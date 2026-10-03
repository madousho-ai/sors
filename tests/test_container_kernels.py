"""Container bundles select pinned artifacts, materialize them, and work offline."""

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from _runner import run

ROOT = Path(__file__).resolve().parents[1]


def bundle_module():
    path = ROOT / "docker" / "kernel_bundle.py"
    assert path.is_file(), "the offline kernel bundle builder is missing"
    spec = importlib.util.spec_from_file_location("container_kernel_bundle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot_fixture(root, repo_id, variant):
    snapshot = root / repo_id.rsplit("/", 1)[1]
    artifact = snapshot / "build" / variant
    artifact.mkdir(parents=True)
    content = (repo_id + " pinned kernel").encode()
    blob = root / (repo_id.rsplit("/", 1)[1] + ".blob")
    blob.write_bytes(content)
    (artifact / "__init__.py").symlink_to(blob)
    digest = base64.b64encode(hashlib.sha256(content).digest()).decode()
    (artifact / "metadata.json").write_text(json.dumps({
        "name": repo_id.rsplit("/", 1)[1], "backend": {"type": "cuda"},
        "digest": {"algorithm": "sha256", "files": {"__init__.py": digest}},
    }))
    return snapshot


def test_sm90_bundle_contains_only_its_pinned_attention_and_convolution():
    module = bundle_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        snapshots = {
            "kernels-community/vllm-flash-attn3": snapshot_fixture(
                root, "kernels-community/vllm-flash-attn3", "torch-stable-abi29-cu130-x86_64-linux"),
            "kernels-community/causal-conv1d": snapshot_fixture(
                root, "kernels-community/causal-conv1d", "torch-stable-abi210-cu130-x86_64-linux"),
        }
        revisions = {
            "kernels-community/vllm-flash-attn3": "867af20731c4690484bf76c4e04c09bcd818e115",
            "kernels-community/causal-conv1d": "5b5f06d4ed57f9410a1837bf0393958e2da15297",
        }

        def download(repo_id, *, revision, repo_type, allow_patterns, **kwargs):
            assert revision == revisions[repo_id] and repo_type == "kernel"
            assert len(allow_patterns) == 1 and allow_patterns[0].startswith("build/torch")
            assert allow_patterns != ["build/*"], "downloaded unrelated platforms"
            return str(snapshots[repo_id])

        dest = root / "bundle"
        with patch("huggingface_hub.snapshot_download", side_effect=download):
            module.prepare("sm90", dest)
        # The final bundle must survive removal of the original Hub blobs/cache.
        for blob in root.glob("*.blob"):
            blob.unlink()
        manifest, env = module.runtime_environment(dest, (9, 0))
        assert manifest["attention"] == "flash_attention_3"
        assert set(manifest["kernels"]) == set(snapshots)
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["HF_DATASETS_OFFLINE"] == "1"
        mappings = dict(item.split("=", 1) for item in env["LOCAL_KERNELS"].split(":"))
        assert set(mappings) == set(snapshots)
        for directory in mappings.values():
            assert Path(directory, "__init__.py").is_file()
            assert not Path(directory, "__init__.py").is_symlink()
        try:
            module.runtime_environment(dest, (8, 6))
        except ValueError as exc:
            assert "sm90" in str(exc)
        else:
            raise AssertionError("Hopper image accepted an Ampere GPU")
        first = Path(next(iter(mappings.values())))
        (first / "__init__.py").write_text("corrupt")
        try:
            module.runtime_environment(dest, (9, 0))
        except ValueError as exc:
            assert "checksum" in str(exc).lower()
        else:
            raise AssertionError("tampered kernel was accepted")


def test_unknown_profile_is_rejected_before_download_or_output_creation():
    module = bundle_module()
    with tempfile.TemporaryDirectory() as tmp, \
         patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unexpected network")):
        dest = Path(tmp) / "bundle"
        try:
            module.prepare("sm999", dest)
        except ValueError as exc:
            assert "sm999" in str(exc)
        else:
            raise AssertionError("unknown SM profile accepted")
        assert not dest.exists()


def test_failed_download_never_publishes_a_partial_bundle():
    module = bundle_module()
    with tempfile.TemporaryDirectory() as tmp, \
         patch("huggingface_hub.snapshot_download", side_effect=OSError("download interrupted")):
        dest = Path(tmp) / "bundle"
        try:
            module.prepare("sm8x", dest)
        except OSError:
            pass
        else:
            raise AssertionError("failed download reported success")
        assert not dest.exists()


def test_all_profiles_select_an_attention_kernel_and_the_shared_convolution():
    module = bundle_module()
    for profile, expected in (("sm8x", "flash_attention_2"), ("sm90", "flash_attention_3"),
                              ("sm100", "flash_attention_4"), ("sm120", "flash_attention_2")):
        config, kernels = module.profile_config(profile)
        assert config["attention"] == expected
        assert len(kernels) == 2
        assert any(k["repo"] == "kernels-community/causal-conv1d" for k in kernels)
        assert all(len(k["revision"]) == 40 for k in kernels)


if __name__ == "__main__":
    run(globals())
