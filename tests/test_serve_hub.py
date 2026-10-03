"""Serving accepts local artifacts and Hub references, including real offline caches."""

import pathlib
import tempfile
from unittest.mock import patch

from _runner import run
from test_serve_cli import _args, _mod

REPO = "example/Sors-demo"
COMMIT = "a" * 40


def resolve(args):
    assert hasattr(_mod, "resolve_init"), "serve CLI lacks HF/local model reference resolution"
    _mod.resolve_init(args)
    return args


def cached_model(root, revision=COMMIT):
    from test_decision import tiny_model
    from sors.core.pretrained import save_pretrained

    repo = root / "models--example--Sors-demo"
    folder = repo / "snapshots" / revision
    model, tok, _, _ = tiny_model("minimal", trainable="full")
    save_pretrained(model, tok, {"layout": "context-first", "type_marker": False,
                               "context_marker": False}, folder)
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs/main").write_text(revision)
    return folder


def test_existing_local_directories_and_checkpoints_never_use_the_hub():
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        checkpoint = root / "trained.safetensors"
        checkpoint.touch()
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("local path reached the Hub")):
            for value in (root, checkpoint):
                args = resolve(_args("--init", str(value)))
                assert pathlib.Path(args.init) == value
            with patch.dict("os.environ", {"HOME": str(root)}):
                args = resolve(_args("--init", "~/trained.safetensors"))
                assert pathlib.Path(args.init) == checkpoint


def test_hub_download_resolves_to_a_loadable_model_and_keeps_the_repo_name():
    assert hasattr(_mod, "resolve_init"), "serve CLI lacks HF/local model reference resolution"
    from sors.serve.engine import load_engine
    from sors.serve.api import Choice

    with tempfile.TemporaryDirectory() as directory:
        folder = cached_model(pathlib.Path(directory))

        def download(repo_id, *, revision, local_files_only):
            assert repo_id == REPO and revision == "v1" and local_files_only is False
            return str(folder)

        with patch("huggingface_hub.snapshot_download", side_effect=download):
            args = resolve(_args("--init", REPO, "--revision", "v1"))
            assert pathlib.Path(args.init) == folder
            assert _mod.served_name(args) == "Sors-demo"
            assert _mod.base_model(args) is None
            engine = load_engine(args.init, device="cpu")
            result = engine.evaluate("red context", {"q": Choice(
                type="choice", instructions="Which color?", criteria={"red": None, "blue": None})})
            assert abs(sum(result.probs["q"]) - 1) < 1e-5
            named = resolve(_args("--init", REPO, "--revision", "v1", "--model-name", "my-model"))
            assert _mod.served_name(named) == "my-model"


def test_real_offline_hf_cache_supports_main_and_pinned_revisions():
    with tempfile.TemporaryDirectory() as directory:
        cache = pathlib.Path(directory)
        first = cached_model(cache)
        second = cached_model(cache, "b" * 40)
        with patch("huggingface_hub.constants.HF_HUB_CACHE", str(cache)), \
             patch("huggingface_hub.HfApi.repo_info", side_effect=AssertionError("offline cache attempted HTTP")):
            args = resolve(_args("--init", REPO, "--local-files-only"))
            assert pathlib.Path(args.init) == second
            pinned = resolve(_args("--init", REPO, "--local-files-only", "--revision", COMMIT))
            assert pathlib.Path(pinned.init) == first
            assert _mod.served_name(pinned) == "Sors-demo"


def test_explicit_missing_paths_fail_locally_and_preserve_the_original_reference():
    with tempfile.TemporaryDirectory() as directory:
        values = (str(pathlib.Path(directory) / "missing"), "./missing-model", "../missing-model",
                  "~/missing-sors-model", "runs/missing.safetensors", "missing.pt", "one/two/three")
        with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("missing local path reached Hub")):
            for value in values:
                args = _args("--init", value)
                try:
                    resolve(args)
                except SystemExit as exc:
                    assert value in str(exc)
                else:
                    raise AssertionError(f"missing local path accepted: {value}")
                assert args.init == value


def test_offline_cache_miss_reports_the_hf_reference_instead_of_a_missing_base():
    with tempfile.TemporaryDirectory() as directory, \
         patch("huggingface_hub.constants.HF_HUB_CACHE", directory), \
         patch("huggingface_hub.HfApi.repo_info", side_effect=AssertionError("offline miss attempted HTTP")):
        try:
            resolve(_args("--init", REPO, "--local-files-only"))
        except SystemExit as exc:
            assert REPO in str(exc) and "--base-model" not in str(exc)
        else:
            raise AssertionError("an uncached HF reference was accepted offline")


def test_hub_model_rejects_a_separate_base_before_downloading():
    with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("unnecessary model download")):
        try:
            resolve(_args("--init", REPO, "--base-model", "other/base"))
        except SystemExit as exc:
            assert "--base-model" in str(exc)
        else:
            raise AssertionError("HF full model accepted a separate base")


def test_main_serves_a_cached_hf_model_through_the_real_http_app():
    from fastapi.testclient import TestClient

    with tempfile.TemporaryDirectory() as directory:
        cached_model(pathlib.Path(directory))
        responses = []

        def run_server(app, *, host, port):
            assert host == "0.0.0.0" and port == 9999
            with TestClient(app) as client:
                models = client.get("/v1/models")
                assert models.status_code == 200
                assert models.json()["models"][0]["name"] == "Sors-demo"
                result = client.post("/v1/systemone", json={
                    "model": "Sors-demo", "state": "red context", "questions": {
                        "q": {"type": "choice", "instructions": "Which color?",
                              "criteria": {"red": None, "blue": None}}}})
                assert result.status_code == 200, result.text
                assert set(result.json()["answers"]["q"]["probabilities"]) == {"red", "blue"}
                assert client.get("/demo/playground/").status_code == 200
                responses.append(result.json())

        argv = ["serve.py", "--init", REPO, "--local-files-only", "--device", "cpu",
                "--demo", "--port", "9999", "--host", "0.0.0.0"]
        with patch("huggingface_hub.constants.HF_HUB_CACHE", directory), \
             patch("huggingface_hub.HfApi.repo_info", side_effect=AssertionError("offline startup attempted HTTP")), \
             patch("sys.argv", argv), patch("uvicorn.run", side_effect=run_server):
            try:
                _mod.main()
            except SystemExit as exc:
                raise AssertionError(f"cached HF CLI startup failed: {exc}") from exc
        assert len(responses) == 1


if __name__ == "__main__":
    run(globals())
