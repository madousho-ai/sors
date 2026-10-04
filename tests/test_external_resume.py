"""Relocation keeps fingerprints stable; data/code edits still refuse recovery."""
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

from _runner import run
from sors.training import resume

SOURCES = ("scripts/train.py", "src/sors/data/synth_v5.py", "src/sors/core/menu.py",
           "src/sors/core/prompt.py", "src/sors/serve/menus.py",
           "src/sors/data/paths.py", "src/sors/training/resume.py")


def _fixture(base):
    code, assets = base / "code", base / "assets"
    for name in SOURCES:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# source fixture\n")
    data = assets / "synth-intents-v5.2"
    data.mkdir(parents=True)
    (data / "schema.py").write_text("# schema fixture\n")
    (data / "toy.contexts.json").write_text('{"contexts":[]}')
    return code, assets


def _refused(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("modified sampling inputs must refuse recovery")


def test_sampling_fingerprint_survives_relocation_and_detects_data_and_code_changes():
    with tempfile.TemporaryDirectory() as tmp:
        base = pathlib.Path(tmp)
        code, assets = _fixture(base)
        before = resume.sampling_fingerprint(code, assets)
        relocated = base / "moved"
        shutil.copytree(assets, relocated)
        assert resume.sampling_fingerprint(code, relocated) == before
        data = relocated / "synth-intents-v5.2/toy.contexts.json"
        data.write_text('{"contexts":[{"id":"changed"}]}')
        assert resume.sampling_fingerprint(code, relocated) != before
        (code / "src/sors/data/paths.py").write_text("# changed resolver\n")
        assert resume.sampling_fingerprint(code, assets) != before


def test_legacy_fingerprint_is_accepted_only_for_the_exact_audited_migration():
    with tempfile.TemporaryDirectory() as tmp:
        code, assets = _fixture(pathlib.Path(tmp))
        current = resume.sampling_fingerprint(code, assets)
        old = hashlib.sha256(b"old checkpoint fingerprint").hexdigest()
        manifest = code / "src/sors/training/dataset_split_compat.json"
        manifest.write_text(json.dumps({"current_fingerprint": current, "legacy_fingerprints": [old],
                                        "legacy_commits": ["a" * 40]}))
        assert resume.verify_sampling_fingerprint(code, assets, old) == current
        assert resume.verify_sampling_fingerprint(code, assets, current) == current
        _refused(lambda: resume.verify_sampling_fingerprint(code, assets, "unknown"))
        resume.verify_sampling_commits(code, assets, "a" * 7)
        _refused(lambda: resume.verify_sampling_commits(code, assets, "b" * 7))
        (assets / "synth-intents-v5.2/schema.py").write_text("# modified validation\n")
        _refused(lambda: resume.verify_sampling_fingerprint(code, assets, old))
        _refused(lambda: resume.verify_sampling_commits(code, assets, "a" * 7))


def test_legacy_compatibility_refuses_a_changed_sampler():
    with tempfile.TemporaryDirectory() as tmp:
        code, assets = _fixture(pathlib.Path(tmp))
        current = resume.sampling_fingerprint(code, assets)
        manifest = code / "src/sors/training/dataset_split_compat.json"
        manifest.write_text(json.dumps({"current_fingerprint": current, "legacy_fingerprints": ["old"],
                                        "legacy_commits": ["a" * 40]}))
        (code / "src/sors/core/menu.py").write_text("# changed sampler\n")
        _refused(lambda: resume.verify_sampling_fingerprint(code, assets, "old"))


def test_split_commit_verification_checks_both_repositories_and_untracked_data():
    with tempfile.TemporaryDirectory() as tmp:
        code, assets = _fixture(pathlib.Path(tmp))
        revisions = []
        for repo, paths in ((code, ["scripts", "src"]), (assets, ["synth-intents-v5.2"])):
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "--", *paths], check=True)
            subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                            "commit", "-qm", "fixture"], check=True)
            revisions.append(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip())
        resume.verify_sampling_commits(code, assets, revisions[1], code_commit=revisions[0])
        (assets / "synth-intents-v5.2/untracked.contexts.json").write_text("{}")
        _refused(lambda: resume.verify_sampling_commits(code, assets, revisions[1], code_commit=revisions[0]))
        (assets / "synth-intents-v5.2/untracked.contexts.json").unlink()
        (code / "scripts/train.py").write_text("# changed training\n")
        _refused(lambda: resume.verify_sampling_commits(code, assets, revisions[1], code_commit=revisions[0]))


def test_resume_cli_accepts_external_assets_and_separate_revisions():
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/resume.py"
    spec = importlib.util.spec_from_file_location("external_resume_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    try:
        args = cli.build_parser().parse_args(["--run", "runs/example", "--datasets-dir", "../assets",
                                             "--expected-code-commit", "abc1234", "--expected-data-commit", "def5678"])
    except SystemExit:
        raise AssertionError("resume CLI must accept an external root and separate code/data revisions") from None
    assert args.datasets_dir == "../assets"
    assert args.expected_code_commit == "abc1234" and args.expected_data_commit == "def5678"


def test_optional_provenance_works_when_git_is_unavailable():
    with tempfile.TemporaryDirectory() as tmp:
        code, assets = _fixture(pathlib.Path(tmp))
        expected = resume.sampling_fingerprint(code, assets)
        with patch("sors.training.resume.subprocess.run", side_effect=FileNotFoundError("git")):
            assert resume.sampling_provenance(code, assets) == {
                "datasets_dir": str(assets), "code_commit": None, "code_dirty": None,
                "data_commit": None, "data_dirty": None}
            assert resume.verify_sampling_fingerprint(code, assets, expected) == expected


def test_resume_resolves_both_environment_names_before_the_saved_dataset_path():
    script = pathlib.Path(__file__).resolve().parents[1] / "scripts/resume.py"
    spec = importlib.util.spec_from_file_location("resume_environment_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    class SelectionReached(Exception):
        pass

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        run_dir = root / "run"
        (run_dir / "checkpoints").mkdir(parents=True)
        saved, legacy, current, explicit = (root / name for name in ("saved", "legacy", "current", "explicit"))
        resume.save_state({"metadata": {"versions": cli._versions(), "args": {"datasets_dir": str(saved)},
                                         "sampling_fingerprint": "fixture"}, "config": {}},
                          run_dir / "checkpoints/latest.trainstate.safetensors")
        selected = []

        def stop_after_selection(repo, datasets_dir, fingerprint):
            selected.append(pathlib.Path(datasets_dir))
            raise SelectionReached

        cases = [({}, [], saved),
                 ({"DECIDOPHOBIA_DATASETS_DIR": str(legacy)}, [], legacy),
                 ({"DECIDOPHOBIA_DATASETS_DIR": str(legacy), "SORS_DATASETS_DIR": str(current)}, [], current),
                 ({"SORS_DATASETS_DIR": str(current)}, ["--datasets-dir", str(explicit)], explicit)]
        for env, flags, expected in cases:
            with patch.dict(os.environ, env, clear=True), \
                 patch.object(sys, "argv", ["resume.py", "--run", str(run_dir), *flags]), \
                 patch.object(cli, "verify_sampling_fingerprint", side_effect=stop_after_selection):
                try:
                    cli.main()
                except SelectionReached:
                    pass
                else:
                    raise AssertionError("resume skipped sampling verification")
            assert selected[-1] == expected, (selected[-1], expected)


if __name__ == "__main__":
    run(globals())
