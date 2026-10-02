"""scripts/serve.py 的参数与默认值. 不加载模型、不起服务.

跑:  PYTHONPATH=src .venv/bin/python tests/test_serve_cli.py
"""

import importlib.util
import json
import os
import pathlib
import tempfile
from unittest.mock import patch

from _runner import run

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "serve.py"
_spec = importlib.util.spec_from_file_location("serve_cli", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _run_dir(model="Qwen/Qwen3-1.7B-Base") -> pathlib.Path:
    d = pathlib.Path(tempfile.mkdtemp(prefix="serve-cli-")) / "20260926-153458-synth-attn"
    (d / "checkpoints").mkdir(parents=True)
    (d / "result.json").write_text(json.dumps({"args": {"model": model}}))
    return d


def _args(*argv):
    return _mod.build_parser().parse_args(list(argv))


def test_the_served_name_defaults_to_the_run_directory_and_a_flag_sets_it():
    d = _run_dir()
    assert _mod.served_name(_args("--init", str(d / "trained.safetensors"))) == "20260926-153458-synth-attn"
    assert _mod.served_name(_args("--init", str(d / "checkpoints" / "step-00100.safetensors"))) \
        == "20260926-153458-synth-attn-step-00100"
    assert _mod.served_name(_args("--init", str(d / "trained.safetensors"), "--model-name", "jev-latest")) == "jev-latest"


def test_the_base_model_comes_from_the_flag_or_else_from_the_run_it_was_trained_in():
    d = _run_dir()
    assert _mod.base_model(_args("--init", str(d / "trained.safetensors"))) == "Qwen/Qwen3-1.7B-Base"
    assert _mod.base_model(_args("--init", str(d / "trained.safetensors"), "--base-model", "x/y")) == "x/y"
    lone = pathlib.Path(tempfile.mkdtemp()) / "trained.safetensors"
    try:
        _mod.base_model(_args("--init", str(lone)))
    except SystemExit as e:
        assert "--base-model" in str(e), str(e)
        return
    raise AssertionError("guessed a base model for a checkpoint outside a run")


def test_the_api_key_comes_from_the_flag_or_the_environment():
    with patch.dict(os.environ, {}, clear=True):
        assert _mod.api_key(_args("--init", "x")) is None
        os.environ["SORS_API_KEY"] = "from-env"
        assert _mod.api_key(_args("--init", "x")) == "from-env"
        assert _mod.api_key(_args("--init", "x", "--api-key", "flag")) == "flag"


def test_existing_api_keys_keep_auth_enabled_and_sors_takes_precedence():
    with patch.dict(os.environ, {"DECIDOPHOBIA_API_KEY": "legacy"}, clear=True):
        assert _mod.api_key(_args("--init", "x")) == "legacy"
        os.environ["SORS_API_KEY"] = "current"
        assert _mod.api_key(_args("--init", "x")) == "current"
        assert _mod.api_key(_args("--init", "x", "--api-key", "flag")) == "flag"


def test_it_listens_on_localhost_by_default():
    a = _args("--init", "x")
    assert (a.host, a.port) == ("127.0.0.1", 8000)


def test_the_state_gets_no_label_unless_the_flag_names_one():
    assert _args("--init", "x").context_label == ""
    assert _args("--init", "x", "--context-label", "Customer message").context_label == "Customer message"


def test_the_demo_pages_are_served_only_with_the_demo_flag():
    assert _mod.demo_dir(_args("--init", "x")) is None
    d = _mod.demo_dir(_args("--init", "x", "--demo"))
    assert d == _SCRIPT.parent.parent / "demos", d
    assert (d / "snake" / "index.html").is_file()


if __name__ == "__main__":
    run(globals())
