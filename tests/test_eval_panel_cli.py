"""scripts/eval-panel.py 的端到端测试: export -> run (对一个本地假服务) -> report.

跑:  .venv/bin/pytest tests/test_eval_panel_cli.py
"""

import importlib.util
import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("eval_panel_cli", ROOT / "scripts" / "eval-panel.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def _tasks(path):
    rows = [{"id": "c-1", "family": "f", "state": "s", "labels": ["a", "b"], "expected": "b",
             "question": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}},
            {"id": "n-1", "family": "f", "state": "s", "labels": ["no", "yes"], "expected": "no",
             "question": {"type": "noul", "instructions": "?"}},
            {"id": "s-1", "family": "f", "state": "s", "labels": ["0", "1", "2"], "expected": 2,
             "question": {"type": "score", "instructions": "?", "criteria": ["x", "y", "z"]}}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class _Fake(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Fake.seen.append(body)
        q = body["questions"]["q"]
        if q["type"] == "choice":
            ans = {"type": "choice", "choice": "b", "probabilities": {"a": .2, "b": .8}}
        elif q["type"] == "noul":
            ans = {"type": "noul", "noul": 0.1}
        else:
            ans = {"type": "score", "score": 2.0, "probabilities": {"0": 0, "1": 0, "2": 1}}
        out = json.dumps({"model": body["model"], "answers": {"q": ans}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_export_run_report_against_a_local_server(tmp_path, capsys):
    tasks = tmp_path / "t.jsonl"
    _tasks(tasks)
    panel = tmp_path / "panel"
    cli.main(["export", "--out", str(panel), "--task", f"mini={tasks}"])
    manifest = json.loads((panel / "manifest.json").read_text())
    assert manifest["mini"]["items"] == 3 and manifest["mini"]["questions"] == 3

    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = tmp_path / "runs" / "m.jsonl"
        cli.main(["run", "--panel", str(panel), "--base-url", f"http://127.0.0.1:{srv.server_port}",
                  "--model", "m", "--out", str(out), "--workers", "2"])
    finally:
        srv.shutdown()
    assert all(b["model"] == "m" and "expected" not in json.dumps(b) for b in _Fake.seen)
    capsys.readouterr()
    cli.main(["report", str(out)])
    table = capsys.readouterr().out
    line = next(x for x in table.splitlines() if x.startswith("mini"))
    assert line.split()[2:] == ["3", "0", "100.0", "100.0", "100.0", "100.0"]
    calib = [x for x in table.splitlines() if x.startswith("mini")][1].split()
    assert calib[-2:] == ["1.000", "1.000"]  # 唯一一道 noul P(yes)=0.1、标准答案 no: 决断且答对
