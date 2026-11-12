"""sors.serve.app 的测试: 路由、鉴权、错误码与响应格式. 引擎换成假的, 不碰模型.

跑:  .venv/bin/pytest tests/test_serve_app.py
"""

import pathlib
import tempfile

from fastapi.testclient import TestClient

from sors.serve.app import create_app
from sors.serve.engine import Evaluation, RequestTooLong

NAME = "sors-test"
BODY = {
    "state": "Help! My payouts have been failing for 3 days.",
    "model": NAME,
    "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages",
                                    "sales": "Pricing, upgrades"}},
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    },
}


class FakeEngine:
    """每道题回一个写死的分布, 记下收到的请求."""

    def __init__(self, probs=None, error=None):
        self.probs = probs or {"department": [0.88, 0.12, 0.0], "is_urgent": [0.05, 0.95], "frustration": [0.0, 0.95, 0.05]}
        self.error = error
        self.calls = []

    def evaluate(self, state, questions, images=()):
        self.calls.append((state, questions))
        if self.error:
            raise self.error
        return Evaluation({qid: self.probs[qid] for qid in questions}, input_tokens=123)

    def prompts(self, state, questions, images=()):
        return {qid: (f"State: {state}\n\n", f"Question: {q.instructions}\nAnswer:") for qid, q in questions.items()}


def _client(engine=None, api_key=None):
    return TestClient(create_app(engine or FakeEngine(), NAME, api_key=api_key, description="a test checkpoint",
                                 release_date="2026-09-26"))


def test_a_request_gets_one_answer_per_question_under_its_own_id_with_the_served_model_name():
    r = _client().post("/v1/systemone", json=BODY)
    assert r.status_code == 200, r.text
    got = r.json()
    assert got["model"] == NAME and list(got["answers"]) == ["department", "is_urgent", "frustration"]
    dep = got["answers"]["department"]
    assert dep["type"] == "choice" and dep["choice"] == "billing"
    assert dep["probabilities"] == {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    assert abs(dep["confidence"] - 0.82) < 1e-9
    assert got["answers"]["is_urgent"] == {"type": "noul", "noul": 0.95}
    fr = got["answers"]["frustration"]
    assert abs(fr["score"] - 1.05) < 1e-9 and fr["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert got["usage"] == {"input_tokens": 123, "output_tokens": 3}


def test_the_engine_receives_the_state_and_the_parsed_questions():
    e = FakeEngine()
    body = {**BODY, "state": {"ticket": {"subject": "Payouts failing"}}}
    _client(e).post("/v1/systemone", json=body)
    state, qs = e.calls[0]
    assert state == {"ticket": {"subject": "Payouts failing"}} and qs["frustration"].criteria[0] == "Calm"


def test_a_request_for_another_model_is_refused_with_422_naming_the_served_one():
    r = _client().post("/v1/systemone", json={**BODY, "model": "jev-latest"})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    assert detail[0]["loc"] == ["body", "model"] and NAME in detail[0]["msg"]


def test_a_malformed_request_is_422_and_names_the_offending_field():
    q = {**BODY["questions"]["department"], "criteria": {"only": "one option"}}
    r = _client().post("/v1/systemone", json={**BODY, "questions": {"department": q}})
    assert r.status_code == 422
    assert any(d["loc"][:4] == ["body", "questions", "department", "choice"] for d in r.json()["detail"]), r.text


def test_a_request_too_long_for_the_model_is_422():
    e = FakeEngine(error=RequestTooLong("the state plus the longest question come to 9000 tokens"))
    r = _client(e).post("/v1/systemone", json=BODY)
    assert r.status_code == 422 and "9000 tokens" in r.json()["detail"][0]["msg"], r.text


def test_a_request_too_long_says_context_window_so_benchmark_clients_record_a_capacity_limit():
    # Decision Index 的 http 引擎只在 422 正文含 "context window" 这类短语时记成 unsupported, 否则记 error 并在续跑时重试
    e = FakeEngine(error=RequestTooLong("candidate option branch exceeds max_length"))
    r = _client(e).post("/v1/systemone", json=BODY)
    assert r.status_code == 422 and "context window" in r.json()["detail"][0]["msg"], r.text


def test_with_an_api_key_set_requests_need_the_bearer_token():
    c = _client(api_key="s3cret")
    assert c.post("/v1/systemone", json=BODY).status_code == 401
    assert c.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert c.get("/v1/models").status_code == 401
    ok = c.post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200, ok.text
    assert c.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_without_an_api_key_any_authorization_header_is_accepted():
    """官方 SDK 总会带上自己的 key; 本地服务不设 key 时不管它带的是什么."""
    r = _client().post("/v1/systemone", json=BODY, headers={"Authorization": "Bearer whatever"})
    assert r.status_code == 200, r.text


def test_models_lists_the_served_model():
    r = _client().get("/v1/models")
    assert r.status_code == 200
    assert r.json() == {"models": [{"name": NAME, "description": "a test checkpoint", "release_date": "2026-09-26"}]}


def test_api_documentation_identifies_the_service_as_sors():
    r = _client().get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"] == "SORS"
    assert "State-conditioned Option Ranking System" in r.json()["info"]["summary"]


def _demo_dir() -> pathlib.Path:
    d = pathlib.Path(tempfile.mkdtemp(prefix="demo-")) / "snake"
    d.mkdir()
    (d / "index.html").write_text("<!doctype html><title>snake</title>")
    (d / "snake.mjs").write_text("export const x = 1;")
    return d.parent


def test_the_demo_pages_are_off_unless_a_demo_directory_is_given():
    assert _client().get("/demo/snake/").status_code == 404


def test_with_a_demo_directory_its_pages_are_served_under_demo_without_a_key():
    """演示页是静态文件, 不要 key; 页面里调 API 时才带 key."""
    c = TestClient(create_app(FakeEngine(), NAME, api_key="s3cret", demo_dir=_demo_dir()))
    page = c.get("/demo/snake/")
    assert page.status_code == 200 and "<title>snake</title>" in page.text
    js = c.get("/demo/snake/snake.mjs")
    assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript"), js.headers
    assert c.post("/v1/systemone", json=BODY).status_code == 401


def test_with_the_demo_on_prompts_shows_the_text_the_model_reads_for_each_question():
    """演示页每一步要显示提示原文. System One API 不返回它, 所以演示开着时另有 POST /demo/prompts:
    请求体与 /v1/systemone 相同, 回 {问题 id: {state, question}}, 不跑模型."""
    e = FakeEngine()
    c = TestClient(create_app(e, NAME, demo_dir=_demo_dir()))
    r = c.post("/demo/prompts", json=BODY)
    assert r.status_code == 200, r.text
    got = r.json()["prompts"]
    assert list(got) == ["department", "is_urgent", "frustration"]
    assert got["department"] == {"state": f"State: {BODY['state']}\n\n",
                                 "question": "Question: Which team should handle this?\nAnswer:"}
    assert e.calls == [], "rendering the prompt must not run the model"


def test_prompts_checks_the_model_and_the_key_like_the_api_and_is_off_without_the_demo():
    c = TestClient(create_app(FakeEngine(), NAME, api_key="s3cret", demo_dir=_demo_dir()))
    assert c.post("/demo/prompts", json=BODY).status_code == 401
    ok = {"Authorization": "Bearer s3cret"}
    assert c.post("/demo/prompts", json={**BODY, "model": "jev-latest"}, headers=ok).status_code == 422
    assert c.post("/demo/prompts", json=BODY, headers=ok).status_code == 200
    assert _client().post("/demo/prompts", json=BODY).status_code == 404


def _logged(path):
    import json
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines()]


def test_with_a_log_file_each_answered_request_appends_one_line_with_the_request_and_the_probabilities():
    log = pathlib.Path(tempfile.mkdtemp()) / "requests.jsonl"
    c = TestClient(create_app(FakeEngine(), NAME, log_path=log))
    assert c.post("/v1/systemone", json=BODY).status_code == 200
    assert c.post("/v1/systemone", json={**BODY, "state": "second"}).status_code == 200
    first, second = _logged(log)
    assert first["request"]["state"] == BODY["state"] and second["request"]["state"] == "second"
    assert first["request"]["questions"]["is_urgent"] == {"type": "noul", "instructions": "Does this convey urgency?"}
    assert first["probabilities"] == {"department": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                                      "is_urgent": {"false": 0.05, "true": 0.95},
                                      "frustration": {"0": 0.0, "1": 0.95, "2": 0.05}}
    assert "time" in first


def test_the_log_keeps_a_digest_of_each_image_in_place_of_its_bytes():
    import base64, hashlib
    data = base64.b64encode(b"not really a png").decode()
    from sors.serve import app as app_module
    req = app_module.SystemOneRequest.model_validate(
        {**BODY, "assets": {"img": {"type": "image", "data_base64": data, "format": "png"}}})
    logged = app_module.logged_request(req)
    assert logged["assets"]["img"] == {"type": "image", "format": "png", "bytes": len(b"not really a png"),
                                       "sha256": hashlib.sha256(b"not really a png").hexdigest()}


def test_refused_requests_and_servers_without_a_log_file_write_nothing():
    log = pathlib.Path(tempfile.mkdtemp()) / "requests.jsonl"
    c = TestClient(create_app(FakeEngine(), NAME, log_path=log))
    assert c.post("/v1/systemone", json={**BODY, "model": "other"}).status_code == 422
    assert not log.exists() or log.read_text() == ""
    assert _client().post("/v1/systemone", json=BODY).status_code == 200
