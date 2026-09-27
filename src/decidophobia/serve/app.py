"""FastAPI 应用: TypeSafe System One API 的两个端点.

  POST /v1/systemone   {state, model, questions} -> {model, answers, usage}
  GET  /v1/models      {models: [{name, description, release_date}]}
引擎只要有 evaluate(state, questions) -> serve.engine.Evaluation; 测试里换成假的.
请求的 model 必须是服务的名字 (启动时定), 否则 422, 与请求体校验失败同一种错误格式 (FastAPI 的 detail 列表).
api_key 给了就要求 Authorization: Bearer <key>, 否则 401; 不给就不查 (官方 SDK 总会带自己的 key, 带什么都放行).
端点是普通函数, FastAPI 放进线程池跑; 引擎自己有锁, 请求在 GPU 上排队.
demo_dir 给了才把这个目录当静态文件挂在 /demo/ 下 (如 demos/snake/ -> /demo/snake/), 默认不挂;
演示页本身不要 key, 页面调 API 时照常鉴权.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles

from decidophobia.serve.api import SystemOneRequest, answer
from decidophobia.serve.engine import RequestTooLong


def _unprocessable(field: str, msg: str) -> HTTPException:
    return HTTPException(422, detail=[{"loc": ["body", field], "msg": msg, "type": "value_error"}])


def create_app(engine, model_name: str, api_key: str | None = None, description: str = "",
               release_date: str = "", demo_dir: str | os.PathLike | None = None) -> FastAPI:
    app = FastAPI(title="decidophobia", summary="TypeSafe System One API, served from a decidophobia checkpoint")

    def authorized(authorization: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        if not hmac.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(401, detail="missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})

    @app.post("/v1/systemone", dependencies=[Depends(authorized)])
    def systemone(req: SystemOneRequest) -> dict:
        if req.model != model_name:
            raise _unprocessable("model", f"model {req.model!r} is not served here; this server serves {model_name!r}")
        try:
            ev = engine.evaluate(req.state, req.questions)
        except RequestTooLong as e:
            raise _unprocessable("questions", str(e)) from None
        return {"model": model_name,
                "answers": {qid: answer(q, ev.probs[qid]) for qid, q in req.questions.items()},
                "usage": {"input_tokens": ev.input_tokens, "output_tokens": len(req.questions)}}

    @app.get("/v1/models", dependencies=[Depends(authorized)])
    def models() -> dict:
        return {"models": [{"name": model_name, "description": description, "release_date": release_date}]}

    if demo_dir is not None:
        app.mount("/demo", StaticFiles(directory=demo_dir, html=True), name="demo")
    return app
