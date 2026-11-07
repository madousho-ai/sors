"""FastAPI 应用: TypeSafe System One API 的两个端点.

  POST /v1/systemone   {state, model, questions, assets?} -> {model, answers, usage}
  GET  /v1/models      {models: [{name, description, release_date}]}
引擎只要有 evaluate(state, questions, images) -> serve.engine.Evaluation; 测试里换成假的.
assets 是请求里的图片 (TypeSafe 的 images 规范), 由 serve.assets 换进 state 再交给引擎: 不合规 400, 超限 413,
与官方相同; 引用串写在问题里是 422 on questions (这个服务只在 state 里读图). TypeSafe-Danger-Zone-Images 头不查.
请求体超过 32 MiB 直接 413, 不解析.
请求的 model 必须是服务的名字 (启动时定), 否则 422, 与请求体校验失败同一种错误格式 (FastAPI 的 detail 列表).
api_key 给了就要求 Authorization: Bearer <key>, 否则 401; 不给就不查 (官方 SDK 总会带自己的 key, 带什么都放行).
端点是普通函数, FastAPI 放进线程池跑; 引擎自己有锁, 请求在 GPU 上排队.
demo_dir 给了才把这个目录当静态文件挂在 /demo/ 下 (如 demos/snake/ -> /demo/snake/), 默认不挂;
演示页本身不要 key, 页面调 API 时照常鉴权. 演示开着时另有 POST /demo/prompts: 请求体同 /v1/systemone,
回 {prompts: {问题 id: {state, question}}}, 即模型读到的提示原文 (引擎的 prompts, 不跑模型).
它不在 System One 规范里, 只给演示页显示提示用, 所以跟着 demo_dir 一起开关.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from sors.serve import assets
from sors.serve.api import SystemOneRequest, answer
from sors.serve.assets import ImageRejected, ImagesOutsideState, ImageTooLarge
from sors.serve.engine import RequestTooLong, UnsupportedInput


def _unprocessable(field: str, msg: str) -> HTTPException:
    return HTTPException(422, detail=[{"loc": ["body", field], "msg": msg, "type": "value_error"}])


def _resolved(req: SystemOneRequest) -> tuple:
    """(state, images): 请求的 assets 换进 state (serve.assets). 状态码照官方: 不合规 400, 超限 413."""
    try:
        return assets.resolve(req.state, req.questions, req.assets)
    except ImageTooLarge as e:
        raise HTTPException(413, detail=str(e)) from None
    except ImageRejected as e:
        raise HTTPException(400, detail=str(e)) from None
    except ImagesOutsideState as e:
        raise _unprocessable("questions", str(e)) from None
    except UnsupportedInput as e:
        raise _unprocessable("state", str(e)) from None


class _BodyLimit:
    """请求体超过 serve.assets.MAX_BODY_BYTES 就回 413, 不读完也不交给 FastAPI 解析. 边收边数, 分块传输也管得住."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        limit = assets.MAX_BODY_BYTES
        declared = dict(scope.get("headers") or ()).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > limit:
            return await self._too_large(scope, receive, send, limit)
        chunks, size, more = [], 0, True
        while more:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunks.append(message.get("body", b""))
            size += len(chunks[-1])
            if size > limit:
                return await self._too_large(scope, receive, send, limit)
            more = message.get("more_body", False)
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if sent:
                return await receive()
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)

    @staticmethod
    async def _too_large(scope, receive, send, limit):
        await JSONResponse({"detail": f"the request body is over {limit} bytes"}, status_code=413)(scope, receive, send)


def create_app(engine, model_name: str, api_key: str | None = None, description: str = "",
               release_date: str = "", demo_dir: str | os.PathLike | None = None) -> FastAPI:
    app = FastAPI(title="SORS", summary="State-conditioned Option Ranking System — TypeSafe System One-compatible API")

    def authorized(authorization: str | None = Header(default=None)) -> None:
        if api_key is None:
            return
        if not hmac.compare_digest(authorization or "", f"Bearer {api_key}"):
            raise HTTPException(401, detail="missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})

    def served(req: SystemOneRequest) -> None:
        if req.model != model_name:
            raise _unprocessable("model", f"model {req.model!r} is not served here; this server serves {model_name!r}")

    @app.post("/v1/systemone", dependencies=[Depends(authorized)])
    def systemone(req: SystemOneRequest) -> dict:
        served(req)
        state, images = _resolved(req)
        try:
            ev = engine.evaluate(state, req.questions, images)
        except RequestTooLong as e:
            raise _unprocessable("questions", str(e)) from None
        except UnsupportedInput as e:
            raise _unprocessable("state", str(e)) from None
        return {"model": model_name,
                "answers": {qid: answer(q, ev.probs[qid]) for qid, q in req.questions.items()},
                "usage": {"input_tokens": ev.input_tokens, "output_tokens": len(req.questions)}}

    @app.get("/v1/models", dependencies=[Depends(authorized)])
    def models() -> dict:
        return {"models": [{"name": model_name, "description": description, "release_date": release_date}]}

    if demo_dir is not None:
        @app.post("/demo/prompts", dependencies=[Depends(authorized)])  # 先于下面的挂载注册, 否则被静态目录拦住
        def prompts(req: SystemOneRequest) -> dict:
            served(req)
            state, images = _resolved(req)
            try:
                pieces = engine.prompts(state, req.questions, images)
            except UnsupportedInput as e:
                raise _unprocessable("state", str(e)) from None
            return {"prompts": {qid: {"state": s, "question": question} for qid, (s, question) in pieces.items()}}

        app.mount("/demo", StaticFiles(directory=demo_dir, html=True), name="demo")
    app.add_middleware(_BodyLimit)
    return app
