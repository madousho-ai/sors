"""TypeSafe System One API 的线上格式 (https://docs.typesafe.ai/api).

请求 {state, model, questions: {问题 id: Question}}. Question 按 type 分三种, 共有 instructions:
  noul    是非题, criteria 可选, 只认 true / false 两个键, 各是一句说明
  choice  选一项, criteria 是 {选项名: 说明或 null}, 2..255 项, 顺序即调用方写的顺序
  score   有序分级, criteria 是分级说明的数组, 2..10 级, 从低到高
state、instructions 与各条说明都可以是字符串、JSON 对象或数组 (SDK 里叫 JSONContent); 数字、布尔、null 不行.
问题对象上多出规范之外的字段一律拒收 (与 SDK 的 additionalProperties: false 相同), 拼错的字段不会被默默丢掉.
instructions 必填且不能为空: SDK 的类型允许省略, API 参考写的是 required, 这里照 API 参考.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, WithJsonSchema

_CONTENT_SCHEMA = {"anyOf": [{"type": "string"}, {"type": "object"}, {"type": "array"}]}


def _content(v: Any) -> Any:
    if not isinstance(v, (str, dict, list)):
        raise ValueError("must be a string, a JSON object or an array")
    return v


def _optional_content(v: Any) -> Any:
    return None if v is None else _content(v)


def _instructions(v: Any) -> Any:
    _content(v)
    if not (v.strip() if isinstance(v, str) else v):
        raise ValueError("must not be empty")
    return v


# 一个校验函数管到底, 而非 str | dict | list 的联合: 联合类型的错误位置会带上成员名, 报不到字段本身
JSONContent = Annotated[Any, AfterValidator(_content), WithJsonSchema(_CONTENT_SCHEMA)]
OptionalContent = Annotated[Any, AfterValidator(_optional_content),
                            WithJsonSchema({"anyOf": [*_CONTENT_SCHEMA["anyOf"], {"type": "null"}]})]
Instructions = Annotated[Any, AfterValidator(_instructions), WithJsonSchema(_CONTENT_SCHEMA)]

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoulCriteria(_Strict):
    true: OptionalContent = None
    false: OptionalContent = None


class Noul(_Strict):
    type: Literal["noul"]
    instructions: Instructions
    criteria: NoulCriteria | None = None


class Choice(_Strict):
    type: Literal["choice"]
    instructions: Instructions
    criteria: dict[str, OptionalContent] = Field(min_length=2, max_length=MAX_CHOICE_OPTIONS)


class Score(_Strict):
    type: Literal["score"]
    instructions: Instructions
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_SCORE_LEVELS)


Question = Annotated[Noul | Choice | Score, Field(discriminator="type")]


class SystemOneRequest(_Strict):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)
