"""decidophobia.serve.api 的测试: 请求体按 TypeSafe API (POST /v1/systemone) 的规范校验. 不碰模型.

跑:  PYTHONPATH=src .venv/bin/python tests/test_serve_api.py
"""

from pydantic import ValidationError

from _runner import run
from decidophobia.serve.api import Choice, Noul, Score, SystemOneRequest


def _req(questions, state="Help! My payouts have been failing for 3 days.", model="m"):
    return {"state": state, "model": model, "questions": questions}


def _choice(n):
    return {"type": "choice", "instructions": "Which one?", "criteria": {f"o{i}": f"option {i}" for i in range(n)}}


def _score(n):
    return {"type": "score", "instructions": "How bad?", "criteria": [f"level {i}" for i in range(n)]}


def _rejects(body) -> list[tuple]:
    """body 过不了校验; 返回各条错误的位置, 断言里拿来看报的是不是出错的那个字段."""
    try:
        SystemOneRequest.model_validate(body)
    except ValidationError as e:
        return [err["loc"] for err in e.errors()]
    raise AssertionError(f"accepted {body}")


def test_a_request_mixing_the_three_question_types_parses_with_criteria_in_the_order_given():
    r = SystemOneRequest.model_validate(_req({
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?",
                      "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"technical": "Bugs, outages, integrations", "billing": "Payments, invoicing, refunds",
                                    "sales": None}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
    }))
    assert list(r.questions) == ["is_urgent", "department", "frustration"]
    noul, choice, score = r.questions.values()
    assert isinstance(noul, Noul) and noul.criteria.true == "Explicitly time-sensitive"
    assert isinstance(choice, Choice) and list(choice.criteria) == ["technical", "billing", "sales"]
    assert choice.criteria["sales"] is None
    assert isinstance(score, Score) and score.criteria == ["Calm", "Frustrated", "Very angry"]


def test_state_is_text_an_object_or_an_array():
    for state in ("plain", {"ticket": {"subject": "x"}, "n": 3}, ["Hi", "My card was charged twice."]):
        assert SystemOneRequest.model_validate(_req({"q": _choice(2)}, state=state)).state == state
    for state in (5, 1.5, True, None):
        assert any(loc[0] == "state" for loc in _rejects(_req({"q": _choice(2)}, state=state))), state


def test_state_model_and_questions_are_required_and_questions_is_not_empty():
    for field in ("state", "model", "questions"):
        body = _req({"q": _choice(2)})
        del body[field]
        assert (field,) in _rejects(body), field
    assert ("questions",) in _rejects(_req({}))


def test_a_question_of_an_unknown_type_is_rejected():
    assert _rejects(_req({"q": {"type": "rank", "instructions": "Rank them", "criteria": ["a", "b"]}}))


def test_a_choice_has_two_to_255_options():
    SystemOneRequest.model_validate(_req({"q": _choice(2)}))
    SystemOneRequest.model_validate(_req({"q": _choice(255)}))
    for n in (0, 1, 256):
        assert ("questions", "q", "choice", "criteria") in _rejects(_req({"q": _choice(n)})), n


def test_a_score_has_two_to_ten_levels():
    SystemOneRequest.model_validate(_req({"q": _score(2)}))
    SystemOneRequest.model_validate(_req({"q": _score(10)}))
    for n in (0, 1, 11):
        assert ("questions", "q", "score", "criteria") in _rejects(_req({"q": _score(n)})), n


def test_fields_outside_the_spec_are_rejected_instead_of_silently_dropped():
    """criterion 这种拼错的字段若被默默丢掉, 调用方会以为自己的选项说明生效了."""
    q = {**_choice(2), "criterion": {"a": "b"}}
    assert ("questions", "q", "choice", "criterion") in _rejects(_req({"q": q}))
    noul = {"type": "noul", "instructions": "Urgent?", "criteria": {"yes": "Explicitly time-sensitive"}}
    assert ("questions", "q", "noul", "criteria", "yes") in _rejects(_req({"q": noul}))


def test_instructions_are_required_and_not_blank():
    for bad in (None, "", "   ", {}, []):
        q = {**_choice(2), "instructions": bad}
        assert ("questions", "q", "choice", "instructions") in _rejects(_req({"q": q})), bad
    q = _choice(2)
    del q["instructions"]
    assert ("questions", "q", "choice", "instructions") in _rejects(_req({"q": q}))


def test_instructions_and_descriptions_may_be_objects_or_arrays():
    """结构化的 instructions: 问句放一个字段, 它引用的数据放别的字段; 选项与分级的说明同样可以是对象或数组."""
    q = {"type": "noul",
         "instructions": {"potential_duplicate": {"name": "John Smith", "location": "Oakland, California"},
                          "question": "Is the resume for the same person as `potential_duplicate`?"}}
    r = SystemOneRequest.model_validate(_req({"q": q, "c": {**_choice(2), "criteria": {"a": {"what": "x"}, "b": ["y"]}},
                                              "s": {**_score(2), "criteria": [{"level": "low"}, ["high"]]}}))
    assert r.questions["q"].instructions["question"].startswith("Is the resume")


def test_descriptions_are_text_objects_or_arrays_and_only_choice_options_may_leave_them_null():
    assert _rejects(_req({"q": {**_choice(2), "criteria": {"a": 3, "b": "x"}}}))
    assert ("questions", "q", "score", "criteria", 1) in _rejects(_req({"q": {**_score(2), "criteria": ["low", None]}}))
    ok = SystemOneRequest.model_validate(_req({"q": {"type": "noul", "instructions": "Urgent?",
                                                     "criteria": {"true": None, "false": "No urgency"}}}))
    assert ok.questions["q"].criteria.true is None


if __name__ == "__main__":
    run(globals())
