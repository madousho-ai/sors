"""sors.serve.menus 的测试: 一道 API 问题 -> 模型看到的菜单样本. 不碰模型.

跑:  PYTHONPATH=src .venv/bin/python tests/test_serve_menus.py
"""

from _runner import run
from sors.core.prompt import render_menu
from sors.serve.api import SystemOneRequest
from sors.serve.menus import to_example

STATE = "Help! My payouts have been failing for 3 days."


def _q(body):
    return SystemOneRequest.model_validate({"state": STATE, "model": "m", "questions": {"q": body}}).questions["q"]


def test_a_choice_becomes_a_menu_of_name_and_description_lines_in_criteria_order():
    """菜单第 i 行是 criteria 第 i 项, 写成「名字: 说明」, 说明是 null 就只写名字; 连续编号 D0, D1, ...
    整条提示与训练时 context-first 的模板逐字相同, 上下文标签由服务指定."""
    q = _q({"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"technical": "Bugs, outages, integrations", "billing": "Payments, invoicing, refunds",
                         "sales": None}})
    ex = to_example(q, STATE, "State")
    assert render_menu(ex) == (
        "State: Help! My payouts have been failing for 3 days.\n\n"
        "Question: Which team should handle this?\n"
        "Options:\n"
        "<|D0|>. technical: Bugs, outages, integrations\n"
        "<|D1|>. billing: Payments, invoicing, refunds\n"
        "<|D2|>. sales\n\n"
        "Answer:")
    assert ex.qtype == "choice" and ex.slot_codes == [0, 1, 2]


def test_a_noul_is_a_no_yes_menu_with_its_criteria_written_after_each_side():
    """与训练里的二元题同形: 两行 no / yes, 类型 bool. criteria 的 false 跟在 no 后面, true 跟在 yes 后面."""
    bare = to_example(_q({"type": "noul", "instructions": "Does this convey urgency?"}), STATE, "State")
    assert bare.option_names == ["no", "yes"] and bare.qtype == "bool"
    ex = to_example(_q({"type": "noul", "instructions": "Does this convey urgency?",
                        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"}}), STATE, "State")
    assert ex.option_names == ["no: No urgency expressed", "yes: Explicitly time-sensitive"]
    half = to_example(_q({"type": "noul", "instructions": "Urgent?", "criteria": {"true": "Time-sensitive"}}), STATE, "S")
    assert half.option_names == ["no", "yes: Time-sensitive"]


def test_a_score_is_a_menu_of_its_levels_from_low_to_high_asked_as_a_choice():
    """分级按顺序排成菜单, 读出来的分布再在 serve.api 里算期望. 训练里没有带 <|score|> 的题,
    所以类型写 choice —— 只有训练时开了 --type-marker 的存档才会把类型写进提示."""
    ex = to_example(_q({"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]}), STATE, "State")
    assert ex.option_names == ["Calm", "Frustrated", "Very angry"] and ex.qtype == "choice"


def test_object_state_and_instructions_are_written_as_indented_json_and_object_descriptions_on_one_line():
    """state 与 instructions 是对象时展开成缩进 2 格的 JSON (core.prompt.state_text, 与 synth-v3 训练时相同);
    菜单一行一项, 选项说明是对象或数组时写成单行 JSON."""
    state = {"ticket": {"subject": "Duplicate charge"}}
    q = _q({"type": "choice",
            "instructions": {"potential_duplicate": {"name": "John Smith"}, "question": "Same person as `potential_duplicate`?"},
            "criteria": {"same": {"what": "Same person", "examples": ["Jon Smith"]}, "different": ["Another person"]}})
    ex = to_example(q, state, "State")
    assert ex.query == '{\n  "ticket": {\n    "subject": "Duplicate charge"\n  }\n}'
    assert ex.question.startswith('{\n  "potential_duplicate": {\n    "name": "John Smith"\n  },')
    assert ex.option_names == ['same: {"what": "Same person", "examples": ["Jon Smith"]}', 'different: ["Another person"]']


def test_line_breaks_inside_an_option_become_spaces_so_each_option_stays_on_its_own_menu_line():
    q = _q({"type": "choice", "instructions": "Which?", "criteria": {"a": "first line\nsecond line", "b\r\nc": None}})
    assert to_example(q, STATE, "State").option_names == ["a: first line second line", "b c"]


if __name__ == "__main__":
    run(globals())
