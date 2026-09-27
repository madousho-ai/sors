// demos/playground/request.mjs 的测试: 手填请求的页面背后的纯函数 —— 表单与请求 JSON 互转、
// 答案画成哪几行、提示原文里标哪一行、服务的错误怎么说给人听. 不碰浏览器与服务.
//
// 跑:  node --test tests/test_playground_demo.mjs

import assert from "node:assert/strict";
import { test } from "node:test";

import { answerRows, blankOption, blankQuestion, errorText, markPicked, template, toForm, toRequest }
  from "../demos/playground/request.mjs";

const form = (questions, over = {}) => ({ model: "m", stateKind: "text", state: "s", questions, ...over });
const q = (type, over = {}) => ({ ...blankQuestion(type), id: "q", instructions: "?", ...over });
const sent = (question) => toRequest(form([question])).request.questions[question.id];

// ---- 模板 ----------------------------------------------------------------------

test("the template asks one choice, one noul and one score question of the model it is given", () => {
  const t = template("m");
  assert.equal(t.model, "m");
  assert.deepEqual(Object.values(t.questions).map((x) => x.type), ["choice", "noul", "score"]);
});

test("the template survives a trip through the form unchanged, question and option order included", () => {
  const t = template("m");
  const back = toRequest(toForm(t).form).request;
  assert.deepEqual(back, t);
  assert.deepEqual(Object.keys(back.questions), Object.keys(t.questions));
  const [choice] = Object.keys(t.questions);
  assert.deepEqual(Object.keys(back.questions[choice].criteria), Object.keys(t.questions[choice].criteria));
});

// ---- 表单 -> 请求 --------------------------------------------------------------

test("a choice sends its options in the form's order, an empty description as null", () => {
  const got = sent(q("choice", { options: [{ name: "b", description: "" }, { name: "a", description: "letter a" }] }));
  assert.deepEqual(got, { type: "choice", instructions: "?", criteria: { b: null, a: "letter a" } });
  assert.deepEqual(Object.keys(got.criteria), ["b", "a"]);
});

test("a noul sends criteria only when yes or no has a text, the other side then null", () => {
  assert.deepEqual(sent(q("noul")), { type: "noul", instructions: "?" });
  assert.deepEqual(sent(q("noul", { yes: "says asap" })).criteria, { true: "says asap", false: null });
});

test("a score sends its levels low to high as an array", () => {
  assert.deepEqual(sent(q("score", { levels: ["low", "mid", "high"] })),
    { type: "score", instructions: "?", criteria: ["low", "mid", "high"] });
});

test("a question keeps what was typed for its other types but sends only its own type's fields", () => {
  const got = sent(q("noul", { options: [{ name: "x", description: "" }, { name: "y", description: "" }], levels: ["a", "b"] }));
  assert.deepEqual(got, { type: "noul", instructions: "?" });
});

test("a JSON state is parsed; text that is not a JSON object or array is an error", () => {
  assert.deepEqual(toRequest(form([], { stateKind: "json", state: '{"a": 1}' })).request.state, { a: 1 });
  assert.deepEqual(toRequest(form([], { stateKind: "json", state: "[1, 2]" })).request.state, [1, 2]);
  for (const bad of ["{a: 1", "42", '"text"']) {
    const r = toRequest(form([], { stateKind: "json", state: bad }));
    assert.equal(r.request, undefined, bad);
    assert.match(r.errors.join("\n"), /state/, bad);
  }
});

test("a text state is sent as the string typed", () => {
  assert.equal(toRequest(form([], { state: '{"a": 1}' })).request.state, '{"a": 1}');
});

test("two questions with one id, or two options with one name, are errors since a JSON object keeps only one", () => {
  const dup = toRequest(form([q("noul", { id: "x" }), q("score", { id: "x" })]));
  assert.equal(dup.request, undefined);
  assert.match(dup.errors.join("\n"), /"x"/);
  const opts = toRequest(form([q("choice", { id: "c", options: [{ name: "a", description: "" }, { name: "a", description: "1" }] })]));
  assert.equal(opts.request, undefined);
  assert.match(opts.errors.join("\n"), /"c".*"a"/);
});

test("a blank question takes the first free id q1, q2, ... and starts with two options and three levels", () => {
  const b = blankQuestion("choice", ["q1", "q3"]);
  assert.equal(b.id, "q2");
  assert.equal(b.type, "choice");
  assert.equal(b.options.length, 2);
  assert.equal(b.levels.length, 3);
  assert.equal(blankQuestion("noul").id, "q1");
});

test("a new option takes the first free name option_a ... option_z, then option_27 on, with no description", () => {
  const named = (...names) => names.map((name) => ({ name, description: "" }));
  assert.deepEqual(blankOption(named("option_a", "option_b")), { name: "option_c", description: "" });
  assert.equal(blankOption(named("option_b")).name, "option_a");
  const all = named(..."abcdefghijklmnopqrstuvwxyz".split("").map((c) => `option_${c}`));
  assert.equal(blankOption(all).name, "option_27");
  assert.equal(blankOption([...all, ...named("option_27")]).name, "option_28");
});

// ---- 请求 -> 表单 --------------------------------------------------------------

test("a pasted request fills the form: a string state as text, an object state as indented JSON", () => {
  const { form: f } = toForm({ state: { ticket: "hi" }, model: "jev", questions: {
    u: { type: "noul", instructions: "Urgent?", criteria: { true: "asap", false: null } },
    c: { type: "choice", instructions: "Which?", criteria: { b: null, a: "letter a" } },
    s: { type: "score", instructions: "How bad?", criteria: ["low", "high"] } } });
  assert.equal(f.model, "jev");
  assert.equal(f.stateKind, "json");
  assert.equal(f.state, '{\n  "ticket": "hi"\n}');
  assert.deepEqual(f.questions.map((x) => [x.id, x.type, x.instructions]),
    [["u", "noul", "Urgent?"], ["c", "choice", "Which?"], ["s", "score", "How bad?"]]);
  assert.deepEqual([f.questions[0].yes, f.questions[0].no], ["asap", ""]);
  assert.deepEqual(f.questions[1].options, [{ name: "b", description: "" }, { name: "a", description: "letter a" }]);
  assert.deepEqual(f.questions[2].levels, ["low", "high"]);
  assert.equal(toForm({ state: "plain", model: "m", questions: {} }).form.stateKind, "text");
});

test("a request the form cannot show is reported, naming the question and the field", () => {
  const one = (question) => toForm({ state: "s", model: "m", questions: { q: question } }).error;
  assert.match(one({ type: "noul", instructions: { ask: "?" } }), /"q".*instructions/);
  assert.match(one({ type: "noul", instructions: "?", weight: 2 }), /"q".*weight/);
  assert.match(one({ type: "rank", instructions: "?" }), /"q".*rank/);
  assert.match(one({ type: "choice", instructions: "?", criteria: { a: { what: "x" } } }), /"q".*"a"/);
  assert.match(toForm([1, 2]).error, /对象/);
  assert.match(toForm({ state: 3, model: "m", questions: {} }).error, /state/);
  assert.match(toForm({ state: "s", model: "m", questions: [] }).error, /questions/);
});

// ---- 答案 -> 画出来的行 --------------------------------------------------------

test("a choice answer is one row per option in the question's order, the picked one marked", () => {
  const question = { type: "choice", instructions: "?", criteria: { b: null, a: "letter a", c: { what: "x" } } };
  const answer = { type: "choice", choice: "a", probabilities: { a: 0.5, b: 0.25, c: 0.25 }, confidence: 0.25 };
  assert.deepEqual(answerRows(question, answer), [
    { label: "b", detail: null, p: 0.25, picked: false },
    { label: "a", detail: "letter a", p: 0.5, picked: true },
    { label: "c", detail: '{"what":"x"}', p: 0.25, picked: false }]);
});

test("a noul answer is two rows, no then yes as on the menu, and an even split counts as no", () => {
  const question = { type: "noul", instructions: "?", criteria: { true: "asap", false: null } };
  assert.deepEqual(answerRows(question, { type: "noul", noul: 0.75 }), [
    { label: "no", detail: null, p: 0.25, picked: false },
    { label: "yes", detail: "asap", p: 0.75, picked: true }]);
  assert.deepEqual(answerRows({ type: "noul", instructions: "?" }, { type: "noul", noul: 0.5 }).map((r) => r.picked), [true, false]);
});

test("a score answer is one row per level low to high, the likeliest level marked, the first on a tie", () => {
  const question = { type: "score", instructions: "?", criteria: ["low", "mid", "high"] };
  const answer = { type: "score", score: 1.25, probabilities: { 0: 0.25, 1: 0.25, 2: 0.5 }, confidence: 0.25 };
  assert.deepEqual(answerRows(question, answer), [
    { label: "0", detail: "low", p: 0.25, picked: false },
    { label: "1", detail: "mid", p: 0.25, picked: false },
    { label: "2", detail: "high", p: 0.5, picked: true }]);
  const tie = { ...answer, probabilities: { 0: 0.4, 1: 0.4, 2: 0.2 } };
  assert.deepEqual(answerRows(question, tie).map((r) => r.picked), [true, false, false]);
});

// ---- 提示原文 ------------------------------------------------------------------

test("the prompt marks the menu line of the picked row and no other, D1 apart from D10", () => {
  const menu = Array.from({ length: 11 }, (_, i) => `<|D${i}|>. option ${i}`);
  const prompt = ["Question: Which?", "Options:", ...menu, "", "Answer:"].join("\n");
  const lines = markPicked(prompt, 1);
  assert.equal(lines.length, 15);
  assert.deepEqual(lines.filter((l) => l.picked), [{ text: "<|D1|>. option 1", picked: true }]);
  assert.deepEqual(lines[0], { text: "Question: Which?", picked: false });
});

// ---- 错误 ----------------------------------------------------------------------

test("a 422 lists each complaint after the field it points at", () => {
  const body = JSON.stringify({ detail: [
    { loc: ["body", "questions", "q", "choice", "criteria"], msg: "List should have at least 2 items", type: "too_short" },
    { loc: ["body", "model"], msg: "model 'jev' is not served here", type: "value_error" }] });
  assert.equal(errorText(422, body),
    "请求没通过校验：\nquestions.q.choice.criteria: List should have at least 2 items\nmodel: model 'jev' is not served here");
});

test("a 401 asks for the key, and an error that is not JSON is shown as it came", () => {
  assert.match(errorText(401, '{"detail":"missing or invalid API key"}'), /API key/);
  assert.equal(errorText(500, "Internal Server Error"), "服务返回 500：Internal Server Error");
});
