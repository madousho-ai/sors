// 页面: 左边表单、右边请求 JSON, 发送到同源的推理服务, 把回答和提示原文画出来.
//
// 发出去的永远是右边 JSON 框里的内容. 两边这样跟着变:
//   - 改表单 -> toRequest -> 重写 JSON 框 (表单有 JSON 装不下的毛病时不写, 报在表单下面)
//   - 改 JSON 框 -> toForm -> 重画表单. JSON 有语法错误、或表单画不出这份请求时, 表单整块变灰锁住:
//     这时再改表单会把 JSON 框里手写的东西冲掉. 修好 JSON 或载入模板就解锁
// 发送时同一个请求体并行发给 POST /demo/prompts 拿提示原文; 取不到只是不显示提示.

import { answerRows, blankOption, blankQuestion, errorText, markPicked, template, toForm, toRequest } from "./request.mjs";

const $ = (id) => document.getElementById(id);
const TYPE_NAME = { choice: "choice 选一项", noul: "noul 是非", score: "score 分级" };
const HINT = {
  choice: "左边是选项名，回答里返回的就是它；右边的说明写给模型看，可留空。2 到 255 项，按这里的顺序排成菜单。",
  noul: "两格都可留空，都空就不发 criteria。回答是「是」的概率。",
  score: "从低到高排，2 到 10 级。回答是期望分（0 到最高级的编号）和每一级的概率。",
};

let form, served = null, busy = false;

const pct = (p) => `${(p * 100).toFixed(1)}%`;
const text = (v) => (typeof v === "string" ? v : JSON.stringify(v, null, 2));
const headers = () => {
  const h = { "Content-Type": "application/json" };
  const key = $("key").value.trim();
  if (key) h.Authorization = `Bearer ${key}`;
  return h;
};

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children.filter((c) => c != null));
  return node;
}

// 文字格: 打字只改表单对象和 JSON 框, 不重画表单 (重画会丢掉光标)
const field = (tag, value, set, props = {}) =>
  el(tag, { value, oninput: (e) => { set(e.target.value); syncFromForm(); }, ...props });

// ---- 两边同步 ------------------------------------------------------------------

function syncFromForm() {
  const r = toRequest(form);
  $("form-errors").textContent = r.errors ? `右边的 JSON 没有跟着改：\n${r.errors.join("\n")}` : "";
  if (r.request) $("json").value = JSON.stringify(r.request, null, 2);
}

function lockForm(message) {
  $("form").disabled = Boolean(message);
  $("json-status").textContent = message;
}

function syncFromJson() {
  let parsed;
  try {
    parsed = JSON.parse($("json").value);
  } catch (e) {
    lockForm(`JSON 语法错误：${e.message}\n表单先锁住，免得改表单冲掉这里的内容。`);
    return;
  }
  const r = toForm(parsed);
  if (r.error) {
    lockForm(`表单画不出这份请求：${r.error}。\n发送以这里的 JSON 为准；表单先锁住。`);
    return;
  }
  form = r.form;
  lockForm("");
  $("form-errors").textContent = "";
  renderForm();
}

function loadTemplate() {
  form = toForm(template(served ?? "")).form;
  lockForm("");
  renderForm();
  syncFromForm();
}

// ---- 表单 ----------------------------------------------------------------------

function renderForm() {
  $("model").value = form.model;
  $("state-kind").value = form.stateKind;
  $("state").value = form.state;
  $("state").classList.toggle("mono", form.stateKind === "json");
  $("questions").replaceChildren(...form.questions.map(questionCard));
}

function restructure(change) {
  change();
  renderForm();
  syncFromForm();
}

function questionCard(q, i) {
  const qs = form.questions;
  const move = (d) => restructure(() => { [qs[i], qs[i + d]] = [qs[i + d], qs[i]]; });
  const type = el("select", { onchange: (e) => restructure(() => { q.type = e.target.value; }) },
    ...Object.entries(TYPE_NAME).map(([v, name]) => el("option", { value: v, textContent: name, selected: v === q.type })));
  const head = el("div", { className: "qhead" },
    el("span", { className: "n", textContent: `#${i + 1}` }),
    field("input", q.id, (v) => { q.id = v; }, { className: "qid mono", placeholder: "问题 id，如 is_urgent", title: "问题 id" }),
    type,
    el("button", { className: "icon", textContent: "↑", title: "上移", disabled: i === 0, onclick: () => move(-1) }),
    el("button", { className: "icon", textContent: "↓", title: "下移", disabled: i === qs.length - 1, onclick: () => move(1) }),
    el("button", { className: "icon", textContent: "✕", title: "删掉这道题", onclick: () => restructure(() => qs.splice(i, 1)) }));
  const ask = field("textarea", q.instructions, (v) => { q.instructions = v; },
    { rows: 2, placeholder: "instructions：要问的问题，如「这张工单要不要升级给人工处理？」" });
  return el("div", { className: "qcard" }, head, ask, criteria(q), el("p", { className: "hint", textContent: HINT[q.type] }));
}

function criteria(q) {
  if (q.type === "choice") {
    const rows = q.options.flatMap((o, j) => [
      field("input", o.name, (v) => { o.name = v; }, { className: "mono", placeholder: "选项名" }),
      field("input", o.description, (v) => { o.description = v; }, { placeholder: "说明（可留空）" }),
      el("button", { className: "icon", textContent: "✕", title: "删掉这个选项",
                     onclick: () => restructure(() => q.options.splice(j, 1)) }),
    ]);
    return el("div", {},
      el("div", { className: "opts" }, ...rows),
      el("button", { textContent: "+ 选项", style: "margin-top:6px",
                     onclick: () => restructure(() => q.options.push(blankOption(q.options))) }));
  }
  if (q.type === "noul") {
    return el("div", { className: "noul" },
      el("span", { className: "hint", style: "margin:0", textContent: "yes（true）" }),
      field("input", q.yes, (v) => { q.yes = v; }, { placeholder: "什么情况算「是」（可留空）" }),
      el("span", { className: "hint", style: "margin:0", textContent: "no（false）" }),
      field("input", q.no, (v) => { q.no = v; }, { placeholder: "什么情况算「否」（可留空）" }));
  }
  const rows = q.levels.flatMap((level, j) => [
    el("span", { className: "lv", textContent: `${j}` }),
    field("input", level, (v) => { q.levels[j] = v; }, { placeholder: j === 0 ? "最低一级的说明" : "这一级的说明" }),
    el("button", { className: "icon", textContent: "✕", title: "删掉这一级",
                   onclick: () => restructure(() => q.levels.splice(j, 1)) }),
  ]);
  return el("div", {},
    el("div", { className: "levels" }, ...rows),
    el("button", { textContent: "+ 一级", style: "margin-top:6px", onclick: () => restructure(() => q.levels.push("")) }));
}

// ---- 发送与回答 ----------------------------------------------------------------

function setStatus(message, kind = "") {
  $("status").textContent = message;
  $("status").className = `status ${kind}`;
}

async function loadServed() {
  try {
    const r = await fetch("/v1/models", { headers: headers() });
    if (!r.ok) throw new Error(errorText(r.status, await r.text()));
    served = (await r.json()).models[0].name;
    $("served").textContent = served;
    if (!form.model && !$("form").disabled) {
      form.model = served;
      renderForm();
      syncFromForm();
    }
    if ($("status").classList.contains("error")) setStatus("");
  } catch (e) {
    served = null;
    $("served").textContent = "未连接";
    setStatus(e.message, "error");
  }
}

async function fetchPrompts(body) {
  try {
    const r = await fetch("/demo/prompts", { method: "POST", headers: headers(), body: JSON.stringify(body) });
    return r.ok ? (await r.json()).prompts : null;
  } catch {
    return null;
  }
}

function summary(q, a) {
  if (q.type === "choice") {
    return el("span", { className: "summary" },
      "选 ", el("b", { textContent: a.choice }), ` · 概率 ${pct(a.probabilities[a.choice])} · confidence ${a.confidence.toFixed(2)}`);
  }
  if (q.type === "noul") {
    return el("span", { className: "summary" },
      `P(yes) = ${a.noul.toFixed(3)} → `, el("b", { textContent: a.noul > 0.5 ? "是" : "否" }));
  }
  const top = answerRows(q, a).findIndex((r) => r.picked);
  return el("span", { className: "summary" },
    "期望分 ", el("b", { textContent: a.score.toFixed(2) }),
    `（0 到 ${q.criteria.length - 1}）· 最可能第 ${top} 级 · confidence ${a.confidence.toFixed(2)}`);
}

function resultCard(qid, q, a, prompt) {
  const rows = answerRows(q, a);
  const bars = rows.flatMap((r) => {
    const cls = r.picked ? "picked" : "";
    return [
      el("div", { className: `name ${cls}` }, r.label, r.detail ? el("small", { textContent: r.detail }) : null),
      el("div", { className: cls, innerHTML: `<div class="bar"><i style="width:${(r.p * 100).toFixed(1)}%"></i></div>` }),
      el("div", { className: `pct ${cls}`, textContent: pct(r.p) }),
    ];
  });
  const picked = rows.findIndex((r) => r.picked);
  const promptBlock = prompt
    ? el("details", { open: true }, el("summary", { textContent: `② 问题段（${prompt.question.length} 字符）` }),
        el("pre", { className: "prompt mono" }, ...markPicked(prompt.question, picked).map((l) =>
          el("span", { className: l.picked ? "line picked" : "line", textContent: l.text || " " }))))
    : null;
  return el("div", { className: "result" },
    el("div", { className: "title" }, el("b", { className: "mono", textContent: qid }),
      el("span", { className: "type", textContent: q.type }), summary(q, a)),
    el("div", { className: "ask", textContent: text(q.instructions) }),
    el("div", { className: "bars" }, ...bars),
    promptBlock);
}

function showPromptState(prompts) {
  const first = prompts && Object.values(prompts)[0];
  $("prompt-state").textContent = first ? first.state : "取不到提示原文（服务要开 --demo 才有 /demo/prompts）。";
  $("state-chars").textContent = first ? `（${first.state.length} 字符）` : "";
}

async function send() {
  if (busy) return;
  let body;
  try {
    body = JSON.parse($("json").value);
  } catch (e) {
    setStatus(`JSON 语法错误，没有发送：${e.message}`, "error");
    return;
  }
  busy = true;
  $("send").disabled = true;
  setStatus("等模型回答…");
  try {
    const t0 = performance.now();
    const prompts = fetchPrompts(body);
    const r = await fetch("/v1/systemone", { method: "POST", headers: headers(), body: JSON.stringify(body) });
    const raw = await r.text();
    const ms = performance.now() - t0;
    if (!r.ok) {
      setStatus(errorText(r.status, raw), "error");
      $("raw").textContent = `${JSON.stringify(body, null, 2)}\n\n—— 服务返回 ${r.status} ——\n${raw}`;
      return;
    }
    const json = JSON.parse(raw), p = await prompts;
    $("results").replaceChildren(...Object.entries(body.questions).map(([qid, q]) =>
      resultCard(qid, q, json.answers[qid], p?.[qid])));
    showPromptState(p);
    $("raw").textContent = JSON.stringify({ request: body, response: json }, null, 2);
    setStatus(`${ms.toFixed(0)} ms · ${Object.keys(body.questions).length} 道题 · input_tokens ${json.usage.input_tokens}`);
  } catch (e) {
    setStatus(`请求没发出去：${e.message}`, "error");
  } finally {
    busy = false;
    $("send").disabled = false;
  }
}

// ---- 接线 ----------------------------------------------------------------------

$("model").oninput = (e) => { form.model = e.target.value; syncFromForm(); };
$("state").oninput = (e) => { form.state = e.target.value; syncFromForm(); };
$("state-kind").onchange = (e) => restructure(() => { form.stateKind = e.target.value; });
$("json").oninput = syncFromJson;
$("load-template").onclick = loadTemplate;
$("send").onclick = send;
$("key").onchange = loadServed;
for (const b of document.querySelectorAll("[data-add]")) {
  b.onclick = () => restructure(() => form.questions.push(blankQuestion(b.dataset.add, form.questions.map((q) => q.id))));
}
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});

loadTemplate();
loadServed();
