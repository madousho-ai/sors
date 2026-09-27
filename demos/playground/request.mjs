// 手填请求页面背后的纯函数: 表单与请求 JSON 互转、答案画成哪几行、提示原文里标哪一行、错误怎么说.
// 不碰 DOM 与网络, 页面逻辑在 app.mjs.
//
// 请求照 System One API: { state, model, questions: { 问题 id: { type, instructions, criteria } } }.
// 表单是另一种形状, 按页面上的格子来:
//   { model, stateKind: "text" | "json", state: 文本框里的字,
//     questions: [{ id, type, instructions, options: [{ name, description }], yes, no, levels: [...] }] }
// 每道题三种题型的格子都留着 (options 给 choice, yes / no 给 noul, levels 给 score), 切题型时别的格子里
// 写过的字不丢; 转成请求时只取当前题型那几格.
//
// 表单只能编辑文字. 请求里 instructions 或选项说明是 JSON 对象、多了规范之外的字段, 表单画不出来,
// toForm 报出是哪道题哪一格; 页面那时以 JSON 为准, 照样能发.

const TYPES = ["choice", "noul", "score"];

const isObject = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const isContainer = (v) => v !== null && typeof v === "object";
const shown = (d) => (d == null ? null : typeof d === "string" ? d : JSON.stringify(d));

/** 三种题型各一道的空壳请求, 每一格里写着该填什么. */
export function template(model) {
  return {
    state: "这里写模型要读的内容：一段文字；也可以把 state 格式切到 JSON，写成对象或数组。同一个请求里的所有问题共用这一份 state。",
    model,
    questions: {
      which_one: {
        type: "choice",
        instructions: "这里写一个问题，模型从下面的选项里挑一个。",
        criteria: {
          option_a: "选项说明，写给模型看；左边的选项名是回答里返回的名字。",
          option_b: "至少 2 个、最多 255 个选项，按这里的顺序排成菜单。",
          option_c: null,
        },
      },
      is_it_true: {
        type: "noul",
        instructions: "这里写一个是非问题，回答是「是」的概率。",
        criteria: { true: "（可选）什么情况算「是」。", false: "（可选）什么情况算「否」。" },
      },
      how_much: {
        type: "score",
        instructions: "这里写一个分级问题，回答是期望分和每一级的概率。",
        criteria: ["最低一级：写清这一级是什么样。", "中间一级。", "最高一级。共 2 到 10 级，从低到高排。"],
      },
    },
  };
}

/** 一道新题: id 取 q1, q2, ... 里第一个没被占的, 各题型的格子都是空的. */
export function blankQuestion(type, taken = []) {
  let n = 1;
  while (taken.includes(`q${n}`)) n++;
  return {
    id: `q${n}`, type, instructions: "",
    options: [{ name: "option_a", description: "" }, { name: "option_b", description: "" }],
    yes: "", no: "", levels: ["", "", ""],
  };
}

// ---- 表单 -> 请求 --------------------------------------------------------------

function sendable(x, errors) {
  if (x.type === "choice") {
    const seen = new Set();
    for (const { name } of x.options) {
      if (seen.has(name)) errors.push(`题 "${x.id}" 有两个选项都叫 "${name}"，JSON 里只会留下一个`);
      seen.add(name);
    }
    return { type: "choice", instructions: x.instructions,
             criteria: Object.fromEntries(x.options.map((o) => [o.name, o.description === "" ? null : o.description])) };
  }
  if (x.type === "noul") {
    const out = { type: "noul", instructions: x.instructions };
    if (x.yes !== "" || x.no !== "") out.criteria = { true: x.yes || null, false: x.no || null };
    return out;
  }
  return { type: "score", instructions: x.instructions, criteria: [...x.levels] };
}

/** 表单 -> { request } 或 { errors: [...] }. 只查 JSON 装不下的毛病, 其余 (选项数、空问题) 交给服务校验. */
export function toRequest(f) {
  const errors = [];
  let state = f.state;
  if (f.stateKind === "json") {
    try {
      state = JSON.parse(f.state);
      if (!isContainer(state)) errors.push("state 选了 JSON，内容要是对象 {…} 或数组 […]");
    } catch (e) {
      errors.push(`state 选了 JSON，但它不是合法的 JSON：${e.message}`);
    }
  }
  const ids = new Set(), entries = [];
  for (const x of f.questions) {
    if (ids.has(x.id)) errors.push(`有两道题的 id 都是 "${x.id}"，JSON 里只会留下一道`);
    ids.add(x.id);
    entries.push([x.id, sendable(x, errors)]);
  }
  return errors.length ? { errors } : { request: { state, model: f.model, questions: Object.fromEntries(entries) } };
}

// ---- 请求 -> 表单 --------------------------------------------------------------

function formQuestion(id, x) {
  const at = `题 "${id}"`;
  if (!isObject(x)) return `${at} 要是一个对象`;
  if (!TYPES.includes(x.type)) return `${at} 的 type 是 ${JSON.stringify(x.type)}，只认 choice / noul / score`;
  const extra = Object.keys(x).filter((k) => !["type", "instructions", "criteria"].includes(k));
  if (extra.length) return `${at} 多了表单没有的字段 ${extra.join("、")}`;
  if (typeof x.instructions !== "string") return `${at} 的 instructions 不是文字，表单只能编辑文字`;
  const q = { ...blankQuestion(x.type), id, instructions: x.instructions };
  const c = x.criteria;
  if (x.type === "choice") {
    if (!isObject(c)) return `${at} 的 criteria 要是 {选项名: 说明} 对象`;
    for (const [name, d] of Object.entries(c)) {
      if (d !== null && typeof d !== "string") return `${at} 的选项 "${name}" 的说明不是文字，表单只能编辑文字`;
    }
    q.options = Object.entries(c).map(([name, d]) => ({ name, description: d ?? "" }));
  } else if (x.type === "noul") {
    if (c != null) {
      if (!isObject(c)) return `${at} 的 criteria 要是 {true: 说明, false: 说明} 对象`;
      const bad = Object.keys(c).find((k) => !["true", "false"].includes(k) || (c[k] !== null && typeof c[k] !== "string"));
      if (bad !== undefined) return `${at} 的 criteria.${bad} 表单画不出：只认 true / false 两格文字`;
      [q.yes, q.no] = [c.true ?? "", c.false ?? ""];
    }
  } else {
    if (!Array.isArray(c)) return `${at} 的 criteria 要是分级说明的数组`;
    const bad = c.findIndex((level) => typeof level !== "string");
    if (bad >= 0) return `${at} 的第 ${bad} 级不是文字，表单只能编辑文字`;
    q.levels = [...c];
  }
  return q;
}

/** 请求 JSON -> { form } 或 { error: 哪一格画不出 }. model 缺着就是空字符串. */
export function toForm(r) {
  if (!isObject(r)) return { error: "请求要是一个 JSON 对象 {state, model, questions}" };
  const extra = Object.keys(r).filter((k) => !["state", "model", "questions"].includes(k));
  if (extra.length) return { error: `请求多了表单没有的字段 ${extra.join("、")}` };
  let stateKind, state;
  if (typeof r.state === "string") [stateKind, state] = ["text", r.state];
  else if (isContainer(r.state)) [stateKind, state] = ["json", JSON.stringify(r.state, null, 2)];
  else return { error: "state 要是字符串、JSON 对象或数组" };
  if (r.model !== undefined && typeof r.model !== "string") return { error: "model 要是字符串" };
  if (!isObject(r.questions)) return { error: "questions 要是一个对象 {问题 id: 问题}" };
  const questions = [];
  for (const [id, x] of Object.entries(r.questions)) {
    const q = formQuestion(id, x);
    if (typeof q === "string") return { error: q };
    questions.push(q);
  }
  return { form: { model: r.model ?? "", stateKind, state, questions } };
}

// ---- 答案 -> 画出来的行 --------------------------------------------------------

/**
 * 一道题的回答画成几行 { label, detail, p, picked }, 行序与模型看到的菜单相同 (serve.menus):
 * choice 按 criteria 的顺序, noul 是 no 再 yes, score 从低到高. picked 与服务选答案的规则一致, 并列取靠前的.
 */
export function answerRows(question, answer) {
  if (question.type === "choice") {
    return Object.entries(question.criteria).map(([name, d]) => (
      { label: name, detail: shown(d), p: answer.probabilities[name], picked: name === answer.choice }));
  }
  if (question.type === "noul") {
    const c = question.criteria ?? {}, yes = answer.noul > 0.5;
    return [{ label: "no", detail: shown(c.false), p: 1 - answer.noul, picked: !yes },
            { label: "yes", detail: shown(c.true), p: answer.noul, picked: yes }];
  }
  const ps = question.criteria.map((_, i) => answer.probabilities[String(i)]);
  const top = ps.indexOf(Math.max(...ps));
  return question.criteria.map((level, i) => ({ label: String(i), detail: shown(level), p: ps[i], picked: i === top }));
}

/** 提示的问题段逐行拆开, 菜单第 row 行 (形如 '<|D2|>. 名字: 说明') 标上 picked. */
export function markPicked(prompt, row) {
  const mark = `<|D${row}|>. `;
  return prompt.split("\n").map((text) => ({ text, picked: text.startsWith(mark) }));
}

// ---- 错误 ----------------------------------------------------------------------

/** 服务回的非 2xx 说成人话. text 是响应原文: 校验失败是 FastAPI 的 {detail: [{loc, msg}]}, 爆显存之类是纯文本. */
export function errorText(status, text) {
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = undefined;
  }
  if (status === 401) return "服务要 API key，或者填的 key 不对：在上面填上再发。";
  const detail = body?.detail;
  if (status === 422 && Array.isArray(detail)) {
    const where = (loc = []) => (loc[0] === "body" ? loc.slice(1) : loc).join(".");
    return ["请求没通过校验：", ...detail.map((d) => `${where(d.loc)}: ${d.msg}`)].join("\n");
  }
  return `服务返回 ${status}：${typeof detail === "string" ? detail : text}`;
}
