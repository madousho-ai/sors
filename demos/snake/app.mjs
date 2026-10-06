// 页面: 把 snake.mjs 的规则接到推理服务和屏幕上.
// 每一步: 局面 -> request() -> POST /v1/systemone -> 模型选的方向 -> step() -> 画出来, 再记一行历史.
// 请求按回合走, 上一个回答回来之前不发下一个.
// 同一个请求体并行发给 POST /demo/prompts (服务 --demo 时才有), 拿回模型读到的提示原文显示出来;
// 取不到 (比如老版本的服务) 只是不显示提示, 游戏照走.
// 拆分模式: 一步发一个请求, 里面是每个不掉头方向的五道小题 (splitRequest); 代码把答案组合成方向 (combine),
// 同时按规则算标准答案 (splitTruth) 逐题判对错, 累计成本局各类小题的正确率.

import { DIRECTIONS, KINDS, combine, grade, newGame, request, splitRequest, splitTruth, step } from "./snake.mjs";

const $ = (id) => document.getElementById(id);
const ARROW = { up: "↑ 上", down: "↓ 下", left: "← 左", right: "→ 右" };
const RESULT = { empty: "移动", food: "吃到食物", wall: "撞墙", self: "撞到自己" };
const END = {
  dead: (g) => (g.last.result === "wall" ? "撞墙了，游戏结束。" : "撞到自己的身子，游戏结束。"),
  won: () => "填满了整个棋盘，赢了！",
  starved: (g) => `连续 ${g.rows * g.cols * 2} 步没吃到食物，判定饿死。`,
};

let game, model = null, running = false, busy = false;
let tally = {}; // 小题类型 -> [答对, 总数], 本局累计

const KIND_NAME = { wall: "是墙?（查表）", body: "蛇身里?（查表）", dead: "会死?（给坐标）",
  cell: "那格是（不给坐标）", closer: "更近?（给坐标）" };
const CELL = { empty: "空", body: "身子", food: "食物", wall: "墙" };

const pct = (p) => `${(p * 100).toFixed(1)}%`;
const headers = () => {
  const h = { "Content-Type": "application/json" };
  const key = $("key").value.trim();
  if (key) h.Authorization = `Bearer ${key}`;
  return h;
};

async function loadModel() {
  try {
    const r = await fetch("/v1/models", { headers: headers() });
    if (r.status === 401) throw new Error("服务要 API key，先在下面填上");
    if (!r.ok) throw new Error(`/v1/models 返回 ${r.status}`);
    model = (await r.json()).models[0].name;
    $("model").textContent = model;
    setStatus("按「开始」让模型来走。");
  } catch (e) {
    model = null;
    $("model").textContent = "未连接";
    setStatus(e.message, "error");
  }
}

function setStatus(text, kind = "") {
  $("status").textContent = text;
  $("status").className = `status ${kind}`;
}

// ---- 棋盘 ----------------------------------------------------------------------

function draw() {
  const cv = $("board"), ctx = cv.getContext("2d");
  const cell = cv.width / game.cols;
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = "#1c2129";
  for (let i = 1; i < game.cols; i++) {
    ctx.beginPath(); ctx.moveTo(i * cell, 0); ctx.lineTo(i * cell, cv.height); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i * cell); ctx.lineTo(cv.width, i * cell); ctx.stroke();
  }
  if (game.food) {
    const [r, c] = game.food;
    ctx.fillStyle = "#e5534b";
    ctx.beginPath(); ctx.arc((c + 0.5) * cell, (r + 0.5) * cell, cell * 0.32, 0, Math.PI * 2); ctx.fill();
  }
  game.snake.forEach(([r, c], i) => {
    ctx.fillStyle = i === 0 ? "#9be7b4" : "#3fb56b";
    const pad = i === 0 ? 1 : 3;
    ctx.fillRect(c * cell + pad, r * cell + pad, cell - 2 * pad, cell - 2 * pad);
  });
  if (game.status === "dead") { // 死的那一下画个叉: 撞身子画在撞上的格子, 撞墙那格在棋盘外, 画在头上
    const [r, c] = game.last.result === "wall" ? game.snake[0] : game.last.cell;
    const x = (c + 0.5) * cell, y = (r + 0.5) * cell, d = cell * 0.3;
    ctx.strokeStyle = "#e3a33b"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(x - d, y - d); ctx.lineTo(x + d, y + d); ctx.moveTo(x + d, y - d); ctx.lineTo(x - d, y + d); ctx.stroke();
    ctx.lineWidth = 1;
  }
  $("length").textContent = game.snake.length;
  $("eaten").textContent = game.eaten;
  $("steps").textContent = game.steps;
}

// ---- 决策面板与历史 ------------------------------------------------------------

function showBars(answer) {
  $("bars").replaceChildren(...DIRECTIONS.flatMap((d) => {
    const offered = answer && d in answer.probabilities; // 勾了「不提供掉头的方向」时少一个
    const p = offered ? answer.probabilities[d] : 0;
    const picked = answer && answer.choice === d ? "picked" : "";
    const name = Object.assign(document.createElement("div"), { className: `name ${picked}`, textContent: ARROW[d] });
    const bar = Object.assign(document.createElement("div"), { className: `bar-wrap ${picked}` });
    bar.innerHTML = `<div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div>`;
    const num = Object.assign(document.createElement("div"), { className: `pct ${picked}`, textContent: offered ? pct(p) : "–" });
    return [name, bar, num];
  }));
}

function showDecision(answer, o, ms, criteria) {
  showBars(answer);
  const ignored = o.ignored ? `<span class="ignored">（掉头，被忽略，继续向${ARROW[o.move].slice(2)}）</span>` : "";
  $("decision").innerHTML = `模型选 <b>${ARROW[answer.choice]}</b>，概率 <b>${pct(answer.probabilities[answer.choice])}</b>，`
    + `confidence <b>${answer.confidence.toFixed(2)}</b>${ignored} → ${RESULT[o.result]}`;
  $("why").hidden = false;
  $("why").textContent = `选项说明：${criteria[answer.choice]}`;
  $("latency").textContent = `${ms.toFixed(0)} ms`;
}

function addHistory(n, answer, o, ms) {
  const tr = document.createElement("tr");
  const probs = DIRECTIONS.map((d) => {
    const cls = d === answer.choice ? ' class="picked"' : "";
    const p = d in answer.probabilities ? (answer.probabilities[d] * 100).toFixed(0) : "–";
    return `<span${cls}>${ARROW[d][0]} ${p}</span>`;
  }).join("");
  const res = o.ignored ? `掉头被忽略 · ${RESULT[o.result]}` : RESULT[o.result];
  tr.innerHTML = `<td>${n}</td><td>${ARROW[answer.choice]}</td><td class="probs">${probs}</td>`
    + `<td>${answer.confidence === null ? "–" : answer.confidence.toFixed(2)}</td><td>${res}</td><td>${ms.toFixed(0)} ms</td>`;
  $("history").prepend(tr);
}

// 提示原文: state 段 (棋盘, 所有题共用, 只前向一次) + 问题段 (问句、四个选项、Answer:).
// 选项行形如 '<|D2|>. left: ...', 模型选中的那一行标出来.
async function fetchPrompt(body) {
  try {
    const r = await fetch("/demo/prompts", { method: "POST", headers: headers(), body: JSON.stringify(body) });
    return r.ok ? (await r.json()).prompts : null;
  } catch {
    return null;
  }
}

function showPrompt(p, choice) {
  if (p && !p.state) return showSplitPrompt(p); // 拆分模式: 好几道题共用一段 state
  if (!p) {
    $("prompt-state").textContent = "取不到提示原文 (服务要开 --demo 才有 /demo/prompts)。";
    $("prompt-question").replaceChildren();
    return;
  }
  $("prompt-state").textContent = p.state;
  $("prompt-question").replaceChildren(...p.question.split("\n").map((line) => {
    const picked = new RegExp(`^<\\|D\\d+\\|>\\. ${choice}:`).test(line);
    return Object.assign(document.createElement("span"), { className: picked ? "line picked" : "line", textContent: line || " " });
  }));
  $("prompt-chars").textContent = `${p.state.length + p.question.length} 字符`;
}

function showSplitPrompt(prompts) {
  const all = Object.entries(prompts);
  $("prompt-state").textContent = all[0][1].state;
  $("prompt-question").replaceChildren(...all.flatMap(([q, p]) => [
    Object.assign(document.createElement("span"), { className: "line picked", textContent: `── ${q} ──` }),
    ...p.question.split("\n").map((line) => Object.assign(document.createElement("span"), { className: "line", textContent: line || " " })),
  ]));
  $("prompt-chars").textContent = `${all.length} 道题，state ${all[0][1].state.length} 字符`;
}

// ---- 拆分模式的小题面板 ----------------------------------------------------------

function showSplit(answers, truth, marks, c) {
  const fmt = (q) => {
    if (!(q in answers)) return "<td>–</td>";
    const a = answers[q];
    const said = a.type === "noul" ? a.noul.toFixed(2) : CELL[a.choice];
    const want = a.type === "noul" ? (truth[q] ? "是" : "否") : CELL[truth[q]];
    return `<td class="${marks[q] ? "ok" : "bad"}" title="标准答案：${want}">${marks[q] ? "✓" : "✗"} ${said}</td>`;
  };
  $("split-rows").replaceChildren(...Object.keys(c.scores).map((d) => {
    const tr = document.createElement("tr");
    if (d === c.choice) tr.className = "picked";
    tr.innerHTML = `<td>${ARROW[d]}</td>${KINDS.map((k) => fmt(`${k}_${d}`)).join("")}`
      + `<td>${c.safe[d].toFixed(2)}</td><td>${c.scores[d].toFixed(2)}</td>`;
    return tr;
  }));
}

function showTally() {
  $("tally").replaceChildren(...KINDS.map((k) => {
    const [ok, n] = tally[k] ?? [0, 0];
    const el = document.createElement("div");
    el.innerHTML = `<span>${KIND_NAME[k]}</span> <b>${n ? pct(ok / n) : "–"}</b> <span>${ok}/${n}</span>`;
    return el;
  }));
}

// ---- 一步 ----------------------------------------------------------------------

async function tick() {
  if (busy || game.status !== "playing") return;
  if (!model) await loadModel();
  if (!model) { stop(); return; }
  busy = true;
  try {
    if ($("mode").value === "split") await splitTick();
    else await singleTick();
  } catch (e) {
    stop();
    setStatus(e.message, "error");
  } finally {
    busy = false;
  }
}

async function post(body) {
  const t0 = performance.now();
  const prompt = fetchPrompt(body); // 与决策并行, 只渲染文本, 不占模型
  const r = await fetch("/v1/systemone", { method: "POST", headers: headers(), body: JSON.stringify(body) });
  const json = await r.json();
  const ms = performance.now() - t0;
  if (!r.ok) throw new Error(`服务返回 ${r.status}：${JSON.stringify(json.detail)}`);
  return { json, ms, prompt };
}

function finishStep(body, json) {
  $("raw").textContent = JSON.stringify({ request: body, response: json }, null, 2);
  draw();
  if (game.status !== "playing") {
    stop();
    setStatus(END[game.status](game), "end");
  }
}

async function splitTick() {
  const body = splitRequest(game, model);
  const { json, ms, prompt } = await post(body);
  const answers = json.answers;
  const truth = splitTruth(game); // 走之前的局面
  const marks = grade(answers, truth);
  for (const [q, ok] of Object.entries(marks)) {
    const k = q.split("_")[0];
    const t = (tally[k] ??= [0, 0]);
    t[0] += ok ? 1 : 0; t[1] += 1;
  }
  const c = combine(game, answers, $("safety").value);
  game = step(game, c.choice);
  const o = game.last;
  showBars({ choice: c.choice, probabilities: c.scores });
  const right = Object.values(marks).filter(Boolean).length;
  $("decision").innerHTML = `代码组合选 <b>${ARROW[c.choice]}</b>，综合分 <b>${c.scores[c.choice].toFixed(2)}</b>，`
    + `这一步小题答对 <b>${right}/${Object.keys(marks).length}</b> → ${RESULT[o.result]}`;
  $("why").hidden = false;
  $("why").textContent = `条形是各方向的综合分（安全 × (1 + P(更近)) / 2），三个分数不相加为 100%。`;
  $("latency").textContent = `${ms.toFixed(0)} ms`;
  showSplit(answers, truth, marks, c);
  showTally();
  showPrompt(await prompt);
  addHistory(game.steps, { choice: c.choice, probabilities: c.scores, confidence: null }, o, ms);
  finishStep(body, json);
}

async function singleTick() {
  const body = request(game, $("style").value, model, { dropReverse: $("drop-reverse").checked });
  const { json, ms, prompt } = await post(body);
  const answer = json.answers.move;
  game = step(game, answer.choice);
  showDecision(answer, game.last, ms, body.questions.move.criteria);
  showPrompt((await prompt)?.move, answer.choice);
  addHistory(game.steps, answer, game.last, ms);
  finishStep(body, json);
}

const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));

async function loop() {
  while (running) {
    await tick();
    if (running) await sleep(Number($("delay").value));
  }
}

function start() {
  if (game.status !== "playing") reset();
  running = true;
  $("run").textContent = "暂停";
  setStatus("模型在走…");
  loop();
}

function stop() {
  running = false;
  $("run").textContent = "开始";
  if (game?.status === "playing") setStatus("暂停中。"); // 页面刚打开时 reset 先于第一局
}

function reset() {
  stop();
  const n = Number($("size").value);
  game = newGame({ rows: n, cols: n });
  $("history").replaceChildren();
  tally = {};
  $("split-rows").replaceChildren();
  showTally();
  $("split-panel").hidden = $("mode").value !== "split";
  $("decision").textContent = "还没有决策。";
  $("why").hidden = true;
  $("latency").textContent = "–";
  $("prompt-state").textContent = "还没有提示。";
  $("prompt-question").replaceChildren();
  $("prompt-chars").textContent = "";
  showBars(null);
  draw();
  setStatus(model ? "按「开始」让模型来走。" : $("status").textContent);
}

$("run").onclick = () => (running ? stop() : start());
$("step").onclick = () => { stop(); if (game.status !== "playing") reset(); tick(); };
$("reset").onclick = reset;
$("size").onchange = reset;
$("mode").onchange = reset;
$("delay").oninput = () => { $("delay-ms").textContent = `${$("delay").value} ms`; };
$("key").onchange = loadModel;

reset();
loadModel();
