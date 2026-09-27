// 页面: 把 snake.mjs 的规则接到推理服务和屏幕上.
// 每一步: 局面 -> request() -> POST /v1/systemone -> 模型选的方向 -> step() -> 画出来, 再记一行历史.
// 请求按回合走, 上一个回答回来之前不发下一个.

import { DIRECTIONS, newGame, request, step } from "./snake.mjs";

const $ = (id) => document.getElementById(id);
const ARROW = { up: "↑ 上", down: "↓ 下", left: "← 左", right: "→ 右" };
const RESULT = { empty: "移动", food: "吃到食物", wall: "撞墙", self: "撞到自己" };
const END = {
  dead: (g) => (g.last.result === "wall" ? "撞墙了，游戏结束。" : "撞到自己的身子，游戏结束。"),
  won: () => "填满了整个棋盘，赢了！",
  starved: (g) => `连续 ${g.rows * g.cols * 2} 步没吃到食物，判定饿死。`,
};

let game, model = null, running = false, busy = false;

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
    const p = answer ? answer.probabilities[d] : 0;
    const picked = answer && answer.choice === d ? "picked" : "";
    const name = Object.assign(document.createElement("div"), { className: `name ${picked}`, textContent: ARROW[d] });
    const bar = Object.assign(document.createElement("div"), { className: `bar-wrap ${picked}` });
    bar.innerHTML = `<div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div>`;
    const num = Object.assign(document.createElement("div"), { className: `pct ${picked}`, textContent: answer ? pct(p) : "–" });
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
    return `<span${cls}>${ARROW[d][0]} ${(answer.probabilities[d] * 100).toFixed(0)}</span>`;
  }).join("");
  const res = o.ignored ? `掉头被忽略 · ${RESULT[o.result]}` : RESULT[o.result];
  tr.innerHTML = `<td>${n}</td><td>${ARROW[answer.choice]}</td><td class="probs">${probs}</td>`
    + `<td>${answer.confidence.toFixed(2)}</td><td>${res}</td><td>${ms.toFixed(0)} ms</td>`;
  $("history").prepend(tr);
}

// ---- 一步 ----------------------------------------------------------------------

async function tick() {
  if (busy || game.status !== "playing") return;
  if (!model) await loadModel();
  if (!model) { stop(); return; }
  busy = true;
  try {
    const body = request(game, $("style").value, model);
    const t0 = performance.now();
    const r = await fetch("/v1/systemone", { method: "POST", headers: headers(), body: JSON.stringify(body) });
    const json = await r.json();
    const ms = performance.now() - t0;
    if (!r.ok) throw new Error(`服务返回 ${r.status}：${JSON.stringify(json.detail)}`);
    const answer = json.answers.move;
    game = step(game, answer.choice);
    showDecision(answer, game.last, ms, body.questions.move.criteria);
    addHistory(game.steps, answer, game.last, ms);
    $("raw").textContent = JSON.stringify({ request: body, response: json }, null, 2);
    draw();
    if (game.status !== "playing") {
      stop();
      setStatus(END[game.status](game), "end");
    }
  } catch (e) {
    stop();
    setStatus(e.message, "error");
  } finally {
    busy = false;
  }
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
  $("decision").textContent = "还没有决策。";
  $("why").hidden = true;
  $("latency").textContent = "–";
  showBars(null);
  draw();
  setStatus(model ? "按「开始」让模型来走。" : $("status").textContent);
}

$("run").onclick = () => (running ? stop() : start());
$("step").onclick = () => { stop(); if (game.status !== "playing") reset(); tick(); };
$("reset").onclick = reset;
$("size").onchange = reset;
$("delay").oninput = () => { $("delay-ms").textContent = `${$("delay").value} ms`; };
$("key").onchange = loadModel;

reset();
loadModel();
