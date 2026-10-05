// 贪吃蛇的规则. 纯函数, 局面是普通对象 (可以原样 JSON 化), 每一步返回新局面, 旧的不动.
//
// 局面: { rows, cols, snake: [[行, 列], ...] 头在前, heading, food: [行, 列] | null,
//         status: "playing" | "dead" | "won" | "starved", steps, eaten, sinceFood, last }
// 行 0 在最上面, 列 0 在最左边. last 是上一步的 outcome (新局面为 null).
//
// 规则:
//   - 四个方向都能给. 与当前朝向相反的那个 (掉头咬到脖子) 不生效, 蛇照原方向走;
//     页面可以勾选不把这个方向放进选项 (request 的 dropReverse)
//   - 走出棋盘或撞到自己的身子就死; 尾巴这一步会让开, 头可以跟进尾巴原来的格子
//   - 吃到食物长一节, 新食物落在随机一个空格; 棋盘填满算赢
//   - 连续 行 × 列 × 2 步没吃到东西算饿死, 免得模型原地兜圈子永远不停

export const DIRECTIONS = ["up", "down", "left", "right"];
const DELTA = { up: [-1, 0], down: [1, 0], left: [0, -1], right: [0, 1] };
export const OPPOSITE = { up: "down", down: "up", left: "right", right: "left" };

const same = (a, b) => a[0] === b[0] && a[1] === b[1];
const key = ([r, c]) => `${r},${c}`;

function emptyCells(rows, cols, snake) {
  const taken = new Set(snake.map(key));
  const out = [];
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) if (!taken.has(`${r},${c}`)) out.push([r, c]);
  return out;
}

function dropFood(rows, cols, snake, rng) {
  const free = emptyCells(rows, cols, snake);
  return free.length ? free[Math.floor(rng() * free.length)] : null;
}

export function newGame({ rows = 10, cols = 10, rng = Math.random, snake, heading = "right", food, sinceFood = 0 } = {}) {
  const r = Math.floor(rows / 2), c = Math.floor(cols / 2);
  snake = snake ?? [[r, c], [r, c - 1], [r, c - 2]];
  return {
    rows, cols, snake, heading, food: food ?? dropFood(rows, cols, snake, rng),
    status: "playing", steps: 0, eaten: 0, sinceFood, last: null,
  };
}

// 头从 from 起, 在身子 body 之外的空格里能走到几格 (不算 from 自己)
function reachable(rows, cols, from, body) {
  const blocked = new Set(body.map(key));
  const seen = new Set([key(from)]);
  const todo = [from];
  while (todo.length) {
    const [r, c] = todo.pop();
    for (const [dr, dc] of Object.values(DELTA)) {
      const n = [r + dr, c + dc];
      const k = key(n);
      if (n[0] < 0 || n[0] >= rows || n[1] < 0 || n[1] >= cols || blocked.has(k) || seen.has(k)) continue;
      seen.add(k);
      todo.push(n);
    }
  }
  return seen.size - 1;
}

// 给出方向 dir 的后果, 不改局面.
//   dir / move   给的方向 / 实际走的方向 (掉头时是原朝向); ignored: 掉头被忽略
//   cell         头要去的格子
//   result       "wall" | "self" | "food" | "empty"
//   distance     走过去之后头到食物的曼哈顿距离; 死了或没有食物是 null
//   reachable    走过去之后头还能走到的空格数; 死了是 null
export function outcome(g, dir) {
  if (!DIRECTIONS.includes(dir)) throw new Error(`unknown direction ${dir}`);
  const ignored = g.snake.length > 1 && dir === OPPOSITE[g.heading];
  const move = ignored ? g.heading : dir;
  const [h0, h1] = g.snake[0];
  const cell = [h0 + DELTA[move][0], h1 + DELTA[move][1]];
  const eats = g.food !== null && same(cell, g.food);
  const body = eats ? g.snake : g.snake.slice(0, -1); // 不吃东西的话尾巴这一步让开
  let result;
  if (cell[0] < 0 || cell[0] >= g.rows || cell[1] < 0 || cell[1] >= g.cols) result = "wall";
  else if (body.some((p) => same(p, cell))) result = "self";
  else result = eats ? "food" : "empty";
  const alive = result === "food" || result === "empty";
  return {
    dir, move, ignored, cell, result,
    distance: alive && g.food ? Math.abs(cell[0] - g.food[0]) + Math.abs(cell[1] - g.food[1]) : null,
    reachable: alive ? reachable(g.rows, g.cols, cell, body) : null,
  };
}

export function step(g, dir, rng = Math.random) {
  if (g.status !== "playing") return g;
  const o = outcome(g, dir);
  const steps = g.steps + 1;
  if (o.result === "wall" || o.result === "self") return { ...g, status: "dead", steps, last: o };
  const eats = o.result === "food";
  const snake = [o.cell, ...(eats ? g.snake : g.snake.slice(0, -1))];
  const food = eats ? dropFood(g.rows, g.cols, snake, rng) : g.food;
  const sinceFood = eats ? 0 : g.sinceFood + 1;
  let status = "playing";
  if (snake.length === g.rows * g.cols) status = "won";
  else if (sinceFood >= g.rows * g.cols * 2) status = "starved";
  return { ...g, snake, heading: o.move, food, status, steps, eaten: g.eaten + (eats ? 1 : 0), sinceFood, last: o };
}

// ---- 发给推理服务的请求 ----------------------------------------------------------
// 一道 choice, 选项默认是四个方向 (dropReverse 时去掉掉头的那个), 顺序固定. state 两种写法共用 (boardState); 两种写法只差选项的说明:
// state 用词写棋盘: grid 是整张棋盘, 每行一个字段, 每格一个词 (empty / head / body / food), 不用符号,
// 每行左右两端各加一个 wall;
// 另有几句话说蛇多长、头在哪、身子从脖子到尾巴依次在哪、食物在哪.
// 服务默认不给 state 加标签, 原样放进提示; 训练时游戏局面的提示以「Game state: 」开头, 所以请求里的 state 是
// 一段字符串: 这个标签接上 boardState 的缩进 JSON (与服务展开对象 state 的写法相同).
// state 里不写朝向 (头和脖子的位置就给出了): 写了 "moving": "right", 1.7B 就挑同名的 right,
// 哪怕它的说明写着 game over —— 头贴右墙时四种行序里 right 占三次; 去掉之后同样的两个贴墙局面八次全挑活路.
//   board         说明只讲这个方向往哪边走, 与局面无关. 撞不撞、离食物远近都要模型自己从棋盘上推
//   consequences  说明写这一步的后果: 撞墙 / 撞身子 (game over)、吃到食物、离食物近了还是远了、
//                 走过去之后还剩多少空格能走; 掉头的那个写明会被忽略, 再接上直走的后果

export const QUESTION = "Which direction should the snake move next?";
export const LABEL = "Game state";
export const STYLES = ["consequences", "board"];

const EDGE = { up: "top", down: "bottom", left: "left", right: "right" };
const WAY = {
  up: "Move the head one row up (row number minus 1)",
  down: "Move the head one row down (row number plus 1)",
  left: "Move the head one column left (column number minus 1)",
  right: "Move the head one column right (column number plus 1)",
};

// 游戏目标. System One 的请求没有 system prompt, 放在 state 的第一个字段: state 排在提示最前, 所有题共用.
// 不写方向词 (up / down / left / right), 理由同上面的朝向.
export const GOAL = "Prioritize staying alive. Eat the food: move the snake's head onto the food's cell. "
  + "Each food eaten makes the snake one cell longer, and new food appears somewhere else. "
  + "Eat as much food as possible and grow as long as possible while staying alive.";

const cellName = ([r, c]) => `row ${r}, column ${c}`;

export function boardState(g) {
  const cells = Array.from({ length: g.rows }, () => Array(g.cols).fill("empty"));
  if (g.food) cells[g.food[0]][g.food[1]] = "food";
  g.snake.forEach(([r, c], i) => { cells[r][c] = i === 0 ? "head" : "body"; });
  const [head, ...body] = g.snake;
  const bodyText = body.length
    ? `Its body, from the neck to the tail, is at ${body.map(cellName).join("; ")}.`
    : "It has no body besides the head.";
  return {
    goal: GOAL,
    board: `${g.rows} rows by ${g.cols} columns; row 0 is the top edge, column 0 is the left edge. `
      + `The grid lists every row: a wall, then each cell as one word (empty, head, body or food) `
      + `from column 0 to column ${g.cols - 1}, then a wall.`,
    grid: Object.fromEntries(cells.map((row, r) => [`row ${r}`, ["wall", ...row, "wall"].join(", ")])),
    snake: `The snake is ${g.snake.length} cells long. Its head is at ${cellName(head)}. ${bodyText}`,
    food: g.food ? `The food is at ${cellName(g.food)}.` : "There is no food on the board.",
  };
}

function consequence(g, o) {
  const at = `row ${o.cell[0]}, column ${o.cell[1]}`;
  if (o.result === "wall") return `Head hits the ${EDGE[o.move]} edge of the board: game over`;
  if (o.result === "self") return `Head runs into the snake's own body at ${at}: game over`;
  const room = `${o.reachable} free cells stay reachable`;
  if (o.result === "food") return `Head eats the food at ${at}: the snake grows by one; ${room}`;
  const [h0, h1] = g.snake[0];
  const before = Math.abs(h0 - g.food[0]) + Math.abs(h1 - g.food[1]);
  const way = o.distance < before ? "closer to" : "farther from";
  return `Head moves to ${at}, one step ${way} the food (${o.distance} steps away); ${room}`;
}

function describe(g, dir) {
  const o = outcome(g, dir);
  const what = consequence(g, o);
  return o.ignored ? `Reverses into the snake's neck, so it is ignored and the snake keeps moving ${o.move}. ${what}` : what;
}

// dropReverse: 选项里去掉掉头的那个方向 (反正会被忽略), 只剩三个; 一节长的蛇没有脖子, 四个照给
export function request(g, style, model, { dropReverse = false } = {}) {
  if (!STYLES.includes(style)) throw new Error(`unknown style ${style}; expected one of ${STYLES.join(", ")}`);
  const dirs = DIRECTIONS.filter((d) => !(dropReverse && outcome(g, d).ignored));
  const criteria = Object.fromEntries(dirs.map((d) => [d, style === "board" ? WAY[d] : describe(g, d)]));
  const state = `${LABEL}: ${JSON.stringify(boardState(g), null, 2)}`;
  return { state, model, questions: { move: { type: "choice", instructions: QUESTION, criteria } } };
}

// ---- 拆分模式: 一步拆成小题, 代码把答案组合成方向 -------------------------------------
// 只考虑不掉头的方向 (掉头会被忽略, 问它没有意义). 每个方向问五道, 按模型要自己推多少分三档:
//   查表   outside  noul    给出棋盘行列范围和目标格坐标, 问这格在不在棋盘外
//          body     noul    给出目标格坐标, 问它在不在 snake 字段列出的头和身子里 (尾巴这一步会让开, 也算在内)
//   给坐标 dead     noul    给出目标格坐标, 问头走过去游戏会不会结束
//          closer   noul    给出目标格坐标, 问走过去是不是离食物更近 (没有食物时不问)
//   不给坐标 cell    choice  只说「头的上方那一格」, 问里面是什么: empty / body / food / outside
// 标准答案 splitTruth 按规则算, 页面拿 grade 逐题判对错.
// 组合 (仿 SayCan 把「能不能做」与「有没有用」两个概率相乘): 安全概率按选定的那一档算,
//   lookup  (1 − P(outside)) × (1 − P(body))
//   given   1 − P(dead)
//   infer   P(cell = empty) + P(cell = food)
// 综合分 = 安全 × (1 + P(closer)) / 2. 括号里落在 0.5..1, 安全占主导: 一个安全但更远的方向
// (0.95 × 0.55) 胜过一个多半会死但更近的方向 (0.3 × 0.95).

export const KINDS = ["outside", "body", "dead", "cell", "closer"];
export const SAFETY = ["lookup", "given", "infer"];

const WORD = { up: "one row up", down: "one row down", left: "one column left", right: "one column right" };
const target = (g, d) => [g.snake[0][0] + DELTA[d][0], g.snake[0][1] + DELTA[d][1]];
const dist = ([a, b], [c, d]) => Math.abs(a - c) + Math.abs(b - d);

export function candidates(g) {
  return DIRECTIONS.filter((d) => !(g.snake.length > 1 && d === OPPOSITE[g.heading]));
}

export function splitQuestions(g) {
  const qs = {};
  for (const d of candidates(g)) {
    const at = cellName(target(g, d));
    const move = `If the snake's head moves ${WORD[d]}, to ${at}`;
    qs[`outside_${d}`] = { type: "noul",
      instructions: `The board has rows 0 to ${g.rows - 1} and columns 0 to ${g.cols - 1}. Is ${at} outside the board?` };
    qs[`body_${d}`] = { type: "noul",
      instructions: `Is "${at}" one of the cells listed for the snake's head or body?` };
    qs[`dead_${d}`] = { type: "noul", instructions: `${move}, does the game end?`,
      criteria: { true: "That cell is off the board or taken by the snake's body", false: "That cell is empty or holds the food" } };
    qs[`cell_${d}`] = { type: "choice", instructions: `What is in the cell ${WORD[d]} from the snake's head?`,
      criteria: { empty: "An empty cell", body: "A cell taken by the snake's body", food: "The cell with the food",
        outside: "No cell: that row or column is off the board" } };
    if (g.food) qs[`closer_${d}`] = { type: "noul", instructions: `${move}, is it closer to the food than it is now?` };
  }
  return qs;
}

export function splitRequest(g, model) {
  return { state: `${LABEL}: ${JSON.stringify(boardState(g), null, 2)}`, model, questions: splitQuestions(g) };
}

export function splitTruth(g) {
  const t = {};
  for (const d of candidates(g)) {
    const [r, c] = target(g, d);
    const outside = r < 0 || r >= g.rows || c < 0 || c >= g.cols;
    const body = g.snake.some((p) => same(p, [r, c]));
    const o = outcome(g, d);
    t[`outside_${d}`] = outside;
    t[`body_${d}`] = body;
    t[`dead_${d}`] = o.result === "wall" || o.result === "self";
    t[`cell_${d}`] = outside ? "outside" : body ? "body" : g.food && same(g.food, [r, c]) ? "food" : "empty";
    if (g.food) t[`closer_${d}`] = dist([r, c], g.food) < dist(g.snake[0], g.food);
  }
  return t;
}

// 逐题判对错: noul 看 P(yes) 落在 0.5 哪一边 (正好 0.5 算 no), choice 看模型选的那一项
export function grade(answers, truth) {
  return Object.fromEntries(Object.entries(truth).filter(([q]) => q in answers).map(([q, want]) => {
    const a = answers[q];
    return [q, a.type === "noul" ? (a.noul > 0.5) === want : a.choice === want];
  }));
}

export function combine(g, answers, safety) {
  if (!SAFETY.includes(safety)) throw new Error(`unknown safety ${safety}; expected one of ${SAFETY.join(", ")}`);
  const yes = (q) => answers[q].noul;
  const safe = {}, closer = {}, scores = {};
  for (const d of candidates(g)) {
    if (safety === "lookup") safe[d] = (1 - yes(`outside_${d}`)) * (1 - yes(`body_${d}`));
    else if (safety === "given") safe[d] = 1 - yes(`dead_${d}`);
    else safe[d] = answers[`cell_${d}`].probabilities.empty + answers[`cell_${d}`].probabilities.food;
    closer[d] = answers[`closer_${d}`] ? yes(`closer_${d}`) : 0.5;
    scores[d] = safe[d] * (1 + closer[d]) / 2;
  }
  const choice = Object.keys(scores).reduce((a, b) => (scores[b] > scores[a] ? b : a));
  return { safe, closer, scores, choice };
}
