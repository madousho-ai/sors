// 贪吃蛇的规则. 纯函数, 局面是普通对象 (可以原样 JSON 化), 每一步返回新局面, 旧的不动.
//
// 局面: { rows, cols, snake: [[行, 列], ...] 头在前, heading, food: [行, 列] | null,
//         status: "playing" | "dead" | "won" | "starved", steps, eaten, sinceFood, last }
// 行 0 在最上面, 列 0 在最左边. last 是上一步的 outcome (新局面为 null).
//
// 规则:
//   - 四个方向都能给. 与当前朝向相反的那个 (掉头咬到脖子) 不生效, 蛇照原方向走
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
// 一道 choice, 选项永远是四个方向, 顺序固定. state 两种写法共用 (boardState); 两种写法只差选项的说明:
// state 用词写棋盘: grid 是整张棋盘, 每行一个字段, 每格一个词 (empty / head / body / food), 不用符号;
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
export const GOAL = "Eat the food: move the snake's head onto the food's cell. Each food eaten makes the snake one cell "
  + "longer, and new food appears somewhere else. The game ends if the head hits a wall or the snake's own body. "
  + "Make the head move toward the food.";

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
      + `The grid lists every row, each cell as one word (empty, head, body or food) from column 0 to column ${g.cols - 1}.`,
    grid: Object.fromEntries(cells.map((row, r) => [`row ${r}`, row.join(", ")])),
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

export function request(g, style, model) {
  if (!STYLES.includes(style)) throw new Error(`unknown style ${style}; expected one of ${STYLES.join(", ")}`);
  const criteria = Object.fromEntries(DIRECTIONS.map((d) => [d, style === "board" ? WAY[d] : describe(g, d)]));
  const state = `${LABEL}: ${JSON.stringify(boardState(g), null, 2)}`;
  return { state, model, questions: { move: { type: "choice", instructions: QUESTION, criteria } } };
}
