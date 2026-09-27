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
