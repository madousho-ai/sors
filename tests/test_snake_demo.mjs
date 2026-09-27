// demos/snake/snake.mjs 的测试: 贪吃蛇的规则与发给推理服务的请求. 纯函数, 不碰浏览器与服务.
//
// 跑:  node --test tests/test_snake_demo.mjs

import assert from "node:assert/strict";
import { test } from "node:test";

import { QUESTION, newGame, outcome, request, step } from "../demos/snake/snake.mjs";

// 给一个固定的随机数序列, 用完就从头再来. 食物落在空格里的第 floor(r * 空格数) 个
const seq = (...xs) => {
  let i = 0;
  return () => xs[i++ % xs.length];
};

// 蛇头朝右, 头在 (2, 3), 身子往左拖: (2, 2), (2, 1)
const game = (over = {}) =>
  newGame({ rows: 5, cols: 6, snake: [[2, 3], [2, 2], [2, 1]], heading: "right", food: [0, 0], ...over });

test("a new game puts a three-cell snake heading right in the middle and the food on an empty cell", () => {
  const g = newGame({ rows: 10, cols: 10, rng: seq(0) });
  assert.deepEqual(g.snake, [[5, 5], [5, 4], [5, 3]]);
  assert.equal(g.heading, "right");
  assert.equal(g.status, "playing");
  assert.deepEqual(g.food, [0, 0]); // 第 0 个空格是左上角
  const last = newGame({ rows: 10, cols: 10, rng: seq(0.9999) });
  assert.deepEqual(last.food, [9, 9]);
});

test("moving on shifts the head one cell and the tail follows", () => {
  const g = step(game(), "right");
  assert.deepEqual(g.snake, [[2, 4], [2, 3], [2, 2]]);
  assert.equal(g.heading, "right");
  assert.equal(g.steps, 1);
});

test("a turn changes the heading", () => {
  const g = step(game(), "up");
  assert.deepEqual(g.snake[0], [1, 3]);
  assert.equal(g.heading, "up");
});

test("the direction opposite the heading is ignored and the snake keeps going", () => {
  const g = step(game(), "left");
  assert.deepEqual(g.snake, [[2, 4], [2, 3], [2, 2]]);
  assert.equal(g.heading, "right");
  assert.equal(g.last.ignored, true);
  assert.equal(g.last.move, "right");
  assert.equal(step(game(), "right").last.ignored, false);
});

test("leaving the board ends the game against the wall", () => {
  const g = step(game({ snake: [[0, 3], [1, 3], [2, 3]], heading: "up" }), "up");
  assert.equal(g.status, "dead");
  assert.equal(g.last.result, "wall");
  assert.deepEqual(g.snake, [[0, 3], [1, 3], [2, 3]], "a dead snake stays where it was");
});

test("running into its own body ends the game, but the cell the tail is leaving is free", () => {
  // 头 (2,2) 朝上, 身子绕一圈: 右边 (2,3) 是身子中段
  const coil = [[2, 2], [3, 2], [3, 3], [2, 3], [1, 3]];
  const hit = step(game({ snake: coil, heading: "up" }), "right");
  assert.equal(hit.status, "dead");
  assert.equal(hit.last.result, "self");
  // 同样的形状少一节, 右边 (2,3) 就是尾巴, 这一步它会让开
  const chase = step(game({ snake: coil.slice(0, 4), heading: "up" }), "right");
  assert.equal(chase.status, "playing");
  assert.deepEqual(chase.snake[0], [2, 3]);
});

test("eating grows the snake by one and drops new food on an empty cell", () => {
  const g = step(game({ food: [2, 4] }), "right", seq(0));
  assert.deepEqual(g.snake, [[2, 4], [2, 3], [2, 2], [2, 1]]);
  assert.equal(g.eaten, 1);
  assert.equal(g.last.result, "food");
  assert.deepEqual(g.food, [0, 0]);
  const g2 = step(game({ food: [2, 4] }), "right", seq(0.9999));
  assert.deepEqual(g2.food, [4, 5]);
});

test("filling the whole board wins", () => {
  const g = step(newGame({ rows: 1, cols: 3, snake: [[0, 1], [0, 0]], heading: "right", food: [0, 2] }), "right");
  assert.equal(g.status, "won");
  assert.equal(g.food, null);
});

test("going twice the board size in moves without eating starves the snake", () => {
  let g = game({ sinceFood: 5 * 6 * 2 - 1 });
  g = step(g, "up");
  assert.equal(g.status, "starved");
});

test("a finished game does not move", () => {
  const dead = step(game({ snake: [[0, 3], [1, 3], [2, 3]], heading: "up" }), "up");
  assert.equal(step(dead, "left"), dead);
});

test("outcome tells what a move would do without making it", () => {
  const g = game({ food: [2, 5] });
  const o = outcome(g, "right");
  assert.deepEqual(o.cell, [2, 4]);
  assert.equal(o.result, "empty");
  assert.equal(o.distance, 1);
  assert.equal(outcome(g, "up").distance, 3);
  assert.equal(outcome(g, "left").move, "right", "the ignored reverse behaves like going straight");
  assert.equal(outcome(game({ snake: [[0, 3], [1, 3], [2, 3]], heading: "up" }), "up").distance, null);
});

test("outcome counts the free cells the head can still reach after the move", () => {
  // 5 x 6 的棋盘, 蛇占 3 格: 走一步之后仍占 3 格 (尾巴让开), 其余 27 格全连通, 头自己不算
  assert.equal(outcome(game(), "right").reachable, 27);
  // 竖着一堵墙把头往右边关进 1 列: 蛇占第 4 列整列 (5 格), 头在 (0, 5) 朝上走不了, 往下走
  const wall = [[0, 5], [0, 4], [1, 4], [2, 4], [3, 4], [4, 4]];
  const g = newGame({ rows: 5, cols: 6, snake: wall, heading: "right", food: [0, 0] });
  // 往下到 (1, 5): 尾巴 (4, 4) 让开, 第 5 列剩 (2..4, 5) 三格, 经 (4, 4) 通到左边 4 列 20 格
  assert.equal(outcome(g, "down").reachable, 3 + 1 + 20);
  assert.equal(outcome(g, "up").reachable, null);
});

// --------------------------------------------------------------------------
// 发给推理服务的请求: 一道 choice, 选项永远是四个方向
// --------------------------------------------------------------------------

test("a request asks one choice question over the four directions of the model it names", () => {
  const body = request(game(), "board", "decidophobia-0.6b");
  assert.equal(body.model, "decidophobia-0.6b");
  assert.deepEqual(Object.keys(body.questions), ["move"]);
  const q = body.questions.move;
  assert.equal(q.type, "choice");
  assert.equal(q.instructions, QUESTION);
  assert.deepEqual(Object.keys(q.criteria), ["up", "down", "left", "right"]);
  assert.deepEqual(Object.keys(request(game(), "consequences", "m").questions.move.criteria), ["up", "down", "left", "right"]);
});

test("the state draws the board as a grid of rows and names the head, the heading and the food", () => {
  const s = request(game({ food: [0, 5] }), "board", "m").state;
  assert.deepEqual(s.grid, [
    ".....F",
    "......",
    ".ooH..",
    "......",
    "......",
  ]);
  assert.match(s.board, /5 rows by 6 columns/);
  assert.match(s.legend, /H = snake head/);
  assert.deepEqual(s.snake_head, { row: 2, column: 3 });
  assert.deepEqual(s.food, { row: 0, column: 5 });
  assert.equal(s.moving, "right");
  assert.equal(s.snake_length, 3);
});

test("in the board style the options only say which way each direction goes", () => {
  const a = request(game(), "board", "m").questions.move.criteria;
  const b = request(game({ snake: [[0, 3], [1, 3], [2, 3]], heading: "up" }), "board", "m").questions.move.criteria;
  assert.deepEqual(a, b, "board-style options do not depend on the game");
  assert.match(a.up, /row/);
  assert.match(a.left, /column/);
});

test("in the consequences style each option says what the move leads to", () => {
  // 头 (2,3) 朝右, 食物 (2,4) 就在右边; 上面 (1,3) 空着; 左边是掉头
  const c = request(game({ food: [2, 4] }), "consequences", "m").questions.move.criteria;
  assert.match(c.right, /eats the food at row 2, column 4/);
  assert.match(c.right, /grows/);
  assert.match(c.up, /row 1, column 3/);
  assert.match(c.up, /farther from the food/);
  assert.match(c.left, /ignored/);
  assert.match(c.left, /keeps moving right/);
  assert.match(c.left, /eats the food/, "the ignored reverse carries the consequence of going straight");
  const n = outcome(game({ food: [2, 4] }), "up").reachable;
  assert.match(c.up, new RegExp(`${n} free cells`));

  const toward = request(game({ food: [0, 3] }), "consequences", "m").questions.move.criteria;
  assert.match(toward.up, /closer to the food/);

  const edge = request(game({ snake: [[0, 3], [1, 3], [2, 3]], heading: "up" }), "consequences", "m").questions.move.criteria;
  assert.match(edge.up, /top edge/);
  assert.match(edge.up, /game over/);

  const coil = [[2, 2], [3, 2], [3, 3], [2, 3], [1, 3]];
  const bite = request(game({ snake: coil, heading: "up" }), "consequences", "m").questions.move.criteria;
  assert.match(bite.right, /own body/);
  assert.match(bite.right, /game over/);
});

test("an unknown style is refused", () => {
  assert.throws(() => request(game(), "ascii", "m"), /style/);
});
