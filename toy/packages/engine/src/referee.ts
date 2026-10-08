/** The referee — numeric equivalence on concrete matrices.
 *
 * Stands in for `catopt_core.egraph.verify_certificate`: it evaluates two
 * programs on the same inputs and reports the largest difference.  A move
 * that changes the answer is a lie the referee catches.
 */

import type { Term } from "./term.ts";
import { isLeaf } from "./term.ts";

export type Matrix = number[][];

function mk(rows: number, cols: number, seed: number): Matrix {
  const out: Matrix = [];
  let s = seed;
  for (let i = 0; i < rows; i++) {
    const row: number[] = [];
    for (let j = 0; j < cols; j++) {
      s = (s * 1103515245 + 12345) % 2147483648;
      row.push((s / 2147483648) * 2 - 1);
    }
    out.push(row);
  }
  return out;
}

/** Fixed values for the toy variables — deterministic, so results reproduce. */
export const VALS: Record<string, Matrix> = {
  a: mk(2, 3, 7),
  b: mk(3, 4, 13),
  c: mk(4, 5, 29),
  x: mk(4, 4, 101),
};

const matmul = (x: Matrix, y: Matrix): Matrix =>
  x.map((row) => y[0].map((_, j) => row.reduce((s, v, k) => s + v * y[k][j], 0)));

const ew = (x: Matrix, y: Matrix, f: (p: number, q: number) => number): Matrix =>
  x.map((r, i) => r.map((v, j) => f(v, y[i][j])));

export function evalTerm(t: Term): Matrix {
  if (isLeaf(t)) return VALS[t.leaf];
  const a = t.args.map(evalTerm);
  if (t.op === "matmul") return matmul(a[0], a[1]);
  if (t.op === "add") return ew(a[0], a[1], (p, q) => p + q);
  if (t.op === "mul") return ew(a[0], a[1], (p, q) => p * q);
  if (t.op === "sq") return a[0].map((r) => r.map((v) => v * v));
  if (t.op === "sqq") {
    return ew(
      a[0].map((r) => r.map((v) => v * v)),
      a[1].map((r) => r.map((v) => v * v)),
      (p, q) => p + q,
    );
  }
  return a[0].map((r) => r.map((v) => 2 * v));
}

export function maxDiff(x: Matrix, y: Matrix): number {
  let m = 0;
  x.forEach((r, i) => r.forEach((v, j) => (m = Math.max(m, Math.abs(v - y[i][j])))));
  return m;
}

/** True when the two programs agree on the toy inputs (within fp tolerance). */
export const equivalent = (x: Term, y: Term): boolean => maxDiff(evalTerm(x), evalTerm(y)) < 1e-9;
