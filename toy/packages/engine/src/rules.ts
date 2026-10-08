/** The legal moves — a tiny law library.
 *
 * Each rule preserves the function by construction, which is exactly why
 * the referee only has to check *numbers*, not algebra.
 */

import type { Term } from "./term.ts";
import { isLeaf, isOp, key } from "./term.ts";

export type RuleId = "assoc" | "comm" | "cse" | "square" | "expand" | "fuse";

export interface Rule {
  id: RuleId;
  label: string;
  law: string;
}

export const RULES: readonly Rule[] = [
  { id: "assoc", label: "associativity", law: "(X·Y)·Z ≡ X·(Y·Z)" },
  { id: "comm", label: "commutativity", law: "X + Y ≡ Y + X" },
  { id: "cse", label: "common-subexpression", law: "E + E ≡ 2·E" },
  // The coherence trio — `square` and `expand` are definitional
  // inverses (one equality, two directions, like the real library's
  // silu_fold / silu_expand pair); `fuse` is the redex `expand` kills.
  { id: "square", label: "sq fold", law: "X·X ≡ sq(X)" },
  { id: "expand", label: "sq expand", law: "sq(X) ≡ X·X" },
  { id: "fuse", label: "pair fuse", law: "sq(X) + sq(Y) ≡ sqq(X, Y)" },
];

export function matches(t: Term, rule: RuleId): boolean {
  if (isLeaf(t)) return false;
  const [x, y] = t.args;
  if (rule === "assoc") {
    return t.op === "matmul" && ((isOp(x) && x.op === "matmul") || (isOp(y) && y.op === "matmul"));
  }
  if (rule === "comm") return t.op === "add";
  if (rule === "cse") return t.op === "add" && key(x) === key(y);
  if (rule === "square") return t.op === "mul" && key(x) === key(y);
  if (rule === "expand") return t.op === "sq";
  if (rule === "fuse") {
    return t.op === "add" && isOp(x) && x.op === "sq" && isOp(y) && y.op === "sq";
  }
  return false;
}

export const hasMatch = (t: Term, rule: RuleId): boolean =>
  matches(t, rule) || (isOp(t) && t.args.some((a) => hasMatch(a, rule)));

export function applyRule(t: Term, rule: RuleId): Term {
  if (isLeaf(t)) return t;
  if (matches(t, rule)) {
    const [x, y] = t.args;
    if (rule === "assoc") {
      if (isOp(x) && x.op === "matmul") {
        const [head, mid] = x.args;
        return { op: "matmul", args: [head, { op: "matmul", args: [mid, y] }] };
      }
      if (isOp(y) && y.op === "matmul") {
        const [mid, tail] = y.args;
        return { op: "matmul", args: [{ op: "matmul", args: [x, mid] }, tail] };
      }
    }
    if (rule === "comm") return { op: "add", args: [y, x] };
    if (rule === "cse") return { op: "scale2", args: [x] };
    if (rule === "square") return { op: "sq", args: [x] };
    if (rule === "expand") return { op: "mul", args: [x, x] };
    if (rule === "fuse" && isOp(x) && isOp(y)) {
      // Canonical operand order by key, so comm-then-fuse and
      // fuse-first land on the same term (a normal form).
      const pair = [x.args[0], y.args[0]].sort((p, q) => key(p).localeCompare(key(q)));
      return { op: "sqq", args: pair };
    }
  }
  return { op: t.op, args: t.args.map((a) => applyRule(a, rule)) };
}

/** Every rule that can fire somewhere in `t`. */
export const applicable = (t: Term): RuleId[] =>
  RULES.map((r) => r.id).filter((id) => hasMatch(t, id));

/** Every term reachable by rewriting exactly one matching position.
 *
 * `rebuild` threads the rewritten subtree back into its parents, so the
 * result is a whole program, not a fragment.
 */
export function rewritesAt(t: Term, rule: RuleId): Term[] {
  const out: Term[] = [];
  const visit = (n: Term, rebuild: (x: Term) => Term): void => {
    if (matches(n, rule)) out.push(rebuild(applyRule(n, rule)));
    if (isOp(n)) {
      n.args.forEach((a, i) =>
        visit(a, (x) => rebuild({ op: n.op, args: n.args.map((y, j) => (j === i ? x : y)) })),
      );
    }
  };
  visit(t, (x) => x);
  return out;
}
