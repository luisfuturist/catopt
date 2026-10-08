/** The starting programs the app offers. */

import type { Term } from "./term.ts";

export const PROGRAMS: Record<string, Term> = {
  chain: {
    op: "matmul",
    args: [{ leaf: "a" }, { op: "matmul", args: [{ leaf: "b" }, { leaf: "c" }] }],
  },
  // four matrices, the *worst* bracketing: associativity alone reaches
  // Catalan(3) = 5 forms, from 220 flops down to 152
  chain4: {
    op: "matmul",
    args: [
      {
        op: "matmul",
        args: [{ leaf: "a" }, { op: "matmul", args: [{ leaf: "b" }, { leaf: "c" }] }],
      },
      { leaf: "d" },
    ],
  },
  dup: {
    op: "add",
    args: [
      { op: "mul", args: [{ leaf: "x" }, { leaf: "x" }] },
      { op: "mul", args: [{ leaf: "x" }, { leaf: "x" }] },
    ],
  },
  // the shared redex for the coherence probe: a fusable squared pair
  sqdup: {
    op: "add",
    args: [
      { op: "sq", args: [{ leaf: "x" }] },
      { op: "sq", args: [{ leaf: "x" }] },
    ],
  },
};

/** The greedy one-step oracle: the rule whose result is cheapest. */
export function bestRule(
  term: Term,
  cost: (t: Term) => number,
  apply: (t: Term, r: string) => Term,
  ids: readonly string[],
): string | null {
  let best: string | null = null;
  let bestCost = cost(term);
  for (const id of ids) {
    const c = cost(apply(term, id));
    if (c < bestCost) {
      bestCost = c;
      best = id;
    }
  }
  return best;
}
