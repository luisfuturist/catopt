import { PROGRAMS, applyRule, clone, hasMatch, total } from "engine";
import type { RuleId, Term } from "engine";

export interface RaceResult {
  k: string;
  cost: number;
  path: RuleId[];
}

const ALL: RuleId[] = ["assoc", "comm", "cse"];

/** Same board, same rules — only the player differs. */
export class Players {
  results: RaceResult[] = [];
  best = Number.POSITIVE_INFINITY;
  start = 0;
  started = false;

  private play(kind: string): RaceResult {
    let t: Term = clone(PROGRAMS.chain);
    const path: RuleId[] = [];
    for (let i = 0; i < 6; i++) {
      const open = ALL.filter((r) => hasMatch(t, r));
      if (!open.length) break;
      let pick: RuleId;
      if (kind === "random") {
        pick = open[Math.floor(Math.random() * open.length)];
      } else {
        const improving = open
          .map((r) => ({ r, c: total(applyRule(t, r)) }))
          .filter((x) => x.c < total(t));
        if (!improving.length) break;
        if (kind === "learned" && improving.some((x) => x.r === "assoc")) {
          pick = "assoc";
        } else {
          pick = [...improving].sort((a, b) => a.c - b.c)[0].r;
        }
      }
      t = applyRule(t, pick);
      path.push(pick);
    }
    return { k: kind, cost: total(t), path };
  }

  race(): void {
    this.start = total(PROGRAMS.chain);
    this.started = true;
    const rows = ["random", "greedy", "learned"].map((k) => this.play(k));
    this.best = Math.min(...rows.map((r) => r.cost));
    this.results = rows;
  }
}
