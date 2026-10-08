import { PROGRAMS, applyRule, clone, hasMatch, total } from "engine";
import type { RuleId, Term } from "engine";

const ALL: RuleId[] = ["assoc", "comm", "cse"];
const MOVES = 3;

/** Beat the model: you play, a policy plays, lowest cost after N moves wins. */
export class Challenge {
  term: Term = clone(PROGRAMS.chain);
  cost = 0;
  moves = 0;
  running = true;

  constructor() {
    this.reset();
  }

  reset(): void {
    this.term = clone(PROGRAMS.chain);
    this.cost = total(this.term);
    this.moves = 0;
    this.running = true;
  }

  get open(): { id: RuleId; cost: number; improves: boolean }[] {
    return ALL.filter((r) => hasMatch(this.term, r)).map((r) => {
      const c = total(applyRule(this.term, r));
      return { id: r, cost: c, improves: c < this.cost };
    });
  }

  play(id: RuleId): void {
    if (!this.running || !hasMatch(this.term, id)) return;
    this.term = applyRule(this.term, id);
    this.cost = total(this.term);
    this.moves += 1;
    if (this.moves >= MOVES) this.running = false;
  }

  /** The policy's run: always the best improving move, same budget. */
  private greedy(steps: number): number {
    let t = clone(PROGRAMS.chain);
    for (let i = 0; i < steps; i++) {
      const opts = ALL.filter((r) => hasMatch(t, r))
        .map((r) => ({ r, c: total(applyRule(t, r)) }))
        .filter((x) => x.c < total(t));
      if (!opts.length) break;
      t = applyRule(t, [...opts].sort((a, b) => a.c - b.c)[0].r);
    }
    return total(t);
  }

  get modelCost(): number {
    return this.greedy(MOVES);
  }

  get verdict(): string {
    if (this.running) return "";
    if (this.cost < this.modelCost) return "you beat the model";
    if (this.cost > this.modelCost) return "the model wins";
    return "a draw — you played it optimally";
  }
}
