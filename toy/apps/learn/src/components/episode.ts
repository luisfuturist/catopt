import { PROGRAMS, applyRule, clone, hasMatch, total } from "engine";
import type { RuleId, Term } from "engine";

export interface EpisodeStep {
  rule: RuleId;
  before: number;
  after: number;
  reward: number;
}

const ALL: RuleId[] = ["assoc", "comm", "cse"];
const HORIZON = 6;
const GAMMA = 0.95;

/** One RL episode, stepped by hand: state -> action -> reward. */
export class Episode {
  steps: EpisodeStep[] = [];
  term: Term = clone(PROGRAMS.chain);
  cost = 0;
  t = 0;
  done = false;

  constructor() {
    this.reset();
  }

  reset(): void {
    this.term = clone(PROGRAMS.chain);
    this.cost = total(this.term);
    this.t = 0;
    this.done = false;
    this.steps = [];
  }

  step(): void {
    if (this.done) return;
    const open = ALL.filter((r) => hasMatch(this.term, r));
    const improving = open
      .map((r) => ({ r, c: total(applyRule(this.term, r)) }))
      .filter((x) => x.c < this.cost);
    if (!improving.length) {
      this.done = true;
      return;
    }
    const pick: RuleId = improving.some((x) => x.r === "assoc")
      ? "assoc"
      : [...improving].sort((a, b) => a.c - b.c)[0].r;
    const before = this.cost;
    this.term = applyRule(this.term, pick);
    const after = total(this.term);
    this.cost = after;
    this.t += 1;
    this.steps = [...this.steps, { rule: pick, before, after, reward: (before - after) / before }];
    if (this.t >= HORIZON) this.done = true;
  }

  get returnSoFar(): number {
    let g = 0;
    for (let i = this.steps.length - 1; i >= 0; i--) {
      g = this.steps[i].reward + GAMMA * g;
    }
    return g;
  }
}
