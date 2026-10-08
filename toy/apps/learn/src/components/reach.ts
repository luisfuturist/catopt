import { PROGRAMS, RULES, clone, key, rewritesAt, total } from "engine";
import type { Term } from "engine";

const ALL = RULES.map((r) => r.id);

export interface Chip {
  k: string;
  cost: number;
  best: boolean;
}

/** The idea, playable: reach is bounded by the *language*, not the search. */
export class Reach {
  mode: "ops" | "structure" = "ops";

  /** Bounded closure: every program the law library generates. */
  private closure(root: Term, max = 40): Term[] {
    const seen = new Map<string, Term>([[key(root), clone(root)]]);
    let frontier: Term[] = [clone(root)];
    while (frontier.length && seen.size < max) {
      const next: Term[] = [];
      for (const t of frontier) {
        for (const r of ALL) {
          for (const u of rewritesAt(t, r)) {
            const k = key(u);
            if (!seen.has(k) && seen.size < max) {
              seen.set(k, u);
              next.push(u);
            }
          }
        }
      }
      frontier = next;
    }
    return [...seen.values()];
  }

  get chips(): Chip[] {
    const terms =
      this.mode === "ops"
        ? [clone(PROGRAMS.chain4), clone(PROGRAMS.dup)]
        : [...this.closure(PROGRAMS.chain4), ...this.closure(PROGRAMS.dup)];
    const uniq = new Map<string, Term>();
    for (const t of terms) uniq.set(key(t), t);
    const costs = [...uniq.values()].map((t) => total(t));
    const best = Math.min(...costs);
    return [...uniq.values()].map((t) => ({
      k: key(t),
      cost: total(t),
      best: total(t) === best,
    }));
  }

  get count(): number {
    return this.chips.length;
  }

  get bestCost(): number {
    return Math.min(...this.chips.map((c) => c.cost));
  }

  setMode(mode: "ops" | "structure"): void {
    this.mode = mode;
  }
}
