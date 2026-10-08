import { PROGRAMS, RULES, clone, key, rewritesAt } from "engine";
import type { RuleId, Term } from "engine";

export type Verdict = "pair-confluent" | "library-mediated" | "divergent" | "no-cofire";

export interface PairDef {
  a: RuleId;
  b: RuleId;
  term: string;
  note: string;
}

export interface Probe {
  def: PairDef;
  ra: string[];
  rb: string[];
  verdict: Verdict;
  mediator?: RuleId;
}

const LIBRARY: RuleId[] = RULES.map((r) => r.id);
const FOLD: RuleId = "square";

/** Pairwise law relations, measured by the toy's own saturation probe.
 *
 * Mirrors `tools/law_coherence.py`: fire each law once on a shared
 * redex, then ask whether the two one-step reducts rejoin — under the
 * pair alone (pair-confluent), else under the whole library
 * (library-mediated), else divergent.
 */
export class Coherence {
  sel = 3;
  withFold = true;

  pairs: PairDef[] = [
    { a: "comm", b: "cse", term: "dup", note: "a reordering inside a fold's redex" },
    { a: "square", b: "cse", term: "dup", note: "one law fires inside the other's redex" },
    { a: "comm", b: "fuse", term: "sqdup", note: "reordering a fusable pair" },
    {
      a: "expand",
      b: "fuse",
      term: "sqdup",
      note: "the critical pair — expansion kills the fuse redex",
    },
    {
      a: "expand",
      b: "square",
      term: "sqdup",
      note: "an inverse pair — only one side has a redex here",
    },
  ];

  /** Bounded closure: every term reachable from `roots` under `rules`. */
  private close(roots: readonly Term[], rules: readonly RuleId[], depth: number): Set<string> {
    const seen = new Set(roots.map(key));
    let frontier = [...roots];
    for (let d = 0; d < depth && frontier.length; d++) {
      const next: Term[] = [];
      for (const t of frontier) {
        for (const r of rules) {
          for (const u of rewritesAt(t, r)) {
            const k = key(u);
            if (!seen.has(k)) {
              seen.add(k);
              next.push(u);
            }
          }
        }
      }
      frontier = next;
    }
    return seen;
  }

  /** Do the two reduct sets rejoin under `rules` within the bound? */
  private rejoin(ra: readonly Term[], rb: readonly Term[], rules: readonly RuleId[]): boolean {
    const a = this.close(ra, rules, 6);
    const b = this.close(rb, rules, 6);
    for (const k of a) if (b.has(k)) return true;
    return false;
  }

  private probeFor(def: PairDef): Probe {
    const t = clone(PROGRAMS[def.term]);
    const ra = rewritesAt(t, def.a);
    const rb = rewritesAt(t, def.b);
    const base = { def, ra: ra.map(key), rb: rb.map(key) };
    if (!ra.length || !rb.length) return { ...base, verdict: "no-cofire" };
    if (this.rejoin(ra, rb, [def.a, def.b])) return { ...base, verdict: "pair-confluent" };
    const lib = LIBRARY.filter((r) => this.withFold || r !== FOLD);
    if (!this.rejoin(ra, rb, lib)) return { ...base, verdict: "divergent" };
    // Find the third law that bridges the reducts, if one does alone.
    for (const r of lib) {
      if (r === def.a || r === def.b) continue;
      if (this.rejoin(ra, rb, [def.a, def.b, r])) {
        return { ...base, verdict: "library-mediated", mediator: r };
      }
    }
    return { ...base, verdict: "library-mediated" };
  }

  get current(): Probe {
    return this.probeFor(this.pairs[this.sel]);
  }

  get verdictLabel(): string {
    const v = this.current.verdict;
    if (v === "pair-confluent") return "confluent — the pair rejoins on its own";
    if (v === "library-mediated") {
      return this.current.mediator
        ? `library-mediated — ${this.current.mediator} bridges the reducts`
        : "library-mediated — the wider library bridges the reducts";
    }
    if (v === "divergent") return "divergent — no common reduct in this library";
    return "no shared redex on this term";
  }

  get verdictTone(): string {
    const v = this.current.verdict;
    return v === "divergent" ? "text-bad" : v === "no-cofire" ? "text-faint" : "text-eva";
  }

  get verdictNote(): string {
    if (this.current.verdict === "divergent" && !this.withFold) {
      return "Expanding first kills the fuse redex for good. Bring the fold back — it is the missing mediator.";
    }
    if (this.current.mediator) {
      return "The fold is the expansion's definitional inverse: it rebuilds the fuse redex. One equality, both directions — the same fix the real library shipped as silu_fold.";
    }
    if (this.current.def.a === "comm" || this.current.def.b === "comm") {
      return "A reordering rarely breaks a redex — most commutation pairs are trivially confluent.";
    }
    return "";
  }

  pick(i: number): void {
    this.sel = i;
  }
}
