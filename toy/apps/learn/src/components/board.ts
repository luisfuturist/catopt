import {
  PROGRAMS,
  RULES,
  applyRule,
  clone,
  evalTerm,
  hasMatch,
  key,
  maxDiff,
  render,
  total,
} from "engine";
import type { RuleId, Term } from "engine";

export interface Variant {
  k: string;
  c: number;
}

export interface CertStep {
  rule: string;
  from: number;
  to: number;
}

/** The board: apply legal moves, price the class, remember the path. */
export class Board {
  picked = "chain";
  root: Term = clone(PROGRAMS.chain);
  orig: Term = clone(PROGRAMS.chain);
  variants: Variant[] = [];
  cert: CertStep[] = [];
  verdict = "";

  constructor() {
    this.start("chain");
  }

  start(name: string): void {
    this.picked = name;
    this.root = clone(PROGRAMS[name]);
    this.orig = clone(PROGRAMS[name]);
    const c = total(this.root);
    this.variants = [{ k: key(this.root), c }];
    this.cert = [{ rule: "(start)", from: c, to: c }];
    this.verdict = "";
  }

  get cost(): number {
    return total(this.root);
  }

  get delta(): number {
    return total(this.orig) - this.cost;
  }

  get treeHtml(): string {
    return render(this.root);
  }

  get maxCost(): number {
    return Math.max(...this.variants.map((v) => v.c), 1);
  }

  get frontier(): Variant[] {
    return [...this.variants].sort((a, b) => a.c - b.c).slice(0, 8);
  }

  get ruleList(): { id: RuleId; label: string; applies: boolean }[] {
    return RULES.map((r) => ({
      id: r.id,
      label: r.label,
      applies: hasMatch(this.root, r.id),
    }));
  }

  apply(r: RuleId): void {
    if (!hasMatch(this.root, r)) return;
    const before = total(this.root);
    this.root = applyRule(this.root, r);
    const after = total(this.root);
    this.cert = [...this.cert, { rule: r, from: before, to: after }];
    const k = key(this.root);
    if (!this.variants.some((v) => v.k === k)) {
      this.variants = [...this.variants, { k, c: after }];
    }
  }

  replay(): void {
    const d = maxDiff(evalTerm(this.orig), evalTerm(this.root));
    this.verdict =
      d < 1e-9
        ? `certificate valid · max|original − current| = ${d.toExponential(1)} · ${this.cert.length - 1} step(s) replayed`
        : `MISMATCH ${d}`;
  }

  get verdictHtml(): string {
    if (!this.verdict) return "";
    const ok = this.verdict.startsWith("certificate");
    const colour = ok ? "#79e0a0" : "#ff6b6b";
    return `<span style="color:${colour}">${ok ? "✓ " : "✕ "}${this.verdict}</span>`;
  }
}
