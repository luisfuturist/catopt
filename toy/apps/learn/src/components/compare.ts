import { PROGRAMS, applyRule, equivalent, total } from "engine";
import type { Term } from "engine";

/** Two trees, one function — and a different price, with the arithmetic shown. */
export class Compare {
  readonly left: Term = PROGRAMS.chain; // a·(b·c)
  readonly right: Term = applyRule(PROGRAMS.chain, "assoc"); // (a·b)·c
  revealed = false;

  /** The dimensions the costs depend on — nothing here is shape-free. */
  get shapes(): string {
    return "a : 2×3   b : 3×4   c : 4×5";
  }

  get rule(): string {
    return "an m×n times an n×p matrix costs 2·m·n·p FLOPs";
  }

  get leftCost(): number {
    return total(this.left);
  }

  get rightCost(): number {
    return total(this.right);
  }

  /** a·(b·c): (b·c) then a·(bc). */
  get leftBreakdown(): string {
    return "2·(3·4·5) + 2·(2·3·5) = 120 + 60";
  }

  /** (a·b)·c: (a·b) then (ab)·c. */
  get rightBreakdown(): string {
    return "2·(2·3·4) + 2·(2·4·5) = 48 + 80";
  }

  get sameFunction(): boolean {
    return equivalent(this.left, this.right);
  }

  reveal(): void {
    this.revealed = true;
  }

  reset(): void {
    this.revealed = false;
  }
}
