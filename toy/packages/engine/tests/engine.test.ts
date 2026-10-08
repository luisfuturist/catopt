import { describe, expect, it } from "vite-plus/test";

import {
  PROGRAMS,
  RULES,
  applicable,
  applyRule,
  clone,
  equivalent,
  features,
  hasMatch,
  key,
  rewritesAt,
  total,
} from "../src/index.ts";
import type { Term } from "../src/index.ts";

describe("cost model", () => {
  it("prices the expensive bracketing", () => {
    expect(total(PROGRAMS.chain)).toBe(180);
  });

  it("improves it by associativity", () => {
    expect(total(applyRule(PROGRAMS.chain, "assoc"))).toBe(128);
  });

  it("prices the duplicated elementwise program", () => {
    expect(total(PROGRAMS.dup)).toBe(48);
    expect(total(applyRule(PROGRAMS.dup, "cse"))).toBe(32);
  });

  it("leaves the cost unchanged under commutativity", () => {
    expect(total(applyRule(PROGRAMS.dup, "comm"))).toBe(48);
  });
});

describe("referee", () => {
  it("holds after every legal move", () => {
    for (const r of RULES) {
      expect(equivalent(PROGRAMS.dup, applyRule(PROGRAMS.dup, r.id))).toBe(true);
    }
    expect(equivalent(PROGRAMS.chain, applyRule(PROGRAMS.chain, "assoc"))).toBe(true);
  });

  it("rejects two different programs", () => {
    expect(equivalent({ leaf: "a" }, { leaf: "b" })).toBe(false);
  });
});

describe("features", () => {
  it("matches catopt_core.features on the chain", () => {
    const f = features(PROGRAMS.chain);
    expect(f.flops).toBe(180);
    expect(f.bytes_read).toBe(212);
    expect(f.bytes_written).toBe(100);
    expect(f.temporary_bytes).toBe(60);
    expect(f.depth).toBe(1);
    expect(f.operations).toBe(2);
    expect(f.reuse).toBeCloseTo(180 / 312, 6);
    expect(f.parallelism).toBe(2);
  });

  it("returns all-zero features for a bare leaf", () => {
    expect(features({ leaf: "x" }).operations).toBe(0);
  });
});

describe("legality", () => {
  it("only offers rules that can fire", () => {
    expect(applicable(PROGRAMS.chain)).toEqual(["assoc"]);
    expect([...applicable(PROGRAMS.dup)].sort()).toEqual(["comm", "cse", "square"]);
    expect([...applicable(PROGRAMS.sqdup)].sort()).toEqual(["comm", "cse", "expand", "fuse"]);
    expect(hasMatch(PROGRAMS.chain, "cse")).toBe(false);
  });

  it("renders a stable key per program", () => {
    expect(key(PROGRAMS.chain)).toBe("matmul(a,matmul(b,c))");
  });
});

describe("coherence laws", () => {
  it("the sq fold stays equivalent on the dup program", () => {
    const folded = applyRule(PROGRAMS.dup, "square");
    expect(key(folded)).toBe("add(sq(x),sq(x))");
    expect(equivalent(PROGRAMS.dup, folded)).toBe(true);
    expect(total(folded)).toBe(48);
  });

  it("the sq expand is the fold's definitional inverse", () => {
    const expanded = applyRule(PROGRAMS.sqdup, "expand");
    expect(key(expanded)).toBe("add(mul(x,x),mul(x,x))");
    expect(equivalent(PROGRAMS.sqdup, expanded)).toBe(true);
    // and expanding only one side kills the fuse redex
    const [oneSided] = rewritesAt(PROGRAMS.sqdup, "expand");
    expect(hasMatch(oneSided, "fuse")).toBe(false);
    expect(rewritesAt(PROGRAMS.sqdup, "expand")).toHaveLength(2);
  });

  it("the pair fuse keeps the value and the flop count — one dispatch", () => {
    const fused = applyRule(PROGRAMS.sqdup, "fuse");
    expect(key(fused)).toBe("sqq(x,x)");
    expect(equivalent(PROGRAMS.sqdup, fused)).toBe(true);
    expect(total(fused)).toBe(total(PROGRAMS.sqdup));
  });
});

describe("reach", () => {
  it("generates every bracketing of a four-matrix chain", () => {
    // Catalan(3) = 5 bracketings of a·b·c·d — the reachable set of one law
    const seen = new Set<string>([key(PROGRAMS.chain4)]);
    let frontier: Term[] = [clone(PROGRAMS.chain4)];
    while (frontier.length) {
      const next: Term[] = [];
      for (const t of frontier) {
        for (const u of rewritesAt(t, "assoc")) {
          const k = key(u);
          if (!seen.has(k)) {
            seen.add(k);
            next.push(u);
          }
        }
      }
      frontier = next;
    }
    expect(seen.size).toBe(5);
  });
});
