import { afterEach, describe, expect, it } from "vite-plus/test";

import { Coherence } from "../src/components/coherence.ts";
import { Compare } from "../src/components/compare.ts";
import { Course, STORAGE_KEY } from "../src/components/course.ts";
import { Reach } from "../src/components/reach.ts";
import { Tensions } from "../src/components/tensions.ts";

/** Install a Map-backed localStorage shim; returns the backing map. */
function withStorage(): Map<string, string> {
  const map = new Map<string, string>();
  (globalThis as unknown as { localStorage: unknown }).localStorage = {
    getItem: (k: string) => map.get(k) ?? null,
    setItem: (k: string, v: string) => {
      map.set(k, v);
    },
    removeItem: (k: string) => {
      map.delete(k);
    },
    clear: () => {
      map.clear();
    },
    key: () => null,
    length: 0,
  };
  return map;
}

afterEach(() => {
  delete (globalThis as unknown as { localStorage?: unknown }).localStorage;
});

describe("course gating", () => {
  it("cannot advance before passing the checkpoint", () => {
    const c = new Course();
    expect(c.i).toBe(0);
    expect(c.canAdvance).toBe(false);
    c.next();
    expect(c.i).toBe(0);
  });

  it("advances once the correct option is picked", () => {
    const c = new Course();
    c.answer(c.level.checkpoint.ok);
    expect(c.canAdvance).toBe(true);
    c.next();
    expect(c.i).toBe(1);
  });

  it("rejects a wrong answer but still explains", () => {
    const c = new Course();
    const wrong = c.level.checkpoint.opts.find((o) => o !== c.level.checkpoint.ok) ?? "";
    c.answer(wrong);
    expect(c.canAdvance).toBe(false);
    expect(c.answered(c.level.id)).toBe(true);
  });

  it("is complete only when every level is passed", () => {
    const c = new Course();
    for (let k = 0; k < c.n; k++) {
      c.goto(k);
      c.answer(c.level.checkpoint.ok);
    }
    expect(c.complete).toBe(true);
    expect(c.done).toBe(c.n);
  });

  it("every checkpoint's correct option is one of its options", () => {
    for (const l of new Course().levels) {
      expect(l.checkpoint.opts).toContain(l.checkpoint.ok);
    }
  });

  it("runs twelve levels, 0 through 11", () => {
    const c = new Course();
    expect(c.n).toBe(12);
    expect(c.levels.at(-1)?.id).toBe("coherence");
    expect(c.levels.map((l) => l.id).length).toBe(new Set(c.levels.map((l) => l.id)).size);
  });

  it("restart clears progress", () => {
    const c = new Course();
    c.answer(c.level.checkpoint.ok);
    c.next();
    c.restart();
    expect(c.i).toBe(0);
    expect(c.done).toBe(0);
  });
});

describe("reach", () => {
  it("structural rewriting reaches strictly more than op-level", () => {
    const r = new Reach();
    const ops = r.count;
    r.setMode("structure");
    expect(r.count).toBeGreaterThan(ops);
    expect(r.bestCost).toBeLessThan(220);
  });
});

describe("compare", () => {
  it("shows two equal programs at different costs", () => {
    const c = new Compare();
    expect(c.sameFunction).toBe(true);
    expect(c.rightCost).toBeLessThan(c.leftCost);
  });

  it("states the dimensions the costs depend on", () => {
    const c = new Compare();
    expect(c.shapes).toContain("2×3");
    expect(c.rule).toContain("2·m·n·p");
  });

  it("its worked arithmetic matches the engine's cost", () => {
    // a·(b·c) = 2·(3·4·5) + 2·(2·3·5) = 120 + 60 = 180
    // (a·b)·c = 2·(2·3·4) + 2·(2·4·5) =  48 + 80 = 128
    const c = new Compare();
    expect(c.leftBreakdown).toBe("2·(3·4·5) + 2·(2·3·5) = 120 + 60");
    expect(c.rightBreakdown).toBe("2·(2·3·4) + 2·(2·4·5) = 48 + 80");
    expect(c.leftCost).toBe(180);
    expect(c.rightCost).toBe(128);
  });
});

describe("persistence", () => {
  it("writes progress on answer and on advance", () => {
    const map = withStorage();
    const c = new Course();
    c.answer(c.level.checkpoint.ok);
    c.next();
    expect(map.has(STORAGE_KEY)).toBe(true);
    expect(c.i).toBe(1);
  });

  it("restores level and answers on init", () => {
    withStorage();
    const a = new Course();
    a.answer(a.level.checkpoint.ok);
    a.next();

    const b = new Course();
    b.init();
    expect(b.i).toBe(1);
    expect(b.restored).toBe(true);
    expect(b.passed(b.levels[0].id)).toBe(true);
    expect(b.savedLabel).toBe("restored from this browser");
  });

  it("resetProgress wipes the stored key and the in-memory state", () => {
    const map = withStorage();
    const c = new Course();
    c.answer(c.level.checkpoint.ok);
    c.next();
    c.resetProgress();
    expect(map.has(STORAGE_KEY)).toBe(false);
    expect(c.i).toBe(0);
    expect(c.done).toBe(0);
    expect(c.settings).toBe(false);
  });

  it("survives corrupt storage instead of crashing", () => {
    const map = withStorage();
    map.set(STORAGE_KEY, "{ not json");
    const c = new Course();
    c.init();
    expect(c.i).toBe(0);
    expect(map.has(STORAGE_KEY)).toBe(false);
  });

  it("clamps an out-of-range saved level", () => {
    const map = withStorage();
    map.set(STORAGE_KEY, JSON.stringify({ i: 999, answers: {} }));
    const c = new Course();
    c.init();
    expect(c.i).toBe(c.n - 1);
  });

  it("does nothing when storage is unavailable", () => {
    const c = new Course();
    c.answer(c.level.checkpoint.ok);
    c.next();
    expect(c.i).toBe(1);
  });
});

describe("tensions", () => {
  it("has a valid correct option per item", () => {
    for (const t of new Tensions().items) {
      expect(t.opts).toContain(t.ok);
    }
  });
});

describe("coherence", () => {
  it("a reordering inside a fold's redex rejoins under the pair alone", () => {
    const c = new Coherence();
    c.pick(0);
    expect(c.current.def.a).toBe("comm");
    expect(c.current.def.b).toBe("cse");
    expect(c.current.verdict).toBe("pair-confluent");
  });

  it("the critical pair diverges without the fold and is mediated with it", () => {
    const c = new Coherence();
    c.pick(3); // expand × fuse
    expect(c.current.ra.length).toBeGreaterThan(0);
    expect(c.current.rb.length).toBeGreaterThan(0);
    c.withFold = false;
    expect(c.current.verdict).toBe("divergent");
    c.withFold = true;
    expect(c.current.verdict).toBe("library-mediated");
    expect(c.current.mediator).toBe("square");
  });

  it("an inverse pair with no shared redex reports no-cofire", () => {
    const c = new Coherence();
    c.pick(4); // expand × square on sqdup — square has no redex there
    expect(c.current.verdict).toBe("no-cofire");
  });
});
