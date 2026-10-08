export interface Checkpoint {
  q: string;
  opts: string[];
  ok: string;
  why: string;
}

export interface Level {
  id: string;
  title: string;
  objective: string;
  checkpoint: Checkpoint;
}

/** Where progress lives. Bump the suffix if the level list changes shape. */
export const STORAGE_KEY = "catopt-course-v1";

interface Saved {
  i: number;
  answers: Record<string, string>;
}

/** A localStorage shim that no-ops when storage is unavailable. */
function store(): Storage | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

/** The course shell: levels, gating, progress, and persistence. */
export class Course {
  levels: Level[] = [
    {
      id: "tree",
      title: "A program is a tree",
      objective: "Read a tensor program as a tree of operations over data.",
      checkpoint: {
        q: "In `(a·b)·c`, which node produces the final result?",
        opts: ["a", "the outer matmul", "c"],
        ok: "the outer matmul",
        why: "The root. Every other node feeds it. Optimization is choosing a different tree that computes the same thing — the root stays the answer.",
      },
    },
    {
      id: "equal",
      title: "Two programs, one function",
      objective: "Different trees can compute the same function — at different costs.",
      checkpoint: {
        q: "With a : 2×3, b : 3×4, c : 4×5, the two trees cost 180 and 128 FLOPs. Same function?",
        opts: ["yes", "no", "only if the shapes match"],
        ok: "yes",
        why: "Matrix multiplication is associative: bracketing changes the cost, never the result. That gap — same function, different cost — is the whole opportunity.",
      },
    },
    {
      id: "reach",
      title: "The reachable set",
      objective: "Reach is the closure of your laws, not the power of your search.",
      checkpoint: {
        q: "What bounds how many equivalent programs you can ever find?",
        opts: ["the search budget", "the laws you have", "the hardware"],
        ok: "the laws you have",
        why: "Search only explores what the laws generate. A bigger budget over a weak law set still finds nothing new.",
      },
    },
    {
      id: "structure",
      title: "Laws come from structure",
      objective: "A law is a consequence of algebraic structure, not an op pattern.",
      checkpoint: {
        q: "Why can't an op-level rule find `(Q·Kᵀ)·V → Q·(Kᵀ·V)`?",
        opts: [
          "it is not a valid rewrite",
          "it is a consequence of associativity plus shapes, not an op pattern",
          "it needs a special kernel",
        ],
        ok: "it is a consequence of associativity plus shapes, not an op pattern",
        why: "No op named 'attention' is involved. The law is associativity; the choice is driven by a shape-aware cost model. An op-pattern language cannot express that.",
      },
    },
    {
      id: "referee",
      title: "Staying correct",
      objective: "Free rewriting is dangerous. Certificates make it safe.",
      checkpoint: {
        q: "What makes a rewrite legal?",
        opts: [
          "the cost model approves it",
          "a law licenses it, and the certificate replays",
          "the search chose it",
        ],
        ok: "a law licenses it, and the certificate replays",
        why: "Legality is semantic. Cost and search have no vote — which is exactly why a learned player cannot produce a wrong program.",
      },
    },
    {
      id: "choice",
      title: "Choosing among equals",
      objective: "Among many equals, do not collapse to one scalar too early.",
      checkpoint: {
        q: "Why keep a Pareto frontier instead of picking the cheapest immediately?",
        opts: ["it is faster", "the hardware decides which region matters", "it uses more memory"],
        ok: "the hardware decides which region matters",
        why: "One scalar bakes in one machine's trade-off. The frontier keeps the options; the target picks.",
      },
    },
    {
      id: "search",
      title: "Searching the space",
      objective: "The space is exponential. A policy decides what to try.",
      checkpoint: {
        q: "What may a search policy never do?",
        opts: ["pick a bad move", "change what is legal", "stop early"],
        ok: "change what is legal",
        why: "A policy only orders legal moves. It can be wrong — it can pick a bad move — but it cannot make a move legal or illegal.",
      },
    },
    {
      id: "learn",
      title: "Learning the policy",
      objective: "A policy can be learned: from trajectories, or from reward.",
      checkpoint: {
        q: "You add a new rewrite rule at runtime. Must the learned policy be retrained?",
        opts: ["yes", "no, if actions are encoded structurally", "only on GPU"],
        ok: "no, if actions are encoded structurally",
        why: "Encode a rule by its shape and an unseen rule is still a point in the same space. Encode it by index and you would have to retrain.",
      },
    },
    {
      id: "measure",
      title: "Measuring the machine",
      objective: "Static features describe; hardware is truth; a model predicts.",
      checkpoint: {
        q: "What may a performance model never do?",
        opts: ["rank candidates", "prune the semantic space", "predict latency"],
        ok: "prune the semantic space",
        why: "Feasibility — can the backend lower this at all? — is a hard bound. Performance is a soft ranking. Pruning on a prediction loses certified alternatives.",
      },
    },
    {
      id: "architecture",
      title: "The architecture",
      objective: "Everything so far is one of four jobs. Keeping them apart is the design.",
      checkpoint: {
        q: "A cost model rejects a rewrite as unsound. Which rule did it break?",
        opts: [
          "search may not decide equivalence",
          "evaluation may not decide equivalence",
          "execution may not decide equivalence",
        ],
        ok: "evaluation may not decide equivalence",
        why: "EVALUATION ranks; SEMANTICS decides equivalence. The certificate is the referee — that separation is the whole architecture.",
      },
    },
    {
      id: "advanced",
      title: "Advanced tensions",
      objective: "Where the design actually gets hard.",
      checkpoint: {
        q: "Why is the certificate a hard filter rather than a reward term?",
        opts: [
          "it is cheaper to compute",
          "a soft reward lets the optimizer trade correctness for cost",
          "rewards are hard to compute",
        ],
        ok: "a soft reward lets the optimizer trade correctness for cost",
        why: "As a reward, correctness becomes spendable. As a hard filter, an unsound program is unrepresentable: search can only lose cost, never truth.",
      },
    },
    {
      id: "coherence",
      title: "Laws about laws",
      objective:
        "Laws relate — derivable, inverse, divergent. Measure the relations; fill the gaps.",
      checkpoint: {
        q: "Two laws fire on the same redex and their reducts never rejoin. What fills the gap?",
        opts: [
          "a bigger search budget",
          "a mediating law — often the definitional inverse of the rule that caused the split",
          "delete one of the two laws",
        ],
        ok: "a mediating law — often the definitional inverse of the rule that caused the split",
        why: "The real library's one measured divergence — silu_expand × swiglu_fuse — was filled by silu_fold, x·σ(x) → silu(x): the inverse of the expansion that killed the fuse redex. Divergent pairs went 2 → 0, and the mediator is a paying law in its own right.",
      },
    },
  ];

  i = 0;
  answers: Record<string, string> = {};
  settings = false;
  confirming = false;
  restored = false;

  /** Alpine calls this on mount: restore any saved progress. */
  init(): void {
    const s = store();
    if (!s) return;
    try {
      const raw = s.getItem(STORAGE_KEY);
      if (!raw) return;
      const data = JSON.parse(raw) as Partial<Saved>;
      if (typeof data.i === "number") {
        this.i = Math.max(0, Math.min(this.n - 1, Math.trunc(data.i)));
      }
      if (data.answers && typeof data.answers === "object") {
        this.answers = { ...data.answers };
      }
      this.restored = true;
    } catch {
      // corrupt or unreadable storage: start clean rather than crash
      s.removeItem(STORAGE_KEY);
    }
  }

  private save(): void {
    const s = store();
    if (!s) return;
    try {
      s.setItem(STORAGE_KEY, JSON.stringify({ i: this.i, answers: this.answers }));
    } catch {
      // quota or private mode: progress simply does not persist
    }
  }

  get level(): Level {
    return this.levels[this.i];
  }

  get n(): number {
    return this.levels.length;
  }

  get done(): number {
    return this.levels.filter((l) => this.passed(l.id)).length;
  }

  get complete(): boolean {
    return this.done === this.n;
  }

  get savedLabel(): string {
    return this.restored ? "restored from this browser" : "saved as you go";
  }

  passed(id: string): boolean {
    const lvl = this.levels.find((l) => l.id === id);
    return lvl ? this.answers[id] === lvl.checkpoint.ok : false;
  }

  answered(id: string): boolean {
    return this.answers[id] !== undefined;
  }

  answer(opt: string): void {
    this.answers = { ...this.answers, [this.level.id]: opt };
    this.save();
  }

  get picked(): string | undefined {
    return this.answers[this.level.id];
  }

  get canAdvance(): boolean {
    return this.passed(this.level.id);
  }

  next(): void {
    if (this.canAdvance && this.i < this.n - 1) {
      this.i += 1;
      this.save();
    }
  }

  prev(): void {
    if (this.i > 0) {
      this.i -= 1;
      this.save();
    }
  }

  goto(i: number): void {
    this.i = Math.max(0, Math.min(this.n - 1, i));
    this.save();
  }

  /** Clear answers and start over — keeps you on the course. */
  restart(): void {
    this.i = 0;
    this.answers = {};
    this.restored = false;
    this.save();
  }

  /** Settings: wipe stored progress entirely. */
  resetProgress(): void {
    const s = store();
    try {
      s?.removeItem(STORAGE_KEY);
    } catch {
      // nothing stored, nothing to clear
    }
    this.i = 0;
    this.answers = {};
    this.restored = false;
    this.confirming = false;
    this.settings = false;
  }

  toggleSettings(): void {
    this.settings = !this.settings;
    this.confirming = false;
  }
}
