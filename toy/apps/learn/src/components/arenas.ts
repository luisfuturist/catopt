export interface Mod {
  n: string;
  isNew?: boolean;
}

export interface Arena {
  key: string;
  name: string;
  q: string;
  owns: string;
  mustNot: string;
  mods: Mod[];
}

/** The four dimensions — click a card to see its contract. */
export class Arenas {
  open: string | null = "sem";

  list: Arena[] = [
    {
      key: "sem",
      name: "SEMANTICS",
      q: "“Is it equivalent?”",
      owns: "ir · egraph · laws · typing · certificates. It decides what is the same function.",
      mustNot: "price, time, or run anything.",
      mods: [{ n: "ir" }, { n: "egraph" }, { n: "laws" }, { n: "typing" }, { n: "certs" }],
    },
    {
      key: "sea",
      name: "SEARCH",
      q: "“What should we explore?”",
      owns: "EGraph.run (saturation), the Engine port, the Strategy seam, the game, and the policies.",
      mustNot: "decide equivalence — it only proposes moves.",
      mods: [
        { n: "EGraph.run" },
        { n: "Engine" },
        { n: "Strategy" },
        { n: "game", isNew: true },
        { n: "policies", isNew: true },
      ],
    },
    {
      key: "eva",
      name: "EVALUATION",
      q: "“How does it look / might it run?”",
      owns: "cost models, features, the Pareto frontier, the performance model, the learned policy.",
      mustNot: "change the function, or prune the semantic space.",
      mods: [
        { n: "cost" },
        { n: "TargetProfile" },
        { n: "features", isNew: true },
        { n: "pareto", isNew: true },
        { n: "perf_model", isNew: true },
        { n: "trajectories", isNew: true },
        { n: "learned_policy", isNew: true },
      ],
    },
    {
      key: "exe",
      name: "EXECUTION",
      q: "“What actually ran?”",
      owns: "the Sink port, the Meter, adapters, runners, the CUDA graph runner, the failure taxonomy.",
      mustNot: "define semantics — it consumes equivalence, never produces it.",
      mods: [
        { n: "Sink" },
        { n: "Meter" },
        { n: "adapters" },
        { n: "runners" },
        { n: "cuda" },
        { n: "failures", isNew: true },
      ],
    },
  ];

  toggle(key: string): void {
    this.open = this.open === key ? null : key;
  }

  get selected(): Arena | undefined {
    return this.list.find((a) => a.key === this.open);
  }
}
