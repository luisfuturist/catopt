export interface Scenario {
  text: string;
  illegal: boolean;
  why: string;
  picked: string;
}

/** Spot the leak: which of these break the four-dimension rule? */
export class Leaks {
  items: Scenario[] = [
    {
      text: "A cost model rejects a rewrite because it looks unsound.",
      illegal: true,
      why: "Cost is EVALUATION. Legality is SEMANTICS — the certificate decides it, never the price.",
      picked: "",
    },
    {
      text: "A policy picks a rule that makes the program slower.",
      illegal: false,
      why: "Legal — and merely bad. SEARCH may order moves any way it likes; the referee still guards correctness.",
      picked: "",
    },
    {
      text: "A static profiler runs the model on the GPU to get its features.",
      illegal: true,
      why: "Profiling is observation, not execution. Running the program is EXECUTION's job.",
      picked: "",
    },
    {
      text: "The performance model drops candidates it predicts will be slow.",
      illegal: true,
      why: "It may RANK, never prune. Pruning on a prediction loses certified alternatives.",
      picked: "",
    },
    {
      text: "A new backend declares which ops it can lower.",
      illegal: false,
      why: "Legal — that is the Sink's supported_ops: a hard feasibility bound, not a performance guess.",
      picked: "",
    },
    {
      text: "The verifier replays the derivation on real inputs.",
      illegal: false,
      why: "Legal — that is the referee's job: the certificate is re-checked, not trusted.",
      picked: "",
    },
  ];

  answer(i: number, illegal: boolean): void {
    this.items[i].picked = illegal ? "illegal" : "legal";
  }

  get score(): number {
    return this.items.filter((s) => s.picked === (s.illegal ? "illegal" : "legal")).length;
  }

  get answered(): number {
    return this.items.filter((s) => s.picked !== "").length;
  }
}
