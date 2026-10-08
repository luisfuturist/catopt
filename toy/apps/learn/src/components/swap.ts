export interface Comp {
  slot: string;
  now: string;
  port: string;
}

export interface Msg {
  ok: boolean;
  text: string;
  detail: string;
}

/** Swap any component — the semantic contract holds. */
export class Swap {
  sel = "engine";
  msg: Msg = { ok: true, text: "", detail: "" };

  comps: Comp[] = [
    { slot: "engine", now: "EGraph (python)", port: "Engine — the saturation core" },
    { slot: "rules", now: "ALL_RULES", port: "RuleSet — the algebra" },
    { slot: "policy", now: "GreedyPolicy", port: "Policy — orders legal moves" },
    { slot: "profiler", now: "StaticProfiler", port: "Profiler — observes, never runs" },
    { slot: "model", now: "AnalyticalPerformanceModel", port: "PerformanceModel — ranks only" },
    { slot: "backend", now: "TorchSink", port: "Sink — lowers & verifies" },
  ];

  options: string[] = [
    "NativeEngine (Rust)",
    "a smaller law set",
    "RLPolicy",
    "a measured profiler",
    "a learned model",
    "a CPU sink",
  ];

  pick(slot: string): void {
    this.sel = slot;
    this.msg = { ok: true, text: "", detail: "" };
  }

  use(option: string): void {
    const keeps: Record<string, string> = {
      engine: "saturation stays exact",
      rules: "reach shrinks; correctness does not",
      policy: "the certificate still referees",
      profiler: "it still only observes",
      model: "it still only ranks",
      backend: "supported_ops re-bounds extraction",
    };
    this.msg = {
      ok: true,
      text: `swapped ${this.sel} → ${option}`,
      detail: `${keeps[this.sel]}. The semantic contract is untouched — that is the point of the four dimensions.`,
    };
  }
}
