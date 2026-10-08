export interface Tension {
  text: string;
  opts: string[];
  ok: string;
  why: string;
  picked: string;
}

/** The hard cases — where the four dimensions earn their keep. */
export class Tensions {
  items: Tension[] = [
    {
      text: "Candidate A is 10% cheaper on compute; B is 40% cheaper but uses 3× the memory. What does the frontier keep?",
      opts: ["A only", "B only", "both"],
      ok: "both",
      why: "Neither dominates — A wins on memory, B on compute. Both stay until a target says which trade-off matters.",
      picked: "",
    },
    {
      text: "A rewrite is provably correct but 2× slower on your GPU. What should the search do?",
      opts: ["discard it", "keep it, price it high", "ban it"],
      ok: "keep it, price it high",
      why: "It is legal and equivalent, so it stays in the class. Cost only decides which member is extracted — and another target may prefer it.",
      picked: "",
    },
    {
      text: "An approximate rewrite stays inside an error budget but breaks a strict pointwise bound. Admissible?",
      opts: ["yes, if the certificate says so", "never", "only without a certificate"],
      ok: "yes, if the certificate says so",
      why: "Bounded rewrites are a lax case: the certificate carries the bound explicitly. Admissible exactly when the stated contract is met — never silently.",
      picked: "",
    },
    {
      text: "The performance model predicts a form will be slow, so the search skips it. What went wrong?",
      opts: ["nothing", "a soft ranking was used as a hard bound", "the model was wrong"],
      ok: "a soft ranking was used as a hard bound",
      why: "Feasibility (can the backend lower it?) may prune. A prediction may only rank. Skipping on a guess can lose the certified optimum.",
      picked: "",
    },
    {
      text: "A machine-discovered candidate law is false but would lower the cost model — and it fires 23 times on real models. What stops it?",
      opts: ["the numeric oracle", "the cost model", "the search policy"],
      ok: "the numeric oracle",
      why: "Real case (reshape_transpose): firing and paying are not truth. The same referee that guards human rewrites judges machine proposals — it said false, and the candidate never shipped.",
      picked: "",
    },
    {
      text: "A proposed law is true, new, and fires 24 times — but it never lowers cost and it blows the reachable set up ~40×. Ship it?",
      opts: ["ship it", "no ship"],
      ok: "no ship",
      why: "Real case (grammar:mul_distribute): firing is not paying, and a 36–44× closure blow-up is a search hazard. The ship gate is a conjunction — truth, novelty, fires, pays, verifies, replays, bounded closure.",
      picked: "",
    },
  ];

  answer(i: number, option: string): void {
    this.items[i].picked = option;
  }

  get score(): number {
    return this.items.filter((t) => t.picked === t.ok).length;
  }
}
