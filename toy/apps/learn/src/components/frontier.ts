export interface Candidate {
  name: string;
  flops: number;
  memory: number;
}

type Dim = "flops" | "memory";

/** The Pareto frontier: keep the landscape, don't scalarize early. */
export class Frontier {
  dims: Record<Dim, boolean> = { flops: true, memory: true };
  scalar = false;
  w = { flops: 0.5 };

  candidates: Candidate[] = [
    { name: "sdpa", flops: 120, memory: 80 },
    { name: "matmul", flops: 200, memory: 40 },
    { name: "reassoc", flops: 150, memory: 60 },
    { name: "pairing", flops: 90, memory: 140 },
    { name: "scan", flops: 170, memory: 30 },
    { name: "streaming", flops: 110, memory: 110 },
  ];

  get max(): Record<Dim, number> {
    return {
      flops: Math.max(...this.candidates.map((c) => c.flops)),
      memory: Math.max(...this.candidates.map((c) => c.memory)),
    };
  }

  get active(): Dim[] {
    const out: Dim[] = [];
    if (this.dims.flops) out.push("flops");
    if (this.dims.memory) out.push("memory");
    return out;
  }

  dominates(a: Candidate, b: Candidate): boolean {
    const dims = this.active;
    if (!dims.length) return false;
    return dims.every((d) => a[d] <= b[d]) && dims.some((d) => a[d] < b[d]);
  }

  onFrontier(c: Candidate): boolean {
    if (!this.active.length) return false;
    return !this.candidates.some((o) => this.dominates(o, c));
  }

  get frontier(): Candidate[] {
    return this.candidates.filter((c) => this.onFrontier(c));
  }

  score(c: Candidate): number {
    return this.active.reduce(
      (s, d) => s + (d === "flops" ? this.w.flops : 1 - this.w.flops) * c[d],
      0,
    );
  }

  get scalarWinner(): string {
    if (!this.active.length) return "—";
    return [...this.candidates].sort((a, b) => this.score(a) - this.score(b))[0].name;
  }
}
