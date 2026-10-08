import { PROGRAMS, features } from "engine";
import type { ProgramFeatures } from "engine";

export interface FeatureBar {
  k: string;
  v: number;
  max: number;
}

/** Features describe; the model predicts. Neither decides. */
export class Evaluation {
  tflops = 1.0;
  gbps = 100;
  launch = 5;
  readonly f: ProgramFeatures = features(PROGRAMS.chain);

  get features(): FeatureBar[] {
    const values = Object.values(this.f) as number[];
    const max = Math.max(...values.map((v) => Math.abs(v)), 1);
    return Object.entries(this.f).map(([k, v]) => ({
      k,
      v: Number((v as number).toFixed(3)),
      max,
    }));
  }

  get predicted(): string {
    const f = this.f;
    const compute = f.flops / (this.tflops * 1e12);
    const mem = (f.bytes_read + f.bytes_written) / (this.gbps * 1e9);
    const overhead = this.launch * 1e-6 * f.operations;
    const s = Math.max(compute, mem) + overhead;
    return s < 1e-6 ? `${(s * 1e9).toFixed(1)} ns` : `${(s * 1e6).toFixed(2)} µs`;
  }
}
