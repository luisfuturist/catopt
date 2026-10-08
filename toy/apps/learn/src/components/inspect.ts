import { PROGRAMS, clone, flops, shape, total } from "engine";
import type { Term } from "engine";

export interface Row {
  depth: number;
  label: string;
  out: string;
  flops: number;
}

/** Read a program as a tree: every node, its shape, and its work. */
export class Inspect {
  program: Term = clone(PROGRAMS.chain);
  picked = "chain";

  set(name: string): void {
    this.picked = name;
    this.program = clone(PROGRAMS[name]);
  }

  get rows(): Row[] {
    const out: Row[] = [];
    const walk = (t: Term, depth: number): void => {
      const s = shape(t);
      out.push({
        depth,
        label: "leaf" in t ? t.leaf : `${t.op}(${t.args.length})`,
        out: `[${s.join("×")}]`,
        flops: flops(t),
      });
      if (!("leaf" in t)) t.args.forEach((a) => walk(a, depth + 1));
    };
    walk(this.program, 0);
    return out;
  }

  get cost(): number {
    return total(this.program);
  }
}
