/** The toy term language and its cost model.
 *
 * Mirrors `catopt_core.ir` + `catopt_core.cost` for a five-op algebra:
 * `matmul`, `add`, `mul`, plus the `scale2` the CSE rule produces and
 * the `sq` / `sqq` pair the coherence level's laws work over.
 */

export interface Leaf {
  leaf: string;
}

export interface Op {
  op: string;
  args: Term[];
}

export type Term = Leaf | Op;

export const isLeaf = (t: Term): t is Leaf => "leaf" in t;
export const isOp = (t: Term): t is Op => !isLeaf(t);

/** Shapes chosen so bracketing changes the FLOP count. */
export const SHAPES: Record<string, number[]> = {
  a: [2, 3],
  b: [3, 4],
  c: [4, 5],
  d: [5, 2],
  x: [4, 4],
};

export function shape(t: Term): number[] {
  if (isLeaf(t)) return SHAPES[t.leaf];
  const a = t.args.map(shape);
  if (t.op === "matmul") return [a[0][0], a[1][1]];
  return a[0];
}

export const numel = (s: number[]): number => s.reduce((p, q) => p * q, 1);

export function flops(t: Term): number {
  if (isLeaf(t)) return 0;
  const n = numel(shape(t));
  if (t.op === "matmul") return 2 * n * shape(t.args[0]).at(-1)!;
  if (t.op === "add" || t.op === "mul" || t.op === "scale2" || t.op === "sq") return n;
  // sqq(x, y) = x² + y² elementwise: two multiplies and an add per
  // element — the fused kernel costs the same flops as add(sq, sq);
  // its win is one dispatch, which a flop-meter cannot see.
  if (t.op === "sqq") return 3 * n;
  return 0;
}

export function total(t: Term): number {
  if (isLeaf(t)) return 0;
  return flops(t) + t.args.reduce((p, c) => p + total(c), 0);
}

export const clone = (t: Term): Term =>
  isLeaf(t) ? { ...t } : { op: t.op, args: t.args.map(clone) };

export const key = (t: Term): string =>
  isLeaf(t) ? t.leaf : `${t.op}(${t.args.map(key).join(",")})`;

export function render(t: Term, ind = ""): string {
  if (isLeaf(t)) return `${ind}<span style="color:#6ea8fe">${t.leaf}</span>\n`;
  return (
    `${ind}<span style="color:#f5a35c">${t.op}</span>\n` +
    t.args.map((a) => render(a, ind + "  ")).join("")
  );
}
