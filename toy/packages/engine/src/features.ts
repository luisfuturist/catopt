/** Static program features — the EVALUATION dimension's torch-free half.
 *
 * Mirrors `catopt_core.features.compute_features`: flops, bytes, depth,
 * reuse — computed from the term alone, never by running it.
 */

import type { Op, Term } from "./term.ts";
import { flops, isLeaf, isOp, key, numel, shape } from "./term.ts";

export interface ProgramFeatures {
  flops: number;
  bytes_read: number;
  bytes_written: number;
  temporary_bytes: number;
  depth: number;
  operations: number;
  param_leaves: number;
  reuse: number;
  parallelism: number;
}

export const DIMENSIONS = [
  "flops",
  "bytes_read",
  "bytes_written",
  "temporary_bytes",
  "depth",
  "operations",
  "param_leaves",
  "reuse",
  "parallelism",
] as const;

/** fp32 — the shipped dtype. */
const ITEMSIZE = 4;

export function features(root: Term): ProgramFeatures {
  const nodes: Op[] = [];
  const walk = (n: Term): void => {
    if (isLeaf(n)) return;
    n.args.forEach(walk);
    nodes.push(n);
  };
  walk(root);
  if (!nodes.length) {
    return {
      flops: 0,
      bytes_read: 0,
      bytes_written: 0,
      temporary_bytes: 0,
      depth: 0,
      operations: 0,
      param_leaves: 0,
      reuse: 0,
      parallelism: 0,
    };
  }
  const memo = new Map<string, number[]>();
  const sh = (n: Term): number[] => {
    const k = key(n);
    const hit = memo.get(k);
    if (hit) return hit;
    const s = shape(n);
    memo.set(k, s);
    return s;
  };
  let fl = 0;
  let read = 0;
  let written = 0;
  let temp = 0;
  const depthOf = new Map<string, number>();
  for (const n of nodes) {
    const out = numel(sh(n));
    written += out * ITEMSIZE;
    if (n !== root) temp += out * ITEMSIZE;
    fl += flops(n);
    read += n.args.reduce((p, a) => p + numel(sh(a)), 0) * ITEMSIZE;
    const kids = n.args.filter(isOp);
    depthOf.set(
      key(n),
      kids.length ? Math.max(...kids.map((a) => (depthOf.get(key(a)) ?? 0) + 1)) : 0,
    );
  }
  const operations = nodes.length;
  const depth = Math.max(0, ...depthOf.values());
  const bytes = read + written;
  return {
    flops: fl,
    bytes_read: read,
    bytes_written: written,
    temporary_bytes: temp,
    depth,
    operations,
    param_leaves: 0,
    reuse: bytes > 0 ? fl / bytes : 0,
    parallelism: depth > 0 ? operations / depth : operations,
  };
}

export const featureVector = (f: ProgramFeatures): number[] => DIMENSIONS.map((d) => f[d]);
