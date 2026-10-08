/** The toy e-graph engine behind the catopt learning app.
 *
 * A five-op algebra (`matmul` / `add` / `mul` / `sq` / `sqq`) with six
 * laws, a static cost model, a static feature profiler, and a numeric
 * referee — a faithful miniature of the four dimensions ADR 0003
 * separates.
 */

export * from "./term.ts";
export * from "./rules.ts";
export * from "./referee.ts";
export * from "./features.ts";
export * from "./programs.ts";
