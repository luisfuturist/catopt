# catopt toy — a course in structural optimization

An interactive course on catopt's architecture: **one idea per level**, a worked
example, an interactive, and a **checkpoint you must pass before advancing**.
Scaffolded as a **Vite+ monorepo** with `vp`.

```bash
vp install             # install the workspace
vp -C apps/learn dev   # http://localhost:5173
vp check               # format + lint + typecheck (oxfmt / oxlint / tsgo)
vp run -r test         # vitest across the workspace
vp run -r build        # engine → dist, app → dist
```

## The ladder (basics → advanced)

| #   | Level                      | Idea                                |
| --- | -------------------------- | ----------------------------------- |
| 0   | a program is a tree        | ops over data, one root             |
| 1   | two programs, one function | equality is not syntactic           |
| 2   | the reachable set          | reach is the closure of your laws   |
| 3   | laws come from structure   | monoid / comonoid / trace / product |
| 4   | staying correct            | the certificate is the referee      |
| 5   | choosing among equals      | keep the frontier, don't scalarize  |
| 6   | searching the space        | a policy orders legal moves         |
| 7   | learning the policy        | trajectories, then reward           |
| 8   | measuring the machine      | features describe, a model predicts |
| 9   | the architecture           | four dimensions, swappable parts    |
| 10  | advanced tensions          | where the design gets hard          |
| 11  | laws about laws            | derivable · inverse · divergent     |

## Layout

```
apps/
  learn/                     the course
    index.html               the twelve levels + checkpoint shell
    src/main.ts              Alpine bootstrap
    src/style.css            Tailwind v4 (@import + @theme tokens)
    src/components/course.ts the level list, gating, progress
    src/components/*.ts      one interactive per level
    tests/course.test.ts     gating, reach, compare, tensions, coherence
packages/
  engine/                    the toy e-graph engine, as a library
    src/term.ts              the term language + cost model
    src/rules.ts             the six laws (+ rewritesAt)
    src/referee.ts           numeric equivalence on concrete matrices
    src/features.ts          ProgramFeatures, mirroring catopt_core.features
    src/programs.ts          the starting programs
    tests/engine.test.ts     vitest, incl. Catalan(3) = 5 bracketings
```

## What is real, and what is a toy

The **board** is a real five-op algebra (`matmul` / `add` / `mul` / `sq` /
`sqq`) with six laws, a static cost model, a static feature profiler, and
a numeric referee. It is a faithful miniature of the four dimensions —
the shapes are chosen so bracketing genuinely changes the FLOP count, the
referee checks `max|original − current|` on concrete matrices, and the
level-11 probe computes real critical-pair verdicts (confluent /
library-mediated / divergent) by bounded saturation.

The **toy's own numbers** — costs, reach counts, verdicts — are computed
live by the engine. The **measured numbers in the prose callouts** come
from the real Python system (`project/retros/`): the pipeline's two
machine-discovered laws (`select_mul` −25.9 % modeled, `softmax_fold`
+13–48 % wall-clock), the false-but-cost-lowering candidate the oracle
rejected, the contraction player's ~1.04× parity with best-of-N greedy at
n = 40, the ~10× enumeration-vs-learned-proposer yield gap, the
executor-routing inversion (modeled −20…30 %, measured 1.9–2.7× slower)
and its measured-corrections fix, and the 30-of-53 basis-law catalogue
whose one divergence `silu_fold` closed.

## Stack

- [Alpine.js](https://alpinejs.dev) — the interactivity
- [Tailwind CSS v4](https://tailwindcss.com) — via `@tailwindcss/vite`
- [Vite+](https://viteplus.dev) — `vp` for install / check / test / build
