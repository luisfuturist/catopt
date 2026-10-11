# catopt

[![ci](https://github.com/luisfuturist/catopt/actions/workflows/ci.yml/badge.svg)](https://github.com/luisfuturist/catopt/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

**catopt is a rewrite game over neural-network computation graphs.** A
program becomes an e-graph; players take legal, certificate-replayable
moves — apply a law, saturate, mint a new operator mid-search — and the
winning extraction is delivered as a *verified, runnable* `nn.Module`.

```bash
uv sync
python demo.py                                                    # ~60s end-to-end tour
python -m catopt_discovery.play --domain torch --deliver          # play a real module; lower + verify the win
python -m catopt_discovery.play --domain gen --probe              # claim → deliver → verify → time per case
python -m catopt_discovery.play --zoo                             # the probe over the held-out model zoo
```

## What it measured

Every number is on the dev box (RTX 2050, fp64, launch-bound sizes —
see [Limits](#limits)); provenance in [`docs/results.md`](docs/results.md).
Timing probes raise dynamo's `recompile_limit`, which otherwise silently
degrades late `torch.compile` calls (baselines included) to eager.

| measurement | result | source |
|---|---|---|
| `chainfuse` — laws + `claim(gen_0)`, Triton-bound | synthetic composite (`x@A@B@C` + 6-op tail, 256×512): eager 315µs · compile 253µs · laws only 181µs · **delivered 86µs = 2.94× vs `torch.compile`**, verified 1.7e-05 — the fused tail exists only via the generated kernel | `genkernel` + `play --probe` |
| delivered vs `torch.compile` | spelled matmul chain **3.96×** (22µs vs 87µs — reassoc folds to 1 GEMM; Inductor keeps all three even under `freezing`); llama-MLP swiglu **2.11×**; pointwise fusion loses 0.82–0.88× — wins are structural, not fusion | `play --domain gen --probe` |
| held-out zoo probe | **11/18 delivered verified modules beat `torch.compile`** — LoRAAdapter 2.18×, WaveNetGate 1.51×, DeepEquilibrium 1.35×; every winner ran `claims=[]` (the `gk.profitable` measured referee filtered the minted claims — the wins are the laws) | `play.zoo_probe` |
| joint fwd+bwd step vs `torch.autograd` | **+13%** (68µs vs 78µs, verified 9.5e-07) — the derived VJP shares the forward's activation under one memo; unshared it costs 197µs | `training.joint_step` |
| weight-chain reassociation vs Inductor | **17.12× vs Inductor** (18.63× vs eager) at (k,d,B·T)=(16,512,4096) | `bench.suites.speedup.reassoc_scale` |
| bounded rewrites on a real checkpoint | **up to 1.32× vs eager / 1.25× vs Inductor** (stories15M) | `bench.suites.bounded.bounded_e2e` |
| contraction player n=40 vs `opt_einsum` | **0.87–0.99× randomised greedy** at equal wall-clock (0.44–0.60× deterministic) | `tools/contraction_guided_restart.py` |
| `rms_norm_fold` on the bench case | **−83.3% modeled cost, bitwise fp64-identical, 2.70× wall-clock** | `bench.suites.correctness.law_bench` |
| `silu_fold` on the bench case | **1.43–2.62× wall-clock** — a machine-found 3-cell mediator that restores confluence where `silu` expansion destroyed the `swiglu_fuse` redex | `bench.suites.correctness.law_bench` |
| `softmax_fold` on `ManualSoftmaxAttention` | **+13–27% eager, +29–48% CUDA-graph** vs raw | `tools/law_wallclock.py` |
| machine law pack | **16/16 store objects deployable** as an opt-in `RuleSet` (`machine_default(store)`); machine-only arm wins 3/22 zoo models the human library misses | `machine_pack` |
| laws as data | **71/71 fully serializable** (pattern + `cond`/`dspec` + derivation) | `catopt_core.laws.serialize` |
| referee totality | a legal move can never crash the referee — declines are honest refusals, never crashes | `catopt_core.egraph` |

### What it finds — and what it doesn't

| Regime | Verdict | Why |
|---|---|---|
| Deep weight chains | **WIN** | the e-graph reaches a weights-first form Inductor's post-grad graph cannot express |
| Shared-input projections / gated blocks | **WIN** | pairing fuses k projections into one GEMM + split views |
| Linear-attention scan lift | **WIN** | the affine-monoid scan lift fires and verifies fp64-exact |
| Generated pointwise fusion | **LOSS** | `claim(gen_*)` pays only where launch count dominates — Inductor already fuses pointwise inside a compiled graph |
| Whole-model E2E | **PARITY** | pairing fires per block and verifies fp64-exact, but wall time is ~parity |
| Launch-bound decode | **NEGATIVE** | fewer launches don't pay where launch overhead already dominates — the hypothesis is falsified |

This is **not a universal speedup.** Attention- and GEMM-bound code is
already optimal — expect parity there. The wins live where structure
exists.

## How it works

A tensor program is a morphism in a hypergraph monoidal category
([ADR 0002](project/adrs/0002-categorical-re-expression-thesis.md)).
Optimization is search in the quotient of the free category on the IR's
generators by the laws of the structures the program instantiates —
monoids, comonoids, traces, contraction. The interpretation is not
unique, and choosing the richest one is the creative act: the same graph
reads as plain composition, a shared-input comonoid (pairing), a
low-rank approximation (bounded), or a contraction problem.

Four dimensions stay independent
([ADR 0003](project/adrs/0003-evaluation-is-an-independent-dimension.md)):
a cost model must not change semantics, a policy must not decide
equivalence, a profiler must not run on the target.

```text
SEMANTICS ──▶ SEARCH ──▶ EVALUATION ──▶ EXECUTION
 "equivalent?" "explore?"  "how does it   "what actually
                            look / run?"    ran?"
```

| Dimension | Question | Reach it with |
|---|---|---|
| semantics | Is it equivalent? | `verify_certificate(ir.root, cert, strict=True)` — every delivered program ships a replayable derivation |
| search | What to explore? | `policy=` on `search`/`Optimizer.optimize`: a `Policy` (random / greedy / learned) orders the rules each iteration |
| evaluation | What might run well? | `criteria=PredictedCriterion(model)`; `SearchResult.frontier({...})`; `StaticProfiler` describes without running |
| execution | What actually ran? | the `Sink`/`Runner`/`Meter` ports; `catopt_core.failures` classifies what went wrong |

One call runs the whole game:

```python
from catopt_orchestrator import Optimizer
from catopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
out = opt(x)        # the same function as model(x) — verified, not spot-checked
```

From associativity alone the board reaches `(Q·Kᵀ)·V → Q·(Kᵀ·V)` — an
O(T²d) intermediate becomes O(Td²). Nobody wrote *this* transform down:
the e-graph derives it from associativity plus a shape-aware cost model,
the certificate proves it, the pipeline delivers it end to end.

## The game

catopt is, structurally, a game played on a **tower of morphisms** —
programs, rewrites, proofs and metarules are the same kind of thing at
different dimensions:

| level | cell | in catopt |
|---|---|---|
| 0 | objects | tensor types / shapes |
| 1 | morphisms | **programs** — a term `a ~> b` |
| 2 | rewrites | **laws** — `lhs ~ rhs` + guards |
| 3 | coherences | **derivations between laws** — a `lemma` proven from premises |
| ↑ | polymorphic forms | **declared objects** — folds, lifts, compositions written *as data* |

Two games are played on that tower:

- **The search game** — moves *apply* 2-cells. The e-graph is the board:
  every program *known equal* to yours, in one place; a move walks it,
  extraction picks the cheapest member.
- **The construction game** — moves *write new cells*: `fold`/`lift`/
  `compose`/`auto_cond`/`relax_guard`/`specialize`/`ingest`, all pure
  data over the `CONSTRUCTORS`/`ACTIONS` registries; the score is
  holdout-only pay the player cannot write into.

**The play layer** (`catopt_discovery.play`) turns it into an RL
environment: a domain is *data* — a `cases` corpus, a `case -> Board`
factory, a `legal` enumerator, a featurizer — behind one CLI
(`--domain meta|joint|search|torch|gen`, `--deliver`, `--probe`,
`--zoo`, `--json`). Boards: the meta-arena (moves mix `fire(law)`,
`saturate`, `declare`, `claim`, a certified `extract`), the *joint*
board (a forward program plus every derived gradient under one root),
`SearchEnv`, real `nn.Module`s under `TorchSource`'s `supported_ops`
bound — and `gen`, where the board **grows its own vocabulary**:
`genkernel` mints a handler entry for each elementwise subterm no
shipped handler names (non-elementwise children become bound
metavariables — `mul(y,relu(y))` over `y = x@A@B@C` yields
`("mul","X1",("relu","X1"))` with `X1` binding the folded matmul);
`claim(tag)` declares the handler's own pattern — sound by construction
— and `deliver` lowers to a verified `nn.Module` calling *generated*
`triton.jit` source written to inspectable `.py` files (the hybrid sink
prefers Triton, falls back to `torch.compile`).

Shared equipment:

- **Rules.** Laws are *derived from category theory*, not enumerated as
  op patterns — associativity, unit/interchange, traced-monoidal axioms,
  products, monoid carriers. Legality is semantic equivalence.
- **Referee.** `verify_certificate` replays the derivation on real terms
  (fp64), plus an eight-stage admission gauntlet for constructed cells.
  A player only chooses among legal moves: it can be slow, never wrong.
- **Score.** A pluggable, per-target cost model — Pareto utilities over
  FLOPs, measured time, closure blowup; in the construction game:
  *admitted objects that pay on held-out code*.

## What's data

**Laws, guards, objects and the op vocabulary are data**, not code — so
the machine can grow the object language without growing a pile of
Python.

- **Laws.** `catopt_core.laws.ALL_RULES` holds **71** rules, each an
  `R(name, lhs, rhs, law=, cond=, dspec=, tags=, derivation=)`; the
  measured kernel taxonomy is **47 axioms / 13 lemmas / 2 redundant**
  (`catopt_discovery.coherence --emit-basis`). All 71 serialize —
  pattern + declarative guards + derivation, no Python.
- **Guards.** Side conditions prefer the declarative `cond` DSL —
  JSON-serializable data such as `("and", ("rank-eq","a","b"))`, used by
  45/71 rules — over procedural hooks; a record that can't carry a hook
  flags `missing_hooks` rather than weakening silently.
- **Objects.** The unit of invention is a *declared object* — data the
  referee replays ([ADR 0004](project/adrs/0004-abstraction-as-move.md)).
  `evidence.py` stores (`--add-object`) and admits
  (`--admit-object KEY --gauntlet`); **ten admitted objects are now
  shipped laws** — the machine-found → shipped loop is closed
  ([promoted-laws.md](project/retros/promoted-laws.md)).
- **Op vocabulary.** `catopt_core.opmeta` is one registry of **166
  ops**; the discovery generator's alphabet is *derived by property
  test* (`catopt_discovery.vocab`), and new structural ops are
  declarable via `catopt_core.opdata` `OpDef`s
  ([plan 0019](project/plans/0019-ops-as-data.md)).

### The admission gauntlet

A synthesized object earns **`usable`** only by clearing the same
eight-stage gauntlet a shipped law faces — a failing gate stops the run,
it does not skip:

> reconstruct → full-data → measure → truth → novelty → typed-pay →
> closure → cert

`truth` is the interesting stage for a conditional candidate: the
gauntlet sweeps the guarded region itself, and `--auto-cond` mints the
smallest declarative `cond` covering the measured domain and re-runs.
`usable` means *usable by the pipeline* — promotion to `laws/` stays
manual ([admission-gauntlet.md](project/retros/admission-gauntlet.md)).

### The referee earns its keep

The verifier has caught real bugs — **including in the shipped
library** (all regression-tested):

| caught | how |
|---|---|
| three shipped matmul laws false on mixed-rank bindings | adversarial witness generation; now guarded |
| `select_mul` latently unsound — `mul` broadcasts along the *selected* axis | the view/index oracle; now guarded by `dim-eq-attr` |
| `reshape_transpose` fires 23× and cost-lowers yet is numerically false | the oracle rejects it |
| seven shipped guarded laws with measured-unclean regions | the derivable-gate audit; a measured counterexample outranks a derivation |
| a `claim`-binding skeleton-vs-concrete hole (two different bodies sharing an op skeleton) | the concrete-occurrence fix — `claim` now declares the handler's own pattern |
| two false `mul_slice_*_id` candidates | the numeric oracle rejected them in the corpus sweep |

## The learned player (RL)

The player is a **`Policy`** — it may only **reorder legal moves**,
never change equivalence. Corpus knowledge is the discovery: the
learned player's value lives where the choice is *not* order-invariant
(contraction ordering, shipped), not in law reordering — the extracted
term is *identical* under ~350 ordering arms while wall-clock swings up
to 40× ([law-meta-game.md](project/retros/law-meta-game.md)).

On the meta-arena's mixed board the learned arm (`LinearPolicy`) beats
scripted on mean reward *and* produces certified structural wins the
playbook can't reach — but only after the `claim(tag)` option move fused
the paying declare→handle line into one refereed action; consolidation
remains sampling luck, not learned preference
([meta-arena-player.md](project/retros/meta-arena-player.md)).

```bash
python tools/train_search_policy.py --device cuda   # supervised rule value
python tools/train_rl_policy.py     --device cuda   # REINFORCE over the search env
```

## Honest negatives

The negatives are load-bearing, not footnotes:

- **Generated pointwise fusion loses to Inductor.** The 2.94×
  `chainfuse` headline is a synthetic composite — laws alone reach
  1.34× and the minted kernel roughly doubles it; on the real zoo every
  delivered winner ran `claims=[]` after the measured referee dropped
  unprofitable claims.
- **`shippable` ≠ `usable`.** Corpus-gate success is not pipeline
  admission, and neither is a shipped law — promotion is manual.
- **Corpus-circularity, measured.** The pipeline's `shippable = 11`
  drops to **0** on real modules only — each firing site was a
  purpose-built spelling
  ([shippable-audit.md](project/retros/shippable-audit.md)).
- **Compile-time, inference-only.** Search is seconds per block; weight
  folding destroys per-layer gradients — no backward rewrite.
- **The framework is general; the law library is narrow.** An unmatched
  block is an opaque boundary, never a wrong answer.

## Related work

catopt sits at the intersection of equality saturation, tensor-graph
superoptimization, and verified rewriting. The contribution is not new
algebra — it is *automatic discovery + certified equivalence + a game
that delivers runnable artifacts*, end to end.

| Line of work | Shares | Differs |
|---|---|---|
| **egg / egglog** | e-graph, saturation, cost-guided extraction | catopt's laws are *categorical*, and it delivers a runnable module + certificate rather than a term. A differential oracle cross-checks against egglog on a law subset (`tests/test_egglog_oracle.py`). |
| **TASO / Tensat** | equivalence-preserving graph rewrites, cost extraction | backtracking substitution vs e-graph closure; catopt's laws are over algebraic structure and reach non-local forms (scan lifts, weight folds). |
| **babble / DreamCoder** | abstraction invention from e-graphs | they mint abstractions for compression/synthesis; catopt binds minted operators to *generated compiled kernels* inside the search, driving a runtime win. |
| **TVM / Ansor / Halide** | schedule search, cost models | they search *schedules over a fixed algorithm*; catopt searches *across algorithms* via algebraic laws. |
| **Alive2 / CompCert** | machine-checked equivalence | catopt's certificate is per-program *derivational replay* on real terms, not a whole-compiler proof. |
| **torch.compile / Inductor** | the measured baseline | op-level fusion cannot express transforms across runtime parameters (weight folding, reassociation) — which is where catopt's wins live. |

## Limits

- **Wins are regime-dependent** — the transform set is structural:
  pairing, folds, reassociation, carrier lifts. A dense-GEMM-bound model
  with no shared structure should expect parity.
- **Search is compile-time work** — seconds per block; monolithic
  saturation slows past ~8 blocks (the `Compositional` strategy exists
  for that).
- **Inference only** — weight folding destroys per-layer gradients.
- **Coverage gaps** — `matmul`+bias and grouped convs aren't pairable;
  reassociation needs unnormalized attention; masks must arrive
  materialized.
- **Dev-box numbers** — measured on an RTX 2050 (4 GB) / CPU.
  `calibrate()` re-targets the cost model, but magnitudes do not
  extrapolate to datacenter hardware.

## Repo layout

A uv-workspace monorepo — there is no `catopt` façade; import the domain
packages directly (plan 0008):

```text
packages/
  catopt-core/         torch-free engine: IR, e-graph, laws, cost, ports
  catopt-torch/        export/import bridge, TorchSink, contraction player
  catopt-carriers/     carrier laws / executors (scan, attention)
  catopt-cuda/         CUDA-graph runner
  catopt-orchestrator/ backend-neutral pipeline + morphisms
  catopt-discovery/    law discovery + the game layer (python -m catopt_discovery.<mod>)
  catopt-native/       optional PyO3/Rust search engine
tests/                 pytest suite (100% coverage on the engine packages)
bench/                 benchmark suites + registry (python -m bench)
tools/                 ratchets, probes, trainers, calibrators
docs/                  mechanism / evaluation / api / results
project/               ADRs, plans, retros (the design record)
demo.py                the ~60-second end-to-end tour
```

The hexagonal boundary is pinned by import-linter: `catopt_cuda` ▶
{`catopt_torch`, `catopt_carriers`} ▶ `catopt_orchestrator` ▶
`catopt_native` ▶ `catopt_core`. Core imports no `torch`, `numpy`, or
GPU library — it is a *sink* for adapter-pushed state. A new backend
implements `Sink`; nothing in core changes.

## Verification

```sh
uv run pytest                 # full suite (~7.5 min, serial — do NOT use -n auto)
.venv/bin/ty check            # typecheck — 0 errors required
.venv/bin/ruff check          # lint
.venv/bin/ruff format --check # formatting
.venv/bin/vulture             # dead code
.venv/bin/lint-imports        # hexagonal boundary contracts
.venv/bin/bandit -c .bandit.yaml -r packages       # security SAST
.venv/bin/semgrep --config .semgrep.yml packages   # dataflow (offline)
.venv/bin/python tools/radon_ratchet.py            # complexity ratchet
```

Coverage is pinned at 100% on the five engine packages;
`catopt-discovery` sits at ~99% under a ratchet floor. Manual stages
(network/slower): `pip-audit`, the semgrep registry scan,
`tools/runtime_types.sh`, `tools/mutmut.sh`. Full details in
[`AGENTS.md`](AGENTS.md).

## Docs, ADRs, and plans

| Doc | What it is |
|---|---|
| [`docs/mechanism.md`](docs/mechanism.md) | the conceptual pipeline — syntax → structure → search → certificate |
| [`docs/evaluation.md`](docs/evaluation.md) | the evaluation dimension — features, frontier, policies |
| [`docs/api.md`](docs/api.md) | the API surface — `Optimizer`, strategies, runners, criteria, ports |
| [`docs/results.md`](docs/results.md) | measured results, generated from the pinned baselines |
| [`bench/README.md`](bench/README.md) | the benchmark harness and its suite catalog |
| [`AGENTS.md`](AGENTS.md) | repo layout, verification commands, port contracts |
| [`project/adrs/`](project/adrs/) | 0002 the thesis · 0003 evaluation is a dimension · 0004 abstraction as a move |
| [`project/retros/`](project/retros/) | the measured record — wins and negatives alike |
| [`project/RESEARCH_WRITEUP.md`](project/RESEARCH_WRITEUP.md) | the claim, the verified results, the honest negatives |
| [`project/REPORT.md`](project/REPORT.md) | the research report — weights as programs, what was falsified |
