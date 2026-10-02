# Stage 0 recon — four-dimension boundary audit

Stage 0 of `project/plans/0016-evaluation-dimension.md`, under
ADR 0003.  Goal: find every place a layer answers another dimension's
question, so the later stages know what to preserve and what to fix.

Method: two read-only passes over the repo (core+orchestrator;
torch/carriers/cuda/bench), then two verification passes on the two
highest-value threads.  Line numbers are as reported by those passes.

The four dimensions (ADR 0003): **semantics / search / evaluation /
execution**.

## Verdict

| Dimension | State |
|---|---|
| SEMANTICS | **Cleanest.** No forbidden imports; laws are cost-blind at the law level; every witness is certified. |
| SEARCH | Clean by imports; coupled to EVALUATION **through control flow** (see B). |
| EVALUATION | Most coupled: a cost signal terminates search, a memory signal truncates it, feasibility is smuggled through the cost channel. |
| EXECUTION | Well-isolated behind `Sink` / `Meter` / executors — but the measurement surface is duplicated and unclassified (see D). |

## A. Compliant — no action

- **Core imports nothing forbidden.** Zero `torch`/`numpy`/
  `catopt_torch`/`catopt_carriers`/`catopt_cuda` imports under
  `packages/catopt-core/src`; import-linter already pins this.
  Adapter state arrives by push only (`ops.py` registration,
  `carriers.get_carriers()`, `regime.py:186-211`).
- **Cost models read only IR terms and shapes.** `params.py:130`
  duck-types `.numel` on a stored value — backend-agnostic, fine.
- **Candidate timing goes through the port.** `autotune.py:730` uses
  `meter.time()`; the remaining `time.monotonic()`/`time.time()` calls
  are wall-clock budgets and stats, not module measurement.
- **Cost never decides equivalence or legality.** Law `check`/`derive`
  guards are shape/typing only; `_INVALID_COST` poisons provably
  ill-typed members; `extract_certified` filters on certified bounds;
  autotune runs `sink.verify` *before* timing (`autotune.py:714`).
- **`bench/` is purely observational.** Suites call `sink.verify` /
  `torch.allclose` as gates *over* pipeline output; nothing feeds an
  equivalence decision back into the e-graph or laws.
- **Adapters consume equivalence, never define it** (`sink.verify` →
  `report.verify_module`, `adapters.py:126-139`).

## B. EVALUATION ↔ SEARCH coupling (the main finding)

Not via imports — via control flow.  Three sites where an evaluation
signal bounds the semantic space:

1. **The cost model terminates saturation.** `stop="improving"` halts
   the run when extracted cost stops improving (`patience`), at
   `egraph/core.py:1499-1615`.  A cost model that underprices a
   not-yet-discovered member starves the search.  *Disposition:* fine
   as documented and opt-in — but the ADR tension is real
   (EVALUATION gates SEARCH completeness); make the truncation
   explicit in `stats`.
2. **A memory meter truncates the search.** `optimize.py:180-237`
   (`_current_memory_mb` / `_check_resources`) reads RSS and
   `meter.device_memory_mb()` and **raises** mid-search, aborting
   saturation.  Execution state decides which certified alternatives
   ever exist.  *Disposition:* rename/document as a **resource
   guardrail**, not a semantic cut, and record the truncation in
   `stats`.
3. **Feasibility is smuggled through the cost channel.**
   `cost/backend.py:129-132` expresses the hard `supported_ops` bound
   as a `+inf` price.  Mechanically honest, but a `+inf` from an
   overpriced *user* model is indistinguishable from "the backend
   cannot lower this".  *Disposition:* fine as designed; optional —
   a boolean `feasible` oracle behind the `Engine` port would separate
   the two signals.

## C. Core "names no tensor library" — letter holds, spirit strains

No `import torch` (import-linter passes), but the **law guards
duck-type torch tensors**.  ~30 sites across four files
(`laws/pairing.py`, `laws/specials.py`, `laws/factored.py`,
`laws/headshare.py`), funnelled through four helpers
(`_is_tensor`, `_detach`, `_sig`, `_exact_equal`).

Concrete torch-only call chains:

- `specials.py:92` — `_detach(s).cpu().contiguous().numpy().tobytes()`
  (bitwise content signature).
- `pairing.py:869-875` — the same chain in
  `share_duplicate_param_slices`; plus `t.dim()` (`:851`),
  `new_zeros` (`:905`), `.detach()` (`:908`).
- `pairing.py:33-37` — `_is_tensor` requires `shape`/`dtype`/`dim`
  (`.dim` is torch-only).

The values are pushed by the torch adapter
(`torch_bridge.py:328,381` — `source_tensors: dict[str, torch.Tensor]`,
cloned from `export_to_ir`) and passed through unchanged by the
orchestrator (`optimize.py:1526,1567,1600`).  The code knows it:
`factored.py:50` and `pairing.py:28` say "no tensor library named".

*Disposition:* **extract-behind-port** — one `WeightInspector` /
`TensorBytes` protocol (shape, dtype, raw-bytes signature, slicing)
registered by the adapter alongside `source_tensors`, replacing the
four helpers; all call sites are already centralised.  Cheaper
alternative: widen the claim in the docstrings to "torch-API
duck-typed".  Either way, this is an honesty item, not a correctness
bug.

## D. The measurement surface — duplicated, unclassified, thin provenance

Four **distinct** measurement mechanisms with no shared contract
except #1:

| # | Mechanism | File | Shared contract? |
|---|---|---|---|
| 1 | `TorchMeter.time` | `catopt_torch/meter.py:40-74` | implements `Meter`/`TimingResult` |
| 2 | `calibrate()` probes | `catopt_torch/calibrate.py:127-526` | parallel loops; does **not** use `Meter` |
| 3 | `benchkit.Runner` | `bench/benchkit/runner.py:64-88` | separate engine |
| 4 | ad-hoc `perf_counter` loops | bench suites, `demo.py` | none |

Gaps against ADR 0003:

1. **Parallel measurement engines** — #2/#3/#4 bypass the `Meter`
   port.  *Disposition:* extend — share one timing helper; `calibrate`
   is the worst offender.
2. **No failure taxonomy.**  Autotune statuses are *stage*-only
   (`build_failed` / `verify_error` / `time_failed`), and the error is
   just `type(e).__name__` (`autotune.py:706-736`); OOM vs timeout vs
   NaN vs device-absent are indistinguishable.  *Disposition:*
   new-work.
3. **Swallowed failures leave no record.**  `catopt_cuda/runners.py:71-72`
   bare `except Exception` → silent `drop_cuda_graph()`;
   `calibrate.py:476-481,633,657,668,681` collapse to DEBUG/fallback;
   `bench/benchkit/runner.py:68-80` kills a case on one variant's
   exception.  *Disposition:* extend — record a classified reason.
4. **Provenance thin on the production path.**  `TargetProfile` carries
   `device` / `measured_at` / `meta={dtype, torch, platform}`
   (`calibrate.py:685-701`) but **no git sha, driver/CUDA version,
   seed, warmup or iteration count**; `TimingResult` carries no
   provenance at all.  Bench is stronger (`bench/benchkit/env.py:62-89`
   records git sha+dirty, torch, CUDA version, device, argv, UTC) but
   still lacks seed and driver.  *Disposition:* new-work — extend
   `TimingResult` / profile `meta`.
5. **No timeout mechanism anywhere.**  Hangs are unclassified crashes.
   *Disposition:* new-work.

## E. Minor

- `regime.py` `_force_carrier` / `prefer_executor` override extraction
  to serve a carrier form — a ranking override with honest
  `forced`/`degraded` flags.  *Fine.*
- Bench suites time ad hoc (`e2e_llm.py:455-468`,
  `bench_e2e.py:43-47`, `e2e_model.py:392-410`) despite
  `benchkit.Runner`'s "sole timing" claim (`runner.py:3-5`).
  *Extend* — route them through the runner.

## What this means for the stages

- **B.7/B.8** → stages 1 and 5: make search truncation explicit in
  `stats`; keep the policy out of the hot loop.
- **B.9** → stage 1: an optional `feasible` oracle beside the cost
  channel.
- **C** → stage 2 (or a small prerequisite port): `WeightInspector`.
- **D.1–D.5** → stage 3, precisely its scope: one measurement contract,
  a failure taxonomy, provenance, a timeout.
- **E.2** → stage 12.

None of these block stages 1–2; D is exactly what stage 3 is for.
