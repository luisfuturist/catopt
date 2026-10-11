# catopt

[![ci](https://github.com/luisfuturist/catopt/actions/workflows/ci.yml/badge.svg)](https://github.com/luisfuturist/catopt/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

**catopt plays a verified rewrite game over neural-network graphs.** A
model is exported to an e-graph; players take only legal,
certificate-replayable moves — apply a law, saturate, mint a new
operator mid-search — and the winning program is delivered as a
runnable, verified `nn.Module`.

```bash
uv sync
python demo.py   # ~60s: optimize a gated block, replay the certificate,
                 # race vs eager and torch.compile
```

```python
from catopt_orchestrator import Optimizer
from catopt_torch import TorchBackend

opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x)
out = opt(x)   # verified-equal to model(x)
```

## Results

Measured on the dev box (RTX 2050, fp64, small sizes — see
[limits](#limits)); full provenance in
[`docs/results.md`](docs/results.md).

| | vs `torch.compile` |
|---|---|
| deep weight chains (reassoc to 1 GEMM) | **17.1×** |
| spelled `x@A@B@C` block | **3.96×** |
| `chainfuse` — laws + a kernel minted mid-search | **2.94×** |
| llama-style SwiGLU MLP | **2.11×** |
| held-out model zoo (22 architectures) | **11/18 verified wins**, best 2.18× |
| joint fwd+bwd step vs `torch.autograd` | **+13%** |
| pure pointwise fusion | **0.82× loss** — Inductor's home turf |

Wins are structural — weight folding, shared-input pairing, scan lifts —
the transforms op-level compilers cannot express. GEMM-bound code with
no structure sees parity; launch-bound decode can lose. This is not a
universal speedup.

## What makes it different

- **Laws from algebra, not op lists.** Rewrite rules derive from
  categorical structure (monoid carriers, traces, products), so the
  reachable set is the closure of a law library — larger than any fixed
  pass list.
- **Proof, not spot checks.** Every delivered program ships a replayable
  derivation; `verify_certificate` re-runs each step on real terms.
- **The board invents operators.** Construction moves write new rules
  mid-search; `claim` can bind a minted elementwise pattern to a
  *generated Triton kernel* and deliver it.
- **Four independent dimensions** — semantics / search / evaluation /
  execution. Cost models never change semantics; policies never decide
  equivalence.

## How it works

```text
module → export → e-graph saturation → extract (cost) → certify → lower
                      ▲                                    ↓
          laws as data (71 rules,      verified, runnable nn.Module
          all serializable)            (+ optional compile/CUDA-graph runner)
```

Discovery runs the same machinery on the law library itself:
`catopt_discovery.pipeline` (census → propose → verify → measure →
rank → emit) has shipped ten machine-found laws, and
`catopt_discovery.play` exposes the whole loop as an RL environment
(`--domain meta|joint|search|torch|gen`).

```bash
python -m catopt_discovery.play --domain torch --deliver  # optimize a real module
python -m catopt_discovery.play --domain gen --probe      # mint → claim → deliver → time
python -m catopt_discovery.play --zoo                     # the probe over held-out models
```

## Limits

- Inference only — weight folding destroys per-layer gradients.
- Compile-time search — seconds per block; very large models need the
  `Compositional` strategy.
- Regime-dependent — wins live where algebraic structure exists.
- Dev-box magnitudes — `tools/calibrate_profile.py` re-targets the cost
  model to your hardware.

## Packages

| package | role |
|---|---|
| `catopt-core` | torch-free engine: IR, e-graph, laws, cost, ports |
| `catopt-torch` | PyTorch export/lower/verify, `TorchBackend` |
| `catopt-carriers` | carrier laws and executors (scan, attention) |
| `catopt-cuda` | CUDA-graph runner |
| `catopt-orchestrator` | backend-neutral pipeline |
| `catopt-discovery` | law discovery + the game layer (`python -m catopt_discovery.<mod>`) |
| `catopt-native` | optional PyO3/Rust search engine |

The hexagonal boundary is enforced by import-linter: core imports no
`torch`, `numpy`, or GPU library — a new backend implements `Sink` and
nothing in core changes.

## Docs

| doc | contents |
|---|---|
| [`docs/mechanism.md`](docs/mechanism.md) | the conceptual pipeline — syntax → structure → search → certificate |
| [`docs/evaluation.md`](docs/evaluation.md) | features, calibration, criteria, policies |
| [`docs/api.md`](docs/api.md) | the API surface — `Optimizer`, strategies, runners, criteria |
| [`docs/results.md`](docs/results.md) | measured results, generated from pinned baselines |
| [`bench/README.md`](bench/README.md) | the benchmark harness |
| [`AGENTS.md`](AGENTS.md) | repo layout and the full verification gate list |

## Verification

```sh
uv run pytest                              # full suite (serial — no -n auto)
.venv/bin/ty check && .venv/bin/ruff check && .venv/bin/ruff format --check
.venv/bin/vulture && .venv/bin/lint-imports
.venv/bin/bandit -c .bandit.yaml -r packages
.venv/bin/semgrep --config .semgrep.yml packages
```
