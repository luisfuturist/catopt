"""Wall-clock verification for the two shipped machine-discovered laws.

``tools/law_impact.py`` / ``tools/law_pipeline.py`` measured the laws
under the pipeline's *own* cost model — ``executor_cost_for``'s
roofline-plus-per-dispatch proxy — and ``law_bench`` timed only the
registered synthetic terms.  Nothing so far has timed the *real
models* end-to-end.  This tool closes that gap for:

* ``select_mul`` — ``mul(select(u,d,i), select(v,d,i)) ->
  select(mul(u,v),d,i)``, which fires on the five SSM-family models
  (SelectiveSSM, DiagDenseSSM, DiagonalSSM, HybridBlock,
  TwoLayerHybrid).  Modeled drop: 17.9-25.9 %.
* ``softmax_fold`` — ``div(exp(u), sum(exp(u),dim,keepdim)) ->
  softmax(u,dim)``, which fires on ManualSoftmaxAttention.  Modeled
  drop: 19.0 %.

For every (model, size) case the tool builds three arms:

* ``raw`` — the unoptimized ``nn.Module``;
* ``opt`` — ``Optimizer(backend=TorchBackend())`` search + lower on
  the composed ``default_rules()`` (the shipped path);
* ``abl`` — the same pipeline on ``default_rules()`` *minus* the law
  under test — the matched ablation isolating this law's marginal
  contribution from every other rewrite the pipeline applies.

Each arm is timed two ways on CUDA:

* ``eager`` — steady-state module calls, per-iteration
  ``torch.cuda.synchronize``, median / IQR over >=200 iterations after
  >=50 warmup;
* ``graph`` — the same call captured into a ``torch.cuda.CUDAGraph``
  (manual capture, identical for all three arms) and timed the same
  way; replay removes Python/launch overhead, isolating the kernel
  work the laws actually change.

It also records the pipeline's modeled costs (``dag_cost`` of the
export root vs the extracted term, and the ablated extraction) so the
measured deltas can be set directly against the modeled claims, plus
the lowering's ``sink.verify`` result.

Honesty contract: if the models are too small for wall-clock to show
the modeled dispatch-count win — or if the "optimized" module is
*slower* — the table says so; this tool exists because the
dispatch-proxy claim needed a real-clock check, not to manufacture
one.

Run::

    .venv/bin/python tools/law_wallclock.py
    .venv/bin/python tools/law_wallclock.py --json /tmp/lw.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from catopt_core.cost import dag_cost
from catopt_orchestrator import Optimizer
from catopt_orchestrator.optimize import default_rules
from catopt_torch.backend import TorchBackend
from catopt_torch.models import ManualSoftmaxAttention
from catopt_torch.models.hybrid import HybridBlock, TwoLayerHybrid
from catopt_torch.models.ssm import (
    DiagDenseSSM,
    DiagonalSSM,
    SelectiveSSM,
)

#: Search budget — mirrors the test suite's pipeline runs.
_MAX_ITERS = 6
_MAX_ENODES = 200_000

#: Timing budget — warmup and timed iterations per arm.
_WARMUP = 50
_ITERS = 200

#: Graph-capture warmup iterations on the side stream.
_CAPTURE_WARMUP = 5

_DEV_CPU = torch.device("cpu")
_DEV_CUDA = torch.device("cuda")


# ---------------------------------------------------------------------------
#  Model cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One (model, input-shape) measurement point."""

    name: str
    law: str
    build: Any  # () -> nn.Module
    shape: tuple[int, ...]


def _cases() -> list[Case]:
    """Return every (model, size) case for both laws.

    ``small`` mirrors the dims the impact/shape retros exported;
    ``large`` doubles the unrolled sequence and the dims — the
    "larger batch" axis — while staying inside a few seconds of
    saturation each.
    """
    return [
        # -- select_mul: the five SSM-family models -------------------
        Case(
            "SelectiveSSM small",
            "select_mul",
            lambda: SelectiveSSM(8, 8, 4),
            (4, 8),
        ),
        Case(
            "SelectiveSSM large",
            "select_mul",
            lambda: SelectiveSSM(16, 16, 8),
            (8, 16),
        ),
        Case(
            "DiagDenseSSM small",
            "select_mul",
            lambda: DiagDenseSSM(8, 8, 4),
            (4, 8),
        ),
        Case(
            "DiagDenseSSM large",
            "select_mul",
            lambda: DiagDenseSSM(16, 16, 8),
            (8, 16),
        ),
        Case(
            "DiagonalSSM small",
            "select_mul",
            lambda: DiagonalSSM(8, 8, 4),
            (4, 8),
        ),
        Case(
            "DiagonalSSM large",
            "select_mul",
            lambda: DiagonalSSM(16, 16, 8),
            (8, 16),
        ),
        Case(
            "HybridBlock small",
            "select_mul",
            lambda: HybridBlock(8, 8, 8, 4, 2),
            (4, 8),
        ),
        Case(
            "HybridBlock large",
            "select_mul",
            lambda: HybridBlock(16, 16, 16, 8, 2),
            (8, 16),
        ),
        Case(
            "TwoLayerHybrid small",
            "select_mul",
            lambda: TwoLayerHybrid(8, 8, 8, 4, 2),
            (4, 8),
        ),
        Case(
            "TwoLayerHybrid large",
            "select_mul",
            lambda: TwoLayerHybrid(16, 16, 16, 8, 2),
            (8, 16),
        ),
        # -- softmax_fold: the manual-softmax attention --------------
        Case(
            "ManualSoftmaxAttn small",
            "softmax_fold",
            lambda: ManualSoftmaxAttention(16),
            (2, 8, 16),
        ),
        Case(
            "ManualSoftmaxAttn large",
            "softmax_fold",
            lambda: ManualSoftmaxAttention(64),
            (8, 32, 64),
        ),
    ]


# ---------------------------------------------------------------------------
#  Optimization arms
# ---------------------------------------------------------------------------


def _without(rules: Any, name: str) -> Any:
    """Return *rules* minus the rule called *name* (matched ablation)."""
    return replace(
        rules,
        name=f"{rules.name}-{name}",
        rules=tuple(r for r in rules.rules if r.name != name),
        priorities={
            k: v for k, v in rules.priorities.items() if k != name
        },
    )


@dataclass
class Arm:
    """One lowered arm plus its modeled-cost record."""

    module: Any
    term_cost: float
    fires: int
    verified: str
    executor: str


@dataclass
class CaseResult:
    """Everything measured for one case."""

    case: str
    law: str
    fires_full: int = 0
    fires_abl: int = 0
    root_cost: float = 0.0
    full_cost: float = 0.0
    abl_cost: float = 0.0
    terms_equal: bool = True
    verified_full: str = "-"
    verified_abl: str = "-"
    executor_full: str = "-"
    executor_abl: str = "-"
    times: dict[str, dict[str, Any]] = field(default_factory=dict)
    note: str = ""


def _optimize(
    opt: Optimizer, model: Any, x: torch.Tensor, rules: Any
) -> tuple[Any, Any]:
    """Search + lower *model* under *rules*; return ``(res, low)``."""
    res = opt.search(
        model,
        x,
        rules=rules,
        max_iterations=_MAX_ITERS,
        max_enodes=_MAX_ENODES,
    )
    low = opt.lower(res, x, verify=True)
    return res, low


def _arms(case: Case) -> tuple[Any, torch.Tensor, CaseResult]:
    """Build the raw/opt/abl modules for *case* (CPU, fp64)."""
    torch.manual_seed(0)
    model = case.build().eval().double()
    x = torch.randn(*case.shape, dtype=torch.float64)

    opt = Optimizer(backend=TorchBackend())
    full = default_rules()
    abl = _without(full, case.law)

    res_f, low_f = _optimize(opt, model, x, full)
    res_a, low_a = _optimize(opt, model, x, abl)

    cost_fn = res_f.cost_fn
    row = CaseResult(
        case=case.name,
        law=case.law,
        fires_full=res_f.stats["rule_fires"].get(case.law, 0),
        fires_abl=res_a.stats["rule_fires"].get(case.law, 0),
        root_cost=dag_cost(res_f.ir.root, cost_fn),
        full_cost=dag_cost(res_f.term, cost_fn),
        abl_cost=dag_cost(res_a.term, cost_fn),
        terms_equal=res_f.term == res_a.term,
        verified_full=(
            "pass"
            if low_f.verified and low_f.verified.passed
            else "FAIL"
        ),
        verified_abl=(
            "pass"
            if low_a.verified and low_a.verified.passed
            else "FAIL"
        ),
        executor_full=type(low_f.module).__name__,
        executor_abl=type(low_a.module).__name__,
    )
    modules = {"raw": model, "opt": low_f.module, "abl": low_a.module}
    return modules, x, row


# ---------------------------------------------------------------------------
#  Timing
# ---------------------------------------------------------------------------


def _synced_median(
    fn: Any, warmup: int = _WARMUP, iters: int = _ITERS
) -> dict[str, float]:
    """Time *fn* per-iteration with device sync; return median/IQR ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {
        "median": statistics.median(ts),
        "p25": statistics.quantiles(ts, n=4)[0],
        "p75": statistics.quantiles(ts, n=4)[2],
        "min": min(ts),
        "mean": statistics.mean(ts),
    }


def _capture(fn: Any, x: torch.Tensor) -> Any:
    """Capture ``fn(x)`` into a CUDA graph; return a replay thunk.

    Manual capture for every arm (raw ``nn.Module``s and the lowered
    executors alike) so the graph path is uniform — no arm benefits
    from a different capture mechanism.
    """
    static_in = x.detach().clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(_CAPTURE_WARMUP):
            fn(static_in)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(static_in)
    return graph.replay


def _to_cuda(module: Any) -> Any:
    """Move a module to CUDA and force eval mode."""
    return module.to(_DEV_CUDA).eval()


def _time_case(
    modules: dict[str, Any], x: torch.Tensor
) -> dict[str, dict[str, Any]]:
    """Time every arm eagerly and under CUDA-graph replay."""
    xc = x.to(_DEV_CUDA)
    out: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for name, module in modules.items():
            mod = _to_cuda(module)
            row: dict[str, Any] = {}
            row["eager"] = _synced_median(lambda m=mod: m(xc))
            try:
                replay = _capture(lambda t, m=mod: m(t), xc)
                row["graph"] = _synced_median(replay)
            except Exception as e:  # honest per-arm failure
                row["graph"] = {"error": f"{type(e).__name__}: {e}"}
            out[name] = row
    return out


# ---------------------------------------------------------------------------
#  Driver + report
# ---------------------------------------------------------------------------


def run() -> list[CaseResult]:
    """Measure every case; return the rows."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "law_wallclock requires CUDA — the dispatch-overhead "
            "claim it tests is a launch-bound story"
        )
    rows: list[CaseResult] = []
    for case in _cases():
        modules, x, row = _arms(case)
        row.times = _time_case(modules, x)
        # sanity: optimized output must agree with the raw module
        xc = x.to(_DEV_CUDA)
        with torch.inference_mode():
            ref = modules["raw"].to(_DEV_CUDA)(xc)
            for arm in ("opt", "abl"):
                got = modules[arm].to(_DEV_CUDA)(xc)
                if not torch.allclose(ref, got, rtol=1e-4, atol=1e-6):
                    row.note += f" {arm} output differs from raw!"
        rows.append(row)
        print(f"  done {case.name}", flush=True)
    return rows


def _ms(v: Any) -> str:
    """Render one timing entry as ``median [p25-p75]`` ms."""
    if "error" in v:
        return f"ERR {v['error'][:24]}"
    return f"{v['median']:.4f} [{v['p25']:.4f}-{v['p75']:.4f}]"


def _ratio(num: float, den: float) -> str:
    """Render *num/den* as a speedup string (>1 is faster)."""
    return f"{den / num:.3f}x" if num else "-"


def _report(rows: list[CaseResult]) -> str:
    """Render the full results table."""
    lines: list[str] = []
    lines.append(
        "== law_wallclock — do the shipped laws pay on REAL wall-clock? "
        "=="
    )
    lines.append(
        f"   device: {torch.cuda.get_device_name(0)}  "
        f"dtype: fp64  warmup {_WARMUP}  iters {_ITERS}  "
        "per-iter sync, median [p25-p75] ms"
    )
    lines.append("")
    for r in rows:
        lines.append(f"-- {r.case}  (law: {r.law}) --")
        lines.append(
            f"   fires full/abl: {r.fires_full}/{r.fires_abl}   "
            f"terms identical: {r.terms_equal}   "
            f"verify: {r.verified_full}/{r.verified_abl}   "
            f"exec: {r.executor_full}/{r.executor_abl}"
        )
        modeled_pipe = (r.root_cost - r.full_cost) / r.root_cost * 100
        modeled_law = (
            (r.abl_cost - r.full_cost) / r.abl_cost * 100
            if r.abl_cost
            else 0.0
        )
        lines.append(
            f"   modeled cost: root {r.root_cost:.4g} -> "
            f"abl {r.abl_cost:.4g} -> full {r.full_cost:.4g}   "
            f"(pipeline -{modeled_pipe:.1f} %, law -{modeled_law:.1f} %)"
        )
        for mode in ("eager", "graph"):
            arms = r.times
            raw = arms["raw"][mode]
            opt_ = arms["opt"][mode]
            abl_ = arms["abl"][mode]
            speed = (
                _ratio(opt_["median"], raw["median"])
                if "median" in opt_ and "median" in raw
                else "-"
            )
            marg = (
                _ratio(opt_["median"], abl_["median"])
                if "median" in opt_ and "median" in abl_
                else "-"
            )
            lines.append(
                f"   {mode:<5} raw {_ms(raw):>24}  opt {_ms(opt_):>24} "
                f" abl {_ms(abl_):>24}   "
                f"opt-vs-raw {speed}  opt-vs-abl {marg}"
            )
        if r.note:
            lines.append(f"   note:{r.note}")
        lines.append("")
    return "\n".join(lines)


def _verdict(rows: list[CaseResult]) -> str:
    """Summarise whether modeled cost predicted measured wall-clock."""
    lines = ["== verdict =="]
    for r in rows:
        e_opt = r.times["opt"]["eager"].get("median")
        e_raw = r.times["raw"]["eager"].get("median")
        g_opt = r.times["opt"]["graph"].get("median")
        g_raw = r.times["raw"]["graph"].get("median")
        modeled = (r.abl_cost - r.full_cost) / r.abl_cost * 100
        lines.append(
            f"  {r.case:<26} modeled law-marginal -{modeled:5.1f} %   "
            f"measured opt/raw eager "
            f"{((e_raw - e_opt) / e_raw * 100) if e_opt else 0:+5.1f} %   "
            f"graph {((g_raw - g_opt) / g_raw * 100) if g_opt else 0:+5.1f} %"
        )
    return "\n".join(lines)


def _dump_json(path: str, rows: list[CaseResult]) -> None:
    """Write the machine-readable result."""
    payload = [
        {
            "case": r.case,
            "law": r.law,
            "fires_full": r.fires_full,
            "fires_abl": r.fires_abl,
            "terms_equal": r.terms_equal,
            "verified_full": r.verified_full,
            "verified_abl": r.verified_abl,
            "executor_full": r.executor_full,
            "executor_abl": r.executor_abl,
            "root_cost": r.root_cost,
            "abl_cost": r.abl_cost,
            "full_cost": r.full_cost,
            "times": r.times,
            "note": r.note,
        }
        for r in rows
    ]
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the wall-clock measurement and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    rows = run()
    print()
    print(_report(rows))
    print(_verdict(rows))
    if args.json:
        _dump_json(args.json, rows)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
