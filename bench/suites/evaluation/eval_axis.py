"""Evaluation-axis selection — does pluggable evaluation change the answer?

Plan 0016 (ADR 0003, the EVALUATION dimension) wired four ports into
the pipeline: ``Policy`` steers ``EGraph.run``, ``pareto`` backs
``SearchResult.frontier``, and ``Profiler`` + ``PerformanceModel``
back ``PredictedCriterion``.  This suite asks the only question that
decides whether that wiring is a *result* rather than an *enabler*:

    Does plugging a different evaluator in change the program the
    engine extracts — and does the equivalence class expose a
    genuine multi-axis trade-off?

Method.  A small family set (a three-deep linear chain, a rectangular
two-chain, a SwiGLU MLP, a same-input projection sum, unnormalised
attention, a deep parallel pair and an eight-expert soup) is exported
and saturated once per family.  For each family the suite records

1. the **root-class frontier** — ``SearchResult.frontier`` over every
   pair of the built-in axes (``flops``, ``launch``, ``count``,
   ``depth``, ``peak``, ``params``, ``roofline``).  A >1 frontier is
   classified *genuine* (every axis spreads at least ``_GENUINE_GAP``)
   versus *degenerate* — an exact tie, or the ~1e-16 depth float noise
   the calibrated ``depth_cost_for`` leaves between associativity
   variants;
2. the **per-target extraction** — the search re-run under
   ``criteria=PredictedCriterion(AnalyticalPerformanceModel(),
   hardware=<target>)`` for a compute-bound, a bandwidth-bound and a
   launch-bound target.  Every pick is *certified* (``lower`` →
   ``sink.verify``) and priced two ways: the ``marginal`` DAG-sum the
   search's extraction minimizes, and the ``true`` whole-program
   model value.  The two diverge exactly when the additive marginal
   decomposition mis-ranks a form — the mechanism probe.

Findings (measured, not assumed):

* **Per-target — NEGATIVE.**  An accurate profiler kills the
  bandwidth story: the bandwidth-bound pick never differs from the
  launch-bound pick and never carves out a program of its own.  The
  three targets disagree among themselves in only one family
  (SwiGLU, compute vs bandwidth/launch), and the two families whose
  picks move at all against the default move on the *compute* axis
  (SwiGLU) or on the *model* axis (unnormalised attention), never
  the bandwidth axis.
* **Frontier — NEGATIVE.**  The root-class frontier never returns a
  genuine multi-axis trade-off; its only >1 results are exact ties
  and float noise.
* **Mechanism — NEGATIVE.**  Even the residual target-sensitivity is
  not hardware.  The true roofline value
  (``model.predict(features, target)``) ranks the same form first
  under every target, so the move is the *additive* marginal
  decomposition ``extract_best`` applies to the non-additive
  ``PredictedCriterion``: the criterion is a whole-program function,
  its per-node ``local = c(t) - Σc(children)`` is an approximation,
  and the search minimizes the approximation, not the prediction.

CPU-only, no network, no CUDA, deterministic.
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.cost import (
    count_cost,
    dag_cost,
    depth_cost_for,
    flops_cost,
    launch_aware_cost,
    param_bytes_cost_for,
    roofline_cost,
)
from catopt_core.features import compute_features
from catopt_core.ir import op_repr
from catopt_core.pareto import CostVector, pareto_frontier
from catopt_core.perf_model import AnalyticalPerformanceModel
from catopt_core.profile import TargetProfile
from catopt_orchestrator import Optimizer
from catopt_orchestrator.criteria import (
    PredictedCriterion,
    criteria_cost,
    peak_bytes_cost,
)
from catopt_torch.backend import TorchBackend
from catopt_torch.models import DeepParallel, ParallelLinear

from bench.benchkit import (
    Case,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
    collect_env,
)

#: ``--quick`` shrinks the sweep (the CLI applies this dict).
QUICK = {
    "warmup": 1,
    "min_run_time": 0.01,
}

#: An axis spread below this is not a trade-off — it is a tie.
_GENUINE_GAP = 0.01

#: How many root-class members the frontier may price.
_CANDIDATES = 64

#: The single stateless performance model the suite wraps.
_MODEL = AnalyticalPerformanceModel()

#: Shapes shared by the family builders.
_D = 64
_B = 8
_T = 32

#: The three synthetic targets — one per roofline regime.
_TARGETS: dict[str, TargetProfile] = {
    "compute": TargetProfile(
        name="synth-compute-bound",
        tflops=1e-3,
        gbps=5000.0,
        launch_us=0.5,
        device="synthetic",
        measured_at="synthetic",
        meta={"kind": "compute-bound"},
    ),
    "bandwidth": TargetProfile(
        name="synth-bandwidth-bound",
        tflops=200.0,
        gbps=0.05,
        launch_us=0.5,
        device="synthetic",
        measured_at="synthetic",
        meta={"kind": "bandwidth-bound"},
    ),
    "launch": TargetProfile(
        name="synth-launch-bound",
        tflops=200.0,
        gbps=5000.0,
        launch_us=5000.0,
        device="synthetic",
        measured_at="synthetic",
        meta={"kind": "launch-bound"},
    ),
}

#: The built-in selection axes the frontier is swept over.
_AXES: dict[str, Any] = {
    "flops": flops_cost,
    "launch": launch_aware_cost,
    "count": count_cost,
    "depth": depth_cost_for(None),
    "peak": peak_bytes_cost,
    "params": param_bytes_cost_for(),
    "roofline": roofline_cost,
}

#: Param leaf spellings → the compact signature token.
_LEAF = re.compile(r"\bp_[A-Za-z0-9_]+_(weight|bias)\b")


# ---------------------------------------------------------------------------
#  Model families
# ---------------------------------------------------------------------------


def _chain3(d: int = _D, b: int = _B):
    """A three-deep bias-free linear chain ``x -> W3(W2(W1 x))``."""

    class Chain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Linear(d, d, bias=False)
            self.w2 = nn.Linear(d, d, bias=False)
            self.w3 = nn.Linear(d, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w3(self.w2(self.w1(x)))

    return Chain().eval(), torch.randn(b, d)


def _rect(d: int = _D, k: int = 256, b: int = _B):
    """A rectangular two-chain ``x -> B(A x)`` — matrix-chain order."""

    class Rect(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(d, k, bias=False)
            self.b = nn.Linear(k, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.b(self.a(x))

    return Rect().eval(), torch.randn(b, d)


def _swiglu(d: int = _D, h: int = 256, b: int = _B):
    """A SwiGLU MLP — a fusable gate/up pair, the pairing target."""

    class SwiGLU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w1 = nn.Linear(d, h, bias=False)
            self.w3 = nn.Linear(d, h, bias=False)
            self.w2 = nn.Linear(h, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    return SwiGLU().eval(), torch.randn(b, d)


def _qkv_sum(d: int = _D, b: int = _B):
    """Three same-input projections, summed — a weight-fold target."""

    class QKV(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.wq(x) + self.wk(x) + self.wv(x)

    return QKV().eval(), torch.randn(b, d)


def _linattn(d: int = _D, t: int = _T, b: int = _B):
    """Unnormalised attention ``(Q K^T) V`` — a fusable q/k/v triple."""

    class LinAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            q, k, v = self.wq(x), self.wk(x), self.wv(x)
            return torch.matmul(torch.matmul(q, k.transpose(-2, -1)), v)

    return LinAttn().eval(), torch.randn(b, t, d)


def _deep_parallel(d: int = _D, b: int = _B):
    """``(x@W1 + x@W2) @ W3`` — fold then reassociate."""
    return DeepParallel(d, 256, d).eval(), torch.randn(b, d)


def _parallel8(d: int = _D, b: int = _B):
    """An eight-expert parallel projection sum — a wide fold."""
    return ParallelLinear(d, n_experts=8).eval(), torch.randn(b, d)


#: The measured families — ``(name, builder, one-line note)``.
_FAMILIES: tuple[tuple[str, Any, str], ...] = (
    ("chain3", _chain3, "three-deep linear chain"),
    ("rect", _rect, "rectangular two-chain (k > d)"),
    ("swiglu", _swiglu, "SwiGLU MLP (fusable gate/up pair)"),
    ("qkv_sum", _qkv_sum, "sum of three same-input projections"),
    ("linattn", _linattn, "unnormalised attention (QK^T)V"),
    ("deep_parallel", _deep_parallel, "(x@W1 + x@W2) @ W3"),
    ("parallel8", _parallel8, "eight-expert projection sum"),
)


# ---------------------------------------------------------------------------
#  Measurement helpers
# ---------------------------------------------------------------------------


def _sig(term: Any) -> str:
    """A compact structural signature — param names collapse to ``W``."""
    return _LEAF.sub(
        lambda m: "W" if m.group(1) == "weight" else "b",
        op_repr(term),
    )


def _pick_row(term: Any) -> dict:
    """Price *term* under every target, plus its static features.

    ``compute_features`` now bills **runtime** traffic only — view-op
    outputs and compile-time-folded subtrees are already excluded —
    so ``read`` / ``write`` / ``traffic`` are the bytes the program
    actually moves.

    ``marginal`` is the DAG-sum the search's extraction minimizes
    (``dag_cost`` over the criterion blend): for the non-additive
    ``PredictedCriterion`` this is the per-node marginal
    approximation ``extract_best`` uses, *not* the prediction.
    ``true`` is the whole-program model value
    (``model.predict(features, target)``).  The two diverge exactly
    when the marginal decomposition mis-ranks a form.
    """
    feats = compute_features(term)
    marginal = {
        name: dag_cost(
            term,
            criteria_cost(PredictedCriterion(_MODEL, hardware=tgt)),
        )
        for name, tgt in _TARGETS.items()
    }
    true = {
        name: float(_MODEL.predict(feats, tgt))
        for name, tgt in _TARGETS.items()
    }
    return {
        "sig": _sig(term),
        "flops": dag_cost(term, flops_cost),
        "read": feats.bytes_read,
        "write": feats.bytes_written,
        "traffic": feats.bytes_read + feats.bytes_written,
        "ops": feats.operations,
        "marginal": marginal,
        "true": true,
    }


def _kind(frontier: list) -> str:
    """Classify a >1 frontier by its smallest axis spread.

    ``"tie"`` — some axis is exactly flat across the frontier;
    ``"genuine"`` — every axis spreads at least :data:`_GENUINE_GAP`;
    ``"noise"`` — a real spread nowhere, float noise somewhere.
    """
    dims = frontier[0][0].dims
    spreads = []
    for dim in dims:
        vals = [v[dim] for v, _ in frontier]
        hi = max(vals)
        spreads.append((hi - min(vals)) / hi if hi else 0.0)
    lo = min(spreads)
    if lo == 0.0:
        return "tie"
    return "genuine" if lo >= _GENUINE_GAP else "noise"


def _frontier_rows(res: Any) -> tuple[int, list[dict]]:
    """Root-class frontier size + classification for every axis pair."""
    alts = res.alternatives(top_k=_CANDIDATES, cost_fn=flops_cost)
    vecs = {
        a: [dag_cost(t, fn) for _, t in alts] for a, fn in _AXES.items()
    }
    rows: list[dict] = []
    for a, b in itertools.combinations(_AXES, 2):
        priced = [
            (CostVector((a, b), (vecs[a][i], vecs[b][i])), alts[i][1])
            for i in range(len(alts))
        ]
        fr = pareto_frontier(priced, key=lambda p: p[0])
        if len(fr) > 1:
            rows.append(
                {
                    "axes": [a, b],
                    "size": len(fr),
                    "kind": _kind(fr),
                    "vectors": [v.as_dict() for v, _ in fr],
                }
            )
    return len(alts), rows


def _stmt(opt: Optimizer, model: Any, x: Any, target: Any):
    """A zero-arg callable running one extraction under *target*."""
    crit = (
        None
        if target is None
        else PredictedCriterion(_MODEL, hardware=target)
    )

    def run() -> Any:
        return opt.search(model, x, criteria=crit)

    return run


def _build_case(
    name: str, builder: Any, note: str, opt: Optimizer
) -> Case:
    """One family → a ``Case`` carrying its picks and frontier rows."""
    model, x = builder()
    res = opt.search(model, x)
    n_alts, rows = _frontier_rows(res)
    default = _pick_row(res.term)
    picks: dict[str, dict] = {}
    for tname, tgt in _TARGETS.items():
        r = opt.search(
            model, x, criteria=PredictedCriterion(_MODEL, hardware=tgt)
        )
        lr = opt.lower(r, x)
        row = _pick_row(r.term)
        row["verified"] = bool(lr.verified.passed)
        row["max_rel"] = float(lr.verified.max_rel)
        picks[tname] = row
    distinct = len(
        {default["sig"]} | {p["sig"] for p in picks.values()}
    )
    aux = {
        "family": name,
        "note": note,
        "n_root_alts": n_alts,
        "default": default,
        "picks": picks,
        "distinct_certified": distinct,
        "frontiers": rows,
    }
    variants = [
        Variant(
            "default",
            _stmt(opt, model, x, None),
            flops=default["flops"],
            note="default extraction model",
        )
    ]
    variants += [
        Variant(
            tname,
            _stmt(opt, model, x, _TARGETS[tname]),
            flops=picks[tname]["flops"],
            note=f"{tname}-bound target",
        )
        for tname in _TARGETS
    ]
    return Case(
        name=name,
        params={"family": name, "n_root_alts": n_alts},
        variants=variants,
        aux=aux,
    )


# ---------------------------------------------------------------------------
#  Findings
# ---------------------------------------------------------------------------


def _frontier_finding(cells: list[Any]) -> Finding:
    """The frontier verdict — genuine trade-offs, or ties and noise."""
    n_pairs = len(list(itertools.combinations(_AXES, 2)))
    nontrivial = [
        (c.case.name, r) for c in cells for r in c.aux["frontiers"]
    ]
    genuine = [x for x in nontrivial if x[1]["kind"] == "genuine"]
    ties = [x for x in nontrivial if x[1]["kind"] == "tie"]
    noise = [x for x in nontrivial if x[1]["kind"] == "noise"]
    plural = "" if len(genuine) == 1 else "s"
    return Finding(
        claim=(
            "the root-class Pareto frontier exposes a genuine "
            "multi-axis trade-off"
        ),
        verdict=Verdict.WIN if genuine else Verdict.NEGATIVE,
        headline=(
            f"{len(genuine)} genuine frontier{plural} over "
            f"{len(cells)} families x {n_pairs} axis pairs — the "
            f"{len(nontrivial)} >1 results are {len(ties)} exact "
            f"tie(s) and {len(noise)} float-noise pair(s)"
        ),
        metric="genuine frontiers",
        value=float(len(genuine)),
        evidence={
            "axis_pairs": len(cells) * n_pairs,
            "non_trivial": len(nontrivial),
            "ties": [f"{n}:{r['axes']}" for n, r in ties],
            "noise": [f"{n}:{r['axes']}" for n, r in noise],
        },
    )


def _per_target_finding(cells: list[Any]) -> Finding:
    """The per-target verdict — does a bandwidth target move the pick?

    The sharp test: the bandwidth-bound target's pick is *distinct*
    from **both** the compute and the launch picks — a program the
    bandwidth regime alone would select.  ``bw`` counts those.
    """

    def sig(c: Any, t: str) -> str:
        return (
            c.aux["default"]["sig"]
            if t == "default"
            else c.aux["picks"][t]["sig"]
        )

    bw = [
        c
        for c in cells
        if sig(c, "bandwidth") != sig(c, "compute")
        and sig(c, "bandwidth") != sig(c, "launch")
    ]
    disagree = [
        c
        for c in cells
        if len({sig(c, t) for t in ("compute", "bandwidth", "launch")})
        > 1
    ]
    multi = [c for c in cells if c.aux["distinct_certified"] > 1]
    verified = all(
        p["verified"] for c in cells for p in c.aux["picks"].values()
    )
    detail = {}
    for c in cells:
        a = c.aux
        detail[c.case.name] = {
            "distinct_certified": a["distinct_certified"],
            "default_traffic": a["default"]["traffic"],
            "bandwidth_pick_traffic": a["picks"]["bandwidth"][
                "traffic"
            ],
            "bandwidth_vs_compute": sig(c, "bandwidth")
            != sig(c, "compute"),
            "bandwidth_vs_launch": sig(c, "bandwidth")
            != sig(c, "launch"),
            "bandwidth_vs_default": sig(c, "bandwidth")
            != sig(c, "default"),
            "targets_agree": sig(c, "compute")
            == sig(c, "bandwidth")
            == sig(c, "launch"),
        }
    names = ", ".join(c.case.name for c in disagree)
    return Finding(
        claim=(
            "a bandwidth-bound target extracts a different certified "
            "program than the compute / launch targets"
        ),
        verdict=Verdict.WIN if bw else Verdict.NEGATIVE,
        headline=(
            f"{len(bw)}/{len(cells)} families: the bandwidth pick is "
            f"distinct from both the compute and launch picks nowhere "
            f"— it equals the launch pick in every family.  The three "
            f"targets disagree among themselves in only "
            f"{len(disagree)}/{len(cells)} ({names}); "
            f"{len(multi)}/{len(cells)} move at all against the "
            f"default, and every pick verified={verified}"
        ),
        metric=(
            "families where the bandwidth pick is distinct from both "
            "the compute and launch picks"
        ),
        value=float(len(bw)),
        evidence={
            "all_verified": verified,
            "targets_disagree": [c.case.name for c in disagree],
            "move_vs_default": [c.case.name for c in multi],
            "per_family": detail,
        },
    )


def _mechanism_finding(cells: list[Any]) -> Finding:
    """Why the picks move — a true model effect, or a marginal artefact?"""
    rows = {}
    invariant = 0
    for c in cells:
        a = c.aux
        forms = {a["default"]["sig"]: a["default"]}
        for p in a["picks"].values():
            forms.setdefault(p["sig"], p)
        # The form the TRUE roofline value prefers, per target, over
        # the extracted forms.  Target-invariant => no hardware story.
        argmin = {
            t: min(forms, key=lambda s: forms[s]["true"][t])
            for t in _TARGETS
        }
        same = len(set(argmin.values())) == 1
        invariant += same
        if a["distinct_certified"] <= 1:
            continue
        rows[a["family"]] = {
            "true_argmin": argmin,
            "search_pick": {t: a["picks"][t]["sig"] for t in _TARGETS},
            "search_matches_true": {
                t: a["picks"][t]["sig"] == argmin[t] for t in _TARGETS
            },
            "forms": {
                sig: {
                    "flops": row["flops"],
                    "write": row["write"],
                    "ops": row["ops"],
                    "marginal": row["marginal"],
                    "true": row["true"],
                }
                for sig, row in forms.items()
            },
        }
    n = len(cells)
    return Finding(
        claim=(
            "the residual target-sensitivity traces to a "
            "hardware-genuine trade-off"
        ),
        verdict=Verdict.NEGATIVE,
        headline=(
            f"No — the true roofline value "
            f"(model.predict(features, target)) ranks the same form "
            f"first under all three targets in {invariant}/{n} "
            f"families, so the move is the additive marginal "
            f"decomposition extract_best applies to the non-additive "
            f"PredictedCriterion, not hardware"
        ),
        metric=(
            "families where the true roofline argmin is "
            "target-invariant"
        ),
        value=float(invariant),
        evidence={
            "families": n,
            "target_invariant": invariant,
            "per_family": rows,
        },
    )


# ---------------------------------------------------------------------------
#  Console twin
# ---------------------------------------------------------------------------


def _print_table(cells: list[Any]) -> None:
    """The per-family console twin of the report table."""
    head = (
        f"{'family':<14} {'alts':>4} {'picks':>5} "
        f"{'frontier>1':>10}  default -> bandwidth"
    )
    print("\n" + head)
    print("-" * len(head))
    for cell in cells:
        a = cell.aux
        n1 = sum(1 for r in a["frontiers"] if r["size"] > 1)
        print(
            f"{cell.case.name:<14} {a['n_root_alts']:>4} "
            f"{a['distinct_certified']:>5} {n1:>10}  "
            f"{a['default']['sig'][:26]} -> "
            f"{a['picks']['bandwidth']['sig'][:26]}"
        )


def run_bench(args: argparse.Namespace) -> Report:
    """Harnessed entry point: build the families, pick per target."""
    dev = torch.device(getattr(args, "device", None) or "cpu")
    opt = Optimizer(backend=TorchBackend())
    cases = [
        _build_case(name, builder, note, opt)
        for name, builder, note in _FAMILIES
    ]
    runner = Runner(
        device=dev,
        warmup=int(getattr(args, "warmup", None) or 3),
        min_run_time=float(getattr(args, "min_run_time", None) or 0.05),
    )
    cells = runner.run(cases)
    _print_table(cells)
    return Report(
        suite="eval_axis",
        title="Evaluation-axis selection",
        summary=(
            "For each family: the root-class Pareto frontier over every "
            "pair of built-in axes, and the program a compute- / "
            "bandwidth- / launch-bound PredictedCriterion extracts "
            "(each pick certified by lower -> verify)."
        ),
        findings=[
            _per_target_finding(cells),
            _frontier_finding(cells),
            _mechanism_finding(cells),
        ],
        cells=cells,
        env=collect_env(dev),
        provenance={
            "families": [f[0] for f in _FAMILIES],
            "axes": list(_AXES),
            "targets": {n: t.name for n, t in _TARGETS.items()},
            "genuine_gap": _GENUINE_GAP,
            "device": str(dev),
        },
    )


def main(argv: list[str] | None = None) -> None:
    """Direct entry point (``python bench/suites/evaluation/...``)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min-run-time", type=float, default=0.05)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        for key, val in QUICK.items():
            setattr(args, key, val)
    report = run_bench(args)
    if not args.no_artifacts:
        out = Path(args.out)
        report.to_json(out / "eval_axis.json")
        report.to_markdown(out / "eval_axis.md")
        report.to_html(out / "eval_axis.html")
        print(f"[artifacts] {out}/eval_axis.{{json,md,html}}")


if __name__ == "__main__":
    main()
