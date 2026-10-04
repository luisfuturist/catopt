"""Measure whether the 11 proposed laws actually pay on real models.

``tools/law_proposal.py`` found 11 genuinely-new, cost-reducing laws
absent from ``ALL_RULES`` — mul-over-add factoring, neg-distribution,
``square(neg x) = square x``, the ``exp`` homomorphism, the
annihilators, ``pow x 1``, ``x - x``, ``x + (-x)`` and ``x / x``.
Every one is numerically true and lowers the extracted cost on the
*synthetic* term it was designed for.  The proposal retro left the
applicability question open: do these laws fire on real graphs, and
when they do, do they pay?

This tool answers that question in three measurements, all CPU-only
and bounded to a few minutes:

1. **Firing** — run each law alone over every bench law case
   (``bench.suites.correctness.law_bench.LAW_CASES``) and over real
   model graphs (``catopt_torch.models``), plus a synthetic control
   (each law's own LHS instance).  Report firing counts, including
   zeros.
2. **Cost delta** — for every firing, the pipeline cost model's
   before/after cost and whether extraction changed the program; when
   it did, lower both through ``_lower_extracted`` and run
   ``sink.verify``.
3. **Reach** — saturate a real graph with ``ALL_RULES`` versus
   ``ALL_RULES + {new laws}``: e-node counts, extracted cost, and
   whether the certificate still replays.

The laws are defined *inside this tool* — they are NOT added to
``packages/`` or the shipped ``ALL_RULES``.  The tool is the honest
test of whether law-finding is worth anything: a firing table of
zeros is a decisive answer, not a bug.

Run::

    .venv/bin/python tools/law_impact.py
    .venv/bin/python tools/law_impact.py --json /tmp/law_impact.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from catopt_core.cost import backend_cost, dag_cost, executor_cost_for
from catopt_core.egraph import (
    EGraph,
    Rewrite,
    verify_certificate,
)
from catopt_core.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
)
from catopt_core.laws import ALL_RULES
from catopt_core.laws import tags as _tags
from catopt_core.laws.base import R
from catopt_orchestrator.optimize import _lower_extracted
from catopt_torch.adapters import TorchSink, TorchSource

# ``bench`` is a repo-root package; running this file puts ``tools/``
# on ``sys.path``, not the repo root, so add the root explicitly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__all__ = [
    "Firing",
    "TermCase",
    "main",
    "model_cases",
    "new_laws",
    "reach_row",
    "run_impact",
    "synthetic_cases",
]

_DEV = torch.device("cpu")

#: Saturation budget for a firing / cost-delta probe.
_FIRING_ITERS = 4
_FIRING_NODES = 40_000

#: Saturation budget for the reach comparison (mirrors the pipeline's
#: bounded saturation: the EXPANSIVE closure generators get a per-rule
#: enode budget so the Catalan blow-up cannot run away).
_REACH_ITERS = 6
_REACH_NODES = 60_000
_EXPANSIVE_BUDGET = 600

#: Verification tolerance for the lowered before/after modules.
_RTOL = 1e-4

#: Scale knob for the bench-case builders.
_BENCH_SIZE = 32

#: Shapes for the synthetic control's Vars.
_D = 4


# ---------------------------------------------------------------------------
#  The 11 proposed laws — encoded here, never in ``packages/``
# ---------------------------------------------------------------------------


def new_laws() -> list[Rewrite]:
    """Return the 11 candidate laws as fireable ``Rewrite`` values.

    Names are prefixed ``cand_`` so a run can never shadow a shipped
    rule.  All are unconditional (the pattern already restricts the
    shape); the conditional caveats the proposal retro records
    (``x/x`` needs ``x != 0``, the annihilators carry the usual
    ``inf``/``nan`` fp caveat) are noted in the companion retro, not
    encoded as ``check`` hooks — the point is to measure whether the
    pattern matches at all.
    """
    x, y, z = "x", "y", "z"
    sim = (_tags.SIMPLIFICATION,)
    return [
        R(
            "cand_mul_factor",
            Op.make("add", Op.make("mul", x, y), Op.make("mul", x, z)),
            Op.make("mul", x, Op.make("add", y, z)),
            law="x*y + x*z = x*(y+z)  (factoring).",
            tags=sim,
        ),
        R(
            "cand_mul_factor_right",
            Op.make("add", Op.make("mul", y, x), Op.make("mul", z, x)),
            Op.make("mul", Op.make("add", y, z), x),
            law="y*x + z*x = (y+z)*x  (right-slot factoring).",
            tags=sim,
        ),
        R(
            "cand_neg_factor",
            Op.make("add", Op.make("neg", x), Op.make("neg", y)),
            Op.make("neg", Op.make("add", x, y)),
            law="-x + -y = -(x+y).",
            tags=sim,
        ),
        R(
            "cand_square_neg",
            Op.make("square", Op.make("neg", x)),
            Op.make("square", x),
            law="(-x)^2 = x^2.",
            tags=sim,
        ),
        R(
            "cand_exp_factor",
            Op.make("mul", Op.make("exp", x), Op.make("exp", y)),
            Op.make("exp", Op.make("add", x, y)),
            law="e^x * e^y = e^(x+y).",
            tags=sim,
        ),
        R(
            "cand_mul_zero",
            Op.make("mul", x, Const(0)),
            Const(0),
            law="x * 0 = 0.",
            tags=sim,
        ),
        R(
            "cand_mul_zero_left",
            Op.make("mul", Const(0), x),
            Const(0),
            law="0 * x = 0.",
            tags=sim,
        ),
        R(
            "cand_pow_one",
            Op.make("pow", x, Const(1)),
            x,
            law="x^1 = x.",
            tags=sim,
        ),
        R(
            "cand_sub_self",
            Op.make("sub", x, x),
            Const(0),
            law="x - x = 0.",
            tags=sim,
        ),
        R(
            "cand_add_inv",
            Op.make("add", x, Op.make("neg", x)),
            Const(0),
            law="x + (-x) = 0.",
            tags=sim,
        ),
        R(
            "cand_div_self",
            Op.make("div", x, x),
            Const(1),
            law="x / x = 1.",
            tags=sim,
        ),
    ]


# ---------------------------------------------------------------------------
#  Term sources
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TermCase:
    """One program to probe, with the leaves to lower and verify it."""

    source: str
    name: str
    term: Any
    inputs: tuple[Var, ...]
    feed: tuple
    param_vals: dict


def _params_of(term: Any) -> dict[str, Param]:
    """Collect the distinct ``Param`` leaves of *term* by name."""
    seen: set[Any] = set()
    out: dict[str, Param] = {}
    stack = [term]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Param):
            out[t.name] = t
        elif isinstance(t, Op):
            stack.extend(t.args)
    return out


def _ir_of(term: Any, inputs: tuple[Var, ...]) -> IR:
    """Wrap *term* as an ``IR`` with the given inputs."""
    return IR(
        root=term,
        inputs=list(inputs),
        input_names={v.name for v in inputs},
        params=_params_of(term),
    )


def _bench_cases() -> tuple[list[TermCase], list[str]]:
    """Build every registered bench law case at ``_BENCH_SIZE``.

    The builders are keyed by *shipped* rule name; they are reused
    here purely as a corpus of small, real-ish, well-typed programs to
    probe the new laws against.  A builder that raises is reported in
    the second list rather than crashing the corpus.
    """
    from bench.suites.correctness.law_bench import LAW_CASES

    out: list[TermCase] = []
    errors: list[str] = []
    for name, build in sorted(LAW_CASES.items()):
        try:
            torch.manual_seed(1000)
            term, env, inputs = build(_BENCH_SIZE, _DEV)
        except Exception as e:  # honest per-case failure
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        inps = tuple(inputs)
        out.append(
            TermCase(
                source="bench",
                name=name,
                term=term,
                inputs=inps,
                feed=tuple(env[v.name] for v in inps),
                param_vals={
                    n: env[n] for n in _params_of(term) if n in env
                },
            )
        )
    return out, errors


def _model_cases() -> list[tuple[str, Any, Any]]:
    """Return ``(name, model, example_input)`` for the real graphs."""
    from catopt_torch import models as M
    from catopt_torch.models import hybrid as H
    from catopt_torch.models import ssm as S

    d = 16
    t = torch.randn(4, d, dtype=torch.float64)
    seq = torch.randn(2, 8, d, dtype=torch.float64)
    st = torch.randn(4, 8, dtype=torch.float64)
    img = torch.randn(1, 8, 4, 4, dtype=torch.float64)
    wav = torch.randn(1, 8, 16, dtype=torch.float64)
    return [
        ("SwiGLU", M.SwiGLU(d, 2).eval().double(), t),
        ("RMSNorm", M.RMSNorm(d).eval().double(), t),
        ("ResidualMLP", M.ResidualMLP(d, 2).eval().double(), t),
        ("ParallelLinear", M.ParallelLinear(d, 2).eval().double(), t),
        ("DeepParallel", M.DeepParallel(d, 8, 4).eval().double(), t),
        ("NormLinear", M.NormLinear(d).eval().double(), t),
        ("MatrixChain", M.MatrixChain(d, 12, 8, 4).eval().double(), t),
        (
            "AttentionBlock",
            M.AttentionBlock(d, 4).eval().double(),
            seq,
        ),
        (
            "GQAAttention",
            M.GQAAttention(d, 4, 2).eval().double(),
            seq,
        ),
        (
            "TransformerBlock",
            M.TransformerBlock(d, 4, 2).eval().double(),
            seq,
        ),
        (
            "ParallelBlock",
            M.ParallelBlock(d, 4, 2).eval().double(),
            seq,
        ),
        (
            "RepeatKVAttention",
            M.RepeatKVAttention(d, 4, 2).eval().double(),
            seq,
        ),
        (
            "EagerAttention",
            M.EagerAttention(d, 4, 8).eval().double(),
            seq,
        ),
        (
            "AdditiveMaskAttention",
            M.AdditiveMaskAttention(d, 4, 8).eval().double(),
            seq,
        ),
        (
            "ParallelConv",
            M.ParallelConv(8, 8, 2, 1).eval().double(),
            img,
        ),
        (
            "LinearRecurrence",
            M.LinearRecurrence(8, 4).eval().double(),
            st,
        ),
        (
            "LinearAttention",
            M.LinearAttention().eval().double(),
            (
                torch.randn(2, 4, 8, dtype=torch.float64),
                torch.randn(2, 4, 8, dtype=torch.float64),
                torch.randn(2, 4, 8, dtype=torch.float64),
            ),
        ),
        ("MoEMLP", M.MoEMLP(d, 32, 3).eval().double(), t),
        ("GegluMLP", M.GegluMLP(d, 2).eval().double(), t),
        (
            "GatedResidualBlock",
            M.GatedResidualBlock(d).eval().double(),
            t,
        ),
        (
            "ResNetBlock",
            M.ResNetBlock(8).eval().double(),
            img,
        ),
        (
            "DepthwiseConvBlock",
            M.DepthwiseConvBlock(8).eval().double(),
            img,
        ),
        (
            "ConvNeXtBlock",
            M.ConvNeXtBlock(8).eval().double(),
            img,
        ),
        (
            "ManualSoftmaxAttention",
            M.ManualSoftmaxAttention(d).eval().double(),
            seq,
        ),
        (
            "PositionalEmbedding",
            M.PositionalEmbedding(64, d).eval().double(),
            seq,
        ),
        (
            "KernelizedAttention",
            M.KernelizedAttention(d).eval().double(),
            seq,
        ),
        (
            "TopKRouter",
            M.TopKRouter(d, 4, 2).eval().double(),
            t,
        ),
        (
            "Wav2VecBlock",
            M.Wav2VecBlock(8).eval().double(),
            wav,
        ),
        (
            "SinusoidalEncoding",
            M.SinusoidalEncoding(d).eval().double(),
            seq,
        ),
        (
            "TrilCausalAttention",
            M.TrilCausalAttention(d).eval().double(),
            seq,
        ),
        ("HardDispatch", M.HardDispatch(d, 4).eval().double(), t),
        (
            "CodebookQuantizer",
            M.CodebookQuantizer(d, 8).eval().double(),
            t,
        ),
        ("MaxoutMLP", M.MaxoutMLP(d, 2).eval().double(), t),
        ("GluMLP", M.GluMLP(d, 2).eval().double(), t),
        (
            "ManualGluMLP",
            M.ManualGluMLP(d, 2).eval().double(),
            t,
        ),
        ("NativeRmsNorm", M.NativeRmsNorm(d).eval().double(), t),
        ("SelectiveSSM", S.SelectiveSSM(8, 8, 4).eval().double(), st),
        ("DiagDenseSSM", S.DiagDenseSSM(8, 8, 4).eval().double(), st),
        ("DiagonalSSM", S.DiagonalSSM(8, 8, 4).eval().double(), st),
        (
            "HybridBlock",
            H.HybridBlock(8, 8, 8, 4, 2).eval().double(),
            st,
        ),
        (
            "TwoLayerHybrid",
            H.TwoLayerHybrid(8, 8, 8, 4, 2).eval().double(),
            st,
        ),
    ]


def model_cases() -> tuple[list[TermCase], list[str]]:
    """Export every real model to IR; return ``(cases, export_errors)``.

    A model that fails to export is reported honestly in the second
    list rather than crashing the run.
    """
    src = TorchSource()
    cases: list[TermCase] = []
    errors: list[str] = []
    for name, model, x in _model_cases():
        try:
            ir, tensors = src.to_ir(model, x)
        except Exception as e:  # honest export failure
            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue
        feed = x if isinstance(x, tuple) else (x,)
        cases.append(
            TermCase(
                source="model",
                name=name,
                term=ir.root,
                inputs=tuple(ir.inputs),
                feed=tuple(feed),
                param_vals=dict(tensors),
            )
        )
    return cases, errors


def _v(name: str, *shape: int) -> Var:
    """Return a named tensor variable."""
    return Var(name, TensorType(tuple(shape)))


def synthetic_cases() -> list[TermCase]:
    """Return one designed witness term per law (the control).

    Each is the law's own LHS instantiated on small, nonzero fp64
    tensors — the term the law *should* fire on.  A zero here would
    mean the harness is broken, not that the law is useless.
    """
    d = _D
    x, y, z = _v("x", d, d), _v("y", d, d), _v("z", d, d)
    xyz = (x, y, z)

    def feed(*vars_: Var) -> tuple:
        return tuple(
            torch.randn(tuple(v.typ.shape), dtype=torch.float64) + 1.0
            for v in vars_
        )

    specs: list[tuple[str, Any, tuple]] = [
        (
            "cand_mul_factor",
            Op.make("add", Op.make("mul", x, y), Op.make("mul", x, z)),
            xyz,
        ),
        (
            "cand_mul_factor_right",
            Op.make("add", Op.make("mul", y, x), Op.make("mul", z, x)),
            xyz,
        ),
        (
            "cand_neg_factor",
            Op.make("add", Op.make("neg", x), Op.make("neg", y)),
            (x, y),
        ),
        (
            "cand_square_neg",
            Op.make("square", Op.make("neg", x)),
            (x,),
        ),
        (
            "cand_exp_factor",
            Op.make("mul", Op.make("exp", x), Op.make("exp", y)),
            (x, y),
        ),
        (
            "cand_mul_zero",
            Op.make("mul", x, Const(0)),
            (x,),
        ),
        (
            "cand_mul_zero_left",
            Op.make("mul", Const(0), x),
            (x,),
        ),
        (
            "cand_pow_one",
            Op.make("pow", x, Const(1)),
            (x,),
        ),
        (
            "cand_sub_self",
            Op.make("sub", x, x),
            (x,),
        ),
        (
            "cand_add_inv",
            Op.make("add", x, Op.make("neg", x)),
            (x,),
        ),
        (
            "cand_div_self",
            Op.make("div", x, x),
            (x,),
        ),
    ]
    return [
        TermCase(
            source="synthetic",
            name=name,
            term=term,
            inputs=tuple(vars_),
            feed=feed(*vars_),
            param_vals={},
        )
        for name, term, vars_ in specs
    ]


# ---------------------------------------------------------------------------
#  Step (a) + (b): firing and cost delta
# ---------------------------------------------------------------------------


@dataclass
class Firing:
    """One (case, law) probe and its measured outcome."""

    source: str
    case: str
    law: str
    fires: int = 0
    base_cost: float = 0.0
    out_cost: float = 0.0
    changed: bool = False
    verified: str = "-"
    note: str = ""

    @property
    def paid(self) -> bool:
        """True iff the law fired and strictly lowered the cost."""
        return self.fires > 0 and self.out_cost < self.base_cost


def _cost_fn(sink: TorchSink) -> Any:
    """Return the pipeline's selection model (roofline + dispatch)."""
    return backend_cost(
        executor_cost_for(lowering="generic"), sink.supported_ops
    )


def _probe(
    case: TermCase,
    law: Rewrite,
    sink: TorchSink,
    cost_fn: Any,
) -> Firing:
    """Run *law* alone on *case*, then price and verify any change."""
    out = Firing(source=case.source, case=case.name, law=law.name)
    eg = EGraph()
    root = eg.add_term(case.term)
    eg.run(
        [law],
        root,
        max_iterations=_FIRING_ITERS,
        max_nodes=_FIRING_NODES,
    )
    out.fires = eg.rule_fires.get(law.name, 0)
    if not out.fires:
        return out
    out.base_cost = dag_cost(case.term, cost_fn)
    best = eg.extract_best(root, cost_fn)
    if best is None:
        out.note = "extraction returned no member"
        return out
    out.out_cost = dag_cost(best, cost_fn)
    out.changed = best != case.term
    if not out.changed:
        out.verified = "same"
        out.note = "member found; extraction kept the input term"
        return out
    try:
        before = _lower_extracted(
            case.term,
            _ir_of(case.term, case.inputs),
            case.param_vals,
            sink,
        )
        after = _lower_extracted(
            best, _ir_of(best, case.inputs), case.param_vals, sink
        )
        vr = sink.verify(before, after, case.feed, rtol=_RTOL)
    except Exception as e:  # honest failure row, not a crash
        out.verified = "error"
        out.note = f"{type(e).__name__}: {e}"
        return out
    out.verified = "pass" if vr.passed else "FAIL"
    if not vr.passed:
        out.note = f"lowered terms differ (max_rel={vr.max_rel:.2e})"
    return out


def _set_fires(case: TermCase, laws: list[Rewrite]) -> dict[str, int]:
    """Run the whole *laws* set on *case*; return its ``rule_fires``."""
    eg = EGraph()
    root = eg.add_term(case.term)
    eg.run(
        laws,
        root,
        max_iterations=_FIRING_ITERS,
        max_nodes=_FIRING_NODES,
    )
    return {
        r.name: eg.rule_fires.get(r.name, 0)
        for r in laws
        if eg.rule_fires.get(r.name, 0)
    }


# ---------------------------------------------------------------------------
#  Relaxed census — is the op present, or is the whole shape absent?
# ---------------------------------------------------------------------------


def _iter_subterms(term: Any) -> list[Any]:
    """Return every distinct subterm of *term* (deduped by identity)."""
    seen: set[int] = set()
    out: list[Any] = []
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append(t)
        if isinstance(t, Op):
            stack.extend(t.args)
    return out


def _relax(lhs: Any) -> Any:
    """Rename each repeated metavariable to a fresh name (wildcards).

    The matcher enforces that two occurrences of one metavariable bind
    the *same* e-class; relaxing that turns the LHS into a pure
    structural pattern.  A subterm matching the relaxed pattern says
    "the op shape is present"; the law then fails only because the
    equality precondition (e.g. both factors identical) does not hold.
    A zero relaxed count says the shape never appears at all.
    """
    counts: dict[str, int] = {}

    def walk(t: Any) -> Any:
        if isinstance(t, Op):
            return Op.make(
                t.op,
                *(walk(a) for a in t.args),
                **dict(t.attrs),
            )
        if isinstance(t, str):
            counts[t] = counts.get(t, 0) + 1
            return t if counts[t] == 1 else f"{t}__{counts[t]}"
        return t

    return walk(lhs)


def _relaxed_census(
    terms: list[Any], laws: list[Rewrite]
) -> dict[str, int]:
    """Count subterms matching each law's LHS with metavars relaxed."""
    from catopt_core.egraph.terms import _term_match

    subterms = [s for t in terms for s in _iter_subterms(t)]
    out: dict[str, int] = {}
    for law in laws:
        pat = _relax(law.lhs)
        out[law.name] = sum(
            1
            for s in subterms
            if isinstance(s, Op) and _term_match(pat, s) is not None
        )
    return out


# ---------------------------------------------------------------------------
#  Step (c): reach — does adding the laws change the fixed point?
# ---------------------------------------------------------------------------


def _rule_budgets(rules: list[Rewrite]) -> dict[str, int]:
    """Per-rule enode budget for the EXPANSIVE closure generators."""
    names = {r.name for r in rules}
    return {
        r.name: _EXPANSIVE_BUDGET
        for r in ALL_RULES
        if r.name in names and _tags.EXPANSIVE in r.tags
    }


def _saturate(
    term: Any, rules: list[Rewrite], cost_fn: Any
) -> tuple[EGraph, int, Any, dict]:
    """Saturate *term* under *rules*; return the graph and extraction."""
    eg = EGraph()
    root = eg.add_term(term)
    stats = eg.run(
        rules,
        root,
        max_iterations=_REACH_ITERS,
        max_nodes=_REACH_NODES,
        rule_budgets=_rule_budgets(rules),
    )
    best = eg.extract_best(root, cost_fn)
    return eg, root, best, stats


def _cert_ok(eg: EGraph, src: Any, best: Any, cost_fn: Any) -> str:
    """Return ``"pass"`` / ``"FAIL"`` / an error tag for the certificate."""
    try:
        cert = eg.certificate(src, best, cost_fn=cost_fn)
        verify_certificate(src, cert)
    except Exception as e:  # honest failure, not a crash
        return f"FAIL ({type(e).__name__})"
    return "pass"


def reach_row(
    case: TermCase, laws: list[Rewrite], sink: TorchSink, cost_fn: Any
) -> dict:
    """Compare saturation with vs without *laws* on one real graph."""
    base_eg, _base_root, base_best, base_stats = _saturate(
        case.term, list(ALL_RULES), cost_fn
    )
    add_eg, _add_root, add_best, add_stats = _saturate(
        case.term, [*ALL_RULES, *laws], cost_fn
    )
    fires = {
        r.name: add_eg.rule_fires.get(r.name, 0)
        for r in laws
        if add_eg.rule_fires.get(r.name, 0)
    }
    base_cost = (
        dag_cost(base_best, cost_fn) if base_best else float("inf")
    )
    add_cost = dag_cost(add_best, cost_fn) if add_best else float("inf")
    return {
        "model": case.name,
        "base_enodes": base_stats["n_enodes"],
        "add_enodes": add_stats["n_enodes"],
        "base_classes": base_stats["n_classes"],
        "add_classes": add_stats["n_classes"],
        "base_stop": base_stats["stop"],
        "add_stop": add_stats["stop"],
        "base_cost": base_cost,
        "add_cost": add_cost,
        "changed": add_best != base_best,
        "new_fires": fires,
        "base_cert": _cert_ok(base_eg, case.term, base_best, cost_fn),
        "add_cert": _cert_ok(add_eg, case.term, add_best, cost_fn),
    }


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _fire_table(firings: list[Firing], laws: list[Rewrite]) -> str:
    """Render the per-law firing table (bench / model / synthetic)."""
    head = (
        f"{'law':<24} {'bench':>6} {'model':>6} {'synth':>6} "
        f"{'total':>6} {'paid':>5}"
    )
    lines = [head, "-" * len(head)]
    for law in laws:
        rows = [f for f in firings if f.law == law.name]
        bench = sum(1 for f in rows if f.source == "bench" and f.fires)
        model = sum(1 for f in rows if f.source == "model" and f.fires)
        synth = sum(
            1 for f in rows if f.source == "synthetic" and f.fires
        )
        total = bench + model + synth
        paid = sum(1 for f in rows if f.paid)
        lines.append(
            f"{law.name:<24} {bench:>6} {model:>6} {synth:>6} "
            f"{total:>6} {paid:>5}"
        )
    return "\n".join(lines)


def _delta_table(firings: list[Firing]) -> str:
    """Render every firing that changed the program, with its verify."""
    fired = [f for f in firings if f.fires]
    if not fired:
        return "  (no law fired anywhere)"
    head = (
        f"{'source':<9} {'case':<20} {'law':<22} {'fires':>5} "
        f"{'cost in':>12} {'cost out':>12} {'chg':>4} {'verify':>7}"
    )
    lines = [head, "-" * len(head)]
    for f in sorted(fired, key=lambda f: (f.source, f.case, f.law)):
        lines.append(
            f"{f.source:<9} {f.case:<20} {f.law:<22} {f.fires:>5} "
            f"{f.base_cost:>12.4g} {f.out_cost:>12.4g} "
            f"{'yes' if f.changed else 'no':>4} {f.verified:>7}"
        )
        if f.note:
            lines.append(f"{'':>9} ↳ {f.note}")
    return "\n".join(lines)


def _census_table(result: dict) -> str:
    """Render the relaxed-pattern census (shape present vs firing)."""
    census: dict[str, int] = result["census"]
    fired = {
        f.law
        for f in result["firings"]
        if f.source in {"bench", "model"} and f.fires
    }
    head = f"{'law':<24} {'relaxed':>8} {'fired':>6}  interpretation"
    lines = [head, "-" * len(head)]
    for name in sorted(census):
        n = census[name]
        if n == 0:
            note = "shape absent in the corpus"
        elif name in fired:
            note = "shape present AND fired"
        else:
            note = "shape present, equality precondition fails"
        lines.append(f"{name:<24} {n:>8} {name in fired:>6}  {note}")
    return "\n".join(lines)


def _reach_table(rows: list[dict]) -> str:
    """Render the reach comparison."""
    if not rows:
        return "  (no reach rows)"
    head = (
        f"{'model':<18} {'enodes':>13} {'classes':>13} {'cost':>22} "
        f"{'chg':>4} {'cert':>6}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        en = f"{r['base_enodes']}->{r['add_enodes']}"
        cl = f"{r['base_classes']}->{r['add_classes']}"
        co = f"{r['base_cost']:.3g}->{r['add_cost']:.3g}"
        cert = "pass" if r["add_cert"] == "pass" else "FAIL"
        lines.append(
            f"{r['model']:<18} {en:>13} {cl:>13} {co:>22} "
            f"{'yes' if r['changed'] else 'no':>4} {cert:>6}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def run_impact() -> dict:
    """Run all three measurements; return the machine-readable result."""
    laws = new_laws()
    sink = TorchSink()
    cost_fn = _cost_fn(sink)

    bench, bench_err = _bench_cases()
    models, model_err = model_cases()
    synth = synthetic_cases()

    firings: list[Firing] = []
    set_fires: dict[str, dict[str, int]] = {}
    for case in [*bench, *models, *synth]:
        for law in laws:
            firings.append(_probe(case, law, sink, cost_fn))
        hit = _set_fires(case, laws)
        if hit:
            set_fires[f"{case.source}:{case.name}"] = hit

    real_terms = [c.term for c in [*bench, *models]]
    census = _relaxed_census(real_terms, laws)

    reach = [reach_row(c, laws, sink, cost_fn) for c in models]

    return {
        "laws": [law.name for law in laws],
        "bench_errors": bench_err,
        "model_export_errors": model_err,
        "firings": firings,
        "set_fires": set_fires,
        "census": census,
        "n_real_subterms": sum(
            len(_iter_subterms(t)) for t in real_terms
        ),
        "reach": reach,
    }


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable result."""
    payload = {
        "laws": result["laws"],
        "bench_errors": result["bench_errors"],
        "model_export_errors": result["model_export_errors"],
        "set_fires": result["set_fires"],
        "census": result["census"],
        "n_real_subterms": result["n_real_subterms"],
        "reach": result["reach"],
        "firings": [
            {
                "source": f.source,
                "case": f.case,
                "law": f.law,
                "fires": f.fires,
                "base_cost": f.base_cost,
                "out_cost": f.out_cost,
                "changed": f.changed,
                "verified": f.verified,
                "paid": f.paid,
                "note": f.note,
            }
            for f in result["firings"]
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def _print_report(result: dict) -> None:
    """Print the full human-readable report."""
    laws = new_laws()
    firings: list[Firing] = result["firings"]
    n_bench = sum(1 for f in firings if f.source == "bench")
    n_model = sum(1 for f in firings if f.source == "model")
    n_synth = sum(1 for f in firings if f.source == "synthetic")

    print(
        "== law_impact — do the 11 proposed laws pay on real models? =="
    )
    print(
        f"   probes: {n_bench // len(laws)} bench x {len(laws)} laws, "
        f"{n_model // len(laws)} model x {len(laws)} laws, "
        f"{n_synth // len(laws)} synthetic x {len(laws)} laws"
    )
    if result["bench_errors"]:
        print(f"   bench build errors: {result['bench_errors']}")
    if result["model_export_errors"]:
        print("   model export failures:")
        for e in result["model_export_errors"]:
            print(f"     - {e}")
    print()

    print("-- (a) firing table (bench / model / synthetic / total) --")
    print(_fire_table(firings, laws))
    print()

    print(
        "-- (a2) relaxed-pattern census over "
        f"{result['n_real_subterms']} real subterms --"
    )
    print(_census_table(result))
    print()

    print("-- (b) cost delta for every firing --")
    print(_delta_table(firings))
    print()

    print("-- whole-set firings (all 11 laws at once) --")
    if result["set_fires"]:
        for key, hit in sorted(result["set_fires"].items()):
            print(f"  {key}: {hit}")
    else:
        print("  (the set fired on no case)")
    print()

    print("-- (c) reach: ALL_RULES vs ALL_RULES + new laws --")
    print(_reach_table(result["reach"]))
    print()

    _verdict(result)


def _verdict(result: dict) -> None:
    """Print the plain verdict the retro records."""
    firings: list[Firing] = result["firings"]
    real = [
        f for f in firings if f.source in {"bench", "model"} and f.fires
    ]
    real_paid = [f for f in real if f.paid]
    model = [f for f in firings if f.source == "model" and f.fires]
    reach_changed = [r for r in result["reach"] if r["changed"]]

    print("== verdict ==")
    if not real:
        print(
            "  NONE of the 11 laws fires on any real bench case or model "
            "graph."
        )
    else:
        print(
            f"  {len(real)} real firing(s); {len(real_paid)} paid "
            f"(lowered the extracted cost)."
        )
    print(f"  model firings: {len(model)}")
    print(f"  reach changed by adding the laws: {len(reach_changed)}")


def main(argv: list[str] | None = None) -> int:
    """Run the impact measurement and print (or dump) the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    result = run_impact()
    _print_report(result)
    if args.json:
        _dump_json(args.json, result)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
