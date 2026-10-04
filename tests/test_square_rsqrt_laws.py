"""Tests for the square/rsqrt canonicalization bridge laws —
``mul_square`` and the three reciprocal-root folds
``div_sqrt_to_rsqrt`` / ``pow_to_rsqrt`` / ``recip_sqrt_to_rsqrt``
(:data:`catopt_core.laws.tensor`).

    mul(u, u)                    -> square(u)
    div(1, sqrt(u))              -> rsqrt(u)
    pow(u, Const(-0.5))          -> rsqrt(u)
    reciprocal(sqrt(u))          -> rsqrt(u)

All four are the recorded-miss bridges from
``project/retros/rms-norm-law.md``: the RMSNorm fold pins
``pow(u, 2)`` inside ``mean`` and ``rsqrt(·)`` outside it, so a graph
spelling x² as ``x·x`` or the reciprocal root as ``1/√(·)`` /
``(·)^-0.5`` could never match.  ``mul_square`` is
``square_expand``'s definitional inverse (a lemma); the rsqrt folds
canonicalize the spellings into the member the fold needs —
``reciprocal(sqrt(u))`` is what ``1/sqrt`` actually exports as (aten
lowers scalar-over-tensor division to ``reciprocal(t) * 1``, and
``id_mul`` strips the unit).

Preconditions live in the pattern or the ``cond`` DSL: the shared
``u`` metavariable binds both mul operands to the same e-class; the
``const-cmp`` predicates accept ``Const(1)``/``Const(1.0)`` numerators
and ``Const(-0.5)`` exponents numerically while a ``Var``/``Param``
binding declines.

Covered surface:

* registration/tagging and kernel kinds;
* term-level match/instantiate round-trips;
* numeric soundness on real fp64 tensors (lhs == rhs);
* firing + RHS-is-member;
* the declines — operand-mismatched mul, non-unit numerator,
  non-``-0.5`` exponent, non-sqrt ``reciprocal`` argument;
* the e-graph reach the laws exist for: every recorded miss spelling
  of a manual RMSNorm (``x·x`` square, ``1/√`` as reciprocal, and
  ``pow(·, -0.5)`` — built as terms AND exported from real modules)
  lands the ``rms_norm`` member in the root e-class under
  ``ALL_RULES`` and survives extraction + ``sink.verify``;
* the ``mul_square``/``square_expand`` inverse pair sharing one
  e-class;
* the laws firing inside the public default pipeline.
"""

from __future__ import annotations

import torch
from catopt_core.cost import backend_cost, dag_cost, executor_cost_for
from catopt_core.egraph import EGraph, verify_certificate
from catopt_core.ir import Const, IR, Op, Param, TensorType, Var
from catopt_core.laws import (
    ALL_RULES,
    DEFAULT,
    DIV_SQRT_TO_RSQRT,
    MUL_SQUARE,
    POW_TO_RSQRT,
    RECIP_SQRT_TO_RSQRT,
    SIMPLIFICATION_RULES,
    SQUARE_EXPAND,
    tags,
)
from catopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)
from catopt_orchestrator import Optimizer
from catopt_orchestrator.optimize import _lower_extracted
from catopt_torch.adapters import TorchSink
from catopt_torch.backend import TorchBackend
from catopt_torch.torch_bridge import export_to_ir

_SINK = TorchSink()


def _cost_fn():
    """The pipeline's selection model (roofline + per-dispatch term)."""
    return backend_cost(
        executor_cost_for(lowering="generic"), _SINK.supported_ops
    )


def _saturate(term, rules, iters=6, nodes=60_000):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(rules, root, max_iterations=iters, max_nodes=nodes)
    return eg, root, eg.extract_best(root, _cost_fn())


def _params_of(term):
    """Every ``Param`` leaf of *term* (DAG-aware)."""
    seen: set[int] = set()
    out: dict[str, Param] = {}
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Param):
            out[t.name] = t
        elif isinstance(t, Op):
            stack.extend(t.args)
    return out


def _ir_of(term, ir):
    """Wrap *term* as an ``IR`` reusing *ir*'s inputs."""
    return IR(
        root=term,
        inputs=list(ir.inputs),
        input_names=set(ir.input_names),
        params=_params_of(term),
    )


def _mean(x: Op | Var) -> Op:
    """``mean(x, dim=(-1,), keepdim=True)`` — the rms reduce."""
    return Op.make("mean", x, dim=(-1,), keepdim=True)


def _rms_reduces(u):
    """Every recorded-miss spelling of ``rsqrt(mean(u²) + eps)`` —
    plus the canonical spelling (the control: it folds with no
    bridges) and the doubly-missed spelling (``x·x`` AND ``1/√``)."""
    mean_pow = _mean(Op.make("pow", u, Const(2)))
    mean_mul = _mean(Op.make("mul", u, u))
    eps_pow = Op.make("add", mean_pow, Const(1e-6))
    eps_mul = Op.make("add", mean_mul, Const(1e-6))
    return {
        "canonical_rsqrt": Op.make("rsqrt", eps_pow),
        "mul_spelled_x2": Op.make("rsqrt", eps_mul),
        "div_one_sqrt": Op.make(
            "div", Const(1), Op.make("sqrt", eps_pow)
        ),
        "recip_sqrt": Op.make("reciprocal", Op.make("sqrt", eps_pow)),
        "pow_neghalf": Op.make("pow", eps_pow, Const(-0.5)),
        "both_missed": Op.make(
            "div", Const(1), Op.make("sqrt", eps_mul)
        ),
    }


def _manual_rms(rec: Op, w: Param) -> Op:
    """``mul(mul(u, rec), w)`` — the gained fold's LHS shape."""
    u = Var("u", TensorType((4, 8)))
    return Op.make("mul", Op.make("mul", u, rec), w)


class _ManualRmsRecip(torch.nn.Module):
    """``x · (1/√mean(x·x)+eps) · w`` — 1/√ spelled ``1.0 / sqrt``."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = (x * x).mean(-1, keepdim=True)
        return x * (1.0 / torch.sqrt(m + 1e-6)) * self.w


class _ManualRmsPow(torch.nn.Module):
    """``x · (mean(x·x)+eps)^{-0.5} · w`` — the pow(-0.5) spelling."""

    def __init__(self) -> None:
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = (x * x).mean(-1, keepdim=True)
        return x * (m + 1e-6).pow(-0.5) * self.w


# ---------------------------------------------------------------------------
#  Registration / tagging / kernel kind
# ---------------------------------------------------------------------------


def test_bridge_laws_registered_and_tagged():
    """All four bridges ship in the simplification + default sets."""
    for rule in (
        MUL_SQUARE,
        DIV_SQRT_TO_RSQRT,
        POW_TO_RSQRT,
        RECIP_SQRT_TO_RSQRT,
    ):
        assert rule in SIMPLIFICATION_RULES
        assert rule.tags == {tags.SIMPLIFICATION}
        assert rule.name in {r.name for r in DEFAULT}
        assert rule.name in {r.name for r in ALL_RULES}
    # mul_square is square_expand's lemma; the rsqrt folds are axioms.
    assert MUL_SQUARE.kind == "lemma"
    assert MUL_SQUARE.derivation == ("square_expand",)
    for rule in (DIV_SQRT_TO_RSQRT, POW_TO_RSQRT, RECIP_SQRT_TO_RSQRT):
        assert rule.kind == "axiom", rule.name


# ---------------------------------------------------------------------------
#  Match / instantiate round-trips
# ---------------------------------------------------------------------------


def test_mul_square_match_instantiate_roundtrip():
    u = Var("u", TensorType((4, 8)))
    assert pattern_metavars(MUL_SQUARE.lhs) == {"u"}
    term = instantiate_pattern(MUL_SQUARE.lhs, {"u": u})
    assert term == Op.make("mul", u, u)
    found = match_pattern(MUL_SQUARE.lhs, term)
    assert found is not None and found == {"u": u}
    assert instantiate_pattern(MUL_SQUARE.rhs, found) == Op.make(
        "square", u
    )


def test_rsqrt_law_match_instantiate_roundtrips():
    u = Var("u", TensorType((4, 8)))
    lhs_terms = {
        DIV_SQRT_TO_RSQRT: Op.make(
            "div", Const(1), Op.make("sqrt", u)
        ),
        POW_TO_RSQRT: Op.make("pow", u, Const(-0.5)),
        RECIP_SQRT_TO_RSQRT: Op.make("reciprocal", Op.make("sqrt", u)),
    }
    for rule, term in lhs_terms.items():
        found = match_pattern(rule.lhs, term)
        assert found is not None, rule.name
        assert found["u"] == u
        assert instantiate_pattern(rule.rhs, found) == Op.make(
            "rsqrt", u
        )


def test_rsqrt_conds_accept_and_reject_const_leaves():
    """``const-cmp`` reads ``Const.value`` numerically — ``Const(1)``,
    ``Const(1.0)`` and ``Const(-0.5)`` accept; a ``Var`` leaf or a
    wrong literal declines."""
    u = Var("u", TensorType((4, 8)))
    w = Var("w", TensorType((4, 8)))
    assert DIV_SQRT_TO_RSQRT.cond is not None
    assert POW_TO_RSQRT.cond is not None
    assert DIV_SQRT_TO_RSQRT.check({"ONE": Const(1), "u": u})
    assert DIV_SQRT_TO_RSQRT.check({"ONE": Const(1.0), "u": u})
    assert not DIV_SQRT_TO_RSQRT.check({"ONE": Const(2), "u": u})
    assert not DIV_SQRT_TO_RSQRT.check({"ONE": w, "u": u})
    assert POW_TO_RSQRT.check({"P": Const(-0.5), "u": u})
    assert POW_TO_RSQRT.check({"P": Const(-0.5), "u": u})
    assert not POW_TO_RSQRT.check({"P": Const(0.5), "u": u})
    assert not POW_TO_RSQRT.check({"P": w, "u": u})


# ---------------------------------------------------------------------------
#  Numeric soundness
# ---------------------------------------------------------------------------


def test_bridge_laws_sound_fp64():
    """lhs == rhs on real fp64 tensors — to ~1e-16 (rsqrt's kernel
    rounds differently than sqrt+div), not bitwise."""
    u = torch.randn(4, 8, dtype=torch.float64).abs() + 0.25
    assert torch.allclose(u * u, torch.square(u), atol=1e-15, rtol=0)
    rsqrt = torch.rsqrt(u)
    assert torch.allclose(
        1.0 / torch.sqrt(u), rsqrt, atol=1e-15, rtol=0
    )
    assert torch.allclose(
        torch.reciprocal(torch.sqrt(u)), rsqrt, atol=1e-15, rtol=0
    )
    assert torch.allclose(
        torch.pow(u, -0.5), rsqrt, atol=1e-15, rtol=0
    )


# ---------------------------------------------------------------------------
#  Firing + the declines
# ---------------------------------------------------------------------------


def test_mul_square_fires_and_rhs_is_member():
    u = Var("u", TensorType((4, 8)))
    eg, root, _best = _saturate(Op.make("mul", u, u), [MUL_SQUARE])
    assert eg.rule_fires.get("mul_square", 0) > 0
    assert list(eg.matches(Op.make("square", u), eg.find(root)))


def test_mul_square_declines_on_operand_mismatch():
    """``mul(u, v)`` — the shared ``u`` metavariable vetoes; no check."""
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    eg, _root, _best = _saturate(Op.make("mul", u, v), [MUL_SQUARE])
    assert eg.rule_fires.get("mul_square", 0) == 0


def test_div_sqrt_to_rsqrt_fires_and_declines():
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    sq = Op.make("sqrt", u)
    # Const(1) and Const(1.0) numerators both fold (numeric compare).
    for one in (Const(1), Const(1.0)):
        eg, root, _best = _saturate(
            Op.make("div", one, sq), [DIV_SQRT_TO_RSQRT]
        )
        assert eg.rule_fires.get("div_sqrt_to_rsqrt", 0) > 0
        assert list(eg.matches(Op.make("rsqrt", u), eg.find(root)))
    # A non-unit numerator or a non-Const leaf declines; a bare
    # numerator-without-sqrt denominator has no redex at all.
    for term in (
        Op.make("div", Const(2), sq),
        Op.make("div", v, sq),
        Op.make("div", Const(1), u),
    ):
        eg, _root, _best = _saturate(term, [DIV_SQRT_TO_RSQRT])
        assert eg.rule_fires.get("div_sqrt_to_rsqrt", 0) == 0


def test_pow_to_rsqrt_fires_and_declines():
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    eg, root, _best = _saturate(
        Op.make("pow", u, Const(-0.5)), [POW_TO_RSQRT]
    )
    assert eg.rule_fires.get("pow_to_rsqrt", 0) > 0
    assert list(eg.matches(Op.make("rsqrt", u), eg.find(root)))
    for exp in (Const(0.5), Const(2), Const(-1), v):
        eg, _root, _best = _saturate(Op.make("pow", u, exp), [POW_TO_RSQRT])
        assert eg.rule_fires.get("pow_to_rsqrt", 0) == 0


def test_recip_sqrt_to_rsqrt_fires_and_declines():
    u = Var("u", TensorType((4, 8)))
    eg, root, _best = _saturate(
        Op.make("reciprocal", Op.make("sqrt", u)),
        [RECIP_SQRT_TO_RSQRT],
    )
    assert eg.rule_fires.get("recip_sqrt_to_rsqrt", 0) > 0
    assert list(eg.matches(Op.make("rsqrt", u), eg.find(root)))
    # reciprocal(u) with no sqrt operand — no match.
    eg, _root, _best = _saturate(
        Op.make("reciprocal", u), [RECIP_SQRT_TO_RSQRT]
    )
    assert eg.rule_fires.get("recip_sqrt_to_rsqrt", 0) == 0


# ---------------------------------------------------------------------------
#  The inverse pair shares one e-class
# ---------------------------------------------------------------------------


def test_mul_square_expand_share_one_eclass():
    """``square(u)`` gains the ``mul(u,u)`` member under
    ``square_expand``; the fold's match on that member is a no-op
    merge — the two directions coexist in one class."""
    u = Var("u", TensorType((4, 8)))
    term = Op.make("square", u)
    eg, root, _best = _saturate(term, [SQUARE_EXPAND, MUL_SQUARE])
    assert eg.rule_fires.get("square_expand", 0) > 0
    assert eg.find(root) == eg.find(eg.add_term(Op.make("mul", u, u)))


# ---------------------------------------------------------------------------
#  The reach the laws exist for — every miss spelling folds to rms_norm
# ---------------------------------------------------------------------------


def test_every_miss_spelling_reaches_rms_norm():
    """Each recorded-miss spelling of the manual RMSNorm lands the
    ``rms_norm`` member in the root e-class under ``ALL_RULES`` — the
    canon laws mint the ``rsqrt``/``pow(·, 2)`` members the fold's
    pattern pins, then ``rms_norm_fold`` fires."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    want = Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)
    for name, rec in _rms_reduces(u).items():
        eg = EGraph()
        root = eg.add_term(Op.make("mul", Op.make("mul", u, rec), w))
        eg.run(ALL_RULES, root, max_iterations=6, max_nodes=60_000)
        fires = eg.rule_fires
        assert eg.find(root) == eg.find(eg.add_term(want)), name
        assert fires.get("rms_norm_fold", 0) > 0, name


def test_canonical_rsqrt_spellings_fold_with_their_bridges():
    """The bridges are load-bearing, not incidental: with the canon
    laws removed, ``mul(x,x)``/``1/√``/``pow(·,-0.5)`` spellings do
    NOT fold — exactly the recorded misses."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    want = Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)
    bridge_names = {
        "mul_square",
        "div_sqrt_to_rsqrt",
        "pow_to_rsqrt",
        "recip_sqrt_to_rsqrt",
    }
    without = [r for r in ALL_RULES if r.name not in bridge_names]
    for name, rec in _rms_reduces(u).items():
        term = Op.make("mul", Op.make("mul", u, rec), w)
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(without, root, max_iterations=6, max_nodes=60_000)
        fires = eg.rule_fires
        folded = eg.find(root) == eg.find(eg.add_term(want))
        if name == "canonical_rsqrt":
            # The canonical spelling folds WITHOUT the bridges —
            # the fold itself was never broken.
            assert folded and fires.get("rms_norm_fold", 0) > 0
        else:
            # Every miss spelling: no fold fires (the canonical
            # ``rsqrt``/``pow(·, 2)`` member is never minted).
            assert not folded, name
            assert fires.get("rms_norm_fold", 0) == 0, name


def test_real_export_spellings_fold_and_verify():
    """The two real-export spellings — ``1/√`` (reciprocal + a mul-by-1
    wrapper) and ``pow(·, -0.5)`` — fold to ``rms_norm`` end to end:
    firing on the exported graph, the extracted member, the replayed
    certificate, and the lowered before/after modules agreeing."""
    x = torch.randn(4, 8, dtype=torch.float64)
    for model in (
        _ManualRmsRecip().eval().double(),
        _ManualRmsPow().eval().double(),
    ):
        ir, leaves = export_to_ir(model, x)
        cost_fn = _cost_fn()
        eg = EGraph()
        root = eg.add_term(ir.root)
        eg.run(ALL_RULES, root, max_iterations=6, max_nodes=60_000)
        assert eg.rule_fires.get("rms_norm_fold", 0) > 0
        best = eg.extract_best(root, cost_fn)
        assert best.op == "rms_norm"
        assert dag_cost(best, cost_fn) <= dag_cost(ir.root, cost_fn)
        cert = eg.certificate(ir.root, best, cost_fn=cost_fn)
        verify_certificate(ir.root, cert)
        m0 = _lower_extracted(
            ir.root, _ir_of(ir.root, ir), leaves, _SINK
        )
        m1 = _lower_extracted(best, _ir_of(best, ir), leaves, _SINK)
        assert _SINK.verify(m0, m1, (x,), rtol=1e-4).passed


def test_bridges_fire_in_default_pipeline():
    """The public default pipeline sees the canonicalization fire on
    the real ``pow(·, -0.5)``-spelled export and verifies."""
    model = _ManualRmsPow().eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x, max_iterations=6, max_enodes=100_000)
    assert res.stats["rule_fires"].get("pow_to_rsqrt", 0) > 0
    assert res.stats["rule_fires"].get("mul_square", 0) > 0
    low = opt.lower(res, x, verify=True)
    assert low.verified is not None and low.verified.passed
