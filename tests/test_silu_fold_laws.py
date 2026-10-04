# ruff: noqa: RUF002, RUF003 — σ and × in docstrings are deliberate
# math notation, per the laws-module convention.
"""Tests for the ``silu_fold`` law — the manual-silu fold
(:data:`catopt_core.laws.tensor.SILU_FOLD`).

    mul(x, sigmoid(x)) -> silu(x)

The definitional inverse of ``silu_expand``, admitted as the
mediating 3-cell of the coherence catalogue's two divergent critical
pairs (``silu_expand × swiglu_fuse`` and ``silu_mul_form ×
swiglu_fuse`` — see ``project/retros/law-coherence-catalogue.md`` §4
and ``project/retros/three-cell-mediator.md``).  Expanding ``silu``
inside a ``mul`` destroys the ``swiglu_fuse`` redex; the fold
transports the expansion back into the gate — ``mul((g·σg), u) →
mul(silu(g), u)`` — so the fuse path is reachable again and the
one-step reducts rejoin.

Like ``softmax_fold``, the precondition is carried by the pattern,
not a ``check`` hook: the shared ``x`` metavariable binds both mul
operands to the same e-class, so ``mul(x, σ(y))`` never fires.  The
term-local inverse of an existing simplification — at most one new
member per e-class, no closure growth.

Covered surface:

* registration/tagging;
* a term-level match/instantiate round-trip;
* numeric soundness on real fp64 tensors (lhs == rhs);
* firing + RHS-is-member;
* the declines — mismatched mul operands, a sigmoid-free mul;
* the critical pair itself: on the expanded SwiGLU reduct the fold
  restores the ``swiglu_fuse`` redex, the fused member lands in the
  root e-class, and the certificate replays;
* an end-to-end run on a real manual-silu export: the law fires, the
  extracted cost drops, the certificate replays, and the lowered
  before/after modules agree under ``sink.verify``;
* the law firing inside the public default pipeline.
"""

from __future__ import annotations

import torch
from catopt_core.cost import backend_cost, dag_cost, executor_cost_for
from catopt_core.egraph import EGraph, verify_certificate
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_core.laws import (
    ALL_RULES,
    DEFAULT,
    SILU_EXPAND,
    SILU_FOLD,
    SIMPLIFICATION_RULES,
    SWIGLU_FUSE,
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


def _saturate(term, rules, cost_fn, iters=6, nodes=60_000):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(rules, root, max_iterations=iters, max_nodes=nodes)
    return eg, root, eg.extract_best(root, cost_fn)


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


def _manual_silu_lhs(u=None, v=None):
    """``mul(u, sigmoid(v or u))`` — *v* defaults to *u* (a legal LHS
    instance; pass a different var to build the declined operand-
    mismatch shape)."""
    u = u if u is not None else Var("u", TensorType((4, 8)))
    v = v if v is not None else u
    return Op.make("mul", u, Op.make("sigmoid", v))


def _expanded_swiglu():
    """The expand-side reduct of the critical pair:
    ``mul(mul(linA, σ(linA)), linB)`` — the fused redex destroyed."""
    x = Var("x", TensorType((4, 8)))
    a = Param("A", TensorType((8, 8)))
    b = Param("B", TensorType((8, 8)))
    lin_a = Op.make("linear", x, a)
    lin_b = Op.make("linear", x, b)
    return Op.make(
        "mul",
        Op.make("mul", lin_a, Op.make("sigmoid", lin_a)),
        lin_b,
    )


def _fused_swiglu():
    """The fuse-side reduct: ``swiglu_fuse``'s RHS on the same subst."""
    x = Var("x", TensorType((4, 8)))
    a = Param("A", TensorType((8, 8)))
    b = Param("B", TensorType((8, 8)))
    fused = Op.make("linear", x, Op.make("concat", a, b, dim=0))
    return Op.make(
        "mul",
        Op.make(
            "silu",
            Op.make("chunk", fused, chunks=2, dim=-1, index=0),
        ),
        Op.make("chunk", fused, chunks=2, dim=-1, index=1),
    )


class _ManualSiLU(torch.nn.Module):
    """``down(h · σ(h))`` — silu spelled by hand inside a gated MLP."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(dim, hidden, bias=False)
        self.down = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gate(x)
        return self.down(h * torch.sigmoid(h))


# ---------------------------------------------------------------------------
#  Registration / tagging
# ---------------------------------------------------------------------------


def test_silu_fold_registered_and_tagged():
    """``silu_fold`` is a shipped simplification, in the default set."""
    assert SILU_FOLD in SIMPLIFICATION_RULES
    assert SILU_FOLD.tags == {tags.SIMPLIFICATION}
    assert "silu_fold" in {r.name for r in DEFAULT}
    assert "silu_fold" in {r.name for r in ALL_RULES}


# ---------------------------------------------------------------------------
#  Match / instantiate round-trip
# ---------------------------------------------------------------------------


def test_silu_fold_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the shared leaf
    and the ``silu`` RHS."""
    u = Var("u", TensorType((4, 8)))
    subst = {"x": u}
    assert pattern_metavars(SILU_FOLD.lhs) == {"x"}
    term = instantiate_pattern(SILU_FOLD.lhs, subst)
    assert term == Op.make("mul", u, Op.make("sigmoid", u))
    found = match_pattern(SILU_FOLD.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(SILU_FOLD.lhs, found) == term
    assert instantiate_pattern(SILU_FOLD.rhs, found) == Op.make(
        "silu", u
    )


# ---------------------------------------------------------------------------
#  Numeric soundness
# ---------------------------------------------------------------------------


def test_silu_fold_sound_fp64():
    """``x · σ(x) == silu(x)`` on real fp64 tensors — to ~1e-16, not
    bitwise: the fused kernel rounds differently."""
    u = torch.randn(4, 8, dtype=torch.float64)
    lhs = u * torch.sigmoid(u)
    rhs = torch.nn.functional.silu(u)
    assert torch.allclose(lhs, rhs, atol=1e-15, rtol=0)


# ---------------------------------------------------------------------------
#  Firing + the matcher precondition declines
# ---------------------------------------------------------------------------


def test_silu_fold_fires_and_rhs_is_member():
    u = Var("u", TensorType((4, 8)))
    eg, root, _best = _saturate(
        _manual_silu_lhs(u=u), [SILU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("silu_fold", 0) > 0
    rhs = Op.make("silu", u)
    assert list(eg.matches(rhs, eg.find(root)))


def test_silu_fold_declines_on_operand_mismatch():
    """``mul(u, σ(v))`` with u ≠ v — the shared ``x`` metavariable
    makes the matcher veto, no check needed."""
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    eg, _root, _best = _saturate(
        _manual_silu_lhs(u=u, v=v), [SILU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("silu_fold", 0) == 0


def test_silu_fold_declines_on_sigmoid_free_mul():
    """A plain ``mul(u, v)`` has no sigmoid operand — no match."""
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    eg, _root, _best = _saturate(
        Op.make("mul", u, v), [SILU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("silu_fold", 0) == 0


# ---------------------------------------------------------------------------
#  The critical pair — the 3-cell this law was admitted for
# ---------------------------------------------------------------------------


def test_silu_fold_mediates_the_swiglu_critical_pair():
    """On the expand-side reduct ``mul((g·σg), u)``, ``swiglu_fuse``
    alone cannot fire (the silu is gone); adding ``silu_fold``
    restores the redex and the fused member lands in the root
    e-class."""
    t_expanded = _expanded_swiglu()
    t_fused = _fused_swiglu()

    # The divergence, pinned: the fuse law alone sees no redex.
    eg0, _r0, _b0 = _saturate(t_expanded, [SWIGLU_FUSE], _cost_fn())
    assert eg0.rule_fires.get("swiglu_fuse", 0) == 0

    # The mediator: fold the gate back, then the fuse fires and both
    # reducts share an e-class.
    eg, root, _best = _saturate(
        t_expanded, [SILU_FOLD, SWIGLU_FUSE], _cost_fn()
    )
    assert eg.rule_fires.get("silu_fold", 0) > 0
    assert eg.rule_fires.get("swiglu_fuse", 0) > 0
    assert eg.find(root) == eg.find(eg.add_term(t_fused))

    # The join replays standalone — a real derivation, not an
    # e-graph-dependent stub.
    cert = eg.certificate(t_expanded, t_fused)
    verify_certificate(t_expanded, cert)
    assert set(cert.rules_used) <= {"silu_fold", "swiglu_fuse"}


# ---------------------------------------------------------------------------
#  End-to-end on a real manual-silu export
# ---------------------------------------------------------------------------


def test_silu_fold_end_to_end_manual_silu():
    """The law fires on a real ``x·σ(x)`` export, the extracted cost
    drops, the certificate replays, and the lowered before/after
    modules agree under ``sink.verify``."""
    torch.manual_seed(0)
    model = _ManualSiLU(8, 16).eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    ir, leaves = export_to_ir(model, x)
    cost_fn = _cost_fn()

    without = [r for r in DEFAULT if r.name != "silu_fold"]
    with_law = list(DEFAULT)

    eg0, _r0, best0 = _saturate(ir.root, without, cost_fn)
    eg1, _r1, best1 = _saturate(ir.root, with_law, cost_fn)

    assert eg1.rule_fires.get("silu_fold", 0) > 0
    assert eg0.rule_fires.get("silu_fold", 0) == 0
    c0, c1 = dag_cost(best0, cost_fn), dag_cost(best1, cost_fn)
    assert c1 < c0
    assert best1 != best0

    # the certificate replays on the real export.
    cert = eg1.certificate(ir.root, best1, cost_fn=cost_fn)
    verify_certificate(ir.root, cert)

    # the lowered before/after modules are numerically equal.
    m0 = _lower_extracted(best0, _ir_of(best0, ir), leaves, _SINK)
    m1 = _lower_extracted(best1, _ir_of(best1, ir), leaves, _SINK)
    assert _SINK.verify(m0, m1, (x,), rtol=1e-4).passed


def test_silu_fold_fires_in_default_pipeline():
    """The public default pipeline sees the law fire and verifies."""
    torch.manual_seed(0)
    model = _ManualSiLU(8, 16).eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x, max_iterations=6, max_enodes=100_000)
    assert res.stats["rule_fires"].get("silu_fold", 0) > 0
    low = opt.lower(res, x, verify=True)
    assert low.verified is not None and low.verified.passed


def test_silu_expand_fold_share_one_eclass():
    """Both directions of the inverse pair live in one e-class: a
    ``silu`` term gains the expanded member under ``silu_expand``.
    The fold's match on that member is a no-op merge — its RHS is
    already in the class — so it is honestly not counted as a fire;
    the *starting-expanded* direction is the firing test above."""
    u = Var("u", TensorType((4, 8)))
    term = Op.make("silu", u)
    eg, root, _best = _saturate(
        term, [SILU_EXPAND, SILU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("silu_expand", 0) > 0
    expanded = Op.make("mul", u, Op.make("sigmoid", u))
    assert eg.find(root) == eg.find(eg.add_term(expanded))
