# ruff: noqa: RUF002, RUF003 — σ and ⊗ in docstrings are deliberate
# math notation, per the laws-module convention.
"""Tests for the ``glu_fold`` law — the manual-GLU fold
(:data:`catopt_core.laws.tensor.GLU_FOLD`).

    mul(chunk(u,2,d,0), sigmoid(chunk(u,2,d,1))) -> glu(u, dim=d)

The ``softmax_fold``/``silu_fold`` recipe one op-family over — the
``project/retros/corpus-expansion-r2.md`` §6 lead: the corpus had the
kernel image (``GluMLP``'s ``glu`` node) but no manual spelling until
``ManualGluMLP`` closed the pair.  The export's getitem fold lands the
slice index on the ``chunk`` node itself, so the manual GLU is exactly
``chunk-half · σ(chunk-half)``.

Most of the precondition is structural — the shared ``u`` metavariable
binds both chunk operands to the same e-class, the shared ``D`` attr
metavariable pins them to the same split axis (and carries it to the
RHS), and the literal ``chunks``/``index`` attrs pin the two-halves
split and the gate order.  The residual guard is parity of the split
axis: ``chunk(·, 2, d)`` splits an odd axis first-big (n=3 → 2+1),
which still broadcasts — folding it would mint a ``glu`` member that
raises at eval.  That needs the attr-named axis, which the cond DSL
cannot index, so it stays a procedural ``check`` over ``cond``'s
expressible front (``u`` shaped, rank ≥ 1).

Covered surface:

* registration/tagging (``SIMPLIFICATION``, in ``DEFAULT``);
* a term-level match/instantiate round-trip;
* numeric soundness on real fp64 tensors (lhs == rhs);
* firing + RHS-is-member;
* the matcher declines — different sources, different dims,
  ``chunks != 2``, swapped index order, sigmoid-free mul;
* the check declines — odd split axis, ``None``/unknown split dim,
  out-of-range dim, non-int dim, scalar ``u``;
* an end-to-end run on the real ``ManualGluMLP`` export: the law
  fires, the extracted cost drops, the certificate replays, and the
  lowered before/after modules agree under ``sink.verify``;
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
    GLU_FOLD,
    SIMPLIFICATION_RULES,
    _check_glu_fold,
    _check_glu_shaped,
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
from catopt_torch.models import ManualGluMLP
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


def _manual_glu_lhs(u=None, v=None, **kw):
    """``mul(chunk(u,i0), sigmoid(chunk(v,i1)))`` — *v* defaults to *u*
    (a legal LHS instance; pass a different var or ``chunks``/``dim``/
    ``index`` kwargs to build the declined shapes)."""
    u = u if u is not None else Var("u", TensorType((4, 8)))
    v = v if v is not None else u
    chunks = kw.get("chunks", 2)
    d0 = kw.get("d0", -1)
    d1 = kw.get("d1", -1)
    i0 = kw.get("i0", 0)
    i1 = kw.get("i1", 1)
    return Op.make(
        "mul",
        Op.make("chunk", u, chunks=chunks, dim=d0, index=i0),
        Op.make(
            "sigmoid",
            Op.make("chunk", v, chunks=chunks, dim=d1, index=i1),
        ),
    )


# ---------------------------------------------------------------------------
#  Registration / tagging
# ---------------------------------------------------------------------------


def test_glu_fold_registered_and_tagged():
    """``glu_fold`` is a shipped simplification, in the default set."""
    assert GLU_FOLD in SIMPLIFICATION_RULES
    assert GLU_FOLD.tags == {tags.SIMPLIFICATION}
    assert GLU_FOLD.kind == "axiom"
    assert "glu_fold" in {r.name for r in DEFAULT}
    assert "glu_fold" in {r.name for r in ALL_RULES}


# ---------------------------------------------------------------------------
#  Match / instantiate round-trip
# ---------------------------------------------------------------------------


def test_glu_fold_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the shared leaf,
    the split-axis attr, and the ``glu`` RHS."""
    u = Var("u", TensorType((4, 8)))
    subst = {"u": u, "$attr:D": -1}
    assert pattern_metavars(GLU_FOLD.lhs) == {"$attr:D", "u"}
    term = instantiate_pattern(GLU_FOLD.lhs, subst)
    assert term == _manual_glu_lhs(u=u)
    found = match_pattern(GLU_FOLD.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(GLU_FOLD.lhs, found) == term
    assert instantiate_pattern(GLU_FOLD.rhs, found) == Op.make(
        "glu", u, dim=-1
    )


# ---------------------------------------------------------------------------
#  Numeric soundness
# ---------------------------------------------------------------------------


def test_glu_fold_sound_fp64():
    """``a · σ(b)`` over the chunk halves == ``F.glu(u)`` — to ~1e-16,
    not bitwise: the fused kernel rounds differently."""
    u = torch.randn(4, 8, dtype=torch.float64)
    a, b = u.chunk(2, dim=-1)
    lhs = a * torch.sigmoid(b)
    rhs = torch.nn.functional.glu(u, dim=-1)
    assert torch.allclose(lhs, rhs, atol=1e-15, rtol=0)


def test_glu_fold_sound_fp64_other_axis():
    """The fold is axis-generic — ``dim`` is carried through the shared
    ``D`` attr metavariable, not pinned to -1."""
    u = torch.randn(8, 4, dtype=torch.float64)
    a, b = u.chunk(2, dim=0)
    lhs = a * torch.sigmoid(b)
    rhs = torch.nn.functional.glu(u, dim=0)
    assert torch.allclose(lhs, rhs, atol=1e-15, rtol=0)


# ---------------------------------------------------------------------------
#  Firing + the matcher precondition declines
# ---------------------------------------------------------------------------


def test_glu_fold_fires_and_rhs_is_member():
    u = Var("u", TensorType((4, 8)))
    eg, root, _best = _saturate(
        _manual_glu_lhs(u=u), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) > 0
    rhs = Op.make("glu", u, dim=-1)
    assert list(eg.matches(rhs, eg.find(root)))


def test_glu_fold_fires_on_inner_axis():
    """``dim=0`` chunks fold too — the axis is data, not a literal."""
    u = Var("u", TensorType((8, 4)))
    eg, root, _best = _saturate(
        _manual_glu_lhs(u=u, d0=0, d1=0), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) > 0
    assert list(eg.matches(Op.make("glu", u, dim=0), eg.find(root)))


def test_glu_fold_declines_on_source_mismatch():
    """``mul(chunk(u), σ(chunk(v)))`` with u ≠ v — the shared ``u``
    metavariable makes the matcher veto, no check needed."""
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    eg, _root, _best = _saturate(
        _manual_glu_lhs(u=u, v=v), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) == 0


def test_glu_fold_declines_on_dim_mismatch():
    """Chunks along different axes cannot be halves of one split."""
    u = Var("u", TensorType((4, 8)))
    eg, _root, _best = _saturate(
        _manual_glu_lhs(u=u, d0=-1, d1=0), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) == 0


def test_glu_fold_declines_on_wider_chunk():
    """``chunks=4`` is a quarters split, not halves — literal attr."""
    u = Var("u", TensorType((4, 8)))
    eg, _root, _best = _saturate(
        _manual_glu_lhs(u=u, chunks=4), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) == 0


def test_glu_fold_declines_on_swapped_gate():
    """``mul(chunk1, σ(chunk0))`` — σ on the wrong half is not glu."""
    u = Var("u", TensorType((4, 8)))
    eg, _root, _best = _saturate(
        _manual_glu_lhs(u=u, i0=1, i1=0), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) == 0


def test_glu_fold_declines_on_sigmoid_free_mul():
    """A plain ``mul(chunk0, chunk1)`` has no sigmoid operand."""
    u = Var("u", TensorType((4, 8)))
    term = Op.make(
        "mul",
        Op.make("chunk", u, chunks=2, dim=-1, index=0),
        Op.make("chunk", u, chunks=2, dim=-1, index=1),
    )
    eg, _root, _best = _saturate(term, [GLU_FOLD], _cost_fn())
    assert eg.rule_fires.get("glu_fold", 0) == 0


# ---------------------------------------------------------------------------
#  The shape guard — parity of the split axis
# ---------------------------------------------------------------------------


def test_glu_fold_declines_on_odd_split_axis():
    """``chunk(u,2)`` on an odd axis splits 2+1 — evaluable under
    broadcast but NOT ``glu``, which halves exactly.  The check vetoes."""
    u = Var("u", TensorType((4, 3)))
    eg, _root, _best = _saturate(
        _manual_glu_lhs(u=u), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) == 0


def test_glu_fold_declines_on_unknown_split_dim():
    """A ``None`` ON the split axis cannot be proven even — decline."""
    u = Var("u", TensorType((4, None)))
    eg, _root, _best = _saturate(
        _manual_glu_lhs(u=u), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) == 0


def test_glu_fold_fires_with_unknown_dims_off_axis():
    """A ``None`` elsewhere does not matter — only the split axis's
    parity is the precondition."""
    u = Var("u", TensorType((None, 8)))
    eg, _root, _best = _saturate(
        _manual_glu_lhs(u=u), [GLU_FOLD], _cost_fn()
    )
    assert eg.rule_fires.get("glu_fold", 0) > 0


def test_check_glu_fold_branches():
    """Direct ``bound`` probes pin every decline path of the check."""
    shaped = Var("u", TensorType((4, 8)))
    assert _check_glu_fold({"u": shaped, "$attr:D": -1})
    # unshaped u — _shape_of returns non-tuple
    assert not _check_glu_fold(
        {"u": Var("u", TensorType(None)), "$attr:D": -1}
    )
    # non-int dim attr
    assert not _check_glu_fold({"u": shaped, "$attr:D": (-1,)})
    # dim out of range
    assert not _check_glu_fold({"u": shaped, "$attr:D": 5})
    # scalar u — rank 0 has no axis to halve (also the cond's veto)
    assert not _check_glu_fold(
        {"u": Var("u", TensorType(())), "$attr:D": 0}
    )
    assert not _check_glu_shaped({"u": Var("u", TensorType(()))})
    assert _check_glu_shaped({"u": shaped})


# ---------------------------------------------------------------------------
#  End-to-end on the real corpus export
# ---------------------------------------------------------------------------


def test_glu_fold_end_to_end_manual_glu():
    """The law fires on the ``ManualGluMLP`` export, the extracted cost
    drops, the certificate replays, and the lowered before/after
    modules agree under ``sink.verify``."""
    torch.manual_seed(0)
    model = ManualGluMLP(8, 2).eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    ir, leaves = export_to_ir(model, x)
    cost_fn = _cost_fn()

    without = [r for r in DEFAULT if r.name != "glu_fold"]
    with_law = list(DEFAULT)

    eg0, _r0, best0 = _saturate(ir.root, without, cost_fn)
    eg1, _r1, best1 = _saturate(ir.root, with_law, cost_fn)

    assert eg1.rule_fires.get("glu_fold", 0) > 0
    assert eg0.rule_fires.get("glu_fold", 0) == 0
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


def test_glu_fold_fires_in_default_pipeline():
    """The public default pipeline sees the law fire and verifies."""
    torch.manual_seed(0)
    model = ManualGluMLP(8, 2).eval().double()
    x = torch.randn(4, 8, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x, max_iterations=6, max_enodes=100_000)
    assert res.stats["rule_fires"].get("glu_fold", 0) > 0
    low = opt.lower(res, x, verify=True)
    assert low.verified is not None and low.verified.passed


def test_glu_fold_does_not_fire_on_kernel_glu():
    """``glu(u)`` itself is not a redex — the fold never loops; a
    ``GluMLP``-shaped term gains no member."""
    u = Var("u", TensorType((4, 8)))
    term = Op.make("glu", u, dim=-1)
    eg, _root, _best = _saturate(term, [GLU_FOLD], _cost_fn())
    assert eg.rule_fires.get("glu_fold", 0) == 0
