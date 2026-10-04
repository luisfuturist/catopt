# ruff: noqa: RUF003 — ⁻¹, σ-style math in docstrings is
# deliberate notation, per the laws-module convention.
"""Tests for the ``rms_norm_fold`` laws — the manual-RMSNorm folds
(:data:`catopt_core.laws.tensor.RMS_NORM_FOLD` and
:data:`catopt_core.laws.tensor.RMS_NORM_FOLD_NOGAIN`).

    mul(mul(u, rsqrt(add(mean(pow(u,2),MD,keepdim=MK),EPS))), w)
        -> rms_norm(u, w, dim=ND, eps=EP)
    mul(u, rsqrt(add(mean(pow(u,2),MD,keepdim=MK),EPS)))
        -> rms_norm(u, dim=ND, eps=EP)

The fourth machine-discovered law family admitted to the library —
the ``project/retros/corpus-expansion-r2.md`` §6 lead and the exact
``softmax_fold`` situation one norm family over: the corpus carries
BOTH spellings of the same math (``RMSNorm``/``NormLinear``/
``TransformerBlock``/``ParallelBlock``'s manual ``x·rms⁻¹·w`` and
``NativeRmsNorm``'s fused ``rms_norm`` kernel op).

The structural precondition is the op-tree itself plus the shared
``u`` metavariable (the numerator and the ``pow`` operand must be the
same e-class).  ``cond`` carries the expressible front — ``keepdim``
is True, ``eps`` is a numeric ``Const`` leaf, the ``pow`` exponent is
the literal 2 — and ``check`` carries what the DSL cannot: the
``mean``'s reduce dims must name exactly u's last ``k`` axes
(``F.rms_norm`` only normalizes a trailing block), and the gained
fold additionally requires ``w``'s shape to BE that trailing block
(``aten.rms_norm`` rejects any other weight shape at eval).
``derive`` mints the two RHS attrs the LHS cannot bind verbatim:
``dim`` is the *normalized shape* ``u.shape[-k:]``, not the reduce
dims, and ``eps`` unwraps the bound ``Const`` into the float attr.

Covered surface:

* registration/tagging (``SIMPLIFICATION``, in ``DEFAULT``);
* a term-level match/instantiate round-trip, derive included;
* the check/derive hooks unit-tested branch by branch;
* numeric soundness on real fp64 tensors (lhs == rhs, bitwise);
* firing + RHS-is-member, tuple and int ``dim`` spellings,
  multi-dim and gain-free sites;
* the cond declines — ``keepdim=False``, tensor ``eps``, exponent
  != 2 — and the check declines — non-trailing / duplicate /
  out-of-range dims, a wrong-shaped ``w`` — plus the matcher
  decline (numerator and ``pow`` operands differ);
* an end-to-end run on real ``RMSNorm``/``NormLinear``/
  ``TransformerBlock``/``ParallelBlock`` exports: the laws fire,
  the extracted cost drops, the certificate replays, and the
  lowered before/after modules agree under ``sink.verify``;
* the laws firing inside the public default pipeline.
"""

from __future__ import annotations

import torch
from catopt_core.cost import backend_cost, dag_cost, executor_cost_for
from catopt_core.egraph import EGraph, verify_certificate
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from catopt_core.laws import (
    DEFAULT,
    RMS_NORM_FOLD,
    RMS_NORM_FOLD_NOGAIN,
    SIMPLIFICATION_RULES,
    _check_rms_consts,
    _check_rms_fold,
    _check_rms_fold_nogain,
    _derive_rms_norm,
    tags,
)
from catopt_core.laws.tensor import _rms_normalized_shape
from catopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)
from catopt_orchestrator import Optimizer
from catopt_orchestrator.optimize import _lower_extracted
from catopt_torch.adapters import TorchSink
from catopt_torch.backend import TorchBackend
from catopt_torch.models import (
    NormLinear,
    ParallelBlock,
    RMSNorm,
    TransformerBlock,
)
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


def _rms_factor(u, dim, keepdim, eps, exponent=Const(2), v=None):
    """``rsqrt(mean(pow(v or u, exponent), dim, keepdim) + eps)``."""
    vv = v if v is not None else u
    return Op.make(
        "rsqrt",
        Op.make(
            "add",
            Op.make(
                "mean", Op.make("pow", vv, exponent), dim=dim,
                keepdim=keepdim,
            ),
            eps,
        ),
    )


def _gained_lhs(u, w, dim, keepdim, eps, **kw):
    """``mul(mul(u, rms(v or u)), w)`` — the gained fold's LHS."""
    return Op.make(
        "mul",
        Op.make("mul", u, _rms_factor(u, dim, keepdim, eps, **kw)),
        w,
    )


# ---------------------------------------------------------------------------
#  Registration / tagging
# ---------------------------------------------------------------------------


def test_rms_norm_folds_registered_and_tagged():
    """Both folds are shipped simplifications, in the default set."""
    for rule in (RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN):
        assert rule in SIMPLIFICATION_RULES
        assert rule.tags == {tags.SIMPLIFICATION}
    names = {r.name for r in DEFAULT}
    assert "rms_norm_fold" in names
    assert "rms_norm_fold_nogain" in names


# ---------------------------------------------------------------------------
#  Match / instantiate round-trip (derive supplies the RHS-only attrs)
# ---------------------------------------------------------------------------


def test_rms_norm_fold_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the same term, and
    ``derive`` produces the ``$attr:ND``/``$attr:EP`` the RHS needs."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    subst = {
        "u": u,
        "w": w,
        "P": Const(2),
        "EPS": Const(1e-6),
        "$attr:MD": (-1,),
        "$attr:MK": True,
    }
    assert pattern_metavars(RMS_NORM_FOLD.lhs) == {
        "u",
        "w",
        "P",
        "EPS",
        "$attr:MD",
        "$attr:MK",
    }
    term = instantiate_pattern(RMS_NORM_FOLD.lhs, subst)
    found = match_pattern(RMS_NORM_FOLD.lhs, term)
    assert found is not None
    assert found == subst
    assert instantiate_pattern(RMS_NORM_FOLD.lhs, found) == term
    # the RHS instantiates to the fused rms_norm — the reduce dims
    # becoming the trailing normalized_shape tuple, the bound Const
    # unwrapped to the float eps attr.
    rhs_subst = {**found, **RMS_NORM_FOLD.derive(found)}
    rhs = instantiate_pattern(RMS_NORM_FOLD.rhs, rhs_subst)
    assert rhs == Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)


def test_rms_norm_fold_nogain_match_instantiate_roundtrip():
    """The gain-free LHS — same metvars minus ``w``."""
    u = Var("u", TensorType((4, 8)))
    subst = {
        "u": u,
        "P": Const(2),
        "EPS": Const(1e-6),
        "$attr:MD": (-1,),
        "$attr:MK": True,
    }
    term = instantiate_pattern(RMS_NORM_FOLD_NOGAIN.lhs, subst)
    found = match_pattern(RMS_NORM_FOLD_NOGAIN.lhs, term)
    assert found == subst
    rhs_subst = {**found, **RMS_NORM_FOLD_NOGAIN.derive(found)}
    rhs = instantiate_pattern(RMS_NORM_FOLD_NOGAIN.rhs, rhs_subst)
    assert rhs == Op.make("rms_norm", u, dim=(8,), eps=1e-6)


# ---------------------------------------------------------------------------
#  check / derive hooks — branch by branch
# ---------------------------------------------------------------------------


def _bound(u_shape, w_shape=None, dims=(-1,), mk=True, eps=1e-6):
    """A ``bound`` dict as the matcher would build it."""
    b = {
        "u": Var("u", TensorType(u_shape)),
        "P": Const(2),
        "EPS": Const(eps),
        "$attr:MD": dims,
        "$attr:MK": mk,
    }
    if w_shape is not None:
        b["w"] = Param("w", TensorType(w_shape))
    return b


def test_check_rms_consts_cond_verdicts():
    """The cond front: keepdim, numeric-Const eps, exponent 2."""
    ok = _bound((4, 8))
    assert _check_rms_consts(ok)
    assert not _check_rms_consts({**ok, "$attr:MK": False})
    assert not _check_rms_consts({**ok, "EPS": Var("e", TensorType(()))})
    assert not _check_rms_consts({**ok, "P": Const(3)})
    assert not _check_rms_consts({**ok, "P": Var("p", TensorType(()))})


def test_rms_normalized_shape_accepts_trailing_blocks():
    """Tuple/int reduce dims over the last k axes mint ns = u.shape[-k:]."""
    assert _rms_normalized_shape(_bound((4, 8), dims=(-1,))) == (8,)
    assert _rms_normalized_shape(_bound((4, 8), dims=-1)) == (8,)
    assert _rms_normalized_shape(_bound((4, 8), dims=(1,))) == (8,)
    assert _rms_normalized_shape(_bound((2, 4, 8), dims=(-2, -1))) == (
        4,
        8,
    )
    assert _rms_normalized_shape(_bound((2, 4, 8), dims=(0, 1, 2))) == (
        2,
        4,
        8,
    )
    assert _rms_normalized_shape(_bound((8,), dims=(-1,))) == (8,)


def test_rms_normalized_shape_declines():
    """Non-trailing, duplicate, out-of-range, non-int, or unshaped."""
    assert _rms_normalized_shape(_bound((4, 8), dims=(0,))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(-1, -1))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(-3,))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(2,))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=())) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=None)) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=True)) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(True,))) is None
    assert _rms_normalized_shape(_bound((4, 8), dims="x")) is None
    assert _rms_normalized_shape(_bound((4, 8), dims=(1.5,))) is None
    # u's shape must be concrete — it IS the minted attr
    assert _rms_normalized_shape(_bound((None, 8))) is None
    assert _rms_normalized_shape(_bound((4, None))) is None


def test_check_rms_fold_weight_shape_gate():
    """The gained fold: ``w``'s shape must BE the normalized shape."""
    assert _check_rms_fold(_bound((4, 8), w_shape=(8,)))
    assert not _check_rms_fold(_bound((4, 8), w_shape=(4,)))
    assert not _check_rms_fold(_bound((4, 8), w_shape=(4, 8)))
    # the (None,)-dim weight cannot prove ns — strict posture
    assert not _check_rms_fold(_bound((4, 8), w_shape=(None,)))
    # multi-dim: w must cover the whole trailing block
    b = _bound((2, 4, 8), w_shape=(4, 8), dims=(-2, -1))
    assert _check_rms_fold(b)
    assert not _check_rms_fold(
        _bound((2, 4, 8), w_shape=(8,), dims=(-2, -1))
    )
    # ...but the gain-free fold fires on the same dims regardless of w
    assert _check_rms_fold_nogain(b)


def test_derive_rms_norm_mints_shape_and_eps():
    """``derive`` returns ``dim`` = u.shape[-k:] and the unwrapped eps."""
    b = _bound((2, 4, 8), dims=(-2, -1), eps=1e-5)
    assert _derive_rms_norm(b) == {
        "$attr:ND": (4, 8),
        "$attr:EP": 1e-5,
    }
    # the int exponent form floats the eps too
    b = _bound((4, 8), eps=0)
    assert _derive_rms_norm(b) == {"$attr:ND": (8,), "$attr:EP": 0.0}
    # defensive vetoes — the hook may be called on declined bindings
    assert _derive_rms_norm(_bound((4, 8), dims=(0,))) is None
    assert (
        _derive_rms_norm({**_bound((4, 8)), "EPS": Var("e", TensorType(()))})
        is None
    )


# ---------------------------------------------------------------------------
#  Numeric soundness
# ---------------------------------------------------------------------------


def test_rms_norm_fold_sound_fp64():
    """``x·rsqrt(mean(x²)+eps)·w == F.rms_norm(x, ns, w, eps)`` — bitwise
    on fp64: the kernel computes the same op sequence."""
    torch.manual_seed(0)
    x = torch.randn(4, 8, dtype=torch.float64)
    w = torch.randn(8, dtype=torch.float64)
    lhs = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    rhs = torch.nn.functional.rms_norm(x, (8,), w, 1e-6)
    assert torch.equal(lhs, rhs)


def test_rms_norm_fold_sound_fp64_multidim_and_weightless():
    """Trailing-block reduces of k>1 and the gain-free spelling."""
    torch.manual_seed(0)
    x = torch.randn(2, 4, 8, dtype=torch.float64)
    w = torch.randn(4, 8, dtype=torch.float64)
    lhs = x * torch.rsqrt(x.pow(2).mean((-2, -1), keepdim=True) + 1e-5)
    assert torch.equal(
        lhs * w, torch.nn.functional.rms_norm(x, (4, 8), w, 1e-5)
    )
    assert torch.equal(
        lhs, torch.nn.functional.rms_norm(x, (4, 8), None, 1e-5)
    )


# ---------------------------------------------------------------------------
#  Firing + the side-condition declines
# ---------------------------------------------------------------------------


def test_rms_norm_fold_fires_and_rhs_is_member():
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    eg, root, best = _saturate(
        _gained_lhs(u, w, (-1,), True, Const(1e-6)),
        [RMS_NORM_FOLD],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) > 0
    rhs = Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)
    assert list(eg.matches(rhs, eg.find(root)))
    assert best == rhs  # the fold is also the cheapest member


def test_rms_norm_fold_fires_on_int_dim_spelling():
    """A hand-minted ``mean(u², dim=-1)`` (int, not tuple) still folds."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    eg, root, _best = _saturate(
        _gained_lhs(u, w, -1, True, Const(1e-6)),
        [RMS_NORM_FOLD],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) > 0
    rhs = Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)
    assert list(eg.matches(rhs, eg.find(root)))


def test_rms_norm_fold_fires_on_multidim_reduce():
    """A trailing-block reduce over k=2 axes folds to ns = (d2, d3)."""
    u = Var("u", TensorType((2, 4, 8)))
    w = Param("w", TensorType((4, 8)))
    eg, root, _best = _saturate(
        _gained_lhs(u, w, (-2, -1), True, Const(1e-5)),
        [RMS_NORM_FOLD],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) > 0
    rhs = Op.make("rms_norm", u, w, dim=(4, 8), eps=1e-5)
    assert list(eg.matches(rhs, eg.find(root)))


def test_rms_norm_fold_nogain_fires_standalone_and_inside_gained():
    """The gain-free LHS is the gained one's inner ``mul``: a bare
    ``x·rms⁻¹`` folds, and inside ``x·rms⁻¹·w`` BOTH folds fire —
    ``mul(rms_norm(u), w)`` and ``rms_norm(u, w)`` share the class."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    eg, root, _best = _saturate(
        Op.make("mul", u, _rms_factor(u, (-1,), True, Const(1e-6))),
        [RMS_NORM_FOLD_NOGAIN],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold_nogain", 0) > 0
    rhs = Op.make("rms_norm", u, dim=(8,), eps=1e-6)
    assert list(eg.matches(rhs, eg.find(root)))

    eg, root, best = _saturate(
        _gained_lhs(u, w, (-1,), True, Const(1e-6)),
        [RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) > 0
    assert eg.rule_fires.get("rms_norm_fold_nogain", 0) > 0
    # the fused gained member wins the class
    assert best == Op.make("rms_norm", u, w, dim=(8,), eps=1e-6)


def test_rms_norm_fold_declines_on_no_keepdim():
    """``keepdim=False`` — the dropped axis cannot broadcast the rms
    back; the cond vetoes."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    eg, _root, _best = _saturate(
        _gained_lhs(u, w, (-1,), False, Const(1e-6)),
        [RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) == 0
    assert eg.rule_fires.get("rms_norm_fold_nogain", 0) == 0


def test_rms_norm_fold_declines_on_operand_mismatch():
    """``mul(mul(u, rms(v)), w)`` with u ≠ v — the shared ``u``
    metavariable vetoes, no check needed."""
    u = Var("u", TensorType((4, 8)))
    v = Var("v", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    eg, _root, _best = _saturate(
        _gained_lhs(u, w, (-1,), True, Const(1e-6), v=v),
        [RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) == 0
    assert eg.rule_fires.get("rms_norm_fold_nogain", 0) == 0


def test_rms_norm_fold_declines_on_non_trailing_dim():
    """``mean(dim=(0,))`` normalizes a leading axis — no rms_norm image."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((4,)))
    eg, _root, _best = _saturate(
        _gained_lhs(u, w, (0,), True, Const(1e-6)),
        [RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) == 0
    assert eg.rule_fires.get("rms_norm_fold_nogain", 0) == 0


def test_rms_norm_fold_declines_on_dup_and_out_of_range_dims():
    """Duplicate or out-of-range reduce axes are not a trailing block."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    for dims in ((-1, -1), (-3,), (2,)):
        eg, _root, _best = _saturate(
            _gained_lhs(u, w, dims, True, Const(1e-6)),
            [RMS_NORM_FOLD_NOGAIN],
            _cost_fn(),
        )
        assert eg.rule_fires.get("rms_norm_fold_nogain", 0) == 0, dims


def test_rms_norm_fold_declines_on_tensor_eps():
    """A Var/Param ``eps`` has no float-attr image — the cond vetoes."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    for eps in (Var("e", TensorType(())), Param("e", TensorType(()))):
        eg, _root, _best = _saturate(
            _gained_lhs(u, w, (-1,), True, eps),
            [RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN],
            _cost_fn(),
        )
        assert eg.rule_fires.get("rms_norm_fold", 0) == 0
        assert eg.rule_fires.get("rms_norm_fold_nogain", 0) == 0


def test_rms_norm_fold_declines_on_wrong_exponent():
    """``pow(u, 3)`` is a cube-mean, not rms — the cond vetoes."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    eg, _root, _best = _saturate(
        _gained_lhs(u, w, (-1,), True, Const(1e-6), exponent=Const(3)),
        [RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) == 0
    assert eg.rule_fires.get("rms_norm_fold_nogain", 0) == 0


def test_rms_norm_fold_fires_on_float_exponent_spelling():
    """``pow(u, 2.0)`` is still the square — ``const-cmp`` is numeric,
    not type-strict."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((8,)))
    eg, _root, _best = _saturate(
        _gained_lhs(u, w, (-1,), True, Const(1e-6), exponent=Const(2.0)),
        [RMS_NORM_FOLD],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) > 0


def test_rms_norm_fold_declines_on_wrong_weight_shape():
    """``w`` shaped other than ns mints a weight ``aten`` rejects — the
    gained fold's check vetoes; the gain-free twin still folds the
    inner mul."""
    u = Var("u", TensorType((4, 8)))
    w = Param("w", TensorType((4,)))
    eg, _root, _best = _saturate(
        _gained_lhs(u, w, (-1,), True, Const(1e-6)),
        [RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) == 0
    assert eg.rule_fires.get("rms_norm_fold_nogain", 0) > 0


def test_rms_norm_fold_declines_on_unshaped_input():
    """Unknown ``u`` shape → no provable normalized_shape → decline."""
    u = Var("u", TensorType((None, 8)))
    w = Param("w", TensorType((8,)))
    eg, _root, _best = _saturate(
        _gained_lhs(u, w, (-1,), True, Const(1e-6)),
        [RMS_NORM_FOLD, RMS_NORM_FOLD_NOGAIN],
        _cost_fn(),
    )
    assert eg.rule_fires.get("rms_norm_fold", 0) == 0
    assert eg.rule_fires.get("rms_norm_fold_nogain", 0) == 0


# ---------------------------------------------------------------------------
#  End-to-end on the real manual-RMSNorm exports
# ---------------------------------------------------------------------------


def test_rms_norm_fold_end_to_end_corpus_models():
    """The laws fire on the four real manual-RMSNorm exports, the
    extracted cost drops, the certificate replays, and the lowered
    before/after modules agree under ``sink.verify``."""
    torch.manual_seed(0)
    d = 16
    cases = [
        ("RMSNorm", RMSNorm(d), torch.randn(4, d, dtype=torch.float64)),
        (
            "NormLinear",
            NormLinear(d),
            torch.randn(4, d, dtype=torch.float64),
        ),
        (
            "TransformerBlock",
            TransformerBlock(d, 4, 2),
            torch.randn(2, 8, d, dtype=torch.float64),
        ),
        (
            "ParallelBlock",
            ParallelBlock(d, 4, 2),
            torch.randn(2, 8, d, dtype=torch.float64),
        ),
    ]
    folds = {"rms_norm_fold", "rms_norm_fold_nogain"}
    without = [r for r in DEFAULT if r.name not in folds]
    cost_fn = _cost_fn()
    for name, model, x in cases:
        model = model.eval().double()
        ir, leaves = export_to_ir(model, x)
        eg0, _r0, best0 = _saturate(ir.root, without, cost_fn)
        eg1, _r1, best1 = _saturate(ir.root, list(DEFAULT), cost_fn)

        assert eg1.rule_fires.get("rms_norm_fold", 0) > 0, name
        assert not (set(eg0.rule_fires) & folds), name
        c0, c1 = dag_cost(best0, cost_fn), dag_cost(best1, cost_fn)
        assert c1 < c0, (name, c0, c1)
        assert best1 != best0, name

        cert = eg1.certificate(ir.root, best1, cost_fn=cost_fn)
        verify_certificate(ir.root, cert)

        m0 = _lower_extracted(best0, _ir_of(best0, ir), leaves, _SINK)
        m1 = _lower_extracted(best1, _ir_of(best1, ir), leaves, _SINK)
        assert _SINK.verify(m0, m1, (x,), rtol=1e-4).passed, name


def test_rms_norm_fold_fires_in_default_pipeline():
    """The public default pipeline sees the laws fire and verifies."""
    torch.manual_seed(0)
    model = NormLinear(16).eval().double()
    x = torch.randn(4, 16, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(model, x, max_iterations=6, max_enodes=100_000)
    assert res.stats["rule_fires"].get("rms_norm_fold", 0) > 0
    low = opt.lower(res, x, verify=True)
    assert low.verified is not None and low.verified.passed
