"""Batched-state carrier lifts — the decode_flat_b8 falsification fix.

The diagonal/dense carrier lift used to fire only on unbatched
vector-state recurrences: ``_Spine``'s chain base had to be a bare
leaf (a ``reshape(h)``-flattened init ended the walk) and
``_consistent_shapes`` required ``(d,)`` states.  On ``(B, d)``/
``(B·d,)``-state decodes the lift declined and lowering stayed
``generic`` — measured ~2x slower than Inductor at B≥8.

What these tests prove:

* the applyd/apply carrier lift now accepts batched states —
  ``(B, d)`` diagonal states (shared ``(d,)`` or per-batch ``(B, d)``
  decays), flattened ``(B·d,)`` states under a ``reshape`` base, and
  dense column-vector states ``(*B, d, 1)`` with ``(*B, d, d)`` maps.
* the level-batched executor lowers them fp64-exactly vs the eager
  recurrence, on both ``(T, B, d)`` and ``(B, T, d)`` input layouts.
* a view-wrapped apply root (``reshape(applyd(…), (B, d))``) still
  routes to the batched executor — the module re-applies the views.
* the trace offer still declines non-flat signatures (F's block
  layout is written against flat widths).
* the strict declines still decline: non-broadcastable parts, dense
  maps that don't close under ``@``, computed inits.

Everything fp64: the reassociation is exact to ~1e-15.
"""

import math

import catopt.trace_lift as TL
import pytest
import torch
import torch.nn as nn
from catopt.egraph import EGraph
from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.optimize import optimize_model
from catopt.scan_lower import (
    build_scan_plan,
    is_scan_apply_term,
    to_batched_scan_module,
)
from catopt.torch_bridge import ir_to_torch_module
from catopt_core.typing import _matmul_shape


@pytest.fixture(autouse=True)
def _fp64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _P(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _eval(term, inputs, env, *xs):
    return ir_to_torch_module(
        IR(root=term, inputs=inputs), param_values=env
    )(*xs)


# ---------------------------------------------------------------------------
#  Term builders — batched recurrence spines
# ---------------------------------------------------------------------------


def _diag_batched_term(T: int, B: int, d: int, shared_a: bool = False):
    """``h_t[b] = a_t[b] ⊙ h_{t-1}[b] + x_t[b]`` over (T,B,d) input.

    ``shared_a=True`` is the retnet shape — one ``(d,)`` decay vector
    broadcast over the batch; False gives per-step ``(B, d)`` gates.
    """
    h0 = _v("h", B, d)
    x = _v("x", T, B, d)
    if shared_a:
        a = _P("pa", d)
        a_terms = [a] * T
    else:
        a = _P("pa", T, B, d)
        a_terms = [
            Op.make("select", a, dim=0, index=t) for t in range(T)
        ]
    h = h0
    for t in range(T):
        x_t = Op.make("select", x, dim=0, index=t)
        h = Op.make("add", Op.make("mul", a_terms[t], h), x_t)
    return h, [x, h0], {"pa": a}


def _flat_term(T: int, B: int, d: int):
    """The decode_flat shape: carried state flattened to (B·d,).

    ``hf = a ⊙ hf + x_t.reshape(B·d)`` with ``hf0 = reshape(h, (B·d,))``
    — the base is a *view* of the input leaf, and the module's output
    reshapes the flat state back to ``(B, d)``.
    """
    h0 = _v("h", B, d)
    x = _v("x", T, B, d)
    a = _P("pa", B * d)
    hf = Op.make("reshape", h0, shape=(B * d,))
    for t in range(T):
        u = Op.make(
            "reshape",
            Op.make("select", x, dim=0, index=t),
            shape=(B * d,),
        )
        hf = Op.make("add", Op.make("mul", a, hf), u)
    return (
        Op.make("reshape", hf, shape=(B, d)),
        [x, h0],
        {"pa": a},
    )


def _dense_column_term(T: int, B: int, d: int, shared_a: bool = False):
    """Dense batched: ``h_t = A_t · h_{t-1} + u_t`` on column states.

    States/inputs are ``(B, d, 1)``; maps are per-batch ``(B, d, d)``
    or one shared ``(d, d)`` (LTI) — both evaluate under the
    bindings' literal ``@``.
    """
    h0 = _v("h", B, d, 1)
    x = _v("x", T, B, d, 1)
    if shared_a:
        A = _P("pA", d, d)
        maps = [A] * T
    else:
        A = _P("pA", T, B, d, d)
        maps = [Op.make("select", A, dim=0, index=t) for t in range(T)]
    h = h0
    for t in range(T):
        u = Op.make("select", x, dim=0, index=t)
        h = Op.make("add", Op.make("matmul", maps[t], h), u)
    return h, [x, h0], {"pA": A}


def _env_diag(T, B, d, shared_a=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    env = {}
    if shared_a:
        env["pa"] = torch.rand(d, generator=g) * 0.9
    else:
        env["pa"] = torch.rand(T, B, d, generator=g) * 0.9
    return env


def _seq_eval_diag(env, T, B, d, shared_a, x, h0):
    """Eager reference for the batched diagonal recurrence."""
    h = h0
    a = env["pa"]
    for t in range(T):
        a_t = a if shared_a else a[t]
        h = a_t * h + x[t]
    return h


def _seq_eval_dense(env, T, B, d, shared_a, x, h0):
    h = h0
    A = env["pA"]
    for t in range(T):
        A_t = A if shared_a else A[t]
        h = A_t @ h + x[t]
    return h


def _applyd_members(eg, root):
    """apply/applyd enodes sitting in the root class."""
    cid = eg.find(root)
    return [
        n for n in eg._classes[cid].nodes if n.op in ("apply", "applyd")
    ]


# ---------------------------------------------------------------------------
#  typing._matmul_shape — the dense-family contract helper
# ---------------------------------------------------------------------------


class TestMatmulShape:
    def test_vector_cases(self):
        assert _matmul_shape((4, 4), (4,)) == (4,)
        assert _matmul_shape((4,), (4, 4)) == (4,)
        assert _matmul_shape((4,), (4,)) == ()
        assert _matmul_shape((4,), (5,)) is None
        assert _matmul_shape((3, 5), (4,)) is None
        assert _matmul_shape((5,), (3, 4)) is None

    def test_matrix_batch_cases(self):
        assert _matmul_shape((2, 3, 3), (2, 3, 1)) == (2, 3, 1)
        assert _matmul_shape((3, 3), (2, 3, 1)) == (2, 3, 1)
        assert _matmul_shape((4,), (2, 4, 5)) == (2, 5)
        # inner-dim / batch mismatch
        assert _matmul_shape((2, 3, 3), (2, 4, 1)) is None
        assert _matmul_shape((2, 3, 3), (5, 3, 3)) is None

    def test_invalid_operands(self):
        assert _matmul_shape(None, (4,)) is None
        assert _matmul_shape((4,), "x") is None
        assert _matmul_shape((), (4,)) is None
        assert _matmul_shape((4,), ()) is None
        # vector-matrix promotion: (k,) @ (…, k, n) -> (…, n)
        assert _matmul_shape((2,), (2, 4)) == (4,)
        assert _matmul_shape((4,), (2, 4)) is None

    def test_none_dims_are_wildcards(self):
        assert _matmul_shape((None, 3, 3), (None, 3, 1)) == (
            None,
            3,
            1,
        )
        assert _matmul_shape((3,), (None,)) == ()


# ---------------------------------------------------------------------------
#  The lift — batched spines
# ---------------------------------------------------------------------------


class TestBatchedLift:
    def test_batched_diag_lifts_applyd(self):
        """(T,B,d) spine with per-batch gates → applyd member."""
        T, B, d = 5, 3, 4
        term, _, _ = _diag_batched_term(T, B, d)
        eg = EGraph()
        root = eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts and all(lft["kind"] == "diag" for lft in lifts)
        assert _applyd_members(eg, root)

    def test_batched_diag_shared_decay(self):
        """retnet shape: a shared (d,) decay over (B,d) states."""
        T, B, d = 6, 4, 8
        term, inputs, _ = _diag_batched_term(T, B, d, shared_a=True)
        env = _env_diag(T, B, d, shared_a=True)
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts
        x = torch.randn(T, B, d)
        h0 = torch.randn(B, d)
        want = _seq_eval_diag(env, T, B, d, True, x, h0)
        for lft in lifts:
            got = _eval(lft["term"], inputs, env, x, h0)
            assert (got - want).abs().max().item() < 1e-13

    def test_flat_state_reshape_base(self):
        """decode_flat_b8 shape: reshape(h) init + reshape output."""
        T, B, d = 4, 2, 8
        term, inputs, _ = _flat_term(T, B, d)
        g = torch.Generator().manual_seed(1)
        env = {"pa": torch.rand(B * d, generator=g) * 0.9}
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts, "flat-state spine not recognised"
        x = torch.randn(T, B, d)
        h0 = torch.randn(B, d)
        hf = h0.reshape(B * d)
        for t in range(T):
            hf = env["pa"] * hf + x[t].reshape(B * d)
        want = hf.reshape(B, d)
        for lft in lifts:
            got = _eval(lft["term"], inputs, env, x, h0)
            assert (got - want.reshape(B * d)).abs().max() < 1e-13

    def test_flat_spine_still_gets_trace(self):
        """A (B·d,) flat state IS a flat signature — the trace offer
        still fires on it (block width B·d)."""
        T, B, d = 4, 2, 8
        term, _, _ = _flat_term(T, B, d)
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg)
        assert lifts and all(lft.d == B * d for lft in lifts)

    def test_batched_dense_column_state(self):
        """Dense ``matmul(A(B,d,d), h(B,d,1))`` spines lift to apply."""
        T, B, d = 4, 3, 4
        term, _, _ = _dense_column_term(T, B, d)
        eg = EGraph()
        root = eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts and all(lft["kind"] == "dense" for lft in lifts)
        assert _applyd_members(eg, root)

    def test_batched_dense_shared_map(self):
        """Shared (d,d) map over (B,d,1) column states (batched LTI)."""
        T, B, d = 4, 3, 4
        term, inputs, _ = _dense_column_term(T, B, d, shared_a=True)
        g = torch.Generator().manual_seed(2)
        env = {
            "pA": torch.randn(d, d, generator=g) / math.sqrt(d) * 0.4
        }
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts
        x = torch.randn(T, B, d, 1)
        h0 = torch.randn(B, d, 1)
        want = _seq_eval_dense(env, T, B, d, True, x, h0)
        for lft in lifts:
            got = _eval(lft["term"], inputs, env, x, h0)
            assert (got - want).abs().max().item() < 1e-12

    def test_batched_diag_trace_declines(self):
        """Batched (non-flat) plans get no trace offer — F's layout is
        written against flat usize only."""
        term, _, _ = _diag_batched_term(5, 3, 4)
        eg = EGraph()
        eg.add_term(term)
        assert TL.lift_scan_to_trace(eg) == []
        # … but the carrier lift still fires on the same class
        assert TL.lift_scan_to_applyd(eg)


# ---------------------------------------------------------------------------
#  Declines — the strictness that remains
# ---------------------------------------------------------------------------


class TestBatchedDeclines:
    def test_incompatible_state_shapes_decline(self):
        """h (B,d) state with a (d2,) input that can't broadcast."""
        T, B, d = 4, 2, 4
        h0 = _v("h", B, d)
        x = _v("x", T, B, d)
        a = _P("pa", T, B, d)
        h = h0
        for t in range(T):
            x_t = Op.make("select", x, dim=0, index=t)
            h = Op.make(
                "add",
                Op.make("mul", Op.make("select", a, dim=0, index=t), h),
                x_t,
            )
        # now corrupt: replace one map with an unbroadcastable shape
        bad = _P("bad", d + 1)
        h = Op.make(
            "add",
            Op.make("mul", bad, h),
            Op.make("select", x, dim=0, index=0),
        )
        eg = EGraph()
        eg.add_term(h)
        assert TL.lift_scan_to_applyd(eg, min_steps=5) == []

    def test_computed_init_still_declines(self):
        """A sigmoid(param) init is computed — not a base (even when
        view-wrapped: reshape(sigmoid(p)) is not a leaf)."""
        T, d = 4, 4
        x = _v("x", T, d)
        a = _P("pa", T, d)
        h = Op.make(
            "reshape",
            Op.make("sigmoid", _P("ph", d)),
            shape=(d,),
        )
        for t in range(T):
            x_t = Op.make("select", x, dim=0, index=t)
            a_t = Op.make("select", a, dim=0, index=t)
            h = Op.make("add", Op.make("mul", a_t, h), x_t)
        eg = EGraph()
        eg.add_term(h)
        assert TL.lift_scan_to_applyd(eg) == []
        assert TL.lift_scan_to_trace(eg) == []

    def test_dense_nonuniform_maps_decline(self):
        """Dense maps of mixed widths can't form one matmul family."""
        T, d = 3, 4
        h0 = _v("h", d)
        x = _v("x", T, d)
        h = h0
        for t in range(T):
            w = _P(f"w{t}", d, d) if t < 2 else _P("wbad", d, d + 1)
            u = Op.make("select", x, dim=0, index=t)
            # keep it well-typed: the bad step's matmul still lands (d,)
            h = (
                Op.make("add", Op.make("matmul", w, h), u)
                if t < 2
                else Op.make(
                    "add",
                    Op.make(
                        "select",
                        Op.make(
                            "matmul", w, Op.make("unsqueeze", h, dim=1)
                        ),
                        dim=-1,
                        index=0,
                    ),
                    u,
                )
            )
        eg = EGraph()
        root = eg.add_term(h)
        lifts = TL.lift_scan_to_applyd(eg)
        # the T=2 uniform prefix may lift — the full mixed-width chain
        # must not (the (d,d+1) map breaks the matmul family)
        assert all(
            eg.find(lft["root_eid"]) != eg.find(root) for lft in lifts
        )
        assert not _applyd_members(eg, root)

    def test_mixed_diag_dense_spine_declines(self):
        """A spine mixing mul and matmul steps is _MIXED — no plan."""
        T, d = 4, 4
        h0 = _v("h", d)
        x = _v("x", T, d)
        a = _P("pa", T, d)
        A = _P("pA", T, d, d)
        h = h0
        for t in range(T):
            u = Op.make("select", x, dim=0, index=t)
            prod = (
                Op.make("mul", Op.make("select", a, dim=0, index=t), h)
                if t % 2 == 0
                else Op.make(
                    "matmul", Op.make("select", A, dim=0, index=t), h
                )
            )
            h = Op.make("add", prod, u)
        eg = EGraph()
        eg.add_term(h)
        assert TL.lift_scan_to_applyd(eg) == []


# ---------------------------------------------------------------------------
#  The executor — batched leaves and wrapped roots
# ---------------------------------------------------------------------------


def _applyd_term_from_lift(eg, root):
    """Extract the offered applyd member as a term."""
    cid = eg.find(root)
    node = next(n for n in eg._classes[cid].nodes if n.op == "applyd")
    return eg.extract_best(root, lambda t: 0.0, overrides={cid: node})


class TestBatchedExecutor:
    def test_diag_batched_lowering_fp64(self):
        """(B,d) states, (T,B,d) inputs: level-batched equals eager."""
        T, B, d = 8, 4, 8
        term, inputs, _ = _diag_batched_term(T, B, d)
        env = _env_diag(T, B, d)
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts
        t = lifts[0]["term"]
        mod = to_batched_scan_module(
            IR(root=t, inputs=inputs), param_values=env
        )
        assert mod.is_batched
        x = torch.randn(T, B, d)
        h0 = torch.randn(B, d)
        want = _seq_eval_diag(env, T, B, d, False, x, h0)
        with torch.no_grad():
            got = mod(x, h0)
        assert (got - want).abs().max().item() < 1e-13
        assert got.shape == (B, d)

    def test_diag_shared_map_batched_state(self):
        """(d,) shared decay + (B,d) inputs → batched executor."""
        T, B, d = 8, 4, 8
        term, inputs, _ = _diag_batched_term(T, B, d, shared_a=True)
        env = _env_diag(T, B, d, shared_a=True)
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        mod = to_batched_scan_module(
            IR(root=lifts[0]["term"], inputs=inputs),
            param_values=env,
        )
        assert mod.is_batched
        x = torch.randn(T, B, d)
        h0 = torch.randn(B, d)
        want = _seq_eval_diag(env, T, B, d, True, x, h0)
        with torch.no_grad():
            got = mod(x, h0)
        assert (got - want).abs().max().item() < 1e-13

    def test_diag_BTd_input_layout(self):
        """(B,T,d) input: leaf b-parts are dim-1 selects — the
        index_select+movedim gather path still feeds (T,B,d) leaves."""
        T, B, d = 6, 3, 4
        h0 = _v("h", B, d)
        x = _v("x", B, T, d)
        a = _P("pa", d)
        h = h0
        for t in range(T):
            x_t = Op.make("select", x, dim=1, index=t)
            h = Op.make("add", Op.make("mul", a, h), x_t)
        eg = EGraph()
        eg.add_term(h)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts
        mod = to_batched_scan_module(
            IR(root=lifts[0]["term"], inputs=[x, h0]),
            param_values={"pa": torch.rand(d) * 0.9},
        )
        assert mod.is_batched
        xv = torch.randn(B, T, d)
        hv = torch.randn(B, d)
        av = mod.eval_mod._param_map["pa"]
        want = hv
        for t in range(T):
            want = av * want + xv[:, t]
        with torch.no_grad():
            got = mod(xv, hv)
        assert (got - want).abs().max().item() < 1e-13

    def test_dense_column_state_batched_lowering(self):
        """Dense (B,d,d) maps over (B,d,1) states lower level-batched."""
        T, B, d = 5, 3, 4
        term, inputs, _ = _dense_column_term(T, B, d)
        g = torch.Generator().manual_seed(3)
        env = {
            "pA": torch.randn(T, B, d, d, generator=g)
            / math.sqrt(d)
            * 0.4
        }
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts
        t = lifts[0]["term"]
        assert is_scan_apply_term(t)
        plan = build_scan_plan(t)
        assert plan["column_state"]
        mod = to_batched_scan_module(
            IR(root=t, inputs=inputs), param_values=env
        )
        assert mod.is_batched
        x = torch.randn(T, B, d, 1)
        h0 = torch.randn(B, d, 1)
        want = _seq_eval_dense(env, T, B, d, False, x, h0)
        with torch.no_grad():
            got = mod(x, h0)
        assert (got - want).abs().max().item() < 1e-12
        assert got.shape == (B, d, 1)

    def test_dense_shared_map_column_state(self):
        """Shared (d,d) map broadcast over (B,d,1) column states."""
        T, B, d = 4, 3, 4
        term, inputs, _ = _dense_column_term(T, B, d, shared_a=True)
        g = torch.Generator().manual_seed(4)
        env = {
            "pA": torch.randn(d, d, generator=g) / math.sqrt(d) * 0.4
        }
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        t = lifts[0]["term"]
        mod = to_batched_scan_module(
            IR(root=t, inputs=inputs), param_values=env
        )
        assert mod.is_batched
        x = torch.randn(T, B, d, 1)
        h0 = torch.randn(B, d, 1)
        want = _seq_eval_dense(env, T, B, d, True, x, h0)
        with torch.no_grad():
            got = mod(x, h0)
        assert (got - want).abs().max().item() < 1e-12

    def test_view_wrapped_root_lowers_batched(self):
        """``reshape(applyd(…), (B,d))`` routes to the batched executor
        and re-applies the reshape — the FlatChunk output shape."""
        T, B, d = 4, 2, 4
        term, inputs, _ = _flat_term(T, B, d)
        g = torch.Generator().manual_seed(5)
        env = {"pa": torch.rand(B * d, generator=g) * 0.9}
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts
        # build the extracted root: reshape(applyd(…), (B,d))
        inner_term = lifts[0]["term"]
        wrapped = Op.make("reshape", inner_term, shape=(B, d))
        assert is_scan_apply_term(wrapped)
        mod = to_batched_scan_module(
            IR(root=wrapped, inputs=inputs), param_values=env
        )
        assert mod.is_batched
        x = torch.randn(T, B, d)
        h0 = torch.randn(B, d)
        hf = h0.reshape(B * d)
        for t in range(T):
            hf = env["pa"] * hf + x[t].reshape(B * d)
        want = hf.reshape(B, d)
        with torch.no_grad():
            got = mod(x, h0)
        assert got.shape == (B, d)
        assert (got - want).abs().max().item() < 1e-13

    def test_mixed_map_widths_diag(self):
        """Per-leaf (d,) and (B,d) decays mixed — the executor
        broadcast-normalises before stacking."""
        T, B, d = 4, 3, 4
        h0 = _v("h", B, d)
        x = _v("x", T, B, d)
        a_shared = _P("pa_s", d)
        a_full = _P("pa_f", T // 2, B, d)
        h = h0
        for t in range(T):
            a_t = (
                a_shared
                if t % 2 == 0
                else Op.make("select", a_full, dim=0, index=t // 2)
            )
            x_t = Op.make("select", x, dim=0, index=t)
            h = Op.make("add", Op.make("mul", a_t, h), x_t)
        g = torch.Generator().manual_seed(6)
        env = {
            "pa_s": torch.rand(d, generator=g) * 0.9,
            "pa_f": torch.rand(T // 2, B, d, generator=g) * 0.9,
        }
        eg = EGraph()
        eg.add_term(h)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts
        mod = to_batched_scan_module(
            IR(root=lifts[0]["term"], inputs=[x, h0]),
            param_values=env,
        )
        assert mod.is_batched
        xv = torch.randn(T, B, d)
        hv = torch.randn(B, d)
        want = hv
        for t in range(T):
            a_t = env["pa_s"] if t % 2 == 0 else env["pa_f"][(t) // 2]
            want = a_t * want + xv[t]
        with torch.no_grad():
            got = mod(xv, hv)
        assert (got - want).abs().max().item() < 1e-13

    def test_fused_eager_batched_diag(self):
        """fused='eager' runs the adjacent-pair schedule on (n,B,d)."""
        T, B, d = 8, 4, 8
        term, inputs, _ = _diag_batched_term(T, B, d)
        env = _env_diag(T, B, d)
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        mod = to_batched_scan_module(
            IR(root=lifts[0]["term"], inputs=inputs),
            param_values=env,
            fused="eager",
        )
        assert mod.is_batched and mod._fused == "eager"
        x = torch.randn(T, B, d)
        h0 = torch.randn(B, d)
        want = _seq_eval_diag(env, T, B, d, False, x, h0)
        with torch.no_grad():
            got = mod(x, h0)
        assert (got - want).abs().max().item() < 1e-13

    def test_diag_shared_map_with_leading_ones(self):
        """A (1,d) shared decay broadcasts up to (B,d) — exercises the
        pad-free slot-expand path (s=(1,d) → tgt (B,d))."""
        T, B, d = 4, 3, 4
        h0 = _v("h", B, d)
        x = _v("x", T, B, d)
        a = _P("pa", 1, d)
        h = h0
        for t in range(T):
            x_t = Op.make("select", x, dim=0, index=t)
            # state-first mul keeps ``a`` as the map operand, so every
            # leaf shares it → leaf_a_shared expand path with a
            # leading-ones slot payload (1,d) → (B,d).
            h = Op.make("add", Op.make("mul", h, a), x_t)
        g = torch.Generator().manual_seed(8)
        env = {"pa": torch.rand(1, d, generator=g) * 0.9}
        eg = EGraph()
        eg.add_term(h)
        lifts = TL.lift_scan_to_applyd(eg)
        assert lifts
        mod = to_batched_scan_module(
            IR(root=lifts[0]["term"], inputs=[x, h0]),
            param_values=env,
        )
        assert mod.is_batched
        xv = torch.randn(T, B, d)
        hv = torch.randn(B, d)
        want = hv
        for t in range(T):
            want = env["pa"] * want + xv[t]
        with torch.no_grad():
            got = mod(xv, hv)
        assert (got - want).abs().max().item() < 1e-13

    def test_fused_declined_on_column_dense(self):
        """fused_dense_levels is (d+1)²-only — column-state plans keep
        the standard slot-gather schedule."""
        T, B, d = 4, 2, 4
        term, inputs, _ = _dense_column_term(T, B, d)
        g = torch.Generator().manual_seed(7)
        env = {
            "pA": torch.randn(T, B, d, d, generator=g)
            / math.sqrt(d)
            * 0.4
        }
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_applyd(eg)
        mod = to_batched_scan_module(
            IR(root=lifts[0]["term"], inputs=inputs),
            param_values=env,
            fused="eager",
        )
        assert mod.is_batched
        assert mod._fused is None  # declined
        x = torch.randn(T, B, d, 1)
        h0 = torch.randn(B, d, 1)
        want = _seq_eval_dense(env, T, B, d, False, x, h0)
        with torch.no_grad():
            got = mod(x, h0)
        assert (got - want).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  _consistent_shapes / _is_base — direct arm coverage
# ---------------------------------------------------------------------------


class TestConsistentShapes:
    def _eg_with(self, *terms):
        eg = EGraph()
        eids = [eg.add_term(t) for t in terms]
        return eg, eids

    def test_diag_broadcast_contract(self):
        eg, (m, i, h) = self._eg_with(
            _P("a", 4), _P("b", 3, 4), _P("h", 3, 4)
        )
        res = TL._consistent_shapes(eg, "diag", [m], [i], h)
        assert res == ((3, 4), False)

    def test_diag_flat_signature(self):
        eg, (m, i, h) = self._eg_with(
            _P("a", 4), _P("b", 4), _P("h", 4)
        )
        assert TL._consistent_shapes(eg, "diag", [m], [i], h) == (
            (4,),
            True,
        )

    def test_diag_rejects_nonconcrete_and_incompatible(self):
        eg, (m_none, i, h) = self._eg_with(
            _P("a", None, 4), _P("b", 3, 4), _P("h", 3, 4)
        )
        assert (
            TL._consistent_shapes(eg, "diag", [m_none], [i], h) is None
        )
        eg2, (m3, i3, h3) = self._eg_with(
            _P("a", 5), _P("b", 4), _P("h", 4)
        )
        assert (
            TL._consistent_shapes(eg2, "diag", [m3], [i3], h3) is None
        )
        # non-tuple / scalar state
        eg3, (m4, i4, h4) = self._eg_with(
            _P("a", 4),
            _P("b", 4),
            _P(
                "h",
            ),
        )
        assert (
            TL._consistent_shapes(eg3, "diag", [m4], [i4], h4) is None
        )

    def test_dense_family_rules(self):
        # classic flat signature
        eg, (m, i, h) = self._eg_with(
            _P("A", 4, 4), _P("b", 4), _P("h", 4)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) == (
            (4,),
            True,
        )
        # batched column signature
        eg, (m, i, h) = self._eg_with(
            _P("A", 3, 4, 4), _P("b", 3, 4, 1), _P("h", 3, 4, 1)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) == (
            (3, 4, 1),
            False,
        )
        # non-square map can't compose with itself
        eg, (m, i, h) = self._eg_with(
            _P("A", 3, 5), _P("b", 5), _P("h", 5)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) is None
        # map doesn't close over the input's broadcast join
        eg, (m, i, h) = self._eg_with(
            _P("A", 3, 4, 4), _P("b", 4, 1), _P("h", 4, 1)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) is None
        # map can't contract the state join at all
        eg, (m, i, h) = self._eg_with(
            _P("A", 4, 4), _P("b", 4), _P("h", 3, 4)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) is None
        # non-concrete map
        eg, (m, i, h) = self._eg_with(
            _P("A", None, 4, 4), _P("b", 3, 4, 1), _P("h", 3, 4, 1)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) is None
        # 1-D map — not a matrix family
        eg, (m, i, h) = self._eg_with(
            _P("A", 4), _P("b", 4), _P("h", 4)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) is None
        # a second map of a different shape
        eg, (m1, m2, i, h) = self._eg_with(
            _P("A", 4, 4), _P("B", 3, 4, 4), _P("b", 4), _P("h", 4)
        )
        assert (
            TL._consistent_shapes(eg, "dense", [m1, m2], [i], h) is None
        )
        # non-concrete input
        eg, (m, i, h) = self._eg_with(
            _P("A", 4, 4), _P("b", None, 4), _P("h", 4)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) is None
        # input broadcasts into S but the map can't contract it
        eg, (m, i, h) = self._eg_with(
            _P("A", 2, 4, 4), _P("b", 1), _P("h", 2, 4, 1)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) is None
        # input that can't even broadcast into S
        eg, (m, i, h) = self._eg_with(
            _P("A", 4, 4), _P("b", 5), _P("h", 4)
        )
        assert TL._consistent_shapes(eg, "dense", [m], [i], h) is None


class TestLeafShapeConsistency:
    """``build_scan_plan`` declines carrier roots whose leaf parts
    can't form one concrete family (``_leaf_shapes_consistent``)."""

    def _applyd(self, *leaves):
        tree = leaves[0]
        for lf in leaves[1:]:
            tree = Op.make("affd_compose", lf, tree)
        return Op.make("applyd", tree, _v("h", 4))

    def test_first_leaf_nonconcrete(self):
        t = self._applyd(
            Op.make("aff_diag", _P("a", None, 4), _P("b", 4)),
            Op.make("aff_diag", _P("a2", 4), _P("b2", 4)),
        )
        assert build_scan_plan(t) is None

    def test_later_leaf_nonconcrete(self):
        t = self._applyd(
            Op.make("aff_diag", _P("a", 4), _P("b", 4)),
            Op.make("aff_diag", _P("a2", 4), _P("b2", None, 4)),
        )
        assert build_scan_plan(t) is None

    def test_later_leaf_b_incompatible(self):
        """a parts agree, a later b can't broadcast into the family."""
        # in-order leaf list is [good, bad]: the (5,) b fails the
        # per-leaf b broadcast fold, not the first-leaf check.
        t = Op.make(
            "applyd",
            Op.make(
                "affd_compose",
                Op.make("aff_diag", _P("a", 4), _P("b", 4)),
                Op.make("aff_diag", _P("a2", 4), _P("b2", 5)),
            ),
            _v("h", 4),
        )
        assert build_scan_plan(t) is None

    def test_broadcastable_diag_leaves_accepted(self):
        t = self._applyd(
            Op.make("aff_diag", _P("a", 4), _P("b", 3, 4)),
            Op.make("aff_diag", _P("a2", 3, 4), _P("b2", 4)),
        )
        plan = build_scan_plan(t)
        assert plan is not None and plan["diagonal"]

    def test_dense_leaf_shape_declines(self):
        """Each dense arm: non-concrete a0, 1-D map, mm(A,A) != A,
        uncontractable b, and a non-vector/column b shape."""
        # non-concrete b0
        t = Op.make(
            "apply",
            Op.make(
                "aff_compose",
                Op.make("aff", _P("A", 4, 4), _P("b", None, 4)),
                Op.make("aff", _P("A2", 4, 4), _P("b2", 4)),
            ),
            _v("h", 4),
        )
        assert build_scan_plan(t) is None
        # A can't compose with itself (non-square)
        t = Op.make(
            "apply",
            Op.make(
                "aff_compose",
                Op.make("aff", _P("A", 4, 5), _P("b", 5)),
                Op.make("aff", _P("A2", 4, 5), _P("b2", 5)),
            ),
            _v("h", 5),
        )
        assert build_scan_plan(t) is None
        # A @ b invalid (b inner-dim mismatches)
        t = Op.make(
            "apply",
            Op.make(
                "aff_compose",
                Op.make("aff", _P("A", 4, 4), _P("b", 5)),
                Op.make("aff", _P("A2", 4, 4), _P("b2", 5)),
            ),
            _v("h", 5),
        )
        assert build_scan_plan(t) is None
        # b is neither vector nor column for the family
        t = Op.make(
            "apply",
            Op.make(
                "aff_compose",
                Op.make("aff", _P("A", 4, 4), _P("b", 4, 4)),
                Op.make("aff", _P("A2", 4, 4), _P("b2", 4, 4)),
            ),
            _v("h", 4, 4),
        )
        assert build_scan_plan(t) is None
        # mixed-map widths across leaves
        t = Op.make(
            "apply",
            Op.make(
                "aff_compose",
                Op.make("aff", _P("A", 4, 4), _P("b", 4)),
                Op.make("aff", _P("A2", 2, 4, 4), _P("b2", 4)),
            ),
            _v("h", 4),
        )
        assert build_scan_plan(t) is None

    def test_mixed_leaf_carrier_ops_decline(self):
        """aff vs aff_diag mixed leaves can't share one schedule."""
        t = Op.make(
            "applyd",
            Op.make(
                "affd_compose",
                Op.make("aff_diag", _P("a", 4), _P("b", 4)),
                Op.make("aff_diag", _P("a2", 4), _P("b2", 4)),
            ),
            _v("h", 4),
        )
        assert build_scan_plan(t) is not None
        # diagonal root over dense leaves — diag b (d,) vs a (d,d):
        # broadcast (d,d) is concrete → accepted, executor just
        # multiplies — but the h side check still applies
        t2 = Op.make(
            "apply",
            Op.make(
                "aff_compose",
                Op.make("aff", _P("A", 4, 4), _P("b", 4)),
                Op.make("aff_diag", _P("a", 4), _P("b2", 4)),
            ),
            _v("h", 4),
        )
        assert build_scan_plan(t2) is None


class TestChainBase:
    def test_view_wrapped_leaf_is_base(self):
        eg = EGraph()
        h = eg.add_term(_v("h", 2, 4))
        rh = eg.add_term(Op.make("reshape", _v("h2", 2, 4), shape=(8,)))
        # independent: point the reshape at the real leaf by union
        eg.union(h, rh)
        sp = TL._Spine(eg)
        assert sp._is_base(rh)
        # memoised second call
        assert sp._is_base(rh)

    def test_computed_is_not_base(self):
        eg = EGraph()
        s = eg.add_term(
            Op.make(
                "reshape",
                Op.make("sigmoid", _P("p", 8)),
                shape=(2, 4),
            )
        )
        assert not TL._Spine(eg)._is_base(s)

    def test_self_referential_class_is_not_base(self):
        """A class containing ``reshape(itself)`` can't bottom out."""
        eg = EGraph()
        a = eg.add_term(_P("p", 8))
        b = eg.add_term(Op.make("reshape", _P("q", 8), shape=(8,)))
        eg.union(a, b)
        cid = eg.find(a)
        # now the class holds {leaf, reshape(class)} — leaf wins
        assert TL._Spine(eg)._is_base(cid)
        # and a pure self-loop class declines: {sigmoid(p),
        # reshape(this class)} — the view can't bottom out.
        eg2 = EGraph()
        c = eg2.add_term(Op.make("sigmoid", _P("p", 8)))
        r = eg2.add_enode("reshape", (eg2.find(c),), {"shape": (8,)})
        eg2.union(r, c)
        cid2 = eg2.find(c)
        assert not TL._Spine(eg2)._is_base(cid2)

    def test_multiarg_view_node_is_not_base(self):
        """A _BASE_VIEW_OPS op with != 1 child isn't a look-through —
        the class's other members still decide."""
        eg = EGraph()
        a = eg.add_term(_P("p", 8))
        c = eg.add_term(Op.make("sigmoid", _P("q", 8)))
        eg.add_enode("reshape", (eg.find(c), eg.find(a)), {})
        cid = eg.find(c)
        # {sigmoid, reshape(c, a)} — the 2-child reshape doesn't count
        assert not TL._Spine(eg)._is_base(cid)


# ---------------------------------------------------------------------------
#  End-to-end — the decode_flat_b8-shaped cell through optimize_model
# ---------------------------------------------------------------------------


class _FlatChunkMod(nn.Module):
    """The decode_flat_b8 shape, locally: flat (B·d) carried state."""

    def __init__(self, d: int, B: int, C: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.B, self.d, self.C = B, d, C
        self.log_decay = nn.Parameter(
            torch.randn(d, generator=g) * 0.1 - 2.0
        )
        self.w = nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
        self.wr = nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, d = self.B, self.d
        hf = h.reshape(B * d)
        a = (
            torch.sigmoid(self.log_decay)
            .unsqueeze(0)
            .expand(B, d)
            .reshape(B * d)
        )
        for t in range(self.C):
            xt = x[t]
            u = torch.sigmoid(xt @ self.wr) * (xt @ self.w)
            hf = a * hf + u.reshape(B * d)
        return hf.reshape(B, d)


class _BatchedChunkMod(nn.Module):
    """(T,B,d) chunked retnet step — non-flat (B,d) carried state."""

    def __init__(self, d: int, C: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.d, self.C = d, C
        self.log_decay = nn.Parameter(
            torch.randn(d, generator=g) * 0.1 - 2.0
        )
        self.w = nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)
        self.wr = nn.Parameter(torch.randn(d, d, generator=g) * d**-0.5)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.log_decay)
        for t in range(self.C):
            xt = x[t]
            u = torch.sigmoid(xt @ self.wr) * (xt @ self.w)
            h = a * h + u
        return h


class TestEndToEnd:
    def test_flat_chunk_lowers_batched(self):
        """optimize_model on the decode_flat_b8-shaped cell."""
        B, d, C = 8, 16, 8
        m = _FlatChunkMod(d, B, C).eval()
        x = torch.randn(C, B, d)
        h = torch.randn(B, d)
        opt, stats = optimize_model(
            m, (x, h), verbose=False, max_iterations=32
        )
        assert stats["lowering"] == "batched"
        assert getattr(opt, "is_batched", False)
        with torch.no_grad():
            got, want = opt(x, h), m(x, h)
        rel = (got - want).abs().max() / want.abs().max()
        assert rel.item() < 1e-6

    def test_batched_chunk_lowers_batched(self):
        """optimize_model on the (T,B,d)-state chunked decode."""
        B, d, C = 4, 16, 8
        m = _BatchedChunkMod(d, C).eval()
        x = torch.randn(C, B, d)
        h = torch.randn(B, d)
        opt, stats = optimize_model(
            m, (x, h), verbose=False, max_iterations=32
        )
        assert stats["lowering"] == "batched"
        assert getattr(opt, "is_batched", False)
        with torch.no_grad():
            got, want = opt(x, h), m(x, h)
        rel = (got - want).abs().max() / want.abs().max()
        assert rel.item() < 1e-6

    def test_batched_chunk_fp64_decode(self):
        """fp64 whole-decode equivalence — the decode gate standard."""
        B, d, C, N = 8, 16, 8, 32
        m = _BatchedChunkMod(d, C).double().eval()
        x = torch.randn(N, B, d, dtype=torch.float64)
        h = torch.randn(B, d, dtype=torch.float64)
        opt, stats = optimize_model(
            m, (x[:C], h), verbose=False, max_iterations=32
        )
        assert stats["lowering"] == "batched"
        with torch.no_grad():
            href, hgot = h, h
            for t0 in range(0, N, C):
                href = m(x[t0 : t0 + C], href)
                hgot = opt(x[t0 : t0 + C], hgot)
        rel = (hgot - href).abs().max() / href.abs().max()
        assert rel.item() < 1e-13
