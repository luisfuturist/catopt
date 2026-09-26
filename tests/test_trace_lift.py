# ruff: noqa: RUF002, RUF003
"""trace_lift — the trm↔apply bridge: lifting unrolled recurrences into
the traced-monoidal representation of catopt/trace.py.

The gap documented in test_trace_pipeline.py: exported/unrolled
recurrences contain no ``trace`` term, so ``TRACE_LAWS`` could never
operate on them.  ``lift_scan_to_trace`` is the non-local pass that
recognises the whole recurrence horizon and constructs the
time-extended nilpotent block-shift matrix — the same diagram-level
intervention as ``pair_shared_input_linears``.

What these tests prove:

* e-graph insertion — the offered member lands in the recurrence's
  e-class (union recorded as e-graph-dependent, like any non-local
  pass; ``verify_certificate`` replays it as a trusted step).
* fp64-exactness — the lifted ``reshape(matmul(trace(F, T·d), v))``
  equals the unrolled recurrence on raw ``add(mul|matmul …)`` spines,
  on ``apply``/``applyd`` carrier trees, and on a real
  ``torch.export``ed DiagonalSSM — both joint and channel-split forms.
* post-lift reachability — on genuinely lifted members (not seeded
  traces), ``tr_superpose`` produces the ``bdiag`` of channel traces
  and ``tr_expand`` exposes the resolvent ``(I−S)⁻¹`` form.
* graceful no-op on non-recurrence graphs.

Everything is fp64 with tight tolerance: the fixpoint is solved, not
iterated, so the only error is float reassociation.
"""

import math

import pytest
import torch

import catopt.trace as cat_trace  # registers torch bindings
import catopt.trace_lift as TL
from catopt_core import laws as R
from catopt.cost import (
    _INVALID_COST,
    dag_cost,
    depth_cost,
    flops_cost,
    roofline_cost,
)
from catopt.egraph import EGraph, verify_certificate
from catopt.ir import IR, Op, Param, TensorType, Var, op_repr
from catopt.models.ssm import DiagDenseSSM, DiagonalSSM
from catopt.torch_bridge import export_to_ir, ir_to_torch_module

TRACE_LAWS = cat_trace.TRACE_LAWS


@pytest.fixture(autouse=True)
def _fp64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# ---------------------------------------------------------------------------
#  Term builders — unrolled recurrences over Var/Param leaves
# ---------------------------------------------------------------------------


def _diag_term(T: int, d: int):
    """h_t = a_t ⊙ h_{t−1} + b_t ⊙ x_t — the Mamba-faithful spine."""
    a = Param("pa", TensorType((T, d)))
    b = Param("pb", TensorType((T, d)))
    x = Var("x", TensorType((T, d)))
    h0 = Param("h0", TensorType((d,)))
    h = h0
    for t in range(T):
        a_t = Op.make("select", a, arg1=0, arg2=t)
        in_t = Op.make(
            "mul",
            Op.make("select", b, arg1=0, arg2=t),
            Op.make("select", x, arg1=0, arg2=t),
        )
        h = Op.make("add", Op.make("mul", a_t, h), in_t)
    return h, [x], {"pa": a, "pb": b, "h0": h0}


def _dense_term(T: int, d: int):
    """h_t = A_t·h_{t−1} + b_t ⊙ x_t — the selective-SSM spine."""
    A = Param("pA", TensorType((T, d, d)))
    b = Param("pb", TensorType((T, d)))
    x = Var("x", TensorType((T, d)))
    h0 = Param("h0", TensorType((d,)))
    h = h0
    for t in range(T):
        A_t = Op.make("select", A, arg1=0, arg2=t)
        in_t = Op.make(
            "mul",
            Op.make("select", b, arg1=0, arg2=t),
            Op.make("select", x, arg1=0, arg2=t),
        )
        h = Op.make("add", Op.make("matmul", A_t, h), in_t)
    return h, [x], {"pA": A, "pb": b, "h0": h0}


def _env(T: int, d: int, seed: int = 0, dense: bool = False):
    g = torch.Generator().manual_seed(seed)
    env = {}
    if dense:
        env["pA"] = (
            torch.randn(T, d, d, generator=g) / math.sqrt(d) * 0.5
        )
    else:
        env["pa"] = torch.rand(T, d, generator=g) * 0.9
    env["pb"] = torch.randn(T, d, generator=g)
    env["h0"] = torch.randn(d, generator=g)
    x = torch.randn(T, d, generator=g)
    return env, x


def _eval(term, inputs, env, x):
    return ir_to_torch_module(
        IR(root=term, inputs=inputs), param_values=env
    )(x)


def _applyd_term(T: int, d: int):
    """The same diagonal recurrence as an applyd(affd-tree, h0) carrier
    — what SCAN_DIAG_LAWS saturation produces.  Right-nested compose
    order: affd_compose(f_T, …(affd_compose(f_2, f_1)))."""
    a = Param("pa", TensorType((T, d)))
    b = Param("pb", TensorType((T, d)))
    x = Var("x", TensorType((T, d)))
    h0 = Param("h0", TensorType((d,)))
    leaves = []
    for t in range(T):
        a_t = Op.make("select", a, arg1=0, arg2=t)
        in_t = Op.make(
            "mul",
            Op.make("select", b, arg1=0, arg2=t),
            Op.make("select", x, arg1=0, arg2=t),
        )
        leaves.append(Op.make("aff_diag", a_t, in_t))
    f = leaves[-1]
    for leaf in reversed(leaves[:-1]):
        f = Op.make("affd_compose", f, leaf)
    return Op.make("applyd", f, h0), [x], {"pa": a, "pb": b, "h0": h0}


# ---------------------------------------------------------------------------
#  (a) e-graph insertion
# ---------------------------------------------------------------------------


class TestInsertion:
    def test_raw_diag_spine_gets_trace_member(self):
        T, d = 5, 4
        term, _, _ = _diag_term(T, d)
        eg = EGraph()
        root = eg.add_term(term)
        n0 = eg.n_enodes
        lifts = TL.lift_scan_to_trace(eg)
        # joint trace + the auto 2-way channel split
        assert len(lifts) == 2
        joint = next(lft for lft in lifts if lft.split is None)
        split = next(lft for lft in lifts if lft.split is not None)
        assert joint.T == T and joint.d == d and joint.kind == "diag"
        assert split.split == (2, 2)
        assert eg.n_enodes > n0
        # the offered member sits in the recurrence's own e-class
        assert eg.find(joint.out_eid) == eg.find(root)
        assert eg.find(split.out_eid) == eg.find(root)
        # … and is recognisable as the lifted fixpoint shape
        pat = Op.make(
            "reshape",
            Op.make("matmul", Op.make("trace", "f", usize="U"), "v"),
            shape="S",
        )
        assert eg.matches(pat, root)

    def test_certificate_records_nonlocal_step(self):
        T, d = 4, 4
        term, _, _ = _diag_term(T, d)
        eg = EGraph()
        root = eg.add_term(term)
        lifts = TL.lift_scan_to_trace(
            eg, root_eid=root, channel_splits=None
        )
        assert len(lifts) == 1
        cert = eg.certificate(term, lifts[0].term, root_eid=root)
        assert cert is not None
        # the lift carries a pointwise witness rule — fully replayable
        assert cert.n_egraph_dependent == 0
        assert verify_certificate(term, cert, strict=True) is not None
        assert op_repr(verify_certificate(term, cert)) == op_repr(
            lifts[0].term
        )

    def test_idempotent(self):
        term, _, _ = _diag_term(4, 4)
        eg = EGraph()
        _root = eg.add_term(term)
        TL.lift_scan_to_trace(eg)
        n1 = eg.n_enodes
        lifts2 = TL.lift_scan_to_trace(eg)
        assert lifts2 and eg.n_enodes == n1  # same hash-consed enodes

    def test_lift_only_root(self):
        """maximal_only drops strict-prefix chains."""
        T, d = 5, 4
        term, _, _ = _diag_term(T, d)
        eg = EGraph()
        root = eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg, channel_splits=None)
        assert len(lifts) == 1
        assert lifts[0].T == T
        assert eg.find(lifts[0].root_eid) == eg.find(root)


# ---------------------------------------------------------------------------
#  (b) fp64 numerical equivalence
# ---------------------------------------------------------------------------


class TestNumerics:
    def test_diag_spine_fp64(self):
        T, d = 8, 6
        term, inputs, _ = _diag_term(T, d)
        env, x = _env(T, d, seed=1)
        eg = EGraph()
        _root = eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg)
        assert {lft.split for lft in lifts} == {None, (3, 3)}
        want = _eval(term, inputs, env, x)
        for lft in lifts:
            got = _eval(lft.term, inputs, env, x)
            err = (got - want).abs().max().item()
            assert err < 1e-11

    def test_dense_spine_fp64(self):
        T, d = 7, 5
        term, inputs, _ = _dense_term(T, d)
        env, x = _env(T, d, seed=2, dense=True)
        eg = EGraph()
        _root = eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg)
        # dense transitions couple channels — no split offered
        assert len(lifts) == 1 and lifts[0].kind == "dense"
        assert lifts[0].split is None
        got = _eval(lifts[0].term, inputs, env, x)
        want = _eval(term, inputs, env, x)
        assert (got - want).abs().max().item() < 1e-11

    def test_applyd_carrier_fp64(self):
        T, d = 6, 8
        term, inputs, _ = _applyd_term(T, d)
        env, x = _env(T, d, seed=3)
        eg = EGraph()
        _root = eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg)
        assert len(lifts) == 2
        want = _eval(term, inputs, env, x)
        for lft in lifts:
            got = _eval(lft.term, inputs, env, x)
            assert (got - want).abs().max().item() < 1e-11

    def test_carrier_lift_after_scan_saturation(self):
        """The realistic path: SCAN_DIAG_LAWS discovers the carrier
        inside a raw spine, then the lift sees it via the class's
        applyd member."""
        T, d = 6, 8
        term, inputs, _ = _diag_term(T, d)
        env, x = _env(T, d, seed=4)
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=8)
        lifts = TL.lift_scan_to_trace(eg, root_eid=root)
        assert lifts
        want = _eval(term, inputs, env, x)
        for lft in lifts:
            got = _eval(lft.term, inputs, env, x)
            assert (got - want).abs().max().item() < 1e-11

    def test_exported_diagonal_ssm_fp64(self):
        """End-to-end on a real torch.export spine — no saturation."""
        torch.manual_seed(0)
        T, d = 8, 8
        m = DiagonalSSM(d_inner=d, d_in=d, steps=T).double().eval()
        x = torch.randn(T, d)
        ir, src_tensors = export_to_ir(m, x)
        eg = EGraph()
        root = eg.add_term(ir.root)
        lifts = TL.lift_scan_to_trace(eg, root_eid=root)
        assert lifts, "exported mul-spine not recognised"
        want = m(x)
        for lft in lifts:
            got = ir_to_torch_module(
                IR(root=lft.term, inputs=ir.inputs),
                param_values=src_tensors,
            )(x)
            assert (got - want).abs().max().item() < 1e-9

    def test_exported_dense_ssm_fp64(self):
        torch.manual_seed(0)
        T, d = 6, 8
        m = DiagDenseSSM(d_inner=d, d_in=d, steps=T).double().eval()
        x = torch.randn(T, d)
        ir, src_tensors = export_to_ir(m, x)
        eg = EGraph()
        root = eg.add_term(ir.root)
        lifts = TL.lift_scan_to_trace(eg, root_eid=root)
        assert lifts and all(lft.kind == "dense" for lft in lifts)
        want = m(x)
        for lft in lifts:
            got = ir_to_torch_module(
                IR(root=lft.term, inputs=ir.inputs),
                param_values=src_tensors,
            )(x)
            assert (got - want).abs().max().item() < 1e-9


# ---------------------------------------------------------------------------
#  (c) post-lift saturation — trace laws on genuinely lifted members
# ---------------------------------------------------------------------------


class TestPostLiftSaturation:
    def test_superpose_exposes_bdiag_of_channel_traces(self):
        """d=8 diagonal recurrence → auto-split parl trace →
        tr_superpose → bdiag(trace(F1), trace(F2)) member."""
        T, d = 4, 8
        term, inputs, _ = _diag_term(T, d)
        env, x = _env(T, d, seed=5)
        eg = EGraph()
        root = eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg)
        split = [lft for lft in lifts if lft.split == (4, 4)]
        assert split, "channel-split parl member not offered"
        eg.run(TRACE_LAWS, root, max_iterations=8)
        assert eg.rule_fires.get("tr_superpose", 0) >= 1

        # a bdiag(trace, trace) enode must now be reachable under the
        # root class's reshape→matmul member
        pat = Op.make(
            "reshape",
            Op.make(
                "matmul",
                Op.make(
                    "bdiag",
                    Op.make("trace", "f1", usize="U1"),
                    Op.make("trace", "f2", usize="U2"),
                ),
                "v",
            ),
            shape="S",
        )
        binds = eg.matches(pat, root)
        assert binds, "bdiag-of-traces not reachable after saturation"

        # extract that member and check it numerically
        tr_cid = eg.find(split[0].trace_eid)
        bdiag_node = None
        for node in eg.get_class(tr_cid).nodes:
            if node.op == "bdiag":
                bdiag_node = node
                break
        assert bdiag_node is not None
        term2 = eg.extract_best(
            root, flops_cost, overrides={tr_cid: bdiag_node}
        )
        got = _eval(term2, inputs, env, x)
        want = _eval(term, inputs, env, x)
        assert (got - want).abs().max().item() < 1e-11

    def test_expand_exposes_resolvent(self):
        """tr_expand on the lifted joint trace → the closed-form
        P + Q(I−S)⁻¹R member — inv/sub/eye enodes appear."""
        T, d = 4, 4
        term, inputs, _ = _diag_term(T, d)
        env, x = _env(T, d, seed=6)
        eg = EGraph()
        root = eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg, channel_splits=None)
        tr_cid = eg.find(lifts[0].trace_eid)
        eg.run(TRACE_LAWS, root, max_iterations=8)
        assert eg.rule_fires.get("tr_expand", 0) >= 1
        resolvent = None
        for node in eg.get_class(tr_cid).nodes:
            if node.op == "add":  # P + Q(I−S)⁻¹R
                resolvent = node
                break
        assert resolvent is not None, "resolvent member absent"
        term2 = eg.extract_best(
            root, flops_cost, overrides={tr_cid: resolvent}
        )
        got = _eval(term2, inputs, env, x)
        want = _eval(term, inputs, env, x)
        assert (got - want).abs().max().item() < 1e-10

    def test_vanish_split_then_expand(self):
        """The tuple-usize joint trace splits into nested traces which
        each expand — the vanishing/closed-form chain is reachable."""
        T, d = 3, 4
        term, _inputs, _ = _diag_term(T, d)
        _envv, _x = _env(T, d, seed=7)
        eg = EGraph()
        root = eg.add_term(term)
        _lifts = TL.lift_scan_to_trace(eg)  # split member has
        eg.run(
            TRACE_LAWS,
            root,  # usize=(Td1, Td2)
            max_iterations=10,
        )
        fires = eg.rule_fires
        assert fires.get("tr_superpose", 0) >= 1
        assert fires.get("tr_vanish_split", 0) >= 1
        assert fires.get("tr_expand", 0) >= 1

    def test_costs_are_sane(self):
        """All cost models see the lifted member as a finite program."""
        T, d = 6, 8
        term, _, _ = _diag_term(T, d)
        eg = EGraph()
        _root = eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg)
        for lft in lifts:
            for cost in (flops_cost, depth_cost, roofline_cost):
                c = dag_cost(lft.term, cost)
                assert 0 < c < _INVALID_COST


# ---------------------------------------------------------------------------
#  (d) minted-F storage bound — one F per horizon, not one per step
# ---------------------------------------------------------------------------


class TestStorageBound:
    def test_saturated_prefixes_do_not_multiply_F(self):
        """maximal_only must drop carrier-path prefix offers.

        After SCAN_DIAG_LAWS saturation every prefix class h_2…h_T
        carries an applyd member, hence a plan.  Carrier plans record
        no step_states, so without structural prefix detection each of
        them mints its own block-matrix F: ~2T offers (joint + split
        per prefix) instead of the ~2 for the whole horizon — the
        ~2T× storage blow-up the raw-spine path's step_states already
        prevented.
        """
        T, d = 14, 4
        term, _, _ = _diag_term(T, d)
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=6)
        n0 = eg.n_enodes
        lifts = TL.lift_scan_to_trace(eg)
        # one joint F + one channel-split F — not ~2T prefix copies
        assert 1 <= len(lifts) <= 2
        assert max(lft.T for lft in lifts) == T
        minted = eg.n_enodes - n0
        # the two offered members are O(T) structure each — far below
        # the ~2T Fs a per-prefix mint produced (~4·T² enodes).
        assert minted < 40 * T

    def test_maximal_only_false_still_offers_prefixes(self):
        """The bound is maximal_only's documented semantics — opting
        out restores per-prefix offers, so the guard is behavioural,
        not a recognition failure."""
        T, d = 10, 4
        term, _, _ = _diag_term(T, d)
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=6)
        lifts = TL.lift_scan_to_trace(eg, maximal_only=False)
        assert len(lifts) > T  # per-prefix minting returns
        lifts_max = TL.lift_scan_to_trace(eg)
        assert len(lifts_max) <= 2

    def test_lifted_param_storage_parity(self):
        """The lifted member stores the same leaves as the loop body —
        no per-step F param materialises in the weights file."""
        from catopt.cost import param_bytes_cost

        T, d = 10, 6
        term, _, _ = _diag_term(T, d)
        eg = EGraph()
        eg.add_term(term)
        lifts = TL.lift_scan_to_trace(eg)
        orig = param_bytes_cost(term)
        for lft in lifts:
            got = param_bytes_cost(lft.term)
            assert got <= 2 * orig  # ~parity, never ~2T×
            assert got == orig  # same named leaves, in fact


# ---------------------------------------------------------------------------
#  (e) graceful no-op
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_non_recurrence_is_untouched(self):
        x = Var("x", TensorType((8,)))
        w = Param("w", TensorType((8, 8)))
        v = Param("v", TensorType((8,)))
        term = Op.make(
            "tanh", Op.make("add", Op.make("matmul", w, x), v)
        )
        eg = EGraph()
        _root = eg.add_term(term)
        n0 = eg.n_enodes
        lifts = TL.lift_scan_to_trace(eg)
        assert lifts == []
        assert eg.n_enodes == n0

    def test_two_node_add_is_not_a_recurrence(self):
        """add(mul(a, x), b) with all-leaf factors: nothing chains, and
        the T=1 degenerate lift is below min_steps."""
        a = Param("a", TensorType((4,)))
        x = Var("x", TensorType((4,)))
        b = Param("b", TensorType((4,)))
        term = Op.make("add", Op.make("mul", a, x), b)
        eg = EGraph()
        eg.add_term(term)
        n0 = eg.n_enodes
        assert TL.lift_scan_to_trace(eg) == []
        assert eg.n_enodes == n0

    def test_short_spine_below_min_steps(self):
        term, _, _ = _diag_term(1, 4)
        eg = EGraph()
        eg.add_term(term)
        n0 = eg.n_enodes
        assert TL.lift_scan_to_trace(eg) == []
        assert eg.n_enodes == n0
