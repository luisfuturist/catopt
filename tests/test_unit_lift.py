"""affd_lift_unit — the a≡1 degenerate case: pure additive accumulation.

``SCAN_DIAG_LAWS`` previously lifted only recurrence steps carrying an
explicit multiplicative factor, ``add(mul(a, h), x)``.  A cumsum — the
recurrence ``h_t = h_{t-1} + x_t`` (running sums, running statistics,
linear attention's KV state ``S_t = S_{t-1} + k_t v_tᵀ``) — has no
``mul`` enode, so the scan monoid never saw it.

``affd_lift_unit`` introduces the unit diagonal:

    add(h, x)  →  applyd(aff_diag(1, x), h)

where ``1`` is spelled ``expand(Const(1.0), shape=S)`` with ``S`` the
add's broadcast shape — NOT a bare scalar.  ``scan_lower``'s
``_leaf_shapes_consistent`` requires every ``aff_diag`` leaf's a-part
to match the state's ``(d,)`` shape (the batched executor stacks leaf
a-parts); a scalar-shaped unit would silently bar the whole carrier
tree from ``BatchedScanModule`` and from ``trace_lift``'s carrier path.

Findings encoded as tests:

* On a hand-built cumsum spine the unit lift fires once per step; the
  step variant composes the carried segment, ``affd_assoc(_rev)``
  rebalances the compose spine, and the min-depth extraction is a
  balanced ``applyd(affd_compose-tree, h0)`` — depth ~log T vs ~T for
  the raw chain — that is fp64-exact against the unrolled module.

* On an exported CumsumSSM (``h_t = h_{t-1} + B x_t``) the same
  happens end-to-end: depth 18 → 7 for T=16, and the extracted term
  satisfies ``is_scan_apply_term`` and lowers to a working
  ``BatchedScanModule`` (fp64 diff ~1e-15).

* ``trace_lift.lift_scan_to_trace`` reads the carrier directly —
  unit-decay diagonals are valid trace plans, and both lift kinds
  (joint + channel split) verify fp64-exact.

* ``_affd_unit_state_like`` suppresses sideways firings: an add whose
  operands are per-step terms (matmul/mul/select) never lifts —
  measured by zero ``affd_lift_unit*`` fires — while a leaf operand
  (the h0 Param) is still admitted as state.
"""

import math

import pytest
import torch
import torch.nn as nn

import catopt.trace  # noqa: F401 — registers trace/eye/parl bindings
from catopt import meta
from catopt_core import laws as R
from catopt import trace_lift as TL
from catopt.egraph import EGraph
from catopt.ir import IR, Const, Op, Param, TensorType, Var, op_repr
from catopt.scan_lower import is_scan_apply_term, to_batched_scan_module
from catopt.torch_bridge import (
    _IR_TO_TORCH,
    export_to_ir,
    ir_to_torch_module,
)


@pytest.fixture(autouse=True)
def _fp64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


def _opdepth(t, memo):
    """Critical-path depth of a term (shared-subterm DAG memoised)."""
    if not isinstance(t, Op):
        return 0
    k = id(t)
    if k not in memo:
        memo[k] = 1 + max(
            (_opdepth(a, memo) for a in t.args), default=0
        )
    return memo[k]


class CumsumSSM(nn.Module):
    """Pure additive accumulation: ``h_t = h_{t-1} + B x_t``.

    Exports as a bare ``add`` spine — no ``mul`` enode for
    ``AFFD_LIFT`` to see.  Before ``affd_lift_unit`` this produced zero
    carrier enodes under ``SCAN_DIAG_LAWS``.
    """

    def __init__(
        self, d_inner: int = 16, d_in: int = 16, steps: int = 16
    ) -> None:
        super().__init__()
        self.B_proj = nn.Linear(d_in, d_inner, bias=False)
        self.h0 = nn.Parameter(torch.zeros(d_inner))
        self.steps = steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.B_proj(x)
        h = self.h0
        for t in range(self.steps):
            h = h + b[t]
        return h


def _cumsum_term(T: int, d: int):
    """Hand-built ``add(add(...(add(h0, x0), x1), ...), x{T-1})`` spine
    over select slices — the shape the exporter emits for CumsumSSM."""
    x = Var("x", TensorType((T, d)))
    h0 = Param("h0", TensorType((d,)))
    h = h0
    for t in range(T):
        h = Op.make("add", h, Op.make("select", x, dim=0, index=t))
    return h, [x]


_UNIT_RULES = (
    "affd_lift_unit",
    "affd_lift_unit_post",
    "affd_lift_unit_step",
    "affd_lift_unit_step_post",
)


def _fires(eg, prefix="affd_lift_unit"):
    return {
        k: v for k, v in eg.rule_fires.items() if k.startswith(prefix)
    }


# ---------------------------------------------------------------------------
#  (a) The unit coefficient is an all-ones diagonal at the step's shape
# ---------------------------------------------------------------------------


def test_unit_diagonal_is_ones_of_the_state_shape():
    """``expand(Const(1.0), shape=S)`` materialises 1_S; the lift writes
    applyd(aff_diag(1_S, x), h) == h + x exactly, and ``scan_lower``
    sees a ``(d,)``-shaped a-part just like a multiplicative step."""
    d = 8
    expand = _IR_TO_TORCH["expand"]
    aff_diag = _IR_TO_TORCH["aff_diag"]
    applyd = _IR_TO_TORCH["applyd"]

    unit = expand(torch.tensor(1.0, dtype=torch.float64), shape=(d,))
    assert unit.shape == (d,)
    assert torch.equal(unit, torch.ones(d, dtype=torch.float64))

    a, b, h = (
        unit,
        torch.randn(d, dtype=torch.float64),
        torch.randn(d, dtype=torch.float64),
    )
    assert torch.equal(applyd(aff_diag(a, b), h), h + b)


def test_unit_lift_fires_on_cumsum_spine():
    """add(add(add(h0, x1), x2), x3) saturates into an applyd/aff_diag
    carrier; the unit coefficient lands as ``expand(1.0, shape=(d,))``
    and the extracted scan is fp64-exact."""
    torch.manual_seed(0)
    T, d = 8, 4
    term, inputs = _cumsum_term(T, d)
    _x = inputs[0]
    ir = IR(root=term, inputs=inputs)

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=8)

    fires = _fires(eg)
    assert fires["affd_lift_unit"] == T  # one per step
    assert fires["affd_lift_unit_step"] > 0  # composed chains

    ops = {n.op for c in eg._classes.values() for n in c.nodes}
    assert {"applyd", "aff_diag", "affd_compose", "expand"} <= ops

    best = eg.extract_min_depth(root)
    rep = op_repr(best)
    assert "applyd" in rep and "affd_compose" in rep
    # the unit diagonal materialises at the state shape, not as a
    # bare scalar — scan_lower's leaf-shape consistency needs (d,)
    assert "(expand 1.0, shape=(4,))" in rep
    assert _opdepth(best, {}) <= 4 * math.ceil(math.log2(T)) + 8

    st = {"h0": torch.randn(d, dtype=torch.float64)}
    xt = torch.randn(T, d, dtype=torch.float64)
    ref = ir_to_torch_module(ir, param_values=st)(xt)
    out = ir_to_torch_module(
        IR(root=best, inputs=inputs), param_values=st
    )(xt)
    assert (out - ref).abs().max().item() < 1e-10


# ---------------------------------------------------------------------------
#  (b) End-to-end: an exported accumulator reaches the balanced scan
# ---------------------------------------------------------------------------


def test_cumsum_ssm_reaches_log_depth_scan():
    """CumsumSSM exports a bare add spine; with the unit lift the
    e-graph finds the balanced applyd/affd_compose scan, ~log T depth,
    fp64-exact against the module."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = CumsumSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st = export_to_ir(m, x)

    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=14, max_nodes=400_000)

    best = eg.extract_min_depth(root)
    rep = op_repr(best)
    assert "applyd" in rep and "affd_compose" in rep
    assert "aff_diag" in rep and "expand" in rep
    assert is_scan_apply_term(best)
    assert _opdepth(best, {}) <= 4 * math.ceil(math.log2(T)) + 8

    opt_ir = IR(
        root=best,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    mod = ir_to_torch_module(opt_ir, param_values=st)
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10


def test_batched_scan_executes_unit_accumulation():
    """The unit-lifted carrier lowers to BatchedScanModule — every
    aff_diag leaf reports a consistent (d,) a-shape, so the leaf-stack
    path accepts it; the level-batched evaluation is fp64-exact."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = CumsumSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st = export_to_ir(m, x)

    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=14, max_nodes=400_000)
    best = eg.extract_min_depth(root)

    assert is_scan_apply_term(best)
    opt_ir = IR(
        root=best,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    mod = to_batched_scan_module(opt_ir, param_values=st)
    assert mod.is_batched
    mod.eval()
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10


def test_trace_lift_accepts_unit_carrier():
    """trace_lift's carrier path (build_scan_plan → expand aff_diag
    leaves into dense maps) handles the unit diagonal: both the joint
    and channel-split lifts verify fp64-exact."""
    torch.manual_seed(0)
    T, D = 8, 8
    m = CumsumSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st = export_to_ir(m, x)

    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=14, max_nodes=400_000)

    lifts = TL.lift_scan_to_trace(eg, root)
    assert len(lifts) >= 1
    with torch.no_grad():
        ref = m(x)
        for lft in lifts:
            mod = ir_to_torch_module(
                IR(root=lft.term, inputs=ir.inputs), param_values=st
            )
            assert (mod(x) - ref).abs().max().item() < 1e-10


def test_stratified_run_lifts_canonicalised_adds():
    """Under meta.stratified_run the add spine is canonicalised (operands
    sorted, tree balanced) — the _post positional variants carry the
    lift, carrier members appear in the e-graph, and the extracted term
    is fp64-exact."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = CumsumSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st = export_to_ir(m, x)

    eg = EGraph()
    out = meta.stratified_run(
        eg,
        R.SCAN_DIAG_LAWS,
        ir.root,
        max_iterations=14,
        max_nodes=400_000,
        extract_fn=eg.extract_min_depth,
    )

    fires = _fires(eg)
    assert sum(fires.values()) > 0
    assert (
        fires.get("affd_lift_unit_post", 0) > 0
        or fires.get("affd_lift_unit_step_post", 0) > 0
    )
    ops = {n.op for c in eg._classes.values() for n in c.nodes}
    assert "applyd" in ops and "aff_diag" in ops

    best = out["canonical_best"]
    opt_ir = IR(
        root=best,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    mod = ir_to_torch_module(opt_ir, param_values=st)
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10


# ---------------------------------------------------------------------------
#  (c) Guard: per-step / non-state adds do not lift
# ---------------------------------------------------------------------------


def test_unit_lift_ignores_non_state_adds():
    """``add(matmul(W,x), mul(a,y))`` — neither operand is state-like —
    fires none of the four unit rules.  (The generic mul-form lifts may
    still see ``mul(a, leaf)``; that permissiveness predates this law
    and is asserted separately.)"""
    x = Var("x", TensorType((4,)))
    y = Var("y", TensorType((4,)))
    W = Param("W", TensorType((4, 4)))
    a = Param("a", TensorType((4,)))
    term = Op.make("add", Op.make("matmul", W, x), Op.make("mul", a, y))

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=8)

    for name in _UNIT_RULES:
        assert eg.rule_fires.get(name, 0) == 0


def test_unit_lift_ignores_add_of_increments():
    """``add(select(x,0,t), select(x,0,s))`` — two per-step increments,
    no state — fires no unit rule.  A Const operand is likewise not a
    state: ``add(Const, x)`` stays put."""
    x = Var("x", TensorType((4, 4)))
    term = Op.make(
        "add",
        Op.make("select", x, dim=0, index=0),
        Op.make("select", x, dim=0, index=1),
    )
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=8)
    for name in _UNIT_RULES:
        assert eg.rule_fires.get(name, 0) == 0

    term2 = Op.make(
        "add", Const(2.0), Op.make("select", x, dim=0, index=0)
    )
    eg2 = EGraph()
    r2 = eg2.add_term(term2)
    eg2.run(R.SCAN_DIAG_LAWS, r2, max_iterations=8)
    for name in _UNIT_RULES:
        assert eg2.rule_fires.get(name, 0) == 0


def test_unit_lift_admits_leaf_state():
    """The guard admits a leaf h operand — that is how ``add(h0, x1)``
    enters the carrier.  Fires exactly the base rule (no step variant:
    no applyd precedes it)."""
    x = Var("x", TensorType((4, 4)))
    h0 = Param("h0", TensorType((4,)))
    term = Op.make("add", h0, Op.make("select", x, dim=0, index=0))

    eg = EGraph()
    root = eg.add_term(term)
    eg.run(R.SCAN_DIAG_LAWS, root, max_iterations=8)

    assert eg.rule_fires.get("affd_lift_unit", 0) == 1
    assert eg.rule_fires.get("affd_lift_unit_step", 0) == 0
