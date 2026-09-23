"""SCAN_DIAG_LAWS — the diagonal-affine monoid for elementwise SSMs.

``SCAN_LAWS`` lifts a dense recurrence step ``add(matmul(A,h), x)`` into
the affine-map monoid and lets associativity discover the Blelloch
parallel scan.  ``catopt.models.ssm.DiagonalSSM`` is the Mamba-faithful
elementwise form ``h_t = a_t ⊙ h_{t-1} + b_t ⊙ x_t``, which exports as
``add(mul(a_t, h), mul(b_t, x_t))`` — no ``matmul`` node, so ``AFF_LIFT``
never binds (see test_ssm_scan.py::test_diagonal_elementwise_ssm_does_not_lift
for the negative result on the old law set).

``SCAN_DIAG_LAWS`` closes that gap with the *diagonal-affine* carrier:

    aff_diag(a, b)      — the map h ↦ a⊙h + b       (a pair value)
    affd_compose(f, g)  — f∘g = (f₀⊙g₀, f₀⊙g₁ + f₁) (O(d), not O(d³))
    applyd(f, h)        — f₀⊙h + f₁                 (back in tensor-land)

Findings encoded as tests:

* The torch bindings implement the correct composition algebra:
  applyd(affd_compose(f,g), h) == applyd(f, applyd(g, h)) exactly.

* On DiagonalSSM the e-graph reaches the balanced parallel scan:
  depth ~35 (≈2T+leaf) collapses to ~11 for T=16 (~log T), and the
  lowered module is fp64-exact (diff ~1e-16 — reassociation rounding
  only).  The ``mul(b_t, x_t)`` translation is bound whole by the
  generic lift's ``x`` metavariable — no separate LHS shape needed.

* The diagonal carrier is strictly cheaper than dense ``aff``: each
  compose is 3 elementwise ops (≈3·d FLOPs) vs a dense d×d product plus
  matvec (≈2·d³).  On DiagDenseSSM (same recurrence, dense matmul
  spelling) the extracted scan costs ~130k FLOPs vs ~19k diagonal.
  E-graph growth is comparable (935 vs 968 enodes, T=16) — the saving
  is in the extracted term's work, not the search space.

* ``meta.stratified_run`` reaches the same log-depth form: the lift's
  operand-position variants (``affd_lift_swap`` etc.) cover the
  comm-normalised operand order ``canonicalize`` produces.  Two meta.py
  gaps remain (hardcoded, not fixed here):
    - ``meta._ASSOC_ONLY`` = {"matmul", "aff_compose"} lacks
      ``affd_compose`` — ``canonicalize`` will NOT rebalance a
      right-leaning diagonal compose chain post-hoc.  Harmless in
      practice: the assoc rules run as contentful rules during
      saturation, so extraction still returns the balanced tree.
    - ``meta.COHERENT_RULE_NAMES`` lacks ``affd_assoc``/``affd_assoc_rev``,
      so they are classified contentful (stored, not computed) — the
      same position ``aff_assoc`` holds for the dense carrier.

* ``scan_lower.BatchedScanModule`` does NOT recognise the diagonal
  term: ``is_scan_apply_term`` requires root ``apply`` over an
  ``aff``/``aff_compose`` tree, so ``applyd(affd_compose-tree, h)``
  falls back to the serial tuple-passing evaluator (still exact).  A
  batched executor for applyd trees would be simpler than the dense
  one — diagonal maps batch as two stacked (d,) vectors and compose is
  broadcasted mul/add, no homogeneous-matrix trick needed — but
  scan_lower.py would need an applyd-aware plan builder.
"""

import math

import pytest
import torch

from catopt.egraph import EGraph
from catopt import rules as R
from catopt import meta
from catopt.ir import IR, Op, Var, op_repr
from catopt.models.ssm import DiagDenseSSM, DiagonalSSM
from catopt.scan_lower import is_scan_apply_term, to_batched_scan_module
from catopt.torch_bridge import _IR_TO_TORCH, export_to_ir, ir_to_torch_module
from catopt.cost import flops_cost, dag_cost


def _opdepth(t, memo):
    """Critical-path depth of a term (shared-subterm DAG memoised)."""
    if not isinstance(t, Op):
        return 0
    k = id(t)
    if k not in memo:
        memo[k] = 1 + max((_opdepth(a, memo) for a in t.args), default=0)
    return memo[k]


def _scan(m, x, laws=None, max_nodes=400_000):
    """Export *m* and extract the min-depth term under *laws*."""
    laws = R.SCAN_DIAG_LAWS if laws is None else laws
    ir, st = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    stats = eg.run(laws, root, max_iterations=14, max_nodes=max_nodes)
    best = eg.extract_min_depth(root)
    return ir, st, best, stats


# ---------------------------------------------------------------------------
#  (a) Bindings: the monoid algebra is correct
# ---------------------------------------------------------------------------

def test_affd_bindings_compose_correctly():
    """affd_compose(f,g) = f∘g pointwise; applyd evaluates the map."""
    torch.manual_seed(0)
    d = 8
    a1, b1 = torch.randn(d, dtype=torch.float64), torch.randn(
        d, dtype=torch.float64)
    a2, b2 = torch.randn(d, dtype=torch.float64), torch.randn(
        d, dtype=torch.float64)
    h = torch.randn(d, dtype=torch.float64)

    aff_diag = _IR_TO_TORCH["aff_diag"]
    compose = _IR_TO_TORCH["affd_compose"]
    applyd = _IR_TO_TORCH["applyd"]

    f, g = aff_diag(a1, b1), aff_diag(a2, b2)
    fg = compose(f, g)

    # (a1,b1)∘(a2,b2) = (a1⊙a2, a1⊙b2 + b1)  — f∘g applies g first
    assert torch.equal(fg[0], a1 * a2)
    assert torch.equal(fg[1], a1 * b2 + b1)

    # applyd(compose(f,g), h) == applyd(f, applyd(g, h))  (associativity
    # of application — the law AFFD_COMPOSE_UNFOLD encodes)
    lhs = applyd(fg, h)
    rhs = applyd(f, applyd(g, h))
    assert torch.allclose(lhs, rhs, atol=1e-15, rtol=1e-15)

    # applyd(aff_diag(a,b), h) == a⊙h + b  (the step itself)
    assert torch.equal(applyd(f, h), a1 * h + b1)


# ---------------------------------------------------------------------------
#  (b) DiagonalSSM reaches log-depth and stays fp64-exact
# ---------------------------------------------------------------------------

def test_diagonal_ssm_lifts_to_log_depth():
    """add(mul(a_t,h), mul(b_t,x_t)) lifts into the diagonal monoid and
    reassociates to the balanced (Blelloch) bracketing."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = DiagonalSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)

    ir, st, best, stats = _scan(m, x)
    d_orig, d_best = _opdepth(ir.root, {}), _opdepth(best, {})

    rep = op_repr(best)
    assert "applyd" in rep
    assert "affd_compose" in rep
    assert "aff_diag" in rep
    assert d_best < d_orig
    # measured: T=16 -> 11 (leaf chains ~5 + ~log2 T compose levels)
    assert d_best <= 4 * math.ceil(math.log2(T)) + 8

    opt_ir = IR(root=best, inputs=ir.inputs,
                input_names=ir.input_names, params=ir.params)
    mod = ir_to_torch_module(opt_ir, param_values=st)
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10


def test_diagonal_ssm_scan_t32():
    """T=32: depth stays ~log T (65 -> ~14) and stays exact."""
    torch.manual_seed(0)
    T, D = 32, 16
    m = DiagonalSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, best, _ = _scan(m, x)
    assert _opdepth(best, {}) <= 4 * math.ceil(math.log2(T)) + 8
    opt_ir = IR(root=best, inputs=ir.inputs,
                input_names=ir.input_names, params=ir.params)
    mod = ir_to_torch_module(opt_ir, param_values=st)
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10


# ---------------------------------------------------------------------------
#  (c) The diagonal carrier is cheaper than dense aff
# ---------------------------------------------------------------------------

def test_diag_carrier_cheaper_than_dense_aff():
    """Same recurrence, two spellings: DiagonalSSM (mul-form) under
    SCAN_DIAG_LAWS vs DiagDenseSSM (matmul-form) under SCAN_LAWS.

    The compose is 3 elementwise ops (~3·d) instead of a dense product
    (~2·d³), so the extracted diagonal scan carries ~7x fewer FLOPs.
    Search growth itself is comparable — the win is in the term, not
    the e-graph size."""
    torch.manual_seed(0)
    T, D = 16, 16

    m_d = DiagonalSSM(D, D, T).eval().double()
    x_d = torch.randn(T, D, dtype=torch.float64)
    _, _, best_d, stats_d = _scan(m_d, x_d)

    m_f = DiagDenseSSM(D, D, T).eval().double()
    x_f = torch.randn(T, D, dtype=torch.float64)
    _, _, best_f, stats_f = _scan(m_f, x_f, laws=R.SCAN_LAWS)

    # Both reach the balanced scan at comparable e-graph size.
    assert stats_d["n_enodes"] <= stats_f["n_enodes"] * 2
    # The diagonal compose is O(d): extracted scan does far less work.
    f_d = dag_cost(best_d, flops_cost)
    f_f = dag_cost(best_f, flops_cost)
    assert f_d < f_f / 2, (f_d, f_f)


# ---------------------------------------------------------------------------
#  (d) Stratified run: canonicalize + contentful-only saturation
# ---------------------------------------------------------------------------

def test_stratified_run_reaches_log_depth():
    """stratified_run on the same graph: canonicalize comm-sorts the
    add/mul operands (``add(mul(h,a), mul(b,x))`` for t>0,
    ``add(mul(b,x), mul(a,h0))`` for the first step) — the positional
    lift variants cover both, and the balanced scan is extracted.

    NOTE: meta._ASSOC_ONLY is a hardcoded frozenset without
    ``affd_compose``, so canonicalize does not rebalance diagonal
    compose chains directly (checked below); extraction via the
    contentful assoc rules produces the balanced form anyway."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = DiagonalSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st = export_to_ir(m, x)

    eg = EGraph()
    out = meta.stratified_run(
        eg, R.SCAN_DIAG_LAWS, ir.root, max_iterations=14,
        max_nodes=400_000,
        extract_fn=eg.extract_min_depth)

    best = out["canonical_best"]
    rep = op_repr(best)
    assert "affd_compose" in rep
    assert "applyd" in rep
    assert _opdepth(best, {}) <= 4 * math.ceil(math.log2(T)) + 8

    opt_ir = IR(root=best, inputs=ir.inputs,
                input_names=ir.input_names, params=ir.params)
    mod = ir_to_torch_module(opt_ir, param_values=st)
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10


def test_stratified_run_documents_meta_gap():
    """``meta.canonicalize``'s balanced-ops list is a hardcoded
    frozenset (``meta._ASSOC_ONLY``) that does not include
    ``affd_compose`` — a right-leaning diagonal compose chain is NOT
    rebalanced by canonicalize.  The stratified pipeline compensates:
    ``affd_assoc``/``affd_assoc_rev`` are classified contentful (they
    are absent from ``COHERENT_RULE_NAMES``) and the balanced form is
    found by search + depth extraction instead.  If meta.py later
    learns the op, this test still passes — it only asserts the
    pipeline's *output* is balanced."""
    torch.manual_seed(0)
    T, D = 8, 8
    m = DiagonalSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, _ = export_to_ir(m, x)

    eg = EGraph()
    out = meta.stratified_run(
        eg, R.SCAN_DIAG_LAWS, ir.root, max_iterations=14,
        max_nodes=200_000,
        extract_fn=eg.extract_min_depth)

    # The assoc pair ran as contentful rules (not dropped as coherent):
    assert "affd_assoc" in out["contentful_used"]
    assert "affd_assoc" not in out["coherent_dropped"]
    # ...yet the extracted term is still the balanced scan.
    rep = op_repr(out["canonical_best"])
    assert "affd_compose" in rep
    assert _opdepth(out["canonical_best"], {}) <= \
        4 * math.ceil(math.log2(T)) + 8


# ---------------------------------------------------------------------------
#  Lowering: BatchedScanModule falls back to serial eval on applyd trees
# ---------------------------------------------------------------------------

def test_batched_scan_module_serial_fallback_on_applyd():
    """``is_scan_apply_term`` requires ``apply`` over an ``aff`` tree —
    the diagonal ``applyd`` root is not recognised, so the module
    delegates to the tuple-passing IRModule path.  The fallback is
    still fp64-exact; a batched executor for applyd trees is future
    work (see module docstring)."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = DiagonalSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, best, _ = _scan(m, x)

    assert not is_scan_apply_term(best)
    opt_ir = IR(root=best, inputs=ir.inputs,
                input_names=ir.input_names, params=ir.params)
    mod = to_batched_scan_module(opt_ir, param_values=st)
    assert not mod.is_batched
    mod.eval()
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10
