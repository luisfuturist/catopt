# ruff: noqa: RUF002
"""SCAN_LAWS on a *selective* SSM — input-dependent per-step dynamics.

``LinearRecurrence`` (test_torch_integration.py) lifts a fixed-A LTI
fold ``h_t = A h_{t-1} + x_t`` into the affine-map monoid.  Here the
transition itself is data-dependent: ``h_t = A_t h_{t-1} + u_t`` with
``A_t = I + Δ_t ⊙ A``, ``Δ_t = tanh(W_δ x_t)`` — the Mamba/S4
"selective" mechanism with a dense discretised transition.

Findings encoded as tests:

* The lift DOES fire: torch.export emits ``add(matmul(A_t, h), u_t)``
  where ``A_t`` is an arbitrary input-dependent term (tanh→linear→
  select→unsqueeze→mul→add); the ``"A"`` metavariable binds it whole.
  Associativity of affine composition then generates the balanced
  (Blelloch) bracketing exactly as in the fixed-A case — depth drops
  from ~2T to ~log T (T=32: 70 → 17) and the lowered module is
  fp64-exact (diff ~1e-14; the residual is reassociation rounding).

* The elementwise-diagonal form ``h_t = a_t ⊙ h_{t-1} + b_t ⊙ x_t``
  exports as ``add(mul(a_t, h), ...)`` — no ``matmul`` node, so
  ``AFF_LIFT`` never matches.  Covering it needs a diagonal lift rule
  (``add(mul(a,h),x) → apply(aff_diag(a,x), h)``) + ``aff_diag``
  lowering — a rules.py/torch_bridge.py change, out of scope here.

* Shape inference is agnostic to A_t being input-dependent: ``aff``/
  ``aff_compose`` report the linear part's shape and ``apply`` reports
  h's — no fixed-A assumption anywhere in the cost/extraction path.
  (The only place a Var-free A would matter is compile-time folding;
  an input-dependent A_t is correctly NOT folded by IRModule.)
"""

import math

import pytest
import torch

from catopt import rules as R
from catopt.egraph import EGraph
from catopt.ir import IR, Op, Var, op_repr
from catopt.models.ssm import DiagDenseSSM, DiagonalSSM, SelectiveSSM
from catopt.torch_bridge import export_to_ir, ir_to_torch_module


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


def _mentions_var(t, seen=None):
    if seen is None:
        seen = set()
    if id(t) in seen:
        return False
    seen.add(id(t))
    if isinstance(t, Var):
        return True
    if isinstance(t, Op):
        return any(_mentions_var(a, seen) for a in t.args)
    return False


def _scan(m, x, max_nodes=400_000):
    ir, st = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    stats = eg.run(
        R.SCAN_LAWS, root, max_iterations=14, max_nodes=max_nodes
    )
    best = eg.extract_min_depth(root)
    return ir, st, best, stats


def test_selective_ssm_export_has_affine_step_shape():
    """torch.export must emit add(matmul(A_t, h), u_t) per step — the
    LHS shape AFF_LIFT matches — with A_t an input-dependent term."""
    torch.manual_seed(0)
    m = SelectiveSSM(16, 16, 8).eval().double()
    x = torch.randn(8, 16, dtype=torch.float64)
    ir, _ = export_to_ir(m, x)
    assert ir.root.op == "add"
    mm, _u = ir.root.args
    assert mm.op == "matmul"
    a_t, _h = mm.args
    # A_t is data-dependent: it must contain the input Var, so the
    # affine lift is matching a *selective* step, not a fixed matrix.
    assert _mentions_var(a_t)
    assert a_t.op != "leaf"


def test_selective_ssm_affine_lift_reaches_log_depth():
    """SCAN_LAWS reassociate the input-dependent fold into the balanced
    parallel-scan bracketing; the lowered module is fp64-exact."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = SelectiveSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)

    ir, st, best, _stats = _scan(m, x)
    d_orig, d_best = _opdepth(ir.root, {}), _opdepth(best, {})

    # The affine monoid was reached: apply/aff_compose appear, and the
    # critical path is log-scale, not linear (~2T + leaf depth).
    assert "aff_compose" in op_repr(best)
    assert "apply" in op_repr(best)
    assert d_best < d_orig
    # measured: T=8→12, T=16→13, T=32→17 — well inside this bound.
    assert d_best <= 4 * math.ceil(math.log2(T)) + 8

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


def test_selective_ssm_scan_t32():
    """Larger unroll: depth stays ~log T (70 -> ~17) and stays exact."""
    torch.manual_seed(0)
    T, D = 32, 16
    m = SelectiveSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, best, _ = _scan(m, x)
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


def test_diag_dense_ssm_affine_lift():
    """A contractive diagonal decay materialised as a dense matmul
    (``(eye * a_t) @ h``) lifts and verifies fp64-exact — diagonal
    dynamics are fine as long as the op is spelled ``matmul``."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = DiagDenseSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, best, _ = _scan(m, x)
    assert "aff_compose" in op_repr(best)
    assert _opdepth(best, {}) < _opdepth(ir.root, {})
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


def test_diagonal_elementwise_ssm_does_not_lift():
    """The honest negative: ``a_t ⊙ h + b_t ⊙ x_t`` exports as
    add(mul, mul) — AFF_LIFT's ``matmul`` LHS never matches, so the
    e-graph contributes nothing (depth and term unchanged).  A
    ``mul``-form lift rule in rules.py would be required."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = DiagonalSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, _st, best, _ = _scan(m, x)
    # Sanity: the step really is add(mul(a_t, h), mul(b_t, x_t)).
    assert ir.root.op == "add"
    assert ir.root.args[0].op == "mul"
    # No affine lift: no apply/aff enodes reachable in the best term.
    assert "apply" not in op_repr(best)
    assert "aff" not in op_repr(best)
    assert _opdepth(best, {}) == _opdepth(ir.root, {})


@pytest.mark.requires_cuda
def test_selective_ssm_gpu_wallclock():
    """Sequential vs scan-extracted wall-clock on CUDA.

    Expected honest negative: the extracted term trades T sequential
    matvecs for ~log T dense d×d products (more FLOPs at small d), and
    a serial CUDA stream can't fill the parallel bracketing — so the
    extracted module typically LOSES.  Only correctness is asserted;
    timings are printed for the record."""
    import time

    torch.manual_seed(0)
    T, D = 32, 32
    # Export on CPU (torch.export is device-agnostic); .cuda() moves the
    # module in place afterwards — do NOT pass m.cpu() into _scan.
    m = SelectiveSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, best, _ = _scan(m, x)
    opt_ir = IR(
        root=best,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    m, x = m.cuda(), x.cuda()
    mod = (
        ir_to_torch_module(
            opt_ir, param_values={k: v.cuda() for k, v in st.items()}
        )
        .cuda()
        .double()
    )

    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-9

    def bench(fn, n=100):
        with torch.no_grad():
            for _ in range(10):
                fn(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn(x)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6  # us

    t_seq, t_opt = bench(m), bench(mod)
    print(
        f"\n[cuda] sequential {t_seq:.1f} us  extracted {t_opt:.1f} us"
        f"  (ratio {t_opt / t_seq:.2f}x)"
    )
