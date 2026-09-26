"""Level-batched lowering for the affine-monoid parallel scan.

The e-graph discovers that an unrolled recurrence ``h_t = A·h_{t-1} + x_t``
is a composition of affine maps and extracts a balanced
``apply(aff_compose-tree, h0)`` term (~2·log2 T depth).  The generic
IRModule evaluates that tree serially via tuple passing;
:class:`catopt.scan_lower.BatchedScanModule` instead batches every tree
level into one matmul pair — O(log T) kernel launches.
"""

import math

import pytest
import torch

from catopt_core import laws as R
from catopt.egraph import EGraph
from catopt.ir import IR, Op, op_repr
from catopt.models import LinearRecurrence, SwiGLU
from catopt.scan_lower import (
    BatchedScanModule,
    build_scan_plan,
    is_scan_apply_term,
    to_batched_scan_module,
)
from catopt.torch_bridge import export_to_ir, ir_to_torch_module


def _opdepth(t, memo):
    """Depth cost — the same extraction objective the scan tests use."""
    if not isinstance(t, Op):
        return 0
    k = id(t)
    if k not in memo:
        memo[k] = 1 + max(
            (_opdepth(a, memo) for a in t.args), default=0
        )
    return memo[k]


def _scan_term(model_dim, steps, dtype=torch.float64, seed=0):
    """Export LinearRecurrence and extract the scan form via SCAN_LAWS."""
    torch.manual_seed(seed)
    m = LinearRecurrence(model_dim, steps).eval().to(dtype)
    x = torch.randn(steps, model_dim, dtype=dtype)
    ir, source = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(R.SCAN_LAWS, root, max_iterations=14, max_nodes=300_000)
    best = eg.extract_min_depth(root)
    opt_ir = IR(
        root=best,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    return m, x, opt_ir, source


def _balanced_scan_ir(model, x_var, ir, steps):
    """Hand-build the canonical balanced ``apply(aff-tree, h0)`` term.

    Deterministic stand-in for the extracted term: leaves are the
    per-step affine maps ``aff(A, x_t)`` in chronological order and the
    compose tree is perfectly balanced — the shape depth-extraction
    converges to.
    """
    p_a = ir.params["p_a"]
    p_h0 = ir.params["p_h0"]

    def leaf(t):
        return Op.make(
            "aff", p_a, Op.make("select", x_var, arg1=0, arg2=t)
        )

    def tree(lo, hi):
        if hi - lo == 1:
            return leaf(lo)
        mid = (lo + hi) // 2
        # compose(f, g) = f ∘ g: the LATER steps sit on the left.
        return Op.make("aff_compose", tree(mid, hi), tree(lo, mid))

    root = Op.make("apply", tree(0, steps), p_h0)
    return IR(
        root=root,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )


# ---------------------------------------------------------------------------
#  Numerical equivalence vs the sequential recurrence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_batched_scan_matches_sequential(steps, dtype):
    """apply(balanced aff-tree, h0) lowered level-batched equals the
    sequential fold h_t = A·h_{t-1} + x_t."""
    torch.manual_seed(0)
    d = 16
    m = LinearRecurrence(d, steps).eval().to(dtype)
    x = torch.randn(steps, d, dtype=dtype)
    ir, source = export_to_ir(m, x)

    scan_ir = _balanced_scan_ir(m, ir.inputs[0], ir, steps)
    mod = to_batched_scan_module(scan_ir, param_values=source)
    assert mod.is_batched
    # one aff leaf per step; depth ~log2(steps) compose levels
    assert mod.n_levels <= math.ceil(math.log2(steps)) + 1

    mod.eval()
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    tol = 1e-10 if dtype == torch.float64 else 1e-5
    assert diff < tol


def test_batched_scan_matches_serial_lowering():
    """Batched execution agrees with the tuple-passing IRModule on the
    identical term (guards against an ordering slip in the batching)."""
    torch.manual_seed(0)
    d, steps = 16, 32
    m = LinearRecurrence(d, steps).eval().double()
    x = torch.randn(steps, d, dtype=torch.float64)
    ir, source = export_to_ir(m, x)
    scan_ir = _balanced_scan_ir(m, ir.inputs[0], ir, steps)

    serial = ir_to_torch_module(scan_ir, param_values=source).eval()
    batched = to_batched_scan_module(
        scan_ir, param_values=source
    ).eval()
    assert batched.is_batched
    with torch.no_grad():
        diff = (serial(x) - batched(x)).abs().max().item()
    assert diff < 1e-12


# ---------------------------------------------------------------------------
#  Detection on the real extracted term (EGraph + SCAN_LAWS + depth cost)
# ---------------------------------------------------------------------------


def test_detects_extracted_scan_term():
    """The module recognises the term SCAN_LAWS + depth extraction emit.

    Same pipeline as test_torch_integration::
    test_affine_monoid_parallel_scan — saturation then op-depth
    extraction must yield an apply(aff-tree, h) root, and the lowered
    module must take the batched path and stay fp64-exact.
    """
    m, x, opt_ir, source = _scan_term(16, 16)
    assert is_scan_apply_term(opt_ir.root), op_repr(opt_ir.root)
    assert "aff_compose" in op_repr(opt_ir.root)

    mod = to_batched_scan_module(opt_ir, param_values=source)
    assert mod.is_batched
    # ~2·log2(T) compose depth ⇒ at most that many batched levels.
    assert mod.n_levels <= 2 * math.ceil(math.log2(16)) + 1

    mod.eval()
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10


def test_extracted_scan_term_t32_equivalent():
    """T=32 end-to-end: whatever bracketing extraction returns must be
    numerically exact (batched when the root is apply, serial fallback
    otherwise — both paths are checked for correctness)."""
    m, x, opt_ir, source = _scan_term(16, 32)
    mod = to_batched_scan_module(opt_ir, param_values=source)
    mod.eval()
    with torch.no_grad():
        diff = (m(x) - mod(x)).abs().max().item()
    assert diff < 1e-10


# ---------------------------------------------------------------------------
#  Plan structure / API surface
# ---------------------------------------------------------------------------


def test_plan_levels_are_independent():
    """build_scan_plan groups composes by depth; leaves = aff nodes."""
    torch.manual_seed(0)
    d, steps = 8, 8
    m = LinearRecurrence(d, steps).eval().double()
    x = torch.randn(steps, d, dtype=torch.float64)
    ir, _ = export_to_ir(m, x)
    scan_ir = _balanced_scan_ir(m, ir.inputs[0], ir, steps)

    plan = build_scan_plan(scan_ir.root)
    assert plan is not None
    assert len(plan["leaves"]) == steps
    assert all(lf.op == "aff" for lf in plan["leaves"])
    # balanced tree over 8 leaves: 4 + 2 + 1 composes on 3 levels
    assert [len(lv) for lv in plan["levels"]] == [4, 2, 1]


def test_plan_rejects_non_scan_root():
    x = torch.randn(4, 4)
    m = SwiGLU(4, hidden_mult=2).eval()
    ir, _ = export_to_ir(m, x)
    assert build_scan_plan(ir.root) is None
    assert not is_scan_apply_term(ir.root)


# ---------------------------------------------------------------------------
#  CUDA graph replay (optional fast path)
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_cuda_graph_replay_matches_eager():
    """capture_cuda_graph replays the identical computation; a changed
    input is picked up via the static input buffers, and
    drop_cuda_graph restores the eager path."""
    torch.manual_seed(0)
    d, steps = 16, 32
    m = LinearRecurrence(d, steps).eval()
    x = torch.randn(steps, d)
    ir, source = export_to_ir(m, x)
    scan_ir = _balanced_scan_ir(m, ir.inputs[0], ir, steps)

    mod = to_batched_scan_module(scan_ir, param_values=source)
    mod.eval().cuda()
    xc = x.cuda()
    mod.capture_cuda_graph(xc)
    assert mod.is_graph_captured

    with torch.no_grad():
        out1 = mod(xc).clone()
        x2 = torch.randn(steps, d, device="cuda")
        out2 = mod(x2).clone()
        m2 = m.cuda()
        assert (m2(xc) - out1).abs().max().item() < 1e-5
        assert (m2(x2) - out2).abs().max().item() < 1e-5

        mod.drop_cuda_graph()
        assert not mod.is_graph_captured
        out3 = mod(x2)
        assert (m2(x2) - out3).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
#  Fallback: non-scan IR keeps working
# ---------------------------------------------------------------------------


def test_fallback_matches_plain_lowering():
    """Non-scan IR: BatchedScanModule delegates to serial evaluation."""
    torch.manual_seed(0)
    m = SwiGLU(16, hidden_mult=2).eval()
    x = torch.randn(2, 3, 16)
    ir, source = export_to_ir(m, x)

    mod = to_batched_scan_module(ir, param_values=source)
    assert isinstance(mod, BatchedScanModule)
    assert not mod.is_batched
    ref = ir_to_torch_module(ir, param_values=source)
    mod.eval()
    ref.eval()
    with torch.no_grad():
        assert torch.equal(mod(x.clone()), ref(x.clone()))
        assert (m(x.clone()) - mod(x.clone())).abs().max().item() < 1e-6


def test_fallback_on_scan_ops_outside_apply():
    """aff ops NOT under a root apply (e.g. mid-extraction forms) fall
    back to tuple-passing eval rather than crashing."""
    torch.manual_seed(0)
    d = 8
    m = LinearRecurrence(d, 4).eval().double()
    x = torch.randn(4, d, dtype=torch.float64)
    ir, source = export_to_ir(m, x)
    p_a = ir.params["p_a"]
    xv = ir.inputs[0]
    # A bare compose tree with no apply wrapper: not scan-shaped, so the
    # module must fall back to serial tuple-passing eval.
    bare = Op.make(
        "aff_compose",
        Op.make("aff", p_a, Op.make("select", xv, arg1=0, arg2=1)),
        Op.make("aff", p_a, Op.make("select", xv, arg1=0, arg2=0)),
    )
    weird_ir = IR(
        root=bare,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    mod = to_batched_scan_module(weird_ir, param_values=source)
    assert not mod.is_batched
    # Serial eval of a bare aff_compose returns the (A, b) pair:
    #   (A,x1) ∘ (A,x0) = (A@A, A@x0 + x1)
    with torch.no_grad():
        out = mod(x)
    assert isinstance(out, tuple) and len(out) == 2
    assert torch.allclose(out[0], source["p_a"] @ source["p_a"])
    assert torch.allclose(out[1], source["p_a"] @ x[0] + x[1])
