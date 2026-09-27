"""Fused level-step for the batched scan executor (fused=True).

``to_batched_scan_module(..., fused=...)`` selects the canonical
adjacent-pair reduction from :mod:`catopt_carriers.scan_fused` — a
re-bracketed (association-invariant) schedule whose levels are pure
strided pointwise/matmul steps on shrinking fresh tensors.  These
tests pin numerical equivalence with the serial IRModule on the same
term (fp64), the DAG/occurrence and pad edges, the compile fallback,
and the routing invariants (``is_batched``/``n_levels``/``_plan``).
"""

import pytest
import torch
from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.scan_lower import (
    BatchedScanModule,
    build_scan_plan,
    to_batched_scan_module,
)
from catopt.torch_bridge import ir_to_torch_module
from catopt_carriers.scan_fused import (
    fused_dense_levels,
    fused_diag_levels,
    occurrence_slots,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _compose(opname, leaves):
    if len(leaves) == 1:
        return leaves[0]
    mid = len(leaves) // 2
    return Op.make(
        opname,
        _compose(opname, leaves[:mid]),
        _compose(opname, leaves[mid:]),
    )


def _diag_ir(T, d):
    """Canonical retnet-style ``applyd(affd-tree, h)`` term."""
    a, h = _p("a", d), _p("h", d)
    x = _v("x", T, d)
    leaves = [
        Op.make("aff_diag", a, Op.make("select", x, dim=0, index=t))
        for t in range(T)
    ]
    root = Op.make("applyd", _compose("affd_compose", leaves), h)
    return IR(root=root, inputs=[x], input_names={"x"}, params={})


def _dense_ir(T, d):
    A, h = _p("A", d, d), _p("h", d)
    x = _v("x", T, d)
    leaves = [
        Op.make("aff", A, Op.make("select", x, dim=0, index=t))
        for t in range(T)
    ]
    root = Op.make("apply", _compose("aff_compose", leaves), h)
    return IR(root=root, inputs=[x], input_names={"x"}, params={})


# ---------------------------------------------------------------------------
#  Diagonal carrier — every fused mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fused", ["eager", True, "compile"])
@pytest.mark.parametrize("steps", [8, 6, 5, 16])
def test_fused_diag_matches_serial(fused, steps):
    """fp64: fused schedule agrees with the tuple-passing IRModule on
    the identical term — including non-power-of-2 leaf counts that
    exercise the identity pad."""
    torch.manual_seed(0)
    d = 4
    ir = _diag_ir(steps, d)
    av = torch.rand(d, dtype=torch.float64) * 0.9 + 0.05
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(steps, d, dtype=torch.float64)
    pv = {"a": av, "h": hv}

    gen = ir_to_torch_module(ir, param_values=pv).eval()
    mod = to_batched_scan_module(ir, param_values=pv, fused=fused)
    assert isinstance(mod, BatchedScanModule) and mod.is_batched
    mod.eval()
    with torch.no_grad():
        out1 = mod(xv)
        out2 = mod(xv)  # second call: compiled body already wrapped
    diff = (out1 - gen(xv)).abs().max().item()
    assert diff < 1e-12
    assert torch.equal(out1, out2)
    # reference: the composed map applies leaf[0] LAST — iterate the
    # leaf order in reverse (same convention as test_cov_scan_lower).
    hh = hv.clone()
    for t in range(steps - 1, -1, -1):
        hh = av * hh + xv[t]
    assert (out1 - hh).abs().max().item() < 1e-12


def test_fused_flag_and_routing_invariants():
    """``is_batched``/``n_levels``/``_plan``/``_root``/``_param_map``
    report the extracted term identically under fused."""
    torch.manual_seed(0)
    ir = _diag_ir(8, 4)
    pv = {
        "a": torch.rand(4, dtype=torch.float64),
        "h": torch.randn(4, dtype=torch.float64),
    }
    plain = to_batched_scan_module(ir, param_values=pv)
    fused = to_batched_scan_module(ir, param_values=pv, fused="eager")
    assert fused.is_batched == plain.is_batched
    assert fused.n_levels == plain.n_levels
    assert fused._plan is not None
    assert fused._root is plain._root
    assert fused._param_map.keys() == plain._param_map.keys()
    assert all(
        torch.equal(fused._param_map[k], plain._param_map[k])
        for k in fused._param_map
    )
    assert fused._fused == "eager" and plain._fused is None


def test_fused_invalid_value_rejected():
    ir = _diag_ir(4, 3)
    with pytest.raises(ValueError, match="fused must be"):
        to_batched_scan_module(ir, fused="turbo")


def test_fused_nonscan_root_falls_back():
    """fused=True on non-scan IR: no plan, transparent serial eval."""
    torch.manual_seed(0)
    x = _v("x", 4)
    ir = IR(
        root=Op.make("relu", x),
        inputs=[x],
        input_names={"x"},
        params={},
    )
    mod = to_batched_scan_module(ir, fused=True)
    assert not mod.is_batched and mod._fused is None
    with torch.no_grad():
        out = mod(torch.full((4,), -1.0))
    assert torch.equal(out, torch.zeros(4))


# ---------------------------------------------------------------------------
#  Dense carrier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fused", ["eager", True])
@pytest.mark.parametrize("steps", [8, 5])
def test_fused_dense_matches_serial(fused, steps):
    torch.manual_seed(0)
    d = 4
    ir = _dense_ir(steps, d)
    Av = torch.randn(d, d, dtype=torch.float64) * 0.3
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(steps, d, dtype=torch.float64)
    pv = {"A": Av, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    mod = to_batched_scan_module(ir, param_values=pv, fused=fused)
    assert mod.is_batched
    with torch.no_grad():
        out = mod(xv)
    assert (out - gen(xv)).abs().max().item() < 1e-12
    hh = hv.clone()
    for t in range(steps - 1, -1, -1):
        hh = Av @ hh + xv[t]
    assert (out - hh).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  Occurrence order — DAG-shared leaves and subtrees
# ---------------------------------------------------------------------------


def test_fused_dag_shared_leaf_diag():
    """``compose(leaf, leaf)`` dedups to one leaf slot but the product
    uses it twice — the fused path gathers by occurrence order."""
    torch.manual_seed(0)
    d = 4
    a, bb, h = _p("a", d), _p("bb", d), _p("h", d)
    leaf = Op.make("aff_diag", a, bb)
    root = Op.make("applyd", Op.make("affd_compose", leaf, leaf), h)
    ir = IR(root=root, inputs=[], input_names=set(), params={})
    plan = build_scan_plan(ir.root)
    assert occurrence_slots(plan["f"], plan["leaves"]) == [0, 0]

    av = torch.rand(d, dtype=torch.float64) * 0.8 + 0.1
    bv = torch.randn(d, dtype=torch.float64)
    hv = torch.randn(d, dtype=torch.float64)
    pv = {"a": av, "bb": bv, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    for fused in ("eager", True):
        mod = to_batched_scan_module(ir, param_values=pv, fused=fused)
        assert mod._fused_occ == [0, 0]
        with torch.no_grad():
            out = mod()
        assert (out - gen()).abs().max().item() < 1e-12
        # explicit reference: (a,b)∘(a,b) applied to h
        assert torch.allclose(out, av * av * hv + av * bv + bv)


def test_fused_dag_shared_leaf_dense():
    """Dense carrier takes the occurrence gather on the packed mats."""
    torch.manual_seed(0)
    d = 3
    A, b, h = _p("A", d, d), _p("b", d), _p("h", d)
    leaf = Op.make("aff", A, b)
    root = Op.make("apply", Op.make("aff_compose", leaf, leaf), h)
    ir = IR(root=root, inputs=[], input_names=set(), params={})
    Av = torch.randn(d, d, dtype=torch.float64) * 0.3
    bv = torch.randn(d, dtype=torch.float64)
    hv = torch.randn(d, dtype=torch.float64)
    pv = {"A": Av, "b": bv, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    for fused in ("eager", True):
        mod = to_batched_scan_module(ir, param_values=pv, fused=fused)
        assert mod._fused_occ == [0, 0]
        with torch.no_grad():
            out = mod()
        assert (out - gen()).abs().max().item() < 1e-12
        assert torch.allclose(out, Av @ (Av @ hv + bv) + bv)


def test_fused_declines_on_occurrence_explosion():
    """Deeply shared subtree: occurrence expansion exceeds the bound
    → fused declines and the standard schedule runs the term."""
    torch.manual_seed(0)
    d = 3
    a, bb, h = _p("a", d), _p("bb", d), _p("h", d)
    leaf = Op.make("aff_diag", a, bb)
    t = leaf
    for _ in range(4):  # occurrences 2^4 = 16 > 8 * 1 leaf
        t = Op.make("affd_compose", t, t)
    ir = IR(
        root=Op.make("applyd", t, h),
        inputs=[],
        input_names=set(),
        params={},
    )
    plan = build_scan_plan(ir.root)
    assert occurrence_slots(plan["f"], plan["leaves"]) is None
    mod = to_batched_scan_module(ir, fused="eager")
    assert mod.is_batched and mod._fused is None
    av = torch.rand(d, dtype=torch.float64) * 0.5 + 0.4
    bv = torch.randn(d, dtype=torch.float64)
    hv = torch.randn(d, dtype=torch.float64)
    pv = {"a": av, "bb": bv, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    mod = to_batched_scan_module(ir, param_values=pv, fused="eager")
    with torch.no_grad():
        assert (mod() - gen()).abs().max().item() < 1e-12


def test_occurrence_slots_terminal_nonleaf():
    """A non-compose terminal inside the walk → decline (None)."""
    leaf = Op.make("aff_diag", _p("a", 4), _p("b", 4))
    assert occurrence_slots(_v("stray", 4), [leaf]) is None


# ---------------------------------------------------------------------------
#  Leaf-eval fast path (evf): Op fallthrough + unregistered Param/Var
# ---------------------------------------------------------------------------


def test_fused_evf_op_fallthrough():
    """Leaf terms that are richer than bare Param/Var (here an
    ``add(select(x), c)`` b-part — gather-ineligible) resolve through
    the full ``eval_term`` path."""
    torch.manual_seed(0)
    T, d = 4, 3
    a, c, h = _p("a", d), _p("c", d), _p("h", d)
    x = _v("x", T, d)
    leaves = [
        Op.make(
            "aff_diag",
            a,
            Op.make(
                "add",
                Op.make("select", x, dim=0, index=t),
                c,
            ),
        )
        for t in range(T)
    ]
    ir = IR(
        root=Op.make("applyd", _compose("affd_compose", leaves), h),
        inputs=[x],
        input_names={"x"},
        params={},
    )
    av = torch.rand(d, dtype=torch.float64) * 0.9 + 0.05
    cv = torch.randn(d, dtype=torch.float64)
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(T, d, dtype=torch.float64)
    pv = {"a": av, "c": cv, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    for fused in (False, "eager", True):
        mod = to_batched_scan_module(ir, param_values=pv, fused=fused)
        assert mod.is_batched
        with torch.no_grad():
            diff = (mod(xv) - gen(xv)).abs().max().item()
        assert diff < 1e-12


def test_fused_evf_var_miss_falls_back():
    """An operand that is a bare Var NOT among the inputs resolves
    through eval_term's ``var_default=x`` fallback — here as the
    applied-to term ``h``."""
    torch.manual_seed(0)
    T, d = 4, 3
    x = _v("x", T, d)
    a, w = _p("a", d), _v("w", d)  # w is not among the inputs
    leaves = [
        Op.make(
            "aff_diag",
            a,
            Op.make("select", x, dim=0, index=t),
        )
        for t in range(T)
    ]
    ir = IR(
        root=Op.make("applyd", _compose("affd_compose", leaves), w),
        inputs=[x],
        input_names={"x"},
        params={},
    )
    av = torch.rand(d, dtype=torch.float64) * 0.5 + 0.4
    xv = torch.randn(T, d, dtype=torch.float64) * 0.3
    gen = ir_to_torch_module(ir, param_values={"a": av}).eval()
    mod = to_batched_scan_module(
        ir, param_values={"a": av}, fused="eager"
    )
    assert mod.is_batched
    with torch.no_grad():
        # both paths resolve w → x (var_default); outputs agree
        diff = (mod(xv) - gen(xv)).abs().max().item()
    assert diff < 1e-12


def test_fused_evf_unregistered_param_falls_back():
    """A leaf Param absent from ``_param_map`` (unregistered) hits the
    eval fallback — simulated by dropping it after construction."""
    torch.manual_seed(0)
    T, d = 4, 3
    ir = _diag_ir(T, d)
    av = torch.rand(d, dtype=torch.float64) * 0.9 + 0.05
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(T, d, dtype=torch.float64)
    pv = {"a": av, "h": hv}
    mod = to_batched_scan_module(ir, param_values=pv, fused="eager")
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    # remove the 'a' param registration → evf must fall through to ev,
    # which materialises the unshaped-Param randn default — just check
    # the call path works and shape is right.
    del mod.eval_mod._param_map["a"]
    with torch.no_grad():
        out = mod(xv)
    assert out.shape == (d,)
    # put it back and confirm normal resolution is unaffected
    mod2 = to_batched_scan_module(ir, param_values=pv, fused="eager")
    with torch.no_grad():
        diff = (mod2(xv) - gen(xv)).abs().max().item()
    assert diff < 1e-12


# ---------------------------------------------------------------------------
#  b-gather variants: non-identity indices and non-dim-0 base
# ---------------------------------------------------------------------------


def test_fused_nonidentity_b_gather():
    torch.manual_seed(0)
    T, d = 4, 3
    a, h = _p("a", d), _p("h", d)
    x = _v("x", 8, d)
    leaves = [
        Op.make(
            "aff_diag",
            a,
            Op.make("select", x, dim=0, index=2 * t),
        )
        for t in range(T)
    ]
    ir = IR(
        root=Op.make("applyd", _compose("affd_compose", leaves), h),
        inputs=[x],
        input_names={"x"},
        params={},
    )
    av = torch.rand(d, dtype=torch.float64) * 0.9 + 0.05
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(8, d, dtype=torch.float64)
    pv = {"a": av, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    for fused in (False, "eager", True):
        mod = to_batched_scan_module(ir, param_values=pv, fused=fused)
        assert not mod._b_identity
        with torch.no_grad():
            diff = (mod(xv) - gen(xv)).abs().max().item()
        assert diff < 1e-12


def test_fused_b_gather_movedim():
    """Leaf b's sliced along dim=1 of a (d, T) base → movedim path."""
    torch.manual_seed(0)
    T, d = 4, 3
    a, h = _p("a", d), _p("h", d)
    x = _v("x", d, T)
    leaves = [
        Op.make(
            "aff_diag",
            a,
            Op.make("select", x, dim=1, index=t),
        )
        for t in range(T)
    ]
    ir = IR(
        root=Op.make("applyd", _compose("affd_compose", leaves), h),
        inputs=[x],
        input_names={"x"},
        params={},
    )
    av = torch.rand(d, dtype=torch.float64) * 0.9 + 0.05
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(d, T, dtype=torch.float64)
    pv = {"a": av, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    for fused in (False, "eager"):
        mod = to_batched_scan_module(ir, param_values=pv, fused=fused)
        with torch.no_grad():
            diff = (mod(xv) - gen(xv)).abs().max().item()
        assert diff < 1e-12


# ---------------------------------------------------------------------------
#  Compile failure → permanent eager fallback
# ---------------------------------------------------------------------------


def test_fused_compile_wrap_failure_falls_back(monkeypatch):
    """``torch.compile`` raising at wrap time permanently selects the
    eager fused body — output stays exact."""
    torch.manual_seed(0)
    T, d = 8, 4
    ir = _diag_ir(T, d)
    av = torch.rand(d, dtype=torch.float64) * 0.9 + 0.05
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(T, d, dtype=torch.float64)
    pv = {"a": av, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    mod = to_batched_scan_module(ir, param_values=pv, fused=True)

    def _boom(*a, **k):
        raise RuntimeError("no inductor")

    monkeypatch.setattr(torch, "compile", _boom)
    with torch.no_grad():
        out = mod(xv)
    assert mod._fused_compile_failed
    assert (out - gen(xv)).abs().max().item() < 1e-12


def test_fused_compiled_call_failure_falls_back(monkeypatch):
    """A compiled callable that raises on invocation falls back to the
    eager fused body — and stays there on later calls."""
    torch.manual_seed(0)
    T, d = 8, 4
    ir = _diag_ir(T, d)
    av = torch.rand(d, dtype=torch.float64) * 0.9 + 0.05
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(T, d, dtype=torch.float64)
    pv = {"a": av, "h": hv}
    gen = ir_to_torch_module(ir, param_values=pv).eval()
    mod = to_batched_scan_module(ir, param_values=pv, fused=True)

    def _bad_compile(*a, **k):
        def _bad(*args):
            raise RuntimeError("inductor backend dead")

        return _bad

    monkeypatch.setattr(torch, "compile", _bad_compile)
    with torch.no_grad():
        out = mod(xv)
        out2 = mod(xv)
    assert mod._fused_compile_failed and mod._fused_c is None
    assert (out - gen(xv)).abs().max().item() < 1e-12
    assert (out2 - gen(xv)).abs().max().item() < 1e-12


# ---------------------------------------------------------------------------
#  Free-function units (scan_fused.py coverage)
# ---------------------------------------------------------------------------


def test_fused_level_bodies_direct():
    """Direct calls: power-of-2 sizes skip the pad branch entirely."""
    torch.manual_seed(0)
    a = torch.rand(8, 4, dtype=torch.float64)
    b = torch.randn(8, 4, dtype=torch.float64)
    h = torch.randn(4, dtype=torch.float64)
    out = fused_diag_levels(a, b, h)
    # serial fold of the same product
    hh = h.clone()
    for t in range(7, -1, -1):
        hh = a[t] * hh + b[t]
    assert torch.allclose(out, hh)
    # dense body on n=2 (no pad): identity mats → identity applied to h
    h3 = h[:3].contiguous()
    m_seq = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    out_m = fused_dense_levels(m_seq, h3)
    assert torch.allclose(out_m, h3)


def test_fused_levels_single_leaf():
    """n=1: the reduction never enters the loop; pad is a no-op."""
    torch.manual_seed(0)
    a = torch.rand(1, 4, dtype=torch.float64)
    b = torch.randn(1, 4, dtype=torch.float64)
    h = torch.randn(4, dtype=torch.float64)
    out = fused_diag_levels(a, b, h)
    assert torch.allclose(out, a[0] * h + b[0])
    h3 = h[:3].contiguous()
    m = torch.eye(4, dtype=torch.float64).unsqueeze(0)
    m[0, :3, 3] = torch.randn(3, dtype=torch.float64)
    out_m = fused_dense_levels(m, h3)
    assert torch.allclose(out_m, m[0, :3, :3] @ h3 + m[0, :3, 3])


# ---------------------------------------------------------------------------
#  CUDA graph on the fused path
# ---------------------------------------------------------------------------


@pytest.mark.requires_cuda
def test_fused_cuda_graph_replay():
    """capture_cuda_graph works with fused="eager" and fused=True:
    the replayed graph reproduces the fused forward on new inputs."""
    torch.manual_seed(0)
    T, d = 32, 16
    ir = _diag_ir(T, d)
    av = torch.rand(d, dtype=torch.float64) * 0.9 + 0.05
    hv = torch.randn(d, dtype=torch.float64)
    xv = torch.randn(T, d, dtype=torch.float64)
    pv = {"a": av, "h": hv}
    ref = ir_to_torch_module(ir, param_values=pv).eval()

    for fused in ("eager", True):
        mod = to_batched_scan_module(
            ir, param_values=pv, fused=fused
        ).eval()
        mod.cuda()
        xc = xv.cuda()
        mod.capture_cuda_graph(xc)
        assert mod.is_graph_captured
        with torch.no_grad():
            out1 = mod(xc).clone()
            x2 = torch.randn(T, d, device="cuda")
            out2 = mod(x2).clone()
            assert (out1 - ref(xv).cuda()).abs().max().item() < 1e-10
            assert (
                out2 - ref(x2.cpu()).cuda()
            ).abs().max().item() < 1e-10
            mod.drop_cuda_graph()
            out3 = mod(x2)
            assert (out3 - out2).abs().max().item() < 1e-10
