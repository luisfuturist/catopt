"""Executor routing for extracted terms — ``_lower_extracted``.

``optimize_model`` lowers the extracted term through the level-batched
carrier executors when the root is a carrier apply
(``apply``/``applyd``/``om_apply``/``omd_apply[m]``); everything else
goes through ``sink.lower``.  The routing matters because term-level
cost is blind to the lowering (``bench/cost_fidelity.py``).
"""

import torch

from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.torch_bridge import ir_to_torch_module
from catopt_optimize.optimize import _lower_extracted
from catopt_torch.adapters import TorchSink


def _T(shape):
    return TensorType(tuple(shape))


def _P(name, shape):
    return Param(name=name, typ=_T(shape))


def _scan_ir(T=4, d=3):
    """applyd(affd_compose(aff_diag leaves over selects), h) — the
    diagonal-scan carrier form LinearRecurrence extracts to."""
    a = _P("p_a", (T, d))
    b = _P("p_b", (T, d))
    h = Var("h", _T((d,)))
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", a, dim=0, index=t),
            Op.make("select", b, dim=0, index=t),
        )
        for t in range(T)
    ]
    tree = Op.make("affd_compose", leaves[0], leaves[1])
    for leaf in leaves[2:]:
        tree = Op.make("affd_compose", leaf, tree)
    ir = IR(
        root=Op.make("applyd", tree, h),
        inputs=[h],
        input_names=["h"],
        params={"p_a": _T((T, d)), "p_b": _T((T, d))},
    )
    env = {
        "p_a": torch.rand(T, d, dtype=torch.float64),
        "p_b": torch.rand(T, d, dtype=torch.float64),
    }
    return ir, h, env


def _om_ir():
    """om_apply(om_elem(s, v)) — the chunked-attention carrier form."""
    s = _P("p_s", (4, 5))
    v = _P("p_v", (5, 6))
    ir = IR(
        root=Op.make("om_apply", Op.make("om_elem", s, v)),
        inputs=[],
        input_names=[],
        params={"p_s": _T((4, 5)), "p_v": _T((5, 6))},
    )
    env = {
        "p_s": torch.rand(4, 5, dtype=torch.float64),
        "p_v": torch.rand(5, 6, dtype=torch.float64),
    }
    return ir, env


def _omd_ir():
    """omd_apply(omd_elem(s, a, b), h) — the deferred-attention form."""
    s = _P("p_s", (4, 5))
    a = _P("p_a", (5, 3))
    b = _P("p_b", (5, 3))
    h = Var("h", _T((3,)))
    ir = IR(
        root=Op.make("omd_apply", Op.make("omd_elem", s, a, b), h),
        inputs=[h],
        input_names=["h"],
        params={
            "p_s": _T((4, 5)),
            "p_a": _T((5, 3)),
            "p_b": _T((5, 3)),
        },
    )
    env = {
        "p_s": torch.rand(4, 5, dtype=torch.float64),
        "p_a": torch.rand(5, 3, dtype=torch.float64),
        "p_b": torch.rand(5, 3, dtype=torch.float64),
    }
    return ir, h, env


def test_scan_root_routes_to_batched_executor():
    ir, h, env = _scan_ir()
    mod = _lower_extracted(ir.root, ir, env, TorchSink())
    assert getattr(mod, "is_batched", False)
    ref = ir_to_torch_module(ir, param_values=env)
    x = torch.rand(3, dtype=torch.float64)
    assert torch.allclose(mod(x), ref(x), atol=1e-12)


def test_om_root_routes_to_batched_executor():
    ir, env = _om_ir()
    mod = _lower_extracted(ir.root, ir, env, TorchSink())
    ref = ir_to_torch_module(ir, param_values=env)
    assert torch.allclose(mod(), ref(), atol=1e-12)


def test_omd_root_routes_to_batched_executor():
    ir, h, env = _omd_ir()
    mod = _lower_extracted(ir.root, ir, env, TorchSink())
    ref = ir_to_torch_module(ir, param_values=env)
    x = torch.rand(3, dtype=torch.float64)
    assert torch.allclose(mod(x), ref(x), atol=1e-12)


def test_plain_term_routes_to_sink_lower():
    x = Var("x", _T((3,)))
    w = _P("p_w", (3,))
    ir = IR(
        root=Op.make("add", x, w),
        inputs=[x],
        input_names=["x"],
        params={"p_w": _T((3,))},
    )
    env = {"p_w": torch.rand(3, dtype=torch.float64)}
    mod = _lower_extracted(ir.root, ir, env, TorchSink())
    assert not getattr(mod, "is_batched", False)
    xin = torch.rand(3, dtype=torch.float64)
    assert torch.allclose(mod(xin), xin + env["p_w"], atol=1e-12)


def test_optimize_model_compile_delivers_fused_module():
    """compile=True wraps the routed module in torch.compile and
    verifies output; stats records what ran."""
    from catopt.optimize import optimize_model

    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8)
    ).eval()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(m, x, verbose=False, compile=True)
    assert stats["compiled"] is True
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_compile_failure_falls_back(monkeypatch):
    """A compile failure keeps the uncompiled module — the optimized
    term still ships."""
    import catopt_optimize.optimize as O

    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8)
    ).eval()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)

        def boom(mod):
            raise RuntimeError("no compiler")

        monkeypatch.setattr(O.torch, "compile", boom)
        mod, stats = O.optimize_model(m, x, verbose=False, compile=True)
    assert stats["compiled"] is False
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)
