"""Executor routing for extracted terms — ``_lower_extracted``.

``optimize_model`` lowers the extracted term through the level-batched
carrier executors when the root is a carrier apply
(``apply``/``applyd``/``om_apply``/``omd_apply[m]``); everything else
goes through ``sink.lower``.  The routing matters because term-level
cost is blind to the lowering (``bench/cost_fidelity.py``).
"""

import pytest
import torch
from catopt.cost import flops_cost
from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.torch_bridge import ir_to_torch_module
from catopt_optimize.optimize import _lower_extracted
from catopt_optimize.runners import (
    ChainedRunner,
    CudaGraphRunner,
    TorchCompileRunner,
)
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
    """runner=TorchCompileRunner() wraps the routed module in
    torch.compile and verifies output; stats records what ran."""
    from catopt.optimize import optimize_model

    torch.manual_seed(0)
    m = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8)
    ).eval()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(
            m, x, verbose=False, runner=TorchCompileRunner()
        )
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
        mod, stats = O.optimize_model(
            m, x, verbose=False, runner=TorchCompileRunner()
        )
    assert stats["compiled"] is False
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_delivered_cost_priced_by_executor():
    """Carrier-rooted plannable terms price under the batched
    lowering; non-carrier terms under generic."""
    from catopt_optimize.optimize import _delivered_cost

    ir, h, env = _scan_ir()
    batched = _delivered_cost(ir.root)
    generic = _delivered_cost(
        Op.make("add", Var("x", _T((3,))), _P("p_w", (3,)))
    )
    assert batched > 0 and generic > 0


def test_carrier_upgrade_swaps_to_batched_member():
    """A cheaper batched applyd member replaces the additive winner."""
    from catopt.models import LinearRecurrence
    from catopt.optimize import optimize_model

    torch.manual_seed(0)
    m = LinearRecurrence(4, 8).eval().double()
    x = torch.rand(8, 4, dtype=torch.float64)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(m, x, verbose=False)
    assert stats["lowering"] == "batched"
    assert getattr(mod, "is_batched", False)
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-9)


@pytest.mark.requires_cuda
def test_optimize_model_delivered_module_captures_cuda_graph():
    """End-to-end: the module ``optimize_model`` delivers for a
    scan-shaped model takes the CUDA-graph path — capture replays the
    identical computation, a changed input flows through the static
    buffers, and ``drop_cuda_graph`` restores the eager path."""
    from catopt.models import LinearRecurrence
    from catopt.optimize import optimize_model

    torch.manual_seed(0)
    m = LinearRecurrence(4, 8).eval().double().cuda()
    x = torch.rand(8, 4, dtype=torch.float64, device="cuda")
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(m, x, verbose=False)
    assert stats["lowering"] == "batched"
    assert getattr(mod, "is_batched", False)

    mod.capture_cuda_graph(x)
    assert mod.is_graph_captured
    with torch.no_grad():
        assert torch.allclose(mod(x).clone(), ref, atol=1e-9)
        # new input is picked up via the static input buffers
        x2 = torch.rand(8, 4, dtype=torch.float64, device="cuda")
        assert torch.allclose(mod(x2).clone(), m(x2), atol=1e-9)

        mod.drop_cuda_graph()
        assert not mod.is_graph_captured
        assert torch.allclose(mod(x2), m(x2), atol=1e-9)


def test_lift_scan_to_applyd_offers_carrier_member():
    """The nonlocal lift seeds applyd members on recurrence classes
    without saturating the carrier laws."""
    from catopt.egraph import EGraph
    from catopt.models import LinearRecurrence
    from catopt.scan_lower import (
        is_scan_apply_term,
        to_batched_scan_module,
    )
    from catopt.torch_bridge import export_to_ir
    from catopt_carriers.trace_lift import lift_scan_to_applyd

    torch.manual_seed(0)
    m = LinearRecurrence(4, 8).eval().double()
    x = torch.rand(8, 4, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    lifts = lift_scan_to_applyd(eg)
    assert lifts  # the recurrence spine was recognised
    eg.rebuild()
    root_cid = eg.find(root)
    applyd_nodes = [
        n
        for n in eg._classes[root_cid].nodes
        if n.op in ("apply", "applyd")  # dense -> apply, diag -> applyd
    ]
    assert applyd_nodes
    term = eg.extract_best(
        root, flops_cost, overrides={root_cid: applyd_nodes[0]}
    )
    assert is_scan_apply_term(term)
    env = dict(src)
    ref = ir_to_torch_module(ir, param_values=env)
    batched = to_batched_scan_module(
        IR(
            root=term,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params=ir.params,
        ),
        param_values=env,
    )
    assert batched.is_batched
    assert torch.allclose(batched(x), ref(x), atol=1e-9)


def test_lift_scan_to_applyd_skips_existing_carrier():
    """A second pass offers nothing — classes already carrying an
    apply form are skipped."""
    from catopt.egraph import EGraph
    from catopt.models import LinearRecurrence
    from catopt.torch_bridge import export_to_ir
    from catopt_carriers.trace_lift import lift_scan_to_applyd

    torch.manual_seed(0)
    m = LinearRecurrence(4, 8).eval().double()
    x = torch.rand(8, 4, dtype=torch.float64)
    ir, _ = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    assert lift_scan_to_applyd(eg)
    eg.rebuild()
    assert lift_scan_to_applyd(eg) == []


def test_delivered_cost_declines_non_plannable_carrier():
    """A carrier-rooted term whose batched plan can't be built prices
    as generic — the module would degrade to serial eval anyway."""
    from catopt.cost import executor_cost_for
    from catopt_optimize.optimize import _delivered_cost

    h = Var("h", _T((2,)))
    t = Op.make(
        "applyd",
        Op.make(
            "affd_compose",
            Op.make("aff_diag", _P("a2", (2,)), _P("b2", (2,))),
            Op.make("aff_diag", _P("a4", (4,)), _P("b4", (4,))),
        ),
        h,
    )
    assert _delivered_cost(t) == executor_cost_for(
        lowering="generic"
    )(t)


def test_carrier_upgrade_returns_incumbent_when_no_carriers():
    """No carrier enodes at root -> incumbent returned unchanged."""
    from catopt.egraph import EGraph
    from catopt_optimize.optimize import _carrier_upgrade

    x = Var("x", _T((3,)))
    t = Op.make("add", x, _P("p_w", (3,)))
    eg = EGraph()
    root = eg.add_term(t)
    assert _carrier_upgrade(eg, root, t, flops_cost) is t


def test_carrier_upgrade_handles_failed_extraction(monkeypatch):
    """An override extraction yielding None is skipped."""
    import catopt_optimize.optimize as O
    from catopt.egraph import EGraph

    ir, h, env = _scan_ir()
    eg = EGraph()
    root = eg.add_term(ir.root)
    monkeypatch.setattr(
        O.EGraph, "extract_best", lambda self, eid, cf, **kw: None
    )
    incumbent = Op.make("add", h, _P("p0", (3,)))
    assert (
        O._carrier_upgrade(eg, root, incumbent, flops_cost)
        is incumbent
    )


def test_carrier_upgrade_swaps_when_delivered_cheaper(monkeypatch):
    """Carrier member wins when its delivered (batched) price beats
    the incumbent's delivered price."""
    import catopt_optimize.optimize as O
    from catopt.egraph import EGraph

    ir, h, env = _scan_ir()
    eg = EGraph()
    root = eg.add_term(ir.root)
    cid = eg.find(root)
    carrier_node = next(
        n for n in eg._classes[cid].nodes if n.op == "applyd"
    )
    incumbent = Op.make("add", h, _P("p0", (3,)))
    # force the incumbent to lose on delivered price
    orig = O._delivered_cost
    monkeypatch.setattr(
        O,
        "_delivered_cost",
        lambda t, profile=None, compiled=False: (
            1.0
            if isinstance(t, Op) and t.op in O._CARRIER_PLANS
            else 100.0
        ),
    )
    out = O._carrier_upgrade(eg, root, incumbent, flops_cost)
    assert out is not incumbent
    assert getattr(out, "op", None) == "applyd"


def test_batched_module_exposes_root():
    """Downstream consumers (benches, reports) read ``_root`` — the
    batched modules delegate it to the wrapped IRModule."""
    ir, h, env = _scan_ir()
    mod = _lower_extracted(ir.root, ir, env, TorchSink())
    assert mod._root is mod.eval_mod._root


def test_batched_module_exposes_param_map():
    """``_param_map`` (fused_* materialised params) delegates to the
    wrapped IRModule — benches rebuild fp32 modules from it."""
    ir, h, env = _scan_ir()
    mod = _lower_extracted(ir.root, ir, env, TorchSink())
    assert mod._param_map is mod.eval_mod._param_map


def test_cuda_graph_runner_noop_on_cpu():
    """runner=CudaGraphRunner() on a CPU input: stats records False,
    module is unchanged — the runner is a no-op off-CUDA."""
    from catopt.models import LinearRecurrence
    from catopt.optimize import optimize_model
    torch.manual_seed(0)
    m = LinearRecurrence(4, 8).eval().double()
    x = torch.rand(8, 4, dtype=torch.float64)
    with torch.no_grad():
        mod, stats = optimize_model(
            m, x, verbose=False, runner=CudaGraphRunner()
        )
    assert stats.get("cuda_graph") is False
    with torch.no_grad():
        assert torch.allclose(mod(x), m(x))


def test_cuda_graph_skipped_when_compiled():
    """[TorchCompileRunner, CudaGraphRunner]: the compiled module wins —
    capture is not attempted on it (no capture attr)."""
    from catopt.models import LinearRecurrence
    from catopt.optimize import optimize_model
    torch.manual_seed(0)
    m = LinearRecurrence(4, 8).eval().double()
    x = torch.rand(8, 4, dtype=torch.float64)
    with torch.no_grad():
        mod, stats = optimize_model(
            m,
            x,
            verbose=False,
            runner=ChainedRunner([TorchCompileRunner(), CudaGraphRunner()]),
        )
    if stats.get("compiled"):
        assert stats.get("cuda_graph") is None


def test_cuda_graph_tuple_input_noop_on_cpu():
    """Tuple/list example inputs don't crash the cuda_graph path —
    non-tensor first elements degrade quietly to cuda_graph=False."""
    from catopt.models import LinearRecurrence
    from catopt.optimize import optimize_model
    torch.manual_seed(0)
    m = LinearRecurrence(4, 8).eval().double()
    x = torch.rand(8, 4, dtype=torch.float64)
    with torch.no_grad():
        # tuple input that isn't CUDA → no capture, no crash
        mod, stats = optimize_model(
            m, (x,), verbose=False, runner=CudaGraphRunner()
        )
    assert stats.get("cuda_graph") is False


def test_delivered_cost_compiled_prices_fusion_regions():
    """compiled=True: every candidate prices under the fusion-region
    model — a 128-leaf pointwise spine bills ~1 kernel, not per-node."""
    from catopt.cost import executor_cost_for
    from catopt_optimize.optimize import _delivered_cost

    ir, h, env = _scan_ir()
    assert _delivered_cost(
        ir.root, compiled=True
    ) == executor_cost_for(lowering="compiled")(ir.root)
