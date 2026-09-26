"""Regime-adaptive architecture selection tests.

Covers:
  * ``regime_frontier`` extracting distinct members per cost model on
    ``DiagonalSSM`` (raw spine vs ``applyd`` scan forms).
  * ``RegimeDispatch`` holding ``{regime: (term, executor_module)}``,
    sharing one set of weights, dispatching by regime name.
  * fp64 equivalence of every served form.
  * level-2 certificates for extracted members.
  * graceful degradation when a regime's preferred carrier is
    unreachable, and collapse reporting when regimes agree.
"""

import pytest
import torch

from catopt import rules as R
from catopt.calibrate import (
    PROFILE_DIR_ENV,
    TargetProfile,
    save_profile,
)
from catopt.cost import (
    flops_cost,
    launch_aware_cost,
    roofline_cost,
    roofline_cost_for,
)
from catopt.egraph import EGraph, verify_certificate
from catopt.ir import IR, Op, TensorType, Var, op_repr
from catopt.models.ssm import DiagonalSSM
from catopt.om import OM_LAWS
from catopt.om_lower import is_om_apply_term
from catopt.regime import (
    EXECUTORS,
    Regime,
    RegimeDispatch,
    architecture_label,
    architecture_signature,
    default_regimes,
    footprint_cost,
    regime_dispatch,
    regime_frontier,
)
from catopt.scan_lower import is_scan_apply_term
from catopt.torch_bridge import export_to_ir, ir_to_torch_module

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _ssm_fixture(T=16, D=16, seed=0):
    torch.manual_seed(seed)
    model = DiagonalSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, source = export_to_ir(model, x)
    eg = EGraph()  # level-2 by default
    root = eg.add_term(ir.root)
    stats = eg.run(
        R.SCAN_DIAG_LAWS, root, max_iterations=14, max_nodes=400_000
    )
    return model, x, ir, source, eg, root, stats


@pytest.fixture(scope="module")
def ssm():
    return _ssm_fixture()


def _nested_cat(ts, dim):
    out = ts[0]
    for t in ts[1:]:
        out = Op.make("concat", out, t, dim=dim)
    return out


def _chunked_attention(n_blocks=4, B=2, H=2, T=16, d=8, dv=8, seed=0):
    """Dense chunked attention term → (ir, inputs, eg, root)."""
    torch.manual_seed(seed)
    _K = T // n_blocks
    q = Var("q", TensorType((B, H, T, d)))
    Kv = Var("K", TensorType((B, H, T, d)))
    Vv = Var("V", TensorType((B, H, T, dv)))
    ks = [
        Op.make("chunk", Kv, chunks=n_blocks, dim=-2, index=i)
        for i in range(n_blocks)
    ]
    vs = [
        Op.make("chunk", Vv, chunks=n_blocks, dim=-2, index=i)
        for i in range(n_blocks)
    ]
    kcat, vcat = _nested_cat(ks, -2), _nested_cat(vs, -2)
    scores = Op.make(
        "matmul", q, Op.make("transpose", kcat, arg1=-2, arg2=-1)
    )
    term = Op.make("matmul", Op.make("softmax", scores, arg1=-1), vcat)
    ir = IR(
        root=term,
        inputs=[q, Kv, Vv],
        input_names={"q", "K", "V"},
        params={},
    )
    inputs = {
        "q": torch.randn(B, H, T, d, dtype=torch.float64),
        "K": torch.randn(B, H, T, d, dtype=torch.float64),
        "V": torch.randn(B, H, T, dv, dtype=torch.float64),
    }
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(OM_LAWS, root, max_iterations=20, max_nodes=300_000)
    return ir, inputs, eg, root


# ---------------------------------------------------------------------------
# frontier on DiagonalSSM
# ---------------------------------------------------------------------------


class TestFrontierSSM:
    def test_distinct_architectures(self, ssm):
        _m, _x, ir, _src, eg, root, _st = ssm
        frontier = regime_frontier(
            eg,
            root,
            {
                "sequential": (flops_cost, "generic"),
                "launch": (launch_aware_cost, "auto"),
                "roofline": (roofline_cost, "auto"),
                "parallel": Regime(
                    "parallel",
                    extract_fn=EGraph.extract_min_depth,
                    executor="auto",
                ),
            },
            ir=ir,
        )
        seq = frontier["sequential"]
        assert seq.signature[0] == "tensor"
        assert seq.term.op == "add"  # raw sequential spine
        for name in ("launch", "roofline", "parallel"):
            ch = frontier[name]
            assert ch.executor == "scan"  # auto-resolved
            assert is_scan_apply_term(ch.term)
            assert ch.native and not ch.degraded
        # at least the spine vs the scan forms must differ
        assert frontier.n_architectures >= 2
        assert seq.signature != frontier["parallel"].signature
        # every term is a real member of the root class
        for ch in frontier:
            assert eg.find(eg.add_term(ch.term)) == eg.find(root)

    def test_flops_prefers_raw_spine(self, ssm):
        _m, _x, ir, _src, eg, root, _st = ssm
        frontier = regime_frontier(
            eg, root, {"work": (flops_cost, "generic")}, ir=ir
        )
        ch = frontier["work"]
        census_roots = op_repr(ch.term)
        assert "applyd" not in census_roots
        assert "aff_diag" not in census_roots

    def test_certificates_replay(self, ssm):
        _m, _x, ir, _src, eg, root, _st = ssm
        frontier = regime_frontier(
            eg,
            root,
            {
                "sequential": (flops_cost, "generic"),
                "parallel": Regime(
                    "parallel",
                    extract_fn=EGraph.extract_min_depth,
                    executor="scan",
                ),
            },
            ir=ir,
            src_term=ir.root,
        )
        for name in frontier.names:
            cert = frontier.certificate(name)
            out = verify_certificate(ir.root, cert)
            assert op_repr(out) == op_repr(frontier[name].term), name

    def test_report_smoke(self, ssm):
        _m, _x, ir, _src, eg, root, _st = ssm
        frontier = regime_frontier(eg, root, ir=ir)
        rep = frontier.report()
        assert "sequential" in rep and "bounded_memory" in rep
        assert "scan[" in rep  # scan arch labelled
        assert "tensor[add]" in rep


# ---------------------------------------------------------------------------
# dispatch on DiagonalSSM
# ---------------------------------------------------------------------------


class TestDispatchSSM:
    def test_end_to_end_and_outputs(self):
        model, x, _ir, _source, _eg, _root, _st = _ssm_fixture()
        disp = regime_dispatch(
            model,
            x,
            regimes=[
                Regime(
                    "sequential", cost_fn=flops_cost, executor="generic"
                ),
                Regime(
                    "parallel",
                    extract_fn=EGraph.extract_min_depth,
                    executor="auto",
                ),
            ],
        )
        assert disp.verification is not None
        for name, v in disp.verification.items():
            assert v["ok"], f"{name}: {v}"
            assert v["max_abs_diff"] < 1e-9
        # the parallel form actually engages the batched scan executor
        assert disp.frontier["parallel"].engaged

    def test_shared_weights(self):
        model, x, *_ = _ssm_fixture()
        disp = regime_dispatch(
            model,
            x,
            regimes=[
                Regime("a", cost_fn=flops_cost, executor="generic"),
                Regime(
                    "b",
                    extract_fn=EGraph.extract_min_depth,
                    executor="auto",
                ),
            ],
        )
        ma = disp.executor_module("a")
        mb = disp.executor_module("b")
        pma = RegimeDispatch._param_map_of(ma)
        pmb = RegimeDispatch._param_map_of(mb)
        shared = [p for p in pma if not p.startswith("fused_")]
        assert shared
        for pname in shared:
            assert pma[pname] is pmb[pname]

    def test_regime_kwarg_and_errors(self):
        model, x, *_ = _ssm_fixture()
        disp = regime_dispatch(
            model,
            x,
            regimes=[
                Regime("a", cost_fn=flops_cost, executor="generic"),
                Regime(
                    "b",
                    extract_fn=EGraph.extract_min_depth,
                    executor="auto",
                ),
            ],
            default="a",
        )
        with torch.no_grad():
            ya = disp(x, regime="a")
            yb = disp(x, regime="b")
            yd = disp(x)
        assert torch.equal(yd, ya)
        assert torch.allclose(ya, yb, atol=1e-9)
        with pytest.raises(KeyError):
            disp(x, regime="nope")
        disp.set_regime("b")
        with torch.no_grad():
            assert torch.equal(disp(x), yb)
        with pytest.raises(KeyError):
            disp.set_regime("nope")

    def test_entries_mapping(self):
        model, x, *_ = _ssm_fixture()
        disp = regime_dispatch(
            model,
            x,
            regimes=[
                Regime("a", cost_fn=flops_cost, executor="generic"),
            ],
        )
        entries = disp.entries
        term, _mod = entries["a"]
        assert op_repr(term) == op_repr(disp.frontier["a"].term)


# ---------------------------------------------------------------------------
# chunked attention: dense vs streamed online-softmax
# ---------------------------------------------------------------------------


class TestFrontierAttention:
    def test_dense_vs_om(self):
        ir, _inputs, eg, root = _chunked_attention()
        frontier = regime_frontier(
            eg,
            root,
            {
                "dense": (flops_cost, "generic"),
                "prefill": (roofline_cost, "om_batched"),
                "bounded_memory": (flops_cost, "om_streaming"),
            },
            ir=ir,
        )
        dense = frontier["dense"]
        assert dense.signature[0] == "tensor"
        # the streaming regime must serve a *composed* om tree even
        # though the dense form is cheaper under flops
        bm = frontier["bounded_memory"]
        assert bm.forced and is_om_apply_term(bm.term)
        assert bm.native and not bm.degraded
        assert bm.cost > bm.cost_best  # premium recorded
        assert frontier.n_architectures >= 2

    def test_dispatch_correctness(self):
        ir, inputs, eg, root = _chunked_attention()
        frontier = regime_frontier(
            eg,
            root,
            {
                "dense": (flops_cost, "generic"),
                "stream": (flops_cost, "om_streaming"),
            },
            ir=ir,
        )
        disp = frontier.build(param_values=None)
        ref = ir_to_torch_module(ir)(
            inputs["q"], inputs["K"], inputs["V"]
        )
        for name in disp.regimes:
            with torch.no_grad():
                y = disp(
                    inputs["q"], inputs["K"], inputs["V"], regime=name
                )
            assert (y - ref).abs().max().item() < 1e-9, name
        assert disp.frontier["stream"].engaged

    def test_roofline_picks_lifted(self):
        ir, _inputs, eg, root = _chunked_attention()
        frontier = regime_frontier(
            eg, root, {"prefill": (roofline_cost, "om_batched")}, ir=ir
        )
        ch = frontier["prefill"]
        assert is_om_apply_term(ch.term)
        assert not ch.forced  # cost-best, not forced


# ---------------------------------------------------------------------------
# degradation / collapse honesty
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_unreachable_carrier_flagged(self, ssm):
        """om_streaming on a pure SSM e-graph: om carrier unreachable."""
        _m, x, ir, src, eg, root, _st = ssm
        frontier = regime_frontier(
            eg,
            root,
            {
                "decode": (launch_aware_cost, "om_streaming"),
            },
            ir=ir,
        )
        ch = frontier["decode"]
        assert ch.degraded
        assert not ch.native and not ch.carrier_present
        # still served — executor falls back to serial eval
        disp = frontier.build(param_values=src)
        model_out = _m(x)
        with torch.no_grad():
            y = disp(x, regime="decode")
        assert (y - model_out).abs().max().item() < 1e-9
        assert not ch.engaged  # serial fallback confirmed

    def test_no_prefer_flag(self, ssm):
        _m, _x, ir, _src, eg, root, _st = ssm
        frontier = regime_frontier(
            eg,
            root,
            {
                "r": Regime(
                    "r",
                    cost_fn=flops_cost,
                    executor="scan",
                    prefer_executor=False,
                ),
            },
            ir=ir,
        )
        ch = frontier["r"]
        assert not ch.forced
        assert ch.degraded  # flops picks raw spine
        assert op_repr(ch.term) == op_repr(ch.cost_term)

    def test_collapse_reported(self):
        ir, _inputs, eg, root = _chunked_attention()
        frontier = regime_frontier(
            eg,
            root,
            {
                "a": (flops_cost, "generic"),
                "b": (flops_cost, "generic"),
            },
            ir=ir,
        )
        collapsed = frontier.collapsed()
        assert collapsed
        assert set(collapsed[0][1]) == {"a", "b"}
        assert "collapsed" in frontier.report()


# ---------------------------------------------------------------------------
# target profiles (catopt.calibrate) wired into regimes
# ---------------------------------------------------------------------------

# Toy profiles spanning the roofline space.  GPUISH is a fat discrete
# GPU; CPUISH a modest host; EDGE_SRAM a compute-starved accelerator
# with huge on-package bandwidth (the corner where the om tree's extra
# FLOPs outweigh the dense form's memory traffic).
GPUISH = TargetProfile(
    "gpu-ish",
    tflops=100.0,
    gbps=2000.0,
    launch_us=2.0,
    device="cuda:0",
    measured_at="t",
)
CPUISH = TargetProfile(
    "cpu-ish",
    tflops=0.2,
    gbps=20.0,
    launch_us=30.0,
    device="cpu",
    measured_at="t",
)
EDGE_SRAM = TargetProfile(
    "edge-sram",
    tflops=0.05,
    gbps=200.0,
    launch_us=10.0,
    device="cpu",
    measured_at="t",
)


def _profile_terms():
    """A compute-bound term and a memory-bound one."""
    x = Var("x", TensorType((256, 256)))
    w = Var("w", TensorType((256, 256)))
    a = Var("a", TensorType((512, 512)))
    b = Var("b", TensorType((512, 512)))
    return Op.make("matmul", x, w), Op.make("add", a, b)


class TestProfiles:
    def test_profile_wires_cost_fn(self):
        r = Regime("p", profile=GPUISH)
        fn = roofline_cost_for(GPUISH)
        assert r.profile is GPUISH
        for t in _profile_terms():
            assert r.cost_fn(t) == pytest.approx(fn(t))
        # and the constants actually differ from the built-in default
        mm, _ = _profile_terms()
        assert r.cost_fn(mm) != pytest.approx(roofline_cost(mm))

    def test_explicit_cost_fn_overrides_profile(self):
        r = Regime("p", profile=GPUISH, cost_fn=flops_cost)
        assert r.cost_fn is flops_cost
        assert r.profile is GPUISH  # kept for provenance
        # no profile → unchanged behaviour
        assert Regime("q").cost_fn is None
        r2 = Regime("q", extract_fn=EGraph.extract_min_depth)
        assert r2.cost_fn is None and r2.profile is None

    def test_profile_dict_spec(self):
        spec = {"tflops": 100.0, "gbps": 2000.0, "launch_us": 2.0}
        r = Regime("p", profile=spec)
        fn = roofline_cost_for(spec)
        mm, ew = _profile_terms()
        assert r.cost_fn(mm) == pytest.approx(fn(mm))
        assert r.cost_fn(ew) == pytest.approx(fn(ew))

    def test_profile_by_name_loads(self, tmp_path, monkeypatch):
        monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path))
        save_profile(EDGE_SRAM)
        r = Regime("p", profile="edge-sram")
        assert r.profile == EDGE_SRAM
        mm, _ = _profile_terms()
        assert r.cost_fn(mm) == pytest.approx(
            roofline_cost_for(EDGE_SRAM)(mm)
        )
        with pytest.raises(FileNotFoundError):
            Regime("q", profile="no-such-target")

    def test_profiles_flip_extraction_attention(self):
        """Same e-graph, two targets, two different served members.

        Chunked attention: the compute-starved EDGE_SRAM profile prices
        the om tree's extra FLOPs above the dense form's memory traffic
        and picks the raw ``matmul(softmax(matmul))``; the GPU profile
        does the reverse and picks the ``om_apply`` tree.
        """
        ir, _inputs, eg, root = _chunked_attention()
        frontier = regime_frontier(
            eg,
            root,
            [
                Regime("edge", profile=EDGE_SRAM, executor="generic"),
                Regime("gpu", profile=GPUISH, executor="om_batched"),
            ],
            ir=ir,
        )
        edge, gpu = frontier["edge"], frontier["gpu"]
        assert edge.term.op == "matmul"  # dense form
        assert is_om_apply_term(gpu.term)  # lifted om tree
        assert edge.signature != gpu.signature
        # sanity: each choice is exactly what direct extraction under
        # that profile's roofline yields
        assert op_repr(edge.term) == op_repr(
            eg.extract_best(root, roofline_cost_for(EDGE_SRAM))
        )
        assert op_repr(gpu.term) == op_repr(
            eg.extract_best(root, roofline_cost_for(GPUISH))
        )

    def test_profiles_flip_ssm_member(self, ssm):
        """CPU-ish vs GPU-ish profiles pick different applyd balances."""
        _m, _x, ir, _src, eg, root, _st = ssm
        frontier = regime_frontier(
            eg,
            root,
            [
                Regime("cpu", profile=CPUISH, executor="scan"),
                Regime("gpu", profile=GPUISH, executor="scan"),
            ],
            ir=ir,
        )
        cpu, gpu = frontier["cpu"], frontier["gpu"]
        assert is_scan_apply_term(cpu.term)
        assert is_scan_apply_term(gpu.term)
        # different members of the same e-class: different amounts of
        # affd_compose reassociation are worthwhile per target
        assert op_repr(cpu.term) != op_repr(gpu.term)
        assert op_repr(cpu.term) == op_repr(
            eg.extract_best(root, roofline_cost_for(CPUISH))
        )
        assert op_repr(gpu.term) == op_repr(
            eg.extract_best(root, roofline_cost_for(GPUISH))
        )

    def test_frontier_profiles_kwarg(self):
        ir, _inputs, eg, root = _chunked_attention()
        frontier = regime_frontier(
            eg,
            root,
            {
                # dict spec without cost_fn: profile supplies the model
                "edge": {"executor": "generic"},
                # explicit profile on the Regime beats the map entry
                "gpu": Regime(
                    "gpu", profile=GPUISH, executor="generic"
                ),
            },
            ir=ir,
            profiles={"edge": EDGE_SRAM, "gpu": EDGE_SRAM},
        )
        edge, _gpu = frontier["edge"], frontier["gpu"]
        assert edge.term.op == "matmul"  # EDGE_SRAM attached
        reg_by_name = {r.name: r for r in frontier.regimes}
        assert reg_by_name["edge"].profile == EDGE_SRAM
        assert reg_by_name["gpu"].profile is GPUISH  # not overridden

    def test_regime_dispatch_calibrate_profile(self):
        """calibrate=<profile> fills profile-less regimes, no measuring."""
        model, x, _ir, _src, _eg, _root, _st = _ssm_fixture()
        disp = regime_dispatch(
            model,
            x,
            regimes=[
                Regime("roof"),  # no cost model at all
                Regime("work", cost_fn=flops_cost, executor="generic"),
            ],
            calibrate=GPUISH,
        )
        regs = {r.name: r for r in disp.frontier.regimes}
        assert regs["roof"].profile is GPUISH
        assert regs["work"].profile is GPUISH  # provenance recorded
        assert regs["work"].cost_fn is flops_cost  # explicit still wins
        # the filled profile drove "roof"'s extraction: on the SSM the
        # roofline objective lifts the recurrence into applyd scan form
        roof = disp.frontier["roof"]
        assert is_scan_apply_term(roof.term)
        assert roof.executor == "scan"  # auto-resolved
        for name, v in disp.verification.items():
            assert v["ok"], f"{name}: {v}"


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_footprint_cost(self):
        _m, _x, _ir, _src, eg, root, _st = _ssm_fixture()
        dense = eg.extract_best(root, flops_cost)
        v = footprint_cost(dense)
        assert isinstance(v, float) and v > 0

    def test_default_regimes_shape(self):
        regs = default_regimes()
        names = [r.name for r in regs]
        assert "sequential" in names and "bounded_memory" in names
        for r in regs:
            assert r.executor in EXECUTORS or r.executor == "auto"
            assert r.cost_fn is not None or r.extract_fn is not None

    def test_signature_labels(self, ssm):
        _m, _x, _ir, _src, eg, root, _st = ssm
        dense = eg.extract_best(root, flops_cost)
        scan = eg.extract_min_depth(root)
        assert architecture_signature(dense)[0] == "tensor"
        assert architecture_signature(scan)[0] == "scan"
        assert "scan[" in architecture_label(scan)

    def test_bad_executor_rejected(self, ssm):
        _m, _x, ir, _src, eg, root, _st = ssm
        with pytest.raises(KeyError):
            regime_frontier(
                eg, root, {"x": (flops_cost, "no_such_executor")}, ir=ir
            )
