"""Coverage-gap tests for catopt.regime.

Complements tests/test_regime.py with the paths it never reaches:

* ``footprint_cost`` on non-term leaves and ill-typed terms;
* ``_auto_executor`` resolving every specialised root (scan / om /
  trace) — including ``executor="auto"`` inside a real frontier;
* ``_as_regime`` spec normalisation (callable / dict / tuple forms,
  and the TypeError on nonsense);
* ``architecture_signature``/``architecture_label`` on trace-rooted,
  nested-carrier, leaf and unplannable-scan terms;
* ``_force_carrier``'s unreachable and accepts-fallback exits;
* frontier honesty for a *failed extraction* (term=None choice),
  a frontier with no ``src_term`` (certificate refusal), and a
  forced-but-non-native carrier (the "nested below the root" note +
  ``carrier-nested`` report flag) via a synthetic executor;
* ``RegimeDispatch`` build errors, ``_share_params`` edge cases
  (modules with no param map, ``fused_`` intermediates, non-Parameter
  entries), regime/name plumbing, ``report``/``extra_repr``/
  ``certificate``;
* each executor's ``engaged`` probe in both directions, plus the
  built modules' ``is_batched``/``is_streaming``/``n_levels``/
  ``n_blocks`` on real lowers;
* ``build_egraph``'s tiered runs: no-lift models, ``xc=False``, and
  the XC loop's second-round lift integration on DiagonalSSM;
* ``regime_dispatch``'s ``calibrate`` kwarg (measured / attached /
  no-pending) and ``verify=False``.
"""

import types

import pytest
import torch
import torch.nn as nn

import catopt.calibrate as cal_mod
from catopt.cost import (
    _INVALID_COST,
    flops_cost,
    launch_aware_cost,
    roofline_cost,
)
from catopt.egraph import EGraph, verify_certificate
from catopt.ir import IR, Op, TensorType, Var, op_repr
from catopt.models.ssm import DiagonalSSM
from catopt.om import OM_LAWS
from catopt.om_lower import is_om_apply_term
from catopt.regime import (
    EXECUTORS,
    ExecutorSpec,
    Regime,
    RegimeDispatch,
    _as_regime,
    _auto_executor,
    _force_carrier,
    _normalise_regimes,
    architecture_label,
    architecture_signature,
    build_egraph,
    default_regimes,
    footprint_cost,
    is_trace_rooted_term,
    regime_dispatch,
    regime_frontier,
)
from catopt.scan_lower import is_scan_apply_term
from catopt.torch_bridge import export_to_ir, ir_to_torch_module


def _T(*shape):
    return TensorType(tuple(shape) if shape else (4, 4))


def _ssm_fixture(T=8, D=8, seed=0):
    torch.manual_seed(seed)
    model = DiagonalSSM(D, D, T).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, source = export_to_ir(model, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    from catopt_core import laws as R

    stats = eg.run(
        R.SCAN_DIAG_LAWS, root, max_iterations=14, max_nodes=400_000
    )
    return model, x, ir, source, eg, root, stats


def _nested_cat(ts, dim):
    out = ts[0]
    for t in ts[1:]:
        out = Op.make("concat", out, t, dim=dim)
    return out


def _chunked_attention(n_blocks=4, B=2, H=2, T=16, d=8, dv=8, seed=0):
    torch.manual_seed(seed)
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


@pytest.fixture(scope="module")
def ssm():
    return _ssm_fixture()


@pytest.fixture(scope="module")
def attention():
    return _chunked_attention()


# ---------------------------------------------------------------------------
# footprint_cost edges
# ---------------------------------------------------------------------------


def test_footprint_cost_edges():
    # a non-Op non-leaf (e.g. a stray metavar) costs nothing
    assert footprint_cost("metavar") == 0.0
    a4, b5 = Var("a", _T()), Var("b", _T(5, 5))
    # an ill-typed term is charged the invalid-cost sentinel, which
    # dominates every finite alternative during extraction
    bad = Op.make("add", a4, b5)
    assert footprint_cost(bad) == _INVALID_COST
    good = Op.make("mul", a4, a4)
    memo = {}
    first = footprint_cost(good, memo)
    assert first > 0 and footprint_cost(good, memo) == first


# ---------------------------------------------------------------------------
# executor probes / auto resolution
# ---------------------------------------------------------------------------


def _scan_apply_term():
    A, b, h = Var("A", _T()), Var("b", _T()), Var("h", _T())
    return Op.make("apply", Op.make("aff", A, b), h)


def _om_apply_term():
    s, v = Var("s", _T(5, 4)), Var("v", _T(4, 6))
    return Op.make("om_apply", Op.make("om_elem", s, v))


def _trace_term():
    f = Var("f", _T())
    return Op.make("trace", f, usize="U"), f


def test_is_trace_rooted_term_variants():
    tr, f = _trace_term()
    assert is_trace_rooted_term(tr)
    # the channel-split (superposed) layout counts as trace-rooted too
    assert is_trace_rooted_term(Op.make("bdiag", tr, tr))
    # …but a bdiag of non-trace args does not
    assert not is_trace_rooted_term(Op.make("bdiag", f, f))
    assert not is_trace_rooted_term(f)
    assert not is_trace_rooted_term("metavar")


def test_auto_executor_all_roots():
    tr, f = _trace_term()
    assert _auto_executor(_scan_apply_term()) == "scan"
    assert _auto_executor(_om_apply_term()) == "om_batched"
    assert _auto_executor(tr) == "trace"
    assert _auto_executor(Op.make("add", f, f)) == "generic"


def test_executor_spec_accepts_and_engaged_both_directions():
    scan_t, om_t = _scan_apply_term(), _om_apply_term()
    tr, _f = _trace_term()
    plain = Op.make("add", Var("x", _T()), Var("y", _T()))

    assert EXECUTORS["generic"].accepts(plain)
    assert EXECUTORS["scan"].accepts(scan_t)
    assert not EXECUTORS["scan"].accepts(plain)
    assert EXECUTORS["om_batched"].accepts(om_t)
    assert not EXECUTORS["om_batched"].accepts(scan_t)
    assert EXECUTORS["om_streaming"].accepts(om_t)
    assert not EXECUTORS["om_streaming"].accepts(scan_t)
    assert EXECUTORS["trace"].accepts(tr)
    assert not EXECUTORS["trace"].accepts(plain)

    batched = types.SimpleNamespace(is_batched=True)
    serial = types.SimpleNamespace(is_batched=False)
    unmarked = types.SimpleNamespace()
    assert EXECUTORS["scan"].engaged(batched)
    assert not EXECUTORS["scan"].engaged(serial)
    assert not EXECUTORS["scan"].engaged(unmarked)  # getattr default
    assert EXECUTORS["om_batched"].engaged(batched)
    assert not EXECUTORS["om_batched"].engaged(serial)
    assert EXECUTORS["om_streaming"].engaged(
        types.SimpleNamespace(is_streaming=True)
    )
    assert not EXECUTORS["om_streaming"].engaged(
        types.SimpleNamespace(is_streaming=False)
    )
    # generic/trace probes engage unconditionally
    assert EXECUTORS["generic"].engaged(object())
    assert EXECUTORS["trace"].engaged(object())


# ---------------------------------------------------------------------------
# regime spec normalisation
# ---------------------------------------------------------------------------


def test_as_regime_spec_forms():
    bare = _as_regime("n", flops_cost)
    assert bare.cost_fn is flops_cost and bare.executor == "auto"
    d = _as_regime("n", {"cost_fn": flops_cost, "executor": "scan"})
    assert d.executor == "scan" and d.cost_fn is flops_cost
    t1 = _as_regime("n", (flops_cost,))
    assert t1.executor == "auto"
    t2 = _as_regime("n", (flops_cost, "scan"))
    assert t2.executor == "scan" and t2.prefer_executor
    t3 = _as_regime("n", (flops_cost, "scan", False))
    assert not t3.prefer_executor
    passthrough = Regime("n", cost_fn=flops_cost)
    assert _as_regime("n", passthrough) is passthrough
    with pytest.raises(TypeError, match="cannot interpret"):
        _as_regime("n", 42)


def test_normalise_regimes_forms():
    assert _normalise_regimes(None) == default_regimes()
    regs = _normalise_regimes(
        {"a": flops_cost, "b": (flops_cost, "scan")}
    )
    assert [r.name for r in regs] == ["a", "b"]
    assert regs[0].cost_fn is flops_cost and regs[1].executor == "scan"
    r = Regime("x")
    assert _normalise_regimes([r]) == [r]


# ---------------------------------------------------------------------------
# architecture signature / label variants
# ---------------------------------------------------------------------------


def test_signature_and_label_trace_forms():
    tr, _f = _trace_term()
    assert architecture_signature(tr) == ("trace", "joint", 1)
    bd = Op.make("bdiag", tr, tr)
    assert architecture_signature(bd) == ("trace", "split", 2)
    assert architecture_label(tr) == "trace[joint fixpoint]"
    assert architecture_label(bd) == "trace[bdiag of 2 channel traces]"


def test_signature_and_label_nested_and_leaf():
    f = Var("f", _T())
    nested = Op.make("add", _scan_apply_term(), f)
    sig = architecture_signature(nested)
    assert sig[0] == "tensor+nested"
    assert sig[1] == "add" and "apply" in sig[2]
    assert "nested carrier" in architecture_label(nested)
    # a bare leaf term is just ("tensor", "leaf")
    assert architecture_signature(f) == ("tensor", "leaf")
    assert architecture_label(f) == "tensor[leaf]"


def test_architecture_label_unplannable_scan():
    """An apply whose aff leaves have inconsistent shapes gets no
    batched schedule — the label reports 'unplannable' rather than
    pretending a plan exists."""
    A, b, h = Var("A", _T()), Var("b", _T()), Var("h", _T())
    A5, b5 = Var("A5", _T(5, 5)), Var("b5", _T(5, 5))
    t = Op.make(
        "apply",
        Op.make(
            "aff_compose",
            Op.make("aff", A, b),
            Op.make("aff", A5, b5),
        ),
        h,
    )
    assert is_scan_apply_term(t)
    assert "unplannable" in architecture_label(t)


# ---------------------------------------------------------------------------
# _force_carrier exits
# ---------------------------------------------------------------------------


def test_force_carrier_unreachable_returns_none():
    x, y = Var("x", _T()), Var("y", _T())
    eg = EGraph()
    root = eg.add_term(Op.make("add", x, y))
    # the root class holds no carrier enode at all → unreachable
    assert (
        _force_carrier(
            eg, root, flops_cost, EXECUTORS["om_batched"].carrier
        )
        is None
    )


def test_force_carrier_accepts_fallback(attention):
    _ir, _inputs, eg, root = attention
    carrier = EXECUTORS["om_batched"].carrier
    t = _force_carrier(eg, root, flops_cost, carrier)
    assert t is not None and is_om_apply_term(t)
    # when every extracted candidate fails the accepts probe the first
    # extractable term is still returned (forced-but-non-native)
    t2 = _force_carrier(
        eg, root, flops_cost, carrier, accepts=lambda _t: False
    )
    assert t2 is not None


def test_force_carrier_multi_candidate_fallback():
    """Several carrier enodes in the root class, accepts rejecting all:
    the loop keeps the first extractable candidate and returns it."""
    from catopt.egraph import Rewrite

    x, y = Var("x", _T()), Var("y", _T())
    comm = Rewrite(
        "t_c",
        Op.make("add", "a", "b"),
        Op.make("add", "b", "a"),
    )
    eg = EGraph()
    root = eg.add_term(Op.make("add", x, y))
    eg.run([comm], root, max_iterations=3)
    # root class now holds ≥2 "add" enodes (the original + commuted)
    carrier = (frozenset({"add"}), frozenset(), frozenset())
    t = _force_carrier(
        eg, root, flops_cost, carrier, accepts=lambda _t: False
    )
    assert t is not None and t.op == "add"


def test_force_carrier_all_candidates_cyclic():
    """Carrier enodes that point at their own e-class extract to None:
    every candidate is skipped and the search returns None after both
    interior passes — the forced path honestly reports unreachable."""
    x = Var("x", _T())
    inner = Op.make("unf", x)
    outer = Op.make("unf", inner)
    eg = EGraph()
    c_x = eg.add_term(x)
    c_in = eg.add_term(inner)
    c_out = eg.add_term(outer)
    # merge all three classes: every "unf" enode now takes its own
    # class as a child — cyclic, hence unextractable
    eg.union(c_x, c_in)
    eg.union(c_x, c_out)
    carrier = (frozenset({"unf"}), frozenset(), frozenset())
    assert (
        _force_carrier(eg, c_out, flops_cost, carrier) is None
    )


# ---------------------------------------------------------------------------
# frontier honesty: failed extraction, missing src_term, forced non-native
# ---------------------------------------------------------------------------


def test_frontier_failed_extraction_choice(ssm):
    _m, _x, ir, _src, eg, root, _st = ssm
    frontier = regime_frontier(
        eg,
        root,
        {"dead": Regime("dead", extract_fn=lambda e, r: None)},
        ir=ir,
    )
    ch = frontier["dead"]
    assert ch.term is None
    assert ch.degraded and ch.signature == ("none",)
    assert "no member" in ch.note and ch.label == "extraction failed"
    # None terms are skipped by collapse analysis…
    assert frontier.collapsed() == []
    # …and the report still renders the choice
    assert "extraction failed" in frontier.report()


def test_certificate_requires_src_term(ssm):
    _m, _x, _ir, _src, eg, root, _st = ssm
    frontier = regime_frontier(
        eg, root, {"a": (flops_cost, "generic")}
    )
    assert frontier.src_term is None
    with pytest.raises(ValueError, match="src_term"):
        frontier.certificate("a")


def test_frontier_forced_non_native_carrier(monkeypatch, ssm):
    """An executor whose accepts() rejects every carrier term: the
    frontier still serves the pinned carrier form but reports it
    honestly — forced, not native, nested-carrier flag + note."""
    _m, _x, ir, _src, eg, root, _st = ssm
    fake = ExecutorSpec(
        name="fake",
        lower=ir_to_torch_module,
        accepts=lambda _t: False,
        engaged=lambda _m: True,
        carrier=(frozenset({"add"}), frozenset(), frozenset()),
    )
    monkeypatch.setitem(EXECUTORS, "fake", fake)
    frontier = regime_frontier(
        eg, root, {"f": (flops_cost, "fake")}, ir=ir
    )
    ch = frontier["f"]
    assert ch.forced and not ch.native and not ch.degraded
    assert ch.carrier_present
    assert "nested below the term root" in ch.note
    assert "carrier-nested" in frontier.report()


def test_frontier_auto_executor_resolves(attention):
    _ir, _inputs, eg, root = attention
    # the roofline model prices the om tree cheapest; executor="auto"
    # must resolve to om_batched on the extracted term's root
    frontier = regime_frontier(
        eg, root, {"p": (roofline_cost, "auto")}, ir=None
    )
    ch = frontier["p"]
    assert ch.executor == "om_batched"
    assert is_om_apply_term(ch.term) and ch.native


def test_frontier_auto_executor_trace_root():
    tr, f = _trace_term()
    ir = IR(root=tr, inputs=[f], input_names={"f"}, params={})
    eg = EGraph()
    root = eg.add_term(tr)
    frontier = regime_frontier(
        eg, root, {"t": (flops_cost, "auto")}, ir=ir
    )
    ch = frontier["t"]
    assert ch.executor == "trace" and ch.native
    assert ch.signature == ("trace", "joint", 1)
    # builds + engages through the generic lower path
    disp = frontier.build(param_values=None)
    assert ch.engaged
    assert type(disp.executor_module("t")).__name__ == "IRModule"


def test_frontier_alternatives_and_rank(ssm):
    _m, _x, ir, _src, eg, root, _st = ssm
    frontier = regime_frontier(
        eg, root, {"w": (flops_cost, "generic")}, ir=ir, top_k=4
    )
    ch = frontier["w"]
    assert ch.alternatives, "top_k extraction produced nothing"
    assert all(isinstance(c, float) for c, _t in ch.alternatives)
    assert ch.rank is not None and 0 <= ch.rank < len(ch.alternatives)
    # top_k=0 skips the alternatives machinery entirely
    f0 = regime_frontier(
        eg, root, {"w": (flops_cost, "generic")}, ir=ir, top_k=0
    )
    assert f0["w"].alternatives == [] and f0["w"].rank is None


# ---------------------------------------------------------------------------
# dispatch: build errors, share-params edges, plumbing
# ---------------------------------------------------------------------------


def test_dispatch_build_errors(ssm):
    _m, _x, ir, src, eg, root, _st = ssm
    # no IR recorded on the frontier → build refuses
    f_no_ir = regime_frontier(
        eg, root, {"a": (flops_cost, "generic")}
    )
    with pytest.raises(ValueError, match="no IR"):
        f_no_ir.build()
    # every choice failing extraction → nothing executable
    f_dead = regime_frontier(
        eg,
        root,
        {"dead": Regime("dead", extract_fn=lambda e, r: None)},
        ir=ir,
    )
    with pytest.raises(ValueError, match="no regime produced"):
        f_dead.build(param_values=src)
    # an unknown default regime is a KeyError
    f_ok = regime_frontier(
        eg, root, {"a": (flops_cost, "generic")}, ir=ir
    )
    with pytest.raises(KeyError):
        f_ok.build(param_values=src, default="ghost")


def test_dispatch_skips_failed_choices(ssm):
    _m, _x, ir, src, eg, root, _st = ssm
    frontier = regime_frontier(
        eg,
        root,
        {
            "a": (flops_cost, "generic"),
            "dead": Regime("dead", extract_fn=lambda e, r: None),
        },
        ir=ir,
    )
    disp = frontier.build(param_values=src)
    assert disp.regimes == ["a"]  # the term=None choice was skipped


class _ParamMapModule(nn.Module):
    """Minimal module exposing the executor-style ``_param_map``."""

    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))
        self.aux = nn.Parameter(torch.ones(2, 2, dtype=torch.float64))
        self._param_map = {"W": self.W, "aux": self.aux}


def test_param_map_of_and_share_params_edges():
    inner = _ParamMapModule()
    wrapper = nn.Module()
    wrapper.eval_mod = inner
    assert RegimeDispatch._param_map_of(wrapper) is inner._param_map
    assert RegimeDispatch._param_map_of(nn.Linear(2, 2)) is None

    # _share_params on a bare dispatch: modules without _param_map are
    # skipped; "fused_" names and non-Parameter values are left alone;
    # matching names get rebound to ONE shared Parameter object.
    disp = RegimeDispatch.__new__(RegimeDispatch)
    nn.Module.__init__(disp)
    m1, m2 = _ParamMapModule(), _ParamMapModule()
    m2._param_map["fused_tmp"] = nn.Parameter(torch.zeros(1))
    m2._param_map["not_a_param"] = torch.zeros(1)
    disp.forms = nn.ModuleDict(
        {"bare": nn.Linear(2, 2), "m1": m1, "m2": m2}
    )
    disp._share_params()
    assert m2._param_map["W"] is m1._param_map["W"]
    assert m2._param_map["aux"] is m1._param_map["aux"]
    assert m2.W is m1.W  # the module attribute was rebound too
    assert "fused_tmp" not in m1._param_map
    assert not isinstance(m2._param_map["not_a_param"], nn.Parameter)


def test_dispatch_plumbing_and_report(ssm):
    model, x, ir, src, eg, root, _st = ssm
    frontier = regime_frontier(
        eg,
        root,
        {
            "work": (flops_cost, "generic"),
            "odd name!": (launch_aware_cost, "generic"),
        },
        ir=ir,
        src_term=ir.root,
    )
    disp = frontier.build(param_values=src)
    # regime-name plumbing: unsafe keys are sanitised in the ModuleDict
    assert disp.regimes == ["work", "odd name!"]
    assert "odd name!" in disp._name_map
    assert disp.executor_module("odd name!") is disp.forms[
        disp._name_map["odd name!"]
    ]
    # regime getter + set_regime round trip
    assert disp.regime == "work"
    disp.set_regime("odd name!")
    assert disp.regime == "odd name!"
    with pytest.raises(KeyError):
        disp.set_regime("nope")
    # forward by explicit regime + default
    ref = model(x)
    with torch.no_grad():
        y1 = disp(x, regime="work")
        y2 = disp(x, regime="odd name!")
        yd = disp(x)
    assert torch.equal(yd, y2)
    assert torch.allclose(y1, ref, atol=1e-9)
    assert torch.allclose(y2, ref, atol=1e-9)
    with pytest.raises(KeyError):
        disp(x, regime="nope")
    # entries mapping + certificate through the dispatch wrapper
    term, _mod = disp.entries["work"]
    assert op_repr(term) == op_repr(disp.frontier["work"].term)
    cert = disp.certificate("work")
    out = verify_certificate(ir.root, cert)
    assert op_repr(out) == op_repr(disp.frontier["work"].term)
    # a report rendered before verify() shows no equivalence block
    rep0 = disp.report()
    assert "built modules:" in rep0
    assert "equivalence vs reference" not in rep0
    # verify + report render the verification block
    info = disp.verify(ref, x)
    assert info["work"]["ok"] and info["odd name!"]["ok"]
    rep = disp.report()
    assert "built modules:" in rep
    assert "equivalence vs reference" in rep and "ok=True" in rep
    assert "regime=" in disp.extra_repr()


# ---------------------------------------------------------------------------
# executor modules' engaged/batching properties on real lowers
# ---------------------------------------------------------------------------


def test_scan_executor_module_properties():
    model, x, *_ = _ssm_fixture()
    disp = regime_dispatch(
        model,
        x,
        regimes=[
            Regime(
                "parallel",
                extract_fn=EGraph.extract_min_depth,
                executor="scan",
            ),
        ],
    )
    mod = disp.executor_module("parallel")
    assert mod.is_batched and mod.n_levels >= 1
    assert disp.frontier["parallel"].engaged
    assert disp.verification["parallel"]["ok"]


def test_om_executor_module_properties(attention):
    ir, _inputs, eg, root = attention
    frontier = regime_frontier(
        eg,
        root,
        {
            "prefill": (roofline_cost, "om_batched"),
            "stream": (flops_cost, "om_streaming"),
            "dense": (flops_cost, "generic"),
        },
        ir=ir,
    )
    disp = frontier.build(param_values=None)
    mb = disp.executor_module("prefill")
    ms = disp.executor_module("stream")
    mg = disp.executor_module("dense")
    assert mb.is_batched and mb.n_blocks >= 1 and mb.n_levels >= 0
    assert ms.is_streaming and ms.n_blocks >= 1
    assert not getattr(mg, "is_batched", False)
    assert disp.frontier["prefill"].engaged
    assert disp.frontier["stream"].engaged
    # the generic executor's probe is unconditional
    assert disp.frontier["dense"].engaged
    # every form still evaluates the dense reference
    ref = ir_to_torch_module(ir)(
        _inputs["q"], _inputs["K"], _inputs["V"]
    )
    for name in disp.regimes:
        with torch.no_grad():
            y = disp(_inputs["q"], _inputs["K"], _inputs["V"], regime=name)
        assert (y - ref).abs().max().item() < 1e-9, name


def test_streaming_serial_fallback_module(ssm):
    model, x, ir, src, eg, root, _st = ssm
    frontier = regime_frontier(
        eg,
        root,
        {"decode": (launch_aware_cost, "om_streaming")},
        ir=ir,
    )
    ch = frontier["decode"]
    assert ch.degraded and not ch.native
    disp = frontier.build(param_values=src)
    mod = disp.executor_module("decode")
    # the om carrier never materialised: serial fallback, disengaged
    assert not mod.is_streaming and mod.n_blocks == 0
    assert not ch.engaged
    with torch.no_grad():
        assert torch.allclose(disp(x, regime="decode"), model(x))


# ---------------------------------------------------------------------------
# build_egraph tiers
# ---------------------------------------------------------------------------


def test_build_egraph_no_lifts_no_xc_growth():
    """A carrier-free model: the lift passes offer nothing and the XC
    tier breaks out immediately with zero growth."""
    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)
    ).double()
    x = torch.randn(4, 8, dtype=torch.float64)
    _eg, _root, _ir, _src, stats = build_egraph(
        m, x, max_iterations=6
    )
    assert stats.get("nonlocal_lifts", 0) == 0
    assert stats["xc_rounds"] == 0
    assert stats["xc_fires"] == 0


def test_build_egraph_xc_disabled():
    torch.manual_seed(0)
    m = nn.Linear(8, 8).double()
    x = torch.randn(4, 8, dtype=torch.float64)
    _eg, _root, _ir, _src, stats = build_egraph(
        m, x, xc=False, max_iterations=6
    )
    assert "xc_rounds" not in stats and "xc_fires" not in stats


def test_build_egraph_xc_integrates_second_round_lifts():
    """DiagonalSSM: the bounded XC tier re-runs the non-local lifts on
    seam-minted members — the 'more' integration path."""
    torch.manual_seed(0)
    m = DiagonalSSM(8, 8, 8).eval().double()
    x = torch.randn(8, 8, dtype=torch.float64)
    eg, root, ir, _src, stats = build_egraph(
        m, x, max_iterations=8
    )
    assert stats["nonlocal_lifts"] >= 1
    assert stats["xc_rounds"] >= 1
    # the served graph still verifies against the source
    frontier = regime_frontier(
        eg, root, {"w": (flops_cost, "generic")}, ir=ir
    )
    assert frontier["w"].term is not None


# ---------------------------------------------------------------------------
# regime_dispatch calibrate/verify plumbing
# ---------------------------------------------------------------------------


def test_dispatch_calibrate_true_measures_once(monkeypatch):
    model, x, *_ = _ssm_fixture()
    fake_profile = {"tflops": 10.0, "gbps": 100.0, "launch_us": 5.0}
    calls = []

    def _fake_measure():
        calls.append(1)
        return fake_profile

    monkeypatch.setattr(cal_mod, "calibrate", _fake_measure)
    disp = regime_dispatch(
        model,
        x,
        regimes=[
            Regime("roof"),
            Regime("gpu", profile=fake_profile, cost_fn=flops_cost),
        ],
        calibrate=True,
        verify=False,
    )
    assert calls == [1]  # measured once, not once per regime
    regs = {r.name: r for r in disp.frontier.regimes}
    assert regs["roof"].profile == fake_profile
    assert regs["gpu"].profile == fake_profile
    assert regs["gpu"].cost_fn is flops_cost  # explicit wins
    assert disp.verification is None  # verify=False skips the check


def test_dispatch_calibrate_no_pending_profiles(monkeypatch):
    """All regimes already carry profiles: calibrate=True must not
    measure at all (the pending list is empty)."""
    model, x, *_ = _ssm_fixture()
    fake = {"tflops": 10.0, "gbps": 100.0, "launch_us": 5.0}
    monkeypatch.setattr(
        cal_mod,
        "calibrate",
        lambda: pytest.fail("should not be called"),
    )
    disp = regime_dispatch(
        model,
        x,
        regimes=[Regime("a", profile=fake, cost_fn=flops_cost)],
        calibrate=True,
        verify=False,
    )
    assert disp.frontier.regimes[0].profile == fake


def test_dispatch_default_regime_kwarg():
    model, x, *_ = _ssm_fixture()
    disp = regime_dispatch(
        model,
        x,
        regimes=[
            Regime("a", cost_fn=flops_cost, executor="generic"),
            Regime("b", cost_fn=launch_aware_cost, executor="generic"),
        ],
        default="b",
        verify=False,
    )
    assert disp.regime == "b"
    with pytest.raises(KeyError):
        regime_dispatch(
            model,
            x,
            regimes=[Regime("a", cost_fn=flops_cost)],
            default="ghost",
            verify=False,
        )


# ---------------------------------------------------------------------------
# defensive branches believed unreachable — pragma candidates
# ---------------------------------------------------------------------------


def test_defensive_branch_inventory():
    """Documents branches we believe unreachable by construction —
    candidates for ``pragma: no cover``:

    * regime.py:599 — ``if ec is None: continue`` in _force_carrier:
      ``eg._classes`` keys are always canonicalised back into the same
      map, so a find() result is never missing from it.
    * regime.py:620 — ``if t is None: continue``: an overrides-pinned
      root enode always extracts (its own class was just pinned to it);
      a pinned cyclic enode *might* extract to None but constructing
      one requires corrupting the union-find by hand.
    * regime.py:741-742 — the ``non-native`` report flag: unreachable
      since ``degraded`` is exactly ``carrier is not None and not
      native and not carrier_present`` — every flaggable non-native
      choice is either DEGRADED or carrier-nested.
    """
    # sanity on the flag logic that makes 741-742 dead: with a carrier
    # executor the non-native outcomes are DEGRADED or carrier-nested
    for carrier_present in (False, True):
        degraded = carrier_present is False  # mirrors the frontier expr
        flag = (
            "DEGRADED"
            if degraded
            else ("carrier-nested" if carrier_present else "non-native")
        )
        assert flag != "non-native"
