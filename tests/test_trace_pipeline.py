"""Traced-monoidal carrier in the *pipeline* — regime saturation,
frontier extraction, and executor dispatch (``catopt.regime``).

``tests/test_trace.py`` proves the JSV axioms as isolated rewrites.
This file proves the pipeline-level story:

* ``TRACE_LAWS`` is part of ``CARRIER_LAWS`` (the law set
  ``build_egraph`` / ``regime_dispatch`` saturate with) and a
  ``"trace"`` executor spec exists so regimes can *prefer* the
  fixpoint carrier.

* **The export boundary — now crossed by a non-local pass.**  Raw
  exports contain no ``trace``/``inv``/``eye`` foothold, and no
  lhs→rhs law can mint one (``tr_collapse``'s resolvent LHS needs
  shared ``split``-projections nothing produces).  The bridge is
  ``catopt.trace_lift.lift_scan_to_trace`` — wired into
  ``build_egraph`` — which *constructs* the nilpotent block-shift F
  from the whole unrolled horizon and offers
  ``matmul(trace(F, T·d), vec)``, witnessed.  Post-lift the JSV laws
  fire on real exports (superpose/expand/vanish/tighten).

* **What IS reachable without the lift.**  Seeded into the same
  pipeline law set, a joint-loop trace saturates normally:
  ``tr_superpose`` splits it into a ``bdiag`` of independent channel
  traces (the parallel-schedule payoff), ``tr_expand`` grows the
  resolvent form, extraction and ``regime_frontier`` serve the
  trace-bearing members, and the generic executor evaluates them
  fp64-exact through the torch bindings.

Everything is fp64 (``.double()``) — verification is exact, not
tolerance-lottery.
"""

import pytest
import torch

import catopt.trace as cat_trace  # noqa: F401  (registers torch bindings)
from catopt import meta
from catopt import rules as R
from catopt.cost import flops_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Op, Param, TensorType, Var, op_repr
from catopt.models.hybrid import HybridBlock
from catopt.models.ssm import DiagonalSSM
from catopt.om import OM_LAWS
from catopt.regime import (
    CARRIER_LAWS,
    EXECUTORS,
    Regime,
    build_egraph,
    is_trace_rooted_term,
    regime_dispatch,
    regime_frontier,
)
from catopt.torch_bridge import export_to_ir, ir_to_torch_module
from catopt.trace import TRACE_LAWS


@pytest.fixture(autouse=True)
def _fp64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _rand(seed: int, *shape: int) -> torch.Tensor:
    return torch.randn(
        *shape, generator=torch.Generator().manual_seed(seed)
    )


def _mkf(
    du: int, dx: int, dy: int, seed: int, s_scale: float = 0.25
) -> torch.Tensor:
    """Feedback-first block matrix [[S,R],[Q,P]], contractive S."""
    S = _rand(seed + 1, du, du) * s_scale
    Rm = _rand(seed + 2, du, dx) * 0.4
    Q = _rand(seed + 3, dy, du) * 0.4
    P = _rand(seed + 4, dy, dx) * 0.4
    return torch.cat(
        [torch.cat([S, Rm], dim=1), torch.cat([Q, P], dim=1)], dim=0
    )


def _op_census(eg):
    from collections import Counter

    return Counter(n.op for n in eg._node_to_class)


def _tr_fires(eg):
    return {
        k: v for k, v in eg.rule_fires.items() if k.startswith("tr_")
    }


_TRACE_FAMILY = ("trace", "bdiag", "parl", "eye", "cswap", "inv")


# ---------------------------------------------------------------------------
#  (a) trace is IN the pipeline law set / executor table
# ---------------------------------------------------------------------------


class TestTraceInPipeline:
    def test_trace_laws_in_carrier_laws(self):
        for r in TRACE_LAWS:
            assert r in CARRIER_LAWS, r.name
        # default_rules() (what build_egraph saturates with) carries them
        from catopt.regime import default_rules

        names = {r.name for r in default_rules()}
        assert {r.name for r in TRACE_LAWS} <= names

    def test_trace_executor_spec(self):
        ex = EXECUTORS["trace"]
        assert ex.carrier is not None
        roots, inner, leaf = ex.carrier
        assert "trace" in roots
        assert {"parl", "bdiag"} <= inner
        assert {"eye", "cswap"} <= leaf

    def test_is_trace_rooted_term(self):
        du, dx, dy = 2, 3, 2
        Fv = Var("F", TensorType((du + dy, du + dx)))
        tr = Op.make("trace", Fv, usize=du)
        assert is_trace_rooted_term(tr)
        split = Op.make("bdiag", tr, Op.make("trace", Fv, usize=du))
        assert is_trace_rooted_term(split)
        assert not is_trace_rooted_term(
            Op.make("matmul", tr, Var("x", TensorType((dx,))))
        )
        assert not is_trace_rooted_term(Fv)


# ---------------------------------------------------------------------------
#  (b) the export boundary — honest negative on real models
# ---------------------------------------------------------------------------


class TestExportBoundary:
    """Raw exports contain no trace foothold — trace enters only via
    the non-local ``lift_scan_to_trace`` pass (wired into
    ``build_egraph``), after which the JSV laws fire on real models."""

    def test_diagonal_ssm_trace_fires_via_lift(self):
        torch.manual_seed(0)
        T, D = 16, 16
        m = DiagonalSSM(D, D, T).eval().double()
        x = torch.randn(T, D)
        # the ACTUAL regime pipeline entry point (CARRIER_LAWS + lifts)
        eg, _root, _ir, _src, stats = build_egraph(m, x)
        assert stats.get("nonlocal_lifts", 0) > 0
        assert _tr_fires(eg), "trace laws should fire post-lift"
        census = _op_census(eg)
        assert census["trace"] > 0
        # the scan carrier lifted too — both domains coexist
        assert census["applyd"] > 0 and census["aff_diag"] > 0

    def test_hybrid_block_saturation_alone_no_trace(self):
        torch.manual_seed(0)
        T, D = 16, 16
        m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
        x = torch.randn(T, D)
        ir, _st = export_to_ir(m, x)
        eg = EGraph()
        # the stratified union harness (more contentful laws than
        # CARRIER_LAWS — stronger negative evidence)
        laws = (
            R.SCAN_DIAG_LAWS
            + OM_LAWS
            + R.SIMPLIFICATION_RULES
            + R.CATEGORICAL_RULES
            + TRACE_LAWS
        )
        _out = meta.stratified_run(
            eg,
            laws,
            ir.root,
            max_iterations=14,
            max_nodes=400_000,
            extract_fn=eg.extract_min_depth,
        )
        assert _tr_fires(eg) == {}
        census = _op_census(eg)
        for op in _TRACE_FAMILY:
            assert census[op] == 0, op
        # both real carriers lifted in the same graph
        assert census["applyd"] > 0 and census["om_apply"] > 0

    def test_why_trace_needs_the_lift(self):
        """Mechanistic check: the raw export contains no ``trace``/
        ``inv``/``eye`` enodes — saturation alone can never reach the
        trace domain.  ``build_egraph``'s non-local ``lift_scan_to_trace``
        pass introduces the block-shift ``trace(F)`` (and the resolvent
        structure the laws then expand), so trace enodes appear only
        *after* the lift."""
        torch.manual_seed(0)
        T, D = 16, 16
        m = DiagonalSSM(D, D, T).eval().double()
        x = torch.randn(T, D)
        # raw export + plain saturation: no trace foothold exists
        ir, _ = export_to_ir(m, x)
        eg0 = EGraph()
        r0 = eg0.add_term(ir.root)
        eg0.run(TRACE_LAWS, r0, max_iterations=3)
        c0 = _op_census(eg0)
        assert c0["trace"] == 0 and c0["inv"] == 0 and c0["eye"] == 0
        # the full pipeline lifts recurrences into trace form
        eg, _root, ir, _src, stats = build_egraph(m, x)
        assert stats.get("nonlocal_lifts", 0) > 0
        assert _op_census(eg)["trace"] > 0

    def test_no_served_member_contains_trace(self):
        """End-to-end dispatch on a real model: every served regime
        member is trace-free (nothing to serve the carrier from)."""
        torch.manual_seed(0)
        T, D = 16, 16
        m = DiagonalSSM(D, D, T).eval().double()
        x = torch.randn(T, D)
        disp = regime_dispatch(
            m,
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
        for name, ch in disp.frontier.choices.items():
            assert "trace" not in op_repr(ch.term), name
        for name, v in disp.verification.items():
            assert v["ok"], f"{name}: {v}"

    def test_trace_executor_degrades_honestly(self):
        """Declaring executor='trace' on a trace-free e-graph must not
        pretend: degraded=True, serial fallback still correct."""
        torch.manual_seed(0)
        T, D = 16, 16
        m = DiagonalSSM(D, D, T).eval().double()
        x = torch.randn(T, D)
        eg, root, ir, src, _stats = build_egraph(m, x)
        frontier = regime_frontier(
            eg,
            root,
            {
                "fixpt": (flops_cost, "trace"),
            },
            ir=ir,
        )
        ch = frontier["fixpt"]
        assert ch.degraded and not ch.native
        assert not ch.carrier_present
        disp = frontier.build(param_values=src)
        ref = m(x)
        with torch.no_grad():
            y = disp(x, regime="fixpt")
        assert (y - ref).abs().max().item() < 1e-9


# ---------------------------------------------------------------------------
#  (c) what IS reachable — seeded trace through the SAME pipeline
# ---------------------------------------------------------------------------


def _seeded_joint_loop():
    """matmul(trace(parl(F,G,u1,u2), (du,dv)), x) — a joint loop over
    two independent channels applied to a data vector."""
    du, dv, dx, dy = 2, 2, 3, 2
    F = _mkf(du, dx, dy, seed=7)
    G = _mkf(dv, dx, dy, seed=9)
    Fp = Param("p_F", TensorType(tuple(F.shape)))
    Gp = Param("p_G", TensorType(tuple(G.shape)))
    xv = Var("x", TensorType((2 * dx,)))
    term = Op.make(
        "matmul",
        Op.make(
            "trace",
            Op.make("parl", Fp, Gp, u1=du, u2=dv),
            usize=(du, dv),
        ),
        xv,
    )
    ir = IR(
        root=term,
        inputs=[xv],
        input_names={"x"},
        params={"p_F": Fp, "p_G": Gp},
    )
    x = _rand(50, 2 * dx)
    env = {"p_F": F, "p_G": G}
    return ir, term, x, env


class TestSeededTracePipeline:
    def test_superpose_channel_split_fires_in_pipeline(self):
        """The payoff rule: ONE saturation under CARRIER_LAWS splits
        the joint loop into bdiag of independent channel traces —
        reachable the moment any trace exists in the graph."""
        _ir, term, _x, _env = _seeded_joint_loop()
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(CARRIER_LAWS, root, max_iterations=10)

        assert eg.rule_fires.get("tr_superpose", 0) >= 1
        assert eg.rule_fires.get("tr_expand", 0) >= 1
        assert eg.rule_fires.get("tr_vanish_split", 0) >= 1
        # the channel-split form is a member of the root e-class
        # (matches() is e-class relative: match the whole rooted term)
        assert eg.matches(
            Op.make(
                "matmul",
                Op.make(
                    "bdiag",
                    Op.make("trace", "f", usize="DU"),
                    Op.make("trace", "g", usize="DV"),
                ),
                "x",
            ),
            root,
        )

    def test_frontier_serves_trace_members_fp64(self):
        """regime_frontier extracts trace-bearing members; both the
        joint fixpoint and the channel-split bdiag verify fp64-exact
        through the generic executor (trace torch bindings)."""
        ir, term, x, env = _seeded_joint_loop()
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(CARRIER_LAWS, root, max_iterations=10)

        frontier = regime_frontier(
            eg,
            root,
            {
                "work": (flops_cost, "auto"),
                "depth": Regime(
                    "depth",
                    extract_fn=EGraph.extract_min_depth,
                    executor="auto",
                ),
                "fixpt": Regime(
                    "fixpt", cost_fn=flops_cost, executor="trace"
                ),
            },
            ir=ir,
            src_term=term,
        )

        ref = ir_to_torch_module(ir, param_values=env)(x)
        saw = {"joint": False, "split": False}
        for name in frontier.names:
            ch = frontier[name]
            assert ch.term is not None and not ch.degraded
            rep = op_repr(ch.term)
            assert "trace" in rep  # a trace member was served
            opt = IR(
                root=ch.term,
                inputs=ir.inputs,
                input_names=ir.input_names,
                params=ir.params,
            )
            y = ir_to_torch_module(opt, param_values=env)(x)
            assert (y - ref).abs().max().item() < 1e-12, name
            if "bdiag" in rep:
                saw["split"] = True
            else:
                saw["joint"] = True
        # across regimes both architectures surfaced
        assert saw["joint"] and saw["split"]

    def test_channel_split_member_is_fp64_exact(self):
        """Extract the bdiag-of-traces member explicitly (pin the root
        class's bdiag enode) and verify it against the joint form."""
        ir, term, x, env = _seeded_joint_loop()
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(CARRIER_LAWS, root, max_iterations=10)

        canon = eg.find(root)
        # find the matmul whose child is the bdiag split form
        pinned = None
        for n in eg.get_class(canon).nodes:
            if n.op != "matmul":
                continue
            child = eg.get_class(eg.find(n.children[0]))
            if any(m.op == "bdiag" for m in child.nodes):
                for m in child.nodes:
                    if m.op == "bdiag":
                        pinned = {eg.find(n.children[0]): m}
                        break
            if pinned:
                break
        assert pinned is not None, (
            "no channel-split member materialised"
        )
        split_term = eg.extract_best(
            canon, flops_cost, overrides=pinned
        )
        rep = op_repr(split_term)
        assert "bdiag" in rep and rep.count("trace") >= 2
        opt = IR(
            root=split_term,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params=ir.params,
        )
        ref = ir_to_torch_module(ir, param_values=env)(x)
        y = ir_to_torch_module(opt, param_values=env)(x)
        assert (y - ref).abs().max().item() < 1e-12

    def test_trace_executor_native_on_trace_root(self):
        """executor='trace' reports native when the served term's root
        carries the fixpoint (auto-resolution also picks it up)."""
        du, dx, dy = 2, 3, 2
        F = _mkf(du, dx, dy, seed=21)
        Fp = Param("p_F", TensorType(tuple(F.shape)))
        xv = Var("x", TensorType((dx,)))
        # trace applied to data through the data wire — root is trace
        term = Op.make("matmul", Op.make("trace", Fp, usize=du), xv)
        ir = IR(
            root=term,
            inputs=[xv],
            input_names={"x"},
            params={"p_F": Fp},
        )
        x = _rand(22, dx)
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(CARRIER_LAWS, root, max_iterations=10)

        frontier = regime_frontier(
            eg,
            root,
            {
                "fixpt": (flops_cost, "trace"),
            },
            ir=ir,
        )
        ch = frontier["fixpt"]
        assert ch.executor == "trace"
        assert ch.carrier_present  # trace census visible in term
        # served member verifies fp64
        ref = ir_to_torch_module(ir, param_values={"p_F": F})(x)
        opt = IR(
            root=ch.term,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params=ir.params,
        )
        y = ir_to_torch_module(opt, param_values={"p_F": F})(x)
        assert (y - ref).abs().max().item() < 1e-12

    def test_expand_then_collapse_roundtrip_in_pipeline(self):
        """Under the full carrier set, expand and collapse are both
        available; the trace root e-class keeps BOTH the fixpoint node
        and its resolvent expansion as members."""
        du, dx, dy = 2, 3, 2
        F = _mkf(du, dx, dy, seed=31)
        Fv = Var("F", TensorType(tuple(F.shape)))
        term = Op.make("trace", Fv, usize=du)
        eg = EGraph()
        eid = eg.add_term(term)
        eg.run(CARRIER_LAWS, eid, max_iterations=10)
        assert eg.rule_fires.get("tr_expand", 0) >= 1
        # trace node still a member of its own class alongside inv/split
        ops = {n.op for n in eg.get_class(eg.find(eid)).nodes}
        assert "trace" in ops
        census = _op_census(eg)
        assert census["inv"] > 0 and census["split"] > 0
