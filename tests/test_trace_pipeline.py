"""Traced-monoidal carrier in the *pipeline* — regime saturation,
frontier extraction, and executor dispatch (``catopt.regime``).

``tests/test_trace.py`` proves the JSV axioms as isolated rewrites.
This file proves the pipeline-level story:

* ``TRACE_LAWS`` is part of ``CARRIER_LAWS`` (the law set
  ``build_egraph`` / ``regime_dispatch`` saturate with) and a
  ``"trace"`` executor spec exists so regimes can *prefer* the
  fixpoint carrier.

* **Honest negative — the export boundary.**  On real exported models
  (``DiagonalSSM``, ``HybridBlock``) NO ``tr_*`` law fires and no
  trace-family enode materialises.  Trace ops are never produced by
  ``torch.export``, and of the 13 trace laws only ``tr_collapse`` has a
  trace-free LHS — but its LHS is the resolvent pattern
  ``add(P, Q·(I−S)⁻¹·R)`` with all four blocks ``split``-projections of
  ONE matrix through ``inv``/``eye``, structure no export and no other
  law family produces.  The missing lift — unrolled recurrence spine →
  ``matmul(trace(F, T·d), vec)`` over the nilpotent block-shift F — is
  the non-local "trm ↔ apply bridge" catopt/trace.py itself documents:
  it must *construct* the time-extended matrix from the whole horizon,
  a pairing pass (like ``pair_shared_input_linears``), not a lhs→rhs
  rule.  These tests pin that boundary instead of pretending.

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

import torch
import pytest

import catopt.trace as cat_trace  # noqa: F401  (registers torch bindings)
from catopt.egraph import EGraph
from catopt import meta
from catopt import rules as R
from catopt.ir import IR, Op, Var, Param, TensorType, op_repr
from catopt.om import OM_LAWS
from catopt.trace import TRACE_LAWS
from catopt.models.ssm import DiagonalSSM
from catopt.models.hybrid import HybridBlock
from catopt.torch_bridge import export_to_ir, ir_to_torch_module
from catopt.cost import flops_cost
from catopt.regime import (
    CARRIER_LAWS, EXECUTORS, Regime,
    build_egraph, regime_frontier, regime_dispatch,
    is_trace_rooted_term,
)


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
    return torch.randn(*shape,
                       generator=torch.Generator().manual_seed(seed))


def _mkf(du: int, dx: int, dy: int, seed: int,
         s_scale: float = 0.25) -> torch.Tensor:
    """Feedback-first block matrix [[S,R],[Q,P]], contractive S."""
    S = _rand(seed + 1, du, du) * s_scale
    Rm = _rand(seed + 2, du, dx) * 0.4
    Q = _rand(seed + 3, dy, du) * 0.4
    P = _rand(seed + 4, dy, dx) * 0.4
    return torch.cat([torch.cat([S, Rm], dim=1),
                      torch.cat([Q, P], dim=1)], dim=0)


def _op_census(eg):
    from collections import Counter
    return Counter(n.op for n in eg._node_to_class)


def _tr_fires(eg):
    return {k: v for k, v in eg.rule_fires.items()
            if k.startswith("tr_")}


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
        split = Op.make("bdiag", tr,
                        Op.make("trace", Fv, usize=du))
        assert is_trace_rooted_term(split)
        assert not is_trace_rooted_term(
            Op.make("matmul", tr, Var("x", TensorType((dx,)))))
        assert not is_trace_rooted_term(Fv)


# ---------------------------------------------------------------------------
#  (b) the export boundary — honest negative on real models
# ---------------------------------------------------------------------------

class TestExportBoundary:
    """No trace law fires on real exports; the only trace-free LHS
    (``tr_collapse``'s resolvent pattern) needs ``inv``/``eye``/shared
    ``split``-projection structure nothing produces."""

    def test_diagonal_ssm_no_trace_fires(self):
        torch.manual_seed(0)
        T, D = 16, 16
        m = DiagonalSSM(D, D, T).eval().double()
        x = torch.randn(T, D)
        # the ACTUAL regime pipeline entry point (CARRIER_LAWS)
        eg, root, ir, src, stats = build_egraph(m, x)
        assert _tr_fires(eg) == {}
        census = _op_census(eg)
        for op in _TRACE_FAMILY:
            assert census[op] == 0, op
        # the scan carrier DID lift — the graph is alive, traces just
        # have no way in
        assert census["applyd"] > 0 and census["aff_diag"] > 0

    def test_hybrid_block_no_trace_fires(self):
        torch.manual_seed(0)
        T, D = 16, 16
        m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
        x = torch.randn(T, D)
        ir, st = export_to_ir(m, x)
        eg = EGraph()
        # the stratified union harness (more contentful laws than
        # CARRIER_LAWS — stronger negative evidence)
        laws = (R.SCAN_DIAG_LAWS + OM_LAWS + R.SIMPLIFICATION_RULES
                + R.CATEGORICAL_RULES + TRACE_LAWS)
        out = meta.stratified_run(
            eg, laws, ir.root, max_iterations=14, max_nodes=400_000,
            extract_fn=eg.extract_min_depth)
        assert _tr_fires(eg) == {}
        census = _op_census(eg)
        for op in _TRACE_FAMILY:
            assert census[op] == 0, op
        # both real carriers lifted in the same graph
        assert census["applyd"] > 0 and census["om_apply"] > 0

    def test_why_tr_collapse_cannot_match(self):
        """Mechanistic check: ``tr_collapse`` is the ONLY trace law
        whose LHS mentions no trace-family op — its pattern is
        ``add(P, matmul(Q, matmul(inv(sub(eye,S)), R)))`` where
        P,Q,R,S are ``split`` projections of one f.  Exports contain no
        ``inv``/``eye``/``split`` enodes at all, so the LHS can never
        be present — the bridge is missing, not vetoed."""
        torch.manual_seed(0)
        T, D = 16, 16
        m = DiagonalSSM(D, D, T).eval().double()
        x = torch.randn(T, D)
        eg, root, ir, src, stats = build_egraph(m, x)
        census = _op_census(eg)
        # none of tr_collapse's distinguishing structure exists
        assert census["inv"] == 0
        assert census["eye"] == 0
        assert census["split"] == 0
        # and no trace law fired — saturation is complete (2 iters)
        assert stats["iterations"] <= 3

    def test_no_served_member_contains_trace(self):
        """End-to-end dispatch on a real model: every served regime
        member is trace-free (nothing to serve the carrier from)."""
        torch.manual_seed(0)
        T, D = 16, 16
        m = DiagonalSSM(D, D, T).eval().double()
        x = torch.randn(T, D)
        disp = regime_dispatch(m, x, regimes=[
            Regime("sequential", cost_fn=flops_cost, executor="generic"),
            Regime("parallel", extract_fn=EGraph.extract_min_depth,
                   executor="auto"),
        ])
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
        eg, root, ir, src, stats = build_egraph(m, x)
        frontier = regime_frontier(eg, root, {
            "fixpt": (flops_cost, "trace"),
        }, ir=ir)
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
        Op.make("trace",
                Op.make("parl", Fp, Gp, u1=du, u2=dv),
                usize=(du, dv)),
        xv)
    ir = IR(root=term, inputs=[xv], input_names={"x"},
            params={"p_F": Fp, "p_G": Gp})
    x = _rand(50, 2 * dx)
    env = {"p_F": F, "p_G": G}
    return ir, term, x, env


class TestSeededTracePipeline:
    def test_superpose_channel_split_fires_in_pipeline(self):
        """The payoff rule: ONE saturation under CARRIER_LAWS splits
        the joint loop into bdiag of independent channel traces —
        reachable the moment any trace exists in the graph."""
        ir, term, x, env = _seeded_joint_loop()
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(CARRIER_LAWS, root, max_iterations=10)

        assert eg.rule_fires.get("tr_superpose", 0) >= 1
        assert eg.rule_fires.get("tr_expand", 0) >= 1
        assert eg.rule_fires.get("tr_vanish_split", 0) >= 1
        # the channel-split form is a member of the root e-class
        # (matches() is e-class relative: match the whole rooted term)
        assert eg.matches(
            Op.make("matmul",
                    Op.make("bdiag",
                            Op.make("trace", "f", usize="DU"),
                            Op.make("trace", "g", usize="DV")),
                    "x"), root)

    def test_frontier_serves_trace_members_fp64(self):
        """regime_frontier extracts trace-bearing members; both the
        joint fixpoint and the channel-split bdiag verify fp64-exact
        through the generic executor (trace torch bindings)."""
        ir, term, x, env = _seeded_joint_loop()
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(CARRIER_LAWS, root, max_iterations=10)

        frontier = regime_frontier(eg, root, {
            "work": (flops_cost, "auto"),
            "depth": Regime("depth",
                            extract_fn=EGraph.extract_min_depth,
                            executor="auto"),
            "fixpt": Regime("fixpt", cost_fn=flops_cost,
                            executor="trace"),
        }, ir=ir, src_term=term)

        ref = ir_to_torch_module(ir, param_values=env)(x)
        saw = {"joint": False, "split": False}
        for name in frontier.names:
            ch = frontier[name]
            assert ch.term is not None and not ch.degraded
            rep = op_repr(ch.term)
            assert "trace" in rep          # a trace member was served
            opt = IR(root=ch.term, inputs=ir.inputs,
                     input_names=ir.input_names, params=ir.params)
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
        assert pinned is not None, "no channel-split member materialised"
        split_term = eg.extract_best(canon, flops_cost,
                                     overrides=pinned)
        rep = op_repr(split_term)
        assert "bdiag" in rep and rep.count("trace") >= 2
        opt = IR(root=split_term, inputs=ir.inputs,
                 input_names=ir.input_names, params=ir.params)
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
        ir = IR(root=term, inputs=[xv], input_names={"x"},
                params={"p_F": Fp})
        x = _rand(22, dx)
        eg = EGraph()
        root = eg.add_term(term)
        eg.run(CARRIER_LAWS, root, max_iterations=10)

        frontier = regime_frontier(eg, root, {
            "fixpt": (flops_cost, "trace"),
        }, ir=ir)
        ch = frontier["fixpt"]
        assert ch.executor == "trace"
        assert ch.carrier_present        # trace census visible in term
        # served member verifies fp64
        ref = ir_to_torch_module(ir, param_values={"p_F": F})(x)
        opt = IR(root=ch.term, inputs=ir.inputs,
                 input_names=ir.input_names, params=ir.params)
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
