# ruff: noqa: RUF002
"""trace_lift — the bounded-horizon opt-out and the wide-emit fast
path on graphs without proof tracking.

``max_trace_T`` bounds the horizon worth materialising the O(T²)
block-shift matrix for: ``flops_cost`` bills the offered ``trace``
op alone ``2·(T·d)³`` and the executor-overhead model adds a 10⁴×
solver surcharge per occurrence, so at that scale the member can
never win extraction — the offer is pure e-graph bloat.  The bound
skips the offer entirely; ``None`` restores the previous
always-emit behaviour (see ``lift_scan_to_trace``).

``truncation_level=1`` graphs keep no per-enode provenance —
``EGraph._track`` is False — so the lifted enodes take the
untracked path in ``_add_enode_dedup``.  The offers still land:
only the certificate records are absent.
"""

from catopt.egraph import EGraph
from catopt.ir import Op, Param, TensorType
from catopt_carriers import trace_lift as TL


def _diag_term(T: int, d: int):
    """h_t = a_t ⊙ h_{t−1} + b_t — a raw diagonal spine."""
    a = Param("pa", TensorType((T, d)))
    b = Param("pb", TensorType((T, d)))
    h0 = Param("h0", TensorType((d,)))
    h = h0
    for t in range(T):
        a_t = Op.make("select", a, dim=0, index=t)
        b_t = Op.make("select", b, dim=0, index=t)
        h = Op.make("add", Op.make("mul", a_t, h), b_t)
    return h


def test_max_trace_T_skips_long_spines():
    T, d = 8, 4
    eg = EGraph()
    eg.add_term(_diag_term(T, d))
    lifts = TL.lift_scan_to_trace(eg, max_trace_T=T - 1)
    assert lifts == []


def test_max_trace_T_admits_at_bound_and_when_none():
    T, d = 8, 4
    eg = EGraph()
    root = eg.add_term(_diag_term(T, d))
    at = TL.lift_scan_to_trace(eg, max_trace_T=T)
    assert {lft.split for lft in at} == {None, (2, 2)}
    assert all(eg.find(lft.out_eid) == eg.find(root) for lft in at)

    eg2 = EGraph()
    root2 = eg2.add_term(_diag_term(T, d))
    unbounded = TL.lift_scan_to_trace(eg2, max_trace_T=None)
    assert {lft.split for lft in unbounded} == {None, (2, 2)}
    assert all(
        eg2.find(lft.out_eid) == eg2.find(root2) for lft in unbounded
    )


def test_lift_on_level1_graph_untracked():
    """``truncation_level=1`` drops per-enode provenance — the
    emission fast path takes its untracked branch — while the
    offered members still land in the recurrence's class."""
    T, d = 6, 4
    eg = EGraph(truncation_level=1)
    root = eg.add_term(_diag_term(T, d))
    lifts = TL.lift_scan_to_trace(eg)
    assert {lft.split for lft in lifts} == {None, (2, 2)}
    assert all(eg.find(lft.out_eid) == eg.find(root) for lft in lifts)
    assert eg.n_proof_edges == 0  # quotient-only graph, no merge log
