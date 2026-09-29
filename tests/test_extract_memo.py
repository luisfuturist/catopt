"""Coverage tests for extract_best's memo-sharing / pre-seeding.

``EGraph.extract_best`` keeps one content-keyed cost memo per cost_fn
on the e-graph (``_cost_memo_for``) and pre-seeds the additive
sub-model entries (``("eo","generic",·)`` dispatch counts and
``("rc",pf,bw,ls,·)`` roofline sums) that executor cost fns would
otherwise recompute by re-walking each candidate's whole subtree.
These tests pin the semantics of those fast paths and cover their
self-gating edge cases.
"""

from catopt_core.egraph import EGraph, ENode
from catopt_core.ir import Op, Param, TensorType, Var, op_repr
from catopt_core.cost import (
    _profile_constants,
    _roofline_cost,
    backend_cost,
    count_cost,
    executor_cost_for,
    executor_overhead,
    flops_cost,
)


def _t(*shape):
    return TensorType(tuple(shape))


def _chain():
    """matmul(matmul(x, A), B) — a small two-level term."""
    x = Var("x", _t(2, 3))
    a = Param("A", _t(3, 4))
    b = Param("B", _t(4, 5))
    return Op.make("matmul", Op.make("matmul", x, a), b)


def test_cost_memo_shared_per_cost_fn():
    """The shared memo persists per cost_fn across extractions."""
    eg = EGraph()
    root = eg.add_term(_chain())
    t1 = eg.extract_best(root, flops_cost)
    memos = eg._cost_memos
    fn_ent = memos[id(flops_cost)]
    assert fn_ent[0] is flops_cost
    t2 = eg.extract_best(root, flops_cost)
    # Same dict reused — second extraction prices nothing new.
    assert memos[id(flops_cost)][1] is fn_ent[1]
    assert op_repr(t2) == op_repr(t1)
    # A different cost_fn gets its own memo — entries never mix.
    t3 = eg.extract_best(root, count_cost)
    assert memos[id(count_cost)][0] is count_cost
    assert memos[id(count_cost)][1] is not fn_ent[1]
    assert t3 is not None


def test_extract_repeated_canonical_child():
    """add(x, x): second occurrence of a canonical child hits the
    per-class cache directly (the ``entry is not None`` branch)."""
    eg = EGraph()
    x = Var("x", _t(2))
    xid = eg.add_term(x)
    add = eg.add_enode("add", (xid, xid))
    best = eg.extract_best(add, count_cost)
    assert best is not None


def test_extract_stale_child_id():
    """A union after enode creation makes stored child ids stale:
    ``cache.get(stale)`` misses, ``find`` canonicalises, and the
    already-processed class is returned from ``best``'s own cache
    check."""
    eg = EGraph()
    x = Var("x", _t(2))
    y = Var("y", _t(2))
    xid = eg.add_term(x)
    yid = eg.add_term(y)
    add = eg.add_enode("add", (xid, xid))
    # Equal ranks -> union(yid, xid) parents xid under yid: the enode's
    # stored child id is now stale while the class it resolves to has
    # already been finalised by the first occurrence.
    eg.union(yid, xid)
    assert eg.find(xid) == yid != xid
    best = eg.extract_best(add, count_cost)
    assert isinstance(best, Op) and best.op == "add"
    assert best.args[0] in (x, y) and best.args[0] == best.args[1]


def test_extract_costfn_uninspectable_profile():
    """A cost_fn carrying an unrelated ``.profile`` attribute must not
    break extraction — roofline seeding just stays off (self-gating
    means even a wrong guess would be harmless)."""

    class BogusProfileCost:
        profile = object()  # no tflops/gbps/launch_us attributes

        def __call__(self, term, memo=None):
            return flops_cost(term, memo=memo)

    eg = EGraph()
    root = eg.add_term(_chain())
    best = eg.extract_best(root, BogusProfileCost())
    assert best is not None
    assert op_repr(best) == op_repr(_chain())


def test_executor_cost_seeds_eo_and_rc():
    """Under executor_cost_for the additive sub-model entries appear in
    the shared memo with exactly the values the models would compute
    — whether they were seeded by extraction or written by the model.
    A second extraction re-encounters the seeded keys."""
    eg = EGraph()
    term = _chain()
    root = eg.add_term(term)
    cf = executor_cost_for(lowering="generic")
    best = eg.extract_best(root, cf)
    memo = eg._cost_memos[id(cf)][1]
    eo = ("eo", "generic", best)
    assert eo in memo
    assert memo[eo] == executor_overhead(best, "generic", {})
    pf, bw, ls = _profile_constants(None)
    rk = ("rc", pf, bw, ls, best)
    assert rk in memo
    assert memo[rk] == _roofline_cost(best, {}, pf, bw, ls)
    # Second pass: every candidate's keys already exist — the
    # ``key not in m`` branches take the False edge.
    best2 = eg.extract_best(root, cf)
    assert op_repr(best2) == op_repr(best)


def test_executor_flops_base_gates_rc():
    """base="flops" never writes ``("rc",…)`` entries, so roofline
    seeding stays gated off while the eo seed still applies."""
    eg = EGraph()
    term = _chain()
    root = eg.add_term(term)
    cf = executor_cost_for(lowering="generic", base="flops")
    best = eg.extract_best(root, cf)
    memo = eg._cost_memos[id(cf)][1]
    pf, bw, ls = _profile_constants(None)
    assert ("rc", pf, bw, ls, best) not in memo
    assert ("eo", "generic", best) in memo


def test_extract_paired_seeds_steered_eo():
    """extract_paired's steered_score pre-seeds eo for member-routing
    candidates: pass-1-evaluated enodes find their key already present
    (the ``not in`` False edge) while a steering enode unreachable from
    the root mints a term pass 1 never priced — its seed is written
    (the True edge) before cost_fn is called."""
    eg = EGraph()
    a = eg.add_term(Var("a", _t(2)))
    b = eg.add_term(Var("b", _t(2)))
    c = eg.add_term(Var("c", _t(2)))
    m = eg.add_enode("mul", (a, b))  # member class
    reach_ok = eg.add_enode("mul", (m, c))  # member-reaching, lowered
    bypass = eg.add_enode("sub", (c, c))  # bypassing enode
    eg.union(reach_ok, bypass)
    root = eg.find(reach_ok)
    # A second steering candidate the root's pass-1 extraction never
    # priced: unreachable from root, so pass 1 cached no entry for the
    # class itself, yet its children (m, a) ARE cached — steered_score
    # evaluates a fresh term and writes the eo seed itself.
    fresh = eg.add_enode("mul", (m, a))
    dead = eg.add_enode("sub", (c, a))
    eg.union(fresh, dead)
    member_enode = ENode("mul", (a, b))
    groups = [{eg.find(m): member_enode}]
    cf = backend_cost(
        executor_cost_for(lowering="generic"),
        {"mul", "sub", "leaf"},
    )
    forced = eg.extract_paired(root, cf, groups)
    assert forced is not None
    assert forced.op == "mul"  # the supported member route won
    memo = eg._cost_memos[id(cf)][1]
    assert ("eo", "generic", forced) in memo


def test_extract_paired_plain_cost_no_seed():
    """With a non-executor cost_fn there are no eo entries to reuse:
    steered_score's seed check gates off (eo_ok False edge)."""
    eg = EGraph()
    a = eg.add_term(Var("a", _t(2)))
    b = eg.add_term(Var("b", _t(2)))
    c = eg.add_term(Var("c", _t(2)))
    m = eg.add_enode("mul", (a, b))
    reach = eg.add_enode("mul", (m, c))
    bypass = eg.add_enode("sub", (c, c))
    eg.union(reach, bypass)
    root = eg.find(reach)
    groups = [{eg.find(m): ENode("mul", (a, b))}]
    forced = eg.extract_paired(root, flops_cost, groups)
    assert forced is not None
    memo = eg._cost_memos[id(flops_cost)][1]
    assert ("eo", "generic", forced) not in memo


def test_param_only_discount_preserved():
    """A param-only subtree still extracts at zero billable local —
    the adj_of merge keeps the discount semantics identical."""
    eg = EGraph()
    x = Var("x", _t(2, 2))
    w = Param("W", _t(2, 2))
    z = Param("Z", _t(2, 2))
    wid = eg.add_term(w)
    zid = eg.add_term(z)
    wz = eg.add_enode("mul", (wid, zid))  # param-only product
    xid = eg.add_term(x)
    root = eg.add_enode("add", (xid, wz))
    best = eg.extract_best(root, flops_cost)
    assert best is not None
