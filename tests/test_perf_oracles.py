"""Plan-0010 correctness oracles.

Each performance lever is checked against the known-good semantics it
replaced:

* **Compiled matcher** — ``EGraph.matches`` now runs a compiled
  ``_Prog`` instead of interpreting the pattern tree.  The oracle is a
  verbatim copy of the interpreted ``_match`` algorithm
  (:func:`_ref_match`); substitution *sequences* must be identical —
  including ``$attr:`` metavariables, shared-metavariable consistency
  and ``max_results`` caps — on the law corpus and on generated terms.
* **Streaming vs list** — ``matches`` is a generator and
  ``apply_rule`` consumes it lazily under a frozen read epoch.  The
  oracle wraps ``_iter_matches`` so the whole enumeration materialises
  *before* the first application (the old semantics); applications,
  merges and the canonical partition must be identical.
* **Incremental congruence** — ``_close_congruence`` is now a worklist
  over a persistent owner map.  The oracle is the closure
  postcondition itself: after ``rebuild`` every e-node's children are
  canonical and no canonical e-node lives in two classes.
"""

from __future__ import annotations

import pytest
from catopt_core.cost import count_cost
from catopt_core.egraph import EGraph
from catopt_core.egraph.core import EGraph as CoreEG
from catopt_core.egraph.types import ENode, _pattern_attrs
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import (
    ALL_RULES,
    ASSOC_ADD,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    COMM_ADD,
    COMM_MUL,
    ID_ADD,
    ID_MUL,
    RuleSet,
)
from hypothesis import given, settings

from tests.test_property_strategies import terms

# ---------------------------------------------------------------------------
#  The interpreted matcher — verbatim port of the pre-0010 ``_match``
# ---------------------------------------------------------------------------


def _ref_match(eg, pattern, eid, subst, results, limit=None):
    if limit is not None and len(results) >= limit:
        return
    eid = eg.find(eid)
    eclass = eg._classes[eid]

    if isinstance(pattern, str):
        if pattern in subst:
            if subst[pattern] == eid:
                results.append(dict(subst))
            return
        subst[pattern] = eid
        results.append(dict(subst))
        del subst[pattern]
        return

    if isinstance(pattern, Op):
        attr_t = _pattern_attrs(pattern)
        for node in eg._nodes_of(eclass, pattern.op):
            if limit is not None and len(results) >= limit:
                return
            if len(node.children) != len(pattern.args):
                continue
            node_attrs = dict(node.attrs)
            if set(node_attrs) != {k for k, _ in attr_t}:
                continue
            attr_substs = [dict(subst)]
            attr_ok = True
            for k, pv in attr_t:
                nv = node_attrs[k]
                if isinstance(pv, str):
                    key = "$attr:" + pv
                    nxt = []
                    for cs in attr_substs:
                        if key in cs:
                            if cs[key] == nv:
                                nxt.append(cs)
                        else:
                            cc = dict(cs)
                            cc[key] = nv
                            nxt.append(cc)
                    attr_substs = nxt
                elif nv != pv:
                    attr_ok = False
                    break
                if not attr_substs:
                    attr_ok = False
                    break
            if not attr_ok:
                continue
            child_substs = attr_substs
            ok = True
            for i, pat_arg in enumerate(pattern.args):
                new_substs = []
                for cs in child_substs:
                    if (
                        limit is not None
                        and len(results) + len(new_substs) >= limit
                    ):
                        ok = False
                        break
                    child_results = []
                    _ref_match(
                        eg,
                        pat_arg,
                        node.children[i],
                        dict(cs),
                        child_results,
                        limit,
                    )
                    new_substs.extend(child_results)
                if not new_substs:
                    ok = False
                    break
                child_substs = new_substs
            if ok:
                if limit is not None:
                    room = limit - len(results)
                    results.extend(child_substs[:room])
                    if len(results) >= limit:
                        return
                else:
                    results.extend(child_substs)
        return

    key = ("key", repr(pattern))
    enode = ENode("leaf", (), (key,))
    if (
        enode in eg._node_to_class
        and eg.find(eg._node_to_class[enode]) == eid
    ):
        results.append(dict(subst))


def _ref_matches(eg, pattern, eid, max_results=None):
    res = []
    _ref_match(eg, pattern, eid, {}, res, max_results)
    return res


def _frozeq(substs):
    return [{k: v for k, v in s.items()} for s in substs]


# ---------------------------------------------------------------------------
#  Oracle 1 — compiled matcher ≡ interpreted matcher
# ---------------------------------------------------------------------------


def _saturated_graph():
    """A graph with merged + multi-member classes to match over."""
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    eg = EGraph(truncation_level=1)
    t = Op.make(
        "add",
        Op.make("mul", x, Const(1)),
        Op.make("mul", y, Const(1)),
    )
    root = eg.add_term(t)
    eg.run(
        [COMM_ADD, COMM_MUL, ID_ADD, ID_MUL, ASSOC_ADD],
        root,
        max_iterations=8,
        max_nodes=5000,
    )
    return eg, root


@pytest.mark.parametrize("rule", ALL_RULES, ids=lambda r: r.name)
def test_compiled_matches_interpreted_everywhere(rule):
    """Every law's LHS: identical substitution sequences per class."""
    eg, _ = _saturated_graph()
    got_classes = set()
    for eid in list(eg._classes.keys()):
        got = list(eg.matches(rule.lhs, eid))
        exp = _ref_matches(eg, rule.lhs, eid)
        assert got == exp, (eid, got, exp)
        if got:
            got_classes.add(eid)
    # attr-metavar / shared-metavar edges are exercised across the
    # corpus; this class loop is the corpus-wide oracle.


@given(terms())
@settings(max_examples=40, deadline=None)
def test_compiled_matches_interpreted_on_generated_terms(t):
    """Random terms: identical substitution sequences per class."""
    eg = EGraph(truncation_level=1)
    root = eg.add_term(t)
    for pat in (
        Op.make("add", "a", "b"),
        Op.make("mul", "a", Op.make("mul", "b", "c")),
        Op.make("add", "a", "a"),
        "a",
        Const(1),
    ):
        got = list(eg.matches(pat, root))
        exp = _ref_matches(eg, pat, root)
        assert got == exp


def test_compiled_matches_interpreted_with_caps():
    """``max_results`` truncation incl. the ok=False node-drop quirk."""
    eg, root = _saturated_graph()
    pat = Op.make("add", "a", Op.make("add", "b", "c"))
    for lim in (0, 1, 2, 3, 5, 256):
        got = list(eg.matches(pat, root, max_results=lim))
        exp = _ref_matches(eg, pat, root, max_results=lim)
        assert got == exp


def test_attr_metavar_consistency_compiled():
    """Repeated ``$attr:`` metavars bind consistently (k="S", k2="S")."""
    eg = EGraph()
    x = Var("x", TensorType((2, 2)))
    a = eg.add_term(Op.make("vf", x, k=7, k2=7))
    b = eg.add_term(Op.make("vf", x, k=7, k2=9))
    pat = Op.make("vf", "a", k="S", k2="S")
    assert list(eg.matches(pat, a)) == _ref_matches(eg, pat, a)
    assert list(eg.matches(pat, b)) == _ref_matches(eg, pat, b) == []
    # and a genuinely binding one
    pat2 = Op.make("vf", "a", k="S")
    assert list(eg.matches(pat2, b)) == _ref_matches(eg, pat2, b)


def test_compiled_prog_cache_and_unhashable():
    """Pattern programs cache per graph; unhashable leaves compile."""
    eg = EGraph()
    x = Var("x", TensorType((2, 2)))
    eid = eg.add_term(Op.make("f", x))
    pat = Op.make("f", "a")
    assert list(eg.matches(pat, eid))
    assert pat in eg._progs  # compiled + cached
    assert list(eg.matches(pat, eid))  # cache hit
    # unhashable non-Op pattern — degenerate leaf, compiles uncached
    assert list(eg.matches([1, 2], eid)) == []


# ---------------------------------------------------------------------------
#  Oracle 2 — streaming ≡ list applications
# ---------------------------------------------------------------------------


def _eager_iter(orig):
    """Wrap ``_iter_matches`` to materialise eagerly (old semantics)."""

    def wrapper(self, prog, eid, limit):
        return iter(list(orig(self, prog, eid, limit)))

    return wrapper


def _apply_seq(eg, rules, root, iters=8):
    st = eg.run(rules, root, max_iterations=iters, max_nodes=8000)
    return st


def _partition(eg):
    """Canonical class id of every e-node, keyed by the node itself."""
    return {
        n: eg.find(c) for c, ec in eg._classes.items() for n in ec.nodes
    }


@pytest.mark.parametrize("level", (1, 2))
def test_streaming_applications_identical(level):
    """Interleaved consumption ≡ eager materialisation, end-to-end."""
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    z = Var("z", TensorType((4, 4)))
    t = Op.make("add", Op.make("add", x, y), Op.make("add", z, x))
    rules = [COMM_ADD, ASSOC_ADD, ID_ADD, ID_MUL, COMM_MUL]

    eg1 = EGraph(truncation_level=level)
    r1 = eg1.add_term(t)
    s1 = _apply_seq(eg1, rules, r1)

    orig = CoreEG._iter_matches
    eg2 = EGraph(truncation_level=level)
    r2 = eg2.add_term(t)
    try:
        CoreEG._iter_matches = _eager_iter(orig)
        s2 = _apply_seq(eg2, rules, r2)
    finally:
        CoreEG._iter_matches = orig

    assert s1["n_enodes"] == s2["n_enodes"]
    assert s1["n_classes"] == s2["n_classes"]
    assert eg1.rule_fires == eg2.rule_fires
    assert _partition(eg1) == _partition(eg2)
    assert len(eg1._applications) == len(eg2._applications)
    assert eg1._applications == eg2._applications
    if level >= 2:
        assert eg1._merge_log == eg2._merge_log


def test_streaming_epoch_survives_mid_enumeration_union():
    """A union fired between yields is invisible to the enumeration."""
    eg = EGraph()
    a = eg.add_term(Var("a", TensorType((2, 2))))
    b = eg.add_term(Var("b", TensorType((2, 2))))
    # class with two matching members: f(a) and f(b)
    fa = eg.add_enode("f", (a,))
    fb = eg.add_enode("f", (b,))
    eg.union(fa, fb)
    cls = eg.find(fa)
    pat = Op.make("f", "v")
    gen = eg.matches(pat, cls)
    s1 = next(gen)
    # mutate mid-enumeration: merge the CHILD class a into b —
    # exactly what a fired application does.
    eg.union(a, b)
    rest = list(gen)
    gen.close()
    # the frozen epoch must replay the pre-union enumeration.
    eager = _ref_matches(eg, pat, cls)
    # NOTE: eager can't reproduce — compare against enumeration done
    # before the union.  Rebuild a fresh graph snapshot instead:
    eg2 = EGraph()
    a2 = eg2.add_term(Var("a", TensorType((2, 2))))
    b2 = eg2.add_term(Var("b", TensorType((2, 2))))
    fa2 = eg2.add_enode("f", (a2,))
    fb2 = eg2.add_enode("f", (b2,))
    eg2.union(fa2, fb2)
    cls2 = eg2.find(fa2)
    exp = _ref_matches(eg2, pat, cls2)
    assert [s1, *rest] == exp
    assert eager  # sanity: matches exist on the mutated graph too


def test_generator_close_releases_epoch():
    """Abandoned enumerations drop their epoch (finally -> remove)."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    f = eg.add_term(Op.make("f", Var("y", TensorType((2, 2)))))
    eg.union(x, f)
    gen = eg.matches(Op.make("f", "v"), eg.find(f))
    next(gen)
    assert len(eg._epochs) == 1
    gen.close()
    assert eg._epochs == []


def test_nested_epochs_journal_both():
    """A union while two enumerations are suspended journals both."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    y = eg.add_term(Var("y", TensorType((2, 2))))
    fx = eg.add_enode("f", (x,))
    fy = eg.add_enode("f", (y,))
    cls = eg.find(fx)
    pat = Op.make("f", "v")
    g1 = eg.matches(pat, cls)
    next(g1)
    g2 = eg.matches(pat, cls)
    next(g2)
    assert len(eg._epochs) == 2
    ep1, ep2 = eg._epochs
    eg.union(fx, fy)  # merges the class mid-both-enumerations
    # the merge journaled into BOTH live epochs
    loser = fx if eg.find(fx) == fy else fy
    assert loser in ep1.ovr and loser in ep2.ovr
    assert loser in ep1.dead and loser in ep2.dead
    # frozen view: each enumeration saw only its pre-union member —
    # the first yield already consumed it; nothing remains.
    r1 = list(g1)
    r2 = list(g2)
    g1.close()
    g2.close()
    assert r1 == r2 == []


# ---------------------------------------------------------------------------
#  Oracle 3 — incremental congruence postcondition
# ---------------------------------------------------------------------------


def _assert_closure_postcondition(eg):
    """The fixed point of congruence closure, stated directly.

    Every e-node's children are canonical, and no canonical e-node
    appears in two different e-classes.
    """
    owner = {}
    for eid, ec in eg._classes.items():
        for node in ec.nodes:
            canon = tuple(eg.find(c) for c in node.children)
            assert canon == node.children, (
                "non-canonical child left in class",
                eid,
                node,
            )
            prev = owner.setdefault(node, eid)
            assert prev == eid, (
                "canonical enode in two classes",
                node,
                prev,
                eid,
            )


@pytest.mark.parametrize("level", (1, 2))
def test_incremental_congruence_postcondition(level):
    """rebuild() reaches the same closure the full rescan produced."""
    x = Var("x", TensorType((2, 2)))
    y = Var("y", TensorType((2, 2)))
    eg = EGraph(truncation_level=level)
    nx = eg.add_term(Op.make("neg", x))
    ny = eg.add_term(Op.make("neg", y))
    nnx = eg.add_term(Op.make("neg", Op.make("neg", x)))
    nny = eg.add_term(Op.make("neg", Op.make("neg", y)))
    eg.union(eg.add_term(x), eg.add_term(y))
    eg.rebuild()
    assert eg.find(nx) == eg.find(ny)
    # congruence through two levels of enodes, via the worklist
    assert eg.find(nnx) == eg.find(nny)
    _assert_closure_postcondition(eg)
    # idempotent: a second rebuild still leaves the closure exact
    eg.rebuild()
    _assert_closure_postcondition(eg)


def test_incremental_congruence_under_saturation(level=2):
    """Closure postcondition holds after full saturation runs."""
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    z = Var("z", TensorType((4, 4)))
    t = Op.make("add", Op.make("add", x, y), Op.make("add", z, x))
    eg = EGraph(truncation_level=level)
    root = eg.add_term(t)
    eg.run(
        [COMM_ADD, ASSOC_ADD, ID_ADD, ID_MUL, COMM_MUL],
        root,
        max_iterations=8,
        max_nodes=8000,
    )
    _assert_closure_postcondition(eg)


def test_congruence_owner_eviction_path():
    """Re-keyed enodes lose their stale owner claim (no false merge)."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    y = eg.add_term(Var("y", TensorType((2, 2))))
    z = eg.add_term(Var("z", TensorType((2, 2))))
    g1 = eg.add_enode("g", (x, z))
    g2 = eg.add_enode("g", (y, z))
    h = eg.add_enode("h", (g1,))
    eg.rebuild()  # build owner map
    eg.union(x, y)  # g1's and g2's children re-canonicalise
    eg.rebuild()
    assert eg.find(g1) == eg.find(g2)
    _assert_closure_postcondition(eg)
    assert eg.find(h) in eg._classes


# ---------------------------------------------------------------------------
#  Lever 1c — rule scheduling
# ---------------------------------------------------------------------------


def test_run_orders_rules_by_ruleset_priorities():
    """EARLY-tagged rules apply before LATE ones within an iteration."""
    fired = []

    class Probe(EGraph):
        def apply_rule(self, rule, root_eid, **kw):
            fired.append(rule.name)
            return super().apply_rule(rule, root_eid, **kw)

    x = Var("x", TensorType((2, 2)))
    eg = Probe()
    root = eg.add_term(Op.make("add", x, Const(0)))
    rules = RuleSet(
        "sched",
        (COMM_ADD, ID_ADD),
    ).with_priorities(comm_add=20, id_add=0)
    eg.run(rules, root, max_iterations=1)
    assert fired[: len(rules)] == ["id_add", "comm_add"]


def test_run_plain_iterable_keeps_order():
    """A plain list has no priority_of — declaration order stands."""
    fired = []

    class Probe(EGraph):
        def apply_rule(self, rule, root_eid, **kw):
            fired.append(rule.name)
            return super().apply_rule(rule, root_eid, **kw)

    x = Var("x", TensorType((2, 2)))
    eg = Probe()
    root = eg.add_term(Op.make("add", x, Const(0)))
    eg.run([COMM_ADD, ID_ADD], root, max_iterations=1)
    assert fired[:2] == ["comm_add", "id_add"]


# ---------------------------------------------------------------------------
#  Lever 1d — lazy saturation
# ---------------------------------------------------------------------------


def test_improving_stop_records_stats():
    """stop='improving' terminates early and reports via stats."""
    x = Var("x", TensorType((4, 4)))
    eg = EGraph()
    root = eg.add_term(Op.make("mul", x, Const(1)))
    stats = eg.run(
        [ID_MUL, COMM_MUL],
        root,
        max_iterations=50,
        stop="improving",
        patience=2,
        cost_fn=count_cost,
    )
    assert stats["stop"] in ("improving", "fixed_point")
    assert "improved" in stats
    assert stats["iterations"] <= 50


def test_improving_requires_cost_fn_and_valid_stop():
    """Validation: bad stop value / missing cost_fn are errors."""
    eg = EGraph()
    root = eg.add_term(Var("x", TensorType((2, 2))))
    with pytest.raises(ValueError, match="stop must be"):
        eg.run([], root, stop="bogus")
    with pytest.raises(ValueError, match="needs a cost_fn"):
        eg.run([], root, stop="improving")


def test_improving_stall_breaks_early():
    """Cost plateau + still-growing graph -> patience stop, stats set."""
    x = Var("x", TensorType((2, 2)))
    y = Var("y", TensorType((2, 2)))
    z = Var("z", TensorType((2, 2)))
    w = Var("w", TensorType((2, 2)))
    eg = EGraph()
    root = eg.add_term(
        Op.make("add", Op.make("add", x, y), Op.make("add", z, w))
    )
    # comm/assoc churn grows the graph while count_cost is flat —
    # the patience counter fires before the fixed point.
    stats = eg.run(
        [COMM_ADD, ASSOC_ADD],
        root,
        max_iterations=50,
        stop="improving",
        patience=2,
        cost_fn=count_cost,
    )
    assert stats["stop"] == "improving"
    assert stats["improved"] == 1
    assert stats["iterations"] == 3


def test_search_plumbs_stop_and_patience():
    """``search(..., stop="improving", patience=..)`` reaches eg.run."""
    import torch
    import torch.nn as nn
    from catopt_orchestrator import search
    from catopt_torch.adapters import TorchSource

    torch.manual_seed(0)
    m = nn.Linear(8, 8, bias=False).double()
    x = torch.randn(2, 8, dtype=torch.float64)
    res = search(
        m,
        x,
        source=TorchSource(),
        stop="improving",
        patience=1,
        max_iterations=10,
    )
    assert res.stats["stop"] in ("improving", "fixed_point")
    assert "improved" in res.stats


def test_fixed_point_is_default_and_identical():
    """Default stop mode keeps the historical stats shape + term."""
    x = Var("x", TensorType((4, 4)))
    eg = EGraph()
    root = eg.add_term(Op.make("mul", x, Const(1)))
    stats = eg.run([ID_MUL, COMM_MUL], root, max_iterations=10)
    assert stats["stop"] == "fixed_point"
    assert "improved" not in stats


def test_max_iterations_stop_reason():
    """A run that never saturates reports stop='max_iterations'."""
    eg = EGraph(truncation_level=1)
    w = Param("w", TensorType((2, 2)))
    v = Param("v", TensorType((2, 2)))
    u = Param("u", TensorType((2, 2)))
    root = eg.add_term(Op.make("matmul", w, Op.make("matmul", v, u)))
    stats = eg.run(
        [ASSOC_MATMUL, ASSOC_MATMUL_REV],
        root,
        max_iterations=1,
    )
    assert stats["stop"] == "max_iterations"


# ---------------------------------------------------------------------------
#  Lever 1b — targeted eligibility
# ---------------------------------------------------------------------------


def test_candidate_classes_child_reqs_skip():
    """Classes failing a child-op constraint never open a match."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    # class has matmul members but none whose 2nd child is a matmul
    m1 = eg.add_enode("matmul", (x, eg.add_enode("leaf_op", (x,))))
    prog = eg._prog_for(
        Op.make("matmul", "a", Op.make("matmul", "b", "c"))
    )
    cands = list(eg._candidate_classes(prog, None))
    # m1's class is eligible by head op but filtered by child_reqs
    assert eg.find(m1) not in cands or all(
        eg._head_ok(prog, c) for c in cands
    )


def test_head_ok_true_when_constraint_met():
    """_head_ok returns True when a member satisfies child_reqs."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    inner = eg.add_enode("matmul", (x, x))
    outer = eg.add_enode("matmul", (x, inner))
    prog = eg._prog_for(
        Op.make("matmul", "a", Op.make("matmul", "b", "c"))
    )
    assert eg._head_ok(prog, eg.find(outer)) is True
    assert eg._head_ok(prog, eg.find(x)) is False
    # arity-skip arm: a class whose only matmul member is arity-3 —
    # the member is skipped for arity and nothing satisfies child_reqs
    m3 = eg.add_enode("matmul", (x, x, x))
    assert eg._head_ok(prog, eg.find(m3)) is False


def test_leaf_root_candidate_class():
    """A concrete-leaf LHS is eligible only at the leaf's class."""
    eg = EGraph()
    c = eg.add_term(Const(7))
    prog = eg._prog_for(Const(7))
    cands = list(eg._candidate_classes(prog, None))
    assert cands == [eg.find(c)]
    # a leaf pattern with no interned leaf — eligible set is empty
    prog2 = eg._prog_for(Const(8))
    assert list(eg._candidate_classes(prog2, None)) == []


def test_metavar_root_candidate_classes():
    """A metavariable LHS is eligible at every class."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    prog = eg._prog_for("v")
    cands = list(eg._candidate_classes(prog, None))
    assert eg.find(x) in cands


# ---------------------------------------------------------------------------
#  Bounded-matcher internals — the ``_m_bounded`` branch surface
# ---------------------------------------------------------------------------


def test_bounded_match_literal_attr_miss_and_arity():
    """Under a cap: literal-attr mismatch and arity mismatch skip."""
    eg = EGraph()
    x = Var("x", TensorType((2, 2)))
    a = eg.add_term(Op.make("vf", x, k=7))
    # literal attr that does not match -> attr_ok=False -> node skipped
    pat = Op.make("vf", "a", k=8)
    assert (
        list(eg.matches(pat, a, max_results=4))
        == _ref_matches(eg, pat, a, max_results=4)
        == []
    )
    # node whose attr KEYS differ from the pattern's -> keyset skip
    pat2 = Op.make("vf", "a", k="S", k2="T")
    assert (
        list(eg.matches(pat2, a, max_results=4))
        == _ref_matches(eg, pat2, a, max_results=4)
        == []
    )
    # an arity-mismatched member in the class: f(x) vs f(x,y)
    f1 = eg.add_enode("f", (a,))
    f2 = eg.add_enode("f", (a, a))
    eg.union(f1, f2)
    pat3 = Op.make("f", "u", "v")
    assert list(eg.matches(pat3, eg.find(f1), max_results=4)) == (
        _ref_matches(eg, pat3, eg.find(f1), max_results=4)
    )


def test_bounded_match_attr_metavar_repeat():
    """Repeated ``$attr:`` metavars under a cap — consistent+conflict."""
    eg = EGraph()
    x = Var("x", TensorType((2, 2)))
    same = eg.add_term(Op.make("vf", x, k=7, k2=7))
    diff = eg.add_term(Op.make("vf", x, k=7, k2=9))
    pat = Op.make("vf", "a", k="S", k2="S")
    for eid, exp_n in ((same, 1), (diff, 0)):
        got = list(eg.matches(pat, eid, max_results=4))
        assert got == _ref_matches(eg, pat, eid, max_results=4)
        assert len(got) == exp_n


def test_bounded_match_shared_metavar_and_literal():
    """Shared-metavar bound arm + literal-attr match arc, under caps."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    y = eg.add_term(Var("y", TensorType((2, 2))))
    f_same = eg.add_enode("f", (x, x))
    f_diff = eg.add_enode("f", (x, y))
    eg.union(f_same, f_diff)
    cls = eg.find(f_same)
    pat = Op.make("f", "v", "v")
    for lim in (1, 4):
        got = list(eg.matches(pat, cls, max_results=lim))
        assert got == _ref_matches(eg, pat, cls, max_results=lim)
        assert got == [{"v": x}]
    # literal attr that MATCHES: the ``nv == pv`` fall-through arc
    z = eg.add_term(Var("z", TensorType((2, 2))))
    vf = eg.add_term(Op.make("vf", Var("z", TensorType((2, 2))), k=7))
    pat2 = Op.make("vf", "a", k=7)
    assert list(eg.matches(pat2, vf, max_results=2)) == [{"a": z}]


def test_bounded_match_leaf_paths():
    """Concrete-leaf patterns under a cap — hit and miss arms."""
    eg = EGraph()
    c = eg.add_term(Const(9))
    x = eg.add_term(Var("x", TensorType((2, 2))))
    eg.union(c, x)  # leaf + var share a class
    cls = eg.find(c)
    assert list(eg.matches(Const(9), cls, max_results=1)) == [{}]
    assert (
        list(eg.matches(Const(4), cls, max_results=1))
        == _ref_matches(eg, Const(4), cls, max_results=1)
        == []
    )


def test_bounded_match_enode_cap_drop():
    """The ok=False drop: a node crossing the cap yields nothing."""
    eg = EGraph()
    # one class, many matching members: f1(v)... each binds differently
    m = [
        eg.add_enode(
            "f", (eg.add_term(Var(f"v{i}", TensorType((2, 2)))),)
        )
        for i in range(4)
    ]
    for other in m[1:]:
        eg.union(m[0], other)
    cls = eg.find(m[0])
    pat = Op.make("f", "v")
    for lim in (1, 2, 3):
        assert list(eg.matches(pat, cls, max_results=lim)) == (
            _ref_matches(eg, pat, cls, max_results=lim)
        )


def _two_pos_graph():
    """P = {f(Cx), f(Cy)} where Ci = {g(i)} — one subst per member."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    y = eg.add_term(Var("y", TensorType((2, 2))))
    gx = eg.add_enode("g", (x,))
    gy = eg.add_enode("g", (y,))
    f1 = eg.add_enode("f", (gx,))
    f2 = eg.add_enode("f", (gy,))
    eg.union(f1, f2)
    return eg, x, y, gx, gy, eg.find(f1)


def test_frozen_read_into_dead_class():
    """A child class merged away mid-enumeration stays readable."""
    eg, x, y, gx, gy, P = _two_pos_graph()
    pat = Op.make("f", Op.make("g", "v"))
    gen = eg.matches(pat, P)
    s1 = next(gen)
    # Kill whichever g-class the enumeration has NOT descended into
    # yet: a size-2 class always wins the union, so this one dies.
    w = eg.add_term(Var("w", TensorType((2, 2))))
    big = eg.add_enode("h", (w,))
    big2 = eg.add_enode("g", (w,))
    eg.union(big, big2)
    victim = gy if s1["v"] == x else gx
    eg.union(eg.find(big), victim)
    rest = list(gen)
    gen.close()
    # the dead class is read through the epoch's ``dead`` overlay:
    # both substitutions are still produced, against frozen ids.
    assert sorted(s["v"] for s in [s1, *rest]) == sorted([x, y])


def test_frozen_members_exclude_mid_epoch_additions():
    """Members merged into a class mid-enumeration stay invisible."""
    eg, x, y, gx, gy, P = _two_pos_graph()
    pat = Op.make("f", Op.make("g", "v"))
    gen = eg.matches(pat, P)
    s1 = next(gen)
    # Grow whichever g-class the enumeration has NOT yet visited:
    # it gets a second member mid-epoch.  (Pad it first so it wins.)
    w = eg.add_term(Var("w", TensorType((2, 2))))
    target = gy if s1["v"] == x else gx
    eg.union(target, eg.add_enode("h", (w,)))
    eg.union(eg.find(target), eg.add_enode("g", (w,)))
    rest = list(gen)
    gen.close()
    # The grown-in g(w) member is post-freeze — the frozen member
    # list subtracts it, so exactly the two pre-freeze substs yield.
    assert sorted(s["v"] for s in [s1, *rest]) == sorted([x, y])


def test_cong_drain_tolerates_stale_pends():
    """The worklist discards pends that are not class members."""
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    y = eg.add_term(Var("y", TensorType((2, 2))))
    g = eg.add_enode("g", (x,))
    # Pad y's class so y survives — x's death pends the g(x) enode,
    # which rebuild() then canonicalises OUT of the class before the
    # stale (gclass, g(x)) pend drains -> ``node = nn`` re-claim arm.
    eg.union(y, eg.add_enode("e", (y,)))
    eg.union(x, y)
    canon = eg.find(g)
    # pend 1: enode that was never a member, children already
    # canonical -> the ``node not in nodes`` arm.
    foreign = ENode("zzz", (eg.find(x),))
    eg._cong_pend.add((canon, foreign))
    # pend 2: enode that re-keys to a non-member -> ``nn not in
    # eclass.nodes`` arm.  A stale child id forces re-keying.
    dead = eg.add_enode("q", (x,))
    keep = eg.add_enode("q2", (x,))
    eg.union(dead, keep)
    loser = dead if eg.find(dead) != dead else keep
    stale = ENode("g", (loser,))  # loser is no longer canonical
    eg._cong_pend.add((canon, stale))
    eg.rebuild()
    _assert_closure_postcondition(eg)


def test_cong_drain_rekey_owner_miss():
    """A mid-drain re-key of an unclaimed member (owner.get -> None).

    Simulates the cascade case: a member whose canonical form changes
    *after* the pass's canonicalisation — union-pends popped during
    the merge rounds hit ``owner.get`` with no prior claim.
    """
    eg = EGraph()
    x = eg.add_term(Var("x", TensorType((2, 2))))
    C = eg.add_enode("g", (x,))
    dead = eg.add_enode("q", (x,))
    live = eg.add_enode("q2", (x,))
    eg.union(dead, live)
    loser = dead if eg.find(dead) != dead else live
    stale = ENode("g", (loser,))  # re-keys: child is not canonical
    ccls = eg.find(C)
    eg._classes[ccls].nodes.add(stale)  # injected mid-drain member
    eg._cong_pend.add((ccls, stale))
    eg.rebuild()
    canon_n = ENode("g", (eg.find(loser),))
    assert canon_n in eg._classes[ccls].nodes
    _assert_closure_postcondition(eg)
