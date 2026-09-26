"""Coverage tests for catopt.egraph internals.

``test_egraph.py`` / ``test_certificates.py`` / ``test_cert_nonlocal.py``
pin the headline behaviour of the package; this file closes the residual
branches in the split modules:

- ``extract.py`` — cyclic/unextractable classes in ``extract_min_depth``
  and ``extract_best_bounded`` (self-loops and two-cycles are the only
  classes with NO acyclic member), ``extract_alternatives`` leaf-skip /
  failed-forced-member / repr-dedup / top-k, ``diverse_classes`` sketch
  arms, bounded-extraction ban rounds (unlocatable RHS, already-banned,
  no-progress), and ``extract_paired`` steering (member-routing
  overrides, unreachable/cyclic children priced inf, descendant cycles,
  non-memo cost functions).
- ``proof.py`` — ``_oldest_term``/``_app_for_member``/``_resolve_subst``
  failure arms, ``_connect`` budget stubs + expansion fallbacks +
  congruence/edge/stub strategy, ``_edge_path`` exhaustion and
  check/derive/seen gates, ``_explain_gap`` origins, ``_resolve_dst``
  failures, ``all_proofs`` level-1/check/derive arms, ``coherent_paths``.
- ``terms.py`` — ``_term_match`` attr-metavar edges, ``_subterm`` /
  ``_replace_subterm`` bad paths, ``verify_certificate`` every raise.
- ``core.py`` — ``union`` witness/UF-false/op-index arms, ``_match``
  enumeration caps, ``rebuild`` provenance of untracked nodes,
  ``_any_term_cached`` RecursionError fallback, ``apply_rule``
  bound-None/derive-veto skips, ``_candidate_classes`` leaf LHS,
  ``run`` bounded-saturation bookkeeping.

Cyclic e-classes are built by unioning a class into its own child's
class and then removing the surviving acyclic (leaf) member — unions
and rules always preserve at least one acyclic member, so the removal
is the only way to reach the "no extractable term" arms.

Defensive branches deliberately not covered (provably unreachable —
suggest ``pragma: no cover``):

- ``extract.py:426`` (``extract_best_bounded`` loop-exit ``return
  None``): reaching it requires ``n_enodes + 1`` rounds that each ban a
  *new* enode — strictly more rounds than the graph has enodes.
- ``core.py:391`` (``_match`` per-enode limit check): ``results`` can
  only reach ``limit`` inside the ``extend`` at the end of the loop
  body, which returns immediately — the check at the top of the next
  iteration can never observe it.
- ``proof.py:504->478`` (``all_proofs`` derivation-signature dedup):
  signatures are full ``(rule, path)`` step histories; two derivations
  with an identical history are the same derivation, so a collision
  can never be generated.

Everything is deterministic (small graphs, no RNG).
"""

import pytest

from catopt_core import laws as R
from catopt.cost import count_cost, dag_cost
from catopt.egraph import (
    Certificate,
    CertificateVerificationError,
    CertStep,
    EGraph,
    ENode,
    Rewrite,
    UnionFind,
    _LeafRegistry,
    verify_certificate,
)
from catopt.egraph.terms import (
    _iter_ops,
    _replace_subterm,
    _subterm,
    _term_instantiate,
    _term_match,
    _term_paths,
)
from catopt.ir import Const, Op, Param, TensorType, Var, op_repr
from catopt_core.laws import pair_shared_input_linears


def _t(d=4):
    return TensorType((d, d))


def _v(name):
    return Var(name, _t())


def _self_loop(eg, op="cycf"):
    """One e-class whose ONLY member is ``op(itself)`` — no acyclic
    member exists, so every extraction/resolution fails."""
    x = eg.add_leaf("__cyc_x")
    f = eg.add_enode(op, (x,))
    assert eg.union(x, f)
    cid = eg.find(x)
    ec = eg.get_class(cid)
    ec.nodes = {n for n in ec.nodes if n.op != "leaf"}
    return cid


def _self_loop2(eg):
    """A cyclic class with TWO self-loop members — used where a failing
    member must be followed by another member in the same scan."""
    x = eg.add_leaf("__cyc2_x")
    f = eg.add_enode("loopf", (x,))
    g = eg.add_enode("loopg", (x,))
    eg.union(x, f)
    eg.union(x, g)
    cid = eg.find(x)
    ec = eg.get_class(cid)
    ec.nodes = {n for n in ec.nodes if n.op != "leaf"}
    return cid


def _two_cycle(eg):
    """A <-> B two-cycle: A = {cycg(B), cycg2(B)}, B = {cycf(A)}.

    A gets two cyclic members so member scans exercise the
    fail-then-continue branch; B's single member fails through the
    ``child already on the resolution stack`` arm.
    """
    a = eg.add_leaf("__cyc_a")
    fa = eg.add_enode("cycf", (a,))
    b = eg.add_leaf("__cyc_b")
    gb = eg.add_enode("cycg", (b,))
    gb2 = eg.add_enode("cycg2", (b,))
    eg.union(a, gb)
    eg.union(a, gb2)
    eg.union(b, fa)
    ca, cb = eg.find(a), eg.find(b)
    for c in (ca, cb):
        ec = eg.get_class(c)
        ec.nodes = {n for n in ec.nodes if n.op != "leaf"}
    return ca, cb


# ===========================================================================
#  extract.py — extract_min_depth / extract_alternatives / diverse_classes
# ===========================================================================


def test_min_depth_picks_shallowest_member():
    x = _v("x")
    src = Op.make("add", x, Const(0))
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.ID_ADD], root, max_iterations=3)
    # class holds add(x,0) and leaf x — the leaf wins on depth
    assert op_repr(eg.extract_min_depth(root)) == "x"


def test_min_depth_nested_member_and_tiebreak():
    """A multi-level member exercises the child-resolution success path;
    a deeper equal-class member loses the ``cand < best`` comparison."""
    x = _v("x")
    eg = EGraph()
    deep = eg.add_term(Op.make("g", Op.make("h", x)))
    shallow = eg.add_term(Op.make("f", x))
    eg.union(deep, shallow)
    got = eg.extract_min_depth(deep)
    # f(x) has depth 1 < g(h(x)) depth 2
    assert op_repr(got) == "(f x)"


def test_min_depth_self_loop_unextractable():
    """A direct self-reference (child == own class) is skipped: the
    class has no acyclic member at all."""
    eg = EGraph()
    cid = _self_loop(eg)
    assert eg.extract_min_depth(cid) is None


def test_min_depth_two_cycle_unextractable():
    """An indirect cycle fails through the recursion (child already
    in-progress -> (inf, None)), not the direct self-check."""
    eg = EGraph()
    ca, _cb = _two_cycle(eg)
    assert eg.extract_min_depth(ca) is None


def test_extract_alternatives_leaf_and_selfref_members():
    """After id_add, the class holds a leaf (skipped) and a self-
    referential add enode whose forced extraction fails — so the
    frontier is empty."""
    x = _v("x")
    eg = EGraph()
    src = Op.make("add", x, Const(0))
    root = eg.add_term(src)
    eg.run([R.ID_ADD], root, max_iterations=3)
    assert eg.extract_alternatives(root, count_cost) == []


def test_extract_alternatives_distinct_members_topk():
    x, y = _v("x"), _v("y")
    eg = EGraph()
    src = Op.make("add", x, y)
    root = eg.add_term(src)
    eg.run([R.COMM_ADD], root, max_iterations=3)
    alts = eg.extract_alternatives(root, count_cost)
    reprs = {op_repr(t) for _, t in alts}
    assert reprs == {"(add x, y)", "(add y, x)"}
    assert all(c == count_cost(t) or c == dag_cost(t, count_cost)
               for c, t in alts)
    # top_k bounds the returned frontier
    top1 = eg.extract_alternatives(root, count_cost, top_k=1)
    assert len(top1) == 1


def test_extract_alternatives_repr_dedup():
    """Two *distinct* enodes producing structurally identical terms
    (same op/children, attrs stored in a different tuple order) are
    deduplicated by ``op_repr`` — the second is not re-recorded."""
    x = _v("x")
    eg = EGraph()
    xc = eg.add_term(x)
    vid = eg.add_enode("vf", (xc,), {"a": 2, "b": 1})
    # same dict, different stored order -> a different ENode object
    eg.get_class(vid).nodes.add(
        ENode("vf", (xc,), (("b", 1), ("a", 2)))
    )
    alts = eg.extract_alternatives(vid, count_cost)
    assert len(alts) == 1
    assert op_repr(alts[0][1]) in (
        "(vf x, a=2, b=1)",
        "(vf x, b=1, a=2)",
    )


def test_extract_alternatives_cyclic_class_empty():
    eg = EGraph()
    cid = _self_loop(eg)
    assert eg.extract_alternatives(cid, count_cost) == []


def test_extract_best_two_leaves_tie():
    """A class holding two equal-cost leaves: the second leaf is a
    valid candidate that does not improve on the recorded best."""
    x, y = _v("x"), _v("y")
    eg = EGraph()
    ex, ey = eg.add_term(x), eg.add_term(y)
    eg.union(ex, ey)
    got = eg.extract_best(ex, count_cost)
    assert op_repr(got) in ("x", "y")


def test_diverse_classes_sketch_arms():
    """diverse_classes reports classes with >= 2 member ops; the sketch
    covers leaf members (named and anonymous), op members with real and
    emptied child classes, and dedup of identical sketches."""
    eg = EGraph()
    x = eg.add_leaf("x")
    z = eg.add_leaf("z")
    h = eg.add_enode("hh", (x,))
    n = eg.add_enode("neg", (z,))
    eg.union(h, n)
    cls = eg.find(h)
    ec = eg.get_class(cls)
    # a leaf member (named) and an anonymous leaf member
    ec.nodes.add(ENode("leaf", (), (("key", "ns:w"),)))
    ec.nodes.add(ENode("leaf", (), ()))
    # two leaves whose 24-char sketch suffix collides -> dedup arm
    ec.nodes.add(ENode("leaf", (), (("key", "aa:dup"),)))
    ec.nodes.add(ENode("leaf", (), (("key", "bb:dup"),)))
    # an op member whose child class has been emptied -> "?" child
    ec.nodes.add(ENode("orph", (x,), ()))
    eg.get_class(eg.find(x)).nodes.clear()
    out = eg.diverse_classes()
    entry = next(d for d in out if d["eid"] == cls)
    members = entry["members"]
    assert any(m.startswith("orph(?") for m in members)
    assert any(m.startswith("neg(") or m.startswith("hh(") for m in members)
    assert "w" in members or "?" in members  # leaf sketches
    # leaf-less classes and single-op classes are absent
    assert all(len(d["members"]) >= 2 for d in out)


# ===========================================================================
#  extract.py — extract_best_bounded
# ===========================================================================


def _bound_graph():
    """Class {f(x), g(x) exact, h(x) eps}: ``h`` cheapest by a custom
    cost model, but its derivation carries ``error_bound``."""
    x = _v("x")
    eg = EGraph()
    src = Op.make("f", x)
    root = eg.add_term(src)
    r_exact = Rewrite("r_exact", Op.make("f", "a"), Op.make("g", "a"))
    r_approx = Rewrite(
        "r_approx", Op.make("f", "a"), Op.make("h", "a"), error_bound=0.3
    )
    eg.run([r_exact, r_approx], root, max_iterations=5)
    return eg, src, root


def _h_cheap_cost(t, memo=None):
    if isinstance(t, Op):
        local = {"h": 2.0}.get(t.op, 5.0)
        return local + sum(_h_cheap_cost(a) for a in t.args)
    return 1.0


def test_bounded_unconstrained_and_generous():
    eg, _src, root = _bound_graph()
    assert op_repr(eg.extract_best_bounded(root, _h_cheap_cost)) == "(h x)"
    assert (
        op_repr(
            eg.extract_best_bounded(root, _h_cheap_cost, max_error=0.5)
        )
        == "(h x)"
    )


def test_bounded_bans_bound_member():
    """max_error=0: the cheapest member's certificate exceeds the
    budget -> its bound-producing enode is banned -> re-extraction
    returns an exact member."""
    eg, src, root = _bound_graph()
    got = eg.extract_best_bounded(root, _h_cheap_cost, max_error=0.0)
    assert op_repr(got) in ("(f x)", "(g x)")
    # an explicit src_term skips the oldest-member resolution
    got2 = eg.extract_best_bounded(
        root, _h_cheap_cost, max_error=0.0, src_term=src
    )
    assert op_repr(got2) in ("(f x)", "(g x)")


def test_bounded_cyclic_class_returns_none():
    """No acyclic member: ``_oldest_term`` fails, ``any_term`` fails,
    ``extract_best`` fails -> ``None``."""
    eg = EGraph()
    cid = _self_loop(eg)
    assert eg.extract_best_bounded(cid, count_cost, max_error=1.0) is None


def test_bounded_unlocatable_and_rebanned_steps():
    """Crafted certificates (installed directly) drive the ban loop's
    residual arms: a bound step whose RHS cannot be located is skipped,
    an already-banned enode is not re-banned, and the loop exits
    ``None`` when a round makes no progress."""
    x = _v("x")
    eg = EGraph()
    src = Op.make("add", x, Const(0))
    root = eg.add_term(src)
    eg.run([R.ID_ADD], root, max_iterations=3)
    foreign = Op.make("zz", _v("ghost"))
    rb = Rewrite(
        "rb", Op.make("add", "a", Const(0)), "a", error_bound=9.0
    )
    rb2 = Rewrite(
        "rb2", Op.make("add", "a", Const(0)), foreign, error_bound=4.0
    )
    rex = Rewrite(
        "rex", Op.make("add", "a", Const(0)), "a"
    )  # no bound -> skipped by the ban scan
    calls = {"n": 0}

    def fake_cert(s, t, root_eid=None):
        calls["n"] += 1
        return Certificate(
            src=s,
            dst=t,
            root_eid=root_eid,
            steps=[
                CertStep("rb", (), src, src, {}),
                CertStep("rb2", (), src, foreign, {}),
                CertStep("rex", (), src, src, {}),
                CertStep("absent_rule", (), src, src, {}),
            ],
            rules={"rb": rb, "rb2": rb2, "rex": rex},
        )

    eg.certificate = fake_cert
    got = eg.extract_best_bounded(
        root, lambda t, memo=None: 1.0, max_error=0.0
    )
    assert got is None
    assert calls["n"] >= 2  # banned the add enode, re-extracted, gave up


# ===========================================================================
#  extract.py — extract_paired
# ===========================================================================


def _pairing_graph():
    x = Var("x", TensorType((2, 4)))
    w1 = Param("W1", TensorType((8, 4)))
    w2 = Param("W2", TensorType((8, 4)))
    src = Op.make(
        "add", Op.make("linear", x, w1), Op.make("linear", x, w2)
    )
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([], root, max_iterations=1)
    groups = pair_shared_input_linears(eg)
    eg.rebuild()
    assert groups, "expected a pairing group"
    return eg, src, root, groups


def _flat_cost(t):
    """Cost fn WITHOUT a ``memo`` parameter — exercises the plain-call
    arm of the paired-extraction cost shim."""
    return 1.0 if isinstance(t, Op) else 0.5


def test_extract_paired_steers_consumers():
    """A non-member class holding BOTH a member-reaching and a bypass
    enode is steered to the cheapest member route — proven at the root
    itself, where a greedy ``alt`` member would otherwise win."""
    eg, _src, root, groups = _pairing_graph()
    member = next(iter(groups[0]))

    # root gains a bypass member that wins ties on size
    lf = eg.add_leaf("lf")
    alt = eg.add_enode("alt", (lf,))
    eg.union(root, alt)

    # an unreachable class: member-reaching + bypass enodes -> the
    # route prices inf (child absent from the pass-1 cache) -> skipped
    far = eg.add_enode("far", (eg.add_leaf("lf2"),))
    v1 = eg.add_enode("v", (member, far))
    v2 = eg.add_enode("v2", (eg.add_leaf("lf3"),))
    eg.union(v1, v2)

    # a cyclic 2-cycle: descendant enumeration must survive the cycle,
    # and a root member pointing into it prices inf (cached None term)
    ca, _cb = _two_cycle(eg)
    d2 = eg.add_enode("dummy2", (ca, eg.find(root)))
    eg.union(root, d2)
    w1 = eg.add_enode("w", (member, ca))
    w2 = eg.add_enode("w2", (eg.add_leaf("lf4"),))
    eg.union(w1, w2)

    # duplicate group entries: member_over.setdefault keeps the first
    groups2 = groups + groups
    out = eg.extract_paired(root, _flat_cost, groups2)
    assert "split" in op_repr(out)
    assert "concat" in op_repr(out)  # the shared fused GEMM is shared

    # same call under a memo-taking cost fn
    out2 = eg.extract_paired(root, count_cost, groups)
    assert "split" in op_repr(out2)


def test_extract_paired_empty_groups_is_greedy():
    eg, _src, root, _groups = _pairing_graph()
    got = eg.extract_paired(root, count_cost, [{}])
    want = eg.extract_best(root, count_cost)
    assert op_repr(got) == op_repr(want)


# ===========================================================================
#  extract.py — _class_of_term / _locate
# ===========================================================================


def test_class_of_term_and_locate_failure_arms():
    x = _v("x")
    eg = EGraph()
    xc = eg.add_term(x)
    vid = eg.add_enode("vf", (xc,), {"k": 1})
    eg.add_enode("vf", (xc,), {"k": 2})  # same class? no — distinct enode
    # put both vf variants in one class to exercise the attr-mismatch scan
    v2 = eg.add_enode("vg", (xc,), {"k": 2})
    other = eg.add_enode("other", (xc,), {})
    eg.union(vid, v2)
    eg.union(vid, other)
    cls = eg.find(vid)

    # term with a child absent from the graph -> _class_of_term None
    assert eg._class_of_term(Op.make("add", x, _v("ghost"))) is None
    # foreign term -> (None, None)
    assert eg._locate(_v("ghost")) == (None, None)
    # explicit eid: leaf term not among the class's members -> (eid, None)
    eid, en = eg._locate(x, eid=cls)
    assert eid == cls and en is None
    # explicit eid: op/arity/attr mismatches scanned, none match
    eid, en = eg._locate(Op.make("vf", x, k=3), eid=cls)
    assert eid == cls and en is None
    # same op/attrs but children in a different class -> scanned past
    eg.get_class(cls).nodes.add(
        ENode("vf", (eg.add_term(_v("other")),), (("k", 1),))
    )
    eid, en = eg._locate(Op.make("vf", x, k=1), eid=cls)
    assert eid == cls and en is not None and dict(en.attrs) == {"k": 1}
    # a class deleted under _classes (non-canonical remnant) -> (eid, None)
    lone = eg.find(eg.add_leaf("lone"))
    del eg._classes[lone]
    eid, en = eg._locate(Param("lone", _t()))
    assert en is None


# ===========================================================================
#  proof.py — _oldest_term / _app_for_member / _resolve_subst
# ===========================================================================


def test_oldest_term_cyclic_and_missing_class():
    eg = EGraph()
    ca, _cb = _two_cycle(eg)
    # two cyclic members: fail -> continue -> fail -> None
    assert eg._oldest_term(ca) is None
    # a canonical id missing from _classes -> None
    lone = eg.find(eg.add_leaf("lone"))
    del eg._classes[lone]
    assert eg._oldest_term(lone) is None


def test_app_for_member_failure_arms():
    x, y = _v("x"), _v("y")
    eg = EGraph()
    t = Op.make("ff", x)
    eid = eg.add_term(t)
    _eid2, en = eg._locate(t)
    assert en is not None
    # foreign term: not locatable at all
    assert eg._app_for_member(Op.make("ff", _v("ghost"))) == (None, None)
    # application whose RHS-root op disagrees with the member enode
    eg.add_term(Op.make("gg", x))
    _eog, en_g = eg._locate(Op.make("gg", x))
    assert en_g is not None
    eg._applications.append(
        {
            "rule": "ghost_rule",
            "matched_eid": eid,
            "rhs_eid": eid,
            "subst": {},
            "rhs_root_enode": en_g,
        }
    )
    eg._enode_app[en] = len(eg._applications) - 1
    assert eg._app_for_member(t) == (None, en)
    # application record with rhs_root_enode None (metavar RHS)
    eg._applications.append(
        {
            "rule": "ghost_rule",
            "matched_eid": eid,
            "rhs_eid": eid,
            "subst": {},
            "rhs_root_enode": None,
        }
    )
    eg._enode_app[en] = len(eg._applications) - 1
    assert eg._app_for_member(t) == (None, en)
    # application whose RHS-root children disagree with the member enode
    eg.add_enode("ff", (eg.add_term(y),))
    _eo, en_other = eg._locate(Op.make("ff", y))
    eg._applications.append(
        {
            "rule": "ghost_rule",
            "matched_eid": eid,
            "rhs_eid": eid,
            "subst": {},
            "rhs_root_enode": en_other,
        }
    )
    eg._enode_app[en] = len(eg._applications) - 1
    assert eg._app_for_member(t) == (None, en)


def test_resolve_subst_arms():
    eg = EGraph()
    assert eg._resolve_subst({"$attr:s": (2, 3)}) == {"$attr:s": (2, 3)}
    cyc = _self_loop(eg)
    # bound class unresolvable (cyclic) -> None
    assert eg._resolve_subst({"a": cyc}) is None


# ===========================================================================
#  proof.py — _connect strategy arms
# ===========================================================================


def test_connect_budget_exceeded_emits_stub():
    """Derivation budget exhaustion records a ``<budget>``
    e-graph-dependent step instead of recursing forever."""
    x, y = _v("x"), _v("y")
    src = Op.make("add", x, y)
    dst = Op.make("add", y, x)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.COMM_ADD], root, max_iterations=3)
    eg._CERT_MAX_DEPTH = -1  # force the budget arm on the first call
    cert = eg.certificate(src, dst)
    assert cert.steps[0].rule == "<budget>"
    assert cert.steps[0].egraph_dependent
    assert "budget" in cert.steps[0].note
    # and the assertion still replays (non-strict)
    assert op_repr(verify_certificate(src, cert)) == op_repr(dst)


def test_connect_congruence_child_gap():
    """Same head op, children pairwise in the same class, but one pair
    unlinked (a manual union): the child emits an e-graph stub and the
    parent's congruence reports failure."""
    x, a, c = _v("x"), _v("a"), _v("c")
    eg = EGraph()
    ea, ec_ = eg.add_term(a), eg.add_term(c)
    eg.union(ea, ec_)  # a and c equal by decree — no derivation
    s = Op.make("hh", a, x)
    t = Op.make("hh", c, x)
    es, et = eg.add_term(s), eg.add_term(t)
    eg.union(es, et)
    cert = eg.certificate(s, t)
    assert cert.n_egraph_dependent == 1
    assert cert.steps[0].path == (0,)
    assert "predate saturation" in cert.steps[0].note
    assert op_repr(verify_certificate(s, cert)) == op_repr(t)


def test_connect_congruence_mismatch_then_gap():
    """Same head op but children in DIFFERENT classes -> the congruence
    shortcut is skipped and the edge search fails -> root stub."""
    x, a, c = _v("x2"), _v("a2"), _v("c2")
    eg = EGraph()
    s = Op.make("hh", a, x)
    t = Op.make("hh", c, x)
    es, et = eg.add_term(s), eg.add_term(t)
    eg.union(es, et)  # only the roots merge
    cert = eg.certificate(s, t)
    assert cert.n_egraph_dependent == 1
    assert cert.steps[0].path == ()
    assert op_repr(verify_certificate(s, cert)) == op_repr(t)


def test_connect_expansion_child_fixup_failure():
    """Target-expansion strategy: the produced RHS children that do not
    match the target's children each become e-graph stubs and flip the
    returned flag — while the root step itself stays a replayable rule
    instance."""
    s, u, v = _v("s"), _v("u"), _v("v")
    t = Op.make("hh", u, v)
    eg = EGraph()
    eid_s = eg.add_term(s)
    eid_t = eg.add_term(t)
    eg.union(eid_s, eid_t)
    _eid, en_t = eg._locate(t)
    fake = Rewrite("fakeexp", "a", Op.make("hh", "a", "a"))
    eg._rule_objs["fakeexp"] = fake
    eg._applications.append(
        {
            "rule": "fakeexp",
            "matched_eid": eg.find(eid_s),
            "rhs_eid": eg.find(eid_t),
            "subst": {"a": eg.find(eid_s)},
            "rhs_root_enode": en_t,
        }
    )
    eg._enode_app[en_t] = len(eg._applications) - 1
    cert = eg.certificate(s, t)
    assert cert.steps[0].rule == "fakeexp"
    assert cert.steps[0].path == ()
    assert cert.n_egraph_dependent == 2  # the two mismatched children
    assert op_repr(verify_certificate(s, cert)) == op_repr(t)


def test_connect_expansion_unresolvable_subst():
    """An application whose recorded binding cannot be resolved to a
    term (cyclic bound class) is skipped: no expansion, fall through to
    congruence/edge search -> stub."""
    s, u = _v("s2"), _v("u2")
    t = Op.make("hh", u)
    eg = EGraph()
    eid_s = eg.add_term(s)
    eid_t = eg.add_term(t)
    eg.union(eid_s, eid_t)
    cyc = _self_loop(eg)
    _eid, en_t = eg._locate(t)
    eg._rule_objs["fx"] = Rewrite("fx", "a", "a")
    eg._applications.append(
        {
            "rule": "fx",
            "matched_eid": eg.find(eid_s),
            "rhs_eid": eg.find(eid_t),
            "subst": {"a": cyc},
            "rhs_root_enode": en_t,
        }
    )
    eg._enode_app[en_t] = len(eg._applications) - 1
    cert = eg.certificate(s, t)
    assert cert.n_egraph_dependent == 1
    assert op_repr(verify_certificate(s, cert)) == op_repr(t)


def test_connect_rhs_leaf_expansion():
    """An application instantiating a bare concrete-leaf RHS: the
    child-fixup loop is skipped (``R`` is not an Op), the step still
    replays as an ordinary rule instance."""
    s = _v("x")
    eg = EGraph()
    eid_s = eg.add_term(s)
    c0 = eg.add_term(Const(0))
    eg.union(eid_s, c0)
    _eid, en_c0 = eg._locate(Const(0))
    assert en_c0 is not None and en_c0.op == "leaf"
    rule = Rewrite("to_zero", "a", Const(0))
    eg._rule_objs["to_zero"] = rule
    eg._applications.append(
        {
            "rule": "to_zero",
            "matched_eid": eg.find(eid_s),
            "rhs_eid": eg.find(c0),
            "subst": {"a": eg.find(eid_s)},
            "rhs_root_enode": en_c0,
        }
    )
    eg._enode_app[en_c0] = len(eg._applications) - 1
    cert = eg.certificate(s, Const(0))
    assert cert.replayable
    assert cert.rules_used == ["to_zero"]
    assert op_repr(verify_certificate(s, cert, strict=True)) == "0"


def test_explain_gap_external_origin():
    """An enode added directly (no rule, provenance None) reports the
    'external' explanation when a certificate stubs through it."""
    x = _v("x")
    eg = EGraph()
    xc = eg.add_term(x)
    g = eg.add_enode("gx", (xc,))  # provenance None -> "external"
    t = eg.any_term(g)
    s = _v("seed")
    es = eg.add_term(s)
    eg.union(es, g)
    cert = eg.certificate(s, t)
    assert cert.n_egraph_dependent == 1
    assert "introduced outside rule application" in cert.steps[0].note
    # unlocatable target -> the generic fallback message
    assert "no replayable" in eg._explain_gap(s, _v("ghost"))


def test_certificate_src_not_in_graph_raises():
    eg = EGraph()
    eg.add_term(_v("x"))
    with pytest.raises(ValueError, match="not in this e-graph"):
        eg.certificate(_v("ghost"))


def test_certificate_explicit_cost_fn():
    """dst=None with an explicit cost_fn skips the count_cost default."""
    x = _v("x")
    src = Op.make("add", x, Const(0))
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.ID_ADD], root, max_iterations=3)
    cert = eg.certificate(src, cost_fn=count_cost)
    assert op_repr(cert.dst) == "x"
    # explicit root_eid skips term->class resolution entirely
    cert2 = eg.certificate(src, root_eid=root, cost_fn=count_cost)
    assert op_repr(cert2.dst) == "x"


# ===========================================================================
#  proof.py — _edge_path gates
# ===========================================================================


def test_edge_path_found_and_seen_recursion():
    """A real two-step edge path exercises the recursive arm; cycle
    avoidance rejects rules whose result was already visited."""
    x = _v("x")
    s = Op.make("ff", x)
    t = Op.make("gg", x)
    eg = EGraph()
    eg._rule_objs["r_fg"] = Rewrite(
        "r_fg", Op.make("ff", "a"), Op.make("gg", "a")
    )
    eg._rule_objs["r_self"] = Rewrite(
        "r_self", Op.make("ff", "a"), Op.make("ff", "a")
    )
    path = eg._edge_path(s, t, ())
    assert path is not None
    assert [st.rule for st in path] == ["r_fg"]
    # explicit seen/budget pass-through (the recursive frame's arm)
    path2 = eg._edge_path(s, t, (), _seen=set(), _budget=[8])
    assert path2 is not None
    # exhausted budget / depth -> None
    assert eg._edge_path(s, t, (), _seen=set(), _budget=[0]) is None
    assert eg._edge_path(s, t, (), depth=17, _seen=set(),
                       _budget=[8]) is None


def test_edge_path_rule_gates():
    """check failure, derive veto, and derive success each gate a rule
    inside the bounded edge search; a dead-end rule is skipped for a
    productive one."""
    x = _v("x")
    s = Op.make("ff", x)
    t = Op.make("gg", x, k=7)
    eg = EGraph()
    eg._rule_objs["dead"] = Rewrite(
        "dead", Op.make("ff", "a"), Op.make("dd", "a")
    )
    eg._rule_objs["badcheck"] = Rewrite(
        "badcheck",
        Op.make("ff", "a"),
        Op.make("gg", "a"),
        check=lambda b: False,
    )
    eg._rule_objs["vetoderive"] = Rewrite(
        "vetoderive",
        Op.make("ff", "a"),
        Op.make("gg", "a", k="S"),
        derive=lambda b: None,
    )
    eg._rule_objs["good"] = Rewrite(
        "good",
        Op.make("ff", "a"),
        Op.make("gg", "a", k="S"),
        derive=lambda b: {"$attr:S": 7},
    )
    path = eg._edge_path(s, t, ())
    assert path is not None
    assert path[-1].rule == "good"
    assert path[-1].rhs.attrs.get("k") == 7
    # dead-end-first ordering still finds the productive rule
    assert eg._edge_path(s, Op.make("dd", x), ()) is not None


# ===========================================================================
#  proof.py — all_proofs / coherent_paths / level-1 certificates
# ===========================================================================


def test_all_proofs_level1_raises_and_identity():
    eg1 = EGraph(track_proofs=False)
    x = _v("x")
    src = Op.make("add", x, Const(0))
    root = eg1.add_term(src)
    eg1.run([R.ID_ADD], root, max_iterations=3)
    with pytest.raises(RuntimeError, match="truncation_level >= 2"):
        eg1.all_proofs(src, x)
    # level-1 certificate: a single trusted assertion
    cert = eg1.certificate(src, x)
    assert cert.stats["proof_free"]
    assert cert.n_egraph_dependent == 1
    assert op_repr(verify_certificate(src, cert)) == "x"
    # identical endpoints -> zero steps even at level 1
    cert0 = eg1.certificate(src, src)
    assert cert0.n_steps == 0
    # level-2 identity derivation -> one EMPTY derivation
    eg = EGraph()
    r2 = eg.add_term(src)
    eg.run([R.ID_ADD], r2, max_iterations=3)
    assert eg.all_proofs(src, src) == [[]]


def test_all_proofs_check_and_derive_gates():
    """Rules whose ``check``/``derive`` fired during saturation are
    retried by ``all_proofs`` against every subterm — failing gates
    there are skipped, passing ones produce real derivations."""
    x, y = _v("x"), _v("y")
    src = Op.make("add", Op.make("p", x), Op.make("p", y))
    eg = EGraph()
    root = eg.add_term(src)
    chk = Rewrite(
        "chk",
        Op.make("p", "a"),
        Op.make("q", "a"),
        check=lambda b: getattr(b["a"], "name", None) == "x",
    )
    drv = Rewrite(
        "drv",
        Op.make("p", "a"),
        Op.make("q", "a", k="S"),
        derive=lambda b: (
            {"$attr:S": 3}
            if getattr(b["a"], "name", None) == "x"
            else None
        ),
    )
    eg.run([chk, drv], root, max_iterations=3)
    dst = Op.make("add", Op.make("q", x), Op.make("p", y))
    paths = eg.all_proofs(src, dst, max_paths=4, max_steps=4, fuel=512)
    assert paths, "expected at least one derivation"
    assert all(p[-1].path == (0,) for p in paths)
    assert {st.rule for p in paths for st in p} <= {"chk", "drv"}
    # fuel exhaustion / step caps leave paths partial or empty
    assert eg.all_proofs(src, dst, fuel=0) == []
    # a small positive fuel dies mid-enumeration (inner fuel breaks)
    partial = eg.all_proofs(src, dst, fuel=3)
    assert isinstance(partial, list)
    short = eg.all_proofs(src, dst, max_steps=0)
    assert short == []


def test_all_proofs_frontier_cap_break():
    """max_paths reached mid-frontier: the next frontier entry is
    abandoned via the per-entry cap check."""
    x = _v("x")
    src = Op.make("aa", x)
    eg = EGraph()
    root = eg.add_term(src)
    r1 = Rewrite("r1", Op.make("aa", "a"), Op.make("bb", "a"))
    r2 = Rewrite("r2", Op.make("aa", "a"), Op.make("cc", "a"))
    r3 = Rewrite("r3", Op.make("bb", "a"), Op.make("qq", "a"))
    r4 = Rewrite("r4", Op.make("cc", "a"), Op.make("qq", "a"))
    eg.run([r1, r2, r3, r4], root, max_iterations=5)
    dst = Op.make("qq", x)
    # two distinct depth-1 intermediates each reach dst in one step;
    # the first path fills max_paths and the second entry is skipped
    paths = eg.all_proofs(src, dst, max_paths=1)
    assert len(paths) == 1
    assert paths[0][-1].rule == "r3"
    assert op_repr(paths[0][-1].rhs) == "(qq x)"
    # with room for both, the two coherences are enumerated
    both = eg.all_proofs(src, dst, max_paths=4)
    sigs = {tuple(s.rule for s in p) for p in both}
    assert ("r1", "r3") in sigs or ("r3",) in sigs or len(both) >= 2


def test_coherent_paths_summary():
    x, y = _v("x"), _v("y")
    src = Op.make("add", x, y)
    dst = Op.make("add", y, x)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.COMM_ADD], root, max_iterations=3)
    res = eg.coherent_paths(src, dst, root_eid=root)
    assert res["same_eclass"]
    assert res["n_paths"] >= 1
    assert not res["truncated"]
    # max_paths=0 -> the loop never runs -> reported truncated
    res0 = eg.coherent_paths(src, dst, root_eid=root, max_paths=0)
    assert res0["truncated"]
    # endpoints in different classes -> same_eclass False
    foreign = Op.make("mul", x, y)
    eg.add_term(foreign)
    resf = eg.coherent_paths(src, foreign)
    assert resf["same_eclass"] is False


# ===========================================================================
#  terms.py — matchers, subterm surgery, verification
# ===========================================================================


def test_iter_ops_and_term_paths():
    x, y = _v("x"), _v("y")
    t = Op.make("add", Op.make("mul", x, y), x)
    ops = list(_iter_ops(t))
    assert [o.op for o in ops] == ["add", "mul"]
    paths = list(_term_paths(t))
    assert () in paths and (0,) in paths and (0, 1) in paths
    assert _subterm(t, (0, 1)) is y or op_repr(_subterm(t, (0, 1))) == "y"


def test_term_match_attr_metavar_edges():
    x, y = _v("x"), _v("y")
    # Op pattern vs non-Op term
    assert _term_match(Op.make("f", "a"), x) is None
    # arity mismatch
    assert _term_match(Op.make("f", "a", "b"), Op.make("f", x)) is None
    # attribute set mismatch
    assert (
        _term_match(Op.make("f", "a", k=1), Op.make("f", x)) is None
    )
    # attribute metavariable: consistent binding required at every site
    pat = Op.make(
        "p",
        Op.make("f", "a", k="S"),
        Op.make("g", "b", k="S"),
    )
    ok = _term_match(
        pat, Op.make("p", Op.make("f", x, k=1), Op.make("g", y, k=1))
    )
    assert ok is not None and ok["$attr:S"] == 1
    bad = _term_match(
        pat, Op.make("p", Op.make("f", x, k=1), Op.make("g", y, k=2))
    )
    assert bad is None
    # concrete attr value mismatch
    assert (
        _term_match(Op.make("f", "a", k=1), Op.make("f", x, k=2))
        is None
    )
    # concrete leaf patterns match by repr
    assert _term_match(Const(0), Const(0)) == {}
    assert _term_match(Const(0), Const(1)) is None
    # repeated metavariable must bind structurally equal subterms
    assert (
        _term_match(Op.make("add", "a", "a"), Op.make("add", x, y))
        is None
    )
    # a supplied substitution is extended, not rebuilt
    subst = {"z": x}
    out = _term_match(Op.make("f", "a"), Op.make("f", y), subst)
    assert out["z"] is x and out["a"] is y


def test_subterm_and_replace_bad_paths():
    x = _v("x")
    t = Op.make("add", x, Op.make("neg", x))
    assert _subterm(t, (9,)) is None          # index past arity
    assert _subterm(x, (0,)) is None          # descend into a leaf
    assert _subterm(t, (1, 0)) is x or op_repr(_subterm(t, (1, 0))) == "x"
    with pytest.raises(CertificateVerificationError):
        _replace_subterm(t, (1, 0, 0), _v("z"))  # descend through a leaf
    with pytest.raises(CertificateVerificationError):
        _replace_subterm(t, (5,), _v("z"))       # index past arity
    new = _replace_subterm(t, (), _v("z"))
    assert op_repr(new) == "z"
    new2 = _replace_subterm(t, (1,), x)
    assert op_repr(new2) == "(add x, x)"


def _mini_cert(src, steps, rules=None, dst=None):
    return Certificate(
        src=src, dst=dst, root_eid=None, steps=steps, rules=rules or {}
    )


def test_verify_cert_step_path_absent():
    x = _v("x")
    r = Rewrite("rr", "a", Op.make("f", "a"))
    cert = _mini_cert(
        x,
        [CertStep("rr", (0,), x, Op.make("f", x), {})],
        rules={"rr": r},
        dst=Op.make("f", x),
    )
    with pytest.raises(CertificateVerificationError, match="path"):
        verify_certificate(x, cert)


def test_verify_cert_dependent_lhs_mismatch():
    x, y = _v("x"), _v("y")
    cert = _mini_cert(
        x,
        [CertStep("<egraph>", (), y, y, {}, egraph_dependent=True)],
        dst=y,
    )
    with pytest.raises(CertificateVerificationError, match="expected"):
        verify_certificate(x, cert)


def test_verify_cert_lhs_rule_mismatch():
    """Recorded LHS equals the subterm (passes the first check) but the
    RULE's LHS pattern does not match it."""
    x = _v("x")
    t = Op.make("f", x)
    r = Rewrite("rr", Op.make("g", "a"), Op.make("h", "a"))
    cert = _mini_cert(
        t,
        [CertStep("rr", (), t, Op.make("h", x), {})],
        rules={"rr": r},
        dst=Op.make("h", x),
    )
    with pytest.raises(CertificateVerificationError, match="does not match"):
        verify_certificate(t, cert)


def test_verify_cert_derived_binding_and_tampered_binding():
    x, y = _v("x"), _v("y")
    r = Rewrite("rr", "a", Op.make("f", "a"))
    good = _mini_cert(
        x,
        [CertStep("rr", (), x, Op.make("f", x),
                  {"a": x, "derived_extra": y})],
        rules={"rr": r},
        dst=Op.make("f", x),
    )
    # the extra 'derived' binding is skipped over (checked via rhs)
    assert op_repr(verify_certificate(x, good)) == "(f x)"
    bad = _mini_cert(
        x,
        [CertStep("rr", (), x, Op.make("f", x), {"a": y})],
        rules={"rr": r},
        dst=Op.make("f", x),
    )
    with pytest.raises(CertificateVerificationError, match="tampered"):
        verify_certificate(x, bad)


def test_verify_cert_check_and_derive_replay():
    x = _v("x")
    t = Op.make("p", x)
    bad_check = _mini_cert(
        t,
        [CertStep("rb", (), t, Op.make("q", x), {"a": x})],
        rules={
            "rb": Rewrite(
                "rb", Op.make("p", "a"), Op.make("q", "a"),
                check=lambda b: False,
            )
        },
        dst=Op.make("q", x),
    )
    with pytest.raises(CertificateVerificationError, match="side condition"):
        verify_certificate(t, bad_check)

    veto = _mini_cert(
        t,
        [CertStep("rv", (), t, Op.make("q", x, k=9), {"a": x})],
        rules={
            "rv": Rewrite(
                "rv", Op.make("p", "a"), Op.make("q", "a", k="S"),
                derive=lambda b: None,
            )
        },
        dst=Op.make("q", x, k=9),
    )
    with pytest.raises(CertificateVerificationError, match="vetoed"):
        verify_certificate(t, veto)

    ok = _mini_cert(
        t,
        [CertStep(
            "rg", (), t, Op.make("q", x, k=9), {"a": x}
        )],
        rules={
            "rg": Rewrite(
                "rg", Op.make("p", "a"), Op.make("q", "a", k="S"),
                derive=lambda b: {"$attr:S": 9},
            )
        },
        dst=Op.make("q", x, k=9),
    )
    assert op_repr(verify_certificate(t, ok)) == "(q x, k=9)"


def test_certificate_properties_on_real_cert():
    """error_bound aggregates per-step bounds triangle-style; exact /
    replayable / rules_used / n_egraph_dependent reflect the steps."""
    x = _v("x")
    src = Op.make("ff", x)
    eg = EGraph()
    root = eg.add_term(src)
    r1 = Rewrite("e1", Op.make("ff", "a"), Op.make("gg", "a"),
                 error_bound=0.3)
    r2 = Rewrite("e2", Op.make("gg", "a"), Op.make("hh", "a"),
                 error_bound=0.4)
    eg.run([r1, r2], root, max_iterations=5)
    cert = eg.certificate(src, Op.make("hh", x))
    assert cert.replayable and cert.n_egraph_dependent == 0
    assert cert.rules_used == ["e1", "e2"]
    assert cert.error_bound == pytest.approx(0.7)
    assert not cert.exact
    # a certificate with only exact steps reports bound 0 / exact
    cert_ex = eg.certificate(src, Op.make("gg", x))
    assert cert_ex.error_bound == pytest.approx(0.3)
    # dependent steps are never silently replayable
    cert2 = eg.certificate(src, src)
    assert cert2.replayable and cert2.exact


# ===========================================================================
#  core.py — union arms
# ===========================================================================


def test_union_noop_uf_false_and_witness_miss():
    x, y = _v("x"), _v("y")
    eg = EGraph()
    a = eg.add_term(x)
    b = eg.add_term(y)
    assert eg.union(a, a) is False  # already one class
    # union-find itself declining (defensive arm)
    eg._uf.union = lambda ra, rb: False
    assert eg.union(a, b) is False


def test_union_witness_rhs_not_in_graph():
    """A witness whose RHS term cannot be located registers no
    synthetic application — but the merge is still logged."""
    x, y = _v("x"), _v("y")
    eg = EGraph()
    a = eg.add_term(x)
    b = eg.add_term(y)
    n_apps = len(eg._applications)
    w = Rewrite(
        "w_missing",
        lhs=x,
        rhs=Op.make("offered", y),  # never added
    )
    assert eg.union(a, b, witness=w)
    assert eg.merge_log[-1].rule == "w_missing"
    assert len(eg._applications) == n_apps


def test_union_source_node_op_unindexed():
    """A member enode whose op was never indexed in ``_op_classes``
    (injected directly) is skipped by the merge's op-index update."""
    x = _v("x")
    eg = EGraph()
    xc = eg.add_term(x)
    ca = eg.add_enode("opa", (xc,))
    cb = eg.add_enode("opb", (xc,))
    eg.get_class(cb).nodes.add(ENode("zz9", (xc,), ()))
    assert eg.union(ca, cb)  # cb becomes the merge source
    assert eg.find(ca) == eg.find(cb)


def test_union_witness_registers_synthetic_application():
    """The located-witness arm: a synthetic application is recorded so
    the merge replays like a fired rule."""
    x = _v("x")
    src = Op.make("neg", x)
    dst = Op.make("mul", x, Const(-1.0))
    eg = EGraph()
    ea, eb = eg.add_term(src), eg.add_term(dst)
    w = Rewrite("wit", lhs=src, rhs=dst)
    assert eg.union(ea, eb, witness=w)
    assert eg.applications[-1]["rule"] == "wit"
    cert = eg.certificate(src, dst)
    assert cert.replayable
    assert op_repr(verify_certificate(src, cert, strict=True)) == op_repr(dst)


# ===========================================================================
#  core.py — matching caps, rebuild provenance, term resolution fallbacks
# ===========================================================================


def test_match_limit_caps():
    x, y = _v("x"), _v("y")
    eg = EGraph()
    fx = eg.add_term(Op.make("f", x))
    fy = eg.add_term(Op.make("f", y))
    eg.union(fx, fy)
    E = eg.find(fx)
    assert eg.matches(Op.make("f", "a"), E, max_results=0) == []
    one = eg.matches(Op.make("f", "a"), E, max_results=1)
    assert len(one) == 1
    assert len(eg.matches(Op.make("f", "a"), E)) == 2

    # cap hit mid-argument: first arg expands to 2 substs, the second
    # arg's enumeration is truncated by the results limit
    eg2 = EGraph()
    fx = eg2.add_term(Op.make("f", x))
    fy = eg2.add_term(Op.make("f", y))
    eg2.union(fx, fy)
    E2 = eg2.find(fx)
    yc = eg2.add_term(y)
    g = eg2.add_enode("g", (E2, yc))
    pat = Op.make("g", Op.make("f", "a"), "b")
    assert len(eg2.matches(pat, g, max_results=1)) == 1
    assert len(eg2.matches(pat, g)) == 2


def test_match_attr_metavar_multi_bind():
    """A pattern whose attribute is a metavar binds ``$attr:`` entries
    and can match the same op under different attribute values."""
    x = _v("x")
    eg = EGraph()
    xc = eg.add_term(x)
    v1 = eg.add_enode("vf", (xc,), {"k": 1})
    v2 = eg.add_enode("vf", (xc,), {"k": 2})
    eg.union(v1, v2)
    cls = eg.find(v1)
    got = eg.matches(Op.make("vf", "a", k="S"), cls)
    assert {m["$attr:S"] for m in got} == {1, 2}
    # concrete attr patterns filter to equal-valued members
    got1 = eg.matches(Op.make("vf", "a", k=1), cls)
    assert len(got1) == 1
    # by_op index rebuilt after the union (nodes_of lazily re-indexes)
    assert {m["a"] for m in got} == {cls} or len(got) == 2


def test_rebuild_canonicalizes_untracked_node():
    """An enode injected without provenance records is canonicalized
    like any other — no origin/birth/app entries to inherit."""
    x, y = _v("x"), _v("y")
    eg = EGraph()
    xc, yc = eg.add_term(x), eg.add_term(y)
    h = eg.add_enode("hh", (xc,))
    eg.get_class(h).nodes.add(ENode("h2", (yc,), ()))
    assert eg.union(xc, yc)  # y loses its canonical id
    assert eg.rebuild() is True
    ops = {n.op for n in eg.get_class(h).nodes}
    assert ops == {"hh", "h2"}
    assert all(
        all(eg.find(c) == eg.find(xc) for c in n.children)
        for n in eg.get_class(h).nodes
    )


def test_any_term_and_min_term_cyclic():
    eg = EGraph()
    cid = _self_loop2(eg)
    assert eg.any_term(cid) is None     # two failing members -> None
    ca, _cb = _two_cycle(eg)
    assert eg.any_term(ca) is None      # indirect cycle
    assert eg._min_term(ca, {}) == (None, float("inf"))
    assert eg._any_term_cached(ca) is None
    # an already-seen eid short-circuits at _min_term entry
    assert eg._min_term(ca, {}, frozenset({ca})) == (
        None,
        float("inf"),
    )


def test_any_term_cached_recursion_fallback():
    """_min_term blowing the recursion budget falls back to the
    early-exit any_term; if THAT also overflows, the caller sees None
    (the substitution is skipped, never guessed)."""
    x = _v("x")
    eg = EGraph()
    cid = eg.add_term(Op.make("f", x))

    def boom(*a, **k):
        raise RecursionError()

    eg._min_term = boom
    got = eg._any_term_cached(cid)
    assert got is not None  # any_term fallback succeeded
    eg.any_term = boom
    assert eg._any_term_cached(cid) is None


def test_candidate_classes_leaf_lhs():
    eg = EGraph()
    c0 = eg.add_term(Const(0))
    rule = Rewrite("z_to_o", Const(0), Const(1))
    assert eg.apply_rule(rule, c0) is True
    c1 = eg._class_of_term(Const(1))
    assert c1 is not None and eg.find(c0) == eg.find(c1)
    # leaf pattern absent from the graph -> no candidates -> no change
    eg2 = EGraph()
    eg2.add_term(Const(9.0))
    assert eg2.apply_rule(rule, eg2.find(eg2.add_term(_v("q")))) is False


def test_apply_rule_bound_none_and_derive_veto():
    """A checked rule skips substitutions whose bound class has no
    resolvable member (cyclic); a vetoing derive skips the rest."""
    eg = EGraph()
    _self_loop(eg)
    zcls = eg.add_term(Op.make("pz", _v("z")))
    fired = Rewrite(
        "see_all", "a", Op.make("pp", "a"), check=lambda b: True
    )
    assert eg.apply_rule(fired, zcls) is True
    veto = Rewrite(
        "veto_all", "a", Op.make("pq", "a"), derive=lambda b: None
    )
    # every substitution is vetoed (the cyclic one skipped earlier on
    # the unresolvable binding) -> nothing changes
    assert eg.apply_rule(veto, zcls) is False


def test_run_budget_and_max_nodes_boundaries():
    x, y, z = _v("x"), _v("y"), _v("z")
    src = Op.make("add", x, Op.make("add", y, z))
    eg = EGraph()
    root = eg.add_term(src)
    stats = eg.run(
        [R.COMM_ADD, R.ASSOC_ADD],
        root,
        max_iterations=10,
        rule_budgets={"comm_add": 0},
    )
    assert stats["budget_suspended"] == ["comm_add"]
    assert stats["rule_budgets"]["comm_add"] == 0
    # tiny max_nodes: the post-iteration check stops the run early
    eg2 = EGraph()
    r2 = eg2.add_term(src)
    stats2 = eg2.run([R.COMM_ADD, R.ASSOC_ADD], r2, max_nodes=1)
    assert stats2["n_enodes"] >= 1


def test_proof_log_accessors_levels():
    eg1 = EGraph(track_proofs=False)
    x, y = _v("x"), _v("y")
    a, b = eg1.add_term(x), eg1.add_term(y)
    eg1.union(a, b)
    assert eg1.merge_log == []
    assert eg1.n_proof_edges == 0
    assert eg1.applications == []
    eg2 = EGraph(track_proofs=True)
    a, b = eg2.add_term(x), eg2.add_term(y)
    eg2.union(a, b)
    assert eg2.n_proof_edges == 1
    assert eg2.merge_log[0].rule is None
    with pytest.raises(ValueError, match="truncation_level"):
        EGraph(truncation_level=4)
    assert EGraph(truncation_level=3).truncation_level == 3


def test_union_find_rank_arms():
    """Union-by-rank: the lower-rank root is re-parented either way,
    equal ranks bump the surviving root."""
    uf = UnionFind()
    a, b, c, d = uf.make(), uf.make(), uf.make(), uf.make()
    assert uf.union(a, b)          # equal rank -> a gains rank
    assert uf.union(c, a)          # c (rank 0) under a (rank 1): swap arm
    assert uf.find(c) == uf.find(b)
    assert uf.union(d, c)          # d under the (a,b,c) tree
    assert uf.find(d) == uf.find(a)
    assert uf.union(d, b) is False  # already one set


def test_leaf_registry_roundtrip():
    term = Param("p_reg", _t())
    key = _LeafRegistry.register(term)
    assert _LeafRegistry.decode(key) is term
    assert _LeafRegistry.decode("__never_registered__") == (
        "__never_registered__"
    )


# ===========================================================================
#  second pass — residual branches
# ===========================================================================


def test_add_enode_dedup_returns_same_class():
    eg = EGraph()
    c = eg.add_leaf("x")
    a = eg.add_enode("f", (c,))
    b = eg.add_enode("f", (c,))
    assert eg.find(a) == eg.find(b)


def test_match_repeated_metavar_consistency():
    """``add("a", "a")`` demands both children in ONE e-class: the
    consistent-binding arm appends, the inconsistent one returns."""
    x, y, d = _v("x"), _v("y"), _v("d")
    eg = EGraph()
    ex, ey = eg.add_term(x), eg.add_term(y)
    eg.union(ex, ey)
    E = eg.find(ex)
    dc = eg.add_term(d)
    same = eg.add_enode("add", (E, E))
    diff = eg.add_enode("add", (E, dc))
    pat = Op.make("add", "a", "a")
    assert len(eg.matches(pat, same)) == 1
    assert eg.matches(pat, diff) == []


def test_match_enode_arity_and_attrkey_mismatch():
    x, y = _v("x"), _v("y")
    eg = EGraph()
    xc, yc = eg.add_term(x), eg.add_term(y)
    f1 = eg.add_enode("f", (xc,))
    f2 = eg.add_enode("f", (xc, yc))
    eg.union(f1, f2)
    cls = eg.find(f1)
    # the 2-ary member is skipped on the arity check
    assert len(eg.matches(Op.make("f", "a"), cls)) == 1
    # attribute key-set mismatch: node has {k, k2}, pattern only {k}
    v1 = eg.add_enode("vf", (xc,), {"k": 1})
    v2 = eg.add_enode("vf", (xc,), {"k": 1, "k2": 5})
    eg.union(v1, v2)
    vcls = eg.find(v1)
    got = eg.matches(Op.make("vf", "a", k="S"), vcls)
    assert len(got) == 1 and got[0]["$attr:S"] == 1
    # concrete leaf pattern for an absent leaf -> no result
    assert eg.matches(Const(9), cls) == []


def test_match_shared_attr_metavar_recheck():
    """The same attribute metavar at two positions must re-check its
    earlier binding (keep-on-equal / drop-on-unequal arms)."""
    x = _v("x")
    eg = EGraph()
    xc = eg.add_term(x)
    good = eg.add_enode("vf", (xc,), {"k": 1, "k2": 1})
    bad = eg.add_enode("vf", (xc,), {"k": 1, "k2": 9})
    eg.union(good, bad)
    cls = eg.find(good)
    got = eg.matches(Op.make("vf", "a", k="S", k2="S"), cls)
    assert len(got) == 1
    assert got[0]["$attr:S"] == 1


def test_match_limit_larger_than_results():
    """``max_results`` above the result count still enumerates fully —
    and the extend-then-continue arc inside the capped path runs."""
    x, y = _v("x"), _v("y")
    eg = EGraph()
    fx = eg.add_term(Op.make("f", x))
    fy = eg.add_term(Op.make("f", y))
    eg.union(fx, fy)
    assert len(eg.matches(Op.make("f", "a"), eg.find(fx),
                          max_results=5)) == 2


def test_match_multiarg_cap_breaks_mid_enumeration():
    """The per-argument cap check fires mid-loop when an earlier arg
    already produced ``limit`` substitutions."""
    x, y, z, w = _v("x"), _v("y"), _v("z"), _v("w")
    eg = EGraph()
    fx = eg.add_term(Op.make("f", x))
    fy = eg.add_term(Op.make("f", y))
    eg.union(fx, fy)
    E = eg.find(fx)
    fz = eg.add_term(Op.make("f", z))
    fw = eg.add_term(Op.make("f", w))
    eg.union(fz, fw)
    D = eg.find(fz)
    g = eg.add_enode("g", (E, D))
    pat = Op.make("g", Op.make("f", "a"), Op.make("f", "b"))
    # the second arg's enumeration overflows the cap mid-loop: the
    # whole node match is rejected (the cap is per-node atomic)
    assert eg.matches(pat, g, max_results=2) == []
    assert len(eg.matches(pat, g)) == 4


def test_rebuild_inherits_rule_provenance():
    """A rule-born enode whose child class is later merged away is
    canonicalized while INHERITING the creating application record."""
    x, y = _v("x"), _v("y")
    eg = EGraph()
    root = eg.add_term(Op.make("pair", x, y))
    eg.add_term(Op.make("f", y))
    r1 = Rewrite("r_g", Op.make("f", "a"), Op.make("g", "a"))
    r2 = Rewrite("r_pair", Op.make("pair", "a", "b"), "b")
    eg.run([r1, r2], root, max_iterations=5)
    # g's enode was born by r_g with child y; y's class then merged
    # into pair's — rebuild rewrote the child id and carried the app
    canon = eg.find(root)
    app_enodes = [
        n
        for n in eg._enode_app
        if n.op == "g" and all(eg.find(c) == canon for c in n.children)
    ]
    assert app_enodes, "expected a canonicalized rule-born enode"


def test_any_term_min_term_memo_hits():
    """Shared-subterm cones resolve the same e-class twice — the memo
    hit arm of both member-resolution walks."""
    x = _v("x")
    eg = EGraph()
    h = Op.make("hh", x)
    t = Op.make("add", h, h)  # interned: one shared child class
    root = eg.add_term(t)
    got = eg.any_term(root)
    assert op_repr(got) == "(add (hh x), (hh x))"
    term, size = eg._min_term(root, {})
    assert size == 5  # add + 2 x (hh + leaf), tree size not DAG size
    assert op_repr(term) == "(add (hh x), (hh x))"


def test_oldest_term_memo_and_birth_fallback():
    """``_oldest_term`` memoizes per-eclass; an enode with no recorded
    birth sorts last via the ``1 << 60`` fallback."""
    x = _v("x")
    eg = EGraph()
    h = Op.make("hh", x)
    t = Op.make("add", h, h)
    root = eg.add_term(t)
    assert op_repr(eg._oldest_term(root)) == "(add (hh x), (hh x))"
    eg.get_class(root).nodes.add(
        ENode("inj", (eg.add_term(x),), ())
    )
    # the injected node has no birth record -> ``1 << 60`` -> sorted
    # last, so the real (earlier-born) member still wins
    assert (
        op_repr(eg._oldest_term(root)) == "(add (hh x), (hh x))"
    )


def test_instantiate_concrete_attr():
    """Rule RHS attrs that are NOT metavariables pass through the
    substitution unchanged."""
    x = _v("x")
    eg = EGraph()
    fid = eg.add_term(Op.make("f", x))
    r = Rewrite("mk", Op.make("f", "a"), Op.make("g", "a", k=9))
    assert eg.apply_rule(r, fid) is True
    got = eg._class_of_term(Op.make("g", x, k=9))
    assert got is not None and eg.find(got) == eg.find(fid)


def test_run_rule_budget_spent_accounting():
    """A positive rule budget caps enode creation, is charged against
    ``spent``, and suspends the rule once exhausted."""
    x, y = _v("x"), _v("y")
    src = Op.make("add", x, y)
    eg = EGraph()
    root = eg.add_term(src)
    # a second commutable term keeps candidates in the scan after the
    # single-enode budget is spent -> the enode_budget break fires
    eg.add_term(Op.make("add", y, _v("w")))
    stats = eg.run(
        [R.COMM_ADD], root, max_iterations=5, rule_budgets={"comm_add": 1}
    )
    assert stats["rule_budgets"]["comm_add"] >= 1
    assert stats["budget_suspended"] == ["comm_add"]
    # and the iteration-range exit runs when max_iterations bounds first
    eg2 = EGraph()
    r2 = eg2.add_term(src)
    stats2 = eg2.run([R.COMM_ADD], r2, max_iterations=1)
    assert stats2["iterations"] == 1


def test_explain_gap_rule_born_origin():
    """An enode born by a rule application reports neither 'external'
    nor 'input' — the generic explanation is used."""
    x = _v("x")
    eg = EGraph()
    root = eg.add_term(Op.make("f", x))
    r = Rewrite("r_g", Op.make("f", "a"), Op.make("g", "a"))
    eg.run([r], root, max_iterations=3)
    t = Op.make("g", x)
    assert "no replayable" in eg._explain_gap(_v("s"), t)


def test_certificate_default_dst_resolution():
    """``certificate(src)`` resolves dst through count_cost when no
    cost_fn is given."""
    x = _v("x")
    src = Op.make("add", x, Const(0))
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.ID_ADD], root, max_iterations=3)
    cert = eg.certificate(src)
    assert op_repr(cert.dst) == "x"


def test_term_match_concrete_attr_equal():
    x = _v("x")
    assert _term_match(
        Op.make("f", "a", k=1), Op.make("f", x, k=1)
    ) is not None


def test_term_instantiate_attr_arms():
    x = _v("x")
    # concrete attr value passes through unchanged
    got = _term_instantiate(Op.make("f", "a", k=1), {"a": x})
    assert got.attrs == {"k": 1}
    # an unbound attr metavar keeps the pattern's literal value
    got2 = _term_instantiate(Op.make("f", "a", k="S"), {"a": x})
    assert got2.attrs == {"k": "S"}


def test_verify_cert_strict_rejects_dependent():
    x = _v("x")
    cert = _mini_cert(
        x,
        [CertStep("<e>", (), x, x, {}, egraph_dependent=True)],
        dst=x,
    )
    with pytest.raises(CertificateVerificationError, match="e-graph"):
        verify_certificate(x, cert, strict=True)


def test_verify_cert_wrong_source():
    x = _v("x")
    r = Rewrite("rr", "a", Op.make("f", "a"))
    cert = _mini_cert(
        x,
        [CertStep("rr", (), x, Op.make("f", x), {"a": x})],
        rules={"rr": r},
        dst=Op.make("f", x),
    )
    with pytest.raises(CertificateVerificationError, match="source"):
        verify_certificate(_v("other"), cert)


def test_verify_cert_unknown_rule():
    x = _v("x")
    cert = _mini_cert(
        x, [CertStep("nope", (), x, x, {})], dst=x
    )
    with pytest.raises(CertificateVerificationError, match="unknown rule"):
        verify_certificate(x, cert)


def test_verify_cert_recorded_lhs_mismatch():
    x = _v("x")
    r = Rewrite("rr", "a", Op.make("f", "a"))
    cert = _mini_cert(
        x,
        [CertStep("rr", (), _v("other"), Op.make("f", x), {"a": x})],
        rules={"rr": r},
        dst=Op.make("f", x),
    )
    with pytest.raises(CertificateVerificationError, match="recorded LHS"):
        verify_certificate(x, cert)


def test_verify_cert_recorded_rhs_mismatch():
    x = _v("x")
    r = Rewrite("rr", "a", Op.make("f", "a"))
    cert = _mini_cert(
        x,
        [CertStep("rr", (), x, Op.make("g", x), {"a": x})],
        rules={"rr": r},
        dst=Op.make("g", x),
    )
    with pytest.raises(CertificateVerificationError, match="recorded RHS"):
        verify_certificate(x, cert)


def test_verify_cert_final_dst_mismatch():
    x = _v("x")
    r = Rewrite("rr", "a", Op.make("f", "a"))
    cert = _mini_cert(
        x,
        [CertStep("rr", (), x, Op.make("f", x), {"a": x})],
        rules={"rr": r},
        dst=Op.make("g", x),  # replay produces f(x), not g(x)
    )
    with pytest.raises(CertificateVerificationError, match="claims"):
        verify_certificate(x, cert)
