"""Tests for ``catopt_discovery.proposal`` + ``.shape_proposal``.

``proposal`` is the candidate-law machinery: structural
duplicate/inverse detection via alpha-normal keys, the numeric
fp64 oracle for equalities the verifier cannot prove, the three
generation strategies (near-miss, composite, schema), usefulness
measured as a real cost drop, and the ranking comparison.

``shape_proposal`` is the shape-aware variant: schemas over
metavariable patterns, ``real_matches`` enforcing repeated-metavar
equalities, ``relaxed_matches`` counting the bare shape, and firing
on real (here: synthetic) term cases.
"""

import json

import pytest
import torch
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op, Param, TensorType, Var, op_repr
from catopt_core.laws import ALL_RULES
from catopt_discovery import proposal as pp
from catopt_discovery import shape_proposal as sp
from catopt_discovery import verifier as vf
from catopt_discovery.impact import TermCase

_BY_NAME = {r.name: r for r in ALL_RULES}
_D = 4


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _sch() -> dict:
    return {s.name: s for s in sp.schemas()}


def _case(name: str, term: Op) -> TermCase:
    return TermCase(
        name=name,
        source="test",
        term=term,
        inputs=(),
        feed=(),
        param_vals={},
    )


# ---------------------------------------------------------------------------
#  Candidate / Outcome records + structural relation
# ---------------------------------------------------------------------------


def test_candidate_as_rule_and_outcome_gates():
    x, y = _v("x", _D, _D), _v("y", _D, _D)
    c = pp.Candidate("s", "lbl", _p("add", x, y), _p("add", y, x))
    r = c.as_rule()
    assert isinstance(r, Rewrite) and r.name == "s:lbl"

    new_true = pp.Outcome(candidate=c, relation="new", num_true=True)
    assert new_true.new and new_true.truth
    dup = pp.Outcome(candidate=c, relation="duplicate", num_true=True)
    assert not dup.new
    inv = pp.Outcome(candidate=c, relation="inverse", num_true=True)
    assert not inv.new
    false = pp.Outcome(candidate=c, relation="new", num_true=False)
    assert not false.truth and not false.useful
    paid = pp.Outcome(
        candidate=c,
        relation="new",
        derivable=True,
        base_cost=10.0,
        cand_cost=5.0,
    )
    assert paid.truth and paid.useful
    flat = pp.Outcome(
        candidate=c,
        relation="new",
        num_true=True,
        base_cost=5.0,
        cand_cost=5.0,
    )
    assert flat.truth and not flat.useful


def test_relation_classification_against_library():
    lib = pp._library_keys()
    x, y = _v("x", _D, _D), _v("y", _D, _D)
    z = _v("z", _D, _D)
    assert pp._relation(x, x, lib) == "tautology"
    rule = _BY_NAME["sub_to_add"]
    assert pp._relation(rule.lhs, rule.rhs, lib) == "duplicate"
    assert pp._relation(rule.rhs, rule.lhs, lib) == "inverse"
    lhs = _p("add", _p("mul", x, y), _p("mul", x, z))
    rhs = _p("mul", x, _p("add", y, z))
    assert pp._relation(lhs, rhs, lib) == "new"


# ---------------------------------------------------------------------------
#  Numeric oracle
# ---------------------------------------------------------------------------


def test_leaves_excludes_consts():
    x = _v("x", _D)
    p = Param("W", TensorType((_D,)))
    t = _p("add", _p("mul", x, Const(2)), p)
    assert pp._leaves(t) == {x, p}
    assert pp._leaves(x) == {x}
    assert pp._leaves(Const(1)) == set()


def test_numeric_true_false_and_undecidable():
    x, y = _v("x", _D), _v("y", _D)
    assert pp._numeric_true(_p("mul", x, y), _p("mul", y, x)) is True
    assert pp._numeric_true(_p("add", x, y), _p("mul", x, y)) is False
    # An unknown leaf shape is undecidable — None, not False.
    u = Var("u", TensorType((None, _D)))
    assert pp._numeric_true(_p("add", u, u), _p("mul", u, u)) is None
    # An op the backend cannot evaluate is likewise undecidable.
    assert pp._numeric_true(_p("bogus_op", x), x) is None


def test_eval_backend_evaluates():
    x = _v("x", 3)
    backend = pp._eval_backend()
    out = backend.eval_term(
        _p("add", x, x), {x: torch.ones(3, dtype=torch.float64)}
    )
    assert torch.equal(out, torch.full((3,), 2.0, dtype=torch.float64))


def test_allclose_strict_shape_dtype_promotion():
    a = torch.randn(4)
    assert pp._allclose(a, a.clone(), 1e-6) is True
    assert pp._allclose(a, torch.randn(4), 1e-6) is False
    # A shape mismatch is a false equality — not broadcastable-True.
    assert pp._allclose(torch.ones(4), torch.ones(1, 4), 1e-6) is False
    # Tuples recurse positionally.
    assert pp._allclose((a,), (a.clone(),), 1e-6) is True
    assert pp._allclose((a,), (a.clone(), a.clone()), 1e-6) is False
    # Integer Consts compare fairly against float results.
    assert (
        pp._allclose(torch.tensor([1]), torch.tensor([1.0]), 1e-6)
        is True
    )
    assert pp._allclose(a, "not-a-tensor", 1e-6) is False
    # A tensor whose comparison itself fails is a False, not a
    # crash — meta tensors cannot allclose.
    m = torch.empty(4, device="meta")
    assert pp._allclose(m, m, 1e-6) is False


# ---------------------------------------------------------------------------
#  Usefulness (saturation cost) helpers
# ---------------------------------------------------------------------------


def test_size_and_tree_dist():
    x, y, z = _v("x", _D), _v("y", _D), _v("z", _D)
    assert pp._size(x) == 0
    assert pp._size(_p("add", _p("mul", x, y), z)) == 2
    assert pp._tree_dist(x, x) == 0
    assert pp._tree_dist(x, y) == 1
    assert pp._tree_dist(_p("add", x, y), _p("add", x, z)) == 1
    # Same shape, different op → charge both subtree sizes.
    assert pp._tree_dist(_p("add", x, y), _p("mul", x, y)) == 2
    # Attr differences charge sizes too.
    a = _p("select", x, dim=0, index=0)
    b = _p("select", x, dim=0, index=1)
    assert pp._tree_dist(a, b) == 2


def test_sat_cost_and_rule_budgets():
    x, y = _v("x", _D, _D), _v("y", _D, _D)
    term = _p("sub", x, _p("neg", y))
    with_rules = pp._sat_cost(term, list(ALL_RULES), flops_cost)
    bare = pp._sat_cost(term, [], flops_cost)
    assert with_rules < bare  # sub+neg folds to a single add
    budgets = pp._rule_budgets(
        [_BY_NAME["comm_add"], _BY_NAME["sub_to_add"]]
    )
    # comm_add is EXPANSIVE-tagged; sub_to_add is not.
    assert budgets == {"comm_add": pp._EXPANSIVE_BUDGET}


def test_cost_delta_factoring_pays():
    x, y, z = (_v(n, _D, _D) for n in "xyz")
    lhs = _p("add", _p("mul", x, y), _p("mul", x, z))
    cand = pp.Candidate(
        "t", "factor", lhs, _p("mul", x, _p("add", y, z))
    )
    base, with_rule = pp._cost_delta(cand, [lhs], flops_cost)
    assert with_rule < base < float("inf")


def test_cost_delta_skips_oversized_programs():
    x = _v("x", _D)
    big = x
    for _ in range(12):
        big = _p("add", big, x)
    assert pp._size(big) > pp._MAX_PROGRAM_SIZE
    cand = pp.Candidate("t", "c", x, x)
    base, with_rule = pp._cost_delta(cand, [big], flops_cost)
    assert base == float("inf") and with_rule == float("inf")


# ---------------------------------------------------------------------------
#  Strategy 1 — near-miss mining
# ---------------------------------------------------------------------------


def test_seed_terms_well_formed():
    seeds = pp.seed_terms()
    assert len(seeds) == 16
    assert all(isinstance(s, Op) for s in seeds)


def test_near_miss_candidates_close_pairs():
    near = pp.near_miss_candidates(pp.seed_terms()[:8], max_pairs=30)
    assert near, "expected some near-miss pairs on the seed corpus"
    for c in near:
        assert c.strategy == "near-miss"
        assert isinstance(c.lhs, Op) and isinstance(c.rhs, Op)
        assert pp._tree_dist(c.lhs, c.rhs) <= 1
    labels = [c.label for c in near]
    assert len(labels) == len(set(labels))


def test_class_reps_deterministic_small():
    seeds = pp.seed_terms()[:4]
    a = [
        op_repr(c.lhs)
        for c in pp.near_miss_candidates(seeds, max_pairs=20)
    ]
    b = [
        op_repr(c.lhs)
        for c in pp.near_miss_candidates(seeds, max_pairs=20)
    ]
    assert a == b  # deterministic reps, not hash-order dependent


def test_class_reps_skips_leaves_and_oversized():
    from catopt_core.egraph import EGraph

    x, y = _v("x", _D, _D), _v("y", _D, _D)
    eg = EGraph()
    root = eg.add_term(_p("add", x, y))
    eg.run(
        [_BY_NAME["comm_add"]], root, max_iterations=2, max_nodes=200
    )
    reps = pp._class_reps(eg, max_size=4)
    # Leaf e-classes produce Var members, which are skipped.
    assert all(isinstance(r, Op) for r in reps)
    assert _p("add", x, y) in reps or _p("add", y, x) in reps
    # A size cap drops the only non-leaf rep entirely.
    assert pp._class_reps(eg, max_size=0) == []


def test_near_miss_max_pairs_early_return():
    near = pp.near_miss_candidates(pp.seed_terms()[:8], max_pairs=1)
    assert len(near) == 1


# ---------------------------------------------------------------------------
#  Strategy 2 — composite of existing laws
# ---------------------------------------------------------------------------


def test_det_rep_cycle_and_seen_guards():
    from catopt_core.egraph import EGraph

    x, y, z = (_v(n, _D, _D) for n in "xyz")
    eg = EGraph()
    root = eg.add_term(_p("add", _p("mul", x, y), z))
    eg.run(
        [_BY_NAME["comm_add"], _BY_NAME["comm_mul"]],
        root,
        max_iterations=3,
        max_nodes=500,
    )
    # The merged add class: two non-leaf members, no leaf node.
    add_cls = next(
        eid
        for eid, cls in eg._classes.items()
        if any(n.op == "add" for n in cls.nodes)
        and all(n.op != "leaf" for n in cls.nodes)
    )
    # The lexicographically-smallest member is the deterministic rep.
    assert pp._det_rep(eg, add_cls, {}) == _p("add", _p("mul", x, y), z)
    # An eid already on the recursion stack is a cycle → None.
    assert pp._det_rep(eg, add_cls, {}, frozenset({add_cls})) is None
    # A class whose only candidates descend into `seen` fails to
    # produce a representative → None (best stays None).
    kids = {
        c
        for n in eg._classes[eg.find(add_cls)].nodes
        for c in n.children
    }
    assert pp._det_rep(eg, add_cls, {}, frozenset(kids)) is None
    # A *grandchild* in seen makes the recursive call itself return
    # None — the `t is None` bail, not the `canon in seen` bail:
    # add(mul(x,y), z) recurses into the mul class, whose own
    # children are the seen leaf classes.
    leaf_eids = {
        eid
        for eid, cls in eg._classes.items()
        if any(n.op == "leaf" for n in cls.nodes)
    }
    assert pp._det_rep(eg, add_cls, {}, frozenset(leaf_eids)) is None


def test_composite_candidates_are_derivable():
    uni = [
        _BY_NAME[n]
        for n in ("sub_to_add", "double_neg", "comm_add", "id_add")
    ]
    out = pp.composite_candidates(uni, max_cands=10)
    assert out, "the small universe should compose at least once"
    for c in out:
        assert c.strategy == "composite"
        assert "+" in c.label
        res = vf.verify_law(c.lhs, c.rhs, uni)
        assert res.derivable is True


def test_composite_skips_non_instanced_and_caps():
    # mul_square has no bench/generic instance — it is skipped as a
    # source (and still appears as a r2 when applicable).
    uni = [
        _BY_NAME[n]
        for n in ("mul_square", "sub_to_add", "comm_add", "double_neg")
    ]
    out = pp.composite_candidates(uni, max_cands=1)
    assert 1 <= len(out) <= 2
    assert not any(c.label.startswith("mul_square+") for c in out)


# ---------------------------------------------------------------------------
#  Strategy 3 — schema enumeration + evaluation
# ---------------------------------------------------------------------------


def test_schema_candidates_mix_true_false_dup():
    cands = pp.schema_candidates()
    assert len(cands) == 23
    labels = {c.label for c in cands}
    assert {"mul_factor", "mul_zero", "sub_to_add_dup"} <= labels
    assert any(lb.startswith("FALSE_") for lb in labels)
    assert all(c.strategy == "schema" for c in cands)


def test_evaluate_outcomes_truth_relation_useful():
    by = {c.label: c for c in pp.schema_candidates()}
    outs = pp.evaluate(
        [by["mul_factor"], by["FALSE_square_add"], by["sub_to_add_dup"]]
    )
    o = {x.candidate.label: x for x in outs}
    fac = o["mul_factor"]
    assert fac.num_true is True
    assert fac.relation == "new"
    assert fac.useful is True
    assert fac.cand_cost < fac.base_cost
    lie = o["FALSE_square_add"]
    assert lie.num_true is False
    assert lie.useful is False
    dupe = o["sub_to_add_dup"]
    assert dupe.relation == "duplicate"
    assert dupe.derivable is True


def test_yield_table_aggregates():
    outs = pp.evaluate(pp.schema_candidates()[:4])
    tab = pp.yield_table(outs)
    row = tab["schema"]
    assert row["proposed"] == 4
    assert row["num_true"] == 4
    assert row["new"] == 4
    assert row["useful"] == 3
    assert row["useful_new"] == 3


# ---------------------------------------------------------------------------
#  Strategy 4 — ranking
# ---------------------------------------------------------------------------


def test_average_precision_orderings():
    def mk(i, useful):
        return pp.Outcome(
            candidate=pp.Candidate("s", str(i), Const(i), Const(i)),
            relation="new",
            num_true=True,
            base_cost=10.0,
            cand_cost=0.0 if useful else 20.0,
        )

    outs = {i: mk(i, i < 2) for i in range(4)}
    assert pp._average_precision([0, 1, 2, 3], outs) == 1.0
    rev = pp._average_precision([2, 3, 0, 1], outs)
    assert rev == pytest.approx((1 / 3 + 2 / 4) / 2)


def test_rank_pool_learned_heuristic_random():
    cands = pp.schema_candidates()[:4]
    outs = pp.evaluate(cands)
    idx = {i: o for i, o in enumerate(outs)}
    x, y = _v("x", _D, _D), _v("y", _D, _D)
    seeds = [cands[0].lhs, _p("sub", x, _p("neg", y))]
    r = pp.rank_pool(cands, idx, seeds=seeds)
    assert r["pool"] == 4 and r["useful"] == 3
    assert r["random_ap"] == pytest.approx(0.75)
    # cost(lhs) - cost(rhs) ranks the single non-useful candidate
    # (the expanding distribute) last → perfect AP.
    assert r["heuristic_ap"] == pytest.approx(1.0)
    assert 0.0 <= r["learned_ap"] <= 1.0


def test_rank_pool_no_positive_is_undefined():
    cands = pp.schema_candidates()[:4]
    outs = pp.evaluate(cands)
    flat = {
        i: pp.Outcome(
            candidate=o.candidate,
            relation="new",
            num_true=True,
            base_cost=1.0,
            cand_cost=1.0,
        )
        for i, o in enumerate(outs)
    }
    assert pp.rank_pool(cands, flat, seeds=[]) == {}
    assert pp._rank_on(cands, list(flat.values()), []) == {}
    # With a useful outcome present, _rank_on returns a real result.
    x, y = _v("x", _D, _D), _v("y", _D, _D)
    seeds = [cands[0].lhs, _p("sub", x, _p("neg", y))]
    r = pp._rank_on(cands, outs, seeds)
    assert r["pool"] == 4 and r["useful"] == 3


def test_pool_features_and_heuristic_score():
    c = pp.schema_candidates()[0]
    assert pp._pool_features(c) is not None
    # mul_factor's LHS has strictly more FLOPs than its RHS.
    assert pp._heuristic_score(c) > 0.0
    flat = pp.Candidate(
        "t",
        "c",
        _p("add", _v("x", _D), _v("y", _D)),
        _p("add", _v("x", _D), _v("y", _D)),
    )
    assert pp._heuristic_score(flat) == 0.0


def test_report_and_dump(tmp_path):
    outs = pp.evaluate(pp.schema_candidates()[:4])
    table = pp.yield_table(outs)
    text = pp._fmt_yield(table)
    assert "schema" in text and "useful" in text
    useful = pp._fmt_useful(outs)
    assert "cost" in useful
    empty = pp._fmt_useful([])
    assert "none" in empty
    path = tmp_path / "pp.json"
    pp._dump_json(str(path), outs, table, {"full": {}, "sub": {}})
    payload = json.loads(path.read_text())
    assert len(payload["outcomes"]) == 4
    assert payload["yield"]["schema"]["proposed"] == 4
    assert payload["outcomes"][0]["lhs"].startswith("(add")


# ---------------------------------------------------------------------------
#  shape_proposal — schemas, matches, firing
# ---------------------------------------------------------------------------


def test_schemas_library_and_as_rule():
    sch = sp.schemas()
    assert len(sch) == 21
    names = {s.name for s in sch}
    assert {"factor_left", "select_mul", "mul_zero"} <= names
    assert "select-naturality" in {s.family for s in sch}
    assert all(s.note for s in sch)
    for s in sch:
        r = s.as_rule()
        assert isinstance(r, Rewrite) and r.name == s.name


def test_relax_renames_only_repeated_metavars():
    fl = _sch()["factor_left"]
    relaxed = sp._relax(fl.lhs)
    assert "A__2" in op_repr(relaxed)
    # Every metavar appears once in the RHS — nothing renamed.
    once = sp._relax(fl.rhs)
    assert "__" not in op_repr(once)


def test_real_and_relaxed_matches():
    x, y, z = (_v(n, _D, _D) for n in "xyz")
    terms = [
        _p("add", _p("mul", x, y), _p("mul", x, z)),  # equality holds
        _p("add", _p("mul", x, y), _p("mul", y, z)),  # shape only
        x,  # leaf: skipped
    ]
    fl = _sch()["factor_left"]
    assert sp.real_matches(terms, fl) == [terms[0]]
    assert sp.relaxed_matches(terms, fl) == 2
    mz = _sch()["mul_zero"]
    assert sp.real_matches(terms, mz) == []
    assert sp.relaxed_matches(terms, mz) == 0


def test_instantiate_rhs_of_match():
    x, y, z = (_v(n, _D, _D) for n in "xyz")
    fl = _sch()["factor_left"]
    sub = _p("add", _p("mul", x, y), _p("mul", x, z))
    assert sp._instantiate(fl, sub) == _p("mul", x, _p("add", y, z))
    assert sp._instantiate(fl, _p("add", x, y)) is None


def test_cost_delta_on_matched_term():
    x, y, z = (_v(n, _D, _D) for n in "xyz")
    fl = _sch()["factor_left"]
    sub = _p("add", _p("mul", x, y), _p("mul", x, z))
    base, cand = sp._cost_delta(fl, [sub], flops_cost)
    assert cand < base < float("inf")
    # The factored form is genuinely cheaper under dag_cost.
    assert dag_cost(sub, flops_cost) == base


def test_fires_on_models_counts_and_pays():
    x, y, z = (_v(n, _D, _D) for n in "xyz")
    sch = _sch()
    term = _p("add", _p("mul", x, y), _p("mul", x, z))
    case = _case("fact", term)
    fires, cases, changed, paid = sp._fires_on_models(
        sch["factor_left"], [case], flops_cost
    )
    assert fires == 1
    assert cases == ("fact",)
    assert changed == 1 and paid == 1
    # A schema whose LHS never appears does nothing.
    out = sp._fires_on_models(sch["mul_zero"], [case], flops_cost)
    assert out == (0, (), 0, 0)


def test_schema_outcome_properties():
    s = _sch()["factor_left"]
    o = sp.SchemaOutcome(
        schema=s,
        relation="new",
        num_true=True,
        base_cost=10.0,
        cand_cost=4.0,
    )
    assert o.new and o.truth and o.useful
    dupe = sp.SchemaOutcome(schema=s, relation="duplicate")
    assert not dupe.new and not dupe.truth and not dupe.useful
    deriv = sp.SchemaOutcome(schema=s, relation="new", derivable=True)
    assert deriv.truth


def test_fires_on_models_unchanged_best_not_changed():
    # A rule that fires but only *adds* a worse member: the extract
    # keeps the original term, so nothing is counted changed/paid.
    x, y = _v("x", _D, _D), _v("y", _D, _D)
    grow = sp.Schema(
        "grow",
        _p("mul", "A", "B"),
        _p("add", _p("mul", "A", "B"), _p("mul", "A", "B")),
    )
    case = _case("m", _p("mul", x, y))
    fires, cases, changed, paid = sp._fires_on_models(
        grow, [case], flops_cost
    )
    assert fires >= 1 and cases == ("m",)
    assert changed == 0 and paid == 0


def test_fires_on_models_changed_but_not_paid():
    # Re-association fires and the extract *changes*, but the two
    # bracketings cost the same — changed without paid.
    x = _v("x", _D, _D)
    assoc = sp.Schema(
        "assoc",
        _p("add", _p("add", "A", "B"), "C"),
        _p("add", "A", _p("add", "B", "C")),
    )
    case = _case("m", _p("add", _p("add", x, x), x))
    fires, cases, changed, paid = sp._fires_on_models(
        assoc, [case], flops_cost
    )
    assert fires >= 1 and cases == ("m",)
    assert changed == 1 and paid == 0


def test_sink_is_cached():
    a = sp._sink()
    assert sp._sink() is a


def test_reach_rows_for_firing_schemas():
    x, y, z = (_v(n, _D, _D) for n in "xyz")
    sch = _sch()["factor_left"]
    term = _p("add", _p("mul", x, y), _p("mul", x, z))
    case = _case("fact", term)
    o = sp.SchemaOutcome(schema=sch, model_fires=1)
    rows = sp._reach([o], [case], flops_cost)
    assert len(rows) == 1
    row = rows[0]
    assert row["schema"] == "factor_left" and row["model"] == "fact"
    assert row["fires"] >= 1 and row["changed"] is True
    assert row["add_cost"] < row["base_cost"]
    # A schema that never fires on a model contributes no row —
    # both via the outcome-level guard and the per-model one.
    quiet = sp.SchemaOutcome(schema=_sch()["mul_zero"])
    assert sp._reach([quiet], [case], flops_cost) == []
    nofire = _case("plain", _p("mul", x, y))
    assert sp._reach([o], [nofire], flops_cost) == []


def test_baseline_counts_law_fires():
    x, y, z = (_v(n, _D, _D) for n in "xyz")
    case = _case("fact", _p("add", _p("mul", x, y), _p("mul", x, z)))
    base = sp._baseline([case])
    assert len(base["laws"]) == 11
    assert base["fires"].get("cand_mul_factor", 0) >= 1


def test_render_tables_and_verdict(tmp_path, capsys):
    s = _sch()["factor_left"]
    good = sp.SchemaOutcome(
        schema=s,
        relaxed=2,
        matches=1,
        example="add((mul x, y), (mul x, z))",
        num_true=True,
        relation="new",
        base_cost=48.0,
        cand_cost=32.0,
        model_fires=1,
        fire_cases=("fact",),
        fire_changed=1,
        fire_paid=1,
    )
    bad = sp.SchemaOutcome(
        schema=_sch()["mul_zero"],
        model_fires=2,
        fire_cases=("m1", "m2", "m3", "m4"),
        fire_paid=2,
        num_true=False,
    )
    outs = [good, bad]
    mt = sp._match_table(outs)
    assert "factor_left" in mt and "yes" in mt
    ft = sp._fire_table(outs)
    assert "fact" in ft
    assert "…" in ft  # >3 fire cases truncate
    assert "no schema" in sp._reach_table([])
    rt = sp._reach_table(
        [
            {
                "schema": "s",
                "model": "m",
                "base_cost": 1.0,
                "add_cost": 0.5,
                "fires": 2,
                "cert": True,
            }
        ]
    )
    assert "s" in rt and "1->0.5" in rt
    result = {
        "outcomes": outs,
        "n_bench": 1,
        "n_models": 1,
        "baseline": {"laws": [], "fires": {}},
        "reach": [],
    }
    sp._verdict(result)
    text = capsys.readouterr().out
    assert "factor_left" in text
    assert "NOT true" in text  # the cost-only proposer warning
    # An all-empty outcome set prints the zero verdicts.
    sp._verdict(
        {
            "outcomes": [],
            "n_bench": 0,
            "n_models": 0,
            "baseline": {"laws": [], "fires": {}},
            "reach": [],
        }
    )
    zeros = capsys.readouterr().out
    assert "schemas: 0" in zeros
    path = tmp_path / "shape.json"
    sp._dump_json(str(path), result)
    payload = json.loads(path.read_text())
    assert payload["outcomes"][0]["schema"] == "factor_left"
    assert payload["outcomes"][0]["useful"] is True
