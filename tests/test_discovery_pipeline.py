"""Tests for ``catopt_discovery.pipeline`` — the end-to-end law loop.

The pipeline composes census -> propose -> verify -> measure -> rank.
These tests exercise it two ways:

* **unit** — the proposer generators (``_census_naturality`` /
  ``_census_mixed_naturality`` / ``_pattern_recognition`` /
  ``_to_pattern`` …), the ``Evidence`` verdict ladder, ``measure``
  on hand-built terms, ``rank`` ordering and ``_evidence_from_row``;
* **integration** — ``run_pipeline`` and ``main`` against a *tiny*
  synthetic corpus (two-to-three ``TermCase``s monkeypatched in for
  the bench/model/intake loaders), keeping the whole loop under a
  second while still running real e-graph saturation, real oracles
  and real torch verification.  The held-out rediscovery paths are
  exercised with ``select_mul`` and ``softmax_fold``, both of which
  the tiny corpus genuinely rediscovers.

The typedness audit (``_bad_subterms`` / ``_typed_probe`` /
``_reach_row`` ill-typed suppression) is pinned by
``test_law_pipeline_typed.py``; this file covers the rest.
"""

import json

import pytest
import torch
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import evidence as ev_store
from catopt_discovery import intake as li
from catopt_discovery import pipeline as pl
from catopt_discovery import proposal as lp
from catopt_discovery.census import (
    CorpusTerm,
    op_tuple_census,
    shape_census,
)
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink

# ---------------------------------------------------------------------------
#  Tiny corpus — two/three well-typed terms standing in for the real one
# ---------------------------------------------------------------------------


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _feed(*vars_: Var) -> tuple:
    return tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in vars_
    )


def _case(name: str, term: Op, *inputs: Var) -> TermCase:
    return TermCase(
        source="test",
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=_feed(*inputs),
        param_vals={},
    )


def _tiny_cases(with_softmax: bool = False) -> list[TermCase]:
    """The stand-in corpus: a factoring site and a select-pair site."""
    torch.manual_seed(0)
    x, y, z = _v("x", 4, 4), _v("y", 4, 4), _v("z", 4, 4)
    u, w = _v("u", 4, 4), _v("w", 4, 4)
    t_factor = _p("add", _p("mul", x, y), _p("mul", x, z))
    t_selmul = _p(
        "mul",
        _p("select", u, dim=0, index=1),
        _p("select", w, dim=0, index=1),
    )
    cases = [
        _case("factor", t_factor, x, y, z),
        _case("selmul", t_selmul, u, w),
    ]
    if with_softmax:
        s = _v("s", 4, 4)
        e = _p("exp", s)
        cases.append(
            _case(
                "softmax",
                _p("div", e, _p("sum", e, dim=(-1,), keepdim=True)),
                s,
            )
        )
    return cases


def _tiny_census(cases: list[TermCase]) -> dict:
    """The real census computation, over the tiny corpus only."""
    cts = [CorpusTerm("test", c.name, c.term) for c in cases]
    op_counts, op_terms = op_tuple_census(cts)
    sh_counts, _ = shape_census(cts)
    return {
        "n_terms": len(cts),
        "n_bench": 0,
        "n_models": len(cts),
        "n_intake": 0,
        "n_op_nodes": sum(op_counts.values()),
        "n_shapes": len(sh_counts),
        "op_tuples": [
            {
                "op": k[0],
                "children": list(k[1]),
                "count": n,
                "terms": len(op_terms[k]),
            }
            for k, n in op_counts.most_common()
        ],
        "shapes": [],
        "sharing": [],
    }


def _patch_tiny(monkeypatch) -> list[TermCase]:
    """Patch every corpus loader to the tiny synthetic corpus."""
    cases = _tiny_cases(with_softmax=True)
    census = _tiny_census(cases)
    monkeypatch.setattr(pl, "run_census", lambda top=400: census)
    monkeypatch.setattr(pl, "_bench_cases", lambda: ([], []))
    monkeypatch.setattr(pl, "model_cases", lambda: (cases, []))
    monkeypatch.setattr(li, "load_cases", lambda *a, **k: [])
    monkeypatch.setattr(li, "probe_cases", lambda *a, **k: [])
    return cases


@pytest.fixture
def lib_keys() -> list:
    return [lp._key(r.lhs, r.rhs) for r in ALL_RULES]


@pytest.fixture
def sink_cost():
    sink = _sink()
    return sink, _cost_fn(sink)


# ---------------------------------------------------------------------------
#  Pure helpers
# ---------------------------------------------------------------------------


def test_op_of_classifies_nodes_and_leaves():
    x = _v("x", 4)
    assert pl._op_of(_p("mul", x, x)) == "mul"
    assert pl._op_of(Const(2)) == "const"
    assert pl._op_of(x) == "·"
    assert pl._op_of("metavar") == "·"


def test_iter_subterms_dedupes_shared_nodes():
    x = _v("x", 4)
    shared = _p("mul", x, x)
    term = _p("add", shared, shared)
    subs = pl._iter_subterms(term)
    assert len(subs) == len({id(s) for s in subs})
    assert term in subs and shared in subs and x in subs


def test_lhs_tuple_keys_like_the_census():
    x, y = _v("x", 4), _v("y", 4)
    term = _p("add", _p("mul", x, y), _p("mul", x, x))
    assert pl._lhs_tuple(term) == ("add", ("mul", "mul"))
    assert pl._lhs_tuple(x) is None
    assert pl._lhs_tuple(Const(1)) is None


def test_to_pattern_abstracts_leaves_consistently():
    x, y = _v("x", 4), _v("y", 4)
    mv: dict[str, str] = {}
    pat = pl._to_pattern(_p("mul", x, _p("add", y, x)), mv)
    # The repeated leaf x is one metavariable; y is another.
    assert pat == _p("mul", "M0", _p("add", "M1", "M0"))
    # Const leaves stay literal.
    assert pl._to_pattern(_p("mul", x, Const(0)), {}) == _p(
        "mul", "M0", Const(0)
    )


def test_view_attrs_collects_only_the_view_op():
    x = _v("x", 4, 4)
    term = _p(
        "mul",
        _p("select", x, dim=0, index=1),
        _p("unsqueeze", x, dim=1),
    )
    assert pl._view_attrs([term], "select") == ["dim", "index"]
    assert pl._view_attrs([term], "slice") == []


def test_view_node_shares_attr_metavars_per_tag():
    a = pl._view_node("select", "U", ["dim", "index"], tag="A")
    b = pl._view_node("select", "U", ["dim", "index"], tag="B")
    assert a.attrs == {"dim": "A_dim", "index": "A_index"}
    assert b.attrs == {"dim": "B_dim", "index": "B_index"}


def test_vocab_sets_hand_and_derived(monkeypatch):
    pointwise, views = pl._vocab_sets("hand")
    assert pointwise == pl._POINTWISE and views == pl._VIEW_OPS

    class _FakeVocab:
        pointwise = ("add",)
        views = ("select",)

    monkeypatch.setattr(
        "catopt_discovery.vocab.derive_vocabulary", lambda: _FakeVocab()
    )
    assert pl._vocab_sets("derived") == (("add",), ("select",))


# ---------------------------------------------------------------------------
#  Census-driven proposers
# ---------------------------------------------------------------------------


def test_census_naturality_same_view_tuple():
    x = _v("x", 4, 4)
    terms = [_p("select", x, dim=0, index=1)]
    out = pl._census_naturality(
        {("mul", ("select", "select")): 3}, terms
    )
    assert [p.name for p in out] == ["census:mul_select"]
    p = out[0]
    # Both select nodes share the attr metavariables — the naturality.
    lhs_sel = p.lhs.args[0]
    assert lhs_sel.attrs == {"dim": "V_dim", "index": "V_index"}
    # Non-view operands never propose.
    assert (
        pl._census_naturality({("add", ("mul", "mul")): 9}, terms) == []
    )


def test_census_mixed_two_view_family():
    terms = [
        _p("select", _v("x", 4, 4), dim=0, index=1),
        _p("slice", _v("y", 4, 4), dim=0, start=0, end=2),
    ]
    out = pl._census_mixed_naturality(
        {("mul", ("select", "slice")): 2}, terms
    )
    names = sorted(p.name for p in out)
    assert names == [
        "mixed:mul_select_slice_id",
        "mixed:mul_select_slice_wl",
        "mixed:mul_select_slice_wr",
    ]
    # The wrap variants keep their own side's attr metavars.
    wl = next(p for p in out if p.name.endswith("_wl"))
    assert wl.rhs.op == "select"
    assert set(wl.rhs.attrs) == {"dim", "index"}


def test_census_mixed_one_view_family_both_sides():
    terms = [_p("select", _v("x", 4, 4), dim=0, index=1)]
    left = pl._census_mixed_naturality(
        {("mul", ("select", "·")): 1}, terms
    )
    right = pl._census_mixed_naturality(
        {("mul", ("·", "select")): 1}, terms
    )
    assert sorted(p.name for p in left) == [
        "mixed:mul_select_l_id",
        "mixed:mul_select_l_w",
    ]
    assert sorted(p.name for p in right) == [
        "mixed:mul_select_r_id",
        "mixed:mul_select_r_w",
    ]
    # view-left vs view-right place the view node accordingly.
    lw = next(p for p in left if p.name.endswith("_w"))
    rw = next(p for p in right if p.name.endswith("_w"))
    assert lw.lhs.args[0].op == "select"
    assert rw.lhs.args[1].op == "select"


def test_reduce_chains_and_pattern_recognition():
    census_op = {
        ("div", ("exp", "sum")): 4,
        ("sum", ("exp",)): 4,
        ("mul", ("·", "·")): 7,
    }
    assert pl._reduce_chains(census_op) == [("div", "exp", "sum")]
    props = pl._pattern_recognition(census_op)
    assert [p.name for p in props] == ["recognize:softmax"]
    assert props[0].check is not None and props[0].derive is not None
    # No reduction-feeding tuple -> no chain -> no proposal.
    assert pl._pattern_recognition({("mul", ("·", "·")): 7}) == []


def test_pattern_recognition_dedup_and_unrecognized():
    # The same chain surfacing from both operand orders is emitted
    # once, and a chain with no registered recognizer stays a census
    # fact — no guessed equality.
    census_op = {
        ("div", ("exp", "sum")): 4,
        ("div", ("sum", "exp")): 4,
        ("sum", ("exp",)): 4,
        ("mul", ("exp", "sum")): 2,
    }
    props = pl._pattern_recognition(census_op)
    assert [p.name for p in props] == ["recognize:softmax"]


def test_check_sum_keepdim_guard():
    assert pl._check_sum_keepdim({"$attr:RK": True, "$attr:RD": (-1,)})
    assert pl._check_sum_keepdim({"$attr:RK": True, "$attr:RD": -1})
    assert not pl._check_sum_keepdim(
        {"$attr:RK": True, "$attr:RD": (0, 1)}
    )
    assert not pl._check_sum_keepdim(
        {"$attr:RK": False, "$attr:RD": (-1,)}
    )
    assert not pl._check_sum_keepdim({"$attr:RD": (-1,)})


def test_derive_softmax_dim_unwraps_tuple():
    assert pl._derive_softmax_dim({"$attr:RD": (-1,)}) == {
        "$attr:SD": -1
    }
    assert pl._derive_softmax_dim({"$attr:RD": 0}) == {"$attr:SD": 0}


def test_has_view_op_scans_both_sides():
    view_p = pl.Proposal(
        "t:v",
        _p("mul", _p("select", "A", dim="D", index="I"), "B"),
        "A",
        family="t",
    )
    plain_p = pl.Proposal(
        "t:p", _p("mul", "A", "B"), _p("mul", "B", "A"), family="t"
    )
    assert pl._has_view_op(view_p)
    assert not pl._has_view_op(plain_p)


def test_search_rules_holds_out_names():
    assert len(pl._search_rules(None)) == len(ALL_RULES)
    out = pl._search_rules("comm_add,select_mul,not_a_rule")
    assert len(out) == len(ALL_RULES) - 2
    assert {r.name for r in out}.isdisjoint({"comm_add", "select_mul"})


def test_subst_key_normalizes_both_record_forms():
    subst = {"A": 3, "$attr:D": 0}
    # The merge-log freezes a binding as sorted ``(key, value)`` items
    # with raw values; both forms key the same multiset.
    as_items = tuple(sorted(subst.items()))
    assert pl._subst_key(subst) == pl._subst_key(as_items)


def test_evals_unknown_dim_is_undecidable():
    assert pl._evals(_v("x", None, 4)) is None
    assert pl._evals(_p("mul", _v("x", 4), _v("y", 4))) is True
    # A leaf whose type cannot give a shape makes the env build fail —
    # reported as undecidable, never as a crash.
    assert pl._evals(_p("neg", Var("x", object()))) is None


def test_bad_subterms_marks_shape_errors_and_bad_dims():
    x = _v("x", 4, 4)
    # One-arg matmul raises inside ``_shape_of`` — flagged, not
    # propagated.
    assert _p("matmul", x) in pl._bad_subterms(_p("matmul", x))
    # A negative reshape dim shape-resolves to a bad tuple.
    bad = _p("reshape", x, shape=(-4,))
    assert bad in pl._bad_subterms(bad)
    # The minted member is ill-typed on eval only (tuple operand read
    # as a tensor): shape-resolves, fails the eval leg.
    u, v = _v("u", 4, 4), _v("v", 4, 1)
    vm = _p("var_mean", u, dim=(-1,), correction=0, keepdim=True)
    term = _p("add", vm, v)
    assert not pl._bad_subterms(term)
    assert pl._evals(term) is False
    assert not pl._term_typed(term)


def test_pick_ill_typed_inherited_badness_is_ambient():
    x, y = _v("x", 3), _v("y", 3, 2)
    bad = _p("mul", x, y)  # broadcast mismatch — does not denote
    # The same bad subterm in the reference is not this rule's mint.
    assert not pl._pick_ill_typed(bad, bad)
    # A fresh ill-typed member is picked out.
    assert pl._pick_ill_typed(bad, _p("mul", _v("a", 3), _v("b", 3)))


# ---------------------------------------------------------------------------
#  _instance_from_match — the check/derive seam over real matches
# ---------------------------------------------------------------------------


def test_instance_from_match_applies_check_and_derive():
    e = _p("exp", "U")
    proposal = pl.Proposal(
        name="t:sm",
        lhs=_p("div", e, _p("sum", e, dim="RD", keepdim="RK")),
        rhs=_p("softmax", "U", dim="SD"),
        family="t",
        check=pl._check_sum_keepdim,
        derive=pl._derive_softmax_dim,
    )
    s = _v("s", 4, 4)
    good = _p(
        "div",
        _p("exp", s),
        _p("sum", _p("exp", s), dim=(-1,), keepdim=True),
    )
    bad = _p(
        "div",
        _p("exp", s),
        _p("sum", _p("exp", s), dim=(-1,), keepdim=False),
    )
    # The vetoed match is skipped; the viable one instantiates.
    inst = pl._instance_from_match(proposal, [bad, good])
    assert inst is not None
    assert inst[0] == good
    assert inst[1] == _p("softmax", s, dim=-1)
    # All matches vetoed -> no instance.
    veto = pl.Proposal(
        "t:v",
        proposal.lhs,
        proposal.rhs,
        "t",
        check=lambda bound: False,
    )
    assert pl._instance_from_match(veto, [good]) is None
    # A derive returning None skips the match exactly like a veto.
    noderive = pl.Proposal(
        "t:d",
        proposal.lhs,
        proposal.rhs,
        "t",
        derive=lambda bound: None,
    )
    assert pl._instance_from_match(noderive, [good]) is None


def test_instance_from_match_skips_nonmatches_and_hook_raises():
    x = _v("x", 4, 4)
    proposal = pl.Proposal(
        "t:sk",
        _p("mul", "A", "B"),
        _p("mul", "B", "A"),
        "t",
        check=lambda bound: True,
    )
    # A subterm the LHS does not match is skipped, never matched.
    assert pl._instance_from_match(proposal, [_p("add", x, x)]) is None

    def _raising_check(bound):
        raise RuntimeError("boom")

    def _raising_derive(bound):
        raise RuntimeError("boom")

    good = _p("mul", x, _v("y", 4, 4))
    raising = pl.Proposal(
        "t:rc",
        proposal.lhs,
        proposal.rhs,
        "t",
        check=_raising_check,
    )
    assert pl._instance_from_match(raising, [good]) is None
    raising_d = pl.Proposal(
        "t:rd",
        proposal.lhs,
        proposal.rhs,
        "t",
        derive=_raising_derive,
    )
    assert pl._instance_from_match(raising_d, [good]) is None


def test_app_typed_refuses_uninstantiable_bindings(sink_cost):
    """An application whose subst cannot instantiate the RHS is
    ill-typed — never silently typed."""
    case = _tiny_cases()[0]
    proposal = _factor_proposal()
    eg = pl.EGraph()
    root = eg.add_term(case.term)
    eg.run([proposal.as_rule()], root, max_iterations=1)
    # A subst missing a metavariable fails to instantiate the RHS.
    real = next(
        a for a in eg.applications if a["rule"] == proposal.name
    )
    partial = {"subst": {"A": real["subst"]["A"]}}
    assert not pl._app_typed(eg, proposal, partial)
    # The full recorded binding instantiates and type-checks.
    assert pl._app_typed(eg, proposal, real)


# ---------------------------------------------------------------------------
#  propose — collection, unification and dedup
# ---------------------------------------------------------------------------


def test_propose_dedup_merges_provenance():
    cases = _tiny_cases(with_softmax=True)
    census_op = {
        (e["op"], tuple(e["children"])): e["count"]
        for e in _tiny_census(cases)["op_tuples"]
    }
    terms = [c.term for c in cases]
    props = pl.propose(census_op, terms, "hand")
    # No alpha-normal key appears twice.
    keys = [lp._key(p.lhs, p.rhs) for p in props]
    assert len(keys) == len(set(keys))
    # The census's mul(select,select) site and the select_mul schema
    # are the same equality — one proposal, merged sources.
    sel = next(p for p in props if p.name == "census:mul_select")
    assert "census-naturality" in sel.sources
    assert "shape-aware" in sel.sources
    # The softmax chain is recognized on the tiny corpus.
    assert any(p.name == "recognize:softmax" for p in props)
    # Grammar candidates keep their concrete instance for the oracles.
    gram = [p for p in props if p.family == "algebraic-grammar"]
    assert gram and all(p.instance is not None for p in gram)


def test_propose_mixed_family_on_asymmetric_tuple():
    x = _v("x", 4, 4)
    terms = [_p("mul", _p("select", x, dim=0, index=1), _v("y", 4))]
    census_op = {("mul", ("select", "·")): 2}
    names = {p.name for p in pl.propose(census_op, terms, "hand")}
    assert "mixed:mul_select_l_w" in names
    assert "mixed:mul_select_l_id" in names


# ---------------------------------------------------------------------------
#  Evidence — the verdict ladder
# ---------------------------------------------------------------------------


def _ev(**kw) -> pl.Evidence:
    ev = pl.Evidence(
        proposal=pl.Proposal("t:x", _p("mul", "A", "B"), "A", "t")
    )
    ev.num_true = True
    ev.relation = "new"
    for k, v in kw.items():
        setattr(ev, k, v)
    return ev


def test_evidence_truth_and_ship_gates():
    assert _ev(derivable=True, num_true=None).truth
    assert _ev(num_true=False).truth is False
    ship = _ev(fires=2, fires_typed=2, paid=1)
    assert ship.shippable and ship.no_ship_reason == ""
    assert not _ev(closure_ratio=3.0).closure_safe


@pytest.mark.parametrize(
    ("kw", "needle"),
    [
        ({"num_true": False}, "false (numeric oracle rejects)"),
        (
            {
                "num_true": None,
                "view_verdict": "conditional",
                "view_guard": "dim 0 only",
            },
            "conditional truth (view-oracle): dim 0 only",
        ),
        (
            {"num_true": None, "view_verdict": "ill-formed"},
            "ill-formed RHS (view-oracle)",
        ),
        (
            {"num_true": None, "matches": 0},
            "inapplicable (no real match)",
        ),
        ({"num_true": None, "matches": 2}, "unproven (no oracle)"),
        (
            {"relation": "inverse", "fires": 1, "paid": 1},
            "not new (inverse)",
        ),
        ({"fires": 0, "paid": 0}, "no firing on a real model"),
        (
            {"fires": 2, "fires_typed": 2, "paid": 0},
            "fires but never lowers cost",
        ),
        (
            {
                "fires": 3,
                "fires_typed": 1,
                "fires_ill_typed": 2,
                "paid": 0,
            },
            "never lowers cost (2/3 fires ill-typed)",
        ),
        (
            {
                "fires": 2,
                "fires_typed": 0,
                "fires_ill_typed": 2,
                "paid": 0,
            },
            "every fire mints an ill-typed member (2/2)",
        ),
        (
            {"fires": 2, "fires_typed": 2, "paid": 1, "verify_fail": 1},
            "lowered modules differ",
        ),
        (
            {"fires": 2, "fires_typed": 2, "paid": 1, "cert_fail": 1},
            "certificate fails to replay",
        ),
        (
            {
                "fires": 2,
                "fires_typed": 2,
                "paid": 1,
                "closure_ratio": 2.5,
            },
            "closure blow-up",
        ),
        (
            {
                "fires": 4,
                "fires_typed": 0,
                "fires_ill_typed": 4,
                "paid": 0,
                "paid_ill_typed": 4,
            },
            "pays only on ill-typed sites (4 suppressed, 4/4 fires "
            "ill-typed)",
        ),
        (
            {
                "fires": 4,
                "fires_typed": 3,
                "fires_ill_typed": 1,
                "paid": 1,
            },
            "mints ill-typed members on real sites (1/4 fires)",
        ),
    ],
)
def test_no_ship_reason_ladder(kw, needle):
    ev = _ev(**kw)
    assert not ev.shippable
    assert needle in ev.no_ship_reason


def test_conditional_verdict_needs_a_guard_string():
    ev = _ev(num_true=None, view_verdict="conditional", view_guard="")
    assert "guard not isolated" in ev.no_ship_reason


# ---------------------------------------------------------------------------
#  measure — oracles + firing + reach on hand-built terms
# ---------------------------------------------------------------------------


def _factor_proposal() -> pl.Proposal:
    return pl.Proposal(
        name="t:factor_left",
        lhs=_p("add", _p("mul", "A", "B"), _p("mul", "A", "C")),
        rhs=_p("mul", "A", _p("add", "B", "C")),
        family="test",
    )


def test_measure_true_paying_candidate(sink_cost, lib_keys):
    sink, cost = sink_cost
    case = _tiny_cases()[0]
    census_op = {("add", ("mul", "mul")): 3}
    ev = pl.measure(
        _factor_proposal(),
        [case.term],
        [case],
        list(ALL_RULES),
        lib_keys,
        census_op,
        sink,
        cost,
    )
    assert ev.census_sites == 3
    assert ev.matches >= 1 and ev.relaxed >= ev.matches
    assert ev.num_true is True
    assert ev.relation == "new"
    assert ev.fires >= 1 and ev.fire_cases == ("factor",)
    assert ev.fires_ill_typed == 0 and ev.changed == 1 and ev.paid == 1
    assert ev.example != "" and ev.match_term is not None
    # The with-rule saturation paid end-to-end and replayed cleanly.
    assert ev.cost_drop > 0
    assert ev.cert_fail == 0 and ev.verify_fail == 0
    assert ev.closure_safe
    assert ev.shippable
    row = ev.reach[0]
    assert row["add_cost"] < row["base_cost"]
    assert row["add_typed"] is True
    assert row["base_cert"] == "pass" and row["add_cert"] == "pass"


def test_measure_rejects_false_equality(sink_cost, lib_keys):
    sink, cost = sink_cost
    case = _tiny_cases()[1]
    false_p = pl.Proposal(
        name="t:false",
        lhs=_p("mul", "A", "B"),
        rhs=_p("add", "A", "B"),
        family="test",
    )
    ev = pl.measure(
        false_p,
        [case.term],
        [case],
        list(ALL_RULES),
        lib_keys,
        {},
        sink,
        cost,
    )
    assert ev.num_true is False
    assert not ev.shippable
    assert "false" in ev.no_ship_reason


def test_measure_duplicate_relation_blocks_ship(sink_cost, lib_keys):
    sink, cost = sink_cost
    # comm_mul spelled with different metavariables — the library key
    # matches structurally, so the proposal is a duplicate.  The
    # concrete instance lets the derivability oracle see the shipped
    # rule prove it directly.
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    dup = pl.Proposal(
        name="t:comm_dup",
        lhs=_p("mul", "P", "Q"),
        rhs=_p("mul", "Q", "P"),
        family="test",
        instance=(_p("mul", x, y), _p("mul", y, x)),
    )
    ev = pl.measure(
        dup, [], [], list(ALL_RULES), lib_keys, {}, sink, cost
    )
    assert ev.relation == "duplicate"
    # A duplicate of a shipped rule is provable under that rule set.
    assert ev.derivable
    assert not ev.shippable
    assert "not new (duplicate)" in ev.no_ship_reason


def test_measure_no_match_is_inapplicable(sink_cost, lib_keys):
    sink, cost = sink_cost
    case = _tiny_cases()[1]
    no_match = pl.Proposal(
        name="t:nomatch",
        lhs=_p("tanh", _p("tanh", "A")),
        rhs=_p("tanh", "A"),
        family="test",
    )
    ev = pl.measure(
        no_match,
        [case.term],
        [case],
        list(ALL_RULES),
        lib_keys,
        {},
        sink,
        cost,
    )
    assert ev.matches == 0 and ev.num_true is None
    assert "inapplicable" in ev.no_ship_reason
    assert ev.fires == 0 and ev.reach == ()


def test_measure_view_oracle_downgrades_conditional(
    sink_cost, lib_keys
):
    """``select_add`` is true on the real match — but the view oracle
    finds failing synthesized instances and reports *conditional*,
    which holds the candidate out of the ship set (num_true back to
    ``None``)."""
    sink, cost = sink_cost
    torch.manual_seed(0)
    u, w = _v("u", 4, 4), _v("w", 4, 4)
    case = _case(
        "seladd",
        _p(
            "add",
            _p("select", u, dim=0, index=1),
            _p("select", w, dim=0, index=1),
        ),
        u,
        w,
    )
    p = pl.Proposal(
        name="t:sel_add",
        lhs=_p(
            "add",
            _p("select", "A", dim="D", index="I"),
            _p("select", "B", dim="D", index="I"),
        ),
        rhs=_p("select", _p("add", "A", "B"), dim="D", index="I"),
        family="test",
    )
    ev = pl.measure(
        p,
        [case.term],
        [case],
        list(ALL_RULES),
        lib_keys,
        {},
        sink,
        cost,
        view_oracle=True,
    )
    assert ev.view_verdict == "conditional"
    assert ev.num_true is None
    assert "conditional truth" in ev.no_ship_reason
    # Without the oracle the single real match reports plain true.
    ev2 = pl.measure(
        p,
        [case.term],
        [case],
        list(ALL_RULES),
        lib_keys,
        {},
        sink,
        cost,
        view_oracle=False,
    )
    assert ev2.num_true is True and ev2.view_verdict == ""


def test_fire_and_reach_helpers(sink_cost):
    sink, cost = sink_cost
    case = _tiny_cases()[0]
    proposal = _factor_proposal()
    rule = proposal.as_rule()
    # A proposal that cannot fire leaves the evidence untouched.
    dead = pl.Proposal("t:dead", _p("tanh", "A"), "A", "t")
    ev = pl.Evidence(proposal=dead)
    pl._fire(dead, [case], sink, cost, ev)
    assert ev.fires == 0 and ev.fire_cases == ()
    # A paying rule: fires, verified, changed, paid.
    ev = pl.Evidence(proposal=proposal)
    pl._fire(proposal, [case], sink, cost, ev)
    assert ev.fires >= 1 and ev.paid == 1 and ev.verify_fail == 0
    pl._reach(proposal, [case], [], cost, ev)
    assert len(ev.reach) == 1
    assert ev.reach[0]["new_fires"].get(rule.name, 0) >= 1
    assert ev.cost_drop > 0 and ev.reach_ill == 0


def test_reach_row_suppresses_untyped_and_reports_fires(sink_cost):
    _, cost = sink_cost
    case = _tiny_cases()[0]
    rule = _factor_proposal().as_rule()
    row = pl._reach_row(case, [], [rule], cost)
    assert row["model"] == "factor"
    assert row["new_fires"].get(rule.name, 0) >= 1
    assert row["changed"] is True
    assert row["add_typed"] is True
    assert row["add_cost"] < row["base_cost"]
    assert row["base_cert"] == "pass" and row["add_cert"] == "pass"


# ---------------------------------------------------------------------------
#  rank + the evidence-store reconstruction
# ---------------------------------------------------------------------------


def test_rank_orders_ship_then_drop_then_fires():
    winner = _ev(fires=4, paid=2, cost_drop=0.5)
    loser = _ev(fires=1, paid=1, cost_drop=0.1)
    false = _ev(num_true=False, cost_drop=1.0)
    ranked = pl.rank([false, loser, winner])
    assert [e.proposal.name for e in ranked] == [
        winner.proposal.name,
        loser.proposal.name,
        false.proposal.name,
    ]
    # Deterministic on identical input.
    assert pl.rank(ranked) == ranked


def test_rank_tiebreaks_are_deterministic():
    a = _ev(fires=1, paid=1, cost_drop=0.2, matches=2)
    b = _ev(fires=1, paid=1, cost_drop=0.2, matches=5)
    ranked = pl.rank([a, b])
    assert ranked[0].matches == 5


def test_evidence_from_row_roundtrips_the_verdict():
    ev = _ev(
        fires=3,
        fires_typed=3,
        paid=2,
        changed=2,
        cost_drop=0.25,
        closure_ratio=1.5,
        matches=4,
        relaxed=6,
        census_sites=7,
        example="mul(a, b)",
        witness=("comm_mul",),
        fire_cases=("m1", "m2"),
    )
    row = ev_store.verdict_row("k", ev, "lhs", "rhs")
    got = pl._evidence_from_row(ev.proposal, row)
    assert got.fires == 3 and got.paid == 2 and got.changed == 2
    assert got.cost_drop == pytest.approx(0.25)
    assert got.closure_ratio == pytest.approx(1.5)
    assert got.witness == ("comm_mul",)
    assert got.fire_cases == ("m1", "m2")
    assert got.shippable
    # Fields with no verdict columns restore at their defaults —
    # the documented under-reporting of cached rows.
    assert got.fires_typed == 0 and got.match_term is None


# ---------------------------------------------------------------------------
#  run_pipeline — the real loop on the tiny corpus
# ---------------------------------------------------------------------------


def test_run_pipeline_ranks_a_ship_candidate(monkeypatch):
    _patch_tiny(monkeypatch)
    result = pl.run_pipeline(vocab="hand", view_oracle=False)
    assert result["n_models"] == 3 and result["n_bench"] == 0
    assert result["n_intake"] == 0
    assert result["n_search_rules"] == len(ALL_RULES)
    assert result["proposals"] > 10
    ranked = result["ranked"]
    assert len(ranked) == result["proposals"]
    # The ranking is best-first: the known payer leads.
    top = ranked[0]
    assert top.proposal.name == "factor_left"
    assert top.shippable and top.fire_cases == ("factor",)
    # Ranking is consistent with the documented key order.
    assert ranked == pl.rank(list(ranked))
    # Every no-ship candidate carries a reason.
    assert all(e.shippable or e.no_ship_reason for e in ranked)
    # The duplicate select_mul shape was proposed and recognized.
    sel = next(
        e for e in ranked if e.proposal.name == "census:mul_select"
    )
    assert sel.relation == "duplicate"
    assert not sel.shippable


def test_run_pipeline_holdout_rediscovers_softmax(monkeypatch):
    _patch_tiny(monkeypatch)
    result = pl.run_pipeline(vocab="hand", holdout="softmax_fold")
    held = result["held_out"]
    assert held["found"] and held["top"]
    assert held["candidate"] == "recognize:softmax"
    assert held["shippable"]
    assert held["census_sites"] >= 1 and held["fires"] >= 1
    assert held["cost_drop"] > 0
    # The search rule set genuinely lost the held-out law.
    assert result["n_search_rules"] == len(ALL_RULES) - 1


def test_run_pipeline_holdout_select_mul(monkeypatch):
    _patch_tiny(monkeypatch)
    result = pl.run_pipeline(
        vocab="hand", holdout="select_mul", view_oracle=False
    )
    held = result["held_out"]
    assert held["found"] and held["shippable"]
    assert held["candidate"] == "census:mul_select"
    assert held["census_generated"]
    assert set(held["sources"]) >= {"census-naturality"}


def test_run_pipeline_view_oracle_gates_select_mul(monkeypatch):
    """With the view oracle on, the rediscovered ``select_mul`` shape
    reports *conditional* — the oracle finds instances where the
    naturality fails, and the ship gate honestly holds it back."""
    _patch_tiny(monkeypatch)
    result = pl.run_pipeline(
        vocab="hand", holdout="select_mul", view_oracle=True
    )
    held = result["held_out"]
    assert held["found"] and not held["shippable"]
    sel = next(
        e
        for e in result["ranked"]
        if e.proposal.name == "census:mul_select"
    )
    assert sel.view_verdict == "conditional"
    assert "conditional truth" in sel.no_ship_reason


def test_run_pipeline_holdout_unknown_rule(monkeypatch):
    _patch_tiny(monkeypatch)
    result = pl.run_pipeline(vocab="hand", holdout="no_such_rule")
    assert result["held_out"]["found"] is False
    assert result["held_out"]["note"] == "unknown rule"


def test_run_pipeline_no_holdout_held_out_record(monkeypatch):
    _patch_tiny(monkeypatch)
    result = pl.run_pipeline(vocab="hand", view_oracle=False)
    assert result["held_out"] == {"holdout": None, "found": False}


def test_run_pipeline_evidence_db_roundtrip(monkeypatch, tmp_path):
    _patch_tiny(monkeypatch)
    db = str(tmp_path / "ev.db")
    first = pl.run_pipeline(
        vocab="hand", evidence_db=db, view_oracle=False
    )
    assert first["cache"]["hits"] == 0
    assert first["cache"]["total"] == first["proposals"]
    second = pl.run_pipeline(
        vocab="hand",
        evidence_db=db,
        use_cache=True,
        view_oracle=False,
    )
    cache = second["cache"]
    assert cache["use_cache"] and cache["hits"] == cache["total"]
    # Cache-served rows reconstruct the verdicts (minus the fields
    # the frozen schema cannot carry).
    for fresh, cached in zip(
        pl.rank(first["ranked"]), second["ranked"], strict=True
    ):
        assert fresh.proposal.name == cached.proposal.name
        assert fresh.shippable == cached.shippable
        assert fresh.fires == cached.fires
        assert fresh.paid == cached.paid
        assert fresh.relation == cached.relation


# ---------------------------------------------------------------------------
#  main — CLI smoke paths
# ---------------------------------------------------------------------------


def test_main_writes_json(monkeypatch, tmp_path, capsys):
    _patch_tiny(monkeypatch)
    out = tmp_path / "p.json"
    rc = pl.main(
        [
            "--vocab",
            "hand",
            "--no-view-oracle",
            "--top",
            "5",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["proposals"] == len(payload["ranked"])
    assert payload["ranked"][0]["name"] == "factor_left"
    assert payload["ranked"][0]["shippable"]
    row = payload["ranked"][0]
    for key in (
        "fires",
        "paid",
        "cost_drop",
        "closure_ratio",
        "no_ship_reason",
        "view_verdict",
    ):
        assert key in row
    printed = capsys.readouterr().out
    assert "ranked candidates" in printed
    assert "held-out rediscovery" in printed


def test_main_evidence_cache_without_db_notes_and_runs(
    monkeypatch, capsys
):
    _patch_tiny(monkeypatch)
    rc = pl.main(
        ["--vocab", "hand", "--no-view-oracle", "--use-evidence-cache"]
    )
    assert rc == 0
    assert "nothing to read, measuring fresh" in capsys.readouterr().out


def test_main_emit_admission_refusal_returns_1(
    monkeypatch, tmp_path, capsys
):
    _patch_tiny(monkeypatch)
    rc = pl.main(
        [
            "--vocab",
            "hand",
            "--no-view-oracle",
            "--emit-admission",
            "grammar:FALSE_mul_factor",
            "--out",
            str(tmp_path / "adm"),
        ]
    )
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().out


def test_main_emit_admission_success(monkeypatch, tmp_path, capsys):
    _patch_tiny(monkeypatch)
    from catopt_discovery import emit

    monkeypatch.setattr(
        emit, "_TENSOR_REL", str(emit.REPO_ROOT / emit._TENSOR_REL)
    )
    out = tmp_path / "adm"
    rc = pl.main(
        [
            "--vocab",
            "hand",
            "--no-view-oracle",
            "--emit-admission",
            "factor_left",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert (out / "admitted_factor_left.py").exists()
    assert (out / "admission_factor_left.patch").exists()
    assert "admission emission" in capsys.readouterr().out


def test_main_emit_admission_unknown_candidate(
    monkeypatch, tmp_path, capsys
):
    _patch_tiny(monkeypatch)
    rc = pl.main(
        [
            "--vocab",
            "hand",
            "--no-view-oracle",
            "--emit-admission",
            "no_such_candidate",
            "--out",
            str(tmp_path / "adm"),
        ]
    )
    assert rc == 1
    assert "unknown candidate" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  The ill-typed suppression end-to-end (the getitem/tuple scenario)
# ---------------------------------------------------------------------------


def _getitem_strip() -> pl.Proposal:
    """The strip candidate whose RHS reads a tuple output as a tensor."""
    return pl.Proposal(
        name="t:add_getitem_id",
        lhs=_p("add", _p("getitem", "U", index="A_i"), "V"),
        rhs=_p("add", "U", "V"),
        family="test",
    )


def _getitem_case() -> TermCase:
    """``var_mean(u)`` is a tuple; ``getitem(·, 0)`` is well-typed — but
    stripping it mints ``add(tuple, v)``, which does not denote."""
    torch.manual_seed(0)
    u, v = _v("u", 4, 4), _v("v", 4, 1)
    vm = _p("var_mean", u, dim=(-1,), correction=0, keepdim=True)
    return _case(
        "tuple_getitem",
        _p("add", _p("getitem", vm, index=0), v),
        u,
        v,
    )


def test_fire_suppresses_ill_typed_pick(sink_cost):
    sink, cost = sink_cost
    proposal = _getitem_strip()
    ev = pl.Evidence(proposal=proposal)
    pl._fire(proposal, [_getitem_case()], sink, cost, ev)
    assert ev.fires == 1
    assert ev.fires_typed == 0 and ev.fires_ill_typed == 1
    # The cheaper extraction is ill-typed: suppressed, not paid — and
    # not a lowering disagreement either.
    assert ev.paid == 0 and ev.paid_ill_typed == 1
    assert ev.changed == 0 and ev.verify_fail == 0
    assert ev.ill_typed_cases == ("tuple_getitem",)


def test_reach_suppresses_ill_typed_drop(sink_cost):
    _, cost = sink_cost
    proposal = _getitem_strip()
    ev = pl.Evidence(proposal=proposal)
    pl._reach(proposal, [_getitem_case()], [], cost, ev)
    # The with-rule cost drop was real but ill-typed — counted in
    # reach_ill, excluded from cost_drop.
    assert ev.reach_ill == 1
    assert ev.cost_drop == 0.0
    assert ev.reach[0]["add_typed"] is False
    assert ev.reach[0]["add_cost"] < ev.reach[0]["base_cost"]


# ---------------------------------------------------------------------------
#  _held_out edge — a real rule no candidate matches
# ---------------------------------------------------------------------------


def test_held_out_known_rule_not_proposed(monkeypatch):
    _patch_tiny(monkeypatch)
    result = pl.run_pipeline(
        vocab="hand", holdout="id_add", view_oracle=False
    )
    held = result["held_out"]
    assert held["holdout"] == "id_add"
    assert held["found"] is False
    assert "note" not in held


# ---------------------------------------------------------------------------
#  main — the remaining report paths
# ---------------------------------------------------------------------------


def test_main_evidence_db_prints_cache_line(
    monkeypatch, tmp_path, capsys
):
    _patch_tiny(monkeypatch)
    db = str(tmp_path / "ev.db")
    args = [
        "--vocab",
        "hand",
        "--no-view-oracle",
        "--evidence-db",
        db,
    ]
    assert pl.main(args) == 0
    capsys.readouterr()
    assert pl.main([*args, "--use-evidence-cache"]) == 0
    out = capsys.readouterr().out
    assert "evidence db:" in out
    assert "verdicts served from cache" in out


def test_main_holdout_prints_rediscovery_verdict(monkeypatch, capsys):
    _patch_tiny(monkeypatch)
    rc = pl.main(
        ["--vocab", "hand", "--holdout", "softmax_fold", "--top", "3"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "held out: softmax_fold — rediscovered as" in out
    assert "recognize:softmax" in out
    assert "verdict: PASS" in out


def test_main_holdout_not_rediscovered(monkeypatch, capsys):
    _patch_tiny(monkeypatch)
    rc = pl.main(
        ["--vocab", "hand", "--holdout", "id_add", "--top", "3"]
    )
    assert rc == 0
    assert (
        "held out: id_add — NOT rediscovered" in capsys.readouterr().out
    )


def test_main_view_oracle_tally_and_conditional_lines(
    monkeypatch, capsys
):
    """With the oracle on, view candidates get verdicts; the report
    prints the tally and each conditional candidate's guard."""
    _patch_tiny(monkeypatch)
    rc = pl.main(
        ["--vocab", "hand", "--holdout", "select_mul", "--top", "3"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "view-oracle:" in out
    assert "conditional" in out
    # The gated rediscovery reports an honest FAIL verdict.
    assert "verdict: FAIL" in out


def test_main_no_ship_candidates_prints_none(monkeypatch, capsys):
    """A corpus nothing ships on gets the honest empty ship list."""
    torch.manual_seed(0)
    x = _v("x", 4, 4)
    cases = [_case("negged", _p("neg", x), x)]
    census = _tiny_census(cases)
    monkeypatch.setattr(pl, "run_census", lambda top=400: census)
    monkeypatch.setattr(pl, "_bench_cases", lambda: ([], []))
    monkeypatch.setattr(pl, "model_cases", lambda: (cases, []))
    monkeypatch.setattr(li, "load_cases", lambda *a, **k: [])
    monkeypatch.setattr(li, "probe_cases", lambda *a, **k: [])
    rc = pl.main(["--vocab", "hand", "--no-view-oracle"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "none — no candidate clears the ship bar" in out
    assert "no holdout" in out


def test_bad_subterms_non_op_is_clean():
    assert pl._bad_subterms(_v("x", 4)) == frozenset()
    assert pl._bad_subterms(Const(1)) == frozenset()


def test_measure_view_oracle_true_verdict(sink_cost, lib_keys):
    """``neg`` commutes with ``select`` on every synthesized instance —
    the oracle's ``true`` verdict sets ``num_true`` where the
    single-instance oracle could not reach."""
    sink, cost = sink_cost
    p = pl.Proposal(
        name="t:neg_sel",
        lhs=_p("neg", _p("select", "A", dim="D", index="I")),
        rhs=_p("select", _p("neg", "A"), dim="D", index="I"),
        family="test",
    )
    ev = pl.measure(
        p, [], [], list(ALL_RULES), lib_keys, {}, sink, cost
    )
    assert ev.view_verdict == "true"
    assert ev.num_true is True
    # No real match -> inapplicable anyway, and honest about it.
    assert not ev.shippable


def test_measure_view_oracle_false_verdict(sink_cost, lib_keys):
    """A view candidate that is equal nowhere is rejected by the
    oracle, not just left unproven."""
    sink, cost = sink_cost
    p = pl.Proposal(
        name="t:false_view",
        lhs=_p("neg", _p("select", "A", dim="D", index="I")),
        rhs=_p("select", _p("mul", "A", "A"), dim="D", index="I"),
        family="test",
    )
    ev = pl.measure(
        p, [], [], list(ALL_RULES), lib_keys, {}, sink, cost
    )
    assert ev.view_verdict == "false"
    assert ev.num_true is False
    assert "false" in ev.no_ship_reason


def test_typed_probe_audit_matches_probe_count(sink_cost):
    """The audit's fire count equals the probe's — both replay the same
    lone-rule run on the same case."""
    _, cost = sink_cost
    case = _tiny_cases()[0]
    proposal = _factor_proposal()
    audit = pl._typed_probe(case, proposal, cost)
    assert audit.fires >= 1
    assert audit.typed == audit.fires and audit.ill == 0
    assert not audit.pick_ill


def test_term_typed_early_false_on_bad_shape():
    x = _v("x", 4, 4)
    # A shape-bad subterm short-circuits before the eval leg runs.
    assert not pl._term_typed(_p("matmul", x))


def test_typed_probe_unchanged_pick(sink_cost):
    """A fire whose extraction keeps the input term is audited typed
    with no pick to inspect."""
    _, cost = sink_cost
    torch.manual_seed(0)
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    case = _case("comm", _p("mul", x, y), x, y)
    p = pl.Proposal(
        "t:comm",
        _p("mul", "A", "B"),
        _p("mul", "B", "A"),
        "t",
    )
    audit = pl._typed_probe(case, p, cost)
    assert audit.fires >= 1 and audit.ill == 0
