"""Tests for ``catopt_discovery.coherence2`` — the depth-2 probes.

The pair catalogue (``catopt_discovery.coherence``) is covered by
``test_discovery_coherence``; this file covers the layer above it on
two hand-picked shrunk universes:

* ``{comm_add, select_mul, t_reorder}`` — a derivable law provable
  from the primitives alone (stratification rank 1);
* ``{silu_expand, silu_fold, silu_mul_form, swiglu_fuse}`` — an
  inverse-pair cyclic SCC, a downstream singleton, and a genuine
  non-pair-confluent critical pair (``silu_expand`` x ``swiglu_fuse``)
  that ``silu_fold`` mediates.

Plus the reach probe on a small real corpus, the cost accounting,
the sanity table, and a ``main`` smoke run over the shrunk universe.
"""

import json

import pytest
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import coherence as lc
from catopt_discovery import coherence2 as c2
from catopt_discovery import verifier as lv


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


_BY_NAME = {r.name: r for r in ALL_RULES}

#: A rule provable from ``comm_add`` alone — the rank-1 derivable.
_T_REORDER = Rewrite(
    "t_reorder",
    _p("add", _p("add", "P", "Q"), "R"),
    _p("add", "R", _p("add", "Q", "P")),
)
_UNI_RANK = [
    _BY_NAME["comm_add"],
    _BY_NAME["select_mul"],
    _T_REORDER,
]
_UNI_CYC = [
    _BY_NAME["silu_expand"],
    _BY_NAME["silu_fold"],
    _BY_NAME["silu_mul_form"],
    _BY_NAME["swiglu_fuse"],
]
_UNI_MED = [
    _BY_NAME["silu_expand"],
    _BY_NAME["silu_fold"],
    _BY_NAME["swiglu_fuse"],
]

#: An unsound collapse pair whose reducts can never rejoin — the
#: honest ``divergent`` mediator row.
_T_INNER = Rewrite("t_inner", _p("add", "P", "Q"), "P")
_T_OUTER = Rewrite(
    "t_outer", _p("mul", _p("add", "A", "B"), "C"), "A"
)


def _inst(rules):
    out = {}
    for r in rules:
        got = lc._instance(r)
        if got is not None:
            out[r.name] = got
    return out


@pytest.fixture(scope="module")
def cat_rank():
    return lc.catalogue(list(_UNI_RANK))


@pytest.fixture(scope="module")
def cat_cyc():
    return lc.catalogue(list(_UNI_CYC))


@pytest.fixture(autouse=True)
def _restore_verify():
    """``_install_counter`` wraps ``lv.verify_law`` permanently —
    restore it after every test."""
    orig = lv.verify_law
    yield
    lv.verify_law = orig


# ---------------------------------------------------------------------------
#  Probe 1 — stratification, SCCs, grounding
# ---------------------------------------------------------------------------


def test_sccs_cycles_and_singletons():
    sccs = c2._sccs(
        {"a", "b", "c", "d"},
        {"a": {"b"}, "b": {"a"}, "c": {"c"}, "d": {"a"}},
    )
    assert ["a", "b"] in sccs
    assert ["c"] in sccs
    assert ["d"] in sccs


def test_premises_unions_all_sources():
    p = lc.LawProfile(
        name="x",
        instanced=True,
        verdict="derivable",
        witness=("a",),
        essential=("b",),
        direct_from=("c",),
    )
    assert c2._premises(p) == {"a", "b", "c"}


def test_stratify_rank_one_derivable(cat_rank):
    inst = _inst(_UNI_RANK)
    strat = c2._stratify(_UNI_RANK, inst, cat_rank["profiles"])
    assert strat["rank"] == {"t_reorder": 1}
    assert strat["remaining"] == []
    assert strat["cyclic_sccs"] == []
    assert strat["n_primitive"] == 2
    assert sorted(strat["prim_names"]) == ["comm_add", "select_mul"]


def test_stratify_cyclic_scc_and_singleton(cat_cyc):
    inst = _inst(_UNI_CYC)
    strat = c2._stratify(_UNI_CYC, inst, cat_cyc["profiles"])
    # The inverse pair cannot stratify: each twin's proof fires the
    # other.  mul_form is a downstream singleton — it needs a cyclic
    # class member, not itself.
    assert strat["rank"] == {}
    assert sorted(strat["remaining"]) == [
        "silu_expand",
        "silu_fold",
        "silu_mul_form",
    ]
    assert strat["cyclic_sccs"] == [["silu_expand", "silu_fold"]]
    # swiglu_fuse is the only primitive in this universe.
    assert strat["n_primitive"] == 1
    assert strat["prim_names"] == ["swiglu_fuse"]


def test_grounding_check_seeds_cycles(cat_cyc):
    inst = _inst(_UNI_CYC)
    strat = c2._stratify(_UNI_CYC, inst, cat_cyc["profiles"])
    ground = c2._grounding_check(_UNI_CYC, inst, strat)
    # One seed per cyclic SCC: silu_expand is alphabetically first.
    assert ground["seeded_cycles"] == ["silu_expand"]
    # Seeded with the primitives plus the SCC rep, every leftover
    # law derives — silu_fold and silu_mul_form included.
    assert ground["uncovered"] == []
    assert set(ground["seeds"]) == {"silu_expand", "swiglu_fuse"}
    assert ground["effective_basis"] == 2


def test_grounding_check_reports_uncovered():
    """A law the seeds cannot derive is named, not hidden.

    ``comm_add`` sits in ``remaining`` but no seed proves it;
    ``select_mul`` is skipped outright because it IS a cyclic rep.
    """
    uni = [_BY_NAME["comm_add"], _BY_NAME["select_mul"]]
    strat = {
        "rank": {},
        "remaining": ["comm_add", "select_mul"],
        "cyclic_sccs": [["select_mul"]],
        "n_primitive": 0,
        "prim_names": [],
    }
    inst = _inst(uni)
    ground = c2._grounding_check(uni, inst, strat)
    assert ground["seeded_cycles"] == ["select_mul"]
    assert ground["uncovered"] == ["comm_add"]
    assert ground["effective_basis"] == 2


def test_derivation_graph_stats_renders(cat_cyc):
    inst = _inst(_UNI_CYC)
    strat = c2._stratify(_UNI_CYC, inst, cat_cyc["profiles"])
    text = c2._derivation_graph_stats(cat_cyc["profiles"], strat)
    assert "premise edges" in text
    assert "unstratified (cyclic" in text
    assert "SCC { silu_expand, silu_fold }" in text
    # mul_form is the downstream singleton of the cyclic class.
    assert "downstream singletons" in text
    assert "silu_mul_form" in text


def test_derivation_graph_stats_ranks(cat_rank):
    inst = _inst(_UNI_RANK)
    strat = c2._stratify(_UNI_RANK, inst, cat_rank["profiles"])
    text = c2._derivation_graph_stats(cat_rank["profiles"], strat)
    assert "rank 1: 1 laws — t_reorder" in text


# ---------------------------------------------------------------------------
#  Cost accounting
# ---------------------------------------------------------------------------


def test_counter_records_calls_and_renders():
    c2._set_phase("t_phase")
    c2._install_counter()
    x = _v("x", 4, 4)
    lv.verify_law(
        _p("mul", x, x), _p("mul", x, x), list(_UNI_RANK)
    )
    assert c2._CALLS["t_phase"][0] >= 1
    text = c2._cost_table()
    assert "t_phase" in text
    assert "TOTAL" in text
    assert "triples" in text


def test_counted_verify_records_each_call():
    before = dict(c2._CALLS)
    c2._set_phase("direct")
    x = _v("x", 4, 4)
    res = c2._counted_verify(
        _p("add", x, x), _p("add", x, x), list(_UNI_RANK)
    )
    assert res.derivable or res.stop
    assert c2._CALLS["direct"][0] == before.get("direct", [0])[0] + 1


# ---------------------------------------------------------------------------
#  Probe 2 — reach under removal
# ---------------------------------------------------------------------------


def test_reach_budgets_caps_expansive():
    budgets = c2._reach_budgets(_UNI_CYC)
    assert set(budgets) <= {r.name for r in _UNI_CYC}
    assert all(n == c2._EXPANSIVE_BUDGET for n in budgets.values())


def test_saturate_once_counts():
    x = _v("x", 4, 4)
    n_en, n_cl = c2._saturate_once(_p("mul", x, x), _UNI_CYC)
    assert n_en > 0 and n_cl > 0
    assert c2._EGRAPH_RUNS["n"] > 0


def test_reach_probe_on_tiny_universe():
    """The real corpus, a 4-rule universe — ~30 small saturations."""
    cat = lc.catalogue(_UNI_CYC)
    rows = c2._reach_probe(_UNI_CYC, cat["profiles"])
    assert len(rows) == len(_UNI_CYC)
    for r in rows:
        assert r["law"] in {x.name for x in _UNI_CYC}
        assert r["verdict"] in ("primitive", "derivable")
        assert r["d_enodes"] >= 0 and r["d_classes"] >= 0
    text = c2._reach_table(rows)
    assert "Reach under removal" in text
    assert "inert on this corpus" in text


def test_reach_table_renders_marks():
    rows = [
        {
            "law": "a",
            "verdict": "derivable",
            "d_enodes": 3,
            "d_classes": 1,
            "cases_affected": 2,
        },
        {
            "law": "b",
            "verdict": "primitive",
            "d_enodes": 0,
            "d_classes": 0,
            "cases_affected": 0,
        },
    ]
    text = c2._reach_table(rows)
    assert "*a " in text.replace("*a", "*a ")
    assert "b" in text
    assert "(* = derivable law" in text
    # every law load-bearing -> no inert list at all.
    busy = [dict(rows[0])]
    assert "inert on this corpus (zero delta): 0" in c2._reach_table(
        busy
    )


# ---------------------------------------------------------------------------
#  Probe 3 — the mediator table
# ---------------------------------------------------------------------------


def test_pair_reducts_cofiring_and_missing():
    inst = _inst(_UNI_MED)
    rd = c2._pair_reducts(
        _BY_NAME["silu_expand"], _BY_NAME["swiglu_fuse"], inst
    )
    assert rd is not None
    base, _, _ = rd
    assert base == "swiglu_fuse"
    # a law missing from inst skips to the other ordering; a pair
    # sharing no firing instance at all returns None.
    inst_comm = _inst([_BY_NAME["comm_add"], _BY_NAME["select_mul"]])
    assert (
        c2._pair_reducts(
            _BY_NAME["comm_add"], _BY_NAME["select_mul"], inst_comm
        )
        is None
    )
    # both orders fail when neither instance hosts a co-fire.
    assert (
        c2._pair_reducts(_T_INNER, _BY_NAME["select_mul"], inst_comm)
        is None
    )


def test_pair_reducts_second_order():
    """The (a, b) order is tried when (b, a) does not co-fire."""
    inst = _inst([_T_INNER, _T_OUTER])
    rd = c2._pair_reducts(_T_OUTER, _T_INNER, inst)
    assert rd is not None
    assert rd[0] == "t_outer"


def test_pair_reducts_apply_veto():
    """A root ``apply`` the rule's own check vetoes is skipped.

    ``first`` co-hosts the fire (``second`` rewrites inside its
    instance) but its own ``check`` refuses the root application —
    the pair is reported as sharing no firing instance.
    """
    a, b, c = _v("a", 4), _v("b", 4), _v("c", 4)
    veto = Rewrite(
        "t_veto",
        _p("mul", _p("add", "A", "B"), "C"),
        "A",
        check=lambda bound: False,
    )
    inst = {
        "t_veto": (_p("mul", _p("add", a, b), c), a),
        "t_inner": (_p("add", a, b), a),
    }
    assert c2._pair_reducts(veto, _T_INNER, inst) is None


def test_mediator_probe_lib_mediated():
    inst = _inst(_UNI_MED)
    row = c2._mediator_probe(
        _BY_NAME["silu_expand"], _BY_NAME["swiglu_fuse"], inst,
        _UNI_MED,
    )
    assert row is not None
    assert row.pair == ("silu_expand", "swiglu_fuse")
    assert row.status == "lib-mediated"
    assert row.singles == ("silu_fold",)
    assert "silu_fold" in row.essential
    assert row.minimal == ("silu_fold",)
    assert row.base == "swiglu_fuse"


def test_mediator_probe_divergent():
    inst = _inst([_T_INNER, _T_OUTER])
    row = c2._mediator_probe(_T_INNER, _T_OUTER, inst, [_T_INNER, _T_OUTER])
    assert row is not None
    assert row.status == "divergent"
    assert row.singles == ()
    assert "no rejoin" in row.note


def test_mediator_probe_pair_confluent_returns_none():
    uni = [
        _BY_NAME["silu_expand"],
        _BY_NAME["silu_fold"],
        _BY_NAME["silu_mul_form"],
    ]
    inst = _inst(uni)
    # expand x mul_form rejoins under the pair alone — no triple row.
    assert (
        c2._mediator_probe(
            _BY_NAME["silu_expand"],
            _BY_NAME["silu_mul_form"],
            inst,
            uni,
        )
        is None
    )


def test_mediator_probe_no_shared_instance_returns_none():
    inst = _inst([_BY_NAME["comm_add"], _BY_NAME["select_mul"]])
    assert (
        c2._mediator_probe(
            _BY_NAME["comm_add"],
            _BY_NAME["select_mul"],
            inst,
            _UNI_MED,
        )
        is None
    )


def test_mediator_probe_combination_paths(monkeypatch):
    """Stub the oracle at the ``verify_law`` seam to reach the
    beyond-triples bookkeeping: a pair the universe rejoins but no
    single rule mediates."""
    a = _BY_NAME["silu_expand"]
    b = _BY_NAME["swiglu_fuse"]
    uni = [a, b, _BY_NAME["silu_fold"], _BY_NAME["comm_add"]]
    inst = _inst(uni)

    def result(derivable: bool, witness=()):
        return lv.LawResult(
            derivable=derivable,
            witness_rules=witness,
            replayable=bool(witness),
        )

    # 1) no single mediator, no witness to reduce — ``cur`` starts
    #    empty and no recheck happens at all.
    def fake1(lhs, rhs, rules, **kw):
        names = {r.name for r in rules}
        if names == {a.name, b.name}:
            return result(False)
        if names == {a.name, b.name, "silu_fold", "comm_add"}:
            return result(True)  # whole universe, opaque witness
        return result(False)

    monkeypatch.setattr(lv, "verify_law", fake1)
    row = c2._mediator_probe(a, b, inst, uni)
    assert row is not None
    assert row.status == "lib-mediated"
    assert row.singles == ()
    assert row.minimal == ()
    assert set(row.essential) <= {a.name, b.name}

    # 2) a non-empty witness combo whose own recheck fails — the
    #    honest "combination" row reporting an empty minimal set.
    calls = {"n": 0}

    def fake_recheck_fails(lhs, rhs, rules, **kw):
        names = {r.name for r in rules}
        if names == {a.name, b.name}:
            return result(False)
        if names == {a.name, b.name, "silu_fold", "comm_add"}:
            calls["n"] += 1
            # the first quartet call (whole-universe) succeeds; the
            # recheck of the found combination does not.
            return result(
                calls["n"] == 1, witness=("silu_fold", "comm_add")
            )
        return result(False)

    monkeypatch.setattr(lv, "verify_law", fake_recheck_fails)
    row = c2._mediator_probe(a, b, inst, uni)
    assert row is not None
    assert row.minimal == ()

    # 3) the witness minus the pair is a real combination; the greedy
    #    reduction drops the member that is not load-bearing.
    def fake2(lhs, rhs, rules, **kw):
        names = {r.name for r in rules}
        if names == {a.name, b.name}:
            return result(False)
        if names == {a.name, b.name, "silu_fold", "comm_add"}:
            return result(True, witness=("silu_fold", "comm_add"))
        # only the full quartet rejoins below — singles stay empty.
        return result(False)

    monkeypatch.setattr(lv, "verify_law", fake2)
    row = c2._mediator_probe(a, b, inst, uni)
    assert row is not None
    assert row.minimal == ("comm_add", "silu_fold")

    # 4) one member of the combination is redundant — the greedy
    #    reduction strips it.  ``{a,b,fold}`` must fail during the
    #    singles census but succeed during the greedy trial — the
    #    two calls share a signature, so order decides.
    seen = {"n": 0}

    def fake4(lhs, rhs, rules, **kw):
        names = {r.name for r in rules}
        if names == {a.name, b.name}:
            return result(False)
        if names == {a.name, b.name, "silu_fold", "comm_add"}:
            return result(True, witness=("silu_fold", "comm_add"))
        if names == {a.name, b.name, "silu_fold"}:
            seen["n"] += 1
            return result(seen["n"] > 1)
        return result(False)

    monkeypatch.setattr(lv, "verify_law", fake4)
    row = c2._mediator_probe(a, b, inst, uni)
    assert row is not None
    assert row.singles == ()
    assert row.minimal == ("silu_fold",)


def test_mediator_table_and_section():
    cat = lc.catalogue(_UNI_MED)
    inst = _inst(_UNI_MED)
    rows = c2._mediator_table(_UNI_MED, inst, cat["confluence"])
    assert len(rows) == 1
    assert rows[0].pair == ("silu_expand", "swiglu_fuse")
    text = c2._mediator_section(rows)
    assert "Mediator table" in text
    assert "silu_fold" in text
    assert "mediator hubs" in text
    # a non-confluent row whose pair shares no instance is skipped.
    conf = [
        *cat["confluence"],
        lc.ConfRow(
            pair=("silu_fold", "swiglu_fuse"),
            base="",
            join_pair=False,
            join_lib=False,
            overlap=False,
        ),
    ]
    rows2 = c2._mediator_table(_UNI_MED, inst, conf)
    assert rows2 == rows


def test_mediator_section_empty_and_multi():
    text = c2._mediator_section([])
    assert "(no non-pair-confluent pairs)" in text
    row = c2.MediatorRow(
        pair=("a", "b"),
        base="a",
        status="lib-mediated",
        singles=(),
        minimal=("c", "d"),
        witness=("c", "d"),
    )
    text = c2._mediator_section([row])
    assert "NO single-rule mediator" in text
    assert "a x b" in text


# ---------------------------------------------------------------------------
#  Sanity, JSON, main
# ---------------------------------------------------------------------------


def test_sanity_section():
    cat = lc.catalogue(_UNI_MED)
    inst = _inst(_UNI_MED)
    med = c2._mediator_table(_UNI_MED, inst, cat["confluence"])
    text = c2._sanity(cat, med, _UNI_MED)
    # the matmul class is not in this universe — an honest FAIL row.
    assert "[FAIL] 4-member matmul" in text
    # silu_fold mediating expand x fuse IS reproduced.
    assert "[PASS] silu_fold mediates silu_expand x swiglu_fuse" in text
    # no layout rules -> the SKIP line.
    assert "[SKIP] linear_to_matmul_t" in text


def test_sanity_layout_rule_present():
    """With a layout rule in the universe the SKIP becomes a count."""
    from catopt_core.laws import ALL_RULES_WITH_LAYOUT

    lm = [
        r
        for r in ALL_RULES_WITH_LAYOUT
        if r.name == "linear_to_matmul_t"
    ]
    assert lm, "linear_to_matmul_t should be a layout rule"
    cat = lc.catalogue(_UNI_MED)
    inst = _inst(_UNI_MED)
    med = c2._mediator_table(_UNI_MED, inst, cat["confluence"])
    text = c2._sanity(cat, med, lm)
    assert "linear_to_matmul_t mediates 0 pairs" in text


def test_jsonable_roundtrips(cat_cyc):
    inst = _inst(_UNI_CYC)
    strat = c2._stratify(_UNI_CYC, inst, cat_cyc["profiles"])
    ground = c2._grounding_check(_UNI_CYC, inst, strat)
    med = c2._mediator_table(_UNI_CYC, inst, cat_cyc["confluence"])
    payload = json.loads(
        json.dumps(
            c2._jsonable(
                strat, ground, med, [{"law": "x", "d_enodes": 0}]
            )
        )
    )
    assert payload["stratification"]["n_primitive"] == 1
    assert payload["grounding"]["effective_basis"] >= 1
    assert isinstance(payload["mediators"], list)
    assert "cost" in payload


def test_main_smoke_tiny_universe(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(c2, "ALL_RULES", list(_UNI_CYC))
    out = tmp_path / "c2.json"
    rc = c2.main(["--skip-reach", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["stratification"]["n_primitive"] == 1
    assert payload["grounding"]["seeded_cycles"] == ["silu_expand"]
    printed = capsys.readouterr().out
    assert "LAW COHERENCE — DEPTH 2" in printed
    assert "Effective basis" in printed
    assert "(reach probe skipped" in printed
    assert "Mediator table" in printed
    assert "Sanity" in printed
    assert "Enumeration cost" in printed
    # a no-flag run executes the reach probe against the real corpus
    # and prints the removal table without writing JSON.
    rc = c2.main([])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "Reach under removal" in printed


def test_main_with_layout_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(
        c2, "ALL_RULES_WITH_LAYOUT", list(_UNI_MED)
    )
    out = tmp_path / "c2l.json"
    rc = c2.main(["--with-layout", "--skip-reach", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["mediators"][0]["pair"] == [
        "silu_expand",
        "swiglu_fuse",
    ]
