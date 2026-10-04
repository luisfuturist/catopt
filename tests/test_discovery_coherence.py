"""Tests for ``catopt_discovery.coherence`` and ``lemma_cert``.

``coherence`` builds the pairwise 3-cell catalogue over a rule
universe: direct derivations ``{A} => B``, whole-library derivability,
essential premises, equivalence classes and bounded local-confluence
probes.  ``lemma_cert`` materializes each ``Rewrite.derivation``
annotation as a replayable certificate and reports the honest verdict
ladder (linear / enumerated / saturation-only / gap).

Both are exercised here over a hand-picked six-rule universe —
``silu_expand``, ``silu_fold``, ``silu_mul_form``, ``comm_add``,
``select_mul`` and a deliberate alpha-duplicate ``zz_comm_twin`` —
which is rich enough to produce every verdict (primitive /
derivable / no-instance), an inverse pair, a duplicate pair,
essential premises, composite direct edges and two genuine
skeleton-overlap critical pairs, all in about a second.
"""

import json

import pytest
from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import coherence as lc
from catopt_discovery import lemma_cert as llc


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


_BY_NAME = {r.name: r for r in ALL_RULES}
_TWIN = Rewrite(
    "zz_comm_twin",
    _p("add", "P", "Q"),
    _p("add", "Q", "P"),
)
_UNIVERSE = [
    _BY_NAME["silu_expand"],
    _BY_NAME["silu_fold"],
    _BY_NAME["silu_mul_form"],
    _BY_NAME["comm_add"],
    _BY_NAME["select_mul"],
    _TWIN,
]


@pytest.fixture(scope="module")
def catalogue():
    return lc.catalogue(list(_UNIVERSE))


# ---------------------------------------------------------------------------
#  Instances
# ---------------------------------------------------------------------------


def test_instance_returns_concrete_pairs():
    lhs, rhs = lc._instance(_BY_NAME["silu_fold"])
    assert lhs.op == "mul" and rhs.op == "silu"


def test_instance_generic_fallback_attr_metavars():
    """A layout-style rule the bench registry does not know gets a
    generic instance: term metavars become (4,4) Vars, ``*0``/``*1``
    attr metavars become the distinct axes 0 and 1."""
    rule = Rewrite(
        "t_transpose",
        _p("transpose", "X", dim0="D0", dim1="D1"),
        "X",
    )
    lhs, rhs = lc._generic_instance(rule)
    assert lhs.op == "transpose"
    assert lhs.attrs == {"dim0": 0, "dim1": 1}
    assert rhs == lhs.args[0]


def test_instance_honest_none_on_check_veto():
    rule = Rewrite(
        "t_veto",
        _p("add", "A", "B"),
        "A",
        check=lambda bound: False,
    )
    assert lc._generic_instance(rule) is None
    assert lc._instance(rule) is None


def test_instance_none_on_unbound_rhs_metavar():
    # RHS mentions a metavariable the LHS never binds — instantiate
    # fails with KeyError, reported as no-instance, never guessed.
    rule = Rewrite("t_free", _p("add", "A", "B"), "C")
    assert lc._generic_instance(rule) is None


# ---------------------------------------------------------------------------
#  The catalogue
# ---------------------------------------------------------------------------


def test_catalogue_instances_every_rule(catalogue):
    assert sorted(catalogue["instanced"]) == sorted(
        r.name for r in _UNIVERSE
    )
    assert catalogue["rules"] == [r.name for r in _UNIVERSE]


def test_catalogue_verdicts_and_witnesses(catalogue):
    profiles = catalogue["profiles"]
    # select_mul is a genuine basis element of this tiny universe —
    # nothing else merges its two instance sides.
    assert profiles["select_mul"].verdict == "primitive"
    # The silu laws are each provable from the rest.
    fold = profiles["silu_fold"]
    assert fold.verdict == "derivable"
    assert fold.witness == ("silu_expand",)
    assert fold.essential == ("silu_expand",)
    # The inverse direction is symmetric — expand proves fold, fold
    # proves expand.
    expand = profiles["silu_expand"]
    assert expand.verdict == "derivable"
    assert expand.witness == ("silu_fold",)
    assert expand.essential == ("silu_fold",)
    # silu_mul_form has two independent routes to its equality — no
    # single premise is essential.
    mf = profiles["silu_mul_form"]
    assert mf.verdict == "derivable"
    assert mf.essential == ()
    assert set(mf.direct_from) == {"silu_expand", "silu_fold"}
    # The duplicate pair prove each other — and each is the other's
    # ONLY route, so both are essential premises.
    comm = profiles["comm_add"]
    assert comm.verdict == "derivable"
    assert comm.witness == ("zz_comm_twin",)
    assert comm.essential == ("zz_comm_twin",)


def test_catalogue_direct_edges(catalogue):
    profiles = catalogue["profiles"]
    assert profiles["silu_expand"].direct_to == (
        "silu_fold",
        "silu_mul_form",
    )
    assert profiles["silu_mul_form"].direct_to == ()
    # The twin's only edge is its duplicate's.
    assert profiles["zz_comm_twin"].direct_from == ("comm_add",)


def test_catalogue_equivalence_classes(catalogue):
    assert catalogue["eq_classes"] == [
        ["comm_add", "zz_comm_twin"],
        ["silu_expand", "silu_fold"],
    ]
    assert ["silu_expand", "silu_fold"] in (
        catalogue["structural"]["inverses"]
    )
    assert ["comm_add", "zz_comm_twin"] in (
        catalogue["structural"]["duplicates"]
    )


def test_catalogue_confluence_row(catalogue):
    conf = {r.pair: r for r in catalogue["confluence"]}
    row = conf[("silu_expand", "silu_mul_form")]
    assert row.base == "silu_mul_form"
    assert row.join_pair is True
    assert row.overlap is True
    # The duplicate pair co-fires — and its one-step reducts rejoin.
    twin = conf[("comm_add", "zz_comm_twin")]
    assert twin.join_pair is True and twin.overlap is True


# ---------------------------------------------------------------------------
#  The confluence probe pieces
# ---------------------------------------------------------------------------


def test_in_skeleton_distinguishes_metavar_descent():
    lhs = _BY_NAME["silu_fold"].lhs  # mul(A, sigmoid(A))
    assert lc._in_skeleton(lhs, ())
    assert lc._in_skeleton(lhs, (1,))
    assert not lc._in_skeleton(lhs, (0,))  # inside the A metavar
    assert not lc._in_skeleton(lhs, (1, 0))  # inside A again
    assert not lc._in_skeleton(lhs, (5,))  # off the pattern


def test_first_fire_reports_path_and_reduct():
    rule = _BY_NAME["silu_expand"]
    x = _v("x", 4, 4)
    fired = lc._first_fire(rule, _p("mul", _p("silu", x), x))
    assert fired is not None
    path, reduct = fired
    assert path == (0,)
    assert reduct == _p("mul", _p("mul", x, _p("sigmoid", x)), x)
    # No firing -> None, never a fabricated reduct.
    assert lc._first_fire(rule, _p("add", x, x)) is None
    # A firing that yields the same term is not a fire.
    comm = _BY_NAME["comm_add"]
    assert lc._first_fire(comm, _p("add", x, x)) is None


def test_confluence_probe_none_without_shared_instance():
    inst = {
        n: lc._instance(_BY_NAME[n])
        for n in ("comm_add", "silu_expand")
    }
    assert (
        lc._confluence_probe(
            _BY_NAME["comm_add"],
            _BY_NAME["silu_expand"],
            inst,
            list(_UNIVERSE),
        )
        is None
    )


def test_confluence_probe_joined_row():
    inst = {
        n: lc._instance(_BY_NAME[n])
        for n in ("silu_expand", "silu_mul_form")
    }
    row = lc._confluence_probe(
        _BY_NAME["silu_expand"],
        _BY_NAME["silu_mul_form"],
        inst,
        list(_UNIVERSE),
    )
    assert row is not None
    assert row.pair == ("silu_expand", "silu_mul_form")
    assert row.join_pair is True
    assert row.overlap is True
    assert row.note == ""


# ---------------------------------------------------------------------------
#  emit_basis — the annotation table
# ---------------------------------------------------------------------------


def test_emit_basis_kinds_and_derivations(catalogue):
    table = lc.emit_basis(catalogue)
    # Primitives plus the alphabetically-first seed of each cycle.
    assert table["comm_add"]["kind"] == "axiom"
    assert table["silu_expand"]["kind"] == "axiom"
    assert table["select_mul"]["kind"] == "axiom"
    assert table["comm_add"]["derivation"] == []
    # The non-seed members are lemmas with a measured premise.
    assert table["silu_fold"] == {
        "kind": "lemma",
        "derivation": ["silu_expand"],
    }
    assert table["silu_mul_form"]["kind"] == "lemma"
    assert table["silu_mul_form"]["derivation"] == ["silu_expand"]
    # The literal re-spelling is redundant — an earlier classmate is
    # already its alpha-duplicate.
    assert table["zz_comm_twin"]["kind"] == "redundant"
    assert table["zz_comm_twin"]["derivation"] == ["comm_add"]


def test_basis_section_renders_annotations(catalogue):
    text = lc._basis_section(lc.emit_basis(catalogue))
    assert "axiom: 3" in text and "lemma: 2" in text
    assert "redundant: 1" in text
    assert 'derivation=("silu_expand",)' in text
    assert "tags+=(tags.REDUNDANT,)" in text


def test_report_covers_every_section(catalogue):
    text = lc._report(catalogue, list(_UNIVERSE))
    assert "LAW COHERENCE CATALOGUE" in text
    assert "universe: 6 rules | instanced: 6" in text
    assert "primitive" in text and "derivable" in text
    # A commutative rule's swap normalizes to itself — the duplicate
    # pair is tagged both ways.
    assert "{ comm_add, zz_comm_twin } (duplicate, inverse)" in text
    assert "{ silu_expand, silu_fold } (inverse)" in text
    # Composite direct edges are starred; structural echoes are not.
    assert "*silu_expand ⇒ silu_mul_form   [composite]" in text
    assert "confluent: 2" in text


def test_jsonable_roundtrips(catalogue):
    payload = json.loads(json.dumps(lc._jsonable(catalogue)))
    assert payload["n_rules"] == 6
    assert payload["profiles"]["select_mul"]["verdict"] == "primitive"
    assert payload["basis"]["zz_comm_twin"]["kind"] == "redundant"
    assert payload["confluence"][0]["join_pair"] is True
    assert payload["confluence"][0]["overlap"] is True


def test_cluster_section_names_known_neighbourhood():
    """The hand-picked cluster section runs against the full layout
    universe — a fixed cut, independent of the test universe."""
    text = lc._cluster_section()
    assert "Cluster skeleton" in text
    assert "transpose_push_mul ⇒ transpose_pull_mul   [inverse]" in text
    # At least one cluster pair is reported confluent.
    assert "confluent" in text


def test_main_on_tiny_universe(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lc, "ALL_RULES", list(_UNIVERSE))
    out = tmp_path / "coh.json"
    rc = lc.main(["--emit-basis", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_rules"] == 6
    assert payload["basis"]["silu_fold"]["kind"] == "lemma"
    printed = capsys.readouterr().out
    assert "LAW COHERENCE CATALOGUE" in printed
    assert "Axiom/lemma basis" in printed
    assert "Cluster skeleton" in printed


# ---------------------------------------------------------------------------
#  lemma_cert — materialize / probe / serialize
# ---------------------------------------------------------------------------


def test_materialize_linear_certificate():
    row, cert = llc.materialize(_BY_NAME["silu_fold"], list(ALL_RULES))
    assert row.verdict == "linear"
    # The inverse twin proves rhs->lhs: the axiom applied to the
    # lemma's RHS instance IS the proof.
    assert row.direction == "rhs->lhs"
    assert row.n_steps == 1
    assert row.rules_used == ("silu_expand",)
    assert row.premise_cover and cert is not None


def test_materialize_composite_path():
    row, cert = llc.materialize(
        _BY_NAME["silu_mul_form"], list(ALL_RULES)
    )
    assert row.verdict == "linear" and row.direction == "lhs->rhs"
    assert [(s.rule, s.path) for s in cert.steps] == [
        ("silu_expand", (0,))
    ]


def test_materialize_gap_on_unprovable_annotation():
    # A rule whose derivation annotation names real premises that do
    # NOT prove its instance — the annotation-drift verdict.
    drift = Rewrite(
        "t_drift",
        _p("add", "A", "B"),
        "A",
        derivation=("comm_add",),
    )
    row, cert = llc.materialize(drift, list(ALL_RULES))
    assert row.verdict == "gap"
    assert cert is None


def test_materialize_bad_derivation_unknown_premise():
    ghost = Rewrite(
        "t_ghost",
        _p("add", "A", "B"),
        "A",
        derivation=("no_such_rule",),
    )
    row, cert = llc.materialize(ghost, list(ALL_RULES))
    assert row.verdict == "bad-derivation"
    assert "no_such_rule" in row.note
    assert cert is None


def test_materialize_no_instance():
    veto = Rewrite(
        "t_veto",
        _p("add", "A", "B"),
        "A",
        check=lambda bound: False,
    )
    row, cert = llc.materialize(veto, list(ALL_RULES))
    assert row.verdict == "no-instance"
    assert cert is None


def test_probe_universe_minus_self():
    row, cert = llc.probe("silu_expand", list(ALL_RULES))
    assert row.verdict == "linear"
    # silu_fold — the lemma — is the replayable witness in the
    # reverse direction.
    assert row.direction == "rhs->lhs"
    assert "silu_fold" in row.rules_used
    assert cert is not None


def test_probe_reports_gap_for_primitive_rule():
    # In the tiny universe the twin would prove comm_add's instance —
    # under the real library minus itself nothing does.
    row, cert = llc.probe("comm_add", list(ALL_RULES))
    assert row.verdict == "gap"
    assert cert is None


def test_materialize_all_only_derivation_carriers():
    rows, certs = llc.materialize_all(list(_UNIVERSE))
    names = {r.name for r in rows}
    # Only the rules carrying a derivation annotation get a row.
    assert names == {"silu_fold", "silu_mul_form"}
    assert names == set(certs)


def test_serialize_demo_roundtrip():
    record, replayed = llc.serialize_demo(
        "silu_mul_form", list(ALL_RULES)
    )
    assert record["version"] == 1
    assert record["replayable"] is True
    assert record["rules_used"] == ["silu_expand"]
    # The replayed term is the certificate's dst — the expanded
    # ``mul(mul(g, sigmoid(g)), u)`` spelling reached by strict replay.
    assert replayed.op == "mul"
    assert replayed.args[0].op == "mul"
    assert replayed.args[0].args[1].op == "sigmoid"


def test_serialize_demo_refuses_certless_rule():
    with pytest.raises(ValueError, match="no certificate"):
        llc.serialize_demo("comm_add", list(ALL_RULES))


def test_lemma_cert_report_and_jsonable():
    rows, certs = llc.materialize_all(list(_UNIVERSE))
    text = llc._report(rows, "TINY")
    assert "LEMMA CERTIFICATES" in text
    assert "verdicts: linear: 2" in text
    payload = json.loads(json.dumps(llc._jsonable(rows, certs)))
    assert {r["name"] for r in payload["lemmas"]} == {
        "silu_fold",
        "silu_mul_form",
    }
    assert set(payload["certificates"]) == set(certs)
    assert payload["certificates"]["silu_fold"]["replayable"] is True


def test_lemma_cert_main_smoke(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(llc, "ALL_RULES", list(_UNIVERSE))
    out = tmp_path / "certs.json"
    rc = llc.main(
        [
            "--json",
            str(out),
            "--probe",
            "silu_expand",
            "--demo",
            "silu_mul_form",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert len(payload["lemmas"]) == 2
    printed = capsys.readouterr().out
    assert "LEMMA CERTIFICATES" in printed
    assert "Probe — silu_expand" in printed
    assert "Serialization demo — silu_mul_form" in printed
    assert "verify_certificate(strict=True): OK" in printed


# ---------------------------------------------------------------------------
#  Second pass — the remaining catalogue/probe edges
# ---------------------------------------------------------------------------


def _veto_rule(name: str) -> Rewrite:
    """A rule whose check vetoes every binding — no honest instance."""
    return Rewrite(
        name, _p("add", "A", "B"), "A", check=lambda bound: False
    )


def test_generic_instance_hook_failures():
    # A check that raises is a veto, not a guess.
    raising = Rewrite(
        "t_raise",
        _p("add", "A", "B"),
        "A",
        check=lambda bound: 1 / 0,
    )
    assert lc._generic_instance(raising) is None
    # A derive that raises, or returns None, yields no instance either.
    for derive in (lambda bound: 1 / 0, lambda bound: None):
        rule = Rewrite(
            "t_derive",
            _p("add", "A", "B"),
            _p("add", "A", "A", extra="E"),
            derive=derive,
        )
        assert lc._generic_instance(rule) is None


def test_in_skeleton_off_pattern_and_const():
    pat = _p("mul", "A", Const(0))
    # A literal Const leaf IS pattern skeleton; a metavariable is not.
    assert lc._in_skeleton(pat, (1,))
    assert not lc._in_skeleton(pat, (0,))
    # A path descending past the end of a leaf is off the pattern.
    assert not lc._in_skeleton(pat, (1, 0))
    assert lc._in_skeleton(pat, ())


def test_confluence_probe_divergent_pair():
    """Two unsound collapses make a real critical pair: the inner
    ``add(A,B) -> A`` fires at path ``(0,)`` inside the outer
    ``mul(add(A,B), C) -> A``'s instance, and its reduct ``mul(A, C)``
    can never rejoin the outer's ``A`` — an honest DIVERGENT row,
    note included."""
    inner = Rewrite("t_inner_collapse", _p("add", "P", "Q"), "P")
    outer = Rewrite(
        "t_outer_collapse", _p("mul", _p("add", "A", "B"), "C"), "A"
    )
    inst = {
        "t_inner_collapse": lc._instance(inner),
        "t_outer_collapse": lc._instance(outer),
    }
    row = lc._confluence_probe(
        inner, outer, inst, [*list(_UNIVERSE), inner, outer]
    )
    assert row is not None
    assert row.pair == ("t_inner_collapse", "t_outer_collapse")
    assert row.base == "t_outer_collapse"
    assert row.join_pair is False
    assert row.join_lib is False
    assert "did not rejoin" in row.note
    assert row.overlap is True


def test_catalogue_no_instance_rules():
    """Rules with no honest instance stay in the universe but carry the
    ``no-instance`` verdict everywhere."""
    veto1, veto2 = _veto_rule("t_v1"), _veto_rule("t_v2")
    cat = lc.catalogue([veto1, veto2, _BY_NAME["silu_expand"]])
    profiles = cat["profiles"]
    assert profiles["t_v1"].verdict == "no-instance"
    assert profiles["t_v1"].instanced is False
    assert cat["instanced"] == ["silu_expand"]
    # No derivability was even attempted on un-instanced rules.
    assert profiles["t_v1"].witness == ()
    table = lc.emit_basis(cat)
    assert table["t_v1"] == {"kind": "no-instance", "derivation": []}
    assert table["t_v2"]["kind"] == "no-instance"


def test_report_no_eq_classes_and_divergence():
    cat = lc.catalogue([_BY_NAME["select_mul"]])
    text = lc._report(cat, [_BY_NAME["select_mul"]])
    assert "(none)" in text
    # A divergent pair is printed with its note.
    inner = Rewrite("t_inner_collapse", _p("add", "P", "Q"), "P")
    outer = Rewrite(
        "t_outer_collapse", _p("mul", _p("add", "A", "B"), "C"), "A"
    )
    uni2 = [inner, outer, _BY_NAME["silu_expand"]]
    cat2 = lc.catalogue(uni2)
    text2 = lc._report(cat2, uni2)
    assert "DIVERGENT" in text2
    assert "did not rejoin" in text2


def test_main_with_layout_flag(monkeypatch, tmp_path):
    """``--with-layout`` selects the layout universe — patched to the
    same tiny one here."""
    monkeypatch.setattr(lc, "ALL_RULES_WITH_LAYOUT", list(_UNIVERSE))
    rc = lc.main(["--with-layout", "--json", str(tmp_path / "c.json")])
    assert rc == 0
    assert json.loads((tmp_path / "c.json").read_text())["n_rules"] == 6


# ---------------------------------------------------------------------------
#  lemma_cert — second pass: enumeration, saturation-only, more edges
# ---------------------------------------------------------------------------


def test_probe_right_factor_linear_enumerated():
    """The documented boundary: ``right_factor_linear`` merges under
    the layout universe but ``certificate()`` emits stubs — the
    ``all_proofs`` enumerator surfaces the real 4-step derivation."""
    from catopt_core.laws import ALL_RULES_WITH_LAYOUT

    row, cert = llc.probe(
        "right_factor_linear", list(ALL_RULES_WITH_LAYOUT)
    )
    assert row.verdict == "enumerated"
    assert cert is not None and cert.n_steps >= 2
    assert set(row.rules_used) <= {
        "linear_to_matmul_t",
        "linear_from_matmul_t",
        "right_factor_matmul",
    }


def test_run_cert_saturation_only_under_tight_fuel(monkeypatch):
    """Merged but no linear derivation within fuel — the honest
    ``saturation-only`` verdict."""
    from catopt_core.laws import ALL_RULES_WITH_LAYOUT

    monkeypatch.setattr(llc, "_PROOF_STEPS", 1)
    monkeypatch.setattr(llc, "_PROOF_FUEL", 200)
    row, cert = llc.probe(
        "right_factor_linear", list(ALL_RULES_WITH_LAYOUT)
    )
    assert row.verdict == "saturation-only"
    assert cert is None
    assert row.note  # the e-graph-dependent stub note survives


def test_probe_no_instance():
    uni = [*list(_UNIVERSE), _veto_rule("t_v")]
    row, cert = llc.probe("t_v", uni)
    assert row.verdict == "no-instance"
    assert cert is None


def test_materialize_with_explicit_instance():
    inst = lc._instance(_BY_NAME["silu_fold"])
    row, cert = llc.materialize(
        _BY_NAME["silu_fold"], list(ALL_RULES), inst=inst
    )
    assert row.verdict == "linear"
    assert cert is not None


def test_materialize_all_skips_certless_rows():
    drift = Rewrite(
        "t_drift",
        _p("add", "A", "B"),
        "A",
        derivation=("comm_add",),
    )
    uni = [*list(_UNIVERSE), drift]
    rows, certs = llc.materialize_all(uni)
    names = {r.name: r.verdict for r in rows}
    assert names["t_drift"] == "gap"
    # A non-linear verdict contributes a row but no certificate.
    assert "t_drift" not in certs
    assert set(certs) <= set(names)


def test_report_prints_nonlinear_notes():
    ghost = Rewrite(
        "t_ghost",
        _p("add", "A", "B"),
        "A",
        derivation=("no_such_rule",),
    )
    row, _ = llc.materialize(ghost, list(ALL_RULES))
    text = llc._report([row], "TINY")
    assert "no_such_rule" in text
    assert "bad-derivation" in text


def test_lemma_cert_main_with_layout(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(llc, "ALL_RULES_WITH_LAYOUT", list(_UNIVERSE))
    rc = llc.main(["--with-layout", "--json", str(tmp_path / "c.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ALL_RULES_WITH_LAYOUT" in out


def test_lemma_cert_main_probe_prints_steps(monkeypatch, capsys):
    monkeypatch.setattr(llc, "ALL_RULES", list(_UNIVERSE))
    rc = llc.main(["--probe", "silu_expand"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "verdict=linear" in out
    assert "silu_fold @" in out


def test_generic_instance_default_attr_and_working_derive():
    # An attr metavar with no 0/1 suffix defaults to 0.
    rule = Rewrite(
        "t_plain_attr",
        _p("transpose", "X", dim0="D", dim1="E"),
        "X",
    )
    lhs, _ = lc._generic_instance(rule)
    assert lhs.attrs == {"dim0": 0, "dim1": 0}
    # A derive hook that produces RHS-only attr bindings feeds the
    # instance — the ``+1`` makes the computation visible.
    derived = Rewrite(
        "t_derived",
        _p("transpose", "X", dim0="D0", dim1="D1"),
        _p("transpose", "X", dim0="E", dim1="D0"),
        derive=lambda bound: {"$attr:E": bound["$attr:D0"] + 1},
    )
    lhs, rhs = lc._generic_instance(derived)
    assert lhs.attrs == {"dim0": 0, "dim1": 1}
    assert rhs.attrs == {"dim0": 1, "dim1": 0}


def test_lemma_cert_main_probe_note(monkeypatch, capsys):
    """A probe whose cert is e-graph-dependent prints the stub note."""
    monkeypatch.setattr(llc, "materialize_all", lambda uni: ([], {}))
    monkeypatch.setattr(llc, "_PROOF_STEPS", 1)
    monkeypatch.setattr(llc, "_PROOF_FUEL", 200)
    rc = llc.main(["--with-layout", "--probe", "right_factor_linear"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "verdict=saturation-only" in out
    assert "note:" in out
