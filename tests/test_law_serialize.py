"""Tests for laws-as-data — ``catopt_core.laws.serialize`` + lemma store.

The lemma-store seam: a law is a 2-cell — pattern pair, side
condition, provenance — and with ``cond`` the whole record is JSON.
This file pins:

* the term codec at its canonical IR home
  (``catopt_core.ir.term_to_data`` / ``term_from_data`` — the same
  scheme ``rulecache`` persists, round-tripped over every shipped
  pattern);
* the ``Rewrite`` record round-trip — ``law_to_data`` /
  ``law_from_data`` — including the exact census of which of the 71
  shipped laws are *full-data* (all 71 now — the last four
  procedural hooks went declarative in stage 0 of plan 0017);
* the honesty contract — a ``serializable: false`` record rebuilds
  its pattern + cond but *not* the dropped hooks: a synthetic
  flagged law fires and mints ``fused(u, dim="ND")`` (the unbound
  attr metavar falls back to its literal name — a visible scar,
  not a silent veto);
* the sqlite ``lemmas`` table — ``store_lemma`` / ``admit_lemma`` /
  ``lemma_rows`` plus the ``--add-lemma`` / ``--admit`` CLI —
  closing the loop: a stored lemma admits into a live ``Rewrite``
  that fires identically to the shipped rule (cond and all).
"""

from __future__ import annotations

import json

import pytest
from catopt_core.egraph import Certificate, EGraph, Rewrite
from catopt_core.egraph.certs import CERT_FORMAT
from catopt_core.ir import (
    Const,
    Op,
    Param,
    TensorType,
    Var,
    term_from_data,
    term_to_data,
)
from catopt_core.laws import ALL_RULES
from catopt_core.laws.serialize import (
    LAW_FORMAT,
    alpha_key,
    law_from_data,
    law_to_data,
    missing_hooks,
)
from catopt_core.laws.tensor import (
    FACTOR_MUL,
    SOFTMAX_FOLD,
)

from catopt_discovery import evidence as le


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _p(name, *shape):
    return Param(name, TensorType(tuple(shape)))


def _json_roundtrip(data):
    return json.loads(json.dumps(data))


# ---------------------------------------------------------------------------
#  The term codec — canonical at the IR layer
# ---------------------------------------------------------------------------


def test_term_codec_roundtrips_every_shipped_pattern():
    for rule in ALL_RULES:
        for side in (rule.lhs, rule.rhs):
            data = _json_roundtrip(term_to_data(side))
            assert term_from_data(data) == side, (rule.name, side)


def test_term_codec_rejects_and_reports():
    with pytest.raises(TypeError):
        term_to_data(42)
    with pytest.raises(ValueError):
        term_from_data({"junk": 1})


# ---------------------------------------------------------------------------
#  alpha_key — the store's identity
# ---------------------------------------------------------------------------


def test_alpha_key_ignores_metavar_names():
    k1 = alpha_key(Op.make("add", "a", "b"), Op.make("add", "b", "a"))
    k2 = alpha_key(Op.make("add", "x", "y"), Op.make("add", "y", "x"))
    assert k1 == k2
    assert k1 != alpha_key(
        Op.make("add", "a", "b"), Op.make("mul", "a", "b")
    )


def test_alpha_key_shared_binding_is_position_sensitive():
    # shared mv map: a,b -> a,b vs a,b -> b,a canonically differ
    k_fwd = alpha_key(Op.make("f", "x", "y"), Op.make("g", "x", "y"))
    k_sw = alpha_key(Op.make("f", "x", "y"), Op.make("g", "y", "x"))
    assert k_fwd != k_sw


def test_alpha_key_canonicalizer_branches():
    """`_attr_canon`/`_canon`: `$attr:` strings, literal and
    unhashable attrs, Const and non-metavar leaves."""
    # $attr:-prefixed attr values serialise under "av" — and stay
    # literal (a bound attr name is not a leaf metavar): different
    # names key differently.
    k = alpha_key(
        Op.make("select", "u", dim="$attr:D", index=0),
        Op.make("select", "u", dim="$attr:E", index=0),
    )
    k2 = alpha_key(
        Op.make("select", "w", dim="$attr:F", index=0),
        Op.make("select", "w", dim="$attr:G", index=0),
    )
    assert k != k2
    # ...but the LEAF metavars inside still abstract — same attr
    # metavar, different leaf names: equal keys.
    assert alpha_key(
        Op.make("select", "u", dim="$attr:D", index=0),
        Op.make("select", "w", dim="$attr:D", index=0),
    ) == alpha_key(
        Op.make("select", "p", dim="$attr:D", index=0),
        Op.make("select", "q", dim="$attr:D", index=0),
    )
    # literal (hashable) vs unhashable attrs fall through lit/repr.
    k3 = alpha_key(
        Op.make("f", "x", n=2, dims=(1, 2)),
        Op.make("f", "x", n=2, dims=[1, 2]),
    )
    assert isinstance(k3, tuple) and len(k3) == 2
    # Const leaves stay literal ("c"), Var leaves hit the repr fallback.
    k4 = alpha_key(
        Op.make("pow", "x", Const(2)),
        Op.make("pow", "x", Const(2)),
    )
    k5 = alpha_key(
        Op.make("add", Var("x", TensorType((2,))), "y"),
        Op.make("add", Var("x", TensorType((2,))), "y"),
    )
    assert k4 != alpha_key(
        Op.make("pow", "x", Const(3)),
        Op.make("pow", "x", Const(3)),
    )
    assert k5 == k5  # self-evident; the repr fallback ran


# ---------------------------------------------------------------------------
#  missing_hooks — the serializability census
# ---------------------------------------------------------------------------


def test_serializability_census_of_shipped_library():
    """Pin the honest partition of the 71 shipped laws."""
    full, need_derive, need_check = [], [], []
    for rule in ALL_RULES:
        missing = missing_hooks(rule)
        if not missing:
            full.append(rule.name)
        elif missing == ("derive",):
            need_derive.append(rule.name)
        else:
            need_check.append((rule.name, missing))
    assert len(ALL_RULES) == 71
    # the whole shipped library is full-data now: the four stragglers
    # went declarative in stage 0 of plan 0017 — glu_fold's parity on
    # a dim-mod predicate, the rms pair's trailing block on the
    # tail-block spec, and gqa_absorb_repeat's repeat-chain /
    # attr-eq-attr / repeat-heads triple — and the ten promoted
    # discovery laws carry pure-data cond/dspec/derivation
    # (project/retros/promoted-laws.md)
    assert len(full) == 71
    assert need_derive == []
    assert need_check == []


def test_missing_hooks_detects_procedural_check_under_cond():
    """A rule whose cond+check conjoin is NOT full data."""
    r = Rewrite(
        name="hybrid",
        lhs=Op.make("add", "a", "b"),
        rhs="a",
        cond=("scalar", "a"),
        check=lambda bound: True,
    )
    assert missing_hooks(r) == ("check",)
    # a bare check with no cond is likewise not data
    r2 = Rewrite(
        name="procedural",
        lhs="a",
        rhs="a",
        check=lambda bound: True,
    )
    assert missing_hooks(r2) == ("check",)
    # and the cond-only twin is clean
    r3 = Rewrite(
        name="declarative",
        lhs=Op.make("add", "a", "b"),
        rhs="a",
        cond=("scalar", "a"),
    )
    assert missing_hooks(r3) == ()
    # same for derive: a dspec is data, a spec+code composite is not
    r4 = Rewrite(
        name="spec",
        lhs=Op.make("add", "a", "b"),
        rhs="a",
        dspec={"D": ("attr0", "RD")},
    )
    assert missing_hooks(r4) == ()
    r5 = Rewrite(
        name="spec_plus_code",
        lhs=Op.make("add", "a", "b"),
        rhs="a",
        dspec={"D": ("attr0", "RD")},
        derive=lambda bound: {"$attr:E": 0},
    )
    assert missing_hooks(r5) == ("derive",)
    # a bare procedural derive flags too
    r6 = Rewrite(
        name="proc_derive",
        lhs="a",
        rhs="a",
        derive=lambda bound: {},
    )
    assert missing_hooks(r6) == ("derive",)


# ---------------------------------------------------------------------------
#  law_to_data / law_from_data — the record round-trip
# ---------------------------------------------------------------------------


def test_law_record_roundtrip_full_data():
    for rule in ALL_RULES:
        if missing_hooks(rule):
            continue
        data = _json_roundtrip(law_to_data(rule))
        rebuilt = law_from_data(data)
        assert law_to_data(rebuilt) == law_to_data(rule), rule.name


def test_law_record_roundtrip_rewrite_equality_unguarded():
    """Hook-free laws rebuild to an ``==``-equal Rewrite."""
    n = 0
    for rule in ALL_RULES:
        if rule.check is None and rule.derive is None:
            rebuilt = law_from_data(_json_roundtrip(law_to_data(rule)))
            assert rebuilt == rule, rule.name
            n += 1
    assert n == 25


def _flagged_rule():
    """A rule data cannot fully carry — a cond + procedural check +
    procedural derive composite.  No shipped law is flagged anymore
    (the census is clean), so the honesty path gets a synthetic
    stand-in."""
    return Rewrite(
        name="flagged_fused",
        lhs=Op.make("mul", "u", Op.make("rsqrt", "u")),
        rhs=Op.make("fused", "u", dim="ND"),
        cond=("rank", "u", ">=", 1),
        check=lambda bound: True,
        derive=lambda bound: {"$attr:ND": 1},
    )


def test_law_record_flagged_hooks_drop_on_rebuild():
    """A check/derive law stores pattern+cond, flagged honestly."""
    rule = _flagged_rule()
    data = law_to_data(rule)
    assert data["serializable"] is False
    assert data["missing_hooks"] == ["check", "derive"]
    rebuilt = law_from_data(_json_roundtrip(data))
    assert rebuilt.derive is None and rebuilt.dspec is None
    # the cond DID travel: the rebuilt rule still guards
    assert rebuilt.cond == rule.cond
    assert rebuilt.check is not None
    # and the record marks the rebuild clean — it has no hooks to drop
    assert law_to_data(rebuilt)["serializable"] is True


def test_dspec_law_record_carries_the_derive():
    """A spec-derived law serializes *whole* — derive and all."""
    data = law_to_data(SOFTMAX_FOLD)
    assert data["serializable"] is True
    assert data["dspec"] == {"SD": ["attr0", "RD"]}
    rebuilt = law_from_data(_json_roundtrip(data))
    assert rebuilt.dspec == SOFTMAX_FOLD.dspec
    assert rebuilt.derive({"$attr:RD": (-1,)}) == {"$attr:SD": -1}


def test_law_record_preserves_provenance_fields():
    data = law_to_data(FACTOR_MUL)
    assert data["derivation"] == list(FACTOR_MUL.derivation)
    assert data["tags"] == sorted(FACTOR_MUL.tags)
    assert data["law"] == FACTOR_MUL.law
    rebuilt = law_from_data(data)
    assert rebuilt.kind == FACTOR_MUL.kind
    assert rebuilt.error_bound == FACTOR_MUL.error_bound
    assert rebuilt.bound_norm == FACTOR_MUL.bound_norm


def test_law_from_data_rejects_bad_version():
    data = law_to_data(FACTOR_MUL)
    data["version"] = LAW_FORMAT + 1
    with pytest.raises(ValueError, match="version"):
        law_from_data(data)


def test_law_record_cert_field_is_optional_v2():
    """``cert`` rides v2: null by default, ignored on rebuild, absent
    in older records — no format bump, no reader rejection."""
    data = law_to_data(FACTOR_MUL)
    assert data["cert"] is None
    cert = Certificate(
        src=FACTOR_MUL.lhs,
        dst=FACTOR_MUL.rhs,
        root_eid=None,
        steps=[],
        rules={},
    )
    with_cert = law_to_data(FACTOR_MUL, cert=cert)
    assert with_cert["cert"]["version"] == CERT_FORMAT
    assert with_cert["cert"]["replayable"] is True
    # the cert is record-level provenance — it does not fold into the
    # reconstructed Rewrite (rebuilt rules aren't == under their
    # folded check closures, so compare the re-serialized records)
    assert law_to_data(law_from_data(with_cert)) == law_to_data(
        law_from_data(data)
    )
    # a pre-cert v2 record (no "cert" key at all) still loads
    del with_cert["cert"]
    assert law_from_data(with_cert).name == FACTOR_MUL.name


# ---------------------------------------------------------------------------
#  Reconstructed laws fire — the proof the data is enough
# ---------------------------------------------------------------------------


def _factor_src():
    x, w = _v("x", 5, 4), _p("W", 4, 3)
    a, b = _p("a", 4, 3), _p("b", 4, 3)
    return (
        Op.make(
            "add",
            Op.make("matmul", x, Op.make("matmul", w, a)),
            Op.make("matmul", x, Op.make("matmul", w, b)),
        ),
        Op.make(
            "matmul", x, Op.make("matmul", w, Op.make("add", a, b))
        ),
    )


def _factor_decline():
    x, a, b = _v("x2", 16), _p("a2", 16), _p("b2", 16, 16)
    return (
        Op.make(
            "add",
            Op.make("matmul", x, a),
            Op.make("matmul", x, b),
        ),
        Op.make("matmul", x, Op.make("add", a, b)),
    )


def _fires(rule, src, want):
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    return eg.find(root) == eg.find(eg.add_term(want))


def test_rebuilt_cond_law_fires_and_declines_identically():
    rebuilt = law_from_data(_json_roundtrip(law_to_data(FACTOR_MUL)))
    src, merged = _factor_src()
    bad, bad_merged = _factor_decline()
    for rule in (FACTOR_MUL, rebuilt):
        assert _fires(rule, src, merged)
        assert not _fires(rule, bad, bad_merged)


def test_rebuilt_dspec_law_fires_identically():
    """``softmax_fold`` round-trips *whole* now — the rebuilt rule
    mints the derived dim, not the ``"SD"`` scar."""
    u = _v("u", 4, 8)
    src = Op.make(
        "div",
        Op.make("exp", u),
        Op.make("sum", Op.make("exp", u), dim=(-1,), keepdim=True),
    )
    want = Op.make("softmax", u, dim=-1)
    rebuilt = law_from_data(_json_roundtrip(law_to_data(SOFTMAX_FOLD)))
    assert _fires(SOFTMAX_FOLD, src, want)
    assert _fires(rebuilt, src, want)


def test_rebuilt_flagged_law_shows_the_honest_scar():
    """A flagged law rebuilt without its hooks fires — and mints the
    unbound attr metavar *literally* (``dim="ND"``).  The flag says
    why this cannot be trusted; the scar shows it."""
    rule = _flagged_rule()
    u = _v("u", 4, 8)
    src = Op.make("mul", u, Op.make("rsqrt", u))
    scar = Op.make("fused", u, dim="ND")
    rebuilt = law_from_data(_json_roundtrip(law_to_data(rule)))
    assert _fires(rebuilt, src, scar)


# ---------------------------------------------------------------------------
#  The lemmas table — store, admit, fire
# ---------------------------------------------------------------------------


def _conn(tmp_path):
    return le.connect(str(tmp_path / "laws.db"))


def test_lemmas_table_write_read_admit(tmp_path):
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, FACTOR_MUL, corpus_hash="abc")
        rows = le.lemma_rows(conn)
        assert len(rows) == 1
        assert rows[0]["alpha_key"] == key
        assert rows[0]["name"] == "factor_matmul"
        assert json.loads(rows[0]["law_json"])["serializable"] is True
        assert json.loads(rows[0]["derivation_json"]) == list(
            FACTOR_MUL.derivation
        )
        got = le.admit_lemma(conn, key)
        assert got is not None
        rule, data = got
        assert data["serializable"] is True
        src, merged = _factor_src()
        assert _fires(rule, src, merged)
        # and the stored cond still vetoes the rank-1 counterexample
        bad, bad_merged = _factor_decline()
        assert not _fires(rule, bad, bad_merged)
    finally:
        conn.close()


def test_admit_lemma_unknown_key_returns_none(tmp_path):
    conn = _conn(tmp_path)
    try:
        assert le.admit_lemma(conn, "nope") is None
    finally:
        conn.close()


def test_admitted_flagged_rule_is_flagged(tmp_path):
    conn = _conn(tmp_path)
    try:
        rule = _flagged_rule()
        key = le.store_lemma(conn, rule)
        admitted, data = le.admit_lemma(conn, key)
        assert data["serializable"] is False
        assert data["missing_hooks"] == ["check", "derive"]
        assert admitted.derive is None and admitted.cond == rule.cond
    finally:
        conn.close()


def test_admitted_softmax_fold_is_full_data(tmp_path):
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, SOFTMAX_FOLD)
        rule, data = le.admit_lemma(conn, key)
        assert data["serializable"] is True
        assert rule.dspec == SOFTMAX_FOLD.dspec
        # and the admitted rule mints the derived dim, not the scar
        u = _v("u", 4, 8)
        src = Op.make(
            "div",
            Op.make("exp", u),
            Op.make("sum", Op.make("exp", u), dim=(-1,), keepdim=True),
        )
        assert _fires(rule, src, Op.make("softmax", u, dim=-1))
    finally:
        conn.close()


def test_lemmas_table_carries_replayable_certificate(tmp_path):
    """``store_lemma`` materializes the recorded derivation;
    ``stored_certificate`` replays it strictly off the record."""
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, FACTOR_MUL)
        stored = json.loads(
            conn.execute(
                "SELECT law_json FROM lemmas WHERE alpha_key = ?",
                (key,),
            ).fetchone()["law_json"]
        )
        assert stored["derivation"] == list(FACTOR_MUL.derivation)
        assert stored["cert"] is not None
        cert = le.stored_certificate(stored)
        assert cert.replayable
        assert cert.n_steps == 1
        assert set(cert.rules_used) <= set(FACTOR_MUL.derivation)
        # and admit hands the same record back — cert intact
        _rule, record = le.admit_lemma(conn, key)
        assert record["cert"] == stored["cert"]
        assert le.stored_certificate(record) is not None
    finally:
        conn.close()


def test_store_lemma_cert_knob(tmp_path):
    """``cert=`` is explicit: ``None`` suppresses materialization, a
    Certificate embeds as given."""
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, FACTOR_MUL, cert=None)
        _rule, data = le.admit_lemma(conn, key)
        assert data["cert"] is None
        empty = Certificate(
            src=FACTOR_MUL.lhs,
            dst=FACTOR_MUL.rhs,
            root_eid=None,
            steps=[],
            rules={},
        )
        key = le.store_lemma(conn, FACTOR_MUL, cert=empty)
        _rule, data = le.admit_lemma(conn, key)
        assert data["cert"]["version"] == CERT_FORMAT
        assert data["cert"]["steps"] == []
    finally:
        conn.close()


def test_lemma_cli_store_then_admit(tmp_path, capsys):
    db = str(tmp_path / "laws.db")
    assert le.main(["--report", db, "--add-lemma", "assoc_matmul"]) == 0
    out = capsys.readouterr().out
    assert "stored assoc_matmul  [full-data]" in out
    key = next(
        line.split("=", 1)[1].strip()
        for line in out.splitlines()
        if "alpha_key" in line
    )
    assert le.main(["--report", db, "--admit", key]) == 0
    out = capsys.readouterr().out
    assert "admitted assoc_matmul  [full-data]" in out
    assert "cert: none recorded" in out


def test_lemma_cli_cert_roundtrip(tmp_path, capsys):
    """A derivation-carrying law stores + admits with its cert —
    ``--admit`` replays it strictly."""
    db = str(tmp_path / "laws.db")
    assert (
        le.main(["--report", db, "--add-lemma", "factor_matmul"]) == 0
    )
    out = capsys.readouterr().out
    assert (
        "cert: 1-step derivation ['distribute_matmul_over_add']" in out
    )
    key = next(
        line.split("=", 1)[1].strip()
        for line in out.splitlines()
        if "alpha_key" in line
    )
    assert le.main(["--report", db, "--admit", key]) == 0
    out = capsys.readouterr().out
    assert "cert: 1-step ['distribute_matmul_over_add']" in out
    assert "replayed strict" in out


def test_lemma_cli_reports_missing_hooks(tmp_path, capsys):
    """Every shipped law is full-data now, so the flagged report comes
    from a synthetic rule stored directly — ``--admit`` prints its
    missing hooks."""
    db = str(tmp_path / "laws.db")
    conn = le.connect(db)
    try:
        key = le.store_lemma(conn, _flagged_rule())
    finally:
        conn.close()
    assert le.main(["--report", db, "--admit", key]) == 0
    out = capsys.readouterr().out
    assert "missing hooks: check, derive" in out


def test_lemma_cli_errors(tmp_path, capsys):
    db = str(tmp_path / "laws.db")
    assert le.main(["--report", db, "--add-lemma", "nosuchlaw"]) == 1
    assert le.main(["--report", db, "--admit", "nope"]) == 1


def test_report_counts_lemmas(tmp_path):
    conn = _conn(tmp_path)
    try:
        le.store_lemma(conn, FACTOR_MUL)
        report = le.render_report(conn, "x.db")
        assert "1 lemmas" in report
    finally:
        conn.close()
