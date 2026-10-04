"""Tests for laws-as-data — ``catopt_core.laws.serialize`` + lemma store.

The lemma-store seam: a law is a 2-cell — pattern pair, side
condition, provenance — and with ``cond`` the whole record is JSON.
This file pins:

* the term codec at its canonical IR home
  (``catopt_core.ir.term_to_data`` / ``term_from_data`` — the same
  scheme ``rulecache`` persists, round-tripped over every shipped
  pattern);
* the ``Rewrite`` record round-trip — ``law_to_data`` /
  ``law_from_data`` — including the exact census of which of the 61
  shipped laws are *full-data* (43) vs pattern(+cond) with a
  ``check`` (2, plus 2 ``check``+``derive``) or ``derive`` (14)
  remainder;
* the honesty contract — a ``serializable: false`` record rebuilds
  its pattern + cond but *not* the dropped hooks: the reconstructed
  ``softmax_fold`` fires and mints ``softmax(u, dim="SD")`` (the
  unbound attr metavar falls back to its literal name — a visible
  scar, not a silent veto);
* the sqlite ``lemmas`` table — ``store_lemma`` / ``admit_lemma`` /
  ``lemma_rows`` plus the ``--add-lemma`` / ``--admit`` CLI —
  closing the loop: a stored lemma admits into a live ``Rewrite``
  that fires identically to the shipped rule (cond and all).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import (
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
from catopt_core.laws.tensor import FACTOR_MUL, SOFTMAX_FOLD

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import law_evidence as le  # noqa: E402


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


# ---------------------------------------------------------------------------
#  missing_hooks — the serializability census
# ---------------------------------------------------------------------------


def test_serializability_census_of_shipped_library():
    """Pin the honest partition of the 61 shipped laws."""
    full, need_derive, need_check = [], [], []
    for rule in ALL_RULES:
        missing = missing_hooks(rule)
        if not missing:
            full.append(rule.name)
        elif missing == ("derive",):
            need_derive.append(rule.name)
        else:
            need_check.append((rule.name, missing))
    assert len(ALL_RULES) == 61
    assert len(full) == 43
    assert len(need_derive) == 14
    assert "softmax_fold" in need_derive
    assert "qkv_fuse_asym" in need_derive
    # glu_fold's split-axis parity, the rms pair's normalized-shape +
    # derive hooks, and gqa_absorb_repeat's repeat-chain side
    # conditions are still procedural
    assert need_check == [
        ("glu_fold", ("check",)),
        ("rms_norm_fold", ("check", "derive")),
        ("rms_norm_fold_nogain", ("check", "derive")),
        ("gqa_absorb_repeat", ("check",)),
    ]


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


def test_law_record_flagged_hooks_drop_on_rebuild():
    """A derive/check law stores pattern+cond, flagged honestly."""
    data = law_to_data(SOFTMAX_FOLD)
    assert data["serializable"] is False
    assert data["missing_hooks"] == ["derive"]
    rebuilt = law_from_data(_json_roundtrip(data))
    assert rebuilt.derive is None
    # the cond DID travel: the rebuilt rule still guards
    assert rebuilt.cond == SOFTMAX_FOLD.cond
    assert rebuilt.check is not None
    # and the record marks the rebuild clean — it has no hooks to drop
    assert law_to_data(rebuilt)["serializable"] is True


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


def test_rebuilt_derive_law_shows_the_honest_scar():
    """``softmax_fold`` without its derive fires — and mints the
    unbound attr metavar *literally* (``dim="SD"``).  The flag says
    why this cannot be trusted; the scar shows it."""
    u = _v("u", 4, 8)
    src = Op.make(
        "div",
        Op.make("exp", u),
        Op.make("sum", Op.make("exp", u), dim=(-1,), keepdim=True),
    )
    rebuilt = law_from_data(_json_roundtrip(law_to_data(SOFTMAX_FOLD)))
    scar = Op.make("softmax", u, dim="SD")
    assert _fires(rebuilt, src, scar)
    # the shipped rule mints the derived dim instead
    assert _fires(SOFTMAX_FOLD, src, Op.make("softmax", u, dim=-1))


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


def test_admitted_softmax_fold_is_flagged(tmp_path):
    conn = _conn(tmp_path)
    try:
        key = le.store_lemma(conn, SOFTMAX_FOLD)
        rule, data = le.admit_lemma(conn, key)
        assert data["serializable"] is False
        assert data["missing_hooks"] == ["derive"]
        assert rule.derive is None and rule.cond == SOFTMAX_FOLD.cond
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


def test_lemma_cli_reports_missing_hooks(tmp_path, capsys):
    db = str(tmp_path / "laws.db")
    assert le.main(["--report", db, "--add-lemma", "softmax_fold"]) == 0
    out = capsys.readouterr().out
    assert "missing hooks: derive" in out


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
