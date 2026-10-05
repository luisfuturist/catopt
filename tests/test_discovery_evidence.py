"""Tests for ``catopt_discovery.evidence`` — the sqlite verdict store.

The store is a content-addressed cache: a verdict row is valid only
under the ``(corpus_hash, rules_hash, code_rev)`` scope it was
measured in, a re-measurement under a new ``run_id`` records "seen
again", and a SHIP candidate can be *stored* as a lemma row and
``--admit``ted back into a live ``Rewrite`` — derivation-carrying
laws materialize a replayable certificate into the record.

The ``lemmas`` table is also the *declared-object* store (ADR 0004,
plan 0017 stage 1): ``store_object`` / ``stored_object`` /
``admit_object`` generalise the lemma seam, the record's ``"kind"``
field marking the declaration's provenance — ``"law"`` (the
default, and every pre-object record), ``"abstraction"`` or
``"bridge"``.  The first inhabitants are laws-as-objects: a shipped
law stored *as* an object record admits and fires identically.
"""

import json

import pytest
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_core.laws.serialize import (
    alpha_key,
    object_from_data,
    object_kind,
    object_to_data,
)
from catopt_discovery import evidence as ev
from catopt_discovery.pipeline import Evidence, Proposal

_BY_NAME = {r.name: r for r in ALL_RULES}


def _meta(**kw):
    """Return a scope dict for ``record_run``."""
    meta = {
        "corpus_hash": ev.corpus_hash(
            ["bench:a:(4,4)", "model:b:(8,)"]
        ),
        "rules_hash": ev.rules_hash(["k1", "k2"]),
        "code_rev": "abc123",
        "run_id": "run0",
        "holdout": "",
        "ts": "2024-01-01T00:00:00+00:00",
    }
    meta.update(kw)
    return meta


def _row(name, key, *, ship=True):
    """Build a ``verdict_row`` dict from a real ``Evidence`` value."""
    prop = Proposal(
        name=name, lhs=None, rhs=None, family="fam", sources=("s",)
    )
    evid = Evidence(
        proposal=prop,
        num_true=True,
        derivable=False,
        census_sites=3,
        relaxed=4,
        matches=2,
        example="add(x, y)",
        fires=5,
        fires_typed=5,
        fire_cases=("m1", "m2"),
        changed=1,
        paid=1 if ship else 0,
        cost_drop=0.25,
        relation="new",
    )
    return ev.verdict_row(key, evid, "add(x, y)", "add(y, x)")


def test_connect_creates_schema(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    assert {"candidates", "verdicts", "lemmas"} <= tables
    conn.close()


def test_run_id_and_timestamp_formats():
    rid = ev.new_run_id()
    assert isinstance(rid, str) and len(rid) == 12
    assert ev.new_run_id() != rid
    ts = ev.now()
    assert "T" in ts and ("+" in ts or ts.endswith("Z"))


def test_hashes_sorted_deterministic_content_keyed():
    assert ev.corpus_hash(["a", "b"]) == ev.corpus_hash(["b", "a"])
    assert ev.corpus_hash(["a", "b"]) != ev.corpus_hash(["a", "c"])
    assert ev.rules_hash(["k1", "k2"]) == ev.rules_hash(["k2", "k1"])
    h = ev.rules_hash(["k1"])
    assert isinstance(h, str) and len(h) == 64


def test_code_rev_real_repo_and_unknown(tmp_path):
    rev = ev.code_rev()
    assert rev and rev != "unknown"
    # A directory git cannot answer for yields "unknown" — written
    # but never served as a cache hit.
    assert ev.code_rev(root=tmp_path) == "unknown"
    assert ev._git(tmp_path, "rev-parse", "HEAD") == ""
    # An unusable cwd makes the subprocess itself fail → "".
    assert ev._git(tmp_path / "missing", "status") == ""


def test_hash_untracked_paths(tmp_path):
    import hashlib

    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f.txt").write_bytes(b"abc")
    h_dir = hashlib.sha256()
    ev._hash_untracked(tmp_path / "d", h_dir)
    h_file = hashlib.sha256()
    ev._hash_untracked(tmp_path / "d" / "f.txt", h_file)
    # Both paths mix the same file contents in.
    assert (
        h_dir.hexdigest()
        == h_file.hexdigest()
        == hashlib.sha256(b"abc").hexdigest()
    )
    # A missing path mixes nothing.
    h_none = hashlib.sha256()
    ev._hash_untracked(tmp_path / "nope", h_none)
    assert h_none.hexdigest() == hashlib.sha256().hexdigest()
    # An unreadable file is skipped, not fatal.
    (tmp_path / "d" / "locked").write_bytes(b"x")
    (tmp_path / "d" / "locked").chmod(0o000)
    h_locked = hashlib.sha256()
    ev._hash_untracked(tmp_path / "d" / "locked", h_locked)
    assert h_locked.hexdigest() == hashlib.sha256().hexdigest()


def test_code_rev_dirty_worktree(tmp_path):
    import subprocess

    root = tmp_path / "repo"
    (root / "packages").mkdir(parents=True)
    (root / "packages" / "a.py").write_text("x = 1\n")

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, check=True
        )

    git("init")
    git("add", "-A")
    git(
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "-m",
        "init",
    )
    clean = ev.code_rev(root=root)
    assert clean and clean != "unknown" and "+" not in clean
    # A tracked edit under a code path extends the rev with a hash.
    (root / "packages" / "a.py").write_text("x = 2\n")
    dirty = ev.code_rev(root=root)
    assert dirty.startswith(clean + "+")
    # Untracked files under code paths are hashed by content too.
    (root / "packages" / "b.py").write_text("y = 1\n")
    dirtier = ev.code_rev(root=root)
    assert dirtier.startswith(clean + "+")
    assert dirtier != dirty


def test_verdict_row_serialization(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    row = _row("law_a", "K1")
    assert row["alpha_key"] == "K1"
    assert row["numeric_true"] == 1
    assert row["drop_pct"] == 25.0  # cost_drop stored in percent
    assert row["verdict"] == "SHIP"
    assert json.loads(row["fire_cases_json"]) == ["m1", "m2"]
    pj = json.loads(row["proposal_json"])
    assert pj["name"] == "law_a" and pj["sources"] == ["s"]

    nay = _row("law_b", "K2", ship=False)
    assert nay["verdict"].startswith("no:")
    conn.close()


def test_numeric_true_none_stays_null(tmp_path):
    prop = Proposal(name="p", lhs=None, rhs=None, family="f")
    evid = Evidence(proposal=prop, num_true=None, fires=1)
    row = ev.verdict_row("K", evid, "l", "r")
    assert row["numeric_true"] is None


def test_record_run_and_latest_verdicts(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    meta = _meta(run_id="r1")
    ev.record_run(
        conn, meta, [_row("law_a", "K1"), _row("law_b", "K2")]
    )
    hits = ev.latest_verdicts(
        conn, meta["corpus_hash"], meta["rules_hash"], "abc123"
    )
    assert set(hits) == {"K1", "K2"}
    assert hits["K1"]["verdict"] == "SHIP"
    assert hits["K1"]["fires"] == 5
    assert hits["K1"]["run_id"] == "r1"
    conn.close()


def test_latest_verdicts_scope_keys_are_strict(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    meta = _meta(run_id="r1")
    ev.record_run(conn, meta, [_row("law_a", "K1")])
    ch, rh = meta["corpus_hash"], meta["rules_hash"]
    assert ev.latest_verdicts(conn, "other", rh, "abc123") == {}
    assert ev.latest_verdicts(conn, ch, "other", "abc123") == {}
    assert ev.latest_verdicts(conn, ch, rh, "zzz") == {}
    # "unknown" revisions never vouch for the verification code.
    assert ev.latest_verdicts(conn, ch, rh, "unknown") == {}
    conn.close()


def test_record_run_newest_wins_replace_and_upsert(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    m1 = _meta(run_id="r1", ts="2024-01-01T00:00:00+00:00")
    m2 = _meta(run_id="r2", ts="2024-01-02T00:00:00+00:00")
    ev.record_run(conn, m1, [_row("law_a", "K1", ship=False)])
    ev.record_run(conn, m2, [_row("law_a", "K1", ship=True)])
    hits = ev.latest_verdicts(
        conn, m1["corpus_hash"], m1["rules_hash"], "abc123"
    )
    assert hits["K1"]["verdict"] == "SHIP"  # newest ts wins
    n = conn.execute("SELECT count(*) FROM verdicts").fetchone()[0]
    assert n == 2  # both runs recorded (history counts them)
    # Same run_id re-recorded → INSERT OR REPLACE, not a duplicate.
    ev.record_run(conn, m2, [_row("law_a", "K1", ship=True)])
    n = conn.execute("SELECT count(*) FROM verdicts").fetchone()[0]
    assert n == 2
    # Candidate upsert refreshes last-seen name/family/proposal.
    ev.record_run(conn, m2, [_row("renamed", "K1", ship=True)])
    cand = conn.execute(
        "SELECT name FROM candidates WHERE alpha_key = ?", ("K1",)
    ).fetchone()
    assert cand["name"] == "renamed"
    conn.close()


def test_store_and_admit_lemma_roundtrip(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    rule = _BY_NAME["id_add"]
    key = ev.store_lemma(conn, rule, corpus_hash="ctx")
    assert key == repr(alpha_key(rule.lhs, rule.rhs))
    got = ev.admit_lemma(conn, key)
    assert got is not None
    rebuilt, data = got
    assert rebuilt.name == "id_add"
    assert data["serializable"] is True
    assert data["missing_hooks"] == []
    # An axiom has no derivation → cert is honestly null.
    assert data["cert"] is None
    assert ev.stored_certificate(data) is None
    row = conn.execute(
        "SELECT corpus_hash FROM lemmas WHERE alpha_key = ?", (key,)
    ).fetchone()
    assert row["corpus_hash"] == "ctx"
    assert ev.admit_lemma(conn, "no-such-key") is None
    conn.close()


def test_store_lemma_materializes_derivation_cert(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    rule = _BY_NAME["silu_fold"]
    assert rule.derivation == ("silu_expand",)
    key = ev.store_lemma(conn, rule)
    _rebuilt, data = ev.admit_lemma(conn, key)
    cert = ev.stored_certificate(data)
    assert cert is not None
    assert list(cert.rules_used) == ["silu_expand"]
    assert cert.n_steps == 1
    # An explicit rules list resolves the step names instead of the
    # shipped ALL_RULES default.
    cert2 = ev.stored_certificate(data, rules=[_BY_NAME["silu_expand"]])
    assert cert2 is not None and cert2.n_steps == 1
    conn.close()


def test_store_lemma_explicit_cert_none(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    # An explicit cert=None overrides materialization: no derivation
    # is stored even though the rule records one.
    key = ev.store_lemma(conn, _BY_NAME["square_to_pow"], cert=None)
    _rebuilt, data = ev.admit_lemma(conn, key)
    assert data["cert"] is None
    assert ev.stored_certificate(data) is None
    conn.close()


def test_lemma_rows_newest_first(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    ev.store_lemma(conn, _BY_NAME["id_add"])
    ev.store_lemma(conn, _BY_NAME["sub_to_add"])
    rows = ev.lemma_rows(conn)
    assert {r["name"] for r in rows} == {"id_add", "sub_to_add"}
    assert rows[0]["name"] == "sub_to_add"
    assert json.loads(rows[0]["law_json"])["serializable"] is True
    conn.close()


def _store_with_flip(tmp_path):
    """Two candidates; ``flips_law`` ships then stops shipping."""
    conn = ev.connect(str(tmp_path / "s.db"))
    m1 = _meta(run_id="r1", ts="2024-01-01T00:00:00+00:00")
    m2 = _meta(run_id="r2", ts="2024-01-02T00:00:00+00:00")
    ev.record_run(
        conn,
        m1,
        [
            _row("flips_law", "K1", ship=True),
            _row("steady_ship", "K2", ship=True),
        ],
    )
    ev.record_run(conn, m2, [_row("flips_law", "K1", ship=False)])
    return conn


def test_render_report_counts_and_tables(tmp_path):
    conn = _store_with_flip(tmp_path)
    out = ev.render_report(conn, "db")
    assert "2 candidates" in out
    assert "3 verdict rows" in out
    assert "flips_law" in out and "steady_ship" in out
    ships = ev.render_report(conn, "db", ships_only=True)
    assert "steady_ship" in ships
    # flips_law ever shipped but its latest verdict is not SHIP.
    line = next(ln for ln in ships.splitlines() if "flips_law" in ln)
    assert line.rstrip().endswith("*")
    conn.close()


def test_render_report_ships_only_skips_never_shipped(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    ev.record_run(
        conn,
        _meta(run_id="r1"),
        [
            _row("shipped", "K1", ship=True),
            _row("never", "K2", ship=False),
        ],
    )
    ships = ev.render_report(conn, "db", ships_only=True)
    assert "shipped" in ships
    assert "never" not in ships
    conn.close()


def test_render_report_flips_and_history(tmp_path):
    conn = _store_with_flip(tmp_path)
    flips = ev.render_report(conn, "db", flips=True)
    assert "flips_law" in flips
    assert "SHIP" in flips and "no:" in flips
    hist = ev.render_report(conn, "db", history="steady")
    assert "steady_ship" in hist
    assert "flips_law" not in hist
    assert "(none)" in ev.render_report(conn, "db", history="zzz")
    conn.close()
    # A stable store reports no flips.
    conn2 = ev.connect(str(tmp_path / "t.db"))
    ev.record_run(conn2, _meta(run_id="r1"), [_row("a", "K")])
    assert "verdicts stable" in ev.render_report(conn2, "t", flips=True)
    conn2.close()


def test_main_report_missing_store(tmp_path, capsys):
    rc = ev.main(["--report", str(tmp_path / "none.db")])
    assert rc == 1
    assert "no evidence store" in capsys.readouterr().out


def test_main_report_modes(tmp_path, capsys):
    conn = _store_with_flip(tmp_path)
    conn.close()
    db = str(tmp_path / "s.db")
    for argv in (
        ["--report", db],
        ["--report", db, "--ships"],
        ["--report", db, "--flips"],
        ["--report", db, "--history", "law"],
    ):
        assert ev.main(argv) == 0
    out = capsys.readouterr().out
    assert "law_evidence" in out


def test_main_add_and_admit_lemma(tmp_path, capsys):
    db = str(tmp_path / "s.db")
    assert ev.main(["--report", db, "--add-lemma", "id_add"]) == 0
    out = capsys.readouterr().out
    assert "stored id_add" in out and "full-data" in out
    key = repr(
        alpha_key(_BY_NAME["id_add"].lhs, _BY_NAME["id_add"].rhs)
    )
    assert ev.main(["--report", db, "--admit", key]) == 0
    out = capsys.readouterr().out
    assert "admitted id_add" in out and "cert: none" in out
    assert ev.main(["--report", db, "--add-lemma", "no_such"]) == 1
    assert ev.main(["--report", db, "--admit", "bogus"]) == 1
    out = capsys.readouterr().out
    assert "no shipped law" in out and "no lemma stored" in out


def test_main_add_lemma_reports_cert(tmp_path, capsys):
    db = str(tmp_path / "s.db")
    assert ev.main(["--report", db, "--add-lemma", "silu_fold"]) == 0
    out = capsys.readouterr().out
    assert "stored silu_fold" in out
    assert "1-step derivation" in out and "silu_expand" in out


def test_main_admit_replays_cert_strict(tmp_path, capsys):
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_lemma(conn, _BY_NAME["silu_fold"])
    conn.close()
    assert (
        ev.main(["--report", str(tmp_path / "s.db"), "--admit", key])
        == 0
    )
    out = capsys.readouterr().out
    assert "replayed strict" in out


def test_main_admit_strict_replay_failure(tmp_path, capsys):
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_lemma(conn, _BY_NAME["silu_fold"])
    row = conn.execute(
        "SELECT law_json FROM lemmas WHERE alpha_key = ?", (key,)
    ).fetchone()
    data = json.loads(row["law_json"])
    # Corrupt the stored derivation: strict replay must fail the
    # admit, not wave it through.
    data["cert"]["steps"][0]["rule"] = "no_such_rule"
    conn.execute(
        "UPDATE lemmas SET law_json = ? WHERE alpha_key = ?",
        (json.dumps(data, sort_keys=True), key),
    )
    conn.commit()
    conn.close()
    assert (
        ev.main(["--report", str(tmp_path / "s.db"), "--admit", key])
        == 1
    )
    out = capsys.readouterr().out
    assert "STRICT REPLAY FAILED" in out


# ---------------------------------------------------------------------------
#  Declared objects — the lemmas table as the object store (ADR 0004)
# ---------------------------------------------------------------------------


def _t(d=4):
    return TensorType((d, d))


def test_object_codec_marks_and_validates_kind():
    """The record codec: kind stamped, defaulted, and validated."""
    rule = _BY_NAME["id_add"]
    data = object_to_data(rule, kind="law")
    assert data["kind"] == "law" and object_kind(data) == "law"
    # a non-law kind stamps provenance; every kind is Rewrite-shaped
    abs_data = object_to_data(rule, kind="abstraction")
    assert abs_data["kind"] == "abstraction"
    assert object_from_data(json.loads(json.dumps(abs_data))).name == (
        "id_add"
    )
    # a legacy record — no "kind" key — reads as a law
    del abs_data["kind"]
    assert object_kind(abs_data) == "law"
    assert object_from_data(abs_data).name == "id_add"
    # unknown kinds fail loudly at both ends — never a silent admit
    with pytest.raises(ValueError, match="unknown object kind"):
        object_to_data(rule, kind="widget")
    with pytest.raises(ValueError, match="unknown object kind"):
        object_from_data({"kind": "widget", "version": 2})


def test_store_and_admit_object_roundtrip(tmp_path):
    """store_object / stored_object / admit_object mirror the lemma
    seam; the record carries its declared kind."""
    conn = ev.connect(str(tmp_path / "s.db"))
    rule = _BY_NAME["id_add"]
    key = ev.store_object(conn, rule, corpus_hash="ctx", kind="law")
    assert key == repr(alpha_key(rule.lhs, rule.rhs))
    record = ev.stored_object(conn, key)
    assert record["kind"] == "law"
    assert record["serializable"] is True
    rebuilt, data = ev.admit_object(conn, key)
    assert rebuilt.name == "id_add" and data == record
    assert ev.stored_object(conn, "no-such-key") is None
    assert ev.admit_object(conn, "no-such-key") is None
    conn.close()


def test_store_object_nonlaw_kind_reads_back(tmp_path):
    """The kind column-free store: provenance lives in the record."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(
        conn, _BY_NAME["sub_to_add"], kind="abstraction"
    )
    assert ev.stored_object(conn, key)["kind"] == "abstraction"
    rebuilt, record = ev.admit_object(conn, key)
    assert rebuilt.name == "sub_to_add"
    assert record["kind"] == "abstraction"
    # a row whose record predates "kind" defaults to "law"
    row = conn.execute(
        "SELECT law_json FROM lemmas WHERE alpha_key = ?", (key,)
    ).fetchone()
    data = json.loads(row["law_json"])
    del data["kind"]
    conn.execute(
        "UPDATE lemmas SET law_json = ? WHERE alpha_key = ?",
        (json.dumps(data, sort_keys=True), key),
    )
    conn.commit()
    assert ev.stored_object(conn, key)["kind"] == "law"
    rebuilt, record = ev.admit_object(conn, key)
    assert record["kind"] == "law"
    conn.close()


def test_admit_object_rejects_unknown_kind(tmp_path):
    """A record claiming a kind the codec does not know is refused."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(conn, _BY_NAME["id_add"])
    record = ev.stored_object(conn, key)
    record["kind"] = "widget"
    conn.execute(
        "UPDATE lemmas SET law_json = ? WHERE alpha_key = ?",
        (json.dumps(record, sort_keys=True), key),
    )
    conn.commit()
    with pytest.raises(ValueError, match="unknown object kind"):
        ev.admit_object(conn, key)
    # stored_object itself only reads — kind still surfaces
    assert ev.stored_object(conn, key)["kind"] == "widget"
    conn.close()


def test_store_object_materializes_cert(tmp_path):
    """Object rows materialize derivations the same way lemma rows
    do — cert replay is part of the record, not the kind."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(
        conn, _BY_NAME["silu_fold"], kind="abstraction"
    )
    record = ev.stored_object(conn, key)
    assert record["kind"] == "abstraction"
    cert = ev.stored_certificate(record)
    assert cert is not None and cert.n_steps == 1
    assert list(cert.rules_used) == ["silu_expand"]
    conn.close()


def test_admitted_object_fires_identically(tmp_path):
    """The seam end-to-end: a shipped law stored AS an object record
    admits into a Rewrite that fires identically — laws are the
    object language's first inhabitants."""
    conn = ev.connect(str(tmp_path / "s.db"))
    rule = _BY_NAME["silu_mul_form"]
    key = ev.store_object(conn, rule, kind="law")
    got = ev.admit_object(conn, key)
    assert got is not None
    rebuilt, record = got
    assert record["kind"] == "law" and record["serializable"] is True
    g, u = Var("g", _t()), Var("u", _t())
    src = Op.make("mul", Op.make("silu", g), u)
    dst = Op.make("mul", Op.make("mul", g, Op.make("sigmoid", g)), u)
    for r in (rule, rebuilt):
        eg = EGraph()
        root = eg.add_term(src)
        eg.run([r], root, max_iterations=4, max_nodes=10_000)
        assert eg.find(root) == eg.find(eg.add_term(dst)), r.name
    # and the stored cert replays strictly off the same record
    cert = ev.stored_certificate(record)
    assert cert is not None and cert.replayable
    assert [(s.rule, s.path) for s in cert.steps] == [
        ("silu_expand", (0,))
    ]
    conn.close()


def test_main_add_and_admit_object(tmp_path, capsys):
    """``--add-object`` / ``--admit-object`` mirror the lemma CLI."""
    db = str(tmp_path / "s.db")
    assert ev.main(["--report", db, "--add-object", "id_add"]) == 0
    out = capsys.readouterr().out
    assert "stored id_add" in out and "full-data" in out
    assert "kind: law" in out
    key = repr(
        alpha_key(_BY_NAME["id_add"].lhs, _BY_NAME["id_add"].rhs)
    )
    assert ev.main(["--report", db, "--admit-object", key]) == 0
    out = capsys.readouterr().out
    assert "admitted id_add" in out and "kind: law" in out
    assert "cert: none" in out
    assert ev.main(["--report", db, "--add-object", "no_such"]) == 1
    assert ev.main(["--report", db, "--admit-object", "bogus"]) == 1
    out = capsys.readouterr().out
    assert "no shipped law" in out and "no lemma stored" in out


def test_main_add_object_with_kind(tmp_path, capsys):
    """``--kind`` stamps the declared provenance on the stored row."""
    db = str(tmp_path / "s.db")
    assert (
        ev.main(
            [
                "--report",
                db,
                "--add-object",
                "sub_to_add",
                "--kind",
                "bridge",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "stored sub_to_add" in out and "kind: bridge" in out
    key = repr(
        alpha_key(
            _BY_NAME["sub_to_add"].lhs, _BY_NAME["sub_to_add"].rhs
        )
    )
    assert ev.main(["--report", db, "--admit-object", key]) == 0
    out = capsys.readouterr().out
    assert "admitted sub_to_add" in out and "kind: bridge" in out


def test_main_admit_object_replays_cert_strict(tmp_path, capsys):
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(
        conn, _BY_NAME["silu_fold"], kind="abstraction"
    )
    conn.close()
    db = str(tmp_path / "s.db")
    assert ev.main(["--report", db, "--admit-object", key]) == 0
    out = capsys.readouterr().out
    assert "kind: abstraction" in out and "replayed strict" in out


def test_main_admit_object_strict_replay_failure(tmp_path, capsys):
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(conn, _BY_NAME["silu_fold"])
    record = ev.stored_object(conn, key)
    record["cert"]["steps"][0]["rule"] = "no_such_rule"
    conn.execute(
        "UPDATE lemmas SET law_json = ? WHERE alpha_key = ?",
        (json.dumps(record, sort_keys=True), key),
    )
    conn.commit()
    conn.close()
    db = str(tmp_path / "s.db")
    assert ev.main(["--report", db, "--admit-object", key]) == 1
    out = capsys.readouterr().out
    assert "STRICT REPLAY FAILED" in out


def test_main_admit_object_unknown_kind_cli(tmp_path, capsys):
    """An unclassifiable record fails the CLI admit, not silently."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(conn, _BY_NAME["id_add"])
    record = ev.stored_object(conn, key)
    record["kind"] = "widget"
    conn.execute(
        "UPDATE lemmas SET law_json = ? WHERE alpha_key = ?",
        (json.dumps(record, sort_keys=True), key),
    )
    conn.commit()
    conn.close()
    db = str(tmp_path / "s.db")
    assert ev.main(["--report", db, "--admit-object", key]) == 1
    out = capsys.readouterr().out
    assert "unknown object kind" in out
