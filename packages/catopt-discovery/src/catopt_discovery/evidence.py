"""Evidence store — a persistent, content-addressed verdict record.

``catopt_discovery.pipeline`` recomputes every verdict on every run
(~25 s) and scatters the results across ``/tmp`` JSON dumps and
prose retros.  A verdict is *deterministic given the corpus, the
search rule set and the verification code* — a perfect cache.  This
module is that store: a small ``sqlite3`` database (stdlib only, no
new dependencies) keyed by content hash.

Three tables:

* ``candidates`` — one row per proposal, keyed by the alpha-normal
  equality key (``laws.serialize.alpha_key``, repr'd — the same
  canonicalisation ``catopt_discovery.proposal._key`` delegates to) with the
  last-seen name / family and a rendered proposal for inspection.
* ``verdicts`` — one row per candidate *per run*, keyed by
  ``(alpha_key, corpus_hash, rules_hash, code_rev, run_id)``.  The
  columns are exactly the fields the pipeline's ``Evidence`` record
  already computes — this module measures nothing, it persists.
* ``lemmas`` — one row per *declared object*: the full
  ``laws.serialize.law_to_data`` record under the same alpha-normal
  key, so ``--admit`` rebuilds a live ``Rewrite`` straight from the
  store.  The record's ``"kind"`` field marks the declaration's
  provenance — ``"law"`` for a shipped/admitted law (the object's
  first inhabitants), ``"abstraction"`` / ``"bridge"`` for objects
  the admission path synthesizes (ADR 0004 — see
  ``project/retros/object-record.md``).  ``"kind"`` lives *inside*
  the record, not a column: the record is the self-describing
  document (the ``cert`` field set the precedent), and a record
  written before the field existed reads as ``"law"`` — no schema
  migration.  The ``serializable`` / ``missing_hooks`` fields inside
  the JSON record say honestly which laws are full-data and which
  are pattern(+cond) with a ``check``/``derive`` remainder that
  still needs code — see ``project/retros/lemma-store.md``.  When
  the law carries a ``derivation`` annotation, :func:`store_lemma`
  (equivalently :func:`store_object` with ``kind="law"``) also
  materializes it as a replayable certificate
  (``catopt_discovery.lemma_cert.materialize``) and stores it in the
  record's ``"cert"`` field — ``--admit`` / ``--admit-object``
  replays it strictly, so a stored object's derivation is a
  verifiable proof, not just provenance metadata.

Scope keys — the honest boundary of a cached verdict:

* ``corpus_hash`` — sha256 over ``source:name:shape-key`` entries of
  every bench + model term the pipeline measured against.
* ``rules_hash`` — sha256 over the alpha-normal keys of the search
  rule set; a ``--holdout`` run is a different context, so its
  verdicts can never poison a full-library cache.
* ``code_rev`` — ``git rev-parse --short HEAD``, extended with a
  hash of the tracked diff (plus the contents of untracked files)
  under ``packages`` / ``tools`` / ``bench`` — the only code a
  verdict can depend on — so uncommitted edits to the verifier,
  corpus or generator invalidate honestly.  ``"unknown"`` (no git)
  rows are written but never served as cache hits.

CLI — history over the store, plus the lemma seam::

    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db
    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db --ships
    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db --flips
    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db \
        --history mul
    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db \
        --add-lemma softmax_fold
    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db \
        --admit '<alpha_key>'
    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db \
        --add-object silu_mul_form --kind abstraction
    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db \
        --admit-object '<alpha_key>'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from catopt_discovery import REPO_ROOT

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS candidates (
    alpha_key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    family TEXT NOT NULL,
    proposal_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    alpha_key TEXT NOT NULL,
    corpus_hash TEXT NOT NULL,
    rules_hash TEXT NOT NULL,
    code_rev TEXT NOT NULL,
    run_id TEXT NOT NULL,
    holdout TEXT NOT NULL,
    numeric_true INTEGER,
    derivable INTEGER NOT NULL,
    witness_json TEXT NOT NULL,
    relation TEXT NOT NULL,
    census_sites INTEGER NOT NULL,
    relaxed INTEGER NOT NULL,
    matches INTEGER NOT NULL,
    example TEXT NOT NULL,
    fires INTEGER NOT NULL,
    fire_cases_json TEXT NOT NULL,
    changed INTEGER NOT NULL,
    paid INTEGER NOT NULL,
    verify_fail INTEGER NOT NULL,
    drop_pct REAL NOT NULL,
    cert INTEGER NOT NULL,
    enode_ratio REAL NOT NULL,
    verdict TEXT NOT NULL,
    ts TEXT NOT NULL,
    PRIMARY KEY (alpha_key, corpus_hash, rules_hash, code_rev, run_id)
);
CREATE TABLE IF NOT EXISTS lemmas (
    alpha_key TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    law_json TEXT NOT NULL,
    derivation_json TEXT NOT NULL,
    corpus_hash TEXT NOT NULL,
    added_ts TEXT NOT NULL
);
"""

#: Verdict measurement columns, in schema order after the key fields.
_VERDICT_COLS = (
    "numeric_true",
    "derivable",
    "witness_json",
    "relation",
    "census_sites",
    "relaxed",
    "matches",
    "example",
    "fires",
    "fire_cases_json",
    "changed",
    "paid",
    "verify_fail",
    "drop_pct",
    "cert",
    "enode_ratio",
    "verdict",
)

#: Repo paths a verdict can depend on — the engine, the tools and the
#: bench corpus builders.  The dirty-worktree hash reads only these,
#: so an unrelated untracked dir (e.g. ``toy/``) does not flip it.
_CODE_PATHS = (
    "packages",
    "tools",
    "bench",
    "pyproject.toml",
    "uv.lock",
)

_REPO = REPO_ROOT


def connect(path: str) -> sqlite3.Connection:
    """Open (creating if needed) the evidence store at *path*."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def new_run_id() -> str:
    """Return a fresh run identifier."""
    return uuid.uuid4().hex[:12]


def now() -> str:
    """Return the current UTC timestamp in ISO form."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def _hash_entries(entries: Iterable[str]) -> str:
    """Return the sha256 over the sorted *entries*."""
    h = hashlib.sha256()
    for e in sorted(entries):
        h.update(e.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def corpus_hash(entries: Iterable[str]) -> str:
    """Content-hash the corpus: one ``source:name:shape-key`` entry each."""
    return _hash_entries(entries)


def rules_hash(keys: Iterable[str]) -> str:
    """Content-hash the search rule set's alpha-normal key reprs."""
    return _hash_entries(keys)


def _git(root: Path, *args: str) -> str:
    """Return the stdout of ``git *args`` at *root* ("" on failure)."""
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            cwd=root,
            check=False,
        )
    except OSError:
        return ""
    return out.stdout if out.returncode == 0 else ""


def _hash_untracked(path: Path, h: Any) -> None:
    """Mix the contents of an untracked file or tree into *h*."""
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
    elif path.is_file():
        files = [path]
    else:
        return
    for f in files:
        try:
            h.update(f.read_bytes())
        except OSError:
            continue


def code_rev(root: Path | None = None) -> str:
    """Return the verification code's revision for cache validity.

    ``git rev-parse --short HEAD`` on a clean tree; when the tracked
    code paths differ from HEAD the sha is extended with a hash of
    the diff (plus untracked file contents), so uncommitted edits to
    the verifier, corpus builders or generator change the key.  A
    new ``*.py`` under ``tools`` is hashed by content, so this file
    itself — and its edits — version the cache.  ``"unknown"`` when
    git cannot answer; ``"unknown"`` rows are written but never
    served as cache hits.
    """
    root = _REPO if root is None else root
    rev = _git(root, "rev-parse", "--short", "HEAD").strip()
    if not rev:
        return "unknown"
    status = _git(root, "status", "--porcelain", "--", *_CODE_PATHS)
    if not status.strip():
        return rev
    h = hashlib.sha256(status.encode("utf-8"))
    h.update(
        _git(root, "diff", "HEAD", "--", *_CODE_PATHS).encode("utf-8")
    )
    for line in status.splitlines():
        if line.startswith("??"):
            _hash_untracked(root / line[3:].strip(), h)
    return f"{rev}+{h.hexdigest()[:8]}"


def verdict_row(alpha_key: str, ev: Any, lhs: str, rhs: str) -> dict:
    """Serialise one ``Evidence``-shaped record as a store row.

    *ev* is the pipeline's ``Evidence`` by duck type — this module
    stays torch/catopt-free.  *lhs* / *rhs* are the rendered pattern
    sides kept in ``proposal_json`` for inspection.  ``drop_pct`` is
    stored in percent (``cost_drop * 100``); ``verdict`` is ``"SHIP"``
    or ``"no:<reason>"``.
    """
    p = ev.proposal
    return {
        "alpha_key": alpha_key,
        "name": p.name,
        "family": p.family,
        "proposal_json": json.dumps(
            {
                "name": p.name,
                "family": p.family,
                "sources": list(p.sources),
                "lhs": lhs,
                "rhs": rhs,
            },
            sort_keys=True,
        ),
        "numeric_true": None
        if ev.num_true is None
        else int(ev.num_true),
        "derivable": int(ev.derivable),
        "witness_json": json.dumps(list(ev.witness)),
        "relation": ev.relation,
        "census_sites": ev.census_sites,
        "relaxed": ev.relaxed,
        "matches": ev.matches,
        "example": ev.example,
        "fires": ev.fires,
        "fire_cases_json": json.dumps(list(ev.fire_cases)),
        "changed": ev.changed,
        "paid": ev.paid,
        "verify_fail": ev.verify_fail,
        "drop_pct": ev.cost_drop * 100.0,
        "cert": ev.cert_fail,
        "enode_ratio": ev.closure_ratio,
        "verdict": (
            "SHIP" if ev.shippable else f"no:{ev.no_ship_reason}"
        ),
    }


def record_run(
    conn: sqlite3.Connection, meta: dict[str, str], rows: Iterable[dict]
) -> None:
    """Upsert the candidates and insert this run's verdict rows.

    *meta* carries the scope: ``corpus_hash``, ``rules_hash``,
    ``code_rev``, ``run_id``, ``holdout`` and ``ts``.  Each *rows*
    entry is a :func:`verdict_row` dict; an identical re-measurement
    under a new ``run_id`` records "seen again", which is what the
    history report counts.
    """
    vcols = ", ".join(_VERDICT_COLS)
    vph = ", ".join("?" for _ in _VERDICT_COLS)
    with conn:
        for r in rows:
            conn.execute(
                "INSERT INTO candidates (alpha_key, name, family,"
                " proposal_json) VALUES (?, ?, ?, ?)"
                " ON CONFLICT(alpha_key) DO UPDATE SET"
                " name=excluded.name, family=excluded.family,"
                " proposal_json=excluded.proposal_json",
                (
                    r["alpha_key"],
                    r["name"],
                    r["family"],
                    r["proposal_json"],
                ),
            )
            conn.execute(
                f"INSERT OR REPLACE INTO verdicts"
                f" (alpha_key, corpus_hash, rules_hash, code_rev,"
                f" run_id, holdout, {vcols}, ts)"
                f" VALUES (?, ?, ?, ?, ?, ?, {vph}, ?)",
                (
                    r["alpha_key"],
                    meta["corpus_hash"],
                    meta["rules_hash"],
                    meta["code_rev"],
                    meta["run_id"],
                    meta["holdout"],
                    *[r[c] for c in _VERDICT_COLS],
                    meta["ts"],
                ),
            )


def latest_verdicts(
    conn: sqlite3.Connection,
    corpus_hash: str,
    rules_hash: str,
    code_rev: str,
) -> dict[str, dict]:
    """Return the newest verdict row per candidate for one context.

    A row is a valid cache entry only when all three scope keys
    match; ``"unknown"`` code revisions cannot vouch for the
    verification code, so they yield no hits.
    """
    if code_rev == "unknown":
        return {}
    out: dict[str, dict] = {}
    for row in conn.execute(
        "SELECT * FROM verdicts"
        " WHERE corpus_hash = ? AND rules_hash = ? AND code_rev = ?"
        " ORDER BY ts DESC, rowid DESC",
        (corpus_hash, rules_hash, code_rev),
    ):
        out.setdefault(row["alpha_key"], dict(row))
    return out


# ---------------------------------------------------------------------------
#  Objects — declared objects as stored data, reconstructable into
#  live rewrites
# ---------------------------------------------------------------------------
#
#  A ``lemmas`` row is the seam between the verdict cache and the
#  library: a SHIP candidate the emit machinery wrote up — or any
#  declared object — can be *stored*, and ``--admit`` turns the
#  stored record back into a ``Rewrite`` object that fires.  The
#  table's name predates the generalisation: it is now the
#  *declared-object* store, and the record's ``"kind"`` field marks
#  what each row declares (ADR 0004, plan 0017 stage 1).  The heavy
#  machinery stays lazy — the verdict-report path never imports
#  catopt.


#: Sentinel for :func:`store_object`'s ``cert=`` — distinguishes the
#: default ("materialize the recorded derivation") from an explicit
#: ``None`` ("store no certificate").
_UNSET: Any = object()


def _materialize_cert(rule: Any, universe: Any) -> Any:
    """Materialize *rule*'s recorded derivation as a Certificate.

    Only derivation-carrying rules are attempted — the certificate
    proves the *annotation's* claim, so a rule with no ``derivation``
    honestly stores ``cert: null``.  Saturation runs under the named
    premises drawn from *universe* (default: the shipped
    ``ALL_RULES``); a non-linear verdict (``saturation-only``,
    ``gap``, ``bad-derivation``) yields ``None`` — a merge witness
    that cannot replay standalone is never stored as a proof.
    """
    if not rule.derivation:
        return None
    from catopt_core.laws import ALL_RULES

    from catopt_discovery import lemma_cert

    uni = list(ALL_RULES) if universe is None else list(universe)
    _row, cert = lemma_cert.materialize(rule, uni)
    return cert


def store_object(
    conn: sqlite3.Connection,
    rule: Any,
    corpus_hash: str = "",
    *,
    kind: str = "law",
    cert: Any = _UNSET,
    universe: Any = None,
) -> str:
    """Persist *rule* as a declared-object row; return its alpha key.

    The row's ``law_json`` is the full object record
    (``catopt_core.laws.serialize.object_to_data``) — pattern pair,
    ``cond``, ``dspec``, tags, derivation, error bound, the
    ``serializable`` / ``missing_hooks`` honesty flags, and the
    ``"kind"`` field marking the declaration's provenance (one of
    ``OBJECT_KINDS``; the codec raises ``ValueError`` on an
    unrecognised kind).  A rule whose ``check``/``derive`` needs code
    is stored *flagged*, not dropped: the record says exactly which
    parts data cannot carry.  ``corpus_hash`` records which corpus
    context the object was measured under (``""`` when none applies
    — e.g. storing a shipped law).

    ``cert=`` controls the record's ``"cert"`` field: the default
    materializes the rule's recorded ``derivation`` as a replayable
    certificate (through ``catopt_discovery.lemma_cert.materialize``
    under *universe*); an explicit :class:`Certificate` embeds as
    given; an explicit ``None`` stores ``cert: null``.  Whatever the
    path, the stored cert is the honest boundary — ``null`` where no
    derivation replays, never a stub.
    """
    from catopt_core.laws.serialize import alpha_key, object_to_data

    if cert is _UNSET:
        cert = _materialize_cert(rule, universe)
    data = object_to_data(rule, cert=cert, kind=kind)
    key = repr(alpha_key(rule.lhs, rule.rhs))
    with conn:
        conn.execute(
            "INSERT INTO lemmas"
            " (alpha_key, name, law_json, derivation_json,"
            "  corpus_hash, added_ts)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(alpha_key) DO UPDATE SET"
            " name=excluded.name, law_json=excluded.law_json,"
            " derivation_json=excluded.derivation_json,"
            " corpus_hash=excluded.corpus_hash,"
            " added_ts=excluded.added_ts",
            (
                key,
                rule.name,
                json.dumps(data, sort_keys=True),
                json.dumps(list(rule.derivation)),
                corpus_hash,
                now(),
            ),
        )
    return key


def store_lemma(
    conn: sqlite3.Connection,
    rule: Any,
    corpus_hash: str = "",
    *,
    cert: Any = _UNSET,
    universe: Any = None,
) -> str:
    """Persist *rule*'s data form in ``lemmas``; return its alpha key.

    The law-flavoured spelling of :func:`store_object` with
    ``kind="law"`` — a stored law is a ``"law"``-kind declared
    object.  See :func:`store_object` for the record fields and the
    ``cert=`` semantics.
    """
    return store_object(
        conn,
        rule,
        corpus_hash,
        kind="law",
        cert=cert,
        universe=universe,
    )


def stored_object(
    conn: sqlite3.Connection, alpha_key: str
) -> dict | None:
    """Return the stored object record for *alpha_key*, or ``None``.

    The parsed ``law_json`` with ``record["kind"]`` made explicit —
    a row written before the field existed defaults to ``"law"``
    (``laws.serialize.object_kind``), since every pre-object record
    is a law by provenance.
    """
    from catopt_core.laws.serialize import object_kind

    row = conn.execute(
        "SELECT law_json FROM lemmas WHERE alpha_key = ?",
        (alpha_key,),
    ).fetchone()
    if row is None:
        return None
    data = json.loads(row["law_json"])
    data["kind"] = object_kind(data)
    return data


def admit_object(
    conn: sqlite3.Connection, alpha_key: str
) -> tuple[Any, dict] | None:
    """Rebuild a stored object as a live ``Rewrite``, or ``None``.

    Returns ``(rule, record)`` — the reconstructed rule plus the
    parsed object record, so the caller can read ``record["kind"]``,
    ``record["serializable"]`` and ``record["missing_hooks"]`` before
    trusting the rule to fire identically to its source.  A
    ``serializable: false`` record still rebuilds — pattern + cond —
    but the reconstructed rule fires without the dropped hooks.  A
    record claiming a kind the codec does not know raises
    ``ValueError`` — a declaration the store cannot classify is a
    failure to surface, never a silent admit.  The record's
    ``"cert"`` field stays on the record — decode + verify it with
    :func:`stored_certificate`.
    """
    from catopt_core.laws.serialize import object_from_data

    data = stored_object(conn, alpha_key)
    if data is None:
        return None
    return object_from_data(data), data


def admit_lemma(
    conn: sqlite3.Connection, alpha_key: str
) -> tuple[Any, dict] | None:
    """Rebuild a stored lemma as a live ``Rewrite``, or ``None``.

    The law-flavoured spelling of :func:`admit_object` — the record
    shape and admission semantics are identical; ``"lemma"`` names
    the ``kind="law"`` inhabitant of the object store.
    """
    return admit_object(conn, alpha_key)


def lemma_rows(conn: sqlite3.Connection) -> list[dict]:
    """Return every stored object row in ``lemmas``, newest first.

    The table predates the generalisation — each row is a declared
    object whose ``law_json`` record carries its ``"kind"``.
    """
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM lemmas ORDER BY added_ts DESC, rowid DESC"
        )
    ]


def stored_certificate(record: dict, rules: Any = None) -> Any:
    """Rebuild + strictly verify the cert an object record carries.

    *record* is the parsed object-record dict (the second half of
    :func:`admit_object`'s return).  *rules* resolves the rule names
    the steps reference — default the shipped ``ALL_RULES``, which
    covers a stored derivation's premises by construction (they are
    shipped laws).  Returns ``None`` when the record claims no
    certificate — ``cert: null`` is an honest absence, never a stub.

    Verification is strict: ``egraph_dependent`` steps are refused,
    and a stored cert that does not replay raises
    ``CertificateVerificationError`` — a corrupt or drifted record is
    a failure to surface, not a flag to read.
    """
    blob = record.get("cert")
    if blob is None:
        return None
    from catopt_core.egraph import cert_from_data, verify_certificate

    if rules is None:
        from catopt_core.laws import ALL_RULES

        rules = ALL_RULES
    cert = cert_from_data(blob, rules)
    verify_certificate(cert.src, cert, strict=True)
    return cert


# ---------------------------------------------------------------------------
#  Reporting — the history a /tmp JSON dump could never answer
# ---------------------------------------------------------------------------


def _verdict_label(v: str) -> str:
    """Return the SHIP/no part of a stored verdict string."""
    return v.split(":", 1)[0]


def _candidate_history(conn: sqlite3.Connection) -> dict:
    """Group every verdict row by candidate, newest first."""
    hist: dict[str, dict] = {}
    rows = conn.execute(
        "SELECT v.*, c.name AS cname, c.family AS cfamily"
        " FROM verdicts v JOIN candidates c"
        " ON c.alpha_key = v.alpha_key"
        " ORDER BY v.ts DESC, v.rowid DESC"
    ).fetchall()
    for r in rows:
        d = dict(r)
        ent = hist.setdefault(
            d["alpha_key"],
            {"name": d["cname"], "family": d["cfamily"], "rows": []},
        )
        ent["rows"].append(d)
    return hist


def _history_table(hist: dict, ships_only: bool = False) -> list[str]:
    """Render the per-candidate latest-verdict table."""
    head = (
        f"{'candidate':<28} {'family':<18} {'verdict':<26} "
        f"{'runs':>4} {'ctx':>3} {'fires':>5} {'paid':>4} "
        f"{'drop%':>6}  last-seen"
    )
    lines = [head, "-" * len(head)]
    ents = sorted(
        hist.values(),
        key=lambda e: (e["rows"][0]["verdict"] != "SHIP", e["name"]),
    )
    for ent in ents:
        rows = ent["rows"]
        latest = rows[0]
        ever_ship = any(r["verdict"] == "SHIP" for r in rows)
        if ships_only and not ever_ship:
            continue
        verdict = latest["verdict"]
        label = verdict if len(verdict) <= 26 else verdict[:25] + "…"
        n_runs = len({r["run_id"] for r in rows})
        n_ctx = len({(r["corpus_hash"], r["rules_hash"]) for r in rows})
        lines.append(
            f"{ent['name']:<28} {ent['family']:<18} {label:<26} "
            f"{n_runs:>4} {n_ctx:>3} {latest['fires']:>5} "
            f"{latest['paid']:>4} {latest['drop_pct']:>6.1f}  "
            f"{latest['ts'][:19]}"
            + ("  *" if ever_ship and verdict != "SHIP" else "")
        )
    return lines


def _flips(hist: dict) -> list[str]:
    """Render candidates whose verdict label changed across runs."""
    lines: list[str] = []
    for ent in sorted(hist.values(), key=lambda e: e["name"]):
        labels = {_verdict_label(r["verdict"]) for r in ent["rows"]}
        if len(labels) < 2:
            continue
        lines.append(f"  {ent['name']} [{ent['family']}]")
        for r in ent["rows"]:
            lines.append(
                f"    {r['ts'][:19]} run={r['run_id']} "
                f"corpus={r['corpus_hash'][:8]} "
                f"rules={r['rules_hash'][:8]} "
                f"rev={r['code_rev']}: {r['verdict']}"
            )
    return lines


def _history_detail(hist: dict, sub: str) -> list[str]:
    """Render every recorded verdict for names containing *sub*."""
    lines: list[str] = []
    for ent in sorted(hist.values(), key=lambda e: e["name"]):
        if sub.lower() not in ent["name"].lower():
            continue
        lines.append(f"  {ent['name']} [{ent['family']}]")
        for r in ent["rows"]:
            lines.append(
                f"    {r['ts'][:19]} run={r['run_id']} "
                f"corpus={r['corpus_hash'][:8]} "
                f"rules={r['rules_hash'][:8]} "
                f"rev={r['code_rev']} holdout={r['holdout'] or '-'}"
            )
            lines.append(
                f"      {r['verdict']}  true={r['numeric_true']} "
                f"deriv={r['derivable']} rel={r['relation']} "
                f"fires={r['fires']} paid={r['paid']} "
                f"drop={r['drop_pct']:.1f}% cert={r['cert']} "
                f"enode={r['enode_ratio']:.2f}x"
            )
    return lines


def render_report(
    conn: sqlite3.Connection,
    path: str,
    ships_only: bool = False,
    flips: bool = False,
    history: str | None = None,
) -> str:
    """Render the store's history report."""
    n_cand = conn.execute("SELECT count(*) FROM candidates").fetchone()[
        0
    ]
    n_verd = conn.execute("SELECT count(*) FROM verdicts").fetchone()[0]
    n_runs = conn.execute(
        "SELECT count(DISTINCT run_id) FROM verdicts"
    ).fetchone()[0]
    n_ctx = conn.execute(
        "SELECT count(DISTINCT corpus_hash || rules_hash) FROM verdicts"
    ).fetchone()[0]
    n_lem = conn.execute("SELECT count(*) FROM lemmas").fetchone()[0]
    lines = [
        f"== law_evidence — {path} ==",
        f"   {n_cand} candidates · {n_verd} verdict rows · "
        f"{n_runs} runs · {n_ctx} corpus/rule contexts · "
        f"{n_lem} lemmas",
        "",
    ]
    hist = _candidate_history(conn)
    if history is not None:
        lines.append(f"-- verdict history matching {history!r} --")
        lines.extend(_history_detail(hist, history) or ["  (none)"])
        return "\n".join(lines)
    if flips:
        lines.append("-- verdict flips across runs --")
        lines.extend(_flips(hist) or ["  (none — verdicts stable)"])
        return "\n".join(lines)
    title = (
        "-- candidates ever marked SHIP --"
        if ships_only
        else "-- candidates (latest verdict, runs seen) --"
    )
    lines.append(title)
    lines.extend(_history_table(hist, ships_only))
    if ships_only:
        lines.append("(* = latest verdict is no longer SHIP)")
    return "\n".join(lines)


def _store_object_cli(
    conn: sqlite3.Connection, name: str, kind: str
) -> int:
    """Store a shipped ``ALL_RULES`` law by name as an object row."""
    from catopt_core.laws import ALL_RULES

    by_name = {r.name: r for r in ALL_RULES}
    rule = by_name.get(name)
    if rule is None:
        print(f"no shipped law named {name!r}")
        return 1
    key = store_object(conn, rule, kind=kind)
    record = stored_object(conn, key)
    if record is None:  # unreachable — the row was just written
        return 1
    state = (
        "full-data"
        if record["serializable"]
        else "missing hooks: " + ", ".join(record["missing_hooks"])
    )
    print(f"stored {rule.name}  [{state}]")
    print(f"  kind: {record['kind']}")
    print(f"  alpha_key = {key}")
    cert = record.get("cert")
    if cert is None:
        print("  cert: none recorded")
    else:
        print(
            f"  cert: {len(cert['steps'])}-step derivation"
            f" {cert['rules_used']}"
        )
    return 0


def _admit_object_cli(conn: sqlite3.Connection, alpha_key: str) -> int:
    """Rebuild a stored object into a live ``Rewrite`` and show it."""
    try:
        got = admit_object(conn, alpha_key)
    except ValueError as exc:
        print(f"cannot admit object under {alpha_key}: {exc}")
        return 1
    if got is None:
        print(f"no lemma stored under {alpha_key}")
        return 1
    rule, data = got
    state = (
        "full-data"
        if data["serializable"]
        else "missing hooks: " + ", ".join(data["missing_hooks"])
    )
    print(f"admitted {rule.name}  [{state}]")
    print(f"  kind: {data['kind']}")
    print(f"  {rule!r}")
    try:
        cert = stored_certificate(data)
    except Exception as exc:
        print(f"  cert: STRICT REPLAY FAILED: {exc}")
        return 1
    if cert is None:
        print("  cert: none recorded")
    else:
        print(
            f"  cert: {cert.n_steps}-step {cert.rules_used}"
            " — replayed strict"
        )
    return 0


#: The ``--kind`` choices the CLI accepts — mirrors
#: ``laws.serialize.OBJECT_KINDS``, kept literal so the report path
#: stays catopt-free; ``object_to_data`` validates authoritatively.
_OBJECT_KIND_CHOICES = ("abstraction", "bridge", "law")


def _op_dispatch(
    args: argparse.Namespace, conn: sqlite3.Connection
) -> int | None:
    """Run a store/admit CLI op, or ``None`` when none was given.

    ``--add-lemma`` / ``--admit`` are the law-flavoured spellings of
    ``--add-object`` / ``--admit-object`` — the general seam is the
    object one.
    """
    if args.add_lemma is not None:
        return _store_object_cli(conn, args.add_lemma, args.kind)
    if args.add_object is not None:
        return _store_object_cli(conn, args.add_object, args.kind)
    if args.admit is not None:
        return _admit_object_cli(conn, args.admit)
    if args.admit_object is not None:
        return _admit_object_cli(conn, args.admit_object)
    return None


def main(argv: list[str] | None = None) -> int:
    """Print the evidence store's history report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        metavar="PATH",
        required=True,
        help="the sqlite evidence store to read",
    )
    parser.add_argument(
        "--ships",
        action="store_true",
        help="only candidates ever marked SHIP",
    )
    parser.add_argument(
        "--flips",
        action="store_true",
        help="candidates whose verdict changed across runs",
    )
    parser.add_argument(
        "--history",
        metavar="SUBSTR",
        help="every recorded verdict for names containing SUBSTR",
    )
    parser.add_argument(
        "--add-lemma",
        metavar="NAME",
        help="store a shipped ALL_RULES law as a lemma row"
        " (--add-object's law-flavoured spelling)",
    )
    parser.add_argument(
        "--admit",
        metavar="ALPHA_KEY",
        help="rebuild a stored lemma into a live Rewrite"
        " (--admit-object's law-flavoured spelling)",
    )
    parser.add_argument(
        "--add-object",
        metavar="NAME",
        help="store a shipped ALL_RULES law as a declared-object row",
    )
    parser.add_argument(
        "--admit-object",
        metavar="ALPHA_KEY",
        help="rebuild a stored declared object into a live Rewrite",
    )
    parser.add_argument(
        "--kind",
        metavar="KIND",
        choices=_OBJECT_KIND_CHOICES,
        default="law",
        help="declaration kind for --add-object / --add-lemma"
        " (default: law)",
    )
    args = parser.parse_args(argv)
    ops = (
        args.add_lemma,
        args.add_object,
        args.admit,
        args.admit_object,
    )
    if (
        not any(o is not None for o in ops)
        and not Path(args.report).is_file()
    ):
        print(f"no evidence store at {args.report}")
        return 1
    conn = connect(args.report)
    try:
        rc = _op_dispatch(args, conn)
        if rc is not None:
            return rc
        print(
            render_report(
                conn,
                args.report,
                ships_only=args.ships,
                flips=args.flips,
                history=args.history,
            )
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
