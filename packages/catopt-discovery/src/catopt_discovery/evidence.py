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
    .venv/bin/python -m catopt_discovery.evidence --report /tmp/laws.db \
        --admit-object '<alpha_key>' --gauntlet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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


def _stored_verdict(ev: Any) -> str:
    """Classify the persisted verdict for one measured ``Evidence``.

    The same precedence :func:`_truth_gate` applies in the gauntlet,
    applied here at the store boundary: a measured ``num_true is
    False`` is a counterexample, and ``derivable`` may waive an
    *absent* measurement — never a measured one.
    ``Evidence.truth``/``shippable`` (``pipeline``) still let a
    derivation override the oracle, so the row re-checks the measured
    site itself rather than trusting ``ev.shippable`` to have
    refused: a ``derivable`` + ``num_true is False`` record persists
    ``no:false (numeric oracle rejects; derivation overridden)``,
    not ``SHIP``.  The reason stem is the pipeline's own
    (``Evidence.no_ship_reason``'s ``"false (numeric oracle
    rejects)"``), so a non-derivable measured-false row is spelled
    exactly as before — only the derivation-override case gains a
    marker, visible in the history report like the gauntlet's
    ``"derivation overridden by a measured counterexample"``.
    """
    if ev.num_true is False:
        over = "; derivation overridden" if ev.derivable else ""
        return f"no:false (numeric oracle rejects{over})"
    return "SHIP" if ev.shippable else f"no:{ev.no_ship_reason}"


def verdict_row(alpha_key: str, ev: Any, lhs: str, rhs: str) -> dict:
    """Serialise one ``Evidence``-shaped record as a store row.

    *ev* is the pipeline's ``Evidence`` by duck type — this module
    stays torch/catopt-free.  *lhs* / *rhs* are the rendered pattern
    sides kept in ``proposal_json`` for inspection.  ``drop_pct`` is
    stored in percent (``cost_drop * 100``); ``verdict`` is ``"SHIP"``
    or ``"no:<reason>"``, classified by :func:`_stored_verdict` — a
    measured ``num_true is False`` never persists as ``SHIP``, no
    matter what the derivation says.
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
        "verdict": _stored_verdict(ev),
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


#: Verdict columns whose value is a measurement *of* the corpus the
#: row was recorded under — match/site counts, the firing probe, pay
#: and the closure sweep.  Rows under different ``corpus_hash``
#: scopes must never have these compared, summed or merged: each
#: answers "on this corpus" only.  ``verdict`` itself is listed here
#: deliberately — ``SHIP`` conflates truth with the pay columns, so
#: the label is a corpus-A answer to a corpus-A question.
#:
#: The corpus-INVARIANT columns — the ones a cross-scope reader may
#: trust — are ``numeric_true`` (an equality measured true on a real
#: instance is a fact about the candidate; the corpus supplied the
#: instance, not the truth), ``derivable`` and ``relation`` (both
#: scoped by ``rules_hash``, which :func:`verdicts_across_scopes`
#: holds fixed), plus the descriptive text (``witness_json`` /
#: ``example`` / the rendered sides name *what* was measured).
CORPUS_DEPENDENT_COLS = (
    "census_sites",
    "matches",
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


def verdicts_across_scopes(
    conn: sqlite3.Connection, meta_base: Mapping[str, str]
) -> dict[str, dict[str, dict]]:
    """Return the newest verdict per candidate, grouped by corpus.

    The cross-scope companion of :func:`latest_verdicts`: where that
    answers "what do we know under *this* context", this answers
    "what was ever measured, under which corpus".  Only the
    ``corpus_hash`` scope key is allowed to vary — *meta_base*'s
    ``rules_hash`` and ``code_rev`` still bind — so the result is a
    family of same-ruleset, same-code verdict tables, one per corpus
    the store has seen::

        {corpus_hash: {alpha_key: verdict_row}}

    Each inner dict is exactly :func:`latest_verdicts`' shape for
    that corpus (newest row per candidate), and every row still
    carries its own ``corpus_hash`` / ``run_id`` / ``ts`` columns —
    attribution survives flattening the grouping.

    The grouping, not a merged bag, is the contract.  The columns in
    :data:`CORPUS_DEPENDENT_COLS` are measurements *of* a corpus and
    are never comparable across groups; ``numeric_true``,
    ``derivable`` and ``relation`` are corpus-invariant and may be
    read across the boundary — "verdict V was measured on corpus A,
    not the current corpus B" is answerable without conflating the
    two.  The current scope is not special-cased out; the caller
    decides which groups are prior.  ``"unknown"`` code revisions
    yield no rows, matching ``latest_verdicts``.
    """
    if meta_base["code_rev"] == "unknown":
        return {}
    out: dict[str, dict[str, dict]] = {}
    for row in conn.execute(
        "SELECT * FROM verdicts"
        " WHERE rules_hash = ? AND code_rev = ?"
        " ORDER BY ts DESC, rowid DESC",
        (meta_base["rules_hash"], meta_base["code_rev"]),
    ):
        scope = out.setdefault(row["corpus_hash"], {})
        scope.setdefault(row["alpha_key"], dict(row))
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
#  The admission gauntlet — a stored object earns "usable" (ADR 0004)
# ---------------------------------------------------------------------------
#
#  ``admit_object`` reconstructs a stored record into a live
#  ``Rewrite`` — but reconstruction is not admission.  An *introduced*
#  object (a candidate the machine synthesized) faces the same
#  adversarial gauntlet a shipped law cleared: the numeric oracle and
#  derivability prover (``pipeline.measure``), the view oracle's
#  guarded-region sweep (``catopt_discovery.oracle`` — the candidate's
#  own ``cond`` decides which bindings the equality must hold on),
#  the typed-pay gate (every merged fire mints a well-typed member and
#  extraction pays somewhere), closure safety, and — when the record
#  carries a derivation — strict certificate replay.  ``run_gauntlet``
#  is that composition; ``usable`` is ``True`` only when every stage
#  passed.  The pipeline's ``Evidence.shippable`` verdict is *not*
#  reused verbatim because it answers a different question — a
#  guarded object is *conditionally* true by construction (the plain
#  view-oracle verdict stays ``conditional``), so the truth gate here
#  is the guarded-region sweep, not the unguarded verdict.


@dataclass(frozen=True)
class GauntletCorpus:
    """The measurement context a stored object's gauntlet runs on.

    The same inputs ``pipeline.run_pipeline`` assembles:
    ``real_terms`` for match enumeration, ``probe`` (``TermCase``
    list) for the firing / typed-pay / reach stages, ``base_rules``
    the search rule set the derivability oracle and the reach
    baseline use, ``census_op`` the op-tuple census, and
    ``sink``/``cost_fn`` the lowering + pricing backend.  Inject a
    small corpus in tests; the default is the real one
    (:func:`default_gauntlet_corpus`).
    """

    real_terms: tuple
    probe: tuple
    base_rules: tuple
    census_op: dict
    sink: Any
    cost_fn: Any


def default_gauntlet_corpus() -> GauntletCorpus:
    """Assemble the real corpus the pipeline measures against.

    The heavy imports stay lazy — the verdict-report path never loads
    catopt or torch.
    """
    from catopt_core.laws import ALL_RULES

    from catopt_discovery import intake as li
    from catopt_discovery import pipeline as pl
    from catopt_discovery.census import run_census
    from catopt_discovery.impact import (
        _bench_cases,
        _cost_fn,
        model_cases,
    )
    from catopt_discovery.shape_proposal import _sink

    census = run_census(pl._CENSUS_TOP)
    census_op = {
        (e["op"], tuple(e["children"])): e["count"]
        for e in census["op_tuples"]
    }
    bench, _be = _bench_cases()
    models, _me = model_cases()
    intake = li.load_cases()
    sink = _sink()
    return GauntletCorpus(
        real_terms=tuple(c.term for c in [*bench, *models, *intake]),
        probe=tuple([*models, *li.probe_cases()]),
        base_rules=tuple(ALL_RULES),
        census_op=census_op,
        sink=sink,
        cost_fn=_cost_fn(sink),
    )


@dataclass(frozen=True)
class GuardedRegion:
    """Outcome counts over the bindings an object's guard accepts.

    ``accepted`` / ``declined`` count the guard's decision;
    ``guard_err`` counts bindings the guard raised on (a non-total
    guard is not admissible data); the rest are the tri-state
    evaluation outcomes over the accepted region — ``rhs_err`` is an
    accepted binding whose instantiated RHS cannot denote, the
    ill-typed-mint signal the typed-pay gate audits at term level.
    ``envs`` is the enumeration cost — the number of synthesized sites
    the sweep evaluated (the window the counts are measured over).
    """

    accepted: int = 0
    declined: int = 0
    guard_err: int = 0
    equal: int = 0
    unequal: int = 0
    rhs_err: int = 0
    other_err: int = 0
    witness: str = ""
    counterexample: str = ""
    envs: int = 0


def _site_outcome(rule: Any, subst: dict, lhs_i: Any) -> str:
    """Evaluate one bound site under *rule*'s guard; return a tag.

    The guard is the rule's own ``check`` — the same environment and
    the same predicate an e-graph firing consults.  A ``derive``
    veto counts as ``declined`` (a firing aborts the same way); an
    uninstantiable RHS counts as ``rhs-err``.  Otherwise the
    instantiated pair is evaluated fp64 and the tri-state outcome —
    ``equal`` / ``unequal`` / ``rhs-err`` / ``lhs-err`` / ``both-err``
    / ``env-err`` — is the oracle's verdict on that binding.
    """
    from catopt_core.egraph.terms import _term_instantiate

    from catopt_discovery import oracle as lvo

    try:
        ok = rule.check is None or bool(rule.check(subst))
    except Exception:
        return "guard-err"
    inst = dict(subst)
    if ok and rule.derive is not None:
        try:
            extra = rule.derive(subst)
        except Exception:
            extra = None
        if extra is None:
            ok = False
        else:
            inst.update(extra)
    if not ok:
        return "declined"
    try:
        rhs_i = _term_instantiate(rule.rhs, inst)
    except Exception:
        return "rhs-err"
    outcome, _note = lvo.eval_instance(lhs_i, rhs_i)
    return outcome


def _guarded_evals(rule: Any, sites: Iterable) -> GuardedRegion:
    """Evaluate ``lhs == rhs`` on every guarded site in *sites*.

    *sites* yields ``(subst, lhs_term)`` pairs — the binding
    environment plus the instantiated LHS — from real matches or the
    synthesized domain.  This is the view oracle's sweep restricted
    to the region the object's own precondition accepts: the honest
    meaning of "the equality holds where the law can fire".
    """
    from catopt_core.ir import op_repr

    acc = dec = gerr = eq = neq = rerr = oerr = envs = 0
    wit = cex = ""
    for subst, lhs_i in sites:
        envs += 1
        out = _site_outcome(rule, subst, lhs_i)
        if out == "declined":
            dec += 1
            continue
        if out == "guard-err":
            gerr += 1
            continue
        acc += 1
        if out == "equal":
            eq += 1
            wit = wit or op_repr(lhs_i)
        elif out == "unequal":
            neq += 1
            cex = cex or op_repr(lhs_i)
        elif out == "rhs-err":
            rerr += 1
        else:
            oerr += 1
    return GuardedRegion(
        accepted=acc,
        declined=dec,
        guard_err=gerr,
        equal=eq,
        unequal=neq,
        rhs_err=rerr,
        other_err=oerr,
        witness=wit,
        counterexample=cex,
        envs=envs,
    )


def _attr_merge(domains: list, attr_combo: tuple, viewed: dict) -> Any:
    """Merge one attr combination into *viewed*; ``None`` on conflict.

    The enumeration machinery lives in the oracle — this delegates so
    the sweep's helpers keep one name.
    """
    from catopt_discovery import oracle as lvo

    return lvo._attr_merge(domains, attr_combo, viewed)


def _lhs_out_shapes(lhs_views: list, base: dict) -> list:
    """Return the instantiated LHS view nodes' output shapes."""
    from catopt_discovery import oracle as lvo

    return lvo._lhs_out_shapes(lhs_views, base)


def _synth_bases(lhs_pat: Any, rhs_pat: Any) -> Iterable:
    """Yield ``(base, viewed_shapes, out_shapes)`` per viewed combo.

    The outer half of the oracle's enumeration — delegated verbatim:
    the viewed-binding banks, the shape-valid attr domains, the
    derived free-operand shapes all live in
    :mod:`catopt_discovery.oracle`.
    """
    from catopt_discovery import oracle as lvo

    yield from lvo._synth_bases(lhs_pat, rhs_pat)


def _inst_pair(lhs_pat: Any, rhs_pat: Any, bound: dict) -> Any:
    """Instantiate both pattern sides under *bound*; ``None`` on failure."""
    from catopt_core.egraph.terms import _term_instantiate

    try:
        return (
            _term_instantiate(lhs_pat, bound),
            _term_instantiate(rhs_pat, bound),
        )
    except Exception:
        return None


def _synth_sites(lhs_pat: Any, rhs_pat: Any, *, limit: int) -> Iterable:
    """Yield ``(bound, lhs_term)`` over the view oracle's domain.

    ``oracle._binding_envs`` IS the enumeration — the same leaf
    binding banks, the same attr domains, the same fair ordering —
    with the evaluation step replaced by a yield of the *binding
    environment*, so the object's own guard, not the oracle, decides
    which region the equality must hold on.  Instantiated
    ``(lhs, rhs)`` pairs are deduped so each evaluable site is
    counted once.
    """
    from catopt_discovery import oracle as lvo

    seen: set = set()
    for full in lvo._binding_envs(lhs_pat, rhs_pat):
        pair = _inst_pair(lhs_pat, rhs_pat, full)
        if pair is None or pair in seen:
            continue
        seen.add(pair)
        yield full, pair[0]
        if len(seen) >= limit:
            return


def _region_clean(region: GuardedRegion) -> bool:
    """Whether a guarded region holds no *measured* counterexample.

    A measured counterexample is a guard-accepted binding the sweep
    *evaluated* and found contradictory: ``unequal`` (the two sides
    differ), ``rhs_err`` (the minted RHS cannot denote), or
    ``guard_err`` (the guard itself raised — a non-total predicate).
    These outrank any derivation: a proof that the equality holds
    cannot coexist with a measured instance where it does not, so the
    derivation is the thing that must yield.
    """
    return not (region.unequal or region.rhs_err or region.guard_err)


def _region_ok(region: GuardedRegion, *, need_equal: bool) -> bool:
    """Whether a guarded region supports the declared equality.

    The region must be contradiction-free (:func:`_region_clean`)
    and, where *need_equal* holds, must exhibit at least one equal
    instance (a guard that accepts nothing provable is vacuous, not
    verified).
    """
    return _region_clean(region) and (
        not need_equal or region.equal >= 1
    )


@dataclass(frozen=True)
class GauntletStage:
    """One gate's verdict: the stage name, pass/fail, evidence line."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class Gauntlet:
    """The adversarial admission verdict for one stored object.

    ``stages`` is the ordered gate list — the object's honest record
    of where it stood.  ``usable`` is ``True`` only when every stage
    passed: the store never reports a synthesized object as usable on
    the strength of reconstruction alone.  ``rule`` / ``record`` are
    the live ``Rewrite`` and the parsed object record (when stage 1
    cleared); ``evidence`` is the pipeline's measured ``Evidence``
    row; ``synth_region`` / ``real_region`` are the guarded-region
    sweeps for ``cond``-carrying objects.  ``auto_cond`` records an
    auto-cond attempt (``run_gauntlet(auto_cond=True)``): the found
    guard's clauses, the measured-domain counts and the refusal detail
    — ``None`` when no attempt ran.
    """

    alpha_key: str
    name: str = ""
    kind: str = ""
    usable: bool = False
    stages: tuple[GauntletStage, ...] = ()
    rule: Any = None
    record: dict | None = None
    evidence: Any = None
    synth_region: GuardedRegion | None = None
    real_region: GuardedRegion | None = None
    auto_cond: dict | None = None

    @property
    def reason(self) -> str:
        """Return the first failing stage, or the cleared verdict."""
        for s in self.stages:
            if not s.passed:
                return f"{s.name}: {s.detail}"
        return "cleared" if self.stages else "no stages ran"


def _truth_detail(evd: Any) -> str:
    """Render the measured truth evidence for the gauntlet report."""
    parts = [f"numeric={evd.num_true}", f"derivable={evd.derivable}"]
    if evd.view_verdict:
        parts.append(f"view={evd.view_verdict} ({evd.view_note})")
    return " ".join(parts)


def _region_detail(synth: GuardedRegion, real: GuardedRegion) -> str:
    """Render the guarded-region sweeps for the gauntlet report."""
    out: list[str] = []
    for label, r in (("synth", synth), ("real", real)):
        out.append(
            f"{label}: {r.equal}eq/{r.unequal}ne/{r.rhs_err}rerr "
            f"({r.accepted} accepted, {r.declined} declined"
            + (f", {r.guard_err} guard-err" if r.guard_err else "")
            + (f", {r.envs} envs" if r.envs else "")
            + ")"
        )
    return " | ".join(out)


def _guarded_truth(
    rule: Any, corpus: GauntletCorpus, limit: int
) -> Any:
    """Sweep the guard-accepted region of *rule*'s domain.

    Returns ``(synth_region, real_region)``: the synthesized domain
    (the oracle's binding banks) filtered by the rule's own ``check``,
    and every real corpus match filtered the same way.  The truth
    question for a guarded object is "equal wherever the rule can
    fire" — this pair answers it on both domains.

    The synthesized sweep is *selective-cap*: it runs at *limit* (the
    default window) and, when that window is starved — the rule is
    guarded and the window accepted nothing, a multi-clause guard's
    accepted corner lying past the default — re-runs once at
    :data:`catopt_discovery.oracle._GUARDED_CAP` (the policy lives in
    :func:`catopt_discovery.oracle.escalate_limit`).  The common case
    keeps the default: an unguarded rule and a guarded rule whose
    window already accepted a site never pay the escalation.
    """
    from catopt_core.egraph.terms import _term_match

    from catopt_discovery import oracle as lvo
    from catopt_discovery.shape_proposal import Schema, real_matches

    schema = Schema(rule.name, rule.lhs, rule.rhs)
    matches = real_matches(list(corpus.real_terms), schema)
    real = _guarded_evals(
        rule,
        (
            (s, m)
            for m in matches
            if (s := _term_match(rule.lhs, m)) is not None
        ),
    )
    synth = _guarded_evals(
        rule, _synth_sites(rule.lhs, rule.rhs, limit=limit)
    )
    eff = lvo.escalate_limit(
        limit, guarded=True, accepted=synth.accepted
    )
    if eff > limit:
        wider = _guarded_evals(
            rule, _synth_sites(rule.lhs, rule.rhs, limit=eff)
        )
        synth = replace(wider, envs=synth.envs + wider.envs)
    return synth, real


def _gate(rep: Gauntlet, name: str, ok: bool, detail: str = "") -> bool:
    """Append one stage verdict to *rep*; return whether it passed."""
    rep.stages = (*rep.stages, GauntletStage(name, bool(ok), detail))
    return ok


def _finish(rep: Gauntlet) -> Gauntlet:
    """Seal the report: ``usable`` iff every recorded stage passed."""
    rep.usable = bool(rep.stages) and all(s.passed for s in rep.stages)
    return rep


def _reconstruct_gate(
    conn: sqlite3.Connection, rep: Gauntlet
) -> tuple[Any, dict] | None:
    """Stages 1-2: the record must rebuild, and rebuild *fully*.

    An unknown kind or a missing row is a reconstruction failure; a
    record that dropped hooks admits a weaker rule than it declares —
    the full-data gate refuses to call that usable.  Returns
    ``(rule, record)`` on success, ``None`` after a failed gate.
    """
    try:
        got = admit_object(conn, rep.alpha_key)
    except ValueError as exc:
        _gate(rep, "reconstruct", False, str(exc))
        return None
    if got is None:
        _gate(rep, "reconstruct", False, "no object under this key")
        return None
    rule, record = got
    rep.rule, rep.record = rule, record
    rep.name, rep.kind = rule.name, record["kind"]
    _gate(rep, "reconstruct", True, f"{record['kind']} rebuilds")
    missing = list(record["missing_hooks"])
    ok = _gate(
        rep,
        "full-data",
        not missing,
        "the record carries every hook"
        if not missing
        else "dropped hooks: " + ", ".join(missing),
    )
    return (rule, record) if ok else None


def _measure_gate(
    rep: Gauntlet,
    rule: Any,
    record: dict,
    corpus: GauntletCorpus,
    synth_limit: int | None,
) -> bool:
    """Stages 3-4: ``pipeline.measure`` on the rebuilt rule, then truth.

    The rebuilt rule's ``check``/``derive`` hooks ride along on the
    ``Proposal``, so every stage measure runs — the numeric oracle,
    derivability, the raw view-oracle verdict, the firing probe and
    the typedness audit — sees the *guarded* rule, not the bare
    pattern.  Truth for a guarded object lives in the region its
    ``cond`` accepts (:func:`_guarded_truth`); the raw view verdict
    stays ``conditional`` there by construction, so it is reported
    but the sweep is the gate.
    """
    from catopt_discovery import pipeline as pl
    from catopt_discovery import proposal as lp

    prop = pl.Proposal(
        name=rule.name,
        lhs=rule.lhs,
        rhs=rule.rhs,
        family=f"object:{record['kind']}",
        check=rule.check,
        derive=rule.derive,
    )
    lib = [lp._key(r.lhs, r.rhs) for r in corpus.base_rules]
    try:
        evd = pl.measure(
            prop,
            list(corpus.real_terms),
            list(corpus.probe),
            list(corpus.base_rules),
            lib,
            corpus.census_op,
            corpus.sink,
            corpus.cost_fn,
        )
    except Exception as exc:
        _gate(rep, "measure", False, f"{type(exc).__name__}: {exc}")
        return False
    rep.evidence = evd
    _gate(
        rep,
        "measure",
        True,
        f"matches={evd.matches} census={evd.census_sites}",
    )
    return _truth_gate(rep, rule, corpus, synth_limit)


def _truth_gate(
    rep: Gauntlet,
    rule: Any,
    corpus: GauntletCorpus,
    synth_limit: int | None,
) -> bool:
    """Stage 4: the declared equality must hold where the rule fires.

    A measured counterexample outranks a derivation.  Derivability is
    legitimate evidence — a proof from shipped axioms — but it is a
    claim about *every* binding, and a measured ``unequal`` /
    ``rhs_err`` / ``guard_err`` site is a fact about one the guard
    accepts.  The two cannot both stand, so the gate blocks on the
    measured site regardless of ``derivable``; ``derivable`` may only
    waive the *starvation* requirement (that the sweep exhibit at
    least one equal site), never a measured counterexample.  An
    unguarded object reads the same rule off the numeric oracle: a
    measured ``num_true is False`` blocks, and ``derivable`` waives
    only the "no measurement" (``num_true is None``) case.

    *synth_limit* sets the sweep's first-phase window (the oracle's
    default when ``None``); a starved window escalates once under the
    selective-cap policy (:func:`catopt_discovery.oracle.escalate_limit`)
    — see :func:`_guarded_truth`.
    """
    from catopt_discovery import oracle as lvo

    evd = rep.evidence
    if rule.cond is None and rule.check is None:
        ok = evd.num_true is not False and (
            evd.derivable or evd.num_true is True
        )
        return _gate(rep, "truth", ok, _truth_detail(evd))
    limit = (
        synth_limit if synth_limit is not None else lvo._MAX_INSTANCES
    )
    rep.synth_region, rep.real_region = _guarded_truth(
        rule, corpus, limit
    )
    clean = _region_clean(rep.synth_region) and _region_clean(
        rep.real_region
    )
    ok = clean and (evd.derivable or rep.synth_region.equal >= 1)
    detail = (
        _truth_detail(evd)
        + " | guarded: "
        + _region_detail(rep.synth_region, rep.real_region)
    )
    if not clean and evd.derivable:
        detail += (
            " | derivation overridden by a measured counterexample"
        )
    return _gate(rep, "truth", ok, detail)


def _typed_pay_gate(rep: Gauntlet) -> bool:
    """Stage 6: fires, all well-typed, pays, lowered modules agree."""
    evd = rep.evidence
    return _gate(
        rep,
        "typed-pay",
        evd.fires > 0
        and evd.fires_ill_typed == 0
        and evd.paid > 0
        and evd.verify_fail == 0,
        f"fires={evd.fires} typed={evd.fires_typed} "
        f"ill={evd.fires_ill_typed} paid={evd.paid} "
        f"verify_fail={evd.verify_fail}",
    )


def _closure_gate(rep: Gauntlet) -> bool:
    """Stage 7: bounded closure and no object-attributable cert break.

    ``evd.cert_fail`` counts every ``add_cert`` failure over the probe
    corpus — including cases the object never fired on and whose
    *base* replay already fails (an ambient corpus instability).  The
    gate the honest contract asks is narrower: failures *attributable*
    to the object — a case whose baseline certificate replayed but
    whose with-rule one did not.  Ambient failures are reported, not
    charged.
    """
    evd = rep.evidence
    cert_att = sum(
        1
        for r in evd.reach
        if r["add_cert"] != "pass" and r["base_cert"] == "pass"
    )
    return _gate(
        rep,
        "closure",
        cert_att == 0 and evd.reach_ill == 0 and evd.closure_safe,
        f"enode={evd.closure_ratio:.2f}x cert_fail={cert_att} "
        f"(ambient={evd.cert_fail - cert_att}) "
        f"reach_ill={evd.reach_ill}",
    )


def _cert_gate(rep: Gauntlet, record: dict) -> bool:
    """Stage 8: a record carrying a derivation replays it strictly."""
    try:
        cert = stored_certificate(record)
    except Exception as exc:
        return _gate(rep, "cert", False, f"strict replay failed: {exc}")
    return _gate(
        rep,
        "cert",
        True,
        "no derivation recorded"
        if cert is None
        else f"{cert.n_steps}-step cert replays strict",
    )


def _auto_cond_retry(
    conn: sqlite3.Connection,
    rep: Gauntlet,
    corpus: GauntletCorpus,
    synth_limit: int | None,
    max_clauses: int,
) -> Gauntlet:
    """Attempt the conditional→guarded rewrite after a refusal.

    Called when a stored object fails the gauntlet on truth (the
    candidate measured *conditional*) or on full-data with exactly the
    ``check`` hook missing (the reconstructed rule is already the bare
    pattern — the missing ``check`` is what a found ``cond`` would
    carry).  ``object_synthesis.auto_cond_object`` searches the
    declarative guard vocabulary for the smallest conjunction covering
    the measured equal sites and declining the measured bad ones; on a
    hit the object record is rewritten — same alpha key, now carrying
    the found ``cond`` and ``kind="abstraction"`` — and the gauntlet
    re-runs on the rewritten record (``auto_cond=False``: one retry,
    not a chase).

    The returned report is the *second* run's, with an ``auto-cond``
    stage prepended carrying the search's detail; the refusal case
    keeps the original report, annotated in ``rep.auto_cond``.
    """
    from catopt_core.laws.cond import cond_to_data

    from catopt_discovery import object_synthesis as obs
    from catopt_discovery import oracle as lvo

    limit = (
        synth_limit if synth_limit is not None else lvo._MAX_INSTANCES
    )
    res = obs.auto_cond_object(
        rep.rule,
        corpus_terms=corpus.real_terms,
        synth_limit=limit,
        max_clauses=max_clauses,
        kind="abstraction",
    )
    rep.auto_cond = {
        "found": res.object is not None,
        "cond": cond_to_data(res.cond),
        "clauses": [cond_to_data(c) for c in res.clauses],
        "measured": res.measured,
        "equal": res.equal,
        "bad": res.bad,
        "unstable": res.unstable,
        "other": res.other,
        "accepted": res.accepted,
        "accepted_other": res.accepted_other,
        "detail": res.detail,
        "first_reason": rep.reason,
    }
    if res.object is None:
        return _finish(rep)
    row = conn.execute(
        "SELECT corpus_hash FROM lemmas WHERE alpha_key = ?",
        (rep.alpha_key,),
    ).fetchone()
    obs.store_constructed(
        conn, res.object, row["corpus_hash"] if row else ""
    )
    rep2 = run_gauntlet(
        conn, rep.alpha_key, corpus=corpus, synth_limit=synth_limit
    )
    rep2.auto_cond = rep.auto_cond
    rep2.stages = (
        GauntletStage("auto-cond", True, res.detail),
        *rep2.stages,
    )
    return rep2


def _auto_cond_armed(rep: Gauntlet, stage: str) -> bool:
    """Whether the auto-cond retry applies to this refusal.

    ``truth``: the candidate measured conditional — the minted guard
    is the missing piece.  ``full-data``: exactly the ``check`` hook
    is missing — the reconstructed rule is already the bare pattern,
    and the found ``cond`` is what the record was missing.  Anything
    else (a rebuild failure, a missing ``dspec``/``derive``) is not
    an auto-cond question.
    """
    if rep.rule is None:
        return False
    if stage == "truth":
        return bool(rep.stages) and rep.stages[-1].name == "truth"
    return rep.record is not None and list(
        rep.record["missing_hooks"]
    ) == ["check"]


def run_gauntlet(
    conn: sqlite3.Connection,
    alpha_key: str,
    *,
    corpus: GauntletCorpus | None = None,
    synth_limit: int | None = None,
    auto_cond: bool = False,
    auto_cond_clauses: int = 3,
) -> Gauntlet:
    """Run the admission gauntlet on a stored declared object.

    The same gates a shipped law cleared, in the ADR's order:
    reconstruct the record → require full data → measure (numeric
    oracle, derivability, the view oracle's raw verdict) → truth
    (unguarded: the measured verdict; guarded: the guarded-region
    sweep — every accepted evaluable binding must be equal, and at
    least one must exist) → novelty (the object is not a library
    spelling) → typed-pay (fires, all well-typed, pays, lowerings
    agree) → closure (no enode blow-up, in-graph certs replay) → the
    stored derivation cert replays strict when present.  A failing
    gate stops the run — later stages are not reached, not skipped.

    *corpus* defaults to the pipeline's real corpus
    (:func:`default_gauntlet_corpus`); tests inject a small one.
    ``usable`` is ``True`` only when every stage passed — the honest
    contract: the store refuses to call a synthesized object usable
    on reconstruction alone.

    ``auto_cond=True`` arms the conditional→guarded retry
    (:func:`_auto_cond_retry`): a truth refusal (or a ``full-data``
    refusal on a missing ``check`` alone) attempts to mint the
    smallest declarative ``cond`` covering the measured domain, then
    re-runs the gauntlet on the rewritten record.  ``auto_cond_clauses``
    bounds the conjunction the search may mint.

    ``synth_limit`` sets the guarded-region sweep's first-phase
    window (the oracle's default when ``None``).  The selective-cap
    policy applies underneath: a guarded rule whose window accepts
    nothing escalates once to
    :data:`catopt_discovery.oracle._GUARDED_CAP`, so a sound law whose
    accepted corner lies past the default is not refused as vacuous.
    """
    rep = Gauntlet(alpha_key=alpha_key)
    got = _reconstruct_gate(conn, rep)
    if got is None:
        if auto_cond and _auto_cond_armed(rep, "full-data"):
            return _auto_cond_retry(
                conn,
                rep,
                corpus or default_gauntlet_corpus(),
                synth_limit,
                auto_cond_clauses,
            )
        return _finish(rep)
    rule, record = got
    if corpus is None:
        corpus = default_gauntlet_corpus()
    if not _measure_gate(rep, rule, record, corpus, synth_limit):
        if auto_cond and _auto_cond_armed(rep, "truth"):
            return _auto_cond_retry(
                conn, rep, corpus, synth_limit, auto_cond_clauses
            )
        return _finish(rep)
    return _admit_tail(rep, record)


def _admit_tail(rep: Gauntlet, record: dict) -> Gauntlet:
    """Stages 5-8: novelty → typed-pay → closure → cert replay.

    *record* is the stored object record for the cert gate.
    """
    # novelty: a stored object duplicating a library spelling is not
    # a new inhabitant.
    evd = rep.evidence
    if not _gate(
        rep,
        "novelty",
        evd.relation == "new",
        f"relation={evd.relation}",
    ):
        return _finish(rep)
    if not _typed_pay_gate(rep) or not _closure_gate(rep):
        return _finish(rep)
    _cert_gate(rep, record)
    return _finish(rep)


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
        print(f"no shipped law named {name!r}")  # stdout-compat
        return 1
    key = store_object(conn, rule, kind=kind)
    # the row was just written — the read-back cannot be empty
    record = cast(dict, stored_object(conn, key))
    state = (
        "full-data"
        if record["serializable"]
        else "missing hooks: " + ", ".join(record["missing_hooks"])
    )
    print(f"stored {rule.name}  [{state}]")  # stdout-compat
    print(f"  kind: {record['kind']}")  # stdout-compat
    print(f"  alpha_key = {key}")  # stdout-compat
    cert = record.get("cert")
    if cert is None:
        print("  cert: none recorded")  # stdout-compat
    else:
        print(  # stdout-compat
            f"  cert: {len(cert['steps'])}-step derivation"
            f" {cert['rules_used']}"
        )
    return 0


def _admit_object_cli(
    conn: sqlite3.Connection,
    alpha_key: str,
    gauntlet: bool = False,
    auto_cond: bool = False,
) -> int:
    """Rebuild a stored object into a live ``Rewrite`` and show it.

    With *gauntlet* the admit runs :func:`run_gauntlet` instead —
    the adversarial admission stages — and reports ``usable`` only
    when every gate passed (exit 1 otherwise).  *auto_cond* arms the
    conditional→guarded retry inside the gauntlet.
    """
    if gauntlet:
        return _gauntlet_cli(conn, alpha_key, auto_cond)
    try:
        got = admit_object(conn, alpha_key)
    except ValueError as exc:
        print(  # stdout-compat
            f"cannot admit object under {alpha_key}: {exc}"
        )
        return 1
    if got is None:
        print(f"no lemma stored under {alpha_key}")  # stdout-compat
        return 1
    rule, data = got
    state = (
        "full-data"
        if data["serializable"]
        else "missing hooks: " + ", ".join(data["missing_hooks"])
    )
    print(f"admitted {rule.name}  [{state}]")  # stdout-compat
    print(f"  kind: {data['kind']}")  # stdout-compat
    print(f"  {rule!r}")  # stdout-compat
    try:
        cert = stored_certificate(data)
    except Exception as exc:
        print(f"  cert: STRICT REPLAY FAILED: {exc}")  # stdout-compat
        return 1
    if cert is None:
        print("  cert: none recorded")  # stdout-compat
    else:
        print(  # stdout-compat
            f"  cert: {cert.n_steps}-step {cert.rules_used}"
            " — replayed strict"
        )
    return 0


def _gauntlet_cli(
    conn: sqlite3.Connection, alpha_key: str, auto_cond: bool = False
) -> int:
    """Run the admission gauntlet on a stored object; report usable."""
    rep = run_gauntlet(conn, alpha_key, auto_cond=auto_cond)
    if rep.rule is None or rep.record is None:
        print(  # stdout-compat
            f"cannot admit object under {alpha_key}: {rep.reason}"
        )
        return 1
    state = (
        "full-data"
        if rep.record["serializable"]
        else "missing hooks: " + ", ".join(rep.record["missing_hooks"])
    )
    print(f"admitted {rep.name}  [{state}]")  # stdout-compat
    print(f"  kind: {rep.kind}")  # stdout-compat
    print(f"  {rep.rule!r}")  # stdout-compat
    print("-- admission gauntlet --")  # stdout-compat
    for s in rep.stages:
        mark = "pass" if s.passed else "FAIL"
        print(f"  [{mark}] {s.name}: {s.detail}")  # stdout-compat
    print(  # stdout-compat
        f"  usable: {'yes' if rep.usable else 'no'} — {rep.reason}"
    )
    return 0 if rep.usable else 1


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
    object one.  ``--gauntlet`` modifies the admit ops only.
    """
    if (
        args.gauntlet
        and args.admit is None
        and args.admit_object is None
    ):
        print(  # stdout-compat
            "--gauntlet needs --admit or --admit-object"
        )
        return 1
    if args.add_lemma is not None:
        return _store_object_cli(conn, args.add_lemma, args.kind)
    if args.add_object is not None:
        return _store_object_cli(conn, args.add_object, args.kind)
    if args.admit is not None:
        return _admit_object_cli(
            conn, args.admit, args.gauntlet, args.auto_cond
        )
    if args.admit_object is not None:
        return _admit_object_cli(
            conn, args.admit_object, args.gauntlet, args.auto_cond
        )
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
    parser.add_argument(
        "--gauntlet",
        action="store_true",
        help="with --admit/--admit-object: run the admission gauntlet"
        " (numeric oracle -> guarded-region truth -> typed-pay ->"
        " closure -> cert replay) and report 'usable' only when every"
        " stage passed",
    )
    parser.add_argument(
        "--auto-cond",
        action="store_true",
        help="with --admit/--admit-object --gauntlet: on a truth"
        " (or missing-check full-data) refusal, attempt to mint the"
        " smallest declarative cond covering the measured domain,"
        " rewrite the record as kind=abstraction, and re-run the"
        " gauntlet",
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
        print(f"no evidence store at {args.report}")  # stdout-compat
        return 1
    conn = connect(args.report)
    try:
        rc = _op_dispatch(args, conn)
        if rc is not None:
            return rc
        print(  # stdout-compat
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
