"""The machine law pack — the store's admitted objects as a ruleset.

The evidence store (:mod:`catopt_discovery.evidence`) persists
*declared objects* — a ``lemmas`` row per object record
(``laws.serialize.object_to_data``: pattern pair, ``cond``/``dspec``
guards, tags, derivation, cert, and the ``serializable`` /
``missing_hooks`` honesty flags).  The admission gauntlet
(:func:`~catopt_discovery.evidence.run_gauntlet`) decides whether a
stored object is *usable* — but that verdict is a runtime report, not
a stored column: nothing in the schema says which rows cleared.

So the pack's admission filter is the **provenance ledger**:
:data:`catopt_core.laws.provenance.MACHINE_STORED` names the objects
the committed tests pin ``usable`` (``tests/test_discovery_*``) plus
the objects the promotion audit records as admitted
(``project/retros/promoted-laws.md``) — sixteen names, all unshipped.
The ledger deliberately omits the colliding names: the store's
``om_lift`` object shares its name with the human carrier law, and
``softsign_fold`` / ``sdpa_fold_{,div_}nomask`` shipped under their own
names — including them would make ``default + pack`` collide on a
name/different-rule pair (``RuleSet.union`` raises).

:func:`load_pack` is the loader: read the store's object rows, keep
the admitted names, reconstruct each into a live ``Rewrite``
(:func:`~catopt_discovery.evidence.admit_object`), and require the
record to be *full-data* — ``missing_hooks == ()`` on both the record
and the rebuilt rule, so a pack member never fires weaker than it
declares.  A row that cannot reconstruct cleanly is **skipped and
counted**, never fatal: the pack reports what it could not load.  For
a caller that wants the verdict re-measured rather than ledgered,
``gauntlet_corpus=`` re-runs the gauntlet on each candidate row
(truth, novelty, typed-pay, closure, cert — against that corpus) and
keeps only ``usable`` objects; note the novelty gate then refuses
objects whose pattern since shipped as a promoted law.

The pack deploys through the existing ``rules=`` seam —
:func:`search <catopt_orchestrator.optimize.search>` and
:class:`~catopt_orchestrator.optimize.Optimizer` already accept a
:class:`~catopt_core.laws.RuleSet`.  :func:`machine_default` is the
one-call opt-in: the orchestrator's composed ``DEFAULT_RULES`` union
the pack.  Machine objects never enter ``DEFAULT`` itself — the
human-bar audit's provenance separation
(``project/retros/human-bar.md``) stays honest by keeping the pack a
distinct set whose members resolve ``machine`` under
``law_provenance``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.laws import MACHINE_STORED, RuleSet
from catopt_core.laws.serialize import missing_hooks

from catopt_discovery import TOOLS
from catopt_discovery import evidence as ev

__all__ = [
    "DEFAULT_STORE",
    "MachinePack",
    "PackSkip",
    "load_pack",
    "machine_default",
    "store_laws",
    "store_ruleset",
]

#: The committed store path — ``tools/evidence.db`` rides the repo.
#: It is the default *location*, not a guarantee of content: a store
#: with no ``lemmas`` rows packs to the empty set, honestly.
DEFAULT_STORE: Path = TOOLS / "evidence.db"


@dataclass(frozen=True)
class PackSkip:
    """One store row the pack declined, with the reason.

    ``name`` is the row's recorded object name (or the rule's, once
    reconstructed); ``reason`` is the refusal class —
    ``"reconstruct: …"`` for a record the codec could not rebuild,
    ``"missing hooks: …"`` for a non-full-data record,
    ``"duplicate rule name"`` for a second row under an already
    loaded name, ``"gauntlet: …"`` for a re-admission refusal.
    """

    name: str
    alpha_key: str
    reason: str


@dataclass(frozen=True)
class MachinePack:
    """The loaded store pack: the rules plus the honest load report.

    ``rules`` are the reconstructed ``Rewrite`` objects — the pack's
    deployable payload.  ``loaded`` / ``skipped`` / ``excluded``
    account for every ``lemmas`` row the loader saw: *loaded* names
    made the pack; *skipped* rows were candidates that failed to load
    cleanly; *excluded* names were filtered out by the admitted-name
    gate (not a load failure — the store holds more than the ledger).
    ``rows`` is the total row count, so the report sums: ``rows ==
    len(loaded) + len(skipped) + len(excluded)``.
    """

    store: str
    rules: tuple[Rewrite, ...]
    loaded: tuple[str, ...]
    skipped: tuple[PackSkip, ...]
    excluded: tuple[str, ...]
    rows: int

    def ruleset(self, name: str = "machine_store") -> RuleSet:
        """Wrap the pack's rules in a named :class:`RuleSet`."""
        return RuleSet(
            name,
            self.rules,
            description=(
                f"machine-admitted store objects from {self.store} — "
                f"{len(self.rules)} loaded, {len(self.skipped)} "
                f"skipped, {len(self.excluded)} excluded"
            ),
        )


def _open(store: Any) -> tuple[sqlite3.Connection, bool]:
    """Resolve *store* to ``(conn, owned)`` — a path opens, a conn lends.

    A path that names no file raises ``FileNotFoundError`` —
    ``evidence.connect`` would otherwise *create* an empty store, and
    the loader must not mint the thing it reads.
    """
    if isinstance(store, sqlite3.Connection):
        return store, False
    if not Path(store).is_file():
        raise FileNotFoundError(f"no evidence store at {store}")
    return ev.connect(str(store)), True


def _reconstruct(
    conn: sqlite3.Connection, row: dict
) -> tuple[Rewrite, dict] | PackSkip:
    """Rebuild one row's object; return ``(rule, record)`` or a skip."""
    key = row["alpha_key"]
    try:
        got = ev.admit_object(conn, key)
    except Exception as exc:
        return PackSkip(row["name"], key, f"reconstruct: {exc}")
    if got is None:
        return PackSkip(row["name"], key, "reconstruct: row vanished")
    rule, record = got
    missing = sorted(
        set(record["missing_hooks"]) | set(missing_hooks(rule))
    )
    if missing:
        return PackSkip(
            rule.name, key, "missing hooks: " + ", ".join(missing)
        )
    return rule, record


def load_pack(
    store: str | Path | sqlite3.Connection = DEFAULT_STORE,
    *,
    names: Iterable[str] | None = MACHINE_STORED,
    gauntlet_corpus: Any = None,
) -> MachinePack:
    """Load the evidence store's admitted objects as a rule tuple.

    *store* is a path (opened and closed) or an open
    ``sqlite3.Connection`` (borrowed, left open).  *names* is the
    admission filter: the default
    :data:`~catopt_core.laws.provenance.MACHINE_STORED` is the
    provenance ledger of gauntlet-cleared unshipped objects;
    ``None`` trusts the store — every row is a candidate.  A row
    reconstructs through
    :func:`~catopt_discovery.evidence.admit_object` and must be
    full-data (``missing_hooks == ()`` on the record AND the rebuilt
    rule) to join the pack; anything else lands in ``skipped`` with
    its reason.  Duplicate rule names dedupe to the newest row —
    ``lemmas`` rows arrive newest-first.

    *gauntlet_corpus* opts into re-admission: each candidate is
    re-run through :func:`~catopt_discovery.evidence.run_gauntlet` on
    that corpus and kept only when ``usable`` — the strict reading of
    "the store's usable objects" for a store the ledger does not
    cover.  Note the gauntlet's novelty gate refuses objects whose
    pattern has since shipped as a promoted law, so a re-admission
    under the current library can yield fewer members than the
    ledger's snapshot records.
    """
    conn, owned = _open(store)
    label = "<connection>" if owned is False else str(store)
    rules: list[Rewrite] = []
    loaded: list[str] = []
    skipped: list[PackSkip] = []
    excluded: list[str] = []
    admitted = None if names is None else set(names)
    try:
        rows = ev.lemma_rows(conn)
        seen: set[str] = set()
        for row in rows:
            if admitted is not None and row["name"] not in admitted:
                excluded.append(row["name"])
                continue
            got = _reconstruct(conn, row)
            if isinstance(got, PackSkip):
                skipped.append(got)
                continue
            rule, _record = got
            if rule.name in seen:
                skipped.append(
                    PackSkip(
                        rule.name,
                        row["alpha_key"],
                        "duplicate rule name",
                    )
                )
                continue
            if gauntlet_corpus is not None:
                rep = ev.run_gauntlet(
                    conn, row["alpha_key"], corpus=gauntlet_corpus
                )
                if not rep.usable:
                    skipped.append(
                        PackSkip(
                            rule.name,
                            row["alpha_key"],
                            f"gauntlet: {rep.reason}",
                        )
                    )
                    continue
            seen.add(rule.name)
            rules.append(rule)
            loaded.append(rule.name)
    finally:
        if owned:
            conn.close()
    return MachinePack(
        store=label,
        rules=tuple(rules),
        loaded=tuple(loaded),
        skipped=tuple(skipped),
        excluded=tuple(excluded),
        rows=len(rows),
    )


def store_laws(
    store: str | Path | sqlite3.Connection = DEFAULT_STORE,
    *,
    names: Iterable[str] | None = MACHINE_STORED,
    gauntlet_corpus: Any = None,
) -> tuple[Rewrite, ...]:
    """Return the pack's rules as a bare tuple — :func:`load_pack`'s payload."""
    return load_pack(
        store, names=names, gauntlet_corpus=gauntlet_corpus
    ).rules


def store_ruleset(
    store: str | Path | sqlite3.Connection = DEFAULT_STORE,
    *,
    names: Iterable[str] | None = MACHINE_STORED,
    gauntlet_corpus: Any = None,
    name: str = "machine_store",
) -> RuleSet:
    """Return the pack as a named :class:`RuleSet` for ``rules=``."""
    return load_pack(
        store, names=names, gauntlet_corpus=gauntlet_corpus
    ).ruleset(name)


def machine_default(
    store: str | Path | sqlite3.Connection = DEFAULT_STORE,
    *,
    base: Any = None,
    names: Iterable[str] | None = MACHINE_STORED,
    gauntlet_corpus: Any = None,
) -> RuleSet:
    """Return ``DEFAULT_RULES + store pack`` — the deployed opt-in set.

    The Optimizer path's opt-in spelling::

        opt = Optimizer(backend=TorchBackend())
        res = opt.search(model, x, rules=machine_default(db_path))

    *base* is the ruleset the pack unions onto — ``None`` resolves
    the orchestrator's composed ``DEFAULT_RULES`` (imported lazily so
    the pack load itself stays torch-free).  The pack is a distinct
    set, never folded into ``DEFAULT`` — the provenance audit reads
    the boundary.
    """
    if base is None:
        from catopt_orchestrator.optimize import default_rules

        base = default_rules()
    return base + store_ruleset(
        store, names=names, gauntlet_corpus=gauntlet_corpus
    )
