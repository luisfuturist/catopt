"""Cells — the fragment-merged store behind every vocabulary.

The project's five spellings of "a cell as data" — ``Rewrite``
laws, ``ConstructedObject`` declarations, ``OpDef`` ops, handler
tables, and evidence-store rows — are views over one record.  A
cell is *role + name + body + provenance*; :func:`merge_fragments`
assembles named :class:`Fragment` bundles into a
:class:`CellStore`, and every engine reads the same store.

This is the handl pattern: language behavior is *data merged at
assembly*, not a parallel subsystem per spelling.  Conflicts
surface at merge (a duplicate ``(role, name)`` is a report entry —
and by default a ``ValueError``, not a silent override), and
alpha-equal laws across fragments are reported honestly:
subsumption is not illegal, but hiding it would be.

Roles — the vocabulary a cell extends:

``"op"``
    An :class:`catopt_core.opdata.OpDef` body — an operation
    declared as data (the Sanada *operation*).
``"law"``
    A :class:`catopt_core.egraph.Rewrite` body — a shipped or
    admitted 2-cell (``kind`` is the record's ``"law"`` mark).
``"object"``
    A ``Rewrite`` body with a non-law :data:`OBJECT_KINDS`
    declaration — ``"abstraction"`` / ``"bridge"``.  Objects are
    stage-1 Rewrite-shaped; the kind is provenance, not payload.
``"handle"``
    A handler-table entry ``{"pattern", "kernel", "args"}`` — a
    denotation binding (the Sanada *handler*): one description,
    this interpretation.

Provenance is *where the cell came from*, not what it is:
``"shipped"`` for the packaged vocabulary, ``"constructed"`` /
``"admitted"`` for machine-built cells (with the construction move
and premise names recorded), ``"minted"`` for generated code cells,
``"declared"`` for user declarations.

What is deliberately *not* here: the engines.  ``CellStore`` is
the assembled vocabulary — saturation, costing, verification, and
play are views downstream code computes over it, matching the
semantics/search/evaluation/execution split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.ir import (
    Const,
    Op,
    Param,
    Var,
    term_from_data,
    term_to_data,
)
from catopt_core.laws.ruleset import RuleSet
from catopt_core.laws.serialize import (
    OBJECT_KINDS,
    alpha_key,
    law_from_data,
    law_to_data,
    object_from_data,
    object_to_data,
)
from catopt_core.opdata import OpDef, opdef_from_data, opdef_to_data

__all__ = [
    "CELL_ROLES",
    "OBJECT_CELL_KINDS",
    "ORIGINS",
    "Cell",
    "CellStore",
    "Fragment",
    "MergeReport",
    "Provenance",
    "cell_from_data",
    "cell_to_data",
    "handle_cell",
    "law_cell",
    "laws_fragment",
    "merge_fragments",
    "object_cell",
    "op_cell",
    "rewrite_of",
    "spec_from_data",
    "spec_to_data",
]

# ---------------------------------------------------------------------------
#  The cell record
# ---------------------------------------------------------------------------

#: The cell roles — which vocabulary a cell extends.
CELL_ROLES: frozenset[str] = frozenset(
    ("op", "law", "object", "handle")
)

#: Object-cell kinds — the non-law OBJECT_KINDS a declared object
#: may carry.  ``"law"``-kind records are role ``"law"`` cells.
OBJECT_CELL_KINDS: frozenset[str] = frozenset(
    k for k in OBJECT_KINDS if k != "law"
)

#: Provenance origins — where a cell entered the store.
ORIGINS: frozenset[str] = frozenset(
    ("shipped", "constructed", "admitted", "minted", "declared")
)

_SHIPPED = "shipped"


@dataclass(frozen=True)
class Provenance:
    """Where a cell came from — the audit trail, not the payload.

    ``origin`` is the admission channel (``ORIGINS``); ``op`` is the
    construction move that produced it (``"compose"`` / ``"fold"`` /
    ``"claim"`` / …) when the origin is machine-driven; ``premises``
    names the cells it was built from — the record
    ``ConstructedObject.construction`` carries, flattened into the
    same shape every role uses.  ``note`` is free-form evidence.
    """

    origin: str = _SHIPPED
    op: str = ""
    premises: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the origin and normalise premises to a tuple."""
        if self.origin not in ORIGINS:
            raise ValueError(
                f"unknown provenance origin {self.origin!r}; "
                f"expected one of {sorted(ORIGINS)}"
            )
        object.__setattr__(self, "premises", tuple(self.premises))


@dataclass(frozen=True)
class Cell:
    """One element of the vocabulary: role + name + body + provenance.

    ``role`` is which vocabulary the cell extends (``CELL_ROLES``);
    ``name`` is its identity within ``(fragment, role)`` — rule name,
    op name, or handler kernel name.  ``body`` is the payload in the
    role's native record type — ``OpDef`` / ``Rewrite`` /
    ``Rewrite`` / handler dict — so a cell never *re*-encodes what a
    spelling already says; it only normalises how the store keys and
    audits it.

    ``kind`` is meaningful for ``role="object"`` only: the
    declaration kind from :data:`OBJECT_CELL_KINDS`.  A ``"law"``-kind
    record is simply a ``role="law"`` cell — one spelling per
    distinction.
    """

    role: str
    name: str
    body: Any
    kind: str = ""
    provenance: Provenance = Provenance()

    def __post_init__(self) -> None:
        """Validate role/kind consistency and the body's record type."""
        if self.role not in CELL_ROLES:
            raise ValueError(
                f"unknown cell role {self.role!r}; "
                f"expected one of {sorted(CELL_ROLES)}"
            )
        if self.role == "object":
            if self.kind not in OBJECT_CELL_KINDS:
                raise ValueError(
                    f"object cell {self.name!r} needs a kind in "
                    f"{sorted(OBJECT_CELL_KINDS)}, got {self.kind!r}"
                )
        elif self.kind:
            raise ValueError(
                f"{self.role} cell {self.name!r} carries no kind "
                f"(kind is object provenance only), got {self.kind!r}"
            )
        want = {
            "op": OpDef,
            "law": Rewrite,
            "object": Rewrite,
            "handle": dict,
        }[self.role]
        if not isinstance(self.body, want):
            raise TypeError(
                f"{self.role} cell {self.name!r} needs a "
                f"{want.__name__} body, got {type(self.body).__name__}"
            )
        if self.role == "handle":
            missing = {"pattern", "kernel", "args"} - set(self.body)
            if missing:
                raise ValueError(
                    f"handle cell {self.name!r} is missing handler "
                    f"keys {sorted(missing)}"
                )


# ---------------------------------------------------------------------------
#  Fragments — named cell bundles
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fragment:
    """A named bundle of cells — one vocabulary slice.

    Fragments are the merge unit: ``"core_laws"`` (the shipped
    library), ``"machine_store"`` (admitted objects), ``"gen"``
    (minted handlers), ``"declared"`` (user ops), an episode's
    constructions.  Names are labels for the merge report and the
    per-cell fragment stamp the store keeps, not identity —
    ``(role, name)`` is.
    """

    name: str
    cells: tuple[Cell, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        """Normalise cells and reject duplicates inside the fragment."""
        cells = tuple(self.cells)
        object.__setattr__(self, "cells", cells)
        seen: set[tuple[str, str]] = set()
        for c in cells:
            key = (c.role, c.name)
            if key in seen:
                raise ValueError(
                    f"duplicate cell {key} in fragment {self.name!r}"
                )
            seen.add(key)


@dataclass(frozen=True)
class MergeReport:
    """What :func:`merge_fragments` saw assembling the store.

    ``conflicts`` — ``(role, name)`` keys claimed by more than one
    fragment.  With ``on_conflict="error"`` this is nonempty only on
    the keep policies; under ``"error"`` the first conflict raises
    before the store assembles.  ``alpha_dups`` — ``(name_a, name_b)``
    law/object pairs whose ``(lhs, rhs)`` alpha-keys coincide across
    fragments: reported, never fatal — subsumption is a discovery
    fact, not a schema error.  ``counts`` — per-role totals in the
    merged store.
    """

    conflicts: tuple[tuple[str, str], ...] = ()
    alpha_dups: tuple[tuple[str, str], ...] = ()
    counts: Mapping[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Cell converters — the five spellings into one record
# ---------------------------------------------------------------------------


def law_cell(
    rule: Rewrite, *, provenance: Provenance = Provenance()
) -> Cell:
    """Wrap a ``Rewrite`` as a ``role="law"`` cell."""
    return Cell(
        role="law", name=rule.name, body=rule, provenance=provenance
    )


def object_cell(
    rule: Rewrite,
    *,
    kind: str,
    provenance: Provenance = Provenance("constructed"),
) -> Cell:
    """Wrap a declared object (a ``Rewrite``-shaped 2-cell) as a cell."""
    return Cell(
        role="object",
        name=rule.name,
        body=rule,
        kind=kind,
        provenance=provenance,
    )


def op_cell(
    opdef: OpDef, *, provenance: Provenance = Provenance("declared")
) -> Cell:
    """Wrap an ``OpDef`` declaration as a ``role="op"`` cell."""
    return Cell(
        role="op", name=opdef.name, body=opdef, provenance=provenance
    )


def handle_cell(
    name: str,
    handler: Mapping[str, Any],
    *,
    provenance: Provenance = Provenance("minted"),
) -> Cell:
    """Wrap a handler-table entry as a ``role="handle"`` cell.

    *name* is the cell's identity — the kernel name is the honest
    choice for generated entries (``gen_<i>_k`` / ``scan_<i>_k``);
    *handler* is the ``{"pattern", "kernel", "args"}`` dict the
    binding builders consume.
    """
    return Cell(
        role="handle",
        name=name,
        body=dict(handler),
        provenance=provenance,
    )


def rewrite_of(cell: Cell) -> Rewrite:
    """Return the ``Rewrite`` a law or object cell carries.

    Views exist so engines keep one convention — the e-graph's rule
    protocol — without knowing which spelling admitted the 2-cell.
    """
    if cell.role not in ("law", "object"):
        raise TypeError(
            f"cell {cell.name!r} (role {cell.role!r}) carries no "
            f"Rewrite body"
        )
    return cell.body


def laws_fragment(
    name: str,
    rules: Iterable[Rewrite],
    *,
    provenance: Provenance = Provenance(),
    note: str = "",
) -> Fragment:
    """Bundle a rule iterable as a law-cell fragment.

    The standard fragment source for shipped rule tuples
    (``ALL_RULES``) and admission stores — each rule becomes a
    ``role="law"`` cell under *name*.
    """
    return Fragment(
        name,
        tuple(law_cell(r, provenance=provenance) for r in rules),
        note=note,
    )


# ---------------------------------------------------------------------------
#  The store — the merged vocabulary
# ---------------------------------------------------------------------------


class CellStore:
    """The fragment assembly every engine reads.

    Cells are keyed ``(role, name)`` — the merge has already run, so
    the store is flat and immutable-by-convention: extend it with
    :meth:`merge`, which re-runs validation rather than mutating.
    ``fragments`` names the sources folded in (assembly order is the
    conflict-resolution order for keep policies); ``report`` is the
    :class:`MergeReport` the merge produced.

    Views are the interface engines use:

    * :meth:`laws` / :meth:`ruleset` — the 2-cells, as ``Rewrite`` values
      (law *and* object roles — both are Rewrite-shaped) or as a
      :class:`RuleSet` for ``EGraph.run``.
    * :meth:`opdefs` — the declared ops, for the opmeta registry.
    * :meth:`handlers` — the handler dicts, keyed by kernel name, in
      the shape the binding builders already consume.
    * :meth:`to_data` / :meth:`from_data` — one schema for the whole
      vocabulary; per-body codecs are the existing law/object/opdef
      ones, so a serialized law cell round-trips through the same
      honesty flags ``law_to_data`` already reports.
    """

    def __init__(
        self,
        cells: Mapping[tuple[str, str], Cell],
        fragments: tuple[str, ...],
        report: MergeReport,
        fragment_of: Mapping[tuple[str, str], str],
    ) -> None:
        """Assemble the store — use :func:`merge_fragments` normally."""
        self._cells = dict(cells)
        self.fragments = tuple(fragments)
        self.report = report
        self._fragment_of = dict(fragment_of)

    # -- lookup ------------------------------------------------------

    def __contains__(self, key: tuple[str, str]) -> bool:
        """``(role, name)`` membership."""
        return key in self._cells

    def __len__(self) -> int:
        """Cell count."""
        return len(self._cells)

    def __iter__(self) -> Any:
        """Iterate cells in assembly order."""
        return iter(self._cells.values())

    def get(self, role: str, name: str, default: Any = None) -> Any:
        """Return the ``(role, name)`` cell, or *default*."""
        return self._cells.get((role, name), default)

    def fragment_of(self, cell: Cell) -> str:
        """Return the fragment a cell arrived in (assembly provenance)."""
        return self._fragment_of.get((cell.role, cell.name), "")

    # -- views --------------------------------------------------------

    def cells(self, role: str) -> tuple[Cell, ...]:
        """All cells of one role, in assembly order."""
        return tuple(c for c in self._cells.values() if c.role == role)

    def laws(self) -> tuple[Rewrite, ...]:
        """Every 2-cell — law *and* object roles — as ``Rewrite`` values."""
        return tuple(
            rewrite_of(c)
            for c in self._cells.values()
            if c.role in ("law", "object")
        )

    def objects(self, kind: str | None = None) -> tuple[Cell, ...]:
        """Object cells, optionally filtered to one declaration kind."""
        objs = self.cells("object")
        if kind is None:
            return objs
        return tuple(c for c in objs if c.kind == kind)

    def opdefs(self) -> tuple[OpDef, ...]:
        """Return the declared ops."""
        return tuple(c.body for c in self.cells("op"))

    def handlers(self) -> dict[str, dict]:
        """Return the handler table, keyed by kernel name."""
        return {
            c.body["kernel"]: dict(c.body) for c in self.cells("handle")
        }

    def ruleset(self, name: str = "cellstore") -> RuleSet:
        """Return the 2-cells as a :class:`RuleSet` for ``EGraph.run``."""
        return RuleSet(name, rules=self.laws())

    # -- merge and codec ------------------------------------------------

    def merge(
        self, *fragments: Fragment, on_conflict: str = "error"
    ) -> CellStore:
        """Fold new fragments into a fresh store (re-running validation)."""
        base = Fragment(
            "+".join(self.fragments) or "store",
            tuple(self._cells.values()),
        )
        return merge_fragments(
            base, *fragments, on_conflict=on_conflict
        )

    def to_data(self) -> dict[str, Any]:
        """Serialise every cell through the one schema."""
        return {
            "fragments": list(self.fragments),
            "cells": [
                dict(
                    cell_to_data(c),
                    fragment=self._fragment_of.get(
                        (c.role, c.name), ""
                    ),
                )
                for c in self._cells.values()
            ],
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> CellStore:
        """Rebuild a store from :meth:`to_data` output."""
        by_fragment: dict[str, list[Cell]] = {}
        order: list[str] = []
        for rec in data.get("cells", ()):
            frag = rec.get("fragment", "")
            if frag not in by_fragment:
                by_fragment[frag] = []
                order.append(frag)
            by_fragment[frag].append(cell_from_data(rec))
        if not order:
            order = list(data.get("fragments", ()))
        frags = tuple(
            Fragment(f, tuple(by_fragment.get(f, ()))) for f in order
        )
        return merge_fragments(*frags, on_conflict="error")


# ---------------------------------------------------------------------------
#  The merge — assembly with validation
# ---------------------------------------------------------------------------


def merge_fragments(
    *fragments: Fragment, on_conflict: str = "error"
) -> CellStore:
    """Assemble fragments into a :class:`CellStore`.

    ``on_conflict`` resolves ``(role, name)`` collisions across
    fragments: ``"error"`` raises ``ValueError`` (the default — a
    silent override is how vocabularies drift), ``"keep_first"``
    keeps the earliest fragment's cell, ``"keep_last"`` the latest.
    Every conflict is recorded in the report under any policy.

    Alpha-equal 2-cells across fragments are *reported* in
    ``report.alpha_dups`` — an admission that duplicates shipped
    semantics is a fact the discovery audit needs, not a schema
    error.
    """
    if on_conflict not in ("error", "keep_first", "keep_last"):
        raise ValueError(f"unknown conflict policy {on_conflict!r}")
    cells: dict[tuple[str, str], Cell] = {}
    fragment_of: dict[tuple[str, str], str] = {}
    conflicts: list[tuple[str, str]] = []
    names: list[str] = []
    for frag in fragments:
        names.append(frag.name)
        for cell in frag.cells:
            key = (cell.role, cell.name)
            if key in cells:
                conflicts.append(key)
                if on_conflict == "error":
                    raise ValueError(
                        f"cell {key} claimed by fragment "
                        f"{frag.name!r} and "
                        f"{fragment_of[key]!r}"
                    )
                if on_conflict == "keep_last":
                    cells[key] = cell
                    fragment_of[key] = frag.name
                continue
            cells[key] = cell
            fragment_of[key] = frag.name
    alpha: dict[tuple[Any, Any], str] = {}
    alpha_dups: list[tuple[str, str]] = []
    for cell in cells.values():
        if cell.role not in ("law", "object"):
            continue
        k = alpha_key(cell.body.lhs, cell.body.rhs)
        if k in alpha:
            alpha_dups.append((alpha[k], cell.name))
        else:
            alpha[k] = cell.name
    counts: dict[str, int] = {}
    for cell in cells.values():
        counts[cell.role] = counts.get(cell.role, 0) + 1
    return CellStore(
        cells,
        tuple(names),
        MergeReport(
            conflicts=tuple(conflicts),
            alpha_dups=tuple(alpha_dups),
            counts=counts,
        ),
        fragment_of,
    )


# ---------------------------------------------------------------------------
#  The codec — cells as data
# ---------------------------------------------------------------------------


def spec_to_data(spec: Any) -> Any:
    """Encode a declarative pattern spec as JSON-safe data.

    The handler-pattern grammar is the same tree ``term_to_data``
    encodes, spelled in spec form: ``str`` metavar leaves,
    numeric ``Const`` leaves, ``(op, *args[, attrs])`` nodes, and
    concrete ``Var`` / ``Param`` / ``Op`` / ``Const`` leaves for
    handlers binding concrete sites.  Each leaf class gets a distinct
    data tag so the decode is unambiguous.
    """
    if isinstance(spec, str):
        return {"mvar": spec}
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return {"const": spec}
    if isinstance(spec, (tuple, list)):
        head, *rest = spec
        attrs = rest[-1] if rest and isinstance(rest[-1], dict) else {}
        args = rest[:-1] if attrs else rest
        return {
            "spec": head,
            "args": [spec_to_data(a) for a in args],
            "attrs": dict(attrs),
        }
    if isinstance(spec, (Var, Param, Op, Const)):
        return {"term": term_to_data(spec)}
    return spec


def spec_from_data(data: Any) -> Any:
    """Decode :func:`spec_to_data` output back to a pattern spec."""
    if not isinstance(data, dict):
        return data
    if "mvar" in data:
        return data["mvar"]
    if "const" in data:
        return data["const"]
    if "spec" in data:
        out: list = [data["spec"]]
        out.extend(spec_from_data(a) for a in data["args"])
        if data.get("attrs"):
            out.append(dict(data["attrs"]))
        return tuple(out)
    if "term" in data:
        return term_from_data(data["term"])
    raise ValueError(f"bad spec encoding: {data!r}")


def cell_to_data(cell: Cell) -> dict[str, Any]:
    """Serialise one cell — the role-specific codec inside one schema.

    The ``"body"`` field is the existing codec's output:
    ``law_to_data`` / ``object_to_data`` / ``opdef_to_data`` for the
    Rewrite/OpDef roles, a plain ``{"pattern", "kernel", "args"}``
    record (pattern through :func:`spec_to_data`) for handlers.  The
    law codecs already flag non-serializable procedural hooks — that
    honesty flag rides through unchanged at ``body.serializable``.
    """
    if cell.role == "op":
        body = opdef_to_data(cell.body)
    elif cell.role == "object":
        body = object_to_data(cell.body, kind=cell.kind)
    elif cell.role == "law":
        body = law_to_data(cell.body)
    else:
        body = {
            "pattern": spec_to_data(cell.body["pattern"]),
            "kernel": cell.body["kernel"],
            "args": list(cell.body["args"]),
        }
    prov = cell.provenance
    return {
        "role": cell.role,
        "name": cell.name,
        "kind": cell.kind,
        "provenance": {
            "origin": prov.origin,
            "op": prov.op,
            "premises": list(prov.premises),
            "note": prov.note,
        },
        "body": body,
    }


def cell_from_data(data: Mapping[str, Any]) -> Cell:
    """Rebuild a cell from :func:`cell_to_data` output.

    Object kinds are validated at decode — an unrecognised kind is a
    ``ValueError``, not a silent admit — and a ``"law"``-kind object
    record rebuilds as a law cell (one spelling per distinction).
    """
    role = data["role"]
    body = data["body"]
    prov_data = data.get("provenance", {})
    prov = Provenance(
        origin=prov_data.get("origin", _SHIPPED),
        op=prov_data.get("op", ""),
        premises=tuple(prov_data.get("premises", ())),
        note=prov_data.get("note", ""),
    )
    if role == "op":
        return Cell(
            role, data["name"], opdef_from_data(body), provenance=prov
        )
    if role == "object":
        return Cell(
            role,
            data["name"],
            object_from_data(body),
            kind=data.get("kind") or body.get("kind", ""),
            provenance=prov,
        )
    if role == "law":
        return Cell(
            role, data["name"], law_from_data(body), provenance=prov
        )
    if role == "handle":
        return Cell(
            role,
            data["name"],
            {
                "pattern": spec_from_data(body["pattern"]),
                "kernel": body["kernel"],
                "args": tuple(body["args"]),
            },
            provenance=prov,
        )
    raise ValueError(f"unknown cell role in data: {role!r}")
