"""Cell adapters — the discovery spellings into the store.

``catopt_core.cells`` defines the record and the merge; this
module holds the adapters that turn the discovery side's real
vocabularies into fragments:

* :func:`cell_of_constructed` / :func:`constructed_fragment` — a
  ``ConstructedObject`` becomes an object cell carrying its
  construction trace as :class:`~catopt_core.cells.Provenance`
  (the move in ``op``, the premise names in ``premises``).
* :func:`handlers_fragment` — a handler table
  (``lawdata.HANDLERS`` / ``genkernel.gen_handlers`` /
  ``scan_handlers``) becomes handle cells, one per entry.
* :func:`machine_fragment` — evidence-store rows become cells:
  each ``law_json`` object record rebuilds its ``Rewrite`` through
  the same codec admission uses, its ``"kind"`` selects law vs
  object role, and the record's ``"provenance"`` field (written by
  ``store_object(..., provenance=)``) restores the construction
  trace — older rows without one report ``origin="admitted"``
  honestly: they entered the store, the trace was not kept.
* :func:`ambient_store` — the standard assembly: shipped laws +
  shipped handlers (+ a machine store when a connection is given)
  merged in one call — the vocabulary a board or probe reads.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from catopt_core.cells import (
    Cell,
    CellStore,
    Fragment,
    Provenance,
    handle_cell,
    law_cell,
    laws_fragment,
    merge_fragments,
    object_cell,
)

__all__ = [
    "ambient_store",
    "cell_of_constructed",
    "constructed_fragment",
    "handlers_fragment",
    "machine_fragment",
]


def _construction_prov(construction: Iterable[str], note: str) -> Any:
    """Flatten ``(move, *premises)`` into a Provenance record."""
    parts = tuple(construction)
    return Provenance(
        "constructed",
        op=parts[0] if parts else "",
        premises=parts[1:],
        note=note,
    )


def cell_of_constructed(obj: Any) -> Cell:
    """Wrap a ``ConstructedObject`` as a provenance-carrying cell."""
    return object_cell(
        obj.rule,
        kind=obj.kind,
        provenance=_construction_prov(obj.construction, obj.note),
    )


def constructed_fragment(name: str, objects: Iterable[Any]) -> Fragment:
    """Bundle constructed objects as an object-cell fragment."""
    return Fragment(
        name, tuple(cell_of_constructed(o) for o in objects)
    )


def handlers_fragment(
    name: str,
    handlers: Mapping[str, dict],
    *,
    provenance: Provenance = Provenance(),
) -> Fragment:
    """Bundle a handler table as a handle-cell fragment.

    One cell per entry, named by kernel — the same identity the
    binding builders (``gen_bindings`` / ``scan_triton_bindings``)
    key their callables by.  *provenance* stamps the whole table:
    shipped for the ambient menu, minted for generated tables.
    """
    return Fragment(
        name,
        tuple(
            handle_cell(h["kernel"], h, provenance=provenance)
            for h in handlers.values()
        ),
    )


def machine_fragment(conn: Any, name: str = "machine") -> Fragment:
    """Rebuild the evidence store's admitted rows as a fragment.

    Every ``lemmas`` row becomes a cell: ``kind="law"`` records are
    law cells, the non-law kinds are object cells of that kind.  A
    record's ``"provenance"`` field restores the construction trace
    written at admission; rows that predate it keep the honest
    ``origin="admitted"`` — the store is the one place their
    provenance is known.
    """
    from catopt_core.laws.serialize import object_from_data

    from catopt_discovery import evidence as ev

    cells: list[Cell] = []
    for row in ev.lemma_rows(conn):
        try:
            record = json.loads(row["law_json"])
            rule = object_from_data(record)
        except (KeyError, TypeError, ValueError):
            continue
        kind = record.get("kind", "law")
        prov = record.get("provenance") or {}
        if prov:
            provenance = Provenance(
                prov.get("origin", "constructed"),
                op=prov.get("op", ""),
                premises=tuple(prov.get("premises", ())),
                note=prov.get("note", ""),
            )
        else:
            provenance = Provenance("admitted")
        if kind == "law":
            cells.append(law_cell(rule, provenance=provenance))
        else:
            cells.append(
                object_cell(rule, kind=kind, provenance=provenance)
            )
    return Fragment(name, tuple(cells))


def ambient_store(
    rules: Iterable[Any] | None = None,
    handlers: Mapping[str, dict] | None = None,
    *,
    machine_conn: Any = None,
) -> CellStore:
    """Assemble the ambient vocabulary into one store.

    *rules* defaults to the shipped ``ALL_RULES``; *handlers* to
    the shipped ``lawdata.HANDLERS``.  *machine_conn* is an open
    evidence-store connection whose admitted objects merge as the
    ``"machine"`` fragment — the same union the discovery sweeps
    assembled by hand.
    """
    if rules is None:
        from catopt_core.laws import ALL_RULES

        rules = ALL_RULES
    if handlers is None:
        from catopt_discovery import lawdata

        handlers = lawdata.HANDLERS
    frags = [
        laws_fragment("shipped", rules),
        handlers_fragment("handlers", handlers),
    ]
    if machine_conn is not None:
        frags.append(machine_fragment(machine_conn))
    return merge_fragments(*frags)
