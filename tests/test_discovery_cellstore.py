"""Tests for the discovery-side cell adapters and board store.

``catopt_discovery.cellstore`` turns the constructed-object,
handler-table, and evidence-row spellings into store fragments;
``MetaArena.cellstore`` exposes the board's live vocabulary as a
merged :class:`CellStore`.  The tests cover provenance through
admission — the construction trace the evidence row now keeps.
"""

from __future__ import annotations

import sqlite3

from catopt_core.cells import merge_fragments
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op, TensorType, Var
from catopt_discovery import cellstore as cs
from catopt_discovery import evidence as ev
from catopt_discovery import meta_arena as ma
from catopt_discovery import object_synthesis as obs


def _var(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _sub_gap() -> Op:
    """``add(x, -y)`` — the standard foldable site."""
    return Op.make(
        "add", _var("x", 4, 4), Op.make("neg", _var("y", 4, 4))
    )


def _fold_obj() -> obs.ConstructedObject:
    """A folded object with a real construction trace."""
    return obs.fold_object(
        "myabs",
        ("add", "X", ("neg", "Y")),
        ("myabs", "X", "Y"),
    )


def _conn() -> sqlite3.Connection:
    return ev.connect(":memory:")


# ---------------------------------------------------------------------------
#  Converters
# ---------------------------------------------------------------------------


class TestConstructedCells:
    """ConstructedObject → object cell with its trace intact."""

    def test_cell_of_constructed(self):
        obj = _fold_obj()
        cell = cs.cell_of_constructed(obj)
        assert cell.role == "object" and cell.kind == obj.kind
        assert cell.body is obj.rule
        prov = cell.provenance
        assert prov.origin == "constructed"
        assert prov.op == obj.construction[0]
        assert prov.premises == tuple(obj.construction[1:])

    def test_constructed_fragment(self):
        frag = cs.constructed_fragment("eps", [_fold_obj()])
        assert frag.name == "eps" and len(frag.cells) == 1
        assert frag.cells[0].role == "object"


class TestHandlerCells:
    """A handler table → handle cells keyed by kernel."""

    def test_handlers_fragment(self):
        handlers = {
            "t1": {
                "pattern": ("mul", "a", "b"),
                "kernel": "k1",
                "args": ("a", "b"),
            }
        }
        frag = cs.handlers_fragment("h", handlers)
        assert frag.cells[0].role == "handle"
        assert frag.cells[0].name == "k1"
        assert frag.cells[0].body["kernel"] == "k1"
        store = merge_fragments(frag)
        assert store.handlers()["k1"]["pattern"] == ("mul", "a", "b")


# ---------------------------------------------------------------------------
#  Evidence rows → cells
# ---------------------------------------------------------------------------


class TestMachineFragment:
    """The evidence store round-trips construction provenance."""

    def test_stored_constructed_restores_trace(self):
        conn = _conn()
        obj = _fold_obj()
        obs.store_constructed(conn, obj, env=None)
        frag = cs.machine_fragment(conn)
        assert len(frag.cells) == 1
        cell = frag.cells[0]
        assert cell.role == "object" and cell.kind == obj.kind
        assert cell.provenance.origin == "constructed"
        assert cell.provenance.op == "fold"
        conn.close()

    def test_row_without_provenance_reports_admitted(self):
        conn = _conn()
        rule = Rewrite(
            name="r",
            lhs=Op.make("mul", "a", "b"),
            rhs=Op.make("mul", "b", "a"),
        )
        ev.store_object(conn, rule, kind="law", cert=None)
        frag = cs.machine_fragment(conn)
        cell = frag.cells[0]
        assert cell.role == "law"
        assert cell.provenance.origin == "admitted"
        conn.close()

    def test_ambient_store_merges_everything(self):
        conn = _conn()
        obs.store_constructed(conn, _fold_obj(), env=None)
        store = cs.ambient_store(
            rules=[], handlers={}, machine_conn=conn
        )
        assert len(store) == 1
        assert [c.name for c in store.objects()] == ["myabs"]
        conn.close()

    def test_ambient_store_defaults_to_shipped(self):
        store = cs.ambient_store()
        assert len(store.laws()) > 0
        assert store.handlers() == {
            h["kernel"]: h
            for h in __import__(
                "catopt_discovery.lawdata", fromlist=["HANDLERS"]
            ).HANDLERS.values()
        }

    def test_corrupt_row_is_skipped(self):
        conn = _conn()
        conn.execute(
            "INSERT INTO lemmas"
            " (alpha_key, name, law_json, derivation_json,"
            "  corpus_hash, added_ts)"
            " VALUES ('k', 'bad', '{not json', '[]', '', 't')"
        )
        frag = cs.machine_fragment(conn)
        assert frag.cells == ()
        conn.close()


# ---------------------------------------------------------------------------
#  The board's live vocabulary as a store
# ---------------------------------------------------------------------------


class TestBoardCellStore:
    """MetaArena.cellstore — same vocabulary, one spelling."""

    def test_fresh_board(self):
        board = ma.MetaArena(_sub_gap())
        store = board.cellstore()
        assert {r.name for r in store.laws()} == {
            r.name for r in board.rules
        }
        assert set(store.handlers()) == {
            h["kernel"] for h in board.handlers.values()
        }

    def test_declare_lands_as_an_object_cell(self):
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(
            ma.Action.declare(
                {
                    "op": "fold",
                    "params": {
                        "name": "myabs",
                        "spelled": ("add", "X", ("neg", "Y")),
                        "kernel": ("myabs", "X", "Y"),
                    },
                }
            )
        )
        assert rep.applied
        store = board.cellstore()
        obj = store.get("object", "myabs")
        assert obj is not None and obj.kind == "abstraction"
        assert obj.provenance.origin == "constructed"
        assert obj.provenance.op == "fold"
        unfold = store.get("law", "myabs_unfold")
        assert unfold is not None
        assert unfold.provenance.op == "unfold"
        assert unfold.provenance.premises == ("myabs",)
        assert {r.name for r in store.laws()} == {
            r.name for r in board.rules
        }

    def test_law_kind_declare_lands_as_a_law_cell(self):
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(
            ma.Action.declare(
                {
                    "op": "fold",
                    "params": {
                        "name": "mylaw",
                        "spelled": ("add", "X", ("neg", "Y")),
                        "kernel": ("mylaw", "X", "Y"),
                        "kind": "law",
                    },
                }
            )
        )
        assert rep.applied
        store = board.cellstore()
        assert store.get("object", "mylaw") is None
        cell = store.get("law", "mylaw")
        assert cell is not None
        assert cell.provenance.origin == "constructed"
