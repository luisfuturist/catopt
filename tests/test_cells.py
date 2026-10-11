"""Tests for the cell store — the fragment-merged vocabulary.

``catopt_core.cells`` is the unification of the project's five
cell spellings: the tests cover the record's validation, fragment
assembly (conflict policies, alpha-dup reporting), the engine
views (laws/objects/opdefs/handlers/ruleset), and the one-schema
codec — including a round-trip over real shipped laws, ops, and
handler entries.
"""

from __future__ import annotations

import json

import pytest
from catopt_core.cells import (
    Cell,
    CellStore,
    Fragment,
    Provenance,
    cell_from_data,
    cell_to_data,
    handle_cell,
    law_cell,
    laws_fragment,
    merge_fragments,
    object_cell,
    op_cell,
    rewrite_of,
    spec_from_data,
    spec_to_data,
)
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op, Var
from catopt_core.laws import ALL_RULES
from catopt_core.laws.serialize import alpha_key
from catopt_core.opdata import OpDef


def _rule(name: str, lhs, rhs, **kw) -> Rewrite:
    """A minimal unconditional rewrite for the tests."""
    return Rewrite(name=name, lhs=lhs, rhs=rhs, **kw)


def _handler(kernel: str = "gen_0_k") -> dict:
    """A minimal minted-handler entry."""
    return {
        "pattern": ("mul", "a", ("add", "b", "c")),
        "kernel": kernel,
        "args": ("a", "b", "c"),
    }


# ---------------------------------------------------------------------------
#  The record
# ---------------------------------------------------------------------------


class TestCellValidation:
    """The cell record rejects malformed vocabulary at construction."""

    def test_unknown_role(self):
        with pytest.raises(ValueError, match="unknown cell role"):
            Cell(role="weird", name="x", body=None)

    def test_object_needs_a_kind(self):
        r = _rule(
            "o", Op.make("add", "a", "b"), Op.make("add", "a", "b")
        )
        with pytest.raises(ValueError, match="needs a kind"):
            Cell(role="object", name="o", body=r)

    def test_non_object_carries_no_kind(self):
        r = _rule(
            "l", Op.make("add", "a", "b"), Op.make("add", "a", "b")
        )
        with pytest.raises(ValueError, match="carries no kind"):
            Cell(role="law", name="l", body=r, kind="abstraction")

    def test_body_type_checked(self):
        with pytest.raises(TypeError, match="needs a Rewrite"):
            Cell(role="law", name="l", body={"not": "a rule"})
        with pytest.raises(TypeError, match="needs a OpDef"):
            Cell(role="op", name="o", body=_rule("x", "a", "a"))

    def test_handle_needs_the_entry_keys(self):
        with pytest.raises(ValueError, match="missing handler keys"):
            Cell(role="handle", name="h", body={"pattern": "x"})

    def test_unknown_provenance_origin(self):
        with pytest.raises(ValueError, match="unknown provenance"):
            Provenance(origin="invented")


class TestConverters:
    """The five spellings wrap into cells, each with honest provenance."""

    def test_law_cell(self):
        r = _rule(
            "comm", Op.make("add", "a", "b"), Op.make("add", "b", "a")
        )
        c = law_cell(r)
        assert c.role == "law" and c.name == "comm" and c.body is r
        assert c.provenance.origin == "shipped"

    def test_object_cell(self):
        r = _rule("abs", "x", "y")
        prov = Provenance(
            origin="constructed", op="compose", premises=("a", "b")
        )
        c = object_cell(r, kind="abstraction", provenance=prov)
        assert c.role == "object" and c.kind == "abstraction"
        assert c.provenance.premises == ("a", "b")

    def test_op_and_handle_cells(self):
        od = OpDef(name="myop", arity=2)
        assert op_cell(od).provenance.origin == "declared"
        h = handle_cell("gen_0_k", _handler())
        assert h.role == "handle" and h.provenance.origin == "minted"
        assert h.name == "gen_0_k"

    def test_rewrite_of_views_both_2cell_roles(self):
        r = _rule("x", "a", "a")
        assert rewrite_of(law_cell(r)) is r
        assert rewrite_of(object_cell(r, kind="bridge")) is r
        with pytest.raises(TypeError, match="no Rewrite body"):
            rewrite_of(op_cell(OpDef(name="z")))

    def test_laws_fragment_bundles_rules(self):
        frag = laws_fragment("f", list(ALL_RULES)[:3])
        assert len(frag.cells) == 3
        assert all(c.role == "law" for c in frag.cells)


# ---------------------------------------------------------------------------
#  The merge
# ---------------------------------------------------------------------------


class TestMerge:
    """merge_fragments assembles with validation, not silent override."""

    def _two_laws(self):
        a = _rule("a", "x", "x")
        b = _rule("b", "y", "y")
        return (
            Fragment("f1", (law_cell(a),)),
            Fragment("f2", (law_cell(b),)),
        )

    def test_basic_merge_and_views(self):
        store = merge_fragments(*self._two_laws())
        assert len(store) == 2 and store.fragments == ("f1", "f2")
        assert {r.name for r in store.laws()} == {"a", "b"}
        assert store.report.counts["law"] == 2
        assert ("law", "a") in store
        assert store.get("law", "b").name == "b"
        assert store.get("law", "absent") is None
        assert store.fragment_of(store.get("law", "a")) == "f1"
        assert [c.name for c in store] == ["a", "b"]

    def test_duplicate_inside_one_fragment_raises(self):
        r = _rule("a", "x", "x")
        with pytest.raises(ValueError, match="duplicate cell"):
            Fragment("f", (law_cell(r), law_cell(r)))

    def test_conflict_errors_by_default(self):
        f1 = Fragment("f1", (law_cell(_rule("a", "x", "x")),))
        f2 = Fragment("f2", (law_cell(_rule("a", "y", "y")),))
        with pytest.raises(ValueError, match="claimed by fragment"):
            merge_fragments(f1, f2)

    def test_keep_policies_record_conflicts(self):
        f1 = Fragment("f1", (law_cell(_rule("a", "x", "x")),))
        f2 = Fragment("f2", (law_cell(_rule("a", "y", "y")),))
        first = merge_fragments(f1, f2, on_conflict="keep_first")
        last = merge_fragments(f1, f2, on_conflict="keep_last")
        assert first.report.conflicts == (("law", "a"),)
        assert first.get("law", "a").body.lhs == "x"
        assert last.get("law", "a").body.lhs == "y"
        assert first.fragment_of(first.get("law", "a")) == "f1"
        assert last.fragment_of(last.get("law", "a")) == "f2"

    def test_unknown_conflict_policy(self):
        with pytest.raises(ValueError, match="conflict policy"):
            merge_fragments(on_conflict="shrug")

    def test_alpha_dups_reported_not_fatal(self):
        add_ab = Op.make("add", "a", "b")
        add_ba = Op.make("add", "b", "a")
        f1 = Fragment("f1", (law_cell(_rule("r1", add_ab, add_ba)),))
        f2 = Fragment(
            "f2",
            (
                law_cell(
                    _rule(
                        "r2",
                        Op.make("add", "s", "t"),
                        Op.make("add", "t", "s"),
                    )
                ),
            ),
        )
        store = merge_fragments(f1, f2)
        assert store.report.alpha_dups == (("r1", "r2"),)

    def test_store_merge_revalidates(self):
        store = merge_fragments(*self._two_laws())
        with pytest.raises(ValueError, match="claimed by fragment"):
            store.merge(
                Fragment("f3", (law_cell(_rule("a", "z", "z")),))
            )
        grown = store.merge(
            Fragment("f3", (handle_cell("k", _handler("k")),))
        )
        assert len(grown) == 3 and "k" in grown.handlers()


# ---------------------------------------------------------------------------
#  The views
# ---------------------------------------------------------------------------


class TestViews:
    """Engines read one store through role views."""

    def _store(self):
        law = _rule(
            "comm", Op.make("add", "a", "b"), Op.make("add", "b", "a")
        )
        obj = _rule("br", "x", "y")
        return merge_fragments(
            Fragment(
                "all",
                (
                    law_cell(law),
                    object_cell(obj, kind="bridge"),
                    op_cell(OpDef(name="myop", arity=1)),
                    handle_cell("gen_0_k", _handler()),
                ),
            )
        )

    def test_role_views(self):
        store = self._store()
        assert [c.name for c in store.cells("handle")] == ["gen_0_k"]
        assert [r.name for r in store.laws()] == ["comm", "br"]
        assert [o.name for o in store.objects()] == ["br"]
        assert store.objects("abstraction") == ()
        assert [o.name for o in store.objects("bridge")] == ["br"]
        assert [o.name for o in store.opdefs()] == ["myop"]
        assert store.handlers()["gen_0_k"]["kernel"] == "gen_0_k"

    def test_ruleset_fires_in_the_egraph(self):
        from catopt_core.egraph import EGraph

        store = self._store()
        eg = EGraph()
        root = eg.add_term(Op.make("add", "p", "q"))
        eg.run(store.ruleset("t").rules, root, max_iterations=4)
        alts = eg.extract_alternatives(root, lambda t: 1)
        assert Op.make("add", "q", "p") in {t for _, t in alts}


# ---------------------------------------------------------------------------
#  The codec
# ---------------------------------------------------------------------------


class TestCodec:
    """One schema for the whole vocabulary."""

    def _roundtrip(self, spec):
        return spec_from_data(
            json.loads(json.dumps(spec_to_data(spec)))
        )

    def test_spec_codec(self):
        assert self._roundtrip("m") == "m"
        assert self._roundtrip(2.5) == 2.5
        spec = ("mul", "a", ("add", "b", 1.0, {"dim": 0}))
        assert self._roundtrip(spec) == spec
        v = Var(
            "x",
            __import__(
                "catopt_core.ir", fromlist=["TensorType"]
            ).TensorType((4,)),
        )
        assert self._roundtrip(v) == v
        assert spec_to_data(True) is True
        assert spec_from_data("plain") == "plain"
        with pytest.raises(ValueError, match="bad spec encoding"):
            spec_from_data({"bogus": 1})

    def test_cell_roundtrips(self):
        rule = _rule(
            "comm", Op.make("add", "a", "b"), Op.make("add", "b", "a")
        )
        cells = [
            law_cell(rule),
            object_cell(
                _rule("br", "x", "y"),
                kind="bridge",
                provenance=Provenance(
                    "constructed", op="fold", premises=("p1",), note="n"
                ),
            ),
            op_cell(OpDef(name="myop", arity=2)),
            handle_cell("gen_0_k", _handler()),
        ]
        out = [
            cell_from_data(json.loads(json.dumps(cell_to_data(c))))
            for c in cells
        ]
        assert [c.role for c in out] == [c.role for c in cells]
        assert out[0].body.lhs == cells[0].body.lhs
        assert out[0].body.rhs == cells[0].body.rhs
        assert out[1].kind == "bridge"
        assert out[1].provenance.op == "fold"
        assert out[1].provenance.premises == ("p1",)
        assert out[2].body.name == "myop"
        assert out[3].body["pattern"] == cells[3].body["pattern"]
        assert out[3].body["args"] == cells[3].body["args"]

    def test_cell_from_data_rejects_junk(self):
        with pytest.raises(ValueError, match="unknown cell role"):
            cell_from_data({"role": "bogus", "name": "x", "body": {}})

    def test_store_roundtrip(self):
        store = merge_fragments(
            Fragment("f1", (law_cell(_rule("a", "x", "x")),)),
            Fragment("f2", (handle_cell("k", _handler("k")),)),
        )
        back = CellStore.from_data(
            json.loads(json.dumps(store.to_data()))
        )
        assert len(back) == 2
        assert {r.name for r in back.laws()} == {"a"}
        assert "k" in back.handlers()
        assert back.fragment_of(back.get("law", "a")) == "f1"

    def test_empty_store_roundtrip(self):
        back = CellStore.from_data({"cells": []})
        assert len(back) == 0

    def test_real_laws_roundtrip_through_the_store(self):
        """Shipped laws survive the cell codec — alpha-key equality."""
        serial = list(ALL_RULES)[:12]
        frag = laws_fragment("core", serial)
        store = merge_fragments(frag)
        back = CellStore.from_data(
            json.loads(json.dumps(store.to_data()))
        )
        keys = {alpha_key(r.lhs, r.rhs) for r in back.laws()}
        want = {alpha_key(r.lhs, r.rhs) for r in serial}
        assert keys == want
