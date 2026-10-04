"""Tests for ``catopt_discovery.census`` — the shape census.

The census counts what real graphs actually contain: op-tuple
frequencies, leaf-abstracted subterm shapes (a leaf that repeats
keeps its placeholder number, so sharing is visible), and the
sharing structure of binary op nodes — the precondition a
factoring/absorption law needs.  Tests run on synthetic
``CorpusTerm`` lists for the counts and on the real corpus for
``run_census`` itself.
"""

import json

from catopt_core.ir import Const, Op, TensorType, Var
from catopt_discovery import census as cs


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _t(name: str, term) -> cs.CorpusTerm:
    return cs.CorpusTerm("test", name, term)


def test_iter_subterms_dedups_shared_nodes():
    x, y = _v("x", 4), _v("y", 4)
    m = _p("mul", x, y)
    t = _p("add", m, m)
    subs = cs._iter_subterms(t)
    # The shared mul node is visited once, not once per path.
    assert len(subs) == 4  # add, mul, x, y


def test_op_of_markers():
    assert cs._op_of(_p("add", _v("x", 2), _v("y", 2))) == "add"
    assert cs._op_of(Const(1)) == "const"
    assert cs._op_of(_v("x", 2)) == "·"


def test_attr_key_order_independent_and_unhashable():
    assert cs._attr_key({"b": 2, "a": 1}) == cs._attr_key(
        {"a": 1, "b": 2}
    )
    assert cs._attr_key({"a": 1, "b": 2}) == (("a", 1), ("b", 2))
    # An unhashable attr value falls back to its repr.
    assert cs._attr_key({"a": [1, 2]}) == (("a", "[1, 2]"),)


def test_shape_key_leaf_numbering_by_identity():
    x, y = _v("x", 4), _v("y", 4)
    shared = cs.shape_key(_p("mul", x, x), {})
    assert shared[2][0] == shared[2][1] == ("leaf", 0)
    assert cs._has_shared_leaf(shared) is True
    distinct = cs.shape_key(_p("mul", x, y), {})
    assert distinct[2][0] != distinct[2][1]
    assert cs._has_shared_leaf(distinct) is False
    # Const nodes keep their literal, not a placeholder.
    c = cs.shape_key(_p("mul", x, Const(0)), {})
    assert c[2][1] == ("const", 0)
    # Two different terms number independently (fresh memo).
    k1 = cs.shape_key(_p("mul", x, y), {})
    k2 = cs.shape_key(_p("mul", _v("a", 4), _v("b", 4)), {})
    assert k1 == k2


def test_leaf_ids_and_shared_leaf_helpers():
    assert cs._leaf_ids(5) == []
    assert cs._leaf_ids(("leaf", 2)) == [2]
    assert cs._leaf_ids(("const", 0)) == []
    key = ("op", (), (("leaf", 0), ("leaf", 1), ("leaf", 0)))
    assert cs._leaf_ids(key) == [0, 1, 0]
    assert cs._has_shared_leaf(key) is True


def test_shape_repr_notation():
    x, y = _v("x", 4), _v("y", 4)
    key = cs.shape_key(_p("add", _p("mul", x, y), x), {})
    assert cs.shape_repr(key) == "add(mul(a, b), a)"
    kattr = cs.shape_key(_p("select", x, dim=0, index=2), {})
    assert cs.shape_repr(kattr) == "select[dim=0,index=2](a)"
    assert cs.shape_repr(("const", 3)) == "3"
    assert cs.shape_repr(("leaf", 30)) == "v30"
    assert cs.shape_repr(("leaf", 1)) == "b"
    assert cs.shape_repr(42) == "42"


def test_op_tuple_census_counts_and_spans():
    x, y, z = _v("x", 4), _v("y", 4), _v("z", 4)
    terms = [
        _t("t1", _p("add", _p("mul", x, y), z)),
        _t("t2", _p("add", _p("mul", x, z), y)),
    ]
    counts, terms_of = cs.op_tuple_census(terms)
    key = ("add", ("mul", "·"))
    assert counts[key] == 2
    assert terms_of[key] == {("test", "t1"), ("test", "t2")}
    assert counts[("mul", ("·", "·"))] == 2
    # A leaf-only term contributes no op nodes.
    counts2, _ = cs.op_tuple_census([_t("leaf", x)])
    assert sum(counts2.values()) == 0
    # Consts are marked distinctly from variable leaves.
    counts3, _ = cs.op_tuple_census([_t("c", _p("mul", x, Const(0)))])
    assert counts3[("mul", ("·", "const"))] == 1


def test_shape_census_groups_equal_shapes():
    x, y = _v("x", 4), _v("y", 4)
    a, b = _v("a", 4), _v("b", 4)
    terms = [_t("t1", _p("mul", x, y)), _t("t2", _p("mul", a, b))]
    counts, terms_of = cs.shape_census(terms)
    key = cs.shape_key(terms[0].term, {})
    assert counts[key] == 2  # same shape, different leaves
    assert terms_of[key] == {("test", "t1"), ("test", "t2")}
    assert sum(counts.values()) == 2  # leaves are not Op nodes


def test_sharing_census_shared_operands():
    # ``shared_child`` compares each operand's shape under a fresh
    # memo — it flags *structurally equal* children (the add(t, t)
    # case); ``shared_leaf`` numbers leaves under ONE memo for the
    # whole node, so it flags true leaf repetition (the factoring
    # precondition).
    x, y, z, w = (_v(n, 4) for n in "xyzw")
    terms = [
        _t("t1", _p("mul", x, x)),
        _t("t2", _p("mul", x, y)),
        _t("t3", _p("mul", x, Const(0))),
        _t("t4", _p("add", _p("mul", x, y), _p("mul", x, z))),
        _t("t5", _p("add", _p("mul", x, x), _p("mul", y, z))),
        _t("t6", _p("add", _p("mul", x, y), _p("mul", z, w))),
    ]
    agg = cs.sharing_census(terms)
    muls = agg[("mul", ("·", "·"))]
    assert muls["sites"] == 8  # 2 top-level + 6 nested in the adds
    # Any two leaf children are shape-equal under fresh memos.
    assert muls["shared_child"] == 8
    # True leaf repetition happens only in the two mul(x, x) nodes.
    assert muls["shared_leaf"] == 2
    consts = agg[("mul", ("·", "const"))]
    assert consts["sites"] == 1
    assert consts["shared_child"] == 0
    adds = agg[("add", ("mul", "mul"))]
    assert adds["sites"] == 3
    # t4/t6 children have identical shapes; t5's differ
    # (mul(a,a) vs mul(b,c)).
    assert adds["shared_child"] == 2
    # t4 and t5 repeat a leaf across the whole node; t6 does not.
    assert adds["shared_leaf"] == 2
    # Unary and non-algebraic ops are not counted at all.
    agg2 = cs.sharing_census([_t("u", _p("neg", _p("matmul", x, y)))])
    assert ("matmul", ("·", "·")) in agg2
    assert all(
        k[0] in {"add", "sub", "mul", "div", "matmul"} for k in agg2
    )
    # A binary-named op with the wrong arity is skipped too.
    agg3 = cs.sharing_census([_t("odd", _p("add", x))])
    assert agg3 == {}


def test_render_tables(tmp_path):
    x, y, z = _v("x", 4), _v("y", 4), _v("z", 4)
    terms = [
        _t("t1", _p("add", _p("mul", x, x), _p("mul", y, z))),
        _t("t2", _p("add", _p("mul", x, y), _p("mul", x, z))),
    ]
    oc, ot = cs.op_tuple_census(terms)
    sc, st = cs.shape_census(terms)
    agg = cs.sharing_census(terms)
    t = cs._op_tuple_table(oc, ot, 5)
    assert "mul(·, ·)" in t and "add(mul, mul)" in t
    s = cs._shape_table(sc, st, 5)
    assert "yes" in s  # a shared-leaf shape is flagged
    assert "mul(a, a)" in s
    sh = cs._sharing_table(agg)
    assert "mul(·, ·)" in sh and "shared-leaf" in sh


def test_run_census_over_real_corpus():
    result = cs.run_census(top=10)
    assert result["n_terms"] > 0
    assert result["n_bench"] > 0 and result["n_models"] > 0
    assert result["n_terms"] == (
        result["n_bench"] + result["n_models"] + result["n_intake"]
    )
    assert result["n_op_nodes"] > 0
    assert result["n_shapes"] > 0
    ot = result["op_tuples"][0]
    assert set(ot) == {"op", "children", "count", "terms"}
    assert ot["count"] >= 1 and ot["terms"] >= 1
    sh = result["shapes"][0]
    assert set(sh) == {"shape", "count", "terms", "shared"}
    assert len(result["op_tuples"]) <= 10
    for row in result["sharing"]:
        assert set(row) >= {
            "op",
            "children",
            "sites",
            "shared_child",
            "shared_leaf",
        }


def test_main_prints_report_and_dumps_json(tmp_path, capsys):
    out = tmp_path / "census.json"
    rc = cs.main(["--json", str(out), "--top", "5"])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_terms"] > 0
    assert len(payload["op_tuples"]) <= 5
    text = capsys.readouterr().out
    assert "law_shape_census" in text
    assert "op-tuple census" in text
    assert "sharing census" in text
