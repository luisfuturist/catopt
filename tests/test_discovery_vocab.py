"""Tests for ``catopt_discovery.vocab`` — property-derived op classes.

The vocabulary is derived by *test*, not lookup: a view is arity-1,
value-preserving and shape-changing; pointwise means shape-preserving
and commuting with a battery of views; a reduction returns fewer,
non-preserved elements.  The classifiers run on synthetic term lists
so every verdict — True / False / untestable — is pinned.
"""

import json

import torch
from catopt_core.ir import Op, TensorType, Var
from catopt_discovery import vocab as vc


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


_TERMS = [
    _p("add", _v("x", 4, 6), _v("y", 4, 6)),
    _p("mul", _v("x", 4, 6), _v("y", 4, 6)),
    _p("relu", _v("x", 4, 6)),
    _p("transpose", _v("x", 4, 6), dim0=0, dim1=1),
    _p("getitem", _v("x", 4, 6), index=0),
    _p("sum", _v("x", 4, 6), dim=0),
]


def test_leaf_rand_numel_helpers():
    assert vc._numel((2, 3)) == 6
    assert vc._numel(()) == 1
    t = vc._rand((2, 3))
    assert t.shape == (2, 3) and t.dtype == torch.float64
    leaf = vc._leaf((2, 3), "w")
    assert leaf.name == "w" and tuple(leaf.typ.shape) == (2, 3)


def test_concrete_shapes():
    assert vc._concrete((2, 3)) is True
    assert vc._concrete(()) is True
    assert vc._concrete((-1, 3)) is False
    assert vc._concrete((2, "x")) is False
    assert vc._concrete("nope") is False


def test_corpus_ops_mines_arities_and_occurrences():
    x, y = _v("x", 4, 6), _v("y", 4, 6)
    m = _p("mul", x, y)
    terms = [_p("add", m, m)]
    occ, arity = vc.corpus_ops(terms)
    assert arity == {"add": 2, "mul": 2}
    # The shared mul node is one occurrence, not two.
    assert len(occ["mul"]) == 1
    attrs, shapes = occ["add"][0]
    assert attrs == {} and shapes == [(4, 6), (4, 6)]


def test_value_preserving():
    x = torch.tensor([1.0, 2.0, 3.0])
    assert vc._value_preserving(x, torch.tensor([3.0, 1.0])) is True
    assert vc._value_preserving(x, torch.tensor([4.0])) is False


def test_battery_is_shape_agnostic():
    small = vc._battery((4,))
    big = vc._battery((4, 6))
    assert len(small) == 4
    assert len(big) == 5
    assert [p[0] for p in big][-1] == "transpose"


def test_is_view_true_false_untestable():
    occ, _arity = vc.corpus_ops(_TERMS)
    assert vc._is_view("transpose", 1, occ) is True
    assert vc._is_view("getitem", 1, occ) is True
    # Arity != 1 is never a view.
    assert vc._is_view("add", 2, occ) is False
    # A computing unary op preserves shape/values? relu does not
    # preserve values (negatives clamp) — but on positive probes it
    # does; either way it never changes shape → not a view.
    assert vc._is_view("relu", 1, occ) is False
    # An op that cannot evaluate is untestable.
    occ2, _ = vc.corpus_ops([_p("not_a_real_op", _v("x", 2, 2))])
    assert vc._is_view("not_a_real_op", 1, occ2) is None


def test_is_pointwise_true_false_arity_gate():
    occ, _arity = vc.corpus_ops(_TERMS)
    assert vc._is_pointwise("add", 2, occ) is True
    assert vc._is_pointwise("mul", 2, occ) is True
    assert vc._is_pointwise("relu", 1, occ) is True
    # transpose changes shape — not pointwise.
    assert vc._is_pointwise("transpose", 1, occ) is False
    # sum returns a different shape — not pointwise.
    assert vc._is_pointwise("sum", 1, occ) is False
    # Arity 3+ is left undecided, and so is an unevaluable op.
    assert vc._is_pointwise("add", 3, occ) is None
    occ2, _ = vc.corpus_ops([_p("not_a_real_op", _v("x", 2, 2))])
    assert vc._is_pointwise("not_a_real_op", 1, occ2) is None


def test_is_reduction_true_false_untestable():
    occ, _arity = vc.corpus_ops(_TERMS)
    assert vc._is_reduction("sum", 1, occ) is True
    assert vc._is_reduction("add", 2, occ) is False
    # An op that cannot evaluate is untestable.
    occ2, _ = vc.corpus_ops([_p("not_a_real_op", _v("x", 2, 2))])
    assert vc._is_reduction("not_a_real_op", 1, occ2) is None


def test_commutes_compose_and_failure():
    # A genuinely pointwise op commutes with a reshape probe.
    assert (
        vc._commutes("neg", {}, [(2, 3)], ("reshape", {"shape": (6,)}))
        is True
    )
    # A malformed arity-1 "add" cannot evaluate on either side —
    # the probe is no evidence either way.
    assert (
        vc._commutes("add", {}, [()], ("reshape", {"shape": (1,)}))
        is None
    )


def test_classify_synthetic_corpus():
    v = vc.classify(_TERMS)
    classes = {c.op: c for c in v.classes}
    assert set(classes) == {
        "add",
        "mul",
        "relu",
        "transpose",
        "getitem",
        "sum",
    }
    assert classes["add"].pointwise is True
    assert classes["add"].arity == 2
    assert classes["relu"].pointwise is True
    assert classes["transpose"].view is True
    assert classes["getitem"].view is True
    assert classes["sum"].reduction is True
    assert classes["transpose"].attribute_carrying is True
    assert classes["getitem"].attribute_carrying is True
    assert classes["add"].attribute_carrying is False
    assert v.pointwise == ("add", "mul")
    assert v.unary_pointwise == ("relu",)
    assert set(v.views) == {"transpose", "getitem"}
    assert len(v.probes) == 5


def test_derived_set_and_vocabulary_views():
    vocab = vc.Vocabulary(
        classes=(
            vc.OpClass("add", 2, True, False, False, False),
            vc.OpClass("relu", 1, True, False, False, False),
            vc.OpClass("tr", 1, None, True, False, True),
        )
    )
    assert vocab.pointwise == ("add",)
    assert vocab.unary_pointwise == ("relu",)
    assert vocab.views == ("tr",)
    assert vc._derived_set(vocab, "view") == frozenset({"tr"})
    assert vc._derived_set(vocab, "binary") == frozenset({"add"})
    assert vc._derived_set(vocab, "unary") == frozenset({"relu"})
    assert vc._derived_set(vocab, "pointwise") == frozenset(
        {"add", "relu"}
    )


def test_validate_reports_agreement_and_corroboration():
    v = vc.classify(_TERMS)
    val = vc.validate(v)
    assert val["corpus_ops"] == sorted(
        {"add", "mul", "relu", "transpose", "getitem", "sum"}
    )
    assert len(val["rows"]) == 7
    rows = {r["table"]: r for r in val["rows"]}
    pipe = rows["pipeline._POINTWISE"]
    assert set(pipe["agree"]) == {"add", "mul"}
    pipe_views = rows["pipeline._VIEW_OPS"]
    assert "transpose" in pipe_views["agree"]
    # getitem is a view the pipeline table missed — corroborated by
    # the morphisms signature table, which does list it.
    assert "getitem" in pipe_views["derived_only"]
    assert "getitem" in pipe_views["corroborated"]
    sig = rows["signature._POINTWISE_OPS"]
    assert "mul" in sig["agree"]
    assert "add" in sig["derived_only"]


def test_print_report_and_dump(tmp_path, capsys):
    v = vc.classify(_TERMS)
    val = vc.validate(v)
    vc._print_report(v, val)
    out = capsys.readouterr().out
    assert "law_vocab" in out
    assert "pointwise (arity 2): add, mul" in out
    assert "corroborated" in out
    path = tmp_path / "vocab.json"
    vc._dump_json(str(path), v, val)
    payload = json.loads(path.read_text())
    assert [c["op"] for c in payload["classes"]] == sorted(
        c.op for c in v.classes
    )
    assert payload["pointwise"] == ["add", "mul"]
    assert "getitem" in payload["views"]
    assert "corpus_ops" in payload["validation"]


def test_derive_vocabulary_over_real_corpus():
    v = vc.derive_vocabulary()
    assert len(v.classes) > 40
    ops = {c.op for c in v.classes}
    assert {"add", "mul", "matmul"} <= ops
    assert "add" in v.pointwise


def test_main_derives_validates_dumps(tmp_path, capsys):
    out = tmp_path / "v.json"
    assert vc.main(["--json", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert len(payload["classes"]) > 40
    assert "corpus_ops" in payload["validation"]
    text = capsys.readouterr().out
    assert "derived vs hand tables" in text
