"""Tests for ``catopt_discovery.verifier`` — the e-graph law oracle.

``verify_law`` decides derivability by saturating both sides of a
claimed equality under a rule universe; on a merge it replays a
positional certificate in both directions and reports the honest
one.  These tests pin all three verdicts — derivable, false, and
true-but-not-derivable — plus the instance builders, the structural
relation classifier, and the three experiments riding on top.
"""

import json

from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op, TensorType, Var, op_repr
from catopt_core.laws import ALL_RULES
from catopt_discovery import verifier as vf

_BY_NAME = {r.name: r for r in ALL_RULES}
_UNI = [
    _BY_NAME[n]
    for n in ("comm_add", "sub_to_add", "double_neg", "id_add")
]


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def test_verify_law_derivable_composite():
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    res = vf.verify_law(
        _p("sub", x, _p("neg", y)), _p("add", x, y), _UNI
    )
    assert res.derivable is True
    assert res.replayable is True
    assert res.direction in ("lhs->rhs", "rhs->lhs")
    assert res.witness_steps >= 1
    assert set(res.witness_rules) <= {
        "sub_to_add",
        "double_neg",
        "comm_add",
    }
    assert res.note == ""
    assert res.stop == "fixed_point"
    assert res.n_enodes > 0 and res.n_classes > 0


def test_verify_law_rejects_false_equality():
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    res = vf.verify_law(_p("add", x, y), _p("mul", x, y), _UNI)
    assert res.derivable is False
    assert res.witness_rules == ()
    assert res.witness_steps == 0
    assert res.replayable is False
    assert res.direction == ""


def test_verify_law_true_but_not_derivable():
    # add(x, 0) = x is true but needs id_add; without it the
    # e-graph cannot close the gap.
    x = _v("x", 4, 4)
    no_id = [r for r in _UNI if r.name != "id_add"]
    res = vf.verify_law(_p("add", x, Const(0)), x, no_id)
    assert res.derivable is False
    res2 = vf.verify_law(_p("add", x, Const(0)), x, _UNI)
    assert res2.derivable is True


def test_verify_law_identical_sides_trivially_merge():
    x = _v("x", 4, 4)
    res = vf.verify_law(_p("add", x, x), _p("add", x, x), [])
    assert res.derivable is True
    assert res.witness_steps == 0


def test_verify_law_node_budget_reports_stop():
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    res = vf.verify_law(
        _p("sub", x, _p("neg", y)),
        _p("add", x, y),
        _UNI,
        max_nodes=4,
    )
    assert res.stop == "max_nodes"
    assert res.n_enodes <= 16


def test_verify_law_replay_failure_is_reported(monkeypatch):
    # A merge is still reported honestly when neither direction's
    # certificate replays — derivable with a "replay failed" note.
    def _boom(_src, _cert):
        raise ValueError("broken cert")

    monkeypatch.setattr(vf, "verify_certificate", _boom)
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    res = vf.verify_law(
        _p("sub", x, _p("neg", y)), _p("add", x, y), _UNI
    )
    assert res.derivable is True
    assert res.replayable is False
    assert res.note.startswith("replay failed: ValueError")
    assert res.direction == "lhs->rhs"


def test_normalize_pattern_alpha_renames_metavars():
    a = vf.normalize_pattern(_p("add", "x", _p("mul", "y", "x")))
    b = vf.normalize_pattern(_p("add", "u", _p("mul", "w", "u")))
    assert op_repr(a) == op_repr(b)
    # First occurrence order drives numbering.
    n = vf.normalize_pattern(_p("mul", "b", "a"))
    assert op_repr(n) == op_repr(_p("mul", "v0", "v1"))
    # Attr metavariables normalize too; non-str leaves pass through.
    t = vf.normalize_pattern(_p("select", "x", dim="D", index=3), {})
    assert t.attrs["dim"] == "v1" and t.attrs["index"] == 3


def test_classify_relation():
    tgt = Rewrite("tgt", _p("add", "x", "y"), _p("mul", "x", "y"))
    dup = Rewrite("dup", _p("add", "a", "b"), _p("mul", "a", "b"))
    inv = Rewrite("inv", _p("mul", "a", "b"), _p("add", "a", "b"))
    oth = Rewrite("oth", _p("neg", "a"), "a")
    assert vf.classify_relation(tgt, dup) == "duplicate"
    assert vf.classify_relation(tgt, inv) == "inverse"
    assert vf.classify_relation(tgt, oth) == "composite"
    # Same name is never reported as a structural relation.
    assert vf.classify_relation(tgt, tgt) == "composite"


def test_instance_of_from_bench_registry():
    inst = vf.instance_of(_BY_NAME["sub_to_add"])
    assert inst is not None
    lhs, rhs = inst
    assert lhs.op == "sub"
    assert rhs.op == "add" and rhs.args[1].op == "neg"


def test_instance_of_generic_fallback():
    # sdpa_fold_add has no bench registry entry — the instance is
    # built from the rule's own pattern via generic_instance.
    inst = vf.instance_of(_BY_NAME["sdpa_fold_add"])
    assert inst is not None
    assert inst[1].op == "sdpa"


def test_generic_instance_unbuildable_returns_none():
    # mul_square's metavariables are outside the SDPA leaf table,
    # so no well-typed instance can be synthesized.
    assert vf.generic_instance(_BY_NAME["mul_square"]) is None


def test_generic_instance_check_and_derive_vetoes():
    src = _BY_NAME["sdpa_fold_add"]
    # A check hook that vetoes or raises yields no instance — the
    # unbuildable case is reported honestly, never guessed.
    veto = Rewrite("veto", src.lhs, src.rhs, check=lambda b: False)
    assert vf.generic_instance(veto) is None

    def _boom(_bound):
        raise ValueError("no")

    err = Rewrite("err", src.lhs, src.rhs, check=_boom)
    assert vf.generic_instance(err) is None
    # A derive hook that vetoes or raises is treated the same way.
    dnone = Rewrite("dnone", src.lhs, src.rhs, derive=lambda b: None)
    assert vf.generic_instance(dnone) is None
    derr = Rewrite("derr", src.lhs, src.rhs, derive=_boom)
    assert vf.generic_instance(derr) is None


def test_generic_instance_unknown_attr_metavar():
    # An attr metavariable outside the defaults table yields None.
    rule = Rewrite(
        "attr",
        _p("sdpa", "Q", "K", "V", "M", scale="ZZ"),
        "Q",
    )
    assert vf.generic_instance(rule) is None


def test_generic_instance_no_hooks_instantiates():
    # No check/derive at all: leaf and attr metavars from the known
    # tables are substituted and both sides instantiate directly.
    rule = Rewrite(
        "plain",
        _p("transpose", "Q", dim0="TD1", dim1="TD2"),
        "Q",
    )
    inst = vf.generic_instance(rule)
    assert inst is not None
    lhs, rhs = inst
    assert lhs.op == "transpose"
    assert lhs.attrs == {"dim0": -2, "dim1": -1}
    assert isinstance(rhs, Var) and rhs.name == "Q"


def test_instance_of_no_lhs_match_returns_none(monkeypatch):
    # A registry entry whose term never matches the rule's LHS is an
    # honest None, not a crash.
    x = _v("x", 4, 4)
    monkeypatch.setitem(
        vf._law_cases(),
        "sub_to_add",
        lambda _size, _dev: (Op.make("add", x, x), {}, ()),
    )
    assert vf.instance_of(_BY_NAME["sub_to_add"]) is None


def test_instance_of_apply_veto_returns_none(monkeypatch):
    # The LHS matches but the rule's check vetoes the rewrite at
    # every position → None.
    x = _v("x", 4, 4)
    monkeypatch.setitem(
        vf._law_cases(),
        "sub_to_add",
        lambda _size, _dev: (Op.make("sub", x, x), {}, ()),
    )
    src = _BY_NAME["sub_to_add"]
    vetoed = Rewrite(
        "sub_to_add", src.lhs, src.rhs, check=lambda _b: False
    )
    assert vf.instance_of(vetoed) is None


def test_law_cases_adds_repo_to_syspath(monkeypatch):
    import sys

    from catopt_discovery import REPO_ROOT

    kept = [p for p in sys.path if p != str(REPO_ROOT)]
    monkeypatch.setattr(sys, "path", kept)
    cases = vf._law_cases()
    assert "sub_to_add" in cases
    assert str(REPO_ROOT) in sys.path


def test_rediscover_three_categories():
    uni = [
        _BY_NAME[n]
        for n in (
            "sub_to_add",
            "double_neg",
            "id_add",
            "comm_add",
            "silu_expand",
            "silu_fold",
            "mul_square",
        )
    ]
    rows = vf.rediscover(uni, max_nodes=50_000)
    by = {r.name: r for r in rows}
    assert by["sub_to_add"].category == "primitive"
    assert by["silu_fold"].category == "derivable"
    assert by["silu_fold"].relation == "inverse"
    assert "silu_expand" in by["silu_fold"].result.witness_rules
    assert by["mul_square"].category == "no-instance"
    assert by["mul_square"].result is None
    assert by["mul_square"].lhs_repr == ""


def test_run_proposals_match_expectations():
    rows = vf.run_proposals()
    assert len(rows) == 8
    assert all(r.ok for r in rows), [r.label for r in rows if not r.ok]
    expects = {r.expectation for r in rows}
    assert expects == {"derivable", "not-derivable"}
    # The conjecture case is true yet non-derivable by design.
    conj = next(r for r in rows if "conjecture" in r.label)
    assert conj.result.derivable is False


def test_run_controls_small_and_empty():
    assert vf.run_controls([_BY_NAME["sub_to_add"]]) == {
        "n": 1,
        "failures": [],
    }
    assert vf.run_controls([]) == {"n": 0, "failures": []}


def test_run_controls_reports_both_failures(monkeypatch):
    # The controls are not decorative: a free merge and a missed
    # self-derivation are both named in the failures list.
    x = _v("x", 4, 4)
    same = (_p("add", x, x), _p("add", x, x))
    diff = (_p("add", x, x), _p("mul", x, x))
    # Both fakes carry LHS patterns that never match — the merges
    # (or lack of them) are decided by the instance sides alone.
    fake_free = Rewrite("f1", _p("nope1", "a"), "a")
    fake_inept = Rewrite("f2", _p("nope2", "a"), "a")
    inst = {"f1": same, "f2": diff}
    monkeypatch.setattr(
        vf, "instance_of", lambda rule, size=16: inst[rule.name]
    )
    out = vf.run_controls([fake_free, fake_inept])
    assert out["n"] == 2
    assert out["failures"] == [
        "f1: merged with NO rules",
        "f2: own rule did not derive it",
    ]


def test_structural_report_inverses_and_duplicates():
    uni = [_BY_NAME[n] for n in ("silu_expand", "silu_fold", "id_add")]
    rep = vf.structural_report(uni)
    assert rep["duplicates"] == []
    assert ["silu_expand", "silu_fold"] in rep["inverses"]
    # A hand-built exact duplicate is reported as such.
    fold = _BY_NAME["silu_fold"]
    clone = Rewrite("silu_fold_clone", fold.lhs, fold.rhs)
    rep2 = vf.structural_report([*uni, clone])
    assert ["silu_fold", "silu_fold_clone"] in rep2["duplicates"]


def test_main_runs_experiments_and_dumps(tmp_path, capsys):
    out = tmp_path / "law.json"
    rc = vf.main(["--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert {
        "rediscovery",
        "proposals",
        "controls",
        "structural",
        "summary",
    } <= set(payload)
    assert payload["controls"]["failures"] == []
    assert all(p["ok"] for p in payload["proposals"])
    counts = payload["summary"]["counts"]
    assert counts["derivable"] > 0 and counts["primitive"] > 0
    text = capsys.readouterr().out
    assert "law rediscovery" in text
    assert "soundness controls" in text


def test_main_reports_mismatches_and_failures(monkeypatch, capsys):
    bad_row = vf.ProposalRow(
        label="bad",
        expectation="derivable",
        lhs_repr="a",
        rhs_repr="b",
        ok=False,
    )
    monkeypatch.setattr(vf, "rediscover", lambda **kw: [])
    monkeypatch.setattr(vf, "run_proposals", lambda **kw: [bad_row])
    monkeypatch.setattr(
        vf,
        "run_controls",
        lambda **kw: {"n": 1, "failures": ["f: no"]},
    )
    monkeypatch.setattr(
        vf,
        "structural_report",
        lambda rules=None: {"duplicates": [], "inverses": []},
    )
    assert vf.main([]) == 1
    out = capsys.readouterr().out
    assert "MISMATCHES" in out and "f: no" in out
