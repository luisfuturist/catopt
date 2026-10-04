"""Tests for ``catopt_discovery.impact`` — the law-impact probe.

``impact`` measures whether the 11 ``cand_*`` laws actually pay:
per-law firing probes (``_probe``/``_set_fires``), the relaxed-shape
census, the ``ALL_RULES`` vs ``ALL_RULES + cand`` reach comparison,
certificate replay and the report/driver surface.  These tests run
the real machinery — real ``EGraph`` saturation, real
``backend_cost`` pricing, real ``TorchSink`` lowering/verify — on
small terms, plus the real bench-case builders and the real model
corpus once each.
"""

import importlib
import json
import sys

import pytest
import torch
import torch.nn as nn
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_core.laws.base import R
from catopt_discovery import impact as im
from catopt_discovery.impact import TermCase
from catopt_torch.adapters import TorchSink


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _feed(*vars_: Var) -> tuple:
    return tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in vars_
    )


def _case(source: str, name: str, term: Op, *inputs: Var) -> TermCase:
    return TermCase(
        source=source,
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=_feed(*inputs),
        param_vals={},
    )


def _synth(name: str) -> TermCase:
    return {c.name: c for c in im.synthetic_cases()}[name]


def _law(name: str):
    return next(law for law in im.new_laws() if law.name == name)


@pytest.fixture(scope="module")
def sink() -> TorchSink:
    return TorchSink()


@pytest.fixture(scope="module")
def cost(sink):
    return im._cost_fn(sink)


@pytest.fixture(scope="module")
def mini_result():
    """``run_impact`` on a four-case corpus — all real machinery.

    The corpus loaders are patched down to one bench case, one model
    case and two synthetic controls so the whole run — 44 real
    probes, the relaxed census and one real reach comparison — stays
    under a second.
    """
    x, y, z = _v("x", 4, 4), _v("y", 4, 4), _v("z", 4, 4)
    u = _v("u", 4, 4)
    bench = TermCase(
        source="bench",
        name="factor",
        term=_p("add", _p("mul", x, y), _p("mul", x, z)),
        inputs=(x, y, z),
        feed=_feed(x, y, z),
        param_vals={},
    )
    model = TermCase(
        source="model",
        name="sqneg",
        term=_p("square", _p("neg", u)),
        inputs=(u,),
        feed=_feed(u),
        param_vals={},
    )
    # A second model on which NO candidate law fires — the whole-set
    # probe must record nothing for it.
    quiet = TermCase(
        source="model",
        name="quiet",
        term=_p("neg", u),
        inputs=(u,),
        feed=_feed(u),
        param_vals={},
    )
    synth = im.synthetic_cases()[:2]
    mp = pytest.MonkeyPatch()
    mp.setattr(im, "_bench_cases", lambda: ([bench], []))
    mp.setattr(im, "model_cases", lambda: ([model, quiet], []))
    mp.setattr(im, "synthetic_cases", lambda: synth)
    try:
        yield im.run_impact()
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
#  Module shim + the candidate laws
# ---------------------------------------------------------------------------


def test_repo_root_path_shim(monkeypatch):
    """``impact`` puts the repo root on ``sys.path`` for ``bench``.

    ``bench`` is a repo-root package, not an installed distribution;
    when the root is absent the module prepends it at import.  A
    reload with the root stripped exercises the shim for real.
    """
    root = str(im.REPO_ROOT)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != root])
    assert root not in sys.path
    importlib.reload(im)
    assert sys.path[0] == root


def test_new_laws_shape():
    laws = im.new_laws()
    assert len(laws) == 11
    assert all(law.name.startswith("cand_") for law in laws)
    assert all(
        law.lhs is not None and law.rhs is not None for law in laws
    )
    by = {law.name: law for law in laws}
    f = by["cand_mul_factor"]
    assert f.lhs.op == "add" and f.rhs.op == "mul"
    assert by["cand_div_self"].rhs == Const(1)
    # No candidate may shadow a shipped rule name.
    assert {law.name for law in laws}.isdisjoint(
        {r.name for r in ALL_RULES}
    )


def test_params_of_and_ir_of():
    x = _v("x", 4, 4)
    w = Param("w", TensorType((4, 4)))
    term = _p("add", _p("mul", x, w), Const(1))
    assert set(im._params_of(term)) == {"w"}
    # A repeated Param counts once; leaves alone are no params.
    term2 = _p("mul", _p("add", w, w), w)
    assert list(im._params_of(term2)) == ["w"]
    assert im._params_of(x) == {}
    ir = im._ir_of(term, (x,))
    assert ir.root is term and ir.inputs == [x]
    assert ir.input_names == {"x"}
    assert set(ir.params) == {"w"}


# ---------------------------------------------------------------------------
#  Term sources — the real bench/model/synthetic corpora
# ---------------------------------------------------------------------------


def test_bench_cases_builds_every_registered_case():
    cases, errors = im._bench_cases()
    assert errors == []
    assert len(cases) >= 50
    for c in cases:
        assert c.source == "bench"
        assert len(c.feed) == len(c.inputs)
        assert isinstance(c.param_vals, dict)


def test_bench_cases_reports_builder_errors(monkeypatch):
    from bench.suites.correctness import law_bench as lb

    real = dict(lb.LAW_CASES)
    good_name = sorted(real)[0]

    def bad(d, dev):
        raise RuntimeError("bench boom")

    monkeypatch.setattr(
        lb, "LAW_CASES", {"zz_bad": bad, good_name: real[good_name]}
    )
    cases, errors = im._bench_cases()
    assert [c.name for c in cases] == [good_name]
    assert errors == ["zz_bad: RuntimeError: bench boom"]


def test_model_cases_exports_the_corpus():
    cases, errors = im.model_cases()
    assert errors == []
    names = {c.name for c in cases}
    assert {"SwiGLU", "RMSNorm", "LSTMSeq", "InstanceNorm"} <= names
    for c in cases:
        assert c.source == "model"
        assert len(c.feed) == len(c.inputs)
        assert isinstance(c.param_vals, dict)
    # A multi-input model rides a tuple feed.
    la = next(c for c in cases if c.name == "LinearAttention")
    assert len(la.feed) == 3


def test_model_cases_reports_export_failures(monkeypatch):
    class DataDependent(nn.Module):
        def forward(self, x):
            if x.sum() > 0:
                return x * 2
            return x

    x = torch.randn(4, 4, dtype=torch.float64)
    monkeypatch.setattr(
        im,
        "_model_cases",
        lambda: [
            ("good", nn.SiLU().eval().double(), x),
            ("bad", DataDependent().eval().double(), x),
        ],
    )
    cases, errors = im.model_cases()
    assert [c.name for c in cases] == ["good"]
    assert len(errors) == 1 and errors[0].startswith("bad: ")


def test_synthetic_cases_match_their_laws():
    cases = im.synthetic_cases()
    assert [c.name for c in cases] == [
        law.name for law in im.new_laws()
    ]
    for c, law in zip(cases, im.new_laws(), strict=True):
        assert c.source == "synthetic"
        # Each witness is the law's own LHS instantiated — the same
        # op spine, metavars filled with concrete Vars.
        lhs_ops = [
            s.op
            for s in im._iter_subterms(law.lhs)
            if isinstance(s, Op)
        ]
        t_ops = [
            s.op for s in im._iter_subterms(c.term) if isinstance(s, Op)
        ]
        assert sorted(lhs_ops) == sorted(t_ops)
        assert all(t.dtype == torch.float64 for t in c.feed)
        assert c.param_vals == {}


# ---------------------------------------------------------------------------
#  _probe / _set_fires — firing + cost delta + verify
# ---------------------------------------------------------------------------


def test_probe_fires_pays_and_verifies(sink, cost):
    f = im._probe(
        _synth("cand_mul_factor"),
        _law("cand_mul_factor"),
        sink,
        cost,
    )
    assert f.fires >= 1
    assert f.changed and f.paid
    assert f.out_cost < f.base_cost
    assert f.verified == "pass"


def test_probe_no_fire_returns_bare_row(sink, cost):
    f = im._probe(
        _synth("cand_mul_factor"),
        _law("cand_square_neg"),
        sink,
        cost,
    )
    assert f.fires == 0 and not f.paid
    assert f.verified == "-" and not f.changed


def test_probe_unchanged_when_rhs_is_pricier(sink, cost):
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    case = _case("test", "mul", _p("mul", x, y), x, y)
    law = R(
        "t_pricier",
        _p("mul", "a", "b"),
        _p("mul", _p("mul", "a", "b"), "b"),
        law="test",
    )
    f = im._probe(case, law, sink, cost)
    assert f.fires == 1 and not f.changed and not f.paid
    assert f.verified == "same"
    assert "kept the input term" in f.note


def test_probe_reports_a_wrong_law_as_fail(sink, cost):
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    case = _case("test", "mul", _p("mul", x, y), x, y)
    law = R(
        "t_wrong",
        _p("mul", "a", "b"),
        _p("add", "a", "b"),
        law="x*y != x+y",
    )
    f = im._probe(case, law, sink, cost)
    assert f.fires == 1 and f.changed
    assert f.verified == "FAIL"
    assert "max_rel" in f.note


def test_probe_verify_error_is_an_honest_row(sink, cost, monkeypatch):
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    case = _case("test", "mul", _p("mul", x, y), x, y)
    law = R(
        "t_err",
        _p("mul", "a", "b"),
        _p("add", "a", "b"),
        law="test",
    )

    def boom(*a, **k):
        raise RuntimeError("lower boom")

    monkeypatch.setattr(im, "_lower_extracted", boom)
    f = im._probe(case, law, sink, cost)
    assert f.fires == 1 and f.changed
    assert f.verified == "error" and "lower boom" in f.note


def test_probe_reports_an_empty_extraction(sink, cost, monkeypatch):
    class _NoExtract(im.EGraph):
        def extract_best(self, *a, **k):
            return None

    monkeypatch.setattr(im, "EGraph", _NoExtract)
    f = im._probe(
        _synth("cand_mul_factor"),
        _law("cand_mul_factor"),
        sink,
        cost,
    )
    assert f.fires >= 1
    assert f.note == "extraction returned no member"
    assert not f.changed


def test_set_fires_counts_the_whole_set(sink, cost):
    laws = im.new_laws()
    fires = im._set_fires(_synth("cand_mul_factor"), laws)
    assert fires == {"cand_mul_factor": 1}
    x = _v("x", 4, 4)
    quiet = _case("test", "neg", _p("neg", x), x)
    assert im._set_fires(quiet, laws) == {}


# ---------------------------------------------------------------------------
#  Relaxed census + saturate/cert + reach
# ---------------------------------------------------------------------------


def test_iter_subterms_dedupes_shared_nodes():
    x = _v("x", 4)
    m = _p("mul", x, x)
    term = _p("add", m, m)
    subs = im._iter_subterms(term)
    assert len(subs) == len({id(s) for s in subs})
    assert subs.count(m) == 1 and term in subs and x in subs


def test_relax_wildcards_repeated_metavars():
    rel = im._relax(_law("cand_sub_self").lhs)
    assert rel == _p("sub", "x", "x__2")
    # Const leaves stay literal — they are not metavariables.
    mz = im._relax(_law("cand_mul_zero").lhs)
    assert mz == _p("mul", "x", Const(0))


def test_relaxed_census_counts_shape_not_equality():
    y, z = _v("y", 4, 4), _v("z", 4, 4)
    terms = [_p("sub", y, y), _p("sub", y, z), _p("add", y, z)]
    census = im._relaxed_census(terms, im.new_laws())
    # ``sub(·, ·)`` appears twice — once self-equal, once not.  The
    # relaxed count sees the SHAPE both times; the strict law only
    # fires on the self-equal one.
    assert census["cand_sub_self"] == 2
    # No square/neg or const-mul shape anywhere -> genuinely absent.
    assert census["cand_square_neg"] == 0
    assert census["cand_mul_zero"] == 0


def test_rule_budgets_only_expansive_shipped_rules():
    from catopt_core.laws import tags

    extra = R("t_x", _p("mul", "a", "b"), "a", law="test")
    budgets = im._rule_budgets([*ALL_RULES, extra])
    expansive = {r.name for r in ALL_RULES if tags.EXPANSIVE in r.tags}
    assert expansive  # premise: shipped rules carry the tag
    assert set(budgets) == expansive
    assert all(v == im._EXPANSIVE_BUDGET for v in budgets.values())
    assert "t_x" not in budgets


def test_saturate_and_cert_ok(sink, cost):
    x = _v("x", 4, 4)
    term = _p("mul", x, x)
    eg, _root, best, stats = im._saturate(term, list(ALL_RULES), cost)
    assert stats["n_enodes"] >= 1 and stats["n_classes"] >= 1
    assert stats["stop"] in {
        "fixed_point",
        "improving",
        "max_iterations",
        "max_nodes",
    }
    assert best is not None
    assert im._cert_ok(eg, term, best, cost) == "pass"

    # A certificate that cannot replay is an honest FAIL tag.
    def boom(*a, **k):
        raise KeyError("no such class")

    eg.certificate = boom
    assert im._cert_ok(eg, term, best, cost) == "FAIL (KeyError)"


def test_reach_row_shows_the_added_law(sink, cost):
    case = _synth("cand_square_neg")
    row = im.reach_row(case, [_law("cand_square_neg")], sink, cost)
    assert row["model"] == "cand_square_neg"
    assert row["new_fires"] == {"cand_square_neg": 1}
    assert row["changed"] and row["add_cost"] < row["base_cost"]
    assert row["add_enodes"] > row["base_enodes"]
    assert row["base_cert"] == "pass" and row["add_cert"] == "pass"
    assert row["base_stop"] and row["add_stop"]


def test_firing_paid_property():
    f = im.Firing(source="t", case="c", law="l")
    assert not f.paid  # never fired
    f.fires, f.base_cost, f.out_cost = 2, 10.0, 5.0
    assert f.paid
    f.out_cost = 10.0
    assert not f.paid  # fired but not cheaper


# ---------------------------------------------------------------------------
#  Report tables + run_impact + driver
# ---------------------------------------------------------------------------


def test_report_tables_render():
    laws = im.new_laws()
    fired = im.Firing(
        source="model",
        case="m",
        law="cand_square_neg",
        fires=1,
        base_cost=10.0,
        out_cost=5.0,
        changed=True,
        verified="pass",
    )
    quiet = im.Firing(source="bench", case="b", law="cand_pow_one")
    t = im._fire_table([fired, quiet], laws)
    assert "cand_square_neg" in t and "cand_pow_one" in t
    d = im._delta_table([fired, quiet])
    assert "cand_square_neg" in d and "yes" in d
    assert im._delta_table([quiet]) == "  (no law fired anywhere)"
    noted = im.Firing(
        source="model",
        case="m2",
        law="cand_pow_one",
        fires=1,
        note="kept the input term",
    )
    assert "kept the input term" in im._delta_table([noted])

    result = {
        "census": {
            "cand_square_neg": 0,
            "cand_sub_self": 3,
            "cand_pow_one": 2,
        },
        "firings": [
            im.Firing(
                source="model",
                case="m",
                law="cand_sub_self",
                fires=1,
            ),
            # A synthetic-only firing does not count as a real fire.
            im.Firing(
                source="synthetic",
                case="s",
                law="cand_pow_one",
                fires=1,
            ),
        ],
    }
    ct = im._census_table(result)
    assert "shape absent in the corpus" in ct
    assert "shape present AND fired" in ct
    assert "shape present, equality precondition fails" in ct


def test_reach_table_renders():
    row = {
        "model": "m",
        "base_enodes": 6,
        "add_enodes": 9,
        "base_classes": 5,
        "add_classes": 7,
        "base_cost": 4.0,
        "add_cost": 2.0,
        "changed": True,
        "base_cert": "pass",
        "add_cert": "pass",
    }
    t = im._reach_table([row])
    assert "6->9" in t and "yes" in t and "pass" in t
    row["add_cert"] = "FAIL (KeyError)"
    assert "FAIL" in im._reach_table([row])
    assert im._reach_table([]) == "  (no reach rows)"


def test_verdict_lines(capsys):
    firing = im.Firing(
        source="model",
        case="m",
        law="l",
        fires=1,
        base_cost=5.0,
        out_cost=2.0,
    )
    im._verdict({"firings": [firing], "reach": [{"changed": True}]})
    out = capsys.readouterr().out
    assert "1 real firing(s); 1 paid" in out
    assert "reach changed by adding the laws: 1" in out

    quiet = im.Firing(source="synthetic", case="s", law="l", fires=1)
    im._verdict({"firings": [quiet], "reach": []})
    out = capsys.readouterr().out
    assert "NONE of the 11 laws fires" in out


def test_run_impact_mini(mini_result):
    res = mini_result
    assert set(res["laws"]) == {law.name for law in im.new_laws()}
    assert res["bench_errors"] == []
    assert res["model_export_errors"] == []
    # Five cases (1 bench + 2 model + 2 synthetic) x 11 laws.
    assert len(res["firings"]) == 55
    fired = {f.law for f in res["firings"] if f.fires}
    assert {"cand_mul_factor", "cand_square_neg"} <= fired
    assert res["census"]["cand_mul_factor"] >= 1
    assert res["census"]["cand_square_neg"] >= 1
    assert res["n_real_subterms"] > 0
    # The quiet model contributes no whole-set firing row.
    assert res["set_fires"]
    assert all(not k.endswith(":quiet") for k in res["set_fires"])
    assert len(res["reach"]) == 2
    row = res["reach"][0]
    assert row["model"] == "sqneg" and row["changed"]
    assert res["reach"][1]["model"] == "quiet"


def test_print_report_and_dump(mini_result, tmp_path, capsys):
    im._print_report(mini_result)
    text = capsys.readouterr().out
    assert "firing table" in text
    assert "relaxed-pattern census" in text
    assert "cost delta" in text and "verdict" in text

    out = tmp_path / "impact.json"
    im._dump_json(str(out), mini_result)
    payload = json.loads(out.read_text())
    assert payload["laws"] == mini_result["laws"]
    assert len(payload["firings"]) == 55
    f0 = payload["firings"][0]
    assert set(f0) == {
        "source",
        "case",
        "law",
        "fires",
        "base_cost",
        "out_cost",
        "changed",
        "verified",
        "paid",
        "note",
    }
    assert payload["reach"][0]["model"] == "sqneg"


def test_print_report_lists_errors(mini_result, capsys):
    result = dict(mini_result)
    result["bench_errors"] = ["b: RuntimeError: x"]
    result["model_export_errors"] = ["m: ExportError: y"]
    result["set_fires"] = {}
    im._print_report(result)
    text = capsys.readouterr().out
    assert "bench build errors" in text
    assert "model export failures" in text
    assert "(the set fired on no case)" in text


def test_main_runs_report_and_json(
    mini_result, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(im, "run_impact", lambda: mini_result)
    out = tmp_path / "r.json"
    assert im.main(["--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "law_impact" in text and f"wrote {out}" in text
    assert json.loads(out.read_text())["laws"]
    # Report-only invocation — no JSON side file.
    assert im.main([]) == 0
