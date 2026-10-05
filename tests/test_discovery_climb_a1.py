"""Coverage climb A1 — the driver paths of ``shape_proposal`` /
``proposal``.

``test_discovery_proposal.py`` exercises the per-schema machinery
(matches, firing, tables) on synthetic cases; the only pieces left
unmeasured are the two ``main``/``run_*`` drivers and a handful of
dedup guards.  These tests drive both drivers end-to-end on a
three-case synthetic corpus — the corpus functions are swapped for
tiny ``TermCase`` lists (the established discovery-test pattern), the
search, oracles and reporting all run for real.
"""

import json

from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import proposal as pp
from catopt_discovery import shape_proposal as sp
from catopt_discovery.impact import TermCase

_BY_NAME = {r.name: r for r in ALL_RULES}
_D = 4


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _sch() -> dict:
    return {s.name: s for s in sp.schemas()}


def _case(name: str, term: Op, source: str = "test") -> TermCase:
    return TermCase(
        name=name,
        source=source,
        term=term,
        inputs=(),
        feed=(),
        param_vals={},
    )


# ---------------------------------------------------------------------------
#  shape_proposal — run_proposal + main on a three-case corpus
# ---------------------------------------------------------------------------


def _tiny_corpus() -> tuple[list[TermCase], list[TermCase]]:
    """A bench/model split where factor_left and neg_add both apply."""
    x, y, z = (_v(n, _D, _D) for n in "xyz")
    bench = [
        _case("b_fact", _p("add", _p("mul", x, y), _p("mul", x, z))),
        _case("b_plain", _p("add", x, y)),
    ]
    models = [
        _case("m_fact", _p("add", _p("mul", x, y), _p("mul", x, z))),
        _case("m_neg", _p("add", _p("neg", x), _p("neg", y))),
    ]
    return bench, models


def test_run_proposal_and_main_tiny_corpus(
    monkeypatch, tmp_path, capsys
):
    bench, models = _tiny_corpus()
    sch = _sch()
    keep = [sch["factor_left"], sch["neg_add"], sch["mul_zero"]]
    monkeypatch.setattr(sp, "_bench_cases", lambda: (bench, []))
    monkeypatch.setattr(sp, "model_cases", lambda: (models, []))
    monkeypatch.setattr(sp, "schemas", lambda: keep)

    result = sp.run_proposal()
    assert result["n_bench"] == 2 and result["n_models"] == 2
    by_name = {o.schema.name: o for o in result["outcomes"]}
    fl = by_name["factor_left"]
    assert fl.matches >= 1 and fl.num_true is True
    assert fl.useful and fl.model_fires >= 1 and fl.fire_paid >= 1
    neg = by_name["neg_add"]
    assert neg.matches >= 1 and neg.model_fires >= 1
    # a schema whose LHS never appears reports honest zeros.
    assert by_name["mul_zero"].matches == 0
    assert by_name["mul_zero"].model_fires == 0
    # the 11-law baseline ran on the same models for comparison.
    assert len(result["baseline"]["laws"]) == 11
    # factor_left fires on m_fact, so the end-to-end reach table has
    # at least one row for it.
    assert any(r["schema"] == "factor_left" for r in result["reach"])

    out = tmp_path / "shape.json"
    rc = sp.main(["--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_bench"] == 2 and payload["n_models"] == 2
    row = {
        o["schema"]: o for o in payload["outcomes"]
    }["factor_left"]
    assert row["useful"] is True and row["fire_paid"] >= 1
    printed = capsys.readouterr().out
    assert "laws aimed at real shapes" in printed
    assert "factor_left" in printed
    assert "== verdict ==" in printed
    # and the plain (no --json) driver exits cleanly.
    assert sp.main([]) == 0
    assert "wrote" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  proposal — _det_rep / _class_reps edges the e-graph can't reach,
#  composite dedup, and main
# ---------------------------------------------------------------------------


def test_composite_skips_tautological_pairs():
    # An inverse pair: r2 folds m1 straight back onto r1's LHS, so the
    # composite is a tautology and is never proposed.
    uni = [_BY_NAME["silu_expand"], _BY_NAME["silu_fold"]]
    assert pp.composite_candidates(uni, max_cands=10) == []


def test_composite_dedups_alpha_identical_candidates():
    # ``factor_matmul`` and ``weight_factor_matmul`` spell the same
    # equality alpha-normally: both instances' ``comm_add`` composites
    # key identically, so only the first survives the ``seen`` dedup.
    uni = [
        _BY_NAME["factor_matmul"],
        _BY_NAME["weight_factor_matmul"],
        _BY_NAME["comm_add"],
    ]
    out = pp.composite_candidates(uni, max_cands=50)
    assert [c.label for c in out] == ["factor_matmul+comm_add"]


def test_proposal_main_small_pool(monkeypatch, tmp_path, capsys):
    """``main`` on a hand-shrunk pool: real evaluation and ranking,
    tiny candidate list."""
    by = {c.label: c for c in pp.schema_candidates()}
    pool = [
        by["mul_factor"],
        by["FALSE_square_add"],
        by["sub_to_add_dup"],
    ]
    monkeypatch.setattr(
        pp, "near_miss_candidates", lambda seeds: pool
    )
    monkeypatch.setattr(pp, "composite_candidates", lambda: [])
    monkeypatch.setattr(pp, "schema_candidates", lambda: [])

    out = tmp_path / "pp.json"
    rc = pp.main(["--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert len(payload["outcomes"]) == 3
    assert payload["yield"]["schema"]["proposed"] == 3
    printed = capsys.readouterr().out
    assert "yield per strategy" in printed
    assert "genuinely-new" in printed
    # mul_factor is useful, so both ranking rows print precisions.
    assert printed.count("average precision") == 2
    assert "wrote" in printed


def test_proposal_main_empty_pool_is_honest(monkeypatch, capsys):
    """No candidates at all: the ranking section says 'undefined'
    rather than fabricating a precision."""
    monkeypatch.setattr(pp, "near_miss_candidates", lambda seeds: [])
    monkeypatch.setattr(pp, "composite_candidates", lambda: [])
    monkeypatch.setattr(pp, "schema_candidates", lambda: [])
    assert pp.main([]) == 0
    printed = capsys.readouterr().out
    assert "no useful candidate — undefined" in printed
    assert "(none" in printed  # _fmt_useful's honest empty row
