"""Tests for ``catopt_discovery.emit`` — the admission-artifact emitter.

``emit_admission`` turns a SHIP-verdict pipeline candidate into the
review artifact: a standalone law module, a generated test file, a
``tensor.py`` review patch and a markdown summary.  These tests run
the real emitter end-to-end against pipeline results measured on a
tiny synthetic corpus (two/three ``TermCase``s patched in for the
corpus loaders — same pattern as ``test_discovery_pipeline.py``), and
pin the refusal paths: non-SHIP candidates, names with no law
spelling, hooks that cannot be source-transcribed, and terms with no
literal rendering.

The emitted module is asserted to be *functional* — imported and its
``R(...)`` rule fired in a bare e-graph — not merely syntactically
plausible.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from catopt_core.egraph import EGraph
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_discovery import emit
from catopt_discovery import intake as li
from catopt_discovery import pipeline as pl
from catopt_discovery.census import (
    CorpusTerm,
    op_tuple_census,
    shape_census,
)
from catopt_discovery.impact import TermCase

_TENSOR_ABS = emit.REPO_ROOT / emit._TENSOR_REL


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _feed(*vars_: Var) -> tuple:
    return tuple(
        torch.randn(tuple(v.typ.shape), dtype=torch.float64)
        for v in vars_
    )


def _case(name: str, term: Op, *inputs: Var) -> TermCase:
    return TermCase(
        source="test",
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=_feed(*inputs),
        param_vals={},
    )


def _tiny_cases() -> list[TermCase]:
    """The emit corpus: a factoring site plus a manual softmax."""
    torch.manual_seed(0)
    x, y, z = _v("x", 4, 4), _v("y", 4, 4), _v("z", 4, 4)
    s = _v("s", 4, 4)
    e = _p("exp", s)
    return [
        _case(
            "factor",
            _p("add", _p("mul", x, y), _p("mul", x, z)),
            x,
            y,
            z,
        ),
        _case(
            "softmax",
            _p("div", e, _p("sum", e, dim=(-1,), keepdim=True)),
            s,
        ),
    ]


def _tiny_census(cases: list[TermCase]) -> dict:
    cts = [CorpusTerm("test", c.name, c.term) for c in cases]
    op_counts, op_terms = op_tuple_census(cts)
    sh_counts, _ = shape_census(cts)
    return {
        "n_terms": len(cts),
        "n_op_nodes": sum(op_counts.values()),
        "n_shapes": len(sh_counts),
        "op_tuples": [
            {
                "op": k[0],
                "children": list(k[1]),
                "count": n,
                "terms": len(op_terms[k]),
            }
            for k, n in op_counts.most_common()
        ],
    }


def _pipeline_result(monkeypatch, **kw) -> dict:
    """Run the real pipeline on the tiny corpus; return the result."""
    cases = _tiny_cases()
    census = _tiny_census(cases)
    monkeypatch.setattr(pl, "run_census", lambda top=400: census)
    monkeypatch.setattr(pl, "_bench_cases", lambda: ([], []))
    monkeypatch.setattr(pl, "model_cases", lambda: (cases, []))
    monkeypatch.setattr(li, "load_cases", lambda *a, **k: [])
    monkeypatch.setattr(li, "probe_cases", lambda *a, **k: [])
    kw.setdefault("vocab", "hand")
    return pl.run_pipeline(**kw)


def _load(path: Path, name: str):
    """Import the emitted module from *path*."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ship_ev(proposal: pl.Proposal, **kw) -> pl.Evidence:
    """A minimally-satisfying SHIP evidence for refusal-path tests."""
    ev = pl.Evidence(proposal=proposal)
    ev.num_true = True
    ev.fires = 1
    ev.fires_typed = 1
    ev.paid = 1
    for k, v in kw.items():
        setattr(ev, k, v)
    assert ev.shippable
    return ev


# ---------------------------------------------------------------------------
#  Rendering — term -> source and hook transcription
# ---------------------------------------------------------------------------


def test_term_src_renders_every_leaf_kind():
    tt = TensorType((2, 3))
    term = _p(
        "mul",
        Var("x", tt),
        _p("add", Param("w", tt), Const(2)),
    )
    src = emit._term_src(term)
    assert src == (
        "Op.make('mul', Var('x', TensorType((2, 3))), "
        "Op.make('add', Param('w', TensorType((2, 3))), Const(2)))"
    )
    # Metavariable leaves stay quoted strings; raw numbers -> Const.
    assert emit._term_src(_p("mul", "A", 5)) == (
        "Op.make('mul', 'A', Const(5))"
    )
    assert emit._term_src("M0") == "'M0'"


def test_term_src_refuses_unspellable_pieces():
    with pytest.raises(emit.Unemittable, match="no source spelling"):
        emit._term_src(object())
    # An attr key that is not a kwarg cannot spell an Op.make call.
    bad = _p("mul", _v("x", 2), _v("y", 2), **{"bad-key": 1})
    with pytest.raises(emit.Unemittable, match="not a kwarg"):
        emit._term_src(bad)
    # An attr value with no literal spelling.
    v = _p("mul", _v("x", 2), _v("y", 2), dim=_v("x", 2))
    with pytest.raises(emit.Unemittable, match="no literal spelling"):
        emit._term_src(v)


def test_literal_roundtrips_and_refuses():
    assert emit._literal((-1,)) == "(-1,)"
    assert emit._literal(True) == "True"
    assert emit._literal("dim") == "'dim'"
    with pytest.raises(emit.Unemittable):
        emit._literal(object())


def test_hook_src_transcribes_named_functions():
    src = emit._hook_src(pl._check_sum_keepdim)
    assert src.startswith("def _check_sum_keepdim(")
    assert "keepdim" in src


def test_hook_src_refuses_lambdas_and_non_functions():
    with pytest.raises(emit.Unemittable, match="not a named function"):
        emit._hook_src(42)
    # A lambda HAS source, but no `def <lambda>(` — the round-trip
    # check refuses it rather than emitting a wrong transcription.
    with pytest.raises(emit.Unemittable, match="did not round-trip"):
        emit._hook_src(lambda bound: True)


def test_safe_name_sanitizes_and_refuses():
    assert emit._safe_name("census:mul_select") == "census_mul_select"
    assert emit._safe_name("recognize:softmax") == "recognize_softmax"
    with pytest.raises(emit.Unemittable, match="no law spelling"):
        emit._safe_name("!!!")
    with pytest.raises(emit.Unemittable):
        emit._safe_name("9lives")


def test_test_name_sanitization():
    assert emit._test_name("$attr:RD", "extend") == "attr_rd_extend"
    assert emit._test_name("Foo", "Bar-Baz") == "foo_bar_baz"


# ---------------------------------------------------------------------------
#  The measured-perturbation machinery
# ---------------------------------------------------------------------------


def test_perturbations_by_value_kind():
    assert emit._perturbations(True) == [("flip", False)]
    assert emit._perturbations(3) == [("plus1", 4)]
    assert emit._perturbations(1.5) == [("plus1", 2.5)]
    assert emit._perturbations((-1,)) == [
        ("extend", (-1, -1)),
        ("scalar", -1),
    ]
    assert emit._perturbations((0, 1)) == [("extend", (0, 1, 1))]
    assert emit._perturbations("no") == []


def test_mv_counts_and_positions():
    lhs = _p(
        "div",
        _p("exp", "U"),
        _p("sum", _p("exp", "U"), dim="RD", keepdim="RK"),
    )
    counts = emit._mv_counts(lhs)
    assert counts["U"] == 2
    assert counts["$attr:RD"] == 1 and counts["$attr:RK"] == 1
    assert emit._mv_positions(lhs, "U") == [(0, 0), (1, 0, 0)]
    assert emit._mv_positions(lhs, "$attr:RD") == [(1,)]


def test_replace_leaf_and_attr():
    x = _v("x", 4)
    term = _p("mul", x, _p("select", x, dim=0, index=1))
    got = emit._replace_leaf(term, (1, 0), _v("z", 4))
    assert got.args[1].args[0].name == "z"
    got2 = emit._replace_attr(term, (1,), "dim", 1)
    assert got2.args[1].attrs["dim"] == 1
    # Path () replaces the root itself.
    assert emit._replace_leaf(term, (), x) is x


def test_subst_for_prefers_firing_case_then_match_term():
    proposal = pl.Proposal(
        "t:f",
        _p("add", _p("mul", "A", "B"), _p("mul", "A", "C")),
        _p("mul", "A", _p("add", "B", "C")),
        "t",
    )
    x, y, z = _v("x", 4), _v("y", 4), _v("z", 4)
    case = _case(
        "factor", _p("add", _p("mul", x, y), _p("mul", x, z)), x, y, z
    )
    subst = emit._subst_for(proposal, None, case)
    assert subst["A"] == x and subst["B"] == y and subst["C"] == z
    # Without a case the evidence's recorded match term is used.
    ev = pl.Evidence(proposal=proposal)
    ev.match_term = case.term
    assert emit._subst_for(proposal, ev, None)["B"] == y
    # Neither -> no honest binding, no guess.
    with pytest.raises(emit.Unemittable, match="no real LHS match"):
        emit._subst_for(proposal, pl.Evidence(proposal=proposal), None)


def test_mismatch_terms_breaks_repeated_metavars():
    proposal = pl.Proposal(
        "t:f",
        _p("add", _p("mul", "A", "B"), _p("mul", "A", "C")),
        _p("mul", "A", _p("add", "B", "C")),
        "t",
    )
    x, y, z = _v("x", 4), _v("y", 4), _v("z", 4)
    subst = {"A": x, "B": y, "C": z}
    mm = emit._mismatch_terms(proposal, subst)
    assert len(mm) == 1
    mv, bad = mm[0]
    assert mv == "A"
    # The second A occurrence was swapped for a different Var — the
    # matcher must veto and the rule must not fire.
    assert bad.args[0].args[0].name == "x"
    assert bad.args[1].args[0].name == "mv_mismatch"
    assert not emit._fires(bad, proposal.as_rule())


def test_probe_attrs_measures_declines_and_variants():
    e = _p("exp", "U")
    proposal = pl.Proposal(
        "t:sm",
        _p("div", e, _p("sum", e, dim="RD", keepdim="RK")),
        _p("softmax", "U", dim="SD"),
        "t",
        check=pl._check_sum_keepdim,
        derive=pl._derive_softmax_dim,
    )
    subst = {"U": _v("s", 4, 4), "$attr:RD": (-1,), "$attr:RK": True}
    probes = emit._probe_attrs(proposal, subst)
    by = {(p.key, p.label): p for p in probes}
    # keepdim flipped -> the check vetoes; extend -> multi-axis veto.
    assert by[("$attr:RK", "flip")].check_ok is False
    assert by[("$attr:RD", "extend")].check_ok is False
    # dim scalar (equivalent spelling) still checks out and fires.
    assert by[("$attr:RD", "scalar")].check_ok is True
    assert by[("$attr:RD", "scalar")].fired is True


def test_fires_in_bare_egraph():
    rule = pl.Proposal(
        "t:c", _p("mul", "A", "B"), _p("mul", "B", "A"), "t"
    ).as_rule()
    assert emit._fires(_p("mul", _v("x", 4), _v("y", 4)), rule)
    assert not emit._fires(_p("add", _v("x", 4), _v("y", 4)), rule)


def test_feed_src_refusal_paths():
    with pytest.raises(emit.Unemittable, match="no firing model"):
        emit._feed_src(None)
    # A term with no Var leaves has no inputs to feed.
    w = Param("w", TensorType((4, 4)))
    no_inputs = _case("params", _p("mul", w, w))
    with pytest.raises(emit.Unemittable, match="no inputs"):
        emit._feed_src(no_inputs)
    # Same Var name at two different types -> ambiguous feed.
    xa, xb = Var("x", TensorType((4,))), Var("x", TensorType((4, 4)))
    amb = _case("amb", _p("add", xa, _p("reshape", xb, shape=(4,))), xa)
    with pytest.raises(emit.Unemittable, match="two different types"):
        emit._feed_src(amb)
    # An unknown dim cannot spell a torch.randn call.
    unk = TermCase(
        source="test",
        name="unk",
        term=_p("neg", _v("x", None, 4)),
        inputs=(),
        feed=(),
        param_vals={},
    )
    with pytest.raises(emit.Unemittable, match="unknown dim"):
        emit._feed_src(unk)


def test_param_feed_src_collects_params_and_refuses_unknown():
    w = Param("w", TensorType((2, 2)))
    src = emit._param_feed_src(_p("mul", w, _v("x", 2)))
    assert "'w'" in src and "torch.randn((2, 2)" in src
    assert emit._param_feed_src(_v("x", 2)) == (
        "def _param_vals():\n    return {}\n"
    )
    bad = Param("b", TensorType((None, 4)))
    with pytest.raises(emit.Unemittable, match="unknown dim"):
        emit._param_feed_src(_p("mul", bad, bad))


# ---------------------------------------------------------------------------
#  emit_admission — refusals
# ---------------------------------------------------------------------------


def test_emit_refuses_unknown_candidate(tmp_path):
    em = emit.emit_admission({"ranked": []}, "ghost", tmp_path)
    assert not em.emitted
    assert "unknown candidate" in em.reason
    assert not list(tmp_path.iterdir())


def test_emit_refuses_non_ship_candidate(tmp_path):
    ev = pl.Evidence(
        proposal=pl.Proposal("t:x", _p("mul", "A", "B"), "A", "t")
    )
    ev.num_true = False
    result = {"ranked": [ev], "models": [], "holdout": None}
    em = emit.emit_admission(result, "t:x", tmp_path)
    assert not em.emitted
    assert "not a SHIP candidate" in em.reason
    assert "false" in em.reason


def test_emit_refuses_name_with_no_law_spelling(tmp_path):
    ev = _ship_ev(pl.Proposal("!!!", _p("mul", "A", "B"), "A", "t"))
    result = {"ranked": [ev], "models": [], "holdout": None}
    em = emit.emit_admission(result, "!!!", tmp_path)
    assert not em.emitted
    assert "no law spelling" in em.reason


def test_emit_refuses_when_no_real_binding(tmp_path):
    ev = _ship_ev(
        pl.Proposal(
            "t:nomatch",
            _p("add", _p("mul", "A", "B"), _p("mul", "A", "C")),
            _p("mul", "A", _p("add", "B", "C")),
            "t",
        )
    )
    result = {"ranked": [ev], "models": [], "holdout": None}
    em = emit.emit_admission(result, "t:nomatch", tmp_path)
    assert not em.emitted
    assert "no real LHS match" in em.reason


def test_emit_refuses_lambda_hook(tmp_path):
    ev = _ship_ev(
        pl.Proposal(
            "t:lam",
            _p("mul", "A", "B"),
            _p("mul", "B", "A"),
            "t",
            check=lambda bound: True,
        ),
        match_term=_p("mul", _v("x", 4), _v("y", 4)),
    )
    result = {"ranked": [ev], "models": [], "holdout": None}
    em = emit.emit_admission(result, "t:lam", tmp_path)
    assert not em.emitted
    assert "did not round-trip" in em.reason


def test_emit_refuses_unspellable_rhs(tmp_path):
    ev = _ship_ev(
        pl.Proposal(
            "t:bad_rhs",
            _p("mul", "A", "B"),
            _p("mul", "B", "A", **{"bad-key": 1}),
            "t",
        ),
        match_term=_p("mul", _v("x", 4), _v("y", 4)),
    )
    result = {"ranked": [ev], "models": [], "holdout": None}
    em = emit.emit_admission(result, "t:bad_rhs", tmp_path)
    assert not em.emitted
    assert "not a kwarg" in em.reason


# ---------------------------------------------------------------------------
#  emit_admission — the happy path, artifacts verified
# ---------------------------------------------------------------------------


def test_emit_factor_left_artifacts(monkeypatch, tmp_path):
    result = _pipeline_result(monkeypatch)
    em = emit.emit_admission(
        result, "factor_left", tmp_path, tensor_path=_TENSOR_ABS
    )
    assert em.emitted, em.reason
    names = {Path(f).name for f in em.files}
    assert names == {
        "admitted_factor_left.py",
        "test_admitted_factor_left.py",
        "admission_factor_left.patch",
        "admission_factor_left.md",
    }
    for f in em.files:
        assert Path(f).exists()

    # The standalone module is importable and its law really fires.
    mod = _load(tmp_path / "admitted_factor_left.py", "admitted_fl")
    rule = mod.FACTOR_LEFT
    assert rule.name == "factor_left"
    x, y, z = _v("x", 4), _v("y", 4), _v("z", 4)
    term = _p("add", _p("mul", x, y), _p("mul", x, z))
    eg = EGraph()
    root = eg.add_term(term)
    eg.run([rule], root, max_iterations=4, max_nodes=20_000)
    assert eg.rule_fires.get("factor_left", 0) >= 1

    # The patch is a review diff: the law block plus registration.
    patch = (tmp_path / "admission_factor_left.patch").read_text()
    assert "+FACTOR_LEFT = R(" in patch
    assert "+    FACTOR_LEFT," in patch
    assert "tests/test_admitted_factor_left.py" in patch
    # The law block lands before the rule collections section.
    assert patch.index("+FACTOR_LEFT = R(") < patch.index(
        "+    FACTOR_LEFT,"
    )

    # The generated test file compiles and carries the real probes.
    test = (tmp_path / "test_admitted_factor_left.py").read_text()
    compile(test, "test_admitted_factor_left.py", "exec")
    assert "def test_match_instantiate_roundtrip" in test
    assert "def test_fires_and_rhs_is_member" in test
    assert "def test_declines_on_a_mismatch" in test
    assert "def test_end_to_end_on_factor" in test

    md = (tmp_path / "admission_factor_left.md").read_text()
    assert "SHIP verdict" in md and "factor_left" in md
    assert any("declines measured" in n for n in em.notes)


def test_emit_hooked_candidate_transcribes_hooks(monkeypatch, tmp_path):
    result = _pipeline_result(monkeypatch, holdout="softmax_fold")
    em = emit.emit_admission(
        result, "recognize:softmax", tmp_path, tensor_path=_TENSOR_ABS
    )
    assert em.emitted, em.reason
    mod = _load(
        tmp_path / "admitted_recognize_softmax.py", "admitted_sm"
    )
    # The hooks arrived verbatim: check vetoes non-keepdim, derive
    # unwraps the reduce-dim tuple.
    assert not mod._check_sum_keepdim(
        {"$attr:RK": False, "$attr:RD": (-1,)}
    )
    assert mod._check_sum_keepdim({"$attr:RK": True, "$attr:RD": (-1,)})
    assert mod._derive_softmax_dim({"$attr:RD": (-1,)}) == {
        "$attr:SD": -1
    }
    # The folded rule fires on the manual softmax.
    s = _v("s", 4, 4)
    e = _p("exp", s)
    term = _p("div", e, _p("sum", e, dim=(-1,), keepdim=True))
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(
        [mod.RECOGNIZE_SOFTMAX],
        root,
        max_iterations=4,
        max_nodes=20_000,
    )
    assert eg.rule_fires.get("recognize_softmax", 0) >= 1

    test = (tmp_path / "test_admitted_recognize_softmax.py").read_text()
    compile(test, "test_admitted_recognize_softmax.py", "exec")
    # The measured declines and the accepted variant made it in.
    assert "def test_declines_on_" in test
    assert "def test_derive_produces_the_rhs_attrs" in test
    assert "def test_check_accepts_the_real_binding" in test
    assert "def test_end_to_end_on_softmax" in test
    # The measured probe counts are reported honestly in the notes.
    assert any("declines measured+emitted: 2" in n for n in em.notes)
    assert any("accepted variants" in n for n in em.notes)
    # The hook names already exist in tensor.py?  The review note is
    # honest either way — assert the note list was built.
    assert any("review points" in n for n in em.notes)


def test_emit_uses_match_term_when_models_absent(monkeypatch, tmp_path):
    """``result['models']`` may be empty (a serialized pipeline run);
    the emitter then binds the tests to the recorded match term and
    simply omits the e2e model test."""
    result = _pipeline_result(monkeypatch)
    ev = next(
        e for e in result["ranked"] if e.proposal.name == "factor_left"
    )
    result["models"] = []
    assert ev.match_term is not None
    em = emit.emit_admission(
        result, "factor_left", tmp_path, tensor_path=_TENSOR_ABS
    )
    assert em.emitted, em.reason
    test = (tmp_path / "test_admitted_factor_left.py").read_text()
    assert "test_match_instantiate_roundtrip" in test
    assert "test_end_to_end_on" not in test


def _check_glu_fold(bound: dict) -> bool:
    """Deliberate name twin of the shipped hook (collision probe)."""
    return True


def test_emit_hook_name_collision_is_noted(tmp_path):
    """A proposer hook whose name already exists in tensor.py lands a
    review note, not a silent shadow."""
    ev = _ship_ev(
        pl.Proposal(
            "t:collide",
            _p("mul", "A", "B"),
            _p("mul", "B", "A"),
            "t",
            check=_check_glu_fold,
        ),
        match_term=_p("mul", _v("x", 4), _v("y", 4)),
    )
    result = {"ranked": [ev], "models": [], "holdout": None}
    em = emit.emit_admission(
        result, "t:collide", tmp_path, tensor_path=_TENSOR_ABS
    )
    assert em.emitted, em.reason
    assert any(
        "_check_glu_fold" in n and "already exists" in n
        for n in em.notes
    )
    # The emitted module still carries the transcription verbatim.
    mod = _load(tmp_path / "admitted_t_collide.py", "admitted_tc")
    assert mod._check_glu_fold({}) is True


# ---------------------------------------------------------------------------
#  Second pass — the remaining rendering and refusal edges
# ---------------------------------------------------------------------------


def test_term_src_bool_and_int_leaves():
    # bool must come before int in the leaf ladder (bool is int).
    assert emit._term_src(True) == "Const(True)"
    assert emit._term_src(7) == "Const(7)"
    assert emit._term_src(0.5) == "Const(0.5)"


def test_hook_src_refuses_sourceless_function():
    ns: dict = {}
    exec("def _no_src(bound):\n    return True\n", ns)
    with pytest.raises(emit.Unemittable, match="no source for hook"):
        emit._hook_src(ns["_no_src"])


def test_mismatch_terms_repeated_attr_metavar():
    """A shared attr metavar is a repeated binding too — the mismatch
    probe perturbs the second occurrence's attr."""
    proposal = pl.Proposal(
        "t:sel_pair",
        _p(
            "mul",
            _p("select", "A", dim="D", index="I"),
            _p("select", "B", dim="D", index="I"),
        ),
        _p("select", _p("mul", "A", "B"), dim="D", index="I"),
        "t",
    )
    u, w = _v("u", 4, 4), _v("w", 4, 4)
    matched = _p(
        "mul",
        _p("select", u, dim=0, index=1),
        _p("select", w, dim=0, index=1),
    )
    from catopt_core.egraph.terms import _term_match

    subst = _term_match(proposal.lhs, matched)
    assert subst is not None
    mm = emit._mismatch_terms(proposal, subst)
    kinds = {mv for mv, _ in mm}
    # Both shared attr metavars get a perturbed twin term.
    assert kinds == {"$attr:D", "$attr:I"}
    for _mv, bad in mm:
        assert not emit._fires(bad, proposal.as_rule())


def test_probe_attrs_without_check_reports_none():
    """A proposal with no ``check`` hook reports ``check_ok=None`` —
    the probe never pretends a veto that does not exist."""
    proposal = pl.Proposal(
        "t:nocheck",
        _p("mul", _p("select", "A", dim="D", index="I"), "B"),
        _p("select", _p("mul", "A", "B"), dim="D", index="I"),
        "t",
    )
    subst = {
        "A": _v("a", 4, 4),
        "B": _v("b", 4, 4),
        "$attr:D": 0,
        "$attr:I": 1,
    }
    probes = emit._probe_attrs(proposal, subst)
    assert probes
    assert all(pr.check_ok is None for pr in probes)
    # dim+1 still fires (a different but valid select).
    assert any(pr.fired for pr in probes)


def _check_admits(bound: dict) -> bool:
    """A named check for the synthetic _test_module proposal."""
    return True


def test_test_module_decline_and_variant_emission():
    """The emitted file carries only measured outcomes: a decline with
    no observable veto is skipped; variants beyond the cap are too."""
    proposal = pl.Proposal(
        "t:p",
        _p("mul", _p("select", "A", dim="D", index="I"), "B"),
        _p("select", _p("mul", "A", "B"), dim="D", index="I"),
        "t",
        check=_check_admits,
    )
    subst = {
        "A": _v("a", 4, 4),
        "B": _v("b", 4, 4),
        "$attr:D": 0,
        "$attr:I": 1,
    }
    probes = [
        # Decline with no check veto and an unmintable LHS — nothing
        # to assert, so no test is emitted for it.
        emit._Probe("$attr:D", "bogus", "?", None, False, False),
        # Decline: check vetoes AND the term still mints — both
        # assertions land.
        emit._Probe("$attr:D", "plus1", 1, False, True, False),
        # Three measured variants — the cap keeps two.
        *[
            emit._Probe("$attr:D", f"v{i}", i, True, True, True)
            for i in range(3)
        ],
    ]
    ev = pl.Evidence(proposal=proposal)
    src = emit._test_module(
        "t_p", "T_P", proposal, ev, None, subst, probes, [], None
    )
    compile(src, "test_admitted_t_p.py", "exec")
    assert src.count("def test_fires_on_variant_") == 2
    assert "def test_declines_on_attr_d_plus1" in src
    # The unassertable decline produced no test.
    assert "bogus" not in src


def test_ruff_format_passthrough_without_ruff(tmp_path, monkeypatch):
    """No vendored ruff -> the source passes through unformatted."""
    monkeypatch.setattr(emit, "REPO_ROOT", tmp_path)
    src = "x   =  1\n"
    assert emit._ruff_format(src) == src


def test_emit_still_succeeds_without_ruff(monkeypatch, tmp_path):
    monkeypatch.setattr(emit, "REPO_ROOT", tmp_path)
    ev = _ship_ev(
        pl.Proposal(
            "t:noruff",
            _p("mul", "A", "B"),
            _p("mul", "B", "A"),
            "t",
        ),
        match_term=_p("mul", _v("x", 4), _v("y", 4)),
    )
    result = {"ranked": [ev], "models": [], "holdout": None}
    em = emit.emit_admission(
        result, "t:noruff", tmp_path / "o", tensor_path=_TENSOR_ABS
    )
    assert em.emitted, em.reason


def test_subst_for_falls_through_unmatched_case():
    """A firing case whose term contains no LHS-shaped subterm is
    skipped honestly — the recorded match term binds instead."""
    proposal = pl.Proposal(
        "t:f",
        _p("add", _p("mul", "A", "B"), _p("mul", "A", "C")),
        _p("mul", "A", _p("add", "B", "C")),
        "t",
    )
    x, y, z = _v("x", 4), _v("y", 4), _v("z", 4)
    # The case's term does not match the LHS at all.
    other = _case("other", _p("neg", x), x)
    ev = pl.Evidence(proposal=proposal)
    ev.match_term = _p("add", _p("mul", x, y), _p("mul", x, z))
    subst = emit._subst_for(proposal, ev, other)
    assert subst["B"] == y and subst["C"] == z


def test_mismatch_terms_skips_unperturbable_and_single_position():
    """A repeated attr metavar bound to an unperturbable value, or one
    used twice at a single node, yields no mismatch term."""
    # ``mode`` metavar bound to a string — no perturbation exists.
    p1 = pl.Proposal(
        "t:s",
        _p("mul", _p("f", "A", mode="M"), _p("f", "B", mode="M")),
        "A",
        "t",
    )
    base1 = _p(
        "mul",
        _p("f", _v("a", 4), mode="x"),
        _p("f", _v("b", 4), mode="x"),
    )
    from catopt_core.egraph.terms import _term_match

    subst1 = _term_match(p1.lhs, base1)
    assert subst1 is not None and "$attr:M" in subst1
    assert emit._mismatch_terms(p1, subst1) == []
    # The same attr metavar twice on ONE node perturbs exactly one of
    # the two slots — a real attr mismatch the matcher vetoes.
    p2 = pl.Proposal(
        "t:one",
        _p("mul", _p("select", "A", dim="D", index="D"), "B"),
        "A",
        "t",
    )
    base2 = _p(
        "mul", _p("select", _v("a", 4, 4), dim=0, index=0), _v("b", 4)
    )
    subst2 = _term_match(p2.lhs, base2)
    assert subst2 is not None
    mm2 = emit._mismatch_terms(p2, subst2)
    assert [mv for mv, _ in mm2] == ["$attr:D"]
    bad = mm2[0][1]
    sel = bad.args[0]
    assert sel.attrs["dim"] != sel.attrs["index"]
    assert not emit._fires(bad, p2.as_rule())
