"""Audit: the 11 rules the round-4 intake corpus reports ``shippable``.

Corpus round 4 (``project/retros/corpus-round-4.md``) reported
``shippable: 0 -> 11`` from the pipeline's own ``measure`` — the
grammar spellings and shared-factor laws the new workloads made
reachable.  That is the pipeline's verdict, not the gauntlet's.
This audit runs each of the 11 through the *full* gauntlet
(``evidence.run_gauntlet``) and pins the result: all 11 clear —
``usable: yes`` — and the pipeline's ``shippable`` and the
gauntlet's ``usable`` agree on every one.

The audit's second finding is *why* they agree.  All 11 are
unguarded (no ``cond``, no ``check``), so the gauntlet's
guarded-region sweep — its one genuinely stricter stage — never
runs for them: the truth gate takes the unguarded branch and the
two bars reduce to the same single-instance numeric oracle.  The
bars do differ systematically in two places, pinned here as unit
facts (the last two tests): the gauntlet's truth is *stricter* when
a derivation meets a measured counterexample, and its closure gate
is *looser* (attributable-only cert failures).  Neither bites the
11: each measures ``num_true is True`` with ``cert_fail == 0``.

The synthetic corpus below carries one concrete spelling site per
rule, so each fires and pays without the real (heavy) intake
corpus.  ``project/retros/shippable-audit.md`` records the real
run.
"""

import functools

import pytest
import torch
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import evidence as ev
from catopt_discovery import pipeline as pl
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink

#: The 11 rules the round-4 intake corpus reports ``shippable``
#: (``corpus-round-4.md``): the shared-factor algebra, the literal
#: grammar spellings and the scalar-corner annihilators.
NAMES = (
    "sub_self",
    "grammar:pow_one",
    "reshape_reshape",
    "square_neg",
    "factor_left",
    "factor_right",
    "factor_sub_left",
    "factor_sub_right",
    "neg_add",
    "exp_add",
    "sub_neg",
)


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _case(name: str, term: Op, *inputs: Var) -> TermCase:
    return TermCase(
        source="test",
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=tuple(
            torch.randn(tuple(x.typ.shape), dtype=torch.float64)
            for x in inputs
        ),
        param_vals={},
    )


def _sites() -> dict:
    """Return one concrete spelling site (its LHS) per rule."""
    u, v, w = _v("u", 4, 8), _v("v", 4, 8), _v("w", 4, 8)
    s = _v("s")
    return {
        "sub_self": _case("c_sub_self", _p("sub", s, s), s),
        "grammar:pow_one": _case(
            "c_pow_one", _p("pow", u, Const(1)), u
        ),
        "reshape_reshape": _case(
            "c_reshape2",
            _p(
                "reshape",
                _p("reshape", u, shape=(2, 4, 4)),
                shape=(2, 16),
            ),
            u,
        ),
        "square_neg": _case(
            "c_square_neg", _p("square", _p("neg", u)), u
        ),
        "factor_left": _case(
            "c_factor_left",
            _p("add", _p("mul", u, v), _p("mul", u, w)),
            u,
            v,
            w,
        ),
        "factor_right": _case(
            "c_factor_right",
            _p("add", _p("mul", v, u), _p("mul", w, u)),
            u,
            v,
            w,
        ),
        "factor_sub_left": _case(
            "c_factor_sub_left",
            _p("sub", _p("mul", u, v), _p("mul", u, w)),
            u,
            v,
            w,
        ),
        "factor_sub_right": _case(
            "c_factor_sub_right",
            _p("sub", _p("mul", v, u), _p("mul", w, u)),
            u,
            v,
            w,
        ),
        "neg_add": _case(
            "c_neg_add",
            _p("add", _p("neg", u), _p("neg", v)),
            u,
            v,
        ),
        "exp_add": _case(
            "c_exp_add",
            _p("mul", _p("exp", u), _p("exp", v)),
            u,
            v,
        ),
        "sub_neg": _case("c_sub_neg", _p("sub", u, _p("neg", v)), u, v),
    }


@functools.lru_cache(maxsize=1)
def _corpus() -> ev.GauntletCorpus:
    """The synthetic corpus: every rule's spelling site, once."""
    sink = _sink()
    sites = _sites()
    return ev.GauntletCorpus(
        real_terms=tuple(c.term for c in sites.values()),
        probe=tuple(sites.values()),
        base_rules=tuple(ALL_RULES),
        census_op={},
        sink=sink,
        cost_fn=_cost_fn(sink),
    )


@functools.lru_cache(maxsize=1)
def _props() -> dict:
    """Return the 11 pipeline proposals keyed by name."""
    return {p.name: p for p in [*pl._shape_aware(), *pl._grammar()]}


def _measure(name: str) -> pl.Evidence:
    """Run the pipeline's ``measure`` on one rule and the corpus."""
    corpus = _corpus()
    lib = [pl.lp._key(r.lhs, r.rhs) for r in corpus.base_rules]
    return pl.measure(
        _props()[name],
        list(corpus.real_terms),
        list(corpus.probe),
        list(corpus.base_rules),
        lib,
        corpus.census_op,
        corpus.sink,
        corpus.cost_fn,
    )


def _gauntlet(tmp_path, name: str) -> ev.Gauntlet:
    """Store one rule as an object and run the full gauntlet."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(conn, _props()[name].as_rule(), kind="law")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus())
    conn.close()
    return rep


# ---------------------------------------------------------------------------
#  The per-rule verdict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_shippable_rule_clears_the_gauntlet(tmp_path, name):
    """Every one of the 11 is ``usable`` — all eight gates pass."""
    rep = _gauntlet(tmp_path, name)
    assert rep.usable, rep.reason
    assert all(s.passed for s in rep.stages)
    # It genuinely fires and pays on its site — not a vacuous admit.
    assert rep.evidence.fires == 1
    assert rep.evidence.fires_ill_typed == 0
    assert rep.evidence.paid == 1
    assert rep.evidence.cert_fail == 0


@pytest.mark.parametrize("name", NAMES)
def test_pipeline_shippable_and_gauntlet_usable_agree(tmp_path, name):
    """No refusal anywhere: the two bars coincide on all 11."""
    evd = _measure(name)
    rep = _gauntlet(tmp_path, name)
    assert evd.shippable is True, evd.no_ship_reason
    assert rep.usable is True, rep.reason
    assert evd.shippable == rep.usable


@pytest.mark.parametrize("name", NAMES)
def test_shippable_rules_are_unguarded(name):
    """The structural reason the bars agree: no guard to sweep."""
    rule = _props()[name].as_rule()
    assert rule.cond is None and rule.check is None


def test_gauntlet_truth_takes_the_unguarded_branch():
    """An unguarded rule's truth gate never sweeps a region."""
    rep = ev.Gauntlet(alpha_key="k")
    rep.evidence = pl.Evidence(
        proposal=_props()["sub_self"], num_true=True
    )
    rule = _props()["sub_self"].as_rule()
    assert ev._truth_gate(rep, rule, None, None) is True
    assert rep.synth_region is None and rep.real_region is None
    assert "guarded:" not in rep.stages[-1].detail


# ---------------------------------------------------------------------------
#  The two systematic asymmetries — real, but dormant for the 11
# ---------------------------------------------------------------------------


def test_gauntlet_truth_is_stricter_on_a_derivable_counterexample():
    """The round-4 tightening: a measured counterexample outranks a
    derivation.  The pipeline's ``truth`` still lets ``derivable``
    win, so the bars diverge here — though not for the 11 (all
    measure ``num_true is True``, not ``False``)."""
    prop = _props()["sub_self"]
    evd = pl.Evidence(proposal=prop, num_true=False, derivable=True)
    assert evd.truth is True  # pipeline: derivation wins
    rep = ev.Gauntlet(alpha_key="k")
    rep.evidence = evd
    assert ev._truth_gate(rep, prop.as_rule(), None, None) is False
    assert rep.stages[-1].detail == "numeric=False derivable=True"


def test_gauntlet_closure_is_looser_on_an_ambient_cert_failure():
    """The gauntlet charges only *attributable* cert failures; the
    pipeline's ``shippable`` demands ``cert_fail == 0``.  An ambient
    failure (the base replay was already broken) refuses the pipeline
    but not the gauntlet — again not biting the 11."""
    prop = _props()["sub_self"]
    evd = pl.Evidence(
        proposal=prop,
        num_true=True,
        relation="new",
        fires=1,
        fires_typed=1,
        paid=1,
        verify_fail=0,
        cert_fail=1,
        closure_ratio=1.0,
    )
    evd.reach = ({"add_cert": "pass", "base_cert": "pass"},)
    assert evd.shippable is False  # pipeline: any cert failure
    assert evd.no_ship_reason == "certificate fails to replay"
    rep = ev.Gauntlet(alpha_key="k")
    rep.evidence = evd
    assert ev._closure_gate(rep) is True
    assert "ambient=1" in rep.stages[-1].detail
