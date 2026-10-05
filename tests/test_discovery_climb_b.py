"""Second-pass coverage climb for ``catopt_discovery``.

The first pass (``test_discovery_*.py``) left a handful of defensive
branches and CLI edges uncovered.  This file reaches every one that a
*real* input can reach:

* ``oracle.synthesize`` — the per-view instantiation probe skips a
  node whose free metavariables are not yet bound (a view wrapped
  around a pointwise skeleton), and skips an instantiation whose
  pattern cannot be re-minted (a ``validate=False`` non-canonical
  attr spelling — the documented escape hatch).
* ``emit`` — ``_mv_counts`` walks concrete leaves/attrs, ``_probe_attrs``
  records a raising check and an uninstantiable binding honestly,
  ``_subst_for`` refuses a ``match_term`` that does not match, and
  ``_insert_tensor`` backs up over narrative lines above the
  collections anchor.
* ``pipeline`` — ``_app_typed`` reports a binding that no longer
  resolves, ``_typed_probe`` on a non-firing rule, an ill-typed
  *tied* extraction (pick_ill without a cost drop — the minted member
  wins the deterministic enode-order tie-break but pays nothing),
  an unchanged-but-ill-typed fire, and a non-``"pass"`` reach
  certificate.
* ``measure`` — the view oracle's ``unproven`` verdict leaves
  ``num_true`` untouched.
* ``coherence._confluence_probe`` — a recorded instance that does
  not host the base rule is skipped, not trusted.
* ``lemma_cert._enumerated_cert`` — an enumerated path that fails
  strict replay is skipped, and a probe row with no certificate and
  no note prints neither.
* ``census`` / ``vocab`` / ``coherence`` ``main`` — the ``--json``-less
  paths.

The defensive lines no honest input can reach are listed in the task
report, not covered here.
"""

import pytest
import torch
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import census as cs
from catopt_discovery import coherence as lc
from catopt_discovery import emit
from catopt_discovery import lemma_cert as llc
from catopt_discovery import oracle as lvo
from catopt_discovery import pipeline as pl
from catopt_discovery import vocab as vc
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink

# ---------------------------------------------------------------------------
#  Helpers (same shapes as test_discovery_pipeline / _emit)
# ---------------------------------------------------------------------------


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


@pytest.fixture
def sink_cost():
    sink = _sink()
    return sink, _cost_fn(sink)


@pytest.fixture
def lib_keys() -> list:
    import catopt_discovery.proposal as lp

    return [lp._key(r.lhs, r.rhs) for r in ALL_RULES]


# ---------------------------------------------------------------------------
#  oracle.py — the instantiation skip paths
# ---------------------------------------------------------------------------


def test_synthesize_skips_unbound_view_probe():
    """A view node wrapping *free* metavars cannot be shape-probed.

    ``lhs_views`` probes each LHS view node's output shape by
    instantiating it under the partial (viewed-only) binding — but
    ``unsqueeze(add(A, B), dim=D)``'s operand holds metavars that are
    bound only later, so the probe instantiation raises ``KeyError``
    and the node is honestly skipped rather than guessed.
    """
    lhs = _p("unsqueeze", _p("add", "A", "B"), dim="D")
    rhs = _p(
        "add",
        _p("unsqueeze", "A", dim="D"),
        _p("unsqueeze", "B", dim="D"),
    )
    insts = lvo.synthesize(lhs, rhs)
    # Enumeration still completes: unsqueeze distributes over add, so
    # real evaluable instances exist alongside the skipped probe.
    assert insts
    assert any(i.outcome == "equal" for i in insts)


def test_synthesize_skips_unmintable_pattern():
    """A non-canonical pattern cannot be instantiated — skip, honestly.

    ``arg9`` is not a declared attr position for ``getitem``, so the
    pattern can only be minted with ``validate=False``.  At
    instantiation the metavariable stays unbound (no domain entry
    produces key ``arg9``), the literal metavar name reaches
    ``Op.make``, and the mint raises ``ValueError`` — every candidate
    instance is skipped and the verdict is honestly ``unproven``.
    """
    lhs = Op.make("getitem", "A", arg9="I", validate=False)
    assert lvo.synthesize(lhs, "A") == []
    v = lvo.verify_view_candidate("t:arg9", lhs, "A", [])
    assert v.verdict == "unproven"
    assert v.note == "no evaluable instance"


# ---------------------------------------------------------------------------
#  emit.py — counting, probing, binding, insertion edges
# ---------------------------------------------------------------------------


def test_mv_counts_visits_concrete_leaves_and_attrs():
    """Concrete (non-metavar) leaves and attr values count nothing."""
    pat = _p("add", "A", _v("x", 2))
    assert emit._mv_counts(pat) == {"A": 1}
    # A concrete attr value is walked but not counted as a metavar.
    pat2 = _p("unsqueeze", "A", dim=0)
    assert emit._mv_counts(pat2) == {"A": 1}


def test_probe_attrs_raising_check_and_uninstantiable():
    """A raising check is a veto; an unmintable term is not fired.

    The ``scalar`` perturbation of the ``(5,)``-bound tuple metavar
    yields an ``int``, which the check subscripts — the raise is
    recorded as ``check_ok=False`` rather than propagated.  The
    perturbed LHS then fails to re-mint (the ``B`` metavar is never
    bound — ``instantiate_pattern`` raises ``KeyError``), so the
    probe records ``mintable=False`` and never reaches ``_fires``
    (which would call the same raising check inside the e-graph).
    """

    def _check_tuple_dim(bound):
        return bound["$attr:T"][0] > 0

    proposal = pl.Proposal(
        "t:td",
        _p("mul", "A", "B"),
        "A",
        "t",
        check=_check_tuple_dim,
    )
    subst = {"A": _v("a", 4, 4), "$attr:T": (5,)}
    probes = emit._probe_attrs(proposal, subst)
    by = {(p.key, p.label): p for p in probes}
    scalar = by[("$attr:T", "scalar")]
    assert scalar.check_ok is False
    assert scalar.mintable is False and scalar.fired is False
    extend = by[("$attr:T", "extend")]
    assert extend.check_ok is True
    assert extend.mintable is False and extend.fired is False


def test_probe_attrs_records_uninstantiable_binding():
    """A binding missing a leaf metavar fails the mint honestly."""
    proposal = pl.Proposal("t:mi", _p("mul", "A", "B"), "A", "t")
    # ``B`` is never bound — instantiate_pattern raises KeyError; the
    # probe records mintable=False rather than guessing a term.
    probes = emit._probe_attrs(
        proposal, {"A": _v("a", 4), "$attr:D": 0}
    )
    assert len(probes) == 1
    (p,) = probes
    assert p.key == "$attr:D" and p.mintable is False
    assert p.fired is False and p.check_ok is None


def test_subst_for_refuses_nonmatching_match_term():
    """A recorded ``match_term`` that does not match is not a binding."""
    proposal = pl.Proposal("t:mm", _p("mul", "A", "B"), "A", "t")
    ev = pl.Evidence(proposal=proposal)
    x, y = _v("x", 4), _v("y", 4)
    ev.match_term = _p("add", x, y)  # no mul anywhere
    with pytest.raises(emit.Unemittable, match="no real LHS match"):
        emit._subst_for(proposal, ev, None)


def test_insert_tensor_backs_up_over_narrative_lines():
    """The collections anchor may sit below non-divider lines.

    When ``#  Rule collections`` is not immediately preceded by the
    ``# ---`` divider, the inserter walks backwards over the
    narrative line(s) before inserting the block above the divider.
    """
    src = (
        "x = 1\n"
        "# ----------------------------------------------------------------------\n"
        "# narrative comment between divider and title\n"
        "#  Rule collections\n"
        "SIMPLIFICATION_RULES = [\n"
        "    FOO,\n"
        "]\n"
    )
    out = emit._insert_tensor(src, "BAR = R()", "BAR")
    assert "BAR = R()" in out
    assert "    BAR,\n" in out
    # The block lands above the divider, not inside the title text.
    assert out.index("BAR = R()") < out.index("#  Rule collections")


# ---------------------------------------------------------------------------
#  pipeline.py — typedness audit + reach edges
# ---------------------------------------------------------------------------


def test_app_typed_unresolvable_binding_is_ill_typed():
    """A subst eid that resolves to no member is honestly ill-typed.

    ``eg._any_term_cached`` returns ``None`` only for a pathological
    class (a recursion-depth overflow on a cyclic cone — see the
    docstring); at the contract level an unresolved binding must
    count as ill-typed, never silently as typed.
    """

    class _NoRep:
        def _any_term_cached(self, eid):
            return None

    proposal = pl.Proposal("t:x", "A", _p("neg", "A"), "t")
    assert not pl._app_typed(_NoRep(), proposal, {"subst": {"A": 0}})


def test_typed_probe_on_nonfiring_rule(sink_cost):
    """A rule that never matches audits to a clean zero."""
    _, cost = sink_cost
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    case = _case("c", _p("add", x, y), x, y)
    proposal = pl.Proposal(
        "t:dead", _p("tanh", "A"), _p("exp", "A"), "t"
    )
    audit = pl._typed_probe(case, proposal, cost)
    assert audit.fires == 0 and audit.typed == 0 and audit.ill == 0
    assert audit.pick_ill is False


def test_fire_ill_typed_pick_without_cost_drop(sink_cost):
    """An ill-typed *tie* pick: extraction prefers it, it saves nothing.

    ``select(u, dim=0, index=-99)`` shape-checks (the axis index is not
    range-validated statically) but fails at evaluation — an
    ill-typed member at exactly the input's cost.  The deterministic
    enode ordering (``repr`` of attrs: ``-99`` sorts before ``1``)
    seats it as the extraction winner, so ``pick_ill`` is set while
    ``paid`` stays False — the suppressed case.
    """
    sink, cost = sink_cost
    u = _v("u", 4, 4)
    case = _case("sel", _p("select", u, dim=0, index=1), u)
    proposal = pl.Proposal(
        "t:sel99",
        _p("select", "A", dim="D", index="I"),
        _p("select", "A", dim="D", index=-99),
        "t",
    )
    ev = pl.Evidence(proposal=proposal)
    pl._fire(proposal, [case], sink, cost, ev)
    assert ev.fires >= 1
    assert ev.ill_typed_cases == ("sel",)
    # Ill-typed pick with no paid drop: counted, never credited.
    assert ev.paid_ill_typed == 0 and ev.paid == 0
    assert ev.fires_ill_typed >= 1


def test_fire_ill_typed_member_keeps_input(sink_cost):
    """A fire that mints only ill-typed members leaves the input picked.

    ``add(A, B) -> add(A, reshape(B, (3,3)))`` mints a member whose
    reshape is shape-invalid — the audit counts it ill, extraction
    keeps the input (``changed`` stays False), and the case
    contributes fires but no cost change.
    """
    sink, cost = sink_cost
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    case = _case("c", _p("add", x, y), x, y)
    proposal = pl.Proposal(
        "t:badr",
        _p("add", "A", "B"),
        _p("add", "A", _p("reshape", "B", shape=(3, 3))),
        "t",
    )
    ev = pl.Evidence(proposal=proposal)
    pl._fire(proposal, [case], sink, cost, ev)
    assert ev.fires >= 1 and ev.fires_ill_typed == ev.fires
    assert ev.changed == 0 and ev.paid == 0 and ev.verify_fail == 0


def test_reach_counts_unreplayable_certificate(monkeypatch, sink_cost):
    """A certificate that cannot replay is counted, not hidden."""
    import catopt_discovery.impact as im

    _, cost = sink_cost
    x, y, z = _v("x", 4, 4), _v("y", 4, 4), _v("z", 4, 4)
    case = _case(
        "factor",
        _p("add", _p("mul", x, y), _p("mul", x, z)),
        x,
        y,
        z,
    )
    proposal = pl.Proposal(
        "t:fac",
        _p("add", _p("mul", "A", "B"), _p("mul", "A", "C")),
        _p("mul", "A", _p("add", "B", "C")),
        "t",
    )

    def _boom(*a, **k):
        raise KeyError("no replay")

    monkeypatch.setattr(im, "verify_certificate", _boom)
    ev = pl.Evidence(proposal=proposal)
    pl._reach(proposal, [case], [], cost, ev)
    assert ev.cert_fail == 1


def test_measure_view_oracle_unproven_keeps_num_true(
    sink_cost, lib_keys
):
    """An ``unproven`` verdict leaves ``num_true`` untouched.

    The proposal carries a ``select`` (which puts it in the view
    oracle's scope) *and* a ``softmax`` attr metavar — an op outside
    the oracle's attr-domain table, so every instantiation is
    skipped honestly and no evaluable instance exists.
    """
    sink, cost = sink_cost
    lhs = _p(
        "mul",
        _p("softmax", "A", dim="D"),
        _p("select", "B", dim="SD", index="SI"),
    )
    proposal = pl.Proposal("t:nodom", lhs, lhs, "t")
    ev = pl.measure(proposal, [], [], [], lib_keys, {}, sink, cost)
    assert ev.view_verdict == "unproven"
    assert ev.num_true is None


# ---------------------------------------------------------------------------
#  coherence.py — confluence probe defensive skip + main without --json
# ---------------------------------------------------------------------------


def test_confluence_probe_skips_foreign_instance():
    """An instance that does not host the base rule is skipped.

    ``inst`` maps rule names to recorded ``(lhs, rhs)`` instances; if
    the entry for *first* does not actually match ``first.lhs`` (a
    name collision or a registry term whose fire was elsewhere),
    ``apply_rewrite_at`` returns ``None`` and the pairing falls
    through honestly rather than probing a bogus base.
    """
    a = Rewrite("ra", _p("div", "A", "B"), _p("mul", "A", "B"))
    b = Rewrite("rb", _p("add", "A", "B"), _p("add", "B", "A"))
    x, y = _v("x", 4), _v("y", 4)
    add_term = _p("add", x, y)
    inst = {
        "ra": (add_term, add_term),
        "rb": (add_term, _p("add", y, x)),
    }
    # b co-fires on a's recorded base, but a's own rule does not
    # re-match there — the probe declines.
    assert lc._confluence_probe(a, b, inst, []) is None


def test_coherence_main_without_json(monkeypatch, capsys):
    """The report path runs without ``--json`` (no dump, rc 0)."""
    by_name = {r.name: r for r in ALL_RULES}
    universe = [
        by_name[n]
        for n in (
            "silu_expand",
            "silu_fold",
            "silu_mul_form",
            "comm_add",
            "select_mul",
        )
    ]
    monkeypatch.setattr(lc, "ALL_RULES", universe)
    rc = lc.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "LAW COHERENCE CATALOGUE" in out


# ---------------------------------------------------------------------------
#  lemma_cert.py — enumerated-path strict-replay skip + print edge
# ---------------------------------------------------------------------------


def test_enumerated_cert_skips_unreplayable_paths(monkeypatch):
    """An enumerated path that fails strict replay is skipped.

    ``all_proofs`` enumerates candidate linear derivations; a path
    that does not verify standalone is ``continue``d — and when every
    candidate fails, ``None`` is returned honestly.
    """
    by_name = {r.name: r for r in ALL_RULES}
    rule = by_name["silu_fold"]
    lhs, rhs = lc._instance(rule)
    eg, r_lhs, r_rhs, _stats = llc._saturate(
        lhs, rhs, [by_name["silu_expand"]]
    )
    assert eg.find(r_lhs) == eg.find(r_rhs)
    # Premise: the rhs->lhs direction enumerates a real path.
    assert eg.all_proofs(
        rhs,
        lhs,
        root_eid=r_rhs,
        max_paths=llc._PROOF_PATHS,
        max_steps=llc._PROOF_STEPS,
        fuel=llc._PROOF_FUEL,
    )

    def _boom(*a, **k):
        raise ValueError("not standalone-replayable")

    monkeypatch.setattr(llc, "verify_certificate", _boom)
    assert llc._enumerated_cert(eg, rhs, lhs, r_rhs, by_name) is None


def test_lemma_cert_main_probe_gap_row(monkeypatch, capsys):
    """A probe row with no certificate and no note prints neither."""
    by_name = {r.name: r for r in ALL_RULES}
    universe = [
        by_name[n]
        for n in (
            "silu_expand",
            "silu_fold",
            "silu_mul_form",
            "comm_add",
            "select_mul",
        )
    ]
    monkeypatch.setattr(llc, "ALL_RULES", universe)
    rc = llc.main(["--probe", "select_mul"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "verdict=gap" in out


# ---------------------------------------------------------------------------
#  census.py / vocab.py — main without --json
# ---------------------------------------------------------------------------


def test_census_main_without_json(capsys):
    """The report path prints and returns 0 without ``--json``."""
    rc = cs.main(["--top", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "op-tuple census" in out
    assert "wrote" not in out


def test_vocab_main_without_json(capsys):
    """The report path prints and returns 0 without ``--json``."""
    rc = vc.main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "op vocabulary by property" in out
    assert "wrote" not in out
