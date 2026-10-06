"""Tests for the admission gauntlet (``catopt_discovery.evidence.run_gauntlet``).

A stored declared object is not usable on the strength of
reconstruction — it must clear the same adversarial gauntlet a
shipped law faced (ADR 0004, plan 0017 stage 3): reconstruct →
full-data → measure (numeric oracle + derivability + the view
oracle's raw verdict) → truth (the *guarded-region* sweep for a
``cond``-carrying object) → novelty → typed-pay → closure → cert
replay.  These tests store a real synthesized candidate —
``mul_unsqueeze_l_id``, the view-oracle's strongest conditional
(``mul(unsqueeze(u,d), v) -> mul(u, v)`` under ``ones-before ∧
bcast-eq``) — as ``kind="abstraction"`` and run the gauntlet on a
tiny injected corpus; each gate is exercised on both sides of its
verdict (a known-false candidate must fail, the conditional-true
object must pass under its cond).
"""

import json

import torch
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import evidence as ev
from catopt_discovery import object_synthesis as synth
from catopt_discovery import oracle as vo
from catopt_discovery import pipeline as pl
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink

_BY_NAME = {r.name: r for r in ALL_RULES}

#: The guard the oracle's conditional analysis named, restated as
#: ``cond`` data (see ``project/retros/cond-dsl-view-guards.md``):
#: every ``u`` dim before the inserted axis is a broadcast-1, and the
#: viewed/unviewed broadcast grids coincide.
_UNSQ_COND = (
    "and",
    ("ones-before", "U", "A_dim"),
    ("bcast-eq", ("unsq-out", "U", "A_dim"), "V", "U", "V"),
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


def _mul_unsqueeze_l_id() -> Rewrite:
    """The first synthesized inhabitant: the guarded unsqueeze strip.

    Post-promotion this pattern IS the shipped ``mul_unsq_pad_l`` —
    the store now reports it as a library duplicate
    (``project/retros/promoted-laws.md``)."""
    return Rewrite(
        name="mul_unsqueeze_l_id",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("mul", "U", "V"),
        law="mul(unsqueeze(u,d),v) = mul(u,v) when the inserted axis "
        "is a broadcast pad",
        cond=_UNSQ_COND,
    )


def _add_unsqueeze_l_id() -> Rewrite:
    """The still-unshipped sibling: the same pad strip under ``add``.

    ``mul``/``sub`` spellings promoted to ``mul_unsq_pad_*`` /
    ``sub_unsq_pad_l``; the ``add`` variant was not measured paying, so
    it stays a store object — exactly the admitted-but-unshipped
    posture these tests exercise.
    """
    return Rewrite(
        name="add_unsqueeze_l_id",
        lhs=_p("add", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("add", "U", "V"),
        law="add(unsqueeze(u,d),v) = add(u,v) when the inserted axis "
        "is a broadcast pad",
        cond=(
            "and",
            ("bcast-eq", ("unsq-out", "U", "A_dim"), "V", "U", "V"),
            ("bcast-into", "U", ("unsq-out", "U", "A_dim")),
        ),
    )


def _mul_transpose_l_id() -> Rewrite:
    """The auto-cond transpose strip — admitted, but not promoted: the
    generic ``transpose_noop`` law subsumes it (see
    ``project/retros/promoted-laws.md``)."""
    return Rewrite(
        name="mul_transpose_l_id",
        lhs=_p(
            "mul", _p("transpose", "U", dim0="A_d0", dim1="A_d1"), "V"
        ),
        rhs=_p("mul", "U", "V"),
        law="mul(transpose(u,d0,d1),v) = mul(u,v) when the swap is a "
        "semantic no-op",
        cond=("axes-noop", "U", "A_d0", "A_d1"),
    )


def _guarded_case() -> TermCase:
    """A real firing site in the guard's region: ``u=(8,8)``,
    ``v=(1,1,1)``, ``dim=0`` — the retro's verified real acceptance."""
    u, v = _v("u", 8, 8), _v("v", 1, 1, 1)
    return _case(
        "unsq_pad",
        _p("mul", _p("unsqueeze", u, dim=0), v),
        u,
        v,
    )


def _guarded_add_case() -> TermCase:
    """The pad site under ``add`` — the unshipped sibling's region."""
    u, v = _v("u", 8, 8), _v("v", 1, 1, 1)
    return _case(
        "add_unsq_pad",
        _p("add", _p("unsqueeze", u, dim=0), v),
        u,
        v,
    )


def _transpose_noop_case() -> TermCase:
    """A real firing site for the no-op transpose strip: ``u=(2,1,1)``
    transposed on ``(-2,-1)`` — both swapped extents are 1."""
    u, v = _v("u", 2, 1, 1), _v("v", 2, 1, 1)
    return _case(
        "transpose_noop",
        _p("mul", _p("transpose", u, dim0=-2, dim1=-1), v),
        u,
        v,
    )


def _unguarded_site_case() -> TermCase:
    """A site outside the guard's region — the strip is false here.

    ``unsq(u,1)`` is ``(8,1,8)``; broadcasting it with ``v=(8,8)``
    gives ``(8,8,8)``, while ``mul(u,v)`` gives ``(8,8)`` — and the
    guard's ``ones-before`` clause declines (``u[0]=8 != 1``).
    """
    u, v = _v("u", 8, 8), _v("v", 8, 8)
    return _case(
        "unsq_nopad",
        _p("mul", _p("unsqueeze", u, dim=1), v),
        u,
        v,
    )


def _corpus(*cases: TermCase) -> ev.GauntletCorpus:
    """The tiny injected corpus: the same pieces the real one has."""
    sink = _sink()
    return ev.GauntletCorpus(
        real_terms=tuple(c.term for c in cases),
        probe=tuple(cases),
        base_rules=tuple(ALL_RULES),
        census_op={},
        sink=sink,
        cost_fn=_cost_fn(sink),
    )


def _stages(rep: ev.Gauntlet) -> dict[str, ev.GauntletStage]:
    return {s.name: s for s in rep.stages}


# ---------------------------------------------------------------------------
#  The first inhabitant — store -> admit -> gauntlet -> usable
# ---------------------------------------------------------------------------


def test_synthesized_object_clears_the_gauntlet(tmp_path):
    """The happy path end-to-end — now run on ``mul_transpose_l_id``,
    the admitted sibling that stayed unshipped (its noop transpose is
    subsumed by the promoted ``transpose_noop`` law; the wrapped
    spelling itself is not in the library)."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(
        conn, _mul_transpose_l_id(), kind="abstraction"
    )
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_transpose_noop_case()))
    conn.close()
    assert rep.usable, rep.reason
    assert (
        rep.name == "mul_transpose_l_id" and rep.kind == "abstraction"
    )
    stages = _stages(rep)
    assert all(s.passed for s in rep.stages)
    # The guarded-region truth: every accepted binding is equal on
    # both domains — and the guard provably bites (it declines).
    assert rep.synth_region.equal > 0
    assert rep.synth_region.unequal == 0
    assert rep.synth_region.rhs_err == 0
    assert rep.synth_region.declined > 0
    assert rep.real_region.accepted == 1
    assert rep.real_region.equal == 1
    # The measured evidence underneath.
    assert rep.evidence.fires >= 1
    assert rep.evidence.fires_ill_typed == 0
    assert rep.evidence.paid == 1
    assert rep.evidence.cert_fail == 0
    assert stages["cert"].detail == "no derivation recorded"


def test_promoted_object_now_reports_library_duplicate(tmp_path):
    """The first inhabitant's post-promotion verdict: ``mul_unsqueeze_*
    l_id`` shipped as ``mul_unsq_pad_l`` (``promoted-laws.md``), so the
    store's novelty gate reports the declared object as a library
    duplicate — the intended terminal state of a promoted object."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(
        conn, _mul_unsqueeze_l_id(), kind="abstraction"
    )
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    assert rep.evidence.relation == "duplicate"
    assert rep.reason.startswith("novelty:")
    # the guarded-region measurement still ran clean — the refusal is
    # novelty alone, not a soundness regression.
    assert rep.synth_region.unequal == 0
    assert rep.synth_region.rhs_err == 0


def test_admitted_object_fires_only_in_guarded_region(tmp_path):
    """The admitted Rewrite fires identically to its declared
    semantics: merges on the guarded site, silent outside it."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(
        conn, _mul_unsqueeze_l_id(), kind="abstraction"
    )
    rule, record = ev.admit_object(conn, key)
    conn.close()
    assert record["serializable"] is True
    u, v = _v("u", 8, 8), _v("v", 1, 1, 1)
    src = _p("mul", _p("unsqueeze", u, dim=0), v)
    dst = _p("mul", u, v)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    assert eg.rule_fires.get(rule.name) == 1
    assert eg.find(root) == eg.find(eg.add_term(dst))
    # Outside the guard the rule must not fire.
    u2, v2 = _v("u2", 8, 8), _v("v2", 8, 8)
    eg2 = EGraph()
    root2 = eg2.add_term(_p("mul", _p("unsqueeze", u2, dim=1), v2))
    eg2.run([rule], root2, max_iterations=4, max_nodes=10_000)
    assert not eg2.rule_fires


def test_guarded_truth_sweep_counts(tmp_path):
    """The guarded-region sweep sees the cond's exact cut: every
    accepted synth binding is equal, the false region is declined.

    The exact equal count rides the enumeration's ordering — the
    operand-shape-led bank (plan 0017 attr-sweep) lands 23 accepted
    equals inside the 360 cap; the region's *shape* is the pin, not
    the count.
    """
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(conn, _mul_unsqueeze_l_id())
    rule, _rec = ev.admit_object(conn, key)
    conn.close()
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=360)
    )
    assert region.equal == 23
    assert region.unequal == 0 and region.rhs_err == 0
    assert region.guard_err == 0
    assert region.declined > 300
    assert region.witness


# ---------------------------------------------------------------------------
#  Each gate refuses — the honest negative side
# ---------------------------------------------------------------------------


def test_unguarded_conditional_object_fails_truth(tmp_path):
    """The same strip with NO cond is only conditionally true —
    the plain verdict is 'conditional' and the object is unusable.
    (The ``mul`` spelling shipped as ``mul_unsq_pad_l`` — its bare
    twin would now read *derivable* on the pad instance and refuse at
    novelty instead — so this pin runs the still-unshipped ``add``
    variant.)"""
    conn = ev.connect(str(tmp_path / "s.db"))
    bare = Rewrite(
        name="add_unsqueeze_l_id_bare",
        lhs=_p("add", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("add", "U", "V"),
    )
    key = ev.store_object(conn, bare, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_add_case()))
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("truth:")
    assert rep.evidence.view_verdict == "conditional"


def test_known_false_candidate_fails_truth(tmp_path):
    """``mul(u,v) -> add(u,v)`` — the numeric oracle rejects it."""
    conn = ev.connect(str(tmp_path / "s.db"))
    false_rule = Rewrite(
        name="mul_is_add",
        lhs=_p("mul", "U", "V"),
        rhs=_p("add", "U", "V"),
    )
    key = ev.store_object(conn, false_rule, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("truth:")
    assert rep.evidence.num_true is False


def test_no_firing_site_fails_typed_pay(tmp_path):
    """A true-under-guard object on a corpus with no firing site
    clears truth but has nothing to pay for — typed-pay refuses.
    (Runs the unshipped ``add`` strip — the ``mul`` twin is a library
    duplicate now.)"""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(
        conn, _add_unsqueeze_l_id(), kind="abstraction"
    )
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    other = _case("plain", _p("add", x, y), x, y)
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(other))
    conn.close()
    assert not rep.usable
    stages = _stages(rep)
    assert stages["truth"].passed
    assert rep.synth_region.equal > 0
    assert not stages["typed-pay"].passed
    assert "fires=0" in stages["typed-pay"].detail


def test_ill_typed_fires_fail_typed_pay(tmp_path, monkeypatch):
    """A candidate minting ill-typed members is refused even when
    the earlier stages pass."""
    conn = ev.connect(str(tmp_path / "s.db"))
    rule = Rewrite(
        name="honest_comm",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "V", "U"),
    )
    key = ev.store_object(conn, rule, kind="abstraction")
    prop = pl.Proposal(
        name=rule.name, lhs=rule.lhs, rhs=rule.rhs, family="t"
    )
    evd = pl.Evidence(
        proposal=prop,
        num_true=True,
        relation="new",
        matches=1,
        fires=2,
        fires_typed=1,
        fires_ill_typed=1,
        paid=1,
    )
    monkeypatch.setattr(pl, "measure", lambda *a, **k: evd)
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    stages = _stages(rep)
    assert stages["truth"].passed and stages["novelty"].passed
    assert not stages["typed-pay"].passed
    assert "ill=1" in stages["typed-pay"].detail


def test_no_pay_fails_typed_pay(tmp_path, monkeypatch):
    """A firing, well-typed object that never lowers cost is not
    usable — 'pays' is part of the gate."""
    conn = ev.connect(str(tmp_path / "s.db"))
    rule = Rewrite(
        name="honest_comm",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "V", "U"),
    )
    key = ev.store_object(conn, rule, kind="abstraction")
    prop = pl.Proposal(
        name=rule.name, lhs=rule.lhs, rhs=rule.rhs, family="t"
    )
    evd = pl.Evidence(
        proposal=prop,
        num_true=True,
        relation="new",
        matches=1,
        fires=1,
        fires_typed=1,
        paid=0,
    )
    monkeypatch.setattr(pl, "measure", lambda *a, **k: evd)
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    assert _stages(rep)["typed-pay"].passed is False


def test_duplicate_spelling_fails_novelty(tmp_path):
    """An object whose equality is already a library rule is not a
    new inhabitant — 'not new' is part of the honest contract."""
    conn = ev.connect(str(tmp_path / "s.db"))
    dup = Rewrite(
        name="comm_mul_again",
        lhs=_p("mul", "P", "Q"),
        rhs=_p("mul", "Q", "P"),
    )
    key = ev.store_object(conn, dup, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    stages = _stages(rep)
    assert stages["truth"].passed  # comm_mul is derivable
    assert not stages["novelty"].passed
    assert "duplicate" in stages["novelty"].detail


def test_missing_object_and_unknown_kind_fail(tmp_path):
    """Reconstruction refusals: no row, or a kind the codec rejects."""
    conn = ev.connect(str(tmp_path / "s.db"))
    rep = ev.run_gauntlet(conn, "no-such-key", corpus=_corpus())
    assert not rep.usable
    assert rep.stages[0].name == "reconstruct"
    # A record claiming an unknown kind fails at the same gate.
    key = ev.store_object(conn, _BY_NAME["id_add"])
    record = ev.stored_object(conn, key)
    record["kind"] = "widget"
    conn.execute(
        "UPDATE lemmas SET law_json = ? WHERE alpha_key = ?",
        (json.dumps(record, sort_keys=True), key),
    )
    conn.commit()
    rep = ev.run_gauntlet(conn, key, corpus=_corpus())
    conn.close()
    assert not rep.usable
    assert "unknown object kind" in rep.reason


def test_non_serializable_record_fails_full_data(tmp_path):
    """A record that dropped hooks admits a weaker rule than it
    declares — the full-data gate refuses to call it usable."""
    conn = ev.connect(str(tmp_path / "s.db"))
    # A procedural ``check`` is code the record cannot carry — the
    # stored object is honestly flagged ``serializable: false``.
    rule = Rewrite(
        name="proc_checked",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "V", "U"),
        check=lambda bound: True,
    )
    key = ev.store_object(conn, rule, kind="abstraction")
    record = ev.stored_object(conn, key)
    assert record["serializable"] is False
    assert record["missing_hooks"] == ["check"]
    rep = ev.run_gauntlet(conn, key, corpus=_corpus())
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("full-data:")


def test_stored_cert_replays_and_corruption_fails(
    tmp_path, monkeypatch
):
    """The cert gate: a carried derivation replays strict; a corrupt
    one refuses the whole object."""
    conn = ev.connect(str(tmp_path / "s.db"))
    # silu_fold records derivation=("silu_expand",) — materializes.
    key = ev.store_object(conn, _BY_NAME["silu_fold"], kind="law")
    prop = pl.Proposal(
        name="x",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "V", "U"),
        family="t",
    )
    good = pl.Evidence(
        proposal=prop,
        num_true=True,
        relation="new",
        matches=1,
        fires=1,
        fires_typed=1,
        paid=1,
    )
    monkeypatch.setattr(pl, "measure", lambda *a, **k: good)
    rep = ev.run_gauntlet(conn, key, corpus=_corpus())
    assert rep.usable, rep.reason
    assert "replays strict" in _stages(rep)["cert"].detail
    # Corrupt the stored derivation — strict replay must refuse.
    record = ev.stored_object(conn, key)
    record["cert"]["steps"][0]["rule"] = "no_such_rule"
    conn.execute(
        "UPDATE lemmas SET law_json = ? WHERE alpha_key = ?",
        (json.dumps(record, sort_keys=True), key),
    )
    conn.commit()
    rep = ev.run_gauntlet(conn, key, corpus=_corpus())
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("cert:")


# ---------------------------------------------------------------------------
#  The CLI path — --admit-object --gauntlet
# ---------------------------------------------------------------------------


def test_cli_gauntlet_usable_and_refusal(tmp_path, capsys, monkeypatch):
    """``--admit-object KEY --gauntlet`` reports usable only on a
    clean gauntlet; exit code follows the verdict."""
    db = str(tmp_path / "s.db")
    conn = ev.connect(db)
    key = ev.store_object(
        conn, _mul_transpose_l_id(), kind="abstraction"
    )
    conn.close()
    monkeypatch.setattr(
        ev,
        "default_gauntlet_corpus",
        lambda: _corpus(_transpose_noop_case()),
    )
    assert (
        ev.main(["--report", db, "--admit-object", key, "--gauntlet"])
        == 0
    )
    out = capsys.readouterr().out
    assert "admitted mul_transpose_l_id" in out
    assert "kind: abstraction" in out
    assert "[pass] truth" in out
    assert "usable: yes" in out
    # A known-false stored object refuses.
    conn = ev.connect(db)
    key2 = ev.store_object(
        conn,
        Rewrite("bad", _p("mul", "U", "V"), _p("add", "U", "V")),
        kind="abstraction",
    )
    conn.close()
    assert (
        ev.main(["--report", db, "--admit-object", key2, "--gauntlet"])
        == 1
    )
    out = capsys.readouterr().out
    assert "usable: no" in out
    # And a promoted object refuses at novelty — it is a library member.
    conn = ev.connect(db)
    key3 = ev.store_object(
        conn, _mul_unsqueeze_l_id(), kind="abstraction"
    )
    conn.close()
    assert (
        ev.main(["--report", db, "--admit-object", key3, "--gauntlet"])
        == 1
    )
    out = capsys.readouterr().out
    assert "usable: no" in out
    assert "novelty" in out


def test_cli_gauntlet_requires_an_admit(tmp_path, capsys):
    db = str(tmp_path / "s.db")
    ev.connect(db).close()
    assert ev.main(["--report", db, "--gauntlet"]) == 1
    assert "--admit" in capsys.readouterr().out


def test_cli_gauntlet_missing_key(tmp_path, capsys):
    """A gauntlet admit on an empty store refuses, not crashes."""
    db = str(tmp_path / "s.db")
    ev.connect(db).close()
    assert (
        ev.main(
            ["--report", db, "--admit-object", "nope", "--gauntlet"]
        )
        == 1
    )
    assert "cannot admit" in capsys.readouterr().out


def test_cli_plain_admit_unchanged(tmp_path, capsys):
    """Without --gauntlet the admit op is the reconstruction report
    it always was — the gauntlet is opt-in, not a surprise."""
    db = str(tmp_path / "s.db")
    conn = ev.connect(db)
    key = ev.store_object(
        conn, _mul_unsqueeze_l_id(), kind="abstraction"
    )
    conn.close()
    assert ev.main(["--report", db, "--admit-object", key]) == 0
    out = capsys.readouterr().out
    assert "admitted mul_unsqueeze_l_id" in out
    assert "usable" not in out


# ---------------------------------------------------------------------------
#  The guarded sweep's honest edges — each bucket exercised
# ---------------------------------------------------------------------------


def test_malformed_cond_fails_measure(tmp_path):
    """A cond the interpreter cannot evaluate is not a guard — a
    real match makes the rule's check raise, and measure surfaces it."""
    conn = ev.connect(str(tmp_path / "s.db"))
    bad = Rewrite(
        name="bad_cond",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("mul", "U", "V"),
        cond=("no-such-pred", "U"),
    )
    key = ev.store_object(conn, bad, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("measure:")


def test_guard_err_bucket(tmp_path):
    """On a corpus with no real match the same malformed guard meets
    the sweep instead — counted as guard-err, refused at truth."""
    conn = ev.connect(str(tmp_path / "s.db"))
    bad = Rewrite(
        name="bad_cond",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("mul", "U", "V"),
        cond=("no-such-pred", "U"),
    )
    key = ev.store_object(conn, bad, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus())
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("truth:")
    assert rep.synth_region.guard_err > 0


def test_accepts_everything_guard_fails_truth(tmp_path):
    """``cond=True`` accepts the false region too — the sweep sees
    the counterexamples the guard failed to exclude."""
    conn = ev.connect(str(tmp_path / "s.db"))
    weak = Rewrite(
        name="weak_guard",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("mul", "U", "V"),
        cond=True,
    )
    key = ev.store_object(conn, weak, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    assert rep.synth_region.unequal > 0
    assert rep.synth_region.counterexample


def test_uninstantiable_view_op_fails_truth(tmp_path):
    """A pattern over an op the oracle cannot attribute-instantiate
    yields no evaluable sites — vacuous, not verified."""
    conn = ev.connect(str(tmp_path / "s.db"))
    mystery = Rewrite(
        name="mystery_view",
        lhs=_p("mul", _p("frobnicate", "U", axis="A_x"), "V"),
        rhs=_p("mul", "U", "V"),
        cond=("leaf", "U"),
    )
    key = ev.store_object(conn, mystery, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("truth:")
    assert rep.synth_region.accepted == 0


def test_rhs_err_fails_truth(tmp_path):
    """An accepted binding whose minted RHS cannot denote (here: an
    out-of-range literal index) is the same firing-time abort the
    engine records — the sweep counts it and the gate refuses."""
    conn = ev.connect(str(tmp_path / "s.db"))
    bad_rhs = Rewrite(
        name="bad_rhs",
        lhs=_p("mul", "U", "V"),
        rhs=_p("select", "V", dim=0, index=9),
        cond=("leaf", "V"),
    )
    key = ev.store_object(conn, bad_rhs, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("truth:")
    assert rep.synth_region.rhs_err > 0


def test_guarded_nonview_object_sweeps_leaf_domain(tmp_path):
    """The sweep is not view-only: a cond on a plain pattern filters
    the leaf-binding domain (and a library twin fails novelty)."""
    conn = ev.connect(str(tmp_path / "s.db"))
    comm = Rewrite(
        name="comm_guarded",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "V", "U"),
        cond=("leaf", "U"),
    )
    key = ev.store_object(conn, comm, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    stages = _stages(rep)
    assert stages["truth"].passed
    assert rep.synth_region.equal > 0 and rep.synth_region.unequal == 0
    assert not stages["novelty"].passed


def test_site_outcome_derive_accept_and_veto():
    """The derive hook rides the sweep with firing semantics: an
    accepted spec mints the ``$attr`` binding, a veto declines."""
    rule = Rewrite(
        name="t_w",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_d"), "V"),
        rhs=_p("unsqueeze", _p("mul", "U", "V"), dim="B_d"),
        cond=("rank", "U", ">=", 1),
        dspec={"B_d": ("attr", "A_d")},
    )
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=120)
    )
    assert region.accepted > 0
    assert region.equal + region.unequal > 0
    # A dspec whose expr cannot resolve vetoes every binding — the
    # firing-time ``derive -> None`` abort, counted as declines.
    veto = Rewrite(
        name="t_veto",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_d"), "V"),
        rhs=_p("unsqueeze", _p("mul", "U", "V"), dim="B_d"),
        cond=("rank", "U", ">=", 1),
        dspec={"B_d": ("dim", "U", 9)},
    )
    region = ev._guarded_evals(
        veto, ev._synth_sites(veto.lhs, veto.rhs, limit=60)
    )
    assert region.accepted == 0 and region.declined > 0


def test_sweep_buckets_guardless_and_lhs_err():
    """``_guarded_evals`` on a check-free rule accepts everything;
    a guard-accepted site whose LHS cannot evaluate counts
    ``other_err`` — honest bookkeeping, not a pass."""
    bare = Rewrite(
        name="bare", lhs=_p("mul", "U", "V"), rhs=_p("mul", "V", "U")
    )
    region = ev._guarded_evals(
        bare, ev._synth_sites(bare.lhs, bare.rhs, limit=30)
    )
    assert region.equal > 0 and region.declined == 0
    # index=9 is out of range on every bank shape: the LHS errs while
    # the (trivial) guard accepts — counted, not blamed on the RHS.
    sel = Rewrite(
        name="sel_oob",
        lhs=_p("select", "U", dim=0, index=9),
        rhs="U",
        cond=("rank", "U", ">=", 1),
    )
    region = ev._guarded_evals(
        sel, ev._synth_sites(sel.lhs, sel.rhs, limit=40)
    )
    assert region.other_err > 0
    assert region.equal == 0 and region.unequal == 0


def test_derive_raise_folds_to_decline(tmp_path):
    """A malformed derive expr raises (a bug, not a decline) — the
    sweep folds the raise into a decline and the object refuses."""
    conn = ev.connect(str(tmp_path / "s.db"))
    bad = Rewrite(
        name="bad_derive",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "V", "U"),
        cond=("leaf", "U"),
        dspec={"X": ("no-such-expr",)},
    )
    key = ev.store_object(conn, bad, kind="abstraction")
    rep = ev.run_gauntlet(conn, key, corpus=_corpus())
    conn.close()
    assert not rep.usable
    assert rep.synth_region.accepted == 0
    assert rep.synth_region.declined > 0


def test_dangling_rhs_fails_measure(tmp_path):
    """An RHS metavar the LHS never binds aborts the pipeline's own
    match-time instantiation — measure refuses before the sweep."""
    conn = ev.connect(str(tmp_path / "s.db"))
    dangling = Rewrite(
        name="dangling",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "U", "W"),
        cond=("leaf", "V"),
    )
    key = ev.store_object(conn, dangling, kind="abstraction")
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_unguarded_site_case())
    )
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("measure:")


def test_site_outcome_rhs_err_direct():
    """The sweep's rhs-err bucket: a guard-accepted real-style site
    whose minted RHS cannot instantiate (``W`` unbound)."""
    rule = Rewrite(
        name="dangling",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "U", "W"),
        cond=("leaf", "V"),
    )
    u, v = _v("u", 2, 2), _v("v", 2, 2)
    region = ev._guarded_evals(
        rule, [({"U": u, "V": v}, _p("mul", u, v))]
    )
    assert region.accepted == 1 and region.rhs_err == 1


def test_attr_merge_conflict_and_literal_attr():
    """``_attr_merge`` merges one attr combo; a name pre-bound to a
    different value declines the combo, literal attrs pass through."""
    node = _p("unsqueeze", "U", dim="A_d")
    domains = [(node, [{"dim": 0}, {"dim": 1}])]
    merged = ev._attr_merge(domains, ({"dim": 0},), {"U": "x"})
    assert merged == {"U": "x", "$attr:A_d": 0}
    # A conflicting pre-bound "$attr:" value vetoes the combination —
    # the same-name-different-value case the oracle guards against.
    assert (
        ev._attr_merge(domains, ({"dim": 0},), {"$attr:A_d": 1}) is None
    )
    # Literal (non-metavar) attrs on the node are not bindings.
    lit = _p("slice", "U", dim="A_d", start=0)
    domains = [(lit, [{"dim": 0, "start": 0}])]
    assert ev._attr_merge(domains, ({"dim": 0, "start": 0},), {}) == {
        "$attr:A_d": 0
    }
    # An attr metavar the combo did not resolve is left unbound.
    node2 = _p("unsqueeze", "U", dim="A_d", pad="A_p")
    domains2 = [(node2, [{"dim": 0}])]
    assert ev._attr_merge(domains2, ({"dim": 0},), {}) == {
        "$attr:A_d": 0
    }


def test_synth_bases_conflict_combo_skipped():
    """A leaf metavar named ``$attr:…`` collides with the attr
    binding channel — the conflicting combo is skipped, not merged."""
    lhs = _p("mul", _p("unsqueeze", "$attr:A_d", dim="A_d"), "V")
    rhs = _p("mul", "$attr:A_d", "V")
    bases = list(ev._synth_bases(lhs, rhs))
    # every combo conflicts (the leaf binds "$attr:A_d" to a Var,
    # the attr domain to an int) — no base survives.
    assert bases == []


def test_lhs_out_shapes_declines_uninstantiable():
    """A view node whose operand cannot instantiate contributes no
    output shape — declined, not guessed."""
    node = _p("unsqueeze", "U", dim="A_d")
    assert ev._lhs_out_shapes([node], {}) == []
    u = _v("u", 2, 3)
    assert ev._lhs_out_shapes([node], {"U": u, "$attr:A_d": 0}) == [
        (1, 2, 3)
    ]
    # A scalar constant instantiates but is shapeless — same decline.
    assert ev._lhs_out_shapes([Const(1.0)], {}) == []


def test_inst_pair_none_on_dangling_metavar():
    """An unbound metavar aborts instantiation — the dedupe filter
    never sees the pair."""
    u, v = _v("u", 2, 2), _v("v", 2, 2)
    lhs = _p("mul", "U", "V")
    assert (
        ev._inst_pair(lhs, _p("mul", "U", "W"), {"U": u, "V": v})
        is None
    )
    assert ev._inst_pair(
        lhs, _p("mul", "V", "U"), {"U": u, "V": v}
    ) == (
        Op.make("mul", u, v),
        Op.make("mul", v, u),
    )


def test_synth_sites_respects_limit():
    """The enumeration honors the cap — the sweep stays bounded."""
    lhs = _p("mul", _p("unsqueeze", "U", dim="A_d"), "V")
    rhs = _p("mul", "U", "V")
    assert len(list(ev._synth_sites(lhs, rhs, limit=5))) == 5


# ---------------------------------------------------------------------------
#  The selective cap policy — the escalation safety net
#
#  The ordering fix (``oracle._diag_groups``) surfaces the headline
#  ``sdpa_fold_*`` folds at the default window, so the escalation is no
#  longer paid for them.  It remains a latent safety net for a guarded
#  rule whose window is still starved by a *different* gap (a shape or
#  kind gap no cap reaches).
# ---------------------------------------------------------------------------


def test_sdpa_fold_addmul_surfaces_at_the_default_cap():
    """The ordering fix: the shipped ``sdpa_fold_addmul`` guard's
    accepted corner now lies inside the default window.

    Its accepted corner is a multi-clause region — the ``axes-last2``
    transpose precondition, ``softmax`` over the last axis, a numeric
    scale — and the shape-major enumeration reached it only at index
    793 (``cap-policy.md``), past the 360 default.  The
    ``(viewed x attr)`` diagonal (``oracle._diag_groups``) surfaces it
    at index 327: the window now accepts and proves the region instead
    of declining every site (``enumeration-fairness.md``).
    """
    rule = _BY_NAME["sdpa_fold_addmul"]
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=360)
    )
    assert region.accepted > 0
    assert region.equal > 0
    assert region.unequal == 0 and region.rhs_err == 0
    assert region.declined > 0
    assert region.envs == 360


def test_sdpa_fold_addmul_region_appears_at_the_guarded_cap():
    """The same rule inside the escalation ceiling: the region is
    there — non-empty, all equal, no counterexample."""
    rule = _BY_NAME["sdpa_fold_addmul"]
    region = ev._guarded_evals(
        rule,
        ev._synth_sites(rule.lhs, rule.rhs, limit=vo._GUARDED_CAP),
    )
    assert region.accepted > 0
    assert region.equal > 0
    assert region.unequal == 0 and region.rhs_err == 0
    assert region.declined > 0


def test_guarded_truth_escalates_a_still_starved_region():
    """``_guarded_truth`` still applies the policy: a guarded rule
    whose default window accepts nothing is swept again at the
    ceiling.

    ``rms_norm_fold`` stays starved — its guard's ``tail-block``
    clause is a shape gap no cap reaches — so the escalation fires and
    pays both windows without a rescue.  The ordering fix surfaced the
    ``sdpa_fold_*`` folds at the default, so this is the remaining
    (honest) cost of the safety net.
    """
    rule = _BY_NAME["rms_norm_fold"]
    synth, _real = ev._guarded_truth(rule, _corpus(), 360)
    assert synth.accepted == 0
    # the cost is reported: both windows were paid.
    assert synth.envs > 360


def test_guarded_truth_keeps_the_default_when_not_starved():
    """A guarded rule whose window already accepts a site is not
    escalated — the common case pays only the default window."""
    rule = _BY_NAME["glu_fold"]
    synth, _real = ev._guarded_truth(rule, _corpus(), 360)
    assert synth.accepted > 0
    assert synth.envs <= 360


def test_guarded_truth_masked_fill_kind_gap_remains():
    """The value bank closes the *value* gap; a *kind* gap keeps the
    rule starved.

    ``sdpa_fold_masked_fill``'s guard requires ``const-cmp F < -1e30``.
    The per-op constant domain (``oracle._CONST_DOMAIN``) now mints
    ``Const(-inf)`` for a ``masked_fill`` leaf, so the escalated window
    *accepts* a site — but the mask leaf itself is a float ``Var`` and
    ``masked_fill`` needs a bool mask, so the accepted site is an
    ``lhs-err``: the region still exhibits no equal instance and the
    truth gate still refuses it as vacuous.  Recorded, not hidden: the
    value bank widens the *values*; the mask *kind* is a separate
    mechanism.
    """
    rule = _BY_NAME["sdpa_fold_masked_fill"]
    synth, _real = ev._guarded_truth(rule, _corpus(), 360)
    assert synth.equal == 0
    assert synth.accepted > 0  # the sentinel is mintable now
    assert synth.other_err > 0  # ... but the float mask cannot denote
    assert synth.envs > 360  # escalated


def test_truth_gate_surfaces_at_the_default_window():
    """The stage-4 wiring: the ordering fix means the shipped
    ``sdpa_fold_addmul`` clears the truth gate on the *default* window
    — the escalation is not paid (it would have been refused as
    vacuous on the old shape-major window)."""
    from types import SimpleNamespace

    rule = _BY_NAME["sdpa_fold_addmul"]
    rep = ev.Gauntlet(
        alpha_key="k",
        evidence=SimpleNamespace(
            num_true=None,
            derivable=False,
            view_verdict="conditional",
            view_note="n",
        ),
    )
    assert ev._truth_gate(rep, rule, _corpus(), None)
    assert rep.synth_region is not None
    assert rep.synth_region.accepted > 0
    assert rep.synth_region.envs == 360


def test_region_detail_reports_the_env_cost():
    """The gauntlet report carries the sweep cost per domain."""
    starved = ev.GuardedRegion(accepted=0, declined=360, envs=360)
    rescued = ev.GuardedRegion(
        accepted=41, declined=1959, equal=9, envs=2360
    )
    detail = ev._region_detail(rescued, starved)
    assert "synth: 9eq/0ne/0rerr (41 accepted, 1959 declined, 2360 envs)" in detail
    assert "real: 0eq/0ne/0rerr (0 accepted, 360 declined, 360 envs)" in detail


# ---------------------------------------------------------------------------
#  A measured counterexample outranks a derivation
# ---------------------------------------------------------------------------
#
#  The hole: the truth gate was ``derivable or (region checks)``, so a
#  rule the saturation *derived* passed without a clean region sweep.
#  The fix: the measured site blocks regardless of derivability;
#  ``derivable`` may only waive the *starvation* case (no equal site
#  measured).
#
#  The gate's logic is pinned on *purpose-built* unclean guards — the
#  pre-tightening guards the derivable-gate audit's fixes replaced (see
#  ``project/retros/shipped-guard-tightening.md``).  A deliberately
#  loose guard keeps the pin independent of the shipped library's
#  regions: the seven laws the audit found unclean are now clean, so
#  the gate must be shown to bite on a guard that still accepts a
#  counterexample.


#: The pre-tightening ``linear_row_scale`` guard (before the
#: ``bcast-into`` clause): accepts ``W=(1,)``, ``r=(1,)``, ``x=(1,)`` —
#: a binding where the two sides differ (the law mis-evaluates a rank-1
#: degenerate output scale).
_LOOSE_ROW_SCALE = Rewrite(
    name="loose_row_scale",
    lhs=_p("linear", _p("mul", "x", "r"), "W"),
    rhs=_p("mul", _p("linear", "x", "W"), "r"),
    cond=("or", ("scalar", "r"), ("dim-eq-const", "r", -1, 1)),
)


#: The pre-tightening ``select_mul`` guard (before the ``bcast``
#: clause): accepts ``u=(2,3)``, ``v=(2,3,1)``, ``D=0`` — a binding
#: whose ``mul(u,v)`` RHS cannot denote (rhs-err).
_LOOSE_SELECT_MUL = Rewrite(
    name="loose_select_mul",
    lhs=_p(
        "mul",
        _p("select", "u", dim="D", index="I"),
        _p("select", "v", dim="D", index="I"),
    ),
    rhs=_p("select", _p("mul", "u", "v"), dim="D", index="I"),
    cond=("dim-eq-attr", "u", "D", "v", "D"),
)


def _channel_then_row() -> synth.ConstructedObject:
    """``compose(linear_channel_scale_rev, linear_row_scale)``.

    Derivable (both premises are shipped), fully declarative (pure
    data).  Its guard is the premises' conjunction, so the audit's
    guard tightening propagated to it: the composite that once
    inherited ``linear_row_scale``'s rank-1 blind spot now sweeps
    clean — see
    :func:`test_tightened_premise_cures_the_composite`.
    """
    obj = synth.compose_objects(
        "channel_then_row_scale",
        _BY_NAME["linear_channel_scale_rev"],
        _BY_NAME["linear_row_scale"],
    )
    assert obj is not None
    return obj


def test_tightened_premise_cures_the_composite():
    """Tightening ``linear_row_scale``'s guard cured the composite that
    inherited its blind spot: ``channel_then_row``'s guarded region now
    sweeps clean (0 unequal / 0 rhs-err), where the derivable-gate audit
    measured 1 unequal."""
    rule = _channel_then_row().rule
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=400)
    )
    assert region.equal > 0
    assert region.unequal == 0 and region.rhs_err == 0


def _stub_evidence(*, derivable: bool, num_true=None) -> object:
    """An ``Evidence``-shaped stub for the gate's own inputs."""
    from types import SimpleNamespace

    return SimpleNamespace(
        num_true=num_true,
        derivable=derivable,
        view_verdict="conditional",
        view_note="n",
    )


def test_derivation_does_not_override_a_measured_counterexample():
    """A derivable rule whose guard-accepted region carries a measured
    ``unequal`` site is refused, and the override is visible in the
    gate detail.

    The unclean guard is purpose-built (the pre-tightening
    ``linear_row_scale`` guard), so the pin survives the shipped
    library's regions changing.
    """
    rep = ev.Gauntlet(
        alpha_key="k", evidence=_stub_evidence(derivable=True)
    )
    assert not ev._truth_gate(rep, _LOOSE_ROW_SCALE, _corpus(), None)
    # measured 3 at the 360-site window; the count rides the bank, the
    # *presence* of the counterexample is the pin.
    assert rep.synth_region.unequal >= 1
    assert "derivation overridden" in _stages(rep)["truth"].detail


def test_derivable_but_clean_rule_still_admits():
    """Do not over-tighten: a derivable rule whose accepted region is
    clean and exhibits an equal site still clears the truth gate."""
    rep = ev.Gauntlet(
        alpha_key="k",
        evidence=_stub_evidence(derivable=True),
    )
    assert ev._truth_gate(rep, _BY_NAME["glu_fold"], _corpus(), None)
    assert rep.synth_region.equal >= 1
    assert rep.synth_region.unequal == 0


def test_derivable_waives_only_the_starvation_requirement():
    """A derivable rule whose window is starved (no equal site) still
    clears truth — derivability waives the vacuity bar, not a
    counterexample.  The same rule *without* derivability is refused.

    The starved guard is a synthetic ``rank ≥ 99`` predicate the bank
    never satisfies, so the test pins the *gate's* logic rather than a
    particular rule's enumeration.
    """
    rule = Rewrite(
        name="starved_guard",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "V", "U"),
        cond=("rank", "U", ">=", 99),
    )
    starved = ev.Gauntlet(
        alpha_key="k", evidence=_stub_evidence(derivable=True)
    )
    assert ev._truth_gate(starved, rule, _corpus(), None)
    assert starved.synth_region.equal == 0
    assert starved.synth_region.accepted == 0
    # The starvation bar bites when there is no derivation to waive it.
    vacuous = ev.Gauntlet(
        alpha_key="k", evidence=_stub_evidence(derivable=False)
    )
    assert not ev._truth_gate(vacuous, rule, _corpus(), None)


def test_derivation_cannot_override_a_measured_rhs_err():
    """``rhs_err`` — an accepted binding whose minted RHS cannot
    denote — is a measured counterexample too: derivability does not
    waive it.  (Purpose-built unclean guard.)"""
    rep = ev.Gauntlet(
        alpha_key="k", evidence=_stub_evidence(derivable=True)
    )
    assert not ev._truth_gate(rep, _LOOSE_SELECT_MUL, _corpus(), None)
    assert rep.synth_region.rhs_err > 0


def test_derivation_cannot_override_an_unequal_site():
    """``unequal`` (the two sides differ on an accepted binding)
    blocks regardless of derivability.  (Purpose-built unclean
    guard.)"""
    rep = ev.Gauntlet(
        alpha_key="k", evidence=_stub_evidence(derivable=True)
    )
    assert not ev._truth_gate(rep, _LOOSE_ROW_SCALE, _corpus(), None)
    assert rep.synth_region.unequal > 0


def test_unguarded_derivable_false_still_refuses():
    """The unguarded branch reads the same rule off the numeric
    oracle: a measured ``num_true is False`` blocks even when the
    candidate is derivable; derivable waives only ``num_true is
    None``."""
    rep = ev.Gauntlet(
        alpha_key="k",
        evidence=_stub_evidence(derivable=True, num_true=False),
    )
    assert not ev._truth_gate(
        rep, _BY_NAME["comm_mul"], _corpus(), None
    )
    rep2 = ev.Gauntlet(
        alpha_key="k",
        evidence=_stub_evidence(derivable=True, num_true=None),
    )
    assert ev._truth_gate(rep2, _BY_NAME["comm_mul"], _corpus(), None)


def test_region_clean_ignores_equal_count():
    """``_region_clean`` is the counterexample test alone — an empty
    region is clean, an unequal one is not."""
    assert ev._region_clean(ev.GuardedRegion())
    assert ev._region_clean(ev.GuardedRegion(equal=5, accepted=5))
    assert not ev._region_clean(ev.GuardedRegion(unequal=1))
    assert not ev._region_clean(ev.GuardedRegion(rhs_err=1))
    assert not ev._region_clean(ev.GuardedRegion(guard_err=1))
    # ``_region_ok`` adds the starvation bar on top.
    assert ev._region_ok(ev.GuardedRegion(), need_equal=False)
    assert not ev._region_ok(ev.GuardedRegion(), need_equal=True)


# ---------------------------------------------------------------------------
#  Non-view attr metavars — the generic attr domain (attr-sweep retro)
# ---------------------------------------------------------------------------

#: The guarded ``sdpa`` fold's transpose-axes precondition — the
#: ``_COND_SCORE_T`` fragment of the shipped ``sdpa_fold_*`` guards.
_COND_LAST2 = (
    "and",
    ("concrete", "K"),
    ("attr-type", "TD1", "int"),
    ("attr-type", "TD2", "int"),
    ("axes-last2", "K", "TD1", "TD2"),
)


def _sdpa_fold_div_nomask_g() -> Rewrite:
    """The guarded twin: ``scale="SC"`` is an attr metavar on a
    *non-view* op — unreachable by the sweep before the generic
    attr domain."""
    return Rewrite(
        name="sdpa_fold_div_nomask_g",
        lhs=_p(
            "matmul",
            _p(
                "softmax",
                _p(
                    "div",
                    _p(
                        "matmul",
                        "Q",
                        _p(
                            "transpose",
                            "K",
                            dim0="TD1",
                            dim1="TD2",
                        ),
                    ),
                    "S",
                ),
                dim=-1,
            ),
            "V",
        ),
        rhs=_p("sdpa", "Q", "K", "V", scale="SC"),
        cond=("and", _COND_LAST2, ("const-num", "S")),
        dspec={"SC": ("recip", ("float", ("const", "S")))},
    )


def _attn_div_case() -> TermCase:
    """``matmul(softmax(q@kᵀ / 4.0, -1), v)`` — the real firing site."""
    q, k, v = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4)
    scores = _p(
        "div",
        _p("matmul", q, _p("transpose", k, dim0=-1, dim1=-2)),
        Const(4.0),
    )
    return _case(
        "attn_div",
        _p("matmul", _p("softmax", scores, dim=-1), v),
        q,
        k,
        v,
    )


def test_nonview_attr_metavar_sweep_is_nonempty():
    """A ``scale="SC"`` metavar on ``sdpa`` no longer empties the
    synthesized domain — the generic attr domain enumerates it."""
    rule = _sdpa_fold_div_nomask_g()
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=360)
    )
    # The guard accepts the evaluable corner and every accepted
    # binding is equal — the region verifies, it is not vacuous.
    assert region.accepted > 0
    assert region.equal > 0
    assert region.unequal == 0 and region.rhs_err == 0
    assert region.guard_err == 0
    assert region.declined > 0


def test_nonview_attr_metavar_object_clears_the_gauntlet(tmp_path):
    """``sdpa_fold_div_nomask_g`` end to end: previously refused at
    truth on an empty synth region — now ``usable`` on its guard."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(
        conn, _sdpa_fold_div_nomask_g(), kind="abstraction"
    )
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_attn_div_case())
    )
    conn.close()
    assert rep.usable, rep.reason
    assert all(s.passed for s in rep.stages)
    assert rep.synth_region.equal >= 1
    assert rep.synth_region.unequal == 0
    assert rep.real_region.equal == 1
    assert rep.evidence.paid >= 1


def test_shipped_nonview_attr_law_sweeps():
    """The shipped ``softmax_fold`` — attr metavars on ``sum`` and
    ``softmax`` — sweeps a nonempty guarded region now (the shipped
    rules were unenumerable before the extension, too)."""
    rule = _BY_NAME["softmax_fold"]
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=360)
    )
    assert region.accepted > 0 and region.equal > 0
    assert region.unequal == 0 and region.rhs_err == 0


def test_measure_exception_is_a_gate_failure(tmp_path, monkeypatch):
    """If the measurement itself fails the object is not usable —
    a measurement error is a failure to surface, not a pass."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = ev.store_object(conn, _mul_unsqueeze_l_id())
    monkeypatch.setattr(
        pl,
        "measure",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_guarded_case()))
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("measure:")


def test_default_gauntlet_corpus_builds():
    """The default corpus is the pipeline's real one — built lazily
    so the report path stays torch-free."""
    corpus = ev.default_gauntlet_corpus()
    assert len(corpus.real_terms) > 0
    assert len(corpus.probe) > 0
    assert len(corpus.base_rules) == len(ALL_RULES)
    assert corpus.census_op and corpus.cost_fn is not None
