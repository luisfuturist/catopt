"""Tests for ``catopt_discovery.arena`` — the construction arena.

Plan 0020: the player's action space is the ADR-0004 construction
vocabulary (fold / lift / compose / auto_cond / ingest) over the
evidence store, refereed by the admission gauntlet.  The arena
splits the corpus — the player acts on the working split while
``fires``/``paid`` are measured on a held-out probe it cannot
write into.  These tests play tiny episodes end to end, pin the
holdout isolation (a player can't see holdout terms), and pin the
dense partial credit (each gauntlet stage cleared counts).
"""

import json
import random

import pytest
import torch
from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op, TensorType, Var, op_repr_dag
from catopt_discovery import arena as ar
from catopt_discovery import evidence as ev
from catopt_discovery import object_synthesis as obs
from catopt_discovery.impact import TermCase


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _case(name: str, term: Op, *inputs: Var, source="test") -> TermCase:
    return TermCase(
        source=source,
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=tuple(
            torch.randn(tuple(x.typ.shape), dtype=torch.float64)
            for x in inputs
        ),
        param_vals={},
    )


def _step_term(a: Var, h: Var, x: Var) -> Op:
    """``add(matmul(a, h), x)`` — the affine-step site the lift targets."""
    return _p("add", _p("matmul", a, h), x)


def _aff_step_case(name: str, n: int = 4) -> TermCase:
    a, h, x = (
        _v(f"{name}_a", n, n),
        _v(f"{name}_h", n),
        _v(f"{name}_x", n),
    )
    return _case(name, _step_term(a, h, x), a, h, x)


def _silu_swap_case(name: str, n: int = 4) -> TermCase:
    x = _v(f"{name}_x", n, n)
    return _case(name, _p("mul", _p("sigmoid", x), x), x)


def _filler_case(name: str = "filler") -> TermCase:
    x = _v(f"{name}_x", 4, 4)
    y = _v(f"{name}_y", 4, 4)
    return _case(name, _p("mul", _p("silu", x), y), x, y)


def _meta() -> dict:
    """A fixed evidence scope (no git probe in tests)."""
    return {"code_rev": "test"}


def _aff_lift() -> ar.Action:
    """The affine-step carrier lift — a construction that clears."""
    return ar.Action.lift(
        ("add", ("matmul", "A", "h"), "x"),
        ("aff", "A", "x"),
        "apply",
        state="h",
        name="aff_step_lift",
    )


def _arena(**kw) -> ar.Arena:
    kw.setdefault("meta", _meta())
    return ar.Arena(**kw)


# ---------------------------------------------------------------------------
#  End to end — an episode runs, the gauntlet referees, reward lands
# ---------------------------------------------------------------------------


def test_episode_runs_end_to_end(tmp_path):
    """FixedRule playbook: ingest first, then the aff-step lift.

    The lift constructs the declared object, the store persists it,
    the gauntlet clears all eight gates — ``usable``, one fire and
    one paying case measured *on the holdout probe*.
    """
    arena = _arena(
        working=[_aff_step_case("w", 4), _filler_case()],
        holdout=[_aff_step_case("holdout", 6)],
        pending=[_aff_step_case("pending", 4)],
    )
    traj = ar.run_episode(arena, ar.FixedRule([_aff_lift()]), 6)
    assert len(traj.reports) == 2
    ingest, lift = traj.reports
    # step 0 — the corpus-direction move: pulls pending into working
    assert ingest.action.op == "ingest" and ingest.ingested == 1
    assert ingest.reward == 0.0 and ingest.stages == ()
    # step 1 — the construction: all eight gates cleared
    assert lift.applied and lift.alpha_key
    assert lift.stages_cleared == 8
    assert lift.usable and lift.holdout_fires >= 1
    assert lift.holdout_paid >= 1
    assert lift.reward == pytest.approx(
        8 + 2 + 0.2 * lift.holdout_fires + 2 * lift.holdout_paid
    )
    # trajectory accounting
    assert traj.usable == 1
    assert traj.holdout_fires == lift.holdout_fires
    assert traj.holdout_paid == lift.holdout_paid
    assert traj.total == pytest.approx(lift.reward)
    # the trace renders the allocation sequence as data
    trace = traj.trace()
    assert [t["op"] for t in trace] == ["ingest", "lift"]
    assert trace[1]["cleared"] == 8 and trace[1]["usable"]
    # hindsight: the unrewarded ingest is credited with the payoff
    # it enabled (the plan's "scored on what it later enables").
    assert traj.hindsight() == [lift.reward, lift.reward]


def test_state_is_read_only_data(tmp_path):
    """``observe`` returns a serializable snapshot over the store."""
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
        pending=[_filler_case("pend")],
    )
    state = arena.observe()
    blob = json.dumps(state.to_dict())  # serializable, no exception
    assert "w" in blob and "pend" in blob
    assert state.epoch == 0 and state.steps == 0
    assert set(state.actions) == {
        "auto_cond",
        "compose",
        "fold",
        "ingest",
        "lift",
        "relax_guard",
        "specialize",
    }
    # the store's object set and the library premises are visible
    state2, _rep = arena.step(_aff_lift())
    assert [o.name for o in state2.objects] == ["aff_step_lift"]
    assert "comm_mul" in state2.premises
    assert "aff_step_lift" in state2.premises
    # the object's measured verdict round-trips through the store
    assert [p.name for p in state2.proposals] == ["aff_step_lift"]
    assert state2.proposals[0].scope == "current"
    assert state2.proposals[0].fires >= 1
    json.dumps(state2.to_dict())


# ---------------------------------------------------------------------------
#  Holdout discipline — the player cannot see holdout terms
# ---------------------------------------------------------------------------


def test_holdout_isolation_pinned(tmp_path):
    """The holdout probe feeds the reward but never the observation."""
    hold = _aff_step_case("secret_holdout", 7)
    arena = _arena(
        working=[_aff_step_case("w", 4), _silu_swap_case("w2")],
        holdout=[hold],
        pending=[_filler_case("pend")],
    )
    term_repr = op_repr_dag(hold.term)
    for state in (arena.observe(), arena.step(_aff_lift())[0]):
        blob = json.dumps(state.to_dict()) + repr(state)
        # the holdout's case name and its term repr are invisible
        assert "secret_holdout" not in blob
        assert term_repr not in blob
        # the working split and the pending pool are all the state sees
        names = {c.name for c in state.corpus}
        assert names <= {"w", "w2", "pend"}
        assert set(state.pending) == {"pend"}
    # the reward still reads the holdout: fires/paid measured there
    rep = arena.reports[next(iter(arena.reports))]
    assert rep.evidence.fire_cases == ("secret_holdout",)


def test_split_corpus_holds_out_real_only():
    """The split bars purpose-built spellings from the holdout."""
    names = [f"real{i}" for i in range(8)]
    cases = [_aff_step_case(n, 4) for n in names] + [
        _aff_step_case("intake:Seeded", 4)
    ]
    pb = {"intake:Seeded"}
    working, holdout = ar.split_corpus(
        cases, seed=0, holdout_frac=0.25, purpose_built=pb
    )
    assert {c.name for c in holdout} <= set(names)
    assert "intake:Seeded" in {c.name for c in working}
    assert len(holdout) == 2  # 25% of the 8 real cases
    # deterministic: same seed → same split
    _w2, h2 = ar.split_corpus(
        cases, seed=0, holdout_frac=0.25, purpose_built=pb
    )
    assert [c.name for c in h2] == [c.name for c in holdout]
    # explicit names are still gated on real-only
    _w3, h3 = ar.split_corpus(
        cases,
        holdout_names={"real1", "intake:Seeded"},
        purpose_built=pb,
    )
    assert {c.name for c in h3} == {"real1"}
    # probe-eligibility: an ineligible real case stays in working
    _w4, h4 = ar.split_corpus(
        cases,
        holdout_names={"real1", "real2"},
        purpose_built=pb,
        probe_eligible={"real2"},
    )
    assert {c.name for c in h4} == {"real2"}
    # nothing eligible → the holdout is honestly empty
    _w5, h5 = ar.split_corpus(
        [_aff_step_case("pb_only", 4)], purpose_built={"pb_only"}
    )
    assert h5 == []


def test_purpose_built_names_reads_the_intake_ledger():
    """``purpose_built_names`` is the intake ledger spelled as cases."""
    pb = ar.purpose_built_names()
    assert "intake:BroadcastPadLeft" in pb  # a ledger entry
    assert "intake:nn.LayerNorm" not in pb  # a real workload
    assert all(n.startswith("intake:") for n in pb)


def test_ingest_rotates_the_evidence_scope(tmp_path):
    """An ingest grows working and re-scopes later verdict rows.

    The object's verdict was measured under the pre-ingest corpus —
    after the rotation it is served as a *prior-scope* proposal,
    never as current evidence (``CORPUS_DEPENDENT_COLS``).
    """
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
        pending=[_filler_case("pend")],
    )
    s1, _lift = arena.step(_aff_lift())
    assert s1.proposals[0].scope == "current"
    s2, ing = arena.step(ar.Action.ingest(("pend",)))
    assert ing.ingested == 1 and s2.epoch == 1
    assert {c.name for c in s2.corpus} == {"w", "pend"}
    # the prior verdict is still served — attributed to its scope
    assert [p.name for p in s2.proposals] == ["aff_step_lift"]
    assert s2.proposals[0].scope != "current"
    assert not arena.pending


# ---------------------------------------------------------------------------
#  Partial credit — every cleared stage is progress
# ---------------------------------------------------------------------------


def test_stage_clear_partial_credit(tmp_path):
    """A false construction still earns the stages it cleared.

    ``add(X,X) → abs(X)`` rebuilds fine, serializes fully and
    measures — then the truth gate refuses it: three stages of
    honest progress, reward 3, not usable, zero holdout payoff.
    """
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
    )
    state, rep = arena.step(
        ar.Action.fold(("add", "X", "X"), "abs", name="bad_fold")
    )
    assert rep.applied
    assert rep.stages_cleared == 3 and rep.stages_total == 4
    assert [s.name for s in rep.stages] == [
        "reconstruct",
        "full-data",
        "measure",
        "truth",
    ]
    assert rep.stages[-1].passed is False
    assert not rep.usable and rep.reward == 3.0
    assert rep.holdout_fires == 0 and rep.holdout_paid == 0
    # the refusal lands in the state view
    (o,) = state.objects
    assert o.usable is False and o.failed == "truth" and o.cleared == 3


def test_unknown_action_kind_is_an_error(tmp_path):
    """The registry is the action space — an unregistered op fails."""
    arena = _arena()
    with pytest.raises(KeyError, match="unregistered"):
        arena.step(ar.Action("nonsense_op", {}))


def test_the_registry_is_the_plug_in_seam(tmp_path):
    """A new constructor is one handler — no match statement.

    The sibling's ``relax_guard``/``specialize`` plug in exactly like
    this: ``(arena, action) -> _Outcome`` behind a name.
    """

    @ar.register_action("probe_noop")
    def _noop(arena: ar.Arena, action: ar.Action) -> ar._Outcome:
        return ar._Outcome(
            applied=False, note=f"noop saw {len(arena.working)} cases"
        )

    try:
        arena = _arena(working=[_aff_step_case("w", 4)])
        state, rep = arena.step(ar.Action("probe_noop", {}))
        assert not rep.applied and "noop saw 1" in rep.note
        assert "probe_noop" in state.actions
    finally:
        ar.ACTIONS.pop("probe_noop", None)


# ---------------------------------------------------------------------------
#  The construction vocabulary — each op over the store
# ---------------------------------------------------------------------------


def test_compose_over_shipped_rules(tmp_path):
    """``compose(comm_mul, silu_fold)`` mints the swapped SiLU fold.

    The composite names shipped premises — the record embeds the
    replayable cert — and the admitted object is usable on the
    holdout's ``mul(sigmoid(x), x)`` site.
    """
    arena = _arena(
        working=[_silu_swap_case("w", 4)],
        holdout=[_silu_swap_case("holdout", 5)],
    )
    state, rep = arena.step(
        ar.Action.compose(
            "comm_mul",
            "silu_fold",
            specialize={"a": ("sigmoid", "X"), "b": "X"},
            name="silu_swapped",
            env={"X": _v("X", 4, 4)},
        )
    )
    assert rep.applied and rep.usable
    assert rep.holdout_fires >= 1 and rep.holdout_paid >= 1
    o = state.objects[0]
    assert o.name == "silu_swapped" and o.serializable


def test_resolve_ref_vocabulary(tmp_path):
    """Object refs resolve: base names, stored keys, stored names."""
    arena = _arena()
    assert arena.resolve_ref("comm_mul") is not None
    assert arena.resolve_ref("no-such") is None
    _st, rep = arena.step(_aff_lift())
    assert arena.resolve_ref(rep.alpha_key) is not None
    assert arena.resolve_ref("aff_step_lift") is not None


def test_compose_declines_honestly(tmp_path):
    """A premise whose LHS matches nowhere declines — applied=False."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    _st, rep = arena.step(
        ar.Action.compose(
            "comm_mul",
            "assoc_add",
            specialize={"a": ("sigmoid", "X"), "b": "X"},
        )
    )
    assert not rep.applied and "matched nowhere" in rep.note
    assert rep.reward == 0.0 and rep.alpha_key == ""
    # unknown premise references decline before construction
    _s2, rep2 = arena.step(ar.Action.compose("comm_mul", "nope"))
    assert not rep2.applied and "unknown premise" in rep2.note


def test_auto_cond_refusal_is_honest(tmp_path):
    """No declarable cover → the refusal is the answer, not a stub."""
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
    )
    arena.step(ar.Action.fold(("add", "X", "X"), "abs", name="bad"))
    _st, rep = arena.step(ar.Action.auto_cond("bad", synth_limit=40))
    assert not rep.applied and rep.reward == 0.0
    assert "equal" in rep.note or "conjunction" in rep.note


def test_auto_cond_success_mints_a_guard(tmp_path):
    """An all-equal domain mints the vacuous guard — applied, re-faced.

    The aff lift's measured domain holds no bad site, so the
    auto-cond search returns ``cond=True`` (a vacuous cover, said
    honestly in the note); the guarded object re-stores under the
    same alpha key and clears the gauntlet again.
    """
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
    )
    _s1, lift = arena.step(_aff_lift())
    assert lift.usable
    _s2, rep = arena.step(
        ar.Action.auto_cond("aff_step_lift", synth_limit=60)
    )
    assert rep.applied and rep.usable
    assert rep.alpha_key == lift.alpha_key
    assert "vacuous cover" in rep.note
    assert rep.stages_cleared == 8 and rep.reward > 0


def test_fixed_rule_drains_to_none(tmp_path):
    """No playbook and nothing to guard → the rule stops."""
    rule = ar.FixedRule(
        [], ingest_first=False, guard_conditionals=False
    )
    assert rule(_arena().observe()) is None
    rule2 = ar.FixedRule([], ingest_first=False)
    assert rule2(_arena().observe()) is None


def test_arena_default_meta_uses_the_git_rev(tmp_path):
    """No ``meta`` override → the scope binds the real code rev."""
    arena = ar.Arena()
    assert arena._meta["code_rev"] != "test"


def test_resolve_ref_scans_stored_names(tmp_path):
    """Stored-name lookup iterates the lemma table; misses are None."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    _s1, _r1 = arena.step(_aff_lift())
    _s2, _r2 = arena.step(
        ar.Action.fold(("mul", "X", "X"), "square", name="sq")
    )
    # second row: the first stored name is scanned past, not matched
    assert arena.resolve_ref("sq") is not None
    assert arena.resolve_ref("aff_step_lift") is not None
    assert arena.resolve_ref("no_such_name") is None


def test_fixed_rule_guards_conditionals(tmp_path):
    """The fixed rule's last move: auto_cond every conditional object.

    With the playbook drained, ``guard_conditionals`` makes the rule
    attempt ``auto_cond`` on each object the truth gate refused —
    the "guard the conditionals" move, once each, honestly refused
    here (``add(X,X)=abs(X)`` has no declarable separating guard).
    """
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
    )
    moves = [
        ar.Action.fold(("add", "X", "X"), "abs", name="bad"),
        _aff_lift(),
    ]
    traj = ar.run_episode(arena, ar.FixedRule(moves), 6)
    ops = [r.action.op for r in traj.reports]
    # the fixed rule replays the playbook then auto_conds the refusal
    assert ops == ["fold", "lift", "auto_cond"]
    assert traj.reports[-1].action.params["ref"]
    assert not traj.reports[-1].applied  # honest refusal
    # and then the player drains — the episode stopped at None
    assert len(traj.reports) == 3


# ---------------------------------------------------------------------------
#  The legal-move enumeration
# ---------------------------------------------------------------------------


def _spec_occurs(hay: tuple, needle: tuple) -> bool:
    """Whether *needle* spec matches a subterm of *hay* spec."""
    from catopt_core.meta import match_pattern

    pat = obs.term_from_spec(needle)
    return any(
        match_pattern(pat, obs.term_from_spec(s), {}) is not None
        for s in ar._sub_specs(hay)
    )


def test_legal_actions_are_grounded_and_constructible(tmp_path):
    """Every enumerated move cites material the state attests."""
    arena = _arena(
        working=[_aff_step_case("w", 4), _silu_swap_case("s", 4)],
        holdout=[_aff_step_case("holdout", 6)],
        pending=[_filler_case("pend"), _filler_case("pend2")],
        base_rules=tuple(),  # isolate: no library premises
    )
    state = arena.observe()
    acts = ar.legal_actions(state)
    assert acts
    keys = [ar._action_key(a) for a in acts]
    assert len(keys) == len(set(keys))  # deduped
    assert acts == ar.legal_actions(state)  # pure function of state
    # deterministic order: family order, sorted inside each family
    order = [a.op for a in acts]
    assert order == sorted(
        order, key=("ingest", "auto_cond", "fold", "lift", "compose").index
    )
    specs = {c.spec for c in state.corpus}
    for a in acts:
        assert a.op in ar.ACTIONS
        p = a.params
        if a.op == "ingest":
            assert set(p["names"]) <= set(state.pending)
        elif a.op == "fold":
            assert ar._spec_size(p["spelled"]) >= 2
            assert any(_spec_occurs(s, p["spelled"]) for s in specs)
            # spelled's own ops are never re-cited as the kernel
            inner = ar._spec_ops(p["spelled"])
            k = p["kernel"]
            assert (k if isinstance(k, str) else k[0]) not in inner
        elif a.op == "lift":
            assert ar._spec_size(p["step"]) >= 2
            assert any(_spec_occurs(s, p["step"]) for s in specs)
            apply = p["apply_op"]
            assert any(
                a_ == apply for _c, a_, _h in state.carriers
            )
        elif a.op == "compose":
            assert p["first"] in state.premises
            assert all(r in state.premises for r in p["rest"])
        else:
            pytest.fail(f"unexpected op {a.op}")
    # the aff-step lift the tests prove is enumerated verbatim
    aff = ar.Action.lift(
        ("add", ("matmul", "X1", "X2"), "X3"),
        ("aff", "X1", "X3"),
        "apply",
        state="X2",
    )
    assert aff in acts
    # ingest: singletons plus the whole pool
    assert ar.Action.ingest(("pend",)) in acts
    assert ar.Action.ingest(("pend", "pend2")) in acts


def test_legal_actions_compose_pairs_and_specialize(tmp_path):
    """Compose enumeration: bare matches plus metavar specializations.

    ``compose(comm_mul, silu_fold)`` is the canonical specialized
    composite — the enumeration must reach a specialize map under
    which the pair actually constructs.
    """
    arena = _arena(working=[_silu_swap_case("w", 4)])
    state = arena.observe()
    acts = ar.legal_actions(state)
    pairs = [
        a for a in acts
        if a.op == "compose"
        and a.params["first"] == "comm_mul"
        and a.params["rest"] == ("silu_fold",)
    ]
    assert pairs
    # at least one emitted specialize map actually constructs
    ok = False
    for a in pairs:
        obj = obs.compose_objects(
            "t",
            arena.resolve_ref("comm_mul"),
            arena.resolve_ref("silu_fold"),
            specialize=dict(a.params["specialize"]),
        )
        ok = ok or obj is not None
    assert ok


def test_legal_actions_auto_cond_targets_unguarded(tmp_path):
    """Unguarded stored objects + conditional verdicts are targets."""
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
    )
    _s1, rep = arena.step(
        ar.Action.fold(("add", "X", "X"), "abs", name="bad")
    )
    assert not rep.usable
    state = arena.observe()
    acts = ar.legal_actions(state)
    assert ar.Action.auto_cond(rep.alpha_key) in acts
    # a stored object carrying a guard is not a target
    _s2, rep2 = arena.step(
        ar.Action.fold(
            ("div", "X", ("add", ("abs", "X"), 1)),
            "softsign",
            cond=("rank", "X", ">=", 1),
            name="guarded",
        )
    )
    state2 = arena.observe()
    acts2 = ar.legal_actions(state2)
    keys = {
        a.params["ref"] for a in acts2 if a.op == "auto_cond"
    }
    assert rep2.alpha_key not in keys
    assert rep.alpha_key in keys


def test_legal_actions_replay_is_idempotent(tmp_path):
    """Re-declaring a stored object stays legal — same alpha key."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    state = arena.observe()
    acts = ar.legal_actions(state)
    fold = next(
        a
        for a in acts
        if a.op == "fold"
        and a.params["spelled"]
        == ("add", ("matmul", "X1", "X2"), "X3")
        and a.params["kernel"] == "abs"
        and a.params["arg"] == "X1"
    )
    _s1, r1 = arena.step(fold)
    acts2 = ar.legal_actions(arena.observe())
    # still legal (re-declaring re-gauntlets the same object)…
    assert fold in acts2
    _s2, r2 = arena.step(fold)
    # …and idempotent: the store key is unchanged
    assert r2.alpha_key == r1.alpha_key


def test_legal_actions_empty_board(tmp_path):
    """No corpus, no pending, no premises → the empty move set."""
    arena = _arena(base_rules=tuple(), carrier_rules=())
    state = arena.observe()
    assert state.carriers == ()
    acts = ar.legal_actions(state)
    assert acts == ()
    assert ar.RandomPlayer(random.Random(0))(state) is None
    traj = ar.run_episode(arena, ar.RandomPlayer(random.Random(0)), 3)
    assert traj.reports == []


def test_carrier_basis_parametrized(tmp_path):
    """``carrier_rules`` binds the lift basis; triples pass through."""
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        carrier_rules=[("om_elem", "om_apply", False)],
    )
    state = arena.observe()
    assert state.carriers == (("om_elem", "om_apply", False),)
    acts = ar.legal_actions(state)
    lifts = [a for a in acts if a.op == "lift"]
    assert lifts and all(
        a.params["apply_op"] == "om_apply"
        and a.params["state"] is None
        for a in lifts
    )
    assert ar.Action.lift(
        ("add", ("matmul", "X1", "X2"), "X3"),
        ("om_elem", "X1", "X2", "X3"),
        "om_apply",
    ) in acts
    # a lift-shaped rule mines its (carrier, apply, state) template
    from catopt_core.laws.scan import AFF_LIFT

    arena2 = _arena(carrier_rules=[AFF_LIFT])
    assert arena2.observe().carriers == (("aff", "apply", True),)


# ---------------------------------------------------------------------------
#  The baselines — random over the legal set, greedy on stages
# ---------------------------------------------------------------------------


def test_random_player_episode_is_deterministic(tmp_path):
    """Same seed → same episode over the live legal set."""

    def _mk():
        arena = _arena(
            working=[_aff_step_case("w", 4)],
            holdout=[_aff_step_case("holdout", 6)],
            pending=[_filler_case("pend")],
            base_rules=tuple(),
        )
        return ar.run_episode(
            arena, ar.RandomPlayer(random.Random(7)), 4
        )

    t1, t2 = _mk(), _mk()
    assert [r.action.op for r in t1.reports] == [
        r.action.op for r in t2.reports
    ]
    assert t1.rewards == t2.rewards
    assert t1.reports  # the episode ran


def test_random_player_plays_only_legal_moves(tmp_path):
    """Every move the random player takes was legal in that state."""
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        pending=[_filler_case("pend")],
        base_rules=tuple(),
    )
    rng = random.Random(0)
    seen: list[str] = []

    class _Checked(ar.RandomPlayer):
        def __call__(self, state):
            action = super().__call__(state)
            if action is not None:
                assert action in ar.legal_actions(state)
                seen.append(action.op)
            return action

    traj = ar.run_episode(arena, _Checked(rng), 5)
    assert seen and len(traj.reports) == len(seen)


def test_greedy_player_picks_the_largest_spec(tmp_path):
    """Greed prefers a construction over ingest, the biggest first."""
    two = _case(
        "two",
        _p("square", _p("add", _v("t_a", 4, 4), _v("t_b", 4, 4))),
        _v("t_a", 4, 4),
        _v("t_b", 4, 4),
    )
    arena = _arena(
        working=[_aff_step_case("w", 4), two],
        pending=[_filler_case("pend")],
        base_rules=tuple(),
        carrier_rules=(),
    )
    move = ar.GreedyPlayer()(arena.observe())
    assert move is not None and move.op in ("fold", "lift", "compose")
    # bigger spelled pattern (2-op square(add(..)) = size 2…3) —
    # the point is greed never takes the zero-scoring ingest
    assert move.op != "ingest"
    # deterministic: same observation, same pick
    move2 = ar.GreedyPlayer()(arena.observe())
    assert ar._action_key(move) == ar._action_key(move2)


def test_greedy_player_legal_hook(tmp_path):
    """The ``legal`` seam restricts what greed sees."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    move = ar.GreedyPlayer(legal=lambda _s: ())(arena.observe())
    assert move is None


# ---------------------------------------------------------------------------
#  The depth probe — players on a small board
# ---------------------------------------------------------------------------


def test_depth_probe_small_board(tmp_path):
    """``depth_probe`` plays all three arms on an injected board."""

    from catopt_core.laws import ALL_RULES

    def factory(_s: int) -> ar.Arena:
        return _arena(
            working=[_aff_step_case("w", 4)],
            holdout=[_aff_step_case("holdout", 6)],
            pending=[_filler_case("pend")],
            base_rules=[
                r
                for r in ALL_RULES
                if r.name in ("comm_mul", "silu_fold")
            ],
        )

    table = ar.depth_probe(
        episodes=1, budget=5, seed=0, arena_factory=factory
    )
    assert set(table) == {"fixed", "random", "greedy"}
    for rows in table.values():
        (r,) = rows
        assert r["episode"] == 0 and r["steps"] <= 5
        assert {"usable", "holdout_fires", "holdout_paid"} <= set(r)
        # the live move set is free to grow (ingest adds spellings)
        assert r["legal_start"] >= 0 and r["legal_end"] >= 0
    # the fixed playbook replays ingest + its three constructions,
    # then may spend steps guarding whatever refused at truth
    fixed = table["fixed"][0]
    assert fixed["steps"] >= 4
    rendered = ar._fmt_probe(table)
    assert "fixed" in rendered and "greedy" in rendered


def test_main_runs_the_probe_and_writes_json(tmp_path, monkeypatch, capsys):
    """``main`` renders the probe table; ``--json`` dumps it."""
    monkeypatch.setattr(
        ar, "depth_probe", lambda **_kw: {"p": [ar._probe_row(
            ar.Trajectory([]), 0, 5, 3
        )]}
    )
    out = tmp_path / "probe.json"
    assert ar.main(["--json", str(out)]) == 0
    blob = json.loads(out.read_text())
    assert blob["p"][0]["usable"] == 0 and blob["p"][0]["legal_start"] == 5
    assert "player" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  Refusals, refs, scope and bookkeeping
# ---------------------------------------------------------------------------


def test_malformed_and_unknown_specs_decline(tmp_path):
    """A malformed spec is an honest ``applied=False``, never a crash."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    _s, r1 = arena.step(ar.Action.fold(("add", "X", ()), "abs"))
    assert not r1.applied and "malformed" in r1.note
    _s, r2 = arena.step(
        ar.Action.lift(("add", "X", ()), ("aff", "A", "x"), "apply")
    )
    assert not r2.applied and "malformed" in r2.note
    _s, r3 = arena.step(
        ar.Action.compose("comm_mul", "assoc_add", specialize={"a": ()})
    )
    assert not r3.applied and "raised" in r3.note
    _s, r4 = arena.step(ar.Action.auto_cond("never_stored"))
    assert not r4.applied and "unknown object" in r4.note


def test_ingest_of_missing_names_is_an_honest_noop(tmp_path):
    """Ingest moves what exists; the rest is named in the note."""
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        pending=[_filler_case("pend")],
    )
    _s, rep = arena.step(ar.Action.ingest(("pend", "ghost")))
    assert rep.applied and rep.ingested == 1
    assert "unknown pending names" in rep.note and "ghost" in rep.note
    _s, rep2 = arena.step(ar.Action.ingest(("ghost2",)))
    assert not rep2.applied and rep2.ingested == 0


def test_arena_explicit_backend_store_and_pending_dict(tmp_path):
    """The board takes an explicit store, rules, sink and dict pool."""
    from catopt_discovery.impact import _cost_fn
    from catopt_discovery.shape_proposal import _sink

    sink = _sink()
    conn = ev.connect(str(tmp_path / "s.db"))
    arena = ar.Arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("h", 6)],
        pending={"pd": _filler_case("pd")},
        conn=conn,
        base_rules=tuple(),
        sink=sink,
        cost_fn=_cost_fn(sink),
        meta=_meta(),
    )
    assert set(arena.pending) == {"pd"}
    assert arena.holdout[0].name == "h"
    state = arena.observe()
    assert state.pending == ("pd",)
    assert {c.name for c in state.corpus} == {"w"}
    conn.close()


def test_resolve_ref_passes_live_objects(tmp_path):
    """A ``Rewrite``/``ConstructedObject`` ref is itself; junk is None."""
    arena = _arena()
    rule = Rewrite(name="t", lhs="X", rhs=("neg", "X"))
    assert arena.resolve_ref(rule) is rule
    obj = obs.ConstructedObject(
        rule=rule, kind="abstraction", construction=("t",)
    )
    assert arena.resolve_ref(obj) is rule
    assert arena.resolve_ref(42) is None


def test_stored_but_ungauntleted_object_view(tmp_path):
    """A record stored outside a step shows ``usable=None`` honestly."""
    arena = _arena()
    key = obs.store_constructed(
        arena.conn,
        obs.fold_object("direct", ("mul", "X", "X"), "square"),
    )
    (o,) = arena.observe().objects
    assert o.alpha_key == key and o.usable is None and o.cleared == 0


def test_key_outcome_regauntlets_and_arms_auto_cond(tmp_path):
    """``_Outcome.key`` re-gauntlets a stored object without re-storing.

    A record that dropped its ``check`` hook fails ``full-data`` —
    and is exactly the object the fixed rule's conditional guard
    arms ``auto_cond`` on.
    """
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("h", 6)],
    )
    rule = Rewrite(
        name="flagged", lhs="X", rhs="X", check=lambda b: True
    )
    obj = obs.ConstructedObject(
        rule=rule, kind="abstraction", construction=("test",)
    )
    key = obs.store_constructed(arena.conn, obj)

    @ar.register_action("re_gauntlet")
    def _again(arena: ar.Arena, action: ar.Action) -> ar._Outcome:
        return ar._Outcome(key=action.params["key"])

    try:
        state, rep = arena.step(ar.Action("re_gauntlet", {"key": key}))
    finally:
        ar.ACTIONS.pop("re_gauntlet", None)
    assert rep.applied and rep.stages_cleared == 1
    assert rep.stages[-1].name == "full-data" and not rep.usable
    (o,) = state.objects
    assert o.failed == "full-data" and o.missing_hooks == ("check",)
    # the fixed rule's conditional guard now arms auto_cond on it
    nxt = ar.FixedRule([], ingest_first=False)(arena.observe())
    assert nxt is not None
    assert nxt.op == "auto_cond" and nxt.params["ref"] == key


def test_guarded_object_reports_region_counts(tmp_path):
    """A ``cond``-carrying construction gets the region sweeps.

    The guarded softsign fold is a library spelling — novelty
    refuses it — but the sweeps ran: ``guard_synth``/``guard_real``
    count the accepted sites, the ``guard_fires`` tally a player
    reads.
    """
    x = _v("w_x", 4, 8)
    site = _case(
        "softsign_site",
        _p("div", x, _p("add", _p("abs", x), Const(1))),
        x,
    )
    arena = _arena(working=[site], holdout=[_aff_step_case("h", 6)])
    state, rep = arena.step(
        ar.Action.fold(
            ("div", "X", ("add", ("abs", "X"), 1)),
            "softsign",
            cond=("rank", "X", ">=", 1),
            name="guarded_softsign",
        )
    )
    (o,) = state.objects
    assert rep.applied and o.failed == "novelty"
    assert o.guard_synth >= 1 and o.guard_real >= 1
    assert state.guard_fires[o.alpha_key] == (
        o.guard_synth + o.guard_real
    )


def test_unnamed_construction_gets_a_digest_name(tmp_path):
    """``params["name"]`` unset → a deterministic ``op:<hash>`` name."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    state, rep = arena.step(ar.Action.fold(("mul", "X", "X"), "square"))
    assert rep.applied
    (o,) = state.objects
    assert o.name.startswith("fold:")


def test_episode_stops_at_budget_and_on_none(tmp_path):
    """``None`` ends early; a never-stopping player hits the budget."""
    arena = _arena()
    traj = ar.run_episode(arena, lambda _s: ar.Action.ingest(()), 3)
    assert len(traj.reports) == 3
    assert all(not r.applied for r in traj.reports)
    empty = ar.run_episode(arena, lambda _s: None, 3)
    assert empty.reports == [] and empty.total == 0.0


# ---------------------------------------------------------------------------
#  The real board — make_arena splits the pipeline corpus honestly
# ---------------------------------------------------------------------------


def test_make_arena_builds_the_real_board(tmp_path):
    """``make_arena`` splits the pipeline's own probe pool.

    The holdout is real-only (the intake ledger bars its
    purpose-built spellings), every held case is probe-eligible
    (feed-bearing), and nothing held out is observable in the state.
    """
    arena = ar.make_arena(seed=0, meta={"code_rev": "test"})
    assert arena.working and arena.holdout
    pb = ar.purpose_built_names()
    held = {c.name for c in arena.holdout}
    assert held and not (held & pb)
    assert all(c.feed for c in arena.holdout)
    state = arena.observe()
    blob = json.dumps(state.to_dict())
    # name-level isolation is strict; a *repr* only counts as a
    # leak when no working case shares that program verbatim.
    visible = {cv.name for cv in state.corpus} | set(state.pending)
    work_reprs = {cv.term for cv in state.corpus}
    for c in arena.holdout:
        assert c.name not in visible
        assert f'"{c.name}"' not in blob
        term_repr = op_repr_dag(c.term)
        if term_repr not in work_reprs:
            assert term_repr not in blob
    # the bench spellings and intake remainder are playable material
    assert state.pending or any(
        c.source == "bench" for c in state.corpus
    )


def test_relax_guard_action_drops_a_clause(tmp_path):
    """``relax_guard`` weakens the stored guard and re-gauntlets."""
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
    )
    _s, rep = arena.step(
        ar.Action.fold(
            ("div", "X", ("add", ("abs", "X"), 1)),
            "softsign",
            cond=("and", ("rank", "X", ">=", 1), ("rank", "X", ">=", 0)),
            name="guarded",
        )
    )
    assert rep.applied
    (o,) = arena.observe().objects
    assert len(o.cond_clauses) == 2
    # drop the second clause — the object re-gauntlets under the key
    _s2, rep2 = arena.step(
        ar.Action.relax_guard(rep.alpha_key, 1)
    )
    assert rep2.applied and rep2.alpha_key == rep.alpha_key
    (o2,) = arena.observe().objects
    assert len(o2.cond_clauses) == 1


def test_relax_guard_declines_without_a_declarative_guard(tmp_path):
    """An unguarded object has no clause to drop — honest refusal."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    _s, rep = arena.step(_aff_lift())
    assert rep.applied
    _s2, rep2 = arena.step(
        ar.Action.relax_guard(rep.alpha_key, 0)
    )
    assert not rep2.applied
    assert "declined" in rep2.note


def test_specialize_action_pins_a_leaf_metavar(tmp_path):
    """``specialize`` narrows the object; a free name is an honest pin."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    _s, rep = arena.step(
        ar.Action.fold(("sub", "X", "Y"), "abs", name="abssub")
    )
    assert rep.applied
    (o,) = arena.observe().objects
    assert set(o.leaf_metavars) == {"X", "Y"}
    _s2, rep2 = arena.step(
        ar.Action.specialize(rep.alpha_key, {"Y": 0})
    )
    assert rep2.applied


def test_specialize_declines_a_name_the_lhs_binds_nowhere(tmp_path):
    """A binding that names nothing is the honest refusal."""
    arena = _arena(working=[_aff_step_case("w", 4)])
    _s, rep = arena.step(_aff_lift())
    _s2, rep2 = arena.step(
        ar.Action.specialize(rep.alpha_key, {"ZZZ": 1})
    )
    assert not rep2.applied


def test_legal_actions_enumerate_relax_and_specialize(tmp_path):
    """A guarded, free-metavar object contributes both move kinds."""
    arena = _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
    )
    _s, rep = arena.step(
        ar.Action.fold(
            ("div", "X", ("add", ("abs", "X"), 1)),
            "softsign",
            cond=("and", ("rank", "X", ">=", 1), ("rank", "X", ">=", 0)),
            name="guarded",
        )
    )
    acts = ar.legal_actions(arena.observe())
    relax = [
        a
        for a in acts
        if a.op == "relax_guard" and a.params["ref"] == rep.alpha_key
    ]
    assert len(relax) == 2  # one per cond clause
    spec = [
        a
        for a in acts
        if a.op == "specialize" and a.params["ref"] == rep.alpha_key
    ]
    # one leaf metavar X x the scalar bank
    assert len(spec) == len(ar.lawdata.SPECIALIZE_SCALARS)
