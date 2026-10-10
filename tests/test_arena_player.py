"""Tests for ``catopt_discovery.arena_player`` — the learned player.

Plan 0021: a softmax linear policy over ``arena.legal_actions``,
trained by episode-return REINFORCE on the cheap board.  These
tests pin the player contract (legal-only moves, determinism,
played-move masking), the featurization schema (features emitted
are exactly the ``lawdata.ARENA_FEATURES`` names), and the
train/eval seed split honesty (no eval board was a training board).
"""


import torch
from catopt_core.ir import Op, TensorType, Var
from catopt_discovery import arena as ar
from catopt_discovery import arena_player as ap
from catopt_discovery import lawdata
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


def _aff_step_case(name: str, n: int = 4) -> TermCase:
    a, h, x = (
        _v(f"{name}_a", n, n),
        _v(f"{name}_h", n),
        _v(f"{name}_x", n),
    )
    return _case(name, _p("add", _p("matmul", a, h), x), a, h, x)


def _filler_case(name: str = "filler") -> TermCase:
    x = _v(f"{name}_x", 4, 4)
    y = _v(f"{name}_y", 4, 4)
    return _case(name, _p("mul", _p("silu", x), y), x, y)


def _meta() -> dict:
    return {"code_rev": "test"}


def _arena(**kw) -> ar.Arena:
    kw.setdefault("meta", _meta())
    return ar.Arena(**kw)


def _factory(**kw):
    """A fresh tiny board per seed — the ``arena_factory`` shape."""

    def mk(_seed: int) -> ar.Arena:
        return _arena(**kw)

    return mk


def _lift_board() -> ar.Arena:
    """A board where the paying aff-step lift is one legal move."""
    return _arena(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
        pending=[_filler_case("pend")],
        carrier_rules=[("aff", "apply", True)],
    )


# ---------------------------------------------------------------------------
#  The feature schema — data-named, subset of lawdata.ARENA_FEATURES
# ---------------------------------------------------------------------------


def test_features_emit_only_declared_names():
    """Every emitted feature name is a declared schema entry."""
    arena = _lift_board()
    state, _rep = arena.step(
        ar.Action.fold(("add", "X", "X"), "abs", name="bad")
    )
    acts = ar.legal_actions(state)
    assert acts
    seen: set = set()
    for a in acts:
        feats = ap.featurize(state, a)
        assert set(feats) <= set(lawdata.ARENA_FEATURES)
        seen |= set(feats)
    # each feature family is exercised on a mixed board
    assert any(n.startswith("st:") for n in seen)
    assert any(n.startswith("h:") for n in seen)
    assert any(n.startswith("a:k:") for n in seen)  # fold kernels
    assert any(n.startswith("a:p1:") for n in seen)  # compose
    assert "a:ingest_frac" in seen
    # the stored unguarded object arms the t:* target features
    assert "t:known" in seen and "t:f_truth" in seen
    assert "a:rescue" in seen


def test_feature_schema_is_internally_consistent():
    """The generated names match the tables that generate them."""
    names = set(lawdata.ARENA_FEATURES)
    assert len(names) == len(lawdata.ARENA_FEATURES)  # unique
    assert {f"op:{o}" for o in lawdata.ARENA_MOVE_ORDER} <= names
    for op in lawdata.ARENA_MOVE_ORDER:
        for stat in lawdata.ARENA_HISTORY_STATS:
            assert f"h:{op}:{stat}" in names
    b = lawdata.ARENA_HASH_BUCKETS
    assert {f"a:k:{i:02d}" for i in range(b["kernel"])} <= names
    assert {f"a:p1:{i:02d}" for i in range(b["premise"])} <= names
    assert {f"a:p2:{i:02d}" for i in range(b["premise"])} <= names
    # the default weight table only names declared features
    assert set(lawdata.ARENA_PLAYER_WEIGHTS) <= names


def test_hash_buckets_are_stable_and_bounded():
    """Same name → same bucket on any board (no process salt)."""
    for kind, width in lawdata.ARENA_HASH_BUCKETS.items():
        got = ap._bucket("softsign", kind)
        assert 0 <= got < width
        assert ap._bucket("softsign", kind) == got


def test_history_records_inferred_outcomes():
    """The player diffs observations into per-op outcome tallies.

    Playing the aff-step lift (a paying construction) then one
    declined move leaves ``h:lift:*`` rates at 1 and a positive
    reward trend — all inferred from ``ArenaState`` alone.
    """
    arena = _lift_board()
    player = ap.LearnedPlayer(seed=0)
    state = arena.observe()
    # play the enumerated aff lift verbatim
    lift = next(
        a
        for a in ar.legal_actions(state)
        if a.op == "lift" and a.params["state"] == "X2"
    )
    _s1, rep = arena.step(lift)
    assert rep.usable and rep.holdout_paid >= 1
    player._last = lift
    player._prev = state
    player._observe(arena.observe())
    feats = player._hist.feats("lift")
    assert feats["h:lift:usable"] == 1.0
    assert feats["h:lift:pay"] == 1.0
    assert player._hist.trend > 0
    # a declined move lands in the decline tally
    ghost = ar.Action.ingest(("ghost",))
    player._last = ghost
    player._prev = arena.observe()
    _s3, g = arena.step(ghost)
    assert not g.applied
    player._observe(arena.observe())
    assert player._hist.decline.get("ingest") == 1


# ---------------------------------------------------------------------------
#  The player contract — legal-only, deterministic, spends moves once
# ---------------------------------------------------------------------------


def test_learned_player_plays_only_legal_moves():
    """Every move the policy picks was legal in that state."""
    arena = _lift_board()
    player = ap.LearnedPlayer(seed=3)
    seen = []

    class _Checked:
        def __call__(self, state):
            move = player(state)
            if move is not None:
                assert move in ar.legal_actions(state)
                seen.append(move.op)
            return move

    traj = ar.run_episode(arena, _Checked(), 8)
    assert seen and len(traj.reports) == len(seen)


def test_learned_player_is_deterministic():
    """Same seed + same board → the same episode."""

    def _mk():
        arena = _lift_board()
        return ar.run_episode(arena, ap.LearnedPlayer(seed=7), 8)

    t1, t2 = _mk(), _mk()
    assert [r.action.op for r in t1.reports] == [
        r.action.op for r in t2.reports
    ]
    assert t1.rewards == t2.rewards


def test_learned_player_masks_played_moves():
    """A played move stays legal but is never re-picked."""
    arena = _lift_board()
    traj = ar.run_episode(arena, ap.LearnedPlayer(seed=0), 12)
    keys = [ar._action_key(r.action) for r in traj.reports]
    assert keys and len(keys) == len(set(keys))
    # and the played set is the masking mechanism: the same action is
    # still legal later yet absent from the reports
    state = arena.observe()
    legal = {ar._action_key(a) for a in ar.legal_actions(state)}
    assert any(k in legal for k in keys)  # replays stayed legal…
    assert len(set(keys)) == len(keys)  # …but never replayed


def test_learned_player_drains_to_none():
    """An exhausted move set ends the episode (``None``).

    Repeated calls on the same observation are not an episode
    boundary — the played set survives until the board steps.
    """

    def _legal(_s):
        return (ar.Action.ingest(("pend",)),)

    arena = _lift_board()
    player = ap.LearnedPlayer(seed=0, legal=_legal)
    assert player(arena.observe()) is not None
    assert player(arena.observe()) is None


def test_greedy_player_is_argmax():
    """``greedy=True`` picks the highest-scoring unplayed move."""
    arena = _lift_board()
    weights = {"op:ingest": -5.0}
    player = ap.LearnedPlayer(seed=0, weights=weights, greedy=True)
    move = player(arena.observe())
    assert move is not None and move.op != "ingest"


# ---------------------------------------------------------------------------
#  The training loop — REINFORCE on episode return
# ---------------------------------------------------------------------------


def test_finish_episode_updates_and_stays_finite():
    """A played episode's return moves the weight table.

    ``budget=1`` makes the episode a true bandit: the return is the
    sampled move's payoff alone, so REINFORCE credit lands on the
    paying lift rather than being smeared over the whole episode
    (the guide-real-run lesson).
    """
    player = ap.LearnedPlayer(seed=0, lr=0.5, legal=_two_move_legal)
    factory = _factory(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
        pending=[_filler_case("pend")],
        base_rules=tuple(),
        carrier_rules=[("aff", "apply", True)],
    )
    totals = ap.train_player(
        player, range(8), budget=1, arena_factory=factory
    )
    assert max(totals) > 10.0  # the paying lift was sampled sometimes
    wd = player.weights_dict()
    assert set(wd) == set(lawdata.ARENA_FEATURES)
    assert all(isinstance(v, float) for v in wd.values())
    assert wd["op:lift"] > wd["op:fold"]
    assert any(abs(v) > 0 for v in wd.values())


def test_frozen_player_never_updates():
    """``learn=False`` keeps the weight table verbatim."""
    player = ap.LearnedPlayer(seed=0, weights={"bias": 1.5})
    frozen = player.frozen(seed=0)
    arena = _lift_board()
    ar.run_episode(arena, frozen, 4)
    frozen.finish_episode(10.0)
    wd = frozen.weights_dict()
    assert wd["bias"] == 1.5
    assert sum(1 for v in wd.values() if v) == 1


def test_advantage_is_clipped():
    """A huge return cannot blow the weights past the clip."""
    player = ap.LearnedPlayer(seed=0, lr=1.0)
    arena = _lift_board()
    ar.run_episode(arena, player, 2)
    player.finish_episode(1e9)
    wd = player.weights_dict()
    assert all(abs(v) <= 1e3 for v in wd.values())


def test_train_player_returns_the_curve():
    """``train_player`` plays one episode per seed and returns totals."""
    player = ap.LearnedPlayer(seed=0)
    totals = ap.train_player(
        player,
        [101, 102],
        budget=4,
        arena_factory=_factory(
            working=[_aff_step_case("w", 4)],
            holdout=[_aff_step_case("holdout", 6)],
        ),
    )
    assert len(totals) == 2
    assert all(isinstance(t, float) for t in totals)


def test_episode_seeds_are_disjoint():
    """Eval boards are never training boards — by construction."""
    train, ev = ap.episode_seeds(0, 30, 10)
    assert len(train) == 30 and len(ev) == 10
    assert set(train).isdisjoint(ev)
    # and each stream is deterministic
    assert (train, ev) == ap.episode_seeds(0, 30, 10)


def test_evaluate_plays_every_arm_on_the_same_seeds():
    """Paired boards: each arm sees the same seed list."""
    seen: dict[str, list] = {}

    def factory(s: int) -> ar.Arena:
        seen.setdefault("seeds", []).append(s)
        return _arena(working=[_aff_step_case("w", 4)])

    arms = {
        "fixed": lambda _e: ar.FixedRule([ar.Action.ingest(())]),
        "learned": lambda _e: ap.LearnedPlayer(seed=0, learn=False),
    }
    table = ap.evaluate(
        arms, [7, 8], budget=3, arena_factory=factory
    )
    assert set(table) == {"fixed", "learned"}
    for rows in table.values():
        assert [r["seed"] for r in rows] == [7, 8]
        assert {"usable", "holdout_paid", "reward"} <= set(rows[0])


# ---------------------------------------------------------------------------
#  Learning signal — a paying move raises its own logit
# ---------------------------------------------------------------------------


def _two_move_legal(_s):
    """A filtered move set: one paying lift, one false fold."""
    return (
        ar.Action.fold(("add", "X", "X"), "abs", name="bad"),
        ar.Action.lift(
            ("add", ("matmul", "X1", "X2"), "X3"),
            ("aff", "X1", "X3"),
            "apply",
            state="X2",
        ),
    )


def test_training_prefers_the_paying_move_on_a_tiny_board():
    """REINFORCE over episode returns lifts the paying move's mass.

    The ``legal`` seam keeps the board honest but small enough that
    six episodes sample the lift many times — on the full ~50k-move
    board a single move is never sampled enough to learn (that is
    the measurement's question, not this test's).
    """
    kw = dict(
        working=[_aff_step_case("w", 4)],
        holdout=[_aff_step_case("holdout", 6)],
        pending=[_filler_case("pend")],
        base_rules=tuple(),
        carrier_rules=[("aff", "apply", True)],
    )
    player = ap.LearnedPlayer(seed=0, lr=0.5, legal=_two_move_legal)
    # uniform start: the paying lift's logit equals the fold's
    wd0 = player.weights_dict()
    assert wd0["op:lift"] == wd0["op:fold"] == 0.0
    totals = ap.train_player(
        player, range(8), budget=1, arena_factory=_factory(**kw)
    )
    assert max(totals) > 10.0  # usable+paid returns dominate
    wd = player.weights_dict()
    assert wd["op:lift"] > wd["op:fold"]
    # the frozen policy then plays the paying move first
    arena = _arena(**kw)
    move = player.frozen(seed=0, greedy=True)(arena.observe())
    assert move is not None and move.op == "lift"


def test_weights_dict_roundtrips_through_the_schema():
    """A trained table loads back by name — the lawdata shape."""
    player = ap.LearnedPlayer(seed=0, weights={"bias": 0.25})
    wd = player.weights_dict()
    assert wd["bias"] == 0.25
    clone = ap.LearnedPlayer(seed=1, weights=wd)
    assert clone.weights_dict() == wd


def test_main_smoke(tmp_path, capsys, monkeypatch):
    """The CLI driver runs end-to-end on a stub board factory."""
    import json

    monkeypatch.setattr(
        ap.ar, "make_arena", lambda seed, **kw: _lift_board()
    )
    out = tmp_path / "run.json"
    rc = ap.main(
        [
            "--train-episodes",
            "1",
            "--episodes",
            "1",
            "--budget",
            "6",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "learned-player comparison" in text
    assert "stage failures" in text
    assert "table" in json.loads(out.read_text())
