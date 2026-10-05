"""Tests for the guide seam in ``catopt_discovery.meta_game``.

The guide is the meta-policy over the generator inventory: an
``Allocation`` chooses which generator draws the next proposals, and
the shared ``Referee`` is the only referee — the guide steers
compute, never semantics.  These tests drive the seam on the same
six-term tiny corpus ``test_discovery_experiments`` uses: real
``pipeline`` proposal queues (the shape-aware and grammar arms emit
statically), real referee verdicts, a real in-memory evidence store,
and the construction player as the ``build`` arm.
"""

import json
import random

import pytest
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_discovery import meta_game as mg
from catopt_discovery import pipeline as lpl
from catopt_discovery.census import (
    CorpusTerm,
    op_tuple_census,
    shape_census,
)
from catopt_discovery.impact import TermCase


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _tiny_terms() -> list:
    """Six small, well-typed terms — the corpus every tool reads."""
    x = _v("x", 4, 4)
    y = _v("y", 4, 4)
    z = _v("z", 4, 4)
    w = Param("w", TensorType((4, 4)))
    return [
        _p("mul", _p("silu", x), y),
        _p("mul", x, _p("sigmoid", x)),
        _p("add", _p("mul", x, y), z),
        _p("mul", x, Const(1)),
        _p("select", _p("mul", x, y), dim=0, index=1),
        _p("add", _p("mul", x, w), y),
    ]


def _tiny_cases() -> list[TermCase]:
    """The corpus as ``TermCase``s, so fired candidates pay."""
    from catopt_discovery import workload_gen as wg

    cases = []
    for i, t in enumerate(_tiny_terms()):
        env = wg._leaf_env(t)
        c = wg.term_to_case(t, f"t{i}", "bench", env)
        cases.append(c)
    return cases


def _vocab():
    terms = _tiny_terms()
    cts = [CorpusTerm("t", f"m{i}", t) for i, t in enumerate(terms)]
    counts, _ = op_tuple_census(cts)
    sh, _ = shape_census(cts)
    return mg.build_vocab(terms, counts, sh, 6)


def _pools() -> dict:
    """Real generator queues over the tiny corpus."""
    terms = _tiny_terms()
    cts = [CorpusTerm("t", f"m{i}", t) for i, t in enumerate(terms)]
    counts, _ = op_tuple_census(cts)
    census_op = {k: v for k, v in counts.items()}
    return mg.generator_pools(census_op, terms, "hand")


def _referee(cases=()):
    return mg.Referee(_tiny_terms(), list(cases), None, None, [])


_META = {
    "corpus_hash": "c",
    "rules_hash": "r",
    "code_rev": "test",
    "run_id": "run",
    "holdout": "",
    "ts": "now",
}


def _arena(cases=(), game=None, **kw):
    pools = kw.pop("pools", None) or _pools()
    return mg.GuideArena(
        pools, _referee(cases), game=game, meta=dict(_META), **kw
    )


# ---------------------------------------------------------------------------
#  The inventory
# ---------------------------------------------------------------------------


def test_generator_pools_are_the_propose_sources():
    pools = _pools()
    assert set(pools) == set(mg.ENUMERATION_ORDER) - {"build"}
    # the static generators always emit; the census-derived arms are
    # honest about a tiny corpus (some may be empty).
    assert pools["shape-aware"]
    assert pools["algebraic-grammar"]
    # de-dup across arms: no alpha-key repeats in any queue.
    from catopt_discovery import proposal as lp

    seen = set()
    for queue in pools.values():
        for p in queue:
            key = lp._key(p.lhs, p.rhs)
            assert key not in seen
            seen.add(key)


def test_generator_pools_derived_vocab():
    terms = _tiny_terms()
    cts = [CorpusTerm("t", f"m{i}", t) for i, t in enumerate(terms)]
    counts, _ = op_tuple_census(cts)
    pools = mg.generator_pools(
        {k: v for k, v in counts.items()}, terms, "derived"
    )
    assert set(pools) == set(mg.ENUMERATION_ORDER) - {"build"}


# ---------------------------------------------------------------------------
#  The arena — draws, accounting, the evidence-store observation
# ---------------------------------------------------------------------------


def test_invest_draws_and_accounts():
    arena = _arena()
    vs = arena.invest(mg.Allocation("shape-aware", 3))
    assert len(vs) == 3
    assert arena.spent == 3
    st = arena.arms["shape-aware"]
    assert st.drawn == 3
    assert st.oracle_calls == arena.ref.oracle_calls
    # adjudicated verdicts reach the evidence store; the observation
    # reads them back through latest_verdicts.
    obs = arena.observation(10)
    assert obs.spent == 3
    assert obs.remaining["shape-aware"] == st.emitted - 3
    stored = set(obs.verdicts)
    adjudicated = {repr(k) for k in arena.ref.by_key}
    assert stored <= adjudicated
    assert len(stored) == len(arena.ref.by_key)


def test_invest_unknown_arm_raises():
    arena = _arena()
    with pytest.raises(KeyError, match="unknown generator"):
        arena.invest(mg.Allocation("nope", 1))


def test_invest_partial_draw_at_queue_end():
    arena = _arena()
    n = arena._remaining("pattern-recognition")
    vs = arena.invest(
        mg.Allocation("pattern-recognition", n + 10)
    )
    assert len(vs) == n


def test_verdict_evidence_mapping():
    p = lpl.Proposal(
        name="t", lhs=_p("mul", "U", Const(1)), rhs="U", family="f"
    )
    v = mg.Verdict(truth=True, reason="true", fires=2, paid=1)
    ev = mg._verdict_evidence(p, v)
    assert ev.num_true is True
    assert ev.fires == 2 and ev.paid == 1
    assert ev.shippable  # cleared the referee's bar
    v2 = mg.Verdict(reason="false")
    assert mg._verdict_evidence(p, v2).num_true is False
    v3 = mg.Verdict(reason="no-instance")
    assert mg._verdict_evidence(p, v3).num_true is None


def test_repeat_verdicts_not_recorded():
    """A cross-arm duplicate is a dedup artifact — not evidence."""
    lhs, rhs = _p("mul", "U", Const(1)), "U"
    dup = lpl.Proposal(name="a", lhs=lhs, rhs=rhs, family="f")
    pools = {
        "shape-aware": [dup],
        "algebraic-grammar": [
            lpl.Proposal(
                name="b",
                lhs=_p("mul", "V", Const(1)),
                rhs="V",
                family="f",
            )
        ],
    }
    arena = _arena(pools=pools)
    vs = arena.invest(mg.Allocation("shape-aware", 1))
    vs2 = arena.invest(mg.Allocation("algebraic-grammar", 1))
    assert vs[0].reason in ("true", "no-instance")
    assert vs2[0].reason.startswith(
        ("repeat", "true", "no-instance", "false")
    )
    obs = arena.observation(10)
    # at most one row per alpha-normal key.
    assert len(obs.verdicts) <= 1


def test_build_arm_plays_and_records():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    arena = _arena(game=game, plays_cap=4, seed_frac=0.0)
    assert "build" in arena.arms
    vs = arena.invest(mg.Allocation("build", 3))
    assert len(vs) == 3
    assert arena.plays_left == 1
    assert arena.arms["build"].drawn == 3


def test_build_arm_seeded_start():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(2))
    arena = _arena(game=game, plays_cap=2, seed_frac=1.0)
    vs = arena.invest(mg.Allocation("build", 1))
    assert len(vs) == 1
    assert vs[0].name.startswith("seed:")


def test_build_arm_without_game_raises():
    """Drawing 'build' with no bound game is a caller error, not a
    silent skip."""
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    arena = _arena(game=game, plays_cap=2)
    arena.game = None
    with pytest.raises(RuntimeError, match="no game"):
        arena.invest(mg.Allocation("build", 1))


def test_build_arm_mint_error_not_recorded(monkeypatch):
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    monkeypatch.setattr(mg.BuildGame, "candidate", lambda self: None)
    arena = _arena(game=game, plays_cap=2, seed_frac=0.0)
    vs = arena.invest(mg.Allocation("build", 1))
    assert vs[0].reason == "mint-error"
    assert arena.observation(10).verdicts == {}


def test_first_true_and_ship_marks():
    cases = _tiny_cases()
    from catopt_discovery.impact import _cost_fn
    from catopt_discovery.shape_proposal import _sink

    sink = _sink()
    ref = mg.Referee(
        _tiny_terms(), cases, sink, _cost_fn(sink), []
    )
    pools = {
        "shape-aware": [
            lpl.Proposal(
                name="id_mul",
                lhs=_p("mul", "U", Const(1)),
                rhs="U",
                family="t",
            ),
            lpl.Proposal(
                name="silu_fold",
                lhs=_p("mul", "U", _p("sigmoid", "U")),
                rhs=_p("silu", "U"),
                family="t",
            ),
        ]
    }
    arena = mg.GuideArena(pools, ref, meta=dict(_META))
    vs = arena.invest(mg.Allocation("shape-aware", 2))
    assert vs[0].truth and vs[0].paid
    assert arena.first_true_at == 1
    assert arena.first_ship_at == 1
    s = arena.summary(2)
    assert s["best"]["generator"] == "shape-aware"
    # both candidates cleared the referee's ship bar; the first-ship
    # mark is set once.
    assert s["arms"]["shape-aware"]["shippable"] == 2


def test_summary_no_verdicts():
    arena = _arena()
    s = arena.summary(0)
    assert s["best"] is None
    assert s["first_true_at"] is None
    assert s["first_ship_at"] is None
    assert s["referee"]["candidates"] == 0


# ---------------------------------------------------------------------------
#  The guides
# ---------------------------------------------------------------------------


def test_guide_base_hooks():
    g = mg.Guide()
    obs = _arena().observation(4)
    with pytest.raises(NotImplementedError):
        g.choose(obs)
    assert g.update(1.0, []) is None  # default: ignore the payoff


def test_enumeration_guide_drains_in_order():
    pools = _pools()
    order = [k for k in pools if pools[k]]
    g = mg.EnumerationGuide(order=order, step=2)
    arena = _arena(pools=pools)
    # two allocations of 2 each drain the first arm before moving on.
    obs = arena.observation(20)
    a1 = g.choose(obs)
    assert a1.generator == order[0]
    arena.invest(a1)
    arena.invest(g.choose(arena.observation(20)))
    if pools[order[0]] and len(pools[order[0]]) <= 4:
        a3 = g.choose(arena.observation(20))
        assert a3.generator != order[0] or len(order) == 1


def test_enumeration_guide_stops_when_drained():
    pools = {"shape-aware": _pools()["shape-aware"][:1]}
    arena = _arena(pools=pools)
    g = mg.EnumerationGuide(order=("shape-aware",), step=5)
    arena.invest(g.choose(arena.observation(10)))
    assert g.choose(arena.observation(10)) is None


def test_random_guide_only_picks_live_arms():
    pools = _pools()
    arena = _arena(pools=pools)
    g = mg.RandomGuide(random.Random(0), step=2)
    dead = {
        k for k, q in pools.items() if not q
    }
    for _ in range(5):
        a = g.choose(arena.observation(20))
        assert a.generator not in dead
        arena.invest(a)
    # drain everything -> None.
    for name, q in pools.items():
        arena.invest(mg.Allocation(name, len(q)))
    assert g.choose(arena.observation(10)) is None


def test_learned_guide_chooses_and_updates():
    pools = _pools()
    arena = _arena(pools=pools)
    g = mg.LearnedGuide(step=2, hidden=8)
    # update before any choose is a no-op.
    g.update(1.0, [])
    rewards = []
    while arena.spent < 12:
        obs = arena.observation(12)
        a = g.choose(obs)
        if a is None:
            break
        vs = arena.invest(a)
        r = sum(v.score for v in vs) / max(len(vs), 1)
        rewards.append(r)
        g.update(r, vs)
    assert rewards
    assert all(a is None or a.n == 2 for a in [a])
    # a drained board ends the game.
    for name in pools:
        arena.invest(mg.Allocation(name, len(pools[name])))
    assert g.choose(arena.observation(10)) is None


def test_learned_guide_feature_shapes():
    g = mg.LearnedGuide(step=1)
    obs = _arena().observation(4)
    sv = g._state_vec(obs)
    assert len(sv) == g._SDIM
    for name in obs.arms:
        assert len(g._arm_vec(obs, name)) == g._ADIM


# ---------------------------------------------------------------------------
#  run_guide / compare_guides — the comparison harness
# ---------------------------------------------------------------------------


def test_run_guide_enumeration_is_the_control():
    """Enumeration over per-arm queues == the fixed-order drain."""
    pools = _pools()
    total = sum(len(q) for q in pools.values())
    arena = _arena(pools=pools)
    g = mg.EnumerationGuide(
        order=[k for k in mg.ENUMERATION_ORDER if k in pools], step=3
    )
    s = mg.run_guide(arena, g, budget=total + 10)
    # the game stops when every arm drains, under the budget.
    assert s["draws"] == total
    assert s["referee"]["candidates"] == arena.ref.summary()[
        "candidates"
    ]
    per_arm = arena.arms
    assert sum(st.drawn for st in per_arm.values()) == total


def test_run_guide_budget_binds_and_none_stops():
    pools = _pools()
    arena = _arena(pools=pools)
    g = mg.EnumerationGuide(
        order=[k for k in mg.ENUMERATION_ORDER if k in pools], step=4
    )
    s = mg.run_guide(arena, g, budget=6)
    assert s["draws"] == 6
    assert s["rounds"] == 2  # 4 + 2 — the tail clamps to the budget


def test_run_guide_stops_on_drained_allocation():
    """A guide that keeps allocating to a dry arm ends the game."""
    pools = _pools()
    arena = _arena(pools=pools)

    class _Stubborn(mg.Guide):
        def choose(self, obs):
            return mg.Allocation("pattern-recognition", 2)

    s = mg.run_guide(arena, _Stubborn(), budget=50)
    n = len(pools["pattern-recognition"])
    assert s["draws"] == n  # drained, then the run stopped


def test_compare_guides_fresh_arenas():
    pools = _pools()
    budget = 8
    out = mg.compare_guides(
        lambda: _arena(pools=pools),
        {
            "enum": lambda: mg.EnumerationGuide(
                order=tuple(pools), step=4
            ),
            "rand": lambda: mg.RandomGuide(random.Random(1), step=4),
        },
        budget,
    )
    assert set(out) == {"enum", "rand"}
    for s in out.values():
        assert s["draws"] == min(budget, sum(len(q) for q in pools.values()))
        assert "referee" in s and "arms" in s


def test_compare_guides_learned_arm():
    pools = _pools()
    out = mg.compare_guides(
        lambda: _arena(pools=pools),
        {"learned": lambda: mg.LearnedGuide(step=4, hidden=8)},
        8,
    )
    s = out["learned"]
    assert s["draws"] == 8
    assert sum(a["drawn"] for a in s["arms"].values()) == 8


def test_build_arm_in_a_run():
    vocab = _vocab()
    pools = {"shape-aware": _pools()["shape-aware"][:2]}

    def make():
        game = mg.BuildGame(vocab, random.Random(0))
        return _arena(pools=pools, game=game, plays_cap=4)

    g = mg.RandomGuide(random.Random(0), step=2)
    s = mg.run_guide(make(), g, budget=6)
    assert s["draws"] == 6
    assert "build" in s["arms"]


# ---------------------------------------------------------------------------
#  run_guide_experiment + the --guide CLI path
# ---------------------------------------------------------------------------


def test_run_guide_experiment_tiny(monkeypatch, tmp_path, capsys):
    """``--guide`` end to end on the tiny corpus: real queues, real
    referees, zero-case slice so nothing pays."""
    cases = _tiny_cases()
    monkeypatch.setattr(mg, "_bench_cases", lambda: (cases[:4], []))
    monkeypatch.setattr(mg, "model_cases", lambda: (cases[4:], []))
    out = tmp_path / "guide.json"
    rc = mg.main(
        [
            "--guide",
            "--budget",
            "8",
            "--guide-step",
            "4",
            "--plays-cap",
            "2",
            "--top-seeds",
            "3",
            "--seed-frac",
            "0.5",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert set(payload["results"]) == {
        "enumeration",
        "uniform",
        "learned",
    }
    assert "yield_tf_ratio_vs_enumeration" in payload
    for s in payload["results"].values():
        assert s["draws"] <= 8
        assert "referee" in s and "arms" in s
    printed = capsys.readouterr().out
    assert "guide seam" in printed
    assert "yield_tf=" in printed


def test_run_guide_experiment_no_build(monkeypatch, capsys):
    cases = _tiny_cases()
    monkeypatch.setattr(mg, "_bench_cases", lambda: (cases[:4], []))
    monkeypatch.setattr(mg, "model_cases", lambda: (cases[4:], []))
    rc = mg.main(
        [
            "--guide",
            "--no-guide-build",
            "--budget",
            "4",
            "--guide-step",
            "4",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "guide seam" in out


def test_run_guide_experiment_trained_build(monkeypatch, capsys):
    """--guide-train pre-trains the build arm with the existing
    ``train`` loop — the seam reuses the trainer, none is new."""
    cases = _tiny_cases()
    monkeypatch.setattr(mg, "_bench_cases", lambda: (cases[:4], []))
    monkeypatch.setattr(mg, "model_cases", lambda: (cases[4:], []))
    rc = mg.main(
        [
            "--guide",
            "--guide-train",
            "2",
            "--budget",
            "6",
            "--guide-step",
            "3",
            "--plays-cap",
            "2",
            "--top-seeds",
            "2",
            "--hidden",
            "8",
        ]
    )
    assert rc == 0


def test_print_guide_report_branches(capsys):
    """The report walks best/first-ship/first-true lines honestly."""
    result = {
        "pools": {"shape-aware": 2},
        "results": {
            "enumeration": {
                "draws": 2,
                "rounds": 1,
                "budget": 2,
                "oracle_calls": 1,
                "probes": 0,
                "arms": {
                    "shape-aware": {
                        "emitted": 2,
                        "drawn": 2,
                        "oracle_calls": 1,
                        "true": 1,
                        "firing": 0,
                        "new_tf": 0,
                        "shippable": 0,
                        "best": 1.5,
                    }
                },
                "referee": {
                    "true": 1,
                    "true_firing": 0,
                    "new_true_firing": 0,
                    "shippable": 0,
                    "yield_tf_per_call": 0.0,
                },
                "best": {
                    "name": "c",
                    "score": 1.5,
                    "generator": "shape-aware",
                    "lhs": "a",
                    "rhs": "b",
                },
                "first_true_at": 1,
                "first_ship_at": None,
            },
            "uniform": {
                "draws": 4,
                "rounds": 1,
                "budget": 4,
                "oracle_calls": 0,
                "probes": 0,
                # no arm drew — the arms line is skipped honestly.
                "arms": {"build": {"drawn": 0, "emitted": 4,
                                   "oracle_calls": 0, "true": 0,
                                   "firing": 0, "new_tf": 0,
                                   "shippable": 0, "best": 0.0}},
                "referee": {
                    "true": 0,
                    "true_firing": 0,
                    "new_true_firing": 0,
                    "shippable": 1,
                    "yield_tf_per_call": 0.0,
                },
                # a ship mark without a scored best prints its line.
                "best": None,
                "first_true_at": None,
                "first_ship_at": 3,
            },
        },
        "yield_tf_ratio_vs_enumeration": {
            "enumeration": 1.0,
            "uniform": None,
        },
    }
    mg._print_guide_report(result)
    out = capsys.readouterr().out
    assert "first true at draw 1" in out
    assert "first shippable at draw 3" in out
    assert "(1.00x enum)" in out
