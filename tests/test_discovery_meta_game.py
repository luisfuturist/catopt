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
from catopt_discovery import evidence as ev_store
from catopt_discovery import meta_game as mg
from catopt_discovery import pipeline as lpl
from catopt_discovery import proposal as lp
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


# ---------------------------------------------------------------------------
#  Corpus arms — scope rotation, legality, the gap re-adjudication
# ---------------------------------------------------------------------------


def _corpus_arena(cases=(), pools=None, **kw):
    """An arena with the corpus arms armed (real sink, real corpus)."""
    from catopt_discovery.impact import _cost_fn
    from catopt_discovery.shape_proposal import _sink

    sink = _sink()
    ref = mg.Referee(
        _tiny_terms(), list(cases), sink, _cost_fn(sink), []
    )
    cts = [
        CorpusTerm("t", f"m{i}", t)
        for i, t in enumerate(_tiny_terms())
    ]
    kw.setdefault("gen_cap", 6)
    kw.setdefault("gen_rng", random.Random(0))
    kw.setdefault("corpus", cts)
    return mg.GuideArena(
        pools if pools is not None else _pools(),
        ref,
        meta=dict(_META),
        **kw,
    )


def _gap_proposal() -> lpl.Proposal:
    """A candidate whose LHS never matches the tiny corpus."""
    return lpl.Proposal(
        name="gaptest",
        lhs=_p("tanh", _p("div", "U", "V")),
        rhs=_p("sigmoid", _p("div", "U", "V")),
        family="t",
        sources=("t",),
    )


def test_corpus_arms_need_supported_ops():
    """Corpus arms without a sink's op table are a caller error."""
    with pytest.raises(ValueError, match="supported_ops"):
        _arena(gen_cap=2)


def test_workload_arm_ingests_and_rotates_scope():
    arena = _corpus_arena()
    assert set(mg.CORPUS_ARMS) <= set(arena.arms)
    obs = arena.observation(10)
    assert obs.remaining["workload_gen"] == 6
    # gap_gen is targeted: no no-instance candidate yet -> not live.
    assert obs.remaining["gap_gen"] == 0
    assert obs.corpus_size == len(_tiny_terms())

    old_hash = arena.meta["corpus_hash"]
    vs = arena.invest(mg.Allocation("workload_gen", 2))
    assert [v.reason for v in vs] == ["generated", "generated"]
    # the draws minted real workloads: both lists grew, the scope
    # rotated once per ingestion, and the run_id is stable.
    assert arena.generated == 2
    assert len(arena.ref.terms) == obs.corpus_size + 2
    assert len(arena.ref.cases) == 2  # nothing fired yet; cases grew
    assert arena.scope_epoch == 2
    assert arena.meta["corpus_hash"] != old_hash
    assert arena.meta["run_id"] == _META["run_id"]
    # a corpus draw is a draw, not an oracle call.
    assert arena.arms["workload_gen"].drawn == 2
    assert arena.arms["workload_gen"].oracle_calls == 0


def test_workload_arm_mints_legal_terms():
    """Ingested terms are verified workloads, not raw samples."""
    from catopt_core.typing import INVALID, _shape_of
    from catopt_discovery import workload_gen as wg

    arena = _corpus_arena(cases=_tiny_cases())
    arena.invest(mg.Allocation("workload_gen", 1))
    case = arena.ref.cases[-1]
    # the term passed the same gates the intake path enforces: an
    # op term, well-typed, Var-bearing, lowerable, and novel.
    assert isinstance(case.term, Op)
    assert _shape_of(case.term) is not INVALID
    assert case.feed  # a real measurable TermCase
    st = wg.corpus_stats(_tiny_cases())
    assert wg.shape_key(case.term) not in st.root_keys


def test_scope_rotation_isolates_verdicts():
    """A verdict under corpus_A is not served as corpus_B evidence."""
    arena = _corpus_arena()
    old_hash = arena.meta["corpus_hash"]
    arena.invest(mg.Allocation("algebraic-grammar", 2))
    n_scoped = len(arena.observation(10).verdicts)
    assert n_scoped >= 1  # rows recorded under corpus_A
    arena.invest(mg.Allocation("workload_gen", 1))
    obs = arena.observation(10)
    # the grown corpus is a new scope: only post-mutation rows serve.
    assert obs.verdicts == {}
    assert obs.scope_epoch == 1
    # ... but the corpus_A rows remain attributable under their hash.
    old_rows = ev_store.latest_verdicts(
        arena.conn, old_hash, "r", "test"
    )
    assert len(old_rows) == n_scoped
    # and a fresh adjudication lands under the new scope.
    arena.invest(mg.Allocation("algebraic-grammar", 3))
    new_hash = arena.meta["corpus_hash"]
    for row in arena.observation(10).verdicts.values():
        assert row["corpus_hash"] == new_hash


def test_gap_arm_witnesses_no_instance_candidate():
    cases = _tiny_cases()
    prop = _gap_proposal()
    arena = _corpus_arena(
        cases=cases, pools={"shape-aware": [prop]}
    )
    vs = arena.invest(mg.Allocation("shape-aware", 1))
    assert vs[0].reason == "no-instance"
    assert arena._remaining("gap_gen") == 1

    old_hash = arena.meta["corpus_hash"]
    vs2 = arena.invest(mg.Allocation("gap_gen", 1))
    v = vs2[0]
    assert arena.arms["gap_gen"].drawn == 1
    # synthesis ingested witness case(s) and rotated the scope; the
    # candidate was re-adjudicated under the grown corpus — the
    # verdict is re-measured (an oracle call was honestly spent).
    assert arena.generated >= 1
    assert arena.scope_epoch >= 1
    assert arena.meta["corpus_hash"] != old_hash
    assert v.reason != "no-instance"
    assert arena.arms["gap_gen"].oracle_calls >= 1
    # the target is spent: gap_gen is not live for it again.
    assert arena._remaining("gap_gen") == 0
    # the re-measured verdict sits under the *new* scope only.
    new_rows = arena.observation(10).verdicts
    assert len(new_rows) == 1
    assert next(iter(new_rows.values()))["corpus_hash"] == (
        arena.meta["corpus_hash"]
    )
    old_rows = ev_store.latest_verdicts(
        arena.conn, old_hash, "r", "test"
    )
    assert len(old_rows) == 1  # the corpus_A no-instance row remains


def test_gap_arm_no_target_is_honest_miss():
    """An untargetable gap draw spends the draw, not a verdict."""
    arena = _corpus_arena()
    # the arm's own remaining is 0 — invest drains nothing, and a
    # forced draw reports honestly instead of inventing a target.
    assert arena._remaining("gap_gen") == 0
    assert arena.invest(mg.Allocation("gap_gen", 1)) == []
    v = arena._draw_gap()
    assert v.reason == "no-target"


def test_gap_arm_synthesis_miss(monkeypatch):
    """A failed synthesis is a ``gap-miss`` draw — no free verdict."""
    arena = _corpus_arena(pools={"shape-aware": [_gap_proposal()]})
    arena.invest(mg.Allocation("shape-aware", 1))
    assert arena._remaining("gap_gen") == 1
    monkeypatch.setattr(
        mg.lgg, "gen_cases_for", lambda *a, **k: ([], {})
    )
    vs = arena.invest(mg.Allocation("gap_gen", 1))
    assert vs[0].reason == "gap-miss"
    assert arena.generated == 0 and arena.scope_epoch == 0


def test_workload_arm_gen_miss(monkeypatch):
    """When no candidate survives the gates the draw is a miss."""
    arena = _corpus_arena(gen_cap=1)
    monkeypatch.setattr(mg.lwg._Resampler, "sample", lambda s: None)
    monkeypatch.setattr(mg.lwg, "mutant_term", lambda *a: None)
    vs = arena.invest(mg.Allocation("workload_gen", 1))
    assert vs[0].reason == "gen-miss"
    assert arena.generated == 0 and arena.scope_epoch == 0
    assert arena.arms["workload_gen"].drawn == 1


def test_workload_arm_gate_and_case_rejects(monkeypatch):
    """The intake gates run per draw; rejections stay uningested."""
    arena = _corpus_arena(gen_cap=2)
    monkeypatch.setattr(mg.lwg, "valid_term", lambda *a: None)
    assert (
        arena.invest(mg.Allocation("workload_gen", 1))[0].reason
        == "gen-miss"
    )
    monkeypatch.setattr(
        mg.lwg,
        "valid_term",
        lambda c, st, seen, sup: {},  # gates pass vacuously
    )
    monkeypatch.setattr(mg.lwg, "term_to_case", lambda *a: None)
    assert (
        arena.invest(mg.Allocation("workload_gen", 1))[0].reason
        == "gen-miss"
    )
    assert arena.generated == 0


def test_corpus_defaults_to_referee_terms():
    """``corpus=None`` wraps the referee's own terms."""
    arena = _corpus_arena(corpus=None)
    assert [c.term for c in arena._corpus_terms] == _tiny_terms()


def test_corpus_arms_explicit_supported():
    """``supported=`` overrides the referee sink's op table."""
    sup = frozenset({"mul", "add", "sigmoid"})
    arena = _corpus_arena(supported=sup)
    assert arena._supported == sup
    assert arena._remaining("workload_gen") == 6


def test_pool_regrowth_only_appends_novel_keys(monkeypatch):
    arena = _corpus_arena()
    before = {k: len(q) for k, q in arena.pools.items()}
    arena._regrow_pools()
    arena._regrow_pools()  # idempotent — nothing re-queued
    assert {k: len(q) for k, q in arena.pools.items()} == before
    keys = set()
    for q in arena.pools.values():
        for p in q:
            keys.add(lp._key(p.lhs, p.rhs))
    assert len(keys) == sum(len(q) for q in arena.pools.values())

    # a genuinely novel candidate IS appended — the grown census
    # feeds the corpus-derived arms; queued and adjudicated keys
    # are both skipped.
    fresh = _gap_proposal()
    assert lp._key(fresh.lhs, fresh.rhs) not in keys
    monkeypatch.setattr(
        mg.lpl, "_census_naturality", lambda *a: [fresh]
    )
    arena._vocab_sets = None
    arena._regrow_pools()
    assert arena.pools["census-naturality"][-1] is fresh
    assert arena.arms["census-naturality"].emitted == 1
    arena._regrow_pools()  # queued key — no double-queue
    assert len(arena.pools["census-naturality"]) == 1
    # an already-adjudicated key is skipped via the referee's dedup.
    arena._queued.discard(lp._key(fresh.lhs, fresh.rhs))
    arena.ref.dedup.add(lp._key(fresh.lhs, fresh.rhs))
    arena._regrow_pools()
    assert len(arena.pools["census-naturality"]) == 1


def test_enumeration_drains_corpus_arms_last():
    """The control's order is fixed-inventory-first, corpus last."""
    pools = {
        "shape-aware": _pools()["shape-aware"][:1],
    }
    arena = _corpus_arena(pools=pools, gen_cap=2)
    g = mg.EnumerationGuide(
        order=mg.ENUMERATION_ORDER + mg.CORPUS_ARMS, step=4
    )
    a1 = g.choose(arena.observation(10))
    assert a1.generator == "shape-aware"
    arena.invest(a1)
    a2 = g.choose(arena.observation(10))
    assert a2.generator == "workload_gen"
    arena.invest(a2)
    a3 = g.choose(arena.observation(10))
    # workload_gen is drained; gap_gen is live only if the earlier
    # draws left a no-instance candidate on the board.
    assert a3 is None or a3.generator == "gap_gen"


def test_corpus_arms_in_a_guided_run():
    """A run over corpus arms terminates and accounts honestly."""
    arena = _corpus_arena(
        pools={"shape-aware": _pools()["shape-aware"][:2]},
        gen_cap=2,
    )

    class _Grower(mg.Guide):
        def choose(self, obs):
            for name in ("workload_gen", "shape-aware"):
                if obs.remaining.get(name, 0) > 0:
                    return mg.Allocation(name, 1)
            return None

    s = mg.run_guide(arena, _Grower(), budget=6)
    assert s["draws"] <= 6
    assert s["generated"] >= 1
    assert s["scope_epoch"] == s["generated"]
    assert "workload_gen" in s["arms"]
    assert s["arms"]["workload_gen"]["drawn"] >= 1


def test_guide_corpus_cli(monkeypatch, tmp_path, capsys):
    """``--guide --guide-corpus`` end to end on the tiny corpus."""
    cases = _tiny_cases()
    monkeypatch.setattr(mg, "_bench_cases", lambda: (cases[:4], []))
    monkeypatch.setattr(mg, "model_cases", lambda: (cases[4:], []))
    out = tmp_path / "guide.json"
    rc = mg.main(
        [
            "--guide",
            "--budget",
            "10",
            "--guide-step",
            "3",
            "--guide-corpus",
            "2",
            "--plays-cap",
            "2",
            "--top-seeds",
            "3",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    # corpus arms armed -> the corpus-first control joins the board.
    assert set(payload["results"]) == {
        "enumeration",
        "enum-corpus-first",
        "uniform",
        "learned",
    }
    for s in payload["results"].values():
        assert "workload_gen" in s["arms"]
        assert "gap_gen" in s["arms"]
        assert "corpus_size" in s and "scope_epoch" in s
