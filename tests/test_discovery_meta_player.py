"""The learned meta-arena player — Q3's arm.

Pins: the featurizer covers the schema and discriminates the paying
line's preconditions (a handleable spec, an unhandled declaration);
the player plays only legal unplayed moves, learns (weights move
under REINFORCE reward-to-go), freezes for eval; the driver's
case corpus mixes paying and non-paying boards.
"""

import random

from catopt_core.cost.basic import count_cost
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import DEFAULT
from catopt_discovery import lawdata
from catopt_discovery import meta_arena as ma
from catopt_discovery import meta_player as mp


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _silu_board():
    """A board whose mined spec is handler-coverable."""
    x = _v("x", 4, 4)
    sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
    return ma.MetaArena(
        Op.make("mul", x, Op.make("sigmoid", x)),
        sans,
        cost_fn=count_cost,
    )


class TestFeaturize:
    def test_schema_subset(self):
        st = _silu_board().observe()
        for a in ma.legal_actions(st):
            feats = mp.featurize(st, a)
            assert set(feats) <= set(lawdata.META_ARENA_FEATURES)
            assert feats["bias"] == 1.0
            assert feats[f"op:{a.op}"] == 1.0

    def test_handleable_spec_detected(self):
        st = _silu_board().observe()
        assert mp._handleable(st) == 1
        # a declare row carries the precondition interaction
        d = [a for a in ma.legal_actions(st) if a.op == "declare"]
        assert d and mp.featurize(st, d[0])["x:handleable:declare"] > 0

    def test_control_board_has_no_handleable(self):
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        arena = ma.MetaArena(
            Op.make(
                "add", Op.make("mul", x, y), Op.make("matmul", x, y)
            ),
            DEFAULT,
            cost_fn=count_cost,
        )
        st = arena.observe()
        assert mp._handleable(st) == 0

    def test_action_params(self):
        st = _silu_board().observe()
        by_op = {}
        for a in ma.legal_actions(st):
            by_op.setdefault(a.op, []).append(a)
        sat_b = [
            a
            for a in by_op["saturate"]
            if a.params.get("budget") is not None
        ]
        sat_u = [
            a
            for a in by_op["saturate"]
            if a.params.get("budget") is None
        ]
        assert mp.featurize(st, sat_b[0])["a:budget"] > 0
        assert mp.featurize(st, sat_u[0])["a:unbounded"] == 1.0
        f = mp.featurize(st, by_op["fire"][0])
        assert any(k.startswith("a:r:") for k in f)

    def test_spec_index_parsing(self):
        named = ma.Action.declare(
            {"op": "fold", "params": {"name": "foldabs_7"}}
        )
        assert mp._spec_index(named) == 7 / 32.0
        unnamed = ma.Action.declare({"op": "fold", "params": {}})
        assert mp._spec_index(unnamed) == 0.5

    def test_hist_rates(self):
        h = mp._Hist()
        h.record("fire")
        h.record("fire")
        h.record("extract")
        assert h.rate("fire") == 2 / 3
        assert h.rate("declare") == 0.0


class TestPlayer:
    def test_plays_legal_unplayed(self):
        p = mp.MetaLearnedPlayer(0, learn=False)
        arena = _silu_board()
        played = set()
        for _ in range(8):
            st = arena.observe()
            if st.done:
                break
            a = p(st)
            assert a is not None and a in ma.legal_actions(st)
            assert mp._key(a) not in played
            played.add(mp._key(a))
            arena.step(a)

    def test_greedy_is_deterministic(self):
        a = mp.MetaLearnedPlayer(0, greedy=True, learn=False)
        b = mp.MetaLearnedPlayer(0, greedy=True, learn=False)
        st = _silu_board().observe()
        assert a(st) == b(st)

    def test_learn_moves_weights(self):
        p = mp.MetaLearnedPlayer(0)
        before = p.weights_dict()
        arena = _silu_board()
        traj = ma.run_episode(arena, p, 24)
        p.finish_episode(
            traj.total, rewards=[r.reward for r in traj.reports]
        )
        after = p.weights_dict()
        assert any(before[k] != after[k] for k in before)

    def test_finish_without_rewards_falls_back(self):
        p = mp.MetaLearnedPlayer(0)
        arena = _silu_board()
        traj = ma.run_episode(arena, p, 6)
        p.finish_episode(traj.total)  # no per-step rewards

    def test_frozen_does_not_learn(self):
        p = mp.MetaLearnedPlayer(0)
        q = p.frozen(3, greedy=True)
        arena = _silu_board()
        traj = ma.run_episode(arena, q, 6)
        q.finish_episode(traj.total)
        assert all(v == 0.0 for v in q.weights_dict().values())

    def test_epsilon_explores(self):
        # epsilon=1 forces the uniform branch; still legal, still plays
        p = mp.MetaLearnedPlayer(0, epsilon=1.0)
        arena = _silu_board()
        traj = ma.run_episode(arena, p, 6)
        assert traj.reports

    def test_drains_to_none(self):
        # a legal set of one: play it, then nothing remains
        def one(state):
            return (ma.Action.extract(),)

        p = mp.MetaLearnedPlayer(0, legal=one)
        arena = _silu_board()
        st = arena.observe()
        assert p(st) is not None
        arena.step(ma.Action.extract())
        st2 = arena.observe()
        assert p(st2) is None

    def test_episode_boundary_resets(self):
        p = mp.MetaLearnedPlayer(0, learn=False)
        arena = _silu_board()
        ma.run_episode(arena, p, 4)
        arena2 = _silu_board()
        st = arena2.observe()
        assert st.steps == 0
        # second board: the played mask must not carry over
        a = p(st)
        assert a is not None


class TestDriver:
    def test_gen_cases_kinds(self):
        cases = mp.gen_cases(0, 20)
        kinds = {c[0].split(":", 1)[1] for c in cases}
        assert len(cases) == 20
        assert {"silu", "control"} <= kinds

    def test_train_and_evaluate(self):
        cases = mp.gen_cases(1, 4)
        p = mp.MetaLearnedPlayer(1)
        totals = mp.train_player(p, cases, budget=8)
        assert len(totals) == 4
        arms = {
            "scripted": lambda i: ma.ScriptedPlayer(),
            "learned": lambda i: p.frozen(i, greedy=True),
        }
        table = mp.evaluate(arms, cases[:2], budget=8)
        assert set(table) == {"scripted", "learned"}
        assert len(table["scripted"]) == 2
        text = mp.player_table(table)
        assert "scripted" in text and "reward" in text

    def test_winning_line_reachable(self):
        # the hand-set policy proves the schema expresses the line:
        # declare+saturate+handle+extract beats scripted on a silu site
        w = {n: 0.0 for n in lawdata.META_ARENA_FEATURES}
        w.update(
            {
                "op:declare": 2.0,
                "op:saturate": 1.5,
                "a:unbounded": 1.0,
                "op:handle": 3.0,
                "op:extract": 0.5,
                "x:extract:improve": 8.0,
            }
        )
        case = [
            c
            for c in mp.gen_cases(7919, 10)
            if "silu" == c[0].split(":")[1]
        ][0]
        scripted = ma.MetaArena(case[1], case[2], cost_fn=count_cost)
        ma.run_episode(scripted, ma.ScriptedPlayer(), 24)
        arena = mp._board(case, count_cost, 20_000)
        p = mp.MetaLearnedPlayer(0, weights=w, greedy=True, learn=False)
        traj = ma.run_episode(arena, p, 24)
        t = traj.terminal
        assert t is not None and t.certificate_ok
        base = scripted.eg.extract_best(
            scripted.root, scripted.feasible_cost
        )
        from catopt_core.cost.params import dag_cost

        assert t.cost < dag_cost(base, scripted.feasible_cost)

    def test_main(self, capsys, monkeypatch):
        real = mp.gen_cases
        monkeypatch.setattr(mp, "gen_cases", lambda s, n: real(0, 2))
        assert (
            mp.main(
                [
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "learned-greedy" in out

    def test_main_json(self, capsys, monkeypatch, tmp_path):
        real = mp.gen_cases
        monkeypatch.setattr(mp, "gen_cases", lambda s, n: real(0, 2))
        out_path = tmp_path / "run.json"
        assert (
            mp.main(
                [
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                    "--json",
                    str(out_path),
                ]
            )
            == 0
        )
        import json

        d = json.loads(out_path.read_text())
        assert "weights" in d and "table" in d


class TestInternals:
    def test_bucket_stable(self):
        assert mp._bucket("silu_fold", "rule") == mp._bucket(
            "silu_fold", "rule"
        )
        assert 0 <= mp._bucket("x", "handler") < 4

    def test_dot(self):
        assert mp._dot({"a": 2.0}, {"a": 3.0, "b": 1.0}) == 6.0

    def test_grad_row(self):
        phi, ephi = mp._grad_row(
            [{"a": 1.0}, {"a": 0.0, "b": 1.0}], [1.0, 1.0], 0
        )
        assert phi == {"a": 1.0}
        assert ephi == {"a": 0.5, "b": 0.5}

    def test_sample_tail_fallback(self):
        assert mp.MetaLearnedPlayer(0)._sample([]) == -1

    def test_foreign_action_gets_no_schema_keys(self):
        st = _silu_board().observe()
        feats = mp.featurize(st, ma.Action("bogus", {}))
        assert set(feats) <= set(lawdata.META_ARENA_FEATURES)
        assert not any(k.startswith("op:") for k in feats)
