"""The generic game engine + the play tool.

Pins: the Board contract runs any conforming domain through one
driver; the joint domain joints forward+gradients into a legal
board program; the search adapter wraps SearchEnv; the learned arm
is domain-agnostic (featurizer injected, not domain-bound).
"""

import random

import pytest

from catopt_core.cost.basic import count_cost
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import DEFAULT
from catopt_discovery import engine, meta_arena as ma
from catopt_discovery import meta_player as mp
from catopt_discovery import play, players
from catopt_discovery.players import LinearPolicy


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


class _ToyBoard:
    """A minimal Board: each step counts; 'stop' ends it."""

    def __init__(self):
        self.n = 0
        self._done = False

    def observe(self):
        return type(
            "S",
            (),
            {"steps": self.n, "done": self._done},
        )()

    def step(self, action):
        self.n += 1
        if action == "stop" or self.n >= 5:
            self._done = True
        return self.observe(), engine.Report(
            action, reward=-1.0, terminal=self._done
        )


def _toy_legal(state):
    return ("go", "go2", "stop") if not state.done else ()


class TestEngine:
    def test_board_protocol(self):
        assert isinstance(_ToyBoard(), engine.Board)

    def test_run_episode_budget(self):
        traj = engine.run_episode(_ToyBoard(), lambda s: "go", 10)
        assert len(traj.reports) == 5  # board's own done at n=5
        assert traj.total == -5.0
        assert traj.terminal is not None

    def test_run_episode_player_none(self):
        traj = engine.run_episode(_ToyBoard(), lambda s: None, 10)
        assert len(traj.reports) == 0
        assert traj.terminal is None

    def test_done_state_stops(self):
        traj = engine.run_episode(_ToyBoard(), lambda s: "stop", 10)
        assert len(traj.reports) == 1
        assert traj.terminal.terminal

    def test_train_policy_calls_finish(self):
        calls = []

        class P:
            def __call__(self, s):
                return "stop"

            def finish_episode(self, total, rewards=None):
                calls.append((total, list(rewards or ())))

        totals = engine.train_policy(
            P(), ["a", "b"], lambda c: _ToyBoard(), budget=4
        )
        assert len(totals) == 2 and len(calls) == 2
        assert calls[0][0] == -1.0 and calls[0][1] == [-1.0]

    def test_evaluate_paired(self):
        arms = {"stopper": lambda i: lambda s: "stop"}
        table = engine.evaluate(
            arms, [("c0",), ("c1",)], lambda c: _ToyBoard(), budget=4
        )
        assert len(table["stopper"]) == 2
        assert table["stopper"][0]["reward"] == -1.0


class TestLinearPolicy:
    def test_generic_domain_plays(self):
        # the same policy class drives a foreign board — featurizer
        # injected, nothing meta-arena-specific
        feats = ("bias",)
        p = LinearPolicy(
            0,
            legal=_toy_legal,
            featurizer=lambda s, a, h: {"bias": 1.0},
            features=feats,
            learn=False,
        )
        assert p.weights_dict() == {"bias": 0.0}
        traj = engine.run_episode(_ToyBoard(), p, 6)
        assert len(traj.reports) <= 3  # go, go2, stop then drain

    def test_learns_on_generic_board(self):
        p = LinearPolicy(
            0,
            legal=_toy_legal,
            featurizer=lambda s, a, h: {
                "bias": 1.0,
                f"act:{a}": 1.0,
            },
            features=("bias", "act:go", "act:go2", "act:stop"),
        )
        engine.train_policy(
            p, range(4), lambda c: _ToyBoard(), budget=6
        )
        w = p.weights_dict()
        assert any(v != 0.0 for v in w.values())

    def test_action_key_nonparams(self):
        assert players._action_key("plain") == ("call", "'plain'")


class TestDomains:
    def test_registry(self):
        assert set(play.DOMAINS) == {"meta", "joint", "search", "torch"}

    def test_torch_domain(self):
        # real modules export and board under the sink bound
        pytest.importorskip("torch")
        cases = play._torch_cases(0, 3)
        assert cases
        name, term, _r, _d, sup = cases[0][:5]
        assert isinstance(sup, frozenset)
        arena = play._torch_board(cases[0])
        assert isinstance(arena, ma.MetaArena)
        # the scripted arm plays the whole game
        traj = play.play("torch", cases[0], ma.ScriptedPlayer(), budget=6)
        assert traj.terminal is not None

    def test_play_api_meta(self):
        traj = play.play(
            "meta",
            play.mp.gen_cases(0, 1)[0],
            ma.ScriptedPlayer(),
            budget=8,
        )
        assert traj.reports and traj.terminal is not None

    def test_joint_cases_and_board(self):
        cases = play._joint_cases(0, 4)
        assert len(cases) == 4
        arena = play._joint_board(cases[0])
        assert isinstance(arena, ma.MetaArena)
        # the joint program is a legal board citizen
        st, rep = arena.step(ma.Action.saturate())
        assert rep.applied

    def test_search_board_contract(self):
        board = play._search_board(play._search_cases(0, 1)[0])
        st = board.observe()
        assert not st.done and st.actions
        st2, rep = board.step(st.actions[0])
        assert isinstance(rep, engine.Report)
        assert st2.steps == 1

    def test_search_scripted(self):
        go = play._scripted_search()
        board = play._search_board(play._search_cases(0, 1)[0])
        st = board.observe()
        a = go(st)
        assert a in st.actions

    def test_main_each_domain(self, capsys):
        for domain in ("meta", "joint", "search"):
            # small real run — the registry must hold together
            rc = play.main(
                [
                    "--domain",
                    domain,
                    "--train-cases",
                    "2",
                    "--eval-cases",
                    "2",
                    "--budget",
                    "6",
                ]
            )
            assert rc == 0
            out = capsys.readouterr().out
            assert domain in out and "learned" in out


class TestEdges:
    def test_joint_all_archetypes(self):
        # sweep seeds until all three archetypes appear
        seen = set()
        for s in range(20):
            for c in play._joint_cases(s, 6):
                seen.add(c[0].split(":")[1])
        assert seen == {"0", "1", "2", "3"}

    def test_scripted_search_drains(self):
        go = play._scripted_search()
        board = play._search_board(play._search_cases(0, 1)[0])
        st = board.observe()
        for _ in range(len(st.actions) + 2):
            a = go(st)
            if a is None:
                break
            st, _ = board.step(a)
        assert go(st) is None

    def test_arms_without_scripted(self):
        d = play.Domain(
            cases=lambda s, n: [],
            board_of=lambda c: None,
            legal=_toy_legal,
            featurizer=lambda s, a, h: {},
        )
        arms = play._arms(d, LinearPolicy(0, legal=_toy_legal,
                                          featurizer=lambda s, a, h: {}), 0)
        assert "scripted" not in arms and "learned" in arms

    def test_train_policy_no_learner(self):
        # a plain callable has no finish_episode — covered arm
        totals = engine.train_policy(
            lambda s: "stop", ["a"], lambda c: _ToyBoard(), budget=3
        )
        assert totals == [-1.0]

    def test_main_json(self, tmp_path, capsys):
        out_path = tmp_path / "run.json"
        rc = play.main(
            [
                "--domain", "search",
                "--train-cases", "2",
                "--eval-cases", "1",
                "--budget", "4",
                "--json", str(out_path),
            ]
        )
        assert rc == 0
        import json

        assert "table" in json.loads(out_path.read_text())

    def test_deliver_torch_artifact(self):
        # the product loop end to end: play -> lower -> verified module
        pytest.importorskip("torch")
        cases = play._torch_cases(0, 2)
        res = play.deliver("torch", cases[0], ma.ScriptedPlayer(), budget=6)
        assert res["delivered"] and res["verified"]
        assert res["cert"] is True
        assert res["max_abs"] == 0.0

    def test_deliver_rejects_deliverless_domain(self):
        with pytest.raises(ValueError, match="no deliver hook"):
            play.deliver("meta", mp.gen_cases(0, 1)[0],
                         ma.ScriptedPlayer(), budget=4)

    def test_torch_spelled_silu_delivers_fused(self):
        # a module spelling x*sigmoid(x) plays to a verified silu
        # module — shipped fold law or claim, the artifact is real
        pytest.importorskip("torch")
        for s in range(8):
            cases = play._torch_cases(s, 6)
            case = next(
                (c for c in cases if "spelled_silu" in c[0]), None
            )
            if case is not None:
                break
        assert case is not None
        res = play.deliver("torch", case, ma.ScriptedPlayer(), budget=10)
        assert res["delivered"] and res["verified"]
        assert res["cost"] is not None and res["cost"] <= 1.0

    def test_deliver_no_member(self, monkeypatch):
        # extraction empty → deliver reports undelivered honestly
        pytest.importorskip("torch")
        cases = play._torch_cases(0, 1)
        board = play._torch_board(cases[0])
        monkeypatch.setattr(
            board.eg, "extract_best", lambda *a, **k: None
        )
        res = play._torch_deliver(board, cases[0],
                                  engine.Trajectory([]))
        assert res == {"delivered": False, "reward": 0.0}

    def test_main_deliver_flag(self, capsys):
        pytest.importorskip("torch")
        rc = play.main(
            [
                "--domain", "torch",
                "--train-cases", "1",
                "--eval-cases", "1",
                "--budget", "6",
                "--deliver",
            ]
        )
        assert rc == 0
        assert "deliver[" in capsys.readouterr().out

    def test_scripted_search_drains_at_all_played(self):
        # every rule played while the board lives → the playbook
        # stops, no dead moves (unreachable under the horizon
        # without a fabricated state — the horizon ends first)
        go = play._scripted_search()
        st = play._SearchState(
            actions=("a", "b"), ops=frozenset({"x"}),
            steps=0, done=False, cost=1.0, baseline=1.0,
        )
        assert go(st) in ("a", "b")
        assert go(st) in ("a", "b")
        assert go(st) is None

    def test_main_bumps_recursion_limit(self, capsys, monkeypatch):
        import sys

        bumped = []
        monkeypatch.setattr(sys, "getrecursionlimit", lambda: 100)
        monkeypatch.setattr(
            sys, "setrecursionlimit", lambda n: bumped.append(n)
        )
        rc = play.main(
            [
                "--domain", "search",
                "--train-cases", "1",
                "--eval-cases", "1",
                "--budget", "4",
            ]
        )
        assert rc == 0 and bumped == [40_000]
        capsys.readouterr()
