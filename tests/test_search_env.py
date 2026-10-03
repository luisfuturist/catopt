"""The search environment (:mod:`catopt_core.search_env`)."""

import pytest
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.search_env import SearchEnv, StepResult


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain():
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    return Op.make("matmul", a, Op.make("matmul", b, c))


class _NoExtract:
    """An e-graph whose root e-class yields no extractable term."""

    def add_term(self, term):
        return 0

    def apply_rule(self, rule, root):
        return True

    def extract_best(self, root, cost_fn):
        return None


def test_reset_state_and_actions():
    env = SearchEnv(_chain(), all_rules())
    f = env.reset()
    assert f.operations == 2
    assert env.cost == 180.0
    assert env.progress == 0.0
    assert len(env.action_names) == len(all_rules())
    assert "assoc_matmul" in env.action_names


def test_step_improves_and_rewards():
    env = SearchEnv(_chain(), all_rules(), horizon=6, patience=2)
    env.reset()
    r = env.step("assoc_matmul")
    assert isinstance(r, StepResult)
    assert r.cost == 128.0
    assert r.reward > 0.0
    assert not r.done
    assert env.progress > 0.0


def test_reward_is_the_improvement_of_that_step():
    """The reward is aligned to the step that produced it."""
    env = SearchEnv(_chain(), all_rules(), horizon=6, patience=5)
    env.reset()
    before = env.cost
    r = env.step("assoc_matmul")
    assert r.reward == (before - r.cost) / before
    assert env.cost == r.cost


def test_patience_stops_the_episode():
    env = SearchEnv(_chain(), all_rules(), horizon=10, patience=1)
    env.reset()
    assert not env.step("assoc_matmul").done  # improves
    assert env.step("assoc_matmul").done  # stalls


def test_horizon_stops_the_episode():
    env = SearchEnv(_chain(), all_rules(), horizon=1, patience=5)
    env.reset()
    assert env.step("assoc_matmul").done


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        SearchEnv(_chain(), all_rules()).step("assoc_matmul")


def test_no_extraction_gives_infinite_cost_and_zero_reward():
    env = SearchEnv(_chain(), all_rules(), horizon=2, patience=2)
    env.reset()
    env.eg = _NoExtract()  # swap in an unextractable e-graph
    env._cost = float("inf")  # and an infinite starting cost
    r = env.step("assoc_matmul")
    assert r.cost == float("inf")
    assert r.reward == 0.0
