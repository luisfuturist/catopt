"""Property tests for the evaluation dimension (ADR 0003)."""

from catopt_core.features import compute_features
from catopt_core.game import Action
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.pareto import CostVector, dominates, pareto_frontier
from catopt_core.policies import GreedyPolicy
from catopt_core.trajectories import RULE_VECTOR_LEN, rule_vector
from hypothesis import given, settings
from hypothesis import strategies as st


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain(n: int) -> Op:
    term: object = _v("x", 2, 2)
    for _ in range(n):
        term = Op.make("matmul", term, _v("w", 2, 2))
    return term


@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=1, max_value=6))
def test_chain_depth_and_operation_count(n):
    f = compute_features(_chain(n))
    assert f.operations == n
    assert f.depth == n - 1
    assert f.parallelism == n / (n - 1) if n > 1 else 1.0


@settings(max_examples=25, deadline=None)
@given(st.integers(min_value=2, max_value=8))
def test_pareto_frontier_is_non_dominated(n):
    items = [
        CostVector(("a", "b"), (float(i), float(n - i)))
        for i in range(n + 1)
    ]
    front = pareto_frontier(items, key=lambda c: c)
    assert front
    assert all(not any(dominates(y, x) for y in front) for x in front)


def test_rule_vector_is_stable_and_sized():
    for rule in all_rules():
        v = rule_vector(rule)
        assert len(v) == RULE_VECTOR_LEN
        assert rule_vector(rule) == v


def test_greedy_policy_returns_a_member():
    acts = [Action(f"r{k}") for k in range(4)]
    assert GreedyPolicy().choose(None, acts) in acts
