"""Learned search policy (:mod:`catopt_torch.learned_policy`)."""

import torch
from catopt_core.features import DIMENSIONS, compute_features
from catopt_core.game import Action, GameState
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.ports import Policy
from catopt_core.trajectories import RULE_VECTOR_LEN, rule_samples
from catopt_torch.learned_policy import (
    LearnedPolicy,
    RuleValueNet,
    input_vector,
    train_rule_value,
)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain():
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    return Op.make("matmul", Op.make("matmul", a, b), c)


def test_input_vector_layout():
    f = compute_features(_chain())
    v = input_vector(f, (0.0,) * RULE_VECTOR_LEN)
    assert len(v) == len(DIMENSIONS) + RULE_VECTOR_LEN


def test_net_forward_shape():
    net = RuleValueNet(hidden=8)
    x = torch.zeros(3, len(DIMENSIONS) + RULE_VECTOR_LEN)
    assert net(x).shape == (3,)


def test_train_choose_and_score():
    rules = all_rules()
    samples = rule_samples(_chain(), rules)
    model = train_rule_value(samples, epochs=5, device="cpu")
    by_name = {r.name: r for r in rules}
    pol = LearnedPolicy(model, by_name, device="cpu")
    assert isinstance(pol, Policy)
    st = GameState(None, 0, features=samples[0].features)
    acts = [Action(r.name) for r in rules]
    assert pol.choose(st, acts).rule in by_name
    assert isinstance(pol.score(samples[0].features, acts[0]), float)


def test_train_default_device():
    samples = rule_samples(_chain(), all_rules())
    model = train_rule_value(samples, epochs=1)
    assert isinstance(model, RuleValueNet)


def test_train_regression_head():
    samples = rule_samples(_chain(), all_rules())
    model = train_rule_value(
        samples, epochs=1, binary=False, device="cpu"
    )
    assert isinstance(model, RuleValueNet)
