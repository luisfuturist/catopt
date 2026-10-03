"""Policy-gradient search policy (:mod:`catopt_torch.rl`)."""

import torch
from catopt_core.features import DIMENSIONS, compute_features
from catopt_core.game import Action, GameState
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.ports import Policy
from catopt_core.trajectories import RULE_VECTOR_LEN
from catopt_torch.rl import (
    PolicyNet,
    RLPolicy,
    state_vector,
    train_reinforce,
)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain():
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    return Op.make("matmul", a, Op.make("matmul", b, c))


def test_state_vector_layout():
    v = state_vector(compute_features(_chain()), 0.5)
    assert len(v) == len(DIMENSIONS) + 1
    assert v[-1] == 0.5


def test_policy_net_forward_shape():
    net = PolicyNet(hidden=8)
    sv = torch.zeros(len(DIMENSIONS) + 1)
    rv = torch.zeros(5, RULE_VECTOR_LEN)
    assert net(sv, rv).shape == (5,)


def test_rl_policy_conforms_and_chooses():
    rules = all_rules()
    pol = RLPolicy(
        PolicyNet(hidden=8), {r.name: r for r in rules}, device="cpu"
    )
    assert isinstance(pol, Policy)
    f = compute_features(_chain())
    acts = [Action(r.name) for r in rules]
    assert pol.choose(GameState(None, 0, features=f), acts).rule in {
        r.name for r in rules
    }
    assert pol.logits(f, 0.0, acts).shape == (len(rules),)


def test_train_reinforce_runs_on_cpu():
    model = train_reinforce(
        [_chain()],
        all_rules(),
        episodes=5,
        horizon=3,
        hidden=8,
        device="cpu",
    )
    assert isinstance(model, PolicyNet)


def test_train_reinforce_default_device_and_logging(capsys):
    train_reinforce(
        [_chain()],
        all_rules(),
        episodes=2,
        horizon=2,
        hidden=8,
        log_every=1,
    )
    assert "episode" in capsys.readouterr().out
