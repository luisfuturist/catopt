"""Policy-gradient search policy (:mod:`catopt_torch.rl`)."""

import logging

import catopt_torch.rl as rl
import torch
from catopt_core.features import DIMENSIONS, compute_features
from catopt_core.game import Action, GameState
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.ports import Policy
from catopt_core.search_env import StepResult
from catopt_core.trajectories import RULE_VECTOR_LEN, rule_vector
from catopt_torch.rl import (
    PolicyNet,
    RLPolicy,
    _advantage,
    _rollout,
    state_vector,
    train_reinforce,
)
from torch.distributions import Categorical


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


def test_train_reinforce_default_device_and_logging(caplog):
    with caplog.at_level(logging.INFO, logger="catopt_torch.rl"):
        train_reinforce(
            [_chain()],
            all_rules(),
            episodes=2,
            horizon=2,
            hidden=8,
            log_every=1,
        )
    assert any("episode" in r.getMessage() for r in caplog.records)


def test_advantage_standardizes_within_episode():
    """The best step scores positive, the rest negative, mean 0."""
    adv = _advantage(torch.tensor([0.4, 0.0, 0.0]))
    assert abs(float(adv.mean())) < 1e-6
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-5
    assert float(adv[0]) > 0.0
    assert float(adv[1]) < 0.0


def test_advantage_is_scale_invariant():
    """A family's reward scale cannot dominate the advantage."""
    r = torch.tensor([0.1, 0.0, 0.0])
    assert torch.allclose(_advantage(r), _advantage(r * 7.0))


def test_advantage_flat_returns_have_no_signal():
    """An episode with no improvement yields a zero advantage."""
    adv = _advantage(torch.zeros(3))
    assert torch.allclose(adv, torch.zeros(3))


# ---------------------------------------------------------------------------
#  Step alignment — the log-probs and the returns must pair per step
# ---------------------------------------------------------------------------


class _ScriptedEnv:
    """A stand-in ``SearchEnv`` with a fixed, hand-checkable script.

    ``_rollout`` only reads ``reset`` / ``horizon`` / ``progress`` /
    ``step``.  Scripting the rewards makes the return-to-go computable
    by hand, and recording ``(progress, name)`` lets each step's
    log-prob be recomputed exactly — so the log-prob ↔ return pairing
    is checkable per step.
    """

    def __init__(
        self,
        rewards: list[float],
        horizon: int,
        stop_after: int | None = None,
    ) -> None:
        self.rewards = list(rewards)
        self.horizon = horizon
        self.stop_after = stop_after
        self._i = 0
        self.seen: list[tuple[float, str]] = []
        self._feats = compute_features(_chain())

    @property
    def progress(self) -> float:
        """The horizon fraction consumed *before* the next step."""
        return self._i / self.horizon

    def reset(self):
        """Restart the script; return the (constant) state."""
        self._i = 0
        self.seen = []
        return self._feats

    def step(self, name: str) -> StepResult:
        """Record ``(progress, name)``; return the next scripted step."""
        self.seen.append((self.progress, name))
        reward = self.rewards[self._i]
        self._i += 1
        stop = self.horizon if self.stop_after is None else self.stop_after
        return StepResult(self._feats, reward, self._i >= stop, 0.0)


def _rule_tensors():
    """Return ``(names, rule_vecs)`` for the full rule set."""
    rules = all_rules()
    names = [r.name for r in rules]
    rule_vecs = torch.tensor(
        [rule_vector(r) for r in rules], dtype=torch.float32
    )
    return names, rule_vecs


def _returns_to_go(rewards: list[float], gamma: float) -> torch.Tensor:
    """Replicate ``_rollout``'s reversed-returns construction."""
    running, out = 0.0, []
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def test_rollout_empty_horizon_yields_no_steps():
    """A zero-horizon episode yields empty, still-aligned sequences."""
    names, rule_vecs = _rule_tensors()
    env = _ScriptedEnv([], horizon=0)
    logps, returns = _rollout(
        PolicyNet(hidden=8), env, rule_vecs, names, "cpu", 0.9
    )
    assert logps == []
    assert returns.numel() == 0


def test_rollout_returns_are_the_discounted_return_to_go():
    """``returns[t]`` is ``sum_{k>=t} gamma^(k-t) r_k``, in step order."""
    rewards = [0.5, -2.0, 3.0, 1.0]
    gamma = 0.5
    names, rule_vecs = _rule_tensors()
    env = _ScriptedEnv(rewards, horizon=len(rewards))
    logps, returns = _rollout(
        PolicyNet(hidden=8), env, rule_vecs, names, "cpu", gamma
    )
    expected = [
        sum(gamma ** (k - i) * rewards[k] for k in range(i, len(rewards)))
        for i in range(len(rewards))
    ]
    assert len(logps) == len(returns) == len(rewards)
    assert torch.allclose(returns, torch.tensor(expected))


def test_rollout_early_done_keeps_logps_and_returns_in_lockstep():
    """An early ``done`` break must not desynchronise the two lists."""
    rewards = [0.5, -2.0, 9.9, 9.9, 9.9]
    names, rule_vecs = _rule_tensors()
    env = _ScriptedEnv(rewards, horizon=5, stop_after=2)
    logps, returns = _rollout(
        PolicyNet(hidden=8), env, rule_vecs, names, "cpu", 0.9
    )
    assert len(logps) == len(returns) == len(env.seen) == 2
    assert torch.allclose(
        returns, torch.tensor([0.5 + 0.9 * -2.0, -2.0])
    )


def test_rollout_logps_pair_with_their_own_step():
    """``logps[t]`` is the log-prob of the action taken at step ``t``."""
    rewards = [0.5, -2.0, 3.0]
    names, rule_vecs = _rule_tensors()
    model = PolicyNet(hidden=8)
    torch.manual_seed(0)
    env = _ScriptedEnv(rewards, horizon=len(rewards))
    logps, _ = _rollout(model, env, rule_vecs, names, "cpu", 0.9)
    feats = compute_features(_chain())
    assert len(logps) == len(env.seen) == len(rewards)
    for i, (progress, name) in enumerate(env.seen):
        sv = torch.tensor(
            state_vector(feats, progress), dtype=torch.float32
        )
        with torch.no_grad():
            logits = model(sv, rule_vecs)
        expected = Categorical(logits=logits).log_prob(
            torch.tensor(names.index(name))
        )
        assert torch.allclose(logps[i], expected)


class _SpyOptim:
    """An optimizer that records the gradient instead of stepping."""

    def __init__(self, params, **_kw) -> None:
        self.params = list(params)
        self.grads: list[torch.Tensor] = []

    def zero_grad(self) -> None:
        """Drop the previous gradients."""
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        """Clone the gradients instead of updating the parameters."""
        self.grads = [p.grad.detach().clone() for p in self.params]


def test_train_reinforce_gradient_is_the_explicitly_paired_sum(monkeypatch):
    """The shipped update equals ``-sum_t logp_t * A_t`` for its steps.

    A step-major / episode-major mismatch — or any permutation of the
    log-probs relative to the returns — would change these gradients,
    so pinning them makes the alignment load-bearing, not incidental.
    """
    rewards = [0.6, -1.5, 0.2]
    horizon, hidden, seed, gamma = len(rewards), 4, 7, 0.9
    holder: dict[str, _ScriptedEnv] = {}
    spies: list[_SpyOptim] = []

    def _env_factory(*_a, **_k):
        holder["env"] = _ScriptedEnv(rewards, horizon)
        return holder["env"]

    def _opt_factory(params, **kw):
        spies.append(_SpyOptim(params, **kw))
        return spies[-1]

    monkeypatch.setattr(rl, "SearchEnv", _env_factory)
    monkeypatch.setattr(torch.optim, "Adam", _opt_factory)

    rules = all_rules()
    train_reinforce(
        [_chain()],
        rules,
        episodes=1,
        horizon=horizon,
        hidden=hidden,
        lr=1e-2,
        gamma=gamma,
        device="cpu",
        seed=seed,
    )

    # Rebuild the initial net (same seed -> same parameters) and replay
    # the *recorded* actions to get the expected per-step log-probs.
    torch.manual_seed(seed)
    model = PolicyNet(hidden)
    names, rule_vecs = _rule_tensors()
    feats = compute_features(_chain())
    logps = [
        Categorical(
            logits=model(
                torch.tensor(
                    state_vector(feats, progress), dtype=torch.float32
                ),
                rule_vecs,
            )
        ).log_prob(torch.tensor(names.index(name)))
        for progress, name in holder["env"].seen
    ]
    adv = _advantage(_returns_to_go(rewards, gamma))
    loss = -(torch.stack(logps) * adv).sum()
    loss.backward()

    expected = [p.grad for p in model.parameters()]
    got = spies[0].grads
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        assert torch.allclose(g, e, atol=1e-6)
