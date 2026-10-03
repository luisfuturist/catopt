"""A policy-gradient search policy — REINFORCE over the search env.

Plan 0016 stage 7, the RL half (ADR 0003).  Where
:mod:`catopt_torch.learned_policy` is *supervised* on one-step
trajectories, this is a full RL player: it samples a rule, applies it
to the e-graph, and is updated from the reward that step produced
(REINFORCE with a per-episode standardized advantage).

The action encoding is structural (``rule_vector``), so an unseen rule
is still scorable.  The policy only *orders* legal moves — the
certificate still decides correctness (ADR 0003 invariant 5).
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from catopt_core.features import DIMENSIONS, ProgramFeatures
from catopt_core.game import Action
from catopt_core.search_env import SearchEnv
from catopt_core.trajectories import RULE_VECTOR_LEN, rule_vector
from torch import nn
from torch.distributions import Categorical

__all__ = ["PolicyNet", "RLPolicy", "state_vector", "train_reinforce"]

logger = logging.getLogger(__name__)

#: State width: the feature vector plus the episode progress.
_STATE_DIM = len(DIMENSIONS) + 1

#: Returns whose spread is below this carry no advantage signal.
_EPS = 1e-12


def state_vector(
    features: ProgramFeatures, progress: float
) -> list[float]:
    """Concatenate program features with the episode progress."""
    return [*features.to_vector(), progress]


class PolicyNet(nn.Module):
    """Scores each rule for a state: ``(state ⊕ rule) -> logit``.

    Scoring per rule (rather than a fixed softmax head) is what keeps
    the action space open: a new rule is a new point in the same
    structural space.
    """

    def __init__(self, hidden: int = 64) -> None:
        """Build the MLP over ``(state ⊕ rule)`` inputs."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(_STATE_DIM + RULE_VECTOR_LEN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, state: torch.Tensor, rule_vecs: torch.Tensor
    ) -> torch.Tensor:
        """Return one logit per rule, shape ``[n_rules]``."""
        n = rule_vecs.shape[0]
        x = torch.cat(
            [state.unsqueeze(0).expand(n, -1), rule_vecs], dim=1
        )
        return self.net(x).squeeze(-1)


class RLPolicy:
    """A :class:`~catopt_core.ports.Policy` backed by :class:`PolicyNet`."""

    name = "rl"

    def __init__(
        self,
        model: nn.Module,
        rule_by_name: Any,
        device: str = "cpu",
    ) -> None:
        """Wrap a trained ``model`` and a rule-name index."""
        self.model = model.to(device).eval()
        self.rule_by_name = dict(rule_by_name)
        self.device = device

    def logits(
        self, features: ProgramFeatures, progress: float, actions: Any
    ) -> torch.Tensor:
        """Return the model's logits for ``actions`` at this state."""
        acts = list(actions)
        rv = torch.tensor(
            [rule_vector(self.rule_by_name[a.rule]) for a in acts],
            dtype=torch.float32,
            device=self.device,
        )
        sv = torch.tensor(
            state_vector(features, progress),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            return self.model(sv, rv)

    def choose(self, state: Any, actions: Any) -> Action:
        """Return the highest-scoring action (greedy at inference)."""
        acts = list(actions)
        progress = float(getattr(state, "progress", 0.0) or 0.0)
        idx = int(
            torch.argmax(self.logits(state.features, progress, acts))
        )
        return acts[idx]


def _rollout(
    model: nn.Module,
    env: SearchEnv,
    rule_vecs: torch.Tensor,
    names: list[str],
    dev: str,
    gamma: float,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Sample one episode; return its log-probs and returns-to-go."""
    feats = env.reset()
    logps: list[torch.Tensor] = []
    rews: list[float] = []
    for _ in range(env.horizon):
        sv = torch.tensor(
            state_vector(feats, env.progress),
            dtype=torch.float32,
            device=dev,
        )
        dist = Categorical(logits=model(sv, rule_vecs))
        idx = dist.sample()
        logps.append(dist.log_prob(idx))
        out = env.step(names[int(idx.item())])
        rews.append(out.reward)
        feats = out.features
        if out.done:
            break
    running, returns = 0.0, []
    for r in reversed(rews):
        running = r + gamma * running
        returns.append(running)
    returns.reverse()
    return logps, torch.tensor(returns, dtype=torch.float32, device=dev)


def _advantage(returns: torch.Tensor) -> torch.Tensor:
    """Return the scale-free advantage of one episode's returns.

    REINFORCE needs a baseline, and a single *running mean* over a
    mixture of program families is dominated by the family with the
    largest normalized reward — the env's reward is
    ``(before - after) / before``, so a chain pays ~0.1 and a fused
    linear ~0.75 — which collapses the policy onto that family.
    Centering and scaling each episode's returns instead makes the
    advantage comparable across families with **no family label**:
    within an episode the best action always scores positive and the
    worst negative, whatever the family's reward scale.  An episode
    with no spread (nothing improved) yields a zero advantage — no
    signal, rather than a spurious one.
    """
    centered = returns - float(returns.mean())
    std = float(returns.std(unbiased=False))
    if std <= _EPS:
        return centered
    return centered / std


def train_reinforce(
    programs: Any,
    rules: Any,
    *,
    episodes: int = 400,
    horizon: int = 6,
    patience: int = 2,
    hidden: int = 64,
    lr: float = 3e-3,
    gamma: float = 0.95,
    device: str | None = None,
    seed: int = 0,
    log_every: int = 0,
) -> nn.Module:
    """Train a :class:`PolicyNet` by REINFORCE; return the model.

    ``programs`` is a cycle of starting programs; each episode samples
    rules, applies them, and updates on the discounted reward with a
    per-episode standardized advantage (see :func:`_advantage`), so the
    advantage is comparable across program families rather than
    dominated by the loudest one.  The device defaults to CUDA when
    available.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = PolicyNet(hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    names = [r.name for r in rules]
    rule_vecs = torch.tensor(
        [rule_vector(r) for r in rules],
        dtype=torch.float32,
        device=dev,
    )
    for ep in range(episodes):
        env = SearchEnv(
            programs[ep % len(programs)],
            rules,
            horizon=horizon,
            patience=patience,
        )
        logps, returns = _rollout(
            model, env, rule_vecs, names, dev, gamma
        )
        adv = _advantage(returns)
        loss = -(torch.stack(logps) * adv).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (ep + 1) % log_every == 0:
            logger.info(
                "episode %d: mean return %.4f",
                ep + 1,
                float(returns.mean()),
            )
    return model.eval()
