"""A learned search policy — rule value from features (torch).

Plan 0016 stage 7.  A small MLP maps (program features ⊕ structural
rule vector) to a predicted cost improvement; the policy picks the
highest-scoring legal action.  Trained by supervised learning on
:class:`catopt_core.trajectories.RuleSample` data (pretraining), then
fine-tuned on real small models (post-training).

The action encoding is **structural** (see
:func:`catopt_core.trajectories.rule_vector`), so a rule the model has
never seen is scored by its shape — no retraining required to add
runtime rules (the plan's out-of-distribution gate).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from catopt_core.features import DIMENSIONS, ProgramFeatures
from catopt_core.game import Action
from catopt_core.trajectories import RULE_VECTOR_LEN, rule_vector
from torch import nn

__all__ = [
    "LearnedPolicy",
    "RuleValueNet",
    "input_vector",
    "train_rule_value",
]


def input_vector(
    features: ProgramFeatures, vec: Sequence[float]
) -> list[float]:
    """Concatenate program features with a rule vector."""
    return list(features.to_vector()) + list(vec)


class RuleValueNet(nn.Module):
    """Predicts a rule's cost improvement from features + rule shape."""

    def __init__(self, hidden: int = 32) -> None:
        """Build the MLP over ``DIMENSIONS`` + rule-vector inputs."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(DIMENSIONS) + RULE_VECTOR_LEN, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted improvement, shape ``[batch]``."""
        return self.net(x).squeeze(-1)


class LearnedPolicy:
    """A :class:`~catopt_core.ports.Policy` backed by a rule-value net.

    ``state`` must carry ``features`` (a
    :class:`~catopt_core.features.ProgramFeatures`); the policy scores
    each action's rule structurally and returns the best.
    """

    name = "learned"

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

    def score(self, features: ProgramFeatures, action: Action) -> float:
        """Predict the improvement of ``action`` for ``features``."""
        rule = self.rule_by_name[action.rule]
        x = torch.tensor(
            [input_vector(features, rule_vector(rule))],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            return float(self.model(x).item())

    def choose(self, state: Any, actions: Sequence[Action]) -> Action:
        """Return the highest-scoring action."""
        acts = list(actions)
        best_i = 0
        best_s: float | None = None
        for i, a in enumerate(acts):
            s = self.score(state.features, a)
            if best_s is None or s > best_s:
                best_s, best_i = s, i
        return acts[best_i]


def train_rule_value(
    samples: Any,
    *,
    hidden: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    device: str | None = None,
    seed: int = 0,
) -> nn.Module:
    """Fit a :class:`RuleValueNet` to ``samples``; return the model.

    ``samples`` is an iterable of
    :class:`~catopt_core.trajectories.RuleSample`.  The device defaults
    to CUDA when available, else CPU.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = torch.tensor(
        [input_vector(s.features, s.rule_vector) for s in samples],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [s.delta_cost for s in samples], dtype=torch.float32
    )
    model = RuleValueNet(hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x.to(dev)), y.to(dev))
        loss.backward()
        opt.step()
    return model.eval()
