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


#: Input width: program features plus a structural rule vector.
_INPUT_DIM = len(DIMENSIONS) + RULE_VECTOR_LEN


class RuleValueNet(nn.Module):
    """Predicts a rule's cost improvement from features + rule shape.

    Inputs are standardised by buffers fitted in
    :func:`train_rule_value` — raw feature scales (thousands of FLOPs
    next to 0/1 flags) otherwise starve the MLP.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, hidden: int = 32) -> None:
        """Build the MLP over ``DIMENSIONS`` + rule-vector inputs."""
        super().__init__()
        self.register_buffer("mean", torch.zeros(_INPUT_DIM))
        self.register_buffer("std", torch.ones(_INPUT_DIM))
        self.net = nn.Sequential(
            nn.Linear(_INPUT_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the predicted improvement logit, shape ``[batch]``."""
        z = (x - self.mean) / self.std
        return self.net(z).squeeze(-1)


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
        # The structural rule vectors are static, so compute them once:
        # the engine consults ``choose`` O(n_rules) times per iteration,
        # and rebuilding every vector per call would put pure-Python
        # encoding work on the search's critical path.
        self._rule_vecs = {
            name: rule_vector(rule)
            for name, rule in self.rule_by_name.items()
        }

    def features_of(self, state: Any) -> ProgramFeatures:
        """Return the program features the policy scores against.

        The base policy reads them straight off the state
        (``state.features``); a subclass that is consulted by the
        engine — whose ``GameState`` carries no features — overrides
        this to derive them (see
        :class:`catopt_torch.graph_policy.GraphFeaturePolicy`).
        """
        return state.features

    def score(self, features: ProgramFeatures, action: Action) -> float:
        """Predict the improvement logit of ``action`` for ``features``."""
        x = torch.tensor(
            [input_vector(features, self._rule_vecs[action.rule])],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            return float(self.model(x).item())

    def scores(
        self, features: ProgramFeatures, actions: Sequence[Action]
    ) -> list[float]:
        """Predict every action's logit in ONE forward pass.

        The engine consults ``choose`` once per rule per iteration, so
        scoring action-by-action would pay a device round-trip (and a
        host sync) for each — on a GPU host that dominates the search
        it is meant to speed up.  Batching the offered set into a
        single forward keeps the decision cost off the critical path.
        """
        acts = list(actions)
        x = torch.tensor(
            [
                input_vector(features, self._rule_vecs[a.rule])
                for a in acts
            ],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            out = self.model(x)
        return [float(v) for v in out]

    def choose(self, state: Any, actions: Sequence[Action]) -> Action:
        """Return the highest-scoring action (first wins a tie)."""
        acts = list(actions)
        vals = self.scores(self.features_of(state), acts)
        best_i = 0
        for i in range(1, len(vals)):
            if vals[i] > vals[best_i]:
                best_i = i
        return acts[best_i]


def train_rule_value(
    samples: Any,
    *,
    hidden: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    device: str | None = None,
    seed: int = 0,
    binary: bool = True,
) -> nn.Module:
    """Fit a :class:`RuleValueNet` to ``samples``; return the model.

    ``samples`` is an iterable of
    :class:`~catopt_core.trajectories.RuleSample`.  With ``binary``
    (the default) the target is ``delta_cost > 0`` and the loss is
    binary cross-entropy on the net's logit — the well-conditioned
    signal, since most rules leave the cost unchanged and raw-delta
    regression collapses to the mean.  With ``binary=False`` the target
    is the raw delta under MSE.

    The device defaults to CUDA when available, else CPU.
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    x = torch.tensor(
        [input_vector(s.features, s.rule_vector) for s in samples],
        dtype=torch.float32,
    )
    if binary:
        y = torch.tensor(
            [1.0 if s.delta_cost > 0 else 0.0 for s in samples],
            dtype=torch.float32,
        )
        loss_fn: nn.Module = nn.BCEWithLogitsLoss()
    else:
        y = torch.tensor(
            [s.delta_cost for s in samples], dtype=torch.float32
        )
        loss_fn = nn.MSELoss()
    model = RuleValueNet(hidden).to(dev)
    with torch.no_grad():
        model.mean.copy_(x.mean(0))
        model.std.copy_(x.std(0).clamp_min(1e-6))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(x.to(dev)), y.to(dev))
        loss.backward()
        opt.step()
    return model.eval()
