"""Players for the generic optimization game.

:class:`LinearPolicy` is the reusable learner: a softmax linear
policy over whatever ``legal(state)`` enumerates, scored by an
injected ``featurizer`` — the same architecture both arena players
use, factored so a new domain is a featurizer and a schema, never a
new policy class.  Training is episode REINFORCE with reward-to-go
credit (a move gets the terminal payout it armed), an EMA baseline
and clipped advantage; *epsilon* mixes uniform exploration into the
sample for sparse-reward boards.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any

__all__ = ["LinearPolicy"]

#: REINFORCE constants — per-episode-return update normalized by
#: step count, EMA running baseline, clipped advantage.
_LR = 0.1
_BASELINE_EMA = 0.2
_CLIP = 40.0


def _dot(w: dict, feats: dict) -> float:
    """Return the linear score ``w·feats`` over a sparse feature dict."""
    s = 0.0
    for k, v in feats.items():
        wv = w.get(k)
        if wv:
            s += wv * v
    return s


def _action_key(action: Any) -> tuple:
    """Stable dedup key — the played-move mask, replayable."""
    params = getattr(action, "params", None)
    if params is None:
        return ("call", repr(action))
    return (
        getattr(action, "op", "?"),
        repr(sorted(params.items(), key=lambda kv: kv[0])),
    )


def _grad_row(
    feats: list[dict], exps: list[float], i: int
) -> tuple[dict, dict]:
    """``(φ_chosen, E_π[φ])`` under the softmax over unplayed moves."""
    z = sum(exps)
    ephi: dict[str, float] = {}
    for f, e in zip(feats, exps, strict=True):
        for k, v in f.items():
            ephi[k] = ephi.get(k, 0.0) + v * e / z
    return feats[i], ephi


class LinearPolicy:
    """A softmax linear policy over a board's legal moves.

    ``score(move) = w·φ(state, move)`` where φ comes from the
    injected *featurizer* ``(state, action, hist) -> dict`` and *w*
    is a ``{feature_name: float}`` table over the *features* schema.
    A step samples the softmax over the *unplayed* legal set
    (``greedy`` → argmax; *temperature* scales logits; *epsilon*
    mixes uniform exploration).  ``learn=False`` freezes the policy
    for eval; ``frozen`` returns a non-learning copy sharing the
    trained weights.

    Episode boundary detection: a fresh state (``steps == 0`` after
    a progressed one) clears the played mask and the history — the
    domain's states must count steps, which both arenas do.
    """

    #: The domain's feature schema — a class attribute a preset
    #: binds (``MetaLearnedPlayer``); ``weights_dict`` reads it for
    #: the full named table.
    FEATURES: Iterable[str] = ()

    def __init__(
        self,
        seed: int = 0,
        *,
        legal: Any,
        featurizer: Any,
        weights: dict | None = None,
        lr: float = _LR,
        temperature: float = 1.0,
        greedy: bool = False,
        learn: bool = True,
        epsilon: float = 0.0,
        features: Iterable[str] | None = None,
    ) -> None:
        """Bind the enumerator, the featurizer, and the sampler."""
        self._rng = random.Random(seed)
        self._legal = legal
        self._featurizer = featurizer
        if features is not None:
            self.FEATURES = tuple(features)
        self._lr = lr
        self._temp = max(temperature, 1e-6)
        self._greedy = greedy
        self._learn = learn
        self._eps = epsilon
        self._w = dict(weights or {})
        self._baseline = 0.0
        self._played: set = set()
        self._hist: Any = None
        self._ep: list = []
        self._prev_steps = -1

    def fresh_hist(self) -> Any:
        """Return the domain's empty history object (overridable)."""
        return None

    def frozen(self, seed: int = 0, *, greedy: bool = False) -> Any:
        """Return a non-learning copy sharing the trained weights."""
        q = type(self)(
            seed=seed,
            legal=self._legal,
            featurizer=self._featurizer,
            weights=self.weights_dict(),
            temperature=self._temp,
            greedy=greedy,
            learn=False,
            epsilon=0.0,
        )
        q.FEATURES = self.FEATURES
        return q

    def weights_dict(self) -> dict[str, float]:
        """Return the weight table — ``{name: w}`` over the schema."""
        return {n: self._w.get(n, 0.0) for n in self.FEATURES}

    def _observe(self, state: Any) -> None:
        """Detect the episode boundary — a fresh board resets state."""
        if self._hist is None or (
            state.steps == 0 and self._prev_steps > 0
        ):
            self._played.clear()
            self._hist = self.fresh_hist()
        self._prev_steps = state.steps

    def __call__(self, state: Any) -> Any | None:
        """Score the unplayed legal moves, sample one, remember it."""
        self._observe(state)
        acts = [
            a
            for a in self._legal(state)
            if _action_key(a) not in self._played
        ]
        if not acts:
            return None
        feats = [self._featurizer(state, a, self._hist) for a in acts]
        logits = [_dot(self._w, f) / self._temp for f in feats]
        top = max(logits)
        exps = [math.exp(x - top) for x in logits]
        if self._greedy:
            i = logits.index(top)
        elif self._eps and self._rng.random() < self._eps:
            i = self._rng.randrange(len(acts))
        else:
            i = self._sample(exps)
        if self._learn:
            self._ep.append(_grad_row(feats, exps, i))
        chosen = acts[i]
        self._played.add(_action_key(chosen))
        self._record(chosen)
        return chosen

    def _record(self, action: Any) -> None:
        """Post-play bookkeeping — the domain's history update hook."""
        rec = getattr(self._hist, "record", None)
        if rec is not None:
            rec(getattr(action, "op", "?"))

    def _sample(self, exps: list[float]) -> int:
        """Draw an index proportionally to the unnormalized weights."""
        r = self._rng.random() * sum(exps)
        acc = 0.0
        for i, e in enumerate(exps):
            acc += e
            if r <= acc:
                return i
        return len(exps) - 1

    def finish_episode(
        self, total: float, rewards: Iterable[float] | None = None
    ) -> None:
        """REINFORCE — reward-to-go per step, not just the total.

        Per step ``t``: ``G_t = Σ_{k≥t} r_k`` — the remaining episode
        reward, so a move gets credit for the terminal payout it
        armed.  *rewards* comes from the trajectory's per-step
        reports; ``None`` falls back to the flat return.  The EMA
        baseline tracks the raw total; advantages are clipped.  No-op
        when ``learn=False``.
        """
        adv = total - self._baseline
        self._baseline += _BASELINE_EMA * adv
        if not (self._learn and self._ep):
            self._ep = []
            return
        rs = list(rewards) if rewards is not None else None
        n = len(self._ep)
        grad: dict[str, float] = {}
        for t, (phi, ephi) in enumerate(self._ep):
            g_t = (
                sum(rs[t:]) if rs is not None and t < len(rs) else total
            )
            a = max(-_CLIP, min(_CLIP, g_t - self._baseline))
            for k, v in phi.items():
                grad[k] = grad.get(k, 0.0) + self._lr * a / n * v
            for k, v in ephi.items():
                grad[k] = grad.get(k, 0.0) - self._lr * a / n * v
        for k, g in grad.items():
            self._w[k] = self._w.get(k, 0.0) + g
        self._ep = []
