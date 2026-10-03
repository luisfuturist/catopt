"""AlphaZero-style contraction ordering: PUCT + a learned value head.

Plan 0016 follow-up.  ``contraction-policy-throughput.md`` removed the
feature-construction bottleneck (4.3x cheaper at n = 40) and showed the
learned policy is no longer compute-starved — yet it still loses to
``opt_einsum``'s randomised greedy at n = 40 by ~1.8x, and the loss is
**structural**: quality *saturates* with rollout count (20 -> 120
rollouts moves the n = 40 ratio only 1.9 -> 1.8).  More compute cannot
close a quality gap, so the policy needs a **better decision**, not
more of them.

This tool is the direct response — an AlphaZero-ification of the
single-player contraction game.

* **Two heads, one trunk.**  The net keeps the existing per-pair policy
  head (a logit per ``(state (+) pair)`` input) and gains a **value
  head** ``V(s)`` predicting the normalised *remaining* cost to complete
  from ``s``.  It replaces the hand-designed analytic critic (the
  greedy-completion cost), which can know nothing the greedy heuristic
  does not.
* **Policy-guided lookahead (PUCT).**  At each decision a small MCTS
  runs over contraction states: select by
  ``-Q + c * P * sqrt(N) / (1 + n)``, expand with the policy prior,
  evaluate the leaf with ``V``, back the value up.  The tree is
  **re-rooted** on the committed child, so later decisions reuse the
  earlier search.
* **Search in the loop.**  Training is expert iteration: the net plays
  PUCT episodes, the **visit distribution** at each root is the policy
  target and the **achieved (normalised) episode cost** is the value
  target.  Search improves the net; the net improves search.

The state/action features are the ``ContractionGame`` features
(``state_features`` / ``pair_feature_matrix``) plus one appended column:
the board's ``log2(greedy_ref)`` (see :func:`_sf`).  That column exists
because the value target is normalised by ``greedy_ref``, which is *not*
otherwise observable to the net — without it the same remaining cost
gets a different label on every board and the value head cannot fit it
(``contraction-value-head.md`` measures the rank collapse: 0.51 -> 0.89
once the normaliser is observable).

Usage::

    uv sync --group einsum --group bench
    .venv/bin/python tools/contraction_az.py --mode compare --device cuda
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import contraction_einsum as ce
import contraction_policy as cp
import contraction_scale as cs
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["main"]

#: State feature width: the game's own features plus the board scale.
_STATE_DIM = cp._STATE_DIM + 1


def _sf(game: cp.ContractionGame) -> list[float]:
    """State features plus the board's ``log2`` greedy reference.

    The value target is normalised by the board's greedy reference
    (``value_unit="ref"``), which is *not* one of the game's features —
    so the net cannot observe the very scale it is asked to predict
    against, and the same remaining cost gets a different label on every
    board.  Appending ``log2(ref)`` makes the normaliser observable and
    the regression target consistent across boards, while keeping the
    target scale-free (so it generalises to larger ``n``).
    """
    ref = game.greedy_ref
    return [
        *game.state_features(),
        math.log2(ref) if ref > 0 else 0.0,
    ]


#: Pair feature width (unchanged).
_PAIR_DIM = cp._PAIR_DIM

#: PUCT exploration constant.
_C_PUCT = 1.5

#: Dirichlet noise concentration / weight for root exploration.
_NOISE_ALPHA = 0.3
_NOISE_EPS = 0.25

#: Clamp on the raw value output before ``expm1`` (an untrained head can
#: emit anything; the log-space target lives in ``[0, ~5]``).
_VALUE_CLAMP = 20.0

#: Value-target normalisation.  ``"ref"`` divides the remaining cost by
#: the *board's* greedy reference (the original choice); ``"abs"`` keeps
#: it absolute.  The board reference is unobservable to the net, so
#: ``"ref"`` makes the regression target inconsistent across boards —
#: the same remaining cost gets a different label on every board, and
#: the net cannot undo it.  The search always works in
#: ``remaining / ref`` units, so every unit is inverted back to that.
_VALUE_UNITS = ("ref", "abs")


def _invert_value(raw: float, unit: str, ref: float) -> float:
    """Invert a value-head output to the search's ``remaining / ref``."""
    if unit == "abs":
        return math.expm1(raw) / ref if ref > 0 else 0.0
    return math.expm1(raw)


#: One collected training sample: state features, pair features, the
#: search's visit distribution and the cost already paid at that state.
_Sample = tuple[list, list, list, float]


# ---------------------------------------------------------------------------
#  The two-head net
# ---------------------------------------------------------------------------


class DualHeadNet(nn.Module):
    """Shared state trunk, per-pair policy head and a scalar value head.

    The trunk consumes the (standardised) state features; the policy head
    scores each ``(trunk (+) pair)`` input — the same open action space
    the old per-pair net had — and the value head reads the trunk alone.
    State and pair columns are standardised separately, fitted from real
    greedy trajectories.
    """

    s_mean: torch.Tensor
    s_std: torch.Tensor
    p_mean: torch.Tensor
    p_std: torch.Tensor

    def __init__(self, hidden: int = 64) -> None:
        """Build the trunk plus the policy and value heads."""
        super().__init__()
        self.register_buffer("s_mean", torch.zeros(_STATE_DIM))
        self.register_buffer("s_std", torch.ones(_STATE_DIM))
        self.register_buffer("p_mean", torch.zeros(_PAIR_DIM))
        self.register_buffer("p_std", torch.ones(_PAIR_DIM))
        self.trunk = nn.Sequential(
            nn.Linear(_STATE_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(hidden + _PAIR_DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, sf: torch.Tensor, pf: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits [B, P], value [B])`` for batched states."""
        h = self.trunk((sf - self.s_mean) / self.s_std)
        b, p, _k = pf.shape
        pn = (pf - self.p_mean) / self.p_std
        x = torch.cat(
            [h.unsqueeze(1).expand(b, p, h.shape[1]), pn], dim=2
        ).reshape(b * p, -1)
        logits = self.policy_head(x).reshape(b, p)
        value = self.value_head(h).squeeze(-1)
        return logits, value


def _fit_norm_dual(
    model: DualHeadNet,
    ns: tuple[int, ...],
    rng: random.Random,
    device: str,
) -> None:
    """Fit the state and pair normalisation buffers from greedy plays."""
    states: list[list[float]] = []
    pairs: list[list[float]] = []
    for _ in range(8):
        for n0 in ns:
            tensors, sizes = ce.random_bond_network(
                n0, rng.randrange(1 << 30)
            )
            g = cp.ContractionGame(
                tensors, sizes, cs.greedy(tensors, sizes)
            )
            while not g.done:
                states.append(_sf(g))
                pairs.extend(g.all_pair_features())
                g.step(*g.pairs[0])
    s = torch.tensor(states, dtype=torch.float32)
    p = torch.tensor(pairs, dtype=torch.float32)
    with torch.no_grad():
        model.s_mean.copy_(s.mean(0))
        model.s_std.copy_(s.std(0).clamp_min(cp._STD_FLOOR))
        model.p_mean.copy_(p.mean(0))
        model.p_std.copy_(p.std(0).clamp_min(cp._STD_FLOOR))


# ---------------------------------------------------------------------------
#  PUCT over contraction states
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """One search node: a partial contraction with visit statistics."""

    game: cp.ContractionGame
    ref: float
    terminal: bool = False
    expanded: bool = False
    priors: np.ndarray | None = None
    value: float = 0.0
    n: np.ndarray | None = None
    w: np.ndarray | None = None
    children: dict[int, _Node] = field(default_factory=dict)


def _expand_many(
    model: DualHeadNet,
    nodes: list[_Node],
    device: str,
    *,
    use_net_value: bool,
    value_unit: str = "ref",
) -> None:
    """Expand a batch of nodes in one net forward.

    The net forward is launch-overhead-bound (~120 us whatever the row
    count), so scoring many nodes together is nearly free — this is what
    makes lockstep search and self-play affordable.  Terminal nodes are
    marked without a forward; the rest are padded to the widest pair
    matrix in the batch.
    """
    live = [nd for nd in nodes if not nd.game.done]
    for nd in nodes:
        if nd.game.done:
            nd.terminal = True
            nd.expanded = True
    if not live:
        return
    widths = [len(nd.game.pairs) for nd in live]
    width = max(widths)
    sf = np.stack([_sf(nd.game) for nd in live])
    pf = np.zeros((len(live), width, _PAIR_DIM), dtype=np.float64)
    for i, nd in enumerate(live):
        m = nd.game.pair_feature_matrix()
        pf[i, : m.shape[0]] = m
    with torch.no_grad():
        logits, value = model(
            torch.as_tensor(sf, dtype=torch.float32, device=device),
            torch.as_tensor(pf, dtype=torch.float32, device=device),
        )
    logits_np = logits.float().cpu().numpy()
    value_np = value.float().cpu().numpy()
    for i, nd in enumerate(live):
        k = widths[i]
        row = logits_np[i, :k]
        e = np.exp(row - row.max())
        nd.priors = e / e.sum()
        if use_net_value:
            # The value head predicts the *remaining* cost in log space
            # under ``value_unit`` (see ``train_search``); the search
            # works in ``remaining / ref`` units, so invert here.
            raw = min(
                max(float(value_np[i]), -_VALUE_CLAMP), _VALUE_CLAMP
            )
            nd.value = _invert_value(raw, value_unit, nd.ref)
        else:
            nd.value = cs.greedy(nd.game.ts, nd.game.sizes) / nd.ref
        nd.n = np.zeros(k, dtype=np.float64)
        nd.w = np.zeros(k, dtype=np.float64)
        nd.expanded = True


def _total_estimate(node: _Node) -> float:
    """Return the node's estimated normalised *total* episode cost.

    The value head predicts the remaining cost from a state; adding the
    cost already paid (which the state carries) gives the total, the unit
    the backup and the unvisited-action default work in.
    """
    return node.game.cost / node.ref + node.value


def _child(node: _Node, a: int) -> _Node:
    """Build the child reached by contracting pair ``a``."""
    g = node.game
    ia, ib = g.pairs[a]
    child = g.clone()
    child.step(ia, ib)
    return _Node(child, node.ref)


def _select(node: _Node) -> int:
    """Return the PUCT action index at an expanded, non-terminal node.

    An unvisited action's ``Q`` defaults to the node's own total-cost
    estimate rather than to zero.  In a zero-sum game zero is neutral, but
    here the value is a *cost*, so a zero default would mark every
    unvisited action as free and starve the search at the ``O(n^2)``
    action counts this game has — the prior must rank the unvisited
    children.
    """
    n = node.n
    assert (
        n is not None and node.priors is not None and node.w is not None
    )
    tot = float(n.sum())
    q = np.where(
        n > 0, node.w / np.maximum(n, 1.0), _total_estimate(node)
    )
    u = _C_PUCT * node.priors * math.sqrt(tot + 1.0) / (1.0 + n)
    return int(np.argmax(-q + u))


def _leaf_value(node: _Node) -> float:
    """Return a leaf's normalised total-cost estimate."""
    if node.terminal:
        return node.game.cost / node.ref
    return _total_estimate(node)


def _descend(root: _Node) -> tuple[list[tuple[_Node, int]], _Node]:
    """Walk PUCT from ``root`` to a fresh (unexpanded) leaf.

    Returns the edge path taken and the leaf reached; the leaf is not yet
    expanded, so a caller may batch several leaves into one net forward.
    """
    node = root
    path: list[tuple[_Node, int]] = []
    while not node.terminal:
        a = _select(node)
        if a not in node.children:
            node.children[a] = _child(node, a)
            path.append((node, a))
            return path, node.children[a]
        path.append((node, a))
        node = node.children[a]
    return path, node


def _backup(path: list[tuple[_Node, int]], value: float) -> None:
    """Add one visit carrying ``value`` to every edge on ``path``."""
    for parent, a in path:
        parent.n[a] += 1.0
        parent.w[a] += value


def _simulate_locked(
    roots: list[_Node],
    model: DualHeadNet,
    device: str,
    *,
    use_net_value: bool = True,
    value_unit: str = "ref",
) -> None:
    """Run one PUCT simulation for each root, batching the leaf evals."""
    walks = [_descend(root) for root in roots]
    _expand_many(
        model,
        [leaf for _p, leaf in walks],
        device,
        use_net_value=use_net_value,
        value_unit=value_unit,
    )
    for path, leaf in walks:
        _backup(path, _leaf_value(leaf))


def _add_noise(node: _Node, rng: random.Random) -> None:
    """Mix Dirichlet noise into an expanded root's priors (AlphaZero)."""
    assert node.priors is not None
    k = len(node.priors)
    draws = [rng.gammavariate(_NOISE_ALPHA, 1.0) for _ in range(k)]
    tot = math.fsum(draws) or 1.0
    noise = np.array([d / tot for d in draws])
    node.priors = (1.0 - _NOISE_EPS) * node.priors + _NOISE_EPS * noise


# ---------------------------------------------------------------------------
#  One PUCT episode, with tree re-rooting
# ---------------------------------------------------------------------------


def puct_episodes(
    model: DualHeadNet,
    instances: list[tuple[Any, dict[int, int], float]],
    *,
    sims: int,
    device: str,
    noise_seeds: list[int | None] | None = None,
    use_net_value: bool = True,
    value_unit: str = "ref",
    collect: bool = False,
) -> list[tuple[list[tuple[int, int]], float, list[_Sample]]]:
    """Play one PUCT episode per instance, in lockstep.

    Every instance gets its own re-rooted tree; the simulations advance
    together so all their leaf evaluations share one net forward — the
    forward is launch-overhead-bound, so this is the single biggest
    speedup available.  Each ``sims`` simulations the most-visited action
    is committed and the tree re-roots on the child, so later decisions
    reuse the earlier search.  Returns ``(order, cost, samples)`` per
    instance; ``samples`` (only when ``collect``) holds
    ``(state_features, pair_features, visit_dist)`` per decision.
    """
    roots = [
        _Node(cp.ContractionGame(tensors, sizes, ref), ref)
        for tensors, sizes, ref in instances
    ]
    _expand_many(
        model,
        roots,
        device,
        use_net_value=use_net_value,
        value_unit=value_unit,
    )
    if noise_seeds is not None:
        for root, nseed in zip(roots, noise_seeds, strict=True):
            if nseed is not None:
                _add_noise(root, random.Random(nseed))
    orders: list[list[tuple[int, int]]] = [[] for _ in roots]
    samples: list[list[_Sample]] = [[] for _ in roots]
    active = list(range(len(roots)))
    while active:
        for _ in range(sims):
            _simulate_locked(
                [roots[i] for i in active],
                model,
                device,
                use_net_value=use_net_value,
                value_unit=value_unit,
            )
        fresh: list[_Node] = []
        for i in active:
            root = roots[i]
            assert root.n is not None and root.priors is not None
            counts = root.n
            total = float(counts.sum())
            pi = counts / total if total > 0 else root.priors
            a = (
                int(np.argmax(counts))
                if total > 0
                else int(np.argmax(root.priors))
            )
            if collect:
                samples[i].append(
                    (
                        _sf(root.game),
                        root.game.pair_feature_matrix().tolist(),
                        pi.tolist(),
                        root.game.cost,
                    )
                )
            ia, ib = root.game.pairs[a]
            orders[i].append((ia, ib))
            if a in root.children:
                child = root.children[a]
            else:
                child = _child(root, a)
                fresh.append(child)
            roots[i] = child
        if fresh:
            _expand_many(
                model,
                fresh,
                device,
                use_net_value=use_net_value,
                value_unit=value_unit,
            )
        active = [i for i in active if not roots[i].terminal]
    return [
        (orders[i], roots[i].game.cost, samples[i])
        for i in range(len(roots))
    ]


def puct_episode(
    model: DualHeadNet,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    *,
    sims: int,
    device: str,
    noise_seed: int | None = None,
    use_net_value: bool = True,
    value_unit: str = "ref",
    collect: bool = False,
) -> tuple[list[tuple[int, int]], float, list[_Sample]]:
    """Play one episode by PUCT; return ``(order, cost, samples)``."""
    return puct_episodes(
        model,
        [(tensors, sizes, greedy_ref)],
        sims=sims,
        device=device,
        noise_seeds=[noise_seed],
        use_net_value=use_net_value,
        value_unit=value_unit,
        collect=collect,
    )[0]


def _puct_prior(
    model: DualHeadNet,
    tensors: Any,
    sizes: dict[int, int],
    device: str,
    *,
    use_net_value: bool = True,
    value_unit: str = "ref",
) -> tuple[float, float]:
    """Warm per-simulation seconds for the PUCT player on this board.

    Returns ``(batch_1, batch_8)``: the unbatched cost sizes a lone
    episode conservatively (a single episode is not amortised), and the
    batched one is the steady-state cost the superbatch sizing uses.
    """
    ref = cs.greedy(tensors, sizes)
    inst = (tensors, sizes, ref)
    steps = max(len(tensors) - 1, 1)
    kw = {"use_net_value": use_net_value, "value_unit": value_unit}
    puct_episodes(model, [inst] * 8, sims=4, device=device, **kw)
    t0 = time.perf_counter()
    puct_episodes(model, [inst], sims=8, device=device, **kw)
    b1 = (time.perf_counter() - t0) / (8 * steps)
    t0 = time.perf_counter()
    puct_episodes(model, [inst] * 8, sims=8, device=device, **kw)
    b8 = (time.perf_counter() - t0) / (8 * 8 * steps)
    return b1, b8


def puct_best_order(
    model: DualHeadNet,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    budget: float,
    device: str,
    *,
    sims: int,
    seed: int,
    per_sim: float,
    per_sim_b1: float,
    use_net_value: bool = True,
    value_unit: str = "ref",
    batch_episodes: int = 8,
) -> tuple[list[tuple[int, int]], float, int, int]:
    """Anytime PUCT: best episode found within ``budget`` seconds.

    Episodes are played in lockstep superbunches (one batched forward per
    simulation across the batch) with Dirichlet root noise seeded per
    episode, and the best order is kept — the same best-of-N anytime
    contract the sampled-policy player gets.  ``sims`` is a *ceiling*: the
    per-decision simulation count is reduced so that a full superbatch
    fits the budget, which keeps a tight budget from starving the player
    (a single un-amortised episode can cost several times a batched one).
    Returns ``(order, secs, episodes, sims)``.
    """
    steps = max(len(tensors) - 1, 0)
    if steps == 0:
        return [], 0.0, 0, 0
    per_sim_ep = max(per_sim, 1e-6) * steps
    per_sim_ep_b1 = max(per_sim_b1, 1e-6) * steps
    sims_eff = max(
        1, min(sims, int(budget / (batch_episodes * per_sim_ep)))
    )
    per_ep = per_sim_ep * sims_eff
    per_ep_b1 = per_sim_ep_b1 * sims_eff
    t0 = time.perf_counter()
    best: list[tuple[int, int]] | None = None
    best_cost = float("inf")
    episodes = 0
    while True:
        elapsed = time.perf_counter() - t0
        remaining = budget - elapsed
        if episodes > 0 and remaining < per_ep_b1:
            break
        batch = max(
            1, min(batch_episodes, int(remaining / per_ep) or 1)
        )
        inst = (tensors, sizes, greedy_ref)
        seeds: list[int | None] = [
            seed + episodes + i for i in range(batch)
        ]
        results = puct_episodes(
            model,
            [inst] * batch,
            sims=sims_eff,
            device=device,
            noise_seeds=seeds,
            use_net_value=use_net_value,
            value_unit=value_unit,
        )
        for order, cost, _ in results:
            if cost < best_cost:
                best, best_cost = order, cost
        episodes += batch
        if time.perf_counter() - t0 >= budget:
            break
    assert best is not None
    secs = time.perf_counter() - t0
    return best, secs, episodes, episodes * sims_eff * steps


# ---------------------------------------------------------------------------
#  Search in the loop — expert iteration
# ---------------------------------------------------------------------------


def _pack_az(
    samples: list[_Sample],
    targets: list[float],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a variable-pair sample batch into tensors for one update."""
    width = max(len(p) for _s, p, _d, _c in samples)
    sf: list[list[float]] = []
    pf: list[list[list[float]]] = []
    pi: list[list[float]] = []
    for s, p, dist, _c in samples:
        pad = width - len(p)
        sf.append(s)
        pf.append(p + [[0.0] * _PAIR_DIM] * pad)
        pi.append(dist + [0.0] * pad)
    return (
        torch.tensor(sf, dtype=torch.float32, device=device),
        torch.tensor(pf, dtype=torch.float32, device=device),
        torch.tensor(pi, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.float32, device=device),
    )


def train_search(
    ns: tuple[int, ...] = (8, 10, 12),
    *,
    batch: int = 24,
    iterations: int = 200,
    sims: int = 24,
    epochs: int = 4,
    minibatch: int = 64,
    hidden: int = 64,
    lr: float = 1e-3,
    value_coef: float = 1.0,
    value_unit: str = "ref",
    device: str = "cuda",
    seed: int = 0,
    log_every: int = 0,
) -> DualHeadNet:
    """Train :class:`DualHeadNet` by expert iteration with PUCT.

    Each iteration draws a fresh batch of small networks, plays one PUCT
    episode per network, and records ``(state, pair, visit_distribution,
    cost-so-far)``.  The policy head is trained to match the **search's
    visit distribution** (cross-entropy) and the value head to regress
    the **remaining cost to completion** (MSE) — the learned replacement
    of the analytic greedy-completion critic, and the loop in which search
    teaches the net and the net sharpens search.

    The value target is ``log1p(remaining / scale)`` with
    ``remaining = episode_cost - cost_so_far`` and ``scale`` set by
    ``value_unit`` (see :data:`_VALUE_UNITS`): the *board's* greedy
    reference (``"ref"``, the original) or nothing (``"abs"``).  The
    remaining cost is
    *path-independent* (a function of the state's tensor multiset alone),
    so it is far easier to fit — and to discriminate siblings by — than
    the total episode cost, which is dominated by the path.  The log
    makes the regression scale-free and the search inverts it with
    ``expm1``.  ``"ref"`` is *not* scale-free across boards: the board
    reference is unobservable to the net, so the same remaining cost
    gets a different label on every board and the net cannot undo it —
    ``contraction_value_probe`` measures the resulting rank collapse.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = DualHeadNet(hidden).to(device)
    _fit_norm_dual(model, ns, rng, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iterations):
        n0 = ns[it % len(ns)]
        insts: list[tuple[Any, dict[int, int], float]] = []
        refs: list[float] = []
        model.eval()
        for _ in range(batch):
            tensors, sizes = ce.random_bond_network(
                n0, rng.randrange(1 << 30)
            )
            ref = cs.greedy(tensors, sizes)
            insts.append((tensors, sizes, ref))
            refs.append(ref)
        results = puct_episodes(
            model,
            insts,
            sims=sims,
            device=device,
            value_unit=value_unit,
            collect=True,
        )
        samples: list[_Sample] = []
        targets: list[float] = []
        for (_order, cost, data), ref in zip(
            results, refs, strict=True
        ):
            for s, p, dist, paid in data:
                samples.append((s, p, dist, paid))
                remaining = max(cost - paid, 0.0)
                scale = ref if value_unit == "ref" else 1.0
                targets.append(
                    math.log1p(remaining / scale) if scale > 0 else 0.0
                )
        model.train()
        last = 0.0
        for _ in range(epochs):
            order = list(range(len(samples)))
            rng.shuffle(order)
            for start in range(0, len(order), minibatch):
                chunk = order[start : start + minibatch]
                s, p, pi, v = _pack_az(
                    [samples[i] for i in chunk],
                    [targets[i] for i in chunk],
                    device,
                )
                logits, value = model(s, p)
                pol = (
                    -(pi * F.log_softmax(logits, dim=-1)).sum(-1).mean()
                )
                val = F.mse_loss(value, v)
                loss = pol + value_coef * val
                opt.zero_grad()
                loss.backward()
                opt.step()
                last = float(loss.detach())
        if log_every and (it + 1) % log_every == 0:
            print(
                f"  az iter {it + 1:>4}: loss {last:.4f} "
                f"samples {len(samples)}"
            )
    return model.eval()


# ---------------------------------------------------------------------------
#  Equal-wall-clock board
# ---------------------------------------------------------------------------


@dataclass
class _AzCfg:
    """The knobs a PUCT board run needs."""

    sims: int
    per_sim: float
    per_sim_b1: float
    budget: float
    seed: int
    batch_episodes: int = 8
    value_unit: str = "ref"


def _run_az_board(
    tensors: Any,
    sizes: dict[int, int],
    models: dict[str, Any],
    device: str,
    cfg: _AzCfg,
    *,
    puct_analytic: bool,
) -> dict[str, ce._Res]:
    """Run every player on one board at an equal wall-clock budget."""
    e = ce.to_einsum(tensors, sizes)
    ref = cs.greedy(tensors, sizes)
    steps = max(len(tensors) - 1, 0)
    out: dict[str, ce._Res] = {}

    t0 = time.perf_counter()
    order = ce.our_greedy_order(tensors, sizes)
    out["our-greedy"] = ce._res(
        tensors, sizes, e, order, time.perf_counter() - t0, steps
    )

    order, secs, rolls = ce.our_restart_order(
        tensors, sizes, cfg.budget, cfg.seed
    )
    out["our-restart"] = ce._res(tensors, sizes, e, order, secs, rolls)

    cur = models["current"]
    prior = ce._policy_prior(cur, tensors, sizes, device)
    order, secs, rolls = ce.policy_best_order(
        cur, tensors, sizes, ref, cfg.budget, device, prior, cfg.seed
    )
    out["current"] = ce._res(tensors, sizes, e, order, secs, rolls)

    az = models["az"]
    order, secs, _eps, sims = puct_best_order(
        az,
        tensors,
        sizes,
        ref,
        cfg.budget,
        device,
        sims=cfg.sims,
        seed=cfg.seed,
        per_sim=cfg.per_sim,
        per_sim_b1=cfg.per_sim_b1,
        value_unit=cfg.value_unit,
        batch_episodes=cfg.batch_episodes,
    )
    out["puct"] = ce._res(tensors, sizes, e, order, secs, sims)
    if puct_analytic:
        a_b1, a_b8 = _puct_prior(
            az, tensors, sizes, device, use_net_value=False
        )
        order, secs, _eps, sims = puct_best_order(
            az,
            tensors,
            sizes,
            ref,
            cfg.budget,
            device,
            sims=cfg.sims,
            seed=cfg.seed,
            per_sim=a_b8,
            per_sim_b1=a_b1,
            use_net_value=False,
            batch_episodes=cfg.batch_episodes,
        )
        out["puct-analytic"] = ce._res(
            tensors, sizes, e, order, secs, sims
        )

    t0 = time.perf_counter()
    order = ce.oe_greedy_order(e)
    out["oe-greedy"] = ce._res(
        tensors, sizes, e, order, time.perf_counter() - t0, steps
    )

    t0 = time.perf_counter()
    order = ce.oe_optimal_order(e, cfg.budget)
    out["oe-optimal"] = ce._res(
        tensors, sizes, e, order, time.perf_counter() - t0, 0.0
    )

    t0 = time.perf_counter()
    order = ce.oe_random_greedy_order(e, cfg.budget)
    out["oe-rand-greedy"] = ce._res(
        tensors, sizes, e, order, time.perf_counter() - t0, float("nan")
    )
    return out


#: Players reported, in table order.
_PLAYERS = (
    "our-greedy",
    "our-restart",
    "current",
    "puct",
    "puct-analytic",
    "oe-greedy",
    "oe-rand-greedy",
    "oe-optimal",
)


def _players(puct_analytic: bool) -> tuple[str, ...]:
    """Return the players to report."""
    return tuple(
        p for p in _PLAYERS if p != "puct-analytic" or puct_analytic
    )


def _measure_az(
    budgets: tuple[float, ...],
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, Any],
    device: str,
    sims: int,
    *,
    puct_analytic: bool,
    batch_episodes: int,
    value_unit: str = "ref",
) -> list[tuple[float, int, list[dict[str, ce._Res]]]]:
    """Run every player on every board once; return the raw results."""
    data: list[tuple[float, int, list[dict[str, ce._Res]]]] = []
    for budget in budgets:
        for n in scales:
            boards = []
            for k in range(instances):
                tensors, sizes = ce.random_bond_network(n, seed + k)
                b1, b8 = _puct_prior(
                    models["az"],
                    tensors,
                    sizes,
                    device,
                    value_unit=value_unit,
                )
                cfg = _AzCfg(
                    sims,
                    b8,
                    b1,
                    budget,
                    seed + k,
                    batch_episodes,
                    value_unit,
                )
                boards.append(
                    _run_az_board(
                        tensors,
                        sizes,
                        models,
                        device,
                        cfg,
                        puct_analytic=puct_analytic,
                    )
                )
            data.append((budget, n, boards))
    return data


def _ladder_az(
    data: list[tuple[float, int, list[dict[str, ce._Res]]]],
    puct_analytic: bool,
) -> None:
    """Print the equal-wall-clock ladder under both cost models."""
    players = _players(puct_analytic)
    for budget in sorted({b for b, _n, _d in data}):
        print()
        print(
            f"== 2. equal wall-clock: {1e3 * budget:.0f} ms per instance =="
        )
        print(
            f"  {'n':>3} {'player':>14} {'our-ratio':>10} "
            f"{'oe-ratio':>10} {'work':>9} {'ms':>7}"
        )
        print("-" * 60)
        for b, n, boards in data:
            if b != budget:
                continue
            for p in players:
                our = [
                    d[p].our / min(r.our for r in d.values())
                    for d in boards
                ]
                oe = [
                    d[p].oe / min(r.oe for r in d.values())
                    for d in boards
                ]
                work = [d[p].work for d in boards]
                secs = [d[p].secs for d in boards]
                print(
                    f"  {n:>3} {p:>14} "
                    f"{statistics.fmean(our):>10.3f} "
                    f"{statistics.fmean(oe):>10.3f} "
                    f"{ce._num(statistics.fmean(work), 9, '.1f')} "
                    f"{1e3 * statistics.fmean(secs):>7.1f}"
                )


def _pairwise_az(
    data: list[tuple[float, int, list[dict[str, ce._Res]]]],
    *,
    puct_analytic: bool,
) -> None:
    """Print PUCT-vs-current and PUCT-vs-opt_einsum ratios."""
    cols = ["current", "oe-greedy", "oe-rand-greedy"]
    print()
    print("== 3. pairwise ratio (mean; <1 = PUCT wins) ==")
    head = (
        f"  {'ms':>6} {'n':>3} "
        + " ".join(f"{'our/' + c:>16}" for c in cols)
        + " "
        + " ".join(f"{'oe/' + c:>16}" for c in cols)
    )
    print(head)
    print("-" * len(head))

    def _cell(vals: list[float]) -> str:
        finite = [v for v in vals if v == v]
        return (
            ce._num(statistics.fmean(finite), 16)
            if finite
            else f"{'DNF':>16}"
        )

    for budget, n, boards in data:
        our_r: dict[str, list[float]] = {c: [] for c in cols}
        oe_r: dict[str, list[float]] = {c: [] for c in cols}
        for d in boards:
            puct = d["puct"]
            for c in cols:
                our_r[c].append(
                    puct.our / d[c].our
                    if d[c].our < float("inf")
                    else float("nan")
                )
                oe_r[c].append(
                    puct.oe / d[c].oe
                    if d[c].oe < float("inf")
                    else float("nan")
                )
        print(
            f"  {1e3 * budget:>6.0f} {n:>3} "
            + " ".join(_cell(our_r[c]) for c in cols)
            + " "
            + " ".join(_cell(oe_r[c]) for c in cols)
        )


def _verdict_az(
    data: list[tuple[float, int, list[dict[str, ce._Res]]]],
    *,
    puct_analytic: bool,
) -> None:
    """Print the honest equal-wall-clock verdict from the rows."""
    print()
    print("== 4. verdict — PUCT + learned value at equal wall-clock ==")
    for budget, n, boards in data:
        cur = statistics.fmean(
            [d["puct"].our / d["current"].our for d in boards]
        )
        rand = statistics.fmean(
            [d["puct"].oe / d["oe-rand-greedy"].oe for d in boards]
        )
        greedy = statistics.fmean(
            [d["puct"].oe / d["oe-greedy"].oe for d in boards]
        )
        work = statistics.fmean([d["puct"].work for d in boards])
        line = (
            f"  {1e3 * budget:>5.0f}ms n={n:>2}: "
            f"puct/current {cur:.3f}  puct/oe-rand {rand:.3f}  "
            f"puct/oe-greedy {greedy:.3f}  (work {work:.0f})"
        )
        if puct_analytic:
            ana = statistics.fmean(
                [d["puct"].our / d["puct-analytic"].our for d in boards]
            )
            line += f"  net-vs-analytic {ana:.3f}"
        print(line)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def _warm_az(
    scales: tuple[int, ...],
    model: DualHeadNet,
    device: str,
    *,
    value_unit: str = "ref",
) -> None:
    """Warm the net's kernels before any board is timed."""
    for n in scales:
        tensors, sizes = ce.random_bond_network(n, 0)
        ref = cs.greedy(tensors, sizes)
        puct_episode(
            model,
            tensors,
            sizes,
            ref,
            sims=4,
            device=device,
            value_unit=value_unit,
        )


def main(argv: list[str] | None = None) -> int:
    """Train the two-head net, then measure it at equal wall-clock."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--sims", type=int, default=24)
    ap.add_argument(
        "--eval-sims",
        type=int,
        default=0,
        help="simulations per decision at evaluation (0 = --sims)",
    )
    ap.add_argument(
        "--batch-episodes",
        type=int,
        default=8,
        help="episodes per lockstep superbatch (smaller = deeper search)",
    )
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument(
        "--scales",
        default="20,30,40",
        help="comma-separated n at scale",
    )
    ap.add_argument(
        "--budgets",
        default="50,200,1000",
        help="comma-separated per-instance budgets (ms)",
    )
    ap.add_argument("--trainer", default="az", choices=("az", "none"))
    ap.add_argument(
        "--az-iterations",
        type=int,
        default=0,
        help="expert-iteration count for the PUCT net (0 = --iterations)",
    )
    ap.add_argument(
        "--load",
        default="",
        help="load a saved PUCT net instead of training one",
    )
    ap.add_argument(
        "--save",
        default="",
        help="save the trained PUCT net state dict",
    )
    ap.add_argument(
        "--puct-analytic",
        action="store_true",
        help="also run PUCT with the analytic greedy-completion value",
    )
    ap.add_argument(
        "--value-unit",
        default="ref",
        choices=_VALUE_UNITS,
        help="value-target scale: 'ref' (board greedy ref, the original) "
        "or 'abs' (absolute remaining cost)",
    )
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  (torch {torch.__version__})")

    az_iters = args.az_iterations or args.iterations
    models: dict[str, Any] = {}
    t0 = time.perf_counter()
    with ce._on_family(ce.random_bond_network):
        models["current"] = cp.train_rl(
            iterations=args.iterations,
            batch=args.batch,
            hidden=args.hidden,
            device=dev,
            seed=args.seed,
        )
        print(
            f"trained current policy in {time.perf_counter() - t0:.1f}s"
        )
        if args.load:
            az = DualHeadNet(args.hidden).to(dev)
            az.load_state_dict(torch.load(args.load, map_location=dev))
            models["az"] = az.eval()
        elif args.trainer == "none":
            raise SystemExit("--trainer none needs --load")
        else:
            t0 = time.perf_counter()
            models["az"] = train_search(
                batch=args.batch,
                iterations=az_iters,
                sims=args.sims,
                epochs=args.epochs,
                hidden=args.hidden,
                value_unit=args.value_unit,
                device=dev,
                seed=args.seed,
            )
            print(
                f"trained PUCT net in {time.perf_counter() - t0:.1f}s"
            )
    if args.save:
        torch.save(models["az"].state_dict(), args.save)
        print(f"saved PUCT net to {args.save}")

    scales = tuple(int(x) for x in args.scales.split(",") if x.strip())
    budgets = tuple(
        1e-3 * float(x) for x in args.budgets.split(",") if x.strip()
    )
    _warm_az(scales, models["az"], dev, value_unit=args.value_unit)
    eval_sims = args.eval_sims or args.sims
    data = _measure_az(
        budgets,
        scales,
        args.instances,
        args.seed,
        models,
        dev,
        eval_sims,
        puct_analytic=args.puct_analytic,
        batch_episodes=args.batch_episodes,
        value_unit=args.value_unit,
    )
    _ladder_az(data, args.puct_analytic)
    _pairwise_az(data, puct_analytic=args.puct_analytic)
    _verdict_az(data, puct_analytic=args.puct_analytic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
