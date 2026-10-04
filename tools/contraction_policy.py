"""Learned contraction ordering — beating the cheap heuristics at scale.

Plan 0016 follow-up.  ``contraction_scale.py`` measured a player ladder
(greedy, one-step, bounded best-first ``search``, randomised
``restart``) on seeded random tensor networks at ``n`` = 20/30/40/60 and
found **headroom above greedy** (up to 13x mean) — but the real
baselines to beat are ``search`` / ``restart``, already 1.00-1.04 of
best-found.  This tool asks the decisive question:

> Can a *learned* contraction-ordering policy beat the cheap heuristics
> at the scale where exact and saturation both fail?

The game is the contraction rule space, framed directly on states (not
on an e-graph): a state is the set of remaining tensors (a partial
contraction), an action is one legal ``contract`` of a tensor pair, and
the reward is the (negative) pairwise cost the move incurs — exactly
the structural cost ``contraction_scale`` prices.  State and action are
described by **scale-free** feature vectors (relative to the state's own
pair-cost spread), so a policy trained on small ``n`` can be applied to
large ``n``.

The policy is a small MLP scoring ``(state-features (+) pair-features)``
(the shape :class:`catopt_torch.rl.PolicyNet` uses), trained on *small*
networks (n = 8-12) where many episodes are affordable, then tested on
*large* unseen networks (n = 20/30/40).  Two trainers are provided:

* ``rl`` — REINFORCE with a **heuristic critic**: the per-step advantage
  is ``V(s) - cost - V(s')`` where ``V`` is the greedy-completion cost (a
  cheap, scale-free baseline).  Summed over an episode it telescopes to
  ``greedy_ref - episode_cost``, so maximising it *is* beating greedy.
* ``imitation`` — supervised on the exact DP-optimal contraction order
  (``dp`` is affordable at n = 8-12), the strongest available signal.

The results above match the *rollout count* (1 vs 1, 64 vs 64), not the
*work*: a learned rollout pays a forward pass per step, a
randomised-greedy rollout pays a heuristic scan.  ``--mode time`` closes
that gap: every player gets the **same wall-clock budget** per instance
(``--budgets`` ms) and the tool reports quality, rollouts and decisions
at that budget, plus the measured per-decision cost of a policy forward
pass against a heuristic scan.  It is the falsification half of the
result — reported as measured, including a negative.

The player machinery itself — the game, the net, the feature derivation,
the rollout driver — is **shipped** in
:mod:`catopt_torch.contraction_policy` (the bundled-weights player,
``contraction-player-artifact.md``); it is imported above so this tool
keeps only the trainers and the measurement ladders.

Usage::

    python tools/contraction_policy.py [--trainer rl] [--seed 0]
    python tools/contraction_policy.py --mode time --budgets 50,200,1000
"""

from __future__ import annotations

import argparse
import functools
import heapq
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import count
from typing import Any

import contraction_scale as cs
import torch
from catopt_torch.contraction_policy import (
    MAX_BATCH as _MAX_BATCH,
)
from catopt_torch.contraction_policy import (
    PAIR_DIM as _PAIR_DIM,
)
from catopt_torch.contraction_policy import (
    SAMPLE_TEMP as _TEMP,
)
from catopt_torch.contraction_policy import (
    STD_FLOOR as _STD_FLOOR,
)
from catopt_torch.contraction_policy import (
    ContractionGame,
    PairPolicyNet,
    run_policy_batch,
)
from catopt_torch.contraction_policy import (
    batch_inputs as _batch_inputs,
)
from catopt_torch.contraction_policy import (
    merge_tensors as _merge_ts,
)
from catopt_torch.contraction_policy import (
    pair_logits as _logits,
)
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

__all__ = ["main"]


@dataclass
class _Rollout:
    """The transitions a sampled episode produced (for the RL update)."""

    logps: list[torch.Tensor]
    advantages: list[float]
    entropies: list[torch.Tensor]


def _fit_norm(
    model: nn.Module,
    ns: tuple[int, ...],
    rng: random.Random,
    device: str,
) -> None:
    """Fit the input mean/std buffers from real greedy trajectories."""
    rows: list[list[float]] = []
    for _ in range(8):
        for n0 in ns:
            tensors, sizes = cs.random_network(
                n0, rng.randrange(1 << 30)
            )
            g = ContractionGame(
                tensors, sizes, cs.greedy(tensors, sizes)
            )
            while not g.done:
                sf = g.state_features()
                rows.extend([sf + f for f in g.all_pair_features()])
                g.step(*g.pairs[0])
    x = torch.tensor(rows, dtype=torch.float32)
    with torch.no_grad():
        model.mean.copy_(x.mean(0))
        model.std.copy_(x.std(0).clamp_min(_STD_FLOOR))


# ---------------------------------------------------------------------------
#  Trainer 1 — REINFORCE with a greedy-completion critic
# ---------------------------------------------------------------------------


def _sample_episode(
    model: nn.Module, games: list[ContractionGame], device: str
) -> _Rollout:
    """Sample one lockstep episode; return its per-step RL signals.

    The advantage at a step is ``V(s) - inc - V(s')`` with ``V`` the
    greedy completion cost: a temporal-difference error against a cheap
    heuristic critic.  It telescopes to ``greedy_ref - episode_cost``,
    so maximising the summed advantage minimises the episode cost.
    Log-probs and advantages are flattened in the *same* (episode-major)
    order, so they pair up element-wise.
    """
    logps: list[list[torch.Tensor]] = [[] for _ in games]
    advantages: list[list[float]] = [[] for _ in games]
    entropies: list[torch.Tensor] = []
    v_prev = [g.greedy_ref for g in games]
    while not games[0].done:
        sf, pf = _batch_inputs(games, device)
        dist = Categorical(logits=_logits(model, sf, pf))
        idx = dist.sample()
        lp = dist.log_prob(idx)
        entropies.append(dist.entropy().mean())
        for j, (g, i) in enumerate(
            zip(games, idx.tolist(), strict=True)
        ):
            inc = g.step(*g.pairs[i])
            v_next = cs.greedy(g.ts, g.sizes)
            advantages[j].append(
                (v_prev[j] - inc - v_next) / g.greedy_ref
            )
            v_prev[j] = v_next
            logps[j].append(lp[j])
    flat_lp = [p for row in logps for p in row]
    flat_adv = [a for row in advantages for a in row]
    return _Rollout(flat_lp, flat_adv, entropies)


def train_rl(
    ns: tuple[int, ...] = (8, 10, 12),
    *,
    batch: int = 24,
    iterations: int = 1800,
    hidden: int = 64,
    lr: float = 1e-3,
    entropy_beta: float = 0.005,
    device: str = "cuda",
    seed: int = 0,
    log_every: int = 0,
) -> nn.Module:
    """Train a :class:`PairPolicyNet` by REINFORCE on small networks.

    Each iteration draws a fresh batch of seeded small networks, samples
    one lockstep episode, and updates on the per-step heuristic-critic
    advantage plus an entropy bonus.  The advantage is standardised over
    the batch — without it the catastrophic-move tail (a huge negative
    ``V(s')``) dominates the gradient and the policy diverges.  The
    device defaults to CUDA.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = PairPolicyNet(hidden).to(device)
    _fit_norm(model, ns, rng, device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for it in range(iterations):
        n0 = ns[it % len(ns)]
        games = []
        for _ in range(batch):
            tensors, sizes = cs.random_network(
                n0, rng.randrange(1 << 30)
            )
            games.append(
                ContractionGame(
                    tensors, sizes, cs.greedy(tensors, sizes)
                )
            )
        roll = _sample_episode(model, games, device)
        adv = torch.tensor(
            roll.advantages, dtype=torch.float32, device=device
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        logps = torch.stack(roll.logps)
        loss = -(adv.detach() * logps).mean()
        ent = torch.stack(roll.entropies).mean()
        loss = loss - entropy_beta * ent
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and (it + 1) % log_every == 0:
            print(
                f"  rl iter {it + 1:>4}: loss {float(loss.detach()):.4f} "
                f"entropy {float(ent.detach()):.3f}"
            )
    return model.eval()


# ---------------------------------------------------------------------------
#  Trainer 2 — imitation of the exact DP-optimal order
# ---------------------------------------------------------------------------


def _xor_table(tensors: Any) -> list[frozenset[int]]:
    """Return ``xor[mask]`` = symmetric difference of the mask's leaves."""
    idx = [frozenset(t) for t in tensors]
    n = len(idx)
    xor: list[frozenset[int]] = [frozenset()] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        xor[mask] = xor[mask ^ low] ^ idx[low.bit_length() - 1]
    return xor


def dp_optimal_order(
    tensors: Any, sizes: dict[int, int]
) -> tuple[float, list[tuple[int, int]], list[frozenset[int]]]:
    """Exact min-cost order: ``(cost, merges, xor)``.

    ``merges`` is a valid sequence of ``(maskL, maskR)`` leaf-mask
    contractions achieving ``cost``; ``xor[mask]`` gives the tensor a
    leaf-mask represents.  Same recurrence as ``contraction_scale.dp``,
    but it records the optimal split so the order can be replayed.
    """
    idx = [frozenset(t) for t in tensors]
    n = len(idx)
    xor = _xor_table(tensors)
    f = [0.0] * (1 << n)
    split = [0] * (1 << n)
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue
        best = float("inf")
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if other:
                c = (
                    f[sub]
                    + f[other]
                    + cs.pair_cost(xor[sub], xor[other], sizes)
                )
                if c < best:
                    best, split[mask] = c, sub
            sub = (sub - 1) & mask
        f[mask] = best
    merges: list[tuple[int, int]] = []

    def _rec(mask: int) -> None:
        if mask & (mask - 1) == 0:
            return
        s = split[mask]
        o = mask ^ s
        _rec(s)
        _rec(o)
        merges.append((s, o))

    _rec((1 << n) - 1)
    return f[(1 << n) - 1], merges, xor


def _dp_index_order(
    tensors: Any, sizes: dict[int, int]
) -> list[tuple[int, int]]:
    """Return a DP-optimal order as current-position index pairs."""
    _cost, merges, _xor = dp_optimal_order(tensors, sizes)
    cur = [1 << i for i in range(len(tensors))]
    order: list[tuple[int, int]] = []
    for mask_l, mask_r in merges:
        a = cur.index(mask_l)
        b = cur.index(mask_r)
        order.append((min(a, b), max(a, b)))
        cur = [m for k, m in enumerate(cur) if k not in (a, b)] + [
            mask_l | mask_r
        ]
    return order


def _random_greedy_order(
    tensors: Any,
    sizes: dict[int, int],
    rng: random.Random,
    top_k: int,
) -> list[tuple[int, int]]:
    """One randomised-greedy episode as current-position index pairs."""
    ts = [frozenset(t) for t in tensors]
    order: list[tuple[int, int]] = []
    while len(ts) > 1:
        cands = sorted(
            (cs.pair_cost(ts[a], ts[b], sizes), a, b)
            for a, b in cs._pairs(ts)
        )
        _c, a, b = cands[rng.randrange(min(top_k, len(cands)))]
        order.append((a, b))
        ts = _merge_ts(ts, a, b)
    return order


def _order_cost(
    tensors: Any, sizes: dict[int, int], order: list[tuple[int, int]]
) -> float:
    """Replay an index-pair order and sum the pairwise cost."""
    ts = [frozenset(t) for t in tensors]
    total = 0.0
    for a, b in order:
        total += cs.pair_cost(ts[a], ts[b], sizes)
        ts = _merge_ts(ts, a, b)
    return total


def _restart_order(
    tensors: Any,
    sizes: dict[int, int],
    rng: random.Random,
    restarts: int,
    top_k: int,
) -> list[tuple[int, int]]:
    """Best-of-``restarts`` randomised-greedy order (the cheap teacher)."""
    best = _random_greedy_order(tensors, sizes, rng, 1)
    best_cost = _order_cost(tensors, sizes, best)
    for _ in range(restarts):
        order = _random_greedy_order(tensors, sizes, rng, top_k)
        cost = _order_cost(tensors, sizes, order)
        if cost < best_cost:
            best, best_cost = order, cost
    return best


def _make_teacher(
    kind: str, dp_max: int, restarts: int, top_k: int
) -> Callable[
    [Any, dict[int, int], random.Random], list[tuple[int, int]]
]:
    """Build a teacher: exact DP where affordable, else best-of-restart.

    The DP is ``O(3^n)`` and so only affordable to ``dp_max``; beyond it
    the teacher falls back to a multi-restart best-of — the strongest
    cheap player available at that scale.  ``kind="dp"`` uses the DP up
    to ``dp_max`` and the restart fallback above; ``kind="restart"``
    uses the restart best-of at every scale.
    """

    def _teacher(
        tensors: Any, sizes: dict[int, int], rng: random.Random
    ) -> list[tuple[int, int]]:
        if kind == "dp" and len(tensors) <= dp_max:
            return _dp_index_order(tensors, sizes)
        return _restart_order(tensors, sizes, rng, restarts, top_k)

    return _teacher


def _replay_samples(
    tensors: Any,
    sizes: dict[int, int],
    order: list[tuple[int, int]],
    n0: int,
) -> list[tuple[ContractionGame, tuple[int, int]]]:
    """Turn a teacher's index-pair order into ``(state, action)`` samples."""
    ref = cs.greedy(tensors, sizes)
    ts = [frozenset(t) for t in tensors]
    acc = 0.0
    out: list[tuple[ContractionGame, tuple[int, int]]] = []
    for a, b in order:
        game = ContractionGame(ts, sizes, ref, n0=n0, cost=acc)
        out.append((game, (min(a, b), max(a, b))))
        acc += cs.pair_cost(ts[a], ts[b], sizes)
        ts = _merge_ts(ts, a, b)
    return out


def _imitation_dataset(
    ns: tuple[int, ...],
    per_n: int,
    rng: random.Random,
    teacher: Callable[
        [Any, dict[int, int], random.Random], list[tuple[int, int]]
    ],
) -> list[tuple[ContractionGame, tuple[int, int]]]:
    """Build ``(state, teacher action)`` samples from teacher orders."""
    data: list[tuple[ContractionGame, tuple[int, int]]] = []
    for n0 in ns:
        for _ in range(per_n):
            tensors, sizes = cs.random_network(
                n0, rng.randrange(1 << 30)
            )
            order = teacher(tensors, sizes, rng)
            data.extend(_replay_samples(tensors, sizes, order, n0))
    return data


def train_imitation(
    ns: tuple[int, ...] = (8, 10, 12),
    *,
    per_n: int = 40,
    epochs: int = 300,
    batch: int = 64,
    hidden: int = 64,
    lr: float = 3e-3,
    device: str = "cuda",
    seed: int = 0,
    teacher: str = "dp",
    dp_max: int = 14,
    restarts: int = 64,
    top_k: int = 3,
) -> nn.Module:
    """Train a :class:`PairPolicyNet` to imitate a teacher's action.

    The label is the teacher's pair at each state along its order; the
    loss is a masked cross-entropy over the legal pairs.  ``teacher``
    is ``"dp"`` (the exact subset DP up to ``dp_max``, the strongest
    signal where it is affordable, falling back to a multi-restart
    best-of beyond it) or ``"restart"`` (the multi-restart best-of at
    every scale).  The teacher is the only thing that scales with ``n``
    — the features, the net and the loss are unchanged.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = PairPolicyNet(hidden).to(device)
    _fit_norm(model, ns, rng, device)
    data = _imitation_dataset(
        ns, per_n, rng, _make_teacher(teacher, dp_max, restarts, top_k)
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        rng.shuffle(data)
        for start in range(0, len(data), batch):
            chunk = data[start : start + batch]
            sf, pf, mask, target = _pack(chunk, device)
            logits = _logits(model, sf, pf).masked_fill(mask == 0, -1e9)
            loss = F.cross_entropy(
                logits, torch.tensor(target, device=device)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def _pack(
    chunk: list[tuple[ContractionGame, tuple[int, int]]], device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Pad a variable-pair batch into tensors plus a validity mask."""
    width = max(len(g.pairs) for g, _ in chunk)
    sf: list[list[float]] = []
    pf: list[list[list[float]]] = []
    mask: list[list[float]] = []
    target: list[int] = []
    for game, action in chunk:
        feats = game.all_pair_features()
        pad = width - len(feats)
        sf.append(game.state_features())
        pf.append(feats + [[0.0] * _PAIR_DIM] * pad)
        mask.append([1.0] * len(feats) + [0.0] * pad)
        target.append(game.pairs.index(action))
    return (
        torch.tensor(sf, dtype=torch.float32, device=device),
        torch.tensor(pf, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
        target,
    )


# ---------------------------------------------------------------------------
#  The ladder
# ---------------------------------------------------------------------------

#: Number of sampled policy rollouts in the ``policy-restart`` player.
_RESTARTS = 64


def _ladder(
    tensors: Any,
    sizes: dict[int, int],
    n: int,
    models: dict[str, nn.Module],
    device: str,
) -> dict[str, float]:
    """Run every feasible player plus the learned policies on one board."""
    ref = cs.greedy(tensors, sizes)
    out = {"greedy": ref, "restart": cs.restart(tensors, sizes)}
    if n <= 20:
        out["one-step"] = cs.one_step(tensors, sizes)
    if n <= 30:
        found = cs.search(
            tensors, sizes, max_states=3000 if n <= 20 else 300
        )
        if found < float("inf"):
            out["search"] = found
    for name, model in models.items():
        out[name] = run_policy_batch(
            model,
            tensors,
            sizes,
            ref,
            samples=1,
            greedy=True,
            temperature=1.0,
            device=device,
        )[0]
        out[f"{name}-restart"] = min(
            run_policy_batch(
                model,
                tensors,
                sizes,
                ref,
                samples=_RESTARTS,
                greedy=False,
                temperature=_TEMP,
                device=device,
            )
        )
    return out


def _ratio_table(
    rows: list[tuple[int, dict[str, list[float]]]],
) -> None:
    """Print the per-scale ratio-to-best-found table."""
    players = [
        "greedy",
        "one-step",
        "search",
        "restart",
        "rl",
        "rl-restart",
        "imitation",
        "imitation-restart",
    ]
    head = f"{'n':>3}  " + " ".join(f"{p:>16}" for p in players)
    print(head)
    print("-" * len(head))
    for n, ratios in rows:
        cells = []
        for p in players:
            vals = ratios.get(p)
            cells.append(
                f"{statistics.fmean(vals):>16.3f}"
                if vals
                else f"{'-':>16}"
            )
        print(f"{n:>3}  " + " ".join(cells))


def _control_table(
    ns: tuple[int, ...],
    instances: int,
    models: dict[str, nn.Module],
    device: str,
) -> None:
    """Print the small-n control: players / exact DP optimum (mean)."""
    print()
    print(
        "== controls: players / exact DP optimum (mean over seeds) =="
    )
    players = ["greedy", "restart"]
    for name in models:
        players += [name, f"{name}-restart"]
    head = f"{'n':>3} " + " ".join(f"{p:>9}" for p in players)
    print(head)
    print("-" * len(head))
    for n in ns:
        acc: dict[str, list[float]] = {p: [] for p in players}
        for k in range(instances):
            tensors, sizes = cs.random_network(n, 1234 + n + k)
            opt = cs.dp(tensors, sizes)
            ref = cs.greedy(tensors, sizes)
            acc["greedy"].append(ref / opt)
            acc["restart"].append(cs.restart(tensors, sizes) / opt)
            for name, model in models.items():
                acc[name].append(
                    run_policy_batch(
                        model,
                        tensors,
                        sizes,
                        ref,
                        samples=1,
                        greedy=True,
                        temperature=1.0,
                        device=device,
                    )[0]
                    / opt
                )
                acc[f"{name}-restart"].append(
                    min(
                        run_policy_batch(
                            model,
                            tensors,
                            sizes,
                            ref,
                            samples=_RESTARTS,
                            greedy=False,
                            temperature=_TEMP,
                            device=device,
                        )
                    )
                    / opt
                )
        line = f"{n:>3} " + " ".join(
            f"{statistics.fmean(acc[p]):>9.3f}" for p in players
        )
        print(line)


def _pairwise_table(
    rows: list[tuple[int, dict[str, list[float]]]],
) -> None:
    """Print learned-vs-cheap pairwise cost ratios (mean over seeds)."""
    print()
    print(
        "== pairwise cost ratio (mean over seeds; <1 = learned wins) =="
    )
    order: list[str] = []
    for learned in ("rl", "imitation"):
        order += [
            f"{learned}/greedy",
            f"{learned}/restart",
            f"{learned}/search",
            f"{learned}-restart/restart",
        ]
    cols = [c for c in order if any(c in pair for _n, pair in rows)]
    head = f"{'n':>3} " + " ".join(f"{c:>18}" for c in cols)
    print(head)
    print("-" * len(head))
    for n, pair in rows:
        cells = []
        for c in cols:
            vals = pair.get(c)
            cells.append(
                f"{statistics.fmean(vals):>18.3f}"
                if vals
                else f"{'-':>18}"
            )
        print(f"{n:>3} " + " ".join(cells))


def _scale_table(
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, nn.Module],
    device: str,
) -> list[tuple[int, dict[str, list[float]]]]:
    """Measure every player at scale; return the ratio rows."""
    print()
    print("== learned policy vs the cheap heuristics at scale ==")
    print("  ratio to best-found: mean over seeds (lower is better)")
    rows = []
    pairs: list[tuple[int, dict[str, list[float]]]] = []
    for n in scales:
        ratios: dict[str, list[float]] = {}
        pair: dict[str, list[float]] = {}
        for k in range(instances):
            tensors, sizes = cs.random_network(n, seed + k)
            found = _ladder(tensors, sizes, n, models, device)
            best = min(found.values())
            for name, v in found.items():
                ratios.setdefault(name, []).append(v / best)
            for learned in models:
                for base in ("greedy", "restart", "search"):
                    if base in found:
                        pair.setdefault(f"{learned}/{base}", []).append(
                            found[learned] / found[base]
                        )
                if "restart" in found:
                    pair.setdefault(
                        f"{learned}-restart/restart", []
                    ).append(
                        found[f"{learned}-restart"] / found["restart"]
                    )
        rows.append((n, ratios))
        pairs.append((n, pair))
    _ratio_table(rows)
    _pairwise_table(pairs)
    return rows


def _verdict(
    rows: list[tuple[int, dict[str, list[float]]]],
) -> None:
    """Print the honest verdict from the measured ratios."""
    print()
    print("== verdict ==")
    for n, ratios in rows:
        cheap = [
            statistics.fmean(ratios[p])
            for p in ("search", "restart")
            if p in ratios
        ]
        cells = f"  n={n}: best cheap {min(cheap):.3f}"
        for name in ratios:
            if name in ("search", "restart"):
                continue
            cells += f"  {name} {statistics.fmean(ratios[name]):.3f}"
        print(cells)


# ---------------------------------------------------------------------------
#  Equal wall-clock: the anytime players
# ---------------------------------------------------------------------------


@dataclass
class _Any:
    """One anytime player's result: best cost, work done, wall time."""

    cost: float
    rollouts: int
    decisions: int
    secs: float


def _all_pairs(ts: list[frozenset[int]]) -> list[tuple[int, int]]:
    """Return every unordered pair of tensor positions."""
    return [
        (a, b) for a in range(len(ts)) for b in range(a + 1, len(ts))
    ]


def _state_key(ts: list[frozenset[int]]) -> tuple:
    """Return a canonical, order-independent key for a state."""
    return tuple(sorted(tuple(sorted(t)) for t in ts))


def _sync(device: str) -> None:
    """Synchronise the device when it is a CUDA device."""
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def anytime_greedy(
    tensors: Any, sizes: dict[int, int], budget: float
) -> _Any:
    """One deterministic greedy episode (extra budget cannot help it)."""
    t0 = time.perf_counter()
    c = cs.greedy(tensors, sizes)
    return _Any(
        c, 1, max(len(tensors) - 1, 0), time.perf_counter() - t0
    )


def anytime_restart(
    tensors: Any,
    sizes: dict[int, int],
    budget: float,
    *,
    top_k: int = 3,
    seed: int = 0,
) -> _Any:
    """Best of as many randomised-greedy episodes as fit in ``budget``."""
    rng = random.Random(seed)
    steps = max(len(tensors) - 1, 0)
    t0 = time.perf_counter()
    best = cs.greedy(tensors, sizes)
    rollouts = 1
    decisions = steps
    while time.perf_counter() - t0 < budget:
        c = cs.greedy(tensors, sizes, rng=rng, top_k=top_k)
        rollouts += 1
        decisions += steps
        if c < best:
            best = c
    return _Any(best, rollouts, decisions, time.perf_counter() - t0)


def anytime_search(
    tensors: Any, sizes: dict[int, int], budget: float
) -> _Any:
    """Best-first search, deadline-bounded inside the expansion loop.

    Identical to ``contraction_scale.search`` except the wall-clock
    deadline is checked before each child is priced, so a tight budget
    bounds the overshoot to a single greedy completion.  The first
    complete state found is returned; if none is found the greedy cost
    is the honest fallback.
    """
    start = [frozenset(t) for t in tensors]
    seq = count()
    heap = [(cs.greedy(start, sizes), 0.0, next(seq), start)]
    seen = {_state_key(start)}
    best = float("inf")
    expansions = 0
    complete = 0
    t0 = time.perf_counter()
    hit = False
    while heap and not hit:
        _prio, cost, _seq, ts = heapq.heappop(heap)
        if len(ts) == 1:
            best = min(best, cost)
            complete += 1
            continue
        expansions += 1
        for a, b in _all_pairs(ts):
            if time.perf_counter() - t0 >= budget:
                hit = True
                break
            c = cs.pair_cost(ts[a], ts[b], sizes)
            nts = _merge_ts(ts, a, b)
            k = _state_key(nts)
            if k in seen:
                continue
            seen.add(k)
            nc = cost + c
            heapq.heappush(
                heap,
                (nc + cs.greedy(nts, sizes), nc, next(seq), nts),
            )
    if best == float("inf"):
        best = cs.greedy(tensors, sizes)
    return _Any(best, complete, expansions, time.perf_counter() - t0)


def anytime_policy(
    model: nn.Module,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    budget: float,
    *,
    per_prior: float,
    device: str,
    temperature: float = _TEMP,
    seed: int = 0,
) -> _Any:
    """Sample policy rollouts in lockstep until ``budget`` is spent.

    ``per_prior`` is the measured per-rollout cost (from the decision
    section) used to size the first lockstep batch to a small slice of
    the budget; every later batch is sized from the *actual* elapsed
    time, so the loop fills the remaining budget and stops once the next
    rollout would not fit.  A small first slice keeps the first batch
    from overrunning when the GPU's measured throughput drifts.  Nothing
    is timed outside the budget — the policy gets the same wall-clock as
    every other player.
    """
    torch.manual_seed(seed)
    steps = max(len(tensors) - 1, 0)
    per = max(per_prior, 1e-6)
    batch = max(1, min(_MAX_BATCH, int(0.25 * budget / per)))
    t0 = time.perf_counter()
    best = float("inf")
    rollouts = 0
    while True:
        costs = run_policy_batch(
            model,
            tensors,
            sizes,
            greedy_ref,
            samples=batch,
            greedy=False,
            temperature=temperature,
            device=device,
        )
        rollouts += batch
        best = min(best, min(costs))
        elapsed = time.perf_counter() - t0
        per = elapsed / rollouts
        remaining = budget - elapsed
        if remaining < per:
            break
        batch = max(1, min(_MAX_BATCH, int(remaining / per)))
    return _Any(
        best, rollouts, rollouts * steps, time.perf_counter() - t0
    )


def _warm_policy(
    scales: tuple[int, ...], model: nn.Module, device: str
) -> None:
    """Warm the policy's kernels and allocator across batch shapes.

    A laptop GPU cools between the CPU-bound heuristic players, and a
    fresh batch shape can trigger a one-off allocation, so the first
    timed policy batch is otherwise severalfold slow.  Running the shapes
    the anytime policy will actually use, once, before any board is
    timed, removes that artefact from every player's budget.
    """
    for n in scales:
        tensors, sizes = cs.random_network(n, 0)
        ref = cs.greedy(tensors, sizes)
        for samples in (1, 8, 32, _MAX_BATCH):
            run_policy_batch(
                model,
                tensors,
                sizes,
                ref,
                samples=samples,
                greedy=False,
                temperature=_TEMP,
                device=device,
            )


def _time_players(
    tensors: Any,
    sizes: dict[int, int],
    models: dict[str, nn.Module],
    device: str,
    budget: float,
    seed: int,
    per_prior: float,
) -> dict[str, _Any]:
    """Run every anytime player on one board under a wall-clock budget."""
    ref = cs.greedy(tensors, sizes)
    out = {
        "greedy": anytime_greedy(tensors, sizes, budget),
        "restart": anytime_restart(tensors, sizes, budget, seed=seed),
        "search": anytime_search(tensors, sizes, budget),
    }
    for name, model in models.items():
        out[name] = anytime_policy(
            model,
            tensors,
            sizes,
            ref,
            budget,
            per_prior=per_prior,
            device=device,
            seed=seed,
        )
    return out


def _bench(
    fn: Callable[[], Any], *, reps: int, warm: int, device: str
) -> float:
    """Return the mean wall seconds per call of ``fn`` on ``device``."""
    for _ in range(warm):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) / reps


def _scan_state(
    ts: list[frozenset[int]],
    sizes: dict[int, int],
    pairs: list[tuple[int, int]],
) -> list[float]:
    """Price every pair of one state — the sweep a greedy step pays."""
    return [cs.pair_cost(ts[a], ts[b], sizes) for a, b in pairs]


def _descend(
    game: ContractionGame, target: int, sizes: dict[int, int]
) -> None:
    """Advance ``game`` by cheapest-pair steps down to ``target`` tensors."""
    while len(game.ts) > target:
        _c, a, b = min(
            (cs.pair_cost(game.ts[x], game.ts[y], sizes), x, y)
            for x, y in game.pairs
        )
        game.step(a, b)


def _decision_section(
    scales: tuple[int, ...],
    models: dict[str, nn.Module],
    device: str,
    *,
    reps: int = 200,
) -> dict[int, float]:
    """Measure the per-decision cost of a forward pass vs a scan.

    One mid-game state per scale: the heuristic scan is the Python
    ``pair_cost`` sweep a greedy step pays; the policy forward scores
    every pair of the state in one MLP pass (batch 1 and batch 64).  The
    per-rollout rows compare a full policy rollout against a full
    randomised-greedy episode, the honest unit the two players trade in.
    Returns ``{n: per-rollout seconds}`` for the anytime policy to size
    its first lockstep batch.
    """
    print()
    print("== per-decision cost: policy forward vs heuristic scan ==")
    print(
        f"  {'n':>3} {'pairs':>6} {'scan us':>9} {'fwd1 us':>9} "
        f"{'fwd64 us':>9} {'pol/roll ms':>12} {'restart/roll ms':>15}"
    )
    print("-" * 70)
    model = next(iter(models.values()))
    priors: dict[int, float] = {}
    for n in scales:
        tensors, sizes = cs.random_network(n, 0)
        ref = cs.greedy(tensors, sizes)
        mid = max(len(tensors) // 2, 4)
        g = ContractionGame(tensors, sizes, ref)
        _descend(g, mid, sizes)
        pairs = list(g.pairs)
        scan = _bench(
            functools.partial(_scan_state, g.ts, sizes, pairs),
            reps=reps,
            warm=20,
            device=device,
        )
        sf1, pf1 = _batch_inputs([g], device)
        fwd1 = _bench(
            functools.partial(_logits, model, sf1, pf1),
            reps=reps,
            warm=20,
            device=device,
        )
        games = [
            ContractionGame(tensors, sizes, ref)
            for _ in range(_RESTARTS)
        ]
        for gg in games:
            _descend(gg, mid, sizes)
        sf64, pf64 = _batch_inputs(games, device)
        fwd64 = _bench(
            functools.partial(_logits, model, sf64, pf64),
            reps=max(reps // 4, 20),
            warm=10,
            device=device,
        )
        roll = (
            _bench(
                functools.partial(
                    run_policy_batch,
                    model,
                    tensors,
                    sizes,
                    ref,
                    samples=_RESTARTS,
                    greedy=False,
                    temperature=_TEMP,
                    device=device,
                ),
                reps=5,
                warm=2,
                device=device,
            )
            / _RESTARTS
        )
        rng = random.Random(0)
        rest = _bench(
            functools.partial(
                cs.greedy, tensors, sizes, rng=rng, top_k=3
            ),
            reps=20,
            warm=5,
            device=device,
        )
        print(
            f"  {n:>3} {len(pairs):>6} {1e6 * scan:>9.1f} "
            f"{1e6 * fwd1:>9.1f} {1e6 * fwd64:>9.1f} "
            f"{1e3 * roll:>12.3f} {1e3 * rest:>15.3f}"
        )
        priors[n] = roll
    return priors


def _time_section(
    budgets: tuple[float, ...],
    scales: tuple[int, ...],
    instances: int,
    seed: int,
    models: dict[str, nn.Module],
    device: str,
    priors: dict[int, float],
) -> list[tuple[float, int, dict[str, list[float]], dict[str, _Any]]]:
    """Measure every anytime player at equal wall-clock per instance.

    Per ``(budget, scale)`` the table reports, mean over seeds: the cost
    ratio to that instance's best-found, the number of completed
    rollouts, the number of decisions (action choices / state
    expansions) and the actual wall time — so the reader can see who got
    more rollouts and who paid more per decision.  Returns the raw rows.
    """
    players = ["greedy", "restart", "search", *models]
    rows: list[
        tuple[float, int, dict[str, list[float]], dict[str, _Any]]
    ] = []
    if models:
        _warm_policy(scales, next(iter(models.values())), device)
    for budget in budgets:
        print()
        print(
            "== equal wall-clock: "
            f"{1e3 * budget:.0f} ms per instance =="
        )
        print(
            f"  {'n':>3} {'player':>8} {'ratio-best':>11} "
            f"{'rollouts':>9} {'decisions':>10} {'ms':>8}"
        )
        print("-" * 56)
        for n in scales:
            ratios: dict[str, list[float]] = {p: [] for p in players}
            work: dict[str, list[_Any]] = {p: [] for p in players}
            for k in range(instances):
                tensors, sizes = cs.random_network(n, seed + k)
                found = _time_players(
                    tensors,
                    sizes,
                    models,
                    device,
                    budget,
                    seed + k,
                    priors[n],
                )
                best = min(r.cost for r in found.values())
                for p, r in found.items():
                    ratios[p].append(r.cost / best)
                    work[p].append(r)
            for p in players:
                print(
                    f"  {n:>3} {p:>8} "
                    f"{statistics.fmean(ratios[p]):>11.3f} "
                    f"{statistics.fmean([r.rollouts for r in work[p]]):>9.1f} "
                    f"{statistics.fmean([r.decisions for r in work[p]]):>10.1f} "
                    f"{1e3 * statistics.fmean([r.secs for r in work[p]]):>8.1f}"
                )
            rows.append((budget, n, ratios, work))
    return rows


def _time_pairs(
    rows: list[
        tuple[float, int, dict[str, list[float]], dict[str, _Any]]
    ],
) -> None:
    """Print learned-vs-cheap pairwise ratios at equal wall-clock."""
    print()
    print(
        "== equal wall-clock: pairwise cost ratio "
        "(mean over seeds; <1 = learned wins) =="
    )
    cols: list[str] = []
    for learned in ("rl", "imitation"):
        for base in ("greedy", "restart", "search"):
            cols.append(f"{learned}/{base}")
    head = f"{'ms':>6} {'n':>3} " + " ".join(f"{c:>18}" for c in cols)
    print(head)
    print("-" * len(head))
    for budget, n, ratios, _work in rows:
        cells = []
        for learned in ("rl", "imitation"):
            for base in ("greedy", "restart", "search"):
                lv = ratios.get(learned)
                bv = ratios.get(base)
                if lv and bv:
                    m = statistics.fmean(
                        a / b
                        for a, b in zip(lv, bv, strict=True)
                        if b > 0
                    )
                    cells.append(f"{m:>18.3f}")
                else:
                    cells.append(f"{'-':>18}")
        print(f"{1e3 * budget:>6.0f} {n:>3} " + " ".join(cells))


def _time_verdict(
    rows: list[
        tuple[float, int, dict[str, list[float]], dict[str, _Any]]
    ],
) -> None:
    """Print the honest equal-compute verdict from the measured rows."""
    print()
    print("== equal-compute verdict ==")
    for budget, n, ratios, work in rows:
        cheap = [
            (statistics.fmean(ratios[p]), p)
            for p in ("search", "restart")
            if p in ratios
        ]
        cheap.sort()
        best_cheap, best_name = cheap[0]
        cells = (
            f"  {1e3 * budget:>5.0f}ms n={n:>2}: "
            f"best cheap {best_name} {best_cheap:.3f}"
        )
        for name in ratios:
            if name in ("search", "restart", "greedy"):
                continue
            r = statistics.fmean(ratios[name])
            w = statistics.fmean([x.rollouts for x in work[name]])
            cells += f"  | {name} {r:.3f} ({w:.0f} roll)"
        print(cells)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Train the policies on small n, then run the ladder at scale."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--trainer",
        default="both",
        choices=("rl", "imitation", "both"),
        help="which learned policy to train",
    )
    ap.add_argument("--iterations", type=int, default=1800)
    ap.add_argument("--per-n", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--instances", type=int, default=3)
    ap.add_argument(
        "--train-scales",
        default="8,10,12",
        help="comma-separated n the policies are trained on",
    )
    ap.add_argument(
        "--teacher",
        default="dp",
        choices=("dp", "restart"),
        help="imitation teacher: exact DP (to --dp-max) or restart",
    )
    ap.add_argument(
        "--dp-max",
        type=int,
        default=14,
        help="largest n the imitation DP teacher runs at",
    )
    ap.add_argument(
        "--scales",
        default="20,30,40",
        help="comma-separated n at scale",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="auto|cpu|cuda (auto prefers cuda)",
    )
    ap.add_argument(
        "--mode",
        default="rollout",
        choices=("rollout", "time", "both"),
        help="rollout: match rollout count; time: equal wall-clock",
    )
    ap.add_argument(
        "--budgets",
        default="50,200,1000",
        help="comma-separated per-instance wall-clock budgets (ms)",
    )
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  (torch {torch.__version__})")
    train_ns = tuple(
        int(x) for x in args.train_scales.split(",") if x.strip()
    )
    print(f"train scales: {train_ns}  (teacher: {args.teacher})")

    models: dict[str, nn.Module] = {}
    if args.trainer in ("rl", "both"):
        t0 = time.perf_counter()
        models["rl"] = train_rl(
            ns=train_ns,
            iterations=args.iterations,
            batch=args.batch,
            hidden=args.hidden,
            device=dev,
            seed=args.seed,
            log_every=args.iterations // 3,
        )
        print(f"trained rl in {time.perf_counter() - t0:.1f}s")
    if args.trainer in ("imitation", "both"):
        t0 = time.perf_counter()
        models["imitation"] = train_imitation(
            ns=train_ns,
            per_n=args.per_n,
            epochs=args.epochs,
            hidden=args.hidden,
            device=dev,
            seed=args.seed,
            teacher=args.teacher,
            dp_max=args.dp_max,
        )
        print(f"trained imitation in {time.perf_counter() - t0:.1f}s")

    _control_table((8, 10, 12), args.instances, models, dev)
    scales = tuple(int(x) for x in args.scales.split(",") if x.strip())
    if args.mode in ("time", "both"):
        budgets = tuple(
            1e-3 * float(x)
            for x in args.budgets.split(",")
            if x.strip()
        )
        priors = _decision_section(scales, models, dev)
        rows = _time_section(
            budgets,
            scales,
            args.instances,
            args.seed,
            models,
            dev,
            priors,
        )
        _time_pairs(rows)
        _time_verdict(rows)
    if args.mode in ("rollout", "both"):
        rows = _scale_table(
            scales, args.instances, args.seed, models, dev
        )
        _verdict(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
