"""Probe whether new features let a net express oe-greedy's choices.

Plan 0016 follow-up to ``contraction-train-scale.md`` §2.  A supervised
net over the shipped ``scale-free-v1`` features fits ``opt_einsum``'s
deterministic greedy at n = 40 with only ~0.60 held-out top-1 — the
features partially express the strong player's choice.  The mechanism
section showed what oe-greedy does that our features may not see: it
**defers outer-product contractions** (0.0 outer-product steps/order vs
our-greedy's 7.7 at n = 40), and its staged scorer effectively looks one
contraction ahead.

This probe measures whether six cheap candidate features close that
sufficiency gap *before* anything is trained or shipped.  The candidates
are computed on top of the shipped :class:`ContractionGame` as
append-only columns — the package derivation is untouched here; only a
proven lift earns a feature-contract bump.

Candidates (all scale-free):

state:
* ``frac-outer`` — fraction of legal pairs sharing no index (an outer
  product in the making).
* ``nonouter-gap`` — log-cost gap from the cheapest pair to the
  cheapest index-sharing pair, over the spread denominator.  Large
  means "every cheap move is an outer product".

pair:
* ``outer?`` — indicator the pair shares no index.
* ``merged-rank`` — rank of the intermediate the pair would produce.
* ``lookahead`` — log-cost of the merged tensor's cheapest *next*
  contraction (one-step lookahead), spread-normalised like ``ul``.
* ``merged-conn`` — fraction of the remaining tensors the merged
  tensor would still share an index with.
* ``mr-score`` — opt_einsum's *actual* greedy heuristic, the
  memory-removed cost ``size(merged) - size(a) - size(b)``,
  normalised by the pair's union (FLOP) size.  This is the decisive
  read of ``opt_einsum.paths.ssa_greedy_optimize``: the teacher never
  scores union-FLOP cost at all, and only index-sharing pairs are
  ever queued (outer products are a forced fallback).
* ``mr-rank`` — quantile rank of ``mr-score`` among the legal pairs
  (the analogue of shipped feature 8 for the teacher's own metric).
* ``pos-a`` / ``pos-b`` — the tensors' normalised positions.  The
  teacher tie-breaks equal scores on ssa ids, which grow in creation
  order — and our list positions track creation order (merges append).

The methodology is the retro's §2 unchanged: replay oe-greedy's order
into ``(state, action)`` samples at n = 40 (12 train / 12 held-out
bond boards), fit the same MLP shape at masked cross-entropy, report
held-out top-1 — for the v1 column subset, the full v2 row, and v2
minus each new column (per-feature ablation).

``opt_einsum`` lives in the opt-in ``einsum`` dependency group; the
tool exits with a sync hint when the group is absent.

Usage::

    uv sync --group einsum
    .venv/bin/python tools/contraction_feature_probe.py --device cuda
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import contraction_einsum as ce
import contraction_policy as cp
import contraction_scale as cs
import numpy as np
import torch
from catopt_torch.contraction_policy import (
    PAIR_DIM,
    STATE_DIM,
    ContractionGame,
)
from torch import nn
from torch.nn import functional as F

__all__ = ["main"]

#: New state features appended after the shipped STATE_DIM.
_EXT_STATE = ("frac-outer", "nonouter-gap")

#: New pair features appended after the shipped PAIR_DIM.
_EXT_PAIR = (
    "outer?",
    "merged-rank",
    "lookahead",
    "merged-conn",
    "mr-score",
    "mr-rank",
    "pos-a",
    "pos-b",
)

#: The second-round features aimed at the teacher's real scoring.
_MR_PAIR = ("mr-score", "mr-rank", "pos-a", "pos-b")

#: Full row width: state (v1 + ext) concatenated with pair (v1 + ext).
_S_FULL = STATE_DIM + len(_EXT_STATE)
_P_FULL = PAIR_DIM + len(_EXT_PAIR)


def _pair_cols(*names: str) -> tuple[int, ...]:
    """Return shipped indices for the named ext pair features."""
    return tuple(PAIR_DIM + _EXT_PAIR.index(n) for n in names)


#: Feature-set specs: (state column idx, pair column idx) pairs.
_SPECS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "v1": (tuple(range(STATE_DIM)), tuple(range(PAIR_DIM))),
    "v2": (
        tuple(range(_S_FULL)),
        tuple(range(PAIR_DIM + 4)),
    ),
    "v3": (
        tuple(range(STATE_DIM)),
        tuple(range(PAIR_DIM)) + _pair_cols(*_MR_PAIR),
    ),
    "v4": (tuple(range(_S_FULL)), tuple(range(_P_FULL))),
}
for _i, _name in enumerate(_EXT_STATE):
    _SPECS[f"no-{_name}"] = (
        tuple(j for j in range(_S_FULL) if j != STATE_DIM + _i),
        tuple(range(_P_FULL)),
    )
for _i, _name in enumerate(_EXT_PAIR):
    _SPECS[f"no-{_name}"] = (
        tuple(range(_S_FULL)),
        tuple(j for j in range(_P_FULL) if j != PAIR_DIM + _i),
    )


# ---------------------------------------------------------------------------
#  The candidate features, computed on the shipped game's caches
# ---------------------------------------------------------------------------


def _ext_state(g: ContractionGame) -> list[float]:
    """Return the two candidate state features for ``g``'s state."""
    ul = g._ul
    if ul.size == 0:
        return [0.0, 0.0]
    outer = g._cnt_u == 0
    non = ~outer
    gap = float(ul[non].min() - g.l_min) if non.any() else g.spread
    return [float(outer.mean()), gap / (g.spread + 1.0)]


def _prod(t: frozenset[int], g: ContractionGame) -> int:
    """Return the exact integer footprint of an index set."""
    s = 1
    for i in t:
        s *= g.sizes[i]
    return s


def _ext_pairs(g: ContractionGame) -> np.ndarray:
    """Return the ``[n_pairs, len(_EXT_PAIR)]`` candidate matrix.

    ``lookahead`` is the merged tensor's cheapest next union cost and
    ``merged-conn`` its connectivity; both need the merged index set
    intersected with every remaining tensor, so this is one
    ``O(n_pairs * m)`` bitmask loop — the same bound the shipped
    incremental ``_advance`` pays per step.  ``mr-score`` is computed
    in log space as ``2^(l12-ul) - 2^(l1-ul) - 2^(l2-ul)`` — every
    exponent is ``<= 0`` (``ul`` is the largest of the three sizes),
    so nothing overflows and the value lands in ``[-2, 1]`` with outer
    products pinned near ``+1`` (their merged size *is* the union).
    ``mr-rank`` ranks the *exact integer* memory-removed cost with the
    teacher's own heap tie-break ``(cost, id2, id1)`` — our list
    positions track its ssa ids because merges append at the end.
    """
    masks = g._masks
    ls = g._ls
    rk = g._rk
    m = len(masks)
    den = g.spread + 1.0
    n_pairs = len(g.pairs)
    out = np.empty((n_pairs, len(_EXT_PAIR)), dtype=np.float64)
    raw_mr: list[int] = []
    for p, (a, b) in enumerate(g.pairs):
        shared = int(g._icount[a, b])
        merged = masks[a] ^ masks[b]
        ml = g._lsize_mask(merged)
        ul = g._ul[p]
        best = math.inf
        conn = 0
        for j in range(m):
            if j in (a, b):
                continue
            both = (masks[a] & masks[j]) ^ (masks[b] & masks[j])
            c = ml + ls[j] - g._lsize_mask(both)
            if c < best:
                best = c
            if both:
                conn += 1
        if best is math.inf:  # m == 2: the merge ends the game
            best = ml
        raw_mr.append(
            _prod(g.ts[a] ^ g.ts[b], g)
            - _prod(g.ts[a], g)
            - _prod(g.ts[b], g)
        )
        out[p, :4] = (
            1.0 if shared == 0 else 0.0,
            (rk[a] + rk[b] - 2.0 * shared) / 4.0,
            (best - g.l_min) / den,
            conn / max(m - 2, 1),
        )
        out[p, 4] = (
            2.0 ** (ml - ul) - 2.0 ** (ls[a] - ul) - 2.0 ** (ls[b] - ul)
        )
        out[p, 6] = (a + 1) / m
        out[p, 7] = (b + 1) / m
    # Rank by the teacher's heap key (raw mr, id2, id1); positions
    # stand in for ssa ids.
    order = sorted(
        range(n_pairs),
        key=lambda p: (raw_mr[p], g.pairs[p][1], g.pairs[p][0]),
    )
    for rank, p in enumerate(order):
        out[p, 5] = rank / max(n_pairs, 1)
    return out


def _rows(
    g: ContractionGame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(state_vec[9], pair_mat[n_pairs, 13])`` — v1 + ext."""
    sf = np.asarray(g.state_features() + _ext_state(g))
    pf = np.concatenate(
        [g.pair_feature_matrix(), _ext_pairs(g)], axis=1
    )
    return sf, pf


# ---------------------------------------------------------------------------
#  The sufficiency fit (retro §2 methodology, column-sliced)
# ---------------------------------------------------------------------------


class _Net(nn.Module):
    """The ``PairPolicyNet`` shape at a variable input width."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, width: int, hidden: int) -> None:
        """Build the MLP over ``width`` inputs."""
        super().__init__()
        self.register_buffer("mean", torch.zeros(width))
        self.register_buffer("std", torch.ones(width))
        self.net = nn.Sequential(
            nn.Linear(width, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one logit per row of ``x``, shape ``[n_rows]``."""
        return self.net((x - self.mean) / self.std).squeeze(-1)


def _encode(
    samples: list[tuple[ContractionGame, tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Pad samples into ``(state[9], pairs[P,13], mask, target)``."""
    width = max(len(g.pairs) for g, _ in samples)
    sf = np.zeros((len(samples), _S_FULL))
    pf = np.zeros((len(samples), width, _P_FULL))
    mask = np.zeros((len(samples), width))
    target: list[int] = []
    for i, (g, action) in enumerate(samples):
        s, p = _rows(g)
        sf[i] = s
        pf[i, : p.shape[0]] = p
        mask[i, : p.shape[0]] = 1.0
        target.append(g.pairs.index(action))
    return sf, pf, mask, target


def _slice(
    sf: np.ndarray,
    pf: np.ndarray,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
) -> np.ndarray:
    """Expand state columns over the pair axis; select pair columns."""
    sc, pc = spec
    n, p, _k = pf.shape
    s = np.broadcast_to(sf[:, list(sc)][:, None, :], (n, p, len(sc)))
    return np.concatenate([s, pf[:, :, list(pc)]], axis=2)


def _fit_spec(
    name: str,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
    train: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    test: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    epochs: int,
    hidden: int,
    rng: random.Random,
    device: str,
) -> tuple[nn.Module, float, float]:
    """Fit one feature spec; return ``(model, train_top1, held_top1)``."""
    width = len(spec[0]) + len(spec[1])
    x_tr = torch.as_tensor(
        _slice(*train[:2], spec), dtype=torch.float32, device=device
    )
    mask_tr = torch.as_tensor(
        train[2], dtype=torch.float32, device=device
    )
    y_tr = torch.tensor(train[3], device=device)
    model = _Net(width, hidden).to(device)
    flat = x_tr[mask_tr.bool()]
    with torch.no_grad():
        model.mean.copy_(flat.mean(0))
        model.std.copy_(flat.std(0).clamp_min(0.1))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    idx = list(range(x_tr.shape[0]))
    for _ in range(epochs):
        rng.shuffle(idx)
        for start in range(0, len(idx), 64):
            sel = idx[start : start + 64]
            logits = (
                model(x_tr[sel].reshape(-1, width))
                .reshape(len(sel), -1)
                .masked_fill(mask_tr[sel] == 0, -1e9)
            )
            loss = F.cross_entropy(logits, y_tr[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return (
        model,
        _top1(model, spec, train, device),
        _top1(model, spec, test, device),
    )


def _top1(
    model: nn.Module,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
    data: tuple[np.ndarray, np.ndarray, np.ndarray, list[int]],
    device: str,
) -> float:
    """Masked top-1 accuracy of ``model`` on an encoded dataset."""
    width = len(spec[0]) + len(spec[1])
    x = torch.as_tensor(
        _slice(*data[:2], spec), dtype=torch.float32, device=device
    )
    mask = torch.as_tensor(data[2], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = (
            model(x.reshape(-1, width))
            .reshape(x.shape[0], -1)
            .masked_fill(mask == 0, -1e9)
        )
        pred = torch.argmax(logits, dim=1)
    hits = sum(
        int(p) == t for p, t in zip(pred.tolist(), data[3], strict=True)
    )
    return hits / max(len(data[3]), 1)


def _rollout_vs_teacher(
    model: nn.Module,
    spec: tuple[tuple[int, ...], tuple[int, ...]],
    boards: list[Any],
    teacher: list[list[tuple[int, int]]],
    device: str,
) -> float:
    """Mean argmax-rollout cost / teacher cost on held-out boards."""
    vals = []
    for (tensors, sizes), order in zip(boards, teacher, strict=True):
        g = ContractionGame(tensors, sizes, cs.greedy(tensors, sizes))
        while not g.done:
            sf, pf = _rows(g)
            sc, pc = spec
            x = torch.as_tensor(
                np.concatenate(
                    [
                        np.broadcast_to(
                            sf[list(sc)], (pf.shape[0], len(sc))
                        ),
                        pf[:, list(pc)],
                    ],
                    axis=1,
                ),
                dtype=torch.float32,
                device=device,
            )
            with torch.no_grad():
                i = int(torch.argmax(model(x)))
            g.step(*g.pairs[i])
        teacher_cost = ce.our_cost_of_order(tensors, sizes, order)
        vals.append(g.cost / teacher_cost)
    return statistics.fmean(vals)


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _datasets(
    n: int,
    boards: int,
    holdout: int,
    seed: int,
    teacher: str,
) -> tuple[Any, Any, list[Any], list[Any]]:
    """Build train/held-out datasets and the held-out teacher orders."""
    tr_b = [ce.random_bond_network(n, seed + k) for k in range(boards)]
    te_b = [
        ce.random_bond_network(n, seed + 10_000 + k)
        for k in range(holdout)
    ]
    order_fn = (
        ce.oe_greedy_order
        if teacher == "oe-greedy"
        else lambda e: ce.oe_random_greedy_order(e, 1.0)
    )

    def _samples(bds: list[Any]) -> list[Any]:
        out: list[Any] = []
        for tensors, sizes in bds:
            order = order_fn(ce.to_einsum(tensors, sizes))
            out.extend(
                cp._replay_samples(tensors, sizes, order, len(tensors))
            )
        return out

    te_orders = [order_fn(ce.to_einsum(t, s)) for t, s in te_b]
    return _samples(tr_b), _samples(te_b), te_b, te_orders


def _teacher_profile(samples: list[Any]) -> None:
    """Print the teacher's outer-product usage for context."""
    outer = 0
    for g, action in samples:
        a, b = action
        if not (g.ts[a] & g.ts[b]):
            outer += 1
    print(
        f"  teacher outer-product steps: {outer}/{len(samples)} "
        f"({outer / max(len(samples), 1):.3f})"
    )


def _e2e(
    path: str | None,
    scales: tuple[int, ...],
    instances: int,
    budgets: tuple[float, ...],
    seed: int,
    device: str,
) -> None:
    """Run the equal-wall-clock ladder for a saved policy artifact.

    Boards and protocol are exactly ``contraction_einsum._measure``'s —
    every player gets ``budget`` seconds per instance and each order is
    scored under both cost models.  ``path=None`` loads the bundled
    artifact (valid only while it matches this checkout's feature
    contract).
    """
    from catopt_torch.contraction_policy import load_contraction_policy

    model = (
        load_contraction_policy(path, device=device).model
        if path
        else load_contraction_policy(device=device).model
    )
    name = Path(path).stem if path else "learned"
    models = {name: model}
    data = ce._measure(budgets, scales, instances, seed, models, device)
    ce._ladder_table(data, models)
    ce._pairwise_table(data, models)


def main(argv: list[str] | None = None) -> int:
    """Fit every feature spec on the oe-greedy teacher; report top-1."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scales", type=int, default=40)
    ap.add_argument("--boards", type=int, default=12)
    ap.add_argument("--holdout", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--teacher", default="oe-greedy")
    ap.add_argument(
        "--e2e",
        metavar="POLICY.pt|bundled",
        nargs="?",
        const="bundled",
        default="",
        help="skip the fit; run the equal-budget ladder for a policy",
    )
    ap.add_argument("--e2e-scales", default="20,30,40")
    ap.add_argument("--e2e-budgets", default="200", help="ms")
    ap.add_argument("--instances", type=int, default=3)
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.e2e:
        path = None if args.e2e == "bundled" else args.e2e
        _e2e(
            path,
            tuple(
                int(x) for x in args.e2e_scales.split(",") if x.strip()
            ),
            args.instances,
            tuple(
                1e-3 * float(x)
                for x in args.e2e_budgets.split(",")
                if x.strip()
            ),
            args.seed,
            dev,
        )
        return 0
    print(f"device: {dev}  (torch {torch.__version__})")
    print(f"specs: {', '.join(_SPECS)}")

    t0 = time.perf_counter()
    tr_s, te_s, te_boards, te_orders = _datasets(
        args.scales,
        args.boards,
        args.holdout,
        args.seed,
        args.teacher,
    )
    print(
        f"datasets: {len(tr_s)} train / {len(te_s)} held-out states "
        f"(n={args.scales}, teacher={args.teacher}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    _teacher_profile(tr_s)

    t0 = time.perf_counter()
    train = _encode(tr_s)
    test = _encode(te_s)
    print(f"encoded in {time.perf_counter() - t0:.1f}s")

    print()
    print(
        f"== feature sufficiency vs {args.teacher} (n={args.scales}) =="
    )
    print(f"  {'spec':>16} {'width':>6} {'train':>8} {'held-out':>9}")
    print("-" * 43)
    rng = random.Random(args.seed)
    fitted: dict[str, nn.Module] = {}
    results: dict[str, tuple[float, float]] = {}
    for name, spec in _SPECS.items():
        torch.manual_seed(args.seed)
        model, tr1, te1 = _fit_spec(
            name, spec, train, test, args.epochs, args.hidden, rng, dev
        )
        fitted[name] = model
        results[name] = (tr1, te1)
        print(
            f"  {name:>16} {len(spec[0]) + len(spec[1]):>6} "
            f"{tr1:>8.3f} {te1:>9.3f}"
        )

    print()
    print("== argmax rollout / teacher order on held-out boards ==")
    print("  (the fit is only useful if it prices a better order)")
    for name in ("v1", "v2", "v3", "v4"):
        roll = _rollout_vs_teacher(
            fitted[name], _SPECS[name], te_boards, te_orders, dev
        )
        print(f"  {name:>16} {roll:>9.3f}")
    print()
    for name in ("v2", "v3", "v4"):
        print(
            f"  held-out delta {name}-v1: "
            f"{results[name][1] - results['v1'][1]:+.3f} "
            "(ship bar: +0.05)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
