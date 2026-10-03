"""Probe the AZ value head against the exact completion cost.

Plan 0016 follow-up.  ``contraction-az-results.md`` found that PUCT +
a learned value head loses to the plain policy, and — the tell — that
the loss *grows* with budget (1.588 -> 3.316 at n = 40 from 200 ms to
1 s).  More simulations should converge toward the search's best, not
away from it.  The retro's hypothesis: **the learned value is not
accurate enough**, so PUCT's selection concentrates on nodes whose
value is misestimated.

This tool measures that value head directly, *before* any fix, against
the exact ``O(3^n)`` subset DP (``contraction_scale.dp``) on states
drawn from real PUCT games at ``n <= 14`` (where the DP is affordable).

Three questions, in order:

1. **How wrong is ``V``?**  For states from real games, compare
   ``V(s)`` (the predicted normalised remaining cost, inverted from log
   space) against the *true* optimal completion cost ``dp(s)``.  Report
   mean/median absolute error, correlation, and — the metric a search
   actually consumes — the **rank** quality: global Spearman, and the
   **sibling** Spearman (does ``V`` order the successors of a state the
   way ``dp`` does?), plus the top-1 pick accuracy and regret.

2. **How good is the target?**  ``V`` is regressed onto the *achieved*
   cost of a PUCT rollout, not the optimal completion.  Quantify the
   gap: the achieved completion is always ``>= dp``, so the target is
   biased upward and path-noisy.  A value net can be accurate about a
   bad target.

3. **Is the value even better than the analytic critic it replaced?**
   The greedy-completion cost is the hand-designed critic PUCT falls
   back on; if the net does not out-rank it, the learned value is a
   regression.

The baselines reported alongside ``V``: the analytic greedy completion
``cs.greedy(s)`` and the (path-dependent) achieved target.

Usage::

    .venv/bin/python tools/contraction_value_probe.py --device cuda
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import contraction_az as az
import contraction_einsum as ce
import contraction_policy as cp
import contraction_scale as cs
import numpy as np

__all__ = ["main"]

#: The scales probed.  ``dp`` costs ``O(3^k)`` in the *remaining* tensor
#: count ``k``, so the full-board root at ``n = 14`` is ~4 s while every
#: later decision is far cheaper.
_SCALES: tuple[int, ...] = (10, 12, 14)

#: Decision simulations per PUCT episode (matches the trainer default).
_SIMS = 24


# ---------------------------------------------------------------------------
#  Rank statistics (no SciPy in the default env)
# ---------------------------------------------------------------------------


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, or ``nan`` when either side is constant."""
    if x.size < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _avg_rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean position), ascending."""
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via average ranks + Pearson."""
    return _pearson(_avg_rank(x), _avg_rank(y))


def _num(x: float, width: int = 8, fmt: str = ".3f") -> str:
    """Render a float, or ``-`` when it is not finite."""
    if x != x or x in (float("inf"), float("-inf")):
        return f"{'-':>{width}}"
    return f"{format(x, fmt):>{width}}"


# ---------------------------------------------------------------------------
#  States from real PUCT games
# ---------------------------------------------------------------------------


@dataclass
class _State:
    """One decision state sampled from a real PUCT episode."""

    ts: list[frozenset[int]]
    sizes: dict[int, int]
    ref: float
    cost: float
    n0: int
    #: Final cost of the episode this state was sampled from (the
    #: achieved target's denominator is ``episode_cost - cost``).
    episode_cost: float
    #: ``V`` in linear units (predicted remaining / ref).
    v: float = float("nan")


def _snap(game: cp.ContractionGame, episode_cost: float) -> _State:
    """Freeze a live game into a probe sample."""
    return _State(
        ts=list(game.ts),
        sizes=dict(game.sizes),
        ref=game.greedy_ref,
        cost=game.cost,
        n0=game.n0,
        episode_cost=episode_cost,
    )


def _puct_states(
    model: az.DualHeadNet,
    tensors: Any,
    sizes: dict[int, int],
    ref: float,
    *,
    sims: int,
    device: str,
    rng: random.Random,
    value_unit: str = "ref",
) -> list[_State]:
    """Play one PUCT episode; return a snapshot per committed decision.

    Mirrors ``contraction_az.puct_episodes`` for a single instance, but
    also records the *full* tensor multiset at each decision — the DP
    needs it, and the trainer's ``collect`` path drops it.
    """
    root = az._Node(cp.ContractionGame(tensors, sizes, ref), ref)
    az._expand_many(
        model, [root], device, use_net_value=True, value_unit=value_unit
    )
    az._add_noise(root, rng)
    snapshots: list[_State] = []
    node = root
    while not node.terminal:
        for _ in range(sims):
            az._simulate_locked(
                [node], model, device, value_unit=value_unit
            )
        assert node.n is not None and node.priors is not None
        total = float(node.n.sum())
        a = (
            int(np.argmax(node.n))
            if total > 0
            else int(np.argmax(node.priors))
        )
        snapshots.append(_snap(node.game, node.game.cost))
        if a in node.children:
            node = node.children[a]
        else:
            node = az._child(node, a)
            az._expand_many(
                model,
                [node],
                device,
                use_net_value=True,
                value_unit=value_unit,
            )
    final = node.game.cost
    for s in snapshots:
        s.episode_cost = final
    return snapshots


def _game(
    ts: list[frozenset[int]],
    sizes: dict[int, int],
    ref: float,
    cost: float,
    n0: int,
) -> cp.ContractionGame:
    """Rebuild a :class:`ContractionGame` from a frozen state."""
    return cp.ContractionGame(ts, sizes, ref, n0=n0, cost=cost)


def _values(
    model: az.DualHeadNet,
    games: list[cp.ContractionGame],
    device: str,
    *,
    value_unit: str = "ref",
) -> list[float]:
    """Return ``V`` (predicted remaining / ref) for each game, batched."""
    nodes = [az._Node(g, g.greedy_ref) for g in games]
    az._expand_many(
        model, nodes, device, use_net_value=True, value_unit=value_unit
    )
    return [nd.value for nd in nodes]


# ---------------------------------------------------------------------------
#  Exact completion cost and the analytic critic
# ---------------------------------------------------------------------------


_DP_CACHE: dict[tuple, float] = {}


def _dp_key(ts: list[frozenset[int]], sizes: dict[int, int]) -> tuple:
    """Canonical cache key: the sorted index-set multiset + sizes."""
    return (
        tuple(sorted(tuple(sorted(t)) for t in ts)),
        tuple(sorted(sizes.items())),
    )


def _true(ts: list[frozenset[int]], sizes: dict[int, int]) -> float:
    """Exact optimal completion cost of a state (cached subset DP)."""
    if len(ts) <= 1:
        return 0.0
    k = _dp_key(ts, sizes)
    v = _DP_CACHE.get(k)
    if v is None:
        v = cs.dp(ts, sizes)
        _DP_CACHE[k] = v
    return v


# ---------------------------------------------------------------------------
#  The measurement tables
# ---------------------------------------------------------------------------


@dataclass
class _Acc:
    """Accumulator of one predictor's errors against the truth."""

    pred: list[float] = field(default_factory=list)
    truth: list[float] = field(default_factory=list)

    def add(self, pred: float, truth: float) -> None:
        """Record one (prediction, truth) pair."""
        self.pred.append(pred)
        self.truth.append(truth)

    def report(self, label: str, width: int = 16) -> str:
        """Render the error / correlation / rank row for this predictor."""
        p = np.asarray(self.pred, dtype=np.float64)
        t = np.asarray(self.truth, dtype=np.float64)
        lp = np.log1p(np.maximum(p, 0.0))
        lt = np.log1p(np.maximum(t, 0.0))
        ratio = np.where(t > 0, p / np.maximum(t, 1e-30), np.nan)
        finite = ratio[np.isfinite(ratio)]
        abs_log = np.abs(lp - lt)
        sp = _spearman(p, t)
        return (
            f"  {label:>{width}} "
            f"{_num(statistics.fmean(abs_log.tolist()), 9)} "
            f"{_num(statistics.median(abs_log.tolist()), 9)} "
            f"{_num(statistics.fmean(finite.tolist()), 8, '.2f')} "
            f"{_num(statistics.median(finite.tolist()), 8, '.2f')} "
            f"{_num(_pearson(lp, lt), 8)} "
            f"{_num(sp, 8)}"
        )


@dataclass
class _Group:
    """One parent's child set: the games, their ``V`` and their truth."""

    games: list[cp.ContractionGame]
    v: list[float]
    truth: list[float]


def _sibling_groups(
    model: az.DualHeadNet,
    states: list[_State],
    max_k: int,
    device: str,
    *,
    value_unit: str = "ref",
) -> list[_Group]:
    """Build one group per sampled non-terminal state.

    For each sampled state (with ``2 <= k <= max_k`` remaining tensors)
    every legal child is built and priced by the exact DP; the group is
    the child set.  ``max_k`` bounds the child DP (``O(3^(k-1))``).  The
    net's ``V`` is scored for every child in one batched forward per
    parent.
    """
    groups: list[_Group] = []
    for s in states:
        k = len(s.ts)
        if k < 2 or k > max_k:
            continue
        g = _game(s.ts, s.sizes, s.ref, s.cost, s.n0)
        children = []
        for a, b in g.pairs:
            child = g.clone()
            child.step(a, b)
            children.append(child)
        if len(children) < 2:
            continue
        groups.append(
            _Group(
                children,
                _values(model, children, device, value_unit=value_unit),
                [_true(c.ts, c.sizes) for c in children],
            )
        )
    return groups


def _sibling_report(
    groups: list[_Group],
    label: str,
    *,
    predictor: Any = None,
) -> tuple[float, float, float]:
    """Print sibling-rank stats; return ``(mean sp, top1, mean regret)``.

    ``predictor`` maps a child game to its score; the default is the
    net's stored ``V``.
    """
    if predictor is not None:
        for grp in groups:
            grp.v = [predictor(c) for c in grp.games]
    sps: list[float] = []
    top1 = 0
    regrets: list[float] = []
    for grp in groups:
        p = np.asarray(grp.v, dtype=np.float64)
        t = np.asarray(grp.truth, dtype=np.float64)
        sp = _spearman(p, t)
        if sp == sp:
            sps.append(sp)
        best = int(np.argmin(t))
        pick = int(np.argmin(p))
        top1 += int(pick == best)
        if t[best] > 0:
            regrets.append(t[pick] / t[best])
    n = len(groups)
    mean_sp = statistics.fmean(sps) if sps else float("nan")
    med_sp = statistics.median(sps) if sps else float("nan")
    frac_neg = (
        sum(1 for x in sps if x < 0.0) / len(sps)
        if sps
        else float("nan")
    )
    frac_good = (
        sum(1 for x in sps if x >= 0.5) / len(sps)
        if sps
        else float("nan")
    )
    mean_reg = statistics.fmean(regrets) if regrets else float("nan")
    print(
        f"  {label:>16} {n:>6} "
        f"{_num(mean_sp, 9)} {_num(med_sp, 9)} "
        f"{_num(frac_neg, 8)} {_num(frac_good, 8)} "
        f"{top1 / max(n, 1):>9.3f} {_num(mean_reg, 9, '.3f')}"
    )
    return mean_sp, top1 / max(n, 1), mean_reg


# ---------------------------------------------------------------------------
#  The probe
# ---------------------------------------------------------------------------


def probe(
    model: az.DualHeadNet,
    *,
    scales: tuple[int, ...],
    episodes: int,
    sims: int,
    device: str,
    seed: int,
    max_sib_k: int,
    value_unit: str = "ref",
) -> dict[str, Any]:
    """Run the full probe and print the three tables."""
    rng = random.Random(seed)
    states: list[_State] = []
    t0 = time.perf_counter()
    for n in scales:
        eps = max(1, episodes // max(len(scales), 1))
        for _ in range(eps):
            tensors, sizes = ce.random_bond_network(
                n, rng.randrange(1 << 30)
            )
            ref = cs.greedy(tensors, sizes)
            states.extend(
                _puct_states(
                    model,
                    tensors,
                    sizes,
                    ref,
                    sims=sims,
                    device=device,
                    rng=rng,
                    value_unit=value_unit,
                )
            )
    # One batched forward for every sampled state's V.
    games = [_game(s.ts, s.sizes, s.ref, s.cost, s.n0) for s in states]
    vals = _values(model, games, device, value_unit=value_unit)
    for s, v in zip(states, vals, strict=True):
        s.v = v
    print(
        f"sampled {len(states)} decision states from "
        f"{len(scales)} scales in {time.perf_counter() - t0:.1f}s"
    )

    # --- 1. V vs the true completion cost --------------------------------
    print()
    print("== 1. V(s) vs the exact optimal completion cost dp(s) ==")
    print(
        "  pred/true are normalised remaining costs; abs-err is |log1p|"
    )
    print(
        f"  {'predictor':>16} {'abs-err':>9} {'med-err':>9} "
        f"{'ratio':>8} {'med-rat':>8} {'pear-l':>8} {'spear':>8}"
    )
    print("-" * 72)
    accs: dict[str, _Acc] = {
        "net V": _Acc(),
        "greedy": _Acc(),
        "target": _Acc(),
    }
    for s in states:
        true = _true(s.ts, s.sizes)
        true_n = true / s.ref if s.ref > 0 else 0.0
        g = _game(s.ts, s.sizes, s.ref, s.cost, s.n0)
        accs["net V"].add(s.v, true_n)
        accs["greedy"].add(
            cs.greedy(g.ts, g.sizes) / s.ref if s.ref > 0 else 0.0,
            true_n,
        )
        achieved = max(s.episode_cost - s.cost, 0.0)
        accs["target"].add(
            achieved / s.ref if s.ref > 0 else 0.0, true_n
        )
    for label, acc in accs.items():
        print(acc.report(label))
    n2 = len(states)
    within2 = sum(
        1
        for s in states
        if 0.5 <= s.v / max(_true(s.ts, s.sizes) / s.ref, 1e-30) <= 2.0
    )
    print(
        f"  V within 2x of dp: {within2}/{n2} "
        f"({100.0 * within2 / max(n2, 1):.0f}%)"
    )

    # --- 2. the training target's quality --------------------------------
    print()
    print("== 2. the training target (achieved PUCT completion) ==")
    tgt = np.asarray(
        [
            max(s.episode_cost - s.cost, 0.0) / s.ref
            if s.ref > 0
            else 0.0
            for s in states
        ],
        dtype=np.float64,
    )
    tru = np.asarray(
        [
            _true(s.ts, s.sizes) / s.ref if s.ref > 0 else 0.0
            for s in states
        ],
        dtype=np.float64,
    )
    viol = int(np.sum(tgt < tru - 1e-9))
    ratio = tgt / np.maximum(tru, 1e-30)
    fin = ratio[np.isfinite(ratio)]
    print(
        f"  target/dp: mean {statistics.fmean(fin.tolist()):.3f}  "
        f"median {statistics.median(fin.tolist()):.3f}  "
        f"p90 {float(np.percentile(fin, 90)):.3f}  "
        f"max {float(fin.max()):.3f}"
    )
    print(
        f"  target < dp (impossible): {viol}/{len(states)}   "
        f"spearman(target, dp) {_spearman(tgt, tru):.3f}   "
        f"pearson(log) {_pearson(np.log1p(tgt), np.log1p(tru)):.3f}"
    )
    log_ratio = np.log1p(tgt) - np.log1p(tru)
    print(
        f"  target log-bias: mean {statistics.fmean(log_ratio.tolist()):.3f}"
        f"  std {float(log_ratio.std()):.3f}"
    )
    # The normalisation choice, isolated: the *same* label is a near-perfect
    # ranker of dp absolute, and a poor one once divided by the board's
    # (unobservable) greedy reference.
    ach = np.asarray(
        [max(s.episode_cost - s.cost, 0.0) for s in states],
        dtype=np.float64,
    )
    dpl = np.asarray(
        [_true(s.ts, s.sizes) for s in states], dtype=np.float64
    )
    refs = np.asarray([s.ref for s in states], dtype=np.float64)
    print(
        f"  label rank vs dp: absolute {_spearman(np.log1p(ach), dpl):.3f}"
        f"   /ref {_spearman(np.log1p(ach / refs), dpl):.3f}"
        f"   dp abs {_spearman(np.log1p(dpl), dpl):.3f}"
        f"   dp/ref {_spearman(np.log1p(dpl / refs), dpl):.3f}"
    )
    print(
        f"  board ref spread: min {refs.min():.0f}  max {refs.max():.0f}  "
        f"({refs.max() / max(refs.min(), 1.0):.0f}x)"
    )

    # --- 3. sibling ranking (the decision PUCT consumes) -----------------
    print()
    print(
        "== 3. sibling ranking over a state's children (k <= "
        f"{max_sib_k}) =="
    )
    print(
        f"  {'predictor':>16} {'groups':>6} {'mean-sp':>9} "
        f"{'med-sp':>9} {'neg':>8} {'>=.5':>8} {'top1':>9} {'regret':>9}"
    )
    print("-" * 72)
    sib = _sibling_groups(
        model, states, max_sib_k, device, value_unit=value_unit
    )
    _sibling_report(sib, "net V")

    def _critic(c: cp.ContractionGame) -> float:
        return cs.greedy(c.ts, c.sizes) / c.greedy_ref

    _sibling_report(sib, "greedy", predictor=_critic)
    return {"states": len(states), "groups": len(sib)}


def main(argv: list[str] | None = None) -> int:
    """Train the AZ net, then probe its value head against the DP."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--sims", type=int, default=_SIMS)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=120)
    ap.add_argument(
        "--scales",
        default=",".join(str(n) for n in _SCALES),
        help="comma-separated n to sample states from",
    )
    ap.add_argument(
        "--max-sib-k",
        type=int,
        default=9,
        help="largest remaining-tensor count for sibling DP",
    )
    ap.add_argument("--trainer", default="az", choices=("az", "none"))
    ap.add_argument(
        "--value-unit",
        default="ref",
        choices=az._VALUE_UNITS,
        help="value-target scale (see contraction_az._VALUE_UNITS)",
    )
    ap.add_argument("--load", default="", help="load a saved net")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if az.torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    torch = az.torch
    if args.load:
        model = az.DualHeadNet(args.hidden).to(dev)
        model.load_state_dict(torch.load(args.load, map_location=dev))
        model.eval()
        print(f"loaded net from {args.load}")
    else:
        if args.trainer == "none":
            raise SystemExit("--trainer none needs --load")
        t0 = time.perf_counter()
        with ce._on_family(ce.random_bond_network):
            model = az.train_search(
                batch=args.batch,
                iterations=args.iterations,
                sims=args.sims,
                epochs=args.epochs,
                hidden=args.hidden,
                value_unit=args.value_unit,
                device=dev,
                seed=args.seed,
            )
        print(f"trained net in {time.perf_counter() - t0:.1f}s")

    scales = tuple(int(x) for x in args.scales.split(",") if x.strip())
    probe(
        model,
        scales=scales,
        episodes=args.episodes,
        sims=args.sims,
        device=dev,
        seed=args.seed,
        max_sib_k=args.max_sib_k,
        value_unit=args.value_unit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
