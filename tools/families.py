"""Training families for the stage-7 learned / RL search policies.

Plan 0016 stage 7, the multi-family gap: the learned and RL policies
were trained and measured on ONE family (matmul chains).  This module
builds training and held-out data for THREE families and scores a
policy's pick per family.

* ``chain``  — right-seeded matmul chains ``a·(b·c)`` with ``k > m``,
  so the bracketing is the expensive one and ``assoc_matmul`` pays.
* ``dup``    — elementwise duplication ``square(t) + t·t``; the CSE
  law ``square_expand`` rewrites ``square(t)`` to ``t·t``, the two
  spellings unify to one e-class, and the shared product collapses.
* ``linear`` — a stacked ``linear → linear → relu`` block exported from
  a small torch model through
  :class:`catopt_torch.adapters.TorchSource`; ``assoc_linear_bias``
  fuses the two linears into one GEMM.

Torch-free except :func:`linear_programs`, which imports the adapter
lazily.  Ranking matches ``bench/suites/evaluation/policy_value.py``:
1-based *average* ranks, largest true delta first, ties share.
"""

from __future__ import annotations

import random
import statistics
from typing import Any

from catopt_core.ir import Op, TensorType, Var
from catopt_core.trajectories import rule_samples

__all__ = [
    "FAMILIES",
    "chain",
    "chain_programs",
    "dup_programs",
    "family_programs",
    "linear_programs",
    "mixture_programs",
    "ranks_desc",
    "rule_deltas",
    "score_pick",
    "summarize",
]

#: The family names, in report order.
FAMILIES: tuple[str, ...] = ("chain", "dup", "linear")

#: ``(d_in, d_hidden, d_out)`` shapes for the linear family — training.
_LINEAR_TRAIN = ((8, 16, 8), (8, 24, 8), (6, 12, 6), (10, 20, 10))

#: Held-out linear shapes (widths unseen in training).
_LINEAR_HELD = ((7, 18, 7), (9, 21, 9), (12, 26, 12), (11, 22, 11))


def _v(name: str, *shape: int) -> Var:
    """Return a named tensor variable."""
    return Var(name, TensorType(tuple(shape)))


def chain(d: int, k: int, m: int) -> Op:
    """Build the right-seeded chain ``a·(b·c)``.

    ``a`` is ``(d, k)``, ``b`` is ``(k, m)``, ``c`` is ``(m, d)``.
    """
    a, b, c = _v("a", d, k), _v("b", k, m), _v("c", m, d)
    return Op.make("matmul", a, Op.make("matmul", b, c))


def chain_programs(
    n: int, seed: int, lo: int = 2, hi: int = 4
) -> list[Op]:
    """``n`` right-seeded chains ``a·(b·c)`` with ``k > m``.

    The existing one-family training family: right seed, ``k > m`` so
    the bracketing is the expensive one and ``assoc_matmul`` always
    pays.  A guaranteed payoff makes a hit a real find, not a coin
    flip.
    """
    rng = random.Random(seed)
    out: list[Op] = []
    for _ in range(n):
        m = rng.randint(lo, hi - 1)
        k = rng.randint(m + 1, hi)
        out.append(chain(rng.randint(lo, hi), k, m))
    return out


def dup_programs(
    n: int, seed: int, lo: int = 2, hi: int = 4
) -> list[Op]:
    """``n`` elementwise-duplication programs ``square(t) + t·t``.

    ``t`` is a bare variable or ``x·y``; either way ``square_expand``
    unifies the two spellings and the duplicate collapses.
    """
    rng = random.Random(seed)
    out: list[Op] = []
    for _ in range(n):
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if rng.random() < 0.5:
            t: Op = _v("x", a, b)
        else:
            t = Op.make("mul", _v("x", a, b), _v("y", a, b))
        out.append(
            Op.make("add", Op.make("square", t), Op.make("mul", t, t))
        )
    return out


def linear_programs(
    n: int, seed: int, shapes: Any = _LINEAR_TRAIN
) -> list[Op]:
    """``n`` stacked ``linear → linear → relu`` blocks exported to IR."""
    import torch
    from catopt_torch.adapters import TorchSource
    from torch import nn

    class _Block(nn.Module):
        def __init__(self, d_in: int, d_h: int, d_out: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_h)
            self.fc2 = nn.Linear(d_h, d_out)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.fc2(self.fc1(x)))

    progs: list[Op] = []
    for i in range(n):
        torch.manual_seed(seed + i)
        d_in, d_h, d_out = shapes[i % len(shapes)]
        ir, _ = TorchSource().to_ir(
            _Block(d_in, d_h, d_out), torch.randn(4, d_in)
        )
        progs.append(ir.root)
    return progs


def family_programs(
    family: str, n: int, seed: int, *, held_out: bool = False
) -> list[Op]:
    """Build ``n`` programs of ``family`` (training or held-out shapes)."""
    lo, hi = (5, 7) if held_out else (2, 4)
    if family == "chain":
        return chain_programs(n, seed, lo=lo, hi=hi)
    if family == "dup":
        return dup_programs(n, seed, lo=lo, hi=hi)
    if family == "linear":
        return linear_programs(
            n, seed, shapes=_LINEAR_HELD if held_out else _LINEAR_TRAIN
        )
    raise ValueError(f"unknown family {family!r}")


def mixture_programs(
    families: Any, per_family: int, seed: int
) -> list[Op]:
    """Interleave ``per_family`` programs of each family."""
    groups = [
        family_programs(f, per_family, seed + i)
        for i, f in enumerate(families)
    ]
    out: list[Op] = []
    for i in range(per_family):
        for group in groups:
            out.append(group[i])
    return out


def rule_deltas(
    term: Any, rules: Any, cost_fn: Any = None
) -> dict[str, float]:
    """Map each rule name to the cost delta it gives alone on ``term``."""
    return {
        s.rule: s.delta_cost for s in rule_samples(term, rules, cost_fn)
    }


def ranks_desc(values: list[float]) -> list[float]:
    """1-based average ranks, largest value first (ties share)."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while (
            j + 1 < len(values)
            and values[order[j + 1]] == values[order[i]]
        ):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def score_pick(
    term: Any, rules: Any, pick_name: str, cost_fn: Any = None
) -> dict[str, Any]:
    """Rank and hit for ``pick_name`` against ``term``'s true deltas."""
    deltas = rule_deltas(term, rules, cost_fn)
    names = [r.name for r in rules]
    rank_of = dict(
        zip(names, ranks_desc([deltas[n] for n in names]), strict=True)
    )
    best = max(names, key=lambda n: deltas[n])
    return {
        "pick": pick_name,
        "rank": rank_of[pick_name],
        "n_rules": len(rules),
        "delta": deltas[pick_name],
        "hit": deltas[pick_name] > 0.0,
        "best": best,
        "best_delta": deltas[best],
    }


def summarize(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean rank, hit rate and mean delta over per-program scores."""
    n = len(scores)
    return {
        "n": n,
        "mean_rank": statistics.fmean([s["rank"] for s in scores]),
        "hit_rate": sum(1 for s in scores if s["hit"]) / n,
        "mean_delta": statistics.fmean([s["delta"] for s in scores]),
    }
