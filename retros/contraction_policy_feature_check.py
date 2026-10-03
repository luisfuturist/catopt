"""Assert the vectorised contraction features are bit-identical.

Companion to ``contraction-policy-throughput.md``.  It defines the
*pre-optimisation* scalar feature construction inline (a faithful copy
of the original ``ContractionGame`` feature code) and drives both it and
the shipped vectorised ``ContractionGame`` (from
``tools/contraction_policy.py``) along the *same* action sequence,
asserting the state and pair feature vectors are equal **bit for bit**
on every step, on both instance families and the four real einsums.

Run::

    .venv/bin/python project/retros/contraction_policy_feature_check.py

It is a retro companion, not a CI test: it imports ``tools/`` by path
and is deliberately kept out of the ``tests/`` suite.
"""

from __future__ import annotations

import bisect
import math
import random
import statistics
import sys

sys.path.insert(0, "tools")

import contraction_policy as new  # noqa: E402
import contraction_scale as cs  # noqa: E402


class ScalarGame:
    """The pre-optimisation scalar feature construction (reference)."""

    def __init__(self, tensors, sizes, greedy_ref, n0=None, cost=0.0):
        """Bind the instance and its greedy reference."""
        self.sizes = dict(sizes)
        self.log = {i: math.log2(s) for i, s in sizes.items()}
        self.greedy_ref = float(greedy_ref)
        self.ts = [frozenset(t) for t in tensors]
        self.n0 = int(n0 if n0 is not None else len(self.ts))
        self.cost = float(cost)
        self._refresh()

    @property
    def done(self):
        """Return whether only one tensor remains."""
        return len(self.ts) <= 1

    def _lsize(self, t):
        """Return ``log2`` of the element count of a tensor."""
        return math.fsum(self.log[i] for i in t)

    def _refresh(self):
        """Recompute the cached pair statistics for the current state."""
        self.lsize = [self._lsize(t) for t in self.ts]
        self.ranks = [len(t) for t in self.ts]
        pairs = []
        ul = []
        for a in range(len(self.ts)):
            for b in range(a + 1, len(self.ts)):
                pairs.append((a, b))
                inter = self._lsize(self.ts[a] & self.ts[b])
                ul.append(self.lsize[a] + self.lsize[b] - inter)
        self.pairs = pairs
        self.union_sorted = sorted(ul)
        self.l_min = min(ul) if ul else 0.0
        self.spread = (max(ul) - self.l_min) if ul else 0.0
        self.mean_rank = (
            statistics.fmean(self.ranks) if self.ranks else 0.0
        )

    def state_features(self):
        """Return the scale-free description of the current state."""
        den = 4.0 * self.mean_rank + 1.0
        ranks = self.ranks or [0]
        return [
            len(self.ts) / self.n0,
            self.cost / self.greedy_ref if self.greedy_ref > 0 else 0.0,
            self.mean_rank / 4.0,
            max(ranks) / 4.0,
            min(ranks) / 4.0,
            self.spread / den,
            self.l_min / den,
        ]

    def pair_features(self, a, b):
        """Return the scale-free description of the ``(a, b)`` action."""
        inter = self.ts[a] & self.ts[b]
        l_inter = self._lsize(inter)
        l_union = self.lsize[a] + self.lsize[b] - l_inter
        l_diff = l_union - l_inter
        den = self.spread + 1.0
        n_pairs = max(len(self.pairs), 1)
        pct = bisect.bisect_left(self.union_sorted, l_union) / n_pairs
        return [
            (l_union - self.l_min) / den,
            (self.lsize[a] - self.l_min) / den,
            (self.lsize[b] - self.l_min) / den,
            (l_diff - self.l_min) / den,
            (l_inter - self.l_min) / den,
            self.ranks[a] / 4.0,
            self.ranks[b] / 4.0,
            len(inter) / max(len(self.ts[a] | self.ts[b]), 1),
            pct,
        ]

    def all_pair_features(self):
        """Return one feature vector per legal action."""
        return [self.pair_features(a, b) for a, b in self.pairs]

    def step(self, a, b):
        """Contract pair ``(a, b)``; charge and return its cost."""
        c = cs.pair_cost(self.ts[a], self.ts[b], self.sizes)
        self.cost += c
        self.ts = cs._merge(self.ts, a, b)
        self._refresh()
        return c


def _actions(tensors, seed):
    """Build a valid, seeded contraction-order action sequence."""
    rng = random.Random(seed)
    ts = [frozenset(t) for t in tensors]
    out = []
    while len(ts) > 1:
        m = len(ts)
        a, b = rng.randrange(m), rng.randrange(m)
        while b == a:
            b = rng.randrange(m)
        if a > b:
            a, b = b, a
        out.append((a, b))
        ts = cs._merge(ts, a, b)
    return out


def _drive(mod, tensors, sizes, ref, n0, cost, actions):
    """Roll a game out and record features at every step."""
    g = mod(tensors, sizes, ref, n0=n0, cost=cost)
    seen = []
    for a, b in actions:
        seen.append(
            (g.state_features(), g.all_pair_features(), list(g.pairs))
        )
        g.step(a, b)
    seen.append((g.state_features(), g.all_pair_features(), list(g.pairs)))
    return seen, g.cost


def _check(tensors, sizes, ref, n0, cost, actions, label):
    """Assert both implementations agree bit for bit."""
    o, oc = _drive(ScalarGame, tensors, sizes, ref, n0, cost, actions)
    n, nc = _drive(
        new.ContractionGame, tensors, sizes, ref, n0, cost, actions
    )
    assert oc == nc, (label, "cost", oc, nc)
    for step, (o_row, n_row) in enumerate(zip(o, n, strict=True)):
        assert o_row[2] == n_row[2], (label, step, "pairs order")
        assert o_row[0] == n_row[0], (label, step, "state", o_row[0])
        for k, (ov, nv) in enumerate(zip(o_row[1], n_row[1], strict=True)):
            assert ov == nv, (label, step, "pair", k, ov, nv)


def main():
    """Check both families and the real einsums; print the count."""
    n_ok = 0
    for n in (8, 10, 12, 20, 30, 40):
        for seed in range(40):
            tensors, sizes = cs.random_network(n, seed)
            ref = cs.greedy(tensors, sizes)
            _check(
                tensors, sizes, ref, None, 0.0, _actions(tensors, seed),
                f"hypergraph n={n} s={seed}",
            )
            n_ok += 1
    for n in (10, 20, 40):
        tensors, sizes = cs.chain_network(n)
        ref = cs.greedy(tensors, sizes)
        _check(
            tensors, sizes, ref, None, 0.0, [(0, 1)] * (n - 1),
            f"chain n={n}",
        )
        n_ok += 1
    # non-zero start cost / explicit n0 (imitation-style construction)
    for n in (8, 10):
        for seed in (0, 1, 2):
            tensors, sizes = cs.random_network(n, seed)
            ref = cs.greedy(tensors, sizes)
            _check(
                tensors, sizes, ref, n, 3.5, _actions(tensors, seed + 7),
                f"n0/cost n={n} s={seed}",
            )
            n_ok += 1
    try:
        import contraction_einsum as ce
    except SystemExit:
        print("opt_einsum absent — bond family skipped (uv sync --group "
              "einsum)")
    else:
        for n in (8, 20, 30, 40):
            for seed in range(20):
                tensors, sizes = ce.random_bond_network(n, seed)
                ref = cs.greedy(tensors, sizes)
                _check(
                    tensors, sizes, ref, None, 0.0, _actions(tensors, seed),
                    f"bond n={n} s={seed}",
                )
                n_ok += 1
        for build in (
            ce.attention_scores,
            ce.attention_context,
            ce.bilinear_pool,
            ce.mlp_stack,
        ):
            tensors, sizes = build()
            ref = cs.greedy(tensors, sizes)
            _check(
                tensors, sizes, ref, None, 0.0, [(0, 1)] * (len(tensors) - 1),
                f"real {build.__name__}",
            )
            n_ok += 1
    print(f"OK — {n_ok} instances, features bit-identical")


if __name__ == "__main__":
    main()
