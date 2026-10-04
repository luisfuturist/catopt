"""A pretrained contraction-ordering player — shipped weights + loader.

The contraction thread's headline is
``project/retros/contraction-synthesis.md``: a policy **distilled on
every trial trajectory** of ``opt_einsum``'s ``RandomGreedy`` teacher
(the ``oe-all`` arm of ``tools/contraction_distill.py``), driven under
the lockstep/affine guided-restart protocol, **beats the teacher at
equal wall-clock at n = 40** — 0.89-1.01 oe-cost ratios across three
seeds at fed budgets, 0.89-0.98 at the best temperature — the first
outright win over the field's best cheap player.  This module ships
that player: the contraction game it runs on, the net, a usable
:class:`ContractionPolicy` wrapper, a lazy
:func:`load_contraction_policy`, and two bundled weight artifacts under
``artifacts/`` — ``contraction_policy_distilled.pt`` (the default — the
oe-all distilled player) and ``contraction_policy_curriculum.pt`` (the
earlier REINFORCE curriculum weights, kept as lineage and loadable via
an explicit path).  The weights are the only artifacts; training stays
a tools concern (``tools/contraction_distill.py`` for the default,
``tools/train_contraction_artifact.py`` for the lineage weights).

**Honest capability statement.**  The default weights were trained on
the :func:`random_bond_network` einsum-valid instance family (every
index a bond or an open leg) at scales 8-24 by masked cross-entropy
over the teacher's trial trajectories.  The player is a *sampler*:
:meth:`ContractionPolicy.best_order` — best-of-N temperature-sampled
rollouts — is the measured winner; the single argmax pass
(:meth:`ContractionPolicy.order`) is a plausible fallback, not the
player (brittle at n = 40, reads 0.9-1.3x ``oe-greedy``).  The player
is throughput-starved under ~50 ms per-instance budgets, where the
affine prior can starve it to one episode.  It generalises to unseen
boards of the same family; other tensor-network distributions are out
of scope.  This is a research artifact behind a documented API —
**nothing wires it into ``Optimizer.optimize``**: the pipeline has no
contraction-ordering hook for it (the contraction diagram search in
``catopt_orchestrator.diagram`` is a different game).

**Game contract.**  A board is ``(tensors, sizes)``: each tensor a
tuple of index labels (a hyperedge), ``sizes`` mapping each label to
its extent.  An action contracts one pair; the move cost is
``prod sizes`` over the union of the pair's index sets — the classic
pairwise contraction cost.  An order is a list of ``(a, b)`` index
pairs into the *current* tensor list; :func:`cost_of_order` replays it.

The feature derivation is **pinned**: the weights are only meaningful
against the exact scale-free features below (state ``STATE_DIM`` +
pair ``PAIR_DIM``), so :class:`ContractionGame` must not change.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

__all__ = [
    "MAX_BATCH",
    "PAIR_DIM",
    "SAMPLE_TEMP",
    "STATE_DIM",
    "STD_FLOOR",
    "ContractionGame",
    "ContractionPolicy",
    "PairPolicyNet",
    "batch_inputs",
    "cost_of_order",
    "greedy",
    "load_contraction_policy",
    "merge_tensors",
    "pair_cost",
    "pair_logits",
    "random_bond_network",
    "rollout_orders",
    "run_policy_batch",
    "save_contraction_policy",
]

#: State feature width.
STATE_DIM = 7

#: Pair (action) feature width.
PAIR_DIM = 9

#: Feature-normalisation floor — stops an all-constant column from
#: exploding when standardised.  Used by the training-side norm fit.
STD_FLOOR = 0.1

#: Largest lockstep batch sampled rollouts run in one forward pass.
#: The vectorised lockstep driver costs one fixed burst of kernel
#: launches per step whatever the batch, so the cap is a memory
#: guard, not a sweet spot.
MAX_BATCH = 512

#: Sampling temperature for the sampled (restart) rollouts.
SAMPLE_TEMP = 1.5

#: Artifact payload format tag; the loader rejects anything else.
_FORMAT = "catopt-contraction-policy/1"

#: Feature contract version — bump if the feature derivation changes.
_FEATURE_CONTRACT = "scale-free-v1"

#: Bundled default weights — the oe-all distilled player, relative to
#: this package (``project/retros/contraction-synthesis.md``).  The
#: earlier REINFORCE curriculum weights remain bundled alongside as
#: ``contraction_policy_curriculum.pt``, loadable via an explicit path.
_DEFAULT_NAME = "contraction_policy_distilled.pt"


# ---------------------------------------------------------------------------
#  Boards: costs, merge, the einsum-valid training family
# ---------------------------------------------------------------------------


def pair_cost(
    x: frozenset[int], y: frozenset[int], sizes: dict[int, int]
) -> float:
    """Scalar multiplies to contract two tensors over index sets."""
    c = 1.0
    for i in x | y:
        c *= sizes[i]
    return c


def _pairs(ts: list[frozenset[int]]) -> Iterator[tuple[int, int]]:
    """Yield every unordered pair of tensor positions."""
    for a in range(len(ts)):
        for b in range(a + 1, len(ts)):
            yield a, b


def merge_tensors(
    ts: list[frozenset[int]], a: int, b: int
) -> list[frozenset[int]]:
    """Replace tensors ``a``/``b`` by their contraction (xor of sets)."""
    merged = ts[a] ^ ts[b]
    return [t for k, t in enumerate(ts) if k not in (a, b)] + [merged]


def greedy(
    tensors: Any,
    sizes: dict[int, int],
    *,
    rng: random.Random | None = None,
    top_k: int = 1,
) -> float:
    """Contract the cheapest pair; ``rng`` randomises among the top-k."""
    ts = [frozenset(t) for t in tensors]
    total = 0.0
    while len(ts) > 1:
        cands = sorted(
            (pair_cost(ts[a], ts[b], sizes), a, b)
            for a, b in _pairs(ts)
        )
        if rng is None:
            c, a, b = cands[0]
        else:
            c, a, b = cands[rng.randrange(min(top_k, len(cands)))]
        total += c
        ts = merge_tensors(ts, a, b)
    return total


def cost_of_order(
    tensors: Any, sizes: dict[int, int], order: list[tuple[int, int]]
) -> float:
    """Replay an index-pair order and sum the pairwise cost."""
    ts = [frozenset(t) for t in tensors]
    total = 0.0
    for a, b in order:
        total += pair_cost(ts[a], ts[b], sizes)
        ts = merge_tensors(ts, a, b)
    return total


#: Maximum tensor rank the bond-network generator will build.
_DEGREE_CAP = 4

#: Probability a bond-network tensor also carries an open (output) leg.
_OPEN_LEG_P = 0.2


def _new_bond(
    sizes: dict[int, int], rng: random.Random, label: int
) -> int:
    """Register a fresh index ``label`` with a random extent."""
    sizes[label] = rng.randint(2, 8)
    return label


def _extra_bonds(
    tensors: list[set[int]],
    sizes: dict[int, int],
    rng: random.Random,
    nxt: int,
) -> int:
    """Add ``n // 4`` extra bonds (cycles) under the degree cap."""
    n = len(tensors)
    extra = max(1, n // 4)
    added = 0
    tries = 0
    while added < extra and tries < 50 * extra + 50:
        tries += 1
        i = rng.randrange(n)
        j = rng.randrange(n)
        if (
            i == j
            or (tensors[i] & tensors[j])
            or len(tensors[i]) >= _DEGREE_CAP
            or len(tensors[j]) >= _DEGREE_CAP
        ):
            continue
        bond = _new_bond(sizes, rng, nxt)
        nxt += 1
        tensors[i].add(bond)
        tensors[j].add(bond)
        added += 1
    return nxt


def _open_legs(
    tensors: list[set[int]],
    sizes: dict[int, int],
    rng: random.Random,
    nxt: int,
) -> int:
    """Give a fraction of tensors an extra open (output) leg."""
    for t in tensors:
        if rng.random() < _OPEN_LEG_P and len(t) < _DEGREE_CAP:
            t.add(_new_bond(sizes, rng, nxt))
            nxt += 1
    return nxt


def random_bond_network(
    n: int, seed: int
) -> tuple[tuple[tuple[int, ...], ...], dict[int, int]]:
    """Seeded random tensor network with every index shared <= 2 times.

    A spanning tree guarantees connectivity, a few extra bonds add
    cycles, and a fraction of tensors carry an open leg.  Every index
    therefore appears in exactly two tensors (a *bond*) or exactly one
    (an *open* leg) — the precondition for a faithful einsum mapping,
    and the family the bundled weights were trained on.
    """
    rng = random.Random(seed)
    tensors: list[set[int]] = [set() for _ in range(n)]
    sizes: dict[int, int] = {}
    nxt = 0
    for i in range(1, n):
        room = [j for j in range(i) if len(tensors[j]) < _DEGREE_CAP]
        j = rng.choice(room) if room else rng.randrange(i)
        bond = _new_bond(sizes, rng, nxt)
        nxt += 1
        tensors[i].add(bond)
        tensors[j].add(bond)
    nxt = _extra_bonds(tensors, sizes, rng, nxt)
    _open_legs(tensors, sizes, rng, nxt)
    return tuple(tuple(sorted(t)) for t in tensors), sizes


# ---------------------------------------------------------------------------
#  The contraction game: state, action, reward
# ---------------------------------------------------------------------------


def _bits(mk: int) -> Iterator[int]:
    """Yield the set bit positions of a non-negative mask, low to high."""
    while mk:
        low = mk & -mk
        yield low.bit_length() - 1
        mk ^= low


#: Cached strict-upper-triangle index arrays, keyed by matrix size.
_TRIU: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _triu(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cached ``(row, col)`` index arrays of the upper triangle."""
    r = _TRIU.get(m)
    if r is None:
        r = np.triu_indices(m, 1)
        _TRIU[m] = r
    return r


class ContractionGame:
    """One contraction episode: the remaining tensors and the cost so far.

    The state is the list of remaining tensors; the legal actions are
    the unordered index pairs; ``step`` contracts one pair and charges
    the classic pairwise cost.  ``greedy_ref`` (the full-network greedy
    cost) anchors the scale-free features.

    Every tensor is mirrored as an integer bitmask over the instance's
    index labels, and the pairwise quantities the features need — the
    ``log2`` intersection size and the intersection cardinality of each
    tensor pair — are maintained **incrementally** across steps: a
    contraction recomputes only the row/column of the freshly merged
    tensor, not the whole ``O(m^2)`` table.  The feature matrices are
    then assembled with NumPy in one shot.  The values are exactly the
    ``math.fsum`` exactly-rounded sums the scalar implementation used
    (``fsum`` is order-independent), so the feature vectors are
    bit-identical and a policy trained on the old features stays valid;
    only the cost changes.
    """

    def __init__(
        self,
        tensors: Any,
        sizes: dict[int, int],
        greedy_ref: float,
        n0: int | None = None,
        cost: float = 0.0,
    ) -> None:
        """Bind the instance, its greedy reference, and a start cost."""
        self.sizes = dict(sizes)
        self.log = {i: math.log2(s) for i, s in sizes.items()}
        self.greedy_ref = float(greedy_ref)
        self.ts = [frozenset(t) for t in tensors]
        self.n0 = int(n0 if n0 is not None else len(self.ts))
        self.cost = float(cost)
        labels = sorted({i for t in self.ts for i in t})
        self._bit = {lab: p for p, lab in enumerate(labels)}
        self._log_bit = [self.log[lab] for lab in labels]
        self._lsize_cache: dict[int, float] = {}
        self._rebuild()

    @property
    def done(self) -> bool:
        """Return whether only one tensor remains."""
        return len(self.ts) <= 1

    def clone(self) -> ContractionGame:
        """Return an independent copy that shares the size caches.

        The pairwise intersection tables are copied, so a ``step`` on
        the clone cannot touch the original.  A tree search expands a
        child per simulation this way, paying one incremental ``step``
        instead of a full ``O(m^2)`` rebuild — the feature values are
        unchanged, only the construction cost.
        """
        g = ContractionGame.__new__(ContractionGame)
        g.sizes = self.sizes
        g.log = self.log
        g.greedy_ref = self.greedy_ref
        g.n0 = self.n0
        g.cost = self.cost
        g._bit = self._bit
        g._log_bit = self._log_bit
        g._lsize_cache = self._lsize_cache
        g.ts = list(self.ts)
        g._masks = list(self._masks)
        g._lsize = list(self._lsize)
        g._ranks = list(self._ranks)
        g._inter = self._inter.copy()
        g._icount = self._icount.copy()
        g._refresh()
        return g

    def _mask_of(self, t: frozenset[int]) -> int:
        """Return the bitmask of an index set."""
        mk = 0
        for lab in t:
            mk |= 1 << self._bit[lab]
        return mk

    def _lsize_mask(self, mk: int) -> float:
        """Return the exactly-rounded ``log2`` volume of a bitmask."""
        r = self._lsize_cache.get(mk)
        if r is None:
            r = math.fsum(self._log_bit[p] for p in _bits(mk))
            self._lsize_cache[mk] = r
        return r

    def _rebuild(self) -> None:
        """Build the mask and pairwise caches from the tensor list."""
        self._masks = [self._mask_of(t) for t in self.ts]
        self._lsize = [self._lsize_mask(mk) for mk in self._masks]
        self._ranks = [len(t) for t in self.ts]
        m = len(self._masks)
        inter = np.zeros((m, m), dtype=np.float64)
        count = np.zeros((m, m), dtype=np.int64)
        for i in range(m):
            inter[i, i] = self._lsize[i]
            count[i, i] = self._ranks[i]
            for j in range(i + 1, m):
                both = self._masks[i] & self._masks[j]
                v = self._lsize_mask(both)
                inter[i, j] = inter[j, i] = v
                c = both.bit_count()
                count[i, j] = count[j, i] = c
        self._inter = inter
        self._icount = count
        self._refresh()

    def _refresh(self) -> None:
        """Recompute the cached state summary for the current state."""
        self.lsize = self._lsize
        self.ranks = self._ranks
        m = len(self._masks)
        ia, ib = _triu(m)
        self._ia = ia
        self._ib = ib
        self.pairs = list(zip(ia.tolist(), ib.tolist(), strict=True))
        self._ls = np.asarray(self._lsize, dtype=np.float64)
        self._rk = np.asarray(self._ranks, dtype=np.float64)
        inter = self._inter[ia, ib]
        ul = self._ls[ia] + self._ls[ib] - inter
        self._inter_u = inter
        self._cnt_u = self._icount[ia, ib]
        self._ul = ul
        self.union_sorted = np.sort(ul)
        self.l_min = float(ul.min()) if ul.size else 0.0
        self.spread = (float(ul.max()) - self.l_min) if ul.size else 0.0
        self.mean_rank = (
            math.fsum(self._ranks) / len(self._ranks)
            if self._ranks
            else 0.0
        )
        self._feat: np.ndarray | None = None

    def state_features(self) -> list[float]:
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

    def _build_feat(self) -> np.ndarray:
        """Assemble the ``[n_pairs, _PAIR_DIM]`` feature matrix.

        Every column is the same IEEE arithmetic the scalar feature
        builder applied (subtract ``l_min``, divide by the spread
        denominator, divide the ranks by four), so the result is
        bit-identical; the normalisation is done in place only to avoid
        temporary arrays.
        """
        ul = self._ul
        inter = self._inter_u
        count = self._cnt_u
        ls = self._ls
        rk = self._rk
        ia, ib = self._ia, self._ib
        den = self.spread + 1.0
        n_pairs = max(len(self.pairs), 1)
        feat = np.empty((ul.shape[0], PAIR_DIM), dtype=np.float64)
        feat[:, 0] = ul
        feat[:, 1] = ls[ia]
        feat[:, 2] = ls[ib]
        feat[:, 3] = ul - inter
        feat[:, 4] = inter
        feat[:, :5] -= self.l_min
        feat[:, :5] /= den
        feat[:, 5] = rk[ia]
        feat[:, 6] = rk[ib]
        feat[:, 5:7] /= 4.0
        feat[:, 7] = count / np.maximum(rk[ia] + rk[ib] - count, 1.0)
        feat[:, 8] = np.searchsorted(self.union_sorted, ul) / n_pairs
        return feat

    def pair_feature_matrix(self) -> np.ndarray:
        """Return the ``[n_pairs, _PAIR_DIM]`` features, ``pairs`` order."""
        feat = self._feat
        if feat is None:
            feat = self._build_feat()
            self._feat = feat
        return feat

    def all_pair_features(self) -> list[list[float]]:
        """Return one feature vector per legal action, in ``pairs`` order."""
        return self.pair_feature_matrix().tolist()

    def _advance(self, a: int, b: int) -> None:
        """Contract ``(a, b)`` in the maintained caches, then refresh."""
        masks = self._masks
        m = len(masks)
        keep = [k for k in range(m) if k != a and k != b]
        new_ts = self.ts[a] ^ self.ts[b]
        new_mask = masks[a] ^ masks[b]
        new_lsize = self._lsize_mask(new_mask)
        k = len(keep)
        row_i = np.empty(k, dtype=np.float64)
        row_c = np.empty(k, dtype=np.int64)
        for pos, j in enumerate(keep):
            both = new_mask & masks[j]
            row_i[pos] = self._lsize_mask(both)
            row_c[pos] = both.bit_count()
        idx = np.asarray(keep, dtype=np.intp)
        inter = np.empty((k + 1, k + 1), dtype=np.float64)
        count = np.empty((k + 1, k + 1), dtype=np.int64)
        inter[:k, :k] = self._inter[np.ix_(idx, idx)]
        inter[k, k] = new_lsize
        inter[:k, k] = row_i
        inter[k, :k] = row_i
        count[:k, :k] = self._icount[np.ix_(idx, idx)]
        count[k, k] = len(new_ts)
        count[:k, k] = row_c
        count[k, :k] = row_c
        self._inter = inter
        self._icount = count
        self.ts = [self.ts[j] for j in keep] + [new_ts]
        self._masks = [masks[j] for j in keep] + [new_mask]
        self._lsize = [self._lsize[j] for j in keep] + [new_lsize]
        self._ranks = [self._ranks[j] for j in keep] + [len(new_ts)]
        self._refresh()

    def step(self, a: int, b: int) -> float:
        """Contract pair ``(a, b)``; charge and return its cost."""
        c = pair_cost(self.ts[a], self.ts[b], self.sizes)
        self.cost += c
        self._advance(a, b)
        return c


# ---------------------------------------------------------------------------
#  The policy net
# ---------------------------------------------------------------------------


class PairPolicyNet(nn.Module):
    """Score a candidate pair from the state: ``(state (+) pair) -> logit``.

    Scoring per action (rather than a fixed softmax head over a fixed
    vocabulary) keeps the action space open: the number of pairs changes
    every step, and a pair is a new point in the same feature space.
    Inputs are standardised by buffers fitted from real trajectories.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, hidden: int = 64) -> None:
        """Build the MLP over ``(state (+) pair)`` inputs."""
        super().__init__()
        width = STATE_DIM + PAIR_DIM
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


def pair_logits(
    model: nn.Module,
    state_feats: torch.Tensor,
    pair_feats: torch.Tensor,
) -> torch.Tensor:
    """Score every pair of every state; return shape ``[B, P]``."""
    b, p, k = pair_feats.shape
    s = state_feats.shape[1]
    x = torch.cat(
        [state_feats.unsqueeze(1).expand(b, p, s), pair_feats], dim=2
    )
    return model(x.reshape(b * p, s + k)).reshape(b, p)


def batch_inputs(
    games: list[ContractionGame], device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack every game's state and pair features into batched tensors."""
    sf = torch.as_tensor(
        np.stack([g.state_features() for g in games]),
        dtype=torch.float32,
        device=device,
    )
    pf = torch.as_tensor(
        np.stack([g.pair_feature_matrix() for g in games]),
        dtype=torch.float32,
        device=device,
    )
    return sf, pf


# ---------------------------------------------------------------------------
#  Rollouts — the vectorised lockstep driver
# ---------------------------------------------------------------------------


#: Per-(device, size) cached upper-triangle pair indices and arange.
_TRIU_T: dict[
    tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _triu_dev(
    m: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cached device ``(row, col, arange)`` index tensors."""
    key = (device, m)
    r = _TRIU_T.get(key)
    if r is None:
        ia, ib = _triu(m)
        r = (
            torch.as_tensor(ia, device=device),
            torch.as_tensor(ib, device=device),
            torch.arange(m, device=device),
        )
        _TRIU_T[key] = r
    return r


#: Below this batch size the scalar per-game driver is faster: the
#: vectorised step costs a fixed burst of kernel launches whatever the
#: batch, so at ``samples <= _SCALAR_MAX`` the old loop wins on CUDA.
_SCALAR_MAX = 2


def _scalar_rollouts(
    model: nn.Module,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    samples: int,
    device: str,
    temperature: float,
    greedy: bool,
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Lockstep rollouts over per-game :class:`ContractionGame` clones.

    The pre-vectorisation driver, kept for tiny batches (the single
    pass ``ContractionPolicy.order`` runs at batch 1): per-game
    feature assembly, one forward pass per step, the sampled indices
    back to the host every step.  Identical feature semantics to
    :func:`_lockstep_rollouts` — it *is* the scalar derivation.
    """
    games = [
        ContractionGame(tensors, sizes, greedy_ref)
        for _ in range(samples)
    ]
    orders: list[list[tuple[int, int]]] = [[] for _ in range(samples)]
    with torch.no_grad():
        while not games[0].done:
            sf, pf = batch_inputs(games, device)
            logits = pair_logits(model, sf, pf)
            if greedy:
                idx = torch.argmax(logits, dim=1)
            else:
                idx = Categorical(logits=logits / temperature).sample()
            for j, (g, i) in enumerate(
                zip(games, idx.tolist(), strict=True)
            ):
                a, b = g.pairs[i]
                orders[j].append((a, b))
                g.step(a, b)
    return orders, [g.cost for g in games]


def _board_tensors(
    ts: list[frozenset[int]], sizes: dict[int, int], device: str
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Pack a board into device tensors for the lockstep driver.

    ``M[t, p]`` is 1.0 when label ``p`` (sorted-label order — the same
    bit order :class:`ContractionGame` uses) belongs to tensor ``t``;
    ``logb[p] = log2 sizes[label_p]``; ``svec[p] = sizes[label_p]``.
    ``T = (M*logb) @ M^T`` is the pairwise intersection log-volume
    table and ``C = M @ M^T`` the intersection-cardinality table — the
    same contents as the scalar game's ``_inter`` / ``_icount``, so
    ``T[t, t]`` is tensor ``t``'s own log-volume and ``C[t, t]`` its
    rank.
    """
    labels = sorted({i for t in ts for i in t})
    bit = {lab: p for p, lab in enumerate(labels)}
    rows = np.zeros((len(ts), len(labels)), dtype=np.float64)
    for t, tensor in enumerate(ts):
        for lab in tensor:
            rows[t, bit[lab]] = 1.0
    logb = np.asarray([math.log2(sizes[lab]) for lab in labels])
    svec = np.asarray([sizes[lab] for lab in labels], dtype=np.float64)
    m0 = torch.as_tensor(rows, device=device)
    logb_t = torch.as_tensor(logb, device=device)
    svec_t = torch.as_tensor(svec, device=device)
    v0 = m0 * logb_t
    t0 = v0 @ m0.transpose(0, 1)
    c0 = m0 @ m0.transpose(0, 1)
    return m0, logb_t, svec_t, t0, c0


def _lockstep_rollouts(
    model: nn.Module,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    samples: int,
    device: str,
    temperature: float,
    greedy: bool,
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Lockstep rollouts over a vectorised batched game state.

    Same semantics as stepping ``samples`` :class:`ContractionGame`
    clones in lockstep — every episode chooses one pair per step from
    the policy's logits (``argmax`` under ``greedy``, else a
    temperature-scaled categorical sample) — but the per-game tensor
    sets live in one ``[B, m, L]`` indicator tensor and the pairwise
    tables in ``[B, m, m]`` tensors maintained **incrementally**: a
    step recomputes only the merged tensor's row/column (a skinny
    batched matmul), mirroring the scalar game's incremental
    ``_advance``.  The chosen-index history and the accumulated costs
    come back to the host in a single copy at the end — there is no
    per-step synchronisation, so a lockstep batch costs one short
    burst of kernel launches per step whatever ``samples`` is.

    Feature fidelity: every quantity is the same IEEE formula the
    scalar derivation applies, evaluated in float64 and cast to
    float32 for the net.  Pairwise sums come from batched GEMMs rather
    than ``math.fsum``, so feature columns 0-4 (and the accumulated
    cost once a product exceeds ``2**53``) can differ from the scalar
    game in the last ulp; columns 5-7 are exact integers; column 8's
    ranks can split *true* volume ties — mathematically equal union
    volumes that ``math.fsum`` evaluates bitwise-identically — by up
    to ``tie_size / n_pairs`` (measured, and empirically harmless, in
    ``project/retros/contraction-throughput-v2.md``).  The move cost
    is the exact union product while it stays below ``2**53``, the
    same regime in which scalar ``pair_cost`` is exact.  Sampling
    under ``greedy=False`` is the Gumbel-max trick — argmax of
    ``logits / T + g`` over iid Gumbel ``g`` — the same categorical
    distribution as ``Categorical(logits / T).sample()`` in fewer
    kernels.
    """
    ts = [frozenset(t) for t in tensors]
    n0 = len(ts)
    if n0 < 2:
        return [[] for _ in range(samples)], [0.0] * samples
    m0, logb, svec, t0, c0 = _board_tensors(ts, sizes, device)
    width = m0.shape[1]
    ind = m0.unsqueeze(0).expand(samples, n0, width)
    inter = t0.unsqueeze(0).expand(samples, n0, n0)
    icount = c0.unsqueeze(0).expand(samples, n0, n0)
    ref_inv = 1.0 / greedy_ref if greedy_ref > 0 else 0.0
    cost = m0.new_zeros(samples)
    hist: list[torch.Tensor] = []
    tiny = torch.finfo(torch.float32).tiny
    with torch.inference_mode():
        for m in range(n0, 1, -1):
            ia, ib, pos = _triu_dev(m, device)
            n_pairs = ia.numel()
            inter_u = inter[:, ia, ib]
            cnt_u = icount[:, ia, ib]
            lsize = inter.diagonal(0, 1, 2)
            ranks = icount.diagonal(0, 1, 2)
            ls_a = lsize[:, ia]
            ls_b = lsize[:, ib]
            ul = ls_a + ls_b - inter_u
            l_min, u_max = torch.aminmax(ul, dim=1)
            spread = u_max - l_min
            den = spread + 1.0
            rk_a = ranks[:, ia]
            rk_b = ranks[:, ib]
            f5 = torch.stack(
                (ul, ls_a, ls_b, ul - inter_u, inter_u), dim=-1
            )
            f5 = (f5 - l_min[:, None, None]) / den[:, None, None]
            c7 = cnt_u / (rk_a + rk_b - cnt_u).clamp_min(1.0)
            srt = ul.sort(dim=1).values
            pct = torch.searchsorted(srt, ul).to(m0.dtype) / n_pairs
            pf = torch.cat(
                (
                    f5,
                    torch.stack((rk_a, rk_b), dim=-1) / 4.0,
                    c7.unsqueeze(-1),
                    pct.unsqueeze(-1),
                ),
                dim=-1,
            ).to(torch.float32)
            mean_rank = ranks.sum(dim=1) / m
            den_s = 4.0 * mean_rank + 1.0
            r_min, r_max = torch.aminmax(ranks, dim=1)
            sf = torch.stack(
                (
                    cost.new_full((samples,), m / n0),
                    cost * ref_inv,
                    mean_rank / 4.0,
                    r_max / 4.0,
                    r_min / 4.0,
                    spread / den_s,
                    l_min / den_s,
                ),
                dim=1,
            ).to(torch.float32)
            logits = pair_logits(model, sf, pf)
            if greedy:
                idx = logits.argmax(dim=1)
            else:
                u = torch.rand_like(logits)
                gum = -torch.log(
                    (-torch.log(u.clamp_min(tiny))).clamp_min(tiny)
                )
                idx = (logits / temperature + gum).argmax(dim=1)
            a = ia[idx]
            b = ib[idx]
            hist.append(torch.stack((a, b), dim=1))
            sel_a = a[:, None, None].expand(samples, 1, width)
            sel_b = b[:, None, None].expand(samples, 1, width)
            m_a = ind.gather(1, sel_a)
            m_b = ind.gather(1, sel_b)
            step_c = (
                torch.where((m_a + m_b) > 0, svec, 1.0)
                .prod(dim=2)
                .squeeze(1)
            )
            cost = cost + step_c
            merged = (m_a - m_b).abs()
            keep = (pos[None, :] != a[:, None]) & (
                pos[None, :] != b[:, None]
            )
            keep_idx = keep.to(torch.int8).argsort(
                dim=1, descending=True, stable=True
            )[:, : m - 2]
            k = m - 2
            ind = torch.cat(
                (
                    ind.gather(
                        1,
                        keep_idx[:, :, None].expand(samples, k, width),
                    ),
                    merged,
                ),
                dim=1,
            )
            mlog = merged * logb
            kept_t = ind[:, :k].transpose(1, 2)
            inter_row = mlog.bmm(kept_t).squeeze(1)
            cnt_row = merged.bmm(kept_t).squeeze(1)
            lsize_new = mlog.sum(dim=-1)
            rank_new = merged.sum(dim=-1)
            ki = keep_idx[:, :, None].expand(samples, k, m)
            kj = keep_idx[:, None, :].expand(samples, k, k)
            t_keep = inter.gather(1, ki).gather(2, kj)
            c_keep = icount.gather(1, ki).gather(2, kj)
            inter = torch.cat(
                (
                    torch.cat((t_keep, inter_row[:, :, None]), dim=2),
                    torch.cat(
                        (inter_row[:, None, :], lsize_new[:, :, None]),
                        dim=2,
                    ),
                ),
                dim=1,
            )
            icount = torch.cat(
                (
                    torch.cat((c_keep, cnt_row[:, :, None]), dim=2),
                    torch.cat(
                        (cnt_row[:, None, :], rank_new[:, :, None]),
                        dim=2,
                    ),
                ),
                dim=1,
            )
    picked = torch.stack(hist, dim=1).tolist()
    orders = [[(a, b) for a, b in row] for row in picked]
    return orders, cost.tolist()


def _rollouts(
    model: nn.Module,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    samples: int,
    device: str,
    temperature: float,
    greedy: bool,
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Dispatch to the scalar or vectorised lockstep driver by batch.

    Tiny batches take :func:`_scalar_rollouts` (the vectorised step
    costs a fixed launch burst whatever ``samples`` is, so a batch of
    one or two is cheaper per game); everything else takes
    :func:`_lockstep_rollouts`.  The split is a measured crossover,
    not a semantic one — both drivers implement the same rollout.
    """
    if samples < 1:
        return [], []
    if samples <= _SCALAR_MAX:
        return _scalar_rollouts(
            model,
            tensors,
            sizes,
            greedy_ref,
            samples,
            device,
            temperature,
            greedy,
        )
    return _lockstep_rollouts(
        model,
        tensors,
        sizes,
        greedy_ref,
        samples,
        device,
        temperature,
        greedy,
    )


def run_policy_batch(
    model: nn.Module,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    *,
    samples: int,
    greedy: bool,
    temperature: float,
    device: str,
) -> list[float]:
    """Roll the policy out ``samples`` times; return the final costs.

    All episodes advance in lockstep, so one forward pass scores the
    whole batch each step — see :func:`_lockstep_rollouts` for the
    batched driver and its float64/last-ulp feature-fidelity notes.
    """
    _orders, costs = _rollouts(
        model,
        tensors,
        sizes,
        greedy_ref,
        samples,
        device,
        temperature,
        greedy,
    )
    return costs


def rollout_orders(
    model: nn.Module,
    tensors: Any,
    sizes: dict[int, int],
    greedy_ref: float,
    samples: int,
    device: str,
    temperature: float,
    *,
    greedy: bool = False,
) -> tuple[list[list[tuple[int, int]]], list[float]]:
    """Lockstep rollouts that also record each episode's chosen order.

    Same vectorised driver as :func:`run_policy_batch`, except it
    returns the ``(a, b)`` index pairs every rollout picked, so an
    order can be re-scored with an independent cost model or replayed.
    With ``greedy=True`` the argmax pair is taken (deterministic);
    otherwise pairs are sampled at ``temperature``.
    """
    return _rollouts(
        model,
        tensors,
        sizes,
        greedy_ref,
        samples,
        device,
        temperature,
        greedy,
    )


# ---------------------------------------------------------------------------
#  The shipped player
# ---------------------------------------------------------------------------


class ContractionPolicy:
    """A usable contraction player wrapping a trained ``PairPolicyNet``.

    Holds the eval-mode model, its device, and the artifact's training
    metadata (``meta``).  ``order`` is the deterministic single pass;
    ``sample_orders`` / ``best_order`` run temperature-sampled rollouts
    under a seed, the anytime player the experiments measured.  Every
    method takes a ``(tensors, sizes)`` board — see the module
    docstring's game contract.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str = "cpu",
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Wrap ``model`` on ``device``; ``meta`` is the artifact's meta."""
        self.model = model.to(device).eval()
        self.device = device
        self.meta = dict(meta or {})

    def order(
        self, tensors: Any, sizes: dict[int, int]
    ) -> list[tuple[int, int]]:
        """Return the deterministic (argmax) contraction order.

        One forward pass per step — the single-pass player.  Same
        board, same weights, same order.
        """
        orders, _costs = rollout_orders(
            self.model,
            tensors,
            sizes,
            greedy(tensors, sizes),
            1,
            self.device,
            1.0,
            greedy=True,
        )
        return orders[0]

    def cost(self, tensors: Any, sizes: dict[int, int]) -> float:
        """Return the replay cost of :meth:`order` on the board."""
        return cost_of_order(tensors, sizes, self.order(tensors, sizes))

    def sample_orders(
        self,
        tensors: Any,
        sizes: dict[int, int],
        *,
        samples: int = 64,
        seed: int = 0,
        temperature: float = SAMPLE_TEMP,
    ) -> list[tuple[list[tuple[int, int]], float]]:
        """Return ``samples`` seeded rollouts as ``(order, cost)`` pairs.

        Deterministic under ``seed`` on a given device — the sampled
        (restart) player the experiments compared at matched rollout
        counts.
        """
        torch.manual_seed(seed)
        orders, costs = rollout_orders(
            self.model,
            tensors,
            sizes,
            greedy(tensors, sizes),
            samples,
            self.device,
            temperature,
        )
        return list(zip(orders, costs, strict=True))

    def best_order(
        self,
        tensors: Any,
        sizes: dict[int, int],
        *,
        samples: int = 64,
        seed: int = 0,
        temperature: float = SAMPLE_TEMP,
    ) -> tuple[list[tuple[int, int]], float]:
        """Return the cheapest of ``samples`` seeded rollouts.

        Best-of-N sampled rollouts — the strongest form of the learned
        player at matched rollout counts.  For a wall-clock-budgeted
        anytime player see ``tools/contraction_einsum.policy_best_order``
        (equal-budget protocol; kept in tools because wall-clock
        measurements are an experiment concern, not an API).
        """
        runs = self.sample_orders(
            tensors,
            sizes,
            samples=samples,
            seed=seed,
            temperature=temperature,
        )
        return min(runs, key=lambda oc: oc[1])


# ---------------------------------------------------------------------------
#  Artifact: save + lazy load
# ---------------------------------------------------------------------------


@dataclass
class _Artifact:
    """Validated artifact payload: state dict + metadata."""

    state_dict: dict[str, torch.Tensor]
    meta: dict[str, Any]


def save_contraction_policy(
    model: nn.Module,
    path: str | Path,
    *,
    meta: dict[str, Any],
) -> None:
    """Serialise a trained net + provenance to ``path`` (``torch.save``).

    The payload carries a format tag, the feature-contract version and
    the training provenance (scales, iterations, seed, git sha, …) so a
    loader can refuse a mismatched artifact instead of silently
    producing a wrong player.  Tensors are stored on CPU — the artifact
    is device-agnostic.  Called by ``tools/train_contraction_artifact``.
    """
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    hidden = int(state["net.0.weight"].shape[0])
    payload = {
        "format": _FORMAT,
        "state_dict": state,
        "meta": {
            "hidden": hidden,
            "state_dim": STATE_DIM,
            "pair_dim": PAIR_DIM,
            "feature_contract": _FEATURE_CONTRACT,
            **meta,
        },
    }
    torch.save(payload, Path(path))


def _read(path: Path) -> _Artifact:
    """Load and validate an artifact payload; raise ``ValueError``."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != _FORMAT
    ):
        raise ValueError(
            f"{path}: not a catopt contraction-policy artifact "
            f"(format tag missing or != {_FORMAT!r})"
        )
    meta = payload.get("meta") or {}
    for key, want in (
        ("state_dim", STATE_DIM),
        ("pair_dim", PAIR_DIM),
        ("feature_contract", _FEATURE_CONTRACT),
    ):
        if meta.get(key) != want:
            raise ValueError(
                f"{path}: feature-spec mismatch — "
                f"meta[{key!r}] is {meta.get(key)!r}, need {want!r}; "
                "the bundled weights and this feature derivation have "
                "drifted"
            )
    if "hidden" not in meta:
        raise ValueError(f"{path}: artifact carries no 'hidden' meta")
    return _Artifact(payload["state_dict"], dict(meta))


def _default_resource() -> Any:
    """Return the bundled default-weights resource."""
    return resources.files("catopt_torch").joinpath(
        "artifacts", _DEFAULT_NAME
    )


def load_contraction_policy(
    path: str | Path | None = None,
    *,
    device: str = "cpu",
) -> ContractionPolicy:
    """Load a contraction player from weights; return it ready to use.

    ``path=None`` loads the bundled **distilled** weights
    (``catopt_torch/artifacts/contraction_policy_distilled.pt`` — the
    oe-all player that beats ``opt_einsum``'s randomised greedy at
    equal wall-clock at n = 40); a user path loads a checkpoint
    written by :func:`save_contraction_policy`.  The earlier
    curriculum-RL weights stay bundled at
    ``artifacts/contraction_policy_curriculum.pt`` — load them with
    ``load_contraction_policy(resources.files("catopt_torch") /
    "artifacts" / "contraction_policy_curriculum.pt")``.  Nothing is
    loaded at import — the ``torch.load`` happens here, on call.
    Raises ``ValueError`` when the payload is not a catopt
    contraction-policy artifact or its feature spec does not match
    this module's derivation.

    The default ``device`` is CPU deliberately: a single forward pass
    per decision is small, and the artifact must work on a CUDA-free
    host.  Pass ``device="cuda"`` for the throughput the experiments
    measured (large sampled batches amortise better on GPU).
    """
    if path is None:
        with resources.as_file(_default_resource()) as p:
            return load_contraction_policy(p, device=device)
    artifact = _read(Path(path))
    model = PairPolicyNet(int(artifact.meta["hidden"]))
    model.load_state_dict(artifact.state_dict)
    return ContractionPolicy(model, device=device, meta=artifact.meta)
