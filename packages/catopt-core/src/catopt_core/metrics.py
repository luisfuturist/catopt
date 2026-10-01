"""Task-level equivalence metrics — the certificate's task contract.

Plan 0015.  A pointwise bound (``max_rel ≤ rtol``) is one statement
of equivalence, but the user's real contract is often *behavioural*:
an LLM cares about the logits' ranking, a classifier about the
argmax, a retrieval model about the cosine ordering of its
embeddings.  A :class:`~catopt_core.ports.TaskMetric` reduces a pair
of module outputs to a scalar ``distance`` gated by a ``tolerance``;
the shipped metrics below are the vocabularies the bound-
amplification study showed matter — ``top-1`` stayed perfect on
drifts ~100x the weight-space bound while every pointwise gate
declined.

Backend-neutral by construction: metrics consume whatever the
lowered modules returned — a tensor-like or a pytree (``tuple`` /
``dict``) of tensor-likes — and evaluate through ``tolist()`` into
plain Python floats.  Core names no tensor library; a torch
``Tensor``, a numpy ``ndarray`` and a nested list all evaluate the
same.  The evaluation is pure Python — intended for the calibration
input a verify already runs, not production-scale sweeps.

Every metric is a frozen value: ``name`` and ``tolerance`` ride the
object so a manifest can record *which* contract gated acceptance
(``stats["task"]`` / the manifest's ``task`` key carry name,
tolerance and the measured distance).  ``tolerance`` defaults are
conservative — agreement/ranking metrics default to ``0.0`` (any
detected drift declines); ``task_tol=`` overrides at the pipeline.

Row convention: each output leaf is flattened to ``(n, d)`` rows —
the last axis is the feature axis, everything upstream folds into
rows.  A leaf distance is computed per row set and the metric's
``distance`` is the *worst* leaf — a multi-output model is as close
as its most-diverged output.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ArgmaxStability",
    "CosineRanking",
    "LogitKL",
    "MaxRel",
    "TopKAgreement",
]

#: The rel-diff denominator floor — the same constant
#: ``verify_equiv`` uses, so :class:`MaxRel` agrees with the
#: historical gate bit-for-bit.
_REL_FLOOR = 1e-8


# ---------------------------------------------------------------------------
#  Output plumbing — pytree pairing and the (n, d) row view
# ---------------------------------------------------------------------------


def _tolist(v: Any) -> Any:
    """Nested Python lists from a tensor-like; pass anything else on."""
    fn = getattr(v, "tolist", None)
    return fn() if callable(fn) else v


def _leaves(ref: Any, opt: Any) -> Iterator[tuple[Any, Any]]:
    """Pair corresponding output leaves of two matching pytrees.

    ``dict`` outputs pair by key, ``tuple`` outputs positionally —
    a mismatched structure is a loud :class:`TypeError`, never a
    silent skip.  Everything else (tensor-likes, lists — a module
    returning a *list* of tensor-likes pools them as one row set) is
    a leaf pair.
    """
    if isinstance(ref, dict) and isinstance(opt, dict):
        if ref.keys() != opt.keys():
            raise TypeError(
                "task metric: output dict keys differ "
                f"({sorted(ref)} vs {sorted(opt)})"
            )
        for k in ref:
            yield from _leaves(ref[k], opt[k])
        return
    if isinstance(ref, tuple) and isinstance(opt, tuple):
        if len(ref) != len(opt):
            raise TypeError(
                "task metric: output tuple arities differ "
                f"({len(ref)} vs {len(opt)})"
            )
        for a, b in zip(ref, opt, strict=True):
            yield from _leaves(a, b)
        return
    yield ref, opt


def _rows(v: Any) -> list[list[float]]:
    """Flatten one output leaf to ``(n, d)`` float rows.

    A scalar or non-sequence is a :class:`TypeError` — a metric can
    only gate on tensor-shaped outputs; a non-tensor output means the
    metric cannot evaluate the delivery, which must be loud.
    """
    v = _tolist(v)
    if not isinstance(v, (list, tuple)):
        raise TypeError(
            "task metrics need numeric tensor-like outputs, got "
            f"{type(v).__name__}"
        )
    if not v:
        return []
    if all(isinstance(e, (int, float)) for e in v):
        return [[float(e) for e in v]]
    rows: list[list[float]] = []
    for e in v:
        rows.extend(_rows(e))
    return rows


def _match(a: list[list[float]], b: list[list[float]]) -> None:
    """Loudly require equal row sets — a shape gap means no metric."""
    if len(a) != len(b):
        raise ValueError(
            f"task metric: row counts differ ({len(a)} vs {len(b)})"
        )
    for ra, rb in zip(a, b, strict=True):
        if len(ra) != len(rb):
            raise ValueError(
                f"task metric: row widths differ "
                f"({len(ra)} vs {len(rb)})"
            )


class _RowMetric:
    """Shared plumbing: pair output pytrees, pool the worst leaf.

    Subclasses implement ``_dist`` over one leaf pair's ``(n, d)``
    row lists; :meth:`distance` runs it on every :func:`_leaves`
    pair and returns the maximum — a multi-output model is as close
    as its most-diverged output.
    """

    def distance(self, ref: Any, opt: Any) -> float:
        """Return the worst leaf distance between the two outputs."""
        d = 0.0
        for a, b in _leaves(ref, opt):
            d = max(d, self._dist(a, b))
        return d

    def _dist(self, ref: Any, opt: Any) -> float:
        """Distance of one leaf pair — subclass-owned."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
#  Small numeric kernels (pure Python — core names no tensor library)
# ---------------------------------------------------------------------------


def _argmax(row: list[float]) -> int:
    """First index of the row maximum (``torch.argmax`` semantics).

    Rows are never empty — :func:`_rows` produces only non-empty
    rows.
    """
    return max(range(len(row)), key=lambda i: row[i])


def _topk_idx(row: list[float], k: int) -> set[int]:
    """Index set of the ``k`` largest values; ties break on index."""
    kk = min(k, len(row))
    return set(sorted(range(len(row)), key=lambda i: (-row[i], i))[:kk])


def _softmax(row: list[float]) -> list[float]:
    """Max-shifted softmax of one row."""
    m = max(row)
    e = [math.exp(v - m) for v in row]
    s = sum(e)
    return [v / s for v in e]


def _kl(p: list[float], q: list[float]) -> float:
    """``KL(p ‖ q)`` in nats; ``inf`` when ``q`` masses where ``p``."""
    total = 0.0
    for pi, qi in zip(p, q, strict=True):
        if pi <= 0.0:
            continue
        if qi <= 0.0:
            return float("inf")
        total += pi * math.log(pi / qi)
    return max(0.0, total)


def _cos_mat(rows: list[list[float]]) -> list[list[float]]:
    """``n * n`` cosine-similarity matrix of the row set."""
    norms = [math.sqrt(sum(v * v for v in r)) for r in rows]

    def cos(i: int, j: int) -> float:
        n = norms[i] * norms[j]
        if n <= 0.0:
            return 0.0
        return (
            sum(x * y for x, y in zip(rows[i], rows[j], strict=True))
            / n
        )

    n = len(rows)
    return [[cos(i, j) for j in range(n)] for i in range(n)]


def _sign(v: float) -> int:
    """-1 / 0 / +1 — the ordering verdict of one cosine pair."""
    if v > 0.0:
        return 1
    if v < 0.0:
        return -1
    return 0


def _kendall(
    ra: list[float], rb: list[float], skip: int
) -> float | None:
    """Discordant-pair fraction of two rankings, skipping index *skip*.

    ``ra``/``rb`` are the ref/opt cosine-similarity rows for one
    query.  Pairs tied in the reference are uninformative and skipped;
    ``None`` when every pair tied — the caller pools non-``None``
    values.
    """
    idxs = [j for j in range(len(ra)) if j != skip]
    discordant = compared = 0
    for x in range(len(idxs)):
        for y in range(x + 1, len(idxs)):
            j, k = idxs[x], idxs[y]
            s_ref = _sign(ra[j] - ra[k])
            if s_ref == 0:
                continue
            compared += 1
            if s_ref != _sign(rb[j] - rb[k]):
                discordant += 1
    return discordant / compared if compared else None


# ---------------------------------------------------------------------------
#  The shipped metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaxRel(_RowMetric):
    """The historical pointwise gate, as a task metric.

    ``distance = max|ref - opt| / (max|ref| + 1e-8)`` — the same
    formula and floor as ``verify_equiv``, so ``MaxRel()`` under a
    default ``task_tol`` is exactly the pointwise certificate.  It is
    the baseline a task metric relaxes: pick it explicitly to restate
    the pointwise contract through the task channel, or pass it with
    a looser ``tolerance``/``task_tol`` to widen the bound by fiat.
    """

    tolerance: float = 1e-4

    @property
    def name(self) -> str:
        """Return ``"max_rel"`` — the metric the default gate uses."""
        return "max_rel"

    def _dist(self, ref: Any, opt: Any) -> float:
        a, b = _rows(ref), _rows(opt)
        _match(a, b)
        num = denom = 0.0
        for ra, rb in zip(a, b, strict=True):
            for x, y in zip(ra, rb, strict=True):
                num = max(num, abs(x - y))
                denom = max(denom, abs(x))
        return num / (denom + _REL_FLOOR)


@dataclass(frozen=True)
class ArgmaxStability(_RowMetric):
    """Fraction of rows whose argmax moved — the classifier contract.

    ``distance`` is the disagreement rate over the row set (``0.0``
    when every argmax survives); the default tolerance ``0.0`` gates
    on *perfect* top-1 retention — relax with ``tolerance`` /
    ``task_tol`` for an explicit slack.  An empty output set has no
    argmaxes to move — distance ``0.0``.
    """

    tolerance: float = 0.0

    @property
    def name(self) -> str:
        """Return ``"argmax_stability"``."""
        return "argmax_stability"

    def _dist(self, ref: Any, opt: Any) -> float:
        a, b = _rows(ref), _rows(opt)
        _match(a, b)
        if not a:
            return 0.0
        bad = sum(
            1
            for ra, rb in zip(a, b, strict=True)
            if _argmax(ra) != _argmax(rb)
        )
        return bad / len(a)


@dataclass(frozen=True)
class TopKAgreement(_RowMetric):
    """``1 - mean`` top-k set overlap per row — the ranking contract.

    For each row the ``k`` largest indices (ties break on index
    order) form a set on each side; agreement scores
    ``|ref_topk ∩ opt_topk| / k``.  ``distance`` is one minus the
    mean score — ``0.0`` when every row's top-k set survives
    verbatim.  ``k=1`` degenerates to argmax agreement; ``k`` larger
    than a row's width clamps to the width.  The recorded ``name``
    carries ``k`` (``"top5_agreement"``), so the manifest names the
    contract precisely.
    """

    k: int = 1
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        """Reject a degenerate ``k`` — a top-0 set is meaningless."""
        if self.k < 1:
            raise ValueError(
                f"TopKAgreement needs k >= 1, got {self.k}"
            )

    @property
    def name(self) -> str:
        """Return ``"top{k}_agreement"``."""
        return f"top{self.k}_agreement"

    def _dist(self, ref: Any, opt: Any) -> float:
        a, b = _rows(ref), _rows(opt)
        _match(a, b)
        if not a:
            return 0.0
        scores = [
            len(_topk_idx(ra, self.k) & _topk_idx(rb, self.k))
            / min(self.k, len(ra))
            for ra, rb in zip(a, b, strict=True)
        ]
        return 1.0 - sum(scores) / len(scores)


@dataclass(frozen=True)
class LogitKL(_RowMetric):
    """Worst-row ``KL(softmax ref ‖ softmax opt)`` in nats.

    Per row the reference output softmaxes to ``p`` and the
    delivered one to ``q``; ``distance`` is the *max* per-row
    KL — the worst softened distribution shift anywhere in the
    output, the conservative reading for a gate.  A row that
    zeroes a probability the reference masses on reports ``inf``
    (any finite ``task_tol`` declines).  Default tolerance is
    ``1e-3`` nats — the order the amplification study measured for
    logit drift that leaves the ranking untouched.
    """

    tolerance: float = 1e-3

    @property
    def name(self) -> str:
        """Return ``"logit_kl"``."""
        return "logit_kl"

    def _dist(self, ref: Any, opt: Any) -> float:
        a, b = _rows(ref), _rows(opt)
        _match(a, b)
        return max(
            (
                _kl(_softmax(ra), _softmax(rb))
                for ra, rb in zip(a, b, strict=True)
            ),
            default=0.0,
        )


@dataclass(frozen=True)
class CosineRanking(_RowMetric):
    """Cosine-similarity ranking preservation — the retrieval contract.

    Treats the output's row set as an embedding collection: for each
    row ``i`` the cosine similarities ``cos(row_i, row_j)`` rank the
    rest of the set, and the metric compares that ranking between
    reference and delivered outputs.  ``distance`` is the mean
    Kendall-tau discordant-pair fraction over all query rows —
    ``0.0`` when every row's cosine ordering survives.  Pairs tied
    in the reference are uninformative and skipped.  Fewer than
    three rows admit no pair ordering — distance ``0.0``.

    The computation is O(n^3) in rows after the O(n^2·d) cosine
    matrix — a calibration-input metric, not a batching one.
    """

    tolerance: float = 0.0

    @property
    def name(self) -> str:
        """Return ``"cosine_ranking"``."""
        return "cosine_ranking"

    def _dist(self, ref: Any, opt: Any) -> float:
        a, b = _rows(ref), _rows(opt)
        _match(a, b)
        n = len(a)
        if n < 3:
            return 0.0
        ca, cb = _cos_mat(a), _cos_mat(b)
        taus = [
            t
            for i in range(n)
            if (t := _kendall(ca[i], cb[i], i)) is not None
        ]
        return sum(taus) / len(taus) if taus else 0.0
