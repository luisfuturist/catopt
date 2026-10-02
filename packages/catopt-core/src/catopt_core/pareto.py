"""Pareto-frontier evaluation landscapes (torch-free).

ADR 0003 keeps *evaluation* a dimension of its own: a candidate's
quality is a point in a space of named cost dimensions (FLOPs,
latency, memory, …), and scalarizing those axes too early bakes a
weighting into the search before the caller has chosen one.  This
module keeps the landscape intact — :class:`CostVector` names the
axes, :func:`dominates` orders two points, and
:func:`pareto_frontier` returns the non-dominated set — while
:func:`best` offers the one scalarized view a caller opts into.

Every dimension is minimized: lower is better.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

__all__ = [
    "CostVector",
    "best",
    "dominates",
    "pareto_frontier",
]

T = TypeVar("T")


@dataclass(frozen=True)
class CostVector:
    """A point in a named cost space, one value per dimension.

    ``dims`` and ``values`` are parallel tuples, so
    ``values[i]`` prices ``dims[i]``.  The type is frozen (hashable
    and safe to share) and validates its own shape on construction.
    All dimensions are minimized.
    """

    dims: tuple[str, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject vectors whose ``dims`` and ``values`` disagree."""
        if len(self.dims) != len(self.values):
            raise ValueError(
                "dims and values must be the same length: "
                f"{len(self.dims)} != {len(self.values)}"
            )

    @classmethod
    def from_mapping(
        cls,
        m: Mapping[str, float],
        dims: tuple[str, ...] | None = None,
    ) -> CostVector:
        """Build a vector from ``m`` in ``dims`` order.

        ``dims`` defaults to the mapping's own key order; a supplied
        order naming a missing key raises ``KeyError``.
        """
        order = tuple(m) if dims is None else dims
        return cls(order, tuple(m[d] for d in order))

    def __getitem__(self, dim: str) -> float:
        """Return the cost on ``dim`` (``KeyError`` if absent)."""
        return self.as_dict()[dim]

    def as_dict(self) -> dict[str, float]:
        """Return the vector as a ``{dim: value}`` mapping."""
        return dict(zip(self.dims, self.values, strict=True))


def dominates(a: CostVector, b: CostVector) -> bool:
    """Return whether ``a`` Pareto-dominates ``b``.

    True iff ``a`` is no worse than ``b`` on every shared dimension
    and strictly better on at least one.  The two vectors must name
    the same dimensions in the same order, else ``ValueError``.
    """
    if a.dims != b.dims:
        raise ValueError(
            "cannot compare vectors with different dims: "
            f"{a.dims} != {b.dims}"
        )
    no_worse = all(
        x <= y for x, y in zip(a.values, b.values, strict=True)
    )
    strictly_better = any(
        x < y for x, y in zip(a.values, b.values, strict=True)
    )
    return no_worse and strictly_better


def pareto_frontier(
    items: Sequence[T],
    key: Callable[[T], CostVector],
) -> list[T]:
    """Return the non-dominated ``items``, in input order.

    An item survives when no other item dominates it; equal vectors
    therefore all survive, since neither strictly beats the other.
    The result is deterministic and order-preserving.
    """
    vectors = [key(item) for item in items]
    return [
        item
        for i, item in enumerate(items)
        if not any(
            dominates(vectors[j], vectors[i])
            for j in range(len(items))
            if j != i
        )
    ]


def best(
    items: Sequence[T],
    key: Callable[[T], CostVector],
    weights: Mapping[str, float] | None = None,
) -> T:
    """Return the scalarized argmin of ``items``.

    Scores each item as ``sum(weights[dim] * value)``; when
    ``weights`` is ``None`` every dimension weighs ``1.0``, and a
    dimension absent from a supplied mapping weighs ``0.0``.  Ties
    break toward the earlier item.  Empty ``items`` raises
    ``ValueError``.
    """
    if not items:
        raise ValueError("best() requires at least one item")

    def score(item: T) -> float:
        vector = key(item)
        if weights is None:
            return sum(vector.values)
        return sum(
            weights.get(dim, 0.0) * value
            for dim, value in zip(
                vector.dims, vector.values, strict=True
            )
        )

    return min(items, key=score)
