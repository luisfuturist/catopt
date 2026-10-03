"""Program features — static, hardware-independent characterization.

The EVALUATION dimension's static half (ADR 0003, plan 0016 stage 2):
a :class:`ProgramFeatures` describes what a program *is* — flops,
bytes, depth, reuse — **without running it**.  Pure Python, torch-free.

Not to be confused with a *profile*: a profile
(:class:`catopt_core.profile.TargetProfile`) is a measured target;
features describe a program (ADR 0003's naming rule).

fp32 (4 bytes per element) is assumed throughout — the shipped dtype.
A program whose shapes cannot be inferred contributes only the
dimensions it can (unknown shapes count zero elements).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catopt_core.cost import _VIEW_OPS, _flops_of
from catopt_core.ir import Op, Param, _dag_postorder
from catopt_core.typing import _numel, _shape_of, has_var_leaf

__all__ = [
    "DIMENSIONS",
    "ProgramFeatures",
    "StaticProfiler",
    "compute_features",
]

#: Bytes per element assumed throughout (fp32).
_ITEMSIZE = 4

#: The ordered feature dimensions a cost vector keys on.
DIMENSIONS: tuple[str, ...] = (
    "flops",
    "bytes_read",
    "bytes_written",
    "temporary_bytes",
    "depth",
    "operations",
    "param_leaves",
    "reuse",
    "parallelism",
)


@dataclass(frozen=True)
class ProgramFeatures:
    """Static, hardware-independent characteristics of a program.

    * ``flops`` — arithmetic operations (the cost model's per-op
      weights).
    * ``bytes_read`` / ``bytes_written`` — **runtime** element traffic,
      fp32.  View ops (`_VIEW_OPS`) and subtrees with no data input
      (folded at compile time) are excluded, so the numbers agree with
      the built-in cost models rather than billing phantom traffic.
    * ``temporary_bytes`` — bytes materialised by non-root nodes (the
      intermediates a fused backend could elide).
    * ``depth`` — longest op chain (the critical path, in ops).
    * ``operations`` — number of distinct op nodes (DAG-deduped).
    * ``param_leaves`` — distinct :class:`~catopt_core.ir.Param`
      operands.
    * ``reuse`` — arithmetic intensity, ``flops / (bytes_read +
      bytes_written)``; ``0.0`` when the program moves no bytes.
    * ``parallelism`` — average width, ``operations / depth`` (a
      proxy: how much of the DAG is independent of the critical
      path); equals ``operations`` for a depth-0 program.
    """

    flops: float
    bytes_read: float
    bytes_written: float
    temporary_bytes: float
    depth: int
    operations: int
    param_leaves: int
    reuse: float
    parallelism: float

    def as_dict(self) -> dict[str, float]:
        """Return the features keyed by dimension name."""
        return {d: float(getattr(self, d)) for d in DIMENSIONS}

    def to_vector(
        self, dims: tuple[str, ...] | None = None
    ) -> tuple[float, ...]:
        """Return the features as an ordered vector.

        ``dims`` selects and orders the dimensions (default
        :data:`DIMENSIONS`), so a learned model can pin its input
        layout.
        """
        return tuple(
            float(getattr(self, d)) for d in (dims or DIMENSIONS)
        )


#: The all-zero record for a leaf (no compute).
_EMPTY = ProgramFeatures(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0.0, 0.0)


def _elems(shape: Any) -> int:
    """Return the element count of a concrete shape, else ``0``."""
    if not isinstance(shape, tuple) or not all(
        isinstance(d, int) for d in shape
    ):
        return 0
    return _numel(shape)


def _read_bytes(node: Op, memo: dict, itemsize: int) -> float:
    """Return the bytes read by ``node``'s operands."""
    return sum(_elems(_shape_of(a, memo)) for a in node.args) * itemsize


def _param_leaves(node: Op) -> int:
    """Count the :class:`~catopt_core.ir.Param` operands of ``node``."""
    return sum(1 for a in node.args if isinstance(a, Param))


def _depth(node: Op, depth_memo: dict[Op, int]) -> int:
    """Return the longest op chain ending at ``node`` (0 for a leaf)."""
    return max(
        (depth_memo[a] + 1 for a in node.args if isinstance(a, Op)),
        default=0,
    )


def _assemble(
    flops: float,
    read: float,
    written: float,
    temp: float,
    nodes: list[Op],
    depth_memo: dict[Op, int],
    n_params: int,
) -> ProgramFeatures:
    """Build the feature record from the walk's accumulators."""
    depth = max((depth_memo[n] for n in nodes), default=0)
    operations = len(nodes)
    total = read + written
    reuse = flops / total if total > 0 else 0.0
    parallelism = operations / depth if depth > 0 else float(operations)
    return ProgramFeatures(
        flops,
        read,
        written,
        temp,
        depth,
        operations,
        n_params,
        reuse,
        parallelism,
    )


def compute_features(
    term: Any, *, itemsize: int = _ITEMSIZE
) -> ProgramFeatures:
    """Compute the static features of ``term``.

    ``term`` is an :class:`Op` DAG or a leaf.  A bare leaf (``Var`` /
    ``Param`` / ``Const``) has no compute: every dimension is zero.
    An op DAG is walked once, children before parents, each node
    charged exactly once.
    """
    if not isinstance(term, Op):
        return _EMPTY
    memo: dict = {}
    flops = 0.0
    read = 0.0
    written = 0.0
    temp = 0.0
    n_params = 0
    depth_memo: dict[Op, int] = {}
    nodes = _dag_postorder(term)
    for node in nodes:
        out_elems = _elems(_shape_of(node, memo))
        # Runtime memory traffic only.  A view op owns no storage (its
        # output aliases an operand's), and a subtree with no data input
        # is folded at compile time — neither writes nor reads anything
        # when the program runs.  The built-in cost models are
        # view/fold-aware (`_VIEW_OPS`, `has_var_leaf`); billing these
        # as traffic makes the profiler disagree with them.
        runtime = node.op not in _VIEW_OPS and has_var_leaf(node, memo)
        if runtime:
            written += out_elems * itemsize
            if node is not term:
                temp += out_elems * itemsize
            read += _read_bytes(node, memo, itemsize)
        flops += _flops_of(node, memo)
        n_params += _param_leaves(node)
        depth_memo[node] = _depth(node, depth_memo)
    return _assemble(
        flops, read, written, temp, nodes, depth_memo, n_params
    )


class StaticProfiler:
    """A :class:`~catopt_core.ports.Profiler` over an IR term.

    Pure Python and torch-free: it observes the program's structure,
    never runs it (ADR 0003 — profiling is not execution).
    """

    name = "static"

    def __init__(self, itemsize: int = _ITEMSIZE) -> None:
        """Fix the assumed element size (bytes)."""
        self.itemsize = itemsize

    def profile(self, program: Any) -> ProgramFeatures:
        """Return the static features of ``program``."""
        return compute_features(program, itemsize=self.itemsize)
