"""Search trajectories — the data a learned policy trains on.

Pretraining data for a search policy (ADR 0003, plan 0016 stage 7):
for a program, the cost delta each rule produces when applied alone.
A :class:`RuleSample` pairs the program's static features and a
**structural** encoding of the rule — so a rule the model has never
seen is still scored by its shape, not a fixed vocabulary index — with
the observed improvement.

Torch-free: this is data generation over the engine, not learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from catopt_core.egraph import EGraph
from catopt_core.features import ProgramFeatures, compute_features
from catopt_core.game import Evaluator, GameState
from catopt_core.ir import op_repr

__all__ = ["RuleSample", "rule_samples", "rule_vector"]

#: Buckets each op name hashes into for the structural rule encoding.
_OP_BUCKETS = 8

#: Buckets each rule side's full pattern tree hashes into.
_TREE_BUCKETS = 16

#: Length of a :func:`rule_vector`.
RULE_VECTOR_LEN = 2 * _OP_BUCKETS + 2 * _TREE_BUCKETS + 4


def _stable_hash(s: str, buckets: int) -> int:
    """Stable hash of a string into ``[0, buckets)``."""
    h = 0
    for i, ch in enumerate(s):
        h = (h * 131 + ord(ch) * (i + 1)) % 1_000_003
    return h % buckets


def _op_bucket(name: str) -> int:
    """Stable hash of an op name into ``[0, _OP_BUCKETS)``."""
    return _stable_hash(name, _OP_BUCKETS)


def _tree(node: Any) -> str:
    """Render one rule side to a stable string (empty for ``None``)."""
    return "" if node is None else op_repr(node)


def rule_vector(rule: Any) -> tuple[float, ...]:
    """Structural encoding of a rule — its shape, not its identity.

    One-hot of each side's root-op bucket, one-hot of each side's
    *full pattern-tree* hash bucket, then LHS arity, RHS arity, and
    two flags (has a ``check``, has a ``derive``).  The tree hash
    separates rules whose root ops coincide (e.g. several ``matmul``
    laws); it is computed from the rule's own pattern, so a new rule
    is still a new point in this space — scorable without retraining.
    """
    lhs = getattr(rule, "lhs", None)
    rhs = getattr(rule, "rhs", None)
    vec = [0.0] * RULE_VECTOR_LEN
    vec[_op_bucket(getattr(lhs, "op", "") or "")] = 1.0
    vec[_OP_BUCKETS + _op_bucket(getattr(rhs, "op", "") or "")] = 1.0
    base = 2 * _OP_BUCKETS
    vec[base + _stable_hash(_tree(lhs), _TREE_BUCKETS)] = 1.0
    vec[
        base + _TREE_BUCKETS + _stable_hash(_tree(rhs), _TREE_BUCKETS)
    ] = 1.0
    tail = base + 2 * _TREE_BUCKETS
    vec[tail] = float(len(getattr(lhs, "args", ()) or ()))
    vec[tail + 1] = float(len(getattr(rhs, "args", ()) or ()))
    vec[tail + 2] = 1.0 if getattr(rule, "check", None) else 0.0
    vec[tail + 3] = 1.0 if getattr(rule, "derive", None) else 0.0
    return tuple(vec)


@dataclass(frozen=True)
class RuleSample:
    """One ``(program, rule, improvement)`` observation."""

    features: ProgramFeatures
    rule_vector: tuple[float, ...]
    rule: str
    delta_cost: float
    program: str


def rule_samples(
    term: Any, rules: Any, cost_fn: Any = None
) -> list[RuleSample]:
    """Cost delta each rule produces when applied alone to ``term``.

    ``delta_cost`` is ``cost_before - cost_after``; positive means the
    rule improved the extracted program.  Each rule gets a fresh
    e-graph, so the deltas are independent of one another.
    """
    feats = compute_features(term)
    program = op_repr(term)
    eg0 = EGraph()
    before = Evaluator(cost_fn).evaluate(
        GameState(eg0, eg0.add_term(term))
    )
    out = []
    for rule in rules:
        eg = EGraph()
        root = eg.add_term(term)
        eg.apply_rule(rule, root)
        after = Evaluator(cost_fn).evaluate(GameState(eg, root))
        out.append(
            RuleSample(
                feats,
                rule_vector(rule),
                rule.name,
                before - after,
                program,
            )
        )
    return out
