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

#: Length of a :func:`rule_vector`.
RULE_VECTOR_LEN = 2 * _OP_BUCKETS + 4


def _op_bucket(name: str) -> int:
    """Stable hash of an op name into ``[0, _OP_BUCKETS)``."""
    return (
        sum(ord(c) * (i + 1) for i, c in enumerate(name)) % _OP_BUCKETS
    )


def rule_vector(rule: Any) -> tuple[float, ...]:
    """Structural encoding of a rule — its shape, not its identity.

    One-hot of the LHS op's bucket, one-hot of the RHS op's bucket,
    then LHS arity, RHS arity, and two flags (has a ``check``, has a
    ``derive``).  A new rule is a new point in this space, so it can
    be scored without retraining.
    """
    lhs = getattr(rule, "lhs", None)
    rhs = getattr(rule, "rhs", None)
    vec = [0.0] * (2 * _OP_BUCKETS)
    vec[_op_bucket(getattr(lhs, "op", "") or "")] = 1.0
    vec[_OP_BUCKETS + _op_bucket(getattr(rhs, "op", "") or "")] = 1.0
    vec.append(float(len(getattr(lhs, "args", ()) or ())))
    vec.append(float(len(getattr(rhs, "args", ()) or ())))
    vec.append(1.0 if getattr(rule, "check", None) else 0.0)
    vec.append(1.0 if getattr(rule, "derive", None) else 0.0)
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
