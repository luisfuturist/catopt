"""Composable rule sets — ``RuleSet`` and the named presets.

A *law* is the mathematical identity (``Rewrite.law``); a *rule* is
one ``Rewrite`` (a direction/instance of a law); a ``RuleSet`` is a
named, composable collection of rules — the first-class replacement
for the loose group lists plus the pipeline's old ``ruleset: str``
switch and its hidden ``_SUBSUMED`` / ``_EXPANSIVE_RULES`` filters.

* Composition is by **rule name** (a rule's stable identity):
  ``A + B`` unions, ``A - B`` subtracts, ``A & B`` intersects.
  Same-name/different-rule collisions **raise** — silent shadowing
  across packages is a real bug source; :meth:`RuleSet.union` takes
  ``override=True`` as the deliberate escape hatch.
* ``tags`` live on the rules themselves (intrinsic metadata — see
  :mod:`catopt_core.laws.tags`); ``priorities`` live on the set
  (scheduling metadata — ``rule name -> int``, lower = earlier;
  :data:`EARLY` / :data:`NORMAL` / :data:`LATE`).
* Presets are *named* sets built with the same algebra, so
  ``DEFAULT + SYMMETRY == FULL`` and every preset is a composition,
  not a new hand-maintained list.

Known limit (recorded in plan 0009): a ``RuleSet`` is a flat list —
there is no way to say "these rules only fire at ``linear`` ops" or
"only at the root".  A ``scoped(op=..., rules=...)`` combinator is the
documented placeholder for carrier- or region-specific rules and is
deliberately not implemented here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.laws import tags
from catopt_core.laws.attention import ATTENTION_RULES
from catopt_core.laws.scan import SCAN_DIAG_LAWS, SCAN_LAWS
from catopt_core.laws.tensor import (
    ALL_RULES_WITH_LAYOUT,
    CATEGORICAL_RULES,
    SIMPLIFICATION_RULES,
)

__all__ = [
    "ATTENTION",
    "CARRIERS",
    "CARRIER_SEARCH",
    "CATEGORICAL",
    "DEFAULT",
    "EARLY",
    "FULL",
    "FUSION",
    "LATE",
    "NORMAL",
    "PRESETS",
    "SIMPLIFICATION",
    "SYMMETRY",
    "WITH_LAYOUT",
    "RuleSet",
    "preset",
]

#: Scheduling priorities — ``rule name -> int`` on the ``RuleSet``,
#: lower fires earlier.  Most rules stay ``NORMAL``; ``EARLY`` /
#: ``LATE`` are the named common cases (finer values stay available —
#: plan 0010's scheduler orders by this map).
EARLY = 0
NORMAL = 10
LATE = 20


def _rules_of(other: Any) -> tuple[Rewrite, ...]:
    """Return the member rules of a ``RuleSet`` or plain iterable."""
    return other.rules if isinstance(other, RuleSet) else tuple(other)


def _name_of(other: Any) -> str:
    """Return a display name for the operand (``RuleSet`` or list)."""
    return str(getattr(other, "name", "rules"))


def _prio_of(other: Any) -> Mapping[str, int]:
    """Return the priority map a ``RuleSet`` carries (``{}`` for lists)."""
    prio = getattr(other, "priorities", None)
    return prio if prio else {}


@dataclass(frozen=True, eq=False)
class RuleSet:
    """A named, composable collection of rewrite rules.

    ``rules`` is the member tuple (deduplicated by rule name at
    construction — duplicate names raise).  ``priorities`` maps rule
    names to scheduling ints (see :data:`EARLY` / :data:`NORMAL` /
    :data:`LATE`); ``description`` is free-form provenance.  Names and
    descriptions are labels, not identity: equality is set equality
    over rule *names* plus the priority map — ``A + B == B + A`` and
    ``(A + B) - B == A`` hold regardless of member order.

    The set satisfies :class:`catopt_core.ports.RuleSetLike` — it is
    directly iterable as ``RuleLike`` objects for ``EGraph.run``.
    """

    name: str
    rules: tuple[Rewrite, ...] = ()
    priorities: Mapping[str, int] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        """Normalise ``rules`` to a tuple and reject duplicate names."""
        rs = tuple(self.rules)
        object.__setattr__(self, "rules", rs)
        seen: set[str] = set()
        for r in rs:
            if r.name in seen:
                raise ValueError(
                    f"duplicate rule name {r.name!r} in "
                    f"RuleSet {self.name!r}"
                )
            seen.add(r.name)
        object.__setattr__(self, "priorities", dict(self.priorities))

    # -- value semantics ------------------------------------------------

    def __eq__(self, other: Any) -> bool:
        """Set equality over rule names, plus the priority map."""
        if not isinstance(other, RuleSet):
            return NotImplemented
        return {r.name for r in self.rules} == {
            r.name for r in other.rules
        } and self.priorities == other.priorities

    def __hash__(self) -> int:
        """Hash matching :meth:`__eq__` — names plus priorities."""
        return hash(
            (
                frozenset(r.name for r in self.rules),
                frozenset(self.priorities.items()),
            )
        )

    # -- container protocol -----------------------------------------

    def __iter__(self) -> Iterator[Rewrite]:
        """Iterate the member rules, in order."""
        return iter(self.rules)

    def __len__(self) -> int:
        """Return the member count."""
        return len(self.rules)

    def __contains__(self, item: Any) -> bool:
        """Membership by rule name — ``"comm_add" in rs`` or a Rewrite."""
        n = (
            item
            if isinstance(item, str)
            else getattr(item, "name", item)
        )
        return any(r.name == n for r in self.rules)

    # -- algebra -----------------------------------------------------

    def __add__(self, other: Any) -> RuleSet:
        """Union by rule name — same-name/different-rule raises."""
        return self.union(other)

    def union(self, other: Any, *, override: bool = False) -> RuleSet:
        """Union with *other* (``RuleSet`` or rule iterable).

        Same name + same rule is a no-op (``A + A == A``); same name +
        *different* rule raises — pass ``override=True`` to replace
        the existing member deliberately.
        """
        out = list(self.rules)
        by_name = {r.name: i for i, r in enumerate(out)}
        for r in _rules_of(other):
            i = by_name.get(r.name)
            if i is None:
                by_name[r.name] = len(out)
                out.append(r)
            elif out[i] != r:
                if not override:
                    raise ValueError(
                        f"rule name {r.name!r} already in "
                        f"{self.name!r} (from {_name_of(other)!r}) — "
                        "pass override=True to replace it deliberately"
                    )
                out[i] = r
        prio = {**_prio_of(other), **self.priorities}
        return RuleSet(
            f"{self.name}+{_name_of(other)}", tuple(out), prio
        )

    def __sub__(self, other: Any) -> RuleSet:
        """Subtract *other*'s rule names — ``(A + B) - B == A``."""
        names = {r.name for r in _rules_of(other)}
        rules = tuple(r for r in self.rules if r.name not in names)
        return RuleSet(
            f"{self.name}-{_name_of(other)}",
            rules,
            self._prio_for(rules),
        )

    def __and__(self, other: Any) -> RuleSet:
        """Intersect by rule name (members taken from ``self``)."""
        names = {r.name for r in _rules_of(other)}
        rules = tuple(r for r in self.rules if r.name in names)
        return RuleSet(
            f"{self.name}&{_name_of(other)}",
            rules,
            self._prio_for(rules),
        )

    # -- subsets ------------------------------------------------------

    def named(self, *names: str) -> RuleSet:
        """Return the subset of members whose rule name is in *names*."""
        want = set(names)
        rules = tuple(r for r in self.rules if r.name in want)
        return RuleSet(
            f"{self.name}[{','.join(names)}]",
            rules,
            self._prio_for(rules),
        )

    def tagged(self, *tags_: str) -> RuleSet:
        """Return the subset of members carrying any of *tags_*."""
        want = set(tags_)
        rules = tuple(r for r in self.rules if r.tags & want)
        return RuleSet(
            f"{self.name}.tagged",
            rules,
            self._prio_for(rules),
        )

    def _prio_for(self, rules: tuple[Rewrite, ...]) -> dict[str, int]:
        """Return this set's priorities restricted to *rules*."""
        names = {r.name for r in rules}
        return {k: v for k, v in self.priorities.items() if k in names}

    # -- scheduling ----------------------------------------------------

    def priority_of(self, rule: Any) -> int:
        """Return the scheduling priority of *rule* (name or Rewrite)."""
        n = rule if isinstance(rule, str) else rule.name
        return self.priorities.get(n, NORMAL)

    def with_priorities(self, **prio: int) -> RuleSet:
        """Return a copy with the priority map merged over ``**prio``.

        Unknown rule names raise — a priority for a rule that is not
        in the set can never fire and is a typo, not a default.
        """
        unknown = sorted(set(prio) - {r.name for r in self.rules})
        if unknown:
            raise KeyError(
                f"with_priorities: {unknown} not in {self.name!r}"
            )
        return replace(self, priorities={**self.priorities, **prio})


# ---------------------------------------------------------------------------
#  Presets — the named sets the pipeline composes
# ---------------------------------------------------------------------------

#: Every core rule, deduplicated by construction (the group lists are
#: disjoint by name).
_ALL = tuple(SIMPLIFICATION_RULES) + tuple(CATEGORICAL_RULES)


def _tagged(pool: Iterable[Rewrite], *t: str) -> tuple[Rewrite, ...]:
    want = set(t)
    return tuple(r for r in pool if r.tags & want)


#: The basic algebraic simplifications — the historical
#: ``SIMPLIFICATION_RULES`` group, including the comm/assoc monoid
#: laws (which also carry the opt-in ``SYMMETRY`` tag).
SIMPLIFICATION = RuleSet(
    "simplification",
    tuple(SIMPLIFICATION_RULES),
    description=(
        "basic algebraic simplification: monoid/group laws plus the "
        "silu/square decompositions"
    ),
)

#: The categorical laws *as the pipeline runs them* — the
#: ``CATEGORICAL_RULES`` group minus the pairing-subsumed product
#: folds (the non-local pairing pass owns those).  This is exactly
#: the historical ``ruleset="categorical"`` selection.
CATEGORICAL = RuleSet(
    "categorical",
    tuple(r for r in CATEGORICAL_RULES if tags.SUBSUMED not in r.tags),
    description=(
        "distributivity, naturality, products and the softmax fold — "
        "minus the pairing-subsumed product folds"
    ),
)

#: Structural folds that collapse a consumer pattern into one kernel:
#: the subsumed product rules (swiglu / parallel-mul / qkv), the sdpa
#: mask-fold family and ``gqa_absorb_repeat``.
FUSION = RuleSet(
    "fusion",
    _tagged(_ALL, tags.FUSION),
    description=(
        "structural folds — fused projections and the flash-attention "
        "transform (includes the pairing-subsumed members)"
    ),
)

#: The attention-path laws — rotary composition and scale
#: commutation, the right-multiply absorb, and the score-scale
#: migration.  Opt-in: the uniform-scale/view commutation pairs are
#: closure-generating on any graph with view ops, and the rotary
#: folds pay only where the exported spellings occur — see
#: :mod:`catopt_core.laws.attention` for the applicability class.
ATTENTION = RuleSet(
    "attention",
    tuple(ATTENTION_RULES),
    description=(
        "the attention-path laws — rope composition/commutation, "
        "right-multiply absorb, score-scale migration (opt-in; the "
        "uniform-scale/view pairs are closure-generating)"
    ),
)

#: The closure-generating symmetry laws — opt-in, never in
#: :data:`DEFAULT`.  comm/assoc plus the scale-hoist naturality rules.
SYMMETRY = RuleSet(
    "symmetry",
    _tagged(_ALL, tags.SYMMETRY),
    description=(
        "comm/assoc and the scale-hoist naturality rules — the "
        "Catalan-blowup closure generators (opt-in)"
    ),
)

#: The core-side share of the ``CARRIERS`` preset — *empty by
#: design*.  The preset names the carrier-package families (om /
#: trace / xc / decode); those live in ``catopt_carriers`` and core
#: cannot import them (the hexagonal boundary).  The orchestrator
#: composes ``carriers.CARRIERS`` into ``DEFAULT_RULES``.  Core's
#: own scan monoids are tagged ``SCAN``, not ``CARRIER``, and sit in
#: :data:`CARRIER_SEARCH`.
CARRIERS = RuleSet(
    "carriers",
    (),
    description=(
        "the carrier-package families (om / trace / xc / decode) — "
        "empty core-side; the orchestrator composes "
        "catopt_carriers.CARRIERS"
    ),
)

#: The core-side share of ``CARRIER_SEARCH`` — today's
#: ``CARRIER_LAWS`` regime set is the scan monoids plus the
#: carrier-package decode / om / trace families; the orchestrator's
#: ``regime`` composes the full set.
CARRIER_SEARCH = RuleSet(
    "carrier_search",
    tuple(SCAN_LAWS) + tuple(SCAN_DIAG_LAWS),
    description=(
        "the regime search set, core-side — the dense and "
        "diagonal-affine scan monoids (the orchestrator composes "
        "decode / om / trace on top)"
    ),
)

#: Today's ``ALL_RULES_WITH_LAYOUT`` — the whole core equational
#: surface plus the transpose/layout migration laws.
WITH_LAYOUT = RuleSet(
    "with_layout",
    tuple(ALL_RULES_WITH_LAYOUT),
    description="the core set plus the layout-migration laws",
)

_BASE = CATEGORICAL + SIMPLIFICATION + FUSION + CARRIERS

#: The recommended set: the categorical + simplification + fusion +
#: carriers groups, minus the pairing-subsumed folds and the opt-in
#: symmetry generators.  (Core-side ``CARRIERS`` is empty — the
#: orchestrator's ``DEFAULT_RULES`` composes
#: ``catopt_carriers.CARRIERS`` on top.)  ``DEFAULT + SYMMETRY`` is
#: the opt-in full set (:data:`FULL`).
DEFAULT = RuleSet(
    "default",
    tuple(
        r for r in _BASE if not r.tags & {tags.SUBSUMED, tags.SYMMETRY}
    ),
    description=(
        "the recommended set — categorical + simplification + fusion "
        "+ carriers, minus subsumed and symmetry (the orchestrator "
        "composes the carrier-package sets on top)"
    ),
)

FULL = replace(
    DEFAULT + SYMMETRY,
    name="full",
    description="DEFAULT + SYMMETRY — the complete core surface",
)

#: Name -> preset lookup (a string may resolve to a preset for
#: convenience — the ``RuleSet`` stays the first-class value).
PRESETS: dict[str, RuleSet] = {
    p.name: p
    for p in (
        SIMPLIFICATION,
        CATEGORICAL,
        FUSION,
        SYMMETRY,
        ATTENTION,
        CARRIERS,
        CARRIER_SEARCH,
        WITH_LAYOUT,
        DEFAULT,
        FULL,
    )
}


def preset(name: str) -> RuleSet:
    """Resolve a preset *name* to its ``RuleSet``.

    Raises ``ValueError`` for unknown names — the same contract the
    retired ``ruleset: str`` switch had.
    """
    rs = PRESETS.get(name)
    if rs is None:
        raise ValueError(
            f"Unknown ruleset: {name!r} — known presets: "
            f"{sorted(PRESETS)}"
        )
    return rs
