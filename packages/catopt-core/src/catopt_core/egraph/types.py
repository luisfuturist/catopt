"""E-graph node/class/union-find/rewrite data types."""

# ruff: noqa: RUF003 — math notation in comments/docstrings
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from catopt_core.ir import Op, op_repr


@dataclass(frozen=True)
class ENode:
    """A term node in the e-graph: an op name with e-class children."""

    op: str
    children: tuple[int, ...]
    attrs: tuple[tuple[str, Any], ...] = ()


@dataclass
class EClass:
    """An equivalence class of semantically-equivalent terms."""

    id: int
    nodes: set[ENode] = field(default_factory=set)
    cache: dict[str, Any] = field(default_factory=dict)
    # Lazily-built ``op -> member enodes`` index used by the matcher;
    # invalidated whenever ``nodes`` is mutated (union / rebuild).
    by_op: dict[str, list] | None = None


class UnionFind:
    """Union-find (disjoint-set with path compression + union by rank)."""

    def __init__(self) -> None:
        """Initialise the empty parent and rank lists."""
        self.parent: list[int] = []
        self.rank: list[int] = []

    def make(self) -> int:
        """Add a new singleton set; return its index."""
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        return idx

    def find(self, x: int) -> int:
        """Return the canonical root of ``x`` (path halving)."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """Merge the sets of ``a`` and ``b``; True if they differed."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass(frozen=True)
class Rewrite:
    """A rewrite rule: lhs -> rhs (both are pattern terms).

    Patterns are terms where leaf variables are *strings* (metavariables).

    ``check`` is an optional side condition: a predicate over the
    *resolved* binding ``{metavar: representative_term}``.  Rules that
    are only valid for particular SHAPES (e.g. a scale that must be
    per-row or per-channel) declare it here — the matcher alone cannot
    see tensor types, and firing without the check produces well-typed
    but semantically wrong terms.

    ``derive`` is an optional computed-attribute hook: given the resolved
    binding it returns extra substitution entries (typically
    ``{"$attr:NAME": value}``) used when instantiating the RHS.  This is
    how a rule computes RHS attributes that are not present verbatim in
    the LHS — e.g. the uneven ``split`` sizes of an asymmetric pairing,
    derived from the bound weight shapes.  Returning ``None`` vetoes the
    rewrite.

    ``tags`` is the rule's intrinsic classification — constants from
    :mod:`catopt_core.laws.tags` (symmetry / expansive / subsumed /
    fusion / carrier / …).  It is *metadata about what the rule is*;
    scheduling priority is a ``RuleSet`` concern and deliberately does
    not live here, so the e-graph type stays free of scheduling policy.

    ``derivation`` is the axiom/lemma bookkeeping (measured by
    ``tools/law_coherence.py``): the names of shipped rules forming
    ONE recorded derivation of this law's instance — typically a
    single premise (a measured direct edge ``{A} ⇒ this``), i.e. a
    one-step proof from the kernel.  The empty tuple designates an
    axiom — a primitive (nothing shipped derives it) or the chosen
    representative of a derivability cycle: inverse pairs each prove
    each other, so which direction is axiom vs lemma is a convention
    (the alphabetical-first member), not a measurement.  See
    :attr:`kind` and ``project/retros/axiom-lemma-split.md``.
    """

    name: str
    lhs: Any
    rhs: Any
    law: str = ""
    check: Any = None  # Callable[[dict[str, Any]], bool] | None
    derive: Any = None  # Callable[[dict], dict | None] | None
    tags: frozenset[str] = frozenset()
    # Bounded-error axis (ε-laws): when set, this rewrite is a
    # *certified approximation* — ``‖lhs − rhs‖ ≤ error_bound`` in the
    # norm named by ``bound_norm`` (e.g. spectral on a substituted
    # weight).  Exact rules leave it None.  Certificates aggregate the
    # per-step bounds conservatively (triangle inequality); extraction
    # can constrain or report the total.
    error_bound: float | None = None
    bound_norm: str = "spectral"
    # Kernel taxonomy premise names — see the class docstring.  Not
    # serialised by ``rulecache`` (synthesised rules carry ``parents``
    # provenance instead); law-emit tooling may replay it into a real
    # ``Derivation`` certificate (:mod:`catopt_core.egraph.certs`).
    derivation: tuple[str, ...] = ()

    def __repr__(self) -> str:
        """Return a ``name: lhs -> rhs`` rendering."""
        return (
            f"{self.name}: {op_repr(self.lhs)} -> {op_repr(self.rhs)}"
        )

    @property
    def kind(self) -> str:
        """Return the kernel-taxonomy kind of this rule.

        ``"axiom"`` when no ``derivation`` is recorded (a kernel
        member), ``"redundant"`` for a derivable rule also carrying
        the ``REDUNDANT`` tag (a literal alpha-duplicate spelling of
        another shipped rule — the constant lives in
        :mod:`catopt_core.laws.tags`, one layer up, so the literal
        string is matched here), else ``"lemma"``.
        """
        if not self.derivation:
            return "axiom"
        if "redundant" in self.tags:
            return "redundant"
        return "lemma"


def _norm_attr_value(v: Any) -> Any:
    """Return the hashable form of an attr value.

    List-valued attrs are unhashable — enode attr keys carry tuples
    instead (the export boundary already canonicalises list→tuple;
    hand-minted terms get the same normalisation here).
    """
    return tuple(v) if isinstance(v, list) else v


def _pattern_attrs(op: Op) -> tuple[tuple[str, Any], ...]:
    return (
        tuple(
            sorted(
                (k, _norm_attr_value(v)) for k, v in op.attrs.items()
            )
        )
        if op.attrs
        else ()
    )


class _LeafRegistry:
    """Maps string keys to leaf term objects and vice-versa."""

    _key_to_term: ClassVar[dict[str, Any]] = {}

    @classmethod
    def register(cls, term: Any) -> str:
        key = repr(term)
        cls._key_to_term[key] = term
        return key

    @classmethod
    def decode(cls, key: str) -> Any:
        return cls._key_to_term.get(key, key)
