"""E-graph node/class/union-find/rewrite data types."""

# ruff: noqa: RUF003 — math notation in comments/docstrings
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, ClassVar

from catopt_core.ir import Op, op_repr


@dataclass(frozen=True, eq=False)
class ENode:
    """A term node in the e-graph: an op name with e-class children.

    ``attrs`` stores values raw — the matcher reads them with ``!=``,
    keeping numeric leniency (a pattern ``dim=0`` still matches a node
    spelled ``dim=0.0``, mirroring the ``Const`` leniency
    ``match_pattern`` keeps for leaves).  Node *identity*, though, is
    spelling-strict: ``__eq__``/``__hash__`` compare the repr-keyed
    ``_sig`` so ``min=0`` and ``min=0.0`` are distinct enodes — the
    :class:`catopt_core.ir.Const` precedent applied to the attr tuple.
    Under field-compare the numeric tower (``0 == 0.0``, shared hash)
    merged both spellings into one e-class — the same silent
    corruption the ``Const`` fix removed from leaf identity.
    """

    op: str
    children: tuple[int, ...]
    attrs: tuple[tuple[str, Any], ...] = ()
    _sig: tuple = field(init=False, repr=False, compare=False)
    _h: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Precompute the spelling-strict signature and its hash."""
        sig = (
            self.op,
            self.children,
            tuple((k, repr(v)) for k, v in self.attrs),
        )
        object.__setattr__(self, "_sig", sig)
        object.__setattr__(self, "_h", hash(sig))

    def __eq__(self, other: Any) -> bool:
        """Compare the strict signature — attr spellings, not numbers."""
        if not isinstance(other, ENode):
            return NotImplemented
        return self._sig == other._sig

    def __hash__(self) -> int:
        """Return the cached signature hash."""
        return self._h


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

    ``cond`` is an optional *declarative* side condition — pure data
    (a tuple tree per :mod:`catopt_core.laws.cond`), serializable to
    JSON, evaluated by that module's interpreter over the same
    ``bound`` environment ``check`` sees.  ``check`` and ``cond`` may
    coexist: ``cond`` carries the expressible part, ``check`` the
    procedural remainder.  ``__post_init__`` folds ``cond`` into the
    ``check`` hook so every evaluation site (apply_rule, certificate
    replay, term-level matching, meta's composite guards) honours the
    conjunction through the single ``rule.check`` convention.

    ``dspec`` is the same move for ``derive``: an optional *declarative*
    derive spec — pure data per :mod:`catopt_core.laws.cond` (a
    ``{NAME: expr}`` map or tuple of pairs), evaluated against the same
    ``bound`` environment to produce the ``{"$attr:NAME": value}`` map
    the ``derive`` contract returns.  ``dspec`` and ``derive`` may
    coexist: ``dspec`` carries the expressible part, ``derive`` the
    procedural remainder; ``__post_init__`` folds ``dspec`` into the
    ``derive`` hook.  For convenience ``derive=`` also *accepts* a spec
    (non-callable data, or an ``as_derive`` partial) — it is recast
    into ``dspec`` at construction.
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
    # provenance instead); ``tools/law_lemma_cert.py`` replays it into
    # a real :class:`Certificate` (see :mod:`catopt_core.egraph.certs`
    # for the data codec).
    derivation: tuple[str, ...] = ()
    # Declarative side condition — pure data (see the class docstring).
    # Stored as data for serialization (the lemma-store seam); folded
    # into ``check`` at construction.
    cond: Any = None
    # Declarative derive spec — pure data (see the class docstring).
    # Folded into ``derive`` at construction.
    dspec: Any = None

    def __post_init__(self) -> None:
        """Fold ``cond``/``dspec`` into the ``check``/``derive`` hooks.

        A rule carrying both evaluates ``cond`` first, then ``check``
        — the conjunction IS the side condition, folded here so every
        evaluation site (``apply_rule``, certificate replay, term-level
        matching, meta's composite guards) keeps the single
        ``rule.check`` convention.  ``dspec`` folds the same way into
        ``rule.derive`` — the spec runs first, then any procedural
        remainder.  The lazy import avoids a cycle:
        ``catopt_core.laws`` depends on this module at load time.
        """
        if self.cond is not None:
            from catopt_core.laws.cond import (
                compile_guard,
                cond_from_data,
            )

            # Canonicalise to the tuple-tree form — a list tree (as
            # ``json.loads`` hands back) evaluates identically but is
            # unhashable and unequal to its canonical twin.
            object.__setattr__(self, "cond", cond_from_data(self.cond))
            object.__setattr__(
                self, "check", compile_guard(self.cond, self.check)
            )
        _fold_dspec(self)

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


def _spec_from_derive(drv: Any) -> Any:
    """Return the derive spec embedded in a ``derive`` argument, if any.

    ``derive=`` accepts the spec itself: non-callable data is a spec
    verbatim, and an ``as_derive(spec)`` partial — recognised by its
    ``eval_derive`` target, one positional arg, no keywords — carries
    the spec it was built from.  Anything else (``None``, a callable)
    carries no spec.
    """
    if drv is not None and not callable(drv):
        return drv
    if (
        isinstance(drv, functools.partial)
        and len(drv.args) == 1
        and not drv.keywords
        and getattr(drv.func, "__module__", "")
        == "catopt_core.laws.cond"
        and getattr(drv.func, "__name__", "") == "eval_derive"
    ):
        return drv.args[0]
    return None


def _fold_dspec(rule: Rewrite) -> None:
    """Fold *rule*'s ``dspec`` into its ``derive`` hook.

    The spec is canonicalised (``derive_from_data`` — dicts and pair
    lists become sorted tuple pairs, hashable and equal to their
    source-spelled twins) and composed with any procedural ``derive``
    through ``compile_derive`` — the same fold ``cond`` gets, keeping
    every evaluation site on the single ``rule.derive`` convention.
    A spec given twice (``dspec`` plus a spec-shaped ``derive``) is a
    ``ValueError``, not a silent choice.  The lazy import avoids a
    cycle: ``catopt_core.laws`` depends on this module at load time.
    """
    spec = rule.dspec
    drv = rule.derive
    source = _spec_from_derive(drv)
    if source is None:
        source = spec
    elif spec is not None:
        raise ValueError(
            f"rule {rule.name!r}: derive spec given twice "
            "(dspec and a non-callable derive)"
        )
    else:
        # A spec sat in the derive slot — recast it to data and
        # re-fold, so the hook is the canonical spec-closure (one code
        # object per shape, which ``laws.serialize._proc_derive``
        # probes) rather than the partial it arrived as.
        drv = None
    if source is None:
        return
    from catopt_core.laws.cond import compile_derive, derive_from_data

    spec = derive_from_data(source)
    object.__setattr__(rule, "dspec", spec)
    object.__setattr__(rule, "derive", compile_derive(spec, drv))


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
