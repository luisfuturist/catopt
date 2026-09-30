"""Compiled LHS patterns and the frozen-read match epoch.

Plan 0010, lever 1a: instead of interpreting the LHS pattern tree on
every candidate (``isinstance`` ladder + ``_pattern_attrs`` recomputed
per node visit), each rule's pattern is compiled **once** into a tree
of :class:`_PVar` / :class:`_PLeaf` / :class:`_POp` nodes carrying
everything the match needs precomputed — the normalised attr tuple,
the attr key set for the exact-equality check, the leaf
:class:`ENode` for registry lookup, and the per-child op constraints
that feed targeted eligibility (lever 1b).

:class:`_Epoch` is the *frozen read view* behind streaming matching
(lever 2b): a match enumeration is lazy, and ``apply_rule`` mutates
the graph between yielded substitutions.  The eager semantics it
replaces enumerated the whole class against the state at call time,
so the streaming matcher must answer ``find``/class-membership
queries *as of enumeration start*.  The epoch journals the only
destructive mutations ``apply_rule`` can perform — class merges —
as four small overlays:

* ``ovr`` — every e-class id that ceased to be canonical maps to
  itself (its frozen root).  Union writes ``uf.parent[root]`` only
  at roots, so a root-level overlay captures every merge.
* ``jph`` — *journal of path halving*: while an epoch is live,
  ``EGraph.find`` still compresses paths but records each touched
  node's pre-write parent here (first write wins — the epoch-start
  parent), so the frozen walk replays the untouched structure.
* ``dead`` — the ``EClass`` object of each deleted canonical id, so
  member lookups for pre-freeze classes still resolve.
* ``added`` — per surviving class, the member e-nodes merged in
  *after* the freeze; frozen membership subtracts them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from catopt_core.egraph.types import EClass, ENode, _pattern_attrs
from catopt_core.ir import Op

__all__ = [
    "_Epoch",
    "_PLeaf",
    "_POp",
    "_PVar",
    "_Prog",
    "compile_pattern",
]


@dataclass(frozen=True)
class _PVar:
    """Metavariable pattern leaf (a bare ``str`` in the pattern)."""

    name: str


@dataclass(frozen=True)
class _PLeaf:
    """Concrete-leaf pattern (``Const``/``Param``/``Var``/other terms).

    ``key`` is the leaf-registry key (``repr`` of the term); ``enode``
    is the prebuilt ``("leaf", (), (("key", key),))`` node the matcher
    probes in ``_node_to_class``; ``term`` is the original pattern
    object — kept so RHS instantiation can re-register it.
    """

    key: str
    term: Any
    enode: ENode


@dataclass(frozen=True)
class _POp:
    """Compiled op pattern: head op, children, attr program.

    ``attrs`` is the normalised attr tuple (``_pattern_attrs`` applied
    once at compile time); ``keyset`` is the attr-name set for the
    exact-equality gate; ``child_reqs`` lists ``(position, op)`` for
    every child position whose pattern is itself an op — the targeted
    eligibility constraints consumed by ``_candidate_classes``.
    """

    op: str
    children: tuple
    attrs: tuple
    keyset: frozenset
    child_reqs: tuple


@dataclass(frozen=True)
class _Prog:
    """A compiled pattern plus its eligibility metadata.

    ``head_op``/``head_arity`` describe the root for the
    ``(op, arity)`` class index (``None``/``-1`` for non-op roots);
    ``leaf_key`` is the registry key for concrete-leaf roots;
    ``child_reqs`` carries the root's child-op constraints.
    """

    root: Any  # _PVar | _PLeaf | _POp
    head_op: str | None
    head_arity: int
    leaf_key: str | None
    child_reqs: tuple


def _pnode(pattern: Any) -> Any:
    """Compile one pattern subtree into its match node."""
    if isinstance(pattern, str):
        return _PVar(pattern)
    if isinstance(pattern, Op):
        children = tuple(_pnode(a) for a in pattern.args)
        attrs = _pattern_attrs(pattern)
        child_reqs = tuple(
            (i, c.op)
            for i, c in enumerate(children)
            if isinstance(c, _POp)
        )
        return _POp(
            pattern.op,
            children,
            attrs,
            frozenset(k for k, _ in attrs),
            child_reqs,
        )
    key = repr(pattern)
    return _PLeaf(key, pattern, ENode("leaf", (), (("key", key),)))


def compile_pattern(pattern: Any) -> _Prog:
    """Compile a whole pattern term into a :class:`_Prog`."""
    root = _pnode(pattern)
    if isinstance(root, _POp):
        return _Prog(
            root, root.op, len(root.children), None, root.child_reqs
        )
    if isinstance(root, _PVar):
        return _Prog(root, None, -1, None, ())
    return _Prog(root, None, -1, root.key, ())


@dataclass
class _Epoch:
    """Frozen-read journal for one streaming match enumeration.

    Populated by :meth:`EGraph.union` while the enumeration is
    suspended between yielded substitutions; read by the frozen
    ``_mfind`` / ``_mclass`` / ``_mnodes`` accessors.  ``watermark``
    is the ``_next_id`` at freeze time — e-class ids at or above it
    are invisible to the epoch (they did not exist when the eager
    semantics enumerated).
    """

    ovr: dict[int, int] = field(default_factory=dict)
    jph: dict[int, int] = field(default_factory=dict)
    dead: dict[int, EClass] = field(default_factory=dict)
    added: dict[int, set] = field(default_factory=dict)
    nlists: dict = field(default_factory=dict)
    watermark: int = -1
