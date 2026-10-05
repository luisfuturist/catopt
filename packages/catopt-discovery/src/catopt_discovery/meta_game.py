"""The law-proposal game — a learned player builds candidate laws.

The pipeline (``catopt_discovery.pipeline``) discovers laws by
*enumeration + filter*: five generators emit a fixed pool, then the
numeric oracle, the derivability oracle and the corpus measurements
referee it.  The proposers are not players — nothing inside them
*chooses* a construction because a previous choice scored well.  The
stated endgame (the RL stage-7 retros, ADR 0003) is a player whose
policy IS the proposer.

This tool is the bounded first cut of that player.

The game
--------

* **state** — a partially-built candidate law: an LHS pattern tree,
  then (once the LHS is complete) an RHS pattern tree, each with open
  *holes* (:class:`_Slot`).  ``BuildGame`` is the board.
* **action** — fill the leftmost hole with an op (creating one hole
  per operand), a metavariable (``U`` / ``V`` / ``W`` / ``X`` —
  reuse binds the same subterm, which is how ``sub(x,x)``-style
  equalities are expressible), or a literal ``Const``.
  Attribute-carrying ops get their attr slots filled automatically
  with *shared* attr metavariables named ``<op>.<key>``, so two
  ``select`` nodes enforce ``dim``/``index`` equality structurally —
  the naturality precondition ``select_mul`` needs.
* **legality** — the rule book is corpus knowledge, not a score: an
  op child must be census-attested at ``(parent, position)`` on the
  LHS (a triple absent from the corpus can never match); a ``Const``
  leaf likewise; an already-bound metavariable may be re-placed only
  where a structural-sharing census attests equal subterms under the
  lowest common ancestor of the bindings; ``_numeric_true``-untestable
  ops are excluded from the vocabulary.  On the RHS — the generated
  side — only the op budget and bound-metavariable rules apply.
* **terminal** — both sides complete (≤ ``_MAX_OPS_SIDE`` ops on the
  LHS, ≤ ``_MAX_OPS_RHS`` on the RHS).  The pair is then a fireable
  candidate: unbound RHS attr metavariables are resolved by a
  generic *attr bridge* (same attr key, singleton-tuple unwrapped —
  e.g. ``sum``'s ``dim=(-1,)`` → ``softmax``'s ``dim=-1``), and a
  generic side condition requires bound *bool* attrs (``keepdim`` …)
  to hold.  Both are honest grammar features, not per-law patches.
* **start states** — with probability ``--seed-frac`` the LHS begins
  as a census-frequent skeleton ``f(g(_), h(_))`` — the ``(op,
  child-ops)`` tuples ``law_shape_census`` counts *and* its
  abstracted per-shape keys — sampled ∝ ``log1p(count)``, so the
  player is pointed at shapes the corpus actually contains.

The reward
----------

The pipeline's own stages are the score, via ``Referee``:

1. **truth gate** — instantiate the candidate on the first real
   corpus-slice match (``_term_match`` + check/derive) and run the
   numeric oracle ``catopt_discovery.proposal._numeric_true``.  One oracle call
   per distinct candidate; a false or undecidable proposal scores 0.
   No match at all costs *no* oracle call (the candidate never
   reached the referee).  Equality is symmetric, so one call
   referees *both* firing orientations, and the dedup cache covers
   the swapped pair.
2. **fires** — ``catopt_discovery.impact._probe`` runs the rule alone over the
   slice (a few models) and counts firings in each direction.
3. **pay** — the extracted-cost delta under the pipeline cost model.

``score = truth · (1 + 0.2·fires + 2·paid - 5·verify_fail +
10·max_rel_drop)``; truth is the referee, fires+pay is the score.

The player
----------

A small ``(state ⊕ action) -> logit`` MLP trained by REINFORCE with a
running-mean baseline (the ``catopt_torch.rl`` shape; the net here is
local because construction actions are not rule names).  A *fixed
corpus prior* is added to the learned logits — census-conditioned
child-op frequency decayed by hole depth, plus "the op occurs on the
other side", the commutation and swap-the-nesting moves, and
bound/fresh metavariable bias — disclosed, not hidden: the prior only
proposes, the oracle still referees, and REINFORCE can learn to
override it.

The comparison that matters
---------------------------

The currency is **oracle calls** (``_numeric_true`` invocations).
Yield = true+firing candidates (and the stricter new-true-firing,
the shippable proxy) per call, for

* the enumerative baseline — ``catopt_discovery.pipeline.propose``'s pool,
* a uniform player (no learning, no prior),
* the trained player (learned logits + prior).

Sanity check: does the player rediscover ``select_mul``- or
``softmax``-shaped equalities in-budget?

The guide seam (ADR 0004, plan 0017 stage 2)
--------------------------------------------

The construction player above is itself only one move-generator.
ADR 0004 §3 makes this module's real role the *meta*-game: the
policy does not pick rewrites, it picks **which generator invests
compute where**.  The investable inventory is the pipeline's own
proposal sources — census naturality, mixed-view naturality,
pattern recognition, the shape-aware schemas, the algebraic grammar
— plus ``"build"``, the construction player itself (one draw is one
play).

An action is an :class:`Allocation` — a generator name plus a draw
count.  The currency is *proposals drawn*, each costing the shared
:class:`Referee` at most one oracle call — the same unit the yield
tables already count.  A :class:`Guide` reads a :class:`GuideObs` —
per-arm tallies plus the evidence store's ``latest_verdicts`` rows
(the observation source is the store, not a parallel accounting) —
and returns the next allocation.  Three baselines ship:
``EnumerationGuide`` (the pipeline's fixed order — the control
arm), ``RandomGuide`` (uniform arm selection) and ``LearnedGuide``
(the ``_PolicyNet`` + REINFORCE machinery, re-pointed at arms).

Run::

    .venv/bin/python -m catopt_discovery.meta_game
    .venv/bin/python -m catopt_discovery.meta_game --episodes 600 --json /tmp/g.json
    .venv/bin/python -m catopt_discovery.meta_game --holdout select_mul
    .venv/bin/python -m catopt_discovery.meta_game --guide --budget 80

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from catopt_core.egraph import Rewrite
from catopt_core.egraph.terms import _term_instantiate, _term_match
from catopt_core.ir import Const, Op, op_repr
from catopt_core.laws import ALL_RULES
from torch import nn
from torch.distributions import Categorical

# Sibling tools own the corpus, the oracles and the measurements.
from catopt_discovery import evidence as ev_store
from catopt_discovery import pipeline as lpl
from catopt_discovery import proposal as lp
from catopt_discovery.census import (
    CorpusTerm,
    op_tuple_census,
    shape_census,
)
from catopt_discovery.impact import (
    TermCase,
    _bench_cases,
    _cost_fn,
    _iter_subterms,
    _probe,
    model_cases,
)
from catopt_discovery.shape_proposal import _sink
from catopt_discovery.vocab import classify, corpus_ops

__all__ = [
    "Allocation",
    "ArmStat",
    "BuildGame",
    "EnumerationGuide",
    "Guide",
    "GuideArena",
    "GuideObs",
    "LearnedGuide",
    "RandomGuide",
    "Referee",
    "Verdict",
    "compare_guides",
    "eval_baseline",
    "generator_pools",
    "main",
    "run_guide",
    "run_guide_experiment",
    "run_player",
    "train",
]

#: Maximum op nodes on one side of a candidate (the task bound).
_MAX_OPS_SIDE = 6

#: Metavariable pool — distinct names bind distinct subterms.
_MV_NAMES = ("U", "V", "W", "X")

#: Literal leaves offered as actions.
_CONSTS = (0, 1)

#: Fixed prior weights added to the learned logits:
#: census child frequency (op, metavariable *and* const children are
#: counted — the census's ``·``/``const`` entries), "op occurs on the
#: other side", "metavariable already bound" (RHS) and "fresh
#: metavariable" (LHS — distinct bindings are what naturality and
#: factoring preconditions need).
_PRIOR_CHILD = 2.0
_PRIOR_REUSE = 2.0
_PRIOR_BOUND_MV = 1.2
_PRIOR_FRESH_MV = 0.4
_PRIOR_COMMUTE = 1.0

#: Swap-the-nesting prior: at the RHS root, an op that sits one level
#: down on the LHS is the naturality move (``f(g u, g v) -> g(f u v)``).
_PRIOR_SWAP_NEST = 1.2

#: Op-placement prior decays with hole depth: the corpus's frequent
#: shapes are shallow (≤ ~3 ops), so below the seeded skeleton level
#: leaves dominate and constructed LHSs stay applicable.
_PRIOR_DEPTH_DECAY = 0.55

#: Op budget per side.  The LHS gets the task bound; the RHS is
#: capped tighter because the laws the corpus rewards are shallow.
_MAX_OPS_RHS = 4

#: Hard ceiling on construction steps (already implied by the op cap).
_MAX_STEPS = 60


# ---------------------------------------------------------------------------
#  Partial pattern trees
# ---------------------------------------------------------------------------


@dataclass
class _Slot:
    """One node of a pattern under construction.

    ``kind`` is ``hole`` (unfilled), ``op``, ``mv`` (metavariable) or
    ``const``.  Op children are slots; attr values are metavariable
    name strings, auto-shared per ``(op, attr key)`` so equal ops
    enforce equal attrs.
    """

    kind: str = "hole"
    op: str = ""
    children: list[_Slot] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    name: str = ""
    value: Any = 0


def _first_hole(slot: _Slot, path: tuple = ()) -> tuple | None:
    """Return the child-index path of the leftmost hole, or ``None``."""
    if slot.kind == "hole":
        return path
    for i, c in enumerate(slot.children):
        p = _first_hole(c, (*path, i))
        if p is not None:
            return p
    return None


def _fill(slot: _Slot, path: tuple, new: _Slot) -> None:
    """Replace the hole at *path* with *new* (mutates *slot*)."""
    parent = slot
    for i in path[:-1]:
        parent = parent.children[i]
    if path:
        parent.children[path[-1]] = new
    else:
        # root replacement — copy fields into the existing object
        slot.kind, slot.op = new.kind, new.op
        slot.children = new.children
        slot.attrs = new.attrs
        slot.name, slot.value = new.name, new.value


def _n_ops(slot: _Slot) -> int:
    """Return the number of placed op nodes in *slot*."""
    if slot.kind != "op":
        return 0
    return 1 + sum(_n_ops(c) for c in slot.children)


def _n_holes(slot: _Slot) -> int:
    """Return the number of open holes in *slot*."""
    if slot.kind == "hole":
        return 1
    return sum(_n_holes(c) for c in slot.children)


def _node_at(slot: _Slot, path: tuple) -> _Slot:
    """Return the slot at *path*."""
    node = slot
    for i in path:
        node = node.children[i]
    return node


def _ops_present(slot: _Slot, out: set | None = None) -> set:
    """Return the set of op names placed in *slot*."""
    out = set() if out is None else out
    if slot.kind == "op":
        out.add(slot.op)
        for c in slot.children:
            _ops_present(c, out)
    return out


def _mint(slot: _Slot) -> Any:
    """Materialize a complete *slot* as a pattern term.

    Metavariable leaves become ``str`` (pattern metavars); attr values
    stay strings (attr metavars).  Returns ``None`` if ``Op.make``
    rejects the node (a malformed attr combination).
    """
    if slot.kind == "mv":
        return slot.name
    if slot.kind == "const":
        return Const(slot.value)
    if slot.kind == "hole":
        return "Z"  # safety net; terminals have no holes
    args = [_mint(c) for c in slot.children]
    try:
        return Op.make(slot.op, *args, **dict(slot.attrs))
    except Exception:
        return None


def _attr_mv_names(pat: Any, out: set | None = None) -> set:
    """Return the attr-metavariable names used in a minted pattern."""
    out = set() if out is None else out
    if isinstance(pat, Op):
        for v in pat.attrs.values():
            if isinstance(v, str):
                out.add(v)
        for a in pat.args:
            _attr_mv_names(a, out)
    return out


def _bool_attr_check(bound: dict) -> bool:
    """Enforce the generic side condition: bound bool attrs must hold.

    Bool attrs are preconditions, not values — a ``keepdim`` bound to
    ``False`` is a different (and differently-typed) site than the one
    the candidate was proposed from.  Requiring them to be truthy is
    the generic version of the shipped softmax fold's own check.
    """
    return all(
        not isinstance(v, bool) or v
        for k, v in bound.items()
        if k.startswith("$attr:")
    )


def _attr_bridge(unbound: frozenset) -> Any:
    """Return a ``derive`` resolving unbound RHS attr metavariables.

    Each unbound name ``<op>.<key>`` is resolved from a bound
    ``$attr:<op2>.<key>`` — the *same attr key* seen on the LHS —
    unwrapping singleton tuples (``sum``'s ``dim=(-1,)`` → -1).  This
    is the one generic 3-cell the game adds beyond pattern copying:
    it is how ``softmax(u, dim)``'s ``dim`` can come from ``sum``'s.
    ``None`` vetoes, exactly like a hand-written derive.
    """

    def derive(bound: dict) -> dict | None:
        out: dict[str, Any] = {}
        for name in sorted(unbound):
            key = name.rsplit(".", 1)[-1]
            cands = sorted(
                k
                for k in bound
                if k.startswith("$attr:") and k.endswith(f".{key}")
            )
            if not cands:
                return None
            v = bound[cands[0]]
            if isinstance(v, tuple) and len(v) == 1:
                v = v[0]
            out[f"$attr:{name}"] = v
        return out

    return derive


# ---------------------------------------------------------------------------
#  Vocabulary — the op alphabet + census priors, all corpus-derived
# ---------------------------------------------------------------------------


@dataclass
class Vocab:
    """Everything the game can place, derived from the corpus."""

    ops: tuple[str, ...]
    arity: dict[str, int]
    attr_keys: dict[str, tuple[str, ...]]
    classes: dict[str, str]  # op -> class label
    tuple_count: dict[tuple, int]  # (op, child-ops) -> count
    head_count: dict[str, int]
    child_count: dict[tuple, int]  # (parent, position, child) -> n
    pos_count: dict[tuple, int]  # (parent, position) -> total n
    share: dict[
        tuple, int
    ]  # (op, child-kinds) -> sites sharing a subterm
    max_count: int
    seeds: list[tuple[Any, int]]  # (skeleton-slot, census count)


def _seed_slot(
    op: str,
    kids: tuple,
    vocab_arity: dict,
    attr_keys: dict,
    ops_set: frozenset,
) -> _Slot | None:
    """Build the LHS skeleton ``f(g(_), h(_))`` for a census tuple."""
    if op not in ops_set or vocab_arity.get(op) != len(kids):
        return None
    root = _Slot(
        kind="op",
        op=op,
        attrs={k: f"{op}.{k}" for k in attr_keys.get(op, ())},
    )
    for kid in kids:
        if kid in ops_set:
            root.children.append(
                _Slot(
                    kind="op",
                    op=kid,
                    children=[_Slot() for _ in range(vocab_arity[kid])],
                    attrs={
                        k: f"{kid}.{k}" for k in attr_keys.get(kid, ())
                    },
                )
            )
        else:
            root.children.append(_Slot())
    return root


def _share_scan(terms: list[Any]) -> dict[tuple, int]:
    """Count, per census tuple, sites with a shared subterm.

    A metavariable placed twice on the LHS is an equality
    precondition: it can only match where two positions bind
    ``op_repr``-equal subterms.  For every op node in the corpus this
    scans its descendant positions for an equal pair and marks the
    node's ``(op, child-kinds)`` tuple.  The game consults it at the
    lowest common ancestor of the two bindings — the node under which
    the sharing must hold.
    """
    share: dict[tuple, int] = {}

    def _kids(s: Op) -> tuple:
        return tuple(
            a.op
            if isinstance(a, Op)
            else ("const" if isinstance(a, Const) else "·")
            for a in s.args
        )

    def _descs(s: Op, out: list, p: tuple) -> None:
        for i, a in enumerate(s.args):
            out.append(((*p, i), a))
            if isinstance(a, Op):
                _descs(a, out, (*p, i))

    for t in terms:
        for s in _iter_subterms(t):
            if not isinstance(s, Op):
                continue
            descs: list = []
            _descs(s, descs, ())
            reps = [op_repr(a) for _, a in descs]
            if len(set(reps)) < len(reps):
                key = (s.op, _kids(s))
                share[key] = share.get(key, 0) + 1
    return share


def _shape_slot(
    key: Any, attr_keys: dict, ops_set: frozenset
) -> _Slot | None:
    """Convert a census *shape* key to a (deeper) LHS skeleton.

    ``catopt_discovery.census.shape_key`` abstracts a real subterm to
    ``(op, attrs, children)`` with leaf placeholders — this re-mints
    it as a partial pattern: ops stay, leaf/const placeholders become
    holes, and concrete attr values become the shared per-``(op,
    key)`` attr metavariables the rest of the game uses.  ``None``
    when the shape uses an out-of-vocab op or exceeds the op budget.
    """
    if key[0] == "leaf":
        return _Slot()
    if key[0] == "const":
        return _Slot(kind="const", value=key[1])
    op, attrs, kids = key
    if op not in ops_set:
        return None
    slot = _Slot(
        kind="op",
        op=op,
        attrs={k: f"{op}.{k}" for k, _ in attrs},
    )
    for k in kids:
        c = _shape_slot(k, attr_keys, ops_set)
        if c is None:
            return None
        slot.children.append(c)
    if _n_ops(slot) > _MAX_OPS_SIDE:
        return None
    return slot


def build_vocab(
    terms: list[Any],
    counts: Any,
    shape_counts: Any,
    top_seeds: int,
) -> Vocab:
    """Derive the op alphabet, attr keys, classes and seed skeletons."""
    occ, arity = corpus_ops(terms)
    cls = {c.op: c for c in classify(terms).classes}
    # Ops the property probes could not evaluate at all can never
    # produce a verifiable equality — an oracle call on them is a
    # guaranteed ``None``, so they are dead moves, not candidates.
    ops = tuple(
        sorted(
            o
            for o, a in arity.items()
            if a <= 2
            and not (
                cls[o].pointwise is None and cls[o].reduction is None
            )
        )
    )
    attr_keys: dict[str, tuple[str, ...]] = {}
    for op in ops:
        keys: set[str] = set()
        for a, _ in occ.get(op, ()):
            keys |= set(a)
        attr_keys[op] = tuple(sorted(keys))
    classes: dict[str, str] = {}
    for op in ops:
        c = cls.get(op)
        if c is None:
            classes[op] = "other"
        elif c.view is True:
            classes[op] = "view"
        elif c.pointwise is True and arity.get(op) == 2:
            classes[op] = "pointwise-bin"
        elif c.pointwise is True:
            classes[op] = "pointwise-un"
        elif c.reduction is True:
            classes[op] = "reduction"
        else:
            classes[op] = "other"
    head: dict[str, int] = {}
    child: dict[tuple, int] = {}
    pos: dict[tuple, int] = {}
    tup = {k: v for k, v in counts.items()}
    for (op, kids), n in counts.items():
        head[op] = head.get(op, 0) + n
        for i, k in enumerate(kids):
            key = (op, i, k)
            child[key] = child.get(key, 0) + n
            pkey = (op, i)
            pos[pkey] = pos.get(pkey, 0) + n
    ops_set = frozenset(ops)
    seeds = [
        (s, n)
        for (op, kids), n in counts.most_common(top_seeds)
        if (s := _seed_slot(op, kids, arity, attr_keys, ops_set))
        is not None
    ]
    seeds += [
        (s, n)
        for k, n in shape_counts.most_common(top_seeds)
        if (s := _shape_slot(k, attr_keys, ops_set)) is not None
    ]
    return Vocab(
        ops=ops,
        arity=dict(arity),
        attr_keys=attr_keys,
        classes=classes,
        tuple_count=tup,
        head_count=head,
        child_count=child,
        pos_count=pos,
        share=_share_scan(terms),
        max_count=max(counts.values()) if counts else 1,
        seeds=seeds,
    )


# ---------------------------------------------------------------------------
#  The game — BuildGame is the board
# ---------------------------------------------------------------------------


class BuildGame:
    """A sequential law-construction game.

    Fills the leftmost hole of the LHS, then of the RHS.  The action
    set is fixed: one ``op:<name>`` per vocabulary op, ``mv:U|V|W``,
    ``const:0|1``.  Legality: op-node cap per side; RHS metavariables
    must be bound on the LHS (an unbound RHS metavar cannot
    instantiate).  ``candidate()`` mints the pattern pair plus the
    generic check / attr-bridge derive hooks.
    """

    def __init__(self, vocab: Vocab, rng: random.Random) -> None:
        """Bind the vocabulary and the RNG (seed sampling)."""
        self.v = vocab
        self.rng = rng
        self.actions: list[tuple[str, Any]] = (
            [("op", o) for o in vocab.ops]
            + [("mv", m) for m in _MV_NAMES]
            + [("const", c) for c in _CONSTS]
        )
        self.lhs = _Slot()
        self.rhs = _Slot()
        self.side = "lhs"
        self.bound: set[str] = set()
        self.mv_paths: dict[str, list[tuple]] = {}
        self.steps = 0
        self.done = False
        self.seed_name = ""

    def reset(self, seed: _Slot | None, name: str = "") -> None:
        """Start a play; *seed* is an optional LHS skeleton."""
        self.lhs = seed if seed is not None else _Slot()
        self.rhs = _Slot()
        self.side = (
            "lhs" if _first_hole(self.lhs) is not None else "rhs"
        )
        self.bound = set()
        self.mv_paths = {}
        self.steps = 0
        self.done = False
        self.seed_name = name

    def _cur(self) -> _Slot:
        return self.lhs if self.side == "lhs" else self.rhs

    def _ops(self, side: str) -> int:
        return _n_ops(self.lhs if side == "lhs" else self.rhs)

    def _op_cap(self) -> int:
        return _MAX_OPS_SIDE if self.side == "lhs" else _MAX_OPS_RHS

    def legal(self) -> list[int]:
        """Return the indices of the legal actions at the hole.

        Legality is the game's rule book, not a heuristic: an op
        placed on the LHS must be census-attested as a child of its
        parent *at that position* (a ``(parent, i, child)`` tuple with
        zero corpus count can never match — the move is dead, like an
        illegal chess move).  Metavariables and consts are always
        legal: a metavar binds whatever the site holds, and a ``Const``
        leaf may match a real constant.  On the RHS — the generated
        side — only the op budget and bound-metavariable rules apply.
        """
        cur = self._cur()
        path = _first_hole(cur)
        if path is None:
            raise RuntimeError("legal() on a term with no hole")
        cur_ops = self._ops(self.side)
        parent_op = None
        pos = 0
        if path:
            parent_op = _node_at(cur, path[:-1]).op
            pos = path[-1]
        lhs = self.side == "lhs"
        out = []
        for i, (kind, val) in enumerate(self.actions):
            if kind == "op":
                if cur_ops >= self._op_cap():
                    continue
                if lhs:
                    if path and not self.v.child_count.get(
                        (parent_op, pos, val), 0
                    ):
                        continue
                    if not path and not self.v.head_count.get(val, 0):
                        continue
                out.append(i)
            elif kind == "const":
                # a Const only matches a Const — legal on the LHS
                # only where the corpus puts a const child, and at
                # the RHS root only for leaf-RHS laws (``-> 0``)
                if lhs:
                    if path and self.v.child_count.get(
                        (parent_op, pos, "const"), 0
                    ):
                        out.append(i)
                else:
                    out.append(i)
            elif lhs:
                if val not in self.bound or all(
                    self._shared_ok(path, p)
                    for p in self.mv_paths.get(val, ())
                ):
                    out.append(i)
            elif val in self.bound:
                out.append(i)
        if not out:
            # a fully-masked hole (every metavariable bound, reuse
            # failing the sharing gate, op budget spent): force-fill
            # with any metavariable — the resulting pattern is dead
            # but the play terminates cleanly
            out = [
                i
                for i, (kind, _) in enumerate(self.actions)
                if kind == "mv"
            ][:1]
        return out

    def _shared_ok(self, path: tuple, prev: tuple) -> bool:
        """Return whether an LHS metavariable reuse is corpus-attested.

        A metavariable bound at *prev* and re-placed at *path* imposes
        a structural-equality precondition on the two sites — only
        matchable where the corpus itself shares a subterm under the
        lowest common ancestor of the bindings (``Vocab.share``).
        """
        i = 0
        while i < min(len(path), len(prev)) and path[i] == prev[i]:
            i += 1
        lca = _node_at(self.lhs, path[:i])
        if lca.kind != "op":
            return False
        kids = tuple(
            c.op
            if c.kind == "op"
            else ("const" if c.kind == "const" else "·")
            for c in lca.children
        )
        return self.v.share.get((lca.op, kids), 0) > 0

    def _autofill(self, slot: _Slot, lhs: bool) -> None:
        """Fill every remaining hole with a metavariable (forced end)."""
        while (p := _first_hole(slot)) is not None:
            name = self.bound.copy().pop() if self.bound else "U"
            if lhs:
                name = "U"
            _fill(slot, p, _Slot(kind="mv", name=name))

    def step(self, idx: int) -> None:
        """Place action *idx* at the leftmost hole."""
        kind, val = self.actions[idx]
        cur = self._cur()
        path = _first_hole(cur)
        if path is None:
            raise RuntimeError("step() on a term with no hole")
        if kind == "op":
            new = _Slot(
                kind="op",
                op=val,
                children=[_Slot() for _ in range(self.v.arity[val])],
                attrs={
                    k: f"{val}.{k}"
                    for k in self.v.attr_keys.get(val, ())
                },
            )
        elif kind == "mv":
            new = _Slot(kind="mv", name=val)
            if self.side == "lhs":
                self.bound.add(val)
                self.mv_paths.setdefault(val, []).append(path)
        else:
            new = _Slot(kind="const", value=val)
        _fill(cur, path, new)
        self.steps += 1
        if _first_hole(self.lhs) is None and self.side == "lhs":
            self.side = "rhs"
        if (
            _first_hole(self.rhs) is None
            and _first_hole(self.lhs) is None
        ):
            self.done = True
        if self.steps >= _MAX_STEPS and not self.done:
            self._autofill(self.lhs, True)
            self._autofill(self.rhs, False)
            self.done = True

    def candidate(self) -> tuple[Any, Any, Any, Any] | None:
        """Mint ``(lhs, rhs, check, derive)`` for the terminal play."""
        lhs, rhs = _mint(self.lhs), _mint(self.rhs)
        if lhs is None or rhs is None:
            return None
        unbound = frozenset(_attr_mv_names(rhs) - _attr_mv_names(lhs))
        derive = _attr_bridge(unbound) if unbound else None
        return lhs, rhs, _bool_attr_check, derive

    # -- featurization -------------------------------------------------

    def _root_tuple(self) -> tuple | None:
        """Return the partial LHS's root ``(op, child-kind)`` key."""
        if self.lhs.kind != "op":
            return None
        kids = tuple(
            c.op
            if c.kind == "op"
            else ("const" if c.kind == "const" else "·")
            for c in self.lhs.children
        )
        return (self.lhs.op, kids)

    def state_vec(self) -> list[float]:
        """Return the state features the policy reads."""
        root = self._root_tuple()
        root_n = (
            self.v.tuple_count.get(root, 0) if root is not None else 0
        )
        return [
            1.0 if self.side == "rhs" else 0.0,
            _n_ops(self.lhs) / _MAX_OPS_SIDE,
            _n_ops(self.rhs) / _MAX_OPS_SIDE,
            _n_holes(self.lhs) / 10,
            _n_holes(self.rhs) / 10,
            len(self.bound) / len(_MV_NAMES),
            self.steps / _MAX_STEPS,
            1.0 if root_n else 0.0,
            root_n / self.v.max_count,
        ]

    def _prior_count(self, idx: int, hole_path: tuple) -> float:
        """Census count of placing action *idx* at *hole_path*.

        Op actions read the ``(parent, position, child-op)`` counts;
        metavariables read the ``·`` (leaf) count at that position and
        consts the ``const`` count — the census already says where
        real children are leaves, which is what keeps a constructed
        LHS *applicable*.  At the LHS root only the op-head frequency
        is known; at the RHS root there is no census context.
        """
        kind, val = self.actions[idx]
        if len(hole_path) == 0:
            if self.side == "lhs" and kind == "op":
                return float(self.v.head_count.get(val, 0))
            return 0.0
        parent = _node_at(self._cur(), hole_path[:-1])
        key = (parent.op, hole_path[-1])
        if kind == "mv":
            # a metavar binds whatever the site holds — its prior is
            # the total census mass at that position
            return float(self.v.pos_count.get(key, 0))
        if kind == "const":
            return float(self.v.child_count.get((*key, "const"), 0))
        return float(self.v.child_count.get((*key, val), 0))

    def action_vec(self, idx: int, hole_path: tuple) -> list[float]:
        """Return the features of one legal action at *hole_path*."""
        kind, val = self.actions[idx]
        n = float(self._prior_count(idx, hole_path))
        prior = math.log1p(n) / math.log1p(self.v.max_count)
        other = self.rhs if self.side == "lhs" else self.lhs
        in_other = (
            1.0 if kind == "op" and val in _ops_present(other) else 0.0
        )
        return [
            1.0 if kind == "op" else 0.0,
            1.0 if kind == "mv" else 0.0,
            1.0 if kind == "const" else 0.0,
            self.v.arity.get(val, 0) / 2 if kind == "op" else 0.0,
            1.0 if self.v.classes.get(val) == "view" else 0.0,
            1.0 if self.v.classes.get(val) == "pointwise-bin" else 0.0,
            1.0 if self.v.classes.get(val) == "pointwise-un" else 0.0,
            1.0 if self.v.classes.get(val) == "reduction" else 0.0,
            prior,
            in_other,
            1.0 if kind == "mv" and val in self.bound else 0.0,
            float(val) / 2 if kind == "const" else 0.0,
        ]

    def prior_logit(self, idx: int, hole_path: tuple) -> float:
        """Return the fixed prior logit for action *idx*."""
        kind, val = self.actions[idx]
        n = self._prior_count(idx, hole_path)
        prior = (
            _PRIOR_CHILD * math.log1p(n) / math.log1p(self.v.max_count)
        )
        if kind == "op":
            prior *= _PRIOR_DEPTH_DECAY ** len(hole_path)
            if self.side == "rhs" and val in _ops_present(self.lhs):
                prior += _PRIOR_REUSE
            if (
                self.side == "rhs"
                and len(hole_path) == 0
                and self.lhs.kind == "op"
            ):
                if val == self.lhs.op:
                    prior += _PRIOR_COMMUTE
                if any(
                    c.kind == "op" and c.op == val
                    for c in self.lhs.children
                ):
                    prior += _PRIOR_SWAP_NEST
        if kind == "mv" and self.side == "rhs" and val in self.bound:
            prior += _PRIOR_BOUND_MV
        if (
            kind == "mv"
            and self.side == "lhs"
            and val not in self.bound
        ):
            prior += _PRIOR_FRESH_MV
        return prior


def _seed_name(slot: _Slot | None) -> str:
    """Return a readable name for a census skeleton slot."""
    if slot is None or slot.kind != "op":
        return ""
    kids = ",".join(
        c.op if c.kind == "op" else "·" for c in slot.children
    )
    return f"seed:{slot.op}({kids})"


# ---------------------------------------------------------------------------
#  The referee — truth gate, then fires + pay (counts oracle calls)
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    """The measured verdict on one distinct candidate."""

    name: str = ""
    lhs_repr: str = ""
    rhs_repr: str = ""
    reason: str = ""  # tautology|repeat|no-instance|false|unknown|true
    truth: bool = False
    relation: str = ""
    instance: bool = False
    fires: int = 0
    paid: int = 0
    changed: int = 0
    verify_fail: int = 0
    rel_drop: float = 0.0
    directions: tuple[str, ...] = ()
    fire_errors: int = 0
    score: float = 0.0


class Referee:
    """Evaluates terminal candidates; counts ``_numeric_true`` calls.

    ``terms`` is the corpus slice searched for a real instantiation;
    ``cases`` the slice the rule is fired over; ``lib`` the
    alpha-normal keys of the search rule set (duplicate detection).
    ``by_key`` dedups candidates: a repeat costs no oracle call.
    """

    def __init__(
        self,
        terms: list[Any],
        cases: list[TermCase],
        sink: Any,
        cost_fn: Any,
        lib: list,
    ) -> None:
        """Bind the corpus slice, the measurement machinery, the lib."""
        self.terms = terms
        self.cases = cases
        self.sink = sink
        self.cost_fn = cost_fn
        self.lib = lib
        self.oracle_calls = 0
        self.probes = 0
        self.repeats = 0
        self.by_key: dict[Any, Verdict] = {}
        #: Every seen equality, in *both* orientations — an equality
        #: is symmetric, so a swapped construction is the same
        #: candidate and must not cost a second oracle call.
        self.dedup: set[Any] = set()
        self._n = 0

    def _instance(self, lhs, rhs, check, derive) -> tuple | None:
        """Instantiate on the first viable corpus-slice match."""
        for t in self.terms:
            for sub in _iter_subterms(t):
                if not isinstance(sub, Op):
                    continue
                subst = _term_match(lhs, sub)
                if subst is None:
                    continue
                if check is not None:
                    try:
                        if not check(subst):
                            continue
                    except Exception:
                        continue
                inst = dict(subst)
                if derive is not None:
                    try:
                        extra = derive(subst)
                    except Exception:
                        continue
                    if extra is None:
                        continue
                    inst.update(extra)
                try:
                    return sub, _term_instantiate(rhs, inst)
                except Exception:
                    continue
        return None

    def evaluate(
        self,
        lhs: Any,
        rhs: Any,
        check: Any = None,
        derive: Any = None,
        *,
        name: str = "",
        instance: tuple | None = None,
    ) -> Verdict:
        """Score one candidate; spend at most one oracle call."""
        v = Verdict(
            name=name or f"cand_{self._n}",
            lhs_repr=op_repr(lhs),
            rhs_repr=op_repr(rhs),
        )
        self._n += 1
        key = lp._key(lhs, rhs)
        if key[0] == key[1]:
            v.reason = "tautology"
            return v
        if key in self.dedup:
            self.repeats += 1
            old = self.by_key.get(key) or self.by_key.get(
                (key[1], key[0])
            )
            return Verdict(
                name=v.name,
                lhs_repr=v.lhs_repr,
                rhs_repr=v.rhs_repr,
                reason=f"repeat({old.reason if old else '?'})",
                score=0.0,
            )
        self.dedup.add(key)
        self.dedup.add((key[1], key[0]))
        self.by_key[key] = v
        inst = instance or self._instance(lhs, rhs, check, derive)
        if inst is None:
            v.reason = "no-instance"
            return v
        v.instance = True
        self.oracle_calls += 1
        truth = lp._numeric_true(inst[0], inst[1])
        if truth is not True:
            v.reason = "false" if truth is False else "unknown"
            return v
        v.truth = True
        v.relation = lp._relation(lhs, rhs, self.lib)
        self._fire(v, lhs, rhs, check, derive)
        v.score = (
            1.0
            + 0.2 * min(v.fires, 40)
            + 2.0 * v.paid
            + 10.0 * v.rel_drop
        )
        return v

    def _fire(self, v: Verdict, lhs, rhs, check, derive) -> None:
        """Fire the candidate over the slice in *both* orientations.

        Equality is symmetric — the numeric oracle already proved
        ``lhs = rhs`` — so one oracle call referees both directions;
        only the *firing* differs (a rule rewrites ``lhs -> rhs``).
        The generic check/derive hooks transfer safely: the bool-attr
        check reads any bound dict, and the attr bridge's extra
        bindings are unused keys under the swapped pattern.  The
        verdict keeps the better direction; ``verify_fail`` counts a
        lowering mismatch in either.
        """
        directions: list[str] = []
        for lo, ro, tag in ((lhs, rhs, "fwd"), (rhs, lhs, "bwd")):
            rule = Rewrite(
                f"{v.name}:{tag}", lo, ro, check=check, derive=derive
            )
            fires = 0
            try:
                for case in self.cases:
                    f = _probe(case, rule, self.sink, self.cost_fn)
                    self.probes += 1
                    if f.verified in ("FAIL", "error"):
                        v.verify_fail += 1
                    if not f.fires:
                        continue
                    fires += f.fires
                    if f.changed:
                        v.changed += 1
                    if f.paid:
                        v.paid += 1
                        if f.base_cost:
                            drop = (
                                f.base_cost - f.out_cost
                            ) / f.base_cost
                            v.rel_drop = max(v.rel_drop, drop)
            except Exception:
                # an unfireable orientation (e.g. RHS-only metavars
                # on the swapped pattern) — honest miss, not a crash
                v.fire_errors += 1
                continue
            v.fires = max(v.fires, fires)
            if fires:
                directions.append(tag)
        v.directions = tuple(directions)
        v.reason = "true"

    def summary(self, plays: int = 0) -> dict:
        """Aggregate the yield table over distinct candidates."""
        vs = list(self.by_key.values())
        true = [v for v in vs if v.truth]
        firing = [v for v in true if v.fires > 0]
        new_tf = [v for v in firing if v.relation == "new"]
        ship = [v for v in new_tf if v.paid > 0 and v.verify_fail == 0]
        return {
            "plays": plays,
            "candidates": len(vs),
            "repeats": self.repeats,
            "oracle_calls": self.oracle_calls,
            "probes": self.probes,
            "no_instance": sum(
                1 for v in vs if v.reason == "no-instance"
            ),
            "false": sum(1 for v in vs if v.reason == "false"),
            "unknown": sum(1 for v in vs if v.reason == "unknown"),
            "tautology": self._n - len(vs) - self.repeats,
            "true": len(true),
            "true_firing": len(firing),
            "new_true_firing": len(new_tf),
            "shippable": len(ship),
            "yield_true_per_call": (
                len(true) / self.oracle_calls
                if self.oracle_calls
                else 0.0
            ),
            "yield_tf_per_call": (
                len(firing) / self.oracle_calls
                if self.oracle_calls
                else 0.0
            ),
            "yield_ship_per_call": (
                len(ship) / self.oracle_calls
                if self.oracle_calls
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
#  The player — a (state ⊕ action) -> logit MLP + fixed corpus prior
# ---------------------------------------------------------------------------


class _PolicyNet(nn.Module):
    """Score each legal action for a state (open action set)."""

    def __init__(self, sdim: int, adim: int, hidden: int = 48) -> None:
        """Build the MLP over ``(state ⊕ action)`` inputs."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sdim + adim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, sv: torch.Tensor, av: torch.Tensor
    ) -> torch.Tensor:
        """Return one logit per action, shape ``[n_actions]``."""
        n = av.shape[0]
        x = torch.cat([sv.unsqueeze(0).expand(n, -1), av], dim=1)
        return self.net(x).squeeze(-1)


def _tensors(
    game: BuildGame, legal: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Featurize the state, the legal actions, and the prior logits."""
    hole = _first_hole(game._cur())
    if hole is None:
        raise RuntimeError("featurize a term with no hole")
    sv = torch.tensor(game.state_vec(), dtype=torch.float32)
    av = torch.tensor(
        [game.action_vec(i, hole) for i in legal],
        dtype=torch.float32,
    )
    pr = torch.tensor(
        [game.prior_logit(i, hole) for i in legal],
        dtype=torch.float32,
    )
    return sv, av, pr


def _play_once(
    game: BuildGame,
    referee: Referee,
    model: nn.Module | None,
    rng: random.Random,
    *,
    prior: bool,
    name: str = "",
) -> tuple[Verdict, list[torch.Tensor]]:
    """Play one construction; return the verdict and the log-probs."""
    logps: list[torch.Tensor] = []
    while not game.done:
        legal = game.legal()
        sv, av, pr = _tensors(game, legal)
        logits = (
            model(sv, av)
            if model is not None
            else torch.zeros(len(legal))
        )
        if prior:
            logits = logits + pr
        dist = Categorical(logits=logits)
        a = dist.sample()
        if model is not None:
            logps.append(dist.log_prob(a))
        game.step(legal[int(a.item())])
    cand = game.candidate()
    if cand is None:
        return Verdict(name=name, reason="mint-error"), logps
    lhs, rhs, check, derive = cand
    v = referee.evaluate(
        lhs, rhs, check, derive, name=name or game.seed_name
    )
    return v, logps


def train(
    game: BuildGame,
    referee: Referee,
    episodes: int,
    *,
    hidden: int = 48,
    lr: float = 3e-3,
    seed_frac: float = 0.55,
    log_every: int = 50,
) -> tuple[nn.Module, list[float]]:
    """Train the policy by REINFORCE; return ``(model, reward hist)``.

    Sparse terminal reward with a running-mean baseline — the
    ``catopt_torch.rl`` advantage, simplified for one terminal payoff
    per play.
    """
    model = _PolicyNet(
        len(game.state_vec()), len(game.action_vec(0, ())), hidden
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    baseline = 0.0
    hist: list[float] = []
    seeds = game.v.seeds
    weights = [math.log1p(n) for _, n in seeds]
    for ep in range(episodes):
        seed = (
            game.rng.choices(seeds, weights=weights, k=1)[0][0]
            if seeds and game.rng.random() < seed_frac
            else None
        )
        # fresh skeleton objects each play (slots are mutated)
        game.reset(_clone(seed), _seed_name(seed))
        v, logps = _play_once(
            game, referee, model, game.rng, prior=True, name=""
        )
        r = v.score
        adv = r - baseline
        baseline = 0.95 * baseline + 0.05 * r
        if logps:
            loss = -(torch.stack(logps).sum() * adv)
            opt.zero_grad()
            loss.backward()
            opt.step()
        hist.append(r)
        if log_every and (ep + 1) % log_every == 0:
            lo = max(0, ep + 1 - log_every)
            print(
                f"  ep {ep + 1:>4}: mean reward "
                f"{sum(hist[lo:]) / (ep + 1 - lo):.3f} "
                f"(baseline {baseline:.3f}, "
                f"oracle {referee.oracle_calls})"
            )
    return model.eval(), hist


def _clone(slot: _Slot | None) -> _Slot | None:
    """Deep-copy a slot tree (seeds are reused across plays)."""
    if slot is None:
        return None
    children = [
        cloned
        for c in slot.children
        if (cloned := _clone(c)) is not None
    ]
    return _Slot(
        kind=slot.kind,
        op=slot.op,
        children=children,
        attrs=dict(slot.attrs),
        name=slot.name,
        value=slot.value,
    )


def run_player(
    game: BuildGame,
    referee: Referee,
    model: nn.Module | None,
    budget: int,
    max_plays: int,
    *,
    prior: bool = True,
    seed_frac: float = 0.55,
) -> dict:
    """Play until *budget* oracle calls are spent; return the summary."""
    plays = 0
    seeds = game.v.seeds
    weights = [math.log1p(n) for _, n in seeds]
    while referee.oracle_calls < budget and plays < max_plays:
        seed = (
            game.rng.choices(seeds, weights=weights, k=1)[0][0]
            if seeds and game.rng.random() < seed_frac
            else None
        )
        game.reset(_clone(seed), _seed_name(seed))
        _play_once(game, referee, model, game.rng, prior=prior)
        plays += 1
    return referee.summary(plays)


def eval_baseline(
    proposals: list, referee: Referee, budget: int
) -> dict:
    """Run the enumerative proposals through the same referee."""
    for p in proposals:
        if referee.oracle_calls >= budget:
            break
        referee.evaluate(
            p.lhs,
            p.rhs,
            p.check,
            p.derive,
            name=p.name,
            instance=p.instance,
        )
    return referee.summary(len(proposals))


# ---------------------------------------------------------------------------
#  The guide seam — which generator invests compute where
# ---------------------------------------------------------------------------
#
#  ``eval_baseline`` is one fixed schedule over the generator
#  inventory; the guide makes the schedule the action space.  An
#  action is an ``Allocation`` — a generator name plus a draw count —
#  and the currency is proposals drawn, each costing the shared
#  ``Referee`` at most one oracle call.  A ``Guide`` reads a
#  ``GuideObs`` and answers "which generator gets the next draws".
#  It steers compute only: every draw still faces the same referee,
#  so the guide can never mint semantics (ADR 0002 Rule 3).


#: The generator inventory in its fixed enumeration order — the
#: five ``pipeline.propose`` sources plus ``"build"`` (the
#: construction player; one draw is one play).  Workload and gap
#: generation are deliberately absent: those generators mutate the
#: corpus mid-game, and a mutated corpus invalidates the evidence
#: store's scope keys — a corpus-mutating action needs a re-scoped
#: observation, not another queue.
ENUMERATION_ORDER = (
    "census-naturality",
    "census-mixed-view",
    "pattern-recognition",
    "shape-aware",
    "algebraic-grammar",
    "build",
)


@dataclass(frozen=True)
class Allocation:
    """One guide move: draw up to ``n`` proposals from a generator."""

    generator: str
    n: int


@dataclass
class ArmStat:
    """The running tally for one generator arm.

    ``oracle_calls`` is the honest cost attribution — a proposal that
    never reaches the referee (no instance, repeat, tautology) spent
    a draw but no call.
    """

    emitted: int = 0
    drawn: int = 0
    oracle_calls: int = 0
    true: int = 0
    firing: int = 0
    new_tf: int = 0
    shippable: int = 0
    best: float = 0.0


@dataclass(frozen=True)
class GuideObs:
    """What a guide sees before each allocation.

    ``verdicts`` is the evidence store's ``latest_verdicts`` for the
    arena's scope — the observation source is the store itself, so a
    guide reads exactly what a later audit would.  ``arms`` carries
    the per-generator tallies (copies — an observation cannot mutate
    the board).
    """

    round: int
    spent: int
    budget: int
    emitted: dict[str, int]
    remaining: dict[str, int]
    arms: dict[str, ArmStat]
    verdicts: dict[str, dict]


def generator_pools(
    census_op: dict, terms: list[Any], vocab: str = "derived"
) -> dict[str, list[lpl.Proposal]]:
    """Return the investable inventory: ``{arm: proposal queue}``.

    The five ``pipeline.propose`` sources, one arm each, deduplicated
    under ``propose``'s own merged-pool semantics — a candidate
    reachable from two generators queues on the first — so the
    enumeration arm replays ``propose``'s order exactly.
    """
    pointwise, views = lpl._vocab_sets(vocab)
    pools: dict[str, list[lpl.Proposal]] = {
        "census-naturality": lpl._census_naturality(
            census_op, terms, pointwise, views
        ),
        "census-mixed-view": lpl._census_mixed_naturality(
            census_op, terms, pointwise, views
        ),
        "pattern-recognition": lpl._pattern_recognition(census_op),
        "shape-aware": lpl._shape_aware(),
        "algebraic-grammar": lpl._grammar(),
    }
    seen: set = set()
    for queue in pools.values():
        kept: list[lpl.Proposal] = []
        for p in queue:
            key = lp._key(p.lhs, p.rhs)
            if key in seen:
                continue
            seen.add(key)
            kept.append(p)
        queue[:] = kept
    return pools


def _verdict_evidence(proposal: lpl.Proposal, v: Verdict) -> Any:
    """Shape a referee ``Verdict`` as the pipeline's ``Evidence``.

    ``evidence.verdict_row`` duck-types this record, so the guide's
    store rows are the pipeline's own schema.  Fields the referee
    does not measure — derivability, census sites, the reach sweep —
    stay at their honest defaults; a stored ``SHIP`` therefore means
    "cleared the referee's bar" and no more.
    """
    ev = lpl.Evidence(proposal=proposal)
    ev.num_true = (
        True if v.truth else (False if v.reason == "false" else None)
    )
    ev.matches = int(v.instance)
    ev.relation = v.relation or "new"
    ev.fires = v.fires
    ev.changed = v.changed
    ev.paid = v.paid
    ev.verify_fail = v.verify_fail
    ev.cost_drop = v.rel_drop
    return ev


class GuideArena:
    """The board the guide plays on: generator queues + referee.

    ``invest`` draws up to ``n`` proposals from an arm and referees
    each through the shared :class:`Referee`; adjudicated verdicts
    (never dedup artifacts) are written to the evidence store via
    ``record_run``, and :meth:`observation` reads them back through
    ``latest_verdicts``.  ``game``/``model`` arm the ``"build"``
    generator — one draw is one ``_play_once`` — and ``plays_cap``
    bounds its queue.
    """

    def __init__(
        self,
        pools: dict[str, list[lpl.Proposal]],
        referee: Referee,
        *,
        game: BuildGame | None = None,
        model: nn.Module | None = None,
        prior: bool = True,
        seed_frac: float = 0.55,
        plays_cap: int = 60,
        conn: Any = None,
        meta: dict[str, str] | None = None,
    ) -> None:
        """Bind the inventory, the referee and the evidence scope."""
        self.pools = {k: list(v) for k, v in pools.items()}
        self.ref = referee
        self.game = game
        self.model = model
        self.prior = prior
        self.seed_frac = seed_frac
        self.plays_left = plays_cap if game is not None else 0
        self.pos = {k: 0 for k in pools}
        self.spent = 0
        self.rounds = 0
        self.arms = {
            k: ArmStat(emitted=len(v)) for k, v in pools.items()
        }
        if game is not None:
            self.arms["build"] = ArmStat(emitted=plays_cap)
        self.conn = (
            conn if conn is not None else ev_store.connect(":memory:")
        )
        if meta is None:
            meta = {
                "corpus_hash": ev_store.corpus_hash(
                    op_repr(t) for t in referee.terms
                ),
                "rules_hash": ev_store.rules_hash(
                    repr(k) for k in referee.lib
                ),
                "code_rev": ev_store.code_rev(),
                "run_id": ev_store.new_run_id(),
                "holdout": "",
                "ts": ev_store.now(),
            }
        self.meta = meta
        self.first_true_at: int | None = None
        self.first_ship_at: int | None = None
        self.best: Verdict | None = None
        self.best_arm = ""

    def _remaining(self, name: str) -> int:
        """Return the arm's undrawn queue length."""
        if name == "build":
            return self.plays_left
        return len(self.pools[name]) - self.pos[name]

    def observation(self, budget: int) -> GuideObs:
        """Return the current observation for a guide."""
        verdicts = ev_store.latest_verdicts(
            self.conn,
            self.meta["corpus_hash"],
            self.meta["rules_hash"],
            self.meta["code_rev"],
        )
        return GuideObs(
            round=self.rounds,
            spent=self.spent,
            budget=budget,
            emitted={k: s.emitted for k, s in self.arms.items()},
            remaining={k: self._remaining(k) for k in self.arms},
            arms={k: replace(s) for k, s in self.arms.items()},
            verdicts=verdicts,
        )

    def invest(self, alloc: Allocation) -> list[Verdict]:
        """Draw and referee up to ``alloc.n`` proposals from the arm."""
        if alloc.generator not in self.arms:
            raise KeyError(f"unknown generator {alloc.generator!r}")
        out: list[Verdict] = []
        while (
            len(out) < alloc.n and self._remaining(alloc.generator) > 0
        ):
            self.spent += 1
            if alloc.generator == "build":
                out.append(self._draw_build())
            else:
                p = self.pools[alloc.generator][
                    self.pos[alloc.generator]
                ]
                self.pos[alloc.generator] += 1
                out.append(self._draw_proposal(alloc.generator, p))
        self.rounds += 1
        return out

    def _draw_proposal(self, arm: str, p: lpl.Proposal) -> Verdict:
        """Referee one queued proposal and account for it."""
        before = self.ref.oracle_calls
        v = self.ref.evaluate(
            p.lhs,
            p.rhs,
            p.check,
            p.derive,
            name=p.name,
            instance=p.instance,
        )
        self._account(
            arm,
            v,
            p.lhs,
            p.rhs,
            proposal=p,
            calls=self.ref.oracle_calls - before,
        )
        return v

    def _draw_build(self) -> Verdict:
        """Play one construction and account for it."""
        g = self.game
        if g is None:
            # unreachable — the "build" arm exists only when a game
            # was bound — but never silently.
            raise RuntimeError("build arm drawn with no game")
        seeds = g.v.seeds
        seed = (
            g.rng.choices(
                seeds, weights=[math.log1p(n) for _, n in seeds], k=1
            )[0][0]
            if seeds and g.rng.random() < self.seed_frac
            else None
        )
        g.reset(_clone(seed), _seed_name(seed))
        before = self.ref.oracle_calls
        v, _ = _play_once(
            g, self.ref, self.model, g.rng, prior=self.prior
        )
        self.plays_left -= 1
        cand = g.candidate()
        lhs, rhs = cand[:2] if cand is not None else (None, None)
        self._account(
            "build",
            v,
            lhs,
            rhs,
            calls=self.ref.oracle_calls - before,
        )
        return v

    def _account(
        self,
        arm: str,
        v: Verdict,
        lhs: Any,
        rhs: Any,
        *,
        proposal: lpl.Proposal | None = None,
        calls: int = 0,
    ) -> None:
        """Tally one draw and record its verdict to the store."""
        st = self.arms[arm]
        st.drawn += 1
        st.oracle_calls += calls
        self._tally(arm, st, v)
        if lhs is None or rhs is None:
            return  # mint-error — nothing was adjudicated
        key = lp._key(lhs, rhs)
        if self.ref.by_key.get(key) is not v:
            return  # tautology/repeat — a dedup artifact, not evidence
        if proposal is None:
            proposal = lpl.Proposal(
                name=v.name,
                lhs=lhs,
                rhs=rhs,
                family=arm,
                sources=(arm,),
            )
        row = ev_store.verdict_row(
            repr(key),
            _verdict_evidence(proposal, v),
            v.lhs_repr,
            v.rhs_repr,
        )
        ev_store.record_run(self.conn, self.meta, [row])

    @staticmethod
    def _ship(v: Verdict) -> bool:
        """Return whether the verdict ships (the pipeline's test)."""
        return (
            v.truth
            and v.relation == "new"
            and v.fires > 0
            and v.paid > 0
            and v.verify_fail == 0
        )

    def _tally(self, arm: str, st: Any, v: Verdict) -> None:
        """Fold one adjudicated verdict into the arm's tallies."""
        if v.truth:
            st.true += 1
            if self.first_true_at is None:
                self.first_true_at = self.spent
        if v.fires:
            st.firing += 1
        if v.truth and v.fires and v.relation == "new":
            st.new_tf += 1
        if self._ship(v):
            st.shippable += 1
            if self.first_ship_at is None:
                self.first_ship_at = self.spent
        self._tally_best(arm, st, v)

    def _tally_best(self, arm: str, st: Any, v: Verdict) -> None:
        """Fold *v* into the per-arm and global best-score tracking."""
        if v.score > st.best:
            st.best = v.score
        if self.best is None or v.score > self.best.score:
            self.best = v
            self.best_arm = arm

    def summary(self, budget: int = 0) -> dict:
        """Aggregate the run: yields, per-arm tallies, the best find."""
        return {
            "draws": self.spent,
            "rounds": self.rounds,
            "budget": budget,
            "oracle_calls": self.ref.oracle_calls,
            "probes": self.ref.probes,
            "arms": {
                k: {
                    "emitted": s.emitted,
                    "drawn": s.drawn,
                    "oracle_calls": s.oracle_calls,
                    "true": s.true,
                    "firing": s.firing,
                    "new_tf": s.new_tf,
                    "shippable": s.shippable,
                    "best": s.best,
                }
                for k, s in self.arms.items()
            },
            "referee": self.ref.summary(),
            "best": (
                {
                    "name": self.best.name,
                    "score": self.best.score,
                    "generator": self.best_arm,
                    "lhs": self.best.lhs_repr,
                    "rhs": self.best.rhs_repr,
                }
                if self.best is not None
                else None
            ),
            "first_true_at": self.first_true_at,
            "first_ship_at": self.first_ship_at,
        }


# ---------------------------------------------------------------------------
#  Guides — policies over the arm inventory
# ---------------------------------------------------------------------------


class Guide:
    """A policy over generator arms — the guide, never the referee.

    ``choose`` reads the observation and returns the next
    :class:`Allocation` (``None`` ends the game); ``update`` is the
    learning hook, called with the allocation's payoff after each
    invest.  The guide never sees a candidate before the referee
    does — it steers compute, not semantics.
    """

    def choose(self, obs: GuideObs) -> Allocation | None:
        """Return the next allocation; ``None`` ends the game."""
        raise NotImplementedError

    def update(self, reward: float, verdicts: list[Verdict]) -> None:
        """Observe the payoff of the last allocation."""


class EnumerationGuide(Guide):
    """The control arm — the pipeline's fixed generator order."""

    def __init__(
        self,
        order: Iterable[str] = ENUMERATION_ORDER,
        step: int = 8,
    ) -> None:
        """Bind the drain order and the per-allocation draw count."""
        self.order = tuple(order)
        self.step = step

    def choose(self, obs: GuideObs) -> Allocation | None:
        """Return the next non-drained arm in fixed order."""
        for name in self.order:
            if obs.remaining.get(name, 0) > 0:
                return Allocation(name, self.step)
        return None


class RandomGuide(Guide):
    """Uniform arm selection — the no-knowledge control."""

    def __init__(self, rng: random.Random, step: int = 8) -> None:
        """Bind the RNG and the per-allocation draw count."""
        self.rng = rng
        self.step = step

    def choose(self, obs: GuideObs) -> Allocation | None:
        """Return a uniform pick among arms with draws left."""
        live = [k for k, r in obs.remaining.items() if r > 0]
        if not live:
            return None
        return Allocation(self.rng.choice(live), self.step)


class LearnedGuide(Guide):
    """A learned arm policy — ``_PolicyNet`` re-pointed at arms.

    The same ``(state ⊕ action) -> logit`` net the construction
    player uses: each live arm is featurized from the observation, a
    ``Categorical`` samples the investment, and ``update`` applies
    the REINFORCE step with the same running-mean baseline ``train``
    uses.  The reward is the batch's mean verdict score — the guide
    inherits the referee's own currency.
    """

    _SDIM = 6
    _ADIM = 7

    def __init__(
        self,
        step: int = 8,
        hidden: int = 16,
        lr: float = 3e-3,
    ) -> None:
        """Build the arm-scoring net and the optimizer."""
        self.step = step
        self.net = _PolicyNet(self._SDIM, self._ADIM, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.baseline = 0.0
        self._logp: torch.Tensor | None = None

    def _state_vec(self, obs: GuideObs) -> list[float]:
        """Featurize the board: budget spent, queues left, progress."""
        total = max(sum(obs.emitted.values()), 1)
        drained = sum(1 for r in obs.remaining.values() if r == 0)
        best = max((s.best for s in obs.arms.values()), default=0.0)
        return [
            1.0,
            obs.spent / max(obs.budget, 1),
            sum(obs.remaining.values()) / total,
            drained / max(len(obs.arms), 1),
            best / 10.0,
            sum(s.true for s in obs.arms.values()) / 10.0,
        ]

    def _arm_vec(self, obs: GuideObs, name: str) -> list[float]:
        """Featurize one arm: exhaustion, hit rates, best score."""
        s = obs.arms[name]
        emitted = max(obs.emitted.get(name, 0), 1)
        drawn = max(s.drawn, 1)
        return [
            1.0,
            s.drawn / emitted,
            obs.remaining.get(name, 0) / emitted,
            s.true / drawn,
            s.firing / drawn,
            s.best / 10.0,
            1.0 if name == "build" else 0.0,
        ]

    def choose(self, obs: GuideObs) -> Allocation | None:
        """Sample an arm from the policy over live arms."""
        live = [k for k, r in obs.remaining.items() if r > 0]
        if not live:
            return None
        sv = torch.tensor(self._state_vec(obs), dtype=torch.float32)
        av = torch.tensor(
            [self._arm_vec(obs, k) for k in live],
            dtype=torch.float32,
        )
        dist = Categorical(logits=self.net(sv, av))
        a = dist.sample()
        self._logp = dist.log_prob(a)
        return Allocation(live[int(a.item())], self.step)

    def update(self, reward: float, verdicts: list[Verdict]) -> None:
        """REINFORCE step on the last allocation's log-prob."""
        if self._logp is None:
            return
        adv = reward - self.baseline
        self.baseline = 0.95 * self.baseline + 0.05 * reward
        loss = -(self._logp * adv)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._logp = None


def run_guide(arena: GuideArena, guide: Guide, budget: int) -> dict:
    """Play the guide game until *budget* proposals are drawn.

    Each round the guide allocates ``(generator, n)``; the arena
    draws and referees them; the guide observes the mean verdict
    score as reward.  The game ends when the budget is spent, the
    guide stops (``None``), or it allocates to a drained arm.
    """
    while arena.spent < budget:
        alloc = guide.choose(arena.observation(budget))
        if alloc is None:
            break
        n = min(alloc.n, budget - arena.spent)
        verdicts = arena.invest(Allocation(alloc.generator, n))
        if not verdicts:
            break
        reward = sum(v.score for v in verdicts) / len(verdicts)
        guide.update(reward, verdicts)
    return arena.summary(budget)


def compare_guides(
    make_arena: Callable[[], GuideArena],
    guides: Mapping[str, Callable[[], Guide]],
    budget: int,
) -> dict:
    """Run each guide factory on a fresh arena; return the summaries.

    ``make_arena`` must rebuild the referee as well as the board —
    dedup state is per-run — while the proposal queues themselves
    are shared (they are read-only).  ``guides`` maps a label to a
    *factory*, so every arm plays a fresh policy.
    """
    return {
        label: run_guide(make_arena(), make(), budget)
        for label, make in guides.items()
    }


def run_guide_experiment(args: argparse.Namespace) -> dict:
    """Run guide-vs-enumeration over the real corpus slice.

    The bounded task: within ``args.budget`` total proposals drawn
    from the generator inventory, how much does each allocation
    policy find — and does any beat the fixed enumeration?  The
    ``build`` arm is the construction player (uniform, or trained
    for ``--guide-train`` episodes first — the existing ``train``
    loop, no new trainer).
    """
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    bench, _be = _bench_cases()
    models, _me = model_cases()
    cts = [
        CorpusTerm(c.source, c.name, c.term) for c in [*bench, *models]
    ]
    counts, _tc = op_tuple_census(cts)
    sh_counts, _st = shape_census(cts)
    all_terms = [c.term for c in [*bench, *models]]
    census_op = {k: v for k, v in counts.items()}

    cases = _slice_cases(bench, models)
    slice_terms = [c.term for c in cases]
    pools = generator_pools(census_op, all_terms, args.vocab)
    vocab = build_vocab(all_terms, counts, sh_counts, args.top_seeds)

    sink = _sink()
    cost_fn = _cost_fn(sink)
    lib = [lp._key(r.lhs, r.rhs) for r in _search_rules(args.holdout)]

    build_model = None
    if args.guide_train:
        tgame = BuildGame(vocab, rng)
        tref = Referee(slice_terms, cases, sink, cost_fn, lib)
        build_model, _hist = train(
            tgame,
            tref,
            args.guide_train,
            hidden=args.hidden,
            lr=args.lr,
            seed_frac=args.seed_frac,
            log_every=0,
        )

    def make_arena() -> GuideArena:
        ref = Referee(slice_terms, cases, sink, cost_fn, lib)
        game = (
            BuildGame(vocab, random.Random(args.seed))
            if args.guide_build
            else None
        )
        return GuideArena(
            pools,
            ref,
            game=game,
            model=build_model,
            seed_frac=args.seed_frac,
            plays_cap=args.plays_cap,
        )

    guides = {
        "enumeration": lambda: EnumerationGuide(step=args.guide_step),
        "uniform": lambda: RandomGuide(
            random.Random(args.seed), step=args.guide_step
        ),
        "learned": lambda: LearnedGuide(step=args.guide_step),
    }
    results = compare_guides(make_arena, guides, args.budget)
    enum_tf = results["enumeration"]["referee"]["yield_tf_per_call"]
    ratios = {
        label: (
            s["referee"]["yield_tf_per_call"] / enum_tf
            if enum_tf
            else None
        )
        for label, s in results.items()
    }
    return {
        "args": vars(args),
        "pools": {k: len(v) for k, v in pools.items()},
        "results": results,
        "yield_tf_ratio_vs_enumeration": ratios,
    }


def _print_guide_report(result: dict) -> None:
    """Print the guide comparison table."""
    print("== guide seam — which generator invests compute where ==")
    print(
        "   pool: "
        + ", ".join(f"{k}={n}" for k, n in result["pools"].items())
    )
    for label, s in result["results"].items():
        r = s["referee"]
        ratio = result["yield_tf_ratio_vs_enumeration"][label]
        line = (
            f"  {label:<12} draws={s['draws']:>4} "
            f"calls={s['oracle_calls']:>4} true={r['true']:>3} "
            f"tf={r['true_firing']:>3} new_tf={r['new_true_firing']:>3} "
            f"ship~={r['shippable']:>2} "
            f"yield_tf={r['yield_tf_per_call']:.3f}"
        )
        if ratio is not None:
            line += f" ({ratio:.2f}x enum)"
        print(line)
        arms = ", ".join(
            f"{k}:{a['drawn']}d/{a['true']}t"
            for k, a in s["arms"].items()
            if a["drawn"]
        )
        if arms:
            print(f"      arms: {arms}")
        best = s["best"]
        if best is not None:
            print(
                f"      best {best['score']:.2f} "
                f"[{best['generator']}] "
                f"{best['lhs']} -> {best['rhs']}"
            )
        if s["first_ship_at"] is not None:
            print(f"      first shippable at draw {s['first_ship_at']}")
        elif s["first_true_at"] is not None:
            print(f"      first true at draw {s['first_true_at']}")


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _slice_cases(
    bench: list[TermCase], models: list[TermCase]
) -> list[TermCase]:
    """Return the small corpus slice the referee measures on.

    A few models spanning the shapes the census says are frequent —
    the SSM/hybrid select family, a manual softmax, an attention and a
    parallel-linear block — plus a handful of bench cases covering
    select/softmax/elementwise shapes.
    """
    want = {
        "SelectiveSSM",
        "DiagDenseSSM",
        "DiagonalSSM",
        "HybridBlock",
        "TwoLayerHybrid",
        "ManualSoftmaxAttention",
        "AttentionBlock",
        "ParallelLinear",
    }
    bench_names = {
        "select_mul",
        "softmax_fold",
        "comm_add",
        "comm_mul",
        "id_mul",
        "assoc_mul",
        "factor_matmul",
        "sdpa_fold_addmul",
    }
    out = [c for c in models if c.name in want]
    out += [c for c in bench if c.name in bench_names]
    return out


def _search_rules(holdout: str | None) -> list[Any]:
    """Return the search rule set, minus the holdout names."""
    return lpl._search_rules(holdout)


def _targets() -> dict[str, tuple]:
    """Return ``{label: (fwd_key, swap_key)}`` for the known winners."""
    out: dict[str, tuple] = {}
    for r in ALL_RULES:
        if r.name == "select_mul":
            k = lp._key(r.lhs, r.rhs)
            out["select_mul"] = (k, (k[1], k[0]))
    sm = lpl._softmax_fold()
    k = lp._key(sm.lhs, sm.rhs)
    out["softmax_fold"] = (k, (k[1], k[0]))
    return out


def _rediscovered(referee: Referee, targets: dict) -> dict:
    """Check which known-winner keys appear among the verdicts."""
    hit: dict[str, list[Verdict]] = {k: [] for k in targets}
    for key, v in referee.by_key.items():
        for label, (fwd, swap) in targets.items():
            if key in (fwd, swap):
                hit[label].append(v)
    return {
        label: [
            {
                "name": v.name,
                "truth": v.truth,
                "fires": v.fires,
                "relation": v.relation,
                "lhs": v.lhs_repr,
                "rhs": v.rhs_repr,
            }
            for v in vs
        ]
        for label, vs in hit.items()
    }


def _reason_table(referee: Referee) -> dict[str, int]:
    """Count terminal outcomes by reason — where the search stalls."""
    out: dict[str, int] = {}
    for v in referee.by_key.values():
        out[v.reason] = out.get(v.reason, 0) + 1
    return out


def _print_yield(tag: str, s: dict) -> None:
    """Print one yield row."""
    print(
        f"  {tag:<22} plays={s['plays']:>4} cands={s['candidates']:>4} "
        f"calls={s['oracle_calls']:>4} true={s['true']:>3} "
        f"true+fire={s['true_firing']:>3} "
        f"new+t+f={s['new_true_firing']:>3} "
        f"ship~={s['shippable']:>2}  "
        f"yield/call tf={s['yield_tf_per_call']:.3f} "
        f"ship={s['yield_ship_per_call']:.3f}"
    )


def _print_top(referee: Referee, top: int) -> None:
    """Print the highest-scoring distinct candidates."""
    vs = sorted(referee.by_key.values(), key=lambda v: -v.score)[:top]
    if not vs:
        print("  (no scored candidates)")
        return
    for v in vs:
        if not v.truth:
            continue
        print(
            f"  {v.score:6.2f} {v.relation:<9} fires={v.fires:>2} "
            f"paid={v.paid} drop={v.rel_drop * 100:4.1f}%  "
            f"{v.lhs_repr} -> {v.rhs_repr}"
        )


def run_experiment(args: argparse.Namespace) -> dict:
    """Run baseline vs players end to end; return the result."""
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    bench, _be = _bench_cases()
    models, _me = model_cases()
    cts = [
        CorpusTerm(c.source, c.name, c.term) for c in [*bench, *models]
    ]
    counts, _tc = op_tuple_census(cts)
    sh_counts, _st = shape_census(cts)
    all_terms = [c.term for c in [*bench, *models]]
    census_op = {k: v for k, v in counts.items()}

    cases = _slice_cases(bench, models)
    slice_terms = [c.term for c in cases]
    print(
        f"corpus: {len(bench)} bench + {len(models)} models; "
        f"slice: {len(cases)} cases"
    )

    vocab = build_vocab(all_terms, counts, sh_counts, args.top_seeds)
    print(
        f"vocab: {len(vocab.ops)} ops, {len(vocab.seeds)} seed "
        f"skeletons (top {args.top_seeds} tuples)"
    )

    sink = _sink()
    cost_fn = _cost_fn(sink)
    base_rules = _search_rules(args.holdout)
    lib = [lp._key(r.lhs, r.rhs) for r in base_rules]

    # -- enumerative baseline ------------------------------------------
    proposals = lpl.propose(census_op, all_terms, args.vocab)
    base_ref = Referee(slice_terms, cases, sink, cost_fn, lib)
    base = eval_baseline(proposals, base_ref, args.budget)

    # -- uniform player -------------------------------------------------
    game = BuildGame(vocab, rng)
    rand_ref = Referee(slice_terms, cases, sink, cost_fn, lib)
    rand = run_player(
        game,
        rand_ref,
        None,
        args.random_calls,
        args.plays_cap,
        prior=False,
        seed_frac=args.seed_frac,
    )

    # -- trained player --------------------------------------------------
    game = BuildGame(vocab, rng)
    train_ref = Referee(slice_terms, cases, sink, cost_fn, lib)
    print(f"-- training ({args.episodes} episodes) --")
    model, hist = train(
        game,
        train_ref,
        args.episodes,
        hidden=args.hidden,
        lr=args.lr,
        seed_frac=args.seed_frac,
        log_every=args.log_every,
    )

    eval_ref = Referee(slice_terms, cases, sink, cost_fn, lib)
    player = run_player(
        game,
        eval_ref,
        model,
        args.budget,
        args.plays_cap,
        prior=True,
        seed_frac=args.seed_frac,
    )

    targets = _targets()
    return {
        "args": vars(args),
        "n_proposals": len(proposals),
        "baseline": base,
        "baseline_reasons": _reason_table(base_ref),
        "random": rand,
        "random_reasons": _reason_table(rand_ref),
        "train_reward_tail": sum(hist[-50:]) / max(len(hist[-50:]), 1),
        "train_oracle": train_ref.oracle_calls,
        "train_true": sum(
            1 for v in train_ref.by_key.values() if v.truth
        ),
        "player": player,
        "player_reasons": _reason_table(eval_ref),
        "player_top": [
            {
                "score": v.score,
                "relation": v.relation,
                "fires": v.fires,
                "paid": v.paid,
                "rel_drop": v.rel_drop,
                "lhs": v.lhs_repr,
                "rhs": v.rhs_repr,
            }
            for v in sorted(
                eval_ref.by_key.values(), key=lambda v: -v.score
            )[: args.top]
            if v.truth
        ],
        "player_rediscovered": _rediscovered(eval_ref, targets),
        "baseline_rediscovered": _rediscovered(base_ref, targets),
        "train_rediscovered": _rediscovered(train_ref, targets),
        "baseline_ref": base_ref,
        "random_ref": rand_ref,
        "player_ref": eval_ref,
        "train_ref": train_ref,
    }


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable result (referee objects dropped)."""
    payload = {
        k: v
        for k, v in result.items()
        if not k.endswith("_ref") and k != "args"
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the game vs the enumerative baseline; print the yields."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--guide",
        action="store_true",
        help="run the guide-seam comparison instead of the player game",
    )
    p.add_argument(
        "--guide-step",
        type=int,
        default=8,
        help="proposals per guide allocation",
    )
    p.add_argument(
        "--guide-build",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the construction player as a 'build' arm",
    )
    p.add_argument(
        "--guide-train",
        type=int,
        default=0,
        help="episodes to pre-train the build arm's player policy",
    )
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument(
        "--budget",
        type=int,
        default=200,
        help="oracle-call budget per evaluation run",
    )
    p.add_argument(
        "--random-calls",
        type=int,
        default=100,
        help="oracle budget for the uniform player",
    )
    p.add_argument(
        "--plays-cap",
        type=int,
        default=2500,
        help="hard cap on plays per run",
    )
    p.add_argument(
        "--seed-frac",
        type=float,
        default=0.55,
        help="fraction of plays starting from a census seed",
    )
    p.add_argument(
        "--top-seeds",
        type=int,
        default=30,
        help="number of census op-tuples used as seeds",
    )
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument(
        "--holdout", help="library rule(s) to hold out of the lib"
    )
    p.add_argument(
        "--vocab",
        choices=("hand", "derived"),
        default="hand",
        help="baseline generator's op alphabet",
    )
    p.add_argument("--json", help="write machine-readable results")
    args = p.parse_args(argv)

    if args.guide:
        r = run_guide_experiment(args)
        _print_guide_report(r)
        if args.json:
            _dump_json(args.json, r)
            print(f"\nwrote {args.json}")
        return 0

    r = run_experiment(args)

    print()
    print("== law meta-game — yield per oracle call ==")
    _print_yield("enumerative", r["baseline"])
    _print_yield("uniform player", r["random"])
    _print_yield("learned player", r["player"])
    print(
        f"  (training: {r['train_oracle']} oracle calls, "
        f"{r['train_true']} true candidates found, "
        f"tail reward {r['train_reward_tail']:.3f})"
    )
    print()
    print("-- where the search stalls (distinct candidates) --")
    for tag in ("baseline_reasons", "random_reasons", "player_reasons"):
        print(f"  {tag[:-8]:<18} {r[tag]}")
    print()
    print("-- uniform player's top candidates --")
    _print_top(r["random_ref"], args.top)
    print()
    print("-- learned player's top candidates --")
    _print_top(r["player_ref"], args.top)
    print()
    print("-- rediscovery of known winners --")
    for label, hits in r["player_rediscovered"].items():
        if hits:
            for h in hits:
                print(
                    f"  {label}: REDISCOVERED truth={h['truth']} "
                    f"fires={h['fires']} rel={h['relation']}"
                )
                print(f"      {h['lhs']} -> {h['rhs']}")
        else:
            print(f"  {label}: not found in-budget")
    if args.json:
        _dump_json(args.json, r)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
