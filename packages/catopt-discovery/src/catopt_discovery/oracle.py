"""View/index oracle — honest instantiation for view-family candidates.

The pipeline's numeric oracle (``catopt_discovery.proposal._numeric_true``) proves a
candidate equality by evaluating *one* concrete instance — the first
viable real match — on random fp64 tensors.  That is sound but
incomplete for the ``mixed:`` view/index family:

* **Single match.**  ``_instance_from_match`` returns the first match
  whose ``check``/``derive`` hooks pass; if the instantiated RHS is
  ill-typed *on that match* (``mul(u, v)`` where ``v`` carries the
  select-output shape, not ``u``'s) the oracle abstains
  (``num_true=None`` → "unproven") even though a later match is
  provably true, and even though *some* shape assignment always
  satisfies both sides.
* **Wrong leaf semantics.**  ``getitem`` in this IR picks either a
  tuple element (``topk``/``var_mean``/``cummax``/``mode`` outputs —
  the corpus's real matches) or a dim-0 tensor index.  A metavar
  sitting under ``getitem`` is *not* honestly a fresh tensor ``Var``:
  instantiating it as one tests the wrong equality (and misses that
  ``add(getitem(u,i),v) → add(u,v)`` is ill-typed, not merely false,
  for tuple-valued ``u``).
* **No satisfiability check.**  A candidate whose RHS can *never* be
  well-typed where the LHS is (pointwise op over a tuple operand) is
  ill-formed — a different verdict from "unproven".

This tool is the missing oracle.  For a pattern pair ``(lhs, rhs)`` it

1. **sweeps every real match** (not the first): instantiate the RHS
   through the proposal's ``check``/``derive`` hooks, evaluate both
   sides tri-state (``equal`` / ``unequal`` / side-error);
2. **synthesizes satisfiable instances**: enumerate the pattern's leaf
   metavariables over a small shape bank (plus ``Const`` scalars for
   free operands and *tuple-producing* terms for a metavar under
   ``getitem``), enumerate each view node's attribute metavariables
   over values *valid for the chosen operand shape*, and evaluate both
   instantiated sides fp64 — an honest "does a well-typed instance
   exist, and does the equality hold there";
3. **resolves a verdict**: ``true`` (every both-sides-evaluable
   instance equal), ``conditional`` (equal and unequal instances both
   exist — the separating shape condition is reported as a guard
   description), ``false`` (both-typed instances exist and none
   agree), ``ill-formed`` (the LHS evaluates but the RHS never can —
   the rewrite's target does not denote), or ``unproven`` (no
   evaluable instance at all).

The verdict feeds ``catopt_discovery.pipeline``'s ``measure`` step
(``--no-view-oracle`` disables).  A ``conditional`` candidate is *not*
auto-admitted: it is evidence for a guarded law — the interesting
outcome — reported for review.

Run::

    .venv/bin/python -m catopt_discovery.oracle
    .venv/bin/python -m catopt_discovery.oracle --json /tmp/view_oracle.json

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from catopt_core.attrs import ATTR_SCHEMA, is_positional_attr
from catopt_core.egraph.terms import _term_instantiate, _term_match
from catopt_core.ir import Const, Op, Param, TensorType, Var, op_repr
from catopt_core.opmeta import REDUCE_DIM_OPS, VIEWISH_OPS

# Sibling tools own the corpus, the eval backend and the comparator;
# reuse them, never duplicate.  The enumeration *banks* (shape banks,
# attr kinds, per-op constants, tuple sources) are pure data in
# ``catopt_discovery.lawdata`` — the table lives there, bound here.
from catopt_discovery import lawdata
from catopt_discovery import proposal as lp

__all__ = [
    "Instance",
    "ViewVerdict",
    "escalate_limit",
    "eval_instance",
    "sweep_real",
    "synthesize",
    "verify_view_candidate",
]

#: Numeric-comparison tolerance (fp64), shared with ``law_proposal``.
_TOL = 1e-6

#: Cap on synthesized instances per candidate — the default window.
_MAX_INSTANCES = 360

#: Second-phase cap for a *starved* guarded-region sweep.  A
#: multi-clause guard's accepted corner sits deep in the fair-order
#: enumeration: the diagonal interleaves every ``(viewed x attr)``
#: base, so the base whose guard admits a site is reached only after
#: each base before it has spent a round.  Measured on the shipped
#: ``sdpa_fold_*`` folds (limit → first equal site): ``sdpa_fold_add``
#: 469, ``sdpa_fold_addmul`` / ``sdpa_fold_adddiv`` 793 — all past the
#: 360 default, so the default window accepts nothing and the truth
#: gate refuses a sound law as vacuous.  The escalation is *selective*
#: (see :func:`escalate_limit`): the common case keeps the default.
_GUARDED_CAP = 2000


def escalate_limit(limit: int, *, guarded: bool, accepted: int) -> int:
    """Return the effective enumeration cap after one guarded sweep.

    The selective cap policy: the default window stays the common
    case.  A sweep escalates to :data:`_GUARDED_CAP` only when it is
    *starved* — the rule carries a guard (``cond`` / ``check``) and
    the window accepted no site at all, i.e. it reached only declines
    and guard errors.  A multi-clause guard's accepted region is a
    corner the fair ordering reaches late; the accepted corner of the
    shipped ``sdpa_fold_*`` folds lies at index 469-793, past the
    default.

    The escalation never fires for an unguarded rule (no guard region
    to starve) nor for a window that already accepted a site (the
    region is non-empty; the cap is not what is biting).  The
    returned cap is never below *limit*.
    """
    if guarded and accepted == 0:
        return max(limit, _GUARDED_CAP)
    return limit


#: View/index ops this oracle knows how to attribute-instantiate.
#: Anything outside the table leaves the attr metavariables unbound —
#: the instance is skipped, honestly.  A projection of
#: :mod:`catopt_core.opmeta` (the ``viewish`` tag).
_VIEWISH = VIEWISH_OPS


# ---------------------------------------------------------------------------
#  Instance records and verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One evaluated instantiation of a candidate equality.

    ``outcome`` is ``equal`` / ``unequal`` when both sides evaluated,
    ``lhs-err`` / ``rhs-err`` / ``both-err`` / ``env-err`` otherwise.
    ``origin`` is ``real`` (a corpus match) or ``synth`` (a synthesized
    binding).  ``feats`` carries the mechanical guard features —
    ``(name, value)`` pairs sorted — used to *name* the condition a
    conditional law would need.
    """

    origin: str
    outcome: str
    lhs_repr: str = ""
    rhs_repr: str = ""
    binds: tuple = ()
    feats: tuple = ()
    note: str = ""


@dataclass
class ViewVerdict:
    """The oracle's resolution of one candidate."""

    name: str
    verdict: str = "unproven"
    guard: str = ""
    note: str = ""
    n_real: int = 0
    real_equal: int = 0
    real_unequal: int = 0
    real_rhs_err: int = 0
    real_lhs_err: int = 0
    n_synth: int = 0
    synth_equal: int = 0
    synth_unequal: int = 0
    synth_rhs_err: int = 0
    witness: str = ""
    counterexample: str = ""
    instances: tuple = ()


# ---------------------------------------------------------------------------
#  Tri-state evaluation
# ---------------------------------------------------------------------------


def _env_for(*terms: Any) -> dict | None:
    """Build the fp64 env for *terms*' leaves; ``None`` on unknown dims."""
    leaves = set()
    for t in terms:
        leaves |= lp._leaves(t)
    env: dict = {}
    for leaf in leaves:
        shape = tuple(leaf.typ.shape) if leaf.typ is not None else ()
        if any(not isinstance(d, int) for d in shape):
            return None
        env[leaf] = torch.randn(shape, dtype=torch.float64)
    return env


def _eval(term: Any, env: dict) -> tuple[bool, Any]:
    """Evaluate *term*; return ``(ok, value-or-error-string)``."""
    try:
        return True, lp._eval_backend().eval_term(term, env)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def eval_instance(lhs: Any, rhs: Any) -> tuple[str, str]:
    """Tri-state numeric check of the concrete pair ``(lhs, rhs)``.

    Returns ``(outcome, note)`` where outcome is one of ``equal``,
    ``unequal``, ``lhs-err``, ``rhs-err``, ``both-err``, ``env-err``.
    Unlike ``lp._numeric_true`` the sides' failures are reported
    separately — an ill-typed RHS is the *signal*, not noise.
    """
    env = _env_for(lhs, rhs)
    if env is None:
        return "env-err", "leaf with non-int dims"
    lok, a = _eval(lhs, env)
    rok, b = _eval(rhs, env)
    if not lok and not rok:
        return "both-err", f"lhs: {a}; rhs: {b}"
    if not lok:
        return "lhs-err", str(a)
    if not rok:
        return "rhs-err", str(b)
    return ("equal" if lp._allclose(a, b, _TOL) else "unequal"), ""


# ---------------------------------------------------------------------------
#  Pattern introspection
# ---------------------------------------------------------------------------


def _leaf_metavars(pat: Any, out: list | None = None) -> list[str]:
    """Ordered unique metavariable strings at leaf positions."""
    out = [] if out is None else out
    if isinstance(pat, str):
        if pat not in out:
            out.append(pat)
    elif isinstance(pat, Op):
        for a in pat.args:
            _leaf_metavars(a, out)
    return out


def _parents(pat: Any) -> dict[str, set[str]]:
    """Map each metavar to the ops of its parent nodes."""
    out: dict[str, set[str]] = {}

    def walk(t: Any) -> None:
        if isinstance(t, Op):
            for a in t.args:
                if isinstance(a, str):
                    out.setdefault(a, set()).add(t.op)
                walk(a)

    walk(pat)
    return out


def _view_nodes(pats: Iterable[Any]) -> list[Op]:
    """Every node carrying a string-valued (metavariable) attribute."""
    nodes: list[Op] = []
    seen: set[int] = set()
    for pat in pats:
        stack = [pat]
        while stack:
            t = stack.pop()
            if id(t) in seen:
                continue
            seen.add(id(t))
            if isinstance(t, Op):
                if any(isinstance(v, str) for v in t.attrs.values()):
                    nodes.append(t)
                stack.extend(t.args)
    return nodes


# ---------------------------------------------------------------------------
#  Attribute domains — values valid for the operand's shape
# ---------------------------------------------------------------------------


def _dims(rank: int) -> list[int]:
    """Return a small set of axis spellings valid for *rank*."""
    if rank <= 0:
        return []
    raw = {0, rank - 1, -1, -rank, 1 if rank > 1 else 0}
    return sorted(raw)


def _norm(d: int, rank: int) -> int:
    return d % rank if rank else d


def _shape_numel(shape: tuple) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


#: Cap on ``_reshape_targets`` — raised from 7 when interior adjacent
#: merges joined the candidate list (rank >= 4 shapes): every existing
#: seat keeps its index; the new candidates land past them.
_RESHAPE_TARGET_CAP = 12


def _interior_merges(shape: tuple) -> Iterable[tuple]:
    """Yield the interior adjacent-pair merges of *shape*."""
    for i in range(1, len(shape) - 2):
        yield (*shape[:i], shape[i] * shape[i + 1], *shape[i + 2 :])


def _expand_targets(sh: tuple, r: int) -> list[tuple]:
    """Broadcastable target shapes for an ``expand`` metavariable."""
    outs = [(2, *sh)] if r else [(2,)]
    ones = [i for i, d in enumerate(sh) if d == 1]
    if not ones:
        return outs
    grown = list(sh)
    grown[ones[0]] = 3
    outs.append(tuple(grown))
    # Each 1-extent grown by a small factor — a repeat-chain's expand
    # grows exactly the unsqueezed axis (the ``ES`` target in
    # ``unsqueeze→expand→reshape``).  Appended after the original
    # grown-first seat to keep domain indices.
    for i in ones:
        for f in (2, 3):
            g = list(sh)
            g[i] = f
            if tuple(g) not in outs:
                outs.append(tuple(g))
    return outs


def _reshape_targets(shape: tuple) -> list[tuple]:
    """Same-numel targets for a reshape ``shape`` metavariable."""
    n = _shape_numel(shape)
    cands = [
        tuple(shape),
        (n,),
        (n, 1),
        (1, n),
        tuple(reversed(shape)),
    ]
    if len(shape) >= 2 and shape[0] > 1:
        cands.append((shape[0] // 2, 2, *shape[1:]))
    if len(shape) >= 3:
        cands.append((shape[0] * shape[1], *shape[2:]))
        cands.append((*shape[:-2], shape[-2] * shape[-1]))
        # Interior adjacent merges — a repeat-chain's reshape merges
        # dims ``d-1``/``d`` (the ``head * r`` pair), which need not
        # be an edge pair.  Appended after the edge merges so existing
        # domain indices keep their seats.
        cands.extend(_interior_merges(shape))
    out: list[tuple] = []
    for c in cands:
        if (
            all(isinstance(d, int) and d > 0 for d in c)
            and c not in out
            and _shape_numel(c) == n
        ):
            out.append(c)
    return out[:_RESHAPE_TARGET_CAP]


def _attr_options(
    op: str, keys: tuple, shape: Any
) -> list[dict] | None:
    """Return concrete attr dicts for *op*'s metavar'd keys, or ``None``.

    *shape* is the operand's inferred shape (``tuple`` or ``()``);
    tuple-producing operands report their element shape.  ``None``
    means the oracle cannot honestly instantiate this op — the
    instance is skipped, not guessed.  View-family ops take the
    hand-tuned tables; anything else falls to
    :func:`_generic_attr_options`, which types each attr through the
    canonical schema and enumerates a small domain per kind — a key
    it cannot type stays ``None``.
    """
    sh = tuple(shape) if isinstance(shape, tuple) else ()
    r = len(sh)
    dims = _dims(r)
    match op:
        case "getitem":
            idx = {0, 1}
            return [{"index": i} for i in sorted(idx)]
        case "select":
            if r < 1:
                return []
            out = []
            for d in dims[:3]:
                ext = sh[_norm(d, r)]
                for i in {0, 1}:
                    if isinstance(ext, int) and i < ext:
                        out.append({"dim": d, "index": i})
            return out[:8]
        case "slice":
            if r < 1:
                return []
            out = []
            for d in dims[:3]:
                ext = sh[_norm(d, r)]
                if not isinstance(ext, int):
                    continue
                for lo, hi in (
                    (0, ext),
                    (0, max(1, ext // 2)),
                    (0, 1),
                    (1, ext),
                ):
                    a = {"dim": d, "start": lo, "end": hi}
                    if "step" in keys:
                        a["step"] = 1
                    out.append(a)
            return out[:10]
        case "unsqueeze":
            ds = {d for d in (0, 1, -1, -2, r) if -(r + 1) <= d <= r}
            return [{"dim": d} for d in sorted(ds)]
        case "squeeze":
            ones = [i for i, d in enumerate(sh) if d == 1] or dims[:2]
            return [{"dim": d} for d in sorted(set(ones))]
        case "transpose" | "t":
            if r < 1:
                return []
            pairs = {(0, 0), (0, 1), (0, -1), (-2, -1), (1, 0)}
            return [
                {"dim0": a, "dim1": b}
                for a, b in sorted(pairs)
                if -r <= a < r and -r <= b < r
            ][:6]
        case "reshape" | "view":
            return [{"shape": s} for s in _reshape_targets(sh)]
        case "expand" | "broadcast_to":
            return [{"shape": s} for s in _expand_targets(sh, r)]
        case "chunk":
            out = []
            for c in (1, 2):
                for d in dims[:2]:
                    ext = sh[_norm(d, r)]
                    if isinstance(ext, int) and ext >= c:
                        for i in range(c):
                            out.append(
                                {"chunks": c, "dim": d, "index": i}
                            )
            return out[:8]
        case "split" | "tensor_split":
            out = []
            for d in dims[:2]:
                ext = sh[_norm(d, r)]
                if isinstance(ext, int) and ext >= 2:
                    out.append(
                        {
                            "sizes": (ext // 2, ext - ext // 2),
                            "dim": d,
                            "index": 0,
                        }
                    )
            return out[:4]
        case "narrow":
            out = []
            for d in dims[:2]:
                ext = sh[_norm(d, r)]
                if isinstance(ext, int):
                    for ln in {1, ext}:
                        out.append({"dim": d, "start": 0, "length": ln})
            return out[:6]
        case "permute":
            perms = [tuple(range(r))]
            if r >= 2:
                perms.append(tuple(reversed(range(r))))
            return [{"dim": p} for p in perms if p]
        case "unbind":
            return [{"dim": d, "index": 0} for d in dims[:2]]
        case "movedim":
            if r < 2:
                return []
            return [
                {"source": 0, "destination": r - 1},
                {"source": r - 1, "destination": 0},
            ]
        case "flatten":
            return [{"start_dim": 0, "end_dim": -1}]
        case _:
            return _generic_attr_options(op, keys, shape)


# ---------------------------------------------------------------------------
#  Generic attr domains — non-view ops keyed by the attr schema
# ---------------------------------------------------------------------------
#
#  The view table above enumerates attrs *shape-valid for a bound
#  operand*; ops outside it used to veto the whole binding — an attr
#  metavariable on ``softmax``/``sdpa``/``sum`` emptied the synthesized
#  domain entirely, even when a rule's ``derive`` would overwrite the
#  value anyway.  The generic fallback restores a non-empty domain:
#  each metavar'd attr key is *typed* — the canonical name resolved
#  through ``catopt_core.attrs.ATTR_SCHEMA`` (so ``arg6`` on ``sdpa``
#  kinds as ``scale``) — and the kind picks a small honest domain.
#  The enumeration proposes candidates; the fp64 eval disposes —
#  an out-of-range axis or ill-typed tuple surfaces as a counted
#  eval error, the same posture the leaf-shape banks take.  An attr
#  key the table cannot type (``attn_mask``, an ``equation`` string)
#  stays honestly unenumerable — ``None``, the binding is skipped.

#: Canonical attr names -> the value kind the sweep can enumerate.
#: ``ATTR_SCHEMA`` *names* every positional attr; this table *types*
#: the names.  The table lives in :mod:`catopt_discovery.lawdata`
#: (:data:`ATTR_KINDS`) — this is the same object bound to the
#: consumer-side name.
_ATTR_KINDS: dict[str, str] = lawdata.ATTR_KINDS

#: ``(op, canonical-attr)`` pairs whose value kind differs from the
#: name default (:data:`catopt_discovery.lawdata.ATTR_KIND_OVERRIDES`).
_ATTR_KIND_OVERRIDES: dict[tuple[str, str], str] = (
    lawdata.ATTR_KIND_OVERRIDES
)

#: Ops whose ``dim`` attr accepts an axis OR a tuple of axes (the
#: aten reduction signature) — the domain enumerates both spellings.
#: A projection of :mod:`catopt_core.opmeta` (the ``reduce-dim`` tag).
_REDUCTION_DIM_OPS = REDUCE_DIM_OPS

#: Cap on one node's generic option dicts (the product over its
#: metavar'd keys) — matches the view tables' per-node caps.
_MAX_GENERIC_OPTIONS = 16

#: Axes offered when the operand's shape is not visible
#: (:data:`catopt_discovery.lawdata.FALLBACK_AXES`).  Out-of-range
#: draws surface as counted eval errors — honest enumeration, honest
#: accounting.
_FALLBACK_AXES: tuple = lawdata.FALLBACK_AXES

#: Trailing tuples offered for a shape-typed attr whose operand
#: shape is unknown (:data:`catopt_discovery.lawdata.FALLBACK_SHAPES`).
_FALLBACK_SHAPES: tuple = lawdata.FALLBACK_SHAPES

#: Sentinel shapes the free operand's derived bank always carries
#: (:data:`catopt_discovery.lawdata.FREE_SENTINELS`).
_FREE_SENTINELS: tuple = lawdata.FREE_SENTINELS


def _attr_kind(op: str, key: str) -> str | None:
    """Return the value kind of *op*'s attr *key* — ``None`` if unknown.

    ``argN`` positional spellings resolve to the canonical name
    through ``ATTR_SCHEMA`` first, then ``(op, name)`` overrides and
    the reduction-``dim`` family take precedence over the name table.
    """
    canon = key
    if is_positional_attr(key):
        canon = (ATTR_SCHEMA.get(op) or {}).get(int(key[3:]), key)
    kind = _ATTR_KIND_OVERRIDES.get((op, canon))
    if kind is not None:
        return kind
    if canon == "dim" and op in _REDUCTION_DIM_OPS:
        return "red-dims"
    return _ATTR_KINDS.get(canon)


def _kind_domain(kind: str, shape: tuple) -> list:
    """Return the candidate values one attr *kind* ranges over.

    *shape* is the operand's shape (``()`` when unknown); kinds that
    need it fall back to a small generic domain — the enumeration
    proposes, the eval decides.
    """
    match kind:
        case "axis":
            return _dims(len(shape)) or list(_FALLBACK_AXES)
        case "red-dims":
            return _red_dims_domain(len(shape))
        case "int":
            return [0, 1, 2]
        case "float":
            return [0.5, 1.0, 1e-5]
        case "bool":
            return [False, True]
        case "shape":
            return _shape_block_domain(shape)
        case _:
            return []


def _red_dims_domain(r: int) -> list:
    """Return the ``reduce``-dim payload — axes plus the tuples."""
    out: list = _dims(r) or list(_FALLBACK_AXES)
    out.append((-1,))
    if r > 1:
        out.append(tuple(range(r)))
    return out


def _shape_block_domain(shape: tuple) -> list:
    """Return the ``weight``-shaped payload — trailing blocks of *shape*."""
    if len(shape):
        return [
            tuple(shape[-k:]) for k in range(1, min(len(shape), 3) + 1)
        ]
    return list(_FALLBACK_SHAPES)


def _generic_attr_options(
    op: str, keys: tuple, shape: Any
) -> list[dict] | None:
    """Return concrete attr dicts for a non-view op, or ``None``.

    Each metavar'd key contributes its kind's domain; the option dicts
    are the Cartesian product, capped at ``_MAX_GENERIC_OPTIONS``.
    ``None`` marks an attr key the table cannot type — the instance is
    skipped, honestly.
    """
    sh = tuple(shape) if isinstance(shape, tuple) else ()
    domains: list[tuple[str, list]] = []
    for k in keys:
        kind = _attr_kind(op, k)
        if kind is None:
            return None
        vals = _kind_domain(kind, sh)
        if not vals:
            return []
        domains.append((k, vals))
    names = [k for k, _ in domains]
    return [
        dict(zip(names, combo, strict=True))
        for combo in itertools.islice(
            itertools.product(*[v for _, v in domains]),
            _MAX_GENERIC_OPTIONS,
        )
    ]


# ---------------------------------------------------------------------------
#  Binding domains for leaf metavariables
# ---------------------------------------------------------------------------


#: Shapes offered to a metavariable that sits under a view op
#: (:data:`catopt_discovery.lawdata.VIEWED_SHAPES`).
_VIEWED_SHAPES: tuple = lawdata.VIEWED_SHAPES

#: Extra viewed-bank shapes for patterns that contain an
#: operand-*chained* view
#: (:data:`catopt_discovery.lawdata.CHAIN_VIEWED_SHAPES`).  Scoped to
#: chained patterns so chain-free enumerations are byte-identical:
#: appending to the universal bank would add free-operand cells at
#: existing index sums and displace already-measured window-tail
#: sites.
_CHAIN_VIEWED_SHAPES: tuple = lawdata.CHAIN_VIEWED_SHAPES


#: Per-op literal-constant domain — the values a metavariable's
#: *parent op* admits, keyed by the op name.  The leaf bank's generic
#: scalar corner mints a single ``Const(0.5)``; a shipped guard can
#: demand a specific literal that corner never reaches (the
#: *domain-gapped* class of ``project/retros/cap-policy.md``).  Only
#: the ops a shipped guard actually constrains carry an entry —
#: ``mul`` / ``add`` identities were measured and left out (a
#: widening with no rescue perturbs the capped order).  The table
#: lives in :mod:`catopt_discovery.lawdata`; see
#: ``project/retros/value-bank.md`` for the measured cost.
_CONST_DOMAIN: dict[str, tuple[int | float, ...]] = lawdata.CONST_DOMAIN


def _const_domain(parents: set[str]) -> list[Const]:
    """Return the literal ``Const`` bindings *parents*' ops admit.

    Keyed by the metavariable's parent op — the op the leaf is an
    operand of.  The values are tabulated in :data:`_CONST_DOMAIN`;
    a metavariable whose parents carry no entry keeps the generic
    ``Const(0.5)`` corner the free bank always carried.  Order is
    stable (sorted op, then the table's order) so the enumeration
    stays deterministic.
    """
    vals: list[int | float] = []
    for op in sorted(parents):
        for v in _CONST_DOMAIN.get(op, ()):
            if v not in vals:
                vals.append(v)
    if not vals:
        vals = [0.5]
    return [Const(v) for v in vals]


def _insert_op_consts(out: list[Any], parents: set[str]) -> None:
    """Splice the op-specific literals just past the generic corner.

    The generic ``Const(0.5)`` (index 1) and scalar ``Var`` (index 2)
    keep their documented seats — a widening must not shift an
    existing accepted corner — so the op literals follow them,
    deduped against the corner (``pow``'s ``0.5`` does not double).
    """
    pos = 3
    for c in _const_domain(parents):
        if any(isinstance(t, Const) and t == c for t in out):
            continue
        out.insert(pos, c)
        pos += 1


def _tuple_sources(mv: str) -> list[Any]:
    """Tuple-producing terms for a metavar under ``getitem``.

    The corpus's real ``getitem`` matches pick elements out of
    ``topk`` / ``var_mean`` / ``cummax`` — a bare ``Var`` only covers
    the dim-0 tensor index.  Both meanings are instantiated; the
    eval decides.
    """
    w = Var(f"{mv}@t", TensorType(lawdata.TUPLE_SOURCE_SHAPE))
    return list(
        map(
            lambda oa: Op.make(oa[0], w, **oa[1]),
            lawdata.TUPLE_SOURCES,
        )
    )


def _leaf_bindings(
    mv: str,
    parents: set[str],
    derived: Iterable[tuple],
    extra_viewed: tuple = (),
) -> list[Any]:
    """Candidate bindings for one leaf metavariable.

    ``derived`` is the extra shape list computed from *other* leaves'
    choices (filled in by the enumerator for the free operand).
    ``extra_viewed`` appends pattern-scoped shapes just past the
    viewed bank — the rank-4 pair a chained-view pattern needs
    (:data:`_CHAIN_VIEWED_SHAPES`); it stays empty otherwise so a
    chain-free enumeration is unchanged.
    """
    out: list[Any] = []
    if "getitem" in parents:
        out.extend(_tuple_sources(mv))
    # The operand-derived shapes lead the bank: a bound operand's own
    # shape is the most conservative instantiation and belongs at
    # index 0 the way the scalar probe did — under a capped
    # enumeration the scalar-led order buries the evaluable corner
    # (measured: for a three-free-operand pattern the first equal
    # site lay ~10⁴ sites into the scalar-led order, ~40 into the
    # operand-led one).
    for s in itertools.chain(derived, _VIEWED_SHAPES, extra_viewed):
        out.append(Var(mv, TensorType(tuple(s))))
    if not parents or all(p not in _VIEWISH for p in parents):
        # A free operand may bind a literal scalar — the corpus does,
        # and a shipped guard may demand a *specific* literal the
        # generic corner never reaches (:func:`_const_domain`).  The
        # generic ``Const(0.5)`` keeps its documented index-1 seat and
        # the scalar ``Var`` its index 2; the op-specific constants
        # follow them (:func:`_insert_op_consts`), so an existing
        # accepted corner is never shifted past the cap (the bank
        # feeds the enumeration — a widening must not bury a corner
        # that already measured).
        out.insert(1, Var(mv, TensorType(())))
        out.insert(1, Const(0.5))
        _insert_op_consts(out, parents)
    return out


def _operand_shape(term: Any) -> tuple:
    """Return the declared shape of a bound operand (``()`` if unknown).

    A compound bound term can carry *unresolved* attr metavariables
    (``reshape(A, shape="S1")`` nested inside another view's operand)
    whose ``_shape_of`` surfaces as non-int dims; those are not a
    shape.  The fallback is the first tensor leaf's declared shape —
    the rank is what the attr domains actually need.
    """
    if isinstance(term, (Var, Param)):
        return tuple(term.typ.shape)
    if isinstance(term, Const):
        return ()
    from catopt_core.typing import _shape_of

    s = _shape_of(term)
    if isinstance(s, tuple) and all(
        isinstance(d, int) and d >= 0 for d in s
    ):
        return tuple(s)
    stack = list(term.args) if isinstance(term, Op) else []
    while stack:
        t = stack.pop()
        if isinstance(t, (Var, Param)):
            return tuple(t.typ.shape)
        if isinstance(t, Op):
            stack.extend(t.args)
    return ()


# ---------------------------------------------------------------------------
#  Instance enumeration
# ---------------------------------------------------------------------------


def _combos_at_sum(lists: tuple, total: int) -> Iterable[tuple]:
    """Yield ``product(*lists)`` tuples whose index-sum is *total*."""
    if not lists:
        if total == 0:
            yield ()
        return
    head, rest = lists[0], lists[1:]
    for i, v in enumerate(head):
        if i > total:
            break
        for tail in _combos_at_sum(rest, total - i):
            yield (v, *tail)


def _diag_product(lists: list) -> Iterable[tuple]:
    """Yield ``product(*lists)`` in Cantor (index-sum) order.

    ``itertools.product`` advances the last coordinate fastest, so a
    capped enumeration barely moves the early coordinates — a binding
    deep in a *first* metavariable's list is never reached (measured:
    three free operands, ``Const`` last, cap 360 → the ``Const``
    binding lay ~10⁴ tuples in).  Diagonal order covers the low-index
    corner of every coordinate first: a cap truncates a *corner* of
    the product space, not a whole dimension.
    """
    sizes = [len(lst) for lst in lists]
    if any(s == 0 for s in sizes):
        return
    for total in range(sum(s - 1 for s in sizes) + 1):
        yield from _combos_at_sum(tuple(lists), total)


def _has_chained_view(nodes: list[Op]) -> bool:
    """Whether *nodes* contain an operand-linked view pair."""
    ids = {id(n) for n in nodes}
    return any(
        n.op in _VIEWISH and n.args and id(n.args[0]) in ids
        for n in nodes
    )


def _viewed_bindings(
    mvs: list[str], parents: dict, extra_viewed: tuple = ()
) -> Iterable[dict]:
    """Yield binding dicts for metavariables under a view node."""
    viewed = [m for m in mvs if parents.get(m, set()) & _VIEWISH]
    lists = [
        _leaf_bindings(m, parents[m], (), extra_viewed) for m in viewed
    ]
    for combo in _diag_product(lists):
        yield dict(zip(viewed, combo, strict=True))


def _derived_free_shapes(
    u_shapes: list[tuple], out_shapes: list[tuple]
) -> list[tuple]:
    """Return the derived shape bank for a free (non-viewed) operand.

    The candidates that probe the boundary region the naturality
    candidates hinge on — the viewed operand's own shape FIRST (it
    leads the free bank: the most conservative instantiation), then
    each view's output shape, every single-axis-1 insertion of each
    (the broadcast-pad cases), a leading pad, and genuinely
    mismatched shapes.
    """
    derived: list[tuple] = []
    for sh in itertools.chain(u_shapes, out_shapes, _FREE_SENTINELS):
        if sh in derived:
            continue
        derived.append(sh)
        for ins in _pad_insertions(sh):
            if ins not in derived:
                derived.append(ins)
    return derived


def _pad_insertions(sh: tuple) -> Iterable[tuple]:
    """Every single-axis-1 insertion of *sh* (the broadcast pads)."""
    for i in range(len(sh) + 1):
        yield (*sh[:i], 1, *sh[i:])


def _mv_names(node: Op) -> tuple:
    """Return the node's sorted attr-metavariable name tuple."""
    return tuple(
        sorted(v for v in node.attrs.values() if isinstance(v, str))
    )


def _attr_components(nodes: list[Op]) -> Any:
    """Union-find ``find`` over *nodes*' attr-metavar components.

    Two merge rules:

    * **shared name set** — nodes whose sorted attr-metavar name
      tuples coincide resolve once (the LHS/RHS ``sdpa`` twin);
    * **operand link** — a *view* node whose first operand *is*
      another metavariable-attred node chains onto it
      (``expand(unsqueeze(k, "UDk"), "ESk")``): the outer view's
      domain depends on the inner draw, so the link enumerates the
      chain's draws jointly.  Non-view consumers (``dropout`` over a
      ``softmax``) keep their own group — their attr domains are
      shape-independent, and linking them would only reshuffle the
      diagonal order.
    """
    idx = {id(n): i for i, n in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple, int] = {}
    for i, node in enumerate(nodes):
        names = _mv_names(node)
        # Nameless nodes bind nothing — keep them out of the groups
        # (a fabricated literal-attr node must not mint a group).
        if not names:
            continue
        if names in seen:
            parent[find(i)] = find(seen[names])
        else:
            seen[names] = i
        if (
            node.op in _VIEWISH
            and node.args
            and id(node.args[0]) in idx
        ):
            parent[find(i)] = find(idx[id(node.args[0])])
    return find


def _attr_groups(nodes: list[Op]) -> list[list[Op]]:
    """Group *nodes* into connected attr-metavar components."""
    find = _attr_components(nodes)
    comps: dict[int, list[Op]] = {}
    order: list[int] = []
    for i, node in enumerate(nodes):
        if not _mv_names(node):
            continue
        r = find(i)
        if r not in comps:
            comps[r] = []
            order.append(r)
        comps[r].append(node)
    return [comps[r] for r in order]


def _chain_order(members: list[Op]) -> list[Op]:
    """Order a chained group innermost-first (operand before consumer).

    ``depth`` counts linked operands below a node; a member whose
    operand is not in the group is a chain root.  The sort is stable,
    so same-depth members keep discovery order.
    """
    ids = {id(n) for n in members}
    depth: dict[int, int] = {}

    def d(n: Op) -> int:
        if id(n) not in depth:
            depth[id(n)] = (
                1 + d(n.args[0])
                if n.args and id(n.args[0]) in ids
                else 0
            )
        return depth[id(n)]

    return sorted(members, key=d)


@dataclass(frozen=True)
class _ChainGroup:
    """Pseudo-node for a chained attr group, for ``_attr_merge``.

    A chained group's option dicts are keyed by *metavariable name*
    (the draws are joint, not per-node), so ``attrs`` maps each
    metavariable name to itself — the merge's ``$attr:name`` lookup
    reads the value straight through.
    """

    attrs: dict


def _chain_shape(node: Op, assign: dict, subst: dict) -> Any:
    """Return the operand's shape under the partial chain draw.

    Instantiating the operand under the drawn ``$attr:`` bindings
    resolves the inner view's attrs, so ``_operand_shape`` returns the
    real intermediate shape — the ``unsqueeze`` output, not the
    pre-view leaf.
    """
    if not node.args:
        return ()
    try:
        bound = _term_instantiate(
            node.args[0],
            {**subst, **{f"$attr:{m}": v for m, v in assign.items()}},
        )
    except Exception:
        return ()
    return _operand_shape(bound)


def _chain_extend(node: Op, acc: list[dict], subst: dict) -> list[dict]:
    """Extend each partial assignment by *node*'s consistent options.

    An option that disagrees with an already-drawn metavariable
    binding is dropped — resolve-once, as ``_attr_merge`` applies
    across groups.  A partial assignment with no consistent option
    dies with it.
    """
    mv_attrs = {
        k: v for k, v in node.attrs.items() if isinstance(v, str)
    }
    nxt: list[dict] = []
    for assign in acc:
        shape = _chain_shape(node, assign, subst)
        opts = _attr_options(node.op, tuple(mv_attrs), shape) or ()
        for o in opts:
            vals = {mv_attrs[k]: o[k] for k in mv_attrs if k in o}
            if any(
                assign[mv] != v
                for mv, v in vals.items()
                if mv in assign
            ):
                continue
            nxt.append({**assign, **vals})
    return nxt


def _chain_domain(members: list[Op], subst: dict) -> list[dict] | None:
    """Enumerate consistent joint attr assignments over a chain.

    Members are processed innermost-first; each node's attr options
    are computed against the operand shape the *drawn* inner attrs
    produce (:func:`_chain_shape`) — the cond DSL's
    ``unsq-out``/``reshape-out``/``transpose-out`` vocabulary made
    concrete through the same typing engine the eval side uses — so
    chained domains no longer fall back to the leaf shape that made
    them unreachable.  ``None`` marks a member the oracle cannot
    instantiate honestly — the whole binding is skipped.
    """
    acc: list[dict] = [{}]
    for node in _chain_order(members):
        acc = _chain_extend(node, acc, subst)
        if not acc:
            return None
    return acc


def _is_chained_group(members: list[Op], member_ids: set) -> bool:
    """Return whether the group links a view onto an attred operand."""
    return len(members) > 1 and any(
        m.op in _VIEWISH and m.args and id(m.args[0]) in member_ids
        for m in members
    )


def _single_domain(node: Op, subst: dict) -> list[dict] | None:
    """Return one unchained node's option dicts, or ``None``.

    For getitem over a tuple source the index domain is the
    tuple arity, not an axis extent — options stay {0,1}.
    Only the metavar'd keys are enumerated — a literal attr
    contributes no binding and must not veto the domain.
    """
    try:
        bound = (
            _term_instantiate(node.args[0], subst)
            if node.args
            else None
        )
    except Exception:
        bound = None
    shape = _operand_shape(bound) if bound is not None else ()
    mv_keys = tuple(
        k for k, v in node.attrs.items() if isinstance(v, str)
    )
    return _attr_options(node.op, mv_keys, shape) or None


def _attr_domains(
    nodes: list[Op], subst: dict
) -> list[tuple[Any, list[dict]]] | None:
    """Return ``(node, option dicts)`` per distinct attr-metavar group.

    Each group is a connected set of nodes — shared metavariable-name
    sets or operand-chained views (:func:`_attr_groups`).  A single
    representative node's option dicts assign concrete values to its
    *attr keys*; a chained group's dicts are keyed by metavariable
    name and carry the whole chain's consistent draw
    (:func:`_chain_domain`).  ``None`` marks a node the oracle cannot
    instantiate honestly.
    """
    out = []
    member_ids = {id(n) for n in nodes}
    for members in _attr_groups(nodes):
        node = members[0]
        if _is_chained_group(members, member_ids):
            assigns = _chain_domain(members, subst)
            if not assigns:
                return None
            mvs = sorted({v for m in members for v in _mv_names(m)})
            out.append(
                (_ChainGroup(attrs={m: m for m in mvs}), assigns)
            )
            continue
        opts = _single_domain(node, subst)
        if not opts:
            return None
        out.append((node, opts))
    return out


def _attr_merge(domains: list, attr_combo: tuple, viewed: dict) -> Any:
    """Merge one attr combination into *viewed*; ``None`` on conflict.

    Shared attr metavariables across two nodes must resolve
    identically — a combo disagreeing with an earlier binding is not
    a legal instantiation.
    """
    base = dict(viewed)
    for (node, _opts), vals in zip(domains, attr_combo, strict=True):
        for k, mv_name in node.attrs.items():
            if not isinstance(mv_name, str):
                continue
            ak = f"$attr:{mv_name}"
            if ak in base and base[ak] != vals.get(k):
                return None
            if k in vals:
                base[ak] = vals[k]
    return base


def _lhs_out_shapes(lhs_views: list, base: dict) -> list:
    """Return the instantiated LHS view nodes' output shapes."""
    out: list = []
    for n in lhs_views:
        try:
            s = _operand_shape(_term_instantiate(n, base))
        except Exception:
            continue
        if s:
            out.append(s)
    return out


def _metavar_parents(lhs_pat: Any, rhs_pat: Any) -> dict:
    """Map each leaf metavar to its parent ops across both patterns."""
    mvs = sorted(
        set(_leaf_metavars(lhs_pat)) | set(_leaf_metavars(rhs_pat))
    )
    parents: dict[str, set] = {m: set() for m in mvs}
    for m, ops in _parents(lhs_pat).items():
        parents.setdefault(m, set()).update(ops)
    for m, ops in _parents(rhs_pat).items():
        parents.setdefault(m, set()).update(ops)
    return parents


class _LazySeq:
    """Indexed pull over a generator — materializes on demand only.

    ``get(i)`` returns the i-th yielded item or ``None`` once the
    generator is exhausted.  The diag interleaves probe element *a* of
    every group before any group's *a+1*, so the enumeration only ever
    materializes the prefix a cap reaches — the eager ``list`` form
    paid the whole ``(viewed x attr)`` product (plus a
    ``_lhs_out_shapes`` per base) before the first env yielded, which
    is what made deep-corner rules like ``gqa_absorb_repeat``
    unmeasurable.
    """

    __slots__ = ("_it", "_items", "done")

    def __init__(self, it: Any) -> None:
        self._it = iter(it)
        self._items: list = []
        self.done = False

    def get(self, i: int) -> Any:
        """Return the i-th item, or ``None`` past the end."""
        while len(self._items) <= i and not self.done:
            try:
                self._items.append(next(self._it))
            except StopIteration:
                self.done = True
        return self._items[i] if i < len(self._items) else None


def _diag_groups(groups: list) -> Iterable:
    """Yield ``groups[v][a]`` in increasing ``v + a`` (Cantor) order.

    The companion of :func:`_diag_product` for a *ragged* product of
    already-materialized lists: ``groups[v]`` is one viewed binding's
    list of attr-combination bases (the lengths differ — a rank-1
    operand's view table admits fewer axis pairs than a rank-2 one).
    Yielding by index-sum keeps the low-index corner of *every* group
    ahead of any group's deep tail, so a cap truncates a corner of the
    ``(viewed x attr)`` space rather than a whole viewed binding — the
    same fairness :func:`_binding_envs` applies one level up.

    Groups may be plain lists or :class:`_LazySeq` pull sequences; a
    probe past a group's end marks it done — the yielded order is
    identical either way.
    """
    lazies = [
        g if isinstance(g, _LazySeq) else _LazySeq(g) for g in groups
    ]
    n = len(lazies)
    if not n:
        return
    total = 0
    while not all(g.done for g in lazies):
        for v in range(min(total, n - 1), -1, -1):
            item = lazies[v].get(total - v)
            if item is not None:
                yield item
        total += 1


def _synth_bases_for(
    domains: list,
    viewed: dict,
    u_shapes: list,
    lhs_views: list,
    need_out: bool,
) -> Iterable:
    """Yield ``(base, u_shapes, out_shapes)`` for one viewed binding.

    The surviving attr merges of the binding's domain product, in
    ``_diag_product`` order; the LHS view output shapes are computed
    per surviving base — lazily, so a capped sweep never pays for the
    combos it did not reach.  ``out_shapes`` feeds only the free
    operand's derived bank, so ``need_out=False`` (no free
    metavariables) leaves it empty — instantiating seven view nodes
    per base for a dead value is measurable cost.
    """
    for combo in _diag_product([d[1] for d in domains]):
        base = _attr_merge(domains, combo, viewed)
        if base is None:
            continue
        out_shapes = (
            _lhs_out_shapes(lhs_views, base) if need_out else []
        )
        yield (base, u_shapes, out_shapes)


def _synth_bases(lhs_pat: Any, rhs_pat: Any) -> Iterable:
    """Yield ``(base, viewed_shapes, out_shapes)`` per viewed combo.

    The outer half of the enumeration: every leaf-metavar binding
    under a view op, merged with each shape-valid attribute
    assignment.  *viewed_shapes* / *out_shapes* are the operand and
    LHS-view output shapes the free operand's bank derives from.

    The two inner dimensions — the viewed binding and the attribute
    combination — are interleaved by index-sum (:func:`_diag_groups`),
    not nested.  A shape-major nesting serves *every* attribute
    combination of one viewed shape before the next shape opens, so a
    guard needing a different operand rank waits on a whole shape's
    attr domain: for the shipped ``sdpa_fold_*`` folds the leading
    rank-1 ``(4,)`` operand spends 24 guard-declined bases before the
    first rank-2 operand, and the accepted corner lands at base 27 of
    324.  The diagonal reaches that corner at base 13 (measured), and
    the same reordering carries to the deeper free dimension through
    :func:`_binding_envs`.
    """
    mvs = sorted(
        set(_leaf_metavars(lhs_pat)) | set(_leaf_metavars(rhs_pat))
    )
    parents = _metavar_parents(lhs_pat, rhs_pat)
    nodes = _view_nodes([lhs_pat, rhs_pat])
    lhs_views = [n for n in _view_nodes([lhs_pat]) if n.args]
    need_out = bool(
        [m for m in parents if not (parents.get(m, set()) & _VIEWISH)]
    )
    extra = _CHAIN_VIEWED_SHAPES if _has_chained_view(nodes) else ()
    groups: list = []
    for viewed in _viewed_bindings(mvs, parents, extra):
        domains = _attr_domains(nodes, viewed)
        if domains is None:
            continue
        u_shapes = [_operand_shape(t) for t in viewed.values()]
        group = _LazySeq(
            _synth_bases_for(
                domains, viewed, u_shapes, lhs_views, need_out
            )
        )
        # A binding whose product is empty (or all merge-conflicts)
        # contributes no group — the eager form's ``if bases:`` check;
        # probing the first element preserves the group indexing (and
        # therefore the yield order) exactly.
        if group.get(0) is None:
            continue
        groups.append(group)
    yield from _diag_groups(groups)


def _pull_bases(
    src: _LazySeq,
    entries: list[list],
    total: int,
    free: list[str],
    parents: dict,
    extra: tuple,
) -> int:
    """Pull every base due at round *total* into *entries*.

    Returns the count appended.  A base's free-combo generator is
    built lazily at pull time, so a capped sweep never materializes
    bases it cannot yield.
    """
    added = 0
    while not src.done and len(entries) <= total:
        got = src.get(len(entries))
        if got is None:
            break
        base, u_shapes, out_shapes = got
        derived = _derived_free_shapes(u_shapes, out_shapes)
        lists = [
            _leaf_bindings(m, parents[m], derived, extra) for m in free
        ]
        entries.append([base, _diag_product(lists), True])
        added += 1
    return added


def _binding_envs(lhs_pat: Any, rhs_pat: Any) -> Iterable[dict]:
    """Yield full binding dicts over the synthesized domain.

    The enumeration is Cantor-ordered across the whole nested
    product — ``(base, free-combo)`` index pairs in increasing
    index-sum — and diagonal inside each base's free domain.
    Ordering matters under a cap: exhausting one base's ~10⁴ free
    combos before the next base opens starves every later dimension
    (measured: a pattern with three free operands enumerates 360
    sites over a single rank-1 viewed binding — the guard's true
    region is never reached).  The diagonal keeps the cap honest:
    it truncates a *corner* of the space, not a whole dimension.
    """
    parents = _metavar_parents(lhs_pat, rhs_pat)
    free = sorted(
        m for m in parents if not (parents.get(m, set()) & _VIEWISH)
    )
    # entries[i] = [base, free-combo generator, alive] — the i-th
    # (viewed x attr) base.  Entries are pulled lazily: round `total`
    # only reaches entries with index <= total, so a capped sweep
    # never materializes bases it cannot yield — the eager form built
    # every base up front (``gqa_absorb_repeat``'s deep corner made
    # that unmeasurable).  The yield order is unchanged: entry *i*
    # still contributes its (total-i)-th free combo at round `total`.
    extra = (
        _CHAIN_VIEWED_SHAPES
        if _has_chained_view(_view_nodes([lhs_pat, rhs_pat]))
        else ()
    )
    entries: list[list] = []
    src = _LazySeq(_synth_bases(lhs_pat, rhs_pat))
    total = 0
    live = 0
    while live or not src.done:
        live += _pull_bases(src, entries, total, free, parents, extra)
        for i in range(min(total, len(entries) - 1), -1, -1):
            entry = entries[i]
            if not entry[2]:
                continue
            try:
                combo = next(entry[1])
            except StopIteration:
                entry[2] = False
                live -= 1
                continue
            yield {**entry[0], **dict(zip(free, combo, strict=True))}
        total += 1


def synthesize(
    lhs_pat: Any,
    rhs_pat: Any,
    *,
    limit: int = _MAX_INSTANCES,
    derive: Any = None,
) -> list[Instance]:
    """Enumerate satisfiable instances; evaluate both sides fp64.

    The enumeration is total over a small domain: every leaf
    metavariable over its binding bank (Vars over the shape bank,
    tuple-producers under ``getitem``, a scalar ``Const`` and derived
    shapes for the free operand), every attr metavariable over values
    valid for its kind — shape-valid axes for view ops, per-kind
    domains for other ops.  When *derive* is supplied it rides each
    binding with firing semantics exactly as ``sweep_real`` applies
    it — the computed ``$attr:`` bindings *override* the enumerated
    placeholder (a metavariable the rule derives is not a free
    choice), and a vetoed binding is not a realizable instance —
    skipped, not counted.  Duplicate instantiations (the same
    ``(lhs, rhs)`` pair from different bindings) are evaluated once.
    """
    out: list[Instance] = []
    seen: set[tuple] = set()
    for full in _binding_envs(lhs_pat, rhs_pat):
        inst = full
        if derive is not None:
            try:
                extra = derive(full)
            except Exception:
                continue
            if extra is None:
                continue
            inst = {**full, **extra}
        try:
            lhs_i = _term_instantiate(lhs_pat, inst)
            rhs_i = _term_instantiate(rhs_pat, inst)
        except Exception:
            continue
        # ``op_repr`` renders a bound Var by name only — the
        # shape lives in the binding, so the sig must carry it
        # or every shape assignment dedups to one instance.
        sig = (
            op_repr(lhs_i),
            op_repr(rhs_i),
            tuple(
                sorted(
                    (k, _bind_desc(v))
                    for k, v in inst.items()
                    if not k.startswith("$attr:")
                )
                + sorted(
                    (k, repr(v))
                    for k, v in inst.items()
                    if k.startswith("$attr:")
                )
            ),
        )
        if sig in seen:
            continue
        seen.add(sig)
        outcome, note = eval_instance(lhs_i, rhs_i)
        feats = _features(lhs_pat, rhs_pat, inst, lhs_i, rhs_i, outcome)
        out.append(
            Instance(
                origin="synth",
                outcome=outcome,
                lhs_repr=op_repr(lhs_i),
                rhs_repr=op_repr(rhs_i),
                binds=tuple(
                    sorted(
                        (k, _bind_desc(v))
                        for k, v in inst.items()
                        if not k.startswith("$attr:")
                    )
                    + sorted(
                        (k, v)
                        for k, v in inst.items()
                        if k.startswith("$attr:")
                    )
                ),
                feats=tuple(sorted(feats.items())),
                note=note,
            )
        )
        if len(out) >= limit:
            return out
    return out


def _bind_desc(term: Any) -> str:
    """Short description of a metavariable binding."""
    if isinstance(term, (Var, Param)):
        return f"Var{tuple(term.typ.shape)}"
    if isinstance(term, Const):
        return f"Const({term.value})"
    return f"{term.op}(·)"


# ---------------------------------------------------------------------------
#  Mechanical guard features
# ---------------------------------------------------------------------------


def _as_tensor(v: Any) -> Any:
    return v if isinstance(v, torch.Tensor) else None


def _view_call(node: Op, arg: Any) -> Any:
    """Apply *node*'s op binding to *arg* with the node's attrs."""
    from catopt_torch.torch_bridge import _IR_TO_TORCH

    fn = _IR_TO_TORCH.get(node.op)
    if fn is None:
        raise KeyError(node.op)
    return fn(arg, **dict(node.attrs))


def _bcast_to(t: Any, shape: tuple) -> Any:
    try:
        return torch.broadcast_to(t, shape)
    except Exception:
        return None


def _feats_id(u: Any, g_u: Any, v: Any) -> dict[str, bool]:
    """``f(g(u),v) == f(u,v)`` pairing/out-shape features."""
    feats: dict[str, bool] = {}
    if not all(isinstance(t, torch.Tensor) for t in (u, g_u, v)):
        return feats
    try:
        g_l = torch.broadcast_shapes(g_u.shape, v.shape)
        g_r = torch.broadcast_shapes(u.shape, v.shape)
        g_all = torch.broadcast_shapes(g_l, g_r, u.shape)
    except Exception:
        return feats
    feats["id:out_shape_eq"] = tuple(g_l) == tuple(g_r)
    ue, ge = _bcast_to(u, g_all), _bcast_to(g_u, g_all)
    if ue is not None and ge is not None:
        feats["id:same_pairing"] = bool(torch.equal(ue, ge))
    return feats


def _feats_w(node_i: Op, u: Any, v: Any, g_out: Any) -> dict[str, bool]:
    """Check whether ``v`` commutes with the view in ``f(g(u),v) = g(f(u,v))``.

    *node_i* is the instantiated RHS view node (concrete attrs); the
    feature applies its op to the free operand broadcast to the
    *unviewed* grid and compares with the broadcast to the *viewed*
    grid — the naturality precondition, checked on tensors.
    """
    feats: dict[str, bool] = {}
    if not all(isinstance(t, torch.Tensor) for t in (u, v, g_out)):
        return feats
    try:
        g_u = torch.broadcast_shapes(u.shape, v.shape)
        g_l = torch.broadcast_shapes(g_out.shape, v.shape)
    except Exception:
        return feats
    ve = _bcast_to(v, g_u)
    if ve is None:
        return feats
    try:
        moved = _view_call(node_i, ve)
    except Exception:
        return feats
    vl = _bcast_to(v, g_l)
    ok = (
        vl is not None
        and isinstance(moved, torch.Tensor)
        and tuple(moved.shape) == tuple(vl.shape)
        and bool(torch.equal(moved, vl))
    )
    feats["w:v_commutes_view"] = ok
    return feats


def _features(
    lhs_pat: Any,
    rhs_pat: Any,
    subst: dict,
    lhs_i: Any,
    rhs_i: Any,
    outcome: str,
) -> dict[str, Any]:
    """Compute the mechanical guard features for one instance.

    Two structural families are recognized:

    * ``_id`` — the RHS drops the view (``f(g(u),v) -> f(u,v)``):
      ``id:same_pairing`` asks whether ``g(u)`` and ``u`` broadcast to
      the *same* elements on the common grid, ``id:out_shape_eq``
      whether the two evaluable output shapes coincide.  Both are
      necessary for the strip to be an equality.
    * ``_w`` — the RHS wraps the pointwise result in the view
      (``f(g(u),v) -> g(f(u,v))``): ``w:v_commutes_view`` asks
      whether the free operand broadcast to the *unviewed* grid and
      then pushed through the view equals its broadcast to the
      *viewed* grid — the naturality precondition, checked
      mechanically on tensors.
    """
    feats: dict[str, Any] = {}
    parents = _parents(lhs_pat)
    mvs = sorted(set(_leaf_metavars(lhs_pat)))
    env = _env_for(lhs_i, rhs_i)
    if env is None:
        return feats
    vals: dict[str, Any] = {}
    for m in mvs:
        ok, v = _eval(_term_instantiate(m, subst), env)
        vals[m] = v if ok else None
        if m in ("U", "V"):
            feats[f"{m.lower()}_shape"] = (
                tuple(v.shape)
                if isinstance(v, torch.Tensor)
                else (
                    f"tuple:{len(v)}" if isinstance(v, tuple) else "?"
                )
            )
    viewed = [m for m in mvs if parents.get(m, set()) & _VIEWISH]
    free = [m for m in mvs if not (parents.get(m, set()) & _VIEWISH)]
    u = vals.get(viewed[0]) if viewed else None
    v = vals.get(free[0]) if free else None
    feats["u_tuple"] = isinstance(u, tuple)
    feats["v_scalar"] = isinstance(v, torch.Tensor) and v.dim() == 0
    feats["v_uniform"] = isinstance(v, torch.Tensor) and all(
        d == 1 for d in v.shape
    )
    vnodes = [
        n for n in _view_nodes([lhs_pat]) if n.args and n.op in _VIEWISH
    ]
    if not vnodes or u is None:
        return feats
    node = vnodes[0]
    ok, g_out = _eval(_term_instantiate(node, subst), env)
    g_out = g_out if ok else None

    def _battr(key: str) -> Any:
        mv = node.attrs.get(key)
        if isinstance(mv, str):
            return subst.get(f"$attr:{mv}")
        return mv

    if isinstance(u, torch.Tensor):
        # No-op / covering features — the degenerate region where a
        # strip (``_id``) candidate can hold.
        if node.op in ("transpose", "t"):
            d0, d1 = _battr("dim0"), _battr("dim1")
            if isinstance(d0, int) and isinstance(d1, int):
                feats["tr:noop"] = (d0 % u.dim()) == (d1 % u.dim())
        elif node.op in ("reshape", "view"):
            s = _battr("shape")
            feats["rs:noop"] = isinstance(s, (tuple, list)) and tuple(
                s
            ) == tuple(u.shape)
        elif node.op == "slice":
            d = _battr("dim")
            s0, e = _battr("start"), _battr("end")
            if isinstance(d, int):
                ext = u.shape[d % u.dim()]
                feats["sl:full"] = (s0 in (0, None)) and (
                    e is None or (isinstance(e, int) and e >= ext)
                )
        elif node.op == "chunk":
            feats["ck:single"] = _battr("chunks") == 1
    if (
        node.op == "unsqueeze"
        and isinstance(u, torch.Tensor)
        and isinstance(v, torch.Tensor)
    ):
        d = subst.get(
            f"$attr:{node.attrs.get('dim')}", node.attrs.get("dim")
        )
        if isinstance(d, int):
            nd = d % (u.dim() + 1)
            feats["unsq:d_in_pad"] = nd < v.dim() - u.dim()
    if not isinstance(rhs_pat, Op) or g_out is None:
        return feats
    if isinstance(lhs_pat, Op) and rhs_pat.op == lhs_pat.op:
        feats.update(_feats_id(u, g_out, v))
    elif rhs_pat.op == node.op:
        rhs_view_i = _term_instantiate(rhs_pat, subst)
        if isinstance(rhs_view_i, Op):
            feats.update(_feats_w(rhs_view_i, u, v, g_out))
    return feats


# ---------------------------------------------------------------------------
#  Real-match sweep
# ---------------------------------------------------------------------------


def sweep_real(
    name: str,
    lhs_pat: Any,
    rhs_pat: Any,
    matches: list[Any],
    *,
    check: Any = None,
    derive: Any = None,
) -> list[Instance]:
    """Evaluate *every* real match, not just the first viable one.

    Applies the proposal's ``check``/``derive`` hooks to each match's
    substitution exactly as a firing would, then evaluates the
    instantiated pair tri-state.  The bound operand's *kind* (tensor
    vs tuple value) is recorded — ``getitem``'s two semantics are the
    difference between an ill-typed RHS and a false one.
    """
    out: list[Instance] = []
    seen: set[tuple] = set()
    for sub in matches:
        subst = _term_match(lhs_pat, sub)
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
            rhs_i = _term_instantiate(rhs_pat, inst)
        except Exception:
            continue
        sig = (op_repr(sub), op_repr(rhs_i))
        if sig in seen:
            continue
        seen.add(sig)
        outcome, note = eval_instance(sub, rhs_i)
        u_kind = ""
        u_term = subst.get("U")
        if u_term is not None:
            env = _env_for(sub, rhs_i)
            if env is not None:
                ok, uv = _eval(u_term, env)
                if ok:
                    u_kind = (
                        "tuple" if isinstance(uv, tuple) else "tensor"
                    )
        out.append(
            Instance(
                origin="real",
                outcome=outcome,
                lhs_repr=op_repr(sub),
                rhs_repr=op_repr(rhs_i),
                binds=tuple(
                    sorted(
                        (k, _bind_desc(v))
                        for k, v in subst.items()
                        if not k.startswith("$attr:")
                    )
                ),
                feats=(
                    ("u_kind", u_kind),
                    *sorted(
                        (f"$attr:{k[6:]}", v)
                        for k, v in subst.items()
                        if k.startswith("$attr:")
                    ),
                ),
                note=note,
            )
        )
    return out


# ---------------------------------------------------------------------------
#  Verdict
# ---------------------------------------------------------------------------


def _separating_feature(instances: list[Instance]) -> str:
    """Find a boolean feature (or pair) separating equal from unequal.

    Among instances where *both* sides evaluated, an absent feature
    counts as ``False`` — a missing pairing feature means there was
    no common broadcast grid, which is itself a failed precondition.
    Returns the feature name (or ``"a ∧ b"`` for a separating
    conjunction) used to *name* the guard a conditional law would
    carry — the description, not an admitted ``cond``.
    """
    both = [i for i in instances if i.outcome in ("equal", "unequal")]

    def key(i: Instance) -> str:
        return i.lhs_repr + i.rhs_repr + repr(i.binds)

    eq = {key(i) for i in both if i.outcome == "equal"}
    feats_of = {key(i): dict(i.feats) for i in both}
    keys = sorted({k for i in both for k, _ in i.feats})

    def sep(pred: Any, label: str) -> str | None:
        pos = [k for k, f in feats_of.items() if pred(f)]
        if pos and all(k in eq for k in pos) and len(pos) == len(eq):
            return label
        return None

    for k in keys:
        hit = sep(lambda f, k=k: f.get(k) is True, k)
        if hit:
            return hit
    for k1, k2 in itertools.combinations(keys, 2):
        hit = sep(
            lambda f, a=k1, b=k2: f.get(a) is True and f.get(b) is True,
            f"{k1} ∧ {k2}",
        )
        if hit:
            return hit
    return ""


def verify_view_candidate(
    name: str,
    lhs_pat: Any,
    rhs_pat: Any,
    matches: list[Any],
    *,
    check: Any = None,
    derive: Any = None,
    synth_limit: int = _MAX_INSTANCES,
) -> ViewVerdict:
    """Resolve a view/index candidate to a verdict.

    ``verdict`` is one of:

    * ``true`` — every both-sides-evaluable instance (real sweep +
      synthesized) agreed;
    * ``conditional`` — equal and unequal instances both exist; the
      separating shape/attr condition is named in ``guard`` when a
      mechanical feature splits them cleanly;
    * ``false`` — both-typed instances exist and none agree;
    * ``ill-formed`` — the LHS evaluates somewhere but the RHS never
      does: the rewrite target cannot denote (e.g. a pointwise op
      over a tuple operand);
    * ``unproven`` — nothing evaluated at all.
    """
    torch.manual_seed(20240513)
    real = sweep_real(
        name, lhs_pat, rhs_pat, matches, check=check, derive=derive
    )
    synth = synthesize(
        lhs_pat, rhs_pat, limit=synth_limit, derive=derive
    )
    insts = [*real, *synth]
    v = ViewVerdict(
        name=name,
        n_real=len(real),
        n_synth=len(synth),
        instances=tuple(insts),
    )
    for i in real:
        if i.outcome == "equal":
            v.real_equal += 1
        elif i.outcome == "unequal":
            v.real_unequal += 1
        elif i.outcome == "rhs-err":
            v.real_rhs_err += 1
        elif i.outcome in ("lhs-err", "both-err", "env-err"):
            v.real_lhs_err += 1
    for i in synth:
        if i.outcome == "equal":
            v.synth_equal += 1
            if not v.witness:
                v.witness = i.lhs_repr
        elif i.outcome == "unequal":
            v.synth_unequal += 1
            if not v.counterexample:
                v.counterexample = i.lhs_repr
        elif i.outcome == "rhs-err":
            v.synth_rhs_err += 1
    for i in real:
        if i.outcome == "equal" and not v.witness:
            v.witness = i.lhs_repr
        if i.outcome == "unequal" and not v.counterexample:
            v.counterexample = i.lhs_repr

    both_ok = [i for i in insts if i.outcome in ("equal", "unequal")]
    eq = [i for i in both_ok if i.outcome == "equal"]
    neq = [i for i in both_ok if i.outcome == "unequal"]
    lhs_ok_any = any(
        i.outcome in ("equal", "unequal", "rhs-err") for i in insts
    )
    if not insts or not lhs_ok_any:
        v.verdict = "unproven"
        v.note = "no evaluable instance"
    elif not both_ok:
        v.verdict = "ill-formed"
        v.note = (
            "RHS never evaluates where the LHS does — the rewrite "
            "target does not denote"
        )
    elif not neq and not any(i.outcome == "rhs-err" for i in insts):
        v.verdict = "true"
        v.note = f"equal on all {len(eq)} evaluable instances"
    elif not eq:
        v.verdict = "false"
        v.note = f"no agreeing instance in {len(both_ok)} evaluable"
    else:
        v.verdict = "conditional"
        sep = _separating_feature(insts)
        rhs_err = sum(1 for i in insts if i.outcome == "rhs-err")
        if not neq:
            v.guard = (
                "RHS well-typedness (the rewrite mints an ill-typed "
                "member where the guard fails)"
            )
        else:
            v.guard = sep or "no single mechanical feature separates"
        v.note = (
            f"{len(eq)} equal / {len(neq)} unequal evaluable "
            f"instances; {rhs_err} ill-typed-RHS"
        )
    return v


# ---------------------------------------------------------------------------
#  Driver — resolve every unresolved view candidate in the pipeline
# ---------------------------------------------------------------------------


def _run() -> list[ViewVerdict]:
    """Run the oracle over the pipeline's current proposal set."""
    from catopt_discovery import intake as li
    from catopt_discovery import pipeline as pl
    from catopt_discovery.census import run_census
    from catopt_discovery.impact import _bench_cases, model_cases
    from catopt_discovery.shape_proposal import Schema, real_matches

    census = run_census(pl._CENSUS_TOP)
    census_op = {
        (e["op"], tuple(e["children"])): e["count"]
        for e in census["op_tuples"]
    }
    bench, _be = _bench_cases()
    models, _me = model_cases()
    intake = li.load_cases()
    real_terms = [c.term for c in [*bench, *models, *intake]]
    proposals = pl.propose(census_op, real_terms, "derived")
    verdicts = []
    for p in proposals:
        has_view = any(
            n.op in _VIEWISH for n in _view_nodes([p.lhs, p.rhs])
        ) or any(
            isinstance(t, Op) and t.op in _VIEWISH
            for t in pl._iter_subterms(p.lhs)
        )
        if not has_view:
            continue
        schema = Schema(p.name, p.lhs, p.rhs)
        ms = real_matches(real_terms, schema)
        verdicts.append(
            verify_view_candidate(
                p.name,
                p.lhs,
                p.rhs,
                ms,
                check=p.check,
                derive=p.derive,
            )
        )
    return verdicts


def _table(verdicts: list[ViewVerdict]) -> str:
    head = (
        f"{'candidate':<32} {'verdict':<12} {'real eq/ne/rerr':<16} "
        f"{'synth eq/ne/rerr':<17} guard"
    )
    lines = [head, "-" * len(head)]
    for v in verdicts:
        lines.append(
            f"{v.name:<32} {v.verdict:<12} "
            f"{v.real_equal}/{v.real_unequal}/{v.real_rhs_err:<8} "
            f"{v.synth_equal}/{v.synth_unequal}/{v.synth_rhs_err:<9} "
            f"{v.guard or '-'}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Resolve every view/index candidate; print (or dump) verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)
    verdicts = _run()
    print(  # stdout-compat
        "== law_view_oracle — view/index candidate resolution =="
    )
    print(_table(verdicts))  # stdout-compat
    if args.json:
        payload = [
            {
                "name": v.name,
                "verdict": v.verdict,
                "guard": v.guard,
                "note": v.note,
                "real": {
                    "n": v.n_real,
                    "equal": v.real_equal,
                    "unequal": v.real_unequal,
                    "rhs_err": v.real_rhs_err,
                    "lhs_err": v.real_lhs_err,
                },
                "synth": {
                    "n": v.n_synth,
                    "equal": v.synth_equal,
                    "unequal": v.synth_unequal,
                    "rhs_err": v.synth_rhs_err,
                },
                "witness": v.witness,
                "counterexample": v.counterexample,
            }
            for v in verdicts
        ]
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
