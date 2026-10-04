"""View/index oracle — honest instantiation for view-family candidates.

The pipeline's numeric oracle (``law_proposal._numeric_true``) proves a
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

The verdict feeds ``tools/law_pipeline.py``'s ``measure`` step
(``--no-view-oracle`` disables).  A ``conditional`` candidate is *not*
auto-admitted: it is evidence for a guarded law — the interesting
outcome — reported for review.

Run::

    .venv/bin/python tools/law_view_oracle.py
    .venv/bin/python tools/law_view_oracle.py --json /tmp/view_oracle.json

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from catopt_core.egraph.terms import _term_instantiate, _term_match
from catopt_core.ir import Const, Op, Param, TensorType, Var, op_repr

# Sibling tools own the corpus, the eval backend and the comparator;
# reuse them, never duplicate.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import law_proposal as lp

__all__ = [
    "Instance",
    "ViewVerdict",
    "eval_instance",
    "sweep_real",
    "synthesize",
    "verify_view_candidate",
]

#: Numeric-comparison tolerance (fp64), shared with ``law_proposal``.
_TOL = 1e-6

#: Cap on synthesized instances per candidate.
_MAX_INSTANCES = 360

#: View/index ops this oracle knows how to attribute-instantiate.
#: Anything outside the table leaves the attr metavariables unbound —
#: the instance is skipped, honestly.
_VIEWISH = frozenset(
    {
        "getitem",
        "select",
        "slice",
        "unsqueeze",
        "squeeze",
        "transpose",
        "reshape",
        "view",
        "expand",
        "broadcast_to",
        "chunk",
        "split",
        "narrow",
        "permute",
        "unbind",
        "movedim",
        "flatten",
    }
)


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
    out: list[tuple] = []
    for c in cands:
        if (
            all(isinstance(d, int) and d > 0 for d in c)
            and c not in out
            and _shape_numel(c) == n
        ):
            out.append(c)
    return out[:7]


def _attr_options(
    op: str, keys: tuple, shape: Any
) -> list[dict] | None:
    """Return concrete attr dicts for *op*'s metavar'd keys, or ``None``.

    *shape* is the operand's inferred shape (``tuple`` or ``()``);
    tuple-producing operands report their element shape.  ``None``
    means the oracle cannot honestly instantiate this op — the
    instance is skipped, not guessed.
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
            outs = [(2, *sh)] if r else [(2,)]
            ones = [i for i, d in enumerate(sh) if d == 1]
            if ones:
                grown = list(sh)
                grown[ones[0]] = 3
                outs.append(tuple(grown))
            return [{"shape": s} for s in outs]
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
            return None


# ---------------------------------------------------------------------------
#  Binding domains for leaf metavariables
# ---------------------------------------------------------------------------


#: Shapes offered to a metavariable that sits under a view op.
_VIEWED_SHAPES: tuple = (
    (4,),
    (2, 3),
    (3, 4),
    (2, 2),
    (2, 3, 4),
    (2, 3, 1),
)


def _tuple_sources(mv: str) -> list[Any]:
    """Tuple-producing terms for a metavar under ``getitem``.

    The corpus's real ``getitem`` matches pick elements out of
    ``topk`` / ``var_mean`` / ``cummax`` — a bare ``Var`` only covers
    the dim-0 tensor index.  Both meanings are instantiated; the
    eval decides.
    """
    w = Var(f"{mv}@t", TensorType((2, 4)))
    return [
        Op.make("topk", w, k=2),
        Op.make("var_mean", w, dim=(-1,), correction=0, keepdim=True),
        Op.make("cummax", w, dim=0),
    ]


def _leaf_bindings(
    mv: str, parents: set[str], derived: Iterable[tuple]
) -> list[Any]:
    """Candidate bindings for one leaf metavariable.

    ``derived`` is the extra shape list computed from *other* leaves'
    choices (filled in by the enumerator for the free operand).
    """
    out: list[Any] = []
    if "getitem" in parents:
        out.extend(_tuple_sources(mv))
    for s in itertools.chain(_VIEWED_SHAPES, derived):
        out.append(Var(mv, TensorType(tuple(s))))
    if not parents or all(p not in _VIEWISH for p in parents):
        # A free operand may bind a literal scalar — the corpus does.
        out.append(Const(0.5))
        out.insert(0, Var(mv, TensorType(())))
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


def _viewed_bindings(mvs: list[str], parents: dict) -> Iterable[dict]:
    """Yield binding dicts for metavariables under a view node."""
    viewed = [m for m in mvs if parents.get(m, set()) & _VIEWISH]
    lists = [_leaf_bindings(m, parents[m], ()) for m in viewed]
    for combo in itertools.product(*lists):
        yield dict(zip(viewed, combo, strict=True))


def _derived_free_shapes(
    u_shapes: list[tuple], out_shapes: list[tuple]
) -> list[tuple]:
    """Return the derived shape bank for a free (non-viewed) operand.

    The candidates that probe the boundary region the naturality
    candidates hinge on: the viewed operand's own shape, the view's
    output shape, every single-axis-1 insertion of each (the
    broadcast-pad cases), a leading pad, and a genuinely mismatched
    shape.
    """
    derived: list[tuple] = [(), (4,), (7, 7)]
    for sh in [*u_shapes, *out_shapes]:
        if not sh or sh in derived:
            continue
        derived.append(sh)
        for i in range(len(sh) + 1):
            ins = (*sh[:i], 1, *sh[i:])
            if ins not in derived:
                derived.append(ins)
    return derived


def _attr_domains(
    nodes: list[Op], subst: dict
) -> list[tuple[Op, list[dict]]] | None:
    """Return ``(node, option dicts)`` per distinct attr-metavar group.

    Each group is one view node's set of attr metavariable names; the
    option dicts assign concrete values to the node's *attr keys*
    (shared metavariable names across LHS/RHS resolve once).
    ``None`` marks a node the oracle cannot instantiate honestly.
    """
    groups: dict[str, tuple[Op, list[str]]] = {}
    order: list[str] = []
    for node in nodes:
        names = tuple(
            sorted(v for v in node.attrs.values() if isinstance(v, str))
        )
        if not names:
            continue
        key = "|".join(names)
        if key not in groups:
            groups[key] = (node, list(node.attrs))
            order.append(key)
    out = []
    for key in order:
        node, _keys = groups[key]
        try:
            bound = (
                _term_instantiate(node.args[0], subst)
                if node.args
                else None
            )
        except Exception:
            bound = None
        shape = _operand_shape(bound) if bound is not None else ()
        # For getitem over a tuple source the index domain is the
        # tuple arity, not an axis extent — options stay {0,1}.
        opts = _attr_options(node.op, tuple(node.attrs), shape)
        if not opts:
            return None
        out.append((node, opts))
    return out


def synthesize(
    lhs_pat: Any,
    rhs_pat: Any,
    *,
    limit: int = _MAX_INSTANCES,
) -> list[Instance]:
    """Enumerate satisfiable instances; evaluate both sides fp64.

    The enumeration is total over a small domain: every leaf
    metavariable over its binding bank (Vars over the shape bank,
    tuple-producers under ``getitem``, a scalar ``Const`` and derived
    shapes for the free operand), every view node over attribute
    values valid for the bound operand's shape.  Duplicate
    instantiations (the same ``(lhs, rhs)`` pair from different
    bindings) are evaluated once.
    """
    mvs = sorted(
        set(_leaf_metavars(lhs_pat)) | set(_leaf_metavars(rhs_pat))
    )
    parents = {m: set() for m in mvs}
    for m, ops in _parents(lhs_pat).items():
        parents.setdefault(m, set()).update(ops)
    for m, ops in _parents(rhs_pat).items():
        parents.setdefault(m, set()).update(ops)
    nodes = _view_nodes([lhs_pat, rhs_pat])
    lhs_views = [n for n in _view_nodes([lhs_pat]) if n.args]
    free = [m for m in mvs if not (parents.get(m, set()) & _VIEWISH)]

    out: list[Instance] = []
    seen: set[tuple] = set()
    for viewed in _viewed_bindings(mvs, parents):
        domains = _attr_domains(nodes, viewed)
        if domains is None:
            continue
        for attr_combo in itertools.product(*[d[1] for d in domains]):
            base = dict(viewed)
            skip = False
            for (node, _opts), vals in zip(
                domains, attr_combo, strict=True
            ):
                for k, mv_name in node.attrs.items():
                    if isinstance(mv_name, str):
                        ak = f"$attr:{mv_name}"
                        if ak in base and base[ak] != vals.get(k):
                            skip = True
                            break
                        if k in vals:
                            base[ak] = vals[k]
                if skip:
                    break
            if skip:
                continue
            # The free operand's bank derives from this exact
            # instantiation: operand shapes and each LHS view node's
            # output shape.
            u_shapes = [_operand_shape(t) for t in viewed.values()]
            out_shapes: list[tuple] = []
            for n in lhs_views:
                try:
                    s = _operand_shape(_term_instantiate(n, base))
                except Exception:
                    continue
                if s:
                    out_shapes.append(s)
            derived = _derived_free_shapes(u_shapes, out_shapes)
            free_lists = [
                _leaf_bindings(m, set(), derived) for m in free
            ]
            for fcombo in itertools.product(*free_lists):
                full = {**base, **dict(zip(free, fcombo, strict=True))}
                try:
                    lhs_i = _term_instantiate(lhs_pat, full)
                    rhs_i = _term_instantiate(rhs_pat, full)
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
                            for k, v in full.items()
                            if not k.startswith("$attr:")
                        )
                        + sorted(
                            (k, repr(v))
                            for k, v in full.items()
                            if k.startswith("$attr:")
                        )
                    ),
                )
                if sig in seen:
                    continue
                seen.add(sig)
                outcome, note = eval_instance(lhs_i, rhs_i)
                feats = _features(
                    lhs_pat, rhs_pat, full, lhs_i, rhs_i, outcome
                )
                out.append(
                    Instance(
                        origin="synth",
                        outcome=outcome,
                        lhs_repr=op_repr(lhs_i),
                        rhs_repr=op_repr(rhs_i),
                        binds=tuple(
                            sorted(
                                (k, _bind_desc(v))
                                for k, v in full.items()
                                if not k.startswith("$attr:")
                            )
                            + sorted(
                                (k, v)
                                for k, v in full.items()
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
    synth = synthesize(lhs_pat, rhs_pat, limit=synth_limit)
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
    import law_intake as li
    import law_pipeline as pl
    from law_impact import _bench_cases, model_cases
    from law_shape_census import run_census
    from law_shape_proposal import Schema, real_matches

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
    print("== law_view_oracle — view/index candidate resolution ==")
    print(_table(verdicts))
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
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
