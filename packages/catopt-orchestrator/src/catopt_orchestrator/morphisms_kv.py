"""KV latent sharing — MLA-style common right-factors.

Plan-0011 follow-on.  When ``N`` blocks project the *same* activation
to K/V (``k_i = x·W_i``), the weights may factor through one latent:
``W_i = D_i·U`` gives ``k_i = (x·Uᵀ)·D_i`` — the shared GEMM
``C = matmul(x, U_t)`` is computed once and each block recovers its
projection with ``linear(C, D_i)``.  Compute drops only when the
common right-factor's rank ``r`` is genuinely below the summed
projection dims; the cost gate and the fp64 pair verify decide.

Applicability, honestly:

* **shared input** — the family's blocks must consume literally the
  same tensor (captured ``in_obj`` object identity); chain and
  residual-stream blocks each see a *different* stream value, so no
  shared latent exists there (the residual-stream generalisation —
  sharing only the embeddings component of the distributed form —
  is documented future work, not a silent assumption);
* **additive outputs** — the fused first slot delivers
  ``Σ b_i(x)`` and the consumed slots become exact zeros, so the
  members' outputs must be additive contributions to one consumed
  value — the model return, the shared input plus it, or another
  block's input — witnessed on the capture AND the perturbed probe;
* **weights must factor** — the stacked effective weight matrices
  need a common row space of rank ``r`` below the savings
  threshold.  Built low-rank families (constructed factorisations,
  weight-sharing schemes, pruned ranks, MLA-style checkpoints)
  factor exactly; an inexact factor inside ``factor_tol`` relative
  Frobenius is accepted and witnessed with its error bound;
  anything beyond declines.  Random full-rank weights factor only
  trivially at ``r = d_in`` and the cost gate drops the match.

* **bounded mode** (plan 0012, opt-in ``budget``) — real models
  carry the shared structure *approximately* (xKV: aligned
  cross-layer singular vectors).  With a budget the common basis is
  *truncated*: directions whose summed per-site Frobenius residual
  fits the budget are dropped, the measured residual becomes the
  certified ``error_bound`` on the witness, the rewrite fires only
  when ``residual ≤ budget``, and the pair verify gates at the
  propagated bound — certified-bound replaces certified-exact,
  never silent: the graft record and match stats carry the bound.

The same machinery applies *inside* one block: a block whose K and V
projections share a data operand and a right-factor is the per-layer
MLA fold — no wiring evidence needed.

Deliberately out of scope: ``matmul`` with the weight on the *left*
(a different latent spelling), non-``Param`` weight expressions, and
fused ``qkv`` weights whose k/v sections do not follow the
q,k,v-equal-thirds convention.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from catopt_core.cost import dag_cost
from catopt_core.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
    op_repr,
)
from catopt_core.laws.pairing import _exact_equal, _is_tensor
from catopt_core.ports import CostFn, Sink
from catopt_core.typing import _shape_of, has_var_leaf

import catopt_orchestrator.morphisms as M

__all__ = ["KVLatentShare"]

log = logging.getLogger("catopt_orchestrator.morphisms_kv")


def _usable_member(graph: Any, name: str) -> bool:
    """Check a record can join a family — lifted, sig-consistent.

    The member's signature must exist (lifted), describe the
    record's *current* IR arity (a sig/IR arity mismatch means the
    signature is stale for this record), and pass the multi-input
    lift verdict — one activation input plus pass-through context.
    """
    ir = graph.record(name).ir
    sig = graph.sig(name)
    return (
        ir is not None
        and sig is not None
        and len(sig.inputs) == len(ir.inputs)
        and M._sig_liftable(sig) is None
    )


#: Weight-name tokens marking a standalone K or V projection.
_KV_TOKENS: tuple[str, ...] = ("k_proj", "v_proj")

#: Weight-name tokens marking a fused qkv projection — its k/v
#: row-sections count as per-block member weights under the
#: q,k,v-equal-thirds convention (the SDPA layout).
_QKV_TOKENS: tuple[str, ...] = ("qkv",)


@dataclass(frozen=True)
class _KVSite:
    """One projection node joining a shared-latent factorisation.

    ``member`` is the block owning ``node``; ``data`` is the operand
    the shared latent is computed on (one common term across the
    family); ``weight`` is the ``Param`` leaf carrying the weight
    values; ``orient`` is the projection spelling — ``"linear"`` or
    ``"qkv"`` (weight rows live in input space) / ``"matmul_r"`` (a
    right-side matmul weight — its *columns* live in input space);
    ``span`` is the ``(start, length)`` row-section for ``"qkv"``.
    """

    member: str
    node: Any
    data: Any
    weight: Param
    orient: str
    span: tuple[int, int] | None = None


def _kv_named(name: str, tokens: tuple[str, ...]) -> bool:
    """Check a weight name carries a KV-projection token."""
    return any(tok in name for tok in tokens)


def _qkv_sites(n: Any, member: str, d: Any, w: Param) -> list[_KVSite]:
    """Two section sites for a fused-qkv weight — or none.

    Only the q,k,v-equal-thirds layout is a candidate (the SDPA
    convention); any other row count declines.
    """
    shp = w.typ.shape
    ok = (
        isinstance(shp, tuple)
        and len(shp) == 2
        and isinstance(shp[0], int)
        and shp[0] > 0
        and shp[0] % 3 == 0
    )
    if not ok:
        return []
    sec = shp[0] // 3
    return [
        _KVSite(member, n, d, w, "qkv", (sec, sec)),
        _KVSite(member, n, d, w, "qkv", (2 * sec, sec)),
    ]


def _linear_sites(
    n: Any,
    member: str,
    tokens: tuple[str, ...],
    qkv_tokens: tuple[str, ...],
) -> list[_KVSite]:
    """Sites for one ``linear`` node — standalone or fused qkv.

    qkv tokens win: ``"qkv_proj"`` also contains ``"v_proj"``.
    """
    if len(n.args) < 2:
        return []
    d, w = n.args[0], n.args[1]
    if not isinstance(w, Param) or not has_var_leaf(d):
        return []
    if _kv_named(w.name, qkv_tokens):
        return _qkv_sites(n, member, d, w)
    if _kv_named(w.name, tokens):
        return [_KVSite(member, n, d, w, "linear")]
    return []


def _matmul_site(
    n: Any, member: str, tokens: tuple[str, ...]
) -> _KVSite | None:
    """One right-matmul site — ``matmul(d, W)`` with a named weight."""
    if len(n.args) != 2:
        return None
    a, b = n.args
    if (
        isinstance(b, Param)
        and has_var_leaf(a)
        and _kv_named(b.name, tokens)
    ):
        return _KVSite(member, n, a, b, "matmul_r")
    return None


def _kv_proj_sites(
    body: Any,
    member: str,
    tokens: tuple[str, ...],
    qkv_tokens: tuple[str, ...],
) -> list[_KVSite]:
    """Projection nodes whose ``Param`` weight carries a KV token.

    Scans ``body`` for ``linear(d, W[, b])`` and ``matmul(d, W)``
    projections with a var-carrying data operand and a named weight —
    everything else (compound weight terms, left-weight matmuls,
    non-KV names) is not a site.
    """
    sites: list[_KVSite] = []
    for n in M._iter_ops(body):
        if n.op == "linear":
            sites.extend(_linear_sites(n, member, tokens, qkv_tokens))
        elif n.op == "matmul":
            s = _matmul_site(n, member, tokens)
            if s is not None:
                sites.append(s)
    return sites


def _latent_groups(
    bodies: dict[str, Any],
    tokens: tuple[str, ...],
    qkv_tokens: tuple[str, ...],
) -> dict[Any, list[_KVSite]]:
    """Group the bodies' KV sites by their shared data operand."""
    groups: dict[Any, list[_KVSite]] = {}
    for name, body in bodies.items():
        for s in _kv_proj_sites(body, name, tokens, qkv_tokens):
            groups.setdefault(s.data, []).append(s)
    return {d: ss for d, ss in groups.items() if len(ss) >= 2}


def _input_families(graph: M.MorphismGraph) -> dict[int, list[str]]:
    """Blocks sharing one input object — the same-``x`` families.

    Keyed by ``id(in_obj)`` — the live object stays referenced by the
    record, so the id cannot be reused; blocks that never ran,
    exported opaquely, or were called more than once cannot join a
    consumable family.
    """
    groups: dict[int, list[str]] = {}
    for n in graph.nodes:
        if n.opaque:
            continue
        r = graph.record(n.name)
        if r.in_obj is None or r.calls != 1:
            continue
        groups.setdefault(id(r.in_obj), []).append(n.name)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _vals_close(
    a: Any, b: Any, *, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Exact-or-near value equality — ``allclose`` without the backend.

    Same convention as the pairing pass's duck-typed tensor equality;
    the tolerant branch covers the reassociation wobble between
    ``(x + b₁) + b₂`` and ``x + (b₁ + b₂)`` — a handful of ulps.
    """
    if a is b:
        return True
    if not (_is_tensor(a) and _is_tensor(b)):
        return False
    if tuple(a.shape) != tuple(b.shape):
        return False
    if _exact_equal(a, b):
        return True
    return bool((abs(a - b) <= atol + rtol * abs(b)).all())


def _sum_vals(vals: Iterable[Any]) -> Any:
    """Left-fold ``+`` over tensor-like values; ``None`` when empty."""
    acc = None
    for v in vals:
        if _is_tensor(v):
            acc = v if acc is None else acc + v
    return acc


def _fanout_clean(
    graph: M.MorphismGraph, members: tuple[str, ...]
) -> bool:
    """Check no member output object is consumed outside the sum.

    A member whose output object is another block's input — or is
    the model's own return — keeps a live non-additive consumer; the
    exact-zero filler would corrupt that path, so the family fails
    closed.
    """
    mem = set(members)
    for n in members:
        oo = graph.record(n).out_obj
        if oo is None:
            continue
        for other in graph.nodes:
            if (
                other.name not in mem
                and graph.record(other.name).in_obj is oo
            ):
                return False
        if any(oo is mo for mo in graph._model_out_objs):
            return False
    return True


def _probe_vals(rec: Any, probe: int) -> tuple[Any, Any]:
    """Return the record's ``(in_val, out_val)`` for one capture."""
    if probe == 1:
        return rec.example, rec.out_val
    return rec.example2, rec.out_val2


def _sum_bases(s: Any, sib: Any, oth: Any, x: Any) -> list[Any]:
    """Candidate sums: ``Σ mem`` plus optional extra addends."""
    bases = [s]
    for extra in (sib, oth):
        if extra is not None:
            bases += [b + extra for b in list(bases)]
    if _is_tensor(x):
        bases += [b + x for b in list(bases)]
    return bases


def _evidence_targets(
    graph: M.MorphismGraph, mem: set, probe: int
) -> list[Any]:
    """Consumed values: model outs + non-member block inputs."""
    outs = graph._model_outs if probe == 1 else graph._model_outs2
    ins = [
        _probe_vals(graph.record(n.name), probe)[0]
        for n in graph.nodes
        if n.name not in mem
    ]
    return [v for v in list(outs) + ins if _is_tensor(v)]


def _side_vals(
    graph: M.MorphismGraph,
    mem: set,
    family_names: list[str],
    first: str,
    probe: int,
) -> tuple[Any, Any, Any]:
    """Return the (sibling sum, other-blocks sum, shared input)."""
    sib_names = [n for n in family_names if n not in mem]
    other_names = [
        n.name
        for n in graph.nodes
        if n.name not in mem and n.name not in sib_names
    ]
    sib = _sum_vals(
        _probe_vals(graph.record(n), probe)[1] for n in sib_names
    )
    oth = _sum_vals(
        _probe_vals(graph.record(n), probe)[1] for n in other_names
    )
    x = _probe_vals(graph.record(first), probe)[0]
    return sib, oth, x


def _family_sum_evidence(
    graph: M.MorphismGraph,
    members: tuple[str, ...],
    family_names: list[str],
    *,
    probe: int,
) -> bool:
    """On one capture: the members' outputs add into a consumed value.

    Candidate bases cover the shapes the parent can actually write:
    ``Σ mem``, plus the shared input, the same-input siblings, and
    the remaining blocks as optional extra addends; targets are the
    model's return and every non-member block's captured input.
    """
    mem = set(members)
    s = _sum_vals(
        _probe_vals(graph.record(n), probe)[1] for n in members
    )
    if s is None:
        return False
    sib, oth, x = _side_vals(
        graph, mem, family_names, members[0], probe
    )
    bases = _sum_bases(s, sib, oth, x)
    targets = _evidence_targets(graph, mem, probe)
    return any(_vals_close(b, t) for b in bases for t in targets)


def _family_evidence(
    graph: M.MorphismGraph,
    members: tuple[str, ...],
    family_names: list[str],
) -> bool:
    """Two-capture additive-consumption evidence + the fan-out guard.

    The perturbed probe confirms the relation is structural, not a
    coincidental value match — the same convention the composer uses
    for residual boundaries.  When the second capture failed
    wholesale (``_probe2`` empty), the single-capture evidence is
    accepted; a present probe missing the members fails closed.
    """
    if not _fanout_clean(graph, members):
        return False
    if not _family_sum_evidence(graph, members, family_names, probe=1):
        return False
    if not graph._probe2:
        return True
    return _family_sum_evidence(graph, members, family_names, probe=2)


# ---------------------------------------------------------------------------
#  The factorisation — duck-typed numerics, no tensor library named
# ---------------------------------------------------------------------------


def _detach(t: Any) -> Any:
    """``t.detach()`` when the backend has it (torch), else ``t``."""
    d = getattr(t, "detach", None)
    return d() if callable(d) else t


def _fnorm(t: Any) -> float:
    """Frobenius norm — duck-typed ``sqrt(sum(t·t))``."""
    return float((t * t).sum()) ** 0.5


def _gs_row_basis(mats: list[Any], tol: float) -> list[Any]:
    """Orthonormal basis for the union of the matrices' row spaces.

    Modified Gram-Schmidt over the stacked rows, tolerance-gated: a
    row whose residual norm drops below ``tol * max_row_norm`` is
    already covered by the current basis — the result is the
    certified common row space at that tolerance.  The result is
    capped at the ambient dimension: a subspace of ``R^d`` admits at
    most ``d`` orthonormal directions, so any residual still above
    threshold once ``d`` vectors are kept is floating-point noise —
    and an over-complete "basis" would hand :func:`_stack_cols` a
    rank larger than the input space its one-hot columns index.
    """
    scale = 0.0
    for m in mats:
        for j in range(m.shape[0]):
            scale = max(scale, _fnorm(m[j]))
    thresh = tol * max(scale, 1e-12)
    basis: list[Any] = []
    dim = 0
    for m in mats:
        for j in range(m.shape[0]):
            v = m[j]
            dim = v.shape[-1]  # the ambient dimension
            for u in basis:
                v = v - u * float((u * v).sum())
            n = _fnorm(v)
            if n > thresh:
                basis.append(v / n)
    return basis[:dim]


def _stack_cols(basis: list[Any]) -> Any:
    """Stack the basis row-vectors as columns of one (d, r) tensor.

    Built duck-typed: ``u[:, None] * e[None, :]`` accumulates an
    outer product per basis vector, where ``e`` is the j-th one-hot
    row — no ``cat``/``stack`` symbol, so no tensor library is named.
    The one-hot ``e`` is carved out of a basis vector (length ``d``),
    so this requires ``r <= d`` — guaranteed by the ambient-
    dimension cap in :func:`_gs_row_basis`.
    """
    r = len(basis)
    u0 = basis[0]
    out = None
    for j, u in enumerate(basis):
        e = (u0 * 0)[:r]
        e[j] = 1.0
        col = u[:, None] * e[None, :]
        out = col if out is None else out + col
    return out


def _site_eff(s: _KVSite, leaves: dict) -> tuple[Any, Any, int] | None:
    """Return the site's ``(weight, effective matrix, d_in)``.

    ``linear`` / ``qkv`` weights contribute their rows, right-matmul
    weights their columns — every effective row lives in the shared
    input space ``R^{d_in}``.
    """
    w = leaves.get(s.weight.name)
    if w is None or not _is_tensor(w) or len(w.shape) != 2:
        return None
    w = _detach(w)
    if s.orient == "matmul_r":
        return w, w.T, int(w.shape[0])
    if s.orient == "qkv":
        st, ln = s.span or (0, 0)
        return w, w[st : st + ln], int(w.shape[1])
    return w, w, int(w.shape[1])


def _certify(
    kept: list[tuple[_KVSite, Any, Any]], Ut: Any, tol: float
) -> float | None:
    """Reconstruction residual bound — ``None`` when it exceeds *tol*."""
    max_err = 0.0
    for _s, _w, eff in kept:
        err = _fnorm(eff - (eff @ Ut) @ Ut.T)
        if err > tol * max(_fnorm(eff), 1e-30):
            return None
        max_err = max(max_err, err)
    return max_err


def _family_residual(
    kept: list[tuple[_KVSite, Any, Any]], Ut: Any
) -> tuple[float, float]:
    """Per-site reconstruction residuals — ``(Σ_i err_i, max_i err_i)``.

    ``err_i = ‖eff_i - (eff_i·U_t)·U_tᵀ‖_F``: the *measured*
    Frobenius error of the factorised weights, never estimated.  The
    sum is the bound the budget gates and the witness certifies;
    the max feeds the per-site ``factor_max_err`` stat.
    """
    errs = [_fnorm(k[2] - (k[2] @ Ut) @ Ut.T) for k in kept]
    return sum(errs), max(errs)


def _basis_masses(mats: list[Any], u: Any) -> list[float]:
    """Squared Frobenius mass one basis vector captures, per matrix.

    For an orthonormal basis vector ``u``, ``Σ_j (u·m_j)²`` is
    exactly the residual² site ``m`` gains when ``u`` is dropped —
    the coin the bounded-mode truncation spends.
    """
    return [
        sum(float((u * m[j]).sum()) ** 2 for j in range(m.shape[0]))
        for m in mats
    ]


def _spend_drops(
    per: list[list[float]],
    resid2: list[float],
    keep: int,
    budget: float,
) -> set[int]:
    """Greedy drop set: basis indices whose mass fits the budget.

    Candidates are tried smallest-mass first — the cheapest signal
    to discard — and a drop lands only while the running summed
    per-site residual (``Σ_i √(resid2_i)``) stays within ``budget``.
    """
    drops: set[int] = set()
    order = sorted(
        (j for j in range(len(per)) if j != keep),
        key=lambda j: sum(per[j]),
    )
    for j in order:
        trial = [r + pm for r, pm in zip(resid2, per[j], strict=True)]
        if sum(t**0.5 for t in trial) <= budget:
            resid2 = trial
            drops.add(j)
    return drops


def _truncate_basis(
    mats: list[Any], basis: list[Any], budget: float
) -> list[Any]:
    """Drop lowest-mass directions while the summed residual ≤ budget.

    Bounded-mode basis selection.  The exact cover keeps every
    direction above ``factor_tol`` — a useless rank on approximate
    weights, where the noise directions outnumber the signal.  Here
    the family may instead *spend* up to ``budget`` of summed
    per-site Frobenius residual.  The heaviest direction is never
    dropped: the law still factors, it does not zero out the family.
    The accept gate re-measures the residual exactly afterwards, so
    the greedy order only ever costs compression, never honesty.
    """
    if not basis or budget <= 0:
        return list(basis)
    per = [_basis_masses(mats, u) for u in basis]
    resid2 = [
        max(_fnorm(m) ** 2 - sum(pm[i] for pm in per), 0.0)
        for i, m in enumerate(mats)
    ]
    keep = max(range(len(basis)), key=lambda j: sum(per[j]))
    drops = _spend_drops(per, resid2, keep, budget)
    return [u for j, u in enumerate(basis) if j not in drops]


def _certified_factor(
    kept: list[tuple[_KVSite, Any, Any]],
    d_in: int | None,
    tol: float,
    budget: float | None,
) -> tuple[Any, int, float, list] | None:
    """Basis selection + certification for the kept triples.

    Exact mode: the ``tol`` cover plus the per-site relative
    residual gate (``_certify``).  Bounded mode: the budget-
    truncated basis (``_truncate_basis``) and the measured summed
    residual as ``err`` — the caller gates it against ``budget``.
    """
    mats = [k[2] for k in kept]
    basis = _gs_row_basis(mats, tol)
    if budget is not None:
        basis = _truncate_basis(mats, basis, budget)
    if not basis or d_in is None:
        return None
    Ut = _stack_cols(basis)
    if budget is None:
        err = _certify(kept, Ut, tol)
    else:
        err, _site_max = _family_residual(kept, Ut)
    if err is None:
        return None
    return Ut, int(d_in), err, kept


def _factor_sites(
    sites: list[_KVSite],
    leaves: dict,
    tol: float,
    budget: float | None = None,
) -> tuple[Any, int, float, list] | None:
    """Factor the sites' weight values through one common right-factor.

    Returns ``(Ut, d_in, err, kept)`` where ``kept`` is
    ``(site, weight, eff)`` triples; ``None`` when a weight is not a
    2-D tensor, the input dims disagree, or — exact mode — the
    reconstruction residual exceeds ``tol`` relative Frobenius.

    With ``budget`` the basis is truncated
    (:func:`_truncate_basis`): directions whose summed residual mass
    fits the budget are dropped, and ``err`` is the measured summed
    per-site Frobenius residual — the certified bound the caller
    gates against ``budget`` and offers on the witness.
    """
    kept: list[tuple[_KVSite, Any, Any]] = []
    d_in: int | None = None
    for s in sites:
        out = _site_eff(s, leaves)
        if out is None:
            return None
        w, eff, d_s = out
        if d_in is None:
            d_in = d_s
        elif d_in != d_s:
            return None
        kept.append((s, w, eff))
    return _certified_factor(kept, d_in, tol, budget)


# ---------------------------------------------------------------------------
#  The term rewrite
# ---------------------------------------------------------------------------


def _replace_nodes(term: Any, mapping: dict) -> Any:
    """Rebuild *term* replacing every node key in *mapping*."""
    if term in mapping:
        return mapping[term]
    if isinstance(term, Op):
        return Op.make(
            term.op,
            *(_replace_nodes(a, mapping) for a in term.args),
            **term.attrs,
        )
    return term


def _add_chain(terms: list[Any]) -> Any:
    """Right-folded ``add`` chain over the given terms."""
    acc = terms[-1]
    for t in reversed(terms[:-1]):
        acc = Op.make("add", t, acc)
    return acc


def _qkv_split_rewrite(
    node: Any,
    sites: list[_KVSite],
    C: Any,
    derived: dict,
    qsec_w: dict,
    qsec_b: dict,
    site_b: dict,
) -> Any:
    """``linear(d, W_qkv[, b])`` → ``concat(q, k', v')`` via the latent.

    The q row-section keeps its ordinary projection on the node's
    data operand (as a derived-leaf weight); each site section
    becomes ``linear(C, D_sec[, b_sec])`` on its derived recovery —
    and, when present, bias — leaves.  Concatenating the sections
    preserves the fused output exactly.
    """
    d = node.args[0]
    ss = sorted(sites, key=lambda s: (s.span or (0, 0))[0])
    q_args: list[Any] = [d, qsec_w[node]]
    if node in qsec_b:
        q_args.append(qsec_b[node])
    pieces = [Op.make("linear", *q_args)]
    for s in ss:
        k_args: list[Any] = [C, derived[s]]
        if s in site_b:
            k_args.append(site_b[s])
        pieces.append(Op.make("linear", *k_args))
    return Op.make("concat", *pieces, dim=-1)


def _kv_rewrite(
    body: Any,
    sites: list[_KVSite],
    C: Any,
    derived: dict,
    qsec_w: dict,
    qsec_b: dict,
    site_b: dict,
) -> Any:
    """Substitute the latent-factored forms into one member's body.

    ``linear(d, W[, b])`` → ``linear(C, D[, b])``;
    ``matmul(d, W)`` → ``matmul(C, D')``; a ``qkv`` node splits into
    per-section pieces.  Every ``D`` is a *derived* ``Param`` leaf —
    the recovery weights are materialised at reify time (the same
    convention as the lowerer's ``fused_*`` params; ``narrow`` terms
    deliberately do NOT fold — see the ``_FOLDABLE_*`` table in
    ``catopt_core.cost`` — so slices live as leaves, not terms).
    """
    repl: dict[Any, Any] = {}
    qkv: dict[Any, list[_KVSite]] = {}
    for s in sites:
        if s.orient == "linear":
            repl[s.node] = Op.make(
                "linear", C, derived[s], *s.node.args[2:]
            )
        elif s.orient == "matmul_r":
            repl[s.node] = Op.make("matmul", C, derived[s])
        else:
            qkv.setdefault(s.node, []).append(s)
    for node, ss in qkv.items():
        repl[node] = _qkv_split_rewrite(
            node, ss, C, derived, qsec_w, qsec_b, site_b
        )
    return _replace_nodes(body, repl)


def _qkv_leaves(
    i: int,
    s: _KVSite,
    w: Any,
    d_in: int,
    params: dict,
    leaves: dict,
    qsec_w: dict,
    qsec_b: dict,
    site_b: dict,
) -> None:
    """Materialise one qkv site's q-section and bias leaves."""
    st, ln = s.span or (0, 0)
    if s.node not in qsec_w:
        qp = Param(f"p_kv_latent_wq{i}", TensorType((ln, d_in)))
        qsec_w[s.node] = qp
        params[qp.name] = qp
        leaves[qp.name] = w[0:ln]
    if not (len(s.node.args) > 2 and isinstance(s.node.args[2], Param)):
        return
    bias = leaves.get(s.node.args[2].name)
    if not _is_tensor(bias):
        return
    bias = _detach(bias)
    if s.node not in qsec_b:
        qb = Param(f"p_kv_latent_qb{i}", TensorType((ln,)))
        qsec_b[s.node] = qb
        params[qb.name] = qb
        leaves[qb.name] = bias[0:ln]
    db = Param(f"p_kv_latent_db{i}", TensorType((ln,)))
    site_b[s] = db
    params[db.name] = db
    leaves[db.name] = bias[st : st + ln]


def _derived_leaves(
    kept: list[tuple[_KVSite, Any, Any]],
    Ut: Any,
    r: int,
    d_in: int,
    params: dict,
    leaves: dict,
) -> tuple[dict, dict, dict, dict]:
    """Materialise the recovery and q-section leaves.

    ``D_i = eff_i·U_t`` (transposed for right-matmul spellings),
    plus the q-section weight/bias leaves a fused-qkv split needs —
    materialised VALUES, not slice terms.
    """
    derived: dict[_KVSite, Param] = {}
    qsec_w: dict[Any, Param] = {}
    qsec_b: dict[Any, Param] = {}
    site_b: dict[_KVSite, Param] = {}
    for i, (s, w, eff) in enumerate(kept):
        if s.orient == "matmul_r":
            dv, dshape = (eff @ Ut).T, (r, int(eff.shape[0]))
        else:
            dv, dshape = eff @ Ut, (int(eff.shape[0]), r)
        dp = Param(f"p_kv_latent_d{i}", TensorType(dshape))
        derived[s] = dp
        params[dp.name] = dp
        leaves[dp.name] = dv
        if s.orient == "qkv":
            _qkv_leaves(
                i, s, w, d_in, params, leaves, qsec_w, qsec_b, site_b
            )
    return derived, qsec_w, qsec_b, site_b


def _zero_slot(
    sink: Sink, var: Var, inputs: tuple | None = None
) -> Any:
    """Lower an exact-zero filler — ``zeros_like`` on the slot input.

    ``inputs`` widens the filler's call signature for multi-input
    member slots; the context args are ignored.
    """
    ins = list(inputs) if inputs is not None else [var]
    ir = IR(
        root=Op.make("mul", var, Const(0)),
        inputs=ins,
        input_names={v.name for v in ins},
        params={},
    )
    return sink.lower(ir, {})


def _family_bodies(
    members: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Prefix+substitute each member's root onto the shared var.

    ``members`` is ``(name, ir)`` pairs — the caller already checked
    every record exported.  Single-input members only: each input var
    maps to the first member's (the family's shared input object,
    proven by capture identity upstream).
    """
    x = members[0][1].inputs[0]
    return {
        n: M._subst(
            M._prefix_params(i.root, M._ns_prefix(n)), i.inputs[0], x
        )
        for n, i in members
    }


def _family_bodies_ctx(
    members: list[tuple[str, Any]],
    recs: list[Any],
    graph: Any,
) -> tuple[dict[str, Any] | None, str | None]:
    """Like :func:`_family_bodies`, but multi-input-aware.

    Each member's *activation* var maps to the first member's;
    non-activation inputs must be the same captured object as one of
    the first member's inputs
    (:func:`catopt_orchestrator.morphisms._ctx_var_map`) — an
    unshared context input returns ``(None, reason)``.
    """
    sig0 = graph.sig(members[0][0])
    act0 = M._act_index(sig0.inputs) if sig0 is not None else 0
    x = members[0][1].inputs[act0]
    bodies: dict[str, Any] = {}
    for (n, i), rec in zip(members, recs, strict=True):
        sig = graph.sig(n)
        act = M._act_index(sig.inputs) if sig is not None else 0
        cmap, why = M._ctx_var_map(recs[0], rec, act)
        if cmap is None:
            return None, why
        body = M._prefix_params(i.root, M._ns_prefix(n))
        body = M._subst(M._subst_ctx(body, cmap), i.inputs[act], x)
        bodies[n] = body
    return bodies, None


def _family_tables(recs: list[Any]) -> tuple[dict, dict]:
    """Build the namespaced param and leaf tables for a family."""
    params: dict[str, Param] = {}
    leaves: dict[str, Any] = {}
    for r in recs:
        p = M._ns_prefix(r.name)
        params.update(
            {p + k: Param(p + k, t.typ) for k, t in r.ir.params.items()}
        )
        leaves.update({p + k: v for k, v in r.leaves.items()})
    return params, leaves


def _family_shape_check(recs: list[Any]) -> str | None:
    """Check the additive-sum graft's shape preconditions.

    All member outputs must share one shape (the sum must exist) and
    every consumed slot needs ``in == out`` so its exact-zero filler
    is shaped like the contribution it replaces.
    """
    oshape = M._shape_tuple(recs[0].out_val)
    if oshape is None:
        return "member outputs not tensors"
    if any(M._shape_tuple(r.out_val) != oshape for r in recs):
        return "member outputs not same shape"
    if any(M._shape_tuple(r.example) != oshape for r in recs[1:]):
        return "consumed slots need in-shape == out-shape"
    return None


def _kv_numbers(
    kept: list[tuple[_KVSite, Any, Any]],
    d_in: int,
    r: int,
    data: Any,
) -> dict[str, float]:
    """Compute the decode-path KV compute/memory deltas for stats.

    Counts the sites' marginal projection work: before, ``2·T·d_in·d``
    per site; after, one ``2·T·d_in·r`` latent GEMM plus
    ``2·T·r·d`` per site.  Bytes count the factored weight elements
    (an unfactored q-section of a ``qkv`` node persists on both
    sides and cancels).
    """
    dshape = _shape_of(data)
    tok = 1
    if isinstance(dshape, tuple):
        for dd in dshape[:-1]:
            tok *= dd if isinstance(dd, int) else 1
    flops_b = sum(2 * tok * d_in * k[2].shape[0] for k in kept)
    flops_a = 2 * tok * d_in * r + sum(
        2 * tok * r * k[2].shape[0] for k in kept
    )
    esize = 8
    for _s, w, _e in kept:
        f = getattr(w, "element_size", None)
        if callable(f):
            esize = int(f())
            break
    bytes_b = sum(k[2].shape[0] * d_in for k in kept) * esize
    bytes_a = (d_in * r + sum(k[2].shape[0] * r for k in kept)) * esize
    return {
        "kv_flops_before": float(flops_b),
        "kv_flops_after": float(flops_a),
        "kv_bytes_before": float(bytes_b),
        "kv_bytes_after": float(bytes_a),
    }


# ---------------------------------------------------------------------------
#  The law
# ---------------------------------------------------------------------------


class KVLatentShare:
    """``k_i = x·W_i → C = x·U`` once + per-block ``D_i`` — MLA-style.

    Signature-level candidacy mirrors :class:`WeightTie`'s: a family
    of ≥2 blocks that captured the *same input object* and carry
    KV-named ``Param`` projections on a shared data operand — plus,
    independently, single blocks whose own K/V projections share an
    operand.  The value side is the reify's job: the stacked weights
    must admit a common right-factor within the factor tolerance,
    the wiring evidence must hold on both captures, and the reified
    program is cost-gated and fp64-verified before grafting.

    Parameters
    ----------
    tokens : tuple[str, ...]
        Substrings marking a weight name as a K/V projection
        (``"k_proj"``, ``"v_proj"``).
    qkv_tokens : tuple[str, ...]
        Substrings marking a fused qkv weight whose k/v row-sections
        join the factorisation (q,k,v equal thirds).
    factor_tol : float
        Relative-Frobenius tolerance for the common-factor
        certification (Gram-Schmidt coverage and the reconstruction
        residual).
    intra, cross : bool
        Enable the single-block (K+V inside one block) and
        cross-block (shared input) forms.
    budget : float | None
        Bounded-error mode (plan 0012).  ``None`` — the default —
        keeps today's exact gate: the common factor must hold within
        ``factor_tol`` and nothing else fires.  A float enables the
        certified-approximate mode: the common basis is truncated to
        the directions whose summed per-site Frobenius residual fits
        the budget, the offer certifies ``error_bound = residual``
        (measured, never claimed smaller), the rewrite fires only
        when ``residual ≤ budget``, and the delivered module is
        verified at the propagated bound rather than left
        unverified — certified-bound instead of certified-exact.

    """

    name = "kv_latent_share"

    def __init__(
        self,
        *,
        tokens: tuple[str, ...] = _KV_TOKENS,
        qkv_tokens: tuple[str, ...] = _QKV_TOKENS,
        factor_tol: float = 1e-8,
        intra: bool = True,
        cross: bool = True,
        budget: float | None = None,
    ) -> None:
        """Store the law's detection and tolerance configuration."""
        self.tokens = tuple(tokens)
        self.qkv_tokens = tuple(qkv_tokens)
        self.factor_tol = float(factor_tol)
        self.intra = bool(intra)
        self.cross = bool(cross)
        self.budget = None if budget is None else float(budget)

    def match(self, graph: M.MorphismGraph) -> list[M.MorphismMatch]:
        """Match shared-data KV families — cross-block and intra."""
        out: list[M.MorphismMatch] = []
        if self.cross:
            for fam in _input_families(graph).values():
                out.extend(self._cross_matches(graph, fam))
        if self.intra:
            for n in graph.nodes:
                if n.opaque:
                    continue
                ir = graph.record(n.name).ir
                if ir is None or not _usable_member(graph, n.name):
                    continue
                for data in _latent_groups(
                    {n.name: ir.root},
                    self.tokens,
                    self.qkv_tokens,
                ):
                    out.append(
                        self._mk_match((n.name,), data, intra=True)
                    )
        # Wider families first — a cross match subsumes its members'
        # intra candidates; a declined cross still leaves them.
        out.sort(key=lambda m: (-len(m.nodes), m.nodes))
        return out

    def _mk_match(
        self, members: tuple[str, ...], data: Any, *, intra: bool
    ) -> M.MorphismMatch:
        """Pack the match: member nodes, boundary, reify payload."""
        return M.MorphismMatch(
            law=self.name,
            nodes=members,
            boundary="intra" if intra else "family",
            reify=M.ReifySpec(
                mode="family",
                rules="compose",
                extra={
                    "tokens": self.tokens,
                    "qkv_tokens": self.qkv_tokens,
                    "factor_tol": self.factor_tol,
                    "budget": self.budget,
                    "data": data,
                },
            ),
            detail=(
                f"{members[0]}: shared-data kv projections"
                if intra
                else f"{'+'.join(members)}: shared-input kv family"
            ),
        )

    def _cross_matches(
        self, graph: M.MorphismGraph, fam: list[str]
    ) -> list[M.MorphismMatch]:
        """Match one input-family's candidate latent groups."""
        usable: list[tuple[str, Any]] = []
        for n in fam:
            ir = graph.record(n).ir
            if _usable_member(graph, n):
                usable.append((n, ir))
        if len(usable) < 2:
            return []
        recs = [graph.record(n) for n, _ in usable]
        bodies, why = _family_bodies_ctx(usable, recs, graph)
        if bodies is None:
            log.debug("kv cross family %s declined: %s", fam, why)
            return []
        out: list[M.MorphismMatch] = []
        for data, sites in _latent_groups(
            bodies, self.tokens, self.qkv_tokens
        ).items():
            members = tuple(
                n for n in fam if any(s.member == n for s in sites)
            )
            if len(members) < 2:
                continue
            if not _family_evidence(graph, members, fam):
                continue
            out.append(self._mk_match(members, data, intra=False))
        return out


# ---------------------------------------------------------------------------
#  Reify — the family rewrite back to verified IR
# ---------------------------------------------------------------------------


def _family_prep(
    match: M.MorphismMatch, graph: M.MorphismGraph
) -> dict[str, Any]:
    """Validate members and build joint/params/leaves — or a decline.

    The returned dict carries the working set on success, or
    ``{"status": "declined", "reason": ...}`` on the honest gates:
    an unexported member, or a member taking more than one input.
    """
    recs = [graph.record(n) for n in match.nodes]
    irs = []
    sigs: dict[str, Any] = {}
    for r_ in recs:
        ir = r_.ir
        sig = graph.sig(r_.name)
        if ir is None or sig is None:
            return {"status": "declined", "reason": "opaque node"}
        if len(sig.inputs) != len(ir.inputs):
            # The sig describes a different-arity IR — a multi-input
            # record the lifted signature does not cover.
            return {"status": "declined", "reason": "multi-input block"}
        why_ = M._sig_liftable(sig)
        if why_ is not None:
            return {"status": "declined", "reason": why_}
        irs.append(ir)
        sigs[r_.name] = sig
    intra = len(recs) == 1
    act0 = M._act_index(sigs[recs[0].name].inputs)
    if intra:
        bodies: dict[str, Any] | None = {recs[0].name: irs[0].root}
        params = dict(irs[0].params)
        leaves = dict(recs[0].leaves)
        joint = irs[0].root
    else:
        bodies, why = _family_bodies_ctx(
            [(r_.name, i_) for r_, i_ in zip(recs, irs, strict=True)],
            recs,
            graph,
        )
        if bodies is None:
            return {"status": "declined", "reason": why}
        params, leaves = _family_tables(recs)
        joint = _add_chain([bodies[n] for n in match.nodes])
    return {
        "recs": recs,
        "irs": irs,
        "x": irs[0].inputs[act0],
        "inputs": tuple(irs[0].inputs),
        "intra": intra,
        "bodies": bodies,
        "params": params,
        "leaves": leaves,
        "joint": joint,
    }


def _latent_rewrite(
    sites: list[_KVSite],
    tol: float,
    data: Any,
    bodies: dict,
    nodes: tuple,
    params: dict,
    leaves: dict,
    intra: bool,
    budget: float | None = None,
) -> tuple[tuple[Any, int, int, float, float, list] | None, str | None]:
    """Certify the common factor and build the factorized joint term.

    Returns ``((joint2, r, d_in, bound, site_max, kept), None)`` —
    the rewritten joint plus the certification record (``bound`` is
    the witnessed error bound: the per-site max in exact mode, the
    summed family residual in bounded mode; ``site_max`` is always
    the max per-site residual) — or ``(None, reason)`` on the
    honest declines: no certified common right-factor, or a
    measured residual over ``budget``.
    """
    fac = _factor_sites(sites, leaves, tol, budget=budget)
    if fac is None:
        return None, "no certified common factor"
    Ut, d_in, err, kept = fac
    if budget is None:
        bound, site_max = err, err
    else:
        bound, site_max = _family_residual(kept, Ut)
        if bound > budget:
            return None, (
                f"bound exceeds budget: {bound:.3e} > {budget:.3e}"
            )
    r = int(Ut.shape[1])
    Ut_p = Param("p_kv_latent_ut", TensorType((d_in, r)))
    params[Ut_p.name] = Ut_p
    leaves[Ut_p.name] = Ut
    C = Op.make("matmul", data, Ut_p)
    maps = _derived_leaves(kept, Ut, r, d_in, params, leaves)
    bodies2 = {
        n: _kv_rewrite(
            bodies[n],
            [s for s in sites if s.member == n],
            C,
            *maps,
        )
        for n in nodes
    }
    joint2 = (
        bodies2[nodes[0]]
        if intra
        else _add_chain([bodies2[n] for n in nodes])
    )
    return (joint2, r, d_in, bound, site_max, kept), None


def _tmax(t: Any) -> float | None:
    """``float(t.abs().max())`` duck-typed; ``None`` when absent."""
    a = getattr(t, "abs", None)
    if not callable(a):
        return None
    m = getattr(a(), "max", None)
    return float(m()) if callable(m) else None


def _propagated_bound(
    recs: list[Any], x_val: Any, residual: float
) -> tuple[float | None, float | None]:
    """Cheap output-space estimate of the certified weight residual.

    Each site's output perturbation is ``x·E_iᵀ``, so the family-sum
    output moves by at most ``‖x‖_F · Σ_i‖E_i‖_F`` on a given input —
    the per-consumer sensitivity estimate propagated through the
    residual distribution.  Returns ``(abs, rel)`` where ``rel``
    normalises by the captured joint-output magnitude — the same
    normalisation the pair verify applies.  Whichever leg the
    captured values cannot supply comes back ``None``.  This is an
    *estimate* — the attention path is not Lipschitz-analysed; the
    certified bound stays the weight-space residual.
    """
    if not _is_tensor(x_val):
        return None, None
    out_abs = _fnorm(x_val) * residual
    scale = _tmax(_sum_vals(r.out_val for r in recs))
    if scale is None:
        return out_abs, None
    return out_abs, out_abs / (scale + 1e-8)


def _bounded_ctx(
    recs: list[Any], verify_tol: float, budget: float, bound: float
) -> tuple[float, str, dict[str, Any]]:
    """Bounded-mode extras: verify tolerance, law suffix, stat block.

    Verified-with-tolerance, not unverified: the pair verify gates
    at the residual propagated to the verify metric's own
    (output-relative) units, never below ``verify_tol``.  The stats
    carry the certified bound — the match record always shows which
    bounded rewrites fired and at what bound.
    """
    out_bound, rel_bound = _propagated_bound(
        recs, recs[0].example, bound
    )
    eff_tol = max(
        verify_tol,
        rel_bound if rel_bound is not None else bound,
    )
    suffix = (
        f" (bounded: measured Frobenius residual {bound:.3e} "
        f"≤ budget {budget:.3e})"
    )
    stats = {
        "bounded": True,
        "error_bound": bound,
        "bound_norm": "frobenius",
        "error_budget": budget,
        "error_bound_out": out_bound,
        "verify_tol": eff_tol,
    }
    return eff_tol, suffix, stats


def _reify_family(
    match: M.MorphismMatch,
    graph: M.MorphismGraph,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    max_iterations: int,
    max_enodes: int,
    symmetry_budget: int | None,
) -> dict[str, Any]:
    """Reify a KV-latent match; return the graft record or a decline.

    Factor the member weights' common row space, rewrite the shared
    projections through one latent GEMM, offer the constructed joint
    term as a witnessed e-graph member (carrying its certified error
    bound), saturate the compose recipe, cost-gate against the
    un-rewritten joint, verify fp64, and emit the per-slot
    replacements: the fused family at the first member's slot (or the
    rewritten block itself for intra matches) plus exact-zero
    fillers for the consumed members.
    """
    spec = match.reify
    extra = spec.extra or {}
    tokens = tuple(extra.get("tokens", _KV_TOKENS))
    qkv_tokens = tuple(extra.get("qkv_tokens", _QKV_TOKENS))
    tol = float(extra.get("factor_tol", 1e-8))
    budget = extra.get("budget")
    if budget is not None:
        budget = float(budget)
    data = extra.get("data")
    prep = _family_prep(match, graph)
    if prep.get("status") == "declined":
        return prep
    recs = prep["recs"]
    irs = prep["irs"]
    x = prep["x"]
    inputs = prep["inputs"]
    intra = prep["intra"]
    bodies = prep["bodies"]
    params = prep["params"]
    leaves = prep["leaves"]
    joint = prep["joint"]
    sites = [
        s
        for n in match.nodes
        for s in _kv_proj_sites(bodies[n], n, tokens, qkv_tokens)
        if s.data == data
    ]
    if len(sites) < 2:
        return {
            "status": "declined",
            "reason": "no shared-data kv sites",
        }
    fac, why = _latent_rewrite(
        sites,
        tol,
        data,
        bodies,
        match.nodes,
        params,
        leaves,
        intra,
        budget=budget,
    )
    if fac is None:
        return {
            "status": "declined",
            "reason": why or "no certified common factor",
        }
    joint2, r, d_in, bound, site_max, kept = fac
    if budget is not None:
        eff_tol, suffix, bstats = _bounded_ctx(
            recs, verify_tol, budget, bound
        )
    else:
        eff_tol, suffix, bstats = verify_tol, "", {}
    law_text = (
        "kv_latent_share: the member weights' rows share a "
        f"certified rank-{r} subspace — W = (W·U_t)·U within "
        "the factor tolerance; morphism-level assertion, "
        "gated by the fp64 verify"
    ) + suffix
    eg, eid = M._saturate(
        joint,
        M._recipe_rules(spec.rules),
        max_iterations=max_iterations,
        max_enodes=max_enodes,
        symmetry_budget=symmetry_budget,
        offers=[
            (
                joint2,
                law_text,
                {
                    "note": "kv_latent_share: common right-factor "
                    "fold over the shared input",
                    "error_bound": bound,
                    "bound_norm": "frobenius",
                },
            )
        ],
    )
    best = eg.extract_best(eid, cost_fn)
    info: dict[str, Any] = {
        "joint": op_repr(joint),
        "reified": op_repr(best),
        "latent_rank": r,
        "d_in": d_in,
        "n_kv_sites": len(sites),
        "factor_max_err": site_max,
        **_kv_numbers(kept, d_in, r, data),
    }
    info.update(bstats)
    return _family_gate(
        match,
        recs,
        irs,
        graph,
        intra=intra,
        sink=sink,
        cost_fn=cost_fn,
        verify_tol=eff_tol,
        joint=joint,
        best=best,
        x=x,
        inputs=inputs,
        params=params,
        leaves=leaves,
        info=info,
    )


def _family_gate(
    match: M.MorphismMatch,
    recs: list[Any],
    irs: list[Any],
    graph: Any,
    *,
    intra: bool,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    joint: Any,
    best: Any,
    x: Any,
    inputs: tuple,
    params: dict,
    leaves: dict,
    info: dict,
) -> dict[str, Any]:
    """Cost-gate, shape-check, fp64-verify, emit the slot reps."""
    info["cost_before"] = dag_cost(joint, cost_fn)
    info["cost_after"] = dag_cost(best, cost_fn)
    if not info["cost_after"] < info["cost_before"]:
        return {
            "status": "declined",
            "reason": "no_improvement",
            **info,
        }
    if not intra:
        decline = _family_shape_check(recs)
        if decline is not None:
            return {"status": "declined", "reason": decline, **info}
    vr = M._verify_pair(
        sink,
        joint,
        best,
        x,
        params,
        leaves,
        recs[0].args,
        verify_tol,
        inputs,
    )
    info["rel_diff"] = vr.max_rel
    if not vr.passed:
        return {
            "status": "declined",
            "reason": f"reify verify failed: {vr.max_rel:.3e}",
            **info,
        }
    reps = {
        match.nodes[0]: M._lower_term(
            best, x, params, leaves, sink, inputs
        )
    }
    if not intra:
        for r_, i_ in zip(recs[1:], irs[1:], strict=True):
            sig_j = graph.sig(r_.name)
            act_j = (
                M._act_index(sig_j.inputs) if sig_j is not None else 0
            )
            reps[r_.name] = _zero_slot(
                sink, i_.inputs[act_j], tuple(i_.inputs)
            )
    return {"status": "grafted", "reps": reps, **info}
