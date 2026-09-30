"""Factored-parameter execution — low-rank detection pass.

``x @ (A·B)`` with ``rank ≪ dims`` computes cheaper as ``(x @ A) @ B``
— the factored order trades one ``2·M·d_in·d_out`` GEMM for two skinny
GEMMs at ``2·M·r·(d_in + d_out)`` flops, winning whenever
``r < d_in·d_out / (d_in + d_out)``.  The equational surface already
carries the association laws that derive both spellings
(``assoc_matmul`` / ``assoc_matmul_rev`` / ``assoc_linear`` /
``assoc_linear_rev``), and cost-driven extraction picks the cheaper
member; ``IRModule._fold_weight_chains`` materialises whichever
param-only product survives.  So a *spelled* factorisation — a LoRA
adapter's ``lora_A``/``lora_B`` pair, a built low-rank projection —
is handled entirely by the laws: extraction keeps it factored when
the rank clears the break-even.

What the laws cannot see is a *dense* parameter whose stored values
happen to be numerically low-rank — the "detected factor" case.  This
module's :func:`offer_low_rank_factors` is a non-local pass in the
:mod:`catopt_core.laws.pairing` family: it scans projection e-nodes
(``matmul(data, W)`` / ``linear(data, W[, b])``), certifies a
Gram-Schmidt rank-``r`` factorisation of the weight's stored value,
and offers the factored member into the consumer's e-class under a
pointwise witness carrying the Frobenius error bound.  Extraction
then decides on cost — the offer is never forced, so a model whose
"low-rank" weight does not actually pay keeps its dense form.
Derived factor parameters are registered into ``source_tensors`` so
lowering materialises them like any weight (the same convention as
``share_duplicate_param_slices``).

Deliberately out of scope: left-weight ``matmul(W, data)`` spellings,
non-leaf weight expressions (a ``matmul(A, B)`` weight *term* is the
laws' job — ``assoc_*`` already derives the factored consumer there),
and 1-D / >2-D weights.
"""

# ruff: noqa: RUF003 -- comments/docstrings use
# mathematical notation (×, ≈, ·) deliberately.

from typing import Any

from catopt_core.egraph.types import _LeafRegistry
from catopt_core.ir import Param, TensorType, Var
from catopt_core.laws.pairing import _exact_equal, _is_tensor

__all__ = ["offer_low_rank_factors"]


# ---------------------------------------------------------------------------
#  Duck-typed numerics — no tensor library named (same contract as the
#  pairing passes and the orchestrator's morphism factorisation)
# ---------------------------------------------------------------------------


def _detach(t: Any) -> Any:
    """``t.detach()`` when the backend has it (torch), else ``t``."""
    d = getattr(t, "detach", None)
    return d() if callable(d) else t


def _fnorm(t: Any) -> float:
    """Frobenius norm — duck-typed ``sqrt(sum(t·t))``."""
    return float((t * t).sum()) ** 0.5


def _row_basis(m: Any, tol: float) -> list[Any]:
    """Orthonormal basis for *m*'s row space, tolerance-gated.

    Modified Gram-Schmidt over the rows: a row whose residual norm
    drops below ``tol * max_row_norm`` is already covered by the
    current basis — the result is the certified row space at that
    tolerance.  (Same construction the KV-latent morphism uses for
    its common right-factor; duplicated here because core cannot
    import the orchestrator.)
    """
    scale = 0.0
    for j in range(m.shape[0]):
        scale = max(scale, _fnorm(m[j]))
    thresh = tol * max(scale, 1e-12)
    basis: list[Any] = []
    for j in range(m.shape[0]):
        v = m[j]
        for u in basis:
            v = v - u * float((u * v).sum())
        n = _fnorm(v)
        if n > thresh:
            basis.append(v / n)
    return basis


def _col_stack(basis: list[Any]) -> Any:
    """Stack basis row-vectors as columns of one ``(d, r)`` tensor.

    Duck-typed outer-product accumulation — ``u[:, None] * e[None, :]``
    per basis vector where ``e`` is the j-th one-hot row — so no
    ``stack``/``cat`` symbol is needed.
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


def _factor_rows(w: Any, tol: float) -> tuple[Any, Any, float] | None:
    """Row-space factorisation ``w ≈ B·A`` for a ``(o, i)`` weight.

    Returns ``(A, B, err)`` with ``A`` the ``(r, i)`` orthonormal row
    basis, ``B = w·Aᵀ`` the ``(o, r)`` coefficient matrix, and ``err``
    the absolute Frobenius reconstruction residual — or ``None`` when
    the row space is empty, is numerically full-rank (a rank at
    ``min(o, i)`` can never clear the ``r·(o+i) < o·i`` break-even —
    and under fp noise the residual rows keep spawning phantom basis
    vectors, so the rank is gated BEFORE the column stack), or the
    residual exceeds ``tol`` relative Frobenius.
    """
    w = _detach(w)
    basis = _row_basis(w, tol)
    if not basis or len(basis) >= min(int(w.shape[0]), int(w.shape[1])):
        return None
    ut = _col_stack(basis)  # (i, r)
    a = _detach(ut.T)  # (r, i) — the orthonormal row basis
    b = _detach(w @ ut)  # (o, r) — coordinates in that basis
    err = _fnorm(w - b @ a)
    if err > tol * max(_fnorm(w), 1e-30):
        return None
    return a, b, err


def _factor_cols(w: Any, tol: float) -> tuple[Any, Any, float] | None:
    """Column-space factorisation ``w ≈ A·B`` for an ``(i, o)`` weight.

    Returns ``(A, B, err)`` with ``A`` the ``(i, r)`` orthonormal
    column basis, ``B = Aᵀ·w`` the ``(r, o)`` coefficient matrix, and
    ``err`` the absolute Frobenius reconstruction residual — same
    gating as :func:`_factor_rows`.
    """
    w = _detach(w)
    basis = _row_basis(w.T, tol)  # rows of w.T = columns of w
    if not basis or len(basis) >= min(int(w.shape[0]), int(w.shape[1])):
        return None
    a = _detach(_col_stack(basis))  # (i, r)
    b = _detach(a.T @ w)  # (r, o)
    err = _fnorm(w - a @ b)
    if err > tol * max(_fnorm(w), 1e-30):
        return None
    return a, b, err


# ---------------------------------------------------------------------------
#  E-graph plumbing — leaf resolution and var-reachability over classes
# ---------------------------------------------------------------------------


def _leaf_params(eg: Any, cid: int) -> list[Param]:
    """``Param`` leaf members of *cid*'s e-class."""
    out: list[Param] = []
    for node in eg._classes[eg.find(cid)].nodes:
        if node.op != "leaf":
            continue
        leaf = _LeafRegistry.decode(node.attrs[0][1])
        if isinstance(leaf, Param):
            out.append(leaf)
    return out


def _cls_has_var(eg: Any, memo: dict[int, bool]) -> Any:
    """Return a predicate testing whether an e-class reaches a ``Var`` leaf.

    Same var-reachability convention as ``_pair_shared_input`` — a
    class-level check, more sound than testing an arbitrary extracted
    representative.
    """

    def go(cid: int, stack: frozenset) -> bool:
        cid = eg.find(cid)
        hit = memo.get(cid)
        if hit is not None:
            return hit
        if cid in stack:
            return False
        res = False
        for n in eg._classes[cid].nodes:
            if n.op == "leaf":
                leaf = _LeafRegistry.decode(n.attrs[0][1])
                res = isinstance(leaf, Var)
            else:
                res = any(
                    go(eg.find(c), stack | {cid}) for c in n.children
                )
            if res:
                break
        memo[cid] = res
        return res

    return go


def _fresh_name(source_tensors: dict, want: str) -> str:
    """First ``want[_n]`` variant not already in *source_tensors*."""
    name = want
    n = 0
    while name in source_tensors:
        n += 1
        name = f"{want}_{n}"
    return name


# ---------------------------------------------------------------------------
#  The pass
# ---------------------------------------------------------------------------


def _register_factor(
    eg: Any,
    source_tensors: dict,
    cache: dict,
    pname: str,
    r: int,
    a: Any,
    b: Any,
) -> tuple[int, int, str, str]:
    """Register the factor tensors; return ``(a_eid, b_eid, a, b)``.

    One ``{name}__lr{r}_{a,b}`` pair per source param per pass —
    consumers of the same weight share the derived leaves (and their
    e-classes).  A name already carrying bitwise-identical values is
    reused, like ``share_duplicate_param_slices``.
    """
    hit = cache.get(pname)
    if hit is not None:
        return hit
    a_want = f"{pname}__lr{r}_a"
    a_name = a_want
    if not (
        a_want in source_tensors
        and _exact_equal(source_tensors[a_want], a)
    ):
        a_name = _fresh_name(source_tensors, a_want)
        source_tensors[a_name] = _detach(a)
    b_want = f"{pname}__lr{r}_b"
    b_name = b_want
    if not (
        b_want in source_tensors
        and _exact_equal(source_tensors[b_want], b)
    ):
        b_name = _fresh_name(source_tensors, b_want)
        source_tensors[b_name] = _detach(b)
    a_eid = eg.add_term(
        Param(a_name, TensorType(tuple(int(d) for d in a.shape))),
        provenance="low_rank_factor",
    )
    b_eid = eg.add_term(
        Param(b_name, TensorType(tuple(int(d) for d in b.shape))),
        provenance="low_rank_factor",
    )
    out = (a_eid, b_eid, a_name, b_name)
    cache[pname] = out
    return out


def _site_orient(node: Any) -> tuple[str, int, int] | None:
    """Classify one e-node as a projection site — or decline.

    Returns ``(orient, data_child, weight_child)``: ``"matmul_r"`` for
    ``matmul(data, W)`` (weight's *columns* live in input space) and
    ``"linear"`` for ``linear(data, W[, b])`` (weight rows).  Every
    other node shape is not a site.
    """
    if node.op == "matmul" and len(node.children) == 2:
        return "matmul_r", node.children[0], node.children[1]
    if node.op == "linear" and len(node.children) in (2, 3):
        return "linear", node.children[0], node.children[1]
    return None


def _build_member(
    eg: Any, orient: str, data_c: int, a_eid: int, b_eid: int, node: Any
) -> int:
    """Intern the factored member; return its root e-class id.

    ``matmul_r``: ``matmul(matmul(data, A), B)``; ``linear``:
    ``linear(linear(data, A), B)``, with a present bias re-added
    outside the chain.  Children stay e-class ids so extraction keeps
    sharing each member's best subterm (the ``_build_grouped_gemm``
    convention).
    """
    op = "matmul" if orient == "matmul_r" else "linear"
    inner = eg.add_enode(
        op,
        (eg.find(data_c), a_eid),
        {},
        provenance="low_rank_factor",
    )
    outer = eg.add_enode(
        op,
        (inner, b_eid),
        {},
        provenance="low_rank_factor",
    )
    if orient == "linear" and len(node.children) == 3:
        outer = eg.add_enode(
            "add",
            (outer, eg.find(node.children[2])),
            {},
            provenance="low_rank_factor",
        )
    return outer


def _offer_one_site(
    eg: Any,
    cid: int,
    node: Any,
    orient: str,
    data_c: int,
    p: Param,
    source_tensors: dict,
    factored: dict,
    rel_tol: float,
    witness: bool,
) -> dict | None:
    """Certify and offer one leaf weight's factored member — or none."""
    w = source_tensors.get(p.name)
    if w is None or not _is_tensor(w) or len(w.shape) != 2:
        return None
    w = _detach(w)
    fac = (
        _factor_cols(w, rel_tol)
        if orient == "matmul_r"
        else _factor_rows(w, rel_tol)
    )
    if fac is None:
        return None
    a, b, err = fac
    if orient == "matmul_r":
        i_, o_, r = int(w.shape[0]), int(w.shape[1]), int(b.shape[0])
    else:
        i_, o_, r = int(w.shape[1]), int(w.shape[0]), int(a.shape[0])
    if r * (i_ + o_) >= i_ * o_:
        return None  # no flop/storage saving possible
    a_eid, b_eid, a_name, b_name = _register_factor(
        eg, source_tensors, factored, p.name, r, a, b
    )
    outer = _build_member(eg, orient, data_c, a_eid, b_eid, node)
    merged = eg._offer_witness(
        cid,
        outer,
        rhs_term=eg.any_term(outer),
        provenance="low_rank_factor",
        law=(
            "pointwise witness for low-rank factoring: the weight's "
            "stored value admits a certified rank-"
            f"{r} factorisation (Gram-Schmidt basis, Frobenius "
            "residual carried as error_bound) — equality established "
            "by the factorisation pass"
        ),
        witness=witness,
        error_bound=err,
        bound_norm="frobenius",
        note=(
            f"offer_low_rank_factors: {p.name} [{orient}] rank {r}, "
            f"rel. Frobenius err {err / max(_fnorm(w), 1e-30):.3e}"
        ),
    )
    if not merged:
        return None
    return {
        "param": p.name,
        "orient": orient,
        "rank": r,
        "in_dim": i_,
        "out_dim": o_,
        "a_param": a_name,
        "b_param": b_name,
        "error": err,
        "eid": outer,
    }


def offer_low_rank_factors(
    eg: Any,
    source_tensors: dict,
    *,
    rel_tol: float = 1e-8,
    witness: bool = True,
) -> list[dict]:
    """Offer factored members for numerically low-rank weight params.

    Scans every ``matmul(data, W)`` and ``linear(data, W[, b])``
    e-node whose weight argument's e-class holds a ``Param`` leaf
    backed by a 2-D ``source_tensors`` value; when that value's
    certified rank ``r`` clears the flop break-even
    ``r·(i + o) < i·o``, the factored member —
    ``matmul(matmul(data, A), B)`` resp. ``linear(linear(data, A),
    B)`` (a present bias re-added outside the chain) — is offered
    into the consumer's e-class under a pointwise witness carrying
    the Frobenius residual bound.

    The member then competes as an ordinary alternative: extraction
    selects it only when the active cost model honestly prefers the
    factored chain — under flop pricing exactly when the pass's own
    break-even holds; under the roofline/executor models only when
    the saved work also beats the extra kernel launch.  There is
    nothing to force and nothing to roll back.  The derived factor
    parameters are registered into ``source_tensors`` so lowering
    materialises them like any weight.

    ``data`` must reach a ``Var`` — a param-only consumer folds at
    compile time either way.  Non-leaf weight expressions, left-side
    ``matmul(W, data)`` weights, non-2-D tensors, and weights whose
    reconstruction exceeds ``rel_tol`` relative Frobenius are
    skipped.

    Returns one record per offered member: ``{param, orient, rank,
    in_dim, out_dim, a_param, b_param, error, eid}``.
    """
    offers: list[dict] = []
    has_var = _cls_has_var(eg, {})
    factored: dict[str, tuple] = {}
    # Snapshot: the offered unions mutate the class table mid-walk —
    # find() always lands on the live canonical id for a snapshot key.
    for cid in list(eg._classes):
        cid = eg.find(cid)
        for node in tuple(eg._classes[cid].nodes):
            site = _site_orient(node)
            if site is None:
                continue
            orient, data_c, w_c = site
            if not has_var(data_c, frozenset()):
                continue  # param-only consumer folds either way
            for p in _leaf_params(eg, w_c):
                rec = _offer_one_site(
                    eg,
                    cid,
                    node,
                    orient,
                    data_c,
                    p,
                    source_tensors,
                    factored,
                    rel_tol,
                    witness,
                )
                if rec is not None:
                    offers.append(rec)
                    break  # one certified factor per consumer site
    return offers
