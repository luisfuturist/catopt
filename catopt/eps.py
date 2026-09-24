"""The ε axis — certified approximations inside the e-graph.

Exact laws preserve semantics; **ε-laws preserve semantics up to a
certified bound**.  A bounded rewrite is an ordinary :class:`Rewrite`
carrying ``error_bound``/``bound_norm``: it enters the e-graph like
any other offer, but every certificate that uses it accumulates the
bound (triangle inequality, conservative).  Extraction can then trade
stored bytes / FLOPs against a *proved* error — quantisation,
low-rank factorisation, and parameter tying all become the same object:
rewrites with an error bound.

This module implements the first ε-pass: **spectral low-rank
factorisation at ``linear`` application sites**.

    For each ``linear(x, W[, b])`` e-node with W (o×i), truncated SVD
    gives  W_r = U_r Σ_r V_rᵀ  with the *exact* Eckart–Young bound

        ‖W − W_r‖₂ = σ_{r+1}          (spectral norm)

    and offers the member ``linear(linear(x, V_r), U_rΣ_r[, b])`` —
    a two-layer chain, NOT ``matmul(U,V)``, because a weight-only
    matmul would be folded back into one full param at lowering.  The
    chained form stores ``r(o+i)`` values and is only offered when
    that is genuinely smaller.

    The certified bound is on the *substituted weight*: for the
    consuming site the output error is ≤ σ_{r+1}·‖x‖₂ (per-token,
    Euclidean).  Whole-model propagation needs per-op Lipschitz
    constants — not yet computed; the certificate reports the sum of
    site-local spectral bounds.

The derived factor Params are injected into ``source_tensors`` so
``ir_to_torch_module`` materialises them as real parameters of the
optimized module — the weights file itself changes shape.

Phase-0 honesty (``measure_weights.py``): on real trained weights
exact structure is absent, so this pass is the *only* weight-space
direction that exists — and its bound is what makes the trade
certified rather than approximate vibes.
"""
from __future__ import annotations

from typing import Any

import torch

from catopt.egraph import EGraph, Rewrite
from catopt.ir import Op, Param, TensorType

__all__ = ["low_rank_params", "kron_linear_params"]


def _kron_member(eg, x_eid, terms, spec):
    """Assemble ``add`` over K ``reshape(matmul(matmul(A, reshape x),
    transpose B), out)`` members — the executable form of
    ``linear(x, Σᵢ Aᵢ⊗Bᵢ)``.  A is (m1×n1), B is (m2×n2):
    ``y[(a2,b2)] = (A·X·Bᵀ)[a2,b2]`` where ``X = reshape(x,(n1,n2))``.
    Batch dims are preserved.  Returns the combined enode id."""
    m1, n1, m2, n2, x_batch, K = spec
    acc = None
    for t in range(K):
        A_t, B_t = terms[t]
        xr = eg.add_enode("reshape", (x_eid,),
                          {"shape": (*x_batch, n1, n2)})
        ax = eg.add_enode("matmul", (eg.add_term(A_t), xr))
        axb = eg.add_enode("matmul", (ax, eg.add_enode(
            "transpose", (eg.add_term(B_t),),
            {"arg1": -2, "arg2": -1})))
        y = eg.add_enode("reshape", (axb,),
                         {"shape": (*x_batch, m1 * m2)})
        acc = y if acc is None else eg.add_enode("add", (acc, y))
    return acc


def kron_linear_params(eg: EGraph, source_tensors: dict, *,
                       rtol: float = 0.05,
                       min_saving: float = 0.8,
                       witness: bool = True) -> list[dict]:
    """Offer a **sum-of-Kronecker** factorisation at each ``linear``
    site:  ``W ≈ Σᵢ Aᵢ⊗Bᵢ``  executes as

        reshape(x, (n1,n2)) → matmul(Aᵢ, ·) → matmul(·, Bᵢᵀ) →
        reshape(·, (m1m2)) →  summed over i, (+ bias)

    chosen over factor pairs (m1·m2=o, n1·n2=i) minimising stored
    values ``K·(m1n1+m2n2)`` subject to the relative Frobenius
    residual ≤ ``rtol`` (the rearrangement is a Frobenius isometry, so
    the bound is exact and certifies ``‖W − Ŵ‖_F``).

    The offered member is a *program* — K composed maps — exactly the
    "weights as programs" object: no dense W materialises, only the
    ``eps_k*`` factor params (injected into ``source_tensors``).
    """
    offers: list[dict] = []
    for cid in list(eg._classes.keys()):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for node in list(ec.nodes):
            if node.op != "linear" or len(node.children) not in (2, 3):
                continue
            wt = eg.any_term(eg.find(node.children[1]))
            if not isinstance(wt, Param) or wt.name not in source_tensors:
                continue
            W = source_tensors[wt.name]
            if not (isinstance(W, torch.Tensor) and W.ndim == 2):
                continue
            o, i = W.shape
            Wd = W.detach().double()
            # choose the (m1,n1) split minimising storage at rtol
            best = None
            for m1 in range(2, int(o ** 0.5) + 2):
                if o % m1:
                    continue
                m2 = o // m1
                for n1 in range(2, int(i ** 0.5) + 2):
                    if i % n1:
                        continue
                    n2 = i // n1
                    R = (Wd.reshape(m1, m2, n1, n2)
                         .permute(0, 2, 1, 3)
                         .reshape(m1 * n1, m2 * n2))
                    try:
                        Ur, S, Vr = torch.linalg.svd(R,
                                                     full_matrices=False)
                    except Exception:
                        continue
                    if S[0] == 0:
                        continue
                    e = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
                    K = int((e < 1 - rtol ** 2).sum().item()) + 1
                    K = min(K, S.numel())
                    stored = K * (m1 * n1 + m2 * n2)
                    if stored < min_saving * o * i and (
                            best is None or stored < best[0]):
                        resid = float(torch.sqrt(
                            (S[K:] ** 2).sum()).item())
                        best = (stored, m1, m2, n1, n2, K, resid,
                                Ur, S, Vr)
            if best is None:
                continue
            stored, m1, m2, n1, n2, K, resid, Ur, S, Vr = best
            terms = []
            for t in range(K):
                aname = f"eps_k{t}_a_{wt.name}_{c}"
                bname = f"eps_k{t}_b_{wt.name}_{c}"
                # R = UΣVᵀ term t :  A_t = reshape(U[:,t]·σ_t, (m1,n1))
                #                     B_t = reshape(V[t],   (m2,n2))
                source_tensors[aname] = (Ur[:, t] * S[t]).reshape(
                    m1, n1).to(W.dtype).contiguous()
                source_tensors[bname] = Vr[t, :].reshape(
                    m2, n2).to(W.dtype).contiguous()
                terms.append((Param(aname, TensorType((m1, n1))),
                              Param(bname, TensorType((m2, n2)))))
            x_eid = node.children[0]
            x_term = eg.any_term(x_eid)
            x_batch = (x_term.typ.shape[:-1]
                       if getattr(x_term, "typ", None) is not None
                       else ())
            outer = _kron_member(eg, x_eid, terms,
                                 (m1, n1, m2, n2, x_batch, K))
            if len(node.children) == 3:
                outer = eg.add_enode("add",
                                     (outer, node.children[2]))
            offer_term = eg.any_term(outer)
            src_term = eg._oldest_term(c) or eg.any_term(c)
            wit = None
            if witness and offer_term is not None and src_term is not None:
                wit = Rewrite(
                    name=f"eps_kron#{outer}",
                    lhs=src_term, rhs=offer_term,
                    law=(f"Kronecker-sum factorisation of {wt.name}: "
                         f"W ≈ Σ_{K} Aᵢ⊗Bᵢ, rearranged SVD residual "
                         f"‖W−Ŵ‖_F = {resid:.3e} (exact)"),
                    error_bound=resid, bound_norm="frobenius")
            eg.union(c, outer, witness=wit,
                     note=(f"eps_kron: {wt.name} ({o}x{i}) -> "
                           f"{K} terms ({m1}x{n1})x({m2}x{n2}), "
                           f"ε_F={resid:.3e}"))
            offers.append({"name": wt.name, "K": K, "bound": resid,
                           "stored": stored, "original": o * i,
                           "site_eid": c,
                           "factors": (m1, n1, m2, n2)})
    return offers


def _shape_of(t: Any):
    from catopt.cost import _shape_of as _so
    return _so(t)


def low_rank_params(eg: EGraph, source_tensors: dict, *,
                    rtol: float = 0.05,
                    min_saving: float = 0.8,
                    witness: bool = True) -> list[dict]:
    """Offer ``linear(linear(x, V_r), U_rΣ_r[, b])`` at each
    ``linear(x, W[, b])`` site whose truncated SVD (i) fits the
    relative spectral budget ``σ_{r+1} ≤ rtol·σ_max`` and (ii) stores
    fewer values: ``r(o+i) < min_saving·o·i``.

    Mutates ``source_tensors`` with the derived factor tensors
    (``eps_u_*``/``eps_v_*`` names) — they are genuine parameters of
    the optimized module.  Returns one dict per offered
    factorisation: ``{name, rank, bound, stored, original, site_eid}``.
    """
    offers: list[dict] = []
    # linear e-nodes whose weight child is a Param leaf in source_tensors
    for cid in list(eg._classes.keys()):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for node in list(ec.nodes):
            if node.op != "linear" or len(node.children) not in (2, 3):
                continue
            w_eid = eg.find(node.children[1])
            wt = eg.any_term(w_eid)
            if not isinstance(wt, Param) or wt.name not in source_tensors:
                continue
            W = source_tensors[wt.name]
            if not (isinstance(W, torch.Tensor) and W.ndim == 2):
                continue
            o, i = W.shape
            Wd = W.detach().double()
            try:
                U, S, Vt = torch.linalg.svd(Wd, full_matrices=False)
            except Exception:
                continue
            if S.numel() == 0 or S[0] == 0:
                continue
            budget = rtol * S[0]
            tail = torch.cat([S, S.new_zeros(1)])
            below = (tail[1:] <= budget).nonzero()
            if len(below) == 0:
                continue
            r = min(int(below[0].item()), S.numel() - 1)
            if r < 1 or r * (o + i) >= min_saving * o * i:
                continue
            bound = float(tail[r])        # σ_{r+1}: Eckart–Young
            # linear(linear(x, V), UΣ) : V is (r,i) applied first,
            # then UΣ is (o,r) — each stored as a factor param.
            Ur = (U[:, :r] * S[:r]).to(W.dtype)
            Vr = Vt[:r, :].to(W.dtype)
            uname = f"eps_u_{wt.name}_{c}"
            vname = f"eps_v_{wt.name}_{c}"
            source_tensors[uname] = Ur
            source_tensors[vname] = Vr
            x_eid = node.children[0]
            V_t = Param(vname, TensorType((r, i)))
            U_t = Param(uname, TensorType((o, r)))
            inner = eg.add_enode("linear", (x_eid, eg.add_term(V_t)))
            outer_children = [inner, eg.add_term(U_t)]
            if len(node.children) == 3:
                outer_children.append(node.children[2])
            outer = eg.add_enode("linear", tuple(outer_children),
                                 dict(node.attrs))
            offer_term = eg.any_term(outer)
            src_term = eg._oldest_term(c) or eg.any_term(c)
            wit = None
            if witness and offer_term is not None and src_term is not None:
                wit = Rewrite(
                    name=f"eps_lr#{outer}",
                    lhs=src_term, rhs=offer_term,
                    law=(f"truncated-SVD factorisation of {wt.name}: "
                         f"‖W − UΣVᵀ‖₂ = σ_{r + 1} = {bound:.3e} "
                         "(exact Eckart–Young bound; output error at "
                         "this site ≤ bound·‖x‖₂)"),
                    error_bound=bound, bound_norm="spectral")
            eg.union(c, outer, witness=wit,
                     note=(f"eps_low_rank: {wt.name} ({o}x{i}) "
                           f"-> rank {r}, ε={bound:.3e}"))
            offers.append({"name": wt.name, "rank": r, "bound": bound,
                           "stored": r * (o + i), "original": o * i,
                           "site_eid": c})
    return offers
