"""The ε axis — certified approximations inside the e-graph.

Optional toolkit: off by default and not part of the core optimizer —
it runs only when ``optimize_model(eps_rtol=…)`` is set or these
functions are called directly.  Phase-5 showed norm bounds do not
predict task quality on trained checkpoints, so this module is kept
for where a norm bound IS the contract (activation paths,
verification, certified deployment).

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

import math
from typing import Any

import torch

from catopt.egraph import EGraph, Rewrite
from catopt.ir import Const, Op, Param, TensorType

__all__ = [
    "kron_linear_params",
    "low_rank_gather",
    "low_rank_params",
    "model_bound",
    "optimize_weight",
    "quant_params",
]


def optimize_weight(
    name: str,
    W: torch.Tensor,
    *,
    rtol: float = 0.05,
    bits: int = 8,
    rules=None,
    witness: bool = True,
) -> dict:
    """A weight *is* a program — searched by catopt itself.

    Builds an e-graph over the leaf ``Param(name)``, offers every
    certified compressed representation the ε-passes know — low-rank
    ``matmul(U_r, V_rᵀ)``, sum-of-Kronecker (executable
    ``reshape/transpose/matmul`` chains), int8 ``mul(float(q), s)`` —
    then saturates with the ordinary rule set (the offers are
    *members*, rewritten by the same laws as any other term) and
    extracts the cheapest under ``param_bytes_cost(by_bytes=True)``.

    Returns ``{term, certificate, source_tensors, bound, bytes,
    offers}`` — the cheapest program for ``W`` plus the proof of how
    wrong it is.

    Caveat: representation search, not task-aware compression —
    Phase-5 showed norm bounds do not predict perplexity.  Use where a
    norm bound IS the contract (activation paths, certified
    deployment, exact structural sharing).
    """
    from catopt.cost import param_bytes_cost_for
    from catopt.rules import all_rules

    src: dict = {name: W}
    eg = EGraph()
    typ = TensorType(tuple(W.shape))
    leaf = Param(name, typ)
    root = eg.add_term(leaf)
    offers = []

    if W.ndim == 2 and W.is_floating_point():
        o, i = W.shape
        Wd = W.detach().double()
        try:
            U, S, Vt = torch.linalg.svd(Wd, full_matrices=False)
        except Exception:
            S = None
        # ---- low-rank offer -----------------------------------------
        if S is not None and S.numel() > 1 and S[0] > 0:
            budget = rtol * float(S[0])
            tail = torch.cat([S, S.new_zeros(1)])
            below = (tail[1:] <= budget).nonzero()
            if len(below):
                r = int(below[0].item()) + 1
                if r * (o + i) < 0.9 * o * i:
                    bound = float(tail[r])
                    un, vn = f"w_{name}_u", f"w_{name}_v"
                    src[un] = (U[:, :r] * S[:r]).to(W.dtype)
                    src[vn] = Vt[:r].contiguous().to(W.dtype)
                    member = Op.make(
                        "matmul",
                        Param(un, TensorType((o, r))),
                        Param(vn, TensorType((r, i))),
                    )
                    meid = eg.add_term(member)
                    wit = (
                        Rewrite(
                            name=f"wlr#{meid}",
                            lhs=leaf,
                            rhs=member,
                            law=f"W ≈ U_rΣVᵀ, ‖ΔW‖₂ = σ_{r + 1}",
                            error_bound=bound,
                            bound_norm="spectral",
                        )
                        if witness
                        else None
                    )
                    eg.union(root, meid, witness=wit)
                    offers.append(("lowrank", r, bound))
        # ---- Kronecker-sum offer (executable form) -------------------
        # A⊗B = reshape(transpose(reshape(Avec@Bvecᵀ,(m1,n1,m2,n2)),
        #               dims 1↔2), (o,i))
        for m1 in range(2, int(o**0.5) + 2):
            if o % m1:
                continue
            m2 = o // m1
            for n1 in range(2, int(i**0.5) + 2):
                if i % n1:
                    continue
                n2 = i // n1
                R = (
                    Wd.reshape(m1, m2, n1, n2)
                    .permute(0, 2, 1, 3)
                    .reshape(m1 * n1, m2 * n2)
                )
                try:
                    Ur, S2, Vr = torch.linalg.svd(
                        R, full_matrices=False
                    )
                except Exception:
                    continue
                e = torch.cumsum(S2**2, 0) / (S2**2).sum()
                ok = (1 - e <= rtol**2).nonzero()
                if not len(ok):
                    continue
                K = int(ok[0].item()) + 1
                if K * (m1 * n1 + m2 * n2) >= 0.9 * o * i:
                    continue
                resid = float(torch.sqrt((S2[K:] ** 2).sum()))
                acc = None
                for t in range(K):
                    an, bn = f"w_{name}_k{t}a", f"w_{name}_k{t}b"
                    src[an] = (
                        (Ur[:, t] * S2[t])
                        .reshape(m1, n1)
                        .to(W.dtype)
                        .contiguous()
                    )
                    src[bn] = (
                        Vr[t].reshape(m2, n2).to(W.dtype).contiguous()
                    )
                    kt = Op.make(
                        "reshape",
                        Op.make(
                            "transpose",
                            Op.make(
                                "reshape",
                                Op.make(
                                    "matmul",
                                    Op.make(
                                        "reshape",
                                        Param(an, TensorType((m1, n1))),
                                        shape=(m1 * n1, 1),
                                    ),
                                    Op.make(
                                        "reshape",
                                        Param(bn, TensorType((m2, n2))),
                                        shape=(1, m2 * n2),
                                    ),
                                ),
                                shape=(m1, n1, m2, n2),
                            ),
                            arg1=1,
                            arg2=2,
                        ),
                        shape=(o, i),
                    )
                    acc = kt if acc is None else Op.make("add", acc, kt)
                meid = eg.add_term(acc)
                wit = (
                    Rewrite(
                        name=f"wkron#{meid}",
                        lhs=leaf,
                        rhs=acc,
                        law=f"W ≈ Σ_{K} Aᵢ⊗Bᵢ, ‖ΔW‖_F = {resid:.3e}",
                        error_bound=resid,
                        bound_norm="frobenius",
                    )
                    if witness
                    else None
                )
                eg.union(root, meid, witness=wit)
                offers.append(("kron", K, resid))
                break
            else:
                continue
            break
        # ---- quantization offer --------------------------------------
        amax = float(Wd.abs().max())
        if amax > 0:
            lv = 2 ** (bits - 1) - 1
            s = amax / lv
            q = torch.clip(torch.round(Wd / s), -lv - 1, lv)
            qname = f"w_{name}_q{bits}"
            src[qname] = q.to(torch.int8)
            bound = float(s / 2 * math.sqrt(W.numel()))
            member = Op.make(
                "mul",
                Op.make(
                    "float",
                    Param(qname, typ),
                    dtype=str(W.dtype).split(".")[-1],
                ),
                Const(float(s)),
            )
            meid = eg.add_term(member)
            wit = (
                Rewrite(
                    name=f"wquant#{meid}",
                    lhs=leaf,
                    rhs=member,
                    law=f"W ≈ int{bits}·s, ‖ΔW‖_F ≤ (s/2)·√n",
                    error_bound=bound,
                    bound_norm="frobenius",
                )
                if witness
                else None
            )
            eg.union(root, meid, witness=wit)
            offers.append(("quant", bits, bound))

    # saturate the generator programs themselves with the ordinary
    # laws — the offers are members, not endpoints
    eg.run(
        all_rules() if rules is None else rules, root, max_iterations=3
    )
    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    cert = eg.certificate(leaf, term, root_eid=root)
    nbytes = param_bytes_cost_for(src, by_bytes=True)(term)
    return {
        "term": term,
        "certificate": cert,
        "source_tensors": src,
        "bound": cert.error_bound,
        "bytes": int(nbytes),
        "orig_bytes": int(W.numel() * W.element_size()),
        "offers": offers,
    }


_LIP_ELEM = {
    "sigmoid": 0.25,
    "tanh": 1.0,
    "relu": 1.0,
    "silu": 1.1,
    "gelu": 1.13,
    "exp": None,
}
_LIP_FREE = {
    "add",
    "sub",
    "neg",
    "reshape",
    "transpose",
    "view",
    "contiguous",
    "broadcast",
    "concat",
    "cat",
    "stack",
    "index_select",
    "select",
    "float",
    "to",
    "alias",
    "embedding",
}


def _term_spectral(t: Any, source_tensors: dict) -> float | None:
    """Spectral norm of a term when determinable: Param -> its tensor;
    Const -> |value|; param-only Op -> evaluate then measure."""
    if isinstance(t, Param):
        W = source_tensors.get(t.name)
        if isinstance(W, torch.Tensor):
            if W.ndim == 2:
                return float(torch.linalg.norm(W.detach().double(), 2))
            return float(W.detach().abs().max())
        return None
    if isinstance(t, Const):
        return abs(float(t.value))
    if isinstance(t, Op):
        # only fold if every leaf is a param/const
        leaves = _leaves(t)
        if all(isinstance(l, (Param, Const)) for l in leaves):
            try:
                from catopt.torch_bridge import _IR_TO_TORCH

                env = {
                    l.name: source_tensors[l.name]
                    for l in leaves
                    if isinstance(l, Param) and l.name in source_tensors
                }

                def ev(x):
                    if isinstance(x, Param):
                        return env[x.name]
                    if isinstance(x, Const):
                        return torch.tensor(x.value)
                    fn = _IR_TO_TORCH.get(x.op)
                    if fn is None:
                        return None
                    args = [ev(a) for a in x.args]
                    if any(a is None for a in args):
                        return None
                    return fn(*args, **dict(x.attrs))

                v = ev(t)
                if isinstance(v, torch.Tensor):
                    return (
                        float(torch.linalg.norm(v.double(), 2))
                        if v.ndim == 2
                        else float(v.abs().max())
                    )
            except Exception:
                return None
    return None


def _leaves(t: Any) -> list:
    if isinstance(t, Op):
        out = []
        for a in t.args:
            out.extend(_leaves(a))
        return out
    return [t]


def _lip_wrt(
    node_op: str,
    i: int,
    children: list,
    source_tensors: dict,
    act_norm: float | None = None,
) -> float | None:
    """Upper bound on ‖∂f/∂child_i‖ — Lipschitz constant of an op w.r.t.
    one input, using spectral norms of the *sibling* operands.
    ``None`` = unknown/unbounded → the caller reports ∞.

    Weight-side edges (``linear(x,W)`` wrt W) are data-dependent:
    ``‖Δy‖ ≤ ‖ΔW‖·‖x‖``.  When ``act_norm`` bounds the sibling
    activation's norm they resolve to a finite constant."""
    if node_op in _LIP_FREE:
        return 1.0
    if node_op in _LIP_ELEM:
        return _LIP_ELEM[node_op]
    if node_op in ("mul", "div"):
        sib = children[1 - i] if len(children) == 2 else None
        return (
            _term_spectral(sib, source_tensors)
            if sib is not None
            else None
        )
    if node_op in ("matmul", "linear", "conv2d"):
        if len(children) >= 2:
            if i == 0:
                return _term_spectral(children[1], source_tensors)
            # weight side: multiplier is the activation norm
            return act_norm
    if node_op == "softmax":
        return 1.0
    if node_op == "sdpa":
        return None
    return None


def _input_sensitivity(
    term: Any,
    source_tensors: dict,
    input_norm: float,
    _path=(),
    _best=0.0,
) -> float:
    """Max over Var-leaf paths of the Lipschitz product — an upper
    bound on ``‖term(x)‖ ≤ input_norm × this`` (activation norm bound).
    """
    from catopt.ir import Var as _Var

    if isinstance(term, _Var):
        return max(_best, 1.0)
    if isinstance(term, Op):
        best = _best
        for i, a in enumerate(term.args):
            lip = _lip_wrt(
                term.op, i, list(term.args), source_tensors, input_norm
            )
            if lip is None:
                continue
            best = max(
                best,
                _input_sensitivity(a, source_tensors, input_norm) * lip,
            )
        return best
    return _best


def _path_sensitivity(
    term: Any,
    path: tuple,
    source_tensors: dict,
    input_norm: float | None = None,
) -> float | None:
    """Product of per-op Lipschitz constants along ``path`` — the
    multiplier from a perturbation at that subterm to the output.
    Data-dependent weight-side edges resolve via ``input_norm`` × the
    input-to-activation sensitivity when provided."""
    sens = 1.0
    t = term
    for i in path:
        if not isinstance(t, Op) or i >= len(t.args):
            return None
        lip = _lip_wrt(t.op, i, list(t.args), source_tensors)
        if (
            lip is None
            and input_norm is not None
            and t.op in ("linear", "matmul", "conv2d")
            and i == 1
        ):
            # weight perturbation: ‖Δy‖ ≤ ‖ΔW‖·‖activation‖ and the
            # activation is bounded by input_norm × input sensitivity
            act = _input_sensitivity(
                t.args[0], source_tensors, input_norm
            )
            lip = input_norm * act if act > 0 else None
        if lip is None:
            return None
        sens *= lip
        t = t.args[i]
    return sens


def _find_subterms(term: Any, target: Any, _path=(), _acc=None):
    """All paths at which ``target`` occurs structurally in ``term``."""
    if _acc is None:
        _acc = []
    if term == target:
        _acc.append(_path)
    if isinstance(term, Op):
        for i, a in enumerate(term.args):
            _find_subterms(a, target, _path + (i,), _acc)
    return _acc


def model_bound(
    root_term: Any,
    cert: Any,
    source_tensors: dict,
    input_norm: float | None = None,
) -> dict:
    """Whole-model certified error bound for an extracted term.

    For each bound-carrying certificate step, locate its produced
    subterm in the final term and multiply the step's local bound by
    the Lipschitz sensitivity of that position.  Steps whose site
    cannot be located (further-rewritten members) contribute at
    program-Lipschitz strength — reported separately as 'unlocated'
    so the total stays honest (inf when unbounded ops intervene).
    """
    total = 0.0
    unlocated = 0.0
    contributions = []
    for step in cert.steps:
        rule = cert.rules.get(step.rule)
        if rule is None or not rule.error_bound:
            continue
        paths = _find_subterms(root_term, step.rhs)
        if not paths:
            unlocated += rule.error_bound
            contributions.append(
                {
                    "rule": step.rule,
                    "bound": rule.error_bound,
                    "site_sensitivity": None,
                }
            )
            continue
        for p in paths:
            s = _path_sensitivity(
                root_term, p, source_tensors, input_norm
            )
            contrib = (
                rule.error_bound * s if s is not None else float("inf")
            )
            total += contrib
            contributions.append(
                {
                    "rule": step.rule,
                    "bound": rule.error_bound,
                    "path": p,
                    "site_sensitivity": s,
                    "contribution": contrib,
                }
            )
    return {
        "bound": total + unlocated,
        "site_contributions": contributions,
        "n_bounded_steps": sum(
            1
            for s in cert.steps
            if cert.rules.get(s.rule) and cert.rules[s.rule].error_bound
        ),
    }


def quant_params(
    eg: EGraph,
    source_tensors: dict,
    *,
    bits: int = 8,
    per_channel: bool = False,
    witness: bool = True,
) -> list[dict]:
    """Quantization-as-ε: offer ``mul(float(W_q), s)`` for every Param
    leaf in the e-graph, where ``W_q`` is per-tensor symmetric int
    (``bits``) and ``s = absmax/levels``.

    Certified bound: each entry rounds off by ≤ s/2, so
    ``‖W − Ŵ‖_F ≤ s/2·√numel`` — Frobenius, exact.  The quantized
    tensor is stored as its integer dtype (``by_bytes`` pricing sees
    the width reduction; ``param_bytes_cost``'s default value count
    does not).  ``float()`` restores the working dtype at use sites.

    This is the third member of the unified object: low-rank, tying,
    and quantization are all rewrites with an error bound.
    """
    levels = 2 ** (bits - 1) - 1
    dtype = {8: torch.int8, 4: torch.int8}[bits]  # int4 stored as int8
    offers: list[dict] = []
    seen: set[str] = set()
    for cid in list(eg._classes.keys()):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for node in list(ec.nodes):
            if node.op != "leaf":
                continue
            wt = eg.any_term(c)
            if not (
                isinstance(wt, Param)
                and wt.name in source_tensors
                and wt.name not in seen
            ):
                continue
            seen.add(wt.name)
            W = source_tensors[wt.name]
            if not (
                isinstance(W, torch.Tensor)
                and W.is_floating_point()
                and W.numel() > 1
            ):
                continue
            Wd = W.detach()
            flat = (
                Wd.reshape(Wd.shape[0], -1)
                if (per_channel and Wd.ndim >= 2)
                else None
            )
            if flat is not None:
                amax_r = flat.abs().amax(-1, keepdim=True)
                sc = torch.where(
                    amax_r == 0,
                    torch.ones_like(amax_r),
                    amax_r / levels,
                )
                q = (
                    torch.round(flat / sc)
                    .clamp(-levels - 1, levels)
                    .to(dtype)
                )
                qname = f"eps_q{bits}c_{wt.name}"
                sname = f"eps_s{bits}c_{wt.name}"
                source_tensors[qname] = q.reshape(W.shape)
                source_tensors[sname] = sc.reshape(
                    *W.shape[:1], *([1] * (W.ndim - 1))
                ).to(W.dtype)
                member = Op.make(
                    "mul",
                    Op.make(
                        "float",
                        Param(qname, wt.typ),
                        dtype=str(W.dtype).split(".")[-1],
                    ),
                    Param(
                        sname,
                        TensorType(tuple(source_tensors[sname].shape)),
                    ),
                )
            else:
                amax = float(Wd.abs().max())
                if amax == 0:
                    continue
                s = amax / levels
                q = (
                    torch.round(Wd / s)
                    .clamp(-levels - 1, levels)
                    .to(dtype)
                )
                qname = f"eps_q{bits}_{wt.name}"
                source_tensors[qname] = q
                member = Op.make(
                    "mul",
                    Op.make(
                        "float",
                        Param(qname, wt.typ),
                        dtype=str(W.dtype).split(".")[-1],
                    ),
                    Const(float(s)),
                )
            member_eid = eg.add_term(member)
            if flat is not None:
                # per-row: ‖ΔW‖_F ≤ (√n_cols/2)·‖s‖₂
                bound = float(
                    (flat.shape[1] ** 0.5)
                    / 2
                    * torch.linalg.norm(sc.squeeze(-1))
                )
                law = (
                    f"per-channel int{bits} of {wt.name}: "
                    f"‖W−Ŵ‖_F ≤ (√n/2)·‖s‖₂ = {bound:.3e}"
                )
            else:
                bound = s / 2 * (W.numel() ** 0.5)
                law = (
                    f"symmetric int{bits} quantization of "
                    f"{wt.name}: ‖W−Ŵ‖_F ≤ (s/2)·√n = "
                    f"{bound:.3e} (s={s:.3e})"
                )
            wit = None
            if witness:
                wit = Rewrite(
                    name=f"eps_q{bits}#{member_eid}",
                    lhs=wt,
                    rhs=member,
                    law=law,
                    error_bound=bound,
                    bound_norm="frobenius",
                )
            eg.union(
                c,
                member_eid,
                witness=wit,
                note=(
                    f"eps_quant: {wt.name} -> int{bits}"
                    f"{'/chan' if flat is not None else ''}"
                ),
            )
            offers.append(
                {
                    "name": wt.name,
                    "bits": bits,
                    "bound": bound,
                    "stored": q.numel(),
                    "original": W.numel(),
                    "eid": member_eid,
                }
            )
    return offers


def low_rank_gather(
    eg: EGraph,
    source_tensors: dict,
    *,
    rtol: float = 0.05,
    min_saving: float = 0.8,
    witness: bool = True,
) -> list[dict]:
    """Low-rank factorisation at ``embedding``/row-gather sites —
    the biggest measured real-weight win (the token embedding is
    ~60% of stories15M and genuinely low-rank).

        embedding(W, idx)  →  matmul(embedding(U_r, idx), V_r)

    gathers ``r``-dimensional rows then projects — storage
    ``r(v+d)`` vs ``v·d``, bound ``σ_{r+1}`` spectral (Eckart–Young),
    propagated: ``‖gathered row − ŵ_j‖₂ ≤ σ_{r+1}``.
    """
    offers: list[dict] = []
    for cid in list(eg._classes.keys()):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for node in list(ec.nodes):
            if node.op != "embedding" or len(node.children) != 2:
                continue
            wt = eg.any_term(eg.find(node.children[0]))
            if (
                not isinstance(wt, Param)
                or wt.name not in source_tensors
            ):
                continue
            W = source_tensors[wt.name]
            if not (isinstance(W, torch.Tensor) and W.ndim == 2):
                continue
            v, d = W.shape
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
            r = min(int(below[0].item()) + 1, S.numel() - 1)
            if r < 1 or r * (v + d) >= min_saving * v * d:
                continue
            bound = float(tail[r])
            uname = f"eps_eu_{wt.name}_{c}"
            vname = f"eps_ev_{wt.name}_{c}"
            source_tensors[uname] = U[:, :r].contiguous().to(W.dtype)
            source_tensors[vname] = (
                (S[:r, None] * Vt[:r]).contiguous().to(W.dtype)
            )
            g = eg.add_enode(
                "embedding",
                (
                    eg.add_term(Param(uname, TensorType((v, r)))),
                    node.children[1],
                ),
                dict(node.attrs),
            )
            outer = eg.add_enode(
                "matmul",
                (g, eg.add_term(Param(vname, TensorType((r, d))))),
            )
            offer_term = eg.any_term(outer)
            src_term = eg._oldest_term(c) or eg.any_term(c)
            wit = None
            if (
                witness
                and offer_term is not None
                and src_term is not None
            ):
                wit = Rewrite(
                    name=f"eps_emb#{outer}",
                    lhs=src_term,
                    rhs=offer_term,
                    law=(
                        f"low-rank embedding of {wt.name}: "
                        f"W ≈ U_rΣ_rV_rᵀ, ‖W−Ŵ‖₂ = σ_{r + 1} = "
                        f"{bound:.3e} (Eckart–Young; per-row error "
                        "≤ bound)"
                    ),
                    error_bound=bound,
                    bound_norm="spectral",
                )
            eg.union(
                c,
                outer,
                witness=wit,
                note=(
                    f"eps_low_rank_gather: {wt.name} "
                    f"({v}x{d}) -> rank {r}, ε={bound:.3e}"
                ),
            )
            offers.append(
                {
                    "name": wt.name,
                    "rank": r,
                    "bound": bound,
                    "stored": r * (v + d),
                    "original": v * d,
                    "site_eid": c,
                }
            )
    return offers


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
        xr = eg.add_enode(
            "reshape", (x_eid,), {"shape": (*x_batch, n1, n2)}
        )
        ax = eg.add_enode("matmul", (eg.add_term(A_t), xr))
        axb = eg.add_enode(
            "matmul",
            (
                ax,
                eg.add_enode(
                    "transpose",
                    (eg.add_term(B_t),),
                    {"arg1": -2, "arg2": -1},
                ),
            ),
        )
        y = eg.add_enode(
            "reshape", (axb,), {"shape": (*x_batch, m1 * m2)}
        )
        acc = y if acc is None else eg.add_enode("add", (acc, y))
    return acc


def kron_linear_params(
    eg: EGraph,
    source_tensors: dict,
    *,
    rtol: float = 0.05,
    min_saving: float = 0.8,
    witness: bool = True,
) -> list[dict]:
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
            if (
                not isinstance(wt, Param)
                or wt.name not in source_tensors
            ):
                continue
            W = source_tensors[wt.name]
            if not (isinstance(W, torch.Tensor) and W.ndim == 2):
                continue
            o, i = W.shape
            Wd = W.detach().double()
            # choose the (m1,n1) split minimising storage at rtol
            best = None
            for m1 in range(2, int(o**0.5) + 2):
                if o % m1:
                    continue
                m2 = o // m1
                for n1 in range(2, int(i**0.5) + 2):
                    if i % n1:
                        continue
                    n2 = i // n1
                    R = (
                        Wd.reshape(m1, m2, n1, n2)
                        .permute(0, 2, 1, 3)
                        .reshape(m1 * n1, m2 * n2)
                    )
                    try:
                        Ur, S, Vr = torch.linalg.svd(
                            R, full_matrices=False
                        )
                    except Exception:
                        continue
                    if S[0] == 0:
                        continue
                    e = torch.cumsum(S**2, 0) / (S**2).sum()
                    K = int((e < 1 - rtol**2).sum().item()) + 1
                    K = min(K, S.numel())
                    stored = K * (m1 * n1 + m2 * n2)
                    if stored < min_saving * o * i and (
                        best is None or stored < best[0]
                    ):
                        resid = float(
                            torch.sqrt((S[K:] ** 2).sum()).item()
                        )
                        best = (
                            stored,
                            m1,
                            m2,
                            n1,
                            n2,
                            K,
                            resid,
                            Ur,
                            S,
                            Vr,
                        )
            if best is None:
                continue
            stored, m1, m2, n1, n2, K, resid, Ur, S, Vr = best
            terms = []
            for t in range(K):
                aname = f"eps_k{t}_a_{wt.name}_{c}"
                bname = f"eps_k{t}_b_{wt.name}_{c}"
                # R = UΣVᵀ term t :  A_t = reshape(U[:,t]·σ_t, (m1,n1))
                #                     B_t = reshape(V[t],   (m2,n2))
                source_tensors[aname] = (
                    (Ur[:, t] * S[t])
                    .reshape(m1, n1)
                    .to(W.dtype)
                    .contiguous()
                )
                source_tensors[bname] = (
                    Vr[t, :].reshape(m2, n2).to(W.dtype).contiguous()
                )
                terms.append(
                    (
                        Param(aname, TensorType((m1, n1))),
                        Param(bname, TensorType((m2, n2))),
                    )
                )
            x_eid = node.children[0]
            x_term = eg.any_term(x_eid)
            x_batch = (
                x_term.typ.shape[:-1]
                if getattr(x_term, "typ", None) is not None
                else ()
            )
            outer = _kron_member(
                eg, x_eid, terms, (m1, n1, m2, n2, x_batch, K)
            )
            if len(node.children) == 3:
                outer = eg.add_enode("add", (outer, node.children[2]))
            offer_term = eg.any_term(outer)
            src_term = eg._oldest_term(c) or eg.any_term(c)
            wit = None
            if (
                witness
                and offer_term is not None
                and src_term is not None
            ):
                wit = Rewrite(
                    name=f"eps_kron#{outer}",
                    lhs=src_term,
                    rhs=offer_term,
                    law=(
                        f"Kronecker-sum factorisation of {wt.name}: "
                        f"W ≈ Σ_{K} Aᵢ⊗Bᵢ, rearranged SVD residual "
                        f"‖W−Ŵ‖_F = {resid:.3e} (exact)"
                    ),
                    error_bound=resid,
                    bound_norm="frobenius",
                )
            eg.union(
                c,
                outer,
                witness=wit,
                note=(
                    f"eps_kron: {wt.name} ({o}x{i}) -> "
                    f"{K} terms ({m1}x{n1})x({m2}x{n2}), "
                    f"ε_F={resid:.3e}"
                ),
            )
            offers.append(
                {
                    "name": wt.name,
                    "K": K,
                    "bound": resid,
                    "stored": stored,
                    "original": o * i,
                    "site_eid": c,
                    "factors": (m1, n1, m2, n2),
                }
            )
    return offers


def _shape_of(t: Any):
    from catopt.cost import _shape_of as _so

    return _so(t)


def low_rank_params(
    eg: EGraph,
    source_tensors: dict,
    *,
    rtol: float = 0.05,
    min_saving: float = 0.8,
    witness: bool = True,
) -> list[dict]:
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
            if (
                not isinstance(wt, Param)
                or wt.name not in source_tensors
            ):
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
            r = min(int(below[0].item()) + 1, S.numel() - 1)
            if r < 1 or r * (o + i) >= min_saving * o * i:
                continue
            bound = float(tail[r])  # σ_{r+1}: Eckart–Young
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
            outer = eg.add_enode(
                "linear", tuple(outer_children), dict(node.attrs)
            )
            offer_term = eg.any_term(outer)
            src_term = eg._oldest_term(c) or eg.any_term(c)
            wit = None
            if (
                witness
                and offer_term is not None
                and src_term is not None
            ):
                wit = Rewrite(
                    name=f"eps_lr#{outer}",
                    lhs=src_term,
                    rhs=offer_term,
                    law=(
                        f"truncated-SVD factorisation of {wt.name}: "
                        f"‖W − UΣVᵀ‖₂ = σ_{r + 1} = {bound:.3e} "
                        "(exact Eckart–Young bound; output error at "
                        "this site ≤ bound·‖x‖₂)"
                    ),
                    error_bound=bound,
                    bound_norm="spectral",
                )
            eg.union(
                c,
                outer,
                witness=wit,
                note=(
                    f"eps_low_rank: {wt.name} ({o}x{i}) "
                    f"-> rank {r}, ε={bound:.3e}"
                ),
            )
            offers.append(
                {
                    "name": wt.name,
                    "rank": r,
                    "bound": bound,
                    "stored": r * (o + i),
                    "original": o * i,
                    "site_eid": c,
                }
            )
    return offers
