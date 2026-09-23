"""Online-softmax monoid — the nonlinear analogue of the scan carrier.

The affine-map monoid (``SCAN_LAWS`` in :mod:`catopt.rules`) showed that
lifting a recurrence into a monoid carrier lets plain associativity
discover parallel bracketings.  The same trick works for the softmax
recurrence — FlashAttention's running (max, sum, numerator) triple.

CARRIER
    ``om(m, l, a)`` is a triple value: running row-max ``m``, running
    denominator ``l = Σ exp(s − m)``, running numerator
    ``a = Σ exp(s − m) · v``.

COMPOSITION (associative *and* commutative)
    ``(m1,l1,a1) ⊕ (m2,l2,a2)``
        = (mx, l1·e^(m1−mx) + l2·e^(m2−mx), a1·e^(m1−mx) + a2·e^(m2−mx))
    with mx = max(m1, m2).  We deliberately do NOT add a commutativity
    rule — comm is the explosive law (permutation space), exactly as in
    SCAN_LAWS; OM_ASSOC/OM_ASSOC_REV suffice for reassociation.

ELEMENT
    ``om_elem(s, v)`` for score block ``s (…,T,K)`` and value block
    ``v (…,K,d)``: ``(rowmax(s), rowsum(exp(s−rowmax)), exp(s−rowmax)@v)``.

HOMOMORPHISM
    ``elem(cat(s1,s2), cat(v1,v2)) = elem(s1,v1) ⊕ elem(s2,v2)`` —
    and ``softmax(s) @ v = a / l``.  This is the law tensor algebra
    cannot state: softmax does not distribute over concat, but its
    monoid carrier does.

``om_apply(f)`` evaluates a carrier back to tensor-land: ``a / l``
(unclamped — a fully-masked row yields NaN exactly like dense softmax).

The rules:
* ``OM_LIFT`` / ``OM_UNLIFT``   — enter / leave the carrier.
* ``OM_SPLIT`` / ``OM_MERGE``   — the homomorphism, both directions.
* ``OM_ASSOC`` / ``OM_ASSOC_REV`` — reassociating the combine gives the
  chunked/blocked attention schedules (FlashAttention tiling is a
  bracketing of this monoid product).
* ``CONCAT_BINARIZE_*``         — n-ary concat → nested binary (the IR's
  concat is variadic; the homomorphism is binary).
* ``MATMUL_T_CONCAT``           — ``q @ cat(k1,k2).T = cat(q@k1.T, q@k2.T)``:
  matmul distributes over concat in the transposed operand, exposing
  per-block score tensors for OM_SPLIT.

Note on concat attrs: torch.export emits ``cat`` with the dim as the
positional ``arg1`` attribute, while rule-produced concats (and
hand-built terms, following rules.py convention) use ``dim``.  LHS
patterns are generated in both spellings; all RHS-produced concats use
``dim``.

MASKED ATTENTION
    A causal/additive mask wraps the score concat in ``masked_fill`` /
    ``add`` / ``where`` and blocks OM_SPLIT.  The law that unblocks it
    is that masking distributes over concat (see the section below):
    each block takes the matching *slice* of the mask, and for causal
    masks the block's positional offset lives inside that slice.
"""

from typing import Any

from catopt.egraph import Rewrite
from catopt.ir import Op
from catopt.rules import R


def _shape_of(t: Any):
    """Best-effort shape of a bound term (delegates to cost model)."""
    from catopt.cost import _shape_of as _so
    return _so(t)


def _dim_eq(a: Any, b: Any) -> bool:
    """cat-compatibility on a non-concatenated axis (None = unknown)."""
    return a is None or b is None or a == b


def _broadcast_ok(a, b) -> bool:
    from catopt.cost import _broadcast, _INVALID
    return _broadcast(a, b) is not _INVALID


# ---------------------------------------------------------------------------
#  Lift / unlift — enter and leave the carrier
# ---------------------------------------------------------------------------

def _check_om_lift(bound: dict) -> bool:
    """softmax must be over the scores' LAST dim (the key axis), and the
    matmul must contract s[...,K] with v[...,K,d].  A softmax over any
    other axis is well-typed but a *different* program — the check is
    load-bearing."""
    sd = bound.get("$attr:SD", -1)
    ss, vs = _shape_of(bound.get("s")), _shape_of(bound.get("v"))
    if not (isinstance(ss, tuple) and isinstance(vs, tuple)
            and len(ss) >= 2 and len(vs) >= 2):
        return False
    if not isinstance(sd, int) or sd % len(ss) != len(ss) - 1:
        return False
    return _dim_eq(ss[-1], vs[-2])


def _om_lift(name: str, attr_key: str | None) -> Rewrite:
    sm = (Op.make("softmax", "s", **{attr_key: "SD"})
          if attr_key is not None else Op.make("softmax", "s"))
    return R(
        name,
        Op.make("matmul", sm, "v"),
        Op.make("om_apply", Op.make("om_elem", "s", "v")),
        law="matmul(softmax(s), v) lifts into the online-softmax monoid: "
            "softmax does not distribute over concat, but elem does — "
            "the carrier (m, l, a) is what distributes.",
        check=_check_om_lift)


#: torch.export emits softmax(x, dim) with the dim in ``arg1``.
OM_LIFT = _om_lift("om_lift", "arg1")
#: Same law for the ``dim=`` kwarg spelling and a bare softmax(-1).
OM_LIFT_DIM = _om_lift("om_lift_dim", "dim")
OM_LIFT_PLAIN = _om_lift("om_lift_plain", None)

OM_UNLIFT = R(
    "om_unlift",
    Op.make("om_apply", Op.make("om_elem", "s", "v")),
    Op.make("matmul", Op.make("softmax", "s", arg1=-1), "v"),
    law="om_apply(elem(s,v)) unfolds back to dense softmax(s) @ v.",
)


# ---------------------------------------------------------------------------
#  The homomorphism — elem distributes over concat
# ---------------------------------------------------------------------------

def _chunks_compatible(s1, s2, v1, v2) -> bool:
    """Chunk pair i must contract s_i[...,K_i] with v_i[...,K_i,d]; the
    two chunks must be cat-compatible off the concatenated axis, and
    each chunk's batch dims must broadcast s_i against v_i."""
    ns, nv = len(s1), len(v1)
    if ns < 2 or nv < 2 or len(s2) != ns or len(v2) != nv:
        return False
    if not all(_dim_eq(s1[i], s2[i]) for i in range(ns - 1)):
        return False
    if not all(_dim_eq(v1[i], v2[i])
               for i in range(nv) if i != nv - 2):
        return False
    if not (_dim_eq(s1[-1], v1[nv - 2])
            and _dim_eq(s2[-1], v2[nv - 2])):
        return False
    return (_broadcast_ok(s1[:-2], v1[:-2])
            and _broadcast_ok(s2[:-2], v2[:-2]))


def _check_om_concat_dims(bound: dict) -> bool:
    """OM_SPLIT fires only when scores concat on the LAST dim (keys)
    and values concat on dim -2 (the same key axis, pre-contraction).
    A cat along any other axis is well-typed but WRONG."""
    sd, vd = bound.get("$attr:SD"), bound.get("$attr:VD")
    s1, s2 = _shape_of(bound.get("s1")), _shape_of(bound.get("s2"))
    v1, v2 = _shape_of(bound.get("v1")), _shape_of(bound.get("v2"))
    if not all(isinstance(x, tuple) for x in (s1, s2, v1, v2)):
        return False
    if not (isinstance(sd, int) and isinstance(vd, int)):
        return False
    if sd % len(s1) != len(s1) - 1:
        return False  # scores must concat along the key axis
    if vd % len(v1) != len(v1) - 2:
        return False  # values concat along the key axis (-2 before @)
    return _chunks_compatible(s1, s2, v1, v2)


def _om_split(name: str, attr_key: str) -> Rewrite:
    return R(
        name,
        Op.make("om_elem",
                Op.make("concat", "s1", "s2", **{attr_key: "SD"}),
                Op.make("concat", "v1", "v2", **{attr_key: "VD"})),
        Op.make("om_compose",
                Op.make("om_elem", "s1", "v1"),
                Op.make("om_elem", "s2", "v2")),
        law="Homomorphism: elem(cat(s1,s2), cat(v1,v2)) = "
            "elem(s1,v1) ⊕ elem(s2,v2) — the law tensor algebra cannot "
            "state (softmax does not distribute over concat).",
        check=_check_om_concat_dims)


OM_SPLIT = _om_split("om_split", "dim")
OM_SPLIT_ARG1 = _om_split("om_split_arg1", "arg1")


def _derive_om_concat_dims(bound: dict) -> dict | None:
    """Concat dims for the merged element, computed from bound shapes:
    scores join on their last dim, values on dim -2.  Vetoes (returns
    None) when the two chunks cannot form a well-typed cat."""
    s1, s2 = _shape_of(bound.get("s1")), _shape_of(bound.get("s2"))
    v1, v2 = _shape_of(bound.get("v1")), _shape_of(bound.get("v2"))
    if not all(isinstance(x, tuple) for x in (s1, s2, v1, v2)):
        return None
    if not _chunks_compatible(s1, s2, v1, v2):
        return None
    return {"$attr:SD": len(s1) - 1, "$attr:VD": len(v1) - 2}


OM_MERGE = R(
    "om_merge",
    Op.make("om_compose",
            Op.make("om_elem", "s1", "v1"),
            Op.make("om_elem", "s2", "v2")),
    Op.make("om_elem",
            Op.make("concat", "s1", "s2", dim="SD"),
            Op.make("concat", "v1", "v2", dim="VD")),
    law="Reverse homomorphism: two block elements merge into the "
        "element of the concatenated block — chunking is reversible.",
    derive=_derive_om_concat_dims)


# ---------------------------------------------------------------------------
#  Associativity — the monoid law that generates blocked schedules
# ---------------------------------------------------------------------------

OM_ASSOC = R(
    "om_assoc",
    Op.make("om_compose",
            Op.make("om_compose", "f", "g"), "h"),
    Op.make("om_compose", "f",
            Op.make("om_compose", "g", "h")),
    law="⊕ is associative: (f ⊕ g) ⊕ h = f ⊕ (g ⊕ h).  Every "
        "bracketing of the block product is a legal attention schedule.")

OM_ASSOC_REV = R(
    "om_assoc_rev",
    Op.make("om_compose", "f",
            Op.make("om_compose", "g", "h")),
    Op.make("om_compose",
            Op.make("om_compose", "f", "g"), "h"),
    law="⊕ is associative: f ⊕ (g ⊕ h) = (f ⊕ g) ⊕ h.")


# ---------------------------------------------------------------------------
#  concat is variadic in the IR; the homomorphism is binary.
# ---------------------------------------------------------------------------

def _concat_binarize(n: int, attr_key: str) -> Rewrite:
    xs = [f"x{i}" for i in range(n)]
    rhs = Op.make("concat", xs[0], xs[1], dim="D")
    for x in xs[2:]:
        rhs = Op.make("concat", rhs, x, dim="D")
    return R(
        f"concat_binarize_{n}_{attr_key}",
        Op.make("concat", *xs, **{attr_key: "D"}),
        rhs,
        law="n-ary concat is iterated binary concat — cat is a binary "
            "tensor product applied repeatedly.")


#: 3..6-ary concats, both attr spellings.  Structural only — needs no
#: shape check (binarisation preserves well-typedness identically).
CONCAT_BINARIZE: list[Rewrite] = [
    _concat_binarize(n, ak)
    for n in range(3, 7) for ak in ("dim", "arg1")
]


# ---------------------------------------------------------------------------
#  matmul distributes over concat in the transposed operand
# ---------------------------------------------------------------------------

def _check_matmul_t_concat(bound: dict) -> bool:
    """q @ cat(k1,k2,dim).T splits only when the key concat is on the
    SEQUENCE axis (dim -2 of k, i.e. keys) and the transpose is exactly
    .T on the last two dims — so that after the transpose the concat
    lands on the scores' last dim."""
    kd, t1, t2 = (bound.get("$attr:KD"), bound.get("$attr:T1"),
                  bound.get("$attr:T2"))
    k1, k2 = _shape_of(bound.get("k1")), _shape_of(bound.get("k2"))
    q = _shape_of(bound.get("q"))
    if not all(isinstance(x, int) for x in (kd, t1, t2)):
        return False
    if not all(isinstance(s, tuple) for s in (k1, k2, q)):
        return False
    nk = len(k1)
    if nk < 2 or len(k2) != nk or len(q) < 2:
        return False
    if kd % nk != nk - 2:
        return False  # keys concat on the sequence axis, not features
    if {t1 % nk, t2 % nk} != {nk - 2, nk - 1}:
        return False  # transpose must be exactly .T (last two dims)
    if not all(_dim_eq(k1[i], k2[i])
               for i in range(nk) if i != nk - 2):
        return False
    # q[...,d] contracts with k.T[...,d,K] — k's LAST dim is d.
    if not _dim_eq(q[-1], k1[-1]):
        return False
    return _broadcast_ok(q[:-2], k1[:-2])


def _derive_score_concat_dim(bound: dict) -> dict | None:
    """The concat dim through the transpose: after .T the key axis is
    last in k.T and lands on the scores' last dim, at index
    rank(out)-1 = max(rank q, rank k) - 1."""
    q, k1 = _shape_of(bound.get("q")), _shape_of(bound.get("k1"))
    if not (isinstance(q, tuple) and isinstance(k1, tuple)):
        return None
    return {"$attr:SD": max(len(q), len(k1)) - 1}


def _matmul_t_concat(name: str, attr_key: str) -> Rewrite:
    return R(
        name,
        Op.make("matmul", "q",
                Op.make("transpose",
                        Op.make("concat", "k1", "k2", **{attr_key: "KD"}),
                        arg1="T1", arg2="T2")),
        Op.make("concat",
                Op.make("matmul", "q",
                        Op.make("transpose", "k1",
                                arg1="T1", arg2="T2")),
                Op.make("matmul", "q",
                        Op.make("transpose", "k2",
                                arg1="T1", arg2="T2")),
                dim="SD"),
        law="matmul distributes over concat in the transposed operand: "
            "q @ cat(k1,k2).T = cat(q@k1.T, q@k2.T) — the score blocks "
            "surface as one concat so OM_SPLIT can chunk the softmax.",
        check=_check_matmul_t_concat,
        derive=_derive_score_concat_dim)


MATMUL_T_CONCAT = _matmul_t_concat("matmul_t_concat", "dim")
MATMUL_T_CONCAT_ARG1 = _matmul_t_concat("matmul_t_concat_arg1", "arg1")


# ---------------------------------------------------------------------------
#  Masks distribute over concat — the law masked chunked attention needs.
#
#  OM_SPLIT needs the score operand to be a literal concat; a causal or
#  additive mask wraps that concat in ``masked_fill``/``add``/``where``
#  and the homomorphism cannot see through it.  The law that unblocks
#  it: masking COMMUTES with concatenation,
#
#      mask(cat(s1, s2), M) = cat(mask(s1, M1), mask(s2, M2))
#
#  where M_i is block i's slice of the full mask along the cat axis —
#  ``split(M, (K1, K2), d, i)`` (the IR's existing projection op; no new
#  generator is needed).  For a causal mask this slice IS the
#  positional dependency: block i's mask columns are the global mask's
#  columns [o_i, o_i + K_i), so the block offset lives inside the
#  slice — no index arithmetic appears in the rewrite.  When the mask
#  broadcasts along the cat axis (extent 1, or the axis absent from the
#  mask entirely) the SAME operand serves both blocks unsliced.
#
#  What this IR still cannot express (documented, not hacked):
#  * synthesising a block-local mask from positions — there are no
#    arange/iota/tril/ones generators, so the offset law
#    ``mask_i = lt(arange(K_i) + o_i, arange(T))`` cannot be written;
#    only slicing a MATERIALISED mask distributes;
#  * implicit masks — ``sdpa(..., is_causal=True)`` has no mask operand
#    to slice; the flag would need its own chunking law;
#  * masks that are not axis-aligned slices of the cat'd operand (e.g.
#    interleaved/block-diagonal layouts) — which don't arise from key
#    chunking anyway.
# ---------------------------------------------------------------------------

def _cat_axis_plan(bound: dict, sliced_key: str,
                   fixed_keys: tuple = ()) -> dict | None:
    """Plan how an elementwise op's operands distribute over
    ``concat(s1, s2, dim=D)``.

    The sliceable operand (``sliced_key`` — the mask/bias) is classified
    by its extent on the cat axis:

    * extent ``K1 + K2`` (the whole concatenated axis) → ``"slice"``:
      block i gets ``split(m, (K1, K2), md, i)`` — for causal masks the
      column slice is exactly the absolute-position offset;
    * extent 1, or the axis absent from the operand (lower rank /
      scalar) → ``"reuse"``: the operand broadcasts, unchanged, into
      both blocks;
    * anything else, or unknown extents → ``None`` (veto: the rule may
      not fire).

    ``fixed_keys`` are other operands that must broadcast along the cat
    axis to be reused (e.g. ``masked_fill``'s fill value, ``where``'s
    other branch): extent 1 — or exactly the block size when both
    blocks are equal — is fine; covering the whole axis would need its
    own slice, which these rules do not build → veto.

    Returns ``{"mode", "sizes", "md", "do"}``: ``md`` is the cat axis in
    the sliced operand's OWN coordinates (mask rank may differ from the
    score rank under broadcasting), ``do`` the cat axis of the
    broadcast result (output rank = max operand rank — a bigger-rank
    mask broadcasts the output up).
    """
    s1, s2 = _shape_of(bound.get("s1")), _shape_of(bound.get("s2"))
    D = bound.get("$attr:D")
    if not (isinstance(s1, tuple) and isinstance(s2, tuple)
            and s1 and len(s1) == len(s2)):
        return None
    if not isinstance(D, int):
        return None
    r, d = len(s1), D % len(s1)
    if not all(_dim_eq(s1[i], s2[i]) for i in range(r) if i != d):
        return None                          # ill-typed cat
    k1, k2 = s1[d], s2[d]
    if not (isinstance(k1, int) and isinstance(k2, int)):
        return None                          # unknown block extents

    shapes: dict[str, tuple] = {}
    out_rank = r
    for key in (sliced_key, *fixed_keys):
        sh = _shape_of(bound.get(key))
        if not isinstance(sh, tuple):
            return None                      # unknown shape: can't prove
        shapes[key] = sh
        out_rank = max(out_rank, len(sh))

    def _off_axis_ok(sh) -> bool:
        """Every operand dim OFF the cat axis must broadcast against the
        blocks (extra leading dims are fine — they rank up the output
        identically on both sides)."""
        for j, ext in enumerate(sh):
            dj = j + r - len(sh)             # operand dim → score dim
            if dj < 0 or dj == d:
                continue
            if not (ext is None or ext == 1 or _dim_eq(ext, s1[dj])):
                return False
        return True

    for key in fixed_keys:
        fs = shapes[key]
        if not _off_axis_ok(fs):
            return None
        fd = d + len(fs) - r                 # cat axis in f's coords
        if fd >= 0:
            ext = fs[fd]
            if not (ext == 1 or (k1 == k2 and ext == k1)):
                return None                  # needs its own slice

    ms = shapes[sliced_key]
    if not _off_axis_ok(ms):
        return None
    md = d + len(ms) - r                     # cat axis in mask coords
    do = d + out_rank - r                    # cat axis of the result
    if md < 0:
        return {"mode": "reuse", "sizes": None, "md": None, "do": do}
    ext = ms[md]
    if ext is None:
        return None
    if ext == k1 + k2:
        return {"mode": "slice", "sizes": (k1, k2), "md": md, "do": do}
    if ext == 1 or (k1 == k2 and ext == k1):
        return {"mode": "reuse", "sizes": None, "md": None, "do": do}
    return None


def _check_mask_cat(sliced_key: str, fixed_keys: tuple, mode: str):
    def check(bound: dict) -> bool:
        plan = _cat_axis_plan(bound, sliced_key, fixed_keys)
        return plan is not None and plan["mode"] == mode
    return check


def _derive_mask_cat(sliced_key: str, fixed_keys: tuple):
    def derive(bound: dict) -> dict | None:
        plan = _cat_axis_plan(bound, sliced_key, fixed_keys)
        if plan is None:
            return None
        out = {"$attr:DO": plan["do"]}
        if plan["mode"] == "slice":
            out["$attr:SZ"] = plan["sizes"]
            out["$attr:MD"] = plan["md"]
        return out
    return derive


def _mask_slice(key: str, i: int) -> Op:
    """Block i's slice of a mask/bias along the cat axis — a projection
    of the full mask, which is exactly where the positional offset of
    block i lives (causal masks included)."""
    return Op.make("split", key, sizes="SZ", dim="MD", index=i)


def _masked_fill_cat(name: str, attr_key: str, mode: str) -> Rewrite:
    """masked_fill(cat(s1,s2,D), m, v) → cat(masked_fill(s_i, m_i, v), D).

    ``m_i`` is the mask's slice on the cat axis (mode "slice") or the
    mask itself when it broadcasts along that axis (mode "reuse")."""
    m1 = _mask_slice("m", 0) if mode == "slice" else "m"
    m2 = _mask_slice("m", 1) if mode == "slice" else "m"
    return R(
        name,
        Op.make("masked_fill",
                Op.make("concat", "s1", "s2", **{attr_key: "D"}),
                "m", "v"),
        Op.make("concat",
                Op.make("masked_fill", "s1", m1, "v"),
                Op.make("masked_fill", "s2", m2, "v"),
                dim="DO"),
        check=_check_mask_cat("m", ("v",), mode),
        derive=_derive_mask_cat("m", ("v",)),
        law="masked_fill distributes over concat: masking a "
            "concatenated score matrix equals concatenating the masked "
            "blocks, each with its slice of the mask — a causal mask's "
            "block offset lives inside the slice.")


def _add_cat(name: str, attr_key: str, mode: str,
             mask_first: bool) -> Rewrite:
    """add(cat(s1,s2,D), m) / add(m, cat(s1,s2,D)) → cat of per-block
    adds — the additive-mask counterpart of masked_fill_cat.  add is
    commutative, but COMM_ADD is deliberately absent from OM_LAWS, so
    both operand orders get a rule."""
    m1 = _mask_slice("m", 0) if mode == "slice" else "m"
    m2 = _mask_slice("m", 1) if mode == "slice" else "m"
    cat = Op.make("concat", "s1", "s2", **{attr_key: "D"})
    lhs = (Op.make("add", "m", cat) if mask_first
           else Op.make("add", cat, "m"))
    b1 = Op.make("add", m1, "s1") if mask_first else Op.make("add", "s1", m1)
    b2 = Op.make("add", m2, "s2") if mask_first else Op.make("add", "s2", m2)
    return R(
        name, lhs,
        Op.make("concat", b1, b2, dim="DO"),
        check=_check_mask_cat("m", (), mode),
        derive=_derive_mask_cat("m", ()),
        law="An additive mask distributes over concat: add(cat s, m) = "
            "cat(add(s_i, m_i)) — concat is a homomorphism for "
            "elementwise ops, with the mask sliced on the cat axis.")


def _where_cat(name: str, attr_key: str, mode: str,
               cat_in_x: bool) -> Rewrite:
    """where(m, cat(s1,s2,D), v) / where(m, v, cat(s1,s2,D)) → cat of
    per-block wheres — the torch.where masking idiom."""
    m1 = _mask_slice("m", 0) if mode == "slice" else "m"
    m2 = _mask_slice("m", 1) if mode == "slice" else "m"
    cat = Op.make("concat", "s1", "s2", **{attr_key: "D"})
    if cat_in_x:
        lhs = Op.make("where", "m", cat, "v")
        b1 = Op.make("where", m1, "s1", "v")
        b2 = Op.make("where", m2, "s2", "v")
    else:
        lhs = Op.make("where", "m", "v", cat)
        b1 = Op.make("where", m1, "v", "s1")
        b2 = Op.make("where", m2, "v", "s2")
    return R(
        name, lhs,
        Op.make("concat", b1, b2, dim="DO"),
        check=_check_mask_cat("m", ("v",), mode),
        derive=_derive_mask_cat("m", ("v",)),
        law="where distributes over concat in the masked operand: "
            "where(m, cat s, v) = cat(where(m_i, s_i, v)).")


def _check_cat_pair(bound: dict) -> int | None:
    """For ``op(cat(a1,a2,DA), cat(b1,b2,DB)) → cat(op(a1,b1), op(a2,b2))``.

    Both cat axes must be the SAME axis of the broadcast result (ranks
    may differ — a lower-rank mask's cat axis is shifted), each operand
    pair must be cat-compatible on its own axis, and the per-block
    broadcast results must be cat-compatible on the shared axis.
    Returns the result cat dim, or None to veto."""
    a1, a2 = _shape_of(bound.get("a1")), _shape_of(bound.get("a2"))
    b1, b2 = _shape_of(bound.get("b1")), _shape_of(bound.get("b2"))
    DA, DB = bound.get("$attr:DA"), bound.get("$attr:DB")
    if not all(isinstance(s, tuple) and s for s in (a1, a2, b1, b2)):
        return None
    if not (isinstance(DA, int) and isinstance(DB, int)):
        return None
    ra, rb = len(a1), len(b1)
    if len(a2) != ra or len(b2) != rb:
        return None
    da, db = DA % ra, DB % rb
    ro = max(ra, rb)
    oa, ob = da + ro - ra, db + ro - rb
    if oa != ob:
        return None                          # different axes — wrong
    if not all(_dim_eq(a1[i], a2[i]) for i in range(ra) if i != da):
        return None
    if not all(_dim_eq(b1[i], b2[i]) for i in range(rb) if i != db):
        return None
    from catopt.cost import _broadcast, _INVALID
    ba, bb = _broadcast(a1, b1), _broadcast(a2, b2)
    if ba is _INVALID or bb is _INVALID:
        return None
    if not all(_dim_eq(ba[i], bb[i]) for i in range(ro) if i != oa):
        return None
    return oa


def _cat_pair_check(bound: dict) -> bool:
    return _check_cat_pair(bound) is not None


def _cat_pair_derive(bound: dict) -> dict | None:
    oa = _check_cat_pair(bound)
    return None if oa is None else {"$attr:DO": oa}


def _cat_hom_add(name: str, attr_key: str) -> Rewrite:
    """add(cat(a1,a2,D), cat(b1,b2,D)) = cat(add(a1,b1), add(a2,b2), D)
    — concat is a homomorphism for elementwise add.  This is the pure
    form of the additive-mask law when the mask is itself concat'd."""
    return R(
        name,
        Op.make("add",
                Op.make("concat", "a1", "a2", **{attr_key: "DA"}),
                Op.make("concat", "b1", "b2", **{attr_key: "DB"})),
        Op.make("concat",
                Op.make("add", "a1", "b1"),
                Op.make("add", "a2", "b2"),
                dim="DO"),
        check=_cat_pair_check,
        derive=_cat_pair_derive,
        law="concat homomorphism over add: cat(a1,a2)+cat(b1,b2) = "
            "cat(a1+b1, a2+b2) — the free case of mask distribution.")


def _cat_hom_masked_fill(name: str, attr_key: str) -> Rewrite:
    """masked_fill(cat(s1,s2,D), cat(m1,m2,DM), v) → cat of per-block
    masked_fills — the mask arrives already concat'd (e.g. chunked
    masks); block i pairs s_i with m_i directly, no split needed."""
    return R(
        name,
        Op.make("masked_fill",
                Op.make("concat", "a1", "a2", **{attr_key: "DA"}),
                Op.make("concat", "b1", "b2", **{attr_key: "DB"}),
                "v"),
        Op.make("concat",
                Op.make("masked_fill", "a1", "b1", "v"),
                Op.make("masked_fill", "a2", "b2", "v"),
                dim="DO"),
        check=_cat_pair_check,
        derive=_cat_pair_derive,
        law="masked_fill over two concat'd operands: the score concat "
            "and mask concat share an axis, so block i's mask is just "
            "m_i — the offset was already paid when the mask was "
            "concatenated.")


#: Elementwise-mask ops pushed through a concat'd operand.  "slice"
#: variants emit ``split`` projections of the mask; "reuse" variants
#: fire when the mask broadcasts along the cat axis.  Both concat attr
#: spellings, both ``add`` operand orders, both ``where`` positions.
MASKED_FILL_CAT: list[Rewrite] = [
    _masked_fill_cat(f"masked_fill_cat_{mode}_{ak}", ak, mode)
    for mode in ("slice", "reuse") for ak in ("dim", "arg1")
]

ADD_MASK_CAT: list[Rewrite] = [
    _add_cat(f"add_{'m' if mask_first else 'cat'}_"
             f"{'cat' if mask_first else 'm'}_{mode}_{ak}",
             ak, mode, mask_first)
    for mode in ("slice", "reuse") for mask_first in (False, True)
    for ak in ("dim", "arg1")
]

WHERE_CAT: list[Rewrite] = [
    _where_cat(f"where_cat_{'x' if cat_in_x else 'y'}_{mode}_{ak}",
               ak, mode, cat_in_x)
    for mode in ("slice", "reuse") for cat_in_x in (True, False)
    for ak in ("dim", "arg1")
]

#: Concat homomorphism when BOTH operands arrive concat'd (a mask that
#: was itself built blockwise).  Same check as the single-operand
#: rules: the two cat axes must coincide on the broadcast result.
CAT_HOM: list[Rewrite] = [
    _cat_hom_add("cat_hom_add_dim", "dim"),
    _cat_hom_add("cat_hom_add_arg1", "arg1"),
    _cat_hom_masked_fill("cat_hom_masked_fill_dim", "dim"),
    _cat_hom_masked_fill("cat_hom_masked_fill_arg1", "arg1"),
]

#: The mask-distribution law set.  Together with OM_SPLIT these turn
#: ``softmax(mask(q @ cat kᵢ.T)) @ cat vᵢ`` — masked or causal — into
#: ``om_apply(⊕ᵢ om_elem(masked sᵢ, vᵢ))``.
OM_MASK_LAWS: list[Rewrite] = [
    *CAT_HOM, *MASKED_FILL_CAT, *ADD_MASK_CAT, *WHERE_CAT,
]


# ---------------------------------------------------------------------------
#  Law set
# ---------------------------------------------------------------------------

#: Minimal law set for chunked-attention discovery.  Deliberately
#: excludes a commutativity rule for om_compose — comm is the explosive
#: law, same as SCAN_LAWS; block order is preserved by the concat
#: structure itself.
OM_LAWS: list[Rewrite] = [
    OM_LIFT, OM_LIFT_DIM, OM_LIFT_PLAIN, OM_UNLIFT,
    OM_SPLIT, OM_SPLIT_ARG1, OM_MERGE,
    OM_ASSOC, OM_ASSOC_REV,
    *CONCAT_BINARIZE,
    MATMUL_T_CONCAT, MATMUL_T_CONCAT_ARG1,
    *OM_MASK_LAWS,
]
