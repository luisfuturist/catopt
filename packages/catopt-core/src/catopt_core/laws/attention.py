"""Attention-path laws: rotary (RoPE) and score-scale algebra.

There is no fused ``rope``/``apply_rotary`` op in the IR — exported
graphs spell rotary position embedding as elementwise chains.  Three
spellings have been observed in ``export_to_ir`` output:

* **half-split cat** (``cat([x1·c − x2·s, x2·c + x1·s], -1)``, the
  ``bench/e2e_model._apply_rope`` / Llama form): contiguous
  ``[0,h)`` / ``[h,2h)`` slices of the feature axis feeding a
  ``sub``/``add`` pair under one ``concat``;
* **rotate-half add** (``x·cos + rotate_half(x)·sin``, the HF
  ``apply_rotary_pos_emb`` form): ``add(mul(x, C),
  mul(concat(neg(x2), x1), S))``;
* **interleaved / strided** pairs (``x[..., ::2]`` slices or
  ``reshape(...,2).unbind(-1)`` complex-mult, the llama2.c /
  GPT-NeoX form): the slices carry a ``step`` attr or the factors
  sit under ``unbind``/``stack`` — the contiguous patterns below
  deliberately do not match them (attr key-sets must equal exactly).

A fourth spelling exists too — ``x @ R`` with a single rank-2 matrix —
a position-independent rotation (or any constant right-multiply).
That one is the only rotary that *folds into a projection weight*.

What the laws exploit, honestly:

* **x-linearity of rotary** — ``rope(a·x) = a·rope(x)`` for a factor
  ``a`` that is scalar or a per-row ``(…,1)`` broadcast: such a factor
  is pair-constant by construction, so it commutes with the whole
  elementwise chain.  Both directions are offered (``*_in`` /
  ``*_out``) so the scale can ride whichever side of a reassociation
  is cheaper — the score-scale fold rides it into the weight.
* **angle addition** — ``rope₂∘rope₁ = rope`` with composed factor
  tables ``C = C₁C₂ − S₁S₂``, ``S = S₁C₂ + C₁S₂`` (rotation
  composition, exact).  The ``unrope∘rope`` cancellation is the same
  pattern with negated ``sin`` — the composite has angle θ₁−θ₂,
  ~identity on true rotation tables (``c²+s²=1`` lives below the
  term algebra; the composed single-rope member is what eqsat sees,
  and the verifier prices it honestly).  Composition happens on the
  *broadcast* factor terms, which is sound only when the two ropes'
  factor shapes agree — mixed broadcast spellings decline.
* **right-multiply absorb** — ``(x@Wᵀ)@R = x@(RᵀW)ᵀ``, i.e.
  ``linear(x,W)@R ≡ linear(x, RᵀW)``: a constant right-multiply
  after a projection (rotary or not) absorbs into the weight at
  compile time.  Rank-2 ``R`` only: ``F.linear`` cannot take a
  batched weight, and a per-position rotation table is not a fixed
  weight anyway — that class declines.  The multi-position spelling
  ``x @ R_t`` can still reassociate through the ``linear ↔ matmul``
  bridge laws (``LINEAR_TO_MM_T`` + ``ASSOC_MATMUL``).
* **score-scale migration** — the companion moves that let a
  ``scores·s`` factor reach a projection weight: scalar into the
  *left* matmul operand (the missing mirror of
  ``naturality_scalar``), a uniform factor through every view op in
  the spellings above, and ``s·linear(x,W) ≡ linear(x, sW)``.  With
  ``naturality_scalar_rev`` (right-operand, existing) plus these,
  ``mul(scores, Const s)`` folds into either ``Wq`` or ``Wk`` — a
  compile-time ``(o,i)`` weight mul replacing a runtime
  ``(B,H,T,T)`` elementwise op.  When the ``sdpa_fold_*mul*`` rules
  fire instead, the scale already lands in the kernel's ``scale``
  attribute at zero cost — the weight-fold is the alternative the
  cost model weighs, and the only path when the softmax is explicit.

Deliberately NOT here: the pair-block commutation
(``rope(q@W)`` vs ``q@(rope-adj W)``) — sound only when ``W``
respects the rotary pair structure, a *value-level* property no term
pattern can see.  It belongs to a non-local pass with
``source_tensors`` access, not an equational law.
"""

# ruff: noqa: RUF001 RUF002 RUF003 -- the law strings and docstrings use
# mathematical notation (·, ⊙, ≡, θ) deliberately; ASCII would misstate it.

from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.ir import Op
from catopt_core.laws import tags
from catopt_core.laws.base import R, _shape_of
from catopt_core.typing import _broadcast, _matmul_shape

#: Tag bundles (see :mod:`catopt_core.laws.tags`): every rule here is
#: opt-in ``ATTENTION``; the structural folds (rope composition, the
#: right-multiply absorbs) also carry ``FUSION``; the bidirectional
#: commutation/scale-migration pairs are closure-generating and carry
#: ``EXPANSIVE`` so the pipeline budgets them.
_FUS = (tags.ATTENTION, tags.FUSION)
_EXP = (tags.ATTENTION, tags.EXPANSIVE)

__all__ = [
    "ATTENTION_RULES",
    "LINEAR_MM_ABSORB",
    "LINEAR_MM_ABSORB_BIAS",
    "LINEAR_MM_ABSORB_BIAS_REV",
    "LINEAR_MM_ABSORB_REV",
    "LINEAR_OUT_SCALE",
    "LINEAR_OUT_SCALE_REV",
    "NATURALITY_SCALAR_LEFT",
    "NATURALITY_SCALAR_LEFT_REV",
    "ROPE_CAT_COMPOSE",
    "ROPE_CAT_SCALE_IN",
    "ROPE_CAT_SCALE_OUT",
    "ROPE_RH_SCALE_IN",
    "ROPE_RH_SCALE_OUT",
]


# ---------------------------------------------------------------------------
#  Pattern builders — the exported rope spellings
# ---------------------------------------------------------------------------


def _slice(t: Any, pre: str) -> Op:
    """``slice`` pattern over *t* with ``pre``-namespaced attr metavars.

    The unstrided canonical form (``dim``/``start``/``end`` — what
    ``torch.export`` emits for ``x[..., a:b]``).  Strided slices carry
    ``step`` and do not match — their attr key-set differs.
    """
    return Op.make(
        "slice", t, dim=f"{pre}D", start=f"{pre}A", end=f"{pre}E"
    )


def _rope_cat(x: Any, c: Any, s: Any, p1: str, p2: str, cd: str) -> Op:
    """Build the half-split rotary pattern over *x* with factors *c*/*s*.

    ``concat(sub(x1·c, x2·s), add(x2·c, x1·s), dim=cd)`` where ``p1`` /
    ``p2`` namespace the two slice patterns' attr metavariables.
    """
    x1, x2 = _slice(x, p1), _slice(x, p2)
    return Op.make(
        "concat",
        Op.make("sub", Op.make("mul", x1, c), Op.make("mul", x2, s)),
        Op.make("add", Op.make("mul", x2, c), Op.make("mul", x1, s)),
        dim=cd,
    )


def _rope_rh(x: Any, c: Any, s: Any, p1: str, p2: str, cd: str) -> Op:
    """Build the rotate-half rotary pattern: ``x·c + cat(−x2, x1)·s``.

    ``p2`` namespaces the negated slice, ``p1`` the plain one.
    """
    return Op.make(
        "add",
        Op.make("mul", x, c),
        Op.make(
            "mul",
            Op.make(
                "concat",
                Op.make("neg", _slice(x, p2)),
                _slice(x, p1),
                dim=cd,
            ),
            s,
        ),
    )


# ---------------------------------------------------------------------------
#  Check hooks — shape guards the matcher cannot see
# ---------------------------------------------------------------------------


def _is_uniform(t: Any) -> bool:
    """Scalar or all-dims-1 broadcast — an elementwise-constant factor."""
    s = _shape_of(t)
    return isinstance(s, tuple) and all(d == 1 for d in s)


def _check_uniform(bound: dict) -> bool:
    """Check ``a`` is a uniform factor — scalar or all-dims-1 broadcast."""
    return _is_uniform(bound.get("a"))


def _check_rope_scale(bound: dict) -> bool:
    """Check ``a`` may slide through a rotary: scalar or per-row ``(…,1)``.

    A last-dim-1 factor broadcasts one scalar per row — pair-constant
    by construction, so ``slice(x·a) ≡ slice(x)·a`` and the whole
    elementwise chain factors.  Anything wider (per-channel tables)
    breaks the rotary pairs and declines; unknown shapes decline for
    the non-scalar case.
    """
    sa = _shape_of(bound.get("a"))
    if not isinstance(sa, tuple):
        return False
    if not sa:
        return True  # scalar multiplies everything uniformly
    if sa[-1] != 1:
        return False
    xs = _shape_of(bound.get("x"))
    return isinstance(xs, tuple) and _broadcast(sa, xs) == xs


def _to_end(e: Any, n: int) -> bool:
    """Whether a bound slice ``end`` covers a dim of extent *n*.

    ``x[..., h:]`` exports ``end`` as the int64 sentinel (≥ any
    extent); ``x[..., h:2h]`` spells it exactly.
    """
    return isinstance(e, int) and e >= n


def _rope_axis(bound: dict, nd: int) -> int | None:
    """Return the shared axis all six rope dims normalize to, or None."""
    dims = [
        bound.get(f"$attr:{k}")
        for k in ("I1D", "I2D", "O1D", "O2D", "CD1", "CD2")
    ]
    ints = tuple(d for d in dims if isinstance(d, int))
    if len(ints) != len(dims):
        return None
    axis = ints[0] % nd
    return axis if all(d % nd == axis for d in ints) else None


def _half_bounds(bound: dict, pa: str, pb: str, width: int) -> int:
    """Return the first-half width when *pa*/*pb* split an axis of *width*.

    The pair must be contiguous ``[0,h)`` + ``[h,2h)`` — the rotary
    pairing.  Returns 0 when the bounds do not describe that split.
    """
    a1 = bound.get(f"$attr:{pa}A")
    e1 = bound.get(f"$attr:{pa}E")
    a2 = bound.get(f"$attr:{pb}A")
    e2 = bound.get(f"$attr:{pb}E")
    if not all(isinstance(v, int) for v in (a1, e1, a2)):
        return 0
    if a1 != 0 or a2 != e1 or width != 2 * (e1 - a1):
        return 0
    return e1 - a1 if _to_end(e2, width) else 0


def _check_rope_cat_compose(bound: dict) -> bool:
    """Nested cat-ropes compose only on a clean half-split.

    The inner rope must split the feature axis into contiguous halves
    (width ``h = extent/2``); the outer rope's slices must recover the
    two ``concat`` children exactly (``[0,h)`` + ``[h,2h)`` on the
    same axis); and the two ropes' factor tables must broadcast
    identically — the composed tables multiply ``C₁·C₂`` elementwise
    on their broadcast shape, sound only when the shapes agree.
    """
    xs = _shape_of(bound.get("x"))
    if not isinstance(xs, tuple) or not xs:
        return False
    axis = _rope_axis(bound, len(xs))
    if axis is None or not isinstance(xs[axis], int):
        return False
    h = _half_bounds(bound, "I1", "I2", xs[axis])
    if h <= 0 or _half_bounds(bound, "O1", "O2", 2 * h) != h:
        return False
    c1, c2 = _shape_of(bound.get("C1")), _shape_of(bound.get("C2"))
    s1, s2 = _shape_of(bound.get("S1")), _shape_of(bound.get("S2"))
    return (
        isinstance(c1, tuple)
        and c1 == c2
        and isinstance(s1, tuple)
        and s1 == s2
    )


def _check_linear_mm_absorb(bound: dict) -> bool:
    """``linear(x,W)@R`` folds only for a rank-2 weight and rank-2 R.

    ``W`` must be ``(o,i)`` and ``R`` ``(o,p)`` — the contraction
    ``R[-2] == o`` must agree (``None`` dims are wildcards).  Rank-3
    per-position tables decline: ``F.linear`` cannot take a batched
    weight, and a position-dependent rotation is not a fixed weight.
    """
    w, r = _shape_of(bound.get("W")), _shape_of(bound.get("R"))
    if not (
        isinstance(w, tuple)
        and len(w) == 2
        and isinstance(r, tuple)
        and len(r) == 2
    ):
        return False
    return r[0] is None or w[0] is None or r[0] == w[0]


def _check_linear_mm_absorb_bias(bound: dict) -> bool:
    """Check the biased absorb: the ``b@R`` matvec needs ``b`` of shape (o,)."""
    if not _check_linear_mm_absorb(bound):
        return False
    b = _shape_of(bound.get("b"))
    if not (isinstance(b, tuple) and len(b) == 1):
        return False
    w = _shape_of(bound.get("W"))
    return b[0] is None or w[0] is None or b[0] == w[0]


# ---------------------------------------------------------------------------
#  Rotary angle composition — rope₂∘rope₁ = rope(θ₁+θ₂)
#
#  cat(x1C₁ − x2S₁, x2C₁ + x1S₁) fed through the same block map gives
#      x1(C₁C₂ − S₁S₂) − x2(S₁C₂ + C₁S₂)   and   the mirrored second
#      child — the composed factor tables C and S below.
# ---------------------------------------------------------------------------

_CM = Op.make(
    "sub", Op.make("mul", "C1", "C2"), Op.make("mul", "S1", "S2")
)
_SM = Op.make(
    "add", Op.make("mul", "S1", "C2"), Op.make("mul", "C1", "S2")
)

ROPE_CAT_COMPOSE = R(
    "rope_cat_compose",
    _rope_cat(
        _rope_cat("x", "C1", "S1", "I1", "I2", "CD1"),
        "C2",
        "S2",
        "O1",
        "O2",
        "CD2",
    ),
    _rope_cat("x", _CM, _SM, "I1", "I2", "CD1"),
    law="Rotation composition is angle addition: rope₂∘rope₁ is one "
    "rotary whose factor tables are C = C₁C₂ − S₁S₂ and "
    "S = S₁C₂ + C₁S₂.  An unrope (negated sin) composes to "
    "θ₁−θ₂ — two elementwise chains collapse to one.",
    check=_check_rope_cat_compose,
    tags=_FUS,
)

# ---------------------------------------------------------------------------
#  Rotary is x-linear — a uniform factor commutes through the chain.
#
#  Both spellings × both directions: the `_out` direction pulls a scale
#  out of the rotary (factoring it toward the score matrix), `_in`
#  pushes it inside (where the view/linear laws route it to a weight).
# ---------------------------------------------------------------------------

_XA = Op.make("mul", "x", "a")

ROPE_CAT_SCALE_OUT = R(
    "rope_cat_scale_out",
    _rope_cat(_XA, "C", "S", "I1", "I2", "CD"),
    Op.make("mul", _rope_cat("x", "C", "S", "I1", "I2", "CD"), "a"),
    law="Rotary is linear in x: rope(a·x) = a·rope(x) for a "
    "pair-constant factor — a scalar or per-row broadcast rides "
    "outside the whole elementwise chain.",
    check=_check_rope_scale,
    tags=_EXP,
)

ROPE_CAT_SCALE_IN = R(
    "rope_cat_scale_in",
    Op.make("mul", _rope_cat("x", "C", "S", "I1", "I2", "CD"), "a"),
    _rope_cat(_XA, "C", "S", "I1", "I2", "CD"),
    law="Reverse: push a uniform factor inside the rotary so the "
    "view/weight laws can route it into a projection.",
    check=_check_rope_scale,
    tags=_EXP,
)

ROPE_RH_SCALE_OUT = R(
    "rope_rh_scale_out",
    _rope_rh(_XA, "C", "S", "I1", "I2", "CD"),
    Op.make("mul", _rope_rh("x", "C", "S", "I1", "I2", "CD"), "a"),
    law="x·cos + rotate_half(x)·sin is also x-linear: a uniform "
    "factor rides out of the HF spelling.",
    check=_check_rope_scale,
    tags=_EXP,
)

ROPE_RH_SCALE_IN = R(
    "rope_rh_scale_in",
    Op.make("mul", _rope_rh("x", "C", "S", "I1", "I2", "CD"), "a"),
    _rope_rh(_XA, "C", "S", "I1", "I2", "CD"),
    law="Reverse: push a uniform factor inside the rotate-half "
    "spelling.",
    check=_check_rope_scale,
    tags=_EXP,
)


# ---------------------------------------------------------------------------
#  Right-multiply absorb — (x@Wᵀ)@R = x@(RᵀW)ᵀ = linear(x, RᵀW)
#
#  The roped-by-matmul class: a SINGLE rank-2 R (position-independent
#  rotation, or any constant right-multiply) after a projection folds
#  into the weight at compile time — the runtime matmul disappears.
#  RᵀW is minted matmul(transpose(R,-2,-1), W): both operands are
#  param-only in the target class, so _fold_weight_chains materialises
#  it once.  The per-position batched-R class declines in check —
#  F.linear has no batched-weight form (and R_t is not a weight).
# ---------------------------------------------------------------------------

_MT_R = Op.make("transpose", "R", dim0=-2, dim1=-1)

LINEAR_MM_ABSORB = R(
    "linear_mm_absorb",
    Op.make("matmul", Op.make("linear", "x", "W"), "R"),
    Op.make("linear", "x", Op.make("matmul", _MT_R, "W")),
    law="(xWᵀ)R = x(RᵀW)ᵀ — a constant right-multiply absorbs into "
    "the projection weight (rotary-by-matmul included).",
    check=_check_linear_mm_absorb,
    tags=_FUS,
)

LINEAR_MM_ABSORB_REV = R(
    "linear_mm_absorb_rev",
    Op.make("linear", "x", Op.make("matmul", _MT_R, "W")),
    Op.make("matmul", Op.make("linear", "x", "W"), "R"),
    law="Reverse absorb — eqsat weighs the folded weight against a "
    "runtime right-multiply.",
    check=_check_linear_mm_absorb,
    tags=_EXP,
)

LINEAR_MM_ABSORB_BIAS = R(
    "linear_mm_absorb_bias",
    Op.make("matmul", Op.make("linear", "x", "W", "b"), "R"),
    Op.make(
        "linear",
        "x",
        Op.make("matmul", _MT_R, "W"),
        Op.make("matmul", "b", "R"),
    ),
    law="Affine right-multiply absorb: (xWᵀ + b)R = x(RᵀW)ᵀ + bR — "
    "the fused bias is the matvec b@R, both products compile-time.",
    check=_check_linear_mm_absorb_bias,
    tags=_FUS,
)

LINEAR_MM_ABSORB_BIAS_REV = R(
    "linear_mm_absorb_bias_rev",
    Op.make(
        "linear",
        "x",
        Op.make("matmul", _MT_R, "W"),
        Op.make("matmul", "b", "R"),
    ),
    Op.make("matmul", Op.make("linear", "x", "W", "b"), "R"),
    law="Reverse biased absorb (eqsat weighs fused vs split).",
    check=_check_linear_mm_absorb_bias,
    tags=_EXP,
)


# ---------------------------------------------------------------------------
#  Score-scale migration — scalar into the left matmul operand, a
#  uniform factor into a linear's weight, and uniform-factor
#  commutation through the view ops of the exported spellings.
#
#  naturality_scalar(_rev) already covers the RIGHT operand
#  (matmul(W, x·s) ↔ matmul(W,x)·s); the left side was missing —
#  needed to push scores·s into q̂.
# ---------------------------------------------------------------------------


def _check_left_scale(bound: dict) -> bool:
    """``s`` slides between ``mul(A@B, s)`` and ``matmul(A·s, B)``.

    Exact for a scalar or a last-dim-1 (per-row/per-batch) factor:
    ``(s·A)@B = s·(A@B)`` row-wise.  The factor must broadcast against
    the product's shape — checked through ``_matmul_shape`` so a
    provably ill-typed member declines.
    """
    sa = _shape_of(bound.get("s"))
    if not isinstance(sa, tuple) or (sa and sa[-1] != 1):
        return False
    out = _matmul_shape(
        _shape_of(bound.get("A")), _shape_of(bound.get("B"))
    )
    if out is None:
        return False
    return not sa or _broadcast(sa, out) == out


NATURALITY_SCALAR_LEFT = R(
    "naturality_scalar_left",
    Op.make("matmul", Op.make("mul", "A", "s"), "B"),
    Op.make("mul", Op.make("matmul", "A", "B"), "s"),
    law="Row-uniform scale slides out of a matmul's left operand: "
    "(s·A)@B = s·(A@B) — the left mirror of naturality_scalar.",
    check=_check_left_scale,
    tags=_EXP,
)

NATURALITY_SCALAR_LEFT_REV = R(
    "naturality_scalar_left_rev",
    Op.make("mul", Op.make("matmul", "A", "B"), "s"),
    Op.make("matmul", Op.make("mul", "A", "s"), "B"),
    law="Reverse: s·(A@B) = (s·A)@B — pushes a score scale into the "
    "left factor, where the view/weight laws route it to Wq.",
    check=_check_left_scale,
    tags=_EXP,
)


def _check_linear_out_scale(bound: dict) -> bool:
    """``s·linear(x,W) ≡ linear(x, sW)`` — uniform factors only.

    A scalar or all-dims-1 ``s`` broadcasts against both the output
    and ``W`` itself.  Per-channel ``(o,)`` or per-row ``(…,1)``
    factors need a different weight spelling (``s[:,None]·W`` /
    x-side) — those commute through ``LINEAR_ROW_SCALE`` /
    ``LINEAR_CHANNEL_SCALE`` instead.
    """
    return _is_uniform(bound.get("s"))


LINEAR_OUT_SCALE = R(
    "linear_out_scale",
    Op.make("mul", Op.make("linear", "x", "W"), "s"),
    Op.make("linear", "x", Op.make("mul", "W", "s")),
    law="Uniform scale folds into the weight: s·(xWᵀ) = x(sW)ᵀ — "
    "the runtime mul becomes a compile-time weight mul.",
    check=_check_linear_out_scale,
    tags=_FUS,
)

LINEAR_OUT_SCALE_REV = R(
    "linear_out_scale_rev",
    Op.make("linear", "x", Op.make("mul", "W", "s")),
    Op.make("mul", Op.make("linear", "x", "W"), "s"),
    law="Reverse uniform fold — eqsat compares weight-folded vs "
    "runtime-scaled forms.",
    check=_check_linear_out_scale,
    tags=_EXP,
)


# ---------------------------------------------------------------------------
#  Uniform factor through views — ``mul(view(t), a) ≡ view(mul(t), a)``
#  for every view op in the attention spellings.  Only uniform factors
#  (scalar / all-dims-1) commute: they broadcast the same value on
#  both sides of any relayout, unlike the equal-rank pointwise moves
#  of ``LAYOUT_RULES``, which *decline* rank-0 broadcast operands.
# ---------------------------------------------------------------------------


def _view_scale_rules() -> list[Rewrite]:
    """Build the uniform-factor/view commutations, both directions."""
    unary: list[tuple[str, Any]] = [
        ("transpose", Op.make("transpose", "t", dim0="D0", dim1="D1")),
        ("reshape", Op.make("reshape", "t", shape="SH")),
        ("slice", Op.make("slice", "t", dim="D", start="A", end="E")),
        (
            "slice_step",
            Op.make(
                "slice", "t", dim="D", start="A", end="E", step="P"
            ),
        ),
        ("unbind", Op.make("unbind", "t", dim="D", index="I")),
        (
            "chunk",
            Op.make("chunk", "t", chunks="N", dim="D", index="I"),
        ),
        ("flatten", Op.make("flatten", "t", start_dim="F0")),
        (
            "flatten2",
            Op.make("flatten", "t", start_dim="F0", end_dim="F1"),
        ),
        ("unsqueeze", Op.make("unsqueeze", "t", dim="D")),
        ("squeeze", Op.make("squeeze", "t", dim="D")),
    ]
    rules: list[Rewrite] = []
    for name, view in unary:
        rules.append(
            R(
                f"scale_in_{name}",
                Op.make("mul", view, "a"),
                _replant(view, Op.make("mul", "t", "a")),
                law=f"Uniform factor commutes into {name}: "
                "a·view(t) = view(a·t) — the elementwise-constant "
                "scale rides through any relayout.",
                check=_check_uniform,
                tags=_EXP,
            )
        )
        rules.append(
            R(
                f"scale_out_{name}",
                _replant(view, Op.make("mul", "t", "a")),
                Op.make("mul", view, "a"),
                law=f"Reverse uniform commutation out of {name} — "
                "factors a shared scale back to the view's output.",
                check=_check_uniform,
                tags=_EXP,
            )
        )
    for name in ("concat", "stack"):
        shared = Op.make(
            name,
            Op.make("mul", "t1", "a"),
            Op.make("mul", "t2", "a"),
            dim="D",
        )
        joined = Op.make(name, "t1", "t2", dim="D")
        rules.append(
            R(
                f"scale_in_{name}",
                Op.make("mul", joined, "a"),
                shared,
                law=f"Uniform factor distributes into {name}: "
                "a·cat(u,v) = cat(a·u, a·v).",
                check=_check_uniform,
                tags=_EXP,
            )
        )
        rules.append(
            R(
                f"scale_out_{name}",
                shared,
                Op.make("mul", joined, "a"),
                law=f"Reverse: shared uniform factors out of {name}.",
                check=_check_uniform,
                tags=_EXP,
            )
        )
    return rules


def _replant(view: Op, child: Any) -> Op:
    """Re-mint *view* with *child* in place of its ``t`` operand."""
    return Op.make(view.op, child, *view.args[1:], **view.attrs)


_VIEW_SCALE_RULES: list[Rewrite] = _view_scale_rules()


# ---------------------------------------------------------------------------
#  Rule collection
# ---------------------------------------------------------------------------

#: The whole attention-law surface — opt-in via
#: :data:`catopt_core.laws.ruleset.ATTENTION` /
#: ``preset("attention")``, never part of ``DEFAULT``/``all_rules()``:
#: the commutation pairs generate closure on any graph with view ops,
#: and the folds pay only where the exported spellings occur.
ATTENTION_RULES: list[Rewrite] = [
    # Rotary algebra
    ROPE_CAT_COMPOSE,
    ROPE_CAT_SCALE_OUT,
    ROPE_CAT_SCALE_IN,
    ROPE_RH_SCALE_OUT,
    ROPE_RH_SCALE_IN,
    # Right-multiply absorb (roped-by-matmul)
    LINEAR_MM_ABSORB,
    LINEAR_MM_ABSORB_REV,
    LINEAR_MM_ABSORB_BIAS,
    LINEAR_MM_ABSORB_BIAS_REV,
    # Score-scale migration
    NATURALITY_SCALAR_LEFT,
    NATURALITY_SCALAR_LEFT_REV,
    LINEAR_OUT_SCALE,
    LINEAR_OUT_SCALE_REV,
    *_VIEW_SCALE_RULES,
]
