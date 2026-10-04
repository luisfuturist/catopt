"""Tensor-algebra laws: the equational rewrite surface.

Each rule is :class:`~catopt_core.egraph.Rewrite` with an LHS pattern (match)
and an RHS pattern (replacement).  Metavariables are Python ``str``
objects that appear as leaves in the pattern trees.

Groups:
* **Monoid laws** — commutativity and associativity of add/mul.
* **Group laws** — identity, inverses, double-negation.
* **Bilinearity / naturality** — distributivity of matmul over add, and
  naturality of scalar multiplication w.r.t. linear maps.
* **Product structure** — the fused-projection laws (SwiGLU, QKV, GQA).
* **Softmax-attention fold** — the flash-attention transform.
* **Decomposition** — silu → x*sigmoid(x), subtraction → a + neg(b).

Collections:
* ``SIMPLIFICATION_RULES`` — basic algebraic simplification.
* ``CATEGORICAL_RULES`` — the categorical insight: distributivity,
  naturality, products, diagonal absorption, SDPA fold.
* ``ALL_RULES`` / :func:`all_rules` — the union.
"""

# ruff: noqa: RUF001 RUF002 RUF003 -- the law strings and docstrings use
# mathematical notation (σ, ⊗, ×) deliberately; ASCII would misstate it.

from typing import Any, cast

from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op
from catopt_core.laws import tags
from catopt_core.laws.base import R, _shape_of
from catopt_core.laws.cond import as_check, as_derive
from catopt_core.laws.layout import LAYOUT_RULES

#: Tag bundles for the rule definitions below (see
#: :mod:`catopt_core.laws.tags`): comm/assoc and the scale-hoist
#: naturality rules are ``SYMMETRY`` + ``EXPANSIVE`` (the Catalan-blowup
#: closure generators — opt-in, never in ``DEFAULT``); the rest of the
#: bilinearity/distributivity/composition algebra is ``CATEGORICAL`` +
#: ``EXPANSIVE`` (today's ``_EXPANSIVE_RULES`` budget set); the term-
#: local product folds the pairing pass owns are ``FUSION`` +
#: ``SUBSUMED``; the remaining folds (sdpa, gqa) are ``FUSION``; the
#: basic algebraic simplifications are ``SIMPLIFICATION``.
_SYM = (tags.SYMMETRY, tags.EXPANSIVE)
_CAT = (tags.CATEGORICAL, tags.EXPANSIVE)
_SUB = (tags.FUSION, tags.SUBSUMED)
_FUS = (tags.FUSION,)
_SIM = (tags.SIMPLIFICATION,)

# ---------------------------------------------------------------------------
#  Monoid laws: commutativity & associativity
# ---------------------------------------------------------------------------

COMM_ADD = R(
    "comm_add",
    Op.make("add", "a", "b"),
    Op.make("add", "b", "a"),
    law="Commutativity in a symmetric monoidal category: σ ∘ (f ⊗ g) = g ⊗ f.",
    tags=_SYM,
)

COMM_MUL = R(
    "comm_mul",
    Op.make("mul", "a", "b"),
    Op.make("mul", "b", "a"),
    law="Hadamard product is commutative (SMC symmetry).",
    tags=_SYM,
)

ASSOC_ADD = R(
    "assoc_add",
    Op.make("add", "a", Op.make("add", "b", "c")),
    Op.make("add", Op.make("add", "a", "b"), "c"),
    law="Associativity of sequential composition in a category.",
    tags=_SYM,
)

ASSOC_MUL = R(
    "assoc_mul",
    Op.make("mul", "a", Op.make("mul", "b", "c")),
    Op.make("mul", Op.make("mul", "a", "b"), "c"),
    law="Associativity of parallel composition in a monoidal category.",
    tags=_SYM,
)


# ---------------------------------------------------------------------------
#  Identity / inverse laws (group structure)
# ---------------------------------------------------------------------------

ID_ADD = R(
    "id_add",
    Op.make("add", "a", Const(0)),
    "a",
    law="Additive identity: a + 0 = a.",
    tags=_SIM,
)

ID_MUL = R(
    "id_mul",
    Op.make("mul", "a", Const(1)),
    "a",
    law="Multiplicative identity: a * 1 = a.",
    tags=_SIM,
)

DOUBLE_NEG = R(
    "double_neg",
    Op.make("neg", Op.make("neg", "a")),
    "a",
    law="Double negation: ¬¬a = a (involution).",
    tags=_SIM,
)


# ---------------------------------------------------------------------------
#  Subtraction and negation
# ---------------------------------------------------------------------------

SUB_TO_ADD = R(
    "sub_to_add",
    Op.make("sub", "a", "b"),
    Op.make("add", "a", Op.make("neg", "b")),
    law="Subtraction as addition of inverse: a - b = a + (-b).",
    tags=_SIM,
)


# ---------------------------------------------------------------------------
#  Decompositions
# ---------------------------------------------------------------------------

SILU_EXPAND = R(
    "silu_expand",
    Op.make("silu", "x"),
    Op.make("mul", "x", Op.make("sigmoid", "x")),
    law="SiLU definition: silu(x) = x · σ(x).",
    tags=_SIM,
)

SQUARE_EXPAND = R(
    "square_expand",
    Op.make("square", "x"),
    Op.make("mul", "x", "x"),
    law="Self-composition: x² = x · x.",
    tags=_SIM,
)

# RMSNorm/pow family: connect x.pow(2) to the mul-based representation.
# Lets square-related and naturality rewrites see RMSNorm's x**2 term.
POW_TO_SQUARE = R(
    "pow_to_square",
    Op.make("pow", "x", Const(2)),
    Op.make("square", "x"),
    law="pow(x, 2) ≡ square(x) ≡ x·x (SwiGLU/RMSNorm bridge).",
    tags=_SIM,
)

SQUARE_TO_POW = R(
    "square_to_pow",
    Op.make("square", "x"),
    Op.make("pow", "x", Const(2)),
    law="Reverse: square(x) ≡ pow(x, 2) for shape/cost reasons.",
    tags=_SIM,
    derivation=("pow_to_square",),
)

# The x·x → square bridge — ``square_expand``'s definitional inverse
# and the missing seed direction the RMSNorm fold's ``pow(u, 2)``
# pattern needs: a graph that spells x² as ``mul(u, u)`` gains a
# ``square`` member here, and ``square_to_pow`` then carries it into
# the ``pow`` spelling ``rms_norm_fold``'s LHS pins (the
# ``mul(x,x)``-spelled source was ``rms-norm-law.md``'s documented
# miss).  The shared ``u`` metavariable is the whole precondition —
# the matcher binds both mul operands to the same e-class, so
# ``mul(u, v)`` never fires, and no check is needed.  Term-local,
# at most one member per e-class — it does not grow the closure.
MUL_SQUARE = R(
    "mul_square",
    Op.make("mul", "u", "u"),
    Op.make("square", "u"),
    law="x·x = square(x): the mul spelling of x² folds to the unary "
    "kernel — the definitional inverse of square_expand.",
    tags=_SIM,
    derivation=("square_expand",),
)

# SwiGLU bridge: the exported graph has silu(linear(...)) followed by
# mul with another linear(...).  Expanding silu exposes the common
# x*sigmoid(x) factor, which lets naturality/distributivity see the
# shared linear prefix.  Already have SILU_EXPAND; add the mul-form.
#
# The library's one emergent law: a measured one-step consequence of
# the silu expand/fold class (the catalogue's only composite direct
# edges — ``silu_expand ⇒ silu_mul_form`` fires inside the ``mul``
# context).  Kept as a lemma: the spelled-out form buys reach on
# hand-written SwiGLU graphs that the kernel path doesn't fire on.
SILU_MUL_FORM = R(
    "silu_mul_form",
    Op.make("mul", Op.make("silu", "g"), "u"),
    Op.make("mul", Op.make("mul", "g", Op.make("sigmoid", "g")), "u"),
    law="SwiGLU: silu(g)*u = (g*sigmoid(g))*u (factor for prefix sharing).",
    tags=_SIM,
    derivation=("silu_expand",),
)

# The definitional inverse of SILU_EXPAND — the manual-silu fold (the
# ``softmax_fold`` analogue for the activation): ``x·σ(x)`` IS silu
# spelled by hand.  The shared ``x`` metavariable is the whole
# precondition — the matcher binds both mul operands to the same
# e-class, so ``mul(x, σ(y))`` never fires — and no check/derive is
# needed.
#
# Beyond folding a hand-spelled kernel to one dispatch, this is the
# missing 3-cell of the coherence catalogue
# (project/retros/law-coherence-catalogue.md §4): expanding ``silu``
# inside ``mul`` destroys the ``swiglu_fuse`` redex, and nothing
# rebuilt it — ``silu_expand × swiglu_fuse`` and
# ``silu_mul_form × swiglu_fuse`` were the library's two divergent
# critical pairs.  The fold transports the expansion back into the
# gate — ``mul((g·σg), u) → mul(silu(g), u)`` — restoring the fuse
# path, so both one-step reducts rejoin under the library.  Inverse
# pairs are kept deliberately (eqsat needs both directions
# reachable); it is term-local and adds at most one member per
# e-class, so it does not grow the closure.
SILU_FOLD = R(
    "silu_fold",
    Op.make("mul", "x", Op.make("sigmoid", "x")),
    Op.make("silu", "x"),
    law="The manual-silu fold: x · σ(x) = silu(x) — the definitional "
    "inverse of silu_expand, and the mediating 3-cell of the "
    "silu_expand/silu_mul_form × swiglu_fuse critical pairs.",
    tags=_SIM,
    derivation=("silu_expand",),
)


# ---------------------------------------------------------------------------
#  View-op naturality — elementwise mul through `select`
# ---------------------------------------------------------------------------

# `select` is a stride view (torch.select) — the same kind of free
# re-layout as `transpose`.  Elementwise `mul` commutes with it: the
# product of two slices is the slice of the product.  This is the
# `select` analogue of the layout family's `transpose_pull_mul`, but
# unlike that bidirectional transpose pair it is single-direction and
# term-local, so it does not generate a saturation closure and belongs
# in the default set rather than the opt-in LAYOUT_RULES (see the
# ALL_RULES note below).
#
# The shared `dim`/`index` attribute metavariables force both operand
# selects to read the same index along the same axis, so the matcher
# enforces the dim/index precondition structurally.  But that is NOT
# the whole precondition — `mul(u, v)` broadcasts when u and v
# disagree, and broadcasting along the *selected* axis changes the
# result: u=(4,), v=(2,4) makes sel(u,0,0)⊙sel(v,0,0) = u[0]·v[0] a
# scalar-times-vector, while sel(mul(u,v),0,0) = (u⊙v)[0] is u·v[0]
# elementwise — different tensors (found by the view-oracle; the 13
# real sites all happened to have agreeing D-dims, so it never bit).
# The cond requires shape(u)[D] == shape(v)[D]; broadcasting on other
# axes is safe (both sides broadcast identically after selection).
SELECT_MUL = R(
    "select_mul",
    Op.make(
        "mul",
        Op.make("select", "u", dim="D", index="I"),
        Op.make("select", "v", dim="D", index="I"),
    ),
    Op.make("select", Op.make("mul", "u", "v"), dim="D", index="I"),
    law="mul commutes with select: sel(u) ⊙ sel(v) = sel(u ⊙ v) — the "
    "elementwise product of two slices is the slice of the product "
    "(naturality of the elementwise action over the select view).  "
    "Removes one dispatched op per site.",
    cond=("dim-eq-attr", "u", "D", "v", "D"),
    tags=_SIM,
)


# ---------------------------------------------------------------------------
#  Kernel recognition — the manual-softmax fold
# ---------------------------------------------------------------------------


# The SECOND machine-discovered law admitted to the library (after
# select_mul) — proposed by the law pipeline's pattern-recognition
# pass, which scanned the model census for composed-then-reduced
# chains and recognised ``div(exp(·), sum(exp(·)))`` as softmax
# spelled by hand (see project/retros/law-proposer-extensions.md).
# It IS the definition of softmax — which is the point: a corpus that
# spells the kernel by hand gets the kernel back, folded to one
# dispatched op, with a certificate.
#
# Unlike select_mul the precondition is NOT structural: the sum's
# `keepdim` must be True (a dropped dim broadcasts wrongly — or not
# at all — against the numerator) and the reduce must cover exactly
# one axis (softmax has no multi-axis image).  `cond` carries that
# side condition; `derive` translates the sum's `dim` tuple `(-1,)`
# to softmax's scalar `dim=-1` (the RHS attr the LHS does not carry
# verbatim).  Single-direction and term-local, so it does not grow
# the closure and belongs in the default set.
_COND_SOFTMAX_FOLD = (
    "and",
    ("attr-is", "RK", True),
    (
        "or",
        ("attr-type", "RD", "int"),
        (
            "and",
            ("attr-type", "RD", "tuple"),
            ("attr-len", "RD", "==", 1),
        ),
    ),
)

#: Compat alias — the test-facing hook; it IS the same data the rule
#: carries in ``cond`` (``as_check`` keeps them from drifting).
_check_sum_keepdim = as_check(_COND_SOFTMAX_FOLD)


#: Unwrap ``sum``'s ``dim`` tuple into ``softmax``'s scalar dim — the
#: derived RHS attr as data (an empty ``dim`` tuple declines; the
#: old Python hook crashed, so the spec's veto is strictly safer).
_DSPEC_SOFTMAX_DIM = {"SD": ("attr0", "RD")}

#: Compat alias — the test-facing hook; it IS the same data the rule
#: carries in ``dspec`` (``as_derive`` keeps them from drifting).
_derive_softmax_dim = as_derive(_DSPEC_SOFTMAX_DIM)


SOFTMAX_FOLD = R(
    "softmax_fold",
    Op.make(
        "div",
        Op.make("exp", "u"),
        Op.make("sum", Op.make("exp", "u"), dim="RD", keepdim="RK"),
    ),
    Op.make("softmax", "u", dim="SD"),
    law="exp(u) / Σ exp(u) = softmax(u): the manual normalization fold "
    "— a composed-then-reduced chain IS the kernel's definition.  "
    "Folds div+exp+sum to one dispatched op.",
    cond=_COND_SOFTMAX_FOLD,
    dspec=_DSPEC_SOFTMAX_DIM,
    tags=_SIM,
)


# The GLU kernel fold — the same recipe one op-family over (the
# ``corpus-expansion-r2.md`` §6 lead): the corpus had the kernel image
# (``GluMLP``'s ``glu`` node) but no manual spelling until
# ``ManualGluMLP`` closed the pair.  ``F.glu(u, d)`` IS
# ``a ⊗ σ(b)`` with ``a, b`` the two halves of ``u`` along ``d`` — and
# the export's getitem fold lands the slice index on the ``chunk``
# node itself, so the manual spelling is exactly
# ``mul(chunk(u,2,d,0), σ(chunk(u,2,d,1)))``.
#
# Most of the precondition is structural: the shared ``u`` metavariable
# binds both chunk operands to the same e-class (chunks of different
# sources never match), the shared ``D`` attr metavariable pins them
# to the same split axis and carries it to the RHS, and the literal
# ``chunks``/``index`` attrs pin the two-equal-halves split and the
# gate order — ``index=0`` ungated, ``index=1`` σ'd, matching
# ``F.glu``'s first-half/second-half convention; the swapped-gate and
# multi-chunk spellings cannot match.
#
# The residual guard is parity of the split axis.  ``glu`` halves its
# ``dim`` exactly, but ``chunk(·, 2, d)`` splits an odd axis
# first-big (n=3 → 2+1) — and (…,2)·(…,1) still broadcasts, so an
# odd-axis redex evaluates while its ``glu`` image raises at eval.
# Evenness of ``u.shape[D]`` is the exact precondition; it needs the
# attr-named axis, which the cond DSL cannot index, so it stays a
# procedural ``check`` — ``cond`` carries the expressible front
# (``u`` shaped, rank ≥ 1; a scalar has no axis to halve).  Unknown or
# ``None`` dims ON the split axis decline (the library's strict
# posture); ``None`` dims elsewhere do not matter.  Term-local and
# single-direction — at most one member per e-class, no closure
# growth.
_COND_GLU_FOLD = ("rank", "u", ">=", 1)

#: Compat alias — the test-facing hook (see ``_check_sum_keepdim``).
_check_glu_shaped = as_check(_COND_GLU_FOLD)


def _check_glu_fold(bound: dict) -> bool:
    """Veto the odd-axis case: ``u.shape[dim]`` must be a known even int."""
    from catopt_core.typing import _shape_of as _so

    s = _so(bound.get("u"))
    d = bound.get("$attr:D")
    if not isinstance(s, tuple) or not isinstance(d, int):
        return False
    if not (-len(s) <= d < len(s)):
        return False
    n = s[d % len(s)]
    return isinstance(n, int) and n % 2 == 0


GLU_FOLD = R(
    "glu_fold",
    Op.make(
        "mul",
        Op.make("chunk", "u", chunks=2, dim="D", index=0),
        Op.make(
            "sigmoid",
            Op.make("chunk", "u", chunks=2, dim="D", index=1),
        ),
    ),
    Op.make("glu", "u", dim="D"),
    law="chunk-half · σ(chunk-half) IS glu(u): the manual-GLU fold — "
    "a ⊗ σ(b) over the two equal halves of u IS the kernel's "
    "definition.  Folds mul+sigmoid+2 chunks to one dispatched op.",
    cond=_COND_GLU_FOLD,
    check=_check_glu_fold,
    tags=_SIM,
)


# The RMSNorm kernel fold — the fourth machine-discovered law admitted
# to the library, the ``corpus-expansion-r2.md`` §6 lead and the exact
# ``softmax_fold`` situation one norm family over: the corpus carries
# BOTH spellings of the same math — ``RMSNorm``/``NormLinear``/
# ``TransformerBlock``/``ParallelBlock``'s manual
# ``x·rsqrt(mean(x²)+eps)·w`` and ``NativeRmsNorm``'s fused ``rms_norm``
# kernel op.  ``x·rms⁻¹·w`` IS ``F.rms_norm`` spelled by hand; folding
# it recovers the kernel as one dispatched op (measured −83 % on
# RMSNorm, −67 % on NormLinear, −31 % on TransformerBlock, −19 % on
# ParallelBlock of extracted cost — see project/retros/rms-norm-law.md).
#
# The pattern pins the export's exact op-tree — ``pow`` (not the
# ``square``/``mul(u,u)`` spellings: under DEFAULT saturation
# ``pow_to_square``/``square_expand`` put all three in one e-class, but
# only a graph that *carries* a ``pow`` enode matches; a ``mul(x,x)``-
# spelled source is a documented miss, not an unsoundness), ``rsqrt``
# (not ``div(1, sqrt)``), and the canonical ``(x·rms)·w`` association
# the exporter writes — alternate mul orderings are reachable only
# through the opt-in SYMMETRY set.
#
# The side condition splits the usual way.  ``cond`` carries the
# expressible front: ``keepdim`` must be True (a dropped axis cannot
# broadcast the rms back over ``u``), ``eps`` must be a numeric
# ``Const`` leaf (the kernel's ``eps`` is a float attr — a tensor
# ``eps`` has no image), and the ``pow`` exponent must be the literal
# 2.  ``check`` carries what the DSL cannot: the ``mean``'s reduce
# dims must name exactly u's last ``k`` axes — ``F.rms_norm`` only
# normalizes a trailing block — and the gain ``w``'s shape must BE
# that trailing block (``aten.rms_norm`` rejects any other weight
# shape at eval, so a mismatch would mint an unlowerable member).
# ``derive`` then computes the two RHS attrs the LHS cannot bind
# verbatim: ``dim`` is the *normalized shape* tuple (u.shape[-k:]),
# not the reduce dims, and ``eps`` unwraps the bound ``Const`` leaf
# into the float attr.  Unknown or ``None`` dims decline — the same
# strict posture as ``_check_glu_fold``; a law never mints an attr it
# cannot verify.
_COND_RMS_FOLD = (
    "and",
    ("attr-is", "MK", True),
    ("const-num", "EPS"),
    ("const-cmp", "P", "==", 2),
)

#: Compat alias — the test-facing hook (see ``_check_sum_keepdim``).
_check_rms_consts = as_check(_COND_RMS_FOLD)


def _rms_dims(bound: dict) -> tuple | None:
    """Return the ``mean``'s reduce dims as a tuple, or ``None``.

    Accepts the exported ``(-1,)`` tuple, a list, or a hand-minted
    bare int; every entry must be a non-``bool`` int.
    """
    dims = bound.get("$attr:MD")
    if isinstance(dims, int) and not isinstance(dims, bool):
        dims = (dims,)
    if not (
        isinstance(dims, (tuple, list))
        and dims
        and all(
            isinstance(d, int) and not isinstance(d, bool) for d in dims
        )
    ):
        return None
    return tuple(dims)


def _rms_normalized_shape(bound: dict):
    """Return the ``normalized_shape`` the LHS proves, or ``None``.

    ``mean(pow(u, 2), dims)`` is the kernel's reduce iff *dims* names
    exactly u's last ``k`` axes — each in range, no duplicates — and
    ``u`` is concretely shaped (the RHS ``dim`` attr IS ``u.shape[-k:]``,
    so an unshaped ``u`` cannot mint it — the strict posture).
    """
    dims = _rms_dims(bound)
    if dims is None:
        return None
    su = _shape_of(bound.get("u"))
    if not (
        isinstance(su, tuple) and all(isinstance(d, int) for d in su)
    ):
        return None
    rank, k = len(su), len(dims)
    if not 1 <= k <= rank or not all(-rank <= d < rank for d in dims):
        return None
    norm = {d % rank for d in dims}
    # ``k`` distinct in-range axes whose minimum is ``rank - k`` IS
    # the trailing block {rank-k … rank-1}.
    if len(norm) != k or min(norm) != rank - k:
        return None
    return tuple(su[rank - k :])


def _check_rms_fold(bound: dict) -> bool:
    """Veto the gained fold when ``w``'s shape isn't the normalized shape."""
    ns = _rms_normalized_shape(bound)
    if ns is None:
        return False
    sw = _shape_of(bound.get("w"))
    return isinstance(sw, tuple) and tuple(sw) == ns


def _check_rms_fold_nogain(bound: dict) -> bool:
    """Veto the gain-free fold when the reduce isn't a trailing block."""
    return _rms_normalized_shape(bound) is not None


def _derive_rms_norm(bound: dict) -> dict | None:
    """Mint the kernel's ``dim`` (a shape, not the reduce dims) + ``eps``."""
    ns = _rms_normalized_shape(bound)
    eps = getattr(bound.get("EPS"), "value", None)
    if ns is None or not isinstance(eps, (int, float)):
        return None
    return {"$attr:ND": ns, "$attr:EP": float(eps)}


def _rms_reduce(u: str = "u") -> Op:
    """``rsqrt(mean(u², MD, keepdim=MK) + EPS)`` — the shared LHS core."""
    return Op.make(
        "rsqrt",
        Op.make(
            "add",
            Op.make(
                "mean",
                Op.make("pow", u, "P"),
                dim="MD",
                keepdim="MK",
            ),
            "EPS",
        ),
    )


RMS_NORM_FOLD = R(
    "rms_norm_fold",
    Op.make("mul", Op.make("mul", "u", _rms_reduce()), "w"),
    Op.make("rms_norm", "u", "w", dim="ND", eps="EP"),
    law="x·rsqrt(mean(x²)+eps)·w IS rms_norm(x, w): the manual-RMSNorm "
    "fold — composed-then-reduced then scaled IS the kernel's "
    "definition.  Folds mul+mul+rsqrt+add+mean+pow to one dispatched "
    "op.",
    cond=_COND_RMS_FOLD,
    check=_check_rms_fold,
    derive=_derive_rms_norm,
    tags=_SIM,
)

# The gain-free twin — ``x·rms⁻¹`` with no channel gain is
# ``F.rms_norm(x, ns, weight=None)``.  Its LHS is the inner ``mul`` of
# the gained fold's, so it also fires inside every gained site (the
# ``mul(rms_norm(u), w)`` member it adds sits in the same e-class as
# the gained fold's ``rms_norm(u, w)`` — the cost model picks the
# fused one).  Term-local and single-direction, like its twin.
RMS_NORM_FOLD_NOGAIN = R(
    "rms_norm_fold_nogain",
    Op.make("mul", "u", _rms_reduce()),
    Op.make("rms_norm", "u", dim="ND", eps="EP"),
    law="x·rsqrt(mean(x²)+eps) IS rms_norm(x): the weight-free "
    "manual-RMSNorm fold — the gained fold's inner ``mul`` and a real "
    "spelling of its own (gain-free RMSNorm blocks).",
    cond=_COND_RMS_FOLD,
    check=_check_rms_fold_nogain,
    derive=_derive_rms_norm,
    tags=_SIM,
)


# ---------------------------------------------------------------------------
#  rsqrt canonicalization — the normalizer's non-canonical spellings.
#
#  ``rsqrt`` is the library's canonical reciprocal-root: the RMSNorm
#  fold pins ``rsqrt(mean(u²)+eps)`` literally, so a corpus graph
#  spelling the same value as ``1/√(·)`` or ``(·)^-0.5`` could never
#  match (``rms-norm-law.md``'s caveats recorded both as misses).
#  These single-direction folds canonicalize both spellings into the
#  member the kernel fold needs.  Both are numerics-exact — rsqrt is
#  *defined* as the reciprocal root — and term-local.
#
#  The literal checks ride the cond DSL: ``const-cmp`` is a NUMERIC
#  comparison, so ``Const(1)``/``Const(1.0)`` numerators and
#  ``Const(-0.5)`` exponents all fold while a ``Var``/``Param``/
#  non-Const binding declines (the DSL's strict posture — the matcher
#  cannot see leaf kinds).
# ---------------------------------------------------------------------------

DIV_SQRT_TO_RSQRT = R(
    "div_sqrt_to_rsqrt",
    Op.make("div", "ONE", Op.make("sqrt", "u")),
    Op.make("rsqrt", "u"),
    law="1 / sqrt(x) = rsqrt(x): the two-op reciprocal-root spelling "
    "folds to the dispatched kernel — the canonical form the RMSNorm "
    "fold's pattern pins.",
    cond=("const-cmp", "ONE", "==", 1),
    tags=_SIM,
)

POW_TO_RSQRT = R(
    "pow_to_rsqrt",
    Op.make("pow", "u", "P"),
    Op.make("rsqrt", "u"),
    law="pow(x, -0.5) = rsqrt(x): the negative-half-power spelling "
    "folds to the dispatched kernel — the same canonicalization as "
    "div_sqrt_to_rsqrt one op-family over.",
    cond=("const-cmp", "P", "==", -0.5),
    tags=_SIM,
)

# The ``1/t`` spelling exports as ``reciprocal(t)`` — aten lowers
# scalar-over-tensor division to ``reciprocal(t) * 1`` (id_mul strips
# the unit), so ``1/sqrt(u)`` reaches the graph as
# ``reciprocal(sqrt(u))``, not the div spelling above.  Fully
# structural — no check.
RECIP_SQRT_TO_RSQRT = R(
    "recip_sqrt_to_rsqrt",
    Op.make("reciprocal", Op.make("sqrt", "u")),
    Op.make("rsqrt", "u"),
    law="reciprocal(sqrt(x)) = rsqrt(x): the exported ``1/sqrt`` "
    "spelling — aten lowers scalar-over-tensor div to reciprocal — "
    "folds to the same canonical kernel.",
    tags=_SIM,
)


# ---------------------------------------------------------------------------
#  Distributivity / naturality (the categorical insight)
#
#  RANK GUARD.  ``matmul`` contracts a rank>=2 right operand's axis -2
#  but a rank-1 operand's ONLY axis, while ``add`` broadcast-aligns
#  TRAILING axes.  Summing a vector with a rank>=2 addend therefore
#  lands the vector's contraction index on the partner's OUTPUT axis
#  and bilinearity fails:
#
#      x(16,)@a(16,) + x(16,)@b(16,16)  !=  x @ (a+b)
#
#  — the scalar dot product broadcasts across the matvec's output axis
#  (max diff ~13 on randn; verified in
#  project/retros/matmul-unsound-fix.md).  Equal-rank addends align
#  axis-for-axis; two rank>=2 addends may still differ in rank —
#  broadcast then only replicates leading/batch axes, never realigning
#  axis -2, so the law holds.  The check vetoes only the provable
#  mismatch (both shapes must be KNOWN — an unshaped member cannot
#  prove its contraction axes align, so it declines, the same strict
#  posture as layout's ``_check_commute_binary``).  The LEFT-operand
#  pair (``right_distribute``/``right_factor``) needs no guard: there
#  the contraction axis IS the last axis for every rank — exactly the
#  axis broadcast aligns.
# ---------------------------------------------------------------------------


#: Contraction-axis alignment of two ``add`` addends, as data.
#:
#: Equal ranks align every axis pairwise.  Two rank>=2 addends may
#: differ: broadcast then pads/replicates leading axes only, leaving
#: each operand's axis -2 (the contraction axis) paired with the
#: other's.  A rank-1 addend's only axis is BOTH its contraction axis
#: and its broadcast tail — against a rank>=2 partner it lands on the
#: partner's output axis, so the identity is false on every evaluable
#: binding of that shape.  Scalar addends can never feed matmul, and
#: an unshaped member cannot prove alignment — every rank op declines
#: it (the same strict posture as layout's axes predicates).
_COND_MM_ADDENDS = (
    "or",
    ("and", ("rank-eq", "a", "b"), ("rank", "a", ">=", 1)),
    ("and", ("rank", "a", ">=", 2), ("rank", "b", ">=", 2)),
)

#: The same alignment on the summed weights ``W``/``W2`` — applied to
#: the ``linear`` spellings too (mathematically safe there, but a
#: rank-mixed weight sum mints a member ``F.linear`` cannot lower).
_COND_MM_WEIGHTS = (
    "or",
    ("and", ("rank-eq", "W", "W2"), ("rank", "W", ">=", 1)),
    ("and", ("rank", "W", ">=", 2), ("rank", "W2", ">=", 2)),
)

#: Compat aliases — the test-facing hooks (see ``_check_sum_keepdim``).
_check_mm_rhs_addends = as_check(_COND_MM_ADDENDS)
_check_mm_rhs_weights = as_check(_COND_MM_WEIGHTS)


# matmul(W, a + b) = matmul(W, a) + matmul(W, b)
DISTRIBUTE_MUL = R(
    "distribute_matmul_over_add",
    Op.make("matmul", "W", Op.make("add", "a", "b")),
    Op.make(
        "add", Op.make("matmul", "W", "a"), Op.make("matmul", "W", "b")
    ),
    law="Distributivity of linear maps over addition (bilinearity).",
    cond=_COND_MM_ADDENDS,
    tags=_CAT,
)

# add(matmul(W, a), matmul(W, b)) → matmul(W, add(a, b))  [reverse]
FACTOR_MUL = R(
    "factor_matmul",
    Op.make(
        "add", Op.make("matmul", "W", "a"), Op.make("matmul", "W", "b")
    ),
    Op.make("matmul", "W", Op.make("add", "a", "b")),
    law="Factoring common linear maps (reverse distributivity).",
    cond=_COND_MM_ADDENDS,
    tags=_CAT,
    derivation=("distribute_matmul_over_add",),
)

# Right-side bilinearity.  In PyTorch, `x @ W` puts the WEIGHT second, so
# the existing left-side rules never match the exported form.  These close
# that gap.  All four require repeated metavariables, which the e-graph
# matcher now enforces as "same e-class" (soundness-tested).

# (a + b) @ W = a@W + b@W
RIGHT_DISTRIBUTE = R(
    "right_distribute_matmul",
    Op.make("matmul", Op.make("add", "a", "b"), "W"),
    Op.make(
        "add", Op.make("matmul", "a", "W"), Op.make("matmul", "b", "W")
    ),
    law="Bilinearity: linear maps distribute over addition in BOTH slots.",
    tags=_CAT,
)

# a@W + b@W = (a + b) @ W
RIGHT_FACTOR = R(
    "right_factor_matmul",
    Op.make(
        "add", Op.make("matmul", "a", "W"), Op.make("matmul", "b", "W")
    ),
    Op.make("matmul", Op.make("add", "a", "b"), "W"),
    law="Factor a shared right-weight (the slot `x @ W` uses).",
    tags=_CAT,
    derivation=("right_distribute_matmul",),
)

# THE WEIGHT-MERGE RULE.  x@W1 + x@W2 = x @ (W1 + W2): two projections of
# the SAME input collapse to one matmul on a summed weight.  This is the
# LoRA/adapter/model-soup merge that deployment tooling does by hand.
# REDUNDANT: alpha-duplicate of factor_matmul (metavar renaming only) —
# the same 2-cell a second time, annotated not deleted (the pair's
# direction coverage lives on the lemma factor_matmul).
WEIGHT_FACTOR = R(
    "weight_factor_matmul",
    Op.make(
        "add", Op.make("matmul", "x", "W"), Op.make("matmul", "x", "W2")
    ),
    Op.make("matmul", "x", Op.make("add", "W", "W2")),
    law="Merge shared-input projections: x@W1 + x@W2 = x@(W1+W2).",
    cond=_COND_MM_WEIGHTS,
    tags=(*_CAT, tags.REDUNDANT),
    derivation=("distribute_matmul_over_add",),
)

# x @ (W1 + W2) = x@W1 + x@W2  [reverse: expand for cost-model choice]
# REDUNDANT: alpha-duplicate of the axiom distribute_matmul_over_add.
WEIGHT_DISTRIBUTE = R(
    "weight_distribute_matmul",
    Op.make("matmul", "x", Op.make("add", "W", "W2")),
    Op.make(
        "add", Op.make("matmul", "x", "W"), Op.make("matmul", "x", "W2")
    ),
    law="Reverse weight merge (lets eqsat weigh fused vs split forms).",
    cond=_COND_MM_WEIGHTS,
    tags=(*_CAT, tags.REDUNDANT),
    derivation=("distribute_matmul_over_add",),
)


# ---------------------------------------------------------------------------
#  `linear` variants — torch.export emits F.linear for every nn.Linear, so
#  the matmul rules above never see the exported form.  F.linear(x, W) is
#  x @ W.T, so transposes distribute over add and flip matrix products.
# ---------------------------------------------------------------------------

# x@W1.T + x@W2.T = x @ (W1+W2).T   ->   linear(x, W1+W2)
WEIGHT_FACTOR_LINEAR = R(
    "weight_factor_linear",
    Op.make(
        "add", Op.make("linear", "x", "W"), Op.make("linear", "x", "W2")
    ),
    Op.make("linear", "x", Op.make("add", "W", "W2")),
    law="Merge shared-input nn.Linears: linear(x,W1)+linear(x,W2)"
    " = linear(x, W1+W2)  (transpose distributes over +).",
    cond=_COND_MM_WEIGHTS,
    tags=_CAT,
    # inverse-pair twin of the axiom weight_distribute_linear (the
    # alphabetical-first member of the {distribute, factor}_linear
    # derivability cycle is the designated kernel representative)
    derivation=("weight_distribute_linear",),
)

# linear(linear(x, A), B) = x @ A.T @ B.T = x @ (B@A).T = linear(x, B@A)
# NOTE the flipped order: fused weight is B @ A, not A @ B.
ASSOC_LINEAR = R(
    "assoc_linear",
    Op.make("linear", Op.make("linear", "x", "A"), "B"),
    Op.make("linear", "x", Op.make("matmul", "B", "A")),
    law="Compose stacked nn.Linears: fused weight is B @ A"
    " (transposes flip the product order).",
    tags=_CAT,
)

# Reverse direction: split a fused weight product back into the
# two-GEMM chain — wins when the inner dim sits below the
# oi/(o+i) break-even (the LoRA / built-low-rank regime).
ASSOC_LINEAR_REV = R(
    "assoc_linear_rev",
    Op.make("linear", "x", Op.make("matmul", "B", "A")),
    Op.make("linear", Op.make("linear", "x", "A"), "B"),
    law="Reverse linear composition: linear(x, B@A) = "
    "linear(linear(x, A), B) (eqsat weighs fused vs split).",
    tags=_CAT,
    derivation=("assoc_linear",),
)

# ------------------------------------------------------------------
#  Biased composition — the affine-map law on the 3-ary `linear`.
#
#  torch.export emits nn.Linear(bias=True) as linear(x, W, b), so the
#  2-ary assoc_linear above never sees the most common stacked-Linear
#  graph.  The composed-bias law is still exact:
#
#      B(Ax + b1) + b2 = (BA)x + (B b1 + b2)
#
#  with the SAME transposed-order weight product B @ A, plus the push
#  of the inner bias through the outer map:  B·b1 + b2.
#
#  NOTE on the RHS spelling.  The textbook form is
#      linear(x, B@A, B@b1 + b2)
#  but that term is unselectable under the cost model: _shape_of has
#  no matvec case, so matmul(B, b1) on a rank-1 bias infers (o, h)
#  instead of (o,), which makes add(B·b1, b2) provably ill-typed
#  (_INVALID) whenever h != o and prices the whole composed member at
#  _INVALID_COST — it could never be extracted.  The associativity-
#  equivalent spelling below keeps every subterm well-typed for ALL
#  h, o: the matvec sits in the linear's bias SLOT (whose inferred
#  shape is ignored by the linear case) and b2 is added at the top
#  level, where the broadcast (…, o) + (o,) is valid.  Both
#  B-products are parameter-only, so _fold_weight_chains materialises
#  (B@A) and (B·b1) at compile time; the runtime keeps one GEMM, one
#  broadcast add, and the b2 leaf.
# ------------------------------------------------------------------


#: Shape guard for the affine compose, as data — the chain dims must
#: compose for B·b1 + b2.  Well-typed when A (h, i), B (o, h), b1 (h,),
#: b2 (o,) or scalar, and x's last dim feeds A's input dim.
#:
#: The matcher cannot see tensor types; without this the rule would
#: also fire on e-nodes whose "bias" slot holds a non-vector term.
#: ``dim-eq`` is *strict* equality on raw dims — unknown dims pass
#: through as equalities on ``None`` (``None == None``), so the guard
#: only vetoes PROVABLE mismatches, the same posture the procedural
#: version took.
_COND_BIAS_COMPOSE = (
    "and",
    ("rank", "A", "==", 2),
    ("rank", "B", "==", 2),
    ("dim-eq", "B", 1, "A", 0),  # B consumes A's out dim
    ("rank", "b1", "==", 1),
    ("dim-eq", "b1", 0, "A", 0),  # inner bias: exactly (h,)
    (
        "or",
        ("scalar", "b2"),
        ("and", ("rank", "b2", "==", 1), ("dim-eq", "b2", 0, "B", 0)),
    ),  # outer bias: scalar|(o,)
    ("rank", "x", ">=", 1),
    ("dim-eq", "x", -1, "A", 1),  # x feeds A's input dim
)

#: Compat alias — the test-facing hook (see ``_check_sum_keepdim``).
_check_linear_bias_compose = as_check(_COND_BIAS_COMPOSE)


ASSOC_LINEAR_BIAS = R(
    "assoc_linear_bias",
    Op.make("linear", Op.make("linear", "x", "A", "b1"), "B", "b2"),
    Op.make(
        "add",
        Op.make(
            "linear",
            "x",
            Op.make("matmul", "B", "A"),
            Op.make("matmul", "B", "b1"),
        ),
        "b2",
    ),
    law="Affine-map composition: (B,b2)∘(A,b1) = (BA, B·b1 + b2).  "
    "Fused weight is B @ A (same transpose flip as assoc_linear); "
    "the fused bias is spelled linear(x, BA, B·b1) + b2 so both "
    "B-products fold at compile time — one GEMM plus one "
    "broadcast add at runtime.",
    cond=_COND_BIAS_COMPOSE,
    tags=_CAT,
)

# Reverse: expand a fused affine member back into the biased chain —
# wins when the hidden dim sits below the oi/(i+o) break-even.
ASSOC_LINEAR_BIAS_REV = R(
    "assoc_linear_bias_rev",
    Op.make(
        "add",
        Op.make(
            "linear",
            "x",
            Op.make("matmul", "B", "A"),
            Op.make("matmul", "B", "b1"),
        ),
        "b2",
    ),
    Op.make("linear", Op.make("linear", "x", "A", "b1"), "B", "b2"),
    law="Reverse affine composition (eqsat weighs fused vs split).",
    cond=_COND_BIAS_COMPOSE,
    tags=_CAT,
    derivation=("assoc_linear_bias",),
)

# a@W.T + b@W.T = (a+b)@W.T   ->   linear(add(a,b), W)
RIGHT_FACTOR_LINEAR = R(
    "right_factor_linear",
    Op.make(
        "add", Op.make("linear", "a", "W"), Op.make("linear", "b", "W")
    ),
    Op.make("linear", Op.make("add", "a", "b"), "W"),
    law="Factor a shared right-hand nn.Linear weight.",
    tags=_CAT,
)

# reverse of weight merge for `linear`
WEIGHT_DISTRIBUTE_LINEAR = R(
    "weight_distribute_linear",
    Op.make("linear", "x", Op.make("add", "W", "W2")),
    Op.make(
        "add", Op.make("linear", "x", "W"), Op.make("linear", "x", "W2")
    ),
    law="Expand a merged nn.Linear so eqsat can compare both forms.",
    cond=_COND_MM_WEIGHTS,
    tags=_CAT,
)

# ---------------------------------------------------------------------------
#  Product structure — the fused-projection rules.
#
#  Categorically: in a category with products, two maps f,g : X -> V with
#  the SAME source pair into a single map <f,g> : X -> V x V (the
#  universal property of the product, mediated by the diagonal Delta).
#  On tensors, pairing two nn.Linears on one input is concatenating the
#  weights along the output dim (one wide GEMM), and the projections
#  pi_i are zero-cost chunks.  This is the MergedColumnParallelLinear /
#  fused-QKV transformation deployment stacks perform manually; a tensor
#  compiler cannot produce it because it must RESHAPE PARAMETERS, which
#  lies outside kernel fusion.
# ---------------------------------------------------------------------------


#: The paired weights share a provably-equal shape, as data.
#:
#: ``concat(A, B, dim=0)`` then ``chunk(·, 2, dim=-1)`` recovers each
#: projection exactly only when the two weights agree on EVERY axis —
#: unequal output dims mis-split the fused GEMM (A (o,i), B (o2,i)
#: with o != o2 makes the first chunk straddle the A/B boundary), and
#: a rank mismatch fails concat outright.  ``None`` dims are
#: wildcards — ``shape-compat`` vetoes only provable mismatches.
#: Without the guard a broadcastable-but-unequal pair (A (4,i),
#: B (1,i)) — whose LHS DOES evaluate — minted a member whose mul
#: operands do not even broadcast (the same matcher-cannot-see-shapes
#: class as the matmul-addend rank guard above).
_COND_FUSE_PAIR = ("shape-compat", "A", "B")

#: Compat alias — the test-facing hook (see ``_check_sum_keepdim``).
_check_fuse_pair = as_check(_COND_FUSE_PAIR)


# silu(x@A.T) * (x@B.T)  ->  y = x@[A;B].T ; silu(y[..., :d]) * y[..., d:]
# The fused linear term is shared (the e-graph stores it once); both
# chunk projections read the same e-class.
SWIGLU_FUSE = R(
    "swiglu_fuse",
    Op.make(
        "mul",
        Op.make("silu", Op.make("linear", "x", "A")),
        Op.make("linear", "x", "B"),
    ),
    Op.make(
        "mul",
        Op.make(
            "silu",
            Op.make(
                "chunk",
                Op.make(
                    "linear", "x", Op.make("concat", "A", "B", dim=0)
                ),
                chunks=2,
                dim=-1,
                index=0,
            ),
        ),
        Op.make(
            "chunk",
            Op.make("linear", "x", Op.make("concat", "A", "B", dim=0)),
            chunks=2,
            dim=-1,
            index=1,
        ),
    ),
    law="Product universal property: <f,g> = (f x g) . Delta.  Two "
    "projections of the same input are ONE GEMM into V x V, then "
    "project.  (Fused SwiGLU gate/up — MergedColumnParallelLinear.)",
    cond=_COND_FUSE_PAIR,
    tags=_SUB,
)

# (x@A.T) * (x@B.T)  ->  chunk form without the gate nonlinearity.
# Covers GLU-style variants and any elementwise-mul pair of parallel
# projections.
PARALLEL_MUL_FUSE = R(
    "parallel_mul_fuse",
    Op.make(
        "mul", Op.make("linear", "x", "A"), Op.make("linear", "x", "B")
    ),
    Op.make(
        "mul",
        Op.make(
            "chunk",
            Op.make("linear", "x", Op.make("concat", "A", "B", dim=0)),
            chunks=2,
            dim=-1,
            index=0,
        ),
        Op.make(
            "chunk",
            Op.make("linear", "x", Op.make("concat", "A", "B", dim=0)),
            chunks=2,
            dim=-1,
            index=1,
        ),
    ),
    law="Pairing without a gate nonlinearity: mul(<pi1 f>, <pi2 g>) "
    "recovers the parallel-product form.",
    cond=_COND_FUSE_PAIR,
    tags=_SUB,
)

# ---------------------------------------------------------------------------
#  Diagonal-scale naturality for `linear` — the RMSNorm-folding rules.
#
#  Two kinds of broadcast scale commute through F.linear differently:
#
#  * CHANNEL scale c ~ (in,) folds INTO the weight:
#        linear(x∘c, W) = (x∘c)@W.T = x@(W∘c).T = linear(x, W∘c)
#    This is the classic "fold the norm's affine gain into the next
#    linear" deployment trick — the diagonal D=diag(c) satisfies
#    (xD)W = x(DW), i.e. a right action absorbed at compile time.
#
#  * ROW scale r ~ (B,T,1) hoists OUT:
#        linear(x∘r, W) = diag(r)·(x@W.T) = linear(x,W) ∘ r
#    A left diagonal commutes with any linear map — naturality of the
#    scalar action.
#
#  CAVEAT: the matcher cannot check broadcast shapes.  Applied to a
#  term where c is sized `out` and out != in, the RHS is ill-typed;
#  the cost model prices provably-ill-typed terms at _INVALID_COST so
#  they can never win, and the verifier is the last line of defence.
# ---------------------------------------------------------------------------

#: Per-CHANNEL scale as data: c broadcasts over the weight's input
#: dim — scalar, ``(in,)``, or ``(1,...,1,in)`` — i.e. every non-last
#: dim of ``c`` is 1 and the last equals ``W``'s last.  (Same verdict
#: as ``base._is_channel_scale``; unshaped ``c``/``W`` declines.)
_COND_CHANNEL_SCALE = (
    "and",
    ("shaped", "c"),
    ("rank", "W", ">=", 1),
    (
        "or",
        ("scalar", "c"),
        ("and", ("dim-eq", "c", -1, "W", -1), ("ones-but-last", "c")),
    ),
)

#: Per-ROW scale as data: broadcasts to ``(B,T,1)`` — scalar or last
#: dim 1.  (Same verdict as ``base._is_row_scale``.)
_COND_ROW_SCALE = ("or", ("scalar", "r"), ("dim-eq-const", "r", -1, 1))

LINEAR_CHANNEL_SCALE = R(
    "linear_channel_scale",
    Op.make("linear", Op.make("mul", "x", "c"), "W"),
    Op.make("linear", "x", Op.make("mul", "W", "c")),
    law="Channel scale is a right diagonal: (xD)W = x(DW).  Folds the "
    "norm's affine gain into the weight at compile time.",
    cond=_COND_CHANNEL_SCALE,
    tags=_SYM,
)

LINEAR_CHANNEL_SCALE_REV = R(
    "linear_channel_scale_rev",
    Op.make("linear", "x", Op.make("mul", "W", "c")),
    Op.make("linear", Op.make("mul", "x", "c"), "W"),
    law="Reverse channel-scale fold (eqsat compares both forms).",
    cond=_COND_CHANNEL_SCALE,
    tags=_SYM,
    derivation=("linear_channel_scale",),
)

LINEAR_ROW_SCALE = R(
    "linear_row_scale",
    Op.make("linear", Op.make("mul", "x", "r"), "W"),
    Op.make("mul", Op.make("linear", "x", "W"), "r"),
    law="Row scale is a left diagonal: commutes through the linear map "
    "to the output (naturality of scalar action).",
    cond=_COND_ROW_SCALE,
    tags=_SYM,
)

LINEAR_ROW_SCALE_REV = R(
    "linear_row_scale_rev",
    Op.make("mul", Op.make("linear", "x", "W"), "r"),
    Op.make("linear", Op.make("mul", "x", "r"), "W"),
    law="Reverse row-scale hoist (eqsat compares both forms).",
    cond=_COND_ROW_SCALE,
    tags=_SYM,
    derivation=("linear_row_scale",),
)


# ---------------------------------------------------------------------------
#  Fused QKV — the product rule applied to attention's three projections.
#
#  sdpa( f(linear(x,Q)), f(linear(x,K)), f(linear(x,V)) )
#    where f = transpose ∘ view is the head-splitting view
#  -->
#  y = linear(x, cat(cat(Q,K),V)) ;
#  sdpa( f(chunk0(y)), f(chunk1(y)), f(chunk2(y)) )
#
#  The view shape and the transpose dims are ATTRIBUTE metavariables
#  (string values in the pattern bind the node's concrete attrs), which
#  is what makes this rule shape-polymorphic.
# ---------------------------------------------------------------------------


def _head(t: Any) -> Op:
    """Build the head-splitting view: view(t, S) then transpose(1, 2)."""
    return Op.make(
        "transpose", Op.make("reshape", t, shape="S"), dim0=1, dim1=2
    )


QKV_FUSE = R(
    "qkv_fuse",
    Op.make(
        "sdpa",
        _head(Op.make("linear", "x", "Q")),
        _head(Op.make("linear", "x", "K")),
        _head(Op.make("linear", "x", "V")),
        scale="SC",
    ),
    Op.make(
        "sdpa",
        _head(
            Op.make(
                "chunk",
                Op.make(
                    "linear",
                    "x",
                    Op.make(
                        "concat",
                        Op.make("concat", "Q", "K", dim=0),
                        "V",
                        dim=0,
                    ),
                ),
                chunks=3,
                dim=-1,
                index=0,
            )
        ),
        _head(
            Op.make(
                "chunk",
                Op.make(
                    "linear",
                    "x",
                    Op.make(
                        "concat",
                        Op.make("concat", "Q", "K", dim=0),
                        "V",
                        dim=0,
                    ),
                ),
                chunks=3,
                dim=-1,
                index=1,
            )
        ),
        _head(
            Op.make(
                "chunk",
                Op.make(
                    "linear",
                    "x",
                    Op.make(
                        "concat",
                        Op.make("concat", "Q", "K", dim=0),
                        "V",
                        dim=0,
                    ),
                ),
                chunks=3,
                dim=-1,
                index=2,
            )
        ),
        scale="SC",
    ),
    law="Triple pairing <q,k,v> : X -> V^3 — three projections of the "
    "same input are ONE GEMM into the product space, then three "
    "zero-cost chunk projections.  (Fused QKV.)",
    tags=_SUB,
)


# ---------------------------------------------------------------------------
#  Asymmetric fused QKV (GQA): q/k/v projections with DIFFERENT output
#  dims.  The product law is identical; only the projections differ —
#  `split` with explicit sizes instead of `chunk`.  The sizes are not
#  present anywhere in the LHS, so a `derive` hook computes them from
#  the bound weight shapes.  (vLLM's QKVParallelLinear.)
# ---------------------------------------------------------------------------


def _head_v(t: Any, shape_var: Any) -> Op:
    """Head view with a per-projection shape metavariable."""
    return Op.make(
        "transpose",
        Op.make("reshape", t, shape=shape_var),
        dim0=1,
        dim1=2,
    )


#: Sizes = (|Q|, |K|, |V|) — each bound weight's *declared* output
#: dim (``leaf-dim`` reads ``.typ.shape``, never inferred: an ``Op``
#: binding or a ``None`` dim declines, exactly the old hook's veto).
_DSPEC_SPLIT_SIZES = {
    "SZ": (
        "tuple",
        ("leaf-dim", "Q", 0),
        ("leaf-dim", "K", 0),
        ("leaf-dim", "V", 0),
    ),
}

#: Compat alias — the test-facing hook (see ``_derive_softmax_dim``).
_derive_split_sizes = as_derive(_DSPEC_SPLIT_SIZES)


_QKV_CAT = Op.make(
    "concat", Op.make("concat", "Q", "K", dim=0), "V", dim=0
)

QKV_FUSE_ASYM = R(
    "qkv_fuse_asym",
    Op.make(
        "sdpa",
        _head_v(Op.make("linear", "x", "Q"), "S1"),
        _head_v(Op.make("linear", "x", "K"), "S2"),
        _head_v(Op.make("linear", "x", "V"), "S3"),
        scale="SC",
        enable_gqa="G",
    ),
    Op.make(
        "sdpa",
        _head_v(
            Op.make(
                "split",
                Op.make("linear", "x", _QKV_CAT),
                sizes="SZ",
                dim=-1,
                index=0,
            ),
            "S1",
        ),
        _head_v(
            Op.make(
                "split",
                Op.make("linear", "x", _QKV_CAT),
                sizes="SZ",
                dim=-1,
                index=1,
            ),
            "S2",
        ),
        _head_v(
            Op.make(
                "split",
                Op.make("linear", "x", _QKV_CAT),
                sizes="SZ",
                dim=-1,
                index=2,
            ),
            "S3",
        ),
        scale="SC",
        enable_gqa="G",
    ),
    law="Asymmetric triple pairing: the same product law as qkv_fuse, "
    "but the three projections have different output dims — one "
    "GEMM, three uneven split views.  (GQA fused QKV.)",
    dspec=_DSPEC_SPLIT_SIZES,
    tags=_SUB,
)


# ---------------------------------------------------------------------------
#  Copy-map absorption:  unsqueeze -> expand -> reshape  is the diagonal
#  Delta_r (duplicate along a new axis, then merge it back = PyTorch's
#  repeat_kv / repeat_interleave).  The diagonal is a *natural* map: it
#  can be pushed inside a consumer that implements the broadcast
#  internally.  SDPA's enable_gqa flag IS that consumer — feeding it the
#  unexpanded k/v computes the same attention without materialising
#  the duplicated heads.  A term-local tensor pass cannot see this: the
#  pattern lives across three view ops plus a fused kernel flag.
# ---------------------------------------------------------------------------


def _check_repeat_chain(bound: dict, pre: str) -> bool:
    """Check reshape(expand(unsqueeze(t, d))) is repeat_interleave.

    Must be exactly repeat_interleave on dim d-1: unsqueeze inserts a 1,
    expand broadcasts only that dim by r, and the reshape merges dims
    d-1,d into one.
    """
    from catopt_core.typing import _shape_of as _so

    d = bound.get(f"$attr:UD{pre}")
    es = bound.get(f"$attr:ES{pre}")
    rs = bound.get(f"$attr:RS{pre}")
    base = bound.get(pre)
    bs = _so(base)
    if not (
        isinstance(d, int)
        and isinstance(es, tuple)
        and isinstance(rs, tuple)
        and isinstance(bs, tuple)
    ):
        return False
    if any(x is None for x in bs):
        return False
    nd = len(bs)
    d = d % (nd + 1)
    us = bs[:d] + (1,) + bs[d:]  # noqa: RUF005
    if len(es) != len(us) or len(rs) != nd or d == 0:
        return False
    r = es[d]
    if not isinstance(r, int) or r <= 1:
        return False
    if any(es[i] != us[i] for i in range(len(us)) if i != d):
        return False  # expand may only grow the inserted dim
    merged = us[: d - 1] + ((us[d - 1] or 0) * r,) + us[d + 1 :]  # noqa: RUF005
    return rs == merged


def _check_gqa_absorb(bound: dict) -> bool:
    """Require k and v to be repeat-chains with the same factor r.

    q's head count must equal kv_heads * r.
    """
    from catopt_core.typing import _shape_of as _so

    for side in ("k", "v"):
        if not _check_repeat_chain(bound, side):
            return False
    if bound["$attr:ESk"] != bound["$attr:ESv"]:
        return False
    d = bound["$attr:UDk"] % (len(cast("tuple", _so(bound["k"]))) + 1)
    r = bound["$attr:ESk"][d]
    qs, ks = _so(bound["q"]), _so(bound["k"])
    if not (
        isinstance(qs, tuple)
        and isinstance(ks, tuple)
        and len(qs) >= 2
        and len(ks) >= 2
    ):
        return False
    if None in qs or None in ks:
        return False
    return qs[-2] == ks[-2] * r  # hq == hkv * n_rep


_REPEAT_KV = Op.make(
    "transpose",
    Op.make(
        "reshape",
        Op.make(
            "expand", Op.make("unsqueeze", "k", dim="UDk"), shape="ESk"
        ),
        shape="RSk",
    ),
    dim0=1,
    dim1=2,
)

_REPEAT_V = Op.make(
    "transpose",
    Op.make(
        "reshape",
        Op.make(
            "expand", Op.make("unsqueeze", "v", dim="UDv"), shape="ESv"
        ),
        shape="RSv",
    ),
    dim0=1,
    dim1=2,
)

GQA_ABSORB = R(
    "gqa_absorb_repeat",
    Op.make(
        "sdpa",
        Op.make("transpose", "q", dim0=1, dim1=2),
        _REPEAT_KV,
        _REPEAT_V,
        arg4="D",
        arg5="C",
    ),
    Op.make(
        "sdpa",
        Op.make("transpose", "q", dim0=1, dim1=2),
        Op.make("transpose", "k", dim0=1, dim1=2),
        Op.make("transpose", "v", dim0=1, dim1=2),
        arg4="D",
        arg5="C",
        arg7=True,
    ),
    law="The diagonal is natural: unsqueeze->expand->reshape copies each "
    "kv head r times (repeat_kv).  SDPA implements that copy inside "
    "the kernel via enable_gqa — pushing Delta into the consumer "
    "deletes the materialisation entirely.",
    check=_check_gqa_absorb,
    tags=_FUS,
)


# ---------------------------------------------------------------------------
#  Softmax-attention fold — the flash-attention transform.
#
#  softmax(q @ k^T * s [+ mask | masked_fill(mask, -inf)]) @ v
#      ==  sdpa(q, k, v, attn_mask=..., scale=s)
#
#  This is the *definition* of scaled_dot_product_attention, so the fold
#  is sound for ANY mask term: additive masks pass straight through,
#  boolean fill masks invert to keep-masks via logical_not.  Inductor
#  fuses the softmax elementwise chain but (for the masked_fill form)
#  never recognises the enclosing matmul pair as SDPA — the pattern
#  spans a softmax nonlinearity and two matmuls.
# ---------------------------------------------------------------------------

_QK_SCORES = Op.make(
    "matmul", "Q", Op.make("transpose", "K", dim0="TD1", dim1="TD2")
)


def _const_val(t):
    return getattr(t, "value", None)


#: K transposed on its last two dims — ``matmul(q, k^T)`` — as data.
#: ``concrete`` asks for a tuple of int dims (the ``% nd`` arithmetic
#: the old check did needs a concrete rank); ``axes-last2`` then does
#: the same normalized-pair ``== {nd-2, nd-1}`` comparison, with the
#: rank-0 crash case degrading to a plain decline.
_COND_SCORE_T = (
    "and",
    ("concrete", "K"),
    ("attr-type", "TD1", "int"),
    ("attr-type", "TD2", "int"),
    ("axes-last2", "K", "TD1", "TD2"),
)

#: Softmax over the last dim (keys) of the score matrix, as data.
_COND_SM_DIM = ("axis", "Q", "SD", -1)

#: The shared precondition of every sdpa_fold rule.
_COND_SDPA_BASE = ("and", _COND_SCORE_T, _COND_SM_DIM)

#: ``+ a numeric scale leaf (the ``S`` Const).``
_COND_SDPA_SCALED = ("and", _COND_SDPA_BASE, ("const-num", "S"))

#: ``+ a boolean masked_fill marker (the ``F`` Const < -1e30).``
_COND_SDPA_MF = ("and", _COND_SDPA_BASE, ("const-cmp", "F", "<", -1e30))

#: masked_fill + scale.
_COND_SDPA_MF_SCALED = ("and", _COND_SDPA_MF, ("const-num", "S"))

#: Compat aliases — the test-facing hooks (see ``_check_sum_keepdim``).
_check_score_transpose = as_check(_COND_SCORE_T)
_check_softmax_dim = as_check(_COND_SM_DIM)
_check_sdpa_base = as_check(_COND_SDPA_BASE)
_check_sdpa_scaled = as_check(_COND_SDPA_SCALED)
_check_sdpa_mf = as_check(_COND_SDPA_MF)
_check_sdpa_mf_scaled = as_check(_COND_SDPA_MF_SCALED)


def _scale_of(bound):
    s = _const_val(bound.get("S"))
    return float(s) if isinstance(s, (int, float)) else None


#: The ``scale`` attr ``sdpa`` takes — minted from the bound scalar
#: leaf ``S`` (mul form), its reciprocal (div form), or the literal
#: 1.0 (scale-free form).  A non-numeric ``S`` declines through the
#: ``float`` op — the old hooks' ``isinstance`` veto verbatim.
_DSPEC_SCALE_MUL = {"SC": ("float", ("const", "S"))}
_DSPEC_SCALE_DIV = {"SC": ("recip", ("float", ("const", "S")))}
_DSPEC_SCALE_ONE = {"SC": 1.0}

#: Compat aliases — the test-facing hooks (see ``_derive_softmax_dim``).
_derive_scale_mul = as_derive(_DSPEC_SCALE_MUL)
_derive_scale_div = as_derive(_DSPEC_SCALE_DIV)
_derive_scale_one = as_derive(_DSPEC_SCALE_ONE)


def _make_sdpa_fold_rules() -> list:
    """6 mask/scale forms × optional eval-mode dropout wrapper."""
    out = []
    scaled = (
        (
            "mul",
            lambda: Op.make("mul", _QK_SCORES, "S"),
            _COND_SDPA_SCALED,
            _COND_SDPA_MF_SCALED,
            _DSPEC_SCALE_MUL,
        ),
        (
            "div",
            lambda: Op.make("div", _QK_SCORES, "S"),
            _COND_SDPA_SCALED,
            _COND_SDPA_MF_SCALED,
            _DSPEC_SCALE_DIV,
        ),
        (
            "",
            lambda: _QK_SCORES,
            _COND_SDPA_BASE,
            _COND_SDPA_MF,
            _DSPEC_SCALE_ONE,
        ),
    )
    wraps = (
        ("", lambda sm: sm),
        (
            "_drop",
            lambda sm: Op.make("dropout", sm, p="DP", train="DT"),
        ),
    )
    for sname, scores, cond_add, cond_mf, dspec in scaled:
        for wname, wrap in wraps:
            sm = lambda inner: wrap(  # noqa: E731, B023
                Op.make("softmax", inner, dim="SD")
            )
            out.append(
                R(
                    f"sdpa_fold_add{sname}{wname}",
                    Op.make(
                        "matmul", sm(Op.make("add", scores(), "M")), "V"
                    ),
                    Op.make("sdpa", "Q", "K", "V", "M", scale="SC"),
                    law="softmax(qk^T s + m) v is sdpa — the additive mask is "
                    "the kernel's attn_mask argument.",
                    cond=cond_add,
                    dspec=dspec,
                    tags=_FUS,
                )
            )
            out.append(
                R(
                    f"sdpa_fold_masked_fill{sname}{wname}",
                    Op.make(
                        "matmul",
                        sm(Op.make("masked_fill", scores(), "MK", "F")),
                        "V",
                    ),
                    Op.make(
                        "sdpa",
                        "Q",
                        "K",
                        "V",
                        Op.make("logical_not", "MK"),
                        scale="SC",
                    ),
                    law="masked_fill(m, -inf) before softmax is a boolean "
                    "attn_mask — logical_not turns the fill-mask into "
                    "SDPA's keep-mask.",
                    cond=cond_mf,
                    dspec=dspec,
                    tags=_FUS,
                )
            )
    return out


SDPA_FOLD_RULES: list = _make_sdpa_fold_rules()

#: Scalar factor as data — the bound term's shape is ``()``.  (Same
#: verdict as ``base._is_scalar``.)
_COND_SCALAR = ("scalar", "c")

# matmul(W, mul(x, c)) = mul(matmul(W, x), c)
# KEY RULE: naturality of scalar multiplication w.r.t. linear maps.
# Lets the optimizer slide an elementwise scaling past a matmul.
NATURALITY_SCALAR = R(
    "naturality_scalar",
    Op.make("matmul", "W", Op.make("mul", "x", "c")),
    Op.make("mul", Op.make("matmul", "W", "x"), "c"),
    law="Naturality: scalar multiplication commutes with linear maps.",
    cond=_COND_SCALAR,
    tags=_CAT,
)

NATURALITY_SCALAR_REV = R(
    "naturality_scalar_rev",
    Op.make("mul", Op.make("matmul", "W", "x"), "c"),
    Op.make("matmul", "W", Op.make("mul", "x", "c")),
    law="Reverse naturality: pull scalar into the matmul's input.",
    cond=_COND_SCALAR,
    tags=_CAT,
    derivation=("naturality_scalar",),
)

# (A @ B) @ C = A @ (B @ C)  — associativity of composition
ASSOC_MATMUL = R(
    "assoc_matmul",
    Op.make("matmul", "A", Op.make("matmul", "B", "C")),
    Op.make("matmul", Op.make("matmul", "A", "B"), "C"),
    law="Associativity of composition in a category: (f∘g)∘h = f∘(g∘h).",
    tags=_CAT,
)

# Reverse direction: explore the other association
ASSOC_MATMUL_REV = R(
    "assoc_matmul_rev",
    Op.make("matmul", Op.make("matmul", "A", "B"), "C"),
    Op.make("matmul", "A", Op.make("matmul", "B", "C")),
    law="Reverse associativity: f∘(g∘h) = (f∘g)∘h.",
    tags=_CAT,
    derivation=("assoc_matmul",),
)
# ---------------------------------------------------------------------------
#  Rule collections
# ---------------------------------------------------------------------------

#: Rules that implement basic algebraic simplification (monoid, group).
SIMPLIFICATION_RULES: list[Rewrite] = [
    COMM_ADD,
    COMM_MUL,
    ASSOC_ADD,
    ASSOC_MUL,
    ID_ADD,
    ID_MUL,
    DOUBLE_NEG,
    SUB_TO_ADD,
    SILU_EXPAND,
    SILU_FOLD,
    SILU_MUL_FORM,
    SQUARE_EXPAND,
    POW_TO_SQUARE,
    SQUARE_TO_POW,
    MUL_SQUARE,
    SELECT_MUL,
    SOFTMAX_FOLD,
    GLU_FOLD,
    RMS_NORM_FOLD,
    RMS_NORM_FOLD_NOGAIN,
    DIV_SQRT_TO_RSQRT,
    POW_TO_RSQRT,
    RECIP_SQRT_TO_RSQRT,
]

#: Rules that implement the categorical insight: distributivity and naturality.
CATEGORICAL_RULES: list[Rewrite] = [
    DISTRIBUTE_MUL,
    FACTOR_MUL,
    RIGHT_DISTRIBUTE,
    RIGHT_FACTOR,
    WEIGHT_FACTOR,
    WEIGHT_DISTRIBUTE,
    # nn.Linear / F.linear forms (what torch.export actually emits)
    WEIGHT_FACTOR_LINEAR,
    WEIGHT_DISTRIBUTE_LINEAR,
    RIGHT_FACTOR_LINEAR,
    ASSOC_LINEAR,
    ASSOC_LINEAR_REV,
    ASSOC_LINEAR_BIAS,
    ASSOC_LINEAR_BIAS_REV,
    NATURALITY_SCALAR,
    NATURALITY_SCALAR_REV,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    # Product structure (fused projections)
    SWIGLU_FUSE,
    PARALLEL_MUL_FUSE,
    QKV_FUSE,
    QKV_FUSE_ASYM,
    # Diagonal-scale naturality (norm folding)
    LINEAR_CHANNEL_SCALE,
    LINEAR_CHANNEL_SCALE_REV,
    LINEAR_ROW_SCALE,
    LINEAR_ROW_SCALE_REV,
    # Diagonal-map absorption (copy pushed inside the kernel)
    GQA_ABSORB,
    # Softmax-attention fold (flash-attention transform)
    *SDPA_FOLD_RULES,
]
#: All rules combined — the default saturation set.
#:
#: The axiom/lemma split (measured by ``tools/law_coherence.py
#: --emit-basis``, documented in
#: ``project/retros/axiom-lemma-split.md``): 46 of these 61 rules are
#: kernel members — ``kind == "axiom"`` — and 15 carry a recorded
#: ``derivation`` from the kernel (13 ``"lemma"`` — inverse twins
#: whose direction buys reach, plus the emergent ``silu_mul_form`` —
#: and 2 ``"redundant"`` alpha-duplicate spellings tagged
#: ``tags.REDUNDANT``).  The kernel itself is the 32 primitives plus
#: one designated representative (alphabetically first) per
#: derivability cycle.
#:
#: LAYOUT_RULES are deliberately NOT in the default: measured on the
#: laws_effect bench they deliver runtime parity (the NT-GEMM form is
#: structurally correct but not faster), while their bidirectional
#: transpose↔pointwise pairs explode the saturation closure
#: (~10–40× search wall on a quarter-B model, enough to push a T=8192
#: block past a 4 GB enode budget and lose the whole cell).  Opt-in
#: via :data:`ALL_RULES_WITH_LAYOUT` or ``eg.run(LAYOUT_RULES, …)``.
ALL_RULES: list[Rewrite] = SIMPLIFICATION_RULES + CATEGORICAL_RULES

#: The full set including layout migration — for callers that want
#: the wider search space and can pay the closure cost.
ALL_RULES_WITH_LAYOUT: list[Rewrite] = ALL_RULES + LAYOUT_RULES


def all_rules() -> list[Rewrite]:
    """Return a fresh list of all rewrite rules."""
    return list(ALL_RULES)
