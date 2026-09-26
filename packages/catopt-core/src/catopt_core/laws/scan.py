"""Scan-monoid laws: the affine and diagonal-affine carriers.

A recurrence step ``h ↦ A·h + x`` is an affine map; affine maps form a
monoid under composition, and the balanced tree (parallel scan,
Blelloch) is another bracketing of the same product — reachable by
associativity alone once steps are lifted into the carrier domain.

* ``SCAN_LAWS`` — the dense affine monoid (``aff``/``aff_compose``/
  ``apply``).
* ``SCAN_DIAG_LAWS`` — the diagonal-affine monoid for elementwise /
  Mamba-faithful SSM steps (``aff_diag``/``affd_compose``/``applyd``),
  including the unit lift for pure accumulations.
"""

# ruff: noqa: RUF003 -- comments/docstrings use
# mathematical notation (⊙, ×, ↦) deliberately.

from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op
from catopt_core.laws.base import R, _shape_of

# ---------------------------------------------------------------------------
# Scan monoid: affine-map domain
# ---------------------------------------------------------------------------
# A recurrence step ``h ↦ A·h + x`` is an affine map.  Affine maps form
# a monoid under composition:
#     (A2,b2) ∘ (A1,b1) = (A2·A1, A2·b1 + b2)
# A sequential fold is the left-associated composition; the balanced
# tree (parallel scan, Blelloch) is another bracketing of the same
# product — reachable by associativity alone once steps are lifted
# into the affine domain.  Matmul/add algebra alone provably cannot
# reach it: the pair (partial-product, partial-sum) is a cross-class
# object no term law synthesises (measured: order-preserving laws
# plateau at ~1.5·T depth; the monoid reaches ~log T).
#
# ``aff(A, b)``      — the map h ↦ A·h + b (a pair value, not a tensor)
# ``aff_compose(f,g)`` — f∘g as an affine object
# ``apply(f, h)``    — evaluate the map on h (back in tensor-land)

AFF_LIFT = R(
    "aff_lift",
    Op.make("add", Op.make("matmul", "A", "h"), "x"),
    Op.make("apply", Op.make("aff", "A", "x"), "h"),
    law="recurrence step is affine-map application",
)

AFF_LIFT_STEP = R(
    "aff_lift_step",
    Op.make(
        "add", Op.make("matmul", "A", Op.make("apply", "f", "h")), "x"
    ),
    Op.make(
        "apply",
        Op.make("aff_compose", Op.make("aff", "A", "x"), "f"),
        "h",
    ),
    law="compose step with the preceding map",
)

AFF_UNLIFT = R(
    "aff_unlift",
    Op.make("apply", Op.make("aff", "A", "x"), "h"),
    Op.make("add", Op.make("matmul", "A", "h"), "x"),
    law="affine application unfolds",
)

AFF_COMPOSE_UNFOLD = R(
    "aff_compose_unfold",
    Op.make("apply", Op.make("aff_compose", "f", "g"), "h"),
    Op.make("apply", "f", Op.make("apply", "g", "h")),
    law="composition is sequential application",
)

AFF_ASSOC = R(
    "aff_assoc",
    Op.make("aff_compose", Op.make("aff_compose", "f", "g"), "h"),
    Op.make("aff_compose", "f", Op.make("aff_compose", "g", "h")),
    law="affine composition is associative",
)

AFF_ASSOC_REV = R(
    "aff_assoc_rev",
    Op.make("aff_compose", "f", Op.make("aff_compose", "g", "h")),
    Op.make("aff_compose", Op.make("aff_compose", "f", "g"), "h"),
    law="affine composition is associative",
)

#: Minimal law set for scan discovery.  Deliberately excludes
#: ``comm_add``: commutativity is the explosive law (permutation space)
#: and Blelloch reassociation is order-preserving.
SCAN_LAWS: list[Rewrite] = [
    AFF_LIFT,
    AFF_LIFT_STEP,
    AFF_UNLIFT,
    AFF_COMPOSE_UNFOLD,
    AFF_ASSOC,
    AFF_ASSOC_REV,
]

# ---------------------------------------------------------------------------
#  Scan monoid: diagonal-affine domain (elementwise / Mamba-faithful SSMs)
# ---------------------------------------------------------------------------
# ``AFF_LIFT`` only sees steps spelled ``add(matmul(A, h), x)``.  A
# Mamba-faithful selective step ``h ↦ a ⊙ h + x`` is a DIAGONAL affine
# map — the same monoid restricted to diagonal linear parts:
#
#     (a2,b2) ∘ (a1,b1) = (a2⊙a1, a2⊙b1 + b2)
#
# a strictly CHEAPER carrier: O(d) elementwise work per compose instead
# of a dense d×d product.  The step exports as ``add(mul(a,h), x)``
# (for ``DiagonalSSM`` the translation x is itself ``mul(b_t, x_t)`` —
# the metavariable binds it whole, so no second LHS shape is needed),
# which ``AFF_LIFT`` cannot see.  These rules mirror ``SCAN_LAWS``
# verbatim in structure:
#
# ``aff_diag(a, b)``     — the map h ↦ a⊙h + b (a pair value)
# ``affd_compose(f, g)`` — f∘g in the diagonal-affine monoid
# ``applyd(f, h)``       — evaluate: f₀⊙h + f₁ (back in tensor-land)


def _affd_state_like(bound: dict) -> bool:
    """Side condition for the diagonal lifts: the ``h`` binding must be
    state-shaped — a previous step's ``add``/``sub`` spine, an already
    lifted application (``applyd``/``apply``), or a leaf (the h0 Param
    or a free Var).  Per-step vectors (a_t, b_t, x_t — select/mul
    terms) are NOT states.

    The check exists for e-graph economy, not soundness — the rewrite
    a⊙h + x ≡ applyd(aff_diag(a,x), h) is valid for ANY h.  Without it,
    the operand-position variants below would each fire a useless
    sideways lift binding an input vector as "h"."""
    t = bound.get("h")
    if isinstance(t, Op):
        return t.op in ("add", "sub", "apply", "applyd")
    return True


# --- the four operand positions --------------------------------------
# add and mul are commutative, and ``meta.canonicalize`` normalises
# operand ORDER (children sorted by op_repr): DiagonalSSM's steps
# canonicalise to ``add(mul(h, a), mul(b, x))`` for t > 0 but
# ``add(mul(b, x), mul(a, h0))`` for the first step.  Since comm_add /
# comm_mul are deliberately absent from the law set, each position the
# state-mul can occupy gets its own LHS so the lift fires on both
# raw-exported AND canonicalised terms.  ``_affd_state_like`` keeps the
# cross-bindings (state vs input swapped) from firing spuriously, so
# exactly one variant fires per ``add`` e-node.

AFFD_LIFT = R(
    "affd_lift",
    Op.make("add", Op.make("mul", "a", "h"), "x"),
    Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
    law="diagonal recurrence step is diagonal-affine "
    "application: a⊙h + x = (aff_diag(a,x))(h)",
    check=_affd_state_like,
)

AFFD_LIFT_SWAP = R(
    "affd_lift_swap",
    Op.make("add", Op.make("mul", "h", "a"), "x"),
    Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
    law="mul-order variant of affd_lift (canonicalised "
    "terms put the state operand first)",
    check=_affd_state_like,
)

AFFD_LIFT_POST = R(
    "affd_lift_post",
    Op.make("add", "x", Op.make("mul", "a", "h")),
    Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
    law="add-order variant of affd_lift (state-mul in "
    "the second add slot)",
    check=_affd_state_like,
)

AFFD_LIFT_POST_SWAP = R(
    "affd_lift_post_swap",
    Op.make("add", "x", Op.make("mul", "h", "a")),
    Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
    law="remaining operand position of affd_lift",
    check=_affd_state_like,
)

# The step rules need no side condition: the ``applyd`` inside the mul
# already pins the state operand — an input e-class contains no
# ``applyd`` enode, so only the true direction matches.
AFFD_LIFT_STEP = R(
    "affd_lift_step",
    Op.make(
        "add", Op.make("mul", "a", Op.make("applyd", "f", "h")), "x"
    ),
    Op.make(
        "applyd",
        Op.make("affd_compose", Op.make("aff_diag", "a", "x"), "f"),
        "h",
    ),
    law="compose step with the preceding map",
)

AFFD_LIFT_STEP_SWAP = R(
    "affd_lift_step_swap",
    Op.make(
        "add", Op.make("mul", Op.make("applyd", "f", "h"), "a"), "x"
    ),
    Op.make(
        "applyd",
        Op.make("affd_compose", Op.make("aff_diag", "a", "x"), "f"),
        "h",
    ),
    law="mul-order variant of affd_lift_step",
)

AFFD_LIFT_STEP_POST = R(
    "affd_lift_step_post",
    Op.make(
        "add", "x", Op.make("mul", "a", Op.make("applyd", "f", "h"))
    ),
    Op.make(
        "applyd",
        Op.make("affd_compose", Op.make("aff_diag", "a", "x"), "f"),
        "h",
    ),
    law="add-order variant of affd_lift_step",
)

AFFD_LIFT_STEP_POST_SWAP = R(
    "affd_lift_step_post_swap",
    Op.make(
        "add", "x", Op.make("mul", Op.make("applyd", "f", "h"), "a")
    ),
    Op.make(
        "applyd",
        Op.make("affd_compose", Op.make("aff_diag", "a", "x"), "f"),
        "h",
    ),
    law="remaining operand position of affd_lift_step",
)

# --- unit lift: pure accumulation -----------------------------------
# ``add(h, x)`` — the recurrence ``h_t = h_{t-1} + x_t`` (cumsum,
# running statistics, linear attention's KV state ``S_t = S_{t-1} +
# k_t v_tᵀ``) — is the ``a ≡ 1`` degenerate case of the diagonal step:
# ``1⊙h + x = h + x``.  No ``mul`` enode exists for ``AFFD_LIFT`` to
# see, so additive accumulations were unreachable by the scan monoid.
# The unit lift writes the step as ``applyd(aff_diag(1, x), h)``; once
# carried, the ordinary step/assoc machinery (affd_compose balancing,
# trace lift, the batched executor) applies verbatim.
#
# The unit ``1`` is spelled ``expand(Const(1.0), shape=US)`` — the
# broadcast shape of the add — NOT a bare scalar: ``aff_diag`` leaves
# must carry the state's ``(d,)`` shape for ``scan_lower``'s
# ``_leaf_shapes_consistent`` (the batched executor *stacks* leaf
# a-parts; a scalar-shaped a would fail the check and bar the whole
# carrier tree from BatchedScanModule AND trace_lift's carrier path).
# ``US`` is an attribute metavariable filled by ``_derive_affd_unit``,
# which also vetoes the firing when the bound shapes are not concrete.


def _affd_unit_state_like(bound: dict) -> bool:
    """Side condition for the unit lifts: the ``h`` binding must be
    state-shaped — a previous step's ``add``/``sub`` spine, an already
    lifted application (``applyd``/``apply``), or a leaf (the h0 Param
    or a free Var).  Same economy guard as ``_affd_state_like``, plus a
    ``Const`` exclusion: a scalar offset is not an accumulating state.
    Per-step increments (select/mul terms) are NOT states."""
    t = bound.get("h")
    if isinstance(t, Op):
        return t.op in ("add", "sub", "apply", "applyd")
    return not isinstance(t, Const)


def _derive_affd_unit(bound: dict) -> dict | None:
    """``US`` := broadcast(shape(h), shape(x)) — the unit diagonal must
    materialise at the add's output shape (all ones).  Vetoes the
    firing when either bound term's shape is non-concrete."""
    from catopt_core.typing import _broadcast

    s = _broadcast(_shape_of(bound.get("h")), _shape_of(bound.get("x")))
    if not (
        isinstance(s, tuple) and all(isinstance(d, int) for d in s)
    ):
        return None  # pragma: no cover — defensive guard
    # The RHS embeds ``Const(1.0)`` as a leaf; ``_instantiate`` adds
    # leaf enodes keyed by repr WITHOUT registering the term, so
    # ``any_term``/extraction would decode the raw string "1.0" unless
    # the leaf is registered here, ahead of instantiation.
    from catopt_core.egraph import _LeafRegistry

    _LeafRegistry.register(Const(1.0))
    return {"$attr:US": tuple(s)}


#: The unit diagonal as a shared pattern fragment: ones of the add's
#: broadcast shape, spelled as a Const broadcast so the carrier leaf
#: reports the same ``(d,)`` shape as every other ``aff_diag``.
_AFFD_UNIT = Op.make("expand", Const(1.0), shape="US")

# Two operand positions — add is commutative and ``canonicalize``
# sorts children by op_repr, so the state operand can sit in either
# slot (mirroring affd_lift / affd_lift_post; there is no ``mul`` and
# hence no ``_swap`` dimension).  ``_affd_unit_state_like`` suppresses
# the sideways bindings.
AFFD_LIFT_UNIT = R(
    "affd_lift_unit",
    Op.make("add", "h", "x"),
    Op.make("applyd", Op.make("aff_diag", _AFFD_UNIT, "x"), "h"),
    law="Unit introduction: pure accumulation h + x IS the diagonal "
    "affine map with a ≡ 1 — applyd(aff_diag(1, x), h).",
    check=_affd_unit_state_like,
    derive=_derive_affd_unit,
)

AFFD_LIFT_UNIT_POST = R(
    "affd_lift_unit_post",
    Op.make("add", "x", "h"),
    Op.make("applyd", Op.make("aff_diag", _AFFD_UNIT, "x"), "h"),
    law="add-order variant of affd_lift_unit (state operand in the "
    "second add slot)",
    check=_affd_unit_state_like,
    derive=_derive_affd_unit,
)

# The step rules need no side condition: the ``applyd`` inside the add
# already pins the state operand — an input e-class contains no
# ``applyd`` enode, so only the true direction matches.  The map
# ordering mirrors AFFD_LIFT_STEP: compose(aff_diag(1,x), f) applies f
# first, then h ↦ 1⊙(f·h) + x = f(h) + x.
AFFD_LIFT_UNIT_STEP = R(
    "affd_lift_unit_step",
    Op.make("add", Op.make("applyd", "f", "h"), "x"),
    Op.make(
        "applyd",
        Op.make(
            "affd_compose", Op.make("aff_diag", _AFFD_UNIT, "x"), "f"
        ),
        "h",
    ),
    law="compose a unit (pure-accumulation) step with the preceding "
    "map — the h ↦ h + x analogue of affd_lift_step",
    derive=_derive_affd_unit,
)

AFFD_LIFT_UNIT_STEP_POST = R(
    "affd_lift_unit_step_post",
    Op.make("add", "x", Op.make("applyd", "f", "h")),
    Op.make(
        "applyd",
        Op.make(
            "affd_compose", Op.make("aff_diag", _AFFD_UNIT, "x"), "f"
        ),
        "h",
    ),
    law="add-order variant of affd_lift_unit_step",
    derive=_derive_affd_unit,
)

AFFD_UNLIFT = R(
    "affd_unlift",
    Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
    Op.make("add", Op.make("mul", "a", "h"), "x"),
    law="diagonal-affine application unfolds",
)

AFFD_COMPOSE_UNFOLD = R(
    "affd_compose_unfold",
    Op.make("applyd", Op.make("affd_compose", "f", "g"), "h"),
    Op.make("applyd", "f", Op.make("applyd", "g", "h")),
    law="composition is sequential application",
)

AFFD_ASSOC = R(
    "affd_assoc",
    Op.make("affd_compose", Op.make("affd_compose", "f", "g"), "h"),
    Op.make("affd_compose", "f", Op.make("affd_compose", "g", "h")),
    law="diagonal-affine composition is associative",
)

AFFD_ASSOC_REV = R(
    "affd_assoc_rev",
    Op.make("affd_compose", "f", Op.make("affd_compose", "g", "h")),
    Op.make("affd_compose", Op.make("affd_compose", "f", "g"), "h"),
    law="diagonal-affine composition is associative",
)

#: Minimal law set for diagonal-scan discovery — the mul-form mirror of
#: ``SCAN_LAWS``, plus the unit lift for pure accumulations.  Covers
#: every operand position the state-mul can take under
#: comm-normalisation; ``_affd_state_like``/``_affd_unit_state_like``
#: suppress the sideways firings.
SCAN_DIAG_LAWS: list[Rewrite] = [
    AFFD_LIFT,
    AFFD_LIFT_SWAP,
    AFFD_LIFT_POST,
    AFFD_LIFT_POST_SWAP,
    AFFD_LIFT_STEP,
    AFFD_LIFT_STEP_SWAP,
    AFFD_LIFT_STEP_POST,
    AFFD_LIFT_STEP_POST_SWAP,
    AFFD_LIFT_UNIT,
    AFFD_LIFT_UNIT_POST,
    AFFD_LIFT_UNIT_STEP,
    AFFD_LIFT_UNIT_STEP_POST,
    AFFD_UNLIFT,
    AFFD_COMPOSE_UNFOLD,
    AFFD_ASSOC,
    AFFD_ASSOC_REV,
]
