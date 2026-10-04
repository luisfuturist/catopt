"""catopt_core.laws — rewrite laws and graph passes, split by domain.

* :mod:`catopt_core.laws.base` — the ``R`` rewrite constructor plus the
  shared check-hook machinery (``_SHAPE_MEMO``, ``_shape_of``, small
  term predicates).
* :mod:`catopt_core.laws.cond` — the declarative side-condition DSL:
  a ``cond`` is a serializable tuple tree interpreted against the same
  ``bound`` dict ``check`` sees (``R(..., cond=...)`` folds it into
  the rule's ``check`` hook at construction).
* :mod:`catopt_core.laws.serialize` — laws as data: the
  ``Rewrite`` ↔ JSON record codec (``law_to_data`` /
  ``law_from_data``), the ``missing_hooks`` honesty flag for
  procedural ``check``/``derive`` remainders, and ``alpha_key``, the
  structural identity the lemma store keys rows by.
* :mod:`catopt_core.laws.tensor` — the tensor-algebra laws:
  ``SIMPLIFICATION_RULES``, ``CATEGORICAL_RULES``, ``SDPA_FOLD_RULES``,
  ``ALL_RULES`` / ``all_rules()``.
* :mod:`catopt_core.laws.scan` — the scan-monoid law sets: ``SCAN_LAWS``
  (dense affine carrier) and ``SCAN_DIAG_LAWS`` (diagonal-affine).
* :mod:`catopt_core.laws.attention` — the attention-path laws:
  ``ATTENTION_RULES`` (rotary composition and scale commutation, the
  right-multiply absorb, the score-scale migration) — opt-in via the
  ``attention`` preset, never in ``DEFAULT``/``all_rules()``.
* :mod:`catopt_core.laws.layout` — the transpose/layout laws:
  ``LAYOUT_RULES`` (pointwise commutation, involution, the
  product-transpose law, and the ``linear``/NT-GEMM bridge), folded
  into ``ALL_RULES``.
* :mod:`catopt_core.laws.pairing` — the non-local passes over the whole
  e-graph (pairing, weight sharing).  These are diagram-level passes,
  not equational laws.
* :mod:`catopt_core.laws.factored` — the factored-parameter path:
  ``offer_low_rank_factors`` detects numerically low-rank weight
  values and offers the ``(x@A)@B`` / ``linear(linear(x,A),B)``
  member into the consumer's e-class under a certified bound.
* :mod:`catopt_core.laws.specials` — the exact-elision sibling:
  ``offer_weight_specials`` detects structurally-special stored
  weights (identity, diagonal, zero, dead/duplicate slices,
  block-diagonal) and offers exact members under a certified zero
  bound.
* :mod:`catopt_core.laws.headshare` — ``share_duplicate_attention_heads``:
  the compute-level fold for bitwise-equal attention heads — an
  ``sdpa`` over provably-identical heads runs on the ``k`` unique
  ones and re-expands by gather (exact, witnessed).
* :mod:`catopt_core.laws.tags` — the rule-tag constants
  (``SYMMETRY`` / ``EXPANSIVE`` / ``SUBSUMED`` / ``FUSION`` / …).
* :mod:`catopt_core.laws.ruleset` — :class:`RuleSet`, the composable
  rule-set value (+/-/& algebra, ``named``/``tagged`` subsets, the
  ``priorities`` scheduling map with ``EARLY``/``NORMAL``/``LATE``),
  and the named presets (``SIMPLIFICATION`` / ``CATEGORICAL`` /
  ``FUSION`` / ``SYMMETRY`` / ``CARRIERS`` / ``WITH_LAYOUT`` /
  ``DEFAULT`` / ``FULL``).

This package is the canonical rewrite surface — the historical
``catopt_core.rules`` module (and its ``catopt.rules`` façade alias)
is gone.  The loose group lists (``*_RULES`` / ``*_LAWS``,
``all_rules()``) remain as thin list aliases during migration; the
first-class values are the ``RuleSet`` presets.
"""

# Public surface (in ``__all__``) plus the private side-condition /
# memo helpers historically reachable through the removed
# ``catopt_core.rules`` shim (and its ``catopt.rules`` alias).  The
# private names are re-exported (hence ``noqa: F401``) so
# ``catopt_core.laws._x`` keeps resolving, but stay out of ``__all__``.
from catopt_core.laws import tags as tags
from catopt_core.laws.attention import (  # noqa: F401
    ATTENTION_RULES,
    LINEAR_MM_ABSORB,
    LINEAR_MM_ABSORB_BIAS,
    LINEAR_MM_ABSORB_BIAS_REV,
    LINEAR_MM_ABSORB_REV,
    LINEAR_OUT_SCALE,
    LINEAR_OUT_SCALE_REV,
    NATURALITY_SCALAR_LEFT,
    NATURALITY_SCALAR_LEFT_REV,
    ROPE_CAT_COMPOSE,
    ROPE_CAT_SCALE_IN,
    ROPE_CAT_SCALE_OUT,
    ROPE_RH_SCALE_IN,
    ROPE_RH_SCALE_OUT,
    _check_left_scale,
    _check_linear_mm_absorb,
    _check_linear_mm_absorb_bias,
    _check_linear_out_scale,
    _check_rope_cat_compose,
    _check_rope_scale,
    _check_uniform,
    _half_bounds,
    _is_uniform,
    _replant,
    _rope_axis,
    _rope_cat,
    _rope_rh,
    _slice,
    _to_end,
    _view_scale_rules,
)
from catopt_core.laws.base import (  # noqa: F401
    _SHAPE_MEMO,
    R,
    _is_channel_scale,
    _is_row_scale,
    _is_scalar,
    _shape_of,
)
from catopt_core.laws.factored import (
    offer_low_rank_factors,
)
from catopt_core.laws.headshare import (
    headshare_keys_hold,
    share_duplicate_attention_heads,
)
from catopt_core.laws.layout import (  # noqa: F401
    LAYOUT_RULES,
    LINEAR_FROM_MM_T,
    LINEAR_FROM_MM_T_BARE,
    LINEAR_TO_MM_T,
    _bound_axes,
    _check_commute_binary,
    _check_commute_unary,
    _check_involution,
    _check_linear_is_mm_t,
    _check_mm_t_is_linear,
    _check_mm_transposes,
    _check_transpose_matmul,
    _is_swap,
    _linear_shapes_ok,
)
from catopt_core.laws.pairing import (  # noqa: F401
    _CONV_ATTR_KEYS,
    _pair_shared_input,
    _term_has_var,
    _wshape,
    pair_shared_input_convs,
    pair_shared_input_linears,
    share_duplicate_param_slices,
    share_duplicate_params,
)
from catopt_core.laws.ruleset import (
    CARRIER_SEARCH,
    CARRIERS,
    CATEGORICAL,
    DEFAULT,
    EARLY,
    FULL,
    FUSION,
    LATE,
    NORMAL,
    PRESETS,
    SIMPLIFICATION,
    SYMMETRY,
    WITH_LAYOUT,
    RuleSet,
    preset,
)
from catopt_core.laws.scan import (  # noqa: F401
    _AFFD_UNIT,
    AFF_ASSOC,
    AFF_ASSOC_REV,
    AFF_COMPOSE_UNFOLD,
    AFF_LIFT,
    AFF_LIFT_STEP,
    AFF_UNLIFT,
    AFFD_ASSOC,
    AFFD_ASSOC_REV,
    AFFD_COMPOSE_UNFOLD,
    AFFD_LIFT,
    AFFD_LIFT_POST,
    AFFD_LIFT_POST_SWAP,
    AFFD_LIFT_STEP,
    AFFD_LIFT_STEP_POST,
    AFFD_LIFT_STEP_POST_SWAP,
    AFFD_LIFT_STEP_SWAP,
    AFFD_LIFT_SWAP,
    AFFD_LIFT_UNIT,
    AFFD_LIFT_UNIT_POST,
    AFFD_LIFT_UNIT_STEP,
    AFFD_LIFT_UNIT_STEP_POST,
    AFFD_UNLIFT,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
    _affd_state_like,
    _affd_unit_state_like,
    _derive_affd_unit,
)
from catopt_core.laws.specials import (
    offer_weight_specials,
)
from catopt_core.laws.tensor import (  # noqa: F401
    _QK_SCORES,
    _QKV_CAT,
    _REPEAT_KV,
    _REPEAT_V,
    ALL_RULES,
    ALL_RULES_WITH_LAYOUT,
    ASSOC_ADD,
    ASSOC_LINEAR,
    ASSOC_LINEAR_BIAS,
    ASSOC_LINEAR_BIAS_REV,
    ASSOC_LINEAR_REV,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    ASSOC_MUL,
    CATEGORICAL_RULES,
    COMM_ADD,
    COMM_MUL,
    DISTRIBUTE_MUL,
    DIV_SQRT_TO_RSQRT,
    DOUBLE_NEG,
    FACTOR_MUL,
    GLU_FOLD,
    GQA_ABSORB,
    ID_ADD,
    ID_MUL,
    LINEAR_CHANNEL_SCALE,
    LINEAR_CHANNEL_SCALE_REV,
    LINEAR_ROW_SCALE,
    LINEAR_ROW_SCALE_REV,
    MUL_SQUARE,
    NATURALITY_SCALAR,
    NATURALITY_SCALAR_REV,
    PARALLEL_MUL_FUSE,
    POW_TO_RSQRT,
    POW_TO_SQUARE,
    QKV_FUSE,
    QKV_FUSE_ASYM,
    RECIP_SQRT_TO_RSQRT,
    RIGHT_DISTRIBUTE,
    RIGHT_FACTOR,
    RIGHT_FACTOR_LINEAR,
    RMS_NORM_FOLD,
    RMS_NORM_FOLD_NOGAIN,
    SDPA_FOLD_RULES,
    SELECT_MUL,
    SILU_EXPAND,
    SILU_FOLD,
    SILU_MUL_FORM,
    SIMPLIFICATION_RULES,
    SOFTMAX_FOLD,
    SQUARE_EXPAND,
    SQUARE_TO_POW,
    SUB_TO_ADD,
    SWIGLU_FUSE,
    WEIGHT_DISTRIBUTE,
    WEIGHT_DISTRIBUTE_LINEAR,
    WEIGHT_FACTOR,
    WEIGHT_FACTOR_LINEAR,
    _check_glu_fold,
    _check_glu_shaped,
    _check_gqa_absorb,
    _check_linear_bias_compose,
    _check_repeat_chain,
    _check_rms_consts,
    _check_rms_fold,
    _check_rms_fold_nogain,
    _check_score_transpose,
    _check_sdpa_base,
    _check_sdpa_mf,
    _check_sdpa_mf_scaled,
    _check_sdpa_scaled,
    _check_softmax_dim,
    _const_val,
    _derive_rms_norm,
    _derive_scale_div,
    _derive_scale_mul,
    _derive_scale_one,
    _derive_split_sizes,
    _head,
    _head_v,
    _make_sdpa_fold_rules,
    _scale_of,
    all_rules,
)

__all__ = [
    "AFFD_ASSOC",
    "AFFD_ASSOC_REV",
    "AFFD_COMPOSE_UNFOLD",
    "AFFD_LIFT",
    "AFFD_LIFT_POST",
    "AFFD_LIFT_POST_SWAP",
    "AFFD_LIFT_STEP",
    "AFFD_LIFT_STEP_POST",
    "AFFD_LIFT_STEP_POST_SWAP",
    "AFFD_LIFT_STEP_SWAP",
    "AFFD_LIFT_SWAP",
    "AFFD_LIFT_UNIT",
    "AFFD_LIFT_UNIT_POST",
    "AFFD_LIFT_UNIT_STEP",
    "AFFD_LIFT_UNIT_STEP_POST",
    "AFFD_UNLIFT",
    "AFF_ASSOC",
    "AFF_ASSOC_REV",
    "AFF_COMPOSE_UNFOLD",
    "AFF_LIFT",
    "AFF_LIFT_STEP",
    "AFF_UNLIFT",
    "ALL_RULES",
    "ALL_RULES_WITH_LAYOUT",
    "ASSOC_ADD",
    "ASSOC_LINEAR",
    "ASSOC_LINEAR_BIAS",
    "ASSOC_LINEAR_BIAS_REV",
    "ASSOC_LINEAR_REV",
    "ASSOC_MATMUL",
    "ASSOC_MATMUL_REV",
    "ASSOC_MUL",
    "ATTENTION_RULES",
    "CARRIERS",
    "CARRIER_SEARCH",
    "CATEGORICAL",
    "CATEGORICAL_RULES",
    "COMM_ADD",
    "COMM_MUL",
    "DEFAULT",
    "DISTRIBUTE_MUL",
    "DIV_SQRT_TO_RSQRT",
    "DOUBLE_NEG",
    "EARLY",
    "FACTOR_MUL",
    "FULL",
    "FUSION",
    "GLU_FOLD",
    "GQA_ABSORB",
    "ID_ADD",
    "ID_MUL",
    "LATE",
    "LAYOUT_RULES",
    "LINEAR_CHANNEL_SCALE",
    "LINEAR_CHANNEL_SCALE_REV",
    "LINEAR_FROM_MM_T",
    "LINEAR_FROM_MM_T_BARE",
    "LINEAR_MM_ABSORB",
    "LINEAR_MM_ABSORB_BIAS",
    "LINEAR_MM_ABSORB_BIAS_REV",
    "LINEAR_MM_ABSORB_REV",
    "LINEAR_OUT_SCALE",
    "LINEAR_OUT_SCALE_REV",
    "LINEAR_ROW_SCALE",
    "LINEAR_ROW_SCALE_REV",
    "LINEAR_TO_MM_T",
    "MUL_SQUARE",
    "NATURALITY_SCALAR",
    "NATURALITY_SCALAR_LEFT",
    "NATURALITY_SCALAR_LEFT_REV",
    "NATURALITY_SCALAR_REV",
    "NORMAL",
    "PARALLEL_MUL_FUSE",
    "POW_TO_RSQRT",
    "POW_TO_SQUARE",
    "PRESETS",
    "QKV_FUSE",
    "QKV_FUSE_ASYM",
    "RECIP_SQRT_TO_RSQRT",
    "RIGHT_DISTRIBUTE",
    "RIGHT_FACTOR",
    "RIGHT_FACTOR_LINEAR",
    "RMS_NORM_FOLD",
    "RMS_NORM_FOLD_NOGAIN",
    "ROPE_CAT_COMPOSE",
    "ROPE_CAT_SCALE_IN",
    "ROPE_CAT_SCALE_OUT",
    "ROPE_RH_SCALE_IN",
    "ROPE_RH_SCALE_OUT",
    "SCAN_DIAG_LAWS",
    "SCAN_LAWS",
    "SDPA_FOLD_RULES",
    "SELECT_MUL",
    "SILU_EXPAND",
    "SILU_FOLD",
    "SILU_MUL_FORM",
    "SIMPLIFICATION",
    "SIMPLIFICATION_RULES",
    "SOFTMAX_FOLD",
    "SQUARE_EXPAND",
    "SQUARE_TO_POW",
    "SUB_TO_ADD",
    "SWIGLU_FUSE",
    "SYMMETRY",
    "WEIGHT_DISTRIBUTE",
    "WEIGHT_DISTRIBUTE_LINEAR",
    "WEIGHT_FACTOR",
    "WEIGHT_FACTOR_LINEAR",
    "WITH_LAYOUT",
    "R",
    "RuleSet",
    "all_rules",
    "headshare_keys_hold",
    "offer_low_rank_factors",
    "offer_weight_specials",
    "pair_shared_input_convs",
    "pair_shared_input_linears",
    "preset",
    "share_duplicate_attention_heads",
    "share_duplicate_param_slices",
    "share_duplicate_params",
    "tags",
]
