"""Compatibility shim — the rewrite surface moved to :mod:`catopt.laws`.

Historically this module mixed three different things: the equational
law definitions (now :mod:`catopt.laws.tensor` and
:mod:`catopt.laws.scan`), the side-condition check helpers (same
modules as their laws), and the non-local whole-graph passes
(:mod:`catopt.laws.pairing`).  Everything previously importable from
``catopt.rules`` — public rule objects, rule collections,
``all_rules()``, the pass functions, and the private check/memo
helpers — still resolves here unchanged.
"""

# ruff: noqa: F401 F403 -- this module is a pure re-export shim.

from typing import Any

from catopt.egraph import Rewrite
from catopt.ir import Const, Op
from catopt.laws import *
from catopt.laws.base import (
    _SHAPE_MEMO,
    _is_channel_scale,
    _is_row_scale,
    _is_scalar,
    _shape_of,
)
from catopt.laws.pairing import (
    _CONV_ATTR_KEYS,
    _pair_shared_input,
    _term_has_var,
    _wshape,
)
from catopt.laws.scan import (
    _AFFD_UNIT,
    _affd_state_like,
    _affd_unit_state_like,
    _derive_affd_unit,
)
from catopt.laws.tensor import (
    _QK_SCORES,
    _QKV_CAT,
    _REPEAT_KV,
    _REPEAT_V,
    _check_gqa_absorb,
    _check_linear_bias_compose,
    _check_repeat_chain,
    _check_score_transpose,
    _check_sdpa_base,
    _check_sdpa_mf,
    _check_sdpa_mf_scaled,
    _check_sdpa_scaled,
    _check_softmax_dim,
    _const_val,
    _derive_scale_div,
    _derive_scale_mul,
    _derive_scale_one,
    _derive_split_sizes,
    _head,
    _head_v,
    _make_sdpa_fold_rules,
    _scale_of,
)
