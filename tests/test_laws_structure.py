"""Structure tests for the Phase-2d split of ``catopt.rules``.

``catopt/rules.py`` was split into the ``catopt.laws`` package by
domain; ``catopt.rules`` is now a pure compatibility shim.  These tests
pin the contract:

* every name that was previously importable from ``catopt.rules``
  still resolves through the shim (public rules, collections, pass
  functions, and the private check/memo helpers);
* ``all_rules()`` returns the same rule set as before;
* the non-local pairing passes resolve identically from both the
  legacy path and ``catopt.laws.pairing``;
* each ``catopt.laws`` submodule is self-contained — it never imports
  the compat shim.
"""

import importlib
import re
import sys

import catopt.laws.pairing
import catopt.laws.scan
import catopt.laws.tensor
import catopt.rules as rules
from catopt.egraph import Rewrite

#: Every public name importable from the old ``catopt.rules`` — the
#: union of all ``from catopt.rules import ...`` sites across catopt/,
#: tests/, and bench/, plus the rule objects accessed as module
#: attributes (``from catopt import rules as R; R.X``).
PUBLIC_NAMES = [
    # rewrite-constructor shorthand (catopt.om/trace/xcarrier use it)
    "R",
    # monoid / group / decomposition laws
    "COMM_ADD",
    "COMM_MUL",
    "ASSOC_ADD",
    "ASSOC_MUL",
    "ID_ADD",
    "ID_MUL",
    "DOUBLE_NEG",
    "SUB_TO_ADD",
    "SILU_EXPAND",
    "SILU_MUL_FORM",
    "SQUARE_EXPAND",
    "POW_TO_SQUARE",
    "SQUARE_TO_POW",
    # distributivity / naturality
    "DISTRIBUTE_MUL",
    "FACTOR_MUL",
    "RIGHT_DISTRIBUTE",
    "RIGHT_FACTOR",
    "WEIGHT_FACTOR",
    "WEIGHT_DISTRIBUTE",
    "WEIGHT_FACTOR_LINEAR",
    "WEIGHT_DISTRIBUTE_LINEAR",
    "RIGHT_FACTOR_LINEAR",
    "ASSOC_LINEAR",
    "ASSOC_LINEAR_BIAS",
    "ASSOC_LINEAR_BIAS_REV",
    "NATURALITY_SCALAR",
    "NATURALITY_SCALAR_REV",
    "ASSOC_MATMUL",
    "ASSOC_MATMUL_REV",
    # product-structure laws
    "SWIGLU_FUSE",
    "PARALLEL_MUL_FUSE",
    "QKV_FUSE",
    "QKV_FUSE_ASYM",
    # diagonal-scale / diagonal-map naturality
    "LINEAR_CHANNEL_SCALE",
    "LINEAR_CHANNEL_SCALE_REV",
    "LINEAR_ROW_SCALE",
    "LINEAR_ROW_SCALE_REV",
    "GQA_ABSORB",
    # softmax-attention fold
    "SDPA_FOLD_RULES",
    # scan monoids
    "AFF_LIFT",
    "AFF_LIFT_STEP",
    "AFF_UNLIFT",
    "AFF_COMPOSE_UNFOLD",
    "AFF_ASSOC",
    "AFF_ASSOC_REV",
    "AFFD_LIFT",
    "AFFD_LIFT_SWAP",
    "AFFD_LIFT_POST",
    "AFFD_LIFT_POST_SWAP",
    "AFFD_LIFT_STEP",
    "AFFD_LIFT_STEP_SWAP",
    "AFFD_LIFT_STEP_POST",
    "AFFD_LIFT_STEP_POST_SWAP",
    "AFFD_LIFT_UNIT",
    "AFFD_LIFT_UNIT_POST",
    "AFFD_LIFT_UNIT_STEP",
    "AFFD_LIFT_UNIT_STEP_POST",
    "AFFD_UNLIFT",
    "AFFD_COMPOSE_UNFOLD",
    "AFFD_ASSOC",
    "AFFD_ASSOC_REV",
    # collections
    "SIMPLIFICATION_RULES",
    "CATEGORICAL_RULES",
    "SCAN_LAWS",
    "SCAN_DIAG_LAWS",
    "ALL_RULES",
    "all_rules",
    # non-local passes
    "pair_shared_input_linears",
    "pair_shared_input_convs",
    "share_duplicate_params",
    "share_duplicate_param_slices",
]

#: Private helpers the old module exposed — tests import
#: ``_check_gqa_absorb``/``_check_softmax_dim`` directly, and
#: ``_SHAPE_MEMO``/``_shape_of`` may be reached as ``rules._x``.
PRIVATE_NAMES = [
    "_SHAPE_MEMO",
    "_shape_of",
    "_is_scalar",
    "_is_row_scale",
    "_is_channel_scale",
    "_check_linear_bias_compose",
    "_head",
    "_head_v",
    "_derive_split_sizes",
    "_QKV_CAT",
    "_check_repeat_chain",
    "_check_gqa_absorb",
    "_REPEAT_KV",
    "_REPEAT_V",
    "_QK_SCORES",
    "_const_val",
    "_check_score_transpose",
    "_check_softmax_dim",
    "_scale_of",
    "_check_sdpa_base",
    "_check_sdpa_scaled",
    "_check_sdpa_mf",
    "_check_sdpa_mf_scaled",
    "_derive_scale_mul",
    "_derive_scale_div",
    "_derive_scale_one",
    "_make_sdpa_fold_rules",
    "_term_has_var",
    "_pair_shared_input",
    "_wshape",
    "_CONV_ATTR_KEYS",
    "_affd_state_like",
    "_affd_unit_state_like",
    "_derive_affd_unit",
    "_AFFD_UNIT",
]

PAIRING_PASSES = [
    "pair_shared_input_linears",
    "pair_shared_input_convs",
    "share_duplicate_params",
    "share_duplicate_param_slices",
]


def test_public_names_resolve_via_shim():
    missing = [n for n in PUBLIC_NAMES if not hasattr(rules, n)]
    assert not missing, f"names no longer importable: {missing}"


def test_private_names_resolve_via_shim():
    missing = [n for n in PRIVATE_NAMES if not hasattr(rules, n)]
    assert not missing, f"helpers no longer importable: {missing}"


def test_shim_exports_every_rewrite():
    """meta._iter_module_rules over the shim must see the full set."""
    from catopt.meta import _iter_module_rules

    found = _iter_module_rules(rules)
    assert all(isinstance(r, Rewrite) for r in found)
    # 50 tensor + 6 dense-scan + 16 diagonal-scan rewrites.
    assert len(found) == 72


def test_all_rules_count_unchanged():
    assert len(rules.all_rules()) == 50
    assert rules.all_rules() == rules.ALL_RULES
    assert len(rules.ALL_RULES) == (
        len(rules.SIMPLIFICATION_RULES) + len(rules.CATEGORICAL_RULES)
    )
    # each call returns a fresh list, not the shared ALL_RULES object
    assert rules.all_rules() is not rules.ALL_RULES


def test_collections_split_by_domain():
    assert len(rules.SIMPLIFICATION_RULES) == 13
    assert len(rules.CATEGORICAL_RULES) == 37
    assert len(rules.SCAN_LAWS) == 6
    assert len(rules.SCAN_DIAG_LAWS) == 16


def test_pairing_passes_resolve_from_both_paths():
    for name in PAIRING_PASSES:
        legacy = getattr(rules, name)
        direct = getattr(catopt.laws.pairing, name)
        assert callable(legacy) and callable(direct)
        assert legacy is direct


def test_laws_modules_are_self_contained():
    """No laws submodule may depend on the catopt.rules compat shim."""
    shim_import = re.compile(
        r"(?:from|import)\s+catopt\.rules\b|catopt\.rules\."
    )
    for mod in (
        catopt.laws.base,
        catopt.laws.tensor,
        catopt.laws.scan,
        catopt.laws.pairing,
    ):
        with open(mod.__file__) as f:
            src = f.read()
        assert not shim_import.search(src), mod.__name__


def test_laws_modules_import_cleanly():
    """Each laws module resolves its own surface standalone."""
    import catopt.laws as laws

    expected = {
        "catopt.laws.base": ["R", "_SHAPE_MEMO", "_shape_of"],
        "catopt.laws.tensor": [
            "SIMPLIFICATION_RULES",
            "CATEGORICAL_RULES",
            "ALL_RULES",
            "all_rules",
        ],
        "catopt.laws.scan": ["SCAN_LAWS", "SCAN_DIAG_LAWS"],
        "catopt.laws.pairing": PAIRING_PASSES,
    }
    for modname, names in expected.items():
        mod = sys.modules.get(modname) or importlib.import_module(modname)
        for n in names:
            assert hasattr(mod, n), f"{modname}.{n}"
    # package-level re-export matches the shim's surface
    assert laws.all_rules() == rules.all_rules()
    assert laws.SCAN_LAWS is rules.SCAN_LAWS
    assert laws.SCAN_DIAG_LAWS is rules.SCAN_DIAG_LAWS


def test_shim_preserves_module_level_objects():
    """Name bindings the old module carried (imports) still resolve."""
    assert rules.Rewrite is Rewrite
    assert isinstance(rules._SHAPE_MEMO, dict)
    assert callable(rules._shape_of)
