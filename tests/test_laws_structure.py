"""Structure tests for the ``catopt_core.laws`` rewrite surface.

``catopt_core.rules`` — and its ``catopt.rules`` façade alias — used
to be a pure compatibility shim over the ``catopt_core.laws``
package.  That
shim is gone; ``catopt_core.laws`` is now the canonical rewrite
surface.  These tests pin the contract:

* every name that was previously importable from the shim — public
  rules, collections, pass functions, and the private check/memo
  helpers — still resolves from ``catopt_core.laws``;
* ``all_rules()`` returns the same rule set as before;
* the non-local pairing passes resolve identically from the package
  and from ``catopt_core.laws.pairing``;
* each ``catopt_core.laws`` submodule is self-contained — it never
  imports the removed shim.
"""

import importlib
import re
import sys

import catopt_core.laws as laws
import catopt_core.laws.layout
import catopt_core.laws.pairing
import catopt_core.laws.scan
import catopt_core.laws.tensor
from catopt_core.egraph import Rewrite

#: Every public name the old ``catopt_core.rules`` shim exposed — the
#: union of all ``from catopt_core.rules import ...`` sites across
#: packages/, the old façade, tests/, and bench/, plus the rule objects
#: accessed as module attributes.
PUBLIC_NAMES = [
    # rewrite-constructor shorthand (om/trace/xcarrier use it)
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
    "SILU_FOLD",
    "SILU_MUL_FORM",
    "SQUARE_EXPAND",
    "POW_TO_SQUARE",
    "SQUARE_TO_POW",
    "SELECT_MUL",
    "SOFTMAX_FOLD",
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
    "LAYOUT_RULES",
    "ALL_RULES",
    "all_rules",
    # non-local passes
    "pair_shared_input_linears",
    "pair_shared_input_convs",
    "share_duplicate_params",
    "share_duplicate_param_slices",
]

#: Private helpers the old shim re-exported — tests import
#: ``_check_gqa_absorb``/``_check_softmax_dim`` directly, and
#: ``_SHAPE_MEMO``/``_shape_of`` may be reached as ``laws._x``.
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


def test_public_names_resolve():
    missing = [n for n in PUBLIC_NAMES if not hasattr(laws, n)]
    assert not missing, f"names no longer importable: {missing}"


def test_private_names_resolve():
    missing = [n for n in PRIVATE_NAMES if not hasattr(laws, n)]
    assert not missing, f"helpers no longer importable: {missing}"


def test_laws_exports_every_rewrite():
    """meta._iter_module_rules over the package must see the full set."""
    from catopt_core.meta import _iter_module_rules

    found = _iter_module_rules(laws)
    assert all(isinstance(r, Rewrite) for r in found)
    # 54 tensor (assoc_linear_rev + select_mul + softmax_fold
    # + silu_fold)
    # + 6 dense-scan + 16 diagonal-scan + 77 layout + 37 attention
    # rewrites.
    assert len(found) == 190


def test_all_rules_count_unchanged():
    # The default set excludes LAYOUT_RULES (opt-in — closure-cost
    # regression documented in laws.tensor); WITH_LAYOUT keeps 131.
    assert len(laws.all_rules()) == 54
    assert laws.all_rules() == laws.ALL_RULES
    assert len(laws.ALL_RULES) == (
        len(laws.SIMPLIFICATION_RULES) + len(laws.CATEGORICAL_RULES)
    )
    assert len(laws.ALL_RULES_WITH_LAYOUT) == 131
    # each call returns a fresh list, not the shared ALL_RULES object
    assert laws.all_rules() is not laws.ALL_RULES


def test_collections_split_by_domain():
    assert len(laws.SIMPLIFICATION_RULES) == 16
    assert len(laws.CATEGORICAL_RULES) == 38
    assert len(laws.SCAN_LAWS) == 6
    assert len(laws.SCAN_DIAG_LAWS) == 16
    # layout: 12 unary + 4 binary pointwise commutations x 2 spellings
    # x 2 directions, 4 involution spellings, 6 product-transpose,
    # 3 linear/NT bridge
    assert len(laws.LAYOUT_RULES) == 77


def test_pairing_passes_resolve_from_both_paths():
    for name in PAIRING_PASSES:
        package = getattr(laws, name)
        direct = getattr(catopt_core.laws.pairing, name)
        assert callable(package) and callable(direct)
        assert package is direct


def test_laws_modules_are_self_contained():
    """No laws submodule may depend on the removed ``*.rules`` shim."""
    shim_import = re.compile(
        r"(?:from|import)\s+catopt(?:_core)?\.rules\b"
        r"|catopt(?:_core)?\.rules\."
    )
    for mod in (
        catopt_core.laws.base,
        catopt_core.laws.tensor,
        catopt_core.laws.scan,
        catopt_core.laws.layout,
        catopt_core.laws.pairing,
    ):
        with open(mod.__file__) as f:
            src = f.read()
        assert not shim_import.search(src), mod.__name__


def test_laws_modules_import_cleanly():
    """Each laws module resolves its own surface standalone."""
    expected = {
        "catopt_core.laws.base": ["R", "_SHAPE_MEMO", "_shape_of"],
        "catopt_core.laws.tensor": [
            "SIMPLIFICATION_RULES",
            "CATEGORICAL_RULES",
            "ALL_RULES",
            "all_rules",
        ],
        "catopt_core.laws.scan": ["SCAN_LAWS", "SCAN_DIAG_LAWS"],
        "catopt_core.laws.layout": ["LAYOUT_RULES"],
        "catopt_core.laws.pairing": PAIRING_PASSES,
        "catopt_core.laws.factored": ["offer_low_rank_factors"],
        "catopt_core.laws.specials": ["offer_weight_specials"],
    }
    for modname, names in expected.items():
        mod = sys.modules.get(modname) or importlib.import_module(
            modname
        )
        for n in names:
            assert hasattr(mod, n), f"{modname}.{n}"
    # package-level re-export matches the submodules' surface
    assert laws.all_rules() == catopt_core.laws.tensor.all_rules()
    assert laws.SCAN_LAWS is catopt_core.laws.scan.SCAN_LAWS
    assert laws.SCAN_DIAG_LAWS is catopt_core.laws.scan.SCAN_DIAG_LAWS


def test_laws_preserves_module_level_objects():
    """Private helpers the old shim carried still resolve from the package."""
    assert isinstance(laws._SHAPE_MEMO, dict)
    assert callable(laws._shape_of)


def test_axiom_lemma_marker_covers_all_rules():
    """The axiom/lemma split — every shipped rule carries a kernel kind.

    Measured by ``tools/law_coherence.py --emit-basis`` (see
    ``project/retros/axiom-lemma-split.md``): the kernel is the 29
    primitives plus one designated member (alphabetically first) per
    derivability cycle = 40 axioms; the other 14 rules carry a
    ``derivation`` naming their premise rules — 12 lemmas (inverse
    twins + the emergent silu_mul_form) and 2 redundant
    alpha-duplicates.
    """
    by_name = {r.name: r for r in laws.ALL_RULES}
    kinds = {"axiom": [], "lemma": [], "redundant": []}
    for r in laws.ALL_RULES:
        assert r.kind in kinds, (r.name, r.kind)
        kinds[r.kind].append(r.name)
        if r.kind == "axiom":
            assert not r.derivation, r.name
        else:
            assert r.derivation, r.name
            for premise in r.derivation:
                # every recorded premise resolves to a shipped axiom —
                # a lemma's proof sketch stays at kernel level
                assert premise in by_name, (r.name, premise)
                assert by_name[premise].kind == "axiom", (
                    r.name,
                    premise,
                )
    assert len(kinds["axiom"]) == 40
    assert len(kinds["lemma"]) == 12
    assert len(kinds["redundant"]) == 2
    # the redundant kind rides on the tags.REDUNDANT constant
    assert {
        r.name
        for r in laws.ALL_RULES
        if laws.tags.REDUNDANT in r.tags
    } == {"weight_distribute_matmul", "weight_factor_matmul"}
