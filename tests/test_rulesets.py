"""Composable rule sets — plan 0009.

Covers the ``RuleSet`` value (name-keyed +/−/& algebra, ``named`` /
``tagged`` subsets, the ``priorities`` scheduling map), the tag
assignments on the shipped rules, the named presets and their
compositions, the orchestrator's ``rules`` resolution, and the
parity contract: ``DEFAULT`` reproduces the retired
``ruleset="all"`` saturation behaviour on a fixture.
"""

import pytest
import torch
import torch.nn as nn

from catopt_core import laws
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op
from catopt_core.laws import (
    ALL_RULES,
    ALL_RULES_WITH_LAYOUT,
    CATEGORICAL_RULES,
    COMM_ADD,
    DISTRIBUTE_MUL,
    GQA_ABSORB,
    ID_ADD,
    LAYOUT_RULES,
    QKV_FUSE,
    R,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
    SDPA_FOLD_RULES,
    SIMPLIFICATION_RULES,
    SWIGLU_FUSE,
    RuleSet,
    all_rules,
    preset,
    tags,
)
from catopt_core.ports import RuleLike, RuleSetLike

SUBSUMED_NAMES = {
    "swiglu_fuse",
    "parallel_mul_fuse",
    "qkv_fuse",
    "qkv_fuse_asym",
}
SYMMETRY_NAMES = {
    "comm_add",
    "comm_mul",
    "assoc_add",
    "assoc_mul",
    "linear_row_scale",
    "linear_row_scale_rev",
    "linear_channel_scale",
    "linear_channel_scale_rev",
}


def _a() -> RuleSet:
    return RuleSet("a", (COMM_ADD, ID_ADD))


def _b() -> RuleSet:
    return RuleSet("b", (DISTRIBUTE_MUL,))


# ---------------------------------------------------------------------------
#  The value itself
# ---------------------------------------------------------------------------


def test_rewrite_gains_tags_and_r_kwarg():
    plain = R("t", Op.make("add", "a", "b"), "a")
    assert plain.tags == frozenset()
    tagged = R(
        "t2",
        Op.make("add", "a", "b"),
        "a",
        tags=[tags.SIMPLIFICATION],
    )
    assert tagged.tags == frozenset({tags.SIMPLIFICATION})
    # Construction order (positional Rewrite) stays intact.
    direct = Rewrite("x", "a", "b")
    assert direct.tags == frozenset()


def test_ruleset_construction_dedup_and_containment():
    rs = RuleSet("t", [COMM_ADD, ID_ADD])
    assert isinstance(rs.rules, tuple)  # normalised
    assert len(rs) == 2
    assert list(rs) == [COMM_ADD, ID_ADD]
    assert "comm_add" in rs
    assert COMM_ADD in rs
    assert "nope" not in rs
    assert DISTRIBUTE_MUL not in rs
    assert 42 not in rs
    with pytest.raises(ValueError, match="duplicate rule name"):
        RuleSet("bad", (COMM_ADD, COMM_ADD))


def test_ruleset_value_semantics():
    a, b = _a(), _b()
    assert a == _a()
    assert a != b
    assert (a == "not a ruleset") is False
    assert hash(a) == hash(_a())
    # name/description are labels, not identity.
    assert a == RuleSet("other", (COMM_ADD, ID_ADD), description="d")


def test_ruleset_union_algebra():
    a, b = _a(), _b()
    assert a + a == a  # idempotent
    assert a + b == b + a  # order-insensitive
    assert len(a + b) == 3
    assert (a + b) - b == a
    assert (a + b) & b == b
    assert (a + b) - b - a == RuleSet("empty", ())
    # union accepts plain iterables too.
    assert a.union([DISTRIBUTE_MUL]) == a + b
    assert (a - [COMM_ADD]) == RuleSet("x", (ID_ADD,))
    assert (a & [COMM_ADD]) == RuleSet("x", (COMM_ADD,))


def test_ruleset_union_name_conflict():
    other = RuleSet(
        "o",
        (R("comm_add", Op.make("add", "x", "y"), "x"),),
    )
    a = _a()
    with pytest.raises(ValueError, match="override=True"):
        a + other
    replaced = a.union(other, override=True)
    assert replaced.rules[0] is other.rules[0]
    assert len(replaced) == 2


def test_ruleset_named_and_tagged_subsets():
    rs = _a() + _b()
    sub = rs.named("comm_add")
    assert isinstance(sub, RuleSet)
    assert list(sub) == [COMM_ADD]
    assert rs.tagged(tags.EXPANSIVE) == RuleSet(
        "exp", (COMM_ADD, DISTRIBUTE_MUL)
    )
    assert rs.tagged("nonexistent") == RuleSet("empty", ())
    # tags / names list through as labels for inspection.
    assert sub.name == "a+b[comm_add]"
    assert rs.tagged(tags.EXPANSIVE).name == "a+b.tagged"


def test_ruleset_priorities():
    rs = _a().with_priorities(comm_add=laws.EARLY)
    assert rs.priority_of(COMM_ADD) == laws.EARLY == 0
    assert rs.priority_of("comm_add") == 0
    assert rs.priority_of(ID_ADD) == laws.NORMAL == 10
    assert rs.priority_of("missing") == laws.NORMAL
    assert laws.LATE == 20
    merged = rs.with_priorities(id_add=laws.LATE)
    assert merged.priority_of("id_add") == 20
    assert merged.priority_of("comm_add") == 0  # merge, not replace
    with pytest.raises(KeyError, match="with_priorities"):
        rs.with_priorities(nope=1)


def test_ruleset_priorities_ride_composition():
    a = _a().with_priorities(comm_add=1)
    b = _b().with_priorities(distribute_matmul_over_add=2)
    u = a + b  # the operand's own priorities merge in
    assert u.priority_of("comm_add") == 1
    assert u.priority_of("distribute_matmul_over_add") == 2
    # subsets keep only the members' priorities.
    assert (u - b).priorities == {"comm_add": 1}
    assert (u & b).priorities == {"distribute_matmul_over_add": 2}
    assert u.named("comm_add").priorities == {"comm_add": 1}
    assert u.tagged(tags.SYMMETRY).priorities == {"comm_add": 1}


def test_ruleset_is_a_rulesetlike():
    assert isinstance(laws.DEFAULT, RuleSetLike)
    assert all(isinstance(r, RuleLike) for r in laws.DEFAULT)


# ---------------------------------------------------------------------------
#  Tag assignments on the shipped rules
# ---------------------------------------------------------------------------


def test_tag_assignments():
    assert COMM_ADD.tags == {tags.SYMMETRY, tags.EXPANSIVE}
    assert SWIGLU_FUSE.tags == {tags.FUSION, tags.SUBSUMED}
    assert ID_ADD.tags == {tags.SIMPLIFICATION}
    assert DISTRIBUTE_MUL.tags == {tags.CATEGORICAL, tags.EXPANSIVE}
    assert all(
        tags.FUSION in r.tags for r in SDPA_FOLD_RULES
    )
    assert GQA_ABSORB.tags == {tags.FUSION}
    assert all(tags.LAYOUT in r.tags for r in LAYOUT_RULES)
    assert all(r.tags == {tags.SCAN} for r in SCAN_LAWS)
    assert all(r.tags == {tags.SCAN} for r in SCAN_DIAG_LAWS)


def test_carrier_package_tags():
    from catopt_carriers.decode_laws import DECODE_LAWS
    from catopt_carriers.om import OM_LAWS
    from catopt_carriers.trace import TRACE_LAWS
    from catopt_carriers.xcarrier import XC_LAWS

    assert all(
        r.tags == {tags.CARRIER, tags.DECODE} for r in DECODE_LAWS
    )
    assert all(r.tags == {tags.CARRIER} for r in OM_LAWS)
    assert all(r.tags == {tags.CARRIER} for r in TRACE_LAWS)
    assert all(r.tags == {tags.CARRIER} for r in XC_LAWS)


def test_every_rule_is_tagged():
    """Tag hygiene: no shipped rule is left untagged."""
    assert all(r.tags for r in all_rules())
    assert all(r.tags for r in LAYOUT_RULES)
    assert all(r.tags for r in SCAN_LAWS)


# ---------------------------------------------------------------------------
#  Presets
# ---------------------------------------------------------------------------


def test_preset_contents():
    assert len(laws.SIMPLIFICATION) == len(SIMPLIFICATION_RULES)
    # the historical "categorical" selection — minus the subsumed folds
    assert len(laws.CATEGORICAL) == len(CATEGORICAL_RULES) - 4
    assert (
        len(laws.FUSION)
        == sum(1 for r in all_rules() if tags.FUSION in r.tags)
        == 17
    )
    assert {r.name for r in laws.SYMMETRY} == SYMMETRY_NAMES
    # CARRIERS names the carrier-package families — empty core-side;
    # the scan monoids are the core share of CARRIER_SEARCH.
    assert not laws.CARRIERS
    assert len(laws.CARRIER_SEARCH) == (
        len(SCAN_LAWS) + len(SCAN_DIAG_LAWS)
    )
    assert len(laws.WITH_LAYOUT) == len(ALL_RULES_WITH_LAYOUT)
    assert laws.PRESETS["default"] is laws.DEFAULT
    assert {p.name for p in laws.PRESETS.values()} == set(laws.PRESETS)


def test_default_excludes_symmetry_and_subsumed():
    names = {r.name for r in laws.DEFAULT}
    assert names.isdisjoint(SYMMETRY_NAMES)
    assert names.isdisjoint(SUBSUMED_NAMES)
    assert not laws.DEFAULT.tagged(tags.SUBSUMED)
    assert not laws.DEFAULT.tagged(tags.SYMMETRY)
    # what stays: categorical + simplification(non-sym) + fusion(non-sub)
    assert "distribute_matmul_over_add" in names
    assert "id_add" in names
    assert "gqa_absorb_repeat" in names
    # the scan monoids stay out of the default search — they are the
    # regime's CARRIER_SEARCH set, not pipeline algebra.
    assert names.isdisjoint({r.name for r in SCAN_LAWS})


def test_full_is_default_plus_symmetry():
    assert laws.DEFAULT + laws.SYMMETRY == laws.FULL
    # FULL is exactly the retired ``ruleset="all"`` selection —
    # the whole core equational surface minus the pairing-subsumed
    # folds (the pairing pass owns those).
    assert {r.name for r in laws.FULL} == {
        r.name for r in ALL_RULES if r.name not in SUBSUMED_NAMES
    }
    assert not laws.FULL.tagged(tags.SUBSUMED)


def test_simplification_plus_categorical_is_legacy_all():
    """SIMPLIFICATION + CATEGORICAL is exactly the old ruleset="all"."""
    legacy = {
        r.name for r in all_rules() if r.name not in SUBSUMED_NAMES
    }
    assert laws.SIMPLIFICATION + laws.CATEGORICAL == RuleSet(
        "legacy", tuple(r for r in all_rules() if r.name in legacy)
    )


def test_preset_lookup():
    assert preset("default") is laws.DEFAULT
    assert preset("full") is laws.FULL
    assert preset("simplification") is laws.SIMPLIFICATION
    with pytest.raises(ValueError, match="Unknown ruleset"):
        preset("bogus")


# ---------------------------------------------------------------------------
#  Orchestrator resolution + the composed default
# ---------------------------------------------------------------------------


def test_resolve_rules_variants():
    from catopt_orchestrator.optimize import (
        _resolve_rules,
        default_rules,
    )

    assert _resolve_rules(None) is default_rules()
    assert _resolve_rules(laws.CATEGORICAL) is laws.CATEGORICAL
    assert _resolve_rules("categorical") is laws.CATEGORICAL
    assert _resolve_rules("default") is default_rules()
    assert _resolve_rules("default_rules") is default_rules()
    # orchestrator-level names resolve to the regime rule sets.
    cs = _resolve_rules("carrier_search")
    assert {r.name for r in laws.CARRIER_SEARCH} <= {
        r.name for r in cs
    }
    assert _resolve_rules("xc") == _resolve_rules("xc")
    custom = _resolve_rules([COMM_ADD, ID_ADD])
    assert isinstance(custom, RuleSet) and len(custom) == 2
    with pytest.raises(ValueError, match="Unknown ruleset"):
        _resolve_rules("bogus")


def test_default_rules_composes_carriers():
    from catopt_carriers.decode_laws import DECODE_LAWS
    from catopt_carriers.om import OM_LAWS
    from catopt_carriers.trace import TRACE_LAWS
    from catopt_carriers.xcarrier import XC_LAWS
    from catopt_orchestrator import DEFAULT_RULES, default_rules

    assert DEFAULT_RULES is default_rules()
    names = {r.name for r in DEFAULT_RULES}
    core = {r.name for r in laws.DEFAULT}
    assert core <= names
    # the carrier-package families (om / trace / xc / decode)
    # compose in on top of the core default.
    for fam in (OM_LAWS, TRACE_LAWS, XC_LAWS, DECODE_LAWS):
        assert {r.name for r in fam} <= names
    # lazy attrs on the package resolve.
    import catopt_carriers

    assert catopt_carriers.CARRIERS.name.startswith("om+")
    assert catopt_carriers.OM_RULES.name == "om"
    assert catopt_carriers.TRACE_RULES.name == "trace"
    assert catopt_carriers.XC_RULES.name == "xc"
    assert catopt_carriers.DECODE_RULES.name == "decode"
    with pytest.raises(AttributeError):
        _ = catopt_carriers.bogus_name


def test_regime_rule_sets():
    import catopt_orchestrator.regime as regime

    search = regime.default_rules()
    assert isinstance(search, RuleSet)
    # CARRIER_SEARCH = the scan monoids + decode / om / trace.
    assert {r.name for r in laws.CARRIER_SEARCH} <= {
        r.name for r in search
    }
    assert regime.CARRIER_LAWS == search
    assert isinstance(regime.XC_LAWS, RuleSet)


def test_optimize_getattr_unknown():
    import catopt_orchestrator.optimize as opt

    assert opt.DEFAULT_RULES is opt.default_rules()
    with pytest.raises(AttributeError):
        _ = opt.bogus_name


def test_composer_joint_rules():
    from catopt_orchestrator.optimize import default_rules
    from catopt_torch.composer import _JOINT_DEFAULT, _joint_rules

    # the pipeline default widens to the layout-inclusive set.
    assert _joint_rules(None) is _JOINT_DEFAULT
    assert _joint_rules(default_rules()) is _JOINT_DEFAULT
    # a caller-chosen set is honoured as-is.
    assert _joint_rules(laws.CATEGORICAL) is laws.CATEGORICAL
    assert set(_JOINT_DEFAULT) >= set(laws.WITH_LAYOUT) - {
        r
        for r in laws.WITH_LAYOUT
        if tags.SUBSUMED in r.tags
    }


# ---------------------------------------------------------------------------
#  Parity — DEFAULT reproduces the retired ruleset="all" path
# ---------------------------------------------------------------------------


class _PairMLP(nn.Module):
    """Two parallel branches — exercises the pairing/fusion rules."""

    def __init__(self):
        super().__init__()
        self.g = nn.Linear(16, 32, bias=False)
        self.u = nn.Linear(16, 32, bias=False)
        self.o = nn.Linear(32, 16, bias=False)

    def forward(self, x):
        import torch.nn.functional as F

        return self.o(F.silu(self.g(x)) * self.u(x))


def test_default_parity_with_legacy_all():
    """``DEFAULT`` reproduces the retired ``ruleset="all"`` behaviour.

    The legacy switch saturated with ``all_rules() - _SUBSUMED`` (46
    rules = today's ``SIMPLIFICATION + CATEGORICAL`` presets).
    ``DEFAULT`` drops the ``SYMMETRY`` generators and adds the core
    carrier laws by design; on this fixture (no scan sources, no
    scale hoists) saturation parity is the same extracted term and
    the same fires for every rule the two sets share — so the preset
    change cannot silently alter a default-pipeline result.
    """
    from catopt_orchestrator import search
    from catopt_torch.adapters import TorchSource
    from catopt_torch.torch_bridge import ir_to_torch_module
    from catopt_core.ir import op_repr

    torch.manual_seed(0)
    m = _PairMLP().eval().double()
    x = torch.randn(2, 16, dtype=torch.float64)

    legacy = RuleSet(
        "legacy-all",
        tuple(
            r for r in all_rules() if r.name not in SUBSUMED_NAMES
        ),
    )
    new = RuleSet(
        "new-default",
        tuple(r for r in laws.DEFAULT if r.name in set(legacy)),
    )
    res_legacy = search(
        m, x, source=TorchSource(), rules=legacy, max_iterations=6
    )
    res_new = search(
        m, x, source=TorchSource(), rules=new, max_iterations=6
    )

    # Same extracted member and same fires on the shared rules.
    assert op_repr(res_new.term) == op_repr(res_legacy.term)
    shared = {r.name for r in new}
    assert res_new.stats["rule_fires"] == {
        k: v
        for k, v in res_legacy.stats["rule_fires"].items()
        if k in shared
    }

    # and the two terms genuinely agree on inputs.
    from catopt_core.ir import IR

    tm_old = ir_to_torch_module(
        IR(root=res_legacy.term, inputs=res_legacy.ir.inputs),
        param_values=res_legacy.param_values,
    )
    tm_new = ir_to_torch_module(
        IR(root=res_new.term, inputs=res_new.ir.inputs),
        param_values=res_new.param_values,
    )
    with torch.no_grad():
        assert torch.allclose(tm_old(x), tm_new(x), atol=1e-12)
        assert torch.allclose(m(x), tm_new(x), atol=1e-12)


def test_full_ruleset_saturation_smoke():
    """The composed values drive ``EGraph.run`` / ``search`` directly."""
    from catopt_orchestrator import search
    from catopt_torch.adapters import TorchSource

    torch.manual_seed(0)
    m = nn.Linear(8, 8, bias=False).double()
    x = torch.randn(2, 8, dtype=torch.float64)
    res = search(
        m,
        x,
        source=TorchSource(),
        rules=laws.DEFAULT.with_priorities(id_add=laws.LATE),
        max_iterations=3,
    )
    # priorities exist in the data: a LATE rule sorts after NORMAL —
    # the scheduler that *consumes* the map is plan 0010.
    rs = laws.DEFAULT.with_priorities(id_add=laws.LATE)
    assert rs.priority_of("id_add") == laws.LATE
    assert rs.priority_of("assoc_matmul") == laws.NORMAL
    assert "rule_fires" in res.stats
