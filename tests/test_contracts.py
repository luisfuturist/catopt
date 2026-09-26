"""Contract tests — plan 0001 phase 1c.

Mechanical pins at the IR boundary, so that binding/attr/shape bugs
(the ``rms_norm``-missing-binding class, the slice-step-dropped
class) fail loudly here instead of shipping silently through the
integration suite.

Sections:

1. **Binding coverage** — every IR op reachable through
   ``_ATEN_TO_IR`` / ``_IR_TO_TORCH_EXTRA`` canonicalization has an
   executable ``_IR_TO_TORCH`` binding.  Parametrized over the full
   table; canonicalized IR names (not raw aten spellings) are what
   must bind.
2. **Shape contracts** — per-op ``cost._shape_of`` pins for the ops
   whose attr spellings have already produced bugs: slice (the
   step), split (``sizes``/``dim``/``index``), concat,
   reshape numel-mismatch, rank-3 transpose, matvec linear,
   broadcast mismatch, rms_norm.
3. **Attr canonicalization round-trip** — ``export_to_ir`` must land
   positional aten spellings in the canonical attr names
   (``dim``/``chunks``/``sizes``/``index``), and must never drop a
   strided slice's ``step``.
4. **Law soundness fuzzer** — seeded random tensor bindings for every
   rule in ``SIMPLIFICATION_RULES`` + a safe subset of
   ``CATEGORICAL_RULES``; lhs and rhs are instantiated with the
   rule's own machinery (``egraph._term_instantiate``) and evaluated
   through the real ``_IR_TO_TORCH`` bindings.  Rules whose terms
   cannot be bound safely are skipped (counted, not failed).

NOTE on plan item 5: the deepcopy-fallback regression
(``optimize_compositional`` must not graft shape-specialized
IRModules into the caller's model when ``copy.deepcopy(model)``
fails) is already pinned at integration level by
``tests/test_batched_verify.py::test_compositional_inplace_fallback_preserves_model``
— that test IS the contract (caller-model immutability across the
in-place fallback), so it is not duplicated here.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from catopt.cost import _INVALID, _shape_of
from catopt.egraph import _term_instantiate
from catopt.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import CATEGORICAL_RULES, SIMPLIFICATION_RULES
from catopt.torch_bridge import (
    _ATEN_TO_IR,
    _IR_TO_TORCH,
    _IR_TO_TORCH_EXTRA,
    _canon_aten_name,
    export_to_ir,
)


def _var(shape: tuple, name: str) -> Var:
    return Var(name=name, typ=TensorType(tuple(shape)))


def _iter_ops(term):
    """Yield every Op node in a term (DAG-aware by identity)."""
    seen = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op) and id(t) not in seen:
            seen.add(id(t))
            yield t
            stack.extend(t.args)


# ---------------------------------------------------------------------------
#  1. Binding coverage
# ---------------------------------------------------------------------------

_RAW_ATEN_NAMES = sorted(set(_ATEN_TO_IR) | set(_IR_TO_TORCH_EXTRA))

#: Every IR op name either map can emit.
_REACHABLE_IR_OPS = sorted(
    set(_ATEN_TO_IR.values()) | set(_IR_TO_TORCH_EXTRA.values())
)


class TestBindingCoverage:
    """Every canonicalized IR op must have an executable binding.

    This is the ``rms_norm``-class guard: an exported graph node whose
    aten name canonicalizes to an op with no ``_IR_TO_TORCH`` entry
    crashes at ``IRModule._eval`` with 'No torch binding for op ...'.
    """

    @pytest.mark.parametrize("ir_op", _REACHABLE_IR_OPS)
    def test_canonical_ir_op_is_bound(self, ir_op):
        fn = _IR_TO_TORCH.get(ir_op)
        assert fn is not None, (
            f"CONTRACT VIOLATION: canonical IR op '{ir_op}' is "
            "reachable via _ATEN_TO_IR/_IR_TO_TORCH_EXTRA but has no "
            "_IR_TO_TORCH binding — exported graphs containing it "
            "cannot be lowered"
        )
        assert callable(fn), f"binding for '{ir_op}' is not callable"

    @pytest.mark.parametrize("raw", _RAW_ATEN_NAMES)
    def test_raw_aten_name_resolves_to_bound_op(self, raw):
        ir_op = _canon_aten_name(raw)
        assert ir_op in _IR_TO_TORCH, (
            f"CONTRACT VIOLATION: aten '{raw}' canonicalizes to "
            f"'{ir_op}', which has no _IR_TO_TORCH binding"
        )
        assert callable(_IR_TO_TORCH[ir_op])

    @pytest.mark.parametrize("raw", _RAW_ATEN_NAMES)
    def test_overload_suffixed_spelling_resolves(self, raw):
        """'x.default'/'x.backend' spellings of every mapped name must
        canonicalize to a bound op too — torch.export emits the
        suffixed overload names."""
        ir_op = _canon_aten_name(raw + ".default")
        assert ir_op in _IR_TO_TORCH, (
            f"CONTRACT VIOLATION: aten '{raw}.default' canonicalizes "
            f"to '{ir_op}', which has no _IR_TO_TORCH binding"
        )

    def test_canonical_op_table_is_subset_of_bindings(self):
        """The set-level contract, in one failure message."""
        unbound = (
            set(_ATEN_TO_IR.values()) | set(_IR_TO_TORCH_EXTRA.values())
        ) - set(_IR_TO_TORCH)
        assert unbound == set(), (
            f"CONTRACT VIOLATION: unbound canonical IR ops: "
            f"{sorted(unbound)}"
        )


# ---------------------------------------------------------------------------
#  2. Shape contracts
# ---------------------------------------------------------------------------

_INT64_MAX = 2**63 - 1


class TestShapeContracts:
    """``cost._shape_of`` per-op pins, built from Op terms directly."""

    # -- slice: ``step`` is the stride (exported positionally)
    def test_slice_strided_last_dim(self):
        # x[..., ::2] on (2, 3, 8): torch.export emits
        # slice(t, dim, start, end, step).
        x = _var((2, 3, 8), "x")
        t = Op.make(
            "slice", x, dim=2, start=0, end=_INT64_MAX, step=2
        )
        assert _shape_of(t) == (2, 3, 4)

    def test_slice_strided_odd_extent(self):
        x = _var((2, 3, 7), "x")
        t = Op.make("slice", x, dim=2, start=0, end=None, step=2)
        assert _shape_of(t) == (2, 3, 4)  # ceil(7/2)

    def test_slice_default_step_is_identity(self):
        x = _var((2, 8), "x")
        t = Op.make("slice", x, dim=1, start=0, end=_INT64_MAX)
        assert _shape_of(t) == (2, 8)

    def test_slice_dim_spelling(self):
        # 'dim' is the canonical axis name.
        x = _var((2, 8), "x")
        t = Op.make("slice", x, dim=1, start=2, end=6)
        assert _shape_of(t) == (2, 4)

    # -- split: canonical ``sizes``/``dim``/``index`` shapes correctly
    def test_split_sizes_index_spelling(self):
        x = _var((2, 12), "x")
        t = Op.make("split", x, sizes=(4, 8), dim=1, index=1)
        assert _shape_of(t) == (2, 8)

    def test_split_sizes_list_spelling(self):
        # exported list-split keeps the list under ``sizes`` and the
        # getitem-fold index under ``index``.
        x = _var((2, 12), "x")
        t = Op.make("split", x, sizes=(4, 8), index=1, dim=1)
        assert _shape_of(t) == (2, 8)

    def test_split_sizes_int_spelling(self):
        # torch.split(x, 4) — equal-size sections, int under ``sizes``.
        x = _var((2, 12), "x")
        t = Op.make("split", x, sizes=4, index=0, dim=1)
        assert _shape_of(t) == (2, 4)

    # -- concat sums operand extents along the cat axis
    def test_concat_sums_along_dim(self):
        a, b, c = _var((2, 4), "a"), _var((2, 8), "b"), _var((2, 1), "c")
        t = Op.make("concat", a, b, c, dim=1)
        assert _shape_of(t) == (2, 13)

    def test_concat_dim_spelling(self):
        a, b = _var((2, 4), "a"), _var((2, 8), "b")
        t = Op.make("concat", a, b, dim=-1)
        assert _shape_of(t) == (2, 12)

    # -- reshape must preserve numel; a mismatch is _INVALID (poisonous)
    def test_reshape_valid(self):
        x = _var((2, 6), "x")
        assert _shape_of(Op.make("reshape", x, shape=(3, 4))) == (3, 4)

    def test_reshape_minus_one_resolved(self):
        x = _var((2, 6), "x")
        assert _shape_of(Op.make("reshape", x, shape=(3, -1))) == (3, 4)

    def test_reshape_numel_mismatch_is_invalid(self):
        x = _var((2, 6), "x")
        assert _shape_of(Op.make("reshape", x, shape=(3, 5))) is _INVALID

    # -- transpose on rank-3
    def test_transpose_rank3(self):
        x = _var((2, 3, 4), "x")
        assert _shape_of(
            Op.make("transpose", x, dim0=1, dim1=2)
        ) == (2, 4, 3)
        assert _shape_of(
            Op.make("transpose", x, dim0=-2, dim1=-1)
        ) == (2, 4, 3)

    # -- matvec / linear rank-1 cases
    def test_matmul_matvec(self):
        w, v = _var((4, 3), "w"), _var((3,), "v")
        assert _shape_of(Op.make("matmul", w, v)) == (4,)

    def test_matmul_dot(self):
        a, b = _var((3,), "a"), _var((3,), "b")
        assert _shape_of(Op.make("matmul", a, b)) == ()

    def test_linear_matvec(self):
        x, w = _var((3,), "x"), _var((4, 3), "w")
        assert _shape_of(Op.make("linear", x, w)) == (4,)

    # -- broadcast mismatch is _INVALID
    def test_broadcast_mismatch_is_invalid(self):
        a, b = _var((2, 3), "a"), _var((2, 4), "b")
        assert _shape_of(Op.make("add", a, b)) is _INVALID
        assert _shape_of(Op.make("mul", a, b)) is _INVALID

    def test_broadcast_valid(self):
        a, b = _var((2, 1), "a"), _var((1, 3), "b")
        assert _shape_of(Op.make("add", a, b)) == (2, 3)

    # -- rms_norm: exported attrs are dim=normalized_shape + eps=eps
    def test_rms_norm_passthrough_shape(self):
        x, w = _var((2, 8), "x"), _var((8,), "w")
        t = Op.make("rms_norm", x, w, dim=(8,), eps=1e-5)
        assert _shape_of(t) == (2, 8)


# ---------------------------------------------------------------------------
#  3. Attr canonicalization round-trip
# ---------------------------------------------------------------------------


class _CatModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.cat([x, y], dim=-1)


class _ChunkModel(torch.nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a + b


class _SplitSizesModel(torch.nn.Module):
    def forward(self, x):
        a, b = torch.split(x, [4, 4], dim=-1)
        return a + b


class _SliceModel(torch.nn.Module):
    def forward(self, x):
        return x[..., ::2]


class _RMSNormModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(8))

    def forward(self, x):
        return F.rms_norm(x, (8,), self.w, eps=1e-5)


def _export(model, *args):
    ir, _tensors = export_to_ir(model.eval(), tuple(args))
    return list(_iter_ops(ir.root))


def _ops_named(ops, name):
    return [t for t in ops if t.op == name]


class TestAttrCanonicalization:
    """Exported IR must carry canonical attr spellings — a bare
    ``argN`` where ``_ATTR_RENAMES``/the canonical schema declares a
    name is a contract violation (rules and ``_shape_of`` only read
    the canonical names)."""

    def test_concat_dim_canonical(self):
        ops = _ops_named(
            _export(_CatModel(), torch.randn(2, 4), torch.randn(2, 4)),
            "concat",
        )
        assert ops, "expected a concat node in the exported graph"
        for t in ops:
            assert "dim" in t.attrs, f"concat missing canonical 'dim': {t.attrs}"
            assert not any(
                k.startswith("arg") for k in t.attrs
            ), f"concat carries bare positional attrs: {t.attrs}"

    def test_chunk_attrs_canonical(self):
        ops = _ops_named(_export(_ChunkModel(), torch.randn(2, 8)), "chunk")
        assert ops, "expected chunk nodes in the exported graph"
        for t in ops:
            assert "chunks" in t.attrs, f"chunk missing 'chunks': {t.attrs}"
            assert "dim" in t.attrs, f"chunk missing 'dim': {t.attrs}"
            assert "index" in t.attrs, (
                "chunk must carry the getitem-folded 'index'"
            )
            assert not any(
                k.startswith("arg") for k in t.attrs
            ), f"chunk carries bare positional attrs: {t.attrs}"

    def test_split_attrs_canonical(self):
        ops = _ops_named(
            _export(_SplitSizesModel(), torch.randn(2, 8)), "split"
        )
        assert ops, "expected split nodes in the exported graph"
        for t in ops:
            assert "sizes" in t.attrs, f"split missing 'sizes': {t.attrs}"
            assert "dim" in t.attrs, (
                "split must carry the canonical 'dim'"
            )
            assert "index" in t.attrs, (
                "split must carry the getitem-folded 'index'"
            )
            assert "arg2" not in t.attrs and "arg3" not in t.attrs, (
                f"split carries bare positional attrs: {t.attrs}"
            )

    def test_strided_slice_step_survives_export(self):
        ops = _ops_named(
            _export(_SliceModel(), torch.randn(2, 3, 8)), "slice"
        )
        assert ops, "expected a slice node in the exported graph"
        (t,) = ops  # exactly one slice
        assert t.attrs.get("step") == 2, (
            "CONTRACT VIOLATION: strided slice lost its step — "
            f"attrs are {t.attrs}"
        )
        # round-trip: the exported term must also shape correctly.
        assert _shape_of(t) == (2, 3, 4)

    def test_rms_norm_attrs_canonical(self):
        ops = _ops_named(
            _export(_RMSNormModel(), torch.randn(2, 8)), "rms_norm"
        )
        assert ops, "expected an rms_norm node in the exported graph"
        for t in ops:
            assert "dim" in t.attrs, (
                "rms_norm normalized_shape must land under 'dim'"
            )
            assert not any(
                k.startswith("arg") for k in t.attrs
            ), (
                "CONTRACT VIOLATION: rms_norm carries bare positional "
                f"attrs {t.attrs} — the canonical name 'eps' exists "
                "in the _IR_TO_TORCH binding but the boundary leaves "
                "argN"
            )


# ---------------------------------------------------------------------------
#  4. Law soundness fuzzer (bounded, seeded)
# ---------------------------------------------------------------------------

#: Per-rule binding specs: metavar -> concrete shape, plus attribute
#: metavar values ("$attr:NAME" -> value).  Shapes are chosen so both
#: sides are well-typed AND every declared ``check``/``derive`` is
#: satisfiable — the point is to prove the LAW, on real tensors, for
#: bindings where the rule legitimately applies.
_LAW_SPECS: dict[str, dict] = {
    # ---- SIMPLIFICATION_RULES (all bindable) ----
    "comm_add": {"a": (2, 3), "b": (2, 3)},
    "comm_mul": {"a": (2, 3), "b": (2, 3)},
    "assoc_add": {"a": (2, 3), "b": (2, 3), "c": (2, 3)},
    "assoc_mul": {"a": (2, 3), "b": (2, 3), "c": (2, 3)},
    "id_add": {"a": (2, 3)},
    "id_mul": {"a": (2, 3)},
    "double_neg": {"a": (2, 3)},
    "sub_to_add": {"a": (2, 3), "b": (2, 3)},
    "silu_expand": {"x": (2, 3)},
    "silu_mul_form": {"g": (2, 3), "u": (2, 3)},
    "square_expand": {"x": (2, 3)},
    "pow_to_square": {"x": (2, 3)},
    "square_to_pow": {"x": (2, 3)},
    # ---- CATEGORICAL_RULES: matmul/linear algebra ----
    "distribute_matmul_over_add": {
        "W": (4, 3),
        "a": (3, 5),
        "b": (3, 5),
    },
    "factor_matmul": {"W": (4, 3), "a": (3, 5), "b": (3, 5)},
    "right_distribute_matmul": {
        "a": (2, 3),
        "b": (2, 3),
        "W": (3, 4),
    },
    "right_factor_matmul": {"a": (2, 3), "b": (2, 3), "W": (3, 4)},
    "weight_factor_matmul": {
        "x": (2, 3),
        "W": (3, 4),
        "W2": (3, 4),
    },
    "weight_distribute_matmul": {
        "x": (2, 3),
        "W": (3, 4),
        "W2": (3, 4),
    },
    "weight_factor_linear": {
        "x": (2, 3),
        "W": (4, 3),
        "W2": (4, 3),
    },
    "weight_distribute_linear": {
        "x": (2, 3),
        "W": (4, 3),
        "W2": (4, 3),
    },
    "right_factor_linear": {
        "a": (2, 3),
        "b": (2, 3),
        "W": (4, 3),
    },
    "assoc_linear": {"x": (2, 3), "A": (5, 3), "B": (4, 5)},
    # check requires A(h,i), B(o,h), b1(h,), b2 scalar|(o,), x[...,i]
    "assoc_linear_bias": {
        "x": (2, 3),
        "A": (5, 3),
        "B": (4, 5),
        "b1": (5,),
        "b2": (4,),
    },
    "assoc_linear_bias_rev": {
        "x": (2, 3),
        "A": (5, 3),
        "B": (4, 5),
        "b1": (5,),
        "b2": (4,),
    },
    # check requires c scalar-shaped
    "naturality_scalar": {"W": (2, 3), "x": (3, 4), "c": ()},
    "naturality_scalar_rev": {"W": (2, 3), "x": (3, 4), "c": ()},
    "assoc_matmul": {"A": (2, 3), "B": (3, 4), "C": (4, 5)},
    "assoc_matmul_rev": {"A": (2, 3), "B": (3, 4), "C": (4, 5)},
    # ---- product structure ----
    "swiglu_fuse": {"x": (2, 3), "A": (4, 3), "B": (4, 3)},
    "parallel_mul_fuse": {"x": (2, 3), "A": (4, 3), "B": (4, 3)},
    "qkv_fuse": {
        "shapes": {"x": (2, 3), "Q": (4, 3), "K": (4, 3), "V": (4, 3)},
        "attrs": {"$attr:S": (2, 2, 2), "$attr:SC": 0.5},
    },
    "qkv_fuse_asym": {
        "shapes": {
            "x": (1, 2, 3),
            "Q": (4, 3),
            "K": (2, 3),
            "V": (2, 3),
        },
        "attrs": {
            "$attr:S1": (1, 2, 2, 2),
            "$attr:S2": (1, 2, 1, 2),
            "$attr:S3": (1, 2, 1, 2),
            "$attr:SC": 0.5,
            "$attr:G": True,
        },
    },
    # ---- diagonal-scale naturality ----
    # channel scale c ~ (in,); row scale r ~ (B,T,1)
    "linear_channel_scale": {"x": (2, 3), "c": (3,), "W": (4, 3)},
    "linear_channel_scale_rev": {"x": (2, 3), "c": (3,), "W": (4, 3)},
    "linear_row_scale": {"x": (2, 2, 3), "r": (2, 2, 1), "W": (4, 3)},
    "linear_row_scale_rev": {
        "x": (2, 2, 3),
        "r": (2, 2, 1),
        "W": (4, 3),
    },
}

#: Rules deliberately not fuzzed: they need side-condition checks that
#: cannot be satisfied by unconstrained random bindings (GQA repeat-
#: chain absorption, the 12 sdpa_fold mask/scale forms) — or carry
#: non-tensor carrier values.  They are counted, not failed.
_ALL_LAW_RULES = SIMPLIFICATION_RULES + CATEGORICAL_RULES

_LAW_TRIALS = 4
_LAW_ATOL = 1e-10


def _eval_bound_term(term, env):
    """Evaluate a concrete (metavar-free) term via the real
    ``_IR_TO_TORCH`` bindings — the same dispatch ``IRModule._eval``
    performs."""
    if isinstance(term, (Var, Param)):
        return env[term.name]
    if isinstance(term, Const):
        return torch.tensor(term.value, dtype=torch.float64)
    if isinstance(term, Op):
        fn = _IR_TO_TORCH.get(term.op)
        if fn is None:
            raise ValueError(f"No torch binding for op '{term.op}'")
        args = [_eval_bound_term(a, env) for a in term.args]
        return fn(*args, **dict(term.attrs))
    raise TypeError(f"cannot evaluate {term!r}")


def _law_spec(rule_name: str) -> dict | None:
    spec = _LAW_SPECS.get(rule_name)
    if spec is None:
        return None
    if "shapes" in spec:
        return spec
    return {"shapes": spec, "attrs": {}}


class TestLawSoundness:
    """lhs(env) == rhs(env) on seeded random fp64 tensors, evaluated
    through the real torch bindings.  A mismatch is a soundness bug in
    the rule OR in a binding — both are contract violations."""

    @pytest.mark.parametrize(
        "rule", _ALL_LAW_RULES, ids=lambda r: r.name
    )
    def test_rule_sound_on_random_bindings(self, rule):
        spec = _law_spec(rule.name)
        if spec is None:
            pytest.skip(
                f"no safe random binding for '{rule.name}' "
                "(side condition not randomly satisfiable)"
            )
        shapes = spec["shapes"]
        for trial in range(_LAW_TRIALS):
            gen = torch.Generator().manual_seed(
                hash((rule.name, trial)) & 0x7FFFFFFF
            )
            subst, env = {}, {}
            for mvar, shape in sorted(shapes.items()):
                leaf = Var(f"v_{mvar}", TensorType(tuple(shape)))
                subst[mvar] = leaf
                env[leaf.name] = torch.randn(
                    shape, generator=gen, dtype=torch.float64
                )
            subst.update(spec.get("attrs", {}))
            # A declared side condition must ACCEPT our binding — a
            # rejection means the spec is wrong, not that the rule is
            # shy; fail loudly either way.
            if rule.check is not None:
                assert rule.check(subst), (
                    f"'{rule.name}' check() rejected the spec binding "
                    f"{shapes} on trial {trial} — bad test spec or "
                    "overly strict guard"
                )
            rhs_subst = dict(subst)
            if rule.derive is not None:
                derived = rule.derive(subst)
                assert derived is not None, (
                    f"'{rule.name}' derive() vetoed the spec binding"
                )
                rhs_subst.update(derived)
            lhs_term = _term_instantiate(rule.lhs, subst)
            rhs_term = _term_instantiate(rule.rhs, rhs_subst)
            lhs_val = _eval_bound_term(lhs_term, env)
            rhs_val = _eval_bound_term(rhs_term, env)
            assert torch.allclose(
                lhs_val, rhs_val, atol=_LAW_ATOL, rtol=0
            ), (
                f"LAW VIOLATION in '{rule.name}' (trial {trial}): "
                f"lhs != rhs, max abs diff "
                f"{(lhs_val - rhs_val).abs().max().item():.3e}"
            )

    def test_fuzzer_coverage_floor(self):
        """The fuzzer must cover at least the promised ~15-rule slice —
        this fails if rules get renamed/dropped and the spec table
        silently goes stale."""
        covered = {r.name for r in _ALL_LAW_RULES} & set(_LAW_SPECS)
        assert len(covered) >= 15, (
            f"only {len(covered)} rules have binding specs; expected "
            ">=15"
        )
