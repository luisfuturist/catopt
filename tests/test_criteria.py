"""Criterion-based selection — ``criteria_cost`` blends and the
``optimize_model(criteria=...)`` param.

The blend is a weighted sum of named cost axes
(:func:`catopt_optimize.criteria.criteria_cost`).  Sum of
additive-per-node models is additive, so extraction's subtractive
local-cost recovery stays exact for the additive axes — the tests
below check that contract, the selection flip when two axes
disagree, the validation surface, and the stats record.
"""

import math

import pytest
import torch
from catopt.cost import (
    _LAUNCH_S,
    _local_roofline,
    dag_cost,
    executor_cost_for,
    flops_cost,
    param_bytes_cost_for,
)
from catopt.egraph import EGraph
from catopt.ir import Op, Param, TensorType, Var
from catopt.ports import CostFn, signature_conforms
from catopt_optimize.criteria import (
    AXES,
    Blend,
    CompiledCriterion,
    Criteria,
    Criterion,
    DepthCriterion,
    FlopsCriterion,
    LatencyCriterion,
    MemoryCriterion,
    criteria_cost,
    peak_bytes_cost,
)


def _v(name: str, *shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape) -> Param:
    return Param(name, TensorType(tuple(shape)))


def _member_storage() -> Op:
    """Compute-light, memory-heavy: one GEMM on a *stored* weight —
    2048 FLOPs, 16 stored values."""
    x = _v("x", 64, 4)
    return Op.make("matmul", x, _p("p_w", 4, 4))


def _member_compute() -> Op:
    """Compute-heavy, memory-free: a larger GEMM on a *data* operand
    plus a neg — 4608 FLOPs, nothing stored."""
    x = _v("x", 64, 4)
    y = _v("y", 4, 8)
    return Op.make("neg", Op.make("matmul", x, y))


def _two_member_graph():
    """One e-class holding the storage member and the compute member
    (the union asserts their equality — the same mechanism a
    witnessed rewrite uses)."""
    a = _member_storage()
    b = _member_compute()
    eg = EGraph()
    ea = eg.add_term(a)
    eb = eg.add_term(b)
    eg.union(ea, eb)
    eg.rebuild()
    return eg, eg.find(ea), a, b


# ---------------------------------------------------------------------------
#  Validation
# ---------------------------------------------------------------------------


def test_criteria_unknown_axis_rejected():
    with pytest.raises(ValueError, match="unknown criteria axes"):
        criteria_cost({"latency": 1.0, "speed": 1.0})
    # every documented axis is accepted
    for axis in AXES:
        criteria_cost({axis: 1.0})


@pytest.mark.parametrize(
    "bad", [-1.0, float("nan"), float("inf"), "x", None]
)
def test_criteria_bad_weight_rejected(bad):
    with pytest.raises(ValueError, match="non-negative weight"):
        criteria_cost({"latency": bad})


def test_criteria_no_positive_weight_rejected():
    with pytest.raises(ValueError, match="positive weight"):
        criteria_cost({})
    with pytest.raises(ValueError, match="positive weight"):
        criteria_cost({"latency": 0.0, "flops": 0.0})


# ---------------------------------------------------------------------------
#  The blend itself
# ---------------------------------------------------------------------------


def test_criteria_none_is_the_shipped_default():
    """criteria=None blends {"latency": 1.0} — exactly
    ``executor_cost_for(lowering="generic")``."""
    t = Op.make("add", _v("x", 4, 8), _p("w", 4, 8))
    assert criteria_cost()(t) == pytest.approx(
        executor_cost_for(lowering="generic")(t)
    )


def test_criteria_weights_normalise_to_one():
    """Only relative weights matter — a uniform rescale is the same
    blend, recorded normalised."""
    t = Op.make("mul", _v("x", 8, 8), _v("y", 8, 8))
    a = criteria_cost({"flops": 2.0, "latency": 6.0})
    b = criteria_cost({"flops": 1.0, "latency": 3.0})
    assert a(t) == pytest.approx(b(t))
    assert a.criteria == {"flops": 0.25, "latency": 0.75}
    assert b.criteria == a.criteria
    assert sum(b.criteria.values()) == pytest.approx(1.0)
    # and the blend value is the weighted sum of the axis models
    assert a(t) == pytest.approx(
        0.25 * flops_cost(t)
        + 0.75 * executor_cost_for(lowering="generic")(t)
    )


def test_criteria_zero_weight_axis_is_dropped():
    c = criteria_cost({"flops": 1.0, "memory": 0.0})
    assert "memory" not in c.criteria
    assert c.charges_param_only is False


def test_criteria_memory_axis_sets_charges_param_only():
    """Storage billing must survive the param-only discount — the
    blend carries param_bytes_cost's marker through."""
    c = criteria_cost({"memory": 1.0})
    assert c.charges_param_only is True
    t = Op.make("linear", _v("x", 4, 64), _p("W", 64, 64))
    assert c(t) == pytest.approx(param_bytes_cost_for()(t))


def test_criteria_conforms_to_costfn_port():
    assert signature_conforms(criteria_cost({"flops": 1.0}), CostFn)


def test_criteria_profile_binds_time_axes():
    """profile threads into the ns-priced axes and rides the closure
    for reporting / extraction's memo pre-seeding."""
    prof = {"tflops": 10.0, "gbps": 100.0, "launch_us": 1.0}
    c = criteria_cost({"latency": 1.0, "depth": 1.0}, profile=prof)
    assert c.profile is prof
    t = Op.make("matmul", _v("x", 32, 32), _p("W", 32, 32))
    assert math.isfinite(c(t))


def test_criteria_all_axes_price_finite():
    """Every axis yields a callable blend — depth/compiled included
    (their whole-DAG contribution is the documented approximation)."""
    c = criteria_cost(
        {a: 1.0 for a in AXES},
        profile={"tflops": 2.5, "gbps": 89.0, "launch_us": 8.7},
    )
    t = Op.make(
        "add",
        Op.make("matmul", _v("x", 16, 16), _p("W", 16, 16)),
        _v("y", 16, 16),
    )
    assert math.isfinite(c(t))


# ---------------------------------------------------------------------------
#  Additive safety — the extract_best contract
# ---------------------------------------------------------------------------


def test_criteria_blend_recovers_exact_local():
    """For additive axes the recovered local
    ``blend(t) - sum(blend(children))`` is exactly the weighted sum of
    the node-level contributions — what extract_best bills."""
    x = _v("x", 16, 32)
    y = _v("y", 16, 32)
    t = Op.make("add", Op.make("mul", x, y), Op.make("neg", x))
    wf, wl = 0.25, 0.75
    blend = criteria_cost({"flops": wf, "latency": wl})
    local = blend(t) - sum(blend(a) for a in t.args)
    # flops: add bills 1 * numel(out) = 512; latency: the node's local
    # roofline plus one generic dispatch (per = launch_s in ns).
    expected = wf * 512.0 + wl * (_local_roofline(t) + _LAUNCH_S * 1e9)
    assert local == pytest.approx(expected)


def test_criteria_extraction_picks_true_blend_min():
    """A saturated class with two members: extraction returns the
    member minimizing the blend — checked against direct prices."""
    eg, root, a, b = _two_member_graph()
    blend = criteria_cost({"flops": 1.0, "memory": 1.0})
    best = eg.extract_best(root, blend)
    assert best == (a if blend(a) < blend(b) else b)


# ---------------------------------------------------------------------------
#  Weighted blend changes selection — two axes disagree
# ---------------------------------------------------------------------------


def test_criteria_blend_flips_selection_between_axes():
    """The storage member wins on FLOPs; the compute member wins on
    memory (it stores nothing).  The blend crosses over as the
    memory weight grows."""
    eg, root, storage, compute = _two_member_graph()
    assert flops_cost(storage) < flops_cost(compute)
    assert param_bytes_cost_for()(compute) < param_bytes_cost_for()(
        storage
    )

    assert (
        eg.extract_best(root, criteria_cost({"flops": 1.0})) == storage
    )
    assert (
        eg.extract_best(root, criteria_cost({"memory": 1.0})) == compute
    )
    # flops-dominant blend: storage still wins
    assert (
        eg.extract_best(
            root, criteria_cost({"flops": 1.0, "memory": 0.001})
        )
        == storage
    )
    # memory-dominant blend: the crossover has passed
    assert (
        eg.extract_best(
            root, criteria_cost({"flops": 1.0, "memory": 1000.0})
        )
        == compute
    )


# ---------------------------------------------------------------------------
#  optimize_model(criteria=...) — precedence + stats record
# ---------------------------------------------------------------------------


def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.SiLU(), torch.nn.Linear(8, 8)
    ).eval()


def test_optimize_model_criteria_records_stats():
    """criteria= builds the blend, stats records the normalised
    weights actually priced, and the module stays equivalent."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(
            m, x, verbose=False, criteria={"latency": 2.0}
        )
    assert stats["criteria"] == {"latency": 1.0}
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_default_run_records_no_criteria():
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        _mod, stats = optimize_model(m, x, verbose=False)
    assert stats["criteria"] is None


def test_optimize_model_explicit_cost_fn_beats_criteria():
    """Precedence: explicit cost_fn > criteria > default — a given
    cost_fn leaves the criteria record empty (none was priced)."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(
            m,
            x,
            verbose=False,
            cost_fn=flops_cost,
            criteria={"memory": 1.0},
        )
    assert stats["criteria"] is None
    with torch.no_grad():
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_bad_criteria_fails_loud():
    """An unknown axis surfaces as ValueError from the optimizer —
    validation isn't deferred to extraction."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with pytest.raises(ValueError, match="unknown criteria axes"):
        optimize_model(m, x, verbose=False, criteria={"speed": 1.0})


# ---------------------------------------------------------------------------
#  Criterion objects — the pluggable axis contract
# ---------------------------------------------------------------------------


class _NamedCriterion:
    """A user criterion: any object with ``cost_fn(profile)`` — here
    an axis billing each op one credit (a count axis)."""

    name = "opcount"

    def cost_fn(self, profile=None):
        del profile  # uncalibrated axis

        def fn(term, memo=None):
            return float(sum(isinstance(t, Op) for t in _walk(term)))

        return fn


def _walk(term):
    seen, stack, out = set(), [term], []
    while stack:
        t = stack.pop()
        if isinstance(t, Op) and t not in seen:
            seen.add(t)
            out.append(t)
            stack.extend(t.args)
    return out


class _NamelessCriterion:
    """A criterion with no ``name`` — the class name stands in."""

    def cost_fn(self, profile=None):
        del profile
        return flops_cost


class _ConstCriterion:
    """A criterion whose cost fn takes NO memo — the blend calls it
    bare."""

    name = "const"

    def cost_fn(self, profile=None):
        del profile

        def fn(term):
            return 7.0

        return fn


class _WeirdSigCriterion:
    """A criterion whose built fn's signature is uninspectable —
    the blend falls back to a bare call."""

    name = "weird"

    def cost_fn(self, profile=None):
        del profile

        class _F:
            @property
            def __signature__(self):
                raise ValueError("uninspectable")

            def __call__(self, term):
                return 3.0

        return _F()


class _StorageCriterion:
    """A criterion declaring the storage-billing marker — propagated
    to the blend."""

    name = "storey"
    charges_param_only = True

    def cost_fn(self, profile=None):
        del profile
        return flops_cost


def test_criterion_protocol_runtime_conformance():
    """Built-ins, containers and a duck-typed user criterion all
    satisfy the runtime-checkable Criterion protocol."""
    assert isinstance(LatencyCriterion(), Criterion)
    assert isinstance(MemoryCriterion("peak"), Criterion)
    assert isinstance(Criteria(FlopsCriterion()), Criterion)
    assert isinstance(Blend([(FlopsCriterion(), 1.0)]), Criterion)
    assert isinstance(_NamedCriterion(), Criterion)
    # a bare callable is NOT a criterion — no cost_fn member
    assert not isinstance(flops_cost, Criterion)


def test_custom_criterion_single_and_named():
    """A user criterion drops straight into criteria_cost — its name
    (or class name) lands in the blend record."""
    t = Op.make("mul", _v("x", 4, 8), _v("y", 4, 8))
    c = criteria_cost(_NamedCriterion())
    assert c.criteria == {"opcount": 1.0}
    assert c(t) == pytest.approx(1.0)  # one op node
    t2 = Op.make(
        "mul",
        Op.make("add", _v("x", 4, 8), _v("y", 4, 8)),
        _v("z", 4, 8),
    )
    assert c(t2) == pytest.approx(2.0)  # add + mul
    # nameless criterion falls back to the class name
    c2 = criteria_cost(_NamelessCriterion())
    assert c2.criteria == {"_NamelessCriterion": 1.0}


def test_custom_criterion_in_blend_and_no_memo_fn():
    """Custom criteria blend with built-ins; a cost fn without a
    memo parameter is called bare; an uninspectable-signature fn is
    treated as memo-free."""
    t = Op.make("mul", _v("x", 4, 8), _v("y", 4, 8))
    blend = criteria_cost([_NamedCriterion(), (_ConstCriterion(), 1.0)])
    assert blend.criteria == {"opcount": 0.5, "const": 0.5}
    assert blend(t) == pytest.approx(0.5 * 1.0 + 0.5 * 7.0)
    weird = criteria_cost(_WeirdSigCriterion())
    assert weird(t) == pytest.approx(3.0)


def test_criterion_marker_aggregation():
    """charges_param_only / charges_shape aggregate: declared on the
    criterion or the built fn, any member turns the blend's flag on."""
    store = criteria_cost(_StorageCriterion())
    assert store.charges_param_only is True
    peak = criteria_cost(MemoryCriterion("peak"))
    assert peak.charges_param_only is False
    assert peak.charges_shape is True
    plain = criteria_cost(_ConstCriterion())
    assert plain.charges_param_only is False
    assert plain.charges_shape is False
    mixed = criteria_cost({"flops": 1.0, "memory": 1.0})
    assert mixed.charges_param_only is True
    assert mixed.charges_shape is True


# ---------------------------------------------------------------------------
#  Composition — `crit * w`, `a + b`, Criteria/Blend containers
# ---------------------------------------------------------------------------


def test_composition_operators_build_blends():
    """LatencyCriterion()*0.7 + MemoryCriterion()*0.3 → a Blend that
    IS a Criterion, normalised when priced."""
    b = LatencyCriterion() * 0.7 + MemoryCriterion() * 0.3
    assert isinstance(b, Blend)
    assert isinstance(b, Criterion)
    assert isinstance(b, Criteria)
    names = [c.name for c, _ in b.terms]
    assert names == ["latency", "memory"]
    assert [w for _, w in b.terms] == [0.7, 0.3]
    fn = b.blend()
    assert signature_conforms(fn, CostFn)
    assert fn.criteria == {"latency": 0.7, "memory": 0.3}
    assert fn.charges_param_only is True
    t = Op.make("add", _v("x", 8, 8), _p("w", 8, 8))
    assert fn(t) == pytest.approx(
        0.7 * executor_cost_for(lowering="generic")(t)
        + 0.3 * param_bytes_cost_for()(t)
    )


def test_composition_variants():
    """float * crit (rmul), crit + crit (unit weights), Blend * k
    (rescales members), Blend + crit, and sum() composition."""
    b1 = 0.5 * FlopsCriterion()
    assert isinstance(b1, Blend)
    assert [w for _, w in b1.terms] == [0.5]

    b2 = FlopsCriterion() + DepthCriterion()
    assert [w for _, w in b2.terms] == [1.0, 1.0]
    assert criteria_cost(b2).criteria == {"flops": 0.5, "depth": 0.5}

    b3 = b1 * 2.0  # rescale all member weights
    assert [w for _, w in b3.terms] == [1.0]

    b4 = b1 + CompiledCriterion()
    assert len(b4.terms) == 2
    assert isinstance(b4.terms[1][0], CompiledCriterion)
    assert b4.terms[1][1] == 1.0

    total = sum([FlopsCriterion() * 0.5, DepthCriterion() * 0.5])
    assert isinstance(total, Blend)
    assert criteria_cost(total).criteria == {
        "flops": 0.5,
        "depth": 0.5,
    }


def test_composition_with_duck_criterion():
    """A criterion lacking the operators still composes on the right:
    duck + builtin hits the builtin's __radd__."""
    duck = _NamedCriterion()
    b = duck + FlopsCriterion()
    assert isinstance(b, Blend)
    assert b.terms[0] == (duck, 1.0)
    assert isinstance(b.terms[1][0], FlopsCriterion)


def test_criteria_container_flattens_and_iterates():
    """Criteria(crit, (crit, w), nested) flattens weighted sub-
    containers; len()/iter expose the member terms."""
    inner = Criteria(FlopsCriterion(), (DepthCriterion(), 3.0))
    outer = Criteria(LatencyCriterion(), (inner, 2.0))
    assert len(outer) == 3
    assert [name for name, _ in ((c.name, w) for c, w in outer)] == [
        "latency",
        "flops",
        "depth",
    ]
    assert [w for _, w in outer.terms] == [1.0, 2.0, 6.0]
    fn = outer.blend()
    assert fn.criteria == pytest.approx(
        {"latency": 1 / 9, "flops": 2 / 9, "depth": 6 / 9}
    )
    # cost_fn is the port spelling of blend()
    assert outer.cost_fn()(
        Op.make("mul", _v("x", 4, 4), _v("y", 4, 4))
    ) == pytest.approx(fn(Op.make("mul", _v("x", 4, 4), _v("y", 4, 4))))


def test_blend_zero_weight_and_empty_members():
    """Zero-weight members price nothing; an empty container has no
    positive axis."""
    c = criteria_cost([FlopsCriterion(), (DepthCriterion(), 0.0)])
    assert c.criteria == {"flops": 1.0}
    with pytest.raises(ValueError, match="positive weight"):
        Criteria().blend()
    with pytest.raises(ValueError, match="positive weight"):
        criteria_cost(Blend())
    with pytest.raises(ValueError, match="positive weight"):
        criteria_cost([(FlopsCriterion(), 0.0)])


def test_same_name_members_merge_in_record():
    """Two members sharing a name merge in the blend's criteria
    record (weights sum) — the priced parts stay distinct."""
    c = criteria_cost([FlopsCriterion(), FlopsCriterion()])
    assert c.criteria == {"flops": 1.0}
    t = Op.make("mul", _v("x", 4, 4), _v("y", 4, 4))
    assert c(t) == pytest.approx(flops_cost(t))


def test_member_spec_errors():
    """Non-criterion members and malformed pairs fail loud."""
    with pytest.raises(TypeError, match="not a Criterion"):
        criteria_cost([42])
    with pytest.raises(TypeError, match="not a Criterion"):
        criteria_cost([(42, 1.0)])
    with pytest.raises(TypeError, match="pair"):
        criteria_cost([(FlopsCriterion(), 1.0, DepthCriterion())])
    with pytest.raises(TypeError, match="not a Criterion"):
        Criteria(42)
    with pytest.raises(ValueError, match="non-negative weight"):
        Criteria((FlopsCriterion(), -1.0))
    with pytest.raises(ValueError, match="non-negative weight"):
        FlopsCriterion() * "x"
    with pytest.raises(TypeError, match="unsupported criteria spec"):
        criteria_cost(42)


# ---------------------------------------------------------------------------
#  The memory axis — weights / peak / combined modes
# ---------------------------------------------------------------------------


def test_memory_mode_validation():
    with pytest.raises(ValueError, match="unknown memory mode"):
        MemoryCriterion("cuda")
    assert MemoryCriterion().name == "memory"
    assert MemoryCriterion("peak").name == "memory:peak"
    assert MemoryCriterion("combined").name == "memory:combined"
    assert MemoryCriterion("weights").charges_param_only is True
    assert MemoryCriterion("peak").charges_param_only is False
    assert MemoryCriterion("combined").charges_param_only is True
    assert MemoryCriterion("peak").charges_shape is True


def test_memory_weights_mode_matches_param_bytes():
    t = Op.make("linear", _v("x", 4, 64), _p("W", 64, 64))
    fn = MemoryCriterion("weights").cost_fn(None)
    assert fn(t) == pytest.approx(param_bytes_cost_for()(t))
    fn2 = MemoryCriterion().cost_fn()
    assert fn2(t) == fn(t)


def test_peak_bytes_liveness_model():
    """Peak = max over the post-order schedule of live bytes (fp32):
    inputs live from entry, intermediates die at their last consumer,
    the root stays live to the end."""
    x, y, z = _v("x", 4, 8), _v("y", 4, 8), _v("z", 4, 8)
    n = 32 * 4.0  # fp32 bytes per (4,8) tensor

    # mul(add(x, y), z): step 0 has x+y+z live plus add's output
    # (inputs are born at entry) — 4 tensors; step 1 has
    # add+mul+z — 3.  Peak = 4 tensors.
    t = Op.make("mul", Op.make("add", x, y), z)
    assert peak_bytes_cost(t) == pytest.approx(4 * n)

    # Params are storage, not activation: add(x, p) peaks at
    # x + out — 2 tensors.
    t = Op.make("add", x, _p("w", 4, 8))
    assert peak_bytes_cost(t) == pytest.approx(2 * n)

    # A param-only subtree folds to a materialised weight — storage:
    # matmul(x, matmul(w1, w2)) prices as x + out only.
    t = Op.make(
        "matmul", x, Op.make("matmul", _p("a", 8, 8), _p("b", 8, 8))
    )
    assert peak_bytes_cost(t) == pytest.approx(2 * n)

    # Views own no storage: neg(transpose(x)) peaks at x + out — the
    # transpose aliases x's buffer (live through the neg).
    t = Op.make("neg", Op.make("transpose", x))
    assert peak_bytes_cost(t) == pytest.approx(2 * n)

    # A lone-view root forwards liveness to its base.
    assert peak_bytes_cost(Op.make("transpose", x)) == pytest.approx(n)

    # Shared subterms are one allocation: mul(a, a) with a = add(x,y)
    # — the second identical arg edge hits the "already later" branch
    # of the last-use update.
    a = Op.make("add", x, y)
    t = Op.make("mul", a, a)
    assert peak_bytes_cost(t) == pytest.approx(3 * n)


def test_peak_bytes_leaf_and_degenerate_terms():
    x = _v("x", 4, 8)
    assert peak_bytes_cost(x) == pytest.approx(128.0)
    assert peak_bytes_cost(_p("w", 4, 8)) == 0.0
    assert peak_bytes_cost("not-a-term") == 0.0
    # An entirely param-foldable root: nothing is scheduled.
    t = Op.make("matmul", _p("a", 8, 8), _p("b", 8, 8))
    assert peak_bytes_cost(t) == 0.0
    # A zero-argument op materialises one (scalar-default) buffer.
    assert peak_bytes_cost(Op.make("some_unlisted_op")) == 4.0
    # An op carrying a non-term arg (constant-morphism spelling):
    # the int operand contributes nothing; the consumer's buffer does.
    assert peak_bytes_cost(Op.make("neg", Op.make("eye", 4))) == 4.0


def test_peak_bytes_memo_and_markers():
    """The model carries its markers and memoizes per term."""
    assert peak_bytes_cost.charges_shape is True
    assert peak_bytes_cost.dag_exact is True
    t = Op.make("mul", _v("x", 4, 8), _v("y", 4, 8))
    memo: dict = {}
    first = peak_bytes_cost(t, memo)
    assert ("pk", t) in memo
    assert peak_bytes_cost(t, memo) == first
    # dag_exact: dag_cost returns the root value verbatim
    assert dag_cost(t, peak_bytes_cost) == pytest.approx(first)


def test_memory_peak_and_combined_modes_price():
    x = _v("x", 8, 8)
    t = Op.make(
        "matmul", x, Op.make("matmul", _p("a", 8, 8), _p("b", 8, 8))
    )
    peak = MemoryCriterion("peak").cost_fn()
    assert peak(t) == pytest.approx(peak_bytes_cost(t))
    combined = MemoryCriterion("combined").cost_fn(profile=None)
    assert combined(t) == pytest.approx(
        param_bytes_cost_for()(t) + peak_bytes_cost(t)
    )
    assert combined.charges_param_only is True
    assert combined.charges_shape is True
    assert combined.dag_exact is True
    assert combined.profile is None


def test_memory_modes_in_dict_and_blend():
    """The memory:* spellings reach the dict spec; modes blend with
    other axes by object."""
    c = criteria_cost({"memory:peak": 1.0})
    assert c.criteria == {"memory:peak": 1.0}
    assert c.charges_param_only is False
    c2 = criteria_cost({"memory:combined": 1.0})
    assert c2.criteria == {"memory:combined": 1.0}
    assert c2.charges_param_only is True
    c3 = criteria_cost({"memory:weights": 1.0})
    assert c3.criteria == {"memory": 1.0}
    b = criteria_cost(
        [(MemoryCriterion("peak"), 1.0), (MemoryCriterion(), 1.0)]
    )
    assert b.criteria == {"memory:peak": 0.5, "memory": 0.5}


def test_peak_mode_flips_selection_toward_thin_intermediates():
    """A member keeping more intermediates live loses to a leaner
    one under the peak axis — the axis flops_cost can't see."""
    eg = EGraph()
    x, y, z = _v("x", 4, 8), _v("y", 4, 8), _v("z", 4, 8)
    big = Op.make("mul", Op.make("add", x, y), z)
    thin = Op.make("mul", x, x)
    assert peak_bytes_cost(big) > peak_bytes_cost(thin)
    eb = eg.add_term(big)
    et = eg.add_term(thin)
    eg.union(eb, et)
    eg.rebuild()
    root = eg.find(eb)
    best = eg.extract_best(root, criteria_cost({"memory:peak": 1.0}))
    assert best == thin


def test_peak_bytes_poisons_ill_typed_members():
    """A provably ill-typed intermediate prices at the near-infinite
    sentinel — extraction must never prefer it."""
    x = _v("x", 4, 8)
    bad = Op.make("mul", Op.make("add", x, _v("y", 64, 64)), x)
    assert peak_bytes_cost(bad) >= 1e15


# ---------------------------------------------------------------------------
#  optimize_model(criteria=...) — new spec forms
# ---------------------------------------------------------------------------


def test_optimize_model_accepts_criterion_objects():
    """criteria=<Criterion> and criteria=<Blend> price through the
    same path as the dict spelling; stats records resolved axes."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(
            m, x, verbose=False, criteria=FlopsCriterion()
        )
        assert stats["criteria"] == {"flops": 1.0}
        assert torch.allclose(mod(x), ref, atol=1e-5)
        mod, stats = optimize_model(
            m,
            x,
            verbose=False,
            criteria=LatencyCriterion() * 0.5 + MemoryCriterion() * 0.5,
        )
        assert stats["criteria"] == {"latency": 0.5, "memory": 0.5}
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_criteria_list_and_peak_mode():
    """A list spec and a mode-qualified dict axis both run
    end-to-end."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with torch.no_grad():
        ref = m(x)
        mod, stats = optimize_model(
            m,
            x,
            verbose=False,
            criteria=[FlopsCriterion(), (DepthCriterion(), 1.0)],
        )
        assert stats["criteria"] == {"flops": 0.5, "depth": 0.5}
        assert torch.allclose(mod(x), ref, atol=1e-5)
        mod, stats = optimize_model(
            m, x, verbose=False, criteria={"memory:peak": 1.0}
        )
        assert stats["criteria"] == {"memory:peak": 1.0}
        assert torch.allclose(mod(x), ref, atol=1e-5)


def test_optimize_model_criteria_spec_errors_surface():
    """Bad spec types raise TypeError; bad members raise their own
    errors — validation isn't deferred to extraction."""
    from catopt.optimize import optimize_model

    m = _tiny_model()
    x = torch.rand(4, 8)
    with pytest.raises(TypeError, match="unsupported criteria spec"):
        optimize_model(m, x, verbose=False, criteria=object())
    with pytest.raises(TypeError, match="not a Criterion"):
        optimize_model(m, x, verbose=False, criteria=[object()])
