"""``_EvalPlan`` — the slot-tape evaluator behind ``IRModule``.

Plan 0003 replaces the recursive ``eval_term`` walk for the generic
(uncompiled) path: term DAGs flatten once into ``(out_slot, op_idx,
arg_slots, attrs)`` steps over a per-call ``vals`` list.  These tests
pin the plan-build edges (DAG dedup, every leaf kind, unplannable
terms), the two runners (positional ``run_xs`` / env ``run_env``), the
arity/attrs call-shape branches, the memo contract ``_eval`` and
``ev_factory`` keep with caller-seeded dicts, and exact equality with
the recursive evaluator on a realistic term.
"""

import pytest
import torch

import catopt_carriers.xcarrier  # noqa: F401 — carrier torch bindings
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from catopt_core.ops import OpTable
from catopt_torch.torch_bridge import _IR_TO_TORCH, ir_to_torch_module

torch.manual_seed(0)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _mod(root, inputs=None, params=None, pv=None, ops=None):
    inputs = [] if inputs is None else inputs
    ir = IR(
        root=root,
        inputs=inputs,
        input_names={v.name for v in inputs},
        params=params or {},
    )
    return ir_to_torch_module(ir, param_values=pv or {}, ops=ops)


# ---------------------------------------------------------------------------
#  Plan build: leaf kinds, DAG dedup, cache
# ---------------------------------------------------------------------------


def test_plan_builds_and_dedupes_shared_subterms():
    """add(m, m) with a shared interned Op: one plan, one slot for m."""
    x = _v("x", 4, 4)
    m = Op.make("mul", x, x)
    root = Op.make("add", m, m)
    mod = _mod(root, inputs=[x])
    plan = mod._root_plan
    assert plan is not None
    # Var + mul + add → 3 slots, 2 op steps; no duplicated mul.
    assert plan.n_slots == 3 and len(plan.steps) == 2
    xv = torch.randn(4, 4)
    assert torch.equal(mod(xv), xv * xv + xv * xv)


def test_plan_cache_hit_returns_same_object():
    x = _v("x", 4, 4)
    t = Op.make("relu", x)
    mod = _mod(t, inputs=[x])
    again = Op.make("relu", x)
    assert again is t  # interned
    assert mod._plan_for(t) is mod._plan_for(t)
    assert mod._plan_for(t) is mod._root_plan


def test_plan_unplannable_non_term_root():
    """A bare non-term root: forward still raises the strict TypeError
    through the eval_term fallback, and the plan cache records the
    unplannable sentinel."""
    mod = _mod(42)
    assert mod._root_plan is None
    with pytest.raises(TypeError, match="Cannot evaluate term"):
        mod(torch.randn(4, 4))
    # cached: second probe takes the _UNPLANNED branch
    assert mod._plan_for(42) is None


def test_plan_unplannable_nested_metavar():
    """A stray non-term arg deep inside makes the plan unplannable —
    eval_term reports it at eval time."""
    x = _v("x", 4, 4)
    root = Op.make("add", Op.make("mul", x, x, validate=False), x)
    # hand-minted term with a raw-string arg — valid to construct,
    # invalid to evaluate.
    bad = Op("add", (root, "metavar"), {})
    mod = _mod(bad, inputs=[x])
    assert mod._root_plan is None
    with pytest.raises(TypeError, match="Cannot evaluate term"):
        mod(torch.randn(4, 4))


def test_plan_leaf_kinds_const_var_param():
    """Const/Var/Param leaves each take their fill path."""
    x = _v("x", 4)
    W = _p("W", 4)
    root = Op.make("add", Op.make("pow", x, Const(2.0)), W)
    mod = _mod(root, inputs=[x], pv={"W": torch.ones(4)})
    plan = mod._root_plan
    assert len(plan.var_fills) == 1
    assert len(plan.param_fills) == 1
    assert len(plan.const_fills) == 1
    xv = torch.randn(4)
    assert torch.equal(mod(xv), xv.pow(2.0) + 1)


def test_plan_const_uses_calltime_default_dtype():
    """``const_fills`` re-mints ``torch.tensor(v)`` per call — flipping
    the default dtype between build and eval must show through."""
    x = _v("x", 4)
    mod = _mod(Op.make("mul", x, Const(0.5)), inputs=[x])
    xv = torch.randn(4)
    assert mod(xv).dtype == torch.float32
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        assert mod(xv.double()).dtype == torch.float64
    finally:
        torch.set_default_dtype(prev)


def test_plan_param_only_and_const_only_roots():
    """Folded/param-only roots evaluate through fills alone."""
    mod = _mod(_p("W", 3), pv={"W": torch.full((3,), 7.0)})
    # no args at all — the zero-``xs`` leg of run_xs
    assert torch.equal(mod(), torch.full((3,), 7.0))
    modc = _mod(Const(5))
    assert torch.equal(modc(), torch.tensor(5))


def test_plan_var_root_passthrough():
    x = _v("x", 2)
    mod = _mod(x, inputs=[x])
    xv = torch.randn(2)
    assert mod(xv) is xv
    # missing args fall back to x — Var fill positional-miss leg
    mod2 = _mod(_v("y", 2), inputs=[])  # 'y' is not an input
    assert mod2(xv) is xv


def test_plan_var_fallback_and_missing_positional():
    """Vars absent from ``inputs`` and short arg tuples both read x."""
    x, y = _v("x", 2), _v("y", 2)
    mod = _mod(Op.make("add", x, y), inputs=[x, y])
    xv = torch.randn(2)
    # y is inputs[1] but only one arg arrives → y reads xs[0]
    assert torch.equal(mod(xv), xv + xv)
    # explicit second arg wins positionally
    yv = torch.randn(2)
    assert torch.equal(mod(xv, yv), xv + yv)


def test_plan_param_unshaped_randn_fallback():
    x = _v("x", 1, 3)
    mod = _mod(
        Op.make("add", x, Param("W", TensorType((None, 3)))), inputs=[x]
    )
    assert mod(torch.randn(1, 3)).shape == (1, 3)


def test_plan_param_recovers_load_state_dict():
    """param_fills read _param_map each call — in-place value updates
    from load_state_dict are visible to the next forward."""
    x = _v("x", 2)
    mod = _mod(
        Op.make("add", x, _p("W", 2)),
        inputs=[x],
        pv={"W": torch.zeros(2)},
    )
    xv = torch.ones(2)
    assert torch.equal(mod(xv), xv)
    mod.load_state_dict({"W": torch.full((2,), 3.0)})
    assert torch.equal(mod(xv), xv + 3.0)


# ---------------------------------------------------------------------------
#  Call-shape branches (arity × attrs)
# ---------------------------------------------------------------------------


def test_plan_call_shapes():
    """Every arity/attrs branch of the tape loop produces correct ops."""
    x, y, z, w = (_v(n, 4) for n in ("x", "y", "z", "w"))
    a, b, c, d = (
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
    )

    cases = {
        # attrs=None, arity 1/2/3/4
        "neg": Op.make("neg", x),
        "add": Op.make("add", x, y),
        "where": Op.make("where", Op.make("gt", x, Const(0)), x, y),
        "concat4": Op.make("concat", x, y, z, w),
        # attrs present, arity 0/1/2/3/4
        "zeros": Op.make("zeros", shape=(4,)),
        "select": Op.make("select", x, dim=0, index=1),
        "stack2": Op.make("stack", x, y, dim=0),
        "stack3": Op.make("stack", x, y, z, dim=0),
        "stack4": Op.make("stack", x, y, z, w, dim=0),
    }
    xs = (a, b, c, d)
    env = {"self": a, "x": a, "y": b, "z": c, "w": d}
    expect = {
        "neg": -a,
        "add": a + b,
        "where": torch.where(a > 0, a, b),
        "concat4": torch.cat([a, b, c, d]),
        "zeros": torch.zeros(4),
        "select": a.select(0, 1),
        "stack2": torch.stack([a, b]),
        "stack3": torch.stack([a, b, c]),
        "stack4": torch.stack([a, b, c, d]),
    }
    for name, term in cases.items():
        mod = _mod(term, inputs=[x, y, z, w])
        got = mod(*xs)
        assert torch.equal(got, expect[name]), name
        # and through the env runner
        got_env = mod._eval(term, env, a, {})
        assert torch.equal(got_env, expect[name]), name


def test_plan_missing_binding_raises_at_eval():
    """A binding deleted post-construction raises the strict
    ValueError through the tape's per-call resolution."""
    x = _v("x", 4)
    table = OpTable.core()
    mod = _mod(Op.make("neg", x), inputs=[x], ops=table)
    del table.torch_bindings["neg"]
    with pytest.raises(ValueError, match="No torch binding"):
        mod(torch.randn(4))


def test_plan_binding_override_reaches_built_module():
    """Ambient-table overrides after construction hit the next call —
    bindings resolve per call, never baked into the plan."""
    x = _v("x", 4)
    mod = _mod(Op.make("relu", x), inputs=[x])
    calls = []
    orig = _IR_TO_TORCH["relu"]
    _IR_TO_TORCH["relu"] = lambda t, *a, **kw: (
        calls.append(1),
        orig(t),
    )[1]
    try:
        xv = torch.randn(4)
        assert torch.equal(mod(xv), xv.clamp_min(0))
        assert len(calls) == 1
    finally:
        _IR_TO_TORCH["relu"] = orig


def test_plan_tuple_output_op():
    """var_mean returns (var, mean) — a non-tensor step result rides
    the slot tape and forward returns it verbatim."""
    x = _v("x", 3, 4)
    mod = _mod(Op.make("var_mean", x, dim=0), inputs=[x])
    xv = torch.randn(3, 4)
    got = mod(xv)
    want = torch.var_mean(xv, dim=0)
    assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


# ---------------------------------------------------------------------------
#  _eval / _eval_fast memo contract
# ---------------------------------------------------------------------------


def test_eval_empty_memo_plan_path_writes_result():
    x = _v("x", 4)
    t = Op.make("relu", x)
    mod = _mod(t, inputs=[x])
    xv = torch.randn(4)
    env = {"self": xv, "x": xv}
    memo: dict = {}
    out = mod._eval(t, env, xv, memo)
    assert memo[t] is out
    # leaf terms never touch memo (eval_term contract)
    assert mod._eval(x, env, xv, memo) is xv
    assert x not in memo
    # memo=None tolerated: plan path skips the write
    assert torch.equal(mod._eval(t, env, xv, None), out)


def test_eval_seeded_memo_honoured_at_depth():
    """A non-empty memo switches to eval_term: seeds hit at ANY depth,
    not just the term being evaluated."""
    x, y = _v("x", 4), _v("y", 4)
    inner = Op.make("mul", x, x)
    root = Op.make("add", inner, y)
    mod = _mod(root, inputs=[x, y])
    xv, yv = torch.randn(4), torch.randn(4)
    env = {"self": xv, "x": xv, "y": yv}
    seed = torch.full((4,), 9.0)
    memo = {inner: seed}
    out = mod._eval(root, env, xv, memo)
    assert torch.equal(out, seed + yv)
    # a top-level hit returns the stored object
    memo2 = {root: seed}
    assert mod._eval(root, env, xv, memo2) is seed


def test_eval_seeded_memo_param_default_and_missing_binding():
    """The recursive (seeded-memo) path still drives eval_term's
    param_default and strict-missing-binding legs."""
    x = _v("x", 4)
    mod = _mod(
        Op.make("add", x, Param("W", TensorType((None, 4)))),
        inputs=[x],
    )
    xv = torch.randn(4)
    env = {"self": xv, "x": xv}
    # seeded memo → eval_term: the unshaped Param materialises via
    # _randn_param (shape (1,4) broadcasts over x's (4,))
    out = mod._eval(mod._root, env, xv, {Op.make("neg", x): None})
    assert out.shape == (1, 4)

    table = OpTable.core()
    mod2 = _mod(Op.make("neg", x), inputs=[x], ops=table)
    del table.torch_bindings["neg"]
    memo = {Op.make("add", x, x): None}
    with pytest.raises(ValueError, match="No torch binding"):
        mod2._eval(mod2._root, env, xv, memo)


def test_eval_env_falsy_resolves_vars_to_x():
    x = _v("x", 4)
    mod = _mod(Op.make("relu", x), inputs=[x])
    xv = torch.randn(4)
    # falsy env → every Var resolves to x, like eval_term's
    # ``(var_env or {}).get`` branch
    assert torch.equal(
        mod._eval(mod._root, {}, xv, {}), xv.clamp_min(0)
    )
    assert torch.equal(
        mod._eval_fast(mod._root, {}, xv), xv.clamp_min(0)
    )


def test_eval_fast_unplannable_raises():
    mod = _mod(_v("x", 4), inputs=[_v("x", 4)])
    with pytest.raises(TypeError, match="Cannot evaluate term"):
        mod._eval_fast("not-a-term", {}, None)


# ---------------------------------------------------------------------------
#  ev_factory memo seeding (the omd carrier contract)
# ---------------------------------------------------------------------------


def _scan_mod():
    import catopt_carriers.scan_lower as sl  # noqa: F401
    from catopt_carriers.scan_lower import to_batched_scan_module

    x = _v("x", 4, 3)
    leaves = [
        Op.make(
            "aff_diag",
            _p("a", 3),
            Op.make("select", x, dim=0, index=t),
        )
        for t in range(4)
    ]

    def comp(ls):
        if len(ls) == 1:
            return ls[0]
        mid = len(ls) // 2
        return Op.make("affd_compose", comp(ls[:mid]), comp(ls[mid:]))

    root = Op.make("applyd", comp(leaves), _p("h", 3))
    ir = IR(root=root, inputs=[x], input_names={"x"}, params={})
    pv = {"a": torch.randn(3) * 0.5, "h": torch.randn(3)}
    return to_batched_scan_module(ir, param_values=pv)


def test_ev_factory_seeded_memo_uses_recursive_path():
    """Foreign keys in the caller's memo (omd-style seeds) force the
    full eval_term walk — inner hits honour seeds at depth."""
    mod = _scan_mod()
    xv = torch.randn(4, 3)
    x, env = mod._input_env((xv,))
    leaf = Op.make("select", _v("x", 4, 3), dim=0, index=1)
    inner = Op.make("relu", leaf)
    seed = torch.full((3,), 4.0)
    memo = {inner: seed}
    ev = mod.ev_factory(env, x, memo)
    out = ev(Op.make("add", inner, leaf))
    assert torch.equal(out, seed + xv[1])


def test_ev_factory_none_memo():
    """A None memo gets a throwaway dict — no crash, fast path."""
    mod = _scan_mod()
    xv = torch.randn(4, 3)
    x, env = mod._input_env((xv,))
    ev = mod.ev_factory(env, x, None)
    leaf = Op.make("select", _v("x", 4, 3), dim=0, index=2)
    assert torch.equal(ev(leaf), xv[2])


def test_ev_factory_memo_roundtrip():
    """Op results memoise under the term; a second call returns the
    same object; non-Op leaves bypass memo entirely."""
    mod = _scan_mod()
    xv = torch.randn(4, 3)
    x, env = mod._input_env((xv,))
    memo: dict = {}
    ev = mod.ev_factory(env, x, memo)
    inp = mod._inputs[0]
    assert ev(inp) is xv
    t = Op.make("relu", inp)
    out = ev(t)
    assert memo[t] is out
    assert ev(t) is out


# ---------------------------------------------------------------------------
#  End-to-end equivalence with the recursive evaluator
# ---------------------------------------------------------------------------


def test_plan_matches_eval_term_on_realistic_term():
    """A dense little DAG (params, consts, multi-arity ops, a tuple
    output) agrees elementwise with eval_term and stays stable over
    repeated calls."""
    x = _v("x", 8, 4)
    W, b = _p("W", 4, 4), _p("b", 4)
    t = Op.make(
        "add",
        Op.make(
            "silu",
            Op.make(
                "add",
                Op.make("matmul", Op.make("tanh", x), W),
                Op.make("mul", x, Const(0.5)),
            ),
        ),
        Op.make(
            "broadcast_to", Op.make("add", b, Const(1.0)), shape=(8, 4)
        ),
    )
    mod = _mod(
        t,
        inputs=[x],
        pv={"W": torch.randn(4, 4) * 0.2, "b": torch.randn(4)},
    )
    xv = torch.randn(8, 4)
    env = {"self": xv, "x": xv}
    # the param-only ``add(b, 1)`` folded to a fused_* Param — eval
    # the FOLDED root (what forward runs) recursively as reference;
    # the dummy seed forces the eval_term path.
    folded = mod._root
    want = mod._eval(folded, env, xv, {Op.make("neg", x): None})
    got = mod(xv)
    assert torch.equal(got, want)
    for _ in range(3):
        assert torch.equal(mod(xv), got)
    # and _eval leaf calls agree with forward, bit for bit
    assert torch.equal(
        mod._eval(folded.args[0], env, xv, {}),
        torch.nn.functional.silu(
            torch.tanh(xv) @ mod._param_map["W"] + xv * 0.5
        ),
    )
