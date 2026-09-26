# ruff: noqa: RUF002, RUF003
"""Coverage tests for catopt.optimize — resource bounds, the OOM
adapter, causal-mask specialization, discovery/eps entry points, the
parameter-diff report, and the compositional driver."""

import json
import types

import pytest
import torch
import torch.nn as nn

from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.optimize import (
    OptimizationResourceError,
    _check_resources,
    _current_memory_mb,
    _eval_const,
    _is_causal_keep_mask,
    _looks_like_oom,
    _oom_to_resource_error,
    _rel_diff,
    _select_blocks,
    _specialize_causal,
    discover_alternatives,
    ir_to_string,
    optimize_compositional,
    optimize_model,
    param_report,
    save_optimized_weights,
    term_cost,
)
from catopt.cost import flops_cost, launch_aware_cost


def _T(*shape):
    return TensorType(tuple(shape))


def _small_mlp(seed=0):
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(16, 16, bias=False)
            s.l2 = nn.Linear(16, 8, bias=False)

        def forward(s, x):
            return s.l2(torch.relu(s.l1(x)))

    return M().eval().double()


# ---------------------------------------------------------------------------
#  OOM detection + the resource-error adapter
# ---------------------------------------------------------------------------


def test_looks_like_oom_variants():
    assert _looks_like_oom(MemoryError("host oom"))
    assert _looks_like_oom(torch.cuda.OutOfMemoryError("cuda oom"))
    assert _looks_like_oom(RuntimeError("CUDA out of memory"))
    assert _looks_like_oom(RuntimeError("can't allocate memory"))
    assert _looks_like_oom(RuntimeError("cannot allocate memory"))
    assert not _looks_like_oom(RuntimeError("shape mismatch"))
    assert not _looks_like_oom(ValueError("out of memory in message"))
    assert not _looks_like_oom(Exception("anything"))


def test_oom_to_resource_error_adapter():
    @_oom_to_resource_error
    def raises(e):
        raise e

    with pytest.raises(OptimizationResourceError):
        raises(MemoryError("boom"))
    with pytest.raises(OptimizationResourceError):
        raises(RuntimeError("CUDA out of memory"))
    # a pre-existing resource error passes through unchanged
    marker = OptimizationResourceError("already")
    with pytest.raises(OptimizationResourceError) as ei:
        raises(marker)
    assert ei.value is marker
    # ordinary errors are not wrapped
    with pytest.raises(ValueError):
        raises(ValueError("plain"))
    # functools.wraps keeps the signature metadata
    assert raises.__name__ == "raises"


def test_current_memory_mb_and_check_resources():
    mb = _current_memory_mb()
    assert isinstance(mb, float) and mb > 0

    eg = types.SimpleNamespace(n_enodes=50)
    _check_resources(eg, max_enodes=100, max_memory_mb=None)
    with pytest.raises(OptimizationResourceError, match="e-nodes"):
        _check_resources(eg, max_enodes=50, max_memory_mb=None)
    with pytest.raises(OptimizationResourceError, match="e-nodes"):
        _check_resources(eg, max_enodes=10, max_memory_mb=None)
    # a memory bound below the real footprint always trips
    with pytest.raises(OptimizationResourceError, match="memory"):
        _check_resources(eg, max_enodes=None, max_memory_mb=0.0)
    # eg=None skips the enode check entirely
    _check_resources(None, max_enodes=1, max_memory_mb=None)
    _check_resources(eg, max_enodes=None, max_memory_mb=None)


def test_optimize_model_resource_bounds():
    m = _small_mlp()
    x = torch.randn(2, 16, dtype=torch.float64)
    with pytest.raises(OptimizationResourceError):
        optimize_model(
            m, x, max_enodes=1, verbose=False
        )
    with pytest.raises(OptimizationResourceError):
        optimize_model(
            m, x, max_memory_mb=0.0, verbose=False
        )


# ---------------------------------------------------------------------------
#  Const eval + causal-mask specialization
# ---------------------------------------------------------------------------


def test_eval_const_paths():
    w = Param("w", _T(2, 2))
    env = {"w": torch.ones(2, 2)}
    assert torch.equal(_eval_const(w, env), torch.ones(2, 2))
    assert float(_eval_const(Const(3.0), env)) == 3.0
    assert _eval_const(Var("x", _T(2, 2)), env) is None
    assert _eval_const(Param("gone", _T(2)), env) is None
    # op without binding → None
    assert _eval_const(Op.make("bogus_op", w), env) is None
    # op that raises inside torch → None
    bad = Op.make("reshape", w, shape=(3, 3))
    assert _eval_const(bad, env) is None
    # composite op on params → concrete tensor
    ok = Op.make("mul", w, Const(2.0))
    assert torch.equal(_eval_const(ok, env), torch.full((2, 2), 2.0))
    # non-term input → None
    assert _eval_const(object(), env) is None


def test_is_causal_keep_mask():
    T = 4
    tril = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert _is_causal_keep_mask(tril, (2, T, T))
    # additive -inf mask keeps the same set
    add = torch.zeros(T, T)
    add.masked_fill_(~tril, float("-inf"))
    assert _is_causal_keep_mask(add, (2, T, T))
    # non-tuple / too-short q shape
    assert not _is_causal_keep_mask(tril, None)
    assert not _is_causal_keep_mask(tril, (T,))
    # non-square mask
    assert not _is_causal_keep_mask(
        torch.ones(T, T + 1, dtype=torch.bool), (2, T, T + 1)
    )
    # mask T mismatch vs q's seq dim
    assert not _is_causal_keep_mask(tril, (2, T + 1, T + 1))
    # non-causal keep pattern
    rand = torch.rand(T, T) > 0.5
    assert not _is_causal_keep_mask(rand, (2, T, T))
    # 1-D mask
    assert not _is_causal_keep_mask(torch.ones(T, dtype=torch.bool), (2, T, T))


def test_specialize_causal_drops_param_tril_mask():
    q = Var("q", _T(1, 4, 4))
    k = Var("k", _T(1, 4, 4))
    v = Var("v", _T(1, 4, 4))
    mp = Param("mask", _T(4, 4))
    tril = torch.tril(torch.ones(4, 4, dtype=torch.bool))
    env = {"mask": tril}
    sdpa = Op.make("sdpa", q, k, v, mp)
    out = _specialize_causal(sdpa, env)
    assert out.attrs.get("arg5") is True
    assert len(out.args) == 3
    # already-causal → unchanged arg count
    sdpa2 = Op.make("sdpa", q, k, v, mp, arg5=True)
    out2 = _specialize_causal(sdpa2, env)
    assert len(out2.args) == 4
    # non-causal mask → unchanged
    env2 = {"mask": torch.ones(4, 4, dtype=torch.bool)}
    out3 = _specialize_causal(sdpa, env2)
    assert len(out3.args) == 4 and not out3.attrs.get("arg5")
    # mask that's a Var (runtime input) → can't specialize
    mv = Var("m", _T(4, 4))
    sdpa3 = Op.make("sdpa", q, k, v, mv)
    out4 = _specialize_causal(sdpa3, env)
    assert len(out4.args) == 4
    # non-Op input passes through; memo hit returns the same object
    leaf = Var("x", _T(2))
    assert _specialize_causal(leaf, env) is leaf
    memo = {}
    o1 = _specialize_causal(sdpa, env, memo)
    o2 = _specialize_causal(sdpa, env, memo)
    assert o2 is o1


# ---------------------------------------------------------------------------
#  optimize_model entry points
# ---------------------------------------------------------------------------


def test_optimize_model_rulesets_and_equivalence():
    m = _small_mlp()
    x = torch.randn(2, 16, dtype=torch.float64)
    for ruleset in ("all", "simpl", "categorical"):
        opt, stats = optimize_model(
            m, x, ruleset=ruleset, verbose=False, max_iterations=3
        )
        with torch.no_grad():
            assert (
                m(x) - opt(x)
            ).abs().max().item() < 1e-6, ruleset
        assert "rule_fires" in stats
    with pytest.raises(ValueError, match="ruleset"):
        optimize_model(m, x, ruleset="nonsense", verbose=False)


def test_optimize_model_cost_fn_and_eps_optin():
    m = _small_mlp()
    x = torch.randn(2, 16, dtype=torch.float64)
    opt, stats = optimize_model(
        m, x, cost_fn=flops_cost, verbose=False, max_iterations=3
    )
    assert opt is not None
    # eps_rtol opts into certified-approximation offers
    opt2, stats2 = optimize_model(
        m, x, eps_rtol=0.9, verbose=False, max_iterations=3
    )
    assert "eps_offers" in stats2
    assert isinstance(stats2["eps_offers"], list)
    assert len(stats2["eps_offers"]) >= 1
    # the lowered module runs; whether the eps member won extraction is
    # cost-model-dependent (an approximate win is still certified, not
    # a silent accuracy trade — it is recorded in eps_offers)
    with torch.no_grad():
        out = opt2(x)
    assert out.shape == m(x).shape and torch.isfinite(out).all()


def test_optimize_model_tuple_input():
    torch.manual_seed(0)

    class TwoIn(nn.Module):
        def __init__(s):
            super().__init__()
            s.w = nn.Linear(8, 8, bias=False)

        def forward(s, a, b):
            return s.w(a + b)

    m = TwoIn().eval().double()
    a = torch.randn(2, 8, dtype=torch.float64)
    b = torch.randn(2, 8, dtype=torch.float64)
    opt, stats = optimize_model(
        m, (a, b), verbose=False, max_iterations=3
    )
    with torch.no_grad():
        ref = m(a, b)
        got = opt(a, b)
    assert (ref - got).abs().max().item() < 1e-6


def test_discover_alternatives():
    m = _small_mlp()
    x = torch.randn(2, 16, dtype=torch.float64)
    res = discover_alternatives(
        m, x, ruleset="all", max_iterations=2, top_k=4
    )
    assert "alternatives" in res and "rule_fires" in res
    assert "diverse_classes" in res and "stats" in res
    assert isinstance(res["alternatives"], list)
    res2 = discover_alternatives(
        m, x, ruleset="simpl", max_iterations=2, top_k=2
    )
    assert isinstance(res2["alternatives"], list)
    res3 = discover_alternatives(
        m,
        x,
        ruleset="categorical",
        max_iterations=2,
        cost_fn=launch_aware_cost,
    )
    assert isinstance(res3["alternatives"], list)


# ---------------------------------------------------------------------------
#  param_report / save_optimized_weights / ir_to_string / term_cost
# ---------------------------------------------------------------------------


def test_param_report_and_weights_file(tmp_path):
    m = _small_mlp()
    x = torch.randn(2, 16, dtype=torch.float64)
    opt, _stats = optimize_model(m, x, verbose=False, max_iterations=3)
    pr = param_report(m, opt)
    assert pr["original_params"] >= 1
    assert pr["optimized_params"] >= 1
    assert pr["original_bytes"] > 0 and pr["optimized_bytes"] > 0
    assert pr["bytes_saved"] == (
        pr["original_bytes"] - pr["optimized_bytes"]
    )
    assert pr["ratio"] > 0
    assert isinstance(pr["eliminated"], list)
    assert isinstance(pr["derived"], list)

    path = tmp_path / "opt_weights.pt"
    save_optimized_weights(opt, str(path))
    sd = torch.load(path, weights_only=True)
    assert len(sd) == pr["optimized_params"]

    x_v = Var("x", _T(2, 4))
    t = Op.make("relu", x_v)
    s = ir_to_string(t)
    assert "relu" in s
    assert term_cost(t) > 0
    assert term_cost(t, flops_cost) > 0


# ---------------------------------------------------------------------------
#  optimize_compositional
# ---------------------------------------------------------------------------


def test_select_blocks_default_and_custom_pred():
    class Wrap(nn.Module):
        def __init__(s):
            super().__init__()
            s.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            s.head = nn.Linear(4, 4)

    m = Wrap()
    blocks = dict(_select_blocks(m, None))
    # "blocks"/"head" are children of a plain Module → not matched by
    # the default pred and not leaves → descend: the ModuleList's own
    # children DO match (parent is a ModuleList)
    assert "blocks.0" in blocks and "blocks.1" in blocks
    assert "head" in blocks  # leaf child → selected
    assert "blocks" not in blocks  # ModuleList was descended, not taken
    # custom pred that selects the ModuleList itself → opaque, no descent
    blocks2 = dict(
        _select_blocks(
            m, lambda p, n, c: isinstance(c, nn.ModuleList)
        )
    )
    assert "blocks" in blocks2 and "blocks.0" not in blocks2
    assert "head" in blocks2


def test_optimize_compositional_sequential():
    torch.manual_seed(0)
    m = nn.Sequential(
        nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8)
    ).double()
    x = torch.randn(4, 8, dtype=torch.float64)
    new_m, stats = optimize_compositional(
        m, x, verbose=False, max_iterations=3
    )
    assert stats["compositional"]
    assert stats["n_blocks"] >= 1
    assert stats["end_to_end"]["max_abs_diff"] < 1e-6
    pr = stats["param_report"]
    assert pr["original_params"] >= stats["n_optimized"]
    assert "blocks" in stats
    for name, entry in stats["blocks"].items():
        assert entry["status"] in (
            "optimized",
            "failed",
            "skipped",
            "not_executed",
        )
    with torch.no_grad():
        assert (m(x) - new_m(x)).abs().max().item() < 1e-6


def test_optimize_compositional_failed_and_skipped_blocks():
    torch.manual_seed(0)

    class DataDep(nn.Module):
        """Runs eagerly; torch.export fails on the data-dependent
        guard — the per-block failure path keeps the original."""

        def forward(s, x):
            if x.sum() > 0:
                return x * 2
            return x

    class Kw(nn.Module):
        def forward(s, x, scale=1.0):
            return x * scale

    class Unused(nn.Module):
        def forward(s, x):
            return x * 0

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.blocks = nn.ModuleList(
                [nn.Linear(8, 8), DataDep(), Kw(), Unused()]
            )
            s.lin = nn.Linear(8, 8)

        def forward(s, x):
            # blocks[0] and [1] run; [2] is called with a kwarg;
            # [3] never executes
            return s.lin(
                s.blocks[1](s.blocks[0](x))
                + s.blocks[2](x, scale=3.0)
            )

    m = M().eval().double()
    x = torch.randn(2, 8, dtype=torch.float64)
    _new, stats = optimize_compositional(
        m, x, verbose=False, max_iterations=2
    )
    blocks = stats["blocks"]
    assert blocks["blocks.0"]["status"] in ("optimized", "failed")
    # export fails on the data-dependent block → kept original
    assert blocks["blocks.1"]["status"] == "failed"
    assert "error" in blocks["blocks.1"]
    # kwargs capture → skipped honestly
    assert blocks["blocks.2"]["status"] == "skipped"
    assert "non-positional" in blocks["blocks.2"]["reason"]
    # never executed → no captured input
    assert blocks["blocks.3"]["status"] == "not_executed"
    assert stats["n_skipped"] == 2
    assert stats["wall_time_s"] > 0


def test_optimize_compositional_max_enodes_resource_limit():
    """A block crossing max_enodes fails with reason=resource_limit —
    the compositional driver records it as an ordinary failure."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8)).double()
    x = torch.randn(2, 8, dtype=torch.float64)
    _new, stats = optimize_compositional(
        m, x, verbose=False, max_enodes=1, max_iterations=2
    )
    reasons = {
        e.get("reason")
        for e in stats["blocks"].values()
        if e.get("status") == "failed"
    }
    assert "resource_limit" in reasons
    assert stats["n_optimized"] == 0
