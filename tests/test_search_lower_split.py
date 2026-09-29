"""The search/lower API seam — plan 0006.

``search`` produces a :class:`SearchResult` (IR + saturated e-graph +
extracted term + search-record stats); ``lower`` consumes it and
returns a :class:`LowerResult` (module + merged stats + verify
report) that unpacks as the historical ``(module, stats)`` tuple.
``Optimizer`` is the configured entry object — required ``source`` /
``sink`` ports, no assumed backend — and ``optimize`` composes the
phases through a :class:`~catopt_core.ports.Strategy`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_core.pipeline import LowerResult, SearchResult
from catopt_core.ports import Capabilities, Sink, Strategy
from catopt_optimize import (
    Autotuned,
    Compositional,
    LatencyCriterion,
    MemoryCriterion,
    Monolithic,
    OptimizationResourceError,
    Optimizer,
    discover_alternatives,
    lower,
    optimize_compositional,
    optimize_model,
    optimize_model_autotuned,
    search,
)
from catopt_optimize.runners import (
    ChainedRunner,
    CudaGraphRunner,
    TorchCompileRunner,
)
from catopt_torch.adapters import TorchSink, TorchSource

from tests.test_pluggable_sink import NumpySink


class _MLP(nn.Module):
    """relu(linear(x)) — small enough to saturate in a few iters."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.w(x))


class _Stack(nn.Module):
    """Two stacked blocks — the compositional strategy's shape."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [nn.Linear(dim, dim), nn.Linear(dim, dim)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = torch.relu(b(x))
        return x


def _make():
    torch.manual_seed(0)
    return _MLP().eval(), torch.randn(2, 8)


def _opt() -> Optimizer:
    return Optimizer(source=TorchSource(), sink=TorchSink())


# ------------------------------------------------------------------
#  Port conformance — Capabilities / Sink / Strategy
# ------------------------------------------------------------------


def test_capabilities_sink_strategy_conformance():
    sink = TorchSink()
    assert isinstance(sink, Capabilities)
    assert isinstance(sink, Sink)
    assert isinstance(NumpySink(), Capabilities)
    assert isinstance(NumpySink(), Sink)
    # Capabilities alone is not a Sink — lower/verify are absent.
    class OnlyCaps:
        @property
        def supported_ops(self):
            return frozenset()

        @property
        def ops(self):
            return None

    assert isinstance(OnlyCaps(), Capabilities)
    assert not isinstance(OnlyCaps(), Sink)
    for st in (Monolithic(), Compositional(), Autotuned()):
        assert isinstance(st, Strategy)
    assert not isinstance(object(), Strategy)


# ------------------------------------------------------------------
#  search → SearchResult
# ------------------------------------------------------------------


def test_search_returns_search_result():
    m, x = _make()
    res = search(
        m, x, source=TorchSource(), capabilities=TorchSink(),
        max_iterations=3,
    )
    assert isinstance(res, SearchResult)
    assert isinstance(res.ir, IR)
    assert isinstance(res.eg, EGraph)
    assert isinstance(res.root_eid, int)
    assert res.term is not None
    assert res.stats["rule_fires"] is not None
    assert res.source is not None and res.model is m
    assert callable(res.cost_fn)


def test_search_without_capabilities_is_backend_agnostic():
    """No capabilities → no backend_cost wrap, no causal fold — the
    search still runs and extracts a term."""
    m, x = _make()
    res = search(m, x, source=TorchSource(), max_iterations=3)
    assert isinstance(res, SearchResult)
    assert "causal_specialized" not in res.stats


def test_specialize_causal_opt_out():
    """specialize_causal=False skips the fold even when the sdpa +
    causal-mask shape is present."""
    from catopt.models import EagerAttention

    torch.manual_seed(0)
    m = EagerAttention(dim=64, n_heads=2, block_size=16).eval()
    x = torch.randn(1, 8, 64)
    on = search(
        m, x, source=TorchSource(), capabilities=TorchSink(),
        ruleset="categorical", max_iterations=4,
    )
    assert on.stats.get("causal_specialized") is True
    off = search(
        m, x, source=TorchSource(), capabilities=TorchSink(),
        ruleset="categorical", max_iterations=4,
        specialize_causal=False,
    )
    assert "causal_specialized" not in off.stats


def test_search_requires_source():
    m, x = _make()
    with pytest.raises(TypeError):
        search(m, x, max_iterations=1)  # type: ignore[call-arg]


# ------------------------------------------------------------------
#  SearchResult methods — alternatives / certificate
# ------------------------------------------------------------------


def test_search_result_alternatives_and_certificate():
    m, x = _make()
    res = search(
        m, x, source=TorchSource(), capabilities=TorchSink(),
        max_iterations=3,
    )
    alts = res.alternatives(top_k=4)
    assert isinstance(alts, list) and alts
    for cost, term in alts:
        assert isinstance(cost, float) and term is not None
    # explicit cost_fn override
    from catopt_core.cost import flops_cost

    alts2 = res.alternatives(top_k=2, cost_fn=flops_cost)
    assert isinstance(alts2, list)
    # certificate: extracted term is provably equal to the export
    cert = res.certificate(res.ir.root, res.term)
    assert cert.src is res.ir.root


def test_search_result_flops_fallback():
    """A hand-built SearchResult with no cost_fn prices alternatives
    under flops_cost."""
    from catopt_core.egraph import EGraph as _EG

    x = Var("x", TensorType((3,)))
    w = Param(name="p_w", typ=TensorType((3,)))
    term = Op.make("add", x, w)
    eg = _EG()
    eid = eg.add_term(term)
    res = SearchResult(
        ir=IR(root=term, inputs=[x], input_names={"x"}, params={}),
        eg=eg,
        root_eid=eid,
        term=term,
        param_values={},
        stats={},
    )
    alts = res.alternatives()
    assert alts == [] or isinstance(alts[0][0], float)
    assert res.model is None and res.source is None


# ------------------------------------------------------------------
#  lower → LowerResult; unpacking; stats merge
# ------------------------------------------------------------------


def test_lower_returns_lower_result_and_unpacks():
    m, x = _make()
    res = search(
        m, x, source=TorchSource(), capabilities=TorchSink(),
        max_iterations=3,
    )
    lr = lower(res, x, sink=TorchSink())
    assert isinstance(lr, LowerResult)
    assert lr.verified is not None and lr.verified.passed
    # LowerResult.stats is search ∪ lower — a fresh dict
    for k in res.stats:
        assert lr.stats[k] == res.stats[k]
    assert lr.stats["lowering"] == "generic"
    assert lr.stats["runner"] == "identity"
    assert res.stats is not lr.stats
    assert "runner" not in res.stats  # search record untouched
    mod, stats = lr  # __iter__ unpacks like the legacy tuple
    assert mod is lr.module and stats is lr.stats


def test_lower_verify_off_and_silent():
    m, x = _make()
    res = search(
        m, x, source=TorchSource(), capabilities=TorchSink(),
        max_iterations=3,
    )
    lr = lower(res, x, sink=TorchSink(), verify=False)
    assert lr.verified is None
    lr2 = lower(res, x, sink=TorchSink(), verify=True, verbose=False)
    assert lr2.verified is not None and lr2.verified.passed


def test_lower_verifies_hand_built_result():
    """A hand-assembled SearchResult (no e-graph history, no model)
    lowers and verifies — the reference is the sink's own lowering of
    the result's IR."""
    x = Var("x", TensorType((3,)))
    w = Param(name="p_w", typ=TensorType((3,)))
    term = Op.make("add", x, w)
    eg = EGraph()
    eid = eg.add_term(term)
    res = SearchResult(
        ir=IR(
            root=term,
            inputs=[x],
            input_names={"x"},
            params={"p_w": TensorType((3,))},
        ),
        eg=eg,
        root_eid=eid,
        term=term,
        param_values={"p_w": torch.randn(3)},
        stats={},
        model=None,
    )
    lr = lower(res, torch.randn(3), sink=TorchSink())
    assert lr.verified is not None and lr.verified.passed


def test_lower_runner_applied_and_recorded():
    m, x = _make()
    res = search(
        m, x, source=TorchSource(), capabilities=TorchSink(),
        max_iterations=3,
    )
    lr = lower(res, x, sink=TorchSink(), runner=TorchCompileRunner())
    assert lr.stats["runner"] == "torch_compile"
    assert lr.stats["compiled"] is True


def test_one_search_many_deliveries():
    """The seam's raison d'être: two runners, one search."""
    m, x = _make()
    res = search(
        m, x, source=TorchSource(), capabilities=TorchSink(),
        max_iterations=3,
    )
    eager = lower(res, x, sink=TorchSink())
    fast = lower(res, x, sink=TorchSink(), runner=TorchCompileRunner())
    assert eager.stats["runner"] == "identity"
    assert fast.stats["runner"] == "torch_compile"
    with torch.no_grad():
        assert torch.allclose(eager.module(x), m(x), atol=1e-5)
        assert torch.allclose(fast.module(x), m(x), atol=1e-5)


def test_lower_requires_sink():
    m, x = _make()
    res = search(m, x, source=TorchSource(), max_iterations=1)
    with pytest.raises(TypeError):
        lower(res, x)  # type: ignore[call-arg]


# ------------------------------------------------------------------
#  Non-torch sink flows through both verbs
# ------------------------------------------------------------------


def test_numpy_sink_flows_through_search_and_lower():
    """NumpySink is the forcing evidence the seam is real: the search
    prices against its supported_ops and lower delivers a numpy
    runnable — verified in the sink's own runtime."""
    from tests.test_pluggable_sink import _model

    model = _model()
    x = torch.randn(2, 8)
    res = search(
        model,
        x,
        source=TorchSource(),
        capabilities=NumpySink(),
        max_iterations=2,
    )
    lr = lower(res, x, sink=NumpySink())
    got = lr.module(x.detach().numpy())
    want = model(x).detach().numpy()
    assert np.allclose(got, want, atol=1e-4)
    assert lr.verified is not None and lr.verified.passed


# ------------------------------------------------------------------
#  Optimizer
# ------------------------------------------------------------------


def test_optimizer_requires_ports():
    with pytest.raises(TypeError):
        Optimizer()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Optimizer(source=TorchSource())  # type: ignore[call-arg]


def test_optimizer_optimize_default_is_monolithic():
    m, x = _make()
    mod, stats = _opt().optimize(m, x, max_iterations=3)
    assert stats["runner"] == "identity"
    assert stats["lowering"] == "generic"
    with torch.no_grad():
        assert torch.allclose(mod(x), m(x), atol=1e-5)


def test_optimizer_search_lower_roundtrip():
    m, x = _make()
    opt = _opt()
    res = opt.search(m, x, max_iterations=3)
    assert isinstance(res, SearchResult)
    lr = opt.lower(res, x)
    assert isinstance(lr, LowerResult)
    assert lr.stats["runner"] == "identity"


def test_optimizer_runner_field_is_default_delivery():
    m, x = _make()
    opt = Optimizer(
        source=TorchSource(),
        sink=TorchSink(),
        runner=TorchCompileRunner(),
    )
    mod, stats = opt.optimize(m, x, max_iterations=3)
    assert stats["runner"] == "torch_compile"
    assert stats["compiled"] is True


def test_optimizer_criteria_field_prices_search():
    m, x = _make()
    opt = Optimizer(
        source=TorchSource(),
        sink=TorchSink(),
        criteria=LatencyCriterion() * 0.7
        + MemoryCriterion("peak") * 0.3,
    )
    _, stats = opt.optimize(m, x, max_iterations=3)
    assert stats["criteria"] is not None


def test_optimizer_discover():
    m, x = _make()
    res = _opt().discover(m, x, max_iterations=2)
    assert isinstance(res, SearchResult)
    assert isinstance(res.alternatives(4), list)


def test_optimize_unknown_kwarg_raises():
    m, x = _make()
    with pytest.raises(TypeError, match="unexpected keywords"):
        _opt().optimize(m, x, bogus_kw=1)


def test_optimize_call_time_runner_overrides_and_hints_search():
    """optimize(..., runner=...) is a delivery override AND a search
    hint (delivers_compiled) — byte-parity with the old runner=
    parameter."""
    m, x = _make()
    mod, stats = _opt().optimize(
        m, x, max_iterations=3, runner=TorchCompileRunner()
    )
    assert stats["runner"] == "torch_compile"
    assert stats["compiled"] is True


def test_custom_strategy_runs_through_optimizer():
    """A user strategy object — name + run — plugs into optimize."""
    m, x = _make()
    seen: dict = {}

    class Halving:
        name = "halving"

        def run(self, model, x, *, optimizer, **kw):
            seen["ran"] = True
            res = optimizer.search(model, x, max_iterations=1)
            return optimizer.lower(res, x, verify=False)

    assert isinstance(Halving(), Strategy)
    mod, stats = _opt().optimize(m, x, strategy=Halving())
    assert seen["ran"] is True
    assert stats["runner"] == "identity"


def test_optimize_strategy_compositional():
    m = _Stack().eval()
    x = torch.randn(2, 8)
    lr = _opt().optimize(
        m,
        x,
        strategy=Compositional(max_cross_pairs=0),
        max_iterations=3,
        verbose=False,
    )
    assert isinstance(lr, LowerResult)
    assert lr.stats["n_blocks"] == 2
    with torch.no_grad():
        assert torch.allclose(lr.module(x), m(x), atol=1e-4)


def test_optimize_strategy_autotuned():
    m, x = _make()
    lr = _opt().optimize(
        m,
        x,
        strategy=Autotuned(candidates=("generic",), n_calls=2,
                           warmup=0),
        max_iterations=3,
        verbose=False,
    )
    assert isinstance(lr, LowerResult)
    assert lr.stats["autotune"]["winner"] == "generic"


# ------------------------------------------------------------------
#  Wrappers reproduce the old output
# ------------------------------------------------------------------


def test_optimize_model_wrapper_parity():
    m, x = _make()
    mod, stats = optimize_model(m, x, verbose=False, max_iterations=3)
    assert stats["runner"] == "identity"
    assert stats["lowering"] == "generic"
    assert "rule_fires" in stats and "compiled" not in stats
    with torch.no_grad():
        assert torch.allclose(mod(x), m(x), atol=1e-5)


def test_optimize_model_verbose_verifies_and_prints(capsys):
    m, x = _make()
    optimize_model(m, x, verbose=True, max_iterations=1)
    out = capsys.readouterr().out
    assert "[Verify] Checking output equivalence..." in out
    assert "Semantically equivalent" in out


def test_wrappers_chained_runner_names():
    m, x = _make()
    _, stats = optimize_model(
        m,
        x,
        verbose=False,
        runner=ChainedRunner([TorchCompileRunner(), CudaGraphRunner()]),
        max_iterations=3,
    )
    assert stats["runner"] == ["torch_compile", "cuda_graph"]
    assert stats["compiled"] is True


def test_optimize_compositional_wrapper_parity():
    torch.manual_seed(0)
    m = _Stack().eval()
    x = torch.randn(2, 8)
    mod, stats = optimize_compositional(
        m, x, max_iterations=3, max_cross_pairs=0, verbose=False
    )
    assert stats["n_blocks"] == 2
    with torch.no_grad():
        assert torch.allclose(mod(x), m(x), atol=1e-4)


def test_optimize_model_autotuned_wrapper_parity():
    m, x = _make()
    mod, stats = optimize_model_autotuned(
        m,
        x,
        candidates=("generic",),
        n_calls=2,
        warmup=0,
        max_iterations=3,
        verbose=False,
    )
    assert stats["autotune"]["winner"] == "generic"
    with torch.no_grad():
        assert torch.allclose(mod(x), m(x), atol=1e-5)


def test_optimize_model_resource_error_still_raises():
    """The OOM adapter still wraps the wrapper."""
    m, x = _make()
    with pytest.raises(OptimizationResourceError):
        optimize_model(
            m, x, verbose=False, max_iterations=3, max_enodes=1
        )


def test_discover_alternatives_returns_search_result():
    m, x = _make()
    res = discover_alternatives(
        m, x, source=TorchSource(), max_iterations=2
    )
    assert isinstance(res, SearchResult)
    assert isinstance(res.alternatives(4), list)
    assert res.eg is not None
