"""Delivery runners — ``catopt_orchestrator.runners``.

``optimize_model``'s ``runner=`` parameter is the sole execution
control: each runner class owns one delivery transform
(``torch.compile``, CUDA-graph capture), chains compose
left-to-right, and a custom duck-typed runner plugs straight in.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from catopt_orchestrator.autotune import AutotuneContext, CandidateUnavailableError



from catopt_orchestrator.runners import ChainedRunner, IdentityRunner, Runner, runner_candidate

from catopt_cuda import CudaGraphRunner
from catopt_torch.runners import TorchCompileRunner

from catopt_torch.adapters import TorchSink
from catopt_orchestrator.optimize import Autotuned
from catopt_orchestrator import Optimizer

from catopt_torch.autotune import TORCH_BUILDERS
from catopt_torch.backend import TorchBackend


class _MLP(nn.Module):
    """relu(linear(x)) — small enough to saturate in a few iters."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.w = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.w(x))


class _Capturable(nn.Module):
    """Stand-in batched executor — records capture/drop attempts."""

    def __init__(self) -> None:
        super().__init__()
        self.captured = False
        self.dropped = False

    def capture_cuda_graph(self, *xs: torch.Tensor) -> None:
        self.captured = True

    def drop_cuda_graph(self) -> None:
        self.dropped = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _make():
    torch.manual_seed(0)
    return _MLP().eval(), torch.randn(2, 8)


# ------------------------------------------------------------------
#  Protocol conformance — duck-typed, no isinstance requirement
# ------------------------------------------------------------------


def test_runner_protocol_conformance():
    for r in (
        IdentityRunner(),
        TorchCompileRunner(),
        CudaGraphRunner(),
        ChainedRunner([IdentityRunner()]),
    ):
        assert isinstance(r, Runner)

    class Custom:
        name = "custom"

        def apply(self, module, example_input, stats):
            return module

    assert isinstance(Custom(), Runner)
    assert not isinstance(object(), Runner)


# ------------------------------------------------------------------
#  IdentityRunner — identity delivery
# ------------------------------------------------------------------


def test_identity_runner():
    mod = nn.Linear(4, 4)
    stats: dict = {}
    out = IdentityRunner().apply(mod, torch.randn(2, 4), stats)
    assert out is mod
    assert stats == {}
    assert IdentityRunner().name == "identity"


# ------------------------------------------------------------------
#  TorchCompileRunner — torch.compile delivery
# ------------------------------------------------------------------


def test_torch_compile_runner_wraps_and_records():
    torch.manual_seed(0)
    mod = nn.Sequential(nn.Linear(8, 8), nn.SiLU()).eval()
    x = torch.randn(2, 8)
    with torch.no_grad():
        ref = mod(x)
    stats: dict = {}
    out = TorchCompileRunner().apply(mod, x, stats)
    assert stats["compiled"] is True
    with torch.no_grad():
        assert torch.allclose(out(x), ref, atol=1e-6)


def test_torch_compile_runner_falls_back_on_failure(monkeypatch):
    """A compile failure keeps the uncompiled module — identical to
    the old inline flag block's contract."""

    def boom(model, **kw):
        raise RuntimeError("no compiler")

    monkeypatch.setattr(torch, "compile", boom)
    mod = nn.Linear(4, 4)
    stats: dict = {}
    out = TorchCompileRunner().apply(mod, torch.randn(2, 4), stats)
    assert out is mod
    assert stats["compiled"] is False


def test_torch_compile_runner_passes_compile_kwargs(monkeypatch):
    seen: dict = {}

    def fake_compile(model, **kw):
        seen.update(kw)
        return model

    monkeypatch.setattr(torch, "compile", fake_compile)
    mod = nn.Linear(4, 4)
    stats: dict = {}
    out = TorchCompileRunner(mode="reduce-overhead").apply(
        mod, torch.randn(2, 4), stats
    )
    assert seen == {"mode": "reduce-overhead"}
    assert stats["compiled"] is True
    assert out is mod


# ------------------------------------------------------------------
#  CudaGraphRunner — CUDA-graph capture delivery
# ------------------------------------------------------------------


def test_cuda_graph_runner_noop_on_cpu_input():
    """CPU example input: quiet degrade — stats False, module
    returned unchanged, no capture attempted."""
    mod = _Capturable()
    stats: dict = {}
    out = CudaGraphRunner().apply(mod, torch.randn(2, 4), stats)
    assert out is mod
    assert stats["cuda_graph"] is False
    assert not mod.captured


def test_cuda_graph_runner_noop_without_capture_attr():
    """A plain module has no capture_cuda_graph — degrade quietly."""
    mod = nn.Linear(4, 4)
    stats: dict = {}
    out = CudaGraphRunner().apply(mod, torch.randn(2, 4), stats)
    assert out is mod
    assert stats["cuda_graph"] is False


def test_cuda_graph_runner_tuple_and_nontensor_inputs():
    """Tuple/list example inputs take the first element; non-tensor
    first elements don't crash — both degrade to cuda_graph=False."""
    mod = _Capturable()
    stats: dict = {}
    out = CudaGraphRunner().apply(mod, (torch.randn(2, 4),), stats)
    assert out is mod
    assert stats["cuda_graph"] is False
    stats2: dict = {}
    out = CudaGraphRunner().apply(mod, ("not-a-tensor",), stats2)
    assert out is mod
    assert stats2["cuda_graph"] is False


def test_cuda_graph_runner_skips_when_compiled():
    """The 'compiled wins' precedence lives inside the runner: a
    successful compile earlier in the chain leaves cuda_graph unset
    and capture unattempted."""
    mod = _Capturable()
    stats: dict = {"compiled": True}
    out = CudaGraphRunner().apply(mod, torch.randn(2, 4), stats)
    assert out is mod
    assert "cuda_graph" not in stats
    assert not mod.captured


@pytest.mark.requires_cuda
def test_cuda_graph_runner_captures_on_cuda_input():
    mod = _Capturable()
    stats: dict = {}
    x = torch.randn(2, 4, device="cuda")
    out = CudaGraphRunner().apply(mod, x, stats)
    assert out is mod
    assert stats["cuda_graph"] is True
    assert mod.captured


@pytest.mark.requires_cuda
def test_cuda_graph_runner_drops_on_failed_capture():
    class Failing(_Capturable):
        def capture_cuda_graph(self, *xs: torch.Tensor) -> None:
            raise RuntimeError("capture broke")

    mod = Failing()
    stats: dict = {}
    x = torch.randn(2, 4, device="cuda")
    out = CudaGraphRunner().apply(mod, x, stats)
    assert out is mod
    assert stats["cuda_graph"] is False
    assert mod.dropped


# ------------------------------------------------------------------
#  ChainedRunner — left-to-right composition
# ------------------------------------------------------------------


def test_chained_runner_applies_left_to_right():
    """Each member sees the previous member's output."""
    seen: list = []

    class Wrap:
        def __init__(self, tag: str) -> None:
            self.name = tag

        def apply(self, module, example_input, stats):
            seen.append(module)
            return nn.Sequential(module)

    mod = nn.Linear(2, 2)
    chain = ChainedRunner([Wrap("a"), Wrap("b")])
    out = chain.apply(mod, torch.randn(1, 2), {})
    assert seen[0] is mod
    assert isinstance(seen[1], nn.Sequential)
    assert seen[1][0] is mod
    assert out[0][0] is mod
    assert chain.names == ["a", "b"]
    assert chain.name == "a+b"


def test_chained_runner_empty():
    chain = ChainedRunner([])
    mod = nn.Linear(2, 2)
    assert chain.apply(mod, torch.randn(1, 2), {}) is mod
    assert chain.name == "identity"
    assert chain.names == []
    assert chain.delivers_compiled is False


def test_delivers_compiled_marker_propagates():
    """``delivers_compiled`` is how extraction knows to price the
    fusion-region model — it aggregates through a chain."""
    assert TorchCompileRunner().delivers_compiled is True
    assert (
        ChainedRunner(
            [CudaGraphRunner(), TorchCompileRunner()]
        ).delivers_compiled
        is True
    )
    assert ChainedRunner([CudaGraphRunner()]).delivers_compiled is False
    assert getattr(IdentityRunner(), "delivers_compiled", False) is False


# ------------------------------------------------------------------
#  optimize_model integration — stats["runner"], explicit runners
# ------------------------------------------------------------------


def test_optimize_model_default_runner_is_identity():
    m, x = _make()
    _, stats = Optimizer(backend=TorchBackend()).optimize(m, x, max_iterations=3, verify=False, verbose=False)

    assert stats["runner"] == "identity"
    assert "compiled" not in stats
    assert "cuda_graph" not in stats


def test_removed_flag_kwargs_raise_typeerror():
    """The legacy compile/cuda_graph flags are gone — runner= is the
    only execution control."""
    m, x = _make()
    with pytest.raises(TypeError):
        Optimizer(backend=TorchBackend()).optimize(m, x, compile=True, verify=False, verbose=False)

    with pytest.raises(TypeError):
        Optimizer(backend=TorchBackend()).optimize(m, x, cuda_graph=True, verify=False, verbose=False)



def test_custom_runner_runs_and_records():
    """A user-defined duck-typed runner is applied to the delivered
    module and can write its own stats keys."""
    m, x = _make()

    class Custom:
        name = "custom"

        def apply(self, module, example_input, stats):
            stats["custom_ran"] = True
            return module

    _, stats = Optimizer(backend=TorchBackend(), runner=Custom()).optimize(m, x, max_iterations=3, verify=False, verbose=False)

    assert stats["runner"] == "custom"
    assert stats["custom_ran"] is True
    assert "compiled" not in stats
    assert "cuda_graph" not in stats


def test_explicit_torch_compile_runner():
    m, x = _make()
    mod, stats = Optimizer(backend=TorchBackend(), runner=TorchCompileRunner()).optimize(m, x, max_iterations=3, verify=False, verbose=False)

    assert stats["runner"] == "torch_compile"
    assert stats["compiled"] is True
    with torch.no_grad():
        assert torch.allclose(mod(x), m(x), atol=1e-5)


def test_explicit_chained_runner():
    m, x = _make()
    _, stats = Optimizer(backend=TorchBackend(), runner=ChainedRunner([TorchCompileRunner(), CudaGraphRunner()])).optimize(m, x, max_iterations=3, verify=False, verbose=False)

    assert stats["runner"] == ["torch_compile", "cuda_graph"]
    assert stats["compiled"] is True


def test_runner_without_name_records_class_name():
    """No ``name`` attribute — still a valid runner (duck-typed);
    stats falls back to the class name."""
    m, x = _make()

    class Nameless:
        def apply(self, module, example_input, stats):
            return module

    _, stats = Optimizer(backend=TorchBackend(), runner=Nameless()).optimize(m, x, max_iterations=3, verify=False, verbose=False)

    assert stats["runner"] == "Nameless"


# ------------------------------------------------------------------
#  runner_candidate — the autotune hook
# ------------------------------------------------------------------


def _ctx(with_ir: bool = True) -> AutotuneContext:
    """An AutotuneContext over a trivial ``add(x, p_w)`` term."""
    from catopt_core.ir import IR, Op, Param, TensorType, Var

    x = Var("x", TensorType((3,)))
    w = Param(name="p_w", typ=TensorType((3,)))
    if with_ir:
        term = Op.make("add", x, w)
        ir: IR | None = IR(
            root=term,
            inputs=[x],
            input_names=["x"],
            params={"p_w": TensorType((3,))},
        )
    else:
        ir = None
        term = None
    return AutotuneContext(
        model=nn.Identity(),
        example_input=torch.randn(3),
        ir=ir,
        term=term,
        param_values={"p_w": torch.randn(3)},
        sink=TorchSink(),
        delivered=nn.Identity(),
        lowering="generic",
    )


def test_runner_candidate_builds_fresh_routed_module():
    """The hook re-lowers the extracted term and applies the runner —
    usable as an autotune ``CandidateBuilder``."""

    class Rec:
        name = "rec"

        def apply(self, module, example_input, stats):
            stats["rec_ran"] = True
            return module

    ctx = _ctx()
    out = runner_candidate(Rec())(ctx)
    w = ctx.param_values["p_w"]
    with torch.no_grad():
        assert torch.allclose(out(torch.ones(3)), torch.ones(3) + w)


def test_runner_candidate_unavailable_without_ir():
    ctx = _ctx(with_ir=False)
    with pytest.raises(CandidateUnavailableError):
        runner_candidate(IdentityRunner())(ctx)


def test_runner_candidate_runs_inside_autotune():
    """End-to-end: a runner-shaped candidate verifies and times."""
    m, x = _make()
    _, stats = Optimizer(backend=TorchBackend()).optimize(m, x, strategy=Autotuned([("gen", runner_candidate(IdentityRunner()))], n_calls=3, warmup=1, verbose=False, builders=TORCH_BUILDERS), max_iterations=3)

    rec = stats["autotune"]["candidates"]["gen"]
    assert rec["status"] == "timed"
    assert stats["autotune"]["winner"] == "gen"
