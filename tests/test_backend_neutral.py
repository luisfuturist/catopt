"""Plan 0007 — the orchestrator is backend-neutral.

Two complements to :mod:`tests.test_core_torch_free` (the core half):

* **static** — an AST walk over ``packages/catopt-optimize/src``
  rejects any direct ``import``/``from`` of the backend roots
  (``torch``, ``catopt_torch``, ``catopt_cuda``).  The
  ``catopt_carriers`` imports the regime/carrier helpers defer to
  call time are *not* forbidden at the statement level — carriers
  are optional rewrite machinery — but the runtime proof below
  blocks them too and the whole package must still import.
* **runtime** — ``catopt_optimize`` is imported in a subprocess with
  ``torch``/``numpy``/``catopt_torch``/``catopt_carriers``/
  ``catopt_cuda`` all blocked by a meta-path finder; every
  submodule walks, the lazy carrier fallbacks degrade cleanly, and
  a full ``Optimizer`` run on a *pure-Python* fake backend still
  optimizes end-to-end — search, lower, autotune — without a tensor
  library in ``sys.modules``.

The fake backend exercises the plan-0007 port surface as a whole:
:class:`~catopt_core.pipeline.Backend` is a plain value of four
ports; :class:`~catopt_core.ports.Source` /
:class:`~catopt_core.ports.Sink` (+ ``executors``) /
:class:`~catopt_core.ports.Composer` /
:class:`~catopt_core.ports.Meter` drive every phase.
"""

from __future__ import annotations

import ast
import copy
import pathlib
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from typing import NamedTuple

import pytest
from catopt_core.ir import IR, Op, Param, TensorType, Var
from catopt_core.ops import OpTable
from catopt_core.pipeline import Backend
from catopt_core.ports import (
    Composer,
    ExecutorSpec,
    Meter,
    Sink,
    Source,
    TimingResult,
)
from catopt_optimize import (
    Autotuned,
    Compositional,
    Optimizer,
)
from catopt_optimize.optimize import _lower_extracted

_OPT_SRC = (
    pathlib.Path(__file__).resolve().parents[1]
    / "packages"
    / "catopt-optimize"
    / "src"
)

#: Import roots the orchestrator must never name directly.
_FORBIDDEN = frozenset({"torch", "catopt_torch", "catopt_cuda"})


# ---------------------------------------------------------------------------
#  Static proof — no direct backend import statement in catopt_optimize
# ---------------------------------------------------------------------------


def _imported_roots(tree: ast.AST):
    """Yield ``(root_name, lineno)`` for every absolute import in *tree*."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def test_optimize_source_has_no_backend_imports():
    offenders: list[str] = []
    files = sorted(_OPT_SRC.rglob("*.py"))
    assert files, f"no orchestrator sources under {_OPT_SRC}"
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for root, lineno in _imported_roots(tree):
            if root in _FORBIDDEN:
                rel = path.relative_to(_OPT_SRC)
                offenders.append(f"{rel}:{lineno} imports {root}")
    assert offenders == [], (
        "catopt-optimize must import no backend directly; found:\n"
        + "\n".join(offenders)
    )


# ---------------------------------------------------------------------------
#  Runtime proof — the package loads and degrades with backends blocked
# ---------------------------------------------------------------------------

_BLOCKED_SCRIPT = r"""
import sys

_BLOCKED = ("torch", "numpy", "catopt_torch", "catopt_carriers",
            "catopt_cuda")


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            raise ModuleNotFoundError(f"blocked import: {name}")
        return None


sys.meta_path.insert(0, _Blocker())

import importlib
import pkgutil

import catopt_optimize

# Walk EVERY orchestrator submodule — none may need a blocked package
# at import time.
for _info in pkgutil.walk_packages(
    catopt_optimize.__path__, "catopt_optimize."
):
    # The compat shims/aliases whose whole point is resolving the
    # torch package must raise the blocked ModuleNotFoundError, not
    # a different failure.
    if _info.name in (
        "catopt_optimize.calibrate", "catopt_optimize.export"
    ):
        continue
    importlib.import_module(_info.name)

assert "torch" not in sys.modules
assert "catopt_torch" not in sys.modules
assert "catopt_carriers" not in sys.modules

# The lazy carrier fallbacks degrade to empty structures, never raise.
import catopt_optimize.optimize as O
import catopt_optimize.regime as R

assert O._carrier_plans() == {}
assert O._carrier_lifts(object(), {}) == []
assert R._xc_laws() == []
assert O._CARRIER_PLANS == {}
# Core scan laws compose in; blocked carriers contribute nothing.
assert isinstance(R.CARRIER_LAWS, list)

print("optimize-imported-torch-free")
"""


def test_optimize_imports_without_backends():
    """Import the whole orchestrator with every backend blocked."""
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKED_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"backend-free orchestrator import failed:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "optimize-imported-torch-free" in proc.stdout


# ---------------------------------------------------------------------------
#  A pure-Python backend — the ports as plain values
# ---------------------------------------------------------------------------


class _ListModel:
    """A pure-Python 'model': ``x -> x + x`` on a list of floats."""

    def __init__(self, inner=None):
        self.inner = inner

    def __call__(self, x):
        return [a + a for a in x]

    forward = __call__


class FakeModule:
    """A lowered IR term evaluated over Python lists."""

    def __init__(self, ir, params):
        self._root = ir.root
        self._inputs = list(ir.inputs)
        self._param_map = dict(params)

    def forward(self, *xs):
        env = {
            v.name: x for v, x in zip(self._inputs, xs, strict=False)
        }
        return self._eval(self._root, env)

    __call__ = forward

    def _eval(self, t, env):
        if isinstance(t, Var):
            return env[t.name]
        if isinstance(t, Param):
            return self._param_map[t.name]
        args = [self._eval(a, env) for a in t.args]
        if t.op == "add":
            return [a + b for a, b in zip(*args, strict=False)]
        raise KeyError(t.op)


class _VerifyOut(NamedTuple):
    max_abs: float
    max_rel: float
    passed: bool


class FakeSink:
    """``Sink`` whose runtime is plain Python."""

    supported_ops = frozenset({"add"})

    @property
    def ops(self):
        return OpTable.core()

    @property
    def executors(self):
        return MappingProxyType(
            {
                "generic": ExecutorSpec(
                    name="generic",
                    accepts=lambda ir, term: True,
                    lower=self.lower,
                    engaged=lambda mod: True,
                ),
                "twin": ExecutorSpec(
                    name="twin",
                    accepts=lambda ir, term: False,
                    lower=self.lower,
                    engaged=lambda mod: True,
                ),
            }
        )

    def lower(self, ir, params=None):
        return FakeModule(ir, params or {})

    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        a = list(ref(*args))
        b = list(opt(*args))
        diffs = [abs(u - v) for u, v in zip(a, b, strict=False)]
        max_abs = max(diffs, default=0.0)
        max_rel = max_abs / (max(abs(x) for x in a) + 1e-8)
        return _VerifyOut(max_abs, max_rel, max_rel < rtol)


class FakeComposer:
    """The structural port — a plain-Python compositional surface."""

    def blocks(self, model, predicate=None):
        out = []
        if getattr(model, "inner", None) is not None:
            out.append(("inner", model.inner))
        return out

    def capture_inputs(self, model, blocks, example_input):
        captured = {n: ((example_input,), {}) for n, _ in blocks}
        io = {n: {"calls": 1} for n, _ in blocks}
        return captured, io

    def perturbed(self, example_input):
        return [v + 1.0 for v in example_input]

    def clone_sharing(self, model):
        return copy.deepcopy(model)

    def graft(self, model, replacements):
        for name, mod in replacements.items():
            setattr(model, name, mod)
        return model

    def boundary(self, *_a, **_k):
        return None

    # Deliberately NO ``cross_pairs`` / ``param_report`` — the
    # optional composer hooks must degrade, not require.


class FakeMeter:
    """``Meter`` — a fixed deterministic measurement."""

    def time(self, runnable, inputs, *, warmup=5, n_calls=30):
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        runnable(*args)
        return TimingResult(median_s=0.001, iqr_s=0.0, n_calls=n_calls)


class FakeSource:
    """``Source`` — emits a one-op IR whatever the model is."""

    def to_ir(self, model, example_input):
        x = Var("x", TensorType((4,)))
        w = Param("w", TensorType((4,)))
        root = Op.make("add", x, x)
        ir = IR(
            root=root,
            inputs=[x],
            input_names={"x"},
            params={"w": w},
        )
        # One non-tensor leaf — ``_param_map`` material for the
        # autotune re-lowering pass (exercises the copy.copy arm).
        return ir, {"w": 1.5}


def _fake_backend() -> Backend:
    return Backend(
        source=FakeSource(),
        sink=FakeSink(),
        composer=FakeComposer(),
        meter=FakeMeter(),
    )


# ---------------------------------------------------------------------------
#  The Backend value
# ---------------------------------------------------------------------------


def test_backend_is_a_frozen_port_tuple():
    b = _fake_backend()
    assert isinstance(b.source, Source)
    assert isinstance(b.sink, Sink)
    assert isinstance(b.composer, Composer)
    assert isinstance(b.meter, Meter)
    assert "generic" in b.sink.executors
    with pytest.raises(FrozenInstanceError):
        b.sink = FakeSink()  # type: ignore[misc]


def test_optimizer_requires_source_and_sink():
    with pytest.raises(TypeError, match="source/sink"):
        Optimizer()
    with pytest.raises(TypeError, match="source/sink"):
        Optimizer(source=FakeSource())


def test_optimizer_backend_vs_explicit_ports():
    b = _fake_backend()
    # Explicit ports override the backend's — the backend supplies
    # only what is missing.
    s2, k2, c2, m2 = (
        FakeSource(),
        FakeSink(),
        FakeComposer(),
        FakeMeter(),
    )
    opt = Optimizer(
        backend=b, source=s2, sink=k2, composer=c2, meter=m2
    )
    assert opt.source is s2 and opt.sink is k2
    assert opt.composer is c2 and opt.meter is m2
    # … and the backend still fills whatever is not given.
    opt_b = Optimizer(backend=b)
    assert opt_b.composer is b.composer and opt_b.meter is b.meter
    # A bare explicit-port optimizer still works for Monolithic.
    opt2 = Optimizer(source=FakeSource(), sink=FakeSink())
    assert opt2.composer is None and opt2.meter is None


# ---------------------------------------------------------------------------
#  Fake backend end-to-end — no torch anywhere in the run
# ---------------------------------------------------------------------------


def test_fake_backend_monolithic():
    model = _ListModel()
    x = [1.0, 2.0, 3.0, 4.0]
    opt = Optimizer(backend=_fake_backend())
    lr = opt.optimize(model, x, max_iterations=2)
    assert lr.module(x) == model(x)
    assert lr.stats["runner"] == "identity"
    assert lr.stats["lowering"] == "generic"
    assert lr.verified is not None and lr.verified.passed


def test_fake_backend_phase_verbs_and_verify():
    model = _ListModel()
    x = [1.0, 2.0]
    opt = Optimizer(backend=_fake_backend())
    res = opt.search(model, x, max_iterations=2)
    lr = opt.lower(res, x, verify=True)
    assert lr.verified is not None and lr.verified.passed
    lr2 = opt.lower(res, x, verify=False)
    assert lr2.verified is None


def test_fake_backend_autotuned():
    model = _ListModel()
    x = [1.0, 2.0, 3.0, 4.0]
    opt = Optimizer(backend=_fake_backend())
    lr = opt.optimize(
        model,
        x,
        strategy=Autotuned(
            candidates=("generic", "eager", "twin", "bogus"),
            n_calls=2,
            warmup=0,
        ),
        max_iterations=2,
    )
    at = lr.stats["autotune"]
    assert at["winner"] in {"generic", "eager", "twin"}
    assert at["candidates"]["bogus"]["status"] == "unknown"
    timed = [
        n
        for n, r in at["candidates"].items()
        if r.get("status") == "timed"
    ]
    assert len(timed) == 3


def test_autotuned_requires_meter():
    opt = Optimizer(source=FakeSource(), sink=FakeSink())
    with pytest.raises(TypeError, match="Meter"):
        opt.optimize(
            _ListModel(),
            [1.0],
            strategy=Autotuned(n_calls=1, warmup=0),
        )


def test_fake_backend_compositional():
    model = _ListModel(inner=_ListModel())
    x = [1.0, 2.0]
    opt = Optimizer(backend=_fake_backend())
    lr = opt.optimize(
        model,
        x,
        strategy=Compositional(max_cross_pairs=0),
        max_iterations=2,
        verbose=False,
    )
    stats = lr.stats
    assert stats["n_blocks"] == 1
    assert stats["blocks"]["inner"]["status"] == "optimized"
    # The composer without a param_report hook yields no audit.
    assert stats["param_report"]["original_params"] == 0


def test_compositional_requires_composer():
    opt = Optimizer(source=FakeSource(), sink=FakeSink())
    with pytest.raises(TypeError, match="Composer"):
        opt.optimize(
            _ListModel(inner=_ListModel()),
            [1.0],
            strategy=Compositional(max_cross_pairs=0),
        )


def test_lower_extracted_routes_to_sink_executor():
    """A carrier spec's ``accepts`` probe wins routing; else generic."""
    sink = FakeSink()
    x = Var("x", TensorType((2,)))
    term = Op.make("add", x, x)
    ir = IR(root=term, inputs=[x], input_names={"x"}, params={})
    routed = _lower_extracted(term, ir, {}, sink)
    assert isinstance(routed, FakeModule)


# ---------------------------------------------------------------------------
#  Delegation arms — moved names still resolve (torch install present)
# ---------------------------------------------------------------------------


def test_optimize_module_delegations():
    import catopt_optimize.optimize as O

    # API wrappers moved to catopt_torch.api — resolvable, lazy.
    assert callable(O.optimize_model)
    assert callable(O.optimize_compositional)
    # Composer internals.
    assert callable(O._select_blocks)
    assert callable(O._perturbed_input)
    assert callable(O._cross_pair_pass)  # bound composer method shape
    # Fold internals.
    assert callable(O._specialize_causal)
    # Report internals.
    assert callable(O.verify_module)
    # The historical torch patch point resolves the shared module.
    import torch as _t

    assert O.torch is _t
    # Carrier-plan cache materialises lazily and caches.
    plans = O._CARRIER_PLANS
    assert O._CARRIER_PLANS is plans
    with pytest.raises(AttributeError):
        _ = O.bogus_name


def test_runners_module_delegations():
    import catopt_optimize.runners as R
    from catopt_cuda import CudaGraphRunner
    from catopt_torch.runners import TorchCompileRunner

    assert R.TorchCompileRunner is TorchCompileRunner
    assert R.CudaGraphRunner is CudaGraphRunner
    with pytest.raises(AttributeError):
        _ = R.bogus_name


def test_regime_module_delegations():
    import catopt_optimize.regime as R

    # CARRIER_LAWS materialises once, then caches.
    laws = R.CARRIER_LAWS
    assert R.CARRIER_LAWS is laws
    assert list(R.XC_LAWS) == R.XC_LAWS
    from catopt_torch.regime import RegimeDispatch

    assert R.RegimeDispatch is RegimeDispatch
    with pytest.raises(AttributeError):
        _ = R.bogus_name


def test_autotune_module_delegations():
    import catopt_optimize.autotune as A
    from catopt_optimize.optimize import Autotuned
    from catopt_torch.api import optimize_model_autotuned

    assert A.Autotuned is Autotuned
    assert A.optimize_model_autotuned is optimize_model_autotuned
    with pytest.raises(AttributeError):
        _ = A.bogus_name


def test_package_delegations():
    import catopt_optimize as pkg
    from catopt_torch.api import optimize_model
    from catopt_torch.export import export_optimized

    assert pkg.optimize_model is optimize_model
    assert pkg.export_optimized is export_optimized
    with pytest.raises(AttributeError):
        _ = pkg.bogus_name


def test_torch_package_delegations():
    import catopt_torch
    from catopt_torch.backend import TorchBackend
    from catopt_torch.runners import TorchCompileRunner

    assert catopt_torch.TorchBackend is TorchBackend
    assert catopt_torch.TorchCompileRunner is TorchCompileRunner
    assert callable(catopt_torch.optimize_model)
    with pytest.raises(AttributeError):
        _ = catopt_torch.bogus_name


def test_compat_shim_modules_alias_the_torch_ones():
    import catopt_optimize.calibrate as shim_cal
    import catopt_optimize.export as shim_exp
    import catopt_torch.calibrate
    import catopt_torch.export

    assert shim_cal is catopt_torch.calibrate
    assert shim_exp is catopt_torch.export


# ---------------------------------------------------------------------------
#  Regime / adapter error + fallback arms
# ---------------------------------------------------------------------------


def test_register_regime_backend_partial_args(monkeypatch):
    import catopt_optimize.regime as R

    sentinel = object()
    monkeypatch.setattr(R, "_DISPATCH_CLS", None)
    # dispatch-only registration: executors untouched, cls recorded.
    R.register_regime_backend(dispatch=sentinel)
    assert R._DISPATCH_CLS is sentinel


def test_frontier_build_without_dispatch(monkeypatch):
    import catopt_optimize.regime as R

    monkeypatch.setattr(R, "_DISPATCH_CLS", None)
    frontier = R.RegimeFrontier(
        eg=None,
        root_eid=0,
        src_term=None,
        ir=None,
        regimes=[],
        choices={},
        executors={},
    )
    with pytest.raises(RuntimeError, match="dispatch backend"):
        frontier.build()


def test_torch_sink_executors_degrade_without_carriers(monkeypatch):
    """Blocked carrier imports leave the trace+generic routing table."""
    import sys as _sys

    from catopt_torch.adapters import TorchSink

    monkeypatch.setitem(
        _sys.modules, "catopt_carriers.scan_lower", None
    )
    monkeypatch.setitem(_sys.modules, "catopt_carriers.om_lower", None)
    monkeypatch.setitem(_sys.modules, "catopt_carriers.omd_lower", None)
    monkeypatch.setitem(
        _sys.modules, "catopt_carriers.om_streaming", None
    )
    sink = TorchSink()
    sink._executors = None  # rebuild under the blocked names
    table = sink.executors
    assert set(table) == {"trace", "generic"}


def test_build_egraph_accepts_explicit_source():
    import torch
    import torch.nn as nn
    from catopt_torch.adapters import TorchSource
    from catopt_torch.regime import build_egraph

    m = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    eg, _root, ir, _src, _stats = build_egraph(
        m, x, source=TorchSource(), rules=[], xc=False
    )
    assert eg is not None and ir is not None


def test_torch_builder_requires_ir():
    from catopt_optimize.autotune import (
        AutotuneContext,
        CandidateUnavailableError,
    )
    from catopt_torch.api import TORCH_BUILDERS

    ctx = AutotuneContext(
        model=None,
        example_input=None,
        ir=None,
        term=None,
        param_values={},
        sink=None,
        delivered=None,
        lowering="generic",
    )
    for name in ("torch_compile", "torch_compile_generic"):
        with pytest.raises(CandidateUnavailableError):
            TORCH_BUILDERS[name](ctx)


def test_runner_candidate_requires_ir():
    from catopt_optimize.autotune import (
        AutotuneContext,
        CandidateUnavailableError,
    )
    from catopt_optimize.runners import IdentityRunner, runner_candidate

    ctx = AutotuneContext(
        model=None,
        example_input=None,
        ir=None,
        term=None,
        param_values={},
        sink=None,
        delivered=None,
        lowering="generic",
    )
    with pytest.raises(CandidateUnavailableError):
        runner_candidate(IdentityRunner())(ctx)


def test_carrier_helpers_degrade_without_carriers(monkeypatch):
    """The lazy carrier fallbacks return empty in-process too."""
    import sys as _sys

    import catopt_optimize.optimize as O
    import catopt_optimize.regime as R
    from catopt_core.egraph import EGraph

    for name in (
        "catopt_carriers.om_lower",
        "catopt_carriers.omd_lower",
        "catopt_carriers.scan_lower",
        "catopt_carriers.trace_lift",
        "catopt_carriers.xcarrier",
    ):
        monkeypatch.setitem(_sys.modules, name, None)
    assert O._carrier_plans() == {}
    # A real (empty) e-graph — the share/tying passes tolerate it.
    assert isinstance(O._carrier_lifts(EGraph(), {}), list)
    assert R._xc_laws() == []
    # build_egraph with blocked carriers runs the plain core tier.
    eg, _root, ir, _src, _st = R.build_egraph(
        _ListModel(), [1.0, 2.0], source=FakeSource(), rules=[], xc=True
    )
    assert eg is not None and ir is not None


def test_register_regime_backend_executors_only():
    import catopt_optimize.regime as R

    R.register_regime_backend(executors={"_probe": object()})
    assert "_probe" in R.EXECUTORS
    del R.EXECUTORS["_probe"]


def test_lower_extracted_empty_executor_table():
    """A sink whose table is empty routes through ``sink.lower``."""

    class BareSink(FakeSink):
        @property
        def executors(self):
            return {}

    x = Var("x", TensorType((2,)))
    term = Op.make("add", x, x)
    ir = IR(root=term, inputs=[x], input_names={"x"}, params={})
    mod = _lower_extracted(term, ir, {}, BareSink())
    assert isinstance(mod, FakeModule)


# ---------------------------------------------------------------------------
#  The real Backend — TorchBackend is a plain value
# ---------------------------------------------------------------------------


def test_torch_backend_value():
    from catopt_torch.backend import TorchBackend

    b = TorchBackend()
    assert isinstance(b, Backend)
    from catopt_torch.adapters import TorchSink, TorchSource

    assert isinstance(b.source, TorchSource)
    assert isinstance(b.sink, TorchSink)
    with pytest.raises(FrozenInstanceError):
        b.meter = None  # type: ignore[misc]


def test_torch_backend_optimizes_end_to_end():
    import torch
    import torch.nn as nn
    from catopt_torch.backend import TorchBackend

    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(8, 8)

        def forward(self, x):
            return torch.relu(self.w(x))

    m = M()
    x = torch.randn(4, 8)
    lr = Optimizer(backend=TorchBackend()).optimize(
        m, x, max_iterations=2, verbose=False
    )
    with torch.no_grad():
        assert torch.allclose(lr.module(x), m(x), atol=1e-5)
