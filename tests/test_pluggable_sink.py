"""A pluggable non-torch sink — the forcing function for Source/Sink.

``NumpySink`` implements ``catopt_core.ports.Sink`` over numpy: it
lowers core-IR ops to numpy closures, reports a numpy op set as
``supported_ops``, and verifies by running both executables.  The
``TorchSource`` frontend still produces the IR (there is no numpy
frontend here), but the *backend* is fully swapped — extraction is
priced against ``NumpySink.supported_ops`` and the returned executable
is numpy, never touching torch at run time.

If the Source/Sink seam were fake (torch leaked into the core), this
would not fit; that it fits is the conformance evidence.
"""

import numpy as np
import torch
import torch.nn as nn
from catopt.adapters import TorchSource
from catopt.cost import backend_cost, launch_aware_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Const, Op, Param, Var
from catopt.ops import OpTable
from catopt.optimize import discover_alternatives, optimize_model
from catopt.ports import Sink
from catopt.report import VerifyReport
from catopt_core.laws import all_rules


def _to_np(v):
    if hasattr(v, "detach"):
        return v.detach().cpu().numpy()
    return np.asarray(v)


def _dim(kw, default=-1):
    d = kw.get("dim", kw.get("axis", default))
    if isinstance(d, (list, tuple)):
        return tuple(int(i) for i in d)
    return int(d)


def _transpose(x, **kw):
    d0 = int(kw.get("dim0", kw.get("arg1", -2)))
    d1 = int(kw.get("dim1", kw.get("arg2", -1)))
    return np.swapaxes(x, d0, d1)


def _linear(x, w, b=None, **kw):
    out = x @ np.swapaxes(w, -1, -2)
    return out if b is None else out + b


_NP_BINDINGS = {
    "add": lambda a, b, **kw: a + b,
    "mul": lambda a, b, **kw: a * b,
    "sub": lambda a, b, **kw: a - b,
    "div": lambda a, b, **kw: a / b,
    "neg": lambda x, **kw: -x,
    "square": lambda x, **kw: x * x,
    "sqrt": lambda x, **kw: np.sqrt(x),
    "exp": lambda x, **kw: np.exp(x),
    "relu": lambda x, **kw: np.maximum(x, 0.0),
    "silu": lambda x, **kw: x / (1.0 + np.exp(-x)),
    "sigmoid": lambda x, **kw: 1.0 / (1.0 + np.exp(-x)),
    "tanh": lambda x, **kw: np.tanh(x),
    "matmul": lambda a, b, **kw: a @ b,
    "linear": _linear,
    "transpose": _transpose,
    "reshape": lambda x, **kw: x.reshape(tuple(kw["shape"])),
    "sum": lambda x, **kw: np.sum(
        x, axis=_dim(kw), keepdims=bool(kw.get("keepdim", False))
    ),
    "mean": lambda x, **kw: np.mean(
        x, axis=_dim(kw), keepdims=bool(kw.get("keepdim", False))
    ),
    "concat": lambda *ts, **kw: np.concatenate(
        list(ts), axis=_dim(kw, 0)
    ),
}


def _ops_of(term):
    out = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op):
            out.add(t.op)
            stack.extend(t.args)
    return out


class _NumpyModule:
    """A lowered IR term evaluated with numpy arrays."""

    def __init__(self, ir, params):
        self._ir = ir
        self._params = params

    def forward(self, *xs):
        env = {
            v.name: x for v, x in zip(self._ir.inputs, xs, strict=False)
        }
        return self._eval(self._ir.root, env)

    __call__ = forward

    def _eval(self, t, env):
        if isinstance(t, Var):
            return env[t.name]
        if isinstance(t, Param):
            return self._params[t.name]
        if isinstance(t, Const):
            return np.asarray(t.value)
        args = [self._eval(a, env) for a in t.args]
        return _NP_BINDINGS[t.op](*args, **dict(t.attrs))


class NumpySink:
    """A ``Sink`` whose runtime is numpy."""

    supported_ops = frozenset(_NP_BINDINGS)

    def __init__(self):
        self._ops = OpTable.full()

    @property
    def ops(self):
        return self._ops

    def lower(self, ir, params=None):
        return _NumpyModule(
            ir, {k: _to_np(v) for k, v in (params or {}).items()}
        )

    def verify(self, ref, opt, inputs, *, rtol=1e-4, atol=None):
        args = inputs if isinstance(inputs, tuple) else (inputs,)
        args = [_to_np(a) for a in args]
        ref_out = np.asarray(ref(*args))
        out = np.asarray(opt(*args))
        max_abs = float(np.max(np.abs(ref_out - out)))
        max_rel = max_abs / (float(np.max(np.abs(ref_out))) + 1e-8)
        passed = bool(max_rel < rtol) and (
            atol is None or bool(max_abs <= atol)
        )
        return VerifyReport(
            max_abs=max_abs, max_rel=max_rel, passed=passed
        )


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))


def _saturate(ir, iterations=3):
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=iterations)
    return eg, root


def test_numpy_sink_is_sink():
    assert isinstance(NumpySink(), Sink)


def test_numpy_sink_lowers_and_runs():
    model = _model()
    x = torch.randn(2, 8)
    ir, params = TorchSource().to_ir(model, x)
    mod = NumpySink().lower(ir, params)
    got = mod(x.detach().numpy())
    want = model(x).detach().numpy()
    assert np.allclose(got, want, atol=1e-5)


def test_backend_relative_extraction_avoids_unsupported_ops():
    """The search commits only to numpy-lowerable forms, and the
    result still equals the original under the numpy runtime."""
    model = _model()
    x = torch.randn(2, 8)
    sink = NumpySink()
    ir, params = TorchSource().to_ir(model, x)
    eg, root = _saturate(ir)
    priced = backend_cost(launch_aware_cost, sink.supported_ops)
    best = eg.extract_best(root, priced)
    assert priced(best) != float("inf")
    orig = sink.lower(ir, params)
    opt = sink.lower(
        IR(
            root=best,
            inputs=ir.inputs,
            input_names=ir.input_names,
            params=ir.params,
        ),
        params,
    )
    rep = sink.verify(orig, opt, x.detach().numpy())
    assert rep.passed


def test_optimize_model_with_numpy_sink():
    model = _model()
    x = torch.randn(2, 8)
    mod, _stats = optimize_model(
        model,
        x,
        source=TorchSource(),
        sink=NumpySink(),
        verbose=False,
        max_iterations=2,
    )
    assert isinstance(mod, _NumpyModule)
    got = mod(x.detach().numpy())
    want = model(x).detach().numpy()
    assert np.allclose(got, want, atol=1e-4)


def test_discover_alternatives_with_custom_source_and_sink():
    model = _model()
    x = torch.randn(2, 8)
    sink = NumpySink()
    res = discover_alternatives(
        model,
        x,
        source=TorchSource(),
        sink=sink,
        max_iterations=2,
        top_k=2,
    )
    assert isinstance(res["alternatives"], list)
    for _cost, term in res["alternatives"]:
        assert sink.supported_ops.issuperset(_ops_of(term))
