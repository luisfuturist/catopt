"""BatchedExecutorBase — the shared machinery of the three batched
executors (extracted in plan 0002 phase A).

CPU-only checks: the real CUDA capture body is pragma'd CUDA-only, so
the capture guards, drop/re-capture bookkeeping, ``_input_env``
prologue, ``ev_factory`` closure, and ``_cached`` device-cache are
exercised on real plans built via ``to_batched_*_module``.
"""

import pytest
import torch

from catopt.executors import BatchedExecutorBase
from catopt.ir import IR, Op, Param, TensorType, Var
from catopt.om_lower import BatchedOMModule, to_batched_om_module
from catopt.omd_lower import (
    BatchedOmdModule,
    _select_index,
    to_batched_omd_module,
)
from catopt.scan_lower import (
    BatchedScanModule,
    to_batched_scan_module,
)
from catopt.torch_bridge import ir_to_torch_module


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _rand(shape, seed=0):
    g = torch.Generator().manual_seed(17 + seed)
    return torch.randn(tuple(shape), dtype=torch.float64, generator=g)


def _compose(opname, leaves):
    """Balanced binary bracketing (chunk order preserved)."""
    if len(leaves) == 1:
        return leaves[0]
    mid = len(leaves) // 2
    return Op.make(
        opname,
        _compose(opname, leaves[:mid]),
        _compose(opname, leaves[mid:]),
    )


def _scan_case():
    """applyd(balanced affd_compose tree of aff_diag(a, x[t]), h)."""
    T, d = 3, 4
    x = _v("x", T, d)
    leaves = [
        Op.make(
            "aff_diag",
            _p("a", d),
            Op.make("select", x, arg1=0, arg2=t),
        )
        for t in range(T)
    ]
    root = Op.make(
        "applyd", _compose("affd_compose", leaves), _p("h", d)
    )
    ir = IR(root=root, inputs=[x], input_names={"x"}, params={})
    pv = {
        "a": _rand((d,), 1).clamp(-0.9, 0.9),
        "h": _rand((d,), 2),
    }
    return ir, pv, (_rand((T, d), 3),)


def _om_case():
    """om_apply(om_compose of om_elem(q @ k_i.T, v_i)) — 5 inputs."""
    B, H, Tq, d, dv = 1, 1, 2, 3, 2
    q = _v("q", B, H, Tq, d)
    ks = [_v(f"k{i}", B, H, 2, d) for i in range(2)]
    vs = [_v(f"v{i}", B, H, 2, dv) for i in range(2)]
    leaves = [
        Op.make(
            "om_elem",
            Op.make(
                "matmul",
                q,
                Op.make("transpose", k, arg1=-2, arg2=-1),
            ),
            vv,
        )
        for k, vv in zip(ks, vs, strict=True)
    ]
    root = Op.make("om_apply", _compose("om_compose", leaves))
    inputs = [q, *ks, *vs]
    ir = IR(
        root=root,
        inputs=inputs,
        input_names={vv.name for vv in inputs},
        params={},
    )
    args = tuple(
        _rand(tuple(vv.typ.shape), 4 + i) for i, vv in enumerate(inputs)
    )
    return ir, {}, args


def _omd_case():
    """omd_apply(omd_elem(s, stack affd_a f_i, stack affd_b f_i), h)."""
    T, Tq, d = 4, 2, 3
    p_a, p_s, p_h = _p("p_a", T, d), _p("p_s", Tq, T), _p("p_h", d)
    x = _v("x", T, d)
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", p_a, arg1=0, arg2=t),
            Op.make("select", x, arg1=0, arg2=t),
        )
        for t in range(T)
    ]
    fs = [leaves[0]]
    for t in range(1, T):
        fs.append(Op.make("affd_compose", leaves[t], fs[-1]))
    a_map = Op.make("stack", *(Op.make("affd_a", f) for f in fs), dim=0)
    b_map = Op.make("stack", *(Op.make("affd_b", f) for f in fs), dim=0)
    term = Op.make(
        "omd_apply", Op.make("omd_elem", p_s, a_map, b_map), p_h
    )
    ir = IR(root=term, inputs=[x], input_names={"x"}, params={})
    pv = {
        "p_a": _rand((T, d), 1),
        "p_s": _rand((Tq, T), 2),
        "p_h": _rand((d,), 3),
    }
    return ir, pv, (_rand((T, d), 4),)


def _cases():
    """The three batched executors on real plans + their arg tuples."""
    ir, pv, args = _scan_case()
    scan = to_batched_scan_module(ir, param_values=pv)
    ir, pv, args_om = _om_case()
    om = to_batched_om_module(ir, param_values=pv)
    ir, pv, args_omd = _omd_case()
    omd = to_batched_omd_module(ir, param_values=pv)
    return [(scan, args), (om, args_om), (omd, args_omd)]


# ---------------------------------------------------------------------------
#  Mixin wiring
# ---------------------------------------------------------------------------


def test_shared_methods_come_from_base():
    """All three classes inherit the mixin's members unmodified."""
    for cls in (BatchedScanModule, BatchedOMModule, BatchedOmdModule):
        assert (
            cls.capture_cuda_graph
            is BatchedExecutorBase.capture_cuda_graph
        )
        assert (
            cls.drop_cuda_graph is BatchedExecutorBase.drop_cuda_graph
        )
        assert (
            cls.is_graph_captured.fget
            is BatchedExecutorBase.is_graph_captured.fget
        )
        assert cls._input_env is BatchedExecutorBase._input_env
        assert cls.ev_factory is BatchedExecutorBase.ev_factory
        assert cls._cached is BatchedExecutorBase._cached


def test_mixin_and_module_wiring():
    """Mixin + nn.Module MRO: super().__init__() still registers
    submodules, is_batched / _plan come from the concrete class."""
    for mod, _ in _cases():
        assert isinstance(mod, BatchedExecutorBase)
        assert isinstance(mod, torch.nn.Module)
        assert mod.is_batched and mod._plan is not None
        # nn.Module machinery reached through the mixin MRO: eval_mod
        # is a registered submodule, not a plain attribute.
        assert "eval_mod" in dict(mod.named_modules())


# ---------------------------------------------------------------------------
#  CUDA-graph bookkeeping (CPU paths)
# ---------------------------------------------------------------------------


def test_capture_cpu_noop_then_cuda_guard(monkeypatch):
    """``is_graph_captured`` False path, the documented CPU no-op, and
    the ValueError guard under a fake CUDA flag."""
    # force the no-CUDA arm regardless of the host device
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for mod, args in _cases():
        assert not mod.is_graph_captured
        # no CUDA → documented no-op returning self, no capture state
        assert mod.capture_cuda_graph(*args) is mod
        assert mod._graph is None
        assert mod._graph_inputs == []
        assert mod._graph_out is None
        # pretend CUDA exists: CPU inputs trip the input guard
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with pytest.raises(ValueError, match="CUDA"):
            mod.capture_cuda_graph(*args)
        assert not mod.is_graph_captured
        monkeypatch.undo()


class _FakeGraph:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def test_graph_replay_drop_and_recapture():
    """A captured-graph slot replays verbatim; drop resets the fields;
    re-capture works and the eager path is restored."""
    for mod, args in _cases():
        sentinel = torch.full((1,), 3.0, dtype=torch.float64)
        mod._graph = _FakeGraph()
        mod._graph_inputs = list(args)
        mod._graph_out = sentinel
        assert mod.is_graph_captured
        assert mod(*[a.clone() for a in args]) is sentinel
        assert mod._graph.replays == 1

        mod.drop_cuda_graph()
        assert not mod.is_graph_captured
        assert mod._graph is None
        assert mod._graph_inputs == []
        assert mod._graph_out is None

        # re-"capture" — _init_graph_state left clean state
        fake = _FakeGraph()
        mod._graph = fake
        mod._graph_inputs = list(args)
        mod._graph_out = sentinel
        assert mod(*args) is sentinel and fake.replays == 1
        mod.drop_cuda_graph()

        # eager path restored — matches the embedded serial evaluator
        torch.testing.assert_close(mod(*args), mod.eval_mod(*args))


# ---------------------------------------------------------------------------
#  _input_env prologue + ev_factory closure
# ---------------------------------------------------------------------------


def test_input_env_matches_forward_prologue():
    """``(x, env)`` is exactly what the old inline prologue computed:
    x = xs[0]/None, env = {"self": x} + positional input bindings."""
    for mod, args in _cases():
        x, env = mod._input_env(args)
        assert x is args[0] and env["self"] is args[0]
        assert set(env) == {"self"} | {inp.name for inp in mod._inputs}
        for i, inp in enumerate(mod._inputs):
            assert env[inp.name] is args[i]

        # fewer args than inputs → missing names default to x
        x, env = mod._input_env(args[:1])
        assert x is args[0]
        for inp in mod._inputs:
            assert env[inp.name] is args[0]

        # zero args → x is None and every binding defaults to it
        x, env = mod._input_env(())
        assert x is None and env["self"] is None
        for inp in mod._inputs:
            assert env[inp.name] is None


def test_ev_factory_evaluates_and_memoises():
    """ev(t) is eval_mod._eval(t, env, x, memo): Vars resolve through
    env, Op results memoise under the term object."""
    for mod, args in _cases():
        x, env = mod._input_env(args)
        memo: dict = {}
        ev = mod.ev_factory(env, x, memo)

        inp = mod._inputs[0]
        assert ev(inp) is args[0]

        t = Op.make("relu", inp)
        out = ev(t)
        torch.testing.assert_close(out, args[0].clamp_min(0))
        assert memo[t] is out
        assert ev(t) is out  # second call hits the caller's memo


def _counted_make(calls):
    def make(like):
        calls.append(like)
        return torch.tensor(
            [0, 1], dtype=torch.long, device=like.device
        )

    return make


def test_cached_device_dtype_cache():
    """``_cached`` memoises per key, remaking only on device/dtype
    mismatch."""
    for mod, args in _cases():
        made = []
        make = _counted_make(made)
        like = args[0]
        a = mod._cached(("k",), like, make, dtype=torch.long)
        assert mod._cached(("k",), like, make, dtype=torch.long) is a
        assert len(made) == 1
        # dtype mismatch remakes and re-seats the cache entry
        b = mod._cached(("k",), like, make, dtype=torch.float64)
        assert b is not a and len(made) == 2
        assert mod._const_cache[("k",)] is b


def test_omd_select_index_getitem_branch():
    """The omd twin of ``scan_lower._select_index``'s getitem
    spelling — kept covered through ``_part_gather``'s recognizer."""
    base = _v("b", 8, 4)
    assert _select_index(Op.make("getitem", base, arg1=2)) == (
        base,
        0,
        2,
    )
    assert (
        _select_index(
            Op.make("getitem", base, arg1="i", validate=False)
        )
        is None
    )


# ---------------------------------------------------------------------------
#  End-to-end: mixin plumbing didn't change the forwards
# ---------------------------------------------------------------------------


def test_forward_matches_serial_eval_on_real_plans():
    """The extracted prologue feeds the same eval_mod — batched output
    agrees with the plain IRModule on each case's args."""
    for mod, args in _cases():
        torch.testing.assert_close(mod(*args), mod.eval_mod(*args))
        # and the input-count edge: extra/positional args still route
        # through _input_env identically
        assert not mod.is_graph_captured


def test_nonbatched_ir_uses_same_helpers():
    """A non-matching root keeps _plan=None: capture is a no-op and
    forward delegates to eval_mod through the shared guards."""
    x = _v("x", 4)
    ir = IR(
        root=Op.make("relu", x),
        inputs=[x],
        input_names={"x"},
        params={},
    )
    for make_mod in (
        to_batched_scan_module,
        to_batched_om_module,
        to_batched_omd_module,
    ):
        mod = make_mod(ir)
        assert isinstance(mod, BatchedExecutorBase)
        assert not mod.is_batched and mod._plan is None
        assert not mod.is_graph_captured
        assert mod.capture_cuda_graph(_rand((4,))) is mod
        mod.drop_cuda_graph()
        xv = _rand((4,), 9)
        torch.testing.assert_close(mod(xv), xv.clamp_min(0))
        # _input_env still works through the serial fallback
        x0, env = mod._input_env((xv,))
        assert x0 is xv and env["x"] is xv
        gen = ir_to_torch_module(ir)
        torch.testing.assert_close(mod(xv), gen(xv))
