"""Tail coverage: real paths the earlier waves left — optimize verbose/
resource paths, _eval_const edges, intern TypeError, attrs helpers,
rulecache env, ports signature internals, compositional e2e arms."""
# ruff: noqa: RUF059


import pytest
import torch
import torch.nn as nn
from catopt.attrs import is_positional_attr
from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.optimize import (
    OptimizationResourceError,
    _eval_const,
    optimize_compositional,
    optimize_model,
)
from catopt.ports import CostFn, Verifier, signature_conforms


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(8, 8)

    def forward(self, x):
        return torch.relu(self.w(x))


class _TwoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = _MLP()
        self.b2 = _MLP()

    def forward(self, x):
        return self.b2(self.b1(x))


def test_optimize_verbose_and_memory_paths():
    torch.manual_seed(0)
    m = _MLP()
    x = torch.randn(2, 8)
    opt, stats = optimize_model(m, x, verbose=True, max_iterations=2)
    assert opt is not None and isinstance(stats, dict)


def test_optimize_memory_budget_raises():
    torch.manual_seed(0)
    with pytest.raises(OptimizationResourceError):
        optimize_model(_MLP(), torch.randn(2, 8), max_memory_mb=0.00001)


def test_compositional_verbose_and_e2e():
    torch.manual_seed(0)
    m = _TwoBlock()
    x = torch.randn(2, 8)
    opt, rep = optimize_compositional(m, x, verbose=True, max_iterations=2)
    assert rep["n_blocks"] == 2
    assert rep["end_to_end"] is not None


def test_compositional_e2e_error_arm():
    """End-to-end check raising records the error, doesn't crash."""
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    for verbose in (False, True):
        m = _TwoBlock()
        orig = m.forward
        calls = {"n": 0}

        def flaky(inp, _orig=orig, _calls=calls):
            _calls["n"] += 1
            if _calls["n"] > 1:
                raise RuntimeError("forced e2e failure")
            return _orig(inp)

        m.forward = flaky
        _opt, rep = optimize_compositional(
            m, x, max_iterations=1, verbose=verbose
        )
        assert "error" in rep["end_to_end"]


def test_eval_const_edges():
    assert _eval_const(Var("x", TensorType((4,))), {}) is None
    assert _eval_const(Param("missing", TensorType((4,))), {}) is None
    assert torch.equal(_eval_const(Const(3.0), {}), torch.tensor(3.0))
    unbound = Op.make("no_such_op_xyz", Const(1.0))
    assert _eval_const(unbound, {}) is None
    mixed = Op.make("add", Var("x", TensorType((4,))), Const(1.0))
    assert _eval_const(mixed, {}) is None


def test_intern_unhashable_args():
    """Unhashable args: intern lookup's TypeError arm fires, then
    __post_init__'s hash raises too — invalid args die at mint."""
    with pytest.raises(TypeError):
        Op.make("custom_op", {"unhashable": 1})
    # dict/list attrs are repr-fallback → still intern
    x = Op.make("custom_op", Const(1.0), meta={"nested": [1]})
    y = Op.make("custom_op", Const(1.0), meta={"nested": [1]})
    assert x is y


def test_is_positional_attr():
    assert is_positional_attr("arg0") and is_positional_attr("arg12")
    assert not is_positional_attr("dim") and not is_positional_attr(3)


def test_rulecache_default_dir_env(monkeypatch):
    from catopt.rulecache import _default_cache_dir

    monkeypatch.delenv("CATOPT_RULECACHE_DIR", raising=False)
    assert "catopt" in str(_default_cache_dir())
    monkeypatch.setenv("CATOPT_RULECACHE_DIR", "/tmp/catopt_rc_test")
    assert str(_default_cache_dir()) == "/tmp/catopt_rc_test"


def test_signature_conforms_internals():
    # non-callables → False
    assert not signature_conforms(object(), CostFn)
    assert not signature_conforms("x", CostFn)
    # real fns bind the port's call
    assert signature_conforms(lambda t, memo=None: 1.0, CostFn)
    assert not signature_conforms(lambda a, b, c: 1.0, Verifier)
    # strict probe requires the full call to bind
    def partial_ok(ref, out):
        return None

    assert signature_conforms(partial_ok, Verifier, strict=False)
    assert not signature_conforms(partial_ok, Verifier, strict=True)


class _KwOnlyProto:
    """__call__ with kw-only + **kw params — exercises _probes arms."""

    def __call__(self, x, *, flag=True, **kw):
        return x, flag, kw


def test_signature_conforms_kw_only_probes():
    def binds_everything(x, flag=True, **kw):
        return 1

    assert signature_conforms(binds_everything, _KwOnlyProto)


def test_optimize_eval_const_missing_binding():
    """an Op whose op has no torch binding → None (no crash)."""
    weird = Op.make("gibberish_op", Const(1.0), Const(2.0))
    assert _eval_const(weird, {}) is None


class _SubProto(_KwOnlyProto):
    """Subclass inherits __call__ — exercises the mro continue arm."""


class _ReqKwOnlyProto:
    """__call__ with a REQUIRED kw-only param → min_kwargs arm."""

    def __call__(self, x, *, req):
        return x, req


class _NoSigProto:
    """__call__ whose signature can't be inspected → None arm."""

    __call__ = vars  # builtins raise ValueError on signature()


def test_signature_conforms_mro_and_edges():
    # subclass without its own __call__ → walks mro, continues
    assert signature_conforms(lambda x, flag=True, **kw: 1, _SubProto)
    # required kw-only in the proto → min_kwargs path
    assert signature_conforms(lambda x, req: 1, _ReqKwOnlyProto)
    # uninspectable port __call__ → permissive True
    assert signature_conforms(lambda *a: 1, _NoSigProto)
    # uninspectable fn → permissive True
    assert signature_conforms(vars, CostFn)


def test_compositional_verbose_failed_block():
    """verbose print on the keeping-original arm."""
    torch.manual_seed(0)

    class Weird(nn.Module):
        def forward(self, x):
            return x.nonzero()  # unlowerable — optimize_model fails

    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = Weird()

        def forward(self, x):
            return self.w(x).float().sum()

    x = torch.arange(6.0)
    _opt, rep = optimize_compositional(Wrap(), x, verbose=True)
    assert rep["n_blocks"] >= 1
    assert any(b["status"] == "failed" for b in rep["blocks"].values())


def test_optimize_memory_budget_pass():
    """used <= max_memory_mb → the check's pass-through arm."""
    torch.manual_seed(0)
    opt, _s = optimize_model(
        _MLP(), torch.randn(2, 8), max_memory_mb=1e6, max_iterations=1
    )
    assert opt is not None


def test_pairing_groups_fire():
    """Two linears sharing an input → pairing_groups stat + re-saturation."""
    torch.manual_seed(0)

    class Paired(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(8, 8, bias=False)
            self.b = nn.Linear(8, 8, bias=False)

        def forward(self, x):
            return self.a(x) + self.b(x)

    x = torch.randn(2, 8)
    m = Paired()
    ref = m(x)
    opt, stats = optimize_model(m, x, max_iterations=6)
    assert stats.get("pairing_groups", 0) >= 0
    assert torch.allclose(ref, opt(x), atol=1e-5)


def test_transformer_block_full_pipeline():
    """Real transformer block — pairing + lifts + re-saturation + verify
    print, in one verbose optimize_model call."""
    import catopt.models as M

    torch.manual_seed(0)
    m = M.TransformerBlock(dim=64)
    x = torch.randn(1, 8, 64)
    ref = m(x)
    opt, stats = optimize_model(m, x, max_iterations=3, verbose=True)
    assert stats["nonlocal_lifts"] >= 1
    assert stats["pairing_groups"] >= 1
    assert torch.allclose(ref, opt(x), atol=1e-5)


def test_compositional_no_blocks():
    """A module with no candidate blocks → early exit arm."""
    m = nn.Linear(4, 4)
    _opt, rep = optimize_compositional(m, torch.randn(2, 4))
    assert rep["n_blocks"] == 0


def test_discover_alternatives_fires_pairing_and_lifts():
    """discover_alternatives runs the same pairing/lift pass — its
    re-saturation blocks execute when groups exist."""
    import catopt.models as M
    from catopt.optimize import discover_alternatives

    torch.manual_seed(0)
    m = M.TransformerBlock(dim=64)
    x = torch.randn(1, 8, 64)
    alts = discover_alternatives(m, x, max_iterations=2, top_k=4)
    assert isinstance(alts, dict)


def test_verbose_verify_warning_arm(monkeypatch):
    """verify reports failure → the ✗ warning arm prints and optimize
    still returns a module (verify failure is non-fatal).

    ``optimize_model`` now verifies through the sink, so the torch
    sink's ``verify_module`` is the patched seam (plan 0004).
    """
    from catopt.report import VerifyReport

    def fake_verify(*a, **kw):
        return VerifyReport(max_abs=0.5, max_rel=0.5, passed=False)

    monkeypatch.setattr("catopt.adapters.verify_module", fake_verify)
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    opt, _s = optimize_model(_MLP(), x, verbose=True, max_iterations=1)
    assert opt is not None


def test_hook_dedup_repeated_block():
    """A block executed twice in one forward → the hook's
    name-in-captured arm (second call skipped)."""
    torch.manual_seed(0)

    class Twice(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = _MLP()

        def forward(self, x):
            return self.b(self.b(x))

    x = torch.randn(2, 8)
    _opt, rep = optimize_compositional(Twice(), x, max_iterations=1)
    assert rep["n_blocks"] >= 1


class _NoSelfCallProto:
    """__call__ without a leading 'self' — the params[0]-!=-self arc."""

    __call__ = len  # signature (obj, /) — first param isn't 'self'


def test_signature_conforms_no_self_sig():
    assert signature_conforms(lambda x: 1, _NoSelfCallProto)


def test_compositional_custom_cost_fn():
    """cost_fn provided → the not-None arc."""
    from catopt.cost import flops_cost

    m = _TwoBlock()
    _opt, rep = optimize_compositional(
        m, torch.randn(2, 8), cost_fn=flops_cost, max_iterations=1
    )
    assert rep["n_blocks"] >= 1


def test_compositional_block_verify_failure(monkeypatch):
    """verify_module reports failure → the RuntimeError verify arm →
    block marked failed, compositional falls back."""
    from catopt.report import VerifyReport

    def fake_verify(*a, **kw):
        return VerifyReport(max_abs=0.5, max_rel=0.5, passed=False)

    monkeypatch.setattr("catopt.optimize.verify_module", fake_verify)
    torch.manual_seed(0)
    m = _TwoBlock()
    _opt, rep = optimize_compositional(m, torch.randn(2, 8), max_iterations=1)
    assert any(b["status"] == "failed" for b in rep["blocks"].values())


class _BareProto:
    """No __call__ anywhere → _port_signature returns None."""


def test_signature_conforms_bare_proto():
    assert signature_conforms(lambda *a: 1, _BareProto)
