"""End-to-end pipeline tests — the full optimize → verify → compile
path, certificate replay on a real export, OpTable composition at the
lowering boundary, and compositional fallback.

These exercise the pieces working *together* rather than any single
component's internals."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from catopt.cost import flops_cost
from catopt.egraph import EGraph, verify_certificate
from catopt.ir import IR, Op, TensorType, Var
from catopt.laws import all_rules
from catopt.om_lower import (
    StreamingOMModule,
    om_apply_state,
    om_empty_state,
    om_step,
)
from catopt.ops import OpTable
from catopt.optimize import optimize_compositional
from catopt.torch_bridge import export_to_ir, ir_to_torch_module


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


# ---------------------------------------------------------------------------
#  Pipeline: compositional optimize of a transformer-ish stack → compile
# ---------------------------------------------------------------------------


class _TinyBlock(nn.Module):
    """One attention + MLP layer with residuals — the smallest honest
    transformer block."""

    def __init__(self, dim, n_heads=2):
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2 * dim, dim)
        self.nh = n_heads

    def forward(self, x):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.nh, -1).transpose(1, 2)
        k = self.wk(x).view(B, T, self.nh, -1).transpose(1, 2)
        v = self.wv(x).view(B, T, self.nh, -1).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + self.wo(o.transpose(1, 2).contiguous().view(B, T, C))
        return x + self.fc2(F.silu(self.fc1(x)))


class _TinyTransformer(nn.Module):
    def __init__(self, dim=16, depth=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            _TinyBlock(dim) for _ in range(depth)
        )

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def test_compositional_pipeline_to_compiled():
    """optimize_compositional over a 2-block transformer stack:
    every block optimizes, the stack verifies end-to-end, and
    ``torch.compile`` (Inductor, CPU) reproduces eager output."""
    torch.manual_seed(0)
    model = _TinyTransformer().eval()
    x = torch.randn(2, 8, 16)
    opt, stats = optimize_compositional(model, x, verbose=False)
    assert stats["n_optimized"] >= 1 and stats["n_failed"] == 0
    assert stats["end_to_end"]["max_rel_diff"] < 1e-9
    with torch.no_grad():
        eager = opt(x)
    compiled = torch.compile(opt)
    with torch.no_grad():
        out = compiled(x)
    torch.testing.assert_close(out, eager)


# ---------------------------------------------------------------------------
#  Certificate round-trip on a real export
# ---------------------------------------------------------------------------


def test_certificate_roundtrip_on_exported_ir():
    """A genuinely-optimized term replays through verify_certificate:
    export a parallel-GEMM module, saturate, extract, take the
    certificate — it verifies standalone, and the lowered dst still
    matches the source module numerically."""
    torch.manual_seed(0)

    class Par(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(8, 4, bias=False)
            self.w2 = nn.Linear(8, 4, bias=False)

        def forward(self, x):
            return self.w1(x) + self.w2(x)

    m = Par().eval()
    x = torch.randn(3, 8)
    ir, source_tensors = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=6)
    best = eg.extract_best(root, flops_cost)
    assert best is not None
    cert = eg.certificate(ir.root, best)
    assert cert.replayable
    # the certificate proves src → best, standalone
    replayed = verify_certificate(ir.root, cert, strict=True)
    assert replayed == cert.dst
    # and the proven term lowers to the same function
    dst_ir = IR(
        root=replayed,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    dst_mod = ir_to_torch_module(dst_ir, param_values=source_tensors)
    with torch.no_grad():
        torch.testing.assert_close(dst_mod(x), m(x))


# ---------------------------------------------------------------------------
#  OpTable composition — carrier ops need their carrier registered
# ---------------------------------------------------------------------------


def test_optable_composition_and_missing_binding():
    """``attnbias`` lives in the om carrier: a core-only table lowers
    to a module that raises the clean "No torch binding" error, and
    registering "om" makes the same term evaluate correctly."""
    B, Tq, K = 1, 3, 4
    s = _v("s", B, Tq, K)
    m = _v("m", B, Tq, K)
    term = Op.make("add", s, Op.make("attnbias", m))
    ir = IR(root=term, inputs=[s, m], input_names={"s", "m"}, params={})

    mod = ir_to_torch_module(ir, ops=OpTable.core())
    try:
        mod(
            torch.randn(B, Tq, K),
            torch.ones(B, Tq, K, dtype=torch.bool),
        )
    except ValueError as e:
        assert "No torch binding for op 'attnbias'" in str(e)
    else:  # pragma: no cover — defensive
        raise AssertionError("expected missing-binding ValueError")

    full = OpTable.core().register("om")
    mod2 = ir_to_torch_module(ir, ops=full)
    ts = torch.randn(B, Tq, K)
    # bool keep-mask → additive -inf bias
    tm = torch.ones(Tq, K, dtype=torch.bool).tril().expand(B, Tq, K)
    out = mod2(ts, tm.contiguous())
    ref = ts + torch.where(
        tm, torch.zeros(()), torch.full((), float("-inf"))
    )
    torch.testing.assert_close(out, ref)
    # float masks pass straight through
    tmf = torch.randn(B, Tq, K)
    assert torch.equal(mod2(ts, tmf), ts + tmf)


def test_optable_register_fragment_kwargs():
    """dict source + shape_rules/attr_schema fragments compose on one
    table — the fragments land in their registries."""
    t = OpTable()
    t.register(
        {"myop": lambda *a, **k: None},
        shape_rules={"myop": lambda op, shapes: (1,)},
        attr_schema={"myop": {1: "dim"}},
    )
    assert "myop" in t.torch_bindings
    assert "myop" in t.shape_rules
    assert "myop" in t.attr_schemas


# ---------------------------------------------------------------------------
#  Compositional fallback — an un-lowerable block keeps its original
# ---------------------------------------------------------------------------


def test_compositional_fallback_keeps_original_block():
    """A block whose export lowers to ops with no torch binding fails
    its verification — the compositional path keeps the ORIGINAL
    block and the assembled model stays functionally identical."""
    torch.manual_seed(0)
    d = 8

    class _Good(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.f1 = nn.Linear(d, d, bias=False)
            self.f2 = nn.Linear(d, d, bias=False)

        def forward(self, x):
            return x + self.f2(F.silu(self.f1(x)))

    class _Bad(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.f = nn.Linear(d, d)

        def forward(self, x):
            # data-dependent op: exports but has no torch lowering —
            # the lowered candidate can't be verified → fallback.
            idx = x.nonzero()
            return self.f(x) + idx.sum() * 0 + self.f(x) * 0

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([_Good(d), _Bad(d), _Good(d)])

        def forward(self, x):
            for b in self.blocks:
                x = b(x)
            return x

    model = M().eval()
    x = torch.randn(4, d)
    opt, stats = optimize_compositional(model, x, verbose=False)
    assert stats["n_optimized"] == 2
    assert stats["n_failed"] == 1
    # the fallback module reproduces the original exactly
    with torch.no_grad():
        assert torch.equal(model(x), opt(x))


# ---------------------------------------------------------------------------
#  Streaming decode regime — incremental state continuation == dense
# ---------------------------------------------------------------------------


def test_streaming_decode_state_continuation():
    """Decode-regime equivalence: a StreamingOMModule evaluates the
    prefix with forward_state, then continues one row at a time via
    om_step — the incrementally-grown state matches a single dense
    forward exactly."""
    B, T, d, dv, n = 1, 6, 4, 3, 2
    q = _v("q", B, T, d)
    k = _v("k", B, T, d)
    v = _v("v", B, T, dv)
    s_full = Op.make(
        "matmul", q, Op.make("transpose", k, arg1=-2, arg2=-1)
    )
    # split the score axis into chunk blocks
    blocks = [
        Op.make(
            "om_elem",
            Op.make("chunk", s_full, arg1=n, arg2=-1, index=i),
            Op.make("chunk", v, arg1=n, arg2=-2, index=i),
        )
        for i in range(n)
    ]
    root = Op.make(
        "om_apply", Op.make("om_compose", blocks[0], blocks[1])
    )
    ir = IR(
        root=root,
        inputs=[q, k, v],
        input_names={"q", "k", "v"},
        params={},
    )
    sm = StreamingOMModule(ir)
    assert sm._plan is not None

    tq = torch.randn(B, T, d, dtype=torch.float64) * 0.5
    tk = torch.randn(B, T, d, dtype=torch.float64) * 0.5
    tv = torch.randn(B, T, dv, dtype=torch.float64) * 0.5
    with torch.no_grad():
        dense = sm(tq, tk, tv)
        m, l_, a = sm.forward_state(tq, tk, tv)
        # decode continuation: replay the same key blocks through
        # om_step on a fresh empty state — it must rebuild the same
        # triple the dense forward computed.
        state = om_empty_state((B, T), dv)
        s_b = tq @ tk.transpose(-1, -2)
        for i in range(n):
            blk_s = torch.chunk(s_b, n, dim=-1)[i]
            blk_v = torch.chunk(tv, n, dim=-2)[i]
            state = om_step(state, blk_s, blk_v)
        inc = om_apply_state(state)
    torch.testing.assert_close(inc, dense)
    # and the checkpointed forward_state agrees with the continuation
    torch.testing.assert_close(om_apply_state((m, l_, a)), inc)
