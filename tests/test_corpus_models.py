"""Corpus-expansion models — export and lowering fidelity.

The law-discovery pipeline reads ``catopt_torch.models`` as its real-
world corpus (``tools/law_impact.model_cases``).  These builders add
architecture families the corpus lacked — soft MoE dispatch, GEGLU,
gated residuals, conv norm blocks, manual-softmax attention, learned
positional embedding and kernelized attention — so the shape census
sees new op-tuples (``sum``, ``exp``, ``gelu``, ``relu``,
``batch_norm``, ``arange``, ``embedding``, ``elu``).

Every builder is exercised end-to-end: forward, ``export_to_ir`` and a
lowered ``IRModule`` verified fp64 against the original module.
"""

import torch
from catopt_core.ir import Op
from catopt_torch.models import (
    ConvNeXtBlock,
    DepthwiseConvBlock,
    GatedResidualBlock,
    GegluMLP,
    KernelizedAttention,
    ManualSoftmaxAttention,
    MoEMLP,
    PositionalEmbedding,
    ResNetBlock,
)
from catopt_torch.torch_bridge import export_to_ir, ir_to_torch_module


def _ops(term, acc):
    if id(term) in acc[1]:
        return
    acc[1].add(id(term))
    if isinstance(term, Op):
        acc[0].add(term.op)
        for a in term.args:
            _ops(a, acc)


def _term_ops(term):
    acc = (set(), set())
    _ops(term, acc)
    return acc[0]


d = 16
VEC = torch.randn(4, d, dtype=torch.float64)
SEQ = torch.randn(2, 8, d, dtype=torch.float64)
IMG = torch.randn(1, 8, 4, 4, dtype=torch.float64)

CASES = [
    ("MoEMLP", MoEMLP(d, 32, 3), VEC, {"softmax", "stack", "sum"}),
    (
        "MoEMLP-default-hidden",
        MoEMLP(d),
        VEC,
        {"softmax", "stack", "sum"},
    ),
    ("GegluMLP", GegluMLP(d, 2), VEC, {"gelu"}),
    (
        "GatedResidualBlock",
        GatedResidualBlock(d),
        VEC,
        {"sigmoid", "sub"},
    ),
    ("ResNetBlock", ResNetBlock(8), IMG, {"conv2d", "batch_norm"}),
    (
        "DepthwiseConvBlock",
        DepthwiseConvBlock(8),
        IMG,
        {"conv2d"},
    ),
    (
        "ConvNeXtBlock",
        ConvNeXtBlock(8),
        IMG,
        {"conv2d", "group_norm", "gelu"},
    ),
    (
        "ManualSoftmaxAttention",
        ManualSoftmaxAttention(d),
        SEQ,
        {"exp", "sum", "div"},
    ),
    (
        "PositionalEmbedding",
        PositionalEmbedding(64, d),
        SEQ,
        {"arange", "embedding"},
    ),
    (
        "KernelizedAttention",
        KernelizedAttention(d),
        SEQ,
        {"elu"},
    ),
]


def test_corpus_models_export_expected_ops():
    """Each builder exports and carries its signature op family."""
    for name, model, x, expected in CASES:
        model = model.eval().double()
        ir, _ = export_to_ir(model, x)
        ops = _term_ops(ir.root)
        missing = expected - ops
        assert not missing, (
            f"{name}: missing {missing} in {sorted(ops)}"
        )


def test_corpus_models_lowered_module_matches():
    """The lowered IRModule reproduces the module output fp64."""
    for name, model, x, _expected in CASES:
        model = model.eval().double()
        with torch.no_grad():
            expected = model(x)
        ir, tensors = export_to_ir(model, x)
        mod = ir_to_torch_module(ir, param_values=tensors)
        with torch.no_grad():
            got = mod(x)
        assert torch.allclose(got, expected, atol=1e-9), name


def test_corpus_models_forward_runs():
    """Plain forwards run under eval+no_grad (coverage for __init__/forward)."""
    for name, model, x, _expected in CASES:
        model = model.eval().double()
        with torch.no_grad():
            out = model(x)
        assert out.numel() > 0, name
