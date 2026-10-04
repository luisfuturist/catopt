"""Corpus-expansion models — export and lowering fidelity.

The law-discovery pipeline reads ``catopt_torch.models`` as its real-
world corpus (``catopt_discovery.impact.model_cases``).  These builders add
architecture families the corpus lacked — soft MoE dispatch, GEGLU,
gated residuals, conv norm blocks, manual-softmax attention, learned
positional embedding, kernelized attention, and the round-2 index /
signal / routing families (top-k routing, causal conv1d, sinusoidal
PE, in-graph mask construction, one-hot dispatch, VQ codebook,
maxout, GLU, native ``rms_norm``) — so the shape census sees new
op-tuples (``sum``, ``exp``, ``gelu``, ``relu``, ``batch_norm``,
``arange``, ``embedding``, ``elu``, ``topk``, ``gather``, ``conv1d``,
``pad``, ``sin``, ``cos``, ``tril``, ``argmax``, ``one_hot``,
``argmin``, ``index_select``, ``maximum``, ``glu``, ``rms_norm``)
— plus the manual-GLU spelling (``chunk`` + ``sigmoid`` + ``mul``)
that pairs with :class:`GluMLP`'s kernel image for ``glu_fold``.

Every builder is exercised end-to-end: forward, ``export_to_ir`` and a
lowered ``IRModule`` verified fp64 against the original module.
"""

import torch
from catopt_core.ir import Op
from catopt_torch.models import (
    CodebookQuantizer,
    ConvNeXtBlock,
    DepthwiseConvBlock,
    GatedResidualBlock,
    GegluMLP,
    GluMLP,
    HardDispatch,
    KernelizedAttention,
    ManualGluMLP,
    ManualSoftmaxAttention,
    MaxoutMLP,
    MoEMLP,
    NativeRmsNorm,
    PositionalEmbedding,
    ResNetBlock,
    SinusoidalEncoding,
    TopKRouter,
    TrilCausalAttention,
    Wav2VecBlock,
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
WAV = torch.randn(1, 8, 16, dtype=torch.float64)

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
    (
        "TopKRouter",
        TopKRouter(d, 4, 2),
        VEC,
        {"topk", "getitem", "gather"},
    ),
    (
        "Wav2VecBlock",
        Wav2VecBlock(8),
        WAV,
        {"conv1d", "pad", "relu"},
    ),
    (
        "SinusoidalEncoding",
        SinusoidalEncoding(d),
        SEQ,
        {"sin", "cos"},
    ),
    (
        "TrilCausalAttention",
        TrilCausalAttention(d),
        SEQ,
        {"tril", "ones", "masked_fill"},
    ),
    (
        "HardDispatch",
        HardDispatch(d, 4),
        VEC,
        {"argmax", "one_hot"},
    ),
    (
        "CodebookQuantizer",
        CodebookQuantizer(d, 8),
        VEC,
        {"argmin", "index_select"},
    ),
    ("MaxoutMLP", MaxoutMLP(d, 2), VEC, {"maximum"}),
    ("GluMLP", GluMLP(d, 2), VEC, {"glu"}),
    (
        "ManualGluMLP",
        ManualGluMLP(d, 2),
        VEC,
        {"chunk", "sigmoid", "mul"},
    ),
    ("NativeRmsNorm", NativeRmsNorm(d), VEC, {"rms_norm"}),
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
