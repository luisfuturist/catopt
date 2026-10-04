"""Intake-defect regressions — lowering fidelity the workload intake caught.

``tools/law_intake.py`` ingests real ``nn.Module``s through the export
boundary and verifies the lowered ``IRModule`` fp64.  Three torch-native
modules exported + bound cleanly but verified *wrong* — lowering
defects no hand-built corpus model had exercised:

* ``nn.MultiheadAttention`` (distinct q/k/v — the cross-attention
  path) splits the packed (3E, E) in-proj weight with
  ``aten.chunk(w, 3)``.  aten elides ``dim`` at its default (0), and
  the boundary recorded nothing — the binding's minted-term default
  (last axis) silently split the WRONG axis.
* ``nn.TransformerDecoderLayer`` spells the same split as
  ``aten.split_with_sizes(w, [E, 2E])`` — same implicit ``dim=0``
  dropped, same wrong axis.
* ``nn.LSTM`` mints ``h0``/``c0`` via ``aten.zeros(shape,
  dtype=torch.float64)`` — the kwargs walk dropped ``torch.dtype``
  values wholesale, so the lowered ``torch.zeros(shape)`` minted an
  fp32 tensor under an fp64 program.

Each case exports + lowers + verifies through the same
``TorchSink.lower``/``verify`` path ``law_intake._verify`` runs.
"""

import torch
from catopt_core.ir import IR, Op
from catopt_torch.adapters import TorchSink
from catopt_torch.torch_bridge import export_to_ir, ir_to_torch_module


def _ops_of(term, acc=None):
    if acc is None:
        acc = []
    if isinstance(term, Op):
        acc.append(term)
        for a in term.args:
            _ops_of(a, acc)
    return acc


def _export_lower_verify(model, feed):
    """The intake's boundary pipeline: export → lower → sink.verify."""
    model = model.eval().double()
    ir, tensors = export_to_ir(model, feed)
    lowered = ir_to_torch_module(ir, param_values=tensors)
    vr = TorchSink().verify(model, lowered, feed)
    return ir, vr


def test_mha_packed_inproj_chunk_dim():
    """Distinct q/k/v: the packed in-proj ``chunk`` splits dim 0."""
    d = 16
    feed = (
        torch.randn(2, 8, d, dtype=torch.float64),
        torch.randn(2, 8, d, dtype=torch.float64),
        torch.randn(2, 8, d, dtype=torch.float64),
    )
    model = torch.nn.MultiheadAttention(d, 4, batch_first=True)
    ir, vr = _export_lower_verify(model, feed)
    chunks = [t for t in _ops_of(ir.root) if t.op == "chunk"]
    # The packed in-proj weight/bias each chunk into q/k/v thirds —
    # aten's implicit dim=0 must be recorded, not guessed.
    assert chunks, "expected the packed in-proj chunk"
    assert all(t.attrs.get("dim") == 0 for t in chunks)
    assert vr.passed, vr


def test_decoder_layer_split_with_sizes_dim():
    """``split_with_sizes`` carries the same implicit dim=0."""
    d = 16
    feed = (
        torch.randn(2, 8, d, dtype=torch.float64),
        torch.randn(2, 8, d, dtype=torch.float64),
    )
    model = torch.nn.TransformerDecoderLayer(
        d, 4, 2 * d, batch_first=True, dropout=0.0
    )
    ir, vr = _export_lower_verify(model, feed)
    splits = [t for t in _ops_of(ir.root) if t.op == "split"]
    assert splits, "expected the cross-attn in-proj split"
    assert all(t.attrs.get("dim") == 0 for t in splits)
    assert vr.passed, vr


def test_lstm_creator_dtype():
    """``aten.zeros(..., dtype=float64)`` must not lower to fp32."""
    model = torch.nn.LSTM(8, 8, batch_first=True)
    feed = (torch.randn(2, 6, 8, dtype=torch.float64),)
    ir, vr = _export_lower_verify(model, feed)
    zeros = [t for t in _ops_of(ir.root) if t.op == "zeros"]
    assert zeros, "expected the h0/c0 zeros ops"
    # The exported dtype lands as a string attr the binding resolves.
    assert all(t.attrs.get("dtype") == "float64" for t in zeros)
    assert vr.passed, vr


def test_exported_split_chunk_dim_explicit_not_overwritten():
    """An explicit dim survives — the injection only fills the elided."""

    class M(torch.nn.Module):
        def forward(self, x):
            a, b = x.chunk(2, dim=1)
            return a * b

    ir, _ = export_to_ir(M().eval(), (torch.randn(2, 8),))
    chunks = [t for t in _ops_of(ir.root) if t.op == "chunk"]
    assert chunks
    assert all(t.attrs.get("dim") == 1 for t in chunks)


def test_creator_dtype_attr_forms():
    """``_creator_dtype`` accepts every spelling the boundary emits.

    Exported terms carry the short string ("float64"); a minted term
    may carry the ``torch.dtype`` object; no attr leaves the torch
    default (fp32).
    """

    def lower(term):
        mod = ir_to_torch_module(
            IR(root=term, inputs=[], input_names=set(), params={})
        )
        return mod()

    z_str = lower(Op.make("zeros", shape=(2, 2), dtype="float64"))
    assert z_str.dtype == torch.float64
    z_obj = lower(Op.make("zeros", shape=(2, 2), dtype=torch.float64))
    assert z_obj.dtype == torch.float64
    z_none = lower(Op.make("zeros", shape=(2, 2)))
    assert z_none.dtype == torch.get_default_dtype()
    # ``*_like`` inherits the operand dtype when no attr is present.
    zl = lower(
        Op.make(
            "zeros_like",
            Op.make("full", shape=(2, 2), dtype="float64"),
        )
    )
    assert zl.dtype == torch.float64


def test_sink_verify_tuple_output_modules():
    """``verify`` compares the modelled first output of tuple modules.

    ``export_to_ir`` keeps the exported graph's first output; an
    ``nn.MultiheadAttention`` (attn_out, weights) or ``nn.LSTM``
    (out, (h_n, c_n)) ref output must not crash the gate — this is
    what let the intake classify them at all.
    """
    d = 16
    feed = (
        torch.randn(2, 8, d, dtype=torch.float64),
        torch.randn(2, 8, d, dtype=torch.float64),
        torch.randn(2, 8, d, dtype=torch.float64),
    )
    model = (
        torch.nn.MultiheadAttention(d, 4, batch_first=True)
        .eval()
        .double()
    )
    ir, tensors = export_to_ir(model, feed)
    lowered = ir_to_torch_module(ir, param_values=tensors)
    vr = TorchSink().verify(model, lowered, feed)
    assert vr.passed, vr
