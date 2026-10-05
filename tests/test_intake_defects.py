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


# ---------------------------------------------------------------------------
#  Round 2 — in-place writes threaded like copy_, dropped dtype casts
# ---------------------------------------------------------------------------


class _SliceFill(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y[:, :4] = 0.0
        return y


class _FillWhole(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y.fill_(0.5)
        return y


class _FillSelect(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y[0].fill_(0.0)
        return y


class _ZeroInit(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y.zero_()
        return y


class _MaskedFillInplace(torch.nn.Module):
    def forward(self, x):
        y = x.clone()
        y.masked_fill_(x > 0, 0.0)
        return y


class _MaskedFillTensor(torch.nn.Module):
    def forward(self, x, v):
        y = x.clone()
        y.masked_fill_(x > 0, v)
        return y


class _OneHotMatmul(torch.nn.Module):
    def forward(self, idx):
        oh = torch.nn.functional.one_hot(idx, 16).to(torch.float64)
        return oh @ torch.eye(16, dtype=torch.float64)


class _TakeAlong(torch.nn.Module):
    def forward(self, x):
        idx = x.argsort(dim=-1, descending=True)[..., :4]
        return x.take_along_dim(idx, dim=-1)


class _Rsub(torch.nn.Module):
    def forward(self, x):
        return 1.0 - torch.sigmoid(x)


class _Gammaln(torch.nn.Module):
    def forward(self, x):
        return torch.special.gammaln(x.abs() + 1.0)


class _InfixBitwise(torch.nn.Module):
    def forward(self, x, y):
        return ((x & y) | (x | y)).to(torch.float64)


def test_fill_tensor_through_slice_view():
    """``y[:, :4] = 0`` — ``fill_.Tensor`` through a ``slice`` view.

    Functionalisation spells a slice write as ``fill_(slice, lifted
    0-dim)``; the mutation threader must scatter the broadcast value
    onto the viewed base, not orphan the write.
    """
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_SliceFill(), feed)
    ops = {t.op for t in _ops_of(ir.root)}
    assert "slice_scatter" in ops and "broadcast_to" in ops
    assert vr.passed, vr


def test_fill_scalar_whole_tensor():
    """``y.fill_(0.5)`` — ``fill_.Scalar`` on a whole tensor."""
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_FillWhole(), feed)
    ops = {t.op for t in _ops_of(ir.root)}
    assert "full" in ops and "copy" in ops
    assert vr.passed, vr


def test_fill_scalar_through_select_view():
    """``y[0].fill_(0)`` — ``fill_.Scalar`` through ``select``."""
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_FillSelect(), feed)
    ops = {t.op for t in _ops_of(ir.root)}
    assert "select_scatter" in ops and "full" in ops
    assert vr.passed, vr


def test_zero_inplace_is_a_fill():
    """``y.zero_()`` — the zero-arg spelling of an in-place fill."""
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_ZeroInit(), feed)
    ops = {t.op for t in _ops_of(ir.root)}
    assert "zeros" in ops and "copy" in ops
    assert vr.passed, vr


def test_masked_fill_inplace_scalar():
    """``masked_fill_.Scalar`` — mints the functional masked_fill."""
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_MaskedFillInplace(), feed)
    ops = {t.op for t in _ops_of(ir.root)}
    assert "masked_fill" in ops
    assert vr.passed, vr


def test_masked_fill_inplace_tensor_value():
    """``masked_fill_.Tensor`` — the tensor-valued fill spelling."""
    feed = (
        torch.randn(4, 16, dtype=torch.float64),
        torch.randn((), dtype=torch.float64),
    )
    ir, vr = _export_lower_verify(_MaskedFillTensor(), feed)
    ops = {t.op for t in _ops_of(ir.root)}
    assert "masked_fill" in ops
    assert vr.passed, vr


def test_to_positional_dtype_cast():
    """``to``'s positional ScalarType must survive to the binding.

    ``one_hot(...)`` returns int64; the ``.to(torch.float64)`` cast is
    positional in ``aten.to.dtype`` and used to drop silently — the
    lowered ``matmul`` then saw long vs float and died.  Now the dtype
    lands as an attr and the binding applies the real cast (the
    carrier ``eye`` binding honours it too).
    """
    feed = (torch.randint(0, 16, (4, 8)),)
    ir, vr = _export_lower_verify(_OneHotMatmul(), feed)
    tos = [t for t in _ops_of(ir.root) if t.op == "to"]
    assert tos and all(t.attrs.get("dtype") == "float64" for t in tos)
    assert vr.passed, vr


def test_to_no_dtype_is_identity():
    """A ``to`` term without a recorded dtype stays an identity."""
    x = Op.make("full", shape=(2, 3), dtype="float64")

    def lower(term):
        mod = ir_to_torch_module(
            IR(root=term, inputs=[], input_names=set(), params={})
        )
        return mod()

    out = lower(Op.make("to", x))
    assert out.shape == (2, 3) and out.dtype == torch.float64


def test_take_along_dim_positional_dim():
    """aten ``take_along_dim(t, indices, dim)`` puts dim at arg 2.

    The schema declared position 1 (``gather``'s layout); the exported
    int landed as an unschema'd ``arg2`` and died at ``Op.make``.
    """
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_TakeAlong(), feed)
    tas = [t for t in _ops_of(ir.root) if t.op == "take_along_dim"]
    assert tas and all(t.attrs.get("dim") == -1 for t in tas)
    assert vr.passed, vr


def test_rsub_scalar_canonicalises_and_lowers():
    """``1 - x`` exports as ``rsub.Scalar`` — canonicalised + bound."""
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_Rsub(), feed)
    assert any(t.op == "rsub" for t in _ops_of(ir.root))
    assert vr.passed, vr


def test_special_gammaln_canonicalises():
    """``torch.special.gammaln`` exports as ``aten.special_gammaln``."""
    feed = (torch.randn(4, 16, dtype=torch.float64),)
    ir, vr = _export_lower_verify(_Gammaln(), feed)
    assert any(t.op == "gammaln" for t in _ops_of(ir.root))
    assert vr.passed, vr


def test_infix_bitwise_canonicalises():
    """``a & b``/``a | b`` spell ``__and__.Tensor``/``__or__.Tensor``."""
    feed = (
        torch.randint(0, 8, (4, 8)),
        torch.randint(0, 8, (4, 8)),
    )
    ir, vr = _export_lower_verify(_InfixBitwise(), feed)
    ops = {t.op for t in _ops_of(ir.root)}
    assert {"bitwise_and", "bitwise_or"} <= ops
    assert vr.passed, vr


def test_eye_honours_exported_dtype():
    """The carrier ``eye`` binding must apply the recorded dtype."""
    eye = Op.make("eye", dim=3, dtype="float64")
    mod = ir_to_torch_module(
        IR(root=eye, inputs=[], input_names=set(), params={})
    )
    assert mod().dtype == torch.float64


def test_copy_family_slice_write_unminted_base_declines():
    """A ``fill_``/``copy_``-family write into a ``slice`` view whose
    BASE was never minted is declined (``env`` unchanged), not
    silently dropped — the write can't be threaded to a base that
    isn't there."""
    from types import SimpleNamespace as NS

    from catopt_torch.torch_bridge import _handle_copy_

    dst_fx = NS(
        name="d",
        target="slice",
        args=(NS(name="unminted"), 0, 0, 4),
        meta={"val": NS(shape=(4,), dtype=torch.float64)},
    )
    env = {"d": Op.make("zeros", shape=(4,), dtype="float64")}
    node = NS(target="fill_.Scalar", args=(dst_fx, 0.0))
    _handle_copy_(node, env)
    assert env["d"] == Op.make("zeros", shape=(4,), dtype="float64")


def test_copy_family_unminted_src_declines():
    """A ``copy_`` whose *source* was never minted is declined —
    the write can't be threaded, and the env is left untouched."""
    from types import SimpleNamespace as NS

    from catopt_torch.torch_bridge import _handle_copy_

    dst_fx = NS(name="d", target="placeholder")
    env = {"d": Op.make("zeros", shape=(4,), dtype="float64")}
    node = NS(target="copy_", args=(dst_fx, NS(name="unminted_src")))
    _handle_copy_(node, env)
    assert env["d"] == Op.make("zeros", shape=(4,), dtype="float64")


# ---------------------------------------------------------------------------
#  Round 3 — Const leaf spelling (``8`` vs ``8.0``) is structural identity
# ---------------------------------------------------------------------------


def test_const_int_float_spellings_are_distinct():
    """``Const(8)`` and ``Const(8.0)`` are different leaves.

    Python's numeric tower makes ``8 == 8.0``; a ``Const`` is a
    structural object (the int-preservation contract in its docstring
    exists because ``x % 2`` needs an int64 operand).  The dataclass
    default ``__eq__``/``__hash__`` leaked that numeric equality into
    structural identity: ``Op.make`` interning returned the
    first-minted spelling, and ``EGraph.add_term``'s content-keyed
    memo mapped the second ``Const`` onto the first's leaf e-class —
    silently rewriting ``* 8.0`` into ``* 8`` in the graph.  The
    e-graph's leaf-key convention (``repr``) is the invariant the
    equality now matches.
    """
    from catopt_core.egraph import EGraph
    from catopt_core.ir import Const

    assert Const(8) != Const(8.0)
    assert Const(8.0) == Const(8.0)
    a = Op.make("mul", Const(8.0), Const(3))
    b = Op.make("mul", Const(8), Const(3))
    assert a is not b and a != b

    eg = EGraph()
    eg.add_term(Op.make("sub", Const(8), Const(8.0)))
    keys = {
        n.attrs[0][1]
        for ec in eg._classes.values()
        for n in ec.nodes
        if n.op == "leaf"
    }
    assert {"8", "8.0"} <= keys


def test_sinc_kernel_certificate_replays():
    """``intake:SincKernel`` — mixed ``8``/``8.0`` leaves certify.

    The Kaiser window spells the scale as ``8.0 *`` (float) and the
    taper as ``t / 8`` (int), so the exported term carries both leaf
    spellings.  Leaf coalescence put ``8.0``'s occurrences in the
    ``8`` e-class, so ``comm_mul``'s recorded binding resolved to
    ``Const(8)`` while the real subterm carries ``8.0`` — replay died
    on "binding b tampered" under the base ruleset (no admitted
    object involved).  With spelling-strict leaves the e-graph keeps
    both classes and the derivation replays end to end.
    """
    from catopt_core.egraph import verify_certificate
    from catopt_core.laws import ALL_RULES
    from catopt_discovery.impact import _cert_ok, _cost_fn, _saturate
    from catopt_discovery.intake import _SincKernel

    d = 16
    feed = (torch.randn(4, d, dtype=torch.float64),)
    model = _SincKernel().eval().double()
    ir, _tensors = export_to_ir(model, feed)
    sink = TorchSink()
    cost_fn = _cost_fn(sink)
    eg, _root, best, _stats = _saturate(ir.root, list(ALL_RULES), cost_fn)
    # The pipeline's base-ruleset certificate must replay on the
    # source term — the intake failure was CertificateVerificationError.
    assert _cert_ok(eg, ir.root, best, cost_fn) == "pass"
    cert = eg.certificate(ir.root, best, cost_fn=cost_fn)
    assert verify_certificate(ir.root, cert) is not None


# ---------------------------------------------------------------------------
#  Round 4 — attr spelling (``min=0`` vs ``min=0.0``) is structural identity
# ---------------------------------------------------------------------------


def test_attr_int_float_spellings_are_distinct_terms():
    """``clamp(min=0)`` and ``clamp(min=0.0)`` are different terms.

    The ``Const`` defect one level down: ``_attr_key`` carried raw
    values and ``Op.__eq__`` compared the attrs dicts, so the numeric
    tower (``0 == 0.0``, ``hash(0) == hash(0.0)``) coalesced both
    spellings into one interned object — the second ``Op.make``
    returned the FIRST spelling's term, silently rewriting the attr
    before any law fired.  Identity is now repr-keyed like ``Const``;
    ``0``/``0.0``/``False`` are distinct spellings, and container
    spellings (``[3,5]`` vs ``(3,5)``, ``(3,5)`` vs ``(3.0,5.0)``)
    stay distinct too.
    """
    x = Op.make("leafless_op")  # operand content is irrelevant here
    a = Op.make("clamp", x, min=0)
    b = Op.make("clamp", x, min=0.0)
    assert a is not b and a != b
    assert Op.make("clamp", x, min=0) is a  # interning still dedups
    assert a.attrs == {"min": 0}  # the first spelling keeps its value
    assert Op.make("softmax", x, dim=True) != Op.make(
        "softmax", x, dim=1
    )
    c = Op.make("unflatten", x, dim=0, sizes=(3, 5))
    d = Op.make("unflatten", x, dim=0, sizes=(3.0, 5.0))
    e = Op.make("unflatten", x, dim=0, sizes=[3, 5])
    assert c != d and c != e


def test_attr_int_float_spellings_are_distinct_enodes():
    """``min=0`` and ``min=0.0`` land in different e-classes.

    ``ENode`` field-compare ran the same numeric-tower equality over
    the attr tuple, so ``add_enode``/``add_term`` merged the two
    spellings into one class — the term-level defect reaching the
    graph.  Identity is now the repr-keyed ``_sig``; both enodes keep
    their own class.
    """
    from catopt_core.egraph import EGraph
    from catopt_core.ir import TensorType, Var

    eg = EGraph()
    x = eg.add_leaf("x")
    a = eg.add_enode("clamp", (x,), {"min": 0})
    b = eg.add_enode("clamp", (x,), {"min": 0.0})
    assert eg.find(a) != eg.find(b)
    assert eg.add_enode("clamp", (x,), {"min": 0}) == eg.find(a)

    # add_term goes through the same strict enode identity — a Var
    # named "x" reprs to the same leaf key, so only ``min`` differs.
    c = eg.add_term(
        Op.make("clamp", Var("x", TensorType(())), min=0.0)
    )
    assert eg.find(c) == eg.find(b)


def test_attr_match_stays_numerically_lenient():
    """Pattern ``min=0`` still matches a term spelled ``min=0.0``.

    Strictness is *identity* — interning, ``__eq__``, enode keys.
    Matching keeps the numeric leniency the matchers always had
    (``nv != pv``), the same split ``Const`` got: strict object,
    lenient matcher.  Attr metavariables bind the node's raw value.
    """
    from catopt_core.egraph import EGraph
    from catopt_core.egraph.terms import _term_match
    from catopt_core.ir import Const
    from catopt_core.meta import match_pattern

    pat = Op.make("clamp", "v", min=0)
    term = Op.make("clamp", Const(1), min=0.0)
    assert _term_match(pat, term) is not None
    assert match_pattern(pat, term) == {"v": Const(1)}

    # The e-graph matcher agrees — and an attr metavar binds the
    # node's raw (un-normalised) spelling, ``0.0`` here.
    eg = EGraph()
    root = eg.add_term(term)
    subs = list(eg.matches(Op.make("clamp", "v", min="m"), root))
    assert subs and subs[0]["$attr:m"] == 0.0
