"""om_lift + flops_cost on a natural multi-head attention graph.

Regression coverage for two T=32 MHA bugs (ScanAttnMH shape from
bench_omd2.py: view+transpose head split over scanned values, causal
scaled-softmax attention, output projection):

* ``_check_om_lift`` veto — metavariable bindings resolve through
  ``EGraph``'s representative member, which on this graph is a
  *carrier* member: the score e-class holds ``applyd`` enodes whose
  ``cost._shape_of`` convention reports the state-slot shape (``()``
  when the state resolves to a scalar member), and the value e-class
  holds an ``apply`` reporting the state shape ``(D,)``.  Judging the
  lift on those convention shapes fails ``len(shape) >= 2`` and vetoes
  a legal rewrite.  ``catopt.om._vshape`` now resolves the *value*
  shape via ``xcarrier._xshape``, so the check sees the true
  ``(nh,T,K)`` / ``(nh,T,d)`` regardless of which class member is
  picked.

* ``flops_cost`` ZeroDivisionError — extraction prices each enode on
  its cheapest resolved children; a ``transpose`` enode over the
  attention-output class resolves through the cheap ``om_apply``
  member whose inferred shape was ``()``, and ``d % len(())`` raised
  ZeroDivisionError.  ``cost._infer_op_shape`` now reports ``None``
  (unknown) whenever a ``()``-shaped operand reaches an op that cannot
  be scalar — axis-indexing ops and carrier convention slots alike —
  instead of crashing or fabricating a scalar.

The e2e tests export the module, saturate the carrier+XC laws exactly
like ``bench_omd2.bounded_build_xc``, assert ``om_apply`` members were
minted, extract under ``flops_cost``, and verify the extracted member
evaluates fp64-identical to the source module.
"""

from __future__ import annotations

import pytest
import torch

import catopt.om as OM  # registers cmask/fill/attnbias
import catopt.xcarrier as XC  # registers omd/affd bindings
from catopt.cost import _infer_op_shape, _shape_of, flops_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Const, Op, Param, TensorType
from catopt.regime import default_rules
from catopt.torch_bridge import export_to_ir, ir_to_torch_module

# ---------------------------------------------------------------------------
#  The module — ScanAttnMH shape (bench_omd2.py)
# ---------------------------------------------------------------------------


class _ScanAttnMH(torch.nn.Module):
    """h_t = a_t⊙h + x_t; v/q/k = W·(stack h or x) split into heads via
    view+transpose; causal scaled softmax attention; output proj."""

    def __init__(self, T: int, D: int, nh: int = 4, hd: int = 16):
        super().__init__()
        self.nh, self.hd = nh, hd
        self.a = torch.nn.Parameter(torch.randn(T, D) * 0.1)
        self.h0 = torch.nn.Parameter(torch.randn(D) * 0.1)
        self.wq = torch.nn.Linear(D, nh * hd, bias=False)
        self.wk = torch.nn.Linear(D, nh * hd, bias=False)
        self.wv = torch.nn.Linear(D, nh * hd, bias=False)
        self.wo = torch.nn.Linear(nh * hd, D, bias=False)
        # Registered buffer, not a Python float — torch.export
        # serialises a bare float attribute through fp32 (a uniform
        # ~1e-8 output deviation; an export artifact, not a catopt
        # issue — same note as bench_omd2.ScanAttnMQA).
        self.register_buffer(
            "sq", torch.tensor(hd**-0.5, dtype=torch.float64)
        )
        mask = torch.zeros(T, T)
        mask.masked_fill_(
            torch.triu(torch.ones(T, T, dtype=torch.bool), 1),
            float("-inf"),
        )
        self.register_buffer("cm", mask)

    def scan(self, x):
        h = self.h0
        outs = []
        for t in range(x.shape[0]):
            h = self.a[t] * h + x[t]
            outs.append(h)
        return torch.stack(outs)

    def forward(self, x):
        T = x.shape[0]
        hs = self.scan(x)
        v = self.wv(hs).view(T, self.nh, self.hd).transpose(0, 1)
        q = self.wq(x).view(T, self.nh, self.hd).transpose(0, 1)
        k = self.wk(x).view(T, self.nh, self.hd).transpose(0, 1)
        s = q @ k.transpose(-1, -2) * self.sq + self.cm
        o = torch.softmax(s, dim=-1) @ v
        return self.wo(o.transpose(0, 1).reshape(T, self.nh * self.hd))


def _build_mha_egraph(T: int, D: int = 64, nh: int = 4, hd: int = 16):
    """Export + saturate exactly like bench_omd2.bounded_build_xc:
    core carrier laws -> non-local lifts -> bounded XC tier -> lifts."""
    torch.manual_seed(0)
    m = _ScanAttnMH(T, D, nh, hd).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(default_rules(), root, max_iterations=6, max_nodes=300_000)
    lifts = (
        XC.gather_applyd_stack(eg)
        + XC.gather_apply_stack(eg)
        + XC.omd_tree_lift(eg)
    )
    if lifts:
        eg.rebuild()
    eg.run(XC.XC_LAWS, root, max_iterations=4, max_nodes=300_000)
    more = (
        XC.gather_applyd_stack(eg)
        + XC.gather_apply_stack(eg)
        + XC.omd_tree_lift(eg)
    )
    if more:
        eg.rebuild()
    return eg, root, ir, src, m, x


def _om_classes(eg: EGraph):
    """(score_cid, value_cid) for every om_elem enode in the graph."""
    out = []
    for cid in list(eg._classes):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        for n in ec.nodes:
            if n.op == "om_elem" and len(n.children) == 2:
                out.append(
                    (eg.find(n.children[0]), eg.find(n.children[1]))
                )
    return out


def _member_term(eg: EGraph, node) -> Op | None:
    """Resolve one enode to a term via any_term on its children."""
    args = [eg.any_term(eg.find(ch)) for ch in node.children]
    if any(a is None for a in args):
        return None
    return Op.make(node.op, *args, **dict(node.attrs))


# ---------------------------------------------------------------------------
#  Unit-level: the () policy in _infer_op_shape
# ---------------------------------------------------------------------------


def test_scalar_operand_axis_ops_report_none():
    """A ``()``-shaped operand under an op that cannot be scalar means
    a carrier-internal member is being read as a tensor — the shape is
    unknown (None), not a fabricated scalar and never a crash."""
    scalar = Const(1.0)
    v4 = Param("v4", TensorType((4,)))
    m44 = Param("m44", TensorType((4, 4)))
    # the ZeroDivisionError site: transpose on a ()-shaped operand
    assert (
        _infer_op_shape(Op.make("transpose", scalar, arg1=0, arg2=1))
        is None
    )
    # every other axis-indexing op takes the same policy
    assert _infer_op_shape(Op.make("squeeze", scalar, arg1=0)) is None
    assert (
        _infer_op_shape(Op.make("select", scalar, arg1=0, arg2=0))
        is None
    )
    assert _infer_op_shape(Op.make("slice", scalar, arg1=0)) is None
    assert _infer_op_shape(Op.make("unbind", scalar, arg1=0)) is None
    assert (
        _infer_op_shape(Op.make("chunk", scalar, dim=0, chunks=2))
        is None
    )
    assert (
        _infer_op_shape(
            Op.make("split", scalar, dim=0, sizes=(1, 1), index=0)
        )
        is None
    )
    assert (
        _infer_op_shape(Op.make("concat", scalar, scalar, dim=0))
        is None
    )
    assert _infer_op_shape(Op.make("flatten", scalar)) is None
    # explicit-dim reduce on a scalar (dim=None full-reduce stays ())
    assert _infer_op_shape(Op.make("sum", scalar, dim=0)) is None
    assert _infer_op_shape(Op.make("sum", scalar)) == ()
    # scalar operand under matmul/linear is ill-typed, not ()
    assert _infer_op_shape(Op.make("matmul", scalar, m44)) is None
    assert _infer_op_shape(Op.make("linear", scalar, m44)) is None
    # carrier convention slots resolve () -> None
    assert _infer_op_shape(Op.make("om_elem", scalar, v4)) is None
    assert (
        _infer_op_shape(
            Op.make("applyd", Op.make("aff_diag", v4, v4), scalar)
        )
        is None
    )
    # genuine scalar results are unaffected
    assert _infer_op_shape(Op.make("matmul", v4, v4)) == ()  # dot
    assert _shape_of(scalar) == ()


def test_om_lift_check_uses_value_shapes():
    """``_check_om_lift`` judged on ANY member of the score/value
    classes — including the carrier members whose cost-convention
    shapes are ``()`` / ``(D,)`` — sees the true value shapes and
    admits the lift."""
    eg, _root, _ir, _src, _m, _x = _build_mha_egraph(T=32)
    pairs = _om_classes(eg)
    assert pairs, "no om_elem enode minted — om_lift did not fire"
    for sc, vc in pairs:
        s_nodes = [n for n in eg.get_class(sc).nodes]
        v_nodes = [n for n in eg.get_class(vc).nodes]
        for sn in s_nodes:
            s_term = _member_term(eg, sn)
            if s_term is None:
                continue
            for vn in v_nodes:
                v_term = _member_term(eg, vn)
                if v_term is None:
                    continue
                assert OM._check_om_lift(
                    {"s": s_term, "v": v_term, "$attr:SD": -1}
                ), (
                    f"om_lift vetoed on s={sn.op} "
                    f"(shape {_shape_of(s_term)}), v={vn.op} "
                    f"(shape {_shape_of(v_term)})"
                )


# ---------------------------------------------------------------------------
#  E2E — T=32 and T=64
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", [32, 64])
def test_mha_om_lift_fires_and_extract_is_exact(T: int):
    """om_lift mints om members on the MHA graph (not vetoed), and
    ``extract_best(flops_cost)`` completes — no ZeroDivisionError —
    returning a term that evaluates fp64-equal to the module."""
    eg, root, ir, src, m, x = _build_mha_egraph(T=T)

    n_om_apply = 0
    for cid in list(eg._classes):
        c = eg.find(cid)
        ec = eg._classes.get(c)
        if ec is None:
            continue
        n_om_apply += sum(1 for n in ec.nodes if n.op == "om_apply")
    assert n_om_apply > 0, "om_lift vetoed — no om_apply member minted"
    assert _om_classes(eg), "no om_elem members"

    # The crash: extraction priced a transpose over an om_apply member
    # whose () convention shape hit `d % len(())`.
    term = eg.extract_best(root, flops_cost)
    assert term is not None

    mod = ir_to_torch_module(
        IR(root=term, inputs=ir.inputs, params=ir.params), src
    )
    with torch.no_grad():
        ref = m(x)
        out = mod(x)
    err = (out - ref).abs().max().item()
    assert err < 1e-10, f"extracted term deviates: max|Δ|={err:.2e}"
