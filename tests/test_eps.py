"""The ε axis — certified bounded-error rewrites.

``low_rank_params`` offers truncated-SVD factorisations of ``linear``
weights as witnessed rewrites carrying an Eckart–Young spectral bound;
certificates aggregate the bound.  These tests pin the mechanics:
the offer exists, it stores fewer values, the bound is real, and the
executed member honours it.
"""
import torch
import torch.nn as nn

from catopt.egraph import EGraph
from catopt.ir import IR, Op, Var, TensorType, op_repr
from catopt.rules import all_rules
from catopt.eps import low_rank_params, kron_linear_params
from catopt.torch_bridge import export_to_ir, ir_to_torch_module


def _lowrank_model(seed=0):
    """Linear whose weight is near-rank-8 (low-rank + small noise)."""
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            U = torch.randn(64, 8)
            V = torch.randn(8, 64)
            self.lin = nn.Linear(64, 64, bias=False)
            self.lin.weight.data = (U @ V) + 0.01 * torch.randn(64, 64)

        def forward(self, x):
            return self.lin(x) + x

    return M().eval().double()


def test_low_rank_offer_exists_with_bound():
    m = _lowrank_model()
    x = torch.randn(4, 64, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=4)
    offers = low_rank_params(eg, src, rtol=0.05)
    assert offers, "expected at least one factorisation offer"
    o = offers[0]
    assert o["stored"] < o["original"]
    assert o["bound"] > 0
    # the chained member entered the site's e-class
    ops = {n.op for n in eg.get_class(o["site_eid"]).nodes}
    assert "linear" in ops
    # derived params materialised into source_tensors
    assert any(n.startswith("eps_u_") for n in src)


def test_low_rank_offer_rejects_full_rank():
    """A full-rank weight with tight rtol gets no offer — the pass
    refuses rather than fabricate a saving."""
    torch.manual_seed(1)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            return self.lin(x)

    m = M().eval().double()
    x = torch.randn(4, 64, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=3)
    offers = low_rank_params(eg, src, rtol=1e-9)
    assert offers == []


def test_eps_bound_certified_and_honoured():
    """The offered member executes within the certified bound and the
    certificate reports the accumulated ε."""
    m = _lowrank_model()
    x = torch.randn(4, 64, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=4)
    offers = low_rank_params(eg, src, rtol=0.05)
    o = offers[0]

    # extract the chained member at the site class
    chained = None
    for n in eg.get_class(o["site_eid"]).nodes:
        c0 = eg.any_term(eg.find(n.children[0]))
        if isinstance(c0, Op) and c0.op == "linear":
            chained = Op.make(
                n.op, c0,
                *[eg.any_term(eg.find(c)) for c in n.children[1:]],
                **dict(n.attrs))
    assert chained is not None

    xv = Var("x", TensorType((4, 64)))
    mod = ir_to_torch_module(
        IR(root=Op.make("add", chained, xv), params={}, inputs=[xv]),
        src)
    with torch.no_grad():
        err = (mod(x) - m(x)).abs().max().item()
    x_norm = torch.linalg.norm(x, dim=-1).max().item()
    assert err <= o["bound"] * x_norm + 1e-9

    # certificate reports the accumulated bound
    cert = eg.certificate(ir.root,
                          Op.make("add", chained, xv))
    assert cert.error_bound >= o["bound"]
    assert not cert.exact
    assert "eps_lr" in " ".join(cert.rules_used)


def _kron_model(seed=0):
    """Linear whose weight is near-Kronecker (A⊗B + small noise)."""
    torch.manual_seed(seed)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            A = torch.randn(16, 16)
            B = torch.randn(4, 4)
            self.lin = nn.Linear(64, 64, bias=False)
            self.lin.weight.data = torch.kron(A, B) \
                + 0.02 * torch.randn(64, 64)

        def forward(self, x):
            return self.lin(x)

    return M().eval().double()


def test_kron_offer_executes_within_frobenius_bound():
    m = _kron_model()
    x = torch.randn(3, 64, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=4)
    offers = kron_linear_params(eg, src, rtol=0.05)
    assert offers, "expected a Kronecker-sum offer"
    o = offers[0]
    assert o["stored"] < o["original"]
    assert all(n.startswith("eps_k") for n in src
               if "eps_" in n)

    cand = None
    for n in eg.get_class(o["site_eid"]).nodes:
        if n.op in ("add", "reshape", "matmul"):
            cand = Op.make(
                n.op,
                *[eg.any_term(eg.find(c)) for c in n.children],
                **dict(n.attrs))
            break
    assert cand is not None

    xv = Var("x", TensorType((3, 64)))
    mod = ir_to_torch_module(
        IR(root=cand, params={}, inputs=[xv]), src)
    with torch.no_grad():
        err = (mod(x) - m(x)).abs().max().item()
    x_norm = torch.linalg.norm(x, dim=-1).max().item()
    assert err <= o["bound"] * x_norm + 1e-9

    cert = eg.certificate(ir.root, cand, root_eid=root)
    assert cert.error_bound == o["bound"]
    assert cert.replayable and not cert.exact
    assert "eps_kron" in " ".join(cert.rules_used)
    # factor params only — the dense weight is gone
    assert sum(t.numel() for t in mod.parameters()) == o["stored"]


def test_kron_rejects_dense_random():
    """A random full-rank weight has no compressible rearrangement."""
    torch.manual_seed(7)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(64, 64, bias=False)

        def forward(self, x):
            return self.lin(x)

    m = M().eval().double()
    x = torch.randn(3, 64, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=3)
    assert kron_linear_params(eg, src, rtol=1e-6) == []
