# ruff: noqa: RUF002
"""The ε axis — certified bounded-error rewrites.

``low_rank_params`` offers truncated-SVD factorisations of ``linear``
weights as witnessed rewrites carrying an Eckart–Young spectral bound;
certificates aggregate the bound.  These tests pin the mechanics:
the offer exists, it stores fewer values, the bound is real, and the
executed member honours it.
"""

import torch
import torch.nn as nn

from catopt.cost import param_bytes_cost_for
from catopt.egraph import EGraph
from catopt.eps import (
    kron_linear_params,
    low_rank_gather,
    low_rank_params,
    quant_params,
)
from catopt.ir import IR, Op, TensorType, Var
from catopt.rules import all_rules
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
                n.op,
                c0,
                *[eg.any_term(eg.find(c)) for c in n.children[1:]],
                **dict(n.attrs),
            )
    assert chained is not None

    xv = Var("x", TensorType((4, 64)))
    mod = ir_to_torch_module(
        IR(root=Op.make("add", chained, xv), params={}, inputs=[xv]),
        src,
    )
    with torch.no_grad():
        err = (mod(x) - m(x)).abs().max().item()
    x_norm = torch.linalg.norm(x, dim=-1).max().item()
    assert err <= o["bound"] * x_norm + 1e-9

    # certificate reports the accumulated bound
    cert = eg.certificate(ir.root, Op.make("add", chained, xv))
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
            self.lin.weight.data = torch.kron(
                A, B
            ) + 0.02 * torch.randn(64, 64)

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
    assert all(n.startswith("eps_k") for n in src if "eps_" in n)

    cand = None
    for n in eg.get_class(o["site_eid"]).nodes:
        if n.op in ("add", "reshape", "matmul"):
            cand = Op.make(
                n.op,
                *[eg.any_term(eg.find(c)) for c in n.children],
                **dict(n.attrs),
            )
            break
    assert cand is not None

    xv = Var("x", TensorType((3, 64)))
    mod = ir_to_torch_module(IR(root=cand, params={}, inputs=[xv]), src)
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


def test_quant_params_int8_certified():
    """Quantization-as-rewrite: W -> mul(float(W_int8), s) with a
    certified Frobenius bound (s/2)·√n.  Byte-aware storage cost
    selects it; the extracted module stores an int8 tensor."""
    torch.manual_seed(0)
    m = nn.Linear(64, 64, bias=False).eval().double()
    x = torch.randn(4, 64, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=2)
    offers = quant_params(eg, src, bits=8)
    assert offers
    o = offers[0]
    assert o["bound"] > 0

    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    xv = Var("x", TensorType((4, 64)))
    mod = ir_to_torch_module(IR(root=term, params={}, inputs=[xv]), src)
    with torch.no_grad():
        err = (mod(x) - m(x)).abs().max().item()
    assert err <= o["bound"] + 1e-9
    assert any(p.dtype == torch.int8 for p in mod.parameters())
    assert (
        sum(t.numel() * t.element_size() for t in mod.parameters())
        < 64 * 64 * 8 / 2
    )  # int8 < fp64
    cert = eg.certificate(ir.root, term, root_eid=root)
    assert cert.error_bound == o["bound"] and cert.replayable
    assert not cert.exact


def test_model_bound_propagates_through_graph():
    """eps.model_bound: site-local bounds × Lipschitz path
    sensitivities -> a finite whole-model certificate."""
    from catopt.eps import model_bound

    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            s.l1 = nn.Linear(32, 64, bias=False)
            s.l2 = nn.Linear(64, 32, bias=False)

        def forward(s, x):
            return s.l2(torch.relu(s.l1(x)))

    m = M().eval().double()
    x = torch.randn(4, 32, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=2)
    offers = quant_params(eg, src, bits=8)
    assert len(offers) == 2
    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    xv = Var("x", TensorType((4, 32)))
    mod = ir_to_torch_module(IR(root=term, params={}, inputs=[xv]), src)
    with torch.no_grad():
        err = (mod(x) - m(x)).abs().max().item()
    cert = eg.certificate(ir.root, term, root_eid=root)
    mb = model_bound(
        term,
        cert,
        src,
        input_norm=torch.linalg.norm(x, dim=-1).max().item(),
    )
    # finite, certified, and above the measured error
    assert mb["bound"] != float("inf")
    assert mb["bound"] >= err
    assert mb["n_bounded_steps"] == 2


def test_eps_families_compose():
    """The unifying claim: gather-low-rank + quantization + exact
    sharing coexist in one e-graph, one extraction, one certificate."""
    from catopt.eps import model_bound

    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            U = torch.randn(1000, 8)
            V = torch.randn(8, 64)
            s.emb = nn.Embedding(1000, 64)
            s.emb.weight.data = U @ V + 0.02 * torch.randn(1000, 64)
            s.l1 = nn.Linear(64, 64, bias=False)
            s.l2 = nn.Linear(64, 64, bias=False)

        def forward(s, x):
            return s.l2(torch.relu(s.l1(s.emb(x))))

    m = M().eval().double()
    idx = torch.randint(0, 1000, (8,))
    orig = sum(p.numel() * p.element_size() for p in m.parameters())
    ir, src = export_to_ir(m, idx)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=3)
    g = low_rank_gather(eg, src, rtol=0.05)
    q = quant_params(eg, src, bits=8)
    assert g and q
    term = eg.extract_best(
        root, param_bytes_cost_for(src, by_bytes=True)
    )
    xv = Var("x", TensorType((8,)))
    mod = ir_to_torch_module(IR(root=term, params={}, inputs=[xv]), src)
    with torch.no_grad():
        err = (mod(idx) - m(idx)).abs().max().item()
    nb = sum(t.numel() * t.element_size() for t in mod.parameters())
    assert nb < orig / 4  # real compression
    cert = eg.certificate(ir.root, term, root_eid=root)
    assert cert.replayable and not cert.exact
    mb = model_bound(term, cert, src, input_norm=1.0)
    assert mb["bound"] >= err or mb["bound"] == float("inf")


def test_optimize_weight_program():
    """A weight IS a program: optimize_weight builds an e-graph over
    the leaf, offers certified realizations, saturates them under the
    ordinary laws, and extracts the cheapest under storage cost."""
    from catopt.eps import optimize_weight
    from catopt.ir import IR, TensorType, Var
    from catopt.torch_bridge import IRModule

    torch.manual_seed(0)
    W = (
        torch.randn(128, 6) @ torch.randn(6, 128)
        + 0.01 * torch.randn(128, 128)
    ).double()
    res = optimize_weight("W", W, rtol=0.02)
    assert res["offers"]  # something was offered
    assert res["bytes"] < res["orig_bytes"]  # storage shrank
    cert = res["certificate"]
    assert cert.replayable
    # the extracted program *executes* and meets its bound
    mod = IRModule(
        IR(
            root=res["term"],
            params={},
            inputs=[Var("x", TensorType((1,)))],
        ),
        res["source_tensors"],
    )
    with torch.no_grad():
        What = mod(torch.randn(1, dtype=torch.float64))
    err = float(torch.linalg.norm(What - W, 2))
    assert err <= cert.error_bound + 1e-9


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


def test_low_rank_gather_factors_embedding():
    """embedding(W, idx) -> matmul(embedding(U_r, idx), V_r): gather
    the small factor, project — the measured real-weight win."""
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(s):
            super().__init__()
            U = torch.randn(1000, 12)
            V = torch.randn(12, 64)
            s.emb = nn.Embedding(1000, 64)
            s.emb.weight.data = U @ V + 0.01 * torch.randn(1000, 64)

        def forward(s, x):
            return s.emb(x)

    m = M().eval().double()
    idx = torch.randint(0, 1000, (8,))
    ir, src = export_to_ir(m, idx)
    eg = EGraph()
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=3)
    offers = low_rank_gather(eg, src, rtol=0.05)
    assert offers
    o = offers[0]
    assert o["stored"] < o["original"]

    cand = None
    for n in eg.get_class(o["site_eid"]).nodes:
        if n.op == "matmul":
            cand = Op.make(
                n.op, *[eg.any_term(eg.find(c)) for c in n.children]
            )
            break
    assert cand is not None
    xv = Var("x", TensorType((8,)))
    mod = ir_to_torch_module(IR(root=cand, params={}, inputs=[xv]), src)
    with torch.no_grad():
        err = (mod(idx) - m(idx)).abs().max().item()
    # per-row spectral bound is conservative here
    assert err <= o["bound"] + 1e-9
    cert = eg.certificate(ir.root, cand, root_eid=root)
    assert cert.error_bound == o["bound"] and cert.replayable
    assert sum(t.numel() for t in mod.parameters()) == o["stored"]
