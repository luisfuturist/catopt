"""Storage-aware extraction and the ε-budget filter.

``param_bytes_cost`` prices the axis the flop-based models cannot see
— how many parameter values a member stores — and
``EGraph.extract_best_bounded`` constrains extraction to members whose
certificate stays within a total error budget (``Certificate.error_bound``
sums the ``Rewrite.error_bound`` of every bound-carrying step).
"""

import pytest
import torch
import torch.nn as nn

from catopt.cost import dag_cost, param_bytes_cost, param_bytes_cost_for
from catopt.egraph import EGraph
from catopt.eps import low_rank_params
from catopt.ir import Op, Param, TensorType, Var, op_repr
from catopt.rules import all_rules
from catopt.torch_bridge import export_to_ir


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


def _saturated(seed=0, truncation_level=2):
    """(ir, source_tensors, eg, root, offers): saturated e-graph with
    the low-rank pass already applied."""
    m = _lowrank_model(seed)
    x = torch.randn(4, 64, dtype=torch.float64)
    ir, src = export_to_ir(m, x)
    eg = EGraph(truncation_level=truncation_level)
    root = eg.add_term(ir.root)
    eg.run(all_rules(), root, max_iterations=4)
    offers = low_rank_params(eg, src, rtol=0.05)
    return ir, src, eg, root, offers


def _param_names(term):
    if isinstance(term, Param):
        return {term.name}
    if isinstance(term, Op):
        out = set()
        for a in term.args:
            out |= _param_names(a)
        return out
    return set()


def _has_chained_linear(term):
    """True if some `linear` applies a `linear` result — the eps member."""
    if not isinstance(term, Op):
        return False
    if (
        term.op == "linear"
        and term.args
        and isinstance(term.args[0], Op)
        and term.args[0].op == "linear"
    ):
        return True
    return any(_has_chained_linear(a) for a in term.args)


# ---------------------------------------------------------------------------
#  param_bytes_cost drives extraction onto the compressed member
# ---------------------------------------------------------------------------


def test_param_bytes_selects_factorised_member():
    """The killer result: pricing parameter storage makes extraction
    pick the certified low-rank chain that flops-based models pass over."""
    ir, src, eg, root, offers = _saturated()
    assert offers, "expected a low-rank offer"
    o = offers[0]

    cost = param_bytes_cost_for(src)
    best = eg.extract_best(root, cost)
    assert best is not None
    assert _has_chained_linear(best), op_repr(best)

    names = _param_names(best)
    assert o["name"] not in names  # dense W is gone
    assert any(n.startswith("eps_u_") for n in names)
    assert any(n.startswith("eps_v_") for n in names)
    # billed bytes = exactly the factorisation's stored count
    # (r·(o+i) for the SVD-chosen rank r — genuinely below 64·64)
    assert param_bytes_cost(best, src) == o["stored"] < 64 * 64

    # DAG-true accounting agrees with the flat tree cost.
    assert dag_cost(best, cost) == pytest.approx(o["stored"])


def test_param_bytes_counts_eps_factors_normally():
    """eps_* derived params are billed like any other Param leaf."""
    ir, src, eg, root, offers = _saturated()
    o = offers[0]
    cost = param_bytes_cost_for(src)
    best = eg.extract_best(root, cost)
    # numel comes from the injected tensors in source_tensors
    uname = next(
        n for n in _param_names(best) if n.startswith("eps_u_")
    )
    vname = next(
        n for n in _param_names(best) if n.startswith("eps_v_")
    )
    assert uname in src and vname in src
    assert src[uname].numel() + src[vname].numel() == o["stored"]


# ---------------------------------------------------------------------------
#  extract_best_bounded: certificate-constrained extraction
# ---------------------------------------------------------------------------


def test_bounded_generous_budget_keeps_factorised():
    """With max_error >= the certified bound, the compressed member
    stays the cheapest eligible one."""
    ir, src, eg, root, offers = _saturated()
    o = offers[0]
    cost = param_bytes_cost_for(src)
    b = eg.extract_best_bounded(
        root, cost, max_error=o["bound"] * 2, src_term=ir.root
    )
    assert b is not None
    assert _has_chained_linear(b)
    cert = eg.certificate(ir.root, b, root_eid=root)
    assert cert.error_bound == pytest.approx(o["bound"])
    assert cert.error_bound <= o["bound"] * 2


def test_bounded_zero_budget_is_exact_only():
    """max_error=0 reproduces exact-only extraction: the dense member
    returns, identical to an e-graph that never saw the eps offer."""
    ir, src, eg, root, offers = _saturated()
    cost = param_bytes_cost_for(src)
    lo = eg.extract_best_bounded(
        root, cost, max_error=0.0, src_term=ir.root
    )
    assert lo is not None
    assert not _has_chained_linear(lo)
    assert not any(n.startswith("eps_") for n in _param_names(lo))
    cert = eg.certificate(ir.root, lo, root_eid=root)
    assert cert.exact

    # Reference: same model, no eps pass — plain extraction.
    m = _lowrank_model()
    x = torch.randn(4, 64, dtype=torch.float64)
    ir2, src2 = export_to_ir(m, x)
    eg2 = EGraph()
    root2 = eg2.add_term(ir2.root)
    eg2.run(all_rules(), root2, max_iterations=4)
    exact_best = eg2.extract_best(root2, param_bytes_cost_for(src2))
    assert op_repr(lo) == op_repr(exact_best)


def test_bounded_budget_between_sweeps_the_choice():
    """max_error just below the bound rejects the offer; just above
    accepts it — the constraint, not the cost model, decides."""
    ir, src, eg, root, offers = _saturated()
    o = offers[0]
    cost = param_bytes_cost_for(src)
    under = eg.extract_best_bounded(
        root, cost, max_error=o["bound"] * 0.5, src_term=ir.root
    )
    over = eg.extract_best_bounded(
        root, cost, max_error=o["bound"] * 1.5, src_term=ir.root
    )
    assert not _has_chained_linear(under)
    assert _has_chained_linear(over)


def test_bounded_exact_graph_unaffected():
    """Exact members always satisfy any max_error: on an e-graph with
    no bound-carrying offers, bounded extraction is plain extraction."""
    x = Var("x", TensorType((4, 8)))
    W = Param("W", TensorType((8, 8)))
    t = Op.make("linear", x, W)
    eg = EGraph()
    root = eg.add_term(t)
    eg.run(all_rules(), root, max_iterations=3)
    plain = eg.extract_best(root, param_bytes_cost)
    for eps in (0.0, 1e-9, 1.0):
        b = eg.extract_best_bounded(
            root, param_bytes_cost, max_error=eps, src_term=t
        )
        assert b is not None
        cert = eg.certificate(t, b, root_eid=root)
        assert cert.exact
        assert cert.error_bound <= eps
    b0 = eg.extract_best_bounded(
        root, param_bytes_cost, max_error=0.0, src_term=t
    )
    assert op_repr(b0) == op_repr(plain)


def test_bounded_none_budget_is_plain_extraction():
    ir, src, eg, root, _ = _saturated()
    cost = param_bytes_cost_for(src)
    assert op_repr(eg.extract_best_bounded(root, cost)) == op_repr(
        eg.extract_best(root, cost)
    )


def test_bounded_level1_is_vacuous():
    """truncation_level=1 records no witnesses: every member certifies
    at bound 0, so the ε constraint is vacuous (documented behaviour)."""
    ir, src, eg, root, offers = _saturated(truncation_level=1)
    assert offers
    b = eg.extract_best_bounded(
        root, param_bytes_cost_for(src), max_error=0.0, src_term=ir.root
    )
    assert b is not None
    assert _has_chained_linear(b)
