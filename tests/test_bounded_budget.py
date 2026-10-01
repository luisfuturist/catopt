"""Bounded-error rewrites — the ``error_budget=`` seam (plan 0012).

Pins the certified-approximation path end to end:

* ``specials.offer_weight_specials(budget=...)`` — nearly-dead
  slices (``max|·| <= budget``) and near-duplicate rows
  (``max|Δ| <= budget``) offer ``elide_bounded`` / ``zero_bounded``
  members whose witness carries the measured ``error_bound``;
  bitwise-exact members stay ``error_bound=0`` and ``budget=None``
  keeps the pass exact.
* ``search(..., error_budget=...)`` — the bound ledger
  (certificate + delivered-member fingerprints) gates extraction,
  ``stats["error_bounds"]`` records every accepted bound, and
  ``lower`` verifies at ``rtol=max(rtol, bound)`` *and* declines
  when the measured ``max_rel`` exceeds the claimed bound.
* ``export_optimized(..., stats=...)`` — the manifest carries the
  bound fields verbatim, including a False ``error_bounds_honored``.
"""

from unittest import mock

import pytest
import torch
import torch.nn as nn
from catopt_core.cost import flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Op, Param, TensorType, Var, op_repr
from catopt_core.laws.specials import (
    _analyse,
    _bounded_elide,
    _m_elide_bounded,
    offer_weight_specials,
)
from catopt_orchestrator import optimize as opt_mod
from catopt_orchestrator.optimize import (
    Compositional,
    OptimizationResourceError,
    Optimizer,
    _ban_bound_members,
    _bounded_gate,
    _bounded_term,
    _check_budget_engine,
)
from catopt_torch.backend import TorchBackend
from catopt_torch.export import export_optimized
from catopt_torch.torch_bridge import IRModule


def _lower(term, x, tensors, xv):
    """Lower *term* through the torch sink and evaluate on *xv*."""
    ir = IR(root=term, inputs=[x], input_names={x.name}, params={})
    mod = IRModule(ir, param_values=dict(tensors))
    with torch.no_grad():
        return mod(xv)


def _site(term):
    """Intern *term* in a fresh graph; return (eg, root_eid)."""
    eg = EGraph()
    return eg, eg.add_term(term)


def _near_dead(o=64, i=32, rows=(16, 48), eps=8e-4, seed=0, scale=5.0):
    """(o, i) weight, ``rows`` nearly dead (one eps element each)."""
    torch.manual_seed(seed)
    wv = torch.randn(o, i, dtype=torch.float64) * scale
    wv[rows[0] : rows[1]] = 0.0
    idx = torch.arange(rows[0], rows[1])
    wv[idx, idx % i] = eps
    return wv


class _NearDead(nn.Module):
    """One Linear whose middle output rows are nearly dead."""

    def __init__(self, i=32, o=64, eps=8e-4, scale=5.0):
        super().__init__()
        self.lin = nn.Linear(i, o, bias=False).double()
        with torch.no_grad():
            self.lin.weight.mul_(scale)
            self.lin.weight[16:48] = 0.0
            idx = torch.arange(16, 48)
            self.lin.weight[idx, idx % i] = eps

    def forward(self, t):
        return self.lin(t)


class _NearDeadSplit(nn.Module):
    """Two near-dead linears on disjoint input halves."""

    def __init__(self, i=32, o=64, eps=8e-4):
        super().__init__()
        self.a = nn.Linear(i, o, bias=False).double()
        self.b = nn.Linear(i, o, bias=False).double()
        for lin in (self.a, self.b):
            with torch.no_grad():
                lin.weight[16:48] = 0.0
                idx = torch.arange(16, 48)
                lin.weight[idx, idx % i] = eps
        self.i = i

    def forward(self, t):
        return self.a(t[..., : self.i]) + self.b(t[..., self.i :])


# ---------------------------------------------------------------------------
#  _bounded_elide — the budget-aware analysis
# ---------------------------------------------------------------------------


def test_analyse_budget_none_is_exact_only():
    """``budget=None`` leaves ``a["b"]`` unset — bitwise-exact only."""
    wv = _near_dead(o=8, i=4, rows=(2, 4))
    a = _analyse(wv, 8, 4)
    assert a["b"] is None
    b = _analyse(wv, 8, 4, budget=1e-3)["b"]
    assert b["bound"] == pytest.approx(8e-4)
    assert b["all_dead"] is False


def test_bounded_elide_near_dead_rows_analysis():
    """Nearly-dead rows join the zero group; the appended synthetic
    zero representative lands at the tail position."""
    wv = _near_dead(o=8, i=4, rows=(2, 6), eps=1e-5)
    b = _bounded_elide(wv, 8, 4, 1e-4)
    assert b["keep"] == (0, 1, 2, 3)
    assert b["firsts"] == (0, 1, 6, 7)
    assert b["zero_appended"] is True
    assert b["imap"] == (0, 1, 4, 4, 4, 4, 2, 3)
    assert b["bound"] == pytest.approx(1e-5)


def test_bounded_elide_reuses_real_zero_row():
    """A bitwise-zero row already in the weight becomes the zero
    representative — no synthetic row is appended."""
    wv = _near_dead(o=8, i=4, rows=(2, 6), eps=1e-5)
    wv[7] = 0.0  # one REAL zero row
    b = _bounded_elide(wv, 8, 4, 1e-4)
    assert b["zero_appended"] is False
    assert b["firsts"] == (0, 1, 6, 7)
    assert b["imap"][7] == 3  # zero rep position = last of firsts
    assert all(b["imap"][r] == 3 for r in range(2, 6))


def test_bounded_elide_columns_and_near_dups():
    """Near-dead columns gather away; near-duplicate rows merge with
    the measured difference as bound."""
    torch.manual_seed(1)
    wv = torch.randn(8, 5, dtype=torch.float64)
    wv[:, 4] = 0.0
    wv[0, 4] = 2e-5  # near-dead column
    wv[6] = wv[5] + 3e-5  # near-duplicate row (exact dedup misses)
    b = _bounded_elide(wv, 8, 5, 1e-4)
    assert b["keep"] == (0, 1, 2, 3)
    assert 6 not in b["firsts"] and b["imap"][6] == b["imap"][5]
    assert b["bound"] == pytest.approx(3e-5)


def test_bounded_elide_all_dead():
    """Every row within budget of zero -> ``all_dead``."""
    wv = torch.full((6, 4), 5e-5, dtype=torch.float64)
    wv[0, 0] = 0.0
    b = _bounded_elide(wv, 6, 4, 1e-4)
    assert b["all_dead"] is True
    assert b["bound"] == pytest.approx(5e-5)


def test_bounded_elide_clean_weight_bound_zero():
    """Live rows + live columns -> bound 0, nothing to elide."""
    torch.manual_seed(2)
    wv = torch.randn(4, 4, dtype=torch.float64)
    b = _bounded_elide(wv, 4, 4, 1e-4)
    assert b["bound"] == 0.0
    assert b["keep"] == (0, 1, 2, 3)
    assert b["firsts"] == (0, 1, 2, 3)


# ---------------------------------------------------------------------------
#  offer_weight_specials(budget=...) — bounded member offers
# ---------------------------------------------------------------------------


def test_offer_bounded_elide_member():
    """A nearly-dead row set offers ``elide_bounded`` with the
    measured ``max|ΔW|`` bound — beside the exact members."""
    o, i = 64, 32
    wv = _near_dead()
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    offers = offer_weight_specials(eg, tensors, budget=1e-3)
    kinds = [r["kind"] for r in offers]
    assert "elide_bounded" in kinds
    rec = next(r for r in offers if r["kind"] == "elide_bounded")
    assert rec["error_bound"] == pytest.approx(8e-4)
    assert rec["bound_norm"] == "max_abs"
    assert tensors[rec["w_param"]].shape == (33, i)
    best = eg.extract_best(eid, flops_cost)
    assert "__bl" in op_repr(best)
    xv = torch.randn(8, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    ref = xv @ wv.T
    # Weight-space bound eps scales by the input magnitude.
    assert (out - ref).abs().max().item() < 8e-4 * xv.abs().max() * 1.1


def test_offer_bounded_beyond_budget_declines():
    """Deviations past the budget offer nothing bounded."""
    o, i = 64, 32
    wv = _near_dead(eps=1e-2)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, _ = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(eg, {"p_w": wv}, budget=1e-3)
    assert offers == []


def test_offer_bounded_exact_dead_stays_exact():
    """Bitwise-dead rows are already exact: under a budget the offer
    stays ``elide`` with ``error_bound == 0`` — no bounded twin."""
    o, i = 64, 32
    torch.manual_seed(3)
    wv = torch.randn(o, i, dtype=torch.float64)
    wv[16:48] = 0.0
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, _ = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(eg, {"p_w": wv}, budget=1e-3)
    assert [r["kind"] for r in offers] == ["elide"]
    assert offers[0]["error_bound"] == 0.0


def test_offer_zero_bounded_member():
    """An all-near-zero weight offers ``zero_bounded`` — the measured
    ``max|W|`` bound, no derived weight parameter."""
    o, i = 8, 4
    wv = torch.full((o, i), 5e-5, dtype=torch.float64)
    x = Var("x", TensorType((2, i)))
    w = Param("p_z", TensorType((o, i)))
    eg, eid = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(eg, {"p_z": wv}, budget=1e-4)
    assert [r["kind"] for r in offers] == ["zero_bounded"]
    assert offers[0]["error_bound"] == pytest.approx(5e-5)
    best = eg.extract_best(eid, flops_cost)
    xv = torch.randn(2, i, dtype=torch.float64)
    out = _lower(best, x, {"p_z": wv}, xv)
    assert out.abs().max().item() == 0.0


def test_offer_bounded_zero_is_exact_stays_zero():
    """A bitwise-zero weight keeps the exact ``zero`` offer even
    under a budget."""
    o, i = 8, 4
    wv = torch.zeros(o, i, dtype=torch.float64)
    x = Var("x", TensorType((2, i)))
    w = Param("p_z", TensorType((o, i)))
    eg, _ = _site(Op.make("linear", x, w))
    offers = offer_weight_specials(eg, {"p_z": wv}, budget=1e-4)
    assert [r["kind"] for r in offers] == ["zero"]
    assert offers[0]["error_bound"] == 0.0


def test_offer_bounded_midrow_permutation_gathers():
    """A single mid-array near-dead row needs the output gather even
    when ``n_sub == o`` — the appended zero rep permutes."""
    o, i = 8, 5
    torch.manual_seed(4)
    wv = torch.randn(o, i, dtype=torch.float64)
    wv[2] = 0.0
    wv[2, 0] = 1e-5
    wv[:, 4] = 0.0
    wv[0, 4] = 2e-5
    x = Var("x", TensorType((4, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    offers = offer_weight_specials(eg, tensors, budget=1e-4)
    assert [r["kind"] for r in offers] == ["elide_bounded"]
    best = eg.extract_best(eid, flops_cost)
    assert best.op == "index_select"
    xv = torch.randn(4, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert (out - xv @ wv.T).abs().max().item() < 1e-4


def test_offer_bounded_single_row_no_saving_declines():
    """One mid-array near-dead row with no dead columns: the bounded
    member keeps ``n_sub == o`` (identity-permute only) — useless, so
    it is not offered."""
    o, i = 8, 4
    torch.manual_seed(5)
    wv = torch.randn(o, i, dtype=torch.float64)
    wv[2] = 0.0
    wv[2, 0] = 1e-5
    x = Var("x", TensorType((4, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, _ = _site(Op.make("linear", x, w))
    assert offer_weight_specials(eg, {"p_w": wv}, budget=1e-4) == []


def test_offer_bounded_bias_readded():
    """``linear(x, W, b)`` — the bounded member wraps in ``add``."""
    o, i = 64, 32
    wv = _near_dead()
    bv = torch.randn(o, dtype=torch.float64)
    x = Var("x", TensorType((8, i)))
    w = Param("p_wb", TensorType((o, i)))
    b = Param("p_bb", TensorType((o,)))
    eg, eid = _site(Op.make("linear", x, w, b))
    tensors = {"p_wb": wv, "p_bb": bv}
    offers = offer_weight_specials(eg, tensors, budget=1e-3)
    assert "elide_bounded" in {r["kind"] for r in offers}
    best = eg.extract_best(eid, flops_cost)
    xv = torch.randn(8, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert torch.allclose(out, xv @ wv.T + bv, atol=1e-2)


def test_offer_bounded_real_zero_rep_member():
    """A bitwise-zero row already in the weight becomes the zero
    representative — the bounded member reuses it (``zero_appended``
    False) and re-expands onto it."""
    o, i = 8, 4
    torch.manual_seed(21)
    wv = torch.randn(o, i, dtype=torch.float64)
    wv[2:6] = 0.0
    idx = torch.arange(2, 6)
    wv[idx, idx % i] = 1e-5
    wv[7] = 0.0  # real zero row joins the zero group as rep
    x = Var("x", TensorType((4, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    offers = offer_weight_specials(eg, tensors, budget=1e-4)
    rec = next(r for r in offers if r["kind"] == "elide_bounded")
    # firsts (0,1,6) + the real zero row (7) — a 4-row sub weight.
    assert tensors[rec["w_param"]].shape == (4, i)
    best = eg.extract_best(eid, flops_cost)
    xv = torch.randn(4, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert (out - xv @ wv.T).abs().max().item() < 1e-4


def test_offer_bounded_cols_only_no_output_gather():
    """Nearly-dead input columns alone: the member gathers the data
    side and needs no output re-expansion (identity ``imap``)."""
    o, i = 8, 5
    torch.manual_seed(22)
    wv = torch.randn(o, i, dtype=torch.float64)
    wv[:, 4] = 0.0
    wv[0, 4] = 2e-5  # the column is nearly dead, rows stay distinct
    x = Var("x", TensorType((4, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    offers = offer_weight_specials(eg, tensors, budget=1e-4)
    rec = next(r for r in offers if r["kind"] == "elide_bounded")
    assert tensors[rec["w_param"]].shape == (o, i - 1)
    best = eg.extract_best(eid, flops_cost)
    assert best.op == "linear" and best.args[0].op == "index_select"
    xv = torch.randn(4, i, dtype=torch.float64)
    out = _lower(best, x, tensors, xv)
    assert (out - xv @ wv.T).abs().max().item() < 1e-4


def test_offer_bounded_matmul_orientation():
    """``matmul(x, W)`` — same bounded analysis on the transposed
    reading."""
    i, o = 32, 64
    torch.manual_seed(6)
    wv = torch.randn(i, o, dtype=torch.float64)
    wv[:, 16:48] = 0.0
    idx = torch.arange(16, 48)
    wv[idx % i, idx] = 8e-4
    x = Var("x", TensorType((8, i)))
    w = Param("p_m", TensorType((i, o)))
    eg, _ = _site(Op.make("matmul", x, w))
    offers = offer_weight_specials(eg, {"p_m": wv}, budget=1e-3)
    rec = next(r for r in offers if r["kind"] == "elide_bounded")
    assert rec["orient"] == "matmul_r"
    assert 0 < rec["error_bound"] <= 1e-3


def test_bounded_witness_carries_measured_bound():
    """The synthetic rewrite records the measured bound and
    ``bound_norm='max_abs'`` — not zero."""
    o, i = 64, 32
    wv = _near_dead()
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, _ = _site(Op.make("linear", x, w))
    offer_weight_specials(eg, {"p_w": wv}, budget=1e-3)
    wits = [
        r
        for n, r in eg._rule_objs.items()
        if n.startswith("weight_special#")
    ]
    bounded = [r for r in wits if r.error_bound]
    assert bounded
    assert all(
        r.bound_norm == "max_abs" and r.error_bound > 0
        for r in bounded
    )


def test_bounded_offer_idempotent():
    """A second pass re-offers nothing — the union is already done."""
    o, i = 64, 32
    wv = _near_dead()
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, _ = _site(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    assert offer_weight_specials(eg, tensors, budget=1e-3)
    assert offer_weight_specials(eg, tensors, budget=1e-3) == []


def test_m_elide_bounded_all_dead_guard():
    """Direct call: an all-dead analysis declines — ``zero_bounded``
    owns the site."""
    o, i = 8, 4
    x = Var("x", TensorType((2, i)))
    w = Param("p_w", TensorType((o, i)))
    eg = EGraph()
    eg.add_term(Op.make("linear", x, w))
    cid = eg.find(eg._class_of_term(Op.make("linear", x, w)))
    node = next(n for n in eg._classes[cid].nodes if n.op == "linear")
    wv = torch.zeros(o, i, dtype=torch.float64)
    a = {
        "o": o,
        "i": i,
        "b": {
            "keep": tuple(range(i)),
            "imap": tuple([0] * o),
            "firsts": (),
            "zero_appended": True,
            "bound": 1e-5,
            "all_dead": True,
        },
    }
    assert (
        _m_elide_bounded(
            eg, node, "linear", node.children[0], w, wv, a, {}
        )
        is None
    )


# ---------------------------------------------------------------------------
#  search(..., error_budget=...) — the bound gate and the ledger
# ---------------------------------------------------------------------------


def test_search_budget_accepts_bounded_member():
    """Within budget: the bounded member is extracted and the ledger
    records it — never silent."""
    torch.manual_seed(7)
    model = _NearDead().eval()
    xv = torch.randn(8, 32, dtype=torch.float64)
    res = Optimizer(backend=TorchBackend()).search(
        model,
        xv,
        detect_specials=True,
        error_budget=1e-3,
        cost_fn=flops_cost,
    )
    assert res.stats["error_budget"] == 1e-3
    assert res.stats["error_bound_total"] == pytest.approx(8e-4)
    entries = res.stats["error_bounds"]
    assert len(entries) == 1
    e = entries[0]
    assert e["rule"].startswith("weight_special#")
    assert e["bound"] == pytest.approx(8e-4)
    assert e["norm"] == "max_abs"
    assert e["measured_max_rel"] is None  # filled by lower's verify
    assert "__bl" in op_repr(res.term)


def test_search_budget_none_unchanged():
    """``error_budget=None`` (default): exact-only, no ledger keys."""
    torch.manual_seed(8)
    model = _NearDead().eval()
    xv = torch.randn(8, 32, dtype=torch.float64)
    res = Optimizer(backend=TorchBackend()).search(
        model, xv, detect_specials=True, cost_fn=flops_cost
    )
    assert "error_budget" not in res.stats
    assert "error_bound_total" not in res.stats
    assert "error_bounds" not in res.stats
    # Near-dead rows are not bitwise structure — the exact pass
    # offers nothing and the extracted term stays dense.
    assert "weight_specials" not in res.stats
    assert "__bl" not in op_repr(res.term)
    assert res.term.op == "linear"


def test_search_budget_engine_guards():
    """``error_budget`` needs the reference engine's witnesses."""
    torch.manual_seed(9)
    model = _NearDead().eval()
    xv = torch.randn(8, 32, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())

    class _StubEngine:
        """Not an ``EGraph`` — satisfies ``_resolve_engine``'s duck."""

        def add_term(self, term):
            return 0

    with pytest.raises(TypeError, match="reference EGraph"):
        opt.search(model, xv, error_budget=1e-3, engine=_StubEngine())
    with pytest.raises(TypeError, match="truncation_level"):
        opt.search(
            model,
            xv,
            error_budget=1e-3,
            engine=EGraph(truncation_level=1),
        )
    # Level-2 EGraph passes; a ``None`` budget skips the check
    # entirely (even for the stub engine).
    _check_budget_engine(EGraph(), 1e-3)
    _check_budget_engine(_StubEngine(), None)


def test_search_budget_over_budget_reextracts_exact():
    """Two bounded members each fit the offer-time budget but their
    sum overflows the search budget — the ban loop falls back to the
    exact members and the ledger stays honest."""
    torch.manual_seed(10)
    model = _NearDeadSplit().eval()
    xv = torch.randn(8, 64, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(
        model,
        xv,
        detect_specials=True,
        error_budget=1e-3,  # each bound is 8e-4; the sum 1.6e-3 is over
        cost_fn=flops_cost,
    )
    assert res.stats["error_bound_total"] <= 1e-3
    assert "__bl" not in op_repr(res.term)
    assert res.stats["error_bounds"] == []
    low = opt.lower(res, xv)
    assert low.verified.passed


def test_search_budget_ledger_two_bounded_sites():
    """Two delivered bounded members: the ledger sums both bounds —
    per-site, never silently dropped."""
    torch.manual_seed(11)
    model = _NearDeadSplit().eval()
    xv = torch.randn(8, 64, dtype=torch.float64)
    res = Optimizer(backend=TorchBackend()).search(
        model,
        xv,
        detect_specials=True,
        error_budget=2e-3,
        cost_fn=flops_cost,
    )
    entries = res.stats["error_bounds"]
    assert len(entries) == 2
    assert all(e["bound"] == pytest.approx(8e-4) for e in entries)
    assert res.stats["error_bound_total"] == pytest.approx(1.6e-3)


def test_bound_ledger_covers_cert_gaps():
    """A derivation gap (``egraph_dependent`` stub) contributes no
    bound to ``cert.error_bound`` — the fingerprint scan still lands
    every delivered bound member in the ledger."""
    eg, eid, _src, _t = _bounded_site()
    term = eg.extract_best(eid, flops_cost)
    assert "__bl" in op_repr(term)
    stub_cert = mock.Mock()
    stub_cert.steps = []
    stub_cert.rules = {}
    stub_cert.error_bound = 0.0  # what a fully-gapped cert reports
    entries, total = opt_mod._bound_ledger(eg, stub_cert, term)
    assert total == pytest.approx(8e-4)
    assert len(entries) == 1
    assert entries[0]["rule"].startswith("weight_special#")
    assert entries[0]["bound"] == pytest.approx(8e-4)
    # And a term delivering no bound member stays empty/zero.
    clean_entries, clean_total = opt_mod._bound_ledger(
        eg, stub_cert, _src
    )
    assert clean_entries == [] and clean_total == 0.0


def test_search_budget_no_bounded_offers():
    """Budget set, nothing bounded offered: ledger keys exist and
    read zero — the request is still on the record."""
    torch.manual_seed(12)

    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(32, 64, bias=False).double()

        def forward(self, t):
            return self.lin(t)

    res = Optimizer(backend=TorchBackend()).search(
        Full().eval(),
        torch.randn(8, 32, dtype=torch.float64),
        error_budget=1e-3,
    )
    assert res.stats["error_budget"] == 1e-3
    assert res.stats["error_bound_total"] == 0.0
    assert res.stats["error_bounds"] == []


# ---------------------------------------------------------------------------
#  lower — verify-with-bound, honored and dishonest deliveries
# ---------------------------------------------------------------------------


def test_lower_verify_honored_bound():
    """Delivered error within the propagated output bound: the bound
    gate holds, and each ledger entry gets the measured value plus
    its propagated ``output_bound``."""
    torch.manual_seed(13)
    model = _NearDead(scale=5.0, eps=8e-4).eval()
    xv = torch.randn(8, 32, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(
        model,
        xv,
        detect_specials=True,
        error_budget=1e-3,
        cost_fn=flops_cost,
    )
    low = opt.lower(res, xv)
    assert low.verified.passed
    assert low.stats["error_bounds_honored"] is True
    e = low.stats["error_bounds"][0]
    assert e["measured_max_rel"] == pytest.approx(low.verified.max_rel)
    # The weight-space bound propagated through the measured site
    # input norm — both units ride the ledger.
    assert e["site_input"] == "site"
    assert e["site_input_norm"] == pytest.approx(
        xv.abs().sum(-1).max().item()
    )
    assert e["output_bound"] == pytest.approx(
        8e-4 * e["site_input_norm"]
    )
    assert low.verified.max_abs <= e["output_bound"]
    assert low.stats["error_bound_output"] == pytest.approx(
        e["output_bound"]
    )


def test_lower_verify_measured_rel_above_weight_bound_honored():
    """The unit fix: a measured ``max_rel`` ABOVE the weight-space
    bound but inside the propagated output bound now delivers —
    the old gate declined it on mismatched units."""
    torch.manual_seed(14)
    model = _NearDead(scale=1.0, eps=8e-4).eval()
    xv = torch.randn(8, 32, dtype=torch.float64)
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(
        model,
        xv,
        detect_specials=True,
        error_budget=1e-3,
        cost_fn=flops_cost,
    )
    bound = res.stats["error_bound_total"]
    low = opt.lower(res, xv, rtol=1e-4)
    # Measured exceeds the weight-space claim (input-norm amplified)…
    assert low.verified.max_rel > bound
    # …but stays within the propagated output bound — honored, so
    # the delivery passes even though the bare rtol was tighter.
    e = low.stats["error_bounds"][0]
    assert low.verified.max_abs <= e["output_bound"]
    assert low.verified.passed
    assert low.stats["error_bounds_honored"] is True


def test_lower_verify_over_bound_declines():
    """A delivery whose measured error exceeds the *propagated*
    bound declines — the gate fails closed on bound violation."""
    stats = {
        "error_bounds": [
            {
                "rule": "weight_special#0",
                "bound": 1e-3,
                "norm": "max_abs",
                "output_bound": 1e-3,
            }
        ]
    }

    class R:
        max_abs = 2e-3  # measured output error exceeds the bound
        max_rel = 2e-3
        passed = True

    out = opt_mod._bound_verify(stats, R(), 1e-3, 1e-3)
    assert out.passed is False
    assert stats["error_bounds_honored"] is False
    assert stats["error_bounds"][0]["measured_max_rel"] == 2e-3


def test_lower_verify_unpropagatable_bound_declines():
    """A bound whose norm has no propagation rule — or whose site
    input cannot be measured — fails closed: ``output_bound`` stays
    ``None`` and the gate declines."""
    stats = {
        "error_bounds": [
            {"rule": "r#1", "bound": 1e-3, "norm": "chebyshev"},
            {
                "rule": "r#2",
                "bound": 1e-3,
                "norm": "max_abs",
            },
        ]
    }
    # Unknown norm -> no exponent -> unpropagatable; and a max_abs
    # entry on a non-tensor input has nothing to amplify by.
    out = opt_mod._propagate_bounds(stats, None, None, {}, None, (3.0,))
    assert out is None
    assert stats["error_bound_output"] is None
    e0, e1 = stats["error_bounds"]
    assert e0["output_bound"] is None and e0["site_input"] == "none"
    assert e1["output_bound"] is None and e1["site_input"] == "none"

    class R:
        max_abs = 1e-6
        max_rel = 1e-6
        passed = True

    out2 = opt_mod._bound_verify(stats, R(), 1e-3, out)
    assert out2.passed is False
    assert stats["error_bounds_honored"] is False


def test_propagate_bounds_block_input_fallback():
    """A bound rule whose witness LHS is not a projection site —
    or whose site subterm cannot be lowered — falls back to the
    module input's row norm, honestly recorded."""
    eg, _eid, _src, _t = _bounded_site()
    name, rule = next(
        (n, r) for n, r in opt_mod._bound_rules(eg).items()
    )
    xv = torch.randn(8, 32, dtype=torch.float64)

    # (a) Non-projection LHS — no isolable site input.
    from catopt_core.egraph.types import Rewrite

    eg._rule_objs["fake#0"] = Rewrite(
        name="fake#0",
        lhs=Var("x", TensorType((8, 32))),
        rhs=rule.rhs,
        error_bound=1e-3,
        bound_norm="max_abs",
    )
    stats = {
        "error_bounds": [
            {"rule": "fake#0", "bound": 1e-3, "norm": "max_abs"}
        ]
    }
    out = opt_mod._propagate_bounds(
        stats, eg, None, {}, sink=None, args=(xv,)
    )
    amp = xv.abs().sum(-1).max().item()
    assert out == pytest.approx(1e-3 * amp)
    assert stats["error_bounds"][0]["site_input"] == "block_input"
    assert stats["error_bounds"][0]["output_bound"] == pytest.approx(
        1e-3 * amp
    )

    # (b) Site resolution works but the site subterm fails to lower —
    # the eval failure still falls back to the input norm.
    xv2 = _src.args[0]
    ir = IR(root=_src, inputs=[xv2], input_names={xv2.name}, params={})
    sink = mock.Mock()
    sink.lower.side_effect = RuntimeError("no lowering")
    stats2 = {
        "error_bounds": [
            {"rule": name, "bound": 8e-4, "norm": "max_abs"}
        ]
    }
    out2 = opt_mod._propagate_bounds(
        stats2, eg, ir, {}, sink, (xv,)
    )
    assert out2 == pytest.approx(8e-4 * amp)
    assert stats2["error_bounds"][0]["site_input"] == "block_input"

    # (c) The site subterm evaluates but to a non-tensor — the row
    # norm cannot measure it, and the fallback still prices the
    # entry.
    mod = mock.Mock()
    mod.forward.return_value = 3.0  # not tensor-like
    sink2 = mock.Mock()
    sink2.lower.return_value = mod
    stats3 = {
        "error_bounds": [
            {"rule": name, "bound": 8e-4, "norm": "max_abs"}
        ]
    }
    out3 = opt_mod._propagate_bounds(
        stats3, eg, ir, {}, sink2, (xv,)
    )
    assert out3 == pytest.approx(8e-4 * amp)
    assert stats3["error_bounds"][0]["site_input"] == "block_input"


def test_bound_helper_units():
    """``_row_norm`` / ``_rel_bound`` / ``_propagate_bounds`` edge
    cases — non-tensor inputs, empty ledgers, degenerate reports."""
    xv = torch.randn(4, 8, dtype=torch.float64)
    p1 = opt_mod._row_norm(xv, 1.0)
    p2 = opt_mod._row_norm(xv, 2.0)
    assert p1 == pytest.approx(xv.abs().sum(-1).max().item())
    assert p2 == pytest.approx(
        (xv.abs() ** 2).sum(-1).sqrt().max().item()
    )
    assert p2 <= p1  # L2 ≤ L1 always
    assert opt_mod._row_norm(3.5, 1.0) is None  # no .abs

    class _WeirdT:
        """Tensor-ish: ``abs`` works but ``sum`` explodes."""

        def abs(self):
            return self

        def sum(self, *a):
            raise RuntimeError("nope")

    assert opt_mod._row_norm(_WeirdT(), 1.0) is None

    # An empty ledger propagates to a zero output bound.
    assert opt_mod._propagate_bounds({}, None, None, {}, None, ()) == 0.0

    class R:
        def __init__(self, max_abs, max_rel):
            self.max_abs = max_abs
            self.max_rel = max_rel

    import math

    assert math.isnan(opt_mod._rel_bound(None, R(1.0, 1.0)))
    assert opt_mod._rel_bound(2.0, R(0.0, 0.0)) == float("inf")
    assert opt_mod._rel_bound(2.0, R(0.5, 0.25)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
#  Ledger/ban units — the certificate-independent accounting
# ---------------------------------------------------------------------------


def _bounded_site(eps=8e-4):
    """A graph + tensors with one near-dead-row linear offered."""
    o, i = 64, 32
    wv = _near_dead(eps=eps)
    x = Var("x", TensorType((8, i)))
    w = Param("p_w", TensorType((o, i)))
    eg, eid = _site(Op.make("linear", x, w))
    tensors = {"p_w": wv}
    offer_weight_specials(eg, tensors, budget=1e-3)
    return eg, eid, Op.make("linear", x, w), tensors


def test_bound_ledger_and_ban_units():
    """``_ban_bound_members``: delivered -> bans; clean term -> no
    progress; second call -> bans already held."""
    eg, eid, src, _t = _bounded_site()
    term = eg.extract_best(eid, flops_cost)
    assert "__bl" in op_repr(term)
    bans: dict[int, set] = {}
    bound = opt_mod._bound_rules(eg)
    assert _ban_bound_members(eg, term, bound, bans) is True
    assert bans
    # The same term bans nothing new a second time.
    assert _ban_bound_members(eg, term, bound, bans) is False
    # The exact member delivers no bound member at all.
    assert _ban_bound_members(eg, src, bound, {}) is False


def test_bounded_term_reextracts_and_raises():
    """``_bounded_term`` returns the cheapest ledger-fitting member;
    exhaustible extraction and a stall both raise."""
    eg, eid, src, _t = _bounded_site()
    term, entries, total = _bounded_term(
        eg, eid, src, flops_cost, error_budget=5e-4
    )
    assert total <= 5e-4
    assert entries == []
    assert "__bl" not in op_repr(term)

    # In-budget: the bounded member comes straight back.
    term2, entries2, total2 = _bounded_term(
        eg, eid, src, flops_cost, error_budget=1e-3
    )
    assert total2 == pytest.approx(8e-4)
    assert [e["rule"] for e in entries2]
    assert "__bl" in op_repr(term2)

    # Degenerate graph: extraction itself returns None.
    dead = EGraph()
    eid0 = dead.add_term(Var("x", TensorType((4, 4))))
    with mock.patch.object(
        dead, "extract_best", return_value=None
    ), pytest.raises(OptimizationResourceError, match="certifies"):
        _bounded_term(dead, eid0, src, flops_cost, 1e-3)

    # Stall: the ban pass cannot name the bound member to exclude.
    eg2, eid2, src2, _t2 = _bounded_site()
    with mock.patch.object(
        opt_mod, "_ban_bound_members", return_value=False
    ), pytest.raises(OptimizationResourceError, match="exclude"):
        _bounded_term(eg2, eid2, src2, flops_cost, 5e-4)


def test_bounded_gate_matrix():
    """``_bounded_gate`` = sink pass AND measured ``max|Δy|`` within
    the propagated OUTPUT bound (absolute units — the bound is in
    output space, not the certificate's weight space)."""

    class R:
        def __init__(self, passed, max_abs):
            self.passed = passed
            self.max_abs = max_abs

    assert _bounded_gate(R(True, 0.5), 0.0)
    assert not _bounded_gate(R(False, 0.0), 0.0)
    assert _bounded_gate(R(True, 0.5), 1.0)
    assert not _bounded_gate(R(True, 1.5), 1.0)
    assert not _bounded_gate(R(False, 0.5), 1.0)
    # An unpropagatable bound (None) fails closed.
    assert not _bounded_gate(R(True, 0.5), None)


# ---------------------------------------------------------------------------
#  Manifest honesty + compositional plumbing
# ---------------------------------------------------------------------------


def test_export_manifest_carries_bound_fields(tmp_path):
    """The bounded ledger keys ride into the manifest verbatim —
    including a False ``error_bounds_honored``."""
    mod = nn.Linear(4, 4).eval()
    stats = {
        "error_budget": 1e-3,
        "error_bound_total": 8e-4,
        "error_bounds": [{"rule": "weight_special#3", "bound": 8e-4}],
        "error_bounds_honored": False,
    }
    manifest = export_optimized(
        None, mod, tmp_path / "w.safetensors", fmt="safetensors", stats=stats
    )
    for k, v in stats.items():
        assert manifest[k] == v

    # A stats dict without bound keys adds nothing.
    m2 = export_optimized(
        None,
        mod,
        tmp_path / "w2.safetensors",
        fmt="safetensors",
        stats={"runner": "x"},
    )
    assert "error_bounds" not in m2
    assert "error_budget" not in m2
    # And no stats at all keeps the old manifest shape.
    m3 = export_optimized(
        None, mod, tmp_path / "w3.safetensors", fmt="safetensors"
    )
    assert "error_bound_total" not in m3


class _BoundBlock(nn.Module):
    """One near-dead Linear — the bounded-offer block fixture."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(32, 32, bias=False).double()
        with torch.no_grad():
            self.lin.weight.mul_(5.0)
            self.lin.weight[8:24] = 0.0
            idx = torch.arange(8, 24)
            self.lin.weight[idx, idx % 32] = 8e-4

    def forward(self, t):
        return self.lin(t)


class _BoundStack(nn.Module):
    """Two ``_BoundBlock`` stages — a composable stack fixture."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(*[_BoundBlock() for _ in range(2)])

    def forward(self, t):
        return self.net(t)


def test_compositional_budget_forwards():
    """The compositional driver forwards ``error_budget`` /
    ``detect_specials`` / ``detect_factors`` to each per-block
    search — a bounded block's verify gates on the propagated
    OUTPUT bound, and the ledger records it."""
    torch.manual_seed(15)

    model = _BoundStack().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=Compositional(),
        error_budget=1e-3,
        detect_specials=True,
        detect_factors=True,
        # flop pricing selects the bounded member — the executor-aware
        # default can honestly decline it (extra gather launches).
        cost_fn=flops_cost,
        verbose=False,
    )
    assert stats["n_optimized"] == 2
    for rep in stats["blocks"].values():
        assert rep["status"] == "optimized"
        assert rep["stats"]["error_budget"] == 1e-3
        assert rep["stats"]["error_bound_total"] <= 1e-3
        # The driver verifies against the propagated output bound and
        # records the verdict on the block's ledger.
        assert rep["stats"]["error_bounds_honored"] is True
        e = rep["stats"]["error_bounds"][0]
        assert e["output_bound"] == pytest.approx(
            e["bound"] * e["site_input_norm"]
        )
        assert e["output_bound"] > e["bound"]
    _ = opt


# ---------------------------------------------------------------------------
#  Morphism lane — error_budget / detect_* forwarding (gap 2)
# ---------------------------------------------------------------------------


def test_morphism_search_forwards_budget_kwargs():
    """``optimize(strategy=MorphismSearch, error_budget=,
    detect_specials=, detect_factors=)`` reaches
    ``_optimize_morphisms`` — forwarded to each per-block fallback
    search, which bound-gates on the propagated output bound."""
    from catopt_orchestrator.morphisms import MorphismSearch

    torch.manual_seed(16)
    model = _BoundStack().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(laws=[], optimize_rest=True),
        error_budget=1e-3,
        detect_specials=True,
        detect_factors=True,
        # flop pricing reaches the per-block fallback when supplied —
        # it is what selects the bounded member over the dense form.
        cost_fn=flops_cost,
        verbose=False,
    )
    assert stats["n_blocks"] == 2
    for rep in stats["blocks"].values():
        assert rep["status"] == "optimized"
        assert rep["stats"]["error_budget"] == 1e-3
        assert 0.0 < rep["stats"]["error_bound_total"] <= 1e-3
        assert rep["stats"]["error_bounds_honored"] is True


def test_morphism_search_accepts_budget_without_blocks():
    """The kwarg is accepted even when nothing needs the fallback."""
    from catopt_orchestrator.morphisms import MorphismSearch

    torch.manual_seed(17)
    model = _BoundStack().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    _mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=MorphismSearch(laws=[], optimize_rest=False),
        error_budget=1e-3,
        detect_specials=True,
    )
    assert stats["blocks"] == {}


def test_arm_law_budgets():
    """``error_budget`` arms budget-aware laws left at ``None`` —
    on a copy, never mutating the caller's law — and leaves
    explicitly-budgeted and budget-free laws alone."""
    from catopt_orchestrator.morphisms import (
        WeightTie,
        _arm_law_budgets,
    )
    from catopt_orchestrator.morphisms_kv import KVLatentShare

    kv = KVLatentShare()
    (armed,) = _arm_law_budgets([kv], 1e-3)
    assert armed is not kv and armed.budget == 1e-3
    assert kv.budget is None  # the caller's object is untouched

    kv2 = KVLatentShare(budget=5e-4)
    (out,) = _arm_law_budgets([kv2], 1e-3)
    assert out is kv2 and out.budget == 5e-4  # explicit budget wins

    tie = WeightTie()  # no budget attribute at all
    (out2,) = _arm_law_budgets([tie], 1e-3)
    assert out2 is tie and not hasattr(tie, "budget")

    # ``None`` budget: laws pass through untouched.
    assert _arm_law_budgets([kv], None) == (kv,)
