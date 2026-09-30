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
    """Delivered error within the accepted bound: verify widens its
    tolerance, the bound gate holds, and each ledger entry gets the
    measured value."""
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
    assert e["measured_max_rel"] <= res.stats["error_bound_total"]


def test_lower_verify_over_bound_declines():
    """A delivery whose measured ``max_rel`` exceeds the claimed
    bound declines — even when the (widened) tolerance would pass."""
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
    low = opt.lower(res, xv, rtol=1e-2)  # wide enough to pass loosely
    assert low.verified.max_rel > bound  # measured exceeds the claim
    assert low.verified.passed is False  # …so the bound gate declines
    assert low.stats["error_bounds_honored"] is False
    assert low.stats["error_bounds"][0]["measured_max_rel"] > bound


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
    """``_bounded_gate`` = sink pass AND measured within bound."""

    class R:
        def __init__(self, passed, max_rel):
            self.passed = passed
            self.max_rel = max_rel

    assert _bounded_gate(R(True, 0.5), 0.0)
    assert not _bounded_gate(R(False, 0.0), 0.0)
    assert _bounded_gate(R(True, 0.5), 1.0)
    assert not _bounded_gate(R(True, 1.5), 1.0)
    assert not _bounded_gate(R(False, 0.5), 1.0)


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


def test_compositional_budget_forwards():
    """The compositional driver forwards ``error_budget`` /
    ``detect_specials`` to each per-block search — a bounded block
    widens its verify tolerance and honors the bound."""
    torch.manual_seed(15)

    class Block(nn.Module):
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

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*[Block() for _ in range(2)])

        def forward(self, t):
            return self.net(t)

    model = Stack().eval()
    x = torch.randn(8, 32, dtype=torch.float64)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=Compositional(),
        error_budget=1e-3,
        detect_specials=True,
        verbose=False,
    )
    assert stats["n_optimized"] == 2
    for rep in stats["blocks"].values():
        assert rep["status"] == "optimized"
        assert rep["stats"]["error_budget"] == 1e-3
        assert rep["stats"]["error_bound_total"] <= 1e-3
    _ = opt
