# ruff: noqa: RUF002
"""Regression pins on the falsified corners — the negatives that must
never silently regress into claimed wins.

Measured background lives on the ``project`` orphan branch
(``adrs/0001-weight-space-structure-falsified.md``,
``retros/weight-as-programs.md``): weight-space compression is closed
(trained weights are entropy-dense; norm bounds are not quality
bounds), and the ε toolkit survives only where a norm bound IS the
contract.  These tests pin the code-side consequences:

  * ``eps.optimize_weight`` on a random (trained-like, entropy-dense)
    weight may offer compressed realizations — but every offer must
    carry an explicit nonzero certified bound; nothing is ever
    "free" compression.
  * Under the storage cost axis, the unmerged-adapter archetype
    reports zero savings — pairing must not bill materialised copies
    as free (the phantom −82.8% win, see tests/test_exact_corner.py).
  * ε offers are opt-in: ``optimize_model`` without ``eps_rtol``
    produces no ``eps_*`` members in the extracted program, even when
    a near-low-rank weight would have attracted one.
"""

import math

import torch
import torch.nn as nn

from catopt.cost import param_bytes_cost_for
from catopt.ir import Op, Param
from catopt.optimize import optimize_model, param_report

BYTES = 8  # fp64


def _randn(shape, g):
    return torch.randn(*shape, generator=g, dtype=torch.float64)


def _walk_term(term):
    """Yield every distinct node of an extracted term DAG."""
    seen = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        yield t
        if isinstance(t, Op):
            stack.extend(t.args)


# ---------------------------------------------------------------------------
#  1. optimize_weight: compressed offers are never bound-free
# ---------------------------------------------------------------------------


def test_optimize_weight_offers_carry_explicit_bounds():
    """A random dense weight is the entropy-dense (trained-like) case:
    certified compression may still be *offered* — e.g. int8 RTN — but
    every offer must carry an explicit nonzero error bound, and if
    extraction selects a smaller realization the certificate must
    report a nonzero accumulated bound.  Nothing is silently free."""
    from catopt.eps import optimize_weight

    g = torch.Generator().manual_seed(0)
    W = _randn((64, 64), g)
    res = optimize_weight("W", W, rtol=0.05)

    # offers are (kind, size, bound) triples; the bound is the contract
    for offer in res["offers"]:
        kind, _, bound = offer
        assert math.isfinite(bound) and bound > 0, (
            f"{kind} offer without a certified nonzero bound: {offer}"
        )

    # if the extracted program stores fewer bytes than the original
    # weight, the certificate must carry a nonzero bound — "free"
    # weight compression is exactly what was falsified.
    if res["bytes"] < res["orig_bytes"]:
        assert res["bound"] > 0
        assert not res["certificate"].exact
    assert res["bound"] == res["certificate"].error_bound


# ---------------------------------------------------------------------------
#  2. unmerged adapter: storage billing never reports a phantom win
# ---------------------------------------------------------------------------


class _AdapterUnmerged(nn.Module):
    """``base(x) + lb(la(x))`` — the corner where leaf-name dedup once
    made a materialised ``concat`` copy look free (a phantom −82.8%
    "saving").  Mirrors ``tests/test_exact_corner.py``; kept small.
    """

    I, O, R = 64, 64, 8  # noqa: E741

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.base = nn.Linear(self.I, self.O, bias=True)
        self.la = nn.Linear(self.I, self.R, bias=False)
        self.lb = nn.Linear(self.R, self.O, bias=False)
        with torch.no_grad():
            for lin in (self.base, self.la, self.lb):
                lin.weight.copy_(_randn(lin.weight.shape, g))

    def forward(self, x):
        return self.base(x) + self.lb(self.la(x))


def test_unmerged_adapter_no_phantom_savings():
    """Under ``param_bytes_cost`` the unmerged adapter has nothing to
    save honestly: the paired concat member re-stores every argument's
    rows and the fused ``linear(x, B@A)`` member materialises the dense
    product — both bill above the originals, so extraction keeps the
    unmerged form.  bytes_saved is exactly 0: never negative (billing
    regression), never positive (phantom compression)."""
    torch.manual_seed(0)
    model = _AdapterUnmerged().eval().double()
    x = torch.randn(4, _AdapterUnmerged.I, dtype=torch.float64)
    low, _stats = optimize_model(
        model, x, cost_fn=param_bytes_cost_for(), verbose=False
    )
    with torch.no_grad():
        ref = model(x.clone())
        out = low(x.clone())
    rel = (out - ref).abs().max().item() / (
        ref.abs().max().item() + 1e-8
    )
    assert rel < 1e-12  # exact

    r = param_report(model, low)
    assert r["bytes_saved"] >= 0  # storage billing never goes negative
    assert r["bytes_saved"] <= 0  # honest bound: nothing to save
    assert r["optimized_bytes"] == r["original_bytes"]
    # no materialised copy survived extraction — neither the paired
    # concat (``fused_*``) nor any ε-derived factor.
    assert not any(n.startswith("fused_") for n in low.state_dict())
    assert not any(n.startswith("eps_") for n in low.state_dict())


# ---------------------------------------------------------------------------
#  3. ε offers are opt-in — default extraction contains no eps_* members
# ---------------------------------------------------------------------------


class _NearLowRank(nn.Module):
    """A near-rank-8 weight — the shape that *would* attract
    ``eps.low_rank_params`` if the ε passes ran.  It must not: the
    toolkit is opt-in."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        U = _randn((64, 8), g)
        V = _randn((8, 64), g)
        self.lin = nn.Linear(64, 64, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(U @ V + 0.01 * _randn((64, 64), g))

    def forward(self, x):
        return self.lin(x) + x


def test_eps_members_require_opt_in():
    """``optimize_model`` without ``eps_rtol`` must produce no
    ``eps_*`` members — not even under the storage cost model that
    would select them if the offers existed.  Walked over the lowered
    term (``IRModule._root``): no op and no param is ``eps_*``; stats
    record no ``eps_offers``."""
    torch.manual_seed(0)
    model = _NearLowRank().eval().double()
    x = torch.randn(4, 64, dtype=torch.float64)
    # param_bytes_cost_for: the axis under which an eps offer, had one
    # been made, would win extraction — the strongest bait.
    low, stats = optimize_model(
        model, x, cost_fn=param_bytes_cost_for(), verbose=False
    )
    assert "eps_offers" not in stats

    ops = {t.op for t in _walk_term(low._root) if isinstance(t, Op)}
    params = {
        t.name for t in _walk_term(low._root) if isinstance(t, Param)
    }
    assert not [o for o in ops if o.startswith("eps_")]
    assert not [n for n in params if n.startswith("eps_")]
    assert not any(n.startswith("eps_") for n in low.state_dict())

    with torch.no_grad():
        ref = model(x.clone())
        out = low(x.clone())
    rel = (out - ref).abs().max().item() / (
        ref.abs().max().item() + 1e-8
    )
    assert rel < 1e-12  # exact
