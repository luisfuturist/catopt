# ruff: noqa: RUF002
"""Regression pins on the falsified corners — the negatives that must
never silently regress into claimed wins.

Measured background lives on the ``project`` orphan branch
(``adrs/0001-weight-space-structure-falsified.md``,
``retros/weight-as-programs.md``): weight-space compression is closed
(trained weights are entropy-dense; norm bounds are not quality
bounds) — the ε axis now lives on the ``weight-eps`` branch.  These
tests pin the code-side consequences on main:

  * Under the storage cost axis, the unmerged-adapter archetype
    reports zero savings — pairing must not bill materialised copies
    as free (the phantom −82.8% win, see tests/test_exact_corner.py).
"""

import torch
import torch.nn as nn

from catopt.cost import param_bytes_cost_for
from catopt.optimize import optimize_model, param_report


def _randn(shape, g):
    return torch.randn(*shape, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------------------
#  unmerged adapter: storage billing never reports a phantom win
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
    # no materialised copy survived extraction (the paired ``fused_*``
    # concat would have shown up as a stored parameter)
    assert not any(n.startswith("fused_") for n in low.state_dict())
