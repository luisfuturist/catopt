"""Partial-install robustness — plan 0003 package split.

A ``catopt-*`` distribution must degrade gracefully when sibling
domain packages are absent:

* ``OpTable.full()`` skips carrier modules that cannot be imported —
  a carrier's terms cannot exist without its package, so the skip is
  semantically correct (e.g. a catopt-core + catopt-torch install);
* the skip is NARROWED to the carrier module itself: an import error
  raised inside an installed carrier (a missing dependency, reported
  under a different module name) still propagates;
* the ``expand`` torch binding accepts a scalar-int shape (a bare
  ``shape``/``dim`` attr) as well as the usual sequence — the old
  ``tuple(shape or dim or args)`` expression raised ``TypeError`` on
  scalar ints.
"""

import importlib
from types import SimpleNamespace

import catopt_core.ops as ops_mod
import pytest
import torch
from catopt.ops import OpTable
from catopt.torch_bridge import _IR_TO_TORCH

_real_import_module = importlib.import_module


def _import_blocker(blocked: str, missing: str | None = None):
    """An ``import_module`` stand-in that reports ``blocked`` as
    absent — under ``missing``'s name (defaults to the blocked module
    itself; a DIFFERENT name simulates a failed import inside the
    carrier, not a missing carrier)."""

    def fake(name, *a, **kw):
        if name == blocked:
            raise ModuleNotFoundError(
                f"No module named {missing or name!r}",
                name=missing or name,
            )
        return _real_import_module(name, *a, **kw)

    return SimpleNamespace(import_module=fake)


def test_full_skips_uninstalled_carrier(monkeypatch):
    """catopt-carriers.om absent → its ops absent from full(), other
    carriers unaffected — and the ambient dict's lazy resolver is
    skipped by the same path."""
    monkeypatch.setattr(
        ops_mod, "importlib", _import_blocker("catopt_carriers.om")
    )
    # A fresh merged dict — never ambient-seeded: om's ops are absent.
    fresh = ops_mod.carrier_torch_bindings()
    for op in ("cmask", "fill", "attnbias"):
        assert op not in fresh, op
    t = OpTable.full()
    # om's bindings may already sit in the ambient dict (seeded by an
    # earlier full() call this session): evict the stale seeds so the
    # checks exercise lazy resolution — which the patched import also
    # skips.
    for op in ("cmask", "fill", "attnbias"):
        monkeypatch.delitem(_IR_TO_TORCH, op, raising=False)
        assert op not in t.torch_bindings
    # every installed carrier still registers
    for op in ("trace", "omd_apply", "aquant"):
        assert op in t.torch_bindings, op


def test_full_reraises_errors_inside_a_carrier(monkeypatch):
    """The skip is narrowed by ``ModuleNotFoundError.name``: a missing
    import inside an installed carrier (reported under the missing
    dependency's name, not the carrier's) still propagates."""
    monkeypatch.setattr(
        ops_mod,
        "importlib",
        _import_blocker("catopt_carriers.om", missing="torch"),
    )
    with pytest.raises(ModuleNotFoundError, match="torch"):
        OpTable.full()


def test_expand_binding_accepts_scalar_and_sequence_shapes():
    """``shape``/``dim`` may arrive as a bare int — pass scalars
    through to ``t.expand``; only sequences are splatted."""
    expand = _IR_TO_TORCH["expand"]
    # zeros/arange, never randn: the assertions are shape-only and an
    # RNG draw here would shift the global stream for later tests.
    t = torch.zeros(2, 1)
    # sequence shapes — the usual export-boundary form
    assert expand(t, shape=(2, 4)).shape == (2, 4)
    assert expand(t, shape=[2, 4]).shape == (2, 4)
    # scalar extents/keep-dim — previously a TypeError.  (torch needs
    # ≥ ndim sizes, so scalars are exercised on a 1-dim tensor.)
    u = torch.zeros(1)
    assert expand(u, shape=4).shape == (4,)
    assert expand(u, shape=-1).shape == (1,)
    assert expand(u, dim=4).shape == (4,)
    # positional fallback still works
    assert expand(t, 2, 4).shape == (2, 4)
