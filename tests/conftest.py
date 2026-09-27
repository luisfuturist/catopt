"""Shared pytest support for catopt.

The ``requires_cuda`` marker marks tests that allocate CUDA tensors.
Such tests are skipped at setup — cleanly, not with a mid-test failure —
when

* ``torch.cuda.is_available()`` is false, or
* the device probes unusable: on a shared / oversubscribed card
  ``is_available()`` can still be true while the first real allocation
  raises ``torch.cuda.OutOfMemoryError`` (or a wedged driver raises at
  init).

The probe runs per test setup (not once per session) so a card that
fills up *between* tests still skips rather than flakes.  Probing is a
~32 MiB alloc + a kernel launch + sync — milliseconds.
"""

import sys
from pathlib import Path

import pytest
import torch

# --- mutmut sandbox ------------------------------------------------
# uv's PEP-660 editable install registers a single ``__editable__*``
# MetaPathFinder mapping every workspace package back to the real
# ``packages/*/src``.  It runs ahead of path-based resolution, so inside
# the flat sandbox mutmut builds (``.mutmut-sandbox/``, see
# tools/mutmut.sh) it would shadow the mutated copies and no mutant
# would ever be imported (mutmut then reports "no test case for any
# mutant").  When running from that sandbox, drop the finder so the
# ``mutants/`` copies win.  It is a no-op for normal runs.
if Path.cwd().name == "mutants":
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if "editable" not in getattr(f, "__module__", "")
    ]

_PROBE_NUMEL = 8 * 1024 * 1024  # float32 zeros → 32 MiB


def _cuda_skip_reason() -> str | None:
    """``None`` when CUDA is genuinely usable; a skip reason otherwise."""
    if not torch.cuda.is_available():
        return "requires_cuda: no CUDA device"
    try:
        x = torch.zeros(_PROBE_NUMEL, device="cuda")
        float(x.sum())  # launch a kernel, not just an alloc
        torch.cuda.synchronize()
        del x
    except Exception as e:
        return (
            f"requires_cuda: CUDA present but unusable "
            f"({type(e).__name__}: {e})"
        )
    return None


def cuda_usable() -> bool:
    """Imperative form for tests that want to branch, not skip."""
    return _cuda_skip_reason() is None


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_cuda") is None:
        return
    reason = _cuda_skip_reason()
    if reason is not None:
        pytest.skip(reason)
