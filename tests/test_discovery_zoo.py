"""Tests for ``catopt_discovery.zoo`` — the held-out model zoo.

The zoo is the holdout set for the honest yield question
(``project/retros/model-zoo-yield.md``): real ``nn.Module``
architecture families the intake corpus never contained, used to
measure whether the shipped laws and the discovery pipeline produce
anything on programs nobody wrote to spell them.  The tests pin the
two properties that make the measurement honest:

* **disjointness** — no zoo name collides with an intake candidate
  (so ``--real-only`` and the full-corpus runs can never have seen
  these modules), nor with the ``_PURPOSE_BUILT`` ledger, the model
  corpus names, the bench law-case names, or the persisted
  ``tools/intake_corpus.json`` records;
* **viability** — every workload's thunk builds and the module runs
  fp64 to a finite output, and the intake's own ``ingest`` boundary
  consumes the zoo unchanged (the holdout is a provenance property,
  not a different path).
"""

import pytest
import torch
from catopt_discovery import intake as li
from catopt_discovery import zoo as zoo_mod
from catopt_torch.adapters import TorchSink

_ZOO = zoo_mod.zoo()


def test_zoo_is_nonempty_and_wellformed():
    names = [w.name for w in _ZOO]
    assert len(names) == len(set(names)) >= 10
    assert all(w.kind == "zoo" for w in _ZOO)
    assert all(not w.purpose_built for w in _ZOO)


def test_zoo_disjoint_from_intake_registry():
    """The holdout property: the intake has never seen these names."""
    zoo_names = {w.name for w in _ZOO}
    intake_names = {w.name for w in li.candidates()}
    assert zoo_names.isdisjoint(intake_names)
    # The ledger only marks intake workloads, but pin the overlap
    # explicitly — a name shared with _PURPOSE_BUILT would let a
    # future edit silently reclassify a zoo member.
    assert zoo_names.isdisjoint(li._PURPOSE_BUILT)


def test_zoo_disjoint_from_model_and_bench_corpus():
    """The corpus is bench + models + intake — check the other two."""
    from catopt_discovery.impact import _model_cases

    from bench.suites.correctness.law_bench import LAW_CASES

    zoo_names = {w.name for w in _ZOO}
    model_names = {name for name, _m, _x in _model_cases()}
    assert zoo_names.isdisjoint(model_names)
    assert zoo_names.isdisjoint(set(LAW_CASES))


def test_zoo_not_persisted_in_intake_sidefile():
    """No zoo term may already sit in the appended census file."""
    zoo_names = {w.name for w in _ZOO}
    persisted = {
        r["name"].removeprefix("intake:") for r in li.load_records()
    }
    assert zoo_names.isdisjoint(persisted)


@pytest.mark.parametrize("w", _ZOO, ids=[w.name for w in _ZOO])
def test_zoo_model_runs_fp64(w):
    """Every zoo workload builds and produces a finite fp64 output."""
    torch.manual_seed(0)
    model, x = w.build()
    feed = x if isinstance(x, tuple) else (x,)
    model = model.eval().double()
    with torch.no_grad():
        out = model(*feed)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "name", ["HighwayGate", "LoRAAdapter", "FiLMHead"]
)
def test_zoo_ingests_through_the_real_boundary(name):
    """The intake's own ``ingest`` consumes the zoo unchanged — the
    holdout is provenance, not a different export path."""
    torch.manual_seed(0)
    picked = [w for w in _ZOO if w.name == name]
    cases, records, rejections = li.ingest(
        cands=picked, sink=TorchSink()
    )
    assert not rejections
    assert [r["status"] for r in records] == ["ingested"]
    assert cases and cases[0].source == "intake"
