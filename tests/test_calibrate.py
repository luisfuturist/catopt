"""Tests for catopt_torch.calibrate (TargetProfile + calibrate) and the
profile-parameterised cost fns in catopt_core.cost."""

import json
import math
from dataclasses import dataclass, field

import pytest
import torch
from catopt_torch.calibrate import (
    PROFILE_DIR_ENV,
    TargetProfile,
    calibrate,
    corrected_price_ns,
    list_profiles,
    load_profile,
    measured_price_ns,
    profile_graph_overhead_us,
    profiles_dir,
    record_measured,
    save_profile,
    shape_bucket,
)
from catopt_core.cost import (
    dag_cost,
    depth_cost_for,
    roofline_cost,
    roofline_cost_for,
)
from catopt_core.ir import Op, Param, TensorType, Var
import catopt_torch.calibrate as cal_mod


# The constants hardcoded in catopt_core.cost — the dev RTX 2050 profile.
RTX2050 = TargetProfile(
    name="RTX 2050",
    tflops=2.5,
    gbps=89.0,
    launch_us=8.7,
    device="cuda:0",
    measured_at="2025-01-01T00:00:00+00:00",
    meta={"dtype": "float32"},
)


def _mm_term():
    """Compute-bound: 256x256 @ 256x256 matmul (~34 MFLOP, ~0.8 MB)."""
    x = Var("x", TensorType((256, 256)))
    w = Param("W", TensorType((256, 256)))
    return Op.make("matmul", x, w)


def _ew_term():
    """Memory-bound: elementwise add over 4096x4096 (~17 MFLOP, ~201 MB)."""
    a = Var("a", TensorType((4096, 4096)))
    b = Var("b", TensorType((4096, 4096)))
    return Op.make("add", a, b)


# ---------------------------------------------------------------------------
# Serialisation / persistence
# ---------------------------------------------------------------------------


def test_profile_json_roundtrip():
    s = RTX2050.to_json()
    assert TargetProfile.from_json(s) == RTX2050
    # bytes and pre-parsed dicts work too
    assert TargetProfile.from_json(s.encode()) == RTX2050
    assert TargetProfile.from_json(json.loads(s)) == RTX2050
    # executor-overhead fields serialise with the rest
    assert "dispatch_us" in s and "leaf_eval_us" in s


def test_profile_executor_overhead_defaults():
    """Profiles built without the new constants — or loaded from a
    pre-probe JSON — get the conservative fallbacks, not zeros."""
    bare = TargetProfile("bare", 1.0, 2.0, 3.0, "cpu", "t")
    assert bare.dispatch_us > 0 and bare.leaf_eval_us > 0
    legacy = json.loads(RTX2050.to_json())
    del legacy["dispatch_us"], legacy["leaf_eval_us"]
    p = TargetProfile.from_json(legacy)
    assert p.dispatch_us == bare.dispatch_us
    assert p.leaf_eval_us == bare.leaf_eval_us


def test_profile_json_ignores_unknown_fields():
    data = json.loads(RTX2050.to_json())
    data["future_field"] = 42
    assert TargetProfile.from_json(data) == RTX2050


def test_profile_op_kernel_ns_defaults_and_roundtrip():
    """The measured kernel table defaults to {} — legacy JSONs and
    hand-built profiles get the empty (pure-roofline) table — and
    serialises with the rest of the profile."""
    bare = TargetProfile("bare", 1.0, 2.0, 3.0, "cpu", "t")
    assert bare.op_kernel_ns == {}
    legacy = json.loads(RTX2050.to_json())
    del legacy["op_kernel_ns"]
    assert TargetProfile.from_json(legacy).op_kernel_ns == {}
    table = {
        "matmul": {"128x128x128": 1.5e4},
        "pointwise": {"1048576": 4.0e5},
    }
    p = TargetProfile(
        "k", 1.0, 2.0, 3.0, "cpu", "t", op_kernel_ns=table
    )
    assert TargetProfile.from_json(p.to_json()) == p
    assert "op_kernel_ns" in p.to_json()


def test_profile_graph_overhead_and_measured_defaults():
    """``graph_overhead_us`` / ``measured_ns`` behave like the other
    profile extras: conservative defaults on hand-built and legacy
    profiles, clean serialisation round-trip."""
    bare = TargetProfile("bare", 1.0, 2.0, 3.0, "cpu", "t")
    assert bare.graph_overhead_us > 0  # fallback, never zero
    assert bare.measured_ns == {}
    legacy = json.loads(RTX2050.to_json())
    del legacy["graph_overhead_us"], legacy["measured_ns"]
    p = TargetProfile.from_json(legacy)
    assert p.graph_overhead_us == bare.graph_overhead_us
    assert p.measured_ns == {}
    measured = {
        "compiled": {"cpu:2^4": {"median_ns": 900.0, "model_ns": 300.0}}
    }
    q = TargetProfile(
        "k",
        1.0,
        2.0,
        3.0,
        "cpu",
        "t",
        graph_overhead_us=42.0,
        measured_ns=measured,
    )
    s = q.to_json()
    assert "graph_overhead_us" in s and "measured_ns" in s
    assert TargetProfile.from_json(s) == q


# ---------------------------------------------------------------------------
# Measured-feedback channel + graph overhead helpers
# ---------------------------------------------------------------------------


def test_shape_bucket():
    """Buckets key on device + ceil(log2 numel) — never global."""
    assert shape_bucket(torch.zeros(16)) == "cpu:2^4"
    assert shape_bucket(torch.zeros(2, 8)) == "cpu:2^4"
    # tuple inputs sum numels: 8 + 9 = 17 → ceil(log2) = 5
    assert shape_bucket((torch.zeros(8), torch.zeros(9))) == "cpu:2^5"
    # non-tensor / empty inputs land in the smallest bucket
    assert shape_bucket(3) == "cpu:2^0"
    assert shape_bucket(torch.zeros(0)) == "cpu:2^0"


def test_profile_graph_overhead_us_reader():
    """Absent / non-positive / unreadable values fall back; real ones
    pass through — the fused per-graph term is never optimistic."""
    fb = cal_mod._FALLBACK_GRAPH_OVERHEAD_US
    assert profile_graph_overhead_us(None) == fb
    assert profile_graph_overhead_us({}) == fb
    assert profile_graph_overhead_us({"graph_overhead_us": 0.0}) == fb
    assert (
        profile_graph_overhead_us({"graph_overhead_us": "junk"}) == fb
    )
    assert (
        profile_graph_overhead_us({"graph_overhead_us": 33.0}) == 33.0
    )
    # attribute profiles read the same way
    assert (
        profile_graph_overhead_us(RTX2050) == RTX2050.graph_overhead_us
    )

    class Obj:
        graph_overhead_us = 12.5

    assert profile_graph_overhead_us(Obj()) == 12.5


def test_measured_price_ns_contract():
    prof = {
        "measured_ns": {
            "compiled": {
                "cpu:2^4": {"median_ns": 900.0, "model_ns": 300.0}
            }
        }
    }
    # residual transfer: model 500 + (measured 900 - recorded 300)
    assert (
        measured_price_ns(prof, "compiled", "cpu:2^4", 500.0) == 1100.0
    )
    # the recorded graph prices at its measurement exactly
    assert (
        measured_price_ns(prof, "compiled", "cpu:2^4", 300.0) == 900.0
    )
    # per-bucket, never global: other buckets / candidates keep the model
    assert (
        measured_price_ns(prof, "compiled", "cpu:2^9", 500.0) == 500.0
    )
    assert measured_price_ns(prof, "generic", "cpu:2^4", 500.0) == 500.0
    # no model price available → measured median substitutes
    assert measured_price_ns(prof, "compiled", "cpu:2^4", None) == 900.0
    # nothing anywhere → unpriced
    assert measured_price_ns({}, "x", "cpu:2^4", None) is None
    # no profile at all → the model price passes through unchanged
    assert measured_price_ns(None, "x", "cpu:2^4", 7.0) == 7.0
    # entry without a recorded model → absolute substitution
    bare = {"measured_ns": {"eager": {"cpu:2^4": {"median_ns": 42.0}}}}
    assert measured_price_ns(bare, "eager", "cpu:2^4", 999.0) == 42.0
    # bare-number entries are allowed too
    num = {"measured_ns": {"eager": {"cpu:2^4": 123.0}}}
    assert measured_price_ns(num, "eager", "cpu:2^4", 999.0) == 123.0
    # a malformed entry (no median) leaves the model price alone
    nomed = {"measured_ns": {"x": {"b": {"model_ns": 1.0}}}}
    assert measured_price_ns(nomed, "x", "b", 7.0) == 7.0
    # malformed candidate table → entry absent → model price
    bad = {"measured_ns": {"x": 5}}
    assert measured_price_ns(bad, "x", "b", 7.0) == 7.0
    # attribute profiles read measured_ns the same way
    tp = TargetProfile(
        "k",
        1.0,
        2.0,
        3.0,
        "cpu",
        "t",
        measured_ns={
            "compiled": {
                "cpu:2^4": {"median_ns": 50.0, "model_ns": 10.0}
            }
        },
    )
    assert measured_price_ns(tp, "compiled", "cpu:2^4", 20.0) == 60.0

    # a non-dict measured_ns on an attribute profile is ignored
    class Weird:
        measured_ns = "junk"

    assert measured_price_ns(Weird(), "x", "b", 7.0) == 7.0


def test_record_measured():
    entry = {"median_ns": 100.0, "model_ns": 40.0}
    # dict profiles update in place
    d = {}
    assert record_measured(d, "compiled", "cpu:2^4", 100.0, 40.0) is d
    assert d["measured_ns"]["compiled"]["cpu:2^4"] == entry
    # accumulating other buckets / candidates preserves earlier entries
    record_measured(d, "compiled", "cpu:2^5", 200.0)
    record_measured(d, "generic", "cpu:2^4", 50.0, 10.0)
    assert set(d["measured_ns"]) == {"compiled", "generic"}
    assert set(d["measured_ns"]["compiled"]) == {"cpu:2^4", "cpu:2^5"}
    # an existing non-dict measured_ns is replaced, not crashed on
    d2 = {"measured_ns": "junk"}
    record_measured(d2, "g", "b", 1.0)
    assert d2["measured_ns"]["g"]["b"] == {"median_ns": 1.0}
    # frozen TargetProfile → a NEW object; the original is untouched
    p = TargetProfile("k", 1.0, 2.0, 3.0, "cpu", "t")
    p2 = record_measured(p, "generic", "cpu:2^4", 55.0, 20.0)
    assert p2 is not p and p.measured_ns == {}
    assert p2.measured_ns["generic"]["cpu:2^4"] == {
        "median_ns": 55.0,
        "model_ns": 20.0,
    }
    # a second write accumulates on the returned copy
    p3 = record_measured(p2, "compiled", "cpu:2^4", 77.0)
    assert set(p3.measured_ns) == {"generic", "compiled"}

    # plain objects get the attribute set and keep prior entries
    class Obj:
        pass

    obj = Obj()
    out = record_measured(obj, "g", "b", 3.0)
    assert out is obj
    record_measured(obj, "h", "b", 4.0)
    assert set(obj.measured_ns) == {"g", "h"}

    # a frozen dataclass WITHOUT the field → replace() raises
    @dataclass(frozen=True)
    class Frozen:
        x: int = 0

    with pytest.raises(TypeError):
        record_measured(Frozen(), "g", "b", 1.0)


# ---------------------------------------------------------------------------
# Learned correction table — record → learn → correct
# ---------------------------------------------------------------------------


def test_record_measured_learns_corrections():
    """Each priced record folds ``median/model`` into the (candidate,
    bucket) correction: a running geometric mean with a growing n."""
    d: dict = {}
    record_measured(d, "compiled", "cpu:2^4", 200.0, 100.0)
    assert d["corrections"]["compiled"]["cpu:2^4"] == {
        "factor": 2.0,
        "n": 1,
    }
    # second observation: geomean(2.0, 8.0) = 4.0
    record_measured(d, "compiled", "cpu:2^4", 800.0, 100.0)
    ent = d["corrections"]["compiled"]["cpu:2^4"]
    assert ent["n"] == 2 and ent["factor"] == pytest.approx(4.0)
    # corrections stay per (candidate, bucket) — never global
    record_measured(d, "generic", "cpu:2^4", 50.0, 10.0)
    record_measured(d, "compiled", "cpu:2^9", 30.0, 10.0)
    assert d["corrections"]["generic"]["cpu:2^4"]["factor"] == 5.0
    assert d["corrections"]["compiled"]["cpu:2^9"]["factor"] == 3.0
    # a measurement without a model price records but does not learn
    record_measured(d, "eager", "cpu:2^4", 42.0)
    assert "eager" not in d["corrections"]
    assert d["measured_ns"]["eager"]["cpu:2^4"] == {"median_ns": 42.0}


def test_record_measured_corrections_robust_inputs():
    """Unusable ratios leave the table unchanged; extreme ratios are
    clamped into the sane factor range before folding in."""
    d: dict = {}
    record_measured(d, "c", "b", 10.0, 0.0)  # division by zero
    record_measured(d, "c", "b", -5.0, 10.0)  # negative median
    record_measured(d, "c", "b", 10.0, -10.0)  # negative ratio
    record_measured(d, "c", "b", 10.0, float("inf"))  # ratio == 0
    record_measured(d, "c", "b", float("inf"), 10.0)  # non-finite
    record_measured(d, "c", "b", 10.0, float("nan"))
    assert d["corrections"] == {}
    # an extreme ratio is clamped to [_CORRECTION_LO, _CORRECTION_HI]
    record_measured(d, "c", "b", 1e6, 10.0)  # ratio 1e5 → 10.0
    assert d["corrections"]["c"]["b"] == {"factor": 10.0, "n": 1}
    record_measured(d, "c", "b", 1e-6, 10.0)  # ratio 1e-7 → 0.1
    assert d["corrections"]["c"]["b"]["factor"] == pytest.approx(1.0)


def test_learned_corrections_windowed_geometric_mean():
    """Once n passes the window, the factor is a bounded-memory
    average — a persistent regime shift still re-learns it."""
    d: dict = {}
    for _ in range(40):  # > _CORRECTION_WINDOW of ratio 2.0
        record_measured(d, "c", "b", 20.0, 10.0)
    ent = d["corrections"]["c"]["b"]
    assert ent["n"] == 40
    assert ent["factor"] == pytest.approx(2.0)
    # a new regime (ratio 8) pulls the saturated estimate toward it
    record_measured(d, "c", "b", 80.0, 10.0)
    f = d["corrections"]["c"]["b"]["factor"]
    assert 2.0 < f < 8.0
    for _ in range(200):
        record_measured(d, "c", "b", 80.0, 10.0)
    assert d["corrections"]["c"]["b"]["factor"] == pytest.approx(
        8.0, rel=0.05
    )


def test_learned_corrections_private_edge_cases():
    """Direct coverage for the paths ``record_measured``'s float()
    casts reject earlier, and malformed stored tables."""
    # non-numeric observation inputs → table unchanged
    assert cal_mod._learned_corrections(None, "c", "b", "x", 1.0) == {}
    assert cal_mod._learned_corrections(None, "c", "b", 1.0, "x") == {}
    # a non-dict stored table is rebuilt, not crashed on
    assert cal_mod._learned_corrections("junk", "c", "b", 2.0, 1.0) == {
        "c": {"b": {"factor": 2.0, "n": 1}}
    }
    # non-dict candidate values are dropped; a malformed entry for the
    # updated bucket re-learns from scratch (n restarts at 1)
    tab = cal_mod._learned_corrections(
        {"bad": 5, "ok": {"b": {"factor": "x", "n": 9}}},
        "ok",
        "b",
        20.0,
        10.0,
    )
    assert "bad" not in tab
    assert tab["ok"]["b"] == {"factor": 2.0, "n": 1}
    # an entry with n <= 0 also re-learns fresh
    tab = cal_mod._learned_corrections(
        {"c": {"b": {"factor": 9.0, "n": 0}}}, "c", "b", 30.0, 10.0
    )
    assert tab["c"]["b"] == {"factor": 3.0, "n": 1}
    # model_ns=None → table passed through untouched
    src = {"c": {"b": {"factor": 2.0, "n": 3}}}
    assert cal_mod._learned_corrections(src, "c", "b", 9.0, None) == src


def test_record_measured_corrections_profile_shapes():
    """Frozen profiles return new objects carrying corrections; dict
    candidates that aren't tables are rebuilt; exotic objects degrade
    gracefully."""
    # frozen TargetProfile → a NEW object; the original is untouched
    p = TargetProfile("k", 1.0, 2.0, 3.0, "cpu", "t")
    p2 = record_measured(p, "generic", "cpu:2^4", 55.0, 20.0)
    assert p2 is not p and p.corrections == {}
    ent = p2.corrections["generic"]["cpu:2^4"]
    assert ent["n"] == 1 and ent["factor"] == pytest.approx(2.75)
    # a second write accumulates on the returned copy
    p3 = record_measured(p2, "generic", "cpu:2^4", 82.5, 20.0)
    ent = p3.corrections["generic"]["cpu:2^4"]
    assert ent["n"] == 2
    assert ent["factor"] == pytest.approx(math.sqrt(2.75 * 4.125))

    # dict profile with a non-dict candidate table → rebuilt
    d = {"measured_ns": {"c": 5}, "corrections": {"c": 9}}
    record_measured(d, "c", "b", 10.0, 5.0)
    assert d["measured_ns"]["c"]["b"]["median_ns"] == 10.0
    assert d["corrections"]["c"]["b"] == {"factor": 2.0, "n": 1}

    # attribute profile whose measured_ns holds non-dict values →
    # they are dropped from the copy, not crashed on
    class Obj:
        pass

    obj = Obj()
    obj.measured_ns = {"x": 5}
    obj.corrections = "junk"
    out = record_measured(obj, "g", "b", 3.0, 1.0)
    assert out is obj
    assert obj.measured_ns["g"]["b"]["median_ns"] == 3.0
    assert "x" not in obj.measured_ns
    assert obj.corrections["g"]["b"] == {"factor": 3.0, "n": 1}

    # an object whose slots reject the corrections attribute still
    # records measured_ns — the learned table is best-effort
    class Slotted:
        __slots__ = ("measured_ns",)

    s = Slotted()
    record_measured(s, "g", "b", 3.0, 1.0)
    assert s.measured_ns["g"]["b"]["median_ns"] == 3.0
    assert not hasattr(s, "corrections")

    # a frozen dataclass with measured_ns but no corrections field:
    # the measured entry lands, learning is skipped
    @dataclass(frozen=True)
    class FrozenM:
        measured_ns: dict = field(default_factory=dict)

    fm = record_measured(FrozenM(), "g", "b", 3.0, 1.0)
    assert fm.measured_ns["g"]["b"]["median_ns"] == 3.0
    assert not hasattr(fm, "corrections")


def test_corrected_price_ns_contract():
    """Learned factors apply once ``n >= min_samples``; the measured_ns
    contract is the fallback ladder below that."""
    prof = {
        "corrections": {
            "compiled": {"cpu:2^4": {"factor": 2.0, "n": 5}}
        }
    }
    assert (
        corrected_price_ns(prof, "compiled", "cpu:2^4", 100.0) == 200.0
    )
    # below min_samples the factor is ignored → model unchanged
    prof1 = {
        "corrections": {
            "compiled": {"cpu:2^4": {"factor": 2.0, "n": 1}}
        }
    }
    assert (
        corrected_price_ns(prof1, "compiled", "cpu:2^4", 100.0) == 100.0
    )
    # ... but min_samples is a knob
    assert (
        corrected_price_ns(
            prof1, "compiled", "cpu:2^4", 100.0, min_samples=1
        )
        == 200.0
    )
    # a direct measured_ns entry is the fallback when n < min_samples
    both = {
        "corrections": {
            "compiled": {"cpu:2^4": {"factor": 2.0, "n": 1}}
        },
        "measured_ns": {
            "compiled": {
                "cpu:2^4": {"median_ns": 90.0, "model_ns": 300.0}
            }
        },
    }
    assert (
        corrected_price_ns(both, "compiled", "cpu:2^4", 300.0) == 90.0
    )
    # once learned, the factor takes precedence over the residual
    both["corrections"]["compiled"]["cpu:2^4"]["n"] = 3
    assert (
        corrected_price_ns(both, "compiled", "cpu:2^4", 300.0) == 600.0
    )
    # unpriceable candidates cannot take a factor → measured path
    assert corrected_price_ns(prof, "compiled", "cpu:2^4", None) is None
    assert corrected_price_ns(both, "compiled", "cpu:2^4", None) == 90.0
    # stored factors are clamped into the sane range at consumption
    wild = {"corrections": {"c": {"b": {"factor": 500.0, "n": 9}}}}
    assert corrected_price_ns(wild, "c", "b", 100.0) == 1000.0
    low = {"corrections": {"c": {"b": {"factor": 0.001, "n": 9}}}}
    assert corrected_price_ns(low, "c", "b", 100.0) == 10.0
    # malformed tables / entries are ignored entirely
    for bad in (
        {"corrections": "junk"},
        {"corrections": {"c": 5}},
        {"corrections": {"c": {"b": 5}}},
        {"corrections": {"c": {"b": {"factor": "x", "n": 9}}}},
        {"corrections": {"c": {"b": {"factor": 2.0}}}},  # missing n
        {"corrections": {"c": {"b": {"factor": float("nan"), "n": 9}}}},
        {"corrections": {"c": {"b": {"factor": -2.0, "n": 9}}}},
    ):
        assert corrected_price_ns(bad, "c", "b", 100.0) == 100.0
    # no profile → model passes through; nothing anywhere → None
    assert corrected_price_ns(None, "c", "b", 7.0) == 7.0
    assert corrected_price_ns(None, "c", "b", None) is None
    # attribute profiles read corrections the same way
    tp = TargetProfile(
        "k",
        1.0,
        2.0,
        3.0,
        "cpu",
        "t",
        corrections={"c": {"b": {"factor": 3.0, "n": 4}}},
    )
    assert corrected_price_ns(tp, "c", "b", 100.0) == 300.0

    class Weird:
        corrections = "junk"

    assert corrected_price_ns(Weird(), "c", "b", 7.0) == 7.0


def test_corrected_price_ns_nearest_bucket():
    """No exact-bucket correction → the nearest same-device bucket's
    factor applies dampened toward 1.0 by ``0.5 ** |Δexponent|``."""
    prof = {
        "corrections": {
            "compiled": {
                "cpu:2^8": {"factor": 4.0, "n": 5},
                "cuda:0:2^9": {"factor": 9.0, "n": 5},
            }
        }
    }
    # exact hit — full strength
    assert (
        corrected_price_ns(prof, "compiled", "cpu:2^8", 100.0) == 400.0
    )
    # one bucket away: 4 ** 0.5 = 2; two away: 4 ** 0.25
    assert corrected_price_ns(
        prof, "compiled", "cpu:2^9", 100.0
    ) == pytest.approx(200.0)
    assert corrected_price_ns(
        prof, "compiled", "cpu:2^10", 100.0
    ) == pytest.approx(100.0 * 4.0**0.25)
    # other candidates / other devices never interpolate
    assert (
        corrected_price_ns(prof, "generic", "cpu:2^9", 100.0) == 100.0
    )
    assert (
        corrected_price_ns(prof, "compiled", "cuda:1:2^8", 100.0)
        == 100.0
    )
    # cuda:0's own factor is exact and undampened — device prefix
    # parsing survives embedded colons
    assert (
        corrected_price_ns(prof, "compiled", "cuda:0:2^9", 100.0)
        == 900.0
    )
    # a nearer LOW-CONFIDENCE entry does not shadow the farther
    # learned one: n=1 is skipped, the 2^4 factor applies at dist 4
    mixed = {
        "corrections": {
            "c": {
                "cpu:2^9": {"factor": 8.0, "n": 1},
                "cpu:2^4": {"factor": 8.0, "n": 3},
            }
        }
    }
    assert corrected_price_ns(
        mixed, "c", "cpu:2^8", 10.0
    ) == pytest.approx(10.0 * 8.0**0.0625)
    # malformed entries are skipped during the nearest scan
    messy = {
        "corrections": {
            "c": {
                "cpu:2^5": "junk",
                "cpu:2^4": {"factor": 3.0, "n": 2},
            }
        }
    }
    assert corrected_price_ns(
        messy, "c", "cpu:2^6", 10.0
    ) == pytest.approx(10.0 * 3.0**0.25)
    # keys that don't parse exact-match but never interpolate
    weird = {"corrections": {"c": {"odd-key": {"factor": 3.0, "n": 5}}}}
    assert corrected_price_ns(weird, "c", "odd-key", 10.0) == 30.0
    assert corrected_price_ns(weird, "c", "cpu:2^4", 10.0) == 10.0
    # malformed query bucket: exact match only, no interpolation
    qc = {"corrections": {"c": {"x": {"factor": 2.0, "n": 2}}}}
    assert corrected_price_ns(qc, "c", "x", 10.0) == 20.0
    assert corrected_price_ns(qc, "c", "y", 10.0) == 10.0
    # entries whose own key doesn't parse are skipped, not crashed on
    bad_key = {
        "corrections": {"c": {"cpu:2^x": {"factor": 2.0, "n": 2}}}
    }
    assert corrected_price_ns(bad_key, "c", "cpu:2^4", 10.0) == 10.0
    # the NEAREST bucket wins: a farther entry seen later does not
    # displace it (cpu:2^5 is distance 1, cpu:2^2 distance 4)
    multi = {
        "corrections": {
            "c": {
                "cpu:2^5": {"factor": 4.0, "n": 3},
                "cpu:2^2": {"factor": 8.0, "n": 3},
            }
        }
    }
    assert corrected_price_ns(
        multi, "c", "cpu:2^6", 10.0
    ) == pytest.approx(10.0 * 4.0**0.5)


def test_bucket_coord_parsing():
    """The nearest-bucket lookup keys on (device, exponent); the
    parser rejects anything outside the ``<device>:2^e`` shape."""
    bc = cal_mod._bucket_coord
    assert bc("cpu:2^4") == ("cpu", 4)
    assert bc("cuda:0:2^10") == ("cuda:0", 10)  # colons in device
    assert bc("x") is None
    assert bc(5) is None
    assert bc(None) is None
    assert bc(":2^3") is None  # empty device
    assert bc("cpu:2^x") is None  # non-integer exponent


def test_correction_loop_flips_ordering_to_measured():
    """The closed loop on a toy profile: the model mis-ranks two
    candidates, one recorded measurement flips the delivered ordering
    to the measured truth, and the learned factor keeps it there."""
    prof: dict = {}
    model = {"generic": 100.0, "compiled": 300.0}
    bucket = "cpu:2^4"
    # before any measurement the (wrong) model ordering stands
    assert corrected_price_ns(
        prof, "compiled", bucket, model["compiled"]
    ) > corrected_price_ns(prof, "generic", bucket, model["generic"])
    # measured truth: compiled is CHEAPER — one record and the direct
    # residual already delivers the measurement exactly
    record_measured(prof, "compiled", bucket, 90.0, model["compiled"])
    assert (
        corrected_price_ns(prof, "compiled", bucket, model["compiled"])
        == 90.0
    )
    assert corrected_price_ns(
        prof, "compiled", bucket, model["compiled"]
    ) < corrected_price_ns(prof, "generic", bucket, model["generic"])
    # a second record reaches min_samples: the learned factor 0.3 now
    # drives the correction — same ordering, from the table
    record_measured(prof, "compiled", bucket, 90.0, model["compiled"])
    ent = prof["corrections"]["compiled"][bucket]
    assert ent["n"] == 2 and ent["factor"] == pytest.approx(0.3)
    assert corrected_price_ns(
        prof, "compiled", bucket, model["compiled"]
    ) == pytest.approx(90.0)
    # the learned ratio transfers to DIFFERENT model prices in the
    # same bucket — a changed graph keeps the correction
    assert corrected_price_ns(
        prof, "compiled", bucket, 500.0
    ) == pytest.approx(150.0)


def test_corrections_serialize_roundtrip():
    """``corrections`` rides the JSON schema like the other maps —
    clean round-trip, empty default on legacy profiles."""
    bare = TargetProfile("bare", 1.0, 2.0, 3.0, "cpu", "t")
    assert bare.corrections == {}
    legacy = json.loads(RTX2050.to_json())
    del legacy["corrections"]
    assert TargetProfile.from_json(legacy).corrections == {}
    corr = {"compiled": {"cpu:2^4": {"factor": 0.3, "n": 7}}}
    q = TargetProfile("k", 1.0, 2.0, 3.0, "cpu", "t", corrections=corr)
    s = q.to_json()
    assert '"corrections"' in s
    assert TargetProfile.from_json(s) == q


def test_measure_graph_overhead_direct(monkeypatch):
    """The probe's timing path is exercised without needing a real
    inductor: ``compile`` → identity keeps the residual ≥ 0."""
    monkeypatch.setattr(torch, "compile", lambda f: f)
    v = cal_mod._measure_graph_overhead(
        torch.device("cpu"), torch.float32, 2, 10, 2, 3e-6
    )
    assert v >= 0.0


def test_calibrate_graph_overhead_probe_failure_falls_back(monkeypatch):
    """A toolchain without inductor (or a failing probe) must not
    crash calibrate — the conservative fallback lands instead."""
    monkeypatch.setattr(
        cal_mod,
        "_measure_graph_overhead",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    p = calibrate(device="cpu", quick=True)
    assert p.graph_overhead_us == cal_mod._FALLBACK_GRAPH_OVERHEAD_US


def test_calibrate_graph_overhead_nonpositive_falls_back(monkeypatch):
    """A probe that returns zero — e.g. a clock coarser than the
    launch constant — is clamped to the fallback, never free."""
    monkeypatch.setattr(
        cal_mod, "_measure_graph_overhead", lambda *a, **k: 0.0
    )
    p = calibrate(device="cpu", quick=True)
    assert p.graph_overhead_us == cal_mod._FALLBACK_GRAPH_OVERHEAD_US


def test_profile_save_load(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path))
    assert profiles_dir() == tmp_path

    path = save_profile(RTX2050)
    assert path.parent == tmp_path and path.exists()

    assert load_profile(RTX2050.name) == RTX2050  # by name
    assert load_profile(path) == RTX2050  # by path
    assert RTX2050.name in list_profiles()

    # dir= parameter bypasses the env default
    other = tmp_path / "elsewhere"
    save_profile(RTX2050, dir=other)
    assert load_profile(RTX2050.name, dir=other) == RTX2050

    with pytest.raises(FileNotFoundError):
        load_profile("nonexistent-target")


# ---------------------------------------------------------------------------
# roofline_cost_for
# ---------------------------------------------------------------------------


def test_roofline_cost_for_matches_default():
    """The RTX 2050 profile must reproduce roofline_cost exactly."""
    fn = roofline_cost_for(RTX2050)
    for term in (
        _mm_term(),
        _ew_term(),
        Op.make("add", _mm_term(), Var("y", TensorType((256, 256)))),
    ):
        assert fn(term) == pytest.approx(roofline_cost(term))


def test_roofline_cost_for_no_profile_is_default():
    fn = roofline_cost_for()
    assert fn(_mm_term()) == pytest.approx(roofline_cost(_mm_term()))


def test_roofline_cost_for_is_a_cost_fn():
    """Standard cost-fn signature: fn(term, memo=None); works in dag_cost."""
    fn = roofline_cost_for(RTX2050)
    term = Op.make("add", _mm_term(), Var("y", TensorType((256, 256))))
    assert fn(term) > 0.0
    assert fn(term, memo={}) == pytest.approx(fn(term))
    # tree (no shared subtrees): dag_cost == additive cost
    assert dag_cost(term, fn) == pytest.approx(fn(term))
    # dict-shaped profiles plug in too
    fn_dict = roofline_cost_for(
        {"tflops": 2.5, "gbps": 89.0, "launch_us": 8.7}
    )
    assert fn_dict(term) == pytest.approx(fn(term))


def test_depth_cost_for_works():
    fn = depth_cost_for(RTX2050)
    x = Var("x", TensorType((64, 64)))
    term = Op.make("add", Op.make("neg", x), Op.make("mul", x, x))
    assert fn(term) > 0.0
    # critical path charges max(children) per op — two parallel unary
    # ops — while roofline charges the sum: depth < additive total.
    assert fn(term) < roofline_cost_for(RTX2050)(term)


def test_profile_flips_ordering():
    """Changing the target profile flips the predicted ordering between a
    compute-bound matmul and a memory-bound elementwise op."""
    mm, ew = _mm_term(), _ew_term()
    weak_compute = TargetProfile(
        "weak-compute",
        tflops=0.01,
        gbps=1000.0,
        launch_us=1.0,
        device="cpu",
        measured_at="t",
    )
    weak_memory = TargetProfile(
        "weak-memory",
        tflops=1000.0,
        gbps=0.01,
        launch_us=1.0,
        device="cpu",
        measured_at="t",
    )
    f_wc, f_wm = (
        roofline_cost_for(weak_compute),
        roofline_cost_for(weak_memory),
    )
    assert f_wc(mm) > f_wc(ew)  # compute-poor target: matmul loses
    assert f_wm(mm) < f_wm(
        ew
    )  # bandwidth-poor target: elementwise loses


# ---------------------------------------------------------------------------
# calibrate()
# ---------------------------------------------------------------------------


def test_calibrate_cpu_sane():
    """Loose, CI-friendly bounds — any real CPU lands inside these."""
    p = calibrate(device="cpu", quick=True)
    assert p.device == "cpu"
    assert p.name
    assert p.measured_at
    # 0.5 GFLOPS .. 10 PFLOPS
    assert 5e-4 < p.tflops < 1e4
    # 100 MB/s .. 100 TB/s
    assert 0.1 < p.gbps < 1e5
    # 10 ns .. 10 ms per launch
    assert 0.01 < p.launch_us < 1e4
    # executor-overhead constants: measured positive on CPU (or a
    # positive fallback — either way never zero)
    assert 0.001 < p.dispatch_us < 1e4
    assert 0.001 < p.leaf_eval_us < 1e4
    # compiled-call per-graph overhead: measured or fallback, > 0
    assert 0.001 < p.graph_overhead_us < 1e5
    # no autotune feedback until optimize_model_autotuned writes some
    assert p.measured_ns == {}
    # and the measured profile yields a working cost fn
    assert roofline_cost_for(p)(_mm_term()) > 0.0
    # the op-kernel table measured every class at positive ns
    for cls in (
        "matmul",
        "pointwise",
        "reduce",
        "concat",
        "stack",
        "index_select",
    ):
        entries = p.op_kernel_ns.get(cls)
        assert entries, f"missing kernel class {cls}"
        assert all(v > 0.0 for v in entries.values())
    # matmul keys are "MxKxN" signatures; the rest are numel strings
    assert all(len(k.split("x")) == 3 for k in p.op_kernel_ns["matmul"])
    assert all(k.isdigit() for k in p.op_kernel_ns["pointwise"])


def test_measure_op_kernels_cpu_direct():
    """The kernel sweep itself returns a positive-ns table covering
    all six classes even outside calibrate() — and it restores the
    torch thread count it pins for the serialized-latency probes."""
    prev = torch.get_num_threads()
    t = cal_mod._measure_op_kernels(
        torch.device("cpu"), torch.float32, quick=True
    )
    assert torch.get_num_threads() == prev
    assert set(t) >= {
        "matmul",
        "pointwise",
        "reduce",
        "concat",
        "stack",
        "index_select",
    }
    assert all(
        v > 0.0 for entries in t.values() for v in entries.values()
    )


def test_measure_op_kernels_probe_failure_dropped(monkeypatch):
    """A probe that cannot run drops only its own entries — here all
    of them — leaving a usable (empty) table rather than a crash."""
    monkeypatch.setattr(
        cal_mod,
        "_kernel_probe",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert (
        cal_mod._measure_op_kernels(
            torch.device("cpu"), torch.float32, quick=True
        )
        == {}
    )


def test_calibrate_op_kernel_sweep_failure_falls_back(monkeypatch):
    """If the whole kernel sweep dies, calibrate still returns a
    profile — with an empty table (pure-roofline pricing)."""
    monkeypatch.setattr(
        cal_mod,
        "_measure_op_kernels",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    p = calibrate(device="cpu", quick=True)
    assert p.op_kernel_ns == {}


@pytest.mark.requires_cuda
def test_calibrate_cuda_sane():
    p = calibrate(device="cuda", quick=True)
    assert p.device.startswith("cuda")
    # 50 GFLOPS .. 10 PFLOPS
    assert 0.05 < p.tflops < 1e4
    # 1 GB/s .. 100 TB/s
    assert 1.0 < p.gbps < 1e5
    assert 0.01 < p.launch_us < 1e4
    assert p.dispatch_us > 0 and p.leaf_eval_us > 0
    assert roofline_cost_for(p)(_mm_term()) > 0.0


def test_calibrate_persists(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path))
    p = calibrate(device="cpu", quick=True, name="ci-cpu", save=True)
    assert load_profile("ci-cpu") == p
