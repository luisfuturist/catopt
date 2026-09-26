"""Tests for catopt.calibrate (TargetProfile + calibrate) and the
profile-parameterised cost fns in catopt.cost."""

import json

import pytest

from catopt.calibrate import (
    PROFILE_DIR_ENV,
    TargetProfile,
    calibrate,
    list_profiles,
    load_profile,
    profiles_dir,
    save_profile,
)
from catopt.cost import (
    dag_cost,
    depth_cost_for,
    roofline_cost,
    roofline_cost_for,
)
from catopt.ir import Op, Param, TensorType, Var

# The constants hardcoded in catopt.cost — the dev RTX 2050 profile.
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


def test_profile_json_ignores_unknown_fields():
    data = json.loads(RTX2050.to_json())
    data["future_field"] = 42
    assert TargetProfile.from_json(data) == RTX2050


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
    # and the measured profile yields a working cost fn
    assert roofline_cost_for(p)(_mm_term()) > 0.0


@pytest.mark.requires_cuda
def test_calibrate_cuda_sane():
    p = calibrate(device="cuda", quick=True)
    assert p.device.startswith("cuda")
    # 50 GFLOPS .. 10 PFLOPS
    assert 0.05 < p.tflops < 1e4
    # 1 GB/s .. 100 TB/s
    assert 1.0 < p.gbps < 1e5
    assert 0.01 < p.launch_us < 1e4
    assert roofline_cost_for(p)(_mm_term()) > 0.0


def test_calibrate_persists(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path))
    p = calibrate(device="cpu", quick=True, name="ci-cpu", save=True)
    assert load_profile("ci-cpu") == p
