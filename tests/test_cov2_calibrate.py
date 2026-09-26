"""Coverage-gap tests for catopt.calibrate.

Beyond the behavioural tests (test_calibrate.py) this file covers:

* ``TargetProfile.from_json`` type errors, and the ``save``/``load``/
  ``cost_fn`` convenience methods;
* ``profiles_dir`` env precedence (``$CATOPT_PROFILE_DIR`` >
  ``$XDG_CACHE_HOME`` > ``~/.cache``) and ``_safe_name`` edges;
* ``load_profile``'s directory-scan fallback — including the corrupt-
  file skip and the non-directory early FileNotFoundError — and the
  same skip path inside ``list_profiles``;
* ``_sync`` dispatching to ``torch.cuda.synchronize`` for CUDA devices
  and being a no-op on CPU;
* the ``dt <= 0`` timing guards inside the FLOP/BW probes (via a
  frozen ``perf_counter``);
* ``_default_name``'s CUDA branch (device-name query + fallback);
* ``calibrate(device="cuda")`` — sweep sizing, tf32 save/restore —
  with the three measurement probes monkeypatched, so the CUDA-only
  code runs on a CPU-only box;
* ``_verbose_ctx`` level restore (both directions);
* real CPU ``calibrate(quick=True)`` plausibility: finite, positive
  constants, parseable ``measured_at``, populated ``meta``.
"""

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import pytest
import torch

import catopt.calibrate as C
from catopt.calibrate import (
    PROFILE_DIR_ENV,
    TargetProfile,
    calibrate,
    list_profiles,
    load_profile,
    profiles_dir,
    save_profile,
)
from catopt.cost import roofline_cost_for
from catopt.ir import Op, Param, TensorType, Var

TOY = TargetProfile(
    "toy", tflops=1.5, gbps=20.0, launch_us=4.0, device="cpu",
    measured_at="2024-01-01T00:00:00+00:00",
)


def _term():
    x = Var("x", TensorType((64, 64)))
    w = Param("W", TensorType((64, 64)))
    return Op.make("matmul", x, w)


# ---------------------------------------------------------------------------
# TargetProfile serde / convenience methods
# ---------------------------------------------------------------------------


def test_from_json_rejects_non_dicts():
    with pytest.raises(TypeError):
        TargetProfile.from_json(42)
    # valid JSON that isn't an object is still a TypeError
    with pytest.raises(TypeError):
        TargetProfile.from_json("[1, 2, 3]")


def test_profile_method_helpers(tmp_path):
    path = TOY.save(tmp_path)
    assert path.exists() and path.parent == tmp_path
    assert TargetProfile.load(TOY.name, tmp_path) == TOY
    assert TargetProfile.load(path) == TOY
    # .cost_fn() is the roofline model built from these constants
    t = _term()
    assert TOY.cost_fn()(t) == pytest.approx(
        roofline_cost_for(TOY)(t)
    )


# ---------------------------------------------------------------------------
# profiles_dir / _safe_name
# ---------------------------------------------------------------------------


def test_profiles_dir_env_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(PROFILE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert profiles_dir() == tmp_path / "xdg" / "catopt" / "profiles"
    # $CATOPT_PROFILE_DIR beats $XDG_CACHE_HOME
    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path / "explicit"))
    assert profiles_dir() == tmp_path / "explicit"
    # neither set → ~/.cache fallback
    monkeypatch.delenv(PROFILE_DIR_ENV)
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert profiles_dir() == Path.home() / ".cache" / "catopt" / "profiles"


def test_safe_name_edges():
    assert C._safe_name("RTX 2050!") == "RTX_2050"
    assert C._safe_name("a b") == "a_b"
    # everything stripped → the generic fallback name
    assert C._safe_name("_") == "profile"
    assert C._safe_name("***") == "profile"


# ---------------------------------------------------------------------------
# load_profile scan fallback / list_profiles corrupt-skip
# ---------------------------------------------------------------------------


def test_load_profile_name_scan_and_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path))
    # stored under a filename that does NOT match the safe name:
    # only the directory scan finds it (stored name field match)
    prof = TargetProfile(
        "real-name", 1.0, 2.0, 3.0, "cpu", "t"
    )
    (tmp_path / "different-file.json").write_text(prof.to_json())
    # plus a corrupt file that both loaders must skip
    (tmp_path / "broken.json").write_text("{not json")
    assert load_profile("real-name") == prof
    with pytest.raises(FileNotFoundError):
        load_profile("ghost")


def test_load_profile_missing_dir(tmp_path):
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        load_profile("ghost", dir=missing)
    assert list_profiles(dir=missing) == []


def test_list_profiles_skips_corrupt(tmp_path):
    (tmp_path / "good.json").write_text(TOY.to_json())
    (tmp_path / "bad.json").write_text(json.dumps([1, 2]))  # non-dict
    (tmp_path / "worse.json").write_text("not json at all")
    assert list_profiles(dir=tmp_path) == ["toy"]


def test_save_profile_explicit_dir(tmp_path):
    path = save_profile(TOY, dir=tmp_path / "sub")
    assert path.name == "toy.json"
    assert load_profile(path) == TOY


# ---------------------------------------------------------------------------
# _sync / _default_name
# ---------------------------------------------------------------------------


def test_sync_dispatches_only_on_cuda(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda dev=None: calls.append(dev)
    )
    dev = torch.device("cuda:0")
    C._sync(dev)
    assert calls == [dev]
    # CPU is a no-op — synchronize is never even invoked
    C._sync(torch.device("cpu"))
    assert calls == [dev]


def test_default_name_cuda_branches(monkeypatch):
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda dev=None: "Fake GPU"
    )
    assert C._default_name(torch.device("cuda:0")) == "Fake GPU"

    def _boom(dev=None):
        raise RuntimeError("driver wedged")

    monkeypatch.setattr(torch.cuda, "get_device_name", _boom)
    assert C._default_name(torch.device("cuda:1")) == "cuda:1"
    assert C._default_name(torch.device("cuda")) == "cuda:0"


def test_default_name_cpu_reports_threads():
    name = C._default_name(torch.device("cpu"))
    assert name.endswith(f"cpu, {torch.get_num_threads()}t)")
    assert name  # processor/machine string or plain 'cpu'


# ---------------------------------------------------------------------------
# measurement probes — dt<=0 guards
# ---------------------------------------------------------------------------


def test_measure_guards_on_zero_dt(monkeypatch):
    """A frozen clock makes dt == 0: every ``if dt > 0`` guard takes
    the False branch and the probes honestly return 0.0."""
    monkeypatch.setattr(time, "perf_counter", lambda: 0.0)
    dev = torch.device("cpu")
    assert C._measure_flops(dev, torch.float32, (8,), 2, 1) == 0.0
    assert C._measure_bandwidth(dev, torch.float32, (1,), 2, 1) == 0.0


def test_measure_launch_returns_per_op_time():
    # unpatched clock: a tiny real measurement is finite and positive
    dt = C._measure_launch(torch.device("cpu"), torch.float32, 50, 10)
    assert math.isfinite(dt) and dt > 0


# ---------------------------------------------------------------------------
# _verbose_ctx
# ---------------------------------------------------------------------------


def test_verbose_ctx_restores_level(caplog):
    log = logging.getLogger("catopt.calibrate")
    prev = log.level
    # verbose=False touches nothing
    with C._verbose_ctx(log, False):
        assert log.level == prev
    assert log.level == prev
    # verbose=True drops to DEBUG for the block, then restores
    with C._verbose_ctx(log, True):
        assert log.level == logging.DEBUG
    assert log.level == prev
    # and DEBUG records actually flow while the context is open
    with caplog.at_level(logging.DEBUG, logger="catopt.calibrate"):
        with C._verbose_ctx(log, True):
            log.debug("surfaced only when verbose")
        # restore honours whatever level was in place (here caplog's)
        assert log.level == logging.DEBUG
    assert "surfaced only when verbose" in caplog.text
    assert log.level == prev  # still back to the session default


# ---------------------------------------------------------------------------
# calibrate() — CPU fallback and CUDA branch
# ---------------------------------------------------------------------------


def test_calibrate_cpu_quick_plausibility():
    if torch.cuda.is_available():
        pytest.skip("CUDA box — CPU fallback assertions don't apply")
    p = calibrate(quick=True)  # default device falls back to CPU
    assert p.device == "cpu" and p.name
    # measured_at is ISO-8601 parseable
    datetime.fromisoformat(p.measured_at)
    # all three constants are finite and positive on any real CPU
    assert math.isfinite(p.tflops) and p.tflops > 0
    assert math.isfinite(p.gbps) and p.gbps > 0
    assert math.isfinite(p.launch_us) and p.launch_us > 0
    assert p.meta["dtype"] == "float32"
    assert p.meta["torch"] == torch.__version__
    assert "platform" in p.meta
    # and the profile drives a working cost model
    assert roofline_cost_for(p)(_term()) > 0


def test_calibrate_cpu_named_and_saved(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_DIR_ENV, str(tmp_path))
    p = calibrate(device="cpu", quick=True, name="ci-cpu2", save=True)
    assert p.name == "ci-cpu2"
    assert (tmp_path / "ci-cpu2.json").exists()
    assert load_profile("ci-cpu2") == p


def test_calibrate_cuda_branch_monkeypatched(monkeypatch, caplog):
    """Run the CUDA sweep-sizing + tf32 save/restore code on a CPU box
    by stubbing the three measurement probes (and the device name)."""
    monkeypatch.setattr(C, "_measure_flops", lambda *a: 2.0e12)
    monkeypatch.setattr(C, "_measure_bandwidth", lambda *a: 9.0e11)
    monkeypatch.setattr(C, "_measure_launch", lambda *a: 4.0e-6)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda dev=None: "Fake GPU"
    )
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with caplog.at_level(logging.DEBUG, logger="catopt.calibrate"):
            p = calibrate(device="cuda", quick=True, verbose=True)
        assert p.device == "cuda"
        assert p.name == "Fake GPU"
        assert p.tflops == pytest.approx(2.0)
        assert p.gbps == pytest.approx(900.0)
        assert p.launch_us == pytest.approx(4.0)
        # tf32 was disabled during measurement and restored after
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert "calibrated" in caplog.text
        assert "calibration raw" in caplog.text
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def test_calibrate_cuda_tf32_access_failure(monkeypatch):
    """When the tf32 flag can't even be read, calibrate proceeds with
    ``prev_tf32=None`` and skips the restore in ``finally``."""
    monkeypatch.setattr(C, "_measure_flops", lambda *a: 1.0e12)
    monkeypatch.setattr(C, "_measure_bandwidth", lambda *a: 1.0e11)
    monkeypatch.setattr(C, "_measure_launch", lambda *a: 1.0e-6)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda d=None: "G")
    monkeypatch.setattr(torch.backends.cuda, "matmul", None)
    p = calibrate(device="cuda", quick=True)
    assert p.device == "cuda" and p.name == "G"


@pytest.mark.requires_cuda
def test_calibrate_cuda_real_device():
    """The real CUDA path — only where a device exists."""
    p = calibrate(device="cuda", quick=True)
    assert p.device.startswith("cuda")
    assert math.isfinite(p.tflops) and p.tflops > 0
    assert math.isfinite(p.gbps) and p.gbps > 0
    assert math.isfinite(p.launch_us) and p.launch_us > 0
