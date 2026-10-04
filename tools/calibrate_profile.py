"""Calibrate a ``TargetProfile`` on this hardware — the public path.

"The cost model is measured, never assumed" only holds on hardware
the model was measured on.  This script is the documented "run this
on your machine" entry point: it writes a loadable
``TargetProfile`` JSON combining

* the roofline/executor constants
  :func:`catopt_torch.calibrate.calibrate` micro-benchmarks — peak
  FLOPS, memory bandwidth, kernel-launch and IR-dispatch overhead,
  batched-scan leaf-eval overhead, compiled-graph call overhead,
  and the per-op-class measured kernel table (``op_kernel_ns``);
* the measured executor-family corrections
  ``tools/executor_cost_probe.py`` derives — per-family geomean of
  ``measured_routed_ns / modeled_ns``, written into the profile's
  ``corrections`` table under each measured case's
  :func:`~catopt_core.profile.shape_bucket`, so
  :func:`~catopt_orchestrator.optimize.delivered_cost_for` and
  ``_carrier_upgrade`` consume them through the shipped
  :func:`~catopt_core.profile.corrected_price_ns` contract.  This
  phase requires CUDA — the probe times CUDA kernels (fp64, the
  probe's methodology); ``--skip-executor-corrections`` produces a
  constants-only profile on any machine.

Usage::

    .venv/bin/python tools/calibrate_profile.py
    .venv/bin/python tools/calibrate_profile.py --out my.json --quick
    .venv/bin/python tools/calibrate_profile.py --cases SSM --warmup 10 --iters 50
    .venv/bin/python tools/calibrate_profile.py --save   # persist to profiles_dir()

then, in the model's process::

    from catopt_core.profile import TargetProfile
    from catopt_orchestrator import Optimizer, delivered_cost_for
    from catopt_torch.backend import TorchBackend

    profile = TargetProfile.load("calibrated_profile.json")
    opt = Optimizer(backend=TorchBackend())
    res = opt.search(
        model, x, cost_fn=delivered_cost_for(profile, x=x)
    )
    mod = opt.lower(res, x, verify=True).module

The simpler ``cost_fn=executor_cost_for(profile)`` uses the measured
constants only (calibrated per-op prices, uncorrected delivered
comparison); ``delivered_cost_for`` additionally bills each term
under the lowering it would be delivered by, corrected by the
profile's measured executor factors — the extraction model the
SSM routing inversion needs.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from catopt_core.profile import save_profile
from catopt_torch.calibrate import calibrate

sys.path.insert(0, str(Path(__file__).resolve().parent))

import executor_cost_probe as ecp
from catopt_orchestrator import Optimizer
from catopt_torch.backend import TorchBackend
from law_wallclock import _cases


def _patch_probe_timing(warmup: int | None, iters: int | None) -> None:
    """Rebind the probe's per-arm timing budget (warmup/iters).

    ``law_wallclock._synced_median``'s budget arguments are bound at
    definition time, so a caller-side override rebinds the probe
    module's imported name to a partially-applied copy — a
    measurement-harness knob, not a shipped-code patch.
    """
    import law_wallclock as lw

    kw: dict[str, int] = {}
    if warmup is not None:
        kw["warmup"] = warmup
    if iters is not None:
        kw["iters"] = iters
    if kw:
        ecp._synced_median = functools.partial(lw._synced_median, **kw)


def _measure_corrections(
    cases: list[Any], *, device: str | None
) -> tuple[dict, dict[str, float]]:
    """Run the probe's measured table over *cases*; return corrections.

    Searches each case, lowers every root-eclass alternative through
    its routed executor (and the generic one), times both on CUDA,
    and pools ``measured/modeled`` ratios per executor family — the
    exact machinery ``tools/executor_cost_probe.py`` runs, imported
    rather than duplicated.  Returns the assembled ``corrections``
    table plus the pooled factors for reporting.
    """
    opt = Optimizer(backend=TorchBackend())
    sink = opt.sink
    _ = sink.executors  # resolve carrier specs (registration effect)

    rows = []
    for case in cases:
        rows.append(ecp._probe_case(case, opt, sink))
        print(f"  measured {case.name}", flush=True)

    factors = ecp._correction_factors(rows)
    counts = {
        fam: len(rs) for fam, rs in ecp._correction_ratios(rows).items()
    }
    corr = ecp._corrections_table(cases, factors, counts, device=device)
    return corr, factors


def main(argv: list[str] | None = None) -> int:
    """Calibrate, measure, and write the profile JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="calibrated_profile.json",
        help="profile JSON path (default: calibrated_profile.json)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device to calibrate (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=("float16", "bfloat16", "float32", "float64"),
        help=(
            "dtype for the constants sweep (default: float64 — the "
            "probe cases run fp64)"
        ),
    )
    parser.add_argument("--name", default=None, help="profile name")
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "shrink calibrate()'s sweeps AND the executor-correction "
            "timing budget (warmup 5 / iters 20)"
        ),
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="substring filter on probe case names (e.g. 'SSM')",
    )
    parser.add_argument(
        "--skip-executor-corrections",
        action="store_true",
        help="constants only — skip the CUDA executor-family probe",
    )
    parser.add_argument(
        "--input-device",
        default="cpu",
        help=(
            "device of the inputs searches will run on — keys the "
            "corrections' shape_bucket (default: cpu, matching the "
            "probe; pass cuda when searching CUDA-side inputs)"
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="probe warmup calls per timed arm",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="probe timed iterations per arm",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="also persist under profiles_dir() (name-keyed)",
    )
    args = parser.parse_args(argv)

    # The calibrate() compiled-graph probe compiles a trivial
    # pointwise chain; cap Inductor's worker pool so the probe stays
    # cheap on modest boxes.  Inductor reads the variable when the
    # compile runs, not at import.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")

    dtype = getattr(torch, args.dtype)
    profile = calibrate(
        device=args.device,
        name=args.name,
        dtype=dtype,
        quick=args.quick,
        verbose=True,
    )

    corrections: dict = {}
    factors: dict[str, float] = {}
    if args.skip_executor_corrections:
        print(
            "executor corrections: skipped (--skip-executor-corrections)"
        )
    elif not torch.cuda.is_available():
        print("executor corrections: skipped (no CUDA)")
    else:
        cases = [
            c
            for c in _cases()
            if args.cases is None
            or args.cases.lower() in c.name.lower()
        ]
        if not cases:
            raise SystemExit(f"--cases {args.cases!r} matched nothing")
        warmup = (
            5 if args.quick and args.warmup is None else args.warmup
        )
        iters = 20 if args.quick and args.iters is None else args.iters
        _patch_probe_timing(warmup, iters)
        print(
            f"executor corrections: {len(cases)} cases "
            f"(warmup {warmup or 50}, iters {iters or 200})"
        )
        corrections, factors = _measure_corrections(
            cases, device=args.input_device or None
        )
        print(f"  factors (eager): {factors}")

    if corrections:
        profile = replace(
            profile,
            corrections=corrections,
            meta={
                **profile.meta,
                "executor_corrections": {
                    "source": "tools/executor_cost_probe.py",
                    "factors_eager": factors,
                    "note": (
                        "pooled same-run factors keyed by "
                        f"{args.input_device!r}-side shape buckets"
                    ),
                },
            },
        )

    out = Path(args.out)
    out.write_text(profile.to_json() + "\n")
    print(f"\nwrote {out}")
    if args.save:
        print(f"persisted {save_profile(profile)}")
    print(
        "\nuse:\n"
        "    from catopt_core.profile import TargetProfile\n"
        "    from catopt_orchestrator import (\n"
        "        Optimizer, delivered_cost_for)\n"
        "    from catopt_torch.backend import TorchBackend\n"
        f"    profile = TargetProfile.load({str(out)!r})\n"
        "    opt = Optimizer(backend=TorchBackend())\n"
        "    res = opt.search(model, x, "
        "cost_fn=delivered_cost_for(profile, x=x))\n"
        "    mod = opt.lower(res, x, verify=True).module"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
