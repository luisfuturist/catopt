"""Timing runner — ``torch.utils.benchmark`` medians + IQR.

``Runner`` is deliberately the *only* place that times anything: suites
describe ``Case``/``Variant`` data and never call a clock themselves.
"""

from __future__ import annotations

import gc

import torch
from torch.utils.benchmark import Timer

from bench.benchkit.model import Case, Cell


class Runner:
    """Times ``Case`` variants with ``Timer.blocked_autorange``.

    ``stmt`` callables are invoked through ``Timer(stmt="_fn()",
    globals={...})`` — torch's Timer only accepts string statements.
    On CUDA devices each timed call is wrapped so it ends in
    ``torch.cuda.synchronize()``, i.e. the measured time includes the
    GPU tail rather than just kernel-launch overhead.  The wrapper
    also runs ``gc.collect(0)`` per call (~2 us): self-referential
    closures in evaluators (or torch internals) can leave young
    reference cycles pinning GPU tensors between steps — on a 4 GB
    card cyclic garbage can outrun the allocator inside a single
    ``blocked_autorange`` window and OOM.  A gen-0 collect reclaims
    young cycles before they pile up; it is a no-op cost for variants
    that don't leak and uniform across variants, so comparisons stay
    fair.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        warmup: int = 5,
        min_run_time: float = 0.2,
        num_threads: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.warmup = warmup
        self.min_run_time = min_run_time
        self.num_threads = num_threads

    @staticmethod
    def _released() -> None:
        """Dead stmt swapped in after timing — see ``run_case``."""
        return None

    def _wrap(self, stmt):
        if self.device.type != "cuda":
            return stmt

        def synced() -> object:
            out = stmt()
            torch.cuda.synchronize()
            gc.collect(0)
            return out

        return synced

    def run_case(self, case: Case) -> Cell:
        """Time every variant of one case."""
        medians: dict[str, float] = {}
        iqrs: dict[str, float] = {}
        for v in case.variants:
            stmt = self._wrap(v.stmt)
            for _ in range(max(self.warmup, 0)):
                stmt()
            kwargs = {}
            if self.num_threads is not None:
                kwargs["num_threads"] = self.num_threads
            timer = Timer(stmt="_fn()", globals={"_fn": stmt}, **kwargs)
            meas = timer.blocked_autorange(
                min_run_time=self.min_run_time
            )
            medians[v.name] = meas.median
            iqrs[v.name] = meas.iqr
            # Timed stmts close over the cell's model + input tensors;
            # a Cell kept for the report would hold that GPU working
            # set for the rest of the sweep.  Timing is the only
            # consumer of ``stmt`` — drop the reference.
            v.stmt = self._released
        return Cell(
            case=case, medians=medians, iqr=iqrs, aux=dict(case.aux)
        )

    def run(self, cases: list[Case], progress=None) -> list[Cell]:
        """Run every case, reporting progress per cell.

        ``progress`` is an optional ``rich``-style callback receiving
        ``(case, cell)``; when ``None`` a plain line is printed.
        """
        cells = []
        for case in cases:
            cell = self.run_case(case)
            cells.append(cell)
            if progress is not None:
                progress(case, cell)
            else:
                coords = " ".join(
                    f"{k}={v}" for k, v in case.params.items()
                )
                times = "  ".join(
                    f"{n}={cell.medians[n] * 1e3:.3f}ms"
                    for n in cell.medians
                )
                print(f"  {case.name} [{coords}]  {times}", flush=True)
        return cells
