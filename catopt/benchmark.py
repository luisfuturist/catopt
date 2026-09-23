"""Benchmarking utilities for the killer experiment (Phase 4).

Compares:
  * Vanilla TorchInductor (``torch.compile`` on the original model).
  * Categorical optimizer + TorchInductor (our optimized IR lowered to torch,
    then compiled with the same torch.compile pipeline).

The comparison table shows whether the categorical optimizer's transformation
survives (or beats) TorchInductor's own optimization passes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    std_ms: float
    n_runs: int
    extra: dict[str, Any] = field(default_factory=dict)


def _bench_once(model: torch.nn.Module, x: torch.Tensor,
                warmup: int = 5, repeats: int = 20) -> BenchResult:
    """Run a model *repeats* times and return timing statistics."""
        # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        times.append((time.perf_counter() - t0) * 1000)  # ms

    times.sort()
    # Trim outliers (drop 10% from each end)
    trim = max(1, len(times) // 10)
    trimmed = times[trim:-trim] if len(times) > 2 * trim else times
    n = len(trimmed)
    mean = sum(trimmed) / n
    var = sum((t - mean) ** 2 for t in trimmed) / n
    return BenchResult(
        name="",
        mean_ms=mean,
        std_ms=var ** 0.5,
        n_runs=n,
    )


def benchmark_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    name: str = "",
    use_compile: bool = True,
    mode: str = "default",
    **compile_kwargs,
) -> BenchResult:
    """Benchmark a model, optionally with TorchInductor compilation."""
    model = model.eval()

    if use_compile:
        compiled = torch.compile(model, mode=mode, **compile_kwargs)
        result = _bench_once(compiled, x)
        result.name = f"{name} (inductor)"
    else:
        result = _bench_once(model, x)
        result.name = f"{name} (eager)"

    return result


def benchmark_comparison(
    original_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    x: torch.Tensor,
    name: str = "",
) -> tuple[BenchResult, BenchResult]:
    """Benchmark both original and optimized models with TorchInductor.

    This is the Phase 4 comparison: does the categorical optimizer's
    transformation still help *after* TorchInductor has run its own
    optimization pipeline?
    """
    # Verify semantic equivalence
    with torch.no_grad():
        orig_out = original_model(x.clone())
        opt_out = optimized_model(x.clone())
        max_diff = (orig_out - opt_out).abs().max().item()
        rel_diff = max_diff / (orig_out.abs().max().item() + 1e-8)
        print(f"  [verify] max_rel_diff = {rel_diff:.2e}")

    # Warm up compile cache
    print(f"  [compile] warming up original model with torch.compile...")
    orig_result = benchmark_model(original_model, x, f"{name} / TorchInductor",
                                  use_compile=True)
    print(f"  [compile] warming up optimized model with torch.compile...")
    opt_result = benchmark_model(optimized_model, x, f"{name} / Categorical+Inductor",
                                 use_compile=True)

    # Also benchmark eager (no compile) for reference
    orig_eager = benchmark_model(original_model, x, f"{name} / TorchInductor",
                                 use_compile=False)
    opt_eager = benchmark_model(optimized_model, x, f"{name} / Categorical+Inductor",
                                use_compile=False)

    return orig_result, opt_result


def print_comparison_table(
    cases: list[tuple[str, str, BenchResult, BenchResult]],
    title: str = "Benchmark Results",
) -> None:
    """Print a comparison table.

    Each entry in *cases* is:
    (program_name, semantically_equivalent, inductor_result, catopt_result)
    """
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Program':<20} {'Equiv?':<10} {'Inductor (ms)':>16} {'CatOpt (ms)':>16} {'Speedup':>10}")
    print(f"{'-'*20} {'-'*10} {'-'*16} {'-'*16} {'-'*10}")

    for prog_name, equiv, ind, cat in cases:
        speedup = ind.mean_ms / (cat.mean_ms + 1e-9)
        speedup_str = f"{speedup:.2f}x" if speedup > 1.0 else f"{speedup:.2f}x"
        flag = "  ↑" if speedup > 1.001 else "  —"
        print(f"{prog_name:<20} {equiv:<10} {ind.mean_ms:>10.3f}±{ind.std_ms:<4.2f}  "
              f"{cat.mean_ms:>10.3f}±{cat.std_ms:<4.2f} {speedup_str:>8} {flag}")

    print(f"{'='*80}")
