"""benchkit self-test — times a trivial case and emits every surface.

.venv/bin/python -m bench.benchkit
.venv/bin/python -m bench.benchkit --out /tmp/benchkit_smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from bench.benchkit import (
    Case,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
    collect_env,
)


def smoke(out: Path) -> None:
    """Time a trivial case end-to-end; write JSON + MD + HTML + plots."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(256, 256, generator=g)
    w = torch.randn(256, 256, generator=g)
    w2 = torch.randn(256, 256, generator=g)
    xb = torch.randn(512, 512, generator=g)
    wb = torch.randn(512, 512, generator=g)
    cases = [
        Case(
            name="matmul",
            params={"n": 256},
            variants=[
                Variant(
                    "one_mm",
                    lambda: x @ w,
                    flops=2 * 256**3,
                    note="single GEMM",
                ),
                Variant(
                    "two_mm",
                    lambda: (x @ w) @ w2,
                    flops=4 * 256**3,
                    note="two chained GEMMs",
                ),
            ],
            aux={"smoke": True, "nested": {"a": [1, 2], "b": 3.5}},
        ),
        Case(
            name="matmul_big",
            params={"n": 512},
            variants=[Variant("one_mm", lambda: xb @ wb)],
        ),
    ]
    runner = Runner(device="cpu", warmup=2, min_run_time=0.05)
    report = Report(
        suite="benchkit_smoke",
        title="benchkit smoke",
        summary="Self-test of the harness + every renderer.",
        findings=[
            Finding(
                claim="two GEMMs are slower than one",
                verdict=Verdict.PARITY,
                headline="harness sanity",
                metric="two_mm/one_mm",
                value=2.0,
            )
        ],
        cells=runner.run(cases),
        env=collect_env("cpu"),
        provenance={"x_param": "n", "speedup_vs": "two_mm"},
    )
    out.mkdir(parents=True, exist_ok=True)
    report.to_json(out / "smoke.json")
    report.to_markdown(out / "smoke.md", speedup_vs="two_mm")
    report.to_html(out / "smoke.html")
    pngs = report.to_plots(
        out / "plots", x_param="n", speedup_vs="two_mm"
    )
    print(f"wrote {out / 'smoke.json'}")
    print(f"wrote {out / 'smoke.md'}")
    print(f"wrote {out / 'smoke.html'}")
    for p in pngs:
        print(f"wrote {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="benchkit self-test")
    ap.add_argument("--out", default="/tmp/benchkit_smoke")
    smoke(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()
