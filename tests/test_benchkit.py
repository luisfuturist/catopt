"""Tests for the bench harness (``bench.benchkit``).

The harness is dev tooling, not shipped code, so it sits outside the
100% coverage floor — but its renderers and the ledger comparison are
pure functions with easy fixtures, and a silent break there would
corrupt every report.  These tests pin the contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from bench import registry  # noqa: E402
from bench.benchkit import (  # noqa: E402
    Case,
    Cell,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
)
from bench.benchkit.compare import compare_baseline  # noqa: E402


def _report() -> Report:
    case = Case(
        name="chain",
        params={"k": 4},
        variants=[
            Variant("eager", lambda: None),
            Variant("catopt", lambda: None),
        ],
        aux={"note": "weight-folded", "nested": {"a": 1, "b": [2, 3]}},
    )
    cell = Cell(
        case,
        {"eager": 1e-3, "catopt": 2e-4},
        {"eager": 1e-5, "catopt": 2e-6},
        case.aux,
    )
    return Report(
        suite="toy",
        title="toy suite",
        summary="does catopt beat eager?",
        findings=[
            Finding(
                claim="catopt folds the chain",
                verdict=Verdict.WIN,
                headline="5.0x vs eager",
                metric="catopt/eager",
                value=5.0,
            )
        ],
        cells=[cell],
        env={"device": "cpu", "torch": "x", "python": "3"},
    )


def test_report_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "toy.json"
    _report().to_json(path)
    payload = json.loads(path.read_text())
    assert payload["schema"] == 1
    assert payload["suite"] == "toy"
    assert payload["findings"][0]["verdict"] == "win"
    assert payload["cells"][0]["median_s"]["catopt"] == 2e-4


def test_markdown_is_readable(tmp_path: Path) -> None:
    path = tmp_path / "toy.md"
    _report().to_markdown(path, speedup_vs="eager")
    text = path.read_text()
    assert "## Findings" in text
    assert "**WIN**" in text
    assert "## Results" in text
    assert "## Speedup vs `eager`" in text
    # aux is pretty-printed inside a collapsible, never a truncated repr.
    assert "<details>" in text
    assert '"nested"' in text
    assert "{" not in text.splitlines()[0]


def test_html_dashboard(tmp_path: Path) -> None:
    path = tmp_path / "toy.html"
    _report().to_html(path)
    text = path.read_text()
    assert "toy suite" in text
    assert 'class="card win"' in text
    assert "<table>" in text


def test_plots_writes_interactive_html(tmp_path: Path) -> None:
    paths = _report().to_plots(
        tmp_path, x_param="k", speedup_vs="eager", static=False
    )
    assert paths
    assert all(p.endswith(".html") for p in paths)
    assert all(Path(p).exists() for p in paths)


def test_quarto_and_slidev(tmp_path: Path) -> None:
    report = _report()
    qmd = tmp_path / "toy.qmd"
    report.to_quarto(qmd, {"grid": "grid.svg"})
    assert qmd.read_text().startswith("---\n")
    assert "![grid](grid.svg)" in qmd.read_text()
    assets = report.to_slidev(tmp_path / "slidev", {"grid": "grid.svg"})
    assert set(assets) == {"json", "markdown"}
    payload = json.loads(Path(assets["json"]).read_text())
    assert payload["findings"][0]["verdict"] == "win"


def test_compare_detects_regression(tmp_path: Path) -> None:
    baseline = tmp_path / "base.json"
    _report().to_json(baseline)
    record = _report().to_dict()
    record["cells"][0]["median_s"]["eager"] = 1e-3 * 1.5  # +50%
    regs = compare_baseline(record, baseline, threshold=0.05)
    assert [r.variant for r in regs] == ["eager"]
    assert regs[0].ratio == pytest.approx(1.5)
    # within threshold -> no regression
    record["cells"][0]["median_s"]["eager"] = 1e-3 * 1.02
    assert compare_baseline(record, baseline, threshold=0.05) == []


def test_registry_lookup() -> None:
    spec = registry.get("reassoc_scale")
    assert spec.intent == "speedup"
    assert spec.module == "bench.suites.speedup.reassoc_scale"
    assert registry.get("bench.suites.speedup.reassoc_scale.py") is spec
    with pytest.raises(KeyError):
        registry.get("does_not_exist")
    assert registry.harnessed()
    # every suite's intent is a declared category with a stated question
    assert {s.intent for s in registry.SUITES} <= set(registry.INTENTS)
    assert all(s.question and s.expects for s in registry.SUITES)


def test_runner_times_a_call() -> None:
    import torch

    x = torch.randn(64, 64)
    case = Case(
        name="mm",
        params={"n": 64},
        variants=[Variant("mm", lambda: x @ x)],
    )
    cells = Runner(device="cpu", warmup=1, min_run_time=0.01).run(
        [case]
    )
    assert cells[0].medians["mm"] > 0
    assert cells[0].iqr["mm"] >= 0


def test_catalog_readme_in_sync() -> None:
    """The README's generated block must match the registry."""
    from bench.benchkit.render import catalog as _catalog

    readme = REPO / "bench/README.md"
    assert _catalog.readme_synced(readme), (
        "bench/README.md catalog drifted — run "
        "`python -m bench catalog --write`"
    )


def test_results_doc_in_sync() -> None:
    """docs/results.md must match the pinned baselines."""
    from bench.benchkit.render.results import render_results_doc

    doc = REPO / "docs/results.md"
    assert doc.exists(), (
        "docs/results.md missing — run `python -m bench results`"
    )
    assert doc.read_text() == render_results_doc(
        REPO / "bench/baselines"
    ), "docs/results.md drifted — run `python -m bench results`"


def test_pinned_baselines_state_findings() -> None:
    """A pinned baseline is not golden unless it states a verdict."""
    for path in sorted((REPO / "bench/baselines").glob("*.json")):
        payload = json.loads(path.read_text())
        assert payload["findings"], f"{path.name} states no finding"


def test_expected_verdict_parses_registry_prose() -> None:
    from bench.benchkit.compare import expected_verdict

    assert expected_verdict("WIN - 8.9-16.1x vs Inductor") == "WIN"
    assert expected_verdict("NEGATIVE on stories15M - 0 accepted") == (
        "NEGATIVE"
    )
    assert expected_verdict("Coverage map - matches/fires") is None
