"""Statistics + tabular helpers shared by the renderers."""

from __future__ import annotations

import math
import re
from typing import Any

from bench.benchkit.model import Cell


def fmt_ms(seconds: float) -> str:
    """Compact millisecond formatting for tables."""
    return f"{seconds * 1e3:.4g}"


def fmt_pm(median: float, iqr: float) -> str:
    """``median ± IQR`` in milliseconds."""
    return f"{fmt_ms(median)} ± {fmt_ms(iqr)}"


def slug(text: str) -> str:
    """Filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def variant_names(cells: list[Cell]) -> list[str]:
    """Union of variant names, ordered by first appearance."""
    names: list[str] = []
    for cell in cells:
        for name in cell.medians:
            if name not in names:
                names.append(name)
    return names


def param_keys(cells: list[Cell]) -> list[str]:
    """Union of sweep-coordinate names, ordered by first appearance."""
    keys: list[str] = []
    for cell in cells:
        for k in cell.case.params:
            if k not in keys:
                keys.append(k)
    return keys


def speedup(cell: Cell, name: str, baseline: str) -> float | None:
    """``baseline / name`` medians (>1 = ``name`` faster), or ``None``."""
    base = cell.medians.get(baseline)
    val = cell.medians.get(name)
    if not base or not val:
        return None
    return base / val


def to_dataframe(cells: list[Cell]):
    """Flatten cells to a ``polars.DataFrame`` (import guarded)."""
    import polars as pl

    rows = []
    for cell in cells:
        row: dict[str, Any] = {
            "case": cell.case.name,
            **cell.case.params,
        }
        for n, s in cell.medians.items():
            row[f"{n}_median_ms"] = s * 1e3
        for n, s in cell.iqr.items():
            row[f"{n}_iqr_ms"] = s * 1e3
        rows.append(row)
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _jsonable(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def jsonable(value: Any) -> Any:
    """Recursively make a value JSON-serializable."""
    return _jsonable(value)
