"""Environment provenance for a report.

Every ``Report`` carries the machine + git state it was measured on, so
a number is never quoted without the context that produced it.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

# Repo root — bench/ sits directly under it.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_state() -> tuple[str, bool]:
    """``(short HEAD sha, dirty)`` for the repo, or ``("unknown", ...)``."""
    try:
        sha = subprocess.run(
            [
                "git",
                "-C",
                str(_REPO_ROOT),
                "rev-parse",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return sha, dirty
    except Exception:
        return "unknown", False


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "cpu"


def collect_env(device: str | torch.device | None = None) -> dict:
    """Provenance for a report: versions, device, git state, UTC time."""
    if device is None:
        dev = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        dev = torch.device(device)
    device_name = (
        torch.cuda.get_device_name(dev)
        if dev.type == "cuda" and torch.cuda.is_available()
        else _cpu_model()
    )
    sha, dirty = _git_state()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(
            timespec="seconds"
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda or "none",
        "device": str(dev),
        "device_name": device_name,
        "cpu_count": os.cpu_count(),
        "git_sha": sha,
        "git_dirty": dirty,
        "argv": list(sys.argv),
    }
