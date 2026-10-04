"""Train and ship the bundled contraction-policy weights.

``project/retros/contraction-train-scale.md`` settled the regime: a
**curriculum** — train on all scales (8, 12, 16, 20, 24) — preserves the
small-board and starvation behaviour of the base policy while nearly
closing the n = 40 gap to the best cheap player.  This tool runs that
regime once (REINFORCE with the greedy-completion critic, ``--iterations
1800``) on the einsum-valid ``random_bond_network`` family and writes
the artifact the shipped loader reads:

    packages/catopt-torch/src/catopt_torch/artifacts/
        contraction_policy_curriculum.pt

so ``catopt_torch.load_contraction_policy()`` has real weights instead
of every experiment retraining from scratch.  The player machinery
itself lives in the package (``catopt_torch.contraction_policy``); the
trainer stays in tools (``contraction_policy.train_rl``) because
training is an experiment concern — only the weights ship.

The family patch mirrors ``contraction_einsum._on_family`` but is
re-stated here so the tool runs without the opt-in ``einsum`` group
(opt_einsum is only needed to *compare*, not to *train*).

Usage::

    .venv/bin/python tools/train_contraction_artifact.py [--device cuda]
    .venv/bin/python tools/train_contraction_artifact.py --iterations 5 \
        --out /tmp/smoke.pt          # smoke run, throwaway artifact
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import contraction_policy as cp
import contraction_scale as cs
import torch
from catopt_torch.contraction_policy import (
    greedy,
    load_contraction_policy,
    random_bond_network,
    rollout_orders,
    save_contraction_policy,
)

__all__ = ["main"]

#: The shipped-artifact destination inside the package.
_DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "catopt-torch"
    / "src"
    / "catopt_torch"
    / "artifacts"
    / "contraction_policy_curriculum.pt"
)

#: The curriculum regime the retro recommends: every scale at once.
_CURRICULUM = (8, 12, 16, 20, 24)


@contextlib.contextmanager
def _on_family(family: Any) -> Any:
    """Train on the einsum-valid bond family, not the hypergraph one."""
    old = cs.random_network
    cs.random_network = family
    try:
        yield
    finally:
        cs.random_network = old


def _git_sha() -> str | None:
    """Best-effort checkout SHA for the artifact's provenance meta."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
            check=False,
        )
    except OSError:
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _sanity_eval(model: Any, device: str, boards: int) -> None:
    """Print the artifact's ratio-to-greedy on held-out bond boards.

    A cheap honest check that the saved weights reload and still play:
    the deterministic rollout's cost over our cheapest-pair greedy's,
    mean per scale — the same metric the retro's section 3 reports.
    """
    print()
    print("== sanity: reload + play held-out bond boards ==")
    print(f"  {'n':>3} {'policy/greedy':>14} {'worst':>7}")
    for n in (12, 16, 20):
        ratios = []
        for k in range(boards):
            tensors, sizes = random_bond_network(n, 777 + 1000 * n + k)
            ref = greedy(tensors, sizes)
            got = rollout_orders(
                model,
                tensors,
                sizes,
                ref,
                1,
                device,
                1.0,
                greedy=True,
            )[1][0]
            ratios.append(got / ref)
        print(
            f"  {n:>3} {statistics.fmean(ratios):>14.3f} "
            f"{max(ratios):>7.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Train the curriculum policy, save it, sanity-check the reload."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=1800)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument(
        "--scales",
        default=",".join(str(n) for n in _CURRICULUM),
        help="comma-separated curriculum n",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--eval-boards", type=int, default=8)
    ap.add_argument(
        "--out",
        default=str(_DEFAULT_OUT),
        help="artifact destination (default: the bundled package path)",
    )
    args = ap.parse_args(argv)

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    ns = tuple(int(x) for x in args.scales.split(",") if x.strip())
    print(f"device: {dev}  (torch {torch.__version__})")
    print(f"curriculum: {ns}  iterations: {args.iterations}")

    t0 = time.perf_counter()
    with _on_family(random_bond_network):
        model = cp.train_rl(
            ns=ns,
            iterations=args.iterations,
            batch=args.batch,
            hidden=args.hidden,
            device=dev,
            seed=args.seed,
            log_every=max(args.iterations // 4, 1),
        )
    print(f"trained in {time.perf_counter() - t0:.1f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_contraction_policy(
        model,
        out,
        meta={
            "trainer": "rl",
            "train_scales": list(ns),
            "iterations": args.iterations,
            "batch": args.batch,
            "seed": args.seed,
            "family": "random_bond_network",
            "git_sha": _git_sha(),
            # ``torch.__version__`` is a ``TorchVersion`` (a str
            # subclass), which ``weights_only`` loads reject — store a
            # plain ``str``.
            "torch": str(torch.__version__),
            "created": time.strftime("%Y-%m-%d"),
        },
    )
    print(f"saved {out} ({out.stat().st_size / 1024:.1f} KiB)")

    # The artifact is only useful if it round-trips through the shipped
    # loader — check that path, not just the in-memory model.
    policy = load_contraction_policy(out, device=dev)
    _sanity_eval(policy.model, dev, args.eval_boards)
    print(f"meta: {policy.meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
