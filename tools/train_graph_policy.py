"""Wire a learned search policy into catopt's real pipeline, and measure.

Plan 0016 stage 7's last mile.  The ``Policy`` seam is already consumed
by ``Optimizer.optimize`` / ``EGraph.run``, but the engine hands the
policy a ``GameState`` with no features, so the base ``LearnedPolicy``
cannot run there.  :class:`catopt_torch.graph_policy.GraphFeaturePolicy`
closes that gap.  This tool:

1. builds supervised trajectories from **real models** (the
   ``catopt_torch.models`` zoo) via
   :func:`catopt_torch.model_trajectories.model_rule_samples`;
2. trains a ``RuleValueNet`` on them (supervised — the retro's dense
   signal, see ``project/retros/stage7-multifamily-results.md``);
3. runs the **real pipeline** (``catopt_orchestrator.optimize.search``)
   with the shipped ordering, the learned policy and a random policy on
   held-out models, reporting enodes / wall time / extracted cost;
4. asserts the safety invariants: the certificate still replays, and a
   random policy yields the identical equivalence class.

Usage::

    python tools/train_graph_policy.py [--device auto|cpu|cuda]
"""

from __future__ import annotations

import argparse
import hashlib
import time

import torch
from catopt_core.cost import flops_cost
from catopt_core.egraph import EGraph
from catopt_core.egraph.terms import verify_certificate
from catopt_core.laws import all_rules
from catopt_core.policies import RandomPolicy
from catopt_orchestrator.optimize import search
from catopt_torch.adapters import TorchSource
from catopt_torch.graph_policy import GraphFeaturePolicy
from catopt_torch.learned_policy import train_rule_value
from catopt_torch.model_trajectories import model_rule_samples
from catopt_torch.models import (
    AttentionBlock,
    DeepParallel,
    GQAAttention,
    MatrixChain,
    NormLinear,
    ParallelLinear,
    ResidualMLP,
    RMSNorm,
    SwiGLU,
    TransformerBlock,
)

#: ``(name, model, example input)`` training cases (small, bounded).
TRAIN_CASES = (
    ("swiglu", SwiGLU(32, 4), torch.randn(4, 32)),
    ("rmsnorm", RMSNorm(64), torch.randn(4, 64)),
    ("attention", AttentionBlock(64, 4), torch.randn(2, 16, 64)),
    ("transformer", TransformerBlock(64, 4, 4), torch.randn(2, 16, 64)),
    ("residual_mlp", ResidualMLP(64, 4), torch.randn(4, 64)),
    ("parallel_linear", ParallelLinear(64, 3), torch.randn(4, 64)),
    ("deep_parallel", DeepParallel(64, 32, 16), torch.randn(4, 64)),
    ("norm_linear", NormLinear(64, 32), torch.randn(4, 64)),
    ("matrix_chain", MatrixChain(128, 64, 32, 8), torch.randn(4, 128)),
    ("gqa", GQAAttention(64, 8, 2), torch.randn(2, 16, 64)),
)

#: Held-out cases — unseen widths / head counts / depth.
HELD_CASES = (
    ("swiglu*", SwiGLU(48, 6), torch.randn(4, 48)),
    ("attention*", AttentionBlock(96, 6), torch.randn(2, 24, 96)),
    (
        "transformer*",
        TransformerBlock(96, 6, 6),
        torch.randn(2, 24, 96),
    ),
    (
        "deep_transformer*",
        torch.nn.Sequential(
            *[TransformerBlock(64, 4, 4) for _ in range(2)]
        ),
        torch.randn(2, 16, 64),
    ),
    ("deep_parallel*", DeepParallel(96, 48, 24), torch.randn(4, 96)),
    ("matrix_chain*", MatrixChain(96, 48, 24, 6), torch.randn(4, 96)),
)


def _train(device: str, epochs: int, hidden: int, seed: int):
    """Build real-model samples and train a rule-value net."""
    rules = all_rules()
    samples = []
    for name, model, x in TRAIN_CASES:
        s = model_rule_samples(model, x, rules)
        samples.extend(s)
        n_pos = sum(1 for r in s if r.delta_cost > 0)
        print(
            f"  train {name:16s} rules={len(s):3d} improving={n_pos:3d}"
        )
    print(f"  total samples: {len(samples)}")
    model = train_rule_value(
        samples, epochs=epochs, hidden=hidden, device=device, seed=seed
    )
    return model, {r.name: r for r in rules}, rules


def _partition(eg):
    """Return an eid-independent canonical form of the e-graph.

    The *equivalence class* is the partition of terms — which terms sit
    in which class — not the enode count and not the eids: reordering
    renumbers eids and can leave a redundant duplicate spelling in a
    class, so ``n_enodes`` and raw ``op_repr`` are not semantic
    invariants.  A Weisfeiler-Lehman-style refinement gives every class
    a label from its enodes' ops/attrs and its children's labels,
    iterated to a fixpoint.  Each round *hashes* the signature, so the
    labels stay fixed-width (no exponential string blowup) and two
    e-graphs compare equal iff they are the same partition up to eid
    renaming and duplicate enodes.
    """
    ids = sorted({eg.find(e) for e in eg._classes})
    label = {e: "" for e in ids}
    for _ in range(len(ids) + 1):
        new = {}
        for e in ids:
            cls = eg.get_class(e)
            sig = "|".join(
                sorted(
                    f"{n.op}({','.join(label[eg.find(c)] for c in n.children)})"
                    f"[{','.join(f'{k}={v}' for k, v in n.attrs)}]"
                    for n in cls.nodes
                )
            )
            new[e] = hashlib.blake2b(
                sig.encode(), digest_size=16
            ).hexdigest()
        if new == label:
            break
        label = new
    return sorted(label.values())


def _run(
    model, x, policy, max_iterations, stop="fixed_point", patience=3
):
    """Run the real pipeline once; return (stats, cost, ms, result)."""
    src = TorchSource()
    t0 = time.perf_counter()
    res = search(
        model,
        x,
        source=src,
        policy=policy,
        max_iterations=max_iterations,
        stop=stop,
        patience=patience,
    )
    dt = time.perf_counter() - t0
    cost = (
        flops_cost(res.term) if res.term is not None else float("inf")
    )
    return res.stats, cost, dt * 1000.0, res


def _run_saturation(model, x, policy, rules, max_iterations):
    """Run saturation alone (``EGraph.run``); return (stats, cost, ms)."""
    ir, _ = TorchSource().to_ir(model, x)
    eg = EGraph()
    root = eg.add_term(ir.root)
    t0 = time.perf_counter()
    stats = eg.run(
        rules, root, max_iterations=max_iterations, policy=policy
    )
    dt = time.perf_counter() - t0
    term = eg.extract_best(root, flops_cost)
    cost = flops_cost(term) if term is not None else float("inf")
    return stats, cost, dt * 1000.0


def _row(label, stats, cost, ms):
    return (
        f"    {label:8s} enodes={stats['n_enodes']:6d} "
        f"classes={stats['n_classes']:6d} "
        f"iters={stats['iterations']:3d} "
        f"stop={stats['stop']:12s} "
        f"cost={cost:14.1f} ms={ms:7.1f}"
    )


def main() -> None:
    """Train on real models, then race the players on held-out ones."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--bounded", type=int, default=1)
    args = ap.parse_args()

    dev = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    print(f"device: {dev}  (torch {torch.__version__})")

    print("\n== training on real models ==")
    model, by_name, rules = _train(
        dev, args.epochs, args.hidden, args.seed
    )
    learned = GraphFeaturePolicy(model, by_name, device=dev)

    for iters, tag in (
        (args.iterations, "fixed point"),
        (args.bounded, f"bounded (max_iterations={args.bounded})"),
    ):
        print(f"\n== held-out race — {tag} ==")
        for name, m, x in HELD_CASES:
            print(f"  {name}:")
            for label, pol in (
                ("default", None),
                ("learned", learned),
                ("random", RandomPolicy(args.seed)),
            ):
                stats, cost, ms, _ = _run(m, x, pol, iters)
                print(_row(label, stats, cost, ms))

    print("\n== held-out saturation only (EGraph.run) — fixed point ==")
    for name, m, x in HELD_CASES:
        print(f"  {name}:")
        for label, pol in (
            ("default", None),
            ("learned", learned),
            ("random", RandomPolicy(args.seed)),
        ):
            stats, cost, ms = _run_saturation(
                m, x, pol, rules, args.iterations
            )
            print(_row(label, stats, cost, ms))

    print("\n== held-out race — lazy (stop='improving') ==")
    for name, m, x in HELD_CASES:
        print(f"  {name}:")
        for label, pol in (
            ("default", None),
            ("learned", learned),
            ("random", RandomPolicy(args.seed)),
        ):
            stats, cost, ms, _ = _run(
                m, x, pol, args.iterations, stop="improving", patience=2
            )
            print(_row(label, stats, cost, ms))

    # -- safety: the certificate still replays ------------------------
    print("\n== safety ==")
    n_ok = 0
    for name, m, x in HELD_CASES:
        src = TorchSource()
        ir, _ = src.to_ir(m, x)
        eg = EGraph()
        root = eg.add_term(ir.root)
        eg.run(
            rules, root, max_iterations=args.iterations, policy=learned
        )
        best = eg.extract_best(root, flops_cost)
        cert = eg.certificate(ir.root, best, cost_fn=flops_cost)
        out = verify_certificate(ir.root, cert)
        assert flops_cost(out) == flops_cost(best), name
        n_ok += 1
    print(
        f"  certificate replays for {n_ok}/{len(HELD_CASES)} models: OK"
    )

    # -- safety: a random policy is the same equivalence class --------
    # Checked twice: through the full pipeline, and at the saturation
    # level (where the policy actually acts) for default / learned /
    # random alike.
    same = 0
    for name, m, x in HELD_CASES:
        d_res = _run(m, x, None, args.iterations)[3]
        r_res = _run(m, x, RandomPolicy(args.seed), args.iterations)[3]
        assert d_res.stats["n_classes"] == r_res.stats["n_classes"], (
            name
        )
        assert _partition(d_res.eg) == _partition(r_res.eg), name
        same += 1
    print(
        f"  random policy == default partition (equivalence "
        f"class): OK ({same})"
    )

    sat = 0
    for name, m, x in HELD_CASES:
        parts = []
        for pol in (None, learned, RandomPolicy(args.seed)):
            ir, _ = TorchSource().to_ir(m, x)
            eg = EGraph()
            root = eg.add_term(ir.root)
            eg.run(
                rules, root, max_iterations=args.iterations, policy=pol
            )
            parts.append(_partition(eg))
        assert parts[0] == parts[1] == parts[2], name
        sat += 1
    print(
        f"  saturation partition identical (default/learned/random): "
        f"OK ({sat})"
    )


if __name__ == "__main__":
    main()
