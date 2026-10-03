"""Measure whether the hand-written pairing coordination is optimal.

``EGraph.extract_paired`` (``catopt_core.egraph.extract``) is the
non-local *coordination* mechanism for the product law.  Per-class
greedy extraction cannot see that ``k`` members each choosing
``split_i(fused)`` share ONE fused GEMM, so the heuristic forces every
pairing member to its split enode, steers member-reaching consumers
through it, and then :func:`_select_best_term` keeps whichever of the
forced term and the greedy term is cheaper (``optimize.py``).

That is an *all-or-nothing* coordination policy: it can fuse every
pairing group or none, and its steering is a fixed greedy route.  This
probe measures how far that policy is from the true optimum over the
coordination action space.

The action space (bounded, stated explicitly):

* every pairing-group **member class** -> any of its enodes (its own
  projection, its split, or a rule-introduced member);
* every **steering-candidate class** (a reachable class holding both a
  member-reaching and a bypassing enode — exactly the classes the
  shipped steering touches) -> any of its enodes.

All other classes stay greedy, exactly as the shipped heuristic leaves
them.  The space is ``prod_c len(nodes(c))`` over those classes; an
instance is skipped when that exceeds ``--space-cap``.  On every
instance small enough to enumerate the *full* per-class space, the
coordination optimum is checked to equal the full optimum (the true
minimum of any extracted term), so this is not a restriction of it.

Run::

    .venv/bin/python tools/coordination_probe.py
    .venv/bin/python tools/coordination_probe.py --instances 80
"""

from __future__ import annotations

import argparse
import itertools
import random
from typing import Any

from catopt_core.cost import (
    count_cost,
    dag_cost,
    executor_cost_for,
    flops_cost,
    launch_aware_cost,
)
from catopt_core.egraph import EGraph
from catopt_core.ir import Op, Param, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_core.laws.pairing import pair_shared_input_linears

#: Terms priced at or above this are provably ill-typed (the models
#: charge ``_INVALID_COST`` = 1e15); such draws are reported, not
#: silently averaged in.
_INVALID = 1e14

#: Default cap on the coordination space enumerated per instance.
_SPACE_CAP = 4096

#: The shipped extraction default (``optimize._default_cost_fn``) plus
#: the kernel-count / FLOP proxies the ladder is scored under.
_MODELS: tuple[tuple[str, Any], ...] = (
    ("executor (default)", executor_cost_for(lowering="generic")),
    ("launch_aware", launch_aware_cost),
    ("count", count_cost),
    ("flops", flops_cost),
)


def _tt(*shape: int) -> TensorType:
    """Return a ``TensorType`` from a raw shape tuple."""
    return TensorType(tuple(shape))


def _node_key(n: Any) -> tuple:
    """Canonical, deterministic enode ordering key."""
    return (n.op, n.children, repr(n.attrs))


def _reachable(eg: EGraph, root: int) -> list[int]:
    """Canonical e-class ids reachable from *root*, in DFS order."""
    root = eg.find(root)
    seen: set[int] = set()
    stack = [root]
    order: list[int] = []
    while stack:
        cid = eg.find(stack.pop())
        if cid in seen:
            continue
        seen.add(cid)
        order.append(cid)
        for n in eg._classes[cid].nodes:
            for ch in n.children:
                stack.append(ch)
    return order


def _member_classes(eg: EGraph, groups: list) -> dict[int, Any]:
    """Map each pairing-member canonical class id to its split enode."""
    out: dict[int, Any] = {}
    for g in groups:
        for cid, node in g.items():
            out[eg.find(cid)] = node
    return out


def _descendants(
    eg: EGraph, cid: int, stack: frozenset = frozenset()
) -> frozenset:
    """Canonical e-class ids reachable from *cid* (cycle-guarded)."""
    cid = eg.find(cid)
    if cid in stack:
        return frozenset({cid})
    out: set[int] = {cid}
    for n in eg._classes[cid].nodes:
        for ch in n.children:
            out |= _descendants(eg, ch, stack | {cid})
    return frozenset(out)


def _steering_candidates(
    eg: EGraph, root: int, groups: list
) -> list[int]:
    """Classes with both a member-reaching and a bypassing enode.

    Exactly the classes the shipped ``extract_paired`` steering
    touches: some enode's descendants hit a pairing member and some do
    not, so the member route and the bypass compete.
    """
    mc = set(_member_classes(eg, groups))
    cands: list[int] = []
    for cid in _reachable(eg, root):
        ec = eg._classes[cid]
        if cid in mc or len(ec.nodes) < 2:
            continue
        reaching = [
            n
            for n in ec.nodes
            if any(mc & _descendants(eg, ch) for ch in n.children)
        ]
        if reaching and len(reaching) < len(ec.nodes):
            cands.append(cid)
    return cands


def _relevant(eg: EGraph, root: int, groups: list) -> list[int]:
    """Return the coordination-relevant classes (members + steering)."""
    mc = _member_classes(eg, groups)
    return sorted(mc) + _steering_candidates(eg, root, groups)


def _space_size(eg: EGraph, rel: list[int]) -> int:
    """Return the number of enode assignments in the space."""
    prod = 1
    for cid in rel:
        prod *= len(eg._classes[cid].nodes)
    return prod


def coordination_optimum(
    eg: EGraph,
    root: int,
    groups: list,
    cost_fn: Any,
    space_cap: int = _SPACE_CAP,
) -> tuple[tuple[float, Any] | None, int]:
    """Exhaustive optimum over the coordination action space.

    Returns ``((cost, term) | None, space_size)``; ``None`` when the
    space exceeds *space_cap*.
    """
    rel = _relevant(eg, root, groups)
    size = _space_size(eg, rel)
    if size > space_cap:
        return None, size
    choices = [sorted(eg._classes[c].nodes, key=_node_key) for c in rel]
    best: tuple[float, Any] = (float("inf"), None)
    for combo in itertools.product(*choices):
        over = dict(zip(rel, combo, strict=True))
        t = eg.extract_best(root, cost_fn, overrides=over)
        if t is None:
            continue
        c = dag_cost(t, cost_fn)
        if c < best[0]:
            best = (c, t)
    return best, size


def shipped_term(
    eg: EGraph, root: int, groups: list, cost_fn: Any
) -> tuple[float, Any]:
    """Return the shipped policy's term: ``min(greedy, paired)``.

    Mirrors ``optimize._select_best_term``'s pairing branch — greedy
    extraction versus the coordinated forced extraction, cheaper wins
    (the forced term on an exact tie, as the shipped ``<=`` does).
    """
    best = eg.extract_best(root, cost_fn)
    bc = dag_cost(best, cost_fn)
    forced = eg.extract_paired(root, cost_fn, groups)
    fc = (
        dag_cost(forced, cost_fn)
        if forced is not None
        else float("inf")
    )
    return (fc, forced) if fc <= bc else (bc, best)


def naive_term(
    eg: EGraph, root: int, cost_fn: Any
) -> tuple[float, Any]:
    """Return the naive player: plain per-class greedy extraction."""
    t = eg.extract_best(root, cost_fn)
    return dag_cost(t, cost_fn), t


def build_egraph(
    src: Op, iters: int, nodes: int
) -> tuple[EGraph, int, list]:
    """Saturate *src*, run the pairing pass, re-saturate briefly.

    Mirrors ``optimize``'s flow: saturate -> ``_pairing_and_lifts``
    (pair + brief re-saturation) -> extraction.
    """
    eg = EGraph()
    root = eg.add_term(src)
    eg.run(ALL_RULES, root, max_iterations=iters, max_nodes=nodes)
    groups = pair_shared_input_linears(eg)
    if groups:
        eg.rebuild()
        eg.run(ALL_RULES, root, max_iterations=1, max_nodes=nodes)
    return eg, root, groups


# -- instance builders -------------------------------------------------


def _group(gi: int, kind: str, k: int = 2) -> Op:
    """Return a shared-input projection group combined by *kind*."""
    x = Var(f"x{gi}", _tt(2, 4))
    members = [
        Op.make("linear", x, Param(f"W{gi}_{j}", _tt(8, 4)))
        for j in range(k)
    ]
    if kind == "sub":
        t = members[0]
        for m in members[1:]:
            t = Op.make("sub", t, m)
        return t
    if kind == "gated":
        return Op.make("mul", Op.make("silu", members[0]), members[1])
    t = members[0]
    for m in members[1:]:
        t = Op.make("add", t, m)
    return t


def structured_instances() -> list[tuple[str, Op]]:
    """Hand-built instances: tie cases and mix cases."""
    cases: list[tuple[str, Op]] = []
    cases.append(
        (
            "one sub group",
            _group(0, "sub"),
        )
    )
    cases.append(
        (
            "one add group (weight-merge bypass)",
            _group(0, "add"),
        )
    )
    cases.append(
        (
            "one gated group",
            _group(0, "gated"),
        )
    )
    # a "fusing wins" group combined with a "fusing loses" group: the
    # shipped all-or-nothing policy cannot fuse one without the other.
    cases.append(
        (
            "sub-group x add-group (mix)",
            Op.make("mul", _group(0, "sub"), _group(1, "add")),
        )
    )
    cases.append(
        (
            "two sub-groups + add-group (mix)",
            Op.make(
                "mul",
                Op.make("mul", _group(0, "sub"), _group(1, "sub")),
                _group(2, "add"),
            ),
        )
    )
    cases.append(
        (
            "three sub-groups (all fuse)",
            Op.make(
                "mul",
                Op.make("mul", _group(0, "sub"), _group(1, "sub")),
                _group(2, "sub"),
            ),
        )
    )
    return cases


# -- seeded random family ----------------------------------------------


_CONSUMERS = ("sub", "add", "mul", "gated", "gated", "sub")


def _rand_consumer(
    rng: random.Random, members: list[Op], kind: str
) -> Op:
    """Combine *members* by a randomly chosen consumer op."""
    if kind == "gated":
        return Op.make("mul", Op.make("silu", members[0]), members[1])
    t = members[0]
    for m in members[1:]:
        t = Op.make(kind, t, m)
    return t


def random_instance(
    seed: int, iters: int, nodes: int
) -> tuple[EGraph, int, list]:
    """Draw a seeded instance: 1-3 groups under a random root op."""
    rng = random.Random(seed)
    parts: list[Op] = []
    for gi in range(rng.choice([1, 2, 2, 3])):
        x = Var(f"x{gi}", _tt(2, 4))
        members = [
            Op.make("linear", x, Param(f"W{gi}_{j}", _tt(8, 4)))
            for j in range(rng.choice([2, 2, 3]))
        ]
        parts.append(
            _rand_consumer(rng, members, rng.choice(_CONSUMERS))
        )
    root_t = parts[0]
    for p in parts[1:]:
        root_t = Op.make(rng.choice(["add", "mul", "sub"]), root_t, p)
    return build_egraph(root_t, iters, nodes)


# -- reporting ---------------------------------------------------------


def _row(
    eg: EGraph,
    root: int,
    groups: list,
    cost_fn: Any,
    space_cap: int,
) -> tuple[float, float, float, int, int] | None:
    """``(naive, shipped, optimal, space, n_relevant)`` or ``None``."""
    opt, size = coordination_optimum(
        eg, root, groups, cost_fn, space_cap
    )
    if opt is None:
        return None
    nv, _ = naive_term(eg, root, cost_fn)
    sv, _ = shipped_term(eg, root, groups, cost_fn)
    ov, _ = opt
    if max(nv, sv, ov) >= _INVALID:
        return None
    return nv, sv, ov, size, len(_relevant(eg, root, groups))


def _ladder(rows: list[tuple[float, float, float]]) -> dict[str, float]:
    """Summarise ``(naive, shipped, opt)`` triples into ladder stats."""
    out: dict[str, float] = {}
    for label, idx in (("naive", 0), ("shipped", 1)):
        ratios = [r[idx] / r[2] if r[2] else 1.0 for r in rows]
        out[f"{label}_mean"] = sum(ratios) / len(ratios)
        out[f"{label}_opt"] = sum(
            1 for r in rows if r[idx] <= r[2] + 1e-9
        )
        out[f"{label}_worst"] = max(ratios)
    return out


def full_space_optimum(
    eg: EGraph, root: int, cost_fn: Any, space_cap: int
) -> tuple[float, Any] | None:
    """Optimum over EVERY reachable multi-enode class (validation).

    The true minimum of any extracted term.  Enumerable only on small
    graphs; used to confirm the coordination optimum is not beaten by
    a choice outside the coordination space.
    """
    rel = [
        c
        for c in _reachable(eg, root)
        if len(eg._classes[c].nodes) >= 2
    ]
    if _space_size(eg, rel) > space_cap:
        return None
    choices = [sorted(eg._classes[c].nodes, key=_node_key) for c in rel]
    best: tuple[float, Any] = (float("inf"), None)
    for combo in itertools.product(*choices):
        over = dict(zip(rel, combo, strict=True))
        t = eg.extract_best(root, cost_fn, overrides=over)
        if t is None:
            continue
        c = dag_cost(t, cost_fn)
        if c < best[0]:
            best = (c, t)
    return best


def report_structured(space_cap: int, iters: int, nodes: int) -> None:
    """Print the per-instance ladder for the hand-built cases."""
    print("== structured instances ==")
    header = (
        f"  {'instance':<38} {'model':<18} "
        f"{'naive':>10} {'shipped':>10} {'opt':>10} "
        f"{'gap%':>7} {'space':>6}"
    )
    print(header)
    for name, src in structured_instances():
        eg, root, groups = build_egraph(src, iters, nodes)
        if not groups:
            print(f"  {name:<38} (no pairing groups)")
            continue
        for mname, cf in _MODELS:
            r = _row(eg, root, groups, cf, space_cap)
            if r is None:
                print(f"  {name:<38} {mname:<18} (space/ill-typed)")
                continue
            nv, sv, ov, size, _ = r
            gap = (sv - ov) / ov * 100 if ov else 0.0
            print(
                f"  {name:<38} {mname:<18} {nv:>10.6g} {sv:>10.6g} "
                f"{ov:>10.6g} {gap:>6.2f}% {size:>6}"
            )
    print()


def report_family(
    n: int, space_cap: int, iters: int, nodes: int
) -> None:
    """Print the ladder over the seeded random family."""
    print(f"== seeded random family (n={n} draws) ==")
    validated = 0
    for mname, cf in _MODELS:
        rows: list[tuple[float, float, float]] = []
        gaps: list[tuple[float, float, int, int]] = []
        spaces: list[int] = []
        ngroups: list[int] = []
        nrel: list[int] = []
        no_groups = 0
        too_big = 0
        for s in range(n):
            eg, root, groups = random_instance(s, iters, nodes)
            if not groups:
                no_groups += 1
                continue
            r = _row(eg, root, groups, cf, space_cap)
            if r is None:
                too_big += 1
                continue
            nv, sv, ov, size, nr = r
            rows.append((nv, sv, ov))
            spaces.append(size)
            ngroups.append(len(groups))
            nrel.append(nr)
            # Validate the coordination optimum against the full
            # per-class optimum wherever that is enumerable (once per
            # instance, on the default model).
            if mname.startswith("executor"):
                full = full_space_optimum(eg, root, cf, space_cap)
                if full is not None:
                    validated += 1
                    if full[0] < ov - 1e-9:
                        print(
                            f"     !! coord optimum beaten by full "
                            f"space on seed {s}"
                        )
            if sv - ov > 1e-9:
                gaps.append((sv - ov, (sv - ov) / ov * 100, s, size))
        if not rows:
            print(f"\n  {mname}: no enumerable instances")
            continue
        st = _ladder(rows)
        nsub = len(rows) - int(st["shipped_opt"])
        nsubn = len(rows) - int(st["naive_opt"])
        print(
            f"\n  -- {mname} --  analysed {len(rows)}/{n} "
            f"(no groups {no_groups}, space>cap {too_big})"
        )
        print("     player   mean ratio   %at-opt   worst    n_subopt")
        for label in ("naive", "shipped"):
            print(
                f"     {label:<8} {st[label + '_mean']:>9.4f}   "
                f"{st[label + '_opt'] / len(rows) * 100:>6.1f}%   "
                f"{st[label + '_worst']:>7.4f}   "
                f"{nsubn if label == 'naive' else nsub:>5}"
            )
        if gaps:
            gaps.sort()
            print(
                f"     shipped gap: n={len(gaps)} "
                f"max abs {gaps[-1][0]:.4g}  "
                f"max rel {max(g[1] for g in gaps):.3f}%  "
                f"median rel {gaps[len(gaps) // 2][1]:.3f}%  "
                f"seeds {sorted(g[2] for g in gaps)}"
            )
        print(
            f"     space |A|: min {min(spaces)} max {max(spaces)}; "
            f"groups min {min(ngroups)} max {max(ngroups)}; "
            f"relevant classes min {min(nrel)} max {max(nrel)}"
        )
    print(
        f"\n  coordination optimum validated == full per-class "
        f"optimum on {validated} instances"
    )
    print()


def main(argv: list[str] | None = None) -> int:
    """Run the coordination ladder and print the findings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        type=int,
        default=60,
        help="seeded random draws to score (default 60)",
    )
    parser.add_argument(
        "--space-cap",
        type=int,
        default=_SPACE_CAP,
        help="max coordination-space size enumerated per instance",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="saturation iterations for the first pass (default 2)",
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=3000,
        help="e-graph node cap per pass (default 3000)",
    )
    args = parser.parse_args(argv)
    report_structured(args.space_cap, args.iterations, args.nodes)
    report_family(
        args.instances, args.space_cap, args.iterations, args.nodes
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
