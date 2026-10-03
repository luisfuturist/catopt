"""Search-policy value bench — does a learned policy pick better rules?

Plan 0016 stage 12 (ADR 0003, the EVALUATION dimension).  The search
game (:mod:`catopt_core.game`) has several players —
:mod:`catopt_core.policies` plus a learned one,
:mod:`catopt_torch.learned_policy` — and this suite asks the only
question that matters about them:

    Does a learned search policy pick better rules than random,
    declaration-order, or a cost-model greedy?

Method.  A small family of two-matmul chains ``a·(b·c)`` / ``(a·b)·c``
is generated; the engine's own evaluator
(:func:`catopt_core.trajectories.rule_samples`) prices every legal rule
applied alone, giving each rule a *true* cost delta.  A tiny
``RuleValueNet`` is trained inline (CPU, fixed seed, a few hundred
epochs) on a disjoint set of small chains, then every policy picks one
rule per held-out program.  For each program and policy this suite
records

* the **rank** of the picked rule by true delta (1 = the best rule),
* the picked rule's **cost delta** (positive = it strictly improves),
* whether the pick is a **hit** (a strictly-improving rule).

The headline numbers are the mean rank and the hit rate per policy.

Players compared:

* ``random``      — ``RandomPolicy`` (seeded), the floor;
* ``declaration`` — ``ExistingPolicy``, the engine's rule order;
* ``greedy``      — ``GreedyPolicy`` keyed by the *negative true
  delta*: the one-step cost-model greedy, i.e. the best any
  single-step player can do (the oracle ceiling);
* ``learned``     — ``LearnedPolicy`` over the trained value net.

Honesty.  Greedy is the single-step optimum by construction, so the
learned policy cannot beat it — the question is whether it *matches*
it.  On this held-out set it does not: it ranks rules far better than
random/declaration-order, but it confuses the associativity *direction*
(it picks ``assoc_matmul`` where ``assoc_matmul_rev`` is correct), so
the finding against greedy is a measured NEGATIVE, not a manufactured
win.  The suite also times each policy's decision (benchkit ``Runner``)
— the learned player pays a per-decision cost the heuristics do not.

CPU-only, no network, no CUDA, deterministic (fixed seeds throughout).
"""

from __future__ import annotations

import argparse
import random
import statistics
from pathlib import Path
from typing import Any

import torch
from catopt_core.features import compute_features
from catopt_core.game import Action, GameState
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.policies import (
    ExistingPolicy,
    GreedyPolicy,
    RandomPolicy,
)
from catopt_core.trajectories import rule_samples
from catopt_torch.learned_policy import LearnedPolicy, train_rule_value

from bench.benchkit import (
    Case,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
    collect_env,
)

#: ``--quick`` shrinks the sweep (the CLI applies this dict).
QUICK = {
    "epochs": 120,
    "train_programs": 24,
    "min_run_time": 0.05,
}

#: Policy names, in report order.
_POLICIES = ("random", "declaration", "greedy", "learned")

#: Dims the pretraining chains draw from (small, fixed).
_TRAIN_DIMS = (2, 3, 4)

#: Held-out chains ``(label, d, k, m, left)`` — shapes the policy never
#: trains on (dims reach 7).  ``left`` selects the ``(a·b)·c``
#: bracketing; the seed bracketing is the expensive one, so the true
#: best rule is ``assoc_matmul`` (right seeds) or ``assoc_matmul_rev``
#: (left seeds) whenever the associativity actually pays.  The last two
#: are already-optimal chains (no improving rule) — honest zero-hit
#: rows for every player.
_HELD_OUT = (
    ("R 3x5x2", 3, 5, 2, False),
    ("R 4x6x3", 4, 6, 3, False),
    ("R 2x5x4", 2, 5, 4, False),
    ("L 5x2x6", 5, 2, 6, True),
    ("L 3x2x5", 3, 2, 5, True),
    ("L 4x2x7", 4, 2, 7, True),
    ("R 5x3x6", 5, 3, 6, False),
    ("L 4x6x3", 4, 6, 3, True),
)


def _v(name: str, *shape: int) -> Var:
    """A named tensor variable."""
    return Var(name, TensorType(tuple(shape)))


def chain(d: int, k: int, m: int, *, left: bool = False) -> Op:
    """A two-matmul chain ``a·(b·c)`` (or ``(a·b)·c`` if ``left``).

    ``a`` is ``(d, k)``, ``b`` is ``(k, m)``, ``c`` is ``(m, d)`` — so
    associativity genuinely pays when the shared dims differ, exactly
    the matrix-chain-ordering problem the cost model solves.
    """
    a, b, c = _v("a", d, k), _v("b", k, m), _v("c", m, d)
    if left:
        return Op.make("matmul", Op.make("matmul", a, b), c)
    return Op.make("matmul", a, Op.make("matmul", b, c))


def _ranks_desc(values: list[float]) -> list[float]:
    """1-based average ranks, largest value ranks first (ties share)."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while (
            j + 1 < len(values)
            and values[order[j + 1]] == values[order[i]]
        ):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def _train_programs(n: int, seed: int) -> list[Op]:
    """Small random matmul chains — the pretraining family."""
    rng = random.Random(seed)
    out: list[Op] = []
    for _ in range(n):
        d, k, m = (rng.choice(_TRAIN_DIMS) for _ in range(3))
        out.append(chain(d, k, m, left=rng.random() < 0.5))
    return out


def _fit_policy(
    programs: list[Op],
    rules: Any,
    *,
    epochs: int,
    hidden: int,
    seed: int,
    device: str,
) -> LearnedPolicy:
    """Train a ``RuleValueNet`` on ``programs``; return the policy."""
    samples = []
    for p in programs:
        samples.extend(rule_samples(p, rules))
    model = train_rule_value(
        samples,
        epochs=epochs,
        hidden=hidden,
        seed=seed,
        device=device,
    )
    return LearnedPolicy(
        model, {r.name: r for r in rules}, device=device
    )


def _stmt(pol: Any, state: GameState, actions: list[Action]):
    """A zero-arg callable timing one policy decision."""

    def run() -> None:
        pol.choose(state, actions)

    return run


def _build_case(
    i: int,
    spec: tuple[str, int, int, int, bool],
    rules: Any,
    names: list[str],
    actions: list[Action],
    policy: LearnedPolicy,
) -> Case:
    """One held-out program → a ``Case`` of the four players."""
    label, d, k, m, left = spec
    prog = chain(d, k, m, left=left)
    deltas = {s.rule: s.delta_cost for s in rule_samples(prog, rules)}
    rank_of = dict(
        zip(names, _ranks_desc([deltas[n] for n in names]), strict=True)
    )
    state = GameState(None, 0, features=compute_features(prog))
    players = {
        "random": RandomPolicy(seed=i),
        "declaration": ExistingPolicy(),
        "greedy": GreedyPolicy({n: -deltas[n] for n in names}),
        "learned": policy,
    }
    picks = {
        p: players[p].choose(state, actions).rule for p in _POLICIES
    }
    best = max(names, key=lambda n: deltas[n])
    aux = {
        "program": label,
        "d": d,
        "k": k,
        "m": m,
        "n_rules": len(rules),
        "best_rule": best,
        "best_delta": deltas[best],
        "policies": {
            p: {
                "pick": picks[p],
                "rank": rank_of[picks[p]],
                "delta": deltas[picks[p]],
                "hit": deltas[picks[p]] > 0.0,
            }
            for p in _POLICIES
        },
    }
    variants = [
        Variant(p, _stmt(players[p], state, actions)) for p in _POLICIES
    ]
    return Case(
        name=label,
        params={"program": label, "d": d, "k": k, "m": m},
        variants=variants,
        aux=aux,
    )


def _aggregate(cells: list[Any], policy: str) -> dict:
    """Mean rank / hit rate / mean delta for one policy across cells."""
    ranks = [c.aux["policies"][policy]["rank"] for c in cells]
    deltas = [c.aux["policies"][policy]["delta"] for c in cells]
    hits = [c.aux["policies"][policy]["hit"] for c in cells]
    return {
        "mean_rank": round(statistics.fmean(ranks), 3),
        "mean_delta": round(statistics.fmean(deltas), 3),
        "hits": int(sum(hits)),
        "n": len(hits),
        "hit_rate": round(sum(hits) / len(hits), 3),
    }


def _findings(cells: list[Any]) -> list[Finding]:
    """The typed conclusions — a win over the floor, a loss to greedy."""
    agg = {p: _aggregate(cells, p) for p in _POLICIES}
    lrn, greedy = agg["learned"], agg["greedy"]
    floor = max(
        agg["random"]["mean_rank"], agg["declaration"]["mean_rank"]
    )
    floor_hits = max(agg["random"]["hits"], agg["declaration"]["hits"])

    matches_greedy = (
        lrn["hits"] == greedy["hits"]
        and abs(lrn["mean_rank"] - greedy["mean_rank"]) < 0.5
    )
    beats_floor = lrn["mean_rank"] < floor and lrn["hits"] > floor_hits
    return [
        Finding(
            claim=(
                "a learned search policy matches the cost-model greedy "
                "on held-out programs"
            ),
            verdict=(
                Verdict.PARITY if matches_greedy else Verdict.NEGATIVE
            ),
            headline=(
                f"learned mean rank {lrn['mean_rank']:.1f} vs greedy "
                f"{greedy['mean_rank']:.1f} (hit {lrn['hits']}/"
                f"{lrn['n']} vs {greedy['hits']}/{greedy['n']})"
            ),
            metric="learned mean rank / greedy mean rank",
            value=(
                round(lrn["mean_rank"] / greedy["mean_rank"], 3)
                if greedy["mean_rank"]
                else None
            ),
            evidence={"learned": lrn, "greedy": greedy},
        ),
        Finding(
            claim=(
                "a learned search policy ranks rules better than random "
                "and declaration-order"
            ),
            verdict=(
                Verdict.WIN if beats_floor else Verdict.INCONCLUSIVE
            ),
            headline=(
                f"learned mean rank {lrn['mean_rank']:.1f} vs "
                f"random/declaration {floor:.1f}; hit rate "
                f"{lrn['hit_rate']:.0%} vs "
                f"{floor_hits / lrn['n']:.0%}"
            ),
            metric="learned mean rank",
            value=lrn["mean_rank"],
            evidence={
                "random": agg["random"],
                "declaration": agg["declaration"],
            },
        ),
    ]


def run_bench(args: argparse.Namespace) -> Report:
    """Harnessed entry point: train the policy, score the players."""
    dev = torch.device(getattr(args, "device", None) or "cpu")
    epochs = int(getattr(args, "epochs", None) or 300)
    hidden = int(getattr(args, "hidden", None) or 32)
    n_train = int(getattr(args, "train_programs", None) or 40)
    seed = int(getattr(args, "seed", None) or 0)

    rules = all_rules()
    names = [r.name for r in rules]
    actions = [Action(n) for n in names]

    print(f"[policy_value] training on {n_train} chains on {dev} …")
    policy = _fit_policy(
        _train_programs(n_train, 0),
        rules,
        epochs=epochs,
        hidden=hidden,
        seed=seed,
        device=str(dev),
    )
    cases = [
        _build_case(i, spec, rules, names, actions, policy)
        for i, spec in enumerate(_HELD_OUT)
    ]
    runner = Runner(
        device=dev,
        warmup=int(getattr(args, "warmup", None) or 5),
        min_run_time=float(getattr(args, "min_run_time", None) or 0.2),
    )
    cells = runner.run(cases)
    _print_table(cells)
    return Report(
        suite="policy_value",
        title="Learned search policy vs heuristics",
        summary=(
            "For held-out matmul chains: the rank (1 = best), cost "
            "delta and hit rate of the rule each player picks — random "
            "/ declaration-order / cost-model greedy / learned — plus "
            "the per-decision cost."
        ),
        findings=_findings(cells),
        cells=cells,
        env=collect_env(dev),
        provenance={
            "train_programs": n_train,
            "epochs": epochs,
            "hidden": hidden,
            "seed": seed,
            "n_rules": len(rules),
            "held_out": len(_HELD_OUT),
            "device": str(dev),
        },
    )


def _print_table(cells: list[Any]) -> None:
    """The per-program console twin of the report table."""
    head = f"{'program':<10} {'best':<18} " + " ".join(
        f"{p:>27}" for p in _POLICIES
    )
    print("\n" + head)
    print("-" * len(head))
    for cell in cells:
        aux = cell.aux
        cells_p = aux["policies"]
        row = " ".join(
            f"{cells_p[p]['pick']:>23} #{cells_p[p]['rank']:<3.0f}"
            for p in _POLICIES
        )
        print(f"{aux['program']:<10} {aux['best_rule']:<18} {row}")


def main(argv: list[str] | None = None) -> None:
    """Direct entry point (``python bench/suites/evaluation/...``)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--train-programs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--min-run-time", type=float, default=0.2)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        for key, val in QUICK.items():
            setattr(args, key, val)
    report = run_bench(args)
    if not args.no_artifacts:
        out = Path(args.out)
        report.to_json(out / "policy_value.json")
        report.to_markdown(out / "policy_value.md")
        report.to_html(out / "policy_value.html")
        print(f"[artifacts] {out}/policy_value.{{json,md,html}}")


if __name__ == "__main__":
    main()
