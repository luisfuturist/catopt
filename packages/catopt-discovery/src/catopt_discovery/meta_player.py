"""The learned meta-arena player — Q3's arm.

Plan 0021's open question: can a *learned* player beat the scripted
``saturate → extract`` playbook on the mixed board?  The probe's
baseline table says the headroom is real — every baseline ties at
baseline on the fold-site cases while a ``declare`` exists that
wins — but no baseline plays the line, because the paying sequence
is four moves deep: ``declare`` a fresh fold of a mined spec,
``saturate`` so the minted rule fires, ``handle`` to bind a
supported kernel, ``extract`` to collect.

``MetaLearnedPlayer`` is the same architecture as
``arena_player.LearnedPlayer`` — a linear softmax policy over the
legal move set, trained by episode-return REINFORCE — featurizing
:class:`meta_arena.MetaState` instead.  The board is small (~dozens
of legal moves vs the construction arena's ~50k), so the softmax
is *exact* over the unplayed set — no perceptual cap, no played-mass
correction; the gradient row is ``φ_chosen - E_π[φ]`` computed
directly.

``gen_cases`` builds the training corpus: seeded variants of the
sites the handlers cover (a ``mul(x, sigmoid(x))`` under a ruleset
with ``silu_fold`` stripped pays exactly one line — the
declare→saturate→handle chain), controls where nothing pays (the
policy must learn *not* to burn steps), and a swiglu site whose
kernel enters the bound explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from collections.abc import Callable, Iterable
from typing import Any

from catopt_core.cost.basic import count_cost
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import DEFAULT

from . import lawdata
from . import meta_arena as ma

__all__ = [
    "MetaLearnedPlayer",
    "evaluate",
    "featurize",
    "gen_cases",
    "player_table",
    "train_player",
]

#: REINFORCE constants — same contract as ``arena_player``'s:
#: per-episode-return update normalized by step count, EMA running
#: baseline, clipped advantage.
_LR = 0.1
_BASELINE_EMA = 0.2
_CLIP = 40.0


# ---------------------------------------------------------------------------
#  Featurization — (MetaState, Action) -> feature row over the schema
# ---------------------------------------------------------------------------


def _bucket(name: Any, kind: str) -> int:
    """Stable-hash *name* into ``META_ARENA_HASH_BUCKETS[kind]``."""
    width = lawdata.META_ARENA_HASH_BUCKETS[kind]
    h = hashlib.sha256(str(name).encode("utf-8")).digest()
    return int.from_bytes(h[:4]) % width


def _dot(w: dict, feats: dict) -> float:
    """Return the linear score ``w·feats`` over a sparse feature dict."""
    s = 0.0
    for k, v in feats.items():
        wv = w.get(k)
        if wv:
            s += wv * v
    return s


class _Hist:
    """Within-episode play counts — the adaptation channel."""

    def __init__(self) -> None:
        """Start an empty episode history."""
        self.counts: dict[str, int] = {}
        self.n = 0

    def record(self, op: str) -> None:
        """Tally one played move of kind *op*."""
        self.counts[op] = self.counts.get(op, 0) + 1
        self.n += 1

    def rate(self, op: str) -> float:
        """Share of this episode's plays of kind *op*."""
        return self.counts.get(op, 0) / max(self.n, 1)


def _state_feats(state: ma.MetaState, hist: _Hist) -> dict[str, float]:
    """Broadcast features — the board summary every move carries."""
    total = state.n_enodes + max(state.budget_left, 0)
    rel = (
        state.best_cost / state.baseline_cost
        if state.baseline_cost > 0
        else 1.0
    )
    return {
        "bias": 1.0,
        "st:steps": state.steps / 32.0,
        "st:enodes_frac": state.n_enodes / max(total, 1),
        "st:classes_frac": state.n_classes / max(state.n_enodes, 1),
        "st:improve": max(0.0, 1.0 - rel),
        "st:declared": len(state.declared) / 8.0,
        "st:handled": len(state.handles) / 8.0,
        "st:unhandled": (
            len(state.declared_bodies) - len(state.handles)
        )
        / 8.0,
        "st:handleable": _handleable(state) / 8.0,
        "st:specs": len(state.specs) / 32.0,
        "st:fires": sum(c for _, c in state.rule_fires) / 64.0,
        "h:fire": hist.rate("fire"),
        "h:saturate": hist.rate("saturate"),
        "h:declare": hist.rate("declare"),
        "h:handle": hist.rate("handle"),
    }


def _handleable(state: ma.MetaState) -> int:
    """Mined specs some handler's canonical pattern covers.

    The precondition of the paying line — ``declare`` a fresh fold
    of a covered spec, then ``handle`` binds it to the kernel.  A
    spec is coverable iff its alpha-canonical form equals a handler
    pattern's (``meta_arena._handle_actions``' own test).
    """
    pats = {p for _, p in state.handler_specs}
    return sum(1 for s in state.specs if ma._canon_spec(s) in pats)


def _spec_index(action: ma.Action) -> float:
    """Mined-spec position a ``declare`` targets (``foldabs_i``)."""
    cons = action.params.get("construction") or {}
    params = cons.get("params") or {}
    suffix = str(params.get("name", "")).rsplit("_", 1)[-1]
    return int(suffix) / 32.0 if suffix.isdigit() else 0.5


def _act_feats(
    action: ma.Action, sf: dict[str, float]
) -> dict[str, float]:
    """Score the move: op one-hot, params, interactions."""
    op = action.op
    out: dict[str, float] = {}
    if f"op:{op}" in lawdata.META_ARENA_FEATURES:
        out[f"op:{op}"] = 1.0
    if op == "fire":
        out[f"a:r:{_bucket(action.params.get('rule'), 'rule'):02d}"] = (
            1.0
        )
    elif op == "saturate":
        budget = action.params.get("budget")
        if budget is None:
            out["a:unbounded"] = 1.0
        else:
            out["a:budget"] = float(budget) / 512.0
        out["x:pressure:saturate"] = sf["st:enodes_frac"]
    elif op == "declare":
        out["a:spec_i"] = _spec_index(action)
        out["x:specs:declare"] = sf["st:specs"]
        out["x:handleable:declare"] = sf["st:handleable"]
    elif op == "handle":
        out[
            f"a:h:{_bucket(action.params.get('handler'), 'handler')}"
        ] = 1.0
        out[f"a:o:{_bucket(action.params.get('object'), 'object')}"] = (
            1.0
        )
        out["x:declared:handle"] = sf["st:declared"]
        out["x:unhandled:handle"] = sf["st:unhandled"]
    elif op == "extract":
        out["x:extract:improve"] = sf["st:improve"]
        out["x:late:extract"] = sf["st:steps"]
    return out


def featurize(
    state: ma.MetaState, action: ma.Action, hist: _Hist | None = None
) -> dict[str, float]:
    """Return the feature row for one legal move — public for tests.

    Keys are a subset of :data:`lawdata.META_ARENA_FEATURES`;
    scoring is ``Σ w[name]·value``.  *hist* defaults to an empty
    episode history.
    """
    hist = _Hist() if hist is None else hist
    sf = _state_feats(state, hist)
    return {**sf, **_act_feats(action, sf)}


# ---------------------------------------------------------------------------
#  The player
# ---------------------------------------------------------------------------


def _key(action: ma.Action) -> tuple:
    """Stable dedup key — played-move mask, replayable."""
    params = action.params
    return (
        action.op,
        repr(sorted(params.items(), key=lambda kv: kv[0])),
    )


class MetaLearnedPlayer:
    """A softmax linear policy over the meta-arena's legal moves.

    ``score(move) = w·φ(state, move)`` over
    :data:`lawdata.META_ARENA_FEATURES`; a step samples the softmax
    over the *unplayed* legal set (``greedy=True`` takes the argmax,
    ``temperature`` scales the logits).  *weights* is a
    ``{feature_name: float}`` table — ``None`` binds the shipped
    :data:`lawdata.META_ARENA_PLAYER_WEIGHTS` (empty ⇒ uniform, the
    honest cold start).  ``learn=False`` freezes the policy for eval.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        weights: dict | None = None,
        lr: float = _LR,
        temperature: float = 1.0,
        greedy: bool = False,
        learn: bool = True,
        legal: Any = None,
        epsilon: float = 0.0,
    ) -> None:
        """Bind the weight table, the sampler and the learn flag.

        *epsilon* mixes uniform exploration into the sample — the
        standard sparse-reward lever: with probability ε the step
        picks uniformly over the unplayed set (still scored into the
        gradient row at its softmax mass).  Eval arms run ε=0.
        """
        self._rng = random.Random(seed)
        self._lr = lr
        self._temp = max(temperature, 1e-6)
        self._greedy = greedy
        self._learn = learn
        self._eps = epsilon
        self._legal = ma.legal_actions if legal is None else legal
        self._w = dict(
            lawdata.META_ARENA_PLAYER_WEIGHTS
            if weights is None
            else weights
        )
        self._baseline = 0.0
        self._played: set = set()
        self._hist = _Hist()
        self._ep: list = []
        self._prev_steps = -1

    def frozen(self, seed: int = 0, *, greedy: bool = False) -> Any:
        """Return a non-learning copy sharing the trained weights."""
        return MetaLearnedPlayer(
            seed=seed,
            weights=self.weights_dict(),
            temperature=self._temp,
            greedy=greedy,
            learn=False,
            legal=self._legal,
        )

    def weights_dict(self) -> dict[str, float]:
        """Return the weight table — ``{name: w}`` over the schema."""
        return {
            n: self._w.get(n, 0.0) for n in lawdata.META_ARENA_FEATURES
        }

    def _observe(self, state: ma.MetaState) -> None:
        """Detect the episode boundary — a fresh board resets state."""
        if state.steps == 0 and self._prev_steps > 0:
            self._played.clear()
            self._hist = _Hist()
        self._prev_steps = state.steps

    def __call__(self, state: ma.MetaState) -> ma.Action | None:
        """Score the unplayed legal moves, sample one, remember it."""
        self._observe(state)
        acts = [
            a for a in self._legal(state) if _key(a) not in self._played
        ]
        if not acts:
            return None
        feats = [featurize(state, a, self._hist) for a in acts]
        logits = [_dot(self._w, f) / self._temp for f in feats]
        top = max(logits)
        exps = [math.exp(x - top) for x in logits]
        if self._greedy:
            i = logits.index(top)
        elif self._eps and self._rng.random() < self._eps:
            i = self._rng.randrange(len(acts))
        else:
            i = self._sample(exps)
        if self._learn:
            self._ep.append(_grad_row(feats, exps, i))
        chosen = acts[i]
        self._played.add(_key(chosen))
        self._hist.record(chosen.op)
        return chosen

    def _sample(self, exps: list[float]) -> int:
        """Draw an index proportionally to the unnormalized weights."""
        r = self._rng.random() * sum(exps)
        acc = 0.0
        for i, e in enumerate(exps):
            acc += e
            if r <= acc:
                return i
        return len(exps) - 1

    def finish_episode(
        self, total: float, rewards: Iterable[float] | None = None
    ) -> None:
        """REINFORCE — reward-to-go per step, not just the total.

        Per step ``t``: ``G_t = Σ_{k≥t} r_k`` (the remaining episode
        reward — the *causal* credit: a ``declare`` gets the terminal
        payout it armed, moves after it don't).  The update is
        ``w += lr·(G_t - baseline)·(φ_t - E_π[φ_t])`` accumulated
        over the episode; *rewards* comes from
        ``Trajectory.reports`` — ``None`` falls back to the flat
        episode return (the causal credit is the point; the fallback
        exists for callers that only have the total).  The EMA
        baseline updates from the raw total; advantages are clipped
        so one paying episode cannot blow the weights up.  No-op
        when ``learn=False``.
        """
        adv = total - self._baseline
        self._baseline += _BASELINE_EMA * adv
        if not (self._learn and self._ep):
            self._ep = []
            return
        rs = list(rewards) if rewards is not None else None
        n = len(self._ep)
        grad: dict[str, float] = {}
        for t, (phi, ephi) in enumerate(self._ep):
            g_t = (
                sum(rs[t:]) if rs is not None and t < len(rs) else total
            )
            a = max(-_CLIP, min(_CLIP, g_t - self._baseline))
            for k, v in phi.items():
                grad[k] = grad.get(k, 0.0) + self._lr * a / n * v
            for k, v in ephi.items():
                grad[k] = grad.get(k, 0.0) - self._lr * a / n * v
        for k, g in grad.items():
            self._w[k] = self._w.get(k, 0.0) + g
        self._ep = []


def _grad_row(
    feats: list[dict], exps: list[float], i: int
) -> tuple[dict, dict]:
    """``(φ_chosen, E_π[φ])`` under the softmax over unplayed moves."""
    z = sum(exps)
    ephi: dict[str, float] = {}
    for f, e in zip(feats, exps, strict=True):
        for k, v in f.items():
            ephi[k] = ephi.get(k, 0.0) + v * e / z
    return feats[i], ephi


# ---------------------------------------------------------------------------
#  Training and evaluation — disjoint case streams
# ---------------------------------------------------------------------------


def gen_cases(seed: int, n: int) -> list:
    """Seeded case variants — the mixed board's paying structures.

    Kinds: ``silu`` sites under a ruleset with ``silu_fold``
    stripped (the declare→handle line is the only path back — silu
    stays in the ambient vocabulary through ``silu_expand`` /
    ``silu_mul_form``); ``swiglu`` sites whose kernel enters the
    bound explicitly; ``softsign`` sites under ``sans`` where the
    handler kernel is *out* of vocabulary — an honest negative, the
    line must decline; and ``control`` terms under ``DEFAULT`` where
    nothing pays and the best play is a cheap extract.  Case tuples
    follow ``meta_probe``'s arity — ``(name, term, rules, decls,
    supported)``.
    """
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
    silu_spelled = Op.make("mul", x, Op.make("sigmoid", x))
    builders = [
        ("silu", lambda: (silu_spelled, sans, None)),
        (
            "silu_nested",
            lambda: (
                Op.make("tanh", Op.make("add", silu_spelled, y)),
                sans,
                None,
            ),
        ),
        (
            "swiglu",
            lambda: (
                Op.make("mul", silu_spelled, y),
                sans,
                ma._supported_or(None, silu_spelled, sans) | {"swiglu"},
            ),
        ),
        (
            "softsign",
            lambda: (
                Op.make(
                    "div",
                    x,
                    Op.make("add", Op.make("abs", x), Const(1)),
                ),
                sans,
                None,
            ),
        ),
        (
            "control",
            lambda: (
                Op.make(
                    "add", Op.make("mul", x, y), Op.make("matmul", x, y)
                ),
                DEFAULT,
                None,
            ),
        ),
    ]
    out = []
    for i in range(n):
        kind, build = builders[rng.randrange(len(builders))]
        term, rules, sup = build()
        out.append((f"gen{i}:{kind}", term, rules, (), sup))
    return out


def _board(case: list | tuple, cf: Any, max_nodes: int) -> ma.MetaArena:
    """Build the arena for one case tuple (probe arity)."""
    rules = case[2] if len(case) > 2 else None
    sup = case[4] if len(case) > 4 else None
    return ma.MetaArena(
        case[1],
        rules,
        cost_fn=cf,
        supported=sup,
        max_nodes=max_nodes,
    )


def train_player(
    player: MetaLearnedPlayer,
    cases: Iterable,
    *,
    budget: int,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> list[float]:
    """Play one episode per case; REINFORCE on each trajectory total."""
    cf = cost_fn or count_cost
    totals: list[float] = []
    for case in cases:
        traj = ma.run_episode(
            _board(case, cf, max_nodes), player, budget
        )
        player.finish_episode(
            traj.total, rewards=[r.reward for r in traj.reports]
        )
        totals.append(traj.total)
    return totals


def evaluate(
    arms: dict[str, Callable[[int], Any]],
    cases: list,
    *,
    budget: int,
    cost_fn: Any = None,
    max_nodes: int = 20_000,
) -> dict[str, list[dict]]:
    """Play every arm on the same cases; paired boards, probe rows."""
    cf = cost_fn or count_cost
    out: dict[str, list[dict]] = {}
    for name, mk in arms.items():
        rows: list[dict] = []
        for i, case in enumerate(cases):
            arena = _board(case, cf, max_nodes)
            traj = ma.run_episode(arena, mk(i), budget)
            t = traj.terminal
            rows.append(
                {
                    "case": case[0],
                    "steps": len(traj.reports),
                    "reward": round(traj.total, 3),
                    "cost": t.cost if t is not None else None,
                    "cert": t.certificate_ok if t is not None else None,
                    "used_declared": (
                        t.used_declared if t is not None else ()
                    ),
                }
            )
        out[name] = rows
    return out


def player_table(table: dict[str, list[dict]]) -> str:
    """Render per-arm means over the eval rows."""
    head = (
        f"{'player':<15} {'cases':>6} {'reward':>9} {'cost':>7}"
        f" {'cert':>5} {'decl':>5} {'steps':>6}"
    )
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        n = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        certs = sum(1 for r in rows if r["cert"])
        decls = sum(1 for r in rows if r["used_declared"])
        lines.append(
            f"{name:<15} {len(rows):>6} "
            f"{sum(r['reward'] for r in rows) / n:>9.2f} "
            f"{(sum(costs) / len(costs)) if costs else float('nan'):>7.2f} "
            f"{certs:>5} {decls:>5} "
            f"{sum(r['steps'] for r in rows) / n:>6.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Train on one case stream, evaluate on a disjoint one."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=40)
    ap.add_argument("--eval-cases", type=int, default=10)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--max-nodes", type=int, default=20_000)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=_LR)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    train = gen_cases(args.seed, args.train_cases)
    ev = gen_cases(args.seed + 7919, args.eval_cases)
    player = MetaLearnedPlayer(
        seed=args.seed,
        temperature=args.temperature,
        lr=args.lr,
        epsilon=args.epsilon,
    )
    totals = train_player(
        player,
        train,
        budget=args.budget,
        max_nodes=args.max_nodes,
    )
    arms: dict[str, Callable[[int], Any]] = {
        "scripted": lambda i: ma.ScriptedPlayer(),
        "greedy": lambda i: ma.GreedyPlayer(),
        "random": lambda i: ma.RandomPlayer(
            random.Random(args.seed + 10_000 + i)
        ),
        "learned": lambda i: player.frozen(args.seed + 20_000 + i),
        "learned-greedy": lambda i: player.frozen(
            args.seed + 30_000 + i, greedy=True
        ),
    }
    table = evaluate(
        arms, ev, budget=args.budget, max_nodes=args.max_nodes
    )
    print(
        "== meta-arena player — learned vs baselines =="
    )  # stdout-compat
    print(player_table(table))  # stdout-compat
    decade = max(len(totals) // 10, 1)
    means = [
        round(sum(totals[i : i + decade]) / decade, 2)
        for i in range(0, len(totals) - decade + 1, decade)
    ]
    print(f"train decade means: {means}")  # stdout-compat
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(
                {
                    "table": table,
                    "train_totals": totals,
                    "weights": player.weights_dict(),
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
