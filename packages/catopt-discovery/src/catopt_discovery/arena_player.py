"""The learned arena player — a softmax linear policy over legal moves.

The construction board's open question (``project/retros/
arena-depth.md``): can a *learned* player beat the authored
``FixedRule`` playbook?  This module is the honest attempt — a
player with the same interface every real player has
(``ArenaState -> Action | None``, moves drawn from
:func:`arena.legal_actions`), whose move scores are a learned linear
function over named features.

The policy
----------

``LearnedPlayer`` scores every legal move as ``w · φ(state, move)``
and samples from the softmax (``temperature`` scales exploration;
``greedy=True`` takes the argmax).  Moves already played this
episode are masked out — the same "spend each move once" discipline
:class:`arena.HeuristicPlayer` uses; re-declaring a stored object is
legal but only re-gauntlets it, so masking keeps the policy's mass
on fresh constructions rather than letting it farm a replayed
payoff.

Featurization is data
---------------------

The feature *names* are content — :data:`lawdata.ARENA_FEATURES` is
the schema, and the learned weight table is a ``{name: float}`` map
over it (:data:`lawdata.ARENA_PLAYER_WEIGHTS`).  Features are scalar
summaries of the serializable state plus the action's own params —
op one-hots, spec size/coverage (``a:cases`` counts the working
cases whose terms contain the spelled subterm — the targeting
signal the frontier baselines lack), kernel/carrier/premise
identity via stable hash buckets (``ARENA_HASH_BUCKETS`` — a linear
model cannot name a kernel op, so the same name lands in the same
bucket on any board), and the referenced object's guard/verdict
state for the object-targeting ops.

Within an episode the player also tracks *what its own moves
produced*: it diffs consecutive observations — a newly stored or
re-gauntleted ``ObjectView`` is the last action's measured result —
into per-op outcome rates (``h:<op>:<stat>`` over
:data:`lawdata.ARENA_HISTORY_STATS`) and a running reward trend
(``st:trend``).  Nothing here reads a ``StepReport``: the player
infers outcomes from the board view alone, which is what makes the
featurization honest — the same information ``ArenaState`` hands
every player.

The training loop
-----------------

:func:`train_player` plays episodes on a per-seed board factory and
applies episode-return REINFORCE: after each episode,
``w += lr·(R - baseline)·Σ_t ∇log π(a_t|s_t)`` where ``R`` is the
``Trajectory.total`` — the rebalanced ``ARENA_REWARD`` composition —
and the baseline is a running mean of past returns (the
guide-real-run lesson: the reward is the episode return, not a
per-step proxy).  :func:`episode_seeds` returns disjoint train/eval
seed ranges so the eval boards are boards the trainer never played;
:func:`evaluate` plays any arm map on a seed list and reports the
honest bars (usable / holdout fires / holdout paid / reward) per
episode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from catopt_discovery import arena as ar
from catopt_discovery import lawdata

__all__ = [
    "FEAT_IDX",
    "LearnedPlayer",
    "episode_seeds",
    "evaluate",
    "featurize",
    "main",
    "train_player",
]

#: Feature-name → column index, aligned to
#: :data:`lawdata.ARENA_FEATURES` — the schema the weight table
#: writes into.
FEAT_IDX = {n: i for i, n in enumerate(lawdata.ARENA_FEATURES)}

#: REINFORCE defaults — resource knobs, not content, so they live
#: here and not in ``lawdata``.  ``_LR`` is per-episode-return and
#: normalized by the episode's step count; ``_BASELINE_EMA`` is the
#: running-mean momentum; ``_CLIP`` bounds the advantage so one
#: paying episode cannot blow the weights up.
_LR = 0.1
_BASELINE_EMA = 0.2
_CLIP = 40.0


def _bucket(name: Any, kind: str) -> int:
    """Stable-hash *name* into ``ARENA_HASH_BUCKETS[kind]`` buckets.

    ``hash()`` is process-salted on strings, so identity features
    must not use it — a trained weight table would silently re-map
    every bucket between runs.  sha256 keeps the same name in the
    same bucket across boards and processes.
    """
    width = lawdata.ARENA_HASH_BUCKETS[kind]
    h = hashlib.sha256(str(name).encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % width


# ---------------------------------------------------------------------------
#  Within-episode history — what the player's own moves produced
# ---------------------------------------------------------------------------


def _view_sig(o: ar.ObjectView) -> tuple:
    """Return an object view's verdict-bearing columns, for diffing."""
    return (
        o.usable,
        o.cleared,
        o.failed,
        o.fires,
        o.paid,
        o.guard_synth,
        o.guard_real,
        o.has_guard,
        len(o.cond_clauses),
    )


def _changed_views(
    prev: ar.ArenaState, state: ar.ArenaState
) -> list[ar.ObjectView]:
    """Return views that are new or whose verdict columns changed."""
    prev_views = {o.alpha_key: o for o in prev.objects}
    return [
        o
        for o in state.objects
        if o.alpha_key not in prev_views
        or _view_sig(o) != _view_sig(prev_views[o.alpha_key])
    ]


def _infer_outcome(
    prev: ar.ArenaState, action: ar.Action, state: ar.ArenaState
) -> tuple[str, float, bool]:
    """Classify what *action* produced, from the observation diff.

    One step touches at most one object: a ``fold``/``lift``/
    ``compose`` (or an ``auto_cond`` minting under a fresh key)
    stores a new ``ObjectView``; ``auto_cond``/``relax_guard``/
    ``specialize`` on a stored key re-gauntlets it, leaving the same
    view with changed verdict columns.  Neither lands → the move was
    declined (or re-measured to an identical view — the honest
    approximation: that outcome is indistinguishable from a refusal
    in the observation).  ``ingest`` applied iff the epoch rotated.
    Returns ``(class, inferred_reward, paid>0)`` — the reward under
    the same ``ARENA_REWARD`` composition the referee reports.
    """
    if action.op == "ingest":
        klass = "corpus" if state.epoch != prev.epoch else "declined"
        return klass, 0.0, False
    hits = _changed_views(prev, state)
    if not hits:
        return "declined", 0.0, False
    o = hits[0]
    rw = lawdata.ARENA_REWARD
    reward = (
        rw["stage"] * o.cleared
        + rw["usable"] * float(bool(o.usable))
        + rw["fire"] * o.fires
        + rw["paid"] * o.paid
    )
    klass = "usable" if o.usable else (o.failed or "cleared")
    return klass, reward, o.paid > 0


@dataclass
class _Hist:
    """Per-op outcome tallies inferred from the observation stream.

    The episode-local second-order features: ``plays``/``usable``/
    ``pay``/``truthfail``/``decline`` are per-op counters;
    ``trend`` is the running mean of inferred step rewards.  The
    feature renderer turns them into the ``h:<op>:<stat>`` /
    ``st:*`` rows — a learned weight on ``h:lift:pay`` answers "did
    lifts pay on boards like this one".
    """

    plays: dict[str, int] = field(default_factory=dict)
    usable: dict[str, int] = field(default_factory=dict)
    pay: dict[str, int] = field(default_factory=dict)
    truthfail: dict[str, int] = field(default_factory=dict)
    decline: dict[str, int] = field(default_factory=dict)
    plays_n: int = 0
    declines_n: int = 0
    trend: float = 0.0
    trend_n: int = 0

    def record(self, op: str, klass: str, paid: bool, reward: float):
        """Fold one inferred outcome into the tallies."""
        self.plays[op] = self.plays.get(op, 0) + 1
        self.plays_n += 1
        if klass == "usable":
            self.usable[op] = self.usable.get(op, 0) + 1
        if paid:
            self.pay[op] = self.pay.get(op, 0) + 1
        if klass == "truth":
            self.truthfail[op] = self.truthfail.get(op, 0) + 1
        if klass == "declined":
            self.decline[op] = self.decline.get(op, 0) + 1
            self.declines_n += 1
        self.trend_n += 1
        self.trend += (reward - self.trend) / self.trend_n

    def feats(self, op: str) -> dict[str, float]:
        """Render this op's tallies as ``h:<op>:<stat>`` features."""
        n = self.plays.get(op, 0)
        d = max(n, 1)
        return {
            f"h:{op}:n": math.log1p(n) / 3.0,
            f"h:{op}:usable": self.usable.get(op, 0) / d,
            f"h:{op}:pay": self.pay.get(op, 0) / d,
            f"h:{op}:truthfail": self.truthfail.get(op, 0) / d,
            f"h:{op}:decline": self.decline.get(op, 0) / d,
        }


# ---------------------------------------------------------------------------
#  Featurization — scalar summaries of the state plus the action
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    """Per-state lookups the per-action features read.

    ``spec_cases`` maps each canonical subterm spec to the number of
    working cases whose term contains it — the "would this fire on a
    real workload" signal.  ``roots`` is the whole-case subset.
    ``premise_sz`` sizes compose's head-premise LHS patterns.
    The three caches memoize the repeated sub-blocks — fold and lift
    enumerate the same specs across kernels/carriers.
    """

    spec_cases: dict
    roots: set
    objects: dict
    proposals: dict
    premise_sz: dict
    pending_n: int
    spell_cache: dict = field(default_factory=dict)
    kern_cache: dict = field(default_factory=dict)
    carr_cache: dict = field(default_factory=dict)

    @classmethod
    def build(cls, state: ar.ArenaState) -> _Ctx:
        """Walk the working corpus once, indexing every subterm spec.

        The three caches are keyed by ``id(spec)``/small param keys —
        the enumerator reuses each canonical spec object across every
        kernel/carrier variant of a move, so the id is stable *for
        the lifetime of one call* (the ctx is rebuilt per step; the
        spec objects it references stay alive in ``acts``).
        """
        spec_cases: dict[str, int] = {}
        roots: set = set()
        for c in state.corpus:
            seen: set = set()
            for s, _n in ar._spec_walk(c.spec):
                seen.add(repr(ar._renumber_spec(s, {})))
            roots.add(repr(ar._renumber_spec(c.spec, {})))
            for k in seen:
                spec_cases[k] = spec_cases.get(k, 0) + 1
        return cls(
            spec_cases=spec_cases,
            roots=roots,
            objects={o.alpha_key: o for o in state.objects},
            proposals={p.alpha_key: p for p in state.proposals},
            premise_sz={
                n: ar._spec_size(lhs) / 8.0
                for n, lhs, _r in state.premise_forms
            },
            pending_n=len(state.pending),
        )


def _obj_counts(objs: tuple) -> tuple[int, int, int, int]:
    """Tally the stored objects' guard/verdict columns once."""
    usable = unguarded = truthfail = guarded = 0
    for o in objs:
        usable += bool(o.usable)
        unguarded += o.usable is False and not o.has_guard
        truthfail += o.failed == "truth"
        guarded += o.has_guard
    return usable, unguarded, truthfail, guarded


def _state_feats(state: ar.ArenaState, hist: _Hist) -> dict[str, float]:
    """Board-level summaries broadcast to every move's row."""
    objs = state.objects
    usable, unguarded, truthfail, guarded = _obj_counts(objs)
    return {
        "bias": 1.0,
        "st:progress": min(state.steps / 48.0, 1.0),
        "st:objects": len(objs) / 16.0,
        "st:usable": usable / 4.0,
        "st:unguarded": unguarded / 8.0,
        "st:truthfail": truthfail / 8.0,
        "st:condprop": (
            sum(
                1 for p in state.proposals if "conditional" in p.verdict
            )
            / 8.0
        ),
        "st:pending": len(state.pending) / 16.0,
        "st:trend": hist.trend,
        "st:decline": hist.declines_n / max(hist.plays_n, 1),
        "st:guarded": guarded / 8.0,
    }


def _op_feats(
    op: str, hist: _Hist, sf: dict[str, float]
) -> dict[str, float]:
    """Return the op block: one-hot, history rates, op-by-state terms.

    A broadcast state feature shifts every logit together and cancels
    in the softmax — state can only steer the policy through these
    product terms (``x:<op>:trend``, ``x:ingest:pending``, ...).
    """
    out = {
        f"op:{op}": 1.0,
        **hist.feats(op),
        f"x:{op}:trend": sf["st:trend"],
    }
    if op == "ingest":
        out["x:ingest:pending"] = sf["st:pending"]
    elif op == "auto_cond":
        out["x:auto_cond:unguarded"] = sf["st:unguarded"]
        out["x:auto_cond:condprop"] = sf["st:condprop"]
    elif op == "compose":
        out["x:compose:objects"] = sf["st:objects"]
    elif op == "relax_guard":
        out["x:relax_guard:guarded"] = sf["st:guarded"]
    elif op == "specialize":
        out["x:specialize:usable"] = sf["st:usable"]
    return out


def _spell_feats(ctx: _Ctx, spec: Any) -> dict[str, float]:
    """Return the corpus-coverage scalars a spelled spec carries.

    Memoized on ``id(spec)`` — the enumerator shares each canonical
    spec object across all kernel/carrier variants of a move, so one
    corpus-coverage computation serves the ~120 rows spelling it.
    """
    out = ctx.spell_cache.get(id(spec))
    if out is None:
        rep = repr(spec)
        out = {
            "a:spec_sz": ar._spec_size(spec) / 8.0,
            "a:mvars": len(ar._spec_metavars(spec)) / 4.0,
            "a:whole": float(rep in ctx.roots),
            "a:cases": ctx.spec_cases.get(rep, 0) / 4.0,
        }
        ctx.spell_cache[id(spec)] = out
    return out


def _kern_feats(ctx: _Ctx, k: Any) -> dict[str, float]:
    """Return a ``fold`` kernel's block — unary vs all-metavar form."""
    kk = ("u", k) if isinstance(k, str) else ("f", k[0], len(k) - 1)
    out = ctx.kern_cache.get(kk)
    if out is None:
        kop = k if kk[0] == "u" else k[0]
        out = {
            "a:k_unary": float(kk[0] == "u"),
            "a:k_full": float(kk[0] == "f"),
            "a:k_arity": kk[2] / 4.0 if kk[0] == "f" else 0.25,
            f"a:k:{_bucket(kop, 'kernel'):02d}": 1.0,
        }
        ctx.kern_cache[kk] = out
    return out


def _fold_feats(ctx: _Ctx, p: dict) -> dict[str, float]:
    """``fold`` params: spelled coverage + kernel identity buckets."""
    return {
        **_spell_feats(ctx, p["spelled"]),
        **_kern_feats(ctx, p["kernel"]),
    }


def _carr_feats(ctx: _Ctx, cop: Any, app: str, stateful: bool) -> dict:
    """Return a ``lift`` carrier's block — stateful + identity."""
    key = (cop, app, stateful)
    out = ctx.carr_cache.get(key)
    if out is None:
        out = {
            "a:c_stateful": float(stateful),
            f"a:c:{_bucket(cop, 'carrier')}": 1.0,
            f"a:ap:{_bucket(app, 'apply')}": 1.0,
        }
        ctx.carr_cache[key] = out
    return out


def _lift_feats(ctx: _Ctx, p: dict) -> dict[str, float]:
    """``lift`` params: step coverage + carrier/apply identity."""
    carrier = p["carrier"]
    cop = carrier[0] if carrier else ""
    return {
        **_spell_feats(ctx, p["step"]),
        **_carr_feats(
            ctx, cop, p["apply_op"], p.get("state") is not None
        ),
    }


def _compose_feats(ctx: _Ctx, p: dict) -> dict[str, float]:
    """``compose`` params: premise identity + specialization flags."""
    first = p["first"]
    rest = tuple(p.get("rest") or ())
    out = {
        "a:spec_sz": ctx.premise_sz.get(first, 0.0),
        "a:compose_spec": float(bool(p.get("specialize"))),
        "a:rest": len(rest) / 2.0,
        "a:self": float(rest[:1] == (first,)),
        f"a:p1:{_bucket(first, 'premise'):02d}": 1.0,
    }
    if rest:
        out[f"a:p2:{_bucket(rest[0], 'premise'):02d}"] = 1.0
    return out


def _target_feats(ctx: _Ctx, a: ar.Action) -> dict[str, float]:
    """Return the referenced object's guard/verdict state (``t:*``).

    Object-targeting ops (``auto_cond``/``relax_guard``/
    ``specialize``) ref a stored alpha key; ``auto_cond`` may also
    name a proposal verdict key (the conditional candidates that
    were never stored).  ``a:rescue`` is the "guard the
    conditionals" signature the fixed playbook ends with — an
    ``auto_cond`` on a truth-failed, check-dropped or
    conditional-verdicted target.
    """
    ref = a.params.get("ref")
    o = ctx.objects.get(ref) if isinstance(ref, str) else None
    prop = ctx.proposals.get(ref) if isinstance(ref, str) else None
    out: dict[str, float] = {}
    if o is not None:
        check_dropped = o.missing_hooks == ("check",)
        out.update(
            {
                "t:known": 1.0,
                "t:usable": float(bool(o.usable)),
                "t:guard": float(o.has_guard),
                "t:f_truth": float(o.failed == "truth"),
                "t:f_fulldata": float(
                    o.failed == "full-data" and check_dropped
                ),
                "t:f_typedpay": float(o.failed == "typed-pay"),
                "t:f_novelty": float(o.failed == "novelty"),
                "t:fires": o.fires / 4.0,
                "t:paid": o.paid / 4.0,
                "t:cleared": o.cleared / 8.0,
            }
        )
    if prop is not None and "conditional" in prop.verdict:
        out["t:conditional"] = 1.0
    if a.op == "auto_cond" and (
        out.get("t:f_truth")
        or out.get("t:f_fulldata")
        or out.get("t:conditional")
    ):
        out["a:rescue"] = 1.0
    return out


def _act_feats(ctx: _Ctx, a: ar.Action) -> dict[str, float]:
    """Return the action-specific feature row (dispatch on the op)."""
    if a.op == "ingest":
        names = a.params.get("names", ())
        return {
            "a:ingest_frac": len(tuple(names)) / max(ctx.pending_n, 1)
        }
    if a.op == "fold":
        return _fold_feats(ctx, a.params)
    if a.op == "lift":
        return _lift_feats(ctx, a.params)
    if a.op == "compose":
        return _compose_feats(ctx, a.params)
    return _target_feats(ctx, a)


def featurize(
    state: ar.ArenaState, action: ar.Action, hist: _Hist | None = None
) -> dict[str, float]:
    """Return the full feature row for one legal move — public for tests.

    The dict's keys are a subset of :data:`lawdata.ARENA_FEATURES`;
    scoring is ``Σ w[name]·value``.  *hist* defaults to an empty
    episode history.
    """
    ctx = _Ctx.build(state)
    hist = _Hist() if hist is None else hist
    sf = _state_feats(state, hist)
    return {
        **sf,
        **_op_feats(action.op, hist, sf),
        **_act_feats(ctx, action),
    }


# ---------------------------------------------------------------------------
#  The player
# ---------------------------------------------------------------------------


class LearnedPlayer:
    """A softmax linear policy over the live legal move set.

    ``score(move) = w·φ(state, move)``; a step samples from the
    softmax over the *unplayed* legal moves (``temperature`` scales
    the logits, ``greedy=True`` takes the argmax).  *weights* is a
    ``{feature_name: float}`` table over
    :data:`lawdata.ARENA_FEATURES` — ``None`` binds the shipped
    :data:`lawdata.ARENA_PLAYER_WEIGHTS` (empty ⇒ uniform, the
    honest cold start).  ``learn=False`` is the eval-time form: the
    player samples but never accumulates gradients — measured yield
    is then attributable to the trained weights, not in-run
    adaptation.  *legal* is the enumerator seam every baseline
    carries.
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
        max_candidates: int | None = None,
    ) -> None:
        """Bind the weight table, the sampler and the learn flag.

        *max_candidates* caps the move set the policy scores each
        step: when the live legal set exceeds the cap a uniform
        seeded slice is drawn — a perceptual limit, not a legality
        change (a player needn't enumerate the whole board, and the
        slice is re-drawn each step so coverage accrues over the
        episode).  ``None`` scores every enumerated move.
        """
        self._rng = random.Random(seed)
        self._seed = seed
        self._lr = lr
        self._temp = max(temperature, 1e-6)
        self._greedy = greedy
        self._learn = learn
        self._cap = max_candidates
        self._legal = ar.legal_actions if legal is None else legal
        self._w = dict(
            lawdata.ARENA_PLAYER_WEIGHTS if weights is None else weights
        )
        self._baseline = 0.0
        self._reset_episode()

    # -- episode bookkeeping ------------------------------------------------

    def _reset_episode(self) -> None:
        """Clear the episode-local state (called on ``steps == 0``)."""
        self._played: dict = {}
        self._hist = _Hist()
        self._ep: list = []
        self._last: ar.Action | None = None
        self._prev: ar.ArenaState | None = None

    def frozen(self, seed: int = 0, *, greedy: bool = False) -> Any:
        """Return a non-learning copy sharing the trained weights.

        The eval-time form: same policy, no gradient accumulation —
        the guide's ``frozen()`` idiom.
        """
        return LearnedPlayer(
            seed=seed,
            weights=self.weights_dict(),
            temperature=self._temp,
            greedy=greedy,
            learn=False,
            legal=self._legal,
            max_candidates=self._cap,
        )

    def weights_dict(self) -> dict[str, float]:
        """Return the learned weight table — ``{name: w}`` over the schema."""
        return {n: self._w.get(n, 0.0) for n in lawdata.ARENA_FEATURES}

    # -- the player's move ---------------------------------------------------

    def _observe(self, state: ar.ArenaState) -> None:
        """Update episode-local state from the new observation.

        Episode boundaries are detected by the step counter: a fresh
        board (``steps == 0``) after a progressed one resets the
        episode-local state.  A repeated call on the *same*
        observation (``steps`` unchanged — tests walk the frontier
        without stepping) is neither a boundary nor an outcome: no
        reset, and no phantom tally for the un-played move.
        """
        if self._prev is None:
            pass  # construction already reset the episode state
        elif state.steps == 0 and self._prev.steps != 0:
            self._reset_episode()
        elif self._last is not None and state.steps > self._prev.steps:
            klass, reward, paid = _infer_outcome(
                self._prev, self._last, state
            )
            self._hist.record(self._last.op, klass, paid, reward)
        self._prev = state

    def _slice(self, acts: list) -> list:
        """The candidate set this step scores — the whole legal set,
        or a uniform seeded slice of it under the cap."""
        if self._cap is not None and 0 < self._cap < len(acts):
            return self._rng.sample(acts, self._cap)
        return acts

    def _sample(self, weights: list[float]) -> int:
        """Draw an index proportionally to the (unscaled) weights."""
        r = self._rng.random() * sum(weights)
        acc = 0.0
        for i, w in enumerate(weights):
            acc += w
            if r <= acc:
                return i
        return len(weights) - 1

    def __call__(self, state: ar.ArenaState) -> ar.Action | None:
        """Score the legal moves, sample an unplayed one, remember it.

        Returns ``None`` when nothing unplayed remains — the same
        drain semantics as the frontier baselines.  The played mask
        is applied by *rejection* on the sampled index (a candidate's
        key is only hashed when drawn) and by a mass correction in
        the gradient's expectation — on the ~50k-move board this is
        ~6x cheaper than keying every enumerated move each step.
        """
        self._observe(state)
        acts = list(self._legal(state))
        if not acts:
            self._last = None
            return None
        acts = self._slice(acts)
        ctx = _Ctx.build(state)
        sf = _state_feats(state, self._hist)
        sf_dot = _dot(self._w, sf)
        of_cache: dict[str, dict] = {}
        afs: list[dict] = []
        logits: list[float] = []
        for a in acts:
            of = of_cache.get(a.op)
            if of is None:
                of = _op_feats(a.op, self._hist, sf)
                of_cache[a.op] = of
            af = _act_feats(ctx, a)
            afs.append(af)
            logits.append(
                (sf_dot + _dot(self._w, of) + _dot(self._w, af))
                / self._temp
            )
        top = max(logits)
        exps = [math.exp(x - top) for x in logits]
        # the played moves' (recomputed) softmax mass — subtracted
        # from the unmasked expectation for the masked policy's grad
        m_exp, played_corr = self._played_mass(
            state, ctx, sf, of_cache, sf_dot, top
        )
        z = sum(exps) - m_exp
        if z <= 0.0:
            self._last = None
            return None  # every enumerated move was already played
        i = self._pick(acts, logits, exps)
        if i is None:
            self._last = None
            return None
        if self._learn:
            self._ep.append(
                _grad_row(
                    sf, of_cache, acts, afs, exps, z, played_corr, i
                )
            )
        chosen = acts[i]
        self._played[_move_key(chosen)] = chosen
        self._last = chosen
        return chosen

    def _played_mass(
        self,
        state: ar.ArenaState,
        ctx: _Ctx,
        sf: dict,
        of_cache: dict,
        sf_dot: float,
        top: float,
    ) -> tuple[float, dict]:
        """``(mass, φ·mass)`` of the played-and-still-legal moves.

        A played move stays legal on this board (re-declaring
        re-gauntlets), so its softmax mass is recomputed under the
        current context and subtracted — the masked policy's
        ``E[φ]`` normalizes over the *unplayed* mass.  ``ingest``
        moves die when their names leave the pending pool (checked
        here); target-class moves can vanish when a guard lands —
        treated as live, a ≤played/50k-mass approximation on the
        gradient, never on the sampled play.
        """
        w = self._w
        m_exp = 0.0
        corr: dict[str, float] = {}
        pend = set(state.pending)
        for b in self._played.values():
            if b.op == "ingest":
                names = set(b.params.get("names") or ())
                if not names <= pend:
                    continue
            of = of_cache.get(b.op)
            if of is None:
                of = _op_feats(b.op, self._hist, sf)
                of_cache[b.op] = of
            af = _act_feats(ctx, b)
            sb = (sf_dot + _dot(w, of) + _dot(w, af)) / self._temp
            eb = math.exp(sb - top)
            m_exp += eb
            for k, v in {**sf, **of, **af}.items():
                corr[k] = corr.get(k, 0.0) + v * eb
        return m_exp, corr

    def _pick(self, acts: list, logits: list, exps: list) -> int | None:
        """Choose an index: argmax or rejection-sampled, unplayed."""
        if self._greedy:
            order = sorted(
                range(len(acts)), key=lambda j: logits[j], reverse=True
            )
            for j in order:
                if _move_key(acts[j]) not in self._played:
                    return j
            return None
        for _try in range(64):
            i = self._sample(exps)
            if _move_key(acts[i]) not in self._played:
                return i
        # pathological board: nearly everything legal is played —
        # fall back to a keyed scan
        for j in range(len(acts)):
            if _move_key(acts[j]) not in self._played:
                return j
        return None

    def finish_episode(self, total: float) -> None:
        """REINFORCE on the episode return (``Trajectory.total``).

        ``w += lr·(R - baseline)·mean_t(φ_t - E_π[φ_t])``; the
        running-mean baseline updates from the raw return, the
        advantage is clipped so one paying episode cannot blow the
        weights up.  No-op when ``learn=False`` — a frozen player
        accumulates no gradient rows anyway.
        """
        adv = total - self._baseline
        self._baseline += _BASELINE_EMA * adv
        if self._learn and self._ep:
            adv = max(-_CLIP, min(_CLIP, adv))
            scale = self._lr * adv / len(self._ep)
            grad: dict[str, float] = {}
            for phi, ephi in self._ep:
                for k, v in phi.items():
                    grad[k] = grad.get(k, 0.0) + v
                for k, v in ephi.items():
                    grad[k] = grad.get(k, 0.0) - v
            for k, g in grad.items():
                self._w[k] = self._w.get(k, 0.0) + scale * g
        self._ep = []


def _dot(w: dict, feats: dict) -> float:
    """Return the linear score ``w·feats`` over a sparse feature dict."""
    s = 0.0
    for k, v in feats.items():
        wv = w.get(k)
        if wv:
            s += wv * v
    return s


def _move_key(a: ar.Action) -> tuple:
    """Return a cheap stable key for the played mask.

    ``arena._action_key`` walks params into nested frozensets —
    ~5 s over the ~50k-move board, paid per candidate per step by
    the frontier baselines.  ``repr`` of the sorted items dedups
    identically (both give 50,947 keys on the cheap board) at ~6x
    less cost — and this player only pays it for *sampled*
    candidates, not the whole enumeration.
    """
    return (a.op, repr(sorted(a.params.items(), key=lambda kv: kv[0])))


def _grad_row(
    sf: dict,
    of_cache: dict,
    acts: list,
    afs: list,
    exps: list,
    z: float,
    played_corr: dict,
    i: int,
) -> tuple[dict, dict]:
    """``(φ_chosen, E_π[φ])`` under the *masked* softmax.

    ``E_π[φ]`` decomposes exactly over the three feature blocks —
    the state row at full mass, each op row at its op's mass, each
    action row at the move's own mass — accumulated in unnormalized
    exp-weights, with the played moves' contribution (``played_corr``,
    recomputed this step) subtracted, all over the unplayed mass ``z``.
    """
    esum: dict[str, float] = {}
    op_esum: dict[str, float] = {}
    tot = 0.0
    for a, e in zip(acts, exps, strict=True):
        tot += e
        op_esum[a.op] = op_esum.get(a.op, 0.0) + e
    for k, v in sf.items():
        esum[k] = tot * v
    for op, feats in of_cache.items():
        m = op_esum.get(op, 0.0)
        if not m:
            continue
        for k, v in feats.items():
            esum[k] = esum.get(k, 0.0) + v * m
    for af, e in zip(afs, exps, strict=True):
        for k, v in af.items():
            esum[k] = esum.get(k, 0.0) + v * e
    for k, v in played_corr.items():
        esum[k] = esum.get(k, 0.0) - v
    ephi = {k: v / z for k, v in esum.items()}
    phi = {**sf, **of_cache[acts[i].op], **afs[i]}
    return phi, ephi


# ---------------------------------------------------------------------------
#  Training and evaluation — disjoint seed streams
# ---------------------------------------------------------------------------


def episode_seeds(
    seed: int, n_train: int, n_eval: int
) -> tuple[list[int], list[int]]:
    """Disjoint ``(train, eval)`` board seeds — the honest split.

    Training rolls ``seed + 1 + ep`` boards; eval boards come
    *after* the whole train range, so no eval board was ever a
    training board — the eval-arena-in-the-training-set mistake
    ``guide-real-run.md`` names, made structurally impossible.
    """
    train = [seed + 1 + i for i in range(n_train)]
    ev = [seed + 1 + n_train + i for i in range(n_eval)]
    return train, ev


def train_player(
    player: LearnedPlayer,
    seeds: Iterable[int],
    *,
    budget: int,
    arena_factory: Callable[[int], ar.Arena],
) -> list[float]:
    """Play one episode per seed; REINFORCE on each episode return.

    *arena_factory* builds a fresh board per seed — the cheap board
    is ``make_arena(seed, max_cases=…, max_holdout=…)``.  Returns the
    per-episode ``Trajectory.total`` sequence (the training curve).
    """
    totals: list[float] = []
    for s in seeds:
        arena = arena_factory(s)
        traj = ar.run_episode(arena, player, budget)
        player.finish_episode(traj.total)
        totals.append(traj.total)
    return totals


def evaluate(
    arms: dict[str, Callable[[int], Any]],
    seeds: Iterable[int],
    *,
    budget: int,
    arena_factory: Callable[[int], ar.Arena],
) -> dict[str, list[dict]]:
    """Play every arm on the same seed list; return the probe rows.

    ``arms`` maps a player name to a factory ``episode_index ->
    player`` (fresh per episode — the learned arm is the frozen
    trained policy).  Paired boards: every arm plays the same
    *seeds*, so means compare like for like.
    """
    out: dict[str, list[dict]] = {}
    for name, mk in arms.items():
        rows: list[dict] = []
        for i, s in enumerate(seeds):
            arena = arena_factory(s)
            traj = ar.run_episode(arena, mk(i), budget)
            rows.append(
                {
                    "episode": i,
                    "seed": s,
                    "steps": len(traj.reports),
                    "usable": traj.usable,
                    "holdout_fires": traj.holdout_fires,
                    "holdout_paid": traj.holdout_paid,
                    "reward": round(traj.total, 3),
                    "failed": ar.stage_failures(traj),
                }
            )
        out[name] = rows
    return out


def _fmt_eval(table: dict) -> str:
    """Render per-arm means over the eval rows."""
    head = (
        f"{'player':<15} {'eps':>4} {'usable':>7} {'fires':>7} "
        f"{'paid':>6} {'reward':>9} {'steps':>6}"
    )
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        n = max(len(rows), 1)
        lines.append(
            f"{name:<15} {len(rows):>4} "
            f"{sum(r['usable'] for r in rows) / n:>7.2f} "
            f"{sum(r['holdout_fires'] for r in rows) / n:>7.2f} "
            f"{sum(r['holdout_paid'] for r in rows) / n:>6.2f} "
            f"{sum(r['reward'] for r in rows) / n:>9.2f} "
            f"{sum(r['steps'] for r in rows) / n:>6.1f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Train the learned player, then run the four-arm comparison.

    Training boards are the FIXED-corpus cheap board at seeds
    ``seed+1..seed+train``; eval boards are the same board shape at
    seeds strictly past the train range — every arm plays each eval
    seed once.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-episodes", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--max-holdout", type=int, default=12)
    parser.add_argument("--lr", type=float, default=_LR)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--candidates",
        type=int,
        default=lawdata.ARENA_PLAYER_CANDIDATES,
        help="cap the legal moves scored per step (0 = uncapped)",
    )
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    def factory(s: int) -> ar.Arena:
        return ar.make_arena(
            seed=s,
            max_cases=args.max_cases,
            max_holdout=args.max_holdout,
            meta={"code_rev": "player"},
        )

    train_seeds, eval_seeds = episode_seeds(
        args.seed, args.train_episodes, args.episodes
    )
    player = LearnedPlayer(
        seed=args.seed,
        lr=args.lr,
        temperature=args.temperature,
        max_candidates=args.candidates,
    )
    totals = train_player(
        player, train_seeds, budget=args.budget, arena_factory=factory
    )
    if totals:
        dec = max(len(totals) // 10, 1)
        curve = [
            round(
                sum(totals[i : i + dec]) / len(totals[i : i + dec]), 2
            )
            for i in range(0, len(totals), dec)
        ]
        print(f"train curve (decade means): {curve}")  # stdout-compat
    top = sorted(
        player.weights_dict().items(), key=lambda kv: -abs(kv[1])
    )[:12]
    print(  # stdout-compat
        "top weights:", [(k, round(v, 3)) for k, v in top]
    )

    cold = LearnedPlayer(seed=args.seed, learn=False)
    arms: dict[str, Callable[[int], Any]] = {
        "fixed": lambda _e: ar.FixedRule(ar._probe_playbook()),
        "random": lambda e: ar.RandomPlayer(
            random.Random(50_000 + args.seed * 100 + e)
        ),
        "heuristic": lambda _e: ar.HeuristicPlayer(),
        "learned-cold": lambda e: cold.frozen(
            seed=70_000 + args.seed * 100 + e
        ),
        "learned": lambda e: player.frozen(
            seed=60_000 + args.seed * 100 + e
        ),
        "learned-greedy": lambda e: player.frozen(
            seed=60_000 + args.seed * 100 + e, greedy=True
        ),
    }
    table = evaluate(
        arms, eval_seeds, budget=args.budget, arena_factory=factory
    )
    print("\n== arena learned-player comparison ==")  # stdout-compat
    print(_fmt_eval(table))  # stdout-compat
    totals_fail = ar.probe_failure_totals(table)
    print("\nstage failures per player:")  # stdout-compat
    for name, stages in totals_fail.items():
        if stages:
            print(f"  {name}: {stages}")  # stdout-compat
    if args.json:
        from pathlib import Path

        blob = {"table": table, "train_totals": totals}
        Path(args.json).write_text(json.dumps(blob, indent=2) + "\n")
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
