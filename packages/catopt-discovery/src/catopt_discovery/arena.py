"""The construction arena — a game over structured object operations.

Plan 0020 (``project/plans/0020-construction-arena.md``): the
scheduling game measured shallow — the trained guide converged to
the fixed rule it cannot beat (``project/retros/guide-real-run.md``).
The space that is actually deep — *which objects to construct,
compose and guard* — is this game's action space.  Where
:mod:`catopt_discovery.meta_game` hands the player generator arms to
schedule, the arena hands it the ADR-0004 construction operations
themselves: the player builds declared objects, and the admission
gauntlet referees them.

The board
---------

An :class:`Arena` binds four things:

* a **working corpus** (``TermCase``s) — the real terms the player
  acts on: its terms feed the gauntlet's match enumeration, real
  guarded-region sweep and numeric-truth instance;
* a **holdout probe** (``TermCase``s) — the firing/typed-pay probe
  the reward reads.  The player cannot write into it: no action adds
  or removes a holdout case, so ``fires``/``paid`` are measured on
  workloads the player never shaped — the zoo-discipline of "pay on
  seeded spellings does not count";
* the **evidence store** — constructed objects persist through
  ``object_synthesis.store_constructed`` and face
  ``evidence.run_gauntlet`` unchanged (the player never decides
  equivalence — ADR 0003);
* a **pending pool** — corpus material the ``ingest`` action can
  pull into the working split (the corpus-direction move).

:func:`split_corpus` makes the split honest: the holdout is drawn
only from the *real* workload set — never the intake's
purpose-built spellings (``intake.Workload.purpose_built`` is the
ledger; :func:`purpose_built_names` reads it).

The moves
---------

An :class:`Action` is data: an ``op`` name plus a params map.  The
shipped vocabulary is the classmethod set — ``fold`` /
``lift`` / ``compose`` / ``auto_cond`` / ``ingest`` — each a thin
wrapper over ``object_synthesis.*`` (and ``evidence``'s store +
gauntlet).  The dispatch is a *registry* (:data:`ACTIONS`,
:func:`register_action`), not a match statement: the sibling's
``relax_guard`` / ``specialize`` constructors plug in without
touching this module — a new ``op`` name plus a
``(arena, action) -> _Outcome`` handler is the whole contract.

The referee
---------

:meth:`Arena.step` applies the construction, stores the result as a
declared object, and runs the eight-stage gauntlet on a corpus
whose ``real_terms`` are the working split and whose ``probe`` is
the holdout.  The :class:`StepReport` returns what the game needs:

* ``stages_cleared`` — the dense partial-credit signal (each gate
  passed is progress: reconstruct → full-data → measure → truth →
  novelty → typed-pay → closure → cert);
* ``usable`` — the gauntlet's honest verdict;
* ``holdout_fires`` / ``holdout_paid`` — measured on the holdout
  probe only.

The reward is the documented composition
``stages_cleared + 2·usable + 0.2·fires + 2·paid``; a refused
construction (malformed spec, unknown premise, compose declined,
auto-cond found no cover) applies ``False`` and scores 0.

The player
----------

A player is any callable ``ArenaState -> Action | None``.
:class:`ArenaState` is the read-only, serializable board view —
working corpus cases, stored objects with their verdicts and
guard-region counts, the proposal-verdict table, pending ingest
names, premise names and guard-fire counts.  It never carries a
holdout term: the player sees the reward *counts* (that is the
signal) but never the workloads they were measured on.

:func:`run_episode` drives a callable until the action budget is
spent; :class:`FixedRule` (an authored playbook plus the
"guard the conditionals" rule the real run measured as the missing
move), :class:`RandomPlayer` (uniform over the live legal set) and
:class:`GreedyPlayer` (the largest immediate-stage estimate) are the
immediately-playable baselines.

The move set
------------

``legal_actions(state)`` enumerates the construction moves a state
affords — the substrate a real player needs (a playbook names moves;
a policy must *see* them).  Every parameter is attested by the
state: corpus composite spellings for ``fold``/``lift``, declared
carrier templates for ``lift``, premise patterns for ``compose``
(emitted only when the firing premise matches inside the head
premise's RHS, bare or after a metavar specialization), unguarded or
conditional-measured objects for ``auto_cond``, the pending pool for
``ingest``.  The list is in deterministic enumeration order and
unique by construction — a pure function of the observation.
Legality is *constructibility*; truth is the gauntlet's question,
not the enumerator's.

:func:`depth_probe` is the plan's phase-2 experiment: the fixed
playbook, uniform-over-legal random and the greedy stage-maximizer
played on ``make_arena`` — does move *choice* matter at all on the
construction board?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, overload

from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op, op_repr, op_repr_dag
from catopt_core.meta import _positions, match_pattern

from catopt_discovery import evidence as ev
from catopt_discovery import lawdata
from catopt_discovery import object_synthesis as obs
from catopt_discovery.census import CorpusTerm, op_tuple_census
from catopt_discovery.impact import TermCase

__all__ = [
    "ACTIONS",
    "Action",
    "Arena",
    "ArenaState",
    "CaseView",
    "FixedRule",
    "GreedyPlayer",
    "ObjectView",
    "ProposalView",
    "RandomPlayer",
    "StageLine",
    "StepReport",
    "Trajectory",
    "depth_probe",
    "legal_actions",
    "main",
    "make_arena",
    "purpose_built_names",
    "register_action",
    "run_episode",
    "split_corpus",
]

# ---------------------------------------------------------------------------
#  Actions — the construction operations as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One construction move — a named operation plus its arguments.

    ``op`` selects the registered handler (see :data:`ACTIONS` and
    :func:`register_action`); ``params`` is its argument map —
    ``object_synthesis.term_from_spec`` term specs, object references
    (a shipped rule name, a stored object's name or alpha key, or a
    live ``Rewrite``/``ConstructedObject``) and plain values.  The
    classmethods are the shipped vocabulary; ``Action("fold",
    {...})`` spelled out is equivalent, so new constructors need no
    new syntax.
    """

    op: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fold(
        cls,
        spelled: Any,
        kernel: Any,
        *,
        name: str | None = None,
        arg: str = "X",
        cond: Any = None,
        dspec: Any = None,
        kind: str = "abstraction",
        tags: tuple = (),
    ) -> Action:
        """``fold(spelled, kernel)`` — the spelled form → its kernel.

        ``object_synthesis.fold_object``: ``div(x, |x|+1) →
        softsign(x)`` spelled ``(("div","X",("add",("abs","X"),1)),
        "softsign")``.  *kernel* may be an op name (unary fold over
        *arg*) or a full term spec.
        """
        return cls(
            "fold",
            {
                "name": name,
                "spelled": spelled,
                "kernel": kernel,
                "arg": arg,
                "cond": cond,
                "dspec": dspec,
                "kind": kind,
                "tags": tuple(tags),
            },
        )

    @classmethod
    def lift(
        cls,
        step: Any,
        carrier: Any,
        apply_op: str,
        *,
        state: Any = None,
        name: str | None = None,
        cond: Any = None,
        dspec: Any = None,
        kind: str = "abstraction",
        tags: tuple = (),
    ) -> Action:
        """``lift(step, carrier, apply_op)`` — a program step → carrier.

        ``object_synthesis.lift_object``: the affine scan lift is
        ``lift(("add",("matmul","A","h"),"x"), ("aff","A","x"),
        "apply", state="h")``.
        """
        return cls(
            "lift",
            {
                "name": name,
                "step": step,
                "carrier": carrier,
                "apply_op": apply_op,
                "state": state,
                "cond": cond,
                "dspec": dspec,
                "kind": kind,
                "tags": tuple(tags),
            },
        )

    @classmethod
    def compose(
        cls,
        first: Any,
        *rest: Any,
        name: str | None = None,
        specialize: dict | None = None,
        cond: Any = None,
        dspec: Any = None,
        kind: str = "abstraction",
        tags: tuple = (),
        env: dict | None = None,
    ) -> Action:
        """``compose(a, b, ...)`` — rewrite a premise's RHS by the rest.

        ``object_synthesis.compose_objects`` with guard transport.
        *first* / *rest* are object references — shipped rule names,
        stored object names or alpha keys.  *env* is the
        metavariable→concrete-leaf instance a derivation certificate
        materializes on (see ``object_synthesis.store_constructed``).
        """
        return cls(
            "compose",
            {
                "name": name,
                "first": first,
                "rest": tuple(rest),
                "specialize": dict(specialize or {}),
                "cond": cond,
                "dspec": dspec,
                "kind": kind,
                "tags": tuple(tags),
                "env": dict(env or {}),
            },
        )

    @classmethod
    def auto_cond(
        cls,
        ref: Any,
        *,
        name: str | None = None,
        max_clauses: int = 3,
        synth_limit: int = 360,
        kind: str = "abstraction",
    ) -> Action:
        """``auto_cond(obj)`` — mint the guard a conditional needs.

        ``object_synthesis.auto_cond_object`` over the *working*
        corpus's measured domain: the smallest declarative ``cond``
        covering every measured equal site and declining every bad
        one.  *ref* is an object reference (stored name or alpha
        key — the conditionals the real run surfaced live in the
        store, not in the working corpus).
        """
        return cls(
            "auto_cond",
            {
                "name": name,
                "ref": ref,
                "max_clauses": max_clauses,
                "synth_limit": synth_limit,
                "kind": kind,
            },
        )

    @classmethod
    def ingest(cls, names: Iterable[str]) -> Action:
        """``ingest(subset)`` — pull named pending cases into working.

        The corpus-direction move: the named cases leave the pending
        pool and join the working split, rotating the evidence scope
        (every later measurement is attributable to the grown
        corpus).  The holdout is untouched — it is not in the pool.
        """
        return cls("ingest", {"names": tuple(names)})

    @classmethod
    def relax_guard(
        cls, ref: Any, clause: Any, *, name: str | None = None
    ) -> Action:
        """``relax_guard(obj, clause)`` — weaken a guarded object.

        ``object_synthesis.relax_guard``: drop one declarative
        ``cond`` conjunct — *clause* is an index into the flattened
        conjunction (``ObjectView.cond_clauses`` order) or the clause
        datum itself.  The relaxed region is a superset of the
        guarded one; the sweep re-measures it and the gauntlet
        decides — relaxing can only grow a region, never prove it.
        """
        return cls(
            "relax_guard",
            {"name": name, "ref": ref, "clause": clause},
        )

    @classmethod
    def specialize(
        cls, ref: Any, binding: dict, *, name: str | None = None
    ) -> Action:
        """``specialize(obj, binding)`` — narrow to a concrete family.

        ``object_synthesis.specialize``: *binding* maps a leaf
        metavariable to a term spec (``{"S": 4.0}`` inlines a
        constant) or an attr metavariable to a concrete value
        (``{"D": 0}`` pins the axis — the pin rides the ``cond``,
        keeping sibling clauses live).  A strict narrowing the
        gauntlet re-referees.
        """
        return cls(
            "specialize",
            {"name": name, "ref": ref, "binding": dict(binding)},
        )


#: Handler contract — ``(arena, action) -> _Outcome``.  A handler
#: constructs (or declines) and reports; ``Arena.step`` owns the
#: storing, the gauntlet and the accounting.  This is the plug-in
#: seam: ``relax_guard`` / ``specialize`` register here.
ActionHandler = Callable[["Arena", Action], "_Outcome"]

#: The action registry — ``op`` name → handler.  Ships the five
#: plan-0020 constructors; later operations plug in via
#: :func:`register_action`.
ACTIONS: dict[str, ActionHandler] = {}


@overload
def register_action(
    op: str,
) -> Callable[[ActionHandler], ActionHandler]: ...


@overload
def register_action(op: str, fn: ActionHandler) -> ActionHandler: ...


def register_action(
    op: str, fn: ActionHandler | None = None
) -> ActionHandler | Callable[[ActionHandler], ActionHandler]:
    """Register *fn* as the constructor behind ``Action(op)``.

    Usable either way — ``register_action("fold", fn)`` or as a
    decorator ``@register_action("relax_guard")`` — so a new
    construction operation is one handler function, no edits here.
    """

    def _bind(f: ActionHandler) -> ActionHandler:
        ACTIONS[op] = f
        return f

    return _bind(fn) if fn is not None else _bind


@dataclass(frozen=True)
class _Outcome:
    """What an action handler left on the board.

    ``obj`` is a :class:`obs.ConstructedObject` to store and referee;
    ``key`` an already-stored alpha key to re-gauntlet; ``env`` the
    instance a derivation certificate materializes on.
    ``applied=False`` is the honest refusal — construction declined,
    nothing stored.  ``ingested`` counts corpus cases the action
    moved.  ``detail`` carries constructor diagnostics (the
    auto-cond domain counts) into the report.
    """

    obj: Any = None
    key: str = ""
    env: dict | None = None
    applied: bool = True
    note: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    ingested: int = 0


def _action_name(action: Action, prefix: str) -> str:
    """Return the declared object name for *action*.

    ``params["name"]`` when the player named its construction;
    otherwise a deterministic digest of the action's params — the
    object's provenance lives in ``construction``/``derivation``,
    the name is display, and collisions share an alpha key anyway.
    """
    given = action.params.get("name")
    if given:
        return str(given)
    digest = hashlib.sha256(
        repr(
            sorted(action.params.items(), key=lambda kv: kv[0])
        ).encode("utf-8")
    ).hexdigest()[:10]
    return f"{prefix}:{digest}"


@register_action("fold")
def _fold(arena: Arena, action: Action) -> _Outcome:
    """Apply ``object_synthesis.fold_object`` to the action's spec."""
    p = action.params
    try:
        obj = obs.fold_object(
            _action_name(action, "fold"),
            p["spelled"],
            p["kernel"],
            arg=p.get("arg", "X"),
            cond=p.get("cond"),
            dspec=p.get("dspec"),
            kind=p.get("kind", "abstraction"),
            tags=tuple(p.get("tags", ())),
        )
    except Exception as exc:
        return _Outcome(
            applied=False,
            note=f"fold spec malformed: {type(exc).__name__}: {exc}",
        )
    return _Outcome(obj=obj, note=f"fold → {obj.rule.name}")


@register_action("lift")
def _lift(arena: Arena, action: Action) -> _Outcome:
    """Apply ``object_synthesis.lift_object`` to the action's spec."""
    p = action.params
    try:
        obj = obs.lift_object(
            _action_name(action, "lift"),
            p["step"],
            p["carrier"],
            p["apply_op"],
            state=p.get("state"),
            cond=p.get("cond"),
            dspec=p.get("dspec"),
            kind=p.get("kind", "abstraction"),
            tags=tuple(p.get("tags", ())),
        )
    except Exception as exc:
        return _Outcome(
            applied=False,
            note=f"lift spec malformed: {type(exc).__name__}: {exc}",
        )
    return _Outcome(obj=obj, note=f"lift → {obj.rule.name}")


@register_action("compose")
def _compose(arena: Arena, action: Action) -> _Outcome:
    """Apply ``object_synthesis.compose_objects`` over resolved refs."""
    p = action.params
    rules: list[Any] = []
    for ref in (p["first"], *p.get("rest", ())):
        rule = arena.resolve_ref(ref)
        if rule is None:
            return _Outcome(
                applied=False, note=f"unknown premise {ref!r}"
            )
        rules.append(rule)
    try:
        obj = obs.compose_objects(
            _action_name(action, "compose"),
            *rules,
            specialize=dict(p.get("specialize", {})),
            cond=p.get("cond"),
            dspec=p.get("dspec"),
            kind=p.get("kind", "abstraction"),
            tags=tuple(p.get("tags", ())),
        )
    except Exception as exc:
        return _Outcome(
            applied=False,
            note=f"compose raised: {type(exc).__name__}: {exc}",
        )
    if obj is None:
        return _Outcome(
            applied=False,
            note="compose declined: a premise matched nowhere",
        )
    return _Outcome(
        obj=obj,
        env=p.get("env") or None,
        note=f"compose → {obj.rule.name}",
    )


@register_action("auto_cond")
def _auto_cond(arena: Arena, action: Action) -> _Outcome:
    """Mint the declarative guard for the referenced object."""
    p = action.params
    rule = arena.resolve_ref(p["ref"])
    if rule is None:
        return _Outcome(
            applied=False, note=f"unknown object {p['ref']!r}"
        )
    res = obs.auto_cond_object(
        rule,
        corpus_terms=[c.term for c in arena.working],
        synth_limit=int(p.get("synth_limit", 360)),
        max_clauses=int(p.get("max_clauses", 3)),
        name=p.get("name"),
        kind=p.get("kind", "abstraction"),
    )
    detail = {
        "measured": res.measured,
        "equal": res.equal,
        "bad": res.bad,
        "unstable": res.unstable,
        "other": res.other,
        "declined": res.declined,
        "accepted": res.accepted,
    }
    if res.object is None:
        return _Outcome(
            applied=False,
            note=res.detail or "auto-cond refused",
            detail=detail,
        )
    return _Outcome(obj=res.object, note=res.detail, detail=detail)


@register_action("relax_guard")
def _relax(arena: Arena, action: Action) -> _Outcome:
    """Drop one declarative clause of the referenced object's cond."""
    p = action.params
    rule = arena.resolve_ref(p["ref"])
    if rule is None:
        return _Outcome(
            applied=False, note=f"unknown object {p['ref']!r}"
        )
    obj = obs.relax_guard(rule, p["clause"], name=p.get("name"))
    if obj is None:
        return _Outcome(
            applied=False,
            note="relax declined (no declarative clause there)",
        )
    return _Outcome(
        obj=obj, note=f"relax_guard → {obj.rule.name} (region widened)"
    )


@register_action("specialize")
def _specialize(arena: Arena, action: Action) -> _Outcome:
    """Bind a metavariable of the referenced object to a value."""
    p = action.params
    rule = arena.resolve_ref(p["ref"])
    if rule is None:
        return _Outcome(
            applied=False, note=f"unknown object {p['ref']!r}"
        )
    obj = obs.specialize(
        rule, dict(p.get("binding", {})), name=p.get("name")
    )
    if obj is None:
        return _Outcome(
            applied=False,
            note="specialize declined (binding names nothing)",
        )
    return _Outcome(
        obj=obj, note=f"specialize → {obj.rule.name} (narrowed)"
    )


@register_action("ingest")
def _ingest(arena: Arena, action: Action) -> _Outcome:
    """Move the named pending cases into the working split."""
    names = tuple(action.params.get("names", ()))
    moved, missing = arena.ingest(names)
    note = f"ingested {moved}"
    if missing:
        note += f"; unknown pending names: {sorted(missing)}"
    return _Outcome(applied=bool(moved), ingested=moved, note=note)


# ---------------------------------------------------------------------------
#  The read-only state — a serializable snapshot of the board
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseView:
    """One working-corpus case, as data the player may read.

    ``spec`` is the term rendered in ``object_synthesis``'s
    declarative spec language — leaves abstracted to metavariables —
    so a ``fold``/``lift`` move may *cite* the composite spellings it
    contains without the state handing out term objects.
    """

    source: str
    name: str
    nodes: int
    ops: tuple
    term: str
    spec: Any = None


@dataclass(frozen=True)
class StageLine:
    """One gauntlet gate's verdict line."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ObjectView:
    """A stored declared object plus its last gauntlet outcome.

    ``usable`` is ``None`` before the object faced the gauntlet;
    ``failed`` names the first refusing gate (``""`` when none).
    ``fires`` / ``paid`` are the holdout-probe counts — the reward
    signal itself; ``guard_synth`` / ``guard_real`` are the
    guarded-region ``accepted`` site counts the sweeps measured.
    ``missing_hooks`` is the record's own honesty flag — a
    ``full-data`` refusal on exactly ``("check",)`` is the
    conditional-candidate case an ``auto_cond`` move addresses.
    ``has_guard`` reports whether the record carries any firing
    guard at all (a declarative ``cond`` or a dropped ``check``) —
    an unguarded object is an ``auto_cond`` target.

    ``cond_clauses`` is the declarative guard flattened to its
    top-level conjuncts (clause data, in ``relax_guard``'s index
    order) — what a ``relax_guard`` move may drop.
    ``leaf_metavars`` / ``attr_metavars`` are the LHS pattern's free
    names — what a ``specialize`` move may bind.  All three are the
    object's own stored data, safe for the player to see.
    """

    alpha_key: str
    name: str
    kind: str
    usable: bool | None
    cleared: int
    stages: tuple[StageLine, ...]
    failed: str
    fires: int
    paid: int
    guard_synth: int
    guard_real: int
    serializable: bool
    missing_hooks: tuple
    has_guard: bool
    cond_clauses: tuple = ()
    leaf_metavars: tuple = ()
    attr_metavars: tuple = ()


@dataclass(frozen=True)
class ProposalView:
    """One measured candidate row, scrubbed for the player.

    Only the corpus-invariant and *count* columns are served: the
    case names inside ``fire_cases`` / ``ill_typed_cases`` identify
    the holdout workloads and never cross into the observation.
    ``scope`` is ``"current"`` for a verdict measured under the
    board's present corpus, ``"prior"`` for one a corpus rotation
    left behind (its ``fires``/``paid`` answer "on that corpus" —
    ``evidence.CORPUS_DEPENDENT_COLS``).
    """

    alpha_key: str
    name: str
    family: str
    verdict: str
    numeric_true: bool | None
    fires: int
    paid: int
    scope: str


@dataclass(frozen=True)
class ArenaState:
    """The player's read-only view of the board — pure data.

    Everything a policy needs and nothing it must not have: the
    working corpus (never the holdout's terms), the stored objects
    with verdicts and guard regions, the measured-proposal table,
    the pending ingest names and the resolvable premise names.
    ``premise_forms`` is ``(name, lhs_spec, rhs_spec)`` per resolvable
    premise — the pattern sides as spec data, what a ``compose``
    ref binds.  ``carriers`` is the board's declared carrier basis —
    ``(carrier_op, apply_op, has_state)`` templates a ``lift`` move
    instantiates.  ``guard_fires`` maps a stored object's alpha key
    to the total guarded sites its sweeps accepted — how wide its
    firing region measured.  ``to_dict`` renders the whole snapshot
    JSON-safe.
    """

    corpus: tuple[CaseView, ...]
    objects: tuple[ObjectView, ...]
    proposals: tuple[ProposalView, ...]
    pending: tuple[str, ...]
    premises: tuple[str, ...]
    premise_forms: tuple[tuple[str, Any, Any], ...]
    carriers: tuple[tuple[str, str, bool], ...]
    guard_fires: dict[str, int]
    actions: tuple[str, ...]
    epoch: int
    steps: int
    scope: str

    def to_dict(self) -> dict:
        """Return the snapshot as a JSON-safe dict."""
        return {
            "corpus": [
                {**c.__dict__, "ops": list(c.ops)} for c in self.corpus
            ],
            "objects": [
                {
                    **o.__dict__,
                    "stages": [
                        [s.name, s.passed, s.detail] for s in o.stages
                    ],
                    "missing_hooks": list(o.missing_hooks),
                }
                for o in self.objects
            ],
            "proposals": [p.__dict__ for p in self.proposals],
            "pending": list(self.pending),
            "premises": list(self.premises),
            "premise_forms": [list(f) for f in self.premise_forms],
            "carriers": [list(c) for c in self.carriers],
            "guard_fires": dict(self.guard_fires),
            "actions": list(self.actions),
            "epoch": self.epoch,
            "steps": self.steps,
            "scope": self.scope,
        }


# ---------------------------------------------------------------------------
#  Step report — the dense reward decomposition
# ---------------------------------------------------------------------------

#: Reward composition — partial credit per cleared gauntlet stage,
#: a bonus on the honest ``usable`` verdict, and the holdout
#: firing/pay columns.  The weights are data in
#: :mod:`catopt_discovery.lawdata` (:data:`ARENA_REWARD`) — a learned
#: player scores its moves on the same table.
_REWARD = lawdata.ARENA_REWARD


@dataclass(frozen=True)
class StepReport:
    """What one arena step measured — the reward decomposition.

    ``applied`` is False on the honest refusals (malformed spec,
    unknown premise, compose declined, auto-cond found no cover,
    unknown pending names): nothing was stored, no stage ran, the
    reward is 0.  Otherwise ``stages`` is the gauntlet's own verdict
    list, ``stages_cleared`` its passed count (the dense-reward
    signal), ``usable`` its verdict and ``holdout_fires`` /
    ``holdout_paid`` the counts measured on the held-out probe only.
    ``reward`` composes them as the module docstring states.
    """

    action: Action
    applied: bool
    note: str
    alpha_key: str = ""
    stages: tuple[StageLine, ...] = ()
    stages_cleared: int = 0
    stages_total: int = 0
    usable: bool = False
    holdout_fires: int = 0
    holdout_paid: int = 0
    ingested: int = 0
    reward: float = 0.0


# ---------------------------------------------------------------------------
#  The arena — board, split, step, episode
# ---------------------------------------------------------------------------


def _scope_entries(
    cases: Iterable[TermCase], tag: str
) -> Iterable[str]:
    """Content-address one case list for the evidence scope hash.

    ``op_repr_dag``, not ``op_repr``: corpus terms are sharing-heavy
    DAGs and the tree-expanding repr is exponential on them.
    """
    return (
        f"{tag}:{c.source}:{c.name}:{op_repr_dag(c.term)}"
        for c in cases
    )


def _census_of(cases: Iterable[TermCase]) -> dict:
    """Op-tuple census over the working terms (for ``census_op``)."""
    cts = [CorpusTerm(c.source, c.name, c.term) for c in cases]
    counts, _ = op_tuple_census(cts)
    return dict(counts)


def _ops_of(term: Any, out: set | None = None) -> set:
    """Return the op names inside *term*."""
    out = set() if out is None else out
    if isinstance(term, Op):
        out.add(term.op)
        for a in term.args:
            _ops_of(a, out)
    return out


def _n_nodes(term: Any) -> int:
    """Return the op-node count of *term*."""
    if isinstance(term, Op):
        return 1 + sum(_n_nodes(a) for a in term.args)
    return 0


def _case_view(case: TermCase) -> CaseView:
    """Render one working case as the player's read-only view."""
    return CaseView(
        source=case.source,
        name=case.name,
        nodes=_n_nodes(case.term),
        ops=tuple(sorted(_ops_of(case.term))),
        term=op_repr_dag(case.term),
        spec=_term_spec(case.term),
    )


def _all_rules() -> tuple:
    """Return the default premise library — the shipped rule set."""
    from catopt_core.laws import ALL_RULES

    return tuple(ALL_RULES)


def _default_sink() -> Any:
    """Return the pipeline's lowering backend."""
    from catopt_discovery.shape_proposal import _sink

    return _sink()


def _default_cost_fn(sink: Any) -> Any:
    """Return the pipeline's pricing function over *sink*."""
    from catopt_discovery.impact import _cost_fn

    return _cost_fn(sink)


def _pending_pool(pending: Iterable | dict) -> dict:
    """Normalize the ingestable pool to a ``{name: TermCase}`` map."""
    if isinstance(pending, dict):
        return {str(k): v for k, v in pending.items()}
    return {c.name: c for c in pending}


def _gauntlet_lines(rep: ev.Gauntlet | None) -> tuple:
    """Render a gauntlet's stage list as ``StageLine``s (or empty)."""
    if rep is None:
        return ()
    return tuple(
        StageLine(s.name, s.passed, s.detail) for s in rep.stages
    )


def _holdout_counts(rep: ev.Gauntlet) -> tuple[int, int]:
    """Return the probe-measured ``(fires, paid)`` — ``(0, 0)`` if none."""
    evd = rep.evidence
    return (evd.fires, evd.paid) if evd is not None else (0, 0)


def _region_accepted(region: ev.GuardedRegion | None) -> int:
    """Return a guarded region's accepted count (``0`` if none ran)."""
    return region.accepted if region is not None else 0


def _has_guard(record: dict) -> bool:
    """Whether the stored record carries any firing guard.

    A declarative ``cond`` is data on the record; a procedural
    ``check`` cannot serialize, so its presence is read off the
    record's ``missing_hooks`` honesty flag.
    """
    return record.get("cond") is not None or "check" in (
        record.get("missing_hooks") or ()
    )


def _cond_clauses(record: dict) -> tuple:
    """Return the record's declarative guard as flattened conjuncts."""
    clauses = obs._conjuncts(record.get("cond"))
    return tuple(clauses) if clauses else ()


def _data_metavars(d: Any) -> tuple:
    """Ordered leaf-metavariable names of a ``term_to_data`` tree.

    A metavariable is ``{"mvar": name}``; typed leaves
    (``var``/``param``) and constants are leaves, not metavars.
    """
    out: list[str] = []
    if not isinstance(d, dict):
        return ()
    if "mvar" in d:
        return (d["mvar"],)
    if "op" in d:
        for a in d.get("args", ()):
            out.extend(m for m in _data_metavars(a) if m not in out)
    return tuple(out)


def _data_attr_metavars(d: Any) -> tuple:
    """Ordered attr-metavariable names of a ``term_to_data`` tree.

    An attr metavariable is a bare ``str`` value inside an ``op``'s
    ``attrs`` — including inside ``{"__list__"/"__tuple__"}`` attrs.
    """
    out: list[str] = []

    def walk(v: Any) -> None:
        if isinstance(v, str) and v not in out:
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    if isinstance(d, dict) and "op" in d:
        for v in d.get("attrs", {}).values():
            walk(v)
        for a in d.get("args", ()):
            out.extend(
                m for m in _data_attr_metavars(a) if m not in out
            )
    return tuple(out)


def _record_metavars(record: dict) -> tuple[tuple, tuple]:
    """Return ``(leaf_metavars, attr_metavars)`` of the LHS record."""
    lhs = record.get("lhs")
    return _data_metavars(lhs), _data_attr_metavars(lhs)


def _bare_object_view(row: dict, record: dict) -> ObjectView:
    """Render a stored object that never faced the gauntlet."""
    leaf_mv, attr_mv = _record_metavars(record)
    return ObjectView(
        alpha_key=row["alpha_key"],
        name=row["name"],
        kind=record.get("kind", "law"),
        usable=None,
        cleared=0,
        stages=(),
        failed="",
        fires=0,
        paid=0,
        guard_synth=0,
        guard_real=0,
        serializable=bool(record.get("serializable", True)),
        missing_hooks=tuple(record.get("missing_hooks") or ()),
        has_guard=_has_guard(record),
        cond_clauses=_cond_clauses(record),
        leaf_metavars=leaf_mv,
        attr_metavars=attr_mv,
    )


def _reported_object_view(
    row: dict, record: dict, rep: ev.Gauntlet
) -> ObjectView:
    """Render a stored object plus its last gauntlet report."""
    fires, paid = _holdout_counts(rep)
    leaf_mv, attr_mv = _record_metavars(record)
    return ObjectView(
        alpha_key=rep.alpha_key,
        name=rep.name or row["name"],
        kind=rep.kind or record.get("kind", "law"),
        usable=rep.usable,
        cleared=sum(1 for s in rep.stages if s.passed),
        stages=_gauntlet_lines(rep),
        failed=next((s.name for s in rep.stages if not s.passed), ""),
        fires=fires,
        paid=paid,
        guard_synth=_region_accepted(rep.synth_region),
        guard_real=_region_accepted(rep.real_region),
        serializable=bool(record.get("serializable", True)),
        missing_hooks=tuple(record.get("missing_hooks") or ()),
        has_guard=_has_guard(record),
        cond_clauses=_cond_clauses(record),
        leaf_metavars=leaf_mv,
        attr_metavars=attr_mv,
    )


def _object_view(row: dict, rep: ev.Gauntlet | None) -> ObjectView:
    """Render one stored object row plus its last gauntlet report."""
    record = json.loads(row["law_json"])
    if rep is None:
        return _bare_object_view(row, record)
    return _reported_object_view(row, record, rep)


def _proposal_view(
    row: dict, names: dict, current: bool
) -> ProposalView:
    """Render one verdict row as the scrubbed proposal view.

    ``verdicts`` rows carry no ``name``/``family`` columns — those
    live on the joined ``candidates`` row *names* supplies.
    """
    name, family = names.get(row["alpha_key"], ("", ""))
    return ProposalView(
        alpha_key=row["alpha_key"],
        name=name,
        family=family,
        verdict=row["verdict"],
        numeric_true=(
            None
            if row["numeric_true"] is None
            else bool(row["numeric_true"])
        ),
        fires=int(row["fires"]),
        paid=int(row["paid"]),
        scope="current" if current else row["corpus_hash"][:8],
    )


def _candidate_names(conn: Any) -> dict:
    """Return ``{alpha_key: (name, family)}`` from the candidates table."""
    return {
        r["alpha_key"]: (r["name"], r["family"])
        for r in conn.execute(
            "SELECT alpha_key, name, family FROM candidates"
        )
    }


class Arena:
    """The board: working corpus, held-out probe, the object store.

    *working* is the split the player acts on — its terms feed every
    gauntlet's match enumeration, real guarded sweep and truth
    instance, and ``ingest`` grows it.  *holdout* is the probe the
    reward reads — fixed at construction, invisible in the state,
    unwritable by any action.  *pending* is the ingestable pool
    (``{name: TermCase}`` or an iterable); *conn* defaults to an
    in-memory evidence store; *base_rules* the library the
    derivability oracle and compose premises draw on; *sink* /
    *cost_fn* the lowering + pricing backend (the pipeline's own
    when omitted); *carrier_rules* the lift exemplars the carrier
    basis mines (the shipped scan/om lift laws when omitted — see
    :func:`_carrier_basis`); *meta* the evidence scope (built over the
    measurement context — working terms plus the holdout probe —
    and rotated by every ingest).
    """

    def __init__(
        self,
        *,
        working: Iterable[TermCase] = (),
        holdout: Iterable[TermCase] = (),
        pending: Iterable[TermCase] | dict = (),
        conn: Any = None,
        base_rules: Iterable | None = None,
        carrier_rules: Iterable | None = None,
        sink: Any = None,
        cost_fn: Any = None,
        meta: dict | None = None,
    ) -> None:
        """Bind the splits, the store, the backend and the scope."""
        self.conn = conn if conn is not None else ev.connect(":memory:")
        self._working = list(working)
        self._holdout = tuple(holdout)
        self._pending = _pending_pool(pending)
        self._base = tuple(
            _all_rules() if base_rules is None else base_rules
        )
        self._carriers = _carrier_basis(
            _default_carrier_rules()
            if carrier_rules is None
            else carrier_rules
        )
        self._sink = _default_sink() if sink is None else sink
        self._cost_fn = (
            _default_cost_fn(self._sink) if cost_fn is None else cost_fn
        )
        self._census = _census_of(self._working)
        self.reports: dict[str, ev.Gauntlet] = {}
        self.epoch = 0
        self.steps = 0
        self._meta = self._initial_meta(meta)

    def _initial_meta(self, meta: dict | None) -> dict:
        """Assemble the evidence scope for the bound board."""
        out = {
            "corpus_hash": self._scope_hash(),
            "rules_hash": ev.rules_hash(
                repr(r) for r in self._library_keys()
            ),
            "run_id": ev.new_run_id(),
            "holdout": "",
            "ts": ev.now(),
            **(meta or {}),
        }
        if "code_rev" not in out:
            # lazy: the git probe only runs when no override supplied one
            out["code_rev"] = ev.code_rev()
        return out

    @property
    def working(self) -> tuple:
        """The working split (read-only view — ingest mutates)."""
        return tuple(self._working)

    @property
    def holdout(self) -> tuple:
        """The held-out probe cases — the reward's measurement set."""
        return self._holdout

    @property
    def pending(self) -> dict:
        """The ingestable pool — a copy; writes go through actions."""
        return dict(self._pending)

    def _library_keys(self) -> list:
        """Alpha keys of the base rule set (for ``rules_hash``)."""
        from catopt_discovery import proposal as lp

        return [lp._key(r.lhs, r.rhs) for r in self._base]

    def _scope_hash(self) -> str:
        """Content-hash the measurement context (working + holdout).

        The scope covers *what was measured*: the working terms feed
        truth/matches, the holdout cases feed fires/paid — a verdict
        row written under this hash answers "on this pair of splits".
        """
        return ev.corpus_hash(
            [
                *_scope_entries(self._working, "work"),
                *_scope_entries(self._holdout, "hold"),
            ]
        )

    def _gauntlet(self) -> ev.GauntletCorpus:
        """Assemble the gauntlet corpus — working terms, holdout probe."""
        return ev.GauntletCorpus(
            real_terms=tuple(c.term for c in self._working),
            probe=self._holdout,
            base_rules=self._base,
            census_op=self._census,
            sink=self._sink,
            cost_fn=self._cost_fn,
        )

    def ingest(self, names: Iterable[str]) -> tuple[int, list[str]]:
        """Move *names* from pending into working; rotate the scope.

        Returns ``(moved, missing)`` — the second element lists the
        names that were never in the pool (the honest count: an
        ingest only moves what exists).  Any move recomputes the
        corpus census and the evidence scope so later verdict rows
        attribute to the grown corpus.
        """
        moved, missing = 0, []
        for n in names:
            case = self._pending.pop(n, None)
            if case is None:
                missing.append(n)
                continue
            self._working.append(case)
            moved += 1
        if moved:
            self._census = _census_of(self._working)
            self._meta["corpus_hash"] = self._scope_hash()
            self._meta["ts"] = ev.now()
            self.epoch += 1
        return moved, missing

    def resolve_ref(self, ref: Any) -> Any:
        """Resolve an object reference to a live ``Rewrite``.

        Accepts a ``Rewrite`` / ``ConstructedObject`` pass-through, a
        base-rule name, a stored alpha key or a stored object name —
        the premise/auto-cond vocabulary ``premises`` advertises.
        ``None`` when the reference resolves to nothing on the board.
        """
        if isinstance(ref, Rewrite):
            return ref
        if isinstance(ref, obs.ConstructedObject):
            return ref.rule
        if not isinstance(ref, str):
            return None
        for r in self._base:
            if r.name == ref:
                return r
        got = ev.admit_object(self.conn, ref)
        if got is not None:
            return got[0]
        for row in ev.lemma_rows(self.conn):
            if row["name"] == ref:
                # resolve by alpha key — the same admit path; a
                # record that fails to rebuild surfaces its error
                return self.resolve_ref(row["alpha_key"])
        return None

    def _premises(self) -> tuple:
        """Names a compose/auto_cond ref may resolve: library + store."""
        return tuple(
            sorted(
                {r.name for r in self._base}
                | {row["name"] for row in ev.lemma_rows(self.conn)}
            )
        )

    def _premise_forms(self) -> tuple:
        """``(name, lhs_spec, rhs_spec)`` per resolvable premise.

        The pattern sides as spec data — what a ``compose`` ref binds
        when :func:`legal_actions` tests composability.  Name
        resolution order matches :meth:`resolve_ref`: the library
        first, then the store's rows in store order.
        """
        out: dict[str, tuple] = {}
        for r in self._base:
            out[r.name] = (_term_spec(r.lhs), _term_spec(r.rhs))
        for row in ev.lemma_rows(self.conn):
            if row["name"] in out:
                continue
            got = ev.admit_object(self.conn, row["alpha_key"])
            if got is not None:
                out[row["name"]] = (
                    _term_spec(got[0].lhs),
                    _term_spec(got[0].rhs),
                )
        return tuple(
            sorted((name, lhs, rhs) for name, (lhs, rhs) in out.items())
        )

    def _record(self, alpha_key: str, rep: ev.Gauntlet) -> None:
        """Persist the object's measured verdict under the scope.

        The gauntlet's ``evidence`` row becomes a store verdict —
        ``proposals`` in the observation reads it back through
        ``latest_verdicts`` / ``verdicts_across_scopes``, so the same
        machinery that attributes the meta-game's verdicts attributes
        a construction's measurements (rotated scopes included).
        """
        evd = rep.evidence
        if evd is None or rep.rule is None:
            return
        row = ev.verdict_row(
            alpha_key, evd, op_repr(rep.rule.lhs), op_repr(rep.rule.rhs)
        )
        ev.record_run(self.conn, self._meta, [row])

    def observe(self) -> ArenaState:
        """Build the read-only snapshot — no store writes, no holdout.

        The holdout enters the state only as the reward *counts* on
        stored objects (``fires``/``paid``); its terms and case names
        never cross — ``proposals`` is scrubbed to counts, and
        ``corpus`` lists the working split alone.
        """
        objects = tuple(
            _object_view(row, self.reports.get(row["alpha_key"]))
            for row in ev.lemma_rows(self.conn)
        )
        names = _candidate_names(self.conn)
        verdicts = ev.latest_verdicts(
            self.conn,
            self._meta["corpus_hash"],
            self._meta["rules_hash"],
            self._meta["code_rev"],
        )
        history = ev.verdicts_across_scopes(self.conn, self._meta)
        history.pop(self._meta["corpus_hash"], None)
        proposals = tuple(
            _proposal_view(r, names, True) for r in verdicts.values()
        ) + tuple(
            _proposal_view(r, names, False)
            for scope in history.values()
            for r in scope.values()
        )
        guard_fires = {
            o.alpha_key: o.guard_synth + o.guard_real for o in objects
        }
        return ArenaState(
            corpus=tuple(_case_view(c) for c in self._working),
            objects=objects,
            proposals=proposals,
            pending=tuple(sorted(self._pending)),
            premises=self._premises(),
            premise_forms=self._premise_forms(),
            carriers=self._carriers,
            guard_fires=guard_fires,
            actions=tuple(sorted(ACTIONS)),
            epoch=self.epoch,
            steps=self.steps,
            scope=self._meta["corpus_hash"][:8],
        )

    def _referee(self, out: _Outcome) -> tuple[str, ev.Gauntlet | None]:
        """Store a produced object and run the gauntlet on it.

        Returns ``(alpha_key, report)`` — the key is empty and the
        report ``None`` when the outcome carried no object (a corpus
        move or a refusal).
        """
        key = ""
        if out.obj is not None:
            key = obs.store_constructed(
                self.conn,
                out.obj,
                self._meta["corpus_hash"],
                env=out.env,
            )
        elif out.key:
            key = out.key
        rep: ev.Gauntlet | None = None
        if key:
            rep = ev.run_gauntlet(
                self.conn, key, corpus=self._gauntlet()
            )
            self.reports[key] = rep
            self._record(key, rep)
        return key, rep

    @staticmethod
    def _step_report(
        action: Action,
        out: _Outcome,
        key: str,
        rep: ev.Gauntlet | None,
    ) -> StepReport:
        """Compose the reward decomposition for one applied step."""
        stages = _gauntlet_lines(rep)
        cleared = sum(1 for s in stages if s.passed)
        fires, paid = (
            _holdout_counts(rep) if rep is not None else (0, 0)
        )
        usable = rep.usable if rep is not None else False
        refereed = out.applied and rep is not None
        reward = (
            _REWARD["stage"] * cleared
            + _REWARD["usable"] * usable
            + _REWARD["fire"] * fires
            + _REWARD["paid"] * paid
            if refereed
            else 0.0
        )
        return StepReport(
            action=action,
            applied=out.applied,
            note=out.note or (rep.reason if rep is not None else ""),
            alpha_key=key,
            stages=stages,
            stages_cleared=cleared,
            stages_total=len(stages),
            usable=usable,
            holdout_fires=fires,
            holdout_paid=paid,
            ingested=out.ingested,
            reward=reward,
        )

    def step(self, action: Action) -> tuple[ArenaState, StepReport]:
        """Apply *action*, referee the result, return state' + report.

        The handler constructs (or declines); a produced object is
        stored through ``object_synthesis.store_constructed`` and
        faces ``evidence.run_gauntlet`` on a corpus whose probe is
        the holdout.  The reward is the documented composition — a
        refusal scores 0, ingest scores 0 immediately (its payoff is
        what it later enables — see ``Trajectory.hindsight``).
        An unregistered ``op`` is a ``KeyError`` — the registry is
        the action space's own contract.
        """
        fn = ACTIONS.get(action.op)
        if fn is None:
            raise KeyError(
                f"unregistered action {action.op!r} "
                f"(registered: {sorted(ACTIONS)})"
            )
        out = fn(self, action)
        key, rep = self._referee(out)
        report = self._step_report(action, out, key, rep)
        self.steps += 1
        return self.observe(), report


# ---------------------------------------------------------------------------
#  Legal moves — the move set the board actually affords
# ---------------------------------------------------------------------------
#
#  ``Arena.step`` accepts any well-formed ``Action``, so a playbook can
#  name its moves — but a *player* needs the converse: which
#  construction moves does this state afford?  ``legal_actions``
#  answers from the observation alone.  The honesty rule is "no
#  fabricated parameters": every term spec is a real corpus subterm,
#  every ref a resolvable premise, every carrier a declared basis
#  template, every ``specialize`` map a binding that makes the firing
#  premise match.  Legality means *constructible* — whether the
#  constructed object is true is the gauntlet's question, never the
#  enumerator's.


def _term_spec(term: Any, names: dict | None = None) -> Any:
    """Render *term* as declarative term-spec data.

    The inverse of ``object_synthesis.term_from_spec``: an ``Op``
    becomes ``(op, *child_specs[, attrs])``, a ``Const`` its value, a
    ``str`` metavariable stays verbatim, and any other leaf
    (``Var``/``Param``) becomes a metavariable named after the leaf's
    own ``name`` — or ``X1``, ``X2``, … in first-occurrence pre-order
    when the leaf carries none.  The spec is pure data a move may
    cite — no term objects cross into the observation.
    """
    names = {} if names is None else names
    if isinstance(term, str):
        return term
    if isinstance(term, Op):
        spec = [term.op]
        spec.extend(_term_spec(a, names) for a in term.args)
        if term.attrs:
            spec.append(dict(term.attrs))
        return tuple(spec)
    if isinstance(term, Const):
        return term.value
    if term not in names:
        names[term] = getattr(term, "name", "") or f"X{len(names) + 1}"
    return names[term]


def _spec_metavars(spec: Any) -> tuple:
    """Ordered leaf-metavariable names of a term spec (pre-order)."""
    out: list[str] = []
    if isinstance(spec, str):
        out.append(spec)
    elif isinstance(spec, (tuple, list)):
        for s in spec[1:]:
            if not isinstance(s, dict):
                out.extend(m for m in _spec_metavars(s) if m not in out)
    return tuple(out)


def _spec_ops(spec: Any) -> set:
    """Return the op names occurring in a term spec."""
    if not isinstance(spec, (tuple, list)) or not spec:
        return set()
    out = {spec[0]}
    for s in spec[1:]:
        if not isinstance(s, dict):
            out |= _spec_ops(s)
    return out


def _spec_size(spec: Any) -> int:
    """Return the op-node count of a term spec."""
    if not isinstance(spec, (tuple, list)) or not spec:
        return 0
    return 1 + sum(
        _spec_size(s) for s in spec[1:] if not isinstance(s, dict)
    )


def _sub_specs(spec: Any) -> Iterable:
    """Yield every subterm spec (tuple node) of *spec*, pre-order."""
    if isinstance(spec, (tuple, list)) and spec:
        yield spec
        for s in spec[1:]:
            if not isinstance(s, dict):
                yield from _sub_specs(s)


def _renumber_spec(spec: Any, names: dict) -> Any:
    """Alpha-normalize a spec's leaf metavars — ``X1``, ``X2``, ….

    Two corpus cases spelling the same subterm under different
    variable names canonicalize to one spec — the same construction,
    enumerated once.  Attr values pass through verbatim: corpus attrs
    are *concrete* values (a string ``mode`` is data, not an attr
    metavariable).
    """
    if isinstance(spec, str):
        if spec not in names:
            names[spec] = f"X{len(names) + 1}"
        return names[spec]
    if isinstance(spec, (tuple, list)) and spec:
        return (
            spec[0],
            *(_renumber_spec(s, names) for s in spec[1:]),
        )
    return spec


def _spec_walk(spec: Any) -> Iterable:
    """Yield ``(subterm_spec, node_count)`` — one post-order pass.

    Post-order means the inner loop's last yield is the child root
    itself, so ``n`` after the loop is that child's size — a single
    pass, no quadratic per-subterm recount on real corpus terms.
    """
    if not isinstance(spec, (tuple, list)) or not spec:
        return
    total = 1
    for s in spec[1:]:
        if isinstance(s, dict):
            continue
        n = 0
        for sub, n in _spec_walk(s):
            yield sub, n
        total += n
    yield spec, total


def _composite_specs(state: ArenaState, cap: int) -> list:
    """Distinct composite (≥2-op) subterm specs across the corpus.

    Canonicalized to ``X1…`` metavariables, sorted by their canonical
    repr, truncated at *cap* — the fold/lift spelling inventory.
    """
    seen: dict[str, Any] = {}
    for c in state.corpus:
        for s, n in _spec_walk(c.spec):
            if n < 2:
                continue
            canon = _renumber_spec(s, {})
            seen.setdefault(repr(canon), canon)
    return [seen[k] for k in sorted(seen)][:cap]


def _op_vocabulary(state: ArenaState) -> list:
    """Sorted op names the state attests — corpus + premise patterns."""
    out = {op for c in state.corpus for op in c.ops}
    for _n, lhs, rhs in state.premise_forms:
        out |= _spec_ops(lhs)
        out |= _spec_ops(rhs)
    return sorted(out)


def _ingest_moves(state: ArenaState) -> list:
    """One ``ingest`` per pending name, plus the whole pool."""
    out = [Action.ingest((n,)) for n in state.pending]
    if len(state.pending) > 1:
        out.append(Action.ingest(tuple(state.pending)))
    return out


def _auto_cond_moves(state: ArenaState) -> list:
    """``auto_cond`` on unguarded objects and conditional verdicts."""
    targets = {
        o.alpha_key for o in state.objects if not o.has_guard
    } | {
        p.alpha_key
        for p in state.proposals
        if "conditional" in p.verdict
    }
    return [Action.auto_cond(k) for k in sorted(targets)]


def _relax_moves(state: ArenaState) -> list:
    """``relax_guard`` moves: each stored object's each cond clause."""
    return [
        Action.relax_guard(o.alpha_key, i)
        for o in state.objects
        for i in range(len(o.cond_clauses))
    ]


def _specialize_moves(
    state: ArenaState, scalars: tuple, axes: tuple
) -> list:
    """``specialize`` moves: metavariable pins over the honest banks.

    Leaf metavariables draw from *scalars* (the lawdata constant
    bank), attr metavariables from *axes* (small axis candidates).
    Both banks are data — a player that wants a richer pin writes a
    richer bank, not Python.
    """
    out: list[Action] = []
    for o in state.objects:
        for mv in o.leaf_metavars:
            for v in scalars:
                out.append(Action.specialize(o.alpha_key, {mv: v}))
        for mv in o.attr_metavars:
            for v in axes:
                out.append(Action.specialize(o.alpha_key, {mv: v}))
    return out


def _fold_moves(state: ArenaState, cap: int) -> list:
    """``fold`` moves: corpus composite spellings vs attested ops.

    A fold names a spelled composite and the *primitive* kernel it
    folds to — an op the state already attests (corpus ops plus the
    premise-pattern vocabulary) that does not occur inside the
    spelled form (a fold introduces a fused kernel, it does not
    restate a node).  Both the unary kernel over each metavariable
    and the all-metavar kernel are legal spellings.
    """
    vocab = _op_vocabulary(state)
    out: list[Action] = []
    for s in _composite_specs(state, cap):
        inner = _spec_ops(s)
        mvs = _spec_metavars(s)
        for k in vocab:
            if k in inner:
                continue
            out.append(Action.fold(s, (k, *mvs)))
            for m in mvs:
                out.append(Action.fold(s, k, arg=m))
    return out


def _lift_moves(state: ArenaState, cap: int) -> list:
    """``lift`` moves: composite steps over the declared carrier basis.

    For a stateful carrier (``apply(aff(A, x), h)``) each metavariable
    in turn is the state choice — the carrier takes the rest; a
    stateless carrier (``om_apply(om_elem(s, v))``) takes them all.
    """
    out: list[Action] = []
    for s in _composite_specs(state, cap):
        mvs = _spec_metavars(s)
        for cop, aop, has_state in state.carriers:
            if not has_state:
                out.append(Action.lift(s, (cop, *mvs), aop))
                continue
            for st in mvs:
                carrier = tuple([cop, *[m for m in mvs if m != st]])
                out.append(Action.lift(s, carrier, aop, state=st))
    return out


def _spec_attrs_match(pat: Op, sub: Op, out: dict) -> bool:
    """Attr compatibility for :func:`_specialize_match` (in *out*)."""
    if set(pat.attrs) != set(sub.attrs):
        return False
    for k, pv in pat.attrs.items():
        sv = sub.attrs[k]
        if isinstance(pv, str):
            continue
        if isinstance(sv, str):
            out[sv] = pv
        elif pv != sv:
            return False
    return True


def _specialize_match(pat: Any, sub: Any) -> dict | None:
    """Derive a metavar map making *pat* match the instantiated *sub*.

    *pat* is the firing premise's LHS, *sub* a subterm of the head
    premise's RHS — both pattern terms, so ``sub``'s leaves are
    metavariables the ``specialize`` map may instantiate.  Returns the
    map (possibly ``{}`` — no instantiation needed), or ``None`` when
    no metavar instantiation can make them match — an op head the
    specialize map cannot rewrite is a hard stop.
    """
    if isinstance(pat, str):
        return {}
    if isinstance(sub, str):
        return {sub: pat}
    if isinstance(pat, Op) and isinstance(sub, Op):
        return _specialize_match_op(pat, sub)
    if isinstance(pat, Const) and isinstance(sub, Const):
        return {} if pat.value == sub.value else None
    return {} if pat == sub else None


def _specialize_match_op(pat: Op, sub: Op) -> dict | None:
    """Derive the specialize map for two same-shaped ``Op`` nodes."""
    if pat.op != sub.op or len(pat.args) != len(sub.args):
        return None
    out: dict = {}
    if not _spec_attrs_match(pat, sub, out):
        return None
    for pa, sa in zip(pat.args, sub.args, strict=True):
        merged = _specialize_match(pa, sa)
        if merged is None:
            return None
        for m, v in merged.items():
            if m in out and out[m] != v:
                return None
            out[m] = v
    return out


def _compose_moves(state: ArenaState, max_specialize: int) -> list:
    """``compose`` pairs the premise patterns structurally support.

    For each ordered premise pair ``(first, second)``: a bare
    structural match of ``second.lhs`` anywhere in ``first.rhs``
    yields ``compose(first, second)``; where the match needs the head
    premise's metavariables instantiated (a metavar leaf facing a
    compound pattern — the ``comm_mul ∘ silu_fold`` case), the
    specialize map :func:`_specialize_match` derives is emitted as
    ``compose(first, second, specialize={metavar: spec})``.
    Legality here is pattern-level composability — the constructor's
    guard transport still referees whether a guarded premise fires.
    """
    forms = {
        n: (obs.term_from_spec(lhs), obs.term_from_spec(rhs))
        for n, lhs, rhs in state.premise_forms
    }
    out: list[Action] = []
    for p1 in sorted(forms):
        subs = [s for _p, s in _positions(forms[p1][1])]
        for p2 in sorted(forms):
            l2 = forms[p2][0]
            if any(match_pattern(l2, s, {}) is not None for s in subs):
                out.append(Action.compose(p1, p2))
            seen: set[str] = set()
            for s in subs:
                sig = _specialize_match(l2, s)
                if not sig:
                    continue
                key = repr(sorted(sig.items(), key=lambda kv: kv[0]))
                if key in seen:
                    continue
                seen.add(key)
                out.append(Action.compose(p1, p2, specialize=dict(sig)))
                if len(seen) >= max_specialize:
                    break
    return out


def _hashable_param(v: Any) -> Any:
    """Hashable form of a param value — dicts → frozensets of items."""
    if isinstance(v, dict):
        return frozenset((k, _hashable_param(x)) for k, x in v.items())
    if isinstance(v, (tuple, list)):
        return tuple(_hashable_param(x) for x in v)
    return v


def _action_key(action: Action) -> Any:
    """Return a stable dedup key for an enumerated move.

    ``(op, hashable params)`` — two actions naming the same
    construction under the same parameters dedupe, without a repr
    walk per candidate (which dominates enumeration on a real
    board).
    """
    return (action.op, _hashable_param(action.params))


def legal_actions(
    state: ArenaState, *, max_specs: int = 128, max_specialize: int = 6
) -> tuple[Action, ...]:
    """Enumerate the construction moves *state* affords, deterministically.

    The live move set across the seven constructors:

    * ``ingest`` — each pending name alone, and the whole pool;
    * ``auto_cond`` — stored objects without a guard, plus proposals
      whose verdict measured ``conditional``;
    * ``fold`` — each distinct composite corpus subterm spec against
      every attested op name not inside the spelling (unary kernels
      over each metavariable, and the all-metavar kernel);
    * ``lift`` — the same composite steps against the declared
      ``carriers`` basis (each metavariable as state choice for a
      stateful carrier);
    * ``compose`` — premise pairs whose patterns compose
      (:func:`_compose_moves`);
    * ``relax_guard`` — each clause of each stored object's
      declarative guard (:func:`_relax_moves`);
    * ``specialize`` — each leaf/attr metavariable of each stored
      object over the lawdata pin banks (:func:`_specialize_moves`).

    Re-declaring an already-stored construction stays legal — it
    re-gauntlets the same alpha key (a legal no-op); filtering it
    would cost one alpha-key computation per candidate, which
    dominates the enumeration on a real board.  *max_specs* bounds
    the composite-spelling inventory; *max_specialize* the
    specialize maps per premise pair.  The result is unique by
    construction — composite specs are deduped canonically, premise
    pairs and targets are enumerated once each — and is returned in
    enumeration order, so the list a player sees is a pure function
    of the observation.
    """
    return tuple(
        [
            *_ingest_moves(state),
            *_auto_cond_moves(state),
            *_fold_moves(state, max_specs),
            *_lift_moves(state, max_specs),
            *_compose_moves(state, max_specialize),
            *_relax_moves(state),
            *_specialize_moves(
                state,
                lawdata.SPECIALIZE_SCALARS,
                lawdata.SPECIALIZE_AXES,
            ),
        ]
    )


# ---------------------------------------------------------------------------
#  The carrier basis — which carrier forms the board declares
# ---------------------------------------------------------------------------


def _default_carrier_rules() -> list:
    """Return the shipped lift exemplars the default basis mines.

    The scan lifts live in core (``aff``/``aff_diag``); the online
    softmax carrier lives in ``catopt_carriers`` — not a declared
    dependency of this package, so its absence degrades the basis to
    the scan carriers rather than failing.
    """
    from catopt_core.laws.scan import SCAN_DIAG_LAWS, SCAN_LAWS

    rules = [*SCAN_LAWS, *SCAN_DIAG_LAWS]
    try:
        from catopt_carriers import om
    except ImportError:
        return rules
    return [*rules, *om.OM_LAWS, *om.OM_MASK_LAWS]


def _carrier_basis(rules: Iterable) -> tuple:
    """Mine ``(carrier_op, apply_op, has_state)`` templates from lifts.

    A lift-shaped rule reads ``step → apply_op(carrier(…)[, state])``
    — an ``*apply*`` root over a compound first child.  Entries that
    are already ``(op, apply, bool)`` triples pass through verbatim,
    so a caller can declare a carrier no shipped law spells.
    """
    out: set[tuple] = set()
    for e in rules:
        if isinstance(e, tuple) and len(e) == 3:
            out.add((e[0], e[1], bool(e[2])))
            continue
        rhs = getattr(e, "rhs", None)
        if (
            isinstance(rhs, Op)
            and rhs.args
            and isinstance(rhs.args[0], Op)
            and "apply" in rhs.op
        ):
            out.add((rhs.args[0].op, rhs.op, len(rhs.args) > 1))
    return tuple(sorted(out))


# ---------------------------------------------------------------------------
#  The honest split — the holdout is real-only
# ---------------------------------------------------------------------------


def purpose_built_names(cands: Any = None) -> frozenset:
    """Return the intake ledger's ``intake:{name}`` spellings.

    ``intake.candidates()`` marks every micro-module written to
    spell one pattern with ``purpose_built=True`` — the corpus-
    circular ledger.  A case named ``intake:{name}`` matching one of
    them may serve the working split but never the holdout.
    """
    from catopt_discovery import intake as li

    src = li.candidates() if cands is None else cands
    return frozenset(f"intake:{w.name}" for w in src if w.purpose_built)


def split_corpus(
    cases: Iterable[TermCase],
    *,
    seed: int = 0,
    holdout_frac: float = 0.25,
    holdout_names: Iterable[str] | None = None,
    purpose_built: Iterable[str] = (),
    probe_eligible: Iterable[str] | None = None,
) -> tuple[list[TermCase], list[TermCase]]:
    """Split *cases* into ``(working, holdout)``, deterministically.

    The holdout is drawn only from cases that are (a) *real* —
    never a name in *purpose_built* (the intake ledger; pay on a
    seeded spelling does not count) — and (b) *probe-eligible* —
    in *probe_eligible*'s name set when one is supplied (a firing
    probe needs a persisted feed; pass ``intake.probe_cases``'s
    names for the real board).  An explicit *holdout_names* is
    intersected with the eligible set the same way — a barred or
    ineligible name silently stays in working, never in holdout.
    Everything else — the purpose-built spellings included — is
    corpus material the player may act on.
    """
    cases = list(cases)
    barred = set(purpose_built)
    probe_ok = None if probe_eligible is None else set(probe_eligible)
    eligible = {
        c.name
        for c in cases
        if c.name not in barred
        and (probe_ok is None or c.name in probe_ok)
    }
    picked = _pick_holdout(eligible, seed, holdout_frac, holdout_names)
    holdout = [c for c in cases if c.name in picked]
    working = [c for c in cases if c.name not in picked]
    return working, holdout


def _pick_holdout(
    eligible: set,
    seed: int,
    holdout_frac: float,
    holdout_names: Iterable[str] | None,
) -> set:
    """Choose the held-out case names from the eligible set.

    An explicit *holdout_names* intersects the eligible set; else a
    deterministic sha256-seeded ranking takes the *holdout_frac*
    share (at least one when anything is eligible).
    """
    if holdout_names is not None:
        return eligible & set(holdout_names)
    n_hold = (
        max(1, round(len(eligible) * holdout_frac)) if eligible else 0
    )
    ranked = sorted(
        eligible,
        key=lambda n: hashlib.sha256(f"{seed}:{n}".encode()).digest(),
    )
    return set(ranked[:n_hold])


def make_arena(
    *,
    seed: int = 0,
    holdout_frac: float = 0.25,
    conn: Any = None,
    meta: dict | None = None,
) -> Arena:
    """Assemble the real board — the pipeline corpus, honestly split.

    The probe-eligible pool is the pipeline's own firing set
    (``model_cases`` + ``intake.probe_cases``); the holdout is a
    deterministic share of its *real-only* members (the intake
    ledger's purpose-built spellings are barred — they stay
    playable working material).  The bench spellings and the
    remaining intake terms (terms only — they feed real_terms but
    carry no probe feed) seed the pending pool, so the first
    ``ingest`` action is the corpus-direction move the board is for.
    """
    from catopt_discovery import intake as li
    from catopt_discovery.impact import _bench_cases, model_cases

    models, _me = model_cases()
    probe = li.probe_cases()
    bench, _be = _bench_cases()
    pool = [*models, *probe]
    working, holdout = split_corpus(
        [*pool, *bench],
        seed=seed,
        holdout_frac=holdout_frac,
        purpose_built=purpose_built_names(),
        probe_eligible={c.name for c in pool},
    )
    seen = {c.name for c in pool} | {c.name for c in bench}
    pending = [c for c in li.load_cases() if c.name not in seen]
    return Arena(
        working=working,
        holdout=holdout,
        pending=pending,
        conn=conn,
        meta=meta,
    )


# ---------------------------------------------------------------------------
#  Players and the episode driver
# ---------------------------------------------------------------------------


class FixedRule:
    """The fixed-order baseline — an authored playbook replayed once.

    The control arm of this game, in the spirit of
    ``meta_game.EnumerationGuide``: a fixed construction order a
    learned player must beat.  The rule is (i) ``ingest`` every
    pending case first (corpus direction before construction),
    (ii) replay *moves* in order — a playbook of ``Action``s the
    caller authors (fold/lift/compose steps plus ``auto_cond`` refs
    spelled by stored name), and (iii) when *guard_conditionals*
    holds, ``auto_cond`` each stored object the gauntlet refused at
    ``truth`` (or at ``full-data`` on a lone ``check`` hook) — the
    "guard the conditional candidates" move ``guide-real-run.md``
    measured as the missing one — once each, in store order.
    """

    def __init__(
        self,
        moves: Iterable[Action] = (),
        *,
        ingest_first: bool = True,
        guard_conditionals: bool = True,
    ) -> None:
        """Bind the playbook and the fixed ordering."""
        self._moves = deque(moves)
        self._ingest_first = ingest_first
        self._guard = guard_conditionals
        self._cond_tried: set = set()
        self._ingested = not ingest_first

    def _conditional_target(self, state: ArenaState) -> Action | None:
        """Return the next conditional object's ``auto_cond`` move."""
        for o in state.objects:
            armed = o.failed == "truth" or (
                o.failed == "full-data"
                and o.missing_hooks == ("check",)
            )
            if (
                o.usable is False
                and armed
                and o.alpha_key not in self._cond_tried
            ):
                self._cond_tried.add(o.alpha_key)
                return Action.auto_cond(o.alpha_key)
        return None

    def __call__(self, state: ArenaState) -> Action | None:
        """Return the next scripted move, or ``None`` when drained."""
        if not self._ingested and state.pending:
            self._ingested = True
            return Action.ingest(state.pending)
        self._ingested = True
        if self._moves:
            return self._moves.popleft()
        if self._guard:
            return self._conditional_target(state)
        return None


class RandomPlayer:
    """Uniform play over the live legal set — the no-knowledge arm.

    Each call enumerates :func:`legal_actions` — the moves the board
    actually affords — and samples uniformly, so a random episode
    reaches ingests, constructions and guards in the proportions the
    board's move masses set.  Seeded — runs replay.  *legal* is the
    enumerator seam: any ``ArenaState -> tuple[Action, ...]``
    callable plugs in (a filtered or learned move generator).
    """

    def __init__(self, rng: random.Random, legal: Any = None) -> None:
        """Bind the PRNG and the legal-move enumerator."""
        self.rng = rng
        self._legal = legal_actions if legal is None else legal

    def __call__(self, state: ArenaState) -> Action | None:
        """Sample uniformly over the live move set."""
        acts = self._legal(state)
        if not acts:
            return None
        return acts[self.rng.randrange(len(acts))]


class GreedyPlayer:
    """Immediate-stage greed — the largest spelled spec first.

    The static estimate of immediate reward: a constructible
    ``fold``/``lift``/``compose`` reconstructs, serializes and
    measures — three near-certain stages — so each scores ``3`` plus
    a small bonus for the size of the spec it spells (a wider pattern
    covers more sites when it holds).  ``auto_cond`` scores ``2``
    (a rewrite attempt, not a fresh construction) and ``ingest``
    ``0`` — its honest immediate reward; the hindsight payoff the
    plan credits it with is invisible to one-step greed, which is
    exactly the bias this baseline exists to measure.  Ties break by
    the deterministic :func:`legal_actions` order — ``max`` keeps
    the first maximum.
    """

    def __init__(self, legal: Any = None) -> None:
        """Bind the legal-move enumerator (default: the arena's)."""
        self._legal = legal_actions if legal is None else legal

    def _score(self, state: ArenaState, action: Action) -> float:
        """Estimate the immediate reward of one move, statically."""
        p = action.params
        if action.op == "auto_cond":
            return 2.0
        if action.op == "compose":
            lhs = next(
                (
                    lhs
                    for n, lhs, _r in state.premise_forms
                    if n == p["first"]
                ),
                (),
            )
            return 3.0 + 0.01 * _spec_size(lhs)
        if action.op in ("fold", "lift"):
            spec = p["spelled"] if action.op == "fold" else p["step"]
            return 3.0 + 0.01 * _spec_size(spec)
        return 0.0

    def __call__(self, state: ArenaState) -> Action | None:
        """Return the highest-scoring legal move, or ``None``."""
        acts = self._legal(state)
        if not acts:
            return None
        return max(acts, key=lambda a: self._score(state, a))


@dataclass
class Trajectory:
    """One episode's step reports, in play order.

    ``reports`` is the ordered :class:`StepReport` list the episode
    driver recorded — each carrying its own reward decomposition.
    The aggregations are the game's honest bars: ``usable`` counts
    admitted objects; ``holdout_fires`` / ``holdout_paid`` sum the
    latest measured counts per distinct object; ``hindsight`` is
    the next-payoff attribution (the plan's "an ingest/compose
    action is scored on what it later enables") — an unrewarded
    step is credited with the first positive reward downstream of
    it, so an enabling move is scored on the construction it made
    possible rather than its own immediate delta.
    """

    reports: list[StepReport]

    @property
    def rewards(self) -> list[float]:
        """The per-step immediate reward sequence."""
        return [r.reward for r in self.reports]

    @property
    def total(self) -> float:
        """The episode's total immediate reward."""
        return sum(self.rewards)

    @property
    def objects(self) -> list[str]:
        """Distinct stored objects the episode refereed (order kept)."""
        return list(
            dict.fromkeys(
                r.alpha_key for r in self.reports if r.alpha_key
            )
        )

    @property
    def usable(self) -> int:
        """Count of distinct objects the gauntlet admitted."""
        return sum(
            1
            for k in self.objects
            if any(r.usable and r.alpha_key == k for r in self.reports)
        )

    @property
    def holdout_fires(self) -> int:
        """Latest measured holdout fires, summed over objects."""
        latest = self._latest()
        return sum(r.holdout_fires for r in latest.values())

    @property
    def holdout_paid(self) -> int:
        """Latest measured holdout pay, summed over objects."""
        latest = self._latest()
        return sum(r.holdout_paid for r in latest.values())

    def _latest(self) -> dict:
        """Return ``{alpha_key: newest StepReport}`` for stored objects."""
        out: dict[str, StepReport] = {}
        for r in self.reports:
            if r.alpha_key:
                out[r.alpha_key] = r
        return out

    def hindsight(self) -> list[float]:
        """Next-payoff attribution — credit enabling moves downstream."""
        out: list[float] = []
        for i, r in enumerate(self.reports):
            if r.reward:
                out.append(r.reward)
                continue
            nxt = next(
                (s.reward for s in self.reports[i + 1 :] if s.reward),
                0.0,
            )
            out.append(nxt)
        return out

    def trace(self) -> list[dict]:
        """Render the episode as one dict per step.

        The reportable allocation sequence, in the spirit of the
        guide's ``"trace"``.
        """
        return [
            {
                "step": i,
                "op": r.action.op,
                "applied": r.applied,
                "name": (r.action.params.get("name") or ""),
                "alpha_key": r.alpha_key[:12],
                "cleared": r.stages_cleared,
                "usable": r.usable,
                "holdout_fires": r.holdout_fires,
                "holdout_paid": r.holdout_paid,
                "ingested": r.ingested,
                "reward": round(r.reward, 3),
                "note": r.note,
            }
            for i, r in enumerate(self.reports)
        ]


def run_episode(
    arena: Arena,
    player: Callable[[ArenaState], Action | None],
    budget: int,
) -> Trajectory:
    """Play *player* on *arena* until *budget* actions are spent.

    A player is any callable ``ArenaState -> Action | None`` —
    ``None`` ends the episode early.  The budget counts actions
    (construction calls), the game's honest currency here: every
    step costs at most one construction plus one gauntlet.
    """
    reports: list[StepReport] = []
    for _ in range(budget):
        action = player(arena.observe())
        if action is None:
            break
        _state, rep = arena.step(action)
        reports.append(rep)
    return Trajectory(reports)


# ---------------------------------------------------------------------------
#  The depth probe — does move choice matter on the construction board?
# ---------------------------------------------------------------------------


def _probe_playbook() -> list[Action]:
    """Return the FixedRule playbook for the depth probe.

    The three constructions the arena tests prove end-to-end — the
    affine-step carrier lift, the SiLU composite and the softsign
    fold.  A reasonable authored baseline, deliberately not tuned to
    the corpus the probe runs on.
    """
    return [
        Action.lift(
            ("add", ("matmul", "A", "h"), "x"),
            ("aff", "A", "x"),
            "apply",
            state="h",
            name="aff_step_lift",
        ),
        Action.compose(
            "comm_mul",
            "silu_fold",
            specialize={"a": ("sigmoid", "X"), "b": "X"},
            name="silu_swapped",
        ),
        Action.fold(
            ("div", "X", ("add", ("abs", "X"), 1)),
            "softsign",
            name="softsign_fold_probe",
        ),
    ]


def _probe_players(seed: int) -> dict:
    """Return the probe's three arms — fixed playbook, uniform, greedy."""
    return {
        "fixed": lambda _e: FixedRule(_probe_playbook()),
        "random": lambda e: RandomPlayer(
            random.Random(1000 * seed + e)
        ),
        "greedy": lambda _e: GreedyPlayer(),
    }


def _probe_row(
    traj: Trajectory, episode: int, legal0: int, legal_end: int
) -> dict:
    """One episode's probe row — the plan's measured bars."""
    return {
        "episode": episode,
        "steps": len(traj.reports),
        "usable": traj.usable,
        "holdout_fires": traj.holdout_fires,
        "holdout_paid": traj.holdout_paid,
        "reward": round(traj.total, 3),
        "legal_start": legal0,
        "legal_end": legal_end,
    }


def depth_probe(
    *,
    episodes: int = 2,
    budget: int = 8,
    seed: int = 0,
    arena_factory: Any = None,
) -> dict:
    """Play the three baselines on fresh arenas; return the probe table.

    Plan 0020 phase 2: ``fixed`` replays the authored playbook,
    ``random`` samples :func:`legal_actions` uniformly, ``greedy``
    takes the largest immediate-stage estimate.  Each episode gets a
    fresh board from *arena_factory* (``make_arena(seed)`` by
    default); every row carries the legal-set size at the episode's
    start and end plus the honest bars — usable objects, holdout
    fires and holdout pay.
    """
    factory = arena_factory or (
        lambda s: make_arena(seed=s, meta={"code_rev": "probe"})
    )
    table: dict[str, list] = {}
    for name, mk in _probe_players(seed).items():
        rows: list[dict] = []
        for ep in range(episodes):
            arena = factory(seed + ep)
            legal0 = len(legal_actions(arena.observe()))
            traj = run_episode(arena, mk(ep), budget)
            legal_end = len(legal_actions(arena.observe()))
            rows.append(_probe_row(traj, ep, legal0, legal_end))
        table[name] = rows
    return table


def _fmt_probe(table: dict) -> str:
    """Render the probe table as fixed-width rows."""
    head = (
        f"{'player':<8} {'ep':>3} {'steps':>5} {'usable':>6} "
        f"{'fires':>6} {'paid':>5} {'reward':>9} "
        f"{'legal@0':>8} {'legal@end':>9}"
    )
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        for r in rows:
            lines.append(
                f"{name:<8} {r['episode']:>3} {r['steps']:>5} "
                f"{r['usable']:>6} {r['holdout_fires']:>6} "
                f"{r['holdout_paid']:>5} {r['reward']:>9} "
                f"{r['legal_start']:>8} {r['legal_end']:>9}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the depth probe on the real board and print the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    table = depth_probe(
        episodes=args.episodes, budget=args.budget, seed=args.seed
    )
    print(  # stdout-compat
        "== arena depth probe — construction board =="
    )
    print(_fmt_probe(table))  # stdout-compat
    if args.json:
        from pathlib import Path

        Path(args.json).write_text(json.dumps(table, indent=2) + "\n")
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
