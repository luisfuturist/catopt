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
move) and :class:`RandomPlayer` (uniform over the live move
vocabulary) are the immediately-playable baselines.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, overload

from catopt_core.egraph import Rewrite
from catopt_core.ir import Op, op_repr, op_repr_dag

from catopt_discovery import evidence as ev
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
    "ObjectView",
    "ProposalView",
    "RandomPlayer",
    "StageLine",
    "StepReport",
    "Trajectory",
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
    """One working-corpus case, as data the player may read."""

    source: str
    name: str
    nodes: int
    ops: tuple
    term: str


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
    ``guard_fires`` maps a stored object's alpha key to the total
    guarded sites its sweeps accepted — how wide its firing region
    measured.  ``to_dict`` renders the whole snapshot JSON-safe.
    """

    corpus: tuple[CaseView, ...]
    objects: tuple[ObjectView, ...]
    proposals: tuple[ProposalView, ...]
    pending: tuple[str, ...]
    premises: tuple[str, ...]
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
#: firing/pay columns (the same ``0.2·fires + 2·paid`` shape the
#: meta-game's referee scored).
_STAGE_CREDIT = 1.0
_USABLE_BONUS = 2.0
_FIRE_CREDIT = 0.2
_PAY_CREDIT = 2.0


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
    )


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


def _bare_object_view(row: dict, record: dict) -> ObjectView:
    """Render a stored object that never faced the gauntlet."""
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
    )


def _reported_object_view(
    row: dict, record: dict, rep: ev.Gauntlet
) -> ObjectView:
    """Render a stored object plus its last gauntlet report."""
    fires, paid = _holdout_counts(rep)
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
    when omitted); *meta* the evidence scope (built over the
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
        sink: Any = None,
        cost_fn: Any = None,
        meta: dict | None = None,
    ) -> None:
        """Bind the splits, the store, the backend and the scope."""
        if base_rules is None:
            from catopt_core.laws import ALL_RULES

            base_rules = tuple(ALL_RULES)
        if sink is None:
            from catopt_discovery.shape_proposal import _sink

            sink = _sink()
        if cost_fn is None:
            from catopt_discovery.impact import _cost_fn

            cost_fn = _cost_fn(sink)
        self.conn = conn if conn is not None else ev.connect(":memory:")
        self._working = list(working)
        self._holdout = tuple(holdout)
        self._pending: dict[str, TermCase] = (
            {str(k): v for k, v in pending.items()}
            if isinstance(pending, dict)
            else {c.name: c for c in pending}
        )
        self._base = tuple(base_rules)
        self._sink = sink
        self._cost_fn = cost_fn
        self._census = _census_of(self._working)
        self.reports: dict[str, ev.Gauntlet] = {}
        self.epoch = 0
        self.steps = 0
        self._meta = {
            "corpus_hash": self._scope_hash(),
            "rules_hash": ev.rules_hash(
                repr(r) for r in self._library_keys()
            ),
            "run_id": ev.new_run_id(),
            "holdout": "",
            "ts": ev.now(),
            **(meta or {}),
        }
        if "code_rev" not in self._meta:
            # lazy: the git probe only runs when no override supplied one
            self._meta["code_rev"] = ev.code_rev()

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
            _STAGE_CREDIT * cleared
            + _USABLE_BONUS * usable
            + _FIRE_CREDIT * fires
            + _PAY_CREDIT * paid
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
    """Uniform play over the live move vocabulary — the no-knowledge arm.

    Each call picks uniformly among three live move kinds: an
    unplayed *moves* library entry (the same construction pool the
    fixed rule replays — random *order*, same vocabulary), an
    ``ingest`` of a random pending subset, and an ``auto_cond`` of a
    random refused object.  The player is seeded — runs replay.
    """

    def __init__(
        self,
        rng: random.Random,
        moves: Iterable[Action] = (),
        *,
        ingest: bool = True,
        guard: bool = True,
    ) -> None:
        """Bind the PRNG, the move library and the live move kinds."""
        self.rng = rng
        self._moves = list(moves)
        self._ingest = ingest
        self._guard = guard
        self._tried: set = set()

    def _live_kinds(
        self, state: ArenaState, refused: list
    ) -> list[str]:
        """Return the move kinds with a live move on this board."""
        kinds: list[str] = []
        if self._moves:
            kinds.append("move")
        if self._ingest and state.pending:
            kinds.append("ingest")
        if self._guard and refused:
            kinds.append("auto_cond")
        return kinds

    def _pick(
        self, kind: str, state: ArenaState, refused: list
    ) -> Action:
        """Sample one move inside the chosen kind."""
        if kind == "move":
            return self._moves.pop(self.rng.randrange(len(self._moves)))
        if kind == "ingest":
            k = self.rng.randint(1, len(state.pending))
            return Action.ingest(
                tuple(self.rng.sample(sorted(state.pending), k))
            )
        target = self.rng.choice(refused)
        self._tried.add(target.alpha_key)
        return Action.auto_cond(target.alpha_key)

    def __call__(self, state: ArenaState) -> Action | None:
        """Sample a live move kind, then a move inside it."""
        refused = [
            o
            for o in state.objects
            if o.usable is False and o.alpha_key not in self._tried
        ]
        kinds = self._live_kinds(state, refused)
        if not kinds:
            return None
        return self._pick(self.rng.choice(kinds), state, refused)


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
