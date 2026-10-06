"""The meta-arena — one program, a live e-graph, the mixed move set.

Plan 0021 (``project/plans/0021-meta-arena.md``) unifies the two
measured boards: the law-order game — *deep for schedule, shallow
for quality* (~350 orderings, one extracted term) — and the
construction arena's deep object game.  The board is *a program
being optimized*: a term plus its live :class:`EGraph` under an
enode budget, and one action space mixes the levers:

* ``fire(rule)`` — apply one named law once: the apply-law move,
  the measured schedule dimension (a *cost* lever — order spends
  board, it does not change the reachable answer);
* ``saturate(rules, budget)`` — a bounded closure step: the old
  game's whole move, demoted to one option;
* ``declare(construction)`` — run a construction action
  (``object_synthesis`` fold / lift / compose / specialize /
  relax_guard on corpus-derived candidates) and insert the result
  into the live ruleset *mid-search* — the move that can change
  what the board reaches (the plan's Q2).  When the constructed RHS
  is a faithful abbreviation — a bare declared op over all of the
  LHS's metavariables — the definitional *unfold* pair rides along
  (the ``opdata`` semantics: a declared op *is* its expansion);
* ``extract()`` — terminal: extract the cheapest member, build and
  replay its certificate, score the cost delta vs baseline.

The referee is unchanged (ADR 0003): ``verify_certificate`` replays
every extracted program — a declared rule that fired sits in the
certificate's rule table.  A terminal move whose certificate fails
scores nothing.  One caveat stays visible: a fold to an
*already-named* op (``div(x,|x|+1) → softsign(x)``) is a *claim* —
the certificate certifies the derivation, not the claim (the
gauntlet that rules on claims lives in ``arena``/``evidence`` and
is deliberately not wired in).  Folds to fresh names are
definitions, not claims, and stay sound.

The honest columns: ``cost`` prices the extraction under
``cost_fn`` — a fresh declared op hits the table default
(1 flop/elem), so a fresh-name fold wins *by cost-model
construction*.  ``cost_unfolded`` expands every declared
abbreviation back and reprices — a real gain survives unfolding; an
unpriced-op gain collapses to baseline.  ``used_declared`` names
the declared kernels the extraction picked.

The reward is :data:`lawdata.META_ARENA_REWARD`, all data: an
applied non-terminal move pays ``step + enode·<enodes added>``;
``extract`` pays ``step`` and collects
``delta·(baseline - cost)/baseline``; a declined move scores 0 —
the construction arena's honest-refusal convention.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, overload

from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import (
    CertificateVerificationError,
    EGraph,
    Rewrite,
    verify_certificate,
)
from catopt_core.egraph.terms import _term_instantiate
from catopt_core.ir import Const, Op, op_repr
from catopt_core.laws import DEFAULT

from catopt_discovery import lawdata
from catopt_discovery import object_synthesis as obs

__all__ = [
    "ACTIONS",
    "Action",
    "GreedyPlayer",
    "MetaArena",
    "MetaState",
    "RandomPlayer",
    "ScriptedPlayer",
    "StepReport",
    "Trajectory",
    "declare_candidates",
    "legal_actions",
    "main",
    "meta_probe",
    "probe_table",
    "register_action",
    "run_episode",
]

# ---------------------------------------------------------------------------
#  Actions — the mixed move set as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Action:
    """One meta-move — a named operation plus its params map.

    ``op`` selects the registered handler (:data:`ACTIONS`,
    :func:`register_action`); ``params`` is plain data.  The
    classmethods are the shipped vocabulary.
    """

    op: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def fire(cls, rule: str) -> Action:
        """``fire(rule)`` — one application of the named live rule."""
        return cls("fire", {"rule": rule})

    @classmethod
    def saturate(
        cls,
        rules: Iterable[str] | None = None,
        *,
        budget: int | None = None,
        iterations: int = 8,
        stop: str = "fixed_point",
        patience: int = 3,
    ) -> Action:
        """``saturate(rules, budget)`` — a bounded closure step.

        *rules* ``None`` runs the whole live ruleset; a name tuple
        restricts the step.  *budget* bounds the NEW enodes each
        selected rule may add over the graph's lifetime (the
        engine's ``rule_budgets`` semantics); the rest forwards to
        :meth:`EGraph.run`.
        """
        return cls(
            "saturate",
            {
                "rules": None if rules is None else tuple(rules),
                "budget": budget,
                "iterations": iterations,
                "stop": stop,
                "patience": patience,
            },
        )

    @classmethod
    def declare(
        cls, construction: Any, *, unfold: bool = True
    ) -> Action:
        """``declare(construction)`` — mint a declared object mid-search.

        *construction* is a construction action as data — a
        ``{"op": "fold", "params": {...}}`` mapping or any object
        carrying ``.op`` / ``.params`` (``arena.Action`` duck-types).
        The ops are ``object_synthesis``'s: ``fold`` / ``lift`` /
        ``compose`` / ``specialize``.  *unfold*
        mints the definitional inverse when the constructed RHS is
        a faithful abbreviation.
        """
        return cls(
            "declare",
            {"construction": construction, "unfold": unfold},
        )

    @classmethod
    def extract(cls) -> Action:
        """``extract()`` — the terminal move: certify and score."""
        return cls("extract", {})


#: Handler contract — ``(arena, action) -> _Outcome``, mirroring
#: the construction arena's registry seam.
ActionHandler = Callable[["MetaArena", Action], "_Outcome"]

#: The action registry — ``op`` name → handler.
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
    """Register *fn* as the handler behind ``Action(op)``.

    ``register_action("fire", fn)`` or ``@register_action("x")`` —
    a new move kind is one handler function, no edits here.
    """

    def _bind(f: ActionHandler) -> ActionHandler:
        ACTIONS[op] = f
        return f

    return _bind(fn) if fn is not None else _bind


@dataclass(frozen=True)
class _Outcome:
    """What a move left on the board.

    ``rules`` are inserted into the live ruleset (``declare``);
    ``defs`` are ``(kernel_op, mvs, spelled)`` abbreviation records
    the ``cost_unfolded`` column reads.  ``applied=False`` is the
    honest refusal — move declined, reward 0.
    """

    applied: bool = True
    note: str = ""
    rules: tuple = ()
    defs: tuple = ()
    terminal: bool = False
    cost: float = float("nan")
    cost_unfolded: float = float("nan")
    certificate_ok: bool | None = None
    used_declared: tuple = ()
    detail: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
#  Term-spec helpers — the mined declare inventory (specs are data)
# ---------------------------------------------------------------------------


def _spec_of(term: Any, names: dict) -> Any:
    """Render *term* as spec data with canonical leaf metavariables.

    ``Op`` → ``(op, *child_specs[, attrs])``; ``Const`` → its value;
    other leaves → ``X1``, ``X2``, … by first-occurrence order —
    one *names* map per call, so each spec is canonicalized.
    """
    if isinstance(term, Op):
        parts = [_spec_of(a, names) for a in term.args]
        if term.attrs:
            parts.append(dict(term.attrs))
        return (term.op, *parts)
    if isinstance(term, Const):
        return term.value
    key = repr(term)
    if key not in names:
        names[key] = f"X{len(names) + 1}"
    return names[key]


def _spec_children(spec: Any) -> Iterable:
    """Yield the child specs of a compound spec (skip the attr dict)."""
    if isinstance(spec, (tuple, list)) and spec:
        for e in spec[1:]:
            if not isinstance(e, dict):
                yield e


def _spec_metavars(spec: Any) -> tuple:
    """Return the distinct metavariable leaves of *spec*, in order."""
    out: list[str] = []

    def walk(s: Any) -> None:
        if isinstance(s, str):
            if s not in out:
                out.append(s)
        elif isinstance(s, (tuple, list)):
            for e in _spec_children(s):
                walk(e)

    walk(spec)
    return tuple(out)


def _spec_ops(spec: Any) -> set:
    """Return the op names inside *spec*."""
    out: set[str] = set()
    if (
        isinstance(spec, (tuple, list))
        and spec
        and isinstance(spec[0], str)
    ):
        out.add(spec[0])
        for e in _spec_children(spec):
            out |= _spec_ops(e)
    return out


def _spec_size(spec: Any) -> int:
    """Return the number of op nodes in *spec*."""
    if (
        isinstance(spec, (tuple, list))
        and spec
        and isinstance(spec[0], str)
    ):
        return 1 + sum(_spec_size(e) for e in _spec_children(spec))
    return 0


def _composite_specs(corpus: Iterable, cap: int) -> tuple:
    """Distinct ≥2-op subterm specs over *corpus*, canonicalized."""
    seen: dict[str, Any] = {}

    def visit(t: Any) -> None:
        if isinstance(t, Op):
            spec = _spec_of(t, {})
            if _spec_size(spec) >= 2:
                seen.setdefault(repr(spec), spec)
            for a in t.args:
                visit(a)

    for term in corpus:
        visit(term)
    return tuple(seen[k] for k in sorted(seen)[:cap])


def _ops_of(term: Any, out: set | None = None) -> set:
    """Every op name appearing in *term*."""
    if out is None:
        out = set()
    if isinstance(term, Op):
        out.add(term.op)
        for a in term.args:
            _ops_of(a, out)
    return out


def _term_metavars(term: Any, out: set | None = None) -> set:
    """Collect the leaf metavariables (``str`` leaves) of a pattern."""
    if out is None:
        out = set()
    if isinstance(term, str):
        out.add(term)
    elif isinstance(term, Op):
        for a in term.args:
            _term_metavars(a, out)
    return out


def _action_name(params: dict, prefix: str) -> str:
    """Return the construction object name (declared or digest)."""
    given = params.get("name")
    if given:
        return str(given)
    digest = hashlib.sha256(
        repr(sorted(params.items(), key=lambda kv: kv[0])).encode()
    ).hexdigest()[:10]
    return f"{prefix}:{digest}"


# ---------------------------------------------------------------------------
#  Construction dispatch — object_synthesis ops over resolved refs
# ---------------------------------------------------------------------------


def _construction_parts(construction: Any) -> tuple[str, dict]:
    """Read ``(op, params)`` off a construction action — dict or attrs."""
    if isinstance(construction, dict):
        return (
            str(construction["op"]),
            dict(construction.get("params", {})),
        )
    return (
        str(construction.op),
        dict(getattr(construction, "params", {})),
    )


def _build_fold(arena: MetaArena, name: str, p: dict) -> Any:
    """``fold(spelled, kernel)`` — the spelled composite → its kernel."""
    return obs.fold_object(
        name,
        p["spelled"],
        p["kernel"],
        arg=p.get("arg", "X"),
        cond=p.get("cond"),
        dspec=p.get("dspec"),
        kind=p.get("kind", "abstraction"),
        tags=tuple(p.get("tags", ())),
    )


def _build_lift(arena: MetaArena, name: str, p: dict) -> Any:
    """``lift(step, carrier, apply_op)`` — a step → carrier form."""
    return obs.lift_object(
        name,
        p["step"],
        p["carrier"],
        p["apply_op"],
        state=p.get("state"),
        cond=p.get("cond"),
        dspec=p.get("dspec"),
        kind=p.get("kind", "abstraction"),
        tags=tuple(p.get("tags", ())),
    )


def _build_compose(arena: MetaArena, name: str, p: dict) -> Any:
    """``compose(first, *rest)`` — rewrite premise RHSs over live refs."""
    rules = []
    for ref in (p["first"], *p.get("rest", ())):
        r = arena.resolve_ref(ref)
        if r is None:
            raise ValueError(f"unknown premise {ref!r}")
        rules.append(r)
    return obs.compose_objects(
        name,
        *rules,
        specialize=dict(p.get("specialize", {})),
        cond=p.get("cond"),
        dspec=p.get("dspec"),
        kind=p.get("kind", "abstraction"),
        tags=tuple(p.get("tags", ())),
    )


def _resolve_one(arena: MetaArena, p: dict) -> Any:
    """Resolve a construction's ``ref`` param against the live ruleset."""
    rule = arena.resolve_ref(p["ref"])
    if rule is None:
        raise ValueError(f"unknown object {p['ref']!r}")
    return rule


def _build_specialize(arena: MetaArena, name: str, p: dict) -> Any:
    """``specialize(ref, binding)`` — narrow a live rule's family."""
    return obs.specialize(
        _resolve_one(arena, p), dict(p["binding"]), name=name
    )


#: The construction ops ``declare`` dispatches — name → builder.
_CONSTRUCTION: dict[str, Any] = {
    "fold": _build_fold,
    "lift": _build_lift,
    "compose": _build_compose,
    "specialize": _build_specialize,
}


def _unfold_for(obj: Any) -> tuple | None:
    """Mint the definitional inverse of a faithful abbreviation.

    ``(unfold_rule, (kernel_op, mvs, spelled))`` when the constructed
    ``rhs`` is a bare op over distinct metavariables that *cover* the
    LHS's metavariable set — then ``kernel(mvs) := spelled`` is a
    definition and both directions live.  ``None`` otherwise: a
    compound RHS or a metavar-dropping kernel is a claim.
    """
    rhs = obj.rule.rhs
    if not isinstance(rhs, Op):
        return None
    if not rhs.args or not all(isinstance(a, str) for a in rhs.args):
        return None
    if len(set(rhs.args)) != len(rhs.args):
        return None
    if not _term_metavars(obj.rule.lhs) <= set(rhs.args):
        return None
    unfold = Rewrite(
        name=f"{obj.rule.name}_unfold",
        lhs=rhs,
        rhs=obj.rule.lhs,
        law=f"definitional unfold of {obj.rule.name}: "
        f"{rhs.op} is the declared fold of its spelled form",
        tags=obj.rule.tags,
    )
    return unfold, (rhs.op, tuple(rhs.args), obj.rule.lhs)


# ---------------------------------------------------------------------------
#  The move handlers
# ---------------------------------------------------------------------------


@register_action("fire")
def _fire(arena: MetaArena, action: Action) -> _Outcome:
    """Apply one named live rule once, then rebuild congruence."""
    name = str(action.params["rule"])
    rule = arena.resolve_ref(name)
    if rule is None:
        return _Outcome(
            applied=False, note=f"fire: unknown rule {name!r}"
        )
    remaining = arena.max_nodes - arena.eg.n_enodes
    if remaining <= 0:
        return _Outcome(
            applied=False, note="fire: enode budget exhausted"
        )
    changed = arena.eg.apply_rule(
        rule, arena.root, enode_budget=remaining
    )
    arena.eg.rebuild()
    return _Outcome(
        note=f"fire {name}: {'changed' if changed else 'no change'}"
    )


@register_action("saturate")
def _saturate(arena: MetaArena, action: Action) -> _Outcome:
    """Run a bounded closure step over the selected live rules."""
    p = action.params
    sel, missing = arena.select_rules(p.get("rules"))
    if missing:
        return _Outcome(
            applied=False,
            note=f"saturate: unknown rules {sorted(missing)!r}",
        )
    budget = p.get("budget")
    budgets = (
        None if budget is None else {r.name: int(budget) for r in sel}
    )
    stats = arena.eg.run(
        sel,
        arena.root,
        max_iterations=int(p.get("iterations", 8)),
        max_nodes=arena.max_nodes,
        rule_budgets=budgets,
        stop=p.get("stop", "fixed_point"),
        patience=int(p.get("patience", 3)),
        cost_fn=arena.cost_fn,
    )
    return _Outcome(
        note=(
            f"saturate[{len(sel)}]: {stats['stop']} "
            f"it={stats['iterations']}"
        ),
        detail={
            "stop": stats["stop"],
            "iterations": stats["iterations"],
        },
    )


@register_action("declare")
def _declare(arena: MetaArena, action: Action) -> _Outcome:
    """Construct an object and offer it for the live ruleset."""
    p = action.params
    cop, cp = _construction_parts(p["construction"])
    builder = _CONSTRUCTION.get(cop)
    if builder is None:
        return _Outcome(
            applied=False, note=f"unknown construction {cop!r}"
        )
    try:
        obj = builder(arena, _action_name(cp, cop), cp)
    except Exception as exc:
        return _Outcome(
            applied=False,
            note=f"{cop} declined: {type(exc).__name__}: {exc}",
        )
    if obj is None:
        return _Outcome(
            applied=False, note=f"{cop} declined: constructor refused"
        )
    rules: list[Any] = [obj.rule]
    defs: list = []
    if p.get("unfold", True):
        pair = _unfold_for(obj)
        if pair is not None:
            rules.append(pair[0])
            defs.append(pair[1])
    return _Outcome(
        note=f"declare {cop} → {obj.rule.name}",
        rules=tuple(rules),
        defs=tuple(defs),
        detail={
            "kind": obj.kind,
            "construction": tuple(obj.construction),
        },
    )


@register_action("extract")
def _extract(arena: MetaArena, action: Action) -> _Outcome:
    """Terminal: extract, certify the derivation, score the delta."""
    best = arena.eg.extract_best(arena.root, arena.cost_fn)
    if best is None:
        return _Outcome(
            applied=False, terminal=True, note="extract: no member"
        )
    cost = float(dag_cost(best, arena.cost_fn))
    ok = False
    note = ""
    try:
        cert = arena.eg.certificate(
            arena.term, best, root_eid=arena.root, cost_fn=arena.cost_fn
        )
        replayed = verify_certificate(arena.term, cert)
        ok = op_repr(replayed) == op_repr(best)
        if not ok:
            note = "replay mismatch"
    except CertificateVerificationError as exc:
        note = f"certificate rejected: {exc}"
    except Exception as exc:
        note = f"certification failed: {type(exc).__name__}: {exc}"
    unfolded = arena.unfold(best)
    used = tuple(sorted(_ops_of(best) & set(arena.definitions)))
    return _Outcome(
        terminal=True,
        cost=cost,
        cost_unfolded=float(dag_cost(unfolded, arena.cost_fn)),
        certificate_ok=ok,
        used_declared=used,
        note=f"extract → {cost:.6g}" + ("" if ok else f" ({note})"),
        detail={
            "extracted": op_repr(best),
            "unfolded": op_repr(unfolded),
        },
    )


# ---------------------------------------------------------------------------
#  The board
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetaState:
    """The read-only board view a player decides on.

    ``rules`` is the live ruleset (base + declared, in play order);
    ``specs`` the mined composite-subterm inventory the ``declare``
    enumerator reads — pure spec data, no term objects cross.
    ``best_cost`` is the current extraction's price.
    """

    rules: tuple[str, ...]
    declared: tuple[str, ...]
    n_enodes: int
    n_classes: int
    rule_fires: tuple[tuple[str, int], ...]
    steps: int
    done: bool
    best_cost: float
    baseline_cost: float
    budget_left: int
    specs: tuple


@dataclass(frozen=True)
class StepReport:
    """One move's accounting — the reward decomposition.

    ``enodes_delta`` is what the move spent of the board's budget;
    ``inserted`` the rule names a declare added.  The terminal
    fields (``cost`` / ``cost_unfolded`` / ``certificate_ok`` /
    ``used_declared``) are populated on ``extract``.
    """

    action: Action
    applied: bool
    note: str
    reward: float
    enodes_delta: int = 0
    terminal: bool = False
    cost: float = float("nan")
    cost_unfolded: float = float("nan")
    certificate_ok: bool | None = None
    used_declared: tuple[str, ...] = ()
    inserted: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)


class MetaArena:
    """The board — a term, its live e-graph, and the live ruleset.

    *term* is the program being optimized; *rules* the initial
    library (:data:`catopt_core.laws.DEFAULT` when omitted — the
    live set is a list ``declare`` appends to mid-episode); *corpus*
    the extra terms the ``declare`` inventory mines (the board term
    itself when omitted); *cost_fn* prices extraction
    (``flops_cost`` default); *max_nodes* the enode budget every
    growth move spends against; *max_specs* caps the mined
    ``declare`` inventory; *weights* overrides
    :data:`lawdata.META_ARENA_REWARD`.
    """

    def __init__(
        self,
        term: Any,
        rules: Any = None,
        *,
        corpus: Iterable = (),
        cost_fn: Any = None,
        max_nodes: int = 20_000,
        max_specs: int = 32,
        weights: dict | None = None,
    ) -> None:
        """Bind the program, the library, and the budget."""
        self.term = term
        self.eg = EGraph()
        self.root = self.eg.add_term(term)
        base = DEFAULT if rules is None else rules
        self.rules = list(base)
        self._by_name = {r.name: r for r in self.rules}
        self.declared: dict[str, Rewrite] = {}
        self.definitions: dict[str, tuple] = {}
        self.corpus = tuple(corpus) or (term,)
        self.cost_fn = cost_fn if cost_fn is not None else flops_cost
        self.max_nodes = int(max_nodes)
        self.weights = {**lawdata.META_ARENA_REWARD, **(weights or {})}
        self.baseline_cost = float(dag_cost(term, self.cost_fn))
        self.steps = 0
        self.done = False
        self._specs = _composite_specs(self.corpus, max_specs)

    def resolve_ref(self, ref: Any) -> Any:
        """Resolve a rule reference against the *live* ruleset.

        ``Rewrite`` / ``ConstructedObject`` pass through; a name
        looks up base and declared rules alike — the premise
        vocabulary compose/specialize cite.
        """
        if isinstance(ref, Rewrite):
            return ref
        if isinstance(ref, obs.ConstructedObject):
            return ref.rule
        if not isinstance(ref, str):
            return None
        return self._by_name.get(ref)

    def select_rules(self, names: Any) -> tuple[list, tuple]:
        """Resolve a ``saturate`` rules param to (rules, missing)."""
        if names is None:
            return list(self.rules), ()
        out: list = []
        missing: list = []
        for n in names:
            r = self._by_name.get(str(n))
            if r is None:
                missing.append(str(n))
            else:
                out.append(r)
        return out, tuple(missing)

    def _status_of(self, rule: Rewrite) -> str:
        """Classify *rule* against the live set — no mutation."""
        old = self._by_name.get(rule.name)
        if old is None:
            return "new"
        same = op_repr(old.lhs) == op_repr(rule.lhs) and op_repr(
            old.rhs
        ) == op_repr(rule.rhs)
        return "same" if same else "conflict"

    def _insert(self, rule: Rewrite) -> None:
        """Add *rule* to the live set (caller checked the status)."""
        self.rules.append(rule)
        self._by_name[rule.name] = rule
        self.declared[rule.name] = rule

    def unfold(self, term: Any) -> Any:
        """Expand every declared abbreviation back to its spelled form.

        The honest-cost view: a ``foldabs(x)`` the extraction picked
        becomes the composite it denotes; non-declared ops pass
        through (``sub`` stays ``sub`` — a *real* op, priced as
        itself; the definition is only recorded for honest
        accounting of fresh names).
        """
        if isinstance(term, Op):
            args = tuple(self.unfold(a) for a in term.args)
            if term.op in self.definitions:
                mvs, spelled = self.definitions[term.op]
                return _term_instantiate(
                    spelled, dict(zip(mvs, args, strict=True))
                )
            return Op.make(term.op, *args, **dict(term.attrs))
        return term

    def observe(self) -> MetaState:
        """Build the read-only snapshot — live ruleset and board size."""
        best = self.eg.extract_best(self.root, self.cost_fn)
        best_cost = (
            float(dag_cost(best, self.cost_fn))
            if best is not None
            else float("inf")
        )
        return MetaState(
            rules=tuple(r.name for r in self.rules),
            declared=tuple(sorted(self.declared)),
            n_enodes=self.eg.n_enodes,
            n_classes=self.eg.n_classes,
            rule_fires=tuple(sorted(self.eg.rule_fires.items())),
            steps=self.steps,
            done=self.done,
            best_cost=best_cost,
            baseline_cost=self.baseline_cost,
            budget_left=max(0, self.max_nodes - self.eg.n_enodes),
            specs=self._specs,
        )

    def _accept(self, out: _Outcome) -> tuple[_Outcome, tuple]:
        """Insert a declare's rules, two-phase; return (out', names).

        A name bound to a *different* rule declines the whole
        declare — no partial insertion.
        """
        statuses = [(r, self._status_of(r)) for r in out.rules]
        conflict = next(
            (r.name for r, s in statuses if s == "conflict"), ""
        )
        if conflict:
            return (
                _Outcome(
                    applied=False,
                    note=(
                        f"declare: name {conflict!r} is bound to a "
                        "different rule"
                    ),
                    terminal=out.terminal,
                ),
                (),
            )
        inserted = tuple(r.name for r, s in statuses if s == "new")
        for r, s in statuses:
            if s == "new":
                self._insert(r)
        for op_name, mvs, spelled in out.defs:
            self.definitions[op_name] = (mvs, spelled)
        return out, inserted

    def _reward(self, out: _Outcome, spent: int) -> float:
        """Compose the step's reward from the weight table."""
        w = self.weights
        if not out.applied:
            return 0.0
        if out.terminal:
            if not out.certificate_ok:
                # the referee refused: spent the move, no payout
                return -w["step"]
            base = self.baseline_cost
            rel = (base - out.cost) / base if base > 0 else 0.0
            return w["delta"] * rel - w["step"]
        return -(w["step"] + w["enode"] * spent)

    def step(self, action: Action) -> tuple[MetaState, StepReport]:
        """Apply *action*, referee, return state' + report.

        ``fire`` / ``saturate`` spend board (enodes) for reach;
        ``declare`` inserts the constructed rule — plus its
        definitional unfold when faithful — a name collision on a
        *different* rule being an honest refusal; ``extract`` ends
        the episode under certificate.  The reward composition is
        :data:`lawdata.META_ARENA_REWARD`: declined moves score 0,
        applied non-terminal moves pay ``step + enode·spent``, a
        certified extract pays ``step`` and collects
        ``delta·(baseline - cost)/baseline``.
        """
        if self.done:
            raise RuntimeError(
                "episode finished — extract() is terminal"
            )
        fn = ACTIONS.get(action.op)
        if fn is None:
            raise KeyError(
                f"unregistered meta-move {action.op!r} "
                f"(registered: {sorted(ACTIONS)})"
            )
        before = self.eg.n_enodes
        out = fn(self, action)
        spent = self.eg.n_enodes - before
        inserted: tuple = ()
        if out.applied and out.rules:
            out, inserted = self._accept(out)
        reward = self._reward(out, spent)
        self.done = self.done or out.terminal
        self.steps += 1
        return self.observe(), StepReport(
            action=action,
            applied=out.applied,
            note=out.note,
            reward=reward,
            enodes_delta=spent,
            terminal=out.terminal,
            cost=out.cost,
            cost_unfolded=out.cost_unfolded,
            certificate_ok=out.certificate_ok,
            used_declared=out.used_declared,
            inserted=tuple(inserted),
            detail=out.detail,
        )


# ---------------------------------------------------------------------------
#  Legal moves — what the board affords a player
# ---------------------------------------------------------------------------


def _fresh_fold(i: int, spec: Any) -> Action:
    """Mint the sound-by-definition declare: spec → fresh op."""
    kernel = f"foldabs_{i}"
    return Action.declare(
        {
            "op": "fold",
            "params": {
                "name": kernel,
                "spelled": spec,
                "kernel": (kernel, *_spec_metavars(spec)),
            },
        }
    )


def legal_actions(state: MetaState) -> tuple[Action, ...]:
    """Enumerate the moves *state* affords, deterministically.

    ``fire`` once per live rule (the whole schedule dimension);
    ``saturate`` once per budget arm in
    :data:`lawdata.META_SATURATE_BUDGETS`; ``declare`` once per mined
    composite spec, folded to the fresh declared op the move names —
    the enumerator only offers *sound-by-definition* candidates
    (existing-kernel folds are claims: playbook-reachable, not
    enumerated); ``extract`` last.
    """
    if state.done:
        return ()
    out = [Action.fire(n) for n in state.rules]
    out += [
        Action.saturate(budget=b) for b in lawdata.META_SATURATE_BUDGETS
    ]
    out += [_fresh_fold(i, s) for i, s in enumerate(state.specs)]
    out.append(Action.extract())
    return tuple(out)


def declare_candidates(
    arena: MetaArena, extra_kernels: Iterable = ()
) -> list[Action]:
    """Return the probe's hypothesis set — fresh folds + claims.

    Per mined composite spec: the fresh-name fold (sound by
    definition), then one fold per kernel in *extra_kernels* — the
    caller's hypothesis vocabulary.  Kernel folds are *claims*
    (module docstring — the certificate certifies the derivation,
    not the claim), so the enumerator does not invent them: a
    kernel vocabulary mined from every rule pattern mostly mints
    nonsense (``add(x,-y) → aff(x,y)``), and an unrefereed junk claim
    can win the probe on the unpriced-op artifact alone.
    """
    kernels = set(extra_kernels)
    state = arena.observe()
    out: list[Action] = []
    for i, spec in enumerate(state.specs):
        mvs = _spec_metavars(spec)
        out.append(_fresh_fold(i, spec))
        for k in sorted(kernels - _spec_ops(spec)):
            out.append(
                Action.declare(
                    {
                        "op": "fold",
                        "params": {
                            "name": f"fold_{k}_{i}",
                            "spelled": spec,
                            "kernel": (k, *mvs),
                        },
                    }
                )
            )
    return out


# ---------------------------------------------------------------------------
#  Players and the episode driver
# ---------------------------------------------------------------------------


def _hashable_param(v: Any) -> Any:
    """Hashable form of a param value — dicts → frozensets of items."""
    if isinstance(v, dict):
        return frozenset((k, _hashable_param(x)) for k, x in v.items())
    if isinstance(v, (tuple, list)):
        return tuple(_hashable_param(x) for x in v)
    return v


def _action_key(action: Action) -> Any:
    """Return a stable dedup key for an enumerated move."""
    return (action.op, _hashable_param(action.params))


class ScriptedPlayer:
    """A fixed playbook — ``saturate → extract`` by default.

    The human-known-good order every other arm must beat: one
    bounded closure step over the live ruleset, then the certified
    terminal move.  *moves* is an iterable of ``Action``s; ``None``
    ends the episode once drained.
    """

    def __init__(self, moves: Iterable[Action] | None = None) -> None:
        """Bind the playbook (``None`` = saturate-then-extract)."""
        default = [Action.saturate(), Action.extract()]
        self._moves = deque(default if moves is None else moves)

    def __call__(self, state: MetaState) -> Action | None:
        """Return the next scripted move, or ``None`` when drained."""
        return self._moves.popleft() if self._moves else None


class RandomPlayer:
    """Uniform play over the live legal set — the no-knowledge arm.

    Seeded — runs replay.  *legal* is the enumerator seam: any
    ``MetaState -> tuple[Action, ...]`` callable plugs in.
    """

    def __init__(self, rng: random.Random, legal: Any = None) -> None:
        """Bind the PRNG and the legal-move enumerator."""
        self.rng = rng
        self._legal = legal_actions if legal is None else legal

    def __call__(self, state: MetaState) -> Action | None:
        """Sample uniformly over the live move set."""
        acts = self._legal(state)
        if not acts:
            return None
        return acts[self.rng.randrange(len(acts))]


class GreedyPlayer:
    """Cheapest-immediate play — take the payout the moment it exists.

    ``extract`` scores ``delta·(baseline - best)/baseline - step`` —
    the terminal payout measured *now*; ``saturate`` scores
    ``-(step + enode·est)`` with *est* the named budget (or the
    ``saturate_est`` weight when unbounded); ``fire`` / ``declare``
    score ``-step`` — cheap, payoff statically unknown.  Played
    moves are skipped (a frontier, not a loop), so the player
    grinds the fire list, tries the free declarations, and cashes
    out the moment the extraction beats the baseline.
    """

    def __init__(
        self, legal: Any = None, weights: dict | None = None
    ) -> None:
        """Bind the enumerator and the reward table."""
        self._legal = legal_actions if legal is None else legal
        self._weights = {**lawdata.META_ARENA_REWARD, **(weights or {})}
        self._played: set = set()

    def _score(self, state: MetaState, action: Action) -> float:
        """Estimate the immediate reward of one move, statically."""
        w = self._weights
        if action.op == "extract":
            base = state.baseline_cost
            rel = (base - state.best_cost) / base if base > 0 else 0.0
            return w["delta"] * rel - w["step"]
        if action.op == "saturate":
            est = action.params.get("budget")
            est = w["saturate_est"] if est is None else float(est)
            return -(w["step"] + w["enode"] * est)
        return -w["step"]

    def __call__(self, state: MetaState) -> Action | None:
        """Return the highest-scoring unplayed move, or ``None``."""
        fresh = [
            a
            for a in self._legal(state)
            if _action_key(a) not in self._played
        ]
        pick = max(
            fresh, key=lambda a: self._score(state, a), default=None
        )
        if pick is not None:
            self._played.add(_action_key(pick))
        return pick


@dataclass
class Trajectory:
    """One episode's step reports, in play order.

    ``terminal`` is the ``extract`` report when the episode ended
    under certification — its ``cost`` / ``cost_unfolded`` /
    ``certificate_ok`` are the Q2 columns.
    """

    reports: list[StepReport]

    @property
    def total(self) -> float:
        """The episode's total immediate reward."""
        return sum(r.reward for r in self.reports)

    @property
    def terminal(self) -> StepReport | None:
        """The terminal report, or ``None`` (episode ran out)."""
        if self.reports and self.reports[-1].terminal:
            return self.reports[-1]
        return None

    @property
    def cost(self) -> float | None:
        """The extracted program's cost, when certified."""
        t = self.terminal
        return t.cost if t is not None else None

    @property
    def cost_unfolded(self) -> float | None:
        """The extraction's cost with declared ops expanded."""
        t = self.terminal
        return t.cost_unfolded if t is not None else None

    @property
    def certificate_ok(self) -> bool | None:
        """The referee's verdict on the extracted program."""
        t = self.terminal
        return t.certificate_ok if t is not None else None

    @property
    def declared_used(self) -> tuple:
        """Declared kernel ops the extracted program picked."""
        t = self.terminal
        return t.used_declared if t is not None else ()


def run_episode(
    arena: MetaArena,
    player: Callable[[MetaState], Action | None],
    budget: int,
) -> Trajectory:
    """Play *player* on *arena* until *budget* actions are spent.

    A player is any callable ``MetaState -> Action | None`` —
    ``None`` ends the episode early.  The budget counts actions:
    every step costs at most one move plus the enodes it spent.
    """
    reports: list[StepReport] = []
    for _ in range(budget):
        state = arena.observe()
        if state.done:
            break
        action = player(state)
        if action is None:
            break
        _state, rep = arena.step(action)
        reports.append(rep)
    return Trajectory(reports)


# ---------------------------------------------------------------------------
#  The Q2 probe — does mid-search abstraction reach what laws don't?
# ---------------------------------------------------------------------------


def _arm_row(traj: Trajectory, baseline: float) -> dict:
    """One player's probe row — cost, delta, certificate, spend."""
    t = traj.terminal
    return {
        "steps": len(traj.reports),
        "cost": t.cost if t is not None else None,
        "unfolded": t.cost_unfolded if t is not None else None,
        "delta": baseline - t.cost if t is not None else None,
        "cert": t.certificate_ok if t is not None else None,
        "reward": round(traj.total, 4),
        "used_declared": t.used_declared if t is not None else (),
        "inserted": tuple(n for r in traj.reports for n in r.inserted),
    }


def meta_probe(
    cases: list,
    rules: Any = None,
    *,
    cost_fn: Any = None,
    seed: int = 0,
    budget: int = 160,
    max_nodes: int = 20_000,
    declare_limit: int = 24,
    extra_kernels: Iterable = (),
) -> list[dict]:
    """Play the arms on 3-5 corpus terms; answer Q2, honestly.

    Per case ``(name, term[, rules[, declares]])`` on a fresh board:

    * ``scripted`` — the human baseline, saturate-then-extract;
    * ``greedy`` / ``random`` — the live legal set under the two
      non-scripted players;
    * ``declare`` — for each candidate (explicit *declares* + mined
      fresh folds + spec/kernel claims, capped at *declare_limit*) a
      scripted ``declare → saturate → extract`` episode; the winner
      is the cheapest certified extraction.  ``unfolded`` is the
      honest price — a gain that collapses to baseline under
      unfolding is a cost-model artifact, not a reached program.
    """
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)
    cf = cost_fn if cost_fn is not None else flops_cost
    rows: list[dict] = []
    for ci, case in enumerate(cases):
        name, term = case[0], case[1]
        rset = case[2] if len(case) > 2 else (rules or DEFAULT)
        decls = tuple(case[3]) if len(case) > 3 else ()

        def board(t: Any = term, rs: Any = rset) -> MetaArena:
            return MetaArena(
                t, rs, corpus=[t], cost_fn=cf, max_nodes=max_nodes
            )

        base = board().baseline_cost
        arms = {
            "scripted": run_episode(board(), ScriptedPlayer(), budget),
            "greedy": run_episode(board(), GreedyPlayer(), budget),
            "random": run_episode(
                board(), RandomPlayer(random.Random(seed + ci)), budget
            ),
        }
        cands = [
            *[Action.declare(d) for d in decls],
            *declare_candidates(board(), extra_kernels=extra_kernels),
        ]
        rows.append(
            {
                "name": name,
                "baseline": base,
                "declare_tried": min(len(cands), declare_limit),
                "arms": {k: _arm_row(t, base) for k, t in arms.items()},
                "declare": _declare_best(board, cands, declare_limit),
            }
        )
    return rows


def _declare_best(
    board: Callable[[], MetaArena], cands: list, limit: int
) -> dict | None:
    """Try each declare-then-search episode; keep the cheapest win."""
    best: dict | None = None
    for cand in cands[:limit]:
        traj = run_episode(
            board(),
            ScriptedPlayer([cand, Action.saturate(), Action.extract()]),
            3,
        )
        t = traj.terminal
        if (
            t is not None
            and t.certificate_ok
            and (best is None or t.cost < best["cost"])
        ):
            best = {
                "cost": t.cost,
                "unfolded": t.cost_unfolded,
                "inserted": tuple(
                    n for r in traj.reports for n in r.inserted
                ),
                "used_declared": t.used_declared,
                "cert": t.certificate_ok,
            }
    return best


def probe_table(rows: list[dict]) -> str:
    """Render the probe rows — the Q2 table the retro ships."""
    head = (
        f"{'case':<14} {'base':>8} {'script':>8} {'greedy':>8} "
        f"{'random':>8} {'declare':>8} {'unfold':>8} {'cert':>5}  winner"
    )
    lines = [head, "-" * len(head)]

    def _c(v: Any) -> str:
        return f"{v:8.3f}" if isinstance(v, float) else f"{'-':>8}"

    for r in rows:
        a = r["arms"]
        d = r["declare"]
        ok = all(v["cert"] is not False for v in a.values()) and (
            d is None or d["cert"]
        )
        win = ",".join(d["inserted"]) if d and d["inserted"] else "-"
        lines.append(
            f"{r['name']:<14} {_c(r['baseline'])}"
            f"{_c(a['scripted']['cost'])}{_c(a['greedy']['cost'])}"
            f"{_c(a['random']['cost'])}{_c(d['cost'] if d else None)}"
            f"{_c(d['unfolded'] if d else None)}{ok!s:>5}  {win}"
        )
    return "\n".join(lines)


def _demo_cases() -> list:
    """Return the probe's 3-5 term corpus — the fold sites Q2 asks.

    The activation-fold sites withhold the shipped fold
    (``DEFAULT - {silu_fold, softsign_fold}``) so the declaration is
    the only path back; ``sub_gap`` runs the full ``DEFAULT`` — no
    shipped law spells ``add(x, -y)`` as ``sub`` at all; ``control``
    offers no fold site.
    """
    from catopt_core.ir import TensorType, Var

    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
    silu_decl = {
        "op": "fold",
        "params": {
            "name": "silu_decl",
            "spelled": ("mul", "X", ("sigmoid", "X")),
            "kernel": "silu",
            "arg": "X",
        },
    }
    soft_decl = {
        "op": "fold",
        "params": {
            "name": "softsign_decl",
            "spelled": ("div", "X", ("add", ("abs", "X"), 1)),
            "kernel": "softsign",
            "arg": "X",
        },
    }
    sub_decl = {
        "op": "fold",
        "params": {
            "name": "sub_decl",
            "spelled": ("add", "X", ("neg", "Y")),
            "kernel": ("sub", "X", "Y"),
        },
    }
    return [
        (
            "silu_site",
            Op.make("mul", x, Op.make("sigmoid", x)),
            sans,
            (silu_decl,),
        ),
        (
            "softsign_site",
            Op.make(
                "div", x, Op.make("add", Op.make("abs", x), Const(1))
            ),
            sans,
            (soft_decl,),
        ),
        (
            "sub_gap",
            Op.make("add", x, Op.make("neg", y)),
            DEFAULT,
            (sub_decl,),
        ),
        (
            "nested_site",
            Op.make(
                "tanh",
                Op.make(
                    "div",
                    x,
                    Op.make("add", Op.make("abs", x), Const(1)),
                ),
            ),
            sans,
            (soft_decl,),
        ),
        (
            "control",
            Op.make(
                "add", Op.make("mul", x, y), Op.make("matmul", x, y)
            ),
            DEFAULT,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the Q2 probe on the demo corpus; print the table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget", type=int, default=160)
    p.add_argument("--declare-limit", type=int, default=24)
    p.add_argument("--max-nodes", type=int, default=20_000)
    args = p.parse_args(argv)
    rows = meta_probe(
        _demo_cases(),
        seed=args.seed,
        budget=args.budget,
        declare_limit=args.declare_limit,
        max_nodes=args.max_nodes,
    )
    print(  # stdout-compat
        "== meta-arena Q2 probe — declare-then-search vs search-only =="
    )
    print(probe_table(rows))  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
