"""Object synthesis — the *construction* half of ADR 0004.

The discovery pipeline (census → propose → verify → measure) *finds*
candidate equalities in the corpus: every candidate it emits is a
pattern observed in — or enumerated over — existing programs.  ADR
0004 names the complementary action space: the machine *constructs*
objects, it does not only find them.  This module is the honest seed
of that space — three construction operations over already-admitted
material:

* :func:`fold_object` — **fold representation**: a spelled-out
  composition folded into one dispatched kernel
  (``div(x, |x|+1) → softsign(x)``) — the same shape every shipped
  machine-found fold law takes.
* :func:`lift_object` — **introduce abstraction**: a carrier-style
  object declared as data — a spelled-out program step lifted into a
  runtime carrier (``add(matmul(A,h),x) → apply(aff(A,x),h)`` — the
  affine-scan monoid; ``softmax(s)·v → om_apply(om_elem(s,v))`` — the
  online-softmax monoid).
* :func:`compose_objects` — **compose abstractions**: rewrite a
  premise's RHS by a second premise at pattern level, producing the
  composite object (``compose(aff_lift, aff_lift)`` — the two-step
  scan lift — exists in no shipped ruleset).

Each operation returns a :class:`ConstructedObject`: the ``Rewrite``
plus its provenance — the construction trace and the premise names.
:func:`store_constructed` persists it through
:func:`catopt_discovery.evidence.store_object` as a declared-object
record (``kind="abstraction"`` by default), materializing the
replayable certificate when the composite's ``derivation`` is given a
concrete instance.  Nothing here admits: the stored record still faces
``evidence.run_gauntlet`` — construction is a claim, the gauntlet is
the referee.

Term specs are small declarative data — the same job ``Op.make``
pattern trees do by hand, but as a value the constructor can carry::

    spec := str                    # a metavariable leaf ("X")
          | int | float            # a Const leaf
          | Const | Op | Var | ... # any concrete term, passed through
          | (op, *specs)           # Op.make(op, *specs)
          | (op, *specs, dict)     # trailing dict = attrs
"""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from catopt_core.egraph import Certificate, Rewrite
from catopt_core.egraph.terms import (
    _replace_subterm,
    _term_instantiate,
    _term_match,
)
from catopt_core.ir import Const, Op
from catopt_core.laws import tags as _tags
from catopt_core.laws.cond import eval_cond
from catopt_core.meta import (
    _positions,
    _subterm,
    apply_rewrite_at,
    instantiate_pattern,
    match_pattern,
)

__all__ = [
    "AutoCond",
    "ConstructedObject",
    "auto_cond_object",
    "compose_objects",
    "fold_object",
    "lift_object",
    "object_record",
    "store_constructed",
    "term_from_spec",
]


@dataclass(frozen=True)
class ConstructedObject:
    """A machine-constructed declared object, before admission.

    ``rule`` is the synthesized ``Rewrite`` — the object body.
    ``kind`` is the declaration kind the record will carry
    (``"abstraction"`` / ``"bridge"``).  ``construction`` is the
    provenance trace — the operation and the premise names it ran
    over, e.g. ``("compose", "comm_mul", "silu_fold")``.  The record's
    honesty is the gauntlet's problem, not the constructor's: a
    ``ConstructedObject`` is a *claim* until
    ``evidence.run_gauntlet`` rules on it.
    """

    rule: Rewrite
    kind: str
    construction: tuple[str, ...]
    note: str = ""


# ---------------------------------------------------------------------------
#  Term specs — declarative pattern data
# ---------------------------------------------------------------------------


def _spec_op(spec: tuple | list) -> Op:
    """Build an :class:`Op` node from a ``(op, *specs[, attrs])`` spec.

    A trailing ``dict`` element is the attr map; every other element
    is a recursive term spec.  A non-``str`` head is a malformed spec
    — ``TypeError``, not a guess.
    """
    head, *rest = spec
    attrs: dict = {}
    if rest and isinstance(rest[-1], dict):
        attrs = dict(rest.pop())
    if not isinstance(head, str):
        raise TypeError(f"spec head must be an op name: {head!r}")
    return Op.make(head, *(term_from_spec(s) for s in rest), **attrs)


def term_from_spec(spec: Any) -> Any:
    """Build a pattern term from declarative *spec* data.

    ``str`` yields a metavariable leaf, ``int``/``float``/``bool`` a
    ``Const``, concrete terms (``Op``/``Const``/``Var``/``Param``)
    pass through, and a non-empty tuple builds an :class:`Op` node via
    :func:`_spec_op`.  An empty or non-tuple non-term is a malformed
    spec — ``TypeError``, not a guess.
    """
    if isinstance(spec, str):
        return spec
    if isinstance(spec, (int, float)):
        return Const(spec)
    if isinstance(spec, (tuple, list)):
        if not spec:
            raise TypeError(f"malformed term spec: {spec!r}")
        return _spec_op(spec)
    if isinstance(spec, (Op, Const)):
        return spec
    # Var/Param and any other concrete leaf pass through unchanged.
    return spec


# ---------------------------------------------------------------------------
#  Construction operations
# ---------------------------------------------------------------------------


def fold_object(
    name: str,
    spelled: Any,
    kernel: Any,
    *,
    arg: str = "X",
    cond: Any = None,
    dspec: Any = None,
    kind: str = "abstraction",
    tags: Any = (_tags.SIMPLIFICATION,),
) -> ConstructedObject:
    """Construct a *fold* object: the spelled composition → the kernel.

    ``fold_object("softsign_fold", ("div", "X", ("add", ("abs", "X"),
    1)), "softsign")`` declares ``div(x, |x|+1) → softsign(x)`` — the
    fused kernel *is* the abstraction; the spelled form is its
    expansion.  *arg* names the metavariable the kernel wraps.

    *kernel* also accepts a full term spec — the general fold
    ``spelled → <term spec>`` for abstractions whose dispatched form
    is not a unary kernel, e.g. the mask-free attention fold
    ``matmul(softmax(q@kᵀ,·),v) → sdpa(q,k,v,scale=1)`` spelled as
    ``("sdpa", "Q", "K", "V", {"scale": 1.0})``.  *cond* / *dspec*
    carry the object's declarative side condition and derive spec —
    pure data, the same serializable shape shipped ``cond=`` /
    ``dspec=`` laws take; the gauntlet's guarded-region sweep reads
    the cond to decide where the declared equality must hold.
    """
    lhs = term_from_spec(spelled)
    rhs = (
        Op.make(kernel, arg)
        if isinstance(kernel, str)
        else term_from_spec(kernel)
    )
    rule = Rewrite(
        name=name,
        lhs=lhs,
        rhs=rhs,
        law="the spelled composition folds to its dispatched form",
        cond=cond,
        dspec=dspec,
        tags=frozenset(tags),
    )
    return ConstructedObject(
        rule=rule,
        kind=kind,
        construction=(
            "fold",
            kernel if isinstance(kernel, str) else repr(kernel),
        ),
    )


def lift_object(
    name: str,
    step: Any,
    carrier: Any,
    apply_op: str,
    *,
    state: Any = None,
    cond: Any = None,
    dspec: Any = None,
    kind: str = "abstraction",
    tags: Any = (_tags.CARRIER,),
) -> ConstructedObject:
    """Construct a *lift* object: a program step → its carrier form.

    Declares ``step → apply_op(carrier[, state])`` as data — the
    carrier-style introduction the ADR names (the affine scan, the
    online-softmax monoid) as a record rather than a hardcoded
    ``R(...)``.  ``lift_object("aff_step_lift",
    ("add", ("matmul", "A", "h"), "x"), ("aff", "A", "x"), "apply",
    state="h")`` declares the affine-step lift;
    ``state=None`` covers the unary applies (``om_apply``).
    *cond* / *dspec* carry the object's declarative guard and derive
    spec — e.g. the state-shape condition the diagonal-scan lifts
    ship with (an economy guard, serializable as data).
    """
    lhs = term_from_spec(step)
    args = [term_from_spec(carrier)]
    if state is not None:
        args.append(term_from_spec(state))
    rhs = Op.make(apply_op, *args)
    rule = Rewrite(
        name=name,
        lhs=lhs,
        rhs=rhs,
        law=f"{apply_op} lift: the step is one carrier application",
        cond=cond,
        dspec=dspec,
        tags=frozenset(tags),
    )
    return ConstructedObject(
        rule=rule, kind=kind, construction=("lift", apply_op)
    )


def _as_rule(obj: Any) -> Rewrite:
    """Accept a ``Rewrite`` or a ``ConstructedObject`` as a premise."""
    return obj.rule if isinstance(obj, ConstructedObject) else obj


# ---------------------------------------------------------------------------
#  Guard transport — composing premises whose guard needs real shapes
# ---------------------------------------------------------------------------
#
#  A premise fires decidable (``apply_rewrite_at``) only when its guard
#  evaluates on the symbolic binding.  A guard that reads real extents —
#  ``om_split``'s ``_check_om_concat_dims`` asks the concat dims to name
#  the right axes and the chunk shapes to align — declines there: the
#  metavariables are leaves, not shaped tensors, so the predicate is
#  undecidable, not false.  The composition used to abort (``None``).
#
#  The honest fix is *guard transport*: fire the premise structurally
#  (its LHS alone) and carry its guard into the composite, so the
#  composite is admissible exactly where every premise it rewrote would
#  have fired.  Two flavours, chosen by what the premise's guard is:
#
#  * **declarative** — a purely declarative ``cond`` is re-expressed
#    (metavariable references renamed) and folded into the composite's
#    own ``cond``.  The composite stays pure data (serializable), so the
#    gauntlet's guarded-region sweep can rule on it.
#  * **procedural** — a ``check``/``derive`` with a code remainder is
#    re-run at fire time on the premise's own binding (the intermediate
#    re-instantiated from the composite's binding).  The composite is
#    sound but non-serializable: the store flags ``missing_hooks``
#    honestly and the gauntlet's full-data gate refuses it — construction
#    is a claim, and a claim data cannot carry is refused as such.


def _structural_fire(rule: Rewrite, term: Any) -> tuple | None:
    """First position where *rule*'s LHS matches *term*, guard ignored.

    The decidable path (:func:`catopt_core.meta.apply_rewrite_at`)
    evaluates the premise's guard on the symbolic binding; a guard that
    needs real shapes declines there.  This retries the *structural*
    match — the LHS alone — and returns ``(path, rewritten, match)`` so
    the caller can transport the premise's guard into the composite.
    The premise's ``derive`` still runs (its RHS attributes must
    instantiate); a ``derive`` that cannot evaluate on the symbolic
    binding declines — a composite whose RHS needs instance-computed
    attributes is not constructible at pattern level, honestly.
    """
    for path, sub in _positions(term):
        subst = match_pattern(rule.lhs, sub, {})
        if subst is None:
            continue
        inst = dict(subst)
        if rule.derive is not None:
            try:
                extra = rule.derive(subst)
            except Exception:
                extra = None
            if extra is None:
                continue
            inst = {**subst, **extra}
        try:
            rhs = instantiate_pattern(rule.rhs, inst)
        except KeyError:
            continue
        nxt = _replace_subterm(term, path, rhs)
        if nxt != term:
            return path, nxt, subst
    return None


def _rename_cond(node: Any, rename: dict[str, str]) -> Any:
    """Rename metavariable references inside a declarative cond tree."""
    if isinstance(node, str):
        return rename.get(node, node)
    if isinstance(node, (tuple, list)):
        return tuple(_rename_cond(x, rename) for x in node)
    return node


def _declarative_clause(rule: Rewrite, match: dict) -> Any:
    """Re-express *rule*'s declarative guard over the composite's metavars.

    Returns the premise's ``cond`` with its metavariable references
    renamed to the composite's, or ``None`` when the guard is not purely
    declarative (a procedural ``check`` remainder) or binds a
    metavariable to a compound subterm — a cond atom addresses
    metavariables by name, so a compound binding is un-expressible as a
    clause and must ride the procedural path.
    """
    from catopt_core.laws.serialize import _proc_check

    if rule.cond is None or _proc_check(rule):
        return None
    rename: dict[str, str] = {}
    for k, v in match.items():
        if not isinstance(v, str):
            return None
        rename[k[len("$attr:") :] if k.startswith("$attr:") else k] = v
    return _rename_cond(rule.cond, rename)


def _guard_transport(steps: list) -> Any:
    """Build the composite's fire-time check from transported premises.

    *steps* is ``(rule, mid_pattern, path)`` per premise fired
    structurally — *mid_pattern* the intermediate term (over the
    composite's metavariables) the premise fired into, *path* the
    position inside it.  On a composite firing the intermediate is
    re-instantiated from the binding and the premise's own ``check``
    re-run on ITS binding (the check sees only the LHS binding, never
    the premise's ``derive`` output — the ``apply_rewrite_at`` order),
    so the composite is admissible exactly where every transported
    premise is.  Total — an un-evaluable re-check declines, never
    raises (the ``apply_rewrite_at`` convention).
    """

    def check(bound: dict) -> bool:
        for rule, mid, path in steps:
            try:
                term = instantiate_pattern(mid, bound)
                m = match_pattern(rule.lhs, _subterm(term, path), {})
                if rule.check is not None and not rule.check(m):
                    return False
            except Exception:
                return False
        return True

    return check


def _fold_and(clauses: list) -> Any:
    """Fold guard clauses into one declarative cond datum (or ``None``)."""
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return ("and", *clauses)


def _fire_premise(rule: Rewrite, cur: Any) -> tuple | None:
    """Fire *rule* on *cur*, decidable first then structurally.

    Returns ``(rewritten, clause, step)`` — the rewritten term, the
    declarative guard clause to fold into the composite's ``cond`` (or
    ``None``), and the procedural ``(rule, mid, path)`` re-check step (or
    ``None``) — or ``None`` when the premise's LHS matches nowhere.  A
    decidable firing transports nothing (its guard held on the symbolic
    binding); a structural firing transports the guard.
    """
    for path, _sub in _positions(cur):
        nxt = apply_rewrite_at(rule, cur, path)
        if nxt is not None and nxt != cur:
            return nxt, None, None
    fired = _structural_fire(rule, cur)
    if fired is None:
        return None
    path, nxt, match = fired
    clause = _declarative_clause(rule, match)
    if clause is not None:
        return nxt, clause, None
    return nxt, None, (rule, cur, path)


def _spec_map(specialize: dict | None) -> dict:
    """Build the ``{metavar: term}`` substitution from a spec map."""
    return {k: term_from_spec(v) for k, v in (specialize or {}).items()}


def _premise_names(rules: list) -> tuple[str, ...]:
    """Return the premises' names, deduplicated, in firing order."""
    return tuple(dict.fromkeys(r.name for r in rules))


def _head_transport(
    head: Rewrite, lhs: Any, has_caller_cond: bool
) -> tuple:
    """Transport the first premise's guard (it guards the composite LHS).

    Returns ``(clause, step)``.  A declaratively re-expressible guard
    becomes a cond clause; otherwise the caller's ``cond`` is taken as
    the declared composite guard (the pre-existing contract) unless none
    was given, in which case the guard rides a procedural fire-time step.
    """
    if head.cond is None and head.check is None:
        return None, None
    m0 = match_pattern(head.lhs, lhs, {})
    clause = _declarative_clause(head, m0) if m0 is not None else None
    if clause is not None:
        return clause, None
    return None, (None if has_caller_cond else (head, lhs, ()))


def _specialize(pat: Any, subst: dict) -> Any:
    """Instantiate *pat* under *subst*, leaving unbound metavars.

    Metavar leaves not in *subst* and unbound ``$attr:`` metavars stay
    metavariables — the composite's pattern keeps them open.
    """
    if isinstance(pat, str):
        return subst.get(pat, pat)
    if isinstance(pat, Op):
        args = tuple(_specialize(a, subst) for a in pat.args)
        attrs = {
            k: (subst.get(v, v) if isinstance(v, str) else v)
            for k, v in pat.attrs.items()
        }
        return Op.make(pat.op, *args, **attrs)
    return pat


def compose_objects(
    name: str,
    first: Any,
    *rest: Any,
    specialize: dict | None = None,
    cond: Any = None,
    dspec: Any = None,
    kind: str = "abstraction",
    tags: Any = (),
) -> ConstructedObject | None:
    """Construct a *composite* object by rewriting premises' RHSs.

    The composite ``first.lhs(specialized) -> rhs`` where ``rhs`` is
    ``first.rhs`` rewritten by each of *rest* in turn, applied at the
    first position where the premise fires.

    A premise fires **decidable** when its guard evaluates on the
    symbolic binding (:func:`catopt_core.meta.apply_rewrite_at` — the
    cond DSL and its shape specs decide leaf/rank/spec clauses without
    concrete values).  A premise whose guard *needs real shapes* —
    ``om_split``'s concat-dim/chunk-alignment check, which reads tensor
    extents a metavariable does not carry — is instead fired
    **structurally** (its LHS alone) and its guard is *transported* into
    the composite (see the module note on guard transport): a purely
    declarative premise ``cond`` becomes an extra clause of the
    composite's own ``cond`` (still pure data), a procedural
    ``check``/``derive`` becomes the composite's fire-time ``check``.
    A premise whose LHS does not match anywhere still fails the
    composition as ``None``, honestly.

    *specialize* instantiates ``first``'s pattern metavariables
    (``{name: spec}``) before composition — how
    ``compose(comm_mul{x ↦ sigmoid(X)}, silu_fold)`` yields
    ``mul(sigmoid(X),X) → silu(X)`` and ``compose(aff_lift, aff_lift,
    specialize={"h": <the step spec>})`` yields the two-step scan
    lift.  The composite's ``derivation`` names the premises —
    deduplicated, in firing order — so a composite over shipped rules
    can carry a replayable certificate.

    *cond* / *dspec* declare the composite's own guard and derive spec
    — the constructor's claim about where the composite is legal.  A
    declaratively transported premise guard is ANDed onto *cond*; a
    procedural one rides ``check`` (the composite then reports
    ``missing_hooks == ["check"]`` at the store, honestly).
    """
    rules = [_as_rule(first), *(_as_rule(r) for r in rest)]
    subst = _spec_map(specialize)
    lhs = _specialize(rules[0].lhs, subst)
    cur = _specialize(rules[0].rhs, subst)
    clauses: list = []
    steps: list = []
    # The first premise's own guard applies to the composite's LHS, so it
    # is transported too (see :func:`_head_transport`).
    head_clause, head_step = _head_transport(
        rules[0], lhs, cond is not None
    )
    if head_clause is not None:
        clauses.append(head_clause)
    if head_step is not None:
        steps.append(head_step)
    for r in rules[1:]:
        fired = _fire_premise(r, cur)
        if fired is None:
            return None
        cur, clause, step = fired
        if clause is not None:
            clauses.append(clause)
        elif step is not None:
            steps.append(step)
    premises = _premise_names(rules)
    rule = Rewrite(
        name=name,
        lhs=lhs,
        rhs=cur,
        law="composite of " + " ∘ ".join(premises),
        check=_guard_transport(steps) if steps else None,
        cond=_fold_and(([cond] if cond is not None else []) + clauses),
        dspec=dspec,
        tags=frozenset(tags),
        derivation=premises,
    )
    return ConstructedObject(
        rule=rule,
        kind=kind,
        construction=("compose", *premises),
    )


# ---------------------------------------------------------------------------
#  Storage — the constructed object becomes a declared record
# ---------------------------------------------------------------------------


def object_record(obj: ConstructedObject, *, cert: Any = None) -> dict:
    """Return the declared-object record for *obj* (unstored).

    The ``laws.serialize.object_to_data`` record with the object's
    kind stamped — the same shape ``evidence.store_object`` persists.
    """
    from catopt_core.laws.serialize import object_to_data

    return object_to_data(obj.rule, cert=cert, kind=obj.kind)


def _certify(
    rule: Rewrite, universe: Any, env: dict
) -> Certificate | None:
    """Materialize *rule*'s derivation cert on the *env* instance.

    *env* maps the pattern's metavariables to concrete leaves; the
    instance ``(lhs(env), rhs(env))`` is saturated under the named
    premises via ``lemma_cert.materialize``.  Returns the replayable
    certificate, or ``None`` for a non-linear verdict — the honest
    absence, never a stub.
    """
    from catopt_core.laws import ALL_RULES

    from catopt_discovery import lemma_cert

    uni = list(ALL_RULES) if universe is None else list(universe)
    inst = (
        _term_instantiate(rule.lhs, env),
        _term_instantiate(rule.rhs, env),
    )
    _row, cert = lemma_cert.materialize(rule, uni, inst=inst)
    return cert


def store_constructed(
    conn: sqlite3.Connection,
    obj: ConstructedObject,
    corpus_hash: str = "",
    *,
    universe: Any = None,
    env: dict | None = None,
) -> str:
    """Store *obj* as a declared-object row; return its alpha key.

    Delegates to :func:`catopt_discovery.evidence.store_object` with
    the construction's declared ``kind``.  When *env* is given and the
    composite carries a ``derivation``, the certificate is
    materialized on that instance first — a constructed object can
    carry its proof, replayed strict at the gauntlet's cert gate.
    """
    from catopt_discovery import evidence as ev

    kwargs: dict[str, Any] = {"kind": obj.kind, "universe": universe}
    if env is not None and obj.rule.derivation:
        kwargs["cert"] = _certify(obj.rule, universe, env)
    return ev.store_object(conn, obj.rule, corpus_hash, **kwargs)


# ---------------------------------------------------------------------------
#  Auto-cond — the conditional → guarded constructor
# ---------------------------------------------------------------------------
#
#  The real-corpus run's honest finding (project/retros/guide-real-run.md):
#  candidates reach ``SHIP`` in the arena but ``usable: yes`` stays 0 —
#  they are *conditional* equalities (``gap_gen`` witnesses mint the
#  instances where they happen to hold; on the real corpus they are
#  conditional).  Admission requires the guard written as declarative
#  ``cond`` data.  :func:`auto_cond_object` is that step:
#
#  1. **Measure the bare domain** — the oracle's synthesized binding
#     envs (:func:`evidence._synth_sites`, the attr-sweep machinery)
#     plus every real corpus match — each site evaluated tri-state
#     exactly as a firing would: the candidate's own ``derive`` rides
#     along (a veto is a firing abort — the site never enters the
#     guard's universe), then both sides evaluate fp64.
#  2. **Enumerate guard predicates** — the declarative vocabulary
#     ``catopt_core.laws.cond`` already interprets, instantiated over
#     the pattern's metavariables and attr metavariables (leaf/shape/
#     rank/broadcast predicates, attr comparisons, the view-op
#     predicates — ``ones-before``/``axes-*``/``bcast-eq`` over the
#     computed shape specs, the attr values *observed* in the measured
#     domain for ``attr-eq``).
#  3. **Minimal conjunction** — a covering set: the smallest set of
#     predicates that accepts every measured ``equal`` site and
#     declines every measured ``unequal`` / ``rhs-err`` site — the
#     smallest guard that keeps all measured equal sites, not a guard
#     tuned to exclude one known false site.  ``other``-bucket sites
#     (lhs-err &c.) are don't-care, with the tie-break preferring the
#     combination that admits fewest of them.
#
#  A refusal is the honest answer when nothing declarable separates:
#  no measured equal site, or no conjunction of ``max_clauses`` covers.
#  The minted object is *pure data* — it carries the found ``cond``
#  (plus the candidate's ``dspec``/procedural remainders, which the
#  store flags honestly) and still faces ``evidence.run_gauntlet`` —
#  construction is a claim, the gauntlet is the referee.


@dataclass(frozen=True)
class AutoCond:
    """The outcome of an auto-cond search over a measured domain.

    ``object`` is the minted guarded :class:`ConstructedObject`, or
    ``None`` on refusal — the refusal is the honest answer, recorded
    in ``detail``.  ``cond`` is the minted declarative guard (``True``
    when the domain measured no bad site — a vacuous cover, flagged
    in ``detail``); ``clauses`` the separating predicates the search
    chose, in bank order.  The site counts partition the measured
    domain: ``equal`` sites where the declared equality held, ``bad``
    the ``unequal``/``rhs-err`` sites the guard must decline,
    ``other`` non-evaluable sites, ``declined`` firing aborts
    (``derive`` vetoes) outside the universe; ``accepted`` /
    ``accepted_other`` count what the minted guard admits.
    """

    object: ConstructedObject | None
    cond: Any = None
    clauses: tuple = ()
    measured: int = 0
    equal: int = 0
    bad: int = 0
    unstable: int = 0
    other: int = 0
    declined: int = 0
    accepted: int = 0
    accepted_other: int = 0
    detail: str = ""


#: Default cap on conjunction size — the minimal cover search is
#: bounded by this, so "no cover found" means "no cover of ≤ k".
_AUTO_MAX_CLAUSES = 3

#: Visit cap for the covering DFS — a guard needing a wider search is
#: refused, not chased.
_AUTO_VISIT_CAP = 200_000


def _mv_names(*pats: Any) -> list[str]:
    """Sorted leaf-metavariable names over *pats*."""
    from catopt_discovery import oracle as lvo

    out: set[str] = set()
    for p in pats:
        out.update(lvo._leaf_metavars(p))
    return sorted(out)


def _attr_mvs(*pats: Any) -> list[str]:
    """Sorted attr-metavariable names over *pats* (``$attr:`` bodies)."""
    from catopt_discovery import oracle as lvo

    out: set[str] = set()
    for pat in pats:
        for node in lvo._view_nodes([pat]):
            out.update(
                v for v in node.attrs.values() if isinstance(v, str)
            )
    return sorted(out)


#: Index views whose commutation guard needs the axis-alignment atoms.
_IDX_ATTR = {"select": "dim", "slice": "dim", "chunk": "dim"}


def _str_attrs(node: Op) -> dict:
    """Return the node's str-valued attrs (attr-metavariable positions)."""
    return {k: v for k, v in node.attrs.items() if isinstance(v, str)}


def _spec_unsq(operand: Any, attrs: dict) -> tuple | None:
    """``unsqueeze`` output spec (the inserted axis is an attr metavar)."""
    d = attrs.get("dim")
    return ("unsq-out", operand, d) if isinstance(d, str) else None


def _spec_reshape(operand: Any, attrs: dict) -> tuple | None:
    """``reshape``/``view`` output spec (the target shape is an attr)."""
    s = attrs.get("shape")
    return ("reshape-out", operand, s) if isinstance(s, str) else None


def _spec_getitem(operand: Any, _attrs: dict) -> tuple | None:
    """``getitem`` output spec (no attrs to carry)."""
    return ("getitem-out", operand)


def _spec_select(operand: Any, attrs: dict) -> tuple | None:
    """``select`` output spec (the picked axis is an attr metavar)."""
    d = attrs.get("dim")
    return ("select-out", operand, d) if isinstance(d, str) else None


def _spec_slice(operand: Any, attrs: dict) -> tuple | None:
    """``slice`` output spec (axis + bounds/step metavariables)."""
    d = attrs.get("dim")
    if not isinstance(d, str):
        return None
    return (
        "slice-out",
        operand,
        d,
        attrs.get("start"),
        attrs.get("end"),
        attrs.get("step"),
    )


def _spec_chunk(operand: Any, attrs: dict) -> tuple | None:
    """``chunk`` output spec (chunk count is an attr metavar)."""
    c = attrs.get("chunks")
    if not isinstance(c, str):
        return None
    return ("chunk-out", operand, c, attrs.get("dim"))


def _spec_transpose(operand: Any, attrs: dict) -> tuple | None:
    """``transpose``/``t`` output spec (the swapped axes are attrs)."""
    d0 = attrs.get("dim0")
    if not isinstance(d0, str):
        return None
    return ("transpose-out", operand, d0, attrs.get("dim1"))


#: The view-output spec table — one entry per specifiable view op.
#: Mirrors the ``_infer_op_shape`` branches the ``laws.cond`` shape
#: specs implement; the bank's specs and the DSL's must agree.
_VIEW_SPEC_FNS = {
    "unsqueeze": _spec_unsq,
    "reshape": _spec_reshape,
    "view": _spec_reshape,
    "getitem": _spec_getitem,
    "select": _spec_select,
    "slice": _spec_slice,
    "chunk": _spec_chunk,
    "transpose": _spec_transpose,
    "t": _spec_transpose,
}


def _view_out_spec(op: str, operand: Any, attrs: dict) -> tuple | None:
    """Return the view-output shape spec for *op*, or ``None``."""
    fn = _VIEW_SPEC_FNS.get(op)
    return fn(operand, attrs) if fn is not None else None


def _view_specs(nodes: Iterable[Op]) -> list[tuple[str, tuple]]:
    """``(operand-mv, output-shape spec)`` per enumerable view node."""
    out: list[tuple[str, tuple]] = []
    for n in nodes:
        if not n.args or not isinstance(n.args[0], str):
            continue
        spec = _view_out_spec(n.op, n.args[0], _str_attrs(n))
        if spec is not None:
            out.append((n.args[0], spec))
    return out


def _view_commute_views(nodes: Iterable[Op]) -> list[tuple]:
    """Return ``(operand-mv, spec, op, attrs)`` per view node."""
    views: list[tuple[str, tuple, str, dict]] = []
    for n in nodes:
        if not n.args or not isinstance(n.args[0], str):
            continue
        u = n.args[0]
        attrs = _str_attrs(n)
        spec = _view_out_spec(n.op, u, attrs)
        if spec is not None:
            views.append((u, spec, n.op, attrs))
    return views


def _wrap_preds(views: list[tuple], mvs: list[str]) -> list[tuple]:
    """Emit the wrap family: ``bcast(g(u), v) == g(bcast(u, v))``."""
    out: list[tuple] = []
    for u, su, op, attrs in views:
        for v in mvs:
            if v == u:
                continue
            suv = _view_out_spec(op, ("bcast", u, v), attrs)
            if suv is None:
                continue
            out.append(("bcast-eq", su, v, suv, suv))
            out.append(("shaped", ("bcast", u, v)))
            k = _IDX_ATTR.get(op)
            if k is not None and k in attrs:
                out.append(("bcast-dim-inv", v, u, attrs[k]))
            if op in ("transpose", "t") and "dim0" in attrs:
                out.append(
                    (
                        "or",
                        ("axes-noop", v, attrs["dim0"], attrs["dim1"]),
                        ("rank", v, "<=", 1),
                    )
                )
    return out


def _pair_preds(views: list[tuple]) -> list[tuple]:
    """Emit the pair family: ``bcast(g(a), g(b)) == g(bcast(a, b))``."""
    out: list[tuple] = []
    for u, su, op, attrs in views:
        for w, sw, op2, attrs2 in views:
            if w == u or op2 != op or attrs2 != attrs:
                continue
            suw = _view_out_spec(op, ("bcast", u, w), attrs)
            if suw is None:
                continue
            out.append(("bcast-eq", su, sw, suw, suw))
            out.append(("shaped", ("bcast", u, w)))
            k = _IDX_ATTR.get(op)
            if k is not None and k in attrs:
                out.append(("axis-align-eq", u, w, attrs[k]))
    return out


def _viewview_preds(nodes: Iterable[Op]) -> list[tuple]:
    """Emit the view-view family: ``g1(g2(a)) == g2(g1(a))``."""
    out: list[tuple] = []
    for n in nodes:
        inner = n.args[0] if n.args else None
        if not isinstance(inner, Op) or not inner.args:
            continue
        if not isinstance(inner.args[0], str):
            continue
        u = inner.args[0]
        inner_attrs = _str_attrs(inner)
        attrs = _str_attrs(n)
        inner_spec = _view_out_spec(inner.op, u, inner_attrs)
        g1 = _view_out_spec(n.op, u, attrs)
        so = (
            _view_out_spec(n.op, inner_spec, attrs)
            if inner_spec is not None
            else None
        )
        g2g1 = (
            _view_out_spec(inner.op, g1, inner_attrs)
            if g1 is not None
            else None
        )
        if so is not None and g2g1 is not None:
            out.append(("shape-eq", so, g2g1))
    return out


def _view_commute_preds(
    nodes: Iterable[Op], mvs: list[str]
) -> list[tuple]:
    """Enumerate the view-commute guard vocabulary over view nodes.

    Three families, all "a view must commute with a broadcast": the
    *wrap* form, the *pair* form with axis alignment, and the
    *view-view* form.
    """
    views = _view_commute_views(nodes)
    return (
        _wrap_preds(views, mvs)
        + _pair_preds(views)
        + _viewview_preds(nodes)
    )


def _mv_preds(mvs: list[str]) -> list[tuple]:
    """Single-metavar predicates — leaf kind, shape, rank, const."""
    bank: list[tuple] = []
    for t in mvs:
        bank += [
            ("concrete", t),
            ("shaped", t),
            ("scalar", t),
            ("uniform", t),
            ("ones-but-last", t),
            ("leaf", t),
            ("const", t),
            ("const-num", t),
            ("not", ("leaf", t)),
            ("not", ("const", t)),
        ]
        for k in (0, 1, 2, 3):
            bank.append(("rank", t, "==", k))
            if k:
                bank.append(("rank", t, ">=", k))
                bank.append(("rank", t, "<=", k))
        bank += [("rank", t, "!=", 0), ("rank", t, ">=", 4)]
        for cmp_ in ("==", "!=", ">", "<="):
            for v in (0, 1):
                bank.append(("const-cmp", t, cmp_, v))
    return bank


def _mv_pair_preds(mvs: list[str]) -> list[tuple]:
    """Two-metavar predicates — shape relations and broadcasts."""
    bank: list[tuple] = []
    for a, b in itertools.combinations(mvs, 2):
        bank += [
            ("rank-eq", a, b),
            ("shape-eq", a, b),
            ("shape-compat", a, b),
            ("term-eq", a, b),
            ("bcast-into", a, b),
            ("bcast-into", b, a),
            ("mm-shape-ok", a, b),
            ("mm-shape-ok", b, a),
        ]
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                bank.append(("dim-eq", a, i, b, j))
                bank.append(("dim-compat", a, i, b, j))
    return bank


def _spec_preds(
    mvs: list[str], specs: list[tuple[str, tuple]]
) -> list[tuple]:
    """Predicates over the computed shape specs (view outputs)."""
    bank: list[tuple] = []
    for u, s in specs:
        for m in mvs:
            bank += [
                ("shape-eq", s, m),
                ("bcast-into", m, s),
                ("bcast-into", s, m),
            ]
            if m != u:
                # the strip family: bcast(view(u), v) == bcast(u, v)
                bank.append(("bcast-eq", s, m, u, m))
                bank.append(("bcast-eq", s, u, u, m))
    return bank


def _attr_shape_preds(n: str, mvs: list[str]) -> list[tuple]:
    """Attr-vs-shape predicates for one attr metavariable."""
    bank: list[tuple] = []
    for t in mvs:
        for k in (-1, 0, 1, 2):
            bank.append(("axis", t, n, k))
        for m_ in (2, 3):
            for r in (0, 1):
                bank.append(("dim-mod", t, n, m_, r))
    return bank


def _attr_observed_preds(n: str, envs: Iterable[dict]) -> list[tuple]:
    """``attr-eq`` over the values the measurement observed for *n*.

    The domain's own vocabulary — no guessed constants beyond the
    small ``attr-is``/``attr-len`` sets.
    """
    seen: list = []
    for env in envs:
        v = env.get(f"$attr:{n}", None)
        if v is not None and not isinstance(v, bool):
            if isinstance(v, list):
                v = tuple(v)
            if v not in seen and len(seen) < 12:
                seen.append(v)
    return [("attr-eq", n, v) for v in seen]


def _attr_preds(
    names: list[str], mvs: list[str], envs: Iterable[dict]
) -> list[tuple]:
    """Attr-level predicates, including observed ``attr-eq`` values."""
    bank: list[tuple] = []
    for n in names:
        for kind in ("int", "float", "number", "bool", "str", "tuple"):
            bank.append(("attr-type", n, kind))
        bank += [
            ("attr-is", n, None),
            ("attr-is", n, True),
            ("attr-is", n, False),
        ]
        for cmp_ in ("==", ">=", "<="):
            for k in (0, 1, 2, 3):
                bank.append(("attr-len", n, cmp_, k))
        bank += _attr_shape_preds(n, mvs)
        bank += _attr_observed_preds(n, envs)
    for n1, n2 in itertools.combinations(names, 2):
        bank.append(("attr-eq-attr", n1, n2))
    for n1 in names:
        for n2 in names:
            for a, b in itertools.combinations(mvs, 2):
                bank.append(("dim-eq-attr", a, n1, b, n2))
            for t in mvs:
                for cmp_ in (">=", "<=", "==", "!="):
                    bank.append(("attr-cmp-dim", n1, cmp_, t, n2))
    return bank


def _op_in_preds(mvs: list[str], envs: Iterable[dict]) -> list[tuple]:
    """``op-in``/``not op-in`` over ops actually observed bound."""
    bank: list[tuple] = []
    for t in mvs:
        ops = tuple(
            sorted({e[t].op for e in envs if isinstance(e.get(t), Op)})
        )
        if ops:
            bank.append(("op-in", t, ops))
            bank.append(("not", ("op-in", t, ops)))
    return bank


def _view_pred_unsqueeze(
    u: str, amv: dict, mvs: list[str], reshape_specs: list
) -> list[tuple]:
    """Emit the unsqueeze strip/naturality guard vocabulary."""
    bank = [("ones-before", u, amv["dim"])]
    for m in mvs:
        if m == u:
            continue
        for s in reshape_specs:
            bank.append(("flat-pair-unsq", u, amv["dim"], m, s))
            bank.append(("flat-map-unsq", u, amv["dim"], m, s))
    return bank


def _view_pred_axes(
    u: str, amv: dict, _mvs: list, _rs: list
) -> list[tuple]:
    """Transpose-pair guards — last-two, distinct, no-op axes."""
    d0, d1 = amv["dim0"], amv["dim1"]
    return [
        ("axes-last2", u, d0, d1),
        ("axes-distinct", u, d0, d1),
        ("axes-noop", u, d0, d1),
    ]


def _view_pred_getitem(
    u: str, amv: dict, _mvs: list, _rs: list
) -> list[tuple]:
    """Getitem strip guards — the scalar-index corner."""
    return [
        ("attr-in", amv["index"], (0, -1)),
        ("dim-eq-const", u, 0, 1),
    ]


def _view_pred_slice(
    u: str, amv: dict, _mvs: list, _rs: list
) -> list[tuple]:
    """Slice strip guards — the no-op/extent corner."""
    bank: list[tuple] = []
    if "start" in amv:
        bank += [
            ("attr-is", amv["start"], None),
            ("attr-eq", amv["start"], 0),
        ]
    if "end" in amv:
        bank += [
            ("attr-is", amv["end"], None),
            ("attr-cmp-dim", amv["end"], ">=", u, amv["dim"]),
        ]
    return bank


def _view_pred_select(
    u: str, amv: dict, mvs: list[str], _rs: list
) -> list[tuple]:
    """Select naturality guards — aligned pick on the other operand."""
    return [
        ("dim-eq-attr", u, amv["dim"], b, amv["dim"])
        for b in mvs
        if b != u
    ]


#: Per-view-op guard vocabulary — ``(required str-attr keys, fn)``.
_VIEW_NODE_PREDS = {
    "unsqueeze": (("dim",), _view_pred_unsqueeze),
    "transpose": (("dim0", "dim1"), _view_pred_axes),
    "t": (("dim0", "dim1"), _view_pred_axes),
    "getitem": (("index",), _view_pred_getitem),
    "slice": (("dim",), _view_pred_slice),
    "select": (("dim",), _view_pred_select),
}


def _view_preds(
    nodes: Iterable[Op], mvs: list[str], specs: list[tuple[str, tuple]]
) -> list[tuple]:
    """View-node predicates — the strip/commute guard vocabulary."""
    bank: list[tuple] = []
    reshape_specs = [s for _u, s in specs if s[0] == "reshape-out"]
    for n in nodes:
        u = n.args[0] if n.args and isinstance(n.args[0], str) else None
        entry = _VIEW_NODE_PREDS.get(n.op)
        if u is None or entry is None:
            continue
        req, fn = entry
        amv = {k: v for k, v in n.attrs.items() if isinstance(v, str)}
        if not set(req) <= set(amv):
            continue
        bank += fn(u, amv, mvs, reshape_specs)
    return bank


def _pred_bank(
    lhs: Any, rhs: Any, envs: list[tuple[dict, str]]
) -> list[tuple]:
    """Enumerate the declarative guard vocabulary over the patterns.

    The bank is the cond DSL's own predicates instantiated over the
    pattern's metavariables and attr metavariables — the same atoms
    shipped ``cond=`` laws carry — plus the observed attr values and
    bound-op names the measured domain supplies.  Deduped by spelling;
    the search dedupes by verdict vector afterwards.
    """
    from catopt_discovery import oracle as lvo

    mvs = _mv_names(lhs, rhs)
    names = _attr_mvs(lhs, rhs)
    nodes = [n for n in lvo._view_nodes([lhs, rhs]) if n.args]
    specs = _view_specs(nodes)
    env_maps = [e for e, _o in envs]
    bank = (
        _mv_preds(mvs)
        + _mv_pair_preds(mvs)
        + _spec_preds(mvs, specs)
        + _attr_preds(names, mvs, env_maps)
        + _op_in_preds(mvs, env_maps)
        + _view_preds(nodes, mvs, specs)
        + _view_commute_preds(nodes, mvs)
    )
    seen: set = set()
    out: list[tuple] = []
    for p in bank:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ---------------------------------------------------------------------------
#  Auto-cond — measurement and the minimal cover
# ---------------------------------------------------------------------------


def _env_key(env: dict) -> str:
    """Content key for a binding env — dedups identical measurements.

    ``Var`` reprs are name-only, so the shape rides the key through
    ``oracle._bind_desc`` — the same signature the oracle's instance
    dedup uses.
    """
    from catopt_discovery import oracle as lvo

    return repr(
        sorted(
            (k, lvo._bind_desc(v))
            for k, v in env.items()
            if not k.startswith("$attr:")
        )
        + sorted(
            (k, repr(v))
            for k, v in env.items()
            if k.startswith("$attr:")
        )
    )


def _oracle_sites(
    lhs_pat: Any, rhs_pat: Any, derive: Any, limit: int
) -> Iterable:
    """Yield ``(env, lhs_i)`` deduped on the oracle's instance signature.

    The second enumeration window: ``_synth_sites`` dedups the
    instantiated pair *before* ``derive`` runs — a derive-vetoed or
    attr-collapsed env still burns a slot, so its 360-window ends
    shallower.  ``oracle.synthesize`` dedups on the post-derive
    instance sig (binds included), reaching different corners of the
    same domain — measured: the no-op-transpose counterexample corner
    of the ``_g`` twin lies inside this window, outside the other.
    The union of the two windows is the measured domain.
    """
    from catopt_core.ir import op_repr

    from catopt_discovery import oracle as lvo

    def sig(inst: dict, lhs_i: Any, rhs_i: Any) -> tuple:
        """Return the ``synthesize`` instance signature — the dedup key."""
        return (
            op_repr(lhs_i),
            op_repr(rhs_i),
            tuple(
                sorted(
                    (k, lvo._bind_desc(v))
                    for k, v in inst.items()
                    if not k.startswith("$attr:")
                )
                + sorted(
                    (k, repr(v))
                    for k, v in inst.items()
                    if k.startswith("$attr:")
                )
            ),
        )

    seen: set[tuple] = set()
    for full in lvo._binding_envs(lhs_pat, rhs_pat):
        inst = full
        if derive is not None:
            try:
                extra = derive(full)
            except Exception:
                continue
            if extra is None:
                continue
            inst = {**full, **extra}
        try:
            lhs_i = _term_instantiate(lhs_pat, inst)
            rhs_i = _term_instantiate(rhs_pat, inst)
        except Exception:
            continue
        s = sig(inst, lhs_i, rhs_i)
        if s in seen:
            continue
        seen.add(s)
        yield full, lhs_i
        if len(seen) >= limit:
            return


def _measure_domain(
    rule: Rewrite,
    corpus_terms: Iterable,
    synth_limit: int,
) -> list[tuple[dict, str]]:
    """Measure ``(bound-env, outcome)`` over the synthesized+real domain.

    The probe is the candidate's *bare* pattern — the existing
    ``cond``/``check`` is dropped so the found guard is measured as a
    complete claim, not a strengthening of a partially-written one
    (``derive`` stays: it is part of how the rule instantiates, and a
    veto is a firing abort, not a site).  Real corpus matches are
    measured the same way — a real ``unequal`` is a site the found
    guard must decline exactly like a synthesized one.  The synth
    domain unions two windows of the same enumeration —
    :func:`evidence._synth_sites` (pair-dedup) and
    :func:`_oracle_sites` (instance-sig dedup, the ``synthesize``
    convention) — since the capped windows expose different corners.
    """
    from catopt_discovery import evidence as ev
    from catopt_discovery.shape_proposal import Schema, real_matches

    probe = Rewrite(rule.name, rule.lhs, rule.rhs, derive=rule.derive)
    out: list[tuple[dict, str]] = []
    seen: set[str] = set()

    def record(env: dict, lhs_i: Any) -> None:
        key = _env_key(env)
        if key in seen:
            return
        seen.add(key)
        out.append((env, _stable_outcome(probe, env, lhs_i)))

    try:
        for subst, lhs_i in ev._synth_sites(
            rule.lhs, rule.rhs, limit=synth_limit
        ):
            record(subst, lhs_i)
        for env, lhs_i in _oracle_sites(
            rule.lhs, rule.rhs, rule.derive, synth_limit
        ):
            record(env, lhs_i)
    except (TypeError, ValueError, KeyError):
        # The enumerator's honest boundary (nested views and friends
        # raise in ``_attr_domains`` — see oracle's pinned edge):
        # whatever sites were collected still measure, plus the real
        # matches below.
        pass
    schema = Schema(rule.name, rule.lhs, rule.rhs)
    for m in real_matches(list(corpus_terms), schema):
        subst = _term_match(rule.lhs, m)
        if subst is not None:
            record(subst, m)
    return out


def _stable_outcome(probe: Rewrite, env: dict, lhs_i: Any) -> str:
    """Score one site, twice — ``unstable`` when the verdicts differ.

    ``eval_instance`` fills leaves with ambient ``torch.randn`` draws,
    so a binding whose evaluative verdict is value-sensitive — a real
    term holding ``pow(param, -0.25)`` that NaNs on a negative draw is
    the measured case — reports a different outcome per draw.  Such a
    site is *not stably measured*: it goes into the must-decline
    class, so the minted guard never admits a firing region whose
    verdict is a coin flip.  ``declined``/``guard-err`` (no numeric
    eval) and the deterministic error outcomes take a single pass.
    """
    from catopt_discovery import evidence as ev

    o1 = ev._site_outcome(probe, env, lhs_i)
    if o1 in ("declined", "guard-err", "env-err"):
        return o1
    o2 = ev._site_outcome(probe, env, lhs_i)
    return o1 if o1 == o2 else "unstable"


def _pred_masks(
    bank: list[tuple], envs: list[dict]
) -> list[tuple[tuple, int]]:
    """Evaluate every predicate on every env; return ``(pred, mask)``.

    A predicate that raises on any env is dropped entirely — a
    non-total guard is not admissible data (it would surface as
    ``guard-err`` at sweep time).  Masks dedupe: two spellings with
    the same verdict vector keep the lexicographically first.
    """
    out: dict[int, tuple] = {}
    for pred in bank:
        mask = 0
        try:
            for i, env in enumerate(envs):
                if eval_cond(pred, env):
                    mask |= 1 << i
        except (ValueError, TypeError, KeyError):
            continue
        prev = out.get(mask)
        if prev is None or repr(prev) > repr(pred):
            out[mask] = pred
    return [
        (p, m)
        for m, p in sorted(out.items(), key=lambda kv: repr(kv[1]))
    ]


def _min_cover(
    useful: list[tuple[tuple, int, int]],
    eq_mask: int,
    bad_mask: int,
    n_sites: int,
    max_clauses: int,
) -> tuple | None:
    """Smallest conjunction covering *eq_mask* and declining *bad_mask*.

    *useful* is ``(pred, accept_mask, kill_mask)`` — every pred covers
    all equal sites and declines ≥1 bad site.  Iterative deepening
    over clause count (a smaller solution is impossible at level *k*
    iff levels below it returned nothing); at each level DFS branches
    on the uncovered bad site with the fewest killers, keeping the
    combination admitting the fewest non-equal sites, determinism via
    sorted order.  ``None`` = no declarable cover of ≤ *max_clauses*.
    """
    killers: dict[int, list[int]] = {}
    for i, (_p, _m, km) in enumerate(useful):
        for site in _bit_sites(km):
            killers.setdefault(site, []).append(i)
    if any(s not in killers for s in _bit_sites(bad_mask)):
        return None
    all_mask = (1 << n_sites) - 1
    state: dict[str, Any] = {
        "key": None,
        "combo": None,
        "cap": _AUTO_VISIT_CAP,
    }

    def visit(
        uncovered: int, acc: int, chosen: tuple, limit: int
    ) -> None:
        if state["cap"] <= 0:
            return
        state["cap"] -= 1
        if not uncovered:
            extra = (acc & ~eq_mask & all_mask).bit_count()
            key = (
                extra,
                tuple(sorted(repr(useful[i][0]) for i in chosen)),
            )
            if state["key"] is None or key < state["key"]:
                state["key"], state["combo"] = key, chosen
            return
        if len(chosen) >= limit:
            return
        site = min(
            (s for s in killers if (uncovered >> s) & 1),
            key=lambda s: len(killers[s]),
        )
        for i in killers[site]:
            if i not in chosen:
                visit(
                    uncovered & ~useful[i][2],
                    acc & useful[i][1],
                    (*chosen, i),
                    limit,
                )

    for limit in range(1, max_clauses + 1):
        state["key"], state["combo"] = None, None
        visit(bad_mask, all_mask, (), limit)
        if state["combo"] is not None:
            return tuple(useful[i][0] for i in state["combo"])
    return None


def _bit_sites(mask: int) -> Iterable[int]:
    """Yield the set-bit indices of *mask*, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _outcome_bitsets(
    envs: list[tuple[dict, str]],
) -> tuple[int, int, int, int]:
    """Partition outcomes into ``(equal, bad, other, declined)`` masks.

    ``bad`` = ``unequal`` + ``rhs-err`` + ``unstable`` — the sites a
    minted guard MUST decline (the guarded-region sweep refuses them,
    and an ``unstable`` site's verdict is a coin flip); ``other``
    (``lhs-err``/``both-err``/``env-err``) is don't-care, counted for
    the report; ``declined`` sites (``derive`` vetoes) sit outside
    the firing region and play no role in the cover.
    """
    eq = bad = other = declined = 0
    for i, (_e, outcome) in enumerate(envs):
        if outcome == "equal":
            eq |= 1 << i
        elif outcome in ("unequal", "rhs-err", "unstable"):
            bad |= 1 << i
        elif outcome in ("declined", "guard-err"):
            declined |= 1 << i
        else:
            other |= 1 << i
    return eq, bad, other, declined


def _auto_refuse(
    measured: int,
    eq: int,
    bad: int,
    other: int,
    declined: int,
    detail: str,
    unstable: int = 0,
) -> AutoCond:
    """Return the honest refusal — reason recorded, no object minted."""
    return AutoCond(
        object=None,
        measured=measured,
        equal=eq.bit_count(),
        bad=bad.bit_count(),
        unstable=unstable,
        other=other.bit_count(),
        declined=declined,
        detail=detail,
    )


def _cover_clauses(
    lhs: Any,
    rhs: Any,
    universe: list[tuple[dict, str]],
    u_eq: int,
    u_bad: int,
    max_clauses: int,
) -> tuple[tuple, dict]:
    """Search the predicate bank for the minimal cover.

    Returns ``(clauses, {pred: accept-mask})`` — ``clauses`` empty
    when no declarable conjunction of ≤ *max_clauses* separates the
    measured classes.
    """
    bank = _pred_bank(lhs, rhs, universe)
    masks = _pred_masks(bank, [e for e, _ in universe])
    useful = [
        (p, m, u_bad & ~m)
        for p, m in masks
        if (m & u_eq) == u_eq and (u_bad & ~m)
    ]
    clauses = (
        _min_cover(useful, u_eq, u_bad, len(universe), max_clauses)
        or ()
    )
    return clauses, {p: m for p, m, _k in useful}


def _mint_guarded(
    rule: Any, cond: Any, *, name: str | None, kind: str, detail: str
) -> ConstructedObject:
    """Mint the found guard as a constructed object — declarative data.

    The minted ``Rewrite`` carries the found ``cond`` — *replacing*
    any declarative guard the candidate had (the measurement ran on
    the bare domain, so the found cond is the complete claim) — and
    keeps the candidate's ``dspec`` plus its *procedural* remainders:
    a callable ``check``/``derive`` the serializer flags as missing
    rides the rebuilt rule (the store marks it honestly).
    """
    from catopt_core.laws.serialize import _proc_check, _proc_derive

    new_rule = Rewrite(
        name=name or rule.name,
        lhs=rule.lhs,
        rhs=rule.rhs,
        law=rule.law or "auto-cond guarded candidate",
        check=rule.check if _proc_check(rule) else None,
        derive=rule.derive if _proc_derive(rule) else None,
        tags=rule.tags,
        error_bound=rule.error_bound,
        bound_norm=rule.bound_norm,
        derivation=rule.derivation,
        cond=cond,
        dspec=rule.dspec,
    )
    return ConstructedObject(
        rule=new_rule,
        kind=kind,
        construction=("auto-cond", rule.name),
        note=detail,
    )


def _prelude_refusal(
    envs: list, universe: list, masks: tuple[int, int, int], n_uns: int
) -> AutoCond | None:
    """Return the early refusal — an empty domain or an empty equal class."""
    n_dec = len(envs) - len(universe)
    _eq, bad, other = masks
    if not universe:
        return _auto_refuse(
            len(envs),
            0,
            0,
            0,
            n_dec,
            "no evaluable site in the measured domain",
        )
    if not _eq:
        return _auto_refuse(
            len(envs),
            0,
            bad,
            other,
            n_dec,
            "no equal site in the measured domain",
            n_uns,
        )
    return None


def _fold_cond(clauses: tuple) -> Any:
    """Fold the clause tuple into a ``cond`` datum."""
    if not clauses:
        return True
    if len(clauses) == 1:
        return clauses[0]
    return ("and", *clauses)


def auto_cond_object(
    rule: Any,
    *,
    corpus_terms: Iterable = (),
    synth_limit: int = 360,
    max_clauses: int = _AUTO_MAX_CLAUSES,
    name: str | None = None,
    kind: str = "abstraction",
) -> AutoCond:
    """Mint the smallest declarative guard making *rule* measured-true.

    *rule* is the conditional candidate — a ``Rewrite`` (or any object
    with the ``Rewrite`` fields).  The search measures the bare
    pattern's domain (the oracle's synthesized envs at *synth_limit*
    plus every real match in *corpus_terms*), enumerates the cond-DSL
    predicate bank over its metavariables, and picks the smallest
    conjunction covering every equal site and declining every
    ``unequal``/``rhs-err``/``unstable`` site.  ``object=None`` inside
    the result is the refusal: no equal site measured, or no
    declarable cover of ≤ *max_clauses*.
    """
    envs = _measure_domain(rule, corpus_terms, synth_limit)
    universe = [
        (e, o) for e, o in envs if o not in ("declined", "guard-err")
    ]
    u_eq, u_bad, u_other, _unf = _outcome_bitsets(universe)
    n_dec = len(envs) - len(universe)
    n_uns = sum(1 for _e, o in universe if o == "unstable")
    refused = _prelude_refusal(
        envs, universe, (u_eq, u_bad, u_other), n_uns
    )
    if refused is not None:
        return refused
    clauses: tuple = ()
    acc = (1 << len(universe)) - 1
    detail = (
        f"vacuous cover — no bad site among "
        f"{len(universe)} measured; cond=True"
    )
    if u_bad:
        clauses, accept_masks = _cover_clauses(
            rule.lhs, rule.rhs, universe, u_eq, u_bad, max_clauses
        )
        if not clauses:
            return _auto_refuse(
                len(envs),
                u_eq,
                u_bad,
                u_other,
                n_dec,
                f"no declarable conjunction of <= {max_clauses} "
                f"covers {u_eq.bit_count()} equal / "
                f"{u_bad.bit_count()} bad sites "
                f"({len(accept_masks)} covering predicates)",
                n_uns,
            )
        for c in clauses:
            acc &= accept_masks[c]
        detail = (
            f"{len(clauses)}-clause guard: covers "
            f"{u_eq.bit_count()} equal, declines "
            f"{u_bad.bit_count()} bad"
            + (
                f", accepts {(acc & u_other).bit_count()} other-err"
                if acc & u_other
                else ""
            )
        )
    cond = _fold_cond(clauses)
    return AutoCond(
        object=_mint_guarded(
            rule, cond, name=name, kind=kind, detail=detail
        ),
        cond=cond,
        clauses=clauses,
        measured=len(envs),
        equal=u_eq.bit_count(),
        bad=u_bad.bit_count(),
        unstable=n_uns,
        other=u_other.bit_count(),
        declined=n_dec,
        accepted=acc.bit_count(),
        accepted_other=(acc & u_other).bit_count(),
        detail=detail,
    )
