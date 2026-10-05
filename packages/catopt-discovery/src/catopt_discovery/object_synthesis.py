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

import sqlite3
from dataclasses import dataclass
from typing import Any

from catopt_core.egraph import Certificate, Rewrite
from catopt_core.egraph.terms import _term_instantiate
from catopt_core.ir import Const, Op
from catopt_core.laws import tags as _tags
from catopt_core.meta import _positions, apply_rewrite_at

__all__ = [
    "ConstructedObject",
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
    first position where the premise fires
    (:func:`catopt_core.meta.apply_rewrite_at` — guards and derives
    are evaluated on the symbolic binding, so a premise whose guard
    cannot evaluate on metavar terms declines and the composition
    fails as ``None``, honestly).

    *specialize* instantiates ``first``'s pattern metavariables
    (``{name: spec}``) before composition — how
    ``compose(comm_mul{x ↦ sigmoid(X)}, silu_fold)`` yields
    ``mul(sigmoid(X),X) → silu(X)`` and ``compose(aff_lift, aff_lift,
    specialize={"h": <the step spec>})`` yields the two-step scan
    lift.  The composite's ``derivation`` names the premises —
    deduplicated, in firing order — so a composite over shipped rules
    can carry a replayable certificate.

    *cond* / *dspec* declare the composite's own guard and derive spec
    — the constructor's claim about where the composite is legal.
    Full *cond*-transport (re-expressing each premise's condition over
    the composite's metavars) is not wired in; the caller states the
    guard the composite should carry and the gauntlet's guarded-region
    sweep rules on it.
    """
    rules = [_as_rule(first), *(_as_rule(r) for r in rest)]
    subst = {
        k: term_from_spec(v) for k, v in (specialize or {}).items()
    }
    lhs = _specialize(rules[0].lhs, subst)
    cur = _specialize(rules[0].rhs, subst)
    for r in rules[1:]:
        nxt = None
        for path, _sub in _positions(cur):
            nxt = apply_rewrite_at(r, cur, path)
            if nxt is not None and nxt != cur:
                break
            nxt = None
        if nxt is None:
            return None
        cur = nxt
    premises = tuple(dict.fromkeys(r.name for r in rules))
    rule = Rewrite(
        name=name,
        lhs=lhs,
        rhs=cur,
        law="composite of " + " ∘ ".join(premises),
        cond=cond,
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
