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
* :func:`relax_guard` — **weaken a guard**: drop one conjunct of an
  object's declarative ``cond``, growing the accepted region — a
  candidate the guarded-region sweep re-measures.
* :func:`specialize` — **narrow an object**: bind a leaf metavariable
  to a concrete term, or pin an attr metavariable to a concrete value
  (``select_mul`` under ``D=0``) — the instance family shrinks.
* :func:`auto_cond_object` — **mint a guard**: the smallest
  declarative ``cond`` separating the measured domain.

:data:`CONSTRUCTORS` is the op-name registry the arena drives;
:func:`construct` is its dispatch entry point (``construct(op_name,
*args, store) -> record | None``).

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
from catopt_core.laws.cond import (
    cond_from_data,
    eval_cond,
    eval_derive,
)
from catopt_core.meta import (
    _positions,
    apply_rewrite_at,
    instantiate_pattern,
    match_pattern,
)

__all__ = [
    "CONSTRUCTORS",
    "AutoCond",
    "ConstructedObject",
    "auto_cond_object",
    "compose_objects",
    "construct",
    "fold_object",
    "lift_object",
    "object_record",
    "relax_guard",
    "specialize",
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
#    (metavariable references renamed — a reference bound to a compound
#    piece becomes its view-output spec, a reference to a *derived*
#    attr becomes the inlined derive expr) and folded into the
#    composite's own ``cond``.  The composite stays pure data
#    (serializable), so the gauntlet's guarded-region sweep can rule
#    on it.
#  * **procedural** — a ``check``/``derive`` with a code remainder is
#    re-run at fire time on the premise's own binding (the premise
#    match re-instantiated from the composite's binding plus the
#    accumulated derive outputs — an earlier premise's minted attrs
#    feed a later premise's check, the chaining the premises' own fire
#    order performs).  The composite is sound but non-serializable:
#    the store flags ``missing_hooks`` honestly and the gauntlet's
#    full-data gate refuses it — construction is a claim, and a claim
#    data cannot carry is refused as such.


def _structural_fire(rule: Rewrite, term: Any) -> tuple | None:
    """First position where *rule*'s LHS matches *term*, guard ignored.

    The decidable path (:func:`catopt_core.meta.apply_rewrite_at`)
    evaluates the premise's guard on the symbolic binding; a guard that
    needs real shapes declines there.  This retries the *structural*
    match — the LHS alone — and returns ``(path, rewritten, match,
    extra)`` so the caller can transport the premise's guard *and*
    derive into the composite (``extra`` is the derive's output on the
    symbolic binding, ``None`` when the premise has none).  The
    premise's ``derive`` still runs (its RHS attributes must
    instantiate); a ``derive`` that cannot evaluate on the symbolic
    binding declines — a composite whose RHS needs instance-computed
    attributes is not constructible at pattern level, honestly.
    """
    for path, sub in _positions(term):
        subst = match_pattern(rule.lhs, sub, {})
        if subst is None:
            continue
        inst = dict(subst)
        extra = None
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
            return path, nxt, subst, extra
    return None


# ---------------------------------------------------------------------------
#  Transport substitution — re-expressing premise data over composite names
#
#  A transported ``cond`` clause or ``dspec`` entry is the premise's own
#  datum with every reference renamed through the premise's match —
#  the map ``premise metavar -> pattern piece over the composite's
#  metavars``.  Three cases per reference:
#
#  * bound to a metavariable — a plain rename;
#  * bound to a *compound* piece — only re-expressible in a shape
#    reading (``T``) position, where the piece becomes its view-output
#    spec (``index_select(t,D,U)`` -> ``("isel-out","t","D","U")``);
#  * bound to a literal — inlines as ``("lit", v)`` in attr positions.
#
#  Attr positions additionally consult *derived*: when the renamed
#  target is an attr the composite derives (a transported ``dspec``
#  entry), the derive expr is inlined in place of the name — a
#  composite ``cond`` evaluates before ``derive`` runs, so only the
#  inline spelling can see the value.  Anything un-expressible is
#  ``None``, sending the premise down the procedural path honestly.
# ---------------------------------------------------------------------------


#: Failure sentinel for the transport substitution — distinct from
#: ``None``, which is a legitimate literal argument (``slice-out``'s
#: optional bounds, an ``attr-is`` ``None`` pin).
_UNX: Any = object()


def _spec_arg(v: Any, derived: dict) -> Any:
    """Return the spec-arg spelling of a pattern attr value.

    A metavariable name stays a name — or becomes its derive expr when
    the composite *derives* that name; a literal tuple/list must ride
    ``("lit", v)`` (a bare tuple is an expr); other literals pass.
    """
    if isinstance(v, str):
        return derived.get(v, v)
    if isinstance(v, (tuple, list)):
        return ("lit", tuple(v))
    return v


def _term_to_spec(node: Any, derived: dict) -> Any:
    """Return a shape spec denoting the compound pattern piece *node*.

    A metavar leaf stays a metavar; a view ``Op`` becomes its
    ``*_-out`` spec over the (recursively spec'd) operand, attr values
    resolved through :func:`_spec_arg`.  ``_UNX`` when the op has no
    spec form or the piece is a concrete leaf — the clause then rides
    the procedural path instead.
    """
    if isinstance(node, str):
        return node
    if not isinstance(node, Op) or not node.args:
        return _UNX
    operand = _term_to_spec(node.args[0], derived)
    if operand is _UNX:
        return _UNX
    attrs = {k: _spec_arg(v, derived) for k, v in node.attrs.items()}
    spec = _view_out_spec(node.op, operand, attrs)
    return _UNX if spec is None else spec


def _tsub(match: dict, derived: dict, x: Any) -> Any:
    """Resolve a ``T``-position arg — metavar name or nested spec."""
    if isinstance(x, str):
        if x not in match:
            return _UNX
        piece = match[x]
        return (
            piece
            if isinstance(piece, str)
            else (_term_to_spec(piece, derived))
        )
    if isinstance(x, (tuple, list)) and x:
        return _spec_subst(x, match, derived)
    return x


def _asub(match: dict, derived: dict, x: Any) -> Any:
    """Resolve an ``A``-position arg — an attr metavar name."""
    if not isinstance(x, str):
        return (
            _dexpr_subst(x, match, derived)
            if isinstance(x, (tuple, list))
            else x
        )
    key = "$attr:" + x
    if key not in match:
        return _UNX
    return _spec_arg(match[key], derived)


def _spec_subst(spec: Any, match: dict, derived: dict) -> Any:
    """Re-express a shape spec's metavar references, or ``_UNX``."""
    if isinstance(spec, str):
        return _tsub(match, derived, spec)
    if not isinstance(spec, (tuple, list)) or not spec:
        return spec
    sig = _SPEC_REFS.get(spec[0])
    args = tuple(spec[1:])
    if sig is None or len(args) != len(sig):
        return _UNX
    out = [
        _subst_arg(k, a, match, derived)
        for k, a in zip(sig, args, strict=True)
    ]
    return _UNX if any(a is _UNX for a in out) else (spec[0], *out)


def _subst_sig(
    node: Any, sigs: dict, match: dict, derived: dict
) -> Any:
    """Positional-substitution tail shared by the two walkers.

    Looks up *node*'s op in *sigs*, substitutes each argument under
    its position kind — the shape both :func:`_dexpr_subst` (derive
    exprs) and :func:`_subst_cond` (predicates) reduce to after their
    own recursion clauses.
    """
    sig = sigs.get(node[0])
    args = tuple(node[1:])
    if sig is None or len(args) != len(sig):
        return _UNX
    out = [
        _subst_arg(k, a, match, derived)
        for k, a in zip(sig, args, strict=True)
    ]
    return _UNX if any(a is _UNX for a in out) else (node[0], *out)


def _dexpr_subst(node: Any, match: dict, derived: dict) -> Any:
    """Re-express a derive expr's metavar references, or ``_UNX``."""
    if node is None or isinstance(node, (bool, int, float)):
        return node
    if not isinstance(node, (tuple, list)) or not node:
        return _UNX
    if node[0] in ("tuple", "concat"):
        out = [_dexpr_subst(a, match, derived) for a in node[1:]]
        return _UNX if any(a is _UNX for a in out) else (node[0], *out)
    return _subst_sig(node, _DEXPR_REFS, match, derived)


def _subst_arg(kind: str, a: Any, match: dict, derived: dict) -> Any:
    """Substitute one datum argument under its position kind."""
    if kind == "T":
        return _tsub(match, derived, a)
    if kind == "A":
        return _asub(match, derived, a)
    if kind == "e":
        return _dexpr_subst(a, match, derived)
    return a


def _subst_comb(node: Any, match: dict, derived: dict) -> Any:
    """Substitute an ``and``/``or``/``not`` node of :func:`_subst_cond`."""
    op = node[0]
    if op == "not":
        inner = (
            _subst_cond(node[1], match, derived)
            if len(node) == 2
            else _UNX
        )
        return _UNX if inner is _UNX else ("not", inner)
    parts = [_subst_cond(c, match, derived) for c in node[1:]]
    if any(p is _UNX for p in parts):
        return _UNX
    return (op, *parts)


def _subst_cond(node: Any, match: dict, derived: dict) -> Any:
    """Re-express a declarative cond over the composite's metavars.

    ``_UNX`` marks a node the match cannot carry — an unknown op or
    arity mismatch, a term metavar bound to a concrete leaf — the
    caller then falls back to the procedural step.
    """
    if isinstance(node, bool):
        return node
    if not isinstance(node, (tuple, list)) or not node:
        return _UNX
    op = node[0]
    if op in ("and", "or", "not"):
        return _subst_comb(node, match, derived)
    return _subst_sig(node, _PRED_REFS, match, derived)


def _transport_clause(rule: Rewrite, match: dict, derived: dict) -> Any:
    """Re-express *rule*'s declarative guard over the composite's metavars.

    Returns the premise's ``cond`` substituted through the premise's
    match — metavar renames, compound pieces as their view-output
    specs, derived-attr reads inlined as their exprs — or ``None``
    when the guard is not purely declarative (a procedural ``check``
    remainder) or a reference cannot be re-expressed; the premise
    then rides the procedural path instead.
    """
    from catopt_core.laws.serialize import _proc_check

    if rule.cond is None or _proc_check(rule):
        return None
    clause = _subst_cond(rule.cond, match, derived)
    return None if clause is _UNX else clause


# ---------------------------------------------------------------------------
#  The procedural chain — re-running premise hooks on their own bindings
# ---------------------------------------------------------------------------
#
#  When a premise's guard or derive cannot ride the composite's data
#  fields (procedural hooks, or a reference the substitution cannot
#  carry), the composite keeps them as fire-time callables.  Each
#  premise's binding is reconstructed from the composite's bound env
#  plus the *accumulated derive outputs* — an earlier premise's minted
#  attrs feed a later premise's check, exactly the chaining the
#  premises' own fire order performs.  The chain is total: a
#  re-expression failure, a guard veto or a derive veto vetoes the
#  firing, never raises.


def _pat_bind(pats: dict, env: dict) -> dict | None:
    """Instantiate a recorded premise binding against *env*.

    *pats* maps premise metavariables to pattern pieces over the
    composite's metavars — the match dict captured at composition.
    ``$attr:`` entries holding a name resolve through ``env`` (which
    carries the accumulated derived attrs); term metavars instantiate
    through it.  ``None`` when a reference is unbound.
    """
    out = {}
    for k, v in pats.items():
        if k.startswith("$attr:"):
            if isinstance(v, str):
                key = "$attr:" + v
                if key not in env:
                    return None
                out[k] = env[key]
            else:
                out[k] = v
        else:
            try:
                out[k] = instantiate_pattern(v, env)
            except Exception:
                return None
    return out


def _emit_map(rule: Rewrite, extra: dict | None) -> dict | str | None:
    """How a premise's derive output maps onto composite attr names.

    ``None`` for no emission — a premise with no ``derive``, a purely
    declarative one (its minted names are covered by the composite's
    ``dspec`` transport or ride upstream names), or a derive whose
    symbolic output carried no metavar-strings to re-mint.  ``"all"``
    emits every output verbatim (the head premise — its minted names
    ARE the composite RHS metavar names).  Otherwise a
    ``{premise-key: composite-key}`` map built from the symbolic
    ``extra``: a str-valued output is the metavar name it left in the
    composite RHS.
    """
    from catopt_core.laws.serialize import _proc_derive

    if rule.derive is None:
        return None
    if _proc_derive(rule):
        if extra is None:
            return "all"
        emit = {
            k: ("$attr:" + v if k.startswith("$attr:") else v)
            for k, v in extra.items()
            if isinstance(v, str)
        }
        return emit or None
    return None


def _chain_step(
    rule: Rewrite,
    pat: dict,
    emit: Any,
    recheck: bool,
    env: dict,
    out: dict,
) -> bool:
    """Run one premise's hooks against its reconstructed binding.

    Re-expresses the premise's binding under the accumulated *env*,
    re-runs its ``check`` when *recheck*, then its ``derive`` — the
    outputs grow *env* for later steps, and *emit* selects which of
    them the composite's substitution reads into *out*.  A binding
    that cannot be re-expressed, a guard veto or a derive veto is a
    plain ``False`` — never a raise.
    """
    b = _pat_bind(pat, env)
    if b is None:
        return False
    try:
        if recheck and rule.check is not None and not rule.check(b):
            return False
        if rule.derive is None:
            return True
        e = rule.derive(b)
    except Exception:
        return False
    if e is None:
        return False
    env.update(e)
    _chain_emit(emit, e, out)
    return True


def _chain_emit(emit: Any, e: dict, out: dict) -> None:
    """Select a premise's derive outputs into the composite env."""
    if emit == "all":
        out.update(e)
        return
    if emit:
        for k, v in e.items():
            tgt = emit.get(k)
            if tgt is not None:
                out[tgt] = v


def _chain_hooks(steps: list) -> tuple:
    """Build the composite's procedural ``(check, derive)`` pair.

    *steps* is ``(rule, pat, emit, recheck)`` per premise in firing
    order — see the section note.  The shared evaluation replays the
    premises' fire order: each premise's binding is re-expressed under
    the accumulated env, its ``check`` re-runs when *recheck* (a guard
    the declarative transport could not carry), and its ``derive``
    always runs — for the env the later steps read and for the outputs
    *emit* selects into the composite's substitution.  Returns
    ``(None, None)`` when no step carries real work.
    """

    def _eval(bound: dict) -> dict | None:
        env = dict(bound)
        out: dict = {}
        for rule, pat, emit, recheck in steps:
            if not _chain_step(rule, pat, emit, recheck, env, out):
                return None
        return out

    if not any(
        s[2] or (s[3] and s[0].check is not None) for s in steps
    ):
        return None, None

    def check(bound: dict) -> bool:
        return _eval(bound) is not None

    def derive(bound: dict) -> dict | None:
        return _eval(bound)

    return (
        check if any(s[3] for s in steps) else None,
        derive if any(s[2] for s in steps) else None,
    )


def _fold_and(clauses: list) -> Any:
    """Fold guard clauses into one declarative cond datum (or ``None``)."""
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return ("and", *clauses)


def _premise_extra(rule: Rewrite, match: dict) -> dict | None:
    """Return the premise's derive output on *match*, or ``None``.

    On the decidable path a derive that cannot evaluate on the
    symbolic binding is tolerated — the premise's emit map simply
    carries nothing; :func:`_structural_fire` holds the stricter
    contract (an unevaluable derive declines the position).
    """
    if rule.derive is None:
        return None
    try:
        return rule.derive(dict(match))
    except Exception:
        return None


def _fire_premise(rule: Rewrite, cur: Any) -> tuple | None:
    """Fire *rule* on *cur*, decidable first then structurally.

    Returns ``(rewritten, match, decidable, extra)`` — the rewritten
    term, the premise's binding as patterns over the composite's
    metavars, whether the firing was decidable (its guard held on the
    symbolic binding — no guard to transport), and the derive's
    symbolic output — or ``None`` when the premise's LHS matches
    nowhere.
    """
    for path, sub in _positions(cur):
        nxt = apply_rewrite_at(rule, cur, path)
        if nxt is None or nxt == cur:
            continue
        match = match_pattern(rule.lhs, sub, {})
        if match is not None:
            return nxt, match, True, _premise_extra(rule, match)
    fired = _structural_fire(rule, cur)
    if fired is None:
        return None
    _path, nxt, match, extra = fired
    return nxt, match, False, extra


def _spec_map(specialize: dict | None) -> dict:
    """Build the ``{metavar: term}`` substitution from a spec map."""
    return {k: term_from_spec(v) for k, v in (specialize or {}).items()}


def _premise_names(rules: list) -> tuple[str, ...]:
    """Return the premises' names, deduplicated, in firing order."""
    return tuple(dict.fromkeys(r.name for r in rules))


def _head_transport(
    head: Rewrite, lhs: Any, has_caller_cond: bool, derived: dict
) -> tuple:
    """Transport the first premise's guard (it guards the composite LHS).

    Returns ``(clause, step, match)`` — *match* is the head's binding
    over the composite's LHS, the renaming map the derive transport
    also needs.  A declaratively re-expressible guard becomes a cond
    clause; otherwise the caller's ``cond`` is taken as the declared
    composite guard (the pre-existing contract) unless none was given,
    in which case the guard rides a procedural fire-time step.  A
    guarded head whose own LHS cannot bind the composite LHS (``None``
    *match*) must decline — the caller checks for it.
    """
    m0 = match_pattern(head.lhs, lhs, {})
    if m0 is None or (head.cond is None and head.check is None):
        return None, None, m0
    clause = _transport_clause(head, m0, derived)
    if clause is not None or has_caller_cond:
        return clause, None, m0
    return None, [head, m0, None, True], m0


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


def _head_derive(
    head: Rewrite, m0: dict, derived: dict, caller: dict, entries: list
) -> tuple:
    """Transport the head premise's derive mints into the composite.

    Pure ``dspec`` entries become composite spec entries — each
    expression substituted through the head's match and the *derived*
    map (so a later premise may inline them).  Returns
    ``(emit, derived)``: ``emit="all"`` when the derive rides the
    procedural chain instead — a procedural ``derive``, or a spec
    entry the substitution cannot express (the accumulated entries
    and the derived map roll back to the caller's own claims).
    """
    from catopt_core.laws.serialize import _proc_derive

    if _proc_derive(head):
        return "all", derived
    for dname, dx in head.dspec:
        e2 = _dexpr_subst(dx, m0, derived)
        if e2 is _UNX:
            entries.clear()
            return "all", dict(caller)
        entries.append((dname, e2))
        derived[dname] = e2
    return None, derived


def _premise_step(r: Rewrite, cur: Any, derived: dict) -> tuple | None:
    """Fire one ``rest`` premise on the composite RHS-in-progress.

    Returns ``(cur, clause, pending)`` — the rewritten RHS, the
    premise's transported declarative guard clause (``None`` when it
    rode the procedural path or was decidable), and the chain step to
    append (``None`` when the premise needs no fire-time hook) — or
    ``None`` when the premise matches nowhere.
    """
    fired = _fire_premise(r, cur)
    if fired is None:
        return None
    cur, match, decidable, extra = fired
    emit = _emit_map(r, extra)
    if decidable:
        pend = [r, match, emit, False] if emit else None
        return cur, None, pend
    clause = _transport_clause(r, match, derived)
    recheck = clause is None and r.check is not None
    pend = None
    if recheck or emit or r.derive is not None:
        pend = [r, match, emit, recheck]
    return cur, clause, pend


def _compose_head(
    head: Rewrite, lhs: Any, caller_cond: Any, caller: dict
) -> tuple | None:
    """Transport the first premise's guard *and* derive mints.

    The head's own guard applies to the composite's LHS
    (:func:`_head_transport`); its ``derive`` mints the attrs its RHS
    carries — pure ``dspec`` entries become composite spec entries
    (:func:`_head_derive`), anything else rides the procedural chain.
    Returns ``(clauses, pending, derived, entries)`` or ``None`` when
    the head's LHS cannot bind the composite's LHS.
    """
    derived = dict(caller)
    entries: list = []
    clause, step, m0 = _head_transport(
        head, lhs, caller_cond is not None, derived
    )
    if m0 is None:
        return None
    clauses = [clause] if clause is not None else []
    pending = [step] if step is not None else []
    if head.derive is not None:
        emit, derived = _head_derive(head, m0, derived, caller, entries)
        if emit is not None:
            if step is not None:
                step[2] = emit
            else:
                pending.append([head, m0, emit, False])
    return clauses, pending, derived, entries


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
    — the constructor's claim about where the composite is legal (on a
    name collision the caller's ``dspec`` entry wins).  A declaratively
    transported premise guard is ANDed onto *cond*; a procedural one
    rides ``check`` (the composite then reports the hooks it still
    carries — ``missing_hooks`` — at the store, honestly).

    Premise *derives* transport the same way (see the module note on
    derive transport): a head premise whose RHS mints attr metavariables
    contributes its ``dspec`` entries to the composite's — renamed
    through its own match — and a procedural premise ``derive`` rides
    the composite's fire-time ``derive``, the chained evaluation that
    reconstructs each premise's binding under the earlier premises'
    minted attrs.
    """
    from catopt_core.laws.cond import derive_from_data

    rules = [_as_rule(first), *(_as_rule(r) for r in rest)]
    subst = _spec_map(specialize)
    lhs = _specialize(rules[0].lhs, subst)
    cur = _specialize(rules[0].rhs, subst)
    caller = dict(derive_from_data(dspec)) if dspec is not None else {}
    head_res = _compose_head(rules[0], lhs, cond, caller)
    if head_res is None:
        return None
    clauses, pending, derived, entries = head_res
    for r in rules[1:]:
        step = _premise_step(r, cur, derived)
        if step is None:
            return None
        cur, clause, pend = step
        if clause is not None:
            clauses.append(clause)
        if pend is not None:
            pending.append(pend)
    return _compose_result(
        name,
        rules,
        lhs,
        cur,
        pending,
        clauses,
        cond,
        caller,
        entries,
        kind,
        tags,
    )


def _compose_result(
    name: str,
    rules: list,
    lhs: Any,
    cur: Any,
    pending: list,
    clauses: list,
    cond: Any,
    caller: dict,
    entries: list,
    kind: str,
    tags: Any,
) -> ConstructedObject:
    """Assemble the composite rule + object from transported parts.

    The premises name the replayable derivation; the pending chain
    becomes the procedural ``(check, derive)`` pair; the caller's
    declared ``dspec`` entries win over transported ones (``caller``
    merges last); the caller's ``cond`` conjoins ahead of the
    transported premise clauses.
    """
    premises = _premise_names(rules)
    ck, dr = _chain_hooks(pending) if pending else (None, None)
    merged = {**dict(entries), **caller}
    rule = Rewrite(
        name=name,
        lhs=lhs,
        rhs=cur,
        law="composite of " + " ∘ ".join(premises),
        check=ck,
        derive=dr,
        cond=_fold_and(([cond] if cond is not None else []) + clauses),
        dspec=merged or None,
        tags=frozenset(tags),
        derivation=premises,
    )
    return ConstructedObject(
        rule=rule,
        kind=kind,
        construction=("compose", *premises),
    )


# ---------------------------------------------------------------------------
#  Guard manipulation — relax / specialize
# ---------------------------------------------------------------------------
#
#  The arena's last two construction moves operate on the *guard* —
#  the lever the real-corpus measurement showed mattering
#  (``project/retros/auto-cond.md``: guards minted correctly but too
#  tight for real sites, or too coarse to fire).  Both mint a plain
#  :class:`ConstructedObject` on the same record path as
#  fold/lift/compose — store-able, reconstructable, gauntlet-able —
#  and both stay honest: the output is a *candidate*, never a proof.
#
#  * :func:`relax_guard` drops one conjunct of a declarative ``cond``.
#    An ``and`` minus a conjunct accepts a superset of the old region
#    — monotonic by construction — so the honest statement is only
#    "the region grew"; whether the grown region is still clean is
#    the guarded sweep's measurement, not the constructor's promise.
#  * :func:`specialize` narrows the object to an instance family: a
#    leaf metavariable binds a concrete term spec (``S ↦ 4.0`` — the
#    ``div`` divisor becomes a literal ``Const``), an attr metavariable
#    takes a value pinned by a fresh ``("attr-eq", NAME, v)`` conjunct
#    (``D=0`` on ``select_mul``).  The pin form keeps the attr
#    metavariable bound at match time, so sibling clauses still read
#    ``$attr:D`` — the narrower family without re-spelling the
#    pattern.


def _obj_kind(obj: Any) -> str:
    """Return the declaration kind a premise carries.

    A :class:`ConstructedObject` keeps its own ``kind``; a bare
    ``Rewrite`` has no declaration kind (its ``.kind`` property is the
    kernel taxonomy — axiom/lemma — not the object record's), so the
    introduced-object default applies.
    """
    return (
        obj.kind
        if isinstance(obj, ConstructedObject)
        else "abstraction"
    )


def _conjuncts(cond: Any) -> list | None:
    """Return *cond*'s top-level conjuncts, or ``None`` if unguarded.

    ``and`` flattens — ``("and", ("and", a, b), c)`` is three
    conjuncts ``[a, b, c]`` (conjunction is associative; a nested
    ``and`` carries no extra meaning).  Any other node — a bare
    predicate, an ``("or", …)``, ``("not", …)``, a ``False`` literal —
    is a single clause.  ``None`` / ``True`` carry no clause to drop.
    """
    if cond is None or cond is True:
        return None
    out: list = []

    def _flatten(n: Any) -> None:
        if isinstance(n, (tuple, list)) and n and n[0] == "and":
            for c in n[1:]:
                _flatten(c)
        else:
            out.append(n)

    _flatten(cond)
    return out


def _clause_index(clauses: list, clause: Any) -> int | None:
    """Resolve *clause* — an index or a clause datum — to a position.

    An ``int`` is a position into the top-level conjunction (Pythonic
    negatives allowed); anything else is matched as a clause datum,
    canonicalised through ``cond_from_data`` so a JSON-shaped list
    finds its tuple-tree twin.  ``None`` when the conjunct is absent.
    """
    if isinstance(clause, int) and not isinstance(clause, bool):
        return (
            clause if -len(clauses) <= clause < len(clauses) else None
        )
    want = cond_from_data(clause)
    for i, c in enumerate(clauses):
        if c == want:
            return i
    return None


def relax_guard(
    obj: Any,
    clause: Any,
    *,
    name: str | None = None,
    kind: str | None = None,
) -> ConstructedObject | None:
    """Drop one clause of *obj*'s declarative ``cond`` — the relaxed claim.

    *obj* is a ``Rewrite`` or :class:`ConstructedObject`; *clause*
    selects the conjunct to remove — an ``int`` index into the
    top-level ``("and", …)`` (or ``0`` for a single-clause guard), or
    the clause datum itself (lists canonicalise, so the JSON spelling
    works).  The result keeps ``lhs``/``rhs``, tags, ``dspec`` and any
    *procedural* ``check``/``derive`` remainder untouched — only the
    declarative conjunction weakens.  Dropping the last conjunct
    unguards the object (``cond=None``).

    The accepted region can only grow — an ``and`` minus a conjunct
    is a superset of itself — so a relaxed object is strictly a
    *candidate*: the guarded-region sweep re-measures whether the
    grown region stays clean, and the gauntlet decides.  The store
    keys records on ``(lhs, rhs)`` — the relaxed object shares the
    premise's alpha key, so storing it rewrites the guard under the
    same row, the in-place update ``auto-cond`` already performs.

    ``None`` is the honest decline: no declarative guard to relax
    (an unguarded object, or one whose guard is pure procedural
    ``check`` — code is not a clause and cannot be dropped), or
    *clause* names nothing in the conjunction.
    """
    from catopt_core.laws.serialize import _proc_check, _proc_derive

    rule = _as_rule(obj)
    clauses = _conjuncts(rule.cond)
    if clauses is None:
        return None
    i = _clause_index(clauses, clause)
    if i is None:
        return None
    dropped = clauses[i]
    new_rule = Rewrite(
        name=name or rule.name,
        lhs=rule.lhs,
        rhs=rule.rhs,
        law=rule.law,
        check=rule.check if _proc_check(rule) else None,
        derive=rule.derive if _proc_derive(rule) else None,
        tags=rule.tags,
        error_bound=rule.error_bound,
        bound_norm=rule.bound_norm,
        cond=_fold_and(clauses[:i] + clauses[i + 1 :]),
        dspec=rule.dspec,
    )
    return ConstructedObject(
        rule=new_rule,
        kind=kind or _obj_kind(obj),
        construction=("relax_guard", rule.name, repr(dropped)),
        note=(
            "dropped conjunct "
            f"{dropped!r} — the accepted region can only grow; "
            "the sweep re-measures it"
        ),
    )


# ---------------------------------------------------------------------------
#  Specialize — bound-env reference analysis for the cond fold
# ---------------------------------------------------------------------------
#
#  Specializing inlines a leaf metavariable into the pattern, which
#  removes the name from the matcher's ``bound`` env — a declarative
#  clause or derive expr that still reads it would dangle (the DSL's
#  strictness declines on a missing key, silently vacuating the rule).
#  The tables below name, per cond predicate / shape spec / derive
#  expr, which argument positions read the bound env: ``"T"`` a term
#  ref or shape spec (a metavar name or a nested spec), ``"A"`` an
#  attr-metavar name (resolved under ``"$attr:"``), ``"e"`` a nested
#  derive expr, ``"_"`` a literal.  The fold then decides a clause
#  exactly when every name it reads is bound, and refuses the
#  construction on a dangling read — never guesses.

#: ``("T" | "A" | "_")`` per argument of each cond predicate.
_PRED_REFS: dict[str, tuple] = {
    "shaped": ("T",),
    "concrete": ("T",),
    "scalar": ("T",),
    "uniform": ("T",),
    "ones-but-last": ("T",),
    "rank": ("T", "_", "_"),
    "rank-eq": ("T", "T"),
    "shape-eq": ("T", "T"),
    "shape-compat": ("T", "T"),
    "dim-eq": ("T", "_", "T", "_"),
    "dim-compat": ("T", "_", "T", "_"),
    "dim-eq-const": ("T", "_", "_"),
    "dim-eq-attr": ("T", "A", "T", "A"),
    "dim-mod": ("T", "A", "_", "_"),
    "bcast-into": ("T", "T"),
    "mm-shape-ok": ("T", "T"),
    "axes-last2": ("T", "A", "A"),
    "axes-distinct": ("T", "A", "A"),
    "axes-eq": ("T", "A", "A", "A", "A"),
    "axis": ("T", "A", "_"),
    "op-in": ("T", "_"),
    "leaf": ("T",),
    "const": ("T",),
    "term-eq": ("T", "T"),
    "const-num": ("T",),
    "const-cmp": ("T", "_", "_"),
    "attr-is": ("A", "_"),
    "attr-eq": ("A", "_"),
    "attr-in": ("A", "_"),
    "attr-type": ("A", "_"),
    "attr-len": ("A", "_", "_"),
    "attr-cmp-dim": ("A", "_", "T", "A"),
    "attr-eq-attr": ("A", "A"),
    "bcast-eq": ("T", "T", "T", "T"),
    "ones-before": ("T", "A"),
    "axes-noop": ("T", "A", "A"),
    "axis-align-eq": ("T", "T", "A"),
    "bcast-dim-inv": ("T", "T", "A"),
    "flat-pair-unsq": ("T", "A", "T", "T"),
    "flat-map-unsq": ("T", "A", "T", "T"),
    "repeat-chain": ("T", "A", "A", "A"),
    "repeat-heads": ("T", "T", "A", "A"),
    "attr-range": ("A", "T", "A"),
}

#: ``("T" | "A")`` per argument of each shape spec — ``None`` literal
#: slots (``slice-out``'s optional bounds) read nothing.
_SPEC_REFS: dict[str, tuple] = {
    "mm-out": ("T", "T"),
    "bcast": ("T", "T"),
    "unsq-out": ("T", "A"),
    "reshape-out": ("T", "A"),
    "getitem-out": ("T",),
    "select-out": ("T", "A"),
    "slice-out": ("T", "A", "A", "A", "A"),
    "chunk-out": ("T", "A", "A"),
    "transpose-out": ("T", "A", "A"),
    "tail-block": ("T", "A"),
    "isel-out": ("T", "A", "A"),
}

#: ``("e" | "T" | "A" | "_")`` per argument of each derive expr —
#: ``tuple``/``concat`` are variadic over ``"e"`` children.
_DEXPR_REFS: dict[str, tuple] = {
    "lit": ("_",),
    "attr": ("A",),
    "attr0": ("A",),
    "const": ("T",),
    "shape": ("T",),
    "dim": ("T", "_"),
    "leaf-dim": ("T", "_"),
    "len": ("e",),
    "gather": ("e", "e"),
    "unique": ("e",),
    "posmap": ("e", "e"),
    "bcast": ("T", "T"),
    "add": ("e", "e"),
    "sub": ("e", "e"),
    "mul": ("e", "e"),
    "fdiv": ("e", "e"),
    "floordiv": ("e", "e"),
    "neg": ("e",),
    "recip": ("e",),
    "float": ("e",),
    "int": ("e",),
}


def _attr_refs(a: Any) -> set | None:
    """Return the bound-env refs of an ``"A"`` position argument.

    A metavariable name reads its own ``$attr:`` binding; a literal
    tuple/list is an inlined derive expr whose refs walk
    :func:`_dexpr_refs`; a bare literal reads nothing.
    """
    if isinstance(a, str):
        return {"$attr:" + a}
    if isinstance(a, (tuple, list)):
        return _dexpr_refs(a)
    return set()


def _sig_refs(sig: tuple, args: tuple, e_fn: Any) -> set | None:
    """Union the bound-env refs of *args* under position table *sig*.

    ``"T"`` args recurse into :func:`_shape_refs`, ``"e"`` args into
    *e_fn*, ``"A"`` args read ``"$attr:<name>"``, ``"_"`` is a
    literal.  ``None`` propagates a malformed node — the caller
    refuses rather than guess.
    """
    refs: set = set()
    for kind, a in zip(sig, args, strict=True):
        if kind == "T":
            r = _shape_refs(a)
        elif kind == "e":
            r = e_fn(a)
        elif kind == "A":
            r = _attr_refs(a)
        else:
            r = set()
        if r is None:
            return None
        refs |= r
    return refs


def _shape_refs(spec: Any) -> set | None:
    """Return the bound-env keys a shape spec may read, or ``None``.

    A ``str`` is a metavariable ref; a tuple dispatches through
    :data:`_SPEC_REFS` (``None`` marks a malformed spec — unknown op
    or arity mismatch); any other literal reads nothing.
    """
    if isinstance(spec, str):
        return {spec}
    if not isinstance(spec, (tuple, list)) or not spec:
        return set()
    sig = _SPEC_REFS.get(spec[0])
    args = tuple(spec[1:])
    if sig is None or len(args) != len(sig):
        return None
    return _sig_refs(sig, args, _shape_refs)


def _cond_refs(node: Any) -> set | None:
    """Return the bound-env keys a cond node may read, or ``None``.

    Term refs come back bare (``bound[name]``), attr refs prefixed
    (``bound["$attr:name"]``).  ``None`` marks a node outside the
    predicate table — the specialize fold refuses rather than guess.
    """
    if isinstance(node, bool):
        return set()
    if not isinstance(node, (tuple, list)) or not node:
        return None
    op = node[0]
    if op in ("and", "or"):
        refs: set = set()
        for c in node[1:]:
            r = _cond_refs(c)
            if r is None:
                return None
            refs |= r
        return refs
    if op == "not":
        return _cond_refs(node[1]) if len(node) == 2 else None
    sig = _PRED_REFS.get(op)
    args = tuple(node[1:])
    if sig is None or len(args) != len(sig):
        return None
    return _sig_refs(sig, args, _cond_refs)


def _dexpr_refs(node: Any) -> set | None:
    """Return the bound-env keys a derive expr may read, or ``None``.

    Bare scalars are literals; a bare ``str`` is *malformed* in expr
    position (``_dexpr`` raises on it) — ``None``, like any unknown
    op or arity mismatch.
    """
    if node is None or isinstance(node, (bool, int, float)):
        return set()
    if not isinstance(node, (tuple, list)) or not node:
        return None
    if node[0] in ("tuple", "concat"):
        refs: set = set()
        for a in node[1:]:
            r = _dexpr_refs(a)
            if r is None:
                return None
            refs |= r
        return refs
    sig = _DEXPR_REFS.get(node[0])
    args = tuple(node[1:])
    if sig is None or len(args) != len(sig):
        return None
    return _sig_refs(sig, args, _dexpr_refs)


def _fold_comb(node: Any, bound: set, dead: set, env: dict) -> Any:
    """Fold the children of an ``and``/``or``/``not`` node.

    See :func:`_fold_bound` for the contract.  ``False`` absorbs an
    ``and`` (the node decides False), ``True`` absorbs an ``or``; the
    opposite constants are identity children and drop away.  A
    ``None`` (unexpressible) child vetoes the whole fold — the caller
    refuses the construction.
    """
    op = node[0]
    if op == "not":
        k = (
            _fold_bound(node[1], bound, dead, env)
            if len(node) == 2
            else None
        )
        if k is None:
            return None
        return not k if isinstance(k, bool) else ("not", k)
    absorbing = op == "and"  # and: False absorbs; or: True absorbs
    kids: list = []
    for c in node[1:]:
        f = _fold_bound(c, bound, dead, env)
        if f is None:
            return None
        if isinstance(f, bool):
            if f != absorbing:
                return f
            continue
        kids.append(f)
    if not kids:
        return absorbing
    return kids[0] if len(kids) == 1 else (op, *kids)


def _fold_bound(node: Any, bound: set, dead: set, env: dict) -> Any:
    """Evaluate *node* against the specialize binding (partial fold).

    *bound* is the bound-env key set the binding determines — the
    inlined leaf metavariables plus the pinned ``$attr:`` names;
    *dead* is the inlined-leaf subset; *env* maps inlined names to
    their terms and pinned attrs to their values.

    Returns ``True``/``False`` for a node the binding decides, the
    node verbatim when every name it reads stays live (a pinned attr
    metavariable still binds at match time, so a mixed clause
    survives), or ``None`` when the node reads an inlined leaf
    alongside live names — a dangling ref the DSL cannot re-express —
    or when a decided node's evaluation itself fails.  The caller
    refuses the construction on ``None`` and on a ``False`` root (a
    guard the binding falsifies is a vacuous family, not an object).
    """
    if isinstance(node, bool):
        return node
    if not isinstance(node, (tuple, list)) or not node:
        return None
    if node[0] in ("and", "or", "not"):
        return _fold_comb(node, bound, dead, env)
    refs = _cond_refs(node)
    if refs is None:
        return None
    if refs <= bound:
        try:
            return bool(eval_cond(node, env))
        except (ValueError, TypeError, KeyError, IndexError):
            return None
    if refs & dead:
        return None
    return node


def _norm_pin(value: Any) -> Any:
    """Return the canonical pin value — list attrs spell as tuples."""
    return tuple(value) if isinstance(value, list) else value


def _pin_clause(name: str, value: Any) -> tuple:
    """Return the clause pinning attr metavar *name* to *value*.

    ``None``/``bool`` compare by identity (``attr-is``), everything
    else by equality (``attr-eq``).
    """
    if value is None or isinstance(value, bool):
        return ("attr-is", name, value)
    return ("attr-eq", name, _norm_pin(value))


def _split_binding(rule: Rewrite, binding: dict) -> tuple | None:
    """Split *binding* into ``(leaf_subst, attr_pins)``, or decline.

    A ``$attr:``-prefixed key always names an attr metavar pin; a
    bare name binds a leaf metavar of the LHS when one exists — the
    value goes through :func:`term_from_spec`, and a bare ``str`` is
    a metavar rename, not a narrowing: declined.  Anything else is a
    *tentative* pin — the caller validates it against the
    *specialized* pattern's attr metavars (a bound spec can carry
    attr metavars of its own).  ``None`` on a non-str key.
    """
    leaf = set(_mv_names(rule.lhs))
    subst: dict = {}
    pins: list[tuple[str, Any]] = []
    for k, v in binding.items():
        if not isinstance(k, str):
            return None
        if k.startswith("$attr:"):
            pins.append((k[len("$attr:") :], _norm_pin(v)))
            continue
        if k in leaf:
            if isinstance(v, str):
                return None
            subst[k] = term_from_spec(v)
        else:
            pins.append((k, _norm_pin(v)))
    return subst, pins


def _specialize_dspec(
    dspec: Any, bound: set, dead: set, env: dict
) -> Any:
    """Rewrite a declarative ``dspec`` under the binding, or refuse.

    Per ``(NAME, expr)`` pair: an expr the binding fully determines
    evaluates once and folds to ``("lit", v)``; an expr reading only
    live names rides verbatim; an expr touching an inlined leaf it
    cannot fully evaluate is unexpressible — ``None``.  ``()`` for a
    ``None`` spec keeps "no derive" distinct from the refusal.
    """
    if dspec is None:
        return ()
    out: list = []
    for name, expr in dspec:
        refs = _dexpr_refs(expr)
        if refs is None:
            return None
        if refs <= bound:
            vals = eval_derive(((name, expr),), env)
            if vals is None:
                return None
            out.append((name, ("lit", vals[f"$attr:{name}"])))
            continue
        if refs & dead:
            return None
        out.append((name, expr))
    return tuple(out)


def _spec_split(rule: Rewrite, binding: dict) -> tuple | None:
    """Resolve *binding* against *rule* — the specialize fold's inputs.

    Returns ``(lhs2, rhs2, pins, bound, dead, env)``: the specialized
    patterns, the attr pins as ``(name, value)`` pairs, the bound-env
    key set the binding determines, the inlined-leaf subset, and the
    evaluation fragment (inlined names to terms, pinned attrs to
    values).  ``None`` is the honest decline — an unbound pin name,
    or a leaf name that is also an attr metavar in the result.
    """
    split = _split_binding(rule, binding)
    if split is None:
        return None
    subst, pins = split
    lhs2 = _specialize(rule.lhs, subst)
    rhs2 = _specialize(rule.rhs, subst)
    # Pin names resolve against the *specialized* LHS: a bound spec
    # can carry attr metavars of its own, and a leaf name that is
    # also an attr metavar would bind ambiguously — decline both.
    live_attrs = set(_attr_mvs(lhs2))
    if any(n not in live_attrs for n, _v in pins):
        return None
    if live_attrs & set(subst):
        return None
    # An inlined name is dead only if no bound spec reintroduced it —
    # ``{"h": ("add", ("mul", "a1", "h"), "x1")}`` leaves a live "h".
    live = set(_mv_names(lhs2)) | set(_mv_names(rhs2))
    dead = {m for m in subst if m not in live}
    bound = dead | {f"$attr:{n}" for n, _v in pins}
    env = {
        **{m: subst[m] for m in dead},
        **{f"$attr:{n}": v for n, v in pins},
    }
    return lhs2, rhs2, pins, bound, dead, env


def _spec_cond(
    cond: Any, bound: set, dead: set, env: dict, pins: list
) -> Any:
    """Fold the guard under the binding and conjunct the pins.

    Returns the new ``cond`` — the folded clauses plus the pin
    conjuncts (``None`` when everything discharged).  ``False`` is
    the refuse sentinel: a clause the binding falsifies or leaves
    dangling means the specialized family is vacuous or
    unexpressible, and the caller declines rather than mint it.
    """
    core = None
    if cond is not None:
        folded = _fold_bound(cond, bound, dead, env)
        if folded is None or folded is False:
            return False
        core = None if folded is True else folded
    return _fold_and(
        ([core] if core is not None else [])
        + [_pin_clause(n, v) for n, v in pins]
    )


def specialize(
    obj: Any,
    binding: dict,
    *,
    name: str | None = None,
    kind: str | None = None,
) -> ConstructedObject | None:
    """Bind metavariables of *obj* to concrete values — the narrower family.

    *obj* is a ``Rewrite`` or :class:`ConstructedObject`.  *binding*
    maps a name to its specialization:

    * a **leaf metavariable** takes a term spec — ``{"S": 4.0}`` turns
      the ``div`` divisor into ``Const(4.0)``; a compound spec narrows
      the metavar to that shape (``{"h": ("add", ("mul", "a1", "h"),
      "x1")}``, the ``compose_objects`` ``specialize=`` idiom).  The
      name leaves the matcher's binding env, so every ``cond`` clause
      and ``dspec`` expr that reads it must *fold* under the bound
      value: a clause decided ``True`` is discharged, ``False`` (or an
      unexpressible dangling read) declines the construction — the
      specialized family would be vacuous or unstatable.
    * an **attr metavariable** — ``{"D": 0}`` or the explicit
      ``{"$attr:D": 0}`` — takes a concrete value pinned by a fresh
      ``("attr-eq", NAME, v)`` conjunct (``attr-is`` for
      ``None``/``bool``).  The metavar stays bound at match time, so
      sibling clauses still see ``$attr:D`` — ``select_mul`` under
      ``D=0`` keeps its ``dim-eq-attr`` guard live while the pin
      narrows the accepted region to the ``D=0`` slice.

    The result is a strict narrowing — every binding the child accepts
      satisfies the parent's guard — and stays pure data: serializable,
      store-able, gauntlet-able.  ``None`` is the honest decline: an
      empty binding, a name the LHS binds nowhere, a bare-string value
      (a metavar rename is not a narrowing), an ambiguous
      leaf-and-attr name, or a cond/dspec fold that cannot carry the
      bound clause.
    """
    from catopt_core.laws.serialize import _proc_check, _proc_derive

    rule = _as_rule(obj)
    if not binding:
        return None
    spec = _spec_split(rule, binding)
    if spec is None:
        return None
    lhs2, rhs2, pins, bound, dead, env = spec
    cond_final = _spec_cond(rule.cond, bound, dead, env, pins)
    if cond_final is False:
        return None
    dspec2 = _specialize_dspec(rule.dspec, bound, dead, env)
    if dspec2 is None:
        return None
    new_rule = Rewrite(
        name=name or rule.name,
        lhs=lhs2,
        rhs=rhs2,
        law=rule.law,
        check=rule.check if _proc_check(rule) else None,
        derive=rule.derive if _proc_derive(rule) else None,
        tags=rule.tags,
        error_bound=rule.error_bound,
        bound_norm=rule.bound_norm,
        cond=cond_final,
        dspec=dspec2 or None,
    )
    desc = ", ".join(f"{k}={v!r}" for k, v in sorted(binding.items()))
    return ConstructedObject(
        rule=new_rule,
        kind=kind or _obj_kind(obj),
        construction=("specialize", rule.name, desc),
        note=f"bound {desc}; the instance family only narrows",
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


def _spec_isel(operand: Any, attrs: dict) -> tuple | None:
    """``index_select`` output spec (dim + index are attrs)."""
    d, i = attrs.get("dim"), attrs.get("index")
    if d is None or i is None:
        return None
    return ("isel-out", operand, d, i)


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
    "index_select": _spec_isel,
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


# ---------------------------------------------------------------------------
#  The constructor registry — the arena's action space as data
# ---------------------------------------------------------------------------
#
#  Plan 0020's moves are op names over this module's constructors.
#  The registry is the seam the arena drives — one name, one
#  constructor — and :func:`construct` is its dispatch entry point:
#  same store-able record path whatever the op.

#: Op name → constructor.  ``auto_cond``'s entry returns an
#: :class:`AutoCond` (its ``.object`` is the minted object); every
#: other entry returns a :class:`ConstructedObject` — or ``None``,
#: the honest decline.
CONSTRUCTORS: dict[str, Any] = {
    "fold": fold_object,
    "lift": lift_object,
    "compose": compose_objects,
    "auto_cond": auto_cond_object,
    "relax_guard": relax_guard,
    "specialize": specialize,
}


def construct(
    op: str,
    *args: Any,
    store: Any = None,
    corpus_hash: str = "",
    env: dict | None = None,
    universe: Any = None,
    **kwargs: Any,
) -> dict | None:
    """Run a construction op by name; return the object record or ``None``.

    The arena's action entry point: *op* names a registered
    constructor (:data:`CONSTRUCTORS`), *args* / *kwargs* forward to
    it verbatim, and the minted :class:`ConstructedObject` persists
    through :func:`store_constructed` when *store* (a sqlite3
    connection) is given — the same object-record path
    fold/lift/compose share, so the result is store-able,
    reconstructable and gauntlet-able regardless of which op minted
    it.  *env* / *universe* reach ``store_constructed``'s certificate
    materialization for derivation-carrying composites.

    Returns the object record — the ``evidence.stored_object`` row
    when stored, else the unstored :func:`object_record` — or
    ``None`` when the construction declines (a compose whose premise
    fires nowhere, an auto-cond refusal, a relax of an absent clause,
    a specialize over an unbound name).  An unknown *op* is a bug,
    not a decline — ``ValueError``.
    """
    fn = CONSTRUCTORS.get(op)
    if fn is None:
        raise ValueError(f"unknown construction op: {op!r}")
    res = fn(*args, **kwargs)
    obj = res.object if isinstance(res, AutoCond) else res
    if obj is None:
        return None
    if store is None:
        return object_record(obj)
    from catopt_discovery import evidence as ev

    key = store_constructed(
        store, obj, corpus_hash, env=env, universe=universe
    )
    return ev.stored_object(store, key)
