# ruff: noqa: RUF002
"""3-cells as computed intelligence: coherence + completion, never stored.

Two meta-level pieces on top of the e-graph kernel:

PART A — coherence stratification
---------------------------------
The rule set splits into *coherent* laws (commutativity, associativity,
identity, involution — pure symmetry that generates every equivalent
bracketing/permutation of the same computation) and *contentful* laws
(distribute, factor, lift, fold, split — rules that change the
computational structure).

Storing coherent variants is what explodes equality saturation: each
commutativity doubles a node's permutation class, each associativity
multiplies bracketings, and the contentful rules then fire on every
single one of them.  The alternative is to *compute* coherence instead
of *storing* it:

* :func:`canonicalize` eagerly rewrites a term into a canonical form —
  commutative-associative chains (add, mul) are flattened, identity
  elements dropped, children sorted by structural key and rebuilt
  balanced; associative-only chains (matmul, aff_compose) are flattened
  order-preservingly and rebuilt balanced (the Blelloch bracketing is a
  *canonical form*, not a search result); double negation cancels.
* :func:`stratified_run` canonicalizes the input and saturates with
  contentful rules only — coherent variants never enter the e-graph.
  On ``LinearRecurrence`` (T=6, d=8) this is the measured difference
  between 100,353 e-nodes (all laws) and 178 (stratified) — ~560x —
  with the extracted term verified numerically identical in fp64.

PART B — rule synthesis via critical-pair completion
----------------------------------------------------
:func:`synthesize_rules` derives new rewrite rules by composing ordered
pairs of existing rules, in two ways:

* *Seed-guided composition* (preferred): apply ``r1`` to a concrete
  seed term at position ``p``, then ``r2`` at position ``q``; the
  composite rule is
  ``abstract(subterm(s, lca(p,q))) -> abstract(subterm(result, lca(p,q)))``
  where abstraction renames every metavariable-bound or untouched leaf
  to a fresh metavariable but keeps leaves that were matched against
  concrete pattern leaves.  The fired binding becomes the derived
  rule's *witness* — validation is guaranteed an instance on which
  both parents' guards actually hold.
* *Symbolic composition*: instantiate ``r1``'s RHS pattern with its own
  metavariables (as opaque symbolic leaves) and try to match ``r2``'s
  LHS against the result.  A match at position ``q`` yields the derived
  rule ``r1.lhs -> (r1.rhs with r2 applied at q)`` — fully general by
  construction, but with no witness.  Guarded symbolic candidates are
  validated against seed-derived instances first (see
  :func:`_validate_candidate`), falling back to the bounded
  instantiation pool.

Guarded parents are handled soundly: a derived rule's ``check`` is the
CONJUNCTION of both parents' checks, each evaluated on its own binding
re-expressed through the intermediate substitution, and a parent's
``derive``-produced attributes travel as namespaced placeholders the
derived rule's own ``derive`` recomputes at fire time.  Compositions
whose side conditions cannot be re-expressed — or cannot be satisfied
on any validation instance — are rejected, never emitted.

Every emitted rule is validated: rhs term metavars ⊆ lhs metavars,
the r1-then-r2 derivation is replayed on a fresh instantiation (or on
the seed derivation's own witness binding), and (when every op has a
torch binding) both sides are evaluated on random tensors and compared
with ``allclose``.  Provenance — which parent pair a rule descends
from — is recorded in ``law``, on ``rule.parents``, and in
:data:`SYNTH_PARENTS`.

Example finding: given ``SCAN_LAWS`` minus ``aff_lift_step`` and a
two-step unrolled recurrence seed, synthesis emits the *unfolded
equivalent* of ``aff_lift_step`` — the two-step fusion

    add(mm(A2, add(mm(A1, h), u)), x)
        -> apply(aff(A2, x), apply(aff(A1, u), h))

which is ``aff_lift_step`` composed with ``aff_compose_unfold``.  The
compose-headed form itself is unreachable by pair-composition because
no rule in the set *introduces* ``aff_compose`` from non-compose
structure — the nested-``apply`` form is the honest canonical
representative of the same critical pair.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import catopt.rules as R
from catopt.egraph import EGraph, Rewrite
from catopt.ir import Const, Op, Param, Var, op_repr

# ---------------------------------------------------------------------------
#  Part A.1 — coherence classification
# ---------------------------------------------------------------------------

#: Rules that express pure coherence: symmetry/associativity/identity/
#: involution.  They generate every *equivalent bracketing* of the same
#: computation — the search-space the e-graph should never store.
#: ``om_assoc``/``om_assoc_rev`` live in catopt.om (same law family:
#: associativity of a monoid's compose).
COHERENT_RULE_NAMES: frozenset[str] = frozenset(
    {
        "comm_add",
        "comm_mul",  # SMC symmetry
        "assoc_add",
        "assoc_mul",  # additive/multiplicative associativity
        "id_add",
        "id_mul",  # monoid units
        "double_neg",  # involution
        "assoc_matmul",
        "assoc_matmul_rev",  # associativity of composition
        "aff_assoc",
        "aff_assoc_rev",  # affine-map monoid associativity
        "affd_assoc",
        "affd_assoc_rev",  # diagonal-affine monoid assoc.
        "om_assoc",
        "om_assoc_rev",  # online-softmax monoid assoc.
    }
)


def module_rules(mod=R) -> list[Rewrite]:
    """Collect every Rewrite object defined in a rules module."""
    return _iter_module_rules(mod)


def _iter_module_rules(mod) -> list[Rewrite]:
    """Collect every Rewrite object defined in a rules module."""
    out: list[Rewrite] = []
    for v in vars(mod).values():
        if isinstance(v, Rewrite):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(x for x in v if isinstance(x, Rewrite))
    seen: set[int] = set()
    uniq = []
    for r in out:
        if id(r) not in seen:
            seen.add(id(r))
            uniq.append(r)
    return uniq


def classify_rules(
    rules: Iterable[Rewrite],
) -> tuple[list[Rewrite], list[Rewrite]]:
    """Split *rules* into ``(coherent, contentful)`` by name."""
    coherent = [r for r in rules if r.name in COHERENT_RULE_NAMES]
    contentful = [r for r in rules if r.name not in COHERENT_RULE_NAMES]
    return coherent, contentful


# ---------------------------------------------------------------------------
#  Part A.2 — canonicalize: eager coherence normalization
# ---------------------------------------------------------------------------

#: Commutative + associative ops with a dropped identity element.
_AC_IDENTITY: dict[str, float] = {"add": 0.0, "mul": 1.0}

#: Associative but NOT commutative ops: chains flatten in order and
#: rebuild balanced.  For ``aff_compose`` the balanced form is the
#: parallel-scan (Blelloch) bracketing — computed here, not searched.
_ASSOC_ONLY: frozenset[str] = frozenset(
    {"matmul", "aff_compose", "affd_compose"}
)


def _sort_key(t: Any) -> str:
    return op_repr(t)


def _balanced(op: str, items: list[Any], attrs: dict) -> Any:
    """Rebuild a flattened chain as a balanced binary tree."""
    if len(items) == 1:
        return items[0]
    mid = len(items) // 2
    return Op.make(
        op,
        _balanced(op, items[:mid], attrs),
        _balanced(op, items[mid:], attrs),
        **attrs,
    )


def _flatten_chain(term: Op) -> list[Any]:
    """Collect the operands of a same-op/same-attrs chain."""
    out: list[Any] = []

    def go(t: Any) -> None:
        if (
            isinstance(t, Op)
            and t.op == term.op
            and dict(t.attrs) == dict(term.attrs)
        ):
            for a in t.args:
                go(a)
        else:
            out.append(t)

    go(term)
    return out


def canonicalize(term: Any, memo: dict | None = None) -> Any:
    """Return the coherence-canonical form of *term*.

    * ``add``/``mul`` chains: flattened, identity elements (0 / 1)
      dropped, remaining children sorted by ``op_repr`` and rebuilt
      balanced.  Collapses every comm/assoc/id-equivalent form.
    * ``matmul``/``aff_compose`` chains: flattened in order (they are
      associative, not commutative) and rebuilt balanced — the
      canonical log-depth bracketing.
    * ``neg(neg(t))`` collapses to ``t``.

    Memoised on the term itself (hash-consed terms are content-keyed)
    so shared-subterm DAGs stay linear.  The function is idempotent
    and semantics-preserving (all transforms are instances of the
    coherent laws).
    """
    if memo is None:
        memo = {}
    key = term
    if key in memo:
        return memo[key]
    if not isinstance(term, Op):
        memo[key] = term
        return term

    args = tuple(canonicalize(a, memo) for a in term.args)
    out = Op.make(term.op, *args, **dict(term.attrs))

    if out.op == "neg":
        inner = out.args[0]
        if isinstance(inner, Op) and inner.op == "neg":
            out = inner.args[0]
    elif out.op in _AC_IDENTITY:
        flat = _flatten_chain(out)
        ident = _AC_IDENTITY[out.op]
        flat = [
            t
            for t in flat
            if not (isinstance(t, Const) and t.value == ident)
        ]
        if not flat:
            out = Const(ident)
        elif len(flat) == 1:
            out = flat[0]
        else:
            flat = sorted(flat, key=_sort_key)
            out = _balanced(out.op, flat, dict(out.attrs))
    elif out.op in _ASSOC_ONLY:
        flat = _flatten_chain(out)
        if len(flat) == 1:  # pragma: no cover — binary chains yield >=2 leaves
            out = flat[0]
        elif len(flat) != len(out.args):
            out = _balanced(out.op, flat, dict(out.attrs))
        # len==2 already binary: keep (canonical balanced form of 2)

    memo[key] = out
    return out


# ---------------------------------------------------------------------------
#  Part A.3 — stratified_run
# ---------------------------------------------------------------------------


def stratified_run(
    eg: EGraph,
    rules: list[Rewrite],
    term: Any,
    *,
    max_iterations: int = 100,
    max_nodes: int = 100_000,
    cost_fn=None,
    extract: bool = True,
    canonicalize_output: bool = True,
    extract_fn=None,
) -> dict:
    """Canonicalize *term*, saturate with the CONTENTFUL rules only.

    The coherent laws are never run: equivalent bracketings/permutations
    are collapsed by :func:`canonicalize` before they can multiply in
    the e-graph.  If ``cost_fn`` is given, the best term is extracted
    and (with ``canonicalize_output``) re-canonicalized — post-hoc
    normalization recovers the balanced coherent form of whatever the
    contentful rules produced.

    Returns a dict with ``root_eid``, ``stats`` (from ``eg.run``), the
    canonicalized input, the dropped/used rule names, and (when
    ``extract``) ``best``/``canonical_best``.
    """
    canon = canonicalize(term)
    root = eg.add_term(canon)
    coherent, contentful = classify_rules(rules)
    stats = eg.run(
        contentful,
        root,
        max_iterations=max_iterations,
        max_nodes=max_nodes,
    )
    out = {
        "root_eid": root,
        "stats": stats,
        "canonical_input": canon,
        "coherent_dropped": [r.name for r in coherent],
        "contentful_used": [r.name for r in contentful],
    }
    if extract and (cost_fn is not None or extract_fn is not None):
        # Depth (and other max-composed measures) are not additive, so
        # extract_best cannot rank them — pass extract_fn =
        # eg.extract_min_depth for those objectives.
        best = (
            extract_fn(root)
            if extract_fn is not None
            else eg.extract_best(root, cost_fn)
        )
        out["best"] = best
        out["canonical_best"] = (
            canonicalize(best)
            if canonicalize_output and best is not None
            else best
        )
    return out


# ---------------------------------------------------------------------------
#  Part B — rule synthesis via critical-pair completion
# ---------------------------------------------------------------------------


#: ops with at least one string attr value carry *attribute*
#: metavariables (bound into the substitution under "$attr:<name>").
def _has_attr_metavars(pat: Any) -> bool:
    if isinstance(pat, Op):
        if any(isinstance(v, str) for v in pat.attrs.values()):
            return True
        return any(_has_attr_metavars(a) for a in pat.args)
    return False


def _synthesizable(r: Rewrite) -> bool:
    """A rule participates in synthesis iff it is *guarded-sound*:
    every metavariable in its RHS is either bound by the LHS or — for
    attribute metavars only — producible by ``derive``.

    ``check``/``derive`` hooks and attribute metavars no longer exclude
    a rule: parent side conditions are re-expressed on the derived
    rule's own substitution (see :func:`_compose_guards`), and
    derive-produced attributes are carried through the derivation as
    namespaced placeholders (``"@1:SD"`` / ``"@2:SD"``) that the derived
    rule's own ``derive`` fills at fire time.  A rule whose RHS needs an
    attribute it can neither match nor derive stays excluded — there is
    no sound way to propagate it."""
    lhs_mv = pattern_metavars(r.lhs)
    lhs_terms = {v for v in lhs_mv if not v.startswith("$attr:")}
    lhs_attrs = lhs_mv - lhs_terms
    for v in pattern_metavars(r.rhs):
        if v.startswith("$attr:"):
            if v not in lhs_attrs and r.derive is None:
                return False
        elif v not in lhs_terms:
            return False
    return True


def match_pattern(
    pat: Any, term: Any, subst: dict | None = None
) -> dict | None:
    """One-sided match of pattern *pat* against a concrete *term*.

    ``str`` leaves in the pattern are metavariables binding any subterm;
    repeated metavars must bind equal subterms (structural equality).
    String attr values are attribute metavariables bound under
    ``"$attr:<name>"`` — mirroring :meth:`EGraph._match`.  Returns the
    substitution dict or ``None``.
    """
    if subst is None:
        subst = {}
    if isinstance(pat, str):
        if pat in subst:
            return subst if subst[pat] == term else None
        subst[pat] = term
        return subst
    if isinstance(pat, Op):
        if (
            not isinstance(term, Op)
            or term.op != pat.op
            or len(term.args) != len(pat.args)
        ):
            return None
        if set(term.attrs) != set(pat.attrs):
            return None
        for k, pv in pat.attrs.items():
            tv = term.attrs[k]
            if isinstance(pv, str):
                key = "$attr:" + pv
                if key in subst:
                    if subst[key] != tv:
                        return None
                else:
                    subst[key] = tv
            elif tv != pv:
                return None
        for pa, ta in zip(pat.args, term.args, strict=True):
            subst = match_pattern(pa, ta, subst)
            if subst is None:
                return None
        return subst
    return subst if pat == term else None


def instantiate_pattern(pat: Any, subst: dict) -> Any:
    """Instantiate a pattern term: str leaves and str attr values are
    looked up in *subst*; everything else is rebuilt as-is."""
    if isinstance(pat, str):
        return subst[pat]
    if isinstance(pat, Op):
        args = tuple(instantiate_pattern(a, subst) for a in pat.args)
        attrs = {}
        for k, v in pat.attrs.items():
            attrs[k] = subst["$attr:" + v] if isinstance(v, str) else v
        return Op.make(pat.op, *args, **attrs)
    return pat


def pattern_metavars(pat: Any) -> set[str]:
    """All metavariables (str leaves + str attr values) in a pattern."""
    out: set[str] = set()
    if isinstance(pat, str):
        out.add(pat)
    elif isinstance(pat, Op):
        for a in pat.args:
            out |= pattern_metavars(a)
        for v in pat.attrs.values():
            if isinstance(v, str):
                out.add("$attr:" + v)
    return out


def _positions(
    term: Any, path: tuple = ()
) -> Iterable[tuple[tuple, Any]]:
    """Yield ``(path, subterm)`` for every position, DFS pre-order."""
    yield path, term
    if isinstance(term, Op):
        for i, a in enumerate(term.args):
            yield from _positions(a, (*path, i))


def _subterm(term: Any, path: tuple) -> Any:
    for i in path:
        term = term.args[i]
    return term


def _replace(term: Any, path: tuple, new: Any) -> Any:
    if not path:
        return new
    i, rest = path[0], path[1:]
    args = list(term.args)
    args[i] = _replace(args[i], rest, new)
    return Op.make(term.op, *args, **dict(term.attrs))


def _common_prefix(p: tuple, q: tuple) -> tuple:
    out = []
    for a, b in zip(p, q, strict=False):
        if a != b:
            break
        out.append(a)
    return tuple(out)


def _overlapping(p: tuple, q: tuple) -> bool:
    """True when one position is an ancestor of (or equal to) the other."""
    lca = _common_prefix(p, q)
    return lca in (p, q)


def apply_rewrite_at(
    rule: Rewrite, term: Any, path: tuple
) -> Any | None:
    """Apply *rule* to ``term`` at *path*; return the rewritten term or
    ``None`` if the LHS doesn't match or a guard vetoes the firing.

    ``check``/``derive`` hooks ARE evaluated here: at the term level the
    fired substitution already maps metavariables to terms, which is
    exactly the ``bound`` convention the e-graph uses (``any_term``-
    resolved bindings).  A raising hook counts as a veto — a side
    condition that cannot be evaluated can never justify a rewrite."""
    subst = match_pattern(rule.lhs, _subterm(term, path), {})
    if subst is None:
        return None
    if rule.check is not None:
        try:
            if not rule.check(subst):
                return None
        except Exception:
            return None
    if rule.derive is not None:
        try:
            extra = rule.derive(subst)
        except Exception:
            return None
        if extra is None:
            return None
        subst = {**subst, **extra}
    try:
        rhs = instantiate_pattern(rule.rhs, subst)
    except KeyError:
        return None
    return _replace(term, path, rhs)


# ---------------------------------------------------------------------------
#  Guarded composition — re-expressing parent side conditions
# ---------------------------------------------------------------------------
#
#: Attribute metavariables produced by a parent's ``derive`` hook are
#: carried through a derivation as *namespaced placeholders*: the
#: string ``"@<i>:<name>"`` where ``i`` is 1 for the first parent and 2
#: for the second.  A placeholder is an ordinary attr metavariable in
#: the derived patterns; the derived rule's own ``derive`` (built by
#: :func:`_compose_guards`) re-runs the parent hooks at fire time and
#: fills ``"$attr:@i:name"`` in the substitution.  Namespacing keeps the
#: parents' metavar namespaces disjoint — two parents may both derive a
#: ``"SD"`` attr with different meanings.
_DRV_PREFIX = ("@1:", "@2:")


class _Unreexpressible(Exception):
    """A parent binding cannot be re-expressed on the derived rule's
    metavariables — the composition must be rejected, never guessed."""


#: Provenance of every emitted rule: name -> (first parent, second parent).
SYNTH_PARENTS: dict[str, tuple[str, str]] = {}


def provenance(rule: Rewrite) -> tuple[str, ...]:
    """The parent rules a synthesized rule descends from (``()`` for a
    non-synthesized rule).  Recorded on the rule as ``.parents`` and in
    :data:`SYNTH_PARENTS`."""
    return getattr(rule, "parents", SYNTH_PARENTS.get(rule.name, ()))


def _fire_guarded(rule: Rewrite, term: Any, path: tuple, ns: str):
    """Fire *rule* on a CONCRETE term like :func:`apply_rewrite_at`, but
    keep ``derive``-produced attributes symbolic under the *ns*
    placeholder prefix.

    Returns ``(subst, rewritten, concrete_attrs)`` where
    ``concrete_attrs`` maps each placeholder name to the value the
    parent actually derived on this instance — needed to evaluate a
    second parent's guards on the intermediate term.  ``None`` on
    no-match or guard veto."""
    subst = match_pattern(rule.lhs, _subterm(term, path), {})
    if subst is None:
        return None
    try:
        if rule.check is not None and not rule.check(subst):
            return None
    except Exception:
        return None
    inst = dict(subst)
    conc: dict[str, Any] = {}
    if rule.derive is not None:
        try:
            extra = rule.derive(subst)
        except Exception:
            return None
        if extra is None:
            return None
        for k, v in extra.items():
            if k.startswith("$attr:"):
                ph = ns + k[len("$attr:") :]
                inst[k] = ph  # stays a metavar in the derived rule
                conc[ph] = v  # but this instance's value is known
            else:
                inst[k] = v
    try:
        rhs = instantiate_pattern(rule.rhs, inst)
    except KeyError:
        return None
    return subst, _replace(term, path, rhs), conc


def _concretize_attrs(t: Any, conc: dict) -> Any:
    """Replace placeholder attr values inside a concrete term using the
    fired *conc* map (placeholder name -> derived value)."""
    if isinstance(t, Op):
        attrs = {
            k: (conc.get(v, v) if isinstance(v, str) else v)
            for k, v in t.attrs.items()
        }
        return Op.make(
            t.op, *(_concretize_attrs(a, conc) for a in t.args), **attrs
        )
    return t


def _reexpress_term(t: Any, names: dict, keep: set) -> Any:
    """Re-express a term bound during a seed-guided derivation as a
    pattern over the derived rule's metavariables.

    Region leaves become their ``names`` metavariable; leaves pinned by
    a parent's concrete pattern (*keep*) and literal constants stay
    concrete.  A metavar leaf (symbolic path) passes through.  Anything
    else — a leaf the derived rule does not bind — is un-reexpressible."""
    if isinstance(t, Op):
        return Op.make(
            t.op,
            *(_reexpress_term(a, names, keep) for a in t.args),
            **dict(t.attrs),
        )
    if isinstance(t, str):
        return t
    if t in keep:
        return t
    if t in names:
        return names[t]
    if isinstance(t, Const):
        return t
    raise _Unreexpressible(t)


def _reexpress_map(m: dict, names: dict, keep: set) -> dict:
    """Re-express a whole fired binding (term metavars + ``$attr:``
    entries).  ``$attr:`` values pass through: concrete ones are baked
    in, string ones are metavar references resolved at fire time."""
    out = {}
    for k, v in m.items():
        if k.startswith("$attr:"):
            out[k] = v
        else:
            out[k] = _reexpress_term(v, names, keep)
    return out


def _reexpress_binding(pats: dict, subst: dict) -> dict | None:
    """Instantiate a re-expressed binding against a fired substitution.

    Term metavars instantiate as patterns; a ``$attr:`` entry that is a
    string means "the attribute metavariable of that name" and is
    resolved through ``subst`` — a missing key is an un-reexpressible
    reference and rejects the composition (returns ``None``)."""
    out = {}
    for k, pat in pats.items():
        if k.startswith("$attr:"):
            if isinstance(pat, str):
                key = "$attr:" + pat
                if key not in subst:
                    return None
                out[k] = subst[key]
            else:
                out[k] = pat
        else:
            try:
                out[k] = instantiate_pattern(pat, subst)
            except (KeyError, Exception):
                return None
    return out


def _compose_guards(r1: Rewrite, r2: Rewrite, pat1: dict, pat2: dict):
    """Build the derived rule's ``(check, derive)`` from the parents'.

    ``pat1``/``pat2`` map each parent's metavariables to patterns over
    the DERIVED rule's metavariables (symbolic path: ``pat1`` is the
    identity and ``pat2`` is r2's symbolic match; seed path: both are
    the fired bindings re-abstracted through the region's leaf names).

    The composite check on a fired binding ``bound`` is:

        r1.check(bound1) ∧ r2.check(bound2)
        where bound_i = instantiate(pat_i, bound ∪ parents' derives)

    — i.e. each parent's side condition evaluated on ITS OWN binding,
    re-expressed through the intermediate substitution.  ``derive``
    hooks are also re-run (they can veto) and their outputs fill the
    ``"@i:"`` placeholders the derived RHS carries.  Returns
    ``(None, None)`` when neither parent is guarded."""
    if (
        r1.check is None
        and r1.derive is None
        and r2.check is None
        and r2.derive is None
    ):
        return None, None
    ns1, ns2 = _DRV_PREFIX

    def _eval(bound: dict):
        """Evaluate both parents' guards on the derived binding.

        Returns ``(bound1, extra1, bound2, extra2)`` or ``None`` if any
        re-expression fails or any hook vetoes/raises."""
        b1 = _reexpress_binding(pat1, bound)
        if b1 is None:
            return None
        try:
            if r1.check is not None and not r1.check(b1):
                return None
        except Exception:
            return None
        # pat2 is instantiated over the DERIVED rule's metavars (bound),
        # never r1's namespace — merging b1 here would clobber a derived
        # metavar that happens to share a name with an r1 metavar (e.g.
        # om_split binds "v1"/"v2", the same names leaf-generalization
        # mints).  Only r1's derive-produced attrs join, namespaced.
        eff1 = dict(bound)
        e1 = None
        if r1.derive is not None:
            try:
                e1 = r1.derive(b1)
            except Exception:
                return None
            if e1 is None:
                return None
            for k, v in e1.items():
                if k.startswith("$attr:"):
                    eff1["$attr:" + ns1 + k[len("$attr:") :]] = v
                else:
                    eff1[k] = v
        b2 = _reexpress_binding(pat2, eff1)
        if b2 is None:
            return None
        try:
            if r2.check is not None and not r2.check(b2):
                return None
        except Exception:
            return None
        e2 = None
        if r2.derive is not None:
            try:
                e2 = r2.derive(b2)
            except Exception:
                return None
            if e2 is None:
                return None
        return b1, e1, b2, e2

    def check(bound: dict) -> bool:
        return _eval(bound) is not None

    def derive(bound: dict) -> dict | None:
        ctx = _eval(bound)
        if ctx is None:
            return None
        _, e1, _, e2 = ctx
        out: dict[str, Any] = {}
        for e, ns in ((e1, ns1), (e2, ns2)):
            if not e:
                continue
            for k, v in e.items():
                if k.startswith("$attr:"):
                    out["$attr:" + ns + k[len("$attr:") :]] = v
                else:
                    out[k] = v
        return out

    return check, derive


def _rhs_derive_placeholders(rhs: Any) -> set[str]:
    """Namespaced derive placeholders (``"@1:X"`` / ``"@2:X"``) that
    appear as attribute metavars in a derived RHS — the keys its
    composite ``derive`` must provide."""
    out: set[str] = set()
    if isinstance(rhs, Op):
        for v in rhs.attrs.values():
            if isinstance(v, str) and any(
                v.startswith(p) for p in _DRV_PREFIX
            ):
                out.add(v)
        for a in rhs.args:
            out |= _rhs_derive_placeholders(a)
    return out


def _concrete_matched_leaves(
    pat: Any, term: Any, out: set | None = None
) -> set:
    """Leaf values that a pattern pinned down with *concrete* leaves.

    Those leaves may NOT be generalized to metavariables in a derived
    rule: the parent rule only fires on that exact constant.
    """
    if out is None:
        out = set()
    if isinstance(pat, str):
        return out  # metavar: bound subterm is free to generalize
    if isinstance(pat, Op):
        if isinstance(term, Op) and term.op == pat.op:
            for pa, ta in zip(pat.args, term.args, strict=False):
                _concrete_matched_leaves(pa, ta, out)
        return out
    out.add(term)  # concrete pattern leaf pinned this value
    return out


def _leaf_generalize(
    region: Any,
    names: dict,
    keep: set,
    counter: list[int],
    assign_fresh: bool,
) -> Any:
    """Rename each leaf of *region* to a metavariable (shared ``names``
    map keeps lhs/rhs consistent), except leaves in *keep* which were
    concrete-matched and must stay literal.  With ``assign_fresh=False``
    (the RHS pass) an unseen leaf is a constant introduced by a rule
    pattern — it stays concrete."""
    if isinstance(region, Op):
        return Op.make(
            region.op,
            *(
                _leaf_generalize(a, names, keep, counter, assign_fresh)
                for a in region.args
            ),
            **dict(region.attrs),
        )
    if region in keep:
        return region
    if region in names:
        return names[region]
    if not assign_fresh:
        return region
    name = f"v{counter[0]}"
    counter[0] += 1
    names[region] = name
    return name


def _alpha_key(lhs: Any, rhs: Any) -> tuple[str, str]:
    """Metavariable-renaming-invariant key for a (lhs, rhs) pair."""
    names: dict[str, str] = []

    def norm(t: Any, table: list) -> Any:
        if isinstance(t, str):
            for i, n in enumerate(table):
                if n == t:
                    return f"${i}"
            table.append(t)
            return f"${len(table) - 1}"
        if isinstance(t, Op):
            return Op.make(
                t.op, *(norm(a, table) for a in t.args), **dict(t.attrs)
            )
        return t

    return op_repr(norm(lhs, names)), op_repr(norm(rhs, names))


def _eval_term(term: Any, env: dict) -> Any:
    """Evaluate a term with concrete leaves against ``env`` (leaf ->
    tensor), using the torch_bridge op bindings.  Returns a tensor or a
    nested tuple (aff/om carriers)."""
    from catopt.torch_bridge import _IR_TO_TORCH

    if isinstance(term, Const):
        import torch

        return torch.tensor(term.value)
    if isinstance(term, (Var, Param)):
        return env[term]
    if isinstance(term, Op):
        fn = _IR_TO_TORCH.get(term.op)
        if fn is None:
            raise KeyError(term.op)
        args = [_eval_term(a, env) for a in term.args]
        return fn(*args, **dict(term.attrs))
    raise TypeError(term)


def _eval_allclose(a: Any, b: Any, tol: float = 1e-6) -> bool:
    import torch

    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(
            _eval_allclose(x, y, tol) for x, y in zip(a, b, strict=True)
        )
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.allclose(a, b, atol=tol, rtol=tol))
    return False


def _term_is_ground(t: Any) -> bool:
    """No leftover str metavariables anywhere."""
    if isinstance(t, str):
        return False
    if isinstance(t, Op):
        if any(isinstance(v, str) for v in t.attrs.values()):
            return False
        return all(_term_is_ground(a) for a in t.args)
    return True


def _replays(
    derived: Rewrite,
    r1: Rewrite,
    r2: Rewrite,
    t0: Any,
    target: Any,
    fuel: int = 400,
) -> bool:
    """Check the derivation replays: some r1 application on *t0* followed
    by some r2 application reaches *target*."""
    spent = 0
    for p, _ in _positions(t0):
        t1 = apply_rewrite_at(r1, t0, p)
        if t1 is None:
            continue
        for q, _ in _positions(t1):
            spent += 1
            if spent > fuel:
                return False
            t2 = apply_rewrite_at(r2, t1, q)
            if t2 is not None and t2 == target:
                return True
    return False


def _fresh_leaves(n: int, shape: tuple = (4, 4)) -> list[Var]:
    from catopt.ir import TensorType

    return [Var(f"_synth_{i}", TensorType(shape)) for i in range(n)]


#: Leaf shapes tried when instantiating a candidate LHS for validation.
#: ``(4,4)`` satisfies shape-checked guards on rank-2 terms; ``()``
#: catches guards requiring scalar bindings.  A candidate whose guards
#: need a mixed/other profile is rejected — conservative, never unsound.
_LEAF_SHAPES: tuple = ((4, 4), ())

#: Values enumerated for attribute metavariables in a candidate LHS
#: (dims first — they dominate; a few shapes for view-style attrs).
_ATTR_POOL: tuple = (
    -1,
    -2,
    1,
    2,
    0,
    -3,
    3,
    4,
    -4,
    (4, 4),
    (4,),
    (2, 4, 4),
)

#: Cap on LHS instantiations tried per leaf-shape profile.
_MAX_INSTANTIATIONS: int = 400


def _instantiation_stream(term_mvars: list[str], attr_mvars: list[str]):
    """Yield candidate-LHS substitutions: fresh typed leaves (per
    :data:`_LEAF_SHAPES` profile) times a bounded product of
    :data:`_ATTR_POOL` values for attribute metavariables."""
    import itertools

    for shape in _LEAF_SHAPES:
        leaves = _fresh_leaves(len(term_mvars), shape)
        base = dict(zip(term_mvars, leaves, strict=True))
        for count, combo in enumerate(
            itertools.product(*([_ATTR_POOL] * len(attr_mvars)))
        ):
            subst = dict(base)
            for a, v in zip(attr_mvars, combo, strict=True):
                subst["$attr:" + a] = v
            yield subst
            if count + 1 >= _MAX_INSTANTIATIONS:
                break


def _leaf_vars(t: Any, out: set | None = None) -> set:
    """Var/Param leaves occurring anywhere inside a (sub)term."""
    if out is None:
        out = set()
    if isinstance(t, (Var, Param)):
        out.add(t)
    elif isinstance(t, Op):
        for a in t.args:
            _leaf_vars(a, out)
    return out


def _tensor_env(subst: dict) -> dict:
    """Random fp64 tensors for the typed leaves in a substitution.

    Bound values may be whole subterms (witness bindings mined from
    seeds), so Var/Param leaves are collected recursively — a scalar or
    tensor needed only deep inside a bound mask still gets an entry."""
    import torch

    env = {}
    for v in subst.values():
        for leaf in _leaf_vars(v):
            shape = getattr(getattr(leaf, "typ", None), "shape", None)
            if shape is not None and all(
                isinstance(d, int) for d in shape
            ):
                env[leaf] = torch.randn(*shape, dtype=torch.float64)
    return env


def _seed_witnesses(lhs: Any, seeds: Iterable[Any]):
    """Substitutions obtained by matching *lhs* against every subterm of
    the seed terms.

    These are the *real witnesses* a guarded candidate needs: concrete
    bindings on which the composite side conditions demonstrably can
    hold (Const-valued scales, broadcast-shaped masks, transpose dims
    tied to an actual rank) — assignments the bounded leaf/attr pool
    cannot produce."""
    for s in seeds or ():
        for _, sub in _positions(s):
            m = match_pattern(lhs, sub, {})
            if m is not None:
                yield m


def _validate_candidate(
    cand: Rewrite,
    r1: Rewrite,
    r2: Rewrite,
    numeric: bool,
    witness: dict | None = None,
    seeds: Iterable[Any] = (),
) -> bool:
    """Well-formedness + derivation replay + (optional) numeric check.

    *witness* (seed path) is the concrete binding the derivation
    actually fired on — instantiating the LHS with it is guaranteed to
    satisfy the composite guards.  Without a witness (symbolic path) a
    GUARDED candidate first tries seed-mined bindings — its LHS matched
    against every seed subterm — since tight side conditions are only
    satisfiable on real instances; the bounded instantiation pool then
    remains as a fallback.  A candidate whose guards are unsatisfiable
    on every tried instance is rejected rather than trusted."""
    lhs_mv = pattern_metavars(cand.lhs)
    rhs_mv = pattern_metavars(cand.rhs)
    lhs_terms = {v for v in lhs_mv if not v.startswith("$attr:")}
    # RHS term metavars must be LHS-bound; RHS attr metavars may also be
    # produced by the candidate's own ``derive``.
    for v in rhs_mv:
        if v.startswith("$attr:"):
            if v not in lhs_mv and cand.derive is None:
                return False
        elif v not in lhs_terms:
            return False
    term_mvars = sorted(lhs_terms)
    attr_mvars = sorted(
        v[len("$attr:") :] for v in lhs_mv if v.startswith("$attr:")
    )

    def attempt(subst: dict, fatal_replay: bool):
        """None: instance vetoed/skipped; False: reject; True: emit."""
        try:
            t0 = instantiate_pattern(cand.lhs, subst)
        except KeyError:
            return None
        # Fire the derived rule itself — this runs the composite
        # check/derive, so a guarded instantiation that vetoes simply
        # moves the enumeration to the next assignment.
        tout = apply_rewrite_at(cand, t0, ())
        if tout is None:
            return None
        if not _term_is_ground(tout):
            return None
        # The claimed derivation r1-then-r2 must actually replay —
        # parent guards are evaluated on this concrete instance too.  A
        # firing instance the derivation cannot reproduce is a genuine
        # counterexample (fatal); a seed-mined match may instead merely
        # be too large for the replay fuel, so it only skips ahead.
        if not _replays(cand, r1, r2, t0, tout):
            return False if fatal_replay else None
        if numeric:
            try:
                env = _tensor_env(subst)
                a = _eval_term(t0, env)
                b = _eval_term(tout, env)
            except Exception:
                return (
                    True  # ops without torch bindings: structural only
                )
            return _eval_allclose(a, b)
        return True

    if witness is not None:
        return attempt(dict(witness), fatal_replay=True) is True

    # Guarded candidates prefer real seed witnesses; a vetoed match is
    # just an instance the side conditions exclude, a verified match is
    # proof on a binding that actually arises.
    if seeds and (cand.check is not None or cand.derive is not None):
        for subst in _seed_witnesses(cand.lhs, seeds):
            res = attempt(subst, fatal_replay=False)
            if res is not None:
                return res
    for subst in _instantiation_stream(term_mvars, attr_mvars):
        res = attempt(subst, fatal_replay=True)
        if res is not None:
            return res
    return False


def _is_tautology(lhs: Any, rhs: Any) -> bool:
    lhs_key, rhs_key = _alpha_key(lhs, rhs)
    return lhs_key == rhs_key


def _subsumed(cand: Rewrite, existing: list[Rewrite]) -> bool:
    """True if *cand* is just an instance of an existing rule (its lhs
    matches some rule's lhs and the corresponding rhs instantiation
    alpha-equals cand.rhs)."""
    for e in existing:
        # A guarded rule cannot subsume: it only fires where its side
        # conditions pass, so *cand* may cover instances it cannot.
        if e.check is not None or e.derive is not None:
            continue
        subst = match_pattern(e.lhs, cand.lhs, {})
        if subst is None:
            continue
        try:
            inst = instantiate_pattern(e.rhs, subst)
        except KeyError:
            continue
        if (
            _alpha_key(inst, inst)[0]
            == _alpha_key(cand.rhs, cand.rhs)[0]
        ):
            return True
    return False


def synthesize_rules(
    rules: list[Rewrite],
    seed_terms: Iterable[Any] = (),
    *,
    fuel: int = 512,
    numeric_check: bool = True,
    require_overlap: bool = True,
    emit_subsumed: bool = False,
) -> list[Rewrite]:
    """Compose ordered pairs of rules into derived rewrite rules.

    Two synthesis paths (seed-guided runs FIRST — it is the preferred
    path for guarded pairs):

    * **Seed-guided** — apply ``r1`` then ``r2`` on concrete seed terms;
      the composite over the smallest region containing both rewrite
      sites is re-abstracted leaf-by-leaf into a new pattern, and the
      fired binding is carried into validation as the candidate's
      *witness*.  A guarded pair's side conditions (Const-valued
      scales, broadcast-shaped masks, rank-tied transpose dims) are in
      general only satisfiable on real instances, so a pair where
      either parent has ``check``/``derive`` needs this path — or a
      seed-mined witness — to emit.  This path also captures
      interactions that only appear in context (e.g. an ``apply`` node
      produced inside a larger ``add(matmul(...), x)``).
    * **Symbolic** — apply ``r1`` to its own LHS symbolically (its RHS
      with metavariable leaves) and match ``r2``'s LHS on the result.
      Produces fully general derived rules ``r1.lhs -> t2``.  Guarded
      symbolic candidates are validated against seed-mined witnesses
      first (the candidate LHS matched against every seed subterm),
      then against the bounded instantiation pool as fallback.

    Guarded parents compose too: the derived rule's ``check`` is the
    CONJUNCTION of both parents' checks, each evaluated on its own
    binding re-expressed through the intermediate substitution (see
    :func:`_compose_guards`), and ``derive``-produced attributes are
    carried as namespaced placeholders the derived rule refills at fire
    time.  A composition whose side condition cannot be re-expressed is
    rejected — never emitted unsound.

    ``seed_terms`` supplies the concrete terms the seed-guided path
    fires on (and that guarded symbolic candidates mine for witness
    bindings); pass real terms on which the guarded parents actually
    fire — e.g. a materialised attention term for the ``sdpa_fold``
    family.  ``fuel`` bounds the total number of rule-application
    attempts.  ``require_overlap`` keeps only pairs whose rewrite sites
    share an ancestor/descendant relation (true critical pairs —
    disjoint pairs are just parallel application).  ``numeric_check``
    evaluates each candidate on random fp64 tensors when its ops have
    torch bindings.

    Returns a list of :class:`Rewrite` objects named
    ``syn_<r1>__<r2>_<i>`` with provenance recorded in ``law``, on
    ``rule.parents``, and in :data:`SYNTH_PARENTS`.
    """
    usable = [r for r in rules if _synthesizable(r)]
    derived: list[Rewrite] = []
    seen: set[tuple[str, str]] = set()
    spent = 0
    counter = [0]
    ns1, ns2 = _DRV_PREFIX

    def offer(
        lhs: Any,
        rhs: Any,
        r1: Rewrite,
        r2: Rewrite,
        via: str,
        check=None,
        derive=None,
        witness: dict | None = None,
        pats: tuple[dict, dict] | None = None,
    ) -> None:
        if _is_tautology(lhs, rhs):
            return
        key = _alpha_key(lhs, rhs)
        if key in seen:
            return
        seen.add(key)
        # A composite derive is only attached when the RHS actually
        # carries placeholders it must fill (the check alone already
        # covers vetoing; extra derives would be dead weight).
        placeholders = _rhs_derive_placeholders(rhs)
        drv = derive if (derive is not None and placeholders) else None
        cand = Rewrite(
            name=f"syn_{r1.name}__{r2.name}_{counter[0]}",
            lhs=lhs,
            rhs=rhs,
            law=f"synthesized: {r1.name} then {r2.name} ({via})",
            check=check,
            derive=drv,
        )
        counter[0] += 1
        # Frozen dataclass: provenance rides on the instance dict plus
        # the module-level registry.
        object.__setattr__(cand, "parents", (r1.name, r2.name))
        SYNTH_PARENTS[cand.name] = (r1.name, r2.name)
        # The guard re-expression maps (pat1, pat2) are pure data — kept
        # on the rule so catopt.rulecache can serialize them and rebuild
        # the composite check/derive at load time via _compose_guards.
        if pats is not None:  # pragma: no cover — every call site passes pats
            object.__setattr__(cand, "guard_pats", pats)
        if not emit_subsumed and _subsumed(cand, usable + derived):
            return
        if not _validate_candidate(
            cand,
            r1,
            r2,
            numeric_check,
            witness=witness,
            seeds=seed_terms,
        ):
            return
        derived.append(cand)

    # -- seed-guided path: r1 then r2 on concrete terms -------------------
    # Seeds run FIRST: a guarded pair's side conditions (Const-valued
    # scales, broadcast-shaped masks, rank-tied dims) are only
    # satisfiable on real instances, and the fired binding rides into
    # validation as the candidate's witness.  The symbolic path below
    # stays as the fully-general fallback.
    for s in seed_terms:
        if spent > fuel:
            break
        for r1 in usable:
            for p, _ in _positions(s):
                if spent > fuel:
                    break
                f1 = _fire_guarded(r1, s, p, ns1)
                if f1 is None:
                    continue
                m1, t1, conc1 = f1
                spent += 1
                keep = _concrete_matched_leaves(r1.lhs, _subterm(s, p))
                for r2 in usable:
                    for q, sub2 in _positions(t1):
                        if spent > fuel:
                            break
                        if require_overlap and not _overlapping(p, q):
                            continue
                        m2 = match_pattern(r2.lhs, sub2, {})
                        if m2 is None:
                            continue
                        spent += 1
                        # r2's guards need the CONCRETE binding: resolve
                        # placeholder attrs inside the matched terms and
                        # in the "$attr:" entries via r1's derived values.
                        m2e = {}
                        bad = False
                        for k, v in m2.items():
                            if k.startswith("$attr:"):
                                if isinstance(v, str):
                                    if v in conc1:
                                        v = conc1[v]
                                    else:
                                        bad = True
                                        break
                                m2e[k] = v
                            else:
                                m2e[k] = _concretize_attrs(v, conc1)
                        if bad:
                            continue
                        try:
                            if r2.check is not None and not r2.check(
                                m2e
                            ):
                                continue
                        except Exception:
                            continue
                        inst2 = dict(m2)
                        if r2.derive is not None:
                            try:
                                e2 = r2.derive(m2e)
                            except Exception:
                                continue
                            if e2 is None:
                                continue
                            for k, v in e2.items():
                                if k.startswith("$attr:"):
                                    inst2[k] = ns2 + k[len("$attr:") :]
                                else:
                                    inst2[k] = v
                        try:
                            rhs2 = instantiate_pattern(r2.rhs, inst2)
                        except KeyError:  # pragma: no cover — _synthesizable guarantees a total subst
                            continue
                        t2 = _replace(t1, q, rhs2)
                        keep2 = _concrete_matched_leaves(r2.lhs, sub2)
                        lca = _common_prefix(p, q)
                        names: dict = {}
                        cnt = [0]
                        lhs_pat = _leaf_generalize(
                            _subterm(s, lca),
                            names,
                            keep | keep2,
                            cnt,
                            assign_fresh=True,
                        )
                        rhs_pat = _leaf_generalize(
                            _subterm(t2, lca),
                            names,
                            keep | keep2,
                            cnt,
                            assign_fresh=False,
                        )
                        # Re-express both fired bindings over the derived
                        # metavariables; an un-reexpressible binding
                        # rejects the pair outright.
                        try:
                            pat1 = _reexpress_map(
                                m1, names, keep | keep2
                            )
                            pat2 = _reexpress_map(
                                m2, names, keep | keep2
                            )
                        except _Unreexpressible:
                            continue
                        chk, drv = _compose_guards(r1, r2, pat1, pat2)
                        witness = {v: leaf for leaf, v in names.items()}
                        offer(
                            lhs_pat,
                            rhs_pat,
                            r1,
                            r2,
                            "seed",
                            check=chk,
                            derive=drv,
                            witness=witness,
                            pats=(pat1, pat2),
                        )

    # -- symbolic path: r1 applied to its own lhs -------------------------
    for r1 in usable:
        if spent > fuel:
            break
        # Identity instantiation: term metavars stay themselves, LHS
        # attr metavars keep their names, and RHS attr metavars that
        # only ``derive`` can produce become "@1:" placeholders.
        subst = {}
        for v in pattern_metavars(r1.lhs):
            subst[v] = (
                v[len("$attr:") :] if v.startswith("$attr:") else v
            )
        for v in pattern_metavars(r1.rhs):
            if v.startswith("$attr:") and v not in subst:
                subst[v] = ns1 + v[len("$attr:") :]
        try:
            t1 = instantiate_pattern(r1.rhs, subst)
        except KeyError:  # pragma: no cover — _synthesizable guarantees a total subst
            continue
        # r1's binding on the derived rule's own subst is the identity.
        pat1 = {
            v: (v[len("$attr:") :] if v.startswith("$attr:") else v)
            for v in pattern_metavars(r1.lhs)
        }
        for q, sub in _positions(t1):
            for r2 in usable:
                spent += 1
                if spent > fuel:
                    break
                m2 = match_pattern(r2.lhs, sub, {})
                if m2 is None:
                    continue
                # r2's RHS attr metavars not bound by its LHS need
                # r2.derive — carried as "@2:" placeholders.
                inst2 = dict(m2)
                ok = True
                for v in pattern_metavars(r2.rhs):
                    if v.startswith("$attr:") and v not in inst2:
                        if r2.derive is None:  # pragma: no cover — filtered by _synthesizable
                            ok = False
                            break
                        inst2[v] = ns2 + v[len("$attr:") :]  # pragma: no cover — r2.derive + unbound attr metavar

                if not ok:  # pragma: no cover — filtered by _synthesizable
                    continue
                try:
                    rhs2 = instantiate_pattern(r2.rhs, inst2)
                except KeyError:  # pragma: no cover — total subst guaranteed
                    continue
                t2 = _replace(t1, q, rhs2)
                chk, drv = _compose_guards(r1, r2, pat1, m2)
                offer(
                    r1.lhs,
                    t2,
                    r1,
                    r2,
                    "symbolic",
                    check=chk,
                    derive=drv,
                    pats=(pat1, m2),
                )

    return derived
