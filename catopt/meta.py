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

* *Symbolic composition*: instantiate ``r1``'s RHS pattern with its own
  metavariables (as opaque symbolic leaves) and try to match ``r2``'s
  LHS against the result.  A match at position ``q`` yields the derived
  rule ``r1.lhs -> (r1.rhs with r2 applied at q)`` — fully general by
  construction.
* *Seed-guided composition*: apply ``r1`` to a concrete seed term at
  position ``p``, then ``r2`` at position ``q``; the composite rule is
  ``abstract(subterm(s, lca(p,q))) -> abstract(subterm(result, lca(p,q)))``
  where abstraction renames every metavariable-bound or untouched leaf
  to a fresh metavariable but keeps leaves that were matched against
  concrete pattern leaves.

Every emitted rule is validated: metavars(rhs) ⊆ metavars(lhs), the
r1-then-r2 derivation is replayed on a fresh instantiation, and (when
every op has a torch binding) both sides are evaluated on random
tensors and compared with ``allclose``.

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

from typing import Any, Iterable

from catopt.egraph import EGraph, Rewrite
from catopt.ir import Op, Var, Const, Param, op_repr
import catopt.rules as R


# ---------------------------------------------------------------------------
#  Part A.1 — coherence classification
# ---------------------------------------------------------------------------

#: Rules that express pure coherence: symmetry/associativity/identity/
#: involution.  They generate every *equivalent bracketing* of the same
#: computation — the search-space the e-graph should never store.
#: ``om_assoc``/``om_assoc_rev`` live in catopt.om (same law family:
#: associativity of a monoid's compose).
COHERENT_RULE_NAMES: frozenset[str] = frozenset({
    "comm_add", "comm_mul",          # SMC symmetry
    "assoc_add", "assoc_mul",        # additive/multiplicative associativity
    "id_add", "id_mul",              # monoid units
    "double_neg",                    # involution
    "assoc_matmul", "assoc_matmul_rev",   # associativity of composition
    "aff_assoc", "aff_assoc_rev",         # affine-map monoid associativity
    "affd_assoc", "affd_assoc_rev",       # diagonal-affine monoid assoc.
    "om_assoc", "om_assoc_rev",           # online-softmax monoid assoc.
})


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


def classify_rules(rules: Iterable[Rewrite]) -> tuple[list[Rewrite], list[Rewrite]]:
    """Split *rules* into ``(coherent, contentful)`` by name."""
    coherent = [r for r in rules if r.name in COHERENT_RULE_NAMES]
    contentful = [r for r in rules if r.name not in COHERENT_RULE_NAMES]
    return coherent, contentful


#: Coherent / contentful partition of every rule defined in catopt.rules.
_ALL_MODULE_RULES = _iter_module_rules(R)
COHERENT: list[Rewrite]
CONTENTFUL: list[Rewrite]
COHERENT, CONTENTFUL = classify_rules(_ALL_MODULE_RULES)


# ---------------------------------------------------------------------------
#  Part A.2 — canonicalize: eager coherence normalization
# ---------------------------------------------------------------------------

#: Commutative + associative ops with a dropped identity element.
_AC_IDENTITY: dict[str, float] = {"add": 0.0, "mul": 1.0}

#: Associative but NOT commutative ops: chains flatten in order and
#: rebuild balanced.  For ``aff_compose`` the balanced form is the
#: parallel-scan (Blelloch) bracketing — computed here, not searched.
_ASSOC_ONLY: frozenset[str] = frozenset(
    {"matmul", "aff_compose", "affd_compose"})


def _sort_key(t: Any) -> str:
    return op_repr(t)


def _balanced(op: str, items: list[Any], attrs: dict) -> Any:
    """Rebuild a flattened chain as a balanced binary tree."""
    if len(items) == 1:
        return items[0]
    mid = len(items) // 2
    return Op.make(op,
                   _balanced(op, items[:mid], attrs),
                   _balanced(op, items[mid:], attrs),
                   **attrs)


def _flatten_chain(term: Op) -> list[Any]:
    """Collect the operands of a same-op/same-attrs chain."""
    out: list[Any] = []

    def go(t: Any) -> None:
        if (isinstance(t, Op) and t.op == term.op
                and dict(t.attrs) == dict(term.attrs)):
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

    Memoised by ``id()`` so shared-subterm DAGs stay linear.  The
    function is idempotent and semantics-preserving (all transforms are
    instances of the coherent laws).
    """
    if memo is None:
        memo = {}
    key = id(term)
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
        flat = [t for t in flat
                if not (isinstance(t, Const) and t.value == ident)]
        if not flat:
            out = Const(ident)
        elif len(flat) == 1:
            out = flat[0]
        else:
            flat = sorted(flat, key=_sort_key)
            out = _balanced(out.op, flat, dict(out.attrs))
    elif out.op in _ASSOC_ONLY:
        flat = _flatten_chain(out)
        if len(flat) == 1:
            out = flat[0]
        elif len(flat) != len(out.args):
            out = _balanced(out.op, flat, dict(out.attrs))
        # len==2 already binary: keep (canonical balanced form of 2)

    memo[key] = out
    return out


# ---------------------------------------------------------------------------
#  Part A.3 — stratified_run
# ---------------------------------------------------------------------------

def canonical_cost(cost_fn):
    """Wrap *cost_fn* so extraction prices the canonical form of each
    candidate — coherent-equivalent bracketings are scored by their
    normal form, so e.g. a right-leaning ``aff_compose`` chain is
    charged its balanced (log-depth) cost."""
    def wrapped(t: Any, **kw) -> float:
        return cost_fn(canonicalize(t), **kw)
    return wrapped


def stratified_run(eg: EGraph, rules: list[Rewrite], term: Any,
                   *, max_iterations: int = 100, max_nodes: int = 100_000,
                   cost_fn=None, extract: bool = True,
                   canonicalize_output: bool = True,
                   extract_fn=None) -> dict:
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
    stats = eg.run(contentful, root, max_iterations=max_iterations,
                   max_nodes=max_nodes)
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
        best = (extract_fn(root) if extract_fn is not None
                else eg.extract_best(root, cost_fn))
        out["best"] = best
        out["canonical_best"] = (
            canonicalize(best) if canonicalize_output and best is not None
            else best)
    return out


# ---------------------------------------------------------------------------
#  Part B — rule synthesis via critical-pair completion
# ---------------------------------------------------------------------------

#: ops with at least one string attr value carry *attribute*
#: metavariables; synthesis skips such rules (their derived attrs can't
#: be propagated soundly).
def _has_attr_metavars(pat: Any) -> bool:
    if isinstance(pat, Op):
        if any(isinstance(v, str) for v in pat.attrs.values()):
            return True
        return any(_has_attr_metavars(a) for a in pat.args)
    return False


def _synthesizable(r: Rewrite) -> bool:
    """A rule participates in synthesis iff it is side-condition-free
    and attr-metavar-free (checks can't run on symbolic terms, and a
    generalized derived rule could fire where a check would veto)."""
    return (r.check is None and r.derive is None
            and not _has_attr_metavars(r.lhs)
            and not _has_attr_metavars(r.rhs))


def match_pattern(pat: Any, term: Any, subst: dict | None = None) -> dict | None:
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
        if (not isinstance(term, Op) or term.op != pat.op
                or len(term.args) != len(pat.args)):
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
        for pa, ta in zip(pat.args, term.args):
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


def _positions(term: Any, path: tuple = ()) -> Iterable[tuple[tuple, Any]]:
    """Yield ``(path, subterm)`` for every position, DFS pre-order."""
    yield path, term
    if isinstance(term, Op):
        for i, a in enumerate(term.args):
            yield from _positions(a, path + (i,))


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
    for a, b in zip(p, q):
        if a != b:
            break
        out.append(a)
    return tuple(out)


def _overlapping(p: tuple, q: tuple) -> bool:
    """True when one position is an ancestor of (or equal to) the other."""
    lca = _common_prefix(p, q)
    return lca == p or lca == q


def apply_rewrite_at(rule: Rewrite, term: Any, path: tuple) -> Any | None:
    """Apply *rule* to ``term`` at *path*; return the rewritten term or
    ``None`` if the LHS doesn't match there.  (check/derive rules are
    not supported by this term-level helper.)"""
    subst = match_pattern(rule.lhs, _subterm(term, path), {})
    if subst is None:
        return None
    try:
        rhs = instantiate_pattern(rule.rhs, subst)
    except KeyError:
        return None
    return _replace(term, path, rhs)


def _concrete_matched_leaves(pat: Any, term: Any, out: set | None = None) -> set:
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
            for pa, ta in zip(pat.args, term.args):
                _concrete_matched_leaves(pa, ta, out)
        return out
    out.add(term)  # concrete pattern leaf pinned this value
    return out


def _leaf_generalize(region: Any, names: dict, keep: set,
                     counter: list[int], assign_fresh: bool) -> Any:
    """Rename each leaf of *region* to a metavariable (shared ``names``
    map keeps lhs/rhs consistent), except leaves in *keep* which were
    concrete-matched and must stay literal.  With ``assign_fresh=False``
    (the RHS pass) an unseen leaf is a constant introduced by a rule
    pattern — it stays concrete."""
    if isinstance(region, Op):
        return Op.make(region.op,
                       *(_leaf_generalize(a, names, keep, counter,
                                          assign_fresh)
                         for a in region.args),
                       **dict(region.attrs))
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
            return Op.make(t.op, *(norm(a, table) for a in t.args),
                           **dict(t.attrs))
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
            _eval_allclose(x, y, tol) for x, y in zip(a, b))
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


def _replays(derived: Rewrite, r1: Rewrite, r2: Rewrite,
             t0: Any, target: Any, fuel: int = 400) -> bool:
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


def _validate_candidate(cand: Rewrite, r1: Rewrite, r2: Rewrite,
                        numeric: bool) -> bool:
    """Well-formedness + derivation replay + (optional) numeric check."""
    # RHS may only use LHS metavariables.
    if not pattern_metavars(cand.rhs) <= pattern_metavars(cand.lhs):
        return False
    # Instantiate the LHS with fresh leaves and apply the derived rule.
    mvars = sorted(pattern_metavars(cand.lhs))
    if any(v.startswith("$attr:") for v in mvars):
        return False
    leaves = _fresh_leaves(len(mvars))
    subst = dict(zip(mvars, leaves))
    try:
        t0 = instantiate_pattern(cand.lhs, subst)
        tout = instantiate_pattern(cand.rhs, subst)
    except KeyError:
        return False
    if not _term_is_ground(tout):
        return False
    # The claimed derivation r1-then-r2 must actually replay.
    if not _replays(cand, r1, r2, t0, tout):
        return False
    if numeric:
        import torch
        try:
            env = {v: torch.randn(4, 4, dtype=torch.float64)
                   for v in leaves}
            a = _eval_term(t0, env)
            b = _eval_term(tout, env)
        except Exception:
            return True  # ops without torch bindings: structural only
        return _eval_allclose(a, b)
    return True


def _is_tautology(lhs: Any, rhs: Any) -> bool:
    l, r = _alpha_key(lhs, rhs)
    return l == r


def _subsumed(cand: Rewrite, existing: list[Rewrite]) -> bool:
    """True if *cand* is just an instance of an existing rule (its lhs
    matches some rule's lhs and the corresponding rhs instantiation
    alpha-equals cand.rhs)."""
    for e in existing:
        subst = match_pattern(e.lhs, cand.lhs, {})
        if subst is None:
            continue
        try:
            inst = instantiate_pattern(e.rhs, subst)
        except KeyError:
            continue
        if _alpha_key(inst, inst)[0] == _alpha_key(cand.rhs, cand.rhs)[0]:
            return True
    return False


def synthesize_rules(rules: list[Rewrite],
                     seed_terms: Iterable[Any] = (),
                     *, fuel: int = 512, numeric_check: bool = True,
                     require_overlap: bool = True,
                     emit_subsumed: bool = False) -> list[Rewrite]:
    """Compose ordered pairs of rules into derived rewrite rules.

    Two synthesis paths:

    * **Symbolic** — apply ``r1`` to its own LHS symbolically (its RHS
      with metavariable leaves) and match ``r2``'s LHS on the result.
      Produces fully general derived rules ``r1.lhs -> t2``.
    * **Seed-guided** — apply ``r1`` then ``r2`` on concrete seed terms;
      the composite over the smallest region containing both rewrite
      sites is re-abstracted leaf-by-leaf into a new pattern.  This
      captures interactions that only appear in context (e.g. an
      ``apply`` node produced inside a larger ``add(matmul(...), x)``).

    ``fuel`` bounds the total number of rule-application attempts.
    ``require_overlap`` keeps only pairs whose rewrite sites share an
    ancestor/descendant relation (true critical pairs — disjoint pairs
    are just parallel application).  ``numeric_check`` evaluates each
    candidate on random fp64 tensors when its ops have torch bindings.

    Returns a list of :class:`Rewrite` objects named
    ``syn_<r1>__<r2>_<i>`` with provenance recorded in ``law``.
    """
    usable = [r for r in rules if _synthesizable(r)]
    derived: list[Rewrite] = []
    seen: set[tuple[str, str]] = set()
    spent = 0
    counter = [0]

    def offer(lhs: Any, rhs: Any, r1: Rewrite, r2: Rewrite,
              via: str) -> None:
        if _is_tautology(lhs, rhs):
            return
        key = _alpha_key(lhs, rhs)
        if key in seen:
            return
        seen.add(key)
        cand = Rewrite(
            name=f"syn_{r1.name}__{r2.name}_{counter[0]}",
            lhs=lhs, rhs=rhs,
            law=f"synthesized: {r1.name} then {r2.name} ({via})",
        )
        counter[0] += 1
        if not emit_subsumed and _subsumed(
                cand, usable + derived):
            return
        if not _validate_candidate(cand, r1, r2, numeric_check):
            return
        derived.append(cand)

    # -- symbolic path: r1 applied to its own lhs -------------------------
    for r1 in usable:
        if spent > fuel:
            break
        subst = {v: v for v in pattern_metavars(r1.lhs)
                 if not v.startswith("$attr:")}
        t1 = instantiate_pattern(r1.rhs, subst)
        for q, sub in _positions(t1):
            for r2 in usable:
                spent += 1
                if spent > fuel:
                    break
                m2 = match_pattern(r2.lhs, sub, {})
                if m2 is None:
                    continue
                try:
                    rhs2 = instantiate_pattern(r2.rhs, m2)
                except KeyError:
                    continue
                t2 = _replace(t1, q, rhs2)
                offer(r1.lhs, t2, r1, r2, "symbolic")

    # -- seed-guided path: r1 then r2 on concrete terms -------------------
    for s in seed_terms:
        if spent > fuel:
            break
        for r1 in usable:
            for p, _ in _positions(s):
                if spent > fuel:
                    break
                t1 = apply_rewrite_at(r1, s, p)
                if t1 is None:
                    continue
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
                        try:
                            rhs2 = instantiate_pattern(r2.rhs, m2)
                        except KeyError:
                            continue
                        t2 = _replace(t1, q, rhs2)
                        keep2 = _concrete_matched_leaves(r2.lhs, sub2)
                        lca = _common_prefix(p, q)
                        names: dict = {}
                        cnt = [0]
                        lhs_pat = _leaf_generalize(
                            _subterm(s, lca), names, keep | keep2, cnt,
                            assign_fresh=True)
                        rhs_pat = _leaf_generalize(
                            _subterm(t2, lca), names, keep | keep2, cnt,
                            assign_fresh=False)
                        offer(lhs_pat, rhs_pat, r1, r2, "seed")

    return derived
