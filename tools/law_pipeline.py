"""Law-discovery pipeline — census -> propose -> verify -> measure -> rank.

The loop that produced ``select_mul`` (retro
``project/retros/law-shape-aware.md``) was a sequence of one-off
tools with a human deciding which experiment to run at each step:
the census (``tools/law_shape_census.py``), the shape-aware proposer
(``tools/law_shape_proposal.py``), the two oracles
(``tools/law_verifier.py`` derivability, ``tools/law_proposal.py``
numeric truth) and the impact measurement
(``tools/law_impact.py``).  Each did its job; nothing ran them end
to end and decided.

This tool is that end-to-end runnable: **one entry point** that

1. **census** — mines the shape frequencies of the real corpus
   (``law_shape_census.run_census``).
2. **propose** — collects candidate equalities from the proposers:
   the census generators (same-view and mixed-view naturality over the
   frequent op-tuples, plus composed-then-reduced pattern
   recognition), ``law_shape_proposal.schemas`` shape-aware schemas
   and ``law_proposal.schema_candidates`` algebraic grammar — unified
   to pattern rules and de-duplicated by alpha-normal key.
3. **verify** — runs *both* oracles on every candidate's concrete
   instance: the numeric-truth oracle (``law_proposal._numeric_true``)
   and the derivability oracle (``law_verifier.verify_law``), the
   latter against the *search rule set* (the library, minus any
   ``--holdout`` names).
4. **measure** — fires the rule alone over every real model
   (``law_impact._probe``: fires, cost delta, lowered-module
   ``sink.verify``) and, for any candidate that fires, saturates each
   model under the search rule set with and without the rule
   (``law_impact._saturate`` / ``_cert_ok``): end-to-end cost delta,
   certificate replay, and a **closure-safety** check — the enode
   ratio ``out/in`` (a rule that explodes the closure is a search
   hazard, not a win).
5. **rank** — a single deterministic ordering over the evidence and a
   **ship / no-ship** verdict per candidate, with the reason.

Held-out rediscovery
--------------------

``select_mul`` is now shipped (``catopt_core.laws.tensor.SELECT_MUL``,
in ``SIMPLIFICATION_RULES``).  To validate the pipeline, remove it —
and only it — from the search rule set with ``--holdout select_mul``,
re-run, and locate the proposal whose equality matches the held-out
law in the ranking.  If the pipeline cannot rank a known winner at the
top, it is not trustworthy; the report says so plainly.

Reuse, not reinvention: every stage calls the shipped tool that
already implements it.  The only new logic here is the composition,
the closure-safety ratio, and the ranking.

Run::

    .venv/bin/python tools/law_pipeline.py
    .venv/bin/python tools/law_pipeline.py --holdout select_mul
    .venv/bin/python tools/law_pipeline.py --json /tmp/pipeline.json

CPU-only, bounded to a few minutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from catopt_core.cost import dag_cost
from catopt_core.egraph import Rewrite
from catopt_core.egraph.terms import _term_instantiate, _term_match
from catopt_core.ir import Const, Op, op_repr
from catopt_core.laws import ALL_RULES

# Sibling tools own every stage; reuse them, never duplicate.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import law_proposal as lp
from law_impact import (
    TermCase,
    _bench_cases,
    _cert_ok,
    _cost_fn,
    _iter_subterms,
    _probe,
    _saturate,
    model_cases,
)
from law_shape_census import _op_of, run_census
from law_shape_proposal import (
    Schema,
    _sink,
    real_matches,
    relaxed_matches,
    schemas,
)
from law_verifier import verify_law

__all__ = [
    "Evidence",
    "Proposal",
    "main",
    "propose",
    "run_pipeline",
]

#: Default number of ranked rows printed.
_DEFAULT_TOP = 20

#: Enode ``out/in`` ratio above which a candidate is closure-unsafe:
#: a rule that grows the e-graph more than this is a search hazard.
_CLOSURE_LIMIT = 2.0

#: Census rows to keep — large enough to hold every distinct op-tuple
#: (the corpus has ~124), so the frequency lookup is exact.
_CENSUS_TOP = 400


# ---------------------------------------------------------------------------
#  Proposal — one candidate equality, unified across the proposers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Proposal:
    """One candidate law: a pattern equality, plus an optional instance.

    ``lhs`` / ``rhs`` are *patterns* (bare ``str`` leaves are
    metavariables), so a single rule fires on any instantiation and
    the census matchers apply uniformly.  ``instance`` is a concrete
    ``(lhs, rhs)`` pair for the numeric / derivability oracles when the
    proposer already supplies one (the algebraic grammar); the
    shape-aware schemas supply patterns only and are instantiated from
    a real match instead.

    ``check`` / ``derive`` are the ``Rewrite`` side-condition hooks,
    for candidates whose RHS needs an attribute the LHS does not carry
    verbatim — e.g. the softmax fold, whose ``dim`` is a single int
    derived from ``sum``'s ``dim`` tuple.  They pass through to
    :meth:`as_rule`, so firing, certificate replay and the oracles all
    see the same side condition.
    """

    name: str
    lhs: Any
    rhs: Any
    family: str
    sources: tuple[str, ...] = ()
    instance: tuple[Any, Any] | None = None
    check: Any = None
    derive: Any = None

    def as_rule(self) -> Rewrite:
        """Return the proposal as a fireable pattern ``Rewrite``."""
        return Rewrite(
            self.name,
            self.lhs,
            self.rhs,
            check=self.check,
            derive=self.derive,
        )


def _to_pattern(term: Any, mv: dict[str, str]) -> Any:
    """Abstract the concrete leaves of *term* to shared metavariables.

    A leaf (``Var`` / ``Param``) keyed by ``repr`` becomes ``M0``,
    ``M1``, … under the shared *mv* map, so two occurrences of the
    same leaf stay the same metavariable and a repeated-slot equality
    survives abstraction.  ``Const`` leaves stay literal.
    """
    if isinstance(term, Op):
        return Op.make(
            term.op,
            *(_to_pattern(a, mv) for a in term.args),
            **dict(term.attrs),
        )
    if isinstance(term, Const):
        return term
    key = repr(term)
    if key not in mv:
        mv[key] = f"M{len(mv)}"
    return mv[key]


#: Pointwise ops a view-naturality law may push through a view op.
_POINTWISE = ("add", "sub", "mul", "div")

#: View / re-layout ops (the census's frequent inner ops).
_VIEW_OPS = (
    "select",
    "slice",
    "reshape",
    "transpose",
    "unsqueeze",
    "squeeze",
    "expand",
)


def _shape_aware() -> list[Proposal]:
    """Return the shape-aware grammar's schemas as proposals."""
    return [
        Proposal(
            name=s.name,
            lhs=s.lhs,
            rhs=s.rhs,
            family=s.family,
            sources=("shape-aware",),
        )
        for s in schemas()
    ]


def _grammar() -> list[Proposal]:
    """Return the algebraic grammar's candidates as pattern proposals."""
    out: list[Proposal] = []
    for c in lp.schema_candidates():
        mv: dict[str, str] = {}
        out.append(
            Proposal(
                name=f"grammar:{c.label}",
                lhs=_to_pattern(c.lhs, mv),
                rhs=_to_pattern(c.rhs, mv),
                family="algebraic-grammar",
                sources=("algebraic-grammar",),
                instance=(c.lhs, c.rhs),
            )
        )
    return out


def _view_attrs(terms: list[Any], view_op: str) -> list[str]:
    """Return the sorted attr keys of every *view_op* node in *terms*."""
    keys: set[str] = set()
    for term in terms:
        for sub in _iter_subterms(term):
            if isinstance(sub, Op) and sub.op == view_op:
                keys |= set(sub.attrs)
    return sorted(keys)


def _view_node(
    view_op: str, leaf: Any, keys: list[str], tag: str = "V"
) -> Any:
    """Return a view-op pattern node with *keys* as attr metavariables.

    The attribute values are shared metavariable names prefixed by
    *tag*, so the two view nodes of a naturality law must carry the
    same attrs — the matcher enforces the precondition structurally.
    Distinct tags let a mixed-view pattern give each view node its own
    attr metavariables (no cross-op attr equality is implied).
    """
    return Op.make(view_op, leaf, **{k: f"{tag}_{k}" for k in keys})


def _vocab_sets(vocab: str) -> tuple[tuple, tuple]:
    """Return ``(pointwise, views)`` for the requested vocabulary.

    ``"hand"`` is the two hand-written tuples below; ``"derived"``
    classifies every corpus op by property (``tools/law_vocab.py``),
    so the generator's op alphabet is machine-derived too.
    """
    if vocab == "derived":
        import law_vocab

        v = law_vocab.derive_vocabulary()
        return v.pointwise, v.views
    return _POINTWISE, _VIEW_OPS


def _census_naturality(
    census_op: dict,
    terms: list[Any],
    pointwise: tuple = _POINTWISE,
    views: tuple = _VIEW_OPS,
) -> list[Proposal]:
    """Mechanically derive view-naturality candidates from the census.

    For every frequent op-tuple ``f(g, g)`` — a pointwise *f* whose two
    operands are the *same* view op *g* — propose the naturality

        f(g(u, A), g(v, A)) -> g(f(u, v), A)

    where *A* are *g*'s own attribute keys, shared across both nodes.
    This is the census -> propose step done by machine: the shape
    comes from ``law_shape_census``'s op-tuple counts, not a hand-
    written schema list.  The truth oracle (not this generator)
    decides whether each candidate is sound.  The op alphabet is the
    ``pointwise`` / ``views`` pair — hand-written by default, or
    property-derived (``--vocab derived``).
    """
    out: list[Proposal] = []
    for op, kids in census_op:
        if op not in pointwise or len(kids) != 2:
            continue
        if kids[0] != kids[1] or kids[0] not in views:
            continue
        g = kids[0]
        keys = _view_attrs(terms, g)
        lhs = Op.make(
            op,
            _view_node(g, "U", keys),
            _view_node(g, "V", keys),
        )
        rhs = _view_node(g, Op.make(op, "U", "V"), keys)
        out.append(
            Proposal(
                name=f"census:{op}_{g}",
                lhs=lhs,
                rhs=rhs,
                family="census-naturality",
                sources=("census-naturality",),
            )
        )
    return out


def _two_view_family(
    op: str, g: str, h: str, terms: list[Any]
) -> list[Proposal]:
    """Emit the RHS family for a two-view tuple ``f(g(u,A), h(v,B))``.

    The plausible wrap targets are the left view, the right view, and
    no view at all (the ``id``/strip variant) — three shallow guesses,
    each letting the oracles be the referee rather than an algebraic
    theory of view inversion.  Each view keeps its own attr
    metavariables (``A_*`` / ``B_*``), so the RHS re-wraps with the
    bound attrs of whichever side it picks.
    """
    kg, kh = _view_attrs(terms, g), _view_attrs(terms, h)
    lhs = Op.make(
        op,
        _view_node(g, "U", kg, tag="A"),
        _view_node(h, "V", kh, tag="B"),
    )
    inner = Op.make(op, "U", "V")
    variants = {
        "wl": _view_node(g, inner, kg, tag="A"),
        "wr": _view_node(h, inner, kh, tag="B"),
        "id": inner,
    }
    return [
        Proposal(
            name=f"mixed:{op}_{g}_{h}_{tag}",
            lhs=lhs,
            rhs=rhs,
            family="census-mixed-view",
            sources=("census-mixed-view",),
        )
        for tag, rhs in variants.items()
    ]


def _one_view_family(
    op: str, view: str, view_left: bool, terms: list[Any]
) -> list[Proposal]:
    """Emit the RHS family for ``f(view(u,A), v)`` with ``v`` opaque.

    The non-view operand stays a bare metavariable — the census tuple
    (``mul(slice, ·)``, ``mul(select, add)``, ``mul(unsqueeze,
    stack)``) only fixes that a view feeds one side of a pointwise
    op, so every asymmetric tuple collapses onto this one pattern and
    de-dup merges the provenance.  The two plausible equalities are
    the push-through ``w(f(u, v))`` and the strip ``f(u, v)``; truth
    is per-instance and the oracles decide.
    """
    keys = _view_attrs(terms, view)
    node = _view_node(view, "U", keys, tag="A")
    lhs = (
        Op.make(op, node, "V") if view_left else Op.make(op, "V", node)
    )
    side = "l" if view_left else "r"
    inner = Op.make(op, "U", "V")
    variants = {
        "w": _view_node(view, inner, keys, tag="A"),
        "id": inner,
    }
    return [
        Proposal(
            name=f"mixed:{op}_{view}_{side}_{tag}",
            lhs=lhs,
            rhs=rhs,
            family="census-mixed-view",
            sources=("census-mixed-view",),
        )
        for tag, rhs in variants.items()
    ]


def _census_mixed_naturality(
    census_op: dict,
    terms: list[Any],
    pointwise: tuple = _POINTWISE,
    views: tuple = _VIEW_OPS,
) -> list[Proposal]:
    """Generalize census naturality to *mixed* operand views.

    :func:`_census_naturality` only reaches ``f(g, g)`` — a pointwise
    op over two copies of the SAME view.  The expanded corpus's more
    frequent shape is asymmetric: ``mul(slice, ·)``, ``mul(select,
    add)``, ``add(matmul, select)``, ``mul(unsqueeze, stack)`` — the
    retro's MoE-dispatch lead.  For every census tuple ``f(g, h)``
    with distinct children where at least one is a view this emits a
    small family of ``f(g(u, A), h(v, B)) -> w(f(u, v))`` candidates
    (``w`` over the views present, plus the identity).  The numeric
    oracle and the measurement gates — not this generator — decide
    truth and worth; a false or worthless variant is an honest
    rejection, not a generator bug.
    """
    out: list[Proposal] = []
    for op, kids in census_op:
        if op not in pointwise or len(kids) != 2:
            continue
        g, h = kids
        if g == h:
            continue  # same-view tuples are census-naturality's job
        gl, hl = g in views, h in views
        if gl and hl:
            out.extend(_two_view_family(op, g, h, terms))
        elif gl or hl:
            out.extend(_one_view_family(op, g if gl else h, gl, terms))
    return out


def _reduce_chains(census_op: dict) -> list[tuple[str, str, str]]:
    """Return ``(f, u, r)`` triples: ``f`` combines ``u(·)`` with ``r(u(·))``.

    The census signature of a composed-then-reduced chain — a manual
    normalization fold: a unary ``u`` feeding a reduction ``r`` (the
    ``(r, (u,))`` tuple) AND a binary ``f`` combining ``u(·)`` with an
    ``r`` node (``(f, (u, r))`` or ``(f, (r, u))``) both occur.  The
    corpus's instance is ``div(exp(·), sum(exp(·)))`` — a softmax
    spelled by hand.
    """
    out: list[tuple[str, str, str]] = []
    for op, kids in census_op:
        if len(kids) != 2:
            continue
        for i in (0, 1):
            u, r = kids[i], kids[1 - i]
            if u in ("·", "const") or r in ("·", "const"):
                continue
            if (r, (u,)) in census_op:
                out.append((op, u, r))
    return out


def _check_sum_keepdim(bound: dict) -> bool:
    """Guard the softmax fold: keepdim and a single reduce axis.

    ``x / sum(x, dim)`` only broadcasts to a softmax when the sum
    keeps its dim (a dropped dim broadcasts wrongly — or not at all —
    against the numerator), and a multi-axis sum has no single-dim
    ``softmax`` image.
    """
    dims = bound.get("$attr:RD")
    if bound.get("$attr:RK") is not True:
        return False
    return isinstance(dims, int) or (
        isinstance(dims, tuple) and len(dims) == 1
    )


def _derive_softmax_dim(bound: dict) -> dict:
    """Unwrap ``sum``'s ``dim`` tuple into ``softmax``'s scalar dim."""
    dims = bound.get("$attr:RD")
    return {"$attr:SD": dims[0] if isinstance(dims, tuple) else dims}


def _softmax_fold() -> Proposal:
    """Return the ``exp/sum`` spelling of softmax — a true shape fold.

    ``div(exp(u), sum(exp(u), dim, keepdim)) -> softmax(u, dim)`` is
    the kernel-recognition candidate ``ManualSoftmaxAttention`` puts
    in the corpus.  ``check``/``derive`` carry the side condition
    (keepdim, single axis) and the attr translation (the sum's
    ``dim`` tuple to softmax's int) — the oracles see them through
    :meth:`Proposal.as_rule` exactly as firing does.
    """
    e = Op.make("exp", "U")
    return Proposal(
        name="recognize:softmax",
        lhs=Op.make(
            "div", e, Op.make("sum", e, dim="RD", keepdim="RK")
        ),
        rhs=Op.make("softmax", "U", dim="SD"),
        family="pattern-recognition",
        sources=("pattern-recognition",),
        check=_check_sum_keepdim,
        derive=_derive_softmax_dim,
    )


#: Composed-then-reduced chains with a known kernel image, keyed by
#: ``(f, u, r)``.  Only the softmax fold is recognized today; a chain
#: with no recognizer stays a census fact, never a guessed equality.
_RECOGNIZERS: dict[tuple[str, str, str], Any] = {
    ("div", "exp", "sum"): _softmax_fold,
}


def _pattern_recognition(census_op: dict) -> list[Proposal]:
    """Emit candidates for recognized composed-then-reduced chains.

    This is the shape-proposal source beyond view naturality: the
    census is scanned for ``f(u(·), r(u(·)))`` chains and each chain
    with a registered recognizer yields one candidate.  Chains with no
    recognizer are evidence of a missing fold, silently skipped — the
    candidate pool stays honest guesses only.
    """
    out: list[Proposal] = []
    seen: set[tuple[str, str, str]] = set()
    for chain in _reduce_chains(census_op):
        if chain in seen:
            continue
        seen.add(chain)
        build = _RECOGNIZERS.get(chain)
        if build is not None:
            out.append(build())
    return out


def propose(
    census_op: dict, terms: list[Any], vocab: str = "hand"
) -> list[Proposal]:
    """Collect, unify and de-duplicate every proposer's candidates.

    Five sources feed the pool: the **census** generators (same-view
    naturality and the mixed-view family, both over the frequent
    op-tuples and the requested op *vocab*), **pattern recognition**
    (composed-then-reduced chains like the ``exp/sum`` softmax fold),
    the shape-aware schemas (``law_shape_proposal.schemas``), and the
    algebraic grammar (``law_proposal.schema_candidates``, abstracted
    to patterns with its concrete instance retained for the oracles).
    De-dup is by ``law_proposal._key`` — the alpha-normal equality —
    and a duplicate's provenance is merged into ``sources``, so a
    candidate reachable from the census generator is recorded as such.
    """
    pointwise, views = _vocab_sets(vocab)
    by_key: dict = {}
    pool = [
        *_census_naturality(census_op, terms, pointwise, views),
        *_census_mixed_naturality(census_op, terms, pointwise, views),
        *_pattern_recognition(census_op),
        *_shape_aware(),
        *_grammar(),
    ]
    for p in pool:
        key = lp._key(p.lhs, p.rhs)
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = p
        else:
            by_key[key] = replace(
                cur,
                sources=tuple(dict.fromkeys(cur.sources + p.sources)),
            )
    return list(by_key.values())


# ---------------------------------------------------------------------------
#  Evidence — the measured record the ranking reads
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """One proposal plus every measured verdict.

    The properties ``truth`` and ``shippable`` are the verdict; every
    field above them is raw evidence, so the report can show *why*.
    """

    proposal: Proposal
    census_sites: int = 0
    relaxed: int = 0
    matches: int = 0
    example: str = ""
    num_true: bool | None = None
    derivable: bool = False
    witness: tuple[str, ...] = ()
    relation: str = "new"
    fires: int = 0
    fire_cases: tuple[str, ...] = ()
    changed: int = 0
    paid: int = 0
    verify_fail: int = 0
    reach: tuple[dict, ...] = ()
    cost_drop: float = 0.0
    cert_fail: int = 0
    closure_ratio: float = 1.0

    @property
    def truth(self) -> bool:
        """True iff derivable, or numerically true on a real instance."""
        return self.derivable or self.num_true is True

    @property
    def closure_safe(self) -> bool:
        """True iff adding the rule did not blow up the closure."""
        return self.closure_ratio <= _CLOSURE_LIMIT

    @property
    def shippable(self) -> bool:
        """True iff every gate a library entry must clear holds."""
        return (
            self.truth
            and self.relation == "new"
            and self.fires > 0
            and self.paid > 0
            and self.verify_fail == 0
            and self.cert_fail == 0
            and self.closure_safe
        )

    @property
    def no_ship_reason(self) -> str:
        """Return the first gate a no-ship candidate fails."""
        if self.shippable:
            return ""
        if not self.truth:
            if self.num_true is False:
                return "false (numeric oracle rejects)"
            if self.matches == 0:
                return "inapplicable (no real match)"
            return "unproven (no oracle)"
        if self.relation != "new":
            return f"not new ({self.relation})"
        if self.fires == 0:
            return "no firing on a real model"
        if self.paid == 0:
            return "fires but never lowers cost"
        if self.verify_fail:
            return "lowered modules differ"
        if self.cert_fail:
            return "certificate fails to replay"
        return "closure blow-up"


# ---------------------------------------------------------------------------
#  Measurement — firing, cost delta, certificate, closure safety
# ---------------------------------------------------------------------------


def _reach_row(
    case: TermCase,
    base_rules: list[Rewrite],
    laws: list[Rewrite],
    cost_fn: Any,
) -> dict:
    """Compare saturation of *case* with vs without *laws*.

    This is ``law_impact.reach_row`` with the base rule set
    parameterized (the impact tool hard-codes ``ALL_RULES``; the
    held-out run needs a different base).  It reuses the impact tool's
    ``_saturate`` / ``_cert_ok`` helpers, and adds nothing but the
    base argument.
    """
    base_eg, _br, base_best, base_stats = _saturate(
        case.term, list(base_rules), cost_fn
    )
    add_eg, _ar, add_best, add_stats = _saturate(
        case.term, [*base_rules, *laws], cost_fn
    )
    fires = {
        r.name: add_eg.rule_fires.get(r.name, 0)
        for r in laws
        if add_eg.rule_fires.get(r.name, 0)
    }
    base_cost = (
        dag_cost(base_best, cost_fn) if base_best else float("inf")
    )
    add_cost = dag_cost(add_best, cost_fn) if add_best else float("inf")
    return {
        "model": case.name,
        "base_enodes": base_stats["n_enodes"],
        "add_enodes": add_stats["n_enodes"],
        "base_cost": base_cost,
        "add_cost": add_cost,
        "changed": add_best != base_best,
        "new_fires": fires,
        "base_cert": _cert_ok(base_eg, case.term, base_best, cost_fn),
        "add_cert": _cert_ok(add_eg, case.term, add_best, cost_fn),
    }


def _instance_from_match(
    proposal: Proposal, matches: list[Any]
) -> tuple[Any, Any] | None:
    """Instantiate *proposal*'s RHS on its first viable real match.

    Applies the proposal's ``check`` / ``derive`` hooks (when present)
    to each match's substitution — a check veto or a ``None`` derive
    skips that match, exactly as an e-graph firing would — so the
    oracles see the same instance the rule would produce.
    """
    for sub in matches:
        subst = _term_match(proposal.lhs, sub)
        if subst is None:
            continue
        if proposal.check is not None:
            try:
                if not proposal.check(subst):
                    continue
            except Exception:
                continue
        inst = dict(subst)
        if proposal.derive is not None:
            try:
                extra = proposal.derive(subst)
            except Exception:
                continue
            if extra is None:
                continue
            inst.update(extra)
        return sub, _term_instantiate(proposal.rhs, inst)
    return None


def _fire(
    proposal: Proposal,
    models: list[TermCase],
    sink: Any,
    cost_fn: Any,
    ev: Evidence,
) -> None:
    """Fire *proposal* alone over every model; fill the firing fields."""
    rule = proposal.as_rule()
    cases: list[str] = []
    for case in models:
        f = _probe(case, rule, sink, cost_fn)
        if f.verified in ("FAIL", "error"):
            ev.verify_fail += 1
        if not f.fires:
            continue
        ev.fires += f.fires
        cases.append(case.name)
        if f.changed:
            ev.changed += 1
        if f.paid:
            ev.paid += 1
    ev.fire_cases = tuple(cases)


def _reach(
    proposal: Proposal,
    models: list[TermCase],
    base_rules: list[Rewrite],
    cost_fn: Any,
    ev: Evidence,
) -> None:
    """Measure end-to-end reach, certificate and closure ratio."""
    rows = tuple(
        _reach_row(c, base_rules, [proposal.as_rule()], cost_fn)
        for c in models
    )
    ev.reach = rows
    drops: list[float] = []
    ratios: list[float] = []
    for r in rows:
        base, add = r["base_cost"], r["add_cost"]
        if base not in (0.0, float("inf")) and add < base:
            drops.append((base - add) / base)
        if r["base_enodes"]:
            ratios.append(r["add_enodes"] / r["base_enodes"])
        if r["add_cert"] != "pass":
            ev.cert_fail += 1
    ev.cost_drop = max(drops) if drops else 0.0
    ev.closure_ratio = max(ratios) if ratios else 1.0


def _lhs_tuple(term: Any) -> tuple | None:
    """Return the census op-tuple key ``(op, child-op tuple)`` of *term*.

    The same key ``law_shape_census.op_tuple_census`` counts, so a
    candidate's LHS can be located in the census directly — the
    census -> propose link made explicit.
    """
    if not isinstance(term, Op):
        return None
    return (term.op, tuple(_op_of(a) for a in term.args))


def measure(
    proposal: Proposal,
    real_terms: list[Any],
    models: list[TermCase],
    base_rules: list[Rewrite],
    lib: list,
    census_op: dict,
    sink: Any,
    cost_fn: Any,
) -> Evidence:
    """Run both oracles and every measurement for one *proposal*."""
    ev = Evidence(proposal=proposal)
    ev.census_sites = census_op.get(_lhs_tuple(proposal.lhs), 0)
    schema = Schema(proposal.name, proposal.lhs, proposal.rhs)
    ev.relaxed = relaxed_matches(real_terms, schema)
    matches = real_matches(real_terms, schema)
    ev.matches = len(matches)
    ev.relation = lp._relation(proposal.lhs, proposal.rhs, lib)
    inst = proposal.instance or _instance_from_match(proposal, matches)
    if inst is not None:
        ev.num_true = lp._numeric_true(inst[0], inst[1])
        res = verify_law(inst[0], inst[1], base_rules)
        ev.derivable = res.derivable
        ev.witness = res.witness_rules
    if matches:
        ev.example = op_repr(matches[0])
    _fire(proposal, models, sink, cost_fn, ev)
    if ev.fires:
        _reach(proposal, models, base_rules, cost_fn, ev)
    return ev


# ---------------------------------------------------------------------------
#  Ranking — the deterministic order and the ship verdict
# ---------------------------------------------------------------------------


def _rank_key(ev: Evidence) -> tuple:
    """Return the documented best-first ordering key for *ev*.

    Shippable candidates first; then the largest end-to-end cost drop,
    the number of models it pays on, and the firing count; then truth,
    applicability and (as a tie-break) the smaller closure ratio.
    """
    return (
        1 if ev.shippable else 0,
        round(ev.cost_drop, 9),
        ev.paid,
        ev.fires,
        1 if ev.truth else 0,
        ev.matches,
        ev.relaxed,
        -ev.closure_ratio,
    )


def rank(evs: list[Evidence]) -> list[Evidence]:
    """Return *evs* sorted best-first by :func:`_rank_key`."""
    return sorted(evs, key=_rank_key, reverse=True)


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _search_rules(holdout: str | None) -> list[Rewrite]:
    """Return the search rule set, minus *holdout* if given."""
    if holdout is None:
        return list(ALL_RULES)
    names = {n for n in holdout.split(",") if n}
    return [r for r in ALL_RULES if r.name not in names]


def run_pipeline(
    holdout: str | None = None, vocab: str = "hand"
) -> dict:
    """Run census -> propose -> verify -> measure -> rank.

    ``holdout`` is a comma-separated list of library rule names to
    remove from the search rule set (the library used for duplicate
    detection, derivability, and the reach baseline) — the held-out
    rediscovery test.  ``vocab`` selects the generator's op alphabet:
    ``"hand"`` (the ``_POINTWISE`` / ``_VIEW_OPS`` tuples) or
    ``"derived"`` (property-classified over the corpus by
    ``tools/law_vocab.py``).
    """
    base_rules = _search_rules(holdout)
    lib = [lp._key(r.lhs, r.rhs) for r in base_rules]
    census = run_census(_CENSUS_TOP)
    census_op = {
        (e["op"], tuple(e["children"])): e["count"]
        for e in census["op_tuples"]
    }

    bench, _be = _bench_cases()
    models, _me = model_cases()
    real_terms = [c.term for c in [*bench, *models]]
    proposals = propose(census_op, real_terms, vocab)
    sink = _sink()
    cost_fn = _cost_fn(sink)

    evs = [
        measure(
            p,
            real_terms,
            models,
            base_rules,
            lib,
            census_op,
            sink,
            cost_fn,
        )
        for p in proposals
    ]
    ranked = rank(evs)
    return {
        "holdout": holdout,
        "vocab": vocab,
        "n_search_rules": len(base_rules),
        "n_bench": len(bench),
        "n_models": len(models),
        "census": {
            "n_terms": census["n_terms"],
            "n_op_nodes": census["n_op_nodes"],
            "n_op_tuples": len(census["op_tuples"]),
            "n_shapes": census["n_shapes"],
        },
        "proposals": len(proposals),
        "ranked": ranked,
        "held_out": _held_out(ranked, holdout),
    }


def _held_out(ranked: list[Evidence], holdout: str | None) -> dict:
    """Locate the held-out law's candidate in the ranking, if any.

    The held-out rule is taken from the *library* (``ALL_RULES``); its
    alpha-normal equality is matched against every ranked proposal so
    the location is structural, not a name lookup.
    """
    if holdout is None:
        return {"holdout": None, "found": False}
    name = next((n for n in holdout.split(",") if n), "")
    rule = next((r for r in ALL_RULES if r.name == name), None)
    if rule is None:
        return {"holdout": name, "found": False, "note": "unknown rule"}
    key = lp._key(rule.lhs, rule.rhs)
    swap = (key[1], key[0])
    for i, ev in enumerate(ranked, start=1):
        k = lp._key(ev.proposal.lhs, ev.proposal.rhs)
        if k in (key, swap):
            return {
                "holdout": name,
                "found": True,
                "rank": i,
                "of": len(ranked),
                "candidate": ev.proposal.name,
                "shippable": ev.shippable,
                "top": i == 1,
                "sources": list(ev.proposal.sources),
                "census_generated": (
                    "census-naturality" in ev.proposal.sources
                ),
                "census_sites": ev.census_sites,
                "matches": ev.matches,
                "fires": ev.fires,
                "paid": ev.paid,
                "cost_drop": ev.cost_drop,
            }
    return {"holdout": name, "found": False}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _row(i: int, ev: Evidence) -> str:
    """Render one ranked evidence row."""
    true = {True: "yes", False: "no", None: "-"}[ev.num_true]
    verdict = "SHIP" if ev.shippable else "no"
    return (
        f"{i:>4} {ev.proposal.name:<26} {ev.proposal.family:<18} "
        f"{ev.census_sites:>6} {true:>4} {ev.relation:<9} "
        f"{ev.matches:>5} {ev.fires:>5} "
        f"{ev.paid:>4} {ev.cost_drop * 100:>6.1f} "
        f"{'pass' if ev.cert_fail == 0 else 'FAIL':>4} "
        f"{ev.closure_ratio:>6.2f}x {verdict:>4}"
    )


def _table(ranked: list[Evidence], top: int) -> str:
    """Render the ranked evidence table."""
    head = (
        f"{'rank':>4} {'candidate':<26} {'family':<18} "
        f"{'census':>6} {'true':>4} {'rel':<9} {'match':>5} "
        f"{'fires':>5} {'paid':>4} {'drop%':>6} {'cert':>4} "
        f"{'enode':>7} {'ship':>4}"
    )
    lines = [head, "-" * len(head)]
    for i, ev in enumerate(ranked[:top], start=1):
        lines.append(_row(i, ev))
    return "\n".join(lines)


def _print_report(result: dict, top: int) -> None:
    """Print the full human-readable pipeline report."""
    ranked: list[Evidence] = result["ranked"]
    c = result["census"]
    print("== law_pipeline — census -> propose -> verify -> measure ==")
    ho = result["holdout"]
    print(
        f"   search rule set: {result['n_search_rules']} rules"
        + (f" (held out: {ho})" if ho else " (ALL_RULES)")
    )
    print(f"   op vocabulary: {result.get('vocab', 'hand')}")
    print(
        f"   corpus: {result['n_bench']} bench + "
        f"{result['n_models']} models"
    )
    print(
        f"   census: {c['n_op_nodes']} op nodes, "
        f"{c['n_op_tuples']} op-tuples, {c['n_shapes']} shapes"
    )
    print(f"   proposals: {result['proposals']}")
    ship = [e for e in ranked if e.shippable]
    print(
        f"   firing on a real model: "
        f"{sum(1 for e in ranked if e.fires)}; "
        f"shippable: {len(ship)}"
    )
    print()
    print(f"-- ranked candidates (top {top}) --")
    print(_table(ranked, top))
    print()
    print("-- ship recommendations (evidence per candidate) --")
    if not ship:
        print("  none — no candidate clears the ship bar")
    for i, ev in enumerate(ranked, start=1):
        if not ev.shippable:
            continue
        print(
            f"  #{i} {ev.proposal.name} [{ev.proposal.family}] — "
            f"true={ev.num_true} new={ev.relation} "
            f"census={ev.census_sites} match={ev.matches} "
            f"fires={ev.fires} paid={ev.paid} "
            f"drop={ev.cost_drop * 100:.1f}% cert=pass "
            f"enode={ev.closure_ratio:.2f}x"
        )
        print(f"      rule: {op_repr(ev.proposal.lhs)}")
        print(f"         -> {op_repr(ev.proposal.rhs)}")
        print(f"      cases: {', '.join(ev.fire_cases)}")
    print()
    print("-- why the rest did not ship --")
    for i, ev in enumerate(ranked, start=1):
        if ev.shippable:
            continue
        print(f"  #{i:>2} {ev.proposal.name:<26} {ev.no_ship_reason}")
    print()
    _print_held_out(result)


def _print_held_out(result: dict) -> None:
    """Print the held-out rediscovery verdict."""
    held = result["held_out"]
    print("-- held-out rediscovery --")
    if not held.get("holdout"):
        print("  (no holdout — run with --holdout <rule> to validate)")
        return
    if not held.get("found"):
        print(
            f"  held out: {held['holdout']} — NOT rediscovered "
            f"(no structurally-matching candidate was proposed)"
        )
        return
    print(
        f"  held out: {held['holdout']} — rediscovered as "
        f"{held['candidate']}: rank {held['rank']} of {held['of']}, "
        f"shippable={held['shippable']}"
    )
    print(
        f"  evidence: census LHS sites={held['census_sites']}, "
        f"match={held['matches']}, fires={held['fires']}, "
        f"paid={held['paid']}, drop={held['cost_drop'] * 100:.1f}%"
    )
    print(
        f"  proposed by: {', '.join(held['sources'])}"
        f" — census-generated={held['census_generated']}"
    )
    print(
        "  verdict: "
        + (
            "PASS — the pipeline ranks the known winner top"
            if held["top"] and held["shippable"]
            else "FAIL — the known winner is not ranked top/shippable"
        )
    )


def _dump_json(path: str, result: dict) -> None:
    """Write the machine-readable pipeline result."""
    ranked: list[Evidence] = result["ranked"]
    payload = {
        "holdout": result["holdout"],
        "vocab": result.get("vocab", "hand"),
        "n_search_rules": result["n_search_rules"],
        "n_bench": result["n_bench"],
        "n_models": result["n_models"],
        "census": result["census"],
        "proposals": result["proposals"],
        "held_out": result["held_out"],
        "ranked": [
            {
                "rank": i,
                "name": ev.proposal.name,
                "family": ev.proposal.family,
                "sources": list(ev.proposal.sources),
                "lhs": op_repr(ev.proposal.lhs),
                "rhs": op_repr(ev.proposal.rhs),
                "census_sites": ev.census_sites,
                "relaxed": ev.relaxed,
                "matches": ev.matches,
                "example": ev.example,
                "num_true": ev.num_true,
                "derivable": ev.derivable,
                "witness": list(ev.witness),
                "relation": ev.relation,
                "fires": ev.fires,
                "fire_cases": list(ev.fire_cases),
                "changed": ev.changed,
                "paid": ev.paid,
                "verify_fail": ev.verify_fail,
                "cost_drop": ev.cost_drop,
                "cert_fail": ev.cert_fail,
                "closure_ratio": ev.closure_ratio,
                "shippable": ev.shippable,
                "no_ship_reason": ev.no_ship_reason,
            }
            for i, ev in enumerate(ranked, start=1)
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run the pipeline and print (or dump) the ranked report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout",
        help="comma-separated library rule names to exclude "
        "(held-out rediscovery)",
    )
    parser.add_argument("--json", help="write machine-readable results")
    parser.add_argument("--top", type=int, default=_DEFAULT_TOP)
    parser.add_argument(
        "--vocab",
        choices=("hand", "derived"),
        default="hand",
        help="the generator's op alphabet: hand-written tuples "
        "(default) or property-classified over the corpus "
        "(tools/law_vocab.py)",
    )
    args = parser.parse_args(argv)

    result = run_pipeline(args.holdout, args.vocab)
    _print_report(result, args.top)
    if args.json:
        _dump_json(args.json, result)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
