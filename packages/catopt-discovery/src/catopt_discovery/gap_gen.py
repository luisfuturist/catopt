"""Gap-targeted workload synthesis — generation-as-search, measured.

The workload-generation retro (``project/retros/law-workload-gen.md``)
showed that *undirected* generation cannot feed the census: resampling
is confined to the census's tuple support (0 new op-tuples by
construction) and mutation's new tuples are semantically arbitrary
(0/48 were law-bearing shapes).  It left one direction open:
**generation as search** — synthesize a workload that *contains* a
census-identified unsatisfied shape, rather than hoping sampling hits
one.

This tool closes that loop.  The pipeline
(``catopt_discovery.pipeline``) reports ~half its proposals as
*inapplicable*: TRUE (or unevaluated) equalities whose LHS pattern
never appears in the real corpus — the ``no firing on a real model``
and ``inapplicable (no real match)`` verdicts.  For every such
candidate this tool

1. **synthesizes a minimal witness** — instantiate the proposal's LHS
   *pattern* to a concrete term: metavariables become fresh ``Var``
   leaves (repeated metavars share one leaf, so ``sub(x,x)`` really is
   self-subtraction) and attribute metavariables are minted against
   the instantiated child shapes (``select``'s ``index`` inside the
   chosen ``dim``'s extent, ``reshape``'s ``shape`` preserving numel,
   ``split``'s ``sizes`` summing right).  The instance must (i) match
   its own pattern under ``_term_match`` — the generation self-check —
   (ii) type-check (``_shape_of`` concrete), (iii) satisfy the
   proposal's ``check``/``derive`` side conditions exactly as an
   e-graph firing would see them (``bound`` carries ``"$attr:"``
   keys), and (iv) pass ``catopt_discovery.workload_gen.valid_term`` — sink-lowered
   ops only, evaluable under the torch oracle, novel vs every corpus
   root AND subterm;
2. **embeds it in a plausible program** — the bare instance (the
   ``min`` case), the instance in an ``add`` context (the ``ctx``
   case — an annihilator like ``x-x=0`` is only *defined* in context:
   at the root its 0-dim RHS cannot stand in for a shaped output),
   and the instance grafted into a real model skeleton at a
   shape-equal leaf slot (the ``graft`` case — the corpus model's own
   surrounding program);
3. **measures it through the pipeline's own referees** —
   ``catopt_discovery.impact._probe`` (does the lone rule fire? does extraction
   pick the rewrite? does the lowered module verify?) and
   ``catopt_discovery.impact._reach_row`` (``ALL_RULES`` vs ``ALL_RULES +
   {rule}``: end-to-end cost drop, certificate replay, enode closure
   ratio) — plus the oracle verdict the corpus never gave: the
   numeric-truth and derivability checks on the *generated* instance,
   which is the first instance the unproven candidates have ever had.

The acceptance test is the verdict movement: does an "inapplicable"
candidate now FIRE, does it PAY, does the generated workload admit it
as a real optimization (extracted cost drops, cert replays, lowered
module verifies)?  "Fires but never pays" is the honest verdict when
the corpus wasn't missing a law — it was missing a *useless* shape.

Run::

    .venv/bin/python -m catopt_discovery.gap_gen
    .venv/bin/python -m catopt_discovery.gap_gen --json /tmp/gap.json
    .venv/bin/python -m catopt_discovery.gap_gen --only \
        factor_left,grammar:pow_one --skip-baseline

``--skip-baseline`` re-derives proposals without re-measuring the
known-zero baseline firings (the cheap half of ``measure`` — matches,
relation, oracles — still runs); the default run executes the full
pipeline first so the target set is computed, not asserted.

CPU-only; the baseline pipeline is a few minutes, synthesis is
seconds, the per-candidate measurement a minute or two.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from catopt_core.egraph.terms import _term_instantiate, _term_match
from catopt_core.ir import Op, Param, TensorType, Var, op_repr
from catopt_core.typing import INVALID, _shape_of
from catopt_torch.adapters import TorchSink

# Sibling tools own the corpus, the census, the oracle, the pipeline
# and the five-gate validity checker; reuse them so a generated term
# is judged by exactly the machinery real models are judged by.
from catopt_discovery import pipeline as lpipe
from catopt_discovery import proposal as lp
from catopt_discovery import workload_gen as lwg
from catopt_discovery.impact import (
    TermCase,
    _bench_cases,
    _cost_fn,
    _probe,
    model_cases,
)
from catopt_discovery.shape_proposal import _sink
from catopt_discovery.verifier import verify_law

__all__ = [
    "GapResult",
    "gen_cases_for",
    "main",
    "synthesize",
    "targets",
]

#: Instantiation attempts per candidate instance.
_ATTEMPTS = 400

#: Distinct instances kept per candidate.
_INSTANCES = 2

#: Leaf-shape pool for metavariable instantiation.  Uniform draws
#: cover the pointwise laws (all metavars one shape); the broadcastable
#: entries (``(1,)``, ``(4,1)``, ``()``) give mixed-view candidates a
#: non-view operand that composes on BOTH sides of the rewrite.
_SHAPE_POOL = [
    (4, 16),
    (16, 16),
    (2, 8, 16),
    (8, 8),
    (16,),
    (4, 8),
    (4, 1),
    (1, 16),
    (1,),
    (),
    (2, 4, 4),
]

#: Uniform base shapes tried first, in order, before random draws.
_BASE_SHAPES = [(4, 16), (16, 16), (2, 8, 16)]


# ---------------------------------------------------------------------------
#  Pattern instantiation — mint a concrete term that matches the LHS
# ---------------------------------------------------------------------------


def _metavars(pattern: Any, out: list | None = None) -> list[str]:
    """Return the pattern's leaf metavariables, first-seen order."""
    out = [] if out is None else out
    if isinstance(pattern, str):
        if pattern not in out:
            out.append(pattern)
    elif isinstance(pattern, Op):
        for a in pattern.args:
            _metavars(a, out)
    return out


def _numel(shape: tuple) -> int:
    """Return the element count of a concrete *shape*."""
    n = 1
    for d in shape:
        n *= d
    return n


def _factorings(shape: tuple) -> list[tuple]:
    """Return alternative shapes with *shape*'s numel (for ``reshape``)."""
    n = _numel(shape)
    out = [(n,), tuple(shape)]
    for a in range(2, n):
        if n % a == 0:
            out.append((a, n // a))
    return out


def _super_shapes(shape: tuple) -> list[tuple]:
    """Return shapes *shape* broadcasts into (for ``expand``)."""
    out = []
    for cand in _SHAPE_POOL:
        if not cand or len(cand) < len(shape):
            continue
        pad = (1,) * (len(cand) - len(shape)) + tuple(shape)
        if all(
            b == c or b == 1 for b, c in zip(pad, cand, strict=True)
        ):
            out.append(cand)
    return out


def _op_attr_candidates(
    op: str, name: str, rank: int, sz: int, attrs: dict
) -> list | None:
    """Return op-specific attr values, or ``None`` (use generic)."""
    table: dict = {
        "select": {
            "dim": list(range(rank)),
            "index": list(range(sz)),
        },
        "slice": {
            "dim": list(range(rank)),
            "start": list(range(max(sz - 1, 1))),
            "end": list(range(int(attrs.get("start", 0)) + 1, sz + 1)),
            "step": [1],
        },
        "reshape": {"shape": None},  # filled below (needs shape0)
        "transpose": {
            "dim0": list(range(rank)),
            "dim1": list(range(rank)),
        },
        "unsqueeze": {"dim": list(range(rank + 1))},
        "squeeze": {"dim": list(range(rank))},
        "expand": {"shape": None},
        "broadcast_to": {"shape": None},
        "softmax": {"dim": list(range(rank))},
        "log_softmax": {"dim": list(range(rank))},
        "flatten": {
            "start_dim": list(range(rank)),
            "end_dim": list(
                range(int(attrs.get("start_dim", 0)), rank)
            ),
        },
        "chunk": {
            "chunks": [2, 4],
            "dim": list(range(rank)),
            "index": list(range(int(attrs.get("chunks", 2)))),
        },
        "split": {
            "sizes": None,  # filled below (needs sz)
            "dim": list(range(rank)),
            "index": [0, 1],
        },
    }
    spec = table.get(op)
    if spec is None:
        return None
    return spec.get(name)  # None -> generic/special-case below


def _attr_candidates(
    op: str, name: str, shape0: tuple | None, attrs: dict
) -> list:
    """Return mintable concrete values for one attr metavariable.

    ``shape0`` is the instantiated first child's shape; ``attrs`` the
    values already minted for THIS node (``index``/``start``/``end``
    read the already-chosen ``dim``).  An empty list vetoes the
    attempt — the enclosing instantiation retries with new leaves.
    """
    r = len(shape0) if shape0 else 0
    d = attrs.get("dim")
    d = d % r if isinstance(d, int) and r else 0
    sz = shape0[d] if shape0 and r else 4
    specific = _op_attr_candidates(op, name, r, sz, attrs)
    if specific is not None:
        return specific
    if op in ("reshape",) and name == "shape":
        return _factorings(shape0 or (4, 4))
    if op in ("expand", "broadcast_to") and name == "shape":
        return _super_shapes(shape0 or ())
    if op == "split" and name == "sizes":
        half = sz // 2
        cands = [[half, sz - half]]
        if sz % 2 == 0:
            cands.append(half)
        return cands
    generic: dict[str, list] = {
        "keepdim": [True, False],
        "index": list(range(min(sz, 4))),
        "start": [0],
        "end": [sz],
        "step": [1],
        "k": [1, 2],
        "correction": [1, 2],
        "diagonal": [1, 2],
        "chunks": [1, 2],
        "ord": [1, 2],
        "p": [1, 2],
        "shifts": [1],
        "equation": [],
    }
    if name in generic:
        return generic[name]
    if name == "dim":
        return [*range(r), *((i,) for i in range(r))]
    if name in ("dim0", "dim1", "start_dim", "end_dim"):
        return list(range(r))
    if name in ("shape", "sizes"):
        return _factorings(shape0 or (4, 4))
    if name in (
        "is_causal",
        "largest",
        "sorted",
        "descending",
        "equal_nan",
        "accumulate",
        "unbiased",
    ):
        return [True, False]
    if name in ("min", "max", "alpha", "beta", "value"):
        return [0.0, 1.0, 0.5]
    if name in ("rtol", "atol", "eps", "threshold", "scale"):
        return [1e-5, 1.0]
    return [0, 1]


#: Attr names whose value depends on a sibling attr minted earlier —
#: they are resolved in a second pass so ``dim`` is already bound.
_LATE_ATTRS = ("index", "start", "end")


def _instantiate(
    node: Any, leaf_subst: dict, attr_memo: dict, rng: random.Random
) -> Any:
    """Instantiate a pattern node under *leaf_subst*; ``None`` on veto.

    Metavariable leaves resolve through *leaf_subst*; attribute
    metavariables are minted once per *name* into *attr_memo* (shared
    across the whole pattern, so the two view nodes of a naturality
    law carry the same concrete attrs — the structural precondition
    the matcher enforces).  Literal attrs pass through.
    """
    if isinstance(node, str):
        return leaf_subst.get(node)
    if not isinstance(node, Op):
        return node
    args = [
        _instantiate(a, leaf_subst, attr_memo, rng) for a in node.args
    ]
    if any(a is None for a in args):
        return None
    shape0 = _shape_of(args[0])
    shape0 = shape0 if isinstance(shape0, tuple) else None
    items = sorted(
        node.attrs.items(), key=lambda kv: kv[0] in _LATE_ATTRS
    )
    attrs: dict = {}
    for k, v in items:
        if not isinstance(v, str):
            attrs[k] = v
            continue
        if v in attr_memo:
            attrs[k] = attr_memo[v]
            continue
        cands = _attr_candidates(node.op, k, shape0, attrs)
        if not cands:
            return None
        val = rng.choice(cands)
        attr_memo[v] = val
        attrs[k] = val
    try:
        return Op.make(node.op, *args, **attrs)
    except ValueError:
        return None


def _check_bound(proposal: Any, subst: dict) -> dict | None:
    """Return the firing-visible bound (subst + derive), or ``None``.

    Mirrors the e-graph's application path: ``check`` may veto,
    ``derive`` may add RHS-side bindings or veto with ``None``.
    """
    try:
        if proposal.check is not None and not proposal.check(subst):
            return None
        if proposal.derive is not None:
            extra = proposal.derive(subst)
            if extra is None:
                return None
            return {**subst, **extra}
    except Exception:
        return None
    return subst


def _instance_valid(
    term: Any,
    st: Any,
    seen: set,
    supported: frozenset,
    require_novel: bool,
) -> dict | None:
    """Gate an instance like ``valid_term``, with a novelty switch.

    Gates 1-4 are identical: op term in the node budget, sink-lowered
    ops only, concrete well-typed shape, ``Var``-bearing, name-
    consistent leaves, torch-evaluable.  Gate 5 (novel vs every corpus
    root and subterm) applies only when *require_novel* — a candidate
    whose LHS already matches the corpus (it matched a bench term but
    never fired on a model) is vetoed by gate 5 for the wrong reason.
    """
    if not isinstance(term, Op):
        return None
    if len(lwg._iter_subterms(term)) > lwg._MAX_NODES:
        return None
    if any(
        s.op not in supported
        for s in lwg._iter_subterms(term)
        if isinstance(s, Op)
    ):
        return None
    if _shape_of(term) is INVALID or _shape_of(term) is None:
        return None
    if not any(isinstance(leaf, Var) for leaf in lwg._leaves(term)):
        return None
    env = lwg._leaf_env(term)
    if env is None:
        return None
    try:
        out = lp._eval_backend().eval_term(term, env)
    except Exception:
        return None
    if not isinstance(out, torch.Tensor):
        return None
    key = lwg.shape_key(term)
    if key in seen:
        return None
    if require_novel and (key in st.root_keys or key in st.sub_keys):
        return None
    return env


def synthesize(
    proposal: Any,
    st: Any,
    supported: frozenset,
    seen: set,
    rng: random.Random,
    n: int = _INSTANCES,
    require_novel: bool = True,
) -> list[tuple[Any, dict, dict]]:
    """Return up to *n* ``(instance, bound, env)`` witnesses for the LHS.

    Each witness passes the generation self-check (the instance
    re-matches its own pattern), the side-condition hooks, and the
    five-gate ``valid_term`` validity (type-check, supported ops,
    torch-evaluable, corpus-novel).  Repeated metavars share one
    ``Var`` so equality preconditions hold by construction.
    """
    mvs = _metavars(proposal.lhs)
    out: list[tuple[Any, dict, dict]] = []
    keys: set = set()
    for attempt in range(_ATTEMPTS):
        if len(out) >= n:
            break
        if attempt < len(_BASE_SHAPES):
            shapes = {m: _BASE_SHAPES[attempt] for m in mvs}
        else:
            shapes = {m: rng.choice(_SHAPE_POOL) for m in mvs}
        leaf_subst = {
            m: Var(f"gw{attempt}_{i}", TensorType(s))
            for i, (m, s) in enumerate(shapes.items())
        }
        term = _instantiate(proposal.lhs, leaf_subst, {}, rng)
        if term is None:
            continue
        subst = _term_match(proposal.lhs, term)
        if subst is None:
            continue  # self-check: the instance must match its pattern
        bound = _check_bound(proposal, subst)
        if bound is None:
            continue
        try:
            rhs = _term_instantiate(proposal.rhs, bound)
        except Exception:
            continue
        if _shape_of(rhs) is INVALID:
            continue
        key = repr(term)
        if key in keys:
            continue
        env = _instance_valid(term, st, seen, supported, require_novel)
        if env is None:
            continue
        keys.add(key)
        out.append((term, bound, env))
    return out


# ---------------------------------------------------------------------------
#  Embedding — the instance inside a plausible surrounding program
# ---------------------------------------------------------------------------


def _leaf_paths(term: Any, path: tuple = ()) -> list:
    """Return ``(path, leaf)`` for every ``Var``/``Param`` in *term*."""
    out: list = []
    if isinstance(term, Op):
        for i, a in enumerate(term.args):
            out.extend(_leaf_paths(a, (*path, i)))
    elif isinstance(term, (Var, Param)):
        out.append((path, term))
    return out


def _concrete(term: Any) -> tuple | None:
    """Return *term*'s shape when it is a concrete all-int tuple."""
    s = _shape_of(term)
    if isinstance(s, tuple) and all(
        isinstance(d, int) and d >= 0 for d in s
    ):
        return s
    return None


def _case_env(term: Any) -> dict | None:
    """Return a fresh eval env for *term* (``None`` when unbindable)."""
    return lwg._leaf_env(term)


def gen_cases_for(
    proposal: Any,
    cases: list[TermCase],
    st: Any,
    supported: frozenset,
    seen: set,
    rng: random.Random,
    require_novel: bool = True,
) -> tuple[list[TermCase], dict]:
    """Build the generated ``TermCase``s for one candidate.

    For each synthesized instance: the bare instance (``min``), the
    instance inside ``add(·, fresh_var)`` (``ctx`` — the context an
    annihilator needs to be well-formed at all), and the instance
    grafted into a real model skeleton at a shape-equal leaf slot
    (``graft``).  Returns ``(cases, provenance)`` — provenance records
    attempts, the instance reprs, and which embeddings succeeded.
    """
    prov: dict[str, Any] = {"instances": [], "embeddings": []}
    out: list[TermCase] = []
    instances = synthesize(
        proposal, st, supported, seen, rng, require_novel=require_novel
    )
    prov["synthesized"] = len(instances)
    prov["_terms"] = [inst for inst, _bound, _env in instances]
    tag = proposal.name.replace(":", "_")
    for i, (inst, _bound, env) in enumerate(instances):
        prov["instances"].append(op_repr(inst))
        seen.add(lwg.shape_key(inst))
        c = lwg.term_to_case(inst, f"gap:{tag}:min{i}", "gen-gap", env)
        if c is not None:
            out.append(c)
            prov["embeddings"].append(f"min{i}")
        out_shape = _concrete(inst)
        if out_shape:
            w = Var(f"gw_ctx{i}", TensorType(out_shape))
            wrapped = Op.make("add", inst, w)
            wenv = lwg.valid_term(wrapped, st, seen, supported)
            if wenv is not None:
                seen.add(lwg.shape_key(wrapped))
                c = lwg.term_to_case(
                    wrapped, f"gap:{tag}:ctx{i}", "gen-gap", wenv
                )
                if c is not None:
                    out.append(c)
                    prov["embeddings"].append(f"ctx{i}")
        graft = _graft(inst, out_shape, cases, st, seen, supported, rng)
        if graft is not None:
            genv = _case_env(graft)
            if genv is not None:
                c = lwg.term_to_case(
                    graft, f"gap:{tag}:graft{i}", "gen-gap", genv
                )
                if c is not None:
                    out.append(c)
                    prov["embeddings"].append(f"graft{i}")
    return out, prov


def _graft(
    inst: Any,
    out_shape: tuple | None,
    cases: list[TermCase],
    st: Any,
    seen: set,
    supported: frozenset,
    rng: random.Random,
) -> Any:
    """Return *inst* grafted into a real skeleton's leaf slot, or ``None``.

    The donor is a corpus model term containing a ``Var`` leaf of
    exactly *inst*'s output shape — the same slot-substitution the
    mutator's ``_graft`` performs, except the donor is the candidate's
    own synthesized LHS, so the shape is present by construction.
    """
    if out_shape is None:
        return None
    donors: list = []
    for case in cases:
        if case.source != "model":
            continue
        for path, leaf in _leaf_paths(case.term):
            if isinstance(leaf, Var) and _concrete(leaf) == out_shape:
                donors.append((case, path))
    if not donors:
        return None
    rng.shuffle(donors)
    for case, path in donors:
        term = lwg._replace(case.term, path, inst)
        if lwg.valid_term(term, st, seen, supported) is not None:
            seen.add(lwg.shape_key(term))
            return term
    return None


# ---------------------------------------------------------------------------
#  Measurement — the pipeline's own referees on the generated workload
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    """One generated case probed through the pipeline's machinery."""

    name: str
    fires: int = 0
    changed: bool = False
    paid: bool = False
    verified: str = "-"
    note: str = ""
    base_cost: float = 0.0
    add_cost: float = 0.0
    cert: str = "-"
    closure: float = 1.0
    reach_fires: int = 0


@dataclass
class GapResult:
    """One inapplicable candidate plus its generated-workload verdict."""

    name: str
    family: str
    relation: str
    base_reason: str
    base_truth: bool
    synthesized: int = 0
    instances: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)
    num_true: bool | None = None
    derivable: bool = False
    rhs_instance: str = ""
    case_results: list = field(default_factory=list)

    @property
    def fires(self) -> int:
        """Total lone-rule firings across the generated cases."""
        return sum(r.fires for r in self.case_results)

    @property
    def paid(self) -> int:
        """Generated cases where the lone rule lowered the cost."""
        return sum(1 for r in self.case_results if r.paid)

    @property
    def verify_fail(self) -> int:
        """Generated cases whose lowered before/after differ."""
        return sum(
            1
            for r in self.case_results
            if r.verified in ("FAIL", "error")
        )

    @property
    def cert_fail(self) -> int:
        """Generated cases whose certificate fails to replay."""
        return sum(
            1
            for r in self.case_results
            if r.cert != "pass" and r.cert != "-"
        )

    @property
    def truth(self) -> bool:
        """Truth on the best evidence (baseline or generated instance).

        A measured ``num_true is False`` is a counterexample — it
        outranks both the baseline verdict and a derivation, matching
        ``pipeline.Evidence.truth`` and the gauntlet's truth gate.
        """
        if self.num_true is False:
            return False
        return (
            self.base_truth or self.num_true is True or self.derivable
        )

    @property
    def gen_drop(self) -> float:
        """Best end-to-end cost drop across the generated cases."""
        drops = [
            (r.base_cost - r.add_cost) / r.base_cost
            for r in self.case_results
            if r.base_cost and r.add_cost < r.base_cost
        ]
        return max(drops) if drops else 0.0

    @property
    def closure_ratio(self) -> float:
        """Worst enode out/in ratio across the generated cases."""
        return max((r.closure for r in self.case_results), default=1.0)

    @property
    def would_ship(self) -> bool:
        """The ship verdict IF the generated shape were real."""
        return (
            self.truth
            and self.relation == "new"
            and self.fires > 0
            and self.paid > 0
            and self.verify_fail == 0
            and self.cert_fail == 0
            and self.closure_ratio <= lpipe._CLOSURE_LIMIT
        )

    @property
    def new_reason(self) -> str:
        """The verdict the pipeline would reach on the gen evidence."""
        if self.would_ship:
            return "SHIP (on generated evidence)"
        if not self.truth:
            if self.num_true is False:
                return "false (numeric oracle rejects)"
            return "unproven (no oracle, even generated)"
        if self.relation != "new":
            return f"not new ({self.relation})"
        if self.fires == 0:
            return "still no firing — synthesis failed"
        if self.paid == 0:
            return "fires but never lowers cost"
        if self.verify_fail:
            return "lowered modules differ"
        if self.cert_fail:
            return "certificate fails to replay"
        if self.closure_ratio > lpipe._CLOSURE_LIMIT:
            return "closure blow-up"
        return "SHIP (on generated evidence)"


def targets(ranked: list, only: set | None = None) -> list:
    """Return the inapplicable-but-not-false candidates to close.

    A candidate is a gap target iff it never fired on a real model and
    the numeric oracle has not already rejected it — the TRUE-but-
    absent laws and the unproven ones.  ``only`` restricts to named
    candidates (dev loops).
    """
    out = []
    for ev in ranked:
        if ev.fires != 0 or ev.num_true is False:
            continue
        if only is not None and ev.proposal.name not in only:
            continue
        out.append(ev)
    return out


def _measure_case(
    case: TermCase,
    rule: Any,
    base_rules: list,
    sink: Any,
    cost_fn: Any,
) -> CaseResult:
    """Probe and reach-measure one generated case."""
    res = CaseResult(name=case.name)
    f = _probe(case, rule, sink, cost_fn)
    res.fires = f.fires
    res.changed = f.changed
    res.paid = f.paid
    res.verified = f.verified
    res.note = f.note
    row = lpipe._reach_row(case, base_rules, [rule], cost_fn)
    res.base_cost = row["base_cost"]
    res.add_cost = row["add_cost"]
    res.cert = row["add_cert"]
    res.reach_fires = sum(row["new_fires"].values())
    res.closure = (
        row["add_enodes"] / row["base_enodes"]
        if row["base_enodes"]
        else 1.0
    )
    return res


def measure_candidate(
    ev: Any,
    cases_for: list[TermCase],
    prov: dict,
    base_rules: list,
    sink: Any,
    cost_fn: Any,
) -> GapResult:
    """Measure one candidate's generated cases through the referees."""
    p = ev.proposal
    res = GapResult(
        name=p.name,
        family=p.family,
        relation=ev.relation,
        base_reason=ev.no_ship_reason,
        base_truth=ev.truth,
        synthesized=prov.get("synthesized", 0),
        instances=prov.get("instances", []),
        embeddings=prov.get("embeddings", []),
    )
    rule = p.as_rule()
    for case in cases_for:
        res.case_results.append(
            _measure_case(case, rule, base_rules, sink, cost_fn)
        )
    if res.instances:
        lhs_inst = prov["_terms"][0]
        sub = _term_match(p.lhs, lhs_inst)
        if sub is not None:
            bound = _check_bound(p, sub)
            if bound is not None:
                # Best-effort probe: an instantiate/eval/verify
                # failure just leaves the defaults on res.
                with suppress(Exception):
                    rhs = _term_instantiate(p.rhs, bound)
                    res.rhs_instance = op_repr(rhs)
                    res.num_true = lp._numeric_true(lhs_inst, rhs)
                    res.derivable = verify_law(
                        lhs_inst, rhs, base_rules
                    ).derivable
    return res


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _target_table(res: list[GapResult]) -> str:
    """Render the per-candidate verdict-movement table."""
    head = (
        f"{'candidate':<28} {'family':<18} {'rel':<9} {'gen':>3} "
        f"{'fires':>5} {'paid':>4} {'vf':>3} {'drop%':>6} "
        f"{'cert':>4} {'cl':>5}  verdict"
    )
    lines = [head, "-" * len(head)]
    for r in res:
        cert = "pass" if r.cert_fail == 0 else "FAIL"
        lines.append(
            f"{r.name:<28} {r.family[:18]:<18} {r.relation:<9} "
            f"{r.synthesized:>3} {r.fires:>5} {r.paid:>4} "
            f"{r.verify_fail:>3} {r.gen_drop * 100:>5.1f} "
            f"{cert:>4} {r.closure_ratio:>5.2f}  {r.new_reason}"
        )
    return "\n".join(lines)


def _case_table(res: list[GapResult]) -> str:
    """Render every generated case's probe/reach row."""
    head = (
        f"{'case':<34} {'fires':>5} {'chg':>4} {'paid':>4} "
        f"{'verify':>7} {'cost':>17} {'cert':>5} {'cl':>5}"
    )
    lines = [head, "-" * len(head)]
    shown = 0
    for r in res:
        for c in r.case_results:
            shown += 1
            lines.append(
                f"{c.name:<34} {c.fires:>5} "
                f"{'yes' if c.changed else 'no':>4} "
                f"{'yes' if c.paid else 'no':>4} {c.verified:>7} "
                f"{c.base_cost:>8.3g}->{c.add_cost:<8.3g} "
                f"{c.cert[:5]:>5} {c.closure:>5.2f}"
            )
            if c.note:
                lines.append(f"{'':>34} ↳ {c.note[:60]}")
    if not shown:
        lines.append("  (no generated cases)")
    return "\n".join(lines)


def _oracle_table(res: list[GapResult]) -> str:
    """Render the truth evidence the generated instance supplied."""
    head = f"{'candidate':<28} {'num_true':>8} {'derivable':>9}  rhs instance"
    lines = [head, "-" * len(head)]
    for r in res:
        if not r.instances:
            continue
        nt = {True: "yes", False: "NO", None: "-"}[r.num_true]
        lines.append(
            f"{r.name:<28} {nt:>8} {r.derivable!s:>9}  "
            f"{r.rhs_instance[:60]}"
        )
    return "\n".join(lines)


def _dump_json(path: str, res: list[GapResult], meta: dict) -> None:
    """Write the machine-readable result."""
    payload = {
        **meta,
        "candidates": [
            {
                "name": r.name,
                "family": r.family,
                "relation": r.relation,
                "base_reason": r.base_reason,
                "synthesized": r.synthesized,
                "instances": r.instances,
                "embeddings": r.embeddings,
                "num_true": r.num_true,
                "derivable": r.derivable,
                "rhs_instance": r.rhs_instance,
                "fires": r.fires,
                "paid": r.paid,
                "verify_fail": r.verify_fail,
                "cert_fail": r.cert_fail,
                "gen_drop": r.gen_drop,
                "closure_ratio": r.closure_ratio,
                "would_ship": r.would_ship,
                "new_reason": r.new_reason,
                "cases": [
                    {
                        "name": c.name,
                        "fires": c.fires,
                        "changed": c.changed,
                        "paid": c.paid,
                        "verified": c.verified,
                        "note": c.note,
                        "base_cost": c.base_cost,
                        "add_cost": c.add_cost,
                        "cert": c.cert,
                        "closure": c.closure,
                        "reach_fires": c.reach_fires,
                    }
                    for c in r.case_results
                ],
            }
            for r in res
        ],
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _named_targets(
    only: set | None,
    cases: list[TermCase],
    vocab: str,
    holdout: str | None,
) -> list | None:
    """Re-derive named proposals with the cheap evidence only.

    The ``--skip-baseline`` dev path: proposals come from
    ``lpipe.propose`` and each named candidate gets its relation,
    real-match count and (when it carries one) instance oracle — the
    firing baseline is asserted zero, not re-measured.  Returns
    ``None`` when ``only`` is unset (targets cannot be named).
    """
    if only is None:
        print(  # stdout-compat
            "   --skip-baseline requires --only (targets must "
            "be named when the baseline is not measured)"
        )
        return None
    from catopt_discovery.census import CorpusTerm, op_tuple_census
    from catopt_discovery.shape_proposal import Schema, real_matches

    base_rules = lpipe._search_rules(holdout)
    lib = [lp._key(r.lhs, r.rhs) for r in base_rules]
    op_counts, _ = op_tuple_census(
        [CorpusTerm(c.source, c.name, c.term) for c in cases]
    )
    real_terms = [c.term for c in cases]
    proposals = lpipe.propose(
        {k: n for k, n in op_counts.items()}, real_terms, vocab
    )
    by_name = {p.name: p for p in proposals}
    evs = []
    for name in sorted(only):
        p = by_name.get(name)
        if p is None:
            print(f"   !! {name}: no such proposal")  # stdout-compat
            continue
        ev = lpipe.Evidence(proposal=p)
        ev.relation = lp._relation(p.lhs, p.rhs, lib)
        ev.matches = len(
            real_matches(real_terms, Schema(p.name, p.lhs, p.rhs))
        )
        if p.instance is not None:
            ev.num_true = lp._numeric_true(*p.instance)
        evs.append(ev)
    print(  # stdout-compat
        f"   targets (named): {[e.proposal.name for e in evs]}"
    )
    return evs


def main(argv: list[str] | None = None) -> int:
    """Close the loop: synthesize the gap shapes, measure the laws."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--vocab", choices=("hand", "derived"), default="derived"
    )
    parser.add_argument("--holdout", help="pipeline holdout rules")
    parser.add_argument(
        "--only",
        help="comma-separated candidate names to restrict to",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="skip the full pipeline run; re-derive proposals and "
        "reuse only the cheap evidence (targets must be named)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=_INSTANCES,
        help="distinct instances to keep per candidate",
    )
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    only = set(args.only.split(",")) if args.only else None

    print(  # stdout-compat
        "== law_gap_targeted_gen — close the inapplicable loop =="
    )
    bench, _be = _bench_cases()
    models, _me = model_cases()
    cases = [*bench, *models]
    st = lwg.corpus_stats(cases)
    supported = TorchSink().supported_ops
    print(  # stdout-compat
        f"   corpus: {len(bench)} bench + {len(models)} models; "
        f"seed={args.seed} vocab={args.vocab}"
    )

    if args.skip_baseline:
        evs = _named_targets(only, cases, args.vocab, args.holdout)
        if evs is None:
            return 1
        base_rules = lpipe._search_rules(args.holdout)
    else:
        print(  # stdout-compat
            "   running the full pipeline for the baseline…"
        )
        result = lpipe.run_pipeline(args.holdout, args.vocab)
        base_rules = lpipe._search_rules(args.holdout)
        evs = targets(result["ranked"], only)
        print(  # stdout-compat
            f"   pipeline: {len(result['ranked'])} candidates, "
            f"{len(evs)} gap targets "
            f"(fires=0, not proven-false)"
        )

    sink = _sink()
    cost_fn = _cost_fn(sink)
    seen: set = set()
    results: list[GapResult] = []
    for ev in evs:
        gen, prov = gen_cases_for(
            ev.proposal,
            cases,
            st,
            supported,
            seen,
            rng,
            require_novel=ev.matches == 0,
        )
        results.append(
            measure_candidate(ev, gen, prov, base_rules, sink, cost_fn)
        )

    print()  # stdout-compat
    print("-- synthesized instances --")  # stdout-compat
    for r in results:
        print(  # stdout-compat
            f"  {r.name:<28} [{r.family}] rel={r.relation}"
        )
        print(f"      was: {r.base_reason}")  # stdout-compat
        for inst in r.instances:
            print(f"      gen: {inst[:66]}")  # stdout-compat
        if not r.instances:
            print(  # stdout-compat
                "      gen: (no valid instance synthesized)"
            )
        print(  # stdout-compat
            f"      embeddings: {', '.join(r.embeddings) or 'none'}"
        )
    print()  # stdout-compat
    print("-- oracle on the generated instance --")  # stdout-compat
    print(_oracle_table(results))  # stdout-compat
    print()  # stdout-compat
    print("-- per-case probe + reach --")  # stdout-compat
    print(_case_table(results))  # stdout-compat
    print()  # stdout-compat
    print("-- verdict movement --")  # stdout-compat
    print(_target_table(results))  # stdout-compat
    print()  # stdout-compat
    fired = [r for r in results if r.fires]
    paid = [r for r in results if r.paid]
    ships = [r for r in results if r.would_ship]
    print("== summary ==")  # stdout-compat
    print(  # stdout-compat
        f"  {len(results)} gap targets; "
        f"{sum(1 for r in results if r.synthesized)} synthesized; "
        f"{len(fired)} now fire; {len(paid)} pay on a generated case; "
        f"{len(ships)} would ship if the shape were real"
    )
    if ships:
        print(  # stdout-compat
            f"  would-ship: {[r.name for r in ships]}"
        )

    if args.json:
        _dump_json(
            args.json,
            results,
            {
                "seed": args.seed,
                "vocab": args.vocab,
                "n_targets": len(results),
            },
        )
        print(f"\nwrote {args.json}")  # stdout-compat
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
