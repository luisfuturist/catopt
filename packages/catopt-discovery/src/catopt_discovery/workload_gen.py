"""Workload generator — feed the law census *new* programs.

The law-discovery pipeline (``catopt_discovery.pipeline``) is validated:
it rediscovers shipped winners under holdout and its verdicts are
honest.  The corpus-expansion retro
(``project/retros/law-corpus-expansion.md``) then showed the corpus is
the bottleneck — real architectures added 31 op-tuples and the
pipeline's answers changed, but no new law shipped because the new
shapes were *deep*, not the parallel-view shapes the naturality
generator consumes.  The meta-game retro
(``project/retros/law-meta-game.md``) proved corpus knowledge IS the
discovery — a learned proposer loses ~10x to enumeration.  The last
untested half of self-play is therefore not a better proposer: it is
**new workloads feeding the census**.

This tool generates workloads at the level the census reads — IR
terms, no ``torch.export`` needed — two ways, then measures whether
they teach the pipeline anything:

* **resample** — census-resampled synthesis.  The census's own
  op-tuple statistics define a generative distribution: sample a
  corpus root op-tuple, expand each child by sampling an
  op-conditional tuple (``P(children | parent)``), mint attrs by
  resampling observed per-op attr dicts, and fill leaf slots with
  corpus-attested leaf shapes conditioned on the (parent, position)
  they fell out of.  Every *node* is a census-attested tuple by
  construction — the novelty is the composition, not the node.
* **mutate** — real-model mutation.  Take an exported model term and
  apply 1-3 point mutations: **swap** an op node for a same-arity
  corpus sibling, **graft** a shape-equal subterm from any corpus
  term into the site, or **lift** a leaf to a frequent subterm of the
  same shape.  Mutations preserve the input/output structure of a
  real program and are the only strategy that can mint a *new*
  op-tuple (a swap or leaf-lift rewrites a node's child signature).

Validity is enforced, not assumed.  Every candidate must

1. mint under ``Op.make``'s attr contract,
2. type-check: ``catopt_core.typing.shape_of`` must not return
   ``INVALID`` (and every leaf shape must be concrete),
3. use only ops the torch sink lowers (``supported_ops``),
4. *evaluate*: ``TorchConcreteEval.eval_term`` on random fp64 leaves
   must return a tensor — the same oracle machinery the pipeline's
   numeric-truth stage uses, applied one level up (whole term, not
   equality), and
5. differ from the corpus: the term's leaf-numbered ``shape_key``
   must not be any corpus root's or any corpus *subterm's* key.

On the lowering boundary, plainly: a generated ``TermCase`` carries
fresh feeds (``randn`` per ``Var`` shape) and param values
(``randn`` per ``Param`` shape), so ``catopt_discovery.impact._probe`` lowers and
verifies it exactly like a real model — no ``nn.Module`` or export
step is needed.  ``eval_term`` is the cheap pre-gate; the pipeline's
own ``_lower_extracted`` + ``sink.verify`` remains the referee.

The acceptance test then runs the census and the pipeline on the
enlarged corpus and reports the delta: new op-tuples, new shapes,
new proposals, new firing candidates, any SHIP.  Resampling can only
*re-encode* the census — new node-level structure is unreachable by
construction — so the honest prior is that yield comes, if at all,
from mutation.  The numbers decide.

Run::

    .venv/bin/python -m catopt_discovery.workload_gen --n 60 --seed 1
    .venv/bin/python -m catopt_discovery.workload_gen --n 60 --skip-pipeline
    .venv/bin/python -m catopt_discovery.workload_gen --json /tmp/gen.json

CPU-only; generation is seconds, the enlarged pipeline a few minutes
(``--gen-models`` caps how many generated terms join the expensive
firing/reach probe — the census and matcher always see all of them).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from catopt_core.ir import (
    Const,
    Op,
    Param,
    TensorType,
    Var,
    op_repr_dag,
)
from catopt_core.typing import INVALID, _shape_of
from catopt_torch.adapters import TorchSink

# Sibling tools own the corpus, the census, the oracle and the
# pipeline; reuse them so generated terms are judged by exactly the
# machinery real models are judged by.
from catopt_discovery import pipeline as lpipe
from catopt_discovery import proposal as lp
from catopt_discovery.census import (
    CorpusTerm,
    op_tuple_census,
    shape_census,
    shape_key,
    shape_repr,
)
from catopt_discovery.impact import (
    TermCase,
    _bench_cases,
    _cost_fn,
    _iter_subterms,
    model_cases,
)
from catopt_discovery.shape_proposal import _sink

__all__ = [
    "CorpusStats",
    "GenStats",
    "census_delta",
    "corpus_stats",
    "main",
    "mutant_term",
    "resampled_term",
    "run_with_workloads",
    "term_to_case",
    "valid_term",
]

_DEV = torch.device("cpu")

#: Depth cap for a synthesized term (corpus terms top out near 10).
_MAX_DEPTH = 10

#: Node budget for a generated term — keeps probes/saturations cheap.
_MAX_NODES = 80

#: Point mutations applied per mutant.
_MAX_MUTATIONS = 3

#: Candidate attempts per accepted term, per strategy.
_ATTEMPTS_PER_TERM = 60


# ---------------------------------------------------------------------------
#  Corpus statistics — the generative distribution the census implies
# ---------------------------------------------------------------------------


def _marker(t: Any) -> str:
    """Return the typed child marker: op name or leaf kind.

    Finer than the census's ``_op_of`` (which collapses ``Var`` and
    ``Param`` to ``·``): leaf *kind* is conditioning the generator
    needs — a ``linear``'s second operand is a ``param`` in every
    corpus occurrence, and resampling should keep it one.
    """
    if isinstance(t, Op):
        return t.op
    if isinstance(t, Const):
        return "const"
    if isinstance(t, Param):
        return "param"
    if isinstance(t, Var):
        return "var"
    return "leaf"


def _concrete_shape(t: Any) -> tuple | None:
    """Return *t*'s shape when it is a concrete all-int tuple."""
    s = _shape_of(t)
    if (
        isinstance(s, tuple)
        and s
        and all(isinstance(d, int) and d >= 0 for d in s)
    ):
        return s
    return None


@dataclass
class CorpusStats:
    """Every statistic the generators sample from.

    All are read off the real corpus — nothing is hand-written.  The
    typed tables distinguish ``var``/``param``/``const`` children (the
    census's ``·`` collapses them); novelty is still measured on the
    census's own collapsed keys.
    """

    # (op, (typed child markers)) -> count; and grouped by parent op.
    typed_tuples: Counter = field(default_factory=Counter)
    tuples_of: dict[str, Counter] = field(default_factory=dict)
    root_tuples: Counter = field(default_factory=Counter)
    # (parent op, position) -> typed child-marker multiset.
    child_at: dict = field(default_factory=dict)
    # op -> [(attrs dict, count), ...] — observed attr combinations.
    attr_dicts: dict[str, Counter] = field(default_factory=dict)
    const_vals: Counter = field(default_factory=Counter)
    # (parent op, position) -> [(kind, shape), ...] for leaf slots.
    leaf_at: dict = field(default_factory=dict)
    var_shapes: Counter = field(default_factory=Counter)
    param_shapes: Counter = field(default_factory=Counter)
    # concrete-shape -> corpus subterms with that output shape.
    pool_by_shape: dict = field(default_factory=dict)
    arity: dict = field(default_factory=dict)
    # Dedup key sets: every corpus root shape-key, and every corpus
    # subterm shape-key (a generated term equal to a real subterm
    # teaches the census nothing).
    root_keys: set = field(default_factory=set)
    sub_keys: set = field(default_factory=set)


def corpus_stats(cases: list[TermCase]) -> CorpusStats:
    """Mine every table the generators read, from *cases*' terms."""
    st = CorpusStats()
    for c in cases:
        t = c.term
        if isinstance(t, Op):
            key = (t.op, tuple(_marker(a) for a in t.args))
            st.root_tuples[key] += 1
        st.root_keys.add(shape_key(t))
        for s in _iter_subterms(t):
            if not isinstance(s, Op):
                continue
            kids = tuple(_marker(a) for a in s.args)
            st.typed_tuples[(s.op, kids)] += 1
            st.tuples_of.setdefault(s.op, Counter())[kids] += 1
            st.arity[s.op] = len(s.args)
            if s.attrs:
                st.attr_dicts.setdefault(s.op, Counter())[
                    tuple(sorted(s.attrs.items()))
                ] += 1
            st.sub_keys.add(shape_key(s))
            sh = _concrete_shape(s)
            if sh is not None:
                st.pool_by_shape.setdefault(sh, []).append(s)
            for i, a in enumerate(s.args):
                m = _marker(a)
                st.child_at.setdefault((s.op, i), Counter())[m] += 1
                if m in ("var", "param"):
                    ash = _concrete_shape(a)
                    if ash is not None:
                        st.leaf_at.setdefault((s.op, i), []).append(
                            (m, ash)
                        )
            for a in s.args:
                if isinstance(a, Const):
                    st.const_vals[a.value] += 1
        for leaf in _leaves(t):
            sh = _concrete_shape(leaf)
            if sh is None:
                continue
            if isinstance(leaf, Param):
                st.param_shapes[sh] += 1
            else:
                st.var_shapes[sh] += 1
    return st


def _leaves(term: Any, out: set | None = None) -> set:
    """Return every non-``Const`` leaf of *term*."""
    out = set() if out is None else out
    if isinstance(term, Op):
        for a in term.args:
            _leaves(a, out)
    elif isinstance(term, (Var, Param)):
        out.add(term)
    return out


def _weighted(rng: random.Random, counter: Counter) -> Any:
    """Sample a key of *counter* proportional to its count."""
    total = sum(counter.values())
    r = rng.uniform(0, total)
    for k, n in counter.items():
        r -= n
        if r <= 0:
            return k
    return next(iter(counter))


# ---------------------------------------------------------------------------
#  Validity — the five-gate acceptance every generated term must pass
# ---------------------------------------------------------------------------


def _leaf_env(term: Any) -> dict | None:
    """Return ``{leaf: randn fp64}`` for *term*, or ``None``.

    ``None`` is the consistency verdict: two ``Var``/``Param`` leaves
    sharing a name but not a shape cannot be bound (``IRModule`` keys
    inputs by name), so the term is not evaluable.
    """
    by_name: dict[str, tuple] = {}
    env: dict = {}
    for leaf in _leaves(term):
        shape = leaf.typ.shape
        if any(d is None or d < 0 for d in shape):
            return None
        prior = by_name.get(leaf.name)
        if prior is not None and prior != (type(leaf), shape):
            return None
        by_name[leaf.name] = (type(leaf), shape)
        env.setdefault(
            leaf,
            torch.randn(tuple(shape), dtype=torch.float64),
        )
    return env


def valid_term(
    term: Any,
    st: CorpusStats,
    seen: set,
    supported: frozenset,
) -> dict | None:
    """Gate *term*; return its eval env on success, else ``None``.

    The gates, in order: (i) is an op term with at least one ``Var``
    leaf (a workload consumes input) and within the node budget;
    (ii) every op lowerable; (iii) shape inference does not prove it
    ill-typed and leaf shapes are concrete + name-consistent;
    (iv) it *evaluates* under the torch concrete-eval oracle; (v) it
    is not a corpus root, corpus subterm, or prior emission.
    """
    if not isinstance(term, Op):
        return None
    if len(_iter_subterms(term)) > _MAX_NODES:
        return None
    if any(
        s.op not in supported
        for s in _iter_subterms(term)
        if isinstance(s, Op)
    ):
        return None
    if _shape_of(term) is INVALID or _shape_of(term) is None:
        return None
    if not any(isinstance(leaf, Var) for leaf in _leaves(term)):
        return None
    env = _leaf_env(term)
    if env is None:
        return None
    try:
        out = lp._eval_backend().eval_term(term, env)
    except Exception:
        return None
    if not isinstance(out, torch.Tensor):
        return None
    key = shape_key(term)
    if key in st.root_keys or key in st.sub_keys or key in seen:
        return None
    return env


# ---------------------------------------------------------------------------
#  Strategy (a) — census-resampled synthesis
# ---------------------------------------------------------------------------


class _Resampler:
    """Sample terms from the census's own conditional distribution."""

    def __init__(self, st: CorpusStats, rng: random.Random) -> None:
        """Bind the statistics and the PRNG; own the leaf counter."""
        self.st = st
        self.rng = rng
        self.leaf_id = 0

    def _leaf(self, parent: str, pos: int, kind: str) -> Any:
        """Mint a leaf for the ``(parent, pos)`` slot of kind *kind*."""
        st, rng = self.st, self.rng
        if kind == "const":
            return Const(_weighted(rng, st.const_vals))
        slot = st.leaf_at.get((parent, pos), [])
        local = [s for k, s in slot if k == kind]
        if local:
            shape = rng.choice(local)
        else:
            pool = st.var_shapes if kind == "var" else st.param_shapes
            if not pool:
                pool = st.var_shapes
            shape = _weighted(rng, pool)
        name = f"g{kind[0]}{self.leaf_id}"
        self.leaf_id += 1
        typ = TensorType(tuple(shape))
        return Param(name, typ) if kind == "param" else Var(name, typ)

    def _attrs(self, op: str) -> dict:
        """Sample an observed attrs dict for *op* verbatim."""
        dist = self.st.attr_dicts.get(op)
        if not dist:
            return {}
        return dict(_weighted(self.rng, dist))

    def _expand(self, op: str, depth: int) -> Any:
        """Expand *op* by sampling one of its corpus tuples."""
        if depth > _MAX_DEPTH or op not in self.st.tuples_of:
            return None
        kids = _weighted(self.rng, self.st.tuples_of[op])
        args: list = []
        for i, m in enumerate(kids):
            if m in ("var", "param", "const", "leaf"):
                args.append(self._leaf(op, i, m))
            else:
                child = self._expand(m, depth + 1)
                if child is None:
                    return None
                args.append(child)
        try:
            return Op.make(op, *args, **self._attrs(op))
        except ValueError:
            return None

    def sample(self) -> Any:
        """Return one candidate term (pre-gate), or ``None``."""
        root = _weighted(self.rng, self.st.root_tuples)
        op, kids = root
        if op not in self.st.tuples_of:
            return None
        args = []
        for i, m in enumerate(kids):
            if m in ("var", "param", "const", "leaf"):
                args.append(self._leaf(op, i, m))
            else:
                child = self._expand(m, 1)
                if child is None:
                    return None
                args.append(child)
        try:
            return Op.make(op, *args, **self._attrs(op))
        except ValueError:
            return None


# ---------------------------------------------------------------------------
#  Strategy (b) — real-model mutation
# ---------------------------------------------------------------------------


def _nodes_with_paths(term: Any) -> list:
    """Return ``(path, node)`` for every ``Op`` occurrence in *term*."""
    out: list = []

    def walk(t: Any, path: tuple) -> None:
        if isinstance(t, Op):
            out.append((path, t))
            for i, a in enumerate(t.args):
                walk(a, (*path, i))

    walk(term, ())
    return out


def _replace(term: Any, path: tuple, new: Any) -> Any:
    """Return *term* with the node at *path* substituted by *new*."""
    if not path:
        return new
    i = path[0]
    args = list(term.args)
    args[i] = _replace(term.args[i], path[1:], new)
    return Op.make(term.op, *args, **dict(term.attrs))


def _parent_slot(term: Any, path: tuple) -> tuple:
    """Return ``(parent_op, position)`` for the node at *path*."""
    if not path:
        return ("<root>", 0)
    parent = term
    for i in path[:-1]:
        parent = parent.args[i]
    return (parent.op, path[-1])


def _swap(
    term: Any,
    path: tuple,
    node: Op,
    st: CorpusStats,
    rng: random.Random,
) -> Any:
    """Swap *node*'s op for a same-arity corpus sibling.

    Siblings come from the census's ``(parent, pos) -> child`` table —
    ops the corpus itself attests in this slot — falling back to all
    same-arity ops when the slot is unexplored.  The replacement takes
    an observed attr dict of its own op; old attrs are not carried
    (a ``select``'s ``dim,index`` means nothing to a ``sigmoid``).
    """
    slot = _parent_slot(term, path)
    attested = [
        m
        for m in st.child_at.get(slot, {})
        if m not in ("var", "param", "const", "leaf", node.op)
    ]
    cands = [m for m in attested if st.arity.get(m) == len(node.args)]
    if not cands:
        cands = [
            o
            for o, a in st.arity.items()
            if a == len(node.args) and o != node.op
        ]
    if not cands:
        return None
    new_op = rng.choice(cands)
    dist = st.attr_dicts.get(new_op)
    attrs = dict(_weighted(rng, dist)) if dist else {}
    try:
        return _replace(
            term, path, Op.make(new_op, *node.args, **attrs)
        )
    except ValueError:
        return None


def _graft(
    term: Any,
    path: tuple,
    node: Op,
    st: CorpusStats,
    rng: random.Random,
) -> Any:
    """Replace *node* by a shape-equal corpus subterm.

    The donor pool is keyed by concrete output shape, so the splice
    preserves the parent's typing by construction; the donor's root
    op need not be attested at the slot — that freedom is exactly
    where new op-tuples come from.
    """
    sh = _concrete_shape(node)
    if sh is None:
        return None
    donors = [d for d in st.pool_by_shape.get(sh, []) if d != node]
    if not donors:
        return None
    return _replace(term, path, rng.choice(donors))


def _lift_leaf(term: Any, st: CorpusStats, rng: random.Random) -> Any:
    """Replace a random leaf by a shape-equal corpus subterm.

    A leaf slot is a "legal hole" with a known shape; grafting a
    frequent subterm into it both deepens the program and rewrites
    the parent node's child signature — the other source of new
    op-tuples.
    """
    paths: list = []

    def walk(t: Any, path: tuple) -> None:
        if isinstance(t, Op):
            for i, a in enumerate(t.args):
                walk(a, (*path, i))
        elif isinstance(t, (Var, Param)):
            paths.append((path, t))

    walk(term, ())
    rng.shuffle(paths)
    for path, leaf in paths:
        sh = _concrete_shape(leaf)
        if sh is None:
            continue
        donors = st.pool_by_shape.get(sh, [])
        if donors:
            return _replace(term, path, rng.choice(donors))
    return None


def mutant_term(
    case: TermCase, st: CorpusStats, rng: random.Random
) -> Any:
    """Return *case*'s term under 1-3 point mutations, or ``None``."""
    term = case.term
    n = rng.randint(1, _MAX_MUTATIONS)
    for _ in range(n):
        nodes = _nodes_with_paths(term)
        if not nodes:
            return None
        kind = rng.random()
        if kind < 0.25:
            new = _lift_leaf(term, st, rng)
        else:
            path, node = rng.choice(nodes)
            new = (
                _swap(term, path, node, st, rng)
                if kind < 0.6
                else _graft(term, path, node, st, rng)
            )
        if new is not None and new is not term:
            term = new
    return term if term is not case.term else None


def resampled_term(
    st: CorpusStats,
    rng: random.Random,
    sampler: _Resampler | None = None,
) -> Any:
    """Return one census-resampled candidate term (pre-gate)."""
    return (sampler or _Resampler(st, rng)).sample()


# ---------------------------------------------------------------------------
#  TermCase plumbing — feeds and params so the pipeline can measure
# ---------------------------------------------------------------------------


@dataclass
class GenStats:
    """Per-run counters for the honesty table."""

    attempted: Counter = field(default_factory=Counter)
    accepted: Counter = field(default_factory=Counter)
    rejected: Counter = field(default_factory=Counter)


def term_to_case(
    term: Any, name: str, source: str, env: dict
) -> TermCase | None:
    """Wrap *term* as a measurable ``TermCase`` using env's values.

    ``inputs`` are the term's ``Var`` leaves in first-encounter
    order; ``feed`` the matching env tensors; ``param_vals`` the
    ``Param`` env tensors keyed by name — exactly the plumbing
    ``catopt_discovery.impact.synthetic_cases`` builds by hand.
    """
    vs: list[Var] = []
    seen: set = set()
    for leaf in _leaves(term):
        if isinstance(leaf, Var) and leaf not in seen:
            seen.add(leaf)
            vs.append(leaf)
    params: dict[str, Param] = {}
    for leaf in _leaves(term):
        if isinstance(leaf, Param):
            params[leaf.name] = leaf
    return TermCase(
        source=source,
        name=name,
        term=term,
        inputs=tuple(vs),
        feed=tuple(env[v] for v in vs),
        param_vals={n: env[p] for n, p in params.items()},
    )


def _generate(
    cases: list[TermCase],
    st: CorpusStats,
    rng: random.Random,
    n: int,
    strategies: tuple,
    supported: frozenset,
) -> tuple[list[TermCase], GenStats]:
    """Draw *n* accepted terms split evenly across *strategies*."""
    stats = GenStats()
    out: list[TermCase] = []
    seen: set = set()
    sampler = _Resampler(st, rng)
    per = n // len(strategies)
    quota = {s: per for s in strategies}
    for _, s in zip(
        range(n - per * len(strategies)), strategies, strict=False
    ):
        quota[s] += 1
    for strategy in strategies:
        got = 0
        for _attempt in range(quota[strategy] * _ATTEMPTS_PER_TERM):
            if got >= quota[strategy]:
                break
            stats.attempted[strategy] += 1
            if strategy == "resample":
                cand = sampler.sample()
            else:
                case = rng.choice(cases)
                cand = mutant_term(case, st, rng)
            if cand is None:
                stats.rejected[(strategy, "mint")] += 1
                continue
            env = valid_term(cand, st, seen, supported)
            if env is None:
                stats.rejected[(strategy, "gate")] += 1
                continue
            c = term_to_case(
                cand, f"gen:{strategy}:{got}", f"gen-{strategy}", env
            )
            if c is None:
                stats.rejected[(strategy, "case")] += 1
                continue
            seen.add(shape_key(cand))
            out.append(c)
            stats.accepted[strategy] += 1
            got += 1
    return out, stats


# ---------------------------------------------------------------------------
#  Novelty — what the census sees that the corpus did not contain
# ---------------------------------------------------------------------------


def _as_corpus_terms(cases: list[TermCase]) -> list[CorpusTerm]:
    """Wrap cases as ``CorpusTerm``s for the census functions."""
    return [CorpusTerm(c.source, c.name, c.term) for c in cases]


def census_delta(cases: list[TermCase], gen: list[TermCase]) -> dict:
    """Diff the generated terms' census keys against the corpus's.

    Returns the *new* op-tuples and *new* shapes the generated set
    contributes, plus per-strategy breakdowns — the generated terms'
    own census counters.
    """
    base_ct = _as_corpus_terms(cases)
    gen_ct = _as_corpus_terms(gen)
    base_op, _ = op_tuple_census(base_ct)
    gen_op, gen_op_terms = op_tuple_census(gen_ct)
    base_sh, _ = shape_census(base_ct)
    gen_sh, gen_sh_terms = shape_census(gen_ct)
    new_tuples = {
        k: (n, len(gen_op_terms[k]))
        for k, n in gen_op.items()
        if k not in base_op
    }
    new_shapes = {
        k: (n, len(gen_sh_terms[k]))
        for k, n in gen_sh.items()
        if k not in base_sh
    }
    per_strategy: dict = {}
    for c in gen:
        op_c, _ = op_tuple_census(
            [CorpusTerm(c.source, c.name, c.term)]
        )
        sh_c, _ = shape_census([CorpusTerm(c.source, c.name, c.term)])
        rec = per_strategy.setdefault(
            c.source,
            {"terms": 0, "new_tuples": set(), "new_shapes": set()},
        )
        rec["terms"] += 1
        rec["new_tuples"] |= {k for k in op_c if k not in base_op}
        rec["new_shapes"] |= {k for k in sh_c if k not in base_sh}
    return {
        "new_op_tuples": new_tuples,
        "new_shapes": new_shapes,
        "gen_tuple_total": len(gen_op),
        "gen_shape_total": len(gen_sh),
        "per_strategy": per_strategy,
    }


# ---------------------------------------------------------------------------
#  Pipeline on the enlarged corpus — reuse law_pipeline verbatim
# ---------------------------------------------------------------------------


def _run_pipeline(
    corpus_cases: list[TermCase],
    probe_models: list[TermCase],
    vocab: str,
    holdout: str | None,
) -> dict:
    """Mirror ``catopt_discovery.pipeline.run_pipeline`` on an explicit corpus.

    ``corpus_cases`` feeds the census AND the matcher (the terms the
    proposals are derived from and checked against); ``probe_models``
    is the (possibly capped) TermCase list the firing/reach stage
    probes.  Splitting them is what lets all generated terms inform
    the census while only a bounded few pay the reach cost.
    """
    base_rules = lpipe._search_rules(holdout)
    lib = [lp._key(r.lhs, r.rhs) for r in base_rules]
    op_counts, _ = op_tuple_census(_as_corpus_terms(corpus_cases))
    census_op = {k: n for k, n in op_counts.items()}
    real_terms = [c.term for c in corpus_cases]
    proposals = lpipe.propose(census_op, real_terms, vocab)
    sink = _sink()
    cost_fn = _cost_fn(sink)
    evs = [
        lpipe.measure(
            p,
            real_terms,
            probe_models,
            base_rules,
            lib,
            census_op,
            sink,
            cost_fn,
        )
        for p in proposals
    ]
    return {
        "proposals": proposals,
        "ranked": lpipe.rank(evs),
        "n_tuples": len(census_op),
        "n_terms": len(corpus_cases),
    }


def run_with_workloads(
    cases: list[TermCase],
    gen: list[TermCase],
    probe_models: list[TermCase],
    gen_models: list[TermCase],
    vocab: str,
    holdout: str | None,
) -> dict:
    """Run the pipeline on the corpus, then corpus+generated."""
    base = _run_pipeline(cases, probe_models, vocab, holdout)
    big = _run_pipeline(
        [*cases, *gen], [*probe_models, *gen_models], vocab, holdout
    )
    return {"baseline": base, "enlarged": big}


# ---------------------------------------------------------------------------
#  Reporting
# ---------------------------------------------------------------------------


def _interleave(gen: list[TermCase], n: int) -> list[TermCase]:
    """Take up to *n* generated cases, round-robin across strategies.

    ``gen`` arrives strategy-blocked (all resampled terms first); a
    plain ``gen[:n]`` would put zero mutants in the firing/reach
    probe.  Round-robin over the source tag keeps every strategy in.
    """
    by_src: dict[str, list] = {}
    for c in gen:
        by_src.setdefault(c.source, []).append(c)
    out: list[TermCase] = []
    srcs = sorted(by_src)
    for i in range(max((len(v) for v in by_src.values()), default=0)):
        for s in srcs:
            if i < len(by_src[s]):
                out.append(by_src[s][i])
                if len(out) >= n:
                    return out
    return out


def _gen_table(gen: list[TermCase], stats: GenStats) -> str:
    """Render the per-strategy yield table."""
    head = (
        f"{'strategy':<10} {'attempts':>8} {'accepted':>8} "
        f"{'rate':>6} {'nodes~':>7} {'depth~':>7}"
    )
    lines = [head, "-" * len(head)]
    for strategy in ("resample", "mutate"):
        got = [g for g in gen if g.source == f"gen-{strategy}"]
        if not stats.attempted.get(strategy):
            continue
        att = stats.attempted[strategy]
        rate = len(got) / att if att else 0.0
        nodes = [len(_iter_subterms(g.term)) for g in got]
        depths = [_depth(g.term) for g in got]
        nm = f"{sum(nodes) / len(nodes):.0f}" if nodes else "-"
        dm = f"{sum(depths) / len(depths):.0f}" if depths else "-"
        lines.append(
            f"{strategy:<10} {att:>8} {len(got):>8} "
            f"{rate * 100:>5.1f}% {nm:>7} {dm:>7}"
        )
    return "\n".join(lines)


def _depth(t: Any) -> int:
    """Return the op-depth of *t* (leaves are 0)."""
    if not isinstance(t, Op):
        return 0
    return 1 + max((_depth(a) for a in t.args), default=0)


def _reject_table(stats: GenStats) -> str:
    """Render why candidates were rejected."""
    lines = [f"{'strategy':<10} {'reason':<8} {'count':>6}"]
    lines.append("-" * 26)
    for (s, why), n in sorted(stats.rejected.items()):
        lines.append(f"{s:<10} {why:<8} {n:>6}")
    return "\n".join(lines)


def _novelty_table(delta: dict, top: int = 20) -> str:
    """Render the new op-tuples and a sample of new shapes."""
    lines = [f"-- new op-tuples ({len(delta['new_op_tuples'])}) --"]
    for (op, kids), (n, nterms) in sorted(
        delta["new_op_tuples"].items(), key=lambda kv: -kv[1][0]
    )[:top]:
        lines.append(f"  {n:>4}x/{nterms}t  {op}({', '.join(kids)})")
    if not delta["new_op_tuples"]:
        lines.append("  (none)")
    lines.append(
        f"-- new shapes ({len(delta['new_shapes'])}; top {top}) --"
    )
    for key, (n, nterms) in sorted(
        delta["new_shapes"].items(), key=lambda kv: -kv[1][0]
    )[:top]:
        lines.append(f"  {n:>4}x/{nterms}t  {shape_repr(key)}")
    if not delta["new_shapes"]:
        lines.append("  (none)")
    return "\n".join(lines)


def _pipeline_delta(res: dict) -> str:
    """Render baseline-vs-enlarged pipeline comparison."""
    b, e = res["baseline"], res["enlarged"]
    b_names = {p.name for p in b["proposals"]}
    e_names = {p.name for p in e["proposals"]}
    b_rank = {ev.proposal.name: ev for ev in b["ranked"]}
    e_rank = {ev.proposal.name: ev for ev in e["ranked"]}
    lines = [
        f"corpus: {b['n_terms']} -> {e['n_terms']} terms, "
        f"{b['n_tuples']} -> {e['n_tuples']} op-tuples",
        f"proposals: {len(b_names)} -> {len(e_names)}",
    ]
    new_props = sorted(e_names - b_names)
    lines.append(f"new proposals ({len(new_props)}):")
    for name in new_props:
        ev = e_rank[name]
        gen_fires = sum(
            1 for c in ev.fire_cases if c.startswith("gen:")
        )
        lines.append(
            f"  {name:<28} [{ev.proposal.family}] "
            f"match={ev.matches} fires={ev.fires} "
            f"(gen={gen_fires}) paid={ev.paid} "
            f"ship={'SHIP' if ev.shippable else ev.no_ship_reason}"
        )
    if not new_props:
        lines.append("  (none)")
    gone = sorted(b_names - e_names)
    if gone:
        lines.append(f"proposals no longer emitted: {gone}")
    lines.append("-- firing/paid deltas on shared proposals --")
    shown = 0
    for name in sorted(b_names & e_names):
        be, ee = b_rank[name], e_rank[name]
        if ee.fires == be.fires and ee.matches == be.matches:
            continue
        gen_fires = sum(
            1 for c in ee.fire_cases if c.startswith("gen:")
        )
        lines.append(
            f"  {name:<28} match {be.matches}->{ee.matches} "
            f"fires {be.fires}->{ee.fires} (gen={gen_fires}) "
            f"paid {be.paid}->{ee.paid} "
            f"ship: {'Y' if ee.shippable else ee.no_ship_reason}"
        )
        shown += 1
    if not shown:
        lines.append("  (no shared proposal changed)")
    b_ship = [ev.proposal.name for ev in b["ranked"] if ev.shippable]
    e_ship = [ev.proposal.name for ev in e["ranked"] if ev.shippable]
    lines.append(
        f"shippable: {len(b_ship)} -> {len(e_ship)}"
        + (
            f"  new: {[s for s in e_ship if s not in b_ship]}"
            if e_ship
            else ""
        )
    )
    return "\n".join(lines)


def _evidence_rows(ranked: list) -> list[dict]:
    """Return the JSON-safe evidence row for every ranked candidate."""
    return [
        {
            "rank": i,
            "name": ev.proposal.name,
            "family": ev.proposal.family,
            "sources": list(ev.proposal.sources),
            "num_true": ev.num_true,
            "derivable": ev.derivable,
            "relation": ev.relation,
            "matches": ev.matches,
            "fires": ev.fires,
            "fire_cases": list(ev.fire_cases),
            "paid": ev.paid,
            "verify_fail": ev.verify_fail,
            "cert_fail": ev.cert_fail,
            "cost_drop": ev.cost_drop,
            "closure_ratio": ev.closure_ratio,
            "shippable": ev.shippable,
            "no_ship_reason": ev.no_ship_reason,
        }
        for i, ev in enumerate(ranked, start=1)
    ]


def _dump_json(path: str, payload: dict) -> None:
    """Write the machine-readable result."""
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Generate workloads, measure novelty, run the pipeline delta."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--strategy",
        choices=("resample", "mutate", "both"),
        default="both",
    )
    parser.add_argument(
        "--gen-models",
        type=int,
        default=16,
        help="max generated terms in the firing/reach probe",
    )
    parser.add_argument(
        "--vocab", choices=("hand", "derived"), default="derived"
    )
    parser.add_argument("--holdout", help="pipeline holdout rules")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    strategies = (
        ("resample", "mutate")
        if args.strategy == "both"
        else (args.strategy,)
    )

    print("== law_workload_gen — new workloads for the census ==")
    bench, _be = _bench_cases()
    models, _me = model_cases()
    cases = [*bench, *models]
    print(
        f"   corpus: {len(bench)} bench + {len(models)} models; "
        f"seed={args.seed} n={args.n} strategies={strategies}"
    )
    st = corpus_stats(cases)
    supported = TorchSink().supported_ops
    gen, gstats = _generate(
        cases, st, rng, args.n, strategies, supported
    )
    print()
    print("-- generation --")
    print(_gen_table(gen, gstats))
    print()
    print(_reject_table(gstats))
    print()

    delta = census_delta(cases, gen)
    print("-- census delta (generated vs corpus) --")
    print(
        f"   generated terms carry {delta['gen_tuple_total']} "
        f"distinct op-tuples ({len(delta['new_op_tuples'])} new), "
        f"{delta['gen_shape_total']} distinct shapes "
        f"({len(delta['new_shapes'])} new)"
    )
    for src, rec in sorted(delta["per_strategy"].items()):
        print(
            f"   {src}: {rec['terms']} terms, "
            f"{len(rec['new_tuples'])} new tuples, "
            f"{len(rec['new_shapes'])} new shapes"
        )
    print()
    print(_novelty_table(delta))
    print()

    payload: dict = {
        "n_requested": args.n,
        "n_generated": len(gen),
        "seed": args.seed,
        "attempted": dict(gstats.attempted),
        "accepted": dict(gstats.accepted),
        "rejected": {
            f"{s}:{w}": n for (s, w), n in gstats.rejected.items()
        },
        "new_op_tuples": {
            f"{k[0]}({','.join(k[1])})": v
            for k, v in delta["new_op_tuples"].items()
        },
        "n_new_shapes": len(delta["new_shapes"]),
        "gen_terms": [op_repr_dag(g.term) for g in gen],
    }

    if not args.skip_pipeline:
        print("-- pipeline delta (baseline vs corpus+generated) --")
        print("   (two full pipeline runs; a few minutes)")
        gen_models = _interleave(gen, args.gen_models)
        res = run_with_workloads(
            cases, gen, models, gen_models, args.vocab, args.holdout
        )
        print(_pipeline_delta(res))
        payload["pipeline"] = {
            "baseline": {
                "n_terms": res["baseline"]["n_terms"],
                "n_tuples": res["baseline"]["n_tuples"],
                "proposals": [
                    ev.proposal.name for ev in res["baseline"]["ranked"]
                ],
                "shippable": [
                    ev.proposal.name
                    for ev in res["baseline"]["ranked"]
                    if ev.shippable
                ],
            },
            "enlarged": {
                "n_terms": res["enlarged"]["n_terms"],
                "n_tuples": res["enlarged"]["n_tuples"],
                "proposals": [
                    ev.proposal.name for ev in res["enlarged"]["ranked"]
                ],
                "shippable": [
                    ev.proposal.name
                    for ev in res["enlarged"]["ranked"]
                    if ev.shippable
                ],
                "new_proposals": sorted(
                    {
                        ev.proposal.name
                        for ev in res["enlarged"]["ranked"]
                    }
                    - {
                        ev.proposal.name
                        for ev in res["baseline"]["ranked"]
                    }
                ),
                "ranked": _evidence_rows(res["enlarged"]["ranked"]),
            },
        }

    if args.json:
        _dump_json(args.json, payload)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
