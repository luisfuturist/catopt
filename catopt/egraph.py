"""E-graph with equality saturation for the categorical IR.

The e-graph maintains a partition of terms (ENodes) into equivalence
classes (EClasses).  Each EClass holds one or more ENodes known to be
semantically equivalent.  A union-find data structure merges classes
as rewrite rules fire.

The saturation loop:
1. Searches for matches of each rewrite rule's LHS pattern in the e-graph.
2. Adds the RHS term (instantiated with the matched bindings).
3. Unions the matched e-class with the new e-class.

Fixed point reached -> extract_best() uses a cost model to find the
minimum-cost representative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from catopt.ir import Op, op_repr


@dataclass(frozen=True)
class ENode:
    """A term node in the e-graph: an op name with e-class children."""

    op: str
    children: tuple[int, ...]
    attrs: tuple[tuple[str, Any], ...] = ()


@dataclass
class EClass:
    """An equivalence class of semantically-equivalent terms."""

    id: int
    nodes: set[ENode] = field(default_factory=set)
    cache: dict[str, Any] = field(default_factory=dict)
    # Lazily-built ``op -> member enodes`` index used by the matcher;
    # invalidated whenever ``nodes`` is mutated (union / rebuild).
    by_op: dict[str, list] | None = None


class UnionFind:
    """Union-find (disjoint-set with path compression + union by rank)."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []

    def make(self) -> int:
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        return idx

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path halving
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass(frozen=True)
class Rewrite:
    """A rewrite rule: lhs -> rhs (both are pattern terms).

    Patterns are terms where leaf variables are *strings* (metavariables).

    ``check`` is an optional side condition: a predicate over the
    *resolved* binding ``{metavar: representative_term}``.  Rules that
    are only valid for particular SHAPES (e.g. a scale that must be
    per-row or per-channel) declare it here — the matcher alone cannot
    see tensor types, and firing without the check produces well-typed
    but semantically wrong terms.

    ``derive`` is an optional computed-attribute hook: given the resolved
    binding it returns extra substitution entries (typically
    ``{"$attr:NAME": value}``) used when instantiating the RHS.  This is
    how a rule computes RHS attributes that are not present verbatim in
    the LHS — e.g. the uneven ``split`` sizes of an asymmetric pairing,
    derived from the bound weight shapes.  Returning ``None`` vetoes the
    rewrite.
    """

    name: str
    lhs: Any
    rhs: Any
    law: str = ""
    check: Any = None  # Callable[[dict[str, Any]], bool] | None
    derive: Any = None  # Callable[[dict], dict | None] | None
    # Bounded-error axis (ε-laws): when set, this rewrite is a
    # *certified approximation* — ``‖lhs − rhs‖ ≤ error_bound`` in the
    # norm named by ``bound_norm`` (e.g. spectral on a substituted
    # weight).  Exact rules leave it None.  Certificates aggregate the
    # per-step bounds conservatively (triangle inequality); extraction
    # can constrain or report the total.
    error_bound: float | None = None
    bound_norm: str = "spectral"

    def __repr__(self) -> str:
        return (
            f"{self.name}: {op_repr(self.lhs)} -> {op_repr(self.rhs)}"
        )


def _pattern_attrs(op: Op) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(op.attrs.items())) if op.attrs else ()


class _LeafRegistry:
    """Maps string keys to leaf term objects and vice-versa."""

    _key_to_term: dict[str, Any] = {}

    @classmethod
    def register(cls, term: Any) -> str:
        key = repr(term)
        cls._key_to_term[key] = term
        return key

    @classmethod
    def decode(cls, key: str) -> Any:
        return cls._key_to_term.get(key, key)


# ---------------------------------------------------------------------------
#  Proof-carrying merges — 2-morphisms as first-class data
# ---------------------------------------------------------------------------
#
#  Every merge records WHY it happened: ``apply_rule`` appends one
#  :class:`ProofEdge` per successful union (the canonical e-class ids on
#  the matched and produced sides plus the fired binding) and tags every
#  enode it instantiates with the creating rule application.  The
#  e-graph quotients the proof space — we keep a single witness per
#  merge (the first discovered), not all proofs.
#
#  :meth:`EGraph.certificate` replays that provenance into a positional
#  derivation — an ordered list of single-rule rewrites — connecting
#  the source term to an extracted member.  :func:`verify_certificate`
#  replays the steps on real terms, independent of the e-graph.  A step
#  that has no standalone justification (non-local merges such as the
#  diagram-level pairing pass, or manual unions) is recorded but flagged
#  ``egraph_dependent`` rather than silently trusted — UNLESS the pass
#  attached a synthesised :class:`Rewrite` via ``union(..., witness=...)``,
#  in which case the merge replays as an ordinary rule step.


@dataclass
class ProofEdge:
    """One recorded merge: the 2-morphism witnessing two e-classes.

    ``rule`` is the ``Rewrite.name`` that fired — ``None`` for merges
    performed outside rule application (non-local passes such as
    ``pair_shared_input_linears``, or direct ``union`` calls), or the
    name of a *synthesised* :class:`Rewrite` when the caller attached a
    replayable ``witness`` to :meth:`EGraph.union` (see there).
    ``a``/``b`` are the canonical e-class ids of the matched (LHS) and
    produced (RHS) sides *before* the merge; ``subst`` is the fired
    binding as ``(key, value)`` pairs — metavariables map to e-class
    ids, ``"$attr:"`` keys carry concrete attribute values.
    """

    rule: str | None
    a: int
    b: int
    subst: tuple = ()
    note: str = ""


@dataclass
class CertStep:
    """A single derivation step: ``rule`` applied at ``path``.

    ``path`` is a tuple of child indices locating the rewritten subterm
    inside the evolving term.  ``lhs``/``rhs`` are the concrete matched
    and produced term instances; ``bindings`` records metavariable ->
    term and ``"$attr:"`` -> value exactly as fired.

    ``egraph_dependent`` marks steps that cannot be replayed as a
    standalone rewrite — context-dependent merges (the pairing pass,
    manual unions) or derivation-budget exhaustion.  Verification
    substitutes them as trusted assertions: they are counted in
    ``Certificate.stats`` and rejected under ``strict=True``, never
    silently passed.
    """

    rule: str
    path: tuple
    lhs: Any = None
    rhs: Any = None
    bindings: dict = field(default_factory=dict)
    egraph_dependent: bool = False
    note: str = ""


class CertificateVerificationError(Exception):
    """The recorded derivation does not connect its claimed endpoints."""


@dataclass
class Certificate:
    """A proof-carrying derivation ``src`` -> ``dst``.

    ``steps``, applied in order, rewrite ``src`` into ``dst``: each is
    one rule application located by ``path``.  ``rules`` carries the
    :class:`Rewrite` objects used, so the certificate is self-contained
    for :func:`verify_certificate`.  ``stats`` summarises coverage —
    ``n_egraph_dependent`` counts steps the e-graph witnessed but that
    cannot replay standalone.
    """

    src: Any
    dst: Any
    root_eid: int | None
    steps: list = field(default_factory=list)
    rules: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def n_egraph_dependent(self) -> int:
        return sum(1 for s in self.steps if s.egraph_dependent)

    @property
    def replayable(self) -> bool:
        """True when every step is a standalone-replayable rewrite."""
        return self.n_egraph_dependent == 0

    @property
    def rules_used(self) -> list[str]:
        return sorted(
            {s.rule for s in self.steps if not s.egraph_dependent}
        )

    @property
    def error_bound(self) -> float:
        """Conservative accumulated error bound: the triangle-inequality
        sum of every step's ``Rewrite.error_bound`` (steps lacking a
        bound contribute 0 — they are exact).  The bound is in whatever
        norm the contributing rules declared; today that is the
        spectral norm on the substituted subterm.  Propagating
        site-local bounds to the model output requires per-op Lipschitz
        constants — not yet computed."""
        total = 0.0
        for s in self.steps:
            r = self.rules.get(s.rule)
            if r is not None and r.error_bound:
                total += r.error_bound
        return total

    @property
    def exact(self) -> bool:
        """True when no step carries an error bound — the derivation is
        an exact equivalence, not a certified approximation."""
        return self.error_bound == 0.0

    def render(self) -> str:
        lines = [
            f"certificate: {op_repr(self.src)}",
            f"        ==> {op_repr(self.dst)}",
        ]
        for i, s in enumerate(self.steps):
            tag = "  [e-graph-dependent]" if s.egraph_dependent else ""
            r = self.rules.get(s.rule)
            if r is not None and r.error_bound:
                tag += f"  [ε≤{r.error_bound:.3e} {r.bound_norm}]"
            lines.append(
                f"  {i:>3}. {s.rule} @{list(s.path)}: "
                f"{op_repr(s.lhs)} -> {op_repr(s.rhs)}{tag}"
            )
        if not self.exact:
            lines.append(f"  total ε bound: {self.error_bound:.3e}")
        return "\n".join(lines)


class EGraph:
    """The equality-saturation data structure.

    The e-graph is a *truncation* of the program ∞-groupoid, and
    ``truncation_level`` selects how much of it is materialised:

    - **1 — pure quotient.**  Only the partition into e-classes is
      kept: no :class:`ProofEdge` records, no per-enode rule
      provenance.  Minimal memory; :meth:`certificate` degrades to a
      proof-free marker and :meth:`all_proofs` is unavailable.
    - **2 — witnesses (default).**  One :class:`ProofEdge` per merge
      plus per-enode provenance — the data :meth:`certificate` needs.
      The cost is O(1) per union/enode: a small record per edge, never
      a second pass over the proof space.
    - **3 — lazy coherences.**  Same storage as level 2, plus
      :meth:`all_proofs` / :meth:`coherent_paths` materialise
      *alternate derivations* between two terms on demand — bounded
      enumeration over the rules that fired, stored nowhere.  (The
      level-2 merge log is a forest — one witness per merge — so
      coherence between proofs is a property of the term-rewriting
      space, computed lazily rather than enumerated eagerly.)

    Backward compatibility: an explicit ``track_proofs`` flag overrides
    the dial — ``True`` selects level 2, ``False`` selects level 1.
    """

    def __init__(
        self,
        track_proofs: bool | None = None,
        truncation_level: int = 2,
    ) -> None:
        if truncation_level not in (1, 2, 3):
            raise ValueError(
                f"truncation_level must be 1, 2, or 3, "
                f"got {truncation_level!r}"
            )
        if track_proofs is not None:
            # Backward-compat override: the old boolean flag maps onto
            # the dial — True -> level 2 (witnesses), False -> level 1
            # (pure quotient).
            truncation_level = 2 if track_proofs else 1
        self.truncation_level = truncation_level
        self._uf = UnionFind()
        self._classes: dict[int, EClass] = {}
        self._node_to_class: dict[ENode, int] = {}
        self._next_id = 0
        self.rule_fires: dict[str, int] = {}
        # -- proof tracking (level >= 2 only) --
        self._track = truncation_level >= 2
        self._merge_log: list[ProofEdge] = []
        self._applications: list[dict] = []
        self._enode_origin: dict[ENode, str] = {}
        self._enode_birth: dict[ENode, int] = {}
        self._enode_app: dict[ENode, int] = {}
        self._rule_objs: dict[str, Rewrite] = {}
        self._tag_rule: str | None = (
            None  # rule context (RHS instantiate)
        )
        self._collect: list | None = None  # enodes born mid-instantiate
        self._inst_last_enode: ENode | None = None
        # -- incremental saturation state --------------------------------
        # The matcher only needs to re-search an e-class when the match
        # results *could* have changed: the class gained enodes, or some
        # class reachable from it through enode children changed.  We
        # maintain the inverted child->parent adjacency so every change
        # can dirty exactly the affected ancestor cone.
        self._dirty: set[int] = set()
        self._parents: dict[int, set[int]] = {}
        # ``op name -> canonical class ids containing a member with that
        # op`` — a rule whose LHS is an Op pattern can only match at
        # those classes, so the per-iteration scan never touches the
        # rest of the graph.
        self._op_classes: dict[str, set[int]] = {}
        # Names of rules already applied to the whole graph once —
        # after the first pass the dirty frontier suffices.
        self._applied_rules: set[str] = set()
        # Per-apply_rule memo for ``any_term`` resolutions (checks fire
        # per substitution and re-resolve the same bound classes).
        self._anyterm_memo: dict[int, Any] | None = None
        # Cumulative ``rule name -> enodes contributed`` under
        # ``rule_budgets`` — persists across ``run`` calls so the
        # budget bounds a rule's expansion over the graph's lifetime,
        # not per call (the post-pass re-saturations in
        # ``optimize_model`` cannot re-open a closed budget).
        self._budget_spent: dict[str, int] = {}

    #: Per-class match-enumeration cap used when a rule runs under an
    #: ``enode_budget``: large enough to cover the substitutions a
    #: non-closure class realistically produces, small enough that a
    #: combinatorially-exploded class cannot dominate a pass.
    _MATCH_CAP: int = 256

    @property
    def n_classes(self) -> int:
        return len(self._classes)

    @property
    def n_enodes(self) -> int:
        return len(self._node_to_class)

    @property
    def n_nodes(self) -> int:
        return sum(len(c.nodes) for c in self._classes.values())

    def find(self, eid: int) -> int:
        return self._uf.find(eid)

    def get_class(self, eid: int) -> EClass:
        return self._classes[self.find(eid)]

    def add_leaf(self, key: str, provenance: str | None = None) -> int:
        """Add a leaf (Var/Const/Param) identified by *key*."""
        enode = ENode("leaf", (), (("key", key),))
        if enode in self._node_to_class:
            return self.find(self._node_to_class[enode])
        return self._add_enode(enode, provenance)

    def add_enode(
        self,
        op: str,
        children: tuple[int, ...],
        attrs: dict[str, Any] | None = None,
        provenance: str | None = None,
    ) -> int:
        """Add an ENode with already-resolved child e-class IDs.

        ``provenance`` optionally names what introduced the enode
        (e.g. a non-local pass); enodes born inside a rule application
        are tagged with the firing rule regardless.
        """
        attr_t = tuple(sorted((attrs or {}).items()))
        enode = ENode(op, tuple(self.find(c) for c in children), attr_t)
        if enode in self._node_to_class:
            return self.find(self._node_to_class[enode])
        return self._add_enode(enode, provenance)

    def _add_enode(
        self, enode: ENode, provenance: str | None = None
    ) -> int:
        eid = self._next_id
        self._next_id += 1
        self._uf.parent.append(eid)
        self._uf.rank.append(0)
        self._classes[eid] = EClass(id=eid)
        self._node_to_class[enode] = eid
        self._classes[eid].nodes.add(enode)
        # Incremental-search bookkeeping: a fresh class must be
        # searched (its enode may match rules), it contributes its op
        # to the class index, and each child class gains a parent edge
        # so future changes below propagate dirtiness upward.
        self._dirty.add(eid)
        self._op_classes.setdefault(enode.op, set()).add(eid)
        for c in enode.children:
            self._parents.setdefault(self.find(c), set()).add(eid)
        if self._track:
            self._enode_birth[enode] = eid
            self._enode_origin[enode] = (
                self._tag_rule or provenance or "external"
            )
            if self._collect is not None:
                self._collect.append(enode)
        return eid

    def add_term(
        self,
        term: Any,
        _memo: dict | None = None,
        provenance: str = "input",
    ) -> int:
        """Add a term (Var/Const/Param/Op) to the e-graph.

        ``_memo`` is a content-keyed cache: exported IR terms are DAGs
        with heavy sharing (residual streams, RoPE tables), and without
        memoisation the recursion re-walks shared subtrees
        exponentially.

        ``provenance`` tags every enode the term creates — ``"input"``
        for the source program, distinguishing it from rule-introduced
        and pass-introduced enodes in certificate generation.
        """
        memo = {} if _memo is None else _memo
        key = term
        if key in memo:
            return memo[key]
        if isinstance(term, Op):
            child_eids = tuple(
                self.add_term(a, memo, provenance) for a in term.args
            )
            attr_t = _pattern_attrs(term)
            enode = ENode(term.op, child_eids, attr_t)
            if enode in self._node_to_class:
                memo[key] = self.find(self._node_to_class[enode])
            else:
                memo[key] = self._add_enode(enode, provenance)
            return memo[key]
        _LeafRegistry.register(term)
        memo[key] = self.add_leaf(repr(term), provenance)
        return memo[key]

    def union(
        self,
        a: int,
        b: int,
        rule: str | None = None,
        subst: dict | None = None,
        note: str = "",
        witness: Rewrite | None = None,
    ) -> bool:
        """Merge the e-classes of ``a`` and ``b``.

        ``rule``/``subst`` optionally record the 2-morphism witnessing
        the merge: which rewrite fired and under what binding.  Only the
        first witness per merge is kept — the e-graph quotients the
        proof space.

        ``witness`` is the API for *non-local passes* — transformations
        that offer a member no LHS pattern could produce (the offered
        term is computed from the whole e-graph, not from a matched
        subterm).  The pass synthesises a concrete :class:`Rewrite`
        justifying *this specific* merge — typically a **pointwise**
        rule whose ``lhs`` is a member of ``a``'s class and whose
        ``rhs`` is the offered term itself — and the merge then replays
        like an ordinary rule application: the witness is registered
        alongside the fired rules, and a synthetic application record
        (keyed on the offered term's root enode) lets
        :meth:`certificate` expand the offered member through it.  The
        emitted :class:`CertStep` is re-matched and re-instantiated by
        :func:`verify_certificate` standalone — no ``egraph_dependent``
        stub.  The witness asserts the equality the pass established
        non-locally; the certificate records *what was asserted* in a
        form that replays on real terms.  (A metavariable-pattern
        witness with a ``derive`` that reconstructs the RHS from bound
        pieces works too; the application record is only registered
        when ``witness.rhs`` locates a concrete enode in the graph.)
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._uf.union(ra, rb):
            new_canon = self.find(ra)
            old_canon = ra if new_canon == rb else rb
            target = self._classes[new_canon]
            source = self._classes[old_canon]
            target.nodes |= source.nodes
            target.cache.clear()
            target.by_op = None
            del self._classes[old_canon]
            # -- incremental-search bookkeeping --------------------------
            # The merged class gained enodes, and every enode anywhere
            # whose child resolved to ``old_canon`` now resolves to
            # ``new_canon`` — both can enable new matches, and those
            # new matches can in turn enable matches in the parents of
            # those classes.  The affected region is exactly the
            # upward closure of the merged class over child->parent
            # edges, so mark it all dirty.
            self._dirty.add(new_canon)
            p_old = self._parents.pop(old_canon, None)
            if p_old:
                self._parents.setdefault(new_canon, set()).update(p_old)
            for n in source.nodes:
                oc = self._op_classes.get(n.op)
                if oc is not None:
                    oc.discard(old_canon)
                    oc.add(new_canon)
            stack = [new_canon]
            while stack:
                c = stack.pop()
                for p0 in self._parents.get(c, ()):
                    p = self.find(p0)
                    if p == c or p in self._dirty:
                        # ``p in _dirty`` prune is sound: an already-
                        # dirty class's ancestors were dirtied when it
                        # became dirty (parent edges are only ever
                        # created pointing at dirty-or-fresh classes).
                        continue
                    self._dirty.add(p)
                    stack.append(p)
            if self._track:
                if witness is not None:
                    rule = witness.name
                    self._rule_objs.setdefault(witness.name, witness)
                    # Locate the offered term's root enode and register
                    # a synthetic application so ``_connect``'s
                    # target-expansion strategy can replay this merge;
                    # ``_edge_path`` also finds the witness on its own
                    # whenever the current subterm matches its LHS.
                    _weid, w_en = self._locate(witness.rhs)
                    if w_en is not None:
                        app_idx = len(self._applications)
                        self._applications.append(
                            {
                                "rule": witness.name,
                                "matched_eid": self.find(ra),
                                "rhs_eid": self.find(rb),
                                "subst": dict(subst or {}),
                                "rhs_root_enode": w_en,
                            }
                        )
                        self._enode_app.setdefault(w_en, app_idx)
                fs = (
                    tuple(sorted(subst.items(), key=lambda kv: kv[0]))
                    if subst
                    else ()
                )
                self._merge_log.append(
                    ProofEdge(rule, ra, rb, fs, note)
                )
            return True
        return False

    # -- proof-log accessors ----------------------------------------------

    @property
    def merge_log(self) -> list[ProofEdge]:
        """Every recorded merge witness, in order of application."""
        return list(self._merge_log)

    @property
    def n_proof_edges(self) -> int:
        return len(self._merge_log)

    @property
    def applications(self) -> list[dict]:
        """Every rule application recorded during saturation."""
        return list(self._applications)

    # -- pattern matching --

    def matches(
        self, pattern: Any, eid: int, max_results: int | None = None
    ) -> list[dict[str, int]]:
        """Find all substitutions that match *pattern* at e-class *eid*.

        ``max_results`` bounds the enumeration: matching stops once
        that many substitutions have been found — deterministic (the
        first-found wins) and used by the saturation loop to enforce
        per-rule expansion budgets inside giant e-classes.
        """
        results: list[dict[str, int]] = []
        self._match(pattern, eid, {}, results, max_results)
        return results

    def _match(
        self,
        pattern: Any,
        eid: int,
        subst: dict[str, int],
        results: list[dict[str, int]],
        limit: int | None = None,
    ) -> None:
        if limit is not None and len(results) >= limit:
            return
        eid = self.find(eid)
        eclass = self._classes[eid]

        if isinstance(pattern, str):
            if pattern in subst:
                if subst[pattern] == eid:
                    results.append(dict(subst))
                return
            else:
                subst[pattern] = eid
                results.append(dict(subst))
                del subst[pattern]
                return

        if isinstance(pattern, Op):
            attr_t = _pattern_attrs(pattern)
            for node in self._nodes_of(eclass, pattern.op):
                if limit is not None and len(results) >= limit:
                    return
                if len(node.children) != len(pattern.args):
                    continue
                # Attribute matching: every pattern attr key must exist in
                # the node with an equal value — UNLESS the pattern value
                # is a string, which makes it an attribute metavariable
                # bound into the substitution under a "$attr:" key.
                # This is how shape-polymorphic rules match e.g.
                # view(t, shape=S) for any concrete S.
                node_attrs = dict(node.attrs)
                if set(node_attrs) != {k for k, _ in attr_t}:
                    continue
                attr_substs: list[dict[str, Any]] = [dict(subst)]
                attr_ok = True
                for k, pv in attr_t:
                    nv = node_attrs[k]
                    if isinstance(pv, str):
                        key = "$attr:" + pv
                        nxt: list[dict[str, Any]] = []
                        for cs in attr_substs:
                            if key in cs:
                                if cs[key] == nv:
                                    nxt.append(cs)
                            else:
                                cc = dict(cs)
                                cc[key] = nv
                                nxt.append(cc)
                        attr_substs = nxt
                    elif nv != pv:
                        attr_ok = False
                        break
                    if not attr_substs:
                        attr_ok = False
                        break
                if not attr_ok:
                    continue
                # Thread the incoming bindings so that a metavariable which
                # appears at several positions (e.g. the shared input x in
                # x@W1 + x@W2) is checked for consistency everywhere.
                # Starting from {} would silently rebind it, turning an
                # unSound rewrite into an apparent match.
                child_substs: list[dict[str, Any]] = attr_substs
                ok = True
                for i, pat_arg in enumerate(pattern.args):
                    new_substs: list[dict[str, int]] = []
                    for cs in child_substs:
                        if (
                            limit is not None
                            and len(results) + len(new_substs) >= limit
                        ):
                            ok = False
                            break
                        child_results: list[dict[str, int]] = []
                        self._match(
                            pat_arg,
                            node.children[i],
                            dict(cs),
                            child_results,
                            limit,
                        )
                        new_substs.extend(child_results)
                    if not new_substs:
                        ok = False
                        break
                    child_substs = new_substs
                if ok:
                    # child_substs already contain the incoming bindings;
                    # conflicts were rejected inside the metavar branch.
                    if limit is not None:
                        room = limit - len(results)
                        results.extend(child_substs[:room])
                        if len(results) >= limit:
                            return
                    else:
                        results.extend(child_substs)
            return

        # Leaf (Const/Param/Var) — match by key
        key = ("key", repr(pattern))
        enode = ENode("leaf", (), (key,))
        if enode in self._node_to_class:
            if self.find(self._node_to_class[enode]) == eid:
                results.append(dict(subst))

    def _nodes_of(self, eclass: EClass, op: str) -> list:
        """Member enodes of *eclass* carrying *op* — lazily indexed.

        ``eclass.by_op`` is invalidated at every ``nodes`` mutation
        (union merge, rebuild), so the bucket never goes stale.
        """
        bo = eclass.by_op
        if bo is None:
            bo = eclass.by_op = {}
        lst = bo.get(op)
        if lst is None:
            lst = [n for n in eclass.nodes if n.op == op]
            bo[op] = lst
        return lst

    # -- rebuild --

    def rebuild(self, classes: Any = None) -> bool:
        """Canonicalize children and merge duplicates.

        ``classes`` optionally restricts the pass to a set of class
        ids: only classes whose nodes' children could have changed
        canonical ids need re-canonicalizing, and those are exactly
        the classes the dirty frontier just searched (plus the ones
        dirtied while searching).  An unrestricted pass is still
        available — and used once at the end of :meth:`run` — for
        callers that mutate the graph outside the saturation loop.
        """
        changed = False
        if classes is None:
            ids: Any = list(self._classes.keys())
        else:
            ids = classes
        seen: set[int] = set()
        for eid0 in ids:
            eid = self.find(eid0)
            if eid in seen or eid not in self._classes:
                continue
            seen.add(eid)
            eclass = self._classes[eid]
            new_nodes: set[ENode] = set()
            for node in eclass.nodes:
                if node.children:
                    canon = tuple(self.find(c) for c in node.children)
                    if canon != node.children:
                        changed = True
                        # ``nn`` replaces ``node``: register the upward
                        # edge for each canonical child so later changes
                        # below propagate dirty to this class.  (``eid``
                        # is already dirty — it is an ancestor of the
                        # merge that forced the canonicalisation.)
                        for c in canon:
                            self._parents.setdefault(
                                self.find(c), set()
                            ).add(eid)
                    nn = ENode(node.op, canon, node.attrs)
                    new_nodes.add(nn)
                    if self._track and nn != node:
                        # The canonicalized enode inherits the original's
                        # provenance — same term, fresh child ids.
                        if node in self._enode_origin:
                            self._enode_origin.setdefault(
                                nn, self._enode_origin[node]
                            )
                        if node in self._enode_birth:
                            self._enode_birth.setdefault(
                                nn, self._enode_birth[node]
                            )
                        if node in self._enode_app:
                            self._enode_app.setdefault(
                                nn, self._enode_app[node]
                            )
                        self._node_to_class.setdefault(nn, eid)
                else:
                    new_nodes.add(node)
            if new_nodes != eclass.nodes:
                eclass.nodes = new_nodes
                eclass.by_op = None
        return changed

    # -- rule application --

    def _instantiate(self, pattern: Any, subst: dict[str, int]) -> int:
        """Instantiate a pattern (RHS) with a substitution.

        When proof tracking is on, ``self._inst_last_enode`` records the
        enode realising the pattern's root (post-order: the outermost
        call assigns last), so the caller can tell which enode the
        instantiated RHS is headed by — or ``None`` when the RHS is a
        bare metavariable/leaf binding.
        """
        if isinstance(pattern, str):
            self._inst_last_enode = None
            return subst[pattern]
        if isinstance(pattern, Op):
            child_eids = tuple(
                self._instantiate(a, subst) for a in pattern.args
            )
            attr_t = _pattern_attrs(pattern)
            # Attribute metavariables (string values) resolve through the
            # substitution's "$attr:" namespace.
            attrs = {}
            for k, v in attr_t:
                if isinstance(v, str):
                    attrs[k] = subst.get("$attr:" + v, v)
                else:
                    attrs[k] = v
            enode = ENode(
                pattern.op,
                tuple(self.find(c) for c in child_eids),
                tuple(sorted(attrs.items())),
            )
            if enode in self._node_to_class:
                eid = self.find(self._node_to_class[enode])
            else:
                eid = self._add_enode(enode)
            self._inst_last_enode = enode
            return eid
        else:
            # Register the concrete leaf BEFORE minting the enode: a
            # leaf key that was never seen by ``add_term`` (a Const in
            # a rule RHS, or a re-used repr name) would otherwise
            # decode to a stale cross-call term — or to the raw key
            # string, which poisons any extracted member it lands in.
            _LeafRegistry.register(pattern)
            eid = self.add_leaf(repr(pattern))
            self._inst_last_enode = ENode(
                "leaf", (), (("key", repr(pattern)),)
            )
            return eid

    def any_term(
        self,
        eid: int,
        _seen: frozenset = frozenset(),
        _memo: dict | None = None,
    ) -> Any:
        """Return any acyclic representative term of an e-class.

        Prefers leaf nodes; used to resolve metavariable bindings to
        concrete terms for rewrite side conditions (shape checks).

        ``_memo`` shares resolutions across the recursive descent —
        e-class subgraphs are heavily shared (residual streams), so
        without it the walk re-expands the same cones exponentially.
        Returns the same member as the unmemoised recursion.
        """
        if _memo is None:
            _memo = {}
        eid = self.find(eid)
        eclass = self._classes[eid]
        for node in eclass.nodes:
            if node.op == "leaf":
                key = node.attrs[0][1] if node.attrs else "??"
                return _LeafRegistry.decode(key)
        hit = _memo.get(eid)
        if hit is not None:
            return hit
        _seen = _seen | {eid}
        for node in eclass.nodes:
            args = []
            ok = True
            for c in node.children:
                canon = self.find(c)
                if canon == eid or canon in _seen:
                    ok = False
                    break
                t = self.any_term(canon, _seen, _memo)
                if t is None:
                    ok = False
                    break
                args.append(t)
            if ok:
                _memo[eid] = Op.make(node.op, *args, **dict(node.attrs))
                return _memo[eid]
        return None

    def _min_term(
        self, eid: int, memo: dict, _seen: frozenset = frozenset()
    ) -> tuple[Any, float]:
        """Smallest (fewest ops) acyclic member of an e-class, as
        ``(term, size)``.

        Used to feed rewrite side conditions: every member of the class
        is an equally valid binding, but a compact representative makes
        shape-inference checks cost O(member) instead of O(the unfolded
        class).  Returns ``(None, inf)`` when every member is cyclic
        under ``_seen``; ``None`` results are never memoised (a class
        that fails under one ``_seen`` may succeed under another).
        """
        eid = self.find(eid)
        eclass = self._classes[eid]
        for node in eclass.nodes:
            if node.op == "leaf":
                key = node.attrs[0][1] if node.attrs else "??"
                res = (_LeafRegistry.decode(key), 1)
                memo[eid] = res
                return res
        hit = memo.get(eid)
        if hit is not None:
            return hit
        if eid in _seen:
            return (None, float("inf"))
        _seen = _seen | {eid}
        best = (None, float("inf"))
        for node in eclass.nodes:
            args = []
            sz = 1
            ok = True
            for c in node.children:
                canon = self.find(c)
                if canon == eid or canon in _seen:
                    ok = False
                    break
                t, s = self._min_term(canon, memo, _seen)
                if t is None:
                    ok = False
                    break
                args.append(t)
                sz += s
            if ok and sz < best[1]:
                best = (Op.make(node.op, *args, **dict(node.attrs)), sz)
        if best[0] is not None:
            memo[eid] = best
        return best

    def _any_term_cached(self, eid: int) -> Any:
        """Member resolution for rewrite side conditions, memoised for
        the duration of one ``apply_rule``.

        Resolves to the class's *minimum-size* member rather than an
        arbitrary one: ``check``/``derive`` contracts only require *a*
        member, and a compact one keeps ``_shape_of`` (an O(term)
        recursive inference) from walking tens of thousands of nodes
        per substitution — the dominant cost of the scale-fold rules
        on stacked blocks.

        The result is also stored in ``eclass.cache`` so the SAME term
        object survives across ``apply_rule`` calls and iterations —
        the rules' content-keyed ``_shape_of`` memo then hits for checks
        re-evaluated on unchanged classes.
        """
        eid = self.find(eid)
        eclass = self._classes[eid]
        cached = eclass.cache.get("min_term")
        if cached is not None:
            return cached[0]
        memo = self._anyterm_memo
        try:
            t, s = self._min_term(eid, memo if memo is not None else {})
        except RecursionError:
            # Pathologically deep cones (unbounded runs at the node
            # cap): fall back to the early-exit walk — ``any_term``
            # returns the first acyclic member it finds rather than
            # scanning every member for the minimum.  If even that
            # exceeds the recursion limit, degrade to ``None`` — the
            # caller treats it as "no representative", skipping the
            # substitution (conservative: never a wrong rewrite).
            try:
                return self.any_term(eid)
            except RecursionError:
                return None
        if t is not None:
            eclass.cache["min_term"] = (t, s)
        return t

    def _candidate_classes(
        self, lhs: Any, search: set | list | None
    ) -> Any:
        """E-classes a rule's LHS could possibly match at.

        ``search=None`` scans the whole graph (the pre-incremental
        behaviour); a set restricts to the dirty frontier.  In both
        cases an Op-rooted pattern additionally filters to classes
        that contain a member with the pattern's head op, a
        metavariable pattern binds at any class, and a concrete-leaf
        pattern can only match at the leaf's own class.
        """
        if isinstance(lhs, Op):
            eligible = {
                self.find(c) for c in self._op_classes.get(lhs.op, ())
            }
        elif isinstance(lhs, str):
            eligible = None
        else:
            leaf = ENode("leaf", (), (("key", repr(lhs)),))
            leid = self._node_to_class.get(leaf)
            eligible = {self.find(leid)} if leid is not None else set()
        # ``_classes`` mutates under us as unions fire — snapshot.
        ids = list(self._classes.keys()) if search is None else search
        for eid0 in ids:
            eid = self.find(eid0)
            if eligible is not None and eid not in eligible:
                continue
            yield eid

    def apply_rule(
        self,
        rule: Rewrite,
        root_eid: int,
        search: set | list | None = None,
        enode_budget: int | None = None,
    ) -> bool:
        """Apply a single rewrite rule across e-classes.

        ``search`` limits the scan to the given class ids (the dirty
        frontier supplied by :meth:`run`); the default searches every
        class — the only sound choice for a rule that has never seen
        this graph.

        ``enode_budget`` bounds how many NEW enodes this call may
        create: the scan stops between candidate classes once the
        budget is spent, and match enumeration inside each class is
        capped at the smaller of the remaining budget and
        ``_MATCH_CAP`` so a single giant e-class cannot spend it all
        in one enumeration.  This is the mechanism behind
        :meth:`run`'s ``rule_budgets`` — a bounded-saturation policy
        for rules whose closure is combinatorially explosive.
        """
        if search is None:
            # A whole-graph scan establishes the rule's frontier;
            # a restricted scan does not.
            self._applied_rules.add(rule.name)
        changed = False
        n_start = self.n_enodes
        self._anyterm_memo = {}
        try:
            candidates = self._candidate_classes(rule.lhs, search)
            for eid in candidates:
                match_cap = None
                if enode_budget is not None:
                    spent = self.n_enodes - n_start
                    if spent >= enode_budget:
                        break
                    match_cap = min(
                        enode_budget - spent, self._MATCH_CAP
                    )
                for subst in self.matches(
                    rule.lhs, eid, max_results=match_cap
                ):
                    if (
                        rule.check is not None
                        or rule.derive is not None
                    ):
                        bound = {
                            k: (
                                v
                                if k.startswith("$attr:")
                                else self._any_term_cached(v)
                            )
                            for k, v in subst.items()
                        }
                        if any(
                            v is None
                            for k, v in bound.items()
                            if not k.startswith("$attr:")
                        ):
                            continue
                        if rule.check is not None and not rule.check(
                            bound
                        ):
                            continue
                        if rule.derive is not None:
                            extra = rule.derive(bound)
                            if extra is None:
                                continue
                            subst = {**subst, **extra}
                    created = None
                    if self._track:
                        self._tag_rule = rule.name
                        self._collect = []
                        self._inst_last_enode = None
                        try:
                            rhs_eid = self._instantiate(rule.rhs, subst)
                        finally:
                            self._tag_rule = None
                            created = self._collect
                            self._collect = None
                    else:
                        rhs_eid = self._instantiate(rule.rhs, subst)
                    self._rule_objs.setdefault(rule.name, rule)
                    merged = self.union(
                        eid, rhs_eid, rule=rule.name, subst=subst
                    )
                    if self._track and (merged or created):
                        # A real merge or freshly-instantiated enodes —
                        # the application is a witness worth keeping.  A
                        # no-op firing (RHS already in the class,
                        # nothing created) adds no 2-morphism, so it is
                        # not recorded.
                        app_idx = len(self._applications)
                        self._applications.append(
                            {
                                "rule": rule.name,
                                "matched_eid": self.find(eid),
                                "rhs_eid": self.find(rhs_eid),
                                "subst": dict(subst),
                                "rhs_root_enode": self._inst_last_enode,
                            }
                        )
                        # First discovered witness wins — a hash-consed
                        # enode keeps the application that created it.
                        for en in created or ():
                            self._enode_app.setdefault(en, app_idx)
                    if merged:
                        changed = True
                        self.rule_fires[rule.name] = (
                            self.rule_fires.get(rule.name, 0) + 1
                        )
        finally:
            self._anyterm_memo = None
        return changed

    # -- saturation --

    def run(
        self,
        rules: list[Rewrite],
        root_eid: int,
        max_iterations: int = 100,
        max_nodes: int = 100_000,
        rule_budgets: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """Run equality saturation until a fixed point.

        Incremental: each iteration re-searches only the *dirty
        frontier* — e-classes whose reachable subgraph changed since
        they were last searched (fresh enodes, merged classes, and
        every ancestor of those, maintained by :meth:`_add_enode` and
        :meth:`union`).  A class outside the frontier cannot produce a
        match that was not already produced, so skipping it preserves
        the fixed point exactly while removing the per-iteration
        whole-graph rescan that made saturation quadratic.

        A rule whose name has never been applied in this e-graph
        searches every class once (it has no established frontier);
        subsequent iterations of the same rule set are frontier-only.

        ``rule_budgets`` maps rule names to a maximum number of NEW
        enodes the rule may contribute over the whole run; once a
        rule exhausts its budget it is suspended.  This is *bounded
        saturation*: unbudgeted rules still reach the exact fixed
        point, while combinatorially explosive rules (pure symmetry
        generators such as comm/assoc, whose closure enumerates every
        bracketing and ordering of a summation) are truncated at a
        point that — measured on the transformer pipeline — already
        covers every rewrite the extractor can exploit.  Budget
        accounting is reported in ``stats["rule_budgets"]``.
        """
        budgets = rule_budgets or {}
        spent = self._budget_spent
        for name in budgets:
            spent.setdefault(name, 0)
        iteration = -1
        for iteration in range(max_iterations):
            n_before = self.n_enodes
            # Snapshot the frontier ONCE per iteration: classes changed
            # while rules fire are re-searched next iteration — the
            # fixed point is unchanged, the schedule is tighter.
            search = sorted({self.find(e) for e in self._dirty})
            self._dirty.clear()
            for rule in rules:
                budget = budgets.get(rule.name)
                if budget is not None:
                    remaining = budget - spent[rule.name]
                    if remaining <= 0:
                        continue  # suspended: budget exhausted
                else:
                    remaining = None
                n0 = self.n_enodes
                if rule.name in self._applied_rules:
                    self.apply_rule(
                        rule,
                        root_eid,
                        search=search,
                        enode_budget=remaining,
                    )
                else:
                    # Never scanned this graph: full scan once —
                    # afterwards the dirty frontier suffices.
                    self.apply_rule(
                        rule, root_eid, enode_budget=remaining
                    )
                if budget is not None:
                    spent[rule.name] += self.n_enodes - n0
            self.rebuild(set(search) | self._dirty)
            n_after = self.n_enodes
            if n_after >= max_nodes:
                print(
                    f"  [egraph] stopping: max_nodes ({max_nodes}) reached"
                )
                break
            if n_after == n_before and not self._dirty:
                print(
                    f"  [egraph] saturation at iteration {iteration + 1}"
                )
                break
        # Leave the graph in the same canonicalised postcondition the
        # unrestricted loop guaranteed (downstream passes and
        # extraction traverse it directly).
        self.rebuild()
        return {
            "iterations": iteration + 1,
            "n_enodes": self.n_enodes,
            "n_classes": self.n_classes,
            "n_proof_edges": len(self._merge_log),
            "truncation_level": self.truncation_level,
            "rule_budgets": {n: spent[n] for n in budgets},
            "budget_suspended": [
                n for n in budgets if spent[n] >= budgets[n]
            ],
        }

    # -- extraction --

    def extract_min_depth(self, eid: int) -> Any:
        """Extract the minimum critical-path-depth member.

        Depth is not additive (``local = cost(term) − Σ cost(children)``
        is meaningless for a max-composed measure), so ``extract_best``
        cannot serve it.  But depth IS decomposable per class —
        ``1 + max(child depths)`` — which this greedy walk computes
        directly.  Deterministic: members are scanned in a canonical
        sorted order so ties resolve identically every run.
        """

        cache: dict[int, tuple[float, Any]] = {}
        in_prog: set[int] = set()

        def go(cid: int) -> tuple[float, Any]:
            cid = self.find(cid)
            if cid in cache:
                return cache[cid]
            if cid in in_prog:
                return (float("inf"), None)
            in_prog.add(cid)
            best = (float("inf"), None)
            for node in sorted(
                self._classes[cid].nodes,
                key=lambda n: (n.op, n.children, repr(n.attrs)),
            ):
                if node.op == "leaf":
                    key = node.attrs[0][1] if node.attrs else "??"
                    cand = (0, _LeafRegistry.decode(key))
                else:
                    kids, dmax, ok = [], 0, True
                    for c in node.children:
                        cc = self.find(c)
                        if cc == cid:
                            ok = False
                            break
                        d, t = go(cc)
                        if t is None:
                            ok = False
                            break
                        kids.append(t)
                        dmax = max(dmax, d)
                    if not ok:
                        continue
                    cand = (
                        1 + dmax,
                        Op.make(node.op, *kids, **dict(node.attrs)),
                    )
                if cand[0] < best[0]:
                    best = cand
            in_prog.discard(cid)
            cache[cid] = best
            return best

        return go(eid)[1]

    def extract_alternatives(
        self, eid: int, cost_fn, top_k: int = 8
    ) -> list[tuple[float, Any]]:
        """Enumerate the root e-class frontier: for each non-leaf enode,
        force extraction through it and record the resulting term's DAG
        cost.  Returns the top-k cheapest *distinct* alternatives —
        i.e. the cheapest members of the semantic equivalence class
        [G], which is what a discovery engine inspects for unexpected
        candidates."""
        from catopt.cost import dag_cost
        from catopt.ir import op_repr

        eid = self.find(eid)
        eclass = self._classes[eid]
        seen: dict[str, tuple[float, Any]] = {}
        for node in eclass.nodes:
            if node.op == "leaf":
                continue
            term = self.extract_best(
                eid, cost_fn, overrides={eid: node}
            )
            if term is None:
                continue
            key = op_repr(term)
            cost = dag_cost(term, cost_fn)
            if key not in seen or cost < seen[key][0]:
                seen[key] = (cost, term)
        ranked = sorted(seen.values(), key=lambda kv: kv[0])
        return ranked[:top_k]

    def diverse_classes(self) -> list[dict]:
        """E-classes containing structurally distinct equivalent terms.

        These are where emergent compositions hide: a class holding both
        `matmul(softmax(mf))` and `sdpa` means the search found that
        two very different programs compute the same thing.  Returns
        classes with >= 2 distinct member ops, each with a one-line
        sketch of every distinct member."""

        out = []
        for eid, ec in self._classes.items():
            ops = {n.op for n in ec.nodes if n.op != "leaf"}
            if len(ops) < 2:
                continue
            sketches = []
            seen_sketch = set()
            for n in ec.nodes:
                if n.op == "leaf":
                    key = n.attrs[0][1] if n.attrs else "?"
                    sk = key.split(":")[-1][:24]
                else:
                    child_ops = [
                        next(iter(self._classes[self.find(c)].nodes)).op
                        if self._classes[self.find(c)].nodes
                        else "?"
                        for c in n.children
                    ]
                    sk = f"{n.op}({','.join(child_ops)})"
                if sk not in seen_sketch:
                    seen_sketch.add(sk)
                    sketches.append(sk)
            out.append({"eid": eid, "members": sketches})
        out.sort(key=lambda d: -len(d["members"]))
        return out

    def extract_best(
        self,
        eid: int,
        cost_fn,
        overrides: dict[int, Any] | None = None,
        bans: dict[int, set] | None = None,
        _cache_out: dict | None = None,
    ) -> Any:
        """Extract the minimum-cost term from the e-class at *eid*.

        ``overrides`` maps canonical e-class ids to a specific ENode:
        extraction is then forced to use that enode for those classes.
        This is how non-local rewrites (the diagram-level product rule)
        get *coordinated* extraction — per-class greedy choice cannot see
        that k members each selecting `split_i(fused)` share ONE fused
        GEMM, since each split's subtree alone looks more expensive than
        the member's own linear.

        ``bans`` is the dual: canonical e-class id -> set of ENodes the
        extraction must skip — the mechanism behind
        :meth:`extract_best_bounded`, which bans the enodes that
        bound-carrying certificate steps produced.

        Cost accounting is DAG-aware: each e-class in the extracted
        expression is charged exactly once, even when several parents
        share it (e.g. one fused GEMM feeding two chunk projections).
        A node's *local* cost is recovered as ``cost_fn(term) - sum of
        cost_fn(children)`` — exact for additive cost functions such as
        ``flops_cost`` and ``count_cost``.

        Classes whose extracted subtree contains no data input (only
        ``Param``/``Const`` leaves) are compile-time work: lowering
        folds them into a materialised parameter, so they are charged
        at zero.  This is what lets the extractor prefer rewrites that
        move computation onto the weights (e.g. ``x@(W1@W2@W3)``) even
        though the weight product itself is not free in FLOP terms.
        Cost models that price parameter *storage*
        (``catopt.cost.param_bytes_cost``) opt out of that discount by
        setting ``charges_param_only`` on the function — a folded
        subtree still stores its leaves' values.

        Cyclic nodes (a class reachable from itself through rewrite-
        introduced unions) are skipped: they cannot be extracted.
        """
        # eid -> (total_cost, term, used_eclass_ids, subtree_is_param_only, nops)
        cache: dict[int, tuple[float, Any, frozenset, bool, int]] = {}
        local_of: dict[int, float] = {}
        param_only_of: dict[int, bool] = {}
        in_progress: set[int] = set()

        # Shared cost/shape memo for the whole extraction: makes the
        # per-candidate cost_fn calls O(1) amortised over the DAG.
        # Terms are content-hashed and interned — memos key on the term
        # object directly and hold it alive; no keepalive needed.
        import inspect

        cost_memo: dict = {}
        takes_memo = "memo" in inspect.signature(cost_fn).parameters
        # Storage-style cost models (param_bytes_cost) bill Param leaves
        # — folding does not shrink the weights file — so the param-only
        # discount does not apply.  getattr(..., "func", ...) unwraps
        # functools.partial bindings of the flagged function.
        bill_params = getattr(
            getattr(cost_fn, "func", cost_fn),
            "charges_param_only",
            False,
        )

        def cfn(t: Any) -> float:
            return (
                cost_fn(t, memo=cost_memo) if takes_memo else cost_fn(t)
            )

        def best(
            eclass_id: int,
        ) -> tuple[float, Any, frozenset, bool, int]:
            eclass_id = self.find(eclass_id)
            if eclass_id in cache:
                return cache[eclass_id]
            if eclass_id in in_progress:
                # Cycle back to an ancestor — not extractable.
                return (float("inf"), None, frozenset(), False, 0)
            in_progress.add(eclass_id)
            eclass = self._classes[eclass_id]
            override = overrides.get(eclass_id) if overrides else None
            # Deterministic member order: eclass.nodes is a set, and
            # equal-cost/equal-size ties otherwise resolve by hash
            # order — the extraction result (and any test asserting a
            # particular extracted shape) must not depend on it.
            nodes = (
                (override,)
                if override is not None
                else sorted(
                    eclass.nodes,
                    key=lambda n: (n.op, n.children, repr(n.attrs)),
                )
            )
            banned = bans.get(eclass_id) if bans else None
            best_total: float | None = None
            best_term: Any = None
            best_used: frozenset = frozenset({eclass_id})
            best_local = 0.0
            best_param_only = False
            best_nops = 0
            for node in nodes:
                if banned and node in banned:
                    continue  # excluded enode (extract_best_bounded)
                if node.op == "leaf":
                    key = node.attrs[0][1] if node.attrs else "??"
                    term = _LeafRegistry.decode(key)
                    total = cfn(term)
                    if best_total is None or total < best_total:
                        from catopt.ir import Var as _Var

                        best_total = total
                        best_term = term
                        best_used = frozenset({eclass_id})
                        best_local = total
                        best_param_only = not isinstance(term, _Var)
                        best_nops = 0
                    continue

                child_terms: list[Any] = []
                child_nops = 0
                used: set[int] = {eclass_id}
                sub_cost = 0.0
                param_only = True
                valid = True
                for child_eid in node.children:
                    canon_child = self.find(child_eid)
                    if canon_child == eclass_id:
                        valid = False  # direct self-reference
                        break
                    ctotal, cterm, cused, cpo, cnops = best(canon_child)
                    if cterm is None:
                        valid = False
                        break
                    child_terms.append(cterm)
                    child_nops += cnops
                    param_only = param_only and cpo
                    # Charge each distinct e-class in the DAG once:
                    # a shared child contributes its subtree cost only
                    # for the classes not already accounted for.
                    # Compile-time (param-only) classes are free —
                    # unless the cost model prices storage, in which
                    # case every class's local is billed.
                    for u in cused - used:
                        if bill_params or not param_only_of.get(
                            u, False
                        ):
                            sub_cost += local_of.get(u, 0.0)
                    used |= cused
                if not valid:
                    continue
                term = Op.make(
                    node.op, *child_terms, **dict(node.attrs)
                )
                local = cfn(term) - sum(cfn(c) for c in child_terms)
                local = max(local, 0.0)
                if param_only and not bill_params:
                    local = 0.0  # whole subtree folds at compile time
                total = local + sub_cost
                # Secondary key: among equal-cost candidates prefer the
                # structurally smallest term (a leaf over add(W, 0) in a
                # param-only class, for example).  Counted incrementally
                # from child caches — no tree walk.
                nops = child_nops + 1
                if (
                    best_total is None
                    or total < best_total
                    or (total == best_total and nops < best_nops)
                ):
                    best_total = total
                    best_term = term
                    best_used = frozenset(used)
                    best_local = local
                    best_param_only = param_only
                    best_nops = nops
            in_progress.discard(eclass_id)
            if best_term is None:
                best_total = float("inf")
            local_of[eclass_id] = best_local
            param_only_of[eclass_id] = best_param_only
            cache[eclass_id] = (
                best_total or 0.0,
                best_term,
                best_used,
                best_param_only,
                best_nops,
            )
            return cache[eclass_id]

        _, term, _, _, _ = best(eid)
        if _cache_out is not None:
            _cache_out.update(cache)
        return term

    def extract_best_bounded(
        self,
        eid: int,
        cost_fn,
        max_error: float | None = None,
        *,
        src_term: Any = None,
        _cache_out: dict | None = None,
    ) -> Any:
        """Extract the minimum-cost member whose derivation certifies
        within ``max_error``.

        A member's ε lives on its *derivation*, not on the member
        itself — the certificate is what knows which bound-carrying
        rewrites (``Rewrite.error_bound``) produced it.  So this uses
        the pragmatic extract-then-filter design: the bound is looked
        up per extracted candidate through :meth:`certificate` (which
        aggregates the per-step bounds, triangle-inequality style),
        and extraction iterates:

        1. extract the minimum-cost member under ``cost_fn``;
        2. build its certificate from ``src_term`` (default: the
           class's oldest member — normally the term originally added);
        3. ``cert.error_bound <= max_error`` -> return it;
        4. otherwise *ban* every enode a bound-carrying step produced
           (located via ``_locate(step.rhs)``) and repeat.  Bans
           accumulate, each round removes at least one enode, and the
           loop is additionally capped at ``n_enodes`` rounds.

        ``max_error=None`` is unconstrained extraction.  ``None`` is
        returned when no member satisfies the budget — including when
        an offending member's enode cannot be located to ban.

        Caveats: at ``truncation_level == 1`` no proof witnesses were
        recorded, so every member certifies at bound 0 and the
        constraint is vacuous; and members introduced by *unwitnessed*
        unions (``egraph_dependent`` steps) carry no rule and so
        contribute no bound — the filter trusts only certified bounds,
        matching ``Certificate.error_bound``.
        """
        if max_error is None:
            return self.extract_best(
                eid, cost_fn, _cache_out=_cache_out
            )
        root = self.find(eid)
        if src_term is None:
            src_term = self._oldest_term(root)
            if src_term is None:
                src_term = self.any_term(root)
        bans: dict[int, set] = {}
        for _ in range(self.n_enodes + 1):
            term = self.extract_best(
                root, cost_fn, bans=bans, _cache_out=_cache_out
            )
            if term is None:
                return None
            cert = self.certificate(src_term, term, root_eid=root)
            if cert.error_bound <= max_error:
                return term
            # Ban the member each bound-carrying step produced; the
            # next round re-extracts through exact members only (or
            # members under a smaller accumulated ε).
            progress = False
            for step in cert.steps:
                rule = cert.rules.get(step.rule)
                if rule is None or not rule.error_bound:
                    continue
                ceid, en = self._locate(step.rhs)
                if en is None:
                    continue
                ceid = self.find(ceid)
                if en not in bans.setdefault(ceid, set()):
                    bans[ceid].add(en)
                    progress = True
            if not progress:
                return None  # bounded member we cannot locate/exclude
        return None

    # -- coordinated (group) extraction ----------------------------------

    def extract_paired(
        self, root_eid: int, cost_fn, groups: list[dict[int, Any]]
    ) -> Any:
        """Extract with pairing groups forced to share their fused GEMM.

        Per-class greedy extraction cannot express the product law's
        non-local choice: member class C_i containing both
        ``linear(x, W_i)`` and ``split_i(fused)`` sees the split's
        subtree cost as the FULL fused GEMM, which always loses locally.
        The fused form only wins when *all* members take it — and
        additionally when every consumer class routes through the member
        classes rather than a specialized-fusion alternative (e.g. an
        ``sdpa`` enode built over rule-introduced ``chunk`` terms).

        So: (1) each member class is overridden to its split enode;
        (2) every other class with multiple enodes is overridden to an
        enode whose descendants reach a member class, when one exists —
        steering consumers through the shared GEMM.  The caller compares
        true DAG cost against the greedy term and keeps the winner.
        """
        member_over: dict[int, Any] = {}
        member_classes: set[int] = set()
        for g in groups:
            for cid, enode in g.items():
                cid = self.find(cid)
                member_over.setdefault(cid, enode)
                member_classes.add(cid)

        # descendant e-class sets, memoized, cycle-guarded
        desc_cache: dict[int, frozenset] = {}

        def desc(cid: int, stack: frozenset) -> frozenset:
            cid = self.find(cid)
            if cid in desc_cache:
                return desc_cache[cid]
            if cid in stack:
                return frozenset({cid})
            out: set[int] = {cid}
            for n in self._classes[cid].nodes:
                for ch in n.children:
                    out |= desc(ch, stack | {cid})
            desc_cache[cid] = frozenset(out)
            return desc_cache[cid]

        def enode_reaches_member(node: Any) -> bool:
            for ch in node.children:
                if member_classes & desc(ch, frozenset()):
                    return True
            return False

        # First pass under member overrides gives a cost table used to
        # score steering candidates: picking member_reaching[0] can grab
        # an arbitrarily expensive alternative (e.g. a distributed form)
        # and inflate the forced term's true DAG cost.
        import inspect

        cost_memo: dict = {}
        takes_memo = "memo" in inspect.signature(cost_fn).parameters

        def cfn(t: Any) -> float:
            return (
                cost_fn(t, memo=cost_memo) if takes_memo else cost_fn(t)
            )

        pass1_cache: dict = {}
        self.extract_best(
            root_eid,
            cost_fn,
            overrides=member_over,
            _cache_out=pass1_cache,
        )

        def steered_score(node: Any) -> float:
            """local cost + children best totals (member-routed pass)."""
            child_terms = []
            sub = 0.0
            for ch in node.children:
                entry = pass1_cache.get(self.find(ch))
                if entry is None or entry[1] is None:
                    return float("inf")
                child_terms.append(entry[1])
                sub += entry[0]
            term = Op.make(node.op, *child_terms, **dict(node.attrs))
            local = cfn(term) - sum(cfn(c) for c in child_terms)
            return max(local, 0.0) + sub

        overrides: dict[int, Any] = dict(member_over)
        for cid, ec in list(self._classes.items()):
            cid = self.find(cid)
            if cid in member_over or len(ec.nodes) < 2:
                continue
            member_reaching = [
                n for n in ec.nodes if enode_reaches_member(n)
            ]
            if member_reaching and len(member_reaching) < len(ec.nodes):
                # class has both member-reaching and bypassing enodes —
                # force the cheapest member route so the shared GEMM
                # is used without dragging in junk subtrees
                route = min(member_reaching, key=steered_score)
                if steered_score(route) != float("inf"):
                    overrides[cid] = route

        return self.extract_best(root_eid, cost_fn, overrides=overrides)

    # -- certificates: derivations reconstructed from proof data ----------

    _CERT_MAX_DEPTH = 400
    _CERT_MAX_STEPS = 20_000

    def _class_of_term(
        self, term: Any, _memo: dict | None = None
    ) -> Any:
        """Canonical e-class id realising *term*, without mutating the graph.

        ``_memo`` is content-keyed on the term (interned Op objects
        hash by structure): ``any_term``/``_min_term`` resolutions are
        shared-DAG objects, and without the memo the recursion re-walks
        shared subtrees exponentially (observed: ~24M calls for one
        pairing witness on a 5-block stack).
        """
        if _memo is None:
            _memo = {}
        hit = _memo.get(term, False)
        if hit is not False:
            return hit
        if isinstance(term, Op):
            cids = []
            for a in term.args:
                c = self._class_of_term(a, _memo)
                if c is None:
                    _memo[term] = None
                    return None
                cids.append(c)
            en = ENode(term.op, tuple(cids), _pattern_attrs(term))
        else:
            en = ENode("leaf", (), (("key", repr(term)),))
        eid = self._node_to_class.get(en)
        res = self.find(eid) if eid is not None else None
        _memo[term] = res
        return res

    def _locate(self, term: Any, eid: int | None = None):
        """``(eid, enode)`` realising *term* inside its e-class.

        The enode is matched by structure — op, attrs, and per-child
        e-classes — so it pins down exactly which member of the class
        the term uses.  ``enode`` is ``None`` when no member matches.
        """
        memo: dict = {}
        if eid is None:
            eid = self._class_of_term(term, memo)
        if eid is None:
            return None, None
        eid = self.find(eid)
        ec = self._classes.get(eid)
        if ec is None:
            return eid, None
        if not isinstance(term, Op):
            for n in ec.nodes:
                if (
                    n.op == "leaf"
                    and n.attrs
                    and n.attrs[0][1] == repr(term)
                ):
                    return eid, n
            return eid, None
        child_cls = [self._class_of_term(a, memo) for a in term.args]
        for n in ec.nodes:
            if n.op != term.op or len(n.children) != len(term.args):
                continue
            if dict(n.attrs) != term.attrs:
                continue
            if all(
                cc is not None and self.find(c) == cc
                for c, cc in zip(n.children, child_cls)
            ):
                return eid, n
        return eid, None

    def _birth(self, enode: ENode) -> int:
        """Creation order of an enode (its birth eid); huge if unknown."""
        b = self._enode_birth.get(enode)
        if b is not None:
            return b
        eid = self._node_to_class.get(enode)
        return eid if eid is not None else 1 << 60

    def _oldest_term(
        self,
        eid: int,
        _stack: frozenset = frozenset(),
        _memo: dict | None = None,
    ):
        """The earliest-created representative term of an e-class.

        Proof-time analogue of :meth:`any_term`: picking the minimum-
        birth enode at every level makes target-side expansion in
        :meth:`_connect` a strictly descending recursion — every rule
        instance's LHS was matched on enodes older than the RHS enodes
        it created.

        ``_memo`` shares resolutions across the descent (e-class cones
        are heavily shared); a memoised member is still a valid
        member, and ``None`` results are never cached (a class blocked
        by ``_stack`` may resolve under another ancestry).
        """
        if _memo is None:
            _memo = {}
        eid = self.find(eid)
        if eid in _stack:
            return None
        hit = _memo.get(eid)
        if hit is not None:
            return hit
        ec = self._classes.get(eid)
        if ec is None:
            return None
        _stack = _stack | {eid}
        for node in sorted(ec.nodes, key=self._birth):
            if node.op == "leaf":
                key = node.attrs[0][1] if node.attrs else "??"
                _memo[eid] = _LeafRegistry.decode(key)
                return _memo[eid]
            args = []
            ok = True
            for c in node.children:
                t = self._oldest_term(c, _stack, _memo)
                if t is None:
                    ok = False
                    break
                args.append(t)
            if ok:
                _memo[eid] = Op.make(node.op, *args, **dict(node.attrs))
                return _memo[eid]
        return None

    def _app_for_member(self, term: Any):
        """The rule application whose RHS root realises *term*'s root.

        Returns ``(app, enode)`` — ``app`` is ``None`` when the member
        enode is not the RHS root of any recorded application (input
        term, internal RHS node, or pass-introduced).
        """
        eid, en = self._locate(term)
        if en is None:
            return None, None
        ai = self._enode_app.get(en)
        if ai is None:
            return None, en
        app = self._applications[ai]
        root_en = app["rhs_root_enode"]
        if root_en is None:
            return None, en
        # ``en`` must be the application's RHS root — children compare
        # modulo canonicalisation (rebuild may have rewritten ids).
        if (
            root_en.op != en.op
            or root_en.attrs != en.attrs
            or len(root_en.children) != len(en.children)
        ):
            return None, en
        if not all(
            self.find(a) == self.find(b)
            for a, b in zip(root_en.children, en.children)
        ):
            return None, en
        return app, en

    def _resolve_subst(self, subst: dict) -> dict | None:
        """Resolve a fired binding to concrete terms.

        Metavariable eids map to their class's :meth:`_oldest_term`;
        ``"$attr:"`` keys carry their concrete value through.  ``None``
        when a bound class is unresolvable (cyclic).
        """
        out = {}
        for k, v in subst.items():
            if k.startswith("$attr:"):
                out[k] = v
            else:
                t = self._oldest_term(v)
                if t is None:
                    return None
                out[k] = t
        return out

    def _connect(
        self, s: Any, t: Any, pos: tuple, steps: list, depth: int
    ) -> bool:
        """Emit steps rewriting the subterm at ``pos`` from ``s`` to ``t``.

        Both terms are members of one e-class.  Returns True when every
        emitted step is a standalone-replayable rule application; False
        means at least one ``egraph_dependent`` stub was emitted.  The
        strategy, in order:

        1. **Expand the target** through the rule application that
           created its root enode: recursively bridge ``s`` to the
           application's LHS instance, emit the rewrite, then fix the
           produced children pairwise.
        2. **Congruence**: same head op/attrs and pairwise-equivalent
           children — descend without emitting a step.
        3. **Edge search**: a short bounded search over the rules that
           fired, covering merges whose RHS is a bare metavariable or a
           pre-existing hash-consed enode (no expandable enode exists).
        4. **e-graph-dependent stub**: the merge is real but has no
           standalone derivation (non-local pass, manual union).
        """
        if op_repr(s) == op_repr(t):
            return True
        if (
            depth > self._CERT_MAX_DEPTH
            or len(steps) > self._CERT_MAX_STEPS
        ):
            steps.append(
                CertStep(
                    "<budget>",
                    pos,
                    s,
                    t,
                    {},
                    egraph_dependent=True,
                    note="derivation budget exceeded",
                )
            )
            return False

        app, _en = self._app_for_member(t)
        if app is not None:
            rule = self._rule_objs.get(app["rule"])
            bound = (
                self._resolve_subst(app["subst"])
                if rule is not None
                else None
            )
            if bound is not None:
                L = _term_instantiate(rule.lhs, bound)
                R = _term_instantiate(rule.rhs, bound)
                ok = self._connect(s, L, pos, steps, depth + 1)
                steps.append(CertStep(rule.name, pos, L, R, bound))
                if isinstance(R, Op) and isinstance(t, Op):
                    for i in range(len(t.args)):
                        if not self._connect(
                            R.args[i],
                            t.args[i],
                            pos + (i,),
                            steps,
                            depth + 1,
                        ):
                            ok = False
                return ok

        if (
            isinstance(s, Op)
            and isinstance(t, Op)
            and s.op == t.op
            and s.attrs == t.attrs
            and len(s.args) == len(t.args)
        ):
            cs = [self._class_of_term(a) for a in s.args]
            ct = [self._class_of_term(a) for a in t.args]
            if all(a is not None and a == b for a, b in zip(cs, ct)):
                ok = True
                for i in range(len(s.args)):
                    if not self._connect(
                        s.args[i],
                        t.args[i],
                        pos + (i,),
                        steps,
                        depth + 1,
                    ):
                        ok = False
                return ok

        path = self._edge_path(s, t, pos)
        if path is not None:
            steps.extend(path)
            return True

        steps.append(
            CertStep(
                "<egraph>",
                pos,
                s,
                t,
                {},
                egraph_dependent=True,
                note=self._explain_gap(s, t),
            )
        )
        return False

    def _edge_path(
        self,
        s: Any,
        t: Any,
        pos: tuple,
        depth: int = 0,
        _seen: set | None = None,
        _budget: list | None = None,
    ) -> list | None:
        """Bounded search for a replayable step sequence ``s -> t``.

        Transitions are the rules that actually fired during this run,
        re-matched directly against the real term ``s`` — each emitted
        step is a genuine rule instance, so any path found is a valid
        derivation.  Returns the list of steps (all located at ``pos``),
        or ``None`` when no short path exists.
        """
        if op_repr(s) == op_repr(t):
            return []
        if _seen is None:
            _seen, _budget = set(), [512]
        if depth > 16 or _budget[0] <= 0:
            return None
        _seen.add(op_repr(s))
        for rule in self._rule_objs.values():
            m = _term_match(rule.lhs, s)
            if m is None:
                continue
            if rule.check is not None and not rule.check(m):
                continue
            inst = dict(m)
            if rule.derive is not None:
                extra = rule.derive(m)
                if extra is None:
                    continue
                inst.update(extra)
            r = _term_instantiate(rule.rhs, inst)
            if op_repr(r) in _seen:
                continue
            _budget[0] -= 1
            rest = self._edge_path(r, t, pos, depth + 1, _seen, _budget)
            if rest is not None:
                return [CertStep(rule.name, pos, s, r, inst)] + rest
        return None

    def _explain_gap(self, s: Any, t: Any) -> str:
        """Why a ``connect`` gap is e-graph-dependent, for the cert note."""
        _eid, en_t = self._locate(t)
        if en_t is not None:
            org = self._enode_origin.get(en_t)
            if org == "external":
                return (
                    "target enode introduced outside rule "
                    "application (non-local pass such as "
                    "pair_shared_input_*, or a manual union)"
                )
            if org == "input":
                return (
                    "both members predate saturation but no fired "
                    "rule links them at this position"
                )
        return "no replayable derivation found"

    def _resolve_dst(
        self,
        src_term: Any,
        dst_term: Any,
        root_eid: int | None,
        cost_fn,
    ) -> tuple[int | None, Any]:
        """Shared endpoint resolution for certificates and coherence."""
        if root_eid is None:
            root_eid = self._class_of_term(src_term)
        if dst_term is None:
            if root_eid is None:
                raise ValueError("src_term is not in this e-graph")
            if cost_fn is None:
                from catopt.cost import count_cost

                cost_fn = count_cost
            dst_term = self.extract_best(root_eid, cost_fn)
        return root_eid, dst_term

    def certificate(
        self,
        src_term: Any,
        dst_term: Any = None,
        *,
        root_eid: int | None = None,
        cost_fn=None,
    ) -> Certificate:
        """Build a proof-carrying derivation ``src_term`` -> ``dst_term``.

        ``src_term`` is the term originally added to the e-graph (the
        certificate's anchor — *not* an arbitrary class member).
        ``dst_term`` defaults to ``extract_best`` under ``cost_fn``
        (``count_cost`` if neither is given).  The certificate's steps
        replay positionally on real terms; ``egraph_dependent`` steps
        mark where the e-graph witnessed an equality that has no
        standalone rule derivation.

        At ``truncation_level == 1`` no proof witnesses were recorded,
        so the certificate degrades to a proof-free marker: a single
        ``egraph_dependent`` step asserting ``src == dst`` (or zero
        steps when the terms are already identical).  It replays under
        :func:`verify_certificate` as a trusted assertion and is
        rejected under ``strict=True``.
        """
        root_eid, dst_term = self._resolve_dst(
            src_term, dst_term, root_eid, cost_fn
        )
        if not self._track:
            if op_repr(src_term) == op_repr(dst_term):
                steps0: list = []
            else:
                steps0 = [
                    CertStep(
                        "<truncated>",
                        (),
                        src_term,
                        dst_term,
                        {},
                        egraph_dependent=True,
                        note="truncation level 1: proof witnesses "
                        "were not recorded",
                    )
                ]
            return Certificate(
                src=src_term,
                dst=dst_term,
                root_eid=root_eid,
                steps=steps0,
                rules={},
                stats={
                    "proof_free": True,
                    "truncation_level": self.truncation_level,
                    "n_steps": len(steps0),
                    "n_egraph_dependent": len(steps0),
                    "rules_used": [],
                    "n_proof_edges": 0,
                    "n_rule_applications": 0,
                },
            )
        steps: list = []
        self._connect(src_term, dst_term, (), steps, 0)
        used = sorted({s.rule for s in steps if not s.egraph_dependent})
        cert = Certificate(
            src=src_term,
            dst=dst_term,
            root_eid=root_eid,
            steps=steps,
            rules={
                n: self._rule_objs[n]
                for n in used
                if n in self._rule_objs
            },
            stats={
                "n_steps": len(steps),
                "n_egraph_dependent": sum(
                    1 for s in steps if s.egraph_dependent
                ),
                "rules_used": used,
                "n_proof_edges": len(self._merge_log),
                "n_rule_applications": len(self._applications),
            },
        )
        return cert

    # -- level 3: lazily-materialised coherences --------------------------
    #
    #  The level-2 merge log is a *forest*: ``union`` only records an
    #  edge between two previously-disconnected classes, so exactly one
    #  class-level witness path exists between any two members.  The
    #  alternate proofs that level 3 cares about live in the
    #  term-rewriting space — different orders/locations of rule
    #  application connecting the same endpoints.  ``all_proofs``
    #  enumerates them on demand by re-firing the recorded rules on
    #  real terms (the same machinery ``_edge_path`` uses), with a fuel
    #  cap and a per-path loop check.  Nothing is stored: coherence is
    #  computed when asked, then thrown away.

    def all_proofs(
        self,
        src_term: Any,
        dst_term: Any = None,
        *,
        root_eid: int | None = None,
        cost_fn=None,
        max_paths: int = 32,
        max_steps: int = 8,
        fuel: int = 8192,
    ) -> list:
        """Enumerate distinct derivations ``src_term`` -> ``dst_term``.

        Bounded BFS over the term-rewriting space generated by the rules
        that actually fired during this run (``self._rule_objs``): each
        step is a genuine rule instance — ``_term_match`` on the LHS,
        ``check``/``derive`` honoured, RHS instantiated — located by a
        child-index ``path``, exactly like a :class:`CertStep`.  So
        every returned derivation is a standalone-replayable proof.

        ``max_paths`` caps the number of derivations returned,
        ``max_steps`` caps derivation length, and ``fuel`` bounds total
        match attempts.  Within one derivation a term is never
        revisited (a loop proves nothing new).  Two derivations are
        *distinct* when their ``(rule, path)`` step signatures differ.

        Returns a list of derivations, each a list of :class:`CertStep`.
        ``[[]]`` — one empty derivation — when ``src == dst``.
        Raises ``RuntimeError`` at truncation level 1, where no rule
        provenance exists to enumerate over.
        """
        if not self._track:
            raise RuntimeError(
                "all_proofs requires truncation_level >= 2: level 1 "
                "records no proof witnesses to enumerate over"
            )
        root_eid, dst_term = self._resolve_dst(
            src_term, dst_term, root_eid, cost_fn
        )
        dst_repr = op_repr(dst_term)
        if op_repr(src_term) == dst_repr:
            return [[]]

        rules = list(self._rule_objs.values())
        # frontier entries: (term, steps_so_far, seen_term_reprs)
        frontier: list[tuple[Any, list, frozenset]] = [
            (src_term, [], frozenset({op_repr(src_term)}))
        ]
        paths: list[list] = []
        sigs: set = set()
        while frontier and fuel > 0 and len(paths) < max_paths:
            nxt: list[tuple[Any, list, frozenset]] = []
            for term, steps, seen in frontier:
                if fuel <= 0 or len(paths) >= max_paths:
                    break
                if len(steps) >= max_steps:
                    continue
                for path in _term_paths(term):
                    if fuel <= 0:
                        break
                    sub = _subterm(term, path)
                    for rule in rules:
                        fuel -= 1
                        if fuel < 0:
                            break
                        m = _term_match(rule.lhs, sub)
                        if m is None:
                            continue
                        if rule.check is not None and not rule.check(m):
                            continue
                        inst = dict(m)
                        if rule.derive is not None:
                            extra = rule.derive(m)
                            if extra is None:
                                continue
                            inst.update(extra)
                        r = _term_instantiate(rule.rhs, inst)
                        new_term = _replace_subterm(term, path, r)
                        new_repr = op_repr(new_term)
                        nsteps = steps + [
                            CertStep(rule.name, path, sub, r, inst)
                        ]
                        if new_repr == dst_repr:
                            sig = tuple(
                                (s.rule, s.path) for s in nsteps
                            )
                            if sig not in sigs:
                                sigs.add(sig)
                                paths.append(nsteps)
                        elif new_repr not in seen:
                            nxt.append(
                                (new_term, nsteps, seen | {new_repr})
                            )
            frontier = nxt
        return paths

    def coherent_paths(
        self,
        src_term: Any,
        dst_term: Any = None,
        *,
        root_eid: int | None = None,
        cost_fn=None,
        max_paths: int = 32,
        max_steps: int = 8,
        fuel: int = 8192,
    ) -> dict:
        """Coherence summary between two terms: how many distinct ways
        does the rewrite space prove them equal?

        Thin wrapper over :meth:`all_proofs` (which requires
        ``truncation_level >= 2``).  Returns a dict with the resolved
        endpoints, ``same_eclass`` (whether the e-graph itself judges
        the terms equal — independent evidence for the derivations),
        ``n_paths``, the ``paths`` themselves, and ``truncated`` —
        True when the ``max_paths`` cap was hit, i.e. more coherences
        may exist than reported.
        """
        root_eid, dst_term = self._resolve_dst(
            src_term, dst_term, root_eid, cost_fn
        )
        paths = self.all_proofs(
            src_term,
            dst_term,
            root_eid=root_eid,
            max_paths=max_paths,
            max_steps=max_steps,
            fuel=fuel,
        )
        cs = self._class_of_term(src_term)
        cd = self._class_of_term(dst_term)
        same = (
            cs is not None
            and cd is not None
            and self.find(cs) == self.find(cd)
        )
        return {
            "src": src_term,
            "dst": dst_term,
            "root_eid": root_eid,
            "same_eclass": same,
            "n_paths": len(paths),
            "paths": paths,
            "truncated": len(paths) >= max_paths,
        }


def _term_paths(term: Any, prefix: tuple = ()):
    """Yield the child-index path of every subterm, root first."""
    yield prefix
    if isinstance(term, Op):
        for i, a in enumerate(term.args):
            yield from _term_paths(a, prefix + (i,))


def _iter_ops(term: Any):
    """Yield every Op node in a term (for the structural-size tie-break)."""
    if isinstance(term, Op):
        yield term
        for a in term.args:
            yield from _iter_ops(a)


# ---------------------------------------------------------------------------
#  Certificate verification — replay on real terms, no e-graph
# ---------------------------------------------------------------------------


def _term_match(
    pattern: Any, term: Any, _subst: dict | None = None
) -> dict | None:
    """Structural match of a pattern against a plain *term* (no e-graph).

    Mirrors :meth:`EGraph._match` semantics at term granularity: string
    leaves in the pattern are metavariables bound to subterms (repeated
    metavariables must bind structurally equal terms); string-valued
    attributes are attribute metavariables bound under ``"$attr:"``
    keys; concrete leaves (Var/Const/Param) match by ``repr``.
    Returns the bindings dict, or ``None`` on mismatch.
    """
    subst = {} if _subst is None else _subst
    if isinstance(pattern, str):
        prev = subst.get(pattern)
        if prev is None:
            subst[pattern] = term
            return subst
        return subst if op_repr(prev) == op_repr(term) else None
    if isinstance(pattern, Op):
        if not isinstance(term, Op) or term.op != pattern.op:
            return None
        if len(term.args) != len(pattern.args):
            return None
        if set(term.attrs) != set(pattern.attrs):
            return None
        for k, pv in pattern.attrs.items():
            nv = term.attrs[k]
            if isinstance(pv, str):
                key = "$attr:" + pv
                if key in subst:
                    if subst[key] != nv:
                        return None
                else:
                    subst[key] = nv
            elif nv != pv:
                return None
        for pa, ta in zip(pattern.args, term.args):
            if _term_match(pa, ta, subst) is None:
                return None
        return subst
    # concrete leaf (Const/Param/Var embedded in the pattern)
    return subst if repr(pattern) == repr(term) else None


def _term_instantiate(pattern: Any, subst: dict) -> Any:
    """Instantiate a pattern with term-valued bindings (pure terms).

    The term-level analogue of :meth:`EGraph._instantiate`: metavariable
    strings map to terms, ``"$attr:"`` keys resolve attribute
    metavariables, concrete leaves pass through unchanged.
    """
    if isinstance(pattern, str):
        return subst[pattern]
    if isinstance(pattern, Op):
        args = [_term_instantiate(a, subst) for a in pattern.args]
        attrs = {}
        for k, v in pattern.attrs.items():
            if isinstance(v, str):
                attrs[k] = subst.get("$attr:" + v, v)
            else:
                attrs[k] = v
        return Op.make(pattern.op, *args, **attrs)
    return pattern


def _subterm(term: Any, path: tuple) -> Any:
    """The subterm of *term* at child-index path, or None if absent."""
    for i in path:
        if not isinstance(term, Op) or i >= len(term.args):
            return None
        term = term.args[i]
    return term


def _replace_subterm(term: Any, path: tuple, new: Any) -> Any:
    """*term* with the subterm at *path* replaced by *new*."""
    if not path:
        return new
    if not isinstance(term, Op) or path[0] >= len(term.args):
        raise CertificateVerificationError(
            f"cannot descend path {list(path)} in {op_repr(term)}"
        )
    i = path[0]
    args = list(term.args)
    args[i] = _replace_subterm(args[i], path[1:], new)
    return Op.make(term.op, *args, **dict(term.attrs))


def verify_certificate(
    src_term: Any, cert: Certificate, *, strict: bool = False
) -> Any:
    """Replay a certificate's rule applications on real terms.

    For each step: descend to ``path`` in the evolving term, re-match
    the rule's LHS against the subterm found there, check the recorded
    bindings and the recorded LHS/RHS instances, re-run ``check``/
    ``derive`` side conditions, instantiate the RHS, and substitute.
    ``egraph_dependent`` steps cannot be replayed standalone — the
    recorded ``lhs`` must still be present at ``path``, then ``rhs`` is
    substituted as a trusted assertion; ``strict=True`` rejects any
    certificate containing them.

    Returns the reconstructed term — ``cert.dst`` on success.
    Raises :class:`CertificateVerificationError` on any mismatch.
    """
    if strict and cert.n_egraph_dependent:
        raise CertificateVerificationError(
            f"{cert.n_egraph_dependent} e-graph-dependent step(s) "
            "cannot be replayed standalone"
        )
    if cert.src is not None and op_repr(src_term) != op_repr(cert.src):
        raise CertificateVerificationError(
            f"source mismatch: certificate proves {op_repr(cert.src)}, "
            f"got {op_repr(src_term)}"
        )
    current = src_term
    for i, step in enumerate(cert.steps):
        sub = _subterm(current, step.path)
        if sub is None:
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): path {list(step.path)} "
                f"absent in {op_repr(current)}"
            )
        if step.egraph_dependent:
            if op_repr(sub) != op_repr(step.lhs):
                raise CertificateVerificationError(
                    f"step {i} (e-graph-dependent): expected "
                    f"{op_repr(step.lhs)} at {list(step.path)}, "
                    f"found {op_repr(sub)}"
                )
            current = _replace_subterm(current, step.path, step.rhs)
            continue
        rule = cert.rules.get(step.rule)
        if rule is None:
            raise CertificateVerificationError(
                f"step {i}: unknown rule {step.rule!r}"
            )
        if op_repr(step.lhs) != op_repr(sub):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): recorded LHS instance "
                f"{op_repr(step.lhs)} != subterm {op_repr(sub)}"
            )
        m = _term_match(rule.lhs, sub)
        if m is None:
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): LHS does not match "
                f"{op_repr(sub)}"
            )
        for k, v in step.bindings.items():
            if k not in m:
                continue  # derived binding — checked via derive/rhs below
            same = (
                m[k] == v
                if k.startswith("$attr:")
                else op_repr(m[k]) == op_repr(v)
            )
            if not same:
                raise CertificateVerificationError(
                    f"step {i} ({step.rule}): binding {k} tampered"
                )
        if rule.check is not None and not rule.check(m):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): side condition fails on replay"
            )
        inst = dict(m)
        for k, v in step.bindings.items():
            inst.setdefault(k, v)  # derived $attr bindings
        if rule.derive is not None:
            extra = rule.derive(m)
            if extra is None:
                raise CertificateVerificationError(
                    f"step {i} ({step.rule}): derive vetoed on replay"
                )
            inst.update(extra)
        rhs = _term_instantiate(rule.rhs, inst)
        if op_repr(rhs) != op_repr(step.rhs):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): recorded RHS "
                f"{op_repr(step.rhs)} != instantiated {op_repr(rhs)}"
            )
        current = _replace_subterm(current, step.path, rhs)
    if op_repr(current) != op_repr(cert.dst):
        raise CertificateVerificationError(
            f"replay produced {op_repr(current)}, "
            f"certificate claims {op_repr(cert.dst)}"
        )
    return current
