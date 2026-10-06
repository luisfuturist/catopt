"""Core e-graph: union-find, matching, saturation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any, cast

from catopt_core.egraph.certs import (
    ProofEdge,
)
from catopt_core.egraph.extract import _ExtractMixin
from catopt_core.egraph.match import (
    _Epoch,
    _PLeaf,
    _POp,
    _Prog,
    _PVar,
    compile_pattern,
)
from catopt_core.egraph.proof import _ProofMixin
from catopt_core.egraph.types import (
    EClass,
    ENode,
    Rewrite,
    UnionFind,
    _LeafRegistry,
    _norm_attr_value,
    _pattern_attrs,
)
from catopt_core.ir import Op

logger = logging.getLogger("catopt_core.egraph.core")


def _policy_name(policy: Any) -> str | None:
    """Name the schedule policy for ``stats`` — ``None`` when unset.

    An unset policy reports ``None`` rather than being absent, so the
    stats dict has one stable shape.
    """
    if policy is None:
        return None
    return str(getattr(policy, "name", type(policy).__name__))


class EGraph(_ExtractMixin, _ProofMixin):
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
        """Initialise the union-find, class maps, and saturation state."""
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
        # ``(op, arity) -> canonical class ids`` — the targeted-
        # eligibility index (plan 0010, lever 1b): a compiled pattern
        # only scans classes holding a member of the right head shape.
        self._opk: dict[tuple[str, int], set[int]] = {}
        # ``pattern -> compiled matcher program`` (lever 1a).  Patterns
        # are hashable and interned, so each distinct LHS compiles once
        # per e-graph.
        self._progs: dict[Any, _Prog] = {}
        # Live frozen-read epochs (lever 2b): a suspended match
        # enumeration registers here so ``union`` can journal the
        # merges it must hide from the matcher.
        self._epochs: list[_Epoch] = []
        # Incremental congruence state (lever 2c): ``_cong_owner`` is
        # the persistent ``canonical enode -> owning class`` map;
        # ``_cong_pend`` is the worklist of ``(class, enode)`` pairs
        # needing an ownership pass — populated on enode birth, on
        # child re-canonicalisation, and on class merges.
        self._cong_owner: dict[ENode, int] = {}
        self._cong_pend: set[tuple[int, ENode]] = set()
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
        """Return the number of e-classes."""
        return len(self._classes)

    @property
    def n_enodes(self) -> int:
        """Return the number of enodes."""
        return len(self._node_to_class)

    def find(self, eid: int) -> int:
        """Return the canonical e-class id of ``eid``.

        While a frozen-read match epoch is live (a suspended
        enumeration), path-halving writes are JOURNALED into every
        active epoch's ``jph`` overlay — first write wins, so the
        epoch keeps the freeze-time parent of every reparented node.
        Answers are identical to plain union-find; only the write
        journaling differs.
        """
        if self._epochs:
            parent = self._uf.parent
            while parent[eid] != eid:
                p = parent[eid]
                gp = parent[p]
                for ep in self._epochs:
                    ep.jph.setdefault(eid, p)
                parent[eid] = gp
                eid = gp
            return eid
        return self._uf.find(eid)

    def get_class(self, eid: int) -> EClass:
        """Return the e-class containing ``eid``."""
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
        attr_t = tuple(
            sorted(
                (k, _norm_attr_value(v))
                for k, v in (attrs or {}).items()
            )
        )
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
        self._opk.setdefault(
            (enode.op, len(enode.children)), set()
        ).add(eid)
        self._cong_pend.add((eid, enode))
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
            if self._epochs:
                # Frozen-read journaling (lever 2b): a suspended match
                # enumeration must keep seeing the pre-merge graph —
                # record the lost root, the deleted class object, and
                # the members that just moved into ``new_canon``.
                moved = source.nodes - target.nodes
                for ep in self._epochs:
                    ep.ovr[old_canon] = old_canon
                    ep.dead[old_canon] = source
                    acc = ep.added.get(new_canon)
                    if acc is None:
                        # Per-epoch copy: coexisting epochs freeze at
                        # different times and must not share the set —
                        # a later union mutating it would leak into an
                        # epoch that should keep the earlier members.
                        ep.added[new_canon] = set(moved)
                    else:
                        acc |= moved
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
                # Incremental congruence (lever 2c): the enodes that
                # re-key because their child resolved to ``old_canon``
                # are exactly the direct parents' members mentioning
                # it — queue them for the ownership worklist.
                for p0 in p_old:
                    # ``find`` returns a live canonical id — the class
                    # is always present.
                    pc = self.find(p0)
                    for n in self._classes[pc].nodes:
                        if old_canon in n.children:
                            self._cong_pend.add((pc, n))
            for n in source.nodes:
                oc = self._op_classes.get(n.op)
                if oc is not None:
                    oc.discard(old_canon)
                    oc.add(new_canon)
                key = (n.op, len(n.children))
                ok = self._opk.get(key)
                if ok is not None:
                    ok.discard(old_canon)
                    ok.add(new_canon)
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
        """Return the number of recorded proof edges."""
        return len(self._merge_log)

    @property
    def applications(self) -> list[dict]:
        """Every rule application recorded during saturation."""
        return list(self._applications)

    # -- non-local offers: the pointwise-witness ritual --------------------

    def _pointwise_witness(
        self,
        cid: int,
        rhs_eid: int,
        *,
        rhs_term: Any = None,
        lhs_term: Any = None,
        provenance: str,
        law: str,
        name_eid: int | None = None,
        error_bound: float | None = None,
        bound_norm: str = "spectral",
    ) -> Rewrite | None:
        """Synthesise the pointwise :class:`Rewrite` for an offered member.

        Certifies one non-locally offered member — see
        :meth:`_offer_witness`.

        ``lhs`` defaults to the *oldest* member of ``cid``'s class —
        the term the e-graph saw first — chosen so the certificate's
        ``src`` side usually *is* the witness LHS and the bridging
        connect is trivial.  Passes asserting a specific member (a
        Param leaf they just interned, say) pass ``lhs_term``
        explicitly.  ``rhs`` is the offered member itself, resolved
        through :meth:`any_term` when ``rhs_term`` is not given;
        passes needing a *deterministic* representative
        (``_any_term_cached``, ``_oldest_term``) resolve it themselves
        and pass the term in.

        The name is ``f"{provenance}#{rhs_eid}"`` — unique per offered
        enode, so several offers in one graph never collide in
        ``_rule_objs`` (``name_eid`` overrides the anchor for a pass
        that names the merge after the other side).

        Returns ``None`` when either side is unresolvable — a
        degenerate fully-cyclic class — so the caller's merge may
        proceed witness-free and stay ``egraph_dependent`` rather than
        carrying a fabricated certificate.  (The older
        ``or any_term`` fallback some sites used can never fire: both
        resolvers return ``None`` exactly when the class has no
        acyclic member.)
        """
        if rhs_term is None:
            rhs_term = self.any_term(rhs_eid)
        src = (
            lhs_term
            if lhs_term is not None
            else self._oldest_term(self.find(cid))
        )
        if src is None or rhs_term is None:
            return None
        return Rewrite(
            name=(
                f"{provenance}#"
                f"{name_eid if name_eid is not None else rhs_eid}"
            ),
            lhs=src,
            rhs=rhs_term,
            law=law,
            error_bound=error_bound,
            bound_norm=bound_norm,
        )

    def _offer_witness(
        self,
        cid: int,
        rhs_eid: int | None = None,
        *,
        rhs_term: Any = None,
        lhs_term: Any = None,
        provenance: str,
        law: str,
        witness: bool = True,
        note: str = "",
        name_eid: int | None = None,
        error_bound: float | None = None,
        bound_norm: str = "spectral",
        term_provenance: str = "input",
    ) -> bool:
        """Offer a non-locally-computed member under a pointwise witness.

        The canonical ritual behind every non-local pass.

        A non-local pass offers a member no LHS pattern could produce
        (it is computed from the whole e-graph, not from a matched
        subterm).  Recording the merge under a synthesised
        :class:`Rewrite` — ``class_member -> offered_term`` — makes
        the asserted equality replayable: :meth:`certificate` emits it
        as a named step and ``verify_certificate`` re-matches and
        re-instantiates it standalone instead of emitting an
        ``egraph_dependent`` stub.

        Steps: intern ``rhs_term`` via :meth:`add_term` when
        ``rhs_eid`` is not given (``term_provenance`` tags the new
        enodes), mint the witness through :meth:`_pointwise_witness`,
        then :meth:`union` the two classes under it.  ``witness``
        toggles the witness only — the merge happens either way, so a
        pass declining certificates stays honestly witness-free.
        ``note`` rides on the :class:`ProofEdge` like every other
        union.  Returns :meth:`union`'s result.
        """
        if rhs_eid is None:
            rhs_eid = self.add_term(
                rhs_term, provenance=term_provenance
            )
        wit = (
            self._pointwise_witness(
                cid,
                rhs_eid,
                rhs_term=rhs_term,
                lhs_term=lhs_term,
                provenance=provenance,
                law=law,
                name_eid=name_eid,
                error_bound=error_bound,
                bound_norm=bound_norm,
            )
            if witness
            else None
        )
        return self.union(cid, rhs_eid, witness=wit, note=note)

    # -- pattern matching --
    #
    # Patterns are compiled once per e-graph into a :class:`_Prog`
    # decision tree (lever 1a) and enumerated lazily (lever 2b):
    # ``matches`` is a generator — no per-class substitution list is
    # ever materialised on the unbounded path.  Between two yielded
    # substitutions ``apply_rule`` may merge classes; the enumeration
    # keeps reading the graph as of call time through the frozen
    # ``_Epoch`` overlays (``_mfind`` / ``_mclass`` / ``_mnodes``), so
    # the substitution stream is *identical* to what eager
    # enumeration produced.  ``max_results`` still caps the count,
    # preserving the bounded semantics exactly — including the quirk
    # that an e-node whose enumeration crosses the cap contributes
    # nothing (``_m_bounded`` mirrors the interpreted algorithm).

    def _prog_for(self, pattern: Any) -> _Prog:
        """Return the compiled matcher for *pattern* (cached per graph)."""
        try:
            prog = self._progs.get(pattern)
        except TypeError:
            # Unhashable non-Op pattern (e.g. a list leaf): compile it
            # uncached — it can only ever be a concrete leaf key.
            return compile_pattern(pattern)
        if prog is None:
            prog = compile_pattern(pattern)
            self._progs[pattern] = prog
        return prog

    def matches(
        self, pattern: Any, eid: int, max_results: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield every substitution matching *pattern* at e-class *eid*.

        The enumeration is LAZY: the caller drives it, and may mutate
        the graph between substitutions — each enumeration reads a
        frozen view of the graph (see ``_Epoch``) so the stream equals
        the list the old eager matcher returned, in the same order.
        Materialise with ``list(...)`` when a sequence is needed.

        ``max_results`` bounds the enumeration: matching stops once
        that many substitutions have been yielded — deterministic (the
        first-found wins) and used by the saturation loop to enforce
        per-rule expansion budgets inside giant e-classes.
        """
        return self._iter_matches(
            self._prog_for(pattern), eid, max_results
        )

    def _iter_matches(
        self, prog: _Prog, eid: int, limit: int | None
    ) -> Iterator[dict[str, Any]]:
        """Drive one compiled match under a fresh frozen-read epoch.

        Only the streaming path registers the epoch: bounded
        enumeration (:meth:`_m_bounded`) completes before the first
        yield, so its overlays would stay empty forever — registering
        it would just tax every union during consumption.  It still
        reads through the same frozen accessors on an *unregistered*
        epoch (empty overlays ≡ live reads), keeping one code path.
        """
        if limit is not None:
            ep = _Epoch(watermark=self._next_id)
            results: list[dict[str, Any]] = []
            self._m_bounded(prog.root, eid, {}, results, limit, ep)
            yield from results
            return
        ep = _Epoch(watermark=self._next_id)
        self._epochs.append(ep)
        try:
            bindings: dict[str, Any] = {}
            for _ in self._m_stream(prog.root, eid, bindings, ep):
                yield dict(bindings)
        finally:
            self._epochs.remove(ep)

    # -- frozen reads (live inside a match epoch) ------------------------

    def _mfind(self, eid: int, ep: _Epoch) -> int:
        """Canonical e-class id of ``eid`` *as of the epoch freeze*.

        Walks ``uf.parent`` with the epoch's overlays: every root
        that lost a merge mid-enumeration maps to itself (``ovr``),
        and every node path-halving reparented mid-enumeration
        resolves through its journaled freeze-time parent (``jph``).
        All other parent pointers are untouched by the only two
        mutations possible mid-epoch (halving, root unions).
        """
        ovr = ep.ovr
        jph = ep.jph
        parent = self._uf.parent
        while True:
            p = ovr.get(eid)
            if p is None:
                p = jph.get(eid)
            if p is None:
                p = parent[eid]
            if p == eid:
                return eid
            eid = p

    def _mclass(self, eid: int, ep: _Epoch) -> EClass:
        """Return the e-class of a frozen id (alive or merged away)."""
        ec = ep.dead.get(eid)
        if ec is not None:
            return ec
        return self._classes[eid]

    def _mnodes(self, eid: int, op: str, ep: _Epoch) -> list:
        """Frozen ``op``-member list of a frozen class, epoch-cached.

        Members merged in after the freeze (``ep.added``) are
        subtracted, preserving the call-time member set — and its
        iteration order (``ec.nodes`` order minus removals).
        """
        key = (eid, op)
        lst = ep.nlists.get(key)
        if lst is None:
            ec = self._mclass(eid, ep)
            added = ep.added.get(eid)
            if added is None:
                # Class unchanged since the freeze — the frozen
                # member list is exactly the live one: reuse the
                # e-class's persistent ``by_op`` cache instead of
                # rebuilding the filter every epoch.
                lst = self._nodes_of(ec, op)
            else:
                lst = [
                    n for n in ec.nodes if n.op == op and n not in added
                ]
            ep.nlists[key] = lst
        return lst

    # -- streaming matcher (unbounded enumeration) -----------------------

    def _m_stream(
        self, pn: Any, eid: int, b: dict, ep: _Epoch
    ) -> Iterator[None]:
        """Yield once per substitution of *pn* at ``eid``.

        Bindings live in the single mutable table ``b`` — bind on
        descend, undo on backtrack (the trail is each frame's own
        undo; no per-branch ``dict`` copies).  When the generator
        yields, ``b`` holds a complete substitution for this subtree.
        """
        cls = pn.__class__
        if cls is _PVar:
            ce = self._mfind(eid, ep)
            if pn.name in b:
                if b[pn.name] == ce:
                    yield
                return
            b[pn.name] = ce
            yield
            del b[pn.name]
            return
        if cls is _PLeaf:
            nid = self._node_to_class.get(pn.enode)
            # ``nid >= watermark``: the leaf was interned mid-
            # enumeration — the eager matcher never saw it.
            if (
                nid is not None
                and nid < ep.watermark
                and self._mfind(nid, ep) == self._mfind(eid, ep)
            ):
                yield
            return
        # _POp
        ce = self._mfind(eid, ep)
        for node in self._mnodes(ce, pn.op, ep):
            if len(node.children) != len(pn.children):
                continue
            node_attrs = dict(node.attrs)
            if set(node_attrs) != pn.keyset:
                continue
            # Attribute matching: literal values must equal — under
            # ``!=``, i.e. numerically (a ``dim=0`` pattern still
            # matches a ``dim=0.0`` enode; match-time leniency while
            # enode identity is spelling-strict, see
            # ``catopt_core.ir.Op``).  A string pattern value is an
            # attribute metavariable bound under "$attr:<name>" —
            # binding must be consistent everywhere it repeats.
            marked: list[str] = []
            ok = True
            for k, pv in pn.attrs:
                nv = node_attrs[k]
                if isinstance(pv, str):
                    key = "$attr:" + pv
                    if key in b:
                        if b[key] != nv:
                            ok = False
                            break
                    else:
                        b[key] = nv
                        marked.append(key)
                elif nv != pv:
                    ok = False
                    break
            if ok:
                yield from self._m_pos(
                    pn.children, node.children, 0, b, ep
                )
            for key in marked:
                del b[key]

    def _m_pos(
        self,
        pats: tuple,
        children: tuple,
        i: int,
        b: dict,
        ep: _Epoch,
    ) -> Iterator[None]:
        """Yield once per joint substitution of child positions ``i..``.

        Threads the incoming bindings so a metavariable appearing at
        several positions (the shared ``x`` in ``x@W1 + x@W2``) is
        checked for consistency — the eager semantics' soundness rule.
        """
        if i == len(pats):
            yield
            return
        for _ in self._m_stream(pats[i], children[i], b, ep):
            yield from self._m_pos(pats, children, i + 1, b, ep)

    # -- bounded matcher (max_results) ------------------------------------

    def _m_bounded(
        self,
        pn: Any,
        eid: int,
        subst: dict[str, Any],
        results: list[dict[str, Any]],
        limit: int,
        ep: _Epoch,
    ) -> None:
        """Enumerate matches under a result cap — the interpreted spec.

        Mirrors the pre-compilation ``_match`` *exactly*, including
        the cap semantics: an e-node whose enumeration crosses
        ``limit`` contributes ZERO substitutions (the ``ok=False``
        drop), while earlier nodes keep theirs.  Bounded memory by
        construction — ``results`` never exceeds ``limit``.
        """
        if len(results) >= limit:
            return
        eid = self._mfind(eid, ep)

        if pn.__class__ is _PVar:
            if pn.name in subst:
                if subst[pn.name] == eid:
                    results.append(dict(subst))
                return
            subst[pn.name] = eid
            results.append(dict(subst))
            del subst[pn.name]
            return

        if pn.__class__ is _POp:
            for node in self._mnodes(eid, pn.op, ep):
                # NB: no cap re-check here — ``results`` only reaches
                # ``limit`` inside the ``ok`` arm below, which returns
                # immediately, so a guard at the loop head can never
                # fire.
                if len(node.children) != len(pn.children):
                    continue
                node_attrs = dict(node.attrs)
                if set(node_attrs) != pn.keyset:
                    continue
                # Same numerically-lenient ``!=`` attr compare as
                # ``_m_stream`` — see its comment for the contract.
                attr_substs: list[dict[str, Any]] = [dict(subst)]
                attr_ok = True
                for k, pv in pn.attrs:
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
                # Thread the incoming bindings so shared metavariables
                # are checked for consistency at every position.
                child_substs: list[dict[str, Any]] = attr_substs
                ok = True
                for i, cp in enumerate(pn.children):
                    new_substs: list[dict[str, Any]] = []
                    for cs in child_substs:
                        if len(results) + len(new_substs) >= limit:
                            ok = False
                            break
                        child_results: list[dict[str, Any]] = []
                        self._m_bounded(
                            cp,
                            node.children[i],
                            dict(cs),
                            child_results,
                            limit,
                            ep,
                        )
                        new_substs.extend(child_results)
                    if not new_substs:
                        ok = False
                        break
                    child_substs = new_substs
                if ok:
                    room = limit - len(results)
                    results.extend(child_substs[:room])
                    if len(results) >= limit:
                        return
            return

        # Concrete leaf — match by registry key.  ``nid >= watermark``
        # means the leaf was interned mid-enumeration (invisible).
        nid = self._node_to_class.get(pn.enode)
        if (
            nid is not None
            and nid < ep.watermark
            and self._mfind(nid, ep) == eid
        ):
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
        callers that mutate the graph outside the saturation loop; it
        also closes congruence (see :meth:`_close_congruence`).
        """
        if classes is not None:
            return self._canonicalise(classes)
        changed = self._canonicalise(list(self._classes.keys()))
        return self._close_congruence() or changed

    def _canonicalise(self, ids: Any) -> bool:
        """Re-canonicalize each class's enode children (one pass)."""
        changed = False
        seen: set[int] = set()
        for eid0 in ids:
            eid = self.find(eid0)
            if eid in seen or eid not in self._classes:
                continue
            seen.add(eid)
            eclass = self._classes[eid]
            new_nodes: set[ENode] = set()
            for node in eclass.nodes:
                nn, node_changed = self._canonical_node(node, eid)
                changed = changed or node_changed
                new_nodes.add(nn)
            if new_nodes != eclass.nodes:
                eclass.nodes = new_nodes
                eclass.by_op = None
        return changed

    def _canonical_node(
        self, node: ENode, eid: int
    ) -> tuple[ENode, bool]:
        """Canonicalize one enode's children; return (enode, changed).

        When children change, register the upward edge for each
        canonical child so later changes propagate dirty to ``eid``
        (already dirty — it is an ancestor of the merge that forced the
        canonicalisation), and carry the original's provenance onto the
        fresh enode (same term, new child ids).
        """
        if not node.children:
            return node, False
        canon = tuple(self.find(c) for c in node.children)
        if canon == node.children:
            return node, False
        for c in canon:
            self._parents.setdefault(self.find(c), set()).add(eid)
        nn = ENode(node.op, canon, node.attrs)
        if self._track:
            self._inherit_provenance(node, nn, eid)
        # Incremental congruence: ``node`` re-keyed out of ``eid`` —
        # drop its ownership claim if this class held it (a stale claim
        # would merge a later owner into the wrong class) and queue the
        # canonical ``nn`` for the ownership worklist.
        prev = self._cong_owner.get(node)
        if prev is not None and self.find(prev) == eid:
            del self._cong_owner[node]
        self._cong_pend.add((eid, nn))
        return nn, True

    def _inherit_provenance(
        self, node: ENode, nn: ENode, eid: int
    ) -> None:
        """Copy ``node``'s tracking metadata onto its canonical ``nn``."""
        if node in self._enode_origin:
            self._enode_origin.setdefault(nn, self._enode_origin[node])
        if node in self._enode_birth:
            self._enode_birth.setdefault(nn, self._enode_birth[node])
        if node in self._enode_app:
            self._enode_app.setdefault(nn, self._enode_app[node])
        self._node_to_class.setdefault(nn, eid)

    def _cong_canon(self, node: ENode) -> ENode:
        """Return ``node`` with canonically-resolved children."""
        canon = tuple(self.find(c) for c in node.children)
        if canon == node.children:
            return node
        return ENode(node.op, canon, node.attrs)

    def _close_congruence(self) -> bool:
        """Union e-classes whose canonical enodes coincide.

        Congruence: two enodes that are identical after child
        canonicalisation must share a class.  This is the INCREMENTAL
        version (plan 0010, lever 2c): ``_cong_owner`` — the
        ``canonical enode -> owning class`` map — persists across
        calls, and the worklist ``_cong_pend`` carries only the enodes
        that could need re-keying since the last pass: freshly added
        members (:meth:`_add_enode`), members re-canonicalised by
        :meth:`_canonicalise`, and members whose child resolved to a
        class that just merged (:meth:`union`).  The pass is
        proportional to the delta, not to ``E``; merges it fires
        enqueue their own deltas, iterating to the same fixed point
        the whole-graph rescan reached.
        """
        owner = self._cong_owner
        pend = self._cong_pend
        changed = False
        while pend:
            merges: list[tuple[int, int]] = []
            while pend:
                eid0, node = pend.pop()
                eid = self.find(eid0)
                # ``find`` returns a live canonical id — always present.
                eclass = self._classes[eid]
                nn = self._cong_canon(node)
                if nn is not node:
                    # Re-keyed out of the class — apply the
                    # canonicalisation pointwise.
                    if node in eclass.nodes:
                        eclass.nodes.discard(node)
                        eclass.nodes.add(nn)
                        eclass.by_op = None
                        if self._track:
                            # ``_inherit_provenance`` also hash-conses
                            # ``nn`` into ``_node_to_class``.
                            self._inherit_provenance(node, nn, eid)
                        for c in nn.children:
                            self._parents.setdefault(
                                self.find(c), set()
                            ).add(eid)
                        # Drop the pre-canonical form's claim — a
                        # claim can only belong to the class holding
                        # the node (``eid`` or a predecessor that
                        # merged into it), so an unconditional pop is
                        # equivalent to checking ``find(prev)==eid``.
                        owner.pop(node, None)
                    if nn not in eclass.nodes:
                        continue
                    node = nn
                elif node not in eclass.nodes:
                    # Stale worklist item — the enode is gone.
                    continue
                prev = owner.setdefault(node, eid)
                if self.find(prev) != eid:
                    merges.append((prev, eid))
            if not merges:
                return changed
            changed = True
            for a, b in merges:
                self.union(a, b)
        return changed

    # -- rule application --

    def _instantiate(self, pn: Any, subst: dict[str, Any]) -> int:
        """Instantiate a compiled pattern (RHS) with a substitution.

        Takes the compiled ``_PNode`` tree (see
        :func:`catopt_core.egraph.match.compile_pattern`) — the same
        recursion the interpreted version ran, minus the per-node
        ``isinstance`` ladder and ``_pattern_attrs`` recomputation.

        When proof tracking is on, ``self._inst_last_enode`` records the
        enode realising the pattern's root (post-order: the outermost
        call assigns last), so the caller can tell which enode the
        instantiated RHS is headed by — or ``None`` when the RHS is a
        bare metavariable/leaf binding.
        """
        cls = pn.__class__
        if cls is _PVar:
            self._inst_last_enode = None
            return subst[pn.name]
        if cls is _POp:
            child_eids = tuple(
                self._instantiate(a, subst) for a in pn.children
            )
            if pn.attrs:
                # Attribute metavariables (string values) resolve
                # through the substitution's "$attr:" namespace.
                attrs = {}
                for k, v in pn.attrs:
                    if isinstance(v, str):
                        attrs[k] = subst.get("$attr:" + v, v)
                    else:
                        attrs[k] = v
                attr_t = tuple(
                    sorted(
                        (k, _norm_attr_value(v))
                        for k, v in attrs.items()
                    )
                )
            else:
                attr_t = ()
            enode = ENode(
                pn.op,
                tuple(self.find(c) for c in child_eids),
                attr_t,
            )
            if enode in self._node_to_class:
                eid = self.find(self._node_to_class[enode])
            else:
                eid = self._add_enode(enode)
            self._inst_last_enode = enode
            return eid
        # _PLeaf
        # Register the concrete leaf BEFORE minting the enode: a
        # leaf key that was never seen by ``add_term`` (a Const in
        # a rule RHS, or a re-used repr name) would otherwise
        # decode to a stale cross-call term — or to the raw key
        # string, which poisons any extracted member it lands in.
        _LeafRegistry.register(pn.term)
        eid = self.add_leaf(pn.key)
        self._inst_last_enode = pn.enode
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
        """Return the smallest acyclic member of an e-class.

        Returns ``(term, size)`` — fewest ops wins.

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
        """Resolve a member for rewrite side conditions.

        Memoised for the duration of one ``apply_rule``.

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

    def _head_ok(self, prog: _Prog, eid: int) -> bool:
        """Cheap child-op eligibility check for one candidate class.

        True iff some ``head_op`` member of the class has, at every
        Op-typed child position of the compiled pattern, a child class
        containing a member of the required op.  Pure existence filter
        — the matcher still does the real work, but a class that
        cannot possibly match never opens a match enumeration.
        """
        eclass = self._classes[eid]
        reqs = prog.child_reqs
        # ``child_reqs`` only exists on ``_POp`` roots — head_op is set.
        head_op = cast(str, prog.head_op)
        for node in self._nodes_of(eclass, head_op):
            if len(node.children) != prog.head_arity:
                continue
            if all(
                self.find(node.children[i])
                in self._op_classes.get(op, ())
                for i, op in reqs
            ):
                return True
        return False

    def _candidate_classes(
        self, prog: _Prog, search: set | list | None
    ) -> Any:
        """E-classes a rule's compiled LHS could possibly match at.

        ``search=None`` scans the whole graph (the pre-incremental
        behaviour); a set restricts to the dirty frontier.  In both
        cases an Op-rooted pattern additionally filters to classes
        that contain a member with the pattern's head ``(op, arity)``
        — the targeted-eligibility index (lever 1b) — plus the
        child-op existence check when the pattern's children are
        Op-typed; a metavariable pattern binds at any class, and a
        concrete-leaf pattern can only match at the leaf's own class.
        """
        if prog.head_op is not None:
            eligible = {
                self.find(c)
                for c in self._opk.get(
                    (prog.head_op, prog.head_arity), ()
                )
            }
        elif prog.leaf_key is None:
            eligible = None
        else:
            leaf = ENode("leaf", (), (("key", prog.leaf_key),))
            leid = self._node_to_class.get(leaf)
            eligible = {self.find(leid)} if leid is not None else set()
        # ``_classes`` mutates under us as unions fire — snapshot.
        ids = list(self._classes.keys()) if search is None else search
        for eid0 in ids:
            eid = self.find(eid0)
            if eligible is not None and eid not in eligible:
                continue
            if prog.child_reqs and not self._head_ok(prog, eid):
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
        lhs_prog = self._prog_for(rule.lhs)
        rhs_prog = self._prog_for(rule.rhs)
        try:
            candidates = self._candidate_classes(lhs_prog, search)
            for eid in candidates:
                match_cap = None
                if enode_budget is not None:
                    spent = self.n_enodes - n_start
                    if spent >= enode_budget:
                        break
                    match_cap = min(
                        enode_budget - spent, self._MATCH_CAP
                    )
                # Streaming consumption (lever 2b): each substitution
                # is applied as it is yielded; the enumeration reads
                # the frozen epoch so the stream equals the eager
                # list.  The epoch is released by the generator's own
                # ``finally`` on exhaustion/close.
                for subst in self._iter_matches(
                    lhs_prog, eid, match_cap
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
                    try:
                        if self._track:
                            self._tag_rule = rule.name
                            self._collect = []
                            self._inst_last_enode = None
                            try:
                                rhs_eid = self._instantiate(
                                    rhs_prog.root, subst
                                )
                            finally:
                                self._tag_rule = None
                                created = self._collect
                                self._collect = None
                        else:
                            rhs_eid = self._instantiate(
                                rhs_prog.root, subst
                            )
                    except KeyError:
                        # A binding that cannot realize the RHS — the
                        # match bound less than the pattern needs (a
                        # free metavariable, e.g. an attr left open by
                        # a constructed object's specialize map).
                        # The firing cannot be built: skip it, like a
                        # check veto.
                        continue
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

    def _scheduled(
        self,
        ordered: list[Rewrite],
        policy: Any,
        root_eid: int,
        iteration: int,
    ) -> list[Rewrite]:
        """Order one iteration's rules — the policy decides when given.

        A policy may only *reorder*: every rule is still returned, so
        the fixed point is unchanged (ADR 0003 invariant 5).  An action
        outside the offered set leaves the remainder in declared order,
        so no rule is ever dropped.
        """
        if policy is None:
            return ordered
        from catopt_core.game import Action, GameState

        state = GameState(self, root_eid, iteration)
        remaining = list(ordered)
        out: list[Rewrite] = []
        while remaining:
            offered = [Action(r.name, root_eid) for r in remaining]
            pick = policy.choose(state, offered)
            name = getattr(pick, "rule", pick)
            idx = next(
                (i for i, r in enumerate(remaining) if r.name == name),
                None,
            )
            if idx is None:
                out.extend(remaining)
                break
            out.append(remaining.pop(idx))
        return out

    def run(
        self,
        rules: Iterable[Rewrite],
        root_eid: int,
        max_iterations: int = 100,
        max_nodes: int = 100_000,
        rule_budgets: dict[str, int] | None = None,
        stop: str = "fixed_point",
        patience: int = 3,
        cost_fn: Any = None,
        policy: Any = None,
    ) -> dict[str, Any]:
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

        Rule scheduling (lever 1c): when ``rules`` carries a
        ``priority_of`` map (a :class:`~catopt_core.laws.RuleSet`), each
        iteration applies the rules in priority order — lower first, so
        cheap/filtering rules fire before the closure-generating ones.
        Ordering is a stable sort, so rules at the default priority
        keep their declared order — the default presets (no
        priorities) behave identically to before.

        Lazy saturation (lever 1d, opt-in): ``stop="improving"``
        extracts the best term after every iteration and stops once
        the cost has not strictly improved for ``patience``
        consecutive iterations.  It is a **heuristic** — it can stop
        before the true optimum — and stays opt-in; the default
        ``"fixed_point"`` is unchanged.  ``cost_fn`` is required for
        ``"improving"``.  Termination is reported in
        ``stats["stop"]`` (``"fixed_point"`` / ``"improving"`` /
        ``"max_nodes"`` / ``"max_iterations"``) and, under
        ``"improving"``, ``stats["improved"]`` counts the iterations
        that lowered the extracted cost.

        ``policy`` (optional) is a
        :class:`~catopt_core.ports.Policy` the schedule consults once
        per iteration.  It may only *reorder* the rules: every rule
        still runs, so the fixed point — and therefore the
        certificate — is unchanged (ADR 0003 invariant 5).  A policy
        that returns something outside the offered set has the
        remainder left in declared order, so no rule is ever dropped.
        The chosen ordering is recorded in ``stats["policy"]``.
        """
        if stop not in ("fixed_point", "improving"):
            raise ValueError(
                f"stop must be 'fixed_point' or 'improving', "
                f"got {stop!r}"
            )
        if stop == "improving" and cost_fn is None:
            raise ValueError(
                "stop='improving' needs a cost_fn to extract with"
            )
        ordered = list(rules)
        priority_of = getattr(rules, "priority_of", None)
        if priority_of is not None:
            # Stable sort — equal priorities keep declaration order.
            ordered.sort(key=priority_of)
        budgets = rule_budgets or {}
        spent = self._budget_spent
        for name in budgets:
            spent.setdefault(name, 0)
        stop_reason = "max_iterations"
        best_cost = float("inf")
        stall = 0
        improved = 0
        iteration = -1
        for iteration in range(max_iterations):
            n_before = self.n_enodes
            # Snapshot the frontier ONCE per iteration: classes changed
            # while rules fire are re-searched next iteration — the
            # fixed point is unchanged, the schedule is tighter.
            search = sorted({self.find(e) for e in self._dirty})
            self._dirty.clear()
            for rule in self._scheduled(
                ordered, policy, root_eid, iteration
            ):
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
                logger.debug(
                    "rule %s: %+d enodes",
                    rule.name,
                    self.n_enodes - n0,
                )
            self.rebuild(set(search) | self._dirty)
            n_after = self.n_enodes
            logger.debug(
                "iteration %d: %d -> %d enodes",
                iteration + 1,
                n_before,
                n_after,
            )
            if n_after >= max_nodes:
                logger.warning(
                    "  [egraph] stopping: max_nodes (%d) reached",
                    max_nodes,
                )
                stop_reason = "max_nodes"
                break
            if n_after == n_before and not self._dirty:
                logger.info(
                    "  [egraph] saturation at iteration %d",
                    iteration + 1,
                    extra={
                        "enodes": n_after,
                        "classes": self.n_classes,
                    },
                )
                stop_reason = "fixed_point"
                break
            if stop == "improving":
                # Lazy saturation: re-extract each iteration — the
                # shared cost memo (``_cost_memo_for``) makes repeat
                # extractions cheap on an only-just-grown graph.
                term = self.extract_best(root_eid, cost_fn)
                cost = (
                    cost_fn(term) if term is not None else float("inf")
                )
                if cost < best_cost:
                    best_cost = cost
                    stall = 0
                    improved += 1
                else:
                    stall += 1
                    if stall >= patience:
                        stop_reason = "improving"
                        break
        # Leave the graph in the same canonicalised postcondition the
        # unrestricted loop guaranteed (downstream passes and
        # extraction traverse it directly).
        self.rebuild()
        # The payload mixes counts with the ``rule_budgets`` map and the
        # ``budget_suspended`` list, so the return type is ``dict[str,
        # Any]``.
        stats: dict[str, Any] = {
            "iterations": iteration + 1,
            "n_enodes": self.n_enodes,
            "n_classes": self.n_classes,
            "n_proof_edges": len(self._merge_log),
            "truncation_level": self.truncation_level,
            "rule_budgets": {n: spent[n] for n in budgets},
            "budget_suspended": [
                n for n in budgets if spent[n] >= budgets[n]
            ],
            "stop": stop_reason,
        }
        if stop == "improving":
            stats["improved"] = improved
        stats["policy"] = _policy_name(policy)
        return stats

    # -- extraction --
