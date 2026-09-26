"""Core e-graph: union-find, matching, saturation."""
from __future__ import annotations

import logging
from typing import Any

from catopt_core.egraph.certs import (
    ProofEdge,
)
from catopt_core.egraph.extract import _ExtractMixin
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
        attr_t = tuple(
            sorted(
                (k, _norm_attr_value(v)) for k, v in (attrs or {}).items()
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
        """Synthesise the pointwise :class:`Rewrite` certifying one
        non-locally offered member — see :meth:`_offer_witness`.

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
        """Offer a non-locally-computed member under a pointwise
        witness — the canonical ritual behind every non-local pass.

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
            rhs_eid = self.add_term(rhs_term, provenance=term_provenance)
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
                    return  # pragma: no cover — limit reached only inside trailing extend
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
        if (
            enode in self._node_to_class
            and self.find(self._node_to_class[enode]) == eid
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
                tuple(
                    sorted(
                        (k, _norm_attr_value(v)) for k, v in attrs.items()
                    )
                ),
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

