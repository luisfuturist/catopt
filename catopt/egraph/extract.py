"""Extraction mixin: cost-based + paired extraction."""
# ruff: noqa: RUF002 — math notation in comments
from __future__ import annotations

import logging
from typing import Any

from catopt.egraph.types import (
    ENode,
    _LeafRegistry,
    _pattern_attrs,
)
from catopt.ir import Op

logger = logging.getLogger("catopt.egraph.extract")


class _ExtractMixin:

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
                    _ctotal, cterm, cused, cpo, cnops = best(canon_child)
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

        total, term, _, _, nops = best(eid)
        if term is None:
            logger.warning(
                "extract_best: no extractable term in eclass %d",
                self.find(eid),
            )
        else:
            logger.info(
                "extract_best: eclass %d -> cost %.4g",
                self.find(eid),
                total,
                extra={"cost": total, "nops": nops},
            )
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
        return None  # pragma: no cover — pigeonhole: each ban removes a new enode

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
                for c, cc in zip(n.children, child_cls, strict=True)
            ):
                return eid, n
        return eid, None

