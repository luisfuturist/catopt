"""Extraction mixin: cost-based + paired extraction."""

# ruff: noqa: RUF002 — math notation in comments
from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING, Any, cast

from catopt_core.cost import (
    _FOLDABLE_ELEMWISE,
    _SOLVER_FACTOR,
    _SOLVER_OPS,
    _local_roofline,
    _memo_dispatch,
    _profile_constants,
    dag_cost,
    fusion_member_key,
)
from catopt_core.egraph.types import (
    ENode,
    _LeafRegistry,
    _pattern_attrs,
)
from catopt_core.ir import Op, Var

if TYPE_CHECKING:
    from catopt_core.egraph.certs import Certificate
    from catopt_core.egraph.types import EClass

logger = logging.getLogger("catopt_core.egraph.extract")

#: Ops ``IRModule._fold_weight_chains`` actually folds into a
#: materialised parameter (mirrors ``cost._folds_to_param``).  The
#: param-only discount applies only to classes whose extracted subtree
#: is built from these — a param-only ``trace``/``inv`` subtree is NOT
#: foldable: it evaluates a solver call at runtime, so it must be
#: billed.  (The e-class level check approximates the per-arg Const
#: rules — see the function for the exact contract.)
_FOLDABLE_OPS = _FOLDABLE_ELEMWISE | {"matmul", "concat"}

#: Largest number of pairing groups for which
#: :meth:`_ExtractMixin.extract_paired` enumerates every ``{0,1}^G``
#: fuse/decline subset (``2**G`` extractions).  The coordination space
#: is small in practice (measured 1-3, the whole gap lives there), so
#: the exhaustive decision is exact for every measured instance; beyond
#: the cap the per-group decision degrades to the bounded
#: :meth:`_ExtractMixin._greedy_paired` pass.
_PAIRING_EXHAUSTIVE_MAX = 3

#: Extraction budget for the :meth:`_ExtractMixin._greedy_paired`
#: fallback: at most this many single-group removals are tried on top of
#: the all-groups-forced baseline.  Keeps the fallback cheap when a graph
#: has many pairing groups (a saturated transformer block can reach ~90,
#: where an O(G) sweep would cost ~90 extractions).
_PAIRING_GREEDY_BUDGET = 4


class _ExtractMixin:
    if TYPE_CHECKING:
        # Interface supplied by ``EGraph`` (and the proof mixin) once
        # the mixins are combined — declared here so ``self``
        # type-checks.  Runtime never executes this block.
        _classes: dict[int, EClass]
        _node_to_class: dict[ENode, int]

        @property
        def n_enodes(self) -> int: ...

        def find(self, eid: int) -> int: ...

        def any_term(
            self,
            eid: int,
            _seen: frozenset = frozenset(),
            _memo: dict | None = None,
        ) -> Any: ...

        def certificate(
            self,
            src_term: Any,
            dst_term: Any = None,
            *,
            root_eid: int | None = None,
            cost_fn: Any = None,
        ) -> Certificate: ...

        def _oldest_term(
            self,
            eid: int,
            _stack: frozenset = frozenset(),
            _memo: dict | None = None,
        ) -> Any: ...

    def _cost_memo_for(self, cost_fn) -> dict:
        """Shared content-keyed cost memo for *cost_fn* on this e-graph.

        Cost values are pure functions of ``(cost_fn, term)`` — terms
        are interned — so one dict can serve every extraction pass
        (greedy, forced/coordinated re-extractions, bounded rounds)
        instead of repricing each distinct candidate term per call.
        Entries are per cost_fn object: a different cost model gets a
        fresh dict.  The stored pair pins the callable so an
        ``id()``-recycled different function can never inherit a stale
        memo.
        """
        memos = getattr(self, "_cost_memos", None)
        if memos is None:
            memos = self._cost_memos = {}
        ent = memos.get(id(cost_fn))
        if ent is None or ent[0] is not cost_fn:
            ent = (cost_fn, {})
            memos[id(cost_fn)] = ent
        return ent[1]

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
        """Enumerate the root e-class frontier.

        For each non-leaf enode, force extraction through it and record
        the resulting term's DAG cost.  Returns the top-k cheapest
        *distinct* alternatives — i.e. the cheapest members of the
        semantic equivalence class [G], which is what a discovery
        engine inspects for unexpected candidates.
        """
        from catopt_core.cost import dag_cost
        from catopt_core.ir import op_repr

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
        sketch of every distinct member.
        """
        out: list[dict[str, Any]] = []
        for eid, ec in self._classes.items():
            ops = {n.op for n in ec.nodes if n.op != "leaf"}
            if len(ops) < 2:
                continue
            sketches: list[str] = []
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
        fusion_epsilon: float = 0.0,
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
        (``catopt_core.cost.param_bytes_cost``) opt out of that discount by
        setting ``charges_param_only`` on the function — a folded
        subtree still stores its leaves' values.

        ``fusion_epsilon`` > 0 enables the fusion-preferred tie-break
        (:meth:`extract_fused` passes 0.05): members whose totals land
        within ``fusion_epsilon`` (relative to the class minimum) are
        near-cost-equal under this model, so they compete on the
        fusion key — ``(len(fusion_regions(term)),
        not exposes_pointwise, nops)`` via
        :func:`catopt_core.cost.fusion_member_key` — instead of
        structural size alone.  The compiled lowering collapses
        pointwise clusters into single kernels, so among near-ties the
        member with fewer predicted regions / a fusible root is the
        one that actually runs cheaper under ``torch.compile``.  The
        probe reuses the shared cost memo — each distinct candidate
        term is region-counted once across all extraction passes —
        and only fires when a class has more than one in-band member.
        ``fusion_epsilon=0`` (default) disables the pass entirely and
        selection is unchanged.

        Cyclic nodes (a class reachable from itself through rewrite-
        introduced unions) are skipped: they cannot be extracted.
        """
        # eid -> (total_cost, term, used_eclass_ids, subtree_is_param_only,
        #         nops, cfn(term), eo_overhead, roofline_cost)
        # The last three entries are sub-cost PROBES used to pre-seed the
        # shared cost memo before pricing a parent candidate (see the
        # seeding note below); ``None`` when the cost model did not
        # produce the corresponding memo key for this class's term.
        cache: dict[int, tuple] = {}
        # eid -> billable local contribution of the class's chosen
        # member: ``local`` unless the param-only discount applies.
        # Replaces the separate local_of/param_only_of lookups so the
        # shared-class billing below is a single dict probe per class.
        adj_of: dict[int, float] = {}
        in_progress: set[int] = set()

        # Shared cost/shape memo — per cost_fn, across ALL extractions
        # on this e-graph (``_cost_memo_for``): repeated passes price
        # each distinct term once, total.  Terms are content-hashed and
        # interned — memos key on the term object directly and hold it
        # alive; no keepalive needed.
        # Storage-style cost models (param_bytes_cost) bill Param leaves
        # — folding does not shrink the weights file — so the param-only
        # discount does not apply.  getattr(..., "func", ...) unwraps
        # functools.partial bindings of the flagged function.
        bill_params = getattr(
            getattr(cost_fn, "func", cost_fn),
            "charges_param_only",
            False,
        )
        m = self._cost_memo_for(cost_fn)
        cfn = _memo_dispatch(cost_fn, m)

        # Memo pre-seeding for the two additive sub-models the executor
        # cost family prices with: ``_generic_overhead`` counts op
        # occurrences under ``("eo","generic",term)`` — additive by
        # construction (eo = w(op) + Σ eo(children), integer-valued so
        # order-exact) — but computes it by re-walking the whole
        # subtree, ignoring the memo for children; ``_roofline_cost``
        # under ``("rc",pf,bw,ls,term)`` recurses with the memo, one
        # call per argument.  When every child's probe value is present
        # (which happens exactly when the cost model prices through
        # those keys — the check is self-gating: other cost fns simply
        # leave the probes ``None`` and seeding never fires), the
        # parent's entry is the child values summed under the same
        # local term — turning each fresh ``cfn(term)`` into O(arity)
        # memo hits instead of an O(subtree) re-walk.  Seeded values
        # are identical: the eo count is integer-exact in any order,
        # and rc replays ``local + Σchildren`` in the same order the
        # model itself accumulates.
        unwrapped = getattr(cost_fn, "func", cost_fn)
        rc_enabled = hasattr(unwrapped, "profile")
        if rc_enabled:
            try:
                rc_pf, rc_bw, rc_ls = _profile_constants(
                    getattr(unwrapped, "profile", None)
                )
            except (AttributeError, TypeError, KeyError):
                # A user cost_fn carrying an unrelated ``.profile``
                # attribute — roofline seeding is self-gating anyway,
                # so disabling it loses only the fast path.
                rc_enabled = False
                rc_pf = rc_bw = rc_ls = None
        else:
            rc_pf = rc_bw = rc_ls = None

        def best(eclass_id: int) -> tuple:
            eclass_id = self.find(eclass_id)
            if eclass_id in cache:
                return cache[eclass_id]
            if eclass_id in in_progress:
                # Cycle back to an ancestor — not extractable.
                return (
                    float("inf"),
                    None,
                    frozenset(),
                    False,
                    0,
                    0.0,
                    None,
                    None,
                )
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
                else (
                    sorted(
                        eclass.nodes,
                        key=lambda n: (n.op, n.children, repr(n.attrs)),
                    )
                    if len(eclass.nodes) > 1
                    else tuple(eclass.nodes)
                )
            )
            banned = bans.get(eclass_id) if bans else None
            best_total: float | None = None
            best_term: Any = None
            best_used: frozenset = frozenset({eclass_id})
            best_local = 0.0
            best_ct = 0.0
            best_param_only = False
            best_nops = 0
            # Every valid candidate, only kept when the fusion
            # tie-break is armed: (total, term, used, local, cfn(term),
            # param_only, nops) — the same fields the incremental
            # winner slots carry, so the post-loop pick can re-seat a
            # near-tie winner without re-walking children.
            cands: list[tuple] = []
            for node in nodes:
                if banned and node in banned:
                    continue  # excluded enode (extract_best_bounded)
                if node.op == "leaf":
                    key = node.attrs[0][1] if node.attrs else "??"
                    term = _LeafRegistry.decode(key)
                    total = cfn(term)
                    if fusion_epsilon:
                        cands.append(
                            (
                                total,
                                term,
                                frozenset({eclass_id}),
                                total,
                                total,
                                not isinstance(term, Var),
                                0,
                            )
                        )
                    if best_total is None or total < best_total:
                        best_total = total
                        best_term = term
                        best_used = frozenset({eclass_id})
                        best_local = total
                        best_ct = total
                        best_param_only = not isinstance(term, Var)
                        best_nops = 0
                    continue

                child_terms: list[Any] = []
                child_rcs: list[Any] = []
                child_tc = 0.0
                child_nops = 0
                eo_sum = 0.0
                eo_ok = True
                rc_ok = rc_enabled
                used: set[int] = {eclass_id}
                sub_cost = 0.0
                param_only = True
                valid = True
                for child_eid in node.children:
                    # Cached entries are keyed by canonical id, so a
                    # hit means child_eid was already canonical — and
                    # cannot be eclass_id itself (in-progress classes
                    # are never cached).
                    entry = cache.get(child_eid)
                    if entry is None:
                        canon_child = self.find(child_eid)
                        if canon_child == eclass_id:
                            valid = False  # direct self-reference
                            break
                        entry = best(canon_child)
                    (
                        _ctotal,
                        cterm,
                        cused,
                        cpo,
                        cnops,
                        ctc,
                        ceo,
                        crc,
                    ) = entry
                    if cterm is None:
                        valid = False
                        break
                    child_terms.append(cterm)
                    if rc_enabled:
                        child_rcs.append(crc)
                    child_tc += ctc
                    child_nops += cnops
                    param_only = param_only and cpo
                    eo_sum += ceo or 0.0
                    eo_ok = eo_ok and ceo is not None
                    rc_ok = rc_ok and crc is not None
                    # Charge each distinct e-class in the DAG once:
                    # a shared child contributes its subtree cost only
                    # for the classes not already accounted for.
                    # Compile-time (param-only) classes are free —
                    # unless the cost model prices storage, in which
                    # case every class's local is billed.  adj_of is
                    # that billable local — zeroed by the discount.
                    new = cused - used
                    if new:
                        # Sequential += in diff-set order — identical
                        # association to billing per element (adding
                        # adj_of's 0.0s is an exact no-op).
                        for u in new:
                            sub_cost += adj_of.get(u, 0.0)
                        used |= new
                if not valid:
                    continue
                term = Op.make(
                    node.op, *child_terms, **dict(node.attrs)
                )
                # Pre-seed the additive sub-model memo entries — see
                # the note above.  Self-gating: fires only when the
                # cost model produced the matching keys on children.
                if eo_ok:
                    ek = ("eo", "generic", term)
                    if ek not in m:
                        m[ek] = eo_sum + (
                            _SOLVER_FACTOR
                            if node.op in _SOLVER_OPS
                            else 1.0
                        )
                if rc_ok:
                    rk = ("rc", rc_pf, rc_bw, rc_ls, term)
                    if rk not in m:
                        rc = _local_roofline(
                            term,
                            m,
                            peak_flops=cast("float", rc_pf),
                            peak_bw=cast("float", rc_bw),
                            launch_s=cast("float", rc_ls),
                        )
                        # Same accumulation order as _roofline_cost:
                        # local first, then children left-to-right.
                        for c in child_rcs:
                            rc += c
                        m[rk] = rc
                # …and this node's own op must be foldable — a
                # param-only subtree over trace/inv still runs its
                # solver call per eval, so it is billed.
                param_only = param_only and node.op in _FOLDABLE_OPS
                ct = cfn(term)
                local = ct - child_tc
                local = max(local, 0.0)
                if param_only and not bill_params:
                    local = 0.0  # whole subtree folds at compile time
                total = local + sub_cost
                # Secondary key: among equal-cost candidates prefer the
                # structurally smallest term (a leaf over add(W, 0) in a
                # param-only class, for example).  Counted incrementally
                # from child caches — no tree walk.
                nops = child_nops + 1
                if fusion_epsilon:
                    cands.append(
                        (
                            total,
                            term,
                            used,
                            local,
                            ct,
                            param_only,
                            nops,
                        )
                    )
                if (
                    best_total is None
                    or total < best_total
                    or (total == best_total and nops < best_nops)
                ):
                    best_total = total
                    best_term = term
                    best_used = frozenset(used)
                    best_local = local
                    best_ct = ct
                    best_param_only = param_only
                    best_nops = nops
            if fusion_epsilon and len(cands) > 1:
                # Fusion-preferred near-tie: members priced within
                # fusion_epsilon of the class minimum are cost-
                # indistinguishable under this model's resolution, so
                # pick the one the compiled lowering runs cheapest —
                # fewest predicted kernels (fusion_regions), then a
                # root a pointwise consumer could absorb, then fewest
                # ops.  Region probes reuse the shared memo `m`, so
                # each distinct candidate is counted once ever; the
                # band is anchored at the minimum, making the pick
                # order-independent (cands is already in the canonical
                # sorted member order, so equal keys stay stable).
                floor = min(c[0] for c in cands)
                band = [
                    c
                    for c in cands
                    if c[0] <= floor + fusion_epsilon * abs(floor)
                ]
                if len(band) > 1:
                    win = min(
                        band,
                        key=lambda c: (
                            *fusion_member_key(c[1], m),
                            c[6],
                        ),
                    )
                    (
                        best_total,
                        best_term,
                        best_used,
                        best_local,
                        best_ct,
                        best_param_only,
                        best_nops,
                    ) = (
                        win[0],
                        win[1],
                        frozenset(win[2]),
                        win[3],
                        win[4],
                        win[5],
                        win[6],
                    )
            in_progress.discard(eclass_id)
            if best_term is None:
                best_total = float("inf")
            adj_of[eclass_id] = (
                best_local
                if (bill_params or not best_param_only)
                else 0.0
            )
            # Sub-cost probes for memo pre-seeding (see the note above):
            # read AFTER cfn(best_term) ran, so an eo/rc-pricing cost
            # model has written the keys this reads back.
            ceo = (
                m.get(("eo", "generic", best_term))
                if best_term is not None
                else None
            )
            crc = (
                m.get(("rc", rc_pf, rc_bw, rc_ls, best_term))
                if (rc_enabled and best_term is not None)
                else None
            )
            cache[eclass_id] = (
                best_total or 0.0,
                best_term,
                best_used,
                best_param_only,
                best_nops,
                best_ct,
                ceo,
                crc,
            )
            return cache[eclass_id]

        total, term, _, _, nops, _, _, _ = best(eid)
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

    def extract_fused(
        self,
        eid: int,
        cost_fn,
        *,
        fusion_epsilon: float = 0.05,
        overrides: dict[int, Any] | None = None,
        bans: dict[int, set] | None = None,
        _cache_out: dict | None = None,
    ) -> Any:
        """Extract preferring fusion-optimal members among near-ties.

        :meth:`extract_best` with ``fusion_epsilon`` armed (default
        5%): within each e-class, members priced inside the band
        compete on :func:`catopt_core.cost.fusion_member_key` —
        predicted kernel count of the extracted member subtree, then
        whether its root can merge into a pointwise consumer's
        region — rather than structural size.  This is the
        extraction-side counterpart of the compiled criterion: local
        per-class pricing cannot see that two near-cost-equal members
        differ by a kernel launch once regions collapse, so the
        tie-break picks the member the ``"compiled"`` lowering would
        fuse best.

        ``overrides``/``bans``/``_cache_out`` pass straight through to
        :meth:`extract_best`; ``fusion_epsilon=0`` reproduces plain
        extraction exactly.
        """
        return self.extract_best(
            eid,
            cost_fn,
            overrides=overrides,
            bans=bans,
            _cache_out=_cache_out,
            fusion_epsilon=fusion_epsilon,
        )

    def extract_best_bounded(
        self,
        eid: int,
        cost_fn,
        max_error: float | None = None,
        *,
        src_term: Any = None,
        _cache_out: dict | None = None,
    ) -> Any:
        """Extract the minimum-cost member certifying within ``max_error``.

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
        """Extract with the pairing-group decision made *per group*.

        Per-class greedy extraction cannot express the product law's
        non-local choice: member class C_i containing both
        ``linear(x, W_i)`` and ``split_i(fused)`` sees the split's
        subtree cost as the FULL fused GEMM, which always loses locally.
        The fused form only wins when *all* members of a group take it —
        and additionally when every consumer class routes through the
        member classes rather than a specialized-fusion alternative
        (e.g. an ``sdpa`` enode built over rule-introduced ``chunk``
        terms).

        A single "force every group" decision is *all-or-nothing*: when
        one group's fusion is unprofitable (an ``add``-consumed group
        gains a weight-merge bypass from ``WEIGHT_FACTOR_LINEAR``) and
        another's is profitable, forcing both is a net loss while
        forcing neither misses the profitable one.  So this enumerates
        the per-group decision ``{0,1}^G`` — every **non-empty** subset
        of groups is forced through :meth:`_paired_subset_term` (which
        also steers consumers through the forced members) and the
        cheapest true DAG-cost term wins.  The empty subset is left to
        the caller's greedy-vs-forced comparison, so ``extract_paired``
        keeps meaning "the best *coordinated* term".

        ``G`` is small in practice (measured 1-3), so ``2**G`` is
        enumerated exhaustively up to ``_PAIRING_EXHAUSTIVE_MAX``;
        beyond the cap the per-group decision degrades to the bounded
        :meth:`_greedy_paired` pass.  The result is never worse than the
        all-groups-forced term the previous all-or-nothing policy
        returned — that term is always one of the candidates.
        """
        per_group: list[dict[int, Any]] = []
        for g in groups:
            over: dict[int, Any] = {}
            for cid, enode in g.items():
                over.setdefault(self.find(cid), enode)
            per_group.append(over)
        n = len(per_group)
        if n == 0:
            return self.extract_best(root_eid, cost_fn)
        memo = self._cost_memo_for(cost_fn)
        best: tuple[float, Any] = (float("inf"), None)
        if n <= _PAIRING_EXHAUSTIVE_MAX:
            for r in range(1, n + 1):
                for comb in itertools.combinations(range(n), r):
                    best = self._score_paired(
                        root_eid, cost_fn, per_group, comb, memo, best
                    )
        else:
            best = self._greedy_paired(
                root_eid, cost_fn, per_group, memo, best
            )
        return best[1]

    def _score_paired(
        self,
        root_eid: int,
        cost_fn,
        per_group: list[dict[int, Any]],
        comb: tuple[int, ...],
        memo: dict,
        best: tuple[float, Any],
    ) -> tuple[float, Any]:
        """Force *comb*'s groups; keep the cheaper of *best* and it."""
        merged: dict[int, Any] = {}
        for i in comb:
            merged.update(per_group[i])
        term = self._paired_subset_term(root_eid, cost_fn, merged)
        cost = (
            dag_cost(term, cost_fn, memo=memo)
            if term is not None
            else float("inf")
        )
        if cost < best[0]:
            return (cost, term)
        return best

    def _greedy_paired(
        self,
        root_eid: int,
        cost_fn,
        per_group: list[dict[int, Any]],
        memo: dict,
        best: tuple[float, Any],
    ) -> tuple[float, Any]:
        """Bounded per-group fallback for large ``G``.

        Starts from the all-groups-forced decision — the previous
        all-or-nothing policy, so the result can never be worse than it
        — then tries single-group removals for the first
        ``_PAIRING_GREEDY_BUDGET`` groups, keeping the best.  Constant
        extraction cost, so the ``{0,1}^G`` decision stays cheap past
        ``_PAIRING_EXHAUSTIVE_MAX`` (a saturated transformer block can
        reach ``G`` ~90, where an O(G) sweep would cost ~90 extractions
        against the 2 the previous policy spent).
        """
        n = len(per_group)
        best = self._score_paired(
            root_eid, cost_fn, per_group, tuple(range(n)), memo, best
        )
        for i in range(min(n, _PAIRING_GREEDY_BUDGET)):
            cand = tuple(j for j in range(n) if j != i)
            if not cand:
                continue  # keep >=1 group: the empty set is the caller's
            best = self._score_paired(
                root_eid, cost_fn, per_group, cand, memo, best
            )
        return best

    def _paired_subset_term(
        self, root_eid: int, cost_fn, member_over: dict[int, Any]
    ) -> Any:
        """Force *member_over*'s members to their splits and steer.

        The forced extraction for one group-subset: (1) each member
        class is overridden to its split enode; (2) every other class
        with multiple enodes is overridden to an enode whose
        descendants reach a member class, when one exists — steering
        consumers through the shared GEMM.  The caller prices the true
        DAG cost against the alternatives.
        """
        member_classes: set[int] = set(member_over)

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
        cfn_memo = self._cost_memo_for(cost_fn)
        cfn = _memo_dispatch(cost_fn, cfn_memo)

        pass1_cache: dict = {}
        self.extract_best(
            root_eid,
            cost_fn,
            overrides=member_over,
            _cache_out=pass1_cache,
        )

        def steered_score(node: Any) -> float:
            """Local cost + children best totals (member-routed pass)."""
            child_terms = []
            sub = 0.0
            child_tc = 0.0
            eo_sum = 0.0
            eo_ok = True
            for ch in node.children:
                entry = pass1_cache.get(self.find(ch))
                if entry is None or entry[1] is None:
                    return float("inf")
                child_terms.append(entry[1])
                sub += entry[0]
                child_tc += entry[5]  # cfn(child's chosen term)
                ceo = entry[6]
                eo_ok = eo_ok and ceo is not None
                eo_sum += ceo or 0.0
            term = Op.make(node.op, *child_terms, **dict(node.attrs))
            # Same additive eo pre-seed as extract_best — keeps each
            # steering probe an O(arity) evaluation.
            if eo_ok:
                ek = ("eo", "generic", term)
                if ek not in cfn_memo:
                    cfn_memo[ek] = eo_sum + (
                        _SOLVER_FACTOR
                        if node.op in _SOLVER_OPS
                        else 1.0
                    )
            local = cfn(term) - child_tc
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
