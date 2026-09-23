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
    check: Any = None   # Callable[[dict[str, Any]], bool] | None
    derive: Any = None  # Callable[[dict], dict | None] | None

    def __repr__(self) -> str:
        return f"{self.name}: {op_repr(self.lhs)} -> {op_repr(self.rhs)}"


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


class EGraph:
    """The equality-saturation data structure."""

    def __init__(self) -> None:
        self._uf = UnionFind()
        self._classes: dict[int, EClass] = {}
        self._node_to_class: dict[ENode, int] = {}
        self._next_id = 0
        self.rule_fires: dict[str, int] = {}

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

    def add_leaf(self, key: str) -> int:
        """Add a leaf (Var/Const/Param) identified by *key*."""
        enode = ENode("leaf", (), (("key", key),))
        if enode in self._node_to_class:
            return self.find(self._node_to_class[enode])
        return self._add_enode(enode)

    def add_enode(self, op: str, children: tuple[int, ...],
                  attrs: dict[str, Any] | None = None) -> int:
        """Add an ENode with already-resolved child e-class IDs."""
        attr_t = tuple(sorted((attrs or {}).items()))
        enode = ENode(op, tuple(self.find(c) for c in children), attr_t)
        if enode in self._node_to_class:
            return self.find(self._node_to_class[enode])
        return self._add_enode(enode)

    def _add_enode(self, enode: ENode) -> int:
        eid = self._next_id
        self._next_id += 1
        self._uf.parent.append(eid)
        self._uf.rank.append(0)
        self._classes[eid] = EClass(id=eid)
        self._node_to_class[enode] = eid
        self._classes[eid].nodes.add(enode)
        return eid

    def add_term(self, term: Any, _memo: dict | None = None) -> int:
        """Add a term (Var/Const/Param/Op) to the e-graph.

        ``_memo`` is an ``id()``-keyed cache: exported IR terms are DAGs
        with heavy sharing (residual streams, RoPE tables), and without
        memoisation the recursion re-walks shared subtrees
        exponentially.
        """
        memo = {} if _memo is None else _memo
        key = id(term)
        if key in memo:
            return memo[key]
        if isinstance(term, Op):
            child_eids = tuple(self.add_term(a, memo) for a in term.args)
            attr_t = _pattern_attrs(term)
            enode = ENode(term.op, child_eids, attr_t)
            if enode in self._node_to_class:
                memo[key] = self.find(self._node_to_class[enode])
            else:
                memo[key] = self._add_enode(enode)
            return memo[key]
        _LeafRegistry.register(term)
        memo[key] = self.add_leaf(repr(term))
        return memo[key]

    def union(self, a: int, b: int) -> bool:
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
            del self._classes[old_canon]
            return True
        return False

    # -- pattern matching --

    def matches(self, pattern: Any, eid: int) -> list[dict[str, int]]:
        """Find all substitutions that match *pattern* at e-class *eid*."""
        results: list[dict[str, int]] = []
        self._match(pattern, eid, {}, results)
        return results

    def _match(self, pattern: Any, eid: int,
               subst: dict[str, int],
               results: list[dict[str, int]]) -> None:
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
            for node in eclass.nodes:
                if node.op != pattern.op:
                    continue
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
                        child_results: list[dict[str, int]] = []
                        self._match(pat_arg, node.children[i],
                                    dict(cs), child_results)
                        new_substs.extend(child_results)
                    if not new_substs:
                        ok = False
                        break
                    child_substs = new_substs
                if ok:
                    # child_substs already contain the incoming bindings;
                    # conflicts were rejected inside the metavar branch.
                    results.extend(child_substs)
            return

        # Leaf (Const/Param/Var) — match by key
        key = ("key", repr(pattern))
        enode = ENode("leaf", (), (key,))
        if enode in self._node_to_class:
            if self.find(self._node_to_class[enode]) == eid:
                results.append(dict(subst))

    # -- rebuild --

    def rebuild(self) -> bool:
        """Canonicalize children and merge duplicates."""
        changed = False
        for eid in list(self._classes.keys()):
            eclass = self._classes[self.find(eid)]
            new_nodes: set[ENode] = set()
            for node in eclass.nodes:
                if node.children:
                    canon = tuple(self.find(c) for c in node.children)
                    if canon != node.children:
                        changed = True
                    new_nodes.add(ENode(node.op, canon, node.attrs))
                else:
                    new_nodes.add(node)
            eclass.nodes = new_nodes
        return changed

    # -- rule application --

    def _instantiate(self, pattern: Any, subst: dict[str, int]) -> int:
        """Instantiate a pattern (RHS) with a substitution."""
        if isinstance(pattern, str):
            return subst[pattern]
        if isinstance(pattern, Op):
            child_eids = tuple(self._instantiate(a, subst) for a in pattern.args)
            attr_t = _pattern_attrs(pattern)
            # Attribute metavariables (string values) resolve through the
            # substitution's "$attr:" namespace.
            attrs = {}
            for k, v in attr_t:
                if isinstance(v, str):
                    attrs[k] = subst.get("$attr:" + v, v)
                else:
                    attrs[k] = v
            return self.add_enode(pattern.op, child_eids, attrs)
        else:
            return self.add_leaf(repr(pattern))

    def any_term(self, eid: int, _seen: frozenset = frozenset()) -> Any:
        """Return any acyclic representative term of an e-class.

        Prefers leaf nodes; used to resolve metavariable bindings to
        concrete terms for rewrite side conditions (shape checks).
        """
        eid = self.find(eid)
        eclass = self._classes[eid]
        for node in eclass.nodes:
            if node.op == "leaf":
                key = node.attrs[0][1] if node.attrs else "??"
                return _LeafRegistry.decode(key)
        for node in eclass.nodes:
            args = []
            ok = True
            for c in node.children:
                canon = self.find(c)
                if canon == eid or canon in _seen:
                    ok = False
                    break
                t = self.any_term(canon, _seen | {eid})
                if t is None:
                    ok = False
                    break
                args.append(t)
            if ok:
                return Op.make(node.op, *args, **dict(node.attrs))
        return None

    def apply_rule(self, rule: Rewrite, root_eid: int) -> bool:
        """Apply a single rewrite rule across all e-classes."""
        changed = False
        for eid in list(self._classes.keys()):
            eid = self.find(eid)
            for subst in self.matches(rule.lhs, eid):
                if rule.check is not None or rule.derive is not None:
                    bound = {
                        k: (v if k.startswith("$attr:")
                            else self.any_term(v))
                        for k, v in subst.items()
                    }
                    if any(v is None for k, v in bound.items()
                           if not k.startswith("$attr:")):
                        continue
                    if rule.check is not None and not rule.check(bound):
                        continue
                    if rule.derive is not None:
                        extra = rule.derive(bound)
                        if extra is None:
                            continue
                        subst = {**subst, **extra}
                rhs_eid = self._instantiate(rule.rhs, subst)
                if self.union(eid, rhs_eid):
                    changed = True
                    self.rule_fires[rule.name] = (
                        self.rule_fires.get(rule.name, 0) + 1)
        return changed

    # -- saturation --

    def run(self, rules: list[Rewrite], root_eid: int,
            max_iterations: int = 100,
            max_nodes: int = 100_000) -> dict[str, int]:
        """Run equality saturation until a fixed point."""
        for iteration in range(max_iterations):
            n_before = self.n_enodes
            for rule in rules:
                self.apply_rule(rule, root_eid)
            self.rebuild()
            n_after = self.n_enodes
            if n_after >= max_nodes:
                print(f"  [egraph] stopping: max_nodes ({max_nodes}) reached")
                break
            if n_after == n_before:
                print(f"  [egraph] saturation at iteration {iteration + 1}")
                break
        return {
            "iterations": iteration + 1,
            "n_enodes": self.n_enodes,
            "n_classes": self.n_classes,
        }

    # -- extraction --

    def extract_alternatives(self, eid: int, cost_fn,
                             top_k: int = 8) -> list[tuple[float, Any]]:
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
            term = self.extract_best(eid, cost_fn,
                                     overrides={eid: node})
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
        from catopt.ir import op_repr
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
                        else "?" for c in n.children]
                    sk = f"{n.op}({','.join(child_ops)})"
                if sk not in seen_sketch:
                    seen_sketch.add(sk)
                    sketches.append(sk)
            out.append({"eid": eid, "members": sketches})
        out.sort(key=lambda d: -len(d["members"]))
        return out

    def extract_best(self, eid: int, cost_fn,
                     overrides: dict[int, Any] | None = None,
                     _cache_out: dict | None = None) -> Any:
        """Extract the minimum-cost term from the e-class at *eid*.

        ``overrides`` maps canonical e-class ids to a specific ENode:
        extraction is then forced to use that enode for those classes.
        This is how non-local rewrites (the diagram-level product rule)
        get *coordinated* extraction — per-class greedy choice cannot see
        that k members each selecting `split_i(fused)` share ONE fused
        GEMM, since each split's subtree alone looks more expensive than
        the member's own linear.

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
        # keepalive retains every candidate term so id() keys in the
        # memo cannot be recycled by the GC mid-extraction.
        import inspect
        cost_memo: dict = {}
        keepalive: list = []
        takes_memo = "memo" in inspect.signature(cost_fn).parameters

        def cfn(t: Any) -> float:
            return cost_fn(t, memo=cost_memo) if takes_memo else cost_fn(t)

        def best(eclass_id: int) -> tuple[float, Any, frozenset, bool, int]:
            eclass_id = self.find(eclass_id)
            if eclass_id in cache:
                return cache[eclass_id]
            if eclass_id in in_progress:
                # Cycle back to an ancestor — not extractable.
                return (float("inf"), None, frozenset(), False, 0)
            in_progress.add(eclass_id)
            eclass = self._classes[eclass_id]
            override = overrides.get(eclass_id) if overrides else None
            nodes = (override,) if override is not None else eclass.nodes
            best_total: float | None = None
            best_term: Any = None
            best_used: frozenset = frozenset({eclass_id})
            best_local = 0.0
            best_param_only = False
            best_nops = 0
            for node in nodes:
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
                    # Compile-time (param-only) classes are free.
                    for u in cused - used:
                        if not param_only_of.get(u, False):
                            sub_cost += local_of.get(u, 0.0)
                    used |= cused
                if not valid:
                    continue
                term = Op.make(node.op, *child_terms, **dict(node.attrs))
                keepalive.append(term)
                local = cfn(term) - sum(
                    cfn(c) for c in child_terms
                )
                local = max(local, 0.0)
                if param_only:
                    local = 0.0  # whole subtree folds at compile time
                total = local + sub_cost
                # Secondary key: among equal-cost candidates prefer the
                # structurally smallest term (a leaf over add(W, 0) in a
                # param-only class, for example).  Counted incrementally
                # from child caches — no tree walk.
                nops = child_nops + 1
                if (best_total is None
                        or total < best_total
                        or (total == best_total and nops < best_nops)):
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
                best_total or 0.0, best_term, best_used, best_param_only,
                best_nops,
            )
            return cache[eclass_id]

        _, term, _, _, _ = best(eid)
        if _cache_out is not None:
            _cache_out.update(cache)
        return term

    # -- coordinated (group) extraction ----------------------------------

    def extract_paired(self, root_eid: int, cost_fn,
                       groups: list[dict[int, Any]]) -> Any:
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
            return cost_fn(t, memo=cost_memo) if takes_memo else cost_fn(t)

        pass1_cache: dict = {}
        self.extract_best(root_eid, cost_fn, overrides=member_over,
                          _cache_out=pass1_cache)

        keepalive: list = []

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
            keepalive.append(term)  # id()-keyed memo: prevent GC reuse
            local = cfn(term) - sum(cfn(c) for c in child_terms)
            return max(local, 0.0) + sub

        overrides: dict[int, Any] = dict(member_over)
        for cid, ec in list(self._classes.items()):
            cid = self.find(cid)
            if cid in member_over or len(ec.nodes) < 2:
                continue
            member_reaching = [n for n in ec.nodes
                               if enode_reaches_member(n)]
            if member_reaching and len(member_reaching) < len(ec.nodes):
                # class has both member-reaching and bypassing enodes —
                # force the cheapest member route so the shared GEMM
                # is used without dragging in junk subtrees
                overrides[cid] = min(member_reaching, key=steered_score)

        return self.extract_best(root_eid, cost_fn, overrides=overrides)


def _iter_ops(term: Any):
    """Yield every Op node in a term (for the structural-size tie-break)."""
    if isinstance(term, Op):
        yield term
        for a in term.args:
            yield from _iter_ops(a)
