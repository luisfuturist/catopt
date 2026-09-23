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
    """
    name: str
    lhs: Any
    rhs: Any
    law: str = ""

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

    def add_term(self, term: Any) -> int:
        """Add a term (Var/Const/Param/Op) to the e-graph."""
        if isinstance(term, Op):
            child_eids = tuple(self.add_term(a) for a in term.args)
            attr_t = _pattern_attrs(term)
            enode = ENode(term.op, child_eids, attr_t)
            if enode in self._node_to_class:
                return self.find(self._node_to_class[enode])
            return self._add_enode(enode)
        else:
            _LeafRegistry.register(term)
            return self.add_leaf(repr(term))

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
                if node.op != pattern.op or node.attrs != attr_t:
                    continue
                if len(node.children) != len(pattern.args):
                    continue
                child_substs: list[dict[str, int]] = [{}]
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
                    for cs in child_substs:
                        full = dict(subst)
                        full.update(cs)
                        results.append(full)
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
            return self.add_enode(pattern.op, child_eids, dict(attr_t))
        else:
            return self.add_leaf(repr(pattern))

    def apply_rule(self, rule: Rewrite, root_eid: int) -> bool:
        """Apply a single rewrite rule across all e-classes."""
        changed = False
        for eid in list(self._classes.keys()):
            eid = self.find(eid)
            for subst in self.matches(rule.lhs, eid):
                rhs_eid = self._instantiate(rule.rhs, subst)
                if self.union(eid, rhs_eid):
                    changed = True
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

    def extract_best(self, eid: int, cost_fn) -> Any:
        """Extract the minimum-cost term from the e-class at *eid*."""
        cache: dict[int, tuple[float, Any]] = {}

        def best(eclass_id: int) -> tuple[float, Any]:
            eclass_id = self.find(eclass_id)
            if eclass_id in cache:
                return cache[eclass_id]
            eclass = self._classes[eclass_id]
            best_cost: float | None = None
            best_term: Any = None
            for node in eclass.nodes:
                if node.op == "leaf":
                    key = node.attrs[0][1] if node.attrs else "??"
                    term = _LeafRegistry.decode(key)
                    cost = cost_fn(term)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_term = term
                else:
                    child_terms = []
                    child_cost = 0.0
                    valid = True
                    for child_eid in node.children:
                        canon_child = self.find(child_eid)
                        if canon_child == eclass_id:
                            # Self-reference — skip this node
                            valid = False
                            break
                        ccost, cterm = best(canon_child)
                        if cterm is None:
                            valid = False
                            break
                        child_terms.append(cterm)
                        child_cost += ccost
                    if not valid:
                        continue
                    term = Op.make(node.op, *child_terms, **dict(node.attrs))
                    total = child_cost + cost_fn(term)
                    if best_cost is None or total < best_cost:
                        best_cost = total
                        best_term = term
            cache[eclass_id] = (best_cost or 0.0, best_term)
            return cache[eclass_id]

        _, term = best(eid)
        return term
