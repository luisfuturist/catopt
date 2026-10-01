"""Cross-block CSE — a shared subterm computed once for a family.

Plan-0011 sibling law (same plumbing as
:func:`catopt_core.laws.factored.offer_low_rank_factors`, one level
up).  The e-graph already hash-conses shared subterms *inside* one
block — or inside one monolithic export — so the gap this law
covers is two *sibling* blocks recomputing the identical
computation on the same input object: the shared-norm /
shared-projection recompute of multi-branch and parallel-block
models (``y = x + A(x) + B(x)`` where ``A`` and ``B`` both open
with ``norm(x)`` / ``x @ W``).

Detection has two halves, both evidence-based:

* **reachability** — :func:`~catopt_orchestrator.morphisms_kv._input_families`
  groups blocks whose captured input is literally the same tensor
  object (``in_obj`` identity, single call, lifted IR).  Only then is
  the producer's intermediate provably the same value the consumer
  recomputes: same input object, identical op tree, leaf-equal
  parameters.  Sequential residual-stream blocks see *different*
  stream values — ``s_j`` is a fresh sum, and the addend
  ``f_i(s_i)`` is never separately consumable at ``j`` — so adjacent
  chain/residual pairs produce no candidates by construction.  The
  honest scope is exactly the same-input family; a general
  cross-sequence CSE is documented non-reach, not a silent
  assumption.
* **structural equality modulo leaf renaming** —
  :func:`_same_computation` compares subterm trees op-by-op:
  ``Var``/``Const`` leaves must be literally equal, ``Param`` leaves
  may differ in name only when the leaf table certifies
  bitwise-equal values (the ``share_duplicate_params`` convention) —
  a shared ``nn.Parameter`` read through two sibling modules lands
  here, as do built-identical weight copies.

Delivery is the only rewire the slot-graft machinery supports — the
additive family fusion of ``kv_latent_share``: the members' outputs
must add into a consumed value (the ``_family_evidence`` wiring
checks on both capture probes, plus the fan-out guard), the fused
first slot computes ``Σ b_i(x)`` as ONE term — where the shared
subterm is a single interned DAG node evaluated once by the lowered
evaluator — and the consumed slots become exact-zero fillers.
Blocks whose outputs are opaque to that evidence (multiplied,
concatenated, fanned out non-additively) get no offer.

The rewritten joint is offered into the joint e-graph under a
pointwise witness (``error_bound=0.0`` — the equality is exact by
construction), saturated under the compose recipe, cost-gated on
true DAG cost, and fp64-verified through the sink before any slot
is grafted — a rewrite that cannot be certified is a decline.
"""

from __future__ import annotations

from typing import Any

from catopt_core.ir import Op, Param, op_repr
from catopt_core.laws.pairing import _exact_equal, _is_tensor
from catopt_core.ports import CostFn, Sink
from catopt_core.typing import has_var_leaf

import catopt_orchestrator.morphisms as M
from catopt_orchestrator.morphisms_kv import (
    _add_chain,
    _family_bodies,
    _family_evidence,
    _family_gate,
    _family_prep,
    _family_tables,
    _input_families,
    _replace_nodes,
)

__all__ = ["CrossBlockCSE"]


# ---------------------------------------------------------------------------
#  Structural equality modulo provable leaf renaming
# ---------------------------------------------------------------------------


def _leaf_same(a: Any, b: Any, leaves: dict) -> bool:
    """Leaf equality under renaming — value-certified for ``Param``.

    ``Var``/``Const``/other leaves must be literally equal (the shared
    input var is the same object in every prefixed body); a ``Param``
    may differ in name only when the leaf table certifies equal values
    — the exact-equality convention ``share_duplicate_params`` uses.
    A param whose value is absent from the table is unprovable.
    """
    if a is b or a == b:
        return True
    if not (isinstance(a, Param) and isinstance(b, Param)):
        return False
    va = leaves.get(a.name)
    vb = leaves.get(b.name)
    if va is None or vb is None:
        return False
    if va is vb:
        return True
    if _is_tensor(va) and _is_tensor(vb):
        return _exact_equal(va, vb)
    eq = va == vb
    return eq if isinstance(eq, bool) else False


def _same_computation(a: Any, b: Any, leaves: dict, memo: dict) -> bool:
    """Structural equality of two subterms modulo provable leaves.

    Ops must agree on name, attrs and arity, recursively; leaves defer
    to :func:`_leaf_same`.  ``memo`` is keyed on the term pair — terms
    are interned/immutable, so the DAG walk stays linear.
    """
    key = (a, b)
    hit = memo.get(key)
    if hit is not None:
        return hit
    if a is b or a == b:
        res = True
    elif isinstance(a, Op) and isinstance(b, Op):
        res = (
            a.op == b.op
            and a.attrs == b.attrs
            and len(a.args) == len(b.args)
            and all(
                _same_computation(x, y, leaves, memo)
                for x, y in zip(a.args, b.args, strict=True)
            )
        )
    else:
        res = _leaf_same(a, b, leaves)
    memo[key] = res
    return res


# ---------------------------------------------------------------------------
#  Shared-subterm scan over a same-input family
# ---------------------------------------------------------------------------


def _op_nodes(term: Any) -> list[Any]:
    """Var-reaching ``Op`` nodes of a body — the share candidates."""
    return [n for n in M._iter_ops(term) if has_var_leaf(n)]


def _family_shares(
    bodies: dict[str, Any], order: list[str], leaves: dict
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Shared subterms per member plus the producer->consumer links.

    Walks each member's body against the *earlier* members' op nodes
    (execution order — the producer must precede): a node whose whole
    subtree is :func:`_same_computation`-equal to an earlier member's
    node is a recompute of that member's intermediate.  An outer match
    subsumes inner ones — a matched node's descendants are not
    re-scanned.  An identical *term object* (``hit is t``) is already
    shared by the interned joint DAG — recording it would offer a
    rewrite that saves nothing, so it is skipped (and not descended:
    the subtree is literally the same object throughout).
    """
    producers: list[tuple[str, Any]] = []
    shares: dict[str, dict] = {}
    links: list[tuple[str, str]] = []
    for name in order:
        mapping: dict[Any, Any] = {}
        memo: dict = {}
        stack = [bodies[name]]
        while stack:
            t = stack.pop()
            if not isinstance(t, Op):
                continue
            hit = None
            owner = ""
            if has_var_leaf(t):
                for pm, p in producers:
                    if _same_computation(p, t, leaves, memo):
                        hit, owner = p, pm
                        break
            if hit is None:
                stack.extend(t.args)
            elif hit is not t:
                mapping[t] = hit
                links.append((owner, name))
        if mapping:
            shares[name] = mapping
        producers.extend((name, n) for n in _op_nodes(bodies[name]))
    return shares, links


def _share_components(
    links: list[tuple[str, str]], order: list[str]
) -> list[tuple[str, ...]]:
    """Group members linked by shares, in execution order."""
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in links:
        parent[find(i)] = find(j)
    comps: dict[str, set] = {}
    for m in parent:
        comps.setdefault(find(m), set()).add(m)
    out = [tuple(n for n in order if n in c) for c in comps.values()]
    return sorted(out, key=lambda c: order.index(c[0]))


# ---------------------------------------------------------------------------
#  The law
# ---------------------------------------------------------------------------


class CrossBlockCSE:
    """Shared-subterm elimination across same-input sibling blocks.

    Signature-level candidacy mirrors :class:`KVLatentShare`'s cross
    form: a family of >=2 blocks that captured the *same input object*
    and whose bodies carry a common var-reaching subterm (identical
    computation modulo leaf renaming — renamed ``Param`` leaves
    certified by bitwise-equal leaf values).  The wiring evidence —
    additive output consumption on both capture probes, no non-additive
    fan-out — is the match-time gate; the reified program is
    cost-gated and fp64-verified before grafting.

    Opt-in: not part of ``DEFAULT_MORPHISM_LAWS`` — select it with
    ``MorphismSearch(laws=[..., CrossBlockCSE()])``.

    Deliberately out of scope — each is an honest non-reach, not a
    skipped case:

    * sequential residual/chain neighbours: block ``j`` sees a
      different stream object than block ``i`` produced on, so no
      intermediate of ``i`` is consumable at ``j``;
    * param-only shared subtrees (weight-side terms fold at compile
      time — the tying passes' domain, not runtime CSE);
    * non-additive member outputs (no slot can deliver the fused
      value) and multi-input / never-executed / opaque blocks.
    """

    name = "cross_block_cse"

    def match(self, graph: M.MorphismGraph) -> list[M.MorphismMatch]:
        """Match same-input families whose bodies share a subterm."""
        out: list[M.MorphismMatch] = []
        for fam in _input_families(graph).values():
            out.extend(self._family_matches(graph, fam))
        return out

    def _family_matches(
        self, graph: M.MorphismGraph, fam: list[str]
    ) -> list[M.MorphismMatch]:
        """Detect share components inside one same-input family."""
        usable = []
        for n in fam:
            ir = graph.record(n).ir
            if ir is not None and len(ir.inputs) == 1:
                usable.append((n, ir))
        if len(usable) < 2:
            return []
        names = [n for n, _ in usable]
        recs = [graph.record(n) for n in names]
        _, leaves = _family_tables(recs)
        bodies = _family_bodies(usable)
        shares, links = _family_shares(bodies, names, leaves)
        return [
            m
            for comp in _share_components(links, names)
            if (m := self._comp_match(graph, comp, fam, shares))
            is not None
        ]

    def _comp_match(
        self,
        graph: M.MorphismGraph,
        comp: tuple[str, ...],
        fam: list[str],
        shares: dict,
    ) -> M.MorphismMatch | None:
        """One share component -> a match, or an evidence decline."""
        members = comp
        if not _family_evidence(graph, members, fam):
            return None
        n_sh = sum(len(shares.get(n, ())) for n in members)
        return M.MorphismMatch(
            law=self.name,
            nodes=members,
            boundary="family",
            reify=M.ReifySpec(mode="cse", rules="compose"),
            detail=(
                f"{'+'.join(members)}: {n_sh} shared "
                "subterm(s) on one input object"
            ),
        )


# ---------------------------------------------------------------------------
#  Reify — the shared-subterm rewrite back to verified IR
# ---------------------------------------------------------------------------


def _reify_cse(
    match: M.MorphismMatch,
    graph: M.MorphismGraph,
    *,
    sink: Sink,
    cost_fn: CostFn,
    verify_tol: float,
    max_iterations: int,
    max_enodes: int,
    symmetry_budget: int | None,
) -> dict[str, Any]:
    """Reify a shared-subterm match; return the graft record or decline.

    Rebuilds the family bodies and the share mapping (deterministic —
    the match carries no term payload), rewrites each consumer's copy
    of a shared subterm to the earliest producer's term, and offers
    the rewritten joint sum into the joint e-graph under a pointwise
    witness carrying ``error_bound=0.0``: the equality is exact by
    construction (same input object + identical computation +
    certified-equal leaves).  Saturation, the true-DAG cost gate, the
    additive-slot shape check and the fp64 pair verify are the shared
    ``_family_gate`` machinery; the grafted slots are the fused first
    member plus exact-zero fillers.
    """
    prep = _family_prep(match, graph)
    if prep.get("status") == "declined":
        return prep
    recs = prep["recs"]
    x = prep["x"]
    bodies = prep["bodies"]
    params = prep["params"]
    leaves = prep["leaves"]
    joint = prep["joint"]
    names = list(match.nodes)
    shares, _ = _family_shares(bodies, names, leaves)
    if not shares:
        return {"status": "declined", "reason": "no shared subterm"}
    bodies2 = {
        n: _replace_nodes(bodies[n], shares.get(n, {})) for n in names
    }
    joint2 = _add_chain([bodies2[n] for n in names])
    eg, eid = M._saturate(
        joint,
        M._recipe_rules(match.reify.rules),
        max_iterations=max_iterations,
        max_enodes=max_enodes,
        symmetry_budget=symmetry_budget,
        offers=[
            (
                joint2,
                "cross_block_cse: the consumer members' subterms are "
                "the identical computation as the earliest producer "
                "member's on the *same* input object (captured in_obj "
                "identity) with value-certified equal leaves — an "
                "exact equality asserted at morphism level, gated by "
                "the fp64 verify",
                {
                    "note": (
                        "cross_block_cse: shared-subterm fold over a "
                        "same-input family"
                    ),
                    "error_bound": 0.0,
                    "bound_norm": "frobenius",
                },
            )
        ],
    )
    best = eg.extract_best(eid, cost_fn)
    info: dict[str, Any] = {
        "joint": op_repr(joint),
        "reified": op_repr(best),
        "shared_sites": sum(len(m) for m in shares.values()),
        "shared_ops": sorted(
            {p.op for m in shares.values() for p in m.values()}
        ),
    }
    return _family_gate(
        match,
        recs,
        prep["irs"],
        graph,
        intra=False,
        sink=sink,
        cost_fn=cost_fn,
        verify_tol=verify_tol,
        joint=joint,
        best=best,
        x=x,
        inputs=prep["inputs"],
        params=params,
        leaves=leaves,
        info=info,
    )
