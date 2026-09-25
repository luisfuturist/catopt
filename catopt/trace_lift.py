"""The trm ↔ apply bridge: lifting unrolled recurrences into trace form.

catopt/trace.py documents the gap this module closes: recognising an
unrolled recurrence ``h_t = A_t·h_{t−1} + b_t`` (dense) or
``h_t = a_t ⊙ h_{t−1} + b_t`` (diagonal) as ``matmul(trace(F, T·d), v)``
is NOT a lhs→rhs rewrite — it must *construct* the time-extended
nilpotent block-shift matrix F from the whole horizon, the same kind
of non-local, diagram-level transformation as
``pair_shared_input_linears`` in catopt/rules.py.  This module is that
pass.

THE ENCODING (concrete T, concrete d — see CAVEATS)
    Over a horizon of T steps with state width d, the feedback wire
    carries all T states ``u = [h_1; …; h_T]`` (``usize = T·d``) and the
    data wire carries the packed input ``v = [b_1; …; b_T; h_0]``
    (width ``T·d + d``).  With feedback-first block layout
    ``F = [[S, R], [Q, P]]``:

    * ``S[t, t−1] = M_{t+1}`` — the per-step transition on the
      strictly-lower block diagonal (``Z ∘ diag(M_t)``): S is nilpotent
      (``S^T = 0``), so the fixpoint ``u = Su + Rv`` is the unrolled
      scan EXACTLY — no convergence qualifier.
    * ``R[t, t] = I`` routes step input ``b_{t+1}`` in;
      ``R[0, T] = M_1`` applies the first transition to ``h_0``.
    * ``Q = [0 … 0 I]`` reads out the last block (``h_T``); ``P = 0``.

    Then ``Tr(F) = P + Q(I−S)⁻¹R`` is a ``(d, T·d+d)`` matrix and
    ``reshape(matmul(Tr(F), v), (d,))`` is exactly the recurrence's
    final state ``h_T`` — which is what gets unioned into the
    recurrence's e-class.

    Per-step transition blocks ``M_t`` are (d,d) matrix terms: for the
    dense carrier the ``A_t`` enode itself; for the diagonal carrier
    ``mul(eye(d), a_t)`` (broadcasting builds ``diag(a_t)``).  The
    zero block ``Z = mul(eye(d), 0)`` and the identity ``eye(d)`` are
    hash-consed once and shared — F costs ~2T enodes, not O(T²).

CHANNEL SPLITTING (diagonal recurrences)
    An elementwise recurrence is independent per channel, so for a
    partition ``d = d1 + d2`` the pass additionally offers

        trace(parl(F_1, F_2, u1=T·d1, u2=T·d2), usize=(T·d1, T·d2))

    built from ``split`` views of the same per-step terms, with the
    packed input re-laid as ``[v_1; v_2]`` (parl's wiring keeps both
    feedback wires first).  Once that member exists, ``tr_superpose``
    fires locally and produces ``bdiag(trace(F_1), trace(F_2))`` —
    the parallel-channel schedule — and ``tr_expand`` reaches each
    channel's resolvent.  Dense ``A_t`` couples channels, so the split
    is only offered for the ``mul``/``aff_diag`` carrier.

RECOGNITION — two spine shapes, one pass:
    * carrier trees: ``apply(aff-tree, h)`` / ``applyd(affd-tree, h)``
      members, detected by reusing ``scan_lower.build_scan_plan``
      (in-order leaves of the compose tree are the steps in
      REVERSE-chronological order — ``aff_compose(f,g)`` applies ``g``
      first — so the leaf list is flipped).
    * raw spines: ``add(mul(a_t, ·), ·)`` / ``add(matmul(A_t, ·), ·)``
      chains walked e-class to e-class back to a leaf (or
      ``apply``/``applyd``) base.  Any chain assembled this way is
      SOUND regardless of length or branch choice — each step mirrors a
      real ``add`` e-node over real e-class members — so ambiguity is
      resolved by preferring the longest chain and vetoing ties.

Each offered member is ``add_enode``-built (children stay e-class ids,
so later rewrites see through) and merged with a ``rule=None`` union —
the merge is recorded as e-graph-dependent in certificates, exactly
like the pairing pass (the equality is real but has no standalone
lhs→rhs derivation).  With ``witness=True`` the pass instead attaches
a synthesised pointwise :class:`Rewrite` — ``oldest_class_member ->
offered_term`` — to each union via ``EGraph.union(..., witness=...)``;
the merge then replays in certificates as an ordinary named rule step
(``verify_certificate`` re-matches and re-instantiates it standalone,
``strict=True`` included) rather than an ``egraph_dependent`` stub.

CAVEATS (when the pass declines):
    * T must be concrete — the spine length is the unrolled horizon;
      symbolic/dynamic loops have no finite F to build.
    * d must be concrete — ``usize`` and every block size are ints.
    * states/inputs must be vectors ``(d,)``; dense maps ``(d,d)``;
      diagonal maps ``(d,)``.
    * the chain base must bottom out at a leaf (Param/Var/Const) or an
      already-carried ``apply``/``applyd`` segment; a computed init
      state simply ends the walk (the segment above still lifts if it
      is ≥ ``min_steps``).
    * idempotent — re-running rebuilds the same hash-consed enodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import catopt.trace as _cat_trace  # noqa: F401  (torch bindings for
                                   # trace/bdiag/parl/eye/cswap/inv)
from catopt.cost import _shape_of
from catopt.egraph import EGraph, Rewrite, _LeafRegistry
from catopt.ir import Const, Op
from catopt.scan_lower import build_scan_plan

__all__ = ["TraceLift", "lift_scan_to_trace"]


#: Ops preferred when picking a class representative for the carrier
#: path — the apply/applyd member over the raw tensor spine.
_CARRIER_OPS = frozenset({
    "apply", "applyd", "aff", "aff_diag", "aff_compose", "affd_compose"})

#: E-node ops that mark a class as a usable chain base: a leaf (the h0
#: Param/Var/Const) or an already-lifted scan segment (whose value the
#: trace simply consumes as its init).
_BASE_OPS = frozenset({"leaf", "apply", "applyd"})


# ---------------------------------------------------------------------------
#  Plans — a recognised recurrence spine, e-class ids throughout
# ---------------------------------------------------------------------------

@dataclass
class _Plan:
    """A recurrence recognised in the e-graph.

    ``maps``/``ins`` are chronological (step 1 first).  For ``kind ==
    "diag"`` each map e-class is the (d,) decay vector; for ``"dense"``
    the (d,d) transition matrix.  ``step_states`` lists the e-class ids
    the spine decomposed through (used to drop prefix chains).
    """
    kind: str
    T: int
    d: int
    maps: list
    ins: list
    h0: int
    step_states: list = field(default_factory=list)


def _term_of(eg: EGraph, eid: int) -> Any:
    return eg.any_term(eg.find(eid))


def _consistent_shapes(eg: EGraph, kind: str, maps: list,
                       ins: list, h0: int):
    """All steps must share one concrete vector width — returns d."""
    hs = _shape_of(_term_of(eg, h0))
    if not (isinstance(hs, tuple) and len(hs) == 1
            and isinstance(hs[0], int) and hs[0] > 0):
        return None
    d = hs[0]
    want_map = (d,) if kind == "diag" else (d, d)
    for e in maps:
        if _shape_of(_term_of(eg, e)) != want_map:
            return None
    for e in ins:
        if _shape_of(_term_of(eg, e)) != (d,):
            return None
    return d


# -- carrier path: apply/applyd members (reuses scan_lower's plan) ----

def _prefer_term(eg: EGraph, cid: int, prefer: frozenset,
                 _seen: frozenset = frozenset()):
    """``any_term`` variant preferring enodes whose op is in *prefer*.

    Picks the apply/applyd-headed member of a class when one exists so
    the carrier tree is visible even when the class also holds the raw
    spine (which ``any_term`` might otherwise return).
    """
    cid = eg.find(cid)
    if cid in _seen:
        return None
    ec = eg._classes.get(cid)
    if ec is None:
        return None

    def rank(n):
        if n.op in prefer:
            return 0
        if n.op == "leaf":
            return 1
        return 2

    for node in sorted(ec.nodes,
                       key=lambda n: (rank(n), n.op, n.children,
                                      repr(n.attrs))):
        if node.op == "leaf":
            key = node.attrs[0][1] if node.attrs else "??"
            return _LeafRegistry.decode(key)
        args = []
        ok = True
        for c in node.children:
            cc = eg.find(c)
            if cc == cid or cc in _seen:
                ok = False
                break
            t = _prefer_term(eg, cc, prefer, _seen | {cid})
            if t is None:
                ok = False
                break
            args.append(t)
        if ok:
            return Op.make(node.op, *args, **dict(node.attrs))
    return None


def _carrier_plan(eg: EGraph, cid: int):
    """Plan from an ``apply``/``applyd`` member of the class, if any."""
    ec = eg._classes.get(cid)
    if ec is None or not any(n.op in ("apply", "applyd")
                             for n in ec.nodes):
        return None
    t = _prefer_term(eg, cid, _CARRIER_OPS)
    if t is None:
        return None
    # build_scan_plan folds nested apply segments, checks the tree is a
    # pure single-domain aff/aff_diag tree, and verifies uniform leaf
    # shapes — the same detection the batched executor uses.
    plan = build_scan_plan(t)
    if plan is None:
        return None
    leaves = plan["leaves"]
    # Compose trees apply their RIGHT subtree first, so the in-order
    # leaf list is reverse-chronological: leaf[-1] is step 1.
    steps = list(reversed(leaves))
    kind = "diag" if plan["diagonal"] else "dense"
    maps = [eg.add_term(leaf.args[0]) for leaf in steps]
    ins = [eg.add_term(leaf.args[1]) for leaf in steps]
    h0 = eg.add_term(plan["h"])
    d = _consistent_shapes(eg, kind, maps, ins, h0)
    if d is None:
        return None
    return _Plan(kind, len(steps), d, maps, ins, h0)


# -- raw-spine path: add(mul|matmul) chains over e-classes ------------

class _Spine:
    """Walk ``add(mul(a_t, s), i_t)`` / ``add(matmul(A_t, s), i_t)``
    chains from e-class to e-class.

    ``_walk(cid)`` returns ``(steps, base_eid)`` — the chronological
    decomposition of the class's value — or ``None`` when the class is
    neither a recognisable step nor a base.  Results are memoised on
    canonical ids; the ``_active`` set cuts cycles (post-union classes
    can be self-referential).  A memoised chain may be shorter than the
    true longest when a cycle cut truncated a branch — a completeness
    caveat only: every emitted chain is a real sequence of ``add``
    e-nodes over e-class members, hence sound regardless of length.
    """

    def __init__(self, eg: EGraph) -> None:
        self.eg = eg
        self._memo: dict[int, Any] = {}
        self._active: set[int] = set()

    def plan(self, cid: int):
        res = self._walk(self.eg.find(cid))
        if res is None:
            return None
        steps, base = res
        if not steps:
            return None
        kinds = {s["kind"] for s in steps}
        if len(kinds) != 1:
            return None          # mixed dense/diagonal spine — decline
        kind = kinds.pop()
        maps = [s["map"] for s in steps]
        ins = [s["in"] for s in steps]
        d = _consistent_shapes(self.eg, kind, maps, ins, base)
        if d is None:
            return None
        return _Plan(kind, len(steps), d, maps, ins, base,
                     step_states=[s["state"] for s in steps])

    def _candidates(self, node) -> list:
        """Decompositions of ``add(c0, c1)`` as map·state + input.

        Both add slots are tried as the product; ``mul`` factors are
        tried in both roles (comm_mul-free graph: operand order is not
        canonical).  ``matmul`` keeps its conventional reading
        ``matmul(A, h)`` — the state is the second operand.
        """
        c0, c1 = node.children
        out = []
        for prod, in_e in ((c0, c1), (c1, c0)):
            pec = self.eg._classes.get(self.eg.find(prod))
            if pec is None:
                continue
            for m in pec.nodes:
                if len(m.children) != 2:
                    continue
                if m.op == "mul":
                    f0, f1 = m.children
                    out.append({"map": f1, "state": f0, "in": in_e,
                                "kind": "diag"})
                    out.append({"map": f0, "state": f1, "in": in_e,
                                "kind": "diag"})
                elif m.op == "matmul":
                    out.append({"map": m.children[0],
                                "state": m.children[1],
                                "in": in_e, "kind": "dense"})
        return out

    def _walk(self, cid: int):
        cid = self.eg.find(cid)
        if cid in self._memo:
            return self._memo[cid]
        if cid in self._active:
            return None                    # cycle cut
        ec = self.eg._classes.get(cid)
        if ec is None:
            return None
        self._active.add(cid)
        try:
            best = None
            tied = False
            for node in ec.nodes:
                if node.op != "add" or len(node.children) != 2:
                    continue
                for cand in self._candidates(node):
                    sub = self._walk(cand["state"])
                    if sub is None:
                        continue
                    cur = (sub[0] + [cand], sub[1])
                    if best is None or len(cur[0]) > len(best[0]):
                        best, tied = cur, False
                    elif len(cur[0]) == len(best[0]):
                        sig = [(s["map"], s["in"], s["state"])
                               for s in cur[0]]
                        bsig = [(s["map"], s["in"], s["state"])
                                for s in best[0]]
                        if sig != bsig:
                            tied = True   # ambiguous: two equal-length
                                          # decompositions — veto
            if best is not None:
                res = None if tied else best
            elif any(n.op in _BASE_OPS for n in ec.nodes):
                res = ([], cid)            # chain base: h0 / carried
            else:                          #   scan segment
                res = None
            self._memo[cid] = res
            return res
        finally:
            self._active.discard(cid)


# ---------------------------------------------------------------------------
#  Construction — the time-extended matrix as terms
# ---------------------------------------------------------------------------

class _Emit:
    """Builds every node twice: as an e-graph enode (eid, children are
    e-class ids so later rewrites see through) and as a concrete Op
    term (for the returned record / direct evaluation)."""

    def __init__(self, eg: EGraph, provenance: str) -> None:
        self.eg = eg
        self.prov = provenance
        self.broken = False

    def op(self, name: str, kids=(), attrs: dict | None = None):
        kids = list(kids)
        attrs = dict(attrs or {})
        eid = self.eg.add_enode(name, tuple(k[0] for k in kids), attrs,
                                provenance=self.prov)
        return (eid, Op.make(name, *(k[1] for k in kids), **attrs))

    def ref(self, eid: int):
        """An (eid, representative term) pair for an existing class."""
        eid = self.eg.find(eid)
        t = self.eg.any_term(eid)
        if t is None:
            self.broken = True
        return (eid, t)

    def leaf(self, t: Any):
        return (self.eg.add_term(t, provenance=self.prov), t)


def _assemble_F(em: _Emit, maps: list, d: int):
    """The time-extended feedback matrix F (feedback-first layout).

    ``maps[i]`` is the (d,d) transition block of step i+1.  Returns the
    (eid, term) pair of the ``(T·d + d) × (2·T·d + d)`` concat.
    """
    T = len(maps)
    eye = em.op("eye", (), {"dim": d})
    zero = em.leaf(Const(0.0))
    Z = em.op("mul", (eye, zero))            # shared (d,d) zero block
    rows = []
    for i in range(T):
        # u'_i = h_{i+1} = M_{i+1}·u_{i-1} + b_{i+1} (+ M_1·h0 for i=0)
        s_cols = [maps[i] if j == i - 1 else Z for j in range(T)]
        r_cols = [eye if j == i else Z for j in range(T)]
        r_cols.append(maps[0] if i == 0 else Z)
        rows.append(em.op("concat", s_cols + r_cols, {"dim": -1}))
    # y = h_T: Q selects the last u block; P = 0.
    rows.append(em.op("concat",
                      [Z] * (T - 1) + [eye] + [Z] * (T + 1),
                      {"dim": -1}))
    return em.op("concat", rows, {"dim": -2})


def _emit_head(em: _Emit, F, ins: list, h0, d: int, usize):
    """``reshape(matmul(trace(F, usize), vec), (d,))`` — the member
    offered to the recurrence's e-class."""
    vec = em.op("concat", list(ins) + [h0], {"dim": 0})
    vec2 = em.op("unsqueeze", (vec,), {"dim": 1})
    tr = em.op("trace", (F,), {"usize": usize})
    mv = em.op("matmul", (tr, vec2))
    out = em.op("reshape", (mv,), {"shape": (d,)})
    return tr, vec, out


def _channel_F(em: _Emit, kind: str, maps: list, d: int):
    """F for one channel block: diagonal maps are densified as
    ``mul(eye(d), a_t)``; dense maps are used directly."""
    if kind == "diag":
        eye = em.op("eye", (), {"dim": d})
        maps = [em.op("mul", (eye, m)) for m in maps]
    return _assemble_F(em, maps, d)


def _partitions(plan: _Plan, channel_splits) -> list:
    """Channel partitions to offer for diagonal recurrences.

    ``"auto"`` — the balanced 2-way split.  An iterable of ``(d1, d2)``
    tuples offers each.  Dense transitions couple channels, so no
    split is provable there — none is offered.
    """
    if plan.kind != "diag":
        return []
    d = plan.d
    if channel_splits == "auto":
        return [(d // 2, d - d // 2)] if d >= 2 else []
    if not channel_splits:
        return []
    out = []
    for part in channel_splits:
        if (isinstance(part, (tuple, list)) and len(part) == 2
                and all(isinstance(x, int) and x > 0 for x in part)
                and part[0] + part[1] == d):
            out.append(tuple(part))
    return out


# ---------------------------------------------------------------------------
#  The pass
# ---------------------------------------------------------------------------

@dataclass
class TraceLift:
    """One member offered by :func:`lift_scan_to_trace`.

    ``term`` is the offered member as a plain Op tree (evaluable via
    ``IRModule`` / the torch bindings); ``out_eid`` is its e-class id,
    unioned into ``root_eid`` — the recurrence's class.  ``split`` is
    the channel partition for the ``parl`` form, else ``None``.
    """
    root_eid: int
    out_eid: int
    trace_eid: int
    f_eid: int
    vec_eid: int
    term: Any
    kind: str
    T: int
    d: int
    split: tuple | None = None


def _lift_witness(eg: EGraph, cid: int, offered: Any, offered_eid: int,
                  provenance: str) -> Rewrite | None:
    """Synthesise the pointwise :class:`Rewrite` certifying one offer.

    ``lhs`` is the *oldest* member of the recurrence class — the term
    the e-graph saw first (the input spine or carrier the pass
    recognised), chosen so the certificate's ``src`` side usually *is*
    the witness LHS and the bridging connect is trivial.  ``rhs`` is
    the offered trace member itself.  The pass asserts this equality by
    construction — it built F so that ``matmul(trace(F), v)`` is the
    fixpoint of THIS unrolled recurrence — and recording the union
    under the synthesised rule makes that assertion replayable:
    ``certificate`` emits it as a named step and
    ``verify_certificate`` re-matches/re-instantiates it on real terms.

    The name is unique per offered enode so several offers in one graph
    never collide in ``_rule_objs``.  Returns ``None`` when the class
    has no resolvable member (degenerate cyclic graph) — the union then
    proceeds witness-free and stays ``egraph_dependent``.
    """
    src = eg._oldest_term(eg.find(cid))
    if src is None:
        return None
    return Rewrite(
        name=f"{provenance}#{offered_eid}",
        lhs=src,
        rhs=offered,
        law=("pointwise witness for a non-local offer: this unrolled "
             "recurrence equals its nilpotent block-shift trace "
             "fixpoint (equality established by construction in "
             "lift_scan_to_trace)"))


def _offer(eg: EGraph, cid: int, plan: _Plan, channel_splits,
           provenance: str, witness: bool) -> list:
    em = _Emit(eg, provenance)
    lifts: list[TraceLift] = []
    d, T = plan.d, plan.T

    # -- the joint trace: usize = T·d ---------------------------------
    maps = [em.ref(m) for m in plan.maps]
    ins = [em.ref(i) for i in plan.ins]
    h0 = em.ref(plan.h0)
    if not em.broken:
        F = _channel_F(em, plan.kind, maps, d)
        tr, vec, out = _emit_head(em, F, ins, h0, d, T * d)
        eg.union(cid, out[0],
                 witness=(_lift_witness(eg, cid, out[1], out[0],
                                        provenance)
                          if witness else None),
                 note=(f"trace_lift: unrolled {plan.kind} recurrence "
                      f"(T={T}, d={d}) → nilpotent block-shift fixpoint"))
        lifts.append(TraceLift(
            root_eid=cid, out_eid=out[0], trace_eid=tr[0],
            f_eid=F[0], vec_eid=vec[0], term=out[1],
            kind=plan.kind, T=T, d=d))

    # -- channel-split form (diagonal carriers only) -------------------
    for part in _partitions(plan, channel_splits):
        em.broken = False
        Fs, vecs = [], []
        for ci, dc in enumerate(part):
            at = {"sizes": part, "dim": 0, "index": ci}
            mc = [em.op("split", (em.ref(m),), at) for m in plan.maps]
            ic = [em.op("split", (em.ref(i),), at) for i in plan.ins]
            hc = em.op("split", (em.ref(plan.h0),), at)
            Fs.append(_channel_F(em, "diag", mc, dc))
            vecs.append(em.op("concat", list(ic) + [hc], {"dim": 0}))
        if em.broken:
            continue
        u1, u2 = T * part[0], T * part[1]
        Fp = em.op("parl", tuple(Fs), {"u1": u1, "u2": u2})
        vecp = em.op("concat", vecs, {"dim": 0})
        vec2 = em.op("unsqueeze", (vecp,), {"dim": 1})
        trp = em.op("trace", (Fp,), {"usize": (u1, u2)})
        mvp = em.op("matmul", (trp, vec2))
        outp = em.op("reshape", (mvp,), {"shape": (d,)})
        eg.union(cid, outp[0],
                 witness=(_lift_witness(eg, cid, outp[1], outp[0],
                                        provenance)
                          if witness else None),
                 note=(f"trace_lift: channel-split {part} of a T={T} "
                      f"diagonal recurrence → joint trace over parl"))
        lifts.append(TraceLift(
            root_eid=cid, out_eid=outp[0], trace_eid=trp[0],
            f_eid=Fp[0], vec_eid=vecp[0], term=outp[1],
            kind=plan.kind, T=T, d=d, split=tuple(part)))
    return lifts


def lift_scan_to_trace(eg: EGraph, root_eid: int | None = None, *,
                       min_steps: int = 2, channel_splits="auto",
                       maximal_only: bool = True,
                       provenance: str = "trace_lift",
                       witness: bool = True) -> list:
    """Offer ``trace`` members for every unrolled recurrence in *eg*.

    For each e-class carrying a recognisable recurrence spine — an
    ``apply``/``applyd`` carrier tree (preferred, via
    ``scan_lower.build_scan_plan``) or a raw ``add(mul|matmul …)``
    chain — construct the time-extended nilpotent block-shift matrix F
    and union ``reshape(matmul(trace(F, T·d), vec), (d,))`` into the
    class.  Diagonal recurrences additionally get the ``parl``
    channel-split form (see module docstring), which ``tr_superpose``
    turns into a ``bdiag`` of independent channel traces on the next
    saturation.

    ``root_eid`` restricts the pass to one class (test/debug hook).
    ``min_steps`` sets the shortest chain worth lifting (a T=1 "step"
    is sound but pointless).  ``maximal_only`` drops chains that are
    strict prefixes of a longer recognised chain.

    ``witness`` attaches a replayable certificate witness to every
    offered union (see :func:`_lift_witness` and
    ``EGraph.union(..., witness=...)``): each offer's merge then shows
    up in :meth:`EGraph.certificate` as a named, standalone-replayable
    rule step instead of an ``egraph_dependent`` stub, so
    ``verify_certificate(..., strict=True)`` accepts it.  On by
    default; pass ``witness=False`` to keep the honest
    "no standalone derivation" marking.

    Returns a list of :class:`TraceLift` records — one per offered
    member — in class-iteration order.  Empty when nothing matches:
    the pass is a no-op on non-recurrence graphs.
    """
    plans: dict[int, _Plan] = {}
    spine = _Spine(eg)
    for cid in list(eg._classes.keys()):
        c = eg.find(cid)
        if c in plans or eg._classes.get(c) is None:
            continue
        if root_eid is not None and c != eg.find(root_eid):
            continue
        plan = _carrier_plan(eg, c)
        if plan is None:
            plan = spine.plan(c)
        if plan is not None and plan.T >= min_steps:
            plans[c] = plan

    interior: set[int] = set()
    if maximal_only:
        for p in plans.values():
            for st in p.step_states:
                if st in plans and plans[st].T < p.T:
                    interior.add(st)
        # Carrier-path plans record no step_states (the carrier tree
        # is a term, not an e-class chain), so the walk above misses
        # every prefix class that only carries an apply/applyd
        # member.  Detect them structurally: a plan is interior to a
        # strictly longer plan when both share the same h0 e-class
        # AND the shorter plan's chronological (maps, ins) eid
        # sequence is a literal prefix of the longer's — same map and
        # input e-classes over the same init compute the same
        # intermediate value, so the shorter chain's class IS a state
        # of the longer one.  Without this, every saturated prefix
        # class mints its own block-matrix F (~2T offers instead of
        # the ~2 for the whole horizon — the ~2T× storage blow-up).
        items = list(plans.items())
        for qc, q in items:
            if qc in interior:
                continue
            qh0 = eg.find(q.h0)
            qmaps = [eg.find(m) for m in q.maps]
            qins = [eg.find(i) for i in q.ins]
            for pc, p in items:
                if p.T <= q.T or p.kind != q.kind:
                    continue
                if eg.find(p.h0) != qh0:
                    continue
                if ([eg.find(m) for m in p.maps[:q.T]] == qmaps
                        and [eg.find(i) for i in p.ins[:q.T]] == qins):
                    interior.add(qc)
                    break

    lifts: list[TraceLift] = []
    for c, p in plans.items():
        if c in interior:
            continue
        lifts.extend(_offer(eg, c, p, channel_splits, provenance,
                            witness))
    return lifts
