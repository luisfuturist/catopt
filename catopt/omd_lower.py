# ruff: noqa: RUF002
"""Schedule-batched lowering for deferred-affine (omd) attention terms.

After the cross-carrier lifts (:func:`catopt.xcarrier.omd_tree_lift`),
chunked attention over scanned values extracts as::

    omd_apply(<omd_compose tree of omd_elem(s_i, a_i, b_i)>, h0)

— the whole of softmax attention as ONE map, affine in the initial
scan state.  The generic :class:`~catopt.torch_bridge.IRModule`
evaluates the member correctly but *serially*: each leaf's coefficient
maps ``a_i``/``b_i`` are ``stack(affd_a f_j)`` / ``stack(affd_b f_j)``
over per-step compose chains — O(T²) unrolled nodes (or O(T) dispatched
ops when the chain shares prefixes), each a Python-dispatched torch
call on a tiny ``(d,)`` tensor.

This module lowers the same term batched — the
:class:`~catopt.scan_lower.BatchedScanModule` analogue for the omd
carrier:

*   The map forest inside the leaf projections is evaluated once.
    When every projected map is a *prefix* of one leaf sequence (the
    emitted-scan pattern ``f_j = affd_compose(leaf_j, f_{j-1})``), all
    prefixes come from a single blocked associative scan over the
    ``(a, b)`` pairs — O(√T) batched compose steps instead of O(T)
    serial ones.  Arbitrary bracketings fall back to *level-batched*
    forest evaluation (compose nodes grouped by depth, one batched
    op pair per level), which is bit-identical to serial evaluation.
*   ``stack(affd_* f_j)`` nodes and lone projections are answered from
    the batched maps by gather/view — seeded into the evaluator's memo
    so surrounding tensor terms (e.g. ``matmul(E, stack(...))``)
    evaluate unchanged.
*   The ``omd_compose`` tree itself is evaluated level-batched on
    stacked ``(m, l, fa, fb)`` components — the ``where``/``isfinite``
    −inf handling of ``_omd_compose`` is replicated verbatim, so
    fully-masked blocks keep their exact NaN semantics.

``to_batched_omd_module`` (and :class:`BatchedOmdModule` itself)
detects the ``omd_apply[m](omd-tree, h)`` shape; any other IR falls
back to ordinary IRModule evaluation, so the class is a drop-in
replacement for ``ir_to_torch_module``.

Numerics: scheduling differs but semantics are identical.  The
level-batched paths evaluate the *same* per-node operations (batched
elementwise ops are bitwise-equal per element).  The chain fast path
reassociates the compose fold (a blocked scan), so results agree with
the generic evaluator to fp64 reassociation tolerance rather than
bitwise — exactly the tolerance the carrier laws are verified at.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import catopt.xcarrier  # noqa: F401 — registers the omd_* / affd_*
from catopt.ir import IR, Op
from catopt.torch_bridge import _IR_TO_TORCH, IRModule
from catopt.typing import _shape_of

# torch bindings into _IR_TO_TORCH.

__all__ = [
    "BatchedOmdModule",
    "build_omd_plan",
    "is_omd_apply_term",
    "to_batched_omd_module",
]

#: Projection ops → (component index into the map tuple, carrier domain).
_PROJECTIONS = {
    "affd_a": (0, "diag"),
    "affd_b": (1, "diag"),
    "aff_A": (0, "dense"),
    "aff_b": (1, "dense"),
}
#: Compose ops → carrier domain.
_COMPOSE_OP = {"affd_compose": "diag", "aff_compose": "dense"}
#: Map-leaf ops → carrier domain.
_LEAF_OP = {"aff_diag": "diag", "aff": "dense"}
#: Ops allowed as leaves of the deferred carrier tree.
_OMD_LEAF_OPS = ("omd_elem", "omd")


def _stack_dim(attrs: dict) -> int:
    d = attrs.get("dim", attrs.get("arg1", 0))
    return d if isinstance(d, int) else 0


def _select_index(term: Any):
    """Decompose ``select(base, dim, i)`` → ``(base, dim, i)`` or None."""
    if (
        not isinstance(term, Op)
        or term.op not in ("select", "getitem")
        or len(term.args) != 1
    ):
        return None
    dim = term.attrs.get("arg1", term.attrs.get("dim", 0))
    idx = term.attrs.get("arg2", term.attrs.get("index"))
    if not isinstance(dim, int) or not isinstance(idx, int):
        return None
    return (term.args[0], dim, idx)


def _concrete(s) -> bool:
    return (
        isinstance(s, tuple)
        and len(s) > 0
        and all(isinstance(d, int) for d in s)
    )


def _is_omd_tree(t: Any, memo: dict | None = None) -> bool:
    """True if ``t`` is an ``omd_compose`` tree over omd leaves.

    Leaves are ``omd_elem`` (deferred affine element) or the ``omd``
    tuple-packaging node; shared subtrees are memoised on the term
    itself (hash-consed terms are content-keyed).
    """
    memo = {} if memo is None else memo
    k = t
    if k in memo:
        return memo[k]
    ok = False
    if isinstance(t, Op):
        if t.op == "omd_compose" and len(t.args) == 2:
            ok = _is_omd_tree(t.args[0], memo) and _is_omd_tree(
                t.args[1], memo
            )
        else:
            ok = t.op in _OMD_LEAF_OPS
    memo[k] = ok
    return ok


def is_omd_apply_term(root: Any) -> bool:
    """True if ``root`` is ``omd_apply[m](<omd tree>, h)`` — deferred
    affine attention applied to the shared initial state."""
    return (
        isinstance(root, Op)
        and root.op in ("omd_apply", "omd_applym")
        and len(root.args) == 2
        and _is_omd_tree(root.args[0])
    )


def _flatten_map(t: Any, out: list) -> None:
    """In-order leaf list of a map term in APPLICATION order.

    ``compose(f, g)`` applies ``g`` first, so the chronological leaf
    sequence traverses the right child before the left.  Iterative —
    chains nest O(T) deep.
    """
    stack = [t]
    while stack:
        u = stack.pop()
        if (
            isinstance(u, Op)
            and u.op in _COMPOSE_OP
            and len(u.args) == 2
        ):
            stack.append(u.args[0])  # left child applies LAST
            stack.append(u.args[1])
        else:
            out.append(u)


def _leaf_sig(leaf: Any, domain: str):
    """Uniform-shape signature for a map leaf.

    Returns ``(domain, a_shape)`` for recognised ``aff_diag``/``aff``
    leaves, ``None`` for opaque leaves (no static guarantee — runtime
    stacking enforces), or ``False`` for a recognised leaf whose shape
    contract is violated (diag needs a == b; dense needs square A with
    b == A[:-1] so compose is shape-stable).
    """
    if not isinstance(leaf, Op) or len(leaf.args) != 2:
        return None
    if leaf.op == "aff_diag":
        a, b = _shape_of(leaf.args[0]), _shape_of(leaf.args[1])
        if _concrete(a) and a == b:
            return ("diag", a)
        return False
    if leaf.op == "aff":
        A, b = _shape_of(leaf.args[0]), _shape_of(leaf.args[1])
        if (
            _concrete(A)
            and len(A) >= 2
            and A[-1] == A[-2]
            and _concrete(b)
            and b == A[:-1]
        ):
            return ("dense", A)
        return False
    return None


def _part_gather(leaves: list, argidx: int):
    """All leaf arg ``argidx`` are ``base[dim, i]`` slices of one base.

    Returns ``(base_term, dim, index_list)`` or ``None`` — same trick
    as ``scan_lower._leaf_b_gather``: one ``index_select`` replaces n
    tiny indexing calls.
    """
    parts = []
    for leaf in leaves:
        if (
            not isinstance(leaf, Op)
            or leaf.op not in _LEAF_OP
            or len(leaf.args) != 2
        ):
            return None
        p = _select_index(leaf.args[argidx])
        if p is None:
            return None
        parts.append(p)
    base0, dim0 = parts[0][0], parts[0][1]
    if any(p[0] is not base0 or p[1] != dim0 for p in parts):
        return None
    return (base0, dim0, [p[2] for p in parts])


def _compose_pair(fa, fb, ga, gb, domain):
    """``compose(f, g)`` = f-after-g, batched over a leading dim.

    Diag: ``(fa⊙ga, fa⊙gb + fb)``; dense: ``(fa@ga, fa@gb + fb)`` —
    the ``affd_compose``/``aff_compose`` bindings verbatim, elementwise
    or per-batch-entry so results are bitwise-equal to serial eval.
    """
    if domain == "dense":
        return fa @ ga, (fa @ gb.unsqueeze(-1)).squeeze(-1) + fb
    return fa * ga, fa * gb + fb


def build_omd_plan(root: Any) -> dict | None:
    """Analyse ``omd_apply[m](omd-tree, h)`` into a batched schedule.

    Returns ``None`` when the root is not a deferred-affine application
    (or shapes/domains are inconsistent — the caller then falls back to
    the serial evaluator).  Otherwise a dict:

    * ``"omd_leaves"`` / ``"omd_gather"`` / ``"omd_root"`` — the om-side
      level schedule (leaf tuples occupy slots ``0..n-1``, each level's
      outputs append in order).
    * ``"map_mode"`` — ``None`` (no projections inside the leaves),
      ``"chain"`` (every projected map is a prefix of ONE leaf sequence
      → single blocked scan), or ``"forest"`` (general bracketing →
      per-domain level-batched compose).
    * ``"chain_*"`` — leaf sequence, domain and index_select gathers.
    * ``"forest"`` — per-domain dicts ``{"leaves", "gather", "slot",
      "a_gather", "b_gather"}``.
    * ``"proj_seeds"`` — ``(node, comp_idx, src, pos)``; ``comp_idx`` 0
      is the map's a-part, 1 the b-part; ``src`` is ``("chain",)`` or
      ``("forest", domain)``.
    * ``"stack_seeds"`` — ``(node, comp_idx, src, positions, dim)`` for
      ``stack(affd_* f_j)`` fast paths.
    * ``"map_seeds"`` — ``(map_node, src, pos)`` so map terms used
      elsewhere (e.g. under a stray ``applyd``) short-circuit to the
      already-computed pair.
    """
    if not is_omd_apply_term(root):
        return None
    f_term, h_term = root.args

    # ---- om side: leaves + compose levels ----------------------------
    omd_leaves: list[Op] = []
    omd_levels: list[list[Op]] = []
    level_of: dict[Any, int] = {}

    def visit(t: Op) -> int:
        tid = t
        if tid in level_of:
            return level_of[tid]
        if t.op == "omd_compose":
            lv = max(visit(t.args[0]), visit(t.args[1])) + 1
            level_of[tid] = lv
            while len(omd_levels) < lv:
                omd_levels.append([])
            omd_levels[lv - 1].append(t)
            return lv
        level_of[tid] = 0
        omd_leaves.append(t)
        return 0

    visit(f_term)
    slot = {lf: i for i, lf in enumerate(omd_leaves)}
    nxt = len(omd_leaves)
    omd_gather = []
    for nodes in omd_levels:
        omd_gather.append(
            (
                [slot[t.args[0]] for t in nodes],
                [slot[t.args[1]] for t in nodes],
            )
        )
        for t in nodes:
            slot[t] = nxt
            nxt += 1

    # ---- scan leaf tensor args (and h) for map projections -----------
    targets: dict[Any, tuple[Any, str]] = {}  # map term -> (term, domain)
    raw_proj: list[tuple[Op, int, Any]] = []  # (node, comp_idx, map)
    raw_stack: list[tuple[Op, int, str, list, int]] = []
    scanned: set[Any] = set()
    bad = False

    def _register(mp: Any, dom: str) -> None:
        nonlocal bad
        prev = targets.get(mp)
        if prev is None:
            targets[mp] = (mp, dom)
        elif prev[1] != dom:
            bad = True

    def scan(t: Any, inside_map_leaf: bool = False) -> None:
        nonlocal bad
        if not isinstance(t, Op) or t in scanned:
            return
        scanned.add(t)
        pinfo = _PROJECTIONS.get(t.op)
        if pinfo is not None and len(t.args) == 1:
            if inside_map_leaf:
                # A map leaf whose own args project another map — the
                # batched leaf level would need map values that depend
                # on leaf values.  Decline; serial eval is correct.
                bad = True
                return
            raw_proj.append((t, pinfo[0], t.args[0]))
            _register(t.args[0], pinfo[1])
            return
        if t.op == "stack" and t.args:
            infos = [
                _PROJECTIONS.get(c.op) if isinstance(c, Op) else None
                for c in t.args
            ]
            if infos[0] is not None and all(
                i == infos[0] for i in infos
            ):
                if inside_map_leaf:
                    bad = True
                    return
                ci, dom = infos[0]
                maps = [c.args[0] for c in t.args]
                for mp in maps:
                    _register(mp, dom)
                raw_stack.append(
                    (t, ci, dom, maps, _stack_dim(dict(t.attrs)))
                )
                for c in t.args:
                    raw_proj.append((c, ci, c.args[0]))
                return
        if t.op in _COMPOSE_OP or t.op in _LEAF_OP:
            # A map-valued node sitting in a tensor position — its
            # innards are map structure, not tensor terms; any
            # projections inside it stay on the serial path.
            return
        for a in t.args:
            scan(a, inside_map_leaf)

    # ---- collect the map forest, per domain --------------------------
    domains: dict[str, dict] = {}

    def _ctx(dom: str) -> dict:
        d = domains.get(dom)
        if d is None:
            d = domains[dom] = {
                "leaves": [],
                "levels": [],
                "level_of": {},
            }
        return d

    def walk_map(root_mp: Any, expected: str) -> None:
        """Register one map term into its domain's leaf/level tables.

        Post-order over the compose structure — iterative, because a
        prefix chain nests O(T) deep.
        """
        nonlocal bad
        stack = [(root_mp, expected, False)]
        while stack:
            mp, exp, done = stack.pop()
            tid = mp
            if (
                isinstance(mp, Op)
                and mp.op in _COMPOSE_OP
                and len(mp.args) == 2
            ):
                dom = _COMPOSE_OP[mp.op]
                if dom != exp:
                    bad = True
                    continue
                d = _ctx(dom)
                if tid in d["level_of"]:
                    continue
                if not done:
                    stack.append((mp, exp, True))
                    for a in mp.args:
                        stack.append((a, dom, False))
                    continue
                lv = (
                    max(d["level_of"].get(a, 0) for a in mp.args)
                    + 1
                )
                d["level_of"][tid] = lv
                while len(d["levels"]) < lv:
                    d["levels"].append([])
                d["levels"][lv - 1].append(mp)
                continue
            # leaf position
            if (
                isinstance(mp, Op)
                and mp.op in _LEAF_OP
                and len(mp.args) == 2
            ):
                dom = _LEAF_OP[mp.op]
                if dom != exp:
                    bad = True
                    continue
            else:
                dom = exp  # opaque leaf (any term evaluating to a pair)
            d = _ctx(dom)
            if tid in d["level_of"]:
                continue
            d["level_of"][tid] = 0
            d["leaves"].append(mp)
            if isinstance(mp, Op):
                for a in mp.args:
                    scan(a, inside_map_leaf=True)

    for leaf in omd_leaves:
        for a in leaf.args:
            scan(a)
    scan(h_term)
    if bad:
        return None

    # Worklist: scanning a leaf's args can register further maps.
    queue = list(targets.values())
    qi = 0
    while qi < len(queue):
        mp, dom = queue[qi]
        qi += 1
        before = len(targets)
        walk_map(mp, dom)
        if len(targets) > before:
            queue.extend(list(targets.values())[before:])
    if bad:
        return None

    plan: dict[str, Any] = {
        "apply_op": root.op,
        "h": h_term,
        "omd_leaves": omd_leaves,
        "omd_gather": omd_gather,
        "omd_root": slot[f_term],
        "map_mode": None,
        "proj_seeds": [],
        "stack_seeds": [],
        "map_seeds": [],
    }
    if not targets:
        return plan

    def _sig_uniform(leaves: list, dom: str) -> bool:
        sig = None
        for leaf in leaves:
            s = _leaf_sig(leaf, dom)
            if s is False:
                return False
            if s is not None:
                if sig is None:
                    sig = s
                elif s != sig:
                    return False
        return True

    # ---- chain check: every projected map a prefix of one sequence ---
    seqs: dict[int, list] = {}
    for tid, (mp, _dom) in targets.items():
        seq: list = []
        _flatten_map(mp, seq)
        seqs[tid] = seq
    base_tid = max(seqs, key=lambda k: len(seqs[k]))
    base = seqs[base_tid]
    tdoms = {dom for _, dom in targets.values()}
    chain = len(tdoms) == 1
    if chain:
        for seq in seqs.values():
            if seq != base[: len(seq)]:
                chain = False
                break
    if chain:
        dom = next(iter(tdoms))
        if any(
            isinstance(lf, Op) and _LEAF_OP.get(lf.op) not in (None, dom)
            for lf in base
        ) or not _sig_uniform(base, dom):
            chain = False

    if chain:
        src = ("chain",)
        pos_of = {tid: len(seqs[tid]) - 1 for tid in seqs}
        plan["map_mode"] = "chain"
        plan["chain_domain"] = dom
        plan["chain_leaves"] = base
        plan["chain_a_gather"] = _part_gather(base, 0)
        plan["chain_b_gather"] = _part_gather(base, 1)
        for node, ci, mp in raw_proj:
            plan["proj_seeds"].append((node, ci, src, pos_of[mp]))
        for node, ci, _d, maps, dim in raw_stack:
            plan["stack_seeds"].append(
                (node, ci, src, [pos_of[m] for m in maps], dim)
            )
        for tid, (mp, _d) in targets.items():
            plan["map_seeds"].append((mp, src, pos_of[tid]))
        return plan

    # ---- forest mode: per-domain level-batched compose ---------------
    for dom, d in domains.items():
        if not _sig_uniform(d["leaves"], dom):
            return None
        slot_d = {lf: i for i, lf in enumerate(d["leaves"])}
        nxt_d = len(d["leaves"])
        gather = []
        for nodes in d["levels"]:
            gather.append(
                (
                    [slot_d[t.args[0]] for t in nodes],
                    [slot_d[t.args[1]] for t in nodes],
                )
            )
            for t in nodes:
                slot_d[t] = nxt_d
                nxt_d += 1
        d["slot"] = slot_d
        d["gather"] = gather
        d["a_gather"] = _part_gather(d["leaves"], 0)
        d["b_gather"] = _part_gather(d["leaves"], 1)
    plan["map_mode"] = "forest"
    plan["forest"] = domains
    for node, ci, mp in raw_proj:
        dom = targets[mp][1]
        plan["proj_seeds"].append(
            (node, ci, ("forest", dom), domains[dom]["slot"][mp])
        )
    for node, ci, dom, maps, dim in raw_stack:
        plan["stack_seeds"].append(
            (
                node,
                ci,
                ("forest", dom),
                [domains[dom]["slot"][m] for m in maps],
                dim,
            )
        )
    for _tid, (mp, dom) in targets.items():
        plan["map_seeds"].append(
            (mp, ("forest", dom), domains[dom]["slot"][mp])
        )
    return plan


class BatchedOmdModule(torch.nn.Module):
    """nn.Module running an ``omd_apply[m](omd-tree, h)`` term batched.

    Parameters
    ----------
    ir : IR | term
        The (post-extraction) IR to lower — or a bare root term, which
        is wrapped in ``IR(root=term)`` (param discovery still works;
        single-input graphs resolve Vars against the argument).
    param_values : dict[str, torch.Tensor] | None
        As for :func:`~catopt.torch_bridge.ir_to_torch_module`.

    Embeds a plain :class:`IRModule` (submodule ``eval_mod``) used for
    leaf/score/state evaluation and as the complete fallback when the
    root is not an omd application — a drop-in replacement for
    ``ir_to_torch_module`` on ANY IR.  Any inconsistency the batched
    path hits at runtime likewise falls back to serial evaluation.

    Ports layer: conforms to :class:`catopt.ports.BatchedExecutor`
    (``forward`` + ``_plan`` + ``is_batched``; ``eval_mod`` is the
    semantic member — see :class:`catopt.ports.PlannedExecutor` for why
    it is not the runtime-checked one).  ``map_mode`` / ``fallbacks`` /
    ``is_graph_captured`` stay class-level API, not port members.
    """

    def __init__(
        self,
        ir: Any,
        param_values: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(ir, IR):
            ir = IR(root=ir)
        self._inputs = ir.inputs
        self.eval_mod = IRModule(ir, param_values)
        # Plan from the FOLDED root — IRModule may have rewritten
        # weight-only subtrees into fused Params at construction.
        self._plan = build_omd_plan(self.eval_mod._root)
        self._const_cache: dict[tuple, torch.Tensor] = {}
        self._graph = None
        self._graph_inputs: list[torch.Tensor] = []
        self._graph_out: torch.Tensor | None = None
        self.fallbacks = 0  # times the batched path declined at runtime

    @property
    def is_batched(self) -> bool:
        """True when the root matched the omd-apply pattern."""
        return self._plan is not None

    @property
    def map_mode(self) -> str | None:
        """``"chain"`` / ``"forest"`` / ``None`` — how the coefficient
        maps inside the leaves are evaluated (None when the leaves hold
        no map projections or the term is not omd-shaped)."""
        return None if self._plan is None else self._plan["map_mode"]

    @property
    def is_graph_captured(self) -> bool:
        return self._graph is not None

    def capture_cuda_graph(
        self,
        *example_inputs: torch.Tensor,
        warmup: int = 3,
    ) -> BatchedOmdModule:
        """Record the batched forward into a CUDA graph.

        Same contract as
        :meth:`catopt.scan_lower.BatchedScanModule.capture_cuda_graph`:
        static input buffers are copied into and the returned tensor is
        the static output (clone it to keep it).
        """
        if self._plan is None or not torch.cuda.is_available():
            return self
        if not all(t.is_cuda for t in example_inputs):
            raise ValueError("capture_cuda_graph requires CUDA inputs")
        static_ins = [t.detach().clone() for t in example_inputs]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self._forward_impl(*static_ins)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self._forward_impl(*static_ins)
        self._graph, self._graph_inputs, self._graph_out = (
            g,
            static_ins,
            out,
        )
        return self

    def drop_cuda_graph(self) -> None:
        self._graph = None
        self._graph_inputs = []
        self._graph_out = None

    # -- execution ----------------------------------------------------

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        g = self._graph
        if (
            g is not None
            and len(xs) == len(self._graph_inputs)
            and all(
                t.shape == b.shape
                and t.dtype == b.dtype
                and t.device == b.device
                for t, b in zip(xs, self._graph_inputs, strict=True)
            )
        ):
            for buf, t in zip(self._graph_inputs, xs, strict=True):
                buf.copy_(t, non_blocking=True)
            g.replay()
            return self._graph_out
        return self._forward_impl(*xs)

    def _cached(
        self,
        key: tuple,
        like: torch.Tensor,
        make,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        want = dtype or like.dtype
        t = self._const_cache.get(key)
        if t is None or t.device != like.device or t.dtype != want:
            t = make(like)
            self._const_cache[key] = t
        return t

    def _gidx(self, slots, like: torch.Tensor) -> torch.Tensor:
        return self._cached(
            ("idx", tuple(slots)),
            like,
            lambda t: torch.tensor(
                slots, dtype=torch.long, device=t.device
            ),
            dtype=torch.long,
        )

    def _leaf_part(self, leaf: Any, i: int, ev) -> torch.Tensor:
        """Part i of a map leaf's value — for ``aff``/``aff_diag`` the
        arg directly; for opaque leaves ``v[i]`` mirrors the generic
        ``f[0]``/``f[1]`` indexing exactly (tuple or tensor)."""
        if (
            isinstance(leaf, Op)
            and leaf.op in _LEAF_OP
            and len(leaf.args) == 2
        ):
            return ev(leaf.args[i])
        v = ev(leaf)
        return v[i]

    def _eval_leaf_parts(self, leaves: list, gather_a, gather_b, ev):
        """Stack all leaf (a, b) parts into (n, ...) tensors."""
        if gather_a is not None:
            base, dim, idx = gather_a
            bt = ev(base)
            if idx == list(range(bt.shape[dim])):
                A = bt
            else:
                A = bt.index_select(dim, self._gidx(idx, bt))
            if dim != 0:
                A = A.movedim(dim, 0)
        else:
            A = torch.stack(
                [self._leaf_part(lf, 0, ev) for lf in leaves]
            )
        if gather_b is not None:
            base, dim, idx = gather_b
            bt = ev(base)
            if idx == list(range(bt.shape[dim])):
                B = bt
            else:
                B = bt.index_select(dim, self._gidx(idx, bt))
            if dim != 0:
                B = B.movedim(dim, 0)
        else:
            B = torch.stack(
                [self._leaf_part(lf, 1, ev) for lf in leaves]
            )
        return A, B

    @staticmethod
    def _id_map(a_shape, b_shape, domain, like):
        """The monoid identity map value: ``(1, 0)`` diag / ``(I, 0)``
        dense — padding with it is exact (1·a = a, a·0 + b = b)."""
        if domain == "dense":
            i = a_shape[-1]
            eye = torch.eye(i, dtype=like.dtype, device=like.device)
            ida = eye.reshape(*([1] * (len(a_shape) - 2)), i, i)
            ida = ida.expand(*a_shape)
        else:
            ida = torch.ones(
                *a_shape, dtype=like.dtype, device=like.device
            )
        idb = torch.zeros(
            *b_shape, dtype=like.dtype, device=like.device
        )
        return ida, idb

    def _prefix_scan(
        self, A: torch.Tensor, B: torch.Tensor, domain: str
    ):
        """All prefix maps of the leaf sequence ``(A_t, B_t)``.

        Two-level blocked associative scan: local prefixes within
        blocks of ~√T (batched across blocks), a carry fold across
        block totals, then one batched compose of locals onto carries.
        O(√T) sequential batched steps, O(T) work — vs O(T) serial
        compose calls for the unrolled chain.  Padding lanes carry the
        exact identity so the tail block needs no masking.
        """
        T = A.shape[0]
        if T == 1:
            return A, B
        blk = math.ceil(math.sqrt(T))
        C = (T + blk - 1) // blk
        pad = C * blk - T
        shape_a, shape_b = A.shape[1:], B.shape[1:]
        ida, idb = self._id_map(shape_a, shape_b, domain, A)
        if pad:
            A = torch.cat([A, ida.unsqueeze(0).expand(pad, *shape_a)])
            B = torch.cat([B, idb.unsqueeze(0).expand(pad, *shape_b)])
        A = A.reshape(C, blk, *shape_a)
        B = B.reshape(C, blk, *shape_b)

        las = [A[:, 0]]
        lbs = [B[:, 0]]
        for k in range(1, blk):
            na, nb = _compose_pair(
                A[:, k], B[:, k], las[-1], lbs[-1], domain
            )
            las.append(na)
            lbs.append(nb)
        La = torch.stack(las, dim=1)  # (C, blk, ...) local prefixes
        Lb = torch.stack(lbs, dim=1)

        # carry into block c = total of blocks < c (latest leftmost)
        cpa = [ida]
        cpb = [idb]
        for c in range(1, C):
            na, nb = _compose_pair(
                La[c - 1, -1], Lb[c - 1, -1], cpa[-1], cpb[-1], domain
            )
            cpa.append(na)
            cpb.append(nb)
        CPa = torch.stack(cpa, dim=0).unsqueeze(1)  # (C, 1, ...)
        CPb = torch.stack(cpb, dim=0).unsqueeze(1)
        Pa, Pb = _compose_pair(La, Lb, CPa, CPb, domain)
        n = C * blk
        return (
            Pa.reshape(n, *shape_a)[:T],
            Pb.reshape(n, *shape_b)[:T],
        )

    def _forward_impl(self, *xs: torch.Tensor) -> torch.Tensor:
        plan = self._plan
        if plan is None:
            return self.eval_mod(*xs)
        x = xs[0] if xs else None
        env: dict[str, torch.Tensor] = {"self": x}
        for i, inp in enumerate(self._inputs):
            env[inp.name] = xs[i] if i < len(xs) else x
        memo: dict[Any, Any] = {}

        def ev(t: Any):
            return self.eval_mod._eval(t, env, x, memo)

        try:
            # ---- 1. coefficient maps ---------------------------------
            stores: dict[tuple, tuple] = {}
            mode = plan["map_mode"]
            if mode == "chain":
                A, B = self._eval_leaf_parts(
                    plan["chain_leaves"],
                    plan["chain_a_gather"],
                    plan["chain_b_gather"],
                    ev,
                )
                stores[("chain",)] = self._prefix_scan(
                    A, B, plan["chain_domain"]
                )
            elif mode == "forest":
                for dom, d in plan["forest"].items():
                    A, B = self._eval_leaf_parts(
                        d["leaves"], d["a_gather"], d["b_gather"], ev
                    )
                    for f_idx, g_idx in d["gather"]:
                        ixf = self._gidx(f_idx, A)
                        ixg = self._gidx(g_idx, A)
                        an, bn = _compose_pair(
                            A.index_select(0, ixf),
                            B.index_select(0, ixf),
                            A.index_select(0, ixg),
                            B.index_select(0, ixg),
                            dom,
                        )
                        A = torch.cat([A, an])
                        B = torch.cat([B, bn])
                    stores[("forest", dom)] = (A, B)

            # ---- 2. memo seeds: projections / stacks / map tuples ----
            views = {
                k: (v[0].unbind(0), v[1].unbind(0))
                for k, v in stores.items()
            }
            for node, ci, src, pos in plan["proj_seeds"]:
                memo[node] = views[src][ci][pos]
            for mp, src, pos in plan["map_seeds"]:
                memo[mp] = (views[src][0][pos], views[src][1][pos])
            for node, ci, src, poss, dim in plan["stack_seeds"]:
                srcs = stores[src][ci]
                t = srcs.index_select(0, self._gidx(poss, srcs))
                dn = dim % t.dim()
                if dn != 0:
                    t = t.movedim(0, dn)
                memo[node] = t

            # ---- 3. omd leaves ---------------------------------------
            omd_elem = _IR_TO_TORCH["omd_elem"]
            vals: list = []
            for leaf in plan["omd_leaves"]:
                if leaf.op == "omd_elem" and len(leaf.args) == 3:
                    vals.append(
                        omd_elem(
                            ev(leaf.args[0]),
                            ev(leaf.args[1]),
                            ev(leaf.args[2]),
                        )
                    )
                else:
                    vals.append(ev(leaf))

            # ---- 4. compose levels ------------------------------------
            uniform = (
                bool(vals)
                and all(
                    isinstance(v, tuple) and len(v) == 4 for v in vals
                )
                and all(
                    all(
                        v[c].shape == vals[0][c].shape for c in range(4)
                    )
                    for v in vals[1:]
                )
            )
            if uniform:
                m_all = torch.stack([v[0] for v in vals])
                l_all = torch.stack([v[1] for v in vals])
                fa_all = torch.stack([v[2] for v in vals])
                fb_all = torch.stack([v[3] for v in vals])
                for f_idx, g_idx in plan["omd_gather"]:
                    ixf = self._gidx(f_idx, m_all)
                    ixg = self._gidx(g_idx, m_all)
                    m1, l1 = (
                        t.index_select(0, ixf) for t in (m_all, l_all)
                    )
                    fa1, fb1 = (
                        t.index_select(0, ixf) for t in (fa_all, fb_all)
                    )
                    m2, l2 = (
                        t.index_select(0, ixg) for t in (m_all, l_all)
                    )
                    fa2, fb2 = (
                        t.index_select(0, ixg) for t in (fa_all, fb_all)
                    )
                    # _omd_compose verbatim, batched over slot dim.
                    mx = torch.maximum(m1, m2)
                    fin1, fin2 = torch.isfinite(m1), torch.isfinite(m2)
                    e1 = torch.where(
                        fin1, torch.exp(m1 - mx), torch.zeros_like(mx)
                    )
                    e2 = torch.where(
                        fin2, torch.exp(m2 - mx), torch.zeros_like(mx)
                    )
                    l_n = torch.where(
                        fin1, l1 * e1, torch.zeros_like(l1)
                    ) + torch.where(fin2, l2 * e2, torch.zeros_like(l2))
                    fa_n = torch.where(
                        fin1, fa1 * e1, torch.zeros_like(fa1)
                    ) + torch.where(
                        fin2, fa2 * e2, torch.zeros_like(fa2)
                    )
                    fb_n = torch.where(
                        fin1, fb1 * e1, torch.zeros_like(fb1)
                    ) + torch.where(
                        fin2, fb2 * e2, torch.zeros_like(fb2)
                    )
                    m_all = torch.cat([m_all, mx])
                    l_all = torch.cat([l_all, l_n])
                    fa_all = torch.cat([fa_all, fa_n])
                    fb_all = torch.cat([fb_all, fb_n])
                r = plan["omd_root"]
                root_t = (m_all[r], l_all[r], fa_all[r], fb_all[r])
            else:
                omd_compose = _IR_TO_TORCH["omd_compose"]
                for f_idx, g_idx in plan["omd_gather"]:
                    for fi, gi in zip(f_idx, g_idx, strict=True):
                        vals.append(omd_compose(vals[fi], vals[gi]))
                root_t = vals[plan["omd_root"]]

            # ---- 5. root apply ---------------------------------------
            return _IR_TO_TORCH[plan["apply_op"]](root_t, ev(plan["h"]))
        except Exception:
            # Any shape/pattern surprise: the serial evaluator is always
            # correct — fall back rather than fail a drop-in lowering.
            self.fallbacks += 1
            return self.eval_mod(*xs)


def to_batched_omd_module(
    ir: Any,
    param_values: dict[str, torch.Tensor] | None = None,
) -> BatchedOmdModule:
    """Lower ``ir`` (or a bare term) to a module, batching any leading
    ``omd_apply[m]`` term.  Non-omd roots transparently delegate to the
    serial IRModule evaluator (check ``mod.is_batched``)."""
    return BatchedOmdModule(ir, param_values=param_values)
