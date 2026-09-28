"""Level-batched lowering for parallel-scan (affine-monoid) IR terms.

After eqsat with ``SCAN_LAWS`` and a depth cost, an unrolled recurrence
``h_t = A·h_{t-1} + x_t`` extracts as::

    apply(<balanced aff_compose tree>, h0)

—a Blelloch-style scan of depth ~2·log2(T).  The generic
:class:`~catopt_torch.torch_bridge.IRModule` evaluates ``aff``/``aff_compose``
by passing ``(A, b)`` tuples through the tree one node at a time: the
parallelism is in the graph but the schedule is serial.

This module lowers the same term *level-batched*.  Each affine map
``(A, b)`` is packed into its homogeneous matrix ``[[A, b], [0, 1]]``,
so ``aff_compose(f, g)`` is literally ``M_f @ M_g`` — one batched
matmul per tree level plus two index_select gathers.  That is O(log T)
kernel launches instead of O(T); optionally the whole forward can be
recorded into a CUDA graph (``capture_cuda_graph``) so a single
``graph.replay()`` replaces every launch.

Usage::

    mod = to_batched_scan_module(opt_ir, param_values=source)
    out = mod(x)
    mod.capture_cuda_graph(x)      # optional, CUDA only
    out = mod(x)                   # graph replay

``to_batched_scan_module`` (and :class:`BatchedScanModule` itself)
detects the ``apply(aff-tree, h)`` shape; any other IR falls back to
the ordinary tuple-passing IRModule evaluation.

``fused=True``/``"compile"`` selects the fused level-step from
:mod:`~catopt_carriers.scan_fused`: the leaf product is *re-bracketed*
into the canonical adjacent-pair reduction — the same composed map
(the monoid product is association-invariant) — and the level loop
runs through ``torch.compile`` so Inductor fuses it to a handful of
kernels; ``fused="eager"`` runs the same schedule without compile.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from catopt_core.ir import IR, Op, Param, Var
from catopt_core.typing import (
    _INVALID,
    _broadcast,
    _matmul_shape,
    _shape_of,
)
from catopt_torch.executors import (
    BatchedExecutorBase,
    level_schedule,
    slot_gathers,
)
from catopt_torch.torch_bridge import IRModule

from catopt_carriers.scan_fused import (
    fused_dense_levels,
    fused_diag_levels,
    occurrence_slots,
)

__all__ = [
    "BatchedScanModule",
    "build_scan_plan",
    "is_scan_apply_term",
    "to_batched_scan_module",
]


#: Pure view/bookkeeping ops — single-arg terms that re-lay their
#: input without computing on it (the same convention as
#: ``cost._VIEW_OPS``, extended with the remaining single-arg aten
#: spellings the exporter emits).  The scan plan is transparent to
#: them on BOTH ends of the term: a chain's base may hide under one
#: (``reshape(h0)`` inits — see ``trace_lift._Spine._is_base``), and
#: the apply root may sit under them (``reshape(applyd(…), (B, d))``
#: when the module reshapes its carried state on the way out) — the
#: batched executor re-applies stripped output views verbatim through
#: the ambient torch bindings, so the lowered value is unchanged.
_VIEW_OPS = frozenset(
    {
        "reshape",
        "view",
        "flatten",
        "unflatten",
        "squeeze",
        "unsqueeze",
        "expand",
        "broadcast_to",
        "contiguous",
        "clone",
        "detach",
        "to",
        "alias",
        "transpose",
        "permute",
        "movedim",
        "t",
        "select",
        "getitem",
        "slice",
        "chunk",
        "split",
        "unbind",
    }
)


def _strip_output_views(root: Any) -> tuple[Any, list]:
    """Peel trailing ``_VIEW_OPS`` off a would-be scan root.

    Returns ``(inner_root, post)`` where ``post`` lists the stripped
    ``(op, attrs)`` wrappers OUTERMOST-first; re-applying them in
    reverse order reproduces the original term's value exactly.
    """
    post: list = []
    while (
        isinstance(root, Op)
        and root.op in _VIEW_OPS
        and len(root.args) == 1
    ):
        post.append((root.op, dict(root.attrs)))
        root = root.args[0]
    return root, post


def _fold_nested_apply(term: Any) -> Any:
    """``apply(f, apply(g, h))`` → ``apply(aff_compose(f, g), h)``.

    Also the ``applyd``/``affd_compose`` diagonal pair.

    Extracted terms are often hybrids — a compose spine with nested
    ``apply`` segments.  Folding them bottom-up turns the whole map
    into a single compose tree the plan builder understands.
    Idempotent and semantics-preserving (composition IS sequential
    application).
    """
    memo: dict[int, Any] = {}

    def go(t: Any) -> Any:
        if not isinstance(t, Op):
            return t
        k = id(t)
        if k in memo:
            return memo[k]
        args = [go(a) for a in t.args]
        t = Op.make(t.op, *args, **t.attrs) if args else t
        if t.op in ("apply", "applyd"):
            comp = "affd_compose" if t.op == "applyd" else "aff_compose"
            f, h = t.args
            while isinstance(h, Op) and h.op == t.op:
                f = Op.make(comp, f, h.args[0])
                h = h.args[1]
            t = Op.make(t.op, f, h)
        memo[k] = t
        return t

    return go(term)


def _is_aff_tree(term: Any, memo: dict | None = None) -> str | None:
    """Return True if ``term`` is a pure map tree in ONE carrier domain.

    Dense: leaves ``aff(A, b)``, internal ``aff_compose``.  Diagonal:
    leaves ``aff_diag(a, b)``, internal ``affd_compose``.  Mixed or
    foreign nodes disqualify the subtree.  Memoised on id() because
    extracted terms are DAGs with shared subtrees.

    Returns the shared carrier DOMAIN (``"aff"``/``"aff_diag"``) when
    the tree is pure, ``None`` otherwise — the domain string doubles as
    the truthy verdict, so the annotation is ``str | None``, not
    ``bool``.
    """
    memo = {} if memo is None else memo
    key = id(term)
    if key in memo:
        return memo[key]
    dom = None
    if isinstance(term, Op) and len(term.args) == 2:
        if term.op in ("aff", "aff_diag"):
            dom = term.op
        elif term.op in ("aff_compose", "affd_compose"):
            kids = [_is_aff_tree(a, memo) for a in term.args]
            # Domain purity: all leaves one carrier, and the compose
            # op must match it (aff_compose over aff leaves, etc.).
            want = {"aff": "aff_compose", "aff_diag": "affd_compose"}
            ok = (
                kids[0] is not None
                and kids[0] == kids[1]
                and term.op == want[kids[0]]
            )
            dom = kids[0] if ok else None
    memo[key] = dom
    return dom


def is_scan_apply_term(root: Any) -> bool:
    """Return True for an ``apply[d](<map tree>, h)`` root.

    Dense or diagonal affine scan application (nested ``apply`` segments
    are first folded into the compose spine), possibly under trailing
    view wrappers (``reshape(applyd(…), (B, d))``).
    """
    root = _fold_nested_apply(root)
    root, _post = _strip_output_views(root)
    return (
        isinstance(root, Op)
        and root.op in ("apply", "applyd")
        and len(root.args) == 2
        and _is_aff_tree(root.args[0]) is not None
        and {"apply": "aff", "applyd": "aff_diag"}[root.op]
        == _is_aff_tree(root.args[0])
    )


def _concrete_tuple(s) -> bool:
    return isinstance(s, tuple) and all(
        isinstance(d, int) and d > 0 for d in s
    )


def _leaf_shapes_consistent(leaves: list[Op]) -> bool:
    """Check the ``aff``/``aff_diag`` leaves form one batched family.

    Diagonal carrier: the (a, b) parts broadcast pointwise, so leaves
    need only broadcast into ONE concrete per-slot shape — a shared
    ``(d,)`` decay under ``(B, d)`` inputs is one family (the executor
    normalises each part up to the join before stacking).

    Dense carrier: the linear parts must share one map signature
    ``A = (*P, d, d)`` and every b must be the vector ``A[:-1]`` or
    the column ``(*P, d, 1)`` — exactly the shapes the homogeneous
    ``[[A, b], [0, 1]]`` packing and the literal ``f[0] @ g[1]``
    compose evaluate.  Matmul-shape checks (``A@A == A``,
    ``A@B0 == B0``) keep the family closed under composition.

    If inference cannot prove consistency we decline the plan and the
    caller falls back to the serial evaluator (which is correct for
    any shape).
    """
    if not leaves:
        return False
    a0 = cast("tuple", _shape_of(leaves[0].args[0]))
    b0 = cast("tuple", _shape_of(leaves[0].args[1]))
    if leaves[0].op == "aff_diag":
        if not (_concrete_tuple(a0) and _concrete_tuple(b0)):
            return False
        eff = _broadcast(a0, b0)
        if eff is _INVALID:
            return False
        for leaf in leaves[1:]:
            a = _shape_of(leaf.args[0])
            b = _shape_of(leaf.args[1])
            if not (_concrete_tuple(a) and _concrete_tuple(b)):
                return False
            eff = _broadcast(eff, a)
            if eff is _INVALID:
                return False
            eff = _broadcast(eff, b)
            if eff is _INVALID:
                return False
        return True
    ok = (
        _concrete_tuple(a0)
        and _concrete_tuple(b0)
        and len(a0) >= 2
        and _matmul_shape(a0, a0) == a0
        and _matmul_shape(a0, b0) == b0
        # vector-b (*P, d) or column-b (…, d, 1) — the signatures
        # _leaf_homogeneous packs and the level matmuls close over.
        # A shared (d,d) map may broadcast into a batched (B,d,1) b.
        and (
            b0 == a0[:-1]
            or (len(b0) >= 2 and b0[-1] == 1 and b0[-2] == a0[-2])
        )
    )
    if not ok:
        return False
    for leaf in leaves[1:]:
        if (
            _shape_of(leaf.args[0]) != a0
            or _shape_of(leaf.args[1]) != b0
        ):
            return False
    return True


def _select_index(term: Any) -> tuple[Any, int, int] | None:
    """Decompose ``select(base, dim, i)``/getitem-style leaf operands.

    Returns ``(base_term, dim, index)`` or ``None``.  Covers the
    ``select(x, dim=…, index=i)`` shape the export boundary emits for
    ``x[i]`` indexing.
    """
    if (
        not isinstance(term, Op)
        or term.op not in ("select", "getitem")
        or len(term.args) != 1
    ):
        return None
    if term.op == "getitem":
        # t[i] — index IS the index; always dim 0 (see the
        # torch_bridge binding).  Do not conflate with select.
        idx = term.attrs.get("index")
        if not isinstance(idx, int):
            return None
        return (term.args[0], 0, idx)
    dim = term.attrs.get("dim", 0)
    idx = term.attrs.get("index")
    if not isinstance(dim, int) or not isinstance(idx, int):
        return None
    return (term.args[0], dim, idx)


def _leaf_b_gather(leaves: list[Op]):
    """Recognise leaf b-parts as ``base[i]`` slices of one tensor.

    Recurrence inputs arrive as ``select(x, 0, t)`` per step — evaluating
    T of them is T tiny aten calls, which dominates a launch-bound
    forward.  When every leaf's b is an index of the SAME base term
    along the SAME dim, the whole leaf level collapses to a single
    ``index_select`` (or the base itself when indices are contiguous).

    Returns ``(base_term, dim, index_list)`` or ``None``.
    """
    parts = [_select_index(leaf.args[1]) for leaf in leaves]
    if any(p is None for p in parts):
        return None
    parts = cast("list[tuple]", parts)
    base0, dim0 = parts[0][0], parts[0][1]
    if any(p[0] is not base0 or p[1] != dim0 for p in parts):
        return None
    return (base0, dim0, [p[2] for p in parts])


def build_scan_plan(root: Any) -> dict | None:
    """Analyse ``apply(aff-tree, h)`` into a level-batched schedule.

    Returns ``None`` when the term is not a scan application (or the
    leaf shapes are non-uniform).  Otherwise a dict:

    * ``"leaves"`` — the ``aff`` nodes, in first-encounter order.
    * ``"levels"`` — ``levels[k]`` lists the ``aff_compose`` nodes whose
      longest child chain has length k+1; every node in a level is
      computable once all earlier levels are done.  Children may sit
      at ANY earlier level (extracted bracketings are not perfectly
      balanced), so levels are indexed by depth.
    * ``"level_gather"`` — per level, ``(f_slots, g_slots)`` indexing a
      running tensor of all aff-matrix results computed so far (leaf
      slots 0..n-1 first, then each level's outputs in order).  These
      are plan-time constants — no per-forward tree walk.
    * ``"root_slot"`` — slot of the complete composed map.
    * ``"leaf_b_gather"`` — optional ``(base_term, dim, indices)`` fast
      path: leaf b-parts are slices of one base tensor.
    * ``"leaf_a_shared"`` — True when every leaf's linear part is the
      same term (LTI recurrence): evaluate once, expand as a stride-0
      batch view instead of stacking T copies.
    * ``"post"`` — shape-only ``(op, attrs)`` wrappers stripped from
      the root (outermost first); the executor re-applies them to the
      scan result.  Empty for a bare apply/applyd root.
    * ``"column_state"`` — dense plans whose b-part is the column
      convention ``(*P, d, 1)`` (batched dense states).
    * ``"f"`` / ``"h"`` — the map term and the applied-to term.
    """
    root = _fold_nested_apply(root)
    root, post = _strip_output_views(root)
    if not is_scan_apply_term(root):
        return None
    f_term, h_term = root.args
    diag = root.op == "applyd"

    leaf_op = "aff_diag" if root.op == "applyd" else "aff"
    # Level schedule, id()-keyed like the historical visit: post-fold
    # the tree is Op.make-interned, so term keys would be equivalent —
    # id() simply makes the per-object dedup explicit.
    leaves, levels = level_schedule(
        f_term,
        lambda t: t.args,
        lambda t: t.op == leaf_op,
        key=id,
    )
    if not _leaf_shapes_consistent(leaves):
        return None

    # Slot assignment: leaves occupy 0..n-1, then each level appends its
    # outputs in order — so a level's operand slots are all < its own.
    slot: dict[int, int] = {id(lf): i for i, lf in enumerate(leaves)}
    level_gather = slot_gathers(
        levels, lambda t: t.args, slot, len(leaves), key=id
    )

    a0 = leaves[0].args[0]
    leaf_a_shared = all(
        leaf.args[0] is a0
        or (
            isinstance(leaf.args[0], Param)
            and isinstance(a0, Param)
            and leaf.args[0].name == a0.name
        )
        for leaf in leaves
    )
    b0s = _shape_of(leaves[0].args[1])
    column_state = bool(
        not diag
        and isinstance(b0s, tuple)
        and len(b0s) >= 2
        and b0s[-1] == 1
    )
    return {
        "leaves": leaves,
        "levels": levels,
        "level_gather": level_gather,
        "root_slot": slot[id(f_term)],
        "leaf_b_gather": _leaf_b_gather(leaves),
        "leaf_a_shared": leaf_a_shared,
        "diagonal": diag,
        "post": post,
        "column_state": column_state,
        "f": f_term,
        "h": h_term,
    }


class BatchedScanModule(BatchedExecutorBase, torch.nn.Module):
    """nn.Module that runs an ``apply(aff-tree, h)`` term level-batched.

    Parameters
    ----------
    ir : IR
        The (post-extraction) IR to lower.
    param_values : dict[str, torch.Tensor] | None
        Mapping from IR param names to tensor values, as for
        :func:`~catopt_torch.torch_bridge.ir_to_torch_module`.

    The module embeds a plain :class:`IRModule` (as submodule
    ``eval_mod``) used both for leaf/h evaluation and as the complete
    fallback when the root is not a scan application — so this class is
    a drop-in replacement for ``ir_to_torch_module`` on ANY IR.

    For launch-bound workloads (small d, large T) the eager path is
    still CPU-dispatch limited; :meth:`capture_cuda_graph` records the
    whole batched forward into a CUDA graph so one ``graph.replay()``
    replaces all per-level launches.

    Ports layer: conforms to :class:`catopt_core.ports.BatchedExecutor`
    (``forward`` + ``_plan`` + ``is_batched``; ``eval_mod`` is the
    semantic member — see :class:`catopt_core.ports.PlannedExecutor` for why
    it is not the runtime-checked one).  ``n_levels`` /
    ``is_graph_captured`` stay class-level API, not port members.

    """

    def __init__(
        self,
        ir: IR,
        param_values: dict[str, torch.Tensor] | None = None,
        *,
        fused: bool | str = False,
    ) -> None:
        """Initialise the module, plan, and graph state."""
        super().__init__()
        self._inputs = ir.inputs
        self.eval_mod = IRModule(ir, param_values)
        # Plan from the FOLDED root: IRModule may have rewritten
        # weight-only subtrees into fused Params during construction.
        self._plan = build_scan_plan(self.eval_mod._root)
        self._const_cache: dict[tuple, torch.Tensor] = {}
        self._init_graph_state()
        self._init_fused(fused)
        gather = (
            self._plan["leaf_b_gather"]
            if self._plan is not None
            else None
        )
        # ``indices == range(len)`` is a plan-time constant — caching it
        # turns the per-forward contiguous check into a len compare.
        self._b_identity = gather is not None and gather[2] == list(
            range(len(gather[2]))
        )

    def _init_fused(self, fused: bool | str) -> None:
        """Resolve the ``fused`` flag against the plan.

        ``self._fused`` ends up ``None`` (standard level-gather
        schedule), ``"eager"`` (canonical adjacent-pair reduction run
        eagerly), or ``"compile"`` (the same body through
        ``torch.compile(fullgraph=True)`` — requested by
        ``fused=True``/``"compile"``).  The fused schedule
        *re-brackets* the leaf sequence into adjacent pairs: the
        monoid product is association-invariant, so the result is the
        same composed map, but every level becomes a pure strided
        pointwise/matmul step on a shrinking fresh tensor — no
        gathers, no growing-buffer ``cat``, and Inductor fuses the
        whole loop to a handful of kernels.

        ``self._fused_occ`` holds the leaf-occurrence slots when the
        extracted term is a DAG (shared leaves/subtrees) — ``None``
        for the common tree case where occurrences are already in
        order.  ``occurrence_slots`` declining (pathological sharing)
        keeps the standard schedule.
        """
        self._fused: str | None = None
        self._fused_occ: list[int] | None = None
        self._fused_c: Any = None
        self._fused_compile_failed = False
        if not fused or self._plan is None:
            return
        if self._plan["column_state"]:
            # fused_dense_levels' root slice is written for (d+1)²
            # matrices only — column-state dense plans keep the
            # standard schedule (fused_diag_levels already broadcasts
            # batched (n, …, d) leaves).
            return
        if fused not in (True, "eager", "compile"):
            raise ValueError(
                f"fused must be True, 'eager', or 'compile'; got {fused!r}"
            )
        occ = occurrence_slots(self._plan["f"], self._plan["leaves"])
        if occ is None:
            return
        if occ != list(range(len(occ))):
            self._fused_occ = occ
        self._fused = "eager" if fused == "eager" else "compile"

    @property
    def is_batched(self) -> bool:
        """True when the root matched the scan pattern."""
        return self._plan is not None

    @property
    def n_levels(self) -> int:
        """Number of batched compose levels (0 when not batched)."""
        return 0 if self._plan is None else len(self._plan["levels"])

    # -- execution ----------------------------------------------------

    def forward(self, *xs: torch.Tensor) -> Any:
        """Run the batched (or serial-fallback) forward pass."""
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
            return cast(torch.Tensor, self._graph_out)
        return self._forward_impl(*xs)

    def _gather_idx(self, slots: list[int], like: torch.Tensor):
        return self._cached(
            ("idx", tuple(slots)),
            like,
            lambda t: torch.tensor(
                slots, dtype=torch.long, device=t.device
            ),
            dtype=torch.long,
        )

    def _forward_impl(self, *xs: torch.Tensor) -> Any:
        if self._plan is None:
            return self.eval_mod(*xs)

        x, env = self._input_env(xs)
        memo: dict[int, torch.Tensor] = {}

        ev = self.ev_factory(env, x, memo)

        def evf(t: Any) -> torch.Tensor:
            """Resolve a leaf term, cheaply for ``Param``/``Var``.

            Full ``eval_term`` dispatch for anything richer (select
            chains, fused subtrees, …).
            """
            if isinstance(t, Param):
                v = self.eval_mod._param_map.get(t.name)
                if v is not None:
                    return v
            elif isinstance(t, Var):
                v = env.get(t.name)
                if v is not None:
                    return v
            return ev(t)

        # ---- Level 0: aff(A_t, b_t) leaves → homogeneous (d+1)² -------
        leaves = self._plan["leaves"]
        n = len(leaves)
        if self._plan["leaf_a_shared"]:
            # LTI recurrence: one shared transition matrix — expand is a
            # zero-copy stride-0 view that batched matmul reads directly,
            # avoiding both the per-leaf evals and the stack kernel.
            A0 = evf(leaves[0].args[0])
            a_vals = A0.unsqueeze(0).expand(n, *A0.shape)
        else:
            leaf_As = [evf(leaf.args[0]) for leaf in leaves]
            if all(a is leaf_As[0] for a in leaf_As):
                a_vals = (
                    leaf_As[0].unsqueeze(0).expand(n, *leaf_As[0].shape)
                )
            else:
                # Batched scans may mix broadcastable map shapes
                # (a shared (d,) decay alongside (B, d) per-step
                # gates) — expand to the join so the stack is uniform.
                tgt = torch.broadcast_shapes(
                    *[a.shape for a in leaf_As]
                )
                a_vals = torch.stack([a.expand(tgt) for a in leaf_As])

        gather = self._plan["leaf_b_gather"]
        if gather is not None:
            # All leaf b-parts are base[i] slices of one tensor — a
            # single index_select replaces n tiny indexing calls.
            base_t, dim, indices = gather
            base = evf(base_t)
            if self._b_identity and base.shape[dim] == len(indices):
                b_vals = base
            else:
                b_vals = base.index_select(
                    dim, self._gather_idx(indices, base)
                )
            if dim != 0:
                b_vals = b_vals.movedim(dim, 0)
        else:
            leaf_bs = [evf(leaf.args[1]) for leaf in leaves]
            tgt = torch.broadcast_shapes(*[b.shape for b in leaf_bs])
            b_vals = torch.stack([b.expand(tgt) for b in leaf_bs])

        if self._plan["diagonal"]:
            # Broadcast every leaf part up to the shared state shape —
            # (d,) maps ride along with (B, d) states (the pointwise
            # algebra is identical under the carrier bindings).
            tgt = torch.broadcast_shapes(
                a_vals.shape[1:], b_vals.shape[1:]
            )
            a_vals = _slot_expand(a_vals, tgt)
            b_vals = _slot_expand(b_vals, tgt)

        if self._fused is not None:
            return self._apply_post(
                self._forward_fused(a_vals, b_vals, evf)
            )

        if self._plan["diagonal"]:
            # Diagonal carrier: pairs (a, b) of state-shaped tensors;
            # compose is broadcasted elementwise —
            # (a_f⊙a_g, a_f⊙b_g + b_f).
            a_all, b_all = a_vals, b_vals
            for f_idx, g_idx in self._plan["level_gather"]:
                a_f = a_all.index_select(
                    0, self._gather_idx(f_idx, a_all)
                )
                a_g = a_all.index_select(
                    0, self._gather_idx(g_idx, a_all)
                )
                b_f = b_all.index_select(
                    0, self._gather_idx(f_idx, b_all)
                )
                b_g = b_all.index_select(
                    0, self._gather_idx(g_idx, b_all)
                )
                a_all = torch.cat([a_all, a_f * a_g])
                b_all = torch.cat([b_all, a_f * b_g + b_f])
            h = evf(self._plan["h"])
            r = self._plan["root_slot"]
            return self._apply_post(a_all[r] * h + b_all[r])

        m_all = self._leaf_homogeneous(a_vals, b_vals)

        # ---- Levels 1..L: batched aff_compose = M_f @ M_g -------------
        for f_idx, g_idx in self._plan["level_gather"]:
            m_f = m_all.index_select(0, self._gather_idx(f_idx, m_all))
            m_g = m_all.index_select(0, self._gather_idx(g_idx, m_all))
            m_new = m_f @ m_g
            m_all = torch.cat([m_all, m_new])

        # ---- Root apply: f(h) = A·h + b --------------------------------
        m_root = m_all[self._plan["root_slot"]]
        h = evf(self._plan["h"])
        d = m_root.shape[-1] - 1
        a_r = m_root[..., :d, :d]
        if h.dim() >= 2 and h.shape[-1] == 1:
            # column-vector state (…, d, 1): the homogeneous b column
            # stays a column — (…, d, d) @ (…, d, 1) + (…, d, 1).
            out = a_r @ h + m_root[..., :d, d:]
        else:
            # vector state (d,) (or a broadcastable init — apply's own
            # binding semantics), kept in the classic matvec spelling.
            out = a_r @ h + m_root[..., :d, d]
        return self._apply_post(out)

    def _apply_post(self, out: torch.Tensor) -> torch.Tensor:
        """Re-apply the output views stripped from the root term.

        ``plan["post"]`` holds the ``(op, attrs)`` wrappers outermost
        first — apply them innermost-first to reproduce the original
        root's value.  The ambient torch bindings give each op its
        exact semantics (``reshape(t, shape=…)`` etc.).
        """
        for op_name, attrs in reversed(
            cast("dict", self._plan)["post"]
        ):
            out = self.eval_mod._torch_bindings[op_name](out, **attrs)
        return out

    # -- fused schedule -------------------------------------------------

    def _leaf_homogeneous(
        self, a_vals: torch.Tensor, b_vals: torch.Tensor
    ):
        """Pack leaf ``(A, b)`` pairs into ``(n, *P, d+1, d+1)`` mats.

        ``M_leaf = [[A, b], [0, …, 0, 1]]`` — the homogeneous-matrix
        trick that makes ``aff_compose`` literal matmul.  ``b`` may be
        a vector ``(*P, d)`` (unsqueezed to a column here) or already
        a column ``(*P, d, 1)``; map and column batch dims broadcast
        to a common ``P`` so shared ``(d, d)`` maps ride along with
        batched ``(B, d, 1)`` states.
        """
        n = a_vals.shape[0]
        if b_vals.dim() == a_vals.dim() - 1:
            b_vals = b_vals.unsqueeze(-1)  # (*P, d) → (*P, d, 1)
        pb = torch.broadcast_shapes(
            a_vals.shape[1:-2], b_vals.shape[1:-2]
        )
        d = a_vals.shape[-1]
        a_vals = _slot_expand(a_vals, (*pb, d, d))
        b_vals = _slot_expand(b_vals, (*pb, d, 1))
        top = torch.cat([a_vals, b_vals], dim=-1)
        bottom = self._cached(
            ("row", len(pb), d + 1), a_vals, _make_bottom_row
        )
        return torch.cat([top, bottom.expand(n, *pb, 1, d + 1)], dim=-2)

    def _occ_index(self, like: torch.Tensor):
        """Return the occurrence-order gather indices, or ``None``.

        Tree-structured terms keep ``_fused_occ is None`` — leaves are
        already in product order; only DAG-shared terms pay the gather.
        """
        occ = self._fused_occ
        if occ is None:
            return None
        return self._cached(
            ("occ", tuple(occ)),
            like,
            lambda t: torch.tensor(
                occ, dtype=torch.long, device=t.device
            ),
            dtype=torch.long,
        )

    def _fused_call(self, body, *args: torch.Tensor) -> torch.Tensor:
        """Run ``body`` — compiled when the mode calls for it.

        ``fused="eager"`` (or a failed compile) runs the plain
        function; ``fused=True``/``"compile"`` wraps it in
        ``torch.compile(fullgraph=True)`` on first use and permanently
        falls back to the eager body if compilation or a compiled call
        ever raises — the fused schedule stays correct either way.
        """
        if self._fused == "eager" or self._fused_compile_failed:
            return body(*args)
        if self._fused_c is None:
            try:
                self._fused_c = torch.compile(body, fullgraph=True)
            except Exception:
                self._fused_compile_failed = True
                return body(*args)
        try:
            return self._fused_c(*args)
        except Exception:
            self._fused_compile_failed = True
            self._fused_c = None
            return body(*args)

    def _forward_fused(self, a_vals, b_vals, ev) -> torch.Tensor:
        """Canonical adjacent-pair reduction over the leaf product.

        Re-brackets to a balanced power-of-two tree: every level is a
        strided ``f = x[0::2], g = x[1::2]`` pair on a fresh tensor —
        the same composed map (the monoid product is
        association-invariant) in ~2 kernels per level eagerly, or a
        handful of Inductor kernels total when compiled.
        """
        plan = cast("dict", self._plan)
        h = ev(plan["h"])
        occ = self._occ_index(a_vals)
        if plan["diagonal"]:
            if occ is not None:
                a_vals = a_vals.index_select(0, occ)
                b_vals = b_vals.index_select(0, occ)
            return self._fused_call(
                fused_diag_levels, a_vals, b_vals, h
            )
        m_seq = self._leaf_homogeneous(a_vals, b_vals)
        if occ is not None:
            m_seq = m_seq.index_select(0, occ)
        return self._fused_call(fused_dense_levels, m_seq, h)


def _slot_expand(
    t: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """``(n, *s) -> (n, *shape)``, broadcasting ``s`` right-aligned.

    The leaf stack's dim 0 is the slot axis — the per-slot payload
    right-aligns against the target like ordinary broadcasting, so a
    ``(d,)`` map becomes ``(1, …, 1, d)`` before expanding to
    ``(B, d)``-style state shapes.
    """
    s = t.shape[1:]
    if tuple(s) == tuple(shape):
        return t
    pad = len(shape) - len(s)
    if pad:
        t = t.reshape(t.shape[0], *((1,) * pad), *s)
    return t.expand(t.shape[0], *shape)


def _make_bottom_row(like: torch.Tensor) -> torch.Tensor:
    """Return the ``[0, …, 0, 1]`` row of a homogeneous affine matrix."""
    d1 = like.shape[-1] + 1 if like.dim() >= 2 else like.shape[-1]
    row = like.new_zeros(1, 1, d1)
    row[..., -1] = 1.0
    return row


def to_batched_scan_module(
    ir: IR,
    param_values: dict[str, torch.Tensor] | None = None,
    *,
    fused: bool | str = False,
) -> BatchedScanModule:
    """Lower ``ir`` to a module, level-batching any leading scan term.

    Always returns a :class:`BatchedScanModule`; when ``ir.root`` is not
    ``apply(aff-tree, h)`` the module transparently delegates to the
    serial IRModule evaluator (check ``mod.is_batched`` to tell which
    path was taken).

    ``fused`` selects the level schedule on scan roots: ``False`` (the
    default) uses the slot-gather schedule the plan describes;
    ``True``/``"compile"`` runs the canonical fused reduction through
    ``torch.compile`` (permanently falling back to the eager fused
    body if compilation fails); ``"eager"`` runs the fused schedule
    without compile.  ``is_batched``/``n_levels``/``_plan`` describe
    the extracted term identically under either schedule.
    """
    return BatchedScanModule(ir, param_values=param_values, fused=fused)
