"""Level-batched lowering for parallel-scan (affine-monoid) IR terms.

After eqsat with ``SCAN_LAWS`` and a depth cost, an unrolled recurrence
``h_t = A·h_{t-1} + x_t`` extracts as::

    apply(<balanced aff_compose tree>, h0)

—a Blelloch-style scan of depth ~2·log2(T).  The generic
:class:`~catopt.torch_bridge.IRModule` evaluates ``aff``/``aff_compose``
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
"""

from __future__ import annotations

from typing import Any

import torch

from catopt.executors import BatchedExecutorBase
from catopt.ir import IR, Op, Param
from catopt.torch_bridge import IRModule
from catopt.typing import _shape_of

__all__ = [
    "BatchedScanModule",
    "build_scan_plan",
    "is_scan_apply_term",
    "to_batched_scan_module",
]


def _fold_nested_apply(term: Any) -> Any:
    """``apply(f, apply(g, h))`` → ``apply(aff_compose(f, g), h)``
    (and the ``applyd``/``affd_compose`` diagonal pair).

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


def _is_aff_tree(term: Any, memo: dict | None = None) -> bool:
    """True if ``term`` is a pure map tree in ONE carrier domain.

    Dense: leaves ``aff(A, b)``, internal ``aff_compose``.  Diagonal:
    leaves ``aff_diag(a, b)``, internal ``affd_compose``.  Mixed or
    foreign nodes disqualify the subtree.  Memoised on id() because
    extracted terms are DAGs with shared subtrees.
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
    """True if ``root`` is ``apply[d](<map tree>, h)`` — dense or
    diagonal affine scan application (nested ``apply`` segments are
    first folded into the compose spine)."""
    root = _fold_nested_apply(root)
    return (
        isinstance(root, Op)
        and root.op in ("apply", "applyd")
        and len(root.args) == 2
        and _is_aff_tree(root.args[0]) is not None
        and {"apply": "aff", "applyd": "aff_diag"}[root.op]
        == _is_aff_tree(root.args[0])
    )


def _leaf_shapes_consistent(leaves: list[Op]) -> bool:
    """Check every ``aff(A, b)`` leaf shares one (d, d) / (d,) signature.

    Batched stacking needs uniform shapes; if inference cannot prove
    them uniform we decline the plan and the caller falls back to the
    serial evaluator (which is correct for any shape).
    """
    if not leaves:
        return False
    a0 = _shape_of(leaves[0].args[0])
    b0 = _shape_of(leaves[0].args[1])
    if leaves[0].op == "aff_diag":
        # Diagonal carrier: a and b are both (d,) vectors.
        ok = (
            isinstance(a0, tuple)
            and isinstance(b0, tuple)
            and a0 == b0
            and all(isinstance(d, int) for d in a0)
        )
    else:
        ok = (
            isinstance(a0, tuple)
            and len(a0) >= 2
            and all(isinstance(d, int) for d in a0)
            and isinstance(b0, tuple)
            and b0 == a0[:-1]
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


def _select_index(term: Op) -> tuple[Any, int, int] | None:
    """Decompose ``select(base, dim, i)``/getitem-style leaf operands.

    Returns ``(base_term, dim, index)`` or ``None``.  Covers the
    ``select(x, arg1=dim, arg2=i)`` shape torch.export emits for
    ``x[i]`` indexing.
    """
    if (
        not isinstance(term, Op)
        or term.op not in ("select", "getitem")
        or len(term.args) != 1
    ):
        return None
    if term.op == "getitem":
        # t[i] — arg1/index IS the index; always dim 0 (see the
        # torch_bridge binding).  Do not conflate with select.
        idx = term.attrs.get("arg1", term.attrs.get("index"))
        if not isinstance(idx, int):
            return None
        return (term.args[0], 0, idx)
    dim = term.attrs.get("arg1", term.attrs.get("dim", 0))
    idx = term.attrs.get("arg2", term.attrs.get("index"))
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
    * ``"f"`` / ``"h"`` — the map term and the applied-to term.
    """
    root = _fold_nested_apply(root)
    if not is_scan_apply_term(root):
        return None
    f_term, h_term = root.args
    diag = root.op == "applyd"

    leaves: list[Op] = []
    levels: list[list[Op]] = []
    level_of: dict[int, int] = {}
    leaf_op = "aff_diag" if root.op == "applyd" else "aff"

    def visit(t: Op) -> int:
        tid = id(t)
        if tid in level_of:
            return level_of[tid]
        if t.op == leaf_op:
            level_of[tid] = 0
            leaves.append(t)
            return 0
        lv = max(visit(t.args[0]), visit(t.args[1])) + 1
        level_of[tid] = lv
        while len(levels) < lv:
            levels.append([])
        levels[lv - 1].append(t)
        return lv

    visit(f_term)
    if not _leaf_shapes_consistent(leaves):
        return None

    # Slot assignment: leaves occupy 0..n-1, then each level appends its
    # outputs in order — so a level's operand slots are all < its own.
    slot: dict[int, int] = {
        id(lf): i for i, lf in enumerate(leaves)
    }
    next_slot = len(leaves)
    level_gather: list[tuple[list[int], list[int]]] = []
    for nodes in levels:
        f_idx = [slot[id(t.args[0])] for t in nodes]
        g_idx = [slot[id(t.args[1])] for t in nodes]
        level_gather.append((f_idx, g_idx))
        for t in nodes:
            slot[id(t)] = next_slot
            next_slot += 1

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
    return {
        "leaves": leaves,
        "levels": levels,
        "level_gather": level_gather,
        "root_slot": slot[id(f_term)],
        "leaf_b_gather": _leaf_b_gather(leaves),
        "leaf_a_shared": leaf_a_shared,
        "diagonal": diag,
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
        :func:`~catopt.torch_bridge.ir_to_torch_module`.

    The module embeds a plain :class:`IRModule` (as submodule
    ``eval_mod``) used both for leaf/h evaluation and as the complete
    fallback when the root is not a scan application — so this class is
    a drop-in replacement for ``ir_to_torch_module`` on ANY IR.

    For launch-bound workloads (small d, large T) the eager path is
    still CPU-dispatch limited; :meth:`capture_cuda_graph` records the
    whole batched forward into a CUDA graph so one ``graph.replay()``
    replaces all per-level launches.

    Ports layer: conforms to :class:`catopt.ports.BatchedExecutor`
    (``forward`` + ``_plan`` + ``is_batched``; ``eval_mod`` is the
    semantic member — see :class:`catopt.ports.PlannedExecutor` for why
    it is not the runtime-checked one).  ``n_levels`` /
    ``is_graph_captured`` stay class-level API, not port members.
    """

    def __init__(
        self,
        ir: IR,
        param_values: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._inputs = ir.inputs
        self.eval_mod = IRModule(ir, param_values)
        # Plan from the FOLDED root: IRModule may have rewritten
        # weight-only subtrees into fused Params during construction.
        self._plan = build_scan_plan(self.eval_mod._root)
        self._const_cache: dict[tuple, torch.Tensor] = {}
        self._init_graph_state()

    @property
    def is_batched(self) -> bool:
        """True when the root matched the scan pattern."""
        return self._plan is not None

    @property
    def n_levels(self) -> int:
        """Number of batched compose levels (0 when not batched)."""
        return 0 if self._plan is None else len(self._plan["levels"])

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

    def _gather_idx(self, slots: list[int], like: torch.Tensor):
        return self._cached(
            ("idx", tuple(slots)),
            like,
            lambda t: torch.tensor(
                slots, dtype=torch.long, device=t.device
            ),
            dtype=torch.long,
        )

    def _forward_impl(self, *xs: torch.Tensor) -> torch.Tensor:
        if self._plan is None:
            return self.eval_mod(*xs)

        x, env = self._input_env(xs)
        memo: dict[int, torch.Tensor] = {}

        ev = self.ev_factory(env, x, memo)

        # ---- Level 0: aff(A_t, b_t) leaves → homogeneous (d+1)² -------
        leaves = self._plan["leaves"]
        n = len(leaves)
        if self._plan["leaf_a_shared"]:
            # LTI recurrence: one shared transition matrix — expand is a
            # zero-copy stride-0 view that batched matmul reads directly,
            # avoiding both the per-leaf evals and the stack kernel.
            A0 = ev(leaves[0].args[0])
            a_vals = A0.unsqueeze(0).expand(n, *A0.shape)
        else:
            leaf_As = [ev(leaf.args[0]) for leaf in leaves]
            if all(a is leaf_As[0] for a in leaf_As):
                a_vals = (
                    leaf_As[0].unsqueeze(0).expand(n, *leaf_As[0].shape)
                )
            else:
                a_vals = torch.stack(leaf_As)

        gather = self._plan["leaf_b_gather"]
        if gather is not None:
            # All leaf b-parts are base[i] slices of one tensor — a
            # single index_select replaces n tiny indexing calls.
            base_t, dim, indices = gather
            base = ev(base_t)
            if indices == list(range(base.shape[dim])):
                b_vals = base
            else:
                b_vals = base.index_select(
                    dim, self._gather_idx(indices, base)
                )
            if dim != 0:
                b_vals = b_vals.movedim(dim, 0)
        else:
            b_vals = torch.stack([ev(leaf.args[1]) for leaf in leaves])

        if self._plan["diagonal"]:
            # Diagonal carrier: pairs (a, b) of (d,) vectors; compose is
            # broadcasted elementwise — (a_f⊙a_g, a_f⊙b_g + b_f).
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
            h = ev(self._plan["h"])
            r = self._plan["root_slot"]
            return a_all[r] * h + b_all[r]

        # M_leaf = [[A, b], [0, …, 0, 1]]  (n, d+1, d+1)
        d = a_vals.shape[-1]
        top = torch.cat([a_vals, b_vals.reshape(n, d, 1)], dim=-1)
        bottom = self._cached(("row", d + 1), a_vals, _make_bottom_row)
        m_all = torch.cat([top, bottom.expand(n, 1, d + 1)], dim=-2)

        # ---- Levels 1..L: batched aff_compose = M_f @ M_g -------------
        for f_idx, g_idx in self._plan["level_gather"]:
            m_f = m_all.index_select(0, self._gather_idx(f_idx, m_all))
            m_g = m_all.index_select(0, self._gather_idx(g_idx, m_all))
            m_new = m_f @ m_g
            m_all = torch.cat([m_all, m_new])

        # ---- Root apply: f(h) = A·h + b --------------------------------
        m_root = m_all[self._plan["root_slot"]]
        h = ev(self._plan["h"])
        return m_root[:d, :d] @ h + m_root[:d, d]


def _make_bottom_row(like: torch.Tensor) -> torch.Tensor:
    """The constant ``[0, …, 0, 1]`` row of a homogeneous affine matrix."""
    d1 = like.shape[-1] + 1 if like.dim() >= 2 else like.shape[-1]
    row = like.new_zeros(1, 1, d1)
    row[..., -1] = 1.0
    return row


def to_batched_scan_module(
    ir: IR,
    param_values: dict[str, torch.Tensor] | None = None,
) -> BatchedScanModule:
    """Lower ``ir`` to a module, level-batching any leading scan term.

    Always returns a :class:`BatchedScanModule`; when ``ir.root`` is not
    ``apply(aff-tree, h)`` the module transparently delegates to the
    serial IRModule evaluator (check ``mod.is_batched`` to tell which
    path was taken).
    """
    return BatchedScanModule(ir, param_values=param_values)
