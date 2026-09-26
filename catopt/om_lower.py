# ruff: noqa: RUF002, RUF003
"""Level-batched lowering for online-softmax monoid (om) IR terms.

After eqsat with ``OM_LAWS`` (see :mod:`catopt.om`), chunked attention
``softmax(q @ cat(k_i).T) @ cat(v_i)`` extracts as::

    om_apply(<om_compose tree over om_elem(s_i, v_i)>)

— a FlashAttention schedule in tree form.  The generic
:class:`~catopt.torch_bridge.IRModule` evaluates it serially by passing
``(m, l, a)`` triples through the tree one node at a time: each
``om_elem`` is ~6 kernels (amax, sub, exp, sum, matmul, packaging) and
each ``om_compose`` ~10 elementwise/where kernels.  The parallelism is
in the graph but the schedule is serial.

This module lowers the same term *level-batched*, mirroring
:mod:`catopt.scan_lower` for the affine monoid:

* **Level 0** — every ``om_elem`` leaf sharing one ``(s, v)`` shape
  signature runs as a single batched element: score blocks stack into
  ``(n, ..., T, K)`` so ``amax``/``exp``/``sum`` and the ``e @ v``
  product are batched ops on one tensor instead of n tiny calls.
* **Levels 1..L** — same-level ``om_compose`` nodes run as one batched
  FlashAttention combine (all elementwise + ``maximum`` — fully
  broadcastable) via two index_select gathers per component.
* **Operand gathers** — when leaf operands are ``chunk``/``split``/
  ``select`` slices of a common base, the whole stack collapses to a
  reshape+permute (zero compute); when every score is
  ``q @ k_i.T`` with the *same* q and the ``k_i`` are chunks of one
  ``K``, all scores come from ONE dense matmul ``q @ K.T`` re-chunked
  by a view — the single most valuable case, since it turns the whole
  elem level into one GEMM plus batched reductions.

Usage::

    mod = to_batched_om_module(opt_ir, param_values=source)
    out = mod(x)
    mod.capture_cuda_graph(x)      # optional, CUDA only
    out = mod(x)                   # graph replay

``to_batched_om_module`` detects ``om_apply(<om tree>)`` roots; any
other IR falls back to the ordinary tuple-passing IRModule evaluation
(check ``mod.is_batched``).
"""

from __future__ import annotations

from typing import Any

import torch

from catopt.ir import IR, Op, Param
from catopt.torch_bridge import IRModule, _om_compose, _om_elem
from catopt.typing import _shape_of

__all__ = [
    "BatchedOMModule",
    "StreamingOMModule",
    "build_om_plan",
    "is_om_apply_term",
    "om_apply_state",
    "om_empty_state",
    "om_step",
    "om_step_qk",
    "to_batched_om_module",
    "to_streaming_om_module",
]


# ---------------------------------------------------------------------------
#  Shape detection
# ---------------------------------------------------------------------------


def _is_om_tree(term: Any, memo: dict | None = None) -> bool:
    """True if ``term`` is a pure om carrier tree.

    Leaves are ``om_elem(s, v)`` (score block, value block) or the raw
    packaging node ``om(m, l, a)``; internal nodes are binary
    ``om_compose(f, g)``.  Memoised on the term itself (hash-consed
    terms are content-keyed) — extracted terms are DAGs.
    """
    memo = {} if memo is None else memo
    key = term
    if key in memo:
        return memo[key]
    ok = isinstance(term, Op) and (
        (term.op == "om_elem" and len(term.args) == 2)
        or (term.op == "om" and len(term.args) == 3)
        or (
            term.op == "om_compose"
            and len(term.args) == 2
            and all(_is_om_tree(a, memo) for a in term.args)
        )
    )
    memo[key] = ok
    return ok


def is_om_apply_term(root: Any) -> bool:
    """True if ``root`` is ``om_apply(<om_elem/om/om_compose tree>)``."""
    return (
        isinstance(root, Op)
        and root.op == "om_apply"
        and len(root.args) == 1
        and _is_om_tree(root.args[0])
    )


def _concrete(shape: Any) -> bool:
    return (
        isinstance(shape, tuple)
        and len(shape) > 0
        and all(isinstance(d, int) for d in shape)
    )


def _same_term(ts: list[Any]) -> bool:
    """All entries are the same term object (or same-named Params)."""
    t0 = ts[0]
    return all(
        t is t0
        or (
            isinstance(t, Param)
            and isinstance(t0, Param)
            and t.name == t0.name
        )
        for t in ts[1:]
    )


# ---------------------------------------------------------------------------
#  Operand gathers — recognise slices of a common base tensor
# ---------------------------------------------------------------------------


def _slice_index(term: Any):
    """Decompose ``select``/``getitem``/``chunk``/``split`` slice ops.

    Returns ``(base, kind, dim, index, nparts, part_size)`` or ``None``:
    ``kind`` is ``"select"`` (indexing removes dim) or ``"chunk"``
    (equal split keeps dim); ``nparts``/``part_size`` describe the
    partition for chunk/split.  ``part_size`` is ``None`` when a split
    cannot be proven equal-size (uneven ``torch.chunk`` tails, unequal
    ``split`` sizes) — such slices can't be restacked by a reshape.
    """
    if not isinstance(term, Op) or len(term.args) != 1:
        return None
    a = term.attrs
    if term.op == "select":
        dim = a.get("arg1", a.get("dim"))
        idx = a.get("arg2", a.get("index"))
        if isinstance(dim, int) and isinstance(idx, int):
            return (term.args[0], "select", dim, idx, None, None)
        return None
    if term.op == "getitem":
        # t[i] — always indexes dim 0 (see torch_bridge binding).
        idx = a.get("arg1", a.get("index"))
        if isinstance(idx, int):
            return (term.args[0], "select", 0, idx, None, None)
        return None
    if term.op in ("chunk", "split"):
        dim = a.get("arg2", a.get("dim"))
        idx = a.get("index", a.get("arg3", 0))
        if not isinstance(dim, int) or not isinstance(idx, int):
            return None
        if term.op == "chunk":
            n = a.get("arg1", a.get("chunks"))
            if not isinstance(n, int):
                return None
            base_shape = _shape_of(term.args[0])
            if _concrete(base_shape):
                D = base_shape[dim % len(base_shape)]
                if D % n == 0:
                    return (term.args[0], "chunk", dim, idx, n, D // n)
            return (term.args[0], "chunk", dim, idx, n, None)
        sizes = a.get("sizes")
        if isinstance(sizes, (list, tuple)) and sizes:
            part = (
                sizes[0] if all(s == sizes[0] for s in sizes) else None
            )
            return (term.args[0], "chunk", dim, idx, len(sizes), part)
        return None
    return None


def _sliced_gather(parts: list) -> tuple | None:
    """Common-base check for a list of ``_slice_index`` results.

    Returns ``(base, kind, dim, nparts, part_size)`` when every leaf is
    the i-th equal slice of the SAME base along the SAME dim, in order —
    so stacking the leaves is one view/permute of the base.  ``None``
    otherwise (caller falls back to per-leaf eval + stack).
    """
    if not parts or any(p is None for p in parts):
        return None
    base0, kind0, dim0, _, n0, part0 = parts[0]
    bshape = _shape_of(base0)
    if not _concrete(bshape):
        return None
    rank = len(bshape)
    dim0n = dim0 % rank
    for i, p in enumerate(parts):
        base, kind, dim, idx, n, part = p
        if base is not base0 or kind != kind0 or dim % rank != dim0n:
            return None
        if idx != i:
            return None
        if kind == "chunk" and (
            n != n0 or part is None or part != part0
        ):
            return None
    if kind0 == "select":
        n = bshape[dim0n]
        if n != len(parts):
            return None
        return (base0, "select", dim0, n, 1)
    if n0 != len(parts):
        return None
    return (base0, "chunk", dim0, n0, part0)


def _qk_parts(term: Any):
    """Recognise ``matmul(q, transpose(k, -2, -1))`` → ``(q, k)``.

    The transpose must be exactly ``.T`` on the last two dims (the same
    check MATMUL_T_CONCAT performs — a partial transpose would scatter
    the key axis elsewhere).
    """
    if not (
        isinstance(term, Op)
        and term.op == "matmul"
        and len(term.args) == 2
    ):
        return None
    tr = term.args[1]
    if not (
        isinstance(tr, Op)
        and tr.op == "transpose"
        and len(tr.args) == 1
    ):
        return None
    d0 = tr.attrs.get("arg1", tr.attrs.get("dim0"))
    d1 = tr.attrs.get("arg2", tr.attrs.get("dim1"))
    if not (isinstance(d0, int) and isinstance(d1, int)):
        return None
    ks = _shape_of(tr.args[0])
    if _concrete(ks):
        r = len(ks)
        if r < 2 or {d0 % r, d1 % r} != {r - 2, r - 1}:
            return None
    elif {d0, d1} not in ({-2, -1}, {-1, -2}):
        return None
    return (term.args[0], tr.args[0])


# ---------------------------------------------------------------------------
#  The plan
# ---------------------------------------------------------------------------


def _analyze_elem_group(members: list[Op]) -> dict:
    """Decide how a uniform-shape group of ``om_elem`` leaves evaluates.

    ``s_mode`` for the score operands:

    * ``"dense_qk"`` — every ``s_i`` is ``q @ k_i.T`` with the same q
      and the ``k_i`` are consecutive equal chunks of one K along the
      key axis: compute ``q @ K.T`` once, view as blocks.  One GEMM for
      the whole level.
    * ``"bmm_qk"`` — every ``s_i`` is ``q @ k_i.T`` with the same q and
      uniform k shapes: stack/gather the ``k_i`` and run ONE batched
      matmul ``q @ K_stack.T``.
    * ``"slice"`` — the ``s_i`` are consecutive slices of one score
      tensor: reshape+permute, no compute.
    * ``"stack"`` — evaluate each ``s_i`` then stack (one cat kernel).

    ``v_gather`` likewise collapses value blocks that slice one base.
    """
    grp: dict[str, Any] = {"kind": "elem", "members": members}
    s_terms = [leaf.args[0] for leaf in members]
    v_terms = [leaf.args[1] for leaf in members]

    grp["v_gather"] = _sliced_gather([_slice_index(t) for t in v_terms])

    qks = [_qk_parts(t) for t in s_terms]
    if all(p is not None for p in qks) and _same_term(
        [p[0] for p in qks]
    ):
        grp["q"] = qks[0][0]
        k_terms = [p[1] for p in qks]
        kg = _sliced_gather([_slice_index(t) for t in k_terms])
        if kg is not None and kg[1] == "chunk":
            # slices along the KEY axis (dim -2 of k) ⇒ concat of the
            # k_i rebuilds K on -2 ⇒ s_i are column chunks of q @ K.T.
            bshape = _shape_of(kg[0])
            if (
                _concrete(bshape)
                and kg[2] % len(bshape) == len(bshape) - 2
            ):
                grp["s_mode"] = "dense_qk"
                grp["k_base"] = kg[0]
                grp["k_gather"] = kg
                return grp
        k_shapes = [_shape_of(t) for t in k_terms]
        if _concrete(k_shapes[0]) and all(
            s == k_shapes[0] for s in k_shapes
        ):
            grp["s_mode"] = "bmm_qk"
            grp["k_terms"] = k_terms
            grp["k_gather"] = kg
        else:
            grp["s_mode"] = "stack"
        return grp

    sg = _sliced_gather([_slice_index(t) for t in s_terms])
    if sg is not None:
        grp["s_mode"] = "slice"
        grp["s_gather"] = sg
    else:
        grp["s_mode"] = "stack"
    return grp


def build_om_plan(root: Any) -> dict | None:
    """Analyse ``om_apply(<om tree>)`` into a level-batched schedule.

    Returns ``None`` when the term is not an om application.  Otherwise
    a dict:

    * ``"leaves"`` — the ``om_elem``/``om`` nodes, first-encounter order.
    * ``"leaf_groups"`` — leaves partitioned for level-0 execution:
      same-signature ``om_elem`` leaves form one batched group each;
      everything else is a ``"serial"`` singleton group evaluated by
      the generic evaluator.  Slot order follows group order.
    * ``"levels"`` / ``"level_gather"`` — as in
      :func:`catopt.scan_lower.build_scan_plan`: ``level_gather[k]`` is
      ``(f_slots, g_slots)`` indexing the running stacked triples.
    * ``"root_slot"`` — slot of the complete composed carrier.
    * ``"f"`` — the carrier term (root of the om tree).
    """
    if not is_om_apply_term(root):
        return None
    f_term = root.args[0]

    leaves: list[Op] = []
    levels: list[list[Op]] = []
    level_of: dict[Any, int] = {}

    def visit(t: Op) -> int:
        tid = t
        if tid in level_of:
            return level_of[tid]
        if t.op in ("om_elem", "om"):
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

    # Leaf multiplicities: the term is a DAG, so a shared leaf/subtree
    # genuinely contributes its carrier once per occurrence (composing
    # a block with itself doubles it — like duplicated keys).  Count
    # paths from the root, memoised per node so this is linear in the
    # DAG, not the expanded tree.
    mult_memo: dict[Any, dict[Any, int]] = {}

    def leaf_counts(t: Op) -> dict[Any, int]:
        tid = t
        if tid in mult_memo:
            return mult_memo[tid]
        if t.op in ("om_elem", "om"):
            c = {tid: 1}
        else:
            c = dict(leaf_counts(t.args[0]))
            for k, v in leaf_counts(t.args[1]).items():
                c[k] = c.get(k, 0) + v
        mult_memo[tid] = c
        return c

    mults_of = leaf_counts(f_term)

    # Partition leaves into batched elem groups (uniform s/v shapes,
    # stackable) vs serial singletons (om packaging leaves, or elem
    # leaves whose shapes cannot be proven uniform).
    groups: list[dict] = []
    by_key: dict[Any, dict] = {}
    for leaf in leaves:
        if leaf.op == "om_elem":
            ss = _shape_of(leaf.args[0])
            vs = _shape_of(leaf.args[1])
            key = (
                ("elem", ss, vs)
                if (_concrete(ss) and _concrete(vs))
                else ("serial", leaf)
            )
        else:
            key = ("serial", leaf)
        grp = by_key.get(key)
        if grp is None:
            grp = {"key": key, "members": [], "mults": []}
            by_key[key] = grp
            groups.append(grp)
        grp["members"].append(leaf)
        grp["mults"].append(mults_of.get(leaf, 1))

    # Slot assignment: leaves occupy slots in group order, each taking
    # `mult` consecutive slots, then each level's outputs append —
    # operand slots are all < the level's own.  (Execution runs the
    # canonical adjacent-pair reduction, so the level gathers are
    # descriptive metadata about the extracted bracketing.)
    slot: dict[Any, int] = {}
    next_slot = 0
    for grp in groups:
        for leaf, c in zip(grp["members"], grp["mults"], strict=True):
            slot[leaf] = next_slot
            next_slot += c
        if grp["key"][0] == "elem":
            grp.update(_analyze_elem_group(grp["members"]))
        else:
            grp["kind"] = "serial"

    level_gather: list[tuple[list[int], list[int]]] = []
    for nodes in levels:
        f_idx = [slot[t.args[0]] for t in nodes]
        g_idx = [slot[t.args[1]] for t in nodes]
        level_gather.append((f_idx, g_idx))
        for t in nodes:
            slot[t] = next_slot
            next_slot += 1

    return {
        "leaves": leaves,
        "leaf_groups": groups,
        "levels": levels,
        "level_gather": level_gather,
        "root_slot": slot[f_term],
        "f": f_term,
    }


# ---------------------------------------------------------------------------
#  The batched kernels
# ---------------------------------------------------------------------------


def _stretch(t: torch.Tensor, tail: torch.Size) -> torch.Tensor:
    """``t``: ``(n, *t_tail)`` → ``(n, *tail)`` broadcasting batch dims.

    Compose results broadcast like their operands; the stacked slot
    tensor needs one common trailing shape, so size-1/missing batch
    dims are expanded (a view — no copy).
    """
    t_tail = tuple(t.shape[1:])
    tail = tuple(tail)
    if t_tail == tail:
        return t
    gap = len(tail) - len(t_tail)
    if gap > 0:
        t = t.reshape(t.shape[0], *((1,) * gap), *t_tail)
    return t.expand(t.shape[0], *tail)


def _batched_elem(s: torch.Tensor, v: torch.Tensor):
    """``(n,...,T,K)`` scores + ``(n,...,K,d)`` values → stacked triple.

    Identical math to the serial ``om_elem`` binding, one batched call
    per op: amax, sub+exp, sum, and the ``e @ v`` batched matmul.
    Fully-masked rows keep their NaNs (m = -inf ⇒ s-m = NaN) exactly
    like the serial path.
    """
    m = s.amax(dim=-1, keepdim=True)
    e = torch.exp(s - m)
    return m, e.sum(dim=-1, keepdim=True), e @ v


def _batched_compose(m1, l1, a1, m2, l2, a2):
    """The FlashAttention combine, batched over a whole reduction level.

    The operands' ``l``/``a`` parts are pre-sanitised (a non-finite
    running max implies zeroed l/a — see ``_forward_impl``), so the
    serial binding's ``where(fin, l*e, 0)`` guards collapse to plain
    ``l*e`` products: same result, ~half the memory traffic on the
    dominant ``a`` component.  The ``e`` factors keep their isfinite
    guard — ``exp(-inf − -inf)`` would be NaN.
    """
    mx = torch.maximum(m1, m2)
    e1 = torch.where(torch.isfinite(m1), torch.exp(m1 - mx), 0.0)
    e2 = torch.where(torch.isfinite(m2), torch.exp(m2 - mx), 0.0)
    return mx, l1 * e1 + l2 * e2, a1 * e1 + a2 * e2


# ---------------------------------------------------------------------------
#  The module
# ---------------------------------------------------------------------------


class BatchedOMModule(torch.nn.Module):
    """nn.Module that runs an ``om_apply(<om tree>)`` term level-batched.

    Parameters
    ----------
    ir : IR
        The (post-extraction) IR to lower.
    param_values : dict[str, torch.Tensor] | None
        Mapping from IR param names to tensor values, as for
        :func:`~catopt.torch_bridge.ir_to_torch_module`.

    The module embeds a plain :class:`IRModule` (as submodule
    ``eval_mod``) used both for leaf-operand evaluation and as the
    complete fallback when the root is not an om application — a drop-
    in replacement for ``ir_to_torch_module`` on ANY IR.

    For launch-bound workloads (many small blocks) the eager path is
    still CPU-dispatch limited; :meth:`capture_cuda_graph` records the
    whole batched forward so one ``graph.replay()`` replaces every
    launch.

    Ports layer: conforms to :class:`catopt.ports.BatchedExecutor`
    (``forward`` + ``_plan`` + ``is_batched``; ``eval_mod`` is the
    semantic member — see :class:`catopt.ports.PlannedExecutor` for why
    it is not the runtime-checked one).  ``n_levels`` / ``n_blocks`` /
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
        self._plan = build_om_plan(self.eval_mod._root)
        self._const_cache: dict[tuple, torch.Tensor] = {}
        self._graph = None
        self._graph_inputs: list[torch.Tensor] = []
        self._graph_out: torch.Tensor | None = None
        self._compiled = None

    @property
    def is_batched(self) -> bool:
        """True when the root matched the om_apply pattern."""
        return self._plan is not None

    @property
    def n_levels(self) -> int:
        """Number of batched compose levels (0 when not batched)."""
        return 0 if self._plan is None else len(self._plan["levels"])

    @property
    def n_blocks(self) -> int:
        """Number of om carrier leaves (0 when not batched)."""
        return 0 if self._plan is None else len(self._plan["leaves"])

    @property
    def is_graph_captured(self) -> bool:
        """True after :meth:`capture_cuda_graph` succeeded."""
        return self._graph is not None

    def capture_cuda_graph(
        self,
        *example_inputs: torch.Tensor,
        warmup: int = 3,
    ) -> BatchedOMModule:
        """Capture the batched forward into a CUDA graph.

        After capture, ``forward`` copies each input into static
        buffers and replays the graph — one launch total.  Returns
        ``self``; a no-op when the module is not batched or CUDA is
        unavailable.  Same caveats as
        :meth:`BatchedScanModule.capture_cuda_graph`: fixed
        shapes/dtypes/devices, parameters baked in by pointer, the
        returned tensor is the static output buffer.
        """
        if self._plan is None or not torch.cuda.is_available():
            return self
        if not all(t.is_cuda for t in example_inputs):
            raise ValueError("capture_cuda_graph requires CUDA inputs")
        static_ins = [t.detach().clone() for t in example_inputs]
        # Warm up on a side stream (also populates _const_cache so no
        # host-side allocation happens during capture).
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
        """Release the captured graph, returning to eager execution."""
        self._graph = None
        self._graph_inputs = []
        self._graph_out = None

    def compile(self, **kwargs) -> BatchedOMModule:
        """Compile the batched forward with ``torch.compile``.

        Inductor fuses the elem-level ``sub``/``exp``/``sum`` chain and
        each compose level's elementwise soup — worth ~2-3x on
        bandwidth-bound shapes (measured on RTX 2050).  Returns
        ``self``; a no-op-ish pass-through when the module isn't
        batched (the fallback path delegates to the plain IRModule).
        """
        if self._plan is None:
            return self
        self._compiled = torch.compile(self._forward_impl, **kwargs)
        return self

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
        if self._compiled is not None:
            return self._compiled(*xs)
        return self._forward_impl(*xs)

    def _cached(
        self,
        key: tuple,
        like: torch.Tensor,
        make,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Device/dtype-aware tensor cache (index tensors)."""
        want = dtype or like.dtype
        t = self._const_cache.get(key)
        if t is None or t.device != like.device or t.dtype != want:
            t = make(like)
            self._const_cache[key] = t
        return t

    def _repeat_idx(self, counts: list[int], like: torch.Tensor):
        """``[0]*c0 + [1]*c1 + ...`` as a cached index tensor — expands a
        group's stacked leaves to their DAG multiplicities."""
        rep = [i for i, c in enumerate(counts) for _ in range(c)]
        return self._cached(
            ("rep", tuple(counts)),
            like,
            lambda t: torch.tensor(
                rep, dtype=torch.long, device=t.device
            ),
            dtype=torch.long,
        )

    def _eval_sliced(self, info: tuple, ev) -> torch.Tensor:
        """Stack ``(n, *leaf_shape)`` from a common-base slice gather.

        ``select`` slices: ``base.movedim(d, 0)``.  ``chunk``/equal
        ``split`` slices: split dim d into ``(n, part)`` by a view,
        then move the block axis front — one permute, zero compute.
        """
        base, kind, dim, n, part = info
        t = ev(base)
        d = dim % t.dim()
        if kind == "select":
            return t.movedim(d, 0)
        t = t.reshape(*t.shape[:d], n, part, *t.shape[d + 1 :])
        return t.movedim(d, 0)

    def _stacked_scores(self, grp: dict, ev) -> torch.Tensor:
        """All of a group's score operands as ``(n, ..., T, K)``."""
        mode = grp["s_mode"]
        if mode == "dense_qk":
            # s_i = q @ chunk(K,-2,i).T — compute the dense score
            # matrix once, then re-chunk the key axis by a view.
            q = ev(grp["q"])
            kb = ev(grp["k_base"])
            s = q @ kb.transpose(-2, -1)  # (..., T, nK)
            n = len(grp["members"])
            k = s.shape[-1] // n
            return s.reshape(*s.shape[:-1], n, k).movedim(-2, 0)
        if mode == "bmm_qk":
            # s_i = q @ k_i.T — one batched matmul over stacked keys.
            # einsum rather than matmul: broadcast-matmul materialises
            # the broadcast operand (n copies of q — a full-size DtoD
            # memcpy), while einsum keeps the expansion as a stride-0
            # view into the batched GEMM (~3x faster measured).
            q = ev(grp["q"])
            kg = grp.get("k_gather")
            if kg is not None:
                k = self._eval_sliced(kg, ev)  # (n, ..., K, d)
            else:
                k = torch.stack([ev(t) for t in grp["k_terms"]])
            # Normalise batch ranks (slot axis excluded): left-pad the
            # rank-poor side with 1s, then stretch mismatched dims —
            # einsum's ellipsis needs equal broadcast shapes.
            gap = k.dim() - 1 - q.dim()  # len(kb) - len(qb)
            if gap > 0:
                q = q.reshape(*((1,) * gap), *q.shape)
            elif gap < 0:
                k = k.reshape(
                    k.shape[0], *((1,) * (-gap)), *k.shape[1:]
                )
            bt = torch.broadcast_shapes(
                tuple(q.shape[:-2]), tuple(k.shape[1:-2])
            )
            if tuple(q.shape[:-2]) != bt:
                q = q.expand(*bt, *q.shape[-2:])
            if tuple(k.shape[1:-2]) != bt:
                k = k.expand(k.shape[0], *bt, *k.shape[-2:])
            return torch.einsum("...td,n...kd->n...tk", q, k)
        if mode == "slice":
            return self._eval_sliced(grp["s_gather"], ev)
        return torch.stack(
            [ev(leaf.args[0]) for leaf in grp["members"]]
        )

    def _stacked_values(self, grp: dict, ev) -> torch.Tensor:
        """All of a group's value operands as ``(n, ..., K, d)``."""
        vg = grp.get("v_gather")
        if vg is not None:
            return self._eval_sliced(vg, ev)
        return torch.stack(
            [ev(leaf.args[1]) for leaf in grp["members"]]
        )

    def _forward_impl(self, *xs: torch.Tensor) -> torch.Tensor:
        if self._plan is None:
            return self.eval_mod(*xs)

        x = xs[0] if xs else None
        env: dict[str, torch.Tensor] = {"self": x}
        for i, inp in enumerate(self._inputs):
            env[inp.name] = xs[i] if i < len(xs) else x
        memo: dict[Any, torch.Tensor] = {}

        def ev(t: Any) -> torch.Tensor:
            return self.eval_mod._eval(t, env, x, memo)

        # ---- Level 0: om_elem leaves → stacked (m, l, a) -------------
        m_parts: list[torch.Tensor] = []
        l_parts: list[torch.Tensor] = []
        a_parts: list[torch.Tensor] = []
        for grp in self._plan["leaf_groups"]:
            if grp["kind"] == "elem":
                s = self._stacked_scores(grp, ev)  # (g,...,T,K)
                v = self._stacked_values(grp, ev)  # (g,...,K,d)
                m, l_, a = _batched_elem(s, v)
                mults = grp["mults"]
                if any(c != 1 for c in mults):
                    # DAG-shared leaves contribute once per occurrence.
                    idx = self._repeat_idx(mults, m)
                    m = m.index_select(0, idx)
                    l_ = l_.index_select(0, idx)
                    a = a.index_select(0, idx)
                m_parts.append(m)
                l_parts.append(l_)
                a_parts.append(a)
            else:
                for leaf, c in zip(grp["members"], grp["mults"], strict=True):
                    f = ev(leaf)  # (m, l, a) triple
                    for _ in range(c):
                        m_parts.append(f[0].unsqueeze(0))
                        l_parts.append(f[1].unsqueeze(0))
                        a_parts.append(f[2].unsqueeze(0))

        # Normalise trailing batch dims to one broadcast shape per
        # component (carriers broadcast like their operands), then cat
        # along the slot axis — one contiguous tensor per component.
        # A single part skips the cat entirely (cat of one tensor is
        # still a full copy).
        m_tail = torch.broadcast_shapes(*[t.shape[1:] for t in m_parts])
        l_tail = torch.broadcast_shapes(*[t.shape[1:] for t in l_parts])
        a_tail = torch.broadcast_shapes(*[t.shape[1:] for t in a_parts])

        def _stack_parts(parts, tail):
            if len(parts) == 1 and tuple(parts[0].shape[1:]) == tuple(
                tail
            ):
                return (
                    parts[0].contiguous()
                    if not parts[0].is_contiguous()
                    else parts[0]
                )
            return torch.cat([_stretch(t, tail) for t in parts])

        m_all = _stack_parts(m_parts, m_tail)
        l_all = _stack_parts(l_parts, l_tail)
        a_all = _stack_parts(a_parts, a_tail)

        # Sanitise once: a non-finite running max means the block was
        # fully masked, so its (NaN) l/a must contribute 0 — the serial
        # binding's per-compose where() guards, hoisted to the leaves.
        # Finite-m rows keep NaN payloads (e.g. NaN in v) exactly like
        # the serial path.
        fin = torch.isfinite(m_all)
        l_all = torch.where(fin, l_all, 0.0)
        a_all = torch.where(fin, a_all, 0.0)

        # ---- Levels 1..L: batched om_compose -------------------------
        # ⊕ is associative AND commutative, so the extracted bracketing
        # is one of many correct schedules.  We run the canonical
        # adjacent-pair reduction instead of replaying the extracted
        # tree's slot order: adjacent pairs of a contiguous slot tensor
        # are strided VIEWS (m[0::2], m[1::2]), so a level needs zero
        # index_select gathers and zero re-cats — only the elementwise
        # combine itself (~10 kernels) touches memory.  An odd slot
        # count carries the last triple down unchanged.
        while m_all.shape[0] > 1:
            n = m_all.shape[0]
            h = n // 2
            m_new, l_new, a_new = _batched_compose(
                m_all[0 : 2 * h : 2],
                l_all[0 : 2 * h : 2],
                a_all[0 : 2 * h : 2],
                m_all[1 : 2 * h : 2],
                l_all[1 : 2 * h : 2],
                a_all[1 : 2 * h : 2],
            )
            if n % 2:
                m_new = torch.cat([m_new, m_all[n - 1 :]])
                l_new = torch.cat([l_new, l_all[n - 1 :]])
                a_new = torch.cat([a_new, a_all[n - 1 :]])
            m_all, l_all, a_all = m_new, l_new, a_new

        # ---- Root apply: a / l (unclamped — NaN semantics kept) ------
        return a_all[0] / l_all[0]


def to_batched_om_module(
    ir: IR,
    param_values: dict[str, torch.Tensor] | None = None,
) -> BatchedOMModule:
    """Lower ``ir`` to a module, level-batching any leading om term.

    Always returns a :class:`BatchedOMModule`; when ``ir.root`` is not
    ``om_apply(<om tree>)`` the module transparently delegates to the
    serial IRModule evaluator (check ``mod.is_batched``).
    """
    return BatchedOMModule(ir, param_values=param_values)


# ---------------------------------------------------------------------------
#  Streaming schedule — bounded-working-set left fold over the om tree
# ---------------------------------------------------------------------------
#
# BatchedOMModule stacks EVERY leaf's score block into one
# (n, ..., T, K) tensor — one dense q@Kᵀ GEMM, but the whole q×T_kv
# score matrix is resident at once.  The streaming schedule below runs
# the SAME term as a left fold over leaves: each om_elem's score/value
# block is evaluated, composed into a running (m, l, a) carrier, and
# released before the next block is touched.  Working set is
# O(one block + carrier), independent of block count — the schedule
# FlashAttention assumes but no executor emitted (benchmarked in
# /tmp/bench_om_regime.py: flat ~89 MiB at 2M keys, where materialising
# K,V or the score matrix is impossible).


def om_empty_state(
    shape: tuple,
    dv: int,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ⊕-identity carrier ``(m, l, a) = (-inf, 0, 0)``.

    ``shape`` is the carrier's broadcast tail excluding the trailing
    component dim — e.g. ``(B, H, Tq)`` gives ``m, l: (B,H,Tq,1)`` and
    ``a: (B,H,Tq,dv)``.  Composing it with any carrier returns that
    carrier, so a decode loop can start from it instead of ``None``.
    """
    m = torch.full(
        (*shape, 1), float("-inf"), device=device, dtype=dtype
    )
    l_ = torch.zeros(*shape, 1, device=device, dtype=dtype)
    a = torch.zeros(*shape, dv, device=device, dtype=dtype)
    return (m, l_, a)


def om_step(
    state: tuple | None,
    s: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Incremental step: ``state ⊕ om_elem(s, v)`` in O(block) work.

    ``state`` is a running ``(m, l, a)`` carrier (or ``None``/the
    :func:`om_empty_state` identity for the first block); ``s`` is the
    new block's score tensor ``(..., Tq, K_blk)`` and ``v`` its value
    block ``(..., K_blk, dv)``.  Work and memory are proportional to
    the block, NOT to the total keys seen so far — the decode /
    growing-cache regime (~260x vs sdpa-recompute at 65k, measured).
    """
    e = _om_elem(s, v)
    return e if state is None else _om_compose(state, e)


def om_step_qk(
    state: tuple | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """om_step for the canonical ``s = q @ k.T`` score form."""
    return om_step(state, q @ k.transpose(-2, -1), v)


def om_apply_state(state: tuple) -> torch.Tensor:
    """The ``om_apply`` readout: ``a / l`` (unclamped — NaN kept)."""
    return state[2] / state[1]


class StreamingOMModule(torch.nn.Module):
    """nn.Module running ``om_apply(<om tree>)`` as a bounded fold.

    Same term as :class:`BatchedOMModule`, different schedule: instead
    of materialising every leaf's carrier and reducing level-by-level,
    the tree is folded left-to-right — each ``om_elem`` leaf's score
    and value operands are evaluated through the embedded IRModule
    (with a *fresh* eval memo per leaf, so block intermediates die with
    the iteration), composed into a running ``(m, l, a)`` state via the
    serial ``om_compose`` binding, then released.

    Peak memory is O(max block size + carrier), not O(total keys): the
    q×K_blk score matrix of ONE block is the largest transient, and the
    input tensors themselves are the only thing that scales with the
    sequence.  Per-leaf operand evaluation recomputes subterms shared
    between leaves (the price of dropping the shared memo) — the right
    tradeoff for the bounded-memory regime.

    For non-om roots the module delegates to serial IRModule eval —
    a drop-in for ``ir_to_torch_module`` on any IR (check
    ``mod.is_streaming``).

    The incremental decode step is exposed both as module-level
    functions (:func:`om_step`, :func:`om_step_qk`,
    :func:`om_empty_state`, :func:`om_apply_state` — pure tensor ops,
    CUDA-graph capturable) and as the staticmethods ``step`` /
    ``step_qk`` / ``apply`` here.

    Ports layer: a :class:`catopt.ports.PlannedExecutor` (``forward`` +
    ``_plan``, with ``eval_mod`` as the serial fallback) but
    deliberately not a :class:`~catopt.ports.BatchedExecutor` — its
    discriminator is ``is_streaming``, not ``is_batched``.
    """

    def __init__(
        self,
        ir: IR,
        param_values: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self._inputs = ir.inputs
        self.eval_mod = IRModule(ir, param_values)
        # Plan from the FOLDED root, exactly like BatchedOMModule.
        self._plan = build_om_plan(self.eval_mod._root)

    @property
    def is_streaming(self) -> bool:
        """True when the root matched the om_apply pattern."""
        return self._plan is not None

    @property
    def n_blocks(self) -> int:
        """Number of om carrier leaves (0 when not streaming)."""
        return 0 if self._plan is None else len(self._plan["leaves"])

    # -- incremental decode API (module-level helpers as statics) -----

    step = staticmethod(om_step)
    step_qk = staticmethod(om_step_qk)
    apply = staticmethod(om_apply_state)
    empty_state = staticmethod(om_empty_state)

    # -- execution ----------------------------------------------------

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        """Streaming fold → ``a / l`` (the om_apply readout)."""
        if self._plan is None:
            return self.eval_mod(*xs)
        return om_apply_state(self.forward_state(*xs))

    def forward_state(
        self,
        *xs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Streaming fold → the raw ``(m, l, a)`` carrier.

        This is the state a decode loop checkpoints: feed it to
        :func:`om_step`/`om_step_qk` as further blocks arrive, and read
        out with :func:`om_apply_state`.  For a non-om root the term
        isn't a carrier — this returns whatever serial eval produces
        (a triple for bare om trees), matching ``eval_mod``.
        """
        if self._plan is None:
            return self.eval_mod(*xs)

        x = xs[0] if xs else None
        env: dict[str, torch.Tensor] = {"self": x}
        for i, inp in enumerate(self._inputs):
            env[inp.name] = xs[i] if i < len(xs) else x

        state: tuple | None = None
        for grp in self._plan["leaf_groups"]:
            for leaf, c in zip(grp["members"], grp["mults"], strict=True):
                # Fresh memo per leaf: score blocks / per-leaf
                # intermediates are dropped with the dict at the next
                # iteration — the bounded-working-set property.  Inputs
                # (Var lookups) and params never enter the memo.
                memo: dict[Any, Any] = {}
                e = self.eval_mod._eval(leaf, env, x, memo)
                # A DAG-shared leaf contributes once per occurrence.
                for _ in range(c):
                    state = (
                        e if state is None else _om_compose(state, e)
                    )
        return state


def to_streaming_om_module(
    ir: IR,
    param_values: dict[str, torch.Tensor] | None = None,
) -> StreamingOMModule:
    """Lower ``ir`` to a module running the streaming om schedule.

    Always returns a :class:`StreamingOMModule`; when ``ir.root`` is
    not ``om_apply(<om tree>)`` it delegates to serial IRModule eval
    (check ``mod.is_streaming``).
    """
    return StreamingOMModule(ir, param_values=param_values)
