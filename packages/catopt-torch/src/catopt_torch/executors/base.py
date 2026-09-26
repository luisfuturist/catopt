"""Shared machinery for the planned batched executors.

The three batched lowering adapters —
:class:`catopt_carriers.scan_lower.BatchedScanModule`,
:class:`catopt_carriers.om_lower.BatchedOMModule`, and
:class:`catopt_carriers.omd_lower.BatchedOmdModule` — all embed a plain
``IRModule`` as submodule ``eval_mod``, analyse their root term into
``self._plan`` once in ``__init__``, and carry byte-identical
CUDA-graph capture state plus input-env/eval-closure plumbing.  This
mixin (extracted in plan 0002 phase A) single-sources that shared
machinery; each concrete class keeps its own ``__init__`` and its own
``_forward_impl`` plan section, which genuinely differ per carrier.

The module also hosts the plan-time **level-schedule skeleton**
(extracted in plan 0002 phase D) — :func:`level_schedule` (post-order
``visit()`` assigning tree levels) and :func:`slot_gathers`
(leaf-first slot numbering + per-level operand gathers) — shared by
``build_scan_plan``/``build_om_plan``/``build_omd_plan``.  The
per-domain seeding stays at the sites: om weights leaf slots by DAG
multiplicity and omd's map forest walks iteratively per domain, so
the helpers expose the invariant, not a policy.

Ports layer: the concrete classes conform to
:class:`catopt_core.ports.BatchedExecutor` (``forward`` + ``_plan`` +
``is_batched``); the members here stay class-level API, not port
members.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import torch

__all__ = ["BatchedExecutorBase", "level_schedule", "slot_gathers"]


def level_schedule(
    root: Any,
    children_of: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool],
    *,
    key: Callable[[Any], Any] = lambda t: t,
) -> tuple[list, list[list]]:
    """Post-order level assignment over a compose tree (or DAG).

    ``visit(t)`` returns ``t``'s level — 0 when ``is_leaf(t)``, else
    ``max(children's levels) + 1`` — appending leaves to ``leaves``
    in first-encounter order and internal nodes to ``levels[lv-1]``,
    so every node's operands sit at strictly earlier levels.  Shared
    subtrees are visited once via the ``key``-projected memo: the
    default is the term itself (post-hash-consing terms are interned
    — content keys); ``id`` keeps strict per-object identity.

    Returns ``(leaves, levels)``; pair with :func:`slot_gathers` for
    the leaf-first slot numbering + per-level operand gathers.
    ``children_of``/``is_leaf`` carry the per-domain vocabulary
    (``aff_compose`` / ``om_compose`` / ``omd_compose``).
    """
    leaves: list = []
    levels: list[list] = []
    level_of: dict[Any, int] = {}

    def visit(t: Any) -> int:
        k = key(t)
        lv = level_of.get(k)
        if lv is not None:
            return lv
        if is_leaf(t):
            level_of[k] = 0
            leaves.append(t)
            return 0
        lv = max(visit(a) for a in children_of(t)) + 1
        level_of[k] = lv
        while len(levels) < lv:
            levels.append([])
        levels[lv - 1].append(t)
        return lv

    visit(root)
    return leaves, levels


def slot_gathers(
    levels: list[list],
    children_of: Callable[[Any], Any],
    slot: dict,
    next_slot: int,
    *,
    key: Callable[[Any], Any] = lambda t: t,
) -> list[tuple[list[int], list[int]]]:
    """Leaf-first slot numbering + per-level operand gathers.

    ``slot`` arrives seeded with the leaf slots — contiguous
    ``0..n-1`` at most sites, or om_lower's multiplicity-weighted
    layout (DAG-shared leaves occupy one slot per occurrence — pass
    the real ``next_slot``).  Each level's outputs then take
    consecutive slots in order, so every operand slot is < its own
    level's outputs.  Returns ``gathers[k] == (f_slots, g_slots)``,
    the index_select lists each level consumes.  ``slot`` is mutated
    (internal nodes added) so the caller reads the root's slot
    afterwards.
    """
    gathers: list[tuple[list[int], list[int]]] = []
    for nodes in levels:
        gathers.append(
            (
                [slot[key(children_of(t)[0])] for t in nodes],
                [slot[key(children_of(t)[1])] for t in nodes],
            )
        )
        for t in nodes:
            slot[key(t)] = next_slot
            next_slot += 1
    return gathers


class BatchedExecutorBase:
    """Mixin holding the batched executors' shared machinery.

    Use as ``class BatchedXModule(BatchedExecutorBase, nn.Module)`` —
    the mixin defines no ``__init__``, so ``super().__init__()`` in
    the concrete class still reaches ``nn.Module.__init__`` through
    the MRO, and ``nn.Module``'s attribute hooks apply normally.

    The concrete ``__init__`` must set ``_inputs`` (``ir.inputs``),
    ``eval_mod`` (a plain ``IRModule``), ``_plan`` (``None`` when the
    root doesn't match the planned pattern), and ``_const_cache``,
    then call :meth:`_init_graph_state`.  ``_forward_impl`` itself is
    supplied by the concrete class.
    """

    # Field contract (annotations only — the concrete __init__
    # initialises the values; nothing here touches them).
    _inputs: list[Any]
    _plan: dict | None
    eval_mod: Any
    _const_cache: dict[tuple, torch.Tensor]
    _graph: torch.cuda.CUDAGraph | None
    _graph_inputs: list[torch.Tensor]
    _graph_out: torch.Tensor | None
    _forward_impl: Callable[..., torch.Tensor]

    def _init_graph_state(self) -> None:
        """(Re)set the CUDA-graph capture fields.

        Concrete ``__init__``s call this once ``eval_mod``/``_plan``
        exist; :meth:`drop_cuda_graph` reuses it.
        """
        self._graph = None
        self._graph_inputs = []
        self._graph_out = None

    @property
    def is_graph_captured(self) -> bool:
        """True after :meth:`capture_cuda_graph` succeeded."""
        return self._graph is not None

    def capture_cuda_graph(  # pragma: no cover — CUDA-only body
        self,
        *example_inputs: torch.Tensor,
        warmup: int = 3,
    ) -> Self:
        """Capture the batched forward into a CUDA graph.

        After capture, ``forward`` copies each input into static
        buffers and replays the graph — one launch total.  Returns
        ``self`` for chaining; a no-op when the module is not batched
        or CUDA is unavailable.

        Caveats (standard CUDA-graph rules): inputs must be CUDA
        tensors; calls with a different shape/dtype/device fall back
        to the eager path; parameters are baked in by pointer —
        in-place updates are replayed, replacing a parameter is not;
        the returned tensor is the static output buffer, overwritten
        by the next call (clone it to keep it).
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
        self._init_graph_state()

    # -- eval plumbing ------------------------------------------------

    def _input_env(
        self, xs: tuple[torch.Tensor, ...]
    ) -> tuple[Any, dict[str, Any]]:
        """The shared ``_forward_impl`` prologue: ``(x, env)``.

        ``x`` is the first positional arg (``None`` on zero args —
        the serial evaluator's "self" fallback); ``env`` maps
        ``"self"`` and every named IR input to the matching
        positional arg, defaulting missing ones to ``x``.
        """
        x = xs[0] if xs else None
        env: dict[str, Any] = {"self": x}
        for i, inp in enumerate(self._inputs):
            env[inp.name] = xs[i] if i < len(xs) else x
        return x, env

    def ev_factory(
        self,
        env: dict[str, Any],
        x: Any,
        memo: dict,
    ) -> Callable[[Any], torch.Tensor]:
        """The ``ev(t)`` leaf-evaluation closure ``_forward_impl``
        uses: ``t → eval_mod._eval(t, env, x, memo)``, memoising into
        the caller's ``memo``."""

        def ev(t: Any) -> torch.Tensor:
            return self.eval_mod._eval(t, env, x, memo)

        return ev

    def _cached(
        self,
        key: tuple,
        like: torch.Tensor,
        make,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Device/dtype-aware tensor cache (index tensors, pad rows).

        ``like`` supplies the device; ``dtype`` defaults to ``like``'s
        but is fixed ``torch.long`` for index tensors.
        """
        want = dtype or like.dtype
        t = self._const_cache.get(key)
        if t is None or t.device != like.device or t.dtype != want:
            t = make(like)
            self._const_cache[key] = t
        return t
