"""TorchComposer — the torch :class:`~catopt_core.ports.Composer` port.

Everything structural the per-block (:class:`Compositional`) strategy
needs from a torch module tree (plan 0007, moved verbatim from
``catopt_orchestrator.optimize``):

* block selection — the ``named_children`` walk picking the top-most
  matching submodules;
* input capture — forward hooks recording each block's first call
  (argument clones + object-identity dataflow evidence);
* grafting — dotted-name replacement into ModuleList/Sequential
  containers;
* ``clone_sharing`` — deepcopy of the module structure with the
  parameter/buffer tensors *shared* via a seeded deepcopy memo;
* boundary probes — the perturbed-input residual check and the
  object-identity ``chain``/``residual``(+``_wrapped``) classifier;
* ``cross_pairs`` — the pairwise joint-optimization pass (joint
  micro-models, joint optimize+verify, cost-gated grafting);
* ``param_report`` — the weight-file diff hook.

The classes the pass grafts — ``_JointPair`` / ``_FusedPair`` /
``_Zero`` — are torch modules, so all of this lives with the adapter.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping
from typing import Any

import torch
from catopt_core import laws as _core_laws
from catopt_core.cost import flops_cost
from catopt_core.ports import CostFn, Sink, Source

from catopt_torch.report import verify_module

__all__ = ["TorchComposer", "param_report"]

#: The joint pass's widened set when the caller used the pipeline
#: default — the historical ``ruleset="all"`` → ``"all+layout"``
#: upgrade: the full core surface plus the layout migration laws,
#: minus the pairing-subsumed folds.
_JOINT_DEFAULT = _core_laws.WITH_LAYOUT - _core_laws.WITH_LAYOUT.tagged(
    _core_laws.tags.SUBSUMED
)


def _joint_rules(rules: Any) -> Any:
    """Return the rule set the joint pair pass saturates with.

    ``None`` — or a set equal to the pipeline's composed default —
    widens to the layout-inclusive surface (the historical
    ``"all"`` → ``"all+layout"`` upgrade); a caller-chosen
    :class:`~catopt_core.laws.RuleSet` is used as-is.
    """
    from catopt_orchestrator.optimize import (
        _resolve_rules,
        default_rules,
    )

    resolved = _resolve_rules(rules)
    if {r.name for r in resolved} == {r.name for r in default_rules()}:
        return _JOINT_DEFAULT
    return resolved


def _joint_optimize(
    joint: torch.nn.Module,
    x: Any,
    *,
    source: Source,
    sink: Sink,
    verbose: bool = True,
    **kw: Any,
) -> tuple[Any, dict[str, Any]]:
    """One joint micro-model optimization — the cross-pair seam.

    Runs the monolithic pipeline on the joint block through the same
    ``source``/``sink`` ports the per-block passes used and returns
    ``(module, stats)``.  Module-level so tests can stub the joint
    run (``monkeypatch.setattr(composer, "_joint_optimize", ...)``).
    """
    from catopt_orchestrator.optimize import Optimizer

    lr = Optimizer(source=source, sink=sink).optimize(
        joint, x, verify=verbose, verbose=verbose, **kw
    )
    return lr.module, lr.stats


def _default_block_pred(
    parent: torch.nn.Module, name: str, module: torch.nn.Module
) -> bool:
    """Select direct children of ``nn.ModuleList`` / ``nn.Sequential``.

    The default block selector for the standard 'stacked blocks'
    structure.
    """
    return isinstance(
        parent, (torch.nn.ModuleList, torch.nn.Sequential)
    )


def _select_blocks(
    model: torch.nn.Module, block_pred: Callable | None
) -> list[tuple[str, torch.nn.Module]]:
    """Pick the top-most submodules to optimize independently.

    Walks the module tree; the top-most matching blocks are chosen.

    A child is selected when it is a leaf (no children of its own) or when
    ``block_pred(parent, child_name, child)`` is true.  Selected blocks are
    opaque: we never descend into them, so e.g. the ``nn.Linear`` leaves
    inside a matched ``ParallelBlock`` are not optimized separately.
    """
    pred = block_pred or _default_block_pred
    blocks: list[tuple[str, torch.nn.Module]] = []

    def visit(module: torch.nn.Module, prefix: str) -> None:
        for child_name, child in module.named_children():
            full = f"{prefix}.{child_name}" if prefix else child_name
            is_leaf = next(child.children(), None) is None
            if is_leaf or pred(module, child_name, child):
                blocks.append((full, child))
            else:
                visit(child, full)

    visit(model, "")
    return blocks


#: ``io`` key under which the capture pass stores the model's own
#: return value — ``<`` is not a legal module-attribute character, so
#: it can never collide with a real block name.
_MODEL_KEY = "<model>"


def _capture_block_inputs(
    model: torch.nn.Module,
    blocks: list[tuple[str, torch.nn.Module]],
    example_input: torch.Tensor | tuple,
) -> tuple[dict[str, tuple[tuple, dict]], dict[str, dict[str, Any]]]:
    """Record each selected block's first forward inputs via hooks.

    Runs the ORIGINAL model once.  Returns ``(captured, io)``:

    * ``captured`` maps ``{name: (args, kwargs)}`` — detached clones of
      the first call's arguments;
    * ``io`` carries the cross-block dataflow evidence the pairwise
      pass reads: per block ``{"calls", "in_objs", "out_obj", "out"}``
      — the call count, the live arg/output OBJECTS of the first call
      (kept referenced so ``is``-identity stays valid: a freed object's
      id could be reused by a later allocation), and a detached clone
      of the first output — plus ``io["<model>"]`` with the model's
      own return.

      The object identity answers "did B literally consume A's
      output?"; the clones answer "was it modified in between?".
    """
    captured: dict[str, tuple[tuple, dict]] = {}
    io: dict[str, dict[str, Any]] = {}
    handles = []

    def make_hook(name: str):
        def hook(mod, args, kwargs, out):
            entry = io.setdefault(
                name,
                {
                    "calls": 0,
                    "in_objs": (),
                    "out_obj": None,
                    "out": None,
                },
            )
            entry["calls"] += 1
            if name not in captured:
                captured[name] = (
                    tuple(
                        a.detach().clone()
                        if isinstance(a, torch.Tensor)
                        else a
                        for a in args
                    ),
                    {
                        k: (
                            v.detach().clone()
                            if isinstance(v, torch.Tensor)
                            else v
                        )
                        for k, v in kwargs.items()
                    },
                )
                entry["in_objs"] = args
                entry["out_obj"] = out
                entry["out"] = (
                    out.detach().clone()
                    if isinstance(out, torch.Tensor)
                    else out
                )

        return hook

    for name, mod in blocks:
        handles.append(
            mod.register_forward_hook(make_hook(name), with_kwargs=True)
        )
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    try:
        model.eval()
        with torch.no_grad():
            out = model(*args)
        io[_MODEL_KEY] = {
            "out_obj": out,
            "out": (
                out.detach().clone()
                if isinstance(out, torch.Tensor)
                else out
            ),
        }
    finally:
        for h in handles:
            h.remove()
    return captured, io


def _replace_submodule(
    model: torch.nn.Module, dotted: str, new_mod: torch.nn.Module
) -> None:
    """Set ``model.<dotted>`` to ``new_mod``.

    Handles ModuleList / Sequential integer children.
    """
    parent_name, _, child_name = dotted.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if child_name.isdigit() and isinstance(
        parent, (torch.nn.ModuleList, torch.nn.Sequential)
    ):
        parent[int(child_name)] = new_mod
    else:
        setattr(parent, child_name, new_mod)


def _shared_param_clone(model: torch.nn.Module) -> torch.nn.Module:
    """Deepcopy ``model``'s module structure without copying tensors.

    A plain ``copy.deepcopy`` of an ``nn.Module`` clones every parameter
    and buffer — at ~0.5B fp16 params that doubles device memory before
    a single optimized block is grafted, which is what OOMed
    compositional recompose on small GPUs.  Recomposing only ever
    *replaces* submodules (``_replace_submodule`` rebinds entries in the
    clone's ``_modules`` dicts); it never mutates a tensor in place, so
    the clone can share the original's tensor storage safely.

    The mechanism is deepcopy's own memo: ``copy.deepcopy`` checks
    ``memo[id(obj)]`` before dispatching to ``__deepcopy__``, so
    pre-seeding every reachable tensor's id makes the clone reuse those
    objects while the module ``__dict__``s, ``_modules`` /
    ``_parameters`` / ``_buffers`` dicts, and plain attributes still
    copy normally — the result is a real clone (``training`` flag,
    hooks, structure) whose weights alias the original's.  Grafting
    into it cannot touch the caller's model.

    Seeding covers registered ``parameters()``/``buffers()`` plus
    *unregistered* tensor attributes — ``mod.foo = tensor`` lands in
    ``__dict__`` (only Parameters go to ``_parameters`` and only
    registered buffers to ``_buffers``), which the default deepcopy
    walk traverses by id, so the memo shares them too.  Tensors nested
    inside non-tensor container/attribute objects (e.g.
    ``self.cache = {"k": t}``) are not memo-hit and still clone; if
    even that fails, the caller's in-place fallback applies.
    """
    memo: dict[int, Any] = {}
    for t in list(model.parameters()) + list(model.buffers()):
        memo[id(t)] = t
    for mod in model.modules():
        for v in vars(mod).values():
            if isinstance(v, torch.Tensor):
                memo[id(v)] = v
    return copy.deepcopy(model, memo)


# ---------------------------------------------------------------------------
#  Pairwise cross-block pass — jointly optimize adjacent block pairs
# ---------------------------------------------------------------------------
#
# Per-block optimization is blind across the boundary: block i's output
# projection can compose with block i+1's input projections (a weight-only
# chain that folds to one stored matrix), and a residual ``+`` between
# them can absorb a shared affine.  For each *adjacent* pair the pass
# classifies the boundary, builds a joint micro-model wrapping
# ``B(A(x))``, runs the ordinary ``optimize_model`` on it, verifies
# it against the eager pair, and grafts the joint module into the clone —
# only when it is verified AND cheaper than the two separately-optimized
# results.

#: Symmetry budget for joint runs.  The joint is a two-block
#: micro-model — the reordering closure that motivated bounded
#: saturation dominates its cost, and the documented break-even is
#: identical extracted cost at every budget ≥ 512.  Truncating the
#: reordering closure can only *miss* rewrites (the pair then simply
#: declines on cost), never produce a wrong one.
_CROSS_PAIR_SYMMETRY_BUDGET = 512


def _perturbed_input(example_input: Any) -> Any:
    """Return a second probe input — different values, same structure.

    The residual-boundary check requires ``b_in == a_in + a_out``; on
    ONE example a coincidental value match could promote a false
    boundary, so the relation must also hold on a perturbed probe.
    Only floating tensors are perturbed — a perturbation would corrupt
    non-float (index) inputs.
    """

    def _perturb(t: Any) -> Any:
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            return t * 1.5 + 0.01
        return t

    if isinstance(example_input, tuple):
        return tuple(_perturb(a) for a in example_input)
    return _perturb(example_input)


def _residual_probe(
    name_a: str,
    name_b: str,
    captured2: Mapping,
    io2: Mapping,
) -> bool:
    """Second-probe confirmation of a residual boundary.

    On the perturbed-input capture, ``b_in == a_in + a_out`` must hold
    again — a coincidence of values on one example can't promote the
    pair.  Any absence (block not executed on the probe path, extra
    call, non-tensor piece) fails closed.
    """
    ca = captured2.get(name_a)
    cb = captured2.get(name_b)
    ia = io2.get(name_a)
    if ca is None or cb is None or ia is None or ia["calls"] != 1:
        return False
    args_a, _ = ca
    args_b, _ = cb
    a_out = ia["out"]
    if len(args_a) != 1 or len(args_b) != 1:
        return False
    a_in, b_in = args_a[0], args_b[0]
    if not (
        isinstance(a_in, torch.Tensor)
        and isinstance(b_in, torch.Tensor)
        and isinstance(a_out, torch.Tensor)
    ):
        return False
    return bool(
        a_in.shape == a_out.shape and torch.equal(b_in, a_in + a_out)
    )


def _executor_flops(mod: Any) -> float:
    """Delivered FLOPs of a lowered executor.

    Prices the module's post-fold root term — the computation that
    actually runs (weight chains are already materialised to single
    params).  Carrier-batched executors expose the serial root through
    ``eval_mod``; anything unpriceable returns ``inf`` so the pair
    comparison simply keeps the separate modules.
    """
    root = getattr(mod, "_root", None)
    if root is None:
        root = getattr(getattr(mod, "eval_mod", None), "_root", None)
    if root is None:
        return float("inf")
    return float(flops_cost(root))


class _JointPair(torch.nn.Module):
    """Joint micro-model for one adjacent pair, in the boundary's mode.

    ``mode`` is the :func:`_pair_boundary` verdict — the joint function
    of A's input ``x`` the optimizer sees:

    * ``"chain"`` — ``x |-> B(A(x))``
    * ``"chain_wrapped"`` — ``x |-> A(x) + B(A(x))`` (the parent's own
      ``y + B(y)`` around B is part of the segment)
    * ``"residual"`` — ``x |-> B(x + A(x))``
    * ``"residual_wrapped"`` — ``x |-> (x + A(x)) + B(x + A(x))``
    """

    def __init__(
        self,
        a: torch.nn.Module,
        b: torch.nn.Module,
        mode: str,
    ) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.a(x)
        if self.mode.startswith("residual"):
            y = x + y
        out = self.b(y)
        if self.mode.endswith("_wrapped"):
            out = y + out
        return out


class _FusedPair(torch.nn.Module):
    """Delivery wrapper grafted at block A's slot for a fused pair.

    ``inner`` is the jointly-optimized executor computing the whole
    segment as a function of A's input; B's slot becomes
    ``nn.Identity`` (B consumed plainly) or :class:`_Zero` (the parent
    residual-wraps B, so its slot must contribute a zero addend).

    For a residual A-boundary the parent's own ``x + ·`` still runs, so
    the wrapper returns the *delta* ``inner(x) - x`` and the outer add
    reconstructs ``inner(x)`` (to within one rounding step).
    """

    def __init__(self, inner: torch.nn.Module, delta: bool) -> None:
        super().__init__()
        self.inner = inner
        self.delta = delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.inner(x)
        return out - x if self.delta else out


class _Zero(torch.nn.Module):
    """Exact-zero placeholder for a consumed, residual-wrapped slot.

    The fused pair delivers the whole segment upstream; the parent's
    ``y + ·`` around B still executes, so this slot contributes an
    exact zero addend — adding literal zeros is lossless, unlike a
    computed ``y - y`` on non-finite values.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def _plain_consumers(
    io: Mapping,
    captured: Mapping,
    out_obj: Any,
    out_val: torch.Tensor,
) -> list[str]:
    """Block names whose captured args hold ``out`` (identity + value).

    Object identity proves the same tensor flowed in; comparing the
    captured clone rules out an in-place rewrite between producer and
    consumer.
    """
    hits = []
    for n, m in io.items():
        c_args = captured.get(n, ((), {}))[0]
        for arg, real in zip(
            c_args, m.get("in_objs", ()), strict=False
        ):
            if (
                real is out_obj
                and isinstance(arg, torch.Tensor)
                and torch.equal(arg, out_val)
            ):
                hits.append(n)
                break
    return hits


def _io_has_value(
    captured: Mapping,
    model_out: Any,
    val: torch.Tensor,
) -> bool:
    """Return True when ``val`` appears verbatim in the captured flow.

    Checks the model's return and every block's captured positional
    args — the evidence that a *computed* sum (like ``b_in + b_out``)
    is what actually flows on.
    """
    if isinstance(model_out, torch.Tensor) and torch.equal(
        model_out, val
    ):
        return True
    return any(
        isinstance(a, torch.Tensor) and torch.equal(a, val)
        for args, _ in captured.values()
        for a in args
    )


def _pair_boundary(
    name_a: str,
    name_b: str,
    captured: Mapping,
    io: Mapping,
    captured2: Mapping,
    io2: Mapping,
) -> str | None:
    """Classify the A→B dataflow of an adjacent block pair.

    Returns the joint-micro-model mode — ``"chain"`` /
    ``"chain_wrapped"`` / ``"residual"`` / ``"residual_wrapped"`` — or
    ``None`` when the boundary is not a simple value flow.

    A-side (what flows INTO B):

    * ``chain`` — B's input IS A's output: the same live tensor object,
      unmodified between the two calls (the captured values still
      compare equal), and consumed by B alone;
    * ``residual`` — B's input is ``a_in + a_out`` (the ``x + A(x)``
      residual pattern), confirmed on the perturbed second-probe
      capture too, and A's output feeds nothing else.

    B-side (how B's OUTPUT is consumed — it decides the graft, because
    the parent's ``b_in + B(b_in)`` wrap makes B's slot an addend, not
    a value):

    * plain — B's output object reaches another block or the model
      return unmodified → B's slot becomes ``nn.Identity``;
    * ``_wrapped`` — the sum ``b_in + b_out`` is what flows on → B's
      slot becomes :class:`_Zero` and the joint absorbs B's residual;
    * both or neither → ``None`` (ambiguous or unknown downstream).

    Identity checks run on the retained live objects (``is``); equality
    on detached clones — a coincidence of values alone never promotes a
    boundary.
    """
    ca = captured.get(name_a)
    cb = captured.get(name_b)
    ia = io.get(name_a)
    ib = io.get(name_b)
    if ca is None or cb is None or ia is None or ib is None:
        return None
    if ia["calls"] != 1 or ib["calls"] != 1:
        # A re-entered block is called again outside the pair window —
        # a fused graft would rewrite that later call too.
        return None
    args_a, kw_a = ca
    args_b, kw_b = cb
    if kw_a or kw_b or len(args_a) != 1 or len(args_b) != 1:
        return None
    a_in, b_in, a_out = args_a[0], args_b[0], ia["out"]
    if not (
        isinstance(a_in, torch.Tensor)
        and isinstance(b_in, torch.Tensor)
        and isinstance(a_out, torch.Tensor)
    ):
        return None
    model = io.get(_MODEL_KEY, {})
    model_out, model_out_obj = model.get("out"), model.get("out_obj")
    a_out_obj = ia["out_obj"]
    if a_out_obj is model_out_obj:
        return None  # A's output escapes the pair entirely
    a_fans = _plain_consumers(io, captured, a_out_obj, a_out)
    if ib["in_objs"][0] is a_out_obj:
        # B literally consumed A's output object — and nothing else did.
        if a_fans != [name_b]:
            return None
        a_mode = "chain"
    elif a_fans:
        # A's output feeds another block as well — not a simple edge.
        return None
    elif not (
        a_in.shape == a_out.shape
        and torch.equal(b_in, a_in + a_out)
        and _residual_probe(name_a, name_b, captured2, io2)
    ):
        return None
    else:
        a_mode = "residual"
    # B-side: how is B's own output consumed?
    b_out, b_out_obj = ib["out"], ib["out_obj"]
    if not isinstance(b_out, torch.Tensor):
        return None
    plain_ev = bool(
        _plain_consumers(io, captured, b_out_obj, b_out)
    ) or (
        b_out_obj is model_out_obj
        and isinstance(model_out, torch.Tensor)
        and torch.equal(model_out, b_out)
    )
    wrapped_ev = b_in.shape == b_out.shape and _io_has_value(
        captured, model_out, b_in + b_out
    )
    if plain_ev == wrapped_ev:
        # Neither evidence, or both (ambiguous downstream) — decline.
        return None
    return a_mode + ("_wrapped" if wrapped_ev else "")


def param_report(
    model: torch.nn.Module, optimized_module: torch.nn.Module
) -> dict:
    """Joint graph+parameter view of the optimized weights file.

    Which original parameters survive in the optimized realization,
    which were eliminated, and which were derived (folded) — the
    'optimized weights file' diff.

    The optimized module's state_dict IS the smaller weights file:
    ``_fold_weight_chains`` materialises derived tensors (``fused_*``)
    and ``_build_params`` registers only parameters the extracted term
    actually references, so eliminated subgraphs drop their weights
    automatically.  This function makes that auditable.
    """
    orig = {n: p for n, p in model.state_dict().items()}
    opt = {n: p for n, p in optimized_module.state_dict().items()}
    orig_names = {f"p_{n.replace('.', '_')}" for n in orig}
    opt_names = set(opt)
    eliminated = sorted(orig_names - opt_names)
    derived = sorted(n for n in opt_names if n not in orig_names)
    orig_bytes = sum(
        p.numel() * p.element_size() for p in orig.values()
    )
    opt_bytes = sum(p.numel() * p.element_size() for p in opt.values())
    return {
        "original_params": len(orig),
        "optimized_params": len(opt),
        "original_bytes": orig_bytes,
        "optimized_bytes": opt_bytes,
        "eliminated": eliminated,
        "derived": derived,
        "bytes_saved": orig_bytes - opt_bytes,
        "ratio": opt_bytes / orig_bytes if orig_bytes else 1.0,
    }


class TorchComposer:
    """The torch :class:`~catopt_core.ports.Composer` implementation.

    Thin port object over the module-level machinery in this file —
    the function names stay importable for the historical private
    paths (``catopt_orchestrator.optimize._select_blocks`` and friends resolve
    through the compatibility delegation too).
    """

    def blocks(
        self, model: Any, *, predicate: Callable | None = None
    ) -> list[tuple[str, Any]]:
        """Pick the top-most submodules to optimize independently."""
        return _select_blocks(model, predicate)

    def capture_inputs(
        self, model: Any, blocks: list[Any], example_input: Any
    ) -> tuple[
        dict[str, tuple[tuple, dict]], dict[str, dict[str, Any]]
    ]:
        """Record each block's first forward inputs via hooks."""
        return _capture_block_inputs(model, blocks, example_input)

    def graft(self, model: Any, replacements: Mapping) -> Any:
        """Install ``{dotted_name: optimized}`` replacements; return model."""
        for name, opt_mod in replacements.items():
            _replace_submodule(model, name, opt_mod)
        return model

    def clone_sharing(self, model: Any) -> Any:
        """Deepcopy structure, sharing the original's tensor storage."""
        return _shared_param_clone(model)

    def boundary(
        self,
        name_a: str,
        name_b: str,
        captured: Mapping,
        io: Mapping,
        captured2: Mapping,
        io2: Mapping,
    ) -> str | None:
        """Classify the pair boundary (:func:`_pair_boundary`)."""
        return _pair_boundary(
            name_a, name_b, captured, io, captured2, io2
        )

    def perturbed(self, example_input: Any) -> Any:
        """Return the second-probe input (:func:`_perturbed_input`)."""
        return _perturbed_input(example_input)

    # -- optional adapter hooks (not Composer protocol members) ------

    def param_report(self, model: Any, optimized: Any) -> dict:
        """Return the weight-file diff (:func:`param_report`)."""
        return param_report(model, optimized)

    def cross_pairs(
        self,
        blocks: list[tuple[str, Any]],
        captured: dict[str, tuple[tuple, dict]],
        io: dict[str, dict[str, Any]],
        captured2: dict[str, tuple[tuple, dict]],
        io2: dict[str, dict[str, Any]],
        replacements: dict[str, Any],
        block_reports: dict[str, dict],
        agg: dict[str, Any],
        *,
        rules: Any,
        max_iterations: int,
        max_enodes: int | None,
        max_memory_mb: float | None,
        cost_fn: CostFn,
        verify_tol: float,
        source: Source,
        sink: Sink,
        max_cross_pairs: int,
        verbose: bool,
    ) -> dict[str, dict[str, Any]]:
        """Jointly optimize adjacent block pairs across their boundary.

        For each consecutive pair ``(blocks[i], blocks[i+1])`` in
        execution order whose boundary is a simple value flow
        (:func:`_pair_boundary`), build the :class:`_JointPair`
        micro-model, run the ordinary ``optimize_model`` on it
        (pairing, residual folds and scale hoists apply across the
        two-block composition), verify the lowered joint against the
        eager pair at ``verify_tol``, and graft it — a
        :class:`_FusedPair` at A's slot, ``nn.Identity`` /
        :class:`_Zero` at B's — only when verified AND its delivered
        FLOPs beat the sum of the two separately-optimized results.

        Combinatorics are capped: adjacent pairs only, no overlap (a
        block consumed by a graft cannot re-pair), and at most
        ``max_cross_pairs`` joint optimization runs.  Every failure is
        a silent decline recorded as ``{pair: {"status", ...}}`` —
        ``"grafted"``, ``"declined"`` (with ``reason``), or
        ``"skipped"`` (with ``reason``).
        ``replacements``/``block_reports``/``agg`` are updated in place
        for grafted pairs so the aggregate param report keeps
        describing what is actually delivered.  ``block_reports``
        entries are the plain report dicts the orchestrator builds.
        """
        reports: dict[str, dict[str, Any]] = {}
        consumed: set[str] = set()
        attempts = 0
        for i in range(len(blocks) - 1):
            name_a, mod_a = blocks[i]
            name_b, mod_b = blocks[i + 1]
            pair = f"{name_a}+{name_b}"
            if name_a in consumed or name_b in consumed:
                reports[pair] = {
                    "status": "skipped",
                    "reason": "member already fused",
                }
                continue
            if name_a not in replacements or name_b not in replacements:
                reports[pair] = {
                    "status": "skipped",
                    "reason": "block not optimized",
                }
                continue
            mode = self.boundary(
                name_a, name_b, captured, io, captured2, io2
            )
            if mode is None:
                reports[pair] = {
                    "status": "skipped",
                    "reason": "no simple boundary",
                }
                continue
            if attempts >= max_cross_pairs:
                reports[pair] = {
                    "status": "skipped",
                    "reason": f"max_cross_pairs={max_cross_pairs}",
                }
                continue
            attempts += 1
            t0 = time.time()
            entry: dict[str, Any] = {"boundary": mode}
            try:
                joint = _JointPair(mod_a, mod_b, mode)
                (x,) = captured[name_a][0]
                # The joint run goes through the same monolithic
                # pipeline the per-block pass did — via the
                # module-level ``_joint_optimize`` seam so a stub can
                # intercept it in tests.
                opt_j, st_j = _joint_optimize(
                    joint,
                    x,
                    source=source,
                    sink=sink,
                    rules=_joint_rules(rules),
                    max_iterations=max_iterations,
                    max_enodes=max_enodes,
                    max_memory_mb=max_memory_mb,
                    cost_fn=cost_fn,
                    symmetry_budget=_CROSS_PAIR_SYMMETRY_BUDGET,
                    verbose=verbose,
                )
                entry["stats"] = st_j
                # Soundness gate, same tolerance convention as the
                # per-block verify — the eager pair vs its lowering.
                vr = verify_module(joint, opt_j, (x,), rtol=verify_tol)
                entry["rel_diff"] = vr.max_rel
                if not vr.passed:
                    entry["status"] = "declined"
                    entry["reason"] = (
                        f"joint verify failed: {vr.max_rel:.3e}"
                    )
                else:
                    j_cost = _executor_flops(opt_j)
                    sep = _executor_flops(
                        replacements[name_a]
                    ) + _executor_flops(replacements[name_b])
                    entry["joint_cost"] = j_cost
                    entry["separate_cost"] = sep
                    if j_cost >= sep:
                        entry["status"] = "declined"
                        entry["reason"] = "no cost improvement"
                    else:
                        fused = _FusedPair(
                            opt_j, delta=mode.startswith("residual")
                        )
                        pr_j = param_report(joint, fused)
                        pa = block_reports[name_a].get(
                            "param_report", {}
                        )
                        pb = block_reports[name_b].get(
                            "param_report", {}
                        )
                        delta = {
                            k: pr_j[k] - pa.get(k, 0) - pb.get(k, 0)
                            for k in (
                                "original_params",
                                "optimized_params",
                                "original_bytes",
                                "optimized_bytes",
                            )
                        }
                        drop = (f"{name_a}:", f"{name_b}:")
                        for k, v in delta.items():
                            agg[k] += v
                        agg["eliminated"] = [
                            e
                            for e in agg["eliminated"]
                            if not e.startswith(drop)
                        ]
                        agg["derived"] = [
                            e
                            for e in agg["derived"]
                            if not e.startswith(drop)
                        ]
                        agg["eliminated"] += [
                            f"{pair}:{n}" for n in pr_j["eliminated"]
                        ]
                        agg["derived"] += [
                            f"{pair}:{n}" for n in pr_j["derived"]
                        ]
                        replacements[name_a] = fused
                        replacements[name_b] = (
                            _Zero()
                            if mode.endswith("_wrapped")
                            else torch.nn.Identity()
                        )
                        block_reports[name_a]["cross_pair"] = pair
                        block_reports[name_b]["cross_pair"] = pair
                        consumed.update((name_a, name_b))
                        entry["status"] = "grafted"
            except Exception as e:
                entry["status"] = "declined"
                entry["reason"] = "error"
                entry["error"] = f"{type(e).__name__}: {e}"
            entry["time_s"] = time.time() - t0
            reports[pair] = entry
            if verbose:
                print(  # stdout-compat — the verbose-mode
                    # [Compositional] prints deliberately stayed stdout
                    # through the phase-3c logging migration (same
                    # family as optimize.py's).
                    f"[Compositional] pair {pair}: "
                    f"{entry['status']} ({mode})"
                )
        return reports
