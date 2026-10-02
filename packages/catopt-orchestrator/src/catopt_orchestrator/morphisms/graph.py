"""Stage 1 — the lifted graph.

:class:`MorphismGraph` lifts a model to block objects and boundary
wires: every node carries a :class:`BlockSig` (or lands opaque), every
wire carries the composer's boundary verdict.  Split out of
:mod:`catopt_orchestrator.morphisms` (plan 0011); the whole surface is
re-exported from that package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from typing import Any

from catopt_core.ir import IR
from catopt_core.laws.pairing import _exact_equal, _is_tensor
from catopt_core.ports import Composer, Source

from .boundary import _mi_boundary
from .signature import (
    BlockSig,
    InputSig,
    _act_index,
    _probe_value,
    _shape_tuple,
    _sig_liftable,
    block_signature,
)

# ---------------------------------------------------------------------------
#  Stage 1 — the lifted graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MorphismNode:
    """One block object in the lifted graph.

    ``opaque`` marks a *boundary node* — the block exported no usable
    signature (export failure, never executed on the probe, non-tensor
    call signature).  Opaque nodes stay wired in place (they are real
    objects in the composition) but no morphism law matches them.
    """

    name: str
    sig: BlockSig | None
    opaque: bool


@dataclass(frozen=True)
class Wire:
    """One arrow between adjacent blocks — the boundary verdict.

    ``kind`` is the composer's joint-mode classification
    (``"chain"`` / ``"chain_wrapped"`` / ``"residual"`` /
    ``"residual_wrapped"``); ``"opaque"`` records a boundary that is
    not a simple value flow (re-entry, fan-out, non-tensor pieces) —
    no morphism law crosses it.
    """

    src: str
    dst: str
    kind: str


@dataclass
class _BlockRecord:
    """The lift record for one node — IR, leaves, captured IO.

    Beyond the exported IR and leaf values, the record keeps the
    captured-IO evidence morphism laws read: ``in_obj`` / ``out_obj``
    are the *live* argument/output objects of the block's first call
    (object identity is the structural proof two blocks shared an
    input or that an output escaped), ``out_val`` its detached output
    clone, ``calls`` the invocation count, and ``example2`` /
    ``out_val2`` the perturbed-probe counterparts.  For multi-input
    blocks ``in_obj``/``example``/``example2`` track the *activation*
    argument (``act`` — the arg position the signature's activation
    input occupies), while ``in_objs``/``in_objs2`` keep the full
    per-call arg object tuples for the context-input identity map.
    """

    name: str
    module: Any
    ir: IR | None = None
    leaves: dict[str, Any] = field(default_factory=dict)
    args: tuple = ()
    example: Any = None
    note: str | None = None
    in_obj: Any = None
    in_objs: tuple = ()
    in_objs2: tuple = ()
    out_obj: Any = None
    out_val: Any = None
    calls: int = 0
    act: int = 0
    example2: Any = None
    out_val2: Any = None


def _io_evidence(
    rec: _BlockRecord, ient: dict, cap2: Any, ient2: dict
) -> None:
    """Store the captured-IO evidence on the record.

    Live objects (``in_obj``/``out_obj``/``in_objs``) carry identity
    evidence — two blocks sharing one input object consume literally
    the same tensor; detached clones (``out_val``/``out_val2``)
    carry the value evidence the additive-consumption checks
    compare.  ``rec.act`` selects the activation argument position.
    """
    in_objs = tuple(ient.get("in_objs") or ())
    rec.in_objs = in_objs
    rec.in_obj = in_objs[rec.act] if rec.act < len(in_objs) else None
    rec.in_objs2 = tuple(ient2.get("in_objs") or ())
    rec.out_obj = ient.get("out_obj")
    rec.out_val = ient.get("out")
    rec.calls = int(ient.get("calls", 0))
    if cap2 is not None and rec.act < len(cap2[0]):
        rec.example2 = cap2[0][rec.act]
    rec.out_val2 = ient2.get("out")


def _aux_outs(io: Any, names: set) -> tuple[tuple, tuple]:
    """Return the non-block io entries' (out values, out objects).

    Entries keyed by something that is not a block name are
    model-level rows — their outputs are the "consumed value"
    targets the family laws check.
    """
    outs: list = []
    objs: list = []
    for k, e in (io or {}).items():
        if k in names or not isinstance(e, dict):
            continue
        if e.get("out") is not None:
            outs.append(e["out"])
        if e.get("out_obj") is not None:
            objs.append(e["out_obj"])
    return tuple(outs), tuple(objs)


class MorphismGraph:
    """The lifted graph: blocks as objects, wires as arrows.

    Built by :func:`lift_graph` from the Composer port's block list
    plus the captured IO (``capture_inputs`` + ``boundary`` — the same
    machinery the compositional strategy drives).  ``nodes`` are in
    execution order; ``wires`` carry the boundary verdict for each
    adjacent pair; :meth:`record` returns the per-block lift record
    the reify recipes read.
    """

    def __init__(
        self,
        nodes: list[MorphismNode],
        wires: list[Wire],
        records: dict[str, _BlockRecord],
        io: Any = None,
        io2: Any = None,
    ) -> None:
        """Store the graph — construction is :func:`lift_graph`'s job.

        ``io`` / ``io2`` are the composer's captured dataflow maps
        (first capture and perturbed probe); entries whose key is not
        a block name are model-level rows — their outputs are the
        "consumed value" targets the family laws check.
        """
        self.nodes = tuple(nodes)
        self.wires = tuple(wires)
        self._records = records
        self._by_name = {n.name: n for n in self.nodes}
        names = set(self._by_name)
        self._model_outs, self._model_out_objs = _aux_outs(io, names)
        self._model_outs2 = _aux_outs(io2, names)[0]
        self._probe2 = bool(io2)

    def node(self, name: str) -> MorphismNode:
        """Return the :class:`MorphismNode` for *name*."""
        return self._by_name[name]

    def sig(self, name: str) -> BlockSig | None:
        """Return the node's signature (``None`` when opaque)."""
        return self._by_name[name].sig

    def record(self, name: str) -> _BlockRecord:
        """Return the block's lift record (IR, leaves, captured args)."""
        return self._records[name]

    def __repr__(self) -> str:
        """Return a compact summary of the lifted graph."""
        return (
            f"MorphismGraph(nodes={len(self.nodes)}, "
            f"wires={len(self.wires)}, "
            f"opaque={sum(1 for n in self.nodes if n.opaque)})"
        )


def _mutates_arg(mod: Any, args: tuple) -> str | None:
    """Probe in-place mutation: rerun the block on perturbed clones.

    A functionalised export drops a write-only mutation outright (the
    written value is never returned), so the IR cannot see it — the
    empirical probe is the honest check: run the block on perturbed
    arg clones and compare each arg against its own pre-run clone.
    Perturbing first defeats idempotent writes (``cache[0] = x[0]``
    reproduces the captured state on an unperturbed replay); a write
    whose value depends on the args — or is a constant — then shows.
    ``None`` = provably no in-place write; a string = the decline
    reason (a mutated position, or an un-runnable probe — a block
    that cannot be replayed standalone is not composable).
    """
    try:
        probe = [_probe_value(a, i) for i, a in enumerate(args)]
        before = [a.clone() if _is_tensor(a) else a for a in probe]
        mod(*probe)
    except Exception as e:
        return f"input-mutation probe failed: {type(e).__name__}"
    for i, (b, a) in enumerate(zip(before, probe, strict=True)):
        if _is_tensor(b) and not _exact_equal(b, a):
            return f"mutates input {i}"
    return None


def _param_bound_ids(model: Any) -> set[int]:
    """Object ids of the model's param/buffer/attr tensors."""
    ids: set[int] = set()
    for coll in (
        getattr(model, "parameters", list)(),
        getattr(model, "buffers", list)(),
    ):
        for t in coll:
            ids.add(id(t))
    for mod in getattr(model, "modules", list)():
        for v in vars(mod).values():
            if _is_tensor(v):
                ids.add(id(v))
    return ids


def _refine_inputs(
    sig: BlockSig,
    rec: _BlockRecord,
    cap2: Any,
    const_ids: set[int],
) -> BlockSig:
    """Refine the provisional kinds with capture/probe evidence.

    A context input whose captured value is identical across both
    probes — or whose live arg object IS a model param/buffer/tensor
    attr — is ``const_table`` (Param/Const-bound context).  A
    context input whose value varies between the captures is
    call-dependent state — ``state``/``read_only`` (a KV cache read
    positionally, a per-step index table).  Activations and opaque
    inputs pass through unchanged.
    """
    refined: list[InputSig] = []
    for inp in sig.inputs:
        if inp.kind != "const_table" or inp.index >= len(rec.in_objs):
            refined.append(inp)
            continue
        obj = rec.in_objs[inp.index]
        v1 = rec.args[inp.index] if inp.index < len(rec.args) else None
        v2 = (
            cap2[0][inp.index]
            if cap2 is not None and inp.index < len(cap2[0])
            else None
        )
        bound = id(obj) in const_ids
        invariant = (
            _is_tensor(v1) and _is_tensor(v2) and _exact_equal(v1, v2)
        )
        if bound or invariant:
            refined.append(inp)
        else:
            refined.append(
                _dc_replace(inp, kind="state", role="read_only")
            )
    return _dc_replace(sig, inputs=tuple(refined))


def _lift_block(
    rec: _BlockRecord,
    mod: Any,
    cap: Any,
    cap2: Any,
    ient: dict,
    ient2: dict,
    source: Source,
    const_ids: set[int],
) -> BlockSig | None:
    """Export, classify, probe and refine one captured block's sig.

    Every decline records ``rec.note`` and returns ``None`` — the
    node lands opaque on the graph.  Success returns the refined
    signature and leaves the record's activation-anchored IO
    evidence populated.
    """
    if cap is None:
        rec.note = "not executed on the captured input"
        return None
    if cap[1] or not cap[0]:
        rec.note = "not a single-positional-arg call"
        return None
    if not all(_is_tensor(a) for a in cap[0]):
        # A non-tensor positional arg shifts the input/arg position
        # correspondence — the lowered module's positional binding
        # could not reproduce the call.
        rec.note = "non-tensor positional argument"
        return None
    rec.args = cap[0]
    rec.example = cap[0][0]
    _io_evidence(rec, ient, cap2, ient2)
    try:
        rec.ir, rec.leaves = source.to_ir(
            mod, cap[0] if len(cap[0]) > 1 else cap[0][0]
        )
    except Exception as e:
        rec.note = f"export failed: {type(e).__name__}: {e}"
        return None
    sig = block_signature(
        rec.ir,
        out_shape=_shape_tuple(ient.get("out")),
    )
    decline = _sig_liftable(sig)
    if decline is not None:
        rec.note = decline
        return None
    # Re-anchor the IO evidence on the activation input.
    rec.act = _act_index(sig.inputs)
    rec.example = cap[0][rec.act]
    if rec.act < len(rec.in_objs):
        rec.in_obj = rec.in_objs[rec.act]
    if cap2 is not None and rec.act < len(cap2[0]):
        rec.example2 = cap2[0][rec.act]
    mut = _mutates_arg(mod, cap[0])
    if mut is not None:
        rec.note = mut
        return None
    return _refine_inputs(sig, rec, cap2, const_ids)


def lift_graph(
    model: Any,
    x: Any,
    *,
    source: Source,
    composer: Composer,
    block_pred: Any = None,
) -> MorphismGraph:
    """Lift a model to its :class:`MorphismGraph` — signature level.

    Blocks come from the composer's ``blocks`` port (the same
    selection the compositional strategy uses); each block's IR is
    exported through the ``source`` port on its captured call args
    and summarised by :func:`block_signature` — *multi-input calls
    lift*: ``(x, cos, sin)`` attention blocks and ``(x, kv_cache)``
    state readers classify their inputs, and the block lifts when
    exactly one input is the activation and the rest are
    pass-through context (``const_table`` / read-only ``state``).
    The honest declines stay boundary nodes: multi-activation calls,
    blocks that mutate an input, unclassifiable input roles,
    non-tensor args, kwargs calls, export failures, and blocks that
    never executed.

    Wires between adjacent blocks come from ``composer.boundary``
    over the two-capture IO evidence; when the composer cannot
    classify a boundary that crosses a multi-input block, the
    position-aware fallback :func:`_mi_boundary` classifies the
    *activation* edge (the context args are not part of the wire —
    they pass through each block's own call).
    """
    blocks = composer.blocks(model, predicate=block_pred)
    captured, io = composer.capture_inputs(model, blocks, x)
    try:
        captured2, io2 = composer.capture_inputs(
            model, blocks, composer.perturbed(x)
        )
    except Exception:
        captured2, io2 = {}, {}
    const_ids = _param_bound_ids(model)
    nodes: list[MorphismNode] = []
    records: dict[str, _BlockRecord] = {}
    for name, mod in blocks:
        rec = _BlockRecord(name=name, module=mod)
        records[name] = rec
        sig = _lift_block(
            rec,
            mod,
            captured.get(name),
            captured2.get(name),
            io.get(name, {}),
            io2.get(name, {}),
            source,
            const_ids,
        )
        nodes.append(
            MorphismNode(name=name, sig=sig, opaque=sig is None)
        )

    node_map = {n.name: n for n in nodes}
    wires = [
        Wire(
            blocks[i][0],
            blocks[i + 1][0],
            composer.boundary(
                blocks[i][0],
                blocks[i + 1][0],
                captured,
                io,
                captured2,
                io2,
            )
            or _mi_boundary(
                blocks[i][0],
                blocks[i + 1][0],
                captured,
                io,
                captured2,
                io2,
                node_map,
            )
            or "opaque",
        )
        for i in range(len(blocks) - 1)
    ]
    return MorphismGraph(nodes, wires, records, io=io, io2=io2)
