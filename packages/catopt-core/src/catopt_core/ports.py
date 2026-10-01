"""Ports — catopt's hexagonal-architecture boundary layer.

The codebase has always had *informal* ports — the callables,
registries and module surfaces the domain logic depends on without
naming their contracts: ``_IR_TO_TORCH`` entries, ``register_shape_rule``
handlers, ``fn(term, memo) -> float`` cost models, the executor
modules, ``verify_equiv``, and the ``OpTable`` composition point.
This module names those contracts.  Each is a ``typing.Protocol``
marked ``@runtime_checkable`` so ``isinstance`` performs a real
structural check.  Zero new dependencies — ``Protocol`` and
``inspect`` are stdlib.

Inside the hexagon (the domain core)
------------------------------------
``catopt_core.ir`` terms, ``catopt_core.egraph`` (union-find, e-matching,
equality saturation, extraction, certificates), ``catopt_core.laws``
(equational rewrites), ``catopt_core.typing`` (shape inference),
``catopt_core.cost`` (pricing).  These know about terms and e-classes —
never about ``torch`` module objects.

The ports (this file)
---------------------
* :class:`CostFn` — extract-time pricing:
  ``(term, memo=None) -> float``.  Every builtin cost model conforms.
* :class:`RuleLike` / :class:`RuleSetLike` / :class:`RuleSetProvider`
  — laws as data: the structural read-surface of ``egraph.Rewrite``
  and the collections ``EGraph.run`` consumes
  (:class:`catopt_core.laws.RuleSet` is the concrete implementation).
* :class:`ShapeRule` — the ``typing.register_shape_rule`` handler
  contract, ``fn(op, shapes) -> shape | _INVALID | None``.
* :class:`Executor` / :class:`PlannedExecutor` /
  :class:`BatchedExecutor` — the lowered-module contract.
* :class:`Verifier` — the semantic-equivalence gate,
  ``(ref_out, out, rtol, atol) -> VerifyResult``.
* :class:`VerifyResult` — the core-owned structural view of a verify
  result (``max_abs`` / ``max_rel`` / ``passed``); the torch adapter's
  ``report.VerifyReport`` satisfies it without core naming it.
* :class:`TaskMetric` — the task-level equivalence contract (plan
  0015): ``distance(ref_out, opt_out) -> float`` against a
  ``tolerance``, named by ``name``; the shipped implementations are
  in :mod:`catopt_core.metrics`.
* :class:`Source` — the whole-graph source port,
  ``model -> (IR, leaves)``.
* :class:`Capabilities` — the backend's declared op surface:
  ``supported_ops`` (the extraction bound) + ``ops`` (the registry
  search-time const folds dispatch through).
* :class:`Sink` — ``Capabilities`` + the whole-graph sink port,
  ``IR -> runnable``, plus the module-level equivalence gate and the
  backend's executor table (``executors``).
* :class:`ExecutorSpec` — one named executor entry of that table
  (``lower`` / ``accepts`` / ``engaged`` / ``carrier``).
* :class:`Composer` — the structural port the compositional strategy
  uses: module-tree block selection, hooked input capture, grafting
  and parameter-sharing clones.
* :class:`Meter` / :class:`TimingResult` — the timing port the
  autotuned strategy uses: ``time(runnable, inputs) -> (median, iqr)``.
* :class:`Runner` — the delivery transform port,
  ``apply(module, example_input, stats) -> module``.
* :class:`Criterion` — one cost-model selection axis,
  ``cost_fn(profile) -> CostFn``.
* :class:`Strategy` — the optimization-policy seam behind
  ``Optimizer.optimize(..., strategy=...)``.
* :class:`Binding` — one op's lowering, ``(*args, **attrs)``
  (``TorchBinding`` is the historical alias of the same protocol).
* :class:`OpRegistry` — the adapter-registry port.
* :class:`Engine` — the saturation *engine* seam (plan 0010, lever 3):
  the search core behind ``search``/``Optimizer`` — ``add_term`` /
  ``find`` / ``run`` / ``extract_best``.  ``EGraph`` is the reference
  implementation and the default; ``catopt_native.NativeEngine`` is the
  optional native accelerator.  Engine selection is explicit only —
  never auto-detected.

Outside the hexagon (the adapters)
----------------------------------
The torch-facing implementations: ``torch_bridge._CORE_TORCH_BINDINGS``
and each carrier module's ``TORCH_BINDINGS`` entries are
``Binding``s; ``IRModule`` / ``BatchedScanModule`` /
``BatchedOMModule`` / ``StreamingOMModule`` / ``BatchedOmdModule`` are
``Executor``s; ``report.verify_equiv`` is the canonical ``Verifier``;
``catopt_torch.adapters.TorchSource`` / ``TorchSink`` are the canonical
``Source`` / ``Sink`` pair; each ``_SHAPE_RULES`` entry is a
``ShapeRule``.  ``catopt_torch.composer.TorchComposer`` /
``catopt_torch.meter.TorchMeter`` are the canonical
``Composer`` / ``Meter``; ``catopt_torch.backend.TorchBackend``
bundles the four into the immutable
:dataclass:`~catopt_core.pipeline.Backend` value the orchestrator
consumes.  ``ops.OpTable`` is the
adapter *registry* — it composes the adapters and is already the right
shape, so ``OpRegistry`` describes its surface rather than re-wrapping
it.  The ambient ``torch_bridge._IR_TO_TORCH`` dict remains the live
binding table a ``full()`` table seats.

Runtime checks
--------------
``isinstance`` on a ``@runtime_checkable`` protocol verifies member
*presence* only: a callable port checks that ``__call__`` exists, a
data member that ``inspect.getattr_static`` resolves it.  Signature
*shape* is not checked — any callable ``isinstance``s as a ``CostFn``.
When the signature itself matters, use :func:`signature_conforms`,
which probes ``fn`` against the port's declared ``__call__`` the same
way the real call sites invoke it.

Two presence-check subtleties discovered the hard way:

* ``torch.nn.Module`` diverts Module/Parameter/Tensor attributes into
  ``_modules``/``_parameters``/``_buffers`` — a static getattr never
  sees them.  That is why ``eval_mod`` (a submodule) cannot be a
  protocol member while ``_plan`` (a plain dict attribute) can — see
  :class:`PlannedExecutor`.
* Per-adapter attrs declared as ``@property`` resolve fine (they live
  on the class), so ``is_batched`` is a valid member but an instance
  ``self.fallbacks = 0`` on *one* executor still isn't common enough
  to port.

Deliberate non-fits
-------------------
* ``optimize_model`` / ``ir_to_torch_module`` / ``to_batched_*`` keep
  their concrete return types (``torch.nn.Module`` / ``IRModule`` /
  ``BatchedOMModule`` ...), not ``Executor``: callers legitimately use
  ``state_dict``/``parameters`` and the per-adapter introspection attrs
  (``is_batched``, ``map_mode``, ``n_levels``, ``n_blocks``,
  ``fallbacks``, ``is_streaming``, ``capture_cuda_graph``) that are not
  port members.
* ``StreamingOMModule`` is an ``Executor`` and a ``PlannedExecutor``
  but not a ``BatchedExecutor`` — its discriminator is named
  ``is_streaming``.
* ``ops`` parameters stay typed ``OpTable | None``: ``OpTable`` IS the
  registry port — ``OpRegistry`` names its surface for structural
  checks; nothing is re-wrapped.
* ``RuleSetLike`` describes the iterable-of-``Rewrite`` surface
  ``EGraph.run`` consumes — plain ``list[Rewrite]`` collections and
  the concrete :class:`catopt_core.laws.RuleSet` both conform.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from catopt_core.ir import IR, Op
    from catopt_core.pipeline import LowerResult

__all__ = [
    "BatchedExecutor",
    "Binding",
    "Capabilities",
    "Composer",
    "CostFn",
    "Criterion",
    "Engine",
    "Executor",
    "ExecutorSpec",
    "Meter",
    "OpRegistry",
    "PlannedExecutor",
    "RuleLike",
    "RuleSetLike",
    "RuleSetProvider",
    "Runner",
    "ShapeRule",
    "Sink",
    "Source",
    "Strategy",
    "TaskMetric",
    "TimingResult",
    "TorchBinding",
    "Verifier",
    "VerifyResult",
    "signature_conforms",
]


# ---------------------------------------------------------------------------
#  Adapter ports — torch-facing semantics
# ---------------------------------------------------------------------------


@runtime_checkable
class Binding(Protocol):
    """One op's lowering: ``fn(*operand_values, **attrs) -> Any``.

    Adapter port.  Entries of ``torch_bridge._CORE_TORCH_BINDINGS``,
    each carrier module's ``TORCH_BINDINGS`` export, and values of
    ``OpTable.torch_bindings`` conform.  The contract is deliberately
    loose — the operand arity is per-op (the e-node's children), not
    fixed by the port — so ``torch.matmul`` and the variadic attr-
    swallowing lambdas in the core table both satisfy it.  Results may
    be tensors *or* carrier values (``(A, b)`` pairs, ``(m, l, a)``
    triples): the carrier packaging ops return tuples by design.

    The port is backend-neutral — a numpy/JAX lowering table's entries
    conform too; the name is historical.  ``TorchBinding`` remains as
    an alias of the same protocol.
    """

    def __call__(self, *args: Any, **attrs: Any) -> Any:
        """Lower one op with its operand args and attrs."""
        ...


#: Historical alias of :class:`Binding` — the lowering port predates the
#: backend-neutral spelling.  The same object, kept so ``isinstance``
#: checks and annotations written against ``TorchBinding`` keep working.
TorchBinding = Binding


@runtime_checkable
class Executor(Protocol):
    """The lowered-module contract every executor already satisfies.

    Adapter port.  A lowered IR term is a ``torch.nn.Module`` callable
    as ``mod(*xs) -> torch.Tensor`` — the ``nn.Module.__call__``
    dispatch reaches ``forward``, so declaring ``forward`` covers both.
    Members: ``IRModule`` (torch_bridge), ``BatchedScanModule``
    (scan_lower), ``BatchedOMModule`` / ``StreamingOMModule``
    (om_lower), ``BatchedOmdModule`` (omd_lower).

    ``xs``/return are ``Any``: core is backend-neutral and names no
    tensor library (the torch adapter's values are ``torch.Tensor``).
    """

    def forward(self, *xs: Any) -> Any:
        """Run the lowered module on ``xs``."""
        ...


@runtime_checkable
class PlannedExecutor(Executor, Protocol):
    """An :class:`Executor` that builds a lowering *plan* once.

    The batched/streaming wrappers all analyse the term in
    ``__init__`` into ``self._plan`` (``None`` when the root doesn't
    match the planned pattern — the serial evaluator then runs the
    whole forward) and keep a plain ``IRModule`` as submodule
    ``eval_mod`` for operand evaluation and as the complete fallback —
    what makes each wrapper a drop-in replacement for
    ``ir_to_torch_module`` on any IR.

    Why ``_plan`` and not ``eval_mod`` as the protocol member:
    ``eval_mod`` is the logical member, but it is not *runtime-
    checkable* — ``@runtime_checkable`` resolves data members with
    ``inspect.getattr_static``, and ``torch.nn.Module`` stores
    submodule values in ``_modules`` rather than ``__dict__``, so a
    static getattr misses it.  ``_plan`` is a plain ``__dict__``
    attribute every wrapper sets, so the presence check is real.
    Static type checkers see ``eval_mod`` on the classes regardless.
    """

    _plan: dict | None


@runtime_checkable
class BatchedExecutor(PlannedExecutor, Protocol):
    """A :class:`PlannedExecutor` with the ``is_batched`` discriminator.

    Shared by ``BatchedScanModule`` / ``BatchedOMModule`` /
    ``BatchedOmdModule`` — which additionally share the CUDA-graph API
    (``capture_cuda_graph`` / ``drop_cuda_graph`` /
    ``is_graph_captured``) and the per-adapter attrs (``n_levels`` /
    ``n_blocks`` / ``map_mode`` / ``fallbacks``); both stay
    class-level API, not port members.  ``StreamingOMModule``
    deliberately diverges: its discriminator is ``is_streaming``.
    """

    @property
    def is_batched(self) -> bool:
        """Whether the executor runs batched."""
        ...


@dataclass(frozen=True)
class ExecutorSpec:
    """One named entry of a backend's :attr:`Sink.executors` table.

    The registry record the executor-routing and regime-planning code
    consumes (promoted to core in plan 0007 — the same contract the
    regime-level ``ExecutorSpec`` always was, with backend-neutral
    member types):

    * ``name`` — the registry key (``"generic"``, ``"scan"``,
      ``"om_batched"``, ``"om_streaming"``, ``"trace"``, …).
    * ``lower`` — build the executor: ``(IR, param_values | None) ->
      Executor``.
    * ``accepts`` — the *term-level* probe: does ``term`` carry this
      executor's native root shape, so its specialised schedule
      actually fires (not just its serial fallback)?
    * ``engaged`` — the *module-level* probe after building: did the
      executor take the fast path (e.g. ``mod.is_batched``)?
    * ``carrier`` — ``(root_ops, inner_ops, leaf_ops)`` frozensets
      naming the carrier family this executor serves, for
      census/diagnosis; ``None`` for executor-agnostic lowerings (the
      generic evaluator).
    """

    name: str
    lower: Any  # Callable[[IR, dict | None], Executor]
    accepts: Any  # Callable[[Any], bool]
    engaged: Any  # Callable[[Executor], bool]
    carrier: Any = None
    # tuple[frozenset[str], frozenset[str], frozenset[str]] | None


@runtime_checkable
class OpRegistry(Protocol):
    """The adapter-registry port: op semantics, by name.

    ``catopt_core.ops.OpTable`` is the canonical (and only) implementation —
    the registry was extracted in plan 0001 phase 2c precisely so that
    binding sets compose explicitly instead of via import-time global
    mutation.  Consumers (``IRModule._eval``, ``optimize_model``'s
    causal specialization) read ``torch_bindings``; ``shape_rules`` /
    ``attr_schemas`` ride along as inspectable fragments;
    :meth:`register` is the composition surface.
    """

    torch_bindings: dict[str, Binding]
    shape_rules: dict[str, ShapeRule]
    attr_schemas: dict[str, dict[int, str]]

    def register(
        self,
        source: Any = None,
        *,
        torch_bindings: dict | None = None,
        shape_rules: dict | None = None,
        attr_schema: dict | None = None,
    ) -> OpRegistry:
        """Compose bindings and shape/attr fragments; return self."""
        ...


# ---------------------------------------------------------------------------
#  Domain ports — what the core calls into
# ---------------------------------------------------------------------------


@runtime_checkable
class ShapeRule(Protocol):
    """The ``typing.register_shape_rule`` handler contract.

    ``fn(op, shapes) -> tuple | _INVALID | None``.

    Domain port — installed in ``catopt_core.typing._SHAPE_RULES`` and
    consulted by ``_infer_op_shape`` after ``_INVALID`` propagation and
    the unknown/empty early-returns, so ``shapes`` contains only
    non-``None`` operand shapes (``()`` for zero-argument constant
    morphisms like ``eye``/``cswap``).  Returns a shape tuple, the
    ``_INVALID`` sentinel, or ``None`` (unknown); a carrier-internal
    ``()`` operand must map to ``None``, never a fabricated scalar.

    The params are positional-only — every call site invokes
    ``rule(op, shapes)`` positionally, which is also what makes the
    ``Callable`` alias ``typing._ShapeRule`` structurally assignable.
    """

    def __call__(self, op: Op, shapes: list, /) -> tuple | str | None:
        """Return the op's output shape, ``_INVALID``, or ``None``."""
        ...


@runtime_checkable
class CostFn(Protocol):
    """Extract-time pricing: ``(term, memo=None) -> float``.

    Domain port — the cost model extraction minimizes
    (``EGraph.extract_best`` / ``extract_alternatives`` /
    ``extract_paired``) and ``optimize_model``'s ``cost_fn`` parameter.
    The ``memo`` parameter is *adaptively* supplied:
    ``egraph.extract`` and ``cost.dag_cost`` pass it only when the cost
    fn declares it (``inspect.signature`` introspection — kept as-is),
    so a plain ``fn(term)`` also conforms at the call sites — see
    :func:`signature_conforms` for that looser reading.  Markers read
    off the function object (``charges_param_only``, ``dag_exact``)
    are per-model extras, not port members.
    """

    def __call__(self, term: Any, memo: dict | None = None) -> float:
        """Price *term*, optionally threading *memo*."""
        ...


@runtime_checkable
class VerifyResult(Protocol):
    """Core's structural view of an equivalence-check result.

    Domain port — the return of :class:`Verifier` and
    :meth:`Sink.verify`.  Core names no adapter type: the torch
    adapter's ``report.VerifyReport`` (a frozen dataclass) structurally
    satisfies it, and so does any backend's own result object.  The
    three members are exactly the fields every call site reads:

    * ``max_abs`` — ``max|ref - out|``;
    * ``max_rel`` — the historical rel-diff metric
      ``max|ref - out| / (max|ref| + 1e-8)``;
    * ``passed`` — the gate outcome.

    The members are declared read-only so a frozen dataclass (the torch
    adapter's ``VerifyReport``) satisfies the port; a mutable object
    satisfies it too (it only needs to be readable).
    """

    @property
    def max_abs(self) -> float:
        """Return ``max|ref - out|``."""
        ...

    @property
    def max_rel(self) -> float:
        """Return the historical rel-diff metric."""
        ...

    @property
    def passed(self) -> bool:
        """Return the gate outcome."""
        ...


@runtime_checkable
class Verifier(Protocol):
    """The semantic-equivalence gate.

    ``(ref_out, out, rtol, atol) -> VerifyResult``.

    Domain port — ``report.verify_equiv`` is the canonical
    implementation (the rel-diff metric every call site shares), with
    ``report.verify_module`` as the module-level driver.  Call sites
    pass ``rtol``/``atol`` by name, so a conforming verifier must
    accept them.
    """

    def __call__(
        self,
        ref_out: Any,
        out: Any,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> VerifyResult:
        """Compare ``ref_out`` and ``out`` under ``rtol``/``atol``."""
        ...


@runtime_checkable
class TaskMetric(Protocol):
    """A task-level equivalence metric — the certificate's task contract.

    Domain port (plan 0015).  A pointwise bound (``max_rel ≤ rtol``)
    is one statement of equivalence, but the user's real contract is
    often *behavioural*: an LLM cares about the logits' ranking, a
    classifier about the argmax, a retrieval model about the cosine
    ordering of its embeddings.  A ``TaskMetric`` reduces a pair of
    module outputs to a scalar ``distance``; ``tolerance`` is the
    metric's default gate (the pipeline's ``task_tol`` overrides it)
    and ``name`` is the label the stats/manifest record.

    ``distance(ref_out, opt_out) -> float`` — smaller is closer;
    ``0.0`` means indistinguishable under the task.  ``ref_out`` /
    ``opt_out`` are whatever the lowered modules returned — a
    tensor-like or a pytree of them — and the metric must evaluate
    them by duck typing (``tolist`` …): core names no tensor library.
    The shipped implementations live in :mod:`catopt_core.metrics`
    (:class:`~catopt_core.metrics.MaxRel` restates the pointwise
    gate itself, so ``verify_metric="max_rel"`` names the same
    metric either way).

    The metric is **calibration-conditioned**: ``distance`` is
    evaluated on the verify input, so the certificate's equivalence
    claim holds on that input distribution — distribution shift is
    the honest caveat, recorded via ``evaluated_on`` in the manifest.
    """

    @property
    def name(self) -> str:
        """The label recorded into ``stats`` and the manifest."""
        ...

    @property
    def tolerance(self) -> float:
        """The metric's default gate; ``task_tol`` overrides it."""
        ...

    def distance(self, ref: Any, opt: Any) -> float:
        """Scalar distance between the two modules' outputs."""
        ...


# ---------------------------------------------------------------------------
#  Graph ports — the whole-graph source / sink boundary
# ---------------------------------------------------------------------------


@runtime_checkable
class Source(Protocol):
    """The graph-source port: a backend-native model -> catopt IR.

    Adapter port.  ``catopt_torch.adapters.TorchSource`` is the canonical
    implementation (``torch.export`` → ATen → IR).  The return is the
    IR plus the concrete leaf values (parameters and buffers) keyed by
    IR param name — the paired :class:`Sink` materialises the lowered
    module from them, and the non-local passes read them for exact
    weight identity.  ``model`` / ``example_inputs`` are ``Any``
    because only the adapter knows the frontend's types.
    """

    def to_ir(
        self, model: Any, example_inputs: Any
    ) -> tuple[IR, dict[str, Any]]:
        """Export ``model`` to IR plus its leaf values."""
        ...


@runtime_checkable
class Capabilities(Protocol):
    """The backend's declared op surface — what the *search* needs.

    Adapter port, split out of :class:`Sink` (plan 0006): the search
    phase consumes only the op surface, not materialisation, so a
    backend that wants to *price* a search — or arm its compile-time
    folds — implements just this.

    * ``supported_ops`` bounds the reachable equivalence class.  It is
      the set of op names this backend can lower; extraction prices any
      member that uses an op outside it at ``+inf`` (see
      :func:`catopt_core.cost.backend_cost`), so the search never
      commits to a form the sink cannot execute — the backend
      counterpart of the semantic-language bound.  A backend without
      ``sdpa`` simply never selects the attention fold.
    * ``ops`` is the lowering registry (an :class:`OpRegistry`) the
      compile-time const folds — e.g. the causal-mask specialization —
      dispatch through.  ``specialize_causal``'s declared need.

    :class:`Sink` adds ``lower`` / ``verify``; every ``Sink`` is a
    ``Capabilities``.
    """

    @property
    def supported_ops(self) -> frozenset[str]:
        """Return the set of op names the backend can lower."""
        ...

    @property
    def ops(self) -> OpRegistry:
        """Return the lowering registry."""
        ...


@runtime_checkable
class Sink(Capabilities, Protocol):
    """The graph-sink port: :class:`Capabilities` + IR -> runnable.

    Adapter port.  ``catopt_torch.adapters.TorchSink`` is the canonical
    implementation.  On top of the :class:`Capabilities` op surface:

    * ``lower`` materialises a runnable :class:`Executor` from an IR and
      its leaf values; ``verify`` runs a reference and an optimized
      executable on the same inputs and returns the equivalence report
      in the backend's own runtime.

    ``verify`` is module-level — ``ref`` / ``opt`` are runnables in the
    sink's runtime, not tensors; the torch implementation delegates to
    ``report.verify_module``.  Call sites pass ``rtol`` / ``atol`` by
    name, matching :class:`Verifier`.
    """

    def lower(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> Executor:
        """Materialise a runnable executor from ``ir`` and leaves."""
        ...

    def verify(
        self,
        ref: Any,
        opt: Any,
        inputs: Any,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> VerifyResult:
        """Run ``ref`` and ``opt`` on ``inputs``; return the report."""
        ...

    @property
    def executors(self) -> Mapping[str, ExecutorSpec]:
        """Return the backend's executor table (may be empty).

        The named :class:`ExecutorSpec`s the orchestrator's routing and
        the regime planner consume.  Mapping order is routing order —
        the pipeline routes a term to the first *carrier* entry
        (``carrier`` not ``None``) whose ``accepts`` probe holds, and
        falls back to :meth:`lower` otherwise; carrier executors
        therefore precede executor-agnostic ones.  A backend without
        executor families returns an empty mapping.
        """
        ...


@runtime_checkable
class Composer(Protocol):
    """The structural port — module-tree composition machinery.

    Adapter port (plan 0007).  Everything the per-block
    (:class:`Compositional`) strategy needs that is backend-native:
    which sub-objects count as blocks, how their real inputs are
    captured, how optimized replacements are grafted back, and how a
    structure-preserving clone shares the original's parameter
    storage.  ``catopt_torch.composer.TorchComposer`` is the canonical
    implementation — module-object specifics stay on the adapter side.

    * ``blocks(model, *, predicate=None)`` — pick the top-most
      sub-objects to optimize independently, in execution order,
      ``[(dotted_name, block)]``.
    * ``capture_inputs(model, blocks, example_input)`` — run the
      original once; return ``(captured, io)`` where ``captured``
      maps each name to its first call's ``(args, kwargs)`` clones and
      ``io`` carries the object-identity dataflow evidence the
      boundary classifier reads.
    * ``graft(model, replacements)`` — install each
      ``{dotted_name: optimized_module}``; returns the model.
    * ``clone_sharing(model)`` — a structure clone that aliases the
      original's parameter storage (grafting must never mutate the
      caller's model).
    * ``boundary(name_a, name_b, captured, io, captured2, io2)`` —
      classify the A→B dataflow of an adjacent pair: a joint-mode
      string or ``None`` (not a simple value flow).
    * ``perturbed(example_input)`` — a second probe input, same
      structure, different values (the residual-boundary
      confirmation pass).

    Optional adapter hooks (read via ``getattr``, not port members):
    ``cross_pairs(...)`` — the pairwise joint-optimization pass;
    ``param_report(model, optimized)`` — the weight-file diff.  A
    composer without them simply skips those phases.
    """

    def blocks(self, model: Any, *, predicate: Any = None) -> list[Any]:
        """Select ``[(name, block)]`` optimization units."""
        ...

    def capture_inputs(
        self, model: Any, blocks: list[Any], example_input: Any
    ) -> Any:
        """Run ``model`` once; return ``(captured, io)``."""
        ...

    def graft(self, model: Any, replacements: Mapping) -> Any:
        """Install ``{name: optimized}`` replacements; return model."""
        ...

    def clone_sharing(self, model: Any) -> Any:
        """Clone structure, sharing the original's parameter storage."""
        ...

    def boundary(
        self,
        name_a: str,
        name_b: str,
        captured: Mapping,
        io: Mapping,
        captured2: Mapping,
        io2: Mapping,
    ) -> str | None:
        """Classify the pair boundary; ``None`` declines the pair."""
        ...

    def perturbed(self, example_input: Any) -> Any:
        """Return a same-structure, different-values probe input."""
        ...


@dataclass(frozen=True)
class TimingResult:
    """One :class:`Meter` measurement — median wall time + spread.

    ``median_s`` is the median seconds of one forward;
    ``iqr_s`` the interquartile spread (0 when fewer than 4 samples);
    ``n_calls`` the number of timed calls the measurement ran.
    """

    median_s: float
    iqr_s: float
    n_calls: int


@runtime_checkable
class Meter(Protocol):
    """The timing port — ``time(runnable, inputs) -> TimingResult``.

    Adapter port (plan 0007): the :class:`Autotuned` strategy measures
    candidates through it instead of running torch-side timing loops.
    ``warmup`` untimed calls first, then ``n_calls`` timed forwards;
    the median decides.  Device synchronisation is the adapter's
    business (``torch.cuda.synchronize`` on CUDA inputs).
    """

    def time(
        self,
        runnable: Any,
        inputs: Any,
        *,
        warmup: int = 5,
        n_calls: int = 30,
    ) -> TimingResult:
        """Time ``runnable(*inputs)``; return median + IQR seconds."""
        ...


@runtime_checkable
class Runner(Protocol):
    """The delivery-stage transform port.

    ``apply`` receives the module the lowering produced plus the
    pipeline's ``example_input`` (tensor or positional-args tuple)
    and returns the module to deliver — possibly a wrapped or
    mutated version of the input.  ``stats`` is the same dict the
    pipeline returns, so a runner records what it did
    (``stats["compiled"]``, ``stats["cuda_graph"]``) and may read
    what earlier runners in a chain did.  Optional marker
    ``delivers_compiled`` hints the search's carrier pricing.
    """

    name: str

    def apply(
        self,
        module: Any,
        example_input: Any,
        stats: dict[str, Any],
    ) -> Any:
        """Return the module to deliver, recording into *stats*."""
        ...


@runtime_checkable
class Criterion(Protocol):
    """One selection axis: a named recipe for a calibrated ``CostFn``.

    Members
    -------
    ``name`` — the axis label; recorded into ``stats["criteria"]``
    and the blend's ``criteria`` dict (same-named members merge).

    ``cost_fn(profile=None) -> CostFn`` — build the axis's pricing
    callable, calibrated to *profile* (a
    ``catopt_core.profile.TargetProfile``-like object/dict, or
    ``None`` for the built-in profile).  The callable follows the
    ``(term, memo=None)`` convention; a member that does not declare
    ``memo`` is called bare.

    Optional markers (read with ``getattr`` defaults, propagated to
    the built callable and aggregated over a blend):

    * ``charges_param_only`` — the axis bills compile-time-foldable
      subtrees (storage-style pricing: a folded subtree still stores
      values).  Extraction reads the marker OFF THE BUILT COST FN to
      keep billing them.
    * ``charges_shape`` — the axis's prices are shape-dependent.
      Informational: blends aggregate the flag so reporters can see
      when a blend cares about inferred shapes.

    Duck-typed in use: ``criteria_cost`` accepts any object with
    a callable ``cost_fn`` member (a missing ``name`` falls back to
    the class name); ``isinstance``-conformance additionally needs
    the ``name`` attribute.
    """

    @property
    def name(self) -> str:
        """The axis label recorded into ``stats["criteria"]``."""
        ...

    def cost_fn(self, profile: Any = None) -> CostFn:
        """Build the axis's pricing callable for *profile*."""
        ...


@runtime_checkable
class Strategy(Protocol):
    """One optimization policy — how ``search`` and ``lower`` compose.

    The seam behind ``Optimizer.optimize(..., strategy=...)`` (plan
    0006): a strategy object receives the model, the example input and
    the owning optimizer — through which it reaches the configured
    ports (``optimizer.source`` / ``optimizer.sink``) and defaults
    (``optimizer.criteria`` / ``optimizer.runner``) — and returns a
    :class:`~catopt_core.pipeline.LowerResult`.

    ``optimizer`` is typed ``Any``: core names the contract, not the
    orchestrator's class — ``catopt_core`` may not import
    ``catopt_orchestrator`` (the hexagonal boundary).  The conforming
    implementations live there: ``Monolithic`` (the default —
    ``lower ∘ search``), ``Compositional`` (per-block), ``Autotuned``
    (one search, N timed deliveries).  ``run`` is keyword-flexible:
    ``**kw`` forwards the caller's phase knobs (``rules``,
    ``max_iterations``, ``verify``, …) and each strategy decides how
    they partition.
    """

    name: str

    def run(
        self, model: Any, x: Any, *, optimizer: Any, **kw: Any
    ) -> LowerResult:
        """Run the policy end to end; return the lower record."""
        ...


# ---------------------------------------------------------------------------
#  Laws as data — the rewrite surface
# ---------------------------------------------------------------------------


@runtime_checkable
class RuleLike(Protocol):
    """The rewrite-rule read surface.

    Structural twin of ``egraph.Rewrite``.

    Every member is read unconditionally somewhere in the e-graph:
    ``name`` (fire counts, budgets, ``_applied_rules``), ``lhs`` /
    ``rhs`` (matching/instantiation), ``check`` / ``derive`` (side
    conditions and computed attrs — ``None`` when absent), ``law``
    (provenance text) and ``error_bound`` / ``bound_norm`` (the
    certified-approximation axis — read unconditionally in
    ``extract_best_bounded``/certificates, so part of the contract,
    not optional).
    """

    name: str
    lhs: Any
    rhs: Any
    law: str
    check: Any  # Callable[[dict[str, Any]], bool] | None
    derive: Any  # Callable[[dict], dict | None] | None
    error_bound: float | None
    bound_norm: str


@runtime_checkable
class RuleSetLike(Protocol):
    """An iterable of :class:`RuleLike` rewrites.

    The surface ``EGraph.run(rules, ...)`` consumes — the renamed
    ``LawSet`` (plan 0009's vocabulary: a *rule* is one ``Rewrite``;
    a *rule set* is the collection).

    ``list[Rewrite]`` collections conform (``ALL_RULES`` /
    ``all_rules()``, ``SIMPLIFICATION_RULES``, ``CATEGORICAL_RULES``,
    ``SDPA_FOLD_RULES``, ``OM_LAWS``), and so does the concrete
    :class:`catopt_core.laws.RuleSet`.  (Runtime check is
    presence-level: any iterable ``isinstance``s; element conformance
    is the static-typing half.)
    """

    def __iter__(self) -> Iterator[RuleLike]:
        """Iterate over the rules in the set."""
        ...


@runtime_checkable
class RuleSetProvider(Protocol):
    """A zero-argument source of a :class:`RuleSetLike`.

    This is the ``all_rules()`` shape — the renamed ``RuleProvider``.
    How a rule-set *name* maps to rules is a preset detail
    (:func:`catopt_core.laws.preset`), not part of this port.
    """

    def __call__(self) -> RuleSetLike:
        """Return the rule set."""
        ...


# ---------------------------------------------------------------------------
#  The engine seam — which saturation core runs the search
# ---------------------------------------------------------------------------


@runtime_checkable
class Engine(Protocol):
    """A saturation engine: the search core behind ``search``.

    The port (plan 0010, lever 3) names the surface
    ``catopt_orchestrator.optimize.search`` consumes, so the pure-Python
    :class:`~catopt_core.egraph.EGraph` (the reference implementation
    and the default) and the optional ``catopt_native.NativeEngine``
    are interchangeable values:

    * ``add_term(term) -> eid`` — intern a program's DAG; each call
      registers every enode and returns the root's e-class.
    * ``find(eid) -> canonical eid`` — union-find lookup.
    * ``run(rules, root_eid, max_iterations, max_nodes, rule_budgets,
      stop, patience, cost_fn) -> stats`` — equality saturation under
      the given schedule; the stats dict carries ``iterations``,
      ``n_enodes``, ``n_classes``, ``rule_budgets``,
      ``budget_suspended``, ``stop``.
    * ``extract_best(eid, cost_fn, **kw) -> term`` — greedy
      minimum-cost member extraction (``overrides`` / ``bans`` /
      ``fusion_epsilon`` ride in ``**kw`` like ``EGraph``'s signature).
    * ``rebuild(classes=None) -> bool`` — canonicalise (+congruence
      on an unrestricted pass).
    * ``rule_fires``, ``n_enodes``, ``n_classes`` — the run record.

    Engines are **never auto-detected**: ``search``/``Optimizer`` take
    ``engine=`` explicitly and ``stats["engine"]`` records which ran
    (``"python"`` by convention when the object does not declare an
    ``engine_name`` — ``EGraph`` is the reference, so it needs no
    marker; ``NativeEngine.engine_name == "native"``).

    The scope boundary is part of the contract: the port covers the
    *search* only.  Proof machinery (merge logs, applications,
    certificates) and the non-local pairing/lift passes are
    Python-engine capabilities — a conforming engine is not required
    to provide them, and the pipeline falls back to skipping the
    non-local passes when the engine is not an ``EGraph``.
    """

    def add_term(
        self,
        term: Any,
        _memo: dict | None = None,
        provenance: str = "input",
    ) -> int:
        """Intern *term*; return the root's e-class id."""
        ...

    def find(self, eid: int) -> int:
        """Return the canonical e-class id of ``eid``."""
        ...

    def rebuild(self, classes: Any = None) -> bool:
        """Canonicalise children; close congruence when unrestricted."""
        ...

    def run(
        self,
        rules: RuleSetLike,
        root_eid: int,
        max_iterations: int = 100,
        max_nodes: int = 100_000,
        rule_budgets: dict[str, int] | None = None,
        stop: str = "fixed_point",
        patience: int = 3,
        cost_fn: CostFn | None = None,
    ) -> dict[str, Any]:
        """Saturate under *rules*; return the run record stats."""
        ...

    def extract_best(self, eid: int, cost_fn: CostFn, **kw: Any) -> Any:
        """Return the minimum-cost member of the e-class at *eid*."""
        ...

    @property
    def n_enodes(self) -> int:
        """Return the number of enodes interned."""
        ...

    @property
    def n_classes(self) -> int:
        """Return the number of e-classes."""
        ...

    @property
    def rule_fires(self) -> dict[str, int]:
        """Return ``{rule_name: merge_count}`` from the last run."""
        ...


# ---------------------------------------------------------------------------
#  Signature-level conformance — the companion to isinstance
# ---------------------------------------------------------------------------

_PROBE = object()


def _port_signature(proto: type) -> inspect.Signature | None:
    """Return the port's ``__call__`` signature, minus ``self``."""
    for klass in getattr(proto, "__mro__", (proto,)):
        fn = klass.__dict__.get("__call__")
        if fn is None:
            continue
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return None
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        return sig.replace(parameters=params)
    return None


def _probes(
    sig: inspect.Signature,
) -> tuple[list, dict, list, dict]:
    """Full and minimal probe args/kwargs for a port signature.

    Required positional params are probed positionally; every defaulted
    or keyword-only param is probed *by name* — the real call sites
    pass them as keywords (``cost_fn(t, memo=memo)``,
    ``verify(out, rtol=..., atol=...)``).  The minimal probe carries
    only what the port requires.
    """
    full_args: list = []
    full_kwargs: dict = {}
    min_args: list = []
    min_kwargs: dict = {}
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        required = p.default is inspect.Parameter.empty
        if p.kind is inspect.Parameter.KEYWORD_ONLY:
            full_kwargs[p.name] = _PROBE
            if required:
                min_kwargs[p.name] = _PROBE
        elif required:
            full_args.append(_PROBE)
            min_args.append(_PROBE)
        else:
            full_kwargs[p.name] = _PROBE
    return full_args, full_kwargs, min_args, min_kwargs


def signature_conforms(
    fn: Any, proto: type, *, strict: bool = False
) -> bool:
    """Signature-level conformance for the callable ports.

    ``isinstance(fn, proto)`` on a ``@runtime_checkable`` protocol only
    proves ``fn`` has ``__call__`` — any function passes.  This helper
    additionally checks ``fn`` can be invoked the way the port's call
    sites invoke it: it builds probe arguments from the port's declared
    ``__call__`` and ``inspect.signature``-binds them against ``fn``.

    * ``strict=False`` (default) — ``fn`` conforms when it binds EITHER
      the full documented call OR the minimal call (required params
      only).  This mirrors the introspection-adaptive call sites:
      ``extract_best`` calls ``cost_fn(t)`` or ``cost_fn(t, memo=...)``
      depending on whether the fn declares ``memo``, so both
      ``flops_cost`` and a ``fn(term)``-only callable (e.g. a
      ``CostModel`` instance) conform to :class:`CostFn`.
    * ``strict=True`` — ``fn`` must bind the full call.  Use it where
      the port's call sites always pass the defaulted params by name
      (:class:`Verifier`'s ``rtol``/``atol``).

    Uninspectable callables (C-level functions without signatures)
    pass — presence-level conformance is all that can be proven.  A
    port whose ``__call__`` is variadic-only (:class:`Binding`)
    probes empty: arity is per-op there, so any callable conforms.
    """
    if not callable(fn):
        return False
    sig = _port_signature(proto)
    if sig is None:
        return True
    try:
        fn_sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    full_args, full_kwargs, min_args, min_kwargs = _probes(sig)
    try:
        fn_sig.bind(*full_args, **full_kwargs)
        return True
    except TypeError:
        pass
    if strict:
        return False
    try:
        fn_sig.bind(*min_args, **min_kwargs)
    except TypeError:
        return False
    return True
