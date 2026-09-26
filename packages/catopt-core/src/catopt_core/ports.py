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
* :class:`RuleLike` / :class:`LawSet` / :class:`RuleProvider` — laws as
  data: the structural read-surface of ``egraph.Rewrite`` and the
  collections ``EGraph.run`` consumes.
* :class:`ShapeRule` — the ``typing.register_shape_rule`` handler
  contract, ``fn(op, shapes) -> shape | _INVALID | None``.
* :class:`Executor` / :class:`PlannedExecutor` /
  :class:`BatchedExecutor` — the lowered-module contract.
* :class:`Verifier` — the semantic-equivalence gate,
  ``(ref_out, out, rtol, atol) -> VerifyResult``.
* :class:`VerifyResult` — the core-owned structural view of a verify
  result (``max_abs`` / ``max_rel`` / ``passed``); the torch adapter's
  ``report.VerifyReport`` satisfies it without core naming it.
* :class:`Source` — the whole-graph source port,
  ``model -> (IR, leaves)``.
* :class:`Sink` — the whole-graph sink port, ``IR -> runnable``, plus
  the backend's supported-op set and module-level equivalence gate.
* :class:`Binding` — one op's lowering, ``(*args, **attrs)``
  (``TorchBinding`` is the historical alias of the same protocol).
* :class:`OpRegistry` — the adapter-registry port.

Outside the hexagon (the adapters)
----------------------------------
The torch-facing implementations: ``torch_bridge._CORE_TORCH_BINDINGS``
and each carrier module's ``TORCH_BINDINGS`` entries are
``Binding``s; ``IRModule`` / ``BatchedScanModule`` /
``BatchedOMModule`` / ``StreamingOMModule`` / ``BatchedOmdModule`` are
``Executor``s; ``report.verify_equiv`` is the canonical ``Verifier``;
``catopt_torch.adapters.TorchSource`` / ``TorchSink`` are the canonical
``Source`` / ``Sink`` pair; each ``_SHAPE_RULES`` entry is a
``ShapeRule``.  ``ops.OpTable`` is the
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
* ``optimize_model`` selects law sets by *name* (``ruleset: str``);
  there is no ``rules`` parameter to annotate.  ``LawSet`` describes
  the list-of-``Rewrite`` surface ``EGraph.run`` consumes.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from catopt_core.ir import IR, Op

__all__ = [
    "BatchedExecutor",
    "Binding",
    "CostFn",
    "Executor",
    "LawSet",
    "OpRegistry",
    "PlannedExecutor",
    "RuleLike",
    "RuleProvider",
    "ShapeRule",
    "Sink",
    "Source",
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

    def __call__(self, *args: Any, **attrs: Any) -> Any: ...


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

    def forward(self, *xs: Any) -> Any: ...


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
    def is_batched(self) -> bool: ...


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
    ) -> OpRegistry: ...


# ---------------------------------------------------------------------------
#  Domain ports — what the core calls into
# ---------------------------------------------------------------------------


@runtime_checkable
class ShapeRule(Protocol):
    """The ``typing.register_shape_rule`` handler contract:
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

    def __call__(self, op: Op, shapes: list, /) -> tuple | str | None: ...


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

    def __call__(self, term: Any, memo: dict | None = None) -> float: ...


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
    """

    max_abs: float
    max_rel: float
    passed: bool


@runtime_checkable
class Verifier(Protocol):
    """The semantic-equivalence gate:
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
    ) -> VerifyResult: ...


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
    ) -> tuple[IR, dict[str, Any]]: ...


@runtime_checkable
class Sink(Protocol):
    """The graph-sink port: IR -> runnable, backend-relative.

    Adapter port.  ``catopt_torch.adapters.TorchSink`` is the canonical
    implementation.  Three responsibilities, all backend-owned:

    * ``supported_ops`` bounds the reachable equivalence class.  It is
      the set of op names this backend can lower; extraction prices any
      member that uses an op outside it at ``+inf`` (see
      :func:`catopt_core.cost.backend_cost`), so the search never
      commits to a form the sink cannot execute — the backend
      counterpart of the semantic-language bound.  A sink without
      ``sdpa`` simply never selects the attention fold.
    * ``ops`` is the lowering registry (an :class:`OpRegistry`) the
      compile-time const folds and causal specialization dispatch
      through.
    * ``lower`` materialises a runnable :class:`Executor` from an IR and
      its leaf values; ``verify`` runs a reference and an optimized
      executable on the same inputs and returns the equivalence report
      in the backend's own runtime.

    ``verify`` is module-level — ``ref`` / ``opt`` are runnables in the
    sink's runtime, not tensors; the torch implementation delegates to
    ``report.verify_module``.  Call sites pass ``rtol`` / ``atol`` by
    name, matching :class:`Verifier`.
    """

    @property
    def supported_ops(self) -> frozenset[str]: ...

    @property
    def ops(self) -> OpRegistry: ...

    def lower(
        self, ir: IR, params: dict[str, Any] | None = None
    ) -> Executor: ...

    def verify(
        self,
        ref: Any,
        opt: Any,
        inputs: Any,
        *,
        rtol: float = 1e-4,
        atol: float | None = None,
    ) -> VerifyResult: ...


# ---------------------------------------------------------------------------
#  Laws as data — the rewrite surface
# ---------------------------------------------------------------------------


@runtime_checkable
class RuleLike(Protocol):
    """The rewrite-rule read surface — structural twin of
    ``egraph.Rewrite``.

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
class LawSet(Protocol):
    """An iterable of :class:`RuleLike` rewrites — the surface
    ``EGraph.run(rules, ...)`` consumes.

    ``list[Rewrite]`` collections conform: ``ALL_RULES`` /
    ``all_rules()``, ``SIMPLIFICATION_RULES``, ``CATEGORICAL_RULES``,
    ``SDPA_FOLD_RULES``, ``OM_LAWS``, and the filtered per-ruleset
    lists ``optimize_model`` builds.  (Runtime check is presence-level:
    any iterable ``isinstance``s; element conformance is the
    static-typing half.)
    """

    def __iter__(self) -> Iterator[RuleLike]: ...


@runtime_checkable
class RuleProvider(Protocol):
    """A zero-argument source of a :class:`LawSet` — the
    ``all_rules()`` shape.  How a ruleset *name* maps to rules is a
    pipeline detail (``optimize_model``'s ``ruleset`` dict), not part
    of this port."""

    def __call__(self) -> LawSet: ...


# ---------------------------------------------------------------------------
#  Signature-level conformance — the companion to isinstance
# ---------------------------------------------------------------------------

_PROBE = object()


def _port_signature(proto: type) -> inspect.Signature | None:
    """The signature of the port's ``__call__``, minus ``self``."""
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
