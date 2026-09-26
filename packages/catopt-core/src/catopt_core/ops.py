"""Explicit op-table composition — plan 0001 phase 2c.

Before this phase the torch-lowering table lived as module-global
mutable state: ``catopt_torch.torch_bridge._IR_TO_TORCH`` was seeded with the
core bindings and every carrier module (``catopt_carriers.trace``,
``catopt_carriers.xcarrier``, ``catopt_carriers.om``, ``catopt_eps.act_eps``) mutated it at
import time.  Whether a term could be lowered therefore depended on
WHICH modules had happened to be imported — an invisible dependency.

:class:`OpTable` replaces that with explicit composition:

* :meth:`OpTable.core` — the base table: the core torch bindings plus
  the ambient shape rules (``catopt_core.typing._SHAPE_RULES``) and the
  attr schema (``catopt_core.attrs.ATTR_SCHEMA``).
* :meth:`OpTable.register` — folds one extension module's declared
  exports in: ``TORCH_BINDINGS`` / ``SHAPE_RULES`` / ``ATTR_SCHEMA``
  dicts the module publishes at top level (importing the module no
  longer mutates anything).
* :meth:`OpTable.full` — ``core()`` plus every carrier module —
  preserves today's ambient behavior and is the default
  ``IRModule``/``optimize_model`` dispatches through.

Carrier modules that opt in (and the ops they contribute):

* :mod:`catopt_carriers.trace` — ``trace`` ``bdiag`` ``parl`` ``eye`` ``cswap``
  ``inv``
* :mod:`catopt_carriers.xcarrier` — ``affd_a`` ``affd_b`` ``aff_A`` ``aff_b``
  ``om_elem_affd`` ``om_elem_aff`` ``omd`` ``omd_elem`` ``omd_compose``
  ``omd_apply`` ``omd_applym``
* :mod:`catopt_carriers.om` — ``cmask`` ``fill`` ``attnbias``
* :mod:`catopt_eps.act_eps` — ``aquant`` ``adequant``

The aff/om/affd carrier PRIMITIVES (``aff`` ``apply`` ``om_elem`` ...)
are core ops — they ship inside ``torch_bridge._CORE_TORCH_BINDINGS``.

Back-compat
-----------
``torch_bridge._IR_TO_TORCH`` survives as the *ambient* table: a dict
whose reads lazily resolve through :func:`carrier_torch_bindings` so
legacy ``_IR_TO_TORCH[op]`` / ``.get`` / ``in`` consumers keep working
without importing carriers for side effects.  ``OpTable.full()`` seats
its ``torch_bindings`` on that same dict, so a post-construction
``_IR_TO_TORCH[op] = fn`` override still reaches an already-built
``IRModule`` — the pre-2c dispatch semantics, preserved deliberately.
A custom ``OpTable`` owns a private plain dict: deleting a binding
there makes the missing op fail loudly at eval ("No torch binding"),
it can never silently re-resolve through the ambient table.

Ports layer
-----------
:class:`OpTable` IS the adapter-registry of the hexagonal boundary in
:mod:`catopt_core.ports` — it structurally conforms to
:class:`catopt_core.ports.OpRegistry` (``torch_bindings`` /
``shape_rules`` / ``attr_schemas`` / ``register``); the protocol names
the surface, nothing is re-wrapped.  Its dicts hold
:class:`~catopt_core.ports.TorchBinding` and
:class:`~catopt_core.ports.ShapeRule` values.
"""

from __future__ import annotations

import importlib
from types import ModuleType, SimpleNamespace
from typing import Any

from catopt_core.ports import ShapeRule, TorchBinding

__all__ = ["OpTable", "carrier_torch_bindings"]

#: Extension modules folded into :meth:`OpTable.full`, in
#: registration order.  Each declares a module-level
#: ``TORCH_BINDINGS: dict[str, Callable]`` (and may declare
#: ``SHAPE_RULES`` / ``ATTR_SCHEMA`` fragments).
_CARRIER_MODULES: tuple[str, ...] = (
    "catopt_carriers.trace",
    "catopt_carriers.xcarrier",
    "catopt_carriers.om",
    "catopt_eps.act_eps",
)


def _carrier_module_objects() -> list[ModuleType]:
    return [importlib.import_module(name) for name in _CARRIER_MODULES]


def carrier_torch_bindings() -> dict[str, TorchBinding]:
    """Every carrier module's ``TORCH_BINDINGS`` merged into one dict.

    Rebuilt per call — later carriers win on a name collision and a
    module's post-import edits to its ``TORCH_BINDINGS`` are honored.
    Importing the carriers here is safe: they export dicts now and
    mutate no shared registry.
    """
    out: dict[str, TorchBinding] = {}
    for mod in _carrier_module_objects():
        out.update(getattr(mod, "TORCH_BINDINGS", None) or {})
    return out


class OpTable:
    """An explicit registry of op semantics.

    The adapter registry of the ports layer — conforms to
    :class:`catopt_core.ports.OpRegistry`.

    Attributes
    ----------
    torch_bindings : dict[str, TorchBinding]
        ``op_name -> lowering fn`` — the table ``IRModule._eval``
        dispatches through.  For :meth:`full` this IS the ambient
        ``torch_bridge._IR_TO_TORCH`` dict (shared, live); for
        ``core()``/custom tables it is a private plain dict.
    shape_rules : dict[str, ShapeRule]
        ``op_name -> fn(op, shapes) -> shape`` — mirrors
        ``catopt_core.typing._SHAPE_RULES``.  Shape inference still
        dispatches through the ambient registry today (threading a
        table into ``_shape_of`` is a later phase); the table carries
        the fragment so composition is inspectable and carrier
        ``SHAPE_RULES`` exports have a declared home.
    attr_schemas : dict[str, dict[int, str]]
        ``op_name -> {position: canonical attr name}`` — mirrors
        ``catopt_core.attrs.ATTR_SCHEMA``.
    """

    def __init__(self) -> None:
        self.torch_bindings: dict[str, TorchBinding] = {}
        self.shape_rules: dict[str, ShapeRule] = {}
        self.attr_schemas: dict[str, dict[int, str]] = {}

    # -- constructors -------------------------------------------------

    @classmethod
    def core(cls) -> OpTable:
        """The base table: core torch bindings + shape rules + attr
        schema.  No carrier ops — ``trace``/``omd_*``/``cmask``/
        ``aquant`` are absent until a carrier module is
        :meth:`register`\\ ed."""
        from catopt_torch.torch_bridge import _CORE_TORCH_BINDINGS

        from catopt_core.attrs import ATTR_SCHEMA
        from catopt_core.typing import _SHAPE_RULES

        t = cls()
        t.torch_bindings.update(_CORE_TORCH_BINDINGS)
        t.shape_rules.update(_SHAPE_RULES)
        t.attr_schemas.update(ATTR_SCHEMA)
        return t

    @classmethod
    def full(cls) -> OpTable:
        """``core()`` plus every carrier module — today's ambient
        behavior, as an explicit object.

        The returned table's ``torch_bindings`` is the ambient
        ``catopt_torch.torch_bridge._IR_TO_TORCH`` dict itself: post-hoc
        overrides (``_IR_TO_TORCH[op] = fn``) and lazy carrier
        resolution keep reaching the evaluators that dispatch through
        this table — exactly what the old global registry did.
        """
        import catopt_torch.torch_bridge as tb

        t = cls.core()
        for mod in _carrier_module_objects():
            t.register(mod)
        ambient = tb._IR_TO_TORCH
        for name, fn in t.torch_bindings.items():
            # setdefault: a live user override already in the ambient
            # dict beats the canonical binding.
            dict.setdefault(ambient, name, fn)
        t.torch_bindings = ambient
        return t

    # -- composition ---------------------------------------------------

    @staticmethod
    def _coerce_source(source: Any) -> Any:
        """Normalize a ``register`` argument to an object exposing
        ``TORCH_BINDINGS``/``SHAPE_RULES``/``ATTR_SCHEMA`` attributes.

        Accepts a module object, a module name (``"catopt_carriers.trace"`` or
        the bare ``"trace"``), or a plain dict of torch bindings.
        """
        if isinstance(source, dict):
            return SimpleNamespace(TORCH_BINDINGS=source)
        if isinstance(source, str):
            name = source if "." in source else f"catopt_carriers.{source}"
            return importlib.import_module(name)
        return source

    def register(
        self,
        source: Any = None,
        *,
        torch_bindings: dict | None = None,
        shape_rules: dict | None = None,
        attr_schema: dict | None = None,
    ) -> OpTable:
        """Fold a module's declared exports into this table.

        ``source`` may be a module (or any object) exposing optional
        ``TORCH_BINDINGS`` / ``SHAPE_RULES`` / ``ATTR_SCHEMA``
        attributes, a module name, or a plain dict of torch bindings.
        Fragments may also be passed directly via keyword.  Returns
        ``self`` for chaining::

            table = OpTable.core().register("trace").register(
                {"my_op": my_torch_fn}
            )
        """
        if source is not None:
            mod = self._coerce_source(source)
            for fragment, target in (
                ("TORCH_BINDINGS", self.torch_bindings),
                ("SHAPE_RULES", self.shape_rules),
                ("ATTR_SCHEMA", self.attr_schemas),
            ):
                part = getattr(mod, fragment, None)
                if part:
                    target.update(part)
        if torch_bindings:
            self.torch_bindings.update(torch_bindings)
        if shape_rules:
            self.shape_rules.update(shape_rules)
        if attr_schema:
            self.attr_schemas.update(attr_schema)
        return self

    def copy(self) -> OpTable:
        """A snapshot copy — private plain dicts, detached from the
        ambient table.  Deleting a binding from the copy makes the op
        fail loudly at eval; it cannot leak back into ``full()``."""
        t = OpTable()
        t.torch_bindings = dict(self.torch_bindings)
        t.shape_rules = dict(self.shape_rules)
        t.attr_schemas = {
            op: dict(schema) for op, schema in self.attr_schemas.items()
        }
        return t
