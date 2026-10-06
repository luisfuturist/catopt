"""Op definitions as data — the declaration seam (plan 0019).

``catopt_core.opmeta`` made an op's *metadata* data (one classification
table, named projections).  The op *set* and each op's *shape
semantics* were still code: a new primitive needed a Python shape
rule, a torch binding, a lowering.  This module is the seam for the
part of that which is genuinely declarable — the **structural slice**
of a view / relayout-class op:

* its *classification* (tags, arity) — already data in
  :mod:`catopt_core.opmeta`;
* its *attr schema* — a positional ``{argN: canonical-name}`` table,
  the shape ``catopt_core.attrs.ATTR_SCHEMA`` already takes;
* its *shape rule* — a spec in the **existing** shape-spec DSL
  (:mod:`catopt_core.laws.cond`), re-bound from law-pattern
  metavariables to *operand positions* (``"arg0"``/``"arg1"``) and
  *attr names* (``"dim"``).  A shape-preserving view is the spec
  ``"arg0"``; ``view_as`` (output = operand 1's shape) is ``"arg1"``;
  ``swapaxes`` is ``("transpose-out", "arg0", "dim0", "dim1")``.

:func:`declare_op` registers a declaration into the live registry
(:func:`catopt_core.opmeta.register_op_meta`), the attr schema
(``ATTR_SCHEMA``/``ATTR_REQUIRED``) and the shape-rule registry
(:func:`catopt_core.typing.register_shape_rule`), so a declared op
shapes, mints and matches through the same machinery a shipped op
does.  :func:`reset_declarations` undoes it — the seam is additive.

What is deliberately *not* declarable here (see
``project/plans/0019-ops-as-data.md``): the kernel body (a hand-written
numeric lowering), the torch/aten spelling (the adapter's
``_ATEN_TO_IR``, validated at the boundary, never moved into core) and
the cost entry (the evaluation dimension's own table).  Those stay
code — the deliverable is the *declaration* seam for the structural
slice, not a general "define any op in JSON".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from catopt_core import opmeta
from catopt_core.ir import Op
from catopt_core.laws.cond import _shape
from catopt_core.typing import _SHAPE_RULES, register_shape_rule

__all__ = [
    "OpDef",
    "declare_op",
    "opdef_from_data",
    "opdef_to_data",
    "reset_declarations",
]


@dataclass(frozen=True)
class OpDef:
    """A declared op — the structural slice of a primitive, as data.

    Attributes
    ----------
    name : str
        The canonical IR spelling — a *new* name; a declaration cannot
        shadow a shipped op (:func:`catopt_core.opmeta.register_op_meta`).
    arity : int | None
        Operand count where it is a defining property; ``None`` when
        undeclared (the registry does not guess).
    tags : frozenset[str]
        The classification — a subset of :mod:`catopt_core.opmeta`'s
        tag vocabulary (``RELAYOUT``, ``VIEWISH``, …).
    attrs : Mapping[int, str]
        The positional attr schema — ``{argN: canonical name}``, the
        shape ``catopt_core.attrs.ATTR_SCHEMA`` takes.  ``Op.make``
        enforces it once the op is declared.
    required : frozenset[str]
        Canonical attrs a *fully-attributed* term must supply.
    shape : Any
        A shape spec in the :mod:`catopt_core.laws.cond` DSL, re-bound
        to operand positions (``"arg0"``) and attr names (``"dim"``);
        ``None`` for an op the default rule already shapes (the first
        operand's).

    """

    name: str
    arity: int | None = None
    tags: frozenset[str] = frozenset()
    attrs: Mapping[int, str] = field(default_factory=dict)
    required: frozenset[str] = frozenset()
    shape: Any = None


def _bound(op: Op) -> dict:
    """Bind an op's operands and attrs for the shape-spec DSL.

    The DSL's metavariable names are re-bound here: operand ``i`` is
    ``"arg{i}"`` and every attr is ``"$attr:{name}"`` — the same
    ``bound`` environment :func:`catopt_core.laws.cond.eval_cond`
    builds, so a shape spec is *the same spec* a law guard uses.
    """
    b: dict = {f"arg{i}": a for i, a in enumerate(op.args)}
    for k, v in op.attrs.items():
        b[f"$attr:{k}"] = v
    return b


def _compile_shape(spec: Any):
    """Compile a shape spec into a ``typing`` shape rule.

    The rule re-binds the op's operands/attrs (:func:`_bound`) and
    delegates to the cond DSL's resolver — so a declared op's shape
    inherits the DSL's strictness (an unprovable spec declines to
    ``None``, never a fabricated shape).
    """

    def rule(op: Op, shapes: list) -> Any:
        return _shape(_bound(op), spec)

    return rule


def declare_op(defn: OpDef | Mapping[str, Any]) -> OpDef:
    """Register an op declaration; return the normalized :class:`OpDef`.

    Accepts an :class:`OpDef` or its JSON-ish data form
    (:func:`opdef_from_data`) — a declaration *is* data the way a
    stored law is.  Registration is additive and process-global:

    * the op joins the live registry
      (:func:`catopt_core.opmeta.register_op_meta`) — ``meta`` /
      ``ops`` / ``tags_of`` / ``arity`` / ``attrs`` / ``required`` all
      see it;
    * its positional attr schema joins ``ATTR_SCHEMA`` /
      ``ATTR_REQUIRED``, so :meth:`catopt_core.ir.Op.make` enforces it
      at mint;
    * its shape spec compiles into ``typing._SHAPE_RULES`` (when one is
      given), so :func:`catopt_core.typing.shape_of` answers it.

    Declaring a name a shipped op already uses is a ``ValueError`` — a
    declaration adds a primitive, it does not shadow one.
    """
    d = defn if isinstance(defn, OpDef) else opdef_from_data(defn)
    opmeta.register_op_meta(
        opmeta.OpMeta(d.name, d.arity, d.tags),
        attrs=d.attrs,
        required=d.required,
    )
    if d.shape is not None:
        register_shape_rule(d.name, _compile_shape(d.shape))
    return d


def reset_declarations() -> list[str]:
    """Remove every op declared through this seam; return the names.

    Restores the registry, the attr schema and the shape-rule table to
    their shipped state.  The seam is process-global, so a test (or a
    re-composition) undoes a declaration here rather than leaking it.
    """
    removed = opmeta.reset_declared()
    for name in removed:
        _SHAPE_RULES.pop(name, None)
    return removed


def _shape_to_data(v: Any) -> Any:
    """Encode a shape spec as JSON-safe data (tuples become lists)."""
    if isinstance(v, tuple):
        return [_shape_to_data(x) for x in v]
    return v


def _shape_from_data(v: Any) -> Any:
    """Decode a shape spec (lists become tuples) — the codec inverse."""
    if isinstance(v, list):
        return tuple(_shape_from_data(x) for x in v)
    return v


def opdef_to_data(d: OpDef) -> dict[str, Any]:
    """Encode an :class:`OpDef` as JSON-safe data.

    Tags/required sort; the attr schema is a ``[[argN, name], …]`` list
    (JSON keys are strings, so the index is carried as a value); the
    shape spec is nested lists.
    """
    return {
        "name": d.name,
        "arity": d.arity,
        "tags": sorted(d.tags),
        "attrs": [[int(i), n] for i, n in sorted(d.attrs.items())],
        "required": sorted(d.required),
        "shape": _shape_to_data(d.shape),
    }


def opdef_from_data(data: Mapping[str, Any]) -> OpDef:
    """Decode :func:`opdef_to_data` output back into an :class:`OpDef`."""
    return OpDef(
        name=str(data["name"]),
        arity=data.get("arity"),
        tags=frozenset(data.get("tags", ())),
        attrs={int(i): n for i, n in data.get("attrs", ())},
        required=frozenset(data.get("required", ())),
        shape=_shape_from_data(data.get("shape")),
    )
