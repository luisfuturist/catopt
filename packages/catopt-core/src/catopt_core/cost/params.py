"""Parameter-storage cost model — the memory axis's pricing side.

Prices the stored scalars a lowered module keeps (the axis the
FLOP-based models cannot see) and the fold predicate
(:func:`_folds_to_param`) that mirrors the lowerer's compile-time
weight folding.  Also hosts :func:`dag_cost`, the true-DAG wrapper
whose contract is defined by this model's ``charges_param_only`` /
``dag_exact`` markers, and :data:`_FUSION_POINTWISE_OPS`, the
pointwise op union the roofline kernel table and the fusion model
both read.
"""
# ruff: noqa: RUF002, RUF003 — math notation in comments

from __future__ import annotations

from typing import Any, cast

from catopt_core.ir import Const, Op, Param
from catopt_core.typing import _numel, _shape_of, has_var_leaf

from .basic import _CostMarkers, _memo_dispatch

# ---------------------------------------------------------------------------
#  Parameter-storage cost model — the memory axis's pricing side
# ---------------------------------------------------------------------------


def param_bytes_cost(
    term: Any,
    source_tensors: dict | None = None,
    memo: dict | None = None,
    by_bytes: bool = False,
) -> float:
    """Cost = stored parameter values — what the LOWERED module keeps.

    The storage axis the flop-based models cannot see: a shared
    (tied) weight or a deduplicated head-stack computes the same
    function from fewer stored scalars.  The unit is *values* (bytes
    at unit width — multiply by dtype size for true bytes); derived
    params registered by the sharing passes count normally, which is
    what lets extraction prefer the storage-cheaper member.

    Pricing mirrors ``IRModule._fold_weight_chains`` /
    ``_build_params`` (catopt_torch.torch_bridge), not the term's leaf list:

    * a ``Param`` leaf reachable after folding is one storage entry,
      deduplicated *by name* — a weight read by two consumers is
      stored once (two ``Param`` objects spelled identically are the
      same leaf in the e-graph anyway: ``repr(Param)`` is the name);
    * a param-only subtree the lowerer materialises (``matmul`` over
      stored weights, elementwise ops, ``concat`` — see
      ``_folds_to_param``) is billed at the fold's OUTPUT numel, once
      per subtree object.  Inside such a fold, occurrences are copies,
      not reads: ``concat(W, W)`` stores ``2·numel(W)`` and a folded
      ``matmul(B, A)`` stores ``numel(B@A)`` — billing the deduped
      leaf names would price phantom storage the weights file never
      had (and hide copies it does).

    A Param's numel comes from ``source_tensors[name]`` when the name
    resolves there — the actual tensor, authoritative for derived
    params — else from its ``TensorType`` (unknown dims
    count 1, matching ``_numel``'s best-effort convention).

    Follows the standard ``(term, memo=None)`` cost-fn convention;
    ``source_tensors`` is bound with :func:`param_bytes_cost_for` (or
    ``functools.partial(param_bytes_cost, source_tensors=...)``) for
    use in :meth:`EGraph.extract_best`.  The model sets
    ``charges_param_only`` so extraction does NOT apply the param-only
    discount — a compile-time-folded subtree still stores values —
    and ``dag_exact`` so :func:`dag_cost` returns this DAG-indexed sum
    verbatim instead of re-deriving it per node.
    """
    memo = {} if memo is None else memo
    return float(
        sum(_param_index(term, source_tensors, memo, by_bytes).values())
    )


# Markers read by EGraph.extract_best / dag_cost: storage pricing does
# not fold away at compile time, so param-only subtrees stay billed;
# and the leaf/fold index is already a true DAG cost, so dag_cost
# must not apply its subtractive per-node decomposition.
cast(_CostMarkers, param_bytes_cost).charges_param_only = True
cast(_CostMarkers, param_bytes_cost).dag_exact = True


def param_bytes_cost_for(
    source_tensors: dict | None = None, by_bytes: bool = False
):
    """Bind ``source_tensors`` and return a standard cost fn.

    Same closure convention as :func:`roofline_cost_for`: the result
    has signature ``fn(term, memo=None)`` — dropping straight into
    ``EGraph.extract_best`` / ``dag_cost`` — and carries the
    ``charges_param_only`` marker through to extraction.  ``by_bytes``
    weights each stored value by its dtype width — the axis under
    which quantized params price below fp32.
    """

    def cost(term: Any, memo: dict | None = None) -> float:
        return param_bytes_cost(term, source_tensors, memo, by_bytes)

    cost.__name__ = "param_bytes_cost_for"
    cast(_CostMarkers, cost).charges_param_only = True
    cast(_CostMarkers, cost).dag_exact = True
    return cost


def _param_numel(
    p: Param, source_tensors: dict | None, by_bytes: bool = False
) -> float:
    """Return the stored scalar count for one Param leaf.

    ``source_tensors`` (name -> tensor, e.g. from ``export_to_ir`` plus
    any derived params a pass injected) is authoritative when the
    name resolves there; else the declared ``TensorType``.  With
    ``by_bytes`` the count is weighted by the stored dtype's width
    (``numel * element_size``) — the axis under which int8 quantization
    prices 4× below fp32; unknown widths default to 4 bytes.
    """
    if source_tensors is not None:
        t = source_tensors.get(p.name)
        if t is not None:
            n = getattr(t, "numel", None)
            if callable(n):
                numel = float(n())
                if by_bytes:
                    esz = getattr(t, "element_size", None)
                    return numel * (esz() if callable(esz) else 4)
                return numel  # torch.Tensor / jax / etc.
            s = getattr(t, "size", None)
            if isinstance(s, (int, float)):
                return float(s)  # numpy .size
    shape = getattr(getattr(p, "typ", None), "shape", None)
    return float(_numel(shape))


#: Elementwise ops ``IRModule._fold_weight_chains`` (catopt_torch.torch_bridge)
#: folds eagerly through the ``_IR_TO_TORCH`` bindings when the whole
#: subtree is param-only.  ``matmul`` and ``concat`` are spelled out
#: separately in ``_folds_to_param`` because they fold under tighter arg
#: rules (real tensor operands, not Consts).  Keep in lock-step.
_FOLDABLE_ELEMWISE = frozenset(
    {
        "add",
        "mul",
        "sub",
        "div",
        "neg",
        "square",
        "sqrt",
        "sigmoid",
        "silu",
        "tanh",
        "gelu",
        "exp",
        "pow",
    }
)


#: Value-preserving view/copy ops the same ``_fold_weight_chains``
#: ``elif`` list folds on param-only subtrees: a ``reshape`` /
#: ``transpose`` of stored weights is compile-time work — unlike the
#: slice/extraction family (``select``/``index_select``/``slice``/
#: ``gather``/…), which deliberately stays runtime so dedup
#: re-materialisation keeps its saving.  ``reshape`` is how the
#: pairing pass spells a stacked expert parameter re-tiled as one
#: fused weight, so pricing it as foldable is what lets that member
#: win extraction.
_FOLDABLE_VIEWS = frozenset(
    {
        "reshape",
        "flatten",
        "unflatten",
        "transpose",
        "permute",
        "movedim",
        "unsqueeze",
        "squeeze",
        "contiguous",
        "detach",
        "detach_",
        "alias",
        "clone",
    }
)


def _has_var_leaf(term: Any, memo: dict) -> bool:
    """Return True iff the subtree reads a data input (Var leaf).

    Private alias kept for this module's fold walkers; delegates to
    :func:`catopt_core.typing.has_var_leaf` — the single implementation —
    which uses the same ``("hv", term)`` key convention, so the memo
    this shares with ``_shape_of``/``_folds_to_param`` keeps its exact
    key/value contents.
    """
    return has_var_leaf(term, memo)


def _param_resolves(p: Param, source_tensors: dict | None) -> bool:
    """Return whether ``p.name`` lands in ``_param_values`` at lowering.

    ``optimize_model`` hands the whole ``source_tensors`` dict to the
    lowerer as ``param_values`` (the sharing passes register their
    derived names into it), so an unbound ``source_tensors`` —
    ``param_bytes_cost_for()`` — assumes every leaf resolves.  With a
    bound dict the check is exact: a leaf absent from it cannot fold
    (it is still stored — ``_build_params`` registers it regardless).
    """
    return source_tensors is None or p.name in source_tensors


def _folds_to_param(
    term: Any, source_tensors: dict | None, memo: dict
) -> bool:
    """Return whether the lowerer folds *term* to a fused Param.

    Mirrors the lowerer bottom-up: a param-only subtree folds when
    every argument reduces to a stored parameter — a resolvable
    ``Param`` leaf or a subtree that itself folds (folded intermediates
    are registered into ``_param_values`` before their parents are
    considered).  ``Const`` operands are allowed only where the
    lowering accepts them: elementwise ops, but not the ``matmul``
    two-Param fold nor ``concat`` (``torch.cat`` has no scalar form).
    """
    if not isinstance(term, Op):
        return False
    k = ("pf", term)
    hit = memo.get(k)
    if hit is not None:
        return hit
    res = False
    if not _has_var_leaf(term, memo):

        def to_param(a: Any, allow_const: bool) -> bool:
            if isinstance(a, Param):
                return _param_resolves(a, source_tensors)
            if isinstance(a, Const):
                return allow_const
            return _folds_to_param(a, source_tensors, memo)

        if term.op in ("matmul", "concat") and len(term.args) >= 2:
            res = all(to_param(a, False) for a in term.args)
        elif term.op in _FOLDABLE_ELEMWISE | _FOLDABLE_VIEWS:
            # Elementwise ops and value-preserving views alike accept
            # Const operands at lowering (``isinstance(a, (_Param,
            # Const))``) — the Const arity rule is uniform.
            res = all(to_param(a, True) for a in term.args)
    memo[k] = res
    return res


def _fold_ewidth(
    term: Any, source_tensors: dict | None
) -> float | None:
    """Return the element width of a materialised fold.

    The widest resolvable leaf's dtype (the fused tensor inherits arg
    dtypes); ``None`` when no leaf carries one, letting the caller
    default to fp32.
    """
    if isinstance(term, Param):
        if source_tensors is not None:
            t = source_tensors.get(term.name)
            esz = getattr(t, "element_size", None)
            if callable(esz):
                return float(esz())
        return 4.0
    if isinstance(term, Op):
        ws = [
            w
            for a in term.args
            if (w := _fold_ewidth(a, source_tensors)) is not None
        ]
        return max(ws) if ws else None
    return None


def _fold_numel(
    term: Op, source_tensors: dict | None, memo: dict, by_bytes: bool
) -> float:
    """Return the stored size of a materialised fold.

    The size is the OUTPUT numel: ``concat`` re-stores every argument's
    rows, a weight ``matmul`` stores the dense product.
    """
    n = float(_numel(_shape_of(term, memo)))
    if by_bytes:
        n *= _fold_ewidth(term, source_tensors) or 4.0
    return n


def _param_index(
    term: Any,
    source_tensors: dict | None,
    memo: dict,
    by_bytes: bool = False,
) -> dict[Any, float]:
    r"""``{key: numel}`` for every *stored* parameter entry in a term DAG.

    Two kinds of entries, mirroring the lowered weights file:

    * ``{param_name: numel}`` — a Param leaf that survives folding;
      deduped by name, so a weight read by several consumers (or by
      several identically-spelled leaves) is stored once;
    * ``{"\x00fold:<id>": out_numel}`` — a param-only subtree the
      lowerer materialises (``_folds_to_param``); keyed by subtree
      object identity, matching ``_fold_memo``/``_build_params``: the
      same object reached twice is one stored tensor, and each
      materialisation is billed at output size regardless of how the
      leaves underneath dedup.
    """
    key = ("pbi", term)
    hit = memo.get(key)
    if hit is not None:
        return hit
    if isinstance(term, Param):
        out = {term.name: _param_numel(term, source_tensors, by_bytes)}
    elif isinstance(term, Op):
        if _folds_to_param(term, source_tensors, memo):
            out = {
                ("\x00fold", term): _fold_numel(
                    term, source_tensors, memo, by_bytes
                )
            }
        else:
            out = {}
            for a in term.args:
                for n, v in _param_index(
                    a, source_tensors, memo, by_bytes
                ).items():
                    out.setdefault(n, v)
    else:
        out = {}
    memo[key] = out
    return out


#: Ops whose torch binding is pure POINTWISE tensor work — the unit a
#: compiled lowering (torch.compile / Inductor) fuses into a single
#: kernel.  The elementwise core set plus the rest of the pointwise
#: bindings in the core table (casts, comparisons, where/clone, the
#: cmask/fill/attnbias mask generators) and the carrier ops whose
#: bodies are ordinary elementwise arithmetic on the carried
#: components — ``applyd`` = f0⊙h + f1, ``affd_compose`` =
#: (f0⊙g0, f0⊙g1 + f1), ``om_compose``/``omd_compose`` = maximum +
#: where-rescaled mul-adds, ``om_apply`` = a/l, ``omd_apply`` =
#: (fa⊙h + fb)/l (torch_bridge._CORE_TORCH_BINDINGS,
#: catopt_carriers.xcarrier.TORCH_BINDINGS).  NOT here: the bindings
#: hiding a contraction or reduction — ``apply``/``aff_compose``/
#: ``omd_applym`` (matmul), ``om_elem``/``omd_elem``/``om_elem_aff``/
#: ``om_elem_affd`` (amax + GEMM) — those stay fusion boundaries.
_FUSION_POINTWISE_OPS: frozenset = _FOLDABLE_ELEMWISE | frozenset(
    {
        "relu",
        "rsqrt",
        "cos",
        "sin",
        "eq",
        "ne",
        "lt",
        "le",
        "gt",
        "ge",
        "logical_not",
        "where",
        "masked_fill",
        "clone",
        "to",
        "type_as",
        "float",
        "cmask",
        "fill",
        "attnbias",
        "affd_compose",
        "applyd",
        "om_compose",
        "om_apply",
        "omd_compose",
        "omd_apply",
    }
)


def dag_cost(term: Any, cost_fn, memo: dict | None = None) -> float:
    """Return the true DAG cost of an extracted term.

    ``cost_fn(term)`` counts shared subtrees once per *parent* (a tree
    walk); extracted terms can share Op objects when two e-class parents
    picked terms over the same e-class (e.g. one fused GEMM under two
    split views).  This sums each distinct node's local cost —
    ``cost_fn(node) - sum(cost_fn(children))`` — deduplicated by object
    identity.

    ``memo`` is forwarded to cost functions that accept it (the
    built-in models do), so children already costed are O(1) lookups.

    Cost models that price parameter *storage* (``param_bytes_cost``)
    opt out of the param-only discount by setting
    ``charges_param_only`` on the function: a folded subtree still
    stores its leaves' values.  Models that already index the whole
    term DAG (``param_bytes_cost`` again — by leaf name plus
    materialised-subtree identity) set ``dag_exact`` instead: the
    function's value on the root IS the DAG cost, so the subtractive
    per-node decomposition below is skipped — it would mis-bill them.
    """
    # getattr(..., "func", ...) unwraps functools.partial bindings.
    bill_params = getattr(
        getattr(cost_fn, "func", cost_fn), "charges_param_only", False
    )
    c = _memo_dispatch(cost_fn, memo)

    # Cost models that already index the whole term DAG — by leaf name
    # and by materialised-subtree identity (``param_bytes_cost``) —
    # compute the true DAG cost directly.  The per-node decomposition
    # ``local = c(t) − Σc(children)`` below would mis-bill them: a leaf
    # nested inside a folded subtree is subtracted at every ancestor
    # fold yet charged once as a leaf, erasing exactly the materialised
    # copies the model exists to count.
    if getattr(getattr(cost_fn, "func", cost_fn), "dag_exact", False):
        return c(term)

    seen: set[int] = set()
    total = 0.0

    var_memo: dict = {}

    def has_var(t: Any) -> bool:
        """Return True if the subtree reads a data input (Var leaf).

        Subtrees over only Param/Const leaves are compile-time work —
        lowering folds them into a materialised parameter — so they are
        charged 0, matching extract_best's param-only discount.
        Delegates to :func:`catopt_core.typing.has_var_leaf` (the shared
        implementation) with this DAG walk's private memo.
        """
        return has_var_leaf(t, var_memo)

    fold_memo: dict = {}

    def rec(t: Any) -> None:
        nonlocal total
        if t in seen:
            return
        seen.add(t)
        if isinstance(t, Op):
            if (
                not has_var(t)
                and not bill_params
                and _folds_to_param(t, None, fold_memo)
            ):
                return  # folds at compile time — free at runtime
            # Param-only but NOT foldable (e.g. a ``trace`` resolvent's
            # hidden linalg.solve) still runs every call — bill it.
            for a in t.args:
                rec(a)
            local = c(t) - sum(c(a) for a in t.args)
            total += max(local, 0.0)
        else:
            total += c(t)

    rec(term)
    return total
