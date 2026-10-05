"""Categorical / string-diagram IR for tensor computation graphs.

Terms form a simply-typed algebra where each generator (operation) has
named inputs and outputs, mirroring morphisms in a symmetric monoidal
category.  Sequential composition is ordinary function application
(f ∘ g), and the monoidal product is parallel/tensor composition.

In the e-graph representation each term is an *ENode*: an operation
label plus a list of child e-class IDs.  The e-graph itself (see
:mod:`catopt_core.egraph`) maintains equivalence(classes of ENodes.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from catopt_core.attrs import validate_attrs

__all__ = [
    "IR",
    "Const",
    "Op",
    "Param",
    "TensorType",
    "Var",
    "attr_from_data",
    "attr_to_data",
    "generator",
    "op_def",
    "op_repr",
    "op_repr_dag",
    "term_from_data",
    "term_to_data",
]


# ---------------------------------------------------------------------------
#  Shape / type system (minimal — enough for our cost model)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorType:
    """A tensor type characterized by its shape."""

    shape: tuple[int | None, ...]

    @property
    def size(self) -> int | None:
        """Return the element count, or None if a dim is unknown."""
        if any(d is None for d in self.shape):
            return None
        result = 1
        for d in self.shape:
            result = result * cast("int", d)
        return result

    def __repr__(self) -> str:
        """Return a debug rendering of the tensor type."""
        inner = ", ".join(
            "?" if d is None else str(d) for d in self.shape
        )
        return f"TensorType({inner})"


# ---------------------------------------------------------------------------
#  Terms  (these become ENodes in the e-graph)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Var:
    """A variable / placeholder (input to the computation)."""

    name: str
    typ: TensorType

    def __repr__(self) -> str:
        """Return the variable's name."""
        return self.name


@dataclass(frozen=True, eq=False)
class Const:
    """A literal constant scalar.

    Int values are preserved (not coerced to float): ``x % 2`` must
    eval to an int64 operand or weak-type promotion goes wrong
    downstream.

    Equality is *spelling-strict* — ``Const(8) != Const(8.0)`` —
    because ``Const`` is a structural object, not a number: it is the
    e-graph's leaf identity (``EGraph.add_leaf`` keys on
    ``repr(term)``, ``_term_match`` compares ``repr``) and it is what
    ``Op`` interning, ``add_term`` memos and every term-keyed dict
    hash and compare on.  Python's numeric tower (``8 == 8.0``,
    ``hash(8) == hash(8.0)``) leaking through the dataclass-default
    ``__eq__`` silently coalesced ``Const(8)`` and ``Const(8.0)`` into
    one interned object / one leaf e-class — a real dtype change in
    the program.  Matchers that *want* numeric leaf leniency say so
    explicitly: ``meta.match_pattern`` compares ``Const.value``
    numerically for the identity laws.
    """

    value: int | float

    def __repr__(self) -> str:
        """Return the constant's value as text."""
        return str(self.value)

    def __eq__(self, other: Any) -> bool:
        """Compare leaf identity: same spelling, not just same number."""
        if not isinstance(other, Const):
            return NotImplemented
        return repr(self.value) == repr(other.value)

    def __hash__(self) -> int:
        """Hash the leaf's repr — the same key the e-graph uses."""
        return hash(repr(self.value))


@dataclass(frozen=True)
class Param:
    """A learnable parameter (weight matrix, bias, etc.)."""

    name: str
    typ: TensorType

    def __repr__(self) -> str:
        """Return the parameter's name."""
        return self.name


def _hashable_attr(v: Any) -> Any:
    """Strict key form of an attr value: the spelling, not the number.

    The intern key and ``Op.__eq__`` follow the same rule as
    :class:`Const` leaf identity — ``repr`` — because Python's numeric
    tower (``0 == 0.0 == False``, shared hashes) is a semantic claim
    the IR is not entitled to make: ``clamp(min=0)`` and
    ``clamp(min=0.0)`` are different programs (an int bound promotes
    differently at lowering; a float ``dim`` is simply malformed).
    repr also keeps ``[3,5]`` distinct from ``(3,5)`` and ``(3,5)``
    from ``(3.0,5.0)``, makes ``nan`` attrs reflexive, and covers the
    unhashables outright.  Matchers that *want* numeric leniency say
    so — the pattern matchers compare attr values with ``!=``.
    """
    return repr(v)


def _attr_key(attrs: dict[str, Any]) -> tuple:
    """Intern/identity key for an attrs dict — repr-keyed, sorted."""
    return tuple(
        sorted((k, _hashable_attr(v)) for k, v in attrs.items())
    )


@dataclass(frozen=True, eq=False)
class Op:
    """An operation (an ENode in the e-graph).

    Terms are **hash-consed**: ``Op.make`` interns by structure, so
    equal terms are the same object.  ``__hash__``/``__eq__`` are
    content-based — term objects are safe dict/set keys, and the old
    ``id(t)``-keyed memos can all key on ``t`` directly (no GC-reuse
    hazard, no keepalive lists).

    Attr identity is *spelling-strict* — the :class:`Const` precedent
    one level down: ``min=0`` and ``min=0.0`` are different terms.
    The numeric tower's ``0 == 0.0`` (with ``hash(0) == hash(0.0)``)
    leaked through the raw-value ``_attr_key`` and the dataclass dict
    compare, coalescing both spellings into one interned object — a
    silent attr rewrite before any law fired.  ``__eq__`` therefore
    compares the repr-keyed ``_ak`` signature rather than the attrs
    dicts.  *Matching* stays numerically lenient — ``_term_match``,
    ``match_pattern`` and ``EGraph._m_stream`` all compare attr
    values with ``!=``.
    """

    op: str
    args: tuple[Any, ...]
    attrs: dict[str, Any] = field(default_factory=dict)
    _h: int = field(init=False, repr=False, compare=False, hash=False)
    _ak: tuple = field(
        init=False, repr=False, compare=False, hash=False
    )

    _INTERN: ClassVar[Any] = None  # weakref table, created at import

    def __post_init__(self) -> None:
        """Cache the strict attr signature and content hash."""
        ak = _attr_key(self.attrs)
        object.__setattr__(self, "_ak", ak)
        object.__setattr__(self, "_h", hash((self.op, self.args, ak)))

    def __eq__(self, other: Any) -> bool:
        """Compare structure — op, args, and the strict attr key."""
        if not isinstance(other, Op):
            return NotImplemented
        return (
            self.op == other.op
            and self.args == other.args
            and self._ak == other._ak
        )

    @staticmethod
    def make(op: str, *args, validate: bool = True, **attrs) -> Op:
        """Mint an op term under the ``catopt_core.attrs`` contract.

        Ops declared in ``ATTR_SCHEMA`` are validated at mint: a
        positional ``argN`` at an undeclared position, or a required
        attr missing from a fully-attributed term, dies here with a
        ``ValueError`` — not silently at eval.  ``argN`` at declared
        positions stays legal (dual-spelling rule variants and
        hand-minted terms consume it); the positional spelling only
        merges — winning — when the canonical name is also supplied.
        ``validate=False`` escapes for deliberate non-canonical mints.

        Interned: structurally identical makes return one object.
        """
        a = validate_attrs(op, attrs, validate=validate)
        try:
            key = (op, tuple(args), _attr_key(a))
            existing = Op._INTERN.get(key)
        except TypeError:
            key = None
            existing = None
        if existing is not None:
            return existing
        t = Op(op, args, a)
        if (
            key is not None
        ):  # pragma: no branch — unhashable args die earlier in __post_init__
            Op._INTERN[key] = t
        return t

    def __hash__(self) -> int:
        """Return the cached content hash."""
        return self._h

    def __repr__(self) -> str:
        """Return an S-expression rendering of the term."""
        parts = [op_repr(a) for a in self.args]
        if self.attrs:
            attr_str = ", ".join(
                f"{k}={v}" for k, v in self.attrs.items()
            )
            parts.append(attr_str)
        inside = ", ".join(parts)
        return f"{self.op}({inside})"


Op._INTERN = weakref.WeakValueDictionary()


def op_repr(term: Any) -> str:
    """Compact S-expression-style rendering of a term."""
    if isinstance(term, Op):
        parts = [op_repr(a) for a in term.args]
        if term.attrs:
            parts.append(
                ", ".join(f"{k}={v}" for k, v in term.attrs.items())
            )
        return f"({term.op} {', '.join(parts)})"
    return repr(term)


def _dag_shared(root: Op) -> set[Op]:
    """Ops inside ``root`` referenced by more than one parent slot."""
    counts: dict[Op, int] = {}
    counted: set[Op] = set()
    todo = [root]
    while todo:
        t = todo.pop()
        if t in counted:
            continue
        counted.add(t)
        for a in t.args:
            if isinstance(a, Op):
                counts[a] = counts.get(a, 0) + 1
                todo.append(a)
    return {t for t, c in counts.items() if c > 1}


def _dag_postorder(root: Op) -> list[Op]:
    """Every op in ``root``'s DAG, children before parents, deduped."""
    order: list[Op] = []
    expanded: set[Op] = set()
    work: list[tuple[Op, bool]] = [(root, False)]
    while work:
        t, done = work.pop()
        if done:
            order.append(t)
            continue
        if t in expanded:
            continue
        expanded.add(t)
        work.append((t, True))
        work.extend(
            (a, False)
            for a in t.args
            if isinstance(a, Op) and a not in expanded
        )
    return order


def _sexp(t: Op, arg_repr: Callable[[Any], str]) -> str:
    """Render one node's ``(op arg, …, k=v)`` s-expression."""
    parts = [arg_repr(a) for a in t.args]
    if t.attrs:
        parts.append(", ".join(f"{k}={v}" for k, v in t.attrs.items()))
    return f"({t.op} {', '.join(parts)})"


def op_repr_dag(term: Any) -> str:
    """Compact S-expression rendering of a sharing-heavy term DAG.

    ``op_repr`` renders a term as a *tree*: a multiply-referenced
    subterm is expanded once per use site.  Terms are hash-consed
    DAGs, and a sharing-heavy one (a morphism window joint, where
    each residual-stream node feeds both the next ``add`` and the
    whole next block body) tree-expands exponentially — the repr
    itself exhausts memory before the string exists.  Here every op
    referenced by more than one parent is emitted once as a ``#n``
    binding in a ``let`` prelude and named at each use site, so the
    output is linear in the number of *distinct* nodes.

    Terms without sharing render exactly as ``op_repr``.
    """
    if not isinstance(term, Op):
        return repr(term)
    shared = _dag_shared(term)
    if not shared:
        return op_repr(term)
    names: dict[Op, str] = {}
    defs: list[str] = []
    rendered: dict[Op, str] = {}
    # Post-order emits every ``#n`` binding before all of its uses.
    for t in _dag_postorder(term):
        s = _sexp(
            t,
            lambda a: (
                (names[a] if a in shared else rendered[a])
                if isinstance(a, Op)
                else repr(a)
            ),
        )
        if t in shared:
            names[t] = f"#{len(defs)}"
            defs.append(s)
        else:
            rendered[t] = s
    bound = " ".join(f"(#{i} {d})" for i, d in enumerate(defs))
    # The root is no node's child, so it is never a shared binding.
    return f"(let ({bound}) {rendered[term]})"


# ---------------------------------------------------------------------------
#  Term serialization — the JSON-safe data form (the lemma-store seam)
# ---------------------------------------------------------------------------


def attr_to_data(v: Any) -> Any:
    """Encode an attribute value as JSON-safe data.

    Also handles a ``$attr:`` binding value — strings stay strings
    since they are metavar references.
    """
    if isinstance(v, tuple):
        return {"__tuple__": [attr_to_data(x) for x in v]}
    if isinstance(v, list):
        return {"__list__": [attr_to_data(x) for x in v]}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    raise TypeError(f"unserializable attribute value: {v!r}")


def attr_from_data(v: Any) -> Any:
    """Decode :func:`attr_to_data` output back to a Python value."""
    if isinstance(v, dict):
        if "__tuple__" in v:
            return tuple(attr_from_data(x) for x in v["__tuple__"])
        if "__list__" in v:
            return [attr_from_data(x) for x in v["__list__"]]
        raise ValueError(f"bad attr encoding: {v!r}")
    return v


def term_to_data(t: Any) -> Any:
    """Encode a (sub)term as JSON-safe data: Op tree, metavar, leaf.

    The scheme — canonical since ``rulecache`` adopted it — is
    ``{"mvar": name}`` for a pattern metavariable (a bare ``str``
    leaf), ``{"const": v}`` for :class:`Const`, ``{"var": name,
    "shape": [...]}`` / ``{"param": name, "shape": [...]}`` for typed
    leaves, and ``{"op": name, "args": [...], "attrs": {k: attr}}``
    for :class:`Op` nodes.  Note ``"var"`` means a typed :class:`Var`
    leaf — a *metavariable* is ``"mvar"``.
    """
    if isinstance(t, str):
        return {"mvar": t}
    if isinstance(t, Const):
        return {"const": t.value}
    if isinstance(t, Var):
        return {"var": t.name, "shape": list(t.typ.shape)}
    if isinstance(t, Param):
        return {"param": t.name, "shape": list(t.typ.shape)}
    if isinstance(t, Op):
        return {
            "op": t.op,
            "args": [term_to_data(a) for a in t.args],
            "attrs": {k: attr_to_data(v) for k, v in t.attrs.items()},
        }
    raise TypeError(f"unserializable term: {t!r}")


def term_from_data(d: Any) -> Any:
    """Decode :func:`term_to_data` output back to a term."""
    if not isinstance(d, dict) or len(d) < 1:
        raise ValueError(f"bad term encoding: {d!r}")
    if "mvar" in d:
        return d["mvar"]
    if "const" in d:
        return Const(d["const"])
    if "var" in d:
        return Var(d["var"], TensorType(tuple(d["shape"])))
    if "param" in d:
        return Param(d["param"], TensorType(tuple(d["shape"])))
    if "op" in d:
        return Op.make(
            d["op"],
            *[term_from_data(a) for a in d["args"]],
            **{k: attr_from_data(v) for k, v in d["attrs"].items()},
        )
    raise ValueError(f"bad term encoding: {d!r}")


# ---------------------------------------------------------------------------
#  Generator registry — declares which ops exist and their signatures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenDef:
    """Definition of a generator (operation) in the categorical semantics.

    Documents the categorical law(s) that make the operation interesting
    for rewriting.
    """

    name: str
    n_in: int  # number of input wires
    n_out: int  # number of output wires
    commutative: bool = False
    associative: bool = False
    identity: Any | None = None
    law: str = ""  # human-readable description of the categorical law


_OP_REGISTRY: dict[str, GenDef] = {}


def op_def(
    name: str,
    n_in: int,
    n_out: int = 1,
    *,
    commutative: bool = False,
    associative: bool = False,
    identity: Any | None = None,
    law: str = "",
) -> GenDef:
    """Register a generator and return its definition."""
    g = GenDef(
        name, n_in, n_out, commutative, associative, identity, law
    )
    _OP_REGISTRY[name] = g
    return g


def generator(name: str) -> GenDef | None:
    """Look up a registered generator by name."""
    return _OP_REGISTRY.get(name)


# ---------------------------------------------------------------------------
#  Predefined generators
# ---------------------------------------------------------------------------

# Linear algebra
op_def(
    "matmul",
    2,
    1,
    law="Bilinear; distributes over elementwise add when weight is fixed.",
)
op_def(
    "add",
    2,
    1,
    commutative=True,
    associative=True,
    identity=0,
    law="Addition forms a commutative monoid (SMC symmetry + associativity).",
)
op_def(
    "mul",
    2,
    1,
    commutative=True,
    associative=True,
    identity=1,
    law="Multiplication forms a commutative monoid (Hadamard product).",
)
op_def(
    "sub",
    2,
    1,
    commutative=False,
    law="Subtraction: a - b = a + neg(b)  (distributivity of negation).",
)
op_def(
    "div",
    2,
    1,
    commutative=False,
    law="Division: a / b = a * (1/b)  (multiplicative inverse).",
)
op_def(
    "pow",
    2,
    1,
    commutative=False,
    law="Power: x**n, with square(x)=pow(x,2) and rsqrt as inverse sqrt.",
)

# Unary element-wise
op_def("square", 1, 1, law="Self-composition: square(x) = mul(x, x).")
op_def("sqrt", 1, 1, law="Square root: sqrt(x)**2 = x.")
op_def(
    "neg",
    1,
    1,
    law="Additive inverse: neg(x) + x = 0  (group inverse).",
)
op_def(
    "rsqrt", 1, 1, law="rsqrt(x)*sqrt(x)=1 (group inverse under mul)."
)
op_def("exp", 1, 1, law="Exponential.")
op_def("sigmoid", 1, 1, law="Logistic sigmoid; silu(x)=x*sigmoid(x).")
op_def("silu", 1, 1, law="SiLU/Swish: silu(x) = x * sigmoid(x).")
op_def("tanh", 1, 1, law="Hyperbolic tangent.")
op_def("gelu", 1, 1, law="Gaussian Error Linear Unit.")

# Reductions
op_def(
    "sum",
    1,
    1,
    law="Sum over axes; distributes over add (additivity of trace/sum).",
)
op_def(
    "mean", 1, 1, law="Mean = sum / size (linearity of expectation)."
)
op_def("max", 1, 1, law="Maximum; sublinear (convexity).")

# Data movement
op_def("transpose", 1, 1, law="Symmetry/isomorphism in the SMC.")
op_def(
    "reshape", 1, 1, law="Relabeling of wires; composes associatively."
)
op_def(
    "broadcast",
    1,
    1,
    law="Monoidal coherence: copy then f = f in parallel.",
)

# Product structure: pairing and projections.
# concat(A, B, dim) is the monoidal product on objects — juxtaposing the
# output spaces of two maps.  chunk(t, n, dim, i) is the projection pi_i
# that selects the i-th component.  Together they express the universal
# property of the product: <f, g> = (f x g) ∘ Δ.
op_def(
    "concat",
    2,
    1,
    law="Pairing on objects: concat(W1, W2) builds the product map "
    "weight.  Wire juxtaposition — a data-movement op.",
)
op_def(
    "chunk",
    1,
    1,
    law="Projection pi_i of a paired output; a zero-cost view like "
    "transpose/reshape.",
)
op_def(
    "split",
    1,
    1,
    law="Projection pi_i with explicit (possibly unequal) section "
    "sizes — the asymmetric counterpart of chunk, for pairing "
    "maps with different output dimensions (e.g. GQA fused QKV).",
)
op_def(
    "index_select",
    1,
    1,
    law="Reindexing along one axis (a gather of rows/blocks): the "
    "generalised projection.  With a repeated-index map it "
    "realises slice-level sharing — the copy map Delta applied "
    "per row-block.",
)
op_def(
    "embedding",
    2,
    1,
    law="Row gather: embedding(W, idx) selects rows of the table; "
    "low-rank tables factor as gather-then-project.",
)
op_def(
    "float",
    1,
    1,
    law="Dtype coercion to working precision — the decode half of "
    "quantization-as-rewrite: mul(float(q), s).",
)

# Attention
op_def(
    "contiguous",
    1,
    1,
    law="Memory-layout coercion; semantically the identity map.",
)
op_def(
    "sdpa",
    3,
    1,
    law="Scaled dot-product attention; a fused nonlinear kernel "
    "whose three arguments are projections of the same source.",
)


# ---------------------------------------------------------------------------
#  IR  — top-level program (a single term with free variables)
# ---------------------------------------------------------------------------


@dataclass
class IR:
    """A program: a term plus the set of free variables (inputs).

    The ``root`` term is the computation; ``inputs`` lists the
    :class:`Var` objects that are the program's inputs.  Parameters
    (:class:`Param`) are implicitly tracked as the learnable weights
    inside the term.
    """

    root: Any  # Var | Const | Param | Op
    inputs: list[Var] = field(default_factory=list)
    input_names: set[str] = field(default_factory=set)
    params: dict[str, Param] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return an S-expression rendering of the root term."""
        return op_repr(self.root)
