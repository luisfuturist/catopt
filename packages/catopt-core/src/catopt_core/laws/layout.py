"""Layout laws: exact transpose movement through pointwise and GEMM ops.

``transpose`` is a pure re-layout — a stride permutation that lowers to
a view (or a BLAS transpose flag), never a copy.  Moving one is a
*layout choice*, not an algebraic simplification, so these laws let the
e-graph hold both the materialised-transpose and the migrated/fused
form and let extraction pick whichever the cost model prefers:

* **Pointwise naturality** — transpose commutes with every elementwise
  map: ``T(f(x)) = f(T(x))`` for unary ``f``, and
  ``T(g(a, b)) = g(T(a), T(b))`` for binary ``g`` whose operands share
  a rank (broadcasting is positional — permuting equal-rank operands by
  the same swap preserves the broadcast dim-for-dim).  Pushed inward,
  a transpose lands on a GEMM operand where it is free.
* **Involution** — ``T(T(x)) = x`` on the same axis pair.
* **Product transpose** — ``(A @ B).mT = B.mT @ A.mT`` for the
  last-two-axis swap (batched included: per batch element it is the
  rank-2 identity).
* **Linear ≡ NT GEMM** — ``matmul(x, W.mT) = linear(x, W)`` for rank-2
  ``W``: ``F.linear`` IS the BLAS-NT call — the weight stays in
  ``(out, in)`` storage and no transpose materialises.

Both ``transpose`` spellings are covered: the canonical exported form
``transpose(x, dim0=…, dim1=…)`` and the bare ``t()``/``.mT`` form (no
attrs — the torch binding reads it as the last-two swap,
``x.t()`` on rank 2 / ``x.transpose(-2, -1)`` otherwise).

``LAYOUT_RULES`` is folded into ``ALL_RULES`` (:mod:`.tensor`), so the
default ``optimize_model`` saturation sees both layouts.

Semantic coverage note: ``conv2d`` exists in the op table, but an NCHW
↔ NHWC rewrite is NOT expressible here — ``permute`` changes the
*logical* shape (a permuted input is a different tensor to
``F.conv2d``, not the same tensor in another storage), and the IR has
no memory-format attribute.  Layout movement for conv would need a
dedicated op; this slice covers matmul/linear/transpose.
"""

from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.ir import Op
from catopt_core.laws import tags as _tags
from catopt_core.laws.base import R as _R
from catopt_core.laws.base import _shape_of
from catopt_core.typing import _axis_pair, _broadcast


def R(name: str, lhs, rhs, **kw) -> Rewrite:
    """Module-local law constructor — layout laws carry the ``LAYOUT`` tag."""
    kw.setdefault("tags", (_tags.LAYOUT,))
    return _R(name, lhs, rhs, **kw)


# ---------------------------------------------------------------------------
#  Pattern/mint helpers
# ---------------------------------------------------------------------------


def _t(term: Any, dimmed: bool, k0: str = "D0", k1: str = "D1") -> Op:
    """Build a ``transpose`` pattern over *term*.

    ``dimmed`` spells the canonical ``dim0``/``dim1`` form with attr
    metavariables *k0*/*k1*; ``False`` spells the bare ``t()`` form
    (no attrs — the implicit last-two swap).
    """
    if dimmed:
        return Op.make("transpose", term, dim0=k0, dim1=k1)
    return Op.make("transpose", term)


def _mt(term: Any) -> Op:
    """Mint the canonical last-two-axis swap — the ``x.mT`` transpose."""
    return Op.make("transpose", term, dim0=-2, dim1=-1)


# ---------------------------------------------------------------------------
#  Check hooks — the matcher sees e-classes, not shapes; every law that
#  is only valid for particular ranks/axis pairs declares it here.
# ---------------------------------------------------------------------------


def _bound_axes(
    bound: dict, k0: str, k1: str, rank: int
) -> tuple[int, int] | None:
    """Normalised axis pair a bound ``transpose`` applies, or ``None``.

    When neither ``$attr:k0`` nor ``$attr:k1`` bound, the pattern was a
    *bare* transpose — the lowering's implicit last-two swap.  A dimmed
    pair normalises through :func:`catopt_core.typing._axis_pair`
    (``None`` for non-int or out-of-range dims).  Rank < 2 declines
    unconditionally — there is no swap to move.
    """
    if rank < 2:
        return None
    if f"$attr:{k0}" not in bound and f"$attr:{k1}" not in bound:
        return (rank - 2, rank - 1)
    return _axis_pair(
        bound.get(f"$attr:{k0}"), bound.get(f"$attr:{k1}"), rank
    )


def _is_swap(pair: tuple[int, int] | None, rank: int) -> bool:
    """Return whether *pair* is the last-two-axis swap on ``rank``."""
    return pair is not None and set(pair) == {rank - 2, rank - 1}


def _check_commute_unary(bound: dict) -> bool:
    """``T(f(x)) = f(T(x))`` — pointwise commutes with relayout.

    The bound ``x`` must carry a real axis pair (a known rank ≥ 2 and
    two distinct axes — a self-pair transpose is a no-op view, not a
    layout worth holding).
    """
    s = _shape_of(bound.get("x"))
    if not isinstance(s, tuple):
        return False
    ax = _bound_axes(bound, "D0", "D1", len(s))
    return ax is not None and ax[0] != ax[1]


def _check_commute_binary(bound: dict) -> bool:
    """``T(g(a, b)) = g(T(a), T(b))`` for equal-rank pointwise ``g``.

    Broadcasting is positional: with ``rank(a) == rank(b)``, applying
    the same axis swap to both operands permutes the broadcast
    dim-for-dim — every output element reads the same operand pair.
    Rank-mismatched operands would realign the trailing dims after the
    swap and compute something else — declined.
    """
    sa, sb = _shape_of(bound.get("a")), _shape_of(bound.get("b"))
    if not (
        isinstance(sa, tuple)
        and isinstance(sb, tuple)
        and len(sa) == len(sb)
    ):
        return False
    ax = _bound_axes(bound, "D0", "D1", len(sa))
    return ax is not None and ax[0] != ax[1]


def _check_involution(bound: dict) -> bool:
    """``T(T(x)) = x`` — both transposes must swap the SAME axis pair."""
    s = _shape_of(bound.get("x"))
    if not isinstance(s, tuple):
        return False
    inner = _bound_axes(bound, "I0", "I1", len(s))
    outer = _bound_axes(bound, "O0", "O1", len(s))
    if (
        inner is None
        or outer is None
        or inner[0] == inner[1]
        or outer[0] == outer[1]
    ):
        return False
    return set(inner) == set(outer)


def _check_transpose_matmul(bound: dict) -> bool:
    """Check ``(A @ B).mT = B.mT @ A.mT`` for a bound matmul.

    The outer transpose must be the last-two-axis swap of the matmul's
    output.

    Rank-1 operands are excluded: ``.mT`` on a vector is the identity
    and the product identity degenerates.  The output rank is the
    broadcast of the two batch prefixes plus the matrix pair — that is
    where the transpose's axes must land.
    """
    sa, sb = _shape_of(bound.get("A")), _shape_of(bound.get("B"))
    if not (
        isinstance(sa, tuple)
        and isinstance(sb, tuple)
        and len(sa) >= 2
        and len(sb) >= 2
    ):
        return False
    batch = _broadcast(sa[:-2], sb[:-2])
    if not isinstance(batch, tuple):
        return False  # provably ill-typed batch dims
    if (
        isinstance(sa[-1], int)
        and isinstance(sb[-2], int)
        and sa[-1] != sb[-2]
    ):
        return False  # contraction dims must agree
    rank = len(batch) + 2
    return _is_swap(_bound_axes(bound, "D0", "D1", rank), rank)


def _check_mm_transposes(bound: dict) -> bool:
    """Check ``P.mT @ Q.mT = (Q @ P).mT``.

    Both inner transposes must be the last-two swap of their own
    operand.
    """
    sp, sq = _shape_of(bound.get("P")), _shape_of(bound.get("Q"))
    if not (
        isinstance(sp, tuple)
        and isinstance(sq, tuple)
        and len(sp) >= 2
        and len(sq) >= 2
    ):
        return False
    return _is_swap(
        _bound_axes(bound, "P0", "P1", len(sp)), len(sp)
    ) and _is_swap(_bound_axes(bound, "Q0", "Q1", len(sq)), len(sq))


def _linear_shapes_ok(bound: dict) -> bool:
    """Check the ``linear`` contract.

    W must be rank-2 ``(out, in)`` and x's last dim must feed ``in``.
    """
    w, x = _shape_of(bound.get("W")), _shape_of(bound.get("x"))
    if not (isinstance(w, tuple) and len(w) == 2):
        return False
    if not (isinstance(x, tuple) and len(x) >= 1):
        return False
    return not (
        isinstance(x[-1], int)
        and isinstance(w[1], int)
        and x[-1] != w[1]
    )


def _check_mm_t_is_linear(bound: dict) -> bool:
    """Check ``matmul(x, W.mT) ≡ linear(x, W)``.

    The bound transpose must be the full swap of W's two axes (rank-2
    makes that the only real one).
    """
    if not _linear_shapes_ok(bound):
        return False
    return _is_swap(_bound_axes(bound, "D0", "D1", 2), 2)


def _check_linear_is_mm_t(bound: dict) -> bool:
    """Check ``linear(x, W) ≡ matmul(x, W.mT)`` — the same contract."""
    return _linear_shapes_ok(bound)


# ---------------------------------------------------------------------------
#  Pointwise commutation families
# ---------------------------------------------------------------------------

#: Unary pointwise ops with a torch binding and no axis/scale attrs —
#: ``f`` is applied elementwise, so any operand permutation commutes.
_POINTWISE_UNARY: tuple[str, ...] = (
    "neg",
    "abs",
    "silu",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "exp",
    "sqrt",
    "rsqrt",
    "square",
    "log",
)

#: Binary pointwise ops — sound under the same-rank guard.
_POINTWISE_BINARY: tuple[str, ...] = ("add", "mul", "sub", "div")


def _commute_rules() -> list[Rewrite]:
    """Build the pointwise-transpose commutations.

    Per op and per spelling (canonical ``dim0``/``dim1`` + bare
    ``t()``), in both directions — *push* distributes ``transpose``
    over the pointwise node, *pull* factors it back out.
    """
    rules: list[Rewrite] = []
    for op in _POINTWISE_UNARY:
        for bare in (False, True):
            dimmed = not bare
            tag = "_bare" if bare else ""
            fx = Op.make(op, "x")
            tx = _t("x", dimmed)
            rules.append(
                R(
                    f"transpose_push_{op}{tag}",
                    _t(fx, dimmed),
                    Op.make(op, tx),
                    law="Transpose commutes with the pointwise map "
                    f"{op}: T·f(x) = f(T·x) — naturality of the "
                    "elementwise action.",
                    check=_check_commute_unary,
                )
            )
            rules.append(
                R(
                    f"transpose_pull_{op}{tag}",
                    Op.make(op, tx),
                    _t(fx, dimmed),
                    law=f"Inverse: f(T·x) = T·f(x) for {op} — hoist "
                    "the layout change back out.",
                    check=_check_commute_unary,
                )
            )
    for op in _POINTWISE_BINARY:
        for bare in (False, True):
            dimmed = not bare
            tag = "_bare" if bare else ""
            fab = Op.make(op, "a", "b")
            # Shared attr metavariables on both operand transposes
            # force an identical swap — positional broadcasting only
            # commutes with a common permutation.
            tab = Op.make(op, _t("a", dimmed), _t("b", dimmed))
            rules.append(
                R(
                    f"transpose_push_{op}{tag}",
                    _t(fab, dimmed),
                    tab,
                    law=f"Transpose distributes over {op}: "
                    "T·(a ⊙ b) = T·a ⊙ T·b — same-rank broadcast "
                    "commutes with relayout.",
                    check=_check_commute_binary,
                )
            )
            rules.append(
                R(
                    f"transpose_pull_{op}{tag}",
                    tab,
                    _t(fab, dimmed),
                    law=f"Inverse: T·a ⊙ T·b = T·(a ⊙ b) for {op}.",
                    check=_check_commute_binary,
                )
            )
    return rules


def _involution_rules() -> list[Rewrite]:
    """``T(T(x)) = x`` over every bare/dimmed spelling combination."""
    out: list[Rewrite] = []
    for inner in (True, False):
        for outer in (True, False):
            tag = ("d" if outer else "b") + ("d" if inner else "b")
            out.append(
                R(
                    f"transpose_transpose_{tag}",
                    _t(
                        _t("x", inner, "I0", "I1"),
                        outer,
                        "O0",
                        "O1",
                    ),
                    "x",
                    law="Transpose is an involution per axis pair: "
                    "T₂∘T₁ = id when both swap the same axes.",
                    check=_check_involution,
                )
            )
    return out


def _matmul_rules() -> list[Rewrite]:
    """Build the product-transpose law, both spellings/directions."""
    mm = Op.make("matmul", "A", "B")
    out = [
        R(
            "transpose_matmul",
            _t(mm, True),
            Op.make("matmul", _mt("B"), _mt("A")),
            law="(A@B).mT = B.mT @ A.mT — the output's last-two swap "
            "folds into per-operand transposes (the BLAS NT/TN flag "
            "read, batched).",
            check=_check_transpose_matmul,
        ),
        R(
            "transpose_matmul_bare",
            _t(mm, False),
            Op.make("matmul", _mt("B"), _mt("A")),
            law="Bare ``t()`` spelling of the product-transpose law — "
            "an implicit last-two swap.",
            check=_check_transpose_matmul,
        ),
    ]
    for pd in (True, False):
        for qd in (True, False):
            tag = ("d" if pd else "b") + ("d" if qd else "b")
            out.append(
                R(
                    f"matmul_transpose_rev_{tag}",
                    Op.make(
                        "matmul",
                        _t("P", pd, "P0", "P1"),
                        _t("Q", qd, "Q0", "Q1"),
                    ),
                    _mt(Op.make("matmul", "Q", "P")),
                    law="P.mT @ Q.mT = (Q@P).mT — hoist the product's "
                    "last-two swap outward, dropping both operand "
                    "transposes.",
                    check=_check_mm_transposes,
                )
            )
    return out


# ---------------------------------------------------------------------------
#  The NT bridge — ``linear`` IS the BLAS-NT form of ``x @ W.mT``
# ---------------------------------------------------------------------------

LINEAR_FROM_MM_T = R(
    "linear_from_matmul_t",
    Op.make("matmul", "x", _t("W", True)),
    Op.make("linear", "x", "W"),
    law="matmul(x, W.mT) ≡ linear(x, W): F.linear IS the BLAS-NT "
    "call — W stays in (out, in) storage and no transpose "
    "materialises.",
    check=_check_mm_t_is_linear,
)

LINEAR_FROM_MM_T_BARE = R(
    "linear_from_matmul_t_bare",
    Op.make("matmul", "x", _t("W", False)),
    Op.make("linear", "x", "W"),
    law="Bare-t() spelling of the NT bridge: x @ W.t() = linear(x, W) "
    "— the exported form of ``x @ W.t()``.",
    check=_check_mm_t_is_linear,
)

LINEAR_TO_MM_T = R(
    "linear_to_matmul_t",
    Op.make("linear", "x", "W"),
    Op.make("matmul", "x", _mt("W")),
    law="Reverse NT bridge — exposes the weight's implicit transpose "
    "so the migration laws can move it.",
    check=_check_linear_is_mm_t,
)


#: The whole layout-law surface — folded into ``ALL_RULES``.
LAYOUT_RULES: list[Rewrite] = [
    *_commute_rules(),
    *_involution_rules(),
    *_matmul_rules(),
    LINEAR_FROM_MM_T,
    LINEAR_FROM_MM_T_BARE,
    LINEAR_TO_MM_T,
]
