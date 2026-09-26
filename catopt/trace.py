# ruff: noqa: RUF001, RUF002, RUF003
"""Traced monoidal structure — feedback loops as first-class rewrite targets.

THEORY
    A traced monoidal category (Joyal–Street–Verity) adds, for every
    ``f : A ⊗ U → B ⊗ U``, a morphism ``Tr^U(f) : A → B`` — feedback of
    the ``U`` output wire into the ``U`` input wire.  The recurrence
    ``h_t = f(h_{t−1}, x_t)`` IS such a trace over time, and the JSV
    axioms are exactly the categorical laws that relate iterative and
    closed/parallel forms (our ``SCAN_LAWS`` scan work is a special
    case).  The axioms:

    * **vanishing**   ``Tr^I(f) = f``  and  ``Tr^{U⊗V}(f) = Tr^V(Tr^U(f))``
    * **superposing** ``Tr^{U⊗V}(f ⊗ g) = Tr^U(f) ⊗ Tr^V(g)``
    * **sliding**     ``Tr(g ∘ (id ⊗ h)) = Tr((id ⊗ h) ∘ g)``   (h : U → U)
    * **tightening**  ``Tr((id ⊗ k) ∘ f ∘ (id ⊗ j)) = k ∘ Tr(f) ∘ j``
    * **yanking**     ``Tr^U(σ_{U,U}) = id_U``

ENCODING — morphisms are matrices
    Tensor networks over ``concat`` wiring are linear-algebraic, so a
    morphism ``f : X ⊗ U → Y ⊗ U`` is a 2-D block matrix.  Convention:
    the FEEDBACK wire occupies the FIRST ``usize`` columns of the input
    and the FIRST ``usize`` rows of the output:

        f : U⊗X → U⊗Y      f = [[S, R], [Q, P]]
        u' = S·u + R·x     y  = Q·u + P·x

    ``trace(f, usize)`` is the trace in FDVect — the linear fixpoint
    (linear-fractional transform / Schur complement):

        Tr(f) = P + Q·(I − S)⁻¹·R

    For contractive or nilpotent ``S`` the resolvent is the Neumann
    series ``Σ Sᵏ`` — the closed/parallel form of the same iteration.
    Recurrences enter via *time-extended* wires: over a horizon ``T``,
    ``h_t = A·h_{t−1} + B·x_t`` is ``trace([[Z·A_blk, R], [I, 0]])``
    with ``Z`` the strictly-lower block shift (``Zᵀ = 0``) — the
    fixpoint is the unrolled scan EXACTLY, no convergence qualifier.

OPS
    ``trace(f, usize)`` — the trace; ``usize`` is an int, or a tuple of
        ints giving the feedback wire's factorisation (vanishing-2).
    ``bdiag(f, g)``     — ``f ⊕ g``, the monoidal product of linear
        maps under concat wiring (``torch.block_diag``).
    ``parl(f, g)``      — ``f ⊗ g`` re-laid so BOTH feedback wires stay
        first: ``[u_f; u_g; x; c] → [u'_f; u'_g; y; d]``.  This is the
        product superposing needs; plain ``bdiag`` interleaves the
        feedback and data wires.
    ``eye(dim=n)``      — the identity map ``I_n`` (a constant morphism).
    ``cswap(d1, d2)``   — the symmetry ``σ : [a; b] ↦ [b; a]``
        (a permutation matrix; yanking needs it).
    ``inv(m)``          — matrix inverse, for the closed-form law.

AXIOMS AS REWRITES — every side of every equation above is a single
term, so all five JSV axioms plus the LFT definition are LOCAL
rewrites:

    ``tr_vanish_unit``    usize = 0   →  f
    ``tr_vanish_split``   usize = (u,v) → nested traces
    ``tr_vanish_merge``   nested traces → product-wire trace
    ``tr_superpose``      joint loop over ``parl`` → ``bdiag`` of
                          independent traces  (channel splitting —
                          independent recurrence channels parallelize)
    ``tr_slide``          h on the loop input  ↔  h on the loop output
    ``tr_tighten_out``    ``Tr((I⊕k)·f) = k·Tr(f)``   post-context out
    ``tr_tighten_in``     ``Tr(f·(I⊕j)) = Tr(f)·j``   pre-context out
    ``tr_yank``           ``Tr(σ_{U,U}) = I_U``
    ``tr_expand``         ``Tr(f) → P + Q·(I−S)⁻¹·R``  (iterative →
                          closed form; the inverse rides the feedback
                          block — this is the law that lands a trace in
                          ordinary matmul/add/inv algebra)
    ``tr_collapse``       closed form → ``Tr(f)``  (closed → iterative)

WHAT IS *NOT* EXPRESSIBLE AS A LOCAL REWRITE (documented, not hacked):

* **Delay-loop trace.**  Executing ``h_t = f(h_{t−1}, x_t)`` stepwise
  with an init state is a trace in a category of *stream* functions.
  The fixpoint trace here only reaches it through the time-extended
  (shift) encoding — where it is exact.  An applied form
  ``trace(f, s0, x)`` carrying an init state would need a second op
  family, and its vanishing-2 (nested delay loops) is NOT local: the
  inner loop's input wire mentions the outer loop's state, which a
  positional-argument term cannot express.
* **The trm ↔ apply bridge.**  Recognising ``Tr(F)`` as
  ``apply(aff_composeᵀ…, h0)`` is a series/fold equality — the same
  kind of non-local pairing ``pair_shared_input_linears`` performs as
  a pass, not a lhs→rhs rule (it also needs nilpotency detection to
  terminate the series).  Verified numerically in tests/test_trace.py;
  not a rewrite.
* **Yanking is fine** here because ``cswap`` exists as a literal
  permutation morphism; more exotic cyclic wiring (traces over
  swapped/self-intersecting ports) would need wire-rewiring ops we
  deliberately do not grow.
* **Bodies are linear maps.**  Affine bodies ``h ↦ A·h + c`` encode by
  augmenting wires with a constant-1 block (the standard homogeneous
  trick); genuinely nonlinear bodies need a function-valued object the
  positional IR lacks.
"""

from typing import Any

import torch

from catopt.egraph import Rewrite
from catopt.ir import Op, op_def
from catopt.rules import R

__all__ = [
    "TORCH_BINDINGS",
    "TRACE_LAWS",
    "TR_COLLAPSE",
    "TR_EXPAND",
    "TR_SLIDE",
    "TR_SLIDE_REV",
    "TR_SUPERPOSE",
    "TR_SUPERPOSE_REV",
    "TR_TIGHTEN_IN",
    "TR_TIGHTEN_IN_REV",
    "TR_TIGHTEN_OUT",
    "TR_TIGHTEN_OUT_REV",
    "TR_VANISH_MERGE",
    "TR_VANISH_SPLIT",
    "TR_VANISH_UNIT",
    "TR_YANK",
]


# ---------------------------------------------------------------------------
#  Generator registration (documentary — the term machinery does not
#  consult _OP_REGISTRY; aff/om carriers are likewise unregistered in
#  ir.py's predefined list, these calls only add registry entries)
# ---------------------------------------------------------------------------

op_def(
    "trace",
    1,
    1,
    law="Tr^U(f): feedback of the first-usize output block into the "
    "first-usize input block; in FDVect the linear fixpoint "
    "P + Q(I−S)⁻¹R.",
)
op_def(
    "bdiag",
    2,
    1,
    law="Monoidal product of linear maps: f ⊗ g is block-diagonal "
    "under concat wiring.",
)
op_def(
    "parl",
    2,
    1,
    law="Tensor product re-laid to keep both feedback wires first — "
    "the product the superposing axiom traces over.",
)
op_def("eye", 0, 1, law="Identity morphism id_n (a constant matrix).")
op_def(
    "cswap",
    0,
    1,
    law="Symmetry σ : [a;b] ↦ [b;a] — a permutation matrix.",
)
op_def(
    "inv",
    1,
    1,
    law="Matrix inverse; appears only inside the closed form of a "
    "trace (the resolvent (I−S)⁻¹).",
)


# ---------------------------------------------------------------------------
#  Small shared helpers
# ---------------------------------------------------------------------------


def _shape_of(t: Any):
    """Best-effort shape of a bound term (delegates to cost model)."""
    from catopt.typing import _shape_of as _so

    return _so(t)


def _usize_total(u: Any) -> int:
    """Total traced size: int usize, or the sum of a tuple factorisation."""
    if isinstance(u, (list, tuple)):
        return int(sum(int(x) for x in u))
    if isinstance(u, (int, float)):
        return int(u)
    return 0


def _dim2(t: Any):
    """Concrete 2-D shape of a bound term, or None."""
    s = _shape_of(t)
    if (
        isinstance(s, tuple)
        and len(s) == 2
        and all(isinstance(d, int) for d in s)
    ):
        return s
    return None


def _both_int(bound: dict, *ks: str) -> bool:
    return all(isinstance(bound.get(k), int) for k in ks)


# ---------------------------------------------------------------------------
#  Vanishing —  Tr^I = id   and   Tr^{U⊗V} = Tr^V ∘ Tr^U
# ---------------------------------------------------------------------------

TR_VANISH_UNIT = R(
    "tr_vanish_unit",
    Op.make("trace", "f", usize="U"),
    "f",
    law="Tr^I(f) = f — tracing over the monoidal unit (a zero-width "
    "feedback wire) is the identity.",
    check=lambda b: _usize_total(b.get("$attr:U")) == 0,
)


def _usize_split2(bound: dict):
    u = bound.get("$attr:UV")
    if (
        isinstance(u, (list, tuple))
        and len(u) == 2
        and all(isinstance(x, int) for x in u)
    ):
        return tuple(u)
    return None


def _check_vanish_split(bound: dict) -> bool:
    uv = _usize_split2(bound)
    if uv is None:
        return False
    fs = _dim2(bound.get("f"))
    return (
        fs is not None
        and fs[0] >= uv[0] + uv[1]
        and fs[1] >= uv[0] + uv[1]
    )


def _derive_vanish_split(bound: dict):
    uv = _usize_split2(bound)
    if uv is None:
        return None
    return {"$attr:DU": uv[0], "$attr:DV": uv[1]}


TR_VANISH_SPLIT = R(
    "tr_vanish_split",
    Op.make("trace", "f", usize="UV"),
    Op.make("trace", Op.make("trace", "f", usize="DU"), usize="DV"),
    law="Tr^{U⊗V}(f) = Tr^V(Tr^U(f)) — a product feedback wire "
    "decomposes into nested traces (feedback-first wiring makes "
    "the inner factor literally the leading block).",
    check=_check_vanish_split,
    derive=_derive_vanish_split,
)


def _check_vanish_merge(bound: dict) -> bool:
    if not _both_int(bound, "$attr:DU", "$attr:DV"):
        return False
    fs = _dim2(bound.get("f"))
    if fs is None:
        return False
    du, dv = bound["$attr:DU"], bound["$attr:DV"]
    return fs[0] >= du + dv and fs[1] >= du + dv


TR_VANISH_MERGE = R(
    "tr_vanish_merge",
    Op.make("trace", Op.make("trace", "f", usize="DU"), usize="DV"),
    Op.make("trace", "f", usize="UV"),
    law="Nested traces fuse into a trace over the product wire "
    "U⊗V — the reverse of tr_vanish_split.",
    check=_check_vanish_merge,
    derive=lambda b: {"$attr:UV": (b["$attr:DU"], b["$attr:DV"])},
)


# ---------------------------------------------------------------------------
#  Superposing —  Tr^{U⊗V}(f ⊗ g) = Tr^U(f) ⊗ Tr^V(g)
#
#  ``parl`` re-lays the tensor product so both feedback wires stay
#  first; tracing the joint loop then splits into a block-diagonal of
#  the two independent traces — independent recurrence channels that
#  may be scheduled in parallel.
# ---------------------------------------------------------------------------


def _check_superpose(bound: dict) -> bool:
    if not _both_int(bound, "$attr:DU", "$attr:DV"):
        return False
    du, dv = bound["$attr:DU"], bound["$attr:DV"]
    if _usize_total(bound.get("$attr:UV")) != du + dv:
        return False
    fs, gs = _dim2(bound.get("f")), _dim2(bound.get("g"))
    return (
        fs is not None
        and gs is not None
        and fs[0] >= du
        and fs[1] >= du
        and gs[0] >= dv
        and gs[1] >= dv
    )


TR_SUPERPOSE = R(
    "tr_superpose",
    Op.make(
        "trace", Op.make("parl", "f", "g", u1="DU", u2="DV"), usize="UV"
    ),
    Op.make(
        "bdiag",
        Op.make("trace", "f", usize="DU"),
        Op.make("trace", "g", usize="DV"),
    ),
    law="Superposing: Tr^{U⊗V}(f ⊗ g) = Tr^U(f) ⊗ Tr^V(g) — a joint "
    "loop over independent channels splits into parallel "
    "independent loops.  (With g untraced, u2 = 0: context that "
    "doesn't feed back simply rides along.)",
    check=_check_superpose,
)

TR_SUPERPOSE_REV = R(
    "tr_superpose_rev",
    Op.make(
        "bdiag",
        Op.make("trace", "f", usize="DU"),
        Op.make("trace", "g", usize="DV"),
    ),
    Op.make(
        "trace", Op.make("parl", "f", "g", u1="DU", u2="DV"), usize="UV"
    ),
    law="Reverse superposing: independent loops fuse into one joint "
    "loop — the eqsat-visible schedule/memory tradeoff.",
    check=lambda b: _both_int(b, "$attr:DU", "$attr:DV"),
    derive=lambda b: {"$attr:UV": (b["$attr:DU"], b["$attr:DV"])},
)


# ---------------------------------------------------------------------------
#  Sliding —  Tr(g·(h⊕I_in)) = Tr((h⊕I_out)·g)   (h : U → U)
#
#  Under feedback-first wiring, ``id ⊗ h`` is ``bdiag(h, eye)``: h acts
#  on the FIRST (feedback) block.  Sliding moves the loop-wire map
#  across the trace boundary — the key law for relocating computation
#  in and out of a loop.  Sound for the linear fixpoint:
#  Q·H·(I−SH)⁻¹·R = Q·(I−HS)⁻¹·H·R.
# ---------------------------------------------------------------------------


def _check_slide(bound: dict) -> bool:
    if not _both_int(bound, "$attr:DU", "$attr:DX"):
        return False
    du, dx = bound["$attr:DU"], bound["$attr:DX"]
    gs, hs = _dim2(bound.get("g")), _dim2(bound.get("h"))
    if gs is None or hs is None:
        return False
    return hs == (du, du) and gs[1] == du + dx and gs[0] > du


def _derive_slide(bound: dict):
    gs = _dim2(bound.get("g"))
    du = bound.get("$attr:DU")
    if gs is None or not isinstance(du, int) or gs[0] <= du:
        return None
    return {"$attr:DY": gs[0] - du}


_TR_SLIDE_LHS = Op.make(
    "trace",
    Op.make(
        "matmul", "g", Op.make("bdiag", "h", Op.make("eye", dim="DX"))
    ),
    usize="DU",
)

_TR_SLIDE_RHS = Op.make(
    "trace",
    Op.make(
        "matmul", Op.make("bdiag", "h", Op.make("eye", dim="DY")), "g"
    ),
    usize="DU",
)

TR_SLIDE = R(
    "tr_slide",
    _TR_SLIDE_LHS,
    _TR_SLIDE_RHS,
    law="Sliding: Tr(g·(h⊕I_in)) = Tr((h⊕I_out)·g) — a map on the "
    "feedback wire crosses the loop boundary.",
    check=_check_slide,
    derive=_derive_slide,
)


def _check_slide_rev(bound: dict) -> bool:
    if not _both_int(bound, "$attr:DU", "$attr:DY"):
        return False
    du, dy = bound["$attr:DU"], bound["$attr:DY"]
    gs, hs = _dim2(bound.get("g")), _dim2(bound.get("h"))
    if gs is None or hs is None:
        return False
    return hs == (du, du) and gs[0] == du + dy and gs[1] > du


def _derive_slide_rev(bound: dict):
    gs = _dim2(bound.get("g"))
    du = bound.get("$attr:DU")
    if gs is None or not isinstance(du, int) or gs[1] <= du:
        return None
    return {"$attr:DX": gs[1] - du}


TR_SLIDE_REV = R(
    "tr_slide_rev",
    _TR_SLIDE_RHS,
    _TR_SLIDE_LHS,
    law="Sliding, reverse direction.",
    check=_check_slide_rev,
    derive=_derive_slide_rev,
)


# ---------------------------------------------------------------------------
#  Tightening / naturality —
#      Tr((I_U ⊕ k)·f) = k·Tr(f)      (post-context exits the loop)
#      Tr(f·(I_U ⊕ j)) = Tr(f)·j      (pre-context exits the loop)
#
#  Under feedback-first wiring ``id_U ⊗ k`` is ``bdiag(eye(du), k)``:
#  the identity rides the FIRST (feedback) block, the context map the
#  data block.
# ---------------------------------------------------------------------------


def _check_tighten_out(bound: dict) -> bool:
    du = bound.get("$attr:DU")
    fs, ks = _dim2(bound.get("f")), _dim2(bound.get("k"))
    return (
        isinstance(du, int)
        and fs is not None
        and ks is not None
        and fs[0] == du + ks[1]
    )


TR_TIGHTEN_OUT = R(
    "tr_tighten_out",
    Op.make(
        "trace",
        Op.make(
            "matmul",
            Op.make("bdiag", Op.make("eye", dim="DU"), "k"),
            "f",
        ),
        usize="DU",
    ),
    Op.make("matmul", "k", Op.make("trace", "f", usize="DU")),
    law="Tightening/naturality: Tr((I⊕k)·f) = k·Tr(f) — a map on the "
    "output wire commutes out of the loop.",
    check=_check_tighten_out,
)

TR_TIGHTEN_OUT_REV = R(
    "tr_tighten_out_rev",
    Op.make("matmul", "k", Op.make("trace", "f", usize="DU")),
    Op.make(
        "trace",
        Op.make(
            "matmul",
            Op.make("bdiag", Op.make("eye", dim="DU"), "k"),
            "f",
        ),
        usize="DU",
    ),
    law="Tightening, reverse: push a post-context map into the loop.",
    check=_check_tighten_out,
)


def _check_tighten_in(bound: dict) -> bool:
    du = bound.get("$attr:DU")
    fs, js = _dim2(bound.get("f")), _dim2(bound.get("j"))
    return (
        isinstance(du, int)
        and fs is not None
        and js is not None
        and fs[1] == du + js[0]
    )


TR_TIGHTEN_IN = R(
    "tr_tighten_in",
    Op.make(
        "trace",
        Op.make(
            "matmul",
            "f",
            Op.make("bdiag", Op.make("eye", dim="DU"), "j"),
        ),
        usize="DU",
    ),
    Op.make("matmul", Op.make("trace", "f", usize="DU"), "j"),
    law="Tightening, input side: Tr(f·(I⊕j)) = Tr(f)·j — a map on the "
    "input wire commutes out of the loop.",
    check=_check_tighten_in,
)

TR_TIGHTEN_IN_REV = R(
    "tr_tighten_in_rev",
    Op.make("matmul", Op.make("trace", "f", usize="DU"), "j"),
    Op.make(
        "trace",
        Op.make(
            "matmul",
            "f",
            Op.make("bdiag", Op.make("eye", dim="DU"), "j"),
        ),
        usize="DU",
    ),
    law="Tightening, reverse: push a pre-context map into the loop.",
    check=_check_tighten_in,
)


# ---------------------------------------------------------------------------
#  Yanking —  Tr^U(σ_{U,U}) = id_U
#
#  With feedback-first wiring, cswap(d, d) has blocks S = 0, R = I,
#  Q = I, P = 0, so Tr = 0 + I·(I−0)⁻¹·I = I_d.  A pure crossover loop
#  collapses to the identity wire.
# ---------------------------------------------------------------------------

TR_YANK = R(
    "tr_yank",
    Op.make("trace", Op.make("cswap", d1="D", d2="D"), usize="D"),
    Op.make("eye", dim="D"),
    law="Yanking: Tr^U(σ_{U,U}) = id_U — a pure crossover loop "
    "collapses to the identity wire.",
    check=lambda b: isinstance(b.get("$attr:D"), int),
)


# ---------------------------------------------------------------------------
#  Closed form —  Tr(f) = P + Q·(I−S)⁻¹·R
#
#  The LFT definition as a rewrite: decompose f's blocks with split
#  projections and reassemble the resolvent form in ordinary
#  matmul/add/sub/inv algebra.  This is the local law that bridges
#  iterative and closed forms — and the direction that lets an
#  e-graph grow a ``trace`` node into tensor-algebra structure the
#  other carrier laws can see.
# ---------------------------------------------------------------------------


def _check_expand(bound: dict) -> bool:
    du = bound.get("$attr:DU")
    fs = _dim2(bound.get("f"))
    return (
        isinstance(du, int)
        and du > 0
        and fs is not None
        and fs[0] > du
        and fs[1] > du
    )


def _derive_expand(bound: dict):
    fs = _dim2(bound.get("f"))
    du = bound.get("$attr:DU")
    if (
        fs is None
        or not isinstance(du, int)
        or du <= 0
        or fs[0] <= du
        or fs[1] <= du
    ):
        return None
    return {"$attr:RS": (du, fs[0] - du), "$attr:CS": (du, fs[1] - du)}


def _blk(t: Any, sizes_attr: str, dim: int, idx: int) -> Op:
    return Op.make("split", t, sizes=sizes_attr, dim=dim, index=idx)


_ROWS_U = _blk("f", "RS", -2, 0)  # the u' rows  [:du]
_ROWS_Y = _blk("f", "RS", -2, 1)  # the y  rows  [du:]
_S = _blk(_ROWS_U, "CS", -1, 0)  # u → u'
_RM = _blk(_ROWS_U, "CS", -1, 1)  # x → u'
_Q = _blk(_ROWS_Y, "CS", -1, 0)  # u → y
_P = _blk(_ROWS_Y, "CS", -1, 1)  # x → y

_TR_EXPANDED = Op.make(
    "add",
    _P,
    Op.make(
        "matmul",
        _Q,
        Op.make(
            "matmul",
            Op.make(
                "inv", Op.make("sub", Op.make("eye", dim="DU"), _S)
            ),
            _RM,
        ),
    ),
)

TR_EXPAND = R(
    "tr_expand",
    Op.make("trace", "f", usize="DU"),
    _TR_EXPANDED,
    law="Closed form of the trace: Tr(f) = P + Q·(I−S)⁻¹·R — the "
    "resolvent lives on the feedback block.  The local bridge "
    "from iterative to closed/parallel form.",
    check=_check_expand,
    derive=_derive_expand,
)


def _check_collapse(bound: dict) -> bool:
    """The four block projections must come from one f with matching
    usize: RS = (du, dy), CS = (du, dx), eye dim = du."""
    du = bound.get("$attr:DU")
    rs, cs = bound.get("$attr:RS"), bound.get("$attr:CS")
    if not (isinstance(du, int) and du > 0):
        return False
    if not (
        isinstance(rs, (list, tuple))
        and len(rs) == 2
        and isinstance(cs, (list, tuple))
        and len(cs) == 2
    ):
        return False
    return rs[0] == du and cs[0] == du


TR_COLLAPSE = R(
    "tr_collapse",
    _TR_EXPANDED,
    Op.make("trace", "f", usize="DU"),
    law="Closed → iterative: the resolvent form folds back into a "
    "single trace node.",
    check=_check_collapse,
)


# ---------------------------------------------------------------------------
#  Law set
# ---------------------------------------------------------------------------

#: The traced-monoidal axioms as a saturation set.  All rules are
#: shape-checked on the bound matrices — a law that would fire on
#: ill-typed wiring is vetoed, matching the codebase's convention
#: (cf. LINEAR_*_SCALE checks in rules.py).
TRACE_LAWS: list[Rewrite] = [
    TR_VANISH_UNIT,
    TR_VANISH_SPLIT,
    TR_VANISH_MERGE,
    TR_SUPERPOSE,
    TR_SUPERPOSE_REV,
    TR_SLIDE,
    TR_SLIDE_REV,
    TR_TIGHTEN_OUT,
    TR_TIGHTEN_OUT_REV,
    TR_TIGHTEN_IN,
    TR_TIGHTEN_IN_REV,
    TR_YANK,
    TR_EXPAND,
    TR_COLLAPSE,
]


# ---------------------------------------------------------------------------
#  Torch bindings — registered into torch_bridge's op table (additive;
#  the table is a plain dict and IRModule._eval looks ops up by name).
#  All verify fp64-exact: the fixpoint is computed by solve, not
#  iterated.
# ---------------------------------------------------------------------------


def _trace_torch(f: torch.Tensor, *a, **kw) -> torch.Tensor:
    """Tr(f) = P + Q·(I−S)⁻¹·R — the linear fixpoint (LFT).

    The feedback wire is the FIRST ``usize`` columns of the input and
    FIRST ``usize`` rows of the output; ``usize`` may be a tuple, in
    which case its sum is the traced width.
    """
    du = _usize_total(kw.get("usize", 0))
    if du <= 0 or not (isinstance(f, torch.Tensor) and f.dim() == 2):
        return f
    S, Rm = f[:du, :du], f[:du, du:]
    Q, P = f[du:, :du], f[du:, du:]
    eye = torch.eye(du, dtype=f.dtype, device=f.device)
    return P + Q @ torch.linalg.solve(eye - S, Rm)


def _parl_torch(
    f: torch.Tensor, g: torch.Tensor, *a, **kw
) -> torch.Tensor:
    """f ⊗ g re-laid to keep BOTH feedback wires first.

    Input  [u_f(u1); u_g(u2); x_f; x_g]
    Output [u'_f;      u'_g;      y_f; y_g]
    """
    u1 = int(kw.get("u1", 0))
    u2 = int(kw.get("u2", 0))
    rf, cf = int(f.shape[0]), int(f.shape[1])
    rg, cg = int(g.shape[0]), int(g.shape[1])
    dyf, dxf = rf - u1, cf - u1
    dyg, dxg = rg - u2, cg - u2
    du, dx, dy = u1 + u2, dxf + dxg, dyf + dyg
    out = f.new_zeros(du + dy, du + dx)
    # f's blocks on the outer (u_f, x_f) wires
    out[:u1, :u1] = f[:u1, :u1]  # S_f
    out[:u1, du : du + dxf] = f[:u1, u1:]  # R_f
    out[du : du + dyf, :u1] = f[u1:, :u1]  # Q_f
    out[du : du + dyf, du : du + dxf] = f[u1:, u1:]  # P_f
    # g's blocks on the inner (u_g, x_g) wires
    out[u1:du, u1:du] = g[:u2, :u2]  # S_g
    out[u1:du, du + dxf :] = g[:u2, u2:]  # R_g
    out[du + dyf :, u1:du] = g[u2:, :u2]  # Q_g
    out[du + dyf :, du + dxf :] = g[u2:, u2:]  # P_g
    return out


def _cswap_torch(*a, **kw) -> torch.Tensor:
    """σ_{d1,d2} : [a; b] ↦ [b; a] — a permutation matrix."""
    d1 = int(kw.get("d1", 0))
    d2 = int(kw.get("d2", 0))
    m = torch.zeros(d1 + d2, d1 + d2, dtype=torch.get_default_dtype())
    m[:d2, d1:] = torch.eye(d2)
    m[d2:, :d1] = torch.eye(d1)
    return m


def _eye_torch(*a, **kw) -> torch.Tensor:
    d = int(kw.get("dim", kw.get("d", 1)))
    return torch.eye(d, dtype=torch.get_default_dtype())


#: Torch lowering bindings — an EXPORT, not an import-time mutation
#: (plan 0001 phase 2c): :class:`catopt.ops.OpTable` folds this dict in
#: via ``register``/``full()``; importing this module registers nothing
#: into ``torch_bridge._IR_TO_TORCH``.
TORCH_BINDINGS: dict[str, Any] = {
    "trace": _trace_torch,
    "bdiag": lambda *ts, **kw: torch.block_diag(*ts),
    "parl": _parl_torch,
    "eye": _eye_torch,
    "cswap": _cswap_torch,
    "inv": lambda t, *a, **kw: torch.linalg.inv(t),
}
