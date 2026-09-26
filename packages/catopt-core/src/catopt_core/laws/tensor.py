"""Tensor-algebra laws: the equational rewrite surface.

Each rule is :class:`~catopt_core.egraph.Rewrite` with an LHS pattern (match)
and an RHS pattern (replacement).  Metavariables are Python ``str``
objects that appear as leaves in the pattern trees.

Groups:
* **Monoid laws** — commutativity and associativity of add/mul.
* **Group laws** — identity, inverses, double-negation.
* **Bilinearity / naturality** — distributivity of matmul over add, and
  naturality of scalar multiplication w.r.t. linear maps.
* **Product structure** — the fused-projection laws (SwiGLU, QKV, GQA).
* **Softmax-attention fold** — the flash-attention transform.
* **Decomposition** — silu → x*sigmoid(x), subtraction → a + neg(b).

Collections:
* ``SIMPLIFICATION_RULES`` — basic algebraic simplification.
* ``CATEGORICAL_RULES`` — the categorical insight: distributivity,
  naturality, products, diagonal absorption, SDPA fold.
* ``ALL_RULES`` / :func:`all_rules` — the union.
"""

# ruff: noqa: RUF001 RUF002 RUF003 -- the law strings and docstrings use
# mathematical notation (σ, ⊗, ×) deliberately; ASCII would misstate it.

from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op
from catopt_core.laws.base import (
    R,
    _is_channel_scale,
    _is_row_scale,
    _is_scalar,
    _shape_of,
)

# ---------------------------------------------------------------------------
#  Monoid laws: commutativity & associativity
# ---------------------------------------------------------------------------

COMM_ADD = R(
    "comm_add",
    Op.make("add", "a", "b"),
    Op.make("add", "b", "a"),
    law="Commutativity in a symmetric monoidal category: σ ∘ (f ⊗ g) = g ⊗ f.",
)

COMM_MUL = R(
    "comm_mul",
    Op.make("mul", "a", "b"),
    Op.make("mul", "b", "a"),
    law="Hadamard product is commutative (SMC symmetry).",
)

ASSOC_ADD = R(
    "assoc_add",
    Op.make("add", "a", Op.make("add", "b", "c")),
    Op.make("add", Op.make("add", "a", "b"), "c"),
    law="Associativity of sequential composition in a category.",
)

ASSOC_MUL = R(
    "assoc_mul",
    Op.make("mul", "a", Op.make("mul", "b", "c")),
    Op.make("mul", Op.make("mul", "a", "b"), "c"),
    law="Associativity of parallel composition in a monoidal category.",
)


# ---------------------------------------------------------------------------
#  Identity / inverse laws (group structure)
# ---------------------------------------------------------------------------

ID_ADD = R(
    "id_add",
    Op.make("add", "a", Const(0)),
    "a",
    law="Additive identity: a + 0 = a.",
)

ID_MUL = R(
    "id_mul",
    Op.make("mul", "a", Const(1)),
    "a",
    law="Multiplicative identity: a * 1 = a.",
)

DOUBLE_NEG = R(
    "double_neg",
    Op.make("neg", Op.make("neg", "a")),
    "a",
    law="Double negation: ¬¬a = a (involution).",
)


# ---------------------------------------------------------------------------
#  Subtraction and negation
# ---------------------------------------------------------------------------

SUB_TO_ADD = R(
    "sub_to_add",
    Op.make("sub", "a", "b"),
    Op.make("add", "a", Op.make("neg", "b")),
    law="Subtraction as addition of inverse: a - b = a + (-b).",
)


# ---------------------------------------------------------------------------
#  Decompositions
# ---------------------------------------------------------------------------

SILU_EXPAND = R(
    "silu_expand",
    Op.make("silu", "x"),
    Op.make("mul", "x", Op.make("sigmoid", "x")),
    law="SiLU definition: silu(x) = x · σ(x).",
)

SQUARE_EXPAND = R(
    "square_expand",
    Op.make("square", "x"),
    Op.make("mul", "x", "x"),
    law="Self-composition: x² = x · x.",
)

# RMSNorm/pow family: connect x.pow(2) to the mul-based representation.
# Lets square-related and naturality rewrites see RMSNorm's x**2 term.
POW_TO_SQUARE = R(
    "pow_to_square",
    Op.make("pow", "x", Const(2)),
    Op.make("square", "x"),
    law="pow(x, 2) ≡ square(x) ≡ x·x (SwiGLU/RMSNorm bridge).",
)

SQUARE_TO_POW = R(
    "square_to_pow",
    Op.make("square", "x"),
    Op.make("pow", "x", Const(2)),
    law="Reverse: square(x) ≡ pow(x, 2) for shape/cost reasons.",
)

# SwiGLU bridge: the exported graph has silu(linear(...)) followed by
# mul with another linear(...).  Expanding silu exposes the common
# x*sigmoid(x) factor, which lets naturality/distributivity see the
# shared linear prefix.  Already have SILU_EXPAND; add the mul-form:
SILU_MUL_FORM = R(
    "silu_mul_form",
    Op.make("mul", Op.make("silu", "g"), "u"),
    Op.make("mul", Op.make("mul", "g", Op.make("sigmoid", "g")), "u"),
    law="SwiGLU: silu(g)*u = (g*sigmoid(g))*u (factor for prefix sharing).",
)


# ---------------------------------------------------------------------------
#  Distributivity / naturality (the categorical insight)
# ---------------------------------------------------------------------------

# matmul(W, a + b) = matmul(W, a) + matmul(W, b)
DISTRIBUTE_MUL = R(
    "distribute_matmul_over_add",
    Op.make("matmul", "W", Op.make("add", "a", "b")),
    Op.make(
        "add", Op.make("matmul", "W", "a"), Op.make("matmul", "W", "b")
    ),
    law="Distributivity of linear maps over addition (bilinearity).",
)

# add(matmul(W, a), matmul(W, b)) → matmul(W, add(a, b))  [reverse]
FACTOR_MUL = R(
    "factor_matmul",
    Op.make(
        "add", Op.make("matmul", "W", "a"), Op.make("matmul", "W", "b")
    ),
    Op.make("matmul", "W", Op.make("add", "a", "b")),
    law="Factoring common linear maps (reverse distributivity).",
)

# Right-side bilinearity.  In PyTorch, `x @ W` puts the WEIGHT second, so
# the existing left-side rules never match the exported form.  These close
# that gap.  All four require repeated metavariables, which the e-graph
# matcher now enforces as "same e-class" (soundness-tested).

# (a + b) @ W = a@W + b@W
RIGHT_DISTRIBUTE = R(
    "right_distribute_matmul",
    Op.make("matmul", Op.make("add", "a", "b"), "W"),
    Op.make(
        "add", Op.make("matmul", "a", "W"), Op.make("matmul", "b", "W")
    ),
    law="Bilinearity: linear maps distribute over addition in BOTH slots.",
)

# a@W + b@W = (a + b) @ W
RIGHT_FACTOR = R(
    "right_factor_matmul",
    Op.make(
        "add", Op.make("matmul", "a", "W"), Op.make("matmul", "b", "W")
    ),
    Op.make("matmul", Op.make("add", "a", "b"), "W"),
    law="Factor a shared right-weight (the slot `x @ W` uses).",
)

# THE WEIGHT-MERGE RULE.  x@W1 + x@W2 = x @ (W1 + W2): two projections of
# the SAME input collapse to one matmul on a summed weight.  This is the
# LoRA/adapter/model-soup merge that deployment tooling does by hand.
WEIGHT_FACTOR = R(
    "weight_factor_matmul",
    Op.make(
        "add", Op.make("matmul", "x", "W"), Op.make("matmul", "x", "W2")
    ),
    Op.make("matmul", "x", Op.make("add", "W", "W2")),
    law="Merge shared-input projections: x@W1 + x@W2 = x@(W1+W2).",
)

# x @ (W1 + W2) = x@W1 + x@W2  [reverse: expand for cost-model choice]
WEIGHT_DISTRIBUTE = R(
    "weight_distribute_matmul",
    Op.make("matmul", "x", Op.make("add", "W", "W2")),
    Op.make(
        "add", Op.make("matmul", "x", "W"), Op.make("matmul", "x", "W2")
    ),
    law="Reverse weight merge (lets eqsat weigh fused vs split forms).",
)


# ---------------------------------------------------------------------------
#  `linear` variants — torch.export emits F.linear for every nn.Linear, so
#  the matmul rules above never see the exported form.  F.linear(x, W) is
#  x @ W.T, so transposes distribute over add and flip matrix products.
# ---------------------------------------------------------------------------

# x@W1.T + x@W2.T = x @ (W1+W2).T   ->   linear(x, W1+W2)
WEIGHT_FACTOR_LINEAR = R(
    "weight_factor_linear",
    Op.make(
        "add", Op.make("linear", "x", "W"), Op.make("linear", "x", "W2")
    ),
    Op.make("linear", "x", Op.make("add", "W", "W2")),
    law="Merge shared-input nn.Linears: linear(x,W1)+linear(x,W2)"
    " = linear(x, W1+W2)  (transpose distributes over +).",
)

# linear(linear(x, A), B) = x @ A.T @ B.T = x @ (B@A).T = linear(x, B@A)
# NOTE the flipped order: fused weight is B @ A, not A @ B.
ASSOC_LINEAR = R(
    "assoc_linear",
    Op.make("linear", Op.make("linear", "x", "A"), "B"),
    Op.make("linear", "x", Op.make("matmul", "B", "A")),
    law="Compose stacked nn.Linears: fused weight is B @ A"
    " (transposes flip the product order).",
)

# ------------------------------------------------------------------
#  Biased composition — the affine-map law on the 3-ary `linear`.
#
#  torch.export emits nn.Linear(bias=True) as linear(x, W, b), so the
#  2-ary assoc_linear above never sees the most common stacked-Linear
#  graph.  The composed-bias law is still exact:
#
#      B(Ax + b1) + b2 = (BA)x + (B b1 + b2)
#
#  with the SAME transposed-order weight product B @ A, plus the push
#  of the inner bias through the outer map:  B·b1 + b2.
#
#  NOTE on the RHS spelling.  The textbook form is
#      linear(x, B@A, B@b1 + b2)
#  but that term is unselectable under the cost model: _shape_of has
#  no matvec case, so matmul(B, b1) on a rank-1 bias infers (o, h)
#  instead of (o,), which makes add(B·b1, b2) provably ill-typed
#  (_INVALID) whenever h != o and prices the whole composed member at
#  _INVALID_COST — it could never be extracted.  The associativity-
#  equivalent spelling below keeps every subterm well-typed for ALL
#  h, o: the matvec sits in the linear's bias SLOT (whose inferred
#  shape is ignored by the linear case) and b2 is added at the top
#  level, where the broadcast (…, o) + (o,) is valid.  Both
#  B-products are parameter-only, so _fold_weight_chains materialises
#  (B@A) and (B·b1) at compile time; the runtime keeps one GEMM, one
#  broadcast add, and the b2 leaf.
# ------------------------------------------------------------------


def _check_linear_bias_compose(bound: dict) -> bool:
    """Shape guard: the chain dims must compose for B·b1 + b2 to be
    well-typed — A (h, i), B (o, h), b1 (h,), b2 (o,) or scalar.

    The matcher cannot see tensor types; without this the rule would
    also fire on e-nodes whose "bias" slot holds a non-vector term.
    Unknown dims pass through as equalities on None (the rewrite is
    exact wherever the LHS is a real computation — the check only
    vetoes PROVABLE mismatches)."""
    x = _shape_of(bound.get("x"))
    a = _shape_of(bound.get("A"))
    b = _shape_of(bound.get("B"))
    b1 = _shape_of(bound.get("b1"))
    b2 = _shape_of(bound.get("b2"))
    if not (
        isinstance(a, tuple)
        and isinstance(b, tuple)
        and len(a) == 2
        and len(b) == 2
    ):
        return False
    h, i, o = a[0], a[1], b[0]
    if b[1] != h:  # B consumes A's out dim
        return False
    if not (isinstance(b1, tuple) and len(b1) == 1 and b1[0] == h):
        return False  # inner bias: exactly (h,)
    if b2 != () and not (
        isinstance(b2, tuple) and len(b2) == 1 and b2[0] == o
    ):
        return False  # outer bias: scalar|(o,)
    if not (  # noqa: SIM103
        isinstance(x, tuple) and len(x) >= 1 and x[-1] == i
    ):
        return False  # x feeds A's input dim
    return True


ASSOC_LINEAR_BIAS = R(
    "assoc_linear_bias",
    Op.make("linear", Op.make("linear", "x", "A", "b1"), "B", "b2"),
    Op.make(
        "add",
        Op.make(
            "linear",
            "x",
            Op.make("matmul", "B", "A"),
            Op.make("matmul", "B", "b1"),
        ),
        "b2",
    ),
    law="Affine-map composition: (B,b2)∘(A,b1) = (BA, B·b1 + b2).  "
    "Fused weight is B @ A (same transpose flip as assoc_linear); "
    "the fused bias is spelled linear(x, BA, B·b1) + b2 so both "
    "B-products fold at compile time — one GEMM plus one "
    "broadcast add at runtime.",
    check=_check_linear_bias_compose,
)

# Reverse: expand a fused affine member back into the biased chain —
# wins when the hidden dim sits below the oi/(i+o) break-even.
ASSOC_LINEAR_BIAS_REV = R(
    "assoc_linear_bias_rev",
    Op.make(
        "add",
        Op.make(
            "linear",
            "x",
            Op.make("matmul", "B", "A"),
            Op.make("matmul", "B", "b1"),
        ),
        "b2",
    ),
    Op.make("linear", Op.make("linear", "x", "A", "b1"), "B", "b2"),
    law="Reverse affine composition (eqsat weighs fused vs split).",
    check=_check_linear_bias_compose,
)

# a@W.T + b@W.T = (a+b)@W.T   ->   linear(add(a,b), W)
RIGHT_FACTOR_LINEAR = R(
    "right_factor_linear",
    Op.make(
        "add", Op.make("linear", "a", "W"), Op.make("linear", "b", "W")
    ),
    Op.make("linear", Op.make("add", "a", "b"), "W"),
    law="Factor a shared right-hand nn.Linear weight.",
)

# reverse of weight merge for `linear`
WEIGHT_DISTRIBUTE_LINEAR = R(
    "weight_distribute_linear",
    Op.make("linear", "x", Op.make("add", "W", "W2")),
    Op.make(
        "add", Op.make("linear", "x", "W"), Op.make("linear", "x", "W2")
    ),
    law="Expand a merged nn.Linear so eqsat can compare both forms.",
)

# ---------------------------------------------------------------------------
#  Product structure — the fused-projection rules.
#
#  Categorically: in a category with products, two maps f,g : X -> V with
#  the SAME source pair into a single map <f,g> : X -> V x V (the
#  universal property of the product, mediated by the diagonal Delta).
#  On tensors, pairing two nn.Linears on one input is concatenating the
#  weights along the output dim (one wide GEMM), and the projections
#  pi_i are zero-cost chunks.  This is the MergedColumnParallelLinear /
#  fused-QKV transformation deployment stacks perform manually; a tensor
#  compiler cannot produce it because it must RESHAPE PARAMETERS, which
#  lies outside kernel fusion.
# ---------------------------------------------------------------------------

# silu(x@A.T) * (x@B.T)  ->  y = x@[A;B].T ; silu(y[..., :d]) * y[..., d:]
# The fused linear term is shared (the e-graph stores it once); both
# chunk projections read the same e-class.
SWIGLU_FUSE = R(
    "swiglu_fuse",
    Op.make(
        "mul",
        Op.make("silu", Op.make("linear", "x", "A")),
        Op.make("linear", "x", "B"),
    ),
    Op.make(
        "mul",
        Op.make(
            "silu",
            Op.make(
                "chunk",
                Op.make(
                    "linear", "x", Op.make("concat", "A", "B", dim=0)
                ),
                chunks=2,
                dim=-1,
                index=0,
            ),
        ),
        Op.make(
            "chunk",
            Op.make("linear", "x", Op.make("concat", "A", "B", dim=0)),
            chunks=2,
            dim=-1,
            index=1,
        ),
    ),
    law="Product universal property: <f,g> = (f x g) . Delta.  Two "
    "projections of the same input are ONE GEMM into V x V, then "
    "project.  (Fused SwiGLU gate/up — MergedColumnParallelLinear.)",
)

# (x@A.T) * (x@B.T)  ->  chunk form without the gate nonlinearity.
# Covers GLU-style variants and any elementwise-mul pair of parallel
# projections.
PARALLEL_MUL_FUSE = R(
    "parallel_mul_fuse",
    Op.make(
        "mul", Op.make("linear", "x", "A"), Op.make("linear", "x", "B")
    ),
    Op.make(
        "mul",
        Op.make(
            "chunk",
            Op.make("linear", "x", Op.make("concat", "A", "B", dim=0)),
            chunks=2,
            dim=-1,
            index=0,
        ),
        Op.make(
            "chunk",
            Op.make("linear", "x", Op.make("concat", "A", "B", dim=0)),
            chunks=2,
            dim=-1,
            index=1,
        ),
    ),
    law="Pairing without a gate nonlinearity: mul(<pi1 f>, <pi2 g>) "
    "recovers the parallel-product form.",
)

# ---------------------------------------------------------------------------
#  Diagonal-scale naturality for `linear` — the RMSNorm-folding rules.
#
#  Two kinds of broadcast scale commute through F.linear differently:
#
#  * CHANNEL scale c ~ (in,) folds INTO the weight:
#        linear(x∘c, W) = (x∘c)@W.T = x@(W∘c).T = linear(x, W∘c)
#    This is the classic "fold the norm's affine gain into the next
#    linear" deployment trick — the diagonal D=diag(c) satisfies
#    (xD)W = x(DW), i.e. a right action absorbed at compile time.
#
#  * ROW scale r ~ (B,T,1) hoists OUT:
#        linear(x∘r, W) = diag(r)·(x@W.T) = linear(x,W) ∘ r
#    A left diagonal commutes with any linear map — naturality of the
#    scalar action.
#
#  CAVEAT: the matcher cannot check broadcast shapes.  Applied to a
#  term where c is sized `out` and out != in, the RHS is ill-typed;
#  the cost model prices provably-ill-typed terms at _INVALID_COST so
#  they can never win, and the verifier is the last line of defence.
# ---------------------------------------------------------------------------

LINEAR_CHANNEL_SCALE = R(
    "linear_channel_scale",
    Op.make("linear", Op.make("mul", "x", "c"), "W"),
    Op.make("linear", "x", Op.make("mul", "W", "c")),
    law="Channel scale is a right diagonal: (xD)W = x(DW).  Folds the "
    "norm's affine gain into the weight at compile time.",
    check=_is_channel_scale,
)

LINEAR_CHANNEL_SCALE_REV = R(
    "linear_channel_scale_rev",
    Op.make("linear", "x", Op.make("mul", "W", "c")),
    Op.make("linear", Op.make("mul", "x", "c"), "W"),
    law="Reverse channel-scale fold (eqsat compares both forms).",
    check=_is_channel_scale,
)

LINEAR_ROW_SCALE = R(
    "linear_row_scale",
    Op.make("linear", Op.make("mul", "x", "r"), "W"),
    Op.make("mul", Op.make("linear", "x", "W"), "r"),
    law="Row scale is a left diagonal: commutes through the linear map "
    "to the output (naturality of scalar action).",
    check=lambda b: _is_row_scale(b["r"]),
)

LINEAR_ROW_SCALE_REV = R(
    "linear_row_scale_rev",
    Op.make("mul", Op.make("linear", "x", "W"), "r"),
    Op.make("linear", Op.make("mul", "x", "r"), "W"),
    law="Reverse row-scale hoist (eqsat compares both forms).",
    check=lambda b: _is_row_scale(b["r"]),
)


# ---------------------------------------------------------------------------
#  Fused QKV — the product rule applied to attention's three projections.
#
#  sdpa( f(linear(x,Q)), f(linear(x,K)), f(linear(x,V)) )
#    where f = transpose ∘ view is the head-splitting view
#  -->
#  y = linear(x, cat(cat(Q,K),V)) ;
#  sdpa( f(chunk0(y)), f(chunk1(y)), f(chunk2(y)) )
#
#  The view shape and the transpose dims are ATTRIBUTE metavariables
#  (string values in the pattern bind the node's concrete attrs), which
#  is what makes this rule shape-polymorphic.
# ---------------------------------------------------------------------------


def _head(t: str) -> Op:
    """The head-splitting view: view(t, S) then transpose(1, 2)."""
    return Op.make(
        "transpose", Op.make("reshape", t, shape="S"), dim0=1, dim1=2
    )


QKV_FUSE = R(
    "qkv_fuse",
    Op.make(
        "sdpa",
        _head(Op.make("linear", "x", "Q")),
        _head(Op.make("linear", "x", "K")),
        _head(Op.make("linear", "x", "V")),
        scale="SC",
    ),
    Op.make(
        "sdpa",
        _head(
            Op.make(
                "chunk",
                Op.make(
                    "linear",
                    "x",
                    Op.make(
                        "concat",
                        Op.make("concat", "Q", "K", dim=0),
                        "V",
                        dim=0,
                    ),
                ),
                chunks=3,
                dim=-1,
                index=0,
            )
        ),
        _head(
            Op.make(
                "chunk",
                Op.make(
                    "linear",
                    "x",
                    Op.make(
                        "concat",
                        Op.make("concat", "Q", "K", dim=0),
                        "V",
                        dim=0,
                    ),
                ),
                chunks=3,
                dim=-1,
                index=1,
            )
        ),
        _head(
            Op.make(
                "chunk",
                Op.make(
                    "linear",
                    "x",
                    Op.make(
                        "concat",
                        Op.make("concat", "Q", "K", dim=0),
                        "V",
                        dim=0,
                    ),
                ),
                chunks=3,
                dim=-1,
                index=2,
            )
        ),
        scale="SC",
    ),
    law="Triple pairing <q,k,v> : X -> V^3 — three projections of the "
    "same input are ONE GEMM into the product space, then three "
    "zero-cost chunk projections.  (Fused QKV.)",
)


# ---------------------------------------------------------------------------
#  Asymmetric fused QKV (GQA): q/k/v projections with DIFFERENT output
#  dims.  The product law is identical; only the projections differ —
#  `split` with explicit sizes instead of `chunk`.  The sizes are not
#  present anywhere in the LHS, so a `derive` hook computes them from
#  the bound weight shapes.  (vLLM's QKVParallelLinear.)
# ---------------------------------------------------------------------------


def _head_v(t: Any, shape_var: str) -> Op:
    """Head view with a per-projection shape metavariable."""
    return Op.make(
        "transpose",
        Op.make("reshape", t, shape=shape_var),
        dim0=1,
        dim1=2,
    )


def _derive_split_sizes(bound: dict) -> dict | None:
    """sizes = (|Q|, |K|, |V|) — each bound weight's output dim."""
    sizes = []
    for k in ("Q", "K", "V"):
        w = bound.get(k)
        shape = getattr(getattr(w, "typ", None), "shape", None)
        if not shape or any(d is None for d in shape):
            return None
        sizes.append(shape[0])
    return {"$attr:SZ": tuple(sizes)}


_QKV_CAT = Op.make(
    "concat", Op.make("concat", "Q", "K", dim=0), "V", dim=0
)

QKV_FUSE_ASYM = R(
    "qkv_fuse_asym",
    Op.make(
        "sdpa",
        _head_v(Op.make("linear", "x", "Q"), "S1"),
        _head_v(Op.make("linear", "x", "K"), "S2"),
        _head_v(Op.make("linear", "x", "V"), "S3"),
        scale="SC",
        enable_gqa="G",
    ),
    Op.make(
        "sdpa",
        _head_v(
            Op.make(
                "split",
                Op.make("linear", "x", _QKV_CAT),
                sizes="SZ",
                dim=-1,
                index=0,
            ),
            "S1",
        ),
        _head_v(
            Op.make(
                "split",
                Op.make("linear", "x", _QKV_CAT),
                sizes="SZ",
                dim=-1,
                index=1,
            ),
            "S2",
        ),
        _head_v(
            Op.make(
                "split",
                Op.make("linear", "x", _QKV_CAT),
                sizes="SZ",
                dim=-1,
                index=2,
            ),
            "S3",
        ),
        scale="SC",
        enable_gqa="G",
    ),
    law="Asymmetric triple pairing: the same product law as qkv_fuse, "
    "but the three projections have different output dims — one "
    "GEMM, three uneven split views.  (GQA fused QKV.)",
    derive=_derive_split_sizes,
)


# ---------------------------------------------------------------------------
#  Copy-map absorption:  unsqueeze -> expand -> reshape  is the diagonal
#  Delta_r (duplicate along a new axis, then merge it back = PyTorch's
#  repeat_kv / repeat_interleave).  The diagonal is a *natural* map: it
#  can be pushed inside a consumer that implements the broadcast
#  internally.  SDPA's enable_gqa flag IS that consumer — feeding it the
#  unexpanded k/v computes the same attention without materialising
#  the duplicated heads.  A term-local tensor pass cannot see this: the
#  pattern lives across three view ops plus a fused kernel flag.
# ---------------------------------------------------------------------------


def _check_repeat_chain(bound: dict, pre: str) -> bool:
    """reshape(expand(unsqueeze(t, d))) must be exactly repeat_interleave
    on dim d-1: unsqueeze inserts a 1, expand broadcasts only that dim
    by r, and the reshape merges dims d-1,d into one."""
    from catopt_core.typing import _shape_of as _so

    d = bound.get(f"$attr:UD{pre}")
    es = bound.get(f"$attr:ES{pre}")
    rs = bound.get(f"$attr:RS{pre}")
    base = bound.get(pre)
    bs = _so(base)
    if not (
        isinstance(d, int)
        and isinstance(es, tuple)
        and isinstance(rs, tuple)
        and isinstance(bs, tuple)
    ):
        return False
    if any(x is None for x in bs):
        return False
    nd = len(bs)
    d = d % (nd + 1)
    us = bs[:d] + (1,) + bs[d:]  # noqa: RUF005
    if len(es) != len(us) or len(rs) != nd or d == 0:
        return False
    r = es[d]
    if not isinstance(r, int) or r <= 1:
        return False
    if any(es[i] != us[i] for i in range(len(us)) if i != d):
        return False  # expand may only grow the inserted dim
    merged = us[: d - 1] + ((us[d - 1] or 0) * r,) + us[d + 1 :]  # noqa: RUF005
    return rs == merged


def _check_gqa_absorb(bound: dict) -> bool:
    """Both k and v must be repeat-chains with the SAME repeat factor r,
    and q's head count must equal kv_heads * r."""
    from catopt_core.typing import _shape_of as _so

    for side in ("k", "v"):
        if not _check_repeat_chain(bound, side):
            return False
    if bound["$attr:ESk"] != bound["$attr:ESv"]:
        return False
    d = bound["$attr:UDk"] % (len(_so(bound["k"])) + 1)
    r = bound["$attr:ESk"][d]
    qs, ks = _so(bound["q"]), _so(bound["k"])
    if not (
        isinstance(qs, tuple)
        and isinstance(ks, tuple)
        and len(qs) >= 2
        and len(ks) >= 2
    ):
        return False
    if None in qs or None in ks:
        return False
    return qs[-2] == ks[-2] * r  # hq == hkv * n_rep


_REPEAT_KV = Op.make(
    "transpose",
    Op.make(
        "reshape",
        Op.make(
            "expand", Op.make("unsqueeze", "k", dim="UDk"), shape="ESk"
        ),
        shape="RSk",
    ),
    dim0=1,
    dim1=2,
)

_REPEAT_V = Op.make(
    "transpose",
    Op.make(
        "reshape",
        Op.make(
            "expand", Op.make("unsqueeze", "v", dim="UDv"), shape="ESv"
        ),
        shape="RSv",
    ),
    dim0=1,
    dim1=2,
)

GQA_ABSORB = R(
    "gqa_absorb_repeat",
    Op.make(
        "sdpa",
        Op.make("transpose", "q", dim0=1, dim1=2),
        _REPEAT_KV,
        _REPEAT_V,
        arg4="D",
        arg5="C",
    ),
    Op.make(
        "sdpa",
        Op.make("transpose", "q", dim0=1, dim1=2),
        Op.make("transpose", "k", dim0=1, dim1=2),
        Op.make("transpose", "v", dim0=1, dim1=2),
        arg4="D",
        arg5="C",
        arg7=True,
    ),
    law="The diagonal is natural: unsqueeze->expand->reshape copies each "
    "kv head r times (repeat_kv).  SDPA implements that copy inside "
    "the kernel via enable_gqa — pushing Delta into the consumer "
    "deletes the materialisation entirely.",
    check=_check_gqa_absorb,
)


# ---------------------------------------------------------------------------
#  Softmax-attention fold — the flash-attention transform.
#
#  softmax(q @ k^T * s [+ mask | masked_fill(mask, -inf)]) @ v
#      ==  sdpa(q, k, v, attn_mask=..., scale=s)
#
#  This is the *definition* of scaled_dot_product_attention, so the fold
#  is sound for ANY mask term: additive masks pass straight through,
#  boolean fill masks invert to keep-masks via logical_not.  Inductor
#  fuses the softmax elementwise chain but (for the masked_fill form)
#  never recognises the enclosing matmul pair as SDPA — the pattern
#  spans a softmax nonlinearity and two matmuls.
# ---------------------------------------------------------------------------

_QK_SCORES = Op.make(
    "matmul", "Q", Op.make("transpose", "K", dim0="TD1", dim1="TD2")
)


def _const_val(t):
    return getattr(t, "value", None)


def _check_score_transpose(bound) -> bool:
    """k must be transposed on its last two dims — matmul(q, k^T)."""
    ks = _shape_of(bound.get("K"))
    d1, d2 = bound.get("$attr:TD1"), bound.get("$attr:TD2")
    if not (
        isinstance(ks, tuple)
        and all(isinstance(x, int) for x in ks)
        and isinstance(d1, int)
        and isinstance(d2, int)
    ):
        return False
    nd = len(ks)
    return {d1 % nd, d2 % nd} == {nd - 2, nd - 1}


def _check_softmax_dim(bound) -> bool:
    """softmax must be over the last dim (keys) of the score matrix."""
    sd = bound.get("$attr:SD")
    qs = _shape_of(bound.get("Q"))
    if not (isinstance(sd, int) and isinstance(qs, tuple) and qs):
        return False
    return sd % len(qs) == len(qs) - 1


def _scale_of(bound):
    s = _const_val(bound.get("S"))
    return float(s) if isinstance(s, (int, float)) else None


def _check_sdpa_base(bound) -> bool:
    return _check_score_transpose(bound) and _check_softmax_dim(bound)


def _check_sdpa_scaled(bound) -> bool:
    return _check_sdpa_base(bound) and _scale_of(bound) is not None


def _check_sdpa_mf(bound) -> bool:
    if not _check_sdpa_base(bound):
        return False
    f = _const_val(bound.get("F"))
    return isinstance(f, (int, float)) and f < -1e30


def _check_sdpa_mf_scaled(bound) -> bool:
    return _check_sdpa_mf(bound) and _scale_of(bound) is not None


def _derive_scale_mul(bound):
    s = _scale_of(bound)
    return {"$attr:SC": s} if s is not None else None


def _derive_scale_div(bound):
    s = _scale_of(bound)
    return {"$attr:SC": 1.0 / s} if s is not None else None


def _derive_scale_one(bound):
    return {"$attr:SC": 1.0}


def _make_sdpa_fold_rules() -> list:
    """6 mask/scale forms × optional eval-mode dropout wrapper."""
    out = []
    scaled = (
        (
            "mul",
            lambda: Op.make("mul", _QK_SCORES, "S"),
            _check_sdpa_scaled,
            _check_sdpa_mf_scaled,
            _derive_scale_mul,
        ),
        (
            "div",
            lambda: Op.make("div", _QK_SCORES, "S"),
            _check_sdpa_scaled,
            _check_sdpa_mf_scaled,
            _derive_scale_div,
        ),
        (
            "",
            lambda: _QK_SCORES,
            _check_sdpa_base,
            _check_sdpa_mf,
            _derive_scale_one,
        ),
    )
    wraps = (
        ("", lambda sm: sm),
        (
            "_drop",
            lambda sm: Op.make("dropout", sm, p="DP", train="DT"),
        ),
    )
    for sname, scores, check_add, check_mf, derive in scaled:
        for wname, wrap in wraps:
            sm = lambda inner: wrap(  # noqa: E731, B023
                Op.make("softmax", inner, dim="SD")
            )
            out.append(
                R(
                    f"sdpa_fold_add{sname}{wname}",
                    Op.make(
                        "matmul", sm(Op.make("add", scores(), "M")), "V"
                    ),
                    Op.make("sdpa", "Q", "K", "V", "M", scale="SC"),
                    law="softmax(qk^T s + m) v is sdpa — the additive mask is "
                    "the kernel's attn_mask argument.",
                    check=check_add,
                    derive=derive,
                )
            )
            out.append(
                R(
                    f"sdpa_fold_masked_fill{sname}{wname}",
                    Op.make(
                        "matmul",
                        sm(Op.make("masked_fill", scores(), "MK", "F")),
                        "V",
                    ),
                    Op.make(
                        "sdpa",
                        "Q",
                        "K",
                        "V",
                        Op.make("logical_not", "MK"),
                        scale="SC",
                    ),
                    law="masked_fill(m, -inf) before softmax is a boolean "
                    "attn_mask — logical_not turns the fill-mask into "
                    "SDPA's keep-mask.",
                    check=check_mf,
                    derive=derive,
                )
            )
    return out


SDPA_FOLD_RULES: list = _make_sdpa_fold_rules()

# matmul(W, mul(x, c)) = mul(matmul(W, x), c)
# KEY RULE: naturality of scalar multiplication w.r.t. linear maps.
# Lets the optimizer slide an elementwise scaling past a matmul.
NATURALITY_SCALAR = R(
    "naturality_scalar",
    Op.make("matmul", "W", Op.make("mul", "x", "c")),
    Op.make("mul", Op.make("matmul", "W", "x"), "c"),
    law="Naturality: scalar multiplication commutes with linear maps.",
    check=lambda b: _is_scalar(b["c"]),
)

NATURALITY_SCALAR_REV = R(
    "naturality_scalar_rev",
    Op.make("mul", Op.make("matmul", "W", "x"), "c"),
    Op.make("matmul", "W", Op.make("mul", "x", "c")),
    law="Reverse naturality: pull scalar into the matmul's input.",
    check=lambda b: _is_scalar(b["c"]),
)

# (A @ B) @ C = A @ (B @ C)  — associativity of composition
ASSOC_MATMUL = R(
    "assoc_matmul",
    Op.make("matmul", "A", Op.make("matmul", "B", "C")),
    Op.make("matmul", Op.make("matmul", "A", "B"), "C"),
    law="Associativity of composition in a category: (f∘g)∘h = f∘(g∘h).",
)

# Reverse direction: explore the other association
ASSOC_MATMUL_REV = R(
    "assoc_matmul_rev",
    Op.make("matmul", Op.make("matmul", "A", "B"), "C"),
    Op.make("matmul", "A", Op.make("matmul", "B", "C")),
    law="Reverse associativity: f∘(g∘h) = (f∘g)∘h.",
)
# ---------------------------------------------------------------------------
#  Rule collections
# ---------------------------------------------------------------------------

#: Rules that implement basic algebraic simplification (monoid, group).
SIMPLIFICATION_RULES: list[Rewrite] = [
    COMM_ADD,
    COMM_MUL,
    ASSOC_ADD,
    ASSOC_MUL,
    ID_ADD,
    ID_MUL,
    DOUBLE_NEG,
    SUB_TO_ADD,
    SILU_EXPAND,
    SILU_MUL_FORM,
    SQUARE_EXPAND,
    POW_TO_SQUARE,
    SQUARE_TO_POW,
]

#: Rules that implement the categorical insight: distributivity and naturality.
CATEGORICAL_RULES: list[Rewrite] = [
    DISTRIBUTE_MUL,
    FACTOR_MUL,
    RIGHT_DISTRIBUTE,
    RIGHT_FACTOR,
    WEIGHT_FACTOR,
    WEIGHT_DISTRIBUTE,
    # nn.Linear / F.linear forms (what torch.export actually emits)
    WEIGHT_FACTOR_LINEAR,
    WEIGHT_DISTRIBUTE_LINEAR,
    RIGHT_FACTOR_LINEAR,
    ASSOC_LINEAR,
    ASSOC_LINEAR_BIAS,
    ASSOC_LINEAR_BIAS_REV,
    NATURALITY_SCALAR,
    NATURALITY_SCALAR_REV,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    # Product structure (fused projections)
    SWIGLU_FUSE,
    PARALLEL_MUL_FUSE,
    QKV_FUSE,
    QKV_FUSE_ASYM,
    # Diagonal-scale naturality (norm folding)
    LINEAR_CHANNEL_SCALE,
    LINEAR_CHANNEL_SCALE_REV,
    LINEAR_ROW_SCALE,
    LINEAR_ROW_SCALE_REV,
    # Diagonal-map absorption (copy pushed inside the kernel)
    GQA_ABSORB,
    # Softmax-attention fold (flash-attention transform)
    *SDPA_FOLD_RULES,
]
#: All rules combined.
ALL_RULES: list[Rewrite] = SIMPLIFICATION_RULES + CATEGORICAL_RULES


def all_rules() -> list[Rewrite]:
    """Return a fresh list of all rewrite rules."""
    return list(ALL_RULES)
