"""Rewrite rules derived from categorical / algebraic laws.

Each rule is :class:`~catopt.egraph.Rewrite` with an LHS pattern (match)
and an RHS pattern (replacement).  Metavariables are Python ``str``
objects that appear as leaves in the pattern trees.

Groups:
* **Monoid laws** — commutativity and associativity of add/mul.
* **Group laws** — identity, inverses, double-negation.
* **Bilinearity / naturality** — distributivity of matmul over add, and
  naturality of scalar multiplication w.r.t. linear maps.
* **Decomposition** — silu → x*sigmoid(x), subtraction → a + neg(b).
* **Constants** — constant folding.
"""

from typing import Any
from catopt.egraph import Rewrite
from catopt.ir import Op, Const


def R(name: str, lhs: Any, rhs: Any, law: str = "", check=None,
      derive=None) -> Rewrite:
    """Shorthand for creating a rewrite rule."""
    return Rewrite(name=name, lhs=lhs, rhs=rhs, law=law, check=check,
                   derive=derive)


def _shape_of(t: Any):
    """Best-effort shape of a bound term (delegates to cost model)."""
    from catopt.cost import _shape_of as _so
    return _so(t)


def _is_scalar(t: Any) -> bool:
    """True if the bound term is a scalar (shape ())."""
    s = _shape_of(t)
    return s == () or s == tuple()


def _is_row_scale(t: Any) -> bool:
    """Per-ROW scale: broadcasts to (B,T,1) — last dim is 1 (or scalar)."""
    s = _shape_of(t)
    if not isinstance(s, tuple):
        return s == ()
    return len(s) == 0 or s[-1] == 1


def _is_channel_scale(bound: dict) -> bool:
    """Per-CHANNEL scale: broadcasts over the weight's input dim.

    c may be scalar, (in,), or (1,...,1,in) — i.e. every non-last dim
    must be 1 and the last must equal W's input feature dim.
    """
    c, w = bound.get("c"), bound.get("W")
    cs, ws = _shape_of(c), _shape_of(w)
    if not isinstance(cs, tuple):
        return cs == ()
    if not isinstance(ws, tuple) or len(ws) < 1:
        return False
    if len(cs) == 0:
        return True
    return cs[-1] == ws[-1] and all(d == 1 for d in cs[:-1])


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
    Op.make("add",
            Op.make("matmul", "W", "a"),
            Op.make("matmul", "W", "b")),
    law="Distributivity of linear maps over addition (bilinearity).",
)

# add(matmul(W, a), matmul(W, b)) → matmul(W, add(a, b))  [reverse]
FACTOR_MUL = R(
    "factor_matmul",
    Op.make("add",
            Op.make("matmul", "W", "a"),
            Op.make("matmul", "W", "b")),
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
    Op.make("add",
            Op.make("matmul", "a", "W"),
            Op.make("matmul", "b", "W")),
    law="Bilinearity: linear maps distribute over addition in BOTH slots.",
)

# a@W + b@W = (a + b) @ W
RIGHT_FACTOR = R(
    "right_factor_matmul",
    Op.make("add",
            Op.make("matmul", "a", "W"),
            Op.make("matmul", "b", "W")),
    Op.make("matmul", Op.make("add", "a", "b"), "W"),
    law="Factor a shared right-weight (the slot `x @ W` uses).",
)

# THE WEIGHT-MERGE RULE.  x@W1 + x@W2 = x @ (W1 + W2): two projections of
# the SAME input collapse to one matmul on a summed weight.  This is the
# LoRA/adapter/model-soup merge that deployment tooling does by hand.
WEIGHT_FACTOR = R(
    "weight_factor_matmul",
    Op.make("add",
            Op.make("matmul", "x", "W"),
            Op.make("matmul", "x", "W2")),
    Op.make("matmul", "x", Op.make("add", "W", "W2")),
    law="Merge shared-input projections: x@W1 + x@W2 = x@(W1+W2).",
)

# x @ (W1 + W2) = x@W1 + x@W2  [reverse: expand for cost-model choice]
WEIGHT_DISTRIBUTE = R(
    "weight_distribute_matmul",
    Op.make("matmul", "x", Op.make("add", "W", "W2")),
    Op.make("add",
            Op.make("matmul", "x", "W"),
            Op.make("matmul", "x", "W2")),
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
    Op.make("add",
            Op.make("linear", "x", "W"),
            Op.make("linear", "x", "W2")),
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
    if not (isinstance(a, tuple) and isinstance(b, tuple)
            and len(a) == 2 and len(b) == 2):
        return False
    h, i, o = a[0], a[1], b[0]
    if b[1] != h:                               # B consumes A's out dim
        return False
    if not (isinstance(b1, tuple) and len(b1) == 1 and b1[0] == h):
        return False                            # inner bias: exactly (h,)
    if b2 != () and not (isinstance(b2, tuple)
                         and len(b2) == 1 and b2[0] == o):
        return False                            # outer bias: scalar|(o,)
    if not (isinstance(x, tuple) and len(x) >= 1 and x[-1] == i):
        return False                            # x feeds A's input dim
    return True


ASSOC_LINEAR_BIAS = R(
    "assoc_linear_bias",
    Op.make("linear", Op.make("linear", "x", "A", "b1"), "B", "b2"),
    Op.make("add",
            Op.make("linear", "x",
                    Op.make("matmul", "B", "A"),
                    Op.make("matmul", "B", "b1")),
            "b2"),
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
    Op.make("add",
            Op.make("linear", "x",
                    Op.make("matmul", "B", "A"),
                    Op.make("matmul", "B", "b1")),
            "b2"),
    Op.make("linear", Op.make("linear", "x", "A", "b1"), "B", "b2"),
    law="Reverse affine composition (eqsat weighs fused vs split).",
    check=_check_linear_bias_compose,
)

# a@W.T + b@W.T = (a+b)@W.T   ->   linear(add(a,b), W)
RIGHT_FACTOR_LINEAR = R(
    "right_factor_linear",
    Op.make("add",
            Op.make("linear", "a", "W"),
            Op.make("linear", "b", "W")),
    Op.make("linear", Op.make("add", "a", "b"), "W"),
    law="Factor a shared right-hand nn.Linear weight.",
)

# reverse of weight merge for `linear`
WEIGHT_DISTRIBUTE_LINEAR = R(
    "weight_distribute_linear",
    Op.make("linear", "x", Op.make("add", "W", "W2")),
    Op.make("add",
            Op.make("linear", "x", "W"),
            Op.make("linear", "x", "W2")),
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
    Op.make("mul",
            Op.make("silu", Op.make("linear", "x", "A")),
            Op.make("linear", "x", "B")),
    Op.make("mul",
            Op.make("silu",
                    Op.make("chunk",
                            Op.make("linear", "x",
                                    Op.make("concat", "A", "B", dim=0)),
                            chunks=2, dim=-1, index=0)),
            Op.make("chunk",
                    Op.make("linear", "x",
                            Op.make("concat", "A", "B", dim=0)),
                    chunks=2, dim=-1, index=1)),
    law="Product universal property: <f,g> = (f x g) . Delta.  Two "
        "projections of the same input are ONE GEMM into V x V, then "
        "project.  (Fused SwiGLU gate/up — MergedColumnParallelLinear.)",
)

# (x@A.T) * (x@B.T)  ->  chunk form without the gate nonlinearity.
# Covers GLU-style variants and any elementwise-mul pair of parallel
# projections.
PARALLEL_MUL_FUSE = R(
    "parallel_mul_fuse",
    Op.make("mul",
            Op.make("linear", "x", "A"),
            Op.make("linear", "x", "B")),
    Op.make("mul",
            Op.make("chunk",
                    Op.make("linear", "x",
                            Op.make("concat", "A", "B", dim=0)),
                    chunks=2, dim=-1, index=0),
            Op.make("chunk",
                    Op.make("linear", "x",
                            Op.make("concat", "A", "B", dim=0)),
                    chunks=2, dim=-1, index=1)),
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
    return Op.make("transpose",
                   Op.make("reshape", t, shape="S"),
                   arg1=1, arg2=2)


QKV_FUSE = R(
    "qkv_fuse",
    Op.make("sdpa",
            _head(Op.make("linear", "x", "Q")),
            _head(Op.make("linear", "x", "K")),
            _head(Op.make("linear", "x", "V")),
            scale="SC"),
    Op.make("sdpa",
            _head(Op.make("chunk",
                          Op.make("linear", "x",
                                  Op.make("concat",
                                          Op.make("concat", "Q", "K",
                                                  dim=0),
                                          "V", dim=0)),
                          chunks=3, dim=-1, index=0)),
            _head(Op.make("chunk",
                          Op.make("linear", "x",
                                  Op.make("concat",
                                          Op.make("concat", "Q", "K",
                                                  dim=0),
                                          "V", dim=0)),
                          chunks=3, dim=-1, index=1)),
            _head(Op.make("chunk",
                          Op.make("linear", "x",
                                  Op.make("concat",
                                          Op.make("concat", "Q", "K",
                                                  dim=0),
                                          "V", dim=0)),
                          chunks=3, dim=-1, index=2)),
            scale="SC"),
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
    return Op.make("transpose",
                   Op.make("reshape", t, shape=shape_var),
                   arg1=1, arg2=2)


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


_QKV_CAT = Op.make("concat",
                   Op.make("concat", "Q", "K", dim=0),
                   "V", dim=0)

QKV_FUSE_ASYM = R(
    "qkv_fuse_asym",
    Op.make("sdpa",
            _head_v(Op.make("linear", "x", "Q"), "S1"),
            _head_v(Op.make("linear", "x", "K"), "S2"),
            _head_v(Op.make("linear", "x", "V"), "S3"),
            scale="SC", enable_gqa="G"),
    Op.make("sdpa",
            _head_v(Op.make("split", Op.make("linear", "x", _QKV_CAT),
                            sizes="SZ", dim=-1, index=0), "S1"),
            _head_v(Op.make("split", Op.make("linear", "x", _QKV_CAT),
                            sizes="SZ", dim=-1, index=1), "S2"),
            _head_v(Op.make("split", Op.make("linear", "x", _QKV_CAT),
                            sizes="SZ", dim=-1, index=2), "S3"),
            scale="SC", enable_gqa="G"),
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
    from catopt.cost import _shape_of as _so
    d = bound.get(f"$attr:UD{pre}")
    es = bound.get(f"$attr:ES{pre}")
    rs = bound.get(f"$attr:RS{pre}")
    base = bound.get(pre)
    bs = _so(base)
    if not (isinstance(d, int) and isinstance(es, tuple)
            and isinstance(rs, tuple) and isinstance(bs, tuple)):
        return False
    if any(x is None for x in bs):
        return False
    nd = len(bs)
    d = d % (nd + 1)
    us = bs[:d] + (1,) + bs[d:]
    if len(es) != len(us) or len(rs) != nd or d == 0:
        return False
    r = es[d]
    if not isinstance(r, int) or r <= 1:
        return False
    if any(es[i] != us[i] for i in range(len(us)) if i != d):
        return False  # expand may only grow the inserted dim
    merged = us[:d - 1] + ((us[d - 1] or 0) * r,) + us[d + 1:]
    return rs == merged


def _check_gqa_absorb(bound: dict) -> bool:
    """Both k and v must be repeat-chains with the SAME repeat factor r,
    and q's head count must equal kv_heads * r."""
    from catopt.cost import _shape_of as _so
    for side in ("k", "v"):
        if not _check_repeat_chain(bound, side):
            return False
    if bound["$attr:ESk"] != bound["$attr:ESv"]:
        return False
    d = bound["$attr:UDk"] % (len(_so(bound["k"])) + 1)
    r = bound["$attr:ESk"][d]
    qs, ks = _so(bound["q"]), _so(bound["k"])
    if not (isinstance(qs, tuple) and isinstance(ks, tuple)
            and len(qs) >= 2 and len(ks) >= 2):
        return False
    if None in qs or None in ks:
        return False
    return qs[-2] == ks[-2] * r  # hq == hkv * n_rep


_REPEAT_KV = Op.make(
    "transpose",
    Op.make("reshape",
            Op.make("expand",
                    Op.make("unsqueeze", "k", arg1="UDk"),
                    shape="ESk"),
            shape="RSk"),
    arg1=1, arg2=2)

_REPEAT_V = Op.make(
    "transpose",
    Op.make("reshape",
            Op.make("expand",
                    Op.make("unsqueeze", "v", arg1="UDv"),
                    shape="ESv"),
            shape="RSv"),
    arg1=1, arg2=2)

GQA_ABSORB = R(
    "gqa_absorb_repeat",
    Op.make("sdpa",
            Op.make("transpose", "q", arg1=1, arg2=2),
            _REPEAT_KV,
            _REPEAT_V,
            arg4="D", arg5="C"),
    Op.make("sdpa",
            Op.make("transpose", "q", arg1=1, arg2=2),
            Op.make("transpose", "k", arg1=1, arg2=2),
            Op.make("transpose", "v", arg1=1, arg2=2),
            arg4="D", arg5="C", arg7=True),
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
    "matmul", "Q", Op.make("transpose", "K", arg1="TD1", arg2="TD2"))


def _const_val(t):
    return getattr(t, "value", None)


def _check_score_transpose(bound) -> bool:
    """k must be transposed on its last two dims — matmul(q, k^T)."""
    ks = _shape_of(bound.get("K"))
    d1, d2 = bound.get("$attr:TD1"), bound.get("$attr:TD2")
    if not (isinstance(ks, tuple) and all(isinstance(x, int) for x in ks)
            and isinstance(d1, int) and isinstance(d2, int)):
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
        ("mul", lambda: Op.make("mul", _QK_SCORES, "S"),
         _check_sdpa_scaled, _check_sdpa_mf_scaled, _derive_scale_mul),
        ("div", lambda: Op.make("div", _QK_SCORES, "S"),
         _check_sdpa_scaled, _check_sdpa_mf_scaled, _derive_scale_div),
        ("", lambda: _QK_SCORES,
         _check_sdpa_base, _check_sdpa_mf, _derive_scale_one),
    )
    wraps = (
        ("", lambda sm: sm),
        ("_drop", lambda sm: Op.make("dropout", sm, arg1="DP", arg2="DT")),
    )
    for sname, scores, check_add, check_mf, derive in scaled:
        for wname, wrap in wraps:
            sm = lambda inner: wrap(
                Op.make("softmax", inner, arg1="SD"))
            out.append(R(
                f"sdpa_fold_add{sname}{wname}",
                Op.make("matmul", sm(Op.make("add", scores(), "M")), "V"),
                Op.make("sdpa", "Q", "K", "V", "M", scale="SC"),
                law="softmax(qk^T s + m) v is sdpa — the additive mask is "
                    "the kernel's attn_mask argument.",
                check=check_add, derive=derive))
            out.append(R(
                f"sdpa_fold_masked_fill{sname}{wname}",
                Op.make("matmul",
                        sm(Op.make("masked_fill", scores(), "MK", "F")),
                        "V"),
                Op.make("sdpa", "Q", "K", "V",
                        Op.make("logical_not", "MK"), scale="SC"),
                law="masked_fill(m, -inf) before softmax is a boolean "
                    "attn_mask — logical_not turns the fill-mask into "
                    "SDPA's keep-mask.",
                check=check_mf, derive=derive))
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
#  Diagram-level pairing pass — the product law in full generality.
#
#  ⟨f₁,…,f_k⟩ = (f₁ × … × f_k) ∘ Δ  is a NON-LOCAL rewrite: it pairs
#  morphisms by their shared domain, not by a consumer pattern.  A
#  term-local lhs→rhs rule can only fire when a specific parent op
#  (mul, sdpa) happens to consume the projections — which is exactly why
#  pattern-matching optimizers miss the general case.  Here we implement
#  it as a pass over the e-graph: group every `linear(x, Wᵢ)` e-node by
#  the e-class of x, then offer each member's class the alternative
#
#      splitᵢ( linear(x, cat(W₁,…,W_k)) )
#
#  so the group may be extracted as ONE GEMM plus k zero-cost views.
#  Subsumes swiglu_fuse / qkv_fuse / qkv_fuse_asym / parallel_mul_fuse.
#  Guards: the shared input must be runtime data (contain a Var), and
#  every weight must be param-only so the concat folds at compile time.
# ---------------------------------------------------------------------------

def _term_has_var(t: Any) -> bool:
    from catopt.ir import Var
    if isinstance(t, Var):
        return True
    if isinstance(t, Op):
        return any(_term_has_var(a) for a in t.args)
    return False


def _pair_shared_input(eg: Any, *, op: str, split_dim: int,
                       cluster_key) -> list[dict[int, Any]]:
    """Product law over an arbitrary projection signature.

    Groups ``op`` e-nodes by shared input e-class and offers each member
    ``split_i(op(x, cat(W_1..W_k)))`` — one fused kernel plus per-member
    views.  ``cluster_key(enode, weight_term)`` returns a hashable
    signature under which members can share one fused kernel (or None to
    exclude a member): for conv2d it captures stride/padding/dilation/
    groups and the trailing weight dims.
    """
    from catopt.cost import _shape_of as _so
    from catopt.egraph import ENode, _LeafRegistry
    from catopt.ir import Var as _Var

    # Per-class "can some representative reach a Var leaf" — memoized and
    # cycle-guarded.  Replaces materializing any_term + _term_has_var per
    # class (quadratic tree walks on deep graphs) and is *more* sound:
    # it detects var-reachability rather than trusting an arbitrary rep.
    hasvar: dict[int, bool] = {}

    def cls_has_var(cid: int, stack: frozenset = frozenset()) -> bool:
        cid = eg.find(cid)
        if cid in hasvar:
            return hasvar[cid]
        if cid in stack:
            return False
        res = False
        for n in eg._classes[cid].nodes:
            if n.op == "leaf":
                t = _LeafRegistry.decode(n.attrs[0][1])
                if isinstance(t, _Var):
                    res = True
                    break
            elif any(cls_has_var(c, stack | {cid}) for c in n.children):
                res = True
                break
        hasvar[cid] = res
        return res

    by_input: dict[int, list[tuple[ENode, int, int]]] = {}
    for cid in list(eg._classes.keys()):
        for node in eg._classes[cid].nodes:
            # Only bias-free projections: a fused bias would need a
            # second concat; keeping the pairing arity at (x, w).
            if node.op != op or len(node.children) != 2:
                continue
            by_input.setdefault(
                eg.find(node.children[0]), []
            ).append((node, cid, eg.find(node.children[1])))

    groups: list[dict[int, Any]] = []
    for x_eid, members in by_input.items():
        if not cls_has_var(x_eid):
            continue  # pairing weight-only chains is compile-time noise
        # Cluster members by compat signature: convs differing only in
        # stride/kernel cannot share one fused conv, but each compatible
        # subset still pairs (e.g. two 1x1 heads pair; a 3x3 stays out).
        clusters: dict[Any, list[tuple[ENode, int, int]]] = {}
        for entry in members:
            node, cid, w = entry
            wt = eg.any_term(w)
            if wt is None:
                continue
            k = cluster_key(node, wt)
            if k is None:
                continue
            clusters.setdefault(k, []).append(entry)

        for cluster in clusters.values():
            weights = sorted({w for _, _, w in cluster})
            if len(weights) < 2:
                continue
            if any(cls_has_var(w) for w in weights):
                continue  # fused weight must fold at compile time
            wts = [eg.any_term(w) for w in weights]
            if any(t is None for t in wts):
                continue
            sizes: list[int] = []
            for t in wts:
                s = _so(t)
                if not (isinstance(s, tuple) and len(s) >= 1 and s[0]):
                    break
                sizes.append(s[0])
            if len(sizes) != len(weights):
                continue
            cat = weights[0]
            for w in weights[1:]:
                cat = eg.add_enode("concat", (cat, w), {"dim": 0})
            fused = eg.add_enode(op, (x_eid, cat),
                                 dict(cluster[0][0].attrs))
            index_of = {w: i for i, w in enumerate(weights)}
            group: dict[int, Any] = {}
            for _, cid, w in cluster:
                enode = ENode("split", (fused,), (
                    ("dim", split_dim), ("index", index_of[w]),
                    ("sizes", tuple(sizes)),
                ))
                split_eid = eg.add_enode("split", (fused,), {
                    "sizes": tuple(sizes), "dim": split_dim,
                    "index": index_of[w]})
                # Replayable witness: the member's own class term ->
                # its section of the fused GEMM.  Pointwise honesty —
                # asserts this instance, exactly what the pass proved.
                src = getattr(eg, "_oldest_term", eg.any_term)(cid)
                split_term = eg.any_term(split_eid)
                wit = None
                if src is not None and split_term is not None:
                    wit = Rewrite(
                        name=f"pair#{split_eid}",
                        lhs=src, rhs=split_term,
                        law=("pointwise witness for a non-local offer: "
                             "this member equals its split section of "
                             "the shared fused weight (equality "
                             "established by the pairing pass)"))
                eg.union(cid, split_eid, witness=wit)
                group.setdefault(cid, enode)
            groups.append(group)
    return groups


def _wshape(t: Any):
    from catopt.cost import _shape_of as _so
    return _so(t)


def pair_shared_input_linears(eg: Any) -> list[dict[int, Any]]:
    """Pair all `linear` e-nodes that share an input e-class.

    Returns one group per shared input: a dict mapping each member's
    canonical class id to the ``split`` ENode that reads its section of
    the shared fused GEMM.  The caller may feed the union of these dicts
    to ``extract_best`` as ``overrides`` — per-class greedy extraction
    cannot see that all members choosing a split share ONE fused GEMM
    (each split's subtree alone costs more than the member's own
    linear), so the coordinated choice must be forced globally.

    Idempotent: re-running rebuilds the same (hash-consed) enodes.
    """
    def key(enode, wt):
        s = _wshape(wt)
        # 2-D weight only (a 1-D "weight" cannot cat along out-dim).
        return ("lin",) if isinstance(s, tuple) and len(s) == 2 else None

    return _pair_shared_input(eg, op="linear", split_dim=-1,
                              cluster_key=key)


_CONV_ATTR_KEYS = ("stride", "padding", "dilation", "groups")


def pair_shared_input_convs(eg: Any) -> list[dict[int, Any]]:
    """Pair `conv2d` e-nodes sharing an input — the same product law.

    The fused weight is ``cat`` along out-channels (dim 0), valid only
    when members share stride/padding/dilation/groups and kernel dims;
    the projections split the output along the channel dim (1).
    """
    def key(enode, wt):
        a = dict(enode.attrs)
        if a.get("groups", 1) != 1:
            return None  # grouped conv: cat on O mixes groups wrongly
        s = _wshape(wt)
        if not (isinstance(s, tuple) and len(s) == 4):
            return None
        # same non-weight attrs AND same (in_ch, kh, kw) — cat on O
        # requires identical trailing weight dims.
        return (tuple(sorted(
            (k, a[k]) for k in _CONV_ATTR_KEYS if k in a)), s[1:])

    return _pair_shared_input(eg, op="conv2d", split_dim=1,
                              cluster_key=key)


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

# ---------------------------------------------------------------------------
# Scan monoid: affine-map domain
# ---------------------------------------------------------------------------
# A recurrence step ``h ↦ A·h + x`` is an affine map.  Affine maps form
# a monoid under composition:
#     (A2,b2) ∘ (A1,b1) = (A2·A1, A2·b1 + b2)
# A sequential fold is the left-associated composition; the balanced
# tree (parallel scan, Blelloch) is another bracketing of the same
# product — reachable by associativity alone once steps are lifted
# into the affine domain.  Matmul/add algebra alone provably cannot
# reach it: the pair (partial-product, partial-sum) is a cross-class
# object no term law synthesises (measured: order-preserving laws
# plateau at ~1.5·T depth; the monoid reaches ~log T).
#
# ``aff(A, b)``      — the map h ↦ A·h + b (a pair value, not a tensor)
# ``aff_compose(f,g)`` — f∘g as an affine object
# ``apply(f, h)``    — evaluate the map on h (back in tensor-land)

AFF_LIFT = R("aff_lift",
             Op.make("add", Op.make("matmul", "A", "h"), "x"),
             Op.make("apply", Op.make("aff", "A", "x"), "h"),
             law="recurrence step is affine-map application")

AFF_LIFT_STEP = R("aff_lift_step",
                  Op.make("add",
                          Op.make("matmul", "A",
                                  Op.make("apply", "f", "h")),
                          "x"),
                  Op.make("apply",
                          Op.make("aff_compose",
                                  Op.make("aff", "A", "x"), "f"),
                          "h"),
                  law="compose step with the preceding map")

AFF_UNLIFT = R("aff_unlift",
               Op.make("apply", Op.make("aff", "A", "x"), "h"),
               Op.make("add", Op.make("matmul", "A", "h"), "x"),
               law="affine application unfolds")

AFF_COMPOSE_UNFOLD = R("aff_compose_unfold",
                       Op.make("apply",
                               Op.make("aff_compose", "f", "g"), "h"),
                       Op.make("apply", "f",
                               Op.make("apply", "g", "h")),
                       law="composition is sequential application")

AFF_ASSOC = R("aff_assoc",
              Op.make("aff_compose",
                      Op.make("aff_compose", "f", "g"), "h"),
              Op.make("aff_compose", "f",
                      Op.make("aff_compose", "g", "h")),
              law="affine composition is associative")

AFF_ASSOC_REV = R("aff_assoc_rev",
                  Op.make("aff_compose", "f",
                          Op.make("aff_compose", "g", "h")),
                  Op.make("aff_compose",
                          Op.make("aff_compose", "f", "g"), "h"),
                  law="affine composition is associative")

#: Minimal law set for scan discovery.  Deliberately excludes
#: ``comm_add``: commutativity is the explosive law (permutation space)
#: and Blelloch reassociation is order-preserving.
SCAN_LAWS: list[Rewrite] = [
    AFF_LIFT, AFF_LIFT_STEP, AFF_UNLIFT, AFF_COMPOSE_UNFOLD,
    AFF_ASSOC, AFF_ASSOC_REV,
]


# ---------------------------------------------------------------------------
#  Scan monoid: diagonal-affine domain (elementwise / Mamba-faithful SSMs)
# ---------------------------------------------------------------------------
# ``AFF_LIFT`` only sees steps spelled ``add(matmul(A, h), x)``.  A
# Mamba-faithful selective step ``h ↦ a ⊙ h + x`` is a DIAGONAL affine
# map — the same monoid restricted to diagonal linear parts:
#
#     (a2,b2) ∘ (a1,b1) = (a2⊙a1, a2⊙b1 + b2)
#
# a strictly CHEAPER carrier: O(d) elementwise work per compose instead
# of a dense d×d product.  The step exports as ``add(mul(a,h), x)``
# (for ``DiagonalSSM`` the translation x is itself ``mul(b_t, x_t)`` —
# the metavariable binds it whole, so no second LHS shape is needed),
# which ``AFF_LIFT`` cannot see.  These rules mirror ``SCAN_LAWS``
# verbatim in structure:
#
# ``aff_diag(a, b)``     — the map h ↦ a⊙h + b (a pair value)
# ``affd_compose(f, g)`` — f∘g in the diagonal-affine monoid
# ``applyd(f, h)``       — evaluate: f₀⊙h + f₁ (back in tensor-land)


def _affd_state_like(bound: dict) -> bool:
    """Side condition for the diagonal lifts: the ``h`` binding must be
    state-shaped — a previous step's ``add``/``sub`` spine, an already
    lifted application (``applyd``/``apply``), or a leaf (the h0 Param
    or a free Var).  Per-step vectors (a_t, b_t, x_t — select/mul
    terms) are NOT states.

    The check exists for e-graph economy, not soundness — the rewrite
    a⊙h + x ≡ applyd(aff_diag(a,x), h) is valid for ANY h.  Without it,
    the operand-position variants below would each fire a useless
    sideways lift binding an input vector as "h"."""
    t = bound.get("h")
    if isinstance(t, Op):
        return t.op in ("add", "sub", "apply", "applyd")
    return True


# --- the four operand positions --------------------------------------
# add and mul are commutative, and ``meta.canonicalize`` normalises
# operand ORDER (children sorted by op_repr): DiagonalSSM's steps
# canonicalise to ``add(mul(h, a), mul(b, x))`` for t > 0 but
# ``add(mul(b, x), mul(a, h0))`` for the first step.  Since comm_add /
# comm_mul are deliberately absent from the law set, each position the
# state-mul can occupy gets its own LHS so the lift fires on both
# raw-exported AND canonicalised terms.  ``_affd_state_like`` keeps the
# cross-bindings (state vs input swapped) from firing spuriously, so
# exactly one variant fires per ``add`` e-node.

AFFD_LIFT = R("affd_lift",
              Op.make("add", Op.make("mul", "a", "h"), "x"),
              Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
              law="diagonal recurrence step is diagonal-affine "
                  "application: a⊙h + x = (aff_diag(a,x))(h)",
              check=_affd_state_like)

AFFD_LIFT_SWAP = R("affd_lift_swap",
                   Op.make("add", Op.make("mul", "h", "a"), "x"),
                   Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
                   law="mul-order variant of affd_lift (canonicalised "
                       "terms put the state operand first)",
                   check=_affd_state_like)

AFFD_LIFT_POST = R("affd_lift_post",
                   Op.make("add", "x", Op.make("mul", "a", "h")),
                   Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
                   law="add-order variant of affd_lift (state-mul in "
                       "the second add slot)",
                   check=_affd_state_like)

AFFD_LIFT_POST_SWAP = R("affd_lift_post_swap",
                        Op.make("add", "x", Op.make("mul", "h", "a")),
                        Op.make("applyd",
                                Op.make("aff_diag", "a", "x"), "h"),
                        law="remaining operand position of affd_lift",
                        check=_affd_state_like)

# The step rules need no side condition: the ``applyd`` inside the mul
# already pins the state operand — an input e-class contains no
# ``applyd`` enode, so only the true direction matches.
AFFD_LIFT_STEP = R("affd_lift_step",
                   Op.make("add",
                           Op.make("mul", "a",
                                   Op.make("applyd", "f", "h")),
                           "x"),
                   Op.make("applyd",
                           Op.make("affd_compose",
                                   Op.make("aff_diag", "a", "x"), "f"),
                           "h"),
                   law="compose step with the preceding map")

AFFD_LIFT_STEP_SWAP = R(
    "affd_lift_step_swap",
    Op.make("add",
            Op.make("mul", Op.make("applyd", "f", "h"), "a"),
            "x"),
    Op.make("applyd",
            Op.make("affd_compose",
                    Op.make("aff_diag", "a", "x"), "f"),
            "h"),
    law="mul-order variant of affd_lift_step")

AFFD_LIFT_STEP_POST = R(
    "affd_lift_step_post",
    Op.make("add", "x",
            Op.make("mul", "a",
                    Op.make("applyd", "f", "h"))),
    Op.make("applyd",
            Op.make("affd_compose",
                    Op.make("aff_diag", "a", "x"), "f"),
            "h"),
    law="add-order variant of affd_lift_step")

AFFD_LIFT_STEP_POST_SWAP = R(
    "affd_lift_step_post_swap",
    Op.make("add", "x",
            Op.make("mul",
                    Op.make("applyd", "f", "h"), "a")),
    Op.make("applyd",
            Op.make("affd_compose",
                    Op.make("aff_diag", "a", "x"), "f"),
            "h"),
    law="remaining operand position of affd_lift_step")

# --- unit lift: pure accumulation -----------------------------------
# ``add(h, x)`` — the recurrence ``h_t = h_{t-1} + x_t`` (cumsum,
# running statistics, linear attention's KV state ``S_t = S_{t-1} +
# k_t v_tᵀ``) — is the ``a ≡ 1`` degenerate case of the diagonal step:
# ``1⊙h + x = h + x``.  No ``mul`` enode exists for ``AFFD_LIFT`` to
# see, so additive accumulations were unreachable by the scan monoid.
# The unit lift writes the step as ``applyd(aff_diag(1, x), h)``; once
# carried, the ordinary step/assoc machinery (affd_compose balancing,
# trace lift, the batched executor) applies verbatim.
#
# The unit ``1`` is spelled ``expand(Const(1.0), shape=US)`` — the
# broadcast shape of the add — NOT a bare scalar: ``aff_diag`` leaves
# must carry the state's ``(d,)`` shape for ``scan_lower``'s
# ``_leaf_shapes_consistent`` (the batched executor *stacks* leaf
# a-parts; a scalar-shaped a would fail the check and bar the whole
# carrier tree from BatchedScanModule AND trace_lift's carrier path).
# ``US`` is an attribute metavariable filled by ``_derive_affd_unit``,
# which also vetoes the firing when the bound shapes are not concrete.


def _affd_unit_state_like(bound: dict) -> bool:
    """Side condition for the unit lifts: the ``h`` binding must be
    state-shaped — a previous step's ``add``/``sub`` spine, an already
    lifted application (``applyd``/``apply``), or a leaf (the h0 Param
    or a free Var).  Same economy guard as ``_affd_state_like``, plus a
    ``Const`` exclusion: a scalar offset is not an accumulating state.
    Per-step increments (select/mul terms) are NOT states."""
    t = bound.get("h")
    if isinstance(t, Op):
        return t.op in ("add", "sub", "apply", "applyd")
    return not isinstance(t, Const)


def _derive_affd_unit(bound: dict) -> dict | None:
    """``US`` := broadcast(shape(h), shape(x)) — the unit diagonal must
    materialise at the add's output shape (all ones).  Vetoes the
    firing when either bound term's shape is non-concrete."""
    from catopt.cost import _broadcast
    s = _broadcast(_shape_of(bound.get("h")), _shape_of(bound.get("x")))
    if not (isinstance(s, tuple)
            and all(isinstance(d, int) for d in s)):
        return None
    # The RHS embeds ``Const(1.0)`` as a leaf; ``_instantiate`` adds
    # leaf enodes keyed by repr WITHOUT registering the term, so
    # ``any_term``/extraction would decode the raw string "1.0" unless
    # the leaf is registered here, ahead of instantiation.
    from catopt.egraph import _LeafRegistry
    _LeafRegistry.register(Const(1.0))
    return {"$attr:US": tuple(s)}


#: The unit diagonal as a shared pattern fragment: ones of the add's
#: broadcast shape, spelled as a Const broadcast so the carrier leaf
#: reports the same ``(d,)`` shape as every other ``aff_diag``.
_AFFD_UNIT = Op.make("expand", Const(1.0), shape="US")

# Two operand positions — add is commutative and ``canonicalize``
# sorts children by op_repr, so the state operand can sit in either
# slot (mirroring affd_lift / affd_lift_post; there is no ``mul`` and
# hence no ``_swap`` dimension).  ``_affd_unit_state_like`` suppresses
# the sideways bindings.
AFFD_LIFT_UNIT = R(
    "affd_lift_unit",
    Op.make("add", "h", "x"),
    Op.make("applyd", Op.make("aff_diag", _AFFD_UNIT, "x"), "h"),
    law="Unit introduction: pure accumulation h + x IS the diagonal "
        "affine map with a ≡ 1 — applyd(aff_diag(1, x), h).",
    check=_affd_unit_state_like,
    derive=_derive_affd_unit)

AFFD_LIFT_UNIT_POST = R(
    "affd_lift_unit_post",
    Op.make("add", "x", "h"),
    Op.make("applyd", Op.make("aff_diag", _AFFD_UNIT, "x"), "h"),
    law="add-order variant of affd_lift_unit (state operand in the "
        "second add slot)",
    check=_affd_unit_state_like,
    derive=_derive_affd_unit)

# The step rules need no side condition: the ``applyd`` inside the add
# already pins the state operand — an input e-class contains no
# ``applyd`` enode, so only the true direction matches.  The map
# ordering mirrors AFFD_LIFT_STEP: compose(aff_diag(1,x), f) applies f
# first, then h ↦ 1⊙(f·h) + x = f(h) + x.
AFFD_LIFT_UNIT_STEP = R(
    "affd_lift_unit_step",
    Op.make("add",
            Op.make("applyd", "f", "h"),
            "x"),
    Op.make("applyd",
            Op.make("affd_compose",
                    Op.make("aff_diag", _AFFD_UNIT, "x"), "f"),
            "h"),
    law="compose a unit (pure-accumulation) step with the preceding "
        "map — the h ↦ h + x analogue of affd_lift_step",
    derive=_derive_affd_unit)

AFFD_LIFT_UNIT_STEP_POST = R(
    "affd_lift_unit_step_post",
    Op.make("add", "x",
            Op.make("applyd", "f", "h")),
    Op.make("applyd",
            Op.make("affd_compose",
                    Op.make("aff_diag", _AFFD_UNIT, "x"), "f"),
            "h"),
    law="add-order variant of affd_lift_unit_step",
    derive=_derive_affd_unit)

AFFD_UNLIFT = R("affd_unlift",
                Op.make("applyd", Op.make("aff_diag", "a", "x"), "h"),
                Op.make("add", Op.make("mul", "a", "h"), "x"),
                law="diagonal-affine application unfolds")

AFFD_COMPOSE_UNFOLD = R("affd_compose_unfold",
                        Op.make("applyd",
                                Op.make("affd_compose", "f", "g"), "h"),
                        Op.make("applyd", "f",
                                Op.make("applyd", "g", "h")),
                        law="composition is sequential application")

AFFD_ASSOC = R("affd_assoc",
               Op.make("affd_compose",
                       Op.make("affd_compose", "f", "g"), "h"),
               Op.make("affd_compose", "f",
                       Op.make("affd_compose", "g", "h")),
               law="diagonal-affine composition is associative")

AFFD_ASSOC_REV = R("affd_assoc_rev",
                   Op.make("affd_compose", "f",
                           Op.make("affd_compose", "g", "h")),
                   Op.make("affd_compose",
                           Op.make("affd_compose", "f", "g"), "h"),
                   law="diagonal-affine composition is associative")

#: Minimal law set for diagonal-scan discovery — the mul-form mirror of
#: ``SCAN_LAWS``, plus the unit lift for pure accumulations.  Covers
#: every operand position the state-mul can take under
#: comm-normalisation; ``_affd_state_like``/``_affd_unit_state_like``
#: suppress the sideways firings.
SCAN_DIAG_LAWS: list[Rewrite] = [
    AFFD_LIFT, AFFD_LIFT_SWAP, AFFD_LIFT_POST, AFFD_LIFT_POST_SWAP,
    AFFD_LIFT_STEP, AFFD_LIFT_STEP_SWAP,
    AFFD_LIFT_STEP_POST, AFFD_LIFT_STEP_POST_SWAP,
    AFFD_LIFT_UNIT, AFFD_LIFT_UNIT_POST,
    AFFD_LIFT_UNIT_STEP, AFFD_LIFT_UNIT_STEP_POST,
    AFFD_UNLIFT, AFFD_COMPOSE_UNFOLD, AFFD_ASSOC, AFFD_ASSOC_REV,
]


#: All rules combined.
ALL_RULES: list[Rewrite] = SIMPLIFICATION_RULES + CATEGORICAL_RULES


def all_rules() -> list[Rewrite]:
    """Return a fresh list of all rewrite rules."""
    return list(ALL_RULES)