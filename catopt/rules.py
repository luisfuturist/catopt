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
    from catopt.cost import _shape_of as _so
    from catopt.egraph import ENode

    by_input: dict[int, list[tuple[int, int]]] = {}
    for cid in list(eg._classes.keys()):
        for node in eg._classes[cid].nodes:
            if node.op != "linear" or len(node.children) != 2:
                continue
            by_input.setdefault(
                eg.find(node.children[0]), []
            ).append((cid, eg.find(node.children[1])))

    groups: list[dict[int, Any]] = []
    for x_eid, members in by_input.items():
        xt = eg.any_term(x_eid)
        if xt is None or not _term_has_var(xt):
            continue  # pairing weight-only chains is compile-time noise
        weights = sorted({w for _, w in members})
        if len(weights) < 2:
            continue
        wts = [eg.any_term(w) for w in weights]
        if any(t is None or _term_has_var(t) for t in wts):
            continue  # fused weight must fold at compile time
        sizes: list[int] = []
        for t in wts:
            s = _so(t)
            if not (isinstance(s, tuple) and len(s) == 2 and s[0]):
                break
            sizes.append(s[0])
        if len(sizes) != len(weights):
            continue
        cat = weights[0]
        for w in weights[1:]:
            cat = eg.add_enode("concat", (cat, w), {"dim": 0})
        fused = eg.add_enode("linear", (x_eid, cat))
        index_of = {w: i for i, w in enumerate(weights)}
        group: dict[int, Any] = {}
        for cid, w in members:
            enode = ENode("split", (fused,), (
                ("dim", -1), ("index", index_of[w]),
                ("sizes", tuple(sizes)),
            ))
            eg.union(cid, eg.add_enode("split", (fused,), {
                "sizes": tuple(sizes), "dim": -1,
                "index": index_of[w]}))
            group.setdefault(cid, enode)
        groups.append(group)
    return groups


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
]

#: All rules combined.
ALL_RULES: list[Rewrite] = SIMPLIFICATION_RULES + CATEGORICAL_RULES


def all_rules() -> list[Rewrite]:
    """Return a fresh list of all rewrite rules."""
    return list(ALL_RULES)