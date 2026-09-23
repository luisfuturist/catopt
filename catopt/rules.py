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


def R(name: str, lhs: Any, rhs: Any, law: str = "") -> Rewrite:
    """Shorthand for creating a rewrite rule."""
    return Rewrite(name=name, lhs=lhs, rhs=rhs, law=law)


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

# matmul(W, mul(x, c)) = mul(matmul(W, x), c)
# KEY RULE: naturality of scalar multiplication w.r.t. linear maps.
# Lets the optimizer slide an elementwise scaling past a matmul.
NATURALITY_SCALAR = R(
    "naturality_scalar",
    Op.make("matmul", "W", Op.make("mul", "x", "c")),
    Op.make("mul", Op.make("matmul", "W", "x"), "c"),
    law="Naturality: scalar multiplication commutes with linear maps.",
)

NATURALITY_SCALAR_REV = R(
    "naturality_scalar_rev",
    Op.make("mul", Op.make("matmul", "W", "x"), "c"),
    Op.make("matmul", "W", Op.make("mul", "x", "c")),
    law="Reverse naturality: pull scalar into the matmul's input.",
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
    NATURALITY_SCALAR,
    NATURALITY_SCALAR_REV,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
]

#: All rules combined.
ALL_RULES: list[Rewrite] = SIMPLIFICATION_RULES + CATEGORICAL_RULES


def all_rules() -> list[Rewrite]:
    """Return a fresh list of all rewrite rules."""
    return list(ALL_RULES)