"""Discovery content tables — everything the player may learn, as data.

The discovery engine's hardcoded tables are *content*, not code:
shape banks an oracle enumerates, a per-op constant domain a guarded
sweep can mint, the schema library a proposal strategy replays, the
generator-arm inventory a guide schedules, the prior and reward
weights a player's score is composed of.  A learned policy could
plausibly choose any of them — so they live here, in one pure-data
module (the ``catopt_core.opmeta`` idiom: the table is the single
home, consumers bind projections or the table itself, and a
consistency test — ``tests/test_lawdata.py`` — pins the relation).

What is deliberately *not* here: budgets and tolerances (resource
knobs, not content), enumeration *algorithms* (the per-op attr-domain
generators in ``catopt_discovery.oracle._attr_options`` are shape-
dependent code), recognizer hooks (``pipeline._RECOGNIZERS`` maps a
census signature to a *function* — a hook, not a record), nn.Module
corpus registries (``impact._model_cases``, ``intake.candidates``,
``zoo`` — kernel bodies are code), and store/SQL schema vocabulary
(``evidence``'s column lists are the database contract).

Term specs
----------

Several tables carry *terms* (seed programs, schema equalities) in
the declarative spec language :func:`object_synthesis.term_from_spec`
reads, extended with two leaf forms the consumers resolve first::

    spec := str                        # metavariable leaf ("A", "x")
          | int | float                # Const leaf
          | ("var", name, (dims, ...)) # Var(name, TensorType(dims))
          | ("param", name, (dims, ..))# Param(name, TensorType(dims))
          | (op, *specs)               # Op.make(op, *specs)
          | (op, *specs, {attrs})      # trailing dict = attr map

so a new seed term, schema or candidate law is a *new data row*, not
a Python edit — the shape a model would write a new entry in.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ARENA_REWARD",
    "ATTR_KINDS",
    "ATTR_KIND_OVERRIDES",
    "BASE_SHAPES",
    "CANDIDATE_LAWS",
    "CHAIN_VIEWED_SHAPES",
    "COHERENCE_CLUSTER",
    "COHERENCE_CORPUS_MODELS",
    "CONST_DOMAIN",
    "CONST_LEAVES",
    "CORPUS_ARMS",
    "FALLBACK_AXES",
    "FALLBACK_SHAPES",
    "FAMILIES",
    "FREE_SENTINELS",
    "GENERATOR_ORDER",
    "GRAMMAR_BINARY_OPS",
    "GRAMMAR_LITERALS",
    "GRAMMAR_SCHEMAS",
    "GRAMMAR_UNARY_OPS",
    "INSTANCE_ATTR_DEFAULTS",
    "INSTANCE_LEAF_SHAPES",
    "INSTANCE_SCALAR_MVARS",
    "LINEAR_HELD_SHAPES",
    "LINEAR_TRAIN_SHAPES",
    "MV_NAMES",
    "PRIOR_WEIGHTS",
    "PURPOSE_BUILT",
    "REFEREE_SCORE",
    "SEED_TERMS",
    "SHAPE_POOL",
    "SHAPE_SCHEMAS",
    "TUPLE_SOURCES",
    "TUPLE_SOURCE_SHAPE",
    "VIEWED_SHAPES",
]

# ---------------------------------------------------------------------------
#  Oracle enumeration banks (catopt_discovery.oracle)
# ---------------------------------------------------------------------------

#: Shapes offered to a metavariable that sits under a view op.
VIEWED_SHAPES: tuple = (
    (4,),
    (2, 3),
    (3, 4),
    (2, 2),
    (2, 3, 4),
    (2, 3, 1),
)

#: Extra viewed-bank shapes for patterns that contain an
#: operand-*chained* view (``reshape(expand(unsqueeze(...)))``) — the
#: rank-4 ``(b, t, h_kv, d)`` / ``(b, t, h, d)`` pair.  The chain's
#: intermediate tensors are rank-4+, and ``gqa_absorb_repeat``'s
#: corner (``repeat-heads``: ``q[-2] == k[-2] * r``) needs the head-dim
#: product only a rank-4 pair supplies.  Scoped to chained patterns so
#: chain-free enumerations are byte-identical.
CHAIN_VIEWED_SHAPES: tuple = (
    (2, 3, 2, 4),
    (2, 3, 4, 4),
)

#: Sentinel shapes the free operand's derived bank always carries —
#: the scalar corner and two generic mismatches.
FREE_SENTINELS: tuple = ((), (4,), (7, 7))

#: Axes offered when the operand's shape is not visible (a free
#: metavariable bound later, or a rank-0 operand): the common last-
#: and first-axis spellings.
FALLBACK_AXES: tuple = (-2, -1, 0, 1)

#: Trailing tuples offered for a shape-typed attr whose operand
#: shape is unknown — plausible normalized-shape spellings.
FALLBACK_SHAPES: tuple = ((1,), (2,), (4,), (2, 4))

#: Per-op literal-constant domain — the values a metavariable's
#: *parent op* admits, keyed by the op name.  The leaf bank's generic
#: scalar corner mints a single ``Const(0.5)``; a shipped guard can
#: demand a specific literal that corner never reaches, so the
#: guarded-region sweep can never construct the site — the
#: *domain-gapped* class of ``project/retros/cap-policy.md``.  Each
#: entry is a measured widening:
#:
#: * ``pow`` — the exponent (``2`` for the square/RMSNorm spelling,
#:   ``-0.5`` for the reciprocal-root; ``0.5`` / ``1`` round it out).
#: * ``masked_fill`` — the softmax mask sentinel; ``-inf`` is the one
#:   that clears the strict ``const-cmp F < -1e30`` (``-1e30`` itself
#:   does not: the comparison is strict), the other two are the
#:   finite analogues.
#: * ``div`` — the scalar identities (``1`` for the numerator of the
#:   ``1 / sqrt(x)`` spelling, ``0`` for the additive-identity probe).
#:
#: Only the ops a shipped guard actually constrains carry an entry;
#: ``mul`` / ``add`` identities were measured and left out (the bank
#: feeds the enumeration — a widening with no rescue to show for it
#: perturbs the capped order).  ``project/retros/value-bank.md``.
CONST_DOMAIN: dict[str, tuple[int | float, ...]] = {
    "pow": (2, -0.5, 0.5, 1),
    "masked_fill": (-float("inf"), -1e30, 1e9),
    "div": (1, 0),
}

#: Canonical attr names -> the value kind the sweep can enumerate.
#: ``ATTR_SCHEMA`` *names* every positional attr; this table *types*
#: the names.  ``dim`` defaults to a plain axis — the reduction ops
#: and the normalized-shape spellings override below.
ATTR_KINDS: dict[str, str] = {
    # axes — valid values are ``-rank..rank-1`` of the operand
    "dim": "axis",
    "dim0": "axis",
    "dim1": "axis",
    "start_dim": "axis",
    "end_dim": "axis",
    "source": "axis",
    "destination": "axis",
    # small ints — indices, counts, bounds, kernel sizes
    "index": "int",
    "start": "int",
    "end": "int",
    "step": "int",
    "length": "int",
    "chunks": "int",
    "k": "int",
    "sections": "int",
    "num_groups": "int",
    "num_classes": "int",
    "num_layers": "int",
    "groups": "int",
    "shifts": "int",
    "diagonal": "int",
    "correction": "int",
    "upscale_factor": "int",
    "downscale_factor": "int",
    "stride": "int",
    "padding": "int",
    "dilation": "int",
    "m": "int",
    # float scalars — scales, epsilons, rates
    "scale": "float",
    "eps": "float",
    "momentum": "float",
    "p": "float",
    "dropout_p": "float",
    "dropout": "float",
    "rtol": "float",
    "atol": "float",
    "alpha": "float",
    "beta": "float",
    "threshold": "float",
    "min": "float",
    "max": "float",
    "input_scale": "float",
    "value": "float",
    "ord": "float",
    # boolean flags
    "keepdim": "bool",
    "is_causal": "bool",
    "enable_gqa": "bool",
    "train": "bool",
    "training": "bool",
    "largest": "bool",
    "sorted": "bool",
    "descending": "bool",
    "accumulate": "bool",
    "equal_nan": "bool",
    "cudnn_enabled": "bool",
    "has_biases": "bool",
    "bidirectional": "bool",
    "batch_first": "bool",
    "use_input_stats": "bool",
    "right": "bool",
    "out_int32": "bool",
    # shape-typed tuples — normalized_shape and friends
    "shape": "shape",
    "sizes": "shape",
    "size": "shape",
    "pad": "shape",
    # int-or-tuple axis lists (roll's ``dims``)
    "dims": "red-dims",
    # honestly unenumerable — string payloads, not scalars
    "equation": "str",
    "mode": "str",
    "reduce": "str",
    "layout": "str",
}

#: ``(op, canonical-attr)`` pairs whose value kind differs from the
#: name default — the attr names are honest but not typed, and these
#: are the measured exceptions.
ATTR_KIND_OVERRIDES: dict[tuple[str, str], str] = {
    # the norms' ``dim`` attr is aten's normalized_shape list, not an
    # axis (``rms_norm(x, ns, w, eps)`` / ``layer_norm``'s arg1).
    ("layer_norm", "dim"): "shape",
    ("rms_norm", "dim"): "shape",
    # ``eye(n)`` / ``eye.m(n, m)`` — sizes, not axes.
    ("eye", "dim"): "int",
    # unfold's ``size`` is a kernel extent; upsample's ``size`` stays
    # a shape tuple.
    ("unfold", "size"): "int",
}

#: Shared operand shape for the tuple-producing sources below.
TUPLE_SOURCE_SHAPE: tuple = (2, 4)

#: Tuple-producing ops for a metavariable under ``getitem`` — the
#: corpus's real ``getitem`` matches pick elements out of ``topk`` /
#: ``var_mean`` / ``cummax`` (a bare ``Var`` only covers the dim-0
#: tensor index).  Each entry is ``(op, attrs)`` built over a
#: ``TUPLE_SOURCE_SHAPE`` operand.
TUPLE_SOURCES: tuple = (
    ("topk", {"k": 2}),
    ("var_mean", {"dim": (-1,), "correction": 0, "keepdim": True}),
    ("cummax", {"dim": 0}),
)

# ---------------------------------------------------------------------------
#  Instantiation defaults (catopt_discovery.verifier.generic_instance)
# ---------------------------------------------------------------------------

#: Leaf shapes for the SDPA-fold family's metavariables — the only
#: rules the bench registry does not cover.  ``(B,H,T,D)`` scores feed
#: ``(B,H,T,T)`` masks and a ``(B,H,T,D)`` value.
INSTANCE_LEAF_SHAPES: dict[str, tuple[int, ...]] = {
    "Q": (2, 4, 8, 4),
    "K": (2, 4, 8, 4),
    "V": (2, 4, 8, 4),
    "M": (2, 4, 8, 8),
    "MK": (2, 4, 8, 8),
}

#: Attr-metavariable defaults for the SDPA-fold family.
INSTANCE_ATTR_DEFAULTS: dict[str, Any] = {
    "TD1": -2,
    "TD2": -1,
    "SD": -1,
    "DP": 0.5,
    "DT": True,
}

#: Scalar metavariable defaults — the ``Const`` a name mints.
INSTANCE_SCALAR_MVARS: dict[str, int | float] = {
    "S": 0.5,
    "F": float("-inf"),
}

# ---------------------------------------------------------------------------
#  Gap-synthesis pools (catopt_discovery.gap_gen)
# ---------------------------------------------------------------------------

#: Leaf-shape pool for metavariable instantiation.  Uniform draws
#: cover the pointwise laws (all metavars one shape); the broadcastable
#: entries (``(1,)``, ``(4,1)``, ``()``) give mixed-view candidates a
#: non-view operand that composes on BOTH sides of the rewrite.
SHAPE_POOL: tuple = (
    (4, 16),
    (16, 16),
    (2, 8, 16),
    (8, 8),
    (16,),
    (4, 8),
    (4, 1),
    (1, 16),
    (1,),
    (),
    (2, 4, 4),
)

#: Uniform base shapes tried first, in order, before random draws.
BASE_SHAPES: tuple = ((4, 16), (16, 16), (2, 8, 16))

# ---------------------------------------------------------------------------
#  The algebraic-grammar alphabet (catopt_discovery.grammar)
# ---------------------------------------------------------------------------

#: Binary ops the search may place at a two-child node.
GRAMMAR_BINARY_OPS: tuple = (
    "add",
    "mul",
    "sub",
    "div",
    "pow",
    "matmul",
)

#: Unary ops the search may wrap a leaf in (and swap a unary node to).
GRAMMAR_UNARY_OPS: tuple = (
    "neg",
    "exp",
    "square",
    "sqrt",
    "rsqrt",
    "sigmoid",
    "silu",
    "tanh",
)

#: Literal constants the search may substitute for a leaf.  Integer
#: spelling, matching the hand-written grammar (``Const(0)``, not
#: ``Const(0.0)``) — the e-graph keys leaves by ``repr``, so a float
#: spelling would make a law miss its own grammar rule and look novel.
GRAMMAR_LITERALS: tuple = (0, 1, 2)

# ---------------------------------------------------------------------------
#  Training families (catopt_discovery.families)
# ---------------------------------------------------------------------------

#: The family names, in report order.
FAMILIES: tuple[str, ...] = ("chain", "dup", "linear")

#: ``(d_in, d_hidden, d_out)`` shapes for the linear family — training.
LINEAR_TRAIN_SHAPES: tuple = (
    (8, 16, 8),
    (8, 24, 8),
    (6, 12, 6),
    (10, 20, 10),
)

#: Held-out linear shapes (widths unseen in training).
LINEAR_HELD_SHAPES: tuple = (
    (7, 18, 7),
    (9, 21, 9),
    (12, 26, 12),
    (11, 22, 11),
)

# ---------------------------------------------------------------------------
#  Coherence probe neighbourhoods
# ---------------------------------------------------------------------------

#: The select_mul / softmax_fold / transpose_push_mul neighbourhood:
#: the folds and the layout/pointwise naturality laws they touch
#: (``catopt_discovery.coherence``'s hand-picked cluster).
COHERENCE_CLUSTER: tuple = (
    "select_mul",
    "softmax_fold",
    "sdpa_fold_add",
    "sdpa_fold_addmul",
    "comm_mul",
    "comm_add",
    "transpose_push_mul",
    "transpose_pull_mul",
    "transpose_push_add",
    "transpose_pull_add",
    "transpose_push_exp",
    "transpose_transpose_dd",
    "transpose_matmul",
)

#: The coherence2 reach corpus — small real exports chosen to cover
#: the interesting law neighbourhoods: silu/swiglu, matmul chains,
#: linear folds, softmax/select, residual adds.
COHERENCE_CORPUS_MODELS: tuple = (
    "SwiGLU",
    "ResidualMLP",
    "GatedResidualBlock",
    "ManualSoftmaxAttention",
    "MatrixChain",
    "ParallelLinear",
    "NormLinear",
)

# ---------------------------------------------------------------------------
#  The intake ledger (catopt_discovery.intake)
# ---------------------------------------------------------------------------

#: The corpus-circularity ledger — the workload names written to spell
#: one specific *pattern* (an auto-cond guard's region), not to stand
#: for a real program.  Round 3's view-identity elementwise spellings
#: exist to fill the guards' empty regions ("Round 3 supplies the
#: no-op spellings directly"); every round-4 workload spells one
#: wrap / mirror / shared-factor / grammar / scalar-corner pattern.
#: ``intake.candidates`` reads it to set ``Workload.purpose_built``;
#: ``real_workloads`` filters on the flag.
PURPOSE_BUILT: frozenset[str] = frozenset(
    {
        # --- round 3 — view-identity elementwise spellings ----------
        "BroadcastPadLeft",
        "BroadcastPadRight",
        "BroadcastPadSub",
        "FullSliceScale",
        "NoopTransposeScale",
        "NoopTransposeAdd",
        "InertReshapeScale",
        "SingleChunkScale",
        "SquaredDistance",
        # --- round 4 — unsqueeze-wrap (gated broadcast) -------------
        "ChannelGateBroadcast",
        "LiftedScalarScale",
        "HeadGateBroadcast",
        "LiftedScalarScaleRight",
        "LiftedScalarCenter",
        "ContrastiveCenter",
        "PairwiseSubLift",
        # --- round 4 — chunked-projection mirrors -------------------
        "ScaledChunkProjection",
        "SingleChunkGate",
        "ChunkHalfScale",
        # --- round 4 — transposed-add mirrors -----------------------
        "NoopTransposeResidual",
        "ScalarTransposeBias",
        "Rank3TransposeAdd",
        # --- round 4 — select naturality ----------------------------
        "SelectGateSum",
        "SelectGateDiff",
        # --- round 4 — shared-factor algebra ------------------------
        "SharedFactorMixture",
        "SharedFactorContrast",
        "SharedFactorMixtureRight",
        "SharedFactorContrastRight",
        "QuadraticFeature",
        # --- round 4 — reshape / neg / exp / square grammar ---------
        "DoubleReshapeHead",
        "NegatedSum",
        "NegDistributeHead",
        "SubNegBias",
        "NegatedScale",
        "ExpProductHead",
        "ExpSumHead",
        "SquareNegHead",
        "SquareMulHead",
        "SigmoidNegGate",
        "PowOneHead",
        "SubAddFactorHead",
        "DivAddHead",
        # --- round 4 — scalar-corner annihilators -------------------
        "ScalarAnnihilator",
        "ScalarSelfCancel",
        "ScalarSelfRatio",
        "ScalarInverseSum",
    }
)

# ---------------------------------------------------------------------------
#  Meta-game — arm inventory, player vocabulary, priors and score
# ---------------------------------------------------------------------------

#: The generator inventory in its fixed enumeration order — the
#: five ``pipeline.propose`` sources plus ``"build"`` (the
#: construction player; one draw is one play).
GENERATOR_ORDER: tuple = (
    "census-naturality",
    "census-mixed-view",
    "pattern-recognition",
    "shape-aware",
    "algebraic-grammar",
    "build",
)

#: The corpus-mutating arms — a draw is one verified generated
#: workload ingested into the arena corpus (``workload_gen``:
#: undirected resample/mutate; ``gap_gen``: a witness workload
#: synthesized for a ``no-instance`` candidate, which is then
#: re-adjudicated under the grown corpus).  They mutate the
#: evidence scope, so they sit outside the pipeline's enumeration;
#: the enumeration control schedules them last (fixed inventory
#: first, then corpus growth).
CORPUS_ARMS: tuple = ("workload_gen", "gap_gen")

#: Metavariable pool — distinct names bind distinct subterms.
MV_NAMES: tuple = ("U", "V", "W", "X")

#: Literal leaves offered as construction actions.
CONST_LEAVES: tuple = (0, 1)

#: Fixed prior weights added to the learned logits.  Each name is a
#: corpus-informed bias the policy may learn to override:
#:
#: * ``child`` — census child frequency (op, metavariable *and*
#:   const children are counted — the census's ``·``/``const``
#:   entries);
#: * ``reuse`` — "the op occurs on the other side";
#: * ``bound_mv`` — "metavariable already bound" (RHS);
#: * ``fresh_mv`` — "fresh metavariable" (LHS — distinct bindings
#:   are what naturality and factoring preconditions need);
#: * ``commute`` — RHS root repeats the LHS root op;
#: * ``swap_nest`` — RHS root lifts an op one level down on the LHS
#:   (the naturality move ``f(g u, g v) -> g(f u v)``);
#: * ``depth_decay`` — op-placement prior decays with hole depth:
#:   the corpus's frequent shapes are shallow (≤ ~3 ops), so below
#:   the seeded skeleton level leaves dominate and constructed LHSs
#:   stay applicable.
PRIOR_WEIGHTS: dict[str, float] = {
    "child": 2.0,
    "reuse": 2.0,
    "bound_mv": 1.2,
    "fresh_mv": 0.4,
    "commute": 1.0,
    "swap_nest": 1.2,
    "depth_decay": 0.55,
}

#: The referee's score composition — ``base + fire·min(fires,
#: fire_cap) + paid·paid + rel_drop·rel_drop``, applied only to a
#: *true* candidate (the truth gate already scored a false one 0).
REFEREE_SCORE: dict[str, int | float] = {
    "base": 1.0,
    "fire": 0.2,
    "fire_cap": 40,
    "paid": 2.0,
    "rel_drop": 10.0,
}

#: The arena reward composition — partial credit per cleared gauntlet
#: stage, a bonus on the honest ``usable`` verdict, and the holdout
#: firing/pay columns (the same ``0.2·fires + 2·paid`` shape the
#: meta-game's referee scored).
ARENA_REWARD: dict[str, float] = {
    "stage": 1.0,
    "usable": 2.0,
    "fire": 0.2,
    "paid": 2.0,
}

# ---------------------------------------------------------------------------
#  The proposal corpus — term specs (the spec language is documented
#  in the module docstring above)
# ---------------------------------------------------------------------------

#: The curated real terms ``proposal.seed_terms`` mines: matmul
#: bracketing, elementwise duplication, weight-merge, stacked linear,
#: swiglu, an attention path, and the elementwise-algebra set.
SEED_TERMS: tuple = (
    # matmul bracketing (associativity)
    (
        "matmul",
        ("var", "a", (3, 5)),
        ("matmul", ("var", "b", (5, 2)), ("var", "c", (2, 3))),
    ),
    # elementwise duplication (CSE)
    (
        "add",
        ("square", ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "mul",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
        ),
    ),
    # weight-merge / distribute
    (
        "add",
        ("matmul", ("var", "x", (4, 4)), ("param", "W1", (4, 4))),
        ("matmul", ("var", "x", (4, 4)), ("param", "W2", (4, 4))),
    ),
    (
        "matmul",
        ("param", "W", (4, 4)),
        ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
    ),
    # stacked linear
    (
        "linear",
        ("linear", ("var", "x", (4, 4)), ("param", "W1", (4, 4))),
        ("param", "W2", (4, 4)),
    ),
    # swiglu
    (
        "mul",
        (
            "silu",
            ("matmul", ("var", "x", (4, 4)), ("param", "W1", (4, 4))),
        ),
        ("matmul", ("var", "x", (4, 4)), ("param", "W2", (4, 4))),
    ),
    # attention path
    (
        "matmul",
        (
            "softmax",
            (
                "matmul",
                ("var", "q", (2, 8, 4)),
                (
                    "transpose",
                    ("var", "k", (2, 8, 4)),
                    {"dim0": -2, "dim1": -1},
                ),
            ),
            {"dim": -1},
        ),
        ("var", "v", (2, 8, 4)),
    ),
    # elementwise algebra
    (
        "add",
        ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
        ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
    ),
    (
        "add",
        ("neg", ("var", "x", (4, 4))),
        ("neg", ("var", "y", (4, 4))),
    ),
    (
        "mul",
        ("exp", ("var", "x", (4, 4))),
        ("exp", ("var", "y", (4, 4))),
    ),
    ("sub", ("var", "x", (4, 4)), ("neg", ("var", "y", (4, 4)))),
    ("square", ("neg", ("var", "x", (4, 4)))),
    ("add", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
    ("mul", ("var", "x", (4, 4)), 0),
    ("pow", ("var", "x", (4, 4)), 1),
    ("sub", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
)

#: The algebraic-grammar schema library — ``(label, lhs, rhs, note)``
#: rows ``proposal.schema_candidates`` instantiates on the ``x``/``y``/
#: ``z`` leaves.  The set deliberately mixes *true* identities with
#: *false* ones (so the numeric oracle is exercised) and includes a
#: library duplicate (so the structural classifier is exercised).
GRAMMAR_SCHEMAS: tuple = (
    # -- true: elementwise distributivity / factoring -----------------
    (
        "mul_factor",
        (
            "add",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        (
            "mul",
            ("var", "x", (4, 4)),
            ("add", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
        ),
        "x*y + x*z = x*(y+z)  (fewer nodes)",
    ),
    (
        "mul_distribute",
        (
            "mul",
            ("var", "x", (4, 4)),
            ("add", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
        ),
        (
            "add",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        "reverse (expands)",
    ),
    (
        "mul_factor_right",
        (
            "add",
            ("mul", ("var", "y", (4, 4)), ("var", "x", (4, 4))),
            ("mul", ("var", "z", (4, 4)), ("var", "x", (4, 4))),
        ),
        (
            "mul",
            ("add", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
            ("var", "x", (4, 4)),
        ),
        "right-slot variant",
    ),
    (
        "neg_factor",
        (
            "add",
            ("neg", ("var", "x", (4, 4))),
            ("neg", ("var", "y", (4, 4))),
        ),
        ("neg", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        "-x + -y = -(x+y)",
    ),
    (
        "neg_distribute",
        ("neg", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "add",
            ("neg", ("var", "x", (4, 4))),
            ("neg", ("var", "y", (4, 4))),
        ),
        "reverse (expands)",
    ),
    (
        "sub_add_factor",
        (
            "sub",
            ("sub", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("var", "z", (4, 4)),
        ),
        (
            "sub",
            ("var", "x", (4, 4)),
            ("add", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
        ),
        "(x-y)-z = x-(y+z)",
    ),
    (
        "div_add",
        (
            "div",
            ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("var", "z", (4, 4)),
        ),
        (
            "add",
            ("div", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
            ("div", ("var", "y", (4, 4)), ("var", "z", (4, 4))),
        ),
        "(x+y)/z = x/z + y/z",
    ),
    (
        "square_neg",
        ("square", ("neg", ("var", "x", (4, 4)))),
        ("square", ("var", "x", (4, 4))),
        "(-x)^2 = x^2",
    ),
    (
        "exp_factor",
        (
            "mul",
            ("exp", ("var", "x", (4, 4))),
            ("exp", ("var", "y", (4, 4))),
        ),
        ("exp", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        "e^x e^y = e^(x+y)",
    ),
    (
        "exp_distribute",
        ("exp", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "mul",
            ("exp", ("var", "x", (4, 4))),
            ("exp", ("var", "y", (4, 4))),
        ),
        "reverse (expands)",
    ),
    (
        "mul_neg",
        ("mul", ("neg", ("var", "x", (4, 4))), ("var", "y", (4, 4))),
        ("neg", ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        "(-x)y = -(xy)",
    ),
    (
        "square_mul",
        ("square", ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "mul",
            ("square", ("var", "x", (4, 4))),
            ("square", ("var", "y", (4, 4))),
        ),
        "(xy)^2 = x^2 y^2",
    ),
    (
        "sigmoid_neg",
        ("sigmoid", ("neg", ("var", "x", (4, 4)))),
        ("sub", 1, ("sigmoid", ("var", "x", (4, 4)))),
        "sigma(-x) = 1 - sigma(x)",
    ),
    # -- true: annihilators / identity elements -----------------------
    ("mul_zero", ("mul", ("var", "x", (4, 4)), 0), 0, "x*0 = 0"),
    ("mul_zero_left", ("mul", 0, ("var", "x", (4, 4))), 0, "0*x = 0"),
    (
        "pow_one",
        ("pow", ("var", "x", (4, 4)), 1),
        ("var", "x", (4, 4)),
        "x^1 = x",
    ),
    (
        "sub_self",
        ("sub", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
        0,
        "x - x = 0",
    ),
    (
        "add_inv",
        ("add", ("var", "x", (4, 4)), ("neg", ("var", "x", (4, 4)))),
        0,
        "x + (-x) = 0",
    ),
    (
        "div_self",
        ("div", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
        1,
        "x / x = 1",
    ),
    # -- duplicate of a library law (classifier control) --------------
    (
        "sub_to_add_dup",
        ("sub", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
        ("add", ("var", "x", (4, 4)), ("neg", ("var", "y", (4, 4)))),
        "already in the library",
    ),
    # -- false identities (numeric-oracle controls) -------------------
    (
        "FALSE_mul_factor",
        (
            "add",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        (
            "mul",
            ("var", "x", (4, 4)),
            ("add", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        "false: x*y + x*z != x*(x+z)",
    ),
    (
        "FALSE_exp_add",
        ("exp", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "add",
            ("exp", ("var", "x", (4, 4))),
            ("exp", ("var", "y", (4, 4))),
        ),
        "false: e^(x+y) != e^x + e^y",
    ),
    (
        "FALSE_square_add",
        ("square", ("add", ("var", "x", (4, 4)), ("var", "y", (4, 4)))),
        (
            "add",
            ("square", ("var", "x", (4, 4))),
            ("square", ("var", "y", (4, 4))),
        ),
        "false: (x+y)^2 != x^2 + y^2",
    ),
)

#: The shape-aware schema library — ``(name, lhs, rhs, note, family)``
#: rows ``shape_proposal.schemas`` returns as ``Schema`` records.
#: Metavariable leaves are the bare strings (``A``/``B``/``C``/``X``);
#: attr metavariables ride the trailing dict (``dim="D"`` …).  Some
#: are deliberate library duplicates (the ``*_factor`` controls) so
#: the duplicate detector is exercised.
SHAPE_SCHEMAS: tuple = (
    # -- elementwise factorization (targets add(mul,mul), sub(mul,mul)) --
    (
        "factor_left",
        ("add", ("mul", "A", "B"), ("mul", "A", "C")),
        ("mul", "A", ("add", "B", "C")),
        "x*y + x*z = x*(y+z) (shared left factor).",
        "elementwise-factor",
    ),
    (
        "factor_right",
        ("add", ("mul", "B", "A"), ("mul", "C", "A")),
        ("mul", ("add", "B", "C"), "A"),
        "y*x + z*x = (y+z)*x (shared right factor).",
        "elementwise-factor",
    ),
    (
        "factor_mid",
        ("add", ("mul", "A", "B"), ("mul", "C", "B")),
        ("mul", ("add", "A", "C"), "B"),
        "x*y + z*y = (x+z)*y (shared right factor, y).",
        "elementwise-factor",
    ),
    (
        "factor_sub_left",
        ("sub", ("mul", "A", "B"), ("mul", "A", "C")),
        ("mul", "A", ("sub", "B", "C")),
        "x*y - x*z = x*(y-z).",
        "elementwise-factor",
    ),
    (
        "factor_sub_right",
        ("sub", ("mul", "A", "B"), ("mul", "C", "B")),
        ("mul", ("sub", "A", "C"), "B"),
        "x*y - z*y = (x-z)*y.",
        "elementwise-factor",
    ),
    # -- select / slice naturality (targets mul(select,select)) --
    (
        "select_mul",
        (
            "mul",
            ("select", "A", {"dim": "D", "index": "I"}),
            ("select", "B", {"dim": "D", "index": "I"}),
        ),
        (
            "select",
            ("mul", "A", "B"),
            {"dim": "D", "index": "I"},
        ),
        "mul commutes with select: sel(x)*sel(y)=sel(x*y).",
        "select-naturality",
    ),
    (
        "select_add",
        (
            "add",
            ("select", "A", {"dim": "D", "index": "I"}),
            ("select", "B", {"dim": "D", "index": "I"}),
        ),
        (
            "select",
            ("add", "A", "B"),
            {"dim": "D", "index": "I"},
        ),
        "add commutes with select: sel(x)+sel(y)=sel(x+y).",
        "select-naturality",
    ),
    (
        "select_sub",
        (
            "sub",
            ("select", "A", {"dim": "D", "index": "I"}),
            ("select", "B", {"dim": "D", "index": "I"}),
        ),
        (
            "select",
            ("sub", "A", "B"),
            {"dim": "D", "index": "I"},
        ),
        "sub commutes with select: sel(x)-sel(y)=sel(x-y).",
        "select-naturality",
    ),
    (
        "slice_mul",
        (
            "mul",
            ("slice", "A", {"dim": "D", "start": "S0", "end": "E0"}),
            ("slice", "B", {"dim": "D", "start": "S0", "end": "E0"}),
        ),
        (
            "slice",
            ("mul", "A", "B"),
            {"dim": "D", "start": "S0", "end": "E0"},
        ),
        "mul commutes with slice (equal range).",
        "slice-naturality",
    ),
    # -- layout (targets transpose(reshape), reshape(reshape)) --
    (
        "reshape_reshape",
        ("reshape", ("reshape", "A", {"shape": "S1"}), {"shape": "S2"}),
        ("reshape", "A", {"shape": "S2"}),
        "consecutive reshapes fuse.",
        "layout",
    ),
    (
        "reshape_transpose",
        (
            "transpose",
            ("reshape", "A", {"shape": "S"}),
            {"dim0": "D", "dim1": "I"},
        ),
        (
            "reshape",
            ("transpose", "A", {"dim0": "D", "dim1": "I"}),
            {"shape": "S"},
        ),
        "conjecture — false in general (oracle must reject).",
        "layout",
    ),
    # -- elementwise algebra (targets mul(silu,linear), neg/dist) --
    (
        "neg_add",
        ("add", ("neg", "A"), ("neg", "B")),
        ("neg", ("add", "A", "B")),
        "-x + -y = -(x+y).",
        "elementwise-algebra",
    ),
    (
        "sub_neg",
        ("sub", "A", ("neg", "B")),
        ("add", "A", "B"),
        "x - (-y) = x + y.",
        "elementwise-algebra",
    ),
    (
        "mul_neg_left",
        ("mul", ("neg", "A"), "B"),
        ("neg", ("mul", "A", "B")),
        "(-x)*y = -(x*y).",
        "elementwise-algebra",
    ),
    (
        "exp_add",
        ("mul", ("exp", "A"), ("exp", "B")),
        ("exp", ("add", "A", "B")),
        "e^x * e^y = e^(x+y).",
        "elementwise-algebra",
    ),
    (
        "square_neg",
        ("square", ("neg", "A")),
        ("square", "A"),
        "(-x)^2 = x^2.",
        "elementwise-algebra",
    ),
    # -- linear/matmul factor (library controls; already shipped) --
    (
        "linear_factor",
        ("add", ("linear", "X", "A"), ("linear", "X", "B")),
        ("linear", "X", ("add", "A", "B")),
        "dup of weight_factor_linear (control).",
        "linear-control",
    ),
    (
        "matmul_factor",
        ("add", ("matmul", "X", "A"), ("matmul", "X", "B")),
        ("matmul", "X", ("add", "A", "B")),
        "dup of weight_factor_matmul (control).",
        "linear-control",
    ),
    # -- constants (annihilator / identity; targets are rare) --
    ("mul_zero", ("mul", "A", 0), 0, "x*0 = 0.", "annihilator"),
    (
        "id_mul_lit",
        ("mul", "A", 1),
        "A",
        "x*1 = x (dup of id_mul).",
        "annihilator",
    ),
    ("sub_self", ("sub", "A", "A"), 0, "x - x = 0.", "annihilator"),
)

#: The 11 proposed candidate laws — ``(name, lhs, rhs, law-note)``
#: rows ``impact.new_laws`` mints as ``Rewrite``s (names prefixed
#: ``cand_`` so a run can never shadow a shipped rule).  All are
#: unconditional; the fp caveats are noted in the proposal retro.
CANDIDATE_LAWS: tuple = (
    (
        "cand_mul_factor",
        ("add", ("mul", "x", "y"), ("mul", "x", "z")),
        ("mul", "x", ("add", "y", "z")),
        "x*y + x*z = x*(y+z)  (factoring).",
    ),
    (
        "cand_mul_factor_right",
        ("add", ("mul", "y", "x"), ("mul", "z", "x")),
        ("mul", ("add", "y", "z"), "x"),
        "y*x + z*x = (y+z)*x  (right-slot factoring).",
    ),
    (
        "cand_neg_factor",
        ("add", ("neg", "x"), ("neg", "y")),
        ("neg", ("add", "x", "y")),
        "-x + -y = -(x+y).",
    ),
    (
        "cand_square_neg",
        ("square", ("neg", "x")),
        ("square", "x"),
        "(-x)^2 = x^2.",
    ),
    (
        "cand_exp_factor",
        ("mul", ("exp", "x"), ("exp", "y")),
        ("exp", ("add", "x", "y")),
        "e^x * e^y = e^(x+y).",
    ),
    ("cand_mul_zero", ("mul", "x", 0), 0, "x * 0 = 0."),
    ("cand_mul_zero_left", ("mul", 0, "x"), 0, "0 * x = 0."),
    ("cand_pow_one", ("pow", "x", 1), "x", "x^1 = x."),
    ("cand_sub_self", ("sub", "x", "x"), 0, "x - x = 0."),
    (
        "cand_add_inv",
        ("add", "x", ("neg", "x")),
        0,
        "x + (-x) = 0.",
    ),
    ("cand_div_self", ("div", "x", "x"), 1, "x / x = 1."),
)

#: The synthetic control corpus — ``(name, term spec, var names)`` —
#: each candidate law's own LHS instantiated on small, nonzero fp64
#: leaves (the ``x``/``y``/``z`` specs are ``(4, 4)`` Vars).  The var
#: names order the case's ``inputs``/``feed``.
SYNTHETIC_CASES: tuple = (
    (
        "cand_mul_factor",
        (
            "add",
            ("mul", ("var", "x", (4, 4)), ("var", "y", (4, 4))),
            ("mul", ("var", "x", (4, 4)), ("var", "z", (4, 4))),
        ),
        ("x", "y", "z"),
    ),
    (
        "cand_mul_factor_right",
        (
            "add",
            ("mul", ("var", "y", (4, 4)), ("var", "x", (4, 4))),
            ("mul", ("var", "z", (4, 4)), ("var", "x", (4, 4))),
        ),
        ("x", "y", "z"),
    ),
    (
        "cand_neg_factor",
        (
            "add",
            ("neg", ("var", "x", (4, 4))),
            ("neg", ("var", "y", (4, 4))),
        ),
        ("x", "y"),
    ),
    (
        "cand_square_neg",
        ("square", ("neg", ("var", "x", (4, 4)))),
        ("x",),
    ),
    (
        "cand_exp_factor",
        (
            "mul",
            ("exp", ("var", "x", (4, 4))),
            ("exp", ("var", "y", (4, 4))),
        ),
        ("x", "y"),
    ),
    ("cand_mul_zero", ("mul", ("var", "x", (4, 4)), 0), ("x",)),
    ("cand_mul_zero_left", ("mul", 0, ("var", "x", (4, 4))), ("x",)),
    ("cand_pow_one", ("pow", ("var", "x", (4, 4)), 1), ("x",)),
    (
        "cand_sub_self",
        ("sub", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
        ("x",),
    ),
    (
        "cand_add_inv",
        ("add", ("var", "x", (4, 4)), ("neg", ("var", "x", (4, 4)))),
        ("x",),
    ),
    (
        "cand_div_self",
        ("div", ("var", "x", (4, 4)), ("var", "x", (4, 4))),
        ("x",),
    ),
)
