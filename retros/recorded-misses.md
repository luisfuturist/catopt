# Recorded misses — per-item ledger

This retro closes the three recorded misses — the
`corpus-expansion-r2.md` §4 binding gaps (`lstm.input`,
`instance_norm`, `upsample_nearest2d.vec`) and the
`rms-norm-law.md` §Caveats canonicalization gaps (`mul(x,x)`-spelled
squares, `1/√` and `(·)^-0.5` reciprocal-roots).  Per item: done or
skipped, the evidence, the exact implementation, what was measured,
and what remains honest limitations.

The law library went 57 → **61 rules**; all four additions are
`SIMPLIFICATION`-tagged, so they ride `DEFAULT` and fire in the
public pipeline.  Three are axioms (the rsqrt folds);
`mul_square` is `square_expand`'s recorded lemma.  All new laws use
`cond=` or are unconditioned — no new `check=` hooks.

## Binding gaps — all three DONE

### `instance_norm` — bound

`nn.InstanceNorm2d` exports `aten.instance_norm` in all four
`affine × track_running_stats` combos; the feasibility check found
the normalization family's full path already exists
(`batch_norm`/`group_norm` in `ATTR_SCHEMA`, `_shape_of`, and the
torch bindings).  Implementation mirrors `group_norm`:

* `ATTR_SCHEMA["instance_norm"]` = `{5: use_input_stats, 6:
  momentum, 7: eps, 8: cudnn_enabled}` — positions 1–4 are the
  operand slots (`w, b, rm, rv`; Nones drop at export), the scalar
  tail is canonical-named.
* `_shape_of` — `instance_norm` joins the rank-preserving
  alternation (`shapes[0] or None`), next to `batch_norm`/`group_norm`.
* `_instance_norm_torch` — the binding disambiguates the operand
  tail on `use_input_stats`: when `True` (input-stats eval, or any
  `track_running_stats=False` module) all operands are the affine
  pair; when `False` the trailing pair is `(running_mean,
  running_var)` (the eval-mode export of a `track_running_stats`
  module).  `cudnn_enabled` is a dispatch hint, dropped like
  `batch_norm`'s.
* malformed spelling `use_input_stats=False` with no stats operands
  fails loudly (aten's own `RuntimeError`), never silently
  normalizes — pinned by test.

Evidence: all four combos export → lower → `torch.equal` bitwise
on fp32; `Optimizer`-path verified under the parametrized export
tests (4 cases) + `BINDING_CASES` (5 spellings) + `SHAPE_CASES`.

### `upsample_nearest2d` — bound (via `.vec` canonicalization)

`nn.Upsample(mode="nearest")` / `F.interpolate` export
`aten.upsample_nearest2d.vec` — the two-list overload carrying
`output_size`/`scale_factors` (exactly one non-None per call).  The
feasibility check found no IR op, no schema, no binding; the fix is
the standard overload-canonicalization path (`_IR_TO_TORCH_EXTRA`),
same seam as `mul.Tensor → mul`:

* `_IR_TO_TORCH_EXTRA["upsample_nearest2d.vec"] =
  "upsample_nearest2d"` — the `.vec` suffix is an overload tag, not
  a different op.  (`upsample_nearest2d_backward` stays unbound —
  distinct op, correctly.)
* `ATTR_SCHEMA["upsample_nearest2d"] = {1: "size", 2: "scale"}` —
  the two list args land under distinct names, so the binding never
  discriminates `int[]` vs `float[]` positionally.  The `.default`
  overload's lone `output_size` shares position 1.
* Typing: `_upsample_nearest2d_shape` registered in `_SHAPE_RULES`
  (the extensible dispatch — the ratchet keeps `_infer_op_shape`
  from absorbing new arms): `(N,C,H,W) → (N,C,H′,W′)`, `size`
  verbatim, `scale` floors `input*scale`; bool/sparse spellings
  report honest `None` extents, sub-rank-4 reports the input shape.
* `_upsample_nearest2d_torch` → `F.interpolate(mode="nearest")`;
  scalar `size`/`scale` spellings broadcast to the (H,W) pair.

Evidence: `nn.Upsample(scale_factor=2)`, `Upsample(size=(5,12))`,
and `F.interpolate` spellings all export → lower bitwise-equal on
fp32; shape cases cover size/scale/scalar/unknown-extent/bool/sub-rank.

### `lstm.input` — bound (aten passthrough)

`nn.LSTM` exports `aten.lstm.input` returning the heterogeneous
`(output, h_n, c_n)` triple, consumed through `getitem` — the
tuple-valued convention `topk`/`sort`/`var_mean` already establish,
so the binding is feasible without new machinery.  What the retro
speculated as "carrier/complex" turns out to be a clean passthrough:
the two operand lists (`hx`, flat `params`) arrive flattened, the
scalar tail is schema'd, and `torch.ops.aten.lstm.input` is the
lowering.

* `ATTR_SCHEMA["lstm.input"]` names positions 3–8 (`has_biases`,
  `num_layers`, `dropout`, `train`, `bidirectional`, `batch_first`);
  positions 1–2 are tensor-list operands.
* Typing: `_lstm_input_shape` → `None` — deliberately.  The triple's
  element shapes differ `(…,D·H)` vs `(D·L,N,H)`, and the
  same-shape-element convention `topk`/`sort` lean on does not
  apply.  Honest unknown, not `x`'s shape.
* `_lstm_input_torch` — repacks `h0/c0` + flat params into the two
  aten lists and forwards the scalar tail.  The op name keeps the
  `.input` qualifier: `aten.lstm.data` (the PackedSequence variant)
  shares the `lstm` base name with a different input type — keeping
  `lstm.input` means `.data` stays *unbound and loud* rather than
  silently binding to a wrong signature.

Evidence: `batch_first`/`seq_first`/`num_layers=2`/`bidirectional`
exports all verify `torch.equal` bitwise (output element through
`getitem(0)`), attrs land canonical, `lstm.input ∈
TorchSink.supported_ops`.

All three ops now sit in `supported_ops` (the binding table IS the
hard feasibility set) and in `tools/law_impact.py::_model_cases` —
the corpus now exercises them (`InstanceNorm`, `UpsampleNearest`,
`LSTMSeq`; 44 cases, zero export errors).

## `mul → square` — DONE

`square` was already a first-class IR op (`square_expand`,
`pow_to_square`, `square_to_pow`, the `square` binding).  The missing
direction — a graph spelling x² as `mul(u,u)` could never seed the
`square`/`pow` members the RMSNorm fold needs — is now bridged:

```python
MUL_SQUARE = R(
    "mul_square",
    Op.make("mul", "u", "u"),
    Op.make("square", "u"),
    ...,
    derivation=("square_expand",),   # its definitional inverse
    tags=_SIM,
)
```

No `check`/`cond` at all — the shared `u` metavariable is the whole
precondition (the matcher binds both operands to the same e-class;
`mul(u, v)` declines).  Term-local, one member per e-class — the
`silu_fold` posture.  Verified: fires on `mul(u,u)`, declines on
mismatched operands, sound fp64 (`x·x ≡ x²`), shares one e-class
with `square_expand`, and is the load-bearing step that makes the
`mul(x,x)`-spelled RMSNorm fold (fires on the real export; the
folds-without-bridges test pins the miss was real).

## Non-canonical RMS spellings — DONE (three folds, not a pass)

`rms-norm-law.md` recorded that only `rsqrt(mean(pow(u,2))+eps)`
matches — a corpus graph spelling the reciprocal root as `1/√(·)` or
`(·)^-0.5` could never fire the fold.  Rather than a pass, three
single-direction canonicalization laws mint the `rsqrt` member —
the pattern's own strictness does the rest:

* `div_sqrt_to_rsqrt`: `div(ONE, sqrt(u)) → rsqrt(u)`,
  `cond=("const-cmp", "ONE", "==", 1)` — numeric, so `Const(1)` and
  `Const(1.0)` both fold; a `Var` numerator declines.
* `pow_to_rsqrt`: `pow(u, P) → rsqrt(u)`,
  `cond=("const-cmp", "P", "==", -0.5)` — the same guard shape the
  rms pair uses for its `P == 2` exponent.
* `recip_sqrt_to_rsqrt`: `reciprocal(sqrt(u)) → rsqrt(u)` —
  structural, no condition.  *This* is the real export spelling:
  `1.0 / t` lowers through aten to `reciprocal(t) * 1.0` (id_mul
  strips the unit), never `div(1, t)` — the recorded-miss wording
  was the *math*, and probing the boundary showed the `div` spelling
  only appears from an explicit `torch.div(1.0, ·)` call.  Adding
  the reciprocal form was the difference between covering the
  spelling and covering the intent.

Evidence: every recorded-miss spelling — `x·x` square, `1/√` (as
both `div(1,·)` and the exported `reciprocal(√·)` form),
`pow(·, -0.5)`, and the doubly-missed `x·x + 1/√` combination — lands
the `rms_norm(u, w, dim, eps)` member in the root e-class under
`ALL_RULES`, extraction *picks* it, the certificate replays, and the
lowered before/after modules agree under `sink.verify` on the real
exports (`_ManualRmsRecip`, `_ManualRmsPow` models in
`tests/test_square_rsqrt_laws.py`).  The negative control pins the
bridges are load-bearing: with the four laws removed, only the
canonical `pow(u,2)`/`rsqrt` spelling folds — exactly the recorded
miss.

Soundness: `rsqrt` IS the reciprocal root — the laws are
numerics-exact identities, verified to ~1e-15 fp64 (the kernel
rounds differently than `sqrt`+`div`, matching the
`silu_fold`-family tolerance convention).  All mint `rsqrt` — an op
`TorchSink` lowers — so no extraction can select an unlowerable
member (the hard `supported_ops` posture).

## Remaining honest limitations

* `div(ones_like(x), sqrt(u))` — the broadcast-ones numerator is
  semantically `1/sqrt(u)` but the pattern cannot see through the
  creator op; skipped (rare spelling, needs an `ones_like`-provenance
  check the DSL lacks).  Similarly `u.pow(-1).sqrt()`-style chains
  and `exp(-0.5·log(u))` spellings stay unfolded.
* `lstm.input` typing stays `None` (heterogeneous triple); if a
  `getattr`-typed tuple convention lands later, element shapes are
  recoverable from `num_layers`/`bidirectional`/x's shape.
* `aten.lstm.data` (PackedSequence) and every non-`nearest`/
  non-`vec` upsample overload remain unbound — loud boundaries,
  deliberate.
* The `upsample_nearest2d` binding reads `size`/`scale` attrs only —
  a caller minting `Op.make("upsample_nearest2d", x)` with neither
  gets `F.interpolate`'s own `RuntimeError`, which is the honest
  failure for a shape-less resize.

## Gates

Scoped per the task (no full suite):

* `ruff check packages tools` — clean; `ruff format --check` — clean
  (both limited to the format-gated surface).
* `ty check` — clean.  `vulture` — clean.
  `tools/radon_ratchet.py` — clean (the two heavier ops registered as
  `_SHAPE_RULES` handlers instead of growing `_infer_op_shape` past
  its baseline).
* Scoped pytest — `test_square_rsqrt_laws.py` (new, 15),
  `test_semantic_ops.py` (354: the BINDING_CASES/SHAPE_CASES/export
  probes for all three bindings), `test_laws_structure.py`,
  `test_law_serialize.py`, `test_cond_laws.py`, `test_contracts.py`
  (count pins updated: 61/23/138/197/46-axiom/13-lemma/43-full/
  25-unguarded; `mul_square` fuzz-specced, the rsqrt pair join the
  Const-binding carve-out), `test_rms_norm_laws.py`, `test_rules.py`,
  `test_property_laws.py`, `test_laws_rewrite_edges.py`,
  `test_silu_fold_laws.py`, `test_glu_fold_law.py`,
  `test_cost_edge_cases.py`, `test_torch_bridge_edges.py`,
  `test_torch_integration.py`, `test_meta.py` — all green.

Not committed; another agent owns `tools/` beyond `law_impact.py`.
