# Routed defects — census attr-key hash + zero-stride conv shape rule

Two crashes the re-measure agent surfaced during the guide corpus-arms
work, both reachable off `gap_gen`/`workload_gen` draws.  Both were
crash-on-contact — neither let a bad term through — and both fixes are
at the root.

## 1. `corpus_stats` — unhashable attr values

**Symptom.**  `TypeError: unhashable type: 'list'` mining a corpus that
contains a list-valued attr.

**Root cause.**  `workload_gen.corpus_stats` keyed the observed-attrs
distribution on `tuple(sorted(s.attrs.items()))` — raw values, so a
`pad=[1,1]` or `split(sizes=[2,2])` node (the `gap_gen` `sizes`
candidates mint exactly that) could not be hashed.  The same key
problem was already solved twice: `ir._attr_key` (interning/identity)
and `census._attr_key` (shape keys) both repr-stand-in for unhashable
values.

**Fix.**  `attr_dicts` now counts `census._attr_key(s.attrs)` — the same
key semantics `shape_key` already applies to `sub_keys`/`root_keys` in
this very function.  A repr stand-in is not a value, so resampling
reads the dict back through a new `attr_exemplars` map (key → observed
attrs dict); `_sample_attrs` is the one consumer, shared by
`_Resampler._attrs` and `_swap`, and falls back to `dict(key)` for
hand-built stats whose keys are raw values (the climb_a2 fixtures).

## 2. `_infer_op_shape` — `// st` on a zero stride

**Symptom.**  `ZeroDivisionError` out of `gap_gen.synthesize`: the
generic attr-candidate fallback (`return [0, 1]`) binds a `stride`
metavar to `0`, and the conv1d/conv2d shape rules then divided by it —
*outside* the `try`/`except` around `_term_instantiate`, so the whole
synthesis loop died.

**Root cause.**  The window math `(n + 2*pd - dl*(k-1) - 1) // st + 1`
trusted the attr blindly.  A non-positive stride is not a shape the
rule cannot see — it is a conv torch refuses at runtime.

**Fix.**  New `_conv_window` helper returns three honest answers: the
extent, `None` when the dim or spelling is undecidable (metavar attrs
stay strings until instantiate), and `_INVALID` when stride or
dilation is a concrete non-positive int — the provably-ill-typed
verdict that poisons the whole term (`_shape_of` propagates it through
enclosing ops) instead of washing out as "unknown".  `_conv_scalar` /
`_conv_pair` do the scalar-vs-tuple attr normalization (a 1-tuple
`stride=[2]` still shapes conv1d; a wrong-length conv2d tuple declines
to unknown dims rather than `IndexError`).  conv1d and conv2d shared
the defect; both now route through the helpers.

## Verdict — was a bad term ever minted?

No, on both counts — the crash-on-contact behavior is the proof, not a
coincidence:

* The list-attr defect was purely on the *read* path (`corpus_stats`
  mines attested terms).  Minting list attrs is legitimate — `Op.make`
  accepts them and `ir._attr_key` repr-keys them for interning — and
  post-fix resampling reproduces the observed spelling verbatim, so no
  `"...repr..."`-as-string attr can be minted silently.
* The zero-stride crash fired inside `synthesize` *before* any term
  was emitted; and even if shape inference had returned something, the
  eval gate (`eval_term` on a conv that torch refuses) would have
  rejected it.  `mutant_term`/`resampled_term` cannot invent a zero
  stride at all — they only resample corpus-attested attr dicts.
  Post-fix the same draw declines honestly: `_shape_of` reports
  `INVALID`, `synthesize` skips, `valid_term` rejects.

Gates: `pytest tests/test_discovery_experiments.py
tests/test_discovery_climb_a2.py -q` (128 passed) + `ty`/`ruff`/`radon`
clean (no baseline update — the helper extraction nets `_infer_op_shape`
*below* its pinned 288).
