# GroupNorm schema + ConvNeXt — the export-boundary fix, 5 new op-tuples, still 0 shippable

`law-corpus-expansion.md` §6 listed the smallest unblocked item first:
**the `group_norm` attr schema**.  `nn.GroupNorm` had been probed and
rejected during the corpus expansion because its export carries a
positional attr (`arg5`) that `ATTR_SCHEMA` could not name — mint died
with "positional attr 'arg5' has no declared canonical name".  This
retro records the one-line schema fix, the ConvNeXt block it unblocks,
and the pipeline's verdict on the enlarged corpus.

The headline is again **honest and negative-but-informative**:

* The schema gap is closed: `arg5` is `cudnn_enabled`, the same
  eps/flag tail `layer_norm` already declares.
* `ConvNeXtBlock` joins the corpus: `group_norm` enters the op
  vocabulary (40 corpus ops, was 39) with **5 new op-tuples** —
  `group_norm(conv2d,·,·)`, `conv2d(group_norm,·)`, `gelu(conv2d)`,
  `conv2d(gelu,·)`, `add(·,conv2d)`.
* The pipeline still reports **0 further shippable**, and — the honest
  part — **no proposal fires on ConvNeXtBlock at all**: no candidate's
  LHS touches `group_norm`, `conv2d` or `gelu`.  The model feeds the
  census, not the current proposer family.
* **Held-out rediscovery still passes**: `--holdout select_mul`
  re-ranks `census:mul_select` **#1 of 50**, SHIP, identical evidence
  (fires 24, paid 5, drop 25.9 %, cert pass, enode 1.27×).

Reproduce:

    .venv/bin/python tools/law_shape_census.py --top 200
    .venv/bin/python tools/law_pipeline.py --vocab derived
    .venv/bin/python tools/law_pipeline.py --vocab derived --holdout select_mul

## 1. The schema fix — `arg5` is `cudnn_enabled`

`torch.export` of `nn.GroupNorm(4, 8)` emits
`aten.group_norm(x, 4, w, b, 1e-05, False)` — positional tail
`(num_groups, weight, bias, eps, cudnn_enabled)`.  The schema already
declared `{1: "num_groups", 4: "eps"}` but not position 5, so the
exporter's `arg5=False` had no canonical name and `Op.make` rejected
the term.  The fix is the layer_norm precedent verbatim:

    "group_norm": {1: "num_groups", 4: "eps", 5: "cudnn_enabled"},

`ATTR_REQUIRED` is unchanged — `num_groups` was already required and
`cudnn_enabled` has an aten-side default, so it is *not* required
(same treatment as every flag attr).  Exported terms now carry
`group_norm(x, w, b, num_groups=4, eps=1e-05, cudnn_enabled=False)` —
fully canonical.  The rest of the path needed **no code**: the
`group_norm` binding in `_CORE_TORCH_BINDINGS` already reads
`num_groups`/`eps` and takes weight/bias as operands, the typing rule
already types it as `shapes[0]`, and `supported_ops` already lists it.
The gap was exactly the one the retro diagnosed: the boundary could
not name the flag, nothing downstream was broken.

Regression test: `tests/test_attrs.py` gained `_GroupNorm` +
`test_group_norm_eps_is_eps_not_cudnn`, the mirror of the layer_norm
regression — same bug class (eps is arg4, the flag is arg5), pinned
both ways.

## 2. The model — ConvNeXtBlock

`catopt_torch.models.ConvNeXtBlock(channels, groups, expansion)` —
the standard modern-ConvNet residual: depthwise 7×7
(`groups=channels`) → `nn.GroupNorm` → 1×1 pointwise expand → GELU →
1×1 pointwise contract → `+ x`.  Registered in
`tools/law_impact._model_cases()` as `ConvNeXtBlock(8)` on the `img`
input, the same seam every prior model uses — zero tool changes.

Exported IR (verified fp64 against the module):

    add(x, conv2d(gelu(conv2d(group_norm(conv2d(x, w_dw, …, groups=8),
                                    w_gn, b_gn, num_groups=4, eps=1e-05,
                                    cudnn_enabled=False),
                             w_pw1)), w_pw2))

Tested by `tests/test_corpus_models.py` the same three ways as every
corpus builder — forward under `eval()+no_grad`, `export_to_ir`
asserting `{conv2d, group_norm, gelu}` are present, and the lowered
`IRModule` verified fp64-exact.

## 3. The census diff

Two baselines matter, and both are reported:

* The corpus-expansion retro's published numbers: **673 op nodes,
  155 op-tuples, 363 shapes, 91 terms**.
* Just before this change the corpus already read **678 / 158 / 368,
  93 terms** — the drift is the two `LAW_CASES` builders shipped with
  the `softmax_fold` and `silu_fold` laws (both admitted after the
  retro), not a recount.
* Now: **684 op nodes, 163 op-tuples, 374 shapes, 94 terms**
  (63 bench + 31 models).

ConvNeXtBlock alone contributes **+6 op nodes, +5 op-tuples, +6
shapes**:

    NEW op-tuples: 5
      group_norm(conv2d, ·, ·)   conv2d(group_norm, ·)   gelu(conv2d)
      conv2d(gelu, ·)            add(·, conv2d)
    NEW shapes: 6  (the block's own subterm chain, incl.
      conv2d[groups=8,padding=(3,3)](a, b) — a second depthwise shape)

`group_norm` is a new corpus op — the first *normalisation* op with
learnable affine params that is neither `batch_norm` (running stats)
nor `layer_norm`/`rms_norm` (the `dim` list-arg convention).  As with
the last expansion, the new tuples are *deep chains* (conv → norm →
conv → act → conv), not the parallel-view shapes the naturality
generators consume — and the census bears that out in §4.

## 4. Pipeline results on the enlarged corpus

| run | proposals | firing on a real model | shippable |
|---|---|---|---|
| `--vocab derived` | 50 | 14 | 0 |
| `--vocab derived --holdout select_mul` | 50 | 14 | **1** (the held-out winner itself) |

Per-candidate notes:

* `census:mul_select` — still correctly `duplicate` of the shipped
  `select_mul`; with the holdout, rank 1 of 50, SHIP — **PASS**,
  identical evidence to the last three retros.
* `recognize:softmax` — now `duplicate`: the `softmax_fold` law the
  corpus expansion *enabled* has since shipped into
  `SIMPLIFICATION_RULES`.  The last retro's §6 item 2 — "exp/sum →
  softmax recognition … ManualSoftmaxAttention gives the corpus a
  real instance" — already paid out.  This is the corpus investment
  compounding: one expansion produced a shipped law.
* `mixed:*` proposals dominate the table — the mixed-view generator
  family shipped since the last retro explains most of the proposal
  growth (35 → 50); none is conv-family.
* `grammar:mul_distribute` — still the interesting non-ship: fires 24,
  matches 12, never lowers cost, blows the closure 35–43×.
* **No candidate fires on ConvNeXtBlock** — checked per-proposal via
  `Evidence.fire_cases`: zero hits, and no proposal's LHS contains
  `group_norm`, `conv2d` or `gelu` at all.  The proposers generate
  over the algebraic/pointwise/view alphabet; conv-norm chains are
  outside every current generator's grammar.  Same lesson as the last
  retro, sharper: *a new op in the census is not a new shape a
  proposer can see.*

## 5. Honest assessment

* **No new law shipped, and the new model is not why** — ConvNeXtBlock
  feeds the census vocabulary (group_norm is now a real corpus op with
  real attr semantics — `num_groups`/`eps`/`cudnn_enabled` all
  canonical) but produces no firing candidate because no proposer
  speaks conv.  The export-boundary fix was still the right fix: it
  closes a diagnosed gap and unblocks the whole ConvNeXt family for
  any future conv-aware proposal.
* **The pipeline remains the honest instrument**: holdout PASS with
  identical evidence, and it correctly flags the one corpus-expansion
  law that did ship (`recognize:softmax` → `duplicate` of
  `softmax_fold`).
* **Where the corpus now stands**: 94 terms, 163 op-tuples, two
  vision-conv families (ResNet bn-chain, ConvNeXt gn-chain), MoE
  dispatch, manual-softmax attention.  The shapes a conv-aware or
  norm-aware proposer would need are *present*; the proposer is the
  missing ingredient, again.

## 6. Gates

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass
* `.venv/bin/ty check` — pass (0 errors)
* `.venv/bin/vulture` — pass
* `pytest tests/test_corpus_models.py tests/test_attrs.py` — **53 passed**
* `pytest tests/test_semantic_ops.py tests/test_torch_bridge_edges.py`
  — **357 passed** (the group_norm binding/ATTR_SCHEMA consumers)
* Full suite not run on this box (resource constraint); coverage
  unchanged in kind — `ConvNeXtBlock` is branch-free and exercised by
  all three corpus tests, the schema entry is data, and the new test
  pins the canonicalisation.

## 7. Where the next law might hide

1. **A conv/norm-aware proposer** — the corpus now has two conv
   families but the generators only speak pointwise/view/algebra.
   `fold(norm-affine → conv)` (the classic BN/GN-into-conv weight
   fold) is a *true* law with real instances here — the first
   candidate a conv grammar could propose.
2. **`gelu` pointwise variants** — `gelu` is now in two corpus models
   (GegluMLP, ConvNeXtBlock); a `gelu ≈ silu`-family or
   gelu-fusion schema would have real sites, though the numeric
   oracle will keep honesty (approximations are not equalities).
3. **Residual-add shapes** — `add(·, conv2d)` joins `add(batch_norm,·)`
   and `add(·, relu)`; three residual-site shapes now exist for any
   absorption/commute law over the residual spine.
