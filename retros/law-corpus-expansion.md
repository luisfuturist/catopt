# Corpus expansion — 8 new real architectures, 31 new op-tuples, no new law (yet)

`law-vocab-derived.md` §5 named the last human ingredient in the
proposal loop: **the corpus itself** — 22 hand-written models plus 61
bench cases.  The pipeline had already reported "0 further shippable"
on that corpus.  This retro records enlarging the corpus with eight
new *real* architectures and re-running the validated pipeline.

The headline is **honest and negative-but-informative**:

* The census grew: **673 op nodes (was 613), 155 op-tuples (was 124),
  363 shapes (was 324)** — 31 new op-tuples, 0 removed, +39 shapes.
* Eight previously-absent ops enter the corpus: `gelu`, `batch_norm`,
  `relu`, `exp`, `sum`, `arange`, `embedding`, `elu`.  The derived
  vocabulary absorbs them automatically (39 corpus ops now; unary
  pointwise grows 7 → 11 with `elu`, `exp`, `gelu`, `relu`; `sum`
  correctly classifies as a reduction).
* The pipeline still reports **0 further shippable** — but the *why*
  changed, and two candidates now fire on real models that could not
  fire before (§3).
* **Held-out rediscovery still passes**: `--holdout select_mul`
  re-ranks `census:mul_select` **#1 of 35**, SHIP, identical evidence
  (fires 24, paid 5, drop 25.9 %, cert pass, enode 1.27×).

Reproduce:

    .venv/bin/python tools/law_shape_census.py
    .venv/bin/python tools/law_pipeline.py --vocab derived
    .venv/bin/python tools/law_pipeline.py --vocab derived --holdout select_mul

## 1. What was added and why

All eight are real architecture families, placed in
`catopt_torch.models` alongside the existing builders and registered
in `tools/law_impact._model_cases` — the same seam every prior model
uses, so the census, vocab and pipeline pick them up with **zero tool
changes**.

| model | architecture | new corpus ops / shapes |
|---|---|---|
| `MoEMLP` | soft Mixture-of-Experts FFN (Mixtral-style router + expert dispatch): `softmax(linear)` router, `stack` of expert FFNs, `mul(unsqueeze(w), stack)` dispatch, `sum` over experts | `sum`, `softmax(linear)`, `stack(linear,…)`, `mul(unsqueeze, stack)`, `sum(mul)`, `unsqueeze(softmax)` |
| `GegluMLP` | PaLM/Gemma GEGLU: `down(gelu(gate(x)) * up(x))` | `gelu`, `mul(gelu, linear)` |
| `GatedResidualBlock` | highway/GRU-style gated residual: `x + σ(g)·(h − x)` | `mul(sigmoid, sub)`, `sub(linear, ·)` |
| `ResNetBlock` | ResNet basic block: conv–bn–relu–conv–bn + skip | `batch_norm`, `relu`, `batch_norm(conv2d,…)`, `relu(batch_norm)`, `conv2d(relu)`, `relu(add)`, `add(batch_norm, ·)` |
| `DepthwiseConvBlock` | MobileNet depthwise-separable: `relu(pw(dw(x)))`, `groups=ch` attr | `conv2d(conv2d)`, `conv2d(relu)`, `relu(conv2d)` |
| `ManualSoftmaxAttention` | pre-kernel attention, softmax spelled `exp / sum` | `exp`, `exp(div)`, `sum(exp)`, `div(exp, sum)`, `matmul(div, linear)` |
| `PositionalEmbedding` | learned positional embedding (BERT stem): `x + emb(arange(T))` | `arange`, `embedding`, `embedding(·, arange)`, `add(·, embedding)` |
| `KernelizedAttention` | linear-transformer feature map `φ = elu + 1`: `φ(Q) @ (φ(K)ᵀ @ V)` | `elu`, `elu(linear)`, `add(elu, const)`, `matmul(add, matmul)` |

Selection rule: each builder had to (a) be a plausible NN block, (b)
export cleanly through `torch.export` → `export_to_ir`, and (c) add
op-tuples the census did not already count.  `nn.GroupNorm` was
probed and rejected — its export carries an undeclared positional
attr (`arg5`) that `ATTR_SCHEMA` cannot yet name; that is an honest
export-boundary gap, not a model bug.

Coverage: `tests/test_corpus_models.py` exercises every builder three
ways — forward under `eval()+no_grad`, `export_to_ir` asserting the
signature op family is present, and the lowered `IRModule` verified
fp64-exact against the original module.  The `hidden or dim*2`
fallback in `MoEMLP` has an explicit default-args case (the suite is
pinned at 100 %).

## 2. The census diff

    before: 613 op nodes, 124 op-tuples, 324 shapes  (83 terms)
    after:  673 op nodes, 155 op-tuples, 363 shapes  (91 terms)

    NEW op-tuples: 31 — including
      softmax(linear)          mul(unsqueeze, stack)    sum(mul)
      batch_norm(conv2d,…,…)   relu(batch_norm)         relu(add)
      exp(div)                 sum(exp)                 div(exp, sum)
      gelu(linear)             mul(gelu, linear)
      elu(linear)              matmul(add, matmul)
      embedding(·, arange)     arange()
      mul(sigmoid, sub)        conv2d(relu)             conv2d(conv2d)
      add(batch_norm, ·)       add(elu, const)          add(·, embedding)
      sub(linear, ·)           matmul(div, linear)      matmul(linear, transpose)
      matmul(transpose, linear) linear(silu, ·)         transpose(add)
      transpose(linear)        unsqueeze(softmax)
    GONE: 0

The `f(g, g)` pattern the census-naturality generator consumes did
**not** appear in a new form — none of the 31 tuples is a binary
pointwise op over two identical view ops.  The new shapes are
*structural* (MoE dispatch, conv-norm chains, manual softmax), so the
yield shows up in matches/firings of existing schemas rather than in
brand-new naturality candidates.

## 3. Pipeline results on the enlarged corpus

| run | proposals | firing on a real model | shippable |
|---|---|---|---|
| `--vocab hand` | 35 | 5 | 0 |
| `--vocab derived` | 35 | 5 | 0 |
| `--vocab derived --holdout select_mul` | 35 | 5 | **1** (the held-out winner itself) |

Per-candidate notes on the enlarged corpus:

* `census:mul_select` — still #1, correctly `duplicate` (it IS
  `select_mul`, shipped).  With the holdout: rank 1, SHIP — PASS.
* `linear_factor` — duplicate, fires 2, pays 2.
* `reshape_transpose` — fires 23 but the **numeric oracle rejects
  it** (a transpose is not a reshape): the truth gate working.
* `grammar:mul_distribute` — **the interesting one**.  The new gated-
  residual / MoE dispatch shapes gave it real targets: it now fires
  24× and matches 12 sites.  It still does not ship — it never lowers
  extracted cost AND blows the closure 36–43× over the 2.0 limit.
  Corpus expansion moved it from "inapplicable" to "fires but unsafe
  + worthless", which is a genuinely better-understood verdict.
* `grammar:sub_to_add_dup` — new `sub` shapes (gated residual) gave it
  its first real firing (1); duplicate anyway.
* The eight new-op schema candidates (`exp_add`, `square_neg`, …) stay
  inapplicable — the ops now exist, but the *equalities* still have no
  real instances.  Presence of an op ≠ presence of a law shape.  This
  is the same lesson `law-impact.md` recorded, now confirmed at
  corpus scale: **op-level novelty and shape-level novelty are
  different currencies.**

## 4. Honest assessment

* **No new law shipped.**  The task's success criterion was feeding
  the pipeline new shapes; the pipeline's verdict is that none of the
  reachable equalities on the enlarged corpus is simultaneously true,
  new, firing, paying, verified and closure-safe.  Saying more would
  be overstating it.
* **The corpus is the bottleneck, not the pipeline.**  The pipeline
  demonstrably finds winners when they exist (holdout PASS with
  identical evidence).  What it cannot do is invent distributive/
  naturality structure the models do not contain — most of the new
  tuples are *deep* (MoE dispatch, conv-norm chains) rather than the
  *parallel-view* shapes naturality laws consume.
* **One export-boundary gap found**: `nn.GroupNorm` positional attrs.
  A future `ATTR_SCHEMA` entry for `group_norm` would unblock
  ConvNeXt-style models.
* **Coverage stayed honest**: `batch_norm`/`embedding`/`arange` are
  vocabulary-visible but property-undecidable (`?` in law_vocab) —
  the derived alphabet did not weaken.

## 5. Gates

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass
* `.venv/bin/ty check` — pass (0 errors)
* `.venv/bin/vulture` — pass
* `tools/radon_ratchet.py` — pass (1903 functions)
* `uv run pytest -q` — **3367 passed, 31 skipped** (~7.7 min)
* `coverage` — pinned at 100 %; new builders fully exercised by
  `tests/test_corpus_models.py` (forward + export + lowered verify +
  default-args branch)

## 6. Where the next law might hide

1. **A naturality for `unsqueeze`/`stack` composition** — the MoE
   dispatch shape `mul(unsqueeze(w), stack(outs))` is *the* grouped-
   GEMM fusion target; it needs a generator that proposes
   `f(v(x), g(y,…))` mixed-view shapes, not only `f(g,g)`.
2. **`exp/sum → softmax` recognition** — `ManualSoftmaxAttention`
   gives the corpus a real instance; a law `div(exp(s), sum(exp(s)))
   → softmax(s)` is true and would fold legacy attention into the
   kernel path.  It is a *shape* proposal beyond view naturality —
   the grammar proposer's next extension.
3. **`group_norm` attr schema** — small bridge fix, unblocks the
   ConvNeXt family.
