# Retro — intake round 2: 166 real workloads, 0 shippable laws

**Status:** landed (`tools/law_intake.py`, `torch_bridge.py`,
`attrs.py`); pipeline delta is an honest negative.

## What landed

`law_intake.py` grew a three-class intake: **ingested** (exported +
all ops bound + fp64-verified lowered module), **census-only**
(exported, missing bindings — the backlog), **rejected** (export
boundary raised; exception recorded verbatim). The candidate table
now covers ~40 real `nn.*` spellings nobody hand-wrote:
pooling (`MaxPool2d`, `AdaptiveAvgPool2d`), `Upsample`
(nearest/bilinear), `Unfold`/`Fold`, `BatchNorm1d/2d`, `GroupNorm`,
`LayerNorm`, `InstanceNorm`, `LocalResponseNorm`, `ConvTranspose2d`,
`Conv3d`, `PixelShuffle`, `Dropout2d`, `Embedding`, `Bilinear`,
`CosineSimilarity`, `CrossEntropyLoss`, `MSELoss`, RNN/GRU/LSTM
families, attention variants, and compound models.

### Corpus growth (vs. post-round-1)

| metric | before | after |
|---|---|---|
| workloads | 44 cases | **166** |
| census terms | 145 | **276** |
| op-tuples | 290 | **588** |
| shapes | 730 | **1084** |
| op nodes | 1167 | **1586** |

### New binding/lowering work (torch_bridge + attrs)

- `fill_.Tensor`/`fill_.Scalar`/`masked_fill_*`/`zero_` — the
  copy_-family mutation threading (BatchNorm buffer writes, loss
  masks); `_fill_src` materialises the scalar write as a term, and
  drops are honest rejections, not silent mints.
- `special_gammaln` → `gammaln`; `__or__`/`__and__` → the bound
  bitwise ops.
- **`take_along_dim` attr-schema bug** — `dim` sits at arg-position
  **2** (the index tensor is position 1), not 1; the schema was
  name-firing on the wrong operand.  Latent defect, real fix.

### Outcomes

- **114 ingested** (verify fp64-clean), **52 census-only**,
  **2 rejections**: `nn.CTCLoss` (known dynamic-output) and
  `nn.GaussianNLLLoss` (same family).  0 verify failures — round 1's
  three defect classes stayed fixed.

## Pipeline delta — the honest part

66 candidates proposed, 33 firing on a real model, **0 shippable**.

The top rankers are `mixed:`-family view/index identities
(`add_getitem_l_id`, `mul_select_l_id`, …): most stay *unproven*
(no oracle for the view-manipulation class), several are *false*
(`add_select_r_id`, `mul_transpose_l_id`, `eq_getitem_l_id` —
numeric rejects: elementwise×indexed views don't distribute
naively), and the recognizable folds (`mul_select`,
`linear_factor`, `softmax`) are already shipped duplicates.

**Read:** the easy laws are mined.  A 3× corpus now surfaces
view/index identities whose verification needs a *different*
oracle (gather/index semantics, not elementwise numeric equality)
rather than more shapes.  That — not another intake round — is the
next yield gate.
