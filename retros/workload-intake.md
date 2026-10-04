# Workload intake — 38 real programs through the export boundary, +84 op-tuples, 3 lowering defects, 0 shippable

`law-workload-gen.md` closed generation-as-sampling and named the
still-untested half of self-play: **real workload intake** — ingesting
programs not written by hand into the census.  `corpus-expansion-r2.md`
proved real architectures produce new law candidates; both fold laws
shipped from corpus additions.  This retro builds the intake path,
`tools/law_intake.py`: candidate `nn.Module`s → `torch.export` →
`export_to_ir` → classify → persist → union into the census and the
pipeline.

The headline is the established law holding for a fourth time, plus
the intake path paying off in a currency no prior round produced:

* **The intake path works.**  39 candidates, 38 exported, **17
  ingested** (bound + lowered + verified fp64), 18 census-only, 3
  verify-failed, 1 export rejection.  The census grew **206 → 290
  op-tuples (+84)** — more than corpus-expansion r2 (+40) and
  workload-gen mutation (+25) combined — and **431 → 730 shapes
  (+299)**, 774 → 1167 op nodes, 107 → 145 terms, **33 new ops**.
  (Measured on the 41-model corpus; the tree then moved again —
  §5.)
* **The pipeline yield is honest and consistent**: proposals 54 → 58
  (+4, all census-mixed-view), **0 → 0 shippable**.  Real structures
  (MoE dispatch masks, transformer reshape/permute chains, loss
  tails) fire candidates — and the equalities are still unproven or
  false.  Presence ≠ law, again.
* **The real yield is the defect list**: three workloads export and
  bind cleanly but **verify wrong** — `nn.MultiheadAttention`,
  `nn.TransformerDecoderLayer`, `nn.LSTM` — lowering defects no test
  caught, plus a measured 17-op binding-gap backlog and one hard
  export rejection.
* **Held-out rediscovery passes on the enlarged corpus**:
  `--holdout select_mul` re-ranks `census:mul_select` **#1 of 58**,
  SHIP, evidence unchanged (fires 24, paid 5, drop 25.9 %, cert
  pass, enode 1.27×).

Environment note for reproducibility: there is **no torchvision,
timm or transformers in the env** (`uv pip list`: torch 2.14 +
numpy).  The "real library code" leg of intake is therefore torch's
own `nn.*` modules — which turned out to be the richer source
anyway: library spellings (`permute`, `unflatten`, `dropout`,
`split_with_sizes`, `feature_dropout`, packed-QKV attention) are
exactly the spellings the hand corpus never wrote.

Reproduce:

    .venv/bin/python tools/law_intake.py --json /tmp/intake.json
    .venv/bin/python tools/law_shape_census.py --top 400
    .venv/bin/python tools/law_pipeline.py --holdout select_mul

## 1. The intake mechanism

`tools/law_intake.py` is self-contained: a candidate registry of
`(name, thunk -> (model, example_input))` — 32 torch-native modules
plus 7 compound assemblies built inside the tool — and the four-step
boundary pipeline.  Every candidate lands in exactly one class:

| status | test | goes to |
|---|---|---|
| `ingested` | exported, all ops in `TorchSink().supported_ops`, lowered module `sink.verify`-passes vs the original | census + matchers + the firing/reach probe (≤ 120 nodes) |
| `census-only` | exported but ≥1 unbound op, or lowering/verify fails | census + matchers; the missing ops are the backlog |
| `rejected` | `torch.export` / `export_to_ir` raised | the rejection record, verbatim |

The census seam is a **side-file, not a code edit**: `--write` emits
`tools/intake_corpus.json` (terms via `term_to_data`, per-workload
metadata, the rejection backlog) and `tools/intake_tensors.pt`
(feeds/params via `torch.save`, `weights_only` load).  Two union
points read it:

* `law_shape_census.corpus()` appends `CorpusTerm("intake", …)` when
  the file exists — `run_census` gains `n_intake`; the census count
  went 103 → 145 terms with zero corpus code changes.
* `law_pipeline.run_pipeline` unions `load_cases()` into
  `real_terms` (matchers + census hash) and `probe_cases()` into the
  firing/reach probe (status `ingested`, feed present, ≤ 120 nodes —
  the corpus's own models top out near 80).

Absent the file both return `[]` — the baseline corpus is untouched,
so the artifact is regenerable, not a hidden dependency.

**Bench terms** (the task's open question): already counted.
`corpus()` unions `_bench_cases()` and bench contributes **45
bench-only op-tuples** of the 206 total — no hidden bench leg to
exploit.

## 2. What ingested

| class | count | the interesting ones |
|---|---|---|
| ingested | 17 | `nn.TransformerEncoderLayer` (35 nodes, `permute`/`unflatten`/`dropout`/`sdpa`), `nn.TransformerEncoder(d=3)` (105 — a stack at real depth), `SharedExpertMoE` (47 — `topk`/`eq`/`any`/`to`/`slice` dispatch spine), `ViTPatchBlock` (38 — `conv2d`→`flatten`→`transpose`→encoder), `Bottleneck` (`batch_norm` chains), `nn.Embedding`, `nn.PixelShuffle`, `nn.InstanceNorm2d`, `nn.Upsample(nearest)`, `nn.BatchNorm{1,2}d`, `nn.GroupNorm`, `nn.PReLU`, `nn.Mish`, `nn.Hardswish`, `LogSoftmaxHead` |
| census-only | 18 | pooling family (`max_pool2d`, `adaptive_avg_pool2d`, `avg_pool3d`), `im2col`/`col2im`, `conv3d`, `conv_transpose2d.input`, `feature_dropout`, `bilinear`, `cosine_similarity`, `upsample_bilinear2d.vec`, recurrents `gru.input`/`rnn_tanh_cell`, losses `cross_entropy_loss`/`mse_loss`/`nll_loss_nd`/`broadcast_tensors` — incl. `TinyCNN`, `LossHead`, `ManualNLL`, `nn.GRU`, `nn.RNNCell` |
| verify-failed | 3 | `nn.MultiheadAttention`, `nn.TransformerDecoderLayer`, `nn.LSTM` — §4 |
| rejected | 1 | `nn.CTCLoss` — `DynamicOutputShapeException: aten._ctc_loss.Tensor` |

`nn.Fold`'s earlier "rejection" was my input spec, not the bridge —
with the right `(N, C·k², L)` it exports `col2im` (unbound).  The
r2-deferred `nn.TransformerEncoderLayer`/`nn.MultiheadAttention`
were re-attempted properly: the encoder layer ingests cleanly; the
bare MHA exposed a real defect (§4).

## 3. The census diff

    before: 774 op nodes, 206 op-tuples, 431 shapes   (107 terms:
            66 bench + 41 models)
    after:  1167 op nodes, 290 op-tuples, 730 shapes  (145 terms:
            66 bench + 41 models + 38 intake)

    NEW op-tuples: 84   NEW corpus ops: 33
      permute unflatten squeeze dropout feature_dropout flatten
      contiguous sdpa(reshape,reshape,reshape) topk(softmax)
      eq(getitem, const) mul(slice, to) any(eq) to(any)
      split(·) log_softmax(linear) nll_loss_nd(log_softmax, ·)
      cross_entropy_loss(linear, ·) embedding(·, ·)
      + the unbound families: pooling, conv3d/conv_transpose,
        im2col/col2im, bilinear, cosine_similarity, recurrents,
        losses

The novelty is *structurally different* from workload-gen's: those
were arbitrary juxtapositions (`neg(neg(slice))`,
`pow(linear,linear)`); these are real program idioms — the
transformer's `reshape→transpose→permute` chains, the MoE dispatch
`topk → eq → any → to → mul(slice,·)` spine, the CNN `conv → bn →
relu → pool → flatten → linear` tail, the loss head `linear →
log_softmax → nll`.  Per-workload attribution is in
`intake_corpus.json` (`+tup`/`+shp` columns): `ViTPatchBlock` alone
contributes 23 new tuples, `TransformerEncoderLayer` 20,
`TransformerEncoder(d=3)` 21, `SharedExpertMoE` 9.

## 4. Pipeline results and the defect list

Pipeline delta (`lwg._run_pipeline` mirror, derived vocab, probe =
the corpus models + 17 probe-eligible intake):

| measure | baseline | +intake |
|---|---|---|
| terms / op-tuples | 107 / 206 | 145 / 290 |
| proposals | 54 | **58** |
| shippable | 0 | **0** |

* **4 new proposals, all census-mixed-view** — `add_transpose_l_{id,w}`
  (ViT's residual-over-transpose shape; fires 1 each) and
  `eq_getitem_l_{id,w}` (SharedExpertMoE's `topk_idx == i` mask; the
  first equality with real MoE-dispatch instances: match=4, fires=4
  each on `intake:SharedExpertMoE`).
* `eq_getitem_l_id` is the round's interesting row: **paid=1** (the
  rewrite dropped cost on a real case) and **false** (numeric oracle
  rejects) — the truth gate catching a paid lie, the same signature
  workload-gen reported for `add_split_l_id`.
* Intake terms give existing candidates their first / more firings:
  `mul_slice_l_{w,id}` 0 → 8 / 0 → 4 on `intake:SharedExpertMoE`
  (`w[:, i:i+1] * e(x)`), `reshape_transpose` 23 → 43 across the
  three transformer stacks, `mul_select_l_{id,w}` stay at 40 (all
  corpus).
* **`--holdout select_mul` PASS**: `census:mul_select` rank 1 of 58,
  SHIP — the enlarged corpus does not disturb the validated core.

**Verify-failures — the intake-specific finding.**  Three workloads
export and *bind* cleanly but the lowered module is wrong:

* `nn.MultiheadAttention` — `mat1 and mat2 shapes cannot be
  multiplied (16x16 and 4x48)`: the packed in-projection spell
  (unflatten/permute/reshape chain) mis-wires an operand through
  `export_to_ir`.  `nn.TransformerEncoderLayer` avoids it because it
  lowers through `sdpa`, not the decomposed MHA path.
* `nn.TransformerDecoderLayer` — `split_with_sizes expects
  split_sizes to sum exactly to 16 … got [16, 32]`: a `split` term
  receives the wrong operand/sizes (cross-attention's packed split).
  Poisons `TransformerDecoder` / `nn.Transformer` too.
* `nn.LSTM` — `mat1 and mat2 must have the same dtype, Float and
  Double`: the `lstm.input` binding (landed mid-session by the other
  agent's work) materializes fp32 weights under the fp64 harness.

None is fixable from `tools/`; each is a documented bridge gap.

**Binding-gap backlog (17 ops, exported, unbound):**
`adaptive_avg_pool2d`, `avg_pool3d`, `bilinear`,
`broadcast_tensors`, `col2im`, `conv3d`, `conv_transpose2d.input`,
`cosine_similarity`, `cross_entropy_loss`, `feature_dropout`,
`gru.input`, `im2col`, `max_pool2d`, `mse_loss`, `nll_loss_nd`,
`rnn_tanh_cell`, `upsample_bilinear2d.vec` — and the hard export
rejection `aten._ctc_loss` (dynamic output shape).  Priority by
family: the pooling trio (blocks real CNN graphs — `TinyCNN`,
`MaxPool2d`, `AdaptiveAvgPool2d` all wait on it), then the loss
family (`cross_entropy_loss` + `nll_loss_nd` block every
training-step graph), then `conv3d`/`conv_transpose2d.input`.

## 5. Caveats

* **The tree moved under the measurement — twice.**  Another agent
  landed `instance_norm`/`upsample_nearest2d`/`lstm.input` bindings,
  attr schemas, typing rules, four rsqrt-family laws, AND three new
  corpus models (`InstanceNorm`, `UpsampleNearest`, `LSTMSeq` — the
  same gaps this retro's backlog names) mid-session.  The §3 diff is
  the post-first-landing state (41-model baseline); the current
  corpus is already 66 bench + **44** models + 38 intake = 148
  terms, 211 base op-tuples — intake-unique is now **80 tuples /
  297 shapes**, the parallel additions having absorbed 4.
  `nn.LSTM`'s verify-fail is on the fresh binding.  The first pass
  (pre-landing) read 15 ingested / 21 census-only.
* **The side-file is uncommitted and regenerable** (`law_intake.py`
  rebuilds it in ~40 s of exports).  While it exists, every
  `corpus()`/`run_pipeline` consumer sees the enlarged corpus — by
  design; delete the file to restore the baseline.
* `intake_corpus.json` is ~19 MB: `term_to_data` serializes each
  term as a tree, so DAG-shared subterms duplicate.  Fine at 38
  terms; a sharing-aware codec is the honest next step before
  scaling intake to hundreds of workloads.
* The file round-trip is not shape-identical in 3 of 38 cases —
  `argN` positionals canonicalize to named attrs on reload
  (`instance_norm`), and the moving bridge renamed
  `upsample_nearest2d.vec` → `upsample_nearest2d`.  Both are the
  serialization faithfully recording the bridge state at write time;
  the delta counts above are computed on the loaded terms.
* `law_workload_gen`'s generators still read `bench+models` only —
  intake terms do not feed its resampler/mutator distribution (left
  unchanged deliberately; wiring it is a one-line union).
* Full pytest/coverage not run per the AGENTS.md resource bound;
  the change surface is `tools/` only.

## Gates

* `.venv/bin/ruff check tools/law_intake.py tools/law_shape_census.py tools/law_pipeline.py` — pass
* `.venv/bin/ruff format --check` (same) — pass
* `.venv/bin/python tools/radon_ratchet.py` — pass (2038 functions)
* `.venv/bin/python tools/law_intake.py` — runs; tables above (~7 min incl. two mirrored pipeline runs)
* `.venv/bin/python tools/law_pipeline.py --holdout select_mul` — PASS, rank 1 of 58 SHIP
* Not run: full pytest suite (task bound); `ty`/`vulture`/`lint-imports`/`bandit`/`semgrep` cover `packages` only, which this change does not touch.

Files: `tools/law_intake.py` (new), `tools/law_shape_census.py`
(+side-file union, `n_intake`), `tools/law_pipeline.py` (+intake in
`real_terms`/probe/corpus-hash/JSON/print), `tools/intake_corpus.json`
+ `tools/intake_tensors.pt` (generated artifacts, uncommitted).
