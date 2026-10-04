# Corpus expansion round 2 — 9 models, 40 new op-tuples, 17 new ops, 0 shippable (but 2 candidates found their first real targets)

`law-corpus-expansion.md` added eight architectures and proved the
loop: one expansion produced a shipped law (`softmax_fold`, enabled by
`ManualSoftmaxAttention`'s `exp/sum` shape).  `groupnorm-convnext-corpus.md`
closed the `group_norm` schema gap and sharpened the lesson: *a new op
in the census is not a new shape a proposer can see* — the generators
only speak pointwise/view/algebra.  This retro records round 2: nine
more real architectures aimed at the gaps the previous rounds named —
the index/selection ops (`topk`/`gather`/`argmax`/`one_hot`/`argmin`),
signal convs (`conv1d`/`pad`), the transcendental pair (`sin`/`cos`),
in-graph mask construction (`tril`/`ones`), the fused `glu` kernel and
the native `rms_norm` op — plus the measured yield.

The headline is again **honest and negative-but-informative**:

* The census grew: **753 op nodes (was 684), 203 op-tuples (was 163),
  422 shapes (was 374), 103 terms (was 94)** — 40 new op-tuples, 0
  removed, +48 shapes, **17 previously-absent ops**.
* The derived vocabulary absorbs the new ops by property, not by table:
  binary pointwise 4 → **6** (`maximum`, `eq`), unary pointwise 11 →
  **14** (`sin`, `cos`, `to`), views 8 → **9** (`getitem`).  `glu`
  correctly classifies as a *reduction* (it halves the last axis and
  computes) — the property tests working.
* The pipeline still reports **0 further shippable** (proposals 50 →
  **52**, firing on a real model 14 → **16**), but two existing
  candidates gained their first real targets from the new dispatch /
  quantizer shapes, and the top non-duplicate is now an *unproven*
  near-miss that fires 40× and pays (§3).
* **Held-out rediscovery still passes**: `--holdout select_mul`
  re-ranks `census:mul_select` **#1 of 52**, SHIP, identical evidence
  (fires 24, paid 5, drop 25.9 %, cert pass, enode 1.27×).
* One real bug found and fixed: `tools/law_vocab._concrete` treated
  `_shape_of`'s `-1` unknown-dim sentinel as concrete, so
  `TopKRouter`'s `expand(-1,-1,d)` index operand crashed the property
  probes.  Non-positive dims are now non-concrete.
* Three honest export-boundary gaps recorded: `nn.LSTM`
  (`lstm.input`), `nn.InstanceNorm` (`instance_norm`),
  `nn.Upsample` (`upsample_nearest2d.vec`) — export terms exist but no
  torch bindings, so they cannot be lowered or verified (§4).

Reproduce:

    .venv/bin/python tools/law_shape_census.py --top 400
    .venv/bin/python tools/law_vocab.py
    .venv/bin/python tools/law_pipeline.py
    .venv/bin/python tools/law_pipeline.py --holdout select_mul

## 1. What was added and why

Nine builders, same seam as every prior round: `catopt_torch.models`
classes registered in `tools/law_impact._model_cases()` — census,
vocab and pipeline pick them up with zero tool changes.  All nine
export cleanly through `torch.export` → `export_to_ir` and verify
fp64-exact after lowering.

| model | architecture | new corpus ops / tuples |
|---|---|---|
| `TopKRouter` | Switch/Mixtral-style top-k dispatch: `topk(router)` → `getitem` picks values, `gather` selects stacked expert outputs, softmax-renormalised weights combine | `topk`, `getitem`, `gather`; `getitem(topk)`, `softmax(getitem)`, `gather(stack, expand)`, `mul(unsqueeze, gather)` |
| `Wav2VecBlock` | wav2vec-style causal depthwise conv1d: `relu(pw(dw(pad(x))))`, `groups=channels` | `conv1d`, `pad`; `conv1d(pad,·)`, `conv1d(conv1d,·)`, `relu(conv1d)` |
| `SinusoidalEncoding` | Vaswani sinusoidal PE: `cat(sin, cos)` of `pos·w` added to tokens | `sin`, `cos`, `to`; `concat(sin, cos)`, `mul(to,·)`, `unsqueeze(arange)`, `to(unsqueeze)`, `add(·, concat)` |
| `TrilCausalAttention` | eager causal attention whose mask is BUILT in-graph: `tril(ones(T,T)) == 0` → `masked_fill` | `tril`, `ones`; `tril(ones)`, `eq(tril, const)`, `masked_fill(div, eq, const)` |
| `HardDispatch` | Switch top-1 hard routing: `argmax(router)` → `one_hot` → `mul(unsqueeze(oh), stack)` | `argmax`, `one_hot`; `argmax(linear)`, `one_hot(argmax)`, `to(one_hot)` |
| `CodebookQuantizer` | VQ-VAE encode step: `argmin(sum((x-cb)^2))` → `index_select` | `argmin`, `index_select`; `pow(sub, const)`, `sum(pow)`, `argmin(sum)`, `reshape(argmin)`, `index_select(·, reshape)`, `reshape(index_select)` |
| `MaxoutMLP` | Goodfellow maxout: `down(maximum(f1(x), f2(x)))` — two affine pieces over the SAME input | `maximum`; `maximum(linear, linear)`, `linear(maximum,·)` |
| `GluMLP` | fused GLU kernel: `down(glu(up(x)))` | `glu`; `glu(linear)`, `linear(glu,·)` |
| `NativeRmsNorm` | native `F.rms_norm` + projection — the folded spelling NormLinear writes by hand | `rms_norm`; `rms_norm(·,·)`, `linear(rms_norm,·)` |

Selection rule (unchanged): each builder had to be a plausible NN
block, export cleanly, and add op-tuples the census did not already
count.  Probed and **deferred** (all export cleanly, recorded for
future rounds): `torch.einsum` — exports but yields a degenerate
one-node term (no tuples); `nn.TransformerEncoderLayer` /
`nn.MultiheadAttention` — export cleanly and would add `permute`,
`unflatten`, `dropout`, but the decomposed graphs are large and their
novel ops deserve smaller hand-spelled builders; `roll`, `flip`,
`mish`, `softplus`, `leaky_relu`, `hardtanh`, `amax` — one-op unary or
view variants, low shape novelty.  Probed and **failed** (export-
boundary findings, §4): `nn.LSTM`, `nn.InstanceNorm`,
`nn.Upsample`.  `nn.RMSNorm` was expected to fail and did not —
`rms_norm` is fully supported and now in the corpus.

Coverage: `tests/test_corpus_models.py` exercises every builder the
same three ways as round 1 — forward under `eval()+no_grad`,
`export_to_ir` asserting the signature op family, and the lowered
`IRModule` verified fp64-exact.

## 2. The census diff

    before: 684 op nodes, 163 op-tuples, 374 shapes  (94 terms:
            63 bench + 31 models)
    after:  753 op nodes, 203 op-tuples, 422 shapes  (103 terms:
            63 bench + 40 models)

    NEW op-tuples: 40
      getitem(topk)            softmax(getitem)        topk(linear)
      gather(stack, expand)    mul(unsqueeze, gather)  unsqueeze(getitem)
      stack(linear,linear,…)   conv1d(pad, ·)          conv1d(conv1d, ·)
      relu(conv1d)             pad(·)                  concat(sin, cos)
      sin(mul)                 cos(mul)                mul(to, ·)
      to(unsqueeze)            to(one_hot)             unsqueeze(arange)
      add(·, concat)           tril(ones)              eq(tril, const)
      ones()                   masked_fill(div, eq, const)
      matmul(softmax, linear)  argmax(linear)          one_hot(argmax)
      argmin(sum)              reshape(argmin)         index_select(·, reshape)
      reshape(index_select)    pow(sub, const)         sum(pow)
      sub(unsqueeze, ·)        maximum(linear, linear) linear(maximum, ·)
      glu(linear)              linear(glu, ·)          rms_norm(·, ·)
      linear(rms_norm, ·)      unsqueeze(to)
    GONE: 0

    NEW corpus ops: 17  (42 → 59 census-visible incl. leaf markers;
    57 real ops in law_vocab's count)
      argmax argmin conv1d cos gather getitem glu index_select
      maximum one_hot ones pad rms_norm sin to topk tril

The dispatch shape `mul(unsqueeze(w), X)` now has **three** corpus
instances — `MoEMLP`'s `X=stack` (round 1), `TopKRouter`'s `X=gather`
and `HardDispatch`'s `X=stack` over a `one_hot` mask — which is why
the `mul_unsqueeze` proposals now fire on real models (§3).  As with
the conv additions, most new tuples are *deep chains* (selection,
index, conv1d) rather than `f(view, view)` naturality pairs; the two
proposals the census actually gains are the `sub(unsqueeze, ·)`
mixed-view pair from `CodebookQuantizer`.

## 3. Pipeline results on the enlarged corpus

| run | proposals | firing on a real model | shippable |
|---|---|---|---|
| `--vocab derived` | 52 | 16 | 0 |
| `--vocab derived --holdout select_mul` | 52 | 16 | **1** (the held-out winner itself) |

Per-candidate notes:

* `census:mul_select` — correctly `duplicate` of the shipped
  `select_mul`; with the holdout, rank 1 of 52, SHIP — **PASS**,
  identical evidence to the last three retros (fires 24, paid 5, drop
  25.9 %, cert pass, enode 1.27×).
* `mixed:mul_select_l_id` — the new **#1 non-duplicate**: fires 40
  (the most of any candidate), paid 3, drop 40 %, cert pass, enode
  1.64× — but **unproven (no oracle)**.  It is `mul(select(u,A), v) →
  mul(u, v)`: strip a `select` off one operand.  Value-true only under
  broadcasting-side-effects the pattern cannot see, so the truth gate
  is right to stay silent; the firing count is the corpus telling the
  proposer where the mass is.
* `mixed:mul_unsqueeze_l_{id,w}` — **the dispatch candidates**.  The
  three `mul(unsqueeze(w), X)` sites give them their first real
  firings (7 each, on `TopKRouter` + `HardDispatch`); `_id` pays once
  (14.3 % drop, cert pass, enode 1.11×) and stays `unproven`, `_w` is
  unproven and unpaid.  The shape the retro named as the grouped-GEMM
  fusion target is now firing — the verdicts just keep saying the
  naive equalities are not laws.
* `mixed:sub_unsqueeze_l_{id,w}` — **the only genuinely new
  proposals** (the +2 in the pool), generated by `CodebookQuantizer`'s
  `sub(unsqueeze, ·)` distance term.  Each fires once, unproven, no
  pay.
* `grammar:sub_to_add_dup` — `CodebookQuantizer`'s `sub` gives it a
  second real firing (2 total); duplicate anyway.
* `grammar:mul_distribute` — unchanged story: fires 24, matches 12,
  never lowers cost, blows the closure ~36–44×.
* **No candidate's LHS touches the 17 new ops** — `topk`, `gather`,
  `argmax`, `one_hot`, `argmin`, `index_select`, `tril`, `glu`,
  `rms_norm`, `conv1d`, `pad`, `ones` are all outside every
  generator's alphabet.  `getitem`/`eq`/`maximum` entered the derived
  vocab, but no `f(v, v)` tuple with those `f`s exists yet, so no
  naturality candidate consumes them.  Third round, same lesson: op
  presence ≠ reachable law shape.

## 4. Honest assessment

* **No new law shipped.**  The yield this round is corpus-level:
  17 ops, 40 tuples, three vocab-alphabet gains, and *two candidates
  that finally fire on real models* (`mul_unsqueeze` family — the
  exact dispatch shape round 1's §6 flagged as the lead).  Saying
  more would be overstating it.
* **A real bug fixed at the vocab seam**: `TopKRouter`'s
  `expand(-1,-1,d)` index operand leaks `-1` into `_shape_of`'s
  inferred shape; `law_vocab._concrete` accepted it as "all-int" and
  `torch.rand(-1,-1,16)` crashed the whole derivation.  Non-positive
  dims are now non-concrete — the probes skip what they cannot
  instantiate.
* **The corpus is still not the bottleneck for the naturality
  family** — 203 tuples and the generators still produce the same
  `select`/`unsqueeze`/`slice`/`reshape`/`transpose` families because
  no model yet gives `f(v, v)` a *new* pointwise `f` or view `v`.
  `maximum(linear, linear)` is the first `f(g, g)` with a new `f`;
  a `maximum(select, select)`-shaped model would complete the
  naturality pattern for it.
* **Export-boundary findings (this round's gaps)**:
  * `nn.LSTM` — `No torch binding for op 'lstm.input'` (the recurrent
    kernel family is outside the bridge entirely).
  * `nn.InstanceNorm` (1d and 2d) — exports a clean `instance_norm`
    term, but *no torch binding*, so it cannot lower or verify.  A
    norm-family gap distinct from the round-1 `group_norm` schema gap:
    the schema/binding pair is missing, not just an attr name.
  * `nn.Upsample` — `upsample_nearest2d.vec` likewise binds nothing.
  * Non-findings worth recording: `torch.topk`, `torch.gather`,
    `F.one_hot`, `torch.index_select`, `F.glu`, `F.rms_norm`,
    `nn.Conv1d`, `F.pad`, `torch.tril`, `torch.ones`,
    `torch.maximum`, `argmin`/`argmax`, `torch.sin`/`torch.cos`,
    `.to()` — all export *and* lower fp64-exact on first try.
* **MoE top-k routing does export** — the round's sketch "probably
  unsupported" was wrong once the dispatch is written honestly
  (`topk` → `getitem` → `gather`, not boolean-mask indexing).

## 5. Gates

* `.venv/bin/ruff check packages tools` — pass
* `.venv/bin/ruff format --check packages tools` — pass
* `.venv/bin/ty check` — pass (0 errors)
* `.venv/bin/vulture` — pass
* `tools/radon_ratchet.py` — pass (2028 functions; the nine builders
  stay under the rank-C threshold)
* `pytest tests/test_corpus_models.py` — **3 passed** (forward +
  export + lowered fp64 verify over all 19 corpus builders)
* Full suite not run on this box (resource constraint, per the
  AGENTS.md parallel-warning); every new class is exercised by all
  three corpus tests and the builders are branch-free except
  `TopKRouter`'s constructor, fully covered.

## 6. Where the next law might hide

1. **`rms_norm` unfold — the clearest recognizer lead.**  The corpus
   now has BOTH spellings of the same math: `NormLinear`'s manual
   `x * rsqrt(mean(x^2)) * w` and `NativeRmsNorm`'s `rms_norm` kernel
   op — exactly the `softmax_fold` situation before it shipped
   (manual softmax vs `softmax`).  A `recognize:rms_norm` pattern on
   the manual side, or an unfold on the kernel side, has real
   instances in four models (`RMSNorm`, `NormLinear`,
   `TransformerBlock`, `ParallelBlock`).
2. **`glu` fold** — `mul(chunk(u, i=0), sigmoid(chunk(u, i=1))) →
   glu(u)`: the kernel image (`glu`) is in the corpus; the manual
   spelling is not yet — a `GegluMLP`-with-`chunk` variant would
   complete the pair the way `ManualSoftmaxAttention` did.
3. **Index-family laws** — `argmin(sum((x-cb)^2)) → index_select`
   (nearest lookup), `topk+getitem+gather` (top-k dispatch),
   `argmax+one_hot → gather`: three real patterns, all outside the
   current generators — the next proposer extension, not the next
   model.
4. **The three binding gaps** (`instance_norm`, `lstm.input`,
   `upsample_nearest2d.vec`) — each is a small bridge addition like
   the `group_norm` schema; `instance_norm` is the highest-value one
   (a third real norm family).
5. **`maximum` / `eq` naturality pairs** — both are now derived
   binary pointwise; a corpus shape `maximum(select, select)` or
   `eq(select, select)` would auto-generate candidates on the next
   expansion.
