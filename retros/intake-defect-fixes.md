# Intake defect fixes — 3 verify-failed workloads now ingest, 17 → 20

`workload-intake.md` closed with a list no previous round could produce:
three real torch modules that **export and bind cleanly but verify
wrong** — `nn.MultiheadAttention`, `nn.TransformerDecoderLayer`,
`nn.LSTM`.  This retro fixes all three.  The headline result first:

    before: 38 exported → 17 ingested, 18 census-only, 3 verify-failed, 1 rejected
    after:  38 exported → 20 ingested, 18 census-only, 0 verify-failed, 1 rejected

All three turned out to be **one defect family**: `export_to_ir` was
dropping information torch.export elides or carries outside the
int/str/bool/list attr channels — an implicit aten `dim` default in
two cases, a `torch.dtype` kwarg in the third.  Nothing in the e-graph,
the laws, the typing layer, or the executor was wrong; the *boundary*
was emitting terms that did not spell the program the FX graph meant.

## 1. `nn.MultiheadAttention` — packed in-proj chunk, wrong axis

**Symptom.**  `RuntimeError: mat1 and mat2 shapes cannot be multiplied
(16x16 and 4x48)` at lower+run.  (The intake thunk feeds three distinct
q/k/v tensors, so torch.export takes the cross-attention path — not the
self-attention fast path where all three collapse to one input.)

**Root cause.**  The packed in-proj weight is `(3E, E) = (48, 16)`; the
export splits it into q/k/v thirds along **dim 0**:

    %chunk = aten.chunk.default(%p_in_proj_weight, 3)

`aten.chunk`'s `dim` parameter defaults to **0** — and aten elides it at
the default, so the FX node carries only `(w, 3)`.  `export_to_ir`
minted `chunk(w, chunks=3)` with no `dim` attr; the `chunk`/`split`
bindings and the typing rule read an absent dim as **the last axis**
(the minted-term convention — every law spells `dim=-1` on the packed
channel axis).  `torch.chunk((48,16), 3, dim=-1)` produces
(48,6)/(48,6)/(48,4) slabs; `linear` then sees a `4×48` mat2.

**Fix.**  `export_to_ir` records the aten default: when `ir_op in
("split", "chunk")` and no `dim` attr was emitted, `attrs["dim"] = 0`
(torch_bridge.py, the post-kwargs mint block).  The minted-term/law
convention (`absent dim → last axis`) is untouched — this is a
boundary-fidelity fix, not a generator-semantics change.  Audit:
`topk`/`sort`/`mode`/`glu` share aten's `dim=-1` default (consistent
already); `unbind`/`tensor_split` default `dim=0` matching the binding.
`split`/`chunk` was the lone mismatch — the same `dim=0` aten default
that `torch.split`/`chunk` document.

## 2. `nn.TransformerDecoderLayer` — same defect, `split_with_sizes` spelling

**Symptom.**  `RuntimeError: split_with_sizes expects split_sizes to sum
exactly to 16 (input tensor's size at dimension -1), but got
split_sizes=[16, 32]`.

**Root cause.**  The decoder's cross-attention packs q/kv differently:
`aten.split_with_sizes(w_3E_E, [E, 2E])` along **dim 0**, implicit at
the aten default → same dropped `dim`.  The binding then tried
`[16, 32]` on the last axis (extent 16).  Same one-line fix as §1 —
`split` and `chunk` share the injection (and `unsafe_split*`/`split.Tensor`
map to `split` upstream, so they inherit it).

## 3. `nn.LSTM` — `zeros` materialized fp32 under fp64

**Symptom.**  `RuntimeError: mat1 and mat2 must have the same dtype, but
got Float and Double`.

**Root cause.**  `nn.LSTM` builds `h0`/`c0` inside forward:
`torch.zeros(shape, dtype=input.dtype)`.  The export records it
faithfully as `aten.zeros(shape, dtype=torch.float64)` — a *kwarg*.
`export_to_ir`'s kwargs walk kept `int`/`float`/`bool`/`list`/`str`
values and **silently dropped `torch.dtype`** (with `torch.device` and
`torch.layout`).  The `zeros` term minted `dtype`-less; the binding's
`torch.zeros(shape)` produced fp32 — which `aten.lstm.input` refuses to
mix with the fp64 hidden state.  (This is on the *`zeros` op*, not the
`lstm.input` binding — the passthrough binding itself was always
dtype-clean.)

**Fix.**  Two halves:

* `export_to_ir` records `torch.dtype` kwargs as the short string
  (`"float64"` — dtype objects don't survive `term_to_data`).
  `device`/`layout`/`pin_memory` stay dropped — dispatch hints, not
  semantics (a cpu-pinned `device` attr would be *less* faithful to a
  lowered module moved to CUDA).
* The creator bindings honor a `dtype` attr via `_creator_dtype(kw)`
  (accepts the string, a live `torch.dtype`, or `None` → torch
  default): `zeros`, `ones`, `empty`, `randn`, `rand`, `full`,
  `zeros_like`, `ones_like`, `full_like`, `new_zeros`, `new_ones`,
  `new_empty`, `new_full`, `arange`.  `*_like`/`new_*` correctly
  inherit the operand dtype when no attr is present.

## 4. `verify_module` — tuple-returning modules couldn't verify at all

A latent fourth defect the first three were *masking*: `nn.MHA` returns
`(attn_out, attn_weights)` and `nn.LSTM` `(out, (h_n, c_n))`, while
`export_to_ir` keeps the exported graph's **first** output.  `verify_module`
subtracted `tuple - Tensor` → `TypeError` — so even a perfectly-lowered
multi-output module could never classify `ingested`.  `report.py` now
compares `ref[0]` when the reference returns a sequence — verifying
exactly what the IR models, no more and no less.

## Tests

`tests/test_intake_defects.py` (6 tests): all three modules through the
intake's own `export_to_ir` → `ir_to_torch_module` → `TorchSink.verify`
path fp64; the boundary contract pinned (`chunk`/`split` terms carry
`dim: 0` on the packed-projection exports, `zeros` carries
`dtype: "float64"`); an explicit `dim=1` chunk proves the injection only
fills the elided slot; `_creator_dtype` exercised in all three spellings
(string / `torch.dtype` object / absent + `*_like` inherit); and the
tuple-ref `verify` branch covered by MHA.

## Gates

ruff check/format ✓ · ty ✓ · vulture ✓ · lint-imports ✓ · bandit 0
findings ✓ · semgrep 0 findings ✓ · radon ratchet regenerated
(`export_to_ir` 66→69, `verify_module` 6→7 — both intentional; the
regeneration also reconciled a stale baseline that predated several
committed refactors) · scoped pytest: `test_intake_defects` (6) +
`test_torch_bridge_edges` + `test_corpus_models` + `test_attrs` +
`test_export` + `test_reports` + `test_batched_verify` + `test_typing*`
+ `test_om_lower_edges` + `test_cost_edge_cases` + `test_semantic_ops`
+ `test_law_serialize` + `test_adapters` — all green.  No full suite
(11 GB box).

## What the intake is actually worth

This is the third round where the honest answer to "did the law pipeline
ship anything" is *no* — and the second where the real yield came from
the harness, not the search.  The intake verification gate is the first
loop that finds **wrong programs**: every previous retro mined corpus
terms for *candidate equalities*; this one caught the boundary emitting
terms that were never the program.  The mechanism that paid off is
boring on purpose — lower the ingested term, run it fp64 against the
original, diff — and it is now a regression test, so the corpus keeps
paying it forward: any future export/lowering drift that breaks these
three shapes fails `test_intake_defects` before it fails the intake.
`tools/intake_corpus.json`/`intake_tensors.pt` (gitignored side files)
were regenerated in place: the persisted terms now carry `dim: 0` /
`dtype: "float64"` and all three records read `ingested`.
