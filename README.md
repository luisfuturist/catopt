# catopt

**Categorical optimization of neural-network computation graphs.**

> **Research question:** Can categorical semantics expose semantics-preserving
> transformations of neural-network computation graphs that are difficult or
> impossible for conventional tensor-level graph optimizers to discover
> efficiently?

---

## Yes. **That is the actual research question.**

And after checking the current state of the field, there is a particularly
interesting reason to pursue it: **the problem is demonstrably still open even
for conventional compiler techniques.**

PyTorch Inductor already has extensive graph rewriting, fusion, elimination,
and simplification passes. ([PyTorch Developer Mailing List][1]) Yet a 2026
PyTorch RFC proposes adding a **polyhedral optimization pass specifically
because existing pattern-matching/heuristic fusion can miss mathematically
valid and profitable fusion opportunities**; its initial SwiGLU/RMSNorm case
reports about a 1.4× speedup. ([PyTorch Developer Mailing List][2])

That's extremely relevant.

## The hypothesis

We could state it very cleanly:

> **Categorical semantics can expose semantics-preserving transformations of
> neural-network computation graphs that are difficult or impossible for
> conventional tensor-level graph optimizers to discover efficiently.**

Not:

> "Category theory makes CUDA faster."

But:

```text
                    same model
                        │
              ┌─────────┴─────────┐
              │                   │
       conventional IR      categorical IR
              │                   │
       existing optimizer    categorical optimizer
              │                   │
              └─────────┬─────────┘
                        ↓
                  same backend
                        ↓
                 Triton / CUDA
                        ↓
                    benchmark
```

That isolates the variable we care about.

### And we can make the claim much stronger

Suppose we find a transformation:

```text
G₁ → G₂
```

where:

1. `G₁` and `G₂` have the **same categorical semantics**.
2. We can formally establish that equivalence.
3. Existing TorchInductor does not produce `G₂` from `G₁`.
4. `G₂` generates measurably better GPU code.
5. The improvement survives comparison against current compiler optimizations.

Then we have something substantial.

Not merely:

> "Category theory is a nice way to represent neural networks."

But:

> **A semantic representation enabled a compiler optimization that a mature
> tensor compiler failed to discover.**

That's a legitimate compiler/PL/ML-systems result.

## There's an especially interesting connection

The 2025 LICS work on **Equivalence Hypergraphs** explicitly develops
categorical semantics for e-graphs and extends equality-saturation techniques
to monoidal categories. The paper frames rewrite optimization precisely as
sequences of semantics-preserving transformations where the ordering of rewrites
can affect the final execution cost. ([DOI][3])

And separate 2025 work shows that string-diagram rewriting can be implemented
as sound and complete hypergraph rewriting for richer categorical structures.
([UCL Discovery][4])

So we don't need to invent the mathematical machinery from scratch.

We can ask:

**Can this machinery actually buy us something on neural-network computation?**

That's the experiment.

## The killer experiment

I would actually avoid starting with an LLM.

Start with something like:

```text
SwiGLU / RMSNorm / attention blocks
```

because we already know current compilers have difficult optimization cases
there. The 2026 PyTorch RFC gives us an excellent baseline challenge.
([PyTorch Developer Mailing List][2])

Then:

### Phase 1 — Equivalence

Take a PyTorch computation graph:

```text
PyTorch
   ↓
torch.export
   ↓
ATen graph
   ↓
categorical/hypergraph IR
```

Define the categorical semantics.

### Phase 2 — Search

Build a rewrite system:

```text
         original graph
               │
       ┌───────┴───────┐
       │               │
     rewrite 1       rewrite 2
       │               │
     rewrite 3       rewrite 4
       │               │
       └───────┬───────┘
               ↓
       equivalent programs
               │
          cost model
               ↓
          best candidate
```

Potentially use **equality saturation** rather than greedily applying rewrites.

### Phase 3 — Lower

Don't write a CUDA compiler. That's unnecessary.

```text
optimized categorical IR
          ↓
       ATen / FX
          ↓
      TorchInductor
          ↓
       Triton/CUDA
```

Let NVIDIA/PyTorch solve the low-level engineering.

### Phase 4 — Compare

This is the critical table:

| Program  | Semantically equivalent? | TorchInductor result | Categorical result |
| -------- | -----------------------: | -------------------: | -----------------: |
| baseline |                        — |                 X ms |               X ms |
| case A   |                      yes |                 X ms |               Y ms |
| case B   |                      yes |                 X ms |               Y ms |
| case C   |                      yes |                 X ms |               Y ms |

And importantly:

**We need examples where TorchInductor already performs its normal optimization
pipeline.**

Otherwise we're just demonstrating that optimization beats no optimization.

---

## Status: first working prototype

`catopt/` now implements the pipeline above end to end. Run `python main.py`
(or `python main.py --large-batch 4096`) to reproduce everything below.

### What works

| Layer | Module | What it does |
| ----- | ------ | ------------ |
| IR | `catopt/ir.py` | Typed term algebra + symmetric-monoidal generator registry with declared laws |
| E-graph | `catopt/egraph.py` | Union-find, e-matching, equality saturation, cycle-safe extraction |
| Rules | `catopt/rules.py` | 18 rules: monoid/group laws, `silu`/`pow` bridges, matmul distributivity, naturality, associativity |
| Cost | `catopt/cost.py` | `count_cost` + shape-aware `flops_cost` (proper `2·M·N·K` matmul/linear) |
| Bridge | `catopt/torch_bridge.py` | `torch.export` → IR; IR → `IRModule`; ATen overload canonicalisation; **compile-time weight fusion** |
| Pipeline | `catopt/optimize.py` | 4-phase `optimize_model` with equivalence verification |

### Measured results (CPU, run on this machine)

| Program | Equivalent? | Inductor (ms) | CatOpt + Inductor (ms) | Speedup |
| ------- | ----------: | ------------: | ---------------------: | ------: |
| MatrixChain b=128  | yes (7e-09) | 0.048 | 0.030 | **1.60×** |
| MatrixChain b=4096 | yes (5e-09) | 0.240 | 0.039 | **6.22×** |
| **DeepParallel b=4096** | yes (2e-06) | 1.362 | 0.451 | **3.02×** |
| **ParallelLinear b=4096** | yes (3e-06) | 0.878 | 0.393 | **2.24×** |
| SwiGLU   | yes (0.0)   | — | — | 1.00× (cost unchanged, 802,816 FLOPs) |
| RMSNorm  | yes (2e-07) | — | — | 1.00× (cost unchanged, 2,128 FLOPs) |

Both paths go through `torch.compile`, so the comparison isolates the
representation: same backend, same model, same weights — only the graph
handed to TorchInductor differs.

`DeepParallel` is the strongest result: `(W1(x) + W2(x)) @ W3` collapses to a
single `linear` via **two composed rules** (`weight_factor_linear` then
`assoc_linear`) plus compile-time weight folding. No single pattern-matching
pass finds it — the distributive merge must fire *before* reassociation makes
folding legal. Hand-derived reference, measured: 0.406 ms (3.35×), so catopt
lands within 11% of what a human can achieve.

### The algebra is error-prone, which is the interesting part

Deriving the closed form for `DeepParallel` by hand, I got the transpose order
wrong **twice** — `F.linear(x, W) = x @ W.T` means the stacked composition
fuses to `B @ A`, not `A @ B`, and with distinct dims the naive order
`(W1+W2) @ W3` isn't even shape-valid (12×8 @ 16×12). catopt's
`assoc_linear` rule encodes `fused = B @ A` and its equivalence verifier
caught both of my errors (`catopt fused == closed form: True`,
`closed form == original: True`).

That is the concrete answer to *"what structural property did the categorical
representation expose?"*: composing **two** laws where the second is only
*applicable* after the first fires, plus a type-level (transpose-order)
constraint the cost model cannot see. Both got caught not by inspection but by
the e-graph's equivalence check.

### Honest reading of the numbers

* **The 6.3× FLOP figure is not a 6.3× speedup.** It counts the one-time
  `W1@(W2@W3)` precompute; at `batch=128` fixed overheads dominate and the
  wall-clock gain is 1.6×. At `batch=4096` the precompute amortises and the
  measured gain climbs to **6.22×**, approaching the runtime-only matmul-FLOP
  ratio of 10.2×. This is exactly the predicted behaviour, and it is why the
  demo prints both.
* **SwiGLU shows 1.00×.** With this cost model the categorical rules confirm
  equivalence but find no strictly cheaper form. Reporting that is the point:
  a rule system that always "wins" would be measuring its cost model, not the
  compiler.
* **RMSNorm reports 1.00×, and an earlier 1.98× claim was a cost-model bug.**
  A version of this table showed 1.98× because `cost._infer_op_shape` returned
  `shapes[0]` for elementwise ops instead of the broadcast shape, so
  `mul((B,T,1),(B,T,C))` was costed on `(B,T,1)`. Both candidate forms do
  identical work (verified by inspecting real output shapes). Broadcasting is
  now handled and the extractor correctly reports 1.000×.

### Known limitations of the current results

* **Every win so far collapses a *linear-only* DAG to one linear layer.**
  Three stacked linears (MatrixChain) and two parallel plus one (DeepParallel)
  both satisfy `no nonlinearity in between ⇒ equals a single linear`, and a
  domain expert would specify that in one line. So condition 4's premise —
  *difficult* for conventional optimizers — is still **not** demonstrated for
  the models themselves. What *is* demonstrated: Inductor genuinely does not
  find it (measured 3.02×–3.35× miss), the derivation is error-prone for
  humans, and two categorical laws must compose to produce it.
* **Weight folding is inference-only.** `x @ (W1@W2@W3)` is value-preserving
  during training, but folding destroys per-layer gradients, so it is only
  sound on frozen graphs. The cost model deliberately counts the one-time
  precompute as if it were runtime, so it only fully fuses above the
  break-even batch — at small batch it correctly keeps the outer layer
  separate (this is why tests use `batch=4096`).
* **The real target is a *nonlinear* case.** The hypothesis is only tested
  once a transformation is found where a nonlinearity sits in between, so the
  graph does **not** collapse trivially — e.g. `silu(x@W1) + x@W2` (MoE-style),
  or RMSNorm fused with a following projection where the per-row scale must
  commute through the matmul (`naturality_scalar`) *while* reassociating the
  reduction above it. Those rules exist; they do not yet compose into a win.
* **Two soundness bugs were found and fixed in this work**, which is the
  strongest evidence the instrument is trustworthy: (1) the matcher did not
  enforce repeated metavariables as "same e-class", so `x@W1 + y@W2` would
  have matched a pattern requiring `x@W1 + x@W2` and emitted a false proof;
  (2) `cost._infer_op_shape` took `shapes[0]` instead of the broadcast shape,
  which had fabricated a spurious 1.98× RMSNorm "win". Both are regression-tested.

### Reproduce

```bash
python main.py                     # associativity, SwiGLU/RMSNorm, naturality
python main.py --large-batch 4096  # measured large-batch timing
python -m pytest tests/ -q         # 61 tests
```

---

## And if we find even ONE convincing case...

Then the project becomes much more interesting.

Because then the follow-up question is:

> **What structural property did the categorical representation expose that the
> tensor representation obscured?**

That is where the theory becomes valuable.

Maybe it's:

* associativity/compositionality,
* symmetry,
* monoidal structure,
* sharing,
* naturality,
* algebraic identities,
* tensor/network contraction structure,
* equivalence classes of diagrams,
* or some combination.

And potentially we could eventually have:

```text
Neural network
      ↓
categorical semantics
      ↓
equivalence class
      ↓
search over mathematically equivalent programs
      ↓
cost model
      ↓
optimal executable representation
```

That is a **much deeper idea than "another graph optimizer."**

The exciting part is that **the existing compiler ecosystem gives us a brutally
strong baseline**. TorchInductor already does sophisticated graph optimization,
and current work is still adding new optimization machinery because there
remain missed opportunities. ([PyTorch Developer Mailing List][1])

So yes: **this is the question I'd build the entire project around.**

## References

[1]: https://dev-discuss.pytorch.org/t/inductor-passes/2742 "Inductor Passes - compiler - PyTorch Developer Mailing List"

[2]: https://dev-discuss.pytorch.org/t/rfc-polyhedral-optimization-pass-for-pytorch-inductor/3341 "RFC: Polyhedral Optimization Pass for PyTorch Inductor - compiler - PyTorch Developer Mailing List"

[3]: https://doi.org/10.1109/LICS65433.2025.00023 "Equivalence Hypergraphs: DPO Rewriting for Monoidal E-Graphs"

[4]: https://discovery.ucl.ac.uk/id/eprint/10211429 "Rewriting for Traced Monoidal Closed Categories - UCL Discovery"
