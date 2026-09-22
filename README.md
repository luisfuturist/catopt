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
