---
theme: lfds
layout: cover
title: CatOpt
info: |
  Automatically discovering and verifying tensor program optimizations.
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide-left
mdc: true
colorSchema: dark
---

<div v-motion :initial="{ opacity: 0, y: 12 }" :enter="{ opacity: 1, y: 0, transition: { duration: 450 } }">

<Kicker>catopt</Kicker>

</div>

<h1 v-motion :initial="{ opacity: 0, y: 18 }" :enter="{ opacity: 1, y: 0, transition: { delay: 140, duration: 620 } }">
CatOpt
</h1>

<div v-motion :initial="{ opacity: 0 }" :enter="{ opacity: 1, transition: { delay: 420, duration: 500 } }">

Automatically discovering and verifying tensor program optimizations.

<div style="margin-top:1.1rem; max-width:44rem; font-size:1rem; color:var(--muted)">
What happens if we search the space of equivalent programs, instead of encoding optimization heuristics by hand?
</div>

<div class="footnote" style="margin-top:1.4rem">Luis Emidio · 2026</div>

</div>

<!--
Framing: a technical demo of one idea. No collaboration ask, no company.
The whole deck answers: is this a genuinely interesting technique, and do
you understand what I built?
-->

---
layout: default
title: The observation
---

# The observation

Compilers optimize what they know how to rewrite.

<div class="reach">
<div class="reach-space" aria-hidden="true"></div>
<div class="footnote">every program equivalent to the input</div>
<div class="reach-inner">
<div class="mono">reachable by a fixed rewrite set</div>
<div class="footnote">op-level patterns only</div>
</div>
</div>

<v-clicks>

- rewrite spaces are manually designed
- opportunities can span multiple algebraic operations
- some transformations require changing the structure of the computation
- discovering these transformations is hard to automate

</v-clicks>

<Callout v-click style="max-width:88%">

Can we **search** for better programs, rather than enumerate the transformations ourselves?

</Callout>

<!--
Modern compilers have powerful passes, but the reachable set is bounded by the
rules someone wrote. If no rule expresses a transformation, the compiler cannot
reach it — no matter how good the search is. That's the observation.
-->

---
layout: default
title: The core idea
---

# The core idea: search over equivalence classes

<div class="equiv">
<div class="equiv-label">same semantics</div>
<div class="equiv-stem" aria-hidden="true"></div>
<div class="equiv-row">
<div class="equiv-node">program A</div>
<div class="equiv-node cheap">program B</div>
<div class="equiv-node">program C</div>
</div>
<div class="equiv-label" style="margin-top:1.1rem">↓ extract the cheapest</div>
</div>

<div class="footnote" style="text-align:center; margin-top:1.6rem; font-size:0.98rem">
CatOpt searches the equivalence class of a program and extracts a low-cost representative.
</div>

<!--
This is the conceptual heart. Equivalence is established by equational laws;
the class is enumerated in an e-graph; a cost model picks the representative.
The surprising part is how much falls out of a small set of generic laws.
-->

---
layout: default
title: What I built
---

# What I built

<div class="cols">
<div>

<PipelineV :stages="['Tensor program', 'Semantic / categorical representation', 'E-graph saturation', 'Equivalent programs', 'Cost-guided extraction', 'Executable program', 'Certificate replay / verification']" />

</div>
<div>

<Kicker>the unusual pieces</Kicker>

- programs are lifted into **monoid objects** — the same law then reaches structures no op-level pattern expresses
- the engine is **backend-agnostic** and torch-free; a backend just implements the `Sink` port
- every delivered program carries a **replayable certificate**, re-checked on real terms

</div>
</div>

<!--
Keep this minimal — the machinery isn't the point yet. The three things worth
flagging: the categorical representation, the backend-agnostic engine, and the
fact that results are proven, not spot-checked. Go deeper only if asked.
-->

---
layout: default
title: A transformation CatOpt discovered
---

# A transformation CatOpt discovered

<div class="xf">
<div class="xf-row">
<div class="xf-tag">before</div>
<div class="xf-expr">(Q · Kᵀ) · V</div>
<div class="xf-note">T×T intermediate · O(T²d)</div>
</div>
<div class="xf-mid">↓ CatOpt</div>
<div class="xf-row after">
<div class="xf-tag">after</div>
<div class="xf-expr">Q · (Kᵀ · V)</div>
<div class="xf-note">d×d intermediate · O(Td²)</div>
</div>
</div>

<Callout style="margin-top:1.3rem; max-width:90%">

**8.0×** at T=2048 — and this isn't a rule I wrote for this example. It emerges from the equivalence search: associativity, plus a shape-aware cost model.

</Callout>

<!--
The strongest slide. It's the same expression, re-associated — but the
intermediate changes from T×T to d×d, so it's a different complexity class,
not a tuning. The novelty isn't the algebra; it's that the optimizer found and
justified it without a hand-written "reassociate attention" rule.
-->

---
layout: default
title: What happens in practice?
---

# What happens in practice?

| Workload | vs baseline | vs Inductor |
|---|---|---|
| reassociation `(QKᵀ)V → Q(KᵀV)`, T=2048 | **8.0×** | — |
| weights-first fold, k=16 | — | **15.9×** |
| streaming attention, 65k cache | **260×** | — |
| projection pairing (QKV, gate·up) | — | **1.24×** |
| stories15M / stories110M (whole model) | — | **~1.0×** |
| launch-bound decode (B=1, T≤64) | — | **0.85–0.96×** |

<div class="cols" style="gap:2rem; margin-top:1.1rem">
<div>

<Kicker>what works</Kicker>

CatOpt finds transformations existing compilers don't naturally reach.

</div>
<div>

<Kicker>what doesn't yet</Kicker>

The cost model and backend interaction aren't good enough to consistently turn discoveries into end-to-end wins.

</div>
</div>

<div class="footnote" style="margin-top:0.7rem">
RTX 2050 · baseline = eager or the alternative implementation · "—" = not measured
</div>

<!--
Honest results. Targeted transformations win big; whole real models are parity;
launch-bound cells lose. The mechanism is real (fewer kernels, fewer FLOPs) but
at these dimensions it doesn't consistently pay. Label both halves explicitly.
-->

---
layout: default
title: What I learned
---

# What I learned

<div class="display" style="font-size:2.5rem;font-weight:500;line-height:1.15">
The interesting problem isn't search — <span class="muted">search works</span>.
</div>

<div style="margin-top:0.9rem">
The hard part is knowing what is <em>actually</em> fast. A mathematically superior program can lose to:
</div>

<div style="margin-top:0.9rem">
<Tag>fusion</Tag>
<Tag>kernel launch overhead</Tag>
<Tag>memory traffic</Tag>
<Tag>backend lowering</Tag>
<Tag>hardware-specific effects</Tag>
</div>

<Callout style="margin-top:1.2rem; max-width:92%">

So the project moved from *"can I discover equivalent programs?"* to **"can I discover equivalent programs that are <span v-mark.underline>actually faster</span>?"**

</Callout>

<!--
The research slide. Search wasn't the bottleneck — the cost model was. That
reframing is the honest contribution: the question changed from discovery to
discovery-plus-realization. Everything above is evidence for this slide.
-->

---
layout: end
title: Where this could go
---

<Kicker>where this could go</Kicker>

<div class="footnote" style="margin-top:0.9rem">
the bigger idea — optimization as program-space search
</div>

<h1 style="font-size:2.5rem;margin-top:0.6em">
I'd like to understand where this approach is actually useful.
</h1>

<div style="margin-top:1.2rem">
<Tag>neural-network blocks</Tag>
<Tag>compiler optimization</Tag>
<Tag>hardware-specific codegen</Tag>
<Tag>architecture search</Tag>
<Tag>discovering optimization strategies</Tag>
<Tag>verified transformation</Tag>
</div>

<div class="footnote" style="margin-top:1.6rem">
github.com/luisfuturist/catopt · luisfuturist
</div>

<!--
No company, no prescribed application. The directions are offered as a prompt,
not a plan. Ending on "where is this useful" hands the evaluation to the
audience — that's where their specific problem enters the conversation.
-->

---
layout: default
title: How the search works (backup)
hideInToc: true
backup: true
---

# How the search works <Tag>backup</Tag>

An **e-graph** is a set of e-classes — equivalence classes of terms — whose e-nodes are function symbols applied to e-classes. A rewrite is a *union*; congruence closure keeps it consistent; matching re-fires until saturation.

<div class="egraph">
<div class="eg-class">
<div class="eg-id">e-class · one term</div>
<div class="eg-nodes">
<span class="eg-node">(a · b) · c</span>
<span class="eg-node same">a · (b · c)</span>
</div>
</div>
<div class="eg-link" aria-hidden="true" />
<div class="eg-class sub">
<div class="eg-id">sub-term class</div>
<div class="eg-nodes">
<span class="eg-node">b · c</span>
</div>
</div>
</div>

<div class="footnote" style="text-align:center">
associativity unions the two parenthesizations into one e-class — every later rewrite sees both
</div>

<div class="cols" style="margin-top:1.1rem">
<div>

<Kicker>the saturation loop</Kicker>

- match every law against every e-node
- union the two sides
- rebuild + congruence-close
- re-match until no new unions

</div>
<div>

<Kicker>extraction</Kicker>

- a fixed point over the e-graph with an **additive** cost function
- additive is required — `min`-over-lowerings is non-additive and corrupts the local decomposition
- only forms the backend can lower are priced

</div>
</div>

<!--
The answer to "how are you representing this?". An e-graph stores a *set* of
equivalent terms per class, so rewrites compose: after union, later laws match
against both sides. Saturation is the fixpoint; extraction is a DP over the
e-graph with an additive cost. The additive constraint is real — it's why
min-over-lowerings can only report, not select.
-->

---
layout: default
title: The categorical representation (backup)
hideInToc: true
backup: true
---

# The categorical representation <Tag>backup</Tag>

Programs are lifted into **carriers** — monoid objects whose axioms *are* the rewrite laws. The law set is the monoid's axioms, not a catalogue of patterns.

| carrier | monoid | reaches |
|---|---|---|
| affine `aff(A,b)` | matmul / add | balanced scan (Blelloch): depth `2T → 2·log₂T` |
| online-softmax `om(m,l,a)` | softmax combine | FlashAttention's combine — derived, not encoded |
| product `⟨f₁,…,f_k⟩` | `(×fᵢ) ∘ Δ` | one GEMM feeding *k* shared-input projections |
| `trace^U` | traced monoidal | closed-form resolvents `P + Q(I−S)⁻¹R` |

Cross-carrier laws compose them — readouts push through scan evaluation, scans fold inside softmax elements, attention over scanned values stays one recurrence.

<Callout style="margin-top:1.1rem; max-width:94%">

The lift is **non-local**: carrier-law saturation is combinatorially explosive at long horizons, so the carrier member is *constructed* from the recognized spine rather than searched for.

</Callout>

<!--
This is where a category-theory audience should be pointed. The contribution
isn't new math — it's that a small set of generic laws (associativity,
homomorphism, the traced monoidal axioms, the product law ⟨f₁,…,f_k⟩ = (×fᵢ)∘Δ)
reaches programs no op-level pattern expresses, and the search finds them
automatically. FlashAttention's combine, the Blelloch scan, fused QKV, and
closed-form resolvents all fall out of the same machinery.
-->

---
layout: default
title: Cost model & verification (backup)
hideInToc: true
backup: true
---

# Cost model & verification <Tag>backup</Tag>

<div class="cols" style="gap:2.2rem">
<div>

<Kicker>cost model</Kicker>

- pluggable — `calibrate()` + `roofline_cost_for`; nothing in the engine names CUDA
- extraction is bounded by `sink.supported_ops` — members using ops the backend can't lower price at `+∞`
- executor-aware pricing (`dispatch_us`, `leaf_eval_us`) fixed an *inverted* model on scan blocks (ρ ≈ −0.3 → correctly ordered)
- residual gap: structural cost can't predict realized kernel fusion

</div>
<div>

<Kicker>verification</Kicker>

- every program carries an ordered, replayable derivation `original → optimized`
- re-checked on **real terms** — derivational equivalence, not numerical spot-checks
- it has caught a false proof, a fabricated **1.98×** "win", and a well-typed but wrong program (diff 9.83)

</div>
</div>

<div class="derive">
<span class="derive-step">original</span>
<span class="derive-sep">→</span>
<span class="derive-step">reassociate</span>
<span class="derive-sep">→</span>
<span class="derive-step">fold params</span>
<span class="derive-sep">→</span>
<span class="derive-step done">optimized ✓</span>
</div>

<!--
Two honest caveats, both worth volunteering. (1) The cost model is the weak
link — it was rank-inverted on scan blocks until executor-aware pricing; the
residual is that structural cost can't see realized fusion. (2) Verification
is the strong link — it's derivational, and it has caught real bugs including a
fabricated speedup. That asymmetry is the point: discovery is cheap, pricing is
hard, and proof is what keeps it honest.
-->
