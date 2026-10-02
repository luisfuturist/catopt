"""The suite catalog the CLI drives.

Every benchmark is one ``SuiteSpec``.  The organizing axis is
``intent`` — *the question the suite answers* — not the layer it
happens to touch, so the catalog reads as a set of claims rather than a
pile of scripts.  ``question`` and ``expects`` are the machine-readable
form of the "expected verdicts" prose, and the docs (README tables,
dashboard grouping) are generated from them, so they cannot drift.

Orthogonal axes:

* ``tier``       — micro / block / model (how big a thing is measured).
* ``status``     — harnessed (exposes ``run_bench``) vs ad-hoc.
* ``mechanisms`` — a controlled vocabulary of the transforms involved.
* ``needs_cuda`` — the suite has a CUDA-only leg.
"""

# ruff: noqa: RUF001 -- math notation (times, minus, rho) in prose.

from __future__ import annotations

from dataclasses import dataclass, field

#: The organizing axis — the question a suite answers, in catalog order.
INTENTS = (
    "correctness",
    "search",
    "cost",
    "structure",
    "speedup",
    "e2e",
    "bounded",
    "integration",
)

#: Controlled vocabulary of mechanisms (for filtering / grouping).
MECHANISMS = (
    "laws",
    "egraph",
    "cost",
    "pairing",
    "autotune",
    "morphism",
    "scan",
    "carriers",
    "decode",
    "cuda-graph",
    "omd",
    "bounded",
    "weights",
    "checkpoint",
    "serving",
)

TIERS = ("micro", "block", "model")


@dataclass(frozen=True)
class SuiteSpec:
    """Metadata for one benchmark suite."""

    name: str
    intent: str
    title: str
    question: str
    expects: str
    tier: str = "block"  # micro | block | model
    status: str = "harnessed"  # harnessed | ad-hoc | retired
    mechanisms: tuple[str, ...] = ()
    needs_cuda: bool = False
    quick: dict = field(default_factory=dict)

    @property
    def module(self) -> str:
        """Dotted import path (the directory is the intent)."""
        return f"bench.suites.{self.intent}.{self.name}"

    @property
    def harnessed(self) -> bool:
        return self.status == "harnessed"


def _s(name, intent, title, question, expects, **kw) -> SuiteSpec:
    assert intent in INTENTS, intent
    for m in kw.get("mechanisms", ()):
        assert m in MECHANISMS, f"{name}: unknown mechanism {m!r}"
    assert kw.get("tier", "block") in TIERS, name
    return SuiteSpec(
        name=name,
        intent=intent,
        title=title,
        question=question,
        expects=expects,
        **kw,
    )


#: The catalog, grouped by intent.
SUITES: list[SuiteSpec] = [
    # -- correctness: does a rewrite fire, get picked, and verify? ------
    _s(
        "law_bench",
        "correctness",
        "Rewrite-law effect bench",
        "Does each registered rewrite law fire, get picked by "
        "extraction, and lower to a verified term?",
        "WIN — every registered law fires/picks/verifies; non-firing "
        "laws report honestly.",
        tier="micro",
        mechanisms=("laws",),
    ),
    _s(
        "laws_effect",
        "correctness",
        "Laws applied to model families",
        "Do the law families pay off at runtime on realistic model "
        "families?",
        "WIN on launch-bound families; honest negatives where "
        "Inductor's fused pointwise kernel wins on CPU.",
        mechanisms=("laws",),
    ),
    _s(
        "morphism_coverage",
        "correctness",
        "Morphism-law coverage on checkpoints",
        "Which morphism laws match and fire on real checkpoints?",
        "Coverage map — matches/fires per law, declines with reasons.",
        tier="model",
        mechanisms=("morphism", "checkpoint"),
    ),
    # -- search: what does the search cost relative to the space? -------
    _s(
        "search_efficiency",
        "search",
        "Search cost vs exponential space",
        "How expensive is saturation relative to the program space it "
        "represents?",
        "WIN — ~Catalan(k−1) programs in O(k³) live e-nodes; exact "
        "saturation fragments past k≈11 (honest limit).",
        tier="micro",
        mechanisms=("egraph",),
    ),
    # -- cost: does the model predict reality? --------------------------
    _s(
        "cost_fidelity",
        "cost",
        "Cost-model fidelity vs measured",
        "Does the pipeline cost model rank candidates like measured "
        "latency?",
        "High rank correlation (ρ) and pick accuracy — term-level cost "
        "tracks the backend.",
        tier="micro",
        mechanisms=("cost",),
    ),
    # -- structure: what structure do real weights carry? ---------------
    _s(
        "structure_census",
        "structure",
        "Weight-structure census",
        "How much catopt-exploitable structure do real trained weights "
        "carry?",
        "Exact mode: ~zero bitwise structure on dense LLMs, real on "
        "structured ones.",
        tier="model",
        status="ad-hoc",
        mechanisms=("weights",),
    ),
    _s(
        "bound_amplification",
        "structure",
        "Bound amplification on real activations",
        "How does a weight-space error bound propagate to the output?",
        "Measured output rel exceeds the certified bound by ~2–8× "
        "(honest).",
        status="ad-hoc",
        mechanisms=("bounded", "weights"),
    ),
    # -- speedup: measured head-to-head wins ----------------------------
    _s(
        "reassoc_scale",
        "speedup",
        "Weight-chain reassociation vs Inductor",
        "Can the e-graph find a form Inductor's post-grad graph cannot "
        "express?",
        "WIN — 8.9–16.1× vs Inductor on k-deep weight chains.",
        mechanisms=("egraph", "pairing"),
    ),
    _s(
        "real_win_hunt",
        "speedup",
        "Structured-block win hunt",
        "Which structured block topologies admit an autotuned win?",
        "WIN — ~1.1–10× vs Inductor on exploitable topologies.",
        mechanisms=("autotune", "pairing"),
    ),
    _s(
        "real_linear_attn",
        "speedup",
        "Linear-attention scan lift",
        "Does the affine-monoid scan lift pay on real linear-attention "
        "blocks?",
        "WIN — ≈2.9× vs eager (retnet T=128); honest negatives where "
        "Inductor's pointwise fusion wins on CPU.",
        mechanisms=("scan", "carriers"),
    ),
    _s(
        "morphism_e2e",
        "speedup",
        "Morphism-window composition",
        "Do morphism windows convert term-flops into wall time?",
        "WIN — 12.4–12.5× measured wall on 4-block chains at "
        "GEMM-bound sizes.",
        mechanisms=("morphism",),
    ),
    _s(
        "decode_scan_bench",
        "speedup",
        "Carrier + CUDA-graph decode",
        "Does the chunked scan carrier beat the best non-carrier "
        "decode schedule?",
        "WIN on launch-bound devices — 1.65–3.4× vs Inductor; CUDA for "
        "the graph leg.",
        mechanisms=("decode", "carriers", "cuda-graph"),
        needs_cuda=True,
    ),
    _s(
        "decode_bench",
        "speedup",
        "Launch-bound decode sweep",
        "Does fewer GEMM launches pay off where launch overhead "
        "dominates?",
        "NEGATIVE — the launch-bound hypothesis is falsified (4–15% "
        "loss at B=1, T≤64).",
        status="ad-hoc",
        mechanisms=("decode",),
    ),
    _s(
        "killer_demo",
        "speedup",
        "Compositional pairing demo",
        "Does per-model lowering autotune pick the measured-fastest "
        "variant?",
        "WIN — the reported pick is the measured winner, never a static "
        "choice.",
        mechanisms=("autotune", "pairing"),
    ),
    _s(
        "bench_omd2",
        "speedup",
        "omd executor on an attention stack",
        "Does the cross-carrier omd lift survive a transformer-shaped "
        "attention?",
        "Exploratory — fires on the mqa case; not a certified path.",
        status="ad-hoc",
        mechanisms=("omd", "carriers"),
    ),
    # -- e2e: whole models, end to end ----------------------------------
    _s(
        "model_bench",
        "e2e",
        "Whole-model optimize + verify",
        "Do whole multi-block models beat Inductor under the autotuned "
        "lowering?",
        "Latency/peak-memory/compile per model, verified; wins where "
        "structure exists.",
        tier="model",
        mechanisms=("autotune", "pairing"),
    ),
    _s(
        "e2e_model",
        "e2e",
        "MiniGPT end-to-end",
        "Does composition (pairing + fold) beat plain Inductor "
        "end-to-end?",
        "PARITY — ~1.05× over Inductor, verified fp64-exact.",
        tier="model",
        mechanisms=("pairing",),
    ),
    _s(
        "e2e_models2",
        "e2e",
        "Model-family end-to-end",
        "Does composition hold across architecture families?",
        "PARITY — in-repo replicas (minilm/vit/conv/llama/moe), "
        "verified per cell.",
        tier="model",
        mechanisms=("pairing",),
    ),
    _s(
        "e2e_llm",
        "e2e",
        "LLM block end-to-end",
        "Does composition help prefill + KV-cache decode on a "
        "llama-scale model?",
        "Measured c+i/ind band; honest per-cell verdicts.",
        tier="model",
        mechanisms=("pairing",),
    ),
    _s(
        "stories15m_bench",
        "e2e",
        "stories15M whole-model pipeline",
        "Does the whole-model pipeline transform and verify a real "
        "checkpoint?",
        "PARITY — all blocks transform+verify, but the mechanism "
        "doesn't pay at 15M/110M.",
        tier="model",
        status="ad-hoc",
        mechanisms=("checkpoint", "pairing"),
    ),
    _s(
        "bench_e2e",
        "e2e",
        "Quick whole-model smoke",
        "Does a MiniGPT optimize and verify at all?",
        "Smoke — sanity only; use the rigorous sweeps for numbers.",
        tier="model",
        status="ad-hoc",
        mechanisms=("pairing",),
    ),
    # -- bounded: certified-approximation rewrites ----------------------
    _s(
        "bounded_e2e",
        "bounded",
        "Certified bounded rewrites",
        "Do error-budget rewrites buy wall-time on a real checkpoint?",
        "NEGATIVE on stories15M — 0 bounded members accepted at every "
        "budget (honest).",
        tier="model",
        mechanisms=("bounded", "checkpoint"),
    ),
    _s(
        "structured_models",
        "bounded",
        "Structured-model families",
        "Do LoRA / pruned / low-rank families admit bounded rewrites?",
        "WIN where structure exists — params shrink / speedups, "
        "verified.",
        mechanisms=("bounded",),
    ),
    # -- integration: catopt against an external stack ------------------
    _s(
        "vllm_compare",
        "integration",
        "catopt vs vLLM and through it",
        "Can vLLM serve a catopt-optimized model token-for-token?",
        "WIN — token-for-token agreement (~1.0); the wins are "
        "complementary, not competing.",
        tier="model",
        mechanisms=("serving", "checkpoint"),
        needs_cuda=True,
    ),
]

_BY_NAME = {s.name: s for s in SUITES}


def get(name: str) -> SuiteSpec:
    """Look up a suite by name (accepts ``bench.x`` / ``x.py``)."""
    key = name.removesuffix(".py").rsplit(".", 1)[-1]
    if key not in _BY_NAME:
        raise KeyError(
            f"unknown suite {name!r}; known: {', '.join(sorted(_BY_NAME))}"
        )
    return _BY_NAME[key]


def by_intent() -> dict[str, list[SuiteSpec]]:
    """Suites grouped by intent, in catalog order."""
    out: dict[str, list[SuiteSpec]] = {}
    for s in SUITES:
        out.setdefault(s.intent, []).append(s)
    return out


def by_mechanism(mech: str) -> list[SuiteSpec]:
    """Suites whose mechanisms include ``mech``."""
    return [s for s in SUITES if mech in s.mechanisms]


def harnessed() -> list[SuiteSpec]:
    """Only the suites that expose ``run_bench``."""
    return [s for s in SUITES if s.harnessed]
