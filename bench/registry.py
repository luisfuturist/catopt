"""The suite catalog the CLI drives.

One ``SuiteSpec`` per benchmark: where it lives, which category it
belongs to, whether it is harnessed (exposes ``run_bench``) or still an
ad-hoc script, and whether it needs CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CATEGORIES = ("core", "algebra", "models", "infra")


@dataclass(frozen=True)
class SuiteSpec:
    """Metadata for one benchmark suite."""

    name: str
    module: str  # dotted import path
    category: str
    title: str
    status: str = "harnessed"  # harnessed | ad-hoc | retired
    tags: tuple[str, ...] = ()
    needs_cuda: bool = False
    quick: dict = field(default_factory=dict)

    @property
    def harnessed(self) -> bool:
        return self.status == "harnessed"


def _s(name, category, title, **kw) -> SuiteSpec:
    return SuiteSpec(
        name=name,
        module=f"bench.suites.{category}.{name}",
        category=category,
        title=title,
        **kw,
    )


#: The catalog.  Harnessed suites expose ``run_bench(args) -> Report``.
SUITES: list[SuiteSpec] = [
    # -- core: the search / cost machinery ------------------------------
    _s("law_bench", "core", "Rewrite-law effect bench", tags=("laws",)),
    _s(
        "search_efficiency",
        "core",
        "Search cost vs exponential space",
        tags=("egraph",),
    ),
    _s(
        "cost_fidelity",
        "core",
        "Cost-model fidelity vs measured",
        tags=("cost",),
    ),
    _s(
        "laws_effect",
        "core",
        "Laws applied to model families",
        tags=("laws",),
    ),
    _s(
        "structure_census",
        "core",
        "Weight-structure census on real checkpoints",
        status="ad-hoc",
        tags=("weights",),
    ),
    _s(
        "bound_amplification",
        "core",
        "Bounded-error bound amplification",
        status="ad-hoc",
        tags=("bounded",),
    ),
    # -- algebra: the head-to-head wins ---------------------------------
    _s(
        "reassoc_scale",
        "algebra",
        "Weight-chain reassociation vs Inductor",
        tags=("headline",),
    ),
    _s(
        "real_linear_attn",
        "algebra",
        "Linear-attention scan lift vs eager/Inductor",
        tags=("headline", "carriers"),
    ),
    _s(
        "decode_scan_bench",
        "algebra",
        "Carrier + CUDA-graph decode",
        tags=("decode",),
        needs_cuda=True,
    ),
    _s(
        "real_win_hunt",
        "algebra",
        "Structured-block win hunt",
        tags=("headline",),
    ),
    _s(
        "morphism_e2e",
        "algebra",
        "Morphism-window composition",
        tags=("morphism",),
    ),
    _s(
        "killer_demo",
        "algebra",
        "Compositional pairing demo",
        tags=("demo",),
    ),
    # -- models: end-to-end on real checkpoints -------------------------
    _s(
        "model_bench",
        "models",
        "Whole-model optimize + verify",
        tags=("e2e",),
    ),
    _s("e2e_model", "models", "MiniGPT end-to-end", tags=("e2e",)),
    _s(
        "e2e_models2",
        "models",
        "Model-family end-to-end",
        tags=("e2e",),
    ),
    _s("e2e_llm", "models", "LLM block end-to-end", tags=("e2e",)),
    _s(
        "bounded_e2e",
        "models",
        "Certified bounded rewrites on a checkpoint",
        tags=("bounded",),
    ),
    _s(
        "structured_models",
        "models",
        "Structured-model families",
        tags=("bounded",),
    ),
    _s(
        "vllm_compare",
        "models",
        "catopt vs vLLM and through it",
        tags=("serving",),
        needs_cuda=True,
    ),
    _s(
        "morphism_coverage",
        "models",
        "Morphism-law coverage on real checkpoints",
        tags=("morphism", "checkpoint"),
    ),
    _s(
        "bench_e2e",
        "models",
        "Quick whole-model smoke",
        status="ad-hoc",
        tags=("smoke",),
    ),
    _s(
        "bench_omd2",
        "models",
        "omd executor on an attention stack",
        status="ad-hoc",
        tags=("omd",),
    ),
    _s(
        "decode_bench",
        "models",
        "Launch-bound decode sweep",
        status="ad-hoc",
        tags=("decode",),
    ),
    _s(
        "stories15m_bench",
        "models",
        "stories15M whole-model pipeline",
        status="ad-hoc",
        tags=("checkpoint",),
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


def by_category() -> dict[str, list[SuiteSpec]]:
    """Suites grouped by category, in catalog order."""
    out: dict[str, list[SuiteSpec]] = {}
    for s in SUITES:
        out.setdefault(s.category, []).append(s)
    return out


def harnessed() -> list[SuiteSpec]:
    """Only the suites that expose ``run_bench``."""
    return [s for s in SUITES if s.harnessed]
