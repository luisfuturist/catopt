# Retro — de-hardcoding the discovery content tables (round 2)

## What the pass did

Round 1 (plan 0018, `dehardcode.md`) gave the *op* vocabulary one
registry (`catopt_core.opmeta`); plan 0019 declared the *ops* as data
(`catopt_core.opdata`).  This pass applies the same discipline to the
second vocabulary the engine carries: the *content tables* — the
banks, libraries, ledgers and weights a learned policy could
plausibly choose — that were still buried as module-level literals in
`catopt_discovery/` and `catopt_core/meta.py`.

Two new pure-data modules hold them (the `opmeta` idiom: the table
lives in exactly one place, consumers bind the **same object**, and a
consistency test pins the relation):

* **`catopt_core/lawdata.py`** — the law-side vocabulary the core
  search machinery reads: the coherence classification
  (`COHERENT_RULE_NAMES`), the canonicalizer's monoid vocabulary
  (`AC_IDENTITY`, `ASSOC_ONLY`), and the bounded instantiation bank
  (`INSTANTIATE_LEAF_SHAPES`, `INSTANTIATE_ATTR_POOL`).
* **`catopt_discovery/lawdata.py`** — every discovery-side content
  table: oracle enumeration banks, verifier instantiation defaults,
  the gap-synthesis shape pools, the grammar alphabet, training
  families, the coherence-probe neighbourhoods, the intake ledger,
  the meta-game arm inventory and prior/score weights, the arena
  reward composition, and — as *term specs* — the proposal corpus
  (`SEED_TERMS`, `GRAMMAR_SCHEMAS`, `SHAPE_SCHEMAS`) and the impact
  corpus (`CANDIDATE_LAWS`, `SYNTHETIC_CASES`).

Both modules are pure data — no logic, no callables — so adding a
bank entry, schema row, arm or weight is a data row, never a Python
edit: the shape a model writes a new entry in.

## Term specs — how the corpora went declarative

The five term corpora (`seed_terms`, `schema_candidates`,
`shape_proposal.schemas`, `impact.new_laws`, `impact.synthetic_cases`)
were `Op.make(...)` call lists — executable code carrying content.
They are now tuples of *specs* in the language
`object_synthesis.term_from_spec` reads (``str`` metavariables,
``int``/``float`` constants, ``(op, *specs[, attrs])`` nodes),
extended with two leaf forms — ``("var", name, dims)`` /
``("param", name, dims)`` — for the typed concrete leaves the seed
corpus and the grammar candidates carry.  One resolver,
`proposal._spec_term`, mints terms; `shape_proposal` and `impact`
share it.  **Every resolved term is byte-identical** to the literal
it replaced — verified by replaying the old builders out of
`git show HEAD:` and comparing `op_repr` on all 16 + 23 + 21 + 11 +
11 entries (and the inputs/feed ordering of every synthetic case).

## What moved, what stayed — the inventory

| Table | Was | Now | Moved? | Why |
|---|---|---|---|---|
| `oracle._ATTR_KINDS` / `_ATTR_KIND_OVERRIDES` | literal dicts | `lawdata.ATTR_KINDS` / `ATTR_KIND_OVERRIDES` | yes | the attr→value-kind typing a sweep (or a learned enumerator) enumerates — content |
| `oracle._VIEWED_SHAPES` / `_CHAIN_VIEWED_SHAPES` | literal tuples | `lawdata.VIEWED_SHAPES` / `CHAIN_VIEWED_SHAPES` | yes | the shape bank a metavariable-under-a-view enumerates — content |
| `oracle._FREE_SENTINELS` / `_FALLBACK_AXES` / `_FALLBACK_SHAPES` | literals | `lawdata.FREE_SENTINELS` / `FALLBACK_AXES` / `FALLBACK_SHAPES` | yes | sentinel + unknown-shape bank entries — content |
| `oracle._CONST_DOMAIN` | literal dict | `lawdata.CONST_DOMAIN` | yes | per-op literal-constant domain (the domain-gapped rescues of `cap-policy.md`) — content |
| `oracle` tuple sources (`_tuple_sources` body) | three `Op.make` calls | `lawdata.TUPLE_SOURCES` + `TUPLE_SOURCE_SHAPE` | yes | the `(op, attrs)` spec of each tuple-producing op the getitem corner mints — content; the instantiation stays code |
| `oracle._attr_options` per-op branches | functions of operand shape | unchanged | no | shape-*dependent* domains are algorithms, not tables; they read the moved static banks |
| `verifier._LEAF_SHAPES` / `_ATTR_DEFAULTS` (+ the `S`/`F` scalar corner) | literal dicts + two `elif` branches | `lawdata.INSTANCE_LEAF_SHAPES` / `INSTANCE_ATTR_DEFAULTS` / `INSTANCE_SCALAR_MVARS` | yes | metavariable→shape/default for `generic_instance` — content; the instantiate/check logic stays |
| `gap_gen._SHAPE_POOL` / `_BASE_SHAPES` | literal lists | `lawdata.SHAPE_POOL` / `BASE_SHAPES` | yes | the instantiation draw pool — content |
| `grammar._BINARY_OPS` / `_UNARY_OPS` / `_LITERALS` | literal tuples | `lawdata.GRAMMAR_*` | yes | the search's alphabet — the set a schema-space player chooses from |
| `families.FAMILIES` / `_LINEAR_TRAIN` / `_LINEAR_HELD` | literals | `lawdata.FAMILIES` / `LINEAR_*_SHAPES` | yes | the training/held-out split — the corpus a generalization claim is measured on |
| `coherence._CLUSTER`, `coherence2._CORPUS_MODELS` | literals | `lawdata.COHERENCE_CLUSTER` / `COHERENCE_CORPUS_MODELS` | yes | hand-picked neighbourhoods — content |
| `intake._PURPOSE_BUILT` | literal frozenset | `lawdata.PURPOSE_BUILT` | yes | the corpus-circularity ledger — already data, now in the one home with the rest |
| `intake.candidates()` registry (the corpus spelling list) | `Workload` records with `nn.Module` thunks | unchanged | mixed | the `Workload` *rows* are already records; their `factory`/`kind` hooks carry kernel bodies — code, never data. The pure-metadata half (`_PURPOSE_BUILT`) moved |
| `impact._model_cases` / `zoo` model registries | `nn.Module` thunks | unchanged | no | model constructors are kernel bodies — code, never data |
| `meta_game.ENUMERATION_ORDER` / `CORPUS_ARMS` / `_ARM_NAMES` | literal tuples | `lawdata.GENERATOR_ORDER` / `CORPUS_ARMS` | yes | the generator-arm inventory the guide schedules — enumerable content |
| `meta_game._MV_NAMES` / `_CONSTS` | literals | `lawdata.MV_NAMES` / `CONST_LEAVES` | yes | the construction player's leaf vocabulary — content |
| `meta_game._PRIOR_*` (7 weights) | literal floats | `lawdata.PRIOR_WEIGHTS` dict | yes | arm/child priors — exactly what a learned policy could choose |
| `meta_game` score composition `1 + 0.2·fires + 2·paid + 10·drop` | literal formula | `lawdata.REFEREE_SCORE` dict | yes | the referee's scoring weights — content (the composition stays code) |
| `arena._STAGE_CREDIT` / `_USABLE_BONUS` / `_FIRE_CREDIT` / `_PAY_CREDIT` | literal floats | `lawdata.ARENA_REWARD` dict | yes | the reward composition — content (the gauntlet math stays) |
| `proposal.seed_terms` corpus | `Op.make` list | `lawdata.SEED_TERMS` specs | yes | the curated seed programs — content |
| `proposal.schema_candidates` grammar | `Op.make` table | `lawdata.GRAMMAR_SCHEMAS` spec rows | yes | the algebraic-identity schema library — content |
| `shape_proposal.schemas()` library | `Op.make` table | `lawdata.SHAPE_SCHEMAS` spec rows | yes | the shape-aware schema library — content |
| `impact.new_laws()` candidates | `R(...)` call list | `lawdata.CANDIDATE_LAWS` spec rows | yes | the 11 proposed laws — content (the `Rewrite` mint + tags stay code) |
| `impact.synthetic_cases()` corpus | `Op.make` list | `lawdata.SYNTHETIC_CASES` `(name, spec, var names)` rows | yes | the per-law witness terms — content (the random `feed` stays code) |
| `meta.COHERENT_RULE_NAMES` | literal frozenset | `catopt_core.lawdata.COHERENT_RULE_NAMES` | yes | the coherence classification — content |
| `meta._AC_IDENTITY` / `_ASSOC_ONLY` | literals | `catopt_core.lawdata.AC_IDENTITY` / `ASSOC_ONLY` | yes | the canonicalizer's monoid vocabulary — content |
| `meta._LEAF_SHAPES` / `_ATTR_POOL` | literal tuples | `catopt_core.lawdata.INSTANTIATE_*` | yes | the candidate-instantiation bank — content |
| `arena.ACTIONS` | registry dict | unchanged | already data | already the action registry — not duplicated |
| `arena.FixedRule` playbook | injected `Action` iterable | unchanged | already data | the playbook is caller-authored (`moves` param) — `FixedRule` is a replay mechanism over injected data; the ingest-first + guard-conditional ordering is policy logic (code) |
| `workload_gen` census pools / mutation probabilities | derived statistics + `0.25`/`0.6` draws | unchanged | no | census pools are *measured* data (recomputed from the corpus), not authored content; the mutation mix is a strategy parameter, not a vocabulary — flagged, not moved (see below) |
| `pipeline._POINTWISE` / `_VIEW_OPS` | sorted tuples | unchanged | already data | opmeta projections since round 1 (`POINTWISE_BINARY_OPS` / `GENERATOR_VIEW_OPS`) |
| `pipeline._RECOGNIZERS` | hook table | unchanged | no | recognizers map signatures to *functions* — a hook, not a record |
| `evidence` column/SQL vocabulary | literal lists | unchanged | no | store schema — the database contract, not player content |
| budgets (`_MAX_*`, `_RTOL`, `_TOL`, iteration/node caps) | literals | unchanged | no | resource knobs, not learnable content |

## Consistency test — `tests/test_lawdata.py` (27 tests)

Same shape as `test_opmeta.py`'s projection checks:

* every consumer name **is** the data object (`is`, not `==`) — the
  anti-drift guarantee across all 14 consumers;
* `COHERENT_RULE_NAMES ⊆` the real law universe (shipped rules +
  scan rules + the carrier modules' rules, enumerated through the
  same `meta.module_rules` seam the engine uses — never a hand list);
* `AC_IDENTITY ⊆ COMMUTATIVE_OPS` (the registry projection);
* `ATTR_KIND_OVERRIDES` references only declared attrs/kinds; every
  tuple source is a registered op with valid attrs;
* `GENERATOR_ORDER` names the five pipeline sources + `build`,
  disjoint from `CORPUS_ARMS`; `PRIOR_WEIGHTS` / `REFEREE_SCORE` /
  `ARENA_REWARD` carry exactly the keys the consumers read;
* each term-spec table resolves to the expected count/labels/family
  and the right leaf types (concrete `Var`/`Param` for seeds and the
  grammar; metavars for the schema/candidate libraries).

## Gates

`ruff check`, `ruff format --check`, `ty check`, `vulture` and the
radon ratchet (no baseline change — each converted function is a
`map` over a data table plus a row resolver under the threshold).
Scoped pytest on every touched consumer's test files — `test_lawdata`
(27 new), `test_opmeta*`, `test_meta*`, `test_synthesis_seeds`,
`test_intake_defects`, `test_law_view_oracle`, `test_discovery_oracle`,
`test_discovery_verifier`, `test_discovery_proposal`, `test_discovery_impact`,
`test_discovery_families`, `test_discovery_coherence*`,
`test_discovery_meta_game`, `test_discovery_arena`, `test_discovery_intake`,
`test_discovery_climb_a1/a2/b`, `test_discovery_experiments` (gap/grammar
arcs), `test_discovery_gauntlet`, `test_discovery_zoo`,
`test_discovery_emit`, `test_discovery_pipeline`,
`test_discovery_synthesis`, `test_discovery_autocond`,
`test_discovery_shippable_audit`, `test_law_order`, `test_cond_laws`,
`test_law_pipeline_typed` — all green (1300+ tests).

## Honest boundaries

* **The attr-domain generators stayed code.** `oracle._attr_options` /
  `gap_gen._attr_candidates` / `gap_gen._op_attr_candidates` compute
  domains *from operand shape* — they are enumeration algorithms that
  read the moved static banks, not tables.  Moving them would mean a
  hook registry, exactly the indirection this pass exists to remove.
* **`workload_gen`'s mutation mix stayed.** The `0.25`/`0.6` strategy
  draws are player-visible policy numbers, but they sit inside one
  RNG-consuming expression; the census pools they feed are *derived*
  statistics.  Moving them is a policy-extraction question (which
  object owns a learned mutator), not a table move.
* **`meta_game`'s `build` arm inventory is data; its legality is code.**
  The action grammar (tree-filling, side budgets, `actions()`) is the
  referee's machinery — algorithms.
* **Budgets stayed.** `meta._MAX_INSTANTIATIONS`, the iteration/node
  caps, `_RTOL`/`_TOL` are resource knobs, not learnable content.
* **`object_synthesis.py` was excluded** — a sibling effort owns it;
  this pass only *consumes* its `term_from_spec` DSL.
* **Laws stay laws.** `CANDIDATE_LAWS` moved the *records*; the
  `Rewrite` mint, tags and bodies are the engine's replayable objects.
