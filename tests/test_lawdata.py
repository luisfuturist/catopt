"""lawdata consistency — the data home and its consumers stay pinned.

``catopt_core.lawdata`` / ``catopt_discovery.lawdata`` are the single
home for the law-side / discovery content tables; consumers bind the
*same object* under their own names (the ``opmeta`` projection idiom).
These tests pin that relation — an unbound copy would silently drift
the two halves apart — and the shape the consumers expect of every
table they read.
"""

from __future__ import annotations

import catopt_core.lawdata as core_data
import catopt_discovery.lawdata as data
from catopt_core import meta
from catopt_core.attrs import ATTR_SCHEMA
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_core.opmeta import COMMUTATIVE_OPS, REGISTRY


class TestCoreLawdata:
    """Core content tables — same-object bindings in ``meta``."""

    def test_coherent_names_is_the_table(self):
        assert meta.COHERENT_RULE_NAMES is core_data.COHERENT_RULE_NAMES

    def test_coherent_names_are_real_rules(self):
        """Every classified name resolves in the law universe."""
        from catopt_core.laws import SCAN_DIAG_LAWS, SCAN_LAWS

        universe = {
            r.name for r in [*ALL_RULES, *SCAN_LAWS, *SCAN_DIAG_LAWS]
        }
        # the carrier rules (om_assoc pair, …) live in the carrier
        # modules — enumerate them through the same seam the engine
        # registers, never a hand list.
        from catopt_core.ops import _CARRIER_MODULES

        for mod_name in _CARRIER_MODULES:
            mod = __import__(mod_name, fromlist=["*"])
            universe |= {r.name for r in meta.module_rules(mod)}
        assert core_data.COHERENT_RULE_NAMES <= universe

    def test_canonicalize_vocab_is_bound(self):
        assert meta._AC_IDENTITY is core_data.AC_IDENTITY
        assert meta._ASSOC_ONLY is core_data.ASSOC_ONLY

    def test_ac_identity_ops_are_commutative(self):
        assert set(core_data.AC_IDENTITY) <= set(COMMUTATIVE_OPS)

    def test_instantiation_bank_is_bound(self):
        assert meta._LEAF_SHAPES is core_data.INSTANTIATE_LEAF_SHAPES
        assert meta._ATTR_POOL is core_data.INSTANTIATE_ATTR_POOL
        assert all(
            isinstance(s, tuple)
            for s in core_data.INSTANTIATE_LEAF_SHAPES
        )


class TestOracleBanks:
    """oracle.py binds every enumeration bank to the same object."""

    def test_shape_banks(self):
        from catopt_discovery import oracle

        assert oracle._VIEWED_SHAPES is data.VIEWED_SHAPES
        assert oracle._CHAIN_VIEWED_SHAPES is data.CHAIN_VIEWED_SHAPES
        assert oracle._FREE_SENTINELS is data.FREE_SENTINELS
        assert oracle._FALLBACK_AXES is data.FALLBACK_AXES
        assert oracle._FALLBACK_SHAPES is data.FALLBACK_SHAPES
        assert oracle._CONST_DOMAIN is data.CONST_DOMAIN

    def test_attr_kind_tables(self):
        from catopt_discovery import oracle

        assert oracle._ATTR_KINDS is data.ATTR_KINDS
        assert oracle._ATTR_KIND_OVERRIDES is data.ATTR_KIND_OVERRIDES

    def test_attr_kind_overrides_reference_known_kinds(self):
        kinds = set(data.ATTR_KINDS.values())
        for (_op, attr), kind in data.ATTR_KIND_OVERRIDES.items():
            assert attr in data.ATTR_KINDS, attr
            assert kind in kinds, kind

    def test_viewed_shapes_are_positive_int_tuples(self):
        for bank in (data.VIEWED_SHAPES, data.CHAIN_VIEWED_SHAPES):
            for sh in bank:
                assert isinstance(sh, tuple)
                assert all(isinstance(d, int) and d > 0 for d in sh)

    def test_tuple_sources_resolve(self):
        """Every tuple source is a registered op with valid attrs."""
        from catopt_discovery import oracle

        srcs = oracle._tuple_sources("U")
        assert [t.op for t in srcs] == [
            op for op, _ in data.TUPLE_SOURCES
        ]
        for t in srcs:
            assert isinstance(t, Op)
            assert t.op in REGISTRY
            (w,) = t.args
            assert isinstance(w, Var)
            assert w.typ.shape == data.TUPLE_SOURCE_SHAPE


class TestVerifierDefaults:
    """verifier.py binds the instantiation defaults."""

    def test_tables_bound(self):
        from catopt_discovery import verifier

        assert verifier._LEAF_SHAPES is data.INSTANCE_LEAF_SHAPES
        assert verifier._ATTR_DEFAULTS is data.INSTANCE_ATTR_DEFAULTS
        assert verifier._SCALAR_MVARS is data.INSTANCE_SCALAR_MVARS


class TestPoolTables:
    """The simple bank tables are bound under the consumers' names."""

    def test_gap_pools(self):
        from catopt_discovery import gap_gen

        assert gap_gen._SHAPE_POOL is data.SHAPE_POOL
        assert gap_gen._BASE_SHAPES is data.BASE_SHAPES

    def test_grammar_alphabet(self):
        from catopt_discovery import grammar

        assert grammar._BINARY_OPS is data.GRAMMAR_BINARY_OPS
        assert grammar._UNARY_OPS is data.GRAMMAR_UNARY_OPS
        assert grammar._LITERALS is data.GRAMMAR_LITERALS

    def test_families(self):
        from catopt_discovery import families

        assert families.FAMILIES is data.FAMILIES
        assert families._LINEAR_TRAIN is data.LINEAR_TRAIN_SHAPES
        assert families._LINEAR_HELD is data.LINEAR_HELD_SHAPES

    def test_coherence_lists(self):
        from catopt_discovery import coherence, coherence2

        assert coherence._CLUSTER is data.COHERENCE_CLUSTER
        assert coherence2._CORPUS_MODELS is data.COHERENCE_CORPUS_MODELS

    def test_intake_ledger(self):
        from catopt_discovery import intake

        assert intake._PURPOSE_BUILT is data.PURPOSE_BUILT
        assert isinstance(data.PURPOSE_BUILT, frozenset)


class TestMetaGameData:
    """meta_game.py reads the arm inventory and weight tables."""

    def test_arm_inventory(self):
        from catopt_discovery import meta_game

        assert meta_game.ENUMERATION_ORDER is data.GENERATOR_ORDER
        assert meta_game.CORPUS_ARMS is data.CORPUS_ARMS
        assert meta_game._MV_NAMES is data.MV_NAMES
        assert meta_game._CONSTS is data.CONST_LEAVES

    def test_arm_inventory_covers_the_pools(self):
        """The generator order names the pipeline's five sources."""
        pipeline_arms = set(data.GENERATOR_ORDER) - {"build"}
        assert pipeline_arms == {
            "census-naturality",
            "census-mixed-view",
            "pattern-recognition",
            "shape-aware",
            "algebraic-grammar",
        }
        assert not set(data.GENERATOR_ORDER) & set(data.CORPUS_ARMS)

    def test_prior_weights_bound(self):
        from catopt_discovery import meta_game

        assert meta_game._PRIOR_CHILD == data.PRIOR_WEIGHTS["child"]
        assert meta_game._PRIOR_REUSE == data.PRIOR_WEIGHTS["reuse"]
        assert (
            meta_game._PRIOR_BOUND_MV == data.PRIOR_WEIGHTS["bound_mv"]
        )
        assert (
            meta_game._PRIOR_FRESH_MV == data.PRIOR_WEIGHTS["fresh_mv"]
        )
        assert meta_game._PRIOR_COMMUTE == data.PRIOR_WEIGHTS["commute"]
        assert (
            meta_game._PRIOR_SWAP_NEST
            == data.PRIOR_WEIGHTS["swap_nest"]
        )
        assert (
            meta_game._PRIOR_DEPTH_DECAY
            == data.PRIOR_WEIGHTS["depth_decay"]
        )
        # the table carries exactly the weights the prior reads
        assert set(data.PRIOR_WEIGHTS) == {
            "child",
            "reuse",
            "bound_mv",
            "fresh_mv",
            "commute",
            "swap_nest",
            "depth_decay",
        }

    def test_score_and_reward_tables(self):
        assert set(data.REFEREE_SCORE) == {
            "base",
            "fire",
            "fire_cap",
            "paid",
            "rel_drop",
        }
        assert set(data.ARENA_REWARD) == {
            "stage",
            "usable",
            "fire",
            "paid",
        }


class TestTermSpecCorpus:
    """The term-spec tables resolve through ``proposal._spec_term``."""

    def test_seed_terms_resolve(self):
        from catopt_discovery import proposal

        seeds = proposal.seed_terms()
        assert len(seeds) == len(data.SEED_TERMS)
        assert all(isinstance(t, Op) for t in seeds)

    def test_seed_term_shapes(self):
        """A spec leaf mints the typed Var the hand corpus carried."""
        from catopt_discovery import proposal

        seed = proposal.seed_terms()[0]
        # matmul(a:(3,5), matmul(b:(5,2), c:(2,3)))
        assert seed.op == "matmul"
        a = seed.args[0]
        assert isinstance(a, Var) and a.typ.shape == (3, 5)

    def test_grammar_schemas_resolve(self):
        from catopt_discovery import proposal

        cands = proposal.schema_candidates()
        assert len(cands) == len(data.GRAMMAR_SCHEMAS)
        assert [c.label for c in cands] == [
            row[0] for row in data.GRAMMAR_SCHEMAS
        ]
        assert all(c.strategy == "schema" for c in cands)
        # the grammar's leaves are concrete Vars, not metavars
        mf = next(c for c in cands if c.label == "mul_factor")
        leaves = {t for t in _walk(mf.lhs) if isinstance(t, Var)}
        assert {v.typ.shape for v in leaves} == {(4, 4)}

    def test_shape_schemas_resolve(self):
        from catopt_discovery import shape_proposal

        sch = shape_proposal.schemas()
        assert len(sch) == len(data.SHAPE_SCHEMAS)
        assert [s.name for s in sch] == [
            row[0] for row in data.SHAPE_SCHEMAS
        ]
        assert {s.family for s in sch} == {
            row[4] for row in data.SHAPE_SCHEMAS
        }

    def test_candidate_laws_resolve(self):
        from catopt_discovery import impact

        laws = impact.new_laws()
        assert [r.name for r in laws] == [
            row[0] for row in data.CANDIDATE_LAWS
        ]
        assert [r.law for r in laws] == [
            row[3] for row in data.CANDIDATE_LAWS
        ]

    def test_synthetic_cases_resolve(self):
        from catopt_discovery import impact

        cases = impact.synthetic_cases()
        assert [c.name for c in cases] == [
            row[0] for row in data.SYNTHETIC_CASES
        ]
        for case, (_, _, names) in zip(
            cases, data.SYNTHETIC_CASES, strict=True
        ):
            assert [v.name for v in case.inputs] == list(names)
            assert len(case.feed) == len(names)

    def test_spec_resolver(self):
        """The leaf forms mint typed leaves; dicts stay attr maps."""
        from catopt_discovery import proposal

        spec = (
            "mul",
            ("var", "x", (4, 4)),
            ("param", "W", (4, 4)),
        )
        t = proposal._spec_term(spec)
        assert t.op == "mul"
        a, b = t.args
        assert isinstance(a, Var) and a.typ == TensorType((4, 4))
        assert isinstance(b, Param) and b.typ == TensorType((4, 4))
        # metavar / const leaves
        t2 = proposal._spec_term(("mul", "A", 0))
        assert t2.args[0] == "A"
        assert t2.args[1] == Const(0)
        # attr dict passes through un-resolved
        t3 = proposal._spec_term(
            ("select", "A", {"dim": "D", "index": "I"})
        )
        assert t3.attrs == {"dim": "D", "index": "I"}


def _walk(term):
    """Yield every node of *term* (leaves included)."""
    yield term
    if isinstance(term, Op):
        for a in term.args:
            yield from _walk(a)
