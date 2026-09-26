# ruff: noqa: RUF002
"""Persistent cache for synthesized rules — catopt.rulecache.

``synthesize_rules`` is the expensive phase of meta-optimization, but
its output is pure data.  ``RuleCache`` serializes the structural part
(name, law, lhs/rhs patterns, provenance) plus each guarded rule's
*re-expression spec* — the ``(pat1, pat2)`` substitution maps recorded
on ``rule.guard_pats`` — and rebuilds the composite ``check``/``derive``
at load time by re-running ``meta._compose_guards`` against the parent
rules.  No callable is ever serialized.

Covered here:

* round-trip — a synthesized set stores, loads, fires identically, and
  verifies fp64 on a concrete instance;
* guarded round-trip — a composite check still vetoes/accepts after
  reload, and a ``derive`` placeholder is genuinely *recomputed* on a
  new instance (rank-3 concat dim = 2, not the seed's baked 1);
* key misses — changed ruleset / seeds / params produce different keys;
* speedup — a repeated synthesis reloads far faster than re-deriving.
"""

import time

import pytest
import torch

from catopt import meta
from catopt import om as OM
from catopt import rules as R
from catopt.egraph import EGraph, Rewrite
from catopt.ir import Const, Op, TensorType, Var, op_repr
from catopt.rulecache import (
    RuleCache,
    cache_key,
    synthesize_rules_cached,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _sdpa_fold(name):
    return next(r for r in R.SDPA_FOLD_RULES if r.name == name)


def _om_lemma_seed():
    """matmul(softmax(cat(s1,s2,-1), -1), cat(v1,v2,-2))."""
    s1 = Var("s1", _T(5, 3))
    s2 = Var("s2", _T(5, 4))
    v1 = Var("v1", _T(3, 6))
    v2 = Var("v2", _T(4, 6))
    term = Op.make(
        "matmul",
        Op.make("softmax", Op.make("concat", s1, s2, dim=-1), arg1=-1),
        Op.make("concat", v1, v2, dim=-2),
    )
    return term, (s1, s2, v1, v2)


def _left_scaled_attention_seed():
    """matmul(softmax(mul(0.5, q@kᵀ) + m, -1), v)."""
    q = Var("q", _T(5, 4))
    k = Var("k", _T(7, 4))
    v = Var("v", _T(7, 6))
    m = Var("m", _T(5, 7))
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, arg1=-2, arg2=-1)
    )
    term = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make("add", Op.make("mul", Const(0.5), scores), m),
            arg1=-1,
        ),
        v,
    )
    return term, (q, k, v, m)


def _env(leaves):
    torch.manual_seed(0)
    return {
        lf: torch.randn(*lf.typ.shape, dtype=torch.float64)
        for lf in leaves
    }


# ---------------------------------------------------------------------------
#  (a) round-trip: store → load → identical firing, fp64-verified
# ---------------------------------------------------------------------------


def test_roundtrip_rules_fire_identically_and_verify_fp64(tmp_path):
    seed, leaves = _om_lemma_seed()
    rules = [OM.OM_LIFT, OM.OM_SPLIT]
    derived = meta.synthesize_rules(rules, [seed], fuel=4000)
    assert derived

    cache = RuleCache(tmp_path)
    key = cache_key(rules, [seed], fuel=4000)
    cache.store(key, derived)
    loaded = cache.load(key, rules)
    assert loaded is not None and len(loaded) == len(derived)

    for orig, rel in zip(derived, loaded, strict=True):
        assert rel.name == orig.name and rel.law == orig.law
        assert meta._alpha_key(rel.lhs, rel.rhs) == meta._alpha_key(
            orig.lhs, orig.rhs
        )
        assert (
            meta.provenance(rel)
            == meta.provenance(orig)
            == ("om_lift", "om_split")
        )
        assert (rel.check is None) == (orig.check is None)
        assert (rel.derive is None) == (orig.derive is None)
        assert meta.SYNTH_PARENTS[rel.name] == ("om_lift", "om_split")

    lemma = loaded[0]
    applied = meta.apply_rewrite_at(lemma, seed, ())
    assert applied is not None
    assert applied == meta.apply_rewrite_at(derived[0], seed, ())

    # fp64-verified on concrete tensors
    env = _env(leaves)
    assert meta._eval_allclose(
        meta._eval_term(seed, env),
        meta._eval_term(applied, env),
        tol=1e-10,
    )

    # and the loaded rule plugs back into the e-graph
    eg = EGraph()
    root = eg.add_term(seed)
    eg.run([lemma], root, max_iterations=3, max_nodes=5_000)
    assert eg.rule_fires.get(lemma.name, 0) > 0
    assert any(
        n.op == "om_apply" for n in eg.get_class(eg.find(root)).nodes
    )


# ---------------------------------------------------------------------------
#  (b) guarded round-trip: composite check vetoes, derive recomputes
# ---------------------------------------------------------------------------


def test_guarded_rule_roundtrip_check_vetoes_after_reload(tmp_path):
    """comm_mul ∘ sdpa_fold_addmul — a guarded derived rule (check +
    derive).  After reload the composite guard still vetoes a non-Const
    scale, and the derive hook recomputes scale=0.5 per firing."""
    seed, (q, k, v, m) = _left_scaled_attention_seed()
    rules = [R.COMM_MUL, _sdpa_fold("sdpa_fold_addmul")]
    derived = meta.synthesize_rules(rules, [seed], fuel=2000)
    hits = [
        d
        for d in derived
        if meta.provenance(d) == ("comm_mul", "sdpa_fold_addmul")
    ]
    assert hits
    d = hits[0]
    assert d.check is not None and d.derive is not None
    assert meta._rhs_derive_placeholders(d.rhs)

    cache = RuleCache(tmp_path)
    key = cache_key(rules, [seed], fuel=2000)
    cache.store(key, derived)
    loaded = next(
        x for x in cache.load(key, rules) if x.name == d.name
    )
    assert loaded.check is not None and loaded.derive is not None
    assert meta._rhs_derive_placeholders(loaded.rhs)

    # accepts the well-formed instance and re-derives scale=0.5
    applied = meta.apply_rewrite_at(loaded, seed, ())
    assert applied is not None and applied.op == "sdpa"
    assert applied.attrs["scale"] == 0.5

    # vetoes: a Var scale fails the composite guard (needs a Const)
    s_var = Var("s", _T())
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, arg1=-2, arg2=-1)
    )
    bad = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make("add", Op.make("mul", s_var, scores), m),
            arg1=-1,
        ),
        v,
    )
    assert meta.apply_rewrite_at(loaded, bad, ()) is None

    # fp64 equivalence on the accepted instance
    env = _env((q, k, v, m))
    assert meta._eval_allclose(
        meta._eval_term(seed, env),
        meta._eval_term(applied, env),
        tol=1e-10,
    )


def test_derive_placeholder_recomputed_on_new_instance(tmp_path):
    """matmul_t_concat (check+derive) ∘ om_lift (check): the RHS's score-
    concat dim rides as a "@1:SD" placeholder.  After reload the rebuilt
    derive re-runs the parent hook — a rank-3 instance gets dim=2, not
    the seed's baked dim=1."""
    q = Var("q", _T(5, 4))
    k1 = Var("k1", _T(3, 4))
    k2 = Var("k2", _T(4, 4))
    v1 = Var("v1", _T(3, 6))
    v2 = Var("v2", _T(4, 6))
    kcat = Op.make("concat", k1, k2, dim=-2)
    vcat = Op.make("concat", v1, v2, dim=-2)
    seed = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make(
                "matmul",
                q,
                Op.make("transpose", kcat, arg1=-2, arg2=-1),
            ),
            arg1=-1,
        ),
        vcat,
    )
    rules = [OM.MATMUL_T_CONCAT, OM.OM_LIFT]
    derived = meta.synthesize_rules(rules, [seed], fuel=6000)
    lemmas = [
        d
        for d in derived
        if "om_elem" in op_repr(d.rhs) and "concat" in op_repr(d.rhs)
    ]
    assert lemmas
    lemma = lemmas[0]

    cache = RuleCache(tmp_path)
    key = cache_key(rules, [seed], fuel=6000)
    cache.store(key, derived)
    loaded = next(
        x for x in cache.load(key, rules) if x.name == lemma.name
    )

    q3 = Var("q3", _T(2, 5, 4))
    k13, k23 = Var("k13", _T(2, 3, 4)), Var("k23", _T(2, 4, 4))
    v13, v23 = Var("v13", _T(2, 3, 6)), Var("v23", _T(2, 4, 6))
    seed3 = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make(
                "matmul",
                q3,
                Op.make(
                    "transpose",
                    Op.make("concat", k13, k23, dim=-2),
                    arg1=-2,
                    arg2=-1,
                ),
            ),
            arg1=-1,
        ),
        Op.make("concat", v13, v23, dim=-2),
    )
    applied3 = meta.apply_rewrite_at(loaded, seed3, ())
    assert applied3 is not None
    score_cat = applied3.args[0].args[0]
    assert score_cat.op == "concat" and score_cat.attrs["dim"] == 2

    env3 = _env((q3, k13, k23, v13, v23))
    ref3 = torch.softmax(
        env3[q3]
        @ torch.cat([env3[k13], env3[k23]], dim=-2).transpose(-2, -1),
        dim=-1,
    ) @ torch.cat([env3[v13], env3[v23]], dim=-2)
    assert meta._eval_allclose(
        meta._eval_term(applied3, env3), ref3, tol=1e-10
    )


# ---------------------------------------------------------------------------
#  (c) cache key misses on changed ruleset / seeds / params
# ---------------------------------------------------------------------------


def test_cache_key_misses_on_changed_inputs(tmp_path):
    seed, _ = _om_lemma_seed()
    seed2, _ = _left_scaled_attention_seed()
    rules = [OM.OM_LIFT, OM.OM_SPLIT]
    derived = meta.synthesize_rules(rules, [seed], fuel=4000)
    cache = RuleCache(tmp_path)
    key = cache_key(rules, [seed], fuel=4000)
    cache.store(key, derived)

    # same inputs hit; any change misses
    assert cache.load(key, rules) is not None
    assert cache_key(rules, [seed], fuel=4000) == key
    assert cache_key(rules, [seed2], fuel=4000) != key
    assert (
        cache_key([*rules, OM.OM_UNLIFT], [seed], fuel=4000) != key
    )
    assert cache_key(rules, [seed], fuel=8000) != key
    assert (
        cache_key(rules, [seed], fuel=4000, numeric_check=False) != key
    )
    assert (
        cache.load(cache_key(rules, [seed2], fuel=4000), rules) is None
    )

    # loading under a ruleset that lacks the parents is a miss, not a
    # half-broken rule set
    other = [R.COMM_MUL]
    assert cache.load(key, other) is None

    # a corrupt entry is a clean miss too
    path = cache.path(key)
    path.write_text("{not json")
    assert cache.load(key, rules) is None


def test_synthesize_rules_cached_wrapper(tmp_path):
    """The convenience wrapper stores on first call and reloads on the
    second, returning equivalent rules both times."""
    seed, _ = _om_lemma_seed()
    rules = [OM.OM_LIFT, OM.OM_SPLIT]
    d1 = synthesize_rules_cached(
        rules, [seed], fuel=4000, cache_dir=tmp_path
    )
    d2 = synthesize_rules_cached(
        rules, [seed], fuel=4000, cache_dir=tmp_path
    )
    assert [r.name for r in d1] == [r.name for r in d2]
    assert meta.apply_rewrite_at(d2[0], seed, ()) is not None


# ---------------------------------------------------------------------------
#  (d) repeated synthesis is measurably faster from cache
# ---------------------------------------------------------------------------


def test_cached_reload_is_faster_than_fresh_synthesis(tmp_path):
    """Full ruleset × several seeds × guarded pairs: the store+reload
    path must undercut a fresh ``synthesize_rules`` run."""
    seeds = [_om_lemma_seed()[0], _left_scaled_attention_seed()[0]]
    rules = meta.module_rules(R) + meta.module_rules(OM)

    t0 = time.perf_counter()
    fresh = meta.synthesize_rules(rules, seeds, fuel=20000)
    t_fresh = time.perf_counter() - t0
    assert fresh

    cache = RuleCache(tmp_path)
    key = cache_key(rules, seeds, fuel=20000)

    t0 = time.perf_counter()
    cache.store(key, fresh)
    t_store = time.perf_counter() - t0

    t0 = time.perf_counter()
    loaded = cache.load(key, rules)
    t_load = time.perf_counter() - t0

    assert loaded is not None
    assert {r.name for r in loaded} == {r.name for r in fresh}
    # store + reload < fresh synthesis
    assert t_store + t_load < t_fresh, (
        f"store={t_store:.4f}s load={t_load:.4f}s fresh={t_fresh:.4f}s"
    )


def test_storing_unguardable_hooks_raises(tmp_path):
    """A hand-built rule with hooks but no guard_pats spec cannot be
    serialized soundly — store must refuse rather than drop the guard."""
    rule = Rewrite(
        "hand_guard",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
        check=lambda bound: True,
    )
    cache = RuleCache(tmp_path)
    with pytest.raises(TypeError):
        cache.store("k", [rule])
