"""Coverage tests for catopt.rulecache — the JSON codec (attrs, terms,
bindings, rule records incl. guarded re-expression), fingerprinting,
the filesystem store/load contract, and the cached-synthesis key
derivation.  Real codec paths only; the synthesis itself is stubbed
where the cache mechanics are what's under test."""
# ruff: noqa: E741 — test-idiom unpacking

import functools
import json

import pytest
import torch

import catopt.meta as meta
from catopt.egraph import Rewrite
from catopt.ir import Const, Op, Param, TensorType, Var
from catopt.rulecache import (
    CACHE_VERSION,
    RuleCache,
    _dec_attr,
    _dec_binding,
    _dec_rule,
    _dec_term,
    _enc_attr,
    _enc_binding,
    _enc_rule,
    _enc_term,
    _hook_sig,
    _sha,
    cache_key,
    ruleset_fingerprint,
    seed_fingerprint,
    synthesize_rules_cached,
)


def _T(*shape):
    return TensorType(tuple(shape))


def _v(name, *shape):
    return Var(name, _T(*shape))


def _p(name, *shape):
    return Param(name, _T(*shape))


def _rw(name, lhs, rhs, **kw):
    return Rewrite(name=name, lhs=lhs, rhs=rhs, law="l", **kw)


# ---------------------------------------------------------------------------
#  attr / term / binding codec
# ---------------------------------------------------------------------------


def test_attr_codec_roundtrips_and_rejections():
    for v in (3, 2.5, "mvar", True, None):
        assert _dec_attr(_enc_attr(v)) == v
    assert _dec_attr(_enc_attr((1, 2))) == (1, 2)
    assert _dec_attr(_enc_attr([1, "a", True])) == [1, "a", True]
    assert _dec_attr(_enc_attr((1, [2, 3]))) == (1, [2, 3])
    # unserializable value → TypeError at encode
    with pytest.raises(TypeError):
        _enc_attr(torch.zeros(2))
    with pytest.raises(TypeError):
        _enc_attr(object())
    # corrupt marker dict → ValueError at decode
    with pytest.raises(ValueError):
        _dec_attr({"__nope__": 1})
    # non-dict marker passes through
    assert _dec_attr("plain") == "plain"


def test_term_codec_all_forms():
    mv = "$a"
    assert _dec_term(_enc_term(mv)) == "$a"
    c = Const(2.5)
    assert _dec_term(_enc_term(c)).value == 2.5
    x = _v("x", 2, 3)
    assert _dec_term(_enc_term(x)) == x
    p = _p("w", 4, 4)
    assert _dec_term(_enc_term(p)) == p
    t = Op.make(
        "linear", x, _p("W", 4, 3), _p("b", 4), arg1=-1
    )
    d = _dec_term(_enc_term(t))
    assert d == t and repr(d) == repr(t)
    # tuple/list attrs survive via the marker dicts
    t2 = Op.make("sum", x, arg1=(0, 1), arg2=[True])
    d2 = _dec_term(_enc_term(t2))
    assert d2.attrs["arg1"] == (0, 1)
    # unserializable term → TypeError
    with pytest.raises(TypeError):
        _enc_term(42)
    with pytest.raises(TypeError):
        _enc_term([1, 2])
    # corrupt encodings → ValueError, never silent garbage
    with pytest.raises(ValueError):
        _dec_term(None)
    with pytest.raises(ValueError):
        _dec_term({})
    with pytest.raises(ValueError):
        _dec_term({"junk": 1})
    with pytest.raises(ValueError):
        _dec_term({"op": "relu", "args": [{"nope": 1}], "attrs": {}})


def test_binding_codec():
    x = _v("x", 4)
    m = {
        "x": Op.make("relu", x),
        "$attr:dim": 1,
        "$attr:beta": "$beta_val",
        "plain": Const(0.5),
    }
    d = _dec_binding(_enc_binding(m))
    assert repr(d["x"]) == repr(m["x"])
    assert d["$attr:dim"] == 1
    assert d["$attr:beta"] == "$beta_val"
    assert d["plain"].value == 0.5


# ---------------------------------------------------------------------------
#  rule records
# ---------------------------------------------------------------------------


def test_rule_codec_unguarded_roundtrip():
    x = _v("x", 4)
    r = _rw(
        "synth_demo",
        Op.make("relu", Op.make("relu", x)),
        Op.make("relu", x),
    )
    object.__setattr__(r, "parents", ("p_a", "p_b"))
    meta.SYNTH_PARENTS.pop("synth_demo", None)
    rec = _enc_rule(r)
    assert rec["parents"] == ["p_a", "p_b"]
    assert rec["guard"] is None
    d = _dec_rule(rec, {})
    assert d.name == "synth_demo" and d.law == "l"
    assert repr(d.lhs) == repr(r.lhs) and repr(d.rhs) == repr(r.rhs)
    assert d.check is None and d.derive is None
    assert d.parents == ("p_a", "p_b")
    # provenance re-registers globally on decode
    assert meta.SYNTH_PARENTS["synth_demo"] == ("p_a", "p_b")
    meta.SYNTH_PARENTS.pop("synth_demo", None)


def test_rule_encode_guarded_without_spec_rejected():
    x = _v("x", 4)
    r = _rw("bad", Op.make("neg", x), x, check=lambda *a: True)
    with pytest.raises(TypeError, match="guard_pats"):
        _enc_rule(r)
    r2 = _rw("bad2", Op.make("neg", x), x, derive=lambda *a: {})
    with pytest.raises(TypeError, match="guard_pats"):
        _enc_rule(r2)


def test_rule_decode_guarded_missing_parent_is_miss(tmp_path):
    """A guarded record whose parents aren't in the caller's ruleset:
    the decode surfaces the miss as a clean load failure."""
    cache = RuleCache(str(tmp_path))
    x = _v("x", 4)
    # hand-minted guarded record (guard_pats spec present but parents
    # unknown to the loader)
    rec = {
        "name": "g",
        "law": "l",
        "parents": ["gone_a", "gone_b"],
        "lhs": _enc_term(x),
        "rhs": _enc_term(x),
        "guard": {
            "pat1": _enc_binding({}),
            "pat2": _enc_binding({}),
        },
    }
    key = "k1"
    p = cache.path(key)
    cache.dir.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {"version": CACHE_VERSION, "key": key, "rules": [rec]}
        )
    )
    assert cache.load(key, rules=[]) is None


# ---------------------------------------------------------------------------
#  fingerprints and cache key
# ---------------------------------------------------------------------------


def _om_rules():
    from catopt_core.laws import all_rules

    return all_rules()


def test_ruleset_fingerprint_order_and_hooks():
    rules = _om_rules()
    fp1 = ruleset_fingerprint(rules)
    fp2 = ruleset_fingerprint(list(reversed(rules)))
    assert fp1 == fp2  # order-independent
    # changing a hook changes the fingerprint
    r0 = rules[0]
    r0p = _rw(r0.name + "_x", r0.lhs, r0.rhs)
    fp3 = ruleset_fingerprint([*rules[1:], r0p])
    assert fp3 != fp1


def test_seed_fingerprint_and_cache_key():
    x, y = _v("x", 4), _v("y", 4)
    a = Op.make("relu", x)
    b = Op.make("relu", y)
    f1 = seed_fingerprint([a, b])
    f2 = seed_fingerprint([b, a])
    assert f1 != f2  # seed order matters
    rules = _om_rules()
    k1 = cache_key(rules, [a], fuel=512)
    k2 = cache_key(rules, [a], fuel=512)
    assert k1 == k2
    assert cache_key(rules, [b], fuel=512) != k1
    assert cache_key(rules, [a], fuel=1024) != k1
    assert (
        cache_key(rules, [a], fuel=512, numeric_check=False) != k1
    )
    assert (
        cache_key(rules, [a], fuel=512, require_overlap=False) != k1
    )
    assert (
        cache_key(rules, [a], fuel=512, emit_subsumed=True) != k1
    )


def test_hook_sig_variants():
    def fn():
        return 1

    assert _hook_sig(None) == ""
    s1 = _hook_sig(fn)
    s2 = _hook_sig(fn)  # deterministic
    assert s1 == s2 and len(s1) == 16
    partial = functools.partial(max, 1)
    s3 = _hook_sig(partial)
    assert s3.startswith("partial(") and s3 != s1
    # a builtin (no source) falls back to module:qualname — stable
    assert _hook_sig(max) == _hook_sig(max)


def test_sha_empty_and_ordered():
    assert len(_sha([])) == 64
    assert _sha(["a", "b"]) != _sha(["b", "a"])
    assert _sha(["ab"]) != _sha(["a", "b"])  # NUL-separated fields


# ---------------------------------------------------------------------------
#  filesystem store/load
# ---------------------------------------------------------------------------


def test_store_load_roundtrip_and_misses(tmp_path):
    cache = RuleCache(str(tmp_path))
    x = _v("x", 4)
    rules = [
        _rw("r1", Op.make("neg", Op.make("neg", x)), x),
        _rw("r2", Op.make("add", x, Const(0.0)), x),
    ]
    key = cache_key(rules, [x], fuel=4)
    p = cache.store(key, rules)
    assert p == cache.path(key) and p.exists()
    raw = json.loads(p.read_text())
    assert raw["version"] == CACHE_VERSION and raw["key"] == key

    loaded = cache.load(key)
    assert loaded is not None and len(loaded) == 2
    for o, l in zip(rules, loaded, strict=True):
        assert o.name == l.name and o.law == l.law
        assert repr(o.lhs) == repr(l.lhs)
        assert repr(o.rhs) == repr(l.rhs)

    # wrong key → miss even though the file exists
    assert cache.load("otherkey") is None
    # missing file → miss
    assert RuleCache(str(tmp_path / "empty")).load(key) is None
    # corrupt file → clean miss
    _bad = tmp_path / f"rules-{key}-bad.json"
    cache2 = RuleCache(str(tmp_path))
    key2 = "corrupt"
    cache2.path(key2).write_text("{nope")
    assert cache2.load(key2) is None
    # version mismatch → miss
    key3 = "vermismatch"
    cache2.path(key3).write_text(
        json.dumps({"version": 999, "key": key3, "rules": []})
    )
    assert cache2.load(key3) is None
    # key inside payload doesn't match request → miss
    key4 = "k4"
    cache2.path(key4).write_text(
        json.dumps(
            {"version": CACHE_VERSION, "key": "other", "rules": []}
        )
    )
    assert cache2.load(key4) is None


def test_store_rejects_unguardable_rule(tmp_path):
    cache = RuleCache(str(tmp_path))
    x = _v("x", 4)
    hooked = _rw(
        "h",
        Op.make("neg", x),
        x,
        check=lambda *a: True,
    )
    with pytest.raises(TypeError):
        cache.store("k", [hooked])


# ---------------------------------------------------------------------------
#  synthesize_rules_cached — hit/miss mechanics
# ---------------------------------------------------------------------------


def test_synthesize_rules_cached_hits_and_misses(
    tmp_path, monkeypatch
):
    calls = {"n": 0}
    x = _v("x", 4)

    def stub(rules, seeds, **kw):
        calls["n"] += 1
        return [
            _rw(f"derived_{calls['n']}", Op.make("relu", x), x)
        ]

    monkeypatch.setattr(meta, "synthesize_rules", stub)
    parents = _om_rules()[:3]
    seeds = [Op.make("relu", x)]
    d1 = synthesize_rules_cached(
        parents, seeds, fuel=64, cache_dir=str(tmp_path)
    )
    assert calls["n"] == 1 and len(d1) == 1
    # identical problem → cache hit, synthesis NOT re-run
    d2 = synthesize_rules_cached(
        parents, seeds, fuel=64, cache_dir=str(tmp_path)
    )
    assert calls["n"] == 1
    assert [r.name for r in d2] == [r.name for r in d1]
    # different fuel → miss → second synthesis
    _d3 = synthesize_rules_cached(
        parents, seeds, fuel=128, cache_dir=str(tmp_path)
    )
    assert calls["n"] == 2
    # different seeds → miss
    synthesize_rules_cached(
        parents, [Op.make("tanh", x)], fuel=128,
        cache_dir=str(tmp_path),
    )
    assert calls["n"] == 3
    # a different cache dir → miss
    synthesize_rules_cached(
        parents,
        seeds,
        fuel=128,
        cache_dir=str(tmp_path / "other"),
    )
    assert calls["n"] == 4
