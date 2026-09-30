"""Compositional structural-signature cache — repeated-block reuse.

Transformer stacks repeat one block structure N times with different
parameters; the compositional pass would otherwise run N identical
searches.  The cache canonicalises each block's exported IR modulo leaf
names (``structural_key``) and replays the cached extracted term under
the hit block's own leaf names/values — re-deriving value-dependent
assumptions (tied params, ``__heads`` dedup stacks, the causal fold)
and verifying numerically before grafting.
"""

import torch
import torch.nn as nn

from catopt_core.cost import param_bytes_cost_for
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from catopt_core.pipeline import SearchResult

from catopt_orchestrator import Compositional, Optimizer, structural_key
from catopt_orchestrator.optimize import (
    _block_key,
    _cache_entry,
    _cache_replay,
    _derive_dedup,
    _leaf_equal,
    _param_order,
    _param_shapes_ok,
    _remap_term,
    _term_param_names,
)

from catopt_torch.backend import TorchBackend
from catopt_torch.models import DeepParallel


def _t(*shape):
    return TensorType(tuple(shape))


def _stack(depth, dim=32, mid=48):
    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                *[DeepParallel(dim, mid, dim) for _ in range(depth)]
            )

        def forward(self, x):
            return self.net(x)

    return Stack().eval()


# ---------------------------------------------------------------------------
#  structural_key — canonicalisation unit tests
# ---------------------------------------------------------------------------


def test_structural_key_renames_leaves():
    """Same structure, different leaf names → identical key."""
    x = Var("x", _t(4, 8))
    t1 = Op.make("matmul", x, Param("p_a", _t(8, 8)))
    t2 = Op.make(
        "matmul", Var("input0", _t(4, 8)), Param("w1", _t(8, 8))
    )
    assert structural_key(t1, inputs=[x]) == structural_key(
        t2, inputs=[Var("input0", _t(4, 8))]
    )


def test_structural_key_distinguishes_structure():
    x = Var("x", _t(4, 8))
    y = Var("y", _t(4, 8))
    p = Param("p", _t(8, 8))

    # op name
    assert structural_key(Op.make("matmul", x, p)) != structural_key(
        Op.make("mul", x, p)
    )
    # Var position (non-commutative operand order)
    assert structural_key(
        Op.make("sub", x, y), inputs=[x, y]
    ) != structural_key(Op.make("sub", y, x), inputs=[x, y])
    # Const value AND const type (``Op.make`` normalises positional
    # scalars, so the type tag is exercised on bare Const leaves)
    assert structural_key(Op.make("pow", x, Const(2))) != structural_key(
        Op.make("pow", x, Const(3))
    )
    assert structural_key(Const(2)) != structural_key(Const(2.0))
    # Var arity — an extra (even unused) input is a different block
    assert structural_key(
        Op.make("matmul", x, p), inputs=[x]
    ) != structural_key(Op.make("matmul", x, p), inputs=[x, y])
    # Param shape
    assert structural_key(
        Op.make("matmul", x, Param("p", _t(8, 8)))
    ) != structural_key(Op.make("matmul", x, Param("p", _t(8, 4))))
    # dtype rides in via leaf_meta (TensorType is shape-only)
    meta_f32 = {"p": "torch.float32", "x": "torch.float32"}
    meta_f64 = {"p": "torch.float64", "x": "torch.float32"}
    t = Op.make("matmul", x, p)
    assert structural_key(t, inputs=[x], leaf_meta=meta_f32) != (
        structural_key(t, inputs=[x], leaf_meta=meta_f64)
    )


def test_structural_key_param_positions_canonical():
    """matmul(p_a, p_b) and matmul(p_b, p_a) share a key — the slot
    correspondence is positional, which is exactly what the remap uses."""
    pa, pb = Param("a", _t(4)), Param("b", _t(4))
    assert structural_key(Op.make("matmul", pa, pb)) == structural_key(
        Op.make("matmul", pb, pa)
    )
    # and first-occurrence order records the distinct slots
    assert _param_order(Op.make("matmul", pa, pb)) == ("a", "b")


def test_structural_key_edge_leaves():
    x = Var("x", _t(4))
    p = Param("p", _t(4))
    # shared subterm — memo-hit branch (terms are interned)
    shared = Op.make("neg", x)
    t = Op.make("add", shared, shared)
    assert isinstance(structural_key(t), tuple)
    # Var absent from inputs falls back to its name
    k_named = structural_key(Op.make("mul", x, p))
    k_other = structural_key(Op.make("mul", Var("z", _t(4)), p))
    assert k_named != k_other
    # sequence / mapping / non-term roots
    assert structural_key([x, p]) != structural_key([p, x])
    assert structural_key({"w": p}) == structural_key({"w": p})
    assert structural_key(42) == structural_key(42)
    # unhashable leaf — memo lookup fails closed to a repr key
    assert structural_key({1, 2}) == structural_key({1, 2})
    # attrs canonicalise: term-valued, list-valued and unhashable attrs
    q = Param("q", _t(4))
    a = Op.make(
        "fake",
        x,
        sub=Op.make("neg", p),
        dims=[1, 2],
        blob={1, 2},
        table={"k": p},
        n=3,
        validate=False,
    )
    b = Op.make(
        "fake",
        x,
        sub=Op.make("neg", q),
        dims=[1, 2],
        blob={1, 2},
        table={"k": q},
        n=3,
        validate=False,
    )
    assert structural_key(a, inputs=[x]) == structural_key(b, inputs=[x])


# ---------------------------------------------------------------------------
#  remap / leaf utilities
# ---------------------------------------------------------------------------


def test_remap_term_renames_params_and_vars():
    x, w = Var("x", _t(4)), Var("w", _t(4))
    term = Op.make("matmul", x, Op.make("mul", Param("a", _t(4)), Const(2)))
    out = _remap_term(
        term,
        {"a": "q"},
        {"x": w},
        {},
    )
    expected = Op.make(
        "matmul", w, Op.make("mul", Param("q", _t(4)), Const(2))
    )
    assert out == expected


def test_remap_term_declines_unmapped_leaves():
    x = Var("x", _t(4))
    term = Op.make("mul", x, Param("a", _t(4)))
    assert _remap_term(term, {}, {"x": x}, {}) is None  # unmapped Param
    assert _remap_term(term, {"a": "b"}, {}, {}) is None  # unmapped Var
    # a non-term leaf passes through untouched
    assert _remap_term(7, {}, {}, {}) == 7


def test_param_leaves_and_names():
    p1, p2 = Param("a", _t(4)), Param("b", _t(4))
    term = Op.make("add", p1, Op.make("mul", p2, Const(2)))
    assert _term_param_names(term) == {"a", "b"}
    # DAG sharing — the memo caps re-walks
    shared = Op.make("neg", p1)
    assert _term_param_names(Op.make("add", shared, shared)) == {"a"}
    # unhashable container args are walked through
    assert _term_param_names([p1, [p2]]) == {"a", "b"}


def test_param_shapes_ok():
    term = Op.make(
        "mul", Param("a", _t(4)), Param("unk", TensorType((None, 4)))
    )
    good = {"a": torch.ones(4)}
    assert _param_shapes_ok(term, good)  # 'unk' skipped: None dims
    bad = {"a": torch.ones(5)}
    assert not _param_shapes_ok(term, bad)


def test_leaf_equal():
    assert _leaf_equal(torch.ones(4), torch.ones(4))
    assert not _leaf_equal(torch.ones(4), torch.zeros(4))
    assert not _leaf_equal(torch.ones(4), torch.ones(5))
    assert not _leaf_equal(None, torch.ones(4))
    assert not _leaf_equal(torch.ones(4), None)
    # scalar comparison path (a == b yields a plain bool)
    assert _leaf_equal(_Scalar(), _Scalar())
    assert not _leaf_equal(_Scalar2(), _Scalar())

    class _Raises:
        @property
        def shape(self):
            raise RuntimeError("no shape")

    assert not _leaf_equal(_Raises(), _Scalar())


class _Scalar:
    shape = ()

    def __eq__(self, other):
        return True


class _Scalar2:
    shape = ()

    def __eq__(self, other):
        return False


# ---------------------------------------------------------------------------
#  _derive_dedup — recomputing ``__heads`` stacks for a new block's values
# ---------------------------------------------------------------------------


def _dup_weight(dup=True):
    w = torch.randn(8, 4)
    if dup:
        w = torch.cat([w[:4], w[:4]], dim=0)
    return w


def test_derive_dedup_success_and_mismatch():
    w = _dup_weight()
    rec = {"base": "w", "heads": 2, "imap": (0, 0)}
    val, name = _derive_dedup(rec, {"w": "w2"}, {"w2": w})
    assert name == "w2__heads2"
    assert val.shape == (1, 4, 4)
    assert torch.equal(val[0], w[:4])
    # same structure but the new block's heads are all distinct
    assert _derive_dedup(rec, {"w": "w3"}, {"w3": _dup_weight(dup=False)}) is None


def test_derive_dedup_declines():
    rec = {"base": "w", "heads": 2, "imap": (0, 0)}
    # unmapped base / missing tensor
    assert _derive_dedup(rec, {}, {"w": _dup_weight()}) is None
    assert _derive_dedup(rec, {"w": "w"}, {}) is None
    # non-2D tensor
    assert _derive_dedup(
        rec, {"w": "w"}, {"w": torch.ones(4)}
    ) is None
    # non-tensor leaf value (no .dim)
    assert _derive_dedup(rec, {"w": "w"}, {"w": 5}) is None
    # degenerate head count / indivisible / stale index map
    assert _derive_dedup(
        {"base": "w", "heads": 1, "imap": (0,)}, {"w": "w"},
        {"w": _dup_weight()},
    ) is None
    assert _derive_dedup(
        {"base": "w", "heads": 3, "imap": (0, 0, 0)}, {"w": "w"},
        {"w": _dup_weight()},
    ) is None
    assert _derive_dedup(
        {"base": "w", "heads": 2, "imap": (0,)}, {"w": "w"},
        {"w": _dup_weight()},
    ) is None

    class _Broken:
        shape = (8, 4)

        def dim(self):
            return 2

        def __getitem__(self, s):
            raise RuntimeError("no slicing")

    assert _derive_dedup(rec, {"w": "w"}, {"w": _Broken()}) is None


# ---------------------------------------------------------------------------
#  _cache_entry / _cache_replay — the template record and its instantiation
# ---------------------------------------------------------------------------


def _fake_result(term, ir, stats=None):
    return SearchResult(
        ir=ir,
        eg=EGraph(),
        root_eid=0,
        term=term,
        param_values={},
        stats=stats or {},
    )


def _mk_ir(term, inputs, params):
    return IR(
        root=term,
        inputs=list(inputs),
        input_names={v.name for v in inputs},
        params={p.name: p for p in params},
    )


def _entry(term, params, inputs, **kw):
    e = {
        "term": term,
        "params": tuple(params),
        "inputs": tuple(inputs),
        "term_params": _term_param_names(term),
        "ties": (),
        "derived": {},
        "stats": {},
    }
    e.update(kw)
    return e


def test_cache_entry_prefers_pre_causal_term():
    x = Var("x", _t(4))
    raw = Op.make("sdpa", x, x, x, validate=False)
    folded = Op.make("sdpa", x, x, x, arg5=True, validate=False)
    ir = _mk_ir(raw, [x], [])
    res = _fake_result(
        folded, ir, {"causal_specialized": True, "pre_causal_term": raw}
    )
    entry = _cache_entry(res)
    assert entry["term"] is raw  # the pre-fold form is stored
    res2 = _fake_result(raw, ir)
    assert _cache_entry(res2)["term"] is raw


def test_cache_entry_records_derived_recipes():
    x = Var("x", _t(4))
    derived_name = "w__heads2"
    dedup = Param(derived_name, _t(1, 2, 4))
    # the EXPORTED ir root references the plain leaf; the extracted
    # term swaps in the derived dedup stack.
    ir = _mk_ir(
        Op.make("mul", x, Param("w", _t(4, 4))), [x], [Param("w", _t(4, 4))]
    )
    term = Op.make(
        "mul",
        x,
        Op.make(
            "reshape",
            Op.make("index_select", dedup, dim=0, index=(0, 0)),
            shape=(4, 4),
        ),
    )
    sharing = {
        "ties": [],
        "derived": {
            derived_name: {"base": "w", "heads": 2, "imap": (0, 0)}
        },
    }
    res = _fake_result(term, ir, {"param_sharing": sharing})
    entry = _cache_entry(res)
    assert entry["derived"][derived_name]["base"] == "w"
    # an unrecorded derived name is an unservable recipe (None)
    term2 = Op.make("mul", Param("mystery", _t(4)), x)
    entry2 = _cache_entry(_fake_result(term2, ir, {}))
    assert entry2["derived"] == {"mystery": None}


def test_cache_replay_renames_and_serves():
    # template: mul(x, a); current block: mul(w, q) — same structure,
    # different leaf names and different values.
    xt, wt = Var("x", _t(4)), Var("w", _t(4))
    q = Param("q", _t(4))
    term = Op.make("mul", xt, Param("a", _t(4)))
    ir = _mk_ir(Op.make("mul", wt, q), [wt], [q])
    entry = _entry(term, ("a",), ("x",))
    res = _cache_replay(
        entry,
        ir,
        {"q": torch.ones(4)},
        sink=object(),  # no specialize_causal hook — fold skipped
        source=object(),
        model=None,
    )
    assert res is not None
    assert res.stats["cache_replay"] is True
    assert res.term == Op.make("mul", wt, q)
    assert res.param_values["q"].shape == (4,)


def test_cache_replay_declines():
    x = Var("x", _t(4))
    term = Op.make("mul", x, Param("a", _t(4)))
    ir = _mk_ir(term, [x], [Param("a", _t(4))])
    tensors = {"a": torch.ones(4)}
    kw = dict(sink=object(), source=object(), model=None)
    entry = _entry(term, ("a",), ("x",))

    assert _cache_replay(None, ir, tensors, **kw) is None
    # param arity mismatch
    bad = _entry(term, ("a", "b"), ("x",))
    assert _cache_replay(bad, ir, tensors, **kw) is None
    # input arity mismatch
    bad = _entry(term, ("a",), ("x", "y"))
    assert _cache_replay(bad, ir, tensors, **kw) is None
    # unrecorded derived recipe
    bad = _entry(term, ("a",), ("x",), derived={"d": None})
    assert _cache_replay(bad, ir, tensors, **kw) is None
    # derived recipe that cannot be honoured for these values
    bad = _entry(
        term,
        ("a",),
        ("x",),
        derived={"d": {"base": "a", "heads": 2, "imap": (0, 0)}},
    )
    assert _cache_replay(bad, ir, tensors, **kw) is None  # 'a' is 1-D
    # unmapped leaf in the term
    bad = _entry(
        Op.make("mul", x, Param("ghost", _t(4))), ("a",), ("x",)
    )
    assert _cache_replay(bad, ir, tensors, **kw) is None
    # term param with no value coverage
    bad = _entry(term, ("a",), ("x",))
    assert _cache_replay(bad, ir, {}, **kw) is None
    # remapped declared shape mismatches the supplied tensor
    bad = _entry(Op.make("mul", x, Param("a", _t(8))), ("a",), ("x",))
    assert _cache_replay(bad, ir, tensors, **kw) is None


def test_cache_replay_tie_assumptions():
    x = Var("x", _t(4))
    # template search: ``add(mul(x,a), mul(x,b))`` with a==b bitwise →
    # the extracted term keeps only 'a'.  The current block's export
    # is the same structure under its own names.
    term = Op.make("mul", x, Param("a", _t(4)))
    a2, b2 = Param("a2", _t(4)), Param("b2", _t(4))
    ir = _mk_ir(
        Op.make("add", Op.make("mul", x, a2), Op.make("mul", x, b2)),
        [x],
        [a2, b2],
    )
    kw = dict(sink=object(), source=object(), model=None)
    entry = _entry(term, ("a", "b"), ("x",), ties=[["a", "b"]])

    # current block honours the tie → served
    ok = {"a2": torch.ones(4), "b2": torch.ones(4)}
    assert _cache_replay(entry, ir, ok, **kw) is not None
    # current block breaks it → decline
    bad = {"a2": torch.ones(4), "b2": torch.zeros(4)}
    assert _cache_replay(entry, ir, bad, **kw) is None
    # a cluster whose members are ALL kept by the term needs no check
    both = Op.make(
        "add", Op.make("mul", x, Param("a", _t(4))),
        Op.make("mul", x, Param("b", _t(4))),
    )
    entry2 = _entry(both, ("a", "b"), ("x",), ties=[["a", "b"]])
    ok2 = {"a2": torch.ones(4), "b2": torch.zeros(4)}
    assert _cache_replay(entry2, ir, ok2, **kw) is not None
    # cluster names absent from the block's params are skipped
    entry3 = _entry(term, ("a", "b"), ("x",), ties=[["a", "ghost"]])
    assert _cache_replay(entry3, ir, ok, **kw) is not None


def test_cache_replay_derived_dedup_serves():
    x = Var("x", _t(4))
    w = _dup_weight()
    derived_name = "w__heads2"
    dedup = Param(derived_name, _t(1, 4, 4))
    term = Op.make(
        "reshape",
        Op.make("index_select", dedup, dim=0, index=(0, 0)),
        shape=(8, 4),
    )
    # the hit block's own export references its plain leaf
    w_cur = Param("w", _t(8, 4))
    ir = _mk_ir(Op.make("mul", x, w_cur), [x], [w_cur])
    entry = _entry(
        term,
        ("w",),
        ("x",),
        derived={
            derived_name: {"base": "w", "heads": 2, "imap": (0, 0)}
        },
    )
    res = _cache_replay(
        entry,
        ir,
        {"w": w},
        sink=object(),
        source=object(),
        model=None,
    )
    assert res is not None
    # the recomputed stack lands under the CURRENT derived name
    stack = res.param_values["w__heads2"]
    assert stack.shape == (1, 4, 4)
    assert torch.equal(stack[0], w[:4])


def test_cache_replay_reruns_causal_fold():
    """A sink hook re-folds the pre-causal term on the hit values."""
    x = Var("x", _t(4))
    raw = Op.make("mul", x, Param("a", _t(4)))
    ir = _mk_ir(raw, [x], [Param("a", _t(4))])
    entry = _entry(raw, ("a",), ("x",))

    class _FoldSink:
        def specialize_causal(self, term, params, memo):
            # stand-in: pretend the fold fires, tagging the term
            return Op.make("tagged", term, validate=False)

    res = _cache_replay(
        entry,
        ir,
        {"a": torch.ones(4)},
        sink=_FoldSink(),
        source=object(),
        model=None,
    )
    assert res is not None and res.term.op == "tagged"


# ---------------------------------------------------------------------------
#  _block_key
# ---------------------------------------------------------------------------


def test_block_key_includes_dtypes_and_arity():
    x = Var("x", _t(4))
    p = Param("p", _t(4))
    ir = _mk_ir(Op.make("mul", x, p), [x], [p])
    t32 = {"p": torch.ones(4)}
    t64 = {"p": torch.ones(4, dtype=torch.float64)}
    args = (torch.ones(4),)
    assert _block_key(ir, t32, args) != _block_key(ir, t64, args)
    # a missing captured arg still yields a key (meta marks it)
    assert _block_key(ir, t32, ()) != _block_key(ir, t32, args)


# ---------------------------------------------------------------------------
#  End-to-end: repeated blocks hit the cache
# ---------------------------------------------------------------------------


def test_repeated_blocks_hit_cache_and_verify():
    """4 structurally identical DeepParallel blocks: 1 search + 3 hits."""
    torch.manual_seed(0)
    model = _stack(4)
    x = torch.randn(8, 32)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=Compositional(), verbose=False
    )
    assert stats["cache"] == {"hits": 3, "misses": 1, "fallbacks": 0}
    assert stats["n_optimized"] == 4
    reps = stats["blocks"]
    assert reps["net.0"]["cache"] == "miss"
    for i in (1, 2, 3):
        assert reps[f"net.{i}"]["cache"] == "hit"
    with torch.no_grad():
        d = (model(x.clone()) - opt(x.clone())).abs().max().item()
    assert d < 1e-4


def test_hit_replays_skip_the_search():
    """A hit must not touch the e-graph — the replay's stats say so."""
    torch.manual_seed(0)
    model = _stack(2)
    x = torch.randn(8, 32)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=Compositional(), verbose=False
    )
    hit = stats["blocks"]["net.1"]
    assert hit["cache"] == "hit"
    assert hit["stats"].get("cache_replay") is True


def test_cache_disabled_reports_nothing():
    torch.manual_seed(0)
    model = _stack(2)
    x = torch.randn(8, 32)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=Compositional(cache=False), verbose=False
    )
    assert "cache" not in stats
    assert "cache" not in stats["blocks"]["net.0"]


def test_persistent_dict_reuse_across_models():
    """A caller-owned cache dict serves a second, same-structure model."""
    torch.manual_seed(0)
    cache = {}
    x = torch.randn(8, 32)
    m1 = _stack(2)
    o1 = Optimizer(backend=TorchBackend())
    _a, s1 = o1.optimize(m1, x, strategy=Compositional(cache=cache),
                         verbose=False)
    assert s1["cache"]["misses"] == 1 and s1["cache"]["hits"] == 1

    torch.manual_seed(1)  # different weights, identical structure
    m2 = _stack(2)
    _b, s2 = o1.optimize(m2, x, strategy=Compositional(cache=cache),
                         verbose=False)
    assert s2["cache"]["misses"] == 0 and s2["cache"]["hits"] == 2
    with torch.no_grad():
        assert (m2(x) - _b(x)).abs().max().item() < 1e-4


def test_distinct_structures_never_share():
    """Different block geometries get different keys → all misses."""

    class Mixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                DeepParallel(32, 48, 32), DeepParallel(32, 16, 32)
            )

        def forward(self, x):
            return self.net(x)

    torch.manual_seed(0)
    model = Mixed().eval()
    x = torch.randn(8, 32)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=Compositional(), verbose=False
    )
    assert stats["cache"] == {"hits": 0, "misses": 2, "fallbacks": 0}


def test_dtype_separates_entries():
    """fp64 blocks never replay an fp32 template."""
    torch.manual_seed(0)
    cache = {}
    x32 = torch.randn(8, 32)
    m32 = _stack(2)
    o = Optimizer(backend=TorchBackend())
    _a, s1 = o.optimize(m32, x32, strategy=Compositional(cache=cache),
                        verbose=False)
    assert s1["cache"]["misses"] >= 1

    x64 = torch.randn(8, 32, dtype=torch.float64)
    m64 = _stack(2).double()
    _b, s2 = o.optimize(m64, x64, strategy=Compositional(cache=cache),
                        verbose=False)
    # the fp64 model's first block could NOT replay the fp32 template —
    # it searched; its twin then hit the fresh fp64 entry.
    assert s2["blocks"]["net.0"]["cache"] == "miss"
    assert s2["blocks"]["net.1"]["cache"] == "hit"
    with torch.no_grad():
        assert (m64(x64) - _b(x64)).abs().max().item() < 1e-8


def test_param_mutation_between_blocks_is_safe():
    """Same-structure blocks with *different* values hit and verify —
    values never enter the key, only the remapped leaf binding."""
    torch.manual_seed(0)
    model = _stack(3)
    # scramble the last block's weights AFTER construction — structure
    # unchanged, values completely different
    with torch.no_grad():
        for p in model.net[2].parameters():
            p.copy_(torch.randn_like(p) * 3.0)
    x = torch.randn(8, 32)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=Compositional(), verbose=False
    )
    assert stats["cache"]["hits"] == 2
    with torch.no_grad():
        assert (model(x.clone()) - opt(x.clone())).abs().max().item() < 1e-3


def test_tampered_entry_declines_and_recovers():
    """A cache entry under the right key but with a swapped param map
    must not serve a wrong module — the hit's own verify declines it."""
    torch.manual_seed(0)
    cache = {}
    x = torch.randn(8, 32)
    opt = Optimizer(backend=TorchBackend())
    _a, s1 = opt.optimize(_stack(2), x, strategy=Compositional(cache=cache),
                          verbose=False)
    assert s1["cache"]["misses"] == 1

    # tamper: swap the template's param-name order — the remapped
    # leaves no longer match the declared shapes, so the entry is
    # declined before any lowering runs.
    for entry in cache.values():
        entry["params"] = tuple(reversed(entry["params"]))

    torch.manual_seed(0)
    m2 = _stack(2)
    _b, s2 = opt.optimize(m2, x, strategy=Compositional(cache=cache),
                          verbose=False)
    assert s2["cache"]["fallbacks"] == 2
    assert s2["cache"]["hits"] == 0
    # every block still optimized — via a fresh verified search
    assert s2["n_optimized"] == 2
    for r in s2["blocks"].values():
        assert r["cache"] == "replay_failed"
    with torch.no_grad():
        assert (_b(x) - m2(x)).abs().max().item() < 1e-3

    # tamper harder: a structurally-plausible but WRONG term — the
    # replay lowers fine but the hit's own verify declines it.
    for entry in cache.values():
        entry["params"] = tuple(reversed(entry["params"]))  # restore
        entry["term"] = Op.make("neg", entry["term"])

    torch.manual_seed(0)
    m3 = _stack(2)
    _c, s3 = opt.optimize(m3, x, strategy=Compositional(cache=cache),
                          verbose=False)
    assert s3["cache"]["fallbacks"] == 2
    assert s3["cache"]["hits"] == 0
    assert s3["n_optimized"] == 2
    with torch.no_grad():
        assert (_c(x) - m3(x)).abs().max().item() < 1e-3


def test_unservable_entry_falls_back():
    """An entry whose recipe cannot be honoured declines BEFORE lowering."""

    class DupHeads(nn.Module):
        """Linear whose weight packs two identical head-blocks."""

        def __init__(self, dup=True):
            super().__init__()
            self.lin = nn.Linear(8, 8, bias=False)
            if dup:
                with torch.no_grad():
                    self.lin.weight[4:] = self.lin.weight[:4]

        def forward(self, x):
            return self.lin(x)

    torch.manual_seed(0)

    class Stack(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                DupHeads(), DupHeads(dup=False), DupHeads()
            )

        def forward(self, x):
            return self.net(x)

    model = Stack().eval()
    x = torch.randn(4, 8)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        x,
        strategy=Compositional(),
        verbose=False,
        cost_fn=param_bytes_cost_for(),
    )
    c = stats["cache"]
    assert c["misses"] == 2 and c["hits"] == 1 and c["fallbacks"] == 1
    assert stats["blocks"]["net.1"]["cache"] == "replay_failed"
    assert stats["n_optimized"] == 3
    with torch.no_grad():
        assert (model(x) - opt(x)).abs().max().item() < 1e-5


def test_dedup_records_param_sharing():
    """The miss's stats carry the dedup recipe the cache replays."""
    torch.manual_seed(0)

    class DupHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.lin.weight[4:] = self.lin.weight[:4]

        def forward(self, x):
            return self.lin(x)

    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(DupHeads())

        def forward(self, x):
            return self.net(x)

    x = torch.randn(4, 8)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        One().eval(), x, strategy=Compositional(), verbose=False,
        cost_fn=param_bytes_cost_for(),
    )
    sharing = stats["blocks"]["net.0"]["stats"].get("param_sharing")
    assert sharing and sharing["derived"]


def test_tied_weights_record_ties():
    """Two bitwise-equal params inside a block record a tie cluster."""
    torch.manual_seed(0)

    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = nn.Linear(8, 8, bias=False)
            self.w2 = nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.w2.weight.copy_(self.w1.weight)

        def forward(self, x):
            return self.w2(x) + self.w1(x) * 0.5

    class One(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(Tied())

        def forward(self, x):
            return self.net(x)

    x = torch.randn(4, 8)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        One().eval(), x, strategy=Compositional(), verbose=False
    )
    sharing = stats["blocks"]["net.0"]["stats"].get("param_sharing")
    assert sharing and sharing["ties"]


def test_verbose_hit_prints(capsys):
    torch.manual_seed(0)
    model = _stack(2)
    x = torch.randn(8, 32)
    _opt, stats = Optimizer(backend=TorchBackend()).optimize(
        model, x, strategy=Compositional(), verbose=True
    )
    assert stats["cache"]["hits"] == 1
    assert "cache hit" in capsys.readouterr().out


def test_replay_lower_failure_falls_back(monkeypatch):
    """A replay that raises inside lower() reverts to the search path."""
    torch.manual_seed(0)
    model = _stack(2)
    x = torch.randn(8, 32)
    opt = Optimizer(backend=TorchBackend())
    real_lower = opt.lower

    def flaky_lower(result, *a, **kw):
        if result.stats.get("cache_replay"):
            raise RuntimeError("boom")
        return real_lower(result, *a, **kw)

    monkeypatch.setattr(opt, "lower", flaky_lower)
    _m, stats = opt.optimize(
        model, x, strategy=Compositional(), verbose=False
    )
    assert stats["cache"]["fallbacks"] == 1
    assert stats["cache"]["misses"] == 2
    assert stats["n_optimized"] == 2
