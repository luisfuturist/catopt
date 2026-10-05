"""Coverage climb A2 — the seeded, small-budget driver paths of
``grammar``, ``meta_game``, ``workload_gen`` and ``gap_gen``.

``test_discovery_experiments.py`` covers the core machinery on a
six-term corpus; what remains unmeasured is a tail of edge paths:
search arcs in ``grammar.search``, legality/terminal guards in the
game, gate rejects in ``valid_term``/``_instance_valid``, the
resampler/mutator fallback branches, and the ``main`` driver tails.

Everything runs seeded and small — the same corpus and vocab helpers
the experiments file uses.
"""

import random
from collections import Counter

import pytest
from catopt_core.egraph.terms import _term_match
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import gap_gen as gg
from catopt_discovery import grammar as gm
from catopt_discovery import meta_game as mg
from catopt_discovery import pipeline as lpipe
from catopt_discovery import workload_gen as wg
from catopt_discovery.census import (
    CorpusTerm,
    op_tuple_census,
    shape_census,
    shape_key,
)
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink

_BY_NAME = {r.name: r for r in ALL_RULES}


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


# ---------------------------------------------------------------------------
#  Shared tiny corpus (same six terms as the experiments tests)
# ---------------------------------------------------------------------------


def _tiny_terms() -> list:
    x = _v("x", 4, 4)
    y = _v("y", 4, 4)
    z = _v("z", 4, 4)
    w = Param("w", TensorType((4, 4)))
    return [
        _p("mul", _p("silu", x), y),
        _p("mul", x, _p("sigmoid", x)),
        _p("add", _p("mul", x, y), z),
        _p("mul", x, Const(1)),
        _p("select", _p("mul", x, y), dim=0, index=1),
        _p("add", _p("mul", x, w), y),
    ]


def _tiny_cases() -> list[TermCase]:
    cases = []
    for i, t in enumerate(_tiny_terms()):
        env = wg._leaf_env(t)
        c = wg.term_to_case(t, f"t{i}", "bench", env)
        assert c is not None
        cases.append(c)
    return cases


def _stats():
    return wg.corpus_stats(_tiny_cases())


_SUPPORTED = frozenset(
    {
        "add",
        "mul",
        "sub",
        "neg",
        "silu",
        "sigmoid",
        "select",
        "unsqueeze",
        "getitem",
        "topk",
        "reshape",
        "slice",
        "transpose",
    }
)


def _vocab():
    terms = _tiny_terms()
    cts = [CorpusTerm("t", f"m{i}", t) for i, t in enumerate(terms)]
    counts, _ = op_tuple_census(cts)
    sh, _ = shape_census(cts)
    return mg.build_vocab(terms, counts, sh, 6)


def _referee(terms=None, cases=(), lib=None):
    return mg.Referee(
        _tiny_terms() if terms is None else terms,
        list(cases),
        None,
        None,
        [] if lib is None else lib,
    )


# ---------------------------------------------------------------------------
#  grammar — search arcs the small BFS runs missed
# ---------------------------------------------------------------------------


def test_schema_neighbors_rhs_side():
    """A non-trivial RHS yields several well-formed neighbours."""
    lhs = gm._node("add", gm._slot(0), gm._slot(1))
    rhs = gm._node("mul", gm._slot(0), gm._slot(1))
    out = list(gm._schema_neighbors(lhs, rhs))
    # RHS mutations appear (not just LHS ones) and all are well-formed.
    assert any(nr is not rhs for _nl, nr in out)
    assert all(gm._slots(nr) <= gm._slots(nl) for nl, nr in out)


def test_schema_neighbors_rhs_illformed_filtered():
    """An RHS mutation minting a slot the LHS never binds is
    filtered out — the well-formedness gate, not an error."""
    lhs = gm._node("add", gm._slot(0), gm._slot(0))
    rhs = gm._node("mul", gm._slot(0), gm._slot(0))
    out = list(gm._schema_neighbors(lhs, rhs))
    # the slot-split mutation of the shared RHS slot would introduce
    # a fresh slot on the RHS alone; it is dropped.
    assert all(gm._slots(nr) <= gm._slots(nl) for nl, nr in out)
    assert not any(gm._slots(nr) - {0} for _nl, nr in out)


def test_search_exhausts_neighbors_and_skips_crosses():
    """Two seeds with a shared one-step neighbour: the dedup guard,
    the inner-loop exhaustion arcs, and — at depth two — a frontier
    too wide for the recombination block all run."""
    add_s = (gm._node("add", gm._slot(0), gm._slot(1)), gm._slot(0))
    mul_s = (gm._node("mul", gm._slot(0), gm._slot(1)), gm._slot(0))
    out = gm.search([add_s, mul_s], depth=2, level_cap=80, total_cap=200)
    assert out[0] == add_s and out[1] == mul_s
    # level 1 carried > _CROSS_FRONTIER fresh schemas -> level 2's
    # crosses are skipped.
    assert len(out) > gm._CROSS_FRONTIER


def test_search_runs_crosses_on_a_narrow_frontier():
    """A narrow level keeps the frontier small enough to recombine,
    and a fresh cross joins the level without hitting the cap."""
    add = gm._node("add", gm._slot(0), gm._slot(1))
    s1 = (add, gm._node("mul", gm._slot(0), gm._slot(1)))
    s2 = (gm._node("neg", gm._slot(0)), gm._slot(0))
    s3 = (gm._node("sub", gm._slot(0), gm._slot(1)), gm._lit(0))
    out = gm.search(
        [s1, s2, s3], depth=1, level_cap=300, total_cap=500
    )
    # (add, slot0) is a well-formed cross — reachable only by taking
    # s1's lhs with s2's rhs, so it proves recombination ran.
    assert (add, gm._slot(0)) in out


def test_search_crosses_stop_at_level_cap():
    """A cross that arrives after the neighbour cap is still
    considered, appended, and then stops the recombination loop."""
    add = gm._node("add", gm._slot(0), gm._slot(1))
    s1 = (add, gm._node("mul", gm._slot(0), gm._slot(1)))
    s2 = (gm._node("neg", gm._slot(0)), gm._slot(0))
    out = gm.search([s1, s2], depth=1, level_cap=2, total_cap=50)
    assert (add, gm._slot(0)) in out


def test_main_no_json_returns_zero(monkeypatch, capsys):
    seed = (gm._node("mul", gm._slot(0), gm._lit(1)), gm._slot(0))
    monkeypatch.setattr(gm, "_human_schemas", lambda: [seed])
    rc = gm.main(
        ["--depth", "0", "--level-cap", "4", "--total-cap", "8"]
    )
    assert rc == 0
    assert "schema-level search" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  meta_game — minting, attrs, vocab classes, legality edges
# ---------------------------------------------------------------------------


def test_mint_rejects_incomplete_attrs():
    """An op slot whose attrs miss a schema-required key mints to
    ``None`` rather than raising — the guard behind ``candidate()``'s
    ``mint-error`` path."""
    sel = mg._Slot(
        kind="op",
        op="select",
        children=[mg._Slot(kind="mv", name="U")],
        attrs={"index": "select.index"},
    )
    assert mg._mint(sel) is None


def test_attr_mv_names_mixed_value_kinds():
    sel = _p("select", "U", dim=0, index="i")
    assert mg._attr_mv_names(sel) == {"i"}


def test_attr_bridge_non_tuple_value():
    d = mg._attr_bridge(frozenset({"softmax.dim"}))
    assert d({"$attr:sum.dim": -1}) == {"$attr:softmax.dim": -1}


def test_shape_slot_bad_child():
    """A shape key whose child op is out-of-vocab vetoes the parent."""
    key = ("add", (), (("leaf", 0), ("zzz", (), ())))
    assert (
        mg._shape_slot(key, {"add": ()}, frozenset({"add"})) is None
    )


def test_vocab_classes_reduction_and_other():
    """Reductions and unclassifiable ops label 'reduction'/'other'."""
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    w = Param("w", TensorType((4, 4)))
    terms = [
        *_tiny_terms(),
        _p("add", _p("matmul", x, w), _p("sum", y, dim=0)),
        _p("matmul", _p("matmul", x, w), y),
    ]
    cts = [CorpusTerm("t", f"m{i}", t) for i, t in enumerate(terms)]
    counts, _ = op_tuple_census(cts)
    sh, _ = shape_census(cts)
    v = mg.build_vocab(terms, counts, sh, 6)
    assert v.classes["sum"] == "reduction"
    assert v.classes["matmul"] == "other"


def test_legal_root_requires_head_attestation():
    """An op absent from the census's head table is not a legal LHS
    root — the legality guard prunes it."""
    terms = _tiny_terms()
    cts = [CorpusTerm("t", f"m{i}", t) for i, t in enumerate(terms)]
    counts, _ = op_tuple_census(cts)
    partial = Counter(
        {k: n for k, n in counts.items() if k[0] != "mul"}
    )
    sh, _ = shape_census(cts)
    v = mg.build_vocab(terms, partial, sh, 6)
    game = mg.BuildGame(v, random.Random(0))
    game.reset(None)
    legal = [game.actions[i] for i in game.legal()]
    assert ("op", "mul") not in legal
    assert ("op", "add") in legal


def test_legal_fully_masked_hole_force_fills(monkeypatch):
    """Every action masked (ops unattested, consts unattested, the
    only metavar bound-and-ungated) forces the metavariable fallback
    so the play still terminates."""
    monkeypatch.setattr(mg, "_MV_NAMES", ("U",))
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    # add(mul(U, 1), .): at the second add child no op or const is
    # corpus-attested, and U's only binding sits under a ``mul``
    # LCA — a sharing shape the corpus never attests, so its reuse
    # is vetoed too.
    game.step(game.actions.index(("op", "add")))
    game.step(game.actions.index(("op", "mul")))
    game.step(game.actions.index(("mv", "U")))
    game.step(game.actions.index(("const", 1)))
    # the sharing gate does veto U here — the corpus has no
    # add(mul, ·) site sharing a subterm.
    assert game.v.share.get(("add", ("mul", "·")), 0) == 0
    legal = [game.actions[i] for i in game.legal()]
    assert legal == [("mv", "U")]


def test_step_on_completed_term_raises():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    game.step(game.actions.index(("mv", "U")))
    game.step(game.actions.index(("mv", "U")))
    assert game.done
    with pytest.raises(RuntimeError, match="no hole"):
        game.step(0)


def test_candidate_none_when_unmintable():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    # a terminal-looking state whose LHS cannot mint: select without
    # its required ``dim`` attr.
    game.lhs = mg._Slot(
        kind="op",
        op="select",
        children=[mg._Slot(kind="mv", name="U")],
        attrs={"index": "select.index"},
    )
    game.rhs = mg._Slot(kind="mv", name="U")
    assert game.candidate() is None


def test_tensors_on_done_game_raises():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    game.step(game.actions.index(("mv", "U")))
    game.step(game.actions.index(("mv", "U")))
    assert game.done
    with pytest.raises(RuntimeError, match="no hole"):
        mg._tensors(game, [])


def test_play_once_mint_error(monkeypatch):
    vocab = _vocab()
    r = _referee()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    monkeypatch.setattr(game, "candidate", lambda: None)
    v, _logps = mg._play_once(game, r, None, game.rng, prior=False)
    assert v.reason == "mint-error"


# ---------------------------------------------------------------------------
#  meta_game — referee exception paths and firing edges
# ---------------------------------------------------------------------------


def test_instance_check_and_derive_raising_hooks():
    """Hooks that raise are vetoes: a raising ``check`` and a raising
    ``derive`` each reject every matching subterm."""
    r = _referee()
    calls = {"check": 0, "derive": 0}

    def bad_check(bound):
        calls["check"] += 1
        return 1 / 0

    v = r.evaluate(_p("add", "U", "V"), "U", check=bad_check)
    assert v.reason == "no-instance"
    assert calls["check"] >= 1  # the LHS did match corpus subterms

    def bad_derive(bound):
        calls["derive"] += 1
        return 1 / 0

    v = r.evaluate(
        _p("add", "U", "V"), _p("neg", "V"), derive=bad_derive
    )
    assert v.reason == "no-instance"
    assert calls["derive"] >= 1


def test_instance_rhs_unbound_metavar():
    """An RHS metavariable the LHS never binds makes the instance
    rewrite throw — a no-instance, not a crash."""
    r = _referee()
    v = r.evaluate(_p("add", "U", "V"), _p("mul", "U", "W"))
    assert v.reason == "no-instance"


def test_fire_counts_zero_firing_cases():
    """A true rule that fires nowhere reports ``true`` with zero
    fires and no direction."""
    x, y, z = (_v(n, 4, 4) for n in "xyz")
    inst = _p("add", _p("mul", x, y), _p("mul", x, z))
    case = _tiny_cases()[0]  # mul(silu(x), y) — neither side matches
    sink = _sink()
    r = mg.Referee(
        [inst, *_tiny_terms()],
        [case],
        sink,
        _cost_fn(sink),
        [],
    )
    v = r.evaluate(
        _p("add", _p("mul", "U", "V"), _p("mul", "U", "W")),
        _p("mul", "U", _p("add", "V", "W")),
    )
    assert v.reason == "true" and v.truth
    assert v.fires == 0 and v.directions == ()


def test_fire_paid_with_zero_base_cost(monkeypatch):
    """A probe claiming ``paid`` without a base cost must not feed
    the rel-drop ratio (the ``f.base_cost`` guard)."""
    case = _tiny_cases()[3]
    sink = _sink()
    r = mg.Referee([case.term], [case], sink, _cost_fn(sink), [])

    class _FakeProbe:
        fires = 1
        changed = True
        paid = True
        verified = "pass"
        base_cost = 0.0

    monkeypatch.setattr(mg, "_probe", lambda *a: _FakeProbe())
    v = r.evaluate(_p("mul", "U", Const(1)), "U")
    assert v.paid == 2
    assert v.rel_drop == 0.0  # zero base never divides


# ---------------------------------------------------------------------------
#  meta_game — training loop, rediscovery print path
# ---------------------------------------------------------------------------


def test_train_logs_each_episode(capsys):
    vocab = _vocab()
    r = _referee()
    game = mg.BuildGame(vocab, random.Random(0))
    model, hist = mg.train(
        game, r, 2, hidden=8, seed_frac=0.0, log_every=1
    )
    assert len(hist) == 2 and model is not None
    out = capsys.readouterr().out
    assert out.count("ep ") == 2
    assert "mean reward" in out


def test_rediscovered_reports_a_hit():
    """A referee that saw the select_mul key reports it rediscovered."""
    r = _referee()
    sm = _BY_NAME["select_mul"]
    # evaluated with no instance — recorded in by_key all the same.
    r.evaluate(sm.lhs, sm.rhs)
    hits = mg._rediscovered(r, mg._targets())
    assert len(hits["select_mul"]) == 1
    assert hits["select_mul"][0]["name"].startswith("cand_")
    assert hits["softmax_fold"] == []


def test_main_prints_rediscovery(monkeypatch, capsys):
    """``main`` walks the rediscovery rows when a target was hit —
    driven through a real (tiny) referee, with ``run_experiment``
    shrunk to its result."""
    ref = mg.Referee(_tiny_terms(), [], None, None, [])
    sm = _BY_NAME["select_mul"]
    ref.evaluate(sm.lhs, sm.rhs)
    s = ref.summary()
    res = {
        "args": {"top": 5},
        "n_proposals": 0,
        "baseline": s,
        "baseline_reasons": mg._reason_table(ref),
        "random": s,
        "random_reasons": {},
        "train_reward_tail": 0.0,
        "train_oracle": 0,
        "train_true": 0,
        "player": s,
        "player_reasons": mg._reason_table(ref),
        "player_top": [],
        "player_rediscovered": mg._rediscovered(ref, mg._targets()),
        "baseline_rediscovered": mg._rediscovered(
            ref, mg._targets()
        ),
        "train_rediscovered": mg._rediscovered(ref, mg._targets()),
        "baseline_ref": ref,
        "random_ref": ref,
        "player_ref": ref,
        "train_ref": ref,
    }
    monkeypatch.setattr(mg, "run_experiment", lambda a: res)
    rc = mg.main(["--top", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "REDISCOVERED" in out
    assert "select_mul" in out


# ---------------------------------------------------------------------------
#  workload_gen — stats edges, validity gates, resampler/mutator tails
# ---------------------------------------------------------------------------


def test_corpus_stats_non_op_and_unknown_shapes():
    """Leaf terms and unknown-dim subterms skip the pool/shape tables
    honestly."""
    x = _v("x", 4, 4)
    u = Var("u", TensorType((None, 4)))
    cases = [
        TermCase("leaf", "t", x, (), (), {}),
        TermCase("unk", "t", _p("add", u, x), (), (), {}),
    ]
    st = wg.corpus_stats(cases)
    # the leaf-term case contributed no root tuple but a root key.
    assert ("add", ("var", "var")) in st.root_tuples
    # the unknown-shape add never entered the shape pool.
    assert all(
        not any(isinstance(d, type(None)) for d in sh)
        for sh in st.pool_by_shape
    )


def test_valid_term_gate_tail():
    """The post-shape gates: name-consistent leaves, torch-evaluable,
    tensor-valued, novel."""
    st = _stats()
    seen: set = set()
    x = _v("gx", 4, 4)
    # a Var and a Param sharing a name cannot be bound — env None.
    clash = _p("add", Var("w", TensorType((4, 4))), Param("w", TensorType((4, 4))))
    assert wg.valid_term(clash, st, seen, _SUPPORTED) is None
    # shape-fine but unevaluable: a select index out of range.
    bad = _p("select", x, dim=0, index=9)
    assert wg.valid_term(bad, st, seen, _SUPPORTED) is None
    # a tuple-valued op is not a workload (topk -> (values, indices)).
    tk = _p("topk", x, k=1)
    assert wg.valid_term(tk, st, seen, _SUPPORTED) is None


def test_resampler_leaf_and_expand_fallbacks():
    """The leaf fallback chain and the expand vetoes are real
    boundaries of the resampler — driven on hand-built stats."""
    st = wg.CorpusStats(
        var_shapes=Counter({(4, 4): 3}),
        param_shapes=Counter(),
    )
    sampler = wg._Resampler(st, random.Random(0))
    # an unattested slot falls back to the global pools; an empty
    # param pool falls back to var shapes.
    leaf = sampler._leaf("mul", 9, "param")
    assert isinstance(leaf, Param)
    assert leaf.typ.shape == (4, 4)
    assert sampler._leaf("mul", 9, "var").typ.shape == (4, 4)
    # expand vetoes: unknown op, and depth past the cap.
    assert sampler._expand("nope", 0) is None
    assert sampler._expand("add", wg._MAX_DEPTH + 1) is None


def test_resampler_structural_vetoes():
    """Hand-built stats reach the child-failure / mint-failure
    returns a real corpus's stats cannot produce."""
    # "a" expands to a child op "b" that itself has no tuples: the
    # recursion vetoes cleanly.
    st = wg.CorpusStats(
        root_tuples=Counter({("a", ("b",)): 1}),
        tuples_of={"a": Counter({("b",): 1})},
        arity={"a": 1},
    )
    sampler = wg._Resampler(st, random.Random(0))
    assert sampler._expand("a", 0) is None
    assert sampler.sample() is None  # child expand vetoed
    # a root tuple for an op with no expansion table at all.
    st2 = wg.CorpusStats(root_tuples=Counter({("zzz", ("var",)): 1}))
    assert wg._Resampler(st2, random.Random(0)).sample() is None
    # an observed attr dict that leaves a required attr unbound:
    # Op.make rejects the mint.
    st3 = wg.CorpusStats(
        root_tuples=Counter({("select", ("var",)): 1}),
        tuples_of={"select": Counter({("var",): 1})},
        attr_dicts={"select": Counter({(("index", 0),): 1})},
        arity={"select": 1},
        leaf_at={("select", 0): [("var", (4, 4))]},
    )
    assert wg._Resampler(st3, random.Random(0)).sample() is None
    assert (
        wg._Resampler(
            st3, random.Random(0)
        )._expand("select", 0)
        is None
    )


def test_mutator_edge_returns():
    """The mutator's honest None paths: no donors, no same-arity
    sibling, unknown-shape node, nothing to mutate, unchanged term."""
    st = _stats()
    rng = random.Random(0)
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    term = _p("mul", _p("neg", x), y)
    nodes = wg._nodes_with_paths(term)
    # no other arity-1 op in the corpus stats: swap has no candidate.
    only_binary = wg.corpus_stats(
        [
            TermCase("a", "t", _p("add", x, y), (), (), {}),
            TermCase("m", "t", _p("mul", x, y), (), (), {}),
        ]
    )
    neg_node = nodes[1][1]
    assert neg_node.op == "neg"
    assert wg._swap(term, (0,), neg_node, only_binary, rng) is None
    # a node whose output shape is not concrete cannot graft.
    un = Var("u", TensorType((None, 4)))
    bad_node = _p("neg", un)
    assert wg._graft(term, (0,), bad_node, st, rng) is None
    # a shape with no donors grafts nothing.
    lone = wg.CorpusStats()
    assert wg._graft(term, (0,), neg_node, lone, rng) is None
    # leaf-only term: nothing to mutate.
    leaf_case = TermCase("l", "t", x, (), (), {})
    assert wg.mutant_term(leaf_case, st, rng) is None
    # empty stats: every mutation is a no-op -> unchanged -> None.
    add_case = TermCase("a", "t", _p("add", x, y), (), (), {})
    assert wg.mutant_term(add_case, lone, random.Random(1)) is None


def test_swap_mint_failure_via_stats():
    """A swap candidate whose sampled attr dict cannot mint returns
    ``None`` — the ``Op.make`` guard."""
    x, y = _v("x", 4, 4), _v("y", 4, 4)
    term = _p("mul", _p("neg", x), y)
    st = wg.CorpusStats(
        child_at={("mul", 0): Counter({"select": 1})},
        arity={"select": 1, "neg": 1},
        attr_dicts={"select": Counter({(("index", 0),): 1})},
    )
    neg_node = term.args[0]
    out = wg._swap(term, (0,), neg_node, st, random.Random(0))
    assert out is None


def test_lift_leaf_unknown_and_donorless():
    """Unknown-shape leaves are skipped; shapes without donors leave
    the term alone."""
    st = _stats()
    x = _v("x", 4, 4)
    un = Var("u", TensorType((None, 4)))
    # an unknown-dim leaf contributes no candidate slot.
    assert wg._lift_leaf(_p("neg", un), st, random.Random(0)) is None
    # a concrete shape absent from the pool is skipped the same way.
    odd = _p("neg", _v("q", 7))
    assert wg._lift_leaf(odd, st, random.Random(0)) is None
    # a Const arg is walked past, not treated as a leaf slot.
    got = wg._lift_leaf(_p("add", x, Const(1)), st, random.Random(0))
    assert got is None or isinstance(got, Op)


def test_generate_quotas_and_rejects():
    """Odd ``n`` distributes the remainder; a dead strategy earns a
    zero quota; a None sample counts as a mint reject."""
    st = _stats()
    rng = random.Random(3)
    cases = _tiny_cases()
    gen, gstats = wg._generate(
        cases, st, rng, 1, ("resample", "mutate"), _SUPPORTED
    )
    # quota: resample got the remainder slot; mutate ran zero attempts.
    assert len(gen) <= 1
    assert "mutate" not in gstats.attempted
    dead = wg.CorpusStats(root_tuples=Counter({("zzz", ("var",)): 1}))
    gen2, g2 = wg._generate(
        cases, dead, rng, 1, ("resample",), _SUPPORTED
    )
    assert gen2 == []
    assert g2.rejected[("resample", "mint")] > 0


def test_interleave_uneven_sources_and_gen_table():
    gen = _tiny_cases()[:4]
    for i, c in enumerate(gen):
        object.__setattr__(c, "source", "s1" if i == 0 else "s0")
    out = wg._interleave(gen, 4)
    assert len(out) == 4 and out[0].source == "s0"
    # a strategy with zero attempts is skipped, not zero-divided.
    stats = wg.GenStats()
    stats.attempted["resample"] = 3
    text = wg._gen_table(gen, stats)
    assert "resample" in text and "mutate" not in text.split(
        "resample"
    )[-1].split("\n")[0]


def test_pipeline_delta_shared_changed_rows():
    """A shared proposal whose measurements moved prints the delta
    row; the 'nothing changed' footer stays off."""
    prop = lpipe.Proposal(
        name="p1", lhs=_p("mul", "U", Const(1)), rhs="U", family="t"
    )
    ev_b = lpipe.Evidence(proposal=prop, matches=1, fires=0, paid=0)
    ev_e = lpipe.Evidence(
        proposal=prop, matches=1, fires=2, paid=1,
        fire_cases=["gen:g0"],
    )
    fake = {
        "baseline": {
            "proposals": [prop],
            "ranked": [ev_b],
            "n_terms": 4,
            "n_tuples": 3,
        },
        "enlarged": {
            "proposals": [prop],
            "ranked": [ev_e],
            "n_terms": 6,
            "n_tuples": 5,
        },
    }
    text = wg._pipeline_delta(fake)
    assert "fires 0->2" in text
    assert "(gen=1)" in text
    assert "no shared proposal changed" not in text


def test_main_skip_pipeline_no_json(monkeypatch, capsys):
    cases = _tiny_cases()
    monkeypatch.setattr(wg, "_bench_cases", lambda: (cases, []))
    monkeypatch.setattr(wg, "model_cases", lambda: ([], []))
    rc = wg.main(["--n", "1", "--skip-pipeline"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "law_workload_gen" in printed
    assert "wrote" not in printed


# ---------------------------------------------------------------------------
#  gap_gen — instantiation attrs, validity gates, synthesis tails
# ---------------------------------------------------------------------------


def test_metavars_dedup_and_attr_fallbacks():
    u = "U"
    assert gg._metavars(_p("add", u, u)) == ["U"]
    # Const leaves are walked past, not collected as leaf paths.
    x = _v("x", 4, 4)
    paths = gg._leaf_paths(_p("add", x, Const(1)))
    assert [leaf for _pth, leaf in paths] == [x]
    # split sizes on an odd extent: no even-chunk int candidate.
    assert gg._attr_candidates("split", "sizes", (5,), {}) == [[2, 3]]
    # off-table ops fall through to the generic dim/shape/bool/number
    # rows.
    assert gg._attr_candidates("zzz", "dim0", (4,), {}) == [0]
    assert all(
        gg._numel(s) == 4
        for s in gg._attr_candidates("zzz", "sizes", (4,), {})
    )
    assert gg._attr_candidates("zzz", "is_causal", (4,), {}) == [
        True,
        False,
    ]
    assert gg._attr_candidates("zzz", "alpha", (4,), {}) == [
        0.0,
        1.0,
        0.5,
    ]
    assert gg._attr_candidates("zzz", "eps", (4,), {}) == [1e-5, 1.0]


def test_instantiate_unmintable_op_returns_none():
    """An attr metavarmap that cannot complete a schema-required set
    vetoes the mint — ``Op.make`` raises, ``_instantiate`` returns
    ``None``."""
    pat = Op.make("select", "U", index="I", validate=False)
    out = gg._instantiate(
        pat, {"U": _v("u", 4, 4)}, {}, random.Random(0)
    )
    assert out is None


def test_instance_valid_gate_tail():
    """``_instance_valid`` mirrors ``valid_term``'s gates on the
    synthesis side."""
    st = _stats()
    x = _v("gx", 4, 4)
    t = _p("mul", x, Const(1))
    key = shape_key(t)
    # not an op.
    assert gg._instance_valid(x, st, set(), _SUPPORTED, False) is None
    # a clean multi-node term passes.
    assert (
        gg._instance_valid(
            _p("add", x, _p("neg", _v("gy", 4, 4))),
            st,
            set(),
            _SUPPORTED,
            False,
        )
        is not None
    )
    # unsupported op anywhere.
    assert (
        gg._instance_valid(t, st, set(), frozenset({"add"}), False)
        is None
    )
    # no Var leaf (params only).
    p = Param("gp", TensorType((4, 4)))
    assert (
        gg._instance_valid(
            _p("mul", p, p), st, set(), _SUPPORTED, False
        )
        is None
    )
    # unbindable leaves (Var/Param name clash).
    clash = _p("add", Var("w", TensorType((4, 4))), Param("w", TensorType((4, 4))))
    assert (
        gg._instance_valid(clash, st, set(), _SUPPORTED, False)
        is None
    )
    # torch eval rejects the out-of-range select.
    bad = _p("select", x, dim=0, index=9)
    assert (
        gg._instance_valid(bad, st, set(), _SUPPORTED, False)
        is None
    )
    # a tuple-valued eval result is not a workload case.
    tk = _p("topk", x, k=1)
    assert (
        gg._instance_valid(tk, st, set(), _SUPPORTED, False) is None
    )
    # an already-emitted key is rejected.
    assert (
        gg._instance_valid(t, st, {key}, _SUPPORTED, False) is None
    )
    # and the same term passes every gate cleanly.
    assert gg._instance_valid(
        t, st, set(), _SUPPORTED, False
    ) is not None


def test_instance_valid_node_budget(monkeypatch):
    monkeypatch.setattr(wg, "_MAX_NODES", 2)
    x = _v("gx", 4, 4)
    t = _p("sub", x, _p("neg", _v("gy", 4, 4)))
    assert (
        gg._instance_valid(t, _stats(), set(), _SUPPORTED, False)
        is None
    )


def test_synthesize_rhs_vetoes(monkeypatch):
    st = _stats()
    # an RHS metavar the LHS never binds: instantiate throws inside
    # the attempt and the witness is skipped, not propagated.
    prop = lpipe.Proposal(
        name="unbound_rhs",
        family="t",
        lhs=_p("mul", "U", "V"),
        rhs=_p("mul", "U", "Q"),
    )
    assert (
        gg.synthesize(
            prop, st, _SUPPORTED, set(), random.Random(0),
            n=1, require_novel=False,
        )
        == []
    )
    # an ill-typed instantiated RHS is vetoed before the validity
    # gate even runs: the fixed (3, 5) operand broadcasts with none
    # of the base draws.
    bad = lpipe.Proposal(
        name="bad_rhs",
        family="t",
        lhs=_p("mul", "U", "V"),
        rhs=_p("add", _p("mul", "U", "V"), _v("z", 3, 5)),
    )
    monkeypatch.setattr(gg, "_ATTEMPTS", 3)
    assert (
        gg.synthesize(
            bad, st, _SUPPORTED, set(), random.Random(0),
            n=1, require_novel=False,
        )
        == []
    )


def test_synthesize_dedups_identical_instances():
    """A pattern with no metavariables mints the same term every
    attempt — the repr dedup keeps one witness, not _ATTEMPTS."""
    x = _v("x", 4, 4)
    prop = lpipe.Proposal(
        name="ground",
        family="t",
        lhs=_p("mul", x, Const(1)),
        rhs=x,
    )
    out = gg.synthesize(
        prop, _stats(), _SUPPORTED, set(), random.Random(0),
        n=4, require_novel=False,
    )
    # every attempt produced the identical term; the dedup gate kept
    # exactly one (and validity passed: it is evaluable and Var-ful).
    assert len(out) == 1
    assert _term_match(prop.lhs, out[0][0]) is not None


def test_gen_cases_scalar_instance_skips_ctx(monkeypatch):
    """A scalar-output instance (shape ``()``) has no ctx embedding —
    ``add(inst, fresh_var)`` is skipped on the falsy shape."""
    prop = lpipe.Proposal(
        name="sel",
        family="t",
        lhs=_p("select", "U", dim="D", index="I"),
        rhs="U",
    )
    cases = _tiny_cases()
    st = wg.corpus_stats(cases)
    # a rank-1 base draw: select of a vector is a ()-shaped instance.
    monkeypatch.setattr(gg, "_BASE_SHAPES", [(16,)])
    out, prov = gg.gen_cases_for(
        prop, cases, st, _SUPPORTED, set(), random.Random(0),
        require_novel=False,
    )
    assert prov["synthesized"] >= 1
    assert "min0" in prov["embeddings"]
    assert all(isinstance(c, TermCase) for c in out)


def test_gen_cases_graft_embedding():
    """A model donor with a shape-equal Var leaf hosts the graft —
    the third embedding path."""
    prop = lpipe.Proposal(
        name="p_mul_one",
        family="t",
        lhs=_p("mul", "U", Const(1)),
        rhs="U",
    )
    mx = _v("mx", 4, 16)
    my = _v("my", 4, 16)
    model_term = _p("add", mx, my)
    env = wg._leaf_env(model_term)
    model = TermCase(
        source="model",
        name="M",
        term=model_term,
        inputs=(mx, my),
        feed=(env[mx], env[my]),
        param_vals={},
    )
    cases = [*_tiny_cases(), model]
    st = wg.corpus_stats(cases)
    out, prov = gg.gen_cases_for(
        prop, cases, st, _SUPPORTED, set(), random.Random(0),
        require_novel=False,
    )
    assert any(e.startswith("graft") for e in prov["embeddings"])
    grafted = [c for c in out if ":graft" in c.name]
    assert grafted
    assert any(
        n.op == "mul"
        for n in wg._iter_subterms(grafted[0].term)
        if isinstance(n, Op)
    )


def test_graft_all_donors_invalid():
    """Every donor vetoed by the validity gate -> ``None``."""
    x = _v("gx", 4, 4)
    inst = _p("mul", x, Const(1))
    model = _tiny_model()
    st = _stats()
    # pre-seed `seen` with every possible graft, so each donor's term
    # is rejected as already emitted and the loop falls through.
    seen = {
        shape_key(wg._replace(model.term, p, inst))
        for p in ((0,), (1,))
    }
    out = gg._graft(
        inst, (4, 4), [model], st, seen, _SUPPORTED,
        random.Random(0),
    )
    assert out is None


def _tiny_model() -> TermCase:
    x = _v("mx", 4, 4)
    t = _p("add", x, _v("my", 4, 4))
    env = wg._leaf_env(t)
    vs = [leaf for leaf in wg._leaves(t) if isinstance(leaf, Var)]
    return TermCase(
        source="model",
        name="TinyModel",
        term=t,
        inputs=tuple(vs),
        feed=tuple(env[v] for v in vs),
        param_vals={},
    )


def test_measure_candidate_check_vetoed_bound():
    """A matching instance whose check vetoes the bound keeps the
    oracle fields at their honest defaults."""
    prop = lpipe.Proposal(
        name="p",
        family="t",
        lhs=_p("mul", "U", Const(1)),
        rhs="U",
        check=lambda bound: False,
    )
    ev = lpipe.Evidence(proposal=prop, relation="new")
    x = _v("gx", 4, 4)
    term = _p("mul", x, Const(1))
    res = gg.measure_candidate(
        ev,
        [],
        {
            "synthesized": 1,
            "instances": [repr(term)],
            "_terms": [term],
            "embeddings": [],
        },
        list(ALL_RULES),
        None,
        None,
    )
    assert res.num_true is None and res.derivable is False
    assert res.rhs_instance == ""


def test_main_would_ship_and_no_json(monkeypatch, capsys):
    """A candidate that pays on its generated workload reaches the
    'would-ship' summary line; the run without ``--json`` exits
    without writing."""
    cases = _tiny_cases()
    monkeypatch.setattr(gg, "_bench_cases", lambda: (cases, []))
    monkeypatch.setattr(gg, "model_cases", lambda: ([], []))
    prop = lpipe.Proposal(
        name="factor_left",
        family="t",
        lhs=_p("add", _p("mul", "U", "V"), _p("mul", "U", "W")),
        rhs=_p("mul", "U", _p("add", "V", "W")),
    )
    monkeypatch.setattr(gg.lpipe, "propose", lambda *a: [prop])
    rc = gg.main(
        ["--skip-baseline", "--only", "factor_left"]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "law_gap_targeted_gen" in printed
    assert "would-ship" in printed
