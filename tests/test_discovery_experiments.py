"""Tests for the discovery experiments line — the seeded, small-
budget machinery of ``grammar``, ``meta_game``, ``workload_gen`` and
``gap_gen``.

All four tools are search loops over the same corpus machinery; the
tests here drive every stage on a hand-built six-term corpus (a few
pointwise/view ops, one shared subterm, one ``Param``) rather than
the real model suite.  Nothing is trained and nothing runs the full
pipeline: ``train`` gets two episodes on a five-op vocabulary,
``_generate`` asks for four terms, ``measure_candidate`` sees a
three-case generated set, and ``main`` runs end-to-end with the
corpus-producing functions shrunk — the *machinery* is real, the
universe is small.
"""

import argparse
import json
import random

import pytest
import torch
from catopt_core.egraph.terms import _term_match
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_core.typing import INVALID, _shape_of
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


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


# ---------------------------------------------------------------------------
#  Shared tiny corpus
# ---------------------------------------------------------------------------


def _tiny_terms() -> list:
    """Six small, well-typed terms — the corpus every tool reads."""
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
    """The corpus as ``TermCase``s (all ``bench``-sourced)."""
    cases = []
    for i, t in enumerate(_tiny_terms()):
        env = wg._leaf_env(t)
        assert env is not None
        c = wg.term_to_case(t, f"t{i}", "bench", env)
        assert c is not None
        cases.append(c)
    return cases


def _tiny_model_case() -> TermCase:
    """A source='model' case with a (4,4) Var leaf — a graft donor."""
    x = _v("mx", 4, 4)
    t = _p("add", _p("mul", x, _v("my", 4, 4)), _v("mz", 4, 4))
    env = wg._leaf_env(t)
    assert env is not None
    vs = [leaf for leaf in wg._leaves(t) if isinstance(leaf, Var)]
    return TermCase(
        source="model",
        name="TinyModel",
        term=t,
        inputs=tuple(vs),
        feed=tuple(env[v] for v in vs),
        param_vals={
            p.name: env[p]
            for p in wg._leaves(t)
            if isinstance(p, Param)
        },
    )


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


# ---------------------------------------------------------------------------
#  grammar — shapes as data
# ---------------------------------------------------------------------------


def test_shape_basics():
    s = gm._node("add", gm._slot(0), gm._node("mul", gm._slot(1), gm._lit(2)))
    assert gm._slots(s) == {0, 1}
    assert gm._fresh_slot(s) == 2
    assert gm._fresh_slot(gm._lit(0)) == 0
    assert gm._count_slot(s, 1) == 1
    shared = gm._node("sub", gm._slot(0), gm._slot(0))
    assert gm._count_slot(shared, 0) == 2
    positions = dict(gm._positions(s))
    assert positions[(1,)].op == "mul"
    assert positions[(1, 0)] == gm._slot(1)
    rep = gm._replace(s, (1, 0), gm._lit(0))
    assert rep.children[1].children[0] == gm._lit(0)
    assert gm._replace(s, (), gm._lit(1)) == gm._lit(1)
    rel = gm._relabel(shared, 0, 1)
    assert rel.children[0] == gm._slot(1)


def test_arity_ops_and_canon():
    assert gm._arity_ops(1) == gm._UNARY_OPS
    assert gm._arity_ops(2) == gm._BINARY_OPS
    assert gm._arity_ops(3) == ()
    # slot renaming by first use makes keys alpha-normal.
    a = gm._node("add", gm._slot(1), gm._slot(0))
    b = gm._node("add", gm._slot(0), gm._slot(1))
    assert gm._schema_key(a, gm._slot(0)) == gm._schema_key(
        b, gm._slot(1)
    )
    assert gm._canon(gm._lit(2), {}) == ("c", 2)


def test_shape_of_term_and_instantiate():
    x, y = _v("x", 2), _v("y", 2)
    slots: dict[str, int] = {}
    lhs = gm.shape_of_term(_p("mul", x, y), slots)
    rhs = gm.shape_of_term(_p("mul", y, x), slots)
    # x and y share slots across the two sides by repr.
    assert gm._slots(lhs) == gm._slots(rhs) == {0, 1}
    env = {0: x, 1: y}
    assert gm._instantiate(lhs, env) == _p("mul", x, y)
    with pytest.raises(ValueError, match="neither slot nor const"):
        gm._instantiate(gm.Shape(), env)
    with pytest.raises(ValueError, match="neither slot nor const"):
        gm._to_pattern(gm.Shape())
    assert gm._to_pattern(lhs) == _p("mul", "s0", "s1")
    assert gm._to_pattern(gm._lit(2)) == Const(2)
    # Const leaves stay literal.
    s = gm.shape_of_term(_p("mul", x, Const(2)), {})
    assert s.children[1] == gm.Shape(const=2)


def test_alpha_coherent_key():
    x, y, a, b = (_v(n, 4, 4) for n in "xyab")
    k1 = gm._alpha_coherent_key(_p("add", x, y))
    # alpha-renaming and commutation share a key.
    assert k1 == gm._alpha_coherent_key(_p("add", a, b))
    assert k1 == gm._alpha_coherent_key(_p("add", y, x))
    assert k1 != gm._alpha_coherent_key(_p("mul", x, y))
    # a leafless term still gets a key.
    assert gm._alpha_coherent_key(_p("add", Const(0), Const(0)))


def test_instantiate_schema():
    lhs = gm._node("mul", gm._slot(0), gm._lit(1))
    tl, tr = gm._instantiate_schema(lhs, gm._slot(0))
    assert tl == _p("mul", _v("x0", 4, 4), Const(1))
    assert tr == _v("x0", 4, 4)


def test_mutations_cover_all_operators():
    s = gm._node("add", gm._slot(0), gm._slot(1))
    muts = list(gm._mutations(s))
    # operand swap
    assert gm._node("add", gm._slot(1), gm._slot(0)) in muts
    # same-arity op swap
    assert gm._node("mul", gm._slot(0), gm._slot(1)) in muts
    # leaf wrap / literal substitution / slot merge
    assert gm._node("add", gm._node("neg", gm._slot(0)), gm._slot(1)) in muts
    assert gm._node("add", gm._lit(0), gm._slot(1)) in muts
    assert gm._node("add", gm._slot(0), gm._slot(0)) in muts
    # a repeated slot can be split.
    shared = gm._node("mul", gm._slot(0), gm._slot(0))
    assert gm._node("mul", gm._slot(1), gm._slot(0)) in list(
        gm._mutations(shared)
    )
    # unary unwrap, and a literal promotes to a fresh slot.
    un = gm._node("neg", gm._slot(0))
    assert gm._slot(0) in list(gm._mutations(un))
    lit = gm._node("mul", gm._lit(0), gm._slot(0))
    assert gm._node("mul", gm._slot(1), gm._slot(0)) in list(
        gm._mutations(lit)
    )


def test_wellformed_and_neighbors():
    lhs = gm._node("add", gm._slot(0), gm._slot(1))
    assert not gm._wellformed(lhs, gm._slot(2))
    assert gm._wellformed(lhs, gm._slot(0))
    for nl, nr in gm._schema_neighbors(lhs, gm._slot(0)):
        assert gm._slots(nr) <= gm._slots(nl)
    # recombination keeps only well-formed crosses.
    schemas = [
        (gm._node("add", gm._slot(0), gm._slot(1)), gm._slot(0)),
        (gm._slot(2), gm._lit(0)),
    ]
    crosses = list(gm._crosses(schemas, 10))
    assert all(
        gm._slots(rs) <= gm._slots(ls) for ls, rs in crosses
    )
    assert crosses
    # the cap is honoured.
    assert len(list(gm._crosses(schemas, 1))) == 1


def test_search_bounded_bfs():
    seed = (gm._node("mul", gm._slot(0), gm._lit(1)), gm._slot(0))
    out = gm.search([seed], depth=1, level_cap=5, total_cap=50)
    assert out[0] == seed
    assert 1 < len(out) <= 6
    # deterministic: same seeds, same order.
    assert gm.search([seed], depth=1, level_cap=5, total_cap=50) == out
    # total_cap is checked between levels, so one level may carry
    # the result past it — the bound is seeds + one full level.
    capped = gm.search([seed], depth=3, level_cap=8, total_cap=4)
    assert len(capped) <= 1 + 8
    assert len(gm.search([seed], depth=0, level_cap=8, total_cap=4)) == 1


def test_human_schemas_and_grammar_keys():
    human = gm._human_schemas()
    assert len(human) == 23
    assert all(
        isinstance(lft, gm.Shape) and isinstance(rgt, gm.Shape)
        for lft, rgt in human
    )
    # 23 candidates collapse to 18 coherent equalities.
    assert len(gm._grammar_keys()) == 18


def test_grammar_closure_budgets_everything():
    rules, budgets = gm._grammar_closure()
    assert len(rules) == len(ALL_RULES) + 23
    assert len(budgets) == len(rules)
    assert any(r.name.startswith("grammar:") for r in rules)


def test_evaluate_schemas_real_oracles():
    """``mul(x,1) -> x`` is derivable — a known truth, measured by
    the same oracles the pipeline uses."""
    schema = (
        gm._node("mul", gm._slot(0), gm._lit(1)),
        gm._slot(0),
    )
    results = gm.evaluate_schemas([schema])
    assert len(results) == 1
    r = results[0]
    assert r.outcome.derivable or r.outcome.num_true is True
    assert "mul" in r.lhs_repr


def test_counts_and_report():
    schema = (gm._node("mul", gm._slot(0), gm._lit(1)), gm._slot(0))
    results = gm.evaluate_schemas([schema])
    counts = gm._counts(results)
    assert counts["evaluated"] == 1
    assert gm._fmt_ratio(1, 2) == "0.500"
    assert gm._fmt_ratio(0, 0) == "0.000"
    args = argparse.Namespace(depth=1, level_cap=5, total_cap=10)
    human = {
        "evaluated": 1,
        "distinct": 1,
        "useful_new": 0,
    }
    searched = {**counts}
    text = gm._fmt_report(human, searched, args)
    assert "schema-level search" in text
    assert "useful per verification" in text
    # no novel laws -> the honest empty section.
    if not searched["novel"]:
        assert "none" in text


def test_dump_json_and_main(tmp_path, capsys, monkeypatch):
    seed = (gm._node("mul", gm._slot(0), gm._lit(1)), gm._slot(0))
    monkeypatch.setattr(gm, "_human_schemas", lambda: [seed])
    out = tmp_path / "g.json"
    rc = gm.main(
        [
            "--depth",
            "1",
            "--level-cap",
            "4",
            "--total-cap",
            "8",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["human"]["evaluated"] == 1
    assert payload["search"]["evaluated"] >= 1
    printed = capsys.readouterr().out
    assert "schema-level search" in printed


# ---------------------------------------------------------------------------
#  meta_game — slots and the board
# ---------------------------------------------------------------------------


def test_slot_helpers():
    root = mg._Slot()
    assert mg._first_hole(root) == ()
    mg._fill(
        root,
        (),
        mg._Slot(kind="op", op="add", children=[mg._Slot(), mg._Slot()]),
    )
    assert mg._first_hole(root) == (0,)
    assert mg._n_ops(root) == 1
    assert mg._n_holes(root) == 2
    mg._fill(root, (0,), mg._Slot(kind="mv", name="U"))
    assert mg._node_at(root, (0,)).name == "U"
    mg._fill(root, (1,), mg._Slot(kind="const", value=1))
    assert mg._first_hole(root) is None
    assert mg._ops_present(root) == {"add"}
    term = mg._mint(root)
    assert term == _p("add", "U", Const(1))
    # a hole mints to the safety-net metavar name.
    assert mg._mint(mg._Slot()) == "Z"
    # attr metavariables are collected off minted patterns.
    sel = _p("select", "U", dim="select.dim", index="select.index")
    assert mg._attr_mv_names(sel) == {"select.dim", "select.index"}


def test_bool_attr_check_and_attr_bridge():
    assert mg._bool_attr_check({"$attr:k": True})
    assert not mg._bool_attr_check({"$attr:k": False})
    assert mg._bool_attr_check({"$attr:k": 3})
    # the bridge resolves <op>.<key> names off bound same-key attrs
    # and unwraps singleton tuples; no candidate -> veto.
    d = mg._attr_bridge(frozenset({"softmax.dim"}))
    out = d({"$attr:sum.dim": (-1,)})
    assert out == {"$attr:softmax.dim": -1}
    assert d({"$attr:other.index": 0}) is None


def _vocab():
    terms = _tiny_terms()
    cts = [CorpusTerm("t", f"m{i}", t) for i, t in enumerate(terms)]
    counts, _ = op_tuple_census(cts)
    sh, _ = shape_census(cts)
    return mg.build_vocab(terms, counts, sh, 6)


def test_seed_slot_and_shape_slot():
    vocab = _vocab()
    s = mg._seed_slot(
        "mul", ("silu", "·"), vocab.arity, vocab.attr_keys,
        frozenset(vocab.ops),
    )
    assert s is not None and s.op == "mul"
    assert mg._seed_name(s) == "seed:mul(silu,·)"
    assert mg._seed_name(None) == ""
    # out-of-vocab op or arity mismatch -> None.
    assert (
        mg._seed_slot(
            "matmul", ("x",), vocab.arity, vocab.attr_keys,
            frozenset(vocab.ops),
        )
        is None
    )
    assert (
        mg._seed_slot(
            "add", ("x",), vocab.arity, vocab.attr_keys,
            frozenset(vocab.ops),
        )
        is None
    )
    # shape keys: leaf / const / op, and vetoes.
    assert mg._shape_slot(("leaf", 0), {}, frozenset()) == mg._Slot()
    c = mg._shape_slot(("const", 1), {}, frozenset())
    assert c.kind == "const" and c.value == 1
    key = ("add", (), (("leaf", 0), ("const", 1)))
    s = mg._shape_slot(key, vocab.attr_keys, frozenset(vocab.ops))
    assert s is not None and s.op == "add"
    bad = ("matmul", (), (("leaf", 0),))
    assert mg._shape_slot(bad, vocab.attr_keys, frozenset(vocab.ops)) is None
    assert mg._shape_slot(("zzz", (), (("leaf", 0),)), {}, frozenset(vocab.ops)) is None
    # deeper than the op budget -> None.
    deep = key
    for _ in range(mg._MAX_OPS_SIDE):
        deep = ("add", (), (deep, ("const", 1)))
    assert (
        mg._shape_slot(deep, vocab.attr_keys, frozenset(vocab.ops))
        is None
    )
    # clone deep-copies a skeleton.
    cl = mg._clone(s)
    assert cl == s and cl is not s
    assert mg._clone(None) is None


def test_share_scan_counts_shared_subterms():
    x = _v("x", 4, 4)
    # sub(x, x) shares a subterm under the sub node.
    terms = [_p("sub", x, x), _p("add", x, _v("y", 4, 4))]
    share = mg._share_scan(terms)
    assert share[("sub", ("·", "·"))] >= 1
    assert ("add", ("·", "·")) not in share


def test_build_vocab_tables():
    vocab = _vocab()
    assert "mul" in vocab.ops
    assert vocab.arity["add"] == 2
    assert vocab.attr_keys["select"] == ("dim", "index")
    assert vocab.classes["mul"] == "pointwise-bin"
    assert vocab.classes["select"] == "view"
    assert vocab.tuple_count[("add", ("mul", "·"))]
    assert vocab.head_count["mul"] >= 1
    assert vocab.child_count[("mul", 0, "silu")] >= 1
    assert vocab.pos_count[("mul", 0)] >= 1
    assert vocab.max_count >= 1
    assert vocab.seeds


def test_game_legalities():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    # at the LHS root: only census-attested head ops, all metavars,
    # no consts (a const matches only where the corpus puts one).
    legal = [game.actions[i] for i in game.legal()]
    assert ("op", "mul") in legal
    assert ("mv", "U") in legal
    assert not any(k == "const" for k, _ in legal)
    # step an op, then a metavar in the first hole.
    idx = game.actions.index(("op", "mul"))
    game.step(idx)
    assert game.side == "lhs"
    legal = [game.actions[i] for i in game.legal()]
    # census-attested children of mul at position 0 are legal.
    assert ("op", "silu") in legal or ("op", "sigmoid") in legal
    game.step(game.actions.index(("mv", "U")))
    game.step(game.actions.index(("mv", "V")))
    assert game.bound == {"U", "V"}
    assert game.side == "rhs"
    # on the RHS only bound metavars are legal.
    legal = [game.actions[i] for i in game.legal()]
    assert ("mv", "U") in legal and ("mv", "V") in legal
    assert ("mv", "W") not in legal
    game.step(game.actions.index(("mv", "U")))
    assert game.done
    cand = game.candidate()
    assert cand is not None
    lhs, rhs, check, _ = cand
    assert lhs == _p("mul", "U", "V")
    assert rhs == "U"
    assert check is mg._bool_attr_check


def test_game_seeded_reset_and_candidate_derive():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(1))
    assert vocab.seeds
    seed = vocab.seeds[0][0]
    game.reset(mg._clone(seed), "seeded")
    # the select skeleton still has holes on the LHS.
    assert game.side == "lhs"
    while game.side == "lhs" and not game.done:
        game.step(game.legal()[0])
    assert game.side == "rhs"
    # mint an RHS reusing the bound metavar — the attr bridge derive
    # fires when RHS attr metavars are unbound.
    while not game.done:
        game.step(game.legal()[0])
    cand = game.candidate()
    assert cand is not None


def test_shared_ok_gate():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    game.step(game.actions.index(("op", "mul")))
    game.step(game.actions.index(("mv", "U")))
    # re-placing U under the same mul node requires the corpus to
    # attest shared (·,·) children under mul — the tiny corpus has
    # none, so the reuse is gated off.
    ok = game._shared_ok((1,), (0,))
    lca = mg._node_at(game.lhs, ())
    assert lca.op == "mul"
    assert ok == (vocab.share.get(("mul", ("·", "·")), 0) > 0)
    assert ok is False


def test_autofill_on_step_cap(monkeypatch):
    monkeypatch.setattr(mg, "_MAX_STEPS", 4)
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    for _ in range(4):
        if not game.done:
            legal = game.legal()
            # prefer op actions to keep holes open.
            op_idx = next(
                (i for i in legal if game.actions[i][0] == "op"),
                legal[0],
            )
            game.step(op_idx)
    assert game.done
    # forced end filled every remaining hole with metavariables.
    assert mg._first_hole(game.lhs) is None
    assert mg._first_hole(game.rhs) is None


def test_state_action_prior_vectors():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    sv = game.state_vec()
    assert len(sv) == 9
    legal = game.legal()
    hole = mg._first_hole(game._cur())
    for i in legal[:3]:
        av = game.action_vec(i, hole)
        assert len(av) == 12
        assert isinstance(game.prior_logit(i, hole), float)
    # an mv action at the LHS root has the fresh-metavar prior.
    mv_i = game.actions.index(("mv", "U"))
    assert mg._first_hole(game.lhs) == ()
    prior = game.prior_logit(mv_i, ())
    assert prior >= mg._PRIOR_FRESH_MV


def test_policy_net_and_tensors():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    legal = game.legal()
    sv, av, _ = mg._tensors(game, legal)
    net = mg._PolicyNet(sv.shape[0], av.shape[1], hidden=8)
    logits = net(sv, av)
    assert logits.shape == (len(legal),)


# ---------------------------------------------------------------------------
#  meta_game — the referee
# ---------------------------------------------------------------------------


def _referee(terms=None, cases=(), lib=None):
    return mg.Referee(
        _tiny_terms() if terms is None else terms,
        list(cases),
        None,
        None,
        [] if lib is None else lib,
    )


def test_referee_verdict_ladder():
    r = _referee()
    # tautology — same key both sides, no oracle call.
    v = r.evaluate(_p("mul", "U", "V"), _p("mul", "U", "V"))
    assert v.reason == "tautology"
    assert r.oracle_calls == 0
    # no-instance — the pattern matches nothing in the corpus.
    v = r.evaluate(
        _p("div", _p("pow", "U", Const(3)), "V"), _p("div", "U", "V")
    )
    assert v.reason == "no-instance"
    assert r.oracle_calls == 0
    # false — instantiates but the oracle rejects.
    v = r.evaluate(_p("mul", "U", "V"), _p("add", "U", "V"))
    assert v.reason == "false"
    assert r.oracle_calls == 1
    # repeat — the swapped equality is the same candidate.
    v2 = r.evaluate(_p("add", "U", "V"), _p("mul", "U", "V"))
    assert v2.reason.startswith("repeat")
    assert r.oracle_calls == 1
    # true — x*1 = x instantiates on the mul(x, Const(1)) corpus term
    # and the oracle confirms it; no cases -> no fires.
    v = r.evaluate(_p("mul", "U", Const(1)), "U")
    assert v.reason == "true"
    assert v.truth and v.fires == 0
    assert r.oracle_calls == 2
    s = r.summary(plays=3)
    assert s["candidates"] == 3
    assert s["tautology"] >= 1
    assert s["true"] == 1


def test_referee_check_and_derive_hooks():
    r = _referee()
    # a check veto skips the match entirely -> no-instance.  (Each
    # case uses a distinct candidate so dedup does not fold them.)
    v = r.evaluate(
        _p("mul", "U", Const(1)),
        "U",
        check=lambda bound: False,
    )
    assert v.reason == "no-instance"
    # a raising check is a veto too.
    v = r.evaluate(
        _p("mul", "U", Const(0)),
        "U",
        check=lambda bound: 1 / 0,
    )
    assert v.reason == "no-instance"
    # a derive that returns None vetoes; one that raises does too.
    # (Distinct ops — the dedup key is alpha-normal, so a renamed
    # metavar is the same candidate.)
    v = r.evaluate(
        _p("add", "U", "V"),
        _p("add", "V", "W"),
        derive=lambda bound: None,
    )
    assert v.reason == "no-instance"
    v = r.evaluate(
        _p("sub", "U", "V"),
        _p("sub", "V", "X"),
        derive=lambda bound: 1 / 0,
    )
    assert v.reason == "no-instance"
    # a derive supplying a needed binding lets the instance through
    # (the lhs must still match a corpus term).
    v = r.evaluate(
        _p("mul", "U", Const(1)),
        _p("add", "U", "U"),
        derive=lambda bound: {"extra": 1},
    )
    assert v.instance


def test_referee_fires_and_pays():
    """The lone rule ``mul(u,1)->u`` fires on the corpus case and
    strictly lowers the extracted cost — the full referee path."""
    case = _tiny_cases()[3]  # mul(x, Const(1))
    sink = _sink()
    r = mg.Referee(
        [case.term],
        [case],
        sink,
        _cost_fn(sink),
        [],
    )
    v = r.evaluate(_p("mul", "U", Const(1)), "U")
    assert v.truth
    assert v.fires >= 1
    assert "fwd" in v.directions
    assert v.paid >= 1
    assert v.score > 1.0


def test_referee_unfireable_orientation():
    """A rule whose check rejects the e-graph's binding records an
    honest fire error, not a crash."""
    case = _tiny_cases()[3]
    sink = _sink()
    r = mg.Referee([case.term], [case], sink, _cost_fn(sink), [])
    calls = {"n": 0}

    def boom(case, rule, sink, cost_fn):
        calls["n"] += 1
        raise RuntimeError("probe blew up")

    import catopt_discovery.meta_game as _mg

    orig = _mg._probe
    _mg._probe = boom
    try:
        v = r.evaluate(_p("mul", "U", Const(1)), "U")
    finally:
        _mg._probe = orig
    assert v.truth
    assert v.fire_errors == 2  # both orientations blew up
    assert v.fires == 0


def test_referee_probe_verify_fail(monkeypatch):
    """A probe reporting a FAIL verification feeds the counter."""
    case = _tiny_cases()[3]
    sink = _sink()
    r = mg.Referee([case.term], [case], sink, _cost_fn(sink), [])

    class _FakeProbe:
        fires = 1
        changed = False
        paid = False
        verified = "FAIL"
        note = ""
        base_cost = 0.0
        added_cost = 0.0
        cert = ""
        closure = 1.0

    monkeypatch.setattr(mg, "_probe", lambda *a: _FakeProbe())
    v = r.evaluate(_p("mul", "U", Const(1)), "U")
    assert v.verify_fail == 2  # once per orientation
    # fires is the better of the two orientations, not the sum.
    assert v.fires == 1
    assert v.directions == ("fwd", "bwd")


def test_tensors_without_hole_raises():
    vocab = _vocab()
    game = mg.BuildGame(vocab, random.Random(0))
    game.reset(None)
    game.step(game.actions.index(("mv", "U")))
    game.step(game.actions.index(("mv", "U")))
    assert game.done
    # featurizing a completed term — no hole anywhere — is a
    # caller error.
    with pytest.raises(RuntimeError, match="no hole"):
        mg._tensors(game, game.legal())


def test_referee_summary_fields():
    r = _referee()
    r.evaluate(_p("mul", "U", Const(1)), "U")
    s = r.summary()
    for k in (
        "oracle_calls",
        "no_instance",
        "false",
        "unknown",
        "true_firing",
        "yield_true_per_call",
        "yield_tf_per_call",
        "yield_ship_per_call",
    ):
        assert k in s
    assert s["yield_true_per_call"] <= 1.0


# ---------------------------------------------------------------------------
#  meta_game — the player loops
# ---------------------------------------------------------------------------


def test_play_once_uniform():
    vocab = _vocab()
    r = _referee(cases=())
    game = mg.BuildGame(vocab, random.Random(3))
    game.reset(None)
    v, logps = mg._play_once(game, r, None, game.rng, prior=False)
    assert v.reason or v.name is not None
    assert logps == []  # no model -> no log-probs


def test_train_two_episodes():
    vocab = _vocab()
    r = _referee(cases=())
    game = mg.BuildGame(vocab, random.Random(0))
    model, hist = mg.train(
        game, r, 2, hidden=8, seed_frac=0.0, log_every=0
    )
    assert len(hist) == 2
    assert isinstance(model, torch.nn.Module)
    assert r.oracle_calls <= 2


def test_run_player_budget():
    vocab = _vocab()
    r = _referee(cases=())
    game = mg.BuildGame(vocab, random.Random(0))
    s = mg.run_player(game, r, None, budget=0, max_plays=5)
    assert s["plays"] == 0
    s = mg.run_player(game, r, None, budget=4, max_plays=3)
    assert s["plays"] <= 3


def test_eval_baseline_with_instance():
    r = _referee(cases=())
    x = _v("x", 4, 4)
    fake = argparse.Namespace(
        lhs=_p("mul", "U", Const(1)),
        rhs="U",
        check=None,
        derive=None,
        name="fake",
        instance=(_p("mul", x, Const(1)), x),
    )
    s = mg.eval_baseline([fake], r, budget=10)
    assert s["candidates"] == 1
    assert r.oracle_calls == 1


def test_driver_helpers(capsys):
    cases = _tiny_cases()
    model = _tiny_model_case()
    # the slice is a fixed name-list: the wanted model name and one
    # wanted bench name make the cut; the rest do not.
    object.__setattr__(model, "name", "SelectiveSSM")
    object.__setattr__(cases[3], "name", "select_mul")
    sliced = mg._slice_cases(cases, [model])
    assert [c.name for c in sliced] == ["SelectiveSSM", "select_mul"]
    rules = mg._search_rules("comm_add")
    assert all(r.name != "comm_add" for r in rules)
    assert len(mg._search_rules(None)) == len(ALL_RULES)
    targets = mg._targets()
    assert "softmax_fold" in targets
    r = _referee()
    red = mg._rediscovered(r, targets)
    assert set(red) == set(targets)
    assert not any(red.values())
    assert mg._reason_table(r) == {}
    mg._print_yield("t", r.summary())
    mg._print_top(r, 5)
    out = capsys.readouterr().out
    assert "yield/call" in out or "no scored" in out
    # with real verdicts on record the printers walk their rows.
    r2 = _referee()
    r2.evaluate(_p("mul", "U", Const(1)), "U")
    r2.evaluate(_p("mul", "U", "V"), _p("add", "U", "V"))
    tbl = mg._reason_table(r2)
    assert sum(tbl.values()) == 2
    mg._print_top(r2, 5)
    out2 = capsys.readouterr().out
    assert "fires=" in out2 or "(no scored candidates)" in out2


def test_run_experiment_tiny(monkeypatch, tmp_path, capsys):
    """``run_experiment`` + ``main`` end to end on the tiny corpus:
    two training episodes, zero oracle budget elsewhere."""
    cases = _tiny_cases()
    monkeypatch.setattr(mg, "_bench_cases", lambda: (cases[:4], []))
    monkeypatch.setattr(mg, "model_cases", lambda: (cases[4:], []))
    out = tmp_path / "mg.json"
    rc = mg.main(
        [
            "--episodes",
            "2",
            "--budget",
            "0",
            "--random-calls",
            "0",
            "--plays-cap",
            "3",
            "--seed-frac",
            "0",
            "--top-seeds",
            "3",
            "--hidden",
            "8",
            "--log-every",
            "0",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    for k in (
        "baseline",
        "random",
        "player",
        "train_oracle",
        "player_rediscovered",
    ):
        assert k in payload
    printed = capsys.readouterr().out
    assert "yield per oracle call" in printed
    assert "where the search stalls" in printed
    # no --json: the same run prints and returns without writing.
    rc = mg.main(
        [
            "--episodes",
            "0",
            "--budget",
            "0",
            "--random-calls",
            "0",
            "--plays-cap",
            "1",
            "--seed-frac",
            "0",
            "--log-every",
            "0",
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
#  workload_gen — stats, gates, strategies
# ---------------------------------------------------------------------------


def _stats():
    return wg.corpus_stats(_tiny_cases())


def test_marker_and_concrete_shape():
    x = _v("x", 2)
    assert wg._marker(_p("add", x, x)) == "add"
    assert wg._marker(Const(1)) == "const"
    assert wg._marker(Param("p", TensorType((2,)))) == "param"
    assert wg._marker(x) == "var"
    assert wg._marker("leaf") == "leaf"
    assert wg._concrete_shape(_p("add", x, x)) == (2,)
    assert wg._concrete_shape(x) == (2,)
    assert wg._concrete_shape(Var("u", TensorType((None,)))) is None
    assert wg._concrete_shape(Const(1)) is None


def test_corpus_stats_tables():
    st = _stats()
    assert st.root_tuples
    assert st.typed_tuples
    assert st.tuples_of["mul"]
    assert st.attr_dicts["select"]
    assert st.const_vals[1] >= 1
    assert st.leaf_at[("mul", 0)]
    assert st.var_shapes and st.param_shapes
    assert st.pool_by_shape[(4, 4)]
    assert st.root_keys and st.sub_keys
    assert st.arity["select"] == 1


def test_corpus_stats_list_attr():
    """A term carrying a list-valued attr counts through
    ``census._attr_key``'s repr stand-in — the crash-on-hash defect —
    and resampling reads the observed dict back verbatim."""
    term = _p("relu", _p("pad", _v("lv", 2, 3), pad=[1, 1]))
    st = wg.corpus_stats([TermCase("bench", "padcase", term, (), (), {})])
    assert len(st.attr_dicts["pad"]) == 1
    key = next(iter(st.attr_dicts["pad"]))
    assert st.attr_dicts["pad"][key] == 1
    # the repr stand-in ("[1, 1]") is not a value — the exemplar
    # table returns the real list, not the string.
    sampled = wg._sample_attrs(st, "pad", random.Random(0))
    assert sampled == {"pad": [1, 1]}
    assert isinstance(sampled["pad"], list)
    # the resampler's own consumer path agrees.
    sampler = wg._Resampler(st, random.Random(0))
    assert sampler._attrs("pad") == {"pad": [1, 1]}


def test_conv_nonpositive_stride_invalid():
    """A conv with a non-positive stride can never lower — torch
    refuses it at runtime — so ``_shape_of`` reports ``INVALID``
    (the provably-ill-typed verdict) instead of dividing by zero."""
    x1 = _v("cx", 1, 1, 10)
    w1 = Param("cw", TensorType((1, 1, 3)))
    bad1 = _p("conv1d", x1, w1, stride=0)
    assert _shape_of(bad1) is INVALID
    # the verdict poisons enclosing ops — it cannot wash out as
    # "unknown" under a broadcast parent.
    assert _shape_of(_p("add", bad1, _v("cy", 1, 1, 8))) is INVALID
    x2 = _v("dx", 1, 1, 8, 8)
    w2 = Param("dw", TensorType((1, 1, 3, 3)))
    assert _shape_of(_p("conv2d", x2, w2, stride=0)) is INVALID
    assert _shape_of(_p("conv2d", x2, w2, stride=[0, 1])) is INVALID
    assert _shape_of(_p("conv2d", x2, w2, stride=(1, -2))) is INVALID
    # non-positive dilation is the same class.
    assert _shape_of(_p("conv1d", x1, w1, dilation=0)) is INVALID
    # a metavar stride is merely unknown — the degraded return.
    assert _shape_of(_p("conv1d", x1, w1, stride="s")) == (1, 1, None)
    # a wrong-length conv2d window tuple declines the same way.
    assert _shape_of(_p("conv2d", x2, w2, stride=[2])) == (
        1,
        1,
        None,
        None,
    )
    # an unknown operand extent with concrete attrs: unknown, not
    # ill-typed — the divide never runs.
    ux = Var("ux", TensorType((1, 1, None)))
    uw = Param("uw", TensorType((1, 1, 3)))
    assert _shape_of(_p("conv1d", ux, uw, stride=2)) == (1, 1, None)
    # and well-formed convs still shape.
    assert _shape_of(_p("conv1d", x1, w1, stride=2)) == (1, 1, 4)
    assert _shape_of(_p("conv1d", x1, w1, stride=[2])) == (1, 1, 4)
    assert _shape_of(_p("conv2d", x2, w2, stride=(2, 2))) == (
        1,
        1,
        3,
        3,
    )
    # the generation gate declines the term, never mints it.
    assert (
        wg.valid_term(bad1, _stats(), set(), _SUPPORTED | {"conv1d"})
        is None
    )


def test_chunk_split_nonpositive_counts_invalid():
    """``chunks=0`` / ``sections=0`` divide by zero — the same class
    as the conv zero-stride defect: decline ``INVALID``, don't crash.
    A non-int count (``2.0``) declines to ``None`` rather than mint a
    float dimension."""
    x = _v("x", 8)
    assert _shape_of(_p("chunk", x, chunks=0)) is INVALID
    assert _shape_of(_p("tensor_split", x, sections=0, index=0)) is INVALID
    assert (
        _shape_of(_p("tensor_split", x, sections=-2, index=0))
        is INVALID
    )
    assert _shape_of(_p("chunk", x, chunks=2.0)) is None
    # valid spellings unchanged.
    assert _shape_of(_p("chunk", x, chunks=2)) == (4,)
    assert _shape_of(_p("tensor_split", x, sections=2, index=0)) == (4,)
    # an index-list sections payload keeps the bounds path.
    assert _shape_of(
        _p("tensor_split", x, sections=[3, 5], index=1)
    ) == (2,)


def test_weighted_and_leaves():
    rng = random.Random(0)
    from collections import Counter

    c = Counter({"a": 3, "b": 1})
    draws = {wg._weighted(rng, c) for _ in range(20)}
    assert draws <= {"a", "b"}
    x = _v("x", 2)
    t = _p("add", x, _p("mul", x, Const(1)))
    assert wg._leaves(t) == {x}


def test_leaf_env_consistency():
    a = Var("a", TensorType((2,)))
    b = Var("a", TensorType((3,)))
    # two same-named vars with different shapes cannot be bound.
    assert wg._leaf_env(_p("add", a, b)) is None
    x = _v("x", 2)
    assert wg._leaf_env(_p("add", x, x)) is not None
    # unknown dims veto.
    assert (
        wg._leaf_env(_p("add", Var("u", TensorType((None,))), x))
        is None
    )


def test_valid_term_gates():
    st = _stats()
    seen: set = set()
    x = _v("gx", 4, 4)
    # not an op.
    assert wg.valid_term(x, st, seen, _SUPPORTED) is None
    # unsupported op.
    assert (
        wg.valid_term(
            _p("frobnicate", x), st, seen, _SUPPORTED
        )
        is None
    )
    # no Var leaf.
    p = Param("gp", TensorType((4, 4)))
    assert (
        wg.valid_term(_p("mul", p, p), st, seen, _SUPPORTED)
        is None
    )
    # a corpus subterm is not novel.
    assert (
        wg.valid_term(
            _tiny_terms()[3], st, seen, _SUPPORTED
        )
        is None
    )
    # a fresh, well-typed, evaluable term passes and returns an env.
    fresh = _p("sub", x, _p("neg", _v("gy", 4, 4)))
    env = wg.valid_term(fresh, st, seen, _SUPPORTED)
    assert env is not None
    # the same term again is a seen duplicate.
    assert wg.valid_term(fresh, st, {shape_key(fresh)}, _SUPPORTED) is None
    # terms the torch eval rejects (unbroadcastable add) fail the
    # eval gate.
    bad = _p("add", x, _v("gz", 3))
    assert wg.valid_term(bad, st, seen, _SUPPORTED) is None


def test_valid_term_node_budget(monkeypatch):
    monkeypatch.setattr(wg, "_MAX_NODES", 2)
    st = _stats()
    x = _v("gx", 4, 4)
    t = _p("sub", x, _p("neg", _v("gy", 4, 4)))
    assert wg.valid_term(t, st, set(), _SUPPORTED) is None


def test_resampler_and_mutant():
    st = _stats()
    rng = random.Random(7)
    sampler = wg._Resampler(st, rng)
    made = [sampler.sample() for _ in range(12)]
    assert any(t is not None for t in made)
    for t in made:
        if t is not None:
            assert isinstance(t, Op)
    assert wg.resampled_term(st, random.Random(1)) is not None
    case = _tiny_cases()[2]
    mut = wg.mutant_term(case, st, random.Random(5))
    # mutation is stochastic; over seeds some mutant survives.
    if mut is None:
        mut = wg.mutant_term(case, st, random.Random(11))
    assert mut is not None
    assert mut is not case.term


def test_swap_graft_lift_directly():
    st = _stats()
    rng = random.Random(0)
    case = _tiny_cases()[0]
    term = case.term
    nodes = wg._nodes_with_paths(term)
    assert nodes[0][0] == ()
    assert wg._parent_slot(term, (0,)) == (term.op, 0)
    assert wg._parent_slot(term, ()) == ("<root>", 0)
    # swap an op node for a same-arity sibling (or None honestly).
    new = wg._swap(term, (0,), nodes[1][1], st, rng)
    if new is not None:
        assert new is not term
    # graft a shape-equal donor into a node slot.
    g = wg._graft(term, (0,), nodes[1][1], st, rng)
    if g is not None:
        assert g is not term
    lifted = wg._lift_leaf(term, st, rng)
    if lifted is not None:
        assert lifted is not term
    # replace rebuilds with the sibling subterm unchanged.  The
    # untouched slot keeps its *value*; object identity is not a
    # contract — ``Op.make`` interning is structural, so a term
    # minted earlier in the process returns the interned object
    # whose children are equal-but-distinct leaves.
    x = _v("x", 4, 4)
    t = _p("add", x, _v("y", 4, 4))
    rep = wg._replace(t, (0,), _v("z", 4, 4))
    assert rep.args[0].name == "z"
    assert rep.args[1] == t.args[1]


def test_term_to_case():
    x = _v("gx", 2, 2)
    t = _p("mul", x, _p("neg", Param("gp", TensorType((2, 2)))))
    env = wg._leaf_env(t)
    c = wg.term_to_case(t, "n", "src", env)
    assert c is not None
    assert [v.name for v in c.inputs] == ["gx"]
    assert len(c.feed) == 1
    assert set(c.param_vals) == {"gp"}


def test_generate_small_and_delta():
    st = _stats()
    rng = random.Random(3)
    cases = _tiny_cases()
    gen, gstats = wg._generate(
        cases, st, rng, 4, ("resample", "mutate"), _SUPPORTED
    )
    assert len(gen) <= 4
    assert sum(gstats.attempted.values()) >= len(gen)
    assert sum(gstats.accepted.values()) == len(gen)
    assert all(c.source.startswith("gen-") for c in gen)
    delta = wg.census_delta(cases, gen)
    assert "new_op_tuples" in delta
    assert "per_strategy" in delta
    text = wg._gen_table(gen, gstats)
    assert "strategy" in text
    assert wg._reject_table(gstats)
    assert wg._novelty_table(delta)
    # an empty delta renders the honest (none) rows.
    assert "(none)" in wg._novelty_table(
        wg.census_delta(_tiny_cases(), [])
    )


def test_interleave_and_depth():
    gen = _tiny_cases()[:4]
    for i, c in enumerate(gen):
        object.__setattr__(c, "source", f"s{i % 2}")
    out = wg._interleave(gen, 3)
    assert len(out) == 3
    # a strategy that runs dry is skipped round-robin, not padded.
    mixed = [gen[0], gen[2], gen[1]]
    object.__setattr__(mixed[0], "source", "s0")
    object.__setattr__(mixed[1], "source", "s1")
    object.__setattr__(mixed[2], "source", "s0")
    assert len(wg._interleave(mixed, 3)) == 3
    x = _v("x", 2)
    assert wg._depth(x) == 0
    assert wg._depth(_p("add", x, _p("mul", x, x))) == 2
    assert wg._as_corpus_terms(gen)[0].source.startswith("s")


def test_run_pipeline_and_delta_tables(monkeypatch):
    """``_run_pipeline`` over the tiny corpus with a shrunk proposal
    pool — the composition is real even when the pool is empty."""
    monkeypatch.setattr(wg.lpipe, "propose", lambda *a: [])
    monkey_cases = _tiny_cases()
    res = wg.run_with_workloads(
        monkey_cases, [], [], [], "hand", None
    )
    # the real pipeline runs here — assert only on the shape of the
    # result, which is corpus-sized, not verdict-sized.
    assert res["baseline"]["n_terms"] == len(monkey_cases)
    assert "ranked" in res["baseline"]
    # the delta printer over fabricated-but-shaped evidence rows.
    prop = lpipe.Proposal(
        name="p1", lhs=_p("mul", "U", Const(1)), rhs="U", family="t"
    )
    ev = lpipe.Evidence(proposal=prop, matches=2, fires=1, paid=1)
    new_prop = lpipe.Proposal(
        name="p_new", lhs=_p("mul", "U", Const(0)), rhs="U", family="t"
    )
    gone_prop = lpipe.Proposal(
        name="p_gone", lhs="V", rhs="V", family="t"
    )
    ev_new = lpipe.Evidence(
        proposal=new_prop,
        matches=1,
        fires=3,
        paid=2,
        fire_cases=["gen:g0", "g1"],
    )
    fake = {
        "baseline": {
            "proposals": [prop, gone_prop],
            "ranked": [ev],
            "n_terms": 4,
            "n_tuples": 3,
        },
        "enlarged": {
            "proposals": [prop, new_prop],
            "ranked": [ev, ev_new],
            "n_terms": 6,
            "n_tuples": 5,
        },
    }
    text = wg._pipeline_delta(fake)
    assert "corpus:" in text and "proposals:" in text
    assert "p_new" in text  # emitted only on the enlarged corpus
    assert "p_gone" in text  # no longer emitted
    assert "(gen=1)" in text  # a generated workload fired it
    rows = wg._evidence_rows([ev])
    assert rows[0]["name"] == "p1" and rows[0]["rank"] == 1


def test_workload_gen_main_skip_pipeline(monkeypatch, tmp_path, capsys):
    cases = _tiny_cases()
    monkeypatch.setattr(wg, "_bench_cases", lambda: (cases, []))
    monkeypatch.setattr(wg, "model_cases", lambda: ([], []))
    out = tmp_path / "wg.json"
    rc = wg.main(
        ["--n", "2", "--skip-pipeline", "--json", str(out)]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_requested"] == 2
    assert "attempted" in payload
    printed = capsys.readouterr().out
    assert "law_workload_gen" in printed
    assert "census delta" in printed


def test_workload_gen_main_pipeline_branch(monkeypatch, tmp_path, capsys):
    """The pipeline-delta branch with a shrunk proposal pool."""
    cases = _tiny_cases()
    monkeypatch.setattr(wg, "_bench_cases", lambda: (cases, []))
    monkeypatch.setattr(wg, "model_cases", lambda: ([], []))
    monkeypatch.setattr(wg.lpipe, "propose", lambda *a: [])
    out = tmp_path / "wg2.json"
    rc = wg.main(["--n", "2", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert "pipeline" in payload
    printed = capsys.readouterr().out
    assert "pipeline delta" in printed


# ---------------------------------------------------------------------------
#  gap_gen — instantiation, embedding, measurement
# ---------------------------------------------------------------------------


def test_gap_attr_candidates():
    assert gg._numel((2, 3, 4)) == 24
    assert all(gg._numel(s) == 24 for s in gg._factorings((2, 3, 4)))
    assert (4, 6) in gg._factorings((2, 3, 4))
    assert gg._super_shapes((4, 1))
    # rank-0 prepends every prefix — the whole pool minus itself.
    assert gg._super_shapes(()) == [s for s in gg._SHAPE_POOL if s]
    # op-specific tables.
    assert gg._op_attr_candidates("select", "dim", 2, 4, {}) == [0, 1]
    assert gg._op_attr_candidates("select", "index", 2, 4, {}) == [
        0,
        1,
        2,
        3,
    ]
    assert gg._op_attr_candidates("unknown_op", "x", 2, 4, {}) is None
    # generic fallbacks and the empty-veto.
    assert gg._attr_candidates("any", "keepdim", (4,), {}) == [
        True,
        False,
    ]
    assert gg._attr_candidates("any", "equation", (4,), {}) == []
    assert gg._attr_candidates("any", "dim", (2, 3), {}) != []
    assert gg._attr_candidates("any", "mystery", (4,), {}) == [0, 1]
    # split sizes are a list summing to the extent, or an int chunk
    # size that divides it evenly.
    for s in gg._attr_candidates("split", "sizes", (4,), {}):
        if isinstance(s, int):
            assert 4 % s == 0
        else:
            assert sum(s) == 4


@pytest.mark.parametrize(
    "op, key, rank, sz, attrs",
    [
        ("slice", "start", 2, 4, {}),
        ("slice", "end", 2, 4, {"start": 2}),
        ("slice", "step", 2, 4, {}),
        ("transpose", "dim0", 3, 4, {}),
        ("unsqueeze", "dim", 2, 4, {}),
        ("squeeze", "dim", 2, 4, {}),
        ("softmax", "dim", 2, 4, {}),
        ("log_softmax", "dim", 3, 8, {}),
        ("flatten", "start_dim", 3, 4, {}),
        ("flatten", "end_dim", 3, 4, {"start_dim": 1}),
        ("chunk", "chunks", 2, 4, {}),
        ("chunk", "index", 2, 4, {"chunks": 4}),
        ("split", "index", 2, 4, {}),
        ("reshape", "shape", 2, 4, {}),
        ("broadcast_to", "shape", 2, 4, {}),
    ],
)
def test_gap_op_attr_table(op, key, rank, sz, attrs):
    out = gg._op_attr_candidates(op, key, rank, sz, attrs)
    # op-specific rows are either a real domain or None (defer to
    # the generic/special-case path).
    assert out is None or out
    # the generic wrapper honours the special cases too.
    gen = gg._attr_candidates(op, key, (4,) * rank, attrs)
    assert gen is not None
    if op == "reshape" and key == "shape":
        assert all(gg._numel(s) == 16 for s in gen)
    if op == "broadcast_to" and key == "shape":
        assert gen == gg._super_shapes((4,) * rank)


def test_gap_instantiate_and_check_bound():
    u = _v("u", 4, 4)
    subst = {"U": u}
    t = gg._instantiate(
        _p("select", "U", dim="D", index="I"), subst, {}, random.Random(0)
    )
    assert t is not None and t.op == "select"
    assert isinstance(t.attrs["dim"], int)
    # shared attr metavar names share one minted value.
    memo: dict = {}
    t1 = gg._instantiate(
        _p("select", "U", dim="D", index=0), subst, memo,
        random.Random(0),
    )
    t2 = gg._instantiate(
        _p("select", "U", dim="D", index=1), subst, memo,
        random.Random(0),
    )
    assert t1.attrs["dim"] == t2.attrs["dim"] == memo["D"]
    # a metavar with no binding yields None; a non-pattern leaf
    # passes through.
    assert gg._instantiate("Q", {}, {}, random.Random(0)) is None
    assert gg._instantiate(Const(3), {}, {}, random.Random(0)) == Const(3)
    # _check_bound mirrors the firing path: check veto, derive None,
    # derive raise, and a good derive.
    prop = lpipe.Proposal(
        name="p",
        lhs="U",
        rhs="U",
        family="t",
        check=lambda bound: False,
    )
    assert gg._check_bound(prop, {"U": u}) is None
    prop2 = lpipe.Proposal(
        name="p",
        lhs="U",
        rhs="U",
        family="t",
        derive=lambda bound: None,
    )
    assert gg._check_bound(prop2, {"U": u}) is None
    prop3 = lpipe.Proposal(
        name="p",
        lhs="U",
        rhs="U",
        family="t",
        derive=lambda bound: 1 / 0,
    )
    assert gg._check_bound(prop3, {"U": u}) is None
    prop4 = lpipe.Proposal(
        name="p",
        lhs="U",
        rhs="U",
        family="t",
        derive=lambda bound: {"x": 1},
    )
    assert gg._check_bound(prop4, {"U": u}) == {"U": u, "x": 1}


def _gap_stats():
    return _stats()


def test_gap_synthesize_witnesses():
    prop = lpipe.Proposal(
        name="mul_sel_w",
        family="t",
        lhs=_p("mul", _p("select", "U", dim="D", index="I"), "V"),
        rhs=_p("select", _p("mul", "U", "V"), dim="D", index="I"),
    )
    insts = gg.synthesize(
        prop,
        _gap_stats(),
        _SUPPORTED,
        set(),
        random.Random(0),
        n=1,
        require_novel=False,
    )
    assert insts
    term, _, env = insts[0]
    # the witness re-matches its own pattern (the self-check).
    assert _term_match(prop.lhs, term) is not None
    assert env


def test_gap_synthesize_respects_vetoes():
    # a proposal whose check vetoes every instance -> no witnesses.
    prop = lpipe.Proposal(
        name="veto",
        family="t",
        lhs=_p("mul", "U", "V"),
        rhs="U",
        check=lambda bound: False,
    )
    assert (
        gg.synthesize(
            prop,
            _gap_stats(),
            _SUPPORTED,
            set(),
            random.Random(0),
            n=1,
        )
        == []
    )


def test_leaf_paths_and_concrete():
    x = _v("x", 4, 4)
    t = _p("add", x, _p("mul", x, Param("p", TensorType((4, 4)))))
    paths = gg._leaf_paths(t)
    assert (0,) in [p for p, _ in paths]
    assert gg._concrete(t) == (4, 4)
    assert gg._concrete(Var("u", TensorType((None,)))) is None
    assert gg._case_env(t) is not None


def test_gen_cases_for_embeddings():
    prop = lpipe.Proposal(
        name="mul_sel_w",
        family="t",
        lhs=_p("mul", _p("select", "U", dim="D", index="I"), "V"),
        rhs=_p("select", _p("mul", "U", "V"), dim="D", index="I"),
    )
    cases = [*_tiny_cases(), _tiny_model_case()]
    out, prov = gg.gen_cases_for(
        prop,
        cases,
        _gap_stats(),
        _SUPPORTED,
        set(),
        random.Random(0),
        require_novel=False,
    )
    assert prov["synthesized"] >= 1
    assert out
    # the bare instance case exists; ctx/graft are best-effort.
    assert any(e.startswith("min") for e in prov["embeddings"])
    assert all(c.source == "gen-gap" for c in out)


def test_gap_graft_donor():
    x = _v("gx", 4, 4)
    inst = _p("mul", x, _v("gy", 4, 4))
    model = _tiny_model_case()
    st = _stats()
    graft = gg._graft(
        inst, (4, 4), [model], st, set(), _SUPPORTED,
        random.Random(0),
    )
    if graft is not None:
        assert any(
            s is inst
            for s in wg._iter_subterms(graft)
        )
    # no donor of the right shape -> None.
    tiny = _p("mul", _v("a", 2), _v("b", 2))
    assert (
        gg._graft(
            tiny, (2,), [model], st, set(), _SUPPORTED,
            random.Random(0),
        )
        is None
    )
    # a non-Op instance / no out shape -> None.
    assert (
        gg._graft(tiny, None, [model], st, set(), _SUPPORTED,
                  random.Random(0))
        is None
    )


def test_gap_result_properties():
    cr = gg.CaseResult(
        name="c",
        fires=2,
        changed=True,
        paid=True,
        verified="pass",
        base_cost=10.0,
        add_cost=8.0,
        cert="pass",
        closure=1.1,
    )
    res = gg.GapResult(
        name="p",
        family="f",
        relation="new",
        base_reason="no firing",
        base_truth=True,
        case_results=[cr],
    )
    assert res.fires == 2
    assert res.paid == 1
    assert res.verify_fail == 0
    assert res.cert_fail == 0
    assert res.truth
    assert res.gen_drop == pytest.approx(0.2)
    assert res.closure_ratio == pytest.approx(1.1)
    assert res.would_ship
    assert res.new_reason == "SHIP (on generated evidence)"
    # the honest failure ladder.
    bad = gg.GapResult(
        name="q", family="f", relation="new",
        base_reason="x", base_truth=False, num_true=False,
    )
    assert bad.new_reason.startswith("false")
    unk = gg.GapResult(
        name="q", family="f", relation="new",
        base_reason="x", base_truth=False, num_true=None,
    )
    assert unk.new_reason.startswith("unproven")
    notnew = gg.GapResult(
        name="q", family="f", relation="duplicate",
        base_reason="x", base_truth=True,
    )
    assert "not new" in notnew.new_reason
    nofire = gg.GapResult(
        name="q", family="f", relation="new", base_reason="x",
        base_truth=True,
    )
    assert "no firing" in nofire.new_reason
    nopay = gg.GapResult(
        name="q", family="f", relation="new", base_reason="x",
        base_truth=True,
        case_results=[gg.CaseResult(name="c", fires=1)],
    )
    assert "never lowers" in nopay.new_reason
    vf = gg.GapResult(
        name="q", family="f", relation="new", base_reason="x",
        base_truth=True,
        case_results=[
            gg.CaseResult(
                name="c", fires=1, paid=True, verified="FAIL"
            )
        ],
    )
    assert "differ" in vf.new_reason
    cert = gg.GapResult(
        name="q", family="f", relation="new", base_reason="x",
        base_truth=True,
        case_results=[
            gg.CaseResult(name="c", fires=1, paid=True, cert="FAIL")
        ],
    )
    assert "certificate" in cert.new_reason
    blow = gg.GapResult(
        name="q", family="f", relation="new", base_reason="x",
        base_truth=True,
        case_results=[
            gg.CaseResult(name="c", fires=1, paid=True, closure=9.9)
        ],
    )
    assert "blow-up" in blow.new_reason


def test_targets_filter():
    prop = lpipe.Proposal(
        name="p", lhs="U", rhs="U", family="t"
    )
    ev_gap = lpipe.Evidence(proposal=prop, fires=0, num_true=None)
    ev_fired = lpipe.Evidence(proposal=prop, fires=2)
    ev_false = lpipe.Evidence(proposal=prop, fires=0, num_true=False)
    got = gg.targets([ev_gap, ev_fired, ev_false])
    assert got == [ev_gap]
    assert gg.targets([ev_gap], only={"nope"}) == []
    assert gg.targets([ev_gap], only={"p"}) == [ev_gap]


def test_measure_candidate_unmatched_instance():
    """A provenance term that does not re-match the pattern leaves
    the oracle fields at their defaults — no crash, no fabrication."""
    prop = lpipe.Proposal(
        name="p",
        family="t",
        lhs=_p("mul", "U", Const(1)),
        rhs="U",
    )
    prop_ev = lpipe.Evidence(proposal=prop, relation="new")
    x = _v("gx", 4, 4)
    res = gg.measure_candidate(
        prop_ev,
        [],
        {
            "synthesized": 0,
            "instances": ["add(x,x)"],
            "_terms": [_p("add", x, x)],
            "embeddings": [],
        },
        list(ALL_RULES),
        None,
        None,
    )
    assert res.num_true is None
    assert res.derivable is False
    assert res.instances == ["add(x,x)"]


def test_measure_case_and_candidate():
    """The real referees on a generated-sized term: the lone rule
    fires, the reach row compares ALL vs ALL+rule."""
    x = _v("gx", 4, 4)
    term = _p("mul", x, Const(1))
    env = wg._leaf_env(term)
    case = wg.term_to_case(term, "g", "gen-gap", env)
    sink = _sink()
    cost_fn = _cost_fn(sink)
    prop = lpipe.Proposal(
        name="p_mul_one",
        family="t",
        lhs=_p("mul", "U", Const(1)),
        rhs="U",
    )
    rule = prop.as_rule()
    res = gg._measure_case(case, rule, list(ALL_RULES), sink, cost_fn)
    assert res.fires >= 1
    assert res.changed and res.verified == "pass"
    assert res.cert == "pass"
    assert res.base_cost >= res.add_cost >= 0.0
    prop_ev = lpipe.Evidence(proposal=prop, relation="new")
    prop_ev.num_true = None
    out = gg.measure_candidate(
        prop_ev, [case], {"synthesized": 1, "instances": [str(term)],
                          "_terms": [term], "embeddings": ["min0"]},
        list(ALL_RULES), sink, cost_fn,
    )
    assert isinstance(out, gg.GapResult)
    assert out.synthesized == 1
    assert out.case_results
    assert out.truth  # mul(u,1)=u is provable true on the instance
    assert out.num_true is True or out.derivable


def test_gap_tables_and_json(tmp_path):
    res = gg.GapResult(
        name="p",
        family="f",
        relation="new",
        base_reason="no firing",
        base_truth=True,
        synthesized=1,
        instances=["mul(x,1)"],
        embeddings=["min0"],
        case_results=[
            gg.CaseResult(
                name="c",
                fires=1,
                changed=True,
                paid=True,
                verified="pass",
                base_cost=10.0,
                add_cost=8.0,
                cert="pass",
                closure=1.0,
                note="probe note",
            )
        ],
    )
    assert "candidate" in gg._target_table([res])
    assert "case" in gg._case_table([res])
    assert "num_true" in gg._oracle_table([res])
    # an empty result set renders the honest empty rows.
    assert "no generated cases" in gg._case_table([])
    out = tmp_path / "g.json"
    gg._dump_json(out, [res], {"seed": 0})
    payload = json.loads(out.read_text())
    assert payload["candidates"][0]["name"] == "p"
    assert payload["candidates"][0]["would_ship"] is True


def test_named_targets_and_main(monkeypatch, tmp_path, capsys):
    """``main --skip-baseline`` requires --only; with a named
    candidate the whole synthesis loop runs on the tiny corpus."""
    cases = [*_tiny_cases(), _tiny_model_case()]
    monkeypatch.setattr(gg, "_bench_cases", lambda: (cases, []))
    monkeypatch.setattr(gg, "model_cases", lambda: ([], []))
    # --only is mandatory on the skip-baseline path.
    assert gg.main(["--skip-baseline"]) == 1
    prop = lpipe.Proposal(
        name="p_mul_one",
        family="t",
        lhs=_p("mul", "U", Const(1)),
        rhs="U",
    )
    monkeypatch.setattr(gg.lpipe, "propose", lambda *a: [prop])
    out = tmp_path / "gap.json"
    rc = gg.main(
        ["--skip-baseline", "--only", "p_mul_one", "--json", str(out)]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["n_targets"] == 1
    assert payload["candidates"][0]["name"] == "p_mul_one"
    printed = capsys.readouterr().out
    assert "law_gap_targeted_gen" in printed
    assert "verdict movement" in printed


def test_named_targets_missing_and_pipeline_path(
    monkeypatch, tmp_path, capsys
):
    cases = _tiny_cases()
    prop = lpipe.Proposal(
        name="p_mul_one",
        family="t",
        lhs=_p("mul", "U", Const(1)),
        rhs="U",
    )
    monkeypatch.setattr(gg.lpipe, "propose", lambda *a: [prop])
    evs = gg._named_targets({"ghost"}, cases, "hand", None)
    assert evs == []  # the missing name is reported, not faked
    assert "no such proposal" in capsys.readouterr().out
    assert gg._named_targets(None, cases, "hand", None) is None
    # a proposal carrying a measured instance lifts num_true along.
    prop2 = lpipe.Proposal(
        name="p_with_inst",
        family="t",
        lhs=_p("mul", "U", Const(1)),
        rhs="U",
        instance=(_p("mul", _v("x", 4, 4), Const(1)), _v("x", 4, 4)),
    )
    monkeypatch.setattr(
        gg.lpipe, "propose", lambda *a: [prop, prop2]
    )
    evs = gg._named_targets(
        {"p_mul_one", "p_with_inst"}, cases, "hand", None
    )
    assert len(evs) == 2
    assert evs[0].proposal.name == "p_mul_one"
    by_name = {e.proposal.name: e for e in evs}
    assert by_name["p_with_inst"].num_true is True
    # the non-skip path reads targets off a pipeline result.
    ev = lpipe.Evidence(proposal=prop, fires=0, num_true=None)
    monkeypatch.setattr(
        gg.lpipe,
        "run_pipeline",
        lambda *a: {"ranked": [ev]},
    )
    monkeypatch.setattr(gg, "_bench_cases", lambda: (cases, []))
    monkeypatch.setattr(gg, "model_cases", lambda: ([], []))
    rc = gg.main(["--json", str(tmp_path / "g.json")])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "gap targets" in printed
