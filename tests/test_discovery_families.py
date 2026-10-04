"""Tests for ``catopt_discovery.families`` — training-program families.

The families build the small IR programs the learned/RL policies are
trained on: right-seeded matmul chains (``k > m`` so ``assoc_matmul``
always pays), elementwise ``square(t) + t*t`` duplications, and
stacked ``linear -> linear -> relu`` blocks exported through torch.
The scoring helpers rank a policy's pick against each rule's true
cost delta.
"""

import pytest
from catopt_core.ir import Op, Var, op_repr
from catopt_core.laws import ALL_RULES
from catopt_discovery import families as fam

_BY_NAME = {r.name: r for r in ALL_RULES}


def _leaf(term):
    """Return the Var leaves of a chain term as ``(a, b, c)``."""
    a, inner = term.args
    b, c = inner.args
    return a, b, c


def test_chain_structure():
    t = fam.chain(3, 5, 2)
    assert t.op == "matmul"
    a, inner = t.args
    assert inner.op == "matmul"
    b, c = inner.args
    assert tuple(a.typ.shape) == (3, 5)
    assert tuple(b.typ.shape) == (5, 2)
    assert tuple(c.typ.shape) == (2, 3)
    assert all(isinstance(v, Var) for v in (a, b, c))


def test_chain_programs_right_seeded_k_gt_m():
    progs = fam.chain_programs(6, seed=0)
    assert len(progs) == 6
    for p in progs:
        a, b, c = _leaf(p)
        k = a.typ.shape[1]
        m = b.typ.shape[1]
        assert k > m, "assoc_matmul must always pay on this family"
        assert b.typ.shape[0] == k
        assert c.typ.shape[0] == m
        assert c.typ.shape[1] == a.typ.shape[0]


def test_chain_programs_seeded_deterministic():
    first = [op_repr(p) for p in fam.chain_programs(4, seed=7)]
    second = [op_repr(p) for p in fam.chain_programs(4, seed=7)]
    assert first == second


def test_dup_programs_both_spellings():
    saw_var = saw_op = False
    for seed in range(8):
        for p in fam.dup_programs(4, seed):
            sq, mm = p.args
            assert p.op == "add"
            assert sq.op == "square" and mm.op == "mul"
            # The same t object is shared across all three slots —
            # that is the duplication ``square_expand`` collapses.
            assert sq.args[0] is mm.args[0] is mm.args[1]
            if isinstance(sq.args[0], Op):
                saw_op = True
            else:
                saw_var = True
    assert saw_var, "the x*y spelling never appeared"
    assert saw_op, "the bare-variable spelling never appeared"


def test_family_programs_dispatch():
    ch = fam.family_programs("chain", 3, 0)
    assert all(p.op == "matmul" for p in ch)
    du = fam.family_programs("dup", 2, 0)
    assert all(p.op == "add" for p in du)
    lin = fam.family_programs("linear", 1, 0)
    assert len(lin) == 1 and lin[0].op == "relu"


def test_family_programs_held_out_wider():
    held = fam.family_programs("chain", 4, 0, held_out=True)
    for p in held:
        a, b, _c = _leaf(p)
        # Held-out dims come from the wider (5, 7) range.
        assert a.typ.shape[1] >= 5
        assert a.typ.shape[1] > b.typ.shape[1]


def test_family_programs_unknown():
    with pytest.raises(ValueError, match="unknown family"):
        fam.family_programs("nope", 1, 0)


def test_linear_programs_export():
    progs = fam.linear_programs(2, 0)
    assert len(progs) == 2
    for p in progs:
        assert isinstance(p, Op)
        assert p.op == "relu"


def test_mixture_programs_interleaves_families():
    out = fam.mixture_programs(["chain", "dup"], 2, 0)
    assert len(out) == 4
    assert [p.op for p in out] == ["matmul", "add", "matmul", "add"]


def test_ranks_desc_ties_share_the_average():
    assert fam.ranks_desc([3.0, 1.0, 3.0, 0.5]) == [1.5, 3.0, 1.5, 4.0]
    assert fam.ranks_desc([5.0, 4.0, 3.0]) == [1.0, 2.0, 3.0]
    assert fam.ranks_desc([2.0, 2.0, 2.0]) == [2.0, 2.0, 2.0]
    assert fam.ranks_desc([1.0]) == [1.0]
    assert fam.ranks_desc([]) == []


def test_rule_deltas_reports_each_rule():
    term = fam.chain(3, 5, 2)
    rules = [
        _BY_NAME[n] for n in ("comm_add", "assoc_matmul", "sub_to_add")
    ]
    deltas = fam.rule_deltas(term, rules)
    assert set(deltas) == {"comm_add", "assoc_matmul", "sub_to_add"}
    # Right seed with k > m: re-bracketing strictly pays.
    assert deltas["assoc_matmul"] > 0.0
    assert deltas["comm_add"] == 0.0


def test_score_pick_hit_and_miss():
    term = fam.chain(3, 5, 2)
    rules = [
        _BY_NAME[n] for n in ("comm_add", "assoc_matmul", "sub_to_add")
    ]
    hit = fam.score_pick(term, rules, "assoc_matmul")
    assert hit["hit"] is True
    assert hit["rank"] == 1.0
    assert hit["best"] == "assoc_matmul"
    assert hit["delta"] == hit["best_delta"] > 0.0
    miss = fam.score_pick(term, rules, "comm_add")
    assert miss["hit"] is False
    assert miss["delta"] == 0.0
    assert miss["rank"] > 1.0
    assert miss["n_rules"] == 3


def test_summarize_aggregates():
    term = fam.chain(3, 5, 2)
    rules = [
        _BY_NAME[n] for n in ("comm_add", "assoc_matmul", "sub_to_add")
    ]
    scores = [
        fam.score_pick(term, rules, "assoc_matmul"),
        fam.score_pick(term, rules, "comm_add"),
    ]
    agg = fam.summarize(scores)
    assert agg["n"] == 2
    assert agg["hit_rate"] == 0.5
    assert (
        agg["mean_rank"] == (scores[0]["rank"] + scores[1]["rank"]) / 2
    )
    assert (
        agg["mean_delta"]
        == (scores[0]["delta"] + scores[1]["delta"]) / 2
    )
