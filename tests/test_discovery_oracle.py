"""Second-tier tests for ``catopt_discovery.oracle``.

``test_law_view_oracle.py`` pins the headline verdicts; this file
covers the machinery underneath — the tri-state evaluator's error
rows, the attribute/leaf binding domains, the instance enumerator's
skip and veto paths, the mechanical guard features one by one, the
real-match sweep's check/derive/dedup paths, the ``ill-formed`` /
``unproven`` verdicts and the ``--json`` driver.

View-under-view chains (``transpose(transpose(u,...),...)`` or the
``unsqueeze -> expand -> reshape`` triple of ``gqa_absorb_repeat``)
are grouped into chained attr domains: the outer node's options are
computed against the operand shape the drawn inner attrs actually
produce (:func:`catopt_discovery.oracle._chain_domain`), so the
joint draws are consistent by construction.
"""

import itertools
import json

import pytest
import torch
from catopt_core.egraph.terms import _term_instantiate
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_discovery import oracle as vo


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


# ---------------------------------------------------------------------------
#  Tri-state evaluation
# ---------------------------------------------------------------------------


def test_eval_instance_env_err():
    """A leaf with non-int dims cannot build an env."""
    x = Var("x", TensorType((None, 4)))
    outcome, note = vo.eval_instance(_p("mul", x, x), _p("mul", x, x))
    assert outcome == "env-err"
    assert "non-int" in note


def test_eval_instance_lhs_err_and_both_err():
    x = _v("x", 2, 3)
    outcome, _ = vo.eval_instance(_p("frobnicate", x), x)
    assert outcome == "lhs-err"
    outcome, _ = vo.eval_instance(
        _p("frobnicate", x), _p("frobnicate", x)
    )
    assert outcome == "both-err"


def test_env_for_builds_randn_per_leaf():
    x, y = _v("x", 2), _v("y", 3, 1)
    env = vo._env_for(_p("add", x, y))
    assert env is not None
    assert {leaf.name for leaf in env} == {"x", "y"}
    # A leaf type of ``None`` reports the empty shape — no dims to
    # check — so the env is built, not vetoed.
    bare = Var("b", None)
    assert vo._env_for(bare) is not None


def test_leaf_metavars_order_dedup_and_non_str():
    pat = _p("mul", _p("add", "A", Const(1)), "A")
    assert vo._leaf_metavars(pat) == ["A"]
    assert vo._leaf_metavars(_p("add", "A", "B")) == ["A", "B"]
    # A Var/Const leaf is neither a metavar nor an Op — ignored.
    assert vo._leaf_metavars(_p("add", "A", _v("x", 2))) == ["A"]
    assert vo._leaf_metavars("A") == ["A"]


def test_parents_maps_metavars_to_parent_ops():
    pat = _p("mul", _p("select", "U", dim="D", index="I"), "U")
    parents = vo._parents(pat)
    assert parents["U"] == {"select", "mul"}


def test_small_helpers():
    t = torch.zeros(2)
    assert vo._as_tensor(t) is t
    assert vo._as_tensor(3.0) is None
    x = _v("x", 2, 2)
    # ``_view_call`` applies the torch binding; an op outside the
    # table is an honest KeyError.
    assert vo._view_call(
        _p("unsqueeze", x, dim=0), torch.zeros(2, 2)
    ).shape == (1, 2, 2)
    with pytest.raises(KeyError):
        vo._view_call(_p("frobnicate", x), torch.zeros(2, 2))
    assert vo._bcast_to(torch.zeros(2), (3, 2)) is not None
    assert vo._bcast_to(torch.zeros(2, 3), (2,)) is None


# ---------------------------------------------------------------------------
#  Attribute domains
# ---------------------------------------------------------------------------


def test_dims_norm_shape_numel():
    assert vo._dims(0) == []
    assert vo._dims(2) == [-2, -1, 0, 1]
    assert vo._norm(-1, 3) == 2
    assert vo._norm(5, 0) == 5
    assert vo._shape_numel((2, 3, 4)) == 24
    assert vo._shape_numel(()) == 1


def test_reshape_targets_same_numel_deduped():
    for s in vo._reshape_targets((2, 3, 4)):
        assert vo._shape_numel(s) == 24
    assert len(vo._reshape_targets((2, 3, 4))) <= vo._RESHAPE_TARGET_CAP
    # Edge merges for rank >= 3 (leading and trailing pairs).
    assert (6, 4) in vo._reshape_targets((2, 3, 4))
    assert (2, 12) in vo._reshape_targets((2, 3, 4))
    # Interior adjacent merges — a repeat-chain's head*r pair is
    # interior for a rank-5 operand.
    assert (2, 3, 4, 4) in vo._reshape_targets((2, 3, 2, 2, 4))
    assert (2, 6, 4) in vo._reshape_targets((2, 3, 2, 4))
    # rank >= 2 with an even first dim splits it.
    assert (2, 2, 2, 3) in vo._reshape_targets((4, 2, 3))


@pytest.mark.parametrize(
    "op, keys, shape, want",
    [
        ("getitem", ("index",), (4,), "nonempty"),
        ("select", ("dim", "index"), (), "empty"),
        ("select", ("dim", "index"), (4,), "nonempty"),
        ("slice", ("dim", "start", "end"), (), "empty"),
        ("slice", ("dim", "start", "end"), (4,), "nonempty"),
        ("slice", ("dim", "start", "end", "step"), (4,), "nonempty"),
        ("unsqueeze", ("dim",), (2, 3), "nonempty"),
        ("squeeze", ("dim",), (1, 3), "nonempty"),
        ("transpose", ("dim0", "dim1"), (), "empty"),
        ("transpose", ("dim0", "dim1"), (2, 3), "nonempty"),
        ("reshape", ("shape",), (2, 4), "nonempty"),
        ("view", ("shape",), (2, 4), "nonempty"),
        ("expand", ("shape",), (1, 4), "nonempty"),
        ("broadcast_to", ("shape",), (), "nonempty"),
        ("chunk", ("chunks", "dim", "index"), (4, 4), "nonempty"),
        ("split", ("sizes", "dim", "index"), (4, 4), "nonempty"),
        ("narrow", ("dim", "start", "length"), (4,), "nonempty"),
        ("permute", ("dim",), (2, 3), "nonempty"),
        ("unbind", ("dim", "index"), (4, 4), "nonempty"),
        ("movedim", ("source", "destination"), (4,), "empty"),
        ("movedim", ("source", "destination"), (2, 3), "nonempty"),
        ("flatten", ("start_dim", "end_dim"), (2, 3), "nonempty"),
        ("mystery", ("x",), (4,), "none"),
    ],
)
def test_attr_options(op, keys, shape, want):
    opts = vo._attr_options(op, keys, shape)
    if want == "none":
        assert opts is None
    elif want == "empty":
        assert opts == []
    else:
        assert opts
        for o in opts:
            assert set(o) <= set(keys)


def test_attr_options_inner_vetoes():
    """Per-key inner conditions of the attr tables."""
    # slice over a non-int extent skips that axis (``continue``).
    opts = vo._attr_options("slice", ("dim", "start", "end"), (4, "x"))
    assert opts and all(o["dim"] != -1 for o in opts)
    # chunk needs extent >= chunks; a (1,)-shaped operand admits
    # chunks=1 only.
    opts = vo._attr_options("chunk", ("chunks", "dim", "index"), (1,))
    assert opts and all(o["chunks"] == 1 for o in opts)
    # split needs extent >= 2.
    assert (
        vo._attr_options("split", ("sizes", "dim", "index"), (1,)) == []
    )
    # narrow over a non-int extent emits no option.
    assert (
        vo._attr_options("narrow", ("dim", "start", "length"), ("x",))
        == []
    )
    # permute at rank 0 has no non-empty permutation.
    assert vo._attr_options("permute", ("dim",), ()) == []


def test_operand_shape_kinds():
    assert vo._operand_shape(_v("x", 2, 3)) == (2, 3)
    p = Param("w", TensorType((4,)))
    assert vo._operand_shape(p) == (4,)
    assert vo._operand_shape(Const(0.5)) == ()
    # A compound term with an unresolved attr metavariable falls back
    # to the first tensor leaf's declared shape.
    u = _v("u", 2, 4)
    t = _p("reshape", u, shape="S1")
    assert vo._operand_shape(t) == (2, 4)
    # No tensor leaf at all -> () — an op the shape inference does
    # not know falls to the walk, which finds only Consts.
    assert vo._operand_shape(_p("add", Const(1), Const(2))) == ()
    assert (
        vo._operand_shape(_p("frobnicate", _p("neg", Const(1)))) == ()
    )
    # ``_shape_of`` may raise on unresolved attrs — that currently
    # propagates (see the nested-view boundary test below).
    u2 = _v("u2", 2, 4)
    nested = _p("transpose", u2, dim0="A", dim1="B")
    with pytest.raises(TypeError):
        vo._operand_shape(nested)


def test_chain_domain_resolves_through_view_outputs():
    """The unsqueeze→expand→reshape chain enumerates jointly.

    Each outer node's attr options are computed against the operand
    shape the drawn inner attrs produce: ``unsqueeze(u, 2)`` on
    ``u=(2,3,4)`` yields ``(2,3,1,4)``; ``expand`` grows that 1-axis
    to ``(2,3,2,4)``; ``reshape``'s interior merge yields ``(2,6,4)``
    — the ``repeat-chain`` shape triple the pre-chaining domains
    could never mint.
    """
    chain = _p(
        "reshape",
        _p("expand", _p("unsqueeze", "u", dim="UD"), shape="ES"),
        shape="RS",
    )
    nodes = vo._view_nodes([chain])
    groups = vo._attr_groups(nodes)
    assert len(groups) == 1 and len(groups[0]) == 3
    assigns = vo._chain_domain(groups[0], {"u": _v("u", 2, 3, 4)})
    # The repeat-merge triple: unsqueeze the last-but-one axis
    # (``dim=-2`` is the spellable form; ``dim=2`` is not in the axis
    # domain), grow the inserted 1 to 2, merge dims 2*3 back.
    assert {"UD": -2, "ES": (2, 3, 2, 4), "RS": (2, 6, 4)} in assigns
    # Every assignment is chain-consistent: instantiating the inner
    # views under it produces exactly the shapes the outer options
    # were drawn for.
    for a in assigns:
        subst = {f"$attr:{k}": v for k, v in a.items()}
        subst["u"] = _v("u", 2, 3, 4)
        us = vo._operand_shape(
            _term_instantiate(chain.args[0].args[0], subst)
        )
        es = vo._operand_shape(_term_instantiate(chain.args[0], subst))
        assert es == a["ES"]
        assert us[a["UD"] % len(us)] == 1
        rs = vo._operand_shape(_term_instantiate(chain, subst))
        assert rs == a["RS"]


def test_chain_domain_edge_paths():
    """``_chain_domain``'s fallback and veto paths.

    - a member whose operand cannot instantiate (an unbound leaf
      metavariable) falls back to the rank-0 shape domain;
    - a member with no options under the drawn operand shape kills
      the partial assignment — the group returns ``None`` when every
      draw dies;
    - a metavariable shared *inside* a chain must agree, so a chain
      whose levels can't share a value enumerates nothing.
    """
    usq = _p("unsqueeze", "u", dim="UD")
    assigns = vo._chain_domain([usq], {})
    assert assigns and all("UD" in a for a in assigns)

    # A fabricated arg-less member falls back to the rank-0 domain.
    bare = _p("unsqueeze", dim="UD")
    assert vo._chain_domain([bare], {})

    sel = _p("select", "u", dim="SD", index="SI")
    assert vo._chain_domain([sel], {}) is None

    # ``D`` is both the unsqueeze axis and the expand shape — no
    # integer axis equals a shape tuple, so the chain is empty and
    # the binding vetoes.
    dead = _p("expand", _p("unsqueeze", "u", dim="D"), shape="D")
    nodes = vo._view_nodes([dead])
    groups = vo._attr_groups(nodes)
    assert len(groups) == 1
    assert vo._attr_domains(nodes, {"u": _v("u", 2, 3)}) is None


def test_synth_bases_skips_fully_conflicted_binding():
    """A binding whose attr product is all merge-conflicts yields no
    bases — the lazy group is probed once and skipped, preserving the
    eager ``if bases:`` group indexing."""
    pat = _p(
        "mul",
        _p("expand", "v", shape="S"),
        _p("transpose", "u", dim0="S", dim1="T"),
    )
    # S is drawn as a shape tuple by ``expand`` and as an int axis by
    # ``transpose`` — disjoint option types, so every combination
    # merge-conflicts.
    assert list(vo._synth_bases(pat, "v")) == []


def test_nested_view_candidate_chains_consistently():
    """A view-under-view candidate enumerates as one chained group.

    ``transpose(transpose(u, A, B), A, B)`` links the outer node's
    operand to the inner node, so both transposes are enumerated
    jointly — the outer domain is computed over the *transposed*
    operand shape (the drawn inner dims), not the pre-view leaf.
    """
    lhs = _p(
        "transpose",
        _p("transpose", "U", dim0="A", dim1="B"),
        dim0="A",
        dim1="B",
    )
    nodes = vo._view_nodes([lhs, "U"])
    groups = vo._attr_groups(nodes)
    assert len(groups) == 1 and len(groups[0]) == 2
    domains = vo._attr_domains(nodes, {"U": _v("U", 2, 3)})
    assert len(domains) == 1
    node, opts = domains[0]
    assert isinstance(node, vo._ChainGroup)
    # Every joint draw instantiates both transposes back to the leaf
    # shape — the same (A, B) resolves at both levels.
    for o in opts:
        subst = {f"$attr:{k}": v for k, v in o.items()}
        subst["U"] = _v("U", 2, 3)
        assert vo._operand_shape(_term_instantiate(lhs, subst)) == (
            2,
            3,
        )
    inst = list(vo.synthesize(lhs, "U", limit=24))
    assert inst and all(i.outcome == "equal" for i in inst)


def test_tuple_sources_and_leaf_bindings():
    srcs = vo._tuple_sources("U")
    assert {t.op for t in srcs} == {"topk", "var_mean", "cummax"}
    got = vo._leaf_bindings("U", {"getitem"}, ())
    assert got[:3] == srcs
    # a getitem-parent metavar also gets the shape bank, no scalar.
    kinds = [vo._bind_desc(t) for t in got]
    assert not any(k.startswith("Const") for k in kinds)
    # a free operand leads with the operand-derived shape Var (the
    # most conservative instantiation), then the Const literal and
    # the scalar Var — the distinct-kind bindings stay early under a
    # capped enumeration.
    free = vo._leaf_bindings("V", set(), ((9, 9),))
    assert isinstance(free[0], Var) and free[0].typ.shape == (9, 9)
    assert free[1] == Const(0.5)
    assert isinstance(free[2], Var) and free[2].typ.shape == ()
    assert Const(0.5) in free
    assert any(
        isinstance(t, Var) and t.typ.shape == (9, 9) for t in free
    )
    # a viewed (non-getitem) metavar gets no Const option.
    viewed = vo._leaf_bindings("U", {"select"}, ())
    assert not any(isinstance(t, Const) for t in viewed)


def test_viewed_bindings_and_derived_free_shapes():
    parents = {"U": {"select"}, "V": set()}
    gens = list(vo._viewed_bindings(["U", "V"], parents))
    assert gens and all(set(g) == {"U"} for g in gens)
    derived = vo._derived_free_shapes([(2, 3)], [(2, 3, 1)])
    assert (2, 3) in derived and (2, 3, 1) in derived
    # the single-axis-1 insertions of each shape are in the bank;
    # (2,3,1) was already derived as an insertion, so its own
    # insertions are not re-expanded.
    assert (1, 2, 3) in derived
    assert (2, 3, 1, 1) not in derived


# ---------------------------------------------------------------------------
#  Per-op constant domain — the value-bank rescue (value-bank retro)
# ---------------------------------------------------------------------------
#
#  The generic leaf bank mints one ``Const(0.5)`` for a free operand;
#  a shipped guard can demand a *specific* literal the corner never
#  reaches (``pow``'s exponent, the softmax mask sentinel, ``div``'s
#  unit numerator).  ``_CONST_DOMAIN`` is the small per-op table that
#  mints them; these tests pin the table, the bank placement, and one
#  rule the widening rescues (``pow_to_rsqrt``).


def test_const_domain_keyed_by_parent_op():
    """``_const_domain`` reads the parent op's tabulated literals."""
    assert vo._const_domain({"pow"}) == [
        Const(2),
        Const(-0.5),
        Const(0.5),
        Const(1),
    ]
    # the masked_fill sentinel: ``-inf`` is the one that clears the
    # strict ``const-cmp F < -1e30``; the finite analogues ride along.
    sentinel = [c.value for c in vo._const_domain({"masked_fill"})]
    assert sentinel[0] == float("-inf")
    assert -1e30 in sentinel and 1e9 in sentinel
    assert vo._const_domain({"div"}) == [Const(1), Const(0)]


def test_const_domain_generic_fallback_and_ordering():
    """An op with no entry — or no parent at all — keeps the generic
    ``Const(0.5)`` corner; several parents resolve in sorted-op order,
    deduped across the shared ``0.5`` / ``1``."""
    assert vo._const_domain(set()) == [Const(0.5)]
    assert vo._const_domain({"frobnicate"}) == [Const(0.5)]
    # sorted({"div","pow"}) -> div then pow; the shared 1 / 0.5 drop.
    assert [c.value for c in vo._const_domain({"div", "pow"})] == [
        1,
        0,
        2,
        -0.5,
        0.5,
    ]


def test_leaf_bindings_op_constants_follow_the_generic_corner():
    """The op literals ride just past the documented index-0/1/2
    corner — the generic ``Const(0.5)`` and scalar ``Var`` keep their
    seats, so an existing accepted corner is never shifted."""
    free = vo._leaf_bindings("P", {"pow"}, ((9, 9),))
    assert isinstance(free[0], Var) and free[0].typ.shape == (9, 9)
    assert free[1] == Const(0.5)
    assert isinstance(free[2], Var) and free[2].typ.shape == ()
    assert Const(2) in free and Const(-0.5) in free and Const(1) in free
    # an untabulated parent op keeps only the generic corner.
    plain = vo._leaf_bindings("V", {"mul"}, ((9, 9),))
    assert [c for c in plain if isinstance(c, Const)] == [Const(0.5)]


def test_pow_to_rsqrt_now_measures_a_region():
    """The rescued domain gap: ``pow``'s exponent bank mints ``-0.5``,
    so the shipped ``pow_to_rsqrt`` guard accepts sites the bank
    previously declined (the window was starved)."""
    from catopt_core.laws import ALL_RULES
    from catopt_discovery import evidence as ev

    rule = {r.name: r for r in ALL_RULES}["pow_to_rsqrt"]
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=360)
    )
    assert region.accepted > 0
    assert region.equal > 0
    # exponents other than -0.5 still decline — the guard still bites.
    assert region.declined > 0


def test_div_sqrt_to_rsqrt_now_measures_a_region():
    """``div``'s identity bank mints the unit numerator ``1``, so the
    ``1 / sqrt(x)`` spelling's guard accepts sites now."""
    from catopt_core.laws import ALL_RULES
    from catopt_discovery import evidence as ev

    rule = {r.name: r for r in ALL_RULES}["div_sqrt_to_rsqrt"]
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=360)
    )
    assert region.accepted > 0
    assert region.equal > 0


def test_attr_domains_groups_and_vetoes():
    node = _p("select", "U", dim="D", index="I")
    # A fabricated node with no str attrs is skipped (not a group).
    plain = _p("select", "U", dim=0, index=0)
    domains = vo._attr_domains([plain, node], {"U": _v("u", 4)})
    assert len(domains) == 1
    assert domains[0][0] is node
    # an op the oracle cannot instantiate vetoes the whole binding.
    bad = _p("frobnicate", "U", axis="A")
    assert vo._attr_domains([bad], {"U": _v("u", 4)}) is None
    # a node with no args instantiates nothing — shape () falls to
    # the rank-0 domain (getitem stays bindable, select vetoes).
    bare_sel = _p("select", dim="D", index="I")
    assert vo._attr_domains([bare_sel], {}) is None


def test_attr_domains_shared_names_resolve_once():
    """Two nodes over the same metavar names form ONE group."""
    a = _p("select", "U", dim="D", index="I")
    b = _p("select", "V", dim="D", index="I")
    domains = vo._attr_domains(
        [a, b], {"U": _v("u", 4), "V": _v("v", 4)}
    )
    assert len(domains) == 1


# ---------------------------------------------------------------------------
#  Generic attr domains — non-view ops (attr-sweep retro)
# ---------------------------------------------------------------------------


def test_attr_kind_canonicalizes_and_overrides():
    """``argN`` resolves through ``ATTR_SCHEMA``; the (op, name)
    exceptions and the reduction-``dim`` family beat the name table."""
    assert vo._attr_kind("sdpa", "scale") == "float"
    assert vo._attr_kind("sdpa", "arg6") == "float"  # -> scale
    assert vo._attr_kind("sdpa", "arg5") == "bool"  # -> is_causal
    assert vo._attr_kind("softmax", "dim") == "axis"
    assert vo._attr_kind("sum", "dim") == "red-dims"
    assert vo._attr_kind("rms_norm", "dim") == "shape"
    assert vo._attr_kind("rms_norm", "eps") == "float"
    assert vo._attr_kind("eye", "dim") == "int"
    assert vo._attr_kind("einsum", "equation") == "str"
    # tensor-valued and unknown attrs stay honestly unenumerable.
    assert vo._attr_kind("sdpa", "attn_mask") is None
    assert vo._attr_kind("frobnicate", "axis") is None
    assert vo._attr_kind("frobnicate", "arg9") is None


def test_kind_domain_shapes():
    """Kind domains are small and shape-aware where it matters."""
    # axis: the shape-valid axes for a rank-3 operand.
    assert set(vo._kind_domain("axis", (2, 3, 4))) <= {
        -3,
        -2,
        -1,
        0,
        1,
        2,
    }
    # unknown shape -> the generic axis fallback, still enumerable.
    assert vo._kind_domain("axis", ())
    # red-dims carry both the scalar and tuple spellings.
    rd = vo._kind_domain("red-dims", (2, 3))
    assert (-1,) in rd and (0, 1) in rd and -1 in rd
    # shape: trailing blocks of the operand's shape.
    assert vo._kind_domain("shape", (8, 4)) == [(4,), (8, 4)]
    assert vo._kind_domain("int", ()) == [0, 1, 2]
    assert 0.5 in vo._kind_domain("float", ())
    assert vo._kind_domain("bool", ()) == [False, True]
    # "str" payloads are honestly unenumerable.
    assert vo._kind_domain("str", ()) == []


def test_generic_attr_options():
    """The product over metavar'd keys is emitted, capped, and
    vetoes only on an untypeable key."""
    opts = vo._generic_attr_options("sdpa", ("scale",), ())
    assert opts == [{"scale": 0.5}, {"scale": 1.0}, {"scale": 1e-05}]
    # two metavar'd keys -> the Cartesian product.
    opts = vo._generic_attr_options("dropout", ("p", "train"), ())
    assert len(opts) == 6 and {"p": 0.5, "train": True} in opts
    # ``argN`` spellings kind through the canonical schema.
    opts = vo._generic_attr_options("sdpa", ("arg6",), ())
    assert opts[0] == {"arg6": 0.5}
    # an untypeable key vetoes the node — the honest skip.
    assert vo._generic_attr_options("sdpa", ("attn_mask",), ()) is None
    assert vo._generic_attr_options("frob", ("wat",), (2, 3)) is None


def test_attr_options_falls_through_to_generic():
    """``_attr_options``'s wildcard routes non-view ops to the
    generic domain; the view tables still take precedence."""
    assert vo._attr_options("sdpa", ("scale",), ()) == [
        {"scale": 0.5},
        {"scale": 1.0},
        {"scale": 1e-05},
    ]
    # ``softmax``'s dim enumerates axes — the old ``None`` wall is gone.
    assert vo._attr_options("softmax", ("dim",), (2, 3))
    # an op whose attr cannot be typed still returns ``None``.
    assert vo._attr_options("mystery", ("x",), (4,)) is None


def test_synthesize_nonview_attr_metavar():
    """``synthesize`` evaluates instances over a non-view attr metavar
    — a ``softmax(dim=D)`` self-map now produces equal instances."""
    insts = vo.synthesize(
        _p("softmax", "U", dim="D"),
        _p("softmax", "U", dim="D"),
        limit=120,
    )
    assert insts
    assert any(i.outcome == "equal" for i in insts)
    dims = {
        dict(i.binds).get("$attr:D")
        for i in insts
        if i.outcome in ("equal", "unequal")
    }
    assert len(dims) > 1


def test_synthesize_sdpa_scale_metavar():
    """The ``_g`` shape: ``sdpa(scale="SC")`` enumerates floats and the
    instantiated RHS evaluates — no longer a domain veto."""
    insts = vo.synthesize(
        _p(
            "matmul",
            _p(
                "softmax",
                _p(
                    "matmul",
                    "Q",
                    _p("transpose", "K", dim0="TD1", dim1="TD2"),
                ),
                dim=-1,
            ),
            "V",
        ),
        _p("sdpa", "Q", "K", "V", scale="SC"),
        limit=120,
    )
    assert insts
    assert any("$attr:SC" in dict(i.binds) for i in insts)


def test_diag_product_orders_corners_first():
    """Cantor order covers the low-index corner of every coordinate
    before deep values of any one — the truncation-fairness pin."""
    a = [f"a{i}" for i in range(3)]
    b = [f"b{i}" for i in range(3)]
    c = [f"c{i}" for i in range(3)]
    combos = list(vo._diag_product([a, b, c]))
    assert len(combos) == 27 and set(combos) == set(
        itertools.product(a, b, c)
    )
    # index-sum ordering: the prefix is the shallow corner.
    assert combos[:4] == [
        ("a0", "b0", "c0"),
        ("a0", "b0", "c1"),
        ("a0", "b1", "c0"),
        ("a1", "b0", "c0"),
    ]
    # every coordinate's index-1 value lands inside the first 4 —
    # ``itertools.product`` would put them ~10 tuples apart.
    # degenerate products behave like ``itertools.product``.
    assert list(vo._diag_product([])) == [()]
    assert list(vo._diag_product([a, [], c])) == []


def test_binding_envs_covers_bases_early():
    """The (viewed x attr) bases all contribute before any goes deep:
    the first envs span several view bindings, not one base's corner."""
    lhs = _p("mul", _p("unsqueeze", "U", dim="A_d"), "V")
    rhs = _p("mul", "U", "V")
    u_shapes = []
    for i, env in enumerate(vo._binding_envs(lhs, rhs)):
        if i >= 12:
            break
        sh = tuple(env["U"].typ.shape)
        if sh not in u_shapes:
            u_shapes.append(sh)
    # several viewed bindings are represented inside the first dozen —
    # the old nesting served one base's whole free product first.
    assert len(u_shapes) > 1


def test_binding_envs_lazy_pull_matches_eager_order():
    """The lazy (pull) enumeration yields the eager order exactly.

    Rebuild the pre-laziness algorithm in the test — materialize the
    whole ``_synth_bases`` space, then round-robin the per-base free
    products — and require the yielded binding dicts coincide with
    ``_binding_envs``'s, which now only materializes the prefix a cap
    reaches.
    """
    lhs = _p("mul", _p("unsqueeze", "U", dim="A_d"), "V")
    rhs = _p("mul", "U", "V")

    parents = vo._metavar_parents(lhs, rhs)
    free = sorted(
        m for m in parents if not (parents.get(m, set()) & vo._VIEWISH)
    )
    entries = []
    for base, u_shapes, out_shapes in vo._synth_bases(lhs, rhs):
        derived = vo._derived_free_shapes(u_shapes, out_shapes)
        lists = [
            vo._leaf_bindings(m, parents[m], derived) for m in free
        ]
        entries.append([base, vo._diag_product(lists), True])
    eager: list[dict] = []
    total = 0
    live = len(entries)
    while live and len(eager) < 60:
        for i in range(min(total, len(entries) - 1), -1, -1):
            entry = entries[i]
            if not entry[2]:
                continue
            try:
                combo = next(entry[1])
            except StopIteration:
                entry[2] = False
                live -= 1
                continue
            eager.append(
                {**entry[0], **dict(zip(free, combo, strict=True))}
            )
        total += 1

    lazy = list(itertools.islice(vo._binding_envs(lhs, rhs), 60))
    # A round appends per entry, so the eager cap overshoots — the
    # shared prefix is the equivalence check.
    assert lazy == eager[:60]


# ---------------------------------------------------------------------------
#  The (viewed x attr) diagonal — the enumeration-fairness reorder
# ---------------------------------------------------------------------------
#
#  ``_diag_groups`` is the ragged-product companion of ``_diag_product``:
#  it interleaves a *viewed binding's* list of attr-combination bases
#  with the next viewed binding's, by index-sum.  The base dimension was
#  the last nesting level still shape-major, and a guard needing a
#  different operand rank waited on a whole shape's attr domain — the
#  ``sdpa_fold_*`` starvation (``enumeration-fairness.md``).


def test_diag_groups_cantor_order_and_ragged():
    """``groups[v][a]`` in increasing ``v + a``; a ragged product
    covers every element exactly once."""
    g0 = [(0, a) for a in range(3)]
    g1 = [(1, a) for a in range(2)]
    g2 = [(2, a) for a in range(4)]
    got = list(vo._diag_groups([g0, g1, g2]))
    assert got == [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (2, 1),
        (2, 2),
        (2, 3),
    ]
    # every element of every group survives, exactly once.
    assert sorted(got) == sorted(g0 + g1 + g2)
    # index-sum is non-decreasing — the Cantor order.
    sums = [v + a for v, a in got]
    assert sums == sorted(sums)
    # degenerate: no groups; a single group is its own order.
    assert list(vo._diag_groups([])) == []
    assert list(vo._diag_groups([g0])) == g0


def test_diag_groups_keeps_a_whole_binding_ahead_of_a_deep_tail():
    """A cap keeps the low-index corners of the early groups, not one
    group's deep tail — the truncation-fairness property."""
    groups = [[(v, a) for a in range(6)] for v in range(4)]
    got = list(vo._diag_groups(groups))
    assert got[:6] == [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)]
    # inside a 10-entry cap every group is represented — the old
    # shape-major order would have spent all ten on group 0's tail.
    assert len({v for v, _a in got[:10]}) == 4
    # every group's index-0 corner precedes any group's index-3 tail.
    assert got.index((3, 0)) < got.index((0, 3))


def test_synth_bases_interleaves_viewed_shapes():
    """The base generator interleaves viewed bindings instead of
    serving one shape's whole attr domain first.

    The ``sdpa_fold_add`` transpose operand enumerates shapes whose
    view table admits differing numbers of attr combos; under the old
    shape-major nesting the leading rank-1 ``(4,)`` spent 24
    guard-declined bases before the first rank-2 operand opened.  The
    diagonal opens the first rank-2 operand at base 1 and reaches the
    guard-satisfying corner (rank-2 ``K``, ``softmax`` over the last
    axis) at base 13 — inside the 360 default.
    """
    from catopt_core.laws import ALL_RULES

    rule = {r.name: r for r in ALL_RULES}["sdpa_fold_add"]
    bases = list(vo._synth_bases(rule.lhs, rule.rhs))
    # more than one viewed shape inside the first dozen bases.
    shapes = []
    for base, _us, _os in bases[:12]:
        sh = tuple(base["K"].typ.shape)
        if sh not in shapes:
            shapes.append(sh)
    assert len(shapes) > 1
    # the first rank-2 K opens at base 1, not after a whole rank-1
    # shape's 24 attr combos.
    assert len(bases[0][0]["K"].typ.shape) == 1
    assert len(bases[1][0]["K"].typ.shape) >= 2
    # the guard-satisfying corner (last-two transpose axes AND the
    # softmax axis at -1) is base 13.
    corner = next(
        i
        for i, (base, _us, _os) in enumerate(bases)
        if len(base["K"].typ.shape) >= 2
        and base.get("$attr:SD") == -1
        and {base.get("$attr:TD1"), base.get("$attr:TD2")} == {-2, -1}
    )
    assert corner == 13


# ---------------------------------------------------------------------------
#  The selective cap policy — escalate_limit
# ---------------------------------------------------------------------------


def test_guarded_cap_is_past_the_default():
    """The escalation ceiling sits above the default window — a
    still-starved guarded rule (a shape/kind gap no cap reaches) pays
    it, so it must stay above the default."""
    assert vo._MAX_INSTANCES < vo._GUARDED_CAP


def test_escalate_limit_starved_guarded_rule_escalates():
    """A guarded rule whose window accepted nothing escalates."""
    assert (
        vo.escalate_limit(vo._MAX_INSTANCES, guarded=True, accepted=0)
        == vo._GUARDED_CAP
    )


def test_escalate_limit_leaves_the_common_case_alone():
    """An unguarded rule (no guard region to starve) and a guarded
    rule whose window already accepted a site keep the default."""
    assert (
        vo.escalate_limit(vo._MAX_INSTANCES, guarded=False, accepted=0)
        == vo._MAX_INSTANCES
    )
    assert (
        vo.escalate_limit(vo._MAX_INSTANCES, guarded=True, accepted=1)
        == vo._MAX_INSTANCES
    )
    assert (
        vo.escalate_limit(vo._MAX_INSTANCES, guarded=True, accepted=53)
        == vo._MAX_INSTANCES
    )


def test_escalate_limit_never_lowers_and_never_raises_past_ceiling():
    """The returned cap is never below *limit*; a window already at or
    past the ceiling is unchanged."""
    assert vo.escalate_limit(4000, guarded=True, accepted=0) == 4000
    assert (
        vo.escalate_limit(vo._GUARDED_CAP, guarded=True, accepted=0)
        == vo._GUARDED_CAP
    )


# ---------------------------------------------------------------------------
#  The starved folds surface at the default cap — the ordering fix
# ---------------------------------------------------------------------------
#
#  ``sdpa_fold_add`` / ``_addmul`` / ``_adddiv`` measured 0 accepted
#  sites at the 360 default under the old shape-major base order (their
#  equal corners sat at index 468 / 792 — ``cap-policy.md``).  The
#  (viewed x attr) diagonal surfaces them inside the default window, so
#  the selective-cap escalation is no longer paid for them.


def _first_equal_index(rule, limit: int) -> int | None:
    """Enumeration index of the first guard-accepted equal site."""
    from catopt_discovery import evidence as ev

    for idx, (subst, lhs_i) in enumerate(
        ev._synth_sites(rule.lhs, rule.rhs, limit=limit)
    ):
        if ev._site_outcome(rule, subst, lhs_i) == "equal":
            return idx
    return None


@pytest.mark.parametrize(
    "name, first_equal",
    [
        ("sdpa_fold_add", 139),
        ("sdpa_fold_addmul", 327),
        ("sdpa_fold_adddiv", 327),
    ],
)
def test_sdpa_folds_measure_at_the_default_cap(name, first_equal):
    """The starved folds now accept and prove inside the 360 window,
    at the measured index the diagonal reaches their corner."""
    from catopt_core.laws import ALL_RULES
    from catopt_discovery import evidence as ev

    rule = {r.name: r for r in ALL_RULES}[name]
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=360)
    )
    assert region.accepted > 0
    assert region.equal > 0
    assert region.unequal == 0 and region.rhs_err == 0
    assert region.declined > 0
    assert region.envs == 360
    assert _first_equal_index(rule, 360) == first_equal


def test_sdpa_fold_drop_twins_surface_too():
    """The ``_drop`` twins (the extra dropout metavars) surfaced at the
    same index as their bare forms — the extra free dimension is
    diagonalized, not left to bury the corner."""
    from catopt_core.laws import ALL_RULES
    from catopt_discovery import evidence as ev

    for name in ("sdpa_fold_add_drop", "sdpa_fold_addmul_drop"):
        rule = {r.name: r for r in ALL_RULES}[name]
        region = ev._guarded_evals(
            rule, ev._synth_sites(rule.lhs, rule.rhs, limit=360)
        )
        assert region.accepted > 0, name
        assert region.equal > 0, name
        assert region.unequal == 0 and region.rhs_err == 0, name


# ---------------------------------------------------------------------------
#  synthesize — enumeration, skip and veto paths
# ---------------------------------------------------------------------------


def test_synthesize_reshape_exercises_targets():
    insts = vo.synthesize(
        _p("mul", _p("reshape", "U", shape="S"), "V"),
        _p("mul", "U", "V"),
        limit=200,
    )
    assert insts
    assert all(i.origin == "synth" for i in insts)
    assert {i.outcome for i in insts} <= {
        "equal",
        "unequal",
        "lhs-err",
        "rhs-err",
        "both-err",
        "env-err",
    }
    # the reshape no-op feature was computed on tensor u.
    assert any(dict(i.feats).get("rs:noop") is True for i in insts)


def test_synthesize_conflicting_shared_attrs_are_skipped():
    """Two nodes sharing *part* of an attr-name set collide.

    ``select(u, dim=A, index=B)`` and ``unsqueeze(v, dim=A)`` form
    two groups (names ``A|B`` and ``A``); when their option draws
    disagree on ``A`` the combo is skipped — the survivors all keep
    ``A`` consistent.
    """
    insts = vo.synthesize(
        _p(
            "add",
            _p("select", "U", dim="A", index="B"),
            _p("unsqueeze", "V", dim="A"),
        ),
        _p("add", "U", "V"),
        limit=160,
    )
    assert insts
    for i in insts:
        binds = dict(i.binds)
        assert "$attr:A" in binds and "$attr:B" in binds


def test_synthesize_unbound_attr_metavar_leaks_literal():
    """An attr metavar no option dict produces is NOT a veto.

    ``_term_instantiate`` leaves an unbound attr metavariable in
    place — the minted instance carries ``extra=E`` verbatim (the
    matcher never sees it; only the eval does).  Honest pin of the
    current behavior: it is not flagged ill-formed.
    """
    insts = vo.synthesize(
        _p(
            "add",
            _p("getitem", "U", index="I", extra="E"),
            "V",
        ),
        _p("add", "U", "V"),
        limit=50,
    )
    assert insts
    assert all("extra=E" in i.lhs_repr for i in insts)


def test_synthesize_respects_limit():
    insts = vo.synthesize(
        _p("mul", "U", "V"), _p("add", "U", "V"), limit=3
    )
    assert 0 < len(insts) <= 3


def test_synthesize_tuple_producers_bind_u():
    insts = vo.synthesize(
        _p("add", _p("getitem", "U", index="I"), "V"),
        _p("getitem", _p("add", "U", "V"), index="I"),
        limit=160,
    )
    assert insts
    # at least one binding made U tuple-valued.
    assert any(dict(i.feats).get("u_tuple") is True for i in insts)


# ---------------------------------------------------------------------------
#  Mechanical guard features — direct calls
# ---------------------------------------------------------------------------


def _feats(lhs_pat, rhs_pat, subst, outcome="equal"):
    lhs_i = _term_instantiate(lhs_pat, subst)
    rhs_i = (
        _term_instantiate(rhs_pat, subst)
        if isinstance(rhs_pat, Op)
        else subst.get(rhs_pat, rhs_pat)
    )
    return vo._features(lhs_pat, rhs_pat, subst, lhs_i, rhs_i, outcome)


def test_features_env_none():
    x = Var("x", TensorType((None,)))
    subst = {"U": x, "V": _v("v", 4)}
    feats = _feats(
        _p("mul", _p("select", "U", dim="D", index="I"), "V"),
        _p("mul", "U", "V"),
        subst,
    )
    assert feats == {}


def test_features_no_view_nodes():
    # ``v`` in the feature block is the first free metavar — here U
    # (a (1,1) tensor): scalar-shaped but not 0-dim.
    subst = {"U": _v("u", 1, 1), "V": _v("v", 2, 2)}
    feats = _feats(_p("mul", "U", "V"), _p("mul", "V", "U"), subst)
    assert feats["u_tuple"] is False
    assert feats["v_scalar"] is False
    assert feats["v_uniform"] is True
    assert not any(k.startswith("id:") for k in feats)


def test_features_mixed_literal_and_metavar_attrs():
    """A view node may mix a literal attr with a metavariable."""
    insts = vo.synthesize(
        _p(
            "mul",
            _p("select", "U", dim=0, index="I"),
            "V",
        ),
        _p(
            "select",
            _p("mul", "U", "V"),
            dim=0,
            index="I",
        ),
        limit=80,
    )
    assert insts
    assert all("dim=0" in i.lhs_repr for i in insts)


def test_features_u_eval_failure():
    # U bound to a term that cannot evaluate -> u is None -> early
    # return with only the shape/leaf features.
    v = _v("v", 2, 2)
    subst = {"U": _p("frobnicate", v), "V": v}
    feats = _feats(
        _p("mul", _p("select", "U", dim="D", index="I"), "V"),
        _p("mul", "U", "V"),
        subst,
    )
    assert feats["u_tuple"] is False
    assert feats["u_shape"] == "?"
    assert "id:out_shape_eq" not in feats


def test_features_tuple_u_skips_tensor_noops():
    w = _v("w", 2, 4)
    subst = {
        "U": _p("topk", w, k=2),
        "V": _v("v", 4),
        "$attr:I": 0,
    }
    feats = _feats(
        _p("add", _p("getitem", "U", index="I"), "V"),
        _p("getitem", _p("add", "U", "V"), index="I"),
        subst,
    )
    assert feats["u_tuple"] is True
    # tuple u skips the no-op/covering block and the id/w family.
    assert "tr:noop" not in feats


def test_features_tr_noop_literal_and_metavar():
    u, v = _v("u", 2, 3), _v("v", 1)
    lhs = _p("mul", _p("transpose", "U", dim0=0, dim1="D"), "V")
    rhs = _p("mul", "U", "V")
    subst = {"U": u, "V": v, "$attr:D": 0}
    feats = _feats(lhs, rhs, subst)
    assert feats["tr:noop"] is True
    subst = {"U": u, "V": v, "$attr:D": 1}
    feats = _feats(lhs, rhs, subst)
    assert feats["tr:noop"] is False


def test_features_rs_noop_and_sl_full_and_ck_single():
    u, v = _v("u", 2, 3), _v("v", 2, 3)
    rhs = _p("mul", "U", "V")
    base = {"U": u, "V": v}
    f = _feats(
        _p("mul", _p("reshape", "U", shape="S"), "V"),
        rhs,
        {**base, "$attr:S": (2, 3)},
    )
    assert f["rs:noop"] is True
    f = _feats(
        _p("mul", _p("reshape", "U", shape="S"), "V"),
        rhs,
        {**base, "$attr:S": (6,)},
    )
    assert f["rs:noop"] is False
    f = _feats(
        _p(
            "mul",
            _p("slice", "U", dim="D", start="S0", end="E"),
            "V",
        ),
        rhs,
        {**base, "$attr:D": 0, "$attr:S0": 0, "$attr:E": 2},
    )
    assert f["sl:full"] is True
    f = _feats(
        _p(
            "mul",
            _p("slice", "U", dim="D", start="S0", end="E"),
            "V",
        ),
        rhs,
        {**base, "$attr:D": 0, "$attr:S0": 1, "$attr:E": 2},
    )
    assert f["sl:full"] is False
    f = _feats(
        _p(
            "mul",
            _p("chunk", "U", chunks="C", dim="D", index="I"),
            "V",
        ),
        rhs,
        {**base, "$attr:C": 1, "$attr:D": 0, "$attr:I": 0},
    )
    assert f["ck:single"] is True
    f = _feats(
        _p(
            "mul",
            _p("chunk", "U", chunks="C", dim="D", index="I"),
            "V",
        ),
        rhs,
        {**base, "$attr:C": 2, "$attr:D": 0, "$attr:I": 0},
    )
    assert f["ck:single"] is False


def test_features_unsq_d_in_pad_bound():
    u, v = _v("u", 3), _v("v", 3, 2)
    feats = _feats(
        _p("mul", _p("unsqueeze", "U", dim="D"), "V"),
        _p("mul", "U", "V"),
        {"U": u, "V": v, "$attr:D": 0},
    )
    assert feats["unsq:d_in_pad"] is True
    feats = _feats(
        _p("mul", _p("unsqueeze", "U", dim="D"), "V"),
        _p("mul", "U", "V"),
        {"U": u, "V": v, "$attr:D": 1},
    )
    assert feats["unsq:d_in_pad"] is False


def test_features_non_uv_metavar_skips_shape_feats():
    subst = {
        "U": _v("u", 2, 3),
        "W": _v("w", 2, 3),
        "$attr:D": 0,
        "$attr:I": 0,
    }
    feats = _feats(
        _p("mul", _p("select", "U", dim="D", index="I"), "W"),
        _p("mul", "U", "W"),
        subst,
    )
    # W is not a U/V feature metavar — only u's shape is recorded.
    assert "u_shape" in feats and "w_shape" not in feats


def test_features_non_int_dim_attrs_skip_noop():
    """A bound attr metavariable that is not an int defeats the
    no-op checks without failing."""
    u, v = _v("u", 2, 3), _v("v", 2, 3)
    rhs = _p("mul", "U", "V")
    feats = _feats(
        _p("mul", _p("transpose", "U", dim0="A", dim1="B"), "V"),
        rhs,
        {"U": u, "V": v, "$attr:A": 0, "$attr:B": (0,)},
    )
    assert "tr:noop" not in feats
    feats = _feats(
        _p(
            "mul",
            _p("slice", "U", dim="D", start="S0", end="E"),
            "V",
        ),
        rhs,
        {
            "U": u,
            "V": v,
            "$attr:D": (0,),
            "$attr:S0": 0,
            "$attr:E": 2,
        },
    )
    assert "sl:full" not in feats
    feats = _feats(
        _p("mul", _p("unsqueeze", "U", dim="D"), "V"),
        rhs,
        {"U": u, "V": v, "$attr:D": (0,)},
    )
    assert "unsq:d_in_pad" not in feats


def test_features_rhs_neither_family():
    """An RHS that is neither the pointwise op nor the view op gets
    no id/w features."""
    u, v = _v("u", 2, 3), _v("v", 2, 3)
    feats = _feats(
        _p("mul", _p("transpose", "U", dim0="A", dim1="B"), "V"),
        _p("add", "U", "V"),
        {"U": u, "V": v, "$attr:A": 0, "$attr:B": 1},
    )
    assert not any(
        k.startswith("id:") or k.startswith("w:") for k in feats
    )


def test_features_g_out_eval_failure():
    # The view node cannot evaluate ("view" is not bound) so g_out
    # is None — no family features are attempted.
    u, v = _v("u", 2, 3), _v("v", 2, 3)
    feats = _feats(
        _p("mul", _p("view", "U", shape="S"), "V"),
        _p("view", _p("mul", "U", "V"), shape="S"),
        {"U": u, "V": v, "$attr:S": (6,)},
    )
    assert "w:v_commutes_view" not in feats


def test_feats_id_broadcast_failure():
    g_u = torch.zeros(3, 1)
    feats = vo._feats_id(torch.randn(3), g_u, torch.randn(5))
    assert feats == {}
    # incompatible broadcast grids -> no pairing feature at all
    u_t, v_t = torch.randn(2, 3), torch.randn(5)
    assert vo._feats_id(u_t, torch.randn(2, 3), v_t) == {}


def test_feats_w_view_call_raises():
    node = _p("view", "U", shape=(6,))
    feats = vo._feats_w(
        node, torch.randn(2, 3), torch.randn(1), torch.randn(6)
    )
    assert feats == {}


# ---------------------------------------------------------------------------
#  sweep_real — check / derive / dedup / u_kind paths
# ---------------------------------------------------------------------------


def test_sweep_real_skips_and_vetoes():
    u, v = _v("u", 3), _v("v", 3)
    lhs_pat = _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V")
    match = _p("mul", _p("unsqueeze", u, dim=1), v)
    # a term that does not match the pattern is skipped silently.
    insts = vo.sweep_real(
        "t", lhs_pat, _p("mul", "U", "V"), [_p("add", u, v), match]
    )
    assert len(insts) == 1
    # a check that vetoes and a check that raises both skip.
    assert (
        vo.sweep_real(
            "t",
            lhs_pat,
            _p("mul", "U", "V"),
            [match],
            check=lambda bound: False,
        )
        == []
    )
    assert (
        vo.sweep_real(
            "t",
            lhs_pat,
            _p("mul", "U", "V"),
            [match],
            check=lambda bound: 1 / 0,
        )
        == []
    )
    # a derive returning None, or raising, skips the match.
    assert (
        vo.sweep_real(
            "t",
            lhs_pat,
            _p("mul", "U", "V"),
            [match],
            derive=lambda bound: None,
        )
        == []
    )
    assert (
        vo.sweep_real(
            "t",
            lhs_pat,
            _p("mul", "U", "V"),
            [match],
            derive=lambda bound: 1 / 0,
        )
        == []
    )


def test_sweep_real_derive_supplies_and_instantiate_fails():
    u, v = _v("u", 3), _v("v", 3)
    match = _p("mul", _p("unsqueeze", u, dim=1), v)
    lhs_pat = _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V")
    # derive fills an extra binding the RHS needs.
    insts = vo.sweep_real(
        "t",
        lhs_pat,
        _p("mul", "U", _p("unsqueeze", "V", dim="A_d2")),
        [match],
        derive=lambda bound: {"$attr:A_d2": 1},
    )
    assert len(insts) == 1
    # an RHS metavar the binding never supplies fails instantiation.
    assert (
        vo.sweep_real("t", lhs_pat, _p("mul", "U", "W"), [match]) == []
    )


def test_sweep_real_check_passes_through():
    u, v = _v("u", 3), _v("v", 3)
    match = _p("mul", _p("unsqueeze", u, dim=1), v)
    insts = vo.sweep_real(
        "t",
        _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        _p("mul", "U", "V"),
        [match],
        check=lambda bound: True,
    )
    assert len(insts) == 1


def test_sweep_real_u_term_eval_failure():
    """A bound U that cannot evaluate records an empty kind."""
    u, v = _v("u", 4), _v("v", 4)
    match = _p("mul", _p("frobnicate", u), v)
    insts = vo.sweep_real("t", _p("mul", "U", "V"), "V", [match])
    assert len(insts) == 1
    assert insts[0].outcome == "lhs-err"
    assert dict(insts[0].feats)["u_kind"] == ""


def test_sweep_real_dedups_identical_pairs():
    u, v = _v("u", 3), _v("v", 3)
    match = _p("mul", _p("unsqueeze", u, dim=1), v)
    insts = vo.sweep_real(
        "t",
        _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        _p("mul", "U", "V"),
        [match, match],
    )
    assert len(insts) == 1


def test_sweep_real_u_kind_and_env_none():
    w = _v("w", 2, 4)
    v = _v("v", 4)
    # tuple-valued U under getitem -> u_kind "tuple".
    match_t = _p("add", _p("getitem", _p("topk", w, k=2), index=0), v)
    insts = vo.sweep_real(
        "t",
        _p("add", _p("getitem", "U", index="I"), "V"),
        _p("add", "U", "V"),
        [match_t],
    )
    assert dict(insts[0].feats)["u_kind"] == "tuple"
    # tensor U -> "tensor".
    u = _v("u", 4)
    match_x = _p("add", _p("getitem", u, index=0), v)
    insts = vo.sweep_real(
        "t",
        _p("add", _p("getitem", "U", index="I"), "V"),
        _p("add", "U", "V"),
        [match_x],
    )
    assert dict(insts[0].feats)["u_kind"] == "tensor"
    # a pattern without "U" records an empty kind.
    insts = vo.sweep_real(
        "t", _p("add", "P", "Q"), _p("add", "Q", "P"), [match_x]
    )
    assert dict(insts[0].feats)["u_kind"] == ""
    # a match whose leaves carry non-int dims env-errs and skips the
    # kind probe.
    bad = Var("b", TensorType((None,)))
    match_bad = _p("mul", bad, v)
    insts = vo.sweep_real(
        "t", _p("mul", "U", "V"), _p("mul", "V", "U"), [match_bad]
    )
    assert insts[0].outcome == "env-err"
    assert dict(insts[0].feats)["u_kind"] == ""


# ---------------------------------------------------------------------------
#  Verdicts — the rows the headline file did not hit
# ---------------------------------------------------------------------------


def test_verdict_true_when_every_instance_agrees():
    v = vo.verify_view_candidate(
        "id_mul_one",
        _p("mul", "U", Const(1)),
        "U",
        [],
        synth_limit=120,
    )
    assert v.verdict == "true"
    assert v.synth_equal > 0
    assert v.synth_unequal == 0
    assert v.witness


def test_verdict_ill_formed_when_rhs_cannot_denote():
    """reshape to a wrong numel: the target never evaluates."""
    v = vo.verify_view_candidate(
        "ill",
        _p("mul", _p("reshape", "U", shape="S"), "V"),
        _p("reshape", _p("mul", "U", "V"), shape=(999,)),
        [],
        synth_limit=120,
    )
    assert v.verdict == "ill-formed"
    assert v.synth_equal == 0 and v.synth_unequal == 0
    assert v.synth_rhs_err > 0
    assert "does not denote" in v.note


def test_verdict_conditional_rhs_welltypedness():
    """Equal everywhere it is typed; tuple U mints an ill-typed RHS."""
    v = vo.verify_view_candidate(
        "sub_gi",
        _p(
            "sub",
            _p("getitem", "U", index="I"),
            _p("getitem", "U", index="I"),
        ),
        _p("getitem", _p("sub", "U", "U"), index="I"),
        [],
        synth_limit=120,
    )
    assert v.verdict == "conditional"
    assert v.synth_unequal == 0
    assert v.synth_equal > 0 and v.synth_rhs_err > 0
    assert "well-typedness" in v.guard


def test_verdict_unproven_when_nothing_evaluates():
    """Instances exist but every LHS errors — still 'unproven'."""
    u = _v("u", 4)
    v = _v("v", 4)
    match = _p("mul", _p("frobnicate", u, axis="A_axis"), v)
    verdict = vo.verify_view_candidate(
        "frob",
        _p("mul", _p("frobnicate", "U", axis="A_axis"), "V"),
        _p("mul", "U", "V"),
        [match],
    )
    assert verdict.verdict == "unproven"
    assert verdict.n_real == 1
    assert verdict.real_lhs_err == 1
    assert verdict.note == "no evaluable instance"


def test_verdict_counts_real_matches():
    """Real matches feed the counters, the witness and the note.

    ``unsqueeze(u, 0)`` is broadcast-transparent (equal wherever it
    types); ``unsqueeze(u, 1)`` against a column-shaped v disagrees.
    """
    u, v = _v("u", 3), _v("v", 4, 3)
    eq_match = _p("mul", _p("unsqueeze", u, dim=0), v)
    w = _v("w", 3, 1)
    neq_match = _p("mul", _p("unsqueeze", u, dim=1), w)
    rerr_match = _p("mul", _p("unsqueeze", u, dim=1), _v("y", 3, 2))
    verdict = vo.verify_view_candidate(
        "t",
        _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        _p("mul", "U", "V"),
        [eq_match, neq_match, rerr_match],
        synth_limit=80,
    )
    assert verdict.n_real == 3
    assert verdict.real_equal == 1
    assert verdict.real_unequal == 1
    assert verdict.real_rhs_err == 1
    assert verdict.verdict == "conditional"
    assert verdict.witness and verdict.counterexample


def test_verdict_true_from_real_only():
    """An attr the domain cannot type kills synthesis; a real match
    still carries the verdict.

    ``attn_mask`` is a tensor-valued attr — outside the scalar
    domains the sweep enumerates — so the ``sdpa`` node's metavar
    vetoes every synthesized binding, honestly.
    """
    q, k, v = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4)
    match = _p("sdpa", q, k, v, attn_mask=None)
    verdict = vo.verify_view_candidate(
        "t",
        _p("sdpa", "Q", "K", "V", attn_mask="AM"),
        _p("sdpa", "Q", "K", "V", attn_mask="AM"),
        [match],
    )
    assert verdict.verdict == "true"
    assert verdict.n_synth == 0
    assert verdict.real_equal == 1
    assert verdict.witness


def test_verdict_false_counterexample_from_real():
    """The real sweep supplies the counterexample when nothing
    synthesizes (same unenumerable-attr shape as the twin above)."""
    q, k, v = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4)
    match = _p("sdpa", q, k, v, attn_mask=None)
    verdict = vo.verify_view_candidate(
        "t",
        _p("sdpa", "Q", "K", "V", attn_mask="AM"),
        _p("mul", "Q", Const(2)),
        [match],
    )
    assert verdict.verdict == "false"
    assert verdict.real_unequal == 1
    assert verdict.counterexample


# ---------------------------------------------------------------------------
#  _separating_feature — fabricated instance sets
# ---------------------------------------------------------------------------


def _inst(outcome: str, feats: dict, tag: str = "") -> vo.Instance:
    return vo.Instance(
        origin="synth",
        outcome=outcome,
        lhs_repr=f"l{tag}",
        rhs_repr=f"r{tag}",
        binds=(("t", tag),),
        feats=tuple(sorted(feats.items())),
    )


def test_separating_feature_single():
    insts = [
        _inst("equal", {"a": True}, "1"),
        _inst("equal", {"a": True}, "2"),
        _inst("unequal", {"a": False}, "3"),
        _inst("unequal", {}, "4"),  # absent counts as False
    ]
    assert vo._separating_feature(insts) == "a"


def test_separating_feature_conjunction():
    insts = [
        _inst("equal", {"a": True, "b": True}, "1"),
        _inst("equal", {"a": True, "b": True}, "2"),
        _inst("unequal", {"a": True, "b": False}, "3"),
        _inst("unequal", {"a": False, "b": True}, "4"),
    ]
    assert vo._separating_feature(insts) == "a ∧ b"


def test_separating_feature_none_and_no_evaluable():
    insts = [
        _inst("equal", {"a": True}, "1"),
        _inst("unequal", {"a": True}, "2"),
    ]
    assert vo._separating_feature(insts) == ""
    assert vo._separating_feature([_inst("rhs-err", {}, "1")]) == ""


# ---------------------------------------------------------------------------
#  Driver — _run / _table / main
# ---------------------------------------------------------------------------


def test_run_driver(monkeypatch):
    """``_run`` filters proposals to the view family and verdicts
    them — here over a one-proposal corpus."""
    from catopt_discovery import pipeline as pl
    from catopt_discovery.impact import TermCase

    u = _v("u", 3, 1)
    case = TermCase("bench", "t", _p("mul", u, Const(1)), (u,), (), {})
    proposal = pl.Proposal(
        name="mul_sel",
        family="t",
        lhs=_p(
            "mul",
            _p("select", "U", dim="A_dim", index="A_index"),
            "V",
        ),
        rhs=_p("mul", "U", "V"),
    )
    monkeypatch.setattr(
        "catopt_discovery.census.run_census",
        lambda top: {"op_tuples": []},
    )
    monkeypatch.setattr(
        "catopt_discovery.impact._bench_cases", lambda: ([case], [])
    )
    monkeypatch.setattr(
        "catopt_discovery.impact.model_cases", lambda: ([], [])
    )
    monkeypatch.setattr(
        "catopt_discovery.intake.load_cases", lambda: []
    )
    monkeypatch.setattr(
        pl,
        "propose",
        lambda census_op, terms, vocab: [
            proposal,
            # a non-view proposal is filtered out by _run.
            pl.Proposal(
                name="plain",
                family="t",
                lhs=_p("mul", "U", Const(1)),
                rhs="U",
            ),
        ],
    )
    verdicts = vo._run()
    assert [v.name for v in verdicts] == ["mul_sel"]
    assert verdicts[0].verdict in ("false", "conditional", "true")
    text = vo._table(verdicts)
    assert "mul_sel" in text and "verdict" in text


def test_main_json(monkeypatch, tmp_path, capsys):
    verdict = vo.ViewVerdict(name="t", verdict="true", note="ok")
    monkeypatch.setattr(vo, "_run", lambda: [verdict])
    out = tmp_path / "o.json"
    rc = vo.main(["--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload[0]["name"] == "t"
    assert payload[0]["verdict"] == "true"
    printed = capsys.readouterr().out
    assert "view/index candidate resolution" in printed
    assert "wrote" in printed
    # without --json nothing is written; the table still prints.
    rc = vo.main([])
    assert rc == 0


# ---------------------------------------------------------------------------
#  Empty guarded regions that are enumeration gaps, not empty regions
# ---------------------------------------------------------------------------
#
#  Two shipped guarded laws measure ``synth: 0 accepted`` not because
#  their guard is unsatisfiable but because the enumeration never mints
#  the binding the guard needs.  Both guards accept a hand-built
#  binding, so the region is non-empty; the gap is in the enumerator's
#  attr domains, which is out of this change's scope (see
#  ``project/retros/guard-residuals.md``).


def test_gqa_absorb_repeat_guard_accepts_a_chained_binding():
    """The empty ``gqa_absorb_repeat`` synth window is an ENUMERATION
    gap, not an empty region.

    The guard's clauses are satisfiable: a hand-built rank-4
    ``(b, t, h, d)`` binding with ``unsqueeze(-2) -> expand(r at -2)
    -> reshape(merge)`` chains and ``q[-2] == k[-2] * r`` clears.  The
    enumeration mints it now: the operand-chained
    ``unsqueeze -> expand -> reshape`` nodes form one chained attr
    group (:func:`catopt_discovery.oracle._chain_domain`), so the
    ``expand``/``reshape`` options are drawn against the real
    intermediate shapes — and the pinned rank-4 corner evaluates
    ``equal``.  (The earlier rank-3 spelling — ``q=(2,6,4)``,
    ``k=v=(2,3,4)`` — satisfied the repeat clauses but its
    ``enable_gqa`` RHS cannot contract; the guard's ``rank == 4``
    clauses now decline it — see ``test_cond_laws``.)
    """
    from catopt_core.laws import ALL_RULES

    rule = next(r for r in ALL_RULES if r.name == "gqa_absorb_repeat")
    bound = {
        "q": _v("q", 2, 3, 4, 4),
        "k": _v("k", 2, 3, 2, 4),
        "v": _v("v", 2, 3, 2, 4),
        "$attr:UDk": -2,
        "$attr:ESk": (2, 3, 2, 2, 4),
        "$attr:RSk": (2, 3, 4, 4),
        "$attr:UDv": -2,
        "$attr:ESv": (2, 3, 2, 2, 4),
        "$attr:RSv": (2, 3, 4, 4),
        "$attr:D": -1,
        "$attr:C": False,
    }
    assert rule.check(bound)
    # the rank-3 spelling the chained enumeration can also mint is
    # declined — its enable_gqa RHS cannot contract (rhs-err).
    assert not rule.check(
        {
            "q": _v("q", 2, 6, 4),
            "k": _v("k", 2, 3, 4),
            "v": _v("v", 2, 3, 4),
            "$attr:UDk": -2,
            "$attr:ESk": (2, 3, 2, 4),
            "$attr:RSk": (2, 6, 4),
            "$attr:UDv": -2,
            "$attr:ESv": (2, 3, 2, 4),
            "$attr:RSv": (2, 6, 4),
            "$attr:D": -1,
            "$attr:C": False,
        }
    )

    # The chained domain contains the consistent triples — the
    # enumerator mints what it previously could not.
    nodes = vo._view_nodes([rule.lhs, rule.rhs])
    viewed = {
        "k": _v("k", 2, 3, 2, 4),
        "q": _v("q", 2, 3, 4, 4),
        "v": _v("v", 2, 3, 2, 4),
    }
    domains = vo._attr_domains(nodes, viewed)
    assert len(domains) == 3  # the {C, D} pair and the two chains
    k_chain = {
        "UDk": -2,
        "ESk": (2, 3, 2, 2, 4),
        "RSk": (2, 3, 4, 4),
    }
    v_chain = {
        "UDv": -2,
        "ESv": (2, 3, 2, 2, 4),
        "RSv": (2, 3, 4, 4),
    }
    assert any(
        all(o.get(m) == val for m, val in k_chain.items())
        for o in domains[1][1] + domains[2][1]
    )
    assert any(
        all(o.get(m) == val for m, val in v_chain.items())
        for o in domains[1][1] + domains[2][1]
    )

    # The corner binding passes the guard and evaluates equal.
    for combo in vo._diag_product([d[1] for d in domains]):
        base = vo._attr_merge(domains, combo, viewed)
        if base is None:
            continue
        if not all(
            base.get(f"$attr:{m}") == val
            for m, val in {**k_chain, **v_chain}.items()
        ):
            continue
        if base.get("$attr:D") != 1e-5 or base.get("$attr:C"):
            continue
        assert rule.check(base)
        lhs_i = _term_instantiate(rule.lhs, base)
        rhs_i = _term_instantiate(rule.rhs, base)
        torch.manual_seed(0)
        assert vo.eval_instance(lhs_i, rhs_i)[0] == "equal"
        break
    else:
        raise AssertionError("chained corner not enumerated")


def test_rms_norm_fold_guard_accepts_a_tail_block_binding():
    """The ``tail-block`` spec is correct: the gained fold's guard
    accepts ``u=(2,3,4)``, ``w=(4,)`` with ``MD`` naming ``u``'s last
    axis (and the ``(0,1,2)`` spelling with ``w=(2,3,4)``).

    The ``0/6000`` synth measurement is an enumeration-ordering
    artifact — the four-free-operand diagonal buries the
    non-scalar-``u`` corner (all 94 front-passing sites at 6000 have
    ``u=()``) — not a mis-specified guard.
    """
    from catopt_core.laws import ALL_RULES

    rules = {r.name: r for r in ALL_RULES}
    gained = rules["rms_norm_fold"]
    nogain = rules["rms_norm_fold_nogain"]
    base = {
        "u": _v("u", 2, 3, 4),
        "w": _v("w", 4),
        "EPS": Const(0.5),
        "P": Const(2),
        "$attr:MK": True,
        "$attr:MD": -1,
    }
    assert gained.check(base)
    assert nogain.check(base)
    multi = dict(base, w=_v("w", 2, 3, 4), **{"$attr:MD": (0, 1, 2)})
    assert gained.check(multi)
    # a scalar u has no trailing block — the guard declines.
    assert not gained.check(dict(base, u=_v("u")))
