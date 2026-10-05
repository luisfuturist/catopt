"""Tests for the auto-cond constructor and its admission wiring.

The real-corpus run's finding (project/retros/guide-real-run.md):
candidates reach arena ``SHIP`` but ``usable: yes`` stays 0 — they are
conditional equalities and admission requires the guard written as
declarative ``cond`` data.  ``object_synthesis.auto_cond_object`` is
the conditional→guarded step: measure the bare pattern's domain (the
oracle's synthesized envs at the sweep's cap plus the real matches),
enumerate the cond-DSL predicate bank over the pattern's metavariables
and attr metavariables, and pick the *smallest* conjunction covering
every measured equal site and declining every measured
``unequal``/``rhs-err`` site.  ``run_gauntlet(auto_cond=True)`` is the
additive admission hook: a truth (or missing-``check`` full-data)
refusal attempts the mint, rewrites the stored record, and re-runs the
gauntlet — the guarded object faces the same gates, nothing weakened.
"""

import torch
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_core.laws.cond import cond_to_data
from catopt_core.laws.serialize import law_from_data, missing_hooks
from catopt_discovery import evidence as ev
from catopt_discovery import object_synthesis as obs
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _case(name: str, term: Op, *inputs: Var) -> TermCase:
    """Probe-case mirroring the guide-corpus spelling."""
    torch.manual_seed(0)
    return TermCase(
        source="t",
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=tuple(
            torch.randn(tuple(x.typ.shape), dtype=torch.float64)
            for x in inputs
        ),
        param_vals={},
    )


def _corpus(*cases: TermCase) -> ev.GauntletCorpus:
    sink = _sink()
    return ev.GauntletCorpus(
        real_terms=tuple(c.term for c in cases),
        probe=tuple(cases),
        base_rules=tuple(ALL_RULES),
        census_op={},
        sink=sink,
        cost_fn=_cost_fn(sink),
    )


def _unsq_bare(name: str = "mul_unsqueeze_l_id") -> Rewrite:
    """The known conditional: strip ``unsqueeze`` off a mul operand."""
    return Rewrite(
        name=name,
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("mul", "U", "V"),
    )


def _guarded_case() -> TermCase:
    """``pad_after_flat``: ``u=(8,8)`` * ``v=(1,1,1)`` pad broadcast."""
    u, v = _v("u", 8, 8), _v("v", 1, 1, 1)
    return _case("unsq_pad", _p("mul", _p("unsqueeze", u, dim=0), v), u, v)


def _accepted_region(rule: Rewrite, limit: int = 360) -> ev.GuardedRegion:
    """The minted guard's sweep over the synthesized sites."""
    sites = list(ev._synth_sites(rule.lhs, rule.rhs, limit=limit))
    return ev._guarded_evals(rule, iter(sites))


def test_auto_cond_unsq_strip_finds_the_pad_guard() -> None:
    """The conditional strip gets a separating declarative guard."""
    torch.manual_seed(0)
    res = obs.auto_cond_object(_unsq_bare(), synth_limit=360)
    assert res.object is not None
    assert res.equal > 0 and res.bad > 0
    assert res.accepted >= res.equal
    assert res.object.kind == "abstraction"
    # pure data — the found cond is the whole claim
    assert missing_hooks(res.object.rule) == ()
    assert res.object.rule.cond is not None
    # the minted guard verifies on its own sweep: no bad site accepted
    region = _accepted_region(res.object.rule)
    assert region.equal > 0
    assert region.unequal == 0
    assert region.rhs_err == 0
    # every clause is load-bearing — drop one and a bad site re-enters
    clauses = res.clauses
    assert len(clauses) >= 1
    for i in range(len(clauses)):
        rest = tuple(c for j, c in enumerate(clauses) if j != i)
        weaker = Rewrite(
            name="weaker",
            lhs=res.object.rule.lhs,
            rhs=res.object.rule.rhs,
            cond=True if not rest else rest[0] if len(rest) == 1
            else ("and", *rest),
        )
        worse = _accepted_region(weaker)
        assert worse.unequal + worse.rhs_err > 0


def test_auto_cond_smallest_conjunct_refuses_below() -> None:
    """A 2-clause cover refuses when capped at one clause."""
    torch.manual_seed(0)
    res = obs.auto_cond_object(_unsq_bare(), synth_limit=360)
    assert res.object is not None and len(res.clauses) == 2
    smaller = obs.auto_cond_object(
        _unsq_bare(), synth_limit=360, max_clauses=len(res.clauses) - 1
    )
    assert smaller.object is None
    assert "no declarable conjunction" in smaller.detail


def test_auto_cond_g_twin_the_attr_case() -> None:
    """The ``_g`` twin — attr metavariables in the separating guard.

    The bare ``sdpa_fold_div_nomask_g`` pattern is conditional on the
    transpose axes: a no-op ``transpose`` pair (``(0, 0)`` on a square
    ``K``) spells ``Q@K`` while ``sdpa`` transposes internally.  The
    domain's union of the two enumeration windows sees the corner —
    the found guard is the last-two-axes check the shipped laws carry.
    """
    torch.manual_seed(0)
    bare = Rewrite(
        name="sdpa_fold_div_nomask_g",
        lhs=_p(
            "matmul",
            _p(
                "softmax",
                _p(
                    "div",
                    _p(
                        "matmul",
                        "Q",
                        _p("transpose", "K", dim0="TD1", dim1="TD2"),
                    ),
                    "S",
                ),
                dim=-1,
            ),
            "V",
        ),
        rhs=_p("sdpa", "Q", "K", "V", scale="SC"),
        dspec={"SC": ("recip", ("float", ("const", "S")))},
    )
    res = obs.auto_cond_object(bare, synth_limit=360)
    assert res.object is not None
    assert res.bad > 0 and res.equal > 0
    flat = repr(res.cond)
    assert "TD1" in flat and "TD2" in flat
    # the derived-scale machinery rides along; the record stays data
    assert res.object.rule.dspec == bare.dspec
    assert missing_hooks(res.object.rule) == ()
    region = _accepted_region(res.object.rule)
    assert region.equal > 0
    assert region.unequal == 0
    assert region.rhs_err == 0


def test_auto_cond_refuses_when_no_site_is_equal() -> None:
    """``mul -> add`` is never true under randn: no cover to mint."""
    torch.manual_seed(0)
    bare = Rewrite(
        name="mul_is_add", lhs=_p("mul", "U", "V"), rhs=_p("add", "U", "V")
    )
    res = obs.auto_cond_object(bare, synth_limit=360)
    assert res.object is None
    assert res.equal == 0
    assert "no equal site" in res.detail


def test_auto_cond_refuses_an_undeclarable_region() -> None:
    """A guard needing a shape no predicate can name is refused.

    ``reshape(mul(u, v), S) -> mul(u, v)`` holds iff ``S`` is the
    broadcast shape of the *compound* operand — the DSL's shape specs
    name metavariables and view outputs, not pattern-internal terms,
    so no bank predicate separates the classes.  The refusal reports
    the measured class sizes and the covering-set size honestly.
    """
    torch.manual_seed(0)
    bare = Rewrite(
        name="mul_reshape_wr",
        lhs=_p("mul", "U", "V"),
        rhs=_p("reshape", _p("mul", "U", "V"), shape="S"),
    )
    res = obs.auto_cond_object(bare, synth_limit=360)
    assert res.object is None
    assert res.equal > 0 and res.bad > 0
    assert "no declarable conjunction" in res.detail


def test_auto_cond_refuses_an_unenumerable_domain() -> None:
    """An attr metavar no op domain emits yields no synth domain."""
    torch.manual_seed(0)
    bare = Rewrite(
        name="mystery",
        lhs=_p("mul", _p("frobnicate", "U", axis="A_x"), "V"),
        rhs=_p("mul", "U", "V"),
    )
    res = obs.auto_cond_object(bare, synth_limit=360)
    assert res.object is None
    assert res.measured == 0
    assert "no evaluable site" in res.detail


def test_gauntlet_auto_cond_rewrites_and_admits() -> None:
    """The additive hook: refusal -> mint -> rewritten record -> admit."""
    torch.manual_seed(0)
    conn = ev.connect(":memory:")
    key = ev.store_object(conn, _unsq_bare(), kind="abstraction")
    corpus = _corpus(_guarded_case())
    rep = ev.run_gauntlet(conn, key, corpus=corpus)
    assert not rep.usable
    assert rep.stages[-1].name == "truth"

    rep2 = ev.run_gauntlet(conn, key, corpus=corpus, auto_cond=True)
    assert rep2.usable
    assert rep2.stages[0].name == "auto-cond"
    ac = rep2.auto_cond
    assert ac is not None and ac["found"]
    assert ac["first_reason"].startswith("truth:")
    assert ac["equal"] > 0 and ac["bad"] > 0
    # the record was rewritten under the same alpha key: declarative
    # cond present, kind kept as abstraction, fully serializable
    rec = ev.stored_object(conn, key)
    assert rec["kind"] == "abstraction"
    assert rec["cond"] is not None
    assert rec["serializable"]
    rule, rec2 = ev.admit_object(conn, key)
    rebuilt = law_from_data(rec2)
    assert rebuilt.cond is not None
    assert rebuilt.cond == rule.cond
    # the stored data round-trips through the declarative serializer
    assert cond_to_data(rule.cond) == rec2["cond"]
    # the guarded-region sweeps populated the report
    assert rep2.synth_region is not None
    assert rep2.synth_region.equal >= 1
    assert rep2.synth_region.unequal == 0
    assert rep2.real_region is not None
    assert rep2.real_region.equal == 1


def test_gauntlet_auto_cond_refusal_keeps_the_verdict() -> None:
    """A refused search changes nothing: the original verdict stands."""
    torch.manual_seed(0)
    bare = Rewrite(
        name="mul_is_add", lhs=_p("mul", "U", "V"), rhs=_p("add", "U", "V")
    )
    conn = ev.connect(":memory:")
    key = ev.store_object(conn, bare, kind="abstraction")
    u, v = _v("u", 4, 4), _v("v", 4, 4)
    corpus = _corpus(_case("mul", _p("mul", u, v), u, v))
    rep = ev.run_gauntlet(conn, key, corpus=corpus, auto_cond=True)
    assert not rep.usable
    assert rep.auto_cond is not None
    assert not rep.auto_cond["found"]
    assert "no equal site" in rep.auto_cond["detail"]
    # the record is untouched — still the bare pattern
    rec = ev.stored_object(conn, key)
    assert rec["cond"] is None


def test_gauntlet_auto_cond_via_missing_check() -> None:
    """The full-data trigger: missing ``check`` alone arms the retry.

    A stored procedural-check candidate reconstructs bare — the
    missing ``check`` is exactly the slot a found ``cond`` fills, so
    the record is rewritten and re-gated; the procedural hook that
    could not serialize is honestly absent from the minted object.
    """
    torch.manual_seed(0)
    proc = Rewrite(
        name="unsq_proc",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("mul", "U", "V"),
        check=lambda bound: True,
    )
    conn = ev.connect(":memory:")
    key = ev.store_object(conn, proc, kind="abstraction")
    assert ev.stored_object(conn, key)["missing_hooks"] == ["check"]
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_guarded_case()), auto_cond=True
    )
    assert rep.usable
    assert rep.auto_cond is not None and rep.auto_cond["found"]
    assert rep.auto_cond["first_reason"].startswith("full-data:")


def test_auto_cond_preserves_the_dspec_and_proc_derive() -> None:
    """Minting keeps declarative ``dspec`` and procedural ``derive``.

    A procedural ``derive`` is part of how the rule instantiates — the
    measured outcomes already reflect its vetoes — so the minted
    object carries it through and the store flags the missing hook.
    """
    torch.manual_seed(0)
    bare = Rewrite(
        name="unsq_derived",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("mul", "U", "V"),
        derive=lambda bound: None if bound.get("$attr:A_dim") == 0 else {},
    )
    res = obs.auto_cond_object(bare, synth_limit=360)
    assert res.object is not None
    assert res.object.rule.derive is bare.derive
    conn = ev.connect(":memory:")
    key = obs.store_constructed(conn, res.object, "c0")
    rec = ev.stored_object(conn, key)
    assert "derive" in rec["missing_hooks"]


def test_stable_outcome_declines_value_flips(monkeypatch) -> None:
    """A site whose verdict flips between draws is ``unstable``.

    ``eval_instance`` draws leaf values from ambient ``randn`` — a
    binding holding e.g. ``pow(param, -0.25)`` NaNs on a negative
    draw.  Two passes that disagree mark the site must-decline.
    """
    probe = _unsq_bare()
    calls = iter(["equal", "unequal"])
    monkeypatch.setattr(ev, "_site_outcome", lambda *a: next(calls))
    assert obs._stable_outcome(probe, {}, None) == "unstable"
    monkeypatch.setattr(ev, "_site_outcome", lambda *a: "equal")
    assert obs._stable_outcome(probe, {}, None) == "equal"
    monkeypatch.setattr(ev, "_site_outcome", lambda *a: "declined")
    assert obs._stable_outcome(probe, {}, None) == "declined"


def test_auto_cond_clause_set_is_minimal_in_size() -> None:
    """``_min_cover`` prefers the fewest clauses, ties by acceptance."""
    # 6 sites: eq = {0,1,2}, bad = {3,4,5}
    eq = 0b000111
    bad = 0b111000
    # pred A covers eq, kills site 3 only; B covers eq, kills 4,5
    useful = [
        (("a",), eq | 0b110000, 0b001000),
        (("b",), eq | 0b001000, 0b110000),
        (("c",), eq, bad),  # one clause kills everything
    ]
    got = obs._min_cover(useful, eq, bad, 6, 3)
    assert got == (("c",),)
    # drop c: a and b together cover, a alone doesn't
    got2 = obs._min_cover(useful[:2], eq, bad, 6, 3)
    assert sorted(map(repr, got2)) == sorted(
        map(repr, (("a",), ("b",)))
    )
    got3 = obs._min_cover(useful[:2], eq, bad, 6, 1)
    assert got3 is None


# ---------------------------------------------------------------------------
#  The widened bank — the view-commute vocabulary (cond-bank retro)
# ---------------------------------------------------------------------------
#
# The stock bank (``object_synthesis._pred_bank``) enumerates the cond
# DSL over the pattern's metavariables, but its view-output specs cover
# only unsqueeze/reshape/getitem and it never emits the *commutation*
# form — ``bcast(view_out(u), v) == view_out(bcast(u, v))`` — that the
# ``f(view(u), v) -> view(f(u, v))`` "wrap" candidates need.  cond.py
# gained the missing view-output specs (``select-out``/``slice-out``/
# ``chunk-out``/``transpose-out``) and the alignment atoms
# (``axis-align-eq``/``bcast-dim-inv``); the generator below emits the
# matching predicates.  The production bank in object_synthesis.py needs
# the same additive emission — wiring it in here measures the delta
# without touching that module.

#: Index views whose commutation guard needs the axis-alignment atoms.
_IDX_ATTR = {"select": "dim", "slice": "dim", "chunk": "dim"}


def _str_attrs(node: Op) -> dict:
    return {k: v for k, v in node.attrs.items() if isinstance(v, str)}


def _view_out_spec(op: str, operand, attrs: dict):
    """The view-output shape spec for *op*, or ``None`` when unbuildable."""
    if op == "unsqueeze" and isinstance(attrs.get("dim"), str):
        return ("unsq-out", operand, attrs["dim"])
    if op in ("reshape", "view") and isinstance(attrs.get("shape"), str):
        return ("reshape-out", operand, attrs["shape"])
    if op == "getitem":
        return ("getitem-out", operand)
    if op == "select" and isinstance(attrs.get("dim"), str):
        return ("select-out", operand, attrs["dim"])
    if op == "slice" and isinstance(attrs.get("dim"), str):
        return (
            "slice-out",
            operand,
            attrs["dim"],
            attrs.get("start"),
            attrs.get("end"),
            attrs.get("step"),
        )
    if op == "chunk" and isinstance(attrs.get("chunks"), str):
        return ("chunk-out", operand, attrs["chunks"], attrs.get("dim"))
    if op in ("transpose", "t") and isinstance(attrs.get("dim0"), str):
        return ("transpose-out", operand, attrs["dim0"], attrs.get("dim1"))
    return None


def _view_commute_preds(lhs, rhs) -> list:
    """The view-commute predicates the widened bank adds over a pattern."""
    from catopt_discovery import oracle as lvo

    mvs = obs._mv_names(lhs, rhs)
    nodes = [n for n in lvo._view_nodes([lhs, rhs]) if n.args]
    views = []
    out: list = []
    for n in nodes:
        u = n.args[0] if isinstance(n.args[0], str) else None
        if u is None:
            continue
        spec = _view_out_spec(n.op, u, _str_attrs(n))
        if spec is None:
            continue
        views.append((u, spec, n.op, _str_attrs(n)))
    for u, su, op, attrs in views:
        for v in mvs:
            if v == u:
                continue
            suv = _view_out_spec(op, ("bcast", u, v), attrs)
            if suv is None:
                continue
            out.append(("bcast-eq", su, v, suv, suv))
            out.append(("shaped", ("bcast", u, v)))
            k = _IDX_ATTR.get(op)
            if k is not None and k in attrs:
                out.append(("bcast-dim-inv", v, u, attrs[k]))
            if op in ("transpose", "t") and "dim0" in attrs:
                out.append(
                    (
                        "or",
                        ("axes-noop", v, attrs["dim0"], attrs["dim1"]),
                        ("rank", v, "<=", 1),
                    )
                )
        for w, sw, op2, attrs2 in views:
            if w == u or op2 != op or attrs2 != attrs:
                continue
            suw = _view_out_spec(op, ("bcast", u, w), attrs)
            if suw is None:
                continue
            out.append(("bcast-eq", su, sw, suw, suw))
            out.append(("shaped", ("bcast", u, w)))
            k = _IDX_ATTR.get(op)
            if k is not None and k in attrs:
                out.append(("axis-align-eq", u, w, attrs[k]))
    for n in nodes:
        inner = n.args[0] if n.args else None
        if not isinstance(inner, Op) or not inner.args:
            continue
        u = inner.args[0] if isinstance(inner.args[0], str) else None
        if u is None:
            continue
        inner_spec = _view_out_spec(inner.op, u, _str_attrs(inner))
        g1 = _view_out_spec(n.op, u, _str_attrs(n))
        so = _view_out_spec(n.op, inner_spec, _str_attrs(n))
        g2g1 = (
            _view_out_spec(inner.op, g1, _str_attrs(inner))
            if g1 is not None
            else None
        )
        if so is not None and g2g1 is not None:
            out.append(("shape-eq", so, g2g1))
    return out


def _install_view_commute_bank(monkeypatch) -> None:
    """Extend ``obs._pred_bank`` with the view-commute vocabulary."""
    stock = obs._pred_bank

    def bank(lhs, rhs, envs):
        return stock(lhs, rhs, envs) + _view_commute_preds(lhs, rhs)

    monkeypatch.setattr(obs, "_pred_bank", bank)


def _select_wrap_bare(name: str = "mul_select_l_w") -> Rewrite:
    """The index-view wrap: ``mul(select(U), V) -> select(mul(U, V))``."""
    return Rewrite(
        name=name,
        lhs=_p("mul", _p("select", "U", dim="A_dim", index="A_index"), "V"),
        rhs=_p(
            "select",
            _p("mul", "U", "V"),
            dim="A_dim",
            index="A_index",
        ),
    )


def test_view_commute_preds_name_the_wrap_guard() -> None:
    """The generator emits the select-wrap commutation predicate."""
    bare = _select_wrap_bare()
    preds = _view_commute_preds(bare.lhs, bare.rhs)
    assert ("bcast-dim-inv", "V", "U", "A_dim") in preds
    assert (
        "bcast-eq",
        ("select-out", "U", "A_dim"),
        "V",
        ("select-out", ("bcast", "U", "V"), "A_dim"),
        ("select-out", ("bcast", "U", "V"), "A_dim"),
    ) in preds


def test_auto_cond_extended_bank_admits_the_select_wrap(
    monkeypatch,
) -> None:
    """The widened bank mints the guard the stock bank cannot declare.

    The wrap form is equal iff the index view commutes with ``V``'s
    broadcast; the stock bank has no predicate over a ``select``'s
    output shape, so it refuses.  cond.py now carries ``select-out`` /
    ``bcast-dim-inv``; the extended bank emits the commutation guard
    and the constructor mints it — verified on its own sweep.
    """
    torch.manual_seed(0)
    bare = _select_wrap_bare()
    stock = obs.auto_cond_object(bare, synth_limit=360)
    assert stock.object is None
    assert stock.equal > 0 and stock.bad > 0
    assert "no declarable conjunction" in stock.detail

    _install_view_commute_bank(monkeypatch)
    res = obs.auto_cond_object(bare, synth_limit=360)
    assert res.object is not None
    assert res.equal > 0 and res.bad > 0
    assert res.object.kind == "abstraction"
    # pure data — the found cond is the whole claim
    assert missing_hooks(res.object.rule) == ()
    assert "select-out" in repr(res.cond)
    # the minted guard verifies on its own sweep: no bad site accepted
    region = _accepted_region(res.object.rule)
    assert region.equal > 0
    assert region.unequal == 0
    assert region.rhs_err == 0
