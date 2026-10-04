"""Tests for the typed-pay gate in ``tools/law_pipeline.py``.

``law_impact._probe`` counts a rule's e-graph fires and whether the
extracted cost dropped — but never asked whether the instantiated
RHS was *well-typed*.  A view/index candidate can mint a member that
does not denote (a broadcast mismatch, an attr ``%``-normalised out
of range, a tuple-valued operand read as a tensor); the member gets
priced — an unknown shape falls back to ~free — picked, and recorded
as a paying fire.  These tests pin the audit:

* ``_bad_subterms`` / ``_term_typed`` — the shape + eval verdict;
* ``_typed_probe`` — the per-fire ``typed`` / ``ill`` split plus the
  ill-typed-extraction check;
* ``_fire`` — ``fires`` unchanged, ``fires_ill_typed`` visible,
  ``paid``/``changed`` suppressed on an ill-typed pick, the
  suppression counted in ``paid_ill_typed``;
* ``_reach_row`` — a with-rule cost drop via an ill-typed member is
  ``add_typed=False``;
* ``Evidence.no_ship_reason`` — surfaces "pays only on ill-typed
  sites" / "mints ill-typed members".

The pipeline lives in ``catopt_discovery.pipeline`` (formerly
``tools/law_pipeline.py``).
"""

import torch
from catopt_core.ir import Op, TensorType, Var
from catopt_discovery import pipeline as pl
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_torch.adapters import TorchSink


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


_SINK = TorchSink()
_COST = _cost_fn(_SINK)


def _case(name: str, term: Op, *inputs: Var) -> TermCase:
    return TermCase(
        source="test",
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=tuple(
            torch.randn(tuple(v.typ.shape), dtype=torch.float64)
            for v in inputs
        ),
        param_vals={},
    )


# The ``mixed:add_getitem_l_id`` shape — a strip candidate whose RHS
# reads the tuple-valued ``var_mean`` output as a tensor.
_GETITEM_LHS = _p("add", _p("getitem", "U", index="A_i"), "V")
_GETITEM_RHS = _p("add", "U", "V")


def _getitem_strip() -> pl.Proposal:
    return pl.Proposal(
        name="t:add_getitem_id",
        lhs=_GETITEM_LHS,
        rhs=_GETITEM_RHS,
        family="test",
    )


def _getitem_case() -> TermCase:
    # var_mean(u) is a tuple; getitem(·, 0) reads the (4,1) var — the
    # input is well-typed.  The strip mints add(var_mean(u), v):
    # shape-checks (4,1), drops the costed getitem — and cannot
    # evaluate, since ``add`` gets a tuple.
    u, v = _v("u", 4, 4), _v("v", 4, 1)
    vm = _p("var_mean", u, dim=(-1,), correction=0, keepdim=True)
    term = _p("add", _p("getitem", vm, index=0), v)
    return _case("tuple_getitem", term, u, v)


# -- the typing predicates ---------------------------------------------------


def test_bad_subterms_flags_broadcast_mismatch():
    x, y = _v("x", 3), _v("y", 3, 2)
    assert pl._bad_subterms(_p("mul", x, y))
    # (4,) broadcast against unsqueezed (1,4) -> (1,1,4): resolves.
    a, b = _v("a", 4), _v("b", 1, 4)
    ok = _p("mul", a, _p("unsqueeze", b, dim=0))
    assert not pl._bad_subterms(ok)


def test_term_typed_shape_and_eval():
    x, y = _v("x", 4), _v("y", 1, 4)
    assert pl._term_typed(_p("mul", _p("unsqueeze", x, dim=0), y))
    # Shape inference % -normalises the out-of-range dim — the member
    # only fails at eval.
    bad_attr = _p("unsqueeze", x, dim=9)
    assert not pl._bad_subterms(bad_attr)
    assert not pl._term_typed(bad_attr)


def test_term_typed_tuple_operand():
    # add(var_mean(u), v) shape-resolves to (4,1) but evaluates to a
    # TypeError — the tuple-valued operand is ill-typed, not merely
    # false.
    u, v = _v("u", 4, 4), _v("v", 4, 1)
    vm = _p("var_mean", u, dim=(-1,), correction=0, keepdim=True)
    term = _p("add", vm, v)
    assert not pl._bad_subterms(term)
    assert not pl._term_typed(term)


# -- the per-fire audit -------------------------------------------------------


def test_typed_probe_counts_ill_typed_fire():
    audit = pl._typed_probe(_getitem_case(), _getitem_strip(), _COST)
    assert audit.fires == 1
    assert audit.typed == 0
    assert audit.ill == 1
    # The minted member is cheaper (it drops getitem) and gets
    # picked — an ill-typed extraction.
    assert audit.pick_ill


def test_typed_probe_well_typed_fire():
    u, v = _v("u", 4), _v("v", 1, 4)
    term = _p("mul", _p("unsqueeze", u, dim=0), v)
    proposal = pl.Proposal(
        name="t:mul_unsqueeze_w",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_d"), "V"),
        rhs=_p("unsqueeze", _p("mul", "U", "V"), dim="A_d"),
        family="test",
    )
    audit = pl._typed_probe(_case("typed", term, u, v), proposal, _COST)
    assert audit.fires >= 1
    assert audit.ill == 0
    assert audit.typed == audit.fires
    assert not audit.pick_ill


def test_fire_gate_suppresses_illusory_paid():
    proposal = _getitem_strip()
    ev = pl.Evidence(proposal=proposal)
    pl._fire(proposal, [_getitem_case()], _SINK, _COST, ev)
    assert ev.fires == 1
    assert ev.fires_typed == 0
    assert ev.fires_ill_typed == 1
    # The probe measured a real cost drop (the minted member drops a
    # costed op) — suppressed, not paid.
    assert ev.paid == 0
    assert ev.changed == 0
    assert ev.paid_ill_typed == 1
    assert ev.ill_typed_cases == ("tuple_getitem",)
    # The doomed lowering is not a verify disagreement.
    assert ev.verify_fail == 0


def test_reach_row_suppresses_ill_typed_drop():
    case = _getitem_case()
    rule = _getitem_strip().as_rule()
    row = pl._reach_row(case, [], [rule], _COST)
    assert row["add_cost"] < row["base_cost"]
    assert row["add_typed"] is False


def test_reach_row_typed_drop_stays():
    u, v = _v("u", 4), _v("v", 1, 4)
    term = _p("mul", _p("unsqueeze", u, dim=0), v)
    proposal = pl.Proposal(
        name="t:mul_unsqueeze_w",
        lhs=_p("mul", _p("unsqueeze", "U", dim="A_d"), "V"),
        rhs=_p("unsqueeze", _p("mul", "U", "V"), dim="A_d"),
        family="test",
    )
    row = pl._reach_row(
        _case("typed", term, u, v), [], [proposal.as_rule()], _COST
    )
    if row["add_cost"] < row["base_cost"]:
        assert row["add_typed"] is True


# -- the verdict surface ------------------------------------------------------


def _truthy_ev(**kw) -> pl.Evidence:
    ev = pl.Evidence(proposal=_getitem_strip())
    ev.num_true = True
    ev.relation = "new"
    for k, val in kw.items():
        setattr(ev, k, val)
    return ev


def test_no_ship_reason_pays_only_ill_typed():
    ev = _truthy_ev(fires=4, fires_ill_typed=4, paid_ill_typed=4)
    assert not ev.shippable
    assert "pays only on ill-typed sites" in ev.no_ship_reason


def test_no_ship_reason_all_fires_ill_typed():
    ev = _truthy_ev(fires=3, fires_ill_typed=3)
    assert not ev.shippable
    assert "ill-typed" in ev.no_ship_reason


def test_ship_gate_rejects_minted_ill_typed_members():
    # Even a candidate that pays on typed sites does not ship while
    # it also mints ill-typed members on real sites.
    ev = _truthy_ev(fires=4, fires_typed=3, fires_ill_typed=1, paid=1)
    assert not ev.shippable
    assert "mints ill-typed members" in ev.no_ship_reason


def test_typed_and_paying_is_still_shippable():
    ev = _truthy_ev(fires=3, fires_typed=3, paid=1)
    assert ev.shippable
    assert ev.no_ship_reason == ""
