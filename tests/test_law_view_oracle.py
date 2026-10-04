"""Tests for the view/index oracle (``catopt_discovery.oracle``).

The oracle resolves law-pipeline candidates the single-instance
numeric oracle cannot: view/index identities whose truth depends on
shape and index conditions.  These tests pin the verdicts on
hand-built candidates whose truth is known:

* ``mul(select(u,d,i), v) -> mul(u,v)`` — false (the strip drops the
  selected axis);
* ``mul(select(u,d,i), v) -> select(mul(u,v),d,i)`` — the naturality,
  conditional on the free operand commuting with the view;
* ``mul(unsqueeze(u,d), v) -> mul(u,v)`` — conditional (true when the
  inserted axis lands in the broadcast pad);
* ``add(getitem(u,i), v) -> add(u,v)`` — false; the ``_w`` variant is
  conditional and its tuple-valued ``u`` region is ill-typed;
* ``mul(transpose(u,d0,d1), v) -> mul(u,v)`` — conditional on
  ``d0 == d1`` (the no-op transpose).

The oracle lives in the ``catopt_discovery`` package.
"""

from catopt_core.ir import Const, Op, TensorType, Var
from catopt_discovery import oracle as vo


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


# -- tri-state eval ---------------------------------------------------------


def test_eval_instance_equal():
    x = _v("x", 2, 3)
    outcome, _ = vo.eval_instance(
        _p("mul", x, Const(2)), _p("mul", Const(2), x)
    )
    assert outcome == "equal"


def test_eval_instance_unequal():
    x = _v("x", 2, 3)
    outcome, _ = vo.eval_instance(_p("mul", x, x), _p("add", x, x))
    assert outcome == "unequal"


def test_eval_instance_rhs_err():
    # unsq(u,1) broadcasts against (3,2); bare u=(3,) does not.
    x, y = _v("x", 3), _v("y", 3, 2)
    outcome, _ = vo.eval_instance(
        _p("mul", _p("unsqueeze", x, dim=1), y), _p("mul", x, y)
    )
    assert outcome == "rhs-err"


# -- verdict classification -------------------------------------------------


def test_select_strip_is_false():
    """``mul(select(u,d,i),v) -> mul(u,v)`` holds nowhere testable."""
    v = vo.verify_view_candidate(
        "mul_select_l_id",
        _p(
            "mul",
            _p("select", "U", dim="A_dim", index="A_index"),
            "V",
        ),
        _p("mul", "U", "V"),
        [],
        synth_limit=240,
    )
    assert v.verdict == "false"
    assert v.synth_unequal > 0
    assert v.synth_equal == 0


def test_select_naturality_is_conditional():
    """``mul(select(u),v) -> select(mul(u,v))`` is the naturality."""
    v = vo.verify_view_candidate(
        "mul_select_l_w",
        _p(
            "mul",
            _p("select", "U", dim="A_dim", index="A_index"),
            "V",
        ),
        _p(
            "select",
            _p("mul", "U", "V"),
            dim="A_dim",
            index="A_index",
        ),
        [],
        synth_limit=240,
    )
    assert v.verdict == "conditional"
    assert v.synth_equal > 0
    assert v.guard == "w:v_commutes_view"


def test_unsqueeze_strip_is_conditional():
    """``mul(unsqueeze(u,d),v) -> mul(u,v)`` holds when the inserted
    axis is a broadcast pad."""
    v = vo.verify_view_candidate(
        "mul_unsqueeze_l_id",
        _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        _p("mul", "U", "V"),
        [],
        synth_limit=240,
    )
    assert v.verdict == "conditional"
    assert v.synth_equal > 0
    assert v.synth_unequal > 0
    # The separating guard is the pairing/out-shape conjunction.
    assert "id:" in v.guard


def test_getitem_strip_is_false():
    """``add(getitem(u,i),v) -> add(u,v)`` — false on tensor ``u``,
    ill-typed on tuple ``u``; never equal."""
    v = vo.verify_view_candidate(
        "add_getitem_l_id",
        _p("add", _p("getitem", "U", index="A_index"), "V"),
        _p("add", "U", "V"),
        [],
        synth_limit=200,
    )
    assert v.verdict == "false"
    assert v.synth_equal == 0
    # Tuple-valued U makes the RHS un-evaluable — recorded as such.
    assert v.synth_rhs_err > 0


def test_getitem_naturality_is_conditional():
    """``add(getitem(u,i),v) -> getitem(add(u,v),i)`` — the dim-0
    naturality for tensor ``u``; tuple ``u`` is ill-typed."""
    v = vo.verify_view_candidate(
        "add_getitem_l_w",
        _p("add", _p("getitem", "U", index="A_index"), "V"),
        _p("getitem", _p("add", "U", "V"), index="A_index"),
        [],
        synth_limit=200,
    )
    assert v.verdict == "conditional"
    assert v.synth_equal > 0


def test_transpose_strip_is_conditional_on_noop():
    """``mul(transpose(u,d0,d1),v) -> mul(u,v)`` only when d0 == d1."""
    v = vo.verify_view_candidate(
        "mul_transpose_l_id",
        _p(
            "mul",
            _p("transpose", "U", dim0="A_d0", dim1="A_d1"),
            "V",
        ),
        _p("mul", "U", "V"),
        [],
        synth_limit=240,
    )
    assert v.verdict == "conditional"
    assert v.guard in ("tr:noop", "id:same_pairing")


# -- mechanical features ----------------------------------------------------


def test_w_feature_predicts_equality():
    """``w:v_commutes_view`` coincides with the equal set."""
    insts = vo.synthesize(
        _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        _p("unsqueeze", _p("mul", "U", "V"), dim="A_dim"),
        limit=240,
    )
    both = [i for i in insts if i.outcome in ("equal", "unequal")]
    assert both
    for i in both:
        feats = dict(i.feats)
        if "w:v_commutes_view" in feats:
            assert feats["w:v_commutes_view"] == (
                i.outcome == "equal"
            ), i.binds


def test_d_in_pad_marks_equal_region():
    """The unsqueeze pad feature is necessary for the strip to hold."""
    insts = vo.synthesize(
        _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        _p("mul", "U", "V"),
        limit=240,
    )
    for i in insts:
        if i.outcome == "equal":
            feats = dict(i.feats)
            assert feats.get("unsq:d_in_pad") is not False


def test_sweep_real_is_tri_state():
    """A real match whose instantiated RHS is ill-typed reports as
    ``rhs-err`` — not as a verdict."""
    u, v = _v("u", 3), _v("v", 3, 2)
    match = _p("mul", _p("unsqueeze", u, dim=1), v)
    insts = vo.sweep_real(
        "t",
        _p("mul", _p("unsqueeze", "U", dim="A_dim"), "V"),
        _p("mul", "U", "V"),
        [match],
    )
    assert len(insts) == 1
    assert insts[0].outcome == "rhs-err"


def test_unprovable_candidate_stays_unproven():
    """A pattern with an unbindable view op reports ``unproven``."""
    v = vo.verify_view_candidate(
        "mystery",
        _p("mul", _p("frobnicate", "U", axis="A_x"), "V"),
        _p("frobnicate", _p("mul", "U", "V"), axis="A_x"),
        [],
    )
    assert v.verdict == "unproven"
