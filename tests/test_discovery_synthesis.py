"""Tests for ``catopt_discovery.object_synthesis`` — constructed objects.

Stage 3 (``tests/test_discovery_gauntlet.py``) admitted objects the
pipeline had *found* — census/proposal candidates declared as data.
These tests admit objects the machinery *constructs* (ADR 0004's
action space — fold representation | introduce abstraction | compose
abstractions): :mod:`catopt_discovery.object_synthesis` builds the
record programmatically, the evidence store persists it, and the
adversarial gauntlet decides whether it is usable.  Construction is a
claim; the gauntlet is the referee — the honest-negatives below are
constructed objects the gates refuse.
"""

import pytest
import torch
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import ALL_RULES, serialize
from catopt_core.laws.cond import eval_cond
from catopt_core.typing import _shape_of
from catopt_discovery import evidence as ev
from catopt_discovery import object_synthesis as synth
from catopt_discovery import oracle as lvo
from catopt_discovery import shape_proposal
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink

_BY_NAME = {r.name: r for r in ALL_RULES}


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _case(name: str, term: Op, *inputs: Var) -> TermCase:
    return TermCase(
        source="test",
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
    """The tiny injected corpus, shaped like the real one."""
    sink = _sink()
    return ev.GauntletCorpus(
        real_terms=tuple(c.term for c in cases),
        probe=tuple(cases),
        base_rules=tuple(ALL_RULES),
        census_op={},
        sink=sink,
        cost_fn=_cost_fn(sink),
    )


def _stages(rep: ev.Gauntlet) -> dict[str, ev.GauntletStage]:
    return {s.name: s for s in rep.stages}


# ---------------------------------------------------------------------------
#  The constructed objects — one per construction operation
# ---------------------------------------------------------------------------


def _softsign_fold() -> synth.ConstructedObject:
    """fold: ``x / (|x| + 1)`` → the fused ``softsign`` kernel.

    The ``nn.Softsign`` intake term spells this out — a real firing
    site — and no shipped law knows the fold.
    """
    return synth.fold_object(
        "softsign_fold",
        ("div", "X", ("add", ("abs", "X"), 1)),
        "softsign",
    )


def _aff_step_lift() -> synth.ConstructedObject:
    """lift: the affine recurrence step into the scan carrier.

    ``add(matmul(A,h),x) -> apply(aff(A,x),h)`` — the runtime
    abstraction the carrier machinery hardcodes, declared as data.
    """
    return synth.lift_object(
        "aff_step_lift",
        ("add", ("matmul", "A", "h"), "x"),
        ("aff", "A", "x"),
        "apply",
        state="h",
    )


def _aff_scan2() -> synth.ConstructedObject:
    """compose: two fused affine steps — one carrier composition.

    ``compose(aff_lift, aff_lift)`` with the first premise's state
    metavar specialized to a second step: the two-block scan lift —
    an object no shipped ruleset contains.
    """
    step = ("add", ("matmul", "A1", "h"), "x1")
    obj = synth.compose_objects(
        "aff_scan2_lift",
        _aff_step_lift(),
        _aff_step_lift(),
        specialize={"A": "A2", "x": "x2", "h": step},
    )
    assert obj is not None
    return obj


def _silu_fold_commuted() -> synth.ConstructedObject:
    """compose: the swapped SiLU spelling, proven by shipped rules.

    ``mul(sigmoid(X), X) -> silu(X)`` — the composite of ``comm_mul``
    (specialized) and ``silu_fold``; ``derivation`` names both, so the
    record can carry a replayable certificate.
    """
    obj = synth.compose_objects(
        "silu_fold_commuted",
        _BY_NAME["comm_mul"],
        _BY_NAME["silu_fold"],
        specialize={"a": ("sigmoid", "X"), "b": "X"},
    )
    assert obj is not None
    return obj


# ---------------------------------------------------------------------------
#  Stage-5 objects — deeper constructions
# ---------------------------------------------------------------------------

#: The transposed operand must swap K's last two axes — the
#: ``_COND_SCORE_T`` fragment of the shipped ``sdpa_fold_*`` guards,
#: restated as data.  (``softmax`` keeps a concrete ``dim=-1`` and the
#: scale is literal — stylistic choice; a str attr on a non-view op
#: is enumerable since the generic attr domain landed.)
_COND_LAST2 = (
    "and",
    ("concrete", "K"),
    ("attr-type", "TD1", "int"),
    ("attr-type", "TD2", "int"),
    ("axes-last2", "K", "TD1", "TD2"),
)


def _sdpa_fold_nomask() -> synth.ConstructedObject:
    """fold (spec RHS): the mask-free attention-score block.

    ``matmul(softmax(q @ kᵀ, -1), v) -> sdpa(q, k, v, scale=1)`` —
    every shipped ``sdpa_fold_*`` requires an ``add``/``masked_fill``
    mask operand; the mask-free spelling exists in no ruleset, yet
    ``MultiheadAttention`` / ``HybridBlock`` / ``TwoLayerHybrid``
    spell it.  Guarded on the transpose axes; the literal scale keeps
    the RHS free of attr metavars so the guarded sweep can enumerate.
    """
    return synth.fold_object(
        "sdpa_fold_nomask",
        (
            "matmul",
            (
                "softmax",
                (
                    "matmul",
                    "Q",
                    ("transpose", "K", {"dim0": "TD1", "dim1": "TD2"}),
                ),
                {"dim": -1},
            ),
            "V",
        ),
        ("sdpa", "Q", "K", "V", {"scale": 1.0}),
        cond=_COND_LAST2,
        tags=("fusion",),
    )


def _sdpa_fold_div_nomask() -> synth.ConstructedObject:
    """fold (spec RHS + dspec): the 1/√d-scaled mask-free block.

    ``matmul(softmax(q@kᵀ / S, -1), v) -> sdpa(q,k,v, scale=1/S)`` —
    ``ManualAttention``'s spelling verbatim.  The transpose axes are
    literal (the pattern pins the last-two swap — no axes cond
    needed) and the scale is *derived*: the ``dspec`` float-veto
    declines a non-``Const`` S at fire time.  What the veto does not
    cover is rank — ``sdpa`` cannot denote below rank 2 while the
    ``matmul`` spelling still evaluates a vector LHS, so once the
    sweep enumerated the non-view ``scale`` metavar (attr-sweep
    retro) the rank-1 bindings measured ``rhs-err`` and the raw
    verdict went honestly ``conditional``.  ``cond`` declares the
    operand ranks the target needs.
    """
    return synth.fold_object(
        "sdpa_fold_div_nomask",
        (
            "matmul",
            (
                "softmax",
                (
                    "div",
                    (
                        "matmul",
                        "Q",
                        ("transpose", "K", {"dim0": -1, "dim1": -2}),
                    ),
                    "S",
                ),
                {"dim": -1},
            ),
            "V",
        ),
        ("sdpa", "Q", "K", "V", {"scale": "SC"}),
        cond=(
            "and",
            ("rank", "Q", ">=", 2),
            ("rank", "K", ">=", 2),
            ("rank", "V", ">=", 2),
        ),
        dspec={"SC": ("recip", ("float", ("const", "S")))},
        tags=("fusion",),
    )


#: The state-shape economy guard the shipped diagonal lifts carry
#: (``scan.py``'s ``_COND_AFFD_STATE``): the bound ``h`` must be a
#: spine/apply or a leaf — not a per-step input vector.
_COND_AFFD_STATE = (
    "or",
    ("op-in", "h", ("add", "sub", "apply", "applyd")),
    ("leaf", "h"),
)


def _affd_step_lift() -> synth.ConstructedObject:
    """lift: the diagonal-affine step into the scan carrier.

    ``add(mul(a,h),x) -> applyd(aff_diag(a,x),h)`` — the Mamba-typed
    selective step the corpus's SSM models spell; a guarded lift
    declared as data (the shipped ``affd_lift`` lives in
    ``SCAN_DIAG_LAWS`` — the constructor's own premise here).
    """
    return synth.lift_object(
        "affd_step_lift",
        ("add", ("mul", "a", "h"), "x"),
        ("aff_diag", "a", "x"),
        "applyd",
        state="h",
        cond=_COND_AFFD_STATE,
    )


def _affd_scan2() -> synth.ConstructedObject:
    """compose: the two-block diagonal scan — in no shipped ruleset.

    ``compose(affd_step_lift, affd_step_lift)`` specialized on the
    inner step: the guarded premise evaluates on the symbolic binding
    (the ``h`` metavar reads as a leaf) and the composite carries the
    state guard as its own ``cond``.
    """
    obj = synth.compose_objects(
        "affd_scan2_lift",
        _affd_step_lift(),
        _affd_step_lift(),
        specialize={
            "a": "a2",
            "x": "x2",
            "h": ("add", ("mul", "a1", "h"), "x1"),
        },
        cond=_COND_AFFD_STATE,
        tags=("carrier",),
    )
    assert obj is not None
    return obj


def _affd_scan4() -> synth.ConstructedObject:
    """compose∘compose — the four-block object over unshipped edges.

    ``compose(affd_scan2_lift, affd_scan2_lift)``: BOTH premises are
    constructed objects (the derivation names ``affd_scan2_lift``
    alone) — a self-composition in which no edge is a shipped law.
    """
    obj = synth.compose_objects(
        "affd_scan4_lift",
        _affd_scan2(),
        _affd_scan2(),
        specialize={
            "h": (
                "add",
                ("mul", "a0", ("add", ("mul", "am1", "h0"), "xm1")),
                "x0",
            ),
        },
        cond=_COND_AFFD_STATE,
        tags=("carrier",),
    )
    assert obj is not None
    return obj


def _softsign_case() -> TermCase:
    x = _v("x", 4, 8)
    return _case(
        "softsign_site",
        _p("div", x, _p("add", _p("abs", x), Const(1))),
        x,
    )


def _step_case() -> TermCase:
    a, h, x = _v("a", 4, 4), _v("h", 4), _v("x", 4)
    return _case(
        "affine_step", _p("add", _p("matmul", a, h), x), a, h, x
    )


def _scan2_case() -> TermCase:
    a1, a2 = _v("a1", 4, 4), _v("a2", 4, 4)
    h, x1, x2 = _v("h", 4), _v("x1", 4), _v("x2", 4)
    inner = _p("add", _p("matmul", a1, h), x1)
    term = _p("add", _p("matmul", a2, inner), x2)
    return _case("two_step", term, a1, a2, h, x1, x2)


def _silu_swap_case() -> TermCase:
    x = _v("x", 4, 4)
    return _case("silu_swapped", _p("mul", _p("sigmoid", x), x), x)


def _attn_nomask_case() -> TermCase:
    """``matmul(softmax(q@kᵀ, -1), v)`` — the unscaled spelled block."""
    q, k, v = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4)
    scores = _p("matmul", q, _p("transpose", k, dim0=-1, dim1=-2))
    term = _p("matmul", _p("softmax", scores, dim=-1), v)
    return _case("attn_nomask", term, q, k, v)


def _attn_div_nomask_case() -> TermCase:
    """``matmul(softmax(q@kᵀ / 4.0, -1), v)`` — ManualAttention's form."""
    q, k, v = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4)
    scores = _p(
        "div",
        _p("matmul", q, _p("transpose", k, dim0=-1, dim1=-2)),
        Const(4.0),
    )
    term = _p("matmul", _p("softmax", scores, dim=-1), v)
    return _case("attn_div_nomask", term, q, k, v)


def _affd_step_case() -> TermCase:
    a, h, x = _v("a", 4), _v("h", 4), _v("x", 4)
    return _case("affd_step", _p("add", _p("mul", a, h), x), a, h, x)


def _affd_scan2_case() -> TermCase:
    a1, a2 = _v("a1", 4), _v("a2", 4)
    h, x1, x2 = _v("h", 4), _v("x1", 4), _v("x2", 4)
    inner = _p("add", _p("mul", a1, h), x1)
    term = _p("add", _p("mul", a2, inner), x2)
    return _case("affd_scan2", term, a1, a2, h, x1, x2)


def _affd_scan4_case() -> TermCase:
    vs = [_v(n, 4) for n in ("a1", "a2", "a3", "a4", "h")]
    xs = [_v(n, 4) for n in ("x1", "x2", "x3", "x4")]
    a1, a2, a3, a4, h = vs
    term = _p("add", _p("mul", a1, h), xs[0])
    for a, x in zip((a2, a3, a4), xs[1:], strict=True):
        term = _p("add", _p("mul", a, term), x)
    return _case("affd_scan4", term, *vs, *xs)


# ---------------------------------------------------------------------------
#  The spec grammar
# ---------------------------------------------------------------------------


def test_term_from_spec():
    spec = ("div", "X", ("add", ("abs", "X"), 1))
    term = synth.term_from_spec(spec)
    assert term == _p("div", "X", _p("add", _p("abs", "X"), Const(1)))
    # A trailing dict is the attr map.
    t2 = synth.term_from_spec(("softmax", "S", {"dim": -1}))
    assert t2 == _p("softmax", "S", dim=-1)


def test_constructed_objects_are_full_data(tmp_path):
    """Every constructed record serializes completely — a declared
    object is data the store can replay, not a Python-hook claim."""
    conn = ev.connect(str(tmp_path / "s.db"))
    for obj in (
        _softsign_fold(),
        _aff_step_lift(),
        _aff_scan2(),
        _silu_fold_commuted(),
    ):
        key = synth.store_constructed(conn, obj)
        record = ev.stored_object(conn, key)
        assert record["kind"] == "abstraction"
        assert record["serializable"] is True
        assert record["missing_hooks"] == []
    conn.close()


# ---------------------------------------------------------------------------
#  Fold — the fused-kernel abstraction
# ---------------------------------------------------------------------------


def test_fold_object_clears_the_gauntlet(tmp_path):
    """Post-promotion, ``softsign_fold`` IS the shipped law — the
    store's novelty gate reports the declared object as a library
    duplicate (``project/retros/promoted-laws.md``).  The measurement
    underneath is unchanged: the numeric oracle still finds it true
    and paying."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _softsign_fold())
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_softsign_case()))
    conn.close()
    assert not rep.usable
    assert rep.name == "softsign_fold" and rep.kind == "abstraction"
    assert rep.reason.startswith("novelty:")
    assert rep.evidence.relation == "duplicate"
    # Unguarded object: truth is the numeric oracle on the real match.
    assert rep.evidence.num_true is True
    assert rep.evidence.fires >= 1 and rep.evidence.paid == 1


def test_fold_object_fires_and_behaves(tmp_path):
    """store → admit → apply: the admitted fold merges the spelled
    site with the kernel; the extracted program is well-typed and
    numerically identical."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _softsign_fold())
    rule, record = ev.admit_object(conn, key)
    conn.close()
    assert record["serializable"] is True
    x = _v("x", 4, 8)
    src = _p("div", x, _p("add", _p("abs", x), Const(1)))
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    assert eg.rule_fires.get(rule.name) == 1
    assert eg.find(root) == eg.find(eg.add_term(_p("softsign", x)))
    # The admitted transform is well-typed and evaluates identically.
    env = {x: torch.randn(4, 8, dtype=torch.float64)}
    out, _ = lvo.eval_instance(src, _p("softsign", x))
    assert out == "equal" and env
    s = _shape_of(_p("softsign", x))
    assert s == (4, 8)


# ---------------------------------------------------------------------------
#  Lift — the carrier abstraction declared as data
# ---------------------------------------------------------------------------


def test_lift_object_clears_the_gauntlet(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _aff_step_lift())
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_step_case()))
    conn.close()
    assert rep.usable, rep.reason
    assert rep.kind == "abstraction"
    assert rep.evidence.paid == 1  # the carrier form prices cheaper
    assert "carrier" in rep.rule.tags


def test_lift_object_introduces_the_carrier_on_terms(tmp_path):
    """The admitted lift fires on a spelled recurrence step and mints
    the carrier term — ``apply(aff(a,x),h)`` — well-typed."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _aff_step_lift())
    rule, _rec = ev.admit_object(conn, key)
    conn.close()
    a, h, x = _v("a", 4, 4), _v("h", 4), _v("x", 4)
    src = _p("add", _p("matmul", a, h), x)
    dst = _p("apply", _p("aff", a, x), h)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    assert eg.rule_fires.get(rule.name) == 1
    assert eg.find(root) == eg.find(eg.add_term(dst))
    assert _shape_of(dst) == (4,)


# ---------------------------------------------------------------------------
#  Compose — the composite object (and its certificate)
# ---------------------------------------------------------------------------


def test_compose_builds_the_two_step_carrier_object():
    """The composition is mechanical: ``aff_lift`` applied to its own
    specialized RHS mints the two-step carrier form."""
    obj = _aff_scan2()
    assert obj.construction == ("compose", "aff_step_lift")
    assert obj.rule.derivation == ("aff_step_lift",)
    assert obj.rule.lhs == _p(
        "add",
        _p("matmul", "A2", _p("add", _p("matmul", "A1", "h"), "x1")),
        "x2",
    )
    assert obj.rule.rhs == _p(
        "apply",
        _p("aff", "A2", "x2"),
        _p("apply", _p("aff", "A1", "x1"), "h"),
    )


def test_composed_carrier_object_clears_the_gauntlet(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _aff_scan2())
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_scan2_case()))
    conn.close()
    assert rep.usable, rep.reason
    assert rep.evidence.num_true is True
    assert rep.evidence.paid == 1
    # Carrier premises are not ALL_RULES members — no derivation
    # cert could be stored; the gate reports the absence honestly.
    assert _stages(rep)["cert"].detail == "no derivation recorded"


def test_composed_object_over_shipped_rules_carries_a_cert(tmp_path):
    """compose(comm_mul, silu_fold): the composite names shipped
    premises, so the record embeds a replayable 2-step certificate —
    the first object whose cert gate does real work."""
    conn = ev.connect(str(tmp_path / "s.db"))
    obj = _silu_fold_commuted()
    assert obj.rule.derivation == ("comm_mul", "silu_fold")
    key = synth.store_constructed(conn, obj, env={"X": _v("X", 4, 4)})
    record = ev.stored_object(conn, key)
    assert record["cert"] is not None
    assert record["cert"]["rules_used"] == ["comm_mul", "silu_fold"]
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_silu_swap_case()))
    conn.close()
    assert rep.usable, rep.reason
    assert rep.evidence.derivable  # provable under the base rules
    assert "replays strict" in _stages(rep)["cert"].detail


def test_composed_object_fires_and_behaves(tmp_path):
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(
        conn, _silu_fold_commuted(), env={"X": _v("X", 4, 4)}
    )
    rule, _rec = ev.admit_object(conn, key)
    conn.close()
    x = _v("x", 4, 4)
    src = _p("mul", _p("sigmoid", x), x)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    assert eg.rule_fires.get(rule.name) == 1
    assert eg.find(root) == eg.find(eg.add_term(_p("silu", x)))
    # The unswapped spelling is untouched — the object is precise.
    eg2 = EGraph()
    root2 = eg2.add_term(_p("mul", x, _p("sigmoid", x)))
    eg2.run([rule], root2, max_iterations=4, max_nodes=10_000)
    assert not eg2.rule_fires


# ---------------------------------------------------------------------------
#  Honest negatives — construction does not buy admission
# ---------------------------------------------------------------------------


def test_compose_fails_when_no_premise_fires():
    """compose(comm_mul, silu_fold) without the specialization: the
    fold never fires on ``mul(b, a)`` — the operation declines."""
    obj = synth.compose_objects(
        "no_compose", _BY_NAME["comm_mul"], _BY_NAME["silu_fold"]
    )
    assert obj is None


def test_lifted_om_object_clears_the_gauntlet(tmp_path):
    """The online-softmax monoid lift, declared as data:
    ``matmul(softmax(S,-1),V) -> om_apply(om_elem(S,V))`` — the
    ADR's flagship carrier example, admitted ``usable``."""
    conn = ev.connect(str(tmp_path / "s.db"))
    obj = synth.lift_object(
        "om_lift",
        ("matmul", ("softmax", "S", {"dim": -1}), "V"),
        ("om_elem", "S", "V"),
        "om_apply",
    )
    key = synth.store_constructed(conn, obj)
    s, v = _v("s", 8, 8), _v("v", 8, 4)
    case = _case(
        "attn",
        _p("matmul", _p("softmax", s, dim=-1), v),
        s,
        v,
    )
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(case))
    conn.close()
    assert rep.usable, rep.reason
    assert rep.kind == "abstraction"
    assert rep.evidence.num_true is True
    assert rep.evidence.paid == 1
    assert rep.evidence.verify_fail == 0


def test_composed_object_that_does_not_pay_fails(tmp_path):
    """``pow(x,2) -> mul(x,x)`` — a real composite of shipped premises
    (derivable, cert-carrying) — still fails typed-pay: the mul
    spelling prices *above* pow under the generic cost model."""
    conn = ev.connect(str(tmp_path / "s.db"))
    obj = synth.compose_objects(
        "pow2_expand",
        _BY_NAME["pow_to_square"],
        _BY_NAME["square_expand"],
    )
    assert obj is not None
    key = synth.store_constructed(conn, obj)
    x = _v("x", 4, 4)
    case = _case("pow2", _p("pow", x, Const(2)), x)
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(case))
    conn.close()
    assert not rep.usable
    stages = _stages(rep)
    assert stages["truth"].passed
    assert not stages["typed-pay"].passed


def test_constructed_falsehood_fails_truth(tmp_path):
    """A constructed object that is false is refused like any other:
    construction is a claim, not a credential."""
    conn = ev.connect(str(tmp_path / "s.db"))
    bad = synth.fold_object(
        "abs_is_square", ("abs", "X"), "square", arg="X"
    )
    key = synth.store_constructed(conn, bad)
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_softsign_case()))
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("truth:") or rep.reason.startswith(
        "typed-pay:"
    )


# ---------------------------------------------------------------------------
#  Spec-RHS folds — objects whose dispatched form is not a unary kernel
# ---------------------------------------------------------------------------


def test_fold_spec_rhs_is_full_data(tmp_path):
    """The multi-arg/attr fold serializes completely — cond and dspec
    are pure data, so the record carries every hook."""
    conn = ev.connect(str(tmp_path / "s.db"))
    for obj in (
        _sdpa_fold_nomask(),
        _sdpa_fold_div_nomask(),
        _affd_step_lift(),
        _affd_scan2(),
        _affd_scan4(),
    ):
        key = synth.store_constructed(conn, obj)
        record = ev.stored_object(conn, key)
        assert record["kind"] == "abstraction"
        assert record["serializable"] is True
        assert record["missing_hooks"] == []
    conn.close()


def test_sdpa_fold_nomask_clears_the_gauntlet(tmp_path):
    """The mask-free score block: guarded, sweep-verified, pays."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _sdpa_fold_nomask())
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_attn_nomask_case())
    )
    conn.close()
    assert rep.usable, rep.reason
    # Guarded truth: the synthesized region and the real match both
    # verify equal wherever the transpose-axes guard accepts.
    assert rep.synth_region.equal >= 1
    assert rep.real_region.equal >= 1
    assert rep.synth_region.unequal == rep.real_region.unequal == 0
    assert rep.evidence.fires >= 1 and rep.evidence.paid >= 1


def test_sdpa_fold_nomask_fires_and_behaves(tmp_path):
    """store → admit → apply: the admitted fold merges the spelled
    block with ``sdpa(q,k,v,scale=1)`` — numerically identical."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _sdpa_fold_nomask())
    rule, record = ev.admit_object(conn, key)
    conn.close()
    assert record["serializable"] is True
    q, k, v = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4)
    scores = _p("matmul", q, _p("transpose", k, dim0=-1, dim1=-2))
    src = _p("matmul", _p("softmax", scores, dim=-1), v)
    dst = _p("sdpa", q, k, v, scale=1.0)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    assert eg.rule_fires.get(rule.name) == 1
    assert eg.find(root) == eg.find(eg.add_term(dst))
    out, _ = lvo.eval_instance(src, dst)
    assert out == "equal"
    # A non-last-two transpose declines the guard — no mint.
    k3 = _v("k3", 2, 2, 4)
    bad = _p(
        "matmul",
        _p(
            "softmax",
            _p("matmul", q, _p("transpose", k3, dim0=0, dim1=1)),
            dim=-1,
        ),
        v,
    )
    eg2 = EGraph()
    root2 = eg2.add_term(bad)
    eg2.run([rule], root2, max_iterations=4, max_nodes=10_000)
    assert not eg2.rule_fires


def test_sdpa_fold_div_nomask_clears_the_gauntlet(tmp_path):
    """The scaled mask-free block — dspec float-veto plus the rank
    precondition ``sdpa`` needs.  The raw oracle correctly stays
    ``conditional`` (rank-1 mints); the guarded sweep verifies the
    declared region."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _sdpa_fold_div_nomask())
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_attn_div_nomask_case())
    )
    conn.close()
    assert rep.usable, rep.reason
    assert rep.synth_region.equal >= 1
    assert rep.real_region.equal >= 1
    assert rep.synth_region.unequal == rep.real_region.unequal == 0
    assert rep.evidence.paid >= 1


def test_sdpa_fold_div_nomask_derive_vetoes_nonconst_scale(tmp_path):
    """``div(scores, s)`` with a *tensor* divisor cannot be an sdpa
    scale — the dspec veto declines the firing instead of minting an
    unsound member."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _sdpa_fold_div_nomask())
    rule, _rec = ev.admit_object(conn, key)
    conn.close()
    q, k, v, s = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4), _v("s")
    scores = _p(
        "div",
        _p("matmul", q, _p("transpose", k, dim0=-1, dim1=-2)),
        s,
    )
    src = _p("matmul", _p("softmax", scores, dim=-1), v)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    assert not eg.rule_fires


# ---------------------------------------------------------------------------
#  Diagonal-scan composites — carrier objects over constructed premises
# ---------------------------------------------------------------------------


def test_affd_scan2_is_composed_from_constructed_premises():
    """The composite's shape and provenance: a 2-block lift whose
    derivation names the constructed step object — no shipped rule
    participates."""
    obj = _affd_scan2()
    assert obj.construction == ("compose", "affd_step_lift")
    assert obj.rule.derivation == ("affd_step_lift",)
    assert obj.rule.lhs == _p(
        "add",
        _p("mul", "a2", _p("add", _p("mul", "a1", "h"), "x1")),
        "x2",
    )
    assert obj.rule.rhs == _p(
        "applyd",
        _p("aff_diag", "a2", "x2"),
        _p("applyd", _p("aff_diag", "a1", "x1"), "h"),
    )


def test_affd_scan4_self_composes():
    """compose(scan2, scan2): the four-block object — every edge is a
    constructed premise; no shipped law is even *named*."""
    obj = _affd_scan4()
    assert obj.construction == ("compose", "affd_scan2_lift")
    assert obj.rule.derivation == ("affd_scan2_lift",)
    # Four nested steps in, four nested applyd out.
    rhs = obj.rule.rhs
    depth = 0
    while isinstance(rhs, Op) and rhs.op == "applyd":
        depth += 1
        rhs = rhs.args[-1]
    assert depth == 4


def test_affd_objects_clear_the_gauntlet(tmp_path):
    """The guarded diagonal family on an injected corpus: sweep-verified
    truth, pays, bounded closure — carrier composites admit when the
    carrier is cheap (the dense ``aff`` analogues did not)."""
    conn = ev.connect(str(tmp_path / "s.db"))
    corpus = _corpus(
        _affd_step_case(), _affd_scan2_case(), _affd_scan4_case()
    )
    for obj in (_affd_step_lift(), _affd_scan2(), _affd_scan4()):
        key = synth.store_constructed(conn, obj)
        rep = ev.run_gauntlet(conn, key, corpus=corpus)
        assert rep.usable, (obj.rule.name, rep.reason)
        assert rep.synth_region.equal >= 1
        assert rep.evidence.verify_fail == 0
    conn.close()


def test_affd_scan2_fires_and_behaves(tmp_path):
    """The admitted composite folds two spelled steps into one nested
    carrier — the e-graph merges them and the value is unchanged."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _affd_scan2())
    rule, _rec = ev.admit_object(conn, key)
    conn.close()
    a1, a2 = _v("a1", 4), _v("a2", 4)
    h, x1, x2 = _v("h", 4), _v("x1", 4), _v("x2", 4)
    inner = _p("add", _p("mul", a1, h), x1)
    src = _p("add", _p("mul", a2, inner), x2)
    dst = _p(
        "applyd",
        _p("aff_diag", a2, x2),
        _p("applyd", _p("aff_diag", a1, x1), h),
    )
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    assert eg.rule_fires.get(rule.name) == 1
    assert eg.find(root) == eg.find(eg.add_term(dst))
    out, _ = lvo.eval_instance(src, dst)
    assert out == "equal"


# ---------------------------------------------------------------------------
#  Guard transport — composing premises whose guard needs real shapes
# ---------------------------------------------------------------------------


def _om_lift_object() -> synth.ConstructedObject:
    """The online-softmax lift, declared as data (see the lift section)."""
    return synth.lift_object(
        "om_lift",
        ("matmul", ("softmax", "S", {"dim": -1}), "V"),
        ("om_elem", "S", "V"),
        "om_apply",
    )


def _om_chunk2() -> synth.ConstructedObject:
    """``compose(om_lift, om_split)`` — the two-block chunked attention.

    ``om_split``'s guard reads *real shapes* (the concat dims must name
    the key axes, the chunk shapes must align) and cannot evaluate on the
    symbolic binding — the composition used to decline (``None``).  Guard
    transport fires the premise structurally and carries its ``check``
    into the composite, so the composite is admissible exactly where
    ``om_split`` would have fired.
    """
    from catopt_carriers.om import OM_LAWS

    om_split = next(r for r in OM_LAWS if r.name == "om_split")
    obj = synth.compose_objects(
        "om_chunk2",
        _om_lift_object(),
        om_split,
        specialize={
            "S": ("concat", "s1", "s2", {"dim": "SD"}),
            "V": ("concat", "v1", "v2", {"dim": "VD"}),
        },
    )
    assert obj is not None
    return obj


def test_guarded_carrier_premise_composes_via_transport():
    """``compose(om_lift, om_split)`` builds now: the shape-reading
    premise guard is *transported* rather than declining the composite."""
    obj = _om_chunk2()
    assert obj.construction == ("compose", "om_lift", "om_split")
    assert obj.rule.lhs == _p(
        "matmul",
        _p("softmax", _p("concat", "s1", "s2", dim="SD"), dim=-1),
        _p("concat", "v1", "v2", dim="VD"),
    )
    assert obj.rule.rhs == _p(
        "om_apply",
        _p(
            "om_compose",
            _p("om_elem", "s1", "v1"),
            _p("om_elem", "s2", "v2"),
        ),
    )
    # ``om_split``'s guard is code (not a declarative cond), so the
    # composite carries it as its fire-time check — and the store reports
    # the honest boundary: the record cannot carry it.
    assert obj.rule.check is not None
    assert serialize.missing_hooks(obj.rule) == ("check",)


def test_transported_guard_fires_exactly_where_the_premise_does():
    """The composite's transported check IS ``om_split``'s guard
    re-evaluated on the premise's own binding: aligned chunk pairs are
    accepted, mis-aligned ones declined."""
    rule = _om_chunk2().rule
    good = {
        "s1": _v("s1", 5, 3),
        "s2": _v("s2", 5, 4),
        "v1": _v("v1", 3, 6),
        "v2": _v("v2", 4, 6),
        "$attr:SD": -1,
        "$attr:VD": -2,
    }
    assert rule.check(good) is True
    # scores concat on the row axis — om_split's last-dim condition fails.
    assert rule.check(dict(good, **{"$attr:SD": 0})) is False
    # v2's key dim (9) no longer contracts with s2's (4).
    assert rule.check(dict(good, **{"v2": _v("v2", 9, 6)})) is False


def test_first_premise_guard_is_transported():
    """A first premise whose own guard is *code* (``om_lift``'s
    ``_check_om_lift`` — softmax over the key axis) rides the composite as
    a procedural step when the caller declares no ``cond``: the composite
    fires only where the head premise would."""
    from catopt_carriers.om import OM_LAWS

    by_name = {r.name: r for r in OM_LAWS}
    obj = synth.compose_objects(
        "om_lift_unlift", by_name["om_lift"], by_name["om_unlift"]
    )
    assert obj is not None
    rule = obj.rule
    assert rule.rhs == _p("matmul", _p("softmax", "s", dim=-1), "v")
    assert serialize.missing_hooks(rule) == ("check",)
    good = {"s": _v("s", 4, 4), "v": _v("v", 4, 6), "$attr:SD": -1}
    assert rule.check(good) is True
    # softmax over dim 0 — om_lift's last-axis condition fails.
    assert rule.check(dict(good, **{"$attr:SD": 0})) is False


def test_structural_fire_runs_derive_and_declines_uninstantiable():
    """The structural path runs a premise's ``derive`` (its RHS attributes
    must instantiate) and declines — as ``None`` — when the RHS cannot
    denote: a derive that vetoes, an attr metavar no derive supplies, and
    a rewrite that leaves the term unchanged."""
    first = Rewrite(
        "t_wrap", _p("wrap", "u"), _p("add", _p("frob", "u"), "u")
    )
    derived = Rewrite(
        "t_frob_derive",
        _p("frob", "u"),
        _p("frob", "u", dim="D"),
        # undecidable on the symbolic binding (u is a metavar string),
        # true once u is a concrete term — the structural path's premise.
        check=lambda b: isinstance(b.get("u"), Var),
        derive=lambda _b: {"$attr:D": 0},
    )
    obj = synth.compose_objects("t_wrap_frob", first, derived)
    assert obj is not None
    assert obj.rule.rhs == _p("add", _p("frob", "u", dim=0), "u")
    assert serialize.missing_hooks(obj.rule) == ("check",)
    assert obj.rule.check({"u": _v("u", 4, 4)}) is True

    # a derive that vetoes on the symbolic binding — the RHS cannot
    # instantiate, so the composition declines.
    veto = Rewrite(
        "t_frob_veto",
        _p("frob", "u"),
        _p("frob", "u", dim="D"),
        derive=lambda _b: None,
    )
    assert synth.compose_objects("t_wrap_veto", first, veto) is None

    # a derive that raises on the symbolic binding — a veto too, never
    # propagated.
    def _raise(_bound):
        raise ValueError("derive boom")

    raising = Rewrite(
        "t_frob_raise",
        _p("frob", "u"),
        _p("frob", "u", dim="D"),
        derive=_raise,
    )
    assert synth.compose_objects("t_wrap_raise", first, raising) is None

    # an RHS attr metavar no derive supplies: uninstantiable — declined.
    unbound = Rewrite(
        "t_frob_unbound", _p("frob", "u"), _p("frob", "u", dim="D")
    )
    assert (
        synth.compose_objects("t_wrap_unbound", first, unbound) is None
    )

    # a no-op rewrite: the structural match changes nothing — declined.
    noop = Rewrite("t_frob_noop", _p("frob", "u"), _p("frob", "u"))
    assert synth.compose_objects("t_wrap_noop", first, noop) is None


def test_transported_recheck_declines_when_the_guard_raises():
    """A transported guard that raises on the composite's concrete binding
    declines — a firing abort, never a crash — so the composite is
    vacuous there."""
    first = Rewrite(
        "t_wrap2", _p("wrap", "u"), _p("add", _p("frob", "u"), "u")
    )

    def _boom(_bound):
        raise ValueError("boom")

    bad = Rewrite(
        "t_frob_boom",
        _p("frob", "u"),
        _p("frob", "u", dim="D"),
        check=_boom,
        derive=lambda _b: {"$attr:D": 0},
    )
    obj = synth.compose_objects("t_wrap_boom", first, bad)
    assert obj is not None
    assert obj.rule.check({"u": _v("u", 4, 4)}) is False


def test_om_chunk2_gauntlet_refuses_the_procedural_guard(tmp_path):
    """The composite is a *claim*: the gauntlet refuses it at full-data —
    the transported guard is code the record cannot carry.  Building is
    unblocked; admission is not bought."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _om_chunk2())
    record = ev.stored_object(conn, key)
    assert record["serializable"] is False
    assert record["missing_hooks"] == ["check"]
    s1, s2 = _v("s1", 5, 3), _v("s2", 5, 4)
    v1, v2 = _v("v1", 3, 6), _v("v2", 4, 6)
    term = _p(
        "matmul",
        _p("softmax", _p("concat", s1, s2, dim=-1), dim=-1),
        _p("concat", v1, v2, dim=-2),
    )
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_case("chunk2", term, s1, s2, v1, v2))
    )
    conn.close()
    assert not rep.usable
    assert rep.reason.startswith("full-data:")
    assert _stages(rep)["full-data"].detail == "dropped hooks: check"


def _channel_then_row() -> synth.ConstructedObject:
    """``compose(linear_channel_scale_rev, linear_row_scale)``.

    Both premises carry *declarative* rank/shape guards that decline on
    the symbolic binding (``rank`` needs a shape); transport folds them
    into the composite's ``cond`` — pure data, so the composite stays
    serializable.  The first premise's guard (channel scale on ``W``/
    ``c``) rides the composite's LHS; the fired premise's row-scale guard
    (on ``c``) is renamed onto the composite's metavariable.
    """
    obj = synth.compose_objects(
        "channel_then_row_scale",
        _BY_NAME["linear_channel_scale_rev"],
        _BY_NAME["linear_row_scale"],
    )
    assert obj is not None
    return obj


def test_declarative_guards_transport_as_cond_clauses():
    """Both premises' declarative guards ride the composite's ``cond``
    (renamed onto the composite's metavariables) — the composite is pure
    data, no procedural remainder."""
    obj = _channel_then_row()
    assert obj.rule.lhs == _p("linear", "x", _p("mul", "W", "c"))
    assert obj.rule.rhs == _p("mul", _p("linear", "x", "W"), "c")
    assert obj.rule.cond is not None
    assert serialize.missing_hooks(obj.rule) == ()
    good = {"x": _v("x", 2, 4), "W": _v("W", 3, 4), "c": Const(2.0)}
    assert eval_cond(obj.rule.cond, good)
    # c=(3,) is neither scalar nor last-dim-1 → the row guard declines.
    assert not eval_cond(obj.rule.cond, dict(good, c=_v("c", 3)))


def test_declarative_transport_cleared_by_the_tightened_premise(
    tmp_path,
):
    """The derivable composite is now admitted: tightening
    ``linear_row_scale``'s guard (the derivable-gate audit) removed the
    counterexample the composite used to inherit.

    The composite's guard is the premises' conjunction, so the premise
    fix propagated: the sweep now measures 0 ``unequal`` / 0
    ``rhs_err`` where the audit found 1, and the truth gate passes on
    the derivation-backed clean region.  Construction is a claim; the
    gauntlet is the referee — and here it clears.
    """
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _channel_then_row())
    x, w = _v("x", 2, 4), _v("W", 3, 4)
    term = _p("linear", x, _p("mul", w, Const(2.0)))
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_case("chrev", term, x, w))
    )
    conn.close()
    assert rep.evidence.derivable
    assert rep.synth_region.unequal == 0
    assert rep.synth_region.rhs_err == 0
    assert _stages(rep)["truth"].passed
    assert "derivation overridden" not in _stages(rep)["truth"].detail


def test_transport_carries_the_tightened_premise_guard():
    """Transport is faithful: the composite's guard IS the premises'
    conjunction, so the tightened ``linear_row_scale`` clause rides the
    composite.  The composite *declines* the rank-1 degenerate binding
    the premise now declines, and its guarded region carries no
    counterexample (the audit's fix cured the inherited blind spot)."""
    rule = _channel_then_row().rule
    # linear_row_scale's old blind-spot binding (x=(1,), W=(1,),
    # r := c = (1,)) — the transported ``bcast-into`` clause declines it.
    bad = {"x": _v("x", 1), "W": _v("W", 1), "c": _v("c", 1)}
    assert not eval_cond(rule.cond, bad)
    region = ev._guarded_evals(
        rule, ev._synth_sites(rule.lhs, rule.rhs, limit=400)
    )
    assert region.equal > 0
    assert region.unequal == 0 and region.rhs_err == 0


# ---------------------------------------------------------------------------
#  Honest negatives — construction depth does not buy admission
# ---------------------------------------------------------------------------


def test_guarded_attr_metavar_object_clears_the_gauntlet(tmp_path):
    """A guarded object whose RHS mints an attr metavar (``scale=SC``)
    on a non-view op: the generic attr domain (attr-sweep retro) lets
    the sweep enumerate the region — the guarded truth gate now
    verifies it where the raw oracle stays ``conditional``."""
    conn = ev.connect(str(tmp_path / "s.db"))
    guarded = synth.fold_object(
        "sdpa_fold_div_nomask_g",
        (
            "matmul",
            (
                "softmax",
                (
                    "div",
                    (
                        "matmul",
                        "Q",
                        (
                            "transpose",
                            "K",
                            {"dim0": "TD1", "dim1": "TD2"},
                        ),
                    ),
                    "S",
                ),
                {"dim": -1},
            ),
            "V",
        ),
        ("sdpa", "Q", "K", "V", {"scale": "SC"}),
        cond=("and", _COND_LAST2, ("const-num", "S")),
        dspec={"SC": ("recip", ("float", ("const", "S")))},
        tags=("fusion",),
    )
    key = synth.store_constructed(conn, guarded)
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_attn_div_nomask_case())
    )
    conn.close()
    assert rep.usable, rep.reason
    assert rep.synth_region.accepted > 0
    assert rep.synth_region.equal >= 1
    assert rep.synth_region.unequal == 0
    assert rep.real_region.equal >= 1


# ---------------------------------------------------------------------------
#  Guard manipulation — relax_guard / specialize (the plan-0020 moves)
# ---------------------------------------------------------------------------


def _guarded_softsign() -> synth.ConstructedObject:
    """A two-clause guarded fold — the relax/specialize fixture.

    ``div(x, |x|+1) -> softsign(x)`` under ``shaped(X) ∧ rank(X)<=2``.
    Both clauses are honest (the equality holds on every binding), so
    the rank conjunct is the tight clause a relax drops.
    """
    return synth.fold_object(
        "softsign_guarded",
        ("div", "X", ("add", ("abs", "X"), 1)),
        "softsign",
        cond=("and", ("shaped", "X"), ("rank", "X", "<=", 2)),
    )


def test_relax_guard_drops_the_named_clause():
    """The clause is addressable by datum (tuple or JSON-shaped list)
    or by index — and what is not there declines honestly."""
    obj = _guarded_softsign()
    by_data = synth.relax_guard(obj, ("rank", "X", "<=", 2))
    by_list = synth.relax_guard(obj, ["rank", "X", "<=", 2])
    by_index = synth.relax_guard(obj, 1)
    for relaxed in (by_data, by_list, by_index):
        assert relaxed is not None
        assert relaxed.rule.cond == ("shaped", "X")
        assert relaxed.kind == "abstraction"
        assert relaxed.construction[0] == "relax_guard"
    # a nested ``and`` flattens — the conjuncts are the leaf clauses
    rlx = synth.relax_guard(
        _BY_NAME["rms_norm_fold"], ("attr-is", "MK", True)
    )
    assert rlx.rule.cond == (
        "and",
        ("const-num", "EPS"),
        ("const-cmp", "P", "==", 2),
        ("shape-eq", "w", ("tail-block", "u", "MD")),
    )
    # …and the declines: a clause not in the guard, an index past the
    # conjunction, an unguarded object, and a procedural guard (code is
    # not a clause — it cannot be dropped).
    assert synth.relax_guard(obj, ("concrete", "X")) is None
    assert synth.relax_guard(obj, 5) is None
    assert synth.relax_guard(_softsign_fold(), 0) is None
    assert synth.relax_guard(_om_chunk2(), 0) is None


def test_relax_guard_to_unguarded():
    """Dropping the last conjunct unguards the object — ``cond=None``
    and no procedural remainder rides along."""
    obj = _guarded_softsign()
    r1 = synth.relax_guard(obj, 0)
    assert r1.rule.cond == ("rank", "X", "<=", 2)
    r2 = synth.relax_guard(r1, 0)
    assert r2.rule.cond is None
    assert r2.rule.check is None


def test_relax_guard_region_grows_monotonically():
    """Measured monotonicity: every binding the guarded object accepts
    the relaxed one accepts (superset), the sweep counts the growth,
    and the grown region re-measures clean — softsign holds on the
    rank-3 sites the dropped clause had declined."""
    obj = _guarded_softsign()
    relaxed = synth.relax_guard(obj, 1).rule
    sites = list(ev._synth_sites(obj.rule.lhs, obj.rule.rhs, limit=200))
    assert sites
    # Pointwise superset — no accepted binding is ever lost.
    for subst, _lhs_i in sites:
        if eval_cond(obj.rule.cond, subst):
            assert eval_cond(relaxed.cond, subst)
    base = ev._guarded_evals(obj.rule, sites)
    grown = ev._guarded_evals(relaxed, sites)
    assert grown.accepted > base.accepted  # the growth is measured
    assert grown.equal >= base.equal
    # The grown sites re-measure clean: no counterexample enters.
    assert grown.unequal == grown.rhs_err == grown.guard_err == 0
    assert grown.equal == grown.accepted


def test_relaxed_object_stores_reconstructs_and_admits(tmp_path):
    """The same record path as every constructor: the relaxed sdpa
    fold — weakened to the load-bearing clauses — serializes,
    rebuilds, and clears the gauntlet on its corpus case."""
    conn = ev.connect(str(tmp_path / "s.db"))
    obj = synth.relax_guard(
        _sdpa_fold_nomask(), ("attr-type", "TD1", "int")
    )
    assert obj is not None and obj.kind == "abstraction"
    key = synth.store_constructed(conn, obj)
    record = ev.stored_object(conn, key)
    assert record["serializable"] is True
    rule, _ = ev.admit_object(conn, key)
    assert rule.cond == (
        "and",
        ("concrete", "K"),
        ("attr-type", "TD2", "int"),
        ("axes-last2", "K", "TD1", "TD2"),
    )
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_attn_nomask_case())
    )
    conn.close()
    assert rep.usable, rep.reason
    assert rep.synth_region.unequal == rep.real_region.unequal == 0


# ---------------------------------------------------------------------------
#  Specialize — the narrower instance family
# ---------------------------------------------------------------------------


def _fold_bound(node, bound, dead, env):
    return synth._fold_bound(node, bound, dead, env)


def test_specialize_fold_bound_contract():
    """The partial evaluator: a fully-bound clause folds, a falsified
    or dangling one refuses, a live one keeps verbatim."""
    env = {"S": Const(4.0), "$attr:D": 0}
    bound = {"S", "$attr:D"}
    dead = {"S"}
    # decided-True → discharged
    assert _fold_bound(("const-num", "S"), bound, dead, env) is True
    # decided-False → the binding falsifies it → refuse
    assert (
        _fold_bound(("const-cmp", "S", ">", 9), bound, dead, env)
        is False
    )
    # dangling (a dead ref mixed with a live one) → unexpressible
    assert (
        _fold_bound(("dim-eq", "S", 0, "T", 0), bound, dead, env)
        is None
    )
    # untouched by the binding → verbatim
    node = ("rank", "Q", ">=", 2)
    assert _fold_bound(node, bound, dead, env) == node
    # combinator fold: True absorbs or, drops from and; not(True)→False
    assert (
        _fold_bound(
            ("or", ("attr-eq", "D", 0), ("leaf", "Q")),
            bound,
            dead,
            env,
        )
        is True
    )
    assert _fold_bound(
        ("and", ("attr-eq", "D", 0), ("leaf", "Q")), bound, dead, env
    ) == ("leaf", "Q")
    assert (
        _fold_bound(("not", ("attr-eq", "D", 0)), bound, dead, env)
        is False
    )


def test_specialize_attr_pin_narrows_the_region():
    """``select_mul`` under ``D=0``: the ``dim`` metavar stays bound —
    the pin rides the cond as a new conjunct — and the measured region
    is exactly the D=0 slice of the parent's."""
    parent = _BY_NAME["select_mul"]
    obj = synth.specialize(parent, {"D": 0})
    assert obj is not None
    rule = obj.rule
    assert rule.lhs == parent.lhs  # the pattern is unspelled
    assert ("attr-eq", "D", 0) in rule.cond[1:]
    assert obj.construction == ("specialize", "select_mul", "D=0")
    # the explicit $attr: spelling is the same binding
    twin = synth.specialize(parent, {"$attr:D": 0})
    assert twin.rule.cond == rule.cond
    sites = list(ev._synth_sites(rule.lhs, rule.rhs, limit=600))
    assert sites
    base = ev._guarded_evals(parent, sites)
    child = ev._guarded_evals(rule, sites)
    # strictly narrower: the parent's D!=0 accepted sites are gone —
    # and every child-accepted env binds D=0
    assert 0 < child.accepted < base.accepted
    assert child.unequal == child.rhs_err == 0
    assert child.equal <= base.equal
    for subst, _lhs_i in sites:
        if eval_cond(rule.cond, subst):
            assert subst["$attr:D"] == 0


def test_specialized_object_stores_reconstructs_and_fires(tmp_path):
    """store → admit → apply: the D=0 instance fires on dim-0 selects
    and declines dim-1 — the pin, not the pattern, narrows."""
    conn = ev.connect(str(tmp_path / "s.db"))
    obj = synth.specialize(_BY_NAME["select_mul"], {"D": 0})
    key = synth.store_constructed(conn, obj)
    record = ev.stored_object(conn, key)
    assert record["serializable"] is True
    rule, _ = ev.admit_object(conn, key)
    conn.close()
    u, v = _v("u", 2, 4), _v("v", 2, 4)
    src = _p(
        "mul",
        _p("select", u, dim=0, index=1),
        _p("select", v, dim=0, index=1),
    )
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule], root, max_iterations=4, max_nodes=10_000)
    assert eg.rule_fires.get(rule.name) == 1
    assert eg.find(root) == eg.find(
        eg.add_term(_p("select", _p("mul", u, v), dim=0, index=1))
    )
    # dim=1 declines the pin — where the un-specialized parent fires.
    src1 = _p(
        "mul",
        _p("select", u, dim=1, index=1),
        _p("select", v, dim=1, index=1),
    )
    eg2 = EGraph()
    r2 = eg2.add_term(src1)
    eg2.run([rule], r2, max_iterations=4, max_nodes=10_000)
    assert not eg2.rule_fires
    eg3 = EGraph()
    r3 = eg3.add_term(src1)
    eg3.run(
        [_BY_NAME["select_mul"]], r3, max_iterations=4, max_nodes=10_000
    )
    assert eg3.rule_fires.get("select_mul") == 1


def test_specialize_leaf_binds_a_const_and_folds_the_derive(tmp_path):
    """``sdpa_fold_div_nomask`` under ``S=4.0``: the divisor becomes a
    literal ``Const``, the scale derive folds to ``("lit", 0.25)``, and
    the narrower object admits on its corpus case."""
    obj = synth.specialize(_sdpa_fold_div_nomask(), {"S": 4.0})
    assert obj is not None
    rule = obj.rule
    assert rule.lhs == _p(
        "matmul",
        _p(
            "softmax",
            _p(
                "div",
                _p(
                    "matmul",
                    "Q",
                    _p("transpose", "K", dim0=-1, dim1=-2),
                ),
                Const(4.0),
            ),
            dim=-1,
        ),
        "V",
    )
    assert rule.dspec == (("SC", ("lit", 0.25)),)
    assert rule.derive({}) == {"$attr:SC": 0.25}
    assert serialize.missing_hooks(rule) == ()
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, obj)
    rule2, record = ev.admit_object(conn, key)
    assert record["serializable"] is True
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_attn_div_nomask_case())
    )
    conn.close()
    assert rep.usable, rep.reason
    # the admitted instance mints sdpa(scale=0.25)
    q, k, v = _v("q", 2, 4), _v("k", 3, 4), _v("v", 3, 4)
    src = _p(
        "matmul",
        _p(
            "softmax",
            _p(
                "div",
                _p("matmul", q, _p("transpose", k, dim0=-1, dim1=-2)),
                Const(4.0),
            ),
            dim=-1,
        ),
        v,
    )
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([rule2], root, max_iterations=4, max_nodes=10_000)
    assert eg.rule_fires.get(rule2.name) == 1
    assert eg.find(root) == eg.find(
        eg.add_term(_p("sdpa", q, k, v, scale=0.25))
    )


def test_specialize_reintroduced_metavar_keeps_its_guard():
    """``h ↦ add(mul(a1,h),x1)`` reintroduces ``h`` — the state guard
    survives verbatim and now reads the *inner* state metavar."""
    obj = synth.specialize(
        _affd_step_lift(), {"h": ("add", ("mul", "a1", "h"), "x1")}
    )
    assert obj is not None
    assert obj.rule.lhs == _p(
        "add",
        _p("mul", "a", _p("add", _p("mul", "a1", "h"), "x1")),
        "x",
    )
    assert obj.rule.cond == _COND_AFFD_STATE


def test_specialize_declines_honestly():
    """``None`` on every non-construction: empty binding, a name the
    LHS binds nowhere, a metavar rename, a contradictory pin, and a
    dangling mixed clause."""
    sel = _BY_NAME["select_mul"]
    assert synth.specialize(sel, {}) is None
    assert synth.specialize(sel, {"ZZ": 0}) is None
    assert synth.specialize(sel, {"u": "w"}) is None  # a rename
    # the pin contradicts a guard that already fixes D
    pinned = Rewrite(
        "t_pinned",
        _p("select", "u", dim="D", index="I"),
        "u",
        cond=("attr-eq", "D", 5),
    )
    assert synth.specialize(pinned, {"D": 0}) is None
    # the bound leaf's clause also reads a live metavar — dangling
    mixed = Rewrite(
        "t_mixed",
        _p("add", "X", "Y"),
        _p("add", "Y", "X"),
        cond=("dim-eq", "X", 0, "Y", 0),
    )
    assert synth.specialize(mixed, {"X": 2}) is None


def test_construct_dispatches_the_ops_by_name(tmp_path):
    """``construct(op, *args, store)`` is the arena entry point: one
    name → one constructor → the same store-able record path."""
    conn = ev.connect(str(tmp_path / "s.db"))
    assert set(synth.CONSTRUCTORS) >= {
        "fold",
        "lift",
        "compose",
        "auto_cond",
        "relax_guard",
        "specialize",
    }
    rec = synth.construct(
        "relax_guard", _guarded_softsign(), 1, store=conn
    )
    assert rec is not None
    assert rec["serializable"] is True
    assert rec["cond"] == ["shaped", "X"]
    assert rec["kind"] == "abstraction"
    # unstored — the bare object record, not a row
    rec2 = synth.construct(
        "specialize", _BY_NAME["select_mul"], {"D": 0}
    )
    assert rec2 is not None and rec2["cond"][-1] == ["attr-eq", "D", 0]
    rec3 = synth.construct(
        "fold",
        "softsign_2",
        ("div", "X", ("add", ("abs", "X"), 1)),
        "softsign",
    )
    assert rec3 is not None and rec3["name"] == "softsign_2"
    # a declined construction is honest None; an unknown op raises
    assert synth.construct("relax_guard", _softsign_fold(), 0) is None
    assert (
        synth.construct(
            "compose", "x", _BY_NAME["comm_mul"], _BY_NAME["silu_fold"]
        )
        is None
    )
    try:
        synth.construct("frobnicate", _softsign_fold())
    except ValueError:
        pass
    else:
        raise AssertionError("unknown op must raise")
    conn.close()


# ---------------------------------------------------------------------------
#  Spec-grammar and specialization internals — the honest declines
# ---------------------------------------------------------------------------


def test_term_from_spec_rejects_a_non_string_head():
    """A spec whose head is not an op name is malformed data."""
    with pytest.raises(TypeError, match="spec head"):
        synth.term_from_spec((42, "X"))


def test_term_from_spec_passes_a_plain_leaf_through():
    """A leaf that is already a term is the identity case — it is
    re-validated nowhere."""
    v = _v("w", 4)
    assert synth.term_from_spec(v) is v


def test_specialize_ref_tables_mark_what_a_clause_reads():
    """The ref tables are the dangling-boundary oracle: ``None`` is
    the refusal, a set is what the node actually reads."""
    # _shape_refs — literal / metavar / unknown spec / bad arity
    assert synth._shape_refs(3) == set()
    assert synth._shape_refs("A") == {"A"}
    assert synth._shape_refs(("nosuch-spec", "A")) is None
    assert synth._shape_refs(("mm-out", "A")) is None
    # _cond_refs — bool/data leaves, unknown predicates, arity
    assert synth._cond_refs(True) == set()
    assert synth._cond_refs("dangling") is None
    assert synth._cond_refs(()) is None
    assert synth._cond_refs(
        ("and", ("leaf", "A"), ("shaped", "B"))
    ) == {
        "A",
        "B",
    }
    # a refusal deep inside a junction propagates upward
    assert (
        synth._cond_refs(
            ("or", ("leaf", "A"), ("nosuch",), ("shaped", "C"))
        )
        is None
    )
    assert synth._cond_refs(("not", ("leaf", "A"))) == {"A"}
    assert synth._cond_refs(("not",)) is None
    assert synth._cond_refs(("nosuch-pred", "A")) is None
    assert synth._cond_refs(("leaf", "A", "B")) is None
    # attr-eq over non-metavar names reads nothing
    assert synth._cond_refs(("attr-eq", 5, 0)) == set()
    # a malformed shape spec inside a T slot refuses
    assert synth._cond_refs(("rank-eq", ("nosuch-spec",), "B")) is None
    # _dexpr_refs — the derive-expression mirror of _cond_refs
    assert synth._dexpr_refs(None) == set()
    assert synth._dexpr_refs(4) == set()
    assert synth._dexpr_refs("X") is None
    assert synth._dexpr_refs(
        ("tuple", ("shape", "A"), ("attr", "D"))
    ) == {"A", "$attr:D"}
    assert synth._dexpr_refs(("concat", ("attr", "D"), "bad")) is None
    assert synth._dexpr_refs(("nosuch", 1)) is None
    assert synth._dexpr_refs(("dim", "A")) is None


def test_fold_bound_combinator_edges():
    """``_fold_bound``/``_fold_comb``: refuse to guess (``None``),
    fold what the binding decides, keep what stays live."""
    env = {"S": Const(4.0), "$attr:D": 0}
    bound, dead = {"S", "$attr:D"}, {"S"}
    fb = synth._fold_bound
    # passthroughs and malformed data
    assert fb(True, bound, dead, env) is True
    assert fb("X", bound, dead, env) is None
    assert fb(("nosuch-pred", "S"), bound, dead, env) is None
    # combinators — an undecidable child vetoes the fold
    assert (
        fb(("not", ("dim-eq", "S", 0, "T", 0)), bound, dead, env)
        is None
    )
    assert (
        fb(
            ("and", ("leaf", "Q"), ("dim-eq", "S", 0, "T", 0)),
            bound,
            dead,
            env,
        )
        is None
    )
    # every child absorbed: the junction folds to its identity's dual
    assert (
        fb(
            ("and", ("const-num", "S"), ("attr-eq", "D", 0)),
            bound,
            dead,
            env,
        )
        is True
    )
    assert (
        fb(
            ("or", ("const-cmp", "S", ">", 9), ("attr-eq", "D", 7)),
            bound,
            dead,
            env,
        )
        is False
    )


def test_a_raising_eval_declines_the_fold_and_drops_the_pred(
    monkeypatch,
):
    """The ``except`` around ``eval_cond`` is the honest
    refuse-to-guess: a raising eval declines the fold outright and a
    raising predicate is dropped from the mask bank."""

    def boom(node, _env):
        if node == ("leaf", "A"):
            raise TypeError("edge")
        return True

    monkeypatch.setattr(synth, "eval_cond", boom)
    assert (
        synth._fold_bound(("leaf", "A"), {"A"}, {"A"}, {"A": Const(1)})
        is None
    )
    masks = synth._pred_masks(
        [("leaf", "A"), ("leaf", "B")], [{"X": 1}]
    )
    assert masks == [(("leaf", "B"), 0b1)]


def test_specialize_pin_clause_spellings_and_declines():
    """``None``/bool pins ride ``attr-is``, other values ``attr-eq``;
    lists normalize to tuples; a non-str key or a leaf reintroduced
    as an attr metavar declines."""
    r = Rewrite(
        "t_sel",
        lhs=_p("select", "u", dim="D", index="I"),
        rhs="u",
        cond=("dim-eq-const", "u", 0, 1),
    )
    obj = synth.specialize(r, {"D": None})
    assert obj.rule.cond == (
        "and",
        ("dim-eq-const", "u", 0, 1),
        ("attr-is", "D", None),
    )
    obj = synth.specialize(r, {"$attr:I": True})
    assert obj.rule.cond == (
        "and",
        ("dim-eq-const", "u", 0, 1),
        ("attr-is", "I", True),
    )
    obj = synth.specialize(r, {"D": [0, 1]})
    assert obj.rule.cond == (
        "and",
        ("dim-eq-const", "u", 0, 1),
        ("attr-eq", "D", (0, 1)),
    )
    assert synth.specialize(r, {4: 0}) is None
    # leaf name "D" is a leaf metavar — a spec that reintroduces "D"
    # as an attr metavar is a collision, not a binding
    amb = Rewrite("t_amb", lhs=_p("add", "D", "X"), rhs="X")
    assert (
        synth.specialize(amb, {"D": ("reshape", "Q", {"shape": "D"})})
        is None
    )


def test_specialize_dspec_folds_vetoes_and_declines():
    """The derive spec folds the same honest way — literals where the
    binding decides, verbatim where live, decline on the dangling."""
    base = _p("add", "A", "B")
    r = Rewrite(
        "t_d", base, _p("add", "B", "A"), dspec={"Z": ("dim", "B", 0)}
    )
    obj = synth.specialize(r, {"A": 4})
    assert obj.rule.dspec == (("Z", ("dim", "B", 0)),)
    r = Rewrite(
        "t_d2", base, _p("add", "B", "A"), dspec={"Z": ("const", "A")}
    )
    obj = synth.specialize(r, {"A": 4})
    assert obj.rule.dspec == (("Z", ("lit", 4)),)
    # the binding decides the expr but it cannot evaluate → veto
    r = Rewrite("t_d3", base, "A", dspec={"Z": ("dim", "A", 0)})
    assert synth.specialize(r, {"A": 4}) is None
    # a dead leaf mixed with a live one is unexpressible → decline
    r = Rewrite(
        "t_d4",
        base,
        "A",
        dspec={"Z": ("add", ("dim", "A", 0), ("dim", "B", 0))},
    )
    assert synth.specialize(r, {"A": 4}) is None
    # malformed expr data → decline, not a guess
    r = Rewrite("t_d5", base, "A", dspec={"Z": "A"})
    assert synth.specialize(r, {"A": 4}) is None


# ---------------------------------------------------------------------------
#  The auto-cond predicate bank — the cond-DSL vocabulary over a pattern
# ---------------------------------------------------------------------------


def test_view_out_spec_table():
    """Every spec fn names its output shape through the attr metavars
    it needs — a missing metavar or a foreign op declines (``None``)."""
    cases = [
        ("unsqueeze", {"dim": "D"}, ("unsq-out", "U", "D")),
        ("unsqueeze", {"dim": 0}, None),
        ("reshape", {"shape": "S"}, ("reshape-out", "U", "S")),
        ("view", {"shape": "S"}, ("reshape-out", "U", "S")),
        ("reshape", {"shape": (1, 2)}, None),
        ("getitem", {"index": "I"}, ("getitem-out", "U")),
        ("getitem", {}, ("getitem-out", "U")),
        ("select", {"dim": "D"}, ("select-out", "U", "D")),
        ("select", {"dim": 0}, None),
        (
            "slice",
            {"dim": "D", "start": "S", "end": "E"},
            ("slice-out", "U", "D", "S", "E", None),
        ),
        ("slice", {"dim": 0}, None),
        (
            "chunk",
            {"chunks": "C", "dim": "D"},
            ("chunk-out", "U", "C", "D"),
        ),
        ("chunk", {"chunks": 2}, None),
        (
            "transpose",
            {"dim0": "A", "dim1": "B"},
            ("transpose-out", "U", "A", "B"),
        ),
        ("transpose", {"dim0": -1}, None),
        ("t", {"dim0": "A"}, ("transpose-out", "U", "A", None)),
        ("softmax", {"dim": "D"}, None),
    ]
    for op, attrs, want in cases:
        assert synth._view_out_spec(op, "U", attrs) == want, (op, attrs)


def _bank_probe_lhs() -> Op:
    """A probe pattern exercising every view-spec slot of the bank."""
    return _p(
        "add",
        _p("mul", _p("getitem", "T", index="I"), "U"),
        _p(
            "add",
            _p(
                "mul",
                _p("unsqueeze", "W", dim="UD"),
                _p("transpose", "V", dim0="T2", dim1="T3"),
            ),
            _p(
                "mul",
                _p(
                    "mul",
                    _p("select", "A", dim="PD", index="PI"),
                    _p("select", "B", dim="PD", index="PI"),
                ),
                _p(
                    "add",
                    _p(
                        "transpose",
                        _p("reshape", "R", shape="RS"),
                        dim0="T0",
                        dim1="T1",
                    ),
                    _p(
                        "mul",
                        _p(
                            "slice",
                            "E",
                            dim="LD",
                            start="LS",
                            end="LE",
                        ),
                        _p("chunk", "C", chunks="CN", dim="CD"),
                    ),
                ),
            ),
        ),
    )


def _bank_probe_rhs() -> Op:
    """Pattern-side nodes that drive the bank's skip branches."""
    return _p(
        "add",
        # dim0 concrete: the node has *some* attr metavar but the spec
        # and the transpose pred's required names are missing
        _p("transpose", "K", dim0=-1, dim1="D1"),
        _p(
            "mul",
            _p(
                "mul",
                # a str attr on a non-view op — no output spec exists
                _p("softmax", _p("mul", "G", "H"), dim="MD"),
                # a concrete ``chunks``: a view node whose spec declines
                Op.make("chunk", "C2", chunks=2, dim="CD2"),
            ),
            _p(
                "mul",
                # a second getitem with identical attrs: a view pair
                # whose op carries no index-attr alignment
                _p("getitem", "T2", index="I"),
                _p(
                    "mul",
                    # slice with a concrete start / a concrete end —
                    # the per-attr emission edges
                    _p("slice", "E2", dim="LD2", start=0, end="LE2"),
                    _p(
                        "slice",
                        "E3",
                        dim="LD3",
                        start="LS3",
                        end=0,
                    ),
                ),
            ),
        ),
        # a view chain whose inner operand's operand is not a metavar
        _p(
            "add",
            _p("unsqueeze", _p("mul", Const(2), "M"), dim="WD"),
            "N",
        ),
    )


def test_pred_bank_enumerates_the_view_vocabulary():
    """The bank is data over the pattern's vocabulary — per-metavar
    atoms, pair atoms, view-output spec atoms, observed attr values,
    op bindings, and the commute/wrap/view-view forms."""
    lhs, rhs = _bank_probe_lhs(), _bank_probe_rhs()
    envs = [
        # T observed bound to an op (op-in); PD observed as a list
        # (normalized to a tuple)
        (
            {
                "T": _p("topk", _v("w", 2, 4), k=2),
                "$attr:I": 0,
                "$attr:PD": [-1, -2],
            },
            "equal",
        ),
        # None/bool observations do not mint ``attr-eq`` pins
        (
            {
                "T": _v("t", 4),
                "$attr:I": 1,
                "$attr:UD": None,
                "$attr:T2": True,
            },
            "unequal",
        ),
    ]
    bank = synth._pred_bank(lhs, rhs, envs)
    # per-metavar atoms
    assert ("leaf", "T") in bank
    assert ("not", ("leaf", "T")) in bank
    assert ("rank", "U", ">=", 2) in bank
    assert ("const-cmp", "U", "!=", 0) in bank
    # metavar-pair atoms
    assert ("shape-eq", "T", "U") in bank
    assert ("shape-compat", "T", "U") in bank
    assert ("bcast-into", "T", "U") in bank
    assert ("dim-eq", "T", 0, "U", -1) in bank
    assert ("dim-compat", "T", 1, "U", -1) in bank
    # view-output spec preds — ``unsq-out`` names unsqueeze's shape
    assert ("shape-eq", ("unsq-out", "W", "UD"), "T") in bank
    assert ("bcast-into", "T", ("unsq-out", "W", "UD")) in bank
    assert ("bcast-eq", ("unsq-out", "W", "UD"), "T", "W", "T") in bank
    # attr-metavar atoms: the type bank, pair pins, dim/attr crosses
    assert ("attr-type", "UD", "int") in bank
    assert ("attr-eq-attr", "I", "UD") in bank
    assert ("dim-eq-attr", "A", "PD", "B", "PD") in bank
    assert ("attr-cmp-dim", "I", "!=", "T", "UD") in bank
    assert ("attr-cmp-dim", "UD", ">=", "W", "T2") in bank
    # observed values — the list normalizes to a tuple; bool/None
    # observations mint no ``attr-eq`` pin
    assert ("attr-eq", "I", 0) in bank
    assert ("attr-eq", "PD", (-1, -2)) in bank
    assert ("attr-eq", "UD", None) not in bank
    assert ("attr-eq", "T2", True) not in bank
    # op bindings observed under a metavariable
    assert ("op-in", "T", ("topk",)) in bank
    assert ("not", ("op-in", "T", ("topk",))) in bank
    assert ("op-in", "U", ("topk",)) not in bank
    # per-view node preds
    assert ("attr-in", "I", (0, -1)) in bank  # getitem
    assert ("dim-eq-const", "T", 0, 1) in bank
    assert ("ones-before", "W", "UD") in bank  # unsqueeze
    assert (
        "flat-pair-unsq",
        "W",
        "UD",
        "A",
        ("reshape-out", "R", "RS"),
    ) in bank
    assert (
        "flat-map-unsq",
        "W",
        "UD",
        "A",
        ("reshape-out", "R", "RS"),
    ) in bank
    assert ("axes-last2", "V", "T2", "T3") in bank  # transpose
    assert ("attr-is", "LS", None) in bank  # slice
    assert ("attr-eq", "LS", 0) in bank
    assert ("attr-is", "LE", None) in bank
    assert ("attr-cmp-dim", "LE", ">=", "E", "LD") in bank
    # the commute/wrap vocabulary
    assert ("bcast-dim-inv", "U", "A", "PD") in bank
    assert ("axis-align-eq", "A", "B", "PD") in bank
    assert (
        "or",
        ("axes-noop", "U", "T2", "T3"),
        ("rank", "U", "<=", 1),
    ) in bank
    assert (
        "bcast-eq",
        ("getitem-out", "T"),
        "U",
        ("getitem-out", ("bcast", "T", "U")),
        ("getitem-out", ("bcast", "T", "U")),
    ) in bank
    # same-op same-attrs view pair — the aligned-index case
    assert (
        "bcast-eq",
        ("select-out", "A", "PD"),
        ("select-out", "B", "PD"),
        ("select-out", ("bcast", "A", "B"), "PD"),
        ("select-out", ("bcast", "A", "B"), "PD"),
    ) in bank
    # view-view commutation: the g1(g2(x)) spec vs g2(g1(x))
    assert (
        "shape-eq",
        ("transpose-out", ("reshape-out", "R", "RS"), "T0", "T1"),
        ("reshape-out", ("transpose-out", "R", "T0", "T1"), "RS"),
    ) in bank
    # predicate spellings dedupe
    assert len(bank) == len(set(bank))


# ---------------------------------------------------------------------------
#  The auto-cond domain machinery — measurement, enumeration, cover
# ---------------------------------------------------------------------------


def test_oracle_sites_derive_and_instantiate_edges(monkeypatch):
    """``_oracle_sites``: derive extras merge; vetoes, raises and
    uninstantiable envs skip; signatures dedupe; ``limit`` caps."""
    lhs = _p("add", "A", "B")
    rhs = _p("add", "B", "A")
    va, vb, vb2 = _v("a", 2), _v("b", 2), _v("b2", 2)
    envs = [
        {"A": va, "B": vb},  # instantiates
        {"A": va},  # missing B — the instantiation raises
        {"A": va, "B": vb2},  # instantiates to a distinct site
        {"A": va, "B": vb},  # a replay — deduped by signature
    ]
    monkeypatch.setattr(lvo, "_binding_envs", lambda _l, _r: iter(envs))
    got = list(synth._oracle_sites(lhs, rhs, None, 10))
    assert len(got) == 2
    assert got[0][0]["B"] is vb and got[1][0]["B"] is vb2
    # the limit caps the enumeration window
    assert len(list(synth._oracle_sites(lhs, rhs, None, 1))) == 1

    def derive(bound):
        if bound.get("B") is vb2:
            return None  # a veto — the site declines
        if bound.get("B") is None:
            raise ValueError("boom")  # a raising derive — skipped
        return {"$attr:Z": 0}

    got = list(synth._oracle_sites(lhs, rhs, derive, 10))
    assert len(got) == 1 and got[0][0]["B"] is vb


def test_measure_domain_records_real_matches_over_an_enum_failure(
    monkeypatch,
):
    """``_measure_domain``'s honest boundary: a raising enumerator
    still reports whatever was collected plus the real matches."""
    torch.manual_seed(0)
    rule = Rewrite(
        "gi",
        lhs=_p("getitem", "T", index="I"),
        rhs=_p("getitem", "T", index=0),
    )
    t = _v("t", 4)
    site = _p("getitem", t, index=0)

    def _boom(*_a, **_k):
        raise TypeError("enumerator edge")

    monkeypatch.setattr(ev, "_synth_sites", _boom)
    out = synth._measure_domain(rule, [site], 60)
    assert out == [({"T": t, "$attr:I": 0}, "equal")]


def test_measure_domain_skips_a_real_match_that_does_not_rebind(
    monkeypatch,
):
    """A schema match that fails the re-binding contributes nothing —
    the second ``_term_match`` is the honest gate."""
    torch.manual_seed(0)
    rule = Rewrite(
        "r", lhs=_p("add", "X", "Y"), rhs=_p("add", "Y", "X")
    )
    x, y = _v("x", 2), _v("y", 2)
    term = _p("add", x, y)
    phantom = _v("phantom", 2)
    monkeypatch.setattr(
        shape_proposal,
        "real_matches",
        lambda _terms, _schema: [phantom],
    )
    out = synth._measure_domain(rule, [term], 8)
    assert out and all(o == "equal" for _e, o in out)
    assert all(e.get("X") is not phantom for e, _o in out)


def test_outcome_bitsets_partitions_every_outcome_class():
    """``(equal, bad, other, declined)`` — declined sites sit outside
    the firing region and play no role in the cover."""
    eq, bad, other, declined = synth._outcome_bitsets(
        [
            ({}, "equal"),
            ({}, "unequal"),
            ({}, "rhs-err"),
            ({}, "unstable"),
            ({}, "lhs-err"),
            ({}, "declined"),
            ({}, "guard-err"),
            ({}, "env-err"),
        ]
    )
    assert eq == 0b00000001
    assert bad == 0b00001110
    assert other == 0b10010000
    assert declined == 0b01100000


def test_min_cover_search_order_and_spent_budget(monkeypatch):
    """Iterative deepening by clause count; a bad site with no killer
    refuses before the search; a spent visit budget returns nothing
    deeper."""
    eq, bad = 0b000111, 0b111000
    useful = [
        (("a",), eq | 0b110000, 0b001000),
        (("b",), eq | 0b001000, 0b110000),
        (("c",), eq, bad),
    ]
    assert synth._min_cover(useful, eq, bad, 6, 3) == (("c",),)
    # bad sites 4/5 have no killer → honest refusal before searching
    assert synth._min_cover(useful[:1], eq, bad, 6, 3) is None
    # competing covers at a level are compared — a worse completion
    # does not replace the recorded best
    useful2 = [
        (("a",), eq, 0b001000),  # kills 3
        (("b",), eq, 0b010000),  # kills 4
        (("c",), eq, 0b100000),  # kills 5
        (("d",), eq, 0b001000),  # kills 3 too
    ]
    got = synth._min_cover(useful2, eq, bad, 6, 3)
    assert got is not None and sorted(p[0] for p in got) == [
        "a",
        "b",
        "c",
    ]
    # a spent visit budget: the deeper levels short-circuit
    monkeypatch.setattr(synth, "_AUTO_VISIT_CAP", 4)
    assert synth._min_cover(useful[:2], eq, bad, 6, 3) is None


def test_stable_outcome_short_circuits_on_declined_sites():
    """A site whose probe declines (a derive veto) reports the veto
    outcome directly — no second draw."""
    rule = Rewrite(
        "veto",
        lhs=_p("add", "X", "Y"),
        rhs=_p("mul", "X", "Y"),
        derive=lambda _b: None,
    )
    env = {"X": _v("x", 2), "Y": _v("y", 2)}
    assert synth._stable_outcome(rule, env, None) == "declined"


def test_auto_cond_mints_a_one_clause_guard():
    """``add(getitem(T,I),U) → add(getitem(T,0),U)`` holds iff I is 0 —
    the smallest cover is the one-atom pin."""
    torch.manual_seed(0)
    rule = Rewrite(
        "gi_first",
        lhs=_p("add", _p("getitem", "T", index="I"), "U"),
        rhs=_p("add", _p("getitem", "T", index=0), "U"),
    )
    res = synth.auto_cond_object(rule, synth_limit=360)
    assert res.object is not None
    assert res.cond == ("attr-eq", "I", 0)
    assert res.clauses == (("attr-eq", "I", 0),)
    assert res.equal > 0 and res.bad > 0 and res.other > 0
    assert (
        res.accepted_other > 0
    )  # don't-care sites counted, not folded
    # pure data — the minted cond is the whole claim
    assert serialize.missing_hooks(res.object.rule) == ()
    # the guard is the real gate: it accepts exactly the I=0 slice
    assert eval_cond(res.cond, {"$attr:I": 0}) is True
    assert eval_cond(res.cond, {"$attr:I": 1}) is False


def test_auto_cond_mints_a_two_clause_guard_and_refuses_below():
    """The unsqueeze strip needs a conjunction — the deepened search
    finds it; capped below the cover it refuses honestly."""
    torch.manual_seed(0)
    rule = Rewrite(
        "add_unsq",
        lhs=_p("add", _p("unsqueeze", "U", dim="A_dim"), "V"),
        rhs=_p("add", "U", "V"),
    )
    res = synth.auto_cond_object(rule, synth_limit=360)
    assert res.object is not None
    assert len(res.clauses) == 2
    assert res.cond == ("and", *res.clauses)
    # capped under the minimal cover → no declarable conjunction
    res2 = synth.auto_cond_object(rule, synth_limit=360, max_clauses=1)
    assert res2.object is None
    assert "no declarable conjunction" in res2.detail


def test_auto_cond_refuses_a_domain_that_never_evaluates():
    """A derive that vetoes every site: measured but never evaluable —
    the refusal names it."""
    torch.manual_seed(0)
    rule = Rewrite(
        "always_veto",
        lhs=_p("add", "X", "Y"),
        rhs=_p("mul", "X", "Y"),
        derive=lambda _b: None,
    )
    res = synth.auto_cond_object(rule, synth_limit=60)
    assert res.object is None
    assert res.measured > 0
    assert res.declined == res.measured
    assert "no evaluable site" in res.detail
