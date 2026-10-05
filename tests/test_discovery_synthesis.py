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

import torch
from catopt_core.egraph import EGraph
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_core.typing import _shape_of
from catopt_discovery import evidence as ev
from catopt_discovery import object_synthesis as synth
from catopt_discovery import oracle as lvo
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
#: scale is literal: a str attr on a non-view op defeats the oracle's
#: enumeration domain — ``_attr_options`` covers view ops only.)
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
    literal (the pattern pins the last-two swap — no cond needed) and
    the scale is *derived*: the ``dspec`` float-veto declines a
    non-``Const`` S at fire time, so the derive is the guard.
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
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _softsign_fold())
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(_softsign_case()))
    conn.close()
    assert rep.usable, rep.reason
    assert rep.name == "softsign_fold" and rep.kind == "abstraction"
    assert all(s.passed for s in rep.stages)
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
    """The scaled mask-free block — the dspec float-veto is the guard."""
    conn = ev.connect(str(tmp_path / "s.db"))
    key = synth.store_constructed(conn, _sdpa_fold_div_nomask())
    rep = ev.run_gauntlet(
        conn, key, corpus=_corpus(_attn_div_nomask_case())
    )
    conn.close()
    assert rep.usable, rep.reason
    assert rep.evidence.num_true is True
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
#  Honest negatives — construction depth does not buy admission
# ---------------------------------------------------------------------------


def test_guarded_carrier_premise_declines_symbolic_compose():
    """``compose(om_lift, om_split)``: ``om_split``'s shape-derived
    guard cannot evaluate on metavar terms — the operation declines
    (``None``), the documented guarded-compose limit."""
    from catopt_carriers.om import OM_LAWS

    om_lift = synth.lift_object(
        "om_lift",
        ("matmul", ("softmax", "S", {"dim": -1}), "V"),
        ("om_elem", "S", "V"),
        "om_apply",
    )
    om_split = next(r for r in OM_LAWS if r.name == "om_split")
    obj = synth.compose_objects(
        "om_chunk2",
        om_lift,
        om_split,
        specialize={
            "S": ("concat", "s1", "s2", {"dim": "SD"}),
            "V": ("concat", "v1", "v2", {"dim": "VD"}),
        },
    )
    assert obj is None


def test_guarded_object_outside_the_sweep_domain_fails_truth(tmp_path):
    """A guarded object whose RHS mints an attr metavar (``scale=SC``)
    leaves the oracle's enumeration domain — zero synthesized sites —
    so the guarded truth gate cannot be satisfied even though the one
    real match verifies equal.  The measured envelope of the sweep."""
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
    assert not rep.usable
    stages = _stages(rep)
    assert not stages["truth"].passed
    assert rep.synth_region.accepted == 0
    assert rep.real_region.equal >= 1
