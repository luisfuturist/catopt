"""Construction-gap tests — the human-bar misses, closed by construction.

The human-bar retro (``project/retros/human-bar.md``) named three
machine-coverage gaps the store could not reach:

- **chain composition** — no machine-built ``assoc_linear``-class
  object, so the machine arm lost the ~-20% ``LoRAAdapter`` win;
- **self-product folds** — no machine-built ``mul(x,x)``-family
  object, so the tiny ``mul_square`` wins were unreachable;
- **gather/dedup** — no ``index_select`` composition objects at all.

This module constructs those objects with the generic construction
operators (:func:`catopt_discovery.object_synthesis.fold_object` /
``lift_object`` / ``compose_objects``) — no new law bodies — stores
each one through the evidence store, and pins the honest gauntlet
verdict: usable where the sweep measured the claim, declined at the
exact stage where it did not.  The per-gap write-up lives in
``project/retros/construction-gaps.md``.
"""

import torch
from catopt_carriers.decode_laws import DECODE_LAWS
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import ALL_RULES
from catopt_discovery import evidence as ev
from catopt_discovery import object_synthesis as synth
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink

_BY_NAME = {r.name: r for r in ALL_RULES}
_DECODE = {r.name: r for r in DECODE_LAWS}

_T = ("transpose", "W", {"dim0": -1, "dim1": -2})


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _w(name: str, *shape: int) -> Param:
    return Param(name, TensorType(tuple(shape)))


def _case(
    name: str, term: Op, *inputs: Var, params: tuple[Param, ...] = ()
) -> TermCase:
    return TermCase(
        source="test",
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=tuple(
            torch.randn(tuple(x.typ.shape), dtype=torch.float64)
            for x in inputs
        ),
        param_vals={
            p.name: torch.randn(tuple(p.typ.shape), dtype=torch.float64)
            for p in params
        },
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


def _gauntlet(obj: synth.ConstructedObject, *cases: TermCase):
    """Store the construction and run the full gauntlet on it."""
    conn = ev.connect(":memory:")
    key = synth.store_constructed(conn, obj)
    rep = ev.run_gauntlet(conn, key, corpus=_corpus(*cases))
    rec = ev.stored_object(conn, key)
    conn.close()
    return rep, rec


# ---------------------------------------------------------------------------
#  Gap 1 — chain composition: linear∘linear by compose over bridge folds
# ---------------------------------------------------------------------------
#
#  ``compose`` cannot mint ``matmul(B, A)`` out of thin air — it rewrites
#  the head premise's RHS through later premises that already mention
#  every op.  The arena-player spelling therefore goes through bridge
#  folds — ``linear(x,W) ≡ matmul(x, Wᵀ)`` and the transposed-product
#  flip ``tA·tB ≡ (B·A)ᵀ`` — composed with shipped matmul
#  associativity.  The bridges are scaffolding: only the composite is
#  stored and gauntleted.


def _lin_as_mm() -> synth.ConstructedObject:
    """Bridge fold: ``linear(x, W)`` spelled as ``matmul(x, Wᵀ)``."""
    return synth.fold_object(
        "lin_as_mm", ("linear", "x", "W"), ("matmul", "x", _T)
    )


def _mm_transpose_flip() -> synth.ConstructedObject:
    """Bridge fold: ``tA·tB ≡ (B·A)ᵀ``."""
    return synth.fold_object(
        "mm_transpose_flip",
        (
            "matmul",
            ("transpose", "A", {"dim0": -1, "dim1": -2}),
            ("transpose", "B", {"dim0": -1, "dim1": -2}),
        ),
        ("transpose", ("matmul", "B", "A"), {"dim0": -1, "dim1": -2}),
    )


def _linear_chain_t() -> synth.ConstructedObject:
    """The paying chain object: ``lin(lin(x,A),B) → matmul(x,(B·A)ᵀ)``.

    ``assoc_linear`` keeps the fused product inside ``linear``; this
    composite exposes it as ``matmul(x, transpose(matmul(B,A)))`` —
    the same fused weight, a different alpha-key, so novelty clears.
    The guard mirrors ``assoc_linear``'s contract: rank-2 weights and
    the chain dims actually feeding.
    """
    obj = synth.compose_objects(
        "linear_chain_t",
        _lin_as_mm(),
        _lin_as_mm(),
        _BY_NAME["assoc_matmul_rev"],
        _mm_transpose_flip(),
        specialize={"x": ("linear", "x", "A"), "W": "B"},
        cond=(
            "and",
            ("rank", "A", "==", 2),
            ("rank", "B", "==", 2),
            ("dim-eq", "x", -1, "A", 1),
            ("dim-eq", "A", 0, "B", 1),
        ),
    )
    assert obj is not None
    return obj


def _chain_case() -> TermCase:
    """``linear(linear(x, A), B)`` with Param weights — the LoRA step."""
    x, a, b = _v("x", 4, 8), _w("A", 3, 8), _w("B", 5, 3)
    return _case(
        "lin_lin", _p("linear", _p("linear", x, a), b), x, params=(a, b)
    )


def test_linear_chain_t_is_usable_and_fires():
    rep, rec = _gauntlet(_linear_chain_t(), _chain_case())
    assert rep.usable, rep.reason
    assert rec["serializable"] is True
    stages = _stages(rep)
    assert stages["truth"].passed and stages["typed-pay"].passed
    # Behavioral pin: the object folds the chain to the fused matmul.
    from catopt_core.cost import dag_cost

    x, a, b = _v("x", 4, 8), _w("A", 3, 8), _w("B", 5, 3)
    term = _p("linear", _p("linear", x, a), b)
    eg = EGraph()
    root = eg.add_term(term)
    eg.run([rep.rule], root, max_iterations=8, max_nodes=10000)
    assert eg.rule_fires.get("linear_chain_t", 0) >= 1
    sink = _sink()
    cost = _cost_fn(sink)
    best = eg.extract_best(eg.find(root), cost)
    assert dag_cost(best, cost) < dag_cost(term, cost)


def test_compose_recovers_assoc_linear_but_novelty_declines():
    """The full bridge reaches ``assoc_linear``'s exact rhs — and the
    novelty gate refuses the alpha-duplicate honestly."""
    mm2lin = synth.fold_object(
        "mm_t_as_linear", ("matmul", "x", _T), ("linear", "x", "W")
    )
    obj = synth.compose_objects(
        "assoc_linear_built",
        _lin_as_mm(),
        _lin_as_mm(),
        _BY_NAME["assoc_matmul_rev"],
        _mm_transpose_flip(),
        mm2lin,
        specialize={"x": ("linear", "x", "A"), "W": "B"},
        cond=("and", ("rank", "A", "==", 2), ("rank", "B", "==", 2)),
    )
    assert obj is not None
    rep, rec = _gauntlet(obj, _chain_case())
    assert rec["serializable"] is True
    stages = _stages(rep)
    assert stages["truth"].passed
    assert not rep.usable
    assert not stages["novelty"].passed


def test_verbatim_assoc_linear_fold_is_a_novelty_decline():
    obj = synth.fold_object(
        "assoc_linear_fold",
        ("linear", ("linear", "X", "A"), "B"),
        ("linear", "X", ("matmul", "B", "A")),
    )
    rep, _rec = _gauntlet(obj, _chain_case())
    assert not rep.usable
    assert not _stages(rep)["novelty"].passed


def test_transpose_spelled_chain_does_not_pay():
    """``matmul(x, tA·tB)`` bills two non-foldable transposes — the
    executor model charges ``transpose`` as a real op even on Params,
    so this spelling clears truth and novelty but not typed-pay."""
    obj = synth.compose_objects(
        "linear_chain_mm",
        _lin_as_mm(),
        _lin_as_mm(),
        _BY_NAME["assoc_matmul_rev"],
        specialize={"x": ("linear", "x", "A"), "W": "B"},
        cond=("and", ("rank", "A", "==", 2), ("rank", "B", "==", 2)),
    )
    assert obj is not None
    rep, _rec = _gauntlet(obj, _chain_case())
    assert not rep.usable
    stages = _stages(rep)
    assert stages["novelty"].passed
    assert not stages["typed-pay"].passed


def test_three_chain_starves_the_guarded_sweep():
    """``lin(lin(lin(x,A),B),C)`` composes cleanly, but its six
    conjunctive guard leaves the synthesized-binding bank with zero
    accepted sites — an honest truth decline, not a wrong object."""
    obj = synth.compose_objects(
        "linear_chain3_t",
        _lin_as_mm(),
        _lin_as_mm(),
        _lin_as_mm(),
        _BY_NAME["assoc_matmul_rev"],
        _BY_NAME["assoc_matmul_rev"],
        _mm_transpose_flip(),
        _mm_transpose_flip(),
        specialize={
            "x": ("linear", ("linear", "x", "A"), "B"),
            "W": "C",
        },
        cond=(
            "and",
            ("rank", "A", "==", 2),
            ("rank", "B", "==", 2),
            ("rank", "C", "==", 2),
            ("dim-eq", "x", -1, "A", 1),
            ("dim-eq", "A", 0, "B", 1),
            ("dim-eq", "B", 0, "C", 1),
        ),
    )
    assert obj is not None
    x, a, b, c = (
        _v("x", 4, 8),
        _w("A", 3, 8),
        _w("B", 5, 3),
        _w("C", 6, 5),
    )
    case = _case(
        "lin3",
        _p("linear", _p("linear", _p("linear", x, a), b), c),
        x,
        params=(a, b, c),
    )
    rep, _rec = _gauntlet(obj, case)
    assert not rep.usable
    assert not _stages(rep)["truth"].passed


# ---------------------------------------------------------------------------
#  Gap 2 — self-product folds
# ---------------------------------------------------------------------------


def _mul_to_pow2() -> synth.ConstructedObject:
    """``mul(x,x) → pow(x,2)`` composed out of shipped square bridges.

    Carries ``derivation=("mul_square","square_to_pow")`` — the cert
    stage replays it strictly against ``ALL_RULES``.
    """
    obj = synth.compose_objects(
        "mul_to_pow2", _BY_NAME["mul_square"], _BY_NAME["square_to_pow"]
    )
    assert obj is not None
    return obj


def _exp_mul_fold() -> synth.ConstructedObject:
    """``e^a·e^b → e^(a+b)`` — the decoder-side exp-product fold."""
    return synth.fold_object(
        "exp_mul_fold",
        ("mul", ("exp", "A"), ("exp", "B")),
        ("exp", ("add", "A", "B")),
    )


def _square_mul_fold() -> synth.ConstructedObject:
    """``a²·b² → (a·b)²``."""
    return synth.fold_object(
        "square_mul_fold",
        ("mul", ("square", "A"), ("square", "B")),
        ("square", ("mul", "A", "B")),
    )


def _neg_neg_mul() -> synth.ConstructedObject:
    """``(-a)·(-b) → a·b``."""
    return synth.fold_object(
        "neg_neg_mul",
        ("mul", ("neg", "A"), ("neg", "B")),
        ("mul", "A", "B"),
    )


def _pow_add_exp() -> synth.ConstructedObject:
    """``x^a·x^b → x^(a+b)`` — exponent addition as a fold."""
    return synth.fold_object(
        "pow_add_exp",
        ("mul", ("pow", "X", "A"), ("pow", "X", "B")),
        ("pow", "X", ("add", "A", "B")),
    )


def _abs_mul_fold() -> synth.ConstructedObject:
    """``|a|·|b| → |a·b|``."""
    return synth.fold_object(
        "abs_mul_fold",
        ("mul", ("abs", "A"), ("abs", "B")),
        ("abs", ("mul", "A", "B")),
    )


def test_mul_to_pow2_is_usable_with_replaying_cert():
    x = _v("x", 4, 8)
    rep, _rec = _gauntlet(
        _mul_to_pow2(), _case("xx", _p("mul", x, x), x)
    )
    assert rep.usable, rep.reason
    cert = _stages(rep)["cert"]
    assert cert.passed and "cert" in cert.detail


def test_elementwise_self_product_folds_are_usable():
    a, b = _v("a", 4, 8), _v("b", 4, 8)
    cases = {
        "exp_mul_fold": (
            _exp_mul_fold(),
            _case("ee", _p("mul", _p("exp", a), _p("exp", b)), a, b),
        ),
        "square_mul_fold": (
            _square_mul_fold(),
            _case(
                "ss", _p("mul", _p("square", a), _p("square", b)), a, b
            ),
        ),
        "neg_neg_mul": (
            _neg_neg_mul(),
            _case("nn", _p("mul", _p("neg", a), _p("neg", b)), a, b),
        ),
        "pow_add_exp": (
            _pow_add_exp(),
            _case(
                "pp",
                _p(
                    "mul",
                    _p("pow", a, Const(2)),
                    _p("pow", a, Const(3)),
                ),
                a,
            ),
        ),
        "abs_mul_fold": (
            _abs_mul_fold(),
            _case("ab", _p("mul", _p("abs", a), _p("abs", b)), a, b),
        ),
    }
    for name, (obj, case) in cases.items():
        rep, _rec = _gauntlet(obj, case)
        assert rep.usable, f"{name}: {rep.reason}"


def test_rsqrt_product_is_a_measured_counterexample():
    """``rsqrt(a)·rsqrt(b) → rsqrt(a·b)`` is FALSE on the measured
    domain: a<0, b<0 gives NaN·NaN on the left and a finite value on
    the right.  No bank predicate can guard it — truth declines."""
    a, b = _v("a", 4, 8), _v("b", 4, 8)
    obj = synth.fold_object(
        "rsqrt_mul_fold",
        ("mul", ("rsqrt", "A"), ("rsqrt", "B")),
        ("rsqrt", ("mul", "A", "B")),
    )
    rep, _rec = _gauntlet(
        obj,
        _case("rr", _p("mul", _p("rsqrt", a), _p("rsqrt", b)), a, b),
    )
    assert not rep.usable
    assert not _stages(rep)["truth"].passed


def test_self_add_clears_pay_but_overflows_closure():
    """``add(x,x) → mul(x,2)`` is true and pays, but iterating the
    comm/assoc family around the minted member overflows the closure
    budget — declined at closure, not stored as usable."""
    x = _v("x", 4, 8)
    obj = synth.fold_object(
        "self_add_fold", ("add", "X", "X"), ("mul", "X", 2)
    )
    rep, _rec = _gauntlet(obj, _case("aa", _p("add", x, x), x))
    assert not rep.usable
    stages = _stages(rep)
    assert stages["typed-pay"].passed
    assert not stages["closure"].passed


# ---------------------------------------------------------------------------
#  Gap 3 — gather / dedup
# ---------------------------------------------------------------------------


def _isel_compose_lit() -> Rewrite:
    """``t[I][J] = t[I[J]]`` on literal index tuples — pure data."""
    return Rewrite(
        name="isel_compose_lit",
        lhs=_p(
            "index_select",
            _p("index_select", "t", dim="D", index=(0, 1, 2, 3)),
            dim="D",
            index=(0, 2),
        ),
        rhs=_p("index_select", "t", dim="D", index=(0, 2)),
        law="Gather-of-gather compose on literal indices.",
    )


def _isel_compose_proc() -> synth.ConstructedObject:
    """The metavar version needs ``K = I[J]`` — index-of-index — which
    the declarative dspec vocabulary cannot express; spelled with a
    procedural ``derive`` it is honest but not full-data."""

    def _derive_gg(bound):
        i, j = bound.get("$attr:I"), bound.get("$attr:J")
        if not (
            isinstance(i, (tuple, list))
            and isinstance(j, (tuple, list))
        ):
            return None
        if any(
            not isinstance(k, int) or not (0 <= k < len(i)) for k in j
        ):
            return None
        return {"$attr:K": tuple(i[k] for k in j)}

    return synth.ConstructedObject(
        rule=Rewrite(
            name="isel_compose",
            lhs=_p(
                "index_select",
                _p("index_select", "t", dim="D", index="I"),
                dim="D",
                index="J",
            ),
            rhs=_p("index_select", "t", dim="D", index="K"),
            derive=_derive_gg,
            law="t[I][J] = t[I[J]] — gather composition.",
        ),
        kind="abstraction",
        construction=("fold", "isel_compose"),
    )


def _cat_isel_dup() -> synth.ConstructedObject:
    """``cat(t[I], t[I]) → t[I++I]`` — dedup via declarative dspec."""
    return synth.fold_object(
        "cat_isel_dup",
        (
            "concat",
            ("index_select", "t", {"dim": "D", "index": "I"}),
            ("index_select", "t", {"dim": "D", "index": "I"}),
            {"dim": "D"},
        ),
        ("index_select", "t", {"dim": "D", "index": "K"}),
        dspec={"K": ("concat", ("attr", "I"), ("attr", "I"))},
    )


def _cat_isel_fold() -> synth.ConstructedObject:
    """``cat(t[I], t[J]) → t[I++J]`` — the general cat-of-gathers."""
    return synth.fold_object(
        "cat_isel_fold",
        (
            "concat",
            ("index_select", "t", {"dim": "D", "index": "I"}),
            ("index_select", "t", {"dim": "D", "index": "J"}),
            {"dim": "D"},
        ),
        ("index_select", "t", {"dim": "D", "index": "K"}),
        dspec={"K": ("concat", ("attr", "I"), ("attr", "J"))},
    )


def _cat_isel_dup_lit() -> Rewrite:
    """The same dedup on a literal index — no dspec needed."""
    return Rewrite(
        name="cat_isel_dup_lit",
        lhs=_p(
            "concat",
            _p("index_select", "t", dim="D", index=(0, 2)),
            _p("index_select", "t", dim="D", index=(0, 2)),
            dim="D",
        ),
        rhs=_p("index_select", "t", dim="D", index=(0, 2, 0, 2)),
        law="cat(t[I], t[I], d) = t[I++I] on the gather axis.",
    )


def _gg_case() -> TermCase:
    t = _v("t", 8, 4)
    return _case(
        "gg",
        _p(
            "index_select",
            _p("index_select", t, dim=0, index=(0, 1, 2, 3)),
            dim=0,
            index=(0, 2),
        ),
        t,
    )


def _cat_case() -> TermCase:
    t = _v("t", 8, 4)
    return _case(
        "cat",
        _p(
            "concat",
            _p("index_select", t, dim=0, index=(0, 2)),
            _p("index_select", t, dim=0, index=(0, 2)),
            dim=0,
        ),
        t,
    )


def _cat_ij_case() -> TermCase:
    t = _v("t", 8, 4)
    return _case(
        "cat_ij",
        _p(
            "concat",
            _p("index_select", t, dim=0, index=(0, 2)),
            _p("index_select", t, dim=0, index=(1, 3)),
            dim=0,
        ),
        t,
    )


def test_isel_compose_literal_indices_is_usable():
    obj = synth.ConstructedObject(
        rule=_isel_compose_lit(),
        kind="abstraction",
        construction=("fold", "isel_compose_lit"),
    )
    rep, _rec = _gauntlet(obj, _gg_case())
    assert rep.usable, rep.reason


def test_cat_gather_dedup_folds_are_usable():
    rep, _rec = _gauntlet(_cat_isel_dup(), _cat_case())
    assert rep.usable, rep.reason

    rep2, _rec2 = _gauntlet(_cat_isel_fold(), _cat_ij_case())
    assert rep2.usable, rep2.reason

    lit = synth.ConstructedObject(
        rule=_cat_isel_dup_lit(),
        kind="abstraction",
        construction=("fold", "cat_isel_dup_lit"),
    )
    rep3, _rec3 = _gauntlet(lit, _cat_case())
    assert rep3.usable, rep3.reason


def test_metavar_gather_compose_needs_a_missing_derive():
    """``t[I][J] = t[I[J]]`` over metavar indices: the dspec tuple
    language has no index-of-index form, so the object can only be
    spelled procedurally — and the full-data stage refuses it."""
    rep, rec = _gauntlet(_isel_compose_proc(), _gg_case())
    assert not rep.usable
    assert rec["missing_hooks"] == ["derive"]
    assert not _stages(rep)["full-data"].passed


def test_compose_over_procedural_gather_laws_stays_procedural():
    """``compose`` transports the shipped gather laws' Python ``check``
    guards; the composite keeps the hook and full-data declines —
    composition does not launder procedural evidence."""
    obj = synth.compose_objects(
        "dedup_then_id",
        _DECODE["index_select_dedup"],
        _DECODE["index_select_id"],
    )
    assert obj is not None
    rep, rec = _gauntlet(obj, _gg_case())
    assert not rep.usable
    assert rec["missing_hooks"] == ["check"]
    assert not _stages(rep)["full-data"].passed


# ---------------------------------------------------------------------------
#  Deployment — the usable objects ship through machine_pack
# ---------------------------------------------------------------------------


def test_usable_constructions_deploy_through_machine_pack(tmp_path):
    """Store the usable objects, load them back by name through
    ``machine_pack.load_pack`` — the ``rules=`` seam — and the pack
    fires on the chain site."""
    from catopt_discovery import machine_pack

    conn = ev.connect(str(tmp_path / "store.db"))
    for obj in (_linear_chain_t(), _mul_to_pow2()):
        synth.store_constructed(conn, obj)
    pack = machine_pack.load_pack(
        conn, names=("linear_chain_t", "mul_to_pow2")
    )
    conn.close()
    assert set(pack.loaded) == {"linear_chain_t", "mul_to_pow2"}
    assert not pack.skipped
    x, a, b = _v("x", 4, 8), _w("A", 3, 8), _w("B", 5, 3)
    term = _p("linear", _p("linear", x, a), b)
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(list(pack.rules), root, max_iterations=8, max_nodes=10000)
    assert eg.rule_fires.get("linear_chain_t", 0) >= 1
