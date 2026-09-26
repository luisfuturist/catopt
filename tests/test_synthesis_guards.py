"""Guarded-rule synthesis — composing side conditions in completion.

``synthesize_rules`` used to skip every rule with a ``check``/``derive``
hook or an attribute metavariable (~half the ruleset: all the om
lemmas, sdpa folds, scale-naturality rules).  Those gates are gone:
a derived rule now carries a COMPOSITE check — each parent's side
condition evaluated on its own binding, re-expressed through the
intermediate substitution — and ``derive``-produced attributes travel
as namespaced placeholders the derived rule refills at fire time.

Covered here:

* gated-rule census — the previously-blocked rule families (om_lift /
  om_split / om_merge / matmul_t_concat / sdpa_fold / *_scale / affd /
  concat_binarize / qkv_fuse / gqa) are all synthesizable now.
* composite check — a rule synthesized from two CHECKED parents
  accepts exactly the instantiations both parents would, and vetoes
  the rest.
* unsound rejection — a composition whose parent check is
  unsatisfiable (or whose derive output is unproducible) is rejected
  during validation, never emitted.
* new lemmas — the chunked-attention homomorphism
  ``softmax(cat s) @ cat v ≡ ⊕ᵢ elem(sᵢ, vᵢ)`` and the score-concat
  lemma ``softmax(q·cat(k)ᵀ)·cat(v) ≡ apply(elem(cat qkᵢᵀ, cat v))``
  are derivable from guarded parents; both verified fp64 on concrete
  tensors, including a rank-3 instance whose derived concat dim is
  re-computed (2, not the seed's baked 1).
"""

import torch

from catopt import meta
from catopt import om as OM
from catopt_core import laws as R
from catopt.egraph import EGraph, Rewrite
from catopt.ir import Op, TensorType, Var, op_repr


def _T(*shape):
    return TensorType(tuple(shape))


def _om_lemma_seed():
    """matmul(softmax(cat(s1,s2,-1), -1), cat(v1,v2,-2)) — dense
    attention over two key blocks, hand-built as an IR term."""
    s1 = Var("s1", _T(5, 3))
    s2 = Var("s2", _T(5, 4))
    v1 = Var("v1", _T(3, 6))
    v2 = Var("v2", _T(4, 6))
    term = Op.make(
        "matmul",
        Op.make("softmax", Op.make("concat", s1, s2, dim=-1), arg1=-1),
        Op.make("concat", v1, v2, dim=-2),
    )
    return term, (s1, s2, v1, v2)


def _attention_seed():
    """softmax(q @ cat(k1,k2,-2).T) @ cat(v1,v2,-2) — the full dense
    chunked-attention term from test_om_monoid."""
    q = Var("q", _T(5, 4))
    k1 = Var("k1", _T(3, 4))
    k2 = Var("k2", _T(4, 4))
    v1 = Var("v1", _T(3, 6))
    v2 = Var("v2", _T(4, 6))
    kcat = Op.make("concat", k1, k2, dim=-2)
    vcat = Op.make("concat", v1, v2, dim=-2)
    scores = Op.make(
        "matmul", q, Op.make("transpose", kcat, arg1=-2, arg2=-1)
    )
    term = Op.make("matmul", Op.make("softmax", scores, arg1=-1), vcat)
    return term, (q, k1, k2, v1, v2)


def _find(derived, parents=None, rhs_has=None):
    out = []
    for d in derived:
        if parents is not None and meta.provenance(d) != parents:
            continue
        if rhs_has is not None and rhs_has not in op_repr(d.rhs):
            continue
        out.append(d)
    return out


# ---------------------------------------------------------------------------
#  census: previously-blocked rules are now synthesizable
# ---------------------------------------------------------------------------


def test_guarded_rule_families_now_synthesizable():
    """Every rule whose only disqualifier was a check/derive hook or an
    attr metavar participates now — including the whole om lemma
    family and the sdpa folds."""
    rules = meta.module_rules(R) + meta.module_rules(OM)
    syn = {r.name for r in rules if meta._synthesizable(r)}
    for name in (
        "om_lift",
        "om_lift_dim",
        "om_lift_plain",
        "om_split",
        "om_split_arg1",
        "om_merge",
        "matmul_t_concat",
        "matmul_t_concat_arg1",
        "naturality_scalar",
        "linear_row_scale",
        "linear_channel_scale",
        "qkv_fuse",
        "qkv_fuse_asym",
        "gqa_absorb_repeat",
        "sdpa_fold_add",
        "sdpa_fold_masked_fillmul",
        "affd_lift",
        "concat_binarize_3_dim",
    ):
        assert name in syn, name
    # and nothing is left gated on the full ruleset: every rule's RHS
    # metavariables are LHS-bound or derive-produced.
    assert all(meta._synthesizable(r) for r in rules)


# ---------------------------------------------------------------------------
#  (a) composite check accepts / rejects correctly
# ---------------------------------------------------------------------------


def test_composite_check_accepts_and_rejects():
    """om_lift∘om_split emits the two-block homomorphism with a
    composite check: it fires exactly where BOTH parents' shape side
    conditions hold."""
    seed, (s1, s2, v1, v2) = _om_lemma_seed()
    derived = meta.synthesize_rules(
        [OM.OM_LIFT, OM.OM_SPLIT], [seed], fuel=4000
    )
    lemmas = _find(derived, ("om_lift", "om_split"), "om_compose")
    assert lemmas, "chunked-attention lemma not synthesized"
    lemma = lemmas[0]
    assert lemma.check is not None
    assert meta.provenance(lemma) == ("om_lift", "om_split")

    # accepts: well-formed instance — and it is fp64-exact.
    applied = meta.apply_rewrite_at(lemma, seed, ())
    assert applied is not None
    torch.manual_seed(0)
    env = {
        s1: torch.randn(5, 3, dtype=torch.float64),
        s2: torch.randn(5, 4, dtype=torch.float64),
        v1: torch.randn(3, 6, dtype=torch.float64),
        v2: torch.randn(4, 6, dtype=torch.float64),
    }
    assert meta._eval_allclose(
        meta._eval_term(seed, env),
        meta._eval_term(applied, env),
        tol=1e-10,
    )

    # rejects: v2's key dim (9) no longer contracts with s2's (4) —
    # om_split's _check_om_concat_dims vetoes through the composite.
    v2_bad = Var("v2_bad", _T(9, 6))
    bad = Op.make(
        "matmul",
        Op.make("softmax", Op.make("concat", s1, s2, dim=-1), arg1=-1),
        Op.make("concat", v1, v2_bad, dim=-2),
    )
    assert meta.apply_rewrite_at(lemma, bad, ()) is None

    # rejects: scores cat along the row axis — om_lift's last-dim
    # softmax condition fails through the composite.
    s2_row = Var("s2_row", _T(6, 3))
    bad2 = Op.make(
        "matmul",
        Op.make(
            "softmax", Op.make("concat", s1, s2_row, dim=0), arg1=-1
        ),
        Op.make("concat", v1, v2, dim=-2),
    )
    assert meta.apply_rewrite_at(lemma, bad2, ()) is None


def test_composite_check_symbolic_path():
    """Symbolic composition also carries guards: om_lift∘om_unlift
    normalizes a softmax's dim to -1, but only where om_lift's own
    check (softmax over the last axis) holds."""
    derived = meta.synthesize_rules(
        [OM.OM_LIFT, OM.OM_UNLIFT], fuel=1000
    )
    lemmas = _find(derived, ("om_lift", "om_unlift"))
    assert lemmas
    lemma = lemmas[0]
    assert lemma.check is not None
    # LHS keeps the attr metavar SD: accepted values are exactly those
    # naming the last axis — the composite check enforces it.
    s = Var("s", _T(4, 4))
    v = Var("v", _T(4, 6))
    good = Op.make("matmul", Op.make("softmax", s, arg1=-1), v)
    bad = Op.make("matmul", Op.make("softmax", s, arg1=0), v)
    assert meta.apply_rewrite_at(lemma, good, ()) is not None
    assert meta.apply_rewrite_at(lemma, bad, ()) is None


# ---------------------------------------------------------------------------
#  (b) unsound compositions are rejected, never emitted
# ---------------------------------------------------------------------------


def test_unsatisfiable_parent_check_rejected():
    """r2's check is always False: the composite can never fire, so
    validation must reject the pair instead of emitting a dead — or
    worse, unjustified — rule."""
    src = Rewrite(
        "t_guard_src",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
    )
    dead = Rewrite(
        "t_guard_dead",
        Op.make("mul", "a", "b"),
        Op.make("matmul", "a", "b"),
        check=lambda bound: False,
    )
    derived = meta.synthesize_rules([src, dead], fuel=500)
    assert all(
        "t_guard_dead" not in meta.provenance(d) for d in derived
    )


def test_unproducible_derive_rejected():
    """r2's RHS needs an attr metavar that only derive could supply,
    but derive vetoes — the composition is un-justifiable and must not
    be emitted."""
    src = Rewrite(
        "t_guard_src2",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
    )
    veto = Rewrite(
        "t_guard_veto",
        Op.make("mul", "a", "b"),
        Op.make("matmul", "a", "b", scale="S"),
        derive=lambda bound: None,
    )
    derived = meta.synthesize_rules([src, veto], fuel=500)
    assert all(
        "t_guard_veto" not in meta.provenance(d) for d in derived
    )


def test_check_referencing_missing_metavar_rejected():
    """A parent check that consults a metavar it never binds is
    un-reexpressible-by-construction (the composite binding can never
    supply it) — the pair is rejected."""
    src = Rewrite(
        "t_guard_src3",
        Op.make("add", "a", "b"),
        Op.make("mul", "a", "b"),
    )
    ghost = Rewrite(
        "t_guard_ghost",
        Op.make("mul", "a", "b"),
        Op.make("matmul", "a", "b"),
        check=lambda bound: bound.get("ghost") is not None,
    )
    derived = meta.synthesize_rules([src, ghost], fuel=500)
    assert all(
        "t_guard_ghost" not in meta.provenance(d) for d in derived
    )


# ---------------------------------------------------------------------------
#  (c) new lemmas — verified numerically on concrete fp64 tensors
# ---------------------------------------------------------------------------


def test_om_chunked_attention_lemma_fp64():
    """The flagship lemma: softmax over concatenated scores @
    concatenated values IS the two-block monoid product.  Previously
    unreachable — both parents are checked."""
    seed, leaves = _om_lemma_seed()
    derived = meta.synthesize_rules(
        [OM.OM_LIFT, OM.OM_SPLIT, OM.OM_UNLIFT], [seed], fuel=4000
    )
    lemmas = _find(derived, ("om_lift", "om_split"), "om_compose")
    assert lemmas, "no om_lift∘om_split lemma emitted"
    lemma = lemmas[0]
    rhs = op_repr(lemma.rhs)
    assert (
        "om_apply" in rhs
        and "om_compose" in rhs
        and rhs.count("om_elem") == 2
    )

    # numerical equivalence on concrete fp64 tensors
    torch.manual_seed(0)
    env = {
        lf: torch.randn(*lf.typ.shape, dtype=torch.float64)
        for lf in leaves
    }
    applied = meta.apply_rewrite_at(lemma, seed, ())
    lhs_val = meta._eval_term(seed, env)
    rhs_val = meta._eval_term(applied, env)
    ref = torch.softmax(
        torch.cat([env[leaves[0]], env[leaves[1]]], dim=-1), dim=-1
    ) @ torch.cat([env[leaves[2]], env[leaves[3]]], dim=-2)
    assert meta._eval_allclose(lhs_val, ref, tol=1e-12)
    assert meta._eval_allclose(rhs_val, ref, tol=1e-12)

    # and the derived rule plugs back into the e-graph: the root
    # e-class gains the om_apply member whose child class holds the
    # two-block compose.
    eg = EGraph()
    root = eg.add_term(seed)
    eg.run([lemma], root, max_iterations=3, max_nodes=5_000)
    assert eg.rule_fires.get(lemma.name, 0) > 0
    applies = [
        n
        for n in eg.get_class(eg.find(root)).nodes
        if n.op == "om_apply"
    ]
    assert applies
    assert any(
        n.op == "om_compose"
        for n in eg.get_class(applies[0].children[0]).nodes
    )


def test_attention_score_concat_lemma_with_derive_fp64():
    """matmul_t_concat (check + derive) ∘ om_lift (check): dense
    attention over concatenated keys lifts to the element of the
    concatenated per-block scores.  The RHS's concat dim is a
    derive-produced placeholder — re-computed per firing, so a rank-3
    instance gets dim=2, not the seed's baked dim=1."""
    seed, (q, k1, k2, v1, v2) = _attention_seed()
    derived = meta.synthesize_rules(
        [OM.MATMUL_T_CONCAT, OM.OM_LIFT], [seed], fuel=6000
    )
    lemmas = [
        d
        for d in derived
        if "om_elem" in op_repr(d.rhs)
        and "concat" in op_repr(d.rhs)
        and set(meta.provenance(d)) == {"matmul_t_concat", "om_lift"}
    ]
    assert lemmas, "score-concat lemma not synthesized"
    lemma = lemmas[0]
    assert lemma.check is not None and lemma.derive is not None
    assert meta._rhs_derive_placeholders(lemma.rhs)

    torch.manual_seed(0)
    env = {
        lf: torch.randn(*lf.typ.shape, dtype=torch.float64)
        for lf in (q, k1, k2, v1, v2)
    }
    applied = meta.apply_rewrite_at(lemma, seed, ())
    assert applied is not None
    ref = torch.softmax(
        env[q]
        @ torch.cat([env[k1], env[k2]], dim=-2).transpose(-2, -1),
        dim=-1,
    ) @ torch.cat([env[v1], env[v2]], dim=-2)
    assert meta._eval_allclose(
        meta._eval_term(applied, env), ref, tol=1e-10
    )

    # rank-3 instance: the composite derive must recompute SD=2.
    q3 = Var("q3", _T(2, 5, 4))
    k13, k23 = Var("k13", _T(2, 3, 4)), Var("k23", _T(2, 4, 4))
    v13, v23 = Var("v13", _T(2, 3, 6)), Var("v23", _T(2, 4, 6))
    seed3 = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make(
                "matmul",
                q3,
                Op.make(
                    "transpose",
                    Op.make("concat", k13, k23, dim=-2),
                    arg1=-2,
                    arg2=-1,
                ),
            ),
            arg1=-1,
        ),
        Op.make("concat", v13, v23, dim=-2),
    )
    applied3 = meta.apply_rewrite_at(lemma, seed3, ())
    assert applied3 is not None
    # the score concat landed on the LAST axis (dim=2), re-derived —
    # not the rank-2 value baked at synthesis time.
    score_cat = applied3.args[0].args[0]
    assert score_cat.op == "concat" and score_cat.attrs["dim"] == 2
    env3 = {
        lf: torch.randn(*lf.typ.shape, dtype=torch.float64)
        for lf in (q3, k13, k23, v13, v23)
    }
    ref3 = torch.softmax(
        env3[q3]
        @ torch.cat([env3[k13], env3[k23]], dim=-2).transpose(-2, -1),
        dim=-1,
    ) @ torch.cat([env3[v13], env3[v23]], dim=-2)
    assert meta._eval_allclose(
        meta._eval_term(applied3, env3), ref3, tol=1e-10
    )


def test_synthesized_rules_carry_provenance():
    """Every emitted rule names its parent derivation."""
    seed, _ = _om_lemma_seed()
    derived = meta.synthesize_rules(
        [OM.OM_LIFT, OM.OM_SPLIT], [seed], fuel=4000
    )
    assert derived
    for d in derived:
        p = meta.provenance(d)
        assert len(p) == 2 and p == d.parents
        assert meta.SYNTH_PARENTS[d.name] == p
        assert p[0] in d.law and p[1] in d.law
