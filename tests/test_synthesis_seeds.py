"""Seed-witness synthesis — guarded compositions that need real witnesses.

``synthesize_rules`` validates every candidate by instantiating its LHS.
Unguarded and loosely-guarded candidates survive the bounded pool
(``_LEAF_SHAPES`` × ``_ATTR_POOL``), but a guarded composition whose side
conditions pin *relationships* between the bound terms — a Const-valued
attention scale, a broadcast-shaped mask, a transpose tied to a real
rank — has no satisfying assignment in that pool: the uniform leaf
shapes give every metavar the same profile, and no pool value is a
``Const`` leaf at all.

The fix is seed witnesses, exercised two ways:

* the seed-guided path fires ``r1`` then ``r2`` on concrete terms and
  carries the fired binding into validation as the candidate's witness
  (guaranteed guard satisfaction by construction);
* a guarded symbolic candidate with no witness mines bindings by
  matching its LHS against the seed subterms — preferred over the
  bounded pool, which remains as fallback.

Covered here:

* ``comm_mul ∘ sdpa_fold_addmul`` — a left-scaled attention term folds
  to ``sdpa`` only after commuting the scale; emits ONLY with a seed.
* ``sdpa_fold_add ∘ sub_to_add`` — fold-then-rewrite inside the mask
  operand; emits ONLY with a seed.
* ``comm_mul ∘ sdpa_fold_masked_fillmul`` — the masked_fill/-inf form,
  verified against a boolean keep-mask in fp64.
* ``linear_channel_scale ∘ comm_mul`` — a symbolic candidate whose
  broadcast guard is unsatisfiable in the bounded pool: rejected
  without seeds, validated by a seed-mined witness with them.
* the bounded pool remains the fallback — an unseeded guarded pair that
  it *can* satisfy still emits.
"""

import torch

from catopt import meta
from catopt import om as OM
from catopt import rules as R
from catopt.egraph import EGraph, Rewrite
from catopt.ir import Const, Op, TensorType, Var, op_repr


def _T(*shape):
    return TensorType(tuple(shape))


def _sdpa_fold(name):
    return next(r for r in R.SDPA_FOLD_RULES if r.name == name)


def _left_scaled_attention_seed():
    """matmul(softmax(mul(0.5, q@kᵀ) + m, -1), v) — the scale sits LEFT
    of the score matmul, so the term only matches ``sdpa_fold_addmul``
    after a commutativity step."""
    q = Var("q", _T(5, 4))
    k = Var("k", _T(7, 4))
    v = Var("v", _T(7, 6))
    m = Var("m", _T(5, 7))
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, arg1=-2, arg2=-1)
    )
    term = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make("add", Op.make("mul", Const(0.5), scores), m),
            arg1=-1,
        ),
        v,
    )
    return term, (q, k, v, m)


def _sub_mask_attention_seed():
    """matmul(softmax(q@kᵀ + (m1 - m2), -1), v) — unscaled fold whose
    additive mask contains a ``sub`` a second rule can rewrite."""
    q = Var("q", _T(5, 4))
    k = Var("k", _T(7, 4))
    v = Var("v", _T(7, 6))
    m1 = Var("m1", _T(5, 7))
    m2 = Var("m2", _T(5, 7))
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, arg1=-2, arg2=-1)
    )
    term = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make("add", scores, Op.make("sub", m1, m2)),
            arg1=-1,
        ),
        v,
    )
    return term, (q, k, v, m1, m2)


def _masked_fill_attention_seed():
    """matmul(softmax(masked_fill(mul(0.5, q@kᵀ), mk, -inf), -1), v)."""
    q = Var("q", _T(5, 4))
    k = Var("k", _T(7, 4))
    v = Var("v", _T(7, 6))
    mk = Var("mk", _T(5, 7))
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, arg1=-2, arg2=-1)
    )
    term = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make(
                "masked_fill",
                Op.make("mul", Const(0.5), scores),
                mk,
                Const(float("-inf")),
            ),
            arg1=-1,
        ),
        v,
    )
    return term, (q, k, v, mk)


def _env(leaves, bools=()):
    torch.manual_seed(0)
    out = {}
    for lf in leaves:
        t = torch.randn(*lf.typ.shape, dtype=torch.float64)
        out[lf] = (t > 0.0) if lf in bools else t
    return out


def _provenance_in(derived, parents):
    return [d for d in derived if meta.provenance(d) == parents]


# ---------------------------------------------------------------------------
#  (a) sdpa_fold-family compositions emit ONLY via a seed witness
# ---------------------------------------------------------------------------


def test_left_scaled_sdpa_fold_emits_only_with_seed():
    """comm_mul ∘ sdpa_fold_addmul: the fold's ``mul(scores, S)`` shape
    only appears after commuting the seed's ``mul(0.5, scores)``.

    Without seeds the pair is unreachable (nothing synthesises a
    softmax-inside-matmul for the symbolic path, and the Const scale
    defeats the instantiation pool anyway).  With the seed it emits a
    guarded, provenance-tagged, fp64-verified rule."""
    seed, leaves = _left_scaled_attention_seed()
    rules = [R.COMM_MUL, _sdpa_fold("sdpa_fold_addmul")]

    bare = meta.synthesize_rules(rules, fuel=2000)
    assert not _provenance_in(bare, ("comm_mul", "sdpa_fold_addmul"))
    assert all("sdpa" not in op_repr(d.rhs) for d in bare)

    derived = meta.synthesize_rules(rules, [seed], fuel=2000)
    hits = _provenance_in(derived, ("comm_mul", "sdpa_fold_addmul"))
    assert hits, "left-scaled sdpa fold not synthesized"
    d = hits[0]
    assert d.check is not None and d.derive is not None
    assert d.parents == ("comm_mul", "sdpa_fold_addmul")
    assert meta.SYNTH_PARENTS[d.name] == d.parents
    assert "sdpa" in op_repr(d.rhs) and "scale" in op_repr(d.rhs)

    # replays on the seed: folded sdpa carries the derived scale 0.5
    applied = meta.apply_rewrite_at(d, seed, ())
    assert applied is not None and applied.op == "sdpa"
    assert applied.attrs["scale"] == 0.5

    # fp64-verified on real tensors
    env = _env(leaves)
    assert meta._eval_allclose(
        meta._eval_term(seed, env),
        meta._eval_term(applied, env),
        tol=1e-10,
    )

    # and it plugs back into the e-graph: the root class gains an sdpa
    eg = EGraph()
    root = eg.add_term(seed)
    eg.run([d], root, max_iterations=3, max_nodes=5_000)
    assert eg.rule_fires.get(d.name, 0) > 0
    assert any(
        n.op == "sdpa" for n in eg.get_class(eg.find(root)).nodes
    )


def test_sdpa_fold_then_rewrite_emits_only_with_seed():
    """sdpa_fold_add ∘ sub_to_add: fold the attention first, then the
    second rule rewrites ``sub`` inside the folded mask operand —
    fold-then-something, again only through a seed."""
    seed, leaves = _sub_mask_attention_seed()
    rules = [_sdpa_fold("sdpa_fold_add"), R.SUB_TO_ADD]

    bare = meta.synthesize_rules(rules, fuel=2000)
    assert not _provenance_in(bare, ("sdpa_fold_add", "sub_to_add"))

    derived = meta.synthesize_rules(rules, [seed], fuel=4000)
    hits = _provenance_in(derived, ("sdpa_fold_add", "sub_to_add"))
    assert hits, "fold-then-rewrite composition not synthesized"
    d = hits[0]
    assert d.check is not None
    rhs = op_repr(d.rhs)
    assert "sdpa" in rhs and "add" in rhs and "neg" in rhs

    applied = meta.apply_rewrite_at(d, seed, ())
    assert applied is not None and applied.op == "sdpa"
    assert applied.attrs["scale"] == 1.0  # unscaled fold derives 1.0
    # the mask operand inside sdpa is now add(m1, neg(m2))
    mask = applied.args[3]
    assert mask.op == "add" and mask.args[1].op == "neg"

    env = _env(leaves)
    assert meta._eval_allclose(
        meta._eval_term(seed, env),
        meta._eval_term(applied, env),
        tol=1e-10,
    )


def test_masked_fill_sdpa_fold_emits_only_with_seed():
    """comm_mul ∘ sdpa_fold_masked_fillmul: the masked_fill/-inf form.
    The fill value and mask shape are exactly the constraints the
    bounded pool cannot satisfy — the seed supplies them, and the
    composite keeps them guarded (only fires on Const fill < -1e30)."""
    seed, (q, k, v, mk) = _masked_fill_attention_seed()
    rules = [R.COMM_MUL, _sdpa_fold("sdpa_fold_masked_fillmul")]

    bare = meta.synthesize_rules(rules, fuel=2000)
    assert not _provenance_in(
        bare, ("comm_mul", "sdpa_fold_masked_fillmul")
    )

    derived = meta.synthesize_rules(rules, [seed], fuel=2000)
    hits = _provenance_in(
        derived, ("comm_mul", "sdpa_fold_masked_fillmul")
    )
    assert hits, "masked_fill sdpa fold not synthesized"
    d = hits[0]
    assert d.check is not None and d.derive is not None
    assert "logical_not" in op_repr(d.rhs)

    applied = meta.apply_rewrite_at(d, seed, ())
    assert applied is not None and applied.op == "sdpa"
    assert applied.attrs["scale"] == 0.5
    # fill-mask became the kernel's keep-mask via logical_not
    assert applied.args[3].op == "logical_not"

    # fp64 on real tensors — the mask leaf gets a boolean tensor
    env = _env((q, k, v), bools=())
    env[mk] = torch.rand(*mk.typ.shape) > 0.4
    assert meta._eval_allclose(
        meta._eval_term(seed, env),
        meta._eval_term(applied, env),
        tol=1e-10,
    )

    # vetoed instance: a non-Const fill fails the composite guard
    f_var = Var("f", _T())
    bad = Op.make(
        "matmul",
        Op.make(
            "softmax",
            Op.make(
                "masked_fill",
                Op.make(
                    "mul",
                    Const(0.5),
                    Op.make(
                        "matmul",
                        q,
                        Op.make("transpose", k, arg1=-2, arg2=-1),
                    ),
                ),
                mk,
                f_var,
            ),
            arg1=-1,
        ),
        v,
    )
    assert meta.apply_rewrite_at(d, bad, ()) is None


# ---------------------------------------------------------------------------
#  (b) a guarded SYMBOLIC candidate is rescued by a seed-mined witness
# ---------------------------------------------------------------------------


def _symbolic_candidate(r1, r2):
    """Build the symbolic composition r1∘r2 exactly as the symbolic
    path does, returning the unvalidated candidate Rewrite."""
    subst = {}
    for v in meta.pattern_metavars(r1.lhs):
        subst[v] = v[len("$attr:") :] if v.startswith("$attr:") else v
    for v in meta.pattern_metavars(r1.rhs):
        if v.startswith("$attr:") and v not in subst:
            subst[v] = "@1:" + v[len("$attr:") :]
    t1 = meta.instantiate_pattern(r1.rhs, subst)
    pat1 = {
        v: (v[len("$attr:") :] if v.startswith("$attr:") else v)
        for v in meta.pattern_metavars(r1.lhs)
    }
    for q, sub in meta._positions(t1):
        m2 = meta.match_pattern(r2.lhs, sub, {})
        if m2 is None:
            continue
        inst2 = dict(m2)
        ok = True
        for v in meta.pattern_metavars(r2.rhs):
            if v.startswith("$attr:") and v not in inst2:
                if r2.derive is None:
                    ok = False
                    break
                inst2[v] = "@2:" + v[len("$attr:") :]
        if not ok:
            continue
        t2 = meta._replace(
            t1, q, meta.instantiate_pattern(r2.rhs, inst2)
        )
        chk, drv = meta._compose_guards(r1, r2, pat1, m2)
        return Rewrite(
            name="cand", lhs=r1.lhs, rhs=t2, check=chk, derive=drv
        )
    raise AssertionError("no symbolic composition exists")


def test_seed_witness_rescues_symbolic_guarded_candidate():
    """linear_channel_scale ∘ comm_mul emits ``linear(x∘c, W) ->
    linear(x, c∘W)`` guarded by channel-scale broadcastability: c must
    be scalar or (in,) while W is a matrix.  No uniform leaf profile in
    the bounded pool gives c and W different ranks, so validation fails
    without seeds — and succeeds on a seed-mined binding."""
    r1, r2 = R.LINEAR_CHANNEL_SCALE, R.COMM_MUL
    cand = _symbolic_candidate(r1, r2)
    assert cand.check is not None

    # the bounded pool cannot satisfy the guard — rejected
    assert not meta._validate_candidate(cand, r1, r2, True)

    # a real witness satisfies it: c is (in,), W is a matrix
    x = Var("x", _T(4, 4))
    c = Var("c", _T(4))
    W = Var("W", _T(4, 4))
    seed = Op.make("linear", Op.make("mul", x, c), W)
    assert meta._validate_candidate(cand, r1, r2, True, seeds=[seed])

    # end to end: same pair emits only when the seed is passed
    bare = meta.synthesize_rules([r1, r2], fuel=2000)
    assert not _provenance_in(
        bare, ("linear_channel_scale", "comm_mul")
    )
    derived = meta.synthesize_rules([r1, r2], [seed], fuel=2000)
    hits = _provenance_in(derived, ("linear_channel_scale", "comm_mul"))
    assert hits
    d = hits[0]
    applied = meta.apply_rewrite_at(d, seed, ())
    assert applied is not None
    env = _env((x, c, W))
    assert meta._eval_allclose(
        meta._eval_term(seed, env),
        meta._eval_term(applied, env),
        tol=1e-10,
    )


# ---------------------------------------------------------------------------
#  (c) the bounded pool remains the fallback for satisfiable guards
# ---------------------------------------------------------------------------


def test_bounded_instantiation_remains_fallback():
    """An unseeded guarded pair whose side conditions ARE reachable in
    the bounded pool still emits — seeds are preferred, not required.
    Also guards the 109/109 synthesizable census."""
    rules = meta.module_rules(R) + meta.module_rules(OM)
    assert all(meta._synthesizable(r) for r in rules)

    derived = meta.synthesize_rules(
        [OM.OM_LIFT, OM.OM_UNLIFT], fuel=1000
    )
    hits = _provenance_in(derived, ("om_lift", "om_unlift"))
    assert hits, "unguarded-seed fallback regressed"
    lemma = hits[0]
    assert lemma.check is not None

    s = Var("s", _T(4, 4))
    v = Var("v", _T(4, 6))
    good = Op.make("matmul", Op.make("softmax", s, arg1=-1), v)
    bad = Op.make("matmul", Op.make("softmax", s, arg1=0), v)
    assert meta.apply_rewrite_at(lemma, good, ()) is not None
    assert meta.apply_rewrite_at(lemma, bad, ()) is None
