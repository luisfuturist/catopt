"""Tests for catopt.trace — JSV traced-monoidal axioms as e-graph rules.

Each axiom is checked two ways:

* the rewrite FIRES in the e-graph (``rule_fires`` / ``matches`` on the
  root e-class), and
* lhs ≡ rhs numerically in fp64 through the torch op bindings (the
  same ``_IR_TO_TORCH`` table ``IRModule._eval`` consults).

The payoff test builds the time-extended recurrence
``h_t = A·h_{t−1} + B·x_t`` as a trace (nilpotent block shift Z, so the
fixpoint is the unrolled scan EXACTLY), and verifies:

* ``Tr(F)`` applied to the input vector equals the step-by-step loop,
* it equals the affine-scan carrier fold
  ``apply(aff_compose^T(…), h0)``, and
* ``tr_expand`` reaches the closed resolvent form in the e-graph.

The nontrivial-transform test is ``tr_superpose``: a joint loop over
two independent recurrence channels splits into a block-diagonal of
independent traces — channels that may be scheduled in parallel.
"""

import pytest
import torch

import catopt.trace as cat_trace  # noqa: F401  (registers torch bindings)
from catopt.cost import count_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Const, Op, Param, TensorType, Var
from catopt.torch_bridge import _IR_TO_TORCH, ir_to_torch_module
from catopt.trace import (
    TR_COLLAPSE,
    TR_EXPAND,
    TR_SLIDE,
    TR_SLIDE_REV,
    TR_SUPERPOSE,
    TR_SUPERPOSE_REV,
    TR_TIGHTEN_IN,
    TR_TIGHTEN_IN_REV,
    TR_TIGHTEN_OUT,
    TR_TIGHTEN_OUT_REV,
    TR_VANISH_MERGE,
    TR_VANISH_SPLIT,
    TR_VANISH_UNIT,
    TR_YANK,
    TRACE_LAWS,
)


@pytest.fixture(autouse=True)
def _fp64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _rand(seed: int, *shape: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _mkf(
    du: int, dx: int, dy: int, seed: int, s_scale: float = 0.25
) -> torch.Tensor:
    """Random f : U⊗X → U⊗Y as a feedback-first block matrix
    ``[[S, R], [Q, P]]`` with contractive S (well-posed fixpoint)."""
    S = _rand(seed + 1, du, du) * s_scale
    R = _rand(seed + 2, du, dx) * 0.4
    Q = _rand(seed + 3, dy, du) * 0.4
    P = _rand(seed + 4, dy, dx) * 0.4
    return torch.cat(
        [torch.cat([S, R], dim=1), torch.cat([Q, P], dim=1)], dim=0
    )


def _var(name: str, shape) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _ev(term, env: dict | None = None) -> torch.Tensor:
    """Evaluate a term through the torch op table (the same table
    ``IRModule._eval`` consults — ``catopt.trace`` registered its ops
    there at import)."""
    env = env or {}

    def go(t):
        if isinstance(t, (Var, Param)):
            return env[t.name]
        if isinstance(t, Const):
            return torch.tensor(
                t.value, dtype=torch.get_default_dtype()
            )
        return _IR_TO_TORCH[t.op](
            *[go(a) for a in t.args], **dict(t.attrs)
        )

    return go(term)


def _all_nodes(eg: EGraph):
    for ec in eg._classes.values():
        yield from ec.nodes


def _trace_enode(eg: EGraph, usize):
    return [
        n
        for n in _all_nodes(eg)
        if n.op == "trace" and dict(n.attrs).get("usize") == usize
    ]


# ---------------------------------------------------------------------------
#  Semantics: trace IS the linear fixpoint
# ---------------------------------------------------------------------------


def test_trace_is_linear_fixpoint():
    du, dx, dy = 3, 4, 2
    F = _mkf(du, dx, dy, seed=0)
    term = Op.make("trace", _var("F", F.shape), usize=du)
    got = _ev(term, {"F": F})

    S, R = F[:du, :du], F[:du, du:]
    Q, P = F[du:, :du], F[du:, du:]
    # iterate the loop to its fixpoint: u* = S·u* + R
    u = torch.zeros(du, dx)
    for _ in range(400):
        u = S @ u + R
    want = Q @ u + P
    assert torch.allclose(got, want, atol=1e-10)

    # closed form (the definition): P + Q·(I−S)⁻¹·R
    closed = P + Q @ torch.linalg.solve(torch.eye(du) - S, R)
    assert (got - closed).abs().max() < 1e-12


def test_irmodule_evaluates_trace():
    """The ops flow through the real IRModule path, not just the raw
    op table."""
    du, dx, dy = 2, 3, 2
    F = _mkf(du, dx, dy, seed=5)
    Fp = Param("p_F", TensorType(tuple(F.shape)))
    term = Op.make("trace", Fp, usize=du)
    ir = IR(
        root=term,
        inputs=[_var("x", (dx, dy))],
        input_names={"x"},
        params={"p_F": Fp},
    )
    mod = ir_to_torch_module(ir, param_values={"p_F": F})
    x = _rand(6, dx, dy)
    got = mod(x)
    want = F[du:, du:] + F[du:, :du] @ torch.linalg.solve(
        torch.eye(du) - F[:du, :du], F[:du, du:]
    )
    assert torch.allclose(got @ x, want @ x, atol=1e-10)
    assert torch.allclose(got, want, atol=1e-10)


# ---------------------------------------------------------------------------
#  Vanishing — Tr^I = id, Tr^{U⊗V} = Tr^V ∘ Tr^U
# ---------------------------------------------------------------------------


def test_vanishing_unit():
    du, dx, dy = 3, 4, 2
    F = _mkf(du, dx, dy, seed=10)
    Fv = _var("F", F.shape)
    term = Op.make("trace", Fv, usize=0)
    # usize = 0: tracing the trivial unit is already the identity
    assert torch.equal(_ev(term, {"F": F}), F)

    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([TR_VANISH_UNIT], eid)
    assert eg.rule_fires.get("tr_vanish_unit", 0) >= 1
    best = eg.extract_best(eid, count_cost)
    assert isinstance(best, Var) and best.name == "F"


def test_vanishing_split_and_merge():
    du, dv, dx, dy = 2, 3, 4, 2
    F = _mkf(du + dv, dx, dy, seed=20)
    Fv = _var("F", F.shape)
    lhs = Op.make("trace", Fv, usize=(du, dv))
    rhs = Op.make("trace", Op.make("trace", Fv, usize=du), usize=dv)
    # numerics: product-wire trace == nested traces
    assert (_ev(lhs, {"F": F}) - _ev(rhs, {"F": F})).abs().max() < 1e-12

    # split fires
    eg = EGraph()
    eid = eg.add_term(lhs)
    eg.run([TR_VANISH_SPLIT], eid)
    assert eg.rule_fires.get("tr_vanish_split", 0) >= 1
    assert eg.matches(
        Op.make("trace", Op.make("trace", "f", usize="DU"), usize="DV"),
        eid,
    )

    # merge fires back
    eg2 = EGraph()
    eid2 = eg2.add_term(rhs)
    eg2.run([TR_VANISH_MERGE], eid2)
    assert eg2.rule_fires.get("tr_vanish_merge", 0) >= 1
    assert _trace_enode(eg2, (du, dv))


# ---------------------------------------------------------------------------
#  Superposing — joint loop splits into independent channels
# ---------------------------------------------------------------------------


def test_superpose_splits_independent_channels():
    """The headline transform: one joint loop over two independent
    recurrences → block-diagonal of two independent traces that can be
    scheduled in parallel."""
    du, dv, dx, dc, dy, dd = 2, 3, 2, 2, 2, 2
    F = _mkf(du, dx, dy, seed=30)
    G = _mkf(dv, dc, dd, seed=40)
    Fv, Gv = _var("F", F.shape), _var("G", G.shape)
    env = {"F": F, "G": G}

    lhs = Op.make(
        "trace", Op.make("parl", Fv, Gv, u1=du, u2=dv), usize=(du, dv)
    )
    rhs = Op.make(
        "bdiag",
        Op.make("trace", Fv, usize=du),
        Op.make("trace", Gv, usize=dv),
    )
    assert (_ev(lhs, env) - _ev(rhs, env)).abs().max() < 1e-12

    eg = EGraph()
    eid = eg.add_term(lhs)
    eg.run([TR_SUPERPOSE], eid)
    assert eg.rule_fires.get("tr_superpose", 0) >= 1
    # the split form is a member of the root e-class
    assert eg.matches(
        Op.make(
            "bdiag",
            Op.make("trace", "f", usize="DU"),
            Op.make("trace", "g", usize="DV"),
        ),
        eid,
    )

    # reverse direction: independent loops fuse back into the joint loop
    eg2 = EGraph()
    eid2 = eg2.add_term(rhs)
    eg2.run([TR_SUPERPOSE_REV], eid2)
    assert eg2.rule_fires.get("tr_superpose_rev", 0) >= 1
    assert _trace_enode(eg2, (du, dv))


def test_superpose_with_untraced_context():
    """Superposing with u2 = 0: an untraced map g simply rides along —
    the joint loop splits into (loop over f) ⊕ (plain g)."""
    du, dx, dy, dc, dd = 2, 3, 2, 4, 5
    F = _mkf(du, dx, dy, seed=50)
    G = _rand(51, dd, dc)  # g : C → D, no feedback wire at all
    Fv, Gv = _var("F", F.shape), _var("G", G.shape)
    env = {"F": F, "G": G}

    lhs = Op.make(
        "trace", Op.make("parl", Fv, Gv, u1=du, u2=0), usize=(du, 0)
    )
    rhs = Op.make(
        "bdiag",
        Op.make("trace", Fv, usize=du),
        Op.make("trace", Gv, usize=0),
    )
    assert (_ev(lhs, env) - _ev(rhs, env)).abs().max() < 1e-12

    # tr_superpose + tr_vanish_unit chain: g's trivial trace vanishes
    eg = EGraph()
    eid = eg.add_term(lhs)
    eg.run([TR_SUPERPOSE, TR_VANISH_UNIT], eid)
    assert eg.rule_fires.get("tr_superpose", 0) >= 1
    assert eg.rule_fires.get("tr_vanish_unit", 0) >= 1
    # root e-class contains bdiag(trace(F, du), G)
    assert eg.matches(
        Op.make("bdiag", Op.make("trace", "f", usize="DU"), "g"), eid
    )


# ---------------------------------------------------------------------------
#  Sliding — a map on the loop wire crosses the trace boundary
# ---------------------------------------------------------------------------


def _slide_terms(du, dx, dy, seed):
    G = _mkf(du, dx, dy, seed=seed)
    H = _rand(seed + 10, du, du) * 0.5
    Gv, Hv = _var("G", G.shape), _var("H", H.shape)
    lhs = Op.make(
        "trace",
        Op.make(
            "matmul", Gv, Op.make("bdiag", Hv, Op.make("eye", dim=dx))
        ),
        usize=du,
    )
    rhs = Op.make(
        "trace",
        Op.make(
            "matmul", Op.make("bdiag", Hv, Op.make("eye", dim=dy)), Gv
        ),
        usize=du,
    )
    return lhs, rhs, {"G": G, "H": H}


def test_slide():
    du, dx, dy = 2, 3, 2
    lhs, rhs, env = _slide_terms(du, dx, dy, 60)
    # numerics: H·(I−SH)⁻¹ = (I−HS)⁻¹·H
    assert (_ev(lhs, env) - _ev(rhs, env)).abs().max() < 1e-12

    eg = EGraph()
    eid = eg.add_term(lhs)
    eg.run([TR_SLIDE], eid)
    assert eg.rule_fires.get("tr_slide", 0) >= 1
    assert eg.matches(
        Op.make(
            "trace",
            Op.make(
                "matmul",
                Op.make("bdiag", "h", Op.make("eye", dim="DY")),
                "g",
            ),
            usize="DU",
        ),
        eid,
    )

    # reverse direction
    eg2 = EGraph()
    eid2 = eg2.add_term(rhs)
    eg2.run([TR_SLIDE_REV], eid2)
    assert eg2.rule_fires.get("tr_slide_rev", 0) >= 1


def test_slide_pulls_matrix_out_of_recurrence():
    """Sliding relocates the per-step map: an h applied INSIDE the loop
    on every iteration is algebraically the same loop with h applied
    on the loop's output instead — the eqsat can pick whichever side
    is cheaper (e.g. h fused into the surrounding affine map)."""
    du, dx, dy = 3, 2, 2
    lhs, rhs, env = _slide_terms(du, dx, dy, 70)
    eg = EGraph()
    eid = eg.add_term(lhs)
    eg.run([TR_SLIDE], eid)
    # both orientations coexist in the class
    assert eg.rule_fires.get("tr_slide", 0) >= 1
    assert eg.matches(
        Op.make(
            "trace",
            Op.make(
                "matmul",
                "g",
                Op.make("bdiag", "h", Op.make("eye", dim="DX")),
            ),
            usize="DU",
        ),
        eid,
    )
    assert eg.matches(
        Op.make(
            "trace",
            Op.make(
                "matmul",
                Op.make("bdiag", "h", Op.make("eye", dim="DY")),
                "g",
            ),
            usize="DU",
        ),
        eid,
    )


# ---------------------------------------------------------------------------
#  Tightening — context maps commute out of the loop
# ---------------------------------------------------------------------------


def test_tighten_out():
    du, dx, dy, dz = 2, 3, 4, 1
    F = _mkf(du, dx, dy, seed=80)
    K = _rand(81, dz, dy) * 0.5  # readout: y → z
    Fv, Kv = _var("F", F.shape), _var("K", K.shape)
    env = {"F": F, "K": K}

    lhs = Op.make(
        "trace",
        Op.make(
            "matmul", Op.make("bdiag", Op.make("eye", dim=du), Kv), Fv
        ),
        usize=du,
    )
    rhs = Op.make("matmul", Kv, Op.make("trace", Fv, usize=du))
    assert (_ev(lhs, env) - _ev(rhs, env)).abs().max() < 1e-12

    eg = EGraph()
    eid = eg.add_term(lhs)
    eg.run([TR_TIGHTEN_OUT], eid)
    assert eg.rule_fires.get("tr_tighten_out", 0) >= 1
    assert eg.matches(
        Op.make("matmul", "k", Op.make("trace", "f", usize="DU")), eid
    )

    eg2 = EGraph()
    eid2 = eg2.add_term(rhs)
    eg2.run([TR_TIGHTEN_OUT_REV], eid2)
    assert eg2.rule_fires.get("tr_tighten_out_rev", 0) >= 1


def test_tighten_in():
    du, dx, dy, dw = 2, 3, 2, 4
    F = _mkf(du, dx, dy, seed=90)
    J = _rand(91, dx, dw) * 0.5  # lift: w → x
    Fv, Jv = _var("F", F.shape), _var("J", J.shape)
    env = {"F": F, "J": J}

    lhs = Op.make(
        "trace",
        Op.make(
            "matmul", Fv, Op.make("bdiag", Op.make("eye", dim=du), Jv)
        ),
        usize=du,
    )
    rhs = Op.make("matmul", Op.make("trace", Fv, usize=du), Jv)
    assert (_ev(lhs, env) - _ev(rhs, env)).abs().max() < 1e-12

    eg = EGraph()
    eid = eg.add_term(lhs)
    eg.run([TR_TIGHTEN_IN], eid)
    assert eg.rule_fires.get("tr_tighten_in", 0) >= 1
    assert eg.matches(
        Op.make("matmul", Op.make("trace", "f", usize="DU"), "j"), eid
    )

    eg2 = EGraph()
    eid2 = eg2.add_term(rhs)
    eg2.run([TR_TIGHTEN_IN_REV], eid2)
    assert eg2.rule_fires.get("tr_tighten_in_rev", 0) >= 1


# ---------------------------------------------------------------------------
#  Yanking — a crossover loop is the identity
# ---------------------------------------------------------------------------


def test_yank():
    d = 4
    swap = Op.make("cswap", d1=d, d2=d)
    term = Op.make("trace", swap, usize=d)
    # Tr(σ_{U,U}) = I_U: S = 0 ⇒ Tr = I·(I−0)⁻¹·I = I
    got = _ev(term, {})
    assert torch.allclose(got, torch.eye(d), atol=1e-12)

    eg = EGraph()
    eid = eg.add_term(term)
    eg.run([TR_YANK], eid)
    assert eg.rule_fires.get("tr_yank", 0) >= 1
    best = eg.extract_best(eid, count_cost)
    assert (
        isinstance(best, Op)
        and best.op == "eye"
        and best.attrs["dim"] == d
    )


# ---------------------------------------------------------------------------
#  Closed form — iterative ↔ resolvent
# ---------------------------------------------------------------------------


def test_expand_reaches_closed_form():
    du, dx, dy = 3, 4, 2
    F = _mkf(du, dx, dy, seed=100)
    Fv = _var("F", F.shape)
    lhs = Op.make("trace", Fv, usize=du)

    # the expanded resolvent term, built explicitly for numerics
    def blk(t, sizes, dim, idx):
        return Op.make("split", t, sizes=sizes, dim=dim, index=idx)

    rows_u = blk(Fv, (du, dy), -2, 0)
    rows_y = blk(Fv, (du, dy), -2, 1)
    S = blk(rows_u, (du, dx), -1, 0)
    R = blk(rows_u, (du, dx), -1, 1)
    Q = blk(rows_y, (du, dx), -1, 0)
    P = blk(rows_y, (du, dx), -1, 1)
    rhs = Op.make(
        "add",
        P,
        Op.make(
            "matmul",
            Q,
            Op.make(
                "matmul",
                Op.make(
                    "inv", Op.make("sub", Op.make("eye", dim=du), S)
                ),
                R,
            ),
        ),
    )
    env = {"F": F}
    assert (_ev(lhs, env) - _ev(rhs, env)).abs().max() < 1e-12

    eg = EGraph()
    eid = eg.add_term(lhs)
    eg.run([TR_EXPAND], eid)
    assert eg.rule_fires.get("tr_expand", 0) >= 1
    # closed-form structure present in the graph
    assert any(n.op == "inv" for n in _all_nodes(eg))


def test_collapse_folds_closed_form_back():
    du, dx, dy = 2, 3, 2
    F = _mkf(du, dx, dy, seed=110)
    Fv = _var("F", F.shape)

    def blk(t, sizes, dim, idx):
        return Op.make("split", t, sizes=sizes, dim=dim, index=idx)

    rows_u = blk(Fv, (du, dy), -2, 0)
    rows_y = blk(Fv, (du, dy), -2, 1)
    closed = Op.make(
        "add",
        blk(rows_y, (du, dx), -1, 1),
        Op.make(
            "matmul",
            blk(rows_y, (du, dx), -1, 0),
            Op.make(
                "matmul",
                Op.make(
                    "inv",
                    Op.make(
                        "sub",
                        Op.make("eye", dim=du),
                        blk(rows_u, (du, dx), -1, 0),
                    ),
                ),
                blk(rows_u, (du, dx), -1, 1),
            ),
        ),
    )
    eg = EGraph()
    eid = eg.add_term(closed)
    eg.run([TR_COLLAPSE], eid)
    assert eg.rule_fires.get("tr_collapse", 0) >= 1
    assert eg.matches(Op.make("trace", "f", usize="DU"), eid)


# ---------------------------------------------------------------------------
#  Payoff: the recurrence h_t = A·h_{t−1} + B·x_t is a trace
# ---------------------------------------------------------------------------


def test_recurrence_is_trace_and_matches_aff_carrier():
    """The payoff: h_t = A·h_{t−1} + B·x_t as a trace equals both the
    unrolled loop AND the affine-scan carrier fold — fp64 exact, since
    the shift is nilpotent (no convergence caveat)."""
    torch.manual_seed(0)
    T, d, dx = 6, 3, 2
    A = _rand(120, d, d) * 0.3
    B = _rand(121, d, dx) * 0.5
    h0 = _rand(122, d)
    X = _rand(123, T, dx)

    du, n_in = T * d, T * dx + d
    F = torch.zeros(du + T * d, du + n_in)
    # S = Z·A_blk (block t, t−1), nilpotent ⇒ fixpoint = finite unroll
    for t in range(1, T):
        F[t * d : (t + 1) * d, (t - 1) * d : t * d] = A
    # R : [x_1..x_T; h0] → [u'_1..u'_T]
    for t in range(T):
        F[t * d : (t + 1) * d, du + t * dx : du + (t + 1) * dx] = B
    F[0:d, du + T * dx : du + T * dx + d] = A  # h0 → u'_1
    # Q = I (u → y), P = 0
    F[du : du + T * d, :du] = torch.eye(T * d)

    Fv = _var("F", F.shape)
    tr_term = Op.make("trace", Fv, usize=du)
    tr_val = _ev(tr_term, {"F": F})  # (T·d) × (T·dx + d)
    vec = torch.cat([X.reshape(-1), h0])
    got = tr_val @ vec  # h_1..h_T

    # 1) the unrolled loop
    h = h0.clone()
    series = []
    for t in range(T):
        h = A @ h + B @ X[t]
        series.append(h)
    want = torch.cat(series)
    assert torch.allclose(got, want, atol=1e-12)

    # 2) the affine-scan carrier fold: apply(aff_compose^T …, h0)
    aff = _IR_TO_TORCH["aff"]
    comp = _IR_TO_TORCH["aff_compose"]
    apply = _IR_TO_TORCH["apply"]
    f = aff(A, B @ X[0])
    for t in range(1, T):
        f = comp(aff(A, B @ X[t]), f)
    hT = apply(f, h0)
    assert torch.allclose(got[-d:], hT, atol=1e-12)

    # 3) the trace reaches the closed resolvent form in the e-graph
    eg = EGraph()
    eid = eg.add_term(tr_term)
    eg.run([TR_EXPAND], eid)
    assert eg.rule_fires.get("tr_expand", 0) >= 1


def test_trace_laws_are_shape_checked():
    """A law must NOT fire on ill-typed wiring: slide needs h on the
    u-block (du×du); an h of the wrong size is vetoed by the check."""
    du, dx, dy = 2, 3, 2
    G = _mkf(du, dx, dy, seed=130)
    H_bad = _rand(131, dx, dx) * 0.5  # dx×dx, not du×du — wrong wire
    Gv, Hv = _var("G", G.shape), _var("H", H_bad.shape)
    bad = Op.make(
        "trace",
        Op.make(
            "matmul", Gv, Op.make("bdiag", Hv, Op.make("eye", dim=dx))
        ),
        usize=du,
    )
    # bdiag(dx×dx, I_dx) is (2dx)×(2dx) = 6×6 ≠ (du+dx)×(du+dx) = 5×5
    # for du≠dx — matmul is already ill-typed; check vetoes.
    eg = EGraph()
    eid = eg.add_term(bad)
    eg.run([TR_SLIDE], eid)
    assert eg.rule_fires.get("tr_slide", 0) == 0


def test_trace_laws_collection_runs_together():
    """TRACE_LAWS saturates without runaway on a mixed term and keeps
    every form reachable."""
    du, dv, dx, dy = 2, 2, 2, 2
    F = _mkf(du + dv, dx, dy, seed=140)
    Fv = _var("F", F.shape)
    term = Op.make("trace", Fv, usize=(du, dv))
    eg = EGraph()
    eid = eg.add_term(term)
    stats = eg.run(TRACE_LAWS, eid, max_iterations=10, max_nodes=20_000)
    assert stats["n_enodes"] < 20_000
    # split form reachable
    assert eg.matches(
        Op.make("trace", Op.make("trace", "f", usize="DU"), usize="DV"),
        eid,
    )
    # closed form reachable
    assert any(n.op == "inv" for n in _all_nodes(eg))
