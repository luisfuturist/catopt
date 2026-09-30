"""Tests for the attention-path laws (:mod:`catopt_core.laws.attention`).

Coverage plan, mirroring ``test_layout_laws.py``:

* **fire + verify** — every law saturates on its canonical term and
  *every* extractable member of the class verifies against the direct
  fp64 evaluation (``assert_close`` at 1e-12; matmul members may pick
  a different kernel, so bitwise equality is not demanded);
* **decline** — each check veto is exercised: terms that structurally
  match the LHS but break the side condition must not gain the RHS
  member;
* **cost-gating** — the folds are shown cheaper under
  :func:`catopt_core.cost.flops_cost`, the model extraction prices;
* **honest reach** — the exported-IR spellings the patterns cover
  (half-split ``concat``, rotate-half ``add``, matmul-roped
  ``linear``) and the classes that decline (strided slices,
  per-position rotation tables, non-uniform factors);
* **wiring** — the ``attention`` preset composes into a ``RuleSet``
  and resolves through the real ``optimize`` path.
"""

from __future__ import annotations

import math

import pytest
import torch
from catopt_core.cost import count_cost, flops_cost
from catopt_core.egraph import EGraph
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
from catopt_core.laws import (
    ATTENTION_RULES,
    NATURALITY_SCALAR_REV,
    preset,
)
from catopt_core.laws.attention import (
    _check_left_scale,
    _check_linear_mm_absorb,
    _check_linear_mm_absorb_bias,
    _check_linear_out_scale,
    _check_rope_cat_compose,
    _check_rope_scale,
    _check_uniform,
    _half_bounds,
    _is_uniform,
    _rope_axis,
    _rope_cat,
    _rope_rh,
    _to_end,
)
from catopt_core.laws.ruleset import ATTENTION, DEFAULT, PRESETS
from catopt_torch.torch_bridge import ir_to_torch_module

torch.manual_seed(0)

_END = 1 << 62  # the int64 sentinel exports spell for x[..., h:]


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _v(name: str, *shape: int | None) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape: int | None) -> Param:
    return Param(name, TensorType(tuple(shape)))


def _saturate(term, rules=None, iters=8, max_nodes=200_000):
    """Build an e-graph over *term* and saturate with ATTENTION_RULES."""
    eg = EGraph()
    eid = eg.add_term(term)
    eg.run(
        ATTENTION_RULES if rules is None else rules,
        eid,
        max_iterations=iters,
        max_nodes=max_nodes,
    )
    return eg, eid


def _eval(term, feeds, params=None):
    """Lower *term* through IRModule and run it (fp64)."""
    xs = {k: v.to(torch.float64) for k, v in feeds.items()}
    ps = {k: v.to(torch.float64) for k, v in (params or {}).items()}
    inputs = [Var(k, TensorType(tuple(v.shape))) for k, v in xs.items()]
    pterms = {
        k: Param(k, TensorType(tuple(v.shape))) for k, v in ps.items()
    }
    mod = ir_to_torch_module(
        IR(
            root=term,
            inputs=inputs,
            input_names=set(xs),
            params=pterms,
        ),
        param_values=ps,
    )
    mod.eval()
    with torch.no_grad():
        return mod(*xs.values())


def _fired(eg, name):
    return eg.rule_fires.get(name, 0) > 0


def _members(eg, eid):
    """Every non-leaf member of the root class, each forced-extracted."""
    root = eg.find(eid)
    out = []
    for n in eg.get_class(eid).nodes:
        if n.op == "leaf":
            continue
        t = eg.extract_best(eid, count_cost, overrides={root: n})
        if t is not None:
            out.append(t)
    return out


def _check_all_members(eg, eid, lhs, feeds, ref, params=None):
    """Assert every member of the class evaluates equal to ``ref``."""
    torch.testing.assert_close(
        _eval(lhs, feeds, params), ref, rtol=0, atol=1e-12
    )
    for m in _members(eg, eid):
        got = _eval(m, feeds, params)
        torch.testing.assert_close(
            got, ref, rtol=0, atol=1e-12, msg=f"member {m!r} mismatch"
        )


def _check_declined(eg, name):
    """The rule must not have fired — no substitution passed check."""
    assert not _fired(eg, name)


def _sl(t, dim, a, e, step=None):
    """Concrete slice term; ``step`` set only when given (export form)."""
    kw = {"dim": dim, "start": a, "end": e}
    if step is not None:
        kw["step"] = step
    return Op.make("slice", t, **kw)


def _cat_rope(x, c, s, h, dim=3, cdim=-1, end=None):
    """Concrete half-split cat-rope: cat(x1·c - x2·s, x2·c + x1·s).

    ``dim`` defaults to the *positive* last-axis index — exports emit
    ``slice(dim=3)`` while ``concat`` keeps ``dim=-1`` (the slice
    binding does not normalise negative dims, so terms must spell the
    exported form).
    """
    e = _END if end is None else end
    x1 = _sl(x, dim, 0, h)
    x2 = _sl(x, dim, h, e)
    return Op.make(
        "concat",
        Op.make("sub", Op.make("mul", x1, c), Op.make("mul", x2, s)),
        Op.make("add", Op.make("mul", x2, c), Op.make("mul", x1, s)),
        dim=cdim,
    )


def _rh_rope(x, c, s, h, dim=3, cdim=-1, end=None):
    """Concrete rotate-half rope: x·c + cat(-x2, x1)·s."""
    e = _END if end is None else end
    x1 = _sl(x, dim, 0, h)
    x2 = _sl(x, dim, h, e)
    rot = Op.make("concat", Op.make("neg", x2), x1, dim=cdim)
    return Op.make("add", Op.make("mul", x, c), Op.make("mul", rot, s))


def _rope_tables(t: int, h: int, base: float = 1.0):
    """True cos/sin tables (c² + s² = 1) shaped (1,1,T,h), fp64."""
    pos = torch.arange(t, dtype=torch.float64)[:, None]
    freq = (
        base
        * torch.exp(
            -torch.arange(h, dtype=torch.float64)
            * (math.log(10_000) / h)
        )[None, :]
    )
    ang = pos * freq
    return torch.cos(ang).reshape(1, 1, t, h), torch.sin(ang).reshape(
        1, 1, t, h
    )


# ---------------------------------------------------------------------------
#  check-hook unit coverage — fabricated bound dicts, no e-graph needed
# ---------------------------------------------------------------------------


def _vbound(**kw):
    """bound dict of Var leaves + $attr entries."""
    return dict(kw)


def test_is_uniform():
    x2 = _v("t", 4, 5)
    assert _is_uniform(Const(0.5))
    assert _is_uniform(_v("u", 1, 1, 1))
    assert not _is_uniform(_v("u", 2))
    assert not _is_uniform(
        Op.make("einsum", x2, x2, equation="ij,jk->ik")
    )  # shape unknown


def test_to_end():
    assert _to_end(_END, 16)
    assert _to_end(16, 16)
    assert not _to_end(8, 16)
    assert not _to_end("e", 16)
    assert not _to_end(None, 16)


def test_rope_axis():
    b = {
        "$attr:I1D": -1,
        "$attr:I2D": 3,
        "$attr:O1D": -1,
        "$attr:O2D": 3,
        "$attr:CD1": -1,
        "$attr:CD2": 3,
    }
    assert _rope_axis(b, 4) == 3
    b["$attr:O2D"] = 1  # a different axis — not one shared rope axis
    assert _rope_axis(b, 4) is None
    b["$attr:O2D"] = "D"  # unbound metavar — not an int
    assert _rope_axis(b, 4) is None


def test_half_bounds():
    b = {
        "$attr:I1A": 0,
        "$attr:I1E": 8,
        "$attr:I2A": 8,
        "$attr:I2E": _END,
    }
    assert _half_bounds(b, "I1", "I2", 16) == 8
    b["$attr:I1A"] = 2  # does not start at 0
    assert _half_bounds(b, "I1", "I2", 16) == 0
    b["$attr:I1A"] = 0
    b["$attr:I2A"] = 9  # not contiguous
    assert _half_bounds(b, "I1", "I2", 16) == 0
    b["$attr:I2A"] = 8
    b["$attr:I2E"] = 10  # does not cover the axis
    assert _half_bounds(b, "I1", "I2", 16) == 0
    b["$attr:I2E"] = _END
    assert _half_bounds(b, "I1", "I2", 15) == 0  # odd extent
    b["$attr:I1E"] = "E"  # unbound metavar
    assert _half_bounds(b, "I1", "I2", 16) == 0


def test_check_rope_scale():
    x = _v("x", 2, 3, 8, 16)
    # scalar — always fine
    assert _check_rope_scale({"a": Const(0.5), "x": x})
    # per-row (…,1) broadcasting to x's shape
    assert _check_rope_scale({"a": _v("a", 2, 3, 8, 1), "x": x})
    # per-channel table — pair-varying, wrong
    assert not _check_rope_scale({"a": _v("a", 2, 3, 8, 16), "x": x})
    # (T,1) row table broadcasts onto x's last dims — accepted
    assert _check_rope_scale({"a": _v("a", 8, 1), "x": x})
    # last-dim-1 but the row dim mismatches — decline
    assert not _check_rope_scale({"a": _v("a", 5, 1), "x": x})
    # x shape unrecoverable — decline for non-scalar a
    y = Op.make("einsum", x, x, equation="ijkl,lm->ijkm")
    assert not _check_rope_scale({"a": _v("a", 1, 1, 1, 1), "x": y})
    # a shape unrecoverable — decline
    assert not _check_rope_scale({"a": y, "x": x})


def test_check_left_scale():
    a = _v("A", 2, 4, 5)
    b = _v("B", 2, 5, 6)
    assert _check_left_scale({"s": Const(0.5), "A": a, "B": b})
    assert _check_left_scale({"s": _v("s", 2, 4, 1), "A": a, "B": b})
    # per-KEY-column scale: (s·A)@B ≠ s·(A@B) — decline
    assert not _check_left_scale(
        {"s": _v("s", 2, 6, 1), "A": a, "B": b}
    )
    # a rank-(…,5) factor is not a row scale — decline
    assert not _check_left_scale({"s": _v("s", 5), "A": a, "B": b})
    # ill-typed product — decline
    bad = _v("B2", 2, 7, 6)
    assert not _check_left_scale({"s": Const(0.5), "A": a, "B": bad})
    # scalar A — the product is provably ill-typed too
    assert not _check_left_scale(
        {"s": Const(0.5), "A": Const(1), "B": b}
    )
    # unrecoverable factor shape — decline
    e = Op.make("einsum", a, b, equation="ijkl,lmn->ijkn")
    assert not _check_left_scale({"s": e, "A": a, "B": b})


def test_check_linear_mm_absorb():
    w = _p("W", 8, 6)
    r = _p("R", 8, 8)
    assert _check_linear_mm_absorb({"W": w, "R": r})
    # wildcards pass — only provable mismatches veto
    assert _check_linear_mm_absorb({"W": _p("W", None, 6), "R": r})
    assert _check_linear_mm_absorb({"W": w, "R": _p("R", None, 8)})
    # contraction mismatch
    assert not _check_linear_mm_absorb({"W": w, "R": _p("R", 9, 8)})
    # rank-3 per-position table — F.linear cannot take a batched weight
    assert not _check_linear_mm_absorb({"W": w, "R": _p("R", 4, 8, 8)})
    # rank ≠ 2 on either side
    assert not _check_linear_mm_absorb({"W": _p("W", 8, 6, 1), "R": r})
    e = Op.make("einsum", w, r, equation="ij,jk->ik")
    assert not _check_linear_mm_absorb({"W": e, "R": r})
    assert not _check_linear_mm_absorb({"W": w, "R": e})


def test_check_linear_mm_absorb_bias():
    w, r, b = _p("W", 8, 6), _p("R", 8, 8), _p("b", 8)
    assert _check_linear_mm_absorb_bias({"W": w, "R": r, "b": b})
    assert _check_linear_mm_absorb_bias(
        {"W": w, "R": r, "b": _p("b", None)}
    )
    # bias must be the (o,) vector of W's out dim
    assert not _check_linear_mm_absorb_bias(
        {"W": w, "R": r, "b": _p("b", 9)}
    )
    assert not _check_linear_mm_absorb_bias(
        {"W": w, "R": r, "b": _p("b", 8, 1)}
    )
    e = Op.make("einsum", b, b, equation="i,i->i")
    assert not _check_linear_mm_absorb_bias({"W": w, "R": r, "b": e})
    # base check fails propagate
    assert not _check_linear_mm_absorb_bias(
        {"W": w, "R": _p("R", 4, 8, 8), "b": b}
    )


def test_check_linear_out_scale():
    assert _check_linear_out_scale({"s": Const(2.0)})
    assert _check_linear_out_scale({"s": _v("s", 1, 1)})
    assert not _check_linear_out_scale({"s": _v("s", 8)})
    assert not _check_linear_out_scale({"s": _v("s", 2, 8, 1)})


def test_check_rope_cat_compose_unit():
    """Direct bound-level coverage of the compose check's arms."""
    x = _v("x", 2, 3, 8, 16)
    c = _v("c", 1, 1, 8, 8)
    s = _v("s", 1, 1, 8, 8)
    base = {
        "x": x,
        "C1": c,
        "C2": _v("c2", 1, 1, 8, 8),
        "S1": s,
        "S2": _v("s2", 1, 1, 8, 8),
        "$attr:I1D": -1,
        "$attr:I2D": 3,
        "$attr:O1D": -1,
        "$attr:O2D": 3,
        "$attr:CD1": -1,
        "$attr:CD2": 3,
        "$attr:I1A": 0,
        "$attr:I1E": 8,
        "$attr:I2A": 8,
        "$attr:I2E": _END,
        "$attr:O1A": 0,
        "$attr:O1E": 8,
        "$attr:O2A": 8,
        "$attr:O2E": _END,
    }
    assert _check_rope_cat_compose(dict(base))
    b = dict(base, x=Const(0))  # ()-shaped x — no axis
    assert not _check_rope_cat_compose(b)
    b = dict(base, x=Op.make("einsum", x, x, equation="i,i->i"))
    assert not _check_rope_cat_compose(b)  # unrecoverable x shape
    b = dict(base)
    b["$attr:O1D"] = 1  # axes disagree
    assert not _check_rope_cat_compose(b)
    b = dict(base, x=_v("x", 2, 3, 8, None))  # extent unknown
    assert not _check_rope_cat_compose(b)
    b = dict(base)
    b["$attr:I1E"] = 7  # inner split not the half
    assert not _check_rope_cat_compose(b)
    b = dict(base)
    b["$attr:O2A"] = 7  # outer cut off the boundary
    assert not _check_rope_cat_compose(b)
    b = dict(base, C1=Op.make("einsum", c, c, equation="i,i->i"))
    assert not _check_rope_cat_compose(b)  # c1 shape unknown
    b = dict(base, C2=_v("c2", 8, 8))  # different broadcast shape
    assert not _check_rope_cat_compose(b)
    b = dict(base, S1=Op.make("einsum", s, s, equation="i,i->i"))
    assert not _check_rope_cat_compose(b)
    b = dict(base, S2=_v("s2", 8, 8))
    assert not _check_rope_cat_compose(b)


def test_check_uniform():
    assert _check_uniform({"a": Const(1.0)})
    assert not _check_uniform({"a": _v("a", 3)})


# ---------------------------------------------------------------------------
#  Rotary composition — rope_cat_compose
# ---------------------------------------------------------------------------


def test_rope_cat_compose_fires_and_verifies():
    """rope₂∘rope₁ gains the composed single-rope member; all verify."""
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c1, s1 = _p("c1", 1, 1, T, dh), _p("s1", 1, 1, T, dh)
    c2, s2 = _p("c2", 1, 1, T, dh), _p("s2", 1, 1, T, dh)
    lhs = _cat_rope(_cat_rope(x, c1, s1, dh), c2, s2, dh)
    eg, eid = _saturate(lhs, rules=[_r_by_name("rope_cat_compose")])
    assert _fired(eg, "rope_cat_compose")
    assert list(eg.matches(_r_by_name("rope_cat_compose").rhs, eid))
    feeds = {"x": torch.randn(B, H, T, 2 * dh)}
    params = {
        "c1": torch.randn(1, 1, T, dh),
        "s1": torch.randn(1, 1, T, dh),
        "c2": torch.randn(1, 1, T, dh),
        "s2": torch.randn(1, 1, T, dh),
    }
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)


def test_rope_cat_compose_unrope_recovers_x():
    """unrope∘rope composes to a single rope with θ≈0 — every member
    evaluates ≈ x on true rotation tables (c² + s² = 1 lives below the
    term algebra; the *numeric* identity is what verifies)."""
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c, s = _p("c", 1, 1, T, dh), _p("s", 1, 1, T, dh)
    ns = Op.make("neg", s)  # unrope = same cos, negated sin
    lhs = _cat_rope(_cat_rope(x, c, s, dh), c, ns, dh)
    eg, eid = _saturate(lhs, rules=[_r_by_name("rope_cat_compose")])
    assert _fired(eg, "rope_cat_compose")
    feeds = {"x": torch.randn(B, H, T, 2 * dh)}
    cv, sv = _rope_tables(T, dh)
    params = {"c": cv, "s": sv}
    ref = _eval(x, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)


def _r_by_name(name):
    return next(r for r in ATTENTION_RULES if r.name == name)


def test_rope_cat_compose_declines():
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c1, s1 = _p("c1", 1, 1, T, dh), _p("s1", 1, 1, T, dh)
    c2, s2 = _p("c2", 1, 1, T, dh), _p("s2", 1, 1, T, dh)
    rule = _r_by_name("rope_cat_compose")

    # Outer slices cut off the concat boundary (h+1 instead of h).
    inner = _cat_rope(x, c1, s1, dh)
    o1 = _sl(inner, 3, 0, dh + 1)
    o2 = _sl(inner, 3, dh + 1, _END)
    bad = Op.make(
        "concat",
        Op.make("sub", Op.make("mul", o1, c2), Op.make("mul", o2, s2)),
        Op.make("add", Op.make("mul", o2, c2), Op.make("mul", o1, s2)),
        dim=-1,
    )
    eg, _ = _saturate(bad, rules=[rule])
    _check_declined(eg, "rope_cat_compose")

    # Outer slices on a different axis than the inner concat's.
    o1 = _sl(inner, 2, 0, T)
    o2 = _sl(inner, 2, T, _END)
    bad = Op.make(
        "concat",
        Op.make("sub", Op.make("mul", o1, c2), Op.make("mul", o2, s2)),
        Op.make("add", Op.make("mul", o2, c2), Op.make("mul", o1, s2)),
        dim=-1,
    )
    eg, _ = _saturate(bad, rules=[rule])
    _check_declined(eg, "rope_cat_compose")

    # Factor tables broadcast differently — composition unsound.
    c2_flat = _p("c2f", T, dh)  # (T,h) vs c1's (1,1,T,h)
    lhs = _cat_rope(inner, c2_flat, s2, dh)
    eg, _ = _saturate(lhs, rules=[rule])
    _check_declined(eg, "rope_cat_compose")

    # Strided slices are a different rope spelling — the unstrided
    # pattern deliberately does not match (attr key-sets differ).
    st1 = _sl(x, 3, 0, _END, step=2)
    st2 = _sl(x, 3, 1, _END, step=2)
    st_rope = Op.make(
        "stack",
        Op.make(
            "sub", Op.make("mul", st1, c1), Op.make("mul", st2, s1)
        ),
        Op.make(
            "add", Op.make("mul", st2, c1), Op.make("mul", st1, s1)
        ),
        dim=-1,
    )
    eg, _ = _saturate(st_rope, rules=[rule])
    _check_declined(eg, "rope_cat_compose")


# ---------------------------------------------------------------------------
#  Rotary scale commutation — rope_*_scale_in / _out
# ---------------------------------------------------------------------------


def test_rope_cat_scale_out_and_in():
    """a·rope(x) ↔ rope(a·x) — scalar and per-row factors, both ways."""
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c, s = _p("c", 1, 1, T, dh), _p("s", 1, 1, T, dh)
    a = _p("a", 1, 1, 1, 1)  # uniform broadcast — pair-constant
    xa = Op.make("mul", x, a)
    lhs = _cat_rope(xa, c, s, dh)
    rules = [
        _r_by_name("rope_cat_scale_out"),
        _r_by_name("rope_cat_scale_in"),
    ]
    eg, eid = _saturate(lhs, rules=rules)
    # scale_out mints mul(rope(x), a); scale_in then re-mints the
    # original member — a no-op union, counted only as a match, not a
    # merge.  The pulled-out member is what fires.
    assert _fired(eg, "rope_cat_scale_out")
    feeds = {"x": torch.randn(B, H, T, 2 * dh)}
    cv, sv = _rope_tables(T, dh)
    params = {"c": cv, "s": sv, "a": torch.randn(1, 1, 1, 1)}
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)


def test_rope_cat_scale_scalar():
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c, s = _p("c", 1, 1, T, dh), _p("s", 1, 1, T, dh)
    a = Const(0.5)
    lhs = Op.make("mul", _cat_rope(x, c, s, dh), a)
    rules = [_r_by_name("rope_cat_scale_in")]
    eg, eid = _saturate(lhs, rules=rules)
    assert _fired(eg, "rope_cat_scale_in")
    feeds = {"x": torch.randn(B, H, T, 2 * dh)}
    cv, sv = _rope_tables(T, dh)
    params = {"c": cv, "s": sv}
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)


def test_rope_cat_scale_declines_per_channel():
    """A per-channel factor (…,2h) breaks the rotary pairs — decline."""
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c, s = _p("c", 1, 1, T, dh), _p("s", 1, 1, T, dh)
    a = _p("a", 1, 1, 1, 2 * dh)  # pair-varying — not commutable
    lhs = _cat_rope(Op.make("mul", x, a), c, s, dh)
    eg, _ = _saturate(lhs, rules=[_r_by_name("rope_cat_scale_out")])
    _check_declined(eg, "rope_cat_scale_out")


def test_rope_rh_scale_out_and_in():
    """The rotate-half add spelling commutes a uniform factor too."""
    B, H, T, dh = 2, 3, 8, 8
    x = _v("x", B, H, T, 2 * dh)
    c, s = _p("c", 1, 1, T, 2 * dh), _p("s", 1, 1, T, 2 * dh)
    a = _p("a", 1, 1, 1, 1)
    lhs = _rh_rope(Op.make("mul", x, a), c, s, dh)
    rules = [
        _r_by_name("rope_rh_scale_out"),
        _r_by_name("rope_rh_scale_in"),
    ]
    eg, eid = _saturate(lhs, rules=rules)
    assert _fired(eg, "rope_rh_scale_out")
    feeds = {"x": torch.randn(B, H, T, 2 * dh)}
    params = {
        "c": torch.randn(1, 1, T, 2 * dh),
        "s": torch.randn(1, 1, T, 2 * dh),
        "a": torch.randn(1, 1, 1, 1),
    }
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)


# ---------------------------------------------------------------------------
#  Right-multiply absorb — linear_mm_absorb(_bias)(_rev)
# ---------------------------------------------------------------------------


def test_linear_mm_absorb_fires_and_is_picked():
    """matmul(linear(x,W), R) gains linear(x, RᵀW); extraction picks it
    — the weight product is a small param-only mul vs a B·T runtime
    GEMM."""
    B, T, i, o, p = 2, 8, 6, 8, 8
    x, w, r = _v("x", B, T, i), _p("W", o, i), _p("R", o, p)
    lhs = Op.make("matmul", Op.make("linear", x, w), r)
    eg, eid = _saturate(lhs, rules=[_r_by_name("linear_mm_absorb")])
    assert _fired(eg, "linear_mm_absorb")
    assert list(eg.matches(_r_by_name("linear_mm_absorb").rhs, eid))
    feeds = {"x": torch.randn(B, T, i)}
    params = {"W": torch.randn(o, i), "R": torch.randn(o, p)}
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)
    best = eg.extract_best(eid, flops_cost)
    assert flops_cost(best) <= flops_cost(lhs)
    torch.testing.assert_close(
        _eval(best, feeds, params), ref, rtol=0, atol=1e-12
    )
    # the picked form carries the folded weight product
    assert "matmul" in repr(best) and "transpose" in repr(best)


def test_linear_mm_absorb_rev():
    B, T, i, o, p = 2, 8, 6, 8, 8
    x, w, r = _v("x", B, T, i), _p("W", o, i), _p("R", o, p)
    lhs = Op.make(
        "linear",
        x,
        Op.make(
            "matmul",
            Op.make("transpose", r, dim0=-2, dim1=-1),
            w,
        ),
    )
    eg, eid = _saturate(lhs, rules=[_r_by_name("linear_mm_absorb_rev")])
    assert _fired(eg, "linear_mm_absorb_rev")
    feeds = {"x": torch.randn(B, T, i)}
    params = {"W": torch.randn(o, i), "R": torch.randn(o, p)}
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)


def test_linear_mm_absorb_rotary_matrix():
    """The R = rotation-matrix case: position-independent rope-by-
    matmul absorbs into W — the runtime matmul disappears."""
    B, T, i, o = 2, 8, 8, 8
    x, w, r = _v("x", B, T, i), _p("W", o, i), _p("R", o, o)
    lhs = Op.make("matmul", Op.make("linear", x, w), r)
    eg, eid = _saturate(lhs, rules=[_r_by_name("linear_mm_absorb")])
    th = 0.7
    rot = torch.tensor(
        [
            [math.cos(th), -math.sin(th), 0.0, 0.0, 0, 0, 0, 0],
            [math.sin(th), math.cos(th), 0.0, 0.0, 0, 0, 0, 0],
            [0.0, 0.0, math.cos(th), -math.sin(th), 0, 0, 0, 0],
            [0.0, 0.0, math.sin(th), math.cos(th), 0, 0, 0, 0],
            *[[float(j == k) for j in range(8)] for k in range(4, 8)],
        ],
        dtype=torch.float64,
    )[:o]
    feeds = {"x": torch.randn(B, T, i)}
    params = {"W": torch.randn(o, i), "R": rot}
    _check_all_members(
        eg, eid, lhs, feeds, _eval(lhs, feeds, params), params
    )


def test_linear_mm_absorb_bias():
    B, T, i, o, p = 2, 8, 6, 8, 8
    x = _v("x", B, T, i)
    w, b, r = _p("W", o, i), _p("b", o), _p("R", o, p)
    lhs = Op.make("matmul", Op.make("linear", x, w, b), r)
    rules = [
        _r_by_name("linear_mm_absorb_bias"),
        _r_by_name("linear_mm_absorb_bias_rev"),
    ]
    eg, eid = _saturate(lhs, rules=rules)
    assert _fired(eg, "linear_mm_absorb_bias")
    # the rev direction re-mints the already-present member — a no-op
    # union (matched, not merged); it fires on a graph seeded with
    # only the folded member.
    folded = Op.make(
        "linear",
        x,
        Op.make("matmul", Op.make("transpose", r, dim0=-2, dim1=-1), w),
        Op.make("matmul", b, r),
    )
    eg2, _eid2 = _saturate(
        folded, rules=[_r_by_name("linear_mm_absorb_bias_rev")]
    )
    assert _fired(eg2, "linear_mm_absorb_bias_rev")
    feeds = {"x": torch.randn(B, T, i)}
    params = {
        "W": torch.randn(o, i),
        "b": torch.randn(o),
        "R": torch.randn(o, p),
    }
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)


def test_linear_mm_absorb_declines():
    o, i = 8, 6
    x, w = _v("x", 2, 8, i), _p("W", o, i)
    rule = _r_by_name("linear_mm_absorb")
    # Per-position rotation table — not a fixed weight; F.linear
    # cannot take a rank-3 weight either.
    r3 = _p("R", 8, o, o)
    eg, _ = _saturate(
        Op.make("matmul", Op.make("linear", x, w), r3), rules=[rule]
    )
    _check_declined(eg, "linear_mm_absorb")
    # Contraction mismatch — R does not consume the linear's out dim.
    rbad = _p("R", o + 1, o)
    eg, _ = _saturate(
        Op.make("matmul", Op.make("linear", x, w), rbad), rules=[rule]
    )
    _check_declined(eg, "linear_mm_absorb")


# ---------------------------------------------------------------------------
#  Score-scale migration — naturality_scalar_left, linear_out_scale,
#  and the view commutations they chain through.
# ---------------------------------------------------------------------------


def test_naturality_scalar_left_both_directions():
    B, m, k, n = 2, 4, 5, 6
    a, b = _v("A", B, m, k), _v("B", B, k, n)
    s = Const(0.5)
    lhs = Op.make("mul", Op.make("matmul", a, b), s)
    rules = [
        _r_by_name("naturality_scalar_left"),
        _r_by_name("naturality_scalar_left_rev"),
    ]
    eg, eid = _saturate(lhs, rules=rules)
    assert _fired(eg, "naturality_scalar_left_rev")
    # the pull-out direction re-mints the seeded member — a no-op
    # union here; it merges when the graph starts inside-out.
    eg2, _ = _saturate(
        Op.make("matmul", Op.make("mul", a, s), b),
        rules=[_r_by_name("naturality_scalar_left")],
    )
    assert _fired(eg2, "naturality_scalar_left")
    feeds = {"A": torch.randn(B, m, k), "B": torch.randn(B, k, n)}
    ref = _eval(lhs, feeds)
    _check_all_members(eg, eid, lhs, feeds, ref)


def test_naturality_scalar_left_row_scale():
    """A per-row (…,1) factor slides too — (s·A)@B = s·(A@B)."""
    B, m, k, n = 2, 4, 5, 6
    a, b = _v("A", B, m, k), _v("B", B, k, n)
    s = _v("s", B, m, 1)
    lhs = Op.make("matmul", Op.make("mul", a, s), b)
    eg, eid = _saturate(
        lhs, rules=[_r_by_name("naturality_scalar_left")]
    )
    assert _fired(eg, "naturality_scalar_left")
    feeds = {
        "A": torch.randn(B, m, k),
        "B": torch.randn(B, k, n),
        "s": torch.randn(B, m, 1),
    }
    ref = _eval(lhs, feeds)
    _check_all_members(eg, eid, lhs, feeds, ref)


def test_linear_out_scale_both_directions():
    B, T, i, o = 2, 8, 6, 8
    x, w = _v("x", B, T, i), _p("W", o, i)
    s = Const(0.5)
    lhs = Op.make("mul", Op.make("linear", x, w), s)
    rules = [
        _r_by_name("linear_out_scale"),
        _r_by_name("linear_out_scale_rev"),
    ]
    eg, eid = _saturate(lhs, rules=rules)
    assert _fired(eg, "linear_out_scale")
    # the rev direction re-mints the seeded member — merges only on
    # a graph seeded with the folded member.
    eg2, _ = _saturate(
        Op.make("linear", x, Op.make("mul", w, s)),
        rules=[_r_by_name("linear_out_scale_rev")],
    )
    assert _fired(eg2, "linear_out_scale_rev")
    feeds = {"x": torch.randn(B, T, i)}
    params = {"W": torch.randn(o, i)}
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)
    best = eg.extract_best(eid, flops_cost)
    assert flops_cost(best) <= flops_cost(lhs)


def test_linear_out_scale_declines():
    B, T, i, o = 2, 8, 6, 8
    x, w = _v("x", B, T, i), _p("W", o, i)
    # Per-out-channel scale: mul(W, s) would mis-broadcast — decline.
    s = _p("s", o)
    eg, _ = _saturate(
        Op.make("mul", Op.make("linear", x, w), s),
        rules=[_r_by_name("linear_out_scale")],
    )
    _check_declined(eg, "linear_out_scale")
    # Per-row scale: belongs to LINEAR_ROW_SCALE, not the weight fold.
    s = _v("s", B, T, 1)
    eg, _ = _saturate(
        Op.make("mul", Op.make("linear", x, w), s),
        rules=[_r_by_name("linear_out_scale")],
    )
    _check_declined(eg, "linear_out_scale")


# ---------------------------------------------------------------------------
#  Uniform-factor view commutation — per-op fire + verify + one decline
# ---------------------------------------------------------------------------


def _view_cases():
    """op-name -> (term over metavar-free leaves, t's shape, out shape)."""
    t = _v("t", 2, 8)
    return {
        "transpose": Op.make("transpose", t, dim0=0, dim1=1),
        "reshape": Op.make("reshape", t, shape=(4, 4)),
        "slice": Op.make("slice", t, dim=1, start=0, end=4),
        "slice_step": Op.make(
            "slice", t, dim=1, start=0, end=8, step=2
        ),
        "unbind": Op.make("unbind", t, dim=0, index=1),
        "chunk": Op.make("chunk", t, chunks=2, dim=1, index=0),
        "flatten": Op.make("flatten", t, start_dim=0),
        "flatten2": Op.make("flatten", t, start_dim=0, end_dim=1),
        "unsqueeze": Op.make("unsqueeze", t, dim=1),
        "squeeze": Op.make("squeeze", _v("t1", 2, 1, 8), dim=1),
    }


@pytest.mark.parametrize("name", sorted(_view_cases()))
def test_scale_view_unary(name):
    view = _view_cases()[name]
    a = Const(0.5)
    lhs = Op.make("mul", view, a)
    rules = [
        _r_by_name(f"scale_in_{name}"),
        _r_by_name(f"scale_out_{name}"),
    ]
    eg, eid = _saturate(lhs, rules=rules)
    assert _fired(eg, f"scale_in_{name}")
    feeds = {"t": torch.randn(2, 8)}
    if "t1" in repr(view):
        feeds = {"t1": torch.randn(2, 1, 8)}
    ref = _eval(lhs, feeds)
    _check_all_members(eg, eid, lhs, feeds, ref)


@pytest.mark.parametrize("name", ["concat", "stack"])
def test_scale_view_binary(name):
    t1, t2 = _v("t1", 2, 8), _v("t2", 2, 8)
    a = Const(0.5)
    shared = Op.make(
        name, Op.make("mul", t1, a), Op.make("mul", t2, a), dim=1
    )
    rules = [
        _r_by_name(f"scale_in_{name}"),
        _r_by_name(f"scale_out_{name}"),
    ]
    eg, eid = _saturate(shared, rules=rules)
    assert _fired(eg, f"scale_out_{name}")
    # scale_in re-mints the seeded member (no-op union) — it merges
    # on a graph seeded with the joined form.
    joined = Op.make("mul", Op.make(name, t1, t2, dim=1), a)
    eg2, _ = _saturate(joined, rules=[_r_by_name(f"scale_in_{name}")])
    assert _fired(eg2, f"scale_in_{name}")
    feeds = {"t1": torch.randn(2, 8), "t2": torch.randn(2, 8)}
    ref = _eval(shared, feeds)
    _check_all_members(eg, eid, shared, feeds, ref)


def test_scale_view_declines_nonuniform():
    t = _v("t", 2, 8)
    a = _v("a", 8)  # per-column factor — not uniform, wrong side
    lhs = Op.make("mul", Op.make("transpose", t, dim0=0, dim1=1), a)
    eg, _ = _saturate(lhs, rules=[_r_by_name("scale_in_transpose")])
    _check_declined(eg, "scale_in_transpose")


# ---------------------------------------------------------------------------
#  End-to-end: score scale folds into a projection weight
# ---------------------------------------------------------------------------


def _head(term, b, t, h, dh):
    """The head-splitting view: reshape to (B,T,H,dh) then transpose."""
    return Op.make(
        "transpose",
        Op.make("reshape", term, shape=(b, t, h, dh)),
        dim0=1,
        dim1=2,
    )


def test_score_scale_folds_into_wq():
    """mul(q̂k̂ᵀ, s) saturates to the form with s inside Wq — a
    compile-time (o,i) mul replacing a runtime (B,H,T,T) one — and
    extraction under flops_cost prefers it."""
    B, H, T, i, dh = 2, 4, 8, 6, 8
    o = H * dh
    x = _v("x", B, T, i)
    wq, wk = _p("Wq", o, i), _p("Wk", o, i)
    hq = _head(Op.make("linear", x, wq), B, T, H, dh)
    hk = _head(Op.make("linear", x, wk), B, T, H, dh)
    kt = Op.make("transpose", hk, dim0=-2, dim1=-1)
    lhs = Op.make("mul", Op.make("matmul", hq, kt), Const(0.5))
    eg, eid = _saturate(lhs, iters=10)
    assert _fired(eg, "naturality_scalar_left_rev")
    assert _fired(eg, "linear_out_scale")
    feeds = {"x": torch.randn(B, T, i)}
    params = {"Wq": torch.randn(o, i), "Wk": torch.randn(o, i)}
    ref = _eval(lhs, feeds, params)
    best = eg.extract_best(eid, flops_cost)
    assert flops_cost(best) < flops_cost(lhs)
    torch.testing.assert_close(
        _eval(best, feeds, params), ref, rtol=0, atol=1e-12
    )
    # a member carrying the weight-folded Wq exists in the class —
    # linear(x, mul(Wq, s)) nested inside the scores structure.
    def _contains(t, pred):
        if isinstance(t, Op):
            return pred(t) or any(
                _contains(a, pred) for a in t.args
            )
        return pred(t)

    def _weight_folded(t):
        return (
            isinstance(t, Op)
            and t.op == "linear"
            and isinstance(t.args[1], Op)
            and t.args[1].op == "mul"
        )

    assert any(
        _contains(m, _weight_folded) for m in _members(eg, eid)
    )


def test_score_scale_can_fold_into_wk_via_existing_law():
    """The right-side path: NATURALITY_SCALAR_REV pushes s into k̂ᵀ and
    the view commutes route it to Wk — existing + new rules compose."""
    B, H, T, i, dh = 2, 4, 8, 6, 8
    o = H * dh
    x = _v("x", B, T, i)
    wq, wk = _p("Wq", o, i), _p("Wk", o, i)
    hq = _head(Op.make("linear", x, wq), B, T, H, dh)
    hk = _head(Op.make("linear", x, wk), B, T, H, dh)
    kt = Op.make("transpose", hk, dim0=-2, dim1=-1)
    lhs = Op.make("mul", Op.make("matmul", hq, kt), Const(0.5))
    eg, eid = _saturate(
        lhs, rules=[*ATTENTION_RULES, NATURALITY_SCALAR_REV], iters=10
    )
    assert _fired(eg, "naturality_scalar_rev")
    feeds = {"x": torch.randn(B, T, i)}
    params = {"Wq": torch.randn(o, i), "Wk": torch.randn(o, i)}
    ref = _eval(lhs, feeds, params)
    _check_all_members(eg, eid, lhs, feeds, ref, params)


# ---------------------------------------------------------------------------
#  Wiring — the attention preset, its opt-in status, and the pipeline
# ---------------------------------------------------------------------------


def test_attention_preset_wiring():
    assert PRESETS["attention"] is ATTENTION
    assert preset("attention") is ATTENTION
    assert set(ATTENTION.rules) == set(ATTENTION_RULES)
    # opt-in only: nothing leaks into DEFAULT
    assert not (set(ATTENTION.rules) & set(DEFAULT.rules))
    # composes by name with the default set
    both = DEFAULT + ATTENTION
    assert "rope_cat_compose" in both
    assert "sdpa_fold_addmul" in both
    # every rule carries the ATTENTION tag
    assert all("attention" in r.tags for r in ATTENTION_RULES)


def test_attention_preset_through_optimizer():
    """rules='attention' resolves through the real pipeline and the
    MiniRope rope spelling still lowers to the same output."""
    import torch.nn as nn

    class MiniRope(nn.Module):
        def forward(self, x, fc, fs):
            b, t, h, d = x.shape
            xr, xi = x.reshape(b, t, h, d // 2, 2).unbind(-1)
            fc = fc.view(1, t, 1, d // 2)
            fs = fs.view(1, t, 1, d // 2)
            out_r = xr * fc - xi * fs
            out_i = xr * fs + xi * fc
            return torch.stack([out_r, out_i], dim=-1).flatten(3)

    from catopt_orchestrator import Optimizer
    from catopt_torch.backend import TorchBackend

    m = MiniRope().eval()
    x = torch.randn(2, 8, 4, 16)
    fc = torch.randn(8, 8)
    fs = torch.randn(8, 8)
    opt, _stats = Optimizer(backend=TorchBackend()).optimize(
        m, (x, fc, fs), rules="attention", verify=False, verbose=False
    )
    with torch.no_grad():
        diff = (m(x, fc, fs) - opt(x, fc, fs)).abs().max().item()
    assert diff < 1e-4


def test_rope_cat_and_rh_pattern_builders():
    """The pattern builders mint the exported spellings (guard against
    drift between law patterns and the IR boundary)."""
    pat = _rope_cat("x", "C", "S", "I1", "I2", "CD")
    assert pat.op == "concat"
    sub, add = pat.args
    assert sub.op == "sub" and add.op == "add"
    assert sub.args[0].args[0].op == "slice"
    rh = _rope_rh("x", "C", "S", "I1", "I2", "CD")
    assert rh.op == "add" and rh.args[1].args[0].op == "concat"
    assert rh.args[1].args[0].args[0].op == "neg"


def test_exported_rope_term_matches_law_pattern():
    """The real exported MiniRope produces terms the laws reason
    about — the half-split cat form matches _rope_cat's structure."""
    import torch.nn as nn
    from catopt_torch.torch_bridge import export_to_ir

    class SliceRope(nn.Module):
        def forward(self, x, cos, sin):
            hd = x.shape[-1]
            x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
            c = cos[None, None, :, :]
            s = sin[None, None, :, :]
            return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)

    m = SliceRope().eval()
    x = torch.randn(2, 4, 8, 16)
    fc, fs = torch.randn(8, 8), torch.randn(8, 8)
    ir, _src = export_to_ir(m, (x, fc, fs))
    eg = EGraph()
    eid = eg.add_term(ir.root)
    # the exported root matches the law's inner-rope pattern shape:
    # a concat whose children are sub/add of slice·table muls
    pat = _rope_cat("x", "C", "S", "I1", "I2", "CD")
    assert list(eg.matches(pat, eid))
