"""law_bench — per-rewrite-law value harness.

Adding a rewrite law to ``catopt_core.laws`` is cheap; knowing
whether it *pays* is the part that used to be expensive.  This
harness turns the value question into a one-command measurement.
For each named law it

1. builds a small synthetic term where the law's LHS applies (the
   ``LAW_CASES`` registry — one builder per rule name; a name with
   no builder is reported honestly as ``no-case``),
2. interns the term in an ``EGraph`` and runs THAT LAW ALONE to
   saturation (``eg.run([rule], ...)``),
3. reports whether the law fired (``eg.rule_fires``), whether its
   RHS landed in the root e-class (``list(eg.matches(rule.rhs, root))``),
   and whether cost-model extraction actually *picked* the new
   member — firing is necessary but not sufficient: an equal-cost
   or pricier member is found but never selected,
4. lowers the input and extracted terms through the same
   ``_lower_extracted`` path ``optimize_model`` uses (carrier-apply
   roots route to the level-batched executors; everything else
   lowers through ``TorchSink`` to ``IRModule``, so param-only
   subtrees fold at compile time exactly as in the pipeline), and
   verifies them against each other with ``sink.verify`` — the
   pipeline's own equivalence gate,
5. times both lowered modules with the benchkit ``Runner``
   (``blocked_autorange`` medians, CUDA-synced on GPU) — the
   measured before/after the law's value is priced on.

A law that never fires — wrong term shape, a ``check`` veto — has
measured value exactly zero; the table says so instead of hiding
the row.  Extraction is priced through
``backend_cost(executor_cost_for(lowering="generic"),
sink.supported_ops)``, the same model ``optimize_model`` selects
with, so "picked?" answers "would the pipeline even choose this
form?".

Usage:
    python bench/law_bench.py --laws id_add,assoc_linear --sizes 256
    python bench/law_bench.py --laws default --sizes 128,256
    python bench/law_bench.py --laws simpl --quick
    python bench/law_bench.py --list          # registry contents

``--laws`` accepts comma-separated rule names, group names
(``simpl``, ``categorical``, ``sdpa_fold``, ``scan``, ``scan_diag``,
``all``), or ``default`` (the curated set below).  Group expansion
silently skips rules with no registered case; explicit names get an
honest ``no-case`` row.  ``--sizes`` is a per-case scale knob —
usually the feature dim ``d``; attention cases derive head dims
from it.
"""
# ruff: noqa: RUF003 — sys.path setup precedes imports;
# math notation (→, ×) in strings is deliberate, per bench convention.

from __future__ import annotations

import argparse
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from benchkit import Case, Report, Runner, Variant, collect_env
from catopt_torch.adapters import TorchSink
from catopt_core.cost import backend_cost, dag_cost, executor_cost_for
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var, op_repr
from catopt_orchestrator.optimize import _lower_extracted


from catopt_torch.torch_bridge import IRModule  # noqa: F401 — docstring ref
from catopt_core.laws import (
    ALL_RULES,
    CATEGORICAL_RULES,
    SCAN_DIAG_LAWS,
    SCAN_LAWS,
    SDPA_FOLD_RULES,
    SIMPLIFICATION_RULES,
)

_B = 512  # batch rows for the matmul/elementwise cases
_SEED = 20240

#: Harnessed-suite overrides for ``run_all.py --quick``.
QUICK = {"sizes": "128", "min_run_time": 0.05}


# ---------------------------------------------------------------------------
#  Term builders — the synthetic "where the law applies" cases
# ---------------------------------------------------------------------------


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(name: str, *shape: int) -> Param:
    return Param(name, TensorType(tuple(shape)))


def _r(dev: torch.device, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, device=dev)


def _xy_op(op: str):
    """``op(x, y)`` over two (B, d) Vars — the monoid-law case."""

    def build(d: int, dev: torch.device):
        x, y = _v("x", _B, d), _v("y", _B, d)
        env = {"x": _r(dev, _B, d), "y": _r(dev, _B, d)}
        return Op.make(op, x, y), env, [x, y]

    return build


def _assoc3(op: str):
    """``op(x, op(y, z))`` — the right-nested associativity case."""

    def build(d: int, dev: torch.device):
        x, y, z = _v("x", _B, d), _v("y", _B, d), _v("z", _B, d)
        env = {n: _r(dev, _B, d) for n in "xyz"}
        return Op.make(op, x, Op.make(op, y, z)), env, [x, y, z]

    return build


def _one_var(leaf_fn):
    """A term over a single (B, d) Var — ``leaf_fn(x) -> term``."""

    def build(d: int, dev: torch.device):
        x = _v("x", _B, d)
        return leaf_fn(x), {"x": _r(dev, _B, d)}, [x]

    return build


def _silu_mul(d: int, dev: torch.device):
    g, u = _v("g", _B, d), _v("u", _B, d)
    env = {"g": _r(dev, _B, d), "u": _r(dev, _B, d)}
    return Op.make("mul", Op.make("silu", g), u), env, [g, u]


# -- matmul / linear family ------------------------------------------------


def _c_distribute(d: int, dev: torch.device):
    w, a, b = _p("W", d, d), _v("a", d, _B), _v("b", d, _B)
    env = {"W": _r(dev, d, d), "a": _r(dev, d, _B), "b": _r(dev, d, _B)}
    t = Op.make("matmul", w, Op.make("add", a, b))
    return t, env, [a, b]


def _c_factor(d: int, dev: torch.device):
    w, a, b = _p("W", d, d), _v("a", d, _B), _v("b", d, _B)
    env = {"W": _r(dev, d, d), "a": _r(dev, d, _B), "b": _r(dev, d, _B)}
    t = Op.make("add", Op.make("matmul", w, a), Op.make("matmul", w, b))
    return t, env, [a, b]


def _c_right_distribute(d: int, dev: torch.device):
    w, a, b = _p("W", d, d), _v("a", _B, d), _v("b", _B, d)
    env = {"W": _r(dev, d, d), "a": _r(dev, _B, d), "b": _r(dev, _B, d)}
    t = Op.make("matmul", Op.make("add", a, b), w)
    return t, env, [a, b]


def _c_right_factor(d: int, dev: torch.device):
    w, a, b = _p("W", d, d), _v("a", _B, d), _v("b", _B, d)
    env = {"W": _r(dev, d, d), "a": _r(dev, _B, d), "b": _r(dev, _B, d)}
    t = Op.make("add", Op.make("matmul", a, w), Op.make("matmul", b, w))
    return t, env, [a, b]


def _c_weight_factor(d: int, dev: torch.device):
    x = _v("x", _B, d)
    w1, w2 = _p("W", d, d), _p("W2", d, d)
    env = {
        "x": _r(dev, _B, d),
        "W": _r(dev, d, d),
        "W2": _r(dev, d, d),
    }
    t = Op.make(
        "add", Op.make("matmul", x, w1), Op.make("matmul", x, w2)
    )
    return t, env, [x]


def _c_weight_distribute(d: int, dev: torch.device):
    x = _v("x", _B, d)
    w1, w2 = _p("W", d, d), _p("W2", d, d)
    env = {
        "x": _r(dev, _B, d),
        "W": _r(dev, d, d),
        "W2": _r(dev, d, d),
    }
    t = Op.make("matmul", x, Op.make("add", w1, w2))
    return t, env, [x]


def _linear_case(d: int, dev: torch.device, form: str):
    """Shared env for the F.linear cases: x (B,d) Vars/Params (d,d)."""
    x = _v("x", _B, d)
    w1, w2 = _p("W", d, d), _p("W2", d, d)
    env = {
        "x": _r(dev, _B, d),
        "W": _r(dev, d, d),
        "W2": _r(dev, d, d),
    }
    if form == "factor":
        t = Op.make(
            "add", Op.make("linear", x, w1), Op.make("linear", x, w2)
        )
    else:  # distribute
        t = Op.make("linear", x, Op.make("add", w1, w2))
    return t, env, [x]


def _c_right_factor_linear(d: int, dev: torch.device):
    w, a, b = _p("W", d, d), _v("a", _B, d), _v("b", _B, d)
    env = {"W": _r(dev, d, d), "a": _r(dev, _B, d), "b": _r(dev, _B, d)}
    t = Op.make("add", Op.make("linear", a, w), Op.make("linear", b, w))
    return t, env, [a, b]


def _c_assoc_linear(d: int, dev: torch.device):
    x = _v("x", _B, d)
    a, b = _p("A", d, d), _p("B", d, d)
    env = {"x": _r(dev, _B, d), "A": _r(dev, d, d), "B": _r(dev, d, d)}
    t = Op.make("linear", Op.make("linear", x, a), b)
    return t, env, [x]


def _biased_chain(d: int, dev: torch.device, fused: bool):
    """The biased compose case in both spellings (fwd / rev laws)."""
    x = _v("x", _B, d)
    a, b = _p("A", d, d), _p("B", d, d)
    b1, b2 = _p("b1", d), _p("b2", d)
    env = {
        "x": _r(dev, _B, d),
        "A": _r(dev, d, d),
        "B": _r(dev, d, d),
        "b1": _r(dev, d),
        "b2": _r(dev, d),
    }
    if fused:
        t = Op.make(
            "add",
            Op.make(
                "linear",
                x,
                Op.make("matmul", b, a),
                Op.make("matmul", b, b1),
            ),
            b2,
        )
    else:
        t = Op.make("linear", Op.make("linear", x, a, b1), b, b2)
    return t, env, [x]


def _c_naturality(d: int, dev: torch.device):
    x = _v("x", d, _B)
    w, c = _p("W", d, d), _p("c")
    env = {
        "x": _r(dev, d, _B),
        "W": _r(dev, d, d),
        "c": torch.randn((), device=dev),
    }
    t = Op.make("matmul", w, Op.make("mul", x, c))
    return t, env, [x]


def _c_naturality_rev(d: int, dev: torch.device):
    x = _v("x", d, _B)
    w, c = _p("W", d, d), _p("c")
    env = {
        "x": _r(dev, d, _B),
        "W": _r(dev, d, d),
        "c": torch.randn((), device=dev),
    }
    t = Op.make("mul", Op.make("matmul", w, x), c)
    return t, env, [x]


def _c_assoc_matmul(d: int, dev: torch.device):
    x = _v("x", _B, d)
    w1, w2 = _p("W1", d, d), _p("W2", d, d)
    env = {
        "x": _r(dev, _B, d),
        "W1": _r(dev, d, d),
        "W2": _r(dev, d, d),
    }
    t = Op.make("matmul", x, Op.make("matmul", w1, w2))
    return t, env, [x]


def _c_assoc_matmul_rev(d: int, dev: torch.device):
    """A left-associated 4-chain — the reassoc_scale miniature."""
    x = _v("x", _B, d)
    ws = [_p(f"W{i}", d, d) for i in range(1, 5)]
    env = {"x": _r(dev, _B, d)}
    env.update({w.name: _r(dev, d, d) for w in ws})
    t = x
    for w in ws:
        t = Op.make("matmul", t, w)
    return t, env, [x]


def _c_swiglu(d: int, dev: torch.device):
    x = _v("x", _B, d)
    a, b = _p("A", d, d), _p("B", d, d)
    env = {"x": _r(dev, _B, d), "A": _r(dev, d, d), "B": _r(dev, d, d)}
    t = Op.make(
        "mul",
        Op.make("silu", Op.make("linear", x, a)),
        Op.make("linear", x, b),
    )
    return t, env, [x]


def _c_parallel_mul(d: int, dev: torch.device):
    x = _v("x", _B, d)
    a, b = _p("A", d, d), _p("B", d, d)
    env = {"x": _r(dev, _B, d), "A": _r(dev, d, d), "B": _r(dev, d, d)}
    t = Op.make("mul", Op.make("linear", x, a), Op.make("linear", x, b))
    return t, env, [x]


def _head_view(t: Any, b: int, tt: int, h: int, dh: int) -> Op:
    return Op.make(
        "transpose",
        Op.make("reshape", t, shape=(b, tt, h, dh)),
        dim0=1,
        dim1=2,
    )


def _c_qkv(d: int, dev: torch.device):
    """Symmetric fused-QKV case: three equal projections + sdpa."""
    b, tt, h = 2, 64, 4
    dh, in_d = max(8, d // 8), d
    out = h * dh
    x = _v("x", b, tt, in_d)
    qs, ks, vs = (_p(n, out, in_d) for n in ("Q", "K", "V"))
    env = {"x": _r(dev, b, tt, in_d) * 0.3}
    for w in (qs, ks, vs):
        env[w.name] = _r(dev, out, in_d) * 0.1
    sc = 1.0 / math.sqrt(dh)
    t = Op.make(
        "sdpa",
        _head_view(Op.make("linear", x, qs), b, tt, h, dh),
        _head_view(Op.make("linear", x, ks), b, tt, h, dh),
        _head_view(Op.make("linear", x, vs), b, tt, h, dh),
        scale=sc,
    )
    return t, env, [x]


def _c_qkv_asym(d: int, dev: torch.device):
    """GQA case: 4 q heads vs 2 kv heads — uneven split sizes."""
    b, tt, hq, hk = 2, 64, 4, 2
    dh, in_d = max(8, d // 8), d
    x = _v("x", b, tt, in_d)
    q = _p("Q", hq * dh, in_d)
    k, v = _p("K", hk * dh, in_d), _p("V", hk * dh, in_d)
    env = {"x": _r(dev, b, tt, in_d) * 0.3}
    for w in (q, k, v):
        env[w.name] = _r(dev, *w.typ.shape) * 0.1
    sc = 1.0 / math.sqrt(dh)
    t = Op.make(
        "sdpa",
        _head_view(Op.make("linear", x, q), b, tt, hq, dh),
        _head_view(Op.make("linear", x, k), b, tt, hk, dh),
        _head_view(Op.make("linear", x, v), b, tt, hk, dh),
        scale=sc,
        enable_gqa=True,
    )
    return t, env, [x]


def _c_channel_scale(d: int, dev: torch.device):
    x = _v("x", _B, d)
    c, w = _p("c", d), _p("W", d, d)
    env = {"x": _r(dev, _B, d), "c": _r(dev, d), "W": _r(dev, d, d)}
    t = Op.make("linear", Op.make("mul", x, c), w)
    return t, env, [x]


def _c_channel_scale_rev(d: int, dev: torch.device):
    x = _v("x", _B, d)
    c, w = _p("c", d), _p("W", d, d)
    env = {"x": _r(dev, _B, d), "c": _r(dev, d), "W": _r(dev, d, d)}
    t = Op.make("linear", x, Op.make("mul", w, c))
    return t, env, [x]


def _c_row_scale(d: int, dev: torch.device):
    x, r = _v("x", _B, d), _v("r", _B, 1)
    w = _p("W", d, d)
    env = {"x": _r(dev, _B, d), "r": _r(dev, _B, 1), "W": _r(dev, d, d)}
    t = Op.make("linear", Op.make("mul", x, r), w)
    return t, env, [x, r]


def _c_row_scale_rev(d: int, dev: torch.device):
    x, r = _v("x", _B, d), _v("r", _B, 1)
    w = _p("W", d, d)
    env = {"x": _r(dev, _B, d), "r": _r(dev, _B, 1), "W": _r(dev, d, d)}
    t = Op.make("mul", Op.make("linear", x, w), r)
    return t, env, [x, r]


def _c_gqa_absorb(d: int, dev: torch.device):
    """unsqueeze→expand→reshape kv-repeat inside sdpa."""
    b, tt, hq, hk, r = 2, 64, 8, 2, 4
    dh = max(8, d // 8)
    q = _v("q", b, tt, hq, dh)
    k, v = _v("k", b, tt, hk, dh), _v("v", b, tt, hk, dh)
    env = {
        "q": _r(dev, b, tt, hq, dh),
        "k": _r(dev, b, tt, hk, dh),
        "v": _r(dev, b, tt, hk, dh),
    }

    def rep(t: Var) -> Op:
        return Op.make(
            "transpose",
            Op.make(
                "reshape",
                Op.make(
                    "expand",
                    Op.make("unsqueeze", t, dim=3),
                    shape=(b, tt, hk, r, dh),
                ),
                shape=(b, tt, hk * r, dh),
            ),
            dim0=1,
            dim1=2,
        )

    t = Op.make(
        "sdpa",
        Op.make("transpose", q, dim0=1, dim1=2),
        rep(k),
        rep(v),
        arg4=0.0,
        arg5=False,
    )
    return t, env, [q, k, v]


# -- softmax-attention fold cases ------------------------------------------


def _causal_add_mask(b: int, h: int, t: int, dev: torch.device):
    """(B,H,T,T) additive causal mask: 0 keep, -inf masked."""
    tril = torch.tril(torch.ones(t, t, dtype=torch.bool))
    m = torch.zeros(t, t, device=dev).masked_fill(
        ~tril.to(dev), -math.inf
    )
    return m.expand(b, h, t, t).contiguous()


def _causal_bool_mask(b: int, h: int, t: int, dev: torch.device):
    """(B,H,T,T) bool mask: True = masked-out position."""
    m = ~torch.tril(torch.ones(t, t, dtype=torch.bool))
    return m.expand(b, h, t, t).contiguous().to(dev)


def _sdpa_scores_case(
    d: int, dev: torch.device, masked_fill: bool, scaled: bool
):
    b, h, tt = 2, 4, 64
    dh = max(8, d // 8)
    q, k, v = (_v(n, b, h, tt, dh) for n in "qkv")
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1)
    )
    if scaled:
        scores = Op.make("mul", scores, Const(1.0 / math.sqrt(dh)))
    if masked_fill:
        mk = _v("mk", b, h, tt, tt)
        inner = Op.make("masked_fill", scores, mk, Const(float("-inf")))
        env_extra = {"mk": _causal_bool_mask(b, h, tt, dev)}
        inputs = [q, k, v, mk]
    else:
        m = _v("m", b, h, tt, tt)
        inner = Op.make("add", scores, m)
        env_extra = {"m": _causal_add_mask(b, h, tt, dev)}
        inputs = [q, k, v, m]
    sm = Op.make("softmax", inner, dim=-1)
    env = {n: _r(dev, b, h, tt, dh) for n in "qkv"}
    env.update(env_extra)
    return Op.make("matmul", sm, v), env, inputs


# -- scan-monoid cases -------------------------------------------------------


def _c_aff_lift(d: int, dev: torch.device):
    a = _p("A", d, d)
    h, x = _v("h", d), _v("x", d)
    env = {
        "A": _r(dev, d, d) * 0.1,
        "h": _r(dev, d),
        "x": _r(dev, d),
    }
    t = Op.make("add", Op.make("matmul", a, h), x)
    return t, env, [h, x]


def _c_affd_lift(d: int, dev: torch.device):
    a = _p("a", d)
    h, x = _v("h", d), _v("x", d)
    env = {"a": _r(dev, d), "h": _r(dev, d), "x": _r(dev, d)}
    t = Op.make("add", Op.make("mul", a, h), x)
    return t, env, [h, x]


def _c_square(d: int, dev: torch.device):
    x = _v("x", _B, d)
    return Op.make("square", x), {"x": _r(dev, _B, d)}, [x]


def _c_pow(d: int, dev: torch.device):
    x = _v("x", _B, d)
    t = Op.make("pow", x, Const(2))
    return t, {"x": _r(dev, _B, d)}, [x]


# ---------------------------------------------------------------------------
#  Registry: rule name -> synthetic case builder
# ---------------------------------------------------------------------------

LAW_CASES: dict[str, Any] = {
    # SIMPLIFICATION_RULES
    "comm_add": _xy_op("add"),
    "comm_mul": _xy_op("mul"),
    "assoc_add": _assoc3("add"),
    "assoc_mul": _assoc3("mul"),
    "id_add": _one_var(lambda x: Op.make("add", x, Const(0))),
    "id_mul": _one_var(lambda x: Op.make("mul", x, Const(1))),
    "double_neg": _one_var(lambda x: Op.make("neg", Op.make("neg", x))),
    "sub_to_add": _xy_op("sub"),
    "silu_expand": _one_var(lambda x: Op.make("silu", x)),
    "silu_mul_form": _silu_mul,
    "square_expand": _c_square,
    "square_to_pow": _c_square,
    "pow_to_square": _c_pow,
    # CATEGORICAL_RULES — bilinearity / merges
    "distribute_matmul_over_add": _c_distribute,
    "factor_matmul": _c_factor,
    "right_distribute_matmul": _c_right_distribute,
    "right_factor_matmul": _c_right_factor,
    "weight_factor_matmul": _c_weight_factor,
    "weight_distribute_matmul": _c_weight_distribute,
    "weight_factor_linear": lambda d, dev: _linear_case(
        d, dev, "factor"
    ),
    "weight_distribute_linear": lambda d, dev: _linear_case(
        d, dev, "distribute"
    ),
    "right_factor_linear": _c_right_factor_linear,
    "assoc_linear": _c_assoc_linear,
    "assoc_linear_bias": lambda d, dev: _biased_chain(d, dev, False),
    "assoc_linear_bias_rev": lambda d, dev: _biased_chain(d, dev, True),
    "naturality_scalar": _c_naturality,
    "naturality_scalar_rev": _c_naturality_rev,
    "assoc_matmul": _c_assoc_matmul,
    "assoc_matmul_rev": _c_assoc_matmul_rev,
    # product structure / norm folding / attention
    "swiglu_fuse": _c_swiglu,
    "parallel_mul_fuse": _c_parallel_mul,
    "qkv_fuse": _c_qkv,
    "qkv_fuse_asym": _c_qkv_asym,
    "linear_channel_scale": _c_channel_scale,
    "linear_channel_scale_rev": _c_channel_scale_rev,
    "linear_row_scale": _c_row_scale,
    "linear_row_scale_rev": _c_row_scale_rev,
    "gqa_absorb_repeat": _c_gqa_absorb,
    "sdpa_fold_addmul": lambda d, dev: _sdpa_scores_case(
        d, dev, masked_fill=False, scaled=True
    ),
    "sdpa_fold_add": lambda d, dev: _sdpa_scores_case(
        d, dev, masked_fill=False, scaled=False
    ),
    "sdpa_fold_masked_fillmul": lambda d, dev: _sdpa_scores_case(
        d, dev, masked_fill=True, scaled=True
    ),
    # SCAN_LAWS / SCAN_DIAG_LAWS
    "aff_lift": _c_aff_lift,
    "affd_lift": _c_affd_lift,
}

#: The curated set: laws that fire on simple terms and span the
#: interesting behaviours — trivial simplifications, equal-cost
#: reorderings, members found but not selected, and real measured
#: wins (weight folds, merged GEMMs, the SDPA fold, a scan lift).
DEFAULT_LAWS = [
    "id_add",
    "double_neg",
    "sub_to_add",
    "pow_to_square",
    "silu_expand",
    "assoc_matmul_rev",
    "assoc_linear",
    "weight_factor_linear",
    "right_factor_linear",
    "linear_channel_scale",
    "parallel_mul_fuse",
    "sdpa_fold_addmul",
    "affd_lift",
]

_RULE_INDEX: dict[str, Rewrite] = {
    r.name: r for r in ALL_RULES + SCAN_LAWS + SCAN_DIAG_LAWS
}

_GROUPS: dict[str, list[Rewrite]] = {
    "simpl": SIMPLIFICATION_RULES,
    "categorical": CATEGORICAL_RULES,
    "sdpa_fold": SDPA_FOLD_RULES,
    "scan": SCAN_LAWS,
    "scan_diag": SCAN_DIAG_LAWS,
    "all": ALL_RULES + SCAN_LAWS + SCAN_DIAG_LAWS,
    "default": [_RULE_INDEX[n] for n in DEFAULT_LAWS],
}


def _select_laws(spec: str) -> tuple[list[Rewrite], list[str]]:
    """Resolve ``--laws`` to ``(rules, implicit)``.

    Implicit names (group expansion) skip case-less rules silently;
    explicit names keep their no-case row.
    """
    rules: list[Rewrite] = []
    explicit: list[str] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in _GROUPS:
            for r in _GROUPS[tok]:
                if r.name in LAW_CASES and r.name not in {
                    x.name for x in rules
                }:
                    rules.append(r)
            continue
        if tok not in _RULE_INDEX:
            raise SystemExit(
                f"unknown law {tok!r} — names: "
                f"{sorted(_RULE_INDEX)}; groups: {sorted(_GROUPS)}"
            )
        if _RULE_INDEX[tok] not in rules:
            rules.append(_RULE_INDEX[tok])
            explicit.append(tok)
    return rules, explicit


# ---------------------------------------------------------------------------
#  Per-law measurement
# ---------------------------------------------------------------------------


def _params_of(term: Any) -> dict[str, Param]:
    seen: set[Any] = set()
    out: dict[str, Param] = {}
    stack = [term]
    while stack:
        t = stack.pop()
        if t in seen:
            continue
        seen.add(t)
        if isinstance(t, Param):
            out[t.name] = t
        elif isinstance(t, Op):
            stack.extend(t.args)
    return out


def _ir_of(term: Any, inputs: list[Var]) -> IR:
    return IR(
        root=term,
        inputs=list(inputs),
        input_names={v.name for v in inputs},
        params=_params_of(term),
    )


def _stmt(mod: Any, feed: tuple):
    def fn() -> None:
        with torch.no_grad():
            mod(*feed)

    return fn


def _bench_one(
    rule: Rewrite,
    d: int,
    dev: torch.device,
    sink: TorchSink,
    cost_fn: Any,
    iters: int,
    max_nodes: int,
    rtol: float,
) -> Case:
    """Run one (law, size) cell; returns the benchkit ``Case``.

    ``aux`` carries the honest status — fired?  RHS in root class?
    picked by extraction?  verified? — even for the rows that never
    reach the timing stage.
    """
    aux: dict[str, Any] = {
        "fired": 0,
        "rhs_member": "-",
        "picked": "-",
        "verified": "-",
    }
    params = {"law": rule.name, "size": d}
    build = LAW_CASES.get(rule.name)
    if build is None:
        aux["note"] = "no synthetic case registered"
        return Case(rule.name, params, [], aux)
    try:
        torch.manual_seed(_SEED + d)
        term, env, inputs = build(d, dev)
        feed = tuple(env[v.name] for v in inputs)
        param_vals = {n: env[n] for n in _params_of(term) if n in env}
        eg = EGraph()
        root = eg.add_term(term)
        eg.run([rule], root, max_iterations=iters, max_nodes=max_nodes)
        fires = eg.rule_fires.get(rule.name, 0)
        aux["fired"] = fires
        cost_in = dag_cost(term, cost_fn)
        aux["cost_in"] = round(cost_in, 4)
        if not fires:
            aux["note"] = "law did not fire on the synthetic term"
            return Case(rule.name, params, [], aux)
        if isinstance(rule.rhs, Op):
            aux["rhs_member"] = (
                "yes" if list(eg.matches(rule.rhs, eg.find(root))) else "no"
            )
        else:
            aux["rhs_member"] = "yes"  # bare metavar: the merge IS it
        best = eg.extract_best(root, cost_fn)
        if best is None:
            aux["note"] = "extraction returned no member"
            return Case(rule.name, params, [], aux)
        cost_out = dag_cost(best, cost_fn)
        aux["cost_out"] = round(cost_out, 4)
        picked = best != term
        aux["picked"] = "yes" if picked else "no"
        if not picked:
            aux["verified"] = "same"
            aux["note"] = (
                "member found; extraction kept the input term "
                "(RHS not cheaper under the pipeline cost model)"
            )
            return Case(rule.name, params, [], aux)
        aux["before"] = op_repr(term)
        aux["after"] = op_repr(best)
        before_mod = _lower_extracted(
            term, _ir_of(term, inputs), param_vals, sink
        )
        after_mod = _lower_extracted(
            best, _ir_of(best, inputs), param_vals, sink
        )
        vr = sink.verify(before_mod, after_mod, feed, rtol=rtol)
        aux["max_rel"] = f"{vr.max_rel:.3e}"
        if not vr.passed:
            aux["verified"] = "FAIL"
            aux["note"] = "lowered terms differ — not timed"
            return Case(rule.name, params, [], aux)
        aux["verified"] = "pass"
        variants = [
            Variant("before", _stmt(before_mod, feed)),
            Variant("after", _stmt(after_mod, feed)),
        ]
        return Case(rule.name, params, variants, aux)
    except Exception as e:  # honest failure row, not a crash
        aux["note"] = f"error: {type(e).__name__}: {e}"
        return Case(rule.name, params, [], aux)


def _print_table(cells: list[Any]) -> None:
    """The per-law summary table — the artifact's console twin."""
    head = (
        f"{'law':<26} {'size':>5} {'fired':>5} {'rhs':>4} "
        f"{'picked':>6} {'verify':>6} {'cost in':>10} "
        f"{'cost out':>10} {'ms in':>9} {'ms out':>9} {'x':>6}"
    )
    print("\n" + head)
    print("-" * len(head))
    for cell in cells:
        p, aux = cell.case.params, cell.aux
        ci = aux.get("cost_in")
        co = aux.get("cost_out")
        mi = cell.medians.get("before")
        mo = cell.medians.get("after")
        spd = f"{mi / mo:.3f}x" if mi and mo else "-"
        print(
            f"{p['law']:<26} {p['size']:>5} {aux['fired']:>5} "
            f"{aux['rhs_member']:>4} {aux['picked']:>6} "
            f"{aux['verified']:>6} "
            f"{f'{ci:.4g}' if ci is not None else '-':>10} "
            f"{f'{co:.4g}' if co is not None else '-':>10} "
            f"{f'{mi * 1e3:.4g}' if mi else '-':>9} "
            f"{f'{mo * 1e3:.4g}' if mo else '-':>9} {spd:>6}"
        )
        note = aux.get("note")
        if note:
            print(f"{'':>26} ↳ {note}")


def run_bench(args: argparse.Namespace) -> Report:
    """Harnessed entry point (``run_all.py`` convention)."""
    dev = torch.device(
        getattr(args, "device", None)
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    sizes = [
        int(s)
        for s in str(getattr(args, "sizes", None) or "256").split(",")
        if s.strip()
    ]
    rules, _ = _select_laws(getattr(args, "laws", None) or "default")
    sink = TorchSink()
    # The pipeline's selection model: roofline + generic-executor
    # dispatch overhead, priced backend-relative so extraction never
    # picks a form the sink cannot lower.
    cost_fn = backend_cost(
        executor_cost_for(lowering="generic"), sink.supported_ops
    )
    cases = [
        _bench_one(
            rule,
            d,
            dev,
            sink,
            cost_fn,
            iters=int(getattr(args, "iters", None) or 25),
            max_nodes=int(getattr(args, "max_nodes", None) or 50_000),
            rtol=float(getattr(args, "rtol", None) or 1e-4),
        )
        for rule in rules
        for d in sizes
    ]
    runner = Runner(
        device=dev,
        warmup=int(getattr(args, "warmup", None) or 5),
        min_run_time=float(getattr(args, "min_run_time", None) or 0.1),
    )
    print(f"[law_bench] {len(cases)} cells on {dev} …")
    cells = runner.run(cases)
    _print_table(cells)
    return Report(suite="law_bench", cells=cells, env=collect_env(dev))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--laws", default="default")
    ap.add_argument("--sizes", default="256")
    ap.add_argument("--device", default=None)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--min-run-time", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--max-nodes", type=int, default=50_000)
    ap.add_argument("--rtol", type=float, default=1e-4)
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--no-artifacts", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    if args.list:
        print(f"{'law':<28} {'group':<14} case?")
        for name, r in sorted(_RULE_INDEX.items()):
            grp = (
                "simpl"
                if r in SIMPLIFICATION_RULES
                else "scan*"
                if r in SCAN_LAWS + SCAN_DIAG_LAWS
                else "categorical"
            )
            print(
                f"{name:<28} {grp:<14} "
                f"{'yes' if name in LAW_CASES else '-'}"
            )
        return

    report = run_bench(args)
    if not args.no_artifacts:
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out = Path(args.out)
        report.to_json(out / f"law_bench_{ts}.json")
        report.to_markdown(
            out / f"law_bench_{ts}.md", speedup_vs="before"
        )
        print(f"[artifacts] {out}/law_bench_{ts}.{{json,md}}")


if __name__ == "__main__":
    main()
