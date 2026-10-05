"""Engine seam + native-vs-Python differential oracle (plan 0010, lever 3).

Two halves:

* **Engine seam** (always runs): the ``engine=`` plumbing —
  ``search``/``Optimizer`` take an explicit engine conforming to
  :class:`catopt_core.ports.Engine`, ``stats["engine"]`` records which
  ran, and non-local (proof-carrying) passes are skipped when the
  engine is not an ``EGraph``.  Exercised with a pure-Python engine
  double so the seam is tested even when the native wheel is absent.
* **Differential oracle** (self-skips when ``catopt_native`` is not
  built): native vs Python engine on a corpus of terms — identical
  e-class counts, identical live enodes, identical extracted terms and
  costs.  The term corpus mirrors ``tests/egglog_oracle.py`` (the
  SPIKE rule subset plus check/derive-heavy folds).

Build the wheel with ``maturin develop --release`` from
``packages/catopt-native`` (or ``pip install packages/catopt-native``).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from catopt_core import laws
from catopt_core.cost import dag_cost, flops_cost
from catopt_core.egraph import EGraph, Rewrite
from catopt_core.ir import (
    IR,
    Const,
    Op,
    Param,
    TensorType,
    Var,
    op_repr,
)
from catopt_core.laws.tensor import (
    ASSOC_ADD,
    ASSOC_LINEAR,
    ASSOC_LINEAR_BIAS,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    ASSOC_MUL,
    COMM_ADD,
    COMM_MUL,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
    NATURALITY_SCALAR,
    NATURALITY_SCALAR_REV,
    SDPA_FOLD_RULES,
)
from catopt_core.ports import Engine
from catopt_orchestrator.optimize import (
    Optimizer,
    _resolve_engine,
    search,
)

try:
    from catopt_native import NativeEngine

    _HAS_NATIVE = True
except ModuleNotFoundError:  # pragma: no cover — wheel absent
    NativeEngine = None
    _HAS_NATIVE = False

requires_native = pytest.mark.skipif(
    not _HAS_NATIVE,
    reason="catopt_native not built (maturin develop in packages/catopt-native)",
)

#: The egglog-oracle rule subset — comm/assoc + matmul/linear algebra.
SPIKE_RULES = [
    COMM_ADD,
    ASSOC_ADD,
    COMM_MUL,
    ASSOC_MUL,
    ASSOC_MATMUL,
    ASSOC_MATMUL_REV,
    DISTRIBUTE_MUL,
    FACTOR_MUL,
    ASSOC_LINEAR,
    ASSOC_LINEAR_BIAS,
]


def _live_enodes(eng) -> int:
    """Live (class-member) enodes — stale hash-cons keys don't count."""
    return sum(len(ec.nodes) for ec in eng._classes.values())


def _matmul_chain() -> Any:
    a = Param("A", TensorType((4, 64)))
    b = Param("B", TensorType((64, 2)))
    x = Var("x", TensorType((2, 4)))
    return Op.make("matmul", a, Op.make("matmul", b, x))


def _linear_chain() -> Any:
    x = Var("x", TensorType((2, 8)))
    w1 = Param("w1", TensorType((4, 8)))
    w2 = Param("w2", TensorType((6, 4)))
    b1 = Param("b1", TensorType((4,)))
    b2 = Param("b2", TensorType((6,)))
    return Op.make("linear", Op.make("linear", x, w1, b1), w2, b2)


def _sdpa_term() -> Any:
    """``softmax(qk^T + 0)v`` — the SDPA fold (check + derive + $attr:)."""
    x = Var("x", TensorType((2, 8, 16)))
    wq = Param("wq", TensorType((16, 16)))
    wk = Param("wk", TensorType((16, 16)))
    wv = Param("wv", TensorType((16, 16)))
    q = Op.make("linear", x, wq)
    k = Op.make("linear", x, wk)
    v = Op.make("linear", x, wv)
    scores = Op.make(
        "matmul", q, Op.make("transpose", k, dim0=-2, dim1=-1)
    )
    sm = Op.make("softmax", Op.make("add", scores, Const(0)), dim=-1)
    return Op.make("matmul", sm, v)


def _residual_mlp() -> Any:
    x = Var("x", TensorType((4, 64)))
    w1 = Param("w1", TensorType((128, 64)))
    w2 = Param("w2", TensorType((64, 128)))
    c = Param("c", TensorType((64,)))
    inner = Op.make(
        "linear",
        Op.make("mul", Op.make("gelu", Op.make("linear", x, w1)), c),
        w2,
    )
    return Op.make("add", x, inner)


# ---------------------------------------------------------------------------
#  The pure-Python engine double — exercises the seam without the wheel
# ---------------------------------------------------------------------------


class _DoubleEngine:
    """A conforming :class:`Engine` delegating to a private ``EGraph``.

    Deliberately lacks ``_classes`` (the port does not require it), so
    ``_carrier_upgrade`` exercises its no-view early return.
    """

    engine_name = "double"

    def __init__(self) -> None:
        self._eg = EGraph()

    def add_term(
        self,
        term: Any,
        _memo: dict | None = None,
        provenance: str = "input",
    ) -> int:
        return self._eg.add_term(term)

    def find(self, eid: int) -> int:
        return self._eg.find(eid)

    def rebuild(self, classes: Any = None) -> bool:
        return self._eg.rebuild(classes)

    def run(self, rules: Any, root_eid: int, **kw: Any) -> dict:
        return self._eg.run(rules, root_eid, **kw)

    def extract_best(self, eid: int, cost_fn: Any, **kw: Any) -> Any:
        return self._eg.extract_best(eid, cost_fn, **kw)

    @property
    def n_enodes(self) -> int:
        return self._eg.n_enodes

    @property
    def n_classes(self) -> int:
        return self._eg.n_classes

    @property
    def rule_fires(self) -> dict[str, int]:
        return dict(self._eg.rule_fires)


class _FakeSource:
    """Source port double: the model IS the ``(ir, params)`` pair."""

    def to_ir(self, model: Any, x: Any) -> tuple[IR, dict]:
        return model


def _toy_model() -> tuple[IR, dict]:
    """``linear(x, W) + linear(x, W2)`` — shared input, pairs under Python."""
    x = Var("x", TensorType((2, 8)))
    w1 = Param("w1", TensorType((4, 8)))
    w2 = Param("w2", TensorType((6, 8)))
    root = Op.make(
        "add", Op.make("linear", x, w1), Op.make("linear", x, w2)
    )
    return IR(root=root, inputs=[x], input_names={"x"}, params={}), {}


# ---------------------------------------------------------------------------
#  Engine seam — runs with or without the wheel
# ---------------------------------------------------------------------------


def test_engine_protocol_conformance() -> None:
    """``EGraph`` and the double satisfy ``Engine``; a bare object does not."""
    assert isinstance(EGraph(), Engine)
    assert isinstance(_DoubleEngine(), Engine)
    assert not isinstance(object(), Engine)
    if _HAS_NATIVE:
        assert isinstance(NativeEngine(), Engine)


def test_resolve_engine_forms() -> None:
    """``_resolve_engine``: None→EGraph, class→instance, factory→called."""
    assert isinstance(_resolve_engine(None), EGraph)
    assert isinstance(_resolve_engine(_DoubleEngine), _DoubleEngine)
    assert isinstance(
        _resolve_engine(lambda: _DoubleEngine()), _DoubleEngine
    )
    inst = _DoubleEngine()
    assert _resolve_engine(inst) is inst


def test_search_default_engine_is_python() -> None:
    """Without ``engine=`` the reference engine runs; non-local passes fire."""
    ir_model = _toy_model()
    res = search(
        ir_model,
        None,
        source=_FakeSource(),
        cost_fn=flops_cost,
        max_iterations=4,
    )
    assert res.stats["engine"] == "python"
    # The Python engine runs the non-local pairing pass — the two
    # shared-input linears pair into one fused GEMM.
    assert res.stats.get("pairing_groups", 0) >= 1
    assert res.term is not None


def test_search_engine_double_skips_nonlocal() -> None:
    """A non-EGraph engine records itself and skips proof-carrying passes."""
    res = search(
        _toy_model(),
        None,
        source=_FakeSource(),
        cost_fn=flops_cost,
        max_iterations=4,
        engine=_DoubleEngine(),
    )
    assert res.stats["engine"] == "double"
    assert res.stats["nonlocal_passes"] == (
        "skipped (engine is not an EGraph)"
    )
    assert "pairing_groups" not in res.stats
    assert res.term is not None
    # Extracted under the same rules/cost — the double searches the
    # same space as Python (pairing aside).
    assert dag_cost(res.term, flops_cost) < float("inf")


class _FakeSink:
    """Minimal ``Capabilities`` for the optimizer seam test."""

    supported_ops = frozenset({"add", "linear", "matmul"})
    executors: ClassVar[dict] = {}

    @property
    def ops(self) -> Any:
        from catopt_core.ops import OpTable

        return OpTable.core()

    def lower(self, ir: Any, params: Any = None) -> Any:
        return object()

    def verify(self, ref: Any, opt: Any, inputs: Any, **kw: Any) -> Any:
        return None


def test_optimizer_engine_field() -> None:
    """``Optimizer(engine=...)`` wires through ``.search``."""
    opt = Optimizer(
        source=_FakeSource(),
        sink=_FakeSink(),
        engine=_DoubleEngine,
    )
    res = opt.search(_toy_model(), None, cost_fn=flops_cost)
    assert res.stats["engine"] == "double"


# ---------------------------------------------------------------------------
#  Differential oracle — native vs Python
# ---------------------------------------------------------------------------


def _run_both(
    term: Any,
    rules: list,
    budgets: dict | None = None,
    max_iterations: int = 40,
) -> tuple:
    """Run both engines; return ``(py_stats, py_term, nat_stats, nat_term)``."""
    py = EGraph()
    proot = py.add_term(term)
    py_stats = py.run(
        rules,
        proot,
        max_iterations=max_iterations,
        max_nodes=500_000,
        rule_budgets=budgets,
    )
    py_term = py.extract_best(proot, flops_cost)

    nat = NativeEngine()
    nroot = nat.add_term(term)
    nat_stats = nat.run(
        rules,
        nroot,
        max_iterations=max_iterations,
        max_nodes=500_000,
        rule_budgets=budgets,
    )
    nat_term = nat.extract_best(nroot, flops_cost)
    return (py, py_stats, py_term, nat, nat_stats, nat_term)


@requires_native
@pytest.mark.parametrize(
    "term",
    [
        pytest.param(_matmul_chain(), id="matmul_chain"),
        pytest.param(_linear_chain(), id="linear_chain"),
        pytest.param(_sdpa_term(), id="sdpa_fold"),
        pytest.param(_residual_mlp(), id="residual_mlp"),
    ],
)
def test_oracle_identical_fixed_point(term: Any) -> None:
    """Unbudgeted saturation: identical classes, live enodes, term, cost."""
    py, ps, pt, nat, ns, nt = _run_both(term, SPIKE_RULES)
    assert pt is not None and nt is not None
    # The fixed point is unique: identical live structure.
    assert ns["n_classes"] == ps["n_classes"]
    assert _live_enodes(nat) == _live_enodes(py)
    assert ns["iterations"] == ps["iterations"]
    # Extraction is byte-identical (the native engine runs the
    # reference extractor over an identical graph).
    assert op_repr(nt) == op_repr(pt)
    assert dag_cost(nt, flops_cost) == pytest.approx(
        dag_cost(pt, flops_cost)
    )
    # Same rules fired (counts are enumeration-order-dependent).
    assert set(nat.rule_fires) == set(py.rule_fires)


@requires_native
def test_oracle_matmul_chain_closure() -> None:
    """The Catalan closure: assoc rules on a k-matmul chain."""
    ws = [Param(f"W{i}", TensorType((8, 8))) for i in range(8)]
    t = ws[0]
    for w in ws[1:]:
        t = Op.make("matmul", t, w)
    rules = [ASSOC_MATMUL, ASSOC_MATMUL_REV]
    py, ps, pt, nat, ns, nt = _run_both(t, rules)
    # One class per contiguous sub-product of >= 2 factors
    # (7+6+...+1 = 28 intervals) plus the k leaf classes.
    assert ns["n_classes"] == ps["n_classes"] == 36
    assert _live_enodes(nat) == _live_enodes(py)
    assert op_repr(nt) == op_repr(pt)


@requires_native
def test_oracle_budgeted_run_same_term() -> None:
    """A generous budget does not bind: identical stats + extraction."""
    term = _matmul_chain()
    budgets = {"assoc_add": 64, "assoc_matmul": 100_000}
    _py, ps, pt, _nat, ns, nt = _run_both(
        term, SPIKE_RULES, budgets=budgets
    )
    assert (
        ns["rule_budgets"]["assoc_matmul"]
        == (ps["rule_budgets"]["assoc_matmul"])
    )
    assert ns["stop"] == ps["stop"] == "fixed_point"
    assert op_repr(nt) == op_repr(pt)
    assert dag_cost(nt, flops_cost) == pytest.approx(
        dag_cost(pt, flops_cost)
    )


@requires_native
def test_budget_plumbing() -> None:
    """A binding budget truncates the expansive rule and reports so."""
    x = Var("x", TensorType((4, 64)))
    ps = [Param(f"p{i}", TensorType((4, 64))) for i in range(5)]
    t = x
    for p in ps:
        t = Op.make("add", t, Op.make("mul", x, p))
    nat = NativeEngine()
    root = nat.add_term(t)
    stats = nat.run(
        SPIKE_RULES,
        root,
        max_iterations=10,
        max_nodes=500_000,
        rule_budgets={"assoc_add": 32},
    )
    # The cap is enforced between candidate classes — an apply can
    # overshoot inside one class (same semantics as the Python engine),
    # so the rule is suspended once its spend crosses the budget.
    assert stats["rule_budgets"]["assoc_add"] >= 32
    assert "assoc_add" in stats["budget_suspended"]
    assert nat.extract_best(root, flops_cost) is not None


@requires_native
def test_oracle_check_derive_rules() -> None:
    """``check``/``derive`` + ``$attr:`` metavars run through Python hooks."""
    term = _sdpa_term()
    rules = [
        *SDPA_FOLD_RULES,
        NATURALITY_SCALAR,
        NATURALITY_SCALAR_REV,
    ]
    py, _ps, pt, nat, _ns, nt = _run_both(term, rules)
    # The sdpa fold fired in both — the fused kernel term extracted.
    assert "sdpa" in op_repr(nt)
    assert op_repr(nt) == op_repr(pt)
    assert set(nat.rule_fires) == set(py.rule_fires)


@requires_native
def test_oracle_full_default_rules() -> None:
    """The pipeline's default rule set on a residual MLP."""
    term = _residual_mlp()
    rules = list(laws.DEFAULT)
    py, ps, pt, nat, ns, nt = _run_both(term, rules, max_iterations=8)
    assert ns["n_classes"] == ps["n_classes"]
    assert _live_enodes(nat) == _live_enodes(py)
    assert op_repr(nt) == op_repr(pt)
    assert dag_cost(nt, flops_cost) == pytest.approx(
        dag_cost(pt, flops_cost)
    )


@requires_native
def test_native_stop_improving() -> None:
    """``stop='improving'`` drives the extract-cost callback per iteration."""
    nat = NativeEngine()
    term = _matmul_chain()
    root = nat.add_term(term)
    stats = nat.run(
        SPIKE_RULES,
        root,
        max_iterations=6,
        stop="improving",
        patience=2,
        cost_fn=flops_cost,
    )
    assert stats["stop"] in ("improving", "fixed_point")
    assert "improved" in stats
    assert nat.extract_best(root, flops_cost) is not None


@requires_native
def test_native_search_e2e() -> None:
    """``search(..., engine=NativeEngine)``: stats + extraction end-to-end."""
    res = search(
        _toy_model(),
        None,
        source=_FakeSource(),
        cost_fn=flops_cost,
        max_iterations=4,
        engine=NativeEngine,
    )
    assert res.stats["engine"] == "native"
    assert res.stats["nonlocal_passes"] == (
        "skipped (engine is not an EGraph)"
    )
    assert res.term is not None
    assert dag_cost(res.term, flops_cost) < float("inf")
    # The inspectable mid-state works: alternatives enumerates the
    # root-class frontier through the mixin's _classes view.
    alts = res.alternatives()
    assert alts and all(c < float("inf") for c, _ in alts)


@requires_native
def test_native_add_term_returns_same_shape() -> None:
    """``add_term`` interning matches Python: same eid for shared DAGs."""
    term = _residual_mlp()
    py = EGraph()
    nat = NativeEngine()
    pe = py.add_term(term)
    ne = nat.add_term(term)
    assert pe == ne == py.find(pe) == nat.find(ne)
    assert nat.n_enodes == py.n_enodes
    assert nat.n_classes == py.n_classes


@requires_native
def test_native_attr_spelling_strict_identity() -> None:
    """``min=0`` and ``min=0.0`` intern to distinct enodes, like Python.

    The Python side keys attr identity on ``repr`` (``ir.py``) —
    numeric-tower equality must not merge the spellings.  The Rust
    ``AttrVal`` mirrors it: variant-strict ``Eq``/``Hash``/``Ord``,
    lenient only at match sites (``loose_eq``).
    """
    x = Var("x", TensorType((4,)))
    t_int = Op.make("clamp", x, min=0, max=1)
    t_float = Op.make("clamp", x, min=0.0, max=1)
    nat = NativeEngine()
    c_int = nat.add_term(t_int)
    c_float = nat.add_term(t_float)
    assert c_int != c_float

    # matching stays lenient: a ``clamp(x, min=0, ...)`` pattern still
    # binds ``min=0.0`` spellings (Python ``_leaf_eq`` parity)
    pat = Op.make("clamp", "a", min=0, max=1)
    rule = Rewrite("clamp_self", pat, "a")
    nat.run([rule], c_float, max_iterations=4)
    assert nat.rule_fires.get("clamp_self", 0) >= 1


@requires_native
def test_native_vs_python_randomised_corpus() -> None:
    """Hypothesis corpus: random add/mul chains through the symmetry rules."""
    pytest.importorskip("hypothesis")
    from hypothesis import HealthCheck, given, settings

    from tests.test_property_strategies import additive_terms

    symmetry = [COMM_ADD, ASSOC_ADD, COMM_MUL, ASSOC_MUL]

    @given(additive_terms())
    @settings(
        max_examples=25,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def check(term: Any) -> None:
        py = EGraph()
        proot = py.add_term(term)
        pstats = py.run(
            symmetry, proot, max_iterations=8, max_nodes=100_000
        )
        nat = NativeEngine()
        nroot = nat.add_term(term)
        nstats = nat.run(
            symmetry, nroot, max_iterations=8, max_nodes=100_000
        )
        assert nstats["n_classes"] == pstats["n_classes"]
        assert _live_enodes(nat) == _live_enodes(py)
        pt = py.extract_best(proot, flops_cost)
        nt = nat.extract_best(nroot, flops_cost)
        assert op_repr(nt) == op_repr(pt)

    check()
