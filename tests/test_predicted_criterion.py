"""The model-backed criterion: a PerformanceModel steering selection.

``PredictedCriterion`` is the seam that makes the EVALUATION dimension
operational — it turns a ``PerformanceModel`` (plus a ``Profiler`` for
the features) into a ``Criterion`` extraction can minimise.
"""

from catopt_core.cost import flops_cost
from catopt_core.ir import Op, TensorType, Var
from catopt_core.perf_model import AnalyticalPerformanceModel
from catopt_core.ports import Criterion, PerformanceModel
from catopt_orchestrator import PredictedCriterion
from catopt_orchestrator.criteria import criteria_cost


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain():
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    return Op.make("matmul", a, Op.make("matmul", b, c))


def test_model_conforms_and_the_criterion_does_too():
    assert isinstance(AnalyticalPerformanceModel(), PerformanceModel)
    assert isinstance(
        PredictedCriterion(AnalyticalPerformanceModel()), Criterion
    )


def test_it_prices_a_term_through_the_model():
    crit = PredictedCriterion(AnalyticalPerformanceModel())
    fn = crit.cost_fn()
    assert fn(_chain()) > 0.0
    # and it is not the FLOP axis: seconds, not FLOPs
    assert fn(_chain()) != flops_cost(_chain())


def test_the_prediction_depends_on_the_target():
    slow = PredictedCriterion(
        AnalyticalPerformanceModel(tflops=0.01, gbps=1.0, launch_us=5.0)
    )
    fast = PredictedCriterion(
        AnalyticalPerformanceModel(
            tflops=100.0, gbps=3000.0, launch_us=1.0
        )
    )
    assert slow.cost_fn()(_chain()) > fast.cost_fn()(_chain())


def test_hardware_override_reaches_the_model():
    class _HW:
        tflops = 1000.0
        gbps = 10000.0
        launch_us = 0.0

    crit = PredictedCriterion(
        AnalyticalPerformanceModel(), hardware=_HW()
    )
    assert crit.cost_fn()(_chain()) > 0.0


def test_criteria_cost_accepts_it():
    fn = criteria_cost(PredictedCriterion(AnalyticalPerformanceModel()))
    assert fn(_chain()) > 0.0
