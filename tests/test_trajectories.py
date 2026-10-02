"""Search trajectory data (:mod:`catopt_core.trajectories`)."""

from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules
from catopt_core.trajectories import (
    RULE_VECTOR_LEN,
    RuleSample,
    rule_samples,
    rule_vector,
)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain():
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    # the EXPENSIVE bracketing a·(b·c): associativity can improve it
    return Op.make("matmul", a, Op.make("matmul", b, c))


def test_rule_vector_is_structural_and_sized():
    v = rule_vector(all_rules()[0])
    assert len(v) == RULE_VECTOR_LEN
    assert sum(v[:8]) == 1.0
    assert sum(v[8:16]) == 1.0
    assert v[-2] in (0.0, 1.0)
    assert v[-1] in (0.0, 1.0)


def test_rule_vector_handles_missing_fields():
    class _Bare:
        lhs = None
        rhs = None

    v = rule_vector(_Bare())
    assert len(v) == RULE_VECTOR_LEN
    assert v[-1] == 0.0
    assert v[-2] == 0.0


def test_rule_samples_one_row_per_rule_with_an_improvement():
    rules = all_rules()
    samples = rule_samples(_chain(), rules)
    assert len(samples) == len(rules)
    assert all(isinstance(s, RuleSample) for s in samples)
    assert all(s.program for s in samples)
    assert max(s.delta_cost for s in samples) > 0.0
