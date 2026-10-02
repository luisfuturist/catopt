"""The search game (:mod:`catopt_core.game`)."""

import pytest
from catopt_core.egraph import EGraph
from catopt_core.game import (
    Action,
    Evaluator,
    GameState,
    RuleBook,
    transition,
)
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _state():
    eg = EGraph()
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    term = Op.make("matmul", Op.make("matmul", a, b), c)
    return eg, GameState(eg, eg.add_term(term))


class _FakeEG:
    def extract_best(self, eid, cost_fn):
        return None


def test_rule_book_actions_and_lookup():
    rules = all_rules()
    book = RuleBook(rules)
    _, st = _state()
    acts = book.actions(st)
    assert len(acts) == len(rules)
    assert all(a.eclass == st.root_eid for a in acts)
    assert book.rule(rules[0].name) is rules[0]
    assert repr(acts[0]) == f"{rules[0].name}@{st.root_eid}"


def test_rule_book_missing_rule_raises():
    with pytest.raises(KeyError):
        RuleBook(()).rule("nope")


def test_action_defaults_and_transition_and_evaluator():
    book = RuleBook(all_rules())
    eg, st = _state()
    ev = Evaluator()
    before = ev.evaluate(st)
    assert before >= 0.0
    assert Action("r").eclass == 0
    assert any(transition(eg, a, book) for a in book.actions(st))
    assert ev.evaluate(st) <= before


def test_evaluator_none_term_is_infinite():
    assert Evaluator().evaluate(GameState(_FakeEG(), 0)) == float("inf")
