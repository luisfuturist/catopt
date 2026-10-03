"""The ``Policy`` hook in ``EGraph.run`` (ADR 0003 invariant 5).

A policy may only *reorder* the rules: every rule still runs, so the
fixed point — and therefore the certificate — is unchanged.
"""

from catopt_core.cost import flops_cost
from catopt_core.egraph import EGraph
from catopt_core.game import Action
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import all_rules


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain():
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    return Op.make("matmul", a, Op.make("matmul", b, c))


class ReversePolicy:
    """Take the last offered action each time — a full reversal."""

    name = "reverse"

    def __init__(self) -> None:
        self.calls = 0

    def choose(self, state, actions):
        self.calls += 1
        return list(actions)[-1]


class BogusPolicy:
    """Return something that was never offered."""

    name = "bogus"

    def choose(self, state, actions):
        return Action("no_such_rule", 0)


def _saturate(policy=None, iterations=50):
    eg = EGraph()
    root = eg.add_term(_chain())
    stats = eg.run(
        all_rules(), root, max_iterations=iterations, policy=policy
    )
    term = eg.extract_best(root, flops_cost)
    return eg, stats, term


def test_no_policy_reports_none():
    _, stats, _ = _saturate()
    assert stats["policy"] is None


def test_policy_name_is_recorded_and_it_is_consulted():
    policy = ReversePolicy()
    _, stats, _ = _saturate(policy)
    assert stats["policy"] == "reverse"
    # one full pass per iteration, at least
    assert policy.calls >= len(all_rules())


def test_reordering_reaches_the_same_fixed_point():
    _, plain, term_plain = _saturate()
    _, reordered, term_reordered = _saturate(ReversePolicy())
    assert plain["stop"] == reordered["stop"] == "fixed_point"
    assert plain["n_enodes"] == reordered["n_enodes"]
    assert plain["n_classes"] == reordered["n_classes"]
    assert flops_cost(term_plain) == flops_cost(term_reordered)


def test_an_unknown_action_never_drops_a_rule():
    _, plain, _ = _saturate(iterations=6)
    _, bogus, _ = _saturate(BogusPolicy(), iterations=6)
    assert plain["n_enodes"] == bogus["n_enodes"]
    assert plain["n_classes"] == bogus["n_classes"]
