"""Search policies (:mod:`catopt_core.policies`)."""

from catopt_core.game import Action
from catopt_core.policies import (
    BeamPolicy,
    ExistingPolicy,
    GreedyPolicy,
    RandomPolicy,
)
from catopt_core.ports import Policy


def _acts() -> list[Action]:
    return [Action("a", 0), Action("b", 0), Action("c", 0)]


def test_all_policies_conform():
    for p in (
        RandomPolicy(),
        ExistingPolicy(),
        GreedyPolicy(),
        BeamPolicy(),
    ):
        assert isinstance(p, Policy)


def test_random_policy_is_seeded_and_in_range():
    p = RandomPolicy(seed=0)
    picks = {p.choose(None, _acts()).rule for _ in range(50)}
    assert picks <= {"a", "b", "c"}
    assert p.name == "random"


def test_existing_policy_keeps_first():
    assert ExistingPolicy().choose(None, _acts()).rule == "a"


def test_greedy_policy_prefers_lowest_priority_number():
    assert (
        GreedyPolicy({"a": 2.0, "b": 1.0, "c": 3.0})
        .choose(None, _acts())
        .rule
        == "b"
    )
    # unmapped rules default to 0.0, so they win; ties keep order
    assert GreedyPolicy({}).choose(None, _acts()).rule == "a"
    assert GreedyPolicy().choose(None, _acts()).rule == "a"


def test_beam_policy_default_is_declaration_order():
    assert BeamPolicy().choose(None, _acts()).rule == "a"
    assert BeamPolicy().score(None, Action("a", 0), 0) == 0.0


def test_beam_policy_uses_scorer():
    table = {"a": 0.0, "b": 5.0, "c": 1.0}
    p = BeamPolicy(scorer=lambda s, a: table[a.rule])
    assert p.choose(None, _acts()).rule == "b"
