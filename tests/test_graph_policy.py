"""The graph-state learned policy + real-model trajectories.

``GraphFeaturePolicy`` closes the wiring gap: ``EGraph.run`` hands the
policy a ``GameState`` with no features, so the base ``LearnedPolicy``
cannot run there.  These tests pin the derivation, the caching, the
error path, and — the load-bearing safety property — that the policy
only *reorders*, so the equivalence class and the certificate are the
shipped ordering's.
"""

import pytest
import torch
from catopt_core.cost import flops_cost
from catopt_core.egraph import EGraph
from catopt_core.features import compute_features
from catopt_core.game import Action, GameState
from catopt_core.ir import Op, TensorType, Var, op_repr
from catopt_core.laws import all_rules
from catopt_core.ports import Policy
from catopt_torch.adapters import TorchSource
from catopt_torch.graph_policy import GraphFeaturePolicy
from catopt_torch.learned_policy import LearnedPolicy, RuleValueNet
from catopt_torch.model_trajectories import model_rule_samples
from catopt_torch.models import LinearAttention, RMSNorm, SwiGLU


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain():
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    return Op.make("matmul", a, Op.make("matmul", b, c))


def _policy(device="cpu", cost_fn=None):
    rules = all_rules()
    net = RuleValueNet(hidden=8)
    by_name = {r.name: r for r in rules}
    return GraphFeaturePolicy(
        net, by_name, cost_fn=cost_fn, device=device
    )


def test_is_a_policy_and_named():
    p = _policy()
    assert isinstance(p, Policy)
    assert p.name == "learned-graph"
    assert p.cost_fn is flops_cost


def test_features_honour_state_features():
    p = _policy()
    f = compute_features(_chain())
    st = GameState(None, 0, features=f)
    assert p.features_of(st) is f


def test_features_derived_from_eg_and_cached():
    p = _policy()
    eg = EGraph()
    root = eg.add_term(_chain())
    st = GameState(eg, root)
    first = p.features_of(st)
    # Pinned contract: features of the *exported* program.
    assert first == compute_features(_chain())
    # Same e-graph: cached, same object, no recompute.
    assert p.features_of(st) is first
    assert p.features_of(GameState(eg, root)) is first
    # A different e-graph recomputes.
    other = EGraph()
    oroot = other.add_term(_v("z", 5, 6))
    assert p.features_of(GameState(other, oroot)) is not first


def test_features_without_eg_or_features_raises():
    p = _policy()
    with pytest.raises(ValueError):
        p.features_of(GameState(None, 0))


def test_choose_through_egraph_run():
    p = _policy()
    eg = EGraph()
    root = eg.add_term(_chain())
    rules = all_rules()
    stats = eg.run(rules, root, max_iterations=10, policy=p)
    assert stats["policy"] == "learned-graph"
    # Every rule was offered at least once (it reorders, never drops).
    assert stats["n_enodes"] > 0
    acts = [Action(r.name) for r in rules]
    assert p.choose(GameState(eg, root), acts).rule in {
        r.name for r in rules
    }


def test_learned_policy_base_features_of_reads_state():
    rules = all_rules()
    net = RuleValueNet(hidden=4)
    base = LearnedPolicy(
        net, {r.name: r for r in rules}, device="cpu"
    )
    f = compute_features(_chain())
    assert base.features_of(GameState(None, 0, features=f)) is f


def test_reordering_matches_the_shipped_equivalence_class():
    """The safety invariant: a policy cannot change what is certified."""
    rules = all_rules()
    plain = EGraph()
    proot = plain.add_term(_chain())
    pstats = plain.run(rules, proot, max_iterations=50)
    pterm = plain.extract_best(proot, flops_cost)

    pol = _policy()
    opt = EGraph()
    oroot = opt.add_term(_chain())
    ostats = opt.run(rules, oroot, max_iterations=50, policy=pol)
    oterm = opt.extract_best(oroot, flops_cost)

    assert pstats["stop"] == ostats["stop"] == "fixed_point"
    assert pstats["n_classes"] == ostats["n_classes"]
    assert pstats["n_enodes"] == ostats["n_enodes"]
    assert flops_cost(pterm) == flops_cost(oterm)


def test_custom_cost_fn_is_used():
    seen = {}

    def spy(term, memo=None):
        seen["called"] = True
        return flops_cost(term)

    p = _policy(cost_fn=spy)
    assert p.cost_fn is spy
    eg = EGraph()
    root = eg.add_term(_chain())
    p.features_of(GameState(eg, root))
    assert seen.get("called") is True


def test_model_rule_samples_from_real_model():
    torch.manual_seed(0)
    rules = all_rules()
    model, x = SwiGLU(8, 2), torch.randn(2, 8)
    samples = model_rule_samples(model, x, rules)
    assert len(samples) == len(rules)
    assert {s.rule for s in samples} == {r.name for r in rules}
    # The state is the exported program's features (the pinned contract).
    ir, _ = TorchSource().to_ir(model, x)
    assert samples[0].features == compute_features(ir.root)
    assert samples[0].program == op_repr(ir.root)


def test_model_rule_samples_custom_source():
    torch.manual_seed(1)
    rules = all_rules()
    src = TorchSource()
    direct = model_rule_samples(
        RMSNorm(16), torch.randn(2, 16), rules, source=src
    )
    assert len(direct) == len(rules)


def test_model_rule_samples_accepts_tuple_inputs():
    rules = all_rules()
    q = torch.randn(2, 8, 4)
    k = torch.randn(2, 8, 4)
    v = torch.randn(2, 8, 4)
    samples = model_rule_samples(LinearAttention(), (q, k, v), rules)
    assert len(samples) == len(rules)
