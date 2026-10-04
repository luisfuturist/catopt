"""The bundled contraction-policy artifact — load, play, determinism.

``catopt_torch.contraction_policy`` ships the pretrained curriculum
weights under ``artifacts/`` plus the game machinery the player runs
on.  These tests check the artifact loads lazily, produces *valid*
contraction orders on small boards, is deterministic under seed, and —
the sanity quality bar — beats our cheapest-pair greedy on held-out
boards of the family it was trained on (``random_bond_network``).
"""

from __future__ import annotations

import math
import random
from collections import Counter

import catopt_torch
import pytest
import torch
from catopt_torch.contraction_policy import (
    PAIR_DIM,
    SAMPLE_TEMP,
    STATE_DIM,
    ContractionGame,
    ContractionPolicy,
    PairPolicyNet,
    batch_inputs,
    cost_of_order,
    greedy,
    load_contraction_policy,
    merge_tensors,
    pair_cost,
    pair_logits,
    random_bond_network,
    rollout_orders,
    run_policy_batch,
    save_contraction_policy,
)

_DEVICE = "cpu"


def _board(n: int = 8, seed: int = 0):
    """One small held-out bond-network board."""
    return random_bond_network(n, seed)


def _replay(tensors, sizes, order):
    """Replay an order; return ``(remaining_tensors, total_cost)``."""
    ts = [frozenset(t) for t in tensors]
    total = 0.0
    for a, b in order:
        total += pair_cost(ts[a], ts[b], sizes)
        ts = merge_tensors(ts, a, b)
    return ts, total


# ---------------------------------------------------------------------------
#  Boards and cost primitives
# ---------------------------------------------------------------------------


def test_pair_cost_is_union_product():
    sizes = {0: 2, 1: 3, 2: 5}
    assert (
        pair_cost(frozenset({0, 1}), frozenset({1, 2}), sizes) == 30.0
    )


def test_merge_tensors_xors_and_reorders():
    ts = [frozenset({0, 1}), frozenset({1, 2}), frozenset({3})]
    out = merge_tensors(ts, 0, 1)
    assert len(out) == 2
    assert out[-1] == frozenset({0, 2})  # xor drops the shared index
    assert out[0] == frozenset({3})


def test_greedy_deterministic_cost():
    tensors, sizes = _board(8, 3)
    assert greedy(tensors, sizes) == greedy(tensors, sizes)
    assert greedy(tensors, sizes) > 0


def test_greedy_randomised_seeded():
    tensors, sizes = _board(10, 1)
    a = greedy(tensors, sizes, rng=random.Random(7), top_k=3)
    b = greedy(tensors, sizes, rng=random.Random(7), top_k=3)
    assert a == b


def test_random_bond_network_is_einsum_valid():
    tensors, sizes = _board(12, 5)
    assert len(tensors) == 12
    counts = Counter(i for t in tensors for i in t)
    assert (
        max(counts.values()) <= 2
    )  # every index a bond or an open leg
    assert set(counts) <= set(sizes)
    assert all(2 <= s <= 8 for s in sizes.values())


def test_cost_of_order_replays():
    tensors, sizes = _board(8, 2)
    ts = [frozenset(t) for t in tensors]
    order: list[tuple[int, int]] = []
    total = 0.0
    while len(ts) > 1:
        c, a, b = min(
            (pair_cost(ts[x], ts[y], sizes), x, y)
            for x in range(len(ts))
            for y in range(x + 1, len(ts))
        )
        order.append((a, b))
        total += c
        ts = merge_tensors(ts, a, b)
    assert total == greedy(tensors, sizes)
    assert cost_of_order(tensors, sizes, order) == total


# ---------------------------------------------------------------------------
#  The game
# ---------------------------------------------------------------------------


def test_game_step_charges_and_shrinks():
    tensors, sizes = _board(6, 0)
    ref = greedy(tensors, sizes)
    g = ContractionGame(tensors, sizes, ref)
    assert not g.done
    assert len(g.pairs) == 15
    a, b = g.pairs[0]
    want = pair_cost(g.ts[a], g.ts[b], sizes)
    got = g.step(a, b)
    assert got == want == g.cost
    assert len(g.ts) == 5
    assert len(g.pairs) == 10


def test_game_full_episode_reaches_done():
    tensors, sizes = _board(5, 1)
    g = ContractionGame(tensors, sizes, greedy(tensors, sizes))
    while not g.done:
        g.step(*g.pairs[0])
    assert len(g.ts) == 1


def test_game_clone_is_independent():
    tensors, sizes = _board(6, 4)
    g = ContractionGame(tensors, sizes, greedy(tensors, sizes))
    c = g.clone()
    c.step(*c.pairs[0])
    assert len(g.ts) == 6 and g.cost == 0.0
    assert len(c.ts) == 5 and c.cost > 0


def test_state_and_pair_features():
    tensors, sizes = _board(7, 2)
    g = ContractionGame(tensors, sizes, greedy(tensors, sizes))
    sf = g.state_features()
    assert len(sf) == STATE_DIM
    assert sf[0] == 1.0  # all tensors remain
    pf = g.pair_feature_matrix()
    assert pf.shape == (len(g.pairs), PAIR_DIM)
    assert g.pair_feature_matrix() is pf  # cached
    assert g.all_pair_features() == pf.tolist()
    feats = g.all_pair_features()
    assert all(len(f) == PAIR_DIM for f in feats)


def test_pair_feature_matrix_mid_game():
    tensors, sizes = _board(6, 0)
    g = ContractionGame(tensors, sizes, greedy(tensors, sizes))
    g.step(*g.pairs[0])
    pf = g.pair_feature_matrix()
    assert pf.shape == (10, PAIR_DIM)
    assert math.isfinite(pf.sum())


# ---------------------------------------------------------------------------
#  The net and rollouts
# ---------------------------------------------------------------------------


def test_net_forward_shape():
    model = PairPolicyNet(8)
    x = torch.zeros(4, STATE_DIM + PAIR_DIM)
    out = model(x)
    assert out.shape == (4,)


def test_batch_inputs_and_pair_logits():
    tensors, sizes = _board(5, 0)
    ref = greedy(tensors, sizes)
    games = [ContractionGame(tensors, sizes, ref) for _ in range(3)]
    sf, pf = batch_inputs(games, _DEVICE)
    assert sf.shape == (3, STATE_DIM)
    assert pf.shape == (3, len(games[0].pairs), PAIR_DIM)
    logits = pair_logits(PairPolicyNet(8), sf, pf)
    assert logits.shape == (3, len(games[0].pairs))


def test_run_policy_batch_greedy_deterministic():
    tensors, sizes = _board(8, 1)
    model = PairPolicyNet(16)
    ref = greedy(tensors, sizes)
    a = run_policy_batch(
        model,
        tensors,
        sizes,
        ref,
        samples=1,
        greedy=True,
        temperature=1.0,
        device=_DEVICE,
    )
    b = run_policy_batch(
        model,
        tensors,
        sizes,
        ref,
        samples=1,
        greedy=True,
        temperature=1.0,
        device=_DEVICE,
    )
    assert a == b and len(a) == 1


def test_run_policy_batch_sampled_count():
    tensors, sizes = _board(6, 0)
    model = PairPolicyNet(16)
    torch.manual_seed(0)
    costs = run_policy_batch(
        model,
        tensors,
        sizes,
        greedy(tensors, sizes),
        samples=4,
        greedy=False,
        temperature=SAMPLE_TEMP,
        device=_DEVICE,
    )
    assert len(costs) == 4
    assert all(c > 0 for c in costs)


def test_rollout_orders_are_valid_and_priced():
    tensors, sizes = _board(7, 3)
    model = PairPolicyNet(16)
    orders, costs = rollout_orders(
        model,
        tensors,
        sizes,
        greedy(tensors, sizes),
        2,
        _DEVICE,
        SAMPLE_TEMP,
    )
    assert len(orders) == len(costs) == 2
    for order, cost in zip(orders, costs, strict=True):
        assert len(order) == len(tensors) - 1
        ts, total = _replay(tensors, sizes, order)
        assert len(ts) == 1
        assert total == cost


def test_rollout_orders_degenerate():
    """A solved board ends immediately; zero samples returns empty."""
    tensors, sizes = ((0,),), {0: 4}
    model = PairPolicyNet(16)
    orders, costs = rollout_orders(
        model, tensors, sizes, 1.0, 2, _DEVICE, 1.0, greedy=True
    )
    assert orders == [[], []]
    assert costs == [0.0, 0.0]
    orders, costs = rollout_orders(
        model, tensors, sizes, 1.0, 3, _DEVICE, 1.0
    )
    assert orders == [[], [], []]
    assert costs == [0.0, 0.0, 0.0]
    orders, costs = rollout_orders(
        model, tensors, sizes, 1.0, 0, _DEVICE, 1.0
    )
    assert orders == [] and costs == []


def test_rollout_orders_vec_greedy_and_priced():
    """The batched driver: greedy argmax is deterministic and priced."""
    tensors, sizes = _board(8, 3)
    model = PairPolicyNet(16)
    ref = greedy(tensors, sizes)
    a, ca = rollout_orders(
        model, tensors, sizes, ref, 4, _DEVICE, 1.0, greedy=True
    )
    b, cb = rollout_orders(
        model, tensors, sizes, ref, 4, _DEVICE, 1.0, greedy=True
    )
    assert a == b and ca == cb
    for order, cost in zip(a, ca, strict=True):
        ts, total = _replay(tensors, sizes, order)
        assert len(ts) == 1
        assert total == cost


def test_rollout_orders_greedy_flag():
    tensors, sizes = _board(6, 1)
    model = PairPolicyNet(16)
    a, _ = rollout_orders(
        model,
        tensors,
        sizes,
        greedy(tensors, sizes),
        1,
        _DEVICE,
        1.0,
        greedy=True,
    )
    b, _ = rollout_orders(
        model,
        tensors,
        sizes,
        greedy(tensors, sizes),
        1,
        _DEVICE,
        1.0,
        greedy=True,
    )
    assert a == b


# ---------------------------------------------------------------------------
#  The player wrapper
# ---------------------------------------------------------------------------


def _trained_ish_policy() -> ContractionPolicy:
    """A policy around random weights — for API mechanics, not quality."""
    torch.manual_seed(0)
    return ContractionPolicy(PairPolicyNet(16), device=_DEVICE)


def test_policy_order_is_valid_and_deterministic():
    policy = _trained_ish_policy()
    tensors, sizes = _board(8, 0)
    o1 = policy.order(tensors, sizes)
    o2 = policy.order(tensors, sizes)
    assert o1 == o2
    ts, total = _replay(tensors, sizes, o1)
    assert len(ts) == 1
    assert policy.cost(tensors, sizes) == total


def test_policy_sample_orders_seeded():
    policy = _trained_ish_policy()
    tensors, sizes = _board(7, 1)
    a = policy.sample_orders(tensors, sizes, samples=4, seed=3)
    b = policy.sample_orders(tensors, sizes, samples=4, seed=3)
    assert a == b
    assert len(a) == 4


def test_policy_best_order_picks_min():
    policy = _trained_ish_policy()
    tensors, sizes = _board(7, 2)
    order, cost = policy.best_order(tensors, sizes, samples=6, seed=0)
    runs = policy.sample_orders(tensors, sizes, samples=6, seed=0)
    assert cost == min(c for _o, c in runs)
    ts, _ = _replay(tensors, sizes, order)
    assert len(ts) == 1


def test_policy_meta_and_device():
    policy = ContractionPolicy(
        PairPolicyNet(8), device=_DEVICE, meta={"k": 1}
    )
    assert policy.device == _DEVICE
    assert policy.meta == {"k": 1}
    empty = ContractionPolicy(PairPolicyNet(8))
    assert empty.meta == {}


# ---------------------------------------------------------------------------
#  Save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = PairPolicyNet(16).eval()
    out = tmp_path / "p.pt"
    save_contraction_policy(
        model, out, meta={"trainer": "test", "train_scales": [8]}
    )
    policy = load_contraction_policy(out, device=_DEVICE)
    assert isinstance(policy, ContractionPolicy)
    assert policy.meta["hidden"] == 16
    assert policy.meta["trainer"] == "test"
    assert policy.meta["feature_contract"] == "scale-free-v1"
    # Same weights ⇒ same deterministic order.
    tensors, sizes = _board(6, 0)
    ref = greedy(tensors, sizes)
    want, _ = rollout_orders(
        model, tensors, sizes, ref, 1, _DEVICE, 1.0, greedy=True
    )
    assert policy.order(tensors, sizes) == want[0]


def test_load_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_contraction_policy(tmp_path / "nope.pt")


def test_load_rejects_wrong_format(tmp_path):
    out = tmp_path / "bad.pt"
    torch.save({"format": "something-else", "state_dict": {}}, out)
    with pytest.raises(ValueError, match="not a catopt"):
        load_contraction_policy(out)


def test_load_rejects_non_dict(tmp_path):
    out = tmp_path / "list.pt"
    torch.save([1, 2, 3], out)
    with pytest.raises(ValueError, match="not a catopt"):
        load_contraction_policy(out)


def test_load_rejects_feature_drift(tmp_path):
    out = tmp_path / "drift.pt"
    torch.save(
        {
            "format": "catopt-contraction-policy/1",
            "state_dict": {},
            "meta": {
                "hidden": 8,
                "state_dim": STATE_DIM,
                "pair_dim": PAIR_DIM + 1,  # drifted
                "feature_contract": "scale-free-v1",
            },
        },
        out,
    )
    with pytest.raises(ValueError, match="feature-spec mismatch"):
        load_contraction_policy(out)


def test_load_rejects_missing_hidden(tmp_path):
    out = tmp_path / "nohidden.pt"
    torch.save(
        {
            "format": "catopt-contraction-policy/1",
            "state_dict": {},
            "meta": {
                "state_dim": STATE_DIM,
                "pair_dim": PAIR_DIM,
                "feature_contract": "scale-free-v1",
            },
        },
        out,
    )
    with pytest.raises(ValueError, match="hidden"):
        load_contraction_policy(out)


def test_loader_is_lazy(monkeypatch):
    """Resolving the name must not ``torch.load`` anything."""
    import catopt_torch.contraction_policy as mod

    def _boom(*_a, **_k):
        raise AssertionError("torch.load called at import/resolve")

    monkeypatch.setattr(mod.torch, "load", _boom)
    assert catopt_torch.load_contraction_policy is not None
    assert catopt_torch.ContractionPolicy is not None


# ---------------------------------------------------------------------------
#  The bundled artifact itself
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bundled() -> ContractionPolicy:
    """The shipped default weights, loaded once per test module."""
    return load_contraction_policy()


def test_bundled_loads_with_meta(bundled):
    assert bundled.meta["trainer"] == "rl"
    assert bundled.meta["train_scales"] == [8, 12, 16, 20, 24]
    assert bundled.meta["family"] == "random_bond_network"
    assert bundled.meta["hidden"] == 64
    assert bundled.meta["git_sha"]


def test_bundled_plays_a_valid_order(bundled):
    tensors, sizes = _board(10, 42)
    order = bundled.order(tensors, sizes)
    ts, total = _replay(tensors, sizes, order)
    assert len(ts) == 1
    assert total > 0


def test_bundled_deterministic_under_seed(bundled):
    tensors, sizes = _board(10, 7)
    a = bundled.best_order(tensors, sizes, samples=8, seed=5)
    b = bundled.best_order(tensors, sizes, samples=8, seed=5)
    assert a == b


def test_bundled_beats_our_greedy(bundled):
    """Sanity quality bar: mean policy/greedy cost on held-out boards.

    The retro measured ratios of ~0.2-0.4 on bond boards near the
    training scales; 0.8 leaves generous headroom for run-to-run
    variation while still proving the artifact is a real player.
    """
    for n in (12, 16):
        ratios = []
        for k in range(5):
            tensors, sizes = random_bond_network(n, 4242 + 100 * n + k)
            ratios.append(
                bundled.cost(tensors, sizes) / greedy(tensors, sizes)
            )
        assert sum(ratios) / len(ratios) < 0.8, (n, ratios)


def test_bundled_via_lazy_surface():
    """``catopt_torch.load_contraction_policy`` resolves to the module."""
    assert (
        catopt_torch.load_contraction_policy is load_contraction_policy
    )
    assert catopt_torch.ContractionPolicy is ContractionPolicy
