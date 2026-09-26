"""Structural-sharing passes against the REAL stories15M layout.

Two directions:

* End-to-end: a synthetic GQA-style attention block built at
  stories15M's exact geometry — dim=288, n_heads=6, head_dim=48 —
  with kv-head replication materialised into the checkpoint (the
  pattern ``share_duplicate_param_slices`` was written for).  We run
  ``optimize_model`` under the storage cost axis and check
  ``param_report`` shows the elimination with fp64-exact output.

* Real checkpoint: /tmp/stories15M.bin — the honest finding from
  /tmp/struct_probe.py is that a fully-trained model contains ZERO
  bitwise duplicates at any granularity, so the passes must offer
  nothing (regression-pinned here), and the tied wcls is stored once.

Honest limit covered too: under the default flop/launch cost the
dedup member does NOT win extraction — param-only subtrees price at 0,
so the plain leaf ties and wins on size.  Storage savings require the
storage cost axis (``param_bytes_cost_for``), matching the documented
extraction contract in rules.share_duplicate_param_slices.
"""

import os

import pytest
import torch
from catopt.cost import flops_cost, param_bytes_cost_for
from catopt.optimize import optimize_model, param_report

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "/tmp/stories15M.bin"

# stories15M geometry — the real head layout this test mirrors.
DIM = 288
N_HEADS = 6
HEAD_DIM = DIM // N_HEADS  # 48


def _replicate_heads(
    unique_blocks: list[torch.Tensor], index_map: list[int]
) -> torch.Tensor:
    """Materialise a (heads*head_dim, dim) weight from unique
    (head_dim, dim) blocks under a per-head index map — the GQA-style
    'replication baked into the checkpoint' pattern."""
    return torch.cat([unique_blocks[i] for i in index_map], dim=0)


class Stories15MAttnShape(torch.nn.Module):
    """Q/K/V/O projections at stories15M's 288-dim, 6-head layout.

    Each projection consumes a DIFFERENT input slice so
    pair_shared_input_linears cannot fuse them into one GEMM — the
    test isolates the slice-sharing pass.
    """

    def __init__(self, kv_map_wk, kv_map_wv, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.wq = torch.nn.Linear(DIM, DIM, bias=False)
        self.wk = torch.nn.Linear(DIM, DIM, bias=False)
        self.wv = torch.nn.Linear(DIM, DIM, bias=False)
        self.wo = torch.nn.Linear(DIM, DIM, bias=False)
        with torch.no_grad():
            for lin in (self.wq, self.wo):
                lin.weight.copy_(
                    torch.randn(
                        DIM, DIM, generator=g, dtype=torch.float64
                    )
                )
            # KV projections: n_kv unique head-blocks repeated to fill
            # n_heads slots — bitwise-equal head slices.
            for lin, imap in (
                (self.wk, kv_map_wk),
                (self.wv, kv_map_wv),
            ):
                uniq = [
                    torch.randn(
                        HEAD_DIM, DIM, generator=g, dtype=torch.float64
                    )
                    for _ in range(max(imap) + 1)
                ]
                lin.weight.copy_(_replicate_heads(uniq, imap))

    def forward(self, x):
        q = self.wq(x[..., :DIM])
        k = self.wk(x[..., DIM : 2 * DIM])
        v = self.wv(x[..., 2 * DIM : 3 * DIM])
        return self.wo(q + k + v)


# ---------------------------------------------------------------------------
#  End-to-end: share_duplicate_param_slices on the real head layout
# ---------------------------------------------------------------------------


def test_gqa_head_dedup_stories15m_layout_end_to_end():
    """kv-head replication materialised in a 6-head, 288-dim checkpoint
    is removed by share_duplicate_param_slices under the storage cost
    axis — the state_dict stores only the unique head-blocks and the
    module is fp64-exact."""
    torch.manual_seed(0)
    # stories15M-style GQA: wk = 3 kv heads doubled, wv = 2 kv heads
    # tripled — index maps are the expand-pattern real GQA uses.
    m = (
        Stories15MAttnShape(
            kv_map_wk=[0, 0, 1, 1, 2, 2], kv_map_wv=[0, 0, 0, 1, 1, 1]
        )
        .eval()
        .double()
    )
    x = torch.randn(4, 3 * DIM, dtype=torch.float64)

    low, stats = optimize_model(
        m, x, cost_fn=param_bytes_cost_for(), verbose=False
    )

    # fp64-equivalent output (other exact rewrites may reassociate
    # fp ops — residual is fp64 noise, ~1e-13, NOT a sharing error).
    with torch.no_grad():
        ref = m(x)
        y = low(x)
    assert (y - ref).abs().max() < 1e-9

    # The weights file shrank by exactly the duplicated head blocks.
    r = param_report(m, low)
    wk_saved = (N_HEADS - 3) * HEAD_DIM * DIM * 8  # 3 unique of 6
    wv_saved = (N_HEADS - 2) * HEAD_DIM * DIM * 8  # 2 unique of 6
    assert r["bytes_saved"] == wk_saved + wv_saved
    assert (
        r["optimized_bytes"] == r["original_bytes"] - r["bytes_saved"]
    )

    # Original kv weights eliminated; dedup stacks derived.
    assert "p_wk_weight" in r["eliminated"]
    assert "p_wv_weight" in r["eliminated"]
    assert not any(n.startswith("p_wq_weight") for n in r["eliminated"])
    assert not any(n.startswith("p_wo_weight") for n in r["eliminated"])
    assert "p_wk_weight__heads6" in r["derived"]
    assert "p_wv_weight__heads6" in r["derived"]

    # The derived stacks hold the unique head-blocks in first-
    # occurrence order — the exact bytes a deduplicated checkpoint
    # would store.
    sd = low.state_dict()
    assert tuple(sd["p_wk_weight__heads6"].shape) == (3, HEAD_DIM, DIM)
    assert tuple(sd["p_wv_weight__heads6"].shape) == (2, HEAD_DIM, DIM)
    orig_wk = m.wk.weight.detach()
    assert torch.equal(sd["p_wk_weight__heads6"][0], orig_wk[:HEAD_DIM])
    assert torch.equal(
        sd["p_wk_weight__heads6"][1],
        orig_wk[2 * HEAD_DIM : 3 * HEAD_DIM],
    )
    assert torch.equal(
        sd["p_wk_weight__heads6"][2],
        orig_wk[4 * HEAD_DIM : 5 * HEAD_DIM],
    )

    # The share itself is BITWISE-exact: gathering the stack under the
    # index map reproduces the original weight byte-for-byte.
    recon = torch.index_select(
        sd["p_wk_weight__heads6"], 0, torch.tensor([0, 0, 1, 1, 2, 2])
    ).reshape(DIM, DIM)
    assert torch.equal(recon, orig_wk)
    orig_wv = m.wv.weight.detach()
    recon_v = torch.index_select(
        sd["p_wv_weight__heads6"], 0, torch.tensor([0, 0, 0, 1, 1, 1])
    ).reshape(DIM, DIM)
    assert torch.equal(recon_v, orig_wv)

    # The pass ran inside the pipeline (witnessed non-local lift).
    assert stats.get("nonlocal_lifts", 0) >= 2


def test_slice_dedup_all_heads_identical():
    """Degenerate GQA: every head block equal (index_map all zeros) —
    one stored head serves all six."""
    torch.manual_seed(1)
    m = (
        Stories15MAttnShape(
            kv_map_wk=[0] * N_HEADS,
            kv_map_wv=[0, 0, 0, 0, 0, 0],
            seed=7,
        )
        .eval()
        .double()
    )
    x = torch.randn(2, 3 * DIM, dtype=torch.float64)
    low, _ = optimize_model(
        m, x, cost_fn=param_bytes_cost_for(), verbose=False
    )
    with torch.no_grad():
        assert (low(x) - m(x)).abs().max() < 1e-9
    sd = low.state_dict()
    assert tuple(sd["p_wk_weight__heads6"].shape) == (1, HEAD_DIM, DIM)


def test_slice_dedup_no_fire_when_heads_distinct():
    """Soundness at the real layout: trained-style all-distinct heads
    (what the real stories15M actually contains) produce NO offer and
    NO storage change — the pass must not touch the module."""
    torch.manual_seed(2)
    m = (
        Stories15MAttnShape(
            kv_map_wk=list(range(N_HEADS)),
            kv_map_wv=list(range(N_HEADS)),
            seed=11,
        )
        .eval()
        .double()
    )
    x = torch.randn(2, 3 * DIM, dtype=torch.float64)
    low, _ = optimize_model(
        m, x, cost_fn=param_bytes_cost_for(), verbose=False
    )
    with torch.no_grad():
        assert (low(x) - m(x)).abs().max() < 1e-9
    r = param_report(m, low)
    assert r["bytes_saved"] == 0
    assert not any("__heads" in n for n in low.state_dict())


def test_slice_dedup_requires_storage_cost_axis():
    """Honest limit: under the default flop/launch cost the dedup
    member does NOT win — param-only subtrees price at 0, so the plain
    leaf ties and wins on size.  The storage win is only selectable
    under param_bytes_cost.  Both are exact; only the realised file
    size differs."""
    torch.manual_seed(3)
    m = (
        Stories15MAttnShape(
            kv_map_wk=[0, 0, 1, 1, 2, 2],
            kv_map_wv=[0, 0, 0, 1, 1, 1],
            seed=13,
        )
        .eval()
        .double()
    )
    x = torch.randn(2, 3 * DIM, dtype=torch.float64)
    low, _ = optimize_model(m, x, cost_fn=flops_cost, verbose=False)
    with torch.no_grad():
        assert (low(x) - m(x)).abs().max() < 1e-9  # still exact
    _r = param_report(m, low)
    # flop-axis extraction keeps the dense leaves: no storage win.
    assert not any("__heads" in n for n in low.state_dict())


# ---------------------------------------------------------------------------
#  Whole-tensor tying — the 'wcls stored twice' scenario
# ---------------------------------------------------------------------------


def test_tied_embedding_classifier_if_stored_twice():
    """stories15M ties wcls to token_embedding and stores it once.  If
    a checkpoint DID materialise the classifier as a second tensor,
    share_duplicate_params merges the two Param leaves bitwise and the
    optimised file stores one copy — the pass would have exposed the
    tying rather than needing it declared."""
    from catopt.optimize import param_report

    torch.manual_seed(4)
    vocab = 512

    class TiedLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, DIM)
            self.head = torch.nn.Linear(DIM, vocab, bias=False)
            # 'stored twice' — distinct Parameter objects, equal bytes.
            self.head.weight.data = self.embed.weight.data.clone()

        def forward(self, idx):
            return self.head(self.embed(idx))

    m = TiedLM().eval().double()
    idx = torch.randint(0, vocab, (8,))
    low, _ = optimize_model(m, idx, verbose=False)
    with torch.no_grad():
        assert (low(idx) - m(idx)).abs().max() < 1e-9
    r = param_report(m, low)
    # one of the two 512x288 tables is gone from the file.
    assert r["optimized_bytes"] == r["original_bytes"] // 2
    assert len(r["eliminated"]) == 1


# ---------------------------------------------------------------------------
#  The real checkpoint itself — honest zero
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(CKPT), reason="stories15M.bin not downloaded"
)
def test_real_stories15m_checkpoint_has_no_exact_duplicates():
    """Pins the probe's honest finding: stories15M contains ZERO
    bitwise-equal tensors and ZERO duplicated head/row-block slices —
    catopt's exact passes legitimately offer nothing on the real file.
    Also verifies wcls is stored once (tail is the rope cache)."""
    import sys

    import numpy as np

    sys.path.insert(0, REPO)
    from catopt.egraph import EGraph
    from catopt.ir import Param, TensorType
    from catopt_core.laws import (
        share_duplicate_param_slices,
        share_duplicate_params,
    )

    from bench.llama2c import load_llama2c

    w = load_llama2c(CKPT)

    # wcls tied by construction: tail after rms_final is seq*(hd/2)*2
    # = 256*24*2 = 12288 rope freqs, NOT a second vocab x dim table.
    assert w["_tail"].size == 256 * (HEAD_DIM // 2) * 2 == 12288
    assert w["_tail"].size != w["token_embedding"].size

    # Flatten layer stacks into per-layer tensors, as named params.
    src = {}
    for name, t in w.items():
        if t.ndim == 3:
            for i in range(t.shape[0]):
                src[f"{name}_{i}"] = torch.from_numpy(
                    np.array(t[i])
                ).clone()
        else:
            src[name] = torch.from_numpy(np.array(t)).clone()

    eg = EGraph()
    for n, t in src.items():
        eg.add_term(Param(n, TensorType(tuple(t.shape))))

    assert share_duplicate_params(eg, src) == []
    assert share_duplicate_param_slices(eg, src) == []
