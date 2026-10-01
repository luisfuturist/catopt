"""Structured-model E2E bench — where does the machinery actually pay?

The structure census (``structure_census.py``) showed the stock dense
checkpoints carry ~zero exploitable slack.  This bench measures the
flip side: **structure-bearing** variants of the real llama2.c
stories15M checkpoint, synthesized the way such models actually ship:

* ``lora_r{r}`` — every block ``nn.Linear`` wrapped in a LoRA adapter
  (``x@W + x@AᵀBᵀ``, rank ``r``, ~5%‖W‖ delta norm).  The adapter is
  *spelled* — the ``assoc_linear`` / ``weight_factor_linear`` laws
  derive the merged ``linear(x, W + B·A)`` member and compile-time
  weight folding materialises it; ``detect_factors`` is armed but
  correctly offers nothing (spelled factors need no detection — its
  remit is *dense* weights that happen to be low-rank).  Baselines:
  eager/inductor on the unmerged model AND a hand-merged model (the
  deployment baseline a human would produce).
* ``pruned_f{frac}`` — ~``frac`` of each layer's FFN hidden channels
  structurally zeroed (``w1``/``w3`` rows, matching ``w2`` columns) —
  a pruned-checkpoint simulation.  ``detect_specials``'s exact
  ``elide`` members (dead input gather + deduped output gather around
  a shrunk weight) are bitwise-exact offers.  Baseline: a
  hand-**narrowed** model (dead channels deleted, the deployment
  equivalent).  A ``catopt_flops`` lane re-runs the same pipeline
  under pure-FLOP extraction (``cost_fn=flops_cost``) to separate
  "the structure is found" from "the executor-aware cost model
  prices the gathers over the saved GEMM work at this size".
* ``lowrank_r{r}_{f32,f64}`` — every Linear weight (blocks AND the
  LM head) replaced by a dense-stored ``B·A`` rank-``r`` product —
  the "built low-rank" checkpoint case ``detect_factors`` exists
  for.  The fp32 lane is an honest negative control: the pass's
  Gram-Schmidt certificate runs in the stored dtype, and
  ``rel_tol=1e-8`` sits BELOW the fp32 noise floor (a rank-32 fp32
  product measures ~3e-6 relative Frobenius residual), so detection
  declines.  The fp64 lane shows the same machinery when the dtype
  leaves the certificate room — detection fires (residual ~1e-11)
  and extraction prices the two skinny GEMMs against the dense one.

Per (variant, budget) cell — budgets ``exact`` then
``{1e-4, 1e-3, 1e-2}`` (``error_budget=`` arms the certified
bounded ``elide_bounded`` / ``zero_bounded`` members):

* timed ``eager`` / ``inductor`` / ``catopt`` (+ per-variant
  deployment baselines and the optional ``catopt+inductor`` /
  ``catopt_flops`` lanes);
* offered-vs-delivered structure ledgers (``weight_specials`` /
  ``low_rank_factors`` offers vs the derived params actually present
  in the delivered module);
* fresh-input ``rel_diff`` and held-out next-token KL / top-1 / top-5
  agreement (``bounded_e2e`` conventions — random + periodic prompts,
  no tokenizer ships with the checkpoint);
* parameter deltas (delivered vs in-model vs merged/narrowed).

The honest payoff table: which structure class pays how much, per
budget — and which offers the default cost model declines.

    .venv/bin/python bench/structured_models.py --device cpu
    .venv/bin/python bench/structured_models.py --quick
    .venv/bin/python bench/structured_models.py --variants lora_r8,pruned_f0.4
"""
# ruff: noqa: E402, RUF003 -- ×, ·, →, −, ‖ in
# strings/docstrings are deliberate math notation; sys.path setup
# must precede the benchkit/catopt imports (bounded_e2e convention).

from __future__ import annotations

import argparse
import copy
import gc
import sys
import time
from collections import Counter
from pathlib import Path

sys.setrecursionlimit(400_000)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import torch.nn as nn
import torch.nn.functional as F
from bench.benchkit import (
    Case,
    Cell,
    Report,
    Runner,
    Variant,
    collect_env,
)
from bench.suites.models.bounded_e2e import (
    _bounded_ledger,
    _fwd_stmt,
    _load_stories,
    _prompt_set,
    _proxy_eval,
)
from catopt_core.cost import flops_cost
from catopt_orchestrator import Compositional, Optimizer
from catopt_torch.backend import TorchBackend
from catopt_torch.report import verify_equiv
from bench.suites.algebra.real_win_hunt import try_compile
from bench.suites.models.stories15m_bench import Block

# run_all.py picks these up for its --quick lane.
QUICK = {
    "budgets": "none,1e-3",
    "lora_ranks": "8",
    "prune_fracs": "0.4",
    "lr_ranks": "32",
    "lr_dtypes": "f64",
    "seq": "64",
    "min_run_time": "0.05",
    "compile_timeout": "45.0",
    "prompts": "2",
    "catopt_inductor": "0",
}

_BUDGET_DEFAULT = "none,1e-4,1e-3,1e-2"
BLOCK_LINEAR_NAMES = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")
#: Trained-adapter magnitude: ‖B·A‖_F ≈ 5% of ‖W‖_F.
_LORA_DELTA_FRAC = 0.05


# ---------------------------------------------------------------------------
#  Variant builders — each returns a model sharing the checkpoint load
# ---------------------------------------------------------------------------


class LoraLinear(nn.Linear):
    """``nn.Linear`` + additive low-rank adapter.

    ``forward = W x + B·(A x)`` — the scaling (``alpha / r``) is
    pre-baked into ``lora_b`` so the exported graph is a plain
    ``add(linear, linear(linear))`` the ``weight_factor_linear`` /
    ``assoc_linear`` laws see directly (a ``mul`` on the delta would
    break the pattern).
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        gen: torch.Generator,
        delta_frac: float = _LORA_DELTA_FRAC,
    ) -> None:
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
        )
        # Share the base weight storage — the checkpoint copy lives on.
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        i, o = base.in_features, base.out_features
        dt = base.weight.dtype
        # fp64 synthesis keeps the delta honest (values then cast to
        # the model dtype — a real adapter file stores them as such).
        a = torch.randn(rank, i, generator=gen, dtype=torch.float64)
        b = torch.randn(o, rank, generator=gen, dtype=torch.float64)
        cur = torch.linalg.norm(b @ a).item()
        want = (
            delta_frac * torch.linalg.norm(base.weight.double()).item()
        )
        b *= want / max(cur, 1e-30)
        self.lora_a = nn.Parameter(a.to(dt))
        self.lora_b = nn.Parameter(b.to(dt))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias) + F.linear(
            F.linear(x, self.lora_a), self.lora_b
        )


def apply_lora(
    model: nn.Module,
    rank: int,
    gen: torch.Generator,
    names: tuple[str, ...] = BLOCK_LINEAR_NAMES,
) -> nn.Module:
    """Wrap every block Linear in ``names`` with a LoRA adapter."""
    for b in model.blocks:
        for nm in names:
            setattr(b, nm, LoraLinear(getattr(b, nm), rank, gen))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Hand-merged reference: fold ``B·A`` into each wrapped weight."""
    m = copy.deepcopy(model)
    for _mod_name, mod in m.named_modules():
        for child_name, child in list(mod.named_children()):
            if isinstance(child, LoraLinear):
                merged = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                with torch.no_grad():
                    merged.weight.copy_(
                        child.weight + child.lora_b @ child.lora_a
                    )
                    if child.bias is not None:
                        merged.bias.copy_(child.bias)
                setattr(mod, child_name, merged)
    return m.eval()


def prune_ffn(
    model: nn.Module, frac: float, gen: torch.Generator
) -> list[torch.Tensor]:
    """Zero ``frac`` of each layer's FFN hidden channels in place.

    A structured-pruning checkpoint: the same channel index is dead
    in ``w1`` (output row), ``w3`` (output row) and ``w2`` (input
    column).  Returns the KEEP index per layer (for the hand-narrowed
    baseline).
    """
    keeps = []
    for li, b in enumerate(model.blocks):
        hidden = b.w1.weight.shape[0]
        k = round(frac * hidden)
        lg = torch.Generator().manual_seed(
            int(gen.initial_seed()) + 1000 + li
        )
        dead = torch.randperm(hidden, generator=lg)[:k]
        with torch.no_grad():
            b.w1.weight[dead] = 0.0
            b.w3.weight[dead] = 0.0
            b.w2.weight[:, dead] = 0.0
        keep = torch.ones(hidden, dtype=torch.bool)
        keep[dead] = False
        keeps.append(keep.nonzero().flatten())
    return keeps


def build_narrowed(
    base: nn.Module, keeps: list[torch.Tensor]
) -> nn.Module:
    """Hand-narrowed deployment baseline: dead channels deleted.

    Same module tree as ``Stories15M`` but each block's FFN hidden
    width shrinks to the kept channels — what a real pruning stack
    would ship.
    """

    class _Narrowed(nn.Module):
        def forward(self, idx):
            T = idx.shape[-1]
            h = self.emb(idx)
            if h.dim() == 3:
                h = h[0]
            cos, sin = self.cos[:T], self.sin[:T]
            for blk in self.blocks:
                h = blk(h, cos, sin)
            h = F.rms_norm(h, (h.shape[-1],), self.rms_final, 1e-5)
            return self.head(h)

    b0 = base.blocks[0]
    dim = b0.wq.in_features
    m = _Narrowed()
    m.emb = base.emb
    m.blocks = nn.ModuleList()
    for b, keep in zip(base.blocks, keeps, strict=True):
        nb = Block(dim, int(keep.numel()), b.nh, b.hd)
        # Attention side unchanged — share the modules outright.
        nb.rms_att = b.rms_att
        nb.rms_ffn = b.rms_ffn
        nb.wq = b.wq
        nb.wk = b.wk
        nb.wv = b.wv
        nb.wo = b.wo
        nb.w1 = nn.Linear(dim, int(keep.numel()), bias=False)
        nb.w3 = nn.Linear(dim, int(keep.numel()), bias=False)
        nb.w2 = nn.Linear(int(keep.numel()), dim, bias=False)
        with torch.no_grad():
            nb.w1.weight.copy_(b.w1.weight[keep])
            nb.w3.weight.copy_(b.w3.weight[keep])
            nb.w2.weight.copy_(b.w2.weight[:, keep])
        m.blocks.append(nb)
    m.rms_final = base.rms_final
    m.head = base.head
    m.register_buffer("cos", base.cos)
    m.register_buffer("sin", base.sin)
    return m.eval()


def build_lowrank(
    model: nn.Module,
    rank: int,
    gen: torch.Generator,
    *,
    sites: tuple[str, ...] = BLOCK_LINEAR_NAMES,
    head: bool = True,
) -> nn.Module:
    """Replace Linear weights with dense-stored ``B·A`` products.

    The factorisation is synthesised in fp64, norm-matched to the
    original weight (a "built low-rank" checkpoint keeps its scale),
    then cast to the model's dtype — the fp32 lane thereby carries
    honest fp32 rounding, which is exactly what the detector's
    ``rel_tol`` gate is measured against.
    """

    def _set(lin: nn.Linear) -> None:
        o, i = lin.weight.shape
        a = torch.randn(rank, i, generator=gen, dtype=torch.float64)
        b = torch.randn(o, rank, generator=gen, dtype=torch.float64)
        w_new = b @ a
        w_new *= torch.linalg.norm(lin.weight.double()) / max(
            torch.linalg.norm(w_new).item(), 1e-30
        )
        with torch.no_grad():
            lin.weight.copy_(w_new.to(lin.weight.dtype))

    for b in model.blocks:
        for nm in sites:
            _set(getattr(b, nm))
    if head:
        _set(model.head)
    return model


# ---------------------------------------------------------------------------
#  Ledgers — what was OFFERED vs what the delivered module carries
# ---------------------------------------------------------------------------

#: Derived-name → structure-kind classifiers (``param_report.derived``
#: carries the delivered module's non-original parameters; the fused_*
#: names are ``_fold_weight_chains`` compile-time folds).
_DERIVED_KINDS = (
    ("__el", "elide"),
    ("__bl", "elide_bounded"),
    ("__lr", "low_rank"),
    ("__diag", "diagonal"),
    ("__heads", "slice_dedup"),
)


def _offer_ledger(stats: dict) -> dict:
    """Roll per-block offer/delivery evidence into one cell ledger."""
    kinds = Counter()
    lr = []
    offer_detail: list[dict] = []
    struct_per_block: dict[str, dict] = {}
    n_cache_hit = 0
    for name, rep in (stats.get("blocks") or {}).items():
        st = rep.get("stats") or {}
        # A cache-hit block's ``stats`` ARE the template's record —
        # counting its offer lists again would inflate the ledger.
        # The hit's own param_report is still real (it diffs the
        # replayed delivery against this block's own weights).
        hit = rep.get("cache") == "hit"
        n_cache_hit += int(hit)
        specials = [] if hit else (st.get("weight_specials") or [])
        factors = [] if hit else (st.get("low_rank_factors") or [])
        for s in specials:
            kinds[s.get("kind", "?")] += 1
            offer_detail.append(
                {
                    "block": name,
                    "param": s.get("param"),
                    "kind": s.get("kind"),
                    "error_bound": s.get("error_bound"),
                }
            )
        for f_ in factors:
            lr.append(
                {
                    "block": name,
                    "param": f_.get("param"),
                    "rank": f_.get("rank"),
                    "error": f_.get("error"),
                }
            )
        pr = rep.get("param_report") or {}
        struct_per_block[name] = {
            "status": rep.get("status"),
            "cache": rep.get("cache"),
            "rel_diff": rep.get("rel_diff"),
            "bytes": [
                pr.get("original_bytes"),
                pr.get("optimized_bytes"),
            ],
            "offers": len(specials),
            "factor_offers": len(factors),
        }
    struct_per_block["<cache_hits>"] = {"n": n_cache_hit}
    # Delivered structure: derived params in the delivered weights file.
    agg = stats.get("param_report") or {}
    delivered = Counter()
    for dname in agg.get("derived") or []:
        leaf = dname.split(":", 1)[-1]
        for tag, kind in _DERIVED_KINDS:
            if tag in leaf:
                delivered[kind] += 1
                break
        else:
            delivered["fused_fold"] += 1
    return {
        "struct_per_block": struct_per_block,
        "offers_by_kind": dict(kinds),
        "offer_detail": offer_detail,
        "low_rank_offers": lr,
        "delivered_derived": dict(delivered),
        "eliminated": agg.get("eliminated") or [],
        "derived": agg.get("derived") or [],
    }


def _factor_probe(w: torch.Tensor) -> dict:
    """What ``offer_low_rank_factors``'s Gram-Schmidt sees on ``w``.

    Runs the pass's own ``_row_basis`` + residual math at the shipped
    ``rel_tol=1e-8`` and at 1e-4 — the dtype-noise evidence for why a
    dense-stored fp32 factor product declines where fp64 fires.
    """
    from catopt_core.laws.factored import _factor_rows, _row_basis

    out: dict = {"o": int(w.shape[0]), "i": int(w.shape[1])}
    for tol in (1e-8, 1e-4):
        basis = _row_basis(w, tol)
        rec: dict = {"basis_rank": len(basis)}
        fac = _factor_rows(w, tol)
        if fac is not None:
            a, _b, err = fac
            fnorm = float(torch.linalg.norm(w).item())
            rec["rank"] = int(a.shape[0])
            rec["rel_residual"] = err / max(fnorm, 1e-30)
            rec["breakeven"] = bool(
                a.shape[0] * (w.shape[0] + w.shape[1])
                < w.shape[0] * w.shape[1]
            )
        out[f"tol_{tol:g}"] = rec
    return out


def _n_params(mod) -> int:
    """Delivered weight-file size — ``state_dict`` covers both
    registered Parameters and the derived tensors the lowerer
    materialises (``param_report`` convention)."""
    try:
        return sum(p.numel() for p in mod.state_dict().values())
    except Exception:
        return -1


# ---------------------------------------------------------------------------
#  One (variant, budget) cell
# ---------------------------------------------------------------------------


def run_cell(
    model,
    idx: torch.Tensor,
    budget: float | None,
    label: str,
    vname: str,
    args,
    dev: torch.device,
    baseline_mods: list,
    prompts: list[torch.Tensor],
    detect_factors: bool,
    flops_ok: bool,
) -> tuple[dict, Case]:
    """Optimize ``model`` under ``budget``, verify, pack the case.

    ``baseline_mods`` carries ``(name, module)`` pairs — the
    per-variant baselines built/compiled once (eager/inductor always;
    merged/narrowed deployment references where the variant has one).
    ``Variant`` stmts are rebuilt per cell: ``Runner`` retires them
    after timing.
    """
    print(f"\n=== {vname} budget={label} ===", flush=True)
    rec: dict = {
        "name": f"{vname}|{label}",
        "params": {
            "variant": vname,
            "budget": label,
            "seq": idx.shape[-1],
        },
        "detect_factors": detect_factors,
    }

    t0 = time.time()
    opt_mod = stats = None
    try:
        opt_mod, stats = Optimizer(backend=TorchBackend()).optimize(
            model,
            idx,
            strategy=Compositional(),
            detect_specials=True,
            detect_factors=detect_factors,
            error_budget=budget,
            verbose=False,
        )
        rec["opt_s"] = round(time.time() - t0, 1)
        rec["n_optimized"] = stats.get("n_optimized")
        rec["n_blocks"] = stats.get("n_blocks")
        rec["cache"] = stats.get("cache")
        rec["e2e"] = stats.get("end_to_end")
        rec.update(_bounded_ledger(stats))
        rec.update(_offer_ledger(stats))
        pr = stats.get("param_report") or {}
        rec["params_in"] = _n_params(model)
        rec["params_delivered"] = _n_params(opt_mod)
        rec["bytes_ratio"] = pr.get("ratio")
        print(
            f"  optimize: {rec['opt_s']}s "
            f"opt={rec['n_optimized']}/{rec['n_blocks']} "
            f"cache={rec['cache']} "
            f"offers={rec['offers_by_kind']} "
            f"lr={len(rec['low_rank_offers'])} "
            f"delivered={rec['delivered_derived']} "
            f"params {rec['params_in']}→{rec['params_delivered']}",
            flush=True,
        )
    except Exception as e:
        rec["opt_s"] = round(time.time() - t0, 1)
        rec["opt_error"] = f"{type(e).__name__}: {e}"
        print(f"  optimize FAILED: {rec['opt_error']}", flush=True)

    variants = [
        Variant(n, _fwd_stmt(bm, idx)) for n, bm in baseline_mods
    ]
    if opt_mod is not None:
        probe = prompts[0]
        with torch.no_grad():
            vr = verify_equiv(model(probe), opt_mod(probe))
        rec["fresh_rel"] = vr.max_rel
        rec["fresh_passed"] = vr.passed
        pe = _proxy_eval(model, opt_mod, prompts)
        rec["proxy"] = pe
        print(
            f"  fresh rel={vr.max_rel:.3e}  "
            f"KL mean={pe['mean_kl']:.3e} max={pe['max_kl']:.3e}  "
            f"top1={pe['top1_agree']:.3f} top5={pe['top5_in']:.3f}",
            flush=True,
        )
        variants.append(Variant("catopt", _fwd_stmt(opt_mod, idx)))
        if getattr(args, "catopt_inductor", None) not in (
            None,
            "0",
            0,
        ):
            ct = float(getattr(args, "compile_timeout", None) or 45.0)
            oc, st_ = try_compile(opt_mod, (idx,), ct)
            rec["catopt_ind_status"] = st_
            if oc is not None:
                with torch.no_grad():
                    vr2 = verify_equiv(model(idx), oc(idx))
                rec["catopt_ind_rel"] = vr2.max_rel
                if vr2.passed:
                    variants.append(
                        Variant("catopt+inductor", _fwd_stmt(oc, idx))
                    )
            print(
                f"  catopt+inductor: {rec['catopt_ind_status']}",
                flush=True,
            )

    # -- flop-priced lane ------------------------------------------------
    # Same pipeline, pure-FLOP extraction: isolates "the structure is
    # found" from "the executor-aware default prices the gathers over
    # the saved GEMM work".  Exact cells only — a bounded search under
    # a different cost model is a different run anyway, and the bound
    # ledger's story lives in the default lane.
    flops_mode = str(getattr(args, "flops_lane", None) or "exact")
    want_flops = (
        flops_mode != "none"
        and (flops_mode == "all" or budget is None)
        and flops_ok
    )
    if want_flops and opt_mod is not None:
        t0 = time.time()
        try:
            opt_f, stats_f = Optimizer(backend=TorchBackend()).optimize(
                model,
                idx,
                strategy=Compositional(),
                detect_specials=True,
                detect_factors=detect_factors,
                error_budget=budget,
                cost_fn=flops_cost,
                verbose=False,
            )
            rec["flops_opt_s"] = round(time.time() - t0, 1)
            rec["flops_ledger"] = _offer_ledger(stats_f)
            fp = stats_f.get("param_report") or {}
            rec["flops_params_delivered"] = _n_params(opt_f)
            rec["flops_bytes_ratio"] = fp.get("ratio")
            probe = prompts[0]
            with torch.no_grad():
                vrf = verify_equiv(model(probe), opt_f(probe))
            rec["flops_fresh_rel"] = vrf.max_rel
            rec["flops_e2e"] = stats_f.get("end_to_end")
            variants.append(
                Variant("catopt_flops", _fwd_stmt(opt_f, idx))
            )
            if getattr(args, "catopt_inductor", None) not in (
                None,
                "0",
                0,
            ):
                ct = float(
                    getattr(args, "compile_timeout", None) or 45.0
                )
                oc2, st2 = try_compile(opt_f, (idx,), ct)
                rec["flops_ind_status"] = st2
                if oc2 is not None:
                    with torch.no_grad():
                        vr3 = verify_equiv(model(idx), oc2(idx))
                    rec["flops_ind_rel"] = vr3.max_rel
                    if vr3.passed:
                        variants.append(
                            Variant(
                                "catopt_flops+ind",
                                _fwd_stmt(oc2, idx),
                            )
                        )
                print(
                    f"  catopt_flops+inductor: "
                    f"{rec['flops_ind_status']}",
                    flush=True,
                )
            print(
                f"  flops lane: {rec['flops_opt_s']}s "
                f"delivered={rec['flops_ledger']['delivered_derived']} "
                f"params→{rec['flops_params_delivered']} "
                f"fresh rel={vrf.max_rel:.3e}",
                flush=True,
            )
        except Exception as e:
            rec["flops_error"] = f"{type(e).__name__}: {e}"
            print(
                f"  flops lane FAILED: {rec['flops_error']}", flush=True
            )

    return rec, Case(
        name=rec["name"],
        params=rec["params"],
        variants=variants,
        aux=rec,
    )


# ---------------------------------------------------------------------------
#  Harness entry point (run_all.py convention)
# ---------------------------------------------------------------------------


def _parse_budgets(spec: str) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(
            (
                tok,
                None
                if tok.lower() in ("none", "exact")
                else float(tok),
            )
        )
    return out


def run_bench(args) -> Report:
    dev = torch.device(getattr(args, "device", None) or "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("  --device cuda but CUDA is unavailable; using cpu")
        dev = torch.device("cpu")
    seq = int(getattr(args, "seq", None) or 128)
    min_run_time = float(getattr(args, "min_run_time", None) or 0.2)
    warmup = int(getattr(args, "warmup", None) or 3)
    n_prompts = int(getattr(args, "prompts", None) or 4)
    ct = float(getattr(args, "compile_timeout", None) or 60.0)
    budgets = _parse_budgets(
        getattr(args, "budgets", None) or _BUDGET_DEFAULT
    )
    lora_ranks = [
        int(s)
        for s in str(
            getattr(args, "lora_ranks", None) or "4,8,16"
        ).split(",")
        if s.strip()
    ]
    prune_fracs = [
        float(s)
        for s in str(getattr(args, "prune_fracs", None) or "0.4").split(
            ","
        )
        if s.strip()
    ]
    lr_ranks = [
        int(s)
        for s in str(getattr(args, "lr_ranks", None) or "32").split(",")
        if s.strip()
    ]
    lr_dtypes = [
        s.strip()
        for s in str(
            getattr(args, "lr_dtypes", None) or "f32,f64"
        ).split(",")
        if s.strip()
    ]

    model, cfg, ckpt_path = _load_stories(
        getattr(args, "ckpt", None), dev
    )
    torch.manual_seed(0)
    idx = torch.randint(0, cfg["vocab"], (1, seq), device=dev)
    prompts = _prompt_set(cfg["vocab"], seq, n_prompts, dev)
    print(
        f"structured_models — {Path(ckpt_path).name} "
        f"dim={cfg['dim']} L={cfg['n_layers']} T={seq} "
        f"device={dev} budgets={[b for b, _ in budgets]}",
        flush=True,
    )
    t_run = time.perf_counter()

    # -- build the variants + their shared baselines -------------------
    # Each entry: (variant name, model, baseline (name, module) pairs,
    # meta).  Baselines are built/compiled ONCE per variant (not per
    # cell); Runner retires Variant stmts after timing, so cells
    # rebuild them from the modules.
    built: list[tuple[str, nn.Module, list, dict]] = []

    want = getattr(args, "variants", None)
    want_set = (
        {s.strip() for s in str(want).split(",") if s.strip()}
        if want
        else None
    )

    def _keep(name: str) -> bool:
        return want_set is None or name in want_set

    for r in lora_ranks:
        vname = f"lora_r{r}"
        if not _keep(vname):
            continue
        lm = copy.deepcopy(model)
        apply_lora(lm, r, torch.Generator().manual_seed(20261000 + r))
        lm = lm.to(dev).eval()
        bases: list = [("eager", lm)]
        ind, st_ = (
            try_compile(copy.deepcopy(lm), (idx,), ct)
            if ct > 0
            else (None, "disabled")
        )
        print(f"  {vname}: inductor {st_}", flush=True)
        bases.append(("inductor", ind if ind is not None else lm))
        ind_status = st_
        merged = merge_lora(lm)
        bases.append(("merged_eager", merged))
        mind, mst = (
            try_compile(copy.deepcopy(merged), (idx,), ct)
            if ct > 0
            else (None, "disabled")
        )
        print(f"  {vname}: merged inductor {mst}", flush=True)
        if mind is not None:
            bases.append(("merged_inductor", mind))
        meta = {
            "params_model": _n_params(lm),
            "params_merged": _n_params(merged),
            # The detector IS part of this variant's story: it must
            # correctly offer NOTHING (spelled LoRA factors are the
            # assoc laws' job — the pass only scans dense weights).
            # Armed at exact mode only — the pass is budget-
            # independent (offer_low_rank_factors takes no budget),
            # so a bounded cell would pay ~2 min of Gram-Schmidt
            # declines for identical offers.
            "detect_factors": True,
            "factors_exact_only": True,
            "inductor_status": ind_status,
            "flops_lane": False,
        }
        built.append((vname, lm, bases, meta))

    for fr in prune_fracs:
        vname = f"pruned_f{fr:g}"
        if not _keep(vname):
            continue
        pm = copy.deepcopy(model)
        keeps = prune_ffn(
            pm, fr, torch.Generator().manual_seed(20261100)
        )
        pm = pm.to(dev).eval()
        bases = [("eager", pm)]
        ind, st_ = (
            try_compile(copy.deepcopy(pm), (idx,), ct)
            if ct > 0
            else (None, "disabled")
        )
        print(f"  {vname}: inductor {st_}", flush=True)
        bases.append(("inductor", ind if ind is not None else pm))
        ind_status = st_
        narrowed = build_narrowed(pm, keeps).to(dev)
        bases.append(("narrowed_eager", narrowed))
        meta = {
            "params_model": _n_params(pm),
            "params_narrowed": _n_params(narrowed),
            "prune_frac": fr,
            # Elision is the detect_specials story; detect_factors
            # would only spend ~2 min/cell Gram-Schmidt-ing dense
            # full-rank weights (the head site alone is 32000 rows).
            "detect_factors": False,
            "inductor_status": ind_status,
            # The informative lane: under the default executor-aware
            # cost the fused paired GEMM beats the elided+gathered
            # form at these sizes; pure-FLOP extraction prices the
            # flop cut alone.  flops lane = exact cells (see run_cell).
            "flops_lane": True,
        }
        built.append((vname, pm, bases, meta))

    for r in lr_ranks:
        for dt in lr_dtypes:
            vname = f"lowrank_r{r}_{dt}"
            if not _keep(vname):
                continue
            rm = copy.deepcopy(model)
            # Cast BEFORE building the factor products: the f32 lane's
            # whole point is the fp32-rounded B·A (detector's 1e-8
            # gate sits below that noise floor); the f64 lane must
            # carry the product exactly or it reads the same noise.
            if dt == "f64":
                rm = rm.double()
            build_lowrank(
                rm, r, torch.Generator().manual_seed(20261200 + r)
            )
            rm = rm.to(dev).eval()
            bases = [("eager", rm)]
            ind, st_ = (
                try_compile(copy.deepcopy(rm), (idx,), ct)
                if ct > 0
                else (None, "disabled")
            )
            print(f"  {vname}: inductor {st_}", flush=True)
            bases.append(("inductor", ind if ind is not None else rm))
            meta = {
                "params_model": _n_params(rm),
                "dtype": dt,
                # The whole point of this variant.
                "detect_factors": True,
                "inductor_status": ind_status,
                "flops_lane": False,
            }
            # Evidence for the detector verdict: the pass's own
            # Gram-Schmidt certificate on one block weight — the
            # basis size and Frobenius residual at its shipped
            # rel_tol and at a loose 1e-4.  fp32 declines because
            # 1e-8 sits below the dtype's noise floor.
            meta["detector_probe"] = _factor_probe(
                rm.blocks[0].wq.weight
            )
            built.append((vname, rm, bases, meta))

    runner = Runner(
        device=dev, warmup=max(warmup, 2), min_run_time=min_run_time
    )
    cells: list[Cell] = []
    for vname, mod, extra, meta in built:
        for label, budget in budgets:
            rec, case = run_cell(
                mod,
                idx,
                budget,
                label,
                vname,
                args,
                dev,
                extra,
                prompts,
                bool(meta.get("detect_factors"))
                and (
                    budget is None or not meta.get("factors_exact_only")
                ),
                bool(meta.get("flops_lane")),
            )
            rec.update(meta)
            cell = runner.run_case(case)
            cell.aux.update(rec)
            cells.append(cell)
            print(
                "  times: "
                + "  ".join(
                    f"{n}={cell.medians[n] * 1e3:.3f}ms"
                    for n in cell.medians
                ),
                flush=True,
            )
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()

    # -- the honest payoff table ----------------------------------------
    print("\n=== payoff table ===", flush=True)
    hdr = (
        f"{'variant':>16} | {'budget':>6} | {'opt s':>6} | "
        f"{'offers':>14} | {'delivered':>18} | {'params Δ':>9} | "
        f"{'fresh rel':>9} | {'KL':>9} | "
        f"{'catopt ms':>9} | {'vs eager':>8} | {'vs ind':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for cell in cells:
        rec = cell.aux
        ms = cell.medians
        e, ind, c = (
            ms.get("eager"),
            ms.get("inductor"),
            ms.get("catopt"),
        )
        p_in = rec.get("params_in") or rec.get("params_model") or 0
        p_out = rec.get("params_delivered") or 0
        p_ratio = (p_out / p_in) if p_in else 0.0
        kl = (rec.get("proxy") or {}).get("mean_kl")
        print(
            f"{rec['params']['variant']:>16} | "
            f"{rec['params']['budget']:>6} | "
            f"{rec.get('opt_s', 0):>6} | "
            f"{str(rec.get('offers_by_kind', {}))[:14]:>14} | "
            f"{str(rec.get('delivered_derived', {}))[:18]:>18} | "
            f"{p_ratio:>8.3f} | "
            f"{(rec.get('fresh_rel') or 0.0):>9.2e} | "
            f"{(kl if kl is not None else 0.0):>9.2e} | "
            + (f"{c * 1e3:>9.3f} | " if c else f"{'—':>9} | ")
            + (f"{e / c:>7.3f}x | " if (c and e) else f"{'—':>8} | ")
            + (f"{ind / c:>7.3f}x" if (c and ind) else f"{'—':>8}")
        )
    print("-" * len(hdr))
    print(
        "  offers = special/factor members the detectors offered; "
        "delivered = derived params present in the shipped weights "
        "(elide/elide_bounded/low_rank/fused_fold).  'params Δ' is "
        "delivered/in-model param count.",
        flush=True,
    )
    print(
        f"  total wall time {time.perf_counter() - t_run:.1f}s",
        flush=True,
    )

    report = Report(
        suite="structured_models", cells=cells, env=collect_env(dev)
    )
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "structured_models.json"
        md_path = out_dir / "structured_models.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="eager")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "structured-model E2E: LoRA'd / pruned / built-low-rank "
            "stories15M through the real per-block pipeline "
            "(detect_specials + detect_factors, exact then bounded "
            "budgets) — measured latency vs eager/inductor plus "
            "deployment baselines, delivered param deltas, and "
            "held-out KL."
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument(
        "--budgets",
        type=str,
        default=_BUDGET_DEFAULT,
        help="comma list; 'none'/'exact' = the exact-search baseline",
    )
    ap.add_argument(
        "--variants",
        type=str,
        default=None,
        help="comma list of variant names to run "
        "(e.g. lora_r8,pruned_f0.4,lowrank_r32_f64); default = all",
    )
    ap.add_argument("--lora-ranks", type=str, default="4,8,16")
    ap.add_argument("--prune-fracs", type=str, default="0.4")
    ap.add_argument("--lr-ranks", type=str, default="32")
    ap.add_argument(
        "--lr-dtypes",
        type=str,
        default="f32,f64",
        help="weight dtypes for the low-rank lane — f32 is the "
        "detector's honest-decline control (rel_tol sits under the "
        "fp32 noise floor); f64 lets the certificate fire",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="checkpoint path — default resolves "
        "$XDG_CACHE_HOME/catopt/stories15M.bin then /tmp (fetch.py)",
    )
    ap.add_argument(
        "--prompts",
        type=int,
        default=4,
        help="random held-out prompts (+1 periodic) for KL/top-k",
    )
    ap.add_argument(
        "--compile-timeout",
        type=float,
        default=60.0,
        help="torch.compile budget for the inductor baselines, s "
        "(0 disables)",
    )
    ap.add_argument(
        "--catopt-inductor",
        default="1",
        help="also compile each delivered module (0 disables)",
    )
    ap.add_argument(
        "--flops-lane",
        default="exact",
        choices=("none", "exact", "all"),
        help="re-run the pipeline under pure-FLOP extraction "
        "(cost_fn=flops_cost): 'exact' = only the budget=none cell "
        "(default), 'all' = every budget, 'none' = off",
    )
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--min-run-time",
        type=float,
        default=0.2,
        help="blocked_autorange window per variant, seconds",
    )
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--out", default="bench/results", help="artifact dir"
    )
    ap.add_argument("--no-artifacts", action="store_true")
    args = ap.parse_args()
    if args.quick:
        for k, v in QUICK.items():
            setattr(args, k, v)
    run_bench(args)


if __name__ == "__main__":
    main()
