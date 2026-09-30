"""Bounded-rewrite E2E bench — what does ``error_budget=`` buy on a
REAL checkpoint?

Plan-0012's certified bounded rewrites (``search(...,
error_budget=)`` + ``detect_specials=True``'s bounded
``elide_bounded``/``zero_bounded`` offers) shipped with unit
certificates; this bench measures them end-to-end on the real
llama2.c **stories15M** checkpoint (the ``Stories15M`` wrapper from
``stories15m_bench`` — sdpa attention, rmsnorm, swiglu, tied head).

Per budget cell (``--budgets``, default
``none,1e-4,1e-3,3e-3,1e-2``):

* ``Optimizer.optimize(strategy=Compositional(),
  detect_specials=True, error_budget=B)`` — the real per-block
  pipeline; per-block ``rel_diff``, the offered specials ledger
  (``stats["weight_specials"]``), the accepted bound ledger
  (``stats["error_bounds"]`` / ``error_bound_total``), the cache
  record and the driver's e2e verify are all captured.
* Timed variants: ``eager`` / ``inductor`` / ``catopt`` (+ optional
  ``catopt+inductor`` — the delivered module re-compiled).
* Held-out impact proxy: fresh seeded prompts (random + a repeated
  phrase — no tokenizer ships with the checkpoint), per-position
  next-token KL(ref‖opt), top-1 and top-5 agreement.

The morphism lane is probed too — ``KVLatentShare(budget=)`` via
``MorphismSearch`` — and the plumbing gap is documented honestly:
``_optimize_morphisms`` accepts no ``error_budget``/
``detect_specials`` kwargs (they raise TypeError through
``optimize``), and stories15M's ``Block(h, cos, sin)`` signature
fails the law's single-input candidacy regardless.

Usage:
    .venv/bin/python bench/bounded_e2e.py --device cpu
    .venv/bin/python bench/bounded_e2e.py --quick
"""
# ruff: noqa: E402, RUF003 -- ×, ·, →, −, ‖ in
# strings/docstrings are deliberate math notation; sys.path setup
# must precede the benchkit/catopt imports (bench_omd2 convention).

from __future__ import annotations

import argparse
import gc
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.setrecursionlimit(400_000)

import catopt_orchestrator.morphisms_kv as K
import numpy as np
import torch
import torch.nn.functional as F
from benchkit import (
    Case,
    Cell,
    Report,
    Runner,
    Variant,
    collect_env,
)
from catopt_orchestrator import (
    Compositional,
    MorphismSearch,
    Optimizer,
    optimize_morphisms,
)
from catopt_torch.backend import TorchBackend
from catopt_torch.report import verify_equiv
from llama2c import load_llama2c
from real_win_hunt import try_compile
from stories15m_bench import Stories15M, resolve_ckpt

# run_all.py picks these up for its --quick lane.
QUICK = {
    "budgets": "none,1e-3",
    "seq": "64",
    "min_run_time": "0.05",
    "compile_timeout": "45.0",
    "prompts": "2",
    "catopt_inductor": "0",
}

_BUDGET_DEFAULT = "none,1e-4,1e-3,3e-3,1e-2"


# ---------------------------------------------------------------------------
#  Model loading (stories15m_bench conventions)
# ---------------------------------------------------------------------------


def _load_stories(ckpt: str | None, device: torch.device):
    """Parse the llama2.c header + weights; return (model, cfg)."""
    path = resolve_ckpt(ckpt)
    w = load_llama2c(path)
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(28), dtype=np.int32)
    cfg = dict(
        dim=int(hdr[0]),
        hidden=int(hdr[1]),
        n_layers=int(hdr[2]),
        n_heads=int(hdr[3]),
        vocab=int(w["token_embedding"].shape[0]),
        seq_len=int(hdr[6]),
    )
    m = Stories15M(w, cfg).eval().to(device)
    return m, cfg, path


# ---------------------------------------------------------------------------
#  Held-out proxy eval — next-token distribution agreement
# ---------------------------------------------------------------------------


def _prompt_set(
    vocab: int, seq: int, n: int, device: torch.device
) -> list[torch.Tensor]:
    """Seeded random prompts + one repeated-phrase prompt.

    No tokenizer ships in the repo's cache (fetch.py pulls only the
    .bin weights), so the held-out probe is token-agnostic: random
    sequences plus a periodic one — the periodic prompt exercises the
    model on a strongly peaked distribution, the randoms on diffuse
    ones.
    """
    g = torch.Generator().manual_seed(20260927)
    prompts = [
        torch.randint(0, vocab, (1, seq), generator=g) for _ in range(n)
    ]
    pat = torch.randint(0, vocab, (1, 8), generator=g)
    prompts.append(pat.repeat(1, (seq + 7) // 8)[:, :seq].contiguous())
    return [p.to(device) for p in prompts]


def _proxy_eval(ref_mod, opt_mod, prompts: list[torch.Tensor]) -> dict:
    """Per-position next-token agreement of ``opt_mod`` vs ``ref_mod``.

    Returns mean/max KL(ref‖opt) in nats over all positions, the
    top-1 agreement fraction, and the fraction of positions where the
    reference argmax sits inside the optimized model's top-5 — the
    cheap stand-in for a perplexity delta when no held-out text (or
    tokenizer) is available.
    """
    kls = []
    top1 = 0
    top5 = 0
    n_pos = 0
    max_abs = 0.0
    with torch.no_grad():
        for p in prompts:
            r = ref_mod(p).float()
            o = opt_mod(p).float()
            lr = F.log_softmax(r, dim=-1)
            lo = F.log_softmax(o, dim=-1)
            kl = (lr.exp() * (lr - lo)).sum(-1)  # (1,T)
            kls.append(kl.flatten())
            r1 = r.argmax(-1).flatten()
            o5 = o.topk(5, dim=-1).indices.flatten(0, -2)
            top1 += int((r.argmax(-1) == o.argmax(-1)).sum())
            top5 += int((o5 == r1[:, None]).any(-1).sum())
            n_pos += r1.numel()
            max_abs = max(max_abs, (r - o).abs().max().item())
    kl_all = torch.cat(kls)
    return {
        "mean_kl": float(kl_all.mean()),
        "max_kl": float(kl_all.max()),
        "top1_agree": top1 / n_pos,
        "top5_in": top5 / n_pos,
        "max_abs_logit": max_abs,
        "n_positions": n_pos,
    }


# ---------------------------------------------------------------------------
#  Timing stmt
# ---------------------------------------------------------------------------


def _fwd_stmt(mod, idx):
    def stmt() -> None:
        with torch.no_grad():
            mod(idx)

    return stmt


# ---------------------------------------------------------------------------
#  One budget cell
# ---------------------------------------------------------------------------


def _bounded_ledger(stats: dict) -> dict:
    """Roll the per-block bound records into one cell ledger."""
    per_block: dict[str, dict] = {}
    n_offers = 0
    n_bounded_offers = 0
    for name, rep in (stats.get("blocks") or {}).items():
        st = rep.get("stats") or {}
        specials = st.get("weight_specials") or []
        bounds = st.get("error_bounds") or []
        n_offers += len(specials)
        n_bounded_offers += sum(
            1 for s in specials if (s.get("error_bound") or 0.0) > 0
        )
        per_block[name] = {
            "status": rep.get("status"),
            "cache": rep.get("cache"),
            "rel_diff": rep.get("rel_diff"),
            "error": rep.get("error"),
            "time_s": round(rep.get("time_s") or 0.0, 1),
            "bound_total": st.get("error_bound_total"),
            "bounds": [
                {
                    k: e.get(k)
                    for k in (
                        "rule",
                        "bound",
                        "norm",
                        "measured_max_rel",
                    )
                }
                for e in bounds
            ],
            "specials_offered": [
                {
                    "param": s.get("param"),
                    "kind": s.get("kind"),
                    "error_bound": s.get("error_bound"),
                }
                for s in specials
            ],
            "bounds_honored": st.get("error_bounds_honored"),
        }
    return {
        "per_block": per_block,
        "n_special_offers": n_offers,
        "n_bounded_offers": n_bounded_offers,
    }


def run_budget_cell(
    model,
    cfg: dict,
    idx: torch.Tensor,
    budget: float | None,
    label: str,
    args,
    dev: torch.device,
    inductor_mod,
    prompts: list[torch.Tensor],
) -> tuple[dict, Case]:
    """Optimize under ``budget``, verify, pack the benchkit case."""
    print(f"\n=== budget={label} ===", flush=True)
    rec: dict = {
        "name": f"budget={label}",
        "params": {"budget": label, "seq": idx.shape[-1]},
    }

    t0 = time.time()
    try:
        opt_mod, stats = Optimizer(backend=TorchBackend()).optimize(
            model,
            idx,
            strategy=Compositional(),
            detect_specials=True,
            error_budget=budget,
            verbose=False,
        )
        rec["opt_s"] = round(time.time() - t0, 1)
        rec["n_optimized"] = stats.get("n_optimized")
        rec["n_blocks"] = stats.get("n_blocks")
        rec["cache"] = stats.get("cache")
        rec["e2e"] = stats.get("end_to_end")
        rec.update(_bounded_ledger(stats))
        n_delivered = sum(
            len(b["bounds"]) for b in rec["per_block"].values()
        )
        rec["n_bounds_delivered"] = n_delivered
        rec["bound_total_max"] = max(
            (
                b["bound_total"] or 0.0
                for b in rec["per_block"].values()
            ),
            default=0.0,
        )
        # A block declined by the honest bound gate never gets its
        # search stats recorded (the driver drops ``rep["stats"]`` on
        # failure) — its delivered bounded members are invisible to
        # the offer/delivery counters above.  Recover the verdict
        # evidence from the driver's error string, which always
        # reports ``(accepted bound …)``.
        declines = []
        for bn, b in rec["per_block"].items():
            err = b.get("error") or ""
            m = re.search(
                r"rel diff (\S+) \(accepted bound (\S+)\)", err
            )
            if m:
                declines.append(
                    {
                        "block": bn,
                        "status": b["status"],
                        "measured_rel": float(m.group(1)),
                        "accepted_bound": float(m.group(2)),
                        "error": err,
                    }
                )
        if declines:
            rec["bound_declines"] = declines
        print(
            f"  optimize: {rec['opt_s']}s "
            f"opt={rec['n_optimized']}/{rec['n_blocks']} "
            f"cache={rec['cache']} "
            f"bounded offers={rec['n_bounded_offers']} "
            f"delivered={n_delivered} "
            f"bound_max={rec['bound_total_max']:.3e}",
            flush=True,
        )
        statuses = {b["status"] for b in rec["per_block"].values()}
        if statuses - {"optimized"}:
            odd = {
                k: (b["status"], b.get("error"))
                for k, b in rec["per_block"].items()
                if b["status"] != "optimized"
            }
            print(f"  non-optimized blocks: {odd}", flush=True)
    except Exception as e:
        rec["opt_s"] = round(time.time() - t0, 1)
        rec["opt_error"] = f"{type(e).__name__}: {e}"
        print(f"  optimize FAILED: {rec['opt_error']}", flush=True)
        opt_mod = None

    if opt_mod is not None:
        # Fresh-input verify + held-out proxy — the pipeline's own
        # e2e check ran on `idx`; this is a different input.
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

    variants = [
        Variant("eager", _fwd_stmt(model, idx)),
        Variant("inductor", _fwd_stmt(inductor_mod, idx)),
    ]
    if opt_mod is not None:
        variants.append(Variant("catopt", _fwd_stmt(opt_mod, idx)))
        if getattr(args, "catopt_inductor", None) not in (None, "0", 0):
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
    return rec, Case(
        name=rec["name"],
        params=rec["params"],
        variants=variants,
        aux=rec,
    )


# ---------------------------------------------------------------------------
#  Morphism-lane probe — does the budget wiring reach KVLatentShare?
# ---------------------------------------------------------------------------


def probe_morphism(model, idx, budget: float) -> dict:
    """Exercise ``MorphismSearch`` + ``KVLatentShare(budget=)`` on the
    real model; also record the search-level plumbing gap.

    ``optimize(..., error_budget=)`` forwards through ``**kw`` — under
    ``MorphismSearch`` it lands on ``_optimize_morphisms``, whose
    signature has no ``error_budget``/``detect_specials`` parameter,
    so the call raises ``TypeError``.  The law-level budget
    (``KVLatentShare(budget=)``) DOES reach the law through
    ``laws=[...]``; the stories15M blocks are still ineligible —
    ``Block.forward(h, cos, sin)`` exports with 3 inputs, failing the
    single-input candidacy, and a sequential chain has no shared-input
    cross-block family.
    """
    rec: dict = {}
    # 1) The plumbing gap — error_budget does not reach the strategy.
    try:
        Optimizer(backend=TorchBackend()).optimize(
            model,
            idx,
            strategy=MorphismSearch(laws=[], optimize_rest=False),
            error_budget=budget,
        )
        rec["error_budget_kw"] = "accepted"
    except TypeError as e:
        rec["error_budget_kw"] = f"TypeError: {e}"
    except Exception as e:
        rec["error_budget_kw"] = f"{type(e).__name__}: {e}"

    # 2) The law-level budget on the real graph (kv names are wk/wv).
    t0 = time.time()
    try:
        lr = optimize_morphisms(
            model,
            idx,
            backend=TorchBackend(),
            strategy=MorphismSearch(
                laws=[
                    K.KVLatentShare(budget=budget, tokens=("wk", "wv"))
                ],
                optimize_rest=False,
            ),
        )
        st = lr.stats
        rec["kv_budget"] = budget
        rec["opt_s"] = round(time.time() - t0, 1)
        rec["n_blocks"] = st.get("n_blocks")
        rec["n_lifted"] = st.get("n_lifted")
        rec["n_rewritten"] = st.get("n_rewritten")
        rec["matches"] = {
            k: {
                kk: v.get(kk)
                for kk in ("status", "reason", "detail", "error")
            }
            for k, v in (st.get("matches") or {}).items()
        }
        rec["end_to_end"] = st.get("end_to_end")
    except Exception as e:
        rec["kv_error"] = f"{type(e).__name__}: {e}"
        rec["opt_s"] = round(time.time() - t0, 1)
    return rec


# ---------------------------------------------------------------------------
#  Harness entry point (run_all.py convention)
# ---------------------------------------------------------------------------


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

    budget_spec = getattr(args, "budgets", None) or _BUDGET_DEFAULT
    budgets: list[tuple[str, float | None]] = []
    for tok in budget_spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        budgets.append(
            (
                tok,
                None
                if tok.lower() in ("none", "exact")
                else float(tok),
            )
        )

    model, cfg, ckpt_path = _load_stories(
        getattr(args, "ckpt", None), dev
    )
    torch.manual_seed(0)
    idx = torch.randint(0, cfg["vocab"], (1, seq), device=dev)
    prompts = _prompt_set(cfg["vocab"], seq, n_prompts, dev)
    print(
        f"bounded_e2e — {Path(ckpt_path).name} "
        f"dim={cfg['dim']} L={cfg['n_layers']} T={seq} "
        f"device={dev} budgets={[b for b, _ in budgets]}",
        flush=True,
    )
    t0 = time.perf_counter()

    # One inductor baseline, shared across cells.
    inductor_mod = model
    ind_status = "disabled (--compile-timeout 0)"
    if ct > 0:
        import copy

        inductor_mod2, ind_status = try_compile(
            copy.deepcopy(model), (idx,), ct
        )
        if inductor_mod2 is not None:
            inductor_mod = inductor_mod2
    print(f"  inductor baseline: {ind_status}", flush=True)

    # Morphism-lane probe — once, at the loosest budget in the sweep.
    loose = max((b for _, b in budgets if b is not None), default=None)
    morph_rec = None
    if loose is not None and getattr(args, "morphism", "1") != "0":
        print("\n=== morphism probe ===", flush=True)
        morph_rec = probe_morphism(model, idx, loose)
        print(f"  {morph_rec}", flush=True)

    runner = Runner(
        device=dev, warmup=max(warmup, 2), min_run_time=min_run_time
    )
    recs: list[dict] = []
    cells: list = []
    for label, budget in budgets:
        rec, case = run_budget_cell(
            model,
            cfg,
            idx,
            budget,
            label,
            args,
            dev,
            inductor_mod,
            prompts,
        )
        recs.append(rec)
        cell = runner.run_case(case)
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

    # -- post-timing: speedups + honest verdict ------------------------------
    print("\n=== sweep ===", flush=True)
    hdr = (
        f"{'budget':>8} | {'opt s':>6} | {'offers':>6} | "
        f"{'deliv':>5} | {'bound':>8} | {'fresh_rel':>9} | "
        f"{'KL mean':>9} | {'top1':>6} | "
        f"{'catopt ms':>9} | {'vs eager':>8} | {'vs ind':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for rec, cell in zip(recs, cells, strict=True):
        ms = cell.medians
        e, ind, c = (
            ms.get("eager"),
            ms.get("inductor"),
            ms.get("catopt"),
        )
        rec["ms_eager"] = e
        rec["ms_inductor"] = ind
        rec["ms_catopt"] = c
        if c and e:
            rec["speedup_vs_eager"] = e / c
        if c and ind:
            rec["speedup_vs_inductor"] = ind / c
        p = rec.get("proxy") or {}
        print(
            f"{rec['params']['budget']:>8} | "
            f"{rec.get('opt_s', 0):>6} | "
            f"{rec.get('n_bounded_offers', 0):>6} | "
            f"{rec.get('n_bounds_delivered', 0):>5} | "
            f"{rec.get('bound_total_max', 0.0):>8.2e} | "
            f"{(rec.get('fresh_rel') or 0.0):>9.2e} | "
            f"{(p.get('mean_kl') or 0.0):>9.2e} | "
            f"{(p.get('top1_agree') or 0.0):>6.3f} | "
            + (f"{c * 1e3:>9.3f} | " if c else f"{'—':>9} | ")
            + (f"{e / c:>7.3f}x | " if (c and e) else f"{'—':>8} | ")
            + (f"{ind / c:>7.3f}x" if (c and ind) else f"{'—':>8}")
        )
        # The honest one-line verdict.
        if rec.get("opt_error"):
            verdict = f"optimize failed: {rec['opt_error']}"
        elif rec.get("bound_declines"):
            d = rec["bound_declines"][0]
            verdict = (
                f"bounded member delivered on {d['block']} but "
                f"DECLINED by the bound gate: measured rel "
                f"{d['measured_rel']:.2e} > certified bound "
                f"{d['accepted_bound']:.2e} — budget bought nothing, "
                "cost pipeline time"
            )
        elif rec.get("n_bounds_delivered", 0) == 0:
            verdict = (
                "no bounded member delivered — budget changes nothing"
            )
            if rec.get("n_bounded_offers", 0) == 0:
                verdict = (
                    "no bounded offer even qualified — no weight is "
                    "within budget of a structural special"
                )
        else:
            sp = rec.get("speedup_vs_eager") or 0.0
            verdict = (
                f"{rec['n_bounds_delivered']} bounded members, "
                f"bound {rec['bound_total_max']:.2e}, "
                f"speedup {sp:.3f}x, "
                f"KL {p.get('mean_kl', 0.0):.2e}"
            )
        rec["verdict"] = verdict
        cell.aux.update(rec)  # rec mutated after run_case copied it
        print(f"{'':>8} | verdict: {verdict}")
    print("-" * len(hdr))
    print(
        "  offers = bounded members the specials pass offered; "
        "deliv = bounded members the extracted term actually "
        "delivers.  KL/top1 measured on held-out prompts.",
        flush=True,
    )
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    if morph_rec is not None:
        # Plumbing/morphism outcome rides as an untimed aux cell so
        # it lands in the JSON/MD artifacts with everything else.
        cells.append(
            Cell(
                case=Case(
                    name="morphism_probe",
                    params={"budget": f"kv:{loose}", "seq": seq},
                    variants=[],
                    aux=morph_rec,
                ),
                medians={},
                iqr={},
                aux=morph_rec,
            )
        )

    report = Report(
        suite="bounded_e2e", cells=cells, env=collect_env(dev)
    )
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(getattr(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "bounded_e2e.json"
        md_path = out_dir / "bounded_e2e.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="eager")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "bounded-rewrite E2E on the real stories15M checkpoint: "
            "sweep search(..., error_budget=B) with "
            "detect_specials=True; measure speedup vs eager/inductor, "
            "the accepted bound ledger, and held-out next-token "
            "agreement."
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
        help="torch.compile budget for the inductor baseline, s "
        "(0 disables)",
    )
    ap.add_argument(
        "--catopt-inductor",
        default="1",
        help="also compile each delivered module (0 disables)",
    )
    ap.add_argument(
        "--morphism",
        default="1",
        help="run the KVLatentShare(budget=) morphism probe (0 skips)",
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
