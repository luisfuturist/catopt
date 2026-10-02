"""Morphism coverage on REAL checkpoints — what the block-level layer sees.

Plan-0011's morphism engine lifts each composer-selected block to a
:class:`BlockSig`, classifies the boundary wires, fires signature laws
(``WindowCompose`` / ``ResidualReassoc`` / ``OutInCompose`` /
``ResidualAbsorb`` / ``NormCascade`` / ``WeightTie`` + the KV-family
law ``KVLatentShare``), reifies each match through a verified joint
e-graph, and grafts the survivors.  The synthetic fixtures
(``bench/suites/algebra/morphism_e2e.py``) show the wins; this suite
asks the honest question on *real* trained weights — the llama2.c
``stories15M`` and ``stories110M`` checkpoints:

1. **Lift** — how many of the composer's blocks lift, and with what
   input signatures (the multi-input ``(h, cos, sin)`` attention
   blocks are the post-signature-landing question), plus the wire
   kinds between them.
2. **Match / fire / graft** — each law's candidacy count from
   ``law.match(graph)`` and its reify verdicts inside a real
   ``MorphismSearch`` run (``stats["matches"]`` /
   ``stats["morphism_fires"]``): which laws see real structure and
   which rewrites the value/cost gates let through.
3. **E2E** — ``optimize_morphisms`` on the real checkpoint: what the
   delivered module is, whether it verifies on a fresh input, and the
   measured latency delta vs eager (benchkit timings).

Honesty guards: the *full default* law stack is attempted in a
**subprocess** with an address-space cap and a wall-clock timeout —
a law whose reify cannot complete on this host (on this tree the
``WindowCompose`` 5/11-block match dies inside ``_reify``'s
``op_repr`` stats-rendering: the joint term is a sharing-heavy DAG
and ``op_repr`` expands it as a tree, so the repr itself explodes
before the cost gate even speaks) must not take the suite down with
it.  The guarded run records progress per match, so a kill is
attributed to the exact match that died.  The in-process delivery
run then proceeds without the resource-bound law — labelled
``window_excluded`` in the record — so timing still measures a real
delivered module.

Usage:
    .venv/bin/python -m bench run morphism_coverage --device cpu
    .venv/bin/python bench/suites/models/morphism_coverage.py \
        --models stories15M --seq 64
"""
# ruff: noqa: E402, RUF001 -- math notation in docstrings is
# deliberate; sys.path setup precedes imports.

from __future__ import annotations

import argparse
import copy
import gc
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.setrecursionlimit(400_000)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import catopt_orchestrator.morphisms as M
import catopt_orchestrator.morphisms_kv as K
import numpy as np
import torch
from catopt_orchestrator import MorphismSearch, Optimizer
from catopt_torch.backend import TorchBackend
from catopt_torch.report import verify_equiv

from bench.benchkit import (
    Case,
    Finding,
    Report,
    Runner,
    Variant,
    Verdict,
    collect_env,
)
from bench.common.llama2c import load_llama2c
from bench.suites.e2e.stories15m_bench import (
    Stories15M,
    cache_dir,
    resolve_ckpt,
)
from bench.suites.speedup.real_win_hunt import try_compile

# run_all picks these up for its --quick lane.
QUICK = {
    "models": "stories15M",
    "seq": "48",
    "inductor": "0",
    "search_timeout": "180",
    "min_run_time": "0.05",
}


def _opt(args, name, default):
    """``getattr`` that honors defaults under the CLI's lax namespace."""
    v = getattr(args, name, None)
    return default if v is None else v


#: Checkpoint registry — ``name -> llama2.c .bin filename``.
#: stories110M runs only when its checkpoint is present.
_CKPTS = {
    "stories15M": "stories15M.bin",
    "stories110M": "stories110M.bin",
}

#: The morphism law registry this suite measures.  ``kv_latent_share``
#: is the plan-0011 family law living in ``morphisms_kv`` — it is not
#: in ``DEFAULT_MORPHISM_LAWS``; the stories checkpoints name their
#: K/V projections ``wk``/``wv`` so the token set follows the model.
_LAWS: dict[str, object] = {
    "window_compose": M.WindowCompose,
    "residual_reassoc": M.ResidualReassoc,
    "out_in_compose": M.OutInCompose,
    "residual_absorb": M.ResidualAbsorb,
    "norm_cascade": M.NormCascade,
    "weight_tie": M.WeightTie,
    "kv_latent_share": lambda: K.KVLatentShare(tokens=("wk", "wv")),
}


def _law_set(spec: str) -> list:
    """``all`` | ``no_window`` | a comma subset of ``_LAWS`` names."""
    if spec == "all":
        return [f() for f in _LAWS.values()]
    if spec == "no_window":
        return [f() for k, f in _LAWS.items() if k != "window_compose"]
    names = [s.strip() for s in spec.split(",") if s.strip()]
    return [_LAWS[n]() for n in names]


# ---------------------------------------------------------------------------
#  Model loading (stories15m_bench conventions — Stories15M wraps any
#  llama2.c-format checkpoint, stories110M included)
# ---------------------------------------------------------------------------


def _have_ckpt(fname: str) -> bool:
    return (cache_dir() / fname).is_file() or Path(
        "/tmp", fname
    ).is_file()


def _load_model(path: str, device: torch.device):
    """Parse the header + weights; return ``(model, cfg)``."""
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
    return Stories15M(w, cfg).eval().to(device), cfg


# ---------------------------------------------------------------------------
#  Phase A — lift coverage + law candidacy (signature level, no reify)
# ---------------------------------------------------------------------------


def _sig_record(sig) -> dict | None:
    if sig is None:
        return None
    return {
        "in_projs": [w.name for w in sig.in_projs],
        "out_proj": [w.name for w in sig.out_proj],
        "norm": sig.norm.kind,
        "norm_affine": sig.norm.affine,
        "norm_pre": sig.norm.pre,
        "act": list(sig.act),
        "residual": sig.residual,
        "shape": [
            list(s) if isinstance(s, tuple) else s for s in sig.shape
        ],
        "inputs": [
            {
                "index": i.index,
                "name": i.name,
                "kind": i.kind,
                "role": i.role,
            }
            for i in sig.inputs
        ],
    }


def lift_phase(model, idx, backend, laws) -> tuple[dict, object]:
    """``lift_graph`` + per-law ``match()`` candidacy — no reify."""
    graph = M.lift_graph(
        model, idx, source=backend.source, composer=backend.composer
    )
    input_kinds: dict[str, int] = {}
    input_roles: dict[str, int] = {}
    blocks: dict[str, dict] = {}
    for n in graph.nodes:
        rec = graph.record(n.name)
        entry = {
            "opaque": n.opaque,
            "note": rec.note,
            "calls": rec.calls,
            "sig": _sig_record(n.sig),
        }
        blocks[n.name] = entry
        if n.sig is not None:
            for i in n.sig.inputs:
                input_kinds[i.kind] = input_kinds.get(i.kind, 0) + 1
                input_roles[i.role] = input_roles.get(i.role, 0) + 1
    wire_kinds: dict[str, int] = {}
    for w in graph.wires:
        wire_kinds[w.kind] = wire_kinds.get(w.kind, 0) + 1

    candidacy: dict[str, dict] = {}
    for law in laws:
        name = getattr(law, "name", type(law).__name__)
        try:
            ms = law.match(graph)
            candidacy[name] = {
                "matches": len(ms),
                "sites": [list(m.nodes) for m in ms],
                "boundaries": [m.boundary for m in ms],
                "details": [m.detail for m in ms],
            }
        except Exception as e:  # a match() crash is itself coverage
            candidacy[name] = {"error": f"{type(e).__name__}: {e}"}

    lift = {
        "n_nodes": len(graph.nodes),
        "n_lifted": sum(1 for n in graph.nodes if not n.opaque),
        "n_opaque": sum(1 for n in graph.nodes if n.opaque),
        "input_kinds": input_kinds,
        "input_roles": input_roles,
        "wire_kinds": wire_kinds,
        "wires": [[w.src, w.dst, w.kind] for w in graph.wires],
        "blocks": blocks,
    }
    return {"lift": lift, "candidacy": candidacy}, graph


# ---------------------------------------------------------------------------
#  Honest-gap probes — structure the signature layer cannot express
# ---------------------------------------------------------------------------


def _tied_params(model) -> list[dict]:
    """Same-shape bitwise-equal parameter pairs in the model.

    Detects the classic embedding/head tie (and any other duplicated
    tensor) *on the nn.Module* — the comparison the signature level
    cannot see because a non-projection block (``nn.Embedding``)
    carries no ``WeightRef``.
    """
    try:
        sd = model.state_dict()
    except Exception:
        return []
    names = list(sd)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ta, tb = sd[a], sd[b]
            if ta.shape == tb.shape and bool(torch.equal(ta, tb)):
                out.append({"a": a, "b": b, "shape": list(ta.shape)})
    return out


def _norm_ops_seen(graph) -> dict:
    """Norm ops present in each block's IR vs the sig's verdict.

    ``_norm_nodes`` recognises ``layer_norm`` and the ``x·rsqrt·w``
    RMS spelling — a fused ``rms_norm`` op (the llama2.c export's
    spelling) counts here while the signature reads ``kind="none"``,
    which is precisely NormCascade's blind spot on this family.
    """
    norm_ops = {"layer_norm", "rms_norm", "batch_norm", "group_norm"}
    per_block: dict[str, dict] = {}
    for n in graph.nodes:
        rec = graph.record(n.name)
        if rec.ir is None:
            continue
        seen = {}
        for op in M._iter_ops(rec.ir.root):
            if op.op in norm_ops:
                seen[op.op] = seen.get(op.op, 0) + 1
        per_block[n.name] = {
            "norm_ops": seen,
            "sig_norm": n.sig.norm.kind if n.sig else None,
        }
    return per_block


# ---------------------------------------------------------------------------
#  Phase B — guarded default-stack run in a subprocess
# ---------------------------------------------------------------------------


def _sanitize(obj, depth: int = 0):
    """JSON-safe stats copy — drops the giant ``op_repr`` payloads."""
    if depth > 12:
        return "<deep>"
    if isinstance(obj, dict):
        return {
            str(k): _sanitize(v, depth + 1)
            for k, v in obj.items()
            if k not in ("joint", "reified", "reps")
        }
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v, depth + 1) for v in obj]
    if isinstance(obj, (bool, int, float)) or obj is None:
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= 500 else obj[:500] + "…"
    if isinstance(obj, torch.Tensor):
        return f"<tensor {tuple(obj.shape)}>"
    return str(obj)[:300]


def _worker_main(wargs) -> int:
    """Subprocess entry: one MorphismSearch run -> stats JSON.

    ``RLIMIT_AS`` caps the address space so a runaway reify declines
    as ``MemoryError`` instead of taking the host into swap death;
    the progress file attributes a kill to the exact match in flight.
    """
    if wargs.mem_gb:
        try:
            import resource

            cap = int(wargs.mem_gb * (1 << 30))
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (ImportError, ValueError, OSError) as e:
            print(f"worker: mem cap unavailable: {e}", flush=True)
    dev = torch.device(wargs.device)
    model, _cfg = _load_model(wargs.ckpt, dev)
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, _cfg["vocab"], (1, wargs.seq), generator=g)
    idx = idx.to(dev)

    orig_reify = M._reify
    with open(wargs.progress, "w", buffering=1) as prog:

        def logged(match, graph, **kw):
            key = f"{match.law}:{'+'.join(match.nodes)}"
            prog.write(f"BEGIN {key}\n")
            try:
                r = orig_reify(match, graph, **kw)
            except Exception as e:
                prog.write(f"RAISE {key} {type(e).__name__}: {e}\n")
                raise
            prog.write(f"END {key} status={r.get('status')}\n")
            return r

        M._reify = logged
        try:
            _mod, stats = Optimizer(backend=TorchBackend()).optimize(
                model,
                idx,
                strategy=MorphismSearch(
                    laws=_law_set(wargs.laws),
                    optimize_rest=bool(wargs.rest),
                ),
            )
        finally:
            M._reify = orig_reify
        Path(wargs.stats).write_text(json.dumps(_sanitize(stats)))
        prog.write("DONE\n")
    return 0


def _read_progress(path: Path) -> dict:
    """Fold the worker's BEGIN/END/RAISE markers into an outcome."""
    lines = path.read_text().splitlines() if path.is_file() else []
    events = []
    for line in lines:
        parts = line.split(" ", 2)
        if len(parts) >= 2:
            events.append(
                [parts[0], parts[1], parts[2] if len(parts) > 2 else ""]
            )
        elif parts:
            events.append([parts[0], "", ""])
    open_ = [k for tag, k, _ in events if tag == "BEGIN"]
    closed = {k for tag, k, _ in events if tag in ("END", "RAISE")}
    return {
        "events": events,
        "in_flight": [k for k in open_ if k not in closed],
        "completed": "DONE" in lines,
    }


def _guarded_search(
    ckpt: str,
    seq: int,
    device: str,
    laws_spec: str,
    rest: int,
    timeout_s: float,
    mem_gb: float,
) -> dict:
    """Run one MorphismSearch config in a bounded subprocess."""
    tmp = Path(tempfile.mkdtemp(prefix="morph_cov_"))
    stats_f = tmp / "stats.json"
    prog_f = tmp / "progress.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--ckpt",
        ckpt,
        "--seq",
        str(seq),
        "--device",
        device,
        "--laws",
        laws_spec,
        "--rest",
        str(rest),
        "--stats",
        str(stats_f),
        "--progress",
        str(prog_f),
        "--mem-gb",
        str(mem_gb),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout_s,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[3]),
        )
        rc: object = proc.returncode
        err_tail = (proc.stderr or "")[-800:]
    except subprocess.TimeoutExpired:
        rc = "timeout"
        err_tail = f"killed at {timeout_s:.0f}s wall limit"
    elapsed = time.time() - t0
    prog = _read_progress(prog_f)
    stats = None
    if stats_f.is_file():
        try:
            stats = json.loads(stats_f.read_text())
        except json.JSONDecodeError:
            stats = {"corrupt": True}
    out = {
        "laws": laws_spec,
        "optimize_rest": bool(rest),
        "exit": rc,
        "seconds": round(elapsed, 1),
        "died_at": prog["in_flight"],
        "progress_tail": prog["events"][-6:],
        "stderr_tail": err_tail,
        "stats": stats,
    }
    if rc == 0 and stats is not None:
        out["outcome"] = "completed"
    elif rc == "timeout":
        out["outcome"] = "timeout"
    elif rc == -9 or rc == 137:
        out["outcome"] = "oom-killed"
    else:
        out["outcome"] = f"failed(rc={rc})"
    return out


# ---------------------------------------------------------------------------
#  Phase C — in-process delivery run (delivered module is needed for
#  verification + timing, so it cannot live in the guarded subprocess)
# ---------------------------------------------------------------------------


def _delivery_run(
    model,
    idx,
    laws,
    rest: bool,
) -> tuple[object, dict]:
    """``Optimizer.optimize`` under ``MorphismSearch`` — in-process."""
    opt_mod, stats = Optimizer(backend=TorchBackend()).optimize(
        model,
        idx,
        strategy=MorphismSearch(laws=laws, optimize_rest=rest),
    )
    return opt_mod, stats


def _match_ledger(stats: dict) -> dict:
    """Per-law match/fire/graft rollup of ``stats["matches"]``."""
    per_law: dict[str, dict] = {}
    rows: dict[str, dict] = {}
    for key, mv in (stats.get("matches") or {}).items():
        law = key.split(":", 1)[0]
        st = mv.get("status")
        rec = per_law.setdefault(
            law,
            {
                "matched": 0,
                "grafted": 0,
                "declined": 0,
                "skipped": 0,
                "reasons": {},
            },
        )
        rec["matched"] += 1
        rec[st] = rec.get(st, 0) + 1
        if mv.get("reason"):
            rec["reasons"][mv["reason"]] = (
                rec["reasons"].get(mv["reason"], 0) + 1
            )
        rows[key] = {
            k: mv.get(k)
            for k in (
                "status",
                "reason",
                "boundary",
                "detail",
                "cost_before",
                "cost_after",
                "rel_diff",
                "latent_rank",
                "n_kv_sites",
                "error_bound",
                "tied",
            )
            if mv.get(k) is not None
        }
        if "tied" in rows[key]:
            rows[key]["tied"] = _sanitize(rows[key]["tied"])
    return {"per_law": per_law, "matches": rows}


def _block_ledger(stats: dict) -> dict:
    return {
        name: {
            k: rep.get(k)
            for k in ("status", "law", "rel_diff", "reason", "error")
            if rep.get(k) is not None
        }
        for name, rep in (stats.get("blocks") or {}).items()
    }


def _fwd_stmt(mod, args: tuple):
    def stmt() -> None:
        with torch.no_grad():
            mod(*args)

    return stmt


# ---------------------------------------------------------------------------
#  Per-model driver
# ---------------------------------------------------------------------------


def run_model(name: str, ckpt: str, args, dev, runner) -> dict:
    """Lift + candidacy + guarded default stack + delivery + timing."""
    seq = int(_opt(args, "seq", 64))
    print(f"\n=== {name} ({Path(ckpt).name}) seq={seq} ===", flush=True)
    t0 = time.time()
    model, cfg = _load_model(ckpt, dev)
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, cfg["vocab"], (1, seq), generator=g).to(dev)
    rec: dict = {
        "model": name,
        "ckpt": ckpt,
        "cfg": cfg,
        "seq": seq,
        "load_s": round(time.time() - t0, 1),
    }
    backend = TorchBackend()
    laws_all = _law_set("all")

    # -- A: lift + candidacy ------------------------------------------
    t1 = time.time()
    cov, _graph = lift_phase(model, idx, backend, laws_all)
    cov["seconds"] = round(time.time() - t1, 2)
    rec["coverage"] = cov
    lift = cov["lift"]
    print(
        f"  lift: {lift['n_lifted']}/{lift['n_nodes']} blocks "
        f"({cov['seconds']}s)"
        f" — kinds={lift['input_kinds']} wires={lift['wire_kinds']}",
        flush=True,
    )
    cand = cov["candidacy"]
    print(
        "  candidacy: "
        + "  ".join(
            f"{ln}={c.get('matches', c.get('error'))}"
            for ln, c in cand.items()
        ),
        flush=True,
    )
    # Honest gaps: ties the signature cannot candidate + norm ops the
    # signature's norm vocabulary misses.
    rec["gaps"] = {
        "tied_param_pairs": _tied_params(model),
        "norm_ops_vs_sig": _norm_ops_seen(_graph),
        "opaque_wires": [
            [w.src, w.dst, w.kind]
            for w in _graph.wires
            if w.kind == "opaque"
        ],
    }
    if rec["gaps"]["tied_param_pairs"]:
        print(
            f"  gap: value-tied params "
            f"{[(t['a'], t['b']) for t in rec['gaps']['tied_param_pairs']]}",
            flush=True,
        )

    # -- B: the guarded default stack -----------------------------------
    rest_flag = 1 if name in _rest_models(args) else 0
    want_window = bool(cand.get("window_compose", {}).get("matches"))
    # With no window match the "all laws" set is identical to the
    # delivery set — a guarded subprocess run would just duplicate it.
    if want_window and str(_opt(args, "window_attempt", "1")) == "1":
        print(
            "  guarded default stack (subprocess, all laws"
            f"{', rest' if rest_flag else ''})…",
            flush=True,
        )
        rec["default_stack"] = _guarded_search(
            ckpt,
            seq,
            str(dev),
            "all",
            rest_flag,
            float(_opt(args, "search_timeout", 300.0)),
            float(_opt(args, "mem_gb", 8.0)),
        )
        print(
            f"    -> {rec['default_stack']['outcome']} in "
            f"{rec['default_stack'].get('seconds')}s "
            f"(died_at={rec['default_stack'].get('died_at')})",
            flush=True,
        )
    else:
        rec["default_stack"] = {
            "outcome": (
                "skipped — no window match"
                if not want_window
                else "skipped — --window-attempt 0"
            )
        }

    # -- C: delivery run ------------------------------------------------
    # The default stack includes WindowCompose.  The in-process run
    # keeps it only when the *guarded* run proved it safe — i.e. a
    # window match actually grafted there.  A declined window under
    # the subprocess memory cap must not be retried uncapped here.
    ds = rec["default_stack"]
    ds_stats = ds.get("stats") or {}
    window_grafted = bool(
        (ds_stats.get("morphism_fires") or {}).get("window_compose")
    )
    window_ok = ds.get("outcome") == "completed" and (
        window_grafted or not want_window
    )
    laws_delivery = _law_set("all" if window_ok else "no_window")
    rec["delivery"] = {
        "laws": [law.name for law in laws_delivery],
        "optimize_rest": bool(rest_flag),
        "window_excluded": not window_ok,
        "window_exclusion_reason": (
            None
            if window_ok
            else (
                f"guarded run: {ds.get('outcome')}"
                if ds.get("outcome") != "completed"
                else "window match declined under the guarded run"
            )
        ),
    }
    t2 = time.time()
    try:
        opt_mod, stats = _delivery_run(
            model, idx, laws_delivery, bool(rest_flag)
        )
        rec["delivery"]["seconds"] = round(time.time() - t2, 1)
        rec["delivery"]["n_rewritten"] = stats.get("n_rewritten")
        rec["delivery"]["morphism_fires"] = stats.get("morphism_fires")
        rec["delivery"]["end_to_end"] = stats.get("end_to_end")
        rec["delivery"]["in_place"] = stats.get("in_place")
        rec["delivery"].update(_match_ledger(stats))
        rec["delivery"]["blocks"] = _block_ledger(stats)
        fires = stats.get("morphism_fires") or {}
        print(
            f"  delivery ({'all laws' if window_ok else 'window excluded'}"
            f"{', rest' if rest_flag else ''}): "
            f"{rec['delivery']['seconds']}s "
            f"rewritten={stats.get('n_rewritten')} fires={fires} "
            f"e2e={stats.get('end_to_end')}",
            flush=True,
        )
    except Exception as e:
        rec["delivery"]["error"] = f"{type(e).__name__}: {e}"
        opt_mod = None
        print(f"  delivery FAILED: {e}", flush=True)

    # -- verify on a FRESH input ---------------------------------------
    if opt_mod is not None:
        g2 = torch.Generator().manual_seed(123)
        idx2 = torch.randint(
            0, cfg["vocab"], (1, seq), generator=g2
        ).to(dev)
        try:
            with torch.no_grad():
                vr = verify_equiv(model(idx2), opt_mod(idx2))
            rec["verify_fresh"] = {
                "passed": bool(vr.passed),
                "max_rel": vr.max_rel,
            }
            print(
                f"  fresh-input verify: passed={vr.passed} "
                f"rel={vr.max_rel:.3e}",
                flush=True,
            )
            if not vr.passed:
                opt_mod = None
        except Exception as e:
            rec["verify_fresh"] = {"error": f"{type(e).__name__}: {e}"}
            opt_mod = None

    # -- timing ----------------------------------------------------------
    variants = [Variant("eager", _fwd_stmt(model, (idx,)))]
    if opt_mod is not None:
        variants.append(Variant("morphism", _fwd_stmt(opt_mod, (idx,))))
    if str(_opt(args, "inductor", "0")) not in ("0", "None"):
        ct = float(_opt(args, "compile_timeout", 45.0))
        cm, st_ = try_compile(copy.deepcopy(model), (idx,), ct)
        rec["inductor_status"] = st_
        if cm is not None:
            try:
                with torch.no_grad():
                    vr = verify_equiv(model(idx), cm(idx))
                if vr.passed:
                    variants.append(
                        Variant("inductor", _fwd_stmt(cm, (idx,)))
                    )
                rec["inductor_rel"] = vr.max_rel
            except Exception as e:
                rec["inductor_status"] = (
                    f"verify failed: {type(e).__name__}"
                )
        print(f"  inductor: {rec['inductor_status']}", flush=True)
    case = Case(
        name=f"{name}@T{seq}",
        params={"model": name, "seq": seq},
        variants=variants,
        aux={},
    )
    cell = runner.run_case(case)
    rec["timing"] = {
        n: round(cell.medians[n] * 1e3, 3) for n in cell.medians
    }
    if "eager" in cell.medians and "morphism" in cell.medians:
        rec["timing"]["speedup"] = round(
            cell.medians["eager"] / cell.medians["morphism"], 4
        )
    print(
        "  times: "
        + "  ".join(
            f"{n}={cell.medians[n] * 1e3:.3f}ms" for n in cell.medians
        ),
        flush=True,
    )
    cell.aux = rec
    return {"rec": rec, "cell": cell}


def _rest_models(args) -> set[str]:
    spec = getattr(args, "rest_models", None)
    if spec is None:
        return {"stories15M"}
    spec = str(spec).strip().lower()
    if spec == "all":
        return set(_CKPTS)
    if spec in ("none", ""):
        return set()
    return {s.strip() for s in spec.split(",") if s.strip()}


# ---------------------------------------------------------------------------
#  Verdicts
# ---------------------------------------------------------------------------


def _findings(recs: list[dict]) -> list[Finding]:
    """The honest verdicts — lift coverage, law traction, gaps."""
    out: list[Finding] = []
    for rec in recs:
        name = rec["model"]
        lift = rec["coverage"]["lift"]
        n_l, n_n = lift["n_lifted"], lift["n_nodes"]
        out.append(
            Finding(
                claim=f"{name}: composer blocks lift to morphism"
                " signatures (incl. multi-input (h, cos, sin)"
                " attention blocks)",
                verdict=(
                    Verdict.WIN
                    if n_l == n_n
                    else Verdict.PARITY
                    if n_l
                    else Verdict.NEGATIVE
                ),
                headline=f"{n_l}/{n_n} blocks lifted",
                metric="lift_fraction",
                value=n_l / n_n if n_n else None,
                evidence={
                    "opaque": {
                        k: v["note"]
                        for k, v in lift["blocks"].items()
                        if v["opaque"]
                    },
                    "input_kinds": lift["input_kinds"],
                    "wire_kinds": lift["wire_kinds"],
                },
            )
        )
        cand = rec["coverage"]["candidacy"]
        dl = rec.get("delivery") or {}
        per_law = dl.get("per_law") or {}
        for law, c in cand.items():
            matched = c.get("matches", 0)
            pl = per_law.get(law, {})
            grafted = pl.get("grafted", 0)
            declined = pl.get("declined", 0)
            if law == "window_compose" and dl.get("window_excluded"):
                ds = rec.get("default_stack") or {}
                ds_stats = ds.get("stats") or {}
                wrows = {
                    k: v
                    for k, v in (ds_stats.get("matches") or {}).items()
                    if k.startswith("window_compose:")
                }
                out.append(
                    Finding(
                        claim=f"{name}: window_compose — "
                        f"{matched} window match(es) on the real chain",
                        verdict=Verdict.NEGATIVE,
                        headline=(
                            f"matched {matched}, grafted 0 — "
                            f"guarded default stack: "
                            f"{ds.get('outcome')} "
                            + (f"({wrows})" if wrows else "")
                        ),
                        metric="grafted",
                        value=0.0,
                        evidence={
                            "sites": c.get("sites"),
                            "default_stack": {
                                k: ds.get(k)
                                for k in (
                                    "outcome",
                                    "seconds",
                                    "died_at",
                                    "exit",
                                )
                            },
                        },
                    )
                )
                continue
            verdict = (
                Verdict.WIN
                if grafted
                else (
                    Verdict.NEGATIVE
                    if matched and declined >= matched
                    else Verdict.INCONCLUSIVE
                )
            )
            reasons = pl.get("reasons") or {}
            out.append(
                Finding(
                    claim=(
                        f"{name}: {law} — signature matches on the "
                        "real graph"
                    ),
                    verdict=verdict,
                    headline=(
                        f"{matched} candidates -> {grafted} grafted"
                        + (
                            f" ({', '.join(f'{r}×{n}' for r, n in reasons.items())})"
                            if reasons
                            else ""
                        )
                    ),
                    metric="grafted",
                    value=float(grafted),
                    evidence={
                        "sites": c.get("sites"),
                        "declined": declined,
                        "skipped": pl.get("skipped", 0),
                        "reasons": reasons,
                    },
                )
            )
        # -- honest gaps -------------------------------------------------
        gaps = rec.get("gaps") or {}
        ties = gaps.get("tied_param_pairs") or []
        if ties:
            tie_sites = cand.get("weight_tie", {}).get("sites") or []
            seen = {n for site in tie_sites for n in site}
            invisible = [
                t
                for t in ties
                if not (
                    t["a"].rsplit(".", 1)[0] in seen
                    or t["b"].rsplit(".", 1)[0] in seen
                )
            ]
            out.append(
                Finding(
                    claim=(
                        f"{name}: value-tied parameters reach "
                        "weight_tie candidacy"
                    ),
                    verdict=(
                        Verdict.NEGATIVE if invisible else Verdict.WIN
                    ),
                    headline=(
                        f"{len(ties)} tied pair(s) in the module; "
                        f"{len(invisible)} invisible to signature "
                        "level"
                    ),
                    metric="invisible_ties",
                    value=float(len(invisible)),
                    evidence={
                        "tied": ties,
                        "weight_tie_sites": sorted(seen),
                    },
                )
            )
        norms = gaps.get("norm_ops_vs_sig") or {}
        blind = {
            k: v
            for k, v in norms.items()
            if v.get("norm_ops") and v.get("sig_norm") == "none"
        }
        if blind:
            out.append(
                Finding(
                    claim=(
                        f"{name}: blocks' norm ops land in NormSig "
                        "(norm_cascade candidacy)"
                    ),
                    verdict=Verdict.NEGATIVE,
                    headline=(
                        f"{len(blind)} lifted block(s) carry norm ops "
                        "the signature reads as kind=none"
                    ),
                    metric="norm_blind_blocks",
                    value=float(len(blind)),
                    evidence={"blocks": blind},
                )
            )
        ow = gaps.get("opaque_wires") or []
        if ow:
            out.append(
                Finding(
                    claim=(
                        f"{name}: every adjacent block boundary "
                        "classifies"
                    ),
                    verdict=Verdict.PARITY,
                    headline=(
                        f"{len(ow)} opaque edge(s) — the lifted "
                        "graph's reach stops at them"
                    ),
                    metric="opaque_wires",
                    value=float(len(ow)),
                    evidence={"wires": ow},
                )
            )
        # E2E row
        tm = rec.get("timing") or {}
        vf = rec.get("verify_fresh") or {}
        dsecs = dl.get("seconds")
        speedup = tm.get("speedup")
        out.append(
            Finding(
                claim=(
                    f"{name}: optimize_morphisms delivers a verified"
                    " module and its wall-time delta vs eager"
                ),
                verdict=(
                    Verdict.WIN
                    if (speedup or 0) > 1.05
                    else (
                        Verdict.REGRESSION
                        if speedup and speedup < 0.95
                        else Verdict.PARITY
                    )
                ),
                headline=(
                    f"rewritten={dl.get('n_rewritten')} "
                    f"rest_optimized={sum(1 for b in (dl.get('blocks') or {}).values() if b.get('status') == 'optimized')} "
                    f"fresh_rel={vf.get('max_rel')} "
                    f"{f'{speedup:.3f}×' if speedup else 'untimed'}"
                ),
                metric="speedup_vs_eager",
                value=speedup,
                evidence={
                    "opt_s": dsecs,
                    "end_to_end": dl.get("end_to_end"),
                    "timing_ms": tm,
                },
            )
        )
    return out


# ---------------------------------------------------------------------------
#  Harness entry point (registry/CLI convention)
# ---------------------------------------------------------------------------


def run_bench(args) -> Report:
    dev = torch.device(getattr(args, "device", None) or "cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        print("  --device cuda but CUDA is unavailable; using cpu")
        dev = torch.device("cpu")
    # Timing floor: the real checkpoints are 25-170 ms/call, so a
    # 0.2 s autorange window measures ~1 sample and reports noise as
    # "regressions".  0.8 s is the smallest window worth reporting.
    min_run_time = max(float(_opt(args, "min_run_time", 0.8)), 0.8)
    warmup = int(_opt(args, "warmup", 3))
    seq = int(_opt(args, "seq", 64))

    models = [
        s.strip()
        for s in (_opt(args, "models", None) or ",".join(_CKPTS)).split(
            ","
        )
        if s.strip()
    ]

    # Resolve checkpoints — stories15M hard-fails, others skip.
    resolved: dict[str, str] = {}
    for name in models:
        fname = _CKPTS.get(name)
        if fname is None:
            print(f"  unknown model {name!r} — skipped", flush=True)
            continue
        if name == "stories15M" or _have_ckpt(fname):
            resolved[name] = resolve_ckpt(
                _opt(args, "ckpt", None)
                if name == "stories15M"
                else None,
                name=fname,
            )
        else:
            print(
                f"  {name}: {fname} not in {cache_dir()} or /tmp "
                "— skipped",
                flush=True,
            )

    print(
        f"morphism_coverage — device={dev} models={list(resolved)} "
        f"seq={seq}",
        flush=True,
    )
    t0 = time.perf_counter()
    runner = Runner(
        device=dev, warmup=max(warmup, 2), min_run_time=min_run_time
    )

    recs: list[dict] = []
    cells: list = []
    for name, ckpt in resolved.items():
        out = run_model(name, ckpt, args, dev, runner)
        recs.append(out["rec"])
        cells.append(out["cell"])
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    findings = _findings(recs)

    # -- console summary -------------------------------------------------
    print("\n=== coverage summary ===")
    for rec in recs:
        lift = rec["coverage"]["lift"]
        print(
            f"{rec['model']}: lifted {lift['n_lifted']}/"
            f"{lift['n_nodes']} — inputs {lift['input_kinds']} "
            f"wires {lift['wire_kinds']}"
        )
        for law, c in rec["coverage"]["candidacy"].items():
            pl = (rec.get("delivery", {}).get("per_law") or {}).get(
                law, {}
            )
            print(
                f"    {law:18} cand={c.get('matches', '?'):>3} "
                f"grafted={pl.get('grafted', 0)} "
                f"declined={pl.get('declined', 0)} "
                f"{pl.get('reasons') or ''}"
            )
        ds = rec.get("default_stack") or {}
        print(f"    default stack: {ds.get('outcome')}")
        tm = rec.get("timing") or {}
        print(f"    timing ms: {tm}")
    print(
        f"  total wall time {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    report = Report(
        suite="morphism_coverage",
        cells=cells,
        env=collect_env(dev),
        title="Morphism-layer coverage on real checkpoints",
        summary=(
            "How much of a real transformer the morphism engine can "
            "see: block-lift rates, per-law match/fire/graft ledgers "
            "on llama2.c stories15M/stories110M, guarded default-stack"
            " outcome, delivered-module verify + latency vs eager."
        ),
        findings=findings,
        provenance={
            "checkpoints": {r["model"]: r["ckpt"] for r in recs},
            "coverage": {r["model"]: r for r in recs},
        },
    )
    if not getattr(args, "no_artifacts", False):
        out_dir = Path(_opt(args, "out", None) or "bench/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "morphism_coverage.json"
        md_path = out_dir / "morphism_coverage.md"
        report.to_json(json_path)
        report.to_markdown(md_path, speedup_vs="eager")
        print(f"  artifacts → {json_path}")
        print(f"            → {md_path}")
    return report


# ---------------------------------------------------------------------------
#  CLI — suite entry + the hidden guarded-worker mode
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "morphism coverage on real checkpoints: lift/fire/graft "
            "rates per morphism law on llama2.c stories models."
        )
    )
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--models",
        default=None,
        help="comma subset of "
        + ",".join(_CKPTS)
        + " (default: all checkpoints found)",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="stories15M checkpoint override (see resolve_ckpt)",
    )
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument(
        "--rest-models",
        default=None,
        help="models that also run the per-block fallback "
        "(default: stories15M; 'all' or 'none' spell out)",
    )
    ap.add_argument(
        "--window-attempt",
        default="1",
        help="run the guarded all-laws subprocess (0 skips)",
    )
    ap.add_argument(
        "--search-timeout",
        type=float,
        default=300.0,
        help="wall-clock cap on the guarded default-stack subprocess",
    )
    ap.add_argument(
        "--mem-gb",
        type=float,
        default=8.0,
        help="RLIMIT_AS cap for the guarded subprocess (0 disables)",
    )
    ap.add_argument(
        "--inductor", default="0", help="1 adds a torch.compile variant"
    )
    ap.add_argument("--compile-timeout", type=float, default=45.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--min-run-time", type=float, default=0.2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="bench/results")
    ap.add_argument("--no-artifacts", action="store_true")
    # -- worker mode (hidden; used by _guarded_search) ------------------
    ap.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    ap.add_argument("--laws", default="all", help=argparse.SUPPRESS)
    ap.add_argument(
        "--rest", type=int, default=0, help=argparse.SUPPRESS
    )
    ap.add_argument("--stats", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--progress", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        sys.exit(_worker_main(args))
    if args.quick:
        for k, v in QUICK.items():
            cur = getattr(args, k)
            setattr(args, k, str(v) if cur is None else type(cur)(v))
    run_bench(args)


if __name__ == "__main__":
    main()
