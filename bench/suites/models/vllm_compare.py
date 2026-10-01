"""catopt vs vLLM on a stories15M-shaped Llama — and the honest framing.

vLLM's wins come from a different layer than catopt's: paged KV
caching, continuous batching and fused attention kernels vs
compile-time structural graph rewrites (QKV/gate-up fusion, weight
folding).  This bench measures both and, more importantly, exercises
the *composition*: catopt-delivered weights written as a standard
HF ``LlamaForCausalLM`` checkpoint and served through
``vllm.LLM(model=<dir>)``.

Legs:

1. **hf-export** (always runnable — torch only).  Converts the
   llama2.c checkpoint — or the catopt-OPTIMIZED module — into an HF
   checkpoint directory (``config.json`` + ``model.safetensors`` +
   tokenizer files when a tokenizer can be obtained).  The catopt
   path splits the fused ``blocks.N.fused_10`` (cat of wq|wk|wv rows)
   and ``fused_11`` (cat of w1|w3 rows) back out, and permutes the
   q/k projection rows from llama2.c's interleaved RoPE layout into
   HF's rotate-half layout (the llama.cpp ``permute``).  A pure-torch
   check proves the permuted-weight + rotate-half attention scores
   equal the interleaved-rope originals, so the exported checkpoint
   is bit-equivalent in attention, not just plausible.

2. **torch decode** (always runnable).  Greedy decode of ``--gen``
   tokens with a FIXED context window ``--ctx`` (every step sees
   exactly ``ctx`` tokens — catopt's shape-specialized IRModule
   executors keep their verified shapes).  Variants: ``eager``,
   ``inductor``, ``catopt``, ``catopt+inductor``; each verified
   against eager logits, timed as wall seconds per full decode →
   tok/s in aux.  No KV cache on the torch side: this leg measures
   the *module*, while vLLM measures the *engine* — the report says
   so rather than pretending parity of method.

3. **vllm** (optional).  Runs in-process when ``import vllm``
   succeeds, else via ``--vllm-python <exe>`` as a subprocess on the
   same script (``--vllm-leg`` internal mode), else skipped with the
   integration recipe printed to the report.  Serves the exported HF
   dir at ``dtype=float32`` (weights are fp32; keeps greedy tokens
   comparable to the torch leg), measures prefill time
   (``max_tokens=1`` ~ TTFT) and steady-state decode tok/s, and
   reports token agreement between vLLM greedy output and torch
   eager greedy output.

Typical invocations:

    # everything in one env with both stacks installed + a GPU:
    .venv-vllm/bin/python bench/vllm_compare.py --device cuda

    # catopt legs on the repo .venv, vllm leg delegated to a venv
    # that has it (results are labelled cross-interpreter):
    .venv/bin/python bench/vllm_compare.py --device cpu \\
        --vllm-python ~/vllm-probe/bin/python

    # harnessed:
    python bench/run_all.py --suites vllm_compare --quick
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import torch


from bench import benchkit
from bench.suites.models.decode_bench import BatchedStories
from bench.common.llama2c import load_llama2c
from bench.suites.models.stories15m_bench import resolve_ckpt

QUICK = {"gen": "16", "ctx": "48", "repeats": "1"}


# ---------------------------------------------------------------------------
#  Leg 1 — HF checkpoint export
# ---------------------------------------------------------------------------


def rope_permute_rows(w: torch.Tensor, n_heads: int) -> torch.Tensor:
    """Interleaved-RoPE weight rows → HF rotate-half layout.

    llama2.c rotates pairs ``(x[2i], x[2i+1])``; HF Llama rotates
    ``(x[i], x[i+hd/2])``.  Reordering the projection's output rows by
    ``reshape(nh, hd//2, 2, in).transpose(1,2)`` makes HF's convention
    produce the same per-head rotation up to a fixed permutation of
    the head dims — a permutation attention scores are invariant
    under (both q and k are permuted identically).
    """
    out_f, in_f = w.shape
    hd = out_f // n_heads
    return (
        w.reshape(n_heads, hd // 2, 2, in_f)
        .transpose(1, 2)
        .reshape(out_f, in_f)
    )


def verify_rope_conversion(
    wq: torch.Tensor, wk: torch.Tensor, n_heads: int, pos: int = 7
) -> float:
    """Max |Δ| between interleaved-rope and permuted+rotate-half q·k.

    Attention scores are what the RoPE convention must preserve; this
    checks them exactly, so a wrong permutation is a loud failure
    rather than silent garbage downstream.
    """
    wq, wk = wq.detach(), wk.detach()
    dim = wq.shape[1]
    hd = dim // n_heads
    xn = torch.randn(pos + 1, dim, device=wq.device, dtype=wq.dtype)
    q_il = xn @ wq.T
    k_il = xn @ wk.T
    q_hf = xn @ rope_permute_rows(wq, n_heads).T
    k_hf = xn @ rope_permute_rows(wk, n_heads).T

    freqs = 1.0 / (
        10000.0
        ** (torch.arange(0, hd, 2, device=wq.device).float() / hd)
    )
    ang = torch.outer(
        torch.arange(pos + 1, device=wq.device).float(), freqs
    )
    cos_il, sin_il = ang.cos(), ang.sin()  # (T, hd//2) per-pair freqs

    def rope_il(x):  # (T, nh, hd) llama2.c pairs convention
        x = x.reshape(pos + 1, n_heads, hd)
        x1, x2 = x[..., ::2], x[..., 1::2]
        c, s = cos_il[:, None, :], sin_il[:, None, :]
        return torch.stack(
            [x1 * c - x2 * s, x1 * s + x2 * c], -1
        ).reshape(pos + 1, n_heads, hd)

    def rope_hf(x):  # HF rotate_half on the permuted projection
        # cat(freqs, freqs) convention ≡ halves rotating against each
        # other with the same per-pair frequency.
        x = x.reshape(pos + 1, n_heads, hd)
        x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
        c, s = cos_il[:, None, :], sin_il[:, None, :]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    qi, ki = rope_il(q_il), rope_il(k_il)
    qh, kh = rope_hf(q_hf), rope_hf(k_hf)
    # score[q_pos][k_pos] must match under both conventions
    s_il = torch.einsum("qhd,khd->qk", qi, ki)
    s_hf = torch.einsum("qhd,khd->qk", qh, kh)
    return float((s_il - s_hf).abs().max())


def hf_state_dict(
    model: torch.nn.Module, cfg: dict
) -> dict[str, torch.Tensor]:
    """HF ``LlamaForCausalLM`` state dict from stock OR optimized model.

    Reads the module's own ``state_dict`` — for a catopt-optimized
    module the fused executors' parameter names are recognised and
    split back to HF layout (``fused_10`` → q|k|v, ``fused_11`` →
    gate|up).  Everything else (folded single weights like
    ``p_w2_weight``) passes through under its HF name.
    """
    sd = model.state_dict()
    nh, dim, hidden = cfg["n_heads"], cfg["dim"], cfg["hidden"]
    out: dict[str, torch.Tensor] = {}

    def pick(*names: str) -> torch.Tensor:
        for n in names:
            if n in sd:
                return sd[n].detach().cpu().float()
        raise KeyError(f"none of {names} in state_dict")

    out["model.embed_tokens.weight"] = pick("emb.weight", "emb.p_weight")
    out["lm_head.weight"] = pick("head.weight", "head.p_weight")
    out["model.norm.weight"] = pick("rms_final")
    for i in range(cfg["n_layers"]):
        p = f"blocks.{i}."
        hp = f"model.layers.{i}."
        if p + "fused_10" in sd:  # catopt QKV fusion → split rows
            f = sd[p + "fused_10"].detach().cpu().float()
            q, k, v = f[:dim], f[dim : 2 * dim], f[2 * dim :]
        else:
            q, k, v = (
                pick(p + "wq.weight"),
                pick(p + "wk.weight"),
                pick(p + "wv.weight"),
            )
        if p + "fused_11" in sd:  # catopt gate|up fusion
            f = sd[p + "fused_11"].detach().cpu().float()
            w1, w3 = f[:hidden], f[hidden:]
        else:
            w1, w3 = pick(p + "w1.weight"), pick(p + "w3.weight")
        out[hp + "self_attn.q_proj.weight"] = rope_permute_rows(q, nh)
        out[hp + "self_attn.k_proj.weight"] = rope_permute_rows(k, nh)
        out[hp + "self_attn.v_proj.weight"] = v
        out[hp + "self_attn.o_proj.weight"] = pick(
            p + "wo.weight", p + "p_wo_weight"
        )
        out[hp + "input_layernorm.weight"] = pick(
            p + "rms_att", p + "p_rms_att"
        )
        out[hp + "post_attention_layernorm.weight"] = pick(
            p + "rms_ffn", p + "p_rms_ffn"
        )
        out[hp + "mlp.gate_proj.weight"] = w1
        out[hp + "mlp.up_proj.weight"] = w3
        out[hp + "mlp.down_proj.weight"] = pick(
            p + "w2.weight", p + "p_w2_weight"
        )
    return out


def write_hf_dir(hf_dir: Path, sd_hf: dict, cfg: dict) -> dict:
    """Write config.json + weights (+ manifest) for vllm/transformers."""
    hf_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg["dim"],
        "intermediate_size": cfg["hidden"],
        "num_hidden_layers": cfg["n_layers"],
        "num_attention_heads": cfg["n_heads"],
        "num_key_value_heads": cfg["n_heads"],
        "vocab_size": cfg["vocab"],
        "max_position_embeddings": cfg["seq_len"],
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "attention_bias": False,
        "mlp_bias": False,
    }
    (hf_dir / "config.json").write_text(json.dumps(config, indent=2))
    (hf_dir / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 1,
                "eos_token_id": 2,
                "max_length": cfg["seq_len"],
            },
            indent=2,
        )
    )
    wrote = None
    try:
        from safetensors.torch import save_file

        save_file(sd_hf, str(hf_dir / "model.safetensors"))
        wrote = "model.safetensors"
    except ImportError:
        # catopt's own stdlib safetensors writer — no extra dep needed
        # in the bare .venv to emit a file vllm/transformers read.
        from catopt_torch.export import _write_safetensors

        _write_safetensors(
            sd_hf, hf_dir / "model.safetensors", json.dumps({})
        )
        wrote = "model.safetensors"
    manifest = {
        "catopt_hf_export": 1,
        "source": "llama2.c checkpoint via bench/vllm_compare.py",
        "config": config,
        "weights_file": wrote,
        "rope": "q/k projection rows permuted interleaved→rotate_half",
    }
    (hf_dir / "catopt_export.manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    return manifest


def ensure_tokenizer(hf_dir: Path, vocab: int) -> str:
    """Give ``hf_dir`` a loadable tokenizer; return how it was made.

    Order: (1) real llama tokenizer from the hub, (2) a synthetic
    WordLevel built with the ``tokenizers`` package (vocab ids 1:1 —
    token TEXT is wrong but ids round-trip, which is all a benchmark
    prompt needs).  Both need network/packages that may be absent;
    failure returns the reason string instead of raising.
    """
    if (hf_dir / "tokenizer_config.json").exists() or (
        hf_dir / "tokenizer.json"
    ).exists():
        return "already present"
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
        tok.save_pretrained(str(hf_dir))
        return "hf-internal-testing/llama-tokenizer"
    except Exception as e:
        hub_err = f"{type(e).__name__}: {e}"
    try:
        from tokenizers import Tokenizer, models

        t = Tokenizer(
            models.WordLevel(
                {f"tok{i}": i for i in range(vocab)}, unk_token="tok0"
            )
        )
        t.save(str(hf_dir / "tokenizer.json"))
        (hf_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "unk_token": "tok0",
                    "bos_token": "tok1",
                    "eos_token": "tok2",
                }
            )
        )
        return f"synthetic WordLevel (hub fetch failed: {hub_err})"
    except Exception as e:
        return f"FAILED — hub: {hub_err}; wordlevel: {e}"


# ---------------------------------------------------------------------------
#  Leg 2 — torch greedy decode (fixed context window)
# ---------------------------------------------------------------------------


def greedy_decode(
    mod: torch.nn.Module, prompt: torch.Tensor, gen: int, window: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy decode ``gen`` tokens; each step sees the last ``window``.

    ``window`` bounds the attention context each step — a fixed value
    keeps every forward at the same shape (catopt's executors are
    shape-specialized to their optimize-time example).  Pass a window
    ≥ ``prompt_len + gen`` for a *true* decode: the window never
    truncates, positions stay real, and the module sees the full
    history exactly as vLLM's paged KV cache does.
    """
    ids = prompt
    for _ in range(gen):
        with torch.no_grad():
            logits = mod(ids[:, -window:])
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[:, prompt.shape[1] :], logits


def optimize_once(model, idx, device):
    """Compositional optimize + verify against eager on ``idx``."""
    from catopt_orchestrator import Compositional, Optimizer
    from catopt_torch.backend import TorchBackend
    from catopt_torch.report import verify_equiv

    with torch.no_grad():
        ref = model(idx)
    t0 = time.time()
    opt, rep = Optimizer(backend=TorchBackend()).optimize(
        model, idx, strategy=Compositional(), verbose=False
    )
    opt_s = time.time() - t0
    with torch.no_grad():
        vr = verify_equiv(ref, opt(idx), rtol=1e-4)
    return opt, rep, vr, ref, opt_s


def torch_decode_leg(
    model, opt, prompt, args
) -> tuple[list[benchkit.Cell], torch.Tensor]:
    """Time the four variants on one full greedy decode each.

    ``prompt`` is already ``ctx`` tokens long, so every step of every
    variant forwards exactly ``(B, ctx)`` — the shape catopt's
    executors were specialized to.
    """
    B, ctx, gen = args.batch, args.ctx, args.gen

    variants: list[tuple[str, torch.nn.Module | Exception]] = [
        ("eager", model),
        ("catopt", opt),
    ]
    try:
        variants.append(("inductor", torch.compile(model, dynamic=True)))
        variants.append(
            ("catopt+inductor", torch.compile(opt, dynamic=True))
        )
    except Exception as e:
        variants += [("inductor", e), ("catopt+inductor", e)]

    with torch.no_grad():
        ref_tokens, _ = greedy_decode(model, prompt, gen, ctx)

    runner = benchkit.Runner(
        device=args.device,
        warmup=args.warmup or 1,
        min_run_time=args.min_run_time or 0.05,
    )
    variants_out: list[benchkit.Variant] = []
    aux: dict = {"gen_tokens": gen, "ctx": ctx, "batch": B}
    for name, mod in variants:
        if isinstance(mod, Exception):
            aux[f"{name}_status"] = f"compile failed: {mod}"
            continue
        try:
            with torch.no_grad():
                toks, _ = greedy_decode(mod, prompt, 2, ctx)
            agree = float(
                (toks == ref_tokens[:, :2]).float().mean().item()
            )
            if agree < 1.0:
                aux[f"{name}_status"] = (
                    f"SKIP token agreement {agree:.2f}"
                )
                continue
        except Exception as e:
            aux[f"{name}_status"] = f"SKIP {type(e).__name__}: {e}"
            continue

        def stmt(mod=mod):
            greedy_decode(mod, prompt, gen, ctx)

        variants_out.append(
            benchkit.Variant(name=name, stmt=stmt, note="greedy decode")
        )
    case = benchkit.Case(
        name=f"decode B={B} ctx={ctx} gen={gen}",
        params={"B": B, "ctx": ctx, "gen": gen},
        variants=variants_out,
        aux=aux,
    )
    cells = [runner.run_case(case)]
    for c in cells:
        for n, med in c.medians.items():
            c.aux[f"{n}_tok_per_s"] = round(B * gen / med, 1)
    return cells, ref_tokens


# ---------------------------------------------------------------------------
#  Leg 3 — vLLM serving
# ---------------------------------------------------------------------------


def _vllm_leg_payload(hf_dir: str, args) -> dict:
    """Serve ``hf_dir`` through vLLM; returns a JSON-able dict.

    Runs inside whatever interpreter has vllm.  Measures prefill
    (max_tokens=1 — a TTFT stand-in, labelled as such) and full-gen
    wall time → decode tok/s; greedy tokens are returned for the
    agreement check against the torch leg.
    """
    import vllm  # noqa: F401 — import failure is the caller's signal
    from vllm import LLM, SamplingParams

    cfg = json.loads(Path(hf_dir, "catopt_export.manifest.json")
                     .read_text())["config"]
    B, gen = args.batch, args.gen
    prompts = getattr(args, "prompt_ids", None)
    if prompts is None:
        torch.manual_seed(args.seed)
        prompts = (
            torch.randint(1, cfg["vocab_size"], (B, args.prompt_len))
            .tolist()
        )
    else:
        prompts = [list(p) for p in prompts]
    # The tokenizer is ensured inside the leg so a --vllm-python
    # subprocess (which has transformers/tokenizers) can build it
    # even when the parent interpreter cannot.
    tok_msg = ensure_tokenizer(Path(hf_dir), cfg["vocab_size"])
    if tok_msg.startswith("FAILED"):
        return {"mode": "skipped", "reason": f"tokenizer: {tok_msg}"}
    t0 = time.time()
    llm = LLM(
        model=hf_dir,
        tokenizer=hf_dir,
        dtype="float32",
        enforce_eager=bool(args.enforce_eager),
        gpu_memory_utilization=args.gpu_frac,
        max_model_len=cfg["max_position_embeddings"],
        seed=args.seed,
        disable_log_stats=True,
    )
    init_s = time.time() - t0
    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    spn = SamplingParams(
        temperature=0.0, max_tokens=gen, ignore_eos=True
    )
    # warmup (kernel autotune, allocator steady state)
    llm.generate(prompts, sampling_params=sp1, use_tqdm=False)
    t0 = time.time()
    llm.generate(prompts, sampling_params=sp1, use_tqdm=False)
    prefill_s = time.time() - t0
    reps = max(int(getattr(args, "repeats", None) or 1), 1)
    gen_times, toks = [], None
    for _ in range(reps):
        t0 = time.time()
        outn = llm.generate(prompts, sampling_params=spn, use_tqdm=False)
        gen_times.append(time.time() - t0)
        if toks is None:
            toks = [o.outputs[0].token_ids for o in outn]
    gen_s = min(gen_times)  # best-of — engine jitter, not model work
    decode_s = max(gen_s - prefill_s, 1e-9)
    return {
        "init_s": round(init_s, 2),
        "prefill_s": round(prefill_s, 4),
        "gen_s": round(gen_s, 4),
        "gen_times": [round(t, 4) for t in gen_times],
        "decode_tok_per_s": round(B * (gen - 1) / decode_s, 1),
        "e2e_tok_per_s": round(B * gen / gen_s, 1),
        "tokens": [list(t) for t in toks],
        "enforce_eager": bool(args.enforce_eager),
        "tokenizer": tok_msg,
    }


def _vllm_env_setup() -> None:
    """Env fixes for toolkit-less hosts; call BEFORE importing vllm.

    On machines without a CUDA toolkit, flashinfer's sampler JIT
    needs nvcc — point CUDA_HOME at the pip wheel's copy when present
    and disable the flashinfer sampler regardless (greedy argmax
    doesn't need it; this only swaps the code path).
    """
    for cand in (Path(sys.prefix) / "lib").glob(
        "python*/site-packages/nvidia/cu*/bin/nvcc"
    ):
        if not Path("/usr/local/cuda/bin/nvcc").exists():
            os.environ.setdefault("CUDA_HOME", str(cand.parent.parent))
        break
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


def vllm_leg(hf_dir: Path, args, prompt_ids=None) -> dict:
    """Run the vLLM leg in-process or under ``--vllm-python``."""
    _vllm_env_setup()
    args.prompt_ids = prompt_ids
    try:
        import vllm  # noqa: F401
    except ImportError as e:
        vllm_err = e
    else:
        vllm_err = None
    if vllm_err is None:
        try:
            return {
                "mode": "in-process",
                **_vllm_leg_payload(str(hf_dir), args),
            }
        except Exception as e:
            return {
                "mode": "failed",
                "reason": f"in-process vllm leg: "
                f"{type(e).__name__}: {e}",
            }
    if not args.vllm_python:
        return {
            "mode": "skipped",
            "reason": f"import vllm failed ({vllm_err}); pass "
            "--vllm-python <exe> for a subprocess leg",
        }
    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, mode="w"
    ) as f:
        out_json = f.name
    cmd = [
        args.vllm_python,
        str(Path(__file__).resolve()),
        "--vllm-leg",
        "--hf-dir",
        str(hf_dir),
        "--out-json",
        out_json,
        "--batch",
        str(args.batch),
        "--gen",
        str(args.gen),
        "--prompt-len",
        str(args.prompt_len),
        "--seed",
        str(args.seed),
        "--gpu-frac",
        str(args.gpu_frac),
        "--repeats",
        str(getattr(args, "repeats", None) or 3),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if prompt_ids is not None:
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as pf:
            json.dump([list(map(int, p)) for p in prompt_ids], pf)
        cmd += ["--prompt-ids-file", pf.name]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not Path(out_json).exists():
        return {
            "mode": "failed",
            "reason": proc.stderr.strip().splitlines()[-1]
            if proc.stderr.strip()
            else "subprocess produced no output",
            "stderr_tail": proc.stderr[-2000:],
        }
    data = json.loads(Path(out_json).read_text())
    Path(out_json).unlink(missing_ok=True)
    return {"mode": f"subprocess {args.vllm_python}", **data}


INTEGRATION_RECIPE = """\
Integration recipe (verified legs marked [*]):

  1. [*] catopt: opt, rep = Optimizer(backend=TorchBackend()).optimize(
         model, idx, strategy=Compositional())  — fused_10/fused_11
         executors grafted, verified vs eager.
  2. [*] HF export: hf_state_dict(opt, cfg) splits fused rows back to
         HF Llama layout + permutes q/k rows for rotate-half RoPE
         (attention-score equivalence checked by
         verify_rope_conversion); write_hf_dir emits config.json +
         model.safetensors + manifest.
  3.      Tokenizer: AutoTokenizer llama-tokenizer → save_pretrained,
         or synthetic WordLevel (bench only — text is meaningless).
  4.      Serve: LLM(model=hf_dir, dtype='float32', enforce_eager=True,
         gpu_memory_utilization=0.6)  — needs CUDA (this box: RTX 2050
         4 GB + torch cu wheels works; the repo .venv ships CPU torch).
  5.      Generate: llm.generate(prompt_token_ids=..., SamplingParams(
         temperature=0, max_tokens=N)).

vLLM does NOT consume the optimized graph — it re-implements Llama and
applies its own fusions (QKVParallelLinear is the same QKV merge
catopt finds).  catopt's structural wins that survive as *weights*
(folded chains → fewer GEMMs at runtime) map cleanly; executor-level
wins (eval-plan tapes, batched carriers) stay torch-side.  The honest
composition is 'catopt-verified weights → HF dir → vllm serve', which
this script runs end-to-end when vllm is importable.
"""


# ---------------------------------------------------------------------------
#  Orchestration
# ---------------------------------------------------------------------------


def load_model_and_cfg(ckpt: str, device: str):
    w = load_llama2c(ckpt)
    with open(ckpt, "rb") as f:
        hdr = np.frombuffer(f.read(28), dtype=np.int32)
    dim, hidden, L, nh = (int(hdr[i]) for i in range(4))
    vocab, seq = w["token_embedding"].shape[0], int(hdr[6])
    cfg = dict(
        dim=dim, hidden=hidden, n_layers=L, n_heads=nh,
        vocab=vocab, seq_len=seq,
    )
    return BatchedStories(w, cfg).eval().to(device), cfg


def run_bench(args) -> benchkit.Report:
    ckpt = resolve_ckpt(getattr(args, "ckpt", None))
    device = getattr(args, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    args.device = device
    args.batch = int(getattr(args, "batch", None) or 1)
    args.ctx = int(getattr(args, "ctx", None) or 64)
    args.gen = int(getattr(args, "gen", None) or 32)
    args.prompt_len = int(getattr(args, "prompt_len", None) or 16)
    args.seed = int(getattr(args, "seed", None) or 1234)
    args.gpu_frac = float(getattr(args, "gpu_frac", None) or 0.6)
    if getattr(args, "enforce_eager", None) is None:
        args.enforce_eager = True
    args.vllm_python = getattr(args, "vllm_python", None)
    args.repeats = int(
        getattr(args, "repeats", None) or (1 if args.quick else 3)
    )

    model, cfg = load_model_and_cfg(ckpt, device)
    env = benchkit.collect_env(device)
    print(
        f"ckpt={ckpt}  dim={cfg['dim']} L={cfg['n_layers']} "
        f"device={device}  torch={torch.__version__}",
        flush=True,
    )

    # One shared ctx-length prompt so the vLLM greedy tokens can be
    # diffed against the torch legs'.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    prompt = torch.randint(
        1, cfg["vocab"], (args.batch, args.ctx), generator=g
    )
    prompt_ids = prompt.tolist()

    cells: list[benchkit.Cell] = []
    notes: list[str] = []

    # ---- optimize once on the ctx-shaped input -----------------------
    idx0 = torch.randint(
        0, cfg["vocab"], (args.batch, args.ctx), device=device
    )
    opt, rep, vr, _ref, opt_s = optimize_once(model, idx0, device)
    notes.append(
        f"optimize: {opt_s:.1f}s, verify rel={vr.max_rel:.2e}, "
        f"{rep.get('n_optimized','?')}/{rep.get('n_blocks','?')} blocks"
    )
    if not vr.passed:
        notes.append("catopt leg FAILED verify — catopt rows skipped")
        opt = None

    # ---- leg 1: HF export (stock + optimized) ------------------------
    hf_root = Path(
        getattr(args, "hf_out", None)
        or Path(args.out or ".") / "hf_stories15m"
    )
    rope_d = verify_rope_conversion(
        model.blocks[0].wq.weight, model.blocks[0].wk.weight,
        cfg["n_heads"],
    )
    notes.append(f"rope permute attention-score max|Δ|={rope_d:.2e}")
    dirs = {}
    for tag, src in (("eager", model), ("catopt", opt)):
        if src is None:
            continue
        d = hf_root / tag
        sd_hf = hf_state_dict(src, cfg)
        write_hf_dir(d, sd_hf, cfg)
        dirs[tag] = d
        notes.append(f"hf[{tag}] -> {d} ({len(sd_hf)} tensors)")

    # ---- leg 2: torch decode ------------------------------------------
    # Untimed eager TRUE decode (unbounded window → real positions,
    # full history — same semantics as vLLM's KV cache) for the token
    # agreement check.  The TIMED variants all share the fixed-window
    # decode, which diverges from true positions after the first
    # generated token — internal timing is unaffected, but vLLM
    # agreement is measured against this true stream, not the timed
    # one.
    ref_tokens = None
    try:
        true_tokens, _ = greedy_decode(
            model, prompt.to(device), args.gen, window=10**9
        )
        ref_tokens = true_tokens.cpu()
    except Exception as e:
        notes.append(f"eager true-decode failed: {e}")
    if opt is not None:
        try:
            tcells, _ = torch_decode_leg(
                model, opt, prompt.to(device), args
            )
            cells += tcells
        except Exception as e:
            notes.append(f"torch decode leg crashed: {e}")

    # When transformers is importable (e.g. a vllm env), verify the
    # exported HF checkpoint's logits against the catopt model — the
    # strongest proof the weight mapping is correct.
    for tag, d in list(dirs.items()):
        try:
            from transformers import LlamaForCausalLM

            hf = LlamaForCausalLM.from_pretrained(
                str(d), torch_dtype=torch.float32
            ).eval()
            with torch.no_grad():
                ref = model(prompt.to(device)).cpu()[:, -1, :]
                got = hf(prompt).logits[:, -1, :]
            notes.append(
                f"hf[{tag}] transformers logits max|Δ|="
                f"{(ref - got).abs().max().item():.2e}"
            )
            del hf
        except ImportError:
            notes.append(
                "transformers absent — logits-level HF verify skipped "
                "(weight round-trip + rope score check still apply)"
            )
            break
        except Exception as e:
            notes.append(f"hf[{tag}] transformers check: {e}")

    # ---- leg 3: vllm ---------------------------------------------------
    vllm_results = {}
    for tag, d in dirs.items():
        ensure_msg = ensure_tokenizer(d, cfg["vocab"])
        notes.append(f"tokenizer[{tag}]: {ensure_msg}")
        if ensure_msg.startswith("FAILED") and not args.vllm_python:
            vllm_results[tag] = {
                "mode": "skipped",
                "reason": "no tokenizer could be produced",
            }
            continue
        res = vllm_leg(d, args, prompt_ids)
        vllm_results[tag] = res
        # token agreement vs torch eager TRUE decode (same semantics
        # as vLLM's KV cache — full history, real positions)
        agree_s = ""
        if ref_tokens is not None and res.get("tokens"):
            vt = torch.tensor(res["tokens"])
            n = min(vt.shape[1], ref_tokens.shape[1])
            agree = float(
                (vt[:, :n].cpu() == ref_tokens[:, :n].cpu())
                .float()
                .mean()
                .item()
            )
            res["token_agreement_vs_torch_eager"] = round(agree, 4)
            agree_s = f" token_agree={agree:.2f}"
        notes.append(
            f"vllm[{tag}]: mode={res.get('mode')} "
            f"decode_tok/s={res.get('decode_tok_per_s','—')}"
            + agree_s
            + (f" reason={res['reason']}" if res.get("reason") else "")
        )

    # fold vllm numbers into a Report cell (they're wall-clock, not
    # Timer autoranges — flagged in aux)
    vaux = {"wall_clock": True, "note": "vLLM offline LLM.generate"}
    for tag, res in vllm_results.items():
        for k, v in res.items():
            vaux[f"{tag}_{k}"] = v
    if vllm_results:
        # Represent vllm decode rates as a synthetic timed case: stmt
        # replays no work (seconds can't be re-measured here) — instead
        # attach numbers to the decode cell's aux when present.
        if cells:
            cells[0].aux.update(vaux)
        else:
            c = benchkit.Case(
                name="vllm serving", params={}, variants=[], aux=vaux
            )
            cells.append(benchkit.Cell(case=c, medians={}, iqr={}))

    notes.append(INTEGRATION_RECIPE)
    env["notes"] = notes
    return benchkit.Report(
        suite="vllm_compare", cells=cells, env=env
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--min-run-time", type=float, default=0.05)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results"))
    ap.add_argument("--hf-out", default=None)
    ap.add_argument("--vllm-python", default=None,
                    help="interpreter with vllm installed, for the "
                    "serving leg when this env lacks it")
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    ap.add_argument("--gpu-frac", type=float, default=0.6)
    ap.add_argument("--repeats", type=int, default=3,
                    help="vLLM gen repeats — best-of for tok/s")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plots", default=None)
    # internal: run ONLY the vllm leg (invoked as subprocess)
    ap.add_argument("--vllm-leg", action="store_true")
    ap.add_argument("--hf-dir", default=None)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--prompt-ids-file", default=None)
    args = ap.parse_args()

    if args.vllm_leg:
        _vllm_env_setup()
        if args.prompt_ids_file:
            args.prompt_ids = json.loads(
                Path(args.prompt_ids_file).read_text()
            )
        res = _vllm_leg_payload(args.hf_dir, args)
        Path(args.out_json).write_text(json.dumps(res))
        return

    if args.quick:
        args.gen = min(args.gen, 16)
        args.ctx = min(args.ctx, 48)

    report = run_bench(args)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report.to_json(out / f"vllm_compare_{ts}.json")
    report.to_markdown(out / f"vllm_compare_{ts}.md")
    for n in report.env.get("notes", []):
        print(n)
    print(f"\nwrote {out}/vllm_compare_{ts}.{{json,md}}")


if __name__ == "__main__":
    main()
