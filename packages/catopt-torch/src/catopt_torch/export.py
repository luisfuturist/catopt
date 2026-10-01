"""Production export of optimized models — catopt-free artifacts.

:class:`~catopt_orchestrator.Optimizer` deliveries
(:meth:`~catopt_orchestrator.Optimizer.optimize` under the
:class:`~catopt_orchestrator.Monolithic` or
:class:`~catopt_orchestrator.Compositional` strategy) return modules whose
forward runs catopt machinery — ``IRModule``'s term-evaluation plan
tapes or the level-batched carrier executors (``BatchedScanModule`` /
``BatchedOMModule`` / ``BatchedOmdModule``).  Loading such a module at
inference time therefore needs catopt installed.  :func:`export_optimized`
materialises the *computation* into portable artifacts that downstream
tooling consumes with nothing but torch — or a weights file plus a
manifest when only the parameters are portable.

Formats (``fmt`` — aliases in parentheses):

``"module"`` (``"pt2"``, ``"exported_program"``) — **default**
    ``torch.export`` the optimized module and save the
    ``ExportedProgram`` (.pt2).  The serialized aten graph IS the
    optimized structure — fused GEMMs, batched carriers and all —
    and ``torch.export.load`` rebuilds a runnable ``nn.Module`` with
    no catopt install.  This is the recommended artifact.  Requires
    ``example_input``; the exported program is specialised to the
    example's shapes unless ``dynamic_shapes`` is passed through.

``"safetensors"``
    Write the optimized ``state_dict()`` in the safetensors container
    (serialised here in stdlib+torch — the ``safetensors`` package is
    NOT required to write, only to read downstream) plus a JSON
    manifest.  Honest limit: the keys are the *optimized* module's
    parameter names — fused/derived weights — so the file loads into a
    model with the same optimized layout (the catopt executor classes
    or a hand-built equivalent).  It is not a drop-in for the
    original architecture's ``load_state_dict``.

``"state_dict"`` (``"sd"``, ``"pth"``)
    ``torch.save`` the CPU state dict — a plain ``torch.load`` with
    ``weights_only=True`` reads it.  Same optimized-layout caveat as
    safetensors.

``"torchscript"`` (``"ts"``, ``"jit"``)
    ``torch.jit.trace`` when ``example_input`` is given,
    ``torch.jit.script`` otherwise.  The carrier executors' ``*args``
    forwards are not scriptable and some are not traceable — failures
    surface as :class:`ExportError` with the underlying error, never
    silently dropped.

GGUF / Ollama / LM Studio
    These runtimes (llama.cpp) reimplement *standard* architectures
    and consume HF ``config.json`` + safetensors weights with fixed
    tensor names.  The optimized graph's fused parameter names do not
    map onto that schema, and GGUF conversion of an ExportedProgram
    is not a supported path anywhere — the honest bridge is exporting
    the ORIGINAL (unoptimized) model through llama.cpp's
    ``convert_hf_to_gguf.py`` and letting the runtime pick its own
    kernels.  Every manifest carries this note in ``notes``.

Every export writes ``<artifact>.manifest.json`` next to the
artifact and returns the manifest dict.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "ExportError",
    "export_optimized",
    "load_optimized",
    "save_optimized",
    "save_optimized_weights",
]


class ExportError(RuntimeError):
    """Export to a catopt-free artifact failed."""


_FORMAT_ALIASES = {
    "module": "pt2",
    "pt2": "pt2",
    "exported_program": "pt2",
    "safetensors": "safetensors",
    "st": "safetensors",
    "torchscript": "torchscript",
    "ts": "torchscript",
    "jit": "torchscript",
    "state_dict": "state_dict",
    "sd": "state_dict",
    "pth": "state_dict",
}

_SUFFIX = {
    "pt2": ".pt2",
    "safetensors": ".safetensors",
    "torchscript": ".pt",
    "state_dict": ".pth",
}

# torch dtype → safetensors dtype tag.  Only dense strided layouts.
_TORCH_TO_ST: dict[torch.dtype, str] = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
}
_ST_TO_TORCH: dict[str, torch.dtype] = {
    v: k for k, v in _TORCH_TO_ST.items()
}


# ---------------------------------------------------------------------------
#  small helpers
# ---------------------------------------------------------------------------


def _canon_fmt(fmt: str) -> str:
    """Resolve ``fmt`` aliases to a canonical format name."""
    try:
        return _FORMAT_ALIASES[str(fmt).lower()]
    except KeyError:
        raise ExportError(
            f"unsupported export format {fmt!r}; expected one of "
            f"{sorted(set(_FORMAT_ALIASES.values()))}"
        ) from None


def _target_path(path: str | Path, fmt: str) -> Path:
    """``path`` as given; when it has no suffix, append the canonical one."""
    p = Path(path)
    if p.suffix:
        return p
    return p.with_name(p.name + _SUFFIX[fmt])


def _manifest_path(artifact: Path) -> Path:
    """``model.pt2`` → ``model.pt2.manifest.json`` (sibling file).

    Name-based, not stem-based: ``opt.pt2`` and ``opt.safetensors``
    must not overwrite each other's manifest.
    """
    return artifact.with_name(f"{artifact.name}.manifest.json")


def _as_args(
    example_input: torch.Tensor | tuple | list | None,
) -> tuple | None:
    """Normalise ``example_input`` to a positional-args tuple."""
    if example_input is None:
        return None
    if isinstance(example_input, torch.Tensor):
        return (example_input,)
    if isinstance(example_input, (tuple, list)):
        return tuple(example_input)
    raise ExportError(
        "example_input must be a Tensor or a tuple/list of Tensors, "
        f"got {type(example_input).__name__}"
    )


def _run(mod: Any, args: tuple, kwargs: dict[str, Any]) -> Any:
    with torch.no_grad():
        return mod(*args, **kwargs)


# ---------------------------------------------------------------------------
#  manifest: which submodules are catopt machinery
# ---------------------------------------------------------------------------


def _original_class(
    model: torch.nn.Module | None, name: str
) -> str | None:
    """Class of ``model.<name>`` pre-optimization, for the manifest."""
    if model is None:
        return None
    try:
        sub = model.get_submodule(name)
    except AttributeError:
        return "<absent>"
    cls = type(sub)
    return f"{cls.__module__}.{cls.__qualname__}"


def _catopt_modules(
    model: torch.nn.Module | None,
    optimized_model: torch.nn.Module,
    state_dict: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Map ``{dotted_name: info}`` for every catopt-defined submodule.

    These are the grafted executors — the pieces that made the
    optimized module un-loadable without catopt.  The manifest records
    each one's class, its pre-optimization counterpart (when ``model``
    is given), and which ``state_dict`` keys belong to it.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, sub in optimized_model.named_modules():
        if not name:
            continue
        cls = type(sub)
        if not cls.__module__.startswith("catopt"):
            continue
        out[name] = {
            "optimized": f"{cls.__module__}.{cls.__qualname__}",
            "original": _original_class(model, name),
            "state_keys": [
                k for k in state_dict if k.startswith(f"{name}.")
            ],
        }
    return out


def _notes(fmt: str, has_catopt: bool) -> list[str]:
    """Human/agent-readable usage + limits notes for the manifest."""
    if fmt == "pt2":
        notes = [
            "torch.export.load(artifact).module() reproduces the "
            "optimized forward — no catopt install required"
        ]
    elif fmt == "torchscript":
        notes = [
            "torch.jit.load(artifact) reproduces the optimized "
            "forward — no catopt install required"
        ]
    else:
        notes = [
            "weights file: keys are the OPTIMIZED module's "
            "state_dict names (fused/derived parameters) — it loads "
            "into the optimized layout (catopt executor classes or a "
            "matching hand-built module), not the original module "
            "tree"
        ]
        if has_catopt:
            notes.append(
                "catopt_modules lists each grafted executor block, "
                "the class it replaced, and its state_dict keys"
            )
    notes.append(
        "GGUF (Ollama/LM Studio): llama.cpp reimplements standard "
        "architectures from HF-layout weights; the optimized fused "
        "parameter names do not map — convert the ORIGINAL model "
        "with llama.cpp convert_hf_to_gguf.py instead"
    )
    return notes


#: Certified bounded-rewrite ledger keys (plan 0012) copied into the
#: manifest when the pipeline stats carry them.
_BOUND_KEYS = (
    "error_budget",
    "error_bound_total",
    "error_bound_output",
    "error_bounds",
    "error_bounds_honored",
)

#: Task-metric certificate keys (plan 0015) — the metric's record
#: (``task``: name / tolerance / measured distance / verdict /
#: ``evaluated_on``, i.e. which calibration input conditioned the
#: certificate), which metric produced the verify verdict, and which
#: contract accepted the delivery.
_TASK_KEYS = ("task", "verify_metric", "accepted_by")


def _bound_manifest(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Slice the bounded-rewrite ledger out of *stats* for the manifest.

    Plan-0012 honesty: an approximate delivery is never silent in the
    exported artifact — a ``False`` ``error_bounds_honored`` is a
    signal, not an omission, so ``is not None`` keeps it.
    """
    if stats is None:
        return {}
    return {
        k: stats[k] for k in _BOUND_KEYS if stats.get(k) is not None
    }


def _task_manifest(stats: dict[str, Any] | None) -> dict[str, Any]:
    """Slice the task-metric certificate out of *stats* for the manifest.

    Plan-0015 honesty: a task-gated delivery is never silent — the
    manifest carries which metric gated acceptance, its tolerance and
    the measured distance on the verify input.  A declared contract
    that was never evaluated (``verify=False``) exports ``task``
    without ``distance``/``passed`` — visibly a promise, not a proof.
    """
    if stats is None:
        return {}
    return {k: stats[k] for k in _TASK_KEYS if stats.get(k) is not None}


# ---------------------------------------------------------------------------
#  forward-output comparison (verify)
# ---------------------------------------------------------------------------


def _diff(a: Any, b: Any) -> float:
    """Largest |a - b| over tensor leaves of two matching pytrees.

    Raises :class:`ExportError` on a numeric or structural mismatch.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        try:
            torch.testing.assert_close(b, a)
        except AssertionError as e:
            raise ExportError(
                f"artifact round-trip output mismatch: {e}"
            ) from e
        if not (a.is_floating_point() or a.is_complex()):
            return 0.0
        if a.numel() == 0:
            return 0.0
        return float((a - b).abs().max().item())
    if (
        isinstance(a, (tuple, list))
        and isinstance(b, type(a))
        and len(a) == len(b)
    ):
        return max(
            (_diff(x, y) for x, y in zip(a, b, strict=True)),
            default=0.0,
        )
    if (
        isinstance(a, dict)
        and isinstance(b, dict)
        and a.keys() == b.keys()
    ):
        return max((_diff(a[k], b[k]) for k in a), default=0.0)
    raise ExportError(
        "artifact output structure differs from the module's: "
        f"{type(a).__name__} vs {type(b).__name__}"
    )


def _verify_forward(
    mod: Any,
    artifact_mod: Any,
    args: tuple,
    kwargs: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    """Re-run ``artifact_mod`` and assert it matches ``mod``."""
    ref = _run(mod, args, kwargs)
    got = _run(artifact_mod, args, kwargs)
    manifest["verify"] = {
        "kind": "forward",
        "max_abs_diff": _diff(ref, got),
    }


def _verify_weights(
    written: dict[str, torch.Tensor],
    reload: dict[str, torch.Tensor],
    manifest: dict[str, Any],
) -> None:
    """Assert a weight file round-trips byte-exactly."""
    bad = [
        k
        for k, v in written.items()
        if k not in reload or not torch.equal(v, reload[k])
    ]
    manifest["verify"] = {
        "kind": "state_dict",
        "tensors_checked": len(written),
        "exact": not bad,
    }
    if bad:
        raise ExportError(f"weight round-trip mismatch on {bad[:8]}")


# ---------------------------------------------------------------------------
#  weight serialisation (safetensors container, no safetensors dep)
# ---------------------------------------------------------------------------


def _tensor_entries(
    state_dict: dict[str, Any],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Split a state_dict into (tensor entries, skipped non-tensor keys)."""
    keep: dict[str, torch.Tensor] = {}
    skip: list[str] = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            # Not contiguous(): sparse/nested layouts legitimately
            # reach torch.save; _tensor_bytes rejects them for
            # safetensors on its own terms.
            keep[k] = v.detach().cpu()
        else:
            skip.append(k)
    return keep, skip


def _tensor_bytes(t: torch.Tensor, key: str) -> bytes:
    """Raw little-endian bytes of a dense CPU tensor (numpy speed)."""
    if t.layout != torch.strided:
        raise ExportError(
            f"safetensors only stores dense strided tensors; "
            f"{key!r} has layout {t.layout}"
        )
    st_dtype = _TORCH_TO_ST.get(t.dtype)
    if st_dtype is None:
        raise ExportError(
            f"safetensors has no dtype for {t.dtype} (key {key!r})"
        )
    try:
        import numpy  # noqa: F401 — required by Tensor.numpy()
    except ImportError as e:
        raise ExportError(
            "safetensors export needs numpy (a declared dependency of "
            "the catopt meta distribution)"
        ) from e
    u8 = t.contiguous().view(torch.uint8).reshape(-1)
    return u8.numpy().tobytes()


def _write_safetensors(
    entries: dict[str, torch.Tensor],
    path: Path,
    manifest_json: str,
) -> None:
    """Write the safetensors container: header len | JSON | raw bytes."""
    header: dict[str, Any] = {
        "__metadata__": {"catopt_manifest": manifest_json}
    }
    blobs: list[bytes] = []
    offset = 0
    for name, t in entries.items():
        raw = _tensor_bytes(t, name)
        header[name] = {
            "dtype": _TORCH_TO_ST[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for blob in blobs:
            f.write(blob)


def _read_safetensors(path: Path) -> dict[str, torch.Tensor]:
    """Parse a safetensors file back into tensors (stdlib+torch)."""
    try:
        data = path.read_bytes()
        (n,) = struct.unpack("<Q", data[:8])
        header = json.loads(data[8 : 8 + n])
    except Exception as e:
        raise ExportError(
            f"{path} is not a safetensors file: {e}"
        ) from e
    base = 8 + n
    out: dict[str, torch.Tensor] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dt = _ST_TO_TORCH.get(info["dtype"])
        if dt is None:
            raise ExportError(
                f"safetensors dtype {info['dtype']!r} is unsupported"
            )
        lo, hi = info["data_offsets"]
        t = torch.frombuffer(
            bytearray(data[base + lo : base + hi]), dtype=dt
        )
        out[name] = t.reshape(info["shape"])
    return out


# ---------------------------------------------------------------------------
#  per-format exporters
# ---------------------------------------------------------------------------


def _export_pt2(
    mod: torch.nn.Module,
    path: Path,
    args: tuple | None,
    kwargs: dict[str, Any],
    dynamic_shapes: Any,
    verify: bool,
    manifest: dict[str, Any],
) -> None:
    """torch.export → ``ExportedProgram`` (.pt2): graph + weights."""
    if args is None:
        raise ExportError(
            "fmt='module' exports the traced computation graph and "
            "needs example_input (a Tensor or tuple of Tensors)"
        )
    try:
        ep = torch.export.export(
            mod,
            args,
            kwargs=kwargs or None,
            dynamic_shapes=dynamic_shapes,
        )
    except Exception as e:
        raise ExportError(f"torch.export failed: {e}") from e
    torch.export.save(ep, str(path))
    if verify:
        loaded = torch.export.load(str(path)).module()
        _verify_forward(mod, loaded, args, kwargs, manifest)


def _export_torchscript(
    mod: torch.nn.Module,
    path: Path,
    args: tuple | None,
    kwargs: dict[str, Any],
    verify: bool,
    manifest: dict[str, Any],
) -> None:
    """TorchScript: trace with an example, else attempt script."""
    if args is None:
        if kwargs:
            raise ExportError(
                "example_kwargs needs example_input for torchscript "
                "export (trace inputs) — script mode takes no inputs"
            )
        try:
            ts_mod = torch.jit.script(mod)
        except Exception as e:
            raise ExportError(
                "torch.jit.script failed — carrier executor forwards "
                "(*args signatures, eval plan tapes) are not "
                f"scriptable; pass example_input to trace instead: {e}"
            ) from e
    else:
        try:
            ts_mod = torch.jit.trace(
                mod, args, example_kwarg_inputs=kwargs or None
            )
        except Exception as e:
            raise ExportError(f"torch.jit.trace failed: {e}") from e
    torch.jit.save(ts_mod, str(path))
    if verify:
        if args is None:
            manifest["verify"] = {
                "kind": "script",
                "skipped": "no example_input to compare on",
            }
        else:
            _verify_forward(
                mod, torch.jit.load(str(path)), args, kwargs, manifest
            )


def _export_weights(
    mod: torch.nn.Module,
    path: Path,
    fmt: str,
    verify: bool,
    manifest: dict[str, Any],
) -> None:
    """Weights-only artifacts: safetensors container or torch.save."""
    entries, skipped = _tensor_entries(mod.state_dict())
    if skipped:
        manifest["skipped_state_entries"] = skipped
    if fmt == "safetensors":
        _write_safetensors(entries, path, json.dumps(manifest))
        reload: dict[str, torch.Tensor] = (
            _read_safetensors(path) if verify else {}
        )
    else:
        torch.save(entries, str(path))
        reload = (
            torch.load(str(path), weights_only=True) if verify else {}
        )
    if verify:
        _verify_weights(entries, reload, manifest)


# ---------------------------------------------------------------------------
#  public API
# ---------------------------------------------------------------------------


def export_optimized(
    model: torch.nn.Module | None,
    optimized_model: torch.nn.Module,
    path: str | Path,
    *,
    fmt: str = "module",
    example_input: torch.Tensor | tuple | list | None = None,
    example_kwargs: dict[str, Any] | None = None,
    dynamic_shapes: Any = None,
    verify: bool = True,
    stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export an optimized model to a catopt-free artifact.

    Parameters
    ----------
    model : torch.nn.Module or None
        The model BEFORE optimization — used only to annotate the
        manifest with which class each grafted executor replaced.
        Pass ``None`` when unknown.
    optimized_model : torch.nn.Module
        The module returned by ``optimize_model`` /
        ``optimize_compositional``.
    path : str or Path
        Artifact path.  When it carries no suffix the canonical one is
        appended (``.pt2`` / ``.safetensors`` / ``.pt`` / ``.pth``).
    fmt : str
        ``"module"`` (default; .pt2 ``ExportedProgram``),
        ``"safetensors"``, ``"state_dict"``, or ``"torchscript"`` —
        see the module docstring for what each delivers and for the
        GGUF/Ollama story.
    example_input : Tensor or tuple/list, optional
        Forward inputs fixing the exported shapes — required for
        ``fmt="module"``; enables tracing + verify for
        ``fmt="torchscript"``.
    example_kwargs : dict, optional
        Keyword args alongside ``example_input``.
    dynamic_shapes : optional
        Passed through to ``torch.export.export`` (e.g. mark the batch
        dim dynamic) — ignored by other formats.
    verify : bool
        Reload the artifact and assert the forward (runnable formats)
        or the weights (weights formats) round-trip.  On mismatch an
        :class:`ExportError` is raised — the manifest is not written.
    stats : dict, optional
        The pipeline's search/lower stats (``LowerResult.stats``).
        When they carry certified bounded members — ``error_budget`` /
        ``error_bound_total`` / ``error_bounds`` /
        ``error_bounds_honored`` — the keys are copied into the
        manifest verbatim, so an approximate delivery is never silent
        about its error envelope.  The task-metric keys (``task`` /
        ``verify_metric`` / ``accepted_by``, plan 0015) copy the same
        way — the manifest records which task metric gated acceptance,
        its tolerance, and the measured distance on the verify input.

    Returns
    -------
    dict
        The export manifest (also written to
        ``<artifact>.manifest.json``).

    """
    fmt_c = _canon_fmt(fmt)
    p = _target_path(path, fmt_c)
    args = _as_args(example_input)
    kwargs = dict(example_kwargs) if example_kwargs else {}

    sd = optimized_model.state_dict()
    catopt_mods = _catopt_modules(model, optimized_model, sd)
    manifest: dict[str, Any] = {
        "catopt_export": 1,
        "format": fmt_c,
        "artifact": str(p),
        "manifest_file": str(_manifest_path(p)),
        "torch_version": torch.__version__,
        "training_mode": optimized_model.training,
        "runnable_standalone": fmt_c in ("pt2", "torchscript"),
        "weights_only": fmt_c in ("safetensors", "state_dict"),
        "n_state_entries": len(sd),
        "catopt_modules": catopt_mods,
    }
    manifest["notes"] = _notes(fmt_c, bool(catopt_mods))
    manifest.update(_bound_manifest(stats))
    manifest.update(_task_manifest(stats))

    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt_c == "pt2":
        _export_pt2(
            optimized_model,
            p,
            args,
            kwargs,
            dynamic_shapes,
            verify,
            manifest,
        )
    elif fmt_c == "torchscript":
        _export_torchscript(
            optimized_model, p, args, kwargs, verify, manifest
        )
    else:
        _export_weights(optimized_model, p, fmt_c, verify, manifest)

    manifest["artifact_bytes"] = p.stat().st_size
    _manifest_path(p).write_text(
        json.dumps(manifest, indent=2, default=str)
    )
    return manifest


# ``save_optimized`` reads better at call sites that pass a filename.
save_optimized = export_optimized


def save_optimized_weights(
    optimized_module: torch.nn.Module, path: str | Path
) -> None:
    """Emit the optimized weights file.

    Only the parameters the certified form actually needs (folded
    derived tensors included).
    """
    torch.save(optimized_module.state_dict(), str(path))


def load_optimized(path: str | Path) -> Any:
    """Load an artifact written by :func:`export_optimized`.

    Returns a runnable ``torch.nn.Module`` for ``pt2``/``torchscript``
    artifacts (no catopt needed) and a ``dict[str, Tensor]`` of the
    optimized state_dict for ``safetensors``/``state_dict`` artifacts.
    The sibling manifest decides the format; without one, the file
    suffix is used (``.pt`` tries TorchScript first, then
    ``torch.load``).
    """
    p = Path(path)
    mpath = _manifest_path(p)
    if mpath.exists():
        fmt = json.loads(mpath.read_text()).get("format")
        if fmt == "pt2":
            return torch.export.load(str(p)).module()
        if fmt == "safetensors":
            return _read_safetensors(p)
        if fmt == "torchscript":
            return torch.jit.load(str(p))
        if fmt == "state_dict":
            return torch.load(str(p), weights_only=True)
        raise ExportError(
            f"manifest for {p} names unknown format {fmt!r}"
        )
    suffix = p.suffix.lower()
    if suffix == ".pt2":
        return torch.export.load(str(p)).module()
    if suffix == ".safetensors":
        return _read_safetensors(p)
    if suffix == ".pth":
        return torch.load(str(p), weights_only=True)
    if suffix == ".pt":
        try:
            return torch.jit.load(str(p))
        except Exception:
            return torch.load(str(p), weights_only=True)
    raise ExportError(
        f"cannot infer artifact format for {p} — no manifest and an "
        f"unrecognised suffix"
    )
