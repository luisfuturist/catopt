"""Production export: optimized model → catopt-free artifacts.

``export_optimized`` materialises an optimized module (IRModule /
batched-carrier grafts and all) into artifacts downstream tooling
loads without catopt: a ``torch.export`` .pt2 program, a safetensors
weights container + manifest, a plain ``torch.save`` state dict, or
TorchScript.  These tests exercise every format's write → load → run
round-trip and the manifest, plus the documented failure modes.
"""

import json
import struct
import subprocess
import sys
import types

import pytest
import torch
import torch.nn as nn
from catopt_torch.export import ExportError, _diff, _read_safetensors, export_optimized, load_optimized, save_optimized



from tests.test_compositional import MiniGPT
from catopt_orchestrator import Compositional, Optimizer


from catopt_torch.backend import TorchBackend


@pytest.fixture
def optimized_stack():
    """A small compositional-optimized MiniGPT + its input."""
    torch.manual_seed(0)
    model = MiniGPT(dim=32, n_heads=2, depth=2, hidden_mult=2).eval()
    x = torch.randn(1, 8, 32)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x, strategy=Compositional(), verbose=False)

    assert stats["n_optimized"] == 2  # IRModules grafted
    return model, opt, x


def _manifest_file(artifact):
    return artifact.with_name(f"{artifact.name}.manifest.json")


# ---------------------------------------------------------------------------
#  fmt="module" — torch.export .pt2
# ---------------------------------------------------------------------------


def test_export_module_pt2_roundtrip(optimized_stack, tmp_path):
    """Default format: the .pt2 reproduces the optimized forward
    EXACTLY (fp32 identical) and the manifest records the grafts."""
    model, opt, x = optimized_stack
    out = tmp_path / "opt.pt2"

    manifest = export_optimized(
        model, opt, out, example_input=x, fmt="module"
    )

    assert manifest["format"] == "pt2"
    assert manifest["runnable_standalone"] is True
    assert manifest["weights_only"] is False
    assert manifest["verify"]["kind"] == "forward"
    assert manifest["verify"]["max_abs_diff"] == 0.0
    assert manifest["artifact_bytes"] == out.stat().st_size

    # The manifest names the grafted catopt executors and what they
    # replaced — ParallelBlock → catopt_torch IRModule.
    blocks = manifest["catopt_modules"]
    assert set(blocks) == {"blocks.0", "blocks.1"}
    assert "IRModule" in blocks["blocks.0"]["optimized"]
    assert "ParallelBlock" in blocks["blocks.0"]["original"]
    assert any(
        k.startswith("blocks.0.")
        for k in blocks["blocks.0"]["state_keys"]
    )
    assert any("GGUF" in n for n in manifest["notes"])

    # Sidecar manifest was written with the same payload.
    on_disk = json.loads(_manifest_file(out).read_text())
    assert on_disk["format"] == "pt2"

    # Round-trip: the loaded program runs without catopt and matches
    # the optimized module bit-for-bit.
    loaded = load_optimized(out)
    with torch.no_grad():
        got = loaded(x.clone())
        ref = opt(x.clone())
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_pt2_artifact_runs_in_clean_interpreter(
    optimized_stack, tmp_path
):
    """The .pt2 loads + runs in a Python process that never imports
    catopt — the actual 'no catopt at inference time' requirement."""
    _, opt, x = optimized_stack
    out = tmp_path / "opt.pt2"
    export_optimized(None, opt, out, example_input=x, fmt="pt2")

    code = (
        "import sys, torch; "
        f"ep = torch.export.load({str(out)!r}); "
        "m = ep.module(); "
        "out = m(torch.randn(1, 8, 32)); "
        "assert out.shape == (1, 8, 32); "
        "assert not any(m.startswith('catopt') for m in sys.modules)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr


def test_export_module_needs_example_input(optimized_stack, tmp_path):
    """.pt2 serialises a traced graph — no example input is an error."""
    _, opt, _ = optimized_stack
    with pytest.raises(ExportError, match="example_input"):
        export_optimized(None, opt, tmp_path / "o.pt2")


def test_export_torch_export_failure_is_wrapped(
    optimized_stack, tmp_path, monkeypatch
):
    """A torch.export failure surfaces as ExportError, not a raw
    dynamo traceback leak."""
    _, opt, x = optimized_stack

    def boom(*_a, **_k):
        raise RuntimeError("dynamo exploded")

    monkeypatch.setattr(torch.export, "export", boom)
    with pytest.raises(ExportError, match=r"torch\.export failed"):
        export_optimized(None, opt, tmp_path / "o.pt2", example_input=x)


def test_verify_catches_roundtrip_mismatch(
    optimized_stack, tmp_path, monkeypatch
):
    """verify=True actually guards: a wrong-structure reload raises."""
    _, opt, x = optimized_stack

    fake_ep = types.SimpleNamespace(
        module=lambda: lambda *a, **k: (x, x)  # wrong structure
    )
    monkeypatch.setattr(torch.export, "load", lambda p: fake_ep)
    with pytest.raises(ExportError, match="structure"):
        export_optimized(None, opt, tmp_path / "o.pt2", example_input=x)


def test_verify_catches_numeric_mismatch(tmp_path, monkeypatch):
    """A numerically wrong reload is an ExportError via assert_close."""
    mod = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    fake_ep = types.SimpleNamespace(
        module=lambda: lambda *a, **k: torch.zeros(2, 4)
    )
    monkeypatch.setattr(torch.export, "load", lambda p: fake_ep)
    with pytest.raises(ExportError, match="mismatch"):
        export_optimized(None, mod, tmp_path / "o.pt2", example_input=x)


def test_dynamic_shapes_passthrough(tmp_path):
    """dynamic_shapes is forwarded to torch.export — a marked batch
    dim stays dynamic in the saved program."""
    mod = nn.Linear(4, 4).eval()
    x = torch.randn(2, 4)
    batch = torch.export.Dim("batch")
    manifest = export_optimized(
        None,
        mod,
        tmp_path / "dyn.pt2",
        example_input=x,
        dynamic_shapes=({0: batch},),
    )
    assert manifest["format"] == "pt2"
    loaded = load_optimized(tmp_path / "dyn.pt2")
    with torch.no_grad():
        out = loaded(torch.randn(5, 4))  # batch 5 ≠ example's 2
    assert out.shape == (5, 4)


def test_example_input_variants(tmp_path):
    """Tensor, tuple, and list inputs all export; a non-sequence
    example_input is a clear error."""
    mod = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    x = torch.randn(2, 4)
    manifest = export_optimized(
        None, mod, tmp_path / "a.pt2", example_input=[x]
    )
    assert manifest["verify"]["max_abs_diff"] == 0.0

    with pytest.raises(ExportError, match="example_input must be"):
        export_optimized(
            None, mod, tmp_path / "b.pt2", example_input=42
        )


def test_example_kwargs_forwarded(tmp_path):
    """Keyword example inputs reach the exported program."""

    class Scaled(nn.Module):
        def forward(self, x, scale=1.0):
            return x * scale

    mod = Scaled().eval()
    x = torch.randn(4)
    manifest = export_optimized(
        None,
        mod,
        tmp_path / "k.pt2",
        example_input=x,
        example_kwargs={"scale": 3.0},
    )
    loaded = load_optimized(tmp_path / "k.pt2")
    with torch.no_grad():
        torch.testing.assert_close(loaded(x, scale=3.0), x * 3.0)
    assert manifest["verify"]["max_abs_diff"] == 0.0


def test_suffix_appended_when_missing(tmp_path):
    """A suffixless path gains the canonical one."""
    mod = nn.Linear(4, 4).eval()
    manifest = export_optimized(
        None, mod, tmp_path / "bare", example_input=torch.randn(2, 4)
    )
    assert manifest["artifact"].endswith("bare.pt2")
    assert (tmp_path / "bare.pt2").exists()


def test_exotic_output_tree_verifies(tmp_path):
    """Dict/tuple/bool/empty-tensor outputs all verify through _diff."""

    class Exotic(nn.Module):
        def forward(self, x):
            return {
                "sum": x.sum(),
                "parts": (x, x > 0),
                "empty": x[:, :0],
            }

    mod = Exotic().eval()
    manifest = export_optimized(
        None, mod, tmp_path / "e.pt2", example_input=torch.randn(2, 3)
    )
    assert manifest["verify"]["max_abs_diff"] == 0.0


def test_diff_helper_branches():
    """Unit-level: _diff walks pytrees, zeros non-floats/empties, and
    rejects structure mismatches."""
    t = torch.randn(3)
    assert _diff(t, t) == 0.0
    assert _diff((t,), (t,)) == 0.0
    assert _diff({"a": t}, {"a": t}) == 0.0
    assert _diff(t > 0, t > 0) == 0.0  # non-float leaf
    assert _diff(torch.empty(0), torch.empty(0)) == 0.0
    assert _diff((), ()) == 0.0  # empty pytree → default
    with pytest.raises(ExportError, match="structure"):
        _diff(t, (t,))
    with pytest.raises(ExportError, match="mismatch"):
        _diff(t, torch.zeros(3))


# ---------------------------------------------------------------------------
#  fmt="safetensors" and fmt="state_dict" — weights only
# ---------------------------------------------------------------------------


def test_safetensors_roundtrip(optimized_stack, tmp_path):
    """The weights container round-trips byte-exactly; the manifest
    (embedded AND sidecar) names the optimized layout honestly."""
    model, opt, _ = optimized_stack
    out = tmp_path / "opt.safetensors"

    manifest = export_optimized(model, opt, out, fmt="safetensors")

    assert manifest["format"] == "safetensors"
    assert manifest["weights_only"] is True
    assert manifest["runnable_standalone"] is False
    assert manifest["verify"]["kind"] == "state_dict"
    assert manifest["verify"]["exact"] is True
    assert any("optimized" in n.lower() for n in manifest["notes"])

    # Embedded metadata carries the manifest into the file itself.
    data = out.read_bytes()
    (n,) = struct.unpack("<Q", data[:8])
    header = json.loads(data[8 : 8 + n])
    embedded = json.loads(header["__metadata__"]["catopt_manifest"])
    assert embedded["format"] == "safetensors"

    # And the tensor bytes decode back to the optimized state_dict.
    reloaded = load_optimized(out)
    for k, v in opt.state_dict().items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(reloaded[k], v.cpu())


def test_safetensors_skips_non_tensor_entries(tmp_path):
    """get_extra_state-style non-tensor state entries are skipped and
    recorded, not serialised into the container."""

    class WithExtra(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def get_extra_state(self):
            return {"tag": "not-a-tensor"}

    mod = WithExtra().eval()
    manifest = export_optimized(
        None, mod, tmp_path / "w.safetensors", fmt="safetensors"
    )
    assert manifest["skipped_state_entries"] == ["_extra_state"]
    reloaded = load_optimized(tmp_path / "w.safetensors")
    assert "_extra_state" not in reloaded
    assert torch.equal(reloaded["lin.weight"], mod.lin.weight)


def test_safetensors_rejects_bad_dtype(tmp_path):
    """A dtype safetensors can't name (complex) is an ExportError."""

    class Complex(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "c", torch.ones(4, dtype=torch.complex64)
            )

    mod = Complex().eval()
    with pytest.raises(ExportError, match="no dtype"):
        export_optimized(
            None, mod, tmp_path / "c.safetensors", fmt="safetensors"
        )


def test_safetensors_rejects_sparse(tmp_path):
    """Sparse layouts can't serialise to the dense container."""

    class Sparse(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("sp", torch.eye(4).to_sparse())

    mod = Sparse().eval()
    with pytest.raises(ExportError, match="strided"):
        export_optimized(
            None, mod, tmp_path / "s.safetensors", fmt="safetensors"
        )


def test_safetensors_needs_numpy(tmp_path, monkeypatch):
    """numpy absent → a clear ExportError (it backs the fast byte path)."""
    mod = nn.Linear(4, 4).eval()
    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(ExportError, match="numpy"):
        export_optimized(
            None, mod, tmp_path / "n.safetensors", fmt="safetensors"
        )


def test_safetensors_plain_model_has_no_catopt_notes(tmp_path):
    """Weights of an unoptimized model carry no graft notes."""
    mod = nn.Linear(4, 4).eval()
    manifest = export_optimized(
        None, mod, tmp_path / "p.safetensors", fmt="safetensors"
    )
    assert manifest["catopt_modules"] == {}
    assert not any("grafted" in n for n in manifest["notes"])


def test_state_dict_roundtrip(optimized_stack, tmp_path):
    """fmt='state_dict' → plain torch.load(weights_only=True) dict."""
    model, opt, _ = optimized_stack
    out = tmp_path / "opt.pth"
    manifest = export_optimized(model, opt, out, fmt="state_dict")

    assert manifest["format"] == "state_dict"
    assert manifest["verify"]["exact"] is True
    reloaded = torch.load(out, weights_only=True)
    assert set(reloaded) == set(opt.state_dict())
    for k, v in reloaded.items():
        assert torch.equal(v, opt.state_dict()[k].cpu())

    # load_optimized returns the same dict
    loaded = load_optimized(out)
    for k, v in loaded.items():
        assert torch.equal(v, reloaded[k])


def test_weights_verify_mismatch(tmp_path, monkeypatch):
    """A corrupted reload (here: everything missing) raises."""
    mod = nn.Linear(4, 4).eval()
    monkeypatch.setattr(torch, "load", lambda *a, **k: {})
    with pytest.raises(ExportError, match="round-trip"):
        export_optimized(
            None, mod, tmp_path / "m.pth", fmt="state_dict"
        )


def test_read_safetensors_bad_file(tmp_path):
    """Garbage bytes are reported, not silently parsed."""
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"this is not safetensors at all")
    with pytest.raises(ExportError, match="not a safetensors"):
        _read_safetensors(bad)


def test_read_safetensors_unknown_dtype(tmp_path):
    """A dtype tag we don't map raises instead of guessing."""
    p = tmp_path / "x.safetensors"
    header = {
        "w": {
            "dtype": "FROBNICATED",
            "shape": [1],
            "data_offsets": [0, 1],
        }
    }
    head = json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(head)) + head + b"\x00")
    with pytest.raises(ExportError, match="unsupported"):
        _read_safetensors(p)


# ---------------------------------------------------------------------------
#  fmt="torchscript"
# ---------------------------------------------------------------------------


def test_torchscript_trace_roundtrip(optimized_stack, tmp_path):
    """jit.trace handles the IRModule-grafted stack and round-trips
    exactly — a second runnable artifact format."""
    model, opt, x = optimized_stack
    out = tmp_path / "opt.ts"

    manifest = export_optimized(
        model, opt, out, fmt="torchscript", example_input=x
    )
    assert manifest["format"] == "torchscript"
    assert manifest["verify"]["max_abs_diff"] == 0.0

    loaded = load_optimized(out)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(x.clone()), opt(x.clone()), rtol=0, atol=0
        )


def test_torchscript_script_without_input(tmp_path):
    """With no example_input, plain modules still script — verify is
    honestly marked skipped."""
    mod = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    manifest = export_optimized(
        None, mod, tmp_path / "s.pt", fmt="torchscript"
    )
    assert manifest["verify"]["kind"] == "script"
    loaded = load_optimized(tmp_path / "s.pt")
    x = torch.randn(2, 4)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), mod(x))


def test_torchscript_script_failure_is_reported(
    optimized_stack, tmp_path
):
    """IRModule's ``*xs`` forward is unscriptable — the failure is a
    clear ExportError telling the caller to pass example_input."""
    _, opt, _ = optimized_stack
    with pytest.raises(ExportError, match="script failed"):
        export_optimized(
            None, opt, tmp_path / "f.pt", fmt="torchscript"
        )


def test_torchscript_trace_failure_is_reported(
    optimized_stack, tmp_path, monkeypatch
):
    _, opt, x = optimized_stack

    def boom(*_a, **_k):
        raise RuntimeError("trace exploded")

    monkeypatch.setattr(torch.jit, "trace", boom)
    with pytest.raises(ExportError, match="trace failed"):
        export_optimized(
            None,
            opt,
            tmp_path / "f.pt",
            fmt="torchscript",
            example_input=x,
        )


def test_torchscript_kwargs_need_input(tmp_path):
    """example_kwargs without example_input can't drive a trace."""
    mod = nn.Linear(4, 4).eval()
    with pytest.raises(ExportError, match="example_kwargs"):
        export_optimized(
            None,
            mod,
            tmp_path / "k.pt",
            fmt="torchscript",
            example_kwargs={"x": torch.randn(2, 4)},
        )


# ---------------------------------------------------------------------------
#  whole-model path + misc API
# ---------------------------------------------------------------------------


def test_optimize_model_output_exports_to_pt2(tmp_path):
    """The monolithic optimize_model executor also exports — the
    .pt2 path isn't compositional-only."""
    torch.manual_seed(0)
    mod = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
    mod = mod.eval()
    x = torch.randn(4, 8)
    opt, _ = Optimizer(backend=TorchBackend()).optimize(mod, x, verify=False, verbose=False)

    manifest = export_optimized(
        mod, opt, tmp_path / "om.pt2", example_input=x
    )
    assert manifest["verify"]["max_abs_diff"] == 0.0
    loaded = load_optimized(tmp_path / "om.pt2")
    with torch.no_grad():
        torch.testing.assert_close(loaded(x.clone()), opt(x.clone()))


def test_save_optimized_alias(optimized_stack, tmp_path):
    """save_optimized is export_optimized under a call-site name."""
    _, opt, x = optimized_stack
    manifest = save_optimized(
        None, opt, tmp_path / "al.pt2", example_input=x
    )
    assert manifest["format"] == "pt2"


def test_unknown_format_error(tmp_path):
    """Garbage format names list the supported set."""
    mod = nn.Linear(4, 4).eval()
    with pytest.raises(ExportError, match="unsupported export format"):
        export_optimized(None, mod, tmp_path / "x", fmt="onnx")


def test_original_absent_marked(optimized_stack, tmp_path):
    """Passing an unrelated ``model`` marks grafts' originals
    '<absent>' rather than crashing."""
    _, opt, x = optimized_stack
    manifest = export_optimized(
        nn.Module(), opt, tmp_path / "o.pt2", example_input=x
    )
    assert (
        manifest["catopt_modules"]["blocks.0"]["original"] == "<absent>"
    )


def test_verify_false_skips_reload(tmp_path):
    """verify=False writes the artifact without a round-trip check —
    for runnable and weights formats alike."""
    mod = nn.Linear(4, 4).eval()
    manifest = export_optimized(
        None,
        mod,
        tmp_path / "nv.pt2",
        example_input=torch.randn(2, 4),
        verify=False,
    )
    assert "verify" not in manifest
    assert (tmp_path / "nv.pt2").exists()

    ts = export_optimized(
        None,
        mod,
        tmp_path / "nv.pt",
        fmt="torchscript",
        example_input=torch.randn(2, 4),
        verify=False,
    )
    assert "verify" not in ts

    st = export_optimized(
        None,
        mod,
        tmp_path / "nv.safetensors",
        fmt="safetensors",
        verify=False,
    )
    assert "verify" not in st
    assert load_optimized(tmp_path / "nv.safetensors")


def test_load_optimized_manifest_unknown_format(tmp_path):
    """A manifest naming an unrecognised format is an ExportError."""
    p = tmp_path / "w.weights"
    p.write_bytes(b"xx")
    _manifest_file(p).write_text(json.dumps({"format": "bogus"}))
    with pytest.raises(ExportError, match="unknown format"):
        load_optimized(p)


def test_load_optimized_suffix_inference(tmp_path):
    """No manifest → the suffix decides; .pt tries TorchScript before
    falling back to torch.load."""
    mod = nn.Linear(4, 4).eval()

    # .pth → torch.load dict
    pth = tmp_path / "w.pth"
    torch.save(mod.state_dict(), pth)
    sd = load_optimized(pth)
    assert torch.equal(sd["weight"], mod.weight)

    # .pt TorchScript (no manifest) → jit.load branch
    ts_p = tmp_path / "m.pt"
    torch.jit.save(torch.jit.script(mod), str(ts_p))
    loaded = load_optimized(ts_p)
    x = torch.randn(1, 4)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), mod(x))

    # .pt plain torch.save (no manifest) → jit fails → torch.load
    raw = tmp_path / "raw.pt"
    torch.save(mod.state_dict(), raw)
    sd2 = load_optimized(raw)
    assert torch.equal(sd2["bias"], mod.bias)

    # Unrecognised suffix, no manifest
    with pytest.raises(ExportError, match="cannot infer"):
        load_optimized(tmp_path / "m.bin")


def test_load_optimized_suffix_pt2_and_safetensors(
    optimized_stack, tmp_path
):
    """Suffix inference for .pt2/.safetensors when the manifest is
    deleted."""
    _, opt, x = optimized_stack
    pt2 = tmp_path / "m.pt2"
    export_optimized(None, opt, pt2, example_input=x)
    _manifest_file(pt2).unlink()
    loaded = load_optimized(pt2)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x.clone()), opt(x.clone()))

    st = tmp_path / "w.safetensors"
    export_optimized(None, opt, st, fmt="safetensors")
    _manifest_file(st).unlink()
    weights = load_optimized(st)
    assert "blocks.0" not in weights  # keys are full state_dict paths
    assert any(k.startswith("blocks.0.") for k in weights)
