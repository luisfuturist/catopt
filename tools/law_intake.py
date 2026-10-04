"""Workload intake — feed the census programs nobody wrote by hand.

The corpus is hand-curated: forty ``catopt_torch.models`` classes
registered in ``law_impact._model_cases``.  ``law-workload-gen.md``
proved resampling cannot mint new op-tuples and
``corpus-expansion-r2.md`` proved real architectures can — but every
expansion round meant writing a new builder by hand and registering
it in code.  The missing half of self-play is an **intake path**: a
tool that takes a ``torch.nn.Module`` + example input, runs the real
export boundary (``torch.export`` → ``export_to_ir``), and appends
whatever survives to the census — recording, just as honestly, what
the boundary rejects.

What counts as a candidate (``candidates()``):

* **torch-native modules** — ``nn.MultiheadAttention``,
  ``nn.TransformerEncoder`` stacks, ``nn.BatchNorm``,
  ``nn.Embedding``, the RNN family, pooling, losses, ….  Library
  code spells ops differently than our hand builders (``permute``,
  ``unflatten``, ``split_with_sizes``, ``feature_dropout``); every
  spelling the bridge cannot lower is a recorded finding, not a
  crash.
* **compound models** — hand-built but realistic assemblies (a
  shared-expert MoE, a ViT patch block, a bottleneck residual, a
  classifier loss head) at real depth.

Every candidate lands in exactly one class:

* **ingested** — exported, every op bound, and the lowered module
  verifies fp64 against the original.  These join the firing/reach
  probe under a node cap — deeper stacks still feed the census and
  the matchers.
* **census-only** — exported but at least one op has no torch
  binding, or the lowered module fails verification.  The term still
  feeds the census (a real program's shape is a real shape); the
  missing ops are the binding-gap backlog this tool exists to
  surface.
* **rejected** — ``torch.export`` / ``export_to_ir`` raised; the
  exception is recorded verbatim.

The census seam is a side-file, not a code edit: a run emits
``tools/intake_corpus.json`` (terms via ``catopt_core.ir.term_to_data``
plus per-workload metadata and the rejection backlog) and
``tools/intake_tensors.pt`` (the feeds/params ``torch.save``'d, so a
loaded term lowers and verifies exactly like an exported model).
``law_shape_census.corpus()`` unions :func:`load_cases` when the
file exists — the file IS the appended census — and
``law_pipeline.run_pipeline`` reads the same file for its matcher
terms and (via :func:`probe_cases`) its firing probe.  Absent the
file, nothing changes: the baseline corpus is untouched.

Run::

    .venv/bin/python tools/law_intake.py            # ingest + report
    .venv/bin/python tools/law_intake.py --skip-pipeline
    .venv/bin/python tools/law_intake.py --json /tmp/intake.json
    .venv/bin/python tools/law_intake.py --holdout select_mul

CPU-only, a few minutes when the pipeline delta runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from catopt_core.ir import (
    Op,
    TensorType,
    Var,
    term_from_data,
    term_to_data,
)
from catopt_torch.adapters import TorchSink
from catopt_torch.torch_bridge import export_to_ir

sys.path.insert(0, str(Path(__file__).resolve().parent))

from law_impact import (
    TermCase,
    _bench_cases,
    _ir_of,
    _iter_subterms,
    model_cases,
)

__all__ = [
    "Rejection",
    "Workload",
    "candidates",
    "census_delta",
    "ingest",
    "load_cases",
    "load_records",
    "main",
    "probe_cases",
    "write_intake",
]

#: The default side-file paths — the appended census + its tensors.
_DEFAULT_JSON = Path(__file__).with_name("intake_corpus.json")
_DEFAULT_TENSORS = Path(__file__).with_name("intake_tensors.pt")

#: Verify tolerance for the lower-the-term-vs-original-model check.
_RTOL = 1e-4

#: Node cap for the firing/reach probe.  Deep stacks (the 105-node
#: TransformerEncoder(3), the 271-node full Transformer) still feed
#: the census and the matchers; the probe stays comparable in cost
#: to the corpus's own models (~80 nodes).
_PROBE_MAX_NODES = 120


# ---------------------------------------------------------------------------
#  Compound candidates — realistic assemblies, not one-op leaves
# ---------------------------------------------------------------------------


class _SharedExpertMoE(nn.Module):
    """DeepSeek-style routed MLP with a shared expert.

    An always-on shared expert plus a softmax-gated top-k dispatch
    over routed experts.
    """

    def __init__(self, d: int, hidden: int, n_exp: int) -> None:
        """Build the gate, the shared expert and the routed bank."""
        super().__init__()
        self.gate = nn.Linear(d, n_exp)
        self.shared = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, d)
        )
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d, hidden),
                nn.SiLU(),
                nn.Linear(hidden, d),
            )
            for _ in range(n_exp)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Combine the shared expert with weight-masked experts."""
        w = torch.softmax(self.gate(x), dim=-1)
        _, idx = torch.topk(w, 2, dim=-1)
        out = self.shared(x)
        for i, e in enumerate(self.experts):
            mask = (idx == i).any(dim=-1, keepdim=True).to(x.dtype)
            out = out + w[:, i : i + 1] * mask * e(x)
        return out


class _ViTPatchBlock(nn.Module):
    """ViT front end: strided-conv patchify, flatten, encoder layer."""

    def __init__(self, ch: int, d: int, heads: int) -> None:
        """Build the patch projection and the encoder layer."""
        super().__init__()
        self.patch = nn.Conv2d(ch, d, 4, 4)
        self.enc = nn.TransformerEncoderLayer(
            d, heads, 2 * d, batch_first=True, dropout=0.0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify then encode — the ``flatten``/``transpose`` spine."""
        t = self.patch(x)
        return self.enc(t.flatten(2).transpose(1, 2))


class _TinyCNN(nn.Module):
    """VGG-ish conv->bn->relu->pool stack at real depth."""

    def __init__(self, ch: int) -> None:
        """Build the two conv+norm stages and the linear head."""
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)
        self.head = nn.Linear(ch, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """conv-bn-relu-pool x2, adaptive pool, flatten, linear."""
        x = F.relu(self.b1(self.c1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.b2(self.c2(x)))
        return self.head(F.adaptive_avg_pool2d(x, 1).flatten(1))


class _LossHead(nn.Module):
    """A training-step forward graph: body plus the CE loss itself."""

    def __init__(self, d: int, ncls: int) -> None:
        """Build the MLP body and the loss module."""
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, ncls)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the scalar loss — a real two-input graph."""
        return self.loss(self.body(x), y)


class _ManualNLL(nn.Module):
    """``log_softmax`` + ``nll_loss`` spelled by hand."""

    def __init__(self, d: int, ncls: int) -> None:
        """Build the classifier head."""
        super().__init__()
        self.head = nn.Linear(d, ncls)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return NLL of the log-softmaxed logits."""
        return F.nll_loss(F.log_softmax(self.head(x), dim=-1), y)


class _Bottleneck(nn.Module):
    """ResNet bottleneck: 1x1 reduce, 3x3, 1x1 expand, plus skip."""

    def __init__(self, ch: int, mid: int) -> None:
        """Build the three conv+norm stages."""
        super().__init__()
        self.c1 = nn.Conv2d(ch, mid, 1)
        self.b1 = nn.BatchNorm2d(mid)
        self.c2 = nn.Conv2d(mid, mid, 3, padding=1)
        self.b2 = nn.BatchNorm2d(mid)
        self.c3 = nn.Conv2d(mid, ch, 1)
        self.b3 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce-compute-expand with a residual add."""
        y = F.relu(self.b1(self.c1(x)))
        y = F.relu(self.b2(self.c2(y)))
        return F.relu(self.b3(self.c3(y)) + x)


class _LogSoftmaxHead(nn.Module):
    """The canonical classifier tail: linear then ``log_softmax``."""

    def __init__(self, d: int, ncls: int) -> None:
        """Build the linear head."""
        super().__init__()
        self.head = nn.Linear(d, ncls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities."""
        return F.log_softmax(self.head(x), dim=-1)


# ---------------------------------------------------------------------------
#  The candidate registry — thunks so import constructs nothing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Workload:
    """One intake candidate: a name and a ``(model, input)`` thunk."""

    name: str
    build: Callable[[], tuple[nn.Module, Any]]
    kind: str = "torch-native"


def _r(*shape: int) -> torch.Tensor:
    """Return a fresh fp64 CPU example input."""
    return torch.randn(*shape, dtype=torch.float64)


def _enc_layer(d: int) -> nn.Module:
    """Return one batch-first encoder layer at the intake size."""
    return nn.TransformerEncoderLayer(
        d, 4, 2 * d, batch_first=True, dropout=0.0
    )


def candidates() -> list[Workload]:
    """Return the intake registry — torch natives + compound models.

    Thunks build a fresh module and example input per call, so the
    registry is cheap to import and every ingestion sees the same
    seeded weights the report records.
    """
    d = 16
    return [
        # --- torch-native: attention / transformer family -----------
        Workload(
            "nn.MultiheadAttention",
            lambda: (
                nn.MultiheadAttention(d, 4, batch_first=True),
                (_r(2, 8, d), _r(2, 8, d), _r(2, 8, d)),
            ),
        ),
        Workload(
            "nn.TransformerEncoderLayer",
            lambda: (_enc_layer(d), _r(2, 8, d)),
        ),
        Workload(
            "nn.TransformerEncoder(d=3)",
            lambda: (
                nn.TransformerEncoder(_enc_layer(d), 3),
                _r(2, 8, d),
            ),
        ),
        Workload(
            "nn.TransformerDecoderLayer",
            lambda: (
                nn.TransformerDecoderLayer(
                    d, 4, 2 * d, batch_first=True, dropout=0.0
                ),
                (_r(2, 8, d), _r(2, 8, d)),
            ),
        ),
        # --- torch-native: norms ------------------------------------
        Workload(
            "nn.BatchNorm1d", lambda: (nn.BatchNorm1d(d), _r(4, d))
        ),
        Workload(
            "nn.BatchNorm2d",
            lambda: (nn.BatchNorm2d(8), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.GroupNorm", lambda: (nn.GroupNorm(4, 8), _r(1, 8, 8, 8))
        ),
        Workload("nn.LayerNorm", lambda: (nn.LayerNorm(d), _r(4, d))),
        Workload(
            "nn.InstanceNorm2d",
            lambda: (nn.InstanceNorm2d(8), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.LocalResponseNorm",
            lambda: (nn.LocalResponseNorm(4), _r(1, 8, 8, 8)),
        ),
        # --- torch-native: conv / pooling / reshaping ---------------
        Workload(
            "nn.ConvTranspose2d",
            lambda: (nn.ConvTranspose2d(8, 8, 3), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Conv3d",
            lambda: (nn.Conv3d(4, 4, 3), _r(1, 4, 4, 4, 4)),
        ),
        Workload(
            "nn.MaxPool2d", lambda: (nn.MaxPool2d(2), _r(1, 8, 8, 8))
        ),
        Workload(
            "nn.AdaptiveAvgPool2d",
            lambda: (nn.AdaptiveAvgPool2d(1), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.PixelShuffle",
            lambda: (nn.PixelShuffle(2), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Upsample(nearest)",
            lambda: (nn.Upsample(scale_factor=2), _r(1, 8, 8, 8)),
        ),
        Workload(
            "nn.Upsample(bilinear)",
            lambda: (
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=False
                ),
                _r(1, 8, 8, 8),
            ),
        ),
        Workload("nn.Unfold", lambda: (nn.Unfold(2), _r(1, 8, 8, 8))),
        Workload(
            "nn.Fold",
            lambda: (nn.Fold((8, 8), (2, 2)), _r(1, 8 * 4, 49)),
        ),
        Workload(
            "nn.Dropout2d", lambda: (nn.Dropout2d(0.3), _r(1, 8, 8, 8))
        ),
        # --- torch-native: embeddings / pairwise / losses -----------
        Workload(
            "nn.Embedding",
            lambda: (nn.Embedding(32, d), torch.randint(0, 32, (2, 8))),
        ),
        Workload(
            "nn.Bilinear",
            lambda: (nn.Bilinear(d, d, d), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CosineSimilarity",
            lambda: (nn.CosineSimilarity(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CrossEntropyLoss",
            lambda: (
                nn.CrossEntropyLoss(),
                (_r(4, 8), torch.randint(0, 8, (4,))),
            ),
        ),
        Workload(
            "nn.MSELoss",
            lambda: (nn.MSELoss(), (_r(4, d), _r(4, d))),
        ),
        Workload(
            "nn.CTCLoss",
            lambda: (
                nn.CTCLoss(),
                (
                    F.log_softmax(_r(8, 2, 6), -1),
                    torch.randint(1, 6, (2, 3)),
                    torch.tensor([8, 8]),
                    torch.tensor([3, 3]),
                ),
            ),
        ),
        # --- torch-native: recurrent family -------------------------
        Workload(
            "nn.LSTM",
            lambda: (nn.LSTM(8, 8, batch_first=True), _r(2, 6, 8)),
        ),
        Workload(
            "nn.GRU",
            lambda: (nn.GRU(8, 8, batch_first=True), _r(2, 6, 8)),
        ),
        Workload(
            "nn.RNNCell",
            lambda: (nn.RNNCell(8, 8), (_r(2, 8), _r(2, 8))),
        ),
        # --- torch-native: assorted pointwise -----------------------
        Workload("nn.PReLU", lambda: (nn.PReLU(), _r(4, d))),
        Workload("nn.Mish", lambda: (nn.Mish(), _r(4, d))),
        Workload("nn.Hardswish", lambda: (nn.Hardswish(), _r(4, d))),
        # --- compound models ----------------------------------------
        Workload(
            "SharedExpertMoE",
            lambda: (_SharedExpertMoE(d, 8, 4), _r(4, d)),
            kind="compound",
        ),
        Workload(
            "ViTPatchBlock",
            lambda: (_ViTPatchBlock(8, d, 4), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "TinyCNN",
            lambda: (_TinyCNN(8), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "Bottleneck",
            lambda: (_Bottleneck(8, 4), _r(1, 8, 8, 8)),
            kind="compound",
        ),
        Workload(
            "LossHead",
            lambda: (
                _LossHead(d, 8),
                (_r(4, d), torch.randint(0, 8, (4,))),
            ),
            kind="compound",
        ),
        Workload(
            "ManualNLL",
            lambda: (
                _ManualNLL(d, 8),
                (_r(4, d), torch.randint(0, 8, (4,))),
            ),
            kind="compound",
        ),
        Workload(
            "LogSoftmaxHead",
            lambda: (_LogSoftmaxHead(d, 8), _r(4, d)),
            kind="compound",
        ),
    ]


# ---------------------------------------------------------------------------
#  Ingestion — export, classify, verify
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rejection:
    """A candidate the export boundary refused, with the reason."""

    name: str
    stage: str
    error: str


def _ops_of(term: Any) -> set[str]:
    """Return the distinct op names inside *term*."""
    return {s.op for s in _iter_subterms(term) if isinstance(s, Op)}


def _n_nodes(term: Any) -> int:
    """Return the op-node count of *term*."""
    return sum(1 for s in _iter_subterms(term) if isinstance(s, Op))


def _verify(
    model: nn.Module,
    ir: Any,
    tensors: dict,
    feed: tuple,
    sink: TorchSink,
) -> str:
    """Lower the exported IR and verify it against the module.

    Returns ``"pass"``, ``"FAIL"`` or an error tag — the same
    ``sink.lower`` + ``sink.verify`` path the corpus tests run.
    """
    try:
        lowered = sink.lower(
            _ir_of(ir.root, tuple(ir.inputs)), dict(tensors)
        )
        vr = sink.verify(model, lowered, feed, rtol=_RTOL)
    except Exception as e:
        return f"error:{type(e).__name__}: {e}"
    return "pass" if vr.passed else f"FAIL max_rel={vr.max_rel:.2e}"


def _ingest_one(
    cand: Workload, sink: TorchSink, supported: frozenset
) -> tuple[TermCase | None, dict | None, Rejection | None]:
    """Run one candidate through the boundary; classify the result."""
    try:
        model, x = cand.build()
    except Exception as e:
        return (
            None,
            None,
            Rejection(cand.name, "build", f"{type(e).__name__}: {e}"),
        )
    feed = x if isinstance(x, tuple) else (x,)
    try:
        ir, tensors = export_to_ir(model.eval().double(), feed)
    except Exception as e:
        return (
            None,
            None,
            Rejection(cand.name, "export", f"{type(e).__name__}: {e}"),
        )
    ops = _ops_of(ir.root)
    missing = sorted(ops - supported)
    rec: dict[str, Any] = {
        "name": f"intake:{cand.name}",
        "builder": cand.kind,
        "n_op_nodes": _n_nodes(ir.root),
        "ops": sorted(ops),
        "unsupported_ops": missing,
        "term": term_to_data(ir.root),
        "inputs": [
            {"name": v.name, "shape": list(v.typ.shape)}
            for v in ir.inputs
        ],
    }
    if missing:
        rec["status"] = "census-only"
        rec["verify"] = "skipped"
        rec["verify_note"] = f"unbound ops: {missing}"
    else:
        v = _verify(model, ir, tensors, feed, sink)
        rec["verify"] = v
        rec["status"] = "ingested" if v == "pass" else "verify-failed"
        rec["verify_note"] = "" if v == "pass" else v
    case = TermCase(
        source="intake",
        name=f"intake:{cand.name}",
        term=ir.root,
        inputs=tuple(ir.inputs),
        feed=tuple(feed),
        param_vals=dict(tensors),
    )
    return case, rec, None


def ingest(
    cands: list[Workload] | None = None,
    sink: TorchSink | None = None,
) -> tuple[list[TermCase], list[dict], list[Rejection]]:
    """Run every candidate through the export boundary.

    Returns ``(cases, records, rejections)``: the ``TermCase``s of
    every term that exported (any status), the per-workload metadata
    records the side-file persists, and the hard rejections.
    """
    sink = sink or TorchSink()
    supported = sink.supported_ops
    cases: list[TermCase] = []
    records: list[dict] = []
    rejections: list[Rejection] = []
    for cand in cands if cands is not None else candidates():
        case, rec, rej = _ingest_one(cand, sink, supported)
        if rej is not None:
            rejections.append(rej)
            continue
        cases.append(case)
        records.append(rec)
    return cases, records, rejections


# ---------------------------------------------------------------------------
#  Persistence — the side-file census
# ---------------------------------------------------------------------------


def write_intake(
    records: list[dict],
    rejections: list[Rejection],
    tensors_of: dict[str, dict],
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
) -> None:
    """Write the census contribution and the tensor blob.

    ``records`` carry the serialized terms; ``tensors_of`` maps each
    workload name to ``{"feed": [...], "params": {...}}`` — kept in a
    ``torch.save`` blob so the JSON stays diffable.
    """
    payload = {
        "format": 1,
        "tool": "law_intake",
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "workloads": records,
        "rejections": [asdict(r) for r in rejections],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    torch.save(tensors_of, tensors_path)


def load_records(path: Path = _DEFAULT_JSON) -> list[dict]:
    """Return the persisted workload records ([] when absent)."""
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("workloads", [])


def load_cases(
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
) -> list[TermCase]:
    """Rebuild intake ``TermCase``s from the side-file census.

    Terms decode through ``term_from_data``; feeds/params ride the
    companion ``torch.save`` blob (absent blob → empty feed — the
    census only reads ``.term`` anyway, and :func:`probe_cases`
    requires a feed).
    """
    tensor_blob: dict = {}
    if tensors_path.exists():
        tensor_blob = torch.load(tensors_path, weights_only=True)
    cases: list[TermCase] = []
    for rec in load_records(path):
        inputs = tuple(
            Var(i["name"], TensorType(tuple(i["shape"])))
            for i in rec["inputs"]
        )
        blob = tensor_blob.get(rec["name"], {})
        cases.append(
            TermCase(
                source="intake",
                name=rec["name"],
                term=term_from_data(rec["term"]),
                inputs=inputs,
                feed=tuple(blob.get("feed", ())),
                param_vals=dict(blob.get("params", {})),
            )
        )
    return cases


def probe_cases(
    path: Path = _DEFAULT_JSON,
    tensors_path: Path = _DEFAULT_TENSORS,
    max_nodes: int = _PROBE_MAX_NODES,
) -> list[TermCase]:
    """Return the intake cases eligible for the firing/reach probe.

    Eligibility is the intake contract: status ``"ingested"`` (all
    ops bound, lowered module verified), a persisted feed, and under
    the node cap that keeps the probe's saturation cost comparable
    to the corpus's own models.
    """
    status = {r["name"]: r["status"] for r in load_records(path)}
    nodes = {r["name"]: r["n_op_nodes"] for r in load_records(path)}
    return [
        c
        for c in load_cases(path, tensors_path)
        if status.get(c.name) == "ingested"
        and c.feed
        and nodes.get(c.name, 0) <= max_nodes
    ]


def _tensors_of(cases: list[TermCase]) -> dict[str, dict]:
    """Collect each case's feed/param tensors for ``torch.save``."""
    return {
        c.name: {"feed": list(c.feed), "params": dict(c.param_vals)}
        for c in cases
    }


# ---------------------------------------------------------------------------
#  Census delta — what the intake adds that the corpus did not have
# ---------------------------------------------------------------------------


def census_delta(base: list[TermCase], intake: list[TermCase]) -> dict:
    """Diff the intake terms' census keys against the base corpus.

    Returns the new op-tuples, new shapes and previously-absent ops,
    plus a per-workload novelty table for the report.
    """
    from law_shape_census import (
        CorpusTerm,
        op_tuple_census,
        shape_census,
    )

    base_ct = [CorpusTerm(c.source, c.name, c.term) for c in base]
    in_ct = [CorpusTerm(c.source, c.name, c.term) for c in intake]
    base_op, _ = op_tuple_census(base_ct)
    base_sh, _ = shape_census(base_ct)
    in_op, _ = op_tuple_census(in_ct)
    in_sh, _ = shape_census(in_ct)
    base_ops = set()
    for c in base:
        base_ops |= _ops_of(c.term)
    in_ops = set()
    for c in intake:
        in_ops |= _ops_of(c.term)
    per_case = {}
    for c in intake:
        ct = CorpusTerm(c.source, c.name, c.term)
        op_c, _ = op_tuple_census([ct])
        sh_c, _ = shape_census([ct])
        per_case[c.name] = (
            sorted(set(op_c) - set(base_op)),
            len(set(sh_c) - set(base_sh)),
        )
    return {
        "n_base_tuples": len(base_op),
        "n_base_shapes": len(base_sh),
        "new_op_tuples": sorted(set(in_op) - set(base_op)),
        "new_shapes": len(set(in_sh) - set(base_sh)),
        "new_ops": sorted(in_ops - base_ops),
        "per_case": per_case,
    }


# ---------------------------------------------------------------------------
#  Pipeline delta — mirror law_workload_gen's harness verbatim
# ---------------------------------------------------------------------------


def _run_delta(
    base_cases: list[TermCase],
    probe_base: list[TermCase],
    intake_cases: list[TermCase],
    probe_intake: list[TermCase],
    vocab: str,
    holdout: str | None,
) -> dict:
    """Run the pipeline on the corpus, then corpus+intake.

    Reuses ``law_workload_gen._run_pipeline`` — the same census ->
    propose -> measure -> rank mirror that tool validated — with the
    intake cases as the corpus extension and the probe-eligible
    subset as the firing/reach additions.
    """
    import law_workload_gen as lwg

    base = lwg._run_pipeline(base_cases, probe_base, vocab, holdout)
    big = lwg._run_pipeline(
        [*base_cases, *intake_cases],
        [*probe_base, *probe_intake],
        vocab,
        holdout,
    )
    return {"baseline": base, "enlarged": big}


def _intake_fires(ev: Any) -> int:
    """Count an evidence row's firings on intake cases."""
    return sum(1 for c in ev.fire_cases if c.startswith("intake:"))


def _delta_table(res: dict) -> str:
    """Render baseline-vs-enlarged pipeline comparison."""
    b, e = res["baseline"], res["enlarged"]
    b_names = {ev.proposal.name for ev in b["ranked"]}
    e_names = {ev.proposal.name for ev in e["ranked"]}
    b_rank = {ev.proposal.name: ev for ev in b["ranked"]}
    e_rank = {ev.proposal.name: ev for ev in e["ranked"]}
    lines = [
        f"corpus: {b['n_terms']} -> {e['n_terms']} terms, "
        f"{b['n_tuples']} -> {e['n_tuples']} op-tuples",
        f"proposals: {len(b_names)} -> {len(e_names)}",
    ]
    new_props = sorted(e_names - b_names)
    lines.append(f"new proposals ({len(new_props)}):")
    for name in new_props:
        ev = e_rank[name]
        lines.append(
            f"  {name:<28} [{ev.proposal.family}] "
            f"match={ev.matches} fires={ev.fires} "
            f"(intake={_intake_fires(ev)}) paid={ev.paid} "
            f"ship={'SHIP' if ev.shippable else ev.no_ship_reason}"
        )
    if not new_props:
        lines.append("  (none)")
    lines.append("-- firing/paid deltas on shared proposals --")
    shown = 0
    for name in sorted(b_names & e_names):
        be, ee = b_rank[name], e_rank[name]
        extra = _intake_fires(ee)
        if not (
            extra or ee.fires != be.fires or ee.matches != be.matches
        ):
            continue
        lines.append(
            f"  {name:<28} match {be.matches}->{ee.matches} "
            f"fires {be.fires}->{ee.fires} (intake={extra}) "
            f"paid {be.paid}->{ee.paid} "
            f"ship: {'Y' if ee.shippable else ee.no_ship_reason}"
        )
        shown += 1
    if not shown:
        lines.append("  (no shared proposal changed)")
    b_ship = [ev.proposal.name for ev in b["ranked"] if ev.shippable]
    e_ship = [ev.proposal.name for ev in e["ranked"] if ev.shippable]
    lines.append(
        f"shippable: {len(b_ship)} -> {len(e_ship)}"
        + (
            f"  new: {[s for s in e_ship if s not in b_ship]}"
            if e_ship
            else ""
        )
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Reporting + driver
# ---------------------------------------------------------------------------


def _status_table(records: list[dict], delta: dict) -> str:
    """Render the per-workload intake table."""
    head = (
        f"{'workload':<36} {'kind':<12} {'nodes':>6} {'status':<13} "
        f"{'+tup':>5} {'+shp':>5}  missing ops"
    )
    lines = [head, "-" * len(head)]
    for rec in records:
        nt, ns = delta["per_case"].get(rec["name"], ([], 0))
        lines.append(
            f"{rec['name']:<36} {rec['builder']:<12} "
            f"{rec['n_op_nodes']:>6} {rec['status']:<13} "
            f"{len(nt):>5} {ns:>5}  "
            f"{', '.join(rec['unsupported_ops']) or '-'}"
        )
    return "\n".join(lines)


def _rejection_table(rejections: list[Rejection]) -> str:
    """Render the export-boundary rejection backlog."""
    lines = [f"{'workload':<36} {'stage':<8} error"]
    lines.append("-" * 80)
    for r in rejections:
        lines.append(f"{r.name:<36} {r.stage:<8} {r.error}")
    return "\n".join(lines)


def _gap_table(records: list[dict]) -> str:
    """Aggregate unbound ops -> the workloads that need them."""
    gaps: dict[str, list[str]] = {}
    for rec in records:
        for op in rec["unsupported_ops"]:
            gaps.setdefault(op, []).append(rec["name"])
    if not gaps:
        return "  (no binding gaps)"
    lines = [f"{'missing op':<26} {'#':>3}  needed by"]
    lines.append("-" * 80)
    for op, names in sorted(gaps.items()):
        lines.append(f"{op:<26} {len(names):>3}  {', '.join(names)}")
    return "\n".join(lines)


def _eligible(cases: list[TermCase], records: list[dict]) -> list:
    """Return the in-memory probe-eligible cases (ingested + cap)."""
    meta = {r["name"]: (r["status"], r["n_op_nodes"]) for r in records}
    return [
        c
        for c in cases
        if meta.get(c.name, ("", 0))[0] == "ingested"
        and meta[c.name][1] <= _PROBE_MAX_NODES
    ]


def _report_pipeline(
    cases: list[TermCase],
    records: list[dict],
    args: argparse.Namespace,
    payload: dict,
) -> None:
    """Run the baseline-vs-enlarged pipeline delta and report it."""
    bench, _be = _bench_cases()
    models, _me = model_cases()
    probe = _eligible(cases, records)
    print(f"   probe-eligible intake cases: {len(probe)}")
    res = _run_delta(
        [*bench, *models],
        models,
        cases,
        probe,
        args.vocab,
        args.holdout,
    )
    print(_delta_table(res))
    e = res["enlarged"]
    payload["pipeline"] = {
        "baseline_terms": res["baseline"]["n_terms"],
        "enlarged_terms": e["n_terms"],
        "baseline_tuples": res["baseline"]["n_tuples"],
        "enlarged_tuples": e["n_tuples"],
        "new_proposals": sorted(
            {ev.proposal.name for ev in e["ranked"]}
            - {ev.proposal.name for ev in res["baseline"]["ranked"]}
        ),
        "shippable": [
            ev.proposal.name for ev in e["ranked"] if ev.shippable
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Ingest candidates, measure the census delta, write the file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_JSON,
        help="side-file census (default tools/intake_corpus.json)",
    )
    parser.add_argument(
        "--tensors",
        type=Path,
        default=_DEFAULT_TENSORS,
        help="feed/param tensor blob (tools/intake_tensors.pt)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="run the measurement without writing the side files",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--vocab", choices=("hand", "derived"), default="derived"
    )
    parser.add_argument("--holdout", help="pipeline holdout rules")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--json", help="write machine-readable results")
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    print("== law_intake — real workloads into the census ==")
    sink = TorchSink()
    cases, records, rejections = ingest(sink=sink)
    n_in = sum(1 for r in records if r["status"] == "ingested")
    n_co = sum(1 for r in records if r["status"] == "census-only")
    n_vf = sum(1 for r in records if r["status"] == "verify-failed")
    print(
        f"   {len(records)} exported: {n_in} ingested, "
        f"{n_co} census-only, {n_vf} verify-failed; "
        f"{len(rejections)} rejected"
    )

    bench, _be = _bench_cases()
    models, _me = model_cases()
    delta = census_delta([*bench, *models], cases)
    print()
    print("-- intake table --")
    print(_status_table(records, delta))
    print()
    print(
        f"-- census delta (intake vs {len(bench)} bench + "
        f"{len(models)} models) --"
    )
    print(
        f"   {delta['n_base_tuples']} -> "
        f"{delta['n_base_tuples'] + len(delta['new_op_tuples'])} "
        f"op-tuples, +{delta['new_shapes']} shapes, "
        f"{len(delta['new_ops'])} new ops: {delta['new_ops']}"
    )
    print()
    print("-- binding-gap backlog (exported, unbound) --")
    print(_gap_table(records))
    print()
    print("-- export rejections --")
    print(_rejection_table(rejections))
    print()

    payload: dict[str, Any] = {
        "n_exported": len(records),
        "n_ingested": n_in,
        "n_census_only": n_co,
        "n_verify_failed": n_vf,
        "rejections": [asdict(r) for r in rejections],
        "census_delta": {
            "new_op_tuples": [
                f"{k[0]}({', '.join(k[1])})"
                for k in delta["new_op_tuples"]
            ],
            "new_shapes": delta["new_shapes"],
            "new_ops": delta["new_ops"],
        },
    }

    if not args.no_write:
        write_intake(
            records,
            rejections,
            _tensors_of(cases),
            args.out,
            args.tensors,
        )
        print(f"wrote {args.out} + {args.tensors}")
    else:
        print("(--no-write: side files not written)")

    if not args.skip_pipeline:
        print()
        print("-- pipeline delta (baseline vs corpus+intake) --")
        _report_pipeline(cases, records, args, payload)

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
