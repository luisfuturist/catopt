"""Canonical per-op attribute schema — plan 0001 phase 1b.

torch.export spells trailing scalar arguments positionally:
``cat(ts, -2)`` exports as ``concat(arg1=-2)``, ``t.transpose(i, j)``
as ``transpose(arg1=i, arg2=j)``, ``F.sdpa(q, k, v, is_causal=True)``
as ``sdpa(arg4=0.0, arg5=True)``.  Everywhere downstream — rules,
typing, cost, the torch bindings — those spellings were defended by
``kw.get("arg1", kw.get("dim", ...))`` chains.  This module is the
single declaration that replaces the folklore: for each op, the
canonical name of every positional attribute.

``ATTR_SCHEMA[op][i]`` is the canonical attr name for the scalar
argument at position ``i`` of the aten call — the position torch
export's ``arg{i}`` reports.  The mapping was verified against real
``torch.export`` output (``x[..., ::2]`` → ``slice(arg1..arg4)``,
``F.layer_norm`` → ``arg4=eps``, ``arg5=cudnn_enabled`` — see
``tests/test_attrs.py``).

The contract has two halves:

* **Boundary** — ``export_to_ir`` names non-node args at declared
  positions by the schema at emission, so exported terms carry ONLY
  canonical spellings (``concat(dim=-2)``, ``sdpa(is_causal=True)``,
  ``transpose(dim0=1, dim1=2)``, ``slice(dim=2, start=0, end=…,
  step=2)``).
* **Mint** — ``Op.make`` *validates*: a positional ``argN`` at a
  position the schema does not declare, or a required attr missing
  from a fully-attributed term, is a ``ValueError`` — a rewrite
  minting a malformed term dies at mint time, not silently at eval.

  Minted ``argN`` at declared positions is PRESERVED, not renamed —
  the schema bounds WHICH positions may be spelled positionally, it
  does not forbid the spelling.  Every downstream *reader* and *rule
  matcher*, however, reads ONLY the canonical name (the dual-spelling
  ``argN`` fallbacks were collapsed here); a term minted with a bare
  positional therefore behaves as if that attr were absent.  The
  canonical spelling is the ONE spelling the system produces and
  consumes.

  When BOTH spellings arrive (``attrs["arg5"] = True`` patched onto a
  term already carrying ``is_causal`` — ``optimize._specialize_causal``
  does this), the positional write wins and lands under the canonical
  name: imperative attr patches must take effect.

``ATTR_REQUIRED`` marks canonical names a *fully-attributed* term must
supply: enforced only when the term carries at least one
schema-declared attr (either spelling satisfies a position) — a
zero-attr term or one carrying only non-declared extras (the
getitem-folded ``index``) is a partial/pattern term and passes.
``Op.make(..., validate=False)`` escapes entirely for callers that
deliberately mint non-canonical terms.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "ATTR_REQUIRED",
    "ATTR_SCHEMA",
    "attr_of",
    "is_positional_attr",
    "validate_attrs",
]

#: arg position (the ``N`` in torch.export's ``argN``) -> canonical
#: attr name.  Only *attribute* positions appear here — positions
#: carrying tensor operands are never attrs.  Every name is the ONE
#: canonical spelling the boundary emits and the readers consume; no
#: position declares an ``argN`` name (see the module docstring).
ATTR_SCHEMA: dict[str, dict[int, str]] = {
    # --- product structure -------------------------------------------------
    "concat": {1: "dim"},
    "stack": {1: "dim"},
    "unbind": {1: "dim"},
    "chunk": {1: "chunks", 2: "dim", 3: "index"},
    "split": {1: "sizes", 2: "dim", 3: "index"},
    "getitem": {1: "index"},
    # --- axis ops -----------------------------------------------------------
    # transpose(x, dim0, dim1) — the axes the rule patterns mint and
    # every reader (typing, the torch binding) reads.
    "transpose": {1: "dim0", 2: "dim1"},
    "unsqueeze": {1: "dim"},
    "squeeze": {1: "dim"},
    "softmax": {1: "dim"},
    "select": {1: "dim", 2: "index"},
    "flatten": {1: "start_dim", 2: "end_dim"},
    "unflatten": {1: "dim", 2: "sizes"},
    "index_select": {1: "dim", 2: "index"},
    "take_along_dim": {1: "dim"},
    "gather": {1: "dim"},
    # scatter(t, dim, index, src|value) — src/value stay operands;
    # scatter_add/index_add same layout minus the value overload.
    "scatter": {1: "dim"},
    "scatter_add": {1: "dim"},
    "scatter_reduce": {1: "dim", 4: "reduce"},
    "index_add": {1: "dim"},
    # index_put(t, [indices], values, accumulate) — indices land as
    # operands with a ``layout`` attr (see export_to_ir).
    "index_put": {3: "accumulate"},
    # slice_scatter(t, src, dim, start, end, step) — the
    # functionalized x[..., lo:hi] = y (KV-cache writes).
    "slice_scatter": {2: "dim", 3: "start", 4: "end", 5: "step"},
    "select_scatter": {2: "dim", 3: "index"},
    "narrow": {1: "dim", 2: "start", 3: "length"},
    # movedim(t, source, destination) — either may be int or a list.
    "movedim": {1: "source", 2: "destination"},
    "roll": {1: "shifts", 2: "dims"},
    "einsum": {0: "equation"},
    # --- reductions -----------------------------------------------------
    # argmax/prod/var/std/any/all share the (dim, keepdim) tail;
    # var/std additionally carry the unbiased ``correction``.
    "argmax": {1: "dim", 2: "keepdim"},
    "argmin": {1: "dim", 2: "keepdim"},
    # aten.max/.min carry the dim positionally too (values+indices
    # pair); amax/amin are the values-only spellings.  var_mean and
    # std_mean add the unbiased ``correction``.
    "max": {1: "dim", 2: "keepdim"},
    "min": {1: "dim", 2: "keepdim"},
    "amax": {1: "dim", 2: "keepdim"},
    "amin": {1: "dim", 2: "keepdim"},
    "var_mean": {1: "dim", 2: "correction", 3: "keepdim"},
    "std_mean": {1: "dim", 2: "correction", 3: "keepdim"},
    "prod": {1: "dim", 2: "keepdim"},
    "nansum": {1: "dim", 2: "keepdim"},
    "nanmean": {1: "dim", 2: "keepdim"},
    "var": {1: "dim", 2: "correction", 3: "keepdim"},
    "std": {1: "dim", 2: "correction", 3: "keepdim"},
    "any": {1: "dim", 2: "keepdim"},
    "all": {1: "dim", 2: "keepdim"},
    "count_nonzero": {1: "dim"},
    "cumsum": {1: "dim"},
    "cumprod": {1: "dim"},
    "cummax": {1: "dim"},
    "cummin": {1: "dim"},
    "logcumsumexp": {1: "dim"},
    "linalg_vector_norm": {1: "ord", 2: "dim", 3: "keepdim"},
    "log_softmax": {1: "dim"},
    "median": {1: "dim", 2: "keepdim"},
    "kthvalue": {1: "k", 2: "dim", 3: "keepdim"},
    "argsort": {1: "dim", 2: "descending"},
    "topk": {1: "k", 2: "dim", 3: "largest", 4: "sorted"},
    "sort": {1: "dim", 2: "descending"},
    # clamp(t, min, max) — naming the bounds keeps the optional-None
    # gap honest: clamp(x, None, hi) lands max=hi, not a Const operand
    # the binding would misread as min.
    "clamp": {1: "min", 2: "max"},
    "clamp_min": {1: "min"},
    "clamp_max": {1: "max"},
    "hardtanh": {1: "min", 2: "max"},
    "elu": {1: "alpha", 2: "scale", 3: "input_scale"},
    "softplus": {1: "beta", 2: "threshold"},
    "triu": {1: "diagonal"},
    "tril": {1: "diagonal"},
    "glu": {1: "dim"},
    # isclose(x, y, rtol, atol, equal_nan) — the tolerances and the
    # flag are semantic attrs the binding must see.
    "isclose": {2: "rtol", 3: "atol", 4: "equal_nan"},
    "searchsorted": {2: "out_int32", 3: "right"},
    "one_hot": {1: "num_classes"},
    "tensor_split": {1: "sections", 2: "dim"},
    "unfold": {1: "dim", 2: "size", 3: "step"},
    "pixel_shuffle": {1: "upscale_factor"},
    "pixel_unshuffle": {1: "downscale_factor"},
    # pad(t, pad_width_list, *, mode, value) — the width list lands
    # under ``pad``; mode/value are kwargs.
    "pad": {1: "pad", 2: "mode", 3: "value"},
    # aten.group_norm(x, num_groups, weight, bias, eps,
    # cudnn_enabled): same eps/cudnn tail as layer_norm (eps is arg4 —
    # NOT arg5, the cudnn flag).
    "group_norm": {1: "num_groups", 4: "eps", 5: "cudnn_enabled"},
    # batch_norm(x, w, b, rm, rv, training, momentum, eps, cudnn).
    "batch_norm": {
        5: "training",
        6: "momentum",
        7: "eps",
        8: "cudnn_enabled",
    },
    # aten.instance_norm(x, w, b, rm, rv, use_input_stats, momentum,
    # eps, cudnn) — positions 1-4 are the operand slots (None affine
    # pair / running stats drop out at export); the scalar tail names
    # here.  ``use_input_stats`` is semantic: False means "use the
    # running stats" (the eval-mode export when
    # track_running_stats=True).
    "instance_norm": {
        5: "use_input_stats",
        6: "momentum",
        7: "eps",
        8: "cudnn_enabled",
    },
    # aten.upsample_nearest2d.vec(x, output_size, scale_factors) —
    # the vec overload carries exactly one non-None list; both land
    # under distinct names so the binding need not discriminate int
    # vs float.  (aten.upsample_nearest2d.default's lone output_size
    # arg shares position 1.)
    "upsample_nearest2d": {1: "size", 2: "scale"},
    # aten.lstm.input(x, hx[2], params[*], has_biases, num_layers,
    # dropout, train, bidirectional, batch_first) — positions 1-2 are
    # tensor LIST operands (flattened into args at export); the
    # scalar tail names here.
    "lstm.input": {
        3: "has_biases",
        4: "num_layers",
        5: "dropout",
        6: "train",
        7: "bidirectional",
        8: "batch_first",
    },
    # aten.slice(t, dim, start, end, step) — the trailing three are
    # the slice bounds/stride, read by typing/the binding.
    "slice": {1: "dim", 2: "start", 3: "end", 4: "step"},
    # --- attention / normalisation -----------------------------------------
    # sdpa(q, k, v, attn_mask, dropout_p, is_causal, scale, enable_gqa);
    # attn_mask is normally a tensor operand, the rest arrive as arg4..7.
    "sdpa": {
        3: "attn_mask",
        4: "dropout_p",
        5: "is_causal",
        6: "scale",
        7: "enable_gqa",
    },
    # aten.rms_norm(x, normalized_shape, weight, eps): normalized_shape
    # lands under ``dim`` (the list-arg convention), weight is a
    # tensor operand, eps is arg3.
    "rms_norm": {1: "dim", 3: "eps"},
    # aten.layer_norm(x, normalized_shape, weight, bias, eps,
    # cudnn_enabled): eps is arg4 — NOT arg5 (the cudnn flag).  The old
    # binding read arg5 for eps and silently produced eps=0.0.
    "layer_norm": {1: "dim", 4: "eps", 5: "cudnn_enabled"},
    # aten.dropout(x, p, train).
    "dropout": {1: "p", 2: "train"},
    # aten.conv2d positional tails — also intercepted operand-side by
    # the exporter (list-valued stride/padding never reach ``argN``).
    "conv2d": {3: "stride", 4: "padding", 5: "dilation", 6: "groups"},
    "conv1d": {3: "stride", 4: "padding", 5: "dilation", 6: "groups"},
    # aten.mode is the median-style (values, indices) pair picked by
    # a getitem consumer.
    "mode": {1: "dim", 2: "keepdim"},
    # aten.eye(n) / eye.m(n, m) — exported positional sizes name into
    # the carrier binding's ``dim`` (``m`` for the rectangular one).
    "eye": {0: "dim", 1: "m"},
}

#: Canonical names that must be present (either spelling) on a
#: fully-attributed term: enforced only when the term supplies at
#: least one schema-declared attr.  Attrs with aten-side defaults
#: (split's ``dim``, chunk's ``dim``, flatten's ``end_dim``, every
#: sdpa flag, dropout's ``p``/``train``) are NOT required — exports
#: legitimately elide them.
ATTR_REQUIRED: dict[str, frozenset[str]] = {
    "concat": frozenset({"dim"}),
    "stack": frozenset({"dim"}),
    "unbind": frozenset({"dim"}),
    "chunk": frozenset({"chunks"}),
    "split": frozenset({"sizes"}),
    "getitem": frozenset({"index"}),
    "transpose": frozenset({"dim0", "dim1"}),
    "unsqueeze": frozenset({"dim"}),
    "squeeze": frozenset({"dim"}),
    "select": frozenset({"dim", "index"}),
    "flatten": frozenset({"start_dim"}),
    "unflatten": frozenset({"dim", "sizes"}),
    "index_select": frozenset({"dim"}),
    "take_along_dim": frozenset({"dim"}),
    "gather": frozenset({"dim"}),
    "scatter": frozenset({"dim"}),
    "scatter_add": frozenset({"dim"}),
    "scatter_reduce": frozenset({"dim", "reduce"}),
    "index_add": frozenset({"dim"}),
    "slice_scatter": frozenset({"dim"}),
    "select_scatter": frozenset({"dim", "index"}),
    "narrow": frozenset({"dim", "start", "length"}),
    "movedim": frozenset({"source", "destination"}),
    "roll": frozenset({"shifts"}),
    "einsum": frozenset({"equation"}),
    "cumsum": frozenset({"dim"}),
    "cumprod": frozenset({"dim"}),
    "cummax": frozenset({"dim"}),
    "cummin": frozenset({"dim"}),
    "logcumsumexp": frozenset({"dim"}),
    "log_softmax": frozenset({"dim"}),
    "median": frozenset({"dim"}),
    "mode": frozenset({"dim"}),
    "kthvalue": frozenset({"k"}),
    "argsort": frozenset({"dim"}),
    "topk": frozenset({"k"}),
    "sort": frozenset({"dim"}),
    "glu": frozenset({"dim"}),
    "one_hot": frozenset({"num_classes"}),
    "tensor_split": frozenset({"sections"}),
    "unfold": frozenset({"dim", "size", "step"}),
    "pixel_shuffle": frozenset({"upscale_factor"}),
    "pixel_unshuffle": frozenset({"downscale_factor"}),
    "pad": frozenset({"pad"}),
    "group_norm": frozenset({"num_groups"}),
    "softmax": frozenset({"dim"}),
    "slice": frozenset({"dim"}),
    "rms_norm": frozenset({"dim"}),
    "layer_norm": frozenset({"dim"}),
}

_ARG_RE = re.compile(r"^arg(\d+)$")


def attr_of(source: Any, *names: str, default: Any = None) -> Any:
    """Return the first present attr across names.

    The canonical + positional dual spelling read.  `source` may be a
    term (reads .attrs) or a dict.
    """
    attrs = (
        source
        if isinstance(source, Mapping)
        else getattr(source, "attrs", None)
    )
    if not isinstance(attrs, Mapping):
        return default
    for name in names:
        if name in attrs:
            return attrs[name]
    return default


def is_positional_attr(key: Any) -> bool:
    """Return True for the ``argN`` positional-attr spelling."""
    return isinstance(key, str) and _ARG_RE.match(key) is not None


def validate_attrs(
    op: str, attrs: dict[str, Any], *, validate: bool = True
) -> dict[str, Any]:
    """``Op.make``'s mint-time contract for schema'd ops.

    * ``argN`` at an undeclared position is malformed → ``ValueError``.
    * ``argN`` at a declared position is a legal alternate spelling and
      is preserved — UNLESS the canonical name is also present, in
      which case the positional write wins and merges under it.
    * A term supplying at least one schema-declared attr (either
      spelling) must satisfy every ``ATTR_REQUIRED`` name →
      ``ValueError`` otherwise.  Zero-attr terms and terms carrying
      only non-declared names pass (partial/pattern terms).
    * Ops with no schema pass through untouched; ``validate=False``
      returns the dict as-is.
    """
    schema = ATTR_SCHEMA.get(op)
    if schema is None or not validate:
        return attrs
    out = dict(attrs)
    declared_keys = set(schema.values()) | {f"arg{i}" for i in schema}
    for key in list(out):
        m = _ARG_RE.match(key)
        if m is None:
            continue
        canon = schema.get(int(m.group(1)))
        if canon is None:
            raise ValueError(
                f"{op}: positional attr '{key}' has no declared "
                f"canonical name — ATTR_SCHEMA[{op!r}] covers "
                f"positions {sorted(schema)}; extend the schema or "
                f"mint with validate=False"
            )
        if canon != key and canon in out:
            # An imperative argN patch onto a canonically-spelled term:
            # the positional write wins and lands under canon.
            out[canon] = out.pop(key)
    required = ATTR_REQUIRED.get(op)
    if required and set(out) & declared_keys:
        name_to_pos = {v: k for k, v in schema.items()}
        missing = [
            name
            for name in required
            if name not in out and f"arg{name_to_pos[name]}" not in out
        ]
        if missing:
            raise ValueError(
                f"{op}: missing required attr(s) {sorted(missing)} — "
                f"ATTR_SCHEMA requires {sorted(required)} (either "
                f"spelling) on a fully-attributed term (got "
                f"{sorted(out)}); mint a bare {op}(...) for a partial "
                f"term or pass validate=False"
            )
    return out
