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
    # every reader (typing, ibp, the torch binding) reads.
    "transpose": {1: "dim0", 2: "dim1"},
    "unsqueeze": {1: "dim"},
    "squeeze": {1: "dim"},
    "softmax": {1: "dim"},
    "select": {1: "dim", 2: "index"},
    "flatten": {1: "start_dim", 2: "end_dim"},
    "index_select": {1: "dim", 2: "index"},
    "gather": {1: "dim"},
    "narrow": {1: "dim", 2: "start", 3: "length"},
    # aten.slice(t, dim, start, end, step) — the trailing three are
    # the slice bounds/stride, read by typing/ibp/the binding.
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
    "index_select": frozenset({"dim"}),
    "gather": frozenset({"dim"}),
    "narrow": frozenset({"dim", "start", "length"}),
    "softmax": frozenset({"dim"}),
    "slice": frozenset({"dim"}),
    "rms_norm": frozenset({"dim"}),
    "layer_norm": frozenset({"dim"}),
}

_ARG_RE = re.compile(r"^arg(\d+)$")


def attr_of(source: Any, *names: str, default: Any = None) -> Any:
    """First present attr across names — the canonical + positional dual
    spelling read.  `source` may be a term (reads .attrs) or a dict."""
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
    """True for the ``argN`` positional-attr spelling."""
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
