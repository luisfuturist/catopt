"""Generated kernels — lowerings the search itself builds.

The handler table (:data:`lawdata.HANDLERS`) is a human-authored
vocabulary: ``silu``/``square``/``swiglu`` are names someone wrote.
This module closes the loop the meta-arena's ``claim`` move implies:
given a *spec* — a canonical pattern tuple mined from a corpus — mint
a handler entry whose kernel is a NEW op name, plus a torch binding
that evaluates the spelled body.  The board's vocabulary grows with
lowerings the search built; nothing in the arena or the sink changes:

* ``gen_handlers(specs)`` mints ``{tag: {pattern, kernel, args}}``
  for each *elementwise* spec not already covered by the shipped
  handlers — the fused-op candidates nobody named.
* ``gen_bindings`` produces each kernel's :class:`~catopt_core.ports.Binding`
  — ``fn(*vals) = eval(spelled body, leaves=vals)`` — optionally
  ``torch.compile``-compiled so the delivered kernel is a real fused
  implementation (one inductor graph), not a replay of the spelling.
* ``gen_sink`` folds the bindings into a private :class:`OpTable`
  copy and wraps it in a ``TorchSink`` — ``supported_ops`` reports
  the gen names (the feasibility bound the board prices against)
  and ``lower`` delivers modules that *call the kernel the board
  invented*.
* ``novel_cost`` — an extraction-priced novelty bonus so the search
  prefers a claimed/generated member when costs tie.

Feasibility stays honest: a gen op is only priced low when it is in
the board's ``supported`` set, and kernels are only *generated* for
elementwise bodies — the narrow claim this module makes (pointwise
fusion), not a general codegen story.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ELEMENTWISE",
    "elementwise_spec",
    "gen_bindings",
    "gen_handlers",
    "gen_sink",
    "novel_cost",
]

#: Ops whose torch bindings are pointwise (shape-preserving) — the
#: bodies a generated kernel may fuse.  Restricted to the plain
#: unary/binary elementwise vocabulary: no reductions, no reshapes,
#: no matmul — the honest narrow claim.
ELEMENTWISE: frozenset = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "square",
        "sigmoid",
        "tanh",
        "relu",
        "gelu",
        "softplus",
        "mish",
        "silu",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "sin",
        "cos",
        "identity",
        "pow",
        "reciprocal",
        "softsign",
        "softsign_gate",
    }
)


def _spec_ops(spec: Any) -> set:
    """Collect op names in a canonical spec tuple (leaves are metas)."""
    if isinstance(spec, (tuple, list)) and spec:
        out = {spec[0]}
        for e in spec[1:]:
            if not isinstance(e, dict):
                out |= _spec_ops(e)
        return out
    return set()


def _spec_metas(spec: Any) -> list:
    """Ordered distinct metavar leaves of a canonical spec."""
    out: list = []

    def walk(s: Any) -> None:
        if isinstance(s, str):
            if s not in out:
                out.append(s)
        elif isinstance(s, (tuple, list)):
            for e in s[1:]:
                if not isinstance(e, dict):
                    walk(e)

    walk(spec)
    return out


def elementwise_spec(spec: Any) -> bool:
    """Check a spec is a pure elementwise body the gen path may fuse."""
    ops = _spec_ops(spec)
    return bool(ops) and ops <= ELEMENTWISE


def gen_handlers(terms: list, *, covered: set | None = None) -> dict:
    """Mint a handler table for concrete subterms or patterns.

    Each entry yields ``gen_<i>: {pattern, kernel, args}`` — the
    same entry shape ``lawdata.HANDLERS`` ships, so ``claim``
    binds them with no new machinery.  Entries whose *concrete*
    pattern already exists — in *covered* (e.g. the shipped
    handlers' canon-concrete patterns) or earlier in the list —
    are skipped: the generated vocabulary adds names only where
    the ambient table has none.
    """
    from catopt_discovery.meta_arena import (
        _canon_concrete,
        _spec_of,
    )

    skip = covered if covered is not None else set()
    out: dict[str, dict] = {}
    for term in terms:
        pattern = (
            term
            if isinstance(term, (tuple, list))
            else _spec_of(term, {})
        )
        canon = _canon_concrete(pattern)
        if canon in skip or not elementwise_spec(pattern):
            continue
        if any(
            _canon_concrete(h["pattern"]) == canon for h in out.values()
        ):
            continue
        tag = f"gen_{len(out)}"
        out[tag] = {
            "pattern": pattern,
            "kernel": f"{tag}_k",
            "args": tuple(_spec_metas(pattern)),
        }
    return out


def _eval_spec(spec: Any, env: dict, bindings: dict) -> Any:
    """Evaluate a canonical spec tuple under a metavar environment."""
    if isinstance(spec, str):
        return env[spec]
    if isinstance(spec, (tuple, list)):
        attrs = next((e for e in spec[1:] if isinstance(e, dict)), {})
        args = [
            _eval_spec(e, env, bindings)
            for e in spec[1:]
            if not isinstance(e, dict)
        ]
        return bindings[spec[0]](*args, **attrs)
    return spec  # Const leaf — the canonical form carries the value


def gen_bindings(
    handlers: dict, *, compile_kernels: bool = True
) -> dict:
    """Build each gen kernel's torch ``Binding``.

    The binding re-evaluates the handler's spelled body against its
    positional metavars — generated code, not a table entry.  With
    ``compile_kernels`` the callable is ``torch.compile``-compiled into a
    fused inductor graph: the delivered module genuinely runs one
    kernel where the source spelled several.
    """
    import catopt_torch.torch_bridge as tb
    import torch

    bindings = tb._IR_TO_TORCH
    out: dict[str, Any] = {}
    for h in handlers.values():
        spec = h["pattern"]

        def fn(*args: Any, s=spec, **_attrs: Any) -> Any:
            names = _spec_metas(s)
            env = dict(zip(names, args, strict=True))
            return _eval_spec(s, env, bindings)

        out[h["kernel"]] = torch.compile(fn) if compile_kernels else fn
    return out


def gen_sink(handlers: dict, *, compile_kernels: bool = True) -> Any:
    """Build a ``TorchSink`` whose vocabulary includes the gen kernels.

    ``supported_ops`` reports the ambient table plus every gen name —
    the feasibility bound the board prices against.
    """
    import catopt_torch.torch_bridge as tb
    from catopt_core.ops import OpTable
    from catopt_torch.adapters import TorchSink

    bindings = dict(tb._IR_TO_TORCH) | gen_bindings(
        handlers, compile_kernels=compile_kernels
    )
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)


def novel_cost(base: Any, is_novel: Any, bonus: float = 0.5) -> Any:
    """Discount members headed by a novel (declared/gen) op.

    Extraction-priced novelty: a member whose head satisfies
    *is_novel* — e.g. ``op.startswith(("claim_", "foldabs_", "gen_"))``
    — bills ``bonus`` less than its spelled equivalent: the search's
    reason to prefer the invented form when costs tie.  Feasibility
    is untouched: the member must already be lowerable.
    """
    from catopt_core.ir import Op

    def cost(node: Any) -> float:
        c = base(node)
        if isinstance(node, Op) and is_novel(node.op):
            return max(c - bonus, 0.0)
        return c

    return cost
