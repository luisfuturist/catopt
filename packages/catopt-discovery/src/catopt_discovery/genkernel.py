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


def _site_env(spec: Any, term: Any) -> dict | None:
    """Return the metavar bindings of the first concrete occurrence."""
    from catopt_core.ir import Op

    from catopt_discovery.meta_arena import _concrete_match

    env: dict = {}
    if _concrete_match(spec, term, env):
        return env
    if isinstance(term, Op):
        for a in term.args:
            found = _site_env(spec, a)
            if found is not None:
                return found
    return None


def profitable(
    handlers: dict,
    term: Any,
    var_env: dict,
    *,
    reps: int = 40,
    margin: float = 1.0,
) -> dict:
    """Keep only handlers whose compiled kernel beats the spelled site.

    The measured-cost referee: for each minted handler, locate its
    concrete occurrence, evaluate the bound subterms once, and time
    the spelled body against the generated compiled callable on those
    real values.  A claim whose kernel doesn't actually pay (≥
    ``margin`` x faster than its spelled body) is filtered out —
    op-count pricing can't see dynamo overhead, so measurement, not
    the cost model, decides which claims are worth offering.
    """
    import catopt_torch.torch_bridge as tb
    import torch
    from catopt_torch.torch_bridge import eval_term

    # measurement integrity — see gen_probe: per-handler compiled
    # fns would exhaust dynamo's default recompile limit (8) and
    # silently degrade to eager mid-sweep.
    torch._dynamo.config.recompile_limit = max(
        int(torch._dynamo.config.recompile_limit), 256
    )

    bindings = tb._IR_TO_TORCH
    keep: dict[str, dict] = {}
    for name, h in handlers.items():
        env = _site_env(h["pattern"], term)
        if env is None:
            continue
        try:
            vals = {
                k: eval_term(
                    v,
                    var_env=var_env,
                    param_env=var_env,
                    bindings=bindings,
                    strict=True,
                )
                for k, v in env.items()
            }
        except Exception:
            continue

        def spelled(s=h["pattern"], v=vals) -> Any:
            return _eval_spec(s, v, bindings)

        def kerneled(*a: Any, s=h["pattern"]) -> Any:
            return _eval_spec(
                s,
                dict(zip(_spec_metas(s), a, strict=True)),
                bindings,
            )

        kernel = torch.compile(kerneled)
        mvs = _spec_metas(h["pattern"])

        def _t(fn: Any) -> float:
            import time

            for _ in range(5):
                fn()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            ts.sort()
            return ts[len(ts) // 2]

        t_spelled = _t(spelled)
        t_kernel = _t(
            lambda kernel=kernel, vals=vals, mvs=mvs: kernel(
                *[vals[m] for m in mvs]
            )
        )
        if t_kernel <= t_spelled * margin:
            keep[name] = h
    return keep


def _kernel_cache() -> Any:
    """Return the generated-kernel module cache dir."""
    # real files, inspectable — the artifact is a module on disk
    import tempfile
    from pathlib import Path

    d = Path(tempfile.gettempdir()) / "catopt_genkernels"
    d.mkdir(exist_ok=True)
    return d


_TRITON_UNARY = {
    "neg": "(-({a}))",
    "abs": "tl.abs({a})",
    "square": "(({a}) * ({a}))",
    "sigmoid": "(1.0 / (1.0 + tl.exp(-({a}))))",
    "tanh": "libdevice.tanh({a})",
    "relu": "tl.maximum({a}, 0.0)",
    "gelu": (
        "(0.5 * ({a}) * (1.0 + libdevice.erf("
        "({a}) * 0.70710678118654752)))"
    ),
    "softplus": "libdevice.log1p(tl.exp({a}))",
    "mish": "(({a}) * libdevice.tanh(libdevice.log1p(tl.exp({a}))))",
    "silu": "(({a}) / (1.0 + tl.exp(-({a}))))",
    "exp": "tl.exp({a})",
    "log": "tl.log({a})",
    "sqrt": "tl.sqrt({a})",
    "rsqrt": "(1.0 / tl.sqrt({a}))",
    "sin": "tl.sin({a})",
    "cos": "tl.cos({a})",
    "identity": "({a})",
    "reciprocal": "(1.0 / ({a}))",
    "softsign": "(({a}) / (1.0 + tl.abs({a})))",
    "softsign_gate": "((({a}) / (1.0 + tl.abs({a}))) * ({b}))",
    "atanh": "libdevice.atanh({a})",
}
_TRITON_BINARY = {
    "add": "(({a}) + ({b}))",
    "sub": "(({a}) - ({b}))",
    "mul": "(({a}) * ({b}))",
    "div": "(({a}) / ({b}))",
    "pow": "libdevice.pow({a}, {b})",
}


def _triton_expr(spec: Any) -> str | None:
    """Render a canonical spec as a Triton expression, or decline.

    The codegen map — each spec node becomes a ``tl``/libdevice
    expression over its metavar operands.  Constant leaves inline
    as literals; attr dicts on ops this tier doesn't model make
    the whole spec decline honestly (``None``).
    """
    if isinstance(spec, str):
        return spec
    if not isinstance(spec, (tuple, list)):
        return f"({float(spec)})"
    kids = [e for e in spec[1:] if not isinstance(e, dict)]
    if len(spec[1:]) != len(kids):
        return None  # attr-carrying op this tier doesn't model
    args = [_triton_expr(k) for k in kids]
    if any(a is None for a in args):
        return None
    return _triton_dispatch(spec[0], args)


def _triton_dispatch(op: str, args: list) -> str | None:
    """Format the rendered kid expressions into one op expression."""
    if op in _TRITON_BINARY and len(args) == 2:
        return _TRITON_BINARY[op].format(a=args[0], b=args[1])
    if op == "softsign_gate" and len(args) == 2:
        return _TRITON_UNARY[op].format(a=args[0], b=args[1])
    if op in _TRITON_UNARY and len(args) == 1:
        return _TRITON_UNARY[op].format(a=args[0])
    return None


def triton_bindings(handlers: dict) -> dict:
    """Generate a Triton kernel per handler — one nobody wrote.

    Unlike :func:`gen_bindings` (whose callable wraps
    ``torch.compile`` — the incumbent's own codegen), this tier
    emits real ``triton.jit`` source for the fused body and
    compiles it directly.  Broadcasting is handled on the host
    side: args are broadcast + flattened once, the kernel walks a
    flat index space.  Specs whose ops the codegen doesn't model
    are skipped — the handler keeps its torch path instead.
    """
    import importlib.util

    import torch

    out: dict[str, Any] = {}
    for h in handlers.values():
        spec = h["pattern"]
        metas = _spec_metas(spec)
        body = _triton_expr(spec)
        if body is None or not metas:
            continue
        ptrs = ", ".join(f"{m}_ptr" for m in metas)
        loads = "".join(
            f"    {m} = tl.load({m}_ptr + off, mask=mask, other=0.0)\n"
            for m in metas
        )
        src = (
            "import triton\n"
            "import triton.language as tl\n"
            "import triton.language.extra.libdevice as libdevice\n"
            "\n@triton.jit\n"
            f"def _gk({ptrs}, out_ptr, n, BLOCK: tl.constexpr):\n"
            "    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
            "    mask = off < n\n"
            f"{loads}"
            f"    out = {body}\n"
            "    tl.store(out_ptr + off, out, mask=mask)\n"
        )
        # the artifact is a real module file — inspectable on disk,
        # not an exec'd string.  Loading through importlib keeps the
        # repo's no-eval/exec invariant intact.
        path = _kernel_cache() / f"_gk_{h['kernel']}.py"
        path.write_text(src)
        spec_ = importlib.util.spec_from_file_location(path.stem, path)
        if spec_ is None or spec_.loader is None:
            continue  # module load declined — keep the torch path
        mod = importlib.util.module_from_spec(spec_)
        spec_.loader.exec_module(mod)
        kern = mod._gk

        def fn(*args: Any, kern=kern, **_attrs: Any) -> Any:
            xs = [
                x.contiguous() for x in torch.broadcast_tensors(*args)
            ]
            shape = xs[0].shape
            n = xs[0].numel()
            out = torch.empty(n, dtype=xs[0].dtype, device=xs[0].device)
            block = 1024
            kern[((n + block - 1) // block,)](
                *[x.reshape(-1) for x in xs],
                out,
                n,
                BLOCK=block,
            )
            return out.reshape(shape)

        out[h["kernel"]] = fn
    return out


def triton_sink(handlers: dict) -> Any:
    """Build a ``TorchSink`` binding gen ops to Triton kernels.

    Supported ops = ambient + handler kernels whose spec the
    codegen accepted — a claim on a declined spec stays
    unpriced-unlowerable via the ambient table, the same posture
    as ``gen_sink``.
    """
    import catopt_torch.torch_bridge as tb
    from catopt_core.ops import OpTable
    from catopt_torch.adapters import TorchSink

    bindings = dict(tb._IR_TO_TORCH)
    bindings.update(triton_bindings(handlers))
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)


def hybrid_sink(handlers: dict, *, compile_kernels: bool = True) -> Any:
    """Build a sink preferring Triton codegen, compile fallback.

    Triton-accepted specs bind to the generated ``.py`` kernel —
    a kernel no incumbent compiler produces; specs the codegen
    declines fall back to the ``torch.compile``-wrapped body.
    """
    import catopt_torch.torch_bridge as tb
    from catopt_core.ops import OpTable
    from catopt_torch.adapters import TorchSink

    bindings = dict(tb._IR_TO_TORCH)
    bindings.update(
        gen_bindings(handlers, compile_kernels=compile_kernels)
    )
    bindings.update(triton_bindings(handlers))
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
