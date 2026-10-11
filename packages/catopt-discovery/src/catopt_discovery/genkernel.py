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

THE SCAN TIER
    ``triton_bindings`` covers pointwise bodies.  The second codegen
    tier lowers *carrier* terms — the diagonal-affine scan the
    ``affd_*_lift`` family mints — into ONE sequential Triton kernel.
    Two spec forms are recognised (:func:`scan_spec`):

    * ``("applyd", ("aff_diag", a, b), h)`` — the *sequence* form.
      The leaf operands are per-step tensors carrying a step axis
      (``axis``, default ``-2`` — the token axis of ``(B, T, d)``);
      the kernel folds ``h_t = a_t ⊙ h + b_t`` along it and returns
      the final state ``h_T``.  Operands may be metavars, constants,
      or elementwise sub-specs — every metavar inside is a sequence
      operand, so ``aff_diag(("sigmoid", "G"), ("mul", "B", "X"))``
      generates the fully-fused selective step.
    * ``("applyd", <affd_compose tree>, h)`` — the *unrolled* form.
      Each ``aff_diag`` leaf is one carrier step (state-shaped
      operands, no step axis); the kernel applies them in
      chronological order — in-order leaves are reverse-chronological
      (``affd_compose(f, g)`` applies ``g`` first) — in one pass.
      This is the carrier semantics *verbatim*.

    The generated binding is a NEW kernel name (``scan_<i>_k``), not
    ``applyd`` itself: the sequence interpretation is the minted op's
    declared semantics, never a silent override of the ambient
    carrier binding.  Call-time shape violations — no step axis, an
    ``h`` that cannot broadcast to the state shape, non-CUDA tensors
    — raise rather than fall back, so a wrong binding fails loudly.

    Delivery seam: ``scan_handlers`` entries merge into a handler
    table exactly as :func:`gen_handlers` output does — the arena's
    ``claim`` binds them with no new machinery, and
    :func:`hybrid_sink` (the sink ``gen_probe`` lowers through)
    consults both codegen tiers, so a minted scan op is deliverable
    end to end.  :func:`scan_triton_sink` is the scan-only sink.
    What is NOT wired is *minting*: raw exported IR carries no
    ``applyd`` sites — carrier members only appear once lifts run —
    so case builders do not call ``scan_handlers``; a caller that
    can vouch for a site mints the entry itself.  A claim's
    soundness is the minter's burden: the unrolled form is the
    ambient carrier semantics verbatim; the sequence form is a NEW
    declared semantics — the ambient
    ``applyd(aff_diag(a, b), h)`` is one map application
    (``a⊙h+b``), not a fold — so mint it only where the fold
    reading is what the site means.
"""

from __future__ import annotations

import keyword
from typing import Any

__all__ = [
    "ELEMENTWISE",
    "elementwise_spec",
    "gen_bindings",
    "gen_handlers",
    "gen_sink",
    "novel_cost",
    "scan_handlers",
    "scan_spec",
    "scan_triton_bindings",
    "scan_triton_sink",
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
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)

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
        # the referee verifies before it times: the kernel must
        # reproduce the spelled body on the bound values — an
        # inductor miscompile or dtype drift would otherwise win
        # the timing race it should have lost.
        if not torch.allclose(
            kerneled(*[vals[m] for m in mvs]),
            spelled(),
            rtol=1e-4,
            atol=1e-5,
        ):
            continue

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


def _load_generated(path: Any, name: str) -> Any:
    """Import a generated ``.py`` kernel module; ``None`` to decline.

    The artifact is a real module file — inspectable on disk, not an
    exec'd string.  Loading through importlib keeps the repo's
    no-eval/exec invariant intact.  A ``None`` spec/loader means the
    import machinery declined the file; the caller keeps whatever
    fallback path it had.
    """
    import importlib.util

    spec_ = importlib.util.spec_from_file_location(path.stem, path)
    if spec_ is None or spec_.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    return getattr(mod, name)


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
        kern = _load_generated(path, "_gk")
        if kern is None:
            continue  # module load declined — keep the torch path

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


def _seq_scan_spec(spec: Any) -> bool:
    """Check *spec* is a scan spec in the sequence (fold) form."""
    info = scan_spec(spec)
    return info is not None and info["kind"] == "seq"


def hybrid_sink(handlers: dict, *, compile_kernels: bool = True) -> Any:
    """Build a sink preferring Triton codegen, compile fallback.

    Both codegen tiers are consulted.  Triton-accepted elementwise
    specs bind to the generated ``.py`` kernel — a kernel no
    incumbent compiler produces — and ``applyd``-carrier specs the
    scan codegen accepts bind to their generated fold kernel.
    Specs the codegen declines fall back to the
    ``torch.compile``-wrapped body — EXCEPT sequence-form scan
    specs: their spelled body evaluates as one map application
    (``a⊙h+b``), not the fold the minted op declares, so a seq spec
    the scan codegen cannot lower stays unbound (unpriced-
    unlowerable) rather than silently delivering a different
    function.  Unrolled compose-tree specs keep the fallback —
    their spelled eval IS the carrier semantics, just serial.
    """
    import catopt_torch.torch_bridge as tb
    from catopt_core.ops import OpTable
    from catopt_torch.adapters import TorchSink

    plain = {
        k: h
        for k, h in handlers.items()
        if not _seq_scan_spec(h["pattern"])
    }
    bindings = dict(tb._IR_TO_TORCH)
    bindings.update(
        gen_bindings(plain, compile_kernels=compile_kernels)
    )
    bindings.update(triton_bindings(plain))
    bindings.update(scan_triton_bindings(handlers))
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


# ---------------------------------------------------------------------------
#  The scan tier — generated kernels for the diagonal-affine carrier
# ---------------------------------------------------------------------------

#: Tile width for generated scan kernels — one Triton program per
#: ``BLOCK`` columns of the flattened state space.
_SCAN_BLOCK = 1024

#: Names the generated scan source already binds — a metavar colliding
#: with one (or with the generated ``*_ptr`` parameter suffix) would
#: silently shadow a program variable, so such specs decline codegen.
_SCAN_RESERVED = frozenset(
    {
        "acc",
        "off",
        "mask",
        "row",
        "n",
        "N",
        "T",
        "t",
        "BLOCK",
        "out",
        "program_id",
    }
)


def _meta_name_ok(name: Any) -> bool:
    """Check *name* is safe as a generated-kernel variable."""
    return (
        isinstance(name, str)
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and not name.endswith("_ptr")
        and name not in _SCAN_RESERVED
    )


def _affd_leaves(f_spec: Any) -> list | None:
    """In-order ``(a, b)`` operand pairs of a pure aff_diag map spec.

    ``None`` when *f_spec* is not an ``aff_diag`` leaf or an
    ``affd_compose`` tree over such leaves (attr-carrying or foreign
    nodes included — the map spec must be exactly the carrier's).
    """
    if not (
        isinstance(f_spec, (tuple, list))
        and len(f_spec) == 3
        and all(not isinstance(e, dict) for e in f_spec)
    ):
        return None
    if f_spec[0] == "aff_diag":
        return [(f_spec[1], f_spec[2])]
    if f_spec[0] != "affd_compose":
        return None
    left = _affd_leaves(f_spec[1])
    right = _affd_leaves(f_spec[2])
    if left is None or right is None:
        return None
    return [*left, *right]


def _leaf_exprs(leaves: list, h: str) -> tuple[list, list] | None:
    """Render leaf operands to Triton exprs, in chronological order.

    In-order leaves of an ``affd_compose`` tree are REVERSE
    chronological — ``affd_compose(f, g)`` applies ``g`` first — so
    the list is flipped here.  Returns ``(metas, steps)`` where
    *metas* are the operand metavars in spec order and *steps* the
    ``(a_expr, b_expr)`` rendered pairs.  ``None`` declines: an
    operand the elementwise renderer cannot express, a metavar that
    collides with kernel names, or the state meta rebound as a
    sequence operand.
    """
    metas: list = []
    for a_spec, b_spec in leaves:
        for m in _spec_metas(("aff_diag", a_spec, b_spec)):
            if m not in metas:
                metas.append(m)
    if (
        not metas
        or h in metas
        or any(not _meta_name_ok(m) for m in metas)
    ):
        return None
    steps = []
    for a_spec, b_spec in reversed(leaves):
        ea, eb = _triton_expr(a_spec), _triton_expr(b_spec)
        if ea is None or eb is None:
            return None
        steps.append((ea, eb))
    return metas, steps


def _applyd_operands(spec: Any) -> tuple | None:
    """Split an ``applyd`` spec into ``(map spec, state meta, attrs)``.

    ``None`` when the node is not a two-operand ``applyd`` over a
    metavar state — malformed arities, a non-metavar ``h``, or a
    kernel-name-colliding state all decline here.
    """
    if not (
        isinstance(spec, (tuple, list))
        and len(spec) >= 3
        and spec[0] == "applyd"
    ):
        return None
    kids = [e for e in spec[1:] if not isinstance(e, dict)]
    if len(kids) != 2 or not _meta_name_ok(kids[1]):
        return None
    attrs = next((e for e in spec[1:] if isinstance(e, dict)), None)
    return kids[0], kids[1], attrs


def scan_spec(spec: Any) -> dict | None:
    """Recognise an ``applyd``-carrier spec the scan codegen accepts.

    Returns a descriptor dict, or ``None`` when the spec is outside
    the tier's grammar (decline — the handler keeps its torch path):

    * ``{"kind": "seq", ...}`` for ``("applyd", ("aff_diag", a, b),
      h)`` — the sequence form.  ``"seq"`` lists the operand metavars
      (each a per-step tensor on the step axis), ``"a"``/``"b"`` the
      rendered per-step expressions, ``"h"`` the state metavar.
    * ``{"kind": "unrolled", ...}`` for ``("applyd",
      <affd_compose tree>, h)`` — ``"steps"`` the rendered ``(a, b)``
      pairs in chronological order.

    Both carry ``"params"`` — the spec's metavars in order, i.e. the
    generated binding's positional signature.  A trailing attr dict
    on the ``applyd`` node may declare ``{"axis": int}`` — surfaced
    as ``info["axis"]``; a ``None``/absent axis defers to the
    binding's default or its derive-from-``h`` rule.
    """
    parts = _applyd_operands(spec)
    if parts is None:
        return None
    f_spec, h_spec, attrs = parts
    leaves = _affd_leaves(f_spec)
    if leaves is None:
        return None
    rendered = _leaf_exprs(leaves, h_spec)
    if rendered is None:
        return None
    metas, steps = rendered
    info: dict[str, Any] = {
        "h": h_spec,
        "seq": metas,
        "params": [*metas, h_spec],
    }
    if len(leaves) == 1:
        (ea, eb) = steps[0]
        info.update({"kind": "seq", "a": ea, "b": eb})
    else:
        info.update({"kind": "unrolled", "steps": steps})
    if attrs is not None and "axis" in attrs:
        info["axis"] = attrs["axis"]
    return info


def _seq_triton_source(info: dict) -> str:
    """Emit the sequential-scan kernel module for a ``seq`` spec.

    One program per ``BLOCK`` columns of the flattened state space;
    each walks the step axis serially — ``acc = a_t * acc + b_t``.
    Row offsets are computed in int64 so ``T * N`` may exceed int32.
    """
    params = ", ".join(f"{m}_ptr" for m in info["params"])
    loads = "".join(
        f"        {m} = tl.load({m}_ptr + row + off,"
        " mask=mask, other=0.0)\n"
        for m in info["seq"]
    )
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        f"def _gs({params}, out_ptr, T, N, BLOCK: tl.constexpr):\n"
        "    off = tl.program_id(0).to(tl.int64) * BLOCK"
        " + tl.arange(0, BLOCK)\n"
        "    mask = off < N\n"
        f"    acc = tl.load({info['h']}_ptr + off,"
        " mask=mask, other=0.0)\n"
        "    for t in range(T):\n"
        "        row = t.to(tl.int64) * N\n"
        f"{loads}"
        f"        acc = ({info['a']}) * acc + ({info['b']})\n"
        "    tl.store(out_ptr + off, acc, mask=mask)\n"
    )


def _unrolled_triton_source(info: dict) -> str:
    """Emit the unrolled carrier kernel for a compose-tree spec.

    One flat pass over the broadcast state space applying each leaf
    step in chronological order — ``acc = a_i * acc + b_i`` — exactly
    the ambient ``applyd``/``affd_compose`` semantics fused into one
    kernel.
    """
    params = ", ".join(f"{m}_ptr" for m in info["params"])
    loads = "".join(
        f"    {m} = tl.load({m}_ptr + off, mask=mask, other=0.0)\n"
        for m in info["seq"]
    )
    steps = "".join(
        f"    acc = ({ea}) * acc + ({eb})\n" for ea, eb in info["steps"]
    )
    return (
        "import triton\n"
        "import triton.language as tl\n"
        "import triton.language.extra.libdevice as libdevice\n"
        "\n\n@triton.jit\n"
        f"def _gs({params}, out_ptr, n, BLOCK: tl.constexpr):\n"
        "    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = off < n\n"
        f"{loads}"
        f"    acc = tl.load({info['h']}_ptr + off,"
        " mask=mask, other=0.0)\n"
        f"{steps}"
        "    tl.store(out_ptr + off, acc, mask=mask)\n"
    )


def _scan_triton_source(info: dict) -> str:
    """Emit the generated scan kernel module for a recognised spec.

    *info* is a :func:`scan_spec` descriptor; ``"seq"`` emits the
    sequential step-axis fold, ``"unrolled"`` the flat chronological
    pass.  The source is a real module — written to the kernel cache
    by the caller.
    """
    if info["kind"] == "seq":
        return _seq_triton_source(info)
    return _unrolled_triton_source(info)


def _broadcasts_into(src: Any, dst: tuple) -> bool:
    """Check a tensor shaped *src* broadcasts to *dst*."""
    import torch

    try:
        got = torch.broadcast_shapes(tuple(src), tuple(dst))
    except RuntimeError:
        return False
    return tuple(got) == tuple(dst)


def _seq_axis(full: tuple, h_shape: Any, axis: int | None) -> int:
    """Resolve the step axis of the common sequence shape.

    An explicit *axis* is normalised against ``len(full)``.  ``None``
    derives it: the step axis is the one whose removal leaves the
    state shape *h* broadcasts into — requiring exactly one
    candidate.  Zero candidates means *h* matches no contraction of
    the operands' shape; several means the shapes alone cannot tell
    (e.g. a square ambiguous state/step extent) — both are honest
    ``ValueError``s, not guesses.
    """
    if axis is not None:
        return axis % len(full)
    cand = [
        i
        for i in range(len(full))
        if _broadcasts_into(h_shape, full[:i] + full[i + 1 :])
    ]
    if len(cand) != 1:
        raise ValueError(
            f"scan axis is not determined by the shapes:"
            f" candidates={tuple(cand)} for full={tuple(full)}"
            f" h={tuple(h_shape)}"
        )
    return cand[0]


def _scan_dtype(vals: dict, metas: list) -> Any:
    """Promoted dtype over the bound meta values."""
    import torch

    dt = vals[metas[0]].dtype
    for m in metas[1:]:
        dt = torch.promote_types(dt, vals[m].dtype)
    return dt


def _seq_args(vals: dict, info: dict, axis: int | None) -> tuple:
    """Normalise bound sequence operands to ``(T, N)`` row tensors.

    Returns ``(rows, h_flat, state_shape, T, n, dtype)`` — the launch
    arguments in kernel order.  Broadcasting is right-aligned: each
    operand's trailing dims land on state axes, so an operand without
    a step axis (a shared ``(d,)`` decay under ``(T, d)`` inputs, or
    a scalar) broadcasts over steps automatically.  Every sequence
    operand is materialised contiguous — the price of generality
    over strides, stated plainly.
    """
    import torch

    xs = torch.broadcast_tensors(*(vals[m] for m in info["seq"]))
    full = xs[0].shape
    if not full:
        raise ValueError("scan operands carry no step axis")
    if not xs[0].is_cuda:
        raise RuntimeError("generated scan kernels need CUDA tensors")
    ax = _seq_axis(full, vals[info["h"]].shape, axis)
    t_len = full[ax]
    state = full[:ax] + full[ax + 1 :]
    dt = _scan_dtype(vals, info["params"])
    rows = {
        m: x.to(dt).movedim(ax, 0).contiguous().view(t_len, -1)
        for m, x in zip(info["seq"], xs, strict=True)
    }
    h_flat = (
        torch.broadcast_to(vals[info["h"]].to(dt), state)
        .contiguous()
        .view(-1)
    )
    return rows, h_flat, state, t_len, h_flat.numel(), dt


def _seq_binding(kern: Any, info: dict, axis: int | None) -> Any:
    """Host wrapper for the sequential-scan kernel.

    ``fn(*vals)`` binds the spec metavars positionally (the handler's
    ``args`` order), normalises the sequence operands, and launches
    the fold.  ``axis`` is the declared step axis — ``None`` derives
    it from the state shape each call.
    """

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(info["params"], args, strict=True))
        rows, h_flat, state, t_len, n, dt = _seq_args(vals, info, axis)
        out = torch.empty(n, dtype=dt, device=h_flat.device)
        kern[((n + _SCAN_BLOCK - 1) // _SCAN_BLOCK,)](
            *(rows[m] for m in info["seq"]),
            h_flat,
            out,
            t_len,
            n,
            BLOCK=_SCAN_BLOCK,
        )
        return out.view(state)

    return fn


def _unrolled_binding(kern: Any, info: dict) -> Any:
    """Host wrapper for the unrolled carrier kernel.

    ``fn(*vals)`` binds the spec metavars positionally, broadcasts
    every operand (and the state) to one flat index space, and
    applies each leaf step in chronological order inside the kernel.
    """

    def fn(*args: Any, **_attrs: Any) -> Any:
        import torch

        vals = dict(zip(info["params"], args, strict=True))
        xs = torch.broadcast_tensors(*(vals[m] for m in info["params"]))
        if not xs[0].is_cuda:
            raise RuntimeError(
                "generated scan kernels need CUDA tensors"
            )
        shape = xs[0].shape
        dt = _scan_dtype(vals, info["params"])
        flats = [x.to(dt).contiguous().view(-1) for x in xs]
        n = flats[0].numel()
        out = torch.empty(n, dtype=dt, device=flats[0].device)
        kern[((n + _SCAN_BLOCK - 1) // _SCAN_BLOCK,)](
            *flats, out, n, BLOCK=_SCAN_BLOCK
        )
        return out.view(shape)

    return fn


def scan_triton_bindings(
    handlers: dict, *, axis: int | None = -2
) -> dict:
    """Generate a Triton scan kernel per ``applyd``-carrier handler.

    The second codegen tier: each handler whose *pattern*
    :func:`scan_spec` recognises gets a binding whose callable runs a
    generated sequential-scan kernel — one launch for the whole
    recurrence, real ``triton.jit`` source on disk under the kernel
    cache.  The step axis defaults to ``-2`` (the token axis of
    ``(B, T, d)`` layouts); a handler may pin ``"axis"`` (or the spec
    may declare it as an ``applyd`` attr), and ``axis=None`` derives
    the axis from the state shape at call time.

    Unrecognised or un-generatable specs are skipped — the handler
    keeps its torch path, exactly as the elementwise tier declines.
    """
    out: dict[str, Any] = {}
    for h in handlers.values():
        info = scan_spec(h["pattern"])
        if info is None:
            continue
        ax = h.get("axis", info.get("axis", axis))
        src = _scan_triton_source(info)
        path = _kernel_cache() / f"_gs_{h['kernel']}.py"
        path.write_text(src)
        kern = _load_generated(path, "_gs")
        if kern is None:
            continue  # module load declined — keep the torch path
        if info["kind"] == "seq":
            out[h["kernel"]] = _seq_binding(kern, info, ax)
        else:
            out[h["kernel"]] = _unrolled_binding(kern, info)
    return out


def scan_handlers(
    terms: list, *, axis: int | None = -2, covered: set | None = None
) -> dict:
    """Mint handler entries for ``applyd``-carrier scan terms.

    The carrier analogue of :func:`gen_handlers`: ``scan_<i>`` entries
    ``{pattern, kernel, args, axis}`` for each term whose spec is a
    recognised scan — the *sequence* form (an ``applyd(aff_diag(a,
    b), h)`` site whose operands carry a step axis) or the *unrolled*
    form (an ``affd_compose`` tree, e.g. what ``affd_scan2_lift`` /
    ``affd_scan4_lift`` leave behind).  ``axis`` is recorded in the
    entry as the declared step axis; ``None`` defers to call-time
    derivation.

    The caller owns the semantics check for the sequence form: the
    minted ``scan_<i>_k`` op's declared meaning is the fold, so the
    handler belongs on sites whose ``a``/``b`` operands genuinely
    carry a step axis — the binding raises on operands that do not.
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
        if scan_spec(pattern) is None:
            continue
        canon = _canon_concrete(pattern)
        if canon in skip or any(
            _canon_concrete(h["pattern"]) == canon for h in out.values()
        ):
            continue
        tag = f"scan_{len(out)}"
        out[tag] = {
            "pattern": pattern,
            "kernel": f"{tag}_k",
            "args": tuple(_spec_metas(pattern)),
            "axis": axis,
        }
    return out


def scan_triton_sink(handlers: dict, *, axis: int | None = -2) -> Any:
    """Build a ``TorchSink`` binding minted scan ops to Triton kernels.

    Same posture as :func:`triton_sink`: ``supported_ops`` reports the
    ambient table plus every generated scan kernel — the feasibility
    bound the board prices the carrier claims against — and a claim
    on an unrecognised spec stays unpriced-unlowerable.
    """
    import catopt_torch.torch_bridge as tb
    from catopt_core.ops import OpTable
    from catopt_torch.adapters import TorchSink

    bindings = dict(tb._IR_TO_TORCH)
    bindings.update(scan_triton_bindings(handlers, axis=axis))
    table = OpTable.core().register(torch_bindings=bindings)
    return TorchSink(ops=table)
