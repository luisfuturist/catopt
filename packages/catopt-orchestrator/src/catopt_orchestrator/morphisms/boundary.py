"""Multi-input boundary fallback — the activation edge of a pair.

When the composer's single-positional-arg ``boundary`` check declines
(one side is a multi-input block), this fallback reads the same
two-probe evidence on the pair's *activation* position.  Split out of
:mod:`catopt_orchestrator.morphisms` (plan 0011).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from catopt_core.laws.pairing import _exact_equal, _is_tensor

from .signature import BlockSig, _act_index

if TYPE_CHECKING:
    from .graph import MorphismNode


# ---------------------------------------------------------------------------
#  Multi-input boundary fallback — the activation edge of a block pair
# ---------------------------------------------------------------------------


def _mi_consumers(
    io: Any, captured: Any, out_obj: Any, out_val: Any
) -> list[str]:
    """Blocks whose captured args hold ``out`` — identity + value.

    The position-agnostic counterpart of the composer's
    ``_plain_consumers``: any arg position (activation or context)
    holding the same live object with the captured value counts.
    """
    hits = []
    for n, m in (io or {}).items():
        c_args = captured.get(n, ((), {}))[0]
        for arg, real in zip(
            c_args, m.get("in_objs", ()), strict=False
        ):
            if (
                real is out_obj
                and _is_tensor(arg)
                and _is_tensor(out_val)
                and _exact_equal(arg, out_val)
            ):
                hits.append(n)
                break
    return hits


def _mi_io_has_value(captured: Any, model_out: Any, val: Any) -> bool:
    """Check ``val`` appears verbatim in the captured flow."""
    if (
        _is_tensor(model_out)
        and _is_tensor(val)
        and _exact_equal(model_out, val)
    ):
        return True
    return any(
        _is_tensor(a) and _is_tensor(val) and _exact_equal(a, val)
        for args, _ in captured.values()
        for a in args
    )


def _mi_residual_probe(
    name_a: str,
    name_b: str,
    captured2: Any,
    io2: Any,
    ka: int,
    kb: int,
) -> bool:
    """Second-probe confirmation of a multi-input residual edge."""
    ca = captured2.get(name_a)
    cb = captured2.get(name_b)
    ia = io2.get(name_a)
    if ca is None or cb is None or ia is None or ia["calls"] != 1:
        return False
    args_a, args_b = ca[0], cb[0]
    if ka >= len(args_a) or kb >= len(args_b):
        return False
    a_in, b_in, a_out = args_a[ka], args_b[kb], ia["out"]
    if not (
        _is_tensor(a_in) and _is_tensor(b_in) and _is_tensor(a_out)
    ):
        return False
    return bool(
        tuple(a_in.shape) == tuple(a_out.shape)
        and _exact_equal(b_in, a_in + a_out)
    )


def _mi_a_mode(
    name_a: str,
    name_b: str,
    captured: Any,
    io: Any,
    captured2: Any,
    io2: Any,
    sig_a: BlockSig | None,
    in_objs_b: tuple,
    kb: int,
    ka: int,
    a_in: Any,
    b_in: Any,
    a_out: Any,
    a_out_obj: Any,
) -> str | None:
    """Classify the A→B activation edge: ``chain`` / ``residual``.

    ``chain`` — B's activation arg is literally A's output object and
    nothing else consumed it; ``residual`` — the arg equals
    ``a_in + a_out`` on both probes.
    """
    a_fans = _mi_consumers(io, captured, a_out_obj, a_out)
    if kb < len(in_objs_b) and in_objs_b[kb] is a_out_obj:
        # B literally consumed A's output at its activation position —
        # chain only when nothing else did.
        if a_fans != [name_b]:
            return None
        return "chain"
    if a_fans or sig_a is None:
        return None
    if not (
        tuple(a_in.shape) == tuple(a_out.shape)
        and _exact_equal(b_in, a_in + a_out)
        and _mi_residual_probe(name_a, name_b, captured2, io2, ka, kb)
    ):
        return None
    return "residual"


def _mi_b_mode(
    captured: Any,
    io: Any,
    ib: dict,
    b_in: Any,
    model_out: Any,
    model_out_obj: Any,
) -> str | None:
    """Classify B's output consumption: plain / ``_wrapped`` / None."""
    b_out, b_out_obj = ib["out"], ib["out_obj"]
    if not _is_tensor(b_out):
        return None
    plain_ev = bool(_mi_consumers(io, captured, b_out_obj, b_out)) or (
        b_out_obj is model_out_obj
        and _is_tensor(model_out)
        and _exact_equal(model_out, b_out)
    )
    wrapped_ev = tuple(b_in.shape) == tuple(
        b_out.shape
    ) and _mi_io_has_value(captured, model_out, b_in + b_out)
    if plain_ev == wrapped_ev:
        return None
    return "_wrapped" if wrapped_ev else ""


def _mi_boundary(
    name_a: str,
    name_b: str,
    captured: Any,
    io: Any,
    captured2: Any,
    io2: Any,
    nodes: dict[str, MorphismNode],
) -> str | None:
    """Classify A→B when the composer's single-arg check declined.

    The composer's ``boundary`` requires a single positional arg on
    both sides; a multi-input block ``B(y, cos, sin)`` returns
    ``None`` unconditionally.  This fallback reads the same evidence
    on B's *activation* position — ``chain`` when B's activation arg
    is literally A's output object (consumed by B alone),
    ``residual`` when it is ``a_in + a_out`` on both probes — and the
    same plain/wrapped downstream check on B's output.  Context args
    are not part of the wire: they enter B's own call site and pass
    through a composition unchanged.
    """
    node_a, node_b = nodes.get(name_a), nodes.get(name_b)
    sig_a = node_a.sig if node_a is not None else None
    sig_b = node_b.sig if node_b is not None else None
    if not (
        (sig_a is not None and len(sig_a.inputs) > 1)
        or (sig_b is not None and len(sig_b.inputs) > 1)
    ):
        return None  # single-input pair — the composer already spoke
    ca = captured.get(name_a)
    cb = captured.get(name_b)
    ia = io.get(name_a)
    ib = io.get(name_b)
    if ca is None or cb is None or ia is None or ib is None:
        return None
    if ia["calls"] != 1 or ib["calls"] != 1:
        return None
    args_a, kw_a = ca
    args_b, kw_b = cb
    ka = _act_index(sig_a.inputs) if sig_a is not None else 0
    kb = _act_index(sig_b.inputs) if sig_b is not None else 0
    if kw_a or kw_b or ka >= len(args_a) or kb >= len(args_b):
        return None
    a_in, b_in, a_out = args_a[ka], args_b[kb], ia["out"]
    if not (
        _is_tensor(a_in) and _is_tensor(b_in) and _is_tensor(a_out)
    ):
        return None
    a_out_obj = ia["out_obj"]
    model = io.get("<model>", {})
    model_out, model_out_obj = model.get("out"), model.get("out_obj")
    if a_out_obj is model_out_obj:
        return None  # A's output escapes the pair entirely
    a_mode = _mi_a_mode(
        name_a,
        name_b,
        captured,
        io,
        captured2,
        io2,
        sig_a,
        ib.get("in_objs") or (),
        kb,
        ka,
        a_in,
        b_in,
        a_out,
        a_out_obj,
    )
    if a_mode is None:
        return None
    suffix = _mi_b_mode(
        captured, io, ib, b_in, model_out, model_out_obj
    )
    if suffix is None:
        return None
    return a_mode + suffix
