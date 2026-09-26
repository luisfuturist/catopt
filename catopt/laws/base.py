"""Shared machinery for the law modules.

``catopt.laws`` is split by domain — tensor algebra (:mod:`.tensor`),
the scan monoids (:mod:`.scan`), and the non-local graph passes
(:mod:`.pairing`).  This module holds the pieces every domain needs:

* :func:`R` — the ``Rewrite`` constructor shorthand every law uses.
* ``_SHAPE_MEMO`` / :func:`_shape_of` — the content-keyed shape memo
  threading ``catopt.typing._shape_of`` through check hooks.
* Small term predicates shared across laws: scalar/row/channel scale
  classification for the diagonal-naturality guards.
"""

from typing import Any

from catopt.egraph import Rewrite


def R(
    name: str,
    lhs: Any,
    rhs: Any,
    law: str = "",
    check=None,
    derive=None,
) -> Rewrite:
    """Shorthand for creating a rewrite rule."""
    return Rewrite(
        name=name, lhs=lhs, rhs=rhs, law=law, check=check, derive=derive
    )


#: Content-keyed shape memo for the check-hook path.  The bound terms
#: it sees are ``EGraph.any_term`` resolutions — interned Op objects —
#: and ``typing._shape_of`` is already memo-aware: threading one memo
#: turns an exponential DAG re-walk into a linear one.  Term keys hold
#: their objects alive, so no keepalive pins are needed.
_SHAPE_MEMO: dict = {}


def _shape_of(t: Any):
    """Best-effort shape of a bound term (delegates to cost model)."""
    from catopt.typing import _shape_of as _so

    return _so(t, _SHAPE_MEMO)


def _is_scalar(t: Any) -> bool:
    """True if the bound term is a scalar (shape ())."""
    s = _shape_of(t)
    return s == () or s == tuple()


def _is_row_scale(t: Any) -> bool:
    """Per-ROW scale: broadcasts to (B,T,1) — last dim is 1 (or scalar)."""
    s = _shape_of(t)
    if not isinstance(s, tuple):
        return s == ()
    return len(s) == 0 or s[-1] == 1


def _is_channel_scale(bound: dict) -> bool:
    """Per-CHANNEL scale: broadcasts over the weight's input dim.

    c may be scalar, (in,), or (1,...,1,in) — i.e. every non-last dim
    must be 1 and the last must equal W's input feature dim.
    """
    c, w = bound.get("c"), bound.get("W")
    cs, ws = _shape_of(c), _shape_of(w)
    if not isinstance(cs, tuple):
        return cs == ()
    if not isinstance(ws, tuple) or len(ws) < 1:
        return False
    if len(cs) == 0:
        return True
    return cs[-1] == ws[-1] and all(d == 1 for d in cs[:-1])
