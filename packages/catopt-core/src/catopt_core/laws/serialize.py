"""Rewrite laws as data — the ``Rewrite`` ↔ JSON record seam.

A law is a 2-cell: a term pair, a side condition, and provenance.
With :mod:`catopt_core.laws.cond` the side condition is data too, so
the whole record round-trips through JSON for the laws that have no
procedural remainder:

* :func:`law_to_data` / :func:`law_from_data` — the record codec.
  Patterns encode through :func:`catopt_core.ir.term_to_data` (the
  same scheme ``rulecache`` persists, now canonical at the IR layer);
  the condition encodes through ``cond_to_data``, the derive spec
  through ``derive_to_data``; ``tags`` / ``derivation`` / the
  error-bound fields are already JSON scalars.
* :func:`missing_hooks` — the honesty boundary.  ``check`` beyond the
  folded ``cond`` and ``derive`` beyond the folded ``dspec`` are
  Python callables; data cannot carry them.  A rule that has them
  still serializes its pattern + ``cond`` + ``dspec``, but the record
  flags ``"serializable": false`` and lists the dropped hooks rather
  than weakening the law silently: a reconstructed ``rms_norm_fold``
  fires and mints ``rms_norm(u, w, dim="ND", eps="EP")`` — the unbound
  attr metavars fall back to their literal names, a visible scar
  instead of a hidden veto.
* :func:`alpha_key` — the alpha-normal ``(lhs, rhs)`` structural key
  the evidence store keys lemma rows by.  The same canonicalisation
  ``tools/law_proposal._key`` always used, lifted to core so the
  store needs no torch-side import to key a row.

Not re-exported by ``catopt_core.laws.__init__`` — import it as
``catopt_core.laws.serialize`` (same posture as ``laws.cond``).
"""

from __future__ import annotations

from typing import Any

from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op, term_from_data, term_to_data
from catopt_core.laws.cond import (
    compile_derive,
    compile_guard,
    cond_to_data,
    derive_to_data,
)

__all__ = [
    "LAW_FORMAT",
    "alpha_key",
    "law_from_data",
    "law_to_data",
    "missing_hooks",
]

#: Bump when the record layout or reconstruction semantics change.
#: v2 adds the ``"dspec"`` field — declarative derive specs are data.
LAW_FORMAT = 2


# ---------------------------------------------------------------------------
#  Alpha-normal structural key — metavar-renamed laws share one key
# ---------------------------------------------------------------------------


def _attr_canon(v: Any, mv: dict) -> Any:
    """Canonical form of one attr value under the metavar map *mv*."""
    if isinstance(v, str):
        if v.startswith("$attr:"):
            return ("av", v)
        return ("m", mv.setdefault(v, len(mv)))
    try:
        hash(v)
        return ("lit", v)
    except TypeError:
        return ("lit", repr(v))


def _canon(term: Any, mv: dict) -> Any:
    """Abstract a term to a leaf-renamed structural key.

    Distinct leaves (``Var`` / ``Param`` / a bare ``str`` metavariable)
    become shared metavar indices; ``Const`` stays literal; attrs are
    canonicalised.  Two terms are alpha-equal iff their keys coincide
    under a *shared* ``mv``.
    """
    if isinstance(term, Op):
        args = tuple(_canon(a, mv) for a in term.args)
        attrs = tuple(
            sorted(
                (k, _attr_canon(v, mv)) for k, v in term.attrs.items()
            )
        )
        return (term.op, args, attrs)
    if isinstance(term, Const):
        return ("c", term.value)
    if isinstance(term, str):
        return ("m", mv.setdefault(term, len(mv)))
    return ("m", mv.setdefault(repr(term), len(mv)))


def alpha_key(lhs: Any, rhs: Any) -> tuple[Any, Any]:
    """Return the alpha-normal ``(lhs, rhs)`` key of a law instance.

    The content-addressed identity the evidence store keys rows by:
    metavariable names are abstracted to shared indices, so the same
    2-cell spelled with different metavar names hashes identically
    (``repr`` the pair for a string key).
    """
    mv: dict = {}
    return _canon(lhs, mv), _canon(rhs, mv)


# ---------------------------------------------------------------------------
#  The honesty boundary — which hooks data cannot carry
# ---------------------------------------------------------------------------


def _proc_check(rule: Rewrite) -> bool:
    """Return True when *rule*'s check is code beyond the folded cond.

    ``Rewrite.__post_init__`` folds ``cond`` into ``check`` through
    :func:`compile_guard`, which builds one of two closures — a
    cond-only guard or a cond-then-check guard.  Those are two
    distinct code objects, so comparing the live check's ``__code__``
    against a fresh cond-only guard's distinguishes them exactly: a
    partial, a composed guard (``meta._compose_guards``), or any
    hand-written callable is reported as procedural — the strict
    posture, since a missed procedural check would silently weaken a
    stored law.
    """
    if rule.check is None:
        return False
    if rule.cond is None:
        return True
    probe = compile_guard(rule.cond, None)
    return getattr(rule.check, "__code__", None) is not probe.__code__


def _proc_derive(rule: Rewrite) -> bool:
    """Return True when *rule*'s derive is code beyond the folded spec.

    Same probe as :func:`_proc_check`: ``Rewrite.__post_init__`` folds
    ``dspec`` into ``derive`` through :func:`compile_derive`, which
    builds one of two closures — spec-only or spec-then-derive.  A
    spec-only fold compares ``__code__``-equal to a fresh probe;
    anything else (a hand-written callable, an ``as_derive`` partial
    never recast, a spec+code composite) is procedural — the strict
    posture.
    """
    if rule.derive is None:
        return False
    if rule.dspec is None:
        return True
    probe = compile_derive(rule.dspec, None)
    return getattr(rule.derive, "__code__", None) is not probe.__code__


def missing_hooks(rule: Rewrite) -> tuple[str, ...]:
    """Return the names of Python hooks *rule* carries beyond data.

    ``"check"`` when the side condition has a procedural remainder
    (or is entirely procedural — no ``cond``); ``"derive"`` when the
    rule computes RHS attributes in code beyond a declarative
    ``dspec``.  An empty tuple means the rule is fully serializable:
    :func:`law_from_data` rebuilds a behaviourally-identical
    ``Rewrite`` from the record alone.
    """
    out: list[str] = []
    if _proc_check(rule):
        out.append("check")
    if _proc_derive(rule):
        out.append("derive")
    return tuple(out)


# ---------------------------------------------------------------------------
#  The record codec
# ---------------------------------------------------------------------------


def law_to_data(rule: Rewrite) -> dict[str, Any]:
    """Serialise *rule* to a JSON-safe record.

    Every field is data: the pattern pair through
    :func:`catopt_core.ir.term_to_data`, the side condition through
    ``cond_to_data``, the derive spec through ``derive_to_data``, and
    the provenance / taxonomy fields as scalars.  ``"serializable"``
    is True only when :func:`missing_hooks` is empty — for a rule with
    ``check``/``derive`` code the record is honest: pattern + cond +
    dspec are stored, ``"missing_hooks"`` names what the data does not
    carry.
    """
    missing = missing_hooks(rule)
    return {
        "version": LAW_FORMAT,
        "name": rule.name,
        "law": rule.law,
        "lhs": term_to_data(rule.lhs),
        "rhs": term_to_data(rule.rhs),
        "cond": cond_to_data(rule.cond),
        "dspec": derive_to_data(rule.dspec),
        "tags": sorted(rule.tags),
        "derivation": list(rule.derivation),
        "error_bound": rule.error_bound,
        "bound_norm": rule.bound_norm,
        "serializable": not missing,
        "missing_hooks": list(missing),
    }


def law_from_data(data: dict[str, Any]) -> Rewrite:
    """Rebuild a :class:`Rewrite` from :func:`law_to_data` output.

    Hooks the record cannot carry are absent on the rebuilt rule —
    check ``data["missing_hooks"]`` (or ``"serializable"``) before
    trusting a reconstructed lemma to fire identically.  A stored
    ``cond`` re-folds into ``check`` and a stored ``dspec`` into
    ``derive`` at construction, so a full-data record reproduces the
    original rule's behaviour.
    """
    if data.get("version") != LAW_FORMAT:
        raise ValueError(
            f"bad law record version: {data.get('version')!r}"
        )
    return Rewrite(
        name=data["name"],
        lhs=term_from_data(data["lhs"]),
        rhs=term_from_data(data["rhs"]),
        law=data.get("law", ""),
        tags=frozenset(data.get("tags", ())),
        error_bound=data.get("error_bound"),
        bound_norm=data.get("bound_norm", "spectral"),
        derivation=tuple(data.get("derivation", ())),
        cond=data.get("cond"),
        dspec=data.get("dspec"),
    )
