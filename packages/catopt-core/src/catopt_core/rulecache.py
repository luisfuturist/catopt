"""Persistent cache for synthesized rewrite rules.

:func:`catopt_core.meta.synthesize_rules` is the expensive phase of the
meta-optimization loop — critical-pair completion plus per-candidate
validation — but its output is pure data.  This module persists it:
``synthesize_rules_cached`` returns the same derived rules on a second
run without re-deriving them.

Serialization scheme
--------------------
A synthesized :class:`Rewrite` has Op-tree lhs/rhs patterns (metavar
strings, ``$attr:`` attribute metavars, ``@1:``/``@2:`` derive
placeholders) plus Python callables for ``check``/``derive``.  The
callables are NOT serialized — instead each guarded rule stores its
*re-expression spec*: the parent pair names and the two substitution
maps ``(pat1, pat2)`` that map each parent's metavariables to patterns
over the derived rule's metavars (recorded on ``rule.guard_pats`` at
synthesis time).  At load time the composite hooks are rebuilt by
re-running :func:`meta._compose_guards` against the parent rules
resolved by name from the caller's ruleset — the composition machinery
is cheap; it is the search/validation that was expensive.

Cache key
---------
``sha256`` over ``{version, ruleset_fp, seeds_fp, params}`` where

* ``ruleset_fp`` — hash of the sorted per-rule records
  ``{name, lhs, rhs, check_sig, derive_sig}`` (hook signatures hash
  source text when inspectable, else the qualified name — worst case a
  stale signature just misses);
* ``seeds_fp`` — hash of the serialized seed terms, in order;
* ``params`` — fuel / numeric_check / require_overlap / emit_subsumed.

Any change to the parent ruleset, seeds, or synthesis parameters
produces a different key — a stale cache misses cleanly rather than
silently loading rules derived under different assumptions.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import catopt_core.meta as M
from catopt_core.egraph import Rewrite
from catopt_core.ir import Const, Op, Param, TensorType, Var

__all__ = [
    "RuleCache",
    "cache_key",
    "ruleset_fingerprint",
    "seed_fingerprint",
    "synthesize_rules_cached",
]

#: Bump when the on-disk format or reconstruction semantics change.
CACHE_VERSION = 1


# ---------------------------------------------------------------------------
#  Term / attribute serialization  (Op trees, metvars, placeholders — pure data)
# ---------------------------------------------------------------------------


def _enc_attr(v: Any) -> Any:
    """JSON-safe encoding of an attribute value (or a ``$attr:`` binding
    value — strings stay strings since they are metavar references)."""
    if isinstance(v, tuple):
        return {"__tuple__": [_enc_attr(x) for x in v]}
    if isinstance(v, list):
        return {"__list__": [_enc_attr(x) for x in v]}
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    raise TypeError(f"unserializable attribute value: {v!r}")


def _dec_attr(v: Any) -> Any:
    if isinstance(v, dict):
        if "__tuple__" in v:
            return tuple(_dec_attr(x) for x in v["__tuple__"])
        if "__list__" in v:
            return [_dec_attr(x) for x in v["__list__"]]
        raise ValueError(f"bad attr encoding: {v!r}")
    return v


def _enc_term(t: Any) -> Any:
    """Encode a (sub)term: Op tree, metavar string, or leaf."""
    if isinstance(t, str):
        return {"mvar": t}
    if isinstance(t, Const):
        return {"const": t.value}
    if isinstance(t, Var):
        return {"var": t.name, "shape": list(t.typ.shape)}
    if isinstance(t, Param):
        return {"param": t.name, "shape": list(t.typ.shape)}
    if isinstance(t, Op):
        return {
            "op": t.op,
            "args": [_enc_term(a) for a in t.args],
            "attrs": {k: _enc_attr(v) for k, v in t.attrs.items()},
        }
    raise TypeError(f"unserializable term: {t!r}")


def _dec_term(d: Any) -> Any:
    if not isinstance(d, dict) or len(d) < 1:
        raise ValueError(f"bad term encoding: {d!r}")
    if "mvar" in d:
        return d["mvar"]
    if "const" in d:
        return Const(d["const"])
    if "var" in d:
        return Var(d["var"], TensorType(tuple(d["shape"])))
    if "param" in d:
        return Param(d["param"], TensorType(tuple(d["shape"])))
    if "op" in d:
        return Op.make(
            d["op"],
            *[_dec_term(a) for a in d["args"]],
            **{k: _dec_attr(v) for k, v in d["attrs"].items()},
        )
    raise ValueError(f"bad term encoding: {d!r}")


def _enc_binding(m: dict) -> list:
    """Encode a substitution/re-expression map.  ``$attr:`` values are
    attribute values (concrete constants or metavar-name strings);
    every other key maps to a term/pattern."""
    out = []
    for k, v in m.items():
        if k.startswith("$attr:"):
            out.append([k, {"a": _enc_attr(v)}])
        else:
            out.append([k, {"t": _enc_term(v)}])
    return out


def _dec_binding(pairs: list) -> dict:
    out = {}
    for k, enc in pairs:
        if "a" in enc:
            out[k] = _dec_attr(enc["a"])
        else:
            out[k] = _dec_term(enc["t"])
    return out


# ---------------------------------------------------------------------------
#  Rule records
# ---------------------------------------------------------------------------


def _enc_rule(rule: Rewrite) -> dict:
    """Serialize a synthesized rule: name, law, provenance, patterns,
    and — when guarded — the (pat1, pat2) re-expression spec recorded on
    ``rule.guard_pats``.  Callables are never serialized."""
    spec = getattr(rule, "guard_pats", None)
    guarded = rule.check is not None or rule.derive is not None
    if guarded and spec is None:
        raise TypeError(
            f"rule {rule.name!r} has check/derive hooks but no "
            "guard_pats re-expression spec — only rules emitted by "
            "meta.synthesize_rules are cacheable"
        )
    return {
        "name": rule.name,
        "law": rule.law,
        "parents": list(
            getattr(rule, "parents", M.SYNTH_PARENTS.get(rule.name, ()))
        ),
        "lhs": _enc_term(rule.lhs),
        "rhs": _enc_term(rule.rhs),
        "guard": (
            {
                "pat1": _enc_binding(spec[0]),
                "pat2": _enc_binding(spec[1]),
            }
            if (guarded and spec is not None)
            else None
        ),
    }


def _dec_rule(
    rec: dict, parents_by_name: dict[str, Rewrite]
) -> Rewrite:
    """Rebuild a rule.  Guarded rules re-run ``_compose_guards`` on the
    stored re-expression maps — the parents must be present (by name)
    in *parents_by_name*, or a ``KeyError`` propagates and the whole
    load is treated as a miss."""
    lhs = _dec_term(rec["lhs"])
    rhs = _dec_term(rec["rhs"])
    check = derive = None
    pats = None
    guard = rec.get("guard")
    if guard is not None:
        p1, p2 = rec["parents"]
        r1 = parents_by_name[p1]
        r2 = parents_by_name[p2]
        pats = (
            _dec_binding(guard["pat1"]),
            _dec_binding(guard["pat2"]),
        )
        check, drv = M._compose_guards(r1, r2, pats[0], pats[1])
        # Same policy as synthesize_rules.offer: the composite derive is
        # only attached when the RHS actually carries "@i:" placeholders.
        if not (drv is not None and M._rhs_derive_placeholders(rhs)):
            drv = None
        derive = drv
    rule = Rewrite(
        name=rec["name"],
        lhs=lhs,
        rhs=rhs,
        law=rec["law"],
        check=check,
        derive=derive,
    )
    parents = tuple(rec.get("parents") or ())
    object.__setattr__(rule, "parents", parents)
    if parents:
        M.SYNTH_PARENTS[rule.name] = parents
    if pats is not None:
        object.__setattr__(rule, "guard_pats", pats)
    return rule


# ---------------------------------------------------------------------------
#  Fingerprints / cache key
# ---------------------------------------------------------------------------


def _sha(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()


def _hook_sig(fn) -> str:
    """Stability signature for a check/derive callable: source text when
    inspectable, else module+qualname.  An unstable signature only ever
    causes a cache MISS, never a wrong hit — conservative by design."""
    if fn is None:
        return ""
    if isinstance(fn, functools.partial):
        return (
            f"partial({_hook_sig(fn.func)};{fn.args!r};{fn.keywords!r})"
        )
    ident = f"{getattr(fn, '__module__', '')}:{getattr(fn, '__qualname__', '')}"
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = ""
    return hashlib.sha256(f"{ident}|{src}".encode()).hexdigest()[:16]


def ruleset_fingerprint(rules: Iterable[Rewrite]) -> str:
    """Hash of the parent ruleset: every rule's name, serialized lhs/rhs
    patterns, and hook signatures — sorted so rule order doesn't matter."""
    entries = sorted(
        json.dumps(
            {
                "name": r.name,
                "lhs": _enc_term(r.lhs),
                "rhs": _enc_term(r.rhs),
                "check": _hook_sig(r.check),
                "derive": _hook_sig(r.derive),
            },
            sort_keys=True,
        )
        for r in rules
    )
    return _sha(entries)


def seed_fingerprint(seed_terms: Iterable[Any]) -> str:
    """Hash of the serialized seed terms, in order (seed order affects
    which derivations are explored and rule naming)."""
    return _sha(
        json.dumps(_enc_term(s), sort_keys=True) for s in seed_terms
    )


def cache_key(
    rules: Iterable[Rewrite],
    seed_terms: Iterable[Any] = (),
    *,
    fuel: int = 512,
    numeric_check: bool = True,
    require_overlap: bool = True,
    emit_subsumed: bool = False,
) -> str:
    """The cache key for one synthesis problem instance: ruleset
    fingerprint + seed fingerprint + synthesis params + format version."""
    rules = list(rules)
    payload = {
        "version": CACHE_VERSION,
        "ruleset": ruleset_fingerprint(rules),
        "seeds": seed_fingerprint(seed_terms),
        "params": {
            "fuel": fuel,
            "numeric_check": numeric_check,
            "require_overlap": require_overlap,
            "emit_subsumed": emit_subsumed,
        },
    }
    return _sha([json.dumps(payload, sort_keys=True)])


# ---------------------------------------------------------------------------
#  RuleCache + convenience wrapper
# ---------------------------------------------------------------------------


def _default_cache_dir() -> Path:
    env = os.environ.get("CATOPT_RULECACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "catopt" / "rulecache"


class RuleCache:
    """Filesystem cache of synthesized rule sets, one JSON file per key."""

    def __init__(self, cache_dir: Any):
        self.dir = Path(cache_dir)

    def path(self, key: str) -> Path:
        return self.dir / f"rules-{key}.json"

    def store(self, key: str, rules: Iterable[Rewrite]) -> Path:
        """Persist *rules* under *key*.  Raises ``TypeError`` if a rule
        carries hooks without a ``guard_pats`` spec (not synthesizable-
        output), rather than silently dropping its side conditions."""
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "key": key,
            "rules": [_enc_rule(r) for r in rules],
        }
        tmp = self.dir / f".rules-{key}.tmp"
        tmp.write_text(json.dumps(payload, sort_keys=True))
        tmp.replace(self.path(key))
        return self.path(key)

    def load(
        self, key: str, rules: Iterable[Rewrite] = ()
    ) -> list[Rewrite] | None:
        """Load the rule set stored under *key*, or ``None`` on a miss.

        *rules* is the parent ruleset: guarded rules rebuild their
        composite check/derive from the parents' hooks, resolved by
        name.  A missing parent, a version/key mismatch, or a corrupt
        file is a clean miss — never a partially-loaded set."""
        p = self.path(key)
        if not p.exists():
            return None
        try:
            payload = json.loads(p.read_text())
            if (
                payload.get("version") != CACHE_VERSION
                or payload.get("key") != key
            ):
                return None
            by_name = {r.name: r for r in rules}
            return [_dec_rule(rec, by_name) for rec in payload["rules"]]
        except Exception:
            return None


def synthesize_rules_cached(
    rules: list[Rewrite],
    seed_terms: Iterable[Any] = (),
    *,
    fuel: int = 512,
    numeric_check: bool = True,
    require_overlap: bool = True,
    emit_subsumed: bool = False,
    cache_dir: Any = None,
) -> list[Rewrite]:
    """``meta.synthesize_rules`` with persistence: on a repeat call with
    the same ruleset, seeds, and parameters the derived rules are loaded
    from *cache_dir* (composite guards rebuilt from the parent hooks)
    instead of re-derived.

    Loaded rules are ordinary ``Rewrite`` objects — they fire in
    e-graphs, carry ``.parents`` provenance, and re-register in
    ``meta.SYNTH_PARENTS``.  ``cache_dir`` defaults to
    ``$CATOPT_RULECACHE_DIR`` or ``~/.cache/catopt/rulecache``."""
    rules = list(rules)
    cache = RuleCache(cache_dir or _default_cache_dir())
    key = cache_key(
        rules,
        seed_terms,
        fuel=fuel,
        numeric_check=numeric_check,
        require_overlap=require_overlap,
        emit_subsumed=emit_subsumed,
    )
    hit = cache.load(key, rules)
    if hit is not None:
        return hit
    derived = M.synthesize_rules(
        rules,
        seed_terms,
        fuel=fuel,
        numeric_check=numeric_check,
        require_overlap=require_overlap,
        emit_subsumed=emit_subsumed,
    )
    cache.store(key, derived)
    return derived
