"""Structural term utilities + certificate verification."""
from __future__ import annotations

from typing import Any

from catopt_core.egraph.certs import (
    Certificate,
    CertificateVerificationError,
)
from catopt_core.ir import Op, op_repr


def _term_paths(term: Any, prefix: tuple = ()):
    """Yield the child-index path of every subterm, root first."""
    yield prefix
    if isinstance(term, Op):
        for i, a in enumerate(term.args):
            yield from _term_paths(a, (*prefix, i))


def _iter_ops(term: Any):
    """Yield every Op node in a term (for the structural-size tie-break)."""
    if isinstance(term, Op):
        yield term
        for a in term.args:
            yield from _iter_ops(a)


# ---------------------------------------------------------------------------
#  Certificate verification — replay on real terms, no e-graph
# ---------------------------------------------------------------------------


def _term_match(
    pattern: Any, term: Any, _subst: dict | None = None
) -> dict | None:
    """Structural match of a pattern against a plain *term* (no e-graph).

    Mirrors :meth:`EGraph._match` semantics at term granularity: string
    leaves in the pattern are metavariables bound to subterms (repeated
    metavariables must bind structurally equal terms); string-valued
    attributes are attribute metavariables bound under ``"$attr:"``
    keys; concrete leaves (Var/Const/Param) match by ``repr``.
    Returns the bindings dict, or ``None`` on mismatch.
    """
    subst = {} if _subst is None else _subst
    if isinstance(pattern, str):
        prev = subst.get(pattern)
        if prev is None:
            subst[pattern] = term
            return subst
        return subst if op_repr(prev) == op_repr(term) else None
    if isinstance(pattern, Op):
        if not isinstance(term, Op) or term.op != pattern.op:
            return None
        if len(term.args) != len(pattern.args):
            return None
        if set(term.attrs) != set(pattern.attrs):
            return None
        for k, pv in pattern.attrs.items():
            nv = term.attrs[k]
            if isinstance(pv, str):
                key = "$attr:" + pv
                if key in subst:
                    if subst[key] != nv:
                        return None
                else:
                    subst[key] = nv
            elif nv != pv:
                return None
        for pa, ta in zip(pattern.args, term.args, strict=True):
            if _term_match(pa, ta, subst) is None:
                return None
        return subst
    # concrete leaf (Const/Param/Var embedded in the pattern)
    return subst if repr(pattern) == repr(term) else None


def _term_instantiate(pattern: Any, subst: dict) -> Any:
    """Instantiate a pattern with term-valued bindings (pure terms).

    The term-level analogue of :meth:`EGraph._instantiate`: metavariable
    strings map to terms, ``"$attr:"`` keys resolve attribute
    metavariables, concrete leaves pass through unchanged.
    """
    if isinstance(pattern, str):
        return subst[pattern]
    if isinstance(pattern, Op):
        args = [_term_instantiate(a, subst) for a in pattern.args]
        attrs = {}
        for k, v in pattern.attrs.items():
            if isinstance(v, str):
                attrs[k] = subst.get("$attr:" + v, v)
            else:
                attrs[k] = v
        return Op.make(pattern.op, *args, **attrs)
    return pattern


def _subterm(term: Any, path: tuple) -> Any:
    """The subterm of *term* at child-index path, or None if absent."""
    for i in path:
        if not isinstance(term, Op) or i >= len(term.args):
            return None
        term = term.args[i]
    return term


def _replace_subterm(term: Any, path: tuple, new: Any) -> Any:
    """*term* with the subterm at *path* replaced by *new*."""
    if not path:
        return new
    if not isinstance(term, Op) or path[0] >= len(term.args):
        raise CertificateVerificationError(
            f"cannot descend path {list(path)} in {op_repr(term)}"
        )
    i = path[0]
    args = list(term.args)
    args[i] = _replace_subterm(args[i], path[1:], new)
    return Op.make(term.op, *args, **dict(term.attrs))


def verify_certificate(
    src_term: Any, cert: Certificate, *, strict: bool = False
) -> Any:
    """Replay a certificate's rule applications on real terms.

    For each step: descend to ``path`` in the evolving term, re-match
    the rule's LHS against the subterm found there, check the recorded
    bindings and the recorded LHS/RHS instances, re-run ``check``/
    ``derive`` side conditions, instantiate the RHS, and substitute.
    ``egraph_dependent`` steps cannot be replayed standalone — the
    recorded ``lhs`` must still be present at ``path``, then ``rhs`` is
    substituted as a trusted assertion; ``strict=True`` rejects any
    certificate containing them.

    Returns the reconstructed term — ``cert.dst`` on success.
    Raises :class:`CertificateVerificationError` on any mismatch.
    """
    if strict and cert.n_egraph_dependent:
        raise CertificateVerificationError(
            f"{cert.n_egraph_dependent} e-graph-dependent step(s) "
            "cannot be replayed standalone"
        )
    if cert.src is not None and op_repr(src_term) != op_repr(cert.src):
        raise CertificateVerificationError(
            f"source mismatch: certificate proves {op_repr(cert.src)}, "
            f"got {op_repr(src_term)}"
        )
    current = src_term
    for i, step in enumerate(cert.steps):
        sub = _subterm(current, step.path)
        if sub is None:
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): path {list(step.path)} "
                f"absent in {op_repr(current)}"
            )
        if step.egraph_dependent:
            if op_repr(sub) != op_repr(step.lhs):
                raise CertificateVerificationError(
                    f"step {i} (e-graph-dependent): expected "
                    f"{op_repr(step.lhs)} at {list(step.path)}, "
                    f"found {op_repr(sub)}"
                )
            current = _replace_subterm(current, step.path, step.rhs)
            continue
        rule = cert.rules.get(step.rule)
        if rule is None:
            raise CertificateVerificationError(
                f"step {i}: unknown rule {step.rule!r}"
            )
        if op_repr(step.lhs) != op_repr(sub):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): recorded LHS instance "
                f"{op_repr(step.lhs)} != subterm {op_repr(sub)}"
            )
        m = _term_match(rule.lhs, sub)
        if m is None:
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): LHS does not match "
                f"{op_repr(sub)}"
            )
        for k, v in step.bindings.items():
            if k not in m:
                continue  # derived binding — checked via derive/rhs below
            same = (
                m[k] == v
                if k.startswith("$attr:")
                else op_repr(m[k]) == op_repr(v)
            )
            if not same:
                raise CertificateVerificationError(
                    f"step {i} ({step.rule}): binding {k} tampered"
                )
        if rule.check is not None and not rule.check(m):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): side condition fails on replay"
            )
        inst = dict(m)
        for k, v in step.bindings.items():
            inst.setdefault(k, v)  # derived $attr bindings
        if rule.derive is not None:
            extra = rule.derive(m)
            if extra is None:
                raise CertificateVerificationError(
                    f"step {i} ({step.rule}): derive vetoed on replay"
                )
            inst.update(extra)
        rhs = _term_instantiate(rule.rhs, inst)
        if op_repr(rhs) != op_repr(step.rhs):
            raise CertificateVerificationError(
                f"step {i} ({step.rule}): recorded RHS "
                f"{op_repr(step.rhs)} != instantiated {op_repr(rhs)}"
            )
        current = _replace_subterm(current, step.path, rhs)
    if op_repr(current) != op_repr(cert.dst):
        raise CertificateVerificationError(
            f"replay produced {op_repr(current)}, "
            f"certificate claims {op_repr(cert.dst)}"
        )
    return current

