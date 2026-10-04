"""Proof-carrying certificate types + their data codec."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from catopt_core.ir import (
    attr_from_data,
    attr_to_data,
    term_from_data,
    term_to_data,
)

# ---------------------------------------------------------------------------
#  Proof-carrying merges — 2-morphisms as first-class data
# ---------------------------------------------------------------------------
#
#  Every merge records WHY it happened: ``apply_rule`` appends one
#  :class:`ProofEdge` per successful union (the canonical e-class ids on
#  the matched and produced sides plus the fired binding) and tags every
#  enode it instantiates with the creating rule application.  The
#  e-graph quotients the proof space — we keep a single witness per
#  merge (the first discovered), not all proofs.
#
#  :meth:`EGraph.certificate` replays that provenance into a positional
#  derivation — an ordered list of single-rule rewrites — connecting
#  the source term to an extracted member.  :func:`verify_certificate`
#  replays the steps on real terms, independent of the e-graph.  A step
#  that has no standalone justification (non-local merges such as the
#  diagram-level pairing pass, or manual unions) is recorded but flagged
#  ``egraph_dependent`` rather than silently trusted — UNLESS the pass
#  attached a synthesised :class:`Rewrite` via ``union(..., witness=...)``,
#  in which case the merge replays as an ordinary rule step.


@dataclass
class ProofEdge:
    """One recorded merge: the 2-morphism witnessing two e-classes.

    ``rule`` is the ``Rewrite.name`` that fired — ``None`` for merges
    performed outside rule application (non-local passes such as
    ``pair_shared_input_linears``, or direct ``union`` calls), or the
    name of a *synthesised* :class:`Rewrite` when the caller attached a
    replayable ``witness`` to :meth:`EGraph.union` (see there).
    ``a``/``b`` are the canonical e-class ids of the matched (LHS) and
    produced (RHS) sides *before* the merge; ``subst`` is the fired
    binding as ``(key, value)`` pairs — metavariables map to e-class
    ids, ``"$attr:"`` keys carry concrete attribute values.
    """

    rule: str | None
    a: int
    b: int
    subst: tuple = ()
    note: str = ""


@dataclass
class CertStep:
    """A single derivation step: ``rule`` applied at ``path``.

    ``path`` is a tuple of child indices locating the rewritten subterm
    inside the evolving term.  ``lhs``/``rhs`` are the concrete matched
    and produced term instances; ``bindings`` records metavariable ->
    term and ``"$attr:"`` -> value exactly as fired.

    ``egraph_dependent`` marks steps that cannot be replayed as a
    standalone rewrite — context-dependent merges (the pairing pass,
    manual unions) or derivation-budget exhaustion.  Verification
    substitutes them as trusted assertions: they are counted in
    ``Certificate.stats`` and rejected under ``strict=True``, never
    silently passed.
    """

    rule: str
    path: tuple
    lhs: Any = None
    rhs: Any = None
    bindings: dict = field(default_factory=dict)
    egraph_dependent: bool = False
    note: str = ""


class CertificateVerificationError(Exception):
    """The recorded derivation does not connect its claimed endpoints."""


@dataclass
class Certificate:
    """A proof-carrying derivation ``src`` -> ``dst``.

    ``steps``, applied in order, rewrite ``src`` into ``dst``: each is
    one rule application located by ``path``.  ``rules`` carries the
    :class:`Rewrite` objects used, so the certificate is self-contained
    for :func:`verify_certificate`.  ``stats`` summarises coverage —
    ``n_egraph_dependent`` counts steps the e-graph witnessed but that
    cannot replay standalone.
    """

    src: Any
    dst: Any
    root_eid: int | None
    steps: list = field(default_factory=list)
    rules: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        """Return the number of derivation steps."""
        return len(self.steps)

    @property
    def n_egraph_dependent(self) -> int:
        """Return the count of e-graph-dependent steps."""
        return sum(1 for s in self.steps if s.egraph_dependent)

    @property
    def replayable(self) -> bool:
        """True when every step is a standalone-replayable rewrite."""
        return self.n_egraph_dependent == 0

    @property
    def rules_used(self) -> list[str]:
        """Return the rules used by replayable steps."""
        return sorted(
            {s.rule for s in self.steps if not s.egraph_dependent}
        )

    @property
    def error_bound(self) -> float:
        """Return the conservative accumulated error bound.

        The triangle-inequality sum of every step's
        ``Rewrite.error_bound`` (steps lacking a bound contribute 0 —
        they are exact).  The bound is in whatever norm the contributing
        rules declared; today that is the spectral norm on the
        substituted subterm.  Propagating site-local bounds to the model
        output requires per-op Lipschitz constants — not yet computed.
        """
        total = 0.0
        for s in self.steps:
            r = self.rules.get(s.rule)
            if r is not None and r.error_bound:
                total += r.error_bound
        return total

    @property
    def exact(self) -> bool:
        """Return True when no step carries an error bound.

        The derivation is then an exact equivalence, not a certified
        approximation.
        """
        return self.error_bound == 0.0


# ---------------------------------------------------------------------------
#  The data codec — certificates as JSON records
# ---------------------------------------------------------------------------
#
#  A certificate is already *data-shaped*: an ordered list of single-rule
#  rewrites, each located by a child-index ``path`` and carrying the
#  concrete ``lhs``/``rhs`` instances plus the fired ``bindings``.  The
#  codec below makes that explicit — every term encodes through
#  :func:`catopt_core.ir.term_to_data` (the same scheme
#  ``laws.serialize`` uses for patterns) and every binding value is
#  either a term (metavariables) or a scalar/tuple attribute
#  (``"$attr:"`` keys).  Rules are referenced *by name*: the record is
#  honest about what data cannot carry — ``check``/``derive`` hooks are
#  code, so :func:`cert_from_data` takes the rule objects to resolve
#  against, exactly as a reconstructed law needs its hooks re-attached.
#
#  ``egraph_dependent`` steps serialize too: they are part of the
#  recorded derivation, and replaying them as trusted assertions is
#  what :func:`verify_certificate` already does under ``strict=False``.
#  A record keeps the flag — the honesty is in the data, not in hiding
#  the stub.

#: Bump when the record layout or reconstruction semantics change.
CERT_FORMAT = 1


def _binding_to_data(key: str, value: Any) -> dict[str, Any]:
    """Encode one fired binding entry as tagged JSON-safe data."""
    if key.startswith("$attr:"):
        return {"attr": attr_to_data(value)}
    return {"term": term_to_data(value)}


def _binding_from_data(key: str, data: dict[str, Any]) -> Any:
    """Decode one tagged binding entry produced by ``_binding_to_data``."""
    if "attr" in data:
        return attr_from_data(data["attr"])
    return term_from_data(data["term"])


def cert_to_data(cert: Certificate) -> dict[str, Any]:
    """Serialise *cert* to a JSON-safe record.

    The record carries the derivation itself — ``src``/``dst`` plus
    every step's rule name, ``path``, concrete ``lhs``/``rhs``
    instances and fired ``bindings`` — not the e-graph provenance it
    was reconstructed from.  ``rules_used`` is derived metadata for
    consumers that want the premise set without walking the steps.
    """
    return {
        "version": CERT_FORMAT,
        "src": term_to_data(cert.src),
        "dst": term_to_data(cert.dst),
        "steps": [
            {
                "rule": s.rule,
                "path": list(s.path),
                "lhs": term_to_data(s.lhs),
                "rhs": term_to_data(s.rhs),
                "bindings": {
                    k: _binding_to_data(k, v)
                    for k, v in s.bindings.items()
                },
                "egraph_dependent": s.egraph_dependent,
                "note": s.note,
            }
            for s in cert.steps
        ],
        "rules_used": cert.rules_used,
        "replayable": cert.replayable,
    }


def _step_from_data(sd: dict[str, Any]) -> CertStep:
    """Rebuild one :class:`CertStep` from its record entry."""
    return CertStep(
        sd["rule"],
        tuple(sd["path"]),
        term_from_data(sd["lhs"]),
        term_from_data(sd["rhs"]),
        {
            k: _binding_from_data(k, v)
            for k, v in sd.get("bindings", {}).items()
        },
        sd.get("egraph_dependent", False),
        sd.get("note", ""),
    )


def cert_from_data(data: dict[str, Any], rules: Any) -> Certificate:
    """Rebuild a :class:`Certificate` from :func:`cert_to_data` output.

    ``rules`` supplies the rule objects the record references by name —
    a name-to-``Rewrite`` mapping or any iterable of rules.  A step
    naming a rule absent from *rules* decodes but cannot verify
    (:func:`verify_certificate` raises "unknown rule"), the same honest
    posture ``law_from_data`` takes toward missing hooks.
    """
    if data.get("version") != CERT_FORMAT:
        raise ValueError(
            f"bad certificate record version: {data.get('version')!r}"
        )
    rmap = (
        rules if isinstance(rules, dict) else {r.name: r for r in rules}
    )
    steps = [_step_from_data(sd) for sd in data.get("steps", [])]
    used = sorted({s.rule for s in steps if not s.egraph_dependent})
    return Certificate(
        src=term_from_data(data["src"]),
        dst=term_from_data(data["dst"]),
        root_eid=None,
        steps=steps,
        rules={n: rmap[n] for n in used if n in rmap},
        stats={
            "from_data": True,
            "n_steps": len(steps),
            "n_egraph_dependent": sum(
                1 for s in steps if s.egraph_dependent
            ),
            "rules_used": used,
        },
    )
