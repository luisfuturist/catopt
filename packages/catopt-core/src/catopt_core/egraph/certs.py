"""Proof-carrying certificate types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
        return len(self.steps)

    @property
    def n_egraph_dependent(self) -> int:
        return sum(1 for s in self.steps if s.egraph_dependent)

    @property
    def replayable(self) -> bool:
        """True when every step is a standalone-replayable rewrite."""
        return self.n_egraph_dependent == 0

    @property
    def rules_used(self) -> list[str]:
        return sorted(
            {s.rule for s in self.steps if not s.egraph_dependent}
        )

    @property
    def error_bound(self) -> float:
        """Conservative accumulated error bound: the triangle-inequality
        sum of every step's ``Rewrite.error_bound`` (steps lacking a
        bound contribute 0 — they are exact).  The bound is in whatever
        norm the contributing rules declared; today that is the
        spectral norm on the substituted subterm.  Propagating
        site-local bounds to the model output requires per-op Lipschitz
        constants — not yet computed."""
        total = 0.0
        for s in self.steps:
            r = self.rules.get(s.rule)
            if r is not None and r.error_bound:
                total += r.error_bound
        return total

    @property
    def exact(self) -> bool:
        """True when no step carries an error bound — the derivation is
        an exact equivalence, not a certified approximation."""
        return self.error_bound == 0.0



