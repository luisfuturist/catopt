"""E-graph package — same public surface as the former monolith.

Split (plan 0001 phase 2b):
  types.py   — ENode, EClass, UnionFind, Rewrite, leaf registry
  certs.py   — ProofEdge, CertStep, Certificate, verification error
  core.py    — EGraph: union-find, matching, saturation, rebuild
  extract.py — _ExtractMixin: extract_best / extract_paired / locate
  proof.py   — _ProofMixin: certificates, coherent paths, explanations
  terms.py   — structural term utils + verify_certificate
"""

from catopt.egraph.certs import (
    Certificate,
    CertificateVerificationError,
    CertStep,
    ProofEdge,
)
from catopt.egraph.core import EGraph
from catopt.egraph.terms import (
    _term_instantiate,  # noqa: F401 — compat re-export
    _term_match,  # noqa: F401 — compat re-export
    verify_certificate,
)
from catopt.egraph.types import (
    EClass,
    ENode,
    Rewrite,
    UnionFind,
    _LeafRegistry,  # noqa: F401 — compat re-export
    _pattern_attrs,  # noqa: F401 — compat re-export
)

__all__ = [
    "CertStep",
    "Certificate",
    "CertificateVerificationError",
    "EClass",
    "EGraph",
    "ENode",
    "ProofEdge",
    "Rewrite",
    "UnionFind",
    "verify_certificate",
]
