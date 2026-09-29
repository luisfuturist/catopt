"""catopt-carriers — the semantic carriers and their executors.

Online-softmax monoids (:mod:`~catopt_carriers.om`,
:mod:`~catopt_carriers.om_lower`), the deferred omd carrier
(:mod:`~catopt_carriers.xcarrier`,
:mod:`~catopt_carriers.omd_lower`), affine scans
(:mod:`~catopt_carriers.scan_lower`), and the traced-monoidal
extensions (:mod:`~catopt_carriers.trace`,
:mod:`~catopt_carriers.trace_lift`).  Each module declares
``TORCH_BINDINGS`` (folded into :meth:`OpTable.full`) and registers
its shape rules on import — OpTable composes them lazily so the
modules themselves stay import-safe.

The carrier law families also own their rule sets:
``catopt_carriers.om.OM_RULES`` /
``catopt_carriers.trace.TRACE_RULES`` /
``catopt_carriers.xcarrier.XC_RULES`` /
``catopt_carriers.decode_laws.DECODE_RULES`` are
:class:`~catopt_core.laws.RuleSet` values, and ``CARRIERS`` (below)
is their composition — the preset the orchestrator's
``DEFAULT_RULES`` adds on top of the core default.  Every name
materialises **lazily**: the law modules import their backend
(torch), so touching them at package import would couple every
``import catopt_carriers`` to torch.
"""

from typing import Any

__all__ = [
    "CARRIERS",
    "DECODE_RULES",
    "OM_RULES",
    "TRACE_RULES",
    "XC_RULES",
]


def __getattr__(name: str) -> Any:
    """Compose the carrier rule sets on first access (torch-coupled).

    Resolved per name so ``decode_laws`` stays reachable in a
    torch-free process — only ``CARRIERS`` needs every family.
    """
    if name == "CARRIERS":
        from catopt_carriers.decode_laws import DECODE_RULES
        from catopt_carriers.om import OM_RULES
        from catopt_carriers.trace import TRACE_RULES
        from catopt_carriers.xcarrier import XC_RULES

        rs = OM_RULES + TRACE_RULES + XC_RULES + DECODE_RULES
    elif name == "OM_RULES":
        from catopt_carriers.om import OM_RULES as rs
    elif name == "TRACE_RULES":
        from catopt_carriers.trace import TRACE_RULES as rs
    elif name == "XC_RULES":
        from catopt_carriers.xcarrier import XC_RULES as rs
    elif name == "DECODE_RULES":
        from catopt_carriers.decode_laws import DECODE_RULES as rs
    else:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    globals()[name] = rs
    return rs
