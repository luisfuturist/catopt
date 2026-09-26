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
"""
