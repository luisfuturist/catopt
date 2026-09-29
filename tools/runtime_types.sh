#!/bin/sh
# Runtime-contract gate (typeguard).
#
# Instruments the torch-free core (``catopt_core``) plus the three
# adapter/orchestrator packages (``catopt_torch``, ``catopt_carriers``,
# ``catopt_orchestrator``) and runs the typeguard-clean test files, so
# annotation drift surfaces as a runtime ``TypeCheckError`` rather than
# silent ``Any``.  After the ty burn-down emptied ``[tool.ty.src]
# exclude`` these tests pass under full instrumentation; grow the list
# as more test files become typeguard-clean.  The torch-integration /
# omd / mha suites are excluded: they are slow under instrumentation
# (minutes each) and/or exercise paths blocked by annotations owned
# elsewhere.
#
# Run directly (``sh tools/runtime_types.sh``) or via the pre-commit
# ``runtime-types`` hook / the CI workflow.
set -eu
exec .venv/bin/python -m pytest -q -p no:cacheprovider \
    --typeguard-packages=catopt_core,catopt_torch,catopt_carriers,catopt_orchestrator \
    tests/test_ir.py \
    tests/test_egraph.py \
    tests/test_laws_structure.py \
    tests/test_cost.py \
    tests/test_interning.py \
    tests/test_attrs.py \
    tests/test_typing.py \
    tests/test_cov2_typing.py \
    tests/test_cov2_cost.py \
    tests/test_cov2_egraph.py \
    tests/test_cov2_meta.py \
    tests/test_meta.py \
    tests/test_rules.py \
    tests/test_property_ir.py \
    tests/test_property_egraph.py \
    tests/test_property_cost.py \
    tests/test_property_laws.py \
    tests/test_certificates.py \
    tests/test_cert_nonlocal.py \
    tests/test_memo_safety.py \
    tests/test_adapters.py \
    tests/test_ports.py \
    tests/test_contracts.py \
    tests/test_reports.py \
    tests/test_executor_base.py \
    tests/test_optimize_routing.py \
    tests/test_cov2_models.py \
    tests/test_om_monoid.py \
    tests/test_om_mask.py \
    tests/test_om_stream.py \
    tests/test_om_batched.py \
    tests/test_cov_om_lower.py \
    tests/test_cov_scan_lower.py \
    tests/test_cov2_om.py \
    tests/test_cov2_xcarrier.py \
    tests/test_scan_batched.py \
    tests/test_backend_cost.py \
    tests/test_registries.py \
    tests/test_partial_install.py \
    tests/test_logging.py \
    tests/test_cov2_bridge.py \
    tests/test_cov3_tail.py \
    tests/test_truncation.py \
    tests/test_unit_lift.py \
    tests/test_exact_corner.py \
    tests/test_falsified_pins.py \
    tests/test_batched_verify.py \
    tests/test_core_torch_free.py \
    tests/test_calibrate.py \
    tests/test_synthesis_guards.py \
    tests/test_synthesis_seeds.py \
    tests/test_rulecache.py \
    tests/test_cov_rulecache.py \
    tests/test_typecheck_smoke.py \
    tests/test_struct_real.py \
    tests/test_e2e_pipeline.py \
    tests/test_compositional.py \
    tests/test_hybrid.py \
    tests/test_om_causal.py \
    tests/test_cov2_regime.py \
    tests/test_regime.py \
    tests/test_cov2_laws.py \
    tests/test_mask_chunk.py \
    tests/test_trace.py \
    tests/test_trace_lift.py \
    tests/test_trace_pipeline.py \
    tests/test_xcarrier.py \
    tests/test_xcarrier_mha.py
