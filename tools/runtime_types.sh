#!/bin/sh
# Runtime-contract gate (typeguard).
#
# Instruments the torch-free core (``catopt_core``) and runs the
# core-focused test files, so annotation drift surfaces as a runtime
# ``TypeCheckError`` rather than silent ``Any``.  After the ty burn-down
# emptied ``[tool.ty.src] exclude`` these 464 tests pass under full-core
# instrumentation; grow the list as more test files become
# typeguard-clean.  The torch-adapter suites are excluded: they exercise
# the eager/compile paths, not the core contracts, and are slow under
# instrumentation.
#
# Run directly (``sh tools/runtime_types.sh``) or via the pre-commit
# ``runtime-types`` hook / the CI workflow.
set -eu
exec .venv/bin/python -m pytest -q -p no:cacheprovider \
    --typeguard-packages=catopt_core \
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
    tests/test_memo_safety.py
