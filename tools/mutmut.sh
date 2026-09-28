#!/bin/sh
# Run mutmut against a flat-layout sandbox.
#
# mutmut 3 derives a mutant's key from its path relative to cwd
# (`packages.catopt-core.src.catopt_core.attrs.x_foo`) and expects the
# module importable under that dotted name — but this monorepo's
# per-package `src/` layout imports as `catopt_core.attrs`.  This script
# builds a flat sandbox (`.mutmut-sandbox/`, gitignored) where each
# package is symlinked at the top level (so path == import name)
# alongside `tests/`, the `catopt` façade and `uv.lock`, writes a
# matching `[tool.mutmut]`, and runs mutmut there.  The sandbox persists
# so `run`/`results`/`browse` share one cache; delete it to reset.
#
# Usage:
#   tools/mutmut.sh run
#   tools/mutmut.sh results
#   tools/mutmut.sh browse
#
# Scope a fast loop with the env vars (globs/paths relative to the
# sandbox, where the package root is top-level):
#   MUTMUT_ONLY='catopt_core/egraph/terms.py' \
#   MUTMUT_TESTS='tests/test_certificates.py tests/test_cert_nonlocal.py' \
#     tools/mutmut.sh run
#
# Without MUTMUT_ONLY every source file is mutated; without MUTMUT_TESTS
# the whole suite runs per mutant.  Surviving mutants are weak-assertion
# findings (the suite is pinned at 100% coverage, so it is not a
# coverage gap).
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
sandbox="$root/.mutmut-sandbox"
mkdir -p "$sandbox"

for spec in \
    catopt-core:catopt_core \
    catopt-torch:catopt_torch \
    catopt-carriers:catopt_carriers \
    catopt-optimize:catopt_optimize
do
    dir=${spec%%:*}
    mod=${spec##*:}
    [ -e "$sandbox/$mod" ] || ln -s "$root/packages/$dir/src/$mod" "$sandbox/$mod"
done
[ -e "$sandbox/tests" ] || ln -s "$root/tests" "$sandbox/tests"
[ -e "$sandbox/catopt" ] || ln -s "$root/catopt" "$sandbox/catopt"
[ -e "$sandbox/uv.lock" ] || ln -s "$root/uv.lock" "$sandbox/uv.lock"

only=${MUTMUT_ONLY:-}
tests=${MUTMUT_TESTS:-tests/}
sel=$(printf '%s' "$tests" | sed 's/ /", "/g')

{
    echo '[tool.mutmut]'
    echo 'source_paths = ["catopt_core", "catopt_torch", "catopt_carriers", "catopt_optimize"]'
    echo 'pytest_add_cli_args = ["-p", "no:cacheprovider", "-q"]'
    echo "pytest_add_cli_args_test_selection = [\"$sel\"]"
    echo 'also_copy = ["catopt/"]'
    [ -n "$only" ] && echo "only_mutate = [\"$only\"]"
} > "$sandbox/pyproject.toml"

cd "$sandbox"
exec "$root/.venv/bin/mutmut" "$@"
