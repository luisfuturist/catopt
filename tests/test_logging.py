"""Phase 3c: owned modules report through ``logging``, not ``print``.

* a saturation run logs the fixed-point iteration line at INFO (and the
  ``max_nodes`` stop at WARNING);
* ``verbose=True`` widens the module logger to DEBUG-level detail;
* no ``print()`` calls remain in the owned sources — an AST-level check
  (docstrings/comments cannot false-positive), with a documented
  ``# stdout-compat`` escape hatch for any line a capsys/capfd test
  asserts on.  No such test exists today, so no compat prints are kept.
"""

import ast
import logging
from pathlib import Path

from catopt.calibrate import calibrate
from catopt.egraph import EGraph
from catopt.ir import Op, Param, TensorType, Var
from catopt.rules import ASSOC_LINEAR_BIAS

_REPO = Path(__file__).resolve().parents[1]

# Files owned by the phase-3c change (print -> logging).
_OWNED = sorted(
    str(p.relative_to(_REPO))
    for p in (_REPO / "packages").rglob("*.py")
    # the owned set is the phase-3c logging-migration files; optimize.py
    # keeps its deliberate verbose-mode stdout prints
    if "src" in p.parts and p.name != "optimize.py"
)


def _print_call_lines(path: Path) -> list[int]:
    """Line numbers of ``print(...)`` *calls* in *path*.

    AST-based: docstring examples (``regime.py``) and prose cannot
    register, only real call nodes.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]


def test_no_print_calls_in_owned_files():
    offenders = []
    for rel in _OWNED:
        path = _REPO / rel
        lines = path.read_text().splitlines()
        for lineno in _print_call_lines(path):
            # A kept print must justify itself on its own line.
            if "stdout-compat" not in lines[lineno - 1]:
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, f"print() calls outside compat: {offenders}"


def _small_graph() -> tuple[EGraph, int]:
    x = Var("x", TensorType((4, 8)))
    A = Param("A", TensorType((16, 8)))
    B = Param("B", TensorType((8, 16)))
    b1 = Param("b1", TensorType((16,)))
    b2 = Param("b2", TensorType((8,)))
    src = Op.make("linear", Op.make("linear", x, A, b1), B, b2)
    eg = EGraph()
    return eg, eg.add_term(src)


def test_saturation_logs_iteration(caplog):
    eg, eid = _small_graph()
    with caplog.at_level(logging.INFO):
        stats = eg.run(
            [ASSOC_LINEAR_BIAS], eid, max_iterations=8, max_nodes=2000
        )
    assert stats["iterations"] >= 1
    hits = [
        r
        for r in caplog.records
        if r.name.startswith("catopt")
        and r.levelno == logging.INFO
        and "saturation at iteration" in r.getMessage()
    ]
    assert hits, "no INFO saturation-iteration line logged"


def test_max_nodes_stop_logs_warning(caplog):
    eg, eid = _small_graph()
    with caplog.at_level(logging.WARNING):
        eg.run([ASSOC_LINEAR_BIAS], eid, max_nodes=1)
    assert any(
        r.name.startswith("catopt")
        and r.levelno == logging.WARNING
        and "max_nodes" in r.getMessage()
        for r in caplog.records
    )


def test_verbose_true_yields_debug(caplog):
    with caplog.at_level(logging.DEBUG):
        calibrate(device="cpu", quick=True, verbose=True)
    assert any(
        r.name == "catopt_optimize.calibrate" and r.levelno == logging.DEBUG
        for r in caplog.records
    )


def test_verbose_restores_logger_level(caplog):
    log = logging.getLogger("catopt.calibrate")
    prev = log.level
    with caplog.at_level(logging.DEBUG):
        calibrate(device="cpu", quick=True, verbose=True)
    assert log.level == prev
