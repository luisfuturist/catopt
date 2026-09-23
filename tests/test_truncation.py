"""Truncation levels: the e-graph as a truncated program ∞-groupoid.

Level 1 keeps only the quotient (e-classes — it forgets paths between
terms).  Level 2 additionally stores one proof witness per merge
(ProofEdges + per-enode provenance) — enough for ``certificate`` to
reconstruct A derivation.  Level 3 adds nothing to storage; it
materialises *coherences* — alternate derivations between the same
endpoints — on demand via ``all_proofs`` / ``coherent_paths``.
"""

import time
import tracemalloc

import pytest

from catopt.egraph import (
    Certificate,
    CertificateVerificationError,
    EGraph,
    verify_certificate,
)
from catopt.ir import Op, Var, Const, Param, TensorType, op_repr
from catopt import rules as R
from catopt.cost import count_cost


def _t(d=4):
    return TensorType((d, d))


def _recurrence(T, d=4):
    """Hand-built unrolled scan h_t = A·h_{t-1} + x_t."""
    A = Param("A", TensorType((d, d)))
    h = Var("h0", TensorType((d,)))
    for t in range(T):
        h = Op.make("add", Op.make("matmul", A, h),
                    Var(f"x{t}", TensorType((d,))))
    return h


def _verify_path(eg, src, dst, path):
    """Replay one enumerated derivation as a standalone certificate."""
    rules = {}
    for s in path:
        if s.rule in eg._rule_objs:
            rules[s.rule] = eg._rule_objs[s.rule]
    cert = Certificate(src=src, dst=dst, root_eid=None,
                       steps=list(path), rules=rules)
    return verify_certificate(src, cert)


# ---------------------------------------------------------------------------
#  (a) level 1 — pure quotient: no proof witnesses are stored
# ---------------------------------------------------------------------------

def test_truncation_level_selection():
    """The dial maps 1/2/3; explicit track_proofs overrides it."""
    assert EGraph().truncation_level == 2
    assert EGraph(truncation_level=1).truncation_level == 1
    assert EGraph(truncation_level=2).truncation_level == 2
    assert EGraph(truncation_level=3).truncation_level == 3
    # backward compat: the old flag pins the level
    assert EGraph(track_proofs=True).truncation_level == 2
    assert EGraph(track_proofs=False).truncation_level == 1
    assert not EGraph(truncation_level=1)._track
    assert EGraph(truncation_level=3)._track
    with pytest.raises(ValueError):
        EGraph(truncation_level=0)
    with pytest.raises(ValueError):
        EGraph(truncation_level=4)


def test_level1_allocates_no_proof_data():
    """Level 1 stores zero ProofEdges and no per-enode provenance —
    same quotient, strictly less bookkeeping."""
    x = Var("x", _t())
    src = Op.make("add", x, Const(0))
    eg = EGraph(truncation_level=1)
    root = eg.add_term(src)
    stats = eg.run([R.ID_ADD, R.COMM_ADD], root, max_iterations=5)

    # saturation itself is unchanged — the quotient still works
    best = eg.extract_best(root, count_cost)
    assert op_repr(best) == "x"
    assert stats["n_proof_edges"] == 0

    # not a single ProofEdge / application / enode-provenance record
    assert eg.n_proof_edges == 0
    assert eg.merge_log == []
    assert eg.applications == []
    assert len(eg._merge_log) == 0
    assert len(eg._applications) == 0
    assert len(eg._enode_origin) == 0
    assert len(eg._enode_birth) == 0
    assert len(eg._enode_app) == 0


def test_level1_certificate_degrades_gracefully():
    """certificate() at level 1 returns a proof-free marker that still
    replays as a trusted assertion (and strict mode rejects it)."""
    x = Var("x", _t())
    src = Op.make("add", x, Const(0))
    eg = EGraph(truncation_level=1)
    root = eg.add_term(src)
    eg.run([R.ID_ADD, R.COMM_ADD], root, max_iterations=5)

    cert = eg.certificate(src)  # dst defaults to extract_best
    assert cert.stats.get("proof_free") is True
    assert not cert.replayable
    assert cert.n_egraph_dependent == 1
    assert cert.steps[0].rule == "<truncated>"

    out = verify_certificate(src, cert)
    assert op_repr(out) == op_repr(cert.dst) == "x"
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, cert, strict=True)

    # identity: no marker needed at all
    cert_id = eg.certificate(src, src)
    assert cert_id.n_steps == 0
    assert op_repr(verify_certificate(src, cert_id)) == op_repr(src)


def test_level1_coherence_unavailable():
    """Level 1 has no witnesses to enumerate — the coherence API says so."""
    x, y = Var("x", _t()), Var("y", _t())
    src = Op.make("add", x, y)
    eg = EGraph(truncation_level=1)
    root = eg.add_term(src)
    eg.run([R.COMM_ADD], root, max_iterations=3)
    with pytest.raises(RuntimeError):
        eg.all_proofs(src, Op.make("add", y, x))
    with pytest.raises(RuntimeError):
        eg.coherent_paths(src, Op.make("add", y, x))


def test_level1_uses_less_memory():
    """Identical saturation run at levels 1 vs 2: tracemalloc peak is
    strictly lower when no ProofEdge/provenance records are allocated."""
    src = _recurrence(6)

    def saturate(level):
        tracemalloc.start()
        try:
            eg = EGraph(truncation_level=level)
            root = eg.add_term(src)
            stats = eg.run(R.SCAN_LAWS, root, max_iterations=12,
                           max_nodes=300_000)
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return eg, stats, peak

    eg1, stats1, mem1 = saturate(1)
    eg2, stats2, mem2 = saturate(2)

    # same search behaviour — proofs observe, they don't steer
    assert stats1["iterations"] == stats2["iterations"]
    assert stats1["n_enodes"] == stats2["n_enodes"]
    assert stats1["n_classes"] == stats2["n_classes"]

    assert eg1.n_proof_edges == 0
    assert eg2.n_proof_edges > 0
    assert mem1 < mem2
    print(f"\n  level1 peak={mem1/1024:.0f}KiB "
          f"level2 peak={mem2/1024:.0f}KiB "
          f"(Δ={(mem2-mem1)/1024:.0f}KiB, "
          f"{eg2.n_proof_edges} ProofEdges)")


# ---------------------------------------------------------------------------
#  (b) level 2 — unchanged witness behaviour
# ---------------------------------------------------------------------------

def test_level2_certificates_still_verify():
    x, y = Var("x", _t()), Var("y", _t())
    src = Op.make("add", x, y)
    eg = EGraph(truncation_level=2)
    root = eg.add_term(src)
    eg.run([R.COMM_ADD], root, max_iterations=5)

    dst = Op.make("add", y, x)
    cert = eg.certificate(src, dst)
    assert op_repr(verify_certificate(src, cert)) == op_repr(dst)
    assert cert.replayable
    assert cert.rules_used == ["comm_add"]
    assert eg.n_proof_edges >= 1


def test_level2_matches_legacy_track_proofs():
    """EGraph(track_proofs=True) and truncation_level=2 behave alike."""
    x = Var("x", _t())
    src = Op.make("add", x, Const(0))
    for eg in (EGraph(track_proofs=True), EGraph(truncation_level=2)):
        root = eg.add_term(src)
        eg.run([R.ID_ADD], root, max_iterations=3)
        cert = eg.certificate(src)
        assert op_repr(verify_certificate(src, cert)) == "x"
        assert eg.n_proof_edges >= 1


# ---------------------------------------------------------------------------
#  (c) level 3 — coherent_paths surfaces alternate derivations
# ---------------------------------------------------------------------------

def _comm_graph(level=3):
    """add(x, add(y,z)) saturated under comm+assoc: many members, and —
    crucially — many *orders* of rule application between them."""
    x, y, z = Var("x", _t()), Var("y", _t()), Var("z", _t())
    src = Op.make("add", x, Op.make("add", y, z))
    eg = EGraph(truncation_level=level)
    root = eg.add_term(src)
    eg.run([R.COMM_ADD, R.ASSOC_ADD], root, max_iterations=8)
    return eg, src, root, (x, y, z)


def test_coherent_paths_finds_alternate_derivations():
    """add(x, add(y,z)) -> add(add(z,y), x) has at least two distinct
    derivations: commute the root first vs. commute the inner pair
    first.  Both must be found — and both must replay."""
    eg, src, root, (x, y, z) = _comm_graph()
    dst = Op.make("add", Op.make("add", z, y), x)

    result = eg.coherent_paths(src, dst)
    assert result["same_eclass"]
    assert result["n_paths"] >= 2, (
        f"expected >= 2 derivations, got {result['n_paths']}")

    first_moves = {p[0].path for p in result["paths"]}
    assert () in first_moves      # commute the root first
    assert (1,) in first_moves    # commute the inner add first

    # every enumerated derivation is a real proof: replays standalone
    for path in result["paths"]:
        out = _verify_path(eg, src, dst, path)
        assert op_repr(out) == op_repr(dst)


def test_all_proofs_distinct_signatures():
    """Derivations are deduplicated by (rule, path) signature."""
    eg, src, root, (x, y, z) = _comm_graph()
    dst = Op.make("add", Op.make("add", z, y), x)
    paths = eg.all_proofs(src, dst, max_paths=16)
    sigs = [tuple((s.rule, s.path) for s in p) for p in paths]
    assert len(sigs) == len(set(sigs))
    assert all(len(p) >= 1 for p in paths)


def test_all_proofs_bounded_enumeration():
    """max_paths caps the result; fuel bounds the search."""
    eg, src, root, (x, y, z) = _comm_graph()
    dst = Op.make("add", Op.make("add", z, y), x)
    capped = eg.coherent_paths(src, dst, max_paths=1)
    assert capped["n_paths"] == 1
    assert capped["truncated"]
    # tiny fuel still returns *something* well-formed (maybe empty)
    tiny = eg.all_proofs(src, dst, fuel=4)
    assert isinstance(tiny, list)


def test_all_proofs_identity_and_level2():
    """src == dst is the trivial (empty) coherence; the API also works
    at level 2 since the same witness data backs it."""
    eg, src, root, _ = _comm_graph(level=2)
    assert eg.all_proofs(src, src) == [[]]
    x, y = Var("x", _t()), Var("y", _t())
    s2 = Op.make("add", x, y)
    eg2 = EGraph(truncation_level=3)
    r2 = eg2.add_term(s2)
    eg2.run([R.COMM_ADD], r2, max_iterations=4)
    res = eg2.coherent_paths(s2, Op.make("add", y, x))
    assert res["n_paths"] == 1
    assert res["paths"][0][0].rule == "comm_add"


# ---------------------------------------------------------------------------
#  (d) overhead curve — reported, not asserted
# ---------------------------------------------------------------------------

def test_overhead_curve_report():
    """Time + memory (enodes / ProofEdges / applications) per level on a
    T=6 scan saturation.  Numbers are reported; only behavioural
    equivalence of the quotient is asserted."""
    src = _recurrence(6)
    rows = []
    for level in (1, 2, 3):
        tracemalloc.start()
        try:
            eg = EGraph(truncation_level=level)
            root = eg.add_term(src)
            t0 = time.perf_counter()
            stats = eg.run(R.SCAN_LAWS, root, max_iterations=12,
                           max_nodes=300_000)
            dt = time.perf_counter() - t0
            _cur, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        rows.append((level, dt, peak, stats, eg))

    print("\n  truncation overhead (T=6 scan, SCAN_LAWS):")
    base_t, base_m = rows[0][1], rows[0][2]
    for level, dt, peak, stats, eg in rows:
        print(f"    level {level}: {dt*1000:7.1f} ms  "
              f"peak {peak/1024:8.0f} KiB  "
              f"enodes={stats['n_enodes']:5d}  "
              f"classes={stats['n_classes']:4d}  "
              f"proof_edges={stats['n_proof_edges']:4d}  "
              f"apps={len(eg.applications):4d}  "
              f"(×{dt/base_t:.2f} time, ×{peak/base_m:.2f} mem)")

    # the dial changes bookkeeping, never the quotient
    assert rows[0][3]["n_enodes"] == rows[1][3]["n_enodes"] \
        == rows[2][3]["n_enodes"]
    assert rows[0][3]["n_classes"] == rows[1][3]["n_classes"] \
        == rows[2][3]["n_classes"]
    assert rows[0][3]["n_proof_edges"] == 0
    assert rows[1][3]["n_proof_edges"] == rows[2][3]["n_proof_edges"] > 0

    # level 3 stores nothing extra — coherence is computed on demand.
    # dst: the source with its innermost step lifted once — a neighbour
    # known to be reachable, with several alternate derivations
    # (direct lift, or lift-outer + lift-inner + unlift-outer detours).
    eg3 = rows[2][4]
    A = Param("A", TensorType((4, 4)))
    h0 = Var("h0", TensorType((4,)))
    dst = Op.make("apply", Op.make("aff", A, Var("x0", TensorType((4,)))),
                  h0)
    for t in range(1, 6):
        dst = Op.make("add", Op.make("matmul", A, dst),
                      Var(f"x{t}", TensorType((4,))))
    res = eg3.coherent_paths(src, dst, max_paths=8, max_steps=6,
                             fuel=20000)
    print(f"    level 3 coherent_paths(src, lifted): "
          f"n_paths={res['n_paths']} same_eclass={res['same_eclass']} "
          f"lens={[len(p) for p in res['paths']]}")
    assert res["n_paths"] >= 2          # alternate derivations surface
    assert res["same_eclass"]
