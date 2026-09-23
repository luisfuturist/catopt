"""Tests for proof-carrying certificates (2-morphisms as data).

Every ``union`` records *why* two e-classes merged — a ProofEdge naming
the rule, the pre-merge canonical class ids, and the fired binding.
``EGraph.certificate`` reconstructs the minimal positional derivation
connecting the source term to an extracted member, and
``verify_certificate`` replays it on real terms (re-match LHS,
instantiate RHS, substitute at path) — derivational equivalence,
independent of the e-graph.
"""

import time

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
from catopt.rules import pair_shared_input_linears


def _t(d=4):
    return TensorType((d, d))


def _opdepth(t, memo):
    """DAG-aware critical-path depth — the scan extraction objective."""
    if not isinstance(t, Op):
        return 0
    k = id(t)
    if k not in memo:
        memo[k] = 1 + max((_opdepth(a, memo) for a in t.args), default=0)
    return memo[k]


def _recurrence(T, d=4):
    """Hand-built unrolled scan h_t = A·h_{t-1} + x_t (the term
    LinearRecurrence exports), avoiding a torch dependency here."""
    A = Param("A", TensorType((d, d)))
    h = Var("h0", TensorType((d,)))
    for t in range(T):
        h = Op.make("add", Op.make("matmul", A, h),
                    Var(f"x{t}", TensorType((d,))))
    return h


# ---------------------------------------------------------------------------
#  (a) saturate -> extract -> certificate replays to the same term
# ---------------------------------------------------------------------------

def test_certificate_comm_add():
    """add(x, y) under comm_add: the commuted member certifies in one step."""
    x, y = Var("x", _t()), Var("y", _t())
    src = Op.make("add", x, y)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.COMM_ADD], root, max_iterations=5)

    dst = Op.make("add", y, x)
    cert = eg.certificate(src, dst)
    out = verify_certificate(src, cert)
    assert op_repr(out) == op_repr(dst)
    assert cert.replayable
    assert cert.rules_used == ["comm_add"]


def test_certificate_id_add_metavar_rhs():
    """add(x, 0) -> x: an RHS that is a bare metavariable creates no new
    enode — the merge edge itself is the proof, replayed via the
    edge-path fallback."""
    x = Var("x", _t())
    src = Op.make("add", x, Const(0))
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.ID_ADD, R.COMM_ADD], root, max_iterations=5)

    best = eg.extract_best(root, count_cost)
    assert op_repr(best) == "x"
    cert = eg.certificate(src, best)
    out = verify_certificate(src, cert)
    assert op_repr(out) == "x"
    assert cert.replayable
    assert "id_add" in cert.rules_used


def test_certificate_weight_factor():
    """x@W1 + x@W2 -> x@(W1+W2): extraction prefers the merged weight
    (the param-only add is compile-time-free); the certificate carries
    exactly the weight_factor_matmul step."""
    x = Var("x", _t())
    W1, W2 = Param("W1", _t()), Param("W2", _t())
    src = Op.make("add", Op.make("matmul", x, W1), Op.make("matmul", x, W2))
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.WEIGHT_FACTOR], root, max_iterations=5)

    best = eg.extract_best(root, count_cost)
    cert = eg.certificate(src, best)
    out = verify_certificate(src, cert)
    assert op_repr(out) == op_repr(best)
    assert cert.replayable
    assert cert.rules_used == ["weight_factor_matmul"]


def test_certificate_scan_laws_t4():
    """SCAN_LAWS on a T=4 recurrence: the balanced aff_compose bracket
    certifies as lift + compose + one associativity step."""
    src = _recurrence(4)
    eg = EGraph()
    root = eg.add_term(src)
    stats = eg.run(R.SCAN_LAWS, root, max_iterations=14,
                   max_nodes=300_000)
    best = eg.extract_best(root, lambda t, **k: _opdepth(t, {}))
    assert "aff_compose" in op_repr(best)  # sanity: scan form reached

    cert = eg.certificate(src, best)
    out = verify_certificate(src, cert)
    assert op_repr(out) == op_repr(best)
    assert cert.replayable
    # a T=4 lift needs >= 3 rule applications (one aff_lift + steps);
    # extraction may pick either bracketing, so associativity is optional
    assert cert.n_steps >= 3
    assert "aff_lift" in cert.rules_used
    assert set(cert.rules_used) <= {
        "aff_lift", "aff_lift_step", "aff_unlift",
        "aff_compose_unfold", "aff_assoc", "aff_assoc_rev"}


def test_certificate_identity_derivation():
    """dst == src: the empty derivation verifies trivially."""
    x, y = Var("x", _t()), Var("y", _t())
    src = Op.make("add", x, y)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.COMM_ADD], root, max_iterations=3)
    cert = eg.certificate(src, src)
    assert cert.n_steps == 0
    assert op_repr(verify_certificate(src, cert)) == op_repr(src)


# ---------------------------------------------------------------------------
#  (b) provenance: the certificate names the rules that actually fired
# ---------------------------------------------------------------------------

def test_certificate_provenance_names_fired_rules():
    x, y = Var("x", _t()), Var("y", _t())
    src = Op.make("sub", Op.make("add", x, Const(0)), y)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.SUB_TO_ADD, R.ID_ADD, R.COMM_ADD], root, max_iterations=5)

    dst = Op.make("add", x, Op.make("neg", y))  # via sub_to_add + id_add
    cert = eg.certificate(src, dst)
    out = verify_certificate(src, cert)
    assert op_repr(out) == op_repr(dst)
    # named rules are a subset of what the e-graph recorded as fired
    assert set(cert.rules_used) <= set(eg.rule_fires)
    assert "sub_to_add" in cert.rules_used
    assert "id_add" in cert.rules_used
    # and every named rule's Rewrite object travels inside the cert
    assert set(cert.rules) == set(cert.rules_used)


def test_proof_edges_record_merges():
    """union() records one ProofEdge per real merge; apply_rule edges
    name the rule and carry the fired binding."""
    x = Var("x", _t())
    eg = EGraph()
    a = eg.add_term(Op.make("neg", x))
    b = eg.add_term(Op.make("square", x))
    eg.union(a, b)  # manual union: no rule
    assert eg.n_proof_edges == 1
    edge = eg.merge_log[0]
    assert edge.rule is None

    eg2 = EGraph()
    src = Op.make("add", x, Const(0))
    root = eg2.add_term(src)
    eg2.run([R.ID_ADD], root, max_iterations=3)
    assert eg2.n_proof_edges >= 1
    assert any(e.rule == "id_add" for e in eg2.merge_log)


# ---------------------------------------------------------------------------
#  (c) tampered certificates fail verification
# ---------------------------------------------------------------------------

@pytest.fixture
def comm_cert():
    x, y = Var("x", _t()), Var("y", _t())
    src = Op.make("add", x, y)
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.COMM_ADD], root, max_iterations=5)
    cert = eg.certificate(src, Op.make("add", y, x))
    return src, cert, x, y


def test_tampered_rhs_fails(comm_cert):
    src, cert, x, y = comm_cert
    cert.steps[0].rhs = Op.make("mul", y, x)  # claim a different result
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, cert)


def test_tampered_rule_name_fails(comm_cert):
    src, cert, x, y = comm_cert
    cert.steps[0].rule = "nonexistent_rule"
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, cert)


def test_tampered_path_fails(comm_cert):
    src, cert, x, y = comm_cert
    cert.steps[0].path = (0,)  # point at a subterm, not the root
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, cert)


def test_wrong_source_fails(comm_cert):
    src, cert, x, y = comm_cert
    with pytest.raises(CertificateVerificationError):
        verify_certificate(Op.make("mul", x, y), cert)


def test_tampered_dst_claim_fails(comm_cert):
    src, cert, x, y = comm_cert
    cert.dst = Op.make("mul", x, y)  # claim a different endpoint
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, cert)


# ---------------------------------------------------------------------------
#  e-graph-dependent steps: recorded, flagged, never silently passed
# ---------------------------------------------------------------------------

def test_pairing_pass_merges_are_egraph_dependent():
    """The non-local pairing pass merges classes without a term-local
    rule — certificate marks those steps e-graph-dependent rather than
    fabricating a derivation."""
    x = Var("x", TensorType((2, 4)))
    W1 = Param("W1", TensorType((8, 4)))
    W2 = Param("W2", TensorType((8, 4)))
    src = Op.make("add", Op.make("linear", x, W1), Op.make("linear", x, W2))
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([], root, max_iterations=1)
    groups = pair_shared_input_linears(eg)
    assert groups, "expected a pairing group"
    eg.rebuild()

    overrides = {eg.find(cid): en for g in groups for cid, en in g.items()}
    dst = eg.extract_best(root, count_cost, overrides=overrides)
    assert "split" in op_repr(dst)

    cert = eg.certificate(src, dst)
    assert cert.n_egraph_dependent >= 2  # one per paired member
    assert not cert.replayable
    notes = " ".join(s.note for s in cert.steps if s.egraph_dependent)
    assert "non-local" in notes or "outside rule" in notes

    # non-strict replay substitutes the trusted assertions -> dst;
    # strict mode refuses to call that a proof.
    out = verify_certificate(src, cert)
    assert op_repr(out) == op_repr(dst)
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, cert, strict=True)


# ---------------------------------------------------------------------------
#  (d) overhead: proof tracking does not perturb saturation
# ---------------------------------------------------------------------------

def test_proof_tracking_overhead_small():
    """Same saturation result with and without tracking; one O(1)
    record per merge/enode — no second pass over the proof space."""
    src = _recurrence(8)

    def saturate(track):
        eg = EGraph(track_proofs=track)
        root = eg.add_term(src)
        t0 = time.perf_counter()
        stats = eg.run(R.SCAN_LAWS, root, max_iterations=14,
                       max_nodes=300_000)
        return eg, stats, time.perf_counter() - t0

    eg_on, stats_on, t_on = saturate(True)
    eg_off, stats_off, t_off = saturate(False)

    # identical search behaviour — proofs observe, they don't steer
    assert stats_on["iterations"] == stats_off["iterations"]
    assert stats_on["n_enodes"] == stats_off["n_enodes"]
    assert stats_on["n_classes"] == stats_off["n_classes"]

    # one witness per merge: |merge_log| <= |applications|, and both
    # are bounded by total enode work, not proof-space size
    assert eg_on.n_proof_edges > 0
    assert eg_on.n_proof_edges <= len(eg_on.applications)

    # generous bound — tracking is dict stores + a tuple append
    assert t_on < t_off * 10 + 1.0
    assert t_on < 30.0  # absolute sanity bound for a T=8 scan

    # and the tracked run still produces a verifiable certificate
    root = eg_on._class_of_term(src)
    best = eg_on.extract_best(root, lambda t, **k: _opdepth(t, {}))
    cert = eg_on.certificate(src, best)
    assert op_repr(verify_certificate(src, cert)) == op_repr(best)


def test_certificate_default_extracts_best():
    """certificate(src) with no dst uses extract_best + count_cost."""
    x = Var("x", _t())
    src = Op.make("add", x, Const(0))
    eg = EGraph()
    root = eg.add_term(src)
    eg.run([R.ID_ADD, R.COMM_ADD], root, max_iterations=5)
    cert = eg.certificate(src)
    assert op_repr(cert.dst) == "x"
    assert op_repr(verify_certificate(src, cert)) == "x"
