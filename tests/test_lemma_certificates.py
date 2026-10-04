"""Lemma certificates — ``Rewrite.derivation`` as replayable proof data.

The axiom/lemma split records each non-kernel rule's ``derivation`` —
the shipped premises proving the law's instance.  These tests pin the
upgrade that turns the annotation into a *certificate*: saturate the
lemma's instance under the recorded premises alone, reconstruct the
positional derivation, and replay it strictly on real terms.  The
measured shape (``tools/law_lemma_cert.py``): every derivation-carrying
rule gets a linear certificate — one step, the recorded premise — and
the replayable direction is the one the *axiom* fires forward.

Also covers the certificate data codec (``cert_to_data`` /
``cert_from_data``): steps are JSON-safe records — terms through
``term_to_data``, bindings tagged term/attr — and a rebuilt
certificate replays under ``verify_certificate`` against a name→rule
map, closing the lemma-store loop (pattern + cond + derivation +
replayable proof).
"""

import json
import sys
from pathlib import Path

import catopt_core.laws.layout
import pytest
from catopt_core import laws as R
from catopt_core.egraph import (
    Certificate,
    CertificateVerificationError,
    CertStep,
    EGraph,
    cert_from_data,
    cert_to_data,
    verify_certificate,
)
from catopt_core.ir import Op, TensorType, Var, op_repr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import law_evidence as le  # noqa: E402
import law_lemma_cert as llc  # noqa: E402


def _t(d=4):
    return TensorType((d, d))


def _merged_pair(lhs, rhs, rules):
    """Intern both sides, saturate under *rules*, return (eg, eids)."""
    eg = EGraph()
    r_lhs = eg.add_term(lhs)
    r_rhs = eg.add_term(rhs)
    eg.run(rules, r_lhs, max_iterations=20, max_nodes=50_000)
    assert eg.find(r_lhs) == eg.find(r_rhs), "sides did not merge"
    return eg, r_lhs, r_rhs


# ---------------------------------------------------------------------------
#  Lemmas certify as one-step derivations of their recorded premise
# ---------------------------------------------------------------------------


def test_silu_fold_certifies_through_axiom():
    """silu_fold <- silu_expand: the axiom's forward step IS the proof.

    The lemma claims ``mul(x, sig(x)) -> silu(x)``; its recorded premise
    fires only in the *expand* direction, so the replayable certificate
    runs ``rhs -> lhs`` — equality is symmetric, and the lemma direction
    is unreachable by forward premise rewriting (the e-graph-dependent
    stub records exactly that limit).
    """
    x = Var("x", _t())
    lhs = Op.make("mul", x, Op.make("sigmoid", x))
    rhs = Op.make("silu", x)
    eg, r_lhs, r_rhs = _merged_pair(lhs, rhs, [R.SILU_EXPAND])

    cert = eg.certificate(rhs, lhs, root_eid=r_rhs)
    verify_certificate(rhs, cert, strict=True)
    assert cert.replayable
    assert cert.rules_used == ["silu_expand"]
    assert cert.n_steps == 1

    fwd = eg.certificate(lhs, rhs, root_eid=r_lhs)
    assert not fwd.replayable
    assert fwd.n_egraph_dependent == 1


def test_silu_mul_form_certifies_inside_context():
    """silu_mul_form <- silu_expand: the premise fires inside ``mul``.

    The one composite derivation in the lemma set — the axiom fires at
    path ``(0,)``, not at the root.
    """
    g, u = Var("g", _t()), Var("u", _t())
    lhs = Op.make("mul", Op.make("silu", g), u)
    rhs = Op.make("mul", Op.make("mul", g, Op.make("sigmoid", g)), u)
    eg, r_lhs, _r_rhs = _merged_pair(lhs, rhs, [R.SILU_EXPAND])
    cert = eg.certificate(lhs, rhs, root_eid=r_lhs)
    verify_certificate(lhs, cert, strict=True)
    assert cert.replayable
    assert [(s.rule, s.path) for s in cert.steps] == [
        ("silu_expand", (0,))
    ]


# ---------------------------------------------------------------------------
#  The serialization seam — certificates round-trip as JSON data
# ---------------------------------------------------------------------------


def test_lemma_certificate_json_roundtrip():
    """cert -> data -> json -> cert -> strict replay reaches dst."""
    g, u = Var("g", _t()), Var("u", _t())
    lhs = Op.make("mul", Op.make("silu", g), u)
    rhs = Op.make("mul", Op.make("mul", g, Op.make("sigmoid", g)), u)
    eg, r_lhs, _ = _merged_pair(lhs, rhs, [R.SILU_EXPAND])
    cert = eg.certificate(lhs, rhs, root_eid=r_lhs)

    record = json.loads(json.dumps(cert_to_data(cert)))
    rebuilt = cert_from_data(record, {r.name: r for r in R.ALL_RULES})
    out = verify_certificate(rebuilt.src, rebuilt, strict=True)
    assert op_repr(out) == op_repr(rhs)
    assert rebuilt.replayable
    assert rebuilt.rules_used == ["silu_expand"]


def test_cert_codec_attr_bindings_roundtrip():
    """``$attr:`` bindings serialize tagged and replay after decode."""
    u, v = Var("u", _t()), Var("v", _t())
    lhs = Op.make(
        "mul",
        Op.make("select", u, dim=0, index=1),
        Op.make("select", v, dim=0, index=1),
    )
    rhs = Op.make("select", Op.make("mul", u, v), dim=0, index=1)
    eg, r_lhs, _ = _merged_pair(lhs, rhs, [R.SELECT_MUL])
    cert = eg.certificate(lhs, rhs, root_eid=r_lhs)
    assert cert.replayable
    assert {
        k for k in cert.steps[0].bindings if k.startswith("$attr:")
    } == {"$attr:D", "$attr:I"}

    rebuilt = cert_from_data(
        json.loads(json.dumps(cert_to_data(cert))), [R.SELECT_MUL]
    )
    step = rebuilt.steps[0]
    assert step.bindings["$attr:D"] == 0
    assert step.bindings["$attr:I"] == 1
    out = verify_certificate(rebuilt.src, rebuilt, strict=True)
    assert op_repr(out) == op_repr(rhs)


def test_cert_codec_dependent_step_and_version():
    """Dependent stubs round-trip as flagged assertions, never silently."""
    x = Var("x", _t())
    src = Op.make("mul", x, Op.make("sigmoid", x))
    dst = Op.make("silu", x)
    stub = CertStep("<egraph>", (), src, dst, {}, egraph_dependent=True)
    cert = Certificate(
        src=src, dst=dst, root_eid=None, steps=[stub], rules={}
    )
    record = json.loads(json.dumps(cert_to_data(cert)))
    rebuilt = cert_from_data(record, [])
    assert not rebuilt.replayable
    # non-strict replay substitutes the stub as a trusted assertion
    out = verify_certificate(src, rebuilt)
    assert op_repr(out) == op_repr(dst)
    # strict replay refuses it outright
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, rebuilt, strict=True)
    # a version the codec does not recognise is rejected, not guessed
    with pytest.raises(ValueError, match="version"):
        cert_from_data({"version": 999}, [])
    # a step whose rule is absent from the map decodes but cannot verify
    lone = Certificate(
        src=src,
        dst=src,
        root_eid=None,
        steps=[CertStep("ghost_rule", (), src, src, {})],
        rules={},
    )
    ghost = cert_from_data(cert_to_data(lone), {"other": R.SILU_EXPAND})
    with pytest.raises(
        CertificateVerificationError, match="ghost_rule"
    ):
        verify_certificate(src, ghost)


def test_cert_codec_empty_derivation():
    """A zero-step certificate (src == dst) round-trips trivially."""
    x = Var("x", _t())
    cert = Certificate(src=x, dst=x, root_eid=None, steps=[], rules={})
    rebuilt = cert_from_data(cert_to_data(cert), [])
    assert rebuilt.steps == []
    assert verify_certificate(x, rebuilt) == x


# ---------------------------------------------------------------------------
#  The honest boundary — merged but reconstruction-hard
# ---------------------------------------------------------------------------


def test_merge_without_connect_cert_enumerates_linear_proof():
    """``certificate()`` can miss a derivation ``all_proofs`` finds.

    ``add(linear(a,W), linear(b,W)) == linear(add(a,b), W)`` merges
    under the NT-bridge + ``right_factor_matmul`` — but neither
    endpoint's root is a rule-application RHS, and ``_edge_path``
    searches root-level rewrites only, so the connect machinery emits
    an e-graph-dependent stub in both directions.  Positional
    enumeration still surfaces the genuine 4-step derivation (the
    measured ``right_factor_linear`` boundary, shrunk to 3 rules).
    """
    a, b, w = Var("a", _t()), Var("b", _t()), Var("W", _t())
    lhs = Op.make(
        "add", Op.make("linear", a, w), Op.make("linear", b, w)
    )
    rhs = Op.make("linear", Op.make("add", a, b), w)
    rules = [
        catopt_core.laws.layout.LINEAR_TO_MM_T,
        catopt_core.laws.layout.LINEAR_FROM_MM_T,
        R.RIGHT_FACTOR,
    ]
    eg, r_lhs, r_rhs = _merged_pair(lhs, rhs, rules)

    for src, eid in ((lhs, r_lhs), (rhs, r_rhs)):
        cert = eg.certificate(
            src, rhs if src is lhs else lhs, root_eid=eid
        )
        assert not cert.replayable

    paths = eg.all_proofs(lhs, rhs, max_steps=10, fuel=50_000)
    assert paths
    shortest = min(paths, key=len)
    cert = Certificate(
        src=lhs,
        dst=rhs,
        root_eid=r_lhs,
        steps=list(shortest),
        rules={r.name: r for r in rules},
    )
    out = verify_certificate(lhs, cert, strict=True)
    assert op_repr(out) == op_repr(rhs)
    assert [s.rule for s in shortest] == [
        "linear_to_matmul_t",
        "linear_to_matmul_t",
        "right_factor_matmul",
        "linear_from_matmul_t",
    ]


# ---------------------------------------------------------------------------
#  The admission seam — a stored lemma carries its certificate
# ---------------------------------------------------------------------------


def test_stored_lemma_cert_roundtrips_through_admit(tmp_path):
    """The full loop: lemma -> record -> admit -> cert replays strict.

    ``store_lemma`` materializes ``silu_mul_form``'s recorded
    derivation (``silu_expand`` at path ``(0,)`` — the one composite
    step in the lemma set); ``admit_lemma`` rebuilds the Rewrite;
    ``stored_certificate`` decodes the record's ``cert`` field and
    verifies it strictly.  The replayed term is the certificate's
    ``dst`` — the proof travels with the law.
    """
    by_name = {r.name: r for r in R.ALL_RULES}
    rule = by_name["silu_mul_form"]
    conn = le.connect(str(tmp_path / "laws.db"))
    try:
        key = le.store_lemma(conn, rule)
        got = le.admit_lemma(conn, key)
        assert got is not None
        rebuilt, record = got
        assert rebuilt.name == "silu_mul_form"
        assert record["cert"] is not None
        cert = le.stored_certificate(record)
        assert cert.replayable
        assert cert.rules_used == list(rule.derivation)
        assert [(s.rule, s.path) for s in cert.steps] == [
            ("silu_expand", (0,))
        ]
        out = verify_certificate(cert.src, cert, strict=True)
        assert op_repr(out) == op_repr(cert.dst)
    finally:
        conn.close()


def test_right_factor_linear_stores_no_cert(tmp_path):
    """The honest boundary: ``cert: null`` where nothing replays.

    ``right_factor_linear`` carries no ``derivation`` annotation and
    under ``ALL_RULES`` its instance sides never even merge (the
    NT-bridge derivation exists only under ``--with-layout`` — and
    the annotation never claimed it).  ``materialize`` yields no
    certificate, and the stored record says so: ``cert`` is ``null``,
    not a stub.
    """
    by_name = {r.name: r for r in R.ALL_RULES}
    rfl = by_name["right_factor_linear"]
    assert rfl.derivation == ()
    row, cert = llc.materialize(rfl, list(R.ALL_RULES))
    assert cert is None
    assert row.verdict == "gap"
    conn = le.connect(str(tmp_path / "laws.db"))
    try:
        le.store_lemma(conn, rfl)
        stored = json.loads(le.lemma_rows(conn)[0]["law_json"])
        assert stored["cert"] is None
        assert le.stored_certificate(stored) is None
    finally:
        conn.close()
