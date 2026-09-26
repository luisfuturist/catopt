"""Certificates for non-local offers — replayable union witnesses.

Non-local passes offer e-class members that no LHS->RHS rewrite could
have produced: the offered term is computed from the whole e-graph
(e.g. ``lift_scan_to_trace`` constructs the time-extended nilpotent
block-shift matrix F from the whole recurrence horizon).  Such merges
used to be irreducibly ``egraph_dependent`` in certificates — real
equalities with no replayable derivation.

``EGraph.union(a, b, witness=...)`` closes that gap: the pass
synthesises a concrete ``Rewrite`` justifying THIS merge (a pointwise
rule ``source_member -> offered_term``), the union registers it like a
fired rule, and ``certificate``/``verify_certificate`` treat the merge
as an ordinary rule step — re-matched and re-instantiated on real
terms, ``strict=True`` included.

What these tests prove:

* a lifted ``trace`` member gets a certificate with ZERO
  e-graph-dependent steps, and strict verification passes;
* the witness is an ordinary Rewrite object travelling inside the
  certificate — it replays standalone (re-match + re-instantiate) and
  the equality it asserts holds numerically on fresh fp64 tensors;
* unions offered WITHOUT a witness keep the pre-fix behaviour:
  honestly flagged ``egraph_dependent``, rejected under strict;
* the mechanism is generic — ``union(witness=Rewrite(...))`` works for
  a bare manual merge too (what ``pair_shared_input_linears`` would
  need to adopt: a synthesised Rewrite per offered member).
"""


import pytest
import torch

import catopt.trace as _cat_trace  # noqa: F401  torch bindings

# (trace/parl/eye/concat)
from catopt.egraph import (
    CertificateVerificationError,
    EGraph,
    Rewrite,
    _term_instantiate,
    _term_match,
    verify_certificate,
)
from catopt.ir import IR, Const, Op, Param, TensorType, Var, op_repr
from catopt.torch_bridge import ir_to_torch_module
from catopt.trace_lift import lift_scan_to_trace


@pytest.fixture(autouse=True)
def _fp64():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(prev)


# ---------------------------------------------------------------------------
#  Term builders (same as tests/test_trace_lift.py)
# ---------------------------------------------------------------------------


def _diag_term(T: int, d: int):
    """h_t = a_t * h_{t-1} + b_t * x_t — the Mamba-faithful spine."""
    a = Param("pa", TensorType((T, d)))
    b = Param("pb", TensorType((T, d)))
    x = Var("x", TensorType((T, d)))
    h0 = Param("h0", TensorType((d,)))
    h = h0
    for t in range(T):
        a_t = Op.make("select", a, arg1=0, arg2=t)
        in_t = Op.make(
            "mul",
            Op.make("select", b, arg1=0, arg2=t),
            Op.make("select", x, arg1=0, arg2=t),
        )
        h = Op.make("add", Op.make("mul", a_t, h), in_t)
    return h, [x], {"pa": a, "pb": b, "h0": h0}


def _env(T: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    env = {
        "pa": torch.rand(T, d, generator=g) * 0.9,
        "pb": torch.randn(T, d, generator=g),
        "h0": torch.randn(d, generator=g),
    }
    x = torch.randn(T, d, generator=g)
    return env, x


def _eval(term, inputs, env, x):
    return ir_to_torch_module(
        IR(root=term, inputs=inputs), param_values=env
    )(x)


# ---------------------------------------------------------------------------
#  (a) witnessed offers give fully replayable certificates
# ---------------------------------------------------------------------------


def test_lifted_trace_certificate_fully_replayable():
    """union(..., witness=pointwise rule) -> cert with no
    egraph_dependent steps; strict=True verification passes."""
    T, d = 4, 4
    term, _, _ = _diag_term(T, d)
    eg = EGraph()
    root = eg.add_term(term)
    lifts = lift_scan_to_trace(
        eg, root_eid=root, channel_splits=None, witness=True
    )
    assert len(lifts) == 1
    cert = eg.certificate(term, lifts[0].term, root_eid=root)
    assert cert.replayable
    assert cert.n_egraph_dependent == 0
    # the whole merge is ONE named witness step at the root
    assert cert.n_steps == 1
    step = cert.steps[0]
    assert step.rule.startswith("trace_lift#")
    assert not step.egraph_dependent
    assert op_repr(step.lhs) == op_repr(term)
    assert op_repr(step.rhs) == op_repr(lifts[0].term)
    # the synthesised Rewrite travels inside the certificate
    assert step.rule in cert.rules
    assert cert.rules_used == [step.rule]
    # strict replay — no trusted assertions
    out = verify_certificate(term, cert, strict=True)
    assert op_repr(out) == op_repr(lifts[0].term)


def test_witnessed_split_offer_also_replays():
    """The channel-split parl offer gets its own witness too."""
    T, d = 4, 8
    term, _, _ = _diag_term(T, d)
    eg = EGraph()
    root = eg.add_term(term)
    lifts = lift_scan_to_trace(eg, root_eid=root, witness=True)
    assert {l.split for l in lifts} == {None, (4, 4)}
    for l in lifts:
        cert = eg.certificate(term, l.term, root_eid=root)
        assert cert.n_egraph_dependent == 0
        out = verify_certificate(term, cert, strict=True)
        assert op_repr(out) == op_repr(l.term)


def test_witness_step_at_subterm_position():
    """The witness replay is positional: it fires wherever the offered
    member sits inside a larger term, not only at the root."""
    T, d = 3, 4
    term, _, _ = _diag_term(T, d)
    eg = EGraph()
    root = eg.add_term(term)
    lift = lift_scan_to_trace(
        eg, root_eid=root, channel_splits=None, witness=True
    )[0]
    # embed src/dst under an outer op so the merge sits at path (0,)
    src2 = Op.make("tanh", term)
    dst2 = Op.make("tanh", lift.term)
    eg.add_term(src2)
    cert = eg.certificate(src2, dst2)
    assert cert.n_egraph_dependent == 0
    assert cert.steps[0].path == (0,)
    out = verify_certificate(src2, cert, strict=True)
    assert op_repr(out) == op_repr(dst2)


# ---------------------------------------------------------------------------
#  (b) the witness replays standalone on fresh tensors
# ---------------------------------------------------------------------------


def test_witness_replays_standalone_fp64():
    """The cert's witness is a self-contained Rewrite: re-match its LHS
    on the real term, re-instantiate the RHS, and evaluate BOTH sides
    on fresh fp64 data — the asserted equality holds numerically."""
    T, d = 5, 4
    term, inputs, _ = _diag_term(T, d)
    eg = EGraph()
    root = eg.add_term(term)
    lift = lift_scan_to_trace(
        eg, root_eid=root, channel_splits=None, witness=True
    )[0]
    cert = eg.certificate(term, lift.term, root_eid=root)
    step = cert.steps[0]
    rule = cert.rules[step.rule]
    assert isinstance(rule, Rewrite)

    # standalone replay of the rule object itself — no e-graph:
    m = _term_match(rule.lhs, term)
    assert m is not None  # concrete LHS matches
    r = _term_instantiate(rule.rhs, m)
    assert op_repr(r) == op_repr(step.rhs)

    # and the equality it asserts is real, not a bookkeeping artifact:
    # fresh fp64 environment, evaluate lhs vs rhs.
    env, x = _env(T, d, seed=123)
    want = _eval(rule.lhs, inputs, env, x)
    got = _eval(rule.rhs, inputs, env, x)
    assert (got - want).abs().max().item() < 1e-11


# ---------------------------------------------------------------------------
#  (c) no witness attached -> pre-fix behaviour preserved
# ---------------------------------------------------------------------------


def test_unwitnessed_offer_stays_egraph_dependent():
    """``witness=False`` keeps the merge honestly flagged and strict
    replay refuses it — the pre-witness behaviour."""
    T, d = 4, 4
    term, _, _ = _diag_term(T, d)
    eg = EGraph()
    root = eg.add_term(term)
    lifts = lift_scan_to_trace(
        eg, root_eid=root, channel_splits=None, witness=False
    )
    assert len(lifts) == 1
    cert = eg.certificate(term, lifts[0].term, root_eid=root)
    assert cert.n_egraph_dependent >= 1
    assert not cert.replayable
    # non-strict replay substitutes the trusted assertion -> dst
    out = verify_certificate(term, cert)
    assert op_repr(out) == op_repr(lifts[0].term)
    with pytest.raises(CertificateVerificationError):
        verify_certificate(term, cert, strict=True)


def test_bare_union_without_witness_untouched():
    """The mechanism is opt-in: a plain union still records rule=None
    and certificates flag it egraph_dependent."""
    x = Var("x", TensorType((4,)))
    a = Op.make("neg", x)
    b = Op.make("square", x)
    eg = EGraph()
    ea, eb = eg.add_term(a), eg.add_term(b)
    assert eg.union(ea, eb)
    assert eg.merge_log[-1].rule is None
    cert = eg.certificate(a, b)
    assert cert.n_egraph_dependent == 1
    with pytest.raises(CertificateVerificationError):
        verify_certificate(a, cert, strict=True)


# ---------------------------------------------------------------------------
#  (d) the generic API — what pair_shared_input_linears would adopt
# ---------------------------------------------------------------------------


def test_union_witness_api_on_manual_merge():
    """``eg.union(a, b, witness=Rewrite(lhs, rhs))``: the pass asserts
    ``lhs == rhs`` for these concrete terms; the certificate replays
    the assertion as a named rewrite step."""
    x = Var("x", TensorType((4,)))
    src = Op.make("neg", x)
    dst = Op.make("mul", x, Const(-1.0))  # a true pointwise equality
    eg = EGraph()
    ea, eb = eg.add_term(src), eg.add_term(dst)
    w = Rewrite(
        "demo_witness", lhs=src, rhs=dst, law="demo: neg(x) = x * -1"
    )
    assert eg.union(ea, eb, witness=w)
    assert eg.merge_log[-1].rule == "demo_witness"

    cert = eg.certificate(src, dst)
    assert cert.n_egraph_dependent == 0
    assert cert.rules_used == ["demo_witness"]
    out = verify_certificate(src, cert, strict=True)
    assert op_repr(out) == op_repr(dst)

    # and the witness survives tampering checks like any rule step
    cert.steps[0].rhs = Op.make("mul", x, Const(2.0))
    with pytest.raises(CertificateVerificationError):
        verify_certificate(src, cert, strict=True)
