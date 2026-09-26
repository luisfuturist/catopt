"""Hybrid SSM+attention blocks — TWO monoid carriers in one e-graph.

``catopt.models.hybrid.HybridBlock`` stacks a diagonal-affine scan
(``aff_diag`` / ``affd_compose`` / ``applyd`` — ``SCAN_DIAG_LAWS``)
under chunked self-attention (``om_elem`` / ``om_compose`` /
``om_apply`` — ``OM_LAWS``).  ``TwoLayerHybrid`` goes SSM → attention →
SSM so the second recurrence's inputs are reads of the attention
output.  The question: does anything *cross* the carrier boundary?

Findings encoded as tests (T=16, d=16 unless noted):

* **Both carriers coexist.**  One stratified saturation of the union
  law set produces aff_diag×16 / applyd×136 / affd_compose×120 AND
  om_elem×4 / om_apply / om_compose in 444 e-nodes (~0.7 s).

* **The om carrier consumes the scan output.**  Every ``om_elem``
  enode's score-input e-class reaches ``applyd`` members — the chunked
  score blocks are ``matmul(q·s, chunk(k_i)ᵀ)`` over
  ``linear(stack(applyd …))``.  No bespoke rule encodes this nesting;
  it falls out of metavariable binding.

* **Exported-attr spelling gap (found, honest negative).**
  torch.export spells ``cat``'s axis ``arg1=-2`` and ``chunk``'s dims
  ``arg1/arg2``; the om rule family produces ``dim=``/``chunks=``/
  ``dim=``.  On the RAW exported IR, ``matmul_t_concat`` is vetoed by
  its shape check (the ``chunk`` mis-read makes k_i look (16,8)) and
  ``om_split`` can never reunite (its LHS wants both concats in ONE
  spelling, but rule-produced score concats are ``dim`` while exported
  value concats are ``arg1``).  A spelling normalisation in the test
  harness unblocks the whole chain — this is a bridge↔rules interface
  bug a real pipeline would fix in the bridge.

* **Extraction does not mix carriers greedily.**  Under ``flops_cost``
  the scan returns to raw ``add(mul)`` steps (5d per lifted step vs 3d
  raw) and attention stays dense softmax; under min-depth the scan
  lifts (``applyd``) but attention stays dense (om path is deeper).
  A term containing BOTH carriers exists and is fp64-exact — it just
  has to be *coordinated* (overrides), like the pairing pass.

* **Cross-layer rewrites DO fire.**  ``linear_row_scale_rev`` +
  ``linear_channel_scale`` cascade the attention ``1/√d`` scale through
  ``q_proj`` into a *weight-only* ``mul(W_q, c)`` — compile-time fold.
  In ``TwoLayerHybrid``, ``assoc_linear`` fuses the attention
  ``out_proj`` into BOTH of the second SSM's input projections —
  linear(linear(u, W_out), W_dec2) → linear(u, W_dec2@W_out) — a real
  inter-layer weight fusion no one pattern-wrote.  And
  ``pair_shared_input_linears`` pairs decay/B on x AND q/k/v on the
  scan stack.

* **Not found:** no rule fuses the *carriers* themselves — no law maps
  a scan of local summaries into an om element (the linear-attention ↔
  scan duality is NOT derivable from these rules), and no ``sdpa``
  fold fires because the attention is unmasked (the sdpa-fold family
  requires an additive-mask or masked_fill shape).  A causal mask would
  ALSO block ``om_split``: the mask op wraps the score concat exactly
  like the scalar mul did — chunked masked attention would need a
  "masked_fill distributes over concat" law nobody wrote.
"""


import torch

from catopt import meta
from catopt import rules as R
from catopt.cost import dag_cost, flops_cost
from catopt.egraph import EGraph
from catopt.ir import IR, Op, Var, op_repr
from catopt.models.hybrid import HybridBlock, TwoLayerHybrid
from catopt.om import OM_LAWS
from catopt.torch_bridge import export_to_ir, ir_to_torch_module

# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _normalize_attrs(term, memo=None):
    """Unify exported positional attr spellings with the rule-side ones.

    torch.export emits ``cat(ts, -2)`` as ``concat(arg1=-2)`` and
    ``t.chunk(n, -2)`` as ``chunk(arg1=n, arg2=-2)``, while every
    rule-produced concat/chunk uses ``dim=``/``chunks=``.  The om
    homomorphism needs ONE spelling on both concat slots; the chunk
    mis-spelling also breaks ``_shape_of`` (reads ``dim``), which is
    what actually vetoes ``matmul_t_concat`` on raw exports.
    Semantically a no-op — the torch bindings accept both spellings.
    """
    if memo is None:
        memo = {}
    k = id(term)
    if k in memo:
        return memo[k]
    if not isinstance(term, Op):
        memo[k] = term
        return term
    args = tuple(_normalize_attrs(a, memo) for a in term.args)
    attrs = dict(term.attrs)
    if term.op == "concat" and "dim" not in attrs and "arg1" in attrs:
        attrs["dim"] = attrs.pop("arg1")
    elif term.op == "chunk":
        if "chunks" not in attrs and "arg1" in attrs:
            attrs["chunks"] = attrs.pop("arg1")
        if "dim" not in attrs and "arg2" in attrs:
            attrs["dim"] = attrs.pop("arg2")
    elif term.op == "split":
        if "dim" not in attrs and "arg2" in attrs:
            attrs["dim"] = attrs.pop("arg2")
    out = Op.make(term.op, *args, **attrs)
    memo[k] = out
    return out


def _opdepth(t, memo):
    """Critical-path depth of a term (shared-subterm DAG memoised)."""
    if not isinstance(t, Op):
        return 0
    k = id(t)
    if k not in memo:
        memo[k] = 1 + max(
            (_opdepth(a, memo) for a in t.args), default=0
        )
    return memo[k]


def _class_has_op(eg, eid, opname, seen=None):
    """Does the e-class dependency cone rooted at eid contain opname?"""
    seen = set() if seen is None else seen
    eid = eg.find(eid)
    if eid in seen:
        return False
    seen.add(eid)
    for n in eg.get_class(eid).nodes:
        if n.op == opname:
            return True
        if any(_class_has_op(eg, c, opname, seen) for c in n.children):
            return True
    return False


def _op_census(eg):
    from collections import Counter

    return Counter(n.op for n in eg._node_to_class)


def _run_hybrid(
    m,
    x,
    laws=None,
    normalize=True,
    max_iterations=14,
    max_nodes=400_000,
):
    """Export, (optionally) attr-normalise, stratified-saturate.

    Returns (ir, source_tensors, eg, stratified_run output).
    """
    laws = _LAWS if laws is None else laws
    ir, st = export_to_ir(m, x)
    root = _normalize_attrs(ir.root) if normalize else ir.root
    eg = EGraph()
    out = meta.stratified_run(
        eg,
        laws,
        root,
        max_iterations=max_iterations,
        max_nodes=max_nodes,
        extract_fn=eg.extract_min_depth,
    )
    return ir, st, eg, out


def _verify(m, x, ir, st, term, tol=1e-10):
    """Lower *term* and check fp64-equivalence against the module."""
    opt_ir = IR(
        root=term,
        inputs=ir.inputs,
        input_names=ir.input_names,
        params=ir.params,
    )
    mod = ir_to_torch_module(opt_ir, param_values=st)
    mod.eval()
    with torch.no_grad():
        return (m(x) - mod(x)).abs().max().item()


#: The union law set: diagonal-scan carrier + online-softmax carrier +
#: core simplification/categorical rules.  ``stratified_run`` drops the
#: coherent members (comm/assoc/id/involution, affd_assoc, om_assoc,
#: assoc_matmul) and saturates with the contentful remainder.
_LAWS = (
    R.SCAN_DIAG_LAWS
    + OM_LAWS
    + R.SIMPLIFICATION_RULES
    + R.CATEGORICAL_RULES
)


# ---------------------------------------------------------------------------
#  (a) export sanity — the shapes the lifts match must be present
# ---------------------------------------------------------------------------


def test_hybrid_export_shape():
    """The module exports as stack(add(mul,mul)… steps) feeding
    softmax(q @ cat(chunk k)ᵀ) @ cat(chunk v)."""
    torch.manual_seed(0)
    T, D = 8, 16
    m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, _ = export_to_ir(m, x)

    ops = _op_census_term(ir.root)
    # T recurrence steps, each add(mul(a_t, h), mul(b_t, x_t))
    assert ops["add"] == T and ops["mul"] >= 2 * T
    assert ops["select"] == 3 * T  # a[t], b[t], x[t] per step
    assert ops["stack"] == 1  # the SSM sequence output
    assert ops["softmax"] == 1 and ops["matmul"] == 2
    # chunked K/V: 2 cats over 2 getitem-folded chunk projections each
    assert ops["concat"] == 2 and ops["chunk"] == 4
    assert ops["linear"] == 6  # decay,B + q,k,v + out
    # root is the output projection over the attention matmul
    assert ir.root.op == "linear"


def _op_census_term(root):
    from collections import Counter

    c = Counter()
    seen = set()

    def go(t):
        if id(t) in seen:
            return
        seen.add(id(t))
        if isinstance(t, Op):
            c[t.op] += 1
            for a in t.args:
                go(a)

    go(root)
    return c


# ---------------------------------------------------------------------------
#  (b) both carriers in one e-graph + which rules fired
# ---------------------------------------------------------------------------


def test_both_carriers_coexist_in_one_egraph():
    """One saturation holds aff_diag/applyd AND om_elem/om_apply."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, eg, out = _run_hybrid(m, x)

    stats = out["stats"]
    census = _op_census(eg)
    print(f"\n[hybrid] stats={stats}")
    print(f"[hybrid] rule_fires={eg.rule_fires}")
    print(
        f"[hybrid] carrier census="
        f" aff_diag={census['aff_diag']} applyd={census['applyd']}"
        f" affd_compose={census['affd_compose']}"
        f" om_elem={census['om_elem']} om_apply={census['om_apply']}"
        f" om_compose={census['om_compose']}"
    )

    # (a) both carrier domains materialised in the SAME e-graph
    assert census["aff_diag"] == T  # one map leaf per step
    assert census["applyd"] > 0 and census["affd_compose"] > 0
    assert census["om_elem"] >= 2 and census["om_apply"] >= 1
    assert census["om_compose"] >= 1  # OM_SPLIT actually chunked

    # The whole om chain fired on a *real exported graph* (post
    # spelling-normalisation): lift → score-concat → split → merge.
    for name in (
        "om_lift",
        "matmul_t_concat",
        "om_split",
        "om_merge",
        "om_unlift",
    ):
        assert eg.rule_fires.get(name, 0) >= 1, name
    # The scan side: step lifts + step composes fired per step.
    assert (
        sum(
            v
            for k, v in eg.rule_fires.items()
            if k.startswith("affd_lift")
        )
        >= T
    )
    # Coherent laws were stratified away — computed, not stored.
    assert "affd_assoc" in out["coherent_dropped"]
    assert "om_assoc" in out["coherent_dropped"]


def test_om_elems_consume_scan_outputs():
    """Cross-domain nesting: every om_elem's score input reaches an
    applyd member — the attention carrier reads the scan stack."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, eg, out = _run_hybrid(m, x)

    elems = [n for n in eg._node_to_class if n.op == "om_elem"]
    assert elems
    for n in elems:
        s_cid = n.children[0]  # score block e-class
        assert _class_has_op(eg, s_cid, "applyd") or _class_has_op(
            eg, s_cid, "aff_diag"
        ), "om_elem score block does not reach the scan carrier"

    # And the root e-class's attention member shows both program forms:
    # dense matmul(softmax) and lifted om_apply in ONE e-class.
    attn_classes = [
        cid
        for cid in eg._classes
        if any(n.op == "om_apply" for n in eg._classes[cid].nodes)
    ]
    assert attn_classes
    members = {n.op for n in eg._classes[attn_classes[0]].nodes}
    assert "matmul" in members and "om_apply" in members


# ---------------------------------------------------------------------------
#  (c) the attr-spelling gap — honest negative on RAW export
# ---------------------------------------------------------------------------


def test_raw_export_fires_om_split():
    """The bridge now canonicalises exported positional spellings
    (``concat(arg1=-2)`` → ``dim=``, ``chunk(arg1,arg2)`` →
    ``chunks=``/``dim=``) at the boundary, so the om homomorphism
    fires on RAW exports — no test-side normalisation needed.
    (Was ``test_raw_export_blocks_om_split``: this used to be the
    documented bridge↔rules interface gap.)"""
    torch.manual_seed(0)
    T, D = 16, 16
    m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, eg, out = _run_hybrid(m, x, normalize=False)

    census = _op_census(eg)
    # Scan carrier still lifts fine (spelling-independent).
    assert census["applyd"] > 0 and census["affd_compose"] > 0
    # The whole om chain fires on the raw export now.
    assert eg.rule_fires.get("om_lift", 0) >= 1
    assert census["om_compose"] > 0
    assert (
        eg.rule_fires.get("om_split", 0)
        + eg.rule_fires.get("om_split_arg1", 0)
    ) > 0
    assert (
        eg.rule_fires.get("matmul_t_concat", 0)
        + eg.rule_fires.get("matmul_t_concat_arg1", 0)
    ) > 0


# ---------------------------------------------------------------------------
#  (d) extraction — carriers coexist but greedy never mixes them
# ---------------------------------------------------------------------------


def test_extraction_carrier_report():
    """Greedy flops: raw adds + dense softmax.  Min-depth: applyd scan +
    dense softmax.  The om form never wins a per-class objective — it
    must be forced (same story as test_om_monoid's _extract_chunked)."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, eg, out = _run_hybrid(m, x)
    root = out["root_eid"]

    best_f = eg.extract_best(root, flops_cost)
    rep_f = op_repr(best_f)
    best_d = out["canonical_best"]
    rep_d = op_repr(best_d)
    print(
        f"\n[hybrid] flops-best: applyd={'applyd' in rep_f}"
        f" om={'om_apply' in rep_f}  cost={dag_cost(best_f, flops_cost):.0f}"
    )
    print(
        f"[hybrid] depth-best: applyd={'applyd' in rep_d}"
        f" om={'om_apply' in rep_d}"
    )

    # flops extraction prefers the raw sequential spine (3d/step beats
    # 5d/step lifted) and dense softmax — honest negative for mixing.
    assert "applyd" not in rep_f and "om_apply" not in rep_f
    # min-depth lifts the scan (applyd) but keeps dense attention.
    assert "applyd" in rep_d and "affd_compose" in rep_d
    assert "om_apply" not in rep_d
    # both extracted forms are fp64-exact
    assert _verify(m, x, ir, st, best_f) < 1e-10
    assert _verify(m, x, ir, st, best_d) < 1e-10


def test_coordinated_extraction_has_both_carriers():
    """Force-extract om_apply(om_compose…) at the attention class AND
    applyd at every stack-arg class: ONE term, BOTH carriers, fp64
    exact — the carriers coexist in [G], the cost model just never
    nominates the mixed member."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, eg, out = _run_hybrid(m, x)
    canon = eg.find(out["root_eid"])

    ov = {}
    # pin every om_compose-holding class to a compose member
    for cid in list(eg._classes):
        c = eg.find(cid)
        comps = [
            n for n in eg._classes[c].nodes if n.op == "om_compose"
        ]
        if comps:
            ov.setdefault(c, comps[0])
    # pin the attention class to a composed om_apply member
    pinned_om = False
    for cid in list(eg._classes):
        c = eg.find(cid)
        for n in eg._classes[c].nodes:
            if n.op == "om_apply" and _class_has_op(
                eg, n.children[0], "om_compose"
            ):
                ov[c] = n
                pinned_om = True
    assert pinned_om, "no composed om_apply member materialised"
    # pin every stack element (y_t) to an applyd member
    stacks = [n for n in eg._node_to_class if n.op == "stack"]
    assert stacks
    pinned_d = 0
    for s in stacks:
        for c in s.children:
            cc = eg.find(c)
            for n in eg._classes[cc].nodes:
                if n.op == "applyd":
                    ov[cc] = n
                    pinned_d += 1
                    break
    assert pinned_d == T

    term = eg.extract_best(canon, flops_cost, overrides=ov)
    rep = op_repr(term)
    assert "applyd" in rep and "om_apply" in rep
    assert "om_compose" in rep and "om_elem" in rep
    assert _verify(m, x, ir, st, term) < 1e-10


# ---------------------------------------------------------------------------
#  (e) cross-layer rewrites — what actually fired across the boundary
# ---------------------------------------------------------------------------


def test_scale_commutes_into_q_proj_weight():
    """The 1/√d attention scale slid through q_proj into a weight-only
    mul: linear_row_scale_rev then linear_channel_scale.  The folded
    mul(W_q, c) is param-only → compile-time work.  A scalar-naturality
    cascade NEITHER carrier law produced."""
    torch.manual_seed(0)
    T, D = 16, 16
    m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, eg, out = _run_hybrid(m, x)

    assert eg.rule_fires.get("linear_row_scale_rev", 0) >= 1
    assert eg.rule_fires.get("linear_channel_scale", 0) >= 1
    # the scale landed inside a weight position: some mul enode has
    # BOTH children param-only (mul(W_q, c)) — pre-rewrite every mul
    # in the graph contains a Var, so this node came from the fold.
    found = False
    for n in eg._node_to_class:
        if n.op == "mul" and len(n.children) == 2:
            ts = [eg.any_term(c) for c in n.children]
            if all(t is not None and not _is_var_sub(t) for t in ts):
                found = True
    assert found, "no weight-folded scale term found"


def _is_var_sub(t):
    return isinstance(t, Var) or (
        isinstance(t, Op) and any(_is_var_sub(a) for a in t.args)
    )


def test_pairing_pass_spans_the_boundary():
    """pair_shared_input_linears pairs projections on BOTH sides of the
    carrier seam: {decay_proj, B_proj} share x; {q,k,v(,W_q·c)} share
    the scan stack y.  The product law doesn't care which monoid reads
    the output."""
    from catopt.rules import pair_shared_input_linears

    torch.manual_seed(0)
    T, D = 16, 16
    m = HybridBlock(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, eg, out = _run_hybrid(m, x)
    root = out["root_eid"]

    groups = pair_shared_input_linears(eg)
    sizes = sorted(len(g) for g in groups)
    print(f"\n[hybrid] pairing groups: {sizes}")
    # {decay,B} on x (2 members) and {q,k,v, + scaled-q variant} on y
    assert len(groups) >= 2 and sizes[0] >= 2 and sizes[-1] >= 3

    # coordinated paired extraction is a legal equivalent program
    eg.rebuild()
    forced = eg.extract_paired(root, flops_cost, groups)
    assert forced is not None
    assert "split" in op_repr(forced)
    assert _verify(m, x, ir, st, forced) < 1e-10


# ---------------------------------------------------------------------------
#  (f) TwoLayerHybrid — SSM → attention → SSM
# ---------------------------------------------------------------------------


def test_two_layer_hybrid_composes_across_boundary():
    """The second recurrence's aff_diag leaves carry the attention
    carrier inside their translation vectors (select(y2) where y2's
    class holds om_apply), and assoc_linear fuses out_proj into BOTH
    second-layer input projections — an inter-layer weight merge."""
    torch.manual_seed(0)
    T, D = 8, 16
    m = TwoLayerHybrid(D, D, 16, T, n_chunks=2).eval().double()
    x = torch.randn(T, D, dtype=torch.float64)
    ir, st, eg, out = _run_hybrid(m, x)
    stats = out["stats"]
    census = _op_census(eg)
    print(f"\n[2layer] stats={stats}")
    print(f"[2layer] rule_fires={eg.rule_fires}")

    # both scans lifted (16 aff_diag leaves = 2 layers × T steps)
    assert census["aff_diag"] == 2 * T
    assert census["om_compose"] >= 1

    # THE cross-layer finding: linear(linear(u, W_out), W) matched on
    # the attention→SSM seam — out_proj fused into decay_proj2/B_proj2.
    assert eg.rule_fires.get("assoc_linear", 0) >= 1

    # every second-layer aff_diag translation reaches the om carrier
    hits = sum(
        1
        for n in eg._node_to_class
        if n.op == "aff_diag"
        and len(n.children) == 2
        and _class_has_op(eg, n.children[1], "om_apply")
    )
    print(f"[2layer] aff_diag leaves containing om_apply: {hits}")
    assert hits >= T

    # extracted term stays fp64-exact
    best = out["canonical_best"]
    assert "applyd" in op_repr(best)
    assert _verify(m, x, ir, st, best) < 1e-10
