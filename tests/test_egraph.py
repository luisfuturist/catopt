"""Tests for the e-graph and equality saturation."""

from catopt.egraph import EGraph, ENode, UnionFind
from catopt.ir import Const, Op, Param, TensorType, Var, op_repr
from catopt.rules import (
    CATEGORICAL_RULES,
)


def test_union_find_basic():
    uf = UnionFind()
    a = uf.make()
    b = uf.make()
    c = uf.make()
    assert uf.find(a) == a
    assert uf.find(b) == b
    assert uf.find(c) == c
    assert uf.union(a, b) is True
    assert uf.find(a) == uf.find(b)
    assert uf.find(c) != uf.find(a)
    assert uf.union(a, b) is False  # already same set
    assert uf.union(a, c) is True
    assert uf.find(a) == uf.find(b) == uf.find(c)


def test_enode_equality():
    e1 = ENode("add", (0, 1))
    e2 = ENode("add", (0, 1))
    e3 = ENode("add", (1, 0))
    assert e1 == e2
    assert e1 != e3


def test_add_leaf():
    eg = EGraph()
    eid = eg.add_leaf("x")
    assert eg.find(eid) == eid
    # Adding the same leaf again returns the same e-class
    eid2 = eg.add_leaf("x")
    assert eg.find(eid) == eg.find(eid2)


def test_add_enode():
    eg = EGraph()
    e1 = eg.add_leaf("a")
    e2 = eg.add_leaf("b")
    eid = eg.add_enode("add", (e1, e2))
    assert eid is not None
    # Adding the same enode returns the same class
    eid2 = eg.add_enode("add", (e1, e2))
    assert eg.find(eid) == eg.find(eid2)


def test_add_term():
    x = Var("x", TensorType((1, 4)))
    y = Var("y", TensorType((1, 4)))
    term = Op.make("add", x, y)
    eg = EGraph()
    eid = eg.add_term(term)
    assert eid is not None
    assert eg.find(eid) == eid


def test_union_merges_classes():
    eg = EGraph()
    x = Var("x", TensorType((1, 4)))
    y = Var("y", TensorType((1, 4)))
    e1 = eg.add_term(Op.make("neg", x))
    e2 = eg.add_term(Op.make("neg", x))  # same term, same class
    assert eg.find(e1) == eg.find(e2)
    # Different leaf
    e3 = eg.add_term(Op.make("neg", y))
    assert eg.find(e1) != eg.find(e3)
    # Union them
    eg.union(e1, e3)
    assert eg.find(e1) == eg.find(e3)


def test_pattern_match_metavar():
    """Match a simple pattern with metavariables."""
    x = Var("x", TensorType((1, 4)))
    y = Var("y", TensorType((1, 4)))
    term = Op.make("add", x, y)
    eg = EGraph()
    eid = eg.add_term(term)

    # Match: add("a", "b") should match add(x, y)
    pattern = Op.make("add", "a", "b")
    matches = eg.matches(pattern, eg.find(eid))
    assert len(matches) == 1
    # The match should bind a and b
    m = matches[0]
    assert "a" in m
    assert "b" in m


def test_repeated_metavar_enforces_same_eclass():
    """A repeated metavariable must bind ONE e-class everywhere.

    Soundness regression test: x@W1 + y@W2 (different inputs) must NOT
    match add(matmul(x,W1), matmul(x,W2)), because accepting it would let
    a rule conclude x@W1 + y@W2 == x@(W1+W2) — a false proof.
    """
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    w1 = Param("W1", TensorType((4, 4)))
    w2 = Param("W2", TensorType((4, 4)))

    # Pattern requiring the same input twice.
    pattern = Op.make(
        "add",
        Op.make("matmul", "x", "W1"),
        Op.make("matmul", "x", "W2"),
    )

    # Case 1: same input x in both — must match.
    same = Op.make(
        "add", Op.make("matmul", x, w1), Op.make("matmul", x, w2)
    )
    eg_same = EGraph()
    eid_same = eg_same.add_term(same)
    assert len(eg_same.matches(pattern, eg_same.find(eid_same))) >= 1

    # Case 2: different inputs x and y — must NOT match.
    diff = Op.make(
        "add", Op.make("matmul", x, w1), Op.make("matmul", y, w2)
    )
    eg_diff = EGraph()
    eid_diff = eg_diff.add_term(diff)
    assert len(eg_diff.matches(pattern, eg_diff.find(eid_diff))) == 0


def test_extract_best():
    """Extract the minimum-cost term from an e-class."""
    x = Var("x", TensorType((1, 4)))
    zero = Const(0)
    y = Var("y", TensorType((1, 4)))

    term = Op.make("add", x, zero)  # add(x, 0)
    eg = EGraph()
    eid = eg.add_term(term)

    # Cost function: prefer fewer ops (count_cost)
    from catopt.cost import count_cost

    best = eg.extract_best(eid, count_cost)
    # Should extract either add(x, 0) or x (they're in the same e-class
    # only if we've applied rewrites, but we haven't here — so best = add(x,0))
    assert best is not None


def test_associativity_rewriting():
    """Test that associativity rewrites explore different association orders."""
    x = Var("x", TensorType((1, 4)))
    A = Param("A", TensorType((4, 4)))
    B = Param("B", TensorType((4, 4)))
    C = Param("C", TensorType((4, 4)))

    # x @ A @ B @ C = ((x @ A) @ B) @ C  (left-nested)
    left_nested = Op.make(
        "matmul", Op.make("matmul", Op.make("matmul", x, A), B), C
    )

    eg = EGraph()
    eid = eg.add_term(left_nested)

    # Run with both associativity directions
    stats = eg.run(
        CATEGORICAL_RULES, eid, max_iterations=10, max_nodes=10000
    )
    print(f"Stats: {stats}")

    # Extract the best (minimum FLOPs) form
    from catopt.cost import flops_cost

    best = eg.extract_best(eid, flops_cost)
    print(f"Best: {op_repr(best)}")

    # The e-graph should have explored both associations
    # and they should be in the same e-class (equivalent)
    assert best is not None

    # Verify both forms exist in the e-graph's root e-class
    root_class = eg.get_class(eid)
    ops_found = set()
    for node in root_class.nodes:
        if node.op == "matmul":
            # Check nesting depth
            ops_found.add(len(node.children))
    # The e-class should contain multiple equivalent forms
    assert len(root_class.nodes) >= 2


def test_naturality_rewriting():
    """Test that naturality rewrites apply correctly.

    W @ (x * c)  ==  (W @ x) * c   (naturality of scalar mult)
    """
    x = Var("x", TensorType((1, 4)))
    W = Param("W", TensorType((4, 4)))
    c = Const(2.0)

    # W @ (x * c)  = matmul(W, mul(x, c))
    original = Op.make("matmul", W, Op.make("mul", x, c))

    eg = EGraph()
    eid = eg.add_term(original)
    eg.run(CATEGORICAL_RULES, eid, max_iterations=5, max_nodes=1000)

    from catopt.cost import count_cost

    best = eg.extract_best(eid, count_cost)

    root_class = eg.get_class(eid)
    # Should contain both forms
    assert len(root_class.nodes) >= 2
    eg.run(CATEGORICAL_RULES, eid, max_iterations=5, max_nodes=1000)

    from catopt.cost import count_cost, flops_cost

    best = eg.extract_best(eid, flops_cost)

    # The naturality rule should have produced: mul(matmul(x, W), c)
    # Both forms should be equivalent
    root_class = eg.get_class(eid)
    assert len(root_class.nodes) >= 2  # at least 2 equivalent forms


def test_saturation_terminates():
    """Equality saturation should reach a fixed point."""
    x = Var("x", TensorType((1, 4)))
    y = Var("y", TensorType((1, 4)))
    z = Var("z", TensorType((1, 4)))

    term = Op.make("add", x, Op.make("add", y, z))  # add(x, add(y, z))
    eg = EGraph()
    eid = eg.add_term(term)

    stats = eg.run(
        CATEGORICAL_RULES, eid, max_iterations=50, max_nodes=10000
    )
    # Should terminate well before 50 iterations
    assert stats["iterations"] < 50


# ---------------------------------------------------------------------------
#  Incremental (dirty-frontier) saturation + bounded saturation
# ---------------------------------------------------------------------------


def test_incremental_frontier_refires_at_dirtied_classes():
    """After saturation, a rule must re-match classes changed by merges.

    The dirty-frontier schedule skips classes whose reachable subgraph
    is unchanged; this test pins the completeness side: a merge that
    creates a NEW match in a class the rule already scanned must still
    fire on the next run() call.
    """
    x = Var("x", TensorType((1, 4)))
    y = Var("y", TensorType((1, 4)))
    z = Var("z", TensorType((1, 4)))
    w = Var("w", TensorType((1, 4)))

    # add(add(x, y), z): assoc_add needs a RIGHT-nested member
    # add(a, add(b, c)) — which only appears once the e-class also
    # holds add(x, add(y, z)) (comm/assoc interaction).
    term = Op.make("add", Op.make("add", x, y), z)
    eg = EGraph()
    eid = eg.add_term(term)
    from catopt.rules import ASSOC_ADD, COMM_ADD

    stats = eg.run([ASSOC_ADD, COMM_ADD], eid, max_iterations=20)
    root = eg.get_class(eid)
    # Saturation reached every bracketing of the 3-summand chain:
    # (x+y)+z, x+(y+z), and the commuted orders.
    assert stats["iterations"] < 20
    # A second run after a manual union still finds new matches.
    eg2 = EGraph()
    e2 = eg2.add_term(Op.make("add", x, Op.make("add", y, z)))
    eg2.run([ASSOC_ADD], e2, max_iterations=5)
    n1 = eg2.n_enodes
    # Union w into x's class — ancestors (all add members containing
    # the leaf class) become dirty and are re-searched.
    weid = eg2.add_term(w)
    xid = eg2.add_term(x)
    eg2.union(weid, xid)
    eg2.run([ASSOC_ADD], e2, max_iterations=5)
    # The merge propagated dirtiness through the parent edges: new
    # bracketings over the merged class were instantiated.
    assert eg2.n_enodes > n1


def test_rule_budgets_bound_expansion():
    """rule_budgets caps how many enodes a rule may contribute."""
    x = Var("x", TensorType((1, 4)))
    ys = [Var(f"y{i}", TensorType((1, 4))) for i in range(5)]
    # A 6-summand right-nested chain: full AC closure is hundreds of
    # bracketings; a budget of 20 must hold assoc_add far below it.
    term = x
    for y in ys:
        term = Op.make("add", term, y)
    eg = EGraph()
    eid = eg.add_term(term)
    from catopt.rules import ASSOC_ADD, COMM_ADD

    eg_full = EGraph()
    eid_full = eg_full.add_term(term)
    stats_full = eg_full.run(
        [ASSOC_ADD, COMM_ADD], eid_full, max_iterations=30
    )
    assert stats_full["n_enodes"] > 100  # the closure is big

    stats = eg.run(
        [ASSOC_ADD, COMM_ADD],
        eid,
        max_iterations=30,
        rule_budgets={"assoc_add": 20, "comm_add": 20},
    )
    assert stats["n_enodes"] < stats_full["n_enodes"]
    assert stats["rule_budgets"]["assoc_add"] >= 20
    assert "assoc_add" in stats["budget_suspended"]
    # Extraction still works and returns a valid member.
    from catopt.cost import count_cost

    best = eg.extract_best(eid, count_cost)
    assert best is not None


def test_rule_budget_persists_across_run_calls():
    """Budget accounting is cumulative over the e-graph's lifetime —
    a second run() cannot re-open an exhausted budget."""
    xs = [Var(f"v{i}", TensorType((1, 4))) for i in range(6)]
    term = xs[0]
    for v in xs[1:]:
        term = Op.make("add", term, v)
    eg = EGraph()
    eid = eg.add_term(term)
    from catopt.rules import ASSOC_ADD, COMM_ADD

    eg.run(
        [ASSOC_ADD, COMM_ADD],
        eid,
        max_iterations=10,
        rule_budgets={"assoc_add": 8},
    )
    spent1 = eg._budget_spent["assoc_add"]
    assert spent1 >= 8
    eg.run(
        [ASSOC_ADD, COMM_ADD],
        eid,
        max_iterations=10,
        rule_budgets={"assoc_add": 8},
    )
    # Still suspended: cumulative spend stays at the first-run level.
    assert eg._budget_spent["assoc_add"] == spent1


def test_matches_max_results_cap():
    """matches(pattern, eid, max_results) truncates enumeration."""
    xs = [Var(f"v{i}", TensorType((1, 4))) for i in range(4)]
    term = Op.make(
        "add",
        xs[0],
        Op.make("add", xs[1], Op.make("add", xs[2], xs[3])),
    )
    eg = EGraph()
    eid = eg.add_term(term)
    from catopt.rules import ASSOC_ADD, COMM_ADD

    eg.run([ASSOC_ADD, COMM_ADD], eid, max_iterations=10)
    root = eg.find(eid)
    full = eg.matches(
        Op.make("add", "a", Op.make("add", "b", "c")), root
    )
    capped = eg.matches(
        Op.make("add", "a", Op.make("add", "b", "c")),
        root,
        max_results=3,
    )
    assert len(capped) == 3
    assert len(full) > 3


def test_min_term_returns_smallest_member():
    """_min_term picks the fewest-op member of an e-class."""
    x = Var("x", TensorType((1, 4)))
    zero = Const(0)
    eg = EGraph()
    eid = eg.add_term(Op.make("add", x, zero))
    from catopt.rules import ID_ADD

    eg.run([ID_ADD], eid, max_iterations=5)
    t, s = eg._min_term(eid, {})
    assert s == 1  # the bare Var beats add(x, 0)


def test_fresh_rule_full_scan_after_saturation():
    """A rule that has never run on this e-graph searches every class.

    The dirty-frontier schedule only covers rules with an established
    frontier; a first-time rule must still see the whole graph — even
    when the dirty set is empty after a prior saturated run.  Without
    this, ``eg.run`` calls with *different* rule sets would silently
    drop every match of the late-arriving rules.
    """
    x = Var("x", TensorType((1, 4)))
    y = Var("y", TensorType((1, 4)))
    zero = Const(0)
    term = Op.make("add", x, Op.make("add", y, zero))
    eg = EGraph()
    eid = eg.add_term(term)
    from catopt.rules import COMM_ADD, ID_ADD

    eg.run([ID_ADD], eid, max_iterations=5)
    assert not eg._dirty  # saturated: frontier is empty
    assert "comm_add" not in eg.rule_fires
    # comm_add has never scanned this graph — it must get a full pass
    # and find the commuted members despite the empty frontier.
    eg.run([ID_ADD, COMM_ADD], eid, max_iterations=5)
    assert eg.rule_fires.get("comm_add", 0) >= 1
    adds = [n for n in eg.get_class(eid).nodes if n.op == "add"]
    assert len(adds) >= 2  # add(x, add(y,0)) and add(add(y,0), x)


def test_incremental_saturation_same_fixed_point():
    """Frontier-driven saturation reaches the same closure a full
    re-scan would: a second run() with the same rules is a no-op."""
    x = Var("x", TensorType((1, 4)))
    A = Param("A", TensorType((4, 4)))
    B = Param("B", TensorType((4, 4)))
    C = Param("C", TensorType((4, 4)))
    term = Op.make(
        "matmul", Op.make("matmul", Op.make("matmul", x, A), B), C
    )
    eg = EGraph()
    eid = eg.add_term(term)
    from catopt.rules import CATEGORICAL_RULES

    eg.run(CATEGORICAL_RULES, eid, max_iterations=10, max_nodes=10000)
    n1 = eg.n_enodes
    stats2 = eg.run(
        CATEGORICAL_RULES, eid, max_iterations=10, max_nodes=10000
    )
    assert stats2["iterations"] <= 1  # clean frontier: no work
    assert eg.n_enodes == n1
    from catopt.cost import flops_cost

    best = eg.extract_best(eid, flops_cost)
    assert best is not None
