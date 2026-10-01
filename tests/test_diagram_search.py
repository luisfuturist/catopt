"""Contraction-state search tests — plan 0013 stage 3.

Sibling of ``test_diagram.py``.  Pins the ``mode="search"`` driver
(:mod:`catopt_orchestrator.diagram_search`): the state space
(per-node reified bodies written back through ``_terms``), the
exact-merge state union, the sealed-node fallback for term-less
foreign moves, the reify cache, the depth/state bounds — and the
proving fixture: on ``_Stacked``, ``split_leaf`` materialises the
stacked weight's slices into plain ``Param``s so ``merge_projs``
can pair them, reaching a fused state (``split_leaf`` then
``merge_projs``) that greedy's claim order can never produce and
that is strictly cheaper than either move alone.
"""

import catopt_orchestrator.diagram as D
import catopt_orchestrator.diagram_search as DS
import catopt_orchestrator.morphisms as M
import torch
import torch.nn as nn
from catopt_core.cost import dag_cost, launch_aware_cost
from catopt_core.ir import Op, Param, TensorType
from catopt_orchestrator import (
    ContractionSearch,
    MorphismMatch,
    Optimizer,
    ReifySpec,
)
from catopt_torch.backend import TorchBackend

from tests.test_diagram import (
    _Chain,
    _lift,
    _Lin,
    _Parallel,
    _ResidOpaque,
    _SliceUser,
    _Stacked,
    _x,
)

# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


class _OwnLinear(nn.Module):
    """``linear(x, self.lin.weight)`` — a plain own-param projection."""

    def __init__(self, w: torch.Tensor) -> None:
        """Hold the weight in a stock ``nn.Linear``."""
        super().__init__()
        self.lin = nn.Linear(
            w.shape[1], w.shape[0], bias=False
        ).double()
        with torch.no_grad():
            self.lin.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project."""
        return self.lin(x)


class _SplitStack(nn.Module):
    """``b0(x) + b1(x)`` — one plain projection, one stacked slice.

    ``b1`` reads slice 1 of a stacked ``S`` whose slice 0 is bitwise
    equal to ``b0``'s private weight — so ``_RelinkLeaf`` can rewrite
    ``b0``'s body to read ``select(S, 0)`` exactly, after which the
    real ``SplitLeaf`` move has a 2-member leaf group it never saw on
    the initial lift.
    """

    def __init__(self, dim: int = 16) -> None:
        """Build w0 / S = stack(w0, w1) deterministically."""
        super().__init__()
        g = torch.Generator().manual_seed(310)
        w0 = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        w1 = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        self.S = nn.Parameter(torch.stack([w0, w1]))
        self.blocks = nn.ModuleList(
            [_OwnLinear(w0), _SliceUser(self.S, 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sum the members."""
        return self.blocks[0](x) + self.blocks[1](x)


class _RelinkLeaf:
    """Test move: relink ``blocks.0``'s private weight to slice 0 of
    the sibling's stacked leaf — value-exact by construction, and
    verified anyway.

    The point: it emits ``_terms``, so the diagram-state search can
    write the rewritten body back and let ``SplitLeaf`` candidacy
    fire on the *new* state — a composition single-application
    moves cannot reach.
    """

    name = "relink_leaf"

    def candidates(self, diagram: D.Diagram) -> list:
        """Fire only while ``blocks.0`` still reads its own leaf."""
        rec = diagram.record("blocks.0")
        if rec.ir is None or "p_s" in rec.leaves:
            return []
        return [
            MorphismMatch(
                law=self.name,
                nodes=("blocks.0",),
                boundary="relink",
                reify=ReifySpec(mode="intra", rules="compose"),
            )
        ]

    def reify(self, match, graph, *, sink, cost_fn, **kw):
        """Rewrite the body to ``linear(x, select(p_s, 0))``; verify."""
        rec = graph.record("blocks.0")
        s_val = next(
            v
            for v in graph.record("blocks.1").leaves.values()
            if len(getattr(v, "shape", ())) == 3
        )
        p = Param("p_s", TensorType(tuple(int(d) for d in s_val.shape)))
        site = Op.make("select", p, dim=0, index=0)
        x = rec.ir.inputs[0]
        root = Op.make("linear", x, site)
        params = {"p_s": p}
        leaves = {"p_s": s_val}
        opt = M._lower_term(
            root, x, params, leaves, sink, tuple(rec.ir.inputs)
        )
        vr = sink.verify(rec.module, opt, rec.args, rtol=1e-9)
        if not vr.passed:
            return {"status": "declined", "reason": "relink verify"}
        return {
            "status": "grafted",
            "reps": {"blocks.0": opt},
            "_terms": {"blocks.0": (root, params, leaves)},
            "cost_before": dag_cost(rec.ir.root, cost_fn),
            "cost_after": dag_cost(root, cost_fn),
            "rel_diff": vr.max_rel,
        }


class _FlatMove:
    """Stub move: graft a fixed node set with re-lowered bodies.

    Emits no ``_terms`` — the graft cannot be re-represented as
    terms, so the search seals the node (honest "body not
    representable").  ``cost_after`` arms the sealed-cost heuristic.
    The lowered rep computes the node's original body, so delivery is
    trivially exact.
    """

    def __init__(
        self,
        name: str,
        nodes: tuple,
        *,
        cost_after: float | None = None,
    ) -> None:
        """Store the fixed candidate and the optional cost claim."""
        self.name = name
        self.nodes = tuple(nodes)
        self.cost_after = cost_after

    def candidates(self, diagram: D.Diagram) -> list:
        """Emit the fixed match."""
        return [
            MorphismMatch(
                law=self.name,
                nodes=self.nodes,
                boundary="flat",
                reify=ReifySpec(mode="intra", rules="compose"),
            )
        ]

    def reify(self, match, graph, *, sink, **kw):
        """Re-lower each node's current body; no ``_terms``."""
        reps = {}
        for n in match.nodes:
            rec = graph.record(n)
            reps[n] = M._lower_term(
                rec.ir.root,
                rec.ir.inputs[0],
                rec.ir.params,
                rec.leaves,
                sink,
                tuple(rec.ir.inputs),
            )
        res = {"status": "grafted", "reps": reps}
        if self.cost_after is not None:
            res["cost_after"] = self.cost_after
        return res


class _DeclineMove:
    """Stub move: always emits its fixed candidate, always declines."""

    name = "decliner"

    def __init__(self, nodes: tuple = ("blocks.0", "blocks.1")) -> None:
        """Store the fixed match."""
        self.nodes = tuple(nodes)

    def candidates(self, diagram: D.Diagram) -> list:
        """Emit the fixed match."""
        return [
            MorphismMatch(
                law=self.name,
                nodes=self.nodes,
                boundary="x",
                reify=ReifySpec(mode="intra", rules="compose"),
            )
        ]

    def reify(self, *a, **k):
        """Decline honestly."""
        return {"status": "declined", "reason": "nope"}


class _BoomMove:
    """Stub move whose reify raises — the ``error`` decline path."""

    name = "boom"

    def candidates(self, diagram: D.Diagram) -> list:
        """Emit one fixed match."""
        return [
            MorphismMatch(
                law=self.name,
                nodes=("blocks.0",),
                boundary="x",
                reify=ReifySpec(mode="intra", rules="compose"),
            )
        ]

    def reify(self, *a, **k):
        """Raise."""
        raise RuntimeError("bang")


class _GhostMove:
    """Stub move: graft a rep for a node name that is not a block.

    Exercises the ``_advance`` guard for rep keys outside the record
    table — the rep is still delivered (graft honours the composer).
    """

    name = "ghost"

    def candidates(self, diagram: D.Diagram) -> list:
        """Emit a match on the ghost name."""
        return [
            MorphismMatch(
                law=self.name,
                nodes=("no.such.block",),
                boundary="ghost",
                reify=ReifySpec(mode="intra", rules="compose"),
            )
        ]

    def reify(self, *a, **k):
        """Graft a ghost rep — unit-test material only."""
        return {
            "status": "grafted",
            "reps": {"no.such.block": object()},
        }


def _opt(model, strat, x=None):
    """Run the contraction pipeline under the torch backend."""
    torch.manual_seed(0)
    return Optimizer(backend=TorchBackend()).optimize(
        model.eval().double(), _x() if x is None else x, strategy=strat
    )


# ---------------------------------------------------------------------------
#  The proving fixture — ordering composes what single moves cannot
# ---------------------------------------------------------------------------


def test_search_split_then_merge_beats_greedy():
    """``split_leaf`` then ``merge_projs`` on ``_Stacked``.

    Single-application reality: ``merge_projs`` grafts the
    select-spelled weights through stacked-tile retiling; the
    materialised slices pair *better* — the fused weight is a free
    concat over plain ``Param``s.  Greedy claims widest-first by
    law name and ``merge_projs`` wins the family before ``split_leaf``
    is tried, so the composed state is unreachable; search explores
    both orderings and lands on the cheaper composition.
    """
    model_g = _Stacked().eval().double()
    _m_g, stats_g = _opt(
        model_g, ContractionSearch(optimize_rest=False)
    )
    model_s = _Stacked().eval().double()
    mod_s, stats_s = _opt(
        model_s,
        ContractionSearch(mode="search", optimize_rest=False),
    )
    g, s = stats_g["diagram_search"], stats_s["diagram_search"]
    # Greedy: merge claimed the family; split never ran.
    assert stats_g["move_fires"] == {"merge_projs": 1}
    # Search: the composed path was found and is strictly cheaper.
    assert stats_s["move_fires"] == {
        "split_leaf": 1,
        "merge_projs": 1,
    }
    assert s["best_path"] == [
        "split_leaf:blocks.0+blocks.1",
        "merge_projs:blocks.0+blocks.1",
    ]
    assert s["best_cost"] < g["final_cost"]
    assert s["best_cost"] < s["initial_cost"]
    assert s["states_generated"] >= 3
    assert s["states_explored"] >= 3
    # Delivered model is the original, fp64-exact.
    assert stats_s["end_to_end"]["max_rel_diff"] < 1e-12
    with torch.no_grad():
        d = (model_s(_x()) - mod_s(_x())).abs().max().item()
    assert d < 1e-12


def test_search_enabling_composition():
    """A rewritten body creates a candidate no initial pass can see.

    ``_SplitStack``: ``blocks.0`` reads its own weight plainly, so the
    initial leaf groups give ``SplitLeaf`` one member and one site —
    no candidate.  ``_RelinkLeaf`` rewrites ``blocks.0``'s body to
    read ``select(S, 0)`` off the sibling's stacked leaf (verified
    exact); on the new state the leaf group is 2-member and
    ``split_leaf`` fires.  Greedy enumerates candidacy once, on the
    original diagram — it can never see the second move.
    """
    moves = (_RelinkLeaf(), D.SplitLeaf())
    model_g = _SplitStack().eval().double()
    _m_g, stats_g = _opt(
        model_g,
        ContractionSearch(moves=moves, optimize_rest=False),
    )
    model_s = _SplitStack().eval().double()
    mod_s, stats_s = _opt(
        model_s,
        ContractionSearch(
            moves=moves, mode="search", optimize_rest=False
        ),
    )
    g, s = stats_g["diagram_search"], stats_s["diagram_search"]
    # Greedy applied the relink and stopped — its candidate list was
    # computed before the rewrite could matter.
    assert stats_g["move_fires"] == {"relink_leaf": 1}
    # Search found the composition.
    assert stats_s["move_fires"] == {
        "relink_leaf": 1,
        "split_leaf": 1,
    }
    assert s["best_path"] == [
        "relink_leaf:blocks.0",
        "split_leaf:blocks.0+blocks.1",
    ]
    # The relink alone is a cost side-grade (adds a select); the
    # composition is a real win against the original bodies.
    assert g["final_cost"] > g["initial_cost"]
    assert s["best_cost"] < s["initial_cost"]
    assert stats_s["end_to_end"]["max_rel_diff"] < 1e-12
    with torch.no_grad():
        d = (model_s(_x()) - mod_s(_x())).abs().max().item()
    assert d < 1e-12


# ---------------------------------------------------------------------------
#  Search mechanics — parity, merging, sealing, caching, bounds
# ---------------------------------------------------------------------------


def test_search_parity_merge_only():
    """A plain family: search grafts the same move greedy does."""
    model = _Parallel().eval().double()
    mod, stats = _opt(
        model, ContractionSearch(mode="search", optimize_rest=False)
    )
    s = stats["diagram_search"]
    assert s["mode"] == "search"
    assert s["best_path"] == ["merge_projs:blocks.0+blocks.1"]
    assert stats["move_fires"] == {"merge_projs": 1}
    assert stats["end_to_end"]["max_rel_diff"] < 1e-12
    with torch.no_grad():
        d = (model(_x()) - mod(_x())).abs().max().item()
    assert d < 1e-12


def test_search_default_mode_is_greedy():
    """``ContractionSearch()`` stays greedy — stage-4 hardens search."""
    strat = ContractionSearch()
    assert strat.mode == "greedy"
    model = _Parallel().eval().double()
    _m, stats = _opt(model, strat.__class__(optimize_rest=False))
    s = stats["diagram_search"]
    assert s["mode"] == "greedy"
    assert s["moves_applied"] == 1
    assert s["final_cost"] < s["initial_cost"]


def test_search_merged_states_commute():
    """Two disjoint sealed grafts commute — the union merge fires.

    ``flat_a`` on ``blocks.0`` and ``flat_b`` on ``blocks.1`` reach the
    same body map in either order: the second ordering's successor
    keys collide in ``seen`` and count as merges; once a node is
    sealed the same move's re-emitted candidate is skipped by the
    prefilter.
    """
    moves = (
        _FlatMove("flat_a", ("blocks.0",), cost_after=100.0),
        _FlatMove("flat_b", ("blocks.1",), cost_after=100.0),
    )
    model = _Parallel().eval().double()
    _m, stats = _opt(
        model,
        ContractionSearch(
            moves=moves, mode="search", optimize_rest=False
        ),
    )
    s = stats["diagram_search"]
    assert s["states_merged"] >= 1
    assert s["moves_sealed"] >= 1
    assert stats["move_fires"] == {"flat_a": 1, "flat_b": 1}
    assert s["best_path"] == [
        "flat_a:blocks.0",
        "flat_b:blocks.1",
    ]
    assert stats["end_to_end"]["max_rel_diff"] < 1e-12


def test_search_sealed_cost_heuristic():
    """A term-less graft is priced by its ``cost_after`` claim.

    With ``cost_after=5`` on a two-node seal, the first rep carries
    the claim and the filler is booked at zero — the successor's cost
    is exactly 5 and it wins the search.
    """
    moves = (
        _FlatMove("flat_all", ("blocks.0", "blocks.1"), cost_after=5.0),
    )
    model = _Parallel().eval().double()
    _m, stats = _opt(
        model,
        ContractionSearch(
            moves=moves, mode="search", optimize_rest=False
        ),
    )
    s = stats["diagram_search"]
    assert s["best_cost"] == 5.0
    assert s["best_path"] == ["flat_all:blocks.0+blocks.1"]
    assert stats["end_to_end"]["max_rel_diff"] < 1e-12


def test_search_sealed_no_cost():
    """A term-less graft without ``cost_after`` keeps old costs."""
    moves = (_FlatMove("flat", ("blocks.0",)),)
    model = _Parallel().eval().double()
    _m, stats = _opt(
        model,
        ContractionSearch(
            moves=moves, mode="search", optimize_rest=False
        ),
    )
    s = stats["diagram_search"]
    # blocks.0 sealed with no price claim — contrib stands, so the
    # sealed state ties the seed on cost; the seed wins by first-found.
    assert s["best_cost"] == s["initial_cost"]
    assert stats["move_fires"] == {}


def test_search_cache_hit_and_error():
    """The reify cache: same candidate, unchanged members — hit.

    ``decliner`` on ``{blocks.0, blocks.1}`` runs at the seed and is
    re-emitted on the state where only ``blocks.2`` was sealed — its
    member records are untouched, so the cached decline is reused.
    ``boom``'s raising reify lands as an ``error`` decline.
    """
    moves = (
        _DeclineMove(),
        _BoomMove(),
        _FlatMove("flat", ("blocks.2",), cost_after=1.0),
    )
    model = _Chain(depth=3).eval().double()
    _m, stats = _opt(
        model,
        ContractionSearch(
            moves=moves, mode="search", optimize_rest=False
        ),
    )
    s = stats["diagram_search"]
    assert s["cache_hits"] >= 1
    assert s["moves_declined"] >= 3
    assert stats["move_fires"] == {"flat": 1}


def test_search_sealed_opaque_node():
    """A graft naming an opaque node seals it — its body was never
    representable (``ir`` was already ``None``)."""

    class _OpaqueGraft:
        name = "opaque_graft"

        def candidates(self, diagram):
            return [
                MorphismMatch(
                    law=self.name,
                    nodes=("blocks.1",),
                    boundary="opaque",
                    reify=ReifySpec(mode="intra", rules="compose"),
                )
            ]

        def reify(self, match, graph, **kw):
            rec = graph.record("blocks.1")
            return {
                "status": "grafted",
                "reps": {"blocks.1": rec.module},
            }

    model = _ResidOpaque().eval().double()
    _m, stats = _opt(
        model,
        ContractionSearch(
            moves=(_OpaqueGraft(),),
            mode="search",
            optimize_rest=False,
        ),
    )
    s = stats["diagram_search"]
    # The graft sealed blocks.1 — but its body map is unchanged
    # (ir was already None), so the successor merges into the seed
    # and delivers nothing.
    assert s["moves_tried"] == 1
    assert s["states_merged"] == 1
    assert stats["move_fires"] == {}
    assert stats["end_to_end"]["max_rel_diff"] < 1e-10


def test_search_depth_bound():
    """``search_depth=1`` — only single-move states exist; merge wins."""
    model = _Stacked().eval().double()
    _m, stats = _opt(
        model,
        ContractionSearch(
            mode="search", optimize_rest=False, search_depth=1
        ),
    )
    s = stats["diagram_search"]
    assert s["bound_depth"] == 1
    assert len(s["best_path"]) == 1
    assert s["best_path"] == ["merge_projs:blocks.0+blocks.1"]
    assert stats["end_to_end"]["max_rel_diff"] < 1e-12


def test_search_states_bound():
    """``search_states=0`` — no expansion at all; the seed stands."""
    model = _Stacked().eval().double()
    mod, stats = _opt(
        model,
        ContractionSearch(
            mode="search", optimize_rest=False, search_states=0
        ),
    )
    s = stats["diagram_search"]
    assert s["bound_hit"] is True
    assert s["states_explored"] == 0
    assert s["moves_applied"] == 0
    assert stats["move_fires"] == {}
    with torch.no_grad():
        d = (model(_x()) - mod(_x())).abs().max().item()
    assert d < 1e-12


def test_search_optimize_rest_fallback():
    """Unconsumed blocks still get the per-block fallback in search."""

    class _ParTail(nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [_Lin(dim, 400), _Lin(dim, 401), _Lin(dim, 402)]
            )

        def forward(self, x):
            return self.blocks[2](self.blocks[0](x) + self.blocks[1](x))

    model = _ParTail().eval().double()
    _m, stats = _opt(model, ContractionSearch(mode="search"))
    assert stats["blocks"]["blocks.0"]["status"] == "rewritten"
    assert stats["blocks"]["blocks.1"]["status"] == "rewritten"
    assert stats["blocks"]["blocks.2"]["status"] == "optimized"
    assert stats["end_to_end"]["max_rel_diff"] < 1e-10


def test_search_bad_mode():
    """An unknown mode is a constructor-time ``ValueError``."""
    import pytest

    with pytest.raises(ValueError, match="greedy"):
        ContractionSearch(mode="bogus")


def test_search_ghost_rep_unit():
    """``_advance`` guards a rep key outside the record table."""
    dg = _lift(_Parallel())
    st = DS._initial_state(dg, launch_aware_cost)
    succ = DS._advance(
        st,
        MorphismMatch(
            law="ghost",
            nodes=("no.such.block",),
            boundary="g",
            reify=ReifySpec(mode="intra", rules="compose"),
        ),
        {"status": "grafted", "reps": {"no.such.block": object()}},
        {"status": "grafted"},
        cost_fn=launch_aware_cost,
    )
    assert "no.such.block" in succ.reps
    assert succ.cost == st.cost
    # Through the driver the ghost's successor IS the seed state (the
    # body map never changed) — it merges immediately and delivers
    # nothing, honestly.
    model = _Parallel().eval().double()
    _m, stats = _opt(
        model,
        ContractionSearch(
            moves=(_GhostMove(),), mode="search", optimize_rest=False
        ),
    )
    assert stats["diagram_search"]["states_merged"] >= 1
    assert stats["move_fires"] == {}


def test_search_verbose(caplog):
    """Verbose mode logs the per-transition cost."""
    import logging as _logging

    torch.manual_seed(0)
    with caplog.at_level(
        _logging.INFO, "catopt_orchestrator.diagram_search"
    ):
        Optimizer(backend=TorchBackend()).optimize(
            _Parallel().eval().double(),
            _x(),
            strategy=ContractionSearch(
                mode="search", optimize_rest=False
            ),
            verbose=True,
        )
    recs = [r.getMessage() for r in caplog.records]
    assert any("[DiagramSearch]" in r for r in recs)
