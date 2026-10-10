"""Tests for ``catopt_discovery.meta_arena`` — the mixed-move board.

Plan 0021: one program under a live e-graph; the action space mixes
search moves (``fire`` / ``saturate`` — the measured schedule
dimension), construction moves (``declare`` — the quality lever that
inserts mid-search objects), and the terminal ``extract`` referee'd
by certificate replay.  These tests play small boards end to end,
pin the honest columns (``cost_unfolded`` and the supported-op
bound — the unpriced-fresh-name artifact), and pin the reward
composition from ``lawdata``.
"""

import math
import random
import sys

import pytest
from catopt_core.cost import (
    _INVALID_COST,
    backend_cost,
    count_cost,
    dag_cost,
    flops_cost,
    param_bytes_cost,
)
from catopt_core.egraph import CertificateVerificationError
from catopt_core.ir import Const, Op, TensorType, Var, op_repr
from catopt_core.laws import DEFAULT
from catopt_core.laws.ruleset import FULL
from catopt_discovery import lawdata
from catopt_discovery import meta_arena as ma
from catopt_discovery import object_synthesis as obs


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _sub_gap() -> Op:
    """``add(x, -y)`` — no shipped law reaches the ``sub`` spelling."""
    return _p("add", _v("x", 4, 4), _p("neg", _v("y", 4, 4)))


def _silu_site() -> Op:
    """``mul(x, sig(x))`` — the hand-spelled silu."""
    x = _v("x", 4, 4)
    return _p("mul", x, _p("sigmoid", x))


def _id_site() -> Op:
    """``mul(x, 1)`` — one ``id_mul`` fire away from its leaf."""
    return _p("mul", _v("x", 4, 4), Const(1))


def _decl(op: str, **params) -> ma.Action:
    """A ``declare`` move carrying a construction spec as data."""
    return ma.Action.declare({"op": op, "params": params})


def _scripted(board: ma.MetaArena, *extra) -> ma.Trajectory:
    """``extra…, saturate, extract`` — the scripted order."""
    return ma.run_episode(
        board,
        ma.ScriptedPlayer(
            [*extra, ma.Action.saturate(), ma.Action.extract()]
        ),
        10,
    )


class TestBoard:
    def test_baseline_and_state(self):
        term = _sub_gap()
        board = ma.MetaArena(term)
        assert board.baseline_cost == dag_cost(term, flops_cost)
        st = board.observe()
        assert st.baseline_cost == board.baseline_cost
        assert st.best_cost == board.baseline_cost
        assert st.rules == tuple(r.name for r in DEFAULT)
        assert st.budget_left == board.max_nodes - st.n_enodes
        assert not st.done and st.specs

    def test_legal_actions_shape(self):
        board = ma.MetaArena(_sub_gap())
        acts = ma.legal_actions(board.observe())
        n_rules = len(DEFAULT)
        fires = [a for a in acts if a.op == "fire"]
        sats = [a for a in acts if a.op == "saturate"]
        decls = [a for a in acts if a.op == "declare"]
        assert len(fires) == n_rules
        assert len(sats) == len(lawdata.META_SATURATE_BUDGETS)
        assert len(decls) == len(board.observe().specs)
        # fresh-name kernels only — the enumerator offers no claims
        assert all(
            a.params["construction"]["params"]["kernel"][0].startswith(
                "foldabs_"
            )
            for a in decls
        )
        assert acts[-1].op == "extract"

    def test_reward_table_is_data(self):
        assert set(lawdata.META_ARENA_REWARD) == {
            "step",
            "enode",
            "delta",
            "saturate_est",
        }
        w = lawdata.META_ARENA_REWARD
        board = ma.MetaArena(_id_site())
        _st, rep = board.step(ma.Action.fire("id_mul"))
        assert rep.reward == -(
            w["step"] + w["enode"] * rep.enodes_delta
        )


class TestMoves:
    def test_fire_changes_graph(self):
        board = ma.MetaArena(_id_site())
        st, rep = board.step(ma.Action.fire("id_mul"))
        assert rep.applied and not rep.terminal
        assert ("id_mul", 1) in st.rule_fires
        # mul(x, 1) ~ x: the root class now prices the bare leaf
        assert st.best_cost == 0.0

    def test_fire_unknown_and_cap_decline(self):
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(ma.Action.fire("no_such_rule"))
        assert not rep.applied and rep.reward == 0.0
        capped = ma.MetaArena(_sub_gap(), max_nodes=1)
        _st, rep2 = capped.step(ma.Action.fire("sub_to_add"))
        assert not rep2.applied and "budget" in rep2.note

    def test_saturate_subset_and_missing(self):
        board = ma.MetaArena(_id_site())
        st, rep = board.step(
            ma.Action.saturate(rules=("id_mul",), iterations=4)
        )
        assert rep.applied and rep.detail["stop"] in (
            "fixed_point",
            "max_iterations",
        )
        assert st.best_cost == 0.0
        _st, rep2 = board.step(
            ma.Action.saturate(rules=("id_mul", "nope"))
        )
        assert not rep2.applied

    def test_declare_inserts_fold_and_unfold(self):
        board = ma.MetaArena(_sub_gap())
        act = _decl(
            "fold",
            name="myabs",
            spelled=("add", "X", ("neg", "Y")),
            kernel=("myabs", "X", "Y"),
        )
        st, rep = board.step(act)
        assert rep.applied
        assert rep.inserted == ("myabs", "myabs_unfold")
        assert set(st.declared) == {"myabs", "myabs_unfold"}
        assert board.definitions["myabs"][0] == ("X", "Y")

    def test_declare_declines_and_conflicts(self):
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(_decl("bogus", x=1))
        assert not rep.applied and rep.reward == 0.0
        _st, rep = board.step(
            _decl("fold", spelled=("add", "X", ("neg", "Y")))
        )
        assert not rep.applied  # missing kernel — honest refusal
        # a name bound to a different rule conflicts, atomically
        st, rep = board.step(
            _decl(
                "fold",
                name="id_mul",
                spelled=("add", "X", ("neg", "Y")),
                kernel=("sub", "X", "Y"),
            )
        )
        assert not rep.applied and "bound to a different" in rep.note
        assert "id_mul_unfold" not in st.declared
        # re-declaring the same object is an idempotent no-conflict
        act = _decl(
            "fold",
            name="mine",
            spelled=("add", "X", ("neg", "Y")),
            kernel=("sub", "X", "Y"),
        )
        board.step(act)
        _st, rep = board.step(act)
        assert rep.applied and rep.inserted == ()

    def test_unregistered_op_is_keyerror(self):
        board = ma.MetaArena(_sub_gap())
        with pytest.raises(KeyError):
            board.step(ma.Action("teleport", {}))


class TestExtract:
    def test_scripted_certifies(self):
        board = ma.MetaArena(_silu_site(), cost_fn=count_cost)
        traj = _scripted(board)
        t = traj.terminal
        assert t is not None and t.certificate_ok
        assert t.cost <= board.baseline_cost
        # no declared ops in play — unfolded is the same program
        assert traj.cost_unfolded == traj.cost

    def test_terminal_and_done(self):
        board = ma.MetaArena(_sub_gap())
        traj = _scripted(board)
        assert traj.terminal is not None
        assert ma.legal_actions(board.observe()) == ()
        with pytest.raises(RuntimeError):
            board.step(ma.Action.fire("comm_add"))

    def test_cert_failure_scores_nothing(self, monkeypatch):
        def _boom(src, cert, **kw):
            raise CertificateVerificationError("forced")

        monkeypatch.setattr(ma, "verify_certificate", _boom)
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(ma.Action.extract())
        assert rep.terminal and rep.certificate_ok is False
        assert rep.reward == -lawdata.META_ARENA_REWARD["step"]

    def test_extract_reward_is_relative_delta(self):
        board = ma.MetaArena(_id_site())
        board.step(ma.Action.fire("id_mul"))
        _st, rep = board.step(ma.Action.extract())
        base = board.baseline_cost
        want = (
            lawdata.META_ARENA_REWARD["delta"]
            * ((base - rep.cost) / base)
            - lawdata.META_ARENA_REWARD["step"]
        )
        assert rep.reward == pytest.approx(want)


class TestDeclareQ2:
    """The Q2 mechanism: mid-search objects reach what laws cannot."""

    def test_declare_reaches_unreachable_spelling(self):
        board = ma.MetaArena(_sub_gap())
        # the search-only baseline: full DEFAULT cannot spell `sub`
        traj0 = _scripted(ma.MetaArena(_sub_gap()))
        assert traj0.cost == pytest.approx(32.0)
        # declare the fold, then search: sub(x, y) becomes reachable
        act = _decl(
            "fold",
            name="sub_fold",
            spelled=("add", "X", ("neg", "Y")),
            kernel=("sub", "X", "Y"),
        )
        traj = _scripted(board, act)
        t = traj.terminal
        assert t.certificate_ok
        assert t.used_declared == ("sub",)
        assert t.cost < board.baseline_cost
        assert "sub" in t.detail["extracted"]

    def test_fresh_name_fold_and_unfold_accounting(self):
        board = ma.MetaArena(_sub_gap())
        act = _decl(
            "fold",
            name="abs0",
            spelled=("add", "X", ("neg", "Y")),
            kernel=("abs0", "X", "Y"),
        )
        traj = _scripted(board, act)
        t = traj.terminal
        assert t.certificate_ok and t.used_declared == ("abs0",)
        # feasibility pricing: abs0 is declared-but-unlowered, so it
        # bills its spelled form — the extraction may still carry the
        # abbreviation (a sound definition), but it can no longer win
        # on the 1-flop default of a name nothing can lower
        assert t.cost == pytest.approx(board.baseline_cost)
        assert t.cost_unfolded == pytest.approx(t.cost)

    def test_unfold_roundtrip(self):
        board = ma.MetaArena(_sub_gap())
        board.step(
            _decl(
                "fold",
                name="abs0",
                spelled=("add", "X", ("neg", "Y")),
                kernel=("abs0", "X", "Y"),
            )
        )
        board.step(ma.Action.saturate())
        best = board.eg.extract_best(board.root, board.feasible_cost)
        assert op_repr(board.unfold(best)) == op_repr(board.term)


class TestFeasibilityPricing:
    """The supported-op bound: extraction can no longer mint names.

    ``feasible_cost`` = ``backend_cost(cost_fn, supported)`` over the
    declared-op expansion — a supported op keeps its model price, a
    declared-but-unlowered op bills its spelled form, and a name
    that is neither is infeasible (``+inf``).
    """

    def test_supported_bound_defaults_to_library_vocabulary(self):
        board = ma.MetaArena(_sub_gap())
        # every op the base ruleset spells is presumed lowerable —
        # ``sub`` is in DEFAULT's vocabulary (sub_to_add's lhs)
        assert "sub" in board.supported and "add" in board.supported
        assert board.feasible_cost(_sub_gap()) == pytest.approx(32.0)
        # a name no rule mentions is unsupported — and, undeclared,
        # infeasible outright (the finite never-win sentinel)
        x = _v("x", 4, 4)
        assert board.feasible_cost(_p("mystery", x)) >= _INVALID_COST

    def test_declared_fresh_name_prices_at_spelled_form(self):
        board = ma.MetaArena(_sub_gap())
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        fresh = _p("abs0", x, y)
        assert board.feasible_cost(fresh) >= _INVALID_COST
        board.step(
            _decl(
                "fold",
                name="abs0",
                spelled=("add", "X", ("neg", "Y")),
                kernel=("abs0", "X", "Y"),
            )
        )
        # the bound froze at board open — the minted name is not in it
        assert "abs0" not in board.supported
        spelled = _p("add", x, _p("neg", y))
        assert board.feasible_cost(fresh) == pytest.approx(
            board.feasible_cost(spelled)
        )
        # and the composition is literally backend_cost over the
        # expanded term (its +inf shows here as _INVALID_COST — the
        # finite never-win sentinel the report layer reads as inf)
        bound = backend_cost(flops_cost, board.supported)
        assert board.feasible_cost(spelled) == bound(spelled)
        mystery = _p("mystery", x)
        assert bound(mystery) == float("inf")
        assert board.feasible_cost(mystery) >= _INVALID_COST

    def test_unfoldless_declare_is_infeasible_not_cheap(self):
        # unfold=False mints no definition — the fresh name is an op
        # nothing can run, so it prices at the never-win sentinel,
        # not the 1-flop default
        board = ma.MetaArena(_sub_gap())
        board.step(
            ma.Action.declare(
                {
                    "op": "fold",
                    "params": {
                        "name": "k0",
                        "spelled": ("add", "X", ("neg", "Y")),
                        "kernel": ("k0", "X", "Y"),
                    },
                },
                unfold=False,
            )
        )
        assert board.definitions == {}
        traj = _scripted(board)
        t = traj.terminal
        assert t.certificate_ok
        assert t.used_declared == ()
        assert t.cost == pytest.approx(board.baseline_cost)

    def test_supported_bound_gates_the_declared_price(self):
        decl = _decl(
            "fold",
            name="subfold",
            spelled=("add", "X", ("neg", "Y")),
            kernel=("sub", "X", "Y"),
        )
        # a sink that knows ``sub``: the claim earns its kernel price
        board = ma.MetaArena(
            _sub_gap(), supported=frozenset({"add", "neg", "sub"})
        )
        traj = _scripted(board, decl)
        assert traj.terminal.certificate_ok
        assert traj.cost == pytest.approx(16.0)
        # the same object on a board that cannot lower ``sub``: the
        # extraction still certifies — at the spelled price
        board2 = ma.MetaArena(
            _sub_gap(), supported=frozenset({"add", "neg"})
        )
        traj2 = _scripted(board2, decl)
        t2 = traj2.terminal
        assert t2.certificate_ok
        assert t2.cost == pytest.approx(32.0)
        assert t2.cost_unfolded == pytest.approx(t2.cost)

    def test_artifact_pays_no_delta(self):
        board = ma.MetaArena(_sub_gap())
        act = _decl(
            "fold",
            name="abs0",
            spelled=("add", "X", ("neg", "Y")),
            kernel=("abs0", "X", "Y"),
        )
        traj = _scripted(board, act)
        t = traj.terminal
        # delta*(base-cost)/base - step: cost == baseline -> -step
        assert t.reward == pytest.approx(
            -lawdata.META_ARENA_REWARD["step"]
        )

    def test_greedy_finds_no_artifact_win(self):
        # under DEFAULT minus the shipped folds the only declares the
        # enumerator offers are fresh names — parity-priced now, so
        # cheapest-immediate play lands back on baseline
        sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
        board = ma.MetaArena(_silu_site(), sans)
        traj = ma.run_episode(board, ma.GreedyPlayer(), 200)
        t = traj.terminal
        assert t is not None and t.certificate_ok
        assert t.cost == pytest.approx(board.baseline_cost)

    def test_unrunnable_baseline_pays_no_delta(self):
        # a bound that excludes the program's own ops is an honest
        # "this board cannot run it" — not a nan payout
        board = ma.MetaArena(_sub_gap(), supported=frozenset())
        assert not math.isfinite(board.baseline_cost)
        assert board.observe().best_cost == float("inf")
        _st, rep = board.step(ma.Action.extract())
        assert rep.terminal and rep.certificate_ok
        assert rep.reward == pytest.approx(
            -lawdata.META_ARENA_REWARD["step"]
        )
        # and a zero-cost baseline (a bare leaf) divides by nothing
        leaf = ma.MetaArena(_v("x", 4, 4))
        assert leaf.baseline_cost == 0.0
        _st, rep = leaf.step(ma.Action.extract())
        assert rep.terminal and rep.reward == pytest.approx(
            -lawdata.META_ARENA_REWARD["step"]
        )

    def test_torch_supported_falls_back(self, monkeypatch):
        # without catopt-torch the demo bound is None — the board
        # falls back to the ambient-vocabulary bound
        monkeypatch.setitem(sys.modules, "catopt_torch.adapters", None)
        assert ma._torch_supported() is None

    def test_feasible_cost_forwards_model_markers(self):
        # a storage-pricing model's markers ride the wrapper so
        # extract_best / dag_cost keep billing params correctly
        board = ma.MetaArena(_sub_gap(), cost_fn=param_bytes_cost)
        assert board.feasible_cost.charges_param_only is True
        assert board.feasible_cost.dag_exact is True


class TestPlayers:
    def test_greedy_extracts_once_improved(self):
        rules = DEFAULT.named("silu_fold")
        board = ma.MetaArena(_silu_site(), rules, cost_fn=count_cost)
        traj = ma.run_episode(board, ma.GreedyPlayer(), 10)
        t = traj.terminal
        assert t is not None and t.certificate_ok
        assert t.cost == pytest.approx(1.0)
        assert len(traj.reports) == 2

    def test_random_player_seeded(self):
        def run() -> list:
            board = ma.MetaArena(_sub_gap())
            return [
                (r.action.op, r.note)
                for r in ma.run_episode(
                    board, ma.RandomPlayer(random.Random(7)), 6
                ).reports
            ]

        assert run() == run()

    def test_run_episode_budget(self):
        board = ma.MetaArena(_sub_gap())
        traj = ma.run_episode(
            board, ma.RandomPlayer(random.Random(0)), 3
        )
        assert len(traj.reports) <= 3


class TestConstructors:
    """The playbook-reachable construction ops over live refs."""

    def test_lift_and_compose_and_specialize(self):
        # FULL carries the symmetry rules (``comm_mul`` is not in
        # DEFAULT) — the premise vocabulary includes them.
        board = ma.MetaArena(_sub_gap(), FULL)
        _st, rep = board.step(
            _decl(
                "lift",
                name="aff_step",
                step=("add", ("matmul", "A", "h"), "x"),
                carrier=("aff", "A", "x"),
                apply_op="apply",
                state="h",
            )
        )
        assert rep.applied and rep.inserted == ("aff_step",)

        _st, rep = board.step(
            _decl(
                "compose",
                name="silu_swapped",
                first="comm_mul",
                rest=("silu_fold",),
                specialize={"a": ("sigmoid", "X"), "b": "X"},
            )
        )
        # the composite mul(sig(X),X) → silu(X) is itself a faithful
        # abbreviation, so the definitional unfold rides along
        assert rep.applied and rep.inserted == (
            "silu_swapped",
            "silu_swapped_unfold",
        )
        assert "comm_mul" in rep.detail["construction"]

        _st, rep = board.step(
            _decl(
                "specialize",
                ref="comm_add",
                binding={"a": ("neg", "b")},
            )
        )
        assert rep.applied

    def test_compose_unknown_premise_declines(self):
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(
            _decl("compose", first="comm_mul", rest=("nope",))
        )
        assert not rep.applied and rep.reward == 0.0

    def test_specialize_unknown_ref_declines(self):
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(
            _decl("specialize", ref="nope", binding={"a": 1})
        )
        assert not rep.applied

    def test_resolve_ref_passthroughs(self):
        board = ma.MetaArena(_sub_gap())
        assert board.resolve_ref(board.rules[0]) is board.rules[0]
        assert board.resolve_ref(3) is None
        # a ConstructedObject resolves to its rule
        obj = board.resolve_ref(
            obs.fold_object("n", ("add", "X", "Y"), "sub")
        )
        assert obj is not None and obj.name == "n"

    def test_unfold_not_minted_on_unfaithful_rhs(self):
        board = ma.MetaArena(_sub_gap())
        cases = [
            # metavar-dropping kernel — a claim, not a definition
            ("k1", ("k1", "X")),
            # duplicate args
            ("k2", ("k2", "X", "X")),
            # a non-metavar arg
            ("k3", ("k3", "X", 1)),
            # arity-zero kernel
            ("k4", ("k4",)),
            # a non-Op kernel outright
            ("k5", 5),
        ]
        for name, kernel in cases:
            _st, rep = board.step(
                _decl(
                    "fold",
                    name=name,
                    spelled=("add", "X", ("neg", "Y")),
                    kernel=kernel,
                )
            )
            assert rep.applied
            assert rep.inserted == (name,)
        assert board.definitions == {}

    def test_unnamed_declare_gets_digest_name(self):
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(
            _decl(
                "fold",
                spelled=("add", "X", ("neg", "Y")),
                kernel=("k", "X", "Y"),
            )
        )
        assert rep.applied and rep.inserted[0].startswith("fold:")


class TestRefereeEdges:
    def test_replay_mismatch_scores_nothing(self, monkeypatch):
        x = _v("z", 4, 4)
        monkeypatch.setattr(
            ma, "verify_certificate", lambda src, cert: x
        )
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(ma.Action.extract())
        assert rep.certificate_ok is False
        assert "replay mismatch" in rep.note

    def test_certification_exception_scores_nothing(self, monkeypatch):
        def _boom(src, cert, **kw):
            raise ValueError("forced")

        monkeypatch.setattr(ma, "verify_certificate", _boom)
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(ma.Action.extract())
        assert rep.certificate_ok is False
        assert "certification failed" in rep.note


class TestEnumerationAndDrivers:
    def test_specs_cover_attrs_and_consts(self):
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        term = _p(
            "add", _p("unsqueeze", x, dim=0), _p("mul", y, Const(1))
        )
        st = ma.MetaArena(term).observe()
        # the whole composite appears, canonicalized with attr data
        whole = (
            "add",
            ("unsqueeze", "X1", {"dim": 0}),
            ("mul", "X2", 1),
        )
        assert whole in st.specs
        assert ma._spec_metavars(whole) == ("X1", "X2")
        assert ma._spec_ops("bare") == set()

    def test_declare_candidates_with_kernels(self):
        board = ma.MetaArena(_sub_gap())
        cands = ma.declare_candidates(board, extra_kernels=("sub",))
        kinds = [
            c.params["construction"]["params"]["kernel"][0]
            for c in cands
        ]
        assert "sub" in kinds
        assert any(k.startswith("foldabs_") for k in kinds)

    def test_players_none_and_empty_legal(self):
        board = ma.MetaArena(_sub_gap())
        # a drained scripted player ends the episode at step 0
        traj = ma.run_episode(board, ma.ScriptedPlayer([]), 5)
        assert traj.reports == []
        assert traj.terminal is None and traj.cost is None
        assert traj.certificate_ok is None and traj.declared_used == ()
        # an empty legal set yields None — the episode never starts
        traj = ma.run_episode(
            board,
            ma.RandomPlayer(random.Random(0), legal=lambda s: ()),
            5,
        )
        assert traj.reports == []

    def test_arm_row_without_terminal(self):
        row = ma._arm_row(ma.Trajectory([]), 10.0)
        assert row["cost"] is None and row["cert"] is None
        assert row["used_declared"] == ()

    def test_main_runs_the_demo_probe(self, capsys):
        assert ma.main([]) == 0
        out = capsys.readouterr().out
        assert "sub_gap" in out and "declare" in out

    def test_construction_parts_duck_typing(self):
        import types

        board = ma.MetaArena(_sub_gap())
        construction = types.SimpleNamespace(
            op="fold",
            params={
                "name": "ns_fold",
                "spelled": ("add", "X", ("neg", "Y")),
                "kernel": ("sub", "X", "Y"),
            },
        )
        _st, rep = board.step(ma.Action.declare(construction))
        assert rep.applied and rep.inserted[0] == "ns_fold"

    def test_compose_none_declines(self):
        board = ma.MetaArena(_sub_gap(), FULL)
        # silu_fold's lhs matches nowhere inside comm_add's rhs —
        # the honest ``None`` decline path of the constructor
        _st, rep = board.step(
            _decl("compose", first="comm_add", rest=("silu_fold",))
        )
        assert not rep.applied and "constructor refused" in rep.note

    def test_declare_without_unfold_pair(self):
        board = ma.MetaArena(_sub_gap())
        _st, rep = board.step(
            ma.Action.declare(
                {
                    "op": "fold",
                    "params": {
                        "name": "nopair",
                        "spelled": ("add", "X", ("neg", "Y")),
                        "kernel": ("sub", "X", "Y"),
                    },
                },
                unfold=False,
            )
        )
        assert rep.applied and rep.inserted == ("nopair",)
        assert board.definitions == {}

    def test_greedy_none_when_exhausted(self):
        board = ma.MetaArena(_sub_gap())
        player = ma.GreedyPlayer(legal=lambda s: ())
        assert player(board.observe()) is None

    def test_spec_helpers_edge(self):
        assert list(ma._spec_children("x")) == []
        assert list(ma._spec_children(())) == []


class TestProbe:
    def test_meta_probe_rows(self):
        x, y = _v("x", 4, 4), _v("y", 4, 4)
        cases = [
            (
                "sub_gap",
                _p("add", x, _p("neg", y)),
                DEFAULT,
                (
                    {
                        "op": "fold",
                        "params": {
                            "name": "sub_decl",
                            "spelled": ("add", "X", ("neg", "Y")),
                            "kernel": ("sub", "X", "Y"),
                        },
                    },
                ),
            ),
            ("control", _p("add", _p("mul", x, y), x)),
        ]
        rows = ma.meta_probe(cases, budget=400, declare_limit=8)
        assert [r["name"] for r in rows] == ["sub_gap", "control"]
        sub = rows[0]
        # declare-then-search beats every search-only arm
        assert sub["declare"]["cost"] < sub["arms"]["scripted"]["cost"]
        assert sub["declare"]["used_declared"] == ("sub",)
        # every certified extraction replays (a random episode that
        # never extracted reports cert=None — not a failure)
        assert sub["declare"]["cert"]
        assert all(
            v["cert"]
            for v in sub["arms"].values()
            if v["cert"] is not None
        )
        assert "sub_gap" in ma.probe_table(rows)

    def test_meta_probe_supported_bound(self):
        # the same kernel claim prices honestly both ways: unnamed
        # under the ambient vocabulary it earns its spelled form;
        # under a bound that knows it, its supported price
        x = _v("x", 4, 4)
        site = _p("div", x, _p("add", _p("abs", x), Const(1)))
        sans = DEFAULT - DEFAULT.named("silu_fold", "softsign_fold")
        decl = {
            "op": "fold",
            "params": {
                "name": "softsign_decl",
                "spelled": ("div", "X", ("add", ("abs", "X"), 1)),
                "kernel": "softsign",
            },
        }
        rows = ma.meta_probe(
            [("ss", site, sans, (decl,))], budget=8, declare_limit=4
        )
        assert rows[0]["declare"]["cost"] == pytest.approx(
            rows[0]["baseline"]
        )
        sup = frozenset({"div", "add", "abs", "softsign"})
        rows = ma.meta_probe(
            [("ss", site, sans, (decl,), sup)],
            budget=8,
            declare_limit=4,
        )
        assert rows[0]["declare"]["cost"] == pytest.approx(16.0)


class TestHandleMove:
    """``handle`` — the scoped-interpretation move (Sanada-style)."""

    def _arena(self, cost_fn=None):
        from catopt_core.cost.basic import count_cost

        x = _v("x", 4, 4)
        c = _v("c", 4, 4)
        term = _p("add", _p("mul", x, _p("sigmoid", x)), c)
        sup = {"add", "mul", "sigmoid", "silu"}
        arena = ma.MetaArena(
            term,
            rules=[],
            supported=sup,
            cost_fn=cost_fn or count_cost,
            max_specs=8,
        )
        arena.step(
            ma.Action.declare(
                {
                    "op": "fold",
                    "params": {
                        "name": "mysilu",
                        "spelled": ("mul", "X1", ("sigmoid", "X1")),
                        "kernel": ("mysilu", "X1"),
                    },
                }
            )
        )
        arena.step(ma.Action.saturate(budget=512))
        return arena

    def test_handle_reprices_to_kernel(self):
        # handled: the declared name reads as one `silu` op (2 ops
        # total incl. add); unhandled: spelled parity (3 ops)
        arena = self._arena()
        _, rep = arena.step(ma.Action.handle("mysilu", "silu"))
        assert rep.applied
        assert rep.detail["kernel"] == "silu"
        st = arena.observe()
        assert st.handles == (("mysilu", "silu"),)
        _, rep = arena.step(ma.Action.extract())
        assert rep.applied and rep.certificate_ok
        assert rep.cost == pytest.approx(2.0)
        # cost_unfolded stays the honest spelled price
        assert rep.cost_unfolded == pytest.approx(3.0)

    def test_unhandled_prices_spelled_parity(self):
        arena = self._arena()
        _, rep = arena.step(ma.Action.extract())
        assert rep.cost == pytest.approx(3.0)
        assert rep.cost_unfolded == pytest.approx(3.0)

    def test_handle_requires_declaration(self):
        arena = self._arena()
        _, rep = arena.step(ma.Action.handle("nope", "silu"))
        assert not rep.applied
        assert "not a declared definition" in rep.note

    def test_handle_requires_known_handler(self):
        arena = self._arena()
        _, rep = arena.step(ma.Action.handle("mysilu", "bogus"))
        assert not rep.applied
        assert "unknown handler" in rep.note

    def test_handle_requires_covering_pattern(self):
        # `square` covers mul(X,X) — mysilu's body has distinct arg
        # positions; the mismatch declines honestly
        arena = self._arena()
        _, rep = arena.step(ma.Action.handle("mysilu", "square"))
        assert not rep.applied
        assert "does not cover" in rep.note

    def test_handle_unsupported_kernel_is_infeasible(self):
        # a handler whose kernel the bound cannot run applies (the
        # interpretation is real) but prices the member infeasible
        x = _v("x", 4, 4)
        c = _v("c", 4, 4)
        term = _p("add", _p("mul", x, _p("sigmoid", x)), c)
        arena = ma.MetaArena(
            term,
            rules=[],
            supported={"add", "mul", "sigmoid"},  # no silu
            cost_fn=None,
            max_specs=8,
        )
        arena.step(
            ma.Action.declare(
                {
                    "op": "fold",
                    "params": {
                        "name": "mysilu",
                        "spelled": ("mul", "X1", ("sigmoid", "X1")),
                        "kernel": ("mysilu", "X1"),
                    },
                }
            )
        )
        arena.step(ma.Action.saturate(budget=512))
        _, rep = arena.step(ma.Action.handle("mysilu", "silu"))
        assert rep.applied
        _, rep = arena.step(ma.Action.extract())
        # extraction falls back to the spelled member — the fused
        # member is infeasible under this bound
        assert rep.applied
        assert rep.cost == pytest.approx(rep.cost_unfolded)

    def test_legal_actions_enumerates_covering_handlers(self):
        arena = self._arena()
        st = arena.observe()
        handles = [a for a in ma.legal_actions(st) if a.op == "handle"]
        assert handles == [ma.Action.handle("mysilu", "silu")]
        # already-handled ops are not re-offered
        arena.step(ma.Action.handle("mysilu", "silu"))
        st = arena.observe()
        assert not any(a.op == "handle" for a in ma.legal_actions(st))

    def test_handle_reassign_replaces_interpretation(self):
        # re-handling the same op swaps its priced kernel
        arena = self._arena()
        arena.step(ma.Action.handle("mysilu", "silu"))
        assert arena.interpretations["mysilu"][0] == "silu"

    def test_declare_probe_reports_handled_cost(self):
        # the handled column prices the interpretation: a silu
        # fold over a silu-shaped site under count_cost reads 3
        # spelled / 2 handled — the fused premium as a column
        from catopt_core.cost.basic import count_cost

        x, c = _v("x", 4, 4), _v("c", 4, 4)
        term = _p("add", _p("mul", x, _p("sigmoid", x)), c)
        sans = DEFAULT - DEFAULT.named(
            "silu_fold", "softsign_fold", "swiglu_fold"
        )
        sup = frozenset({"add", "mul", "sigmoid", "silu"})
        rows = ma.meta_probe(
            [("silu_site", term, sans, (), sup)],
            cost_fn=count_cost,
            budget=8,
            declare_limit=12,
        )
        d = rows[0]["declare"]
        assert d is not None and d["cert"]
        assert d["handled"] is not None
        assert d["handled"] < d["cost"]


class TestClaimMove:
    """``claim`` — the option move: declare + fire + handle, fused."""

    def _arena(self):
        from catopt_core.cost.basic import count_cost

        x = _v("x", 4, 4)
        term = _p("mul", x, _p("sigmoid", x))
        return ma.MetaArena(
            term,
            rules=[],
            supported={"mul", "sigmoid", "silu"},
            cost_fn=count_cost,
            max_specs=8,
        )

    def test_claim_pays_in_one_move(self):
        # the declare→saturate→handle line as one action: claim
        # lands the handled member; extract prices it at the kernel
        arena = self._arena()
        st = arena.observe()
        claims = [a for a in ma.legal_actions(st) if a.op == "claim"]
        assert [a.params["handler"] for a in claims] == ["silu"]
        st, rep = arena.step(claims[0])
        assert rep.applied
        assert rep.detail["object"] == "claim_silu_0"
        _, rep = arena.step(ma.Action.extract())
        assert rep.applied and rep.certificate_ok
        assert rep.cost == pytest.approx(1.0)
        assert rep.cost_unfolded == pytest.approx(2.0)

    def test_claim_reoffers_until_covered_specs_claimed(self):
        # one covered spec → after claiming, no further claim offers
        arena = self._arena()
        arena.step(ma.Action.claim("silu"))
        st = arena.observe()
        assert not [a for a in ma.legal_actions(st) if a.op == "claim"]

    def test_claim_unknown_handler_declines(self):
        arena = self._arena()
        _, rep = arena.step(ma.Action.claim("bogus"))
        assert not rep.applied
        assert "unknown handler" in rep.note

    def test_claim_no_covered_spec_declines(self):
        from catopt_core.cost.basic import count_cost

        x = _v("x", 4, 4)
        arena = ma.MetaArena(
            _p("add", x, x),
            rules=[],
            supported={"add", "silu"},
            cost_fn=count_cost,
            max_specs=8,
        )
        _, rep = arena.step(ma.Action.claim("silu"))
        assert not rep.applied
        assert "no uncovered spec" in rep.note

    def test_claim_unsupported_kernel_stays_parity(self):
        # claim mints and binds even when the kernel is unsupported —
        # feasibility pricing then keeps the spelled form honest
        from catopt_core.cost.basic import count_cost

        x = _v("x", 4, 4)
        arena = ma.MetaArena(
            _p("mul", x, _p("sigmoid", x)),
            rules=[],
            supported={"mul", "sigmoid"},  # silu NOT supported
            cost_fn=count_cost,
            max_specs=8,
        )
        st = arena.observe()
        claims = [a for a in ma.legal_actions(st) if a.op == "claim"]
        assert claims  # legality is structural, not support-based
        arena.step(claims[0])
        _, rep = arena.step(ma.Action.extract())
        assert rep.applied and rep.certificate_ok
        assert rep.cost == pytest.approx(2.0)  # spelled parity

    def test_claim_fire_skipped_when_budget_spent(self):
        # enode budget exhausted before claim: the fused move still
        # declares and binds — firing is conditional, the outcome
        # honestly reports the claim
        arena = self._arena()
        arena.max_nodes = 0
        st = arena.observe()
        claims = [a for a in ma.legal_actions(st) if a.op == "claim"]
        _, rep = arena.step(claims[0])
        assert rep.applied
        assert rep.detail["object"] == "claim_silu_0"

    def test_claim_declines_when_handle_cannot_bind(self):
        # a handler whose args index metavars the spec does not have:
        # declare lands, the bind refuses, claim reports declined
        arena = self._arena()
        arena.handlers["wide"] = {
            "pattern": ("mul", "X1", ("sigmoid", "X1")),
            "args": ("X1", "X2", "X3"),
            "kernel": "silu",
        }
        _, rep = arena.step(ma.Action.claim("wide"))
        assert not rep.applied
        assert "handle declined" in rep.note

    def test_claim_declines_when_declare_refuses(self, monkeypatch):
        # the fold constructor refuses → claim relays the decline
        monkeypatch.setitem(
            ma._CONSTRUCTION, "fold", lambda arena, name, cp: None
        )
        arena = self._arena()
        _, rep = arena.step(ma.Action.claim("silu"))
        assert not rep.applied
        assert "declare declined" in rep.note

    def test_deliverable_realizes_handled_ops(self):
        # claim_silu_0 → silu (the kernel the sink lowers), while
        # unfold spells it back — the two honest views of one win
        from catopt_core.ir import op_repr

        arena = self._arena()
        arena.step(ma.Action.claim("silu"))
        arena.step(ma.Action.extract())
        best = arena.eg.extract_best(arena.root, arena.feasible_cost)
        assert op_repr(arena.deliverable(best)) == "(silu x)"
        assert op_repr(arena.unfold(best)) == "(mul x, (sigmoid x))"

    def test_deliverable_spells_unhandled(self):
        # declared but never handled: deliverable expands to the
        # spelled body — no fresh name reaches lowering
        from catopt_core.ir import op_repr

        arena = self._arena()
        arena.step(
            ma.Action.declare(
                {
                    "op": "fold",
                    "params": {
                        "name": "mysilu",
                        "spelled": ("mul", "X1", ("sigmoid", "X1")),
                        "kernel": ("mysilu", "X1"),
                    },
                }
            )
        )
        arena.step(ma.Action.saturate(budget=512))
        arena.step(ma.Action.extract())
        best = arena.eg.extract_best(arena.root, arena.feasible_cost)
        assert "mysilu" not in op_repr(arena.deliverable(best))
