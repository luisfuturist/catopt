"""The law-order board probe (:mod:`catopt_discovery.law_order`).

Plan 0021 depth question 1 — measured in
``project/retros/law-order-board.md``.  The pins here are the probe's
safety/determinism contract, not the corpus numbers:

* **deterministic seeding** — the same ordering arm replays to the
  same observables;
* **safety** — the extracted term's certificate verifies under every
  ordering (a policy can only reorder, never change what is proved);
* **the measured invariant** — on terms where the laws fire, the
  extracted cost/spelling/frontier is order-invariant while the
  materialized e-graph contents are not.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from catopt_core.game import Action
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import DEFAULT, tags
from catopt_core.policies import GreedyPolicy, RandomPolicy
from catopt_discovery import law_order as lo


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _chain():
    """``a @ (b @ c)`` — ``assoc_matmul`` has a firing site."""
    a, b, c = _v("a", 2, 3), _v("b", 3, 4), _v("c", 4, 5)
    return Op.make("matmul", a, Op.make("matmul", b, c))


def _swiglu():
    """``silu(a@b) * (a@c)`` — the expand/fold critical pair.

    Order-sensitive *contents* (see the retro): expand-first makes
    ``silu_mul_form``'s instantiated RHS collapse onto the existing
    outer enode through the quotient; mul-form-first materializes a
    distinct enode that ``silu_fold`` later merges.
    """
    a, b, c = _v("a", 4, 8), _v("b", 8, 4), _v("c", 8, 4)
    return Op.make(
        "mul",
        Op.make("silu", Op.make("matmul", a, b)),
        Op.make("matmul", a, c),
    )


def _dead_end():
    """A lone unary op no rule matches — the null board."""
    return Op.make("relu", _v("x", 4, 8))


_FIRE_TERMS = [_chain(), _swiglu()]


def _nonwall(run: lo.BoardRun) -> lo.BoardRun:
    """Every :class:`BoardRun` field except wall-clock."""
    return replace(run, wall_s=0.0)


class TestOrderArms:
    def test_arm_set(self):
        arms = lo.order_arms(DEFAULT, seeds=(3,))
        names = [a for a, _ in arms]
        assert names == [
            "declared",
            "reversed",
            "random:3",
            "expansive_last",
            "expansive_first",
        ]
        assert arms[0][1] is None
        assert isinstance(arms[1][1], GreedyPolicy)
        assert isinstance(arms[2][1], RandomPolicy)

    def test_reversed_actually_reorders(self):
        """The reversed arm picks the last-declared rule first."""
        arms = lo.order_arms(DEFAULT)
        state = SimpleNamespace(root_eid=0)
        first = DEFAULT.rules[0].name
        last = DEFAULT.rules[-1].name
        offered = [Action(r.name, 0) for r in DEFAULT.rules]
        assert arms[0][1] is None  # declared = engine order
        assert arms[1][1].choose(state, offered).rule == last
        # and never the declared-first rule when alternatives exist
        assert arms[1][1].choose(state, offered).rule != first

    def test_production_budgets(self):
        bud = lo.production_budgets(DEFAULT, 64)
        expansive = {r.name for r in DEFAULT.tagged(tags.EXPANSIVE)}
        assert bud == {name: 64 for name in expansive}
        assert lo.production_budgets(DEFAULT, None) is None


class TestBoardRun:
    def test_run_board_records_observables(self):
        run = lo.run_board(
            _chain(), DEFAULT, max_iterations=20, max_nodes=10_000
        )
        assert run.arm == "declared"
        assert run.stop == "fixed_point"
        assert run.n_proof_edges >= 1  # assoc_matmul fired
        assert run.cost == 128.0  # (a@b)@c reassociation wins
        assert run.frontier
        assert run.certificate_ok

    def test_deterministic_seeded(self):
        """Same seeded arms replayed give identical observables.

        ``arms`` are rebuilt per probe: ``RandomPolicy`` carries its
        RNG stream, so a reused arm would sample *continuations* of
        one stream, not replays.
        """
        kw = dict(
            rule_budgets=lo.production_budgets(DEFAULT, 256),
            max_nodes=20_000,
        )
        first = lo.probe_term(
            _swiglu(),
            DEFAULT,
            arms=lo.order_arms(DEFAULT, seeds=(0, 1)),
            **kw,
        )
        second = lo.probe_term(
            _swiglu(),
            DEFAULT,
            arms=lo.order_arms(DEFAULT, seeds=(0, 1)),
            **kw,
        )
        assert [_nonwall(r) for r in first] == [
            _nonwall(r) for r in second
        ]

    def test_null_board_is_order_invariant(self):
        runs = lo.probe_term(_dead_end(), DEFAULT)
        v = lo.compare_runs(runs)
        assert v.contents_equal and v.answer_equal
        assert v.frontier_equal and v.partition_equal

    def test_verify_off_leaves_certificate_unchecked(self):
        run = lo.run_board(
            _chain(), DEFAULT, verify=False, max_iterations=5
        )
        assert not run.certificate_ok
        v = lo.compare_runs((run,))
        assert not v.certificates_ok

    def test_failed_replay_marks_the_arm(self, monkeypatch):
        from catopt_core.egraph import CertificateVerificationError

        def boom(*a, **k):
            raise CertificateVerificationError("synthetic")

        monkeypatch.setattr(lo, "verify_certificate", boom)
        run = lo.run_board(_chain(), DEFAULT, max_iterations=5)
        assert not run.certificate_ok


class TestSafety:
    """The safety property: certificates verify regardless of order."""

    @pytest.mark.parametrize("term", _FIRE_TERMS)
    def test_certificates_verify_under_every_order(self, term):
        arms = lo.order_arms(DEFAULT, seeds=(0, 7, 11))
        runs = lo.probe_term(
            term,
            DEFAULT,
            arms=arms,
            rule_budgets=lo.production_budgets(DEFAULT, 256),
        )
        assert all(r.certificate_ok for r in runs)
        assert lo.compare_runs(runs).certificates_ok


class TestVerdict:
    def test_contents_differ_but_answer_is_invariant(self):
        """The measured split on the swiglu critical pair.

        Ordering moves the materialized enode set (expand-first
        collapses mul-form's RHS through the quotient) but never the
        extracted answer on this corpus.
        """
        runs = lo.probe_term(_swiglu(), DEFAULT)
        v = lo.compare_runs(runs)
        assert not v.contents_equal  # the board is real for contents
        assert v.partition_equal  # but the quotient converges
        assert v.answer_equal and v.frontier_equal
        assert v.certificates_ok

    def test_wall_ratio_is_finite(self):
        runs = lo.probe_term(_chain(), DEFAULT)
        assert lo.compare_runs(runs).wall_ratio >= 1.0

    def test_wall_ratio_zero_floor(self):
        """A zero-cost wall sample degrades to ratio 1.0, not inf."""
        runs = lo.probe_term(_chain(), DEFAULT)
        zeroed = tuple(replace(r, wall_s=0.0) for r in runs[:2])
        assert lo.compare_runs(zeroed).wall_ratio == 1.0


class TestCorpusAndReport:
    def test_sweep_and_report_lines(self, monkeypatch):
        cases = [("chain", _chain()), ("swiglu", _swiglu())]
        regimes = {
            "prod": {
                "rule_budgets": lo.production_budgets(DEFAULT, 128)
            },
            "improving": {
                "stop": "improving",
                "patience": 2,
                "cost_fn": None,
            },
        }
        arms = lo.order_arms(DEFAULT, seeds=(0,))
        rows = lo.sweep(cases, DEFAULT, regimes, arms=arms)
        assert len(rows) == 4
        lines = lo.report_lines(rows)
        assert len(lines) == 4
        assert all("cost=" in line for line in lines)
        # the swiglu board registers a contents diff somewhere
        assert any("contents" in line for line in lines)

    def test_report_lines_render_every_flag(self):
        """Fabricated worst-case cell: every diff flag renders."""
        v = lo.BoardVerdict(
            n_arms=2,
            contents_equal=False,
            partition_equal=False,
            answer_equal=False,
            frontier_equal=False,
            certificates_ok=False,
            cost=1.0,
            wall_ratio=2.0,
        )
        b = lo.BoardRun(
            arm="a",
            iterations=1,
            n_enodes=1,
            n_classes=1,
            n_proof_edges=0,
            rule_fires=(),
            suspended=(),
            stop="fixed_point",
            cost=1.0,
            term_repr="t",
            frontier=(),
            wall_s=0.0,
            certificate_ok=False,
        )
        (line,) = lo.report_lines(
            [{"name": "x", "regime": "r", "verdict": v, "base": b}]
        )
        for flag in (
            "contents",
            "partition",
            "ANSWER",
            "frontier",
            "CERT-FAIL",
        ):
            assert flag in line

    def test_zoo_terms_via_stubbed_zoo(self, monkeypatch):
        """Corpus loading works through the real export boundary."""
        import torch
        from catopt_discovery import zoo as zoo_mod
        from catopt_discovery.intake import Workload

        def one():
            return [
                Workload(
                    "Tiny",
                    lambda: (
                        torch.nn.ReLU().double(),
                        torch.randn(2, 4, dtype=torch.float64),
                    ),
                    kind="zoo",
                )
            ]

        monkeypatch.setattr(zoo_mod, "zoo", one)
        terms, errors = lo.zoo_terms()
        assert errors == []
        assert [n for n, _ in terms] == ["Tiny"]

    def test_zoo_terms_reports_export_errors(self, monkeypatch):
        from catopt_discovery import zoo as zoo_mod
        from catopt_discovery.intake import Workload

        def one():
            return [
                Workload(
                    "Bad",
                    lambda: (_ for _ in ()).throw(ValueError("nope")),
                )
            ]

        monkeypatch.setattr(zoo_mod, "zoo", one)
        terms, errors = lo.zoo_terms()
        assert terms == []
        assert len(errors) == 1 and "Bad" in errors[0]

    def test_model_terms_via_stubbed_cases(self, monkeypatch):
        from catopt_discovery import impact

        cases = [
            SimpleNamespace(name="stub", term=_chain()),
        ]
        monkeypatch.setattr(
            impact, "model_cases", lambda: (cases, ["err"])
        )
        terms, errors = lo.model_terms()
        assert [n for n, _ in terms] == ["stub"]
        assert errors == ["err"]


class TestMain:
    def test_main_runs_report(self, monkeypatch, capsys):
        cases = [("chain", _chain()), ("swiglu", _swiglu())]
        monkeypatch.setattr(
            lo, "model_terms", lambda: (cases, ["SomeErr: x"])
        )
        rc = lo.main(
            ["--corpus", "models", "--regime", "prod", "--seeds", "0"]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "law-order board" in out
        assert "answer-diff=0" in out
        assert "cert-fail=0" in out
        assert "export error" in out

    def test_main_regime_filter_and_limit(self, monkeypatch, capsys):
        cases = [("chain", _chain()), ("swiglu", _swiglu())]
        monkeypatch.setattr(lo, "model_terms", lambda: (cases, []))
        rc = lo.main(
            [
                "--corpus",
                "models",
                "--regime",
                "improving",
                "--limit",
                "1",
                "--seeds",
                "0",
                "1",
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "1 cases" in out

    def test_main_all_regimes(self, monkeypatch, capsys):
        """The default --regime=all path runs every regime."""
        cases = [("chain", _chain())]
        monkeypatch.setattr(lo, "model_terms", lambda: (cases, []))
        rc = lo.main(["--corpus", "models", "--seeds", "0"])
        assert rc == 0
        out = capsys.readouterr().out
        for regime in ("prod", "tight", "improving"):
            assert regime in out

    def test_main_zoo_branch(self, monkeypatch, capsys):
        cases = [("swiglu", _swiglu())]
        monkeypatch.setattr(lo, "zoo_terms", lambda: (cases, []))
        rc = lo.main(
            ["--corpus", "zoo", "--regime", "tight", "--seeds", "0"]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "zoo" in out and "answer-diff=0" in out
