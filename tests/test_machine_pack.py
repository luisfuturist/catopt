"""The machine law pack — a store's admitted objects as a ruleset.

``catopt_discovery.machine_pack`` turns the evidence store's
``lemmas`` rows into a deployable :class:`~catopt_core.laws.RuleSet`:
the loader keeps the gauntlet-cleared objects the provenance ledger
(:data:`~catopt_core.laws.provenance.MACHINE_STORED`) attests,
requires every member to be full-data (``missing_hooks == ()``), and
reports — never crashes on — rows that cannot reconstruct.  The
behavioral pin is the human-bar measurement's headline
(``project/retros/human-bar.md``): ``affd_step_lift``'s AdaLNBlock
win, shipped through ``machine_default`` — the opt-in
``DEFAULT_RULES + pack`` composition.
"""

import json

import catopt_carriers  # noqa: F401 — register carriers before rules
import pytest
import torch
from catopt_core.cost import dag_cost
from catopt_core.egraph import Rewrite
from catopt_core.ir import Op, TensorType, Var
from catopt_core.laws import ALL_RULES, serialize
from catopt_discovery import evidence as ev
from catopt_discovery import machine_pack as mp
from catopt_discovery import object_synthesis as synth
from catopt_discovery import zoo as zoo_mod
from catopt_discovery.impact import TermCase, _cost_fn
from catopt_discovery.shape_proposal import _sink
from catopt_orchestrator.optimize import Optimizer, default_rules
from catopt_torch.backend import TorchBackend

_ZOO = {w.name: w for w in zoo_mod.zoo()}


def _p(op, *args, **attrs):
    return Op.make(op, *args, **attrs)


def _v(name, *shape):
    return Var(name, TensorType(tuple(shape)))


def _case(name, term, *inputs):
    return TermCase(
        source="test",
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=tuple(
            torch.randn(tuple(x.typ.shape), dtype=torch.float64)
            for x in inputs
        ),
        param_vals={},
    )


def _corpus(*cases):
    """The tiny injected gauntlet corpus (the gauntlet tests' shape)."""
    sink = _sink()
    return ev.GauntletCorpus(
        real_terms=tuple(c.term for c in cases),
        probe=tuple(cases),
        base_rules=tuple(ALL_RULES),
        census_op={},
        sink=sink,
        cost_fn=_cost_fn(sink),
    )


def _affd_step_case():
    """A firing site for ``affd_step_lift``: ``a*h + x`` on vectors."""
    a, h, x = _v("a", 4), _v("h", 4), _v("x", 4)
    return _case("affd_step", _p("add", _p("mul", a, h), x), a, h, x)


# ---------------------------------------------------------------------------
#  Store fixtures — the recorded object spellings (as the gauntlet tests
#  construct them) plus honest negatives
# ---------------------------------------------------------------------------


def _mul_transpose_l_id():
    """The admitted transpose strip — a ``MACHINE_STORED`` member."""
    return Rewrite(
        name="mul_transpose_l_id",
        lhs=_p(
            "mul",
            _p("transpose", "U", dim0="A_d0", dim1="A_d1"),
            "V",
        ),
        rhs=_p("mul", "U", "V"),
        law="mul(transpose(u,d0,d1),v) = mul(u,v) on a no-op swap",
        cond=("axes-noop", "U", "A_d0", "A_d1"),
    )


def _affd_step_lift():
    """The diagonal-scan lift — the AdaLN winner, ``MACHINE_STORED``."""
    return synth.lift_object(
        "affd_step_lift",
        ("add", ("mul", "a", "h"), "x"),
        ("aff_diag", "a", "x"),
        "applyd",
        state="h",
        cond=(
            "or",
            ("op-in", "h", ("add", "sub", "apply", "applyd")),
            ("leaf", "h"),
        ),
    ).rule


def _om_lift_object():
    """A stored object NOT on the ledger — the ``om_lift`` collision."""
    return synth.lift_object(
        "om_lift",
        ("matmul", ("softmax", "S", {"dim": -1}), "V"),
        ("om_elem", "S", "V"),
        "om_apply",
    ).rule


def _populate(conn):
    """Seed a known store: two admitted objects plus one foreign."""
    ev.store_object(conn, _mul_transpose_l_id(), kind="abstraction")
    ev.store_object(conn, _affd_step_lift(), kind="abstraction")
    ev.store_object(conn, _om_lift_object(), kind="abstraction")


# ---------------------------------------------------------------------------
#  The loader — load, skip, exclude
# ---------------------------------------------------------------------------


def test_pack_loads_only_the_admitted_names(tmp_path):
    """The default filter is the provenance ledger: ``om_lift`` is a
    stored object but not a ``MACHINE_STORED`` name — excluded, not
    loaded, and counted honestly."""
    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    conn.close()
    pack = mp.load_pack(tmp_path / "s.db")
    assert sorted(pack.loaded) == [
        "affd_step_lift",
        "mul_transpose_l_id",
    ]
    assert pack.excluded == ("om_lift",)
    assert pack.skipped == ()
    assert pack.rows == 3 == len(pack.loaded) + len(pack.excluded)
    assert {r.name for r in pack.rules} == set(pack.loaded)


def test_names_none_trusts_the_store(tmp_path):
    """``names=None`` loads every stored object — the caller's store is
    the admission record; the ledger filter is the default, not a cage."""
    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    conn.close()
    pack = mp.load_pack(tmp_path / "s.db", names=None)
    assert len(pack.rules) == 3
    assert "om_lift" in pack.loaded
    assert pack.excluded == ()


def test_missing_store_path_raises(tmp_path):
    """A path that names no store is an error, not a silently minted
    empty database — the loader never creates the thing it reads."""
    with pytest.raises(FileNotFoundError):
        mp.load_pack(tmp_path / "no_such.db")


def test_borrowed_connection_stays_open(tmp_path):
    """A caller-supplied connection is borrowed, not closed."""
    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    pack = mp.load_pack(conn)
    assert pack.store == "<connection>"
    conn.execute("SELECT 1").fetchone()  # still usable
    conn.close()


def test_every_pack_member_is_full_data(tmp_path):
    """``missing_hooks() == ()`` on every member — the pack is data:
    each rule round-trips through the record codec unchanged."""
    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    pack = mp.load_pack(conn)
    conn.close()
    assert pack.rules
    for rule in pack.rules:
        assert serialize.missing_hooks(rule) == ()
        rebuilt = serialize.object_from_data(
            serialize.object_to_data(rule, kind="abstraction")
        )
        assert rebuilt.lhs == rule.lhs and rebuilt.rhs == rule.rhs


def test_partial_and_corrupt_rows_skip_not_crash(tmp_path):
    """The honest skip paths: a procedural-remainder record (missing
    hooks) and a record claiming a kind the codec rejects both count
    as skips — the pack still loads what it can."""
    conn = ev.connect(str(tmp_path / "s.db"))
    ev.store_object(conn, _affd_step_lift(), kind="abstraction")
    # an admitted name whose record dropped its check hook
    ev.store_object(
        conn,
        Rewrite(
            name="aff_step_lift",
            lhs=_p("add", _p("matmul", "A", "h"), "x"),
            rhs=_p("apply", _p("aff", "A", "x"), "h"),
            check=lambda bound: True,
        ),
        kind="abstraction",
    )
    # an admitted name whose record claims an unknown kind
    key = ev.store_object(
        conn, _mul_transpose_l_id(), kind="abstraction"
    )
    rec = ev.stored_object(conn, key)
    rec["kind"] = "widget"
    conn.execute(
        "UPDATE lemmas SET law_json = ? WHERE alpha_key = ?",
        (json.dumps(rec, sort_keys=True), key),
    )
    conn.commit()
    pack = mp.load_pack(conn)
    conn.close()
    assert pack.loaded == ("affd_step_lift",)
    reasons = {s.name: s.reason for s in pack.skipped}
    assert reasons["aff_step_lift"].startswith("missing hooks: check")
    assert reasons["mul_transpose_l_id"].startswith("reconstruct:")
    assert pack.rows == len(pack.loaded) + len(pack.skipped)


def test_vanished_row_skips(tmp_path, monkeypatch):
    """A row that vanishes between listing and admit is a skip, not a
    crash — the loader never trusts an absent record."""
    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    calls = {"n": 0}
    orig = ev.admit_object

    def admit_once_then_vanish(c, k):
        calls["n"] += 1
        if calls["n"] == 2:
            return None
        return orig(c, k)

    monkeypatch.setattr(ev, "admit_object", admit_once_then_vanish)
    pack = mp.load_pack(conn)
    conn.close()
    assert len(pack.skipped) == 1
    assert pack.skipped[0].reason == "reconstruct: row vanished"
    assert len(pack.loaded) == 1  # the other admitted object loads


def test_duplicate_names_dedupe_to_newest(tmp_path):
    """Two rows sharing a rule name (different bodies) keep the newest
    row — ``lemma_rows`` arrives newest-first — and the older reports
    a duplicate-name skip."""
    conn = ev.connect(str(tmp_path / "s.db"))
    ev.store_object(
        conn,
        Rewrite(
            name="aff_step_lift",
            lhs=_p("mul", "P", "Q"),
            rhs=_p("mul", "Q", "P"),
        ),
        kind="abstraction",
    )
    ev.store_object(
        conn,
        Rewrite(
            name="aff_step_lift",
            lhs=_p("add", "P", "Q"),
            rhs=_p("add", "Q", "P"),
        ),
        kind="abstraction",
    )
    pack = mp.load_pack(conn)
    conn.close()
    assert pack.loaded == ("aff_step_lift",)
    assert pack.rules[0].lhs == _p("add", "P", "Q")  # newest wins
    assert [s.reason for s in pack.skipped] == ["duplicate rule name"]


def test_gauntlet_corpus_re_admits(tmp_path):
    """``gauntlet_corpus=`` re-runs the admission gauntlet per row:
    only ``usable`` objects join the pack — the strict reading of
    "the store's usable objects" for a store the ledger does not
    cover."""
    conn = ev.connect(str(tmp_path / "s.db"))
    ev.store_object(conn, _affd_step_lift(), kind="abstraction")
    # a known-false candidate shares the store — the gauntlet refuses it
    ev.store_object(
        conn,
        Rewrite(
            name="mul_is_add",
            lhs=_p("mul", "U", "V"),
            rhs=_p("add", "U", "V"),
        ),
        kind="abstraction",
    )
    pack = mp.load_pack(
        conn, names=None, gauntlet_corpus=_corpus(_affd_step_case())
    )
    conn.close()
    assert pack.loaded == ("affd_step_lift",)
    assert len(pack.skipped) == 1
    assert pack.skipped[0].reason.startswith("gauntlet: truth:")


# ---------------------------------------------------------------------------
#  The ruleset seam — the Optimizer's opt-in arm
# ---------------------------------------------------------------------------


def test_store_ruleset_and_laws_views(tmp_path):
    """The three faces agree: ``store_laws`` is the bare tuple,
    ``store_ruleset`` the named RuleSet, ``MachinePack.ruleset`` equal."""
    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    laws = mp.store_laws(conn)
    rs = mp.store_ruleset(conn)
    pack = mp.load_pack(conn)
    conn.close()
    assert isinstance(laws, tuple)
    # Each load reconstructs fresh Rewrite objects — compare by name.
    assert [r.name for r in laws] == [r.name for r in pack.rules]
    assert [r.name for r in rs.rules] == [r.name for r in pack.rules]
    assert rs.name == "machine_store"
    assert pack.ruleset() == rs  # RuleSet equality is by rule name
    assert {r.name for r in rs} == set(pack.loaded)


def test_machine_default_composes_with_default_rules(tmp_path):
    """The opt-in set: ``DEFAULT_RULES + pack`` — no name collisions
    (the ledger's collision policy holds: ``om_lift`` is excluded), and
    the pack members sit alongside the human library."""
    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    rs = mp.machine_default(conn)
    conn.close()
    base_names = {r.name for r in default_rules()}
    assert {r.name for r in rs} == base_names | {
        "affd_step_lift",
        "mul_transpose_l_id",
    }
    assert len(rs.rules) == len(default_rules().rules) + 2
    # the store's om_lift stayed out of the pack — the "om_lift" in the
    # union is the carrier preset's own human law, the same object
    carrier_om = next(r for r in default_rules() if r.name == "om_lift")
    assert carrier_om in rs.rules


def test_machine_default_composes_onto_a_caller_base(tmp_path):
    """``base=`` swaps the union's left side — the pack composes onto
    whatever ruleset the caller ships."""
    from catopt_core.laws import RuleSet

    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    rs = mp.machine_default(conn, base=RuleSet("bare", ()))
    conn.close()
    assert {r.name for r in rs} == {
        "affd_step_lift",
        "mul_transpose_l_id",
    }


# ---------------------------------------------------------------------------
#  The zoo delta — the store yield ships
# ---------------------------------------------------------------------------


def _search(w_name, rules):
    """Build the zoo model, run the search arm, return the result."""
    torch.manual_seed(0)
    model, x = _ZOO[w_name].build()
    model = model.eval().double()
    opt = Optimizer(backend=TorchBackend())
    return opt.search(model, x, rules=rules), x


def test_machine_pack_ships_the_adaln_yield(tmp_path):
    """The headline the human-bar retro promised: a store holding the
    admitted ``affd_step_lift`` deploys it — ``DEFAULT + pack`` fires
    the lift on ``AdaLNBlock``, verifies fp64, and prices under the
    default's own extraction."""
    conn = ev.connect(str(tmp_path / "s.db"))
    _populate(conn)
    rules = mp.machine_default(conn)
    conn.close()
    res, x = _search("AdaLNBlock", rules)
    assert "affd_step_lift" in set(res.stats["rule_fires"])
    cin = dag_cost(res.ir.root, res.cost_fn)
    cout = dag_cost(res.term, res.cost_fn)
    assert cout < cin - 1
    opt = Optimizer(backend=TorchBackend())
    low = opt.lower(res, x)
    assert low.verified is not None and low.verified.passed
    # The default alone cannot produce the member.
    res2, _x2 = _search("AdaLNBlock", default_rules())
    assert "affd_step_lift" not in set(res2.stats["rule_fires"])
