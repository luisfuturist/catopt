"""Tests for ``catopt_discovery.intake`` — the real-workload intake.

The intake feeds the census programs nobody wrote by hand: a
``torch.nn.Module`` + example input is pushed through the real export
boundary (``torch.export`` → ``export_to_ir``), classified
ingested / census-only / rejected, and — when it survives — appended
to the census through the ``tools/intake_corpus.json`` +
``intake_tensors.pt`` side-files.

These tests exercise the real boundary on a handful of the
registry's own candidates — ``nn.LayerNorm``/``AngleLog`` ingest,
``nn.MaxPool1d`` is census-only (``max_pool1d`` is unbound),
``nn.CTCLoss`` is rejected by ``torch.export`` — plus the side-file
round-trip, the probe-eligibility filter, the census delta, the
report tables and ``main``'s flag surface with the corpus loaders
and the multi-minute pipeline delta stubbed to a tiny fabricated
result.  The full 166-workload ``main`` is a tool run, not a test.
"""

import argparse
import json

import pytest
import torch
import torch.nn as nn
from catopt_core.ir import Op, TensorType, Var, term_from_data
from catopt_discovery import intake as li
from catopt_discovery import pipeline as pl
from catopt_discovery import workload_gen as lwg
from catopt_discovery.impact import TermCase
from catopt_torch.adapters import TorchSink
from catopt_torch.report import VerifyReport
from catopt_torch.torch_bridge import export_to_ir


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _p(op: str, *args, **attrs) -> Op:
    return Op.make(op, *args, **attrs)


def _case(name: str, term: Op, *inputs: Var) -> TermCase:
    return TermCase(
        source="test",
        name=name,
        term=term,
        inputs=tuple(inputs),
        feed=tuple(
            torch.randn(tuple(v.typ.shape), dtype=torch.float64)
            for v in inputs
        ),
        param_vals={},
    )


# The intake registry is thunks — building the dict is free.
_CANDS = {w.name: w for w in li.candidates()}


@pytest.fixture(scope="module")
def sink() -> TorchSink:
    return TorchSink()


@pytest.fixture(scope="module")
def tiny_ingest(sink):
    """Run four registry candidates through the real boundary.

    ``nn.LayerNorm`` and ``AngleLog`` (a two-input feed) ingest;
    ``nn.MaxPool1d`` exports but ``max_pool1d`` has no binding
    (census-only); ``nn.CTCLoss`` is rejected by ``torch.export``.
    """
    torch.manual_seed(0)
    return li.ingest(
        cands=[
            _CANDS["nn.LayerNorm"],
            _CANDS["AngleLog"],
            _CANDS["nn.MaxPool1d"],
            _CANDS["nn.CTCLoss"],
        ],
        sink=sink,
    )


# ---------------------------------------------------------------------------
#  The candidate registry
# ---------------------------------------------------------------------------


def test_candidates_build_and_run_eager():
    """Every registry thunk yields a runnable ``(model, input)`` pair.

    ``_ingest_one`` normalises ``model.eval().double()`` before the
    export boundary — the same normalisation must also run the model
    eagerly, since export traces exactly this call.  A candidate that
    cannot run eager could never export.
    """
    torch.manual_seed(0)
    cands = li.candidates()
    assert {w.kind for w in cands} == {"torch-native", "compound"}
    assert len(cands) > 100
    failures = []
    for w in cands:
        model, x = w.build()
        assert isinstance(model, nn.Module), w.name
        model.eval().double()
        feed = x if isinstance(x, tuple) else (x,)
        try:
            with torch.no_grad():
                out = model(*feed)
        except Exception as e:
            failures.append(f"{w.name}: {type(e).__name__}: {e}")
            continue
        first = out[0] if isinstance(out, (tuple, list)) else out
        assert isinstance(first, torch.Tensor), w.name
    assert failures == []


# ---------------------------------------------------------------------------
#  _ingest_one / ingest — export, classify, verify
# ---------------------------------------------------------------------------


def test_ingest_one_ingested(sink):
    case, rec, rej = li._ingest_one(
        _CANDS["nn.LayerNorm"], sink, sink.supported_ops
    )
    assert rej is None
    assert rec["status"] == "ingested"
    assert rec["verify"] == "pass" and rec["verify_note"] == ""
    assert rec["unsupported_ops"] == []
    assert rec["name"] == "intake:nn.LayerNorm"
    assert rec["builder"] == "torch-native"
    assert rec["n_op_nodes"] >= 1 and rec["ops"]
    # The persisted term decodes back to the case's root.
    assert term_from_data(rec["term"]) == case.term
    assert case.source == "intake" and case.name == rec["name"]
    assert len(case.feed) == len(case.inputs) == len(rec["inputs"])
    # LayerNorm carries weight/bias — real params ride the case.
    assert case.param_vals
    in_meta = {v.name: list(v.typ.shape) for v in case.inputs}
    assert in_meta == {i["name"]: i["shape"] for i in rec["inputs"]}


def test_ingest_one_two_input_feed(sink):
    case, rec, rej = li._ingest_one(
        _CANDS["AngleLog"], sink, sink.supported_ops
    )
    assert rej is None and rec["status"] == "ingested"
    # A tuple example input stays a two-tensor feed.
    assert len(case.feed) == len(case.inputs) == 2


def test_ingest_one_census_only_unbound_op(sink):
    case, rec, rej = li._ingest_one(
        _CANDS["nn.MaxPool1d"], sink, sink.supported_ops
    )
    assert rej is None
    assert rec["status"] == "census-only"
    assert rec["verify"] == "skipped"
    assert rec["unsupported_ops"] == ["max_pool1d"]
    assert "max_pool1d" in rec["verify_note"]
    # Census-only still yields a TermCase — the term feeds the census.
    assert case is not None and case.term is not None


def test_ingest_one_export_rejection(sink):
    case, rec, rej = li._ingest_one(
        _CANDS["nn.CTCLoss"], sink, sink.supported_ops
    )
    assert case is None and rec is None
    assert rej.name == "nn.CTCLoss" and rej.stage == "export"
    assert "ctc" in rej.error.lower()


def test_ingest_one_build_rejection(sink):
    def boom():
        raise RuntimeError("cannot build")

    cand = li.Workload("broken", boom, kind="compound")
    case, rec, rej = li._ingest_one(cand, sink, sink.supported_ops)
    assert case is None and rec is None
    assert rej.stage == "build" and "cannot build" in rej.error


def test_ingest_one_verify_failed():
    s = TorchSink()
    s.verify = lambda *a, **k: VerifyReport(1.0, 2.5, False)
    case, rec, rej = li._ingest_one(
        _CANDS["nn.LayerNorm"], s, s.supported_ops
    )
    assert rej is None
    assert rec["status"] == "verify-failed"
    assert rec["verify"].startswith("FAIL max_rel=")
    assert rec["verify_note"] == rec["verify"]
    assert case is not None


def test_verify_pass_fail_and_error(sink):
    model = nn.LayerNorm(8)
    feed = (torch.randn(4, 8, dtype=torch.float64),)
    ir, tensors = export_to_ir(model.eval().double(), feed)
    assert li._verify(model, ir, tensors, feed, sink) == "pass"

    class Wrong(nn.Module):
        def forward(self, x):
            return x * 3.0

    bad = TorchSink()
    bad.lower = lambda *a, **k: Wrong()
    v = li._verify(model, ir, tensors, feed, bad)
    assert v.startswith("FAIL max_rel=")

    def raising(*a, **k):
        raise RuntimeError("lower boom")

    boom = TorchSink()
    boom.lower = raising
    v2 = li._verify(model, ir, tensors, feed, boom)
    assert v2.startswith("error:RuntimeError: lower boom")


def test_ingest_collects_outcomes(tiny_ingest, sink, monkeypatch):
    cases, records, rejections = tiny_ingest
    assert len(records) == 3 and len(rejections) == 1
    # The rejected candidate leaves no record or case; the
    # census-only one leaves both.
    assert len(cases) == 3
    by_name = {r["name"]: r for r in records}
    assert by_name["intake:nn.LayerNorm"]["status"] == "ingested"
    assert by_name["intake:nn.MaxPool1d"]["status"] == "census-only"
    assert rejections[0].stage == "export"
    # ``cands=None`` reads the registry — patched to stay tiny.
    monkeypatch.setattr(
        li, "candidates", lambda: [_CANDS["nn.MaxPool1d"]]
    )
    c2, r2, j2 = li.ingest(sink=sink)
    assert [r["name"] for r in r2] == ["intake:nn.MaxPool1d"]
    assert j2 == [] and len(c2) == 1
    # And ``sink=None`` builds the default sink.
    c3, r3, j3 = li.ingest(cands=[_CANDS["nn.MaxPool1d"]])
    assert [r["status"] for r in r3] == ["census-only"]
    assert len(c3) == 1 and j3 == []


# ---------------------------------------------------------------------------
#  Persistence — the side-file census
# ---------------------------------------------------------------------------


def test_side_file_roundtrip(tiny_ingest, tmp_path):
    cases, records, rejections = tiny_ingest
    jpath = tmp_path / "corpus.json"
    tpath = tmp_path / "tensors.pt"
    li.write_intake(
        records, rejections, li._tensors_of(cases), jpath, tpath
    )
    payload = json.loads(jpath.read_text())
    assert payload["format"] == 1 and payload["tool"] == "law_intake"
    assert len(payload["workloads"]) == 3
    assert payload["rejections"][0]["name"] == "nn.CTCLoss"

    recs = li.load_records(jpath)
    assert [r["name"] for r in recs] == [r["name"] for r in records]

    loaded = li.load_cases(jpath, tpath)
    assert len(loaded) == 3
    orig = {c.name: c for c in cases}
    for c in loaded:
        o = orig[c.name]
        assert c.source == "intake" and c.term == o.term
        assert len(c.feed) == len(o.feed)
        for a, b in zip(c.feed, o.feed, strict=True):
            assert torch.equal(a, b)
        assert set(c.param_vals) == set(o.param_vals)
        for k, p in c.param_vals.items():
            assert torch.equal(p, o.param_vals[k])
        in_shape = {v.name: tuple(v.typ.shape) for v in c.inputs}
        assert in_shape == {
            v.name: tuple(v.typ.shape) for v in o.inputs
        }


def test_probe_cases_eligibility(tiny_ingest, tmp_path):
    cases, records, rejections = tiny_ingest
    jpath = tmp_path / "c.json"
    tpath = tmp_path / "t.pt"
    li.write_intake(
        records, rejections, li._tensors_of(cases), jpath, tpath
    )
    probe = li.probe_cases(jpath, tpath)
    # Only the verified, in-budget workloads qualify — census-only
    # keeps feeding the census but never the firing probe.
    assert sorted(c.name for c in probe) == [
        "intake:AngleLog",
        "intake:nn.LayerNorm",
    ]
    assert all(c.feed for c in probe)
    assert li.probe_cases(jpath, tpath, max_nodes=0) == []


def test_load_records_and_cases_missing(tiny_ingest, tmp_path):
    missing = tmp_path / "nope.json"
    assert li.load_records(missing) == []
    assert li.load_cases(missing, tmp_path / "nope.pt") == []
    # JSON present, tensor blob absent -> empty feed, empty params;
    # the term still decodes because the census reads only ``.term``.
    cases, records, rejections = tiny_ingest
    jpath = tmp_path / "c.json"
    li.write_intake(records, rejections, {}, jpath, tmp_path / "t.pt")
    loaded = li.load_cases(jpath, tmp_path / "absent.pt")
    assert len(loaded) == 3
    orig = {c.name: c.term for c in cases}
    for c in loaded:
        assert c.feed == () and c.param_vals == {}
        assert c.term == orig[c.name]


def test_tensors_of_collects_feeds_and_params(tiny_ingest):
    cases, _, _ = tiny_ingest
    blob = li._tensors_of(cases)
    by_name = {c.name: c for c in cases}
    assert set(blob) == set(by_name)
    for name, pack in blob.items():
        c = by_name[name]
        assert pack["feed"] == list(c.feed)
        assert pack["params"] == dict(c.param_vals)


# ---------------------------------------------------------------------------
#  Census delta
# ---------------------------------------------------------------------------


def test_ops_of_and_n_nodes():
    x, y = _v("x", 4), _v("y", 4)
    shared = _p("mul", x, y)
    term = _p("add", shared, _p("neg", shared))
    assert li._ops_of(term) == {"add", "mul", "neg"}
    # The shared mul counts once — subterms are identity-deduped.
    assert li._n_nodes(term) == 3


def test_census_delta_reports_novelty():
    x, y, z = _v("x", 4, 4), _v("y", 4, 4), _v("z", 4, 4)
    base = [_case("b_mul", _p("mul", x, y), x, y)]
    intake_cases = [
        _case(
            "i_factor",
            _p("add", _p("mul", x, y), _p("mul", x, z)),
            x,
            y,
            z,
        ),
        _case("i_sig", _p("sigmoid", y), y),
    ]
    delta = li.census_delta(base, intake_cases)
    assert delta["n_base_tuples"] >= 1
    assert delta["n_base_shapes"] >= 1
    assert delta["new_ops"] == ["add", "sigmoid"]
    assert ("add", ("mul", "mul")) in delta["new_op_tuples"]
    assert delta["new_shapes"] >= 1
    assert set(delta["per_case"]) == {"i_factor", "i_sig"}
    new_tuples, n_shapes = delta["per_case"]["i_factor"]
    assert ("add", ("mul", "mul")) in new_tuples
    assert n_shapes >= 0


# ---------------------------------------------------------------------------
#  Report tables + the pipeline-delta seam
# ---------------------------------------------------------------------------


def test_report_tables_render():
    records = [
        {
            "name": "intake:a",
            "builder": "torch-native",
            "n_op_nodes": 3,
            "status": "ingested",
            "unsupported_ops": [],
        },
        {
            "name": "intake:b",
            "builder": "compound",
            "n_op_nodes": 1,
            "status": "census-only",
            "unsupported_ops": ["max_pool1d"],
        },
    ]
    delta = {"per_case": {"intake:a": ([("add", ("mul", "mul"))], 2)}}
    t = li._status_table(records, delta)
    assert "intake:a" in t and "ingested" in t
    assert "max_pool1d" in t and "census-only" in t

    rt = li._rejection_table(
        [
            li.Rejection("w", "export", "boom"),
            li.Rejection("b", "build", "bad"),
        ]
    )
    assert "export" in rt and "boom" in rt and "bad" in rt

    gt = li._gap_table(records)
    assert "max_pool1d" in gt and "intake:b" in gt
    assert li._gap_table(records[:1]) == "  (no binding gaps)"


def test_eligible_filters_by_status_and_nodes():
    x = _v("x", 4)
    cases = [
        _case("a", _p("mul", x, x), x),
        _case("b", _p("mul", x, x), x),
        _case("c", _p("mul", x, x), x),
    ]
    records = [
        {"name": "a", "status": "ingested", "n_op_nodes": 1},
        {"name": "b", "status": "census-only", "n_op_nodes": 1},
        {"name": "c", "status": "ingested", "n_op_nodes": 999},
    ]
    assert [c.name for c in li._eligible(cases, records)] == ["a"]


def _prop(name: str) -> pl.Proposal:
    return pl.Proposal(name, _p("mul", "A", "B"), "A", family="test")


def _ev(name: str, **kw) -> pl.Evidence:
    ev = pl.Evidence(proposal=_prop(name))
    ev.num_true = True
    ev.relation = "new"
    for k, v in kw.items():
        setattr(ev, k, v)
    return ev


def test_intake_fires_counts_intake_cases():
    ev = _ev("p", fire_cases=("intake:a", "intake:b", "model:c"))
    assert li._intake_fires(ev) == 2


def test_delta_table_renders_deltas():
    shared_base = _ev(
        "shared", fires=1, fires_typed=1, paid=0, matches=2
    )
    shared = _ev(
        "shared",
        fires=3,
        fires_typed=3,
        paid=1,
        matches=2,
        fire_cases=("intake:w1",),
    )
    newp = _ev(
        "new_law",
        fires=2,
        fires_typed=2,
        paid=2,
        matches=1,
        fire_cases=("intake:w2", "intake:w3", "model:m"),
    )
    res = {
        "baseline": {
            "n_terms": 2,
            "n_tuples": 2,
            "ranked": [shared_base],
        },
        "enlarged": {
            "n_terms": 4,
            "n_tuples": 3,
            "ranked": [shared, newp],
        },
    }
    out = li._delta_table(res)
    assert "new proposals (1)" in out and "new_law" in out
    assert "intake=2" in out  # two intake firings on new_law
    assert "fires 1->3" in out  # the shared proposal's delta row
    assert "shippable: 0 -> 2" in out and "new:" in out

    quiet = _ev("same", fires=1, fires_typed=1, paid=0)
    res2 = {
        "baseline": {"n_terms": 1, "n_tuples": 1, "ranked": [quiet]},
        "enlarged": {"n_terms": 1, "n_tuples": 1, "ranked": [quiet]},
    }
    out2 = li._delta_table(res2)
    assert "  (none)" in out2
    assert "(no shared proposal changed)" in out2
    assert "shippable: 0 -> 0" in out2


def test_run_delta_delegates_to_workload_gen(monkeypatch):
    calls = []

    def fake(corpus, probe, vocab, holdout):
        calls.append((len(corpus), len(probe), vocab, holdout))
        return {
            "n_terms": len(corpus),
            "n_tuples": 1,
            "ranked": [],
        }

    monkeypatch.setattr(lwg, "_run_pipeline", fake)
    x = _v("x", 4)
    b = [_case("b", _p("mul", x, x), x)]
    i = [_case("i", _p("add", x, x), x)]
    res = li._run_delta(b, b, i, i, "hand", "sel")
    assert set(res) == {"baseline", "enlarged"}
    assert res["baseline"]["n_terms"] == 1
    assert res["enlarged"]["n_terms"] == 2
    assert calls == [(1, 1, "hand", "sel"), (2, 2, "hand", "sel")]


def test_report_pipeline_populates_payload(monkeypatch, capsys):
    x = _v("x", 4)
    c = _case("m", _p("mul", x, x), x)
    monkeypatch.setattr(li, "_bench_cases", lambda: ([c], []))
    monkeypatch.setattr(li, "model_cases", lambda: ([c], []))
    records = [{"name": "m", "status": "ingested", "n_op_nodes": 1}]
    seen = {}

    def fake(base, probe_b, intake, probe_i, vocab, holdout):
        seen.update(
            n_base=len(base),
            n_probe=len(probe_b),
            n_intake=len(intake),
            n_probe_i=len(probe_i),
            vocab=vocab,
            holdout=holdout,
        )
        ev = _ev("p", fires=1, fires_typed=1, paid=1)
        return {
            "baseline": {"n_terms": 1, "n_tuples": 1, "ranked": []},
            "enlarged": {
                "n_terms": 2,
                "n_tuples": 2,
                "ranked": [ev],
            },
        }

    monkeypatch.setattr(li, "_run_delta", fake)
    args = argparse.Namespace(vocab="hand", holdout="sel")
    payload: dict = {}
    li._report_pipeline([c], records, args, payload)
    assert seen["vocab"] == "hand" and seen["holdout"] == "sel"
    assert seen["n_intake"] == 1 and seen["n_probe_i"] == 1
    pipe = payload["pipeline"]
    assert pipe["baseline_terms"] == 1 and pipe["enlarged_terms"] == 2
    assert pipe["new_proposals"] == ["p"]
    assert pipe["shippable"] == ["p"]
    assert "probe-eligible intake cases: 1" in capsys.readouterr().out


# ---------------------------------------------------------------------------
#  main — the flag surface, not the multi-minute corpus run
# ---------------------------------------------------------------------------


def _patch_main(monkeypatch, tiny_ingest) -> None:
    x = _v("x", 4)
    c = _case("m", _p("mul", x, x), x)
    monkeypatch.setattr(
        li, "ingest", lambda cands=None, sink=None: tiny_ingest
    )
    monkeypatch.setattr(li, "_bench_cases", lambda: ([c], []))
    monkeypatch.setattr(li, "model_cases", lambda: ([c], []))


def test_main_no_write_skip_pipeline(
    tiny_ingest, monkeypatch, tmp_path, capsys
):
    _patch_main(monkeypatch, tiny_ingest)
    out = tmp_path / "report.json"
    rc = li.main(
        [
            "--no-write",
            "--skip-pipeline",
            "--seed",
            "7",
            "--vocab",
            "hand",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    text = capsys.readouterr().out
    assert "law_intake" in text and "intake table" in text
    assert "binding-gap backlog" in text
    assert "export rejections" in text
    assert "--no-write: side files not written" in text
    payload = json.loads(out.read_text())
    assert payload["n_exported"] == 3
    assert payload["n_ingested"] == 2
    assert payload["n_census_only"] == 1
    assert payload["n_verify_failed"] == 0
    assert len(payload["rejections"]) == 1
    cd = payload["census_delta"]
    assert "new_op_tuples" in cd and "new_shapes" in cd


def test_main_writes_side_files(tiny_ingest, monkeypatch, tmp_path):
    _patch_main(monkeypatch, tiny_ingest)
    jpath = tmp_path / "c.json"
    tpath = tmp_path / "t.pt"
    rc = li.main(
        [
            "--out",
            str(jpath),
            "--tensors",
            str(tpath),
            "--skip-pipeline",
        ]
    )
    assert rc == 0
    assert jpath.exists() and tpath.exists()
    assert len(li.load_records(jpath)) == 3
    probe = li.probe_cases(jpath, tpath)
    assert sorted(p.name for p in probe) == [
        "intake:AngleLog",
        "intake:nn.LayerNorm",
    ]


def test_main_runs_pipeline_delta(
    tiny_ingest, monkeypatch, tmp_path, capsys
):
    _patch_main(monkeypatch, tiny_ingest)
    seen = {}
    ev = _ev("census:mul_x", fires=2, fires_typed=2, paid=1)

    def fake(base, probe_b, intake, probe_i, vocab, holdout):
        seen.update(vocab=vocab, holdout=holdout)
        return {
            "baseline": {"n_terms": 1, "n_tuples": 1, "ranked": []},
            "enlarged": {
                "n_terms": 4,
                "n_tuples": 2,
                "ranked": [ev],
            },
        }

    monkeypatch.setattr(li, "_run_delta", fake)
    out = tmp_path / "r.json"
    rc = li.main(
        [
            "--no-write",
            "--seed",
            "3",
            "--vocab",
            "hand",
            "--holdout",
            "select_mul",
            "--json",
            str(out),
        ]
    )
    assert rc == 0
    assert seen == {"vocab": "hand", "holdout": "select_mul"}
    text = capsys.readouterr().out
    assert "pipeline delta" in text
    assert "probe-eligible intake cases: 2" in text
    payload = json.loads(out.read_text())
    assert payload["pipeline"]["new_proposals"] == ["census:mul_x"]
    assert payload["pipeline"]["shippable"] == ["census:mul_x"]
