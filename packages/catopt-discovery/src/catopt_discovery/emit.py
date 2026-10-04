"""Admission-artifact emitter — the last step of the discovery loop.

``catopt_discovery.pipeline`` ends with a ranked SHIP/no-ship report, and
until now closing the loop meant a human hand-wrote the ``R(...)`` in
``catopt_core/laws/tensor.py`` plus its test module (``select_mul``,
``softmax_fold`` — see ``project/retros/law-softmax-fold-shipped.md``).
This tool emits that artifact mechanically for a SHIP-verdict
candidate:

* **the law** — the proposal's pattern trees rendered as ``Op.make``
  calls with metavar / attr-metavar leaves, its ``check`` / ``derive``
  hooks re-emitted as named ``def _check_<law>`` / ``_derive_<law>``
  functions (source-transcribed from the proposer's callables — never
  lambdas, so emission refuses when a hook has no source);
* **the tests** — ``test_admitted_<law>.py`` covering a
  match/instantiate round-trip, hook unit tests, firing plus
  *measured* decline cases (the emitter probes each attr-metavar
  perturbation against ``check`` and the e-graph before emitting a
  decline, so every decline in the file is one the law really takes),
  repeated-metavar mismatches, and an end-to-end
  fire → cost-delta → cert-replay → ``sink.verify`` on the model the
  candidate fired on;
* **the patch** — ``admission_<law>.patch``: a unified diff inserting
  the law block + a ``SIMPLIFICATION_RULES`` registration line into
  ``tensor.py`` and adding the test file — a *report* for human
  review, not a mutation.  Nothing under ``packages/`` is touched;
  ``admitted_<law>.py`` is the standalone module (same code,
  importable) so the artifact can be loaded and diffed against the
  library before the patch is ever applied.

Emission is refused — honestly, with the reason — when the candidate
is not SHIP, when a hook cannot be source-transcribed, or when a term
carries a leaf / attr value with no literal spelling.  A candidate
that needs a *novel* side condition the proposer did not carry is out
of scope: the emitter transcribes what the proposal knows, and the
summary says which parts still need human prose (``law=``, tags).

Run through the pipeline::

    .venv/bin/python -m catopt_discovery.pipeline \
        --holdout softmax_fold \
        --emit-admission recognize:softmax --out /tmp/admission
"""

from __future__ import annotations

import ast
import difflib
import inspect
import re
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from catopt_core.egraph import EGraph
from catopt_core.egraph.terms import _term_match
from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.meta import instantiate_pattern, pattern_metavars

# Sibling tools own the corpus/probe machinery; reuse them, never
# duplicate.
from catopt_discovery import REPO_ROOT
from catopt_discovery import impact as li

#: Repo-root-relative path of the module the patch edits.
_TENSOR_REL = "packages/catopt-core/src/catopt_core/laws/tensor.py"

#: Saturation budget for the emit-time decline/variant probes.
_PROBE_ITERS = 4
_PROBE_NODES = 40_000

#: Perturbations tried per attr-metavar binding: each produces either
#: a measured decline (``check`` vetoes / firing does not happen) or a
#: measured *accepted variant* (still fires) — only measured outcomes
#: are emitted as tests.
_MAX_VARIANT_TESTS = 2


class Unemittable(Exception):
    """A piece of the artifact has no honest source spelling."""


# ---------------------------------------------------------------------------
#  Term -> source rendering
# ---------------------------------------------------------------------------


def _literal(v: Any) -> str:
    """Return ``repr(v)`` if it round-trips through ``literal_eval``."""
    s = repr(v)
    try:
        ast.literal_eval(s)
    except (ValueError, SyntaxError) as e:
        raise Unemittable(
            f"attr value {s!r} has no literal spelling"
        ) from e
    return s


def _term_src(t: Any) -> str:
    """Render *t* (pattern or concrete term) as Python source.

    ``Op`` nodes become ``Op.make`` calls (``str`` attr values stay
    quoted — as pattern metavars or literal strings, both spell the
    same source); ``str`` leaves stay quoted metavars; ``Var`` /
    ``Param`` / ``Const`` leaves reconstruct by constructor; raw
    ``int`` / ``float`` leaves emit as ``Const(...)`` — the exporter's
    canonical leaf form.
    """
    if isinstance(t, Op):
        parts = [repr(t.op), *(_term_src(a) for a in t.args)]
        for k, v in t.attrs.items():
            if not k.isidentifier() or k in ("op", "validate"):
                raise Unemittable(f"attr key {k!r} is not a kwarg")
            parts.append(f"{k}={_literal(v)}")
        return f"Op.make({', '.join(parts)})"
    if isinstance(t, str):
        return repr(t)
    if isinstance(t, (Var, Param)):
        kind = type(t).__name__
        return f"{kind}({t.name!r}, TensorType({t.typ.shape!r}))"
    if isinstance(t, Const):
        return f"Const({t.value!r})"
    if isinstance(t, bool):
        return f"Const({t!r})"
    if isinstance(t, (int, float)):
        return f"Const({t!r})"
    raise Unemittable(
        f"leaf {t!r} ({type(t).__name__}) has no source spelling"
    )


def _hook_src(fn: Any) -> str:
    """Transcribe *fn*'s ``def`` block verbatim, keeping its name.

    Only named module-level functions are emittable: a lambda or a
    computed callable (``partial``, a closure factory) is novel logic
    the artifact cannot spell — emission refuses instead of guessing.
    The proposer's name is kept as-is (proposers name hooks after the
    shipped shapes — ``_check_sum_keepdim``, ``_derive_softmax_dim``);
    a collision with an existing library hook name is reported in the
    summary, not silently resolved.
    """
    if not (inspect.isfunction(fn) or inspect.ismethod(fn)):
        raise Unemittable(
            f"hook {fn!r} is not a named function — emit needs a "
            "source-level `def` the proposer carried"
        )
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as e:
        raise Unemittable(
            f"no source for hook {getattr(fn, '__name__', fn)!r}"
        ) from e
    name = fn.__name__
    if not re.search(rf"def\s+{re.escape(name)}\s*\(", src):
        raise Unemittable(
            f"source transcription of hook {name!r} did not "
            f"round-trip its own def — getsource returned "
            f"{src[:120]!r}"
        )
    return src.rstrip()


def _mv_counts(pat: Any, counts: dict[str, int] | None = None) -> dict:
    """Count occurrences of each metavariable (leaf and attr) in *pat*."""
    counts = {} if counts is None else counts
    if isinstance(pat, str):
        counts[pat] = counts.get(pat, 0) + 1
    elif isinstance(pat, Op):
        for a in pat.args:
            _mv_counts(a, counts)
        for v in pat.attrs.values():
            if isinstance(v, str):
                counts["$attr:" + v] = counts.get("$attr:" + v, 0) + 1
    return counts


def _mv_positions(pat: Any, mv: str) -> list[tuple]:
    """Return child-index paths where *mv* occurs in *pat*."""
    out: list[tuple] = []
    attr_mv = mv.removeprefix("$attr:")

    def walk(t: Any, path: tuple) -> None:
        if isinstance(t, Op):
            for i, a in enumerate(t.args):
                walk(a, (*path, i))
            for v in t.attrs.values():
                if (
                    mv.startswith("$attr:")
                    and v == attr_mv
                    and isinstance(v, str)
                ):
                    out.append(path)
        elif t == mv:
            out.append(path)

    walk(pat, ())
    return out


def _subterm(term: Any, path: tuple) -> Any:
    for i in path:
        term = term.args[i]
    return term


def _replace_leaf(term: Any, path: tuple, new: Any) -> Any:
    """*term* with the subterm at *path* replaced by *new*."""
    if not path:
        return new
    i = path[0]
    args = list(term.args)
    args[i] = _replace_leaf(args[i], path[1:], new)
    return Op.make(term.op, *args, **dict(term.attrs))


def _replace_attr(term: Any, path: tuple, key: str, value: Any) -> Any:
    """*term* with ``attrs[key]`` at *path* set to *value*."""
    if not path:
        attrs = dict(term.attrs)
        attrs[key] = value
        return Op.make(term.op, *term.args, **attrs)
    i = path[0]
    args = list(term.args)
    args[i] = _replace_attr(args[i], path[1:], key, value)
    return Op.make(term.op, *args, **dict(term.attrs))


def _perturbations(v: Any) -> list[tuple[str, Any]]:
    """Return ``(label, value)`` perturbation candidates for *v*."""
    if isinstance(v, bool):
        return [("flip", not v)]
    if isinstance(v, int):
        return [("plus1", v + 1)]
    if isinstance(v, float):
        return [("plus1", v + 1.0)]
    if isinstance(v, tuple):
        out = [("extend", (*v, v[-1] if v else 0))]
        if len(v) == 1:
            out.append(("scalar", v[0]))
        return out
    return []


# ---------------------------------------------------------------------------
#  Emit-time probes — every emitted decline/variant is measured first
# ---------------------------------------------------------------------------


def _fires(term: Any, rule: Any) -> bool:
    """Return True iff *rule* fires on *term* in a bare e-graph."""
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(
        [rule],
        root,
        max_iterations=_PROBE_ITERS,
        max_nodes=_PROBE_NODES,
    )
    return eg.rule_fires.get(rule.name, 0) > 0


@dataclass
class _Probe:
    """One measured perturbation outcome for the emitted test file."""

    key: str
    label: str
    value: Any
    check_ok: bool | None  # None when the rule carries no check
    mintable: bool  # False when the perturbed LHS fails to mint
    fired: bool


def _probe_attrs(proposal: Any, subst: dict) -> list[_Probe]:
    """Probe every attr-metavar perturbation against check and firing."""
    out: list[_Probe] = []
    for key in sorted(k for k in subst if k.startswith("$attr:")):
        for label, value in _perturbations(subst[key]):
            bad = {**subst, key: value}
            check_ok: bool | None = None
            if proposal.check is not None:
                try:
                    check_ok = bool(proposal.check(bad))
                except Exception:
                    check_ok = False
            try:
                bad_term = instantiate_pattern(proposal.lhs, bad)
            except Exception:
                mintable, fired = False, False
            else:
                mintable = True
                fired = _fires(bad_term, proposal.as_rule())
            if check_ok is False or fired is False:
                out.append(
                    _Probe(key, label, value, check_ok, mintable, False)
                )
            elif fired:
                out.append(
                    _Probe(key, label, value, check_ok, mintable, True)
                )
    return out


# ---------------------------------------------------------------------------
#  Emission
# ---------------------------------------------------------------------------


@dataclass
class Emission:
    """The emitted artifact set plus the review summary."""

    emitted: bool
    law: str = ""
    reason: str = ""
    files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _safe_name(name: str) -> str:
    """Sanitize a candidate name into a snake_case rule name."""
    out = re.sub(r"[^0-9a-zA-Z_]+", "_", name).strip("_").lower()
    out = re.sub(r"_+", "_", out)
    if not out or out[0].isdigit():
        raise Unemittable(
            f"candidate name {name!r} has no law spelling"
        )
    return out


def _test_name(*parts: str) -> str:
    """Sanitize test-name components into a valid function suffix."""
    return re.sub(r"[^a-z0-9]+", "_", "_".join(parts).lower()).strip(
        "_"
    )


def _subst_for(proposal: Any, ev: Any, case: Any) -> dict:
    """Return the real binding — preferring the firing model's match."""
    if case is not None:
        for sub in li._iter_subterms(case.term):
            s = _term_match(proposal.lhs, sub)
            if s is not None:
                return s
    if getattr(ev, "match_term", None) is not None:
        s = _term_match(proposal.lhs, ev.match_term)
        if s is not None:
            return s
    raise Unemittable("no real LHS match to bind the tests to")


def _law_block(
    law: str, const: str, proposal: Any, ev: Any, holdout: str | None
) -> str:
    """Render the tensor.py block: hooks + the ``R(...)`` definition."""
    lines = [
        "# " + "-" * 74,
        f"#  Emitted admission artifact — {law}",
        "# " + "-" * 74,
        "",
        f"# Emitted by catopt_discovery.pipeline --emit-admission "
        f"{proposal.name}"
        + (f" --holdout {holdout}" if holdout else ""),
        "# Review before apply: the `law=` prose, the tag set, and the",
        "# generated names.  Evidence at emission: "
        f"fires={ev.fires} paid={ev.paid} "
        f"drop={ev.cost_drop * 100:.1f}% cert="
        f"{'pass' if ev.cert_fail == 0 else 'FAIL'} "
        f"enode={ev.closure_ratio:.2f}x",
        "",
    ]
    check_name = (
        proposal.check.__name__ if proposal.check is not None else ""
    )
    derive_name = (
        proposal.derive.__name__ if proposal.derive is not None else ""
    )
    if proposal.check is not None:
        lines += [_hook_src(proposal.check), "", ""]
    if proposal.derive is not None:
        lines += [_hook_src(proposal.derive), "", ""]
    extras = ""
    if proposal.check is not None:
        extras += f"    check={check_name},\n"
    if proposal.derive is not None:
        extras += f"    derive={derive_name},\n"
    lines.append(
        f"{const} = R(\n"
        f'    "{law}",\n'
        f"    {_term_src(proposal.lhs)},\n"
        f"    {_term_src(proposal.rhs)},\n"
        f'    law="TODO(reviewer): describe the equality — emitted from "\n'
        f'    "pipeline candidate {proposal.name!r}.",\n'
        f"{extras}"
        f"    tags=_SIM,\n"
        f")"
    )
    return "\n".join(lines)


def _module_src(law: str, const: str, block: str, proposal: Any) -> str:
    """Render the standalone importable module wrapping *block*."""
    # The block's tensor.py style (`tags=_SIM`, section comments) is
    # adapted to a self-contained module here.
    body = block.replace("tags=_SIM", "tags=(tags.SIMPLIFICATION,)")
    i = body.index("# Emitted by")
    body = body[i:]
    return f'''"""Emitted admission artifact for ``{proposal.name}`` — preview.

Generated by ``catopt_discovery.pipeline --emit-admission``.  The same
code lands inside ``catopt_core/laws/tensor.py`` via the companion
patch; this module exists so the artifact can be imported, compiled
and diffed against the library *before* review applies anything.
"""

from catopt_core.ir import Const, Op, Param, TensorType, Var
from catopt_core.laws import tags
from catopt_core.laws.base import R

{body}
'''


def _insert_tensor(src: str, block: str, const: str) -> str:
    """Return *src* with the law block and registration inserted."""
    lines = src.splitlines(keepends=True)
    anchor = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("#  Rule collections")
    )
    # back up over the divider line above the title
    while anchor > 0 and not lines[anchor - 1].startswith("# ---"):
        anchor -= 1
    anchor -= 1  # the divider itself
    lines[anchor:anchor] = [block + "\n", "\n", "\n"]
    reg = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("SIMPLIFICATION_RULES")
    )
    end = next(
        i
        for i in range(reg + 1, len(lines))
        if lines[i].rstrip() == "]"
    )
    lines[end:end] = [f"    {const},\n"]
    return "".join(lines)


def _patch(
    orig: str, patched: str, test_src: str, law: str, test_rel: str
) -> str:
    """Return the unified diff: tensor.py edit + the new test file."""
    d1 = difflib.unified_diff(
        orig.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=f"a/{_TENSOR_REL}",
        tofile=f"b/{_TENSOR_REL}",
        n=3,
    )
    d2 = difflib.unified_diff(
        [],
        test_src.splitlines(keepends=True),
        fromfile="/dev/null",
        tofile=f"b/{test_rel}",
        n=3,
    )
    body = "".join(d1)
    body += (
        f"diff --git a/{test_rel} b/{test_rel}\nnew file mode 100644\n"
    )
    body += "".join(d2)
    return body


# ---------------------------------------------------------------------------
#  The emitted test module
# ---------------------------------------------------------------------------


def _feed_src(case: Any) -> str:
    """Emit the input-feed builder for the e2e test.

    The feed order must match the emitted ``_ir_of``'s input order —
    both walk the term the same way (DFS, deduped).
    """
    if case is None:
        raise Unemittable("no firing model case to bind the e2e test")
    args = []
    # Inputs are keyed by *name* — a Var occurring at several leaves is
    # still one input, and the feed must bind it consistently.
    inputs: dict[str, Var] = {}
    for s in li._iter_subterms(case.term):
        if not isinstance(s, Var):
            continue
        prev = inputs.get(s.name)
        if prev is not None and prev.typ != s.typ:
            raise Unemittable(
                f"input {s.name!r} occurs with two different types "
                "— cannot emit an unambiguous feed"
            )
        inputs[s.name] = s
    for v in inputs.values():
        shape = v.typ.shape
        if any(d is None for d in shape):
            raise Unemittable(
                f"input {v.name!r} has an unknown dim — cannot "
                "emit a torch.randn feed"
            )
        args.append(
            f"        torch.randn({shape!r}, dtype=torch.float64),"
        )
    if not args:
        raise Unemittable("the firing case has no inputs")
    return (
        "def _feed():\n    torch.manual_seed(0)\n    return (\n"
        + "\n".join(args)
        + "\n    )\n"
    )


def _param_feed_src(term: Any) -> str:
    """Emit the param-feed builder (one randn per ``Param`` leaf)."""
    params = {
        s.name: s.typ.shape
        for s in li._iter_subterms(term)
        if isinstance(s, Param)
    }
    if not params:
        return "def _param_vals():\n    return {}\n"
    if any(any(d is None for d in s) for s in params.values()):
        raise Unemittable(
            "a Param leaf has an unknown dim — cannot emit a feed"
        )
    rows = "\n".join(
        f"        {n!r}: torch.randn({tuple(s)!r}, "
        "dtype=torch.float64),"
        for n, s in sorted(params.items())
    )
    return (
        "def _param_vals():\n    torch.manual_seed(1)\n"
        "    return {\n" + rows + "\n    }\n"
    )


def _test_module(
    law: str,
    const: str,
    proposal: Any,
    ev: Any,
    case: Any,
    subst: dict,
    probes: list[_Probe],
    mismatches: list[tuple[str, Any]],
    holdout: str | None,
) -> str:
    """Render ``test_admitted_<law>.py``."""
    p = proposal
    has_check = p.check is not None
    has_derive = p.derive is not None
    rule_name = law  # the emitted R() name, not the candidate label
    mvs = sorted(pattern_metavars(p.lhs))

    bound_defs = []
    subst_rows = []
    for k in mvs:
        if k.startswith("$attr:"):
            subst_rows.append(f'    "{k}": {_literal(subst[k])},')
        else:
            name = "_B_" + _test_name(k).upper()
            bound_defs.append(f"{name} = {_term_src(subst[k])}")
            subst_rows.append(f'    "{k}": {name},')
    bound_block = "\n\n".join(bound_defs)
    subst_block = "_SUBST = {\n" + "\n".join(subst_rows) + "\n}"

    imports = (
        f"from catopt_core.laws.tensor import (\n"
        f"    {const} as _RULE,\n"
        + (f"    {p.check.__name__} as _check,\n" if has_check else "")
        + (
            f"    {p.derive.__name__} as _derive,\n"
            if has_derive
            else ""
        )
        + ")"
    )

    tests: list[str] = []

    # -- round-trip --
    mv_set = "{" + ", ".join(repr(m) for m in mvs) + "}"
    tests.append(
        f'''
def test_match_instantiate_roundtrip():
    """Instantiating the LHS and re-matching recovers the binding."""
    assert pattern_metavars(_RULE.lhs) == {mv_set}
    term = instantiate_pattern(_RULE.lhs, _SUBST)
    found = match_pattern(_RULE.lhs, term)
    assert found is not None
    assert instantiate_pattern(_RULE.lhs, found) == term'''
    )
    if has_derive:
        tests.append(
            """
    rhs_subst = dict(found)
    rhs_subst.update(_RULE.derive(rhs_subst))
    rhs = instantiate_pattern(_RULE.rhs, rhs_subst)
    assert rhs is not None"""
        )

    # -- hook unit tests --
    if has_check:
        tests.append(
            '''
def test_check_accepts_the_real_binding():
    """``check`` admits the binding the real match produced."""
    assert _check(dict(_SUBST))'''
        )
    if has_derive:
        derived = p.derive(dict(subst)) or {}
        lit = (
            "{"
            + ", ".join(
                f"{k!r}: {_literal(v)}" for k, v in derived.items()
            )
            + "}"
        )
        tests.append(
            f'''
def test_derive_produces_the_rhs_attrs():
    """``derive`` computes exactly the RHS-only attr bindings."""
    assert _derive(dict(_SUBST)) == {lit}'''
        )

    # -- firing --
    if has_derive:
        rhs_note = (
            "    rhs_subst = dict(_SUBST)\n"
            "    rhs_subst.update(_RULE.derive(rhs_subst))\n"
            "    rhs = instantiate_pattern(_RULE.rhs, rhs_subst)"
        )
    else:
        rhs_note = "    rhs = instantiate_pattern(_RULE.rhs, _SUBST)"
    tests.append(
        f'''
def test_fires_and_rhs_is_member():
    """The emitted law fires on its own LHS instance; the folded RHS
    is a member of the root class."""
    term = instantiate_pattern(_RULE.lhs, _SUBST)
    eg, root, _best = _saturate(term, [_RULE], _cost_fn())
    assert eg.rule_fires.get("{rule_name}", 0) > 0
{rhs_note}
    assert list(eg.matches(rhs, eg.find(root)))'''
    )

    # -- measured declines / accepted variants --
    variant_budget = _MAX_VARIANT_TESTS
    for pr in probes:
        bad = f"{{**_SUBST, {pr.key!r}: {_literal(pr.value)}}}"
        if pr.fired:
            if variant_budget <= 0:
                continue
            variant_budget -= 1
            tests.append(
                f'''
def test_fires_on_variant_{_test_name(pr.key, pr.label)}():
    """An accepted spelling the probe measured firing on."""
    bad = {bad}
    term = instantiate_pattern(_RULE.lhs, bad)
    eg, _root, _best = _saturate(term, [_RULE], _cost_fn())
    assert eg.rule_fires.get("{rule_name}", 0) > 0'''
            )
        else:
            check_line = ""
            if has_check and pr.check_ok is False:
                check_line = (
                    f"    assert not _check({{**_SUBST, "
                    f"{pr.key!r}: {_literal(pr.value)}}})\n"
                )
            fire_lines = ""
            if pr.mintable:
                fire_lines = (
                    f"    bad = {{**_SUBST, {pr.key!r}: "
                    f"{_literal(pr.value)}}}\n"
                    "    term = instantiate_pattern(_RULE.lhs, bad)\n"
                    "    eg, _root, _best = _saturate(\n"
                    "        term, [_RULE], _cost_fn()\n"
                    "    )\n"
                    f'    assert eg.rule_fires.get("{rule_name}", 0)'
                    " == 0"
                )
            if not check_line and not fire_lines:
                continue
            tests.append(
                f'''
def test_declines_on_{_test_name(pr.key, pr.label)}():
    """Attr-violating input — measured non-firing at emission."""
{check_line}{fire_lines}'''
            )

    # -- repeated-metavar mismatches --
    for j, (mv, bad_term) in enumerate(mismatches):
        mname = _test_name(mv.lstrip("$").replace("attr:", "attr"))
        tests.append(
            f'''
_BAD_{_test_name(mv).upper()}_{j} = {_term_src(bad_term)}


def test_declines_on_{mname}_mismatch():
    """Two occurrences of a repeated metavar bound differently — the
    matcher vetoes before ``check`` is ever consulted."""
    assert match_pattern(_RULE.lhs, _BAD_{_test_name(mv).upper()}_{j}) is None
    eg, _root, _best = _saturate(
        _BAD_{_test_name(mv).upper()}_{j}, [_RULE], _cost_fn()
    )
    assert eg.rule_fires.get("{rule_name}", 0) == 0'''
        )

    # -- e2e on the firing model --
    if case is not None:
        cname = _test_name(case.name)
        tests.append(
            f'''
_MODEL_TERM = {_term_src(case.term)}


def test_end_to_end_on_{cname}():
    """Fire -> cost delta -> certificate replay -> ``sink.verify`` on
    the real ``{case.name}`` export the candidate fired on."""
    term = _MODEL_TERM
    eg, root, best = _saturate(term, [_RULE], _cost_fn())
    assert eg.rule_fires.get("{rule_name}", 0) > 0
    c0, c1 = dag_cost(term, _cost_fn()), dag_cost(best, _cost_fn())
    assert c1 < c0
    cert = eg.certificate(term, best, cost_fn=_cost_fn())
    verify_certificate(term, cert)
    m0 = _lower_extracted(
        term, _ir_of(term), _param_vals(), _SINK
    )
    m1 = _lower_extracted(
        best, _ir_of(best), _param_vals(), _SINK
    )
    assert _SINK.verify(m0, m1, _feed(), rtol=1e-4).passed'''
        )

    feed_src = _feed_src(case) if case is not None else ""
    pfeed_src = _param_feed_src(case.term) if case is not None else ""
    ho = f" (holdout: ``{holdout}``)" if holdout else ""
    evidence = (
        f"fires={ev.fires} paid={ev.paid} "
        f"drop={ev.cost_drop * 100:.1f}% "
        f"cert={'pass' if ev.cert_fail == 0 else 'FAIL'} "
        f"enode={ev.closure_ratio:.2f}x"
    )

    body_tests = "\n\n".join(t.rstrip() for t in tests)
    return f'''"""Tests for the emitted ``{law}`` law — GENERATED artifact.

Emitted by ``catopt_discovery.pipeline --emit-admission
{proposal.name}``{ho}; review with
``admission_{law}.patch``.  On apply this file lands as
``tests/test_admitted_{law}.py`` and imports the law from
``catopt_core.laws.tensor``.

    {proposal.lhs!r}
        -> {proposal.rhs!r}

Evidence at emission: {evidence}.
"""

from __future__ import annotations

import torch
from catopt_core.cost import backend_cost, dag_cost, executor_cost_for
from catopt_core.egraph import EGraph, verify_certificate
from catopt_core.ir import IR, Const, Op, Param, TensorType, Var
{imports}
from catopt_core.meta import (
    instantiate_pattern,
    match_pattern,
    pattern_metavars,
)
from catopt_orchestrator.optimize import _lower_extracted
from catopt_torch.adapters import TorchSink

_SINK = TorchSink()


def _cost_fn():
    """The pipeline's selection model (roofline + per-dispatch term)."""
    return backend_cost(
        executor_cost_for(lowering="generic"), _SINK.supported_ops
    )


def _saturate(term, rules, cost_fn, iters=6, nodes=60_000):
    eg = EGraph()
    root = eg.add_term(term)
    eg.run(rules, root, max_iterations=iters, max_nodes=nodes)
    return eg, root, eg.extract_best(root, cost_fn)


def _params_of(term):
    """Every ``Param`` leaf of *term* (DAG-aware)."""
    seen: set[int] = set()
    out: dict[str, Param] = {{}}
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        if isinstance(t, Param):
            out[t.name] = t
        elif isinstance(t, Op):
            stack.extend(t.args)
    return out


def _ir_of(term):
    """Wrap *term* as an ``IR`` with its ``Var`` leaves as inputs."""
    inputs = {{v.name: v for v in _walk(term) if isinstance(v, Var)}}
    return IR(
        root=term,
        inputs=list(inputs.values()),
        input_names=set(inputs),
        params=_params_of(term),
    )


def _walk(term):
    """Yield every distinct subterm of *term* (DAG-aware)."""
    seen: set[int] = set()
    out = []
    stack = [term]
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen.add(id(t))
        out.append(t)
        if isinstance(t, Op):
            stack.extend(t.args)
    return out


{feed_src}

{pfeed_src}

#: The real binding — what the matcher recovers on the firing model.
{bound_block}

{subst_block}


{body_tests}
'''


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------


def _find_ranked(result: dict, name: str) -> Any | None:
    for ev in result["ranked"]:
        if ev.proposal.name == name:
            return ev
    return None


def _compile_check(src: str, fname: str) -> None:
    compile(src, fname, "exec")


def _ruff_format(src: str) -> str:
    """Format *src* with the repo's ruff (best effort, stdin round-trip)."""
    ruff = REPO_ROOT / ".venv/bin/ruff"
    if not ruff.exists():
        return src
    r = subprocess.run(
        [str(ruff), "format", "-"],
        input=src,
        capture_output=True,
        text=True,
        check=False,
    )
    return r.stdout if r.returncode == 0 and r.stdout else src


def _mismatch_terms(
    proposal: Any, subst: dict
) -> list[tuple[str, Any]]:
    """Build one attr/leaf-mismatched term per repeated metavariable."""
    out: list[tuple[str, Any]] = []
    counts = _mv_counts(proposal.lhs)
    base = instantiate_pattern(proposal.lhs, dict(subst))
    for mv, n in sorted(counts.items()):
        if n < 2:
            continue
        paths = _mv_positions(proposal.lhs, mv)
        if len(paths) < 2:
            continue
        second = paths[1]
        if mv.startswith("$attr:"):
            key = next(
                k
                for k, v in _subterm(proposal.lhs, second).attrs.items()
                if v == mv.removeprefix("$attr:")
            )
            bound = subst[mv]
            pert = _perturbations(bound)
            if not pert:
                continue
            bad = _replace_attr(base, second, key, pert[0][1])
        else:
            bound = subst[mv]
            typ = (
                bound.typ if isinstance(bound, Var) else TensorType(())
            )
            bad = _replace_leaf(base, second, Var("mv_mismatch", typ))
        out.append((mv, bad))
    return out


def emit_admission(
    result: dict,
    name: str,
    out_dir: str | Path,
    tensor_path: Path | None = None,
) -> Emission:
    """Emit the admission artifact for a ranked candidate.

    *result* is the ``run_pipeline`` return value (its ``ranked``
    evidence and ``models`` cases); *name* the proposal name as shown
    in the report; *out_dir* receives the patch, the standalone law
    module, the test file and the markdown summary.  Emission is
    refused — ``Emission(emitted=False, reason=...)`` — for a
    non-SHIP candidate or a piece with no honest source spelling.
    """
    out = Path(out_dir)
    ev = _find_ranked(result, name)
    if ev is None:
        return Emission(
            emitted=False,
            reason=f"unknown candidate {name!r} — not in the "
            "ranked proposal list",
        )
    if not ev.shippable:
        return Emission(
            emitted=False,
            reason=f"not a SHIP candidate: {ev.no_ship_reason}",
        )
    em = Emission(emitted=True)
    try:
        law = _safe_name(name)
        const = law.upper()
        p = ev.proposal
        holdout = result.get("holdout")

        models = {c.name: c for c in result.get("models", [])}
        case = next(
            (models[c] for c in ev.fire_cases if c in models), None
        )
        subst = _subst_for(p, ev, case)
        probes = _probe_attrs(p, subst)
        mismatches = _mismatch_terms(p, subst)

        block = _law_block(law, const, p, ev, holdout)
        module = _module_src(law, const, block, p)
        test_rel = f"tests/test_admitted_{law}.py"
        test = _test_module(
            law, const, p, ev, case, subst, probes, mismatches, holdout
        )

        tpath = tensor_path or Path(_TENSOR_REL)
        orig = tpath.read_text()
        # Format every emitted artifact in-memory so the patch is the
        # post-`ruff format` text — the file it lands in is already
        # format-clean, so the diff stays confined to the insertion.
        module = _ruff_format(module)
        test = _ruff_format(test)
        patched = _ruff_format(_insert_tensor(orig, block, const))
        patch = _patch(orig, patched, test, law, test_rel)

        _compile_check(module, f"admitted_{law}.py")
        _compile_check(test, f"test_admitted_{law}.py")
        _compile_check(patched, "tensor.py(patched)")
    except Unemittable as e:
        return Emission(emitted=False, reason=str(e))

    out.mkdir(parents=True, exist_ok=True)
    n_decline = sum(1 for pr in probes if not pr.fired)
    n_variant = sum(1 for pr in probes if pr.fired)
    em.notes = [
        f"declines measured+emitted: {n_decline}",
        f"accepted variants measured+emitted: "
        f"{min(n_variant, _MAX_VARIANT_TESTS)}",
        f"repeated-metavar mismatches: {len(mismatches)}",
        "review points: law= prose, tags=_SIM default, "
        "generated names, decline coverage is probe-measured only",
    ]
    for hook in (p.check, p.derive):
        if hook is not None and re.search(
            rf"def\s+{re.escape(hook.__name__)}\s*\(", orig
        ):
            em.notes.append(
                f"hook name {hook.__name__!r} already exists in "
                "tensor.py — the patch shadows it; the reviewer "
                "must dedupe (the emitted def is the proposer's "
                "verbatim source, so identical bodies collapse)"
            )
    names = [
        f"admitted_{law}.py",
        f"test_admitted_{law}.py",
        f"admission_{law}.patch",
        f"admission_{law}.md",
    ]
    em.files = [str(out / n) for n in names]
    files = {
        names[0]: module,
        names[1]: test,
        names[2]: patch,
        names[3]: _summary_md(law, const, p, ev, em, holdout),
    }
    for fname, text in files.items():
        (out / fname).write_text(text)
    return em


def _summary_md(
    law: str, const: str, p: Any, ev: Any, em: Emission, holdout: Any
) -> str:
    """Render the human-review summary for one emitted admission."""
    files = "\n".join(f"* `{f}`" for f in em.files)
    notes = "\n".join(f"* {n}" for n in em.notes)
    return f"""# Admission artifact — `{law}` (emitted, unapplied)

Emitted by `catopt_discovery.pipeline --emit-admission {p.name}`{
        f" `--holdout {holdout}`" if holdout else ""
    }.

Candidate `{p.name}` [{p.family}] — SHIP verdict:
`fires={ev.fires} paid={ev.paid} drop={ev.cost_drop * 100:.1f}%`
`cert={"pass" if ev.cert_fail == 0 else "FAIL"}`
`enode={ev.closure_ratio:.2f}x` — cases: {", ".join(ev.fire_cases)}.

    {p.lhs!r}
        -> {p.rhs!r}

## Files

{files}

* `admission_{law}.patch` — the review diff: `{const}` + hooks +
  `SIMPLIFICATION_RULES` registration into `tensor.py`, plus the new
  test file (`git apply` to a worktree; nothing was applied).
* `admitted_{law}.py` — the same law block as a standalone importable
  module (for diffing against the library before review).
* `test_admitted_{law}.py` — the generated tests.

## Emission notes

{notes}

## Review checklist

* `law=` is a placeholder — write the equality prose.
* `tags=_SIM` is the default; an expansive law belongs in
  `CATEGORICAL_RULES` (the pipeline's closure gate makes that
  unlikely for a SHIP verdict, not impossible).
* Decline tests cover only what the emit-time probes *measured* —
  semantic declines beyond mechanical attr perturbation are still
  the reviewer's to add.
"""
