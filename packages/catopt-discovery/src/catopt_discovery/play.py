"""``play`` — the optimization game as a tool.

One registry of *domains* (a board family plus its player
vocabulary), one entrypoint: pick a domain, pick arms, get the
paired-boards table.  Adding a domain is data — a case generator, a
``case -> Board`` factory, the legal-move enumerator and the
learned player's featurizer — no driver code:

.. code-block:: text

    DOMAINS[name] = Domain(
        cases=gen_cases,            # seed, n -> [case, ...]
        board_of=...,               # case -> Board
        legal=...,                  # state -> tuple[action, ...]
        featurizer=...,             # (state, action, hist) -> dict
        scripted=...,               # optional fixed-playbook arm
    )

Shipped domains:

* ``meta`` — the meta-arena: fire/saturate/declare/handle/extract
  over the mined-spec corpus (:func:`meta_player.gen_cases`);
* ``joint`` — training graphs: a forward term plus every derived
  gradient under one ``joint`` root, played on the same meta-arena
  board (the reverse handler's programs are ordinary terms);
* ``search`` — the core :class:`~catopt_core.search_env.SearchEnv`:
  one rule per step under a horizon/patience bound, adapted to the
  :class:`~catopt_discovery.engine.Board` contract;
* ``torch`` — real ``nn.Module``s lifted through
  ``catopt_torch.adapters.TorchSource`` and played under the real
  sink's ``supported_ops`` bound; the ``--deliver`` flag lowers
  the winning extraction to a verified runnable module.

Run it::

    python -m catopt_discovery.play --domain meta --train-cases 40 \
        --eval-cases 10
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from typing import Any

from catopt_core.cost.basic import count_cost
from catopt_core.ir import Const, Op, TensorType, Var
from catopt_core.laws import DEFAULT as DEFAULT_SEARCH_RULES
from catopt_core.search_env import SearchEnv

from . import engine, lawdata, training
from . import meta_arena as ma
from . import meta_player as mp
from .meta_player import _bucket
from .players import LinearPolicy

__all__ = [
    "DOMAINS",
    "Domain",
    "SearchBoard",
    "play",
]


@dataclass(frozen=True)
class Domain:
    """One board family's registration — data, not code."""

    cases: Any  # (seed, n) -> [case, ...]
    board_of: Any  # case -> Board
    legal: Any  # state -> tuple[action, ...]
    featurizer: Any  # (state, action, hist) -> dict
    features: Any = ()  # the learned arm's schema
    scripted: Any = None  # () -> player, or None
    deliver: Any = None  # (board, case, traj) -> artifact | None


def play(
    domain: str,
    case: Any,
    player: Any,
    *,
    budget: int = 24,
) -> engine.Trajectory:
    """Play one episode of *domain*/*case* under *player*."""
    d = DOMAINS[domain]
    return engine.run_episode(d.board_of(case), player, budget)


def deliver(
    domain: str,
    case: Any,
    player: Any,
    *,
    budget: int = 24,
) -> dict:
    """Play an episode on *case*; return the lowered artifact.

    The product loop: the board's winning extraction resolved by
    ``MetaArena.deliverable`` (handled ops instantiate their
    kernels, unhandled abbreviations spell back), lowered through
    the domain's sink, verified against the source model.  Domains
    without a ``deliver`` hook raise — the game score needs no
    artifact; delivery does.
    """
    d = DOMAINS[domain]
    if d.deliver is None:
        raise ValueError(f"domain {domain!r} has no deliver hook")
    board = d.board_of(case)
    traj = engine.run_episode(board, player, budget)
    return d.deliver(board, case, traj)


# ---------------------------------------------------------------------------
#  Domain: meta — the mixed board
# ---------------------------------------------------------------------------


def _meta_board(case: Any) -> ma.MetaArena:
    """Probe-arity case tuple -> MetaArena (count_cost)."""
    return mp._board(case, count_cost, 20_000)


# ---------------------------------------------------------------------------
#  Domain: joint — forward + gradients on one board
# ---------------------------------------------------------------------------


def _joint_cases(seed: int, n: int) -> list:
    """Joint programs over shape-varied forward archetypes.

    Each case joints a forward (silu / softmax-spine / linear+tanh
    at a random width) with every derived gradient under one
    ``joint`` root — the reverse handler's programs are ordinary
    terms, so the meta-arena's move set applies unchanged.
    """
    rng = random.Random(seed)
    out = []
    for i in range(n):
        d, b = 4 * rng.choice((1, 2)), 4 * rng.choice((1, 2))
        x = Var("x", TensorType((b, d)))
        w = Var("w", TensorType((d, d)))
        archetype = rng.randrange(4)
        if archetype == 0:
            fwd = Op.make("mul", x, Op.make("sigmoid", x))
        elif archetype == 1:
            fwd = Op.make("softmax", x, dim=-1)
        elif archetype == 2:
            fwd = Op.make(
                "div",
                Op.make("exp", x),
                Op.make("sum", Op.make("exp", x), dim=1, keepdim=True),
            )
        else:
            fwd = Op.make("tanh", Op.make("matmul", x, w))
        cot = training._cotangent_for(fwd)
        grads = training.backward(fwd, cotangent=cot)
        joint = Op.make(
            "joint",
            fwd,
            *[grads[k] for k in sorted(grads)],
            validate=False,
        )
        out.append((f"joint{i}:{archetype}", joint))
    return out


def _joint_board(case: Any) -> ma.MetaArena:
    """Board a joint program under the meta-arena's moves."""
    return ma.MetaArena(case[1], cost_fn=count_cost)


# ---------------------------------------------------------------------------
#  Domain: torch — real modules under the sink's bound
# ---------------------------------------------------------------------------


def _torch_cases(seed: int, n: int) -> list:
    """Small real ``nn.Module``s lifted to IR — the product board.

    Each case is a torch module exported through
    ``catopt_torch.adapters.TorchSource`` and played under the real
    sink's ``supported_ops`` bound — kernel claims price at sink
    costs, not ambient vocabulary.  Modules that fail to export are
    skipped honestly.  Torch is imported lazily — the registry
    itself stays torch-free.
    """
    import torch
    import torch.nn as nn
    from catopt_torch.adapters import TorchSink, TorchSource

    class _SpelledSiLU(nn.Module):
        """x·sigmoid(x) spelled out — the claim move's live site."""

        def forward(self, x):
            """Spell ``x·sigmoid(x)`` — no fused op named."""
            return x * torch.sigmoid(x)

    rng = random.Random(seed)
    supported = TorchSink().supported_ops
    src = TorchSource()
    builders = [
        ("spelled_silu", lambda d: _SpelledSiLU()),
        (
            "mlp",
            lambda d: nn.Sequential(
                nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d)
            ),
        ),
        (
            "relu_lin",
            lambda d: nn.Sequential(nn.Linear(d, d), nn.ReLU()),
        ),
        ("lin", lambda d: nn.Linear(d, d)),
        (
            "softsign_mlp",
            lambda d: nn.Sequential(nn.Linear(d, d), nn.Softsign()),
        ),
    ]
    out = []
    for i in range(n):
        d = 4 * rng.choice((1, 2, 4))
        kind, mk = builders[rng.randrange(len(builders))]
        torch.manual_seed(seed * 10_007 + i)
        model = mk(d).eval()
        example = torch.randn(4, d)
        ir, leaves = src.to_ir(model, example)
        out.append(
            (
                f"torch{i}:{kind}",
                ir.root,
                None,
                (),
                supported,
                {
                    "ir": ir,
                    "leaves": leaves,
                    "input": example,
                    "model": model,
                },
            )
        )
    return out


def _torch_board(case: Any) -> ma.MetaArena:
    """Board a torch-lifted program under the real sink bound."""
    rules = case[2] if len(case) > 2 else None
    return ma.MetaArena(
        case[1],
        rules,
        cost_fn=count_cost,
        supported=case[4] if len(case) > 4 else None,
    )


def _torch_deliver(board: ma.MetaArena, case: Any, traj: Any) -> dict:
    """Lower the winning extraction and verify it against the model.

    ``deliverable`` resolves handled claims to their kernels and
    unhandled abbreviations to spelled form; ``TorchSink.lower``
    rebuilds the module from it; ``verify`` runs both on the case's
    example input.  The game score becomes a delivered artifact.
    """
    from catopt_core.ir import IR
    from catopt_torch.adapters import TorchSink

    ex = case[5]
    best = board.eg.extract_best(board.root, board.feasible_cost)
    if best is None:
        return {"delivered": False, "reward": traj.total}
    term = board.deliverable(best)
    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    sink = ex.get("sink") or TorchSink()
    module = sink.lower(ir, params=ex["leaves"])
    rep = sink.verify(ex["model"], module, (ex["input"],))
    t = traj.terminal
    return {
        "delivered": True,
        "module": module,
        "verified": rep.passed,
        "max_abs": rep.max_abs,
        "cost": getattr(t, "cost", None),
        "cert": getattr(t, "certificate_ok", None),
        "reward": traj.total,
    }


# ---------------------------------------------------------------------------
#  Domain: gen — generated kernels over real torch modules
# ---------------------------------------------------------------------------
#
#  The "discovery builds lowerings" story end to end: spell fusion
#  patterns the ambient handler table has no name for (``x·relu(x)``,
#  ``x·gelu(x)``, …) plus stock-model pieces; the board mints a gen
#  handler + kernel per occurring elementwise subterm; ``claim``
#  binds it; ``deliver`` lowers it through a ``TorchSink`` whose op
#  table includes the generated binding — a runnable module calling
#  a kernel nobody wrote.


def _fusion_pattern(t: Any, names: dict) -> Any:
    """Return the elementwise-closure pattern at *t*, leaves abstracted.

    A subterm fuses iff its *head* is elementwise — children below
    the elementwise boundary (matmuls, reductions, …) become bound
    metavariables, consistently named by subtree identity.  So
    ``mul(y, relu(y))`` over ``y = mm(…)`` yields
    ``("mul", "X1", ("relu", "X1"))`` — ``X1`` binds the whole
    folded matmul, and the generated kernel computes it once.
    """
    from catopt_core.ir import Const

    from catopt_discovery import genkernel as gk

    if isinstance(t, Op) and t.op in gk.ELEMENTWISE:
        kids = [_fusion_pattern(a, names) for a in t.args]
        if t.attrs:
            kids.append(dict(t.attrs))
        return (t.op, *kids)
    if isinstance(t, Const):
        return t.value
    key = repr(t)
    if key not in names:
        names[key] = f"X{len(names) + 1}"
    return names[key]


def _fusion_sites(term: Any) -> list:
    """Elementwise-connex subterms of size ≥2, leaves abstracted.

    Each entry is a *pattern tuple* (metavar leaves for the
    non-elementwise boundary), the shape ``gen_handlers`` consumes.
    """
    from catopt_discovery import genkernel as gk

    out: list = []

    def walk(t: Any) -> None:
        if isinstance(t, Op) and t.op in gk.ELEMENTWISE:
            spec = _fusion_pattern(t, {})
            ops = {e for e in _flat_ops(spec)}
            if len(ops) >= 2:
                out.append(spec)
        if isinstance(t, Op):
            for a in t.args:
                walk(a)

    walk(term)
    return out


def _flat_ops(spec: Any):
    """Yield op names inside a fusion pattern tuple."""
    if isinstance(spec, (tuple, list)) and spec:
        yield spec[0]
        for e in spec[1:]:
            if not isinstance(e, dict):
                yield from _flat_ops(e)


def _gen_cases(seed: int, n: int) -> list:
    """Torch modules over fusion-spelling and stock-model builders.

    The fusion builders spell pointwise patterns the ambient
    handler table has no fused op for — the candidates ``claim``
    can only reach through a *generated* kernel.  Stock pieces
    (mlp/attn chains) give the board real shape.
    """
    import torch
    import torch.nn as nn
    from catopt_torch.adapters import TorchSource

    from catopt_discovery import genkernel as gk
    from catopt_discovery import lawdata
    from catopt_discovery.meta_arena import _canon_concrete

    class _Fuse(nn.Module):
        def __init__(self, kind: str) -> None:
            super().__init__()
            self.kind = kind

        def forward(self, x):
            """Spell the pointwise pattern — no fused op named."""
            if self.kind == "mulrelu":
                return x * torch.relu(x)
            if self.kind == "mulgelu":
                return x * torch.nn.functional.gelu(x)
            if self.kind == "mulsoftplus":
                return x * torch.nn.functional.softplus(x)
            if self.kind == "bigfuse":
                return (
                    x * torch.relu(x)
                    + torch.sigmoid(x)
                    - torch.tanh(x) * torch.abs(x)
                    + x * x
                )
            return x * torch.tanh(x)

    class _SwigluMLP(nn.Module):
        """Stock llama-MLP shape: ``down(silu(gate) * up)`` — spelled."""

        def __init__(self, d: int = 64) -> None:
            super().__init__()
            self.gate = nn.Linear(d, d * 2, bias=False)
            self.up = nn.Linear(d, d * 2, bias=False)
            self.down = nn.Linear(d * 2, d, bias=False)

        def forward(self, x):
            """Spell the swiglu — the site a claim can name."""
            g = self.gate(x)
            return self.down(g * torch.sigmoid(g) * self.up(x))

    class _ChainMLP(nn.Module):
        """A spelled matmul chain + pointwise tail — the reassoc story."""

        def __init__(self, d: int = 64) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.b = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.c = nn.Parameter(torch.randn(d, d) / d**0.5)

        def forward(self, x):
            """Three spelled mms, then a pointwise tail."""
            y = x @ self.a @ self.b @ self.c
            return y * torch.relu(y)

    class _ChainFuse(nn.Module):
        """Fold-able mm chain + a 6-op pointwise tail needing gen.

        The folded chain is a laws win; the tail's fused form exists
        only via ``claim(gen_*)`` — the win that needs the generated
        kernel.
        """

        def __init__(self, d: int = 512) -> None:
            super().__init__()
            self.a = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.b = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.c = nn.Parameter(torch.randn(d, d) / d**0.5)

        def forward(self, x):
            """Chain, then a long elementwise tail over the result."""
            y = x @ self.a @ self.b @ self.c
            return (
                y * torch.relu(y)
                - torch.nn.functional.softplus(y)
                + torch.tanh(y * y)
                * torch.nn.functional.softplus(y * torch.sigmoid(y))
            )

    rng = random.Random(seed)
    src = TorchSource()
    covered = {
        _canon_concrete(h["pattern"]) for h in lawdata.HANDLERS.values()
    }
    builders = [
        ("mulrelu", lambda: _Fuse("mulrelu"), (256, 512)),
        ("mulgelu", lambda: _Fuse("mulgelu"), (256, 512)),
        ("mulsoftplus", lambda: _Fuse("mulsoftplus"), (256, 512)),
        ("multanh", lambda: _Fuse("multanh"), (256, 512)),
        ("bigfuse", lambda: _Fuse("bigfuse"), (256, 512)),
        ("swiglu_mlp", _SwigluMLP, (16, 64)),
        ("chain_mlp", _ChainMLP, (16, 64)),
        ("chainfuse", _ChainFuse, (256, 512)),
        (
            "mlp",
            lambda d=64: nn.Sequential(
                nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d)
            ),
            (8, 64),
        ),
    ]
    out = []
    for i in range(n):
        name, build, shape = builders[rng.randrange(len(builders))]
        model = build().eval()
        gen = torch.Generator().manual_seed(seed * 997 + i)
        x = torch.randn(*shape, generator=gen)
        ir, leaves = src.to_ir(model, x)
        sites = _fusion_sites(ir.root)
        genh = gk.gen_handlers(sites, covered=covered)
        sink = gk.gen_sink(genh, compile_kernels=False)
        handlers = dict(lawdata.HANDLERS) | genh
        out.append(
            (
                f"gen{i}:{name}",
                ir.root,
                {"gen": len(genh)},
                name,
                sink.supported_ops,
                {
                    "ir": ir,
                    "leaves": leaves,
                    "model": model,
                    "input": x,
                    "sink": sink,
                    "handlers": handlers,
                    "gen_handlers": genh,
                },
            )
        )
    return out


def _gen_board(case: Any) -> ma.MetaArena:
    """Board a gen case under its generated handler table."""
    from catopt_core.cost.basic import count_cost

    from catopt_discovery import genkernel as gk

    ex = case[5]

    def novel(op: str) -> bool:
        return op.startswith(("claim_", "foldabs_", "gen_"))

    return ma.MetaArena(
        case[1],
        handlers=ex["handlers"],
        supported=case[4],
        corpus=[case[1]],
        cost_fn=gk.novel_cost(count_cost, novel, 0.5),
    )


def _probe_play(board: Any) -> Any:
    """Play the maximal automated line; return the best extraction.

    Every claim offer, then ``saturate`` (the shipped laws —
    reassociation &c.), then ``extract``.
    """
    from catopt_discovery import meta_arena as ma

    st = board.observe()
    for a in [a for a in ma.legal_actions(st) if a.op == "claim"]:
        board.step(a)
    board.step(ma.Action.saturate())
    board.step(ma.Action.extract())
    return board.eg.extract_best(board.root, board.feasible_cost)


def gen_probe(case: Any, *, budget: int = 12, reps: int = 200) -> dict:
    """Measure the novelty claim: baseline vs generated-kernel module.

    For every claim the board offers: play it, extract, deliver
    through the case's gen sink (bindings compiled — the fused
    kernel is a real inductor graph), verify numerically against
    the source module, and time both sides.  The row answers the
    product question: did the search produce a *runnable module
    that is faster* than the spelled source, not just a new
    spelling.
    """
    import torch

    from catopt_discovery import genkernel as gk

    # measurement integrity: each gen kernel is a distinct code
    # object recompiled under varying global state; dynamo's
    # default limit (8) would silently degrade late compiles —
    # and late *baseline* compiles — to eager.  Raise it so both
    # sides are measured as compiled, not corrupted by the cache.
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)

    ex = case[5]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    board = _gen_board(case)
    tags = list(board.observe().claim_tags)
    best = _probe_play(board)
    rows: list[dict] = []
    base_model = ex["model"].to(device)
    xin = (
        tuple(t.to(device) for t in ex["input"])
        if isinstance(ex["input"], (tuple, list))
        else ex["input"].to(device)
    )
    try:
        t_base = _time_module(base_model, xin, reps)
    except Exception as exc:
        return {
            "case": case[0],
            "delivered": False,
            "note": f"baseline failed: {type(exc).__name__}: {exc}",
        }
    if best is None:
        return {"case": case[0], "delivered": False, "claims": tags}
    import torch as _torch

    compiled = _torch.compile(base_model)
    t_comp = _time_module(compiled, xin, reps)
    sink = gk.hybrid_sink(ex["gen_handlers"], compile_kernels=True)
    term = board.deliverable(best)
    from catopt_core.ir import IR

    ir = IR(
        root=term,
        inputs=list(ex["ir"].inputs),
        input_names=set(ex["ir"].input_names),
        params=dict(ex["ir"].params),
    )
    try:
        rep, t_gen = _probe_delivery(
            sink, ir, ex, base_model, xin, device, reps
        )
    except Exception as exc:
        return {
            "case": case[0],
            "claims": tags,
            "delivered": False,
            "note": f"deliver failed: {type(exc).__name__}: {exc}",
            "baseline_us": t_base,
            "compiled_us": t_comp,
        }
    rows.append(
        {
            "case": case[0],
            "claims": tags,
            "delivered": True,
            "verified": rep.passed,
            "max_abs": rep.max_abs,
            "cost": getattr(board.observe(), "cost", float("nan")),
            "baseline_us": t_base,
            "compiled_us": t_comp,
            "gen_us": t_gen,
            "speedup": round(t_base / t_gen, 3) if t_gen else None,
            "vs_compiled": round(t_comp / t_gen, 3) if t_gen else None,
            "device": device,
        }
    )
    return rows[0]


def _probe_delivery(
    sink: Any,
    ir: Any,
    ex: dict,
    base_model: Any,
    xin: Any,
    device: str,
    reps: int,
) -> Any:
    """Lower → verify → time the delivered module; returns (rep, µs)."""
    module = sink.lower(ir, params=ex["leaves"]).to(device)
    rep = sink.verify(
        base_model,
        module,
        xin if isinstance(xin, (tuple, list)) else (xin,),
    )
    return rep, _time_module(module, xin, reps)


def _zoo_case(wl: Any) -> Any:
    """Build a held-out zoo workload as a gen-domain case tuple."""
    from catopt_torch.adapters import TorchSource

    from catopt_discovery import genkernel as gk
    from catopt_discovery.meta_arena import _canon_concrete

    model, feed = wl.build()
    model = model.double()  # the intake convention is fp64 feeds
    ir, leaves = TorchSource().to_ir(model, feed)
    sites = _fusion_sites(ir.root)
    covered = {
        _canon_concrete(h["pattern"]) for h in lawdata.HANDLERS.values()
    }
    genh = gk.gen_handlers(sites, covered=covered)
    # the measured-cost referee: mint only claims whose generated
    # kernel actually beats its spelled site on the real values;
    # the env must bind the input vars too — a param-only env makes
    # every site touching the feed eval-fail and silently skip.
    feeds = tuple(feed) if isinstance(feed, (tuple, list)) else (feed,)
    var_env = dict(leaves)
    var_env.update(
        {v.name: t for v, t in zip(ir.inputs, feeds, strict=False)}
    )
    genh = gk.profitable(genh, ir.root, var_env, reps=20)
    sink = gk.gen_sink(genh, compile_kernels=False)
    return (
        f"zoo:{wl.name}",
        ir.root,
        {"gen": len(genh)},
        wl.name,
        sink.supported_ops,
        {
            "ir": ir,
            "leaves": leaves,
            "model": model,
            "input": feed,
            "sink": sink,
            "handlers": dict(lawdata.HANDLERS) | genh,
            "gen_handlers": genh,
        },
    )


def _zoo_rows() -> None:
    """Print the zoo probe rows — the ``--zoo`` CLI arm."""
    for row in zoo_probe():
        print(f"zoo[{row['case']}]: {row}")  # stdout-compat


def zoo_probe(*, reps: int = 100) -> list[dict]:
    """``gen_probe`` over the held-out model zoo — the real corpus.

    Untraceable workloads are reported honestly rather than dropped
    silently; each surviving row is the full claim → deliver →
    verify → time table vs eager *and* ``torch.compile``.
    """
    from catopt_discovery import zoo as _zoo

    rows: list[dict] = []
    for wl in _zoo.zoo():
        try:
            case = _zoo_case(wl)
        except Exception as exc:
            rows.append(
                {
                    "case": f"zoo:{wl.name}",
                    "delivered": False,
                    "note": (
                        f"export declined: {type(exc).__name__}: {exc}"
                    ),
                }
            )
            continue
        rows.append(gen_probe(case, reps=reps))
    return rows


def _time_module(model: Any, x: Any, reps: int) -> float:
    """Median per-call latency (µs) of ``model(*x)`` on its device.

    ``x`` is a tuple of positional inputs — a single-tensor feed is
    ``(x,)``.
    """
    import torch

    args = x if isinstance(x, (tuple, list)) else (x,)
    dev = args[0].device
    for _ in range(10):
        model(*args)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    import time

    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model(*args)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


# ---------------------------------------------------------------------------
#  The scan-tier probe — generated fold kernel vs today's delivery
# ---------------------------------------------------------------------------


def _spelled_scan(a: Any, b: Any, h: Any) -> Any:
    """Fold the torch-loop spelling: ``h_t = a_t*h + b_t`` (axis 1)."""
    for t in range(a.shape[1]):
        h = a.select(1, t) * h + b.select(1, t)
    return h


def _scan_carrier_ir(batch: int, steps: int, width: int) -> Any:
    """``applyd(<balanced affd_compose tree>, h)`` over select leaves.

    The form ``affd`` lifts leave behind — per-step ``aff_diag``
    leaves bracketed into a balanced compose tree (in-order leaves
    are reverse-chronological, so later steps sit in the LEFT
    subtree: ``affd_compose(f, g)`` applies ``g`` first).
    """
    from catopt_core.ir import IR

    a = Var("a", TensorType((batch, steps, width)))
    b = Var("b", TensorType((batch, steps, width)))
    h = Var("h", TensorType((batch, width)))
    leaves = [
        Op.make(
            "aff_diag",
            Op.make("select", a, dim=1, index=t),
            Op.make("select", b, dim=1, index=t),
        )
        for t in range(steps)
    ]

    def tree(ls: list) -> Any:
        if len(ls) == 1:
            return ls[0]
        k = len(ls) // 2
        return Op.make("affd_compose", tree(ls[k:]), tree(ls[:k]))

    return IR(
        root=Op.make("applyd", tree(leaves), h),
        inputs=[a, b, h],
        input_names={"a", "b", "h"},
        params={},
    )


#: Default probe shapes — AdaLNBlock-style ``(B, T, d)`` fp64 scans.
_SCAN_PROBE_SHAPES = ((2, 64, 128), (8, 128, 512))


def scan_probe(
    *,
    shapes: Any = None,
    reps: int = 100,
    fused: bool = True,
) -> list[dict]:
    """Time the generated scan kernel against today's delivery paths.

    The honest-number arm for the scan tier: per ``(B, T, d)`` shape
    — fp64 CUDA, AdaLNBlock-style token axis — the row reports median
    µs for the spelled torch loop (eager and ``torch.compile``-fused),
    the carrier executor path (generic ``TorchSink`` tuple-passing
    eval of the ``affd_compose`` tree, the level-batched
    ``to_batched_scan_module`` slot-gather schedule, and — with
    *fused* — its ``scan_fused`` compiled schedule), and the
    generated Triton fold kernel from
    :func:`~catopt_discovery.genkernel.scan_triton_bindings`.
    Every arm is verified against the spelled loop before timing.
    On a host without CUDA+triton the row reports ``delivered:
    False`` honestly.
    """
    import torch
    from catopt_carriers.scan_lower import to_batched_scan_module
    from catopt_torch.adapters import TorchSink

    from catopt_discovery import genkernel as gk

    if shapes is None:
        shapes = _SCAN_PROBE_SHAPES
    if not (torch.cuda.is_available() and _has_triton()):
        return [{"delivered": False, "note": "needs cuda + triton"}]
    _dyno_cfg: Any = torch._dynamo.config
    _dyno_cfg.recompile_limit = max(int(_dyno_cfg.recompile_limit), 256)
    kern = gk.scan_triton_bindings(
        {
            "s": {
                "pattern": ("applyd", ("aff_diag", "A", "B"), "H"),
                "kernel": "sc_probe",
                "args": ("A", "B", "H"),
            }
        }
    )["sc_probe"]
    compiled = torch.compile(_spelled_scan)
    dt, dev = torch.float64, "cuda"
    rows: list[dict] = []
    for batch, steps, width in shapes:
        a = torch.rand(batch, steps, width, device=dev, dtype=dt)
        a = a * 0.5 + 0.3
        b = torch.randn(batch, steps, width, device=dev, dtype=dt)
        h = torch.randn(batch, width, device=dev, dtype=dt)
        want = _spelled_scan(a, b, h)
        ir = _scan_carrier_ir(batch, steps, width)
        generic = TorchSink().lower(ir, params={})
        batched = to_batched_scan_module(ir)
        arms: dict[str, Any] = {
            "spelled": _spelled_scan,
            "spelled_compiled": compiled,
            "executor_generic": generic,
            "executor_batched": batched,
            "triton_scan": kern,
        }
        if fused:
            arms["executor_fused"] = to_batched_scan_module(
                ir, fused="compile"
            )
        row: dict[str, Any] = {
            "shape": (batch, steps, width),
            "delivered": True,
        }
        for name, fn in arms.items():
            torch.testing.assert_close(
                fn(a, b, h), want, atol=1e-9, rtol=1e-9
            )
            row[f"{name}_us"] = _time_module(fn, (a, b, h), reps)
        row["vs_spelled"] = round(
            row["spelled_us"] / row["triton_scan_us"], 3
        )
        rows.append(row)
    return rows


def _has_triton() -> bool:
    """Check triton is importable (the codegen tiers' runtime)."""
    import importlib.util

    return importlib.util.find_spec("triton") is not None


# ---------------------------------------------------------------------------
#  Domain: search — SearchEnv adapted to the Board contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SearchState:
    """The search board's observable: rule names, cursor, terminator."""

    actions: tuple[str, ...]
    ops: frozenset
    steps: int
    done: bool
    cost: float
    baseline: float


class SearchBoard:
    """``SearchEnv`` under the :class:`engine.Board` contract.

    Actions are rule names; ``observe`` returns a
    :class:`_SearchState`; ``step`` fires the rule once and reports
    the env's normalized cost improvement as reward.  Episode ends
    at the env's horizon/patience, or when the player drains.
    """

    def __init__(
        self,
        term: Any,
        rules: Any,
        *,
        horizon: int = 12,
        patience: int | None = None,
    ) -> None:
        """Bind the program, rules, and episode bounds.

        *patience* defaults to the ruleset size — a playbook that
        tries every rule once must not stall out mid-way; the
        episode still ends at *horizon* or when the player drains.
        """
        rules = tuple(rules)
        self.env = SearchEnv(
            term,
            rules,
            cost_fn=count_cost,
            horizon=horizon,
            patience=len(rules) if patience is None else patience,
        )
        self._h = horizon
        self._p = patience
        self._steps = 0
        self._done = True
        self._baseline = float("inf")
        self._ops = frozenset(t.op for t in _ops_walk(term))

    def observe(self) -> _SearchState:
        """Return the board view — reset lazily on first look."""
        if self._done and self._steps == 0:
            self.env.reset()
            self._done = False
            self._baseline = self.env.cost
        return _SearchState(
            actions=self.env.action_names,
            ops=self._ops,
            steps=self._steps,
            done=self._done,
            cost=self.env.cost,
            baseline=self._baseline,
        )

    def step(self, action: Any) -> tuple[_SearchState, engine.Report]:
        """Fire one rule; report the cost improvement as reward."""
        self.observe()  # ensure reset
        name = (
            action if isinstance(action, str) else action.params["rule"]
        )
        res = self.env.step(name)
        self._steps += 1
        self._done = res.done
        return self.observe(), engine.Report(
            action, reward=res.reward, terminal=res.done, cost=res.cost
        )


def _search_cases(seed: int, n: int) -> list:
    """Rule-fire boards — half the pool carries a paying fold.

    ``add(x,0)`` / ``mul(x,1)`` / ``neg(neg(x))`` collapse under
    the identity laws (cost→0); the silu spelling folds to the
    fused op; the rest are honest flat boards — the corpus the
    stoppable-vs-search signal lives on.
    """
    rng = random.Random(seed)
    x = Var("x", TensorType((4, 4)))
    y = Var("y", TensorType((4, 4)))
    pool = [
        Op.make("mul", x, Op.make("sigmoid", x)),
        Op.make("add", Op.make("mul", x, y), Op.make("matmul", x, y)),
        Op.make("mul", x, x),
        Op.make("div", x, Op.make("add", Op.make("abs", x), Const(1))),
        Op.make("add", x, Const(0)),
        Op.make("mul", x, Const(1)),
        Op.make("neg", Op.make("neg", x)),
    ]
    return [
        (f"search{i}", pool[rng.randrange(len(pool))]) for i in range(n)
    ]


def _search_board(case: Any) -> SearchBoard:
    """Board a search case under the rule-fire game."""
    return SearchBoard(case[1], DEFAULT_SEARCH_RULES)


def _ops_walk(term: Any):
    """Yield every Op node reachable in *term* (args-first)."""
    if isinstance(term, Op):
        yield term
        for a in term.args:
            yield from _ops_walk(a)


def _search_legal(state: _SearchState) -> tuple:
    """Every rule name is a legal move until the env says done."""
    return state.actions


def _search_featurizer(
    state: _SearchState, action: Any, _hist: Any
) -> dict:
    """Minimal row: rule bucket plus the board summary."""
    return {
        "bias": 1.0,
        "st:steps": state.steps / 8.0,
        "st:improve": max(
            0.0, 1.0 - state.cost / max(state.baseline, 1e-9)
        ),
        f"a:r:{_bucket(action, 'rule'):02d}": 1.0,
    }


SEARCH_FEATURES: tuple = (
    "bias",
    "st:steps",
    "st:improve",
    *(
        f"a:r:{b:02d}"
        for b in range(lawdata.META_ARENA_HASH_BUCKETS["rule"])
    ),
)


def _random_policy(legal: Any) -> Any:
    """Return a uniform player over any domain's legal set."""

    class _R:
        def __init__(self, seed: int) -> None:
            self._r = random.Random(seed)

        def __call__(self, state: Any) -> Any | None:
            acts = legal(state)
            return acts[self._r.randrange(len(acts))] if acts else None

    return _R


def _scripted_search() -> Any:
    """Fire each rule once, matching-head-first — the playbook.

    Rules whose LHS root op occurs in the program go first (the
    only ones that can change the graph); the rest are dead budget.
    """
    heads = {
        r.name: getattr(r.lhs, "op", None) for r in DEFAULT_SEARCH_RULES
    }
    played: set = set()

    def go(state: _SearchState) -> Any | None:
        if state.done:
            return None
        for name in state.actions:
            if name not in played and heads.get(name) in state.ops:
                played.add(name)
                return name
        for name in state.actions:
            if name not in played:
                played.add(name)
                return name
        return None

    return go


DOMAINS: dict[str, Domain] = {
    "meta": Domain(
        cases=mp.gen_cases,
        board_of=_meta_board,
        legal=ma.legal_actions,
        featurizer=mp.featurize,
        features=lawdata.META_ARENA_FEATURES,
        scripted=ma.ScriptedPlayer,
    ),
    "joint": Domain(
        cases=_joint_cases,
        board_of=_joint_board,
        legal=ma.legal_actions,
        featurizer=mp.featurize,
        features=lawdata.META_ARENA_FEATURES,
        scripted=ma.ScriptedPlayer,
    ),
    "torch": Domain(
        cases=_torch_cases,
        board_of=_torch_board,
        legal=ma.legal_actions,
        featurizer=mp.featurize,
        features=lawdata.META_ARENA_FEATURES,
        scripted=ma.ScriptedPlayer,
        deliver=_torch_deliver,
    ),
    "search": Domain(
        cases=_search_cases,
        board_of=_search_board,
        legal=_search_legal,
        featurizer=_search_featurizer,
        features=SEARCH_FEATURES,
        scripted=_scripted_search,
    ),
    "gen": Domain(
        cases=_gen_cases,
        board_of=_gen_board,
        legal=ma.legal_actions,
        featurizer=mp.featurize,
        features=lawdata.META_ARENA_FEATURES,
        scripted=ma.ScriptedPlayer,
        deliver=_torch_deliver,
    ),
}


def _arms(domain: Domain, player: LinearPolicy, seed: int) -> dict:
    """Build the arm set for one domain: baselines + learned pair."""
    rnd = _random_policy(domain.legal)
    arms: dict[str, Any] = {
        "random": lambda i: rnd(seed + 10_000 + i),
        "learned": lambda i: player.frozen(seed + 20_000 + i),
        "learned-greedy": lambda i: player.frozen(
            seed + 30_000 + i, greedy=True
        ),
    }
    if domain.scripted is not None:
        arms = {"scripted": lambda i: domain.scripted(), **arms}
    return arms


def _table(table: dict[str, list[dict]]) -> str:
    """Render per-arm means over eval rows — generic edition."""
    head = f"{'player':<15} {'cases':>6} {'reward':>9} {'cost':>7} {'cert':>5} {'steps':>6}"
    lines = [head, "-" * len(head)]
    for name, rows in table.items():
        n = max(len(rows), 1)
        costs = [r["cost"] for r in rows if r["cost"] is not None]
        certs = sum(1 for r in rows if r["cert"])
        lines.append(
            f"{name:<15} {len(rows):>6} "
            f"{sum(r['reward'] for r in rows) / n:>9.2f} "
            f"{(sum(costs) / len(costs)) if costs else float('nan'):>7.2f} "
            f"{certs:>5} "
            f"{sum(r['steps'] for r in rows) / n:>6.1f}"
        )
    return "\n".join(lines)


def _extras(args: Any, d: Any, ev: list, player: Any) -> None:
    """Run the ``--deliver`` / ``--probe`` / ``--zoo`` report arms."""
    if args.deliver:
        arm = (
            d.scripted() if d.scripted is not None else player.frozen(0)
        )
        res = deliver(args.domain, ev[0], arm, budget=args.budget)
        shown = {k: v for k, v in res.items() if k != "module"}
        print(f"deliver[{ev[0][0]}]: {shown}")  # stdout-compat
    if args.probe:
        for case in ev:
            if DOMAINS[args.domain] is DOMAINS["gen"]:
                print(  # stdout-compat
                    f"probe[{case[0]}]: {gen_probe(case)}"
                )
    if args.zoo:
        _zoo_rows()
    if args.scan_probe:
        for row in scan_probe():
            print(
                f"scan[{row.get('shape', '-')}]: {row}"
            )  # stdout-compat


def main(argv: list[str] | None = None) -> int:
    """Train the learned arm on one case stream, eval on another."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--domain", choices=sorted(DOMAINS), default="meta")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cases", type=int, default=40)
    ap.add_argument("--eval-cases", type=int, default=10)
    ap.add_argument("--budget", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument(
        "--deliver",
        action="store_true",
        help="lower + verify the first eval case's extraction",
    )
    ap.add_argument(
        "--probe",
        action="store_true",
        help="run the gen domain's claim→deliver→time probe per eval case",
    )
    ap.add_argument(
        "--zoo",
        action="store_true",
        help="run the claim→deliver→time probe over the held-out zoo",
    )
    ap.add_argument(
        "--scan-probe",
        action="store_true",
        help=(
            "time the generated Triton scan kernel against the "
            "spelled loop and the carrier executor path (CUDA)"
        ),
    )
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args(argv)
    if sys.getrecursionlimit() < 40_000:
        sys.setrecursionlimit(40_000)

    d = DOMAINS[args.domain]
    train = d.cases(args.seed, args.train_cases)
    ev = d.cases(args.seed + 7919, args.eval_cases)
    player = LinearPolicy(
        seed=args.seed,
        legal=d.legal,
        featurizer=d.featurizer,
        features=d.features,
        temperature=args.temperature,
        epsilon=args.epsilon,
    )
    totals = engine.train_policy(
        player, train, d.board_of, budget=args.budget
    )
    table = engine.evaluate(
        _arms(d, player, args.seed), ev, d.board_of, budget=args.budget
    )
    print(f"== play — domain {args.domain} ==")  # stdout-compat
    print(_table(table))  # stdout-compat
    decade = max(len(totals) // 10, 1)
    means = [
        round(sum(totals[i : i + decade]) / decade, 2)
        for i in range(0, len(totals) - decade + 1, decade)
    ]
    print(f"train decade means: {means}")  # stdout-compat
    _extras(args, d, ev, player)
    if args.json:
        import json

        with open(args.json, "w") as f:
            json.dump(
                {"table": table, "train_totals": totals}, f, indent=1
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
