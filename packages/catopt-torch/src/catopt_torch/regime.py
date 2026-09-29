"""The torch half of regime-adaptive selection (plan 0007).

The torch-coupled pieces moved out of ``catopt_orchestrator.regime``:
:class:`RegimeDispatch` — the ``nn.Module`` holding every executor —
and the end-to-end :func:`regime_dispatch` / :func:`build_egraph`
wrappers with their historical torch defaults.  Importing this module
also registers the ambient regime backend (the torch executor table +
this dispatch class) with the neutral frontier, so
:func:`catopt_orchestrator.regime.regime_frontier` and the historical
``catopt_torch.regime.EXECUTORS`` surface keep working unchanged.

Every dispatched executor is still the lowered form of a certified
e-graph member — see :mod:`catopt_orchestrator.regime` for the frontier
machinery and honesty contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from catopt_orchestrator.regime import (
    CARRIER_LAWS,
    EXECUTORS,
    ExecutorSpec,
    Regime,
    RegimeChoice,
    RegimeFrontier,
    architecture_label,
    architecture_signature,
    default_regimes,
    default_rules,
    footprint_cost,
    is_trace_rooted_term,
    regime_frontier,
    register_regime_backend,
)
from catopt_orchestrator.regime import (
    build_egraph as _build_egraph,
)

# Compat re-exports — the private helpers tests reach through the
# historical ``catopt_torch.regime`` path.  The redundant alias marks each
# as an intentional re-export, not an unused import.
from catopt_orchestrator.regime import (  # isort: skip
    _as_regime as _as_regime,
)
from catopt_orchestrator.regime import (  # isort: skip
    _attach_profiles as _attach_profiles,
)
from catopt_orchestrator.regime import (  # isort: skip
    _auto_executor as _auto_executor,
)
from catopt_orchestrator.regime import (  # isort: skip
    _force_carrier as _force_carrier,
)
from catopt_orchestrator.regime import (  # isort: skip
    _normalise_regimes as _normalise_regimes,
)

from catopt_torch.adapters import TorchSink, TorchSource

__all__ = [
    "CARRIER_LAWS",
    "EXECUTORS",
    "ExecutorSpec",
    "Regime",
    "RegimeChoice",
    "RegimeDispatch",
    "RegimeFrontier",
    "architecture_label",
    "architecture_signature",
    "build_egraph",
    "default_regimes",
    "default_rules",
    "footprint_cost",
    "is_trace_rooted_term",
    "regime_dispatch",
    "regime_frontier",
    "register_regime_backend",
]


def build_egraph(
    model: torch.nn.Module,
    example_input: Any,
    *,
    source: Any = None,
    rules: Any = None,
    xc: bool = True,
    max_iterations: int = 14,
    max_nodes: int = 400_000,
):
    """Export ``model`` and saturate an e-graph — torch defaults.

    Returns ``(eg, root_eid, ir, source_tensors, stats)``.

    ``source`` defaults to :class:`TorchSource` (``torch.export``);
    pass a different :class:`~catopt_core.ports.Source` to steer the
    export.  Everything else delegates to the backend-neutral
    :func:`catopt_orchestrator.regime.build_egraph` — ``rules`` selects
    the core saturating set (default the carrier law list), ``xc``
    adds the bounded cross-carrier seam tier.
    """
    if source is None:
        source = TorchSource()
    return _build_egraph(
        model,
        example_input,
        source=source,
        rules=rules,
        xc=xc,
        max_iterations=max_iterations,
        max_nodes=max_nodes,
    )


def _safe_key(name: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch == "_" else "_" for ch in name
    )


class RegimeDispatch(torch.nn.Module):
    """One set of weights, one architecture per regime.

    Holds ``{regime_name: (extracted_term, executor_module)}``.  Source
    parameters are shared objects across all executor modules — every
    form reads the same weights.  ``forward(*xs, regime=None)`` routes
    to the named regime (or the default).
    """

    def __init__(
        self,
        frontier: RegimeFrontier,
        param_values: dict | None = None,
        *,
        default: str | None = None,
    ):
        """Build the dispatch by lowering every regime's member."""
        super().__init__()
        if frontier.ir is None:
            raise ValueError(
                "frontier has no IR; pass ir= to regime_frontier to "
                "build a dispatch"
            )
        self.frontier = frontier
        ir = frontier.ir

        self._name_map: dict[str, str] = {}
        self._meta: dict[str, dict] = {}
        forms: dict[str, torch.nn.Module] = {}
        for name, ch in frontier.choices.items():
            if ch.term is None:
                continue
            spec = frontier.executors[ch.executor]
            ir_i = ir.__class__(
                root=ch.term,
                inputs=ir.inputs,
                input_names=ir.input_names,
                params=ir.params,
            )
            mod = spec.lower(ir_i, param_values)
            key = _safe_key(name)
            self._name_map[name] = key
            forms[key] = mod
            ch.engaged = spec.engaged(mod)
            self._meta[name] = {
                "term": ch.term,
                "module": mod,
                "choice": ch,
            }
        self.forms = torch.nn.ModuleDict(forms)
        self._share_params()

        names = [n for n in frontier.choices if n in self._meta]
        if not names:
            raise ValueError("no regime produced an executable form")
        self._regime = default if default is not None else names[0]
        if self._regime not in self._meta:
            raise KeyError(f"unknown default regime {default!r}")
        self.verification: dict[str, dict] | None = None

    # -- one set of weights ------------------------------------------
    @staticmethod
    def _param_map_of(mod: torch.nn.Module) -> dict | None:
        """Return the ``{ir_name: Parameter}`` map of a module.

        On the module itself for ``IRModule``, on ``.eval_mod`` for the
        specialised wrappers (scan/om executors).
        """
        pm = getattr(mod, "_param_map", None)
        if pm is None:
            inner = getattr(mod, "eval_mod", None)
            pm = (
                getattr(inner, "_param_map", None)
                if inner is not None
                else None
            )
        return pm

    def _share_params(self) -> None:
        """Rebind source parameters to one shared object per name.

        Applies across all executor modules.
        """
        shared: dict[str, torch.nn.Parameter] = {}
        for mod in self.forms.values():
            pm = self._param_map_of(mod)
            if pm is None:
                continue
            host = (
                mod
                if getattr(mod, "_param_map", None) is pm
                else mod.eval_mod
            )
            for pname in pm:
                if pname.startswith("fused_"):
                    continue  # per-module fold intermediates
                cur = pm[pname]
                if not isinstance(cur, torch.nn.Parameter):
                    continue
                if pname in shared:
                    setattr(host, pname, shared[pname])
                    pm[pname] = shared[pname]
                else:
                    shared[pname] = cur

    # -- API ---------------------------------------------------------
    @property
    def regimes(self) -> list[str]:
        """Return the dispatched regime names."""
        return list(self._meta)

    @property
    def regime(self) -> str:
        """Return the current default regime name."""
        return self._regime

    def set_regime(self, name: str) -> None:
        """Select the default regime by name."""
        if name not in self._meta:
            raise KeyError(
                f"unknown regime {name!r}; available: "
                f"{sorted(self._meta)}"
            )
        self._regime = name

    @property
    def entries(self) -> dict[str, tuple[Any, torch.nn.Module]]:
        """``{regime_name: (extracted_term, executor_module)}``."""
        return {
            n: (m["term"], m["module"]) for n, m in self._meta.items()
        }

    def executor_module(self, name: str) -> torch.nn.Module:
        """Return the executor module serving ``name``."""
        return self.forms[self._name_map[name]]

    def forward(
        self, *xs: torch.Tensor, regime: str | None = None
    ) -> torch.Tensor:
        """Run the form for ``regime`` (default: the current one)."""
        name = regime if regime is not None else self._regime
        if name not in self._meta:
            raise KeyError(
                f"unknown regime {name!r}; available: "
                f"{sorted(self._meta)}"
            )
        return self.forms[self._name_map[name]](*xs)

    # -- equivalence -------------------------------------------------
    def certificate(self, name: str):
        """Level-2 certificate: source term → this regime's member."""
        return self.frontier.certificate(name)

    def max_diff(
        self,
        reference: Any,
        *xs: torch.Tensor,
        regime: str | None = None,
    ) -> dict[str, float]:
        """Max |form(x) - reference| per regime (or one named regime)."""
        ref = reference(*xs) if callable(reference) else reference
        names = [regime] if regime is not None else self.regimes
        out: dict[str, float] = {}
        with torch.no_grad():
            for n in names:
                y = self.forward(*xs, regime=n)
                out[n] = (y - ref).abs().max().item()
        return out

    def verify(
        self, reference: Any, *xs: torch.Tensor, atol: float = 1e-9
    ) -> dict[str, dict]:
        """Check every dispatched form against a reference output.

        ``reference`` is a tensor or a callable producing it from
        ``*xs``.  Returns ``{regime: {"max_abs_diff": d, "ok": bool}}``
        and caches it on ``self.verification``.
        """
        diffs = self.max_diff(reference, *xs)
        self.verification = {
            n: {"max_abs_diff": d, "ok": d <= atol}
            for n, d in diffs.items()
        }
        return self.verification

    def report(self) -> str:
        """Render the dispatch and its verification as text."""
        lines = [self.frontier.report(), "", "built modules:"]
        for name, m in self._meta.items():
            ch = m["choice"]
            lines.append(
                f"  {name:<16} {type(m['module']).__name__:<22} "
                f"engaged={ch.engaged}"
            )
        if self.verification:
            lines.append("equivalence vs reference:")
            for n, v in self.verification.items():
                lines.append(
                    f"  {n:<16} max|Δ|={v['max_abs_diff']:.3e} "
                    f"ok={v['ok']}"
                )
        return "\n".join(lines)

    def extra_repr(self) -> str:
        """Return the ``nn.Module`` repr extras."""
        return f"regime={self._regime!r}, forms={list(self._meta)}"


def regime_dispatch(
    model: torch.nn.Module,
    example_input: Any,
    regimes: Any = None,
    *,
    rules: Any = None,
    xc: bool = True,
    max_iterations: int = 14,
    max_nodes: int = 400_000,
    default: str | None = None,
    verify: bool = True,
    atol: float = 1e-9,
    profiles: dict[str, Any] | None = None,
    calibrate: Any = None,
    source: Any = None,
    executors: Mapping[str, ExecutorSpec] | None = None,
) -> RegimeDispatch:
    """End-to-end: export → saturate → frontier → build → verify.

    Returns a :class:`RegimeDispatch` whose ``.frontier`` records every
    regime's choice.  With ``verify=True`` each form is checked against
    the model's own output on ``example_input`` (fp64 recommended).

    ``profiles`` is a ``{regime_name: profile_spec}`` map forwarded to
    :func:`~catopt_orchestrator.regime.regime_frontier` — it fills
    ``profile`` on named regimes that don't carry one.  ``calibrate``
    is a convenience for "price this model against the current
    device": ``calibrate=True`` calls
    :func:`catopt_torch.calibrate.calibrate` once and attaches the
    measured profile to every regime still lacking one; a
    ``TargetProfile`` / dict / persisted name does the same without
    measuring.  Since an explicit ``cost_fn`` always wins over a
    profile, ``calibrate`` changes *extraction* only for regimes that
    declare no cost model — elsewhere it is recorded for provenance.
    ``calibrate=None`` (the default) is the old behaviour.

    ``source`` defaults to :class:`TorchSource`
    (``torch.export``); ``executors`` defaults to the ambient
    :data:`EXECUTORS` — the torch table this module registers — pass
    a backend's ``sink.executors`` to steer lowering explicitly.
    """
    args = (
        example_input
        if isinstance(example_input, tuple)
        else (example_input,)
    )
    eg, root, ir, source_tensors, _stats = build_egraph(
        model,
        example_input,
        source=source,
        rules=rules,
        xc=xc,
        max_iterations=max_iterations,
        max_nodes=max_nodes,
    )
    regime_list = _attach_profiles(
        _normalise_regimes(regimes), profiles
    )
    if calibrate:
        pending = [r.name for r in regime_list if r.profile is None]
        if pending:
            if calibrate is True:
                from catopt_torch.calibrate import (
                    calibrate as _measure,
                )

                prof: Any = _measure()
            else:
                prof = calibrate
            regime_list = _attach_profiles(
                regime_list, {n: prof for n in pending}
            )
    frontier = regime_frontier(
        eg,
        root,
        regime_list,
        ir=ir,
        src_term=ir.root,
        executors=executors,
    )
    disp = frontier.build(param_values=source_tensors, default=default)
    if verify:
        was_training = model.training
        try:
            model.eval()
            with torch.no_grad():
                ref = model(*[a.clone() for a in args])
                disp.verify(ref, *args, atol=atol)
        finally:
            model.train(was_training)
    return disp


# ---------------------------------------------------------------------------
# Ambient registration — the torch executor table + this dispatch class
# become the defaults the neutral frontier consults (plan 0007).
# ---------------------------------------------------------------------------

register_regime_backend(
    executors=TorchSink().executors,
    dispatch=RegimeDispatch,
)
