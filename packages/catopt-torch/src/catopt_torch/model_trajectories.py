"""Supervised trajectories built from real models.

Plan 0016 stage 7.  :mod:`catopt_core.trajectories` produces the
``(features, rule_vector, rule, delta_cost, program)`` record from any
IR term; this module supplies the term from a **real torch model** —
exported through the torch source — so the learned policy trains on the
programs catopt actually optimizes (SwiGLU, attention, transformer
blocks, …) rather than only on synthetic families.

The encoding is *reused verbatim*: this is a thin bridge, not a second
trajectory format.  A policy trained here scores rules structurally
(``rule_vector``) against the program's static features, exactly as
:func:`~catopt_torch.learned_policy.train_rule_value` expects.
"""

from __future__ import annotations

from typing import Any

from catopt_core.trajectories import RuleSample, rule_samples

from catopt_torch.adapters import TorchSource

__all__ = ["model_rule_samples"]


def model_rule_samples(
    model: Any,
    x: Any,
    rules: Any,
    cost_fn: Any = None,
    source: Any = None,
) -> list[RuleSample]:
    """Return the per-rule cost deltas a real ``model`` yields.

    Exports ``model`` at ``x`` through ``source`` (default
    :class:`~catopt_torch.adapters.TorchSource`) and delegates to
    :func:`catopt_core.trajectories.rule_samples` on the exported IR
    root.  ``x`` is the example input the source traces with — a tensor
    or a positional-args tuple for a multi-input model.

    The training label the policy consumes is ``delta_cost``: the
    extracted-cost improvement the rule produces when applied *alone*,
    so ``delta_cost > 0`` is the dense ``rule improved this program``
    signal :func:`~catopt_torch.learned_policy.train_rule_value`
    classifies.
    """
    src = source if source is not None else TorchSource()
    ir, _ = src.to_ir(model, x)
    return rule_samples(ir.root, rules, cost_fn)
