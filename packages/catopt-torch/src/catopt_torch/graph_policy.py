"""The graph-state learned policy — features derived from the e-graph.

Plan 0016 stage 7's last mile.  The ``Policy`` seam is already
consumed by the real pipeline — ``EGraph.run(..., policy=...)`` and
``Optimizer.optimize(..., policy=...)`` both consult a policy once per
saturation iteration — but the engine hands the policy a
:class:`~catopt_core.game.GameState` that carries the e-graph and the
root class and **no program features** (the engine never needs them).
:class:`~catopt_torch.learned_policy.LearnedPolicy` reads
``state.features`` and therefore cannot run there.  This module closes
that gap additively: :class:`GraphFeaturePolicy` derives the state
features itself.

**The pinned state contract.**  A learned policy is coupled to the
semantics of its features — ``project/retros/stage7-multifamily-
results.md`` records a correctness fix elsewhere silently regressing a
policy trained on the old features.  So the derivation is pinned, not
recomputed per iteration:

* the features are computed **once per e-graph**, on the *first*
  consult — iteration 0, before any rule has fired — and cached for the
  rest of the run;
* at that instant the e-graph holds exactly the exported program, so
  the features are ``compute_features(ir.root)`` — bit-identical to the
  training contract of
  :func:`catopt_core.trajectories.rule_samples` and
  :func:`~catopt_torch.learned_policy.train_rule_value`.

Recomputing features from a mid-search rewrite instead would move the
input off the trained distribution; pinning the derivation here is what
keeps the net in distribution, and the docstring is the contract any
future change to :class:`~catopt_core.features.ProgramFeatures` must
re-check.

The policy only *orders* the legal moves the law library licenses, so
it can never change what is certified (ADR 0003 invariant 5): the
extracted equivalence class and the certificate are the same as the
shipped ordering's.
"""

from __future__ import annotations

from typing import Any

from catopt_core.cost import flops_cost
from catopt_core.features import ProgramFeatures, compute_features

from catopt_torch.learned_policy import LearnedPolicy

__all__ = ["GraphFeaturePolicy"]


class GraphFeaturePolicy(LearnedPolicy):
    """A learned policy that derives its state features from the graph.

    Extends :class:`~catopt_torch.learned_policy.LearnedPolicy` — same
    net, same structural action encoding, same argmax — with a
    :meth:`features_of` that reads the state's e-graph instead of
    ``state.features``, so the policy is usable straight from
    ``EGraph.run`` / ``Optimizer.optimize``.

    ``cost_fn`` prices the root class's best member on the first
    consult (default ``flops_cost``); at iteration 0 the class holds a
    single member, so the choice does not affect the features.
    """

    name = "learned-graph"

    def __init__(
        self,
        model: Any,
        rule_by_name: Any,
        cost_fn: Any = None,
        device: str = "cpu",
    ) -> None:
        """Wrap a trained ``model`` and a rule-name index."""
        super().__init__(model, rule_by_name, device=device)
        self.cost_fn = cost_fn if cost_fn is not None else flops_cost
        self._eg: Any = None
        self._features: ProgramFeatures = compute_features(None)

    def features_of(self, state: Any) -> ProgramFeatures:
        """Return the state's program features, cached per e-graph.

        A state that already carries features (``state.features is not
        None``) is honoured verbatim, so the policy still works under
        :class:`~catopt_core.search_env.SearchEnv` and the bare
        ``GameState(None, 0, features=...)`` the training tools build.
        Otherwise the features are derived once from the root class's
        best member and cached until a different e-graph appears.

        Raises :class:`ValueError` when neither is available — a state
        with no features and no e-graph has nothing to score against.
        """
        given = getattr(state, "features", None)
        if given is not None:
            return given
        eg = getattr(state, "eg", None)
        if eg is None:
            raise ValueError(
                "GraphFeaturePolicy needs state.features or state.eg"
            )
        if eg is not self._eg:
            term = eg.extract_best(state.root_eid, self.cost_fn)
            self._features = compute_features(term)
            self._eg = eg
        return self._features
