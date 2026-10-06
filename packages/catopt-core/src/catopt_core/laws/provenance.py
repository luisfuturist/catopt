"""Law provenance — which shipped rewrites are machine-admitted.

The library mixes two authorship classes and the yield measurement
(``project/retros/human-bar.md``) needs to tell them apart without
editing the laws themselves:

* **human-authored** — a law a person wrote into this package
  (including laws whose *discovery* was machine-assisted but which
  were hand-integrated before the admission machinery existed:
  ``select_mul``, ``softmax_fold``, ``glu_fold``, ``rms_norm_fold``);
* **machine-admitted** — an object the discovery side produced
  (``catopt_discovery.object_synthesis`` constructions or
  census/proposal candidates), cleared the admission gauntlet
  (``catopt_discovery.evidence.run_gauntlet``) and was then promoted
  verbatim into the library
  (``project/retros/promoted-laws.md``).

The marker is **registry-level**, not a per-law field: a ``Rewrite``
does not carry a provenance slot, so the classification lives here as
two name sets — :data:`MACHINE_SHIPPED` (the promoted laws now in
:data:`~catopt_core.laws.ALL_RULES`) and :data:`MACHINE_STORED`
(gauntlet-cleared objects that stayed in the evidence store,
unshipped).  :func:`law_provenance` is *total* over rule names —
every shipped law resolves, and a name unknown to both sets classifies
as human-authored (the default authorship of the library).

The committed ``tools/evidence.db`` holds no ``lemmas`` rows — the
admission runs recorded their objects in ephemeral stores — so
:data:`MACHINE_STORED` is the honest ledger of *names* the committed
tests pin as ``usable`` (``tests/test_discovery_synthesis.py`` /
``test_discovery_gauntlet.py`` / ``test_discovery_autocond.py``)
plus the objects the promotion audit records as admitted
(``project/retros/promoted-laws.md``).
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "MACHINE_LAWS",
    "MACHINE_SHIPPED",
    "MACHINE_STORED",
    "law_provenance",
]

#: The ten machine-admitted objects promoted into the shipped
#: ``ALL_RULES`` (``promoted-laws.md`` — the promotion table's shipped
#: names, including the two objects that shipped as the strictly more
#: general view identities ``transpose_noop`` / ``chunk_single``).
MACHINE_SHIPPED: frozenset[str] = frozenset(
    {
        "softsign_fold",
        "sdpa_fold_nomask",
        "sdpa_fold_div_nomask",
        "mul_unsq_pad_l",
        "mul_unsq_pad_r",
        "sub_unsq_pad_l",
        "mul_reshape_inert_l",
        "transpose_noop",
        "chunk_single",
        "linear_channel_to_row_scale",
    }
)

#: Machine-constructed / machine-guarded objects that cleared the
#: admission gauntlet but were never written into the library — the
#: unshipped store population, by recorded object name.  Three groups:
#:
#: * guarded view-identity strips (auto-cond minted or oracle-named
#:   guards) — the ``*_id`` spellings the generic promoted laws
#:   subsume or leave unshipped;
#: * constructed folds — ``fold_object`` products (the metavar-attr
#:   ``sdpa_fold_div_nomask_g`` twin that stayed unshipped);
#: * constructed lifts/composites — ``lift_object`` /
#:   ``compose_objects`` products (carrier introductions and
#:   multi-block composites).
#:
#: Two name collisions bound what a *name-level* registry can say:
#:
#: * ``om_lift`` — the store holds a constructed ``om_lift`` object
#:   (``lift_object``, literal ``dim=-1`` pattern) while
#:   ``catopt_carriers.om.OM_LIFT`` is a *human-authored* carrier law
#:   with a metavar-dim guard.  The name is excluded below: the
#:   registry resolves the shipped law's class, and the store
#:   object's provenance rides its ``ConstructedObject.construction``
#:   record, not this table.  Arm-level attribution must compare rule
#:   objects, not names, at exactly this seam.
#: * ``softsign_fold`` / ``sdpa_fold_nomask`` / ``sdpa_fold_div_nomask``
#:   — store objects under the same names as promoted laws; already
#:   classified machine via :data:`MACHINE_SHIPPED`, so omitted here.
MACHINE_STORED: frozenset[str] = frozenset(
    {
        # -- guarded view-identity strips (auto-cond / oracle-named)
        "mul_unsqueeze_l_id",
        "mul_unsqueeze_r_id",
        "sub_unsqueeze_l_id",
        "add_unsqueeze_l_id",
        "mul_transpose_l_id",
        "add_transpose_l_id",
        "mul_chunk_l_id",
        "mul_reshape_l_id",
        # -- constructed folds (fold_object)
        "sdpa_fold_div_nomask_g",
        # -- constructed lifts and composites (lift_object/compose)
        "aff_step_lift",
        "aff_scan2_lift",
        "affd_step_lift",
        "affd_scan2_lift",
        "affd_scan4_lift",
        "silu_fold_commuted",
        "channel_then_row_scale",
    }
)

#: Every machine-authored rewrite name — shipped promotions plus the
#: unshipped store population.
MACHINE_LAWS: frozenset[str] = MACHINE_SHIPPED | MACHINE_STORED


def law_provenance(rule: object) -> Literal["machine", "human"]:
    """Classify a rewrite's authorship: ``"machine"`` or ``"human"``.

    Accepts a rule name or a :class:`~catopt_core.egraph.Rewrite`
    (reads ``.name``).  A name in :data:`MACHINE_LAWS` —
    machine-shipped promotions or machine-held store objects —
    reports ``"machine"``; anything else is ``"human"``, the
    library's default authorship.  The function is total over names
    by design: provenance is a property of the *registry*, not a flag
    each rule had to be edited to carry.
    """
    name = (
        rule if isinstance(rule, str) else getattr(rule, "name", rule)
    )
    return "machine" if name in MACHINE_LAWS else "human"
