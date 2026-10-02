"""The morphism engine — block-level optimization over signatures.

Plan 0011, stages 0+1.  Whole-model equality saturation is bounded by
the tensor-level enode cliff; the morphism engine lifts each *block*
to a structural signature and rewrites the tiny block-level graph
(N objects + wires, not N*ops enodes).

``lift_graph`` : model -> :class:`MorphismGraph` — objects are the
composer-selected blocks, arrows are the per-boundary value-flow
verdicts (``chain`` / ``residual`` / ``*_wrapped``), and every node
carries a :class:`BlockSig` read off the block's IR term — the
signature describes *structure* (projections, norms, residuals),
never torch modules.  Blocks that cannot be lifted (export failure, not
executed, non-tensor call signature) stay in the graph as **opaque
boundary nodes**: wires connect through them but no law matches them —
honest partial coverage, same convention as the carrier executors.

A :class:`MorphismLaw` matches on signatures only and returns
:class:`MorphismMatch` rewrites.  Each match carries a
:class:`ReifySpec` — the recipe that maps the morphism-level rewrite
back to concrete IR terms: the joint term is built by *substituting*
block IRs into each other (backend-neutral term composition, no
``nn.Module`` surgery), the term-level laws that already exist
(``assoc_linear`` / ``weight_factor_*`` / ``linear_channel_scale`` /
the exact-tying pass) re-derive the rewrite inside an e-graph, and the
resulting program is cost-gated and **verified** through the sink
before anything is grafted.  Reified steps either replay as witnessed
e-graph merges or ship under the numeric verify gate — a rewrite that
cannot be certified is a decline, never a graft.

Signature coverage, explicitly: signatures recognise ``linear`` /
``matmul`` projections with param-only weights, ``layer_norm`` and
RMS/diagonal-scale norms (the pointwise ``x·rms⁻¹·w`` chain and the
fused ``rms_norm`` op alike), ``embedding`` gather-source weights as
tying candidates, known activation ops, and the residual
``x + f(x)`` spine.  Attention internals (``sdpa``), carrier ops, and
everything else flow through untouched — they make the block richer,
not opaque.  Only blocks whose *export itself* fails (or that never
executed on the captured input) are opaque.
"""

from catopt_orchestrator.crossblock_cse import (
    CrossBlockCSE as CrossBlockCSE,
)
from catopt_orchestrator.morphisms_kv import (
    KVLatentShare as KVLatentShare,
)

from .boundary import (
    _mi_a_mode as _mi_a_mode,
)
from .boundary import (
    _mi_b_mode as _mi_b_mode,
)
from .boundary import (
    _mi_boundary as _mi_boundary,
)
from .boundary import (
    _mi_consumers as _mi_consumers,
)
from .boundary import (
    _mi_io_has_value as _mi_io_has_value,
)
from .boundary import (
    _mi_residual_probe as _mi_residual_probe,
)
from .graph import (
    MorphismGraph as MorphismGraph,
)
from .graph import (
    MorphismNode as MorphismNode,
)
from .graph import (
    Wire as Wire,
)
from .graph import (
    _aux_outs as _aux_outs,
)
from .graph import (
    _BlockRecord as _BlockRecord,
)
from .graph import (
    _dc_replace as _dc_replace,
)
from .graph import (
    _io_evidence as _io_evidence,
)
from .graph import (
    _lift_block as _lift_block,
)
from .graph import (
    _mutates_arg as _mutates_arg,
)
from .graph import (
    _param_bound_ids as _param_bound_ids,
)
from .graph import (
    _refine_inputs as _refine_inputs,
)
from .graph import (
    lift_graph as lift_graph,
)
from .laws import (
    MorphismLaw as MorphismLaw,
)
from .laws import (
    MorphismMatch as MorphismMatch,
)
from .laws import (
    NormCascade as NormCascade,
)
from .laws import (
    OutInCompose as OutInCompose,
)
from .laws import (
    ReifySpec as ReifySpec,
)
from .laws import (
    ResidualAbsorb as ResidualAbsorb,
)
from .laws import (
    ResidualReassoc as ResidualReassoc,
)
from .laws import (
    WeightTie as WeightTie,
)
from .laws import (
    WindowCompose as WindowCompose,
)
from .laws import (
    _compose_pair_ok as _compose_pair_ok,
)
from .laws import (
    _dims_compatible as _dims_compatible,
)
from .laws import (
    _is_pure_diagonal as _is_pure_diagonal,
)
from .laws import (
    _pair_matches as _pair_matches,
)
from .laws import (
    _stream_commutes as _stream_commutes,
)
from .laws import (
    _stream_pair_ok as _stream_pair_ok,
)
from .laws import (
    _window_nodes as _window_nodes,
)
from .laws import (
    _wire_composes as _wire_composes,
)
from .reify import (
    DEFAULT_MORPHISM_LAWS as DEFAULT_MORPHISM_LAWS,
)
from .reify import (
    Const as Const,
)
from .reify import (
    MorphismSearch as MorphismSearch,
)
from .reify import (
    _arm_law_budgets as _arm_law_budgets,
)
from .reify import (
    _ctx_hit as _ctx_hit,
)
from .reify import (
    _ctx_var_map as _ctx_var_map,
)
from .reify import (
    _distribute_offers as _distribute_offers,
)
from .reify import (
    _distribute_over as _distribute_over,
)
from .reify import (
    _fallback_cost_kw as _fallback_cost_kw,
)
from .reify import (
    _ins_list as _ins_list,
)
from .reify import (
    _intra_joint as _intra_joint,
)
from .reify import (
    _joint_parts as _joint_parts,
)
from .reify import (
    _joint_parts_window as _joint_parts_window,
)
from .reify import (
    _law_tags as _law_tags,
)
from .reify import (
    _lower_term as _lower_term,
)
from .reify import (
    _norm_unfolds as _norm_unfolds,
)
from .reify import (
    _ns_prefix as _ns_prefix,
)
from .reify import (
    _optimize_morphisms as _optimize_morphisms,
)
from .reify import (
    _pair_joint as _pair_joint,
)
from .reify import (
    _prefix_params as _prefix_params,
)
from .reify import (
    _rec_act as _rec_act,
)
from .reify import (
    _recipe_rules as _recipe_rules,
)
from .reify import (
    _reify as _reify,
)
from .reify import (
    _reify_terms as _reify_terms,
)
from .reify import (
    _reify_tie as _reify_tie,
)
from .reify import (
    _res_stats as _res_stats,
)
from .reify import (
    _resolve_joint as _resolve_joint,
)
from .reify import (
    _rest_example as _rest_example,
)
from .reify import (
    _saturate as _saturate,
)
from .reify import (
    _sig_dict as _sig_dict,
)
from .reify import (
    _slot_filler as _slot_filler,
)
from .reify import (
    _subst as _subst,
)
from .reify import (
    _subst_ctx as _subst_ctx,
)
from .reify import (
    _verify_pair as _verify_pair,
)
from .reify import (
    _window_chain as _window_chain,
)
from .reify import (
    _window_joint as _window_joint,
)
from .reify import (
    _window_reps as _window_reps,
)
from .reify import (
    _window_residual as _window_residual,
)
from .reify import (
    _witness_offer as _witness_offer,
)
from .reify import (
    flops_cost as flops_cost,
)
from .reify import (
    optimize_morphisms as optimize_morphisms,
)
from .signature import (
    BlockSig as BlockSig,
)
from .signature import (
    InputSig as InputSig,
)
from .signature import (
    NormSig as NormSig,
)
from .signature import (
    WeightRef as WeightRef,
)
from .signature import (
    _act_index as _act_index,
)
from .signature import (
    _classify_inputs as _classify_inputs,
)
from .signature import (
    _iter_ops as _iter_ops,
)
from .signature import (
    _norm_nodes as _norm_nodes,
)
from .signature import (
    _probe_value as _probe_value,
)
from .signature import (
    _residual_spine as _residual_spine,
)
from .signature import (
    _shape_tuple as _shape_tuple,
)
from .signature import (
    _sig_liftable as _sig_liftable,
)
from .signature import (
    block_signature as block_signature,
)
from .signature import (
    weights_tied as weights_tied,
)

__all__ = [
    "DEFAULT_MORPHISM_LAWS",
    "BlockSig",
    "CrossBlockCSE",
    "InputSig",
    "KVLatentShare",
    "MorphismGraph",
    "MorphismLaw",
    "MorphismMatch",
    "MorphismNode",
    "MorphismSearch",
    "NormCascade",
    "NormSig",
    "OutInCompose",
    "ReifySpec",
    "ResidualAbsorb",
    "ResidualReassoc",
    "WeightRef",
    "WeightTie",
    "WindowCompose",
    "Wire",
    "block_signature",
    "lift_graph",
    "optimize_morphisms",
    "weights_tied",
]
