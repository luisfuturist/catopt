"""Stage 1 — morphism laws.

Signature-level rewrite laws: :class:`MorphismLaw` matches on
:class:`MorphismGraph` signatures/wires only and returns
:class:`MorphismMatch` rewrites carrying a :class:`ReifySpec`.  Split
out of :mod:`catopt_orchestrator.morphisms` (plan 0011); the whole
surface is re-exported from that package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .graph import MorphismGraph, Wire
from .signature import BlockSig, weights_tied

# ---------------------------------------------------------------------------
#  Stage 1 — morphism laws
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReifySpec:
    """The recipe mapping a morphism rewrite back to concrete terms.

    * ``mode`` — the boundary mode the joint term is built in
      (``"chain"`` / ``"residual"`` / their ``_wrapped`` forms), or
      ``"intra"`` (single-block rewrite on its own term) / ``"tie"``
      (exact weight sharing across the nodes' joint e-graph).
    * ``rules`` — the named term-level recipe saturated on the joint
      e-graph: ``"compose"`` (the bilinearity/associativity/factor
      family that folds ``A.out ∘ B.in``), ``"scale"`` (the diagonal
      naturality family folding norm gains into weights), ``"tie"``
      (no saturation — the exact-tying pass).
    * ``distribute`` — first offer the bilinear expansion of the
      residual sum into each in-projection
      (``linear(x + A(x), W) = linear(x,W) + linear(A(x),W)``) as a
      witnessed member — the data-slot distributivity step the rule
      set has no ``linear``-spelling law for, asserted at morphism
      level and gated by the pair verify.
    * ``share`` — run the value-exact weight-tying pass on the joint
      e-graph before extraction.
    * ``kinds`` — for *window* rewrites (``len(nodes) >= 3``), the
      per-wire boundary kinds in arrow order (``len(nodes) - 1``
      entries).  ``mode`` then mirrors ``kinds[-1]``: it still drives
      the first-slot delta and last-slot filler conventions.  Empty
      for pair / intra / tie matches.
    * ``extra`` — an opaque law-carried payload the mode's reify
      reads back: the ``"family"`` mode (KV latent sharing,
      :mod:`catopt_orchestrator.morphisms_kv`) packs the name-token
      sets, the factor tolerance and the shared data term
      identifying the latent group.
    """

    mode: str
    rules: str = "compose"
    distribute: bool = False
    share: bool = False
    kinds: tuple[str, ...] = ()
    extra: Any = None


@dataclass(frozen=True)
class MorphismMatch:
    """One law firing on the lifted graph.

    ``nodes`` are the block names in arrow order; ``boundary`` is the
    wire kind (or ``"intra"`` / ``"tie"``); ``reify`` is the
    :class:`ReifySpec` the engine executes; ``detail`` is the human
    record of *why* the law saw this match.
    """

    law: str
    nodes: tuple[str, ...]
    boundary: str
    reify: ReifySpec
    detail: str = ""


@runtime_checkable
class MorphismLaw(Protocol):
    """A rewrite law on the morphism graph — signatures only.

    ``match`` reads :class:`MorphismGraph` signatures/wires and returns
    :class:`MorphismMatch` rewrites; it never touches tensor-level
    terms (those arrive at *reify* time, through the spec the match
    carries).  Conformance is duck-typed: a ``name`` plus a
    ``match(graph) -> list`` method.
    """

    name: str

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Return the law's firings on *graph*."""
        ...


def _dims_compatible(a: BlockSig, b: BlockSig) -> bool:
    """Check A's output feeds B's input (last dims agree).

    Unknown shapes pass — the match is a candidate; the verify gate at
    reify time is authoritative.
    """
    sa, sb = a.shape[1], b.shape[0]
    if (
        isinstance(sa, tuple)
        and isinstance(sb, tuple)
        and sa
        and sb
        and sa[-1] is not None
        and sb[-1] is not None
    ):
        return sa[-1] == sb[-1]
    return True


def _is_pure_diagonal(sig: BlockSig) -> bool:
    """Check the whole block is one diagonal map (``x ∘ s``)."""
    return (
        sig.norm.kind == "diag"
        and not sig.in_projs
        and not sig.out_proj
        and not sig.residual
    )


def _pair_matches(
    graph: MorphismGraph,
    law: str,
    kinds: frozenset,
    pred: Any,
    spec: Any,
    detail: str,
) -> list[MorphismMatch]:
    """Emit a match for each wire whose boundary and sigs qualify."""
    out = []
    for w in graph.wires:
        if w.kind not in kinds:
            continue
        a, b = graph.sig(w.src), graph.sig(w.dst)
        if a is None or b is None or not pred(a, b):
            continue
        out.append(
            MorphismMatch(
                law=law,
                nodes=(w.src, w.dst),
                boundary=w.kind,
                reify=spec(w.kind),
                detail=detail.format(a=w.src, b=w.dst),
            )
        )
    return out


class OutInCompose:
    """``A.out_proj ∘ B.in_proj`` — compose projections across a chain.

    The signature-level form of the cross-pair weight fold: A's
    terminal projection composes with each of B's input projections.
    Matches chain boundaries (plain or wrapped) between lifted blocks
    that both carry projections; reify builds the joint term
    ``B(A(x))`` and saturates the compose recipe — the term-level
    ``assoc_linear`` / ``weight_factor_*`` laws fold the chain into a
    single weight.
    """

    name = "out_in_compose"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Match chain wires with projections on both sides."""
        return _pair_matches(
            graph,
            self.name,
            frozenset({"chain", "chain_wrapped"}),
            lambda a, b: (
                bool(a.out_proj)
                and bool(b.in_projs)
                and _dims_compatible(a, b)
            ),
            lambda kind: ReifySpec(mode=kind, rules="compose"),
            "{a}.out_proj ∘ {b}.in_projs",
        )


class ResidualAbsorb:
    """``x + A(x) → B`` — the residual add is absorbable into B's projs.

    Matches residual boundaries where B has input projections: the
    reified program distributes each ``linear(x + A(x), W)`` over the
    add (bilinearity — the witnessed morphism step), then the compose
    recipe folds the ``A(x)`` side's ``A.out ∘ W`` composition.  What
    remains is ``linear(x, W) + linear(A_inner, W @ A_out)`` — B
    absorbing A's output projection into its input weights.
    """

    name = "residual_absorb"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Match residual wires into blocks with input projections."""
        return _pair_matches(
            graph,
            self.name,
            frozenset({"residual", "residual_wrapped"}),
            lambda a, b: bool(b.in_projs),
            lambda kind: ReifySpec(
                mode=kind, rules="compose", distribute=True
            ),
            "{b} absorbs the residual add over {a}",
        )


class NormCascade:
    """Norm diagonals cascade through block chains into next weights.

    Two forms:

    * *pair* — a pure diagonal block (``x ∘ s`` — a standalone gain or
      scale) on a chain boundary: its diagonal commutes into B's input
      projections (``linear(x ∘ s, W) = linear(x, W ∘ s)``).
    * *node* — an affine *pre*-norm inside one block: the gain folds
      into the block's own in-projection weights (the RMSNorm→Linear
      fold the signature already sees).
    """

    name = "norm_cascade"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Match diagonal-feeding wires and affine pre-norm nodes."""
        out = _pair_matches(
            graph,
            self.name,
            frozenset({"chain", "chain_wrapped"}),
            lambda a, b: _is_pure_diagonal(a) and bool(b.in_projs),
            lambda kind: ReifySpec(mode=kind, rules="scale"),
            "{a} diagonal cascades into {b}.in_projs",
        )
        for n in graph.nodes:
            sig = n.sig
            if sig is None:
                continue
            if sig.norm.affine and sig.norm.pre and sig.in_projs:
                out.append(
                    MorphismMatch(
                        law=self.name,
                        nodes=(n.name,),
                        boundary="intra",
                        reify=ReifySpec(mode="intra", rules="scale"),
                        detail=(
                            f"{n.name} pre-norm gain folds into "
                            "in_projs"
                        ),
                    )
                )
        return out


class WeightTie:
    """Weight tying: identical param shapes + name stem → share.

    Candidate generation is signature-level: two nodes (possibly the
    same node — duplicated branch weights inside one block) carrying
    weights with equal shapes, plus an equal name stem for the
    cross-block case (a shared ``nn.Parameter`` exports under the same
    leaf name in both blocks).  The candidate set is a block's
    projection refs *plus* its table weights — ``embedding``'s gather
    source is how the tied emb/head pair (llama2.c ``wcls``) becomes
    visible.  Reify interns the involved blocks'
    terms into one e-graph and runs :func:`share_duplicate_params` —
    the value-exact pass decides whether the candidate tie is real; a
    shape+name coincidence that does not share *values* declines.
    """

    name = "weight_tie"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Match intra-block shape dups and cross-block stem+shape ties."""
        out = []
        lifted = [
            (n.name, s) for n in graph.nodes if (s := n.sig) is not None
        ]
        for name, sig in lifted:
            refs = tuple(
                dict.fromkeys(sig.in_projs + sig.out_proj + sig.tables)
            )
            if len(refs) > 1 and any(
                x.shape is not None and x.shape == y.shape
                for i, x in enumerate(refs)
                for y in refs[i + 1 :]
            ):
                out.append(
                    MorphismMatch(
                        law=self.name,
                        nodes=(name,),
                        boundary="tie",
                        reify=ReifySpec(
                            mode="intra", rules="tie", share=True
                        ),
                        detail=f"{name}: same-shape weights",
                    )
                )
        for i, (name_a, sa) in enumerate(lifted):
            for name_b, sb in lifted[i + 1 :]:
                refs_a = sa.in_projs + sa.out_proj + sa.tables
                refs_b = sb.in_projs + sb.out_proj + sb.tables
                if any(
                    weights_tied(wa, wb)
                    for wa in refs_a
                    for wb in refs_b
                ):
                    out.append(
                        MorphismMatch(
                            law=self.name,
                            nodes=(name_a, name_b),
                            boundary="tie",
                            reify=ReifySpec(
                                mode="tie", rules="tie", share=True
                            ),
                            detail=(f"{name_a}~{name_b}: tied weights"),
                        )
                    )
        return out


def _compose_pair_ok(a: BlockSig | None, b: BlockSig | None) -> bool:
    """Check the out∘in signature predicate on one adjacent pair."""
    return (
        a is not None
        and b is not None
        and bool(a.out_proj)
        and bool(b.in_projs)
        and _dims_compatible(a, b)
    )


def _wire_composes(
    graph: MorphismGraph, w: Wire, kinds: frozenset[str]
) -> bool:
    """Check a wire's boundary kind and both sides' signatures."""
    return w.kind in kinds and _compose_pair_ok(
        graph.sig(w.src), graph.sig(w.dst)
    )


def _window_nodes(wires: tuple[Wire, ...], i: int, j: int) -> tuple:
    """Block names for wires ``i..j`` inclusive — arrow order."""
    return (
        *(wires[t].src for t in range(i, j + 1)),
        wires[j].dst,
    )


def _stream_commutes(
    graph: MorphismGraph, nodes: tuple[str, ...]
) -> bool:
    """Check a residual window carries a real commute opportunity.

    Every node must be lifted (an opaque block is a boundary, never
    crossed), and some earlier block's out-projection must compose
    into a *later* block's input projections.  The receiving block's
    projections must read the stream with no pre-norm in the way —
    ``norm.pre`` marks a nonlinear normaliser bilinearity cannot
    cross.
    """
    sigs = [graph.sig(n) for n in nodes]
    if any(s is None for s in sigs):
        return False
    lifted = [s for s in sigs if s is not None]
    return any(
        _stream_pair_ok(si, sj)
        for i, si in enumerate(lifted[:-1])
        for sj in lifted[i + 1 :]
    )


def _stream_pair_ok(a: BlockSig, b: BlockSig) -> bool:
    """One contributing pair: out-proj into a pre-norm-free receiver."""
    return (
        bool(a.out_proj)
        and bool(b.in_projs)
        and not b.norm.pre
        and _dims_compatible(a, b)
    )


class WindowCompose:
    """``A.out ∘ B.in ∘ C.in ∘ …`` — compose a whole chain window.

    Generalises :class:`OutInCompose` from a boundary pair to a
    maximal run of ≥3 blocks: interior wires must be plain ``chain``
    (each block's output is consumed by exactly the next input), and
    the final wire may additionally be ``chain_wrapped`` (the
    parent's ``y + B(y)`` wrap around the last block).  One
    :class:`MorphismMatch` per window carries one :class:`ReifySpec`
    — a single joint term, a single joint e-graph, a single verify —
    instead of k-1 pairwise passes that would consume the blocks two
    at a time.
    """

    name = "window_compose"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Grow maximal composable chain windows left-to-right."""
        out: list[MorphismMatch] = []
        wires = graph.wires
        i = 0
        while i < len(wires):
            if not _wire_composes(
                graph, wires[i], frozenset({"chain"})
            ):
                i += 1
                continue
            j = i
            while j + 1 < len(wires) and _wire_composes(
                graph, wires[j + 1], frozenset({"chain"})
            ):
                j += 1
            kinds = [wires[t].kind for t in range(i, j + 1)]
            # A wrapped tail may close the window (last block only).
            if j + 1 < len(wires) and _wire_composes(
                graph, wires[j + 1], frozenset({"chain_wrapped"})
            ):
                j += 1
                kinds.append("chain_wrapped")
            if j - i >= 1:  # >=2 wires -> >=3 blocks
                nodes = _window_nodes(wires, i, j)
                out.append(
                    MorphismMatch(
                        law=self.name,
                        nodes=nodes,
                        boundary="+".join(kinds),
                        reify=ReifySpec(
                            mode=kinds[-1],
                            rules="compose",
                            kinds=tuple(kinds),
                        ),
                        detail=(
                            f"{nodes[0]}->…->{nodes[-1]}: "
                            f"{len(nodes)}-block out∘in window"
                        ),
                    )
                )
            i = j + 1
        return out


class ResidualReassoc:
    """The residual ``+`` monoid commutes receivers past blocks.

    On a residual-stream run ``s = x + f0(x) + f1(·) + …`` the stream
    every block reads is a *sum* of all earlier contributions, so a
    later block's input projections may legally distribute over
    addends a non-adjacent block produced:
    ``linear(s, W) = linear(x, W) + Σ linear(f_i(·), W)`` — and each
    ``linear(f_i(·), W)`` is the ``f_i.out ∘ W`` composition the pair
    laws reach only for adjacent blocks.  The additive monoid
    (associativity + commutativity of ``+``) is the legal commute
    path; bilinearity does the absorption.

    Matches maximal windows of ≥3 blocks whose interior wires are
    ``residual_wrapped`` (the stream flows on), optionally closed by
    a plain ``residual`` receiver; emits one match whose reify offers
    the stream-distributed joint — a constructed equality asserted at
    morphism level and gated by the fp64 verify + the cost gate.
    """

    name = "residual_reassoc"

    def match(self, graph: MorphismGraph) -> list[MorphismMatch]:
        """Grow maximal residual-stream windows left-to-right."""
        out: list[MorphismMatch] = []
        wires = graph.wires
        i = 0
        while i < len(wires):
            if wires[i].kind != "residual_wrapped":
                i += 1
                continue
            j = i
            while (
                j + 1 < len(wires)
                and wires[j + 1].kind == "residual_wrapped"
            ):
                j += 1
            # An unwrapped receiver may close the window.
            if j + 1 < len(wires) and wires[j + 1].kind == "residual":
                j += 1
            if j - i >= 1:  # >=2 wires -> >=3 blocks on the stream
                nodes = _window_nodes(wires, i, j)
                if _stream_commutes(graph, nodes):
                    kinds = [wires[t].kind for t in range(i, j + 1)]
                    out.append(
                        MorphismMatch(
                            law=self.name,
                            nodes=nodes,
                            boundary="+".join(kinds),
                            reify=ReifySpec(
                                mode=kinds[-1],
                                rules="compose",
                                distribute=True,
                                kinds=tuple(kinds),
                            ),
                            detail=(
                                f"{nodes[0]}->…->{nodes[-1]}: residual"
                                " stream reassociation over "
                                f"{len(nodes)} blocks"
                            ),
                        )
                    )
            i = j + 1
        return out
