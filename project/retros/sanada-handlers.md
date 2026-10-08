# Retro — Sanada's arrow handlers, mapped onto catopt

Two papers by Takahiro Sanada that HANDL's GMSAC kernel builds on
(`handl-lang/platform/docs/explanation/graded-modal-sanada-arrow-calculus.md`):

1. **Algebraic effects and handlers for arrows** (JFP 2024) —
   arrows + `Op`/`Handle`: an `Op` is a *pure effect request* with
   an explicit continuation; a handler is a scoped interpretation
   (an A-homomorphism between A-algebras of a strong promonad in
   Prof).  Descriptions and implementations are separate data.
2. **Programming backpropagation with reverse handlers for arrows**
   (ICFP 2026) — backprop itself is a *reverse handler* over arrows
   on reverse differential restriction categories; networks are
   symbolic descriptions, implementations assigned by handlers.

## The mapping

| Sanada / HANDL GMSAC | catopt |
|---|---|
| `Op` — uninterpreted request (data) | a `declare`d object / unlowered op |
| `handle p with H` — scoped interpretation | an **execution-strategy move**: handler choice is a first-class move, not a backend flag |
| grade row `~[(Op, Query)]~>` | `cond` — the preconditions under which an interpretation is legal |
| A-algebra + homomorphism | sinks/cost models — verify, price, run are three interpretations of one syntax |
| reverse handler | backprop as *interpretation*: the gradient program is derived by a handler |

## What it buys us

- **ADR-0003 formalized.**  Semantics/search/evaluation/execution
  separation is "one effectful arrow, many handlers".  A declared
  object mints the *request*; the handler choice (which kernel,
  carrier, or cost model interprets it) is a move in the
  meta-arena — `change execution` becomes `handle ... with ...`.
- **Purity = serializability = learnable.**  `Op`/`Handle` are
  terms — pure data — so a handler is something the player can
  *write*, not just choose: a data-level construction move for the
  evaluation/execution dimensions we currently express as code.
- **A second domain for free.**  Reverse handlers mean the
  *gradient* program is derivable inside the same universe — the
  optimizer's laws/search/referee could optimize training graphs
  (not just inference) with no new semantics: an untapped corpus
  and a real test of the tower's generality.
- **The interpreter is data too** — handlers compose and scope;
  reinterpretation (verify-then-bench, model-then-measure) is
  handler composition, which is what the pipeline already does
  informally.  Making it explicit names the thing.

## Open questions for the arena

- Does `handle`-as-move give the player an execution-strategy
  action space (choose the interpreter per declared object) that
  is *deep* — where the order/choice of interpretation changes
  measured pay?
- Can the gauntlet be a handler chain (interpret the construction
  under `verify`, then `measure`, then `truth` — each a grade)?
- Reverse handlers make "run the program backwards" a legal
  construction — is there an optimizer move that *inverts* a
  handler (search the program that a cheaper interpretation would
  have produced)?


## Landed: both directions

* **`handle` as a move** (`meta_arena.Action.handle` +
  `lawdata.HANDLERS`): a declared object's interpretation is a
  scoped, legal-checked, repricing move — the "execution-strategy
  action" the table above predicted.  Legality is alpha-coverage
  of the object's spelled body; a handled name prices at its
  kernel under the supported bound (measurably cheaper under
  op-count, honestly flat under flops).

* **Reverse handlers** (`catopt_discovery.training` +
  `lawdata.REVERSE`): `backward(term)` derives the gradient
  program as an *interpretation of the forward term* — a data
  VJP table, not a second semantics stack.  The result is an
  ordinary term: `tests/test_discovery_training.py` checks it
  against `torch.autograd` numerically (elementwise, shared-leaf
  accumulation, matmul with a shaped cotangent, a matmul+tanh
  chain) and shows it lives on the board — saturate/extract/
  certify run on the gradient program unchanged.  Missing VJP
  rows decline (`ValueError`), never silently zero.

The honest limit: the REVERSE table covers shape-free ops only —
reductions need a broadcast spelled with output shape (expressible
via a caller-supplied cotangent), and view ops need shape
information the spec grammar does not bind yet.  Extending the
table is a data edit, not an engine change — that was the point.

*Update — the shape gap closed.*  ``$shape:N`` splices an arg's
inferred shape (``_shape_of``), ``$rcount`` counts reduced
elements, ``$invdims`` inverts a permute, and alternative rows
with ``requires`` guards (``"$ABSENT"`` = attr must be missing)
encode conditional VJPs.  ``sum``/``mean`` (full-reduce or
keepdim), ``reshape`` and ``permute`` are live; ``sum`` with
``dim`` + ``keepdim=False`` still declines honestly — the
cotangent needs an un-reduce the grammar does not spell.
Everything checked against ``torch.autograd``.

And the interpretation is now a *probe column*: ``meta_probe``'s
declare arm reports ``handled`` — the cheapest certified
extraction after applying every afforded ``handle`` — so a
candidate worthless under spelled parity but cheap once
interpreted surfaces on the board.


*Update 2 — the corpus grew.*  ``expand``/``broadcast_to`` VJPs
landed via ``$expdims`` (dims the target grew over the
left-padded input) spelled as ``reshape(sum(g, dims,
keepdim), shape=in)``; all checked against autograd.  And the
domain claim is now a running probe — ``python -m
catopt_discovery.training`` derives gradients for real-shaped
forwards (silu, keepdim softmax, matmul+bias+tanh) and
saturate→extract→certifies each on the meta-arena board: all
three certify, laws fire on the gradient programs (enodes grow
12–18).  A training graph is a board citizen today — measured,
not just designed.


*Update 3 — a real module's backward, end-to-end.*  ``relu``
(``mul(g, gt(x,0))``), ``squeeze``/``unsqueeze`` (attr-swap), and
``linear`` (``dx = gW``, ``dW = g^Tx``, ``db = $batchsum`` —
cotangent summed over leading dims) joined the table.  The check
that matters: ``TorchSource``-exported ``Linear→ReLU`` module →
``backward(ir.root)`` produces param gradient terms that match
``torch.autograd`` numerically — the reverse handler on real
lowered code, not hand-built terms.
