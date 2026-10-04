# Law-proposal as a sequential game — a player over constructions

The pipeline's law discovery today is **enumeration + filter**: the
proposers emit complete candidate identities (algebraic grammar,
shape-aware schemas, census naturality) and the referee filters them
through the numeric oracle, derivability and impact.  This retro tests
the first step of the alternative the project is aimed at — a **player
that proposes laws**: a sequential construction game where the state
is a partially-built candidate and the policy picks the next
construction move.  The oracle stays the referee; the player only
chooses *which* candidates to spend the oracle budget on.

The verdict, plainly: **the game stands up and the player learns, but
its yield per oracle call is ~10× behind the enumerative baseline
in-budget.**  The construction player beats a uniform player over the
same game by ~3–5× and finds genuinely *new* true+firing candidates —
including an emergent family nobody authored (involution/view padding
laws) and textbook commutations — but neither `select_mul` nor
`softmax_fold` is rediscovered within budget, and nothing it finds
pays end-to-end.  The binding constraint is the density of *true*
RHSs under a terminal-only truth reward, not applicability.

Reproduce (the measured run, ~5 min CPU):

    .venv/bin/python tools/law_meta_game.py \
        --episodes 1500 --budget 300 --random-calls 150 \
        --seed-frac 0.55 --seed 1 --json /tmp/mg.json

`packages/` is untouched; everything lives in
`tools/law_meta_game.py`.

## 1. The game

* **State** — a partially-built LHS pattern tree, a partially-built
  RHS tree, the *hole pointer* (DFS position of the open slot), the
  set of bound metavariables, and each bound metavariable's binding
  paths (needed for the sharing gate below).  LHS is built first,
  then RHS; the op budget is ≤6 nodes on the LHS and ≤4 on the RHS
  (the task bound, split toward the matching side).
* **Action** — place one vocabulary element at the current hole: an
  op from the `law_vocab`-derived alphabet (32 ops of arity ≤2 that
  are *testable* — ops the property probes cannot evaluate, e.g.
  `arange`/`embedding`, are dead moves and excluded), a metavariable
  from a pool of 4 names, or a literal (`0`, `1`).  An op placed on
  the LHS carries auto-minted attr metavariables (`select.dim`,
  `select.index`) so real sites' attrs bind; on the RHS the same
  names are reused when the op already appeared on the LHS (so an
  RHS `select` inherits the LHS's bound attrs), else fresh names that
  a `derive` bridge resolves at instantiation.
* **Terminal** — both sides filled; the pair mints to a
  `catopt_core.meta.Rewrite` with a generic bool-attr `check` and a
  suffix-matching attr `derive`.

### 1.1 Legality is corpus knowledge, stated as a rule book

The game's legal moves — not its scores — encode the shape-aware
lesson: a construction that can *never* match the corpus is an
illegal move, the same way an illegal chess move is illegal.

* An op placed at LHS position `i` under parent `p` must be
  census-attested there: `child_count[(p, i, op)] > 0`.  A
  `(parent, pos, child)` triple absent from the corpus can never
  match — the move is dead before it costs an oracle call.
* A `Const` leaf on the LHS is legal only where the corpus puts a
  `const` child; a metavariable is always legal (it binds whatever
  the site holds — leaf *or* op subtree).
* **Metavariable reuse is gated by a structural-sharing census.**
  Placing an already-bound metavariable imposes a
  structural-equality precondition on the two sites; it is legal only
  where the corpus itself shares an `op_repr`-equal pair of
  descendants under the *lowest common ancestor* of the bindings
  (a per-`(op, child-tuple)` site count computed by `_share_scan`).
  `linear(U,U)` is dead (no `linear` site shares), while
  `add(mul(U,V), mul(U,W))`, `mul(sel(U), sel(U))` and
  `div(exp(U), sum(exp(U)))` stay legal — exactly the sharing the
  census records.
* On the RHS only the op budget and the bound-metavariable rule
  apply — the generated side is unconstrained by applicability.

### 1.2 Priors — soft bias, not legality

On top of legality each legal action carries a fixed prior logit:
the corpus child-frequency at the hole (`log1p`-scaled, with a
0.55^depth decay so the LHS stays shallow and applicable), plus
honest structural biases on the RHS — *op occurs on the other side*
(+2.0), *same root op as LHS* (+1.0, the commutation move), *LHS
depth-1 op at the RHS root* (+1.2, the naturality
`f(g·,g·) -> g(f(·,·))` move), *bound metavariable* (+1.2) and
*fresh metavariable* on the LHS (+0.4).

### 1.3 Start states

55% of plays start from a census skeleton, sampled ∝
`log1p(count)`: the top-30 op-tuple seeds (`mul(select,·)`,
`add(mul,mul)`, `div(exp,sum)`, …) **and** the top-30 *shape-census*
skeletons — real abstracted subterm shapes (`mul(select(lin,·),select(·))`,
`transpose(reshape(·))`, …) re-minted as patterns with holes at the
leaves.  The rest start from an empty LHS.

## 2. The player and the referee

* **Players.**  `BuildGame` plays until terminal.  The uniform
  player samples legal actions uniformly; the learned player is a
  tiny MLP scoring `(state-vector, action-vector)` pairs — logits
  `W·[φ(s);φ(a)] + prior` — trained with REINFORCE over terminal
  rewards (EMA baseline; the same policy-gradient discipline as
  `catopt_torch.rl`, recast onto construction actions since the
  library `PolicyNet` scores `(state, rule)` over *existing* rules).
* **Referee** (`Referee`) reuses the pipeline pieces rather than
  inventing a disconnected objective: find the first corpus-slice
  subterm the LHS matches (`_term_match`, free — no oracle call),
  then `law_proposal._numeric_true` on one real instance (the bounded
  oracle call — hard gate), then `law_impact._probe` over the whole
  16-case slice for firing/cost/verify.  Equality is symmetric, so
  one oracle call referees **both** firing orientations; dedup keys
  cover swapped pairs so a construction's orientation never buys a
  second call.  `relation` is computed against the shipped library
  (`new`/`duplicate`/`inverse`/`tautology`).
* **Reward.**  `score = truth · (1 + fires/5 + 2·paid −
  5·verify_fail + 10·rel_drop)`; false or inapplicable proposals
  score 0.  Plays are cheap until they match a corpus term; only
  then does a `_numeric_true` call fire — the same currency the
  enumerative baseline is charged.
* **Baseline** is the pipeline's own proposal stage (census
  naturality + mixed-view + shape-schemas + grammar), 50 proposals,
  the same slice and the same referee.

## 3. Results

### 3.1 Yield per oracle call

| player | plays | distinct | oracle calls | true | true+fire | new+t+f | ship~ | yield/call |
|---|---|---|---|---|---|---|---|---|
| enumerative baseline | 50 | 47 | 24 | 15 | 7 | 2 | 0 | **0.292** |
| uniform player | 2037 | 2017 | 150 | 1 | 1 | 1 | 1 | 0.007 |
| learned player | 1400 | 1337 | 300 | 9 | 9 | 8 | 0 | **0.030** |

Two further runs bound the variance (hits are Poisson-rare): at
3000 training episodes the eval ran to its 2500-play cap at 98 calls
with 2 true+fire (0.020); at 2000 episodes with `--seed-frac 0.7`
it spent the full 400 calls for 3 true+fire (0.007).  Ordering is
stable: **baseline ≫ learned ≈ 3–5× uniform**.

Training itself spent 300–534 oracle calls over 1500–3000 episodes
and surfaced 2–9 true candidates — the policy's reward signal exists
(tail mean reward positive) but is thin.

### 3.2 What the player actually found

The learned player's top-scoring distinct candidates:

* `mul(V,U) -> mul(U,V)` (56 fires) and `add(W,V) -> add(V,W)` (32
  fires, `paid=2`) — **real commutations**, independently proposed;
  duplicates of shipped laws.
* `V -> mul(V,1)` — true, fires 332×; the **inverse** of a shipped
  identity (flagged by `relation`, not counted as new).
* `V -> alias(V)` — new, true, fires 332×.
* **An emergent family: involution/view padding.**  `X ->
  transpose²(X)`, `X -> reshapeⁿ(X)`,
  `transpose(reshape(X)) -> transpose(reshape(alias(reshape(X))))`,
  `X -> reshape(reshape(expand(reshape(X))))` — all *new, true and
  firing*, all `paid=0`.  The policy learned that wrapping a matched
  form in a cancelling view sequence is the cheapest way to score
  `truth + fires` — a legitimate (if useless) reward-exploit family.
  The `paid` gate correctly keeps every one of them out of "ship~".
* `unsqueeze(select(X)) -> select(unsqueeze(X))` — a real
  view-commutation of a sort the schemas never proposed (8 fires,
  `paid=0`).
* `select(V,d,i) -> unsqueeze(select(V,d,i))` — scored **true** by
  the oracle because `torch.allclose` broadcasts `(n,)` vs `(1,n)`:
  see §4.

### 3.3 Where the search stalls

Distinct-candidate outcomes on the measured run:

```
baseline   {'true': 15, 'no-instance': 23, 'false': 9}
random     {'no-instance': 1867, 'unknown': 65, 'false': 84, 'true': 1}
player     {'unknown': 148, 'no-instance': 1037, 'false': 143, 'true': 9}
```

* **`no-instance` still dominates** (~78%): census-legality makes
  every construction *locally* plausible, but deeper placements under
  seed leaf-holes produce *path-inconsistent* trees that no real
  subterm satisfies.  The corpus can certify a move, not a whole
  path.
* **`unknown` (~14%)**: RHS ops that fail to evaluate on the bound
  leaves' shapes (`matmul` on rank-1, `reshape` numel mismatch, …) —
  the oracle returns `None`, the call is spent anyway.
* **`false` (~7%)**: the honest gate doing its job — these reached
  the oracle and were rejected.
* **true (~0.4%)**: the player's wins are shallow identities,
  commutations and padding — *density of true RHSs in the
  construction space is the wall*, not applicability and not legality.

### 3.4 Rediscovery of the shipped winners

**Neither flagship shape was rediscovered in-budget.**  The alpha-
normal keys of `select_mul` (`mul(sel_k U, sel_k V) -> sel_k(mul U V)`)
and `softmax_fold` (`div(exp U, sum_k exp U) -> softmax_k U`) never
appear among the player's true candidates.  Both are *reachable* —
the seeds contain the LHS shapes and the census/sharing gates admit
their metavars — but landing the exact RHS (`select(mul(U,V))` under
`mul(select,select)`, or `softmax` appearing on the RHS *not* present
in the LHS) is a ~10⁻³–10⁻⁴-per-play draw, and a few hundred oracle
calls of training signal is not enough for the policy to climb
there.

## 4. Honesty about what this does and does not show

* **The player did not beat enumeration — report it straight.**
  Yield/call 0.007–0.030 vs 0.292.  The enumerative baseline's
  proposals are *true-by-design* (schemas built to be identities);
  the player proposes constructions and pays an oracle call for each
  near-miss.  Closing a ~10× gap needs a fundamentally denser
  truth signal (guided RHS completion, search over shared structure,
  or a much longer curriculum) — not prior tuning.
* **The corpus is read into the rule book and the start state.**
  Census legality, sharing gates and shape seeds are all corpus
  statistics: the player's *applicable* LHSs are largely the corpus's
  own shapes by construction.  That is the intended design — a
  proposer reads the corpus — but it means the "rediscovery" bar
  applies to the RHS and the pairing, not the LHS scaffold.
* **The single-instance oracle is the spike's bound, honestly
  labeled.**  `_numeric_true` checks one matched instance for speed;
  a candidate true on that instance can be false elsewhere.  Same
  caveat as the shipped pipeline, which only escalates checks
  downstream.  Relatedly, the shipped `_allclose` accepts
  broadcastable rank mismatches — it certified `select(X) =
  unsqueeze(select(X))`, a rank-changing "equality".  Worth a line in
  the oracle's file eventually; for now the meta-game inherits it.
* **The yield metric counts the padding family.**  `X ->
  viewⁿ(X)` candidates are genuinely new+true+firing; they inflate
  `new+t+f` while paying nothing.  The ship gate (paid>0, verify,
  closure) is what separates them from a proposal worth shipping,
  and on that stricter metric the player scored 0.
* **The learned-vs-uniform gap is real but small-sample.**  Ordering
  held across three configs (learned 3–5× uniform per call), but
  each is a single seed; the true-hit counts (2–9 per 300–400 calls)
  are a handful of Poisson draws.
* **REINFORCE is honest but thin here.**  Terminal-only reward over
  ~6–20-step trajectories, sparse positives; the training curves
  show reward appearing late (ep ~450+).  The architecture is a
  first-cut proof of the framing, not a tuned RL effort.

## 5. Recommendation (reported, not made)

`packages/` untouched; this is a tool-level spike.  On this evidence:

* **Keep enumeration as the proposer of record.**  Its yield per
  oracle call is an order of magnitude ahead at every budget tried.
  The player's job — *proposal policy over constructions* — is not
  yet justified by its yield.
* **The meta-game itself is a keeper as infrastructure.**  The rule
  book (census legality + structural-sharing gate) and the
  oracle-budget referee are the reusable pieces: they make *any*
  construction player pay only for corpus-plausible candidates, and
  they caught real structure (commutations, a new naturality-ish
  view commutation) and a real reward exploit (padding) — both
  diagnostic value.
* **If the line continues, the next move is denser truth signal, not
  more episodes** — e.g. an RHS-completer that proposes completions
  around the *bound* subtrees of a matched instance, or a beam over
  partial candidates scored by a learned value on structure, where
  the sparse oracle stays the only judge of truth.  Whether that is
  worth doing is a project-level decision; the negative yield number
  is the evidence to weigh.
* **The emergent findings are worth a footnote elsewhere**: the
  padding family is a natural adversary test for the proposal
  pipeline (true, new, firing, useless — the ship gate is what
  stands between it and the library), and the broadcast-`allclose`
  caveat in the numeric oracle is a real, if narrow, hole.

## Gates

Run from the main worktree; `tools/law_meta_game.py` added, no
`packages/` change:

* `.venv/bin/ruff check tools/law_meta_game.py` — pass
* `.venv/bin/ruff format --check tools/law_meta_game.py` — pass
* `.venv/bin/python tools/law_meta_game.py --episodes 1500 --budget
  300 --random-calls 150 --seed 1` — runs to completion; yield table
  above

Per the task bound the full pytest/coverage gate was **not** run
(11 GB host; the full gate takes ~25 min); the tool's only shipped
surface is `tools/`.
