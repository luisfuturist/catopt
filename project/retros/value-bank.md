# Retro: the per-op value bank — rescuing the domain-gapped guards

Date: 2026-10-07
Context: `cap-policy.md`'s remaining blind spot #1 — "the value bank
is still finite".  The selective cap widened the *window*; it never
widened the *values*.  A guarded rule whose `const-cmp` demands a
literal the free leaf bank never minted measured **0 accepted sites**
at every cap, so the gauntlet's truth gate refused it as *vacuous*
("a guard that accepts nothing provable is not verified") even though
the guard is shipped and sound.  `evidence.py`, `intake.py`,
`object_synthesis.py`, `_binding_envs` ordering were not touched.

## The gap, characterized

For a free (non-viewed) operand the leaf bank minted exactly one
literal — `Const(0.5)` — plus scalar `Var`s.  A guard's `const-cmp`
reads the bound leaf's `.value` (a `Const`), so a guard demanding
`1`, `2`, `-0.5` or `< -1e30` could never be satisfied: the site was
not *declined* because the guard was wrong, but because the bank
could not construct the value the guard names.

Each shipped domain-gapped guard names a literal its **parent op**
makes meaningful:

| rule | guard clause | literal | parent op | after the bank |
|------|--------------|---------|-----------|----------------|
| `div_sqrt_to_rsqrt` | `const-cmp ONE == 1` | `1` | `div` | **rescued** (15 accepted) |
| `pow_to_rsqrt` | `const-cmp P == -0.5` | `-0.5` | `pow` | **rescued** (18 accepted) |
| `rms_norm_fold` / `_nogain` | `const-cmp P == 2` | `2` | `pow` | still starved — see below |
| `sdpa_fold_masked_fill` | `const-cmp F < -1e30` | `-inf` | `masked_fill` | accepted, all `lhs-err` — see below |

**Every needed value is class (a) — derivable from the op's
semantics.**  `1` is `div`'s unit numerator, `-0.5` is the
reciprocal-root exponent `pow` already means (`x**-0.5 == rsqrt(x)`),
`2` is the square exponent, `-inf` is the additive-mask sentinel
`masked_fill` fills with.  None is class (b) — an arbitrary per-op
declaration.  So the table is a *typing* of the ops' own semantics,
not a new vocabulary: the values are read off the ops, not invented.

Two guards the value bank alone does **not** close (recorded, not
hidden):

* `rms_norm_fold*` — `P == 2` is now mintable, but the guard's real
  blocker is its trailing `shape-eq w (tail-block u MD)` clause: it
  passes **0** times over the whole enumerated domain (measured to
  6000 sites).  That is a *shape* gap, not a value gap — the value
  bank cannot reach it.
* `sdpa_fold_masked_fill` — the `-inf` sentinel makes the guard
  *accept* a site, but the mask leaf `MK` is a float `Var` and
  `masked_fill` needs a bool mask, so the accepted site is an
  `lhs-err`: the region still shows no equal instance.  That is a
  *kind* gap (a bool leaf), separate from the value gap.

## What changed

One declarative table and a two-line splice, all in
`catopt_discovery.oracle` (the leaf/value bank):

1. **`oracle._CONST_DOMAIN`** — a small per-op table keyed by the
   metavariable's *parent op*:

   ```python
   _CONST_DOMAIN = {
       "pow": (2, -0.5, 0.5, 1),
       "masked_fill": (-inf, -1e30, 1e9),
       "div": (1, 0),
   }
   ```

   `-inf` (not `-1e30`) is the entry that clears the *strict*
   `F < -1e30`; the finite analogues ride along honestly.

2. **`oracle._const_domain(parents)`** — resolves a metavariable's
   parent ops to their literals (sorted op, deduped), falling back to
   the generic `Const(0.5)` when no parent carries an entry.

3. **`oracle._insert_op_consts` / `_leaf_bindings`** — the op
   literals are spliced in **just past** the generic corner, so the
   documented index-0 (derived shape) / index-1 (`Const(0.5)`) /
   index-2 (scalar `Var`) seats are unshifted.  A widening must not
   bury a corner that already measured.

4. **`_binding_envs`** now passes each free metavariable's real
   parent set to `_leaf_bindings` (`parents[m]` instead of `set()`) so
   the table can be keyed.  The **ordering is untouched** — the
   Cantor diagonal, the base/free interleave, the bank's first three
   entries are byte-for-byte what they were.

`mul` / `add` scalar identities were measured and **left out**: no
domain-gapped guard needs them, and minting them into every free
operand under those ops perturbs the enumeration enough to push an
*already* rescued corner out of the window (`mul_unsqueeze_l_id`'s
equal count moves 23 → 21).  The bank feeds the enumeration, so a
widening with no rescue to show for it is not paid for.

## Measured

Across all **36 shipped guarded rules**, old bank vs new (the old
bank is reproduced exactly by an empty `_CONST_DOMAIN`):

| metric | old bank | per-op bank |
|--------|----------|-------------|
| rules with a non-empty accepted region | 20 | **23** |
| total envs evaluated | 44270 | **43949** |
| wall (guarded sweeps) | ~5.2 s | ~5.1 s |

**Delta: 3 rules** go from `synth: 0 accepted` to a non-empty region —
`div_sqrt_to_rsqrt` (15 accepted), `pow_to_rsqrt` (18),
`sdpa_fold_masked_fill` (1).  The cost is *negative*: the rescued
rules now accept at the **first** window, so they no longer pay the
escalation — `div_sqrt_to_rsqrt` 450 → 255 envs, `pow_to_rsqrt`
450 → 324.  `sdpa_fold_adddiv` (already rescued) gains 6 accepted at
the same 2360 envs; the still-gapped `masked_fill*` variants and
`rms_norm_fold*` pay the same escalation as before.

## Remaining blind spots

1. **The `nan` artifact.**  The rescued `div_sqrt_to_rsqrt` /
   `pow_to_rsqrt` regions carry `unequal` sites that are `nan ==
   nan`: the random env (`torch.randn`) feeds negative operands, both
   sides are `sqrt`/`rsqrt` of a negative → `nan`, and
   `torch.allclose` (with `equal_nan=False`) calls that *unequal*.
   So the guard is sound but the truth gate still sees a
   counterexample.  The fix is a tolerance question in
   `proposal._allclose` (or a positive-only env), not a value-bank
   one — recorded here.
2. **The `kind` gap.**  `sdpa_fold_masked_fill*` needs a **bool**
   mask leaf; the bank mints float `Var`s.  A per-op *dtype* bank (the
   natural twin of this per-op *value* bank) is the next step; the
   scaled `_drop` variants also need a deeper cap than the ceiling.
3. **The `shape` gap.**  `rms_norm_fold*` needs its `tail-block`
   clause satisfied — a shape fact, not a value; the value bank is
   orthogonal.
4. **The table is still hand-keyed.**  The entries are read off the
   shipped guards' `const-cmp` demands; a guard on a new op needs a
   new row.  A *derived* domain — the guard's own `dspec`-style
   proposal of the literal it wants — would close the class instead
   of the instances (the cap-policy retro's option (b), still not
   taken).
