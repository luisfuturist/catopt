# Retro: the derivable gate — a measured counterexample outranks a proof

**Plan 0017 / ADR 0004** — the truth gate's derivability
short-circuit.  Date: 2026-10.

## The hole

`evidence._truth_gate` decided stage 4 with

```python
ok = evd.derivable or (
    _region_ok(rep.synth_region, need_equal=True)
    and _region_ok(rep.real_region, need_equal=False)
)
```

for a `cond`-carrying object, and `evd.derivable or evd.num_true is
True` for an unguarded one.  A rule the saturation **derived** therefore
passed *without* a clean guarded-region sweep: derivability short-
circuited the region check entirely.  The newly composable
`channel_then_row` (guarded composition, plan 0017 stage 3) measured
19 equal / **1 unequal** and was admitted anyway — the same class as the
`select_mul` unsoundness: a sound-looking derivation over a latent,
unsound region.

The unequal site is *inherited*, not introduced: `channel_then_row =
compose(linear_channel_scale_rev, linear_row_scale)`, and the composite
guard is the premises' conjunction, so it is exactly as tight as
`linear_row_scale`'s own guard — which accepts a rank-1 degenerate
weight (`W = c = (1,)`) where `linear` mis-evaluates.  Guard transport
is faithful (it inherits, never adds, a blind spot), and derivability
was laundering that blind spot past the sweep.

## The audit

Every currently-admitted object was swept for measured-unclean
regions — `unequal > 0` or `rhs_err > 0` in the guard-accepted region
(`_guarded_truth` at the default 360-site window).  The store's
admitted set is the objects the gauntlet clears; the shipped guarded
rules are the library's `cond`/`check`-carrying entries.

> *Measurement state.*  The counts below were taken against the
> working tree, which carries a concurrent, uncommitted `oracle.py`
> change (a `_CONST_DOMAIN` value-bank widening — sibling work) that
> feeds the enumeration.  The classification (clean vs unclean) is
> the finding; the exact counts are window- and bank-dependent, as
> they always were.

### Shipped guarded rules (36 of 61 carry a guard)

| rule | synth eq / ne / rerr / accepted |
|---|---|
| `select_mul` | 57 / 0 / **8** / 81 |
| `distribute_matmul_over_add` | 5 / 0 / **4** / 48 |
| `weight_distribute_matmul` | 5 / 0 / **4** / 41 |
| `weight_distribute_linear` | 4 / 0 / **4** / 41 |
| `linear_channel_scale` | 24 / 0 / **16** / 128 |
| `linear_row_scale` | 22 / **3** / **9** / 232 |
| `linear_row_scale_rev` | 22 / **3** / 0 / 232 |

`gqa_absorb_repeat` **crashes the sweep**
(`TypeError: not all arguments converted during string formatting`,
`typing._infer_op_shape` — a str attr metavar reaching `d % (…)`) — a
pre-existing oracle-domain bug, recorded here, not this change.

### Store-admitted objects (constructed, gauntlet-cleared)

| object | synth eq / ne / rerr / accepted |
|---|---|
| `channel_then_row_scale` | 19 / **1** / 0 / 112 |
| `sdpa_fold_nomask` | 49 / 0 / 0 / 224 |
| `sdpa_fold_div_nomask` | 19 / 0 / 0 / 19 |
| `sdpa_fold_div_nomask_g` | 16 / 0 / 0 / 53 |
| `affd_step_lift` | 299 / 0 / 0 / 360 |
| `affd_scan2_lift` | 360 / 0 / 0 / 360 |
| `affd_scan4` | 360 / 0 / 0 / 360 |

(`mul_unsqueeze_l_id` / `sub_unsqueeze_l_id` / `softsign_fold` /
`aff_step_lift` / `aff_scan2_lift` / `silu_fold_commuted` / `om_lift`
are the other admitted objects; all clean.)

**The exposure.**  Seven shipped guarded laws and one admitted
composite carry a measured-unclean guard-accepted region.  Under the
old gate the composite was *usable*; the shipped laws were never
gauntleted (human-admitted), so the audit is the first time their
regions were measured at all.  The `rhs_err` sites are the honest
"minted RHS cannot denote" signal the sweep already records (a rank-1
or ill-shaped corner the guard lets through); the `unequal` sites are
the same latent-unsoundness class as `select_mul`.

## The rule

Derivability is legitimate evidence — a derivation from shipped axioms
*is* a proof.  But a **measured counterexample outranks it**: a proof
that the equality holds everywhere cannot coexist with a measured
instance, inside the guard's own region, where it does not.  The proof
is the thing that must yield.

So the gate now reads:

```python
clean = _region_clean(rep.synth_region) and _region_clean(
    rep.real_region
)
ok = clean and (evd.derivable or rep.synth_region.equal >= 1)
```

- `_region_clean(region)` is the counterexample test alone —
  `not (unequal or rhs_err or guard_err)`.
- A measured counterexample blocks **regardless** of `derivable`.
- `derivable` may waive **only** the starvation requirement
  (`synth_region.equal >= 1`) — the "the guard accepted nothing
  provable, but the derivation is the proof" case.
- The gate detail appends `" | derivation overridden by a measured
  counterexample"` when a derivation loses to a measured site, so the
  override is visible, not silent.

`_region_ok(region, *, need_equal)` is refactored to
`_region_clean(region) and (not need_equal or region.equal >= 1)` —
behaviourally identical to before, now expressed in the two primitives.

The **unguarded** branch gets the same rule off the numeric oracle:

```python
ok = evd.num_true is not False and (
    evd.derivable or evd.num_true is True
)
```

A measured `num_true is False` blocks; `derivable` waives only the
"no measurement" (`num_true is None`) case.

`grep derivable evidence.py` finds no other *gate* use — lines 108 /
140 / 295 / 398 / 440 / 1017 / 1563 are schema, serialisation, the
corpus-invariant column contract and report rendering.  The **sibling
instance** lives in `pipeline.Evidence.truth` (`derivable or num_true
is True`), which `shippable` / `no_ship_reason` consult; it has the
same shape but is *out of this change's scope* (the pipeline's ship
bar for candidates, not the gauntlet) — flagged here as a follow-up,
not silently fixed.

## Premise blind-spot propagation — honest design

*Should the derivation's own premises be checked?*  In principle a
derivation from a premise with a known unclean region inherits it —
`channel_then_row` is the worked example.  In practice a proactive
premise gate is **not cleanly implementable**:

1. **Reachability is undecidable from the guard.**  A premise's blind
   spot is a property of *its* pattern; whether the composite can
   instantiate that pattern's LHS in a binding the guard accepts is not
   decidable from the guard alone.  A conservative "refuse if any
   premise is unclean" would over-tighten — it would reject composites
   whose premise's blind spot is unreachable in the composite (the
   honest direction the gate must not go).
2. **Premises are not always shipped.**  `affd_scan4`'s derivation
   names a *constructed* premise (`affd_scan2_lift`); resolving
   premises to rules is a partial map, and "known unclean" would need a
   measured premise-region table the store does not keep.

What *is* cleanly implementable — and what the fix does — is the
**direct** composite sweep: the inherited blind spot lands *in the
composite's own region* (1 `unequal` here), so the gate measures it and
blocks.  That is the honest epistemic bound: an inherited blind spot
outside the swept window is not caught, exactly as a counterexample
outside the finite banks is not caught.  The fix moves the gate from
"derivation wins over the sweep" to "the sweep is the referee, and a
derivation is the tie-breaker when the sweep is starved".

## The re-admission delta

Re-running the gauntlet over the affected set:

| object | before | after |
|---|---|---|
| `channel_then_row_scale` | **usable** | refused — `truth` |
| `mul_unsqueeze_l_id`, `sub_unsqueeze_l_id` | usable | usable (clean region) |
| `sdpa_fold_nomask`, `sdpa_fold_div_nomask`, `sdpa_fold_div_nomask_g` | usable | usable (clean region) |
| `affd_step_lift`, `affd_scan2_lift`, `affd_scan4` | usable | usable (clean region) |
| `softsign_fold`, `aff_step_lift`, `aff_scan2_lift`, `silu_fold_commuted`, `om_lift` | usable | usable (unguarded, `num_true is True`) |
| `comm_guarded` (comm_mul + `leaf`) | truth pass, novelty fail | unchanged |

**Exactly one re-refusal** — `channel_then_row_scale`, the object the
hole admitted.  Nothing else moves: the fix is not a blanket
tightening, it is the one site where a derivation was standing on a
measured contradiction.  The seven shipped guarded laws above would
also refuse *were they gauntleted*; they are human-admitted, so the
audit is the exposure, not a delta.

## Tests

- `tests/test_discovery_gauntlet.py` — a new section *A measured
  counterexample outranks a derivation*:
  `test_derivation_does_not_override_a_measured_counterexample` (the
  full gauntlet on `channel_then_row_scale`: derivable **and** refused,
  1 `unequal`); `test_derivable_but_clean_rule_still_admits` (the
  no-over-tightening side); `test_derivable_waives_only_the_starvation_requirement`
  (`sdpa_fold_masked_fill`: derivable + starved passes, the same rule
  without derivability is refused as vacuous);
  `test_derivation_cannot_override_a_measured_rhs_err`
  (`linear_channel_scale`, 16 rerr);
  `test_derivation_cannot_override_an_unequal_site`
  (`linear_row_scale`, 3 ne); `test_unguarded_derivable_false_still_refuses`
  (the numeric-oracle branch); `test_region_clean_ignores_equal_count`
  (the primitive).
- `tests/test_discovery_synthesis.py` —
  `test_declarative_transport_clears_the_gauntlet` becomes
  `test_declarative_transport_refused_by_an_inherited_counterexample`:
  the composite that used to clear the gauntlet now refuses at `truth`,
  and the gate detail carries the override.

## Gates

`pytest tests/test_discovery_gauntlet.py tests/test_discovery_evidence.py
tests/test_discovery_autocond.py tests/test_discovery_synthesis.py`
(137 passed), `ruff check` / `ruff format --check`, `ty check`,
`radon_ratchet`, `vulture` — all clean **against the committed
`oracle.py`** (verified in an isolated `git worktree` at HEAD with
this change applied).  No new suppressions.  `pipeline.py`,
`oracle.py`, `intake.py`, `object_synthesis.py`, `cond.py` untouched.

> *Concurrent working-tree state.*  While this change was being
> verified, the working tree's `oracle.py` carried an in-flight,
> uncommitted sibling edit (the `_CONST_DOMAIN` value-bank widening,
> `project/retros/value-bank.md`).  That edit moves the enumeration
> and, on its own, fails the radon ratchet (`oracle.py::_leaf_bindings`
> 6 → 10) and a handful of tests that read region counts directly
> (`test_guarded_truth_sweep_counts`, `test_guarded_truth_domain_gap_is_not_rescued`,
> and the autocond `mul_is_add` refusals — the widened bank now mints
> the degenerate `Const(0)` corner where `mul(u,v) == add(u,v)`).  The
> exact set drifts as that sibling edit iterates; none of those tests
> touch `_truth_gate`, and all pass again against HEAD's `oracle.py`.
> The eight new tests here are written to pin the *gate's* logic, not
> a particular bank, so they hold across the change.

