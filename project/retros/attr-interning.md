# Attr interning — strict identity, lenient matching (the Const fix, one level down)

Verdict: **fixed strict.** `Op.make("clamp", x, min=0)` and
`Op.make("clamp", x, min=0.0)` are now different terms, and
`add_enode` puts them in different e-classes.  Pattern matching stays
numerically lenient (`dim=0` still matches a node spelled `dim=0.0`),
exactly the split commit `5dc62d8` gave `Const` leaves.

## 1. The defect

The numeric tower (`0 == 0.0 == False`, shared hashes) leaked into
attr identity at every layer:

* `ir._attr_key` carried *raw* values (`_hashable_attr` only repr'd
  the unhashables), so `Op.make`'s intern key and `Op.__hash__`
  coalesced `min=0` with `min=0.0`.
* `Op.__eq__` (dataclass default) compared the `attrs` dicts — the
  same leniency on the equality side.
* `egraph.types.ENode` field-compare ran `==` over the attr tuple —
  `add_enode`/`add_term` merged both spellings into one e-class, the
  term-level corruption reaching the graph.

Same class as the `Const` bug: interning was making a semantic
decision the IR is not entitled to make — int vs float spellings are
real programs (an int `min` promotes differently at lowering; a float
`dim` is malformed).  Cert replay compares `op_repr` (spelling-
strict), so a mixed-spelling workload would have recorded honest
mismatches — the intake scan found zero collisions today, but the
merge was silent *before* any law fired.

## 2. The fix — mirror `Const`

* `ir._hashable_attr` now keys every value by `repr` — `0`/`0.0`/
  `False`/`"0"` distinct, `[3,5]` vs `(3,5)` vs `(3.0,5.0)` distinct,
  `nan` reflexive.  One code path, no hash-probe fallback.
* `Op` is `eq=False` with a custom `__eq__` comparing `(op, args, _ak)`
  where `_ak` is the cached `_attr_key` — eq and hash now agree on the
  strict partition (a lenient eq over a strict hash would have been
  the dict-corruption bug all over again).
* `ENode` is `eq=False` with a precomputed repr-keyed `_sig` + cached
  `_h`.  The stored `attrs` tuple keeps raw values — the matcher reads
  them, and changing what it sees would have changed binding payloads
  (`$attr:` values reach `check`/`derive` and RHS instantiation).
* Matchers unchanged and deliberately lenient: `_term_match`,
  `meta.match_pattern`, `EGraph._m_stream`/`_m_bounded` all compare
  attr values with `!=`, and `$attr:` metavariables still bind the
  node's raw spelling.  Docstrings at each site now say so.

## 3. Why not document-and-keep-leniency

Considered, rejected: the leniency was not load-bearing for canonical
forms — the export boundary emits `int` dims and `float` eps/bounds
consistently, so the merge only ever fired on *mixed spellings*,
which are precisely the defect.  Coalescing also *repaired* bad
spellings silently (`dim=0.0` minted where `dim=0` existed returned
the int-spelled term), masking malformed attrs that strictness now
surfaces.  The strict-repr precedent was already established by
`Const`; uniform leniency would have left the same corruption class
alive at the enode layer.

## 4. Follow-ups / honest edges

* **catopt-native parity gap.** `packages/catopt-native/src/attr.rs`
  was written to *mirror* the Python semantics — `AttrVal`'s
  `Eq`/`Hash`/`Ord` still implement Python-numeric equality
  (`Int(1) == Float(1.0) == Bool(true)`).  Under `engine=` the Rust
  side still merges `0`/`0.0` enodes.  Port the strictness there
  (type-tag the ordering/hash) the next time the native package
  builds; until then mixed-spelling workloads can extract different
  members under the two engines.
* `catopt_discovery.census._attr_key` is a separate (raw-value) copy
  for census dedup — same latent conflation, lower stakes (a
  reporting key, not an intern table); aligned when it next matters.
* Raw scalars in `Op.args` (off-contract: operands should be `Const`)
  still hash/compare tower-lenient — consistent eq/hash, so sound,
  just not strict.  Leaf args are `Const`-strict already.
* `_sig`/`_ak`/`_h` add one tuple + one int per `ENode`/`Op` — the
  same per-object overhead `Op._h` already pays.
