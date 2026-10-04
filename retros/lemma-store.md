# Lemma store — laws as data, stored and re-admitted live

The condition-DSL retro (`condition-dsl.md`) closed the last data gap
in a rewrite law: a law is a 2-cell — a pattern pair, a side
condition, provenance — and with `cond` the condition is pure data.
What was still missing was the *record*: nothing serialised a whole
`Rewrite`, and nothing stored one.  This change adds the codec
(`catopt_core.laws.serialize`), lifts the term codec and the
alpha-normal key to core where the seam can reach them, and gives
`tools/law_evidence.py` a `lemmas` table with a `--admit` path that
rebuilds a stored row into a live `Rewrite` that fires.

Reproduce:

    .venv/bin/python -m pytest tests/test_law_serialize.py -q
    .venv/bin/python tools/law_evidence.py --report /tmp/l.db \
        --add-lemma softmax_fold
    .venv/bin/python tools/law_evidence.py --report /tmp/l.db \
        --admit '<alpha_key printed above>'

## 1. The mechanism

* `catopt_core.ir` — the term codec moved home:
  `term_to_data` / `term_from_data` / `attr_to_data` /
  `attr_from_data`, verbatim from `rulecache`.  The scheme is
  unchanged — `{"mvar": …}` for metavariables, `{"const": …}`,
  `{"var"/"param": name, "shape": […]}`, `{"op"/"args"/"attrs": …}`
  with `__tuple__`/`__list__` attr markers — so existing caches and
  fingerprints hold.  `rulecache` keeps its private names as
  aliases; the codec is now an IR-layer primitive, not cache
  machinery, which is what lets `laws.serialize` use it without
  pulling in `meta`.
* `catopt_core.laws.serialize` — the record codec:
  * `law_to_data(rule)` — every field as data: pattern pair through
    `term_to_data`, condition through `cond_to_data`, `tags` /
    `derivation` / `error_bound` / `bound_norm` as scalars, plus the
    honesty fields `serializable` and `missing_hooks`.
  * `law_from_data(data)` — rebuilds the `Rewrite`; a stored `cond`
    re-folds into `check` at construction, so a full-data record
    reproduces behaviour, not just structure.
  * `missing_hooks(rule)` — names the Python callables data cannot
    carry: `"check"` when the side condition is procedural (in whole
    or as a remainder over `cond`), `"derive"` when RHS attributes
    are computed in code.  The check detection compares the folded
    guard's `__code__` against a fresh `compile_guard(cond, None)`
    closure — the two guard branches in `compile_guard` are
    different code objects, so cond-only vs cond-plus-code is
    decided exactly, and anything else (a partial, a composed guard,
    a hand-written callable) is reported procedural: the strict
    posture, since a missed procedural check would silently weaken a
    stored law.
  * `alpha_key(lhs, rhs)` — the alpha-normal structural key, lifted
    from `tools/law_proposal` where it lived as `_key` (that name now
    delegates).  The evidence store keys lemma rows by `repr` of the
    pair — the same identity the `candidates`/`verdicts` tables
    already use, so a lemma row joins to the verdict that earned it.
* `tools/law_evidence.py` — a third table and two CLI verbs:

      CREATE TABLE lemmas (
          alpha_key TEXT PRIMARY KEY,   -- repr(alpha_key(lhs, rhs))
          name TEXT NOT NULL,
          law_json TEXT NOT NULL,       -- full law_to_data record
          derivation_json TEXT NOT NULL,
          corpus_hash TEXT NOT NULL,
          added_ts TEXT NOT NULL
      );

  `store_lemma(conn, rule, corpus_hash)` upserts;
  `admit_lemma(conn, alpha_key)` returns `(rule, record)` or
  `None` — the live `Rewrite` plus the stored record, so the caller
  sees `serializable`/`missing_hooks` before trusting it.
  `--add-lemma NAME` stores a shipped `ALL_RULES` law;
  `--admit KEY` rebuilds and prints it.  Both lazily import catopt —
  the verdict-report path stays dependency-free.

## 2. What is serializable — the honest census

`missing_hooks` over the 54 shipped laws (pinned by
`test_serializability_census_of_shipped_library`):

| class | count | members |
|---|---|---|
| **full-data** | **39** | 23 unguarded + 16 `cond`-only laws |
| pattern+cond, `derive` needs code | 14 | the 12 `sdpa_fold_*` family, `softmax_fold`, `qkv_fuse_asym` |
| pattern-only, `check` needs code | 1 | `gqa_absorb_repeat` |

`derive` stays Python by design — it *computes* RHS attributes, a
different role from a verdict — so the 14 derive laws are stored
*flagged*, not dropped.  The demo shows what that flag means, and
it is deliberately loud: admitting the stored `softmax_fold`
rebuilds pattern+cond, the rule **fires**, and instantiates
`softmax(u, dim="SD")` — the unbound `$attr:` metavar falls back to
its literal name (`_term_instantiate`'s `subst.get(k, v)`), a
visible scar instead of a silent veto.  `gqa_absorb_repeat` is the
one law whose side condition never made it to `cond` at all — the
pattern stores; the guard is `missing_hooks: ["check"]`.

## 3. The admit demo

    $ tools/law_evidence.py --report /tmp/l.db --add-lemma weight_factor_matmul
    stored weight_factor_matmul  [full-data]
      alpha_key = (('add', (('matmul', …
    $ tools/law_evidence.py --report /tmp/l.db --add-lemma softmax_fold
    stored softmax_fold  [missing hooks: derive]
    $ tools/law_evidence.py --report /tmp/l.db --admit '<key>'
    admitted weight_factor_matmul  [full-data]
      weight_factor_matmul: (add (matmul 'x', 'W'), (matmul 'x', 'W2'))
          -> (matmul 'x', (add 'W', 'W2'))

`tests/test_law_serialize.py` closes the loop a CLI print cannot:
the admitted `factor_matmul` fires on a known instance *and* declines
the documented rank-1 counterexample — the `cond` travelled through
sqlite and still vetoes.  The admitted `softmax_fold` fires and
mints the `dim="SD"` scar, asserted against the shipped rule's
derived `dim=-1`.

## 4. Scope — the seam, not the pipeline

This is deliberately the *seam*: laws-as-data, a store table, and
reconstruct-and-fire.  The emit machinery (`tools/law_emit.py`)
still emits code patches — `admitted_<law>.py` modules and unified
diffs — because a SHIP candidate's `check`/`derive` hooks are
transcribed Python source, not data.  The two now interlock
honestly: a lemma row's `alpha_key` is the same identity the
`verdicts` table records a SHIP under, so the store knows *which*
verdict justified a law; `missing_hooks` names which parts of the
law still need the code path.  Admission-from-DB — keying an
emitted artifact by alpha-key and attaching its transcribed hooks
back onto a stored pattern — is the next step, not this one.

## 5. What it unlocks

* A lemma is now a *row*: portable, diffable, loadable without the
  proposer's code — the axiom/lemma taxonomy (`rule.kind`) has a
  physical form.
* The 39 full-data laws can ship between processes as JSON with no
  loss; the 15 remainder are stored with an exact account of what is
  missing, so a future `derive`-DSL or named-hook registry knows
  precisely what to close.
* `alpha_key` in core means anything — store, cache, pipeline —
  can name a law without importing the proposer's torch stack.
