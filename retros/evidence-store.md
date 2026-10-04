# The evidence store — verdicts are a pure function, so cache them

`tools/law_pipeline.py` recomputed every verdict on every run (~2 min
on this box; the retro's "~25 s" was a quieter machine) and scattered
the results across `/tmp` JSON dumps and prose retros — so "what did
we measure, and did it change?" was answered by grepping prose.  A
verdict is **deterministic given the corpus, the search rule set and
the verification code** — a perfect cache.  `tools/law_evidence.py`
is that store: a `sqlite3` database (stdlib only, no new
dependencies) wired into the pipeline as a pure side effect.

Reproduce:

    .venv/bin/python tools/law_pipeline.py --evidence-db /tmp/laws.db
    .venv/bin/python tools/law_pipeline.py --evidence-db /tmp/laws.db \
        --use-evidence-cache
    .venv/bin/python tools/law_evidence.py --report /tmp/laws.db
    .venv/bin/python tools/law_evidence.py --report /tmp/laws.db --ships

## 1. Schema

```sql
candidates(alpha_key TEXT PRIMARY KEY, name, family, proposal_json)
verdicts(alpha_key, corpus_hash, rules_hash, code_rev, run_id,
         holdout, numeric_true, derivable, witness_json, relation,
         census_sites, relaxed, matches, example, fires,
         fire_cases_json, changed, paid, verify_fail, drop_pct,
         cert, enode_ratio, verdict, ts,
         PRIMARY KEY (alpha_key, corpus_hash, rules_hash,
                      code_rev, run_id))
```

`alpha_key` is `repr(law_proposal._key(lhs, rhs))` — the alpha-normal
equality the pipeline already de-duplicates by.  The verdict columns
are exactly the fields the `Evidence` record computes; the store
measures nothing, it persists.  `drop_pct` is percent, `numeric_true`
is NULL/0/1, `verdict` is `"SHIP"` or `"no:<reason>"`.

## 2. The scope keys — when a cached verdict is valid

A hit requires **all three** scope keys to match:

| key | hashed over | invalidates on |
|---|---|---|
| `corpus_hash` | `source:name:shape_key(term)` per bench+model case | corpus change (new model, new shape) |
| `rules_hash` | `repr(_key)` of every search rule | `--holdout`, or any library edit |
| `code_rev` | `git rev-parse --short HEAD`, `+<diffhash>` of `packages`/`tools`/`bench` when dirty | any verification-code edit, committed or not |

A `"unknown"` rev (git cannot answer) is written but never served.
Two fields are not round-tripped: `match_term` (not serializable —
the admission emitter binds from `fire_cases`' live model terms) and
`reach`'s per-model rows (only their aggregates are read downstream).

## 3. Measured

Same corpus, same code, second run:

    evidence db: /tmp/laws.db (rev 76623dd+4089e750,
        corpus cffaf641ea55, rules 85f1cd92ec69)
        — 50/50 verdicts served from cache

**50/50 hits; 128 s → 6.6 s** (the residual is building the corpus
and running census/propose — required to know the keys).  A JSON diff
of a cache-read run vs a fresh-measure run: all 50 rows equal except
`closure_ratio` on 3 rows (2.35 vs 2.47, 1.64 vs 1.61, 35.43 vs
35.40) — see §5; verdicts, ranking and every other field identical.

## 4. The query that pays for the store

    $ law_evidence.py --report /tmp/laws.db --flips
      census:mul_select [census-naturality]
        05:24 run=bc39e932 corpus=cffaf641 rules=e505362d: SHIP
        05:20 run=c81a405c corpus=cffaf641 rules=85f1cd92: no:not new (duplicate)

One row of history tells the whole held-out story: `census:mul_select`
is a duplicate under the full library (`rules=85f1…`) and SHIP the
moment `select_mul` is held out (`rules=e505…`) — the verdict flip is
the *context* flip, visible in the same table.  `--ships` answers
"every candidate ever marked SHIP"; `--history <substr>` dumps every
recorded verdict for a candidate.

## 5. Honesty — what a hit means and what it does not

* **Scope, not prophecy.**  A verdict is `(corpus, rules, code)`-
  scoped.  A corpus edit, a library edit, or a `tensor.py` change
  under a dirty tree all correctly invalidate — the holdout run above
  took `rules_hash` and `code_rev` misses (0/50) exactly as designed.
  Never read a cached verdict as "still true" — read it as "true for
  that context".
* **`enode_ratio` is noisy at the source.**  Two *fresh* runs on an
  identical corpus produce different enode counts (set-iteration
  order differs per process — `PYTHONHASHSEED`).  The cache faithfully
  replays the recorded value; it does not manufacture the noise.
  Verdicts are stable because the ship gate reads the ratio against
  a 2.0× bound, far from the noise floor — a candidate near the bound
  could flip across fresh runs and the store honestly records both.
* **The db must live outside `tools/`** (or any hashed path): an
  untracked file under `tools/` is hashed into `code_rev`, so a store
  there would invalidate itself on every write.
* Re-measurement is never hidden: `--evidence-db` without
  `--use-evidence-cache` always measures fresh and appends; the
  `run_id` count in the report is literally "how many runs saw this
  verdict".

## Gates

`ruff check` / `ruff format --check` clean on `tools/law_evidence.py`
and `tools/law_pipeline.py`; pipeline runs to completion fresh,
cached, and under `--holdout select_mul` (rediscovery still PASS —
`census:mul_select` SHIP at rank 1 of 50).  No test suite run —
tooling-only change.
