# The object record — laws as the first declared objects

**Plan 0017, stage 1** (implements ADR 0004 §1, "objects-as-data").
Date: 2026-10.

## What landed

The `lemmas` table of the evidence store
(`catopt_discovery.evidence`) generalised from "verdicts + law
records" to the **declared-object store**: a row is a serialized
*declaration* — data the referee can replay into a live object the
same way `--admit` already reconstructs a stored lemma.

The record is the `law_to_data` v2 record **plus one field**:

| field | content |
|---|---|
| `version` | `LAW_FORMAT` — still **2** |
| `kind` | the declaration's provenance: `"law"` \| `"abstraction"` \| `"bridge"` (`OBJECT_KINDS`) |
| `name`, `law` | the object's identity and statement |
| `lhs`, `rhs` | the pattern pair (`term_to_data`) |
| `cond`, `dspec` | declarative guard / derive data |
| `tags`, `derivation` | taxonomy + axiom-lemma bookkeeping |
| `cert` | optional replayable certificate (`cert_to_data`) |
| `error_bound`, `bound_norm` | the ε-law axis |
| `serializable`, `missing_hooks` | the honesty flags |

`law_to_data` emits `"kind": "law"`; `object_to_data(rule, cert=*,
kind=…)` stamps one of `OBJECT_KINDS` (a required keyword — a
declaration must say what it declares); `object_from_data` rebuilds
the `Rewrite` for any *known* kind and raises `ValueError` on an
unrecognised one. `object_kind(data)` reads the field, defaulting
absent to `"law"`.

## Why kind-in-record, not a column

Two placements were viable — a `kind TEXT` column on `lemmas`, or
the `kind:` field inside `law_json`. The record field won on the
smaller *honest* diff:

1. **Zero migration.** The schema is `CREATE TABLE IF NOT EXISTS` —
   a no-op on existing stores, so a column would have needed an
   `ALTER TABLE … ADD COLUMN` probe (`PRAGMA table_info`) in
   `connect()`. A record field needs none: readers that predate it
   ignore it (the `law_from_data` explicit-keys posture — the same
   argument that let `cert` ride v2 without a bump), and writers
   that predate it produce rows that still load, reading as
   `"law"`. LAW_FORMAT stays 2; only a change to *reconstruction
   semantics* would earn a bump, and kind is record-level
   provenance, not part of the 2-cell.
2. **The record is the self-describing document.** `version`,
   `serializable`, `cert` already live inside it — provenance
   travels with the payload, and a column would duplicate data the
   JSON already carries, drifting when the two disagree.
3. The cost is honest and small: SQL-level kind queries need
   `json_extract(law_json, '$.kind')` — acceptable until a query
   actually wants it.

Naming caveat kept explicit: the record's `"kind"` is **not**
`Rewrite.kind` — that property is the kernel taxonomy
(axiom/lemma/redundant) and stays derivable from `derivation`.
The record field is the *declaration kind*: what the store row
claims to be.

## The provenance distinction

`"kind"` marks *where the record came from*, not what it can do:

- `"law"` — shipped-law provenance: an `R(...)` in `laws/*.py` that
  went through code review, or a lemma the discovery pipeline
  verified and stored. Every record written before this stage is
  `"law"` by construction.
- `"abstraction"` — an introduced abstraction: an object the
  construction machinery synthesizes and declares as data (the
  carrier-style objects of ADR 0004 §context: online-softmax
  monoids, `omd`, affine scans).
- `"bridge"` — a spelling bridge: a declared object that connects
  surface spellings (the second shape every shipped machine-found
  law takes — fold or bridge).

Only `"law"` has inhabitants today — deliberately. Stage 1 is the
*seam*, not the object: `store_object` / `stored_object` /
`admit_object` and `--add-object` / `--admit-object` (with the
lemma names kept as law-flavoured spellings of the same path, and
`--kind` declaring non-law provenance). The demo test
(`test_admitted_object_fires_identically`) stores `silu_mul_form`
as an object record, admits it, and shows it fires identically on
a small EGraph with its certificate replaying strict.

## What a novel (non-law) object still needs

The record shape is necessary, not sufficient — a stored
`"abstraction"` is a *claim*, and stage 3's gauntlet is what makes
it admissible:

- **a `cond` precondition that exists as data** — an object whose
  side condition can't be written declaratively is not admissible
  as data (ADR 0004: the Python-hook path stays, under the same
  review bar a hand-written `R(...)` faces; the
  `w:v_commutes_view`/`id:same_pairing` gaps are the standing
  `cond` work);
- **the adversarial gate** — numeric oracle → view/index oracle →
  typed-pay → certificate replay, with the sweep testing the
  `cond`'s *false* region too (the `select_mul` lesson);
- **`serializable: true`** — a novel object carrying
  `missing_hooks` is stored flagged, trusted exactly as far as a
  flagged law is today: it rebuilds pattern+cond and no further;
- **a kind-appropriate body** — every current kind is
  `Rewrite`-shaped; when an abstraction needs a non-rewrite body
  (e.g. a fresh operator the patterns reference), the dispatch
  lands in `object_from_data`, and `OBJECT_KINDS` grows a schema,
  not a side-door.

## Boundaries

- `cert: null` remains the honest absence — `cert=` suppression and
  materialization failures behave exactly as for lemma rows.
- `store_lemma` / `admit_lemma` / `--add-lemma` / `--admit` are
  unchanged in behaviour — law-flavoured spellings delegating to
  the object seam.
- The report's `N lemmas` count covers every object row — same
  table, same rows; a kind-broken-down report is left until a
  reader wants it.
