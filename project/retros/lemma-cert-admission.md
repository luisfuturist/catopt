# Lemma-cert admission — the stored lemma carries its proof

`lemma-certificates.md` materialized each `Rewrite.derivation` as a
replayable `Certificate`, and `lemma-store.md` made an admitted law a
row in the `lemmas` table.  What was missing was the join: a stored
lemma carried `derivation` — the premise *index* — but not the
certificate, the *checkable artifact*.  This change wires the
certificate into the admission seam: `store_lemma` materializes the
recorded derivation at write time and the record's new `"cert"` field
carries it; `--admit` replays it strictly off the row.

Reproduce:

    .venv/bin/python -m pytest tests/test_lemma_certificates.py \
        tests/test_law_serialize.py tests/test_derive_laws.py -q
    .venv/bin/python tools/law_evidence.py --report /tmp/l.db \
        --add-lemma factor_matmul
    .venv/bin/python tools/law_evidence.py --report /tmp/l.db \
        --admit '<alpha_key printed above>'

## 1. What got wired — the existing seam, extended

The narrowest honest wiring turned out to be the `lemmas`-table seam
itself — no new pipeline stage, no change to
`law_emit.emit_admission` (an emitted artifact is a candidate's
*source*, not yet a `Rewrite`; the cert attaches when the admitted
law is stored):

* `catopt_core.laws.serialize.law_to_data(rule, cert=None)` — the
  record gains `"cert"`: `cert_to_data` output when a
  `Certificate` is supplied, `null` otherwise.  The caller builds the
  certificate (`tools/law_lemma_cert.materialize`); the codec only
  carries it.  `law_from_data` ignores the field — the certificate is
  record-level provenance, not part of the 2-cell; nothing about the
  reconstructed `Rewrite` changes.
* `tools/law_evidence.store_lemma(..., cert=_UNSET, universe=None)` —
  the default materializes the rule's recorded `derivation` through
  `materialize` under the named premises drawn from `universe`
  (default `ALL_RULES`) and embeds the resulting `Certificate`.
  `cert=None` suppresses materialization; an explicit `Certificate`
  embeds as given.  A non-linear verdict (`saturation-only`, `gap`,
  `bad-derivation`) stores `cert: null` — a merge witness that cannot
  replay standalone is never stored as a proof.
* `tools/law_evidence.stored_certificate(record, rules=None)` — the
  read half: decodes `record["cert"]` via `cert_from_data` (rules
  resolved by name, default `ALL_RULES` — a stored derivation's
  premises are shipped laws by construction) and runs
  `verify_certificate(strict=True)`.  `cert: null` returns `None`;
  a cert that fails strict replay raises — a corrupt or drifted row
  is a failure to surface, not a flag.
* CLI — `--add-lemma` prints the stored cert's shape
  (`cert: 1-step derivation ['distribute_matmul_over_add']` or
  `cert: none recorded`); `--admit` rebuilds the `Rewrite` and
  replays the cert strictly, exiting 1 on a failed replay.

## 2. What the cert field looks like

`"cert"` is verbatim `cert_to_data` output — `CERT_FORMAT = 1`
records — embedded in the `law_to_data` record:

    "cert": {
      "version": 1,
      "src":  <term_to_data of the instance's src>,
      "dst":  <term_to_data of the instance's dst>,
      "steps": [{"rule": "distribute_matmul_over_add", "path": [],
                 "lhs": <term>, "rhs": <term>,
                 "bindings": {"W": {"term": …}, "$attr:D": {"attr": …},
                              …},
                 "egraph_dependent": false, "note": ""}],
      "rules_used": ["distribute_matmul_over_add"],
      "replayable": true
    }

Three notes on the shape:

* **Instance, not pattern.**  `src`/`dst` are the *concrete* instance
  the derivation was materialized on (`lc._instance(rule)`) — the
  same pair `materialize` proved.  The certificate says "these two
  terms connect under these steps", and that is exactly what
  `verify_certificate` re-checks.
* **Rules by name.**  Steps reference rule *names*; the name→rule map
  at decode time (`ALL_RULES` by default) is what re-attaches the
  hooks data cannot carry — the same posture as `missing_hooks`.
* **Direction is the axiom's.**  13 of the 15 derivation-carrying
  rules certify `rhs->lhs` — the premise fires forward on the lemma's
  RHS instance (equality is symmetric; the lemma direction is
  unreachable by forward premise rewriting, which is why the twin
  ships).  The stored cert records the direction it actually
  replays, `src`/`dst` included.

## 3. Format — `cert` rides `LAW_FORMAT = 2`, no bump

`law_from_data` reads explicit keys only, so an unknown field was
never going to be a format change: older readers ignore `"cert"`,
and older records (no `"cert"` key) load unchanged.  Adding it as an
optional v2 field is the honest choice — the alternative (v3) would
have been a version bump to signal nothing.

## 4. The honest boundary

* **`cert: null` is an absence claim, not a stub.**  A rule with no
  `derivation` annotation is never even attempted — the certificate
  proves the *annotation's* claim.  `right_factor_linear` is the
  pinned case: `derivation == ()`, `materialize` under `ALL_RULES`
  is `gap` (the NT-bridge derivation exists only under
  `--with-layout`, and the annotation never claimed it), and the
  stored record says `cert: null`.  No certificate is claimed where
  none replays.
* **A cert proves the derivation, not the law.**  The certificate is
  a replayable witness that `src -> dst` connects under the named
  premises — on ONE concrete instance.  It certifies the recorded
  derivation replays; it does not make a false law true, does not
  re-check the law's `cond`/`check` side conditions at other
  bindings, and does not certify semantic validity for instances the
  materializer didn't see.  A law that is *unsound* but whose
  derivation happens to replay would store a perfectly valid cert —
  the proof is of the derivation, not the law.
* **Verification is strict, and failures raise.**  `stored_certificate`
  refuses `egraph_dependent` steps outright; a stored cert that stops
  replaying after a library edit raises at admit time — the same
  drift-detection posture the lemma-cert tool took, now reachable
  from the store.
* **Premise universe is ambient.**  Steps resolve against the rule
  map supplied at decode (`ALL_RULES` by default).  A lemma whose
  premises were other *admitted* (not yet shipped) lemmas would need
  that map passed in — flagged, not handled: today's 15 derivations
  all ground out in shipped rules.

## 5. Status after this change

* All 15 derivation-carrying rules store a `cert` on `--add-lemma`
  (each is `linear`, 1 step, `premise_cover` — the table
  `tools/law_lemma_cert.py` measures); `--admit` replays each
  strictly.
* The lemma row is now complete end-to-end: pattern + cond (+ dspec)
  + derivation *and* its replayable proof — `pattern + cond +
  derivation + certificate`, where `derivation` names the premises
  and `cert` is the artifact a verifier can check without the
  e-graph.
* Not done (deliberately): `emit_admission` does not attach certs —
  a candidate becomes a `Rewrite` only after its review patch lands,
  and the cert materializes at `store_lemma` time, which is where an
  admitted law's derivation is actually recorded anyway.  Certifying
  a *proposal* would require computing its derivation first
  (`law_coherence --emit-basis`), a different step.
