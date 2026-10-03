# baselines/

Committed golden reports, one `<suite>.json` per baselined suite, used
by `python -m bench compare` / `python -m bench gate` to catch
regressions.

Pin the current numbers for a suite:

```bash
python -m bench run reassoc_scale --device cpu --pin
```

`--pin` writes the **baseline projection** — the canonical report
minus the live spread (`iqr_s`).  `compare` / `gate` read `median_s`
only, so a golden reference carries exactly what it gates on: the
median plus provenance and findings.  The spread stays in every live
report (`bench/results/*.json`, the ledger), where the renderers show
`median ± IQR`; it is left out of the pinned file so a later change to
the timing contract (`catopt_core.timing`) cannot leave a frozen
baseline recording a number the harness no longer agrees with.

Then a later run is compared against it:

```bash
python -m bench run reassoc_scale --device cpu      # appends to the ledger
python -m bench compare reassoc_scale               # fails on >5% slowdown
python -m bench gate                                # every baselined suite
```

Baselines are device-specific — pin them from the same hardware you
gate on (CPU baselines and CUDA baselines are not interchangeable).
