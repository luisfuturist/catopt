# baselines/

Committed golden reports, one `<suite>.json` per baselined suite, used
by `python -m bench compare` / `python -m bench gate` to catch
regressions.

Pin the current numbers for a suite:

```bash
python -m bench run reassoc_scale --device cpu --out bench/results
cp bench/results/reassoc_scale.json bench/baselines/reassoc_scale.json
```

Then a later run is compared against it:

```bash
python -m bench run reassoc_scale --device cpu      # appends to the ledger
python -m bench compare reassoc_scale               # fails on >5% slowdown
python -m bench gate                                # every baselined suite
```

Baselines are device-specific — pin them from the same hardware you
gate on (CPU baselines and CUDA baselines are not interchangeable).
