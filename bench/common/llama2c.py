"""llama2.c v2-style checkpoint parser — the ONLY thing the benches
need out of the old root `measure_weights.py` — the rest was the
weight-space probe axis, archived at `project/retros/` on the
project branch next to REPORT.md)."""

from __future__ import annotations

import numpy as np


def load_llama2c(path: str) -> dict[str, np.ndarray]:
    """Parse a llama2.c v2-style .bin: 7-int32 header
    (dim, hidden, n_layers, n_heads, n_kv_heads, vocab, seq) followed by
    fp32 weights in the documented order.  stories15M ties wcls to the
    embedding table."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(28), dtype=np.int32)
        dim, hidden, L, _h, _kv, vocab, _seq = (int(v) for v in hdr)
        data = np.frombuffer(f.read(), dtype=np.float32)

    w: dict[str, np.ndarray] = {}
    off = 0

    def take(name, shape):
        nonlocal off
        n = int(np.prod(shape))
        w[name] = data[off : off + n].reshape(shape)
        off += n

    take("token_embedding", (vocab, dim))
    take("rms_att", (L, dim))
    for _i, nm in enumerate(("wq", "wk", "wv", "wo")):
        take(nm, (L, dim, dim))
    take("rms_ffn", (L, dim))
    take("w1", (L, dim, hidden))
    take("w2", (L, hidden, dim))
    take("w3", (L, dim, hidden))
    take("rms_final", (dim,))
    if off < len(data):
        w["_tail"] = data[off:]
    return w
