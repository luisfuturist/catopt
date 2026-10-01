"""Fetch the llama2.c TinyStories checkpoints into the catopt cache.

Downloads ``stories15M.bin`` / ``stories110M.bin`` from Hugging Face
(``karpathy/tinyllamas``) into ``$XDG_CACHE_HOME/catopt``
(default ``~/.cache/catopt``), verifies the 7-int32 header and the
file size, and skips files already present and valid.  A valid copy
already sitting in ``/tmp`` (the path older runs of these benches
used) is copied into the cache instead of re-downloaded.

    python bench/fetch.py                    # both models
    python bench/fetch.py --models 15M       # just one
    python bench/fetch.py --models 15M,110M
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = "https://huggingface.co/karpathy/tinyllamas/resolve/main"
MODELS = {
    "15M": "stories15M.bin",
    "110M": "stories110M.bin",
}
CHUNK = 1 << 20  # 1 MiB


def cache_dir() -> Path:
    """$XDG_CACHE_HOME/catopt, falling back to ~/.cache/catopt."""
    root = os.environ.get("XDG_CACHE_HOME") or str(
        Path.home() / ".cache"
    )
    return Path(root) / "catopt"


def _core_floats(
    dim: int, hidden: int, n_layers: int, vocab: int
) -> int:
    """fp32 count of the documented llama2.c weight layout.

    Checkpoints may carry an extra fp32 tail (freq tables, untied
    classifier) — the loader reads a ``_tail`` key for it — so this is
    a *minimum* size, not an exact one.
    """
    L = n_layers
    return (
        vocab * dim  # token_embedding
        + L * dim  # rms_att
        + 4 * L * dim * dim  # wq, wk, wv, wo
        + L * dim  # rms_ffn
        + 3 * L * dim * hidden  # w1, w2, w3
        + dim
    )  # rms_final


def validate(path: Path) -> tuple[bool, str]:
    """Sanity-check a .bin: 7-int32 header + plausible fp32 body."""
    try:
        size = path.stat().st_size
    except OSError:
        return False, "unreadable"
    if size < 32:
        return False, f"too small ({size} B)"
    try:
        with open(path, "rb") as f:
            hdr = np.frombuffer(f.read(28), dtype=np.int32)
    except OSError as e:
        return False, f"unreadable: {e}"
    if hdr.size != 7:
        return False, "short header"
    dim, hidden, n_layers, _nh, _kv, vocab, _seq = (int(v) for v in hdr)
    vocab = abs(vocab)  # llama2.c: negative means tied wcls
    if dim <= 0 or hidden <= 0 or n_layers <= 0 or vocab <= 0:
        return False, f"bad header {hdr.tolist()}"
    core = 28 + 4 * _core_floats(dim, hidden, n_layers, vocab)
    if size < core:
        return (
            False,
            f"truncated: {size} B < {core} B implied by header",
        )
    if (size - 28) % 4:
        return False, f"size {size} B not fp32-aligned"
    return True, (
        f"dim={dim} hidden={hidden} n_layers={n_layers} "
        f"vocab={vocab} — {size / 1e6:.1f} MB"
    )


def download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a .part temp file, with progress."""
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(
        url, headers={"User-Agent": "catopt-fetch"}
    )
    with (
        urllib.request.urlopen(req, timeout=60) as r,
        open(tmp, "wb") as f,
    ):
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            tail = f"/{total / 1e6:.1f}" if total else ""
            print(f"\r  {got / 1e6:.1f}{tail} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def fetch_one(name: str, dest: Path) -> bool:
    """Ensure ``dest`` holds a valid checkpoint. Returns success."""
    if dest.exists():
        ok, detail = validate(dest)
        if ok:
            print(f"{name}: present, valid — skipping ({detail})")
            return True
        print(f"{name}: present but invalid ({detail}) — refetching")
        dest.unlink()

    tmp_copy = Path("/tmp") / name
    if tmp_copy.exists():
        ok, detail = validate(tmp_copy)
        if ok:
            print(
                f"{name}: valid copy at {tmp_copy} — copying into cache"
            )
            shutil.copyfile(tmp_copy, dest)
            print(f"  ok: {detail}")
            return True
        print(f"{name}: /tmp copy invalid ({detail}) — downloading")

    url = f"{BASE_URL}/{name}"
    print(f"{name}: downloading {url}")
    print(f"  -> {dest}")
    try:
        download(url, dest)
    except Exception as e:
        dest.with_name(dest.name + ".part").unlink(missing_ok=True)
        print(f"  FAILED: {e}")
        return False
    ok, detail = validate(dest)
    if not ok:
        print(f"  FAILED post-download validation: {detail}")
        dest.unlink(missing_ok=True)
        return False
    print(f"  ok: {detail}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--models",
        default="15M,110M",
        help="comma list — choices: " + ",".join(MODELS),
    )
    args = ap.parse_args()

    keys = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in keys if m not in MODELS]
    if bad or not keys:
        ap.error(
            f"--models must be a comma list of {sorted(MODELS)}; "
            f"got {args.models!r}"
        )

    dest_dir = cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"cache dir: {dest_dir}")

    failed = [
        k
        for k in keys
        if not fetch_one(MODELS[k], dest_dir / MODELS[k])
    ]
    if failed:
        sys.exit(f"fetch failed for: {', '.join(failed)}")
    print("done — all requested checkpoints present and valid")


if __name__ == "__main__":
    main()
