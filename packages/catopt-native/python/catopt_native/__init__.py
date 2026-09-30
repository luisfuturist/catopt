"""``catopt_native`` — the optional native (Rust) search engine.

An accelerator, never a default: the pure-Python
``catopt_core.egraph.EGraph`` remains the reference implementation and
the engine the pipeline uses unless ``engine=`` is passed explicitly.
See ``catopt_native.engine.NativeEngine`` — an
:class:`catopt_core.ports.Engine` over the compiled search core.
"""

from catopt_native._native import __version__
from catopt_native.engine import NativeEngine

__all__ = ["NativeEngine", "__version__"]
