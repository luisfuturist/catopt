"""catopt-cuda — the CUDA-graph delivery mechanism (plan 0007).

:class:`CudaGraphRunner` — the :class:`~catopt_core.ports.Runner`
that captures a delivered executor into a CUDA graph, collapsing a
carrier executor's per-level kernel launches into one replayable
graph.  It lives in its own package so the CUDA mechanism — the only
piece of the pipeline that is *device*-coupled, not merely
torch-coupled — is separately installable and separately evolvable.

Delivery runners are composable: ``optimize_model``-family entry
points accept ``runner=CudaGraphRunner()`` or a
``ChainedRunner([TorchCompileRunner(), CudaGraphRunner()])`` — the
compiled-wins precedence lives inside this runner itself.
"""

from catopt_cuda.runners import CudaGraphRunner

__all__ = ["CudaGraphRunner"]
