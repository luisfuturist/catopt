"""Semantic op coverage — the widening ``_CORE_TORCH_BINDINGS`` set.

Every entry in the extended binding table is exercised end to end:
``Op.make`` terms lower through ``eval_term``/``IRModule`` and must
match the aten call they stand for, and ``_shape_of`` must report the
shape the same call produces.  The export-boundary fixes this work
surfaced are pinned too:

* ``aten.index``'s ``layout`` attr — ``x[:, i]`` is NOT ``x[i]``;
* lifted tensor constants (``torch.tensor(0.5)`` inside forward)
  become Params, never silently-substituted Vars;
* string kwargs (``approximate``, ``mode``, ``reduce``) pass through
  to the binding;
* dim-less reductions (``x.sum()``) are FULL reduces — the old
  ``dim=-1`` default mislowered them;
* ``amax``/``amin`` (values only) vs ``max``/``min`` (values+indices
  pair) are distinct ops — a getitem consumer picks the pair apart;
* an UNBOUND op is a boundary, not a failure: the graph around it
  still optimizes, lowering is loud, and ``optimize_compositional``
  falls back per block.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from catopt_core.ir import Const, IR, Op, TensorType, Var
from catopt_torch.torch_bridge import (
    _IR_TO_TORCH,
    export_to_ir,
    ir_to_torch_module,
)
from catopt_core.typing import _INVALID, _shape_of
from catopt_orchestrator import Compositional, Optimizer

from catopt_torch.backend import TorchBackend


def _v(name: str, *shape: int) -> Var:
    return Var(name, TensorType(tuple(shape)))


def _t(*shape: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _ti(*shape: int, seed: int = 1, high: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, high, shape, generator=g)


# ---------------------------------------------------------------------------
#  1. Bindings: each new table entry lowers to the aten call it names
# ---------------------------------------------------------------------------


def _eval_binding(op: str, tensors, attrs):
    return _IR_TO_TORCH[op](*tensors, **attrs)


BINDING_CASES = [
    # (op, tensors, attrs, expected-expression)
    (
        "broadcast_to",
        (_t(2, 1, 4),),
        {"shape": (2, 4, 4)},
        lambda a: torch.broadcast_to(a, (2, 4, 4)),
    ),
    (
        "repeat",
        (_t(2, 4),),
        {"shape": (3, 2)},
        lambda a: a.repeat(3, 2),
    ),
    # Minted expand with a scalar dim attr — t.expand(n).
    (
        "expand",
        (
            _t(
                1,
            ),
        ),
        {"dim": 8},
        lambda a: a.expand(8),
    ),
    (
        "unflatten",
        (_t(2, 4),),
        {"dim": 1, "sizes": (2, 2)},
        lambda a: a.unflatten(1, (2, 2)),
    ),
    (
        "permute",
        (_t(2, 3, 4),),
        {"dim": (2, 0, 1)},
        lambda a: a.permute(2, 0, 1),
    ),
    (
        "movedim",
        (_t(2, 3, 4),),
        {"source": 0, "destination": 2},
        lambda a: torch.movedim(a, 0, 2),
    ),
    (
        "flip",
        (_t(2, 4),),
        {"dim": (-1,)},
        lambda a: torch.flip(a, (-1,)),
    ),
    (
        "roll",
        (_t(2, 4),),
        {"shifts": 1, "dims": -1},
        lambda a: torch.roll(a, 1, -1),
    ),
    (
        "expand_as",
        (_t(4, 1), _t(4, 8)),
        {},
        lambda a, b: a.expand_as(b),
    ),
    (
        "narrow",
        (_t(2, 8),),
        {"dim": 1, "start": 2, "length": 3},
        lambda a: a.narrow(1, 2, 3),
    ),
    (
        "gather",
        (_t(2, 4), _ti(2, 4, high=4)),
        {"dim": 1},
        lambda a, i: a.gather(1, i),
    ),
    (
        "take_along_dim",
        (_t(2, 4), _ti(2, 4, high=4)),
        {"dim": 1},
        lambda a, i: torch.take_along_dim(a, i, dim=1),
    ),
    (
        "scatter",
        (_t(2, 4), _ti(2, 2, high=4), _t(2, 2)),
        {"dim": 1},
        lambda a, i, s: a.scatter(1, i, s),
    ),
    (
        "scatter",
        (_t(2, 4), _ti(2, 2, high=4), torch.tensor(2.5)),
        {"dim": 1},
        lambda a, i, v: a.scatter(1, i, value=v),
    ),
    (
        "scatter_add",
        (_t(2, 4), _ti(2, 2, high=4), _t(2, 2)),
        {"dim": 1},
        lambda a, i, s: a.scatter_add(1, i, s),
    ),
    (
        "scatter_reduce",
        (_t(2, 4), _ti(2, 2, high=4), _t(2, 2)),
        {"dim": 1, "reduce": "prod"},
        lambda a, i, s: a.scatter_reduce(1, i, s, reduce="prod"),
    ),
    (
        "index_add",
        (_t(2, 4), _ti(2, high=4), _t(2, 2)),
        {"dim": 1},
        lambda a, i, s: a.index_add(1, i, s),
    ),
    (
        "index",
        (_t(2, 4, 4), _ti(3, high=2)),
        {"layout": (False, True)},
        lambda a, i: a[:, i],
    ),
    (
        "index_put",
        (_t(2, 4), _ti(2, high=2), _t(2, 4)),
        {"layout": (True,)},
        lambda a, i, v: torch.index_put(a, (i,), v),
    ),
    (
        "index_put",
        (_t(2, 4), _ti(2, high=2), _t(2, 4)),
        {"layout": (True,), "accumulate": True},
        lambda a, i, v: torch.index_put(a, (i,), v, accumulate=True),
    ),
    (
        "slice_scatter",
        (_t(2, 4), _t(2, 2)),
        {"dim": 1, "start": 1, "end": 3, "step": 1},
        lambda a, s: a.slice_scatter(s, dim=1, start=1, end=3, step=1),
    ),
    (
        "select_scatter",
        (_t(2, 4), _t(4)),
        {"dim": 0, "index": 1},
        lambda a, s: a.select_scatter(s, dim=0, index=1),
    ),
    (
        "argmax",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: torch.argmax(a, dim=-1),
    ),
    (
        "argmax",
        (_t(2, 4),),
        {},
        lambda a: torch.argmax(a),
    ),
    (
        "argmin",
        (_t(2, 4),),
        {"dim": 0, "keepdim": True},
        lambda a: torch.argmin(a, dim=0, keepdim=True),
    ),
    (
        "prod",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: a.prod(-1),
    ),
    (
        "prod",
        (_t(2, 4),),
        {},
        lambda a: a.prod(),
    ),
    (
        "var",
        (_t(2, 4),),
        {"dim": (-1,), "correction": 0, "keepdim": True},
        lambda a: a.var(dim=(-1,), correction=0, keepdim=True),
    ),
    (
        "std",
        (_t(2, 4),),
        {"dim": (-1,)},
        lambda a: a.std(dim=(-1,)),
    ),
    (
        "any",
        (_t(2, 4) > 0,),
        {"dim": 0},
        lambda a: a.any(0),
    ),
    (
        "all",
        (_t(2, 4) > 0,),
        {},
        lambda a: a.all(),
    ),
    (
        "nansum",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: torch.nansum(a, -1),
    ),
    (
        "nanmean",
        (_t(2, 4),),
        {},
        lambda a: torch.nanmean(a),
    ),
    (
        "count_nonzero",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: torch.count_nonzero(a, dim=-1),
    ),
    (
        "cumsum",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: torch.cumsum(a, -1),
    ),
    (
        "cumprod",
        (_t(2, 4).abs() + 0.5,),
        {"dim": 0},
        lambda a: torch.cumprod(a, 0),
    ),
    (
        "logcumsumexp",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: torch.logcumsumexp(a, -1),
    ),
    (
        "cummax",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: torch.cummax(a, -1),
    ),
    (
        "cummin",
        (_t(2, 4),),
        {"dim": 0},
        lambda a: torch.cummin(a, 0),
    ),
    (
        "median",
        (_t(2, 4),),
        {},
        lambda a: torch.median(a),
    ),
    (
        "median",
        (_t(2, 4),),
        {"dim": -1, "keepdim": True},
        lambda a: torch.median(a, dim=-1, keepdim=True),
    ),
    (
        "kthvalue",
        (_t(2, 4),),
        {"k": 2, "dim": -1},
        lambda a: torch.kthvalue(a, 2, dim=-1),
    ),
    (
        "argsort",
        (_t(2, 4),),
        {"dim": -1, "descending": True},
        lambda a: torch.argsort(a, dim=-1, descending=True),
    ),
    (
        "topk",
        (_t(2, 4),),
        {"k": 2, "dim": -1},
        lambda a: torch.topk(a, 2, dim=-1),
    ),
    (
        "sort",
        (_t(2, 4),),
        {"dim": 0, "descending": True},
        lambda a: torch.sort(a, dim=0, descending=True),
    ),
    (
        "mode",
        (_t(2, 4).round(),),
        {"dim": -1},
        lambda a: torch.mode(a, dim=-1),
    ),
    (
        "linalg_vector_norm",
        (_t(2, 4),),
        {"ord": 2.0, "dim": (-1,), "keepdim": True},
        lambda a: torch.linalg.vector_norm(
            a, ord=2, dim=(-1,), keepdim=True
        ),
    ),
    (
        "diag_sum",
        (_t(4, 4),),
        {},
        lambda a: torch.trace(a),
    ),
    (
        "clamp",
        (_t(2, 4),),
        {"min": -0.5, "max": 0.5},
        lambda a: torch.clamp(a, -0.5, 0.5),
    ),
    (
        "clamp",
        (_t(2, 4),),
        {"max": 0.5},
        lambda a: torch.clamp(a, max=0.5),
    ),
    (
        "clamp_min",
        (_t(2, 4),),
        {"min": 0.1},
        lambda a: torch.clamp_min(a, 0.1),
    ),
    (
        "clamp_max",
        (_t(2, 4),),
        {"max": 0.9},
        lambda a: torch.clamp_max(a, 0.9),
    ),
    (
        "hardtanh",
        (_t(2, 4),),
        {"min": -0.5, "max": 0.5},
        lambda a: F.hardtanh(a, -0.5, 0.5),
    ),
    (
        "leaky_relu",
        (_t(2, 4), torch.tensor(0.05)),
        {},
        lambda a, s: F.leaky_relu(a, float(s)),
    ),
    (
        "leaky_relu",
        (_t(2, 4),),
        {"negative_slope": 0.2},
        lambda a: F.leaky_relu(a, 0.2),
    ),
    (
        "elu",
        (_t(2, 4),),
        {"alpha": 0.8, "scale": 1.0, "input_scale": 1.0},
        lambda a: F.elu(a, alpha=0.8),
    ),
    (
        "celu",
        (_t(2, 4),),
        {},
        lambda a: F.celu(a),
    ),
    (
        "softplus",
        (_t(2, 4),),
        {"beta": 2.0, "threshold": 10.0},
        lambda a: F.softplus(a, beta=2.0, threshold=10.0),
    ),
    ("softsign", (_t(2, 4),), {}, F.softsign),
    ("hardswish", (_t(2, 4),), {}, F.hardswish),
    ("hardsigmoid", (_t(2, 4),), {}, F.hardsigmoid),
    ("mish", (_t(2, 4),), {}, F.mish),
    ("relu6", (_t(2, 4),), {}, F.relu6),
    (
        "glu",
        (_t(2, 8),),
        {"dim": -1},
        lambda a: F.glu(a, dim=-1),
    ),
    (
        "prelu",
        (_t(2, 4), _t(4)),
        {},
        lambda a, w: F.prelu(a, w),
    ),
    (
        "maximum",
        (_t(2, 4), _t(4)),
        {},
        lambda a, b: torch.maximum(a, b),
    ),
    (
        "minimum",
        (_t(2, 4), _t(4)),
        {},
        lambda a, b: torch.minimum(a, b),
    ),
    ("fmax", (_t(2, 4), _t(4)), {}, torch.fmax),
    ("fmin", (_t(2, 4), _t(4)), {}, torch.fmin),
    ("fmod", (_t(2, 4), _t(4).abs() + 1), {}, torch.fmod),
    ("remainder", (_t(2, 4), _t(4).abs() + 1), {}, torch.remainder),
    (
        "xlogy",
        (_t(2, 4).abs() + 0.5, _t(2, 4).abs() + 1),
        {},
        torch.xlogy,
    ),
    ("atan2", (_t(2, 4), _t(4)), {}, torch.atan2),
    ("heaviside", (_t(2, 4), _t(4)), {}, torch.heaviside),
    ("isclose", (_t(2, 4), _t(2, 4)), {}, torch.isclose),
    (
        "lerp",
        (_t(2, 4), _t(2, 4), torch.tensor(0.5)),
        {},
        lambda a, b, w: torch.lerp(a, b, w),
    ),
    (
        "addcmul",
        (_t(2, 4), _t(2, 4), _t(2, 4)),
        {"value": 2.0},
        lambda a, b, c: torch.addcmul(a, b, c, value=2.0),
    ),
    (
        "addcdiv",
        (_t(2, 4), _t(2, 4), _t(2, 4).abs() + 1),
        {},
        lambda a, b, c: torch.addcdiv(a, b, c),
    ),
    (
        "logical_and",
        (_t(2, 4) > 0, _t(2, 4) > 0),
        {},
        lambda a, b: torch.logical_and(a, b),
    ),
    (
        "logical_or",
        (_t(2, 4) > 0, _t(2, 4) > 0),
        {},
        lambda a, b: torch.logical_or(a, b),
    ),
    (
        "logical_xor",
        (_t(2, 4) > 0, _t(2, 4) > 0),
        {},
        lambda a, b: torch.logical_xor(a, b),
    ),
    (
        "bitwise_and",
        (_ti(2, 4, high=8), _ti(2, 4, high=8)),
        {},
        lambda a, b: torch.bitwise_and(a, b),
    ),
    (
        "bitwise_or",
        (_ti(2, 4, high=8), _ti(2, 4, high=8)),
        {},
        lambda a, b: torch.bitwise_or(a, b),
    ),
    (
        "log_softmax",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: F.log_softmax(a, dim=-1),
    ),
    ("abs", (_t(2, 4),), {}, torch.abs),
    ("sign", (_t(2, 4),), {}, torch.sign),
    ("floor", (_t(2, 4),), {}, torch.floor),
    ("ceil", (_t(2, 4),), {}, torch.ceil),
    ("round", (_t(2, 4),), {}, torch.round),
    ("frac", (_t(2, 4),), {}, torch.frac),
    ("trunc", (_t(2, 4),), {}, torch.trunc),
    ("log", (_t(2, 4).abs() + 0.5,), {}, torch.log),
    ("log2", (_t(2, 4).abs() + 0.5,), {}, torch.log2),
    ("log10", (_t(2, 4).abs() + 0.5,), {}, torch.log10),
    ("log1p", (_t(2, 4).abs(),), {}, torch.log1p),
    ("exp2", (_t(2, 4),), {}, torch.exp2),
    ("expm1", (_t(2, 4),), {}, torch.expm1),
    ("erf", (_t(2, 4),), {}, torch.erf),
    ("erfc", (_t(2, 4),), {}, torch.erfc),
    ("erfinv", (_t(2, 4) * 0.5,), {}, torch.erfinv),
    ("gammaln", (_t(2, 4).abs() + 0.5,), {}, torch.special.gammaln),
    ("digamma", (_t(2, 4).abs() + 0.5,), {}, torch.digamma),
    ("i0", (_t(2, 4),), {}, torch.i0),
    ("sinc", (_t(2, 4),), {}, torch.sinc),
    ("reciprocal", (_t(2, 4).abs() + 0.5,), {}, torch.reciprocal),
    (
        "nan_to_num",
        (torch.tensor([[float("nan"), float("inf")]]),),
        {"nan": 0.0},
        lambda a: torch.nan_to_num(a),
    ),
    ("isnan", (_t(2, 4),), {}, torch.isnan),
    ("isinf", (_t(2, 4),), {}, torch.isinf),
    ("isfinite", (_t(2, 4),), {}, torch.isfinite),
    ("isposinf", (_t(2, 4),), {}, torch.isposinf),
    ("isneginf", (_t(2, 4),), {}, torch.isneginf),
    ("asin", (_t(2, 4) * 0.4,), {}, torch.asin),
    ("acos", (_t(2, 4) * 0.4,), {}, torch.acos),
    ("atan", (_t(2, 4),), {}, torch.atan),
    ("sinh", (_t(2, 4),), {}, torch.sinh),
    ("cosh", (_t(2, 4),), {}, torch.cosh),
    ("asinh", (_t(2, 4),), {}, torch.asinh),
    ("acosh", (_t(2, 4).abs() + 1.5,), {}, torch.acosh),
    ("atanh", (_t(2, 4) * 0.4,), {}, torch.atanh),
    (
        "triu",
        (_t(4, 4),),
        {"diagonal": 1},
        lambda a: torch.triu(a, diagonal=1),
    ),
    (
        "tril",
        (_t(4, 4),),
        {},
        lambda a: torch.tril(a),
    ),
    (
        "pad",
        (_t(2, 4),),
        {"pad": (1, 2, 0, 1)},
        lambda a: F.pad(a, (1, 2, 0, 1)),
    ),
    (
        "pad",
        (_t(2, 4),),
        {"pad": (1, 1), "mode": "replicate"},
        lambda a: F.pad(a, (1, 1), mode="replicate"),
    ),
    (
        "one_hot",
        (_ti(2, 4, high=5),),
        {"num_classes": 5},
        lambda a: F.one_hot(a, num_classes=5),
    ),
    (
        "batch_norm",
        (_t(2, 4, 4), _t(4), _t(4), _t(4), _t(4)),
        {"training": False, "momentum": 0.1, "eps": 1e-5},
        lambda x, w, b, rm, rv: F.batch_norm(
            x, rm, rv, w, b, False, 0.1, 1e-5
        ),
    ),
    (
        "batch_norm",
        (_t(2, 4, 4), _t(4), _t(4)),
        {"training": False, "eps": 1e-5},
        lambda x, rm, rv: F.batch_norm(x, rm, rv, training=False),
    ),
    (
        "group_norm",
        (_t(2, 8, 4), _t(8), _t(8)),
        {"num_groups": 2, "eps": 1e-5},
        lambda x, w, b: F.group_norm(x, 2, w, b, eps=1e-5),
    ),
    (
        "group_norm",
        (_t(2, 8, 4),),
        {"num_groups": 4},
        lambda x: F.group_norm(x, 4),
    ),
    # instance_norm's operand layout: the affine pair leads, the
    # running-stats pair tails iff use_input_stats is False.
    (
        "instance_norm",
        (_t(2, 4, 4), _t(4), _t(4)),
        {"use_input_stats": True, "eps": 1e-4},
        lambda x, w, b: F.instance_norm(
            x, weight=w, bias=b, use_input_stats=True, eps=1e-4
        ),
    ),
    (
        "instance_norm",
        (_t(2, 4, 4),),
        {"use_input_stats": True},
        lambda x: F.instance_norm(x, use_input_stats=True),
    ),
    # A lone mid operand reads as weight (batch_norm's convention).
    (
        "instance_norm",
        (_t(2, 4, 4), _t(4)),
        {"use_input_stats": True},
        lambda x, w: F.instance_norm(x, weight=w, use_input_stats=True),
    ),
    # use_input_stats=False: the tail pair is (running_mean,
    # running_var) — the eval-mode export of track_running_stats.
    (
        "instance_norm",
        (_t(2, 4, 4), _t(4).abs() + 0.5, _t(4).abs() + 0.5),
        {"use_input_stats": False},
        lambda x, rm, rv: F.instance_norm(
            x, rm, rv, use_input_stats=False
        ),
    ),
    (
        "instance_norm",
        (_t(2, 4, 4), _t(4), _t(4), _t(4).abs() + 0.5, _t(4).abs() + 0.5),
        {"use_input_stats": False, "momentum": 0.2},
        lambda x, w, b, rm, rv: F.instance_norm(
            x, rm, rv, w, b, False, 0.2, 1e-5
        ),
    ),
    # upsample_nearest2d — the canonical size/scale attr spellings.
    (
        "upsample_nearest2d",
        (_t(2, 4, 8, 8),),
        {"size": (16, 16)},
        lambda x: F.interpolate(x, size=(16, 16), mode="nearest"),
    ),
    (
        "upsample_nearest2d",
        (_t(2, 4, 8, 8),),
        {"scale": (2.0, 3.0)},
        lambda x: F.interpolate(
            x, scale_factor=(2.0, 3.0), mode="nearest"
        ),
    ),
    # Scalar spellings broadcast to the (H, W) pair.
    (
        "upsample_nearest2d",
        (_t(2, 4, 8, 8),),
        {"size": 16},
        lambda x: F.interpolate(x, size=(16, 16), mode="nearest"),
    ),
    (
        "upsample_nearest2d",
        (_t(2, 4, 8, 8),),
        {"scale": 2.0},
        lambda x: F.interpolate(x, scale_factor=(2.0, 2.0), mode="nearest"),
    ),
    # lstm.input — the (x, h0, c0, *params) operand tail, returning
    # the (output, h_n, c_n) triple.
    (
        "lstm.input",
        (_t(2, 5, 8), _t(1, 2, 6), _t(1, 2, 6),
         _t(24, 8), _t(24, 6), _t(24), _t(24)),
        {
            "has_biases": True,
            "num_layers": 1,
            "dropout": 0.0,
            "train": False,
            "bidirectional": False,
            "batch_first": True,
        },
        lambda x, h0, c0, w1, w2, b1, b2: torch.ops.aten.lstm.input(
            x, [h0, c0], [w1, w2, b1, b2], True, 1, 0.0, False,
            False, True
        ),
    ),
    (
        "conv1d",
        (_t(1, 3, 8), _t(4, 3, 3), _t(4)),
        {"stride": 2, "padding": 1},
        lambda x, w, b: F.conv1d(x, w, b, stride=2, padding=1),
    ),
    (
        "searchsorted",
        (torch.linspace(0, 1, 8), _t(4)),
        {},
        lambda s, v: torch.searchsorted(s, v),
    ),
    ("nonzero", (_ti(2, 4, high=2),), {}, torch.nonzero),
    ("outer", (_t(3), _t(4)), {}, torch.outer),
    (
        "einsum",
        (_t(2, 4, 3), _t(2, 3, 4)),
        {"equation": "bij,bjk->bik"},
        lambda a, b: torch.einsum("bij,bjk->bik", a, b),
    ),
    (
        "tensor_split",
        (_t(2, 8),),
        {"sections": 4, "dim": -1, "index": 2},
        lambda a: torch.tensor_split(a, 4, dim=-1)[2],
    ),
    (
        "unfold",
        (_t(2, 8),),
        {"dim": 1, "size": 3, "step": 2},
        lambda a: a.unfold(1, 3, 2),
    ),
    (
        "pixel_shuffle",
        (_t(1, 8, 2, 2),),
        {"upscale_factor": 2},
        lambda a: F.pixel_shuffle(a, 2),
    ),
    (
        "pixel_unshuffle",
        (_t(1, 2, 4, 4),),
        {"downscale_factor": 2},
        lambda a: F.pixel_unshuffle(a, 2),
    ),
    (
        "hstack",
        (_t(2, 2), _t(2, 3)),
        {},
        lambda a, b: torch.hstack([a, b]),
    ),
    (
        "vstack",
        (_t(2, 4), _t(3, 4)),
        {},
        lambda a, b: torch.vstack([a, b]),
    ),
    ("arange", (), {"arg0": 5}, lambda: torch.arange(5)),
    (
        "arange",
        (),
        {"arg0": 1, "arg1": 6, "arg2": 2},
        lambda: torch.arange(1, 6, 2),
    ),
    (
        "zeros",
        (),
        {"shape": (2, 3)},
        lambda: torch.zeros(2, 3),
    ),
    ("ones", (), {"shape": (2, 3)}, lambda: torch.ones(2, 3)),
    ("empty", (), {"shape": (4,)}, lambda s=None: None),  # shape only
    (
        "full",
        (torch.tensor(2.5),),
        {"shape": (2, 2)},
        lambda v: torch.full((2, 2), v.item()),
    ),
    ("zeros_like", (_t(2, 4),), {}, torch.zeros_like),
    ("ones_like", (_t(2, 4),), {}, torch.ones_like),
    (
        "full_like",
        (_t(2, 4), torch.tensor(3.0)),
        {},
        lambda a, v: torch.full_like(a, 3.0),
    ),
    (
        "new_zeros",
        (
            _t(
                4,
            ),
        ),
        {"shape": (2, 3)},
        lambda a: a.new_zeros((2, 3)),
    ),
    (
        "new_ones",
        (
            _t(
                4,
            ),
        ),
        {"shape": (2, 3)},
        lambda a: a.new_ones((2, 3)),
    ),
    (
        "new_full",
        (
            _t(
                4,
            ),
            torch.tensor(7.0),
        ),
        {"shape": (2, 3)},
        lambda a, v: a.new_full((2, 3), 7.0),
    ),
    ("int", (_t(2, 4),), {}, lambda a: a.int()),
    ("long", (_t(2, 4),), {}, lambda a: a.long()),
    ("double", (_t(2, 4),), {}, lambda a: a.double()),
    ("half", (_t(2, 4),), {}, lambda a: a.half()),
    ("bfloat16", (_t(2, 4),), {}, lambda a: a.bfloat16()),
    ("bool", (_t(2, 4),), {}, lambda a: a.bool()),
    ("byte", (_t(2, 4),), {}, lambda a: a.byte()),
    ("char", (_t(2, 4),), {}, lambda a: a.char()),
    ("short", (_t(2, 4),), {}, lambda a: a.short()),
    ("detach", (_t(2, 4),), {}, lambda a: a.detach()),
    ("detach_", (_t(2, 4),), {}, lambda a: a.detach()),
    ("_assert_tensor_metadata", (_t(2, 4),), {}, lambda a: a),
    (
        "copy",
        (_t(2, 4), _t(1)),
        {},
        lambda a, s: torch.broadcast_to(s, a.shape).clone(),
    ),
    ("item", (_t(1),), {}, lambda a: a.item()),
    ("numel", (_t(2, 4),), {}, lambda a: a.numel()),
    # Namedtuple pairs: binding returns the pair; getitem picks.
    (
        "max",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: torch.max(a, dim=-1),
    ),
    ("max", (_t(2, 4),), {}, lambda a: torch.max(a)),
    (
        "min",
        (_t(2, 4),),
        {"dim": 0, "keepdim": True},
        lambda a: torch.min(a, dim=0, keepdim=True),
    ),
    (
        "amax",
        (_t(2, 4),),
        {"dim": -1},
        lambda a: a.amax(-1),
    ),
    ("amin", (_t(2, 4),), {}, lambda a: a.amin()),
    (
        "var_mean",
        (_t(2, 4),),
        {"dim": (-1,), "keepdim": True},
        lambda a: torch.var_mean(a, dim=(-1,), keepdim=True),
    ),
    (
        "std_mean",
        (_t(2, 4),),
        {},
        lambda a: torch.std_mean(a),
    ),
    (
        "squeeze",
        (_t(2, 1, 4, 1),),
        {},
        lambda a: a.squeeze(),
    ),
    (
        "inv",
        (torch.eye(4) + _t(4, 4) * 0.05,),
        {},
        torch.linalg.inv,
    ),
]


@pytest.mark.parametrize(
    "op,tensors,attrs,expected",
    BINDING_CASES,
    ids=[c[0] + "/" + str(i) for i, c in enumerate(BINDING_CASES)],
)
def test_binding_lowers_to_aten_semantics(op, tensors, attrs, expected):
    out = _eval_binding(op, tensors, attrs)
    want = expected(*tensors)
    if op == "empty":
        assert out.shape == (4,)
        return
    if op == "item":
        assert out == want or abs(out - want) < 1e-6
        return
    if op in ("arange",):
        assert torch.equal(out, want)
        return
    if isinstance(out, tuple):
        for o, w in zip(out, want, strict=True):
            torch.testing.assert_close(o, w, equal_nan=True)
        return
    torch.testing.assert_close(out, want, equal_nan=True)


# ---------------------------------------------------------------------------
#  2. Shape inference — every new arm, and the ()-policy edge cases
# ---------------------------------------------------------------------------


def _term(op, shapes, **attrs):
    return Op.make(
        op, *[_v(f"s{i}", *s) for i, s in enumerate(shapes)], **attrs
    )


SHAPE_CASES = [
    # broadcast families
    (("maximum", [(2, 4), (4,)], {}), (2, 4)),
    (("lerp", [(2, 4), (2, 4), ()], {}), (2, 4)),
    # Broadcast conflict inside the ternary chain -> _INVALID.
    (("lerp", [(2, 4), (5,), ()], {}), _INVALID),
    (("addcmul", [(2, 4), (4,), ()], {}), (2, 4)),
    (("logical_and", [(2, 4), (4,)], {}), (2, 4)),
    # same-shape unaries + axis-keep
    (("abs", [(2, 4)], {}), (2, 4)),
    (("cumsum", [(2, 4)], {"dim": -1}), (2, 4)),
    (("cumsum", [()], {"dim": -1}), None),
    (("scatter", [(2, 4), (2, 2), (2, 2)], {"dim": 1}), (2, 4)),
    (("index_put", [(2, 4), (2,), (2, 4)], {}), (2, 4)),
    (("sort", [(2, 4)], {"dim": -1}), (2, 4)),
    (("topk", [(2, 4)], {"k": 2, "dim": -1}), (2, 4)),
    (("batch_norm", [(2, 4, 4)], {}), (2, 4, 4)),
    (("instance_norm", [(2, 4, 4)], {"use_input_stats": True}), (2, 4, 4)),
    (("_assert_tensor_metadata", [(2, 4)], {}), (2, 4)),
    # upsample_nearest2d — the size attr spells the output extents;
    # scale multiplies them (aten floors input*scale); neither known
    # -> (N,C,?,?) extents unknown.
    (
        ("upsample_nearest2d", [(2, 4, 8, 8)], {"size": (16, 16)}),
        (2, 4, 16, 16),
    ),
    (
        ("upsample_nearest2d", [(2, 4, 8, 8)], {"scale": (2.0, 3.0)}),
        (2, 4, 16, 24),
    ),
    (
        ("upsample_nearest2d", [(2, 4, 8, 8)], {"scale": 2.0}),
        (2, 4, 16, 16),
    ),
    (("upsample_nearest2d", [(2, 4, 8, 8)], {}), (2, 4, None, None)),
    # A lone-element list cannot spell the (H, W) pair — unknown.
    (
        ("upsample_nearest2d", [(2, 4, 8, 8)], {"scale": (1.5,)}),
        (2, 4, None, None),
    ),
    # A bool is not a scale (isinstance(True, int) would read it 1).
    (
        ("upsample_nearest2d", [(2, 4, 8, 8)], {"scale": True}),
        (2, 4, None, None),
    ),
    # Unknown input extent -> the scaled extent stays unknown.
    (
        ("upsample_nearest2d", [(2, 4, None, 8)], {"scale": (2.0, 2.0)}),
        (2, 4, None, 16),
    ),
    # Sub-rank-4 input: report the input shape.
    (("upsample_nearest2d", [(2, 4, 8)], {"size": (16, 16)}), (2, 4, 8)),
    # lstm.input — a heterogeneous (output, h_n, c_n) triple: honest
    # unknown rather than x's shape.
    (
        (
            "lstm.input",
            [(2, 5, 8), (1, 2, 6), (1, 2, 6), (24, 8), (24, 6), (24,), (24,)],
            {"num_layers": 1},
        ),
        None,
    ),
    # reductions
    (("prod", [(2, 4, 3)], {"dim": 1, "keepdim": True}), (2, 1, 3)),
    (("prod", [(2, 4)], {}), ()),
    (("argmax", [(2, 4)], {"dim": -1}), (2,)),
    (("argmin", [(2, 4)], {}), ()),
    (("var", [(2, 4)], {"dim": (-1,), "keepdim": True}), (2, 1)),
    (("std", [(2, 4)], {}), ()),
    (("any", [(2, 4)], {"dim": 0}), (4,)),
    (("all", [(2, 4)], {}), ()),
    (("max", [(2, 4)], {"dim": -1}), (2,)),
    (("max", [(2, 4)], {}), ()),
    (("amax", [(2, 4)], {"dim": 0}), (4,)),
    (("var_mean", [(2, 4)], {"dim": (-1,)}), (2,)),
    (("count_nonzero", [(2, 4)], {"dim": 1}), (2,)),
    (
        (
            "linalg_vector_norm",
            [(2, 4)],
            {"dim": (-1,), "keepdim": True},
        ),
        (2, 1),
    ),
    (("diag_sum", [(4, 4)], {}), ()),
    (("item", [(1,)], {}), ()),
    (("numel", [(2, 4)], {}), ()),
    # shape ops
    (("broadcast_to", [(2, 1, 4)], {"shape": (2, 4, 4)}), (2, 4, 4)),
    (("repeat", [(2, 4)], {"shape": (3, 2)}), (6, 8)),
    (("repeat", [(2, 4)], {"shape": (2, 2, 3)}), (2, 4, 12)),
    (("repeat", [(2, 4)], {}), (2, 4)),
    # Malformed: len(reps) < ndim is a torch error — report base.
    (("repeat", [(2, 4)], {"shape": (2,)}), (2, 4)),
    (("permute", [(2, 3, 4)], {"dim": (2, 0, 1)}), (4, 2, 3)),
    (("permute", [(2, 3, 4)], {}), (4, 3, 2)),
    (("permute", [()], {}), None),
    (
        ("movedim", [(2, 3, 4)], {"source": 0, "destination": 2}),
        (3, 4, 2),
    ),
    (("movedim", [()], {"source": 0, "destination": 1}), None),
    (
        (
            "movedim",
            [(2, 3, 4)],
            {"source": (0, 1), "destination": (1, 2)},
        ),
        (4, 2, 3),
    ),
    (("movedim", [(2, 3, 4)], {}), (2, 3, 4)),
    (("expand_as", [(4, 1), (4, 8)], {}), (4, 8)),
    (("expand_as", [(4, 1)], {}), (4, 1)),
    (("narrow", [(2, 8)], {"dim": 1, "start": 0, "length": 3}), (2, 3)),
    (("narrow", [()], {"dim": 0, "start": 0, "length": 1}), None),
    (("unflatten", [(2, 8)], {"dim": 1, "sizes": (2, 4)}), (2, 2, 4)),
    (("unflatten", [(2, 8)], {"dim": 1, "sizes": (2, -1)}), (2, 2, 4)),
    (("unflatten", [(2, 8)], {}), (2, 8)),
    (("unflatten", [()], {"dim": 0, "sizes": (2,)}), None),
    (("flip", [(2, 4)], {"dim": (-1,)}), (2, 4)),
    (("roll", [(2, 4)], {"shifts": 1, "dims": -1}), (2, 4)),
    (("triu", [(4, 4)], {}), (4, 4)),
    (("pad", [(2, 4)], {"pad": (1, 2, 0, 1)}), (3, 7)),
    (("pad", [(2, 4)], {}), (2, 4)),
    # pad list longer than 2*ndim: the loop runs out of axes.
    (("pad", [(4,)], {"pad": (1, 1, 1, 1)}), (6,)),
    (("pad", [()], {"pad": (1, 1)}), None),
    # Unknown extents stay unknown rather than fabricating ints.
    (("pad", [(None, 4)], {"pad": (0, 0, 1, 1)}), (None, 4)),
    (("glu", [(2, 8)], {"dim": -1}), (2, 4)),
    (("glu", [()], {"dim": -1}), None),
    (
        ("pixel_shuffle", [(1, 8, 2, 2)], {"upscale_factor": 2}),
        (1, 2, 4, 4),
    ),
    (
        ("pixel_unshuffle", [(1, 2, 4, 4)], {"downscale_factor": 2}),
        (1, 8, 2, 2),
    ),
    (("pixel_shuffle", [(2, 4)], {"upscale_factor": 2}), (2, 4)),
    (
        ("pixel_unshuffle", [(2, 4)], {"downscale_factor": "x"}),
        (2, 4),
    ),
    (("unfold", [(2, 8)], {"dim": 1, "size": 3, "step": 2}), (2, 3, 3)),
    (("unfold", [()], {"dim": 0, "size": 2, "step": 1}), None),
    (
        ("unfold", [(2, 8)], {"dim": 1, "size": "x", "step": 1}),
        (2, None, None),
    ),
    (("hstack", [(2, 2), (2, 3)], {}), (2, 5)),
    (("hstack", [(4,), (3,)], {}), (7,)),
    (("vstack", [(2, 4), (3, 4)], {}), (5, 4)),
    (("vstack", [(4,), (4,)], {}), (2, 4)),
    (("vstack", [(2, 4), (3,)], {}), (2, 4)),
    (("hstack", [()], {}), None),
    # indexing
    (("gather", [(2, 4), (2, 2)], {"dim": 1}), (2, 2)),
    (("gather", [(2, 4)], {}), (2, 4)),
    (("take_along_dim", [(2, 4), (2, 3)], {"dim": 1}), (2, 3)),
    (("searchsorted", [(8,), (4,)], {}), (4,)),
    (("nonzero", [(2, 4)], {}), (None, 2)),
    (("one_hot", [(2, 4)], {"num_classes": 5}), (2, 4, 5)),
    (("one_hot", [(2, 4)], {}), (2, 4, None)),
    (("outer", [(3,), (4,)], {}), (3, 4)),
    (("einsum", [(2, 4), (4, 4)], {"equation": "ij,jk->ik"}), None),
    # addmm family
    (("addmm", [(4,), (4, 8), (8, 4)], {}), (4, 4)),
    (("addmv", [(4,), (4, 8), (8,)], {}), (4,)),
    (("baddbmm", [(2, 1, 4), (2, 4, 8), (2, 8, 4)], {}), (2, 4, 4)),
    (("addmm", [(4,), (4,), ()], {}), (4,)),
    # conv1d
    (
        ("conv1d", [(1, 3, 8), (4, 3, 3)], {"stride": 2, "padding": 1}),
        (1, 4, 4),
    ),
    (
        ("conv1d", [(1, 3, 8), (4, 3, 3)], {"padding": "same"}),
        (1, 4, None),
    ),
    (("conv1d", [(1, 3), (4, 3)], {}), (1, 4, None)),
    (("conv1d", [(), (4, 3, 3)], {}), None),
    # unknown input extent -> ol is None
    (("conv1d", [(1, 3, None), (4, 3, 3)], {}), (1, 4, None)),
    # tensor_split
    (
        (
            "tensor_split",
            [(2, 8)],
            {"sections": 4, "dim": -1, "index": 1},
        ),
        (2, 2),
    ),
    (
        (
            "tensor_split",
            [(2, 8)],
            {"sections": (3, 5), "dim": -1, "index": 1},
        ),
        (2, 2),
    ),
    (
        ("tensor_split", [(2, 8)], {"sections": "x", "dim": -1}),
        (2, None),
    ),
    (("tensor_split", [()], {"sections": 2}), None),
    # index — advanced indexing layouts
    (
        ("index", [(2, 4, 4), (3,)], {"layout": (False, True)}),
        (2, 3, 4),
    ),
    (
        ("index", [(2, 4, 4), (3,)], {"layout": (True, False)}),
        (3, 4, 4),
    ),
    (
        (
            "index",
            [(2, 4, 4), (3,), (3,)],
            {"layout": (True, False, True)},
        ),
        (3, 4),
    ),
    (("index", [(2, 4, 4), (3,)], {}), (3, 4, 4)),
    (("index", [(2, 4, 4), (5,), (2,)], {}), _INVALID),
    (("index", [()], {}), None),
    # squeeze no-dim arm
    (("squeeze", [(2, 1, 4, 1)], {}), (2, 4)),
    (("squeeze", [(2, 1, 4)], {"dim": 1}), (2, 4)),
    # creators
    (("zeros", [], {"shape": (2, 3)}), (2, 3)),
    (("full", [], {"shape": (4,)}), (4,)),
    (("full", [], {}), None),
    (("new_zeros", [(2,)], {"shape": (2, 2)}), (2, 2)),
    (("arange", [], {"arg0": 4}), (4,)),
    (("arange", [], {"arg0": 2, "arg1": 6}), (4,)),
    (("arange", [], {"arg0": 1, "arg1": 7, "arg2": 2}), (3,)),
    (("arange", [], {"arg0": 0.0, "arg1": 0.5, "arg2": 0.25}), (2,)),
    (("arange", [], {}), (None,)),
]


@pytest.mark.parametrize(
    "spec,expected",
    SHAPE_CASES,
    ids=[f"{s[0]}/{i}" for i, (s, _) in enumerate(SHAPE_CASES)],
)
def test_shape_inference(spec, expected):
    op, shapes, attrs = spec
    term = _term(op, shapes, **attrs)
    got = _shape_of(term)
    if expected is _INVALID:
        assert got is _INVALID
    elif expected is None:
        assert got is None
    else:
        assert got == tuple(expected)


def test_arange_shape_const_operands():
    t = Op.make("arange", Const(0.0), Const(6.0), Const(2.0))
    assert _shape_of(t) == (3,)


def test_arange_binding_const_operands():
    # Minted Const-operand spelling also lowers.
    out = _IR_TO_TORCH["arange"](torch.tensor(0.0), torch.tensor(4.0))
    assert torch.equal(out, torch.arange(4))


def test_scaled_extent_bool_guard():
    """``isinstance(True, int)`` is True — a bool dim/scale is not a
    concrete extent, and ``_scaled_extent`` declines it."""
    from catopt_core.typing import _scaled_extent

    assert _scaled_extent(8, True) is None
    assert _scaled_extent(True, 2.0) is None


def test_pow_shape_broadcasts_exponent():
    """``pow``'s result broadcasts the exponent too — not just the
    base.  ``pow((), (4,))`` evaluates to ``(4,)``; the shape rule
    must agree."""
    x = Var("x", TensorType(()))
    y = Var("y", TensorType((4,)))
    assert _shape_of(Op.make("pow", x, y)) == (4,)
    z = Var("z", TensorType((2, 4)))
    assert _shape_of(Op.make("pow", y, z)) == (2, 4)


# ---------------------------------------------------------------------------
#  3. Export probes — the boundary fixes end to end
# ---------------------------------------------------------------------------


def _export_and_run(model, *args):
    model = model.eval()
    ir, tensors = export_to_ir(model, args)
    mod = ir_to_torch_module(ir, param_values=tensors)
    with torch.no_grad():
        ref = model(*args)
        out = mod(*args)
    return ir, ref, out


def _iter_ops(term):
    seen = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op) and id(t) not in seen:
            seen.add(id(t))
            yield t
            stack.extend(t.args)


def test_export_advanced_index_layout():
    """``x[:, i]`` must not collapse into ``x[i]`` — the layout attr
    preserves the leading slice position."""

    class M(torch.nn.Module):
        def forward(self, x, i):
            return x[:, i]

    x, i = _t(2, 4, 4), torch.arange(2)
    ir, ref, out = _export_and_run(M(), x, i)
    torch.testing.assert_close(out, ref)
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "index"]
    assert t.attrs["layout"] == (False, True)


def test_export_index_leading():
    class M(torch.nn.Module):
        def forward(self, x, i):
            return x[i]

    x, i = _t(2, 4, 4), torch.arange(2)
    _, ref, out = _export_and_run(M(), x, i)
    torch.testing.assert_close(out, ref)


def test_export_index_separated():
    class M(torch.nn.Module):
        def forward(self, x, i, j):
            return x[i, :, j]

    x, i, j = _t(3, 4, 4), torch.arange(2), torch.arange(2)
    ir, ref, out = _export_and_run(M(), x, i, j)
    torch.testing.assert_close(out, ref)
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "index"]
    assert t.attrs["layout"] == (True, False, True)


def test_export_index_put():
    class M(torch.nn.Module):
        def forward(self, x, v):
            y = x.clone()
            y[torch.arange(2)] = v
            return y

    _, ref, out = _export_and_run(M(), _t(3, 4), _t(2, 4))
    torch.testing.assert_close(out, ref)


def test_export_slice_scatter():
    class M(torch.nn.Module):
        def forward(self, x):
            y = torch.zeros(2, 4, 4)
            y[:, 1:3] = x[:, :2]
            return y + x

    _, ref, out = _export_and_run(M(), _t(2, 4, 4))
    torch.testing.assert_close(out, ref)


def test_export_select_scatter_copy_threading():
    """``x[i] = v`` on an integer index is copy_ through a select
    view — the select_scatter threading covers it."""

    class M(torch.nn.Module):
        def forward(self, x):
            y = x.clone()
            y[1] = x[0]
            return y

    _, ref, out = _export_and_run(M(), _t(3, 4))
    torch.testing.assert_close(out, ref)


def test_export_whole_tensor_copy():
    """``param.copy_(v)`` writes the whole tensor — the ``copy`` op
    spelling (not a view-scatter)."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.buf = torch.nn.Parameter(_t(3, 4))

        def forward(self, x):
            self.buf.copy_(x[0])
            return self.buf + x

    _, ref, out = _export_and_run(M(), _t(3, 4))
    torch.testing.assert_close(out, ref)


def test_export_non_persistent_buffer():
    """A non-persistent buffer is absent from the state dict — its
    tensor resolves through the module attribute path."""

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer(
                "mask", torch.full((4,), 2.0), persistent=False
            )

        def forward(self, x):
            return x * self.mask

    _, ref, out = _export_and_run(M(), _t(2, 4))
    torch.testing.assert_close(out, ref)


def test_export_isclose_named_positionals():
    """isclose's (rtol, atol, equal_nan) tail is schema'd — equal_nan
    must reach the binding or nan-compare silently flips."""

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.isclose(x, x + 1e-6, equal_nan=True)

    ir, ref, out = _export_and_run(M(), _t(2, 4))
    torch.testing.assert_close(out, ref)
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "isclose"]
    assert t.attrs["equal_nan"] is True


def test_export_unschemed_bool_positional_argN():
    """A bool positional on an unschemed op lands as ``argN`` — the
    multinomial ``replacement`` flag."""

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.multinomial(
                x.softmax(-1), 2, replacement=True
            ).float()

    ir, _ = export_to_ir(M().eval(), (_t(4, 8),))
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "multinomial"]
    assert t.attrs["arg2"] is True


def test_export_unschemed_scalar_positional_argN():
    """An op with no schema keeps the legacy ``argN`` spelling for a
    scalar positional — repeat_interleave's repeats count."""

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.repeat_interleave(x, 2, dim=1)

    ir, _ = export_to_ir(M().eval(), (_t(2, 4),))
    (t,) = [
        o
        for o in _iter_ops(ir.root)
        if o.op.startswith("repeat_interleave")
    ]
    assert t.attrs["arg1"] == 2


def test_export_gelu_tanh_approximate():
    """The ``approximate`` string kwarg reaches the binding — a
    tanh-approx gelu must not silently become exact gelu."""

    class M(torch.nn.Module):
        def forward(self, x):
            return F.gelu(x, approximate="tanh")

    ir, ref, out = _export_and_run(M(), _t(4, 8))
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "gelu"]
    assert t.attrs["approximate"] == "tanh"


def test_export_div_rounding_mode():
    """``rounding_mode`` is a semantic kwarg — a floor div must not
    lower to plain trunc ``x/y``."""

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.div(
                x, torch.ones_like(x), rounding_mode="floor"
            )

    _, ref, out = _export_and_run(
        M(), torch.tensor([-2.5, 3.5, -1.0, 4.0])
    )
    torch.testing.assert_close(out, ref)


def test_export_round_decimals():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.round(x, decimals=1)

    _, ref, out = _export_and_run(M(), _t(2, 4))
    torch.testing.assert_close(out, ref)


def test_export_pad_mode():
    class M(torch.nn.Module):
        def forward(self, x):
            return F.pad(x, (1, 1), mode="replicate")

    _, ref, out = _export_and_run(M(), _t(2, 4))
    torch.testing.assert_close(out, ref)


def test_export_lifted_tensor_constant():
    """``torch.tensor(0.5)`` inside forward lifts to a Param — not a
    Var that the eval "self" fallback would silently substitute."""

    class M(torch.nn.Module):
        def forward(self, x):
            return torch.maximum(x, torch.tensor(0.5))

    x = _t(2, 4)
    ir, ref, out = _export_and_run(M(), x)
    torch.testing.assert_close(out, ref)
    assert len(ir.inputs) == 1
    assert any(p.typ.shape == () for p in ir.params.values())


def test_export_dimless_sum_is_full_reduce():
    """x.sum() exports aten.sum (no dims) — it is a full reduce, not
    x.sum(-1)."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x.sum() + x.mean() + x.amax()

    _, ref, out = _export_and_run(M(), _t(2, 4))
    torch.testing.assert_close(out, ref)


def test_export_dimless_squeeze_removes_all_ones():
    class M(torch.nn.Module):
        def forward(self, x):
            return x.squeeze()

    _, ref, out = _export_and_run(M(), _t(2, 1, 4, 1))
    torch.testing.assert_close(out, ref)


def test_export_max_min_pairs():
    """x.max(-1) returns a (values, indices) pair — getitem picks the
    pair apart correctly (the old amax collapse returned only
    values)."""

    class M(torch.nn.Module):
        def forward(self, x):
            v, i = x.max(-1)
            lv, li = x.min(-1)
            return (
                v.sum() + i.float().sum() + lv.sum() + li.float().sum()
            )

    _, ref, out = _export_and_run(M(), _t(2, 4))
    torch.testing.assert_close(out, ref)


def test_export_topk_sort_getitems():
    class M(torch.nn.Module):
        def forward(self, x):
            v, i = torch.topk(x, 2, dim=-1)
            sv, si = torch.sort(x, dim=-1, descending=True)
            return (
                v.sum() + i.float().sum() + sv.sum() + si.float().sum()
            )

    _, ref, out = _export_and_run(M(), _t(2, 4))
    torch.testing.assert_close(out, ref)


def test_export_einsum():
    class M(torch.nn.Module):
        def forward(self, x, y):
            return torch.einsum("bij,bjk->bik", x, y)

    _, ref, out = _export_and_run(M(), _t(2, 4, 3), _t(2, 3, 4))
    torch.testing.assert_close(out, ref)


def test_export_tensor_split_fold():
    class M(torch.nn.Module):
        def forward(self, x):
            a, b = torch.tensor_split(x, 2, dim=-1)
            return a + b

    ir, ref, out = _export_and_run(M(), _t(2, 8))
    torch.testing.assert_close(out, ref)
    splits = [o for o in _iter_ops(ir.root) if o.op == "tensor_split"]
    assert splits and all("index" in t.attrs for t in splits)


def test_export_arange_stays_int64():
    """arange's bounds are attr ints, not Consts — a Const would
    erase intness and produce a float index tensor."""

    class M(torch.nn.Module):
        def forward(self, x):
            return x[torch.arange(x.shape[1]) % 2]

    ir, ref, out = _export_and_run(M(), _t(2, 4, 4))
    torch.testing.assert_close(out, ref)
    rng = [o for o in _iter_ops(ir.root) if o.op == "arange"]
    assert rng and all(
        isinstance(v, int) for v in rng[0].attrs.values()
    )


def test_export_scatter_reduce():
    class M(torch.nn.Module):
        def forward(self, x):
            idx = torch.arange(2).view(1, 2, 1).expand(2, 2, 4)
            return x.scatter_reduce(
                1, idx, torch.ones(2, 2, 4), reduce="mean"
            )

    _, ref, out = _export_and_run(M(), _t(2, 4, 4))
    torch.testing.assert_close(out, ref)


def test_export_linalg_inv():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.linalg.inv(x[0] + torch.eye(4)).unsqueeze(0)

    _, ref, out = _export_and_run(M(), _t(1, 4, 4))
    torch.testing.assert_close(out, ref)


def test_export_tile_maps_to_repeat():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.tile(x, (1, 2))

    ir, ref, out = _export_and_run(M(), _t(2, 3))
    torch.testing.assert_close(out, ref)
    assert any(o.op == "repeat" for o in _iter_ops(ir.root))


def test_export_mm_bmm_mv_unify_matmul():
    class M(torch.nn.Module):
        def forward(self, x):
            return torch.mm(x[0], x[0].t()) + torch.bmm(x, x)

    ir, ref, out = _export_and_run(M(), _t(3, 4, 4))
    torch.testing.assert_close(out, ref)
    assert not [o for o in _iter_ops(ir.root) if o.op in ("mm", "bmm")]


class _InstanceNormModel(torch.nn.Module):
    """``nn.InstanceNorm2d`` — the corpus-expansion gap export."""

    def __init__(self, affine: bool, track_running_stats: bool) -> None:
        super().__init__()
        self.norm = torch.nn.InstanceNorm2d(
            4, affine=affine, track_running_stats=track_running_stats
        )
        if track_running_stats:
            self.norm.running_mean.copy_(torch.arange(4.0))
            self.norm.running_var.copy_(torch.arange(1.0, 5.0))

    def forward(self, x):
        return self.norm(x)


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("track_running_stats", [True, False])
def test_export_instance_norm(affine, track_running_stats):
    """All four affine/track_running_stats combos export the
    ``instance_norm`` term, carry the canonical scalar attrs, and
    lower numerically-equal (the running-stats operand pair is
    present iff use_input_stats is False)."""
    m = _InstanceNormModel(affine, track_running_stats).eval()
    x = _t(2, 4, 8, 8)
    ir, ref, out = _export_and_run(m, x)
    torch.testing.assert_close(out, ref)
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "instance_norm"]
    assert t.attrs["use_input_stats"] is not track_running_stats
    assert t.attrs["eps"] == pytest.approx(1e-5)
    assert not any(k.startswith("arg") for k in t.attrs), t.attrs
    # operands: x + affine pair? + running-stats pair iff eval-stats.
    assert len(t.args) == 1 + 2 * affine + 2 * track_running_stats


class _UpsampleModel(torch.nn.Module):
    def __init__(self, **kw) -> None:
        super().__init__()
        self.up = torch.nn.Upsample(mode="nearest", **kw)

    def forward(self, x):
        return self.up(x)


def test_export_upsample_nearest2d_scale():
    """``nn.Upsample(scale_factor=2)`` exports
    ``upsample_nearest2d.vec`` — canonicalized to
    ``upsample_nearest2d`` with the ``scale`` attr."""
    m = _UpsampleModel(scale_factor=2).eval()
    x = _t(2, 4, 8, 8)
    ir, ref, out = _export_and_run(m, x)
    torch.testing.assert_close(out, ref)
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "upsample_nearest2d"]
    assert t.attrs["scale"] == (2.0, 2.0)
    assert not any(k.startswith("arg") for k in t.attrs), t.attrs


def test_export_upsample_nearest2d_size():
    """The ``size``-spelled interpolate lands under the ``size``
    attr of the same canonical op."""
    m = _UpsampleModel(size=(5, 12)).eval()
    x = _t(2, 4, 8, 8)
    ir, ref, out = _export_and_run(m, x)
    torch.testing.assert_close(out, ref)
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "upsample_nearest2d"]
    assert t.attrs["size"] == (5, 12)
    assert "scale" not in t.attrs


class _LstmModel(torch.nn.Module):
    """``nn.LSTM`` — the (output, h_n, c_n) triple; the model reads
    element 0 through the ``getitem`` consumer."""

    def __init__(self, **kw) -> None:
        super().__init__()
        self.lstm = torch.nn.LSTM(8, 6, **kw)

    def forward(self, x):
        y, _ = self.lstm(x)
        return y


@pytest.mark.parametrize(
    "kw", [
        {"batch_first": True},
        {},
        {"batch_first": True, "num_layers": 2},
        {"batch_first": True, "bidirectional": True},
    ]
)
def test_export_lstm_input(kw):
    """The ``lstm.input`` term lowers through the aten passthrough —
    batch_first/seq-first, stacked, and bidirectional exports all
    verify; the scalar tail lands under canonical names."""
    m = _LstmModel(**kw).eval()
    x = _t(2, 5, 8)
    ir, ref, out = _export_and_run(m, x)
    torch.testing.assert_close(out, ref)
    (t,) = [o for o in _iter_ops(ir.root) if o.op == "lstm.input"]
    assert t.attrs["num_layers"] == kw.get("num_layers", 1)
    assert t.attrs["batch_first"] == kw.get("batch_first", False)
    assert t.attrs["bidirectional"] == kw.get("bidirectional", False)
    assert t.attrs["train"] is False
    assert not any(k.startswith("arg") for k in t.attrs), t.attrs
    # the consumer is getitem(0) — the triple's output element.
    g = [o for o in _iter_ops(ir.root) if o.op == "getitem"]
    assert g and g[0].args[0] is t


def test_new_bindings_are_supported_ops():
    """supported_ops coverage — the three gap ops are bound, so the
    backend can lower them (extraction's hard feasibility set)."""
    from catopt_torch.adapters import TorchSink

    assert {
        "instance_norm",
        "upsample_nearest2d",
        "lstm.input",
    } <= TorchSink().supported_ops


def test_instance_norm_uis_false_without_stats_is_loud():
    """``use_input_stats=False`` with no running-stat operands is a
    malformed spelling — the binding must not silently normalize
    with input stats instead; aten raises."""
    with pytest.raises(RuntimeError, match="running_mean"):
        _IR_TO_TORCH["instance_norm"](
            _t(2, 4, 4), use_input_stats=False
        )


# ---------------------------------------------------------------------------
#  4. The opaque boundary — an unbound op is a boundary, not a failure
# ---------------------------------------------------------------------------


class _UnboundWrap(torch.nn.Module):
    """``fft_fft`` stays unbound — an opaque root over an optimizable
    matmul chain."""

    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(8, 8))
        self.w2 = torch.nn.Parameter(torch.randn(8, 8))

    def forward(self, x):
        y = x @ self.w1 @ self.w2
        return torch.fft.fft(y).real


def test_unbound_op_is_a_boundary_not_a_failure():
    """``optimize_model`` on a graph containing an unbound op still
    optimizes everything around it: the param chain folds to a fused
    Param at lowering while the opaque op sits outside, and the
    delivered module fails loudly ("No torch binding"), never
    silently."""


    m = _UnboundWrap().eval()
    x = _t(4, 8)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(m, x, max_iterations=2, verify=False, verbose=False)

    # The bound subtree was still optimized — the matmul chain folded
    # into one fused parameter at lowering.
    fused = [k for k in opt.state_dict() if k.startswith("fused")]
    assert fused, (
        f"expected a fused param, got {list(opt.state_dict())}"
    )
    # The opaque op is a loud boundary, not a silent wrong result.
    with pytest.raises(ValueError, match="No torch binding for op"):
        opt(x)


def test_compositional_falls_back_around_unbound_op():
    """A block whose graph contains an unbound op falls back to the
    original; a sibling block still optimizes and the recomposed
    model stays correct."""
    import torch.nn as nn


    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8, bias=False)
            self.fft = _UnboundWrap()

        def forward(self, x):
            return self.fft(self.lin(x))

    model = Outer().eval()
    x = _t(4, 8)
    opt, stats = Optimizer(backend=TorchBackend()).optimize(model, x, strategy=Compositional(), verbose=False)

    # The linear block optimized; the fft block fell back (its eval
    # failure is a boundary, not a wrong result).
    lin = stats["blocks"]["lin"]
    fft = stats["blocks"]["fft"]
    assert lin["status"] == "optimized"
    assert fft["status"] == "failed"
    # End-to-end the recomposed model still reproduces the original.
    with torch.no_grad():
        torch.testing.assert_close(opt(x), model(x))


# ---------------------------------------------------------------------------
#  5. Boundary internals — the copy_ threader's defensive edges and the
#     minted (layout-free) index_put spelling
# ---------------------------------------------------------------------------


def test_minted_index_put_without_layout():
    """A minted ``index_put(t, idx, v)`` (no layout attr) treats all
    but the last operand as leading-axis indices."""
    x, i, v = _t(3, 4), torch.arange(2), _t(2, 4)
    out = _IR_TO_TORCH["index_put"](x, i, v)
    torch.testing.assert_close(
        out, torch.index_put(x, (i,), v, accumulate=False)
    )


def test_handle_copy_early_returns():
    """``_handle_copy_`` no-ops on missing args, missing env entries,
    and a missing viewed base."""
    from catopt_torch.torch_bridge import _handle_copy_

    class _N:
        def __init__(self, name, args=(), target=None):
            self.name = name
            self.args = args
            self.target = target

    env = {}
    _handle_copy_(_N("c", args=()), env)  # len < 2
    assert env == {}

    dst = _N("d", target="x")
    src = _N("s", target="x")
    env = {}
    _handle_copy_(_N("c", args=(dst, src)), env)
    assert env == {}  # src/dst not in env -> skip

    # A slice dst whose BASE is absent from env -> skip.
    base = _N("b")
    view = _N("v", args=(base, 0, 1, 2), target="slice")
    node = _N("c", args=(view, src))
    env = {"v": Var("v", TensorType((2, 4))), "s": Const(1.0)}
    _handle_copy_(node, env)
    assert env == {"v": env["v"], "s": env["s"]}

    # Same for a select view missing its base.
    sview = _N("sv", args=(base, 0, 1), target="select")
    node2 = _N("c2", args=(sview, src))
    env = {"sv": Var("sv", TensorType((4,))), "s": Const(1.0)}
    _handle_copy_(node2, env)
    assert env == {"sv": env["sv"], "s": env["s"]}


def test_handle_copy_unrecognised_view_is_whole_tensor_copy():
    """A copy_ through a view we don't decompose (e.g. a transpose)
    rewrites the view's own readers to the ``copy`` op."""
    from catopt_torch.torch_bridge import _handle_copy_

    class _N:
        def __init__(self, name, args=(), target=None):
            self.name = name
            self.args = args
            self.target = target

    base = Var("b", TensorType((2, 4)))
    src = Var("s", TensorType((2, 4)))
    view = _N("v", target="transpose")
    node = _N("c", args=(view, _N("s")))
    env = {"v": base, "s": src}
    _handle_copy_(node, env)
    assert env["v"].op == "copy"
    assert env["c"] is env["v"]


def test_resolve_attr_walks_dotted_names():
    from catopt_torch.torch_bridge import _resolve_attr

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(_t(3))

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Inner()

    m = Outer()
    assert _resolve_attr(m, "inner.w") is m.inner.w
