"""Attr-schema contract tests — plan 0001 phase 1b.

``catopt.attrs.ATTR_SCHEMA`` declares the canonical positional order
per op; ``export_to_ir`` lands aten ``argN`` spellings in those names
at the boundary, and ``Op.make`` canonicalises + validates at mint so
a malformed term dies loudly at union time rather than silently at
eval.  These tests pin all three:

1. **Export canonicalisation** — per-op probe modules through
   ``export_to_ir``; every schema'd op must carry the canonical names
   and no bare ``argN`` at a schema-declared position.
2. **Mint-time contract** — ``Op.make`` renames declared positionals,
   rejects ``argN`` at undeclared positions and missing required
   attrs, keeps partial/zero-attr terms, and honours ``validate=False``.
3. **Round-trip** — a batched rope-style block (the rank-5
   ``stack``→``reshape`` failure shape) exports, lowers through
   ``ir_to_torch_module``, and reproduces the module's outputs with
   canonical attrs everywhere.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from catopt.attrs import ATTR_REQUIRED, ATTR_SCHEMA, canonicalize_attrs
from catopt.ir import Op, TensorType, Var
from catopt.torch_bridge import export_to_ir, ir_to_torch_module

_INT64_MAX = 2**63 - 1


def _var(shape: tuple, name: str) -> Var:
    return Var(name=name, typ=TensorType(tuple(shape)))


def _iter_ops(term):
    """Yield every Op node in a term (DAG-aware by identity)."""
    seen = set()
    stack = [term]
    while stack:
        t = stack.pop()
        if isinstance(t, Op) and id(t) not in seen:
            seen.add(id(t))
            yield t
            stack.extend(t.args)


def _export(model, *args):
    ir, tensors = export_to_ir(model.eval(), tuple(args))
    return ir, tensors, list(_iter_ops(ir.root))


def _ops_named(ops, name):
    return [t for t in ops if t.op == name]


def _assert_no_noncanonical_argn(t: Op):
    """No ``argN`` may survive where the schema declares a name."""
    schema = ATTR_SCHEMA.get(t.op, {})
    for key in t.attrs:
        if not (isinstance(key, str) and key.startswith("arg")):
            continue
        pos = int(key[3:])
        assert schema.get(pos) == key, (
            f"{t.op}: bare positional '{key}' but schema declares "
            f"canonical name {schema.get(pos)!r} — attrs {t.attrs}"
        )


# ---------------------------------------------------------------------------
#  Probe modules — one per schema'd op family
# ---------------------------------------------------------------------------


class _M(torch.nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, *xs):
        return self.f(*xs)


class _RMSNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(8))

    def forward(self, x):
        return F.rms_norm(x, (8,), self.w, eps=1e-5)


class _LayerNorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(8))
        self.b = torch.nn.Parameter(torch.zeros(8))

    def forward(self, x):
        return F.layer_norm(x, (8,), self.w, self.b, eps=1e-5)


class _SDPA(torch.nn.Module):
    def forward(self, q, k, v):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=0.5
        )


class _SDPADropout(torch.nn.Module):
    def forward(self, q, k, v):
        # eval() → train=False positional arg; dropout_p is arg4.
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)


class _Conv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)

    def forward(self, x):
        return self.c(x)


class _Chunk(torch.nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return a + b


class _SplitList(torch.nn.Module):
    def forward(self, x):
        a, b = torch.split(x, [3, 5], dim=-1)
        return a.sum() + b.sum()


class _SplitInt(torch.nn.Module):
    def forward(self, x):
        a, b = torch.split(x, 4, dim=-1)
        return a + b


class _Cat(torch.nn.Module):
    def forward(self, x, y):
        return torch.cat([x, y], dim=-1)


class _Stack(torch.nn.Module):
    def forward(self, x, y):
        return torch.stack([x, y], dim=1)


class _Unbind(torch.nn.Module):
    def forward(self, x):
        a, b = x.unbind(-1)
        return a + b


# ---------------------------------------------------------------------------
#  1. Export canonicalisation
# ---------------------------------------------------------------------------


class TestExportCanonicalization:
    """Every schema'd op's exported attrs land canonical."""

    def _check(self, model, args, op, expected):
        _, _, ops = _export(model, *args)
        found = _ops_named(ops, op)
        assert found, f"expected a {op} node in the exported graph"
        for t in found:
            _assert_no_noncanonical_argn(t)
        return found

    def test_slice_strided(self):
        (t,) = self._check(
            _M(lambda x: x[..., ::2]),
            (torch.randn(2, 3, 8),),
            "slice",
            None,
        )
        # dim is canonical; start/end/step stay positional — declared
        # as the canonical argN spelling (typing reads them literally).
        assert t.attrs["dim"] == 2
        assert t.attrs["arg2"] == 0
        assert t.attrs["arg3"] == _INT64_MAX
        assert t.attrs["arg4"] == 2

    def test_slice_plain(self):
        (t,) = self._check(
            _M(lambda x: x[:, 1:5]),
            (torch.randn(2, 3, 8),),
            "slice",
            None,
        )
        assert t.attrs["dim"] == 1
        assert t.attrs["arg2"] == 1
        assert t.attrs["arg3"] == 5

    def test_select(self):
        (t,) = self._check(
            _M(lambda x: x.select(1, 2)),
            (torch.randn(2, 3, 8),),
            "select",
            None,
        )
        assert t.attrs == {"dim": 1, "index": 2}

    def test_unsqueeze(self):
        for t in self._check(
            _M(lambda x: x.unsqueeze(1)),
            (torch.randn(2, 3, 8),),
            "unsqueeze",
            None,
        ):
            # arg1 IS the canonical spelling for unsqueeze — rule
            # patterns mint arg1; no dim-variant rules exist.
            assert t.attrs == {"arg1": 1}

    def test_squeeze(self):
        for t in self._check(
            _M(lambda x: x.unsqueeze(1).squeeze(1)),
            (torch.randn(2, 3, 8),),
            "squeeze",
            None,
        ):
            assert t.attrs == {"arg1": 1}

    def test_transpose(self):
        (t,) = self._check(
            _M(lambda x: x.transpose(0, 2)),
            (torch.randn(2, 3, 8),),
            "transpose",
            None,
        )
        # arg1/arg2 are the canonical spellings for transpose today —
        # every rule pattern mints them.
        assert t.attrs == {"arg1": 0, "arg2": 2}

    def test_flatten(self):
        (t,) = self._check(
            _M(lambda x: x.flatten(1, 2)),
            (torch.randn(2, 3, 8),),
            "flatten",
            None,
        )
        assert t.attrs == {"start_dim": 1, "end_dim": 2}

    def test_softmax(self):
        (t,) = self._check(
            _M(lambda x: F.softmax(x, dim=-1)),
            (torch.randn(2, 3, 8),),
            "softmax",
            None,
        )
        assert t.attrs == {"arg1": -1}

    def test_concat(self):
        (t,) = self._check(
            _Cat(),
            (torch.randn(2, 4), torch.randn(2, 4)),
            "concat",
            None,
        )
        assert t.attrs == {"dim": -1}

    def test_stack(self):
        (t,) = self._check(
            _Stack(),
            (torch.randn(2, 4), torch.randn(2, 4)),
            "stack",
            None,
        )
        assert t.attrs == {"dim": 1}

    def test_chunk(self):
        found = self._check(
            _Chunk(), (torch.randn(2, 8),), "chunk", None
        )
        for t in found:
            assert t.attrs["chunks"] == 2
            assert t.attrs["dim"] == -1
            assert "index" in t.attrs  # the getitem fold

    def test_split_sizes(self):
        found = self._check(
            _SplitList(), (torch.randn(2, 8),), "split", None
        )
        for t in found:
            assert t.attrs["sizes"] == (3, 5)
            assert t.attrs["dim"] == -1
            assert "index" in t.attrs

    def test_split_int(self):
        found = self._check(
            _SplitInt(), (torch.randn(2, 8),), "split", None
        )
        for t in found:
            # int split size is the canonical ``sizes`` attr.
            assert t.attrs["sizes"] == 4
            assert t.attrs["dim"] == -1
            assert "index" in t.attrs

    def test_unbind(self):
        found = self._check(
            _Unbind(), (torch.randn(2, 2),), "unbind", None
        )
        for t in found:
            assert t.attrs["dim"] == -1
            assert "index" in t.attrs

    def test_unbind_default_dim_has_no_dim_attr(self):
        # aten.unbind(x, 0) elides the default dim — the folded term
        # carries only ``index`` and must stay legal (partial attrs).
        class _U0(torch.nn.Module):
            def forward(self, x):
                a, b = x.unbind(0)
                return a + b

        found = self._check(_U0(), (torch.randn(2, 8),), "unbind", None)
        assert found
        for t in found:
            assert "dim" not in t.attrs or t.attrs["dim"] == 0
            assert "index" in t.attrs

    def test_index_select(self):
        (t,) = self._check(
            _M(lambda x: x.index_select(1, torch.tensor([0, 2]))),
            (torch.randn(2, 8),),
            "index_select",
            None,
        )
        assert t.attrs["dim"] == 1

    def test_narrow(self):
        (t,) = self._check(
            _M(lambda x: x.narrow(1, 0, 2)),
            (torch.randn(2, 8),),
            "narrow",
            None,
        )
        assert t.attrs == {"dim": 1, "start": 0, "length": 2}

    def test_rms_norm(self):
        (t,) = self._check(
            _RMSNorm(), (torch.randn(2, 8),), "rms_norm", None
        )
        assert t.attrs["dim"] == (8,)
        assert t.attrs["eps"] == pytest.approx(1e-5)

    def test_layer_norm_eps_is_eps_not_cudnn(self):
        """Regression: aten.layer_norm's eps is arg4 — arg5 is the
        cudnn flag.  The old binding read arg5 and silently set
        eps=0.0."""
        (t,) = self._check(
            _LayerNorm(), (torch.randn(2, 8),), "layer_norm", None
        )
        assert t.attrs["dim"] == (8,)
        assert t.attrs["eps"] == pytest.approx(1e-5)
        assert t.attrs.get("cudnn_enabled") is False

    def test_sdpa_causal_scale(self):
        (t,) = self._check(
            _SDPA(),
            (
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 2, 4, 8),
            ),
            "sdpa",
            None,
        )
        assert t.attrs["is_causal"] is True
        assert t.attrs["dropout_p"] == 0.0
        assert t.attrs["scale"] == 0.5

    def test_sdpa_dropout_p(self):
        (t,) = self._check(
            _SDPADropout(),
            (
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 2, 4, 8),
            ),
            "sdpa",
            None,
        )
        assert t.attrs["dropout_p"] == pytest.approx(0.1)

    def test_dropout_p_train(self):
        (t,) = self._check(
            _M(lambda x: F.dropout(x, p=0.5)),
            (torch.randn(4, 8),),
            "dropout",
            None,
        )
        # arg1=p / arg2=train are the canonical spellings (rules mint
        # them — rules.py's dropout-in-attention pattern).
        assert t.attrs == {"arg1": 0.5, "arg2": True}

    def test_conv2d_positional_attrs(self):
        (t,) = self._check(
            _Conv(), (torch.randn(1, 3, 8, 8),), "conv2d", None
        )
        # torch.export elides defaulted tails — stride/padding arrive
        # as named attrs (never argN); dilation/groups only when set.
        assert t.attrs["stride"] == (2, 2)
        assert t.attrs["padding"] == (1, 1)


# ---------------------------------------------------------------------------
#  2. Mint-time contract — Op.make validates and fails loudly
# ---------------------------------------------------------------------------


class TestMintValidation:
    x = _var((2, 8), "x")

    # -- declared-position argN is a legal minted spelling (preserved) --
    # the *_arg1 rule variants in om.py/xcarrier.py pattern-match
    # arg-spelled e-nodes — renaming at mint would kill them.
    def test_argN_at_declared_position_preserved(self):
        t = Op.make("concat", self.x, self.x, arg1=-1)
        assert t.attrs == {"arg1": -1}

    def test_transpose_argN_preserved(self):
        t = Op.make("transpose", self.x, arg1=0, arg2=1)
        assert t.attrs == {"arg1": 0, "arg2": 1}

    def test_split_argN_preserved(self):
        t = Op.make("split", self.x, arg1=4, arg2=-1, arg3=0)
        assert t.attrs == {"arg1": 4, "arg2": -1, "arg3": 0}

    def test_sdpa_argN_preserved(self):
        t = Op.make(
            "sdpa",
            self.x,
            self.x,
            self.x,
            arg4=0.0,
            arg5=True,
            arg7=True,
        )
        assert t.attrs == {"arg4": 0.0, "arg5": True, "arg7": True}

    def test_slice_positional_canonical_spelling_kept(self):
        # arg2/arg3/arg4 ARE the canonical names for slice today.
        t = Op.make("slice", self.x, arg1=2, arg2=0, arg3=4, arg4=2)
        assert t.attrs == {"arg1": 2, "arg2": 0, "arg3": 4, "arg4": 2}

    def test_mixed_spellings_coexist(self):
        # the dual-spelling ecosystem: canonical dim + positional
        # arg2/arg3 — exactly what test_contracts pins on _shape_of.
        t = Op.make("split", self.x, arg1=(4, 4), arg3=1, dim=0)
        assert t.attrs == {"arg1": (4, 4), "arg3": 1, "dim": 0}

    def test_minted_argN_terms_evaluate_through_bindings(self):
        # Preserved minted spellings still lower: the bindings' argN
        # fallbacks are live code for these terms.
        from catopt.torch_bridge import _IR_TO_TORCH

        out = _IR_TO_TORCH["transpose"](
            torch.arange(6).reshape(2, 3), arg1=0, arg2=1
        )
        assert out.shape == (3, 2)
        out = _IR_TO_TORCH["select"](
            torch.arange(6).reshape(2, 3), arg1=0, arg2=1
        )
        assert out.shape == (3,)
        out = _IR_TO_TORCH["concat"](
            torch.ones(2, 2), torch.ones(2, 2), arg1=-1
        )
        assert out.shape == (2, 4)

    # -- loud failures ---------------------------------------------------
    def test_argN_past_schema_fails(self):
        with pytest.raises(ValueError, match=r"arg9"):
            Op.make("split", self.x, arg9=1, sizes=(4, 4), dim=-1)

    def test_argN_undeclared_position_fails(self):
        # sdpa position 1 is the k operand — never an attr.
        with pytest.raises(ValueError, match="arg1"):
            Op.make(
                "sdpa", self.x, self.x, self.x, arg1=1, dropout_p=0.0
            )

    def test_missing_required_attr_fails(self):
        # split carrying schema attrs but no ``sizes`` is malformed.
        with pytest.raises(ValueError, match="sizes"):
            Op.make("split", self.x, dim=-1)

    def test_missing_required_transpose_dim1_fails(self):
        with pytest.raises(ValueError, match="arg2"):
            Op.make("transpose", self.x, arg1=0)

    # -- partial / bare terms stay legal ----------------------------------
    def test_zero_attr_term_is_partial(self):
        # Pattern terms (e.g. om_lift_plain's softmax("s")) carry no
        # attrs — required checks don't apply to them.
        t = Op.make("softmax", self.x)
        assert t.attrs == {}
        t = Op.make("split", self.x)
        assert t.attrs == {}

    def test_extra_named_attrs_allowed(self):
        # Non-positional names outside the schema (e.g. getitem-fold
        # ``index`` on unbind, binding-only kwargs) are legal — and a
        # term carrying only them is partial, not required-checked.
        t = Op.make("unbind", self.x, arg1=-1, index=3)
        assert t.attrs == {"arg1": -1, "index": 3}
        t = Op.make("unbind", self.x, index=3)
        assert t.attrs == {"index": 3}

    def test_ops_without_schema_untouched(self):
        t = Op.make("some_op", self.x, arg1=1, whatever=2)
        assert t.attrs == {"arg1": 1, "whatever": 2}

    def test_validate_false_escape(self):
        t = Op.make("split", self.x, arg9=1, validate=False)
        assert t.attrs == {"arg9": 1}

    # -- positional patch wins over a pre-supplied canonical key --------
    def test_argN_overrides_duplicate_canonical(self):
        # optimize._specialize_causal patches attrs["arg5"]=True onto
        # an sdpa term that may already carry is_causal — the positional
        # write wins and the stored spelling stays canonical.
        t = Op.make(
            "sdpa",
            self.x,
            self.x,
            self.x,
            is_causal=False,
            arg5=True,
        )
        assert t.attrs == {"is_causal": True}

    # -- canonicalize_attrs: the explicit full-rename helper -------------
    def test_canonicalize_renames_declared_positions(self):
        assert canonicalize_attrs(
            "split", {"arg1": 4, "arg2": -1, "arg3": 0}
        ) == {"sizes": 4, "dim": -1, "index": 0}
        # identity-canonical positions are a no-op
        assert canonicalize_attrs(
            "transpose", {"arg1": 1, "arg2": 2}
        ) == {
            "arg1": 1,
            "arg2": 2,
        }
        # undeclared positions pass through
        assert canonicalize_attrs("split", {"arg9": 1}) == {"arg9": 1}
        # no schema → untouched
        assert canonicalize_attrs("zzz", {"arg1": 1}) == {"arg1": 1}

    def test_canonicalize_idempotent(self):
        a = canonicalize_attrs("split", {"sizes": (4, 4), "dim": -1})
        b = canonicalize_attrs("split", a)
        assert a == b == {"sizes": (4, 4), "dim": -1}


# ---------------------------------------------------------------------------
#  3. Schema hygiene
# ---------------------------------------------------------------------------


class TestSchemaHygiene:
    def test_required_names_are_schema_names(self):
        for op, req in ATTR_REQUIRED.items():
            names = set(ATTR_SCHEMA[op].values())
            assert req <= names, (
                f"{op}: required {sorted(req)} not all canonical names"
            )

    def test_no_duplicate_canonical_names_per_op(self):
        for op, schema in ATTR_SCHEMA.items():
            names = list(schema.values())
            assert len(names) == len(set(names)), (
                f"{op}: duplicate canonical names in schema"
            )


# ---------------------------------------------------------------------------
#  4. Round-trip — batched rope block (the rank-5 stack→reshape shape)
# ---------------------------------------------------------------------------


class _RopeBlock(torch.nn.Module):
    """Batched rope-style block: rank-5 stack+reshape rope plus qkv
    GEMMs — the reported decode-bench failure shape."""

    def __init__(self, dim: int = 48, nh: int = 4, hd: int = 12):
        super().__init__()
        self.nh, self.hd = nh, hd
        self.wq = torch.nn.Linear(dim, nh * hd, bias=False)
        self.wk = torch.nn.Linear(dim, nh * hd, bias=False)
        self.wv = torch.nn.Linear(dim, nh * hd, bias=False)
        self.wo = torch.nn.Linear(nh * hd, dim, bias=False)

    def rope(self, x, cos, sin):
        B, T = x.shape[0], x.shape[1]
        x = x.reshape(B, T, self.nh, self.hd)
        x1, x2 = x[..., ::2], x[..., 1::2]
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1)
        return out.reshape(B, T, self.nh * self.hd)

    def forward(self, h, cos, sin):
        B, T = h.shape[0], h.shape[1]
        q = self.rope(self.wq(h), cos, sin)
        k = self.rope(self.wk(h), cos, sin)
        v = self.wv(h)
        q = q.reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.reshape(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.reshape(B, T, self.nh, self.hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


def test_batched_rope_roundtrip_canonical_attrs():
    """Export → IRModule round-trip: correct output AND canonical
    attrs on every schema'd op in the exported term."""
    torch.manual_seed(0)
    B, T = 2, 8
    model = _RopeBlock().eval()
    h = torch.randn(B, T, 48)
    cos = torch.randn(T, 6)
    sin = torch.randn(T, 6)

    ir, tensors, ops = _export(model, h, cos, sin)
    # no argN survives where a canonical name exists
    for t in ops:
        _assert_no_noncanonical_argn(t)
    # and the rope machinery shows up in canonical form
    assert _ops_named(ops, "slice"), "expected rope slices"
    assert _ops_named(ops, "stack"), "expected rope stack"
    assert _ops_named(ops, "sdpa"), "expected the attention node"
    for t in _ops_named(ops, "stack"):
        assert "dim" in t.attrs
    for t in _ops_named(ops, "sdpa"):
        assert t.attrs.get("is_causal") is True

    lowered = ir_to_torch_module(ir, param_values=tensors)
    with torch.no_grad():
        ref = model(h, cos, sin)
        out = lowered(h, cos, sin)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
