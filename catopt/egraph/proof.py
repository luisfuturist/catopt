"""Proof mixin: certificates, coherence, explanations."""
from __future__ import annotations

from typing import Any

from catopt.egraph.certs import (
    Certificate,
    CertStep,
)
from catopt.egraph.terms import (
    _replace_subterm,
    _subterm,
    _term_instantiate,
    _term_match,
    _term_paths,
)
from catopt.egraph.types import ENode, _LeafRegistry
from catopt.ir import Op, op_repr


class _ProofMixin:

    def _birth(self, enode: ENode) -> int:
        """Creation order of an enode (its birth eid); huge if unknown."""
        b = self._enode_birth.get(enode)
        if b is not None:
            return b
        eid = self._node_to_class.get(enode)
        return eid if eid is not None else 1 << 60

    def _oldest_term(
        self,
        eid: int,
        _stack: frozenset = frozenset(),
        _memo: dict | None = None,
    ):
        """The earliest-created representative term of an e-class.

        Proof-time analogue of :meth:`any_term`: picking the minimum-
        birth enode at every level makes target-side expansion in
        :meth:`_connect` a strictly descending recursion — every rule
        instance's LHS was matched on enodes older than the RHS enodes
        it created.

        ``_memo`` shares resolutions across the descent (e-class cones
        are heavily shared); a memoised member is still a valid
        member, and ``None`` results are never cached (a class blocked
        by ``_stack`` may resolve under another ancestry).
        """
        if _memo is None:
            _memo = {}
        eid = self.find(eid)
        if eid in _stack:
            return None
        hit = _memo.get(eid)
        if hit is not None:
            return hit
        ec = self._classes.get(eid)
        if ec is None:
            return None
        _stack = _stack | {eid}
        for node in sorted(ec.nodes, key=self._birth):
            if node.op == "leaf":
                key = node.attrs[0][1] if node.attrs else "??"
                _memo[eid] = _LeafRegistry.decode(key)
                return _memo[eid]
            args = []
            ok = True
            for c in node.children:
                t = self._oldest_term(c, _stack, _memo)
                if t is None:
                    ok = False
                    break
                args.append(t)
            if ok:
                _memo[eid] = Op.make(node.op, *args, **dict(node.attrs))
                return _memo[eid]
        return None

    def _app_for_member(self, term: Any):
        """The rule application whose RHS root realises *term*'s root.

        Returns ``(app, enode)`` — ``app`` is ``None`` when the member
        enode is not the RHS root of any recorded application (input
        term, internal RHS node, or pass-introduced).
        """
        _eid, en = self._locate(term)
        if en is None:
            return None, None
        ai = self._enode_app.get(en)
        if ai is None:
            return None, en
        app = self._applications[ai]
        root_en = app["rhs_root_enode"]
        if root_en is None:
            return None, en
        # ``en`` must be the application's RHS root — children compare
        # modulo canonicalisation (rebuild may have rewritten ids).
        if (
            root_en.op != en.op
            or root_en.attrs != en.attrs
            or len(root_en.children) != len(en.children)
        ):
            return None, en
        if not all(
            self.find(a) == self.find(b)
            for a, b in zip(root_en.children, en.children, strict=True)
        ):
            return None, en
        return app, en

    def _resolve_subst(self, subst: dict) -> dict | None:
        """Resolve a fired binding to concrete terms.

        Metavariable eids map to their class's :meth:`_oldest_term`;
        ``"$attr:"`` keys carry their concrete value through.  ``None``
        when a bound class is unresolvable (cyclic).
        """
        out = {}
        for k, v in subst.items():
            if k.startswith("$attr:"):
                out[k] = v
            else:
                t = self._oldest_term(v)
                if t is None:
                    return None
                out[k] = t
        return out

    def _connect(
        self, s: Any, t: Any, pos: tuple, steps: list, depth: int
    ) -> bool:
        """Emit steps rewriting the subterm at ``pos`` from ``s`` to ``t``.

        Both terms are members of one e-class.  Returns True when every
        emitted step is a standalone-replayable rule application; False
        means at least one ``egraph_dependent`` stub was emitted.  The
        strategy, in order:

        1. **Expand the target** through the rule application that
           created its root enode: recursively bridge ``s`` to the
           application's LHS instance, emit the rewrite, then fix the
           produced children pairwise.
        2. **Congruence**: same head op/attrs and pairwise-equivalent
           children — descend without emitting a step.
        3. **Edge search**: a short bounded search over the rules that
           fired, covering merges whose RHS is a bare metavariable or a
           pre-existing hash-consed enode (no expandable enode exists).
        4. **e-graph-dependent stub**: the merge is real but has no
           standalone derivation (non-local pass, manual union).
        """
        if op_repr(s) == op_repr(t):
            return True
        if (
            depth > self._CERT_MAX_DEPTH
            or len(steps) > self._CERT_MAX_STEPS
        ):
            steps.append(
                CertStep(
                    "<budget>",
                    pos,
                    s,
                    t,
                    {},
                    egraph_dependent=True,
                    note="derivation budget exceeded",
                )
            )
            return False

        app, _en = self._app_for_member(t)
        if app is not None:
            rule = self._rule_objs.get(app["rule"])
            bound = (
                self._resolve_subst(app["subst"])
                if rule is not None
                else None
            )
            if bound is not None:
                L = _term_instantiate(rule.lhs, bound)
                R = _term_instantiate(rule.rhs, bound)
                ok = self._connect(s, L, pos, steps, depth + 1)
                steps.append(CertStep(rule.name, pos, L, R, bound))
                if isinstance(R, Op) and isinstance(t, Op):
                    for i in range(len(t.args)):
                        if not self._connect(
                            R.args[i],
                            t.args[i],
                            (*pos, i),
                            steps,
                            depth + 1,
                        ):
                            ok = False
                return ok

        if (
            isinstance(s, Op)
            and isinstance(t, Op)
            and s.op == t.op
            and s.attrs == t.attrs
            and len(s.args) == len(t.args)
        ):
            cs = [self._class_of_term(a) for a in s.args]
            ct = [self._class_of_term(a) for a in t.args]
            if all(a is not None and a == b for a, b in zip(cs, ct, strict=True)):
                ok = True
                for i in range(len(s.args)):
                    if not self._connect(
                        s.args[i],
                        t.args[i],
                        (*pos, i),
                        steps,
                        depth + 1,
                    ):
                        ok = False
                return ok

        path = self._edge_path(s, t, pos)
        if path is not None:
            steps.extend(path)
            return True

        steps.append(
            CertStep(
                "<egraph>",
                pos,
                s,
                t,
                {},
                egraph_dependent=True,
                note=self._explain_gap(s, t),
            )
        )
        return False

    def _edge_path(
        self,
        s: Any,
        t: Any,
        pos: tuple,
        depth: int = 0,
        _seen: set | None = None,
        _budget: list | None = None,
    ) -> list | None:
        """Bounded search for a replayable step sequence ``s -> t``.

        Transitions are the rules that actually fired during this run,
        re-matched directly against the real term ``s`` — each emitted
        step is a genuine rule instance, so any path found is a valid
        derivation.  Returns the list of steps (all located at ``pos``),
        or ``None`` when no short path exists.
        """
        if op_repr(s) == op_repr(t):
            return []
        if _seen is None:
            _seen, _budget = set(), [512]
        if depth > 16 or _budget[0] <= 0:
            return None
        _seen.add(op_repr(s))
        for rule in self._rule_objs.values():
            m = _term_match(rule.lhs, s)
            if m is None:
                continue
            if rule.check is not None and not rule.check(m):
                continue
            inst = dict(m)
            if rule.derive is not None:
                extra = rule.derive(m)
                if extra is None:
                    continue
                inst.update(extra)
            r = _term_instantiate(rule.rhs, inst)
            if op_repr(r) in _seen:
                continue
            _budget[0] -= 1
            rest = self._edge_path(r, t, pos, depth + 1, _seen, _budget)
            if rest is not None:
                return [CertStep(rule.name, pos, s, r, inst), *rest]
        return None

    def _explain_gap(self, s: Any, t: Any) -> str:
        """Why a ``connect`` gap is e-graph-dependent, for the cert note."""
        _eid, en_t = self._locate(t)
        if en_t is not None:
            org = self._enode_origin.get(en_t)
            if org == "external":
                return (
                    "target enode introduced outside rule "
                    "application (non-local pass such as "
                    "pair_shared_input_*, or a manual union)"
                )
            if org == "input":
                return (
                    "both members predate saturation but no fired "
                    "rule links them at this position"
                )
        return "no replayable derivation found"

    def _resolve_dst(
        self,
        src_term: Any,
        dst_term: Any,
        root_eid: int | None,
        cost_fn,
    ) -> tuple[int | None, Any]:
        """Shared endpoint resolution for certificates and coherence."""
        if root_eid is None:
            root_eid = self._class_of_term(src_term)
        if dst_term is None:
            if root_eid is None:
                raise ValueError("src_term is not in this e-graph")
            if cost_fn is None:
                from catopt.cost import count_cost

                cost_fn = count_cost
            dst_term = self.extract_best(root_eid, cost_fn)
        return root_eid, dst_term

    def certificate(
        self,
        src_term: Any,
        dst_term: Any = None,
        *,
        root_eid: int | None = None,
        cost_fn=None,
    ) -> Certificate:
        """Build a proof-carrying derivation ``src_term`` -> ``dst_term``.

        ``src_term`` is the term originally added to the e-graph (the
        certificate's anchor — *not* an arbitrary class member).
        ``dst_term`` defaults to ``extract_best`` under ``cost_fn``
        (``count_cost`` if neither is given).  The certificate's steps
        replay positionally on real terms; ``egraph_dependent`` steps
        mark where the e-graph witnessed an equality that has no
        standalone rule derivation.

        At ``truncation_level == 1`` no proof witnesses were recorded,
        so the certificate degrades to a proof-free marker: a single
        ``egraph_dependent`` step asserting ``src == dst`` (or zero
        steps when the terms are already identical).  It replays under
        :func:`verify_certificate` as a trusted assertion and is
        rejected under ``strict=True``.
        """
        root_eid, dst_term = self._resolve_dst(
            src_term, dst_term, root_eid, cost_fn
        )
        if not self._track:
            if op_repr(src_term) == op_repr(dst_term):
                steps0: list = []
            else:
                steps0 = [
                    CertStep(
                        "<truncated>",
                        (),
                        src_term,
                        dst_term,
                        {},
                        egraph_dependent=True,
                        note="truncation level 1: proof witnesses "
                        "were not recorded",
                    )
                ]
            return Certificate(
                src=src_term,
                dst=dst_term,
                root_eid=root_eid,
                steps=steps0,
                rules={},
                stats={
                    "proof_free": True,
                    "truncation_level": self.truncation_level,
                    "n_steps": len(steps0),
                    "n_egraph_dependent": len(steps0),
                    "rules_used": [],
                    "n_proof_edges": 0,
                    "n_rule_applications": 0,
                },
            )
        steps: list = []
        self._connect(src_term, dst_term, (), steps, 0)
        used = sorted({s.rule for s in steps if not s.egraph_dependent})
        cert = Certificate(
            src=src_term,
            dst=dst_term,
            root_eid=root_eid,
            steps=steps,
            rules={
                n: self._rule_objs[n]
                for n in used
                if n in self._rule_objs
            },
            stats={
                "n_steps": len(steps),
                "n_egraph_dependent": sum(
                    1 for s in steps if s.egraph_dependent
                ),
                "rules_used": used,
                "n_proof_edges": len(self._merge_log),
                "n_rule_applications": len(self._applications),
            },
        )
        return cert

    # -- level 3: lazily-materialised coherences --------------------------
    #
    #  The level-2 merge log is a *forest*: ``union`` only records an
    #  edge between two previously-disconnected classes, so exactly one
    #  class-level witness path exists between any two members.  The
    #  alternate proofs that level 3 cares about live in the
    #  term-rewriting space — different orders/locations of rule
    #  application connecting the same endpoints.  ``all_proofs``
    #  enumerates them on demand by re-firing the recorded rules on
    #  real terms (the same machinery ``_edge_path`` uses), with a fuel
    #  cap and a per-path loop check.  Nothing is stored: coherence is
    #  computed when asked, then thrown away.

    def all_proofs(
        self,
        src_term: Any,
        dst_term: Any = None,
        *,
        root_eid: int | None = None,
        cost_fn=None,
        max_paths: int = 32,
        max_steps: int = 8,
        fuel: int = 8192,
    ) -> list:
        """Enumerate distinct derivations ``src_term`` -> ``dst_term``.

        Bounded BFS over the term-rewriting space generated by the rules
        that actually fired during this run (``self._rule_objs``): each
        step is a genuine rule instance — ``_term_match`` on the LHS,
        ``check``/``derive`` honoured, RHS instantiated — located by a
        child-index ``path``, exactly like a :class:`CertStep`.  So
        every returned derivation is a standalone-replayable proof.

        ``max_paths`` caps the number of derivations returned,
        ``max_steps`` caps derivation length, and ``fuel`` bounds total
        match attempts.  Within one derivation a term is never
        revisited (a loop proves nothing new).  Two derivations are
        *distinct* when their ``(rule, path)`` step signatures differ.

        Returns a list of derivations, each a list of :class:`CertStep`.
        ``[[]]`` — one empty derivation — when ``src == dst``.
        Raises ``RuntimeError`` at truncation level 1, where no rule
        provenance exists to enumerate over.
        """
        if not self._track:
            raise RuntimeError(
                "all_proofs requires truncation_level >= 2: level 1 "
                "records no proof witnesses to enumerate over"
            )
        root_eid, dst_term = self._resolve_dst(
            src_term, dst_term, root_eid, cost_fn
        )
        dst_repr = op_repr(dst_term)
        if op_repr(src_term) == dst_repr:
            return [[]]

        rules = list(self._rule_objs.values())
        # frontier entries: (term, steps_so_far, seen_term_reprs)
        frontier: list[tuple[Any, list, frozenset]] = [
            (src_term, [], frozenset({op_repr(src_term)}))
        ]
        paths: list[list] = []
        sigs: set = set()
        while frontier and fuel > 0 and len(paths) < max_paths:
            nxt: list[tuple[Any, list, frozenset]] = []
            for term, steps, seen in frontier:
                if fuel <= 0 or len(paths) >= max_paths:
                    break
                if len(steps) >= max_steps:
                    continue
                for path in _term_paths(term):
                    if fuel <= 0:
                        break
                    sub = _subterm(term, path)
                    for rule in rules:
                        fuel -= 1
                        if fuel < 0:
                            break
                        m = _term_match(rule.lhs, sub)
                        if m is None:
                            continue
                        if rule.check is not None and not rule.check(m):
                            continue
                        inst = dict(m)
                        if rule.derive is not None:
                            extra = rule.derive(m)
                            if extra is None:
                                continue
                            inst.update(extra)
                        r = _term_instantiate(rule.rhs, inst)
                        new_term = _replace_subterm(term, path, r)
                        new_repr = op_repr(new_term)
                        nsteps = [
                            *steps,
                            CertStep(rule.name, path, sub, r, inst),
                        ]
                        if new_repr == dst_repr:
                            sig = tuple(
                                (s.rule, s.path) for s in nsteps
                            )
                            if sig not in sigs:  # pragma: no branch — full (rule,path) histories can't collide
                                sigs.add(sig)
                                paths.append(nsteps)
                        elif new_repr not in seen:
                            nxt.append(
                                (new_term, nsteps, seen | {new_repr})
                            )
            frontier = nxt
        return paths

    def coherent_paths(
        self,
        src_term: Any,
        dst_term: Any = None,
        *,
        root_eid: int | None = None,
        cost_fn=None,
        max_paths: int = 32,
        max_steps: int = 8,
        fuel: int = 8192,
    ) -> dict:
        """Coherence summary between two terms: how many distinct ways
        does the rewrite space prove them equal?

        Thin wrapper over :meth:`all_proofs` (which requires
        ``truncation_level >= 2``).  Returns a dict with the resolved
        endpoints, ``same_eclass`` (whether the e-graph itself judges
        the terms equal — independent evidence for the derivations),
        ``n_paths``, the ``paths`` themselves, and ``truncated`` —
        True when the ``max_paths`` cap was hit, i.e. more coherences
        may exist than reported.
        """
        root_eid, dst_term = self._resolve_dst(
            src_term, dst_term, root_eid, cost_fn
        )
        paths = self.all_proofs(
            src_term,
            dst_term,
            root_eid=root_eid,
            max_paths=max_paths,
            max_steps=max_steps,
            fuel=fuel,
        )
        cs = self._class_of_term(src_term)
        cd = self._class_of_term(dst_term)
        same = (
            cs is not None
            and cd is not None
            and self.find(cs) == self.find(cd)
        )
        return {
            "src": src_term,
            "dst": dst_term,
            "root_eid": root_eid,
            "same_eclass": same,
            "n_paths": len(paths),
            "paths": paths,
            "truncated": len(paths) >= max_paths,
        }


