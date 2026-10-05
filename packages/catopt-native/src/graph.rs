//! The native search core: union-find, enode storage, compiled
//! matching, rebuild/congruence and the saturation loop.
//!
//! This is a faithful port of ``catopt_core.egraph.core.EGraph``'s
//! *search* surface — no proof machinery (``_track`` / merge_log /
//! applications / certificates): those stay in Python.  Matching
//! mirrors the bounded ``_m_bounded`` semantics exactly (the unbounded
//! streaming matcher enumerates the same substitution set); class
//! member order is the deterministic BTreeSet order rather than Python
//! set-hash order, which changes enumeration ORDER but not the
//! substitution SET — the fixed point, and therefore extraction, is
//! identical.

use crate::attr::AttrVal;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::rc::Rc;

pub type ClassId = u32;

/// Per-class match-enumeration cap under an enode budget — mirrors
/// ``EGraph._MATCH_CAP``.
const MATCH_CAP: u64 = 256;

// ---------------------------------------------------------------------------
//  E-nodes / e-classes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ENode {
    pub op: String,
    pub children: Vec<ClassId>,
    /// Sorted ``(key, value)`` pairs — same canonical form as Python's
    /// ``_pattern_attrs`` output.
    pub attrs: Vec<(String, AttrVal)>,
}

#[derive(Default)]
pub struct EClass {
    pub nodes: BTreeSet<ENode>,
    /// Lazily-built ``op -> members`` index (invalidated on mutation).
    /// Rc-shared: ``nodes_of`` hands out the list without cloning the
    /// member enodes per call.
    pub by_op: Option<HashMap<String, Rc<Vec<ENode>>>>,
    /// Positive ``_min_term`` resolutions — mirrors
    /// ``eclass.cache["min_term"]``; cleared when the class merges.
    pub min_term: Option<Rc<SerTerm>>,
}

// ---------------------------------------------------------------------------
//  Patterns (compiled once per rule, mirroring egraph/match.py)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum PAttr {
    Lit(AttrVal),
    Mvar(String),
}

#[derive(Clone, Debug)]
pub enum PNode {
    Var(String),
    Leaf(String),
    Op {
        op: String,
        children: Vec<PNode>,
        attrs: Vec<(String, PAttr)>,
        keyset: HashSet<String>,
    },
}

#[derive(Clone)]
pub struct Prog {
    pub root: PNode,
    pub head_op: Option<String>,
    pub head_arity: usize,
    pub leaf_key: Option<String>,
    pub child_reqs: Vec<(usize, String)>,
}

pub fn compile_pattern(root: PNode) -> Prog {
    match &root {
        PNode::Op {
            op,
            children,
            ..
        } => {
            let reqs = children
                .iter()
                .enumerate()
                .filter_map(|(i, c)| match c {
                    PNode::Op { op, .. } => Some((i, op.clone())),
                    _ => None,
                })
                .collect();
            Prog {
                head_op: Some(op.clone()),
                head_arity: children.len(),
                leaf_key: None,
                child_reqs: reqs,
                root,
            }
        }
        PNode::Var(_) => Prog {
            root,
            head_op: None,
            head_arity: usize::MAX,
            leaf_key: None,
            child_reqs: Vec::new(),
        },
        PNode::Leaf(k) => Prog {
            leaf_key: Some(k.clone()),
            root,
            head_op: None,
            head_arity: usize::MAX,
            child_reqs: Vec::new(),
        },
    }
}

// ---------------------------------------------------------------------------
//  Rules / substitutions / serialized terms
// ---------------------------------------------------------------------------

pub struct RRule {
    pub name: String,
    pub lhs: Prog,
    pub rhs: Prog,
    pub check: Option<Py<PyAny>>,
    pub derive: Option<Py<PyAny>>,
}

#[derive(Clone)]
pub enum SVal {
    Cls(ClassId),
    Attr(AttrVal),
}

pub type Subst = HashMap<String, SVal>;

/// A term materialized Rust-side (a bound metavariable's resolved
/// representative) — converted to a Python ``Op``/leaf through the
/// injected ``build_term`` callback.
#[derive(Clone, Debug)]
pub enum SerTerm {
    Leaf(String),
    Op {
        op: String,
        args: Vec<SerTerm>,
        attrs: Vec<(String, AttrVal)>,
    },
}

fn py_str(py: Python<'_>, s: &str) -> Py<PyAny> {
    use pyo3::types::PyString;

    PyString::new(py, s).unbind().into()
}

impl SerTerm {
    /// Nested-tuple form the Python ``build_term`` callback decodes:
    /// ``("leaf", key)`` | ``("op", name, children, attrs)``.
    pub fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            SerTerm::Leaf(key) => Ok(PyTuple::new(
                py,
                vec![py_str(py, "leaf"), py_str(py, key)],
            )?
            .unbind()
            .into()),
            SerTerm::Op { op, args, attrs } => {
                let kids: Vec<Py<PyAny>> = args
                    .iter()
                    .map(|a| a.to_py(py))
                    .collect::<PyResult<_>>()?;
                let attr_pairs: Vec<Py<PyAny>> = attrs
                    .iter()
                    .map(|(k, v)| -> PyResult<Py<PyAny>> {
                        Ok(PyTuple::new(
                            py,
                            vec![py_str(py, k), v.to_py(py)?],
                        )?
                        .unbind()
                        .into())
                    })
                    .collect::<PyResult<_>>()?;
                Ok(PyTuple::new(
                    py,
                    vec![
                        py_str(py, "op"),
                        py_str(py, op),
                        PyTuple::new(py, kids)?.unbind().into(),
                        PyTuple::new(py, attr_pairs)?.unbind().into(),
                    ],
                )?
                .unbind()
                .into())
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  The engine
// ---------------------------------------------------------------------------

pub struct NativeCore {
    parent: Vec<ClassId>,
    rank: Vec<u32>,
    /// Canonical id -> class.  BTreeMap keeps the deterministic
    /// ascending-id iteration Python gets from dict insertion order.
    classes: BTreeMap<ClassId, EClass>,
    node_to_class: HashMap<ENode, ClassId>,
    next_id: ClassId,
    pub rule_fires: HashMap<String, u64>,
    rules: Vec<RRule>,
    rule_index: HashMap<String, usize>,
    dirty: BTreeSet<ClassId>,
    parents: HashMap<ClassId, HashSet<ClassId>>,
    op_classes: HashMap<String, HashSet<ClassId>>,
    opk: HashMap<(String, usize), HashSet<ClassId>>,
    applied_rules: HashSet<String>,
    budget_spent: HashMap<String, u64>,
    /// Python callable ``fn(ser_term) -> term`` used to materialise
    /// bound metavariables for ``check``/``derive``.
    build_term: Py<PyAny>,
}

impl NativeCore {
    pub fn new(build_term: Py<PyAny>) -> Self {
        NativeCore {
            parent: Vec::new(),
            rank: Vec::new(),
            classes: BTreeMap::new(),
            node_to_class: HashMap::new(),
            next_id: 0,
            rule_fires: HashMap::new(),
            rules: Vec::new(),
            rule_index: HashMap::new(),
            dirty: BTreeSet::new(),
            parents: HashMap::new(),
            op_classes: HashMap::new(),
            opk: HashMap::new(),
            applied_rules: HashSet::new(),
            budget_spent: HashMap::new(),
            build_term,
        }
    }

    pub fn n_enodes(&self) -> usize {
        self.node_to_class.len()
    }

    pub fn n_classes(&self) -> usize {
        self.classes.len()
    }

    // -- union-find ---------------------------------------------------

    pub fn find(&mut self, mut x: ClassId) -> ClassId {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp; // path halving
            x = gp;
        }
        x
    }

    // -- enode/class construction --------------------------------------

    fn add_enode(&mut self, enode: ENode) -> ClassId {
        let eid = self.next_id;
        self.next_id += 1;
        self.parent.push(eid);
        self.rank.push(0);
        let mut ec = EClass::default();
        ec.nodes.insert(enode.clone());
        self.classes.insert(eid, ec);
        self.node_to_class.insert(enode.clone(), eid);
        self.dirty.insert(eid);
        self.op_classes
            .entry(enode.op.clone())
            .or_default()
            .insert(eid);
        self.opk
            .entry((enode.op.clone(), enode.children.len()))
            .or_default()
            .insert(eid);
        for c in &enode.children {
            let cc = self.find(*c);
            self.parents.entry(cc).or_default().insert(eid);
        }
        eid
    }

    pub fn add_leaf(&mut self, key: &str) -> ClassId {
        let enode = ENode {
            op: "leaf".to_string(),
            children: Vec::new(),
            attrs: vec![("key".to_string(), AttrVal::Str(key.to_string()))],
        };
        if let Some(&c) = self.node_to_class.get(&enode) {
            return self.find(c);
        }
        self.add_enode(enode)
    }

    /// Add an enode with already-resolved child class ids (mirrors
    /// ``EGraph.add_enode``).
    pub fn add_op(
        &mut self,
        op: &str,
        children: Vec<ClassId>,
        attrs: Vec<(String, AttrVal)>,
    ) -> ClassId {
        let canon: Vec<ClassId> =
            children.iter().map(|&c| self.find(c)).collect();
        let enode = ENode {
            op: op.to_string(),
            children: canon,
            attrs,
        };
        if let Some(&c) = self.node_to_class.get(&enode) {
            return self.find(c);
        }
        self.add_enode(enode)
    }

    // -- union ----------------------------------------------------------

    /// ``EGraph.union`` minus the proof bookkeeping — same merge +
    /// dirty/parents/index bookkeeping.
    pub fn union(&mut self, a: ClassId, b: ClassId) -> bool {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return false;
        }
        // Union by rank.
        let (hi, lo) = if self.rank[ra as usize] < self.rank[rb as usize] {
            (rb, ra)
        } else {
            (ra, rb)
        };
        self.parent[lo as usize] = hi;
        if self.rank[hi as usize] == self.rank[lo as usize] {
            self.rank[hi as usize] += 1;
        }
        let new_canon = self.find(hi);
        let old_canon = if new_canon == rb { ra } else { rb };
        // Move members.
        let mut source = self.classes.remove(&old_canon).unwrap();
        let moved: Vec<ENode> =
            source.nodes.iter().cloned().collect();
        let target = self.classes.get_mut(&new_canon).unwrap();
        target.nodes.append(&mut source.nodes);
        target.by_op = None;
        target.min_term = None; // cache cleared on merge (Python: cache.clear())
        // -- incremental bookkeeping (mirrors EGraph.union) --
        self.dirty.insert(new_canon);
        if let Some(p_old) = self.parents.remove(&old_canon) {
            let ent = self.parents.entry(new_canon).or_default();
            for p in &p_old {
                ent.insert(*p);
            }
        }
        for n in &moved {
            if let Some(oc) = self.op_classes.get_mut(&n.op) {
                oc.remove(&old_canon);
                oc.insert(new_canon);
            }
            if let Some(ok) =
                self.opk.get_mut(&(n.op.clone(), n.children.len()))
            {
                ok.remove(&old_canon);
                ok.insert(new_canon);
            }
        }
        // Dirty the upward closure over child->parent edges.
        let mut stack = vec![new_canon];
        while let Some(c) = stack.pop() {
            let ps: Vec<ClassId> = self
                .parents
                .get(&c)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            for p0 in ps {
                let p = self.find(p0);
                if p == c || self.dirty.contains(&p) {
                    continue;
                }
                self.dirty.insert(p);
                stack.push(p);
            }
        }
        true
    }

    // -- member indexing -------------------------------------------------

    /// ``_nodes_of`` — the class's members carrying *op* (lazy index,
    /// Rc-shared so repeat calls don't re-copy the member list).
    fn nodes_of(&mut self, eid: ClassId, op: &str) -> Rc<Vec<ENode>> {
        let ec = self.classes.get_mut(&eid).unwrap();
        if let Some(lst) =
            ec.by_op.as_ref().and_then(|bo| bo.get(op))
        {
            return lst.clone();
        }
        let lst: Rc<Vec<ENode>> = Rc::new(
            ec.nodes
                .iter()
                .filter(|n| n.op == op)
                .cloned()
                .collect(),
        );
        ec.by_op
            .get_or_insert_with(HashMap::new)
            .insert(op.to_string(), lst.clone());
        lst
    }

    // -- matching (the _m_bounded semantics) -----------------------------

    /// Port of ``EGraph._m_bounded``: enumerate substitutions matching
    /// *pn* at *eid* into *results*, capped at *limit* — including the
    /// cap quirk that an e-node whose enumeration crosses the limit
    /// contributes nothing.
    fn m_bounded(
        &mut self,
        pn: &PNode,
        eid: ClassId,
        subst: &Subst,
        results: &mut Vec<Subst>,
        limit: usize,
    ) {
        if results.len() >= limit {
            return;
        }
        let eid = self.find(eid);
        match pn {
            PNode::Var(name) => {
                if let Some(SVal::Cls(prev)) = subst.get(name) {
                    if *prev == eid {
                        results.push(subst.clone());
                    }
                    return;
                }
                let mut s2 = subst.clone();
                s2.insert(name.clone(), SVal::Cls(eid));
                results.push(s2);
            }
            PNode::Op {
                op,
                children,
                attrs,
                keyset,
            } => {
                let members = self.nodes_of(eid, op);
                for node in members.iter() {
                    if node.children.len() != children.len() {
                        continue;
                    }
                    let node_keyset: HashSet<&String> =
                        node.attrs.iter().map(|(k, _)| k).collect();
                    if node_keyset.len() != keyset.len()
                        || !node_keyset
                            .iter()
                            .all(|k| keyset.contains(*k))
                    {
                        continue;
                    }
                    // Attribute matching — literal values equal; a
                    // metavar binds under "$attr:<name>" consistently.
                    let mut attr_substs: Vec<Subst> = vec![subst.clone()];
                    let mut attr_ok = true;
                    for (k, pv) in attrs {
                        let nv = &node
                            .attrs
                            .iter()
                            .find(|(nk, _)| nk == k)
                            .unwrap()
                            .1;
                        match pv {
                            PAttr::Mvar(mv) => {
                                let key = format!("$attr:{}", mv);
                                let mut nxt: Vec<Subst> = Vec::new();
                                for cs in &attr_substs {
                                    match cs.get(&key) {
                                        // lenient: the matcher accepts
                                        // 0 for 0.0 — strict eq would
                                        // break consistent-binding
                                        Some(SVal::Attr(av))
                                            if av.loose_eq(nv) =>
                                        {
                                            nxt.push(cs.clone())
                                        }
                                        Some(SVal::Attr(_)) => {}
                                        Some(SVal::Cls(_)) => {}
                                        None => {
                                            let mut cc = cs.clone();
                                            cc.insert(
                                                key.clone(),
                                                SVal::Attr(nv.clone()),
                                            );
                                            nxt.push(cc);
                                        }
                                    }
                                }
                                attr_substs = nxt;
                            }
                            PAttr::Lit(lit) => {
                                // lenient match — literal 0 accepts
                                // attr 0.0 (Python parity)
                                if !nv.loose_eq(lit) {
                                    attr_ok = false;
                                    break;
                                }
                            }
                        }
                        if attr_substs.is_empty() {
                            attr_ok = false;
                            break;
                        }
                    }
                    if !attr_ok {
                        continue;
                    }
                    // Positional matching with shared-metavar
                    // consistency (the incoming substs thread through).
                    let mut child_substs = attr_substs;
                    let mut ok = true;
                    for (i, cp) in children.iter().enumerate() {
                        let mut new_substs: Vec<Subst> = Vec::new();
                        for cs in &child_substs {
                            if results.len() + new_substs.len() >= limit {
                                ok = false;
                                break;
                            }
                            let mut child_results: Vec<Subst> = Vec::new();
                            self.m_bounded(
                                cp,
                                node.children[i],
                                cs,
                                &mut child_results,
                                limit,
                            );
                            new_substs.extend(child_results);
                        }
                        if new_substs.is_empty() {
                            ok = false;
                            break;
                        }
                        child_substs = new_substs;
                    }
                    if ok {
                        let room = limit - results.len();
                        results
                            .extend(child_substs.into_iter().take(room));
                        if results.len() >= limit {
                            return;
                        }
                    }
                }
            }
            PNode::Leaf(key) => {
                let enode = ENode {
                    op: "leaf".to_string(),
                    children: Vec::new(),
                    attrs: vec![(
                        "key".to_string(),
                        AttrVal::Str(key.clone()),
                    )],
                };
                if let Some(&nid) = self.node_to_class.get(&enode) {
                    if self.find(nid) == eid {
                        results.push(subst.clone());
                    }
                }
            }
        }
    }

    // -- instantiation -----------------------------------------------------

    /// ``_instantiate`` — realise the RHS pattern under *subst*.
    fn instantiate(&mut self, pn: &PNode, subst: &Subst) -> ClassId {
        match pn {
            PNode::Var(name) => match subst.get(name) {
                Some(SVal::Cls(c)) => *c,
                _ => panic!("unbound metavar {name} in RHS"),
            },
            PNode::Op {
                op,
                children,
                attrs,
                ..
            } => {
                let child_eids: Vec<ClassId> = children
                    .iter()
                    .map(|a| self.instantiate(a, subst))
                    .collect();
                let mut attr_map: Vec<(String, AttrVal)> = Vec::new();
                for (k, v) in attrs {
                    match v {
                        PAttr::Mvar(mv) => {
                            let key = format!("$attr:{}", mv);
                            match subst.get(&key) {
                                Some(SVal::Attr(av)) => {
                                    attr_map.push((k.clone(), av.clone()))
                                }
                                // Python parity: unbound attr metavars
                                // instantiate as the metavar string.
                                _ => attr_map.push((
                                    k.clone(),
                                    AttrVal::Str(mv.clone()),
                                )),
                            }
                        }
                        PAttr::Lit(v) => attr_map.push((k.clone(), v.clone())),
                    }
                }
                attr_map.sort_by(|a, b| a.0.cmp(&b.0));
                let canon: Vec<ClassId> =
                    child_eids.iter().map(|&c| self.find(c)).collect();
                let enode = ENode {
                    op: op.clone(),
                    children: canon,
                    attrs: attr_map,
                };
                if let Some(&c) = self.node_to_class.get(&enode) {
                    self.find(c)
                } else {
                    self.add_enode(enode)
                }
            }
            PNode::Leaf(key) => self.add_leaf(key),
        }
    }

    // -- min_term (member resolution for check/derive) ---------------------

    /// ``_any_term_cached``: the class's minimum-size acyclic member,
    /// cached per-class (cleared on merge) and per apply_rule call.
    fn any_term_cached(
        &mut self,
        eid: ClassId,
        memo: &mut HashMap<ClassId, (Rc<SerTerm>, usize)>,
    ) -> Option<Rc<SerTerm>> {
        let eid = self.find(eid);
        if let Some(t) = &self.classes[&eid].min_term {
            return Some(t.clone());
        }
        let mut seen: HashSet<ClassId> = HashSet::new();
        let (t, _sz) = self.min_term(eid, memo, &mut seen);
        if let Some(ref tt) = t {
            self.classes.get_mut(&eid).unwrap().min_term = Some(tt.clone());
        }
        t
    }

    /// Port of ``EGraph._min_term`` — ``(term, op-count)`` or None.
    /// Leaf members win immediately (size 1); only positive results
    /// are memoised.
    fn min_term(
        &mut self,
        eid: ClassId,
        memo: &mut HashMap<ClassId, (Rc<SerTerm>, usize)>,
        seen: &mut HashSet<ClassId>,
    ) -> (Option<Rc<SerTerm>>, usize) {
        let eid = self.find(eid);
        let ec = &self.classes[&eid];
        for node in &ec.nodes {
            if node.op == "leaf" {
                let key = match &node.attrs[0].1 {
                    AttrVal::Str(s) => s.clone(),
                    _ => "??".to_string(),
                };
                let res = Rc::new(SerTerm::Leaf(key));
                memo.insert(eid, (res.clone(), 1));
                return (Some(res), 1);
            }
        }
        if let Some(hit) = memo.get(&eid) {
            return (Some(hit.0.clone()), hit.1);
        }
        if seen.contains(&eid) {
            return (None, usize::MAX);
        }
        seen.insert(eid);
        let mut best: (Option<Rc<SerTerm>>, usize) = (None, usize::MAX);
        let nodes: Vec<ENode> = ec.nodes.iter().cloned().collect();
        for node in nodes {
            let mut args: Vec<SerTerm> = Vec::new();
            let mut sz = 1usize;
            let mut ok = true;
            for c in &node.children {
                let canon = self.find(*c);
                if canon == eid || seen.contains(&canon) {
                    ok = false;
                    break;
                }
                let (t, s) = self.min_term(canon, memo, seen);
                match t {
                    Some(tt) => {
                        args.push((*tt).clone());
                        sz = sz.saturating_add(s);
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok && (best.0.is_none() || sz < best.1) {
                best = (
                    Some(Rc::new(SerTerm::Op {
                        op: node.op.clone(),
                        args,
                        attrs: node.attrs.clone(),
                    })),
                    sz,
                );
            }
        }
        seen.remove(&eid);
        if let Some(ref tt) = best.0 {
            memo.insert(eid, (tt.clone(), best.1));
        }
        best
    }

    // -- bound dict (check/derive bridge) ---------------------------------

    /// Build the Python ``bound`` dict for a substitution:
    /// ``{metavar: resolved_term, "$attr:k": attr_value}``.  Returns
    /// ``Ok(None)`` when a metavar's class has no acyclic member (the
    /// Python path skips such substitutions).
    fn build_bound(
        &mut self,
        py: Python<'_>,
        subst: &Subst,
        memo: &mut HashMap<ClassId, (Rc<SerTerm>, usize)>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let d = PyDict::new(py);
        for (k, v) in subst {
            match v {
                SVal::Attr(av) => {
                    d.set_item(k, av.to_py(py)?)?;
                }
                SVal::Cls(cid) => {
                    let t = self.any_term_cached(*cid, memo);
                    match t {
                        Some(tt) => {
                            let ser = tt.to_py(py)?;
                            let term =
                                self.build_term.call1(py, (ser,))?;
                            d.set_item(k, term)?;
                        }
                        None => return Ok(None),
                    }
                }
            }
        }
        Ok(Some(d.unbind().into()))
    }

    // -- candidate classes --------------------------------------------------

    fn head_ok(&mut self, prog: &Prog, eid: ClassId) -> bool {
        let head_op = prog.head_op.clone().unwrap();
        let reqs = prog.child_reqs.clone();
        let members = self.nodes_of(eid, &head_op);
        for node in members.iter() {
            if node.children.len() != prog.head_arity {
                continue;
            }
            let mut all = true;
            for (i, rop) in &reqs {
                let cc = self.find(node.children[*i]);
                match self.op_classes.get(rop) {
                    Some(s) if s.contains(&cc) => {}
                    _ => {
                        all = false;
                        break;
                    }
                }
            }
            if all {
                return true;
            }
        }
        false
    }

    /// ``_candidate_classes`` — snapshot-based (the search set / class
    /// ids are materialised, each re-found at use time — same as the
    /// Python generator).
    fn candidate_classes(
        &mut self,
        prog: &Prog,
        search: Option<&BTreeSet<ClassId>>,
    ) -> Vec<ClassId> {
        let eligible: Option<HashSet<ClassId>> = if let Some(hop) = &prog.head_op
        {
            let key = (hop.clone(), prog.head_arity);
            let raw: Vec<ClassId> = self
                .opk
                .get(&key)
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            Some(raw.iter().map(|&c| self.find(c)).collect())
        } else if prog.leaf_key.is_none() {
            None
        } else {
            let enode = ENode {
                op: "leaf".to_string(),
                children: Vec::new(),
                attrs: vec![(
                    "key".to_string(),
                    AttrVal::Str(prog.leaf_key.clone().unwrap()),
                )],
            };
            match self.node_to_class.get(&enode) {
                Some(&leid) => {
                    let mut s = HashSet::new();
                    s.insert(self.find(leid));
                    Some(s)
                }
                None => Some(HashSet::new()),
            }
        };
        let ids: Vec<ClassId> = match search {
            None => self.classes.keys().copied().collect(),
            Some(s) => s.iter().copied().collect(),
        };
        let mut out = Vec::new();
        for eid0 in ids {
            let eid = self.find(eid0);
            if let Some(el) = &eligible {
                if !el.contains(&eid) {
                    continue;
                }
            }
            if !prog.child_reqs.is_empty() && !self.head_ok(prog, eid) {
                continue;
            }
            out.push(eid);
        }
        out
    }

    // -- rule application ----------------------------------------------------

    /// ``apply_rule`` — match the LHS across (a subset of) classes and
    /// union each match with its instantiated RHS.  ``enode_budget``
    /// bounds NEW enodes created this call.
    pub fn apply_rule(
        &mut self,
        py: Python<'_>,
        ri: usize,
        search: Option<&BTreeSet<ClassId>>,
        enode_budget: Option<u64>,
    ) -> PyResult<bool> {
        if search.is_none() {
            self.applied_rules.insert(self.rules[ri].name.clone());
        }
        let mut changed = false;
        let n_start = self.n_enodes();
        let mut anyterm_memo: HashMap<ClassId, (Rc<SerTerm>, usize)> =
            HashMap::new();
        let candidates =
            self.candidate_classes(&self.rules[ri].lhs.clone(), search);
        for eid in candidates {
            let match_cap: usize = match enode_budget {
                Some(budget) => {
                    let spent = (self.n_enodes() - n_start) as u64;
                    if spent >= budget {
                        break;
                    }
                    std::cmp::min(budget - spent, MATCH_CAP) as usize
                }
                None => usize::MAX,
            };
            let mut substs: Vec<Subst> = Vec::new();
            // Cloned to keep the borrow of self.rules separate from
            // &mut self in the matcher.
            let lhs_root = self.rules[ri].lhs.root.clone();
            self.m_bounded(&lhs_root, eid, &Subst::new(), &mut substs, match_cap);
            for subst in substs {
                let has_check = self.rules[ri].check.is_some();
                let has_derive = self.rules[ri].derive.is_some();
                let mut subst = subst;
                if has_check || has_derive {
                    let bound = self.build_bound(py, &subst, &mut anyterm_memo)?;
                    let Some(bound) = bound else {
                        continue;
                    };
                    if let Some(chk) = &self.rules[ri].check {
                        let ok: bool =
                            chk.call1(py, (&bound,))?.extract(py)?;
                        if !ok {
                            continue;
                        }
                    }
                    if let Some(drv) = &self.rules[ri].derive {
                        let extra = drv.call1(py, (&bound,))?;
                        if extra.is_none(py) {
                            continue;
                        }
                        let d = extra.bind(py).cast::<PyDict>()?;
                        for (k, v) in d.iter() {
                            let key: String = k.extract()?;
                            if key.starts_with("$attr:") {
                                subst.insert(
                                    key.clone(),
                                    SVal::Attr(AttrVal::from_py(&v)?),
                                );
                            } else if let Ok(ci) = v.extract::<i64>() {
                                subst.insert(
                                    key,
                                    SVal::Cls(ci as ClassId),
                                );
                            } else {
                                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                                    "derive returned non-attr key {key:?} with non-class value"
                                )));
                            }
                        }
                    }
                }
                let rhs_root = self.rules[ri].rhs.root.clone();
                let rhs_eid = self.instantiate(&rhs_root, &subst);
                let name = self.rules[ri].name.clone();
                if self.union(eid, rhs_eid) {
                    changed = true;
                    *self.rule_fires.entry(name).or_insert(0) += 1;
                }
            }
        }
        Ok(changed)
    }

    // -- rebuild / congruence -------------------------------------------------

    fn cong_canon(&mut self, node: &ENode) -> ENode {
        let canon: Vec<ClassId> =
            node.children.iter().map(|&c| self.find(c)).collect();
        if canon == node.children {
            node.clone()
        } else {
            ENode {
                op: node.op.clone(),
                children: canon,
                attrs: node.attrs.clone(),
            }
        }
    }

    /// ``_canonicalise`` over the given class ids (one pass).
    fn canonicalise(&mut self, ids: &[ClassId]) -> bool {
        let mut changed = false;
        let mut seen: HashSet<ClassId> = HashSet::new();
        for &eid0 in ids {
            let eid = self.find(eid0);
            if !seen.insert(eid) || !self.classes.contains_key(&eid) {
                continue;
            }
            let ec = self.classes.get_mut(&eid).unwrap();
            let old_nodes: Vec<ENode> = ec.nodes.iter().cloned().collect();
            let mut new_nodes: BTreeSet<ENode> = BTreeSet::new();
            let mut dirty_eclass = false;
            for node in old_nodes {
                if node.children.is_empty() {
                    new_nodes.insert(node);
                    continue;
                }
                let canon: Vec<ClassId> =
                    node.children.iter().map(|&c| self.find(c)).collect();
                if canon == node.children {
                    new_nodes.insert(node);
                    continue;
                }
                changed = true;
                dirty_eclass = true;
                for &c in &canon {
                    let cc = self.find(c);
                    self.parents.entry(cc).or_default().insert(eid);
                }
                let nn = ENode {
                    op: node.op.clone(),
                    children: canon,
                    attrs: node.attrs.clone(),
                };
                // Mirrors _inherit_provenance's hash-cons (level>=2).
                self.node_to_class.entry(nn.clone()).or_insert(eid);
                new_nodes.insert(nn);
            }
            if dirty_eclass {
                let ec = self.classes.get_mut(&eid).unwrap();
                ec.nodes = new_nodes;
                ec.by_op = None;
            }
        }
        changed
    }

    /// ``_close_congruence`` — the same fixed point computed with a
    /// per-round owner scan (the incremental worklist and the whole-
    /// graph rescan converge identically).
    fn close_congruence(&mut self) -> bool {
        let mut changed = false;
        loop {
            // Snapshot every (class, enode) — matches mutate classes.
            let pend: Vec<(ClassId, ENode)> = self
                .classes
                .iter()
                .flat_map(|(&eid, c)| {
                    c.nodes.iter().cloned().map(move |n| (eid, n))
                })
                .collect();
            let mut owner: HashMap<ENode, ClassId> = HashMap::new();
            let mut merges: Vec<(ClassId, ClassId)> = Vec::new();
            for (eid0, node) in pend {
                let eid = self.find(eid0);
                if !self.classes.contains_key(&eid) {
                    continue;
                }
                let nn = self.cong_canon(&node);
                let node = if nn != node {
                    // Re-keyed out of the class — apply pointwise.
                    let ec = self.classes.get_mut(&eid).unwrap();
                    if ec.nodes.remove(&node) {
                        ec.nodes.insert(nn.clone());
                        ec.by_op = None;
                        for &c in &nn.children {
                            let cc = self.find(c);
                            self.parents
                                .entry(cc)
                                .or_default()
                                .insert(eid);
                        }
                        self.node_to_class
                            .entry(nn.clone())
                            .or_insert(eid);
                    }
                    if !self.classes[&eid].nodes.contains(&nn) {
                        continue;
                    }
                    nn
                } else {
                    if !self.classes[&eid].nodes.contains(&node) {
                        continue; // stale worklist item
                    }
                    node
                };
                match owner.get(&node) {
                    Some(&prev) => {
                        if self.find(prev) != eid {
                            merges.push((prev, eid));
                        }
                    }
                    None => {
                        owner.insert(node, eid);
                    }
                }
            }
            if merges.is_empty() {
                return changed;
            }
            changed = true;
            for (a, b) in merges {
                self.union(a, b);
            }
        }
    }

    /// ``rebuild`` — ``classes=None`` canonicalises everything and
    /// closes congruence; a set restricts to a one-pass canonicalise.
    pub fn rebuild(&mut self, classes: Option<Vec<ClassId>>) -> bool {
        match classes {
            Some(ids) => self.canonicalise(&ids),
            None => {
                let ids: Vec<ClassId> =
                    self.classes.keys().copied().collect();
                let changed = self.canonicalise(&ids);
                self.close_congruence() || changed
            }
        }
    }

    // -- rules ----------------------------------------------------------------

    /// Register (or replace) a compiled rule.  Returns its index.
    pub fn add_rule(
        &mut self,
        name: &str,
        lhs: PNode,
        rhs: PNode,
        check: Option<Py<PyAny>>,
        derive: Option<Py<PyAny>>,
    ) -> usize {
        let rule = RRule {
            name: name.to_string(),
            lhs: compile_pattern(lhs),
            rhs: compile_pattern(rhs),
            check,
            derive,
        };
        if let Some(&i) = self.rule_index.get(name) {
            self.rules[i] = rule;
            i
        } else {
            self.rule_index.insert(name.to_string(), self.rules.len());
            self.rules.push(rule);
            self.rules.len() - 1
        }
    }

    // -- saturation -------------------------------------------------------------

    /// One saturation iteration of ``EGraph.run``.
    ///
    /// Snapshots the dirty frontier, applies every scheduled rule
    /// (full scan for rules never applied to this graph, frontier-only
    /// otherwise), rebuilds ``search | dirty``, then classifies the
    /// iteration: ``"max_nodes"`` when the node cap is crossed,
    /// ``"fixed_point"`` when nothing changed, else ``"continue"``.
    /// Splitting the loop this way lets the Python wrapper drive the
    /// ``stop="improving"`` policy itself — the per-iteration
    /// re-extraction runs the reference mixin between calls, so no
    /// reentrant borrow of the extension object is ever needed.
    pub fn run_iteration(
        &mut self,
        py: Python<'_>,
        rule_names: &[String],
        max_nodes: usize,
        rule_budgets: &HashMap<String, u64>,
    ) -> PyResult<&'static str> {
        for name in rule_budgets.keys() {
            self.budget_spent.entry(name.clone()).or_insert(0);
        }
        let n_before = self.n_enodes();
        // Snapshot the frontier once per iteration.
        let dirty_ids: Vec<ClassId> =
            self.dirty.iter().copied().collect();
        let mut search: BTreeSet<ClassId> = BTreeSet::new();
        for e in dirty_ids {
            search.insert(self.find(e));
        }
        self.dirty.clear();
        for name in rule_names {
            let Some(&ri) = self.rule_index.get(name) else {
                continue;
            };
            let budget = rule_budgets.get(name);
            let remaining = budget.map(|b| {
                let spent = *self.budget_spent.get(name).unwrap_or(&0);
                b.saturating_sub(spent)
            });
            if budget.is_some() && remaining == Some(0) {
                continue; // suspended: budget exhausted
            }
            let n0 = self.n_enodes();
            let seen_before = self.applied_rules.contains(name);
            self.apply_rule(
                py,
                ri,
                if seen_before { Some(&search) } else { None },
                remaining,
            )?;
            if budget.is_some() {
                *self.budget_spent.entry(name.clone()).or_insert(0) +=
                    (self.n_enodes() - n0) as u64;
            }
        }
        // rebuild(set(search) | dirty)
        let mut ids: Vec<ClassId> = search.into_iter().collect();
        ids.extend(self.dirty.iter().copied());
        self.rebuild(Some(ids));
        let n_after = self.n_enodes();
        if n_after >= max_nodes {
            return Ok("max_nodes");
        }
        if n_after == n_before && self.dirty.is_empty() {
            return Ok("fixed_point");
        }
        Ok("continue")
    }

    /// The final canonicalise ``EGraph.run`` performs after its loop.
    pub fn finish(&mut self) {
        self.rebuild(None);
    }

    /// Cumulative ``rule -> enodes contributed`` under ``rule_budgets``.
    pub fn budget_spent(&self) -> &HashMap<String, u64> {
        &self.budget_spent
    }

    /// ``EGraph.run`` — the dirty-frontier incremental saturation loop
    /// (``stop="fixed_point"`` policy; the ``"improving"`` policy is
    /// driven from Python over :meth:`run_iteration`).
    pub fn run(
        &mut self,
        py: Python<'_>,
        rule_names: &[String],
        max_iterations: usize,
        max_nodes: usize,
        rule_budgets: &HashMap<String, u64>,
    ) -> PyResult<Py<PyAny>> {
        let mut stop_reason = "max_iterations";
        let mut iterations = 0usize;
        for it in 0..max_iterations {
            iterations = it + 1;
            let r = self.run_iteration(
                py,
                rule_names,
                max_nodes,
                rule_budgets,
            )?;
            if r != "continue" {
                stop_reason = r;
                break;
            }
        }
        // Postcondition: canonicalised graph (unrestricted rebuild).
        self.finish();
        let stats = PyDict::new(py);
        stats.set_item("iterations", iterations)?;
        stats.set_item("n_enodes", self.n_enodes())?;
        stats.set_item("n_classes", self.n_classes())?;
        // Proof-tracking fields kept for stats-shape parity — the
        // native core records none (Python-engine concern).
        stats.set_item("n_proof_edges", 0usize)?;
        stats.set_item("truncation_level", 1usize)?;
        let budgets = PyDict::new(py);
        for name in rule_budgets.keys() {
            budgets.set_item(
                name,
                *self.budget_spent.get(name).unwrap_or(&0),
            )?;
        }
        stats.set_item("rule_budgets", budgets)?;
        let suspended: Vec<&String> = rule_budgets
            .iter()
            .filter(|(n, b)| {
                *self.budget_spent.get(*n).unwrap_or(&0) >= **b
            })
            .map(|(n, _)| n)
            .collect();
        stats.set_item("budget_suspended", suspended)?;
        stats.set_item("stop", stop_reason)?;
        Ok(stats.unbind().into())
    }

    // -- introspection for the Python adapter --------------------------------

    /// ``(canonical_id, [(op, children, [(k, pyval), ...]), ...])`` —
    /// materialises the ``_classes`` view the Python extraction mixin
    /// consumes.
    pub fn dump_classes(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<Vec<(ClassId, Vec<(String, Vec<ClassId>, Vec<(String, Py<PyAny>)>)>)>>
    {
        let ids: Vec<ClassId> = self.classes.keys().copied().collect();
        let mut out = Vec::with_capacity(ids.len());
        for eid in ids {
            let canon = self.find(eid);
            let nodes: Vec<(String, Vec<ClassId>, Vec<(String, Py<PyAny>)>)> =
                self.classes[&canon]
                    .nodes
                    .iter()
                    .map(|n| {
                        let attrs: Vec<(String, Py<PyAny>)> = n
                            .attrs
                            .iter()
                            .map(|(k, v)| {
                                Ok((k.clone(), v.to_py(py)?))
                            })
                            .collect::<PyResult<_>>()?;
                        Ok((n.op.clone(), n.children.clone(), attrs))
                    })
                    .collect::<PyResult<_>>()?;
            out.push((canon, nodes));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
//  Tests — py-free internals plus a GIL-bridged iteration
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;
    use std::sync::Once;

    /// One-shot interpreter init, then a core with a dummy
    /// ``build_term`` hook (unused until check/derive runs).
    fn core() -> NativeCore {
        static INIT: Once = Once::new();
        INIT.call_once(pyo3::Python::initialize);
        Python::attach(|py| NativeCore::new(py.None()))
    }

    fn pvar(name: &str) -> PNode {
        PNode::Var(name.to_string())
    }

    fn pop(op: &str, children: Vec<PNode>) -> PNode {
        PNode::Op {
            op: op.to_string(),
            children,
            attrs: vec![],
            keyset: HashSet::new(),
        }
    }

    #[test]
    fn intern_dedups_and_finds() {
        let mut g = core();
        let a = g.add_leaf("a");
        let b = g.add_leaf("b");
        let n = g.add_op("add", vec![a, b], vec![]);
        assert_eq!(g.find(n), n);
        assert_eq!(g.n_enodes(), 3);
        assert_eq!(g.n_classes(), 3);
        // Structural hash-consing: an identical enode maps to the
        // same class, no new interning.
        let n2 = g.add_op("add", vec![a, b], vec![]);
        assert_eq!(n2, n);
        assert_eq!(g.n_enodes(), 3);
        // Commuted children are a *different* enode.
        let n3 = g.add_op("add", vec![b, a], vec![]);
        assert_ne!(n3, n);
    }

    #[test]
    fn union_updates_indices_and_dirty() {
        let mut g = core();
        let a = g.add_leaf("a");
        let b = g.add_leaf("b");
        let ab = g.add_op("add", vec![a, b], vec![]);
        let ba = g.add_op("add", vec![b, a], vec![]);
        assert!(g.union(ab, ba));
        let canon = g.find(ab);
        assert_eq!(g.find(ba), canon);
        assert!(g.dirty.contains(&canon));
        assert_eq!(g.n_classes(), 3);
        // Both members live in the merged class.
        assert_eq!(g.classes[&canon].nodes.len(), 2);
        // Eligibility index tracks the merge.
        assert!(g.op_classes["add"].contains(&canon));
    }

    #[test]
    fn canonicalise_rekeys_after_merge() {
        let mut g = core();
        let a = g.add_leaf("a");
        let b = g.add_leaf("b");
        let c = g.add_leaf("c");
        let ab = g.add_op("add", vec![a, b], vec![]);
        g.union(b, c);
        g.rebuild(None);
        // ab's child b now canonicalises to c's class — the member is
        // re-keyed, and a fresh identical node hash-conses to it.
        let canon_child = g.find(b);
        let same = g.add_op("add", vec![a, canon_child], vec![]);
        assert_eq!(same, g.find(ab));
    }

    #[test]
    fn congruence_merges_parents() {
        let mut g = core();
        let a = g.add_leaf("a");
        let b = g.add_leaf("b");
        let c = g.add_leaf("c");
        let ab = g.add_op("add", vec![a, b], vec![]);
        let ac = g.add_op("add", vec![a, c], vec![]);
        assert_ne!(g.find(ab), g.find(ac));
        g.union(b, c);
        g.rebuild(None); // canonicalise + close_congruence
        assert_eq!(g.find(ab), g.find(ac));
    }

    #[test]
    fn one_iteration_applies_rule() {
        let mut g = core();
        let a = g.add_leaf("a");
        let b = g.add_leaf("b");
        let c = g.add_leaf("c");
        let ab = g.add_op("add", vec![a, b], vec![]);
        let root = g.add_op("add", vec![ab, c], vec![]);
        // assoc_add: ("add", ("add", $x, $y), $z)
        //          -> ("add", $x, ("add", $y, $z))
        g.add_rule(
            "assoc_add",
            pop(
                "add",
                vec![pop("add", vec![pvar("x"), pvar("y")]), pvar("z")],
            ),
            pop(
                "add",
                vec![pvar("x"), pop("add", vec![pvar("y"), pvar("z")])],
            ),
            None,
            None,
        );
        let names = vec!["assoc_add".to_string()];
        let budgets = HashMap::new();
        Python::attach(|py| {
            let r = g
                .run_iteration(py, &names, usize::MAX, &budgets)
                .unwrap();
            // First pass: the rule fires, work continues.
            assert_eq!(r, "continue");
            // The right-associated member landed in the root class.
            let canon = g.find(root);
            assert!(g.classes[&canon].nodes.len() >= 2);
            // Second iteration reaches the fixed point.
            let r2 = g
                .run_iteration(py, &names, usize::MAX, &budgets)
                .unwrap();
            assert_eq!(r2, "fixed_point");
            assert_eq!(g.rule_fires["assoc_add"], 1);
        });
    }
}
