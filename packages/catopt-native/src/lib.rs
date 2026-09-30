//! PyO3 bindings for the catopt native search core.
//!
//! The exposed surface is deliberately small and *data-shaped*: all
//! terms and patterns cross the boundary as plain tuples, and the only
//! Python objects held are the injected ``build_term`` converter plus
//! each rule's ``check`` / ``derive`` callables.  See
//! ``python/catopt_native/engine.py`` for the adapter that produces
//! these payloads.
//!
//! Serialization formats
//! ---------------------
//! Flat term (``add_term``) — a post-order list of nodes whose
//! children are indices into that list (DAG-flattened, so shared
//! subtrees intern once):
//!
//! * ``("leaf", key)`` — a leaf enode (``ENode("leaf", (), key)``);
//! * ``("op", name, (i0, i1, ...), ((k, v), ...))`` — an op node.
//!
//! Pattern (``add_rule``) — a nested tree, same node grammar where
//! children are sub-trees instead of indices:
//!
//! * ``("var", name)`` — a metavariable leaf;
//! * ``("leaf", key)`` — a concrete leaf (matched by registry key);
//! * ``("op", name, (children...), ((k, v), ...))`` — an op pattern;
//!   a *string* attr value ``v`` is an attribute metavariable
//!   (``"$attr:"`` namespace), exactly as in the Python matcher.

mod attr;
mod graph;

use attr::AttrVal;
use graph::{NativeCore, PAttr, PNode};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};
use std::collections::HashMap;

fn parse_pattr(obj: &Bound<'_, PyAny>) -> PyResult<PAttr> {
    // A string attr value in a pattern is an attribute metavariable —
    // identical to the Python matcher's `isinstance(pv, str)` rule.
    if obj.is_instance_of::<PyString>() {
        return Ok(PAttr::Mvar(obj.extract::<String>()?));
    }
    Ok(PAttr::Lit(AttrVal::from_py(obj)?))
}

fn parse_pnode(obj: &Bound<'_, PyAny>) -> PyResult<PNode> {
    let t = obj.cast::<PyTuple>()?;
    let kind: String = t.get_item(0)?.extract()?;
    match kind.as_str() {
        "var" => Ok(PNode::Var(t.get_item(1)?.extract()?)),
        "leaf" => Ok(PNode::Leaf(t.get_item(1)?.extract()?)),
        "op" => {
            let op: String = t.get_item(1)?.extract()?;
            let children_obj = t.get_item(2)?;
            let children = children_obj.cast::<PyTuple>()?;
            let mut kids = Vec::with_capacity(children.len());
            for c in children.iter() {
                kids.push(parse_pnode(&c)?);
            }
            let attrs_obj = t.get_item(3)?;
            let attrs_t = attrs_obj.cast::<PyTuple>()?;
            let mut attrs = Vec::with_capacity(attrs_t.len());
            let mut keyset = std::collections::HashSet::new();
            for a in attrs_t.iter() {
                let pair = a.cast::<PyTuple>()?;
                let k: String = pair.get_item(0)?.extract()?;
                let v = parse_pattr(&pair.get_item(1)?)?;
                keyset.insert(k.clone());
                attrs.push((k, v));
            }
            Ok(PNode::Op {
                op,
                children: kids,
                attrs,
                keyset,
            })
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown serialized node kind {other:?}"
        ))),
    }
}

/// The native e-graph — search only, no proof machinery.
#[pyclass(name = "NativeEGraph", unsendable)]
struct PyEGraph {
    core: NativeCore,
}

#[pymethods]
impl PyEGraph {
    /// ``build_term(ser) -> term`` materialises a bound metavariable's
    /// serialized representative into a Python term for
    /// ``check``/``derive``.
    #[new]
    fn new(build_term: Py<PyAny>) -> Self {
        PyEGraph {
            core: NativeCore::new(build_term),
        }
    }

    #[getter]
    fn n_enodes(&self) -> usize {
        self.core.n_enodes()
    }

    #[getter]
    fn n_classes(&self) -> usize {
        self.core.n_classes()
    }

    /// ``find(eid)`` — canonical e-class id.
    fn find(&mut self, eid: u32) -> u32 {
        self.core.find(eid)
    }

    /// ``union(a, b) -> bool`` — merge two e-classes.
    fn union(&mut self, a: u32, b: u32) -> bool {
        self.core.union(a, b)
    }

    /// ``add_leaf(key) -> eid``.
    fn add_leaf(&mut self, key: &str) -> u32 {
        self.core.add_leaf(key)
    }

    /// ``add_term(flat) -> eid`` — flat is a post-order node list;
    /// returns the e-class of the LAST node.
    fn add_term(&mut self, flat: &Bound<'_, PyAny>) -> PyResult<u32> {
        let mut eids: Vec<u32> = Vec::new();
        for item in flat.cast::<PyList>()?.iter() {
            let t = item.cast::<PyTuple>()?;
            let kind: String = t.get_item(0)?.extract()?;
            let eid = match kind.as_str() {
                "leaf" => {
                    let key: String = t.get_item(1)?.extract()?;
                    self.core.add_leaf(&key)
                }
                "op" => {
                    let op: String = t.get_item(1)?.extract()?;
                    let kids_obj = t.get_item(2)?;
                    let kids = kids_obj.cast::<PyTuple>()?;
                    let mut child_eids = Vec::with_capacity(kids.len());
                    for c in kids.iter() {
                        let i: usize = c.extract()?;
                        child_eids.push(eids[i]);
                    }
                    let attrs_obj = t.get_item(3)?;
                    let attrs_t = attrs_obj.cast::<PyTuple>()?;
                    let mut attrs = Vec::with_capacity(attrs_t.len());
                    for a in attrs_t.iter() {
                        let pair = a.cast::<PyTuple>()?;
                        let k: String = pair.get_item(0)?.extract()?;
                        let v = AttrVal::from_py(&pair.get_item(1)?)?;
                        attrs.push((k, v));
                    }
                    attrs.sort_by(|a, b| a.0.cmp(&b.0));
                    self.core.add_op(&op, child_eids, attrs)
                }
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        format!("unknown term node kind {other:?}"),
                    ))
                }
            };
            eids.push(eid);
        }
        eids.last().copied().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("empty term node list")
        })
    }

    /// ``add_rule(name, lhs, rhs, check=None, derive=None)`` — compile
    /// and register a rewrite.  ``check``/``derive`` are Python
    /// callables invoked with the resolved ``bound`` dict.
    #[pyo3(signature = (name, lhs, rhs, check=None, derive=None))]
    fn add_rule(
        &mut self,
        name: &str,
        lhs: &Bound<'_, PyAny>,
        rhs: &Bound<'_, PyAny>,
        check: Option<Py<PyAny>>,
        derive: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let lhs = parse_pnode(lhs)?;
        let rhs = parse_pnode(rhs)?;
        self.core.add_rule(name, lhs, rhs, check, derive);
        Ok(())
    }

    /// ``rebuild(classes=None) -> bool`` — canonicalise (+congruence
    /// when unrestricted).
    #[pyo3(signature = (classes=None))]
    fn rebuild(&mut self, classes: Option<Vec<u32>>) -> bool {
        self.core.rebuild(classes)
    }

    /// ``run_iteration(rule_names, max_nodes, rule_budgets) -> str`` —
    /// one saturation iteration; returns ``"continue"`` /
    /// ``"fixed_point"`` / ``"max_nodes"``.  Lets the Python engine
    /// drive the ``stop="improving"`` policy itself (re-extraction is
    /// the Python mixin's job — calling back into this pyclass from
    /// inside ``run`` would trip PyO3's reentrancy guard).
    #[pyo3(signature = (rule_names, max_nodes=100_000, rule_budgets=None))]
    fn run_iteration(
        &mut self,
        py: Python<'_>,
        rule_names: Vec<String>,
        max_nodes: usize,
        rule_budgets: Option<HashMap<String, u64>>,
    ) -> PyResult<&'static str> {
        self.core.run_iteration(
            py,
            &rule_names,
            max_nodes,
            &rule_budgets.unwrap_or_default(),
        )
    }

    /// ``finish()`` — the final canonicalise ``EGraph.run`` performs
    /// after its loop.
    fn finish(&mut self) {
        self.core.finish();
    }

    /// ``budget_spent()`` — ``{rule_name: enodes_contributed}``.
    fn budget_spent(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in self.core.budget_spent() {
            d.set_item(k, v)?;
        }
        Ok(d.unbind())
    }

    /// ``run(rule_names, max_iterations, max_nodes, rule_budgets) ->
    /// dict`` — saturate to the fixed point under the given schedule;
    /// stats mirror ``EGraph.run``.
    #[pyo3(signature = (
        rule_names,
        max_iterations=100,
        max_nodes=100_000,
        rule_budgets=None,
    ))]
    fn run(
        &mut self,
        py: Python<'_>,
        rule_names: Vec<String>,
        max_iterations: usize,
        max_nodes: usize,
        rule_budgets: Option<HashMap<String, u64>>,
    ) -> PyResult<Py<PyAny>> {
        self.core.run(
            py,
            &rule_names,
            max_iterations,
            max_nodes,
            &rule_budgets.unwrap_or_default(),
        )
    }

    /// ``classes()`` — ``[(canonical_id, [(op, children, attrs), ...])]``
    /// for the Python ``_classes`` view consumed by the extraction
    /// mixin and the carrier-upgrade pass.
    fn classes(
        &mut self,
        py: Python<'_>,
    ) -> PyResult<
        Vec<(u32, Vec<(String, Vec<u32>, Vec<(String, Py<PyAny>)>)>)>,
    > {
        self.core.dump_classes(py)
    }

    /// ``rule_fires()`` — ``{rule_name: merge_count}``.
    fn rule_fires(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in &self.core.rule_fires {
            d.set_item(k, v)?;
        }
        Ok(d.unbind())
    }
}

/// ``catopt_native._native`` — the compiled extension module.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<PyEGraph>()?;
    Ok(())
}
