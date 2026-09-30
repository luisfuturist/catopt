//! Attribute values — the hashable payloads of enode attrs and
//! attribute-metavariable bindings.
//!
//! Mirrors the Python semantics of ``catopt_core.egraph.types``:
//! attribute values are ints, floats, bools, strings, tuples or None.
//! Equality follows Python (``1 == 1.0 == True`` numerically); the hash
//! is canonical so equal values hash identically (``hash(1) ==
//! hash(1.0)`` in Python — we normalise the same way).

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString, PyTuple};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

#[derive(Clone, Debug)]
pub enum AttrVal {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Tuple(Vec<AttrVal>),
}

impl AttrVal {
    /// Numeric value used for cross-type equality / ordering (Python
    /// compares int/float/bool numerically).
    fn as_f64(&self) -> Option<f64> {
        match self {
            AttrVal::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            AttrVal::Int(i) => Some(*i as f64),
            AttrVal::Float(f) => Some(*f),
            _ => Option::None,
        }
    }

    /// Rank used for the deterministic total order (member iteration in
    /// Rust is BTreeSet-ordered; Python iterates hash sets, so any
    /// canonical order is equally faithful — we just need one).
    fn rank(&self) -> u8 {
        match self {
            AttrVal::None => 0,
            AttrVal::Bool(_) | AttrVal::Int(_) | AttrVal::Float(_) => 1,
            AttrVal::Str(_) => 2,
            AttrVal::Tuple(_) => 3,
        }
    }

    /// Convert to a Python object.
    pub fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(match self {
            AttrVal::None => py.None(),
            AttrVal::Bool(b) => b.into_pyobject(py)?.to_owned().unbind().into(),
            AttrVal::Int(i) => i.into_pyobject(py)?.unbind().into(),
            AttrVal::Float(f) => f.into_pyobject(py)?.unbind().into(),
            AttrVal::Str(s) => s.into_pyobject(py)?.unbind().into(),
            AttrVal::Tuple(items) => {
                let parts: Vec<Py<PyAny>> = items
                    .iter()
                    .map(|v| v.to_py(py))
                    .collect::<PyResult<_>>()?;
                PyTuple::new(py, parts)?.unbind().into()
            }
        })
    }

    /// Extract from a Python object — the mirror of
    /// ``_norm_attr_value`` (lists normalise to tuples).
    pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<AttrVal> {
        if obj.is_none() {
            return Ok(AttrVal::None);
        }
        // bool BEFORE int — Python bool is a subclass of int.
        if obj.is_instance_of::<PyBool>() {
            return Ok(AttrVal::Bool(obj.extract::<bool>()?));
        }
        if obj.is_instance_of::<PyInt>() {
            if let Ok(i) = obj.extract::<i64>() {
                return Ok(AttrVal::Int(i));
            }
            // Arbitrary-precision ints fall back to float semantics
            // only if they fit; otherwise keep the repr.
            if let Ok(f) = obj.extract::<f64>() {
                return Ok(AttrVal::Float(f));
            }
            return Err(pyo3::exceptions::PyOverflowError::new_err(
                "attr int does not fit i64/f64",
            ));
        }
        if obj.is_instance_of::<PyFloat>() {
            return Ok(AttrVal::Float(obj.extract::<f64>()?));
        }
        if obj.is_instance_of::<PyString>() {
            return Ok(AttrVal::Str(obj.extract::<String>()?));
        }
        if let Ok(t) = obj.cast::<PyTuple>() {
            let mut items = Vec::with_capacity(t.len());
            for it in t.iter() {
                items.push(AttrVal::from_py(&it)?);
            }
            return Ok(AttrVal::Tuple(items));
        }
        if let Ok(l) = obj.cast::<PyList>() {
            let mut items = Vec::with_capacity(l.len());
            for it in l.iter() {
                items.push(AttrVal::from_py(&it)?);
            }
            return Ok(AttrVal::Tuple(items));
        }
        Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "unsupported attr value type: {}",
            obj.get_type().name()?,
        )))
    }
}

impl PartialEq for AttrVal {
    fn eq(&self, other: &AttrVal) -> bool {
        match (self.as_f64(), other.as_f64()) {
            (Some(a), Some(b)) => {
                // Python numeric equality; for i64-exact compare keep
                // integer precision when both are ints.
                if let (AttrVal::Int(x), AttrVal::Int(y)) = (self, other) {
                    return x == y;
                }
                a == b
            }
            _ => match (self, other) {
                (AttrVal::None, AttrVal::None) => true,
                (AttrVal::Str(a), AttrVal::Str(b)) => a == b,
                (AttrVal::Tuple(a), AttrVal::Tuple(b)) => a == b,
                _ => false,
            },
        }
    }
}

impl Eq for AttrVal {}

impl Hash for AttrVal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            AttrVal::None => state.write_u8(0),
            // All numerics hash through a canonical form so that
            // Int(1), Float(1.0) and Bool(true) collide identically
            // (Python: hash(1) == hash(1.0) == hash(True)).
            AttrVal::Bool(b) => {
                state.write_u8(1);
                state.write_i64(if *b { 1 } else { 0 });
            }
            AttrVal::Int(i) => {
                state.write_u8(1);
                state.write_i64(*i);
            }
            AttrVal::Float(f) => {
                state.write_u8(1);
                let v = if *f == 0.0 { 0.0 } else { *f }; // -0.0 == 0.0
                if v.fract() == 0.0 && v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                    state.write_i64(v as i64);
                } else {
                    state.write_u64(v.to_bits());
                }
            }
            AttrVal::Str(s) => {
                state.write_u8(2);
                s.hash(state);
            }
            AttrVal::Tuple(items) => {
                state.write_u8(3);
                items.hash(state);
            }
        }
    }
}

impl PartialOrd for AttrVal {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AttrVal {
    fn cmp(&self, other: &Self) -> Ordering {
        let (ra, rb) = (self.rank(), other.rank());
        if ra != rb {
            return ra.cmp(&rb);
        }
        match (self, other) {
            (AttrVal::None, AttrVal::None) => Ordering::Equal,
            // Numeric ordering across int/float/bool.
            (a, b) if a.as_f64().is_some() && b.as_f64().is_some() => {
                if let (AttrVal::Int(x), AttrVal::Int(y)) = (self, other) {
                    return x.cmp(y);
                }
                let (fa, fb) = (a.as_f64().unwrap(), b.as_f64().unwrap());
                fa.partial_cmp(&fb).unwrap_or(Ordering::Equal)
            }
            (AttrVal::Str(a), AttrVal::Str(b)) => a.cmp(b),
            (AttrVal::Tuple(a), AttrVal::Tuple(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}
