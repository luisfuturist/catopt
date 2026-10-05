//! Attribute values — the hashable payloads of enode attrs and
//! attribute-metavariable bindings.
//!
//! Mirrors the Python semantics of ``catopt_core.egraph.types``:
//! attribute values are ints, floats, bools, strings, tuples or None.
//! Identity is *spelling-strict* — ``Int(1)``, ``Float(1.0)`` and
//! ``Bool(true)`` are distinct enode members (Python: ``repr``-keyed
//! ``_attr_key``/``Op.__eq__``; ``0.0`` != ``-0.0``; NaN is reflexive).
//! Matching stays numerically lenient through ``loose_eq`` — the same
//! split Python keeps between identity and the match sites.

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

    /// Rank used for the deterministic total order — one rank per
    /// variant so ``Ord`` agrees with the strict ``Eq`` partition
    /// (``BTreeSet`` dedup runs through ``Ord``, so a lenient order
    /// would merge ``0``/``0.0`` enodes even with strict equality).
    fn rank(&self) -> u8 {
        match self {
            AttrVal::None => 0,
            AttrVal::Bool(_) => 1,
            AttrVal::Int(_) => 2,
            AttrVal::Float(_) => 3,
            AttrVal::Str(_) => 4,
            AttrVal::Tuple(_) => 5,
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

impl AttrVal {
    /// Numerically lenient equality — the *matcher* contract: Python's
    /// match sites deliberately accept ``0`` for ``0.0`` (the
    /// ``match_pattern`` / ``_term_match`` attr leniency is docstring-
    /// pinned there).  Use only at match points, never for identity —
    /// interning/dedup go through the strict ``==``/``Hash``/``Ord``.
    pub fn loose_eq(&self, other: &AttrVal) -> bool {
        match (self.as_f64(), other.as_f64()) {
            (Some(a), Some(b)) => {
                if let (AttrVal::Int(x), AttrVal::Int(y)) = (self, other) {
                    return x == y;
                }
                a == b
            }
            _ => match (self, other) {
                (AttrVal::None, AttrVal::None) => true,
                (AttrVal::Str(a), AttrVal::Str(b)) => a == b,
                (AttrVal::Tuple(a), AttrVal::Tuple(b)) => {
                    a.len() == b.len()
                        && a.iter().zip(b.iter()).all(|(x, y)| x.loose_eq(y))
                }
                _ => false,
            },
        }
    }
}

impl PartialEq for AttrVal {
    /// Strict (repr-keyed) identity — mirrors Python's ``Op.__eq__`` /
    /// ``_attr_key`` post the attr-interning fix: ``Int(1)``,
    /// ``Float(1.0)`` and ``Bool(true)`` are distinct; ``0.0`` !=
    /// ``-0.0``; every NaN compares equal (Python ``repr`` keys them
    /// all as ``"nan"``).
    fn eq(&self, other: &AttrVal) -> bool {
        match (self, other) {
            (AttrVal::None, AttrVal::None) => true,
            (AttrVal::Bool(a), AttrVal::Bool(b)) => a == b,
            (AttrVal::Int(a), AttrVal::Int(b)) => a == b,
            (AttrVal::Float(a), AttrVal::Float(b)) => {
                a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
            }
            (AttrVal::Str(a), AttrVal::Str(b)) => a == b,
            (AttrVal::Tuple(a), AttrVal::Tuple(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for AttrVal {}

impl Hash for AttrVal {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            AttrVal::None => state.write_u8(0),
            // Strict: the tag distinguishes Bool/Int/Float so the hash
            // agrees with the repr-keyed ``==`` partition.
            AttrVal::Bool(b) => {
                state.write_u8(1);
                b.hash(state);
            }
            AttrVal::Int(i) => {
                state.write_u8(2);
                i.hash(state);
            }
            AttrVal::Float(f) => {
                state.write_u8(3);
                // bits distinguish 0.0/-0.0; all NaN payloads share
                // one canonical hash (repr "nan").
                let bits =
                    if f.is_nan() { f64::NAN.to_bits() } else { f.to_bits() };
                state.write_u64(bits);
            }
            AttrVal::Str(s) => {
                state.write_u8(4);
                s.hash(state);
            }
            AttrVal::Tuple(items) => {
                state.write_u8(5);
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
            (AttrVal::Bool(a), AttrVal::Bool(b)) => a.cmp(b),
            (AttrVal::Int(a), AttrVal::Int(b)) => a.cmp(b),
            // IEEE total order; NaN payloads canonicalise (consistent
            // with the nan-reflexive ``Eq``).
            (AttrVal::Float(a), AttrVal::Float(b)) => {
                let fa = if a.is_nan() { f64::NAN } else { *a };
                let fb = if b.is_nan() { f64::NAN } else { *b };
                fa.total_cmp(&fb)
            }
            (AttrVal::Str(a), AttrVal::Str(b)) => a.cmp(b),
            (AttrVal::Tuple(a), AttrVal::Tuple(b)) => a.cmp(b),
            _ => Ordering::Equal,
        }
    }
}
