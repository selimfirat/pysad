//! Optional native kernels for PySAD.
//!
//! This module currently contains one kernel. It routes one point through all
//! half-space trees in the same heap layout used by `HalfSpaceTrees`.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Return the mass cell indices visited by one point in every tree.
///
/// The split arrays use the heap layout of `HalfSpaceTrees`. Each internal node
/// stores one feature index and one split value. A point goes right when its
/// feature value is greater than the split value.
///
/// Args:
///     x: One point with shape `(num_features,)` and dtype `float64`.
///     split_atts: Split feature indices with shape `(num_trees, num_splits)`
///         and dtype `uint16`.
///     split_values: Split values with shape `(num_trees, num_splits)` and
///         dtype `float64`.
///     max_depth: The maximum depth of every tree.
///
/// Returns:
///     An `int64` array with shape `(max_depth + 1, num_trees)`. Each column
///     contains the flat mass cell indices visited in one tree.
#[pyfunction]
#[pyo3(signature = (x, split_atts, split_values, max_depth))]
fn half_space_trees_path(
    py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    split_atts: PyReadonlyArray2<'_, u16>,
    split_values: PyReadonlyArray2<'_, f64>,
    max_depth: usize,
) -> PyResult<Py<PyArray2<i64>>> {
    let x = x.as_array();
    let split_atts = split_atts.as_array();
    let split_values = split_values.as_array();

    if split_atts.shape() != split_values.shape() {
        return Err(PyValueError::new_err(
            "split_atts and split_values must have the same shape.",
        ));
    }

    let depth = u32::try_from(max_depth)
        .map_err(|_| PyValueError::new_err("max_depth is too large."))?;
    let expected_splits = 1usize
        .checked_shl(depth)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| PyValueError::new_err("max_depth is too large."))?;

    let num_trees = split_atts.shape()[0];
    let num_splits = split_atts.shape()[1];
    if num_splits != expected_splits {
        return Err(PyValueError::new_err(format!(
            "split_atts must have {expected_splits} columns for max_depth {max_depth}."
        )));
    }

    let num_nodes = 2 * num_splits + 1;
    let mut paths = vec![vec![0i64; num_trees]; max_depth + 1];

    for tree in 0..num_trees {
        let mass_base = tree * num_nodes;
        let mut heap_index = 1usize;
        paths[0][tree] = mass_base as i64;

        for depth_index in 0..max_depth {
            let node = heap_index - 1;
            let feature = split_atts[[tree, node]] as usize;
            let goes_right = x[feature] > split_values[[tree, node]];
            heap_index = 2 * heap_index + usize::from(goes_right);
            paths[depth_index + 1][tree] = (mass_base + heap_index - 1) as i64;
        }
    }

    Ok(PyArray2::from_vec2(py, &paths)?.into())
}

/// Define the `pysad_kernels` Python module.
#[pymodule]
fn pysad_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(half_space_trees_path, module)?)?;
    Ok(())
}
