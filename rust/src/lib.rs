//! Optional native kernels for PySAD.
//!
//! This module contains two families of kernels:
//!
//! 1. `half_space_trees_path` - routes one point through all half-space trees
//!    in the heap layout used by `HalfSpaceTrees`.
//!
//! 2. RRCF (Robust Random Cut Forest) kernels for tree traversal and collusive
//!    displacement (CoDisp) computation. The RRCF trees use a dynamic
//!    structure-of-arrays layout with explicit parent/child/sibling pointers.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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

/// Route one point down each RRCF tree to find the leaf it falls into.
///
/// The RRCF trees use a structure-of-arrays layout with explicit pointers.
/// Each tree is independent and stored in flat arrays indexed by node ID.
///
/// Args:
///     x: One point with shape `(num_features,)` and dtype `float64`.
///     roots: Root node index for each tree, shape `(num_trees,)`, dtype `int32`.
///         A value of -1 means the tree is empty.
///     dim: Split dimension for each node, shape `(num_nodes,)`, dtype `int32`.
///         A value of -1 marks a leaf node.
///     cut: Split value for each node, shape `(num_nodes,)`, dtype `float64`.
///     left: Left child index for each node, shape `(num_nodes,)`, dtype `int32`.
///     right: Right child index for each node, shape `(num_nodes,)`, dtype `int32`.
///
/// Returns:
///     An `int32` array with shape `(num_trees,)` containing the leaf node
///     index for each tree. Empty trees return -1.
#[pyfunction]
#[pyo3(signature = (x, roots, dim, cut, left, right))]
fn rrcf_route_to_leaf(
    _py: Python<'_>,
    x: PyReadonlyArray1<'_, f64>,
    roots: PyReadonlyArray1<'_, i32>,
    dim: PyReadonlyArray1<'_, i32>,
    cut: PyReadonlyArray1<'_, f64>,
    left: PyReadonlyArray1<'_, i32>,
    right: PyReadonlyArray1<'_, i32>,
) -> PyResult<Py<PyArray1<i32>>> {
    let x = x.as_array();
    let roots = roots.as_array();
    let dim = dim.as_array();
    let cut = cut.as_array();
    let left = left.as_array();
    let right = right.as_array();

    let num_trees = roots.len();
    let mut leaves = vec![-1i32; num_trees];

    for tree in 0..num_trees {
        let mut node = roots[tree];
        if node < 0 {
            leaves[tree] = -1;
            continue;
        }

        loop {
            let d = dim[node as usize];
            if d < 0 {
                leaves[tree] = node;
                break;
            }
            let d = d as usize;
            if x[d] <= cut[node as usize] {
                node = left[node as usize];
            } else {
                node = right[node as usize];
            }
            if node < 0 {
                // Should not happen in a valid tree
                leaves[tree] = -1;
                break;
            }
        }
    }

    Ok(PyArray1::from_vec(_py, leaves).into())
}

/// Compute the collusive displacement (CoDisp) for a set of leaf nodes.
///
/// CoDisp is the maximum ratio `count[sibling] / (count[node] + 1)` along the
/// path from the leaf to the root.
///
/// Args:
///     leaves: Leaf node indices for each tree, shape `(num_trees,)`, dtype `int32`.
///         A value of -1 means the tree is empty or the point is a duplicate.
///     init_counts: Initial count values for each leaf, shape `(num_trees,)`, dtype `int32`.
///         Use 0 for duplicate leaves, otherwise use the leaf's count.
///     parent: Parent index for each node, shape `(num_nodes,)`, dtype `int32`.
///     sibling: Sibling index for each node, shape `(num_nodes,)`, dtype `int32`.
///     count: Count (number of points in subtree) for each node, shape `(num_nodes,)`, dtype `int32`.
///
/// Returns:
///     A float64 scalar: the sum of CoDisp values across all trees.
#[pyfunction]
#[pyo3(signature = (leaves, init_counts, parent, sibling, count))]
fn rrcf_codisp_sum(
    _py: Python<'_>,
    leaves: PyReadonlyArray1<'_, i32>,
    init_counts: PyReadonlyArray1<'_, i32>,
    parent: PyReadonlyArray1<'_, i32>,
    sibling: PyReadonlyArray1<'_, i32>,
    count: PyReadonlyArray1<'_, i32>,
) -> PyResult<f64> {
    let leaves = leaves.as_array();
    let init_counts = init_counts.as_array();
    let parent = parent.as_array();
    let sibling = sibling.as_array();
    let count = count.as_array();

    let num_trees = leaves.len();
    let mut total: f64 = 0.0;

    for tree in 0..num_trees {
        let leaf = leaves[tree];
        if leaf < 0 {
            continue;
        }

        let init_count = init_counts[tree] as f64;
        let mut best = init_count;
        let mut curr = leaf;

        loop {
            let p = parent[curr as usize];
            if p < 0 {
                break;
            }
            let sib = sibling[curr as usize];
            if sib >= 0 {
                let ratio = count[sib as usize] as f64 / (count[curr as usize] as f64 + 1.0);
                if ratio > best {
                    best = ratio;
                }
            }
            curr = p;
        }

        total += best;
    }

    Ok(total)
}

/// Define the `pysad_kernels` Python module.
#[pymodule]
fn pysad_kernels(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(half_space_trees_path, module)?)?;
    module.add_function(wrap_pyfunction!(rrcf_route_to_leaf, module)?)?;
    module.add_function(wrap_pyfunction!(rrcf_codisp_sum, module)?)?;
    Ok(())
}
