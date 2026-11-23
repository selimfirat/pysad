import pickle

import numpy as np
import pytest

from pysad import _kernels
from pysad.models import HalfSpaceTrees, RobustRandomCutForest


def _reference_hst_path(x, split_atts, split_values, max_depth):
    assert x.dtype == np.float64
    assert split_atts.dtype == np.uint16
    assert split_values.dtype == np.float64

    num_trees, num_splits = split_atts.shape
    num_nodes = 2 * num_splits + 1
    paths = np.empty((max_depth + 1, num_trees), dtype=np.int64)
    for tree in range(num_trees):
        mass_base = tree * num_nodes
        heap_idx = 1
        paths[0, tree] = mass_base
        for k in range(max_depth):
            node = heap_idx - 1
            goes_right = x[int(split_atts[tree, node])] > split_values[tree, node]
            heap_idx = 2 * heap_idx + int(goes_right)
            paths[k + 1, tree] = mass_base + heap_idx - 1
    return paths


def _reference_rrcf_route(x, roots, dim, cut, left, right):
    assert x.dtype == np.float64
    for arr in (roots, dim, left, right):
        assert arr.dtype == np.int32
    assert cut.dtype == np.float64

    leaves = np.full(len(roots), -1, dtype=np.int32)
    for tree in range(len(roots)):
        node = int(roots[tree])
        if node < 0:
            continue
        while True:
            d = int(dim[node])
            if d < 0:
                leaves[tree] = node
                break
            node = int(left[node]) if x[d] <= cut[node] else int(right[node])
            if node < 0:
                leaves[tree] = -1
                break
    return leaves


def _reference_rrcf_codisp(leaves, init_counts, parent, sibling, count):
    for arr in (leaves, init_counts, parent, sibling, count):
        assert arr.dtype == np.int32

    total = 0.0
    for tree in range(len(leaves)):
        leaf = int(leaves[tree])
        if leaf < 0:
            continue
        best = float(init_counts[tree])
        curr = leaf
        while (p := int(parent[curr])) >= 0:
            sib = int(sibling[curr])
            if sib >= 0:
                ratio = float(count[sib]) / (float(count[curr]) + 1.0)
                if ratio > best:
                    best = ratio
            curr = p
        total += best
    return total


def test_hst_rust_dispatch_equivalence_and_pickling(monkeypatch) -> None:
    rng = np.random.default_rng(7)
    stream = rng.random((40, 4), dtype=np.float64)

    monkeypatch.setattr(_kernels, "half_space_trees_path", None)
    model_np = HalfSpaceTrees(
        feature_mins=np.zeros(4),
        feature_maxes=np.ones(4),
        window_size=10,
        num_trees=5,
        max_depth=4,
        random_state=7,
    )
    scores_np = [model_np.fit_score_partial(x) for x in stream]
    exp_np = model_np.explain_partial(stream[-1])

    calls = 0

    def wrapped_hst_path(x, split_atts, split_values, max_depth):
        nonlocal calls
        calls += 1
        return _reference_hst_path(x, split_atts, split_values, max_depth)

    monkeypatch.setattr(_kernels, "half_space_trees_path", wrapped_hst_path)
    model_rs = HalfSpaceTrees(
        feature_mins=np.zeros(4),
        feature_maxes=np.ones(4),
        window_size=10,
        num_trees=5,
        max_depth=4,
        random_state=7,
    )
    assert model_rs.split_atts.dtype == np.uint8
    scores_rs = [model_rs.fit_score_partial(x) for x in stream]
    exp_rs = model_rs.explain_partial(stream[-1])

    assert calls == len(stream) + 1
    assert model_rs._split_atts_u16 is not None
    assert model_rs._split_atts_u16.dtype == np.uint16
    assert scores_rs == scores_np
    np.testing.assert_array_equal(exp_rs, exp_np)

    restored = pickle.loads(pickle.dumps(model_rs))
    assert restored.score_partial(stream[0]) == model_rs.score_partial(stream[0])


def test_hst_falls_back_for_bad_shape_and_high_dimension(monkeypatch) -> None:
    calls = 0

    def fail_if_called(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("Rust kernel should not be called")

    monkeypatch.setattr(_kernels, "half_space_trees_path", fail_if_called)

    model = HalfSpaceTrees(
        feature_mins=np.zeros(4),
        feature_maxes=np.ones(4),
        window_size=10,
        num_trees=3,
        max_depth=3,
        random_state=0,
    )
    with pytest.raises((IndexError, ValueError)):
        model.score_partial(np.array([0.1, 0.2]))
    assert calls == 0

    high_dim = HalfSpaceTrees(
        feature_mins=np.zeros(65537),
        feature_maxes=np.ones(65537),
        window_size=4,
        num_trees=1,
        max_depth=1,
        random_state=0,
    )
    score = high_dim.fit_score_partial(np.full(65537, 0.5))
    assert isinstance(score, float)
    assert calls == 0


def test_rrcf_rust_dispatch_equivalence_and_pickling(monkeypatch) -> None:
    rng = np.random.default_rng(11)
    stream = rng.random((45, 3), dtype=np.float64)
    stream[12] = stream[5]

    monkeypatch.setattr(_kernels, "rrcf_route_to_leaf", None)
    monkeypatch.setattr(_kernels, "rrcf_codisp_sum", None)
    model_np = RobustRandomCutForest(num_trees=8, shingle_size=1, tree_size=16, random_state=11)
    scores_np = [model_np.fit_score_partial(x) for x in stream]

    route_calls = 0
    codisp_calls = 0

    def wrapped_route(x, roots, dim, cut, left, right):
        nonlocal route_calls
        route_calls += 1
        return _reference_rrcf_route(x, roots, dim, cut, left, right)

    def wrapped_codisp(leaves, init_counts, parent, sibling, count):
        nonlocal codisp_calls
        codisp_calls += 1
        return _reference_rrcf_codisp(leaves, init_counts, parent, sibling, count)

    monkeypatch.setattr(_kernels, "rrcf_route_to_leaf", wrapped_route)
    monkeypatch.setattr(_kernels, "rrcf_codisp_sum", wrapped_codisp)
    model_rs = RobustRandomCutForest(num_trees=8, shingle_size=1, tree_size=16, random_state=11)
    scores_rs = [model_rs.fit_score_partial(x) for x in stream]

    assert route_calls > 0
    assert codisp_calls > 0
    np.testing.assert_allclose(scores_rs, scores_np, rtol=1e-12, atol=1e-12)

    restored = pickle.loads(pickle.dumps(model_rs))
    assert restored.score_partial(stream[5]) == pytest.approx(model_rs.score_partial(stream[5]))
