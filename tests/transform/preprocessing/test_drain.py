import numpy as np
import pytest

from pysad.transform.preprocessing import DrainLogParser


def _ids(parser, lines):
    """Returns the event IDs that the parser hands out for each line, in order."""
    out = []
    for line in lines:
        out.append(int(parser.fit_transform_partial(np.array(line, dtype=object))[0]))
    return out


def _tree_stats(node):
    """Returns (max depth, max children, nodes, clusters) of the tree below ``node``."""
    max_depth = node.depth
    max_children = len(node.children)
    nodes = 1
    clusters = len(node.clusters)
    for child in node.children.values():
        child_depth, child_children, child_nodes, child_clusters = _tree_stats(child)
        max_depth = max(max_depth, child_depth)
        max_children = max(max_children, child_children)
        nodes += child_nodes
        clusters += child_clusters
    return max_depth, max_children, nodes, clusters


def test_same_line_keeps_the_same_event_id():
    parser = DrainLogParser()
    ids = _ids(parser, ["Error: failed to connect", "Error: failed to connect", "Info: started"])
    assert ids == [1, 1, 2]


def test_event_ids_follow_file_order_and_reuse_seen_templates():
    parser = DrainLogParser()
    ids = _ids(parser, ["a", "b", "a", "c", "b", "a"])
    assert ids == [1, 2, 1, 3, 2, 1]


def test_similar_lines_merge_and_different_lines_split():
    parser = DrainLogParser(depth=3, st=0.5)
    # The two lines share the first three tokens and differ only in the last one, so they reach the
    # same leaf and their similarity is 3 / 4 >= 0.5. The third line diverges at the third token.
    ids = _ids(
        parser,
        [
            "Error failed connect host1",
            "Error failed connect host2",
            "Error failed disconnect host3",
        ],
    )
    assert ids == [1, 1, 2]


def test_a_line_below_the_threshold_gets_a_new_event_id():
    parser = DrainLogParser(depth=3, st=0.9)
    # Same leaf, but 3 / 4 = 0.75 < 0.9, so the second line gets a new template.
    ids = _ids(parser, ["Error failed connect host1", "Error failed connect host2"])
    assert ids == [1, 2]


def test_transform_partial_returns_a_float_array_and_does_not_learn():
    parser = DrainLogParser()
    parser.fit_transform_partial("a b c")
    # An unseen line has no template yet.
    np.testing.assert_array_equal(parser.transform_partial("x y z"), [-1.0])
    # The unseen line did not create a template: it still gets a new ID when it is fitted.
    assert int(parser.fit_transform_partial("x y z")[0]) == 2


def test_fit_transform_partial_returns_one_event_id_per_line():
    parser = DrainLogParser()
    out = parser.fit_transform_partial("hello world")
    assert isinstance(out, np.ndarray)
    assert out.shape == (1,)
    assert out.dtype == np.float64
    assert int(out[0]) == 1


def test_invalid_depth_raises():
    with pytest.raises(TypeError):
        DrainLogParser(depth="4")
    with pytest.raises(TypeError):
        DrainLogParser(depth=True)
    with pytest.raises(ValueError):
        DrainLogParser(depth=0)


def test_invalid_st_raises():
    with pytest.raises(TypeError):
        DrainLogParser(st="0.4")
    with pytest.raises(TypeError):
        DrainLogParser(st=True)
    with pytest.raises(ValueError):
        DrainLogParser(st=-0.1)
    with pytest.raises(ValueError):
        DrainLogParser(st=1.1)


def test_invalid_max_child_raises():
    with pytest.raises(TypeError):
        DrainLogParser(max_child="100")
    with pytest.raises(TypeError):
        DrainLogParser(max_child=True)
    with pytest.raises(ValueError):
        DrainLogParser(max_child=0)


def test_non_string_input_raises():
    parser = DrainLogParser()
    with pytest.raises(TypeError):
        parser.fit_transform_partial(np.array(1.5))
    with pytest.raises(ValueError):
        parser.fit_transform_partial(np.array(["a", "b"]))
    with pytest.raises(ValueError):
        parser.fit_transform_partial(np.array([["a", "b"]]))


def test_tree_depth_is_bounded_by_depth():
    parser = DrainLogParser(depth=3)
    _ids(
        parser,
        [f"token{i} some message here {i}" for i in range(200)],
    )
    max_depth, _, _, _ = _tree_stats(parser.root_node)
    assert max_depth == 3


def test_children_per_node_are_bounded_by_max_child():
    parser = DrainLogParser(depth=4, max_child=2)
    _ids(parser, [f"distinct{i} message" for i in range(10)])
    _, max_children, _, _ = _tree_stats(parser.root_node)
    assert max_children <= 2


def test_parsing_a_fresh_leaf_touches_no_other_template():
    parser = DrainLogParser(depth=2, st=1.0)
    # 50 distinct templates share the leaf reached by ("common", "fixed").
    _ids(parser, [f"common fixed extra{i}" for i in range(50)])
    leaf = parser.root_node.children["common"].children["fixed"]
    assert len(leaf.clusters) == 50
    # A line reaching a fresh leaf gets a new ID and does not change the other leaf.
    before = len(leaf.clusters)
    assert _ids(parser, ["other only"])[0] == 51
    assert len(leaf.clusters) == before
    fresh = parser.root_node.children["other"].children["only"]
    assert len(fresh.clusters) == 1


def test_batch_fit_and_transform_return_ids_in_order():
    parser = DrainLogParser()
    lines = np.array(["a", "b", "a", "c"], dtype=object)
    parser.fit(lines)
    ids = parser.transform(lines)
    assert ids.shape == (4, 1)
    assert ids.dtype == np.float64
    assert [int(v[0]) for v in ids] == [1, 2, 1, 3]

    ids = parser.fit_transform(lines)
    assert [int(v[0]) for v in ids] == [1, 2, 1, 3]


def test_empty_line_is_a_template():
    parser = DrainLogParser()
    assert _ids(parser, ["", ""])[0] == 1
    assert _ids(parser, [""])[-1] == 1
