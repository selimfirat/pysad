import numpy as np
import pytest

from pysad.transform.preprocessing import DrainLogParser


def _ids(parser, lines):
    """Returns the event IDs that the parser hands out for each line, in order."""
    out = []
    for line in lines:
        out.append(int(parser.fit_transform_partial(np.array(line, dtype=object))[0]))
    return out


def test_same_line_keeps_the_same_event_id():
    parser = DrainLogParser()
    ids = _ids(parser, ["Error: failed to connect", "Error: failed to connect", "Info: started"])
    assert ids == [1, 1, 2]


def test_event_ids_follow_file_order_and_reuse_seen_templates():
    parser = DrainLogParser()
    ids = _ids(parser, ["a", "b", "a", "c", "b", "a"])
    assert ids == [1, 2, 1, 3, 2, 1]


def test_similar_lines_merge_and_different_lines_split():
    parser = DrainLogParser(depth=5, st=0.5)
    # The tree has one length layer and three token layers, so the lines share their first three
    # tokens. The first two lines agree in three of four tokens, so the second line reaches the same
    # leaf and matches the template of the first one. The third line differs in the third token, so
    # it reaches another leaf.
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


def test_a_line_at_the_threshold_matches_the_template():
    parser = DrainLogParser(st=1.0)
    # The second line is identical to the first one, so its similarity is exactly 1.0.
    ids = _ids(parser, ["Error failed connect host1", "Error failed connect host1"])
    assert ids == [1, 1]


def test_the_first_template_wins_a_tie():
    parser = DrainLogParser(depth=2, st=0.4)
    # The lines have the same length, so they reach the same node. The second line agrees with the
    # first template in one of four tokens, so it gets its own template. The third line agrees with
    # both templates in two of four tokens, so both templates match it with the same similarity.
    ids = _ids(parser, ["a b c d", "a x e f", "a b e x"])
    assert ids == [1, 2, 1]


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
    with pytest.raises(ValueError):
        DrainLogParser(depth=1)


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


def test_batch_methods_reject_non_string_lines():
    parser = DrainLogParser()
    with pytest.raises(TypeError):
        parser.fit(np.array([1.5, 2.5]))
    # The failed fit did not turn 1.5 into the template "1.5".
    np.testing.assert_array_equal(parser.transform("1.5"), [[-1.0]])
    with pytest.raises(TypeError):
        parser.transform(np.array([1.5]))
    with pytest.raises(TypeError):
        parser.transform(np.array(["a b", 1.5], dtype=object))
    with pytest.raises(TypeError):
        parser.fit_transform(np.array([1.5, 2.5]))


def test_the_length_layer_separates_lines_of_different_lengths():
    parser = DrainLogParser()
    # The first and third lines have four tokens, and the second line has five tokens. The length
    # layer sends the second line to another node, so it gets its own event ID. The third line has
    # the length of the first line, so it reuses the event ID of the first line.
    ids = _ids(parser, ["INFO a b c", "INFO a b c d", "INFO a b c"])
    assert ids == [1, 2, 1]
    # A line with the length of the second line reuses its event ID.
    assert _ids(parser, ["INFO a b c e"]) == [2]


def test_tokens_with_digits_share_one_node():
    parser = DrainLogParser()
    # The first token holds a digit, so all three lines reach the same node.
    ids = _ids(parser, ["120 bytes received", "121 bytes received", "99 bytes received"])
    assert ids == [1, 1, 1]
    # The second token holds a digit, so these lines also reach the same node.
    assert _ids(parser, ["bytes 120 received", "bytes 121 received"]) == [2, 2]


def test_depth_two_uses_only_the_length_layer():
    parser = DrainLogParser(depth=2, st=0.4)
    # The tree has one length layer and no token layer. The lines have the same length, so they
    # reach the same node and the parser compares them.
    ids = _ids(parser, ["a b c", "x b c"])
    assert ids == [1, 1]


def test_depth_three_adds_one_token_layer():
    parser = DrainLogParser(depth=3, st=0.4)
    # The tree has one length layer and one token layer. The first tokens differ, so the lines reach
    # different nodes and the parser never compares them.
    ids = _ids(parser, ["a b c", "x b c"])
    assert ids == [1, 2]


def test_the_default_depth_keeps_two_token_layers():
    parser = DrainLogParser()
    # The tree has one length layer and two token layers. The lines share their first two tokens, so
    # they reach the same leaf. The first two lines agree in two of four tokens, so the second line
    # matches the template of the first one.
    ids = _ids(parser, ["a b 1 x", "a b 2 y", "a b 3 z"])
    assert ids == [1, 1, 1]


def test_the_depth_bounds_the_path_to_a_leaf():
    parser = DrainLogParser(depth=3, st=0.7)
    # The path uses the first token, so both lines reach the same leaf.
    ids = _ids(parser, ["a b c d", "a b c e"])
    # Three of four tokens agree, so the second line matches the template of the first line.
    assert ids == [1, 1]


def test_lines_that_differ_within_the_depth_are_never_compared():
    parser = DrainLogParser(depth=4, st=0.7)
    # The second token differs, so the lines reach different leaves.
    ids = _ids(parser, ["a b c d", "a x c d"])
    # Three of four tokens agree, but the parser never compares lines of different leaves.
    assert ids == [1, 2]


def test_max_child_stops_a_node_from_growing():
    parser = DrainLogParser(depth=2, max_child=1, st=0.4)
    # The root holds one child, so the length node of a second token count cannot be created. The
    # template of the second line is stored at the root.
    # The third line agrees with that template in two of three tokens, so it matches it.
    ids = _ids(parser, ["a x", "b y c", "b z c"])
    assert ids == [1, 2, 2]


def test_parsing_a_fresh_leaf_keeps_the_event_ids_of_other_templates():
    parser = DrainLogParser(depth=2, st=0.9)
    # The 50 lines reach the same leaf, and two of three tokens agree, so each one gets a template.
    ids = _ids(parser, [f"common fixed extra{i}" for i in range(50)])
    assert ids == list(range(1, 51))
    # A line that reaches a fresh leaf gets a new event ID.
    assert _ids(parser, ["other only"]) == [51]
    # The templates of the other leaf keep their event IDs.
    assert _ids(parser, ["common fixed extra0"]) == [1]


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
