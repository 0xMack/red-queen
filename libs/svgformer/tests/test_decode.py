from svgformer import tokens as T
from svgformer.decode import REPEAT_WINDOW, _Segments


def n(v: int) -> int:
    return T.NUM0 + v


def feed(tokens: list[int]) -> _Segments:
    s = _Segments()
    for t in tokens:
        s.push(t)
    return s


PATH, M, L, C = T.TYPE_ID["path"], T.COMMAND_ID["M"], T.COMMAND_ID["L"], T.COMMAND_ID["C"]


def test_a_segment_going_nowhere_is_banned():
    s = feed([T.BOS, PATH, M, n(10), n(20), L, n(10)])
    assert n(20) in s.banned()  # L 10 20 from (10, 20)
    assert feed([T.BOS, PATH, M, n(10), n(20), L, n(11)]).banned() == []


def test_a_recent_segment_is_not_repeated():
    s = feed([T.BOS, PATH, M, n(0), n(0), L, n(5), n(6), L, n(7), n(8), L, n(5)])
    assert s.banned() == [n(6)]
    curve = [C, n(1), n(2), n(3), n(4), n(5), n(6)]
    s = feed([T.BOS, PATH, M, n(0), n(0), *curve, *curve[:-1]])
    assert s.banned() == [n(6)]


def test_only_the_window_counts_and_only_inside_a_path():
    segments = [t for k in range(REPEAT_WINDOW + 1) for t in (L, n(100 + k), n(200 + k))]
    s = feed([T.BOS, PATH, M, n(0), n(0), *segments, L, n(100)])
    assert s.banned() == []  # L 100 200 is older than the window
    s = feed([T.BOS, PATH, M, n(0), n(0), L, n(5), n(6), T.PATH_END, PATH, M, n(0), n(0), L, n(5)])
    assert s.banned() == []  # a new path starts afresh
    assert feed([T.BOS, T.TYPE_ID["rect"], n(1)]).banned() == []


def test_a_path_drifting_in_place_has_stalled():
    from svgformer.decode import STALL_SEGMENTS

    drift = [t for k in range(STALL_SEGMENTS) for t in (L, n(640 + k % 2), n(810 + k))]
    assert feed([T.BOS, PATH, M, n(0), n(0), *drift]).stalled
    assert not feed([T.BOS, PATH, M, n(0), n(0), *drift[:-3]]).stalled  # one segment short
    spread = [t for k in range(STALL_SEGMENTS) for t in (L, n(100 * k), n(5))]
    assert not feed([T.BOS, PATH, M, n(0), n(0), *spread]).stalled
    assert not feed([T.BOS, PATH, M, n(0), n(0), *drift, L, n(1)]).stalled  # mid-segment: nothing to end yet
