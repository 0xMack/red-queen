"""copycat.report's comparison, which decides what a change is said to have done."""

from evaluator.protocols.copycat_report import compare


def group(mean, **per_image):
    return {"mean": mean, "per_image": per_image}


def test_a_comparison_counts_the_images_that_moved_largest_first():
    before = {"icons/train": group(20.0, a=10.0, b=20.0, c=30.0, d=20.0)}
    after = {"icons/train": group(21.0, a=15.0, b=19.0, c=30.01, d=20.0)}
    row = compare(before, after)["icons/train"]
    assert (row["before"], row["after"], row["up"], row["down"]) == (20.0, 21.0, 1, 1)
    assert list(row["moved"].items()) == [("a", 5.0), ("b", -1.0)]


def test_a_group_only_one_report_measured_is_left_out():
    after = {"icons/heldout": group(23.0, a=23.0), "icons/train": group(25.0, a=25.0)}
    assert list(compare({"icons/train": group(25.0, a=25.0)}, after)) == ["icons/train"]
