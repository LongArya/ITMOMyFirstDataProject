import pytest
from general.utils import get_xyxy_boxes_iou, get_intervals_intersection, xyxy_box_area


def test_xyxy_box_area():
    box = [0, 0, 2, 2]
    assert xyxy_box_area(box) == 4


def test_intervals_intersection():
    interval1 = (1, 4)
    interval2 = (3, 5)
    assert get_intervals_intersection(interval1, interval2) == 1


def test_intervals_intersection_reversed_order():
    interval1 = (3, 5)
    interval2 = (1, 4)
    assert get_intervals_intersection(interval1, interval2) == 1


def test_nested_intervals_intersection():
    interval1 = (1, 5)
    interval2 = (2, 4)
    assert get_intervals_intersection(interval1, interval2) == 2


def test_nested_intervals_intersection_reversed_order():
    interval1 = (2, 4)
    interval2 = (1, 5)
    assert get_intervals_intersection(interval1, interval2) == 2


def test_intervals_zero_intersection():
    interval1 = (1, 2)
    interval2 = (3, 5)
    assert get_intervals_intersection(interval1, interval2) == 0


def test_intervals_zero_intersection_reversed_order():
    interval1 = (3, 5)
    interval2 = (1, 2)
    assert get_intervals_intersection(interval1, interval2) == 0


def test_boxes_iou():
    box1 = [1, 1, 3, 3]
    box2 = [2, 2, 4, 4]
    assert get_xyxy_boxes_iou(box1, box2) == pytest.approx(1 / 7)


def test_boxes_iou_reversed_order():
    box1 = [2, 2, 4, 4]
    box2 = [1, 1, 3, 3]
    assert get_xyxy_boxes_iou(box1, box2) == pytest.approx(1 / 7)


def test_boxes_zero_iou():
    box1 = [1, 1, 3, 3]
    box2 = [4, 4, 5, 5]
    assert get_xyxy_boxes_iou(box1, box2) == 0


def test_boxes_zero_iou_reversed_order():
    box1 = [4, 4, 5, 5]
    box2 = [1, 1, 3, 3]
    assert get_xyxy_boxes_iou(box1, box2) == 0


def test_max_boxes_iou():
    box = [1, 1, 3, 3]
    assert get_xyxy_boxes_iou(box, box) == 1


def test_nested_boxes_iou():
    box1 = [1, 1, 5, 5]
    box2 = [2, 2, 4, 4]
    assert get_xyxy_boxes_iou(box1, box2) == pytest.approx(1 / 4)


def test_nested_boxes_iou_reversed_order():
    box1 = [2, 2, 4, 4]
    box2 = [1, 1, 5, 5]
    assert get_xyxy_boxes_iou(box1, box2) == pytest.approx(1 / 4)
