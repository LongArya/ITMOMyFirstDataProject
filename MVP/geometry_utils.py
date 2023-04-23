import arcade
from typing import Iterable


def from_opencv_coordinate_system_to_arcade(
    point_in_opencv_system: arcade.NamedPoint, width: int, height: int
) -> arcade.NamedPoint:
    """Converts point from opencv like system with origin being in top left corner,
    to arcade coordinate system where origin is at bottom right corner"""
    point_in_arcade_system = arcade.NamedPoint(
        x=point_in_opencv_system.x, y=height - point_in_opencv_system.y
    )
    return point_in_arcade_system


def change_point_origin(
    opencv_like_point: arcade.NamedPoint,
    current_origin: arcade.NamedPoint,
    new_origin: arcade.NamedPoint,
) -> arcade.NamedPoint:
    global_point: arcade.NamedPoint = arcade.NamedPoint(
        x=opencv_like_point.x + current_origin.x,
        y=opencv_like_point.y + current_origin.y,
    )
    point_wrt_new_origin = arcade.NamedPoint(
        x=global_point.x - new_origin.x, y=global_point.y - new_origin.y
    )
    return point_wrt_new_origin


def project_point_to_rectangle(
    point: arcade.NamedPoint,
    projection_dimensions: arcade.NamedPoint,
    original_space_dimensions: arcade.NamedPoint,
) -> arcade.NamedPoint:
    norm_x, norm_y = (
        point.x / original_space_dimensions.x,
        point.y / original_space_dimensions.y,
    )
    projected_space_x, projected_space_y = (
        norm_x * projection_dimensions.x,
        norm_y * projection_dimensions.y,
    )
    projection: arcade.NamedPoint = arcade.NamedPoint(
        x=projected_space_x, y=projected_space_y
    )
    return projection


def get_box_center(xyxy_box: Iterable[float]) -> arcade.NamedPoint:
    x1, y1, x2, y2 = xyxy_box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    bbox_center = arcade.NamedPoint(x=x_center, y=y_center)
    return bbox_center
