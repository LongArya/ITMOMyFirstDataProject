import arcade
from MVP.data_structures.rect import Rect


def draw_progression_as_rectangle_part(
    rectangle: Rect, progression_part: float, color: arcade.Color
):
    """Draws part of rectangle, such that drawn rectangle width / original width = progression_part"""
    arcade.draw_lrtb_rectangle_filled(
        left=rectangle.top_left_x,
        top=rectangle.top_left_y,
        bottom=rectangle.top_left_y - rectangle.height,
        right=rectangle.top_left_x + rectangle.width * progression_part,
        color=color,
    )
