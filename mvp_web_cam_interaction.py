import arcade
import hydra
import os
import numpy as np
import PIL
from PIL import Image
import cv2
from dataclasses import dataclass
from static_gesture_classification.static_gesture import StaticGesture
from typing import Optional, Dict, List
from enum import Enum, auto
import random
from yolo_detection import YoloInferece
from hydra.core.config_store import ConfigStore
from static_gesture_classification.config import StaticGestureConfig
from const import (
    STATIC_GESTURE_CFG_NAME,
    STATIC_GESTURE_CFG_ROOT,
    YOLO_V7_HAND_DETECTION,
    YOLO_V7_INPUT_RESOLUTION,
)
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
    init_augmentations_from_config,
)
from typing import Callable, List, Iterable
from PIL import Image
from static_gesture_classification.static_gesture import StaticGesture
from general.data_structures.data_split import DataSplit

os.environ["HYDRA_FULL_ERROR"] = "1"
cs = ConfigStore.instance()
cs.store(name=STATIC_GESTURE_CFG_NAME, node=StaticGestureConfig)

# Constants
SCREEN_WIDTH = 1500
SCREEN_HEIGHT = 1000
CAMERA_OUTPUT_W = 640
CAMERA_OUTPUT_H = 480
SCREEN_TITLE = "Rock Paper Scissors Game"
ORIGIN = arcade.NamedPoint(x=0, y=0)


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
    point: arcade.NamedPoint,
    current_origin: arcade.NamedPoint,
    new_origin: arcade.NamedPoint,
) -> arcade.NamedPoint:
    global_point: arcade.NamedPoint = arcade.NamedPoint(
        x=point.x + current_origin.x, y=point.y + current_origin.y
    )
    point_wrt_new_origin = arcade.NamedPoint(
        x=global_point.x - new_origin.x, y=global_point.y - new_origin.y
    )
    return point_wrt_new_origin


class WebCamInteractionDemo(arcade.View):
    def __init__(self, detector: YoloInferece) -> None:
        super().__init__()
        self.frame_count = 0
        self.detector = detector
        self.cap = cv2.VideoCapture(0)
        self.current_frame: Optional[np.ndarray] = None
        self.current_boxes: Optional[np.ndarray] = [[10, 10, 100, 100]]
        self.frame_top_left_opencv_position = arcade.NamedPoint(
            x=SCREEN_WIDTH - CAMERA_OUTPUT_W,
            y=SCREEN_HEIGHT - CAMERA_OUTPUT_H,
        )
        self.current_gestures: List[StaticGesture] = [StaticGesture.BACKGROUND]
        self.text = arcade.Text(
            text="TEST",
            start_x=SCREEN_WIDTH // 2,
            start_y=SCREEN_HEIGHT // 2,
            color=arcade.color.GREEN,
            font_size=20,
        )
        self.web_camera_input_texture = arcade.Texture.create_empty(
            name="web_camera_input", size=(CAMERA_OUTPUT_W, CAMERA_OUTPUT_H)
        )
        self.atlas = arcade.TextureAtlas.create_from_texture_sequence(
            [self.web_camera_input_texture]
        )

    def draw_text(self, text: str, point: arcade.NamedPoint):
        arcade.draw_text(
            text,
            start_x=point.x,
            start_y=point.y,
            color=arcade.color.GREEN,
            font_size=20,
            width=SCREEN_WIDTH,
            align="center",
        )

    def _draw_box_in_current_frame(self, xyxy_box: Iterable[float]):
        x1, y1, x2, y2 = xyxy_box
        left_top_point = arcade.NamedPoint(x=x1, y=y1)
        left_top_point = change_point_origin(
            left_top_point,
            current_origin=self.frame_top_left_opencv_position,
            new_origin=ORIGIN,
        )
        left_top_point = from_opencv_coordinate_system_to_arcade(
            left_top_point, width=SCREEN_WIDTH, height=SCREEN_HEIGHT
        )
        right_bottom_point = arcade.NamedPoint(x=x2, y=y2)
        right_bottom_point = change_point_origin(
            right_bottom_point,
            current_origin=self.frame_top_left_opencv_position,
            new_origin=ORIGIN,
        )
        right_bottom_point = from_opencv_coordinate_system_to_arcade(
            right_bottom_point, width=SCREEN_WIDTH, height=SCREEN_HEIGHT
        )
        arcade.draw_lrtb_rectangle_outline(
            left=left_top_point.x,
            right=right_bottom_point.x,
            top=left_top_point.y,
            bottom=right_bottom_point.y,
            color=arcade.color.GREEN,
            border_width=2,
        )

    def on_draw(self):
        self.clear()
        pil_image = Image.fromarray(self.current_frame)
        self.web_camera_input_texture.image = pil_image
        self.atlas.update_texture_image(self.web_camera_input_texture)
        # web_camera_input_texture = arcade.Texture(
        #     name=f"t_{self.frame_count}", image=pil_image
        # )
        frame_center_opencv = arcade.NamedPoint(
            x=self.frame_top_left_opencv_position.x + CAMERA_OUTPUT_W / 2,
            y=self.frame_top_left_opencv_position.y + CAMERA_OUTPUT_H / 2,
        )
        frame_center_arcade = from_opencv_coordinate_system_to_arcade(
            frame_center_opencv, width=SCREEN_WIDTH, height=SCREEN_HEIGHT
        )
        arcade.draw_texture_rectangle(
            center_x=frame_center_arcade.x,
            center_y=frame_center_arcade.y,
            width=CAMERA_OUTPUT_W,
            height=CAMERA_OUTPUT_H,
            texture=self.web_camera_input_texture,
        )
        image_coordinate_point = arcade.NamedPoint(10, 10)
        image_coordinate_point = change_point_origin(
            point=image_coordinate_point,
            current_origin=self.frame_top_left_opencv_position,
            new_origin=ORIGIN,
        )
        image_coordinate_point = from_opencv_coordinate_system_to_arcade(
            image_coordinate_point, width=SCREEN_WIDTH, height=SCREEN_HEIGHT
        )
        if self.current_boxes is not None:
            for box in self.current_boxes:
                self._draw_box_in_current_frame(box)

    def on_update(self, delta_time: float):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.fliplr(frame)
        if not ret:
            return
        # self.frame_count += 1
        self.current_frame = frame
        # if self.frame_count % 100 == 0:
        #     arcade.cleanup_texture_cache()
        outputs = self.detector(frame)
        self.current_boxes = outputs[:, :4]
        return super().on_update(delta_time)


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def main(cfg: StaticGestureConfig):
    """Main function"""
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    yolo_hand_detector = YoloInferece(
        model_path=YOLO_V7_HAND_DETECTION, input_resolution=YOLO_V7_INPUT_RESOLUTION
    )
    gesture_view = WebCamInteractionDemo(detector=yolo_hand_detector)
    window.show_view(gesture_view)
    arcade.run()


if __name__ == "__main__":
    list = arcade.TextureAtlas

    main()
