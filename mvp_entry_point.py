from __future__ import annotations
import cv2
import logging
import arcade
import numpy as np
from const import (
    YOLO_V7_HAND_DETECTION,
    YOLO_V7_INPUT_RESOLUTION,
)
from MVP.yolo_detection import YoloInferece
from MVP.sprite_collection import SpriteCollection
from MVP.ui_const import GAME_RESOUSES_DIR, SCREEN_HEIGHT, SCREEN_WIDTH
from MVP.geometry_utils import (
    change_point_origin,
    from_opencv_coordinate_system_to_arcade,
    project_point_to_rectangle,
    get_box_center,
)
from typing import Optional, Any
from MVP.hands_detection_state import HandDetectionState
from MVP.data_structures.gesture_detection import GestureDetection
from MVP.menu_manager_view import MenuManagerView

arcade.configure_logging(level=logging.ERROR)
from MVP.onnx_static_gesture_classifier import ONNXResnet18StaticGestureClassifier


class GameCorePlaceHolder:
    def __init__(
        self,
        hand_detection: HandDetectionState,
        sprites_collection: SpriteCollection,
        display_camera: bool,
    ):
        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        self.camera_frame_height, self.camera_frame_width = frame.shape[:2]
        self.hand_detection_state = hand_detection
        self.sprites_collection = sprites_collection
        self.frame_number = 0
        self.scene = arcade.Scene()
        self.display_camera = display_camera
        self.window: Optional[Any] = None
        self.window_name: Optional[str] = None
        if self.display_camera:
            self.window_name = "WebCameraDemo"
            self.window = cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.web_camera_sprite_list_name = "web_camera_preview"

    def recreate_scene(self):
        self.scene = arcade.Scene()

    def update_inner_state(self, time_delta: float) -> None:
        ret, bgr_frame = self.cap.read()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.fliplr(rgb_frame)
        if self.display_camera:
            cv2.imshow(self.window_name, np.fliplr(bgr_frame))
        self.hand_detection_state.update_inner_state(
            image=rgb_frame, time_delta=time_delta, frame_number=self.frame_number
        )
        self.frame_number += 1

    def setup_web_camera_preview_in_scene(self) -> None:
        self.scene.add_sprite_list(self.web_camera_sprite_list_name)
        self.sprites_collection.web_camera_preview.center_x = (
            SCREEN_WIDTH - self.sprites_collection.web_camera_preview.width // 2
        )
        self.sprites_collection.web_camera_preview.center_y = (
            self.sprites_collection.web_camera_preview.height // 2
        )
        self.scene.add_sprite(
            name=self.web_camera_sprite_list_name,
            sprite=self.sprites_collection.web_camera_preview,
        )

    def _remove_gestures_sprites_from_web_camera_preview(self) -> None:
        gesture_sprite: arcade.Sprite
        for gesture_sprite in self.sprites_collection.active_gestures_sprites.values():
            gesture_sprite.remove_from_sprite_lists()
        for (
            gesture_sprite
        ) in self.sprites_collection.not_active_gestures_sprites.values():
            gesture_sprite.remove_from_sprite_lists()

    def draw_gesture_detection_in_web_camera(
        self, gesture_detection: GestureDetection, active: bool
    ) -> None:
        self._remove_gestures_sprites_from_web_camera_preview()
        # select_target_sprite
        target_sprite: arcade.Sprite
        if active:
            target_sprite = self.sprites_collection.active_gestures_sprites[
                gesture_detection.gesture
            ]
        else:
            target_sprite = self.sprites_collection.not_active_gestures_sprites[
                gesture_detection.gesture
            ]
        # count gesture position in web camera preview
        box_center: arcade.NamedPoint = get_box_center(
            xyxy_box=gesture_detection.xyxy_box
        )
        box_center = project_point_to_rectangle(
            point=box_center,
            projection_dimensions=arcade.NamedPoint(
                x=self.sprites_collection.web_camera_preview.width,
                y=self.sprites_collection.web_camera_preview.height,
            ),
            original_space_dimensions=arcade.NamedPoint(
                x=self.camera_frame_width, y=self.camera_frame_height
            ),
        )
        box_center = change_point_origin(
            opencv_like_point=box_center,
            current_origin=arcade.NamedPoint(
                x=SCREEN_WIDTH - self.sprites_collection.web_camera_preview.width,
                y=SCREEN_HEIGHT - self.sprites_collection.web_camera_preview.height,
            ),  # FIXME temporary
            new_origin=arcade.NamedPoint(0, 0),
        )
        box_center = from_opencv_coordinate_system_to_arcade(
            point_in_opencv_system=box_center, width=SCREEN_WIDTH, height=SCREEN_HEIGHT
        )
        target_sprite.center_x = box_center.x
        target_sprite.center_y = box_center.y
        self.scene.add_sprite(self.web_camera_sprite_list_name, target_sprite)


def run_mvp():
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, "Demo")
    yolo_hand_detector = YoloInferece(
        model_path=YOLO_V7_HAND_DETECTION, input_resolution=YOLO_V7_INPUT_RESOLUTION
    )
    gesture_classifier = ONNXResnet18StaticGestureClassifier(
        "checkpoint_epoch=49-val_weighted_F1=0.88.ckpt.onnx"
    )
    hands_det_state = HandDetectionState(
        gesture_classifier=gesture_classifier,
        hand_detector=yolo_hand_detector,
        tracks_buffer_size=20,
    )
    game_core = GameCorePlaceHolder(
        hand_detection=hands_det_state,
        sprites_collection=SpriteCollection(GAME_RESOUSES_DIR),
        display_camera=True,
    )
    menu_view = MenuManagerView(game_core=game_core)
    menu_view.setup()
    window.show_view(menu_view)
    arcade.run()


if __name__ == "__main__":
    run_mvp()
