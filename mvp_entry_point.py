from __future__ import annotations
import logging
import arcade
from const import (
    YOLO_V7_ONNX_HAND_DETECTION,
    YOLO_V7_INPUT_RESOLUTION,
    RESNET18_ONNX_CLASSIFIER,
)
from MVP.onnx_networks.models.yolo_detection import YoloInferece
from MVP.sprite_collection import SpriteCollection
from MVP.ui_const import GAME_RESOUSES_DIR, SCREEN_HEIGHT, SCREEN_WIDTH
from MVP.hands_detection_state import HandDetectionState
from MVP.menu_manager_view import MenuManagerView

arcade.configure_logging(level=logging.ERROR)
from MVP.onnx_networks.models.onnx_resnet18_static_gesture_classifier import (
    ONNXResnet18StaticGestureClassifier,
)
from MVP.game_core import GameCore


def run_mvp() -> None:
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, "Demo")
    yolo_hand_detector = YoloInferece(
        model_path=YOLO_V7_ONNX_HAND_DETECTION,
        input_resolution=YOLO_V7_INPUT_RESOLUTION,
    )
    gesture_classifier = ONNXResnet18StaticGestureClassifier(RESNET18_ONNX_CLASSIFIER)
    hands_det_state = HandDetectionState(
        gesture_classifier=gesture_classifier,
        hand_detector=yolo_hand_detector,
        tracks_buffer_size=20,
    )
    game_core = GameCore(
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
