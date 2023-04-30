import arcade
from typing import Protocol
from MVP.hands_detection_state import HandDetectionState
from MVP.data_structures.gesture_detection import GestureDetection
from MVP.sprite_collection import SpriteCollection


class GameCoreProtocol(Protocol):
    hand_detection_state: HandDetectionState
    sprites_collection: SpriteCollection
    scene: arcade.Scene
    camera_frame_height: int
    camera_frame_width: int

    def recreate_scene(self):
        pass

    def update_inner_state(self, time_delta: float) -> None:
        pass

    def setup_web_camera_preview_in_scene(self) -> None:
        pass

    def draw_gesture_detection_in_web_camera(
        self, gesture_detection: GestureDetection, active: bool
    ) -> None:
        pass
