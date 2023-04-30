from __future__ import annotations
import arcade
from MVP.ui_const import SCREEN_WIDTH, SCREEN_HEIGHT
from arcade.application import Window
from MVP.game_core_protocol import GameCoreProtocol
from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.gesture_detection import GestureDetection
from static_gesture_classification.static_gesture import StaticGesture
from typing import Optional
from MVP.data_structures.rect import Rect
from MVP.draw_utils import draw_progression_as_rectangle_part
from typing import List


class MemoryGameMemorizationStage(arcade.View):
    def __init__(
        self,
        game_core: GameCoreProtocol,
        game_manager: MemoryGameManager,
        memorization_gestures: List[StaticGesture],
    ):
        super().__init__()
        self.game_core = game_core
        self.menu_manager = game_manager
        self.memorization_gestures: List[StaticGesture] = memorization_gestures
        self.main_sprite_list_name: str = "MemoryGameMemorizationStage"
        self.memorization_cards_centers_arcade: List[arcade.NamedPoint] = [
            arcade.NamedPoint(x=179.5, y=377),
            arcade.NamedPoint(x=305.5, y=377),
            arcade.NamedPoint(x=431.5, y=377),
            arcade.NamedPoint(x=557.5, y=377),
            arcade.NamedPoint(x=683.5, y=377),
        ]
        self.time_progress_bar_rectangle: Rect = Rect(
            top_left_x=191, top_left_y=185, width=547, height=53
        )
        self.time_progress_bar_color: arcade.Color = [255, 145, 103]
        self.memorization_time_limit_seconds = 20
        self.spent_time_seconds: float = 0

    def setup(self, memorization_gestures: List[StaticGesture]) -> None:
        self.memorization_gestures = memorization_gestures
        self.spent_time_seconds = 0
        self.game_core.recreate_scene()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        self.game_core.scene.add_sprite_list(self.main_sprite_list_name)
        self.game_core.sprites_collection.memory_game_memorization_bg.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.memory_game_memorization_bg.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.memory_game_memorization_bg,
        )
        gesture: StaticGesture
        draw_position: arcade.NamedPoint
        for gesture, draw_position in zip(
            self.memorization_gestures, self.memorization_cards_centers_arcade
        ):
            self.game_core.sprites_collection.not_active_gestures_cards_paths[
                gesture
            ].center_x = draw_position.x
            self.game_core.sprites_collection.not_active_gestures_cards_paths[
                gesture
            ].center_y = draw_position.y
            self.game_core.scene.add_sprite(
                self.main_sprite_list_name,
                self.game_core.sprites_collection.not_active_gestures_cards_paths[
                    gesture
                ],
            )

    def draw_left_time_progress_bar(self) -> None:
        spent_time_proportion: float = (
            self.spent_time_seconds / self.memorization_time_limit_seconds
        )
        spent_time_proportion = min(spent_time_proportion, 1)
        left_time_proportion: float = 1 - spent_time_proportion
        draw_progression_as_rectangle_part(
            rectangle=self.time_progress_bar_rectangle,
            progression_part=left_time_proportion,
            color=self.time_progress_bar_color,
        )

    def on_draw(self):
        self.clear()
        self.game_core.scene.draw()
        self.draw_left_time_progress_bar()
        if self.game_core.hand_detection_state.active_track is not None:
            gesture_detection: GestureDetection = (
                self.game_core.hand_detection_state.active_track.last.object
            )
            self.game_core.draw_gesture_detection_in_web_camera(
                gesture_detection=gesture_detection, active=True
            )

    def on_update(self, delta_time: float):
        self.game_core.update_inner_state(delta_time)
        self.spent_time_seconds += delta_time


from MVP.memory_game.memory_game_manager import MemoryGameManager
