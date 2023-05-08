from __future__ import annotations
import arcade
from MVP.ui_const import SCREEN_WIDTH, SCREEN_HEIGHT
from arcade.application import Window
from MVP.game_core import GameCore
from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.gesture_detection import GestureDetection
from static_gesture_classification.static_gesture import StaticGesture
from typing import Optional, List, Dict
from MVP.data_structures.rect import Rect
from MVP.draw_utils import draw_progression_as_rectangle_part, draw_rectangle


class MemoryGameGesturesDemonstrationView(arcade.View):
    def __init__(
        self,
        game_core: GameCore,
        game_manager: MemoryGameManager,
    ) -> None:
        super().__init__()
        self.game_core: GameCore = game_core
        self.game_manager: MemoryGameManager = game_manager
        self.shown_gesture_index = MEMORIZATION_GAME_GESTURES_NUM - 1
        self.current_gesture_selection: Optional[
            TimeBasedSelection[StaticGesture]
        ] = None
        self.progress_bar_rectangles: List[Rect] = [
            Rect(top_left_x=126, top_left_y=483, width=107, height=17),
            Rect(top_left_x=252, top_left_y=483, width=107, height=17),
            Rect(top_left_x=379, top_left_y=483, width=107, height=17),
            Rect(top_left_x=504, top_left_y=483, width=107, height=17),
            Rect(top_left_x=630, top_left_y=483, width=107, height=17),
        ]
        self.cards_center_locations: List[arcade.NamedPoint] = [
            arcade.NamedPoint(x=179.5, y=377),
            arcade.NamedPoint(x=305.5, y=377),
            arcade.NamedPoint(x=431.5, y=377),
            arcade.NamedPoint(x=557.5, y=377),
            arcade.NamedPoint(x=683.5, y=377),
        ]
        self.gesture_demonstration_time_requirement_seconds: float = 3
        self.shown_gestures: List[StaticGesture] = [
            None for _ in range(MEMORIZATION_GAME_GESTURES_NUM)
        ]
        self.main_sprite_list_name: str = "MemoryGameGesturesDemonstrationView"
        self.time_progress_bar_color: arcade.Color = [255, 145, 103]
        self.time_progress_bar_background_color: arcade.Color = [248, 241, 215]
        self.not_active_sprites_for_different_positions: Dict[
            StaticGesture, List[arcade.Sprite]
        ] = {
            gesture: [
                arcade.Sprite(
                    filename=self.game_core.sprites_collection.not_active_gestures_cards_paths[
                        gesture
                    ]
                )
                for _ in self.cards_center_locations
            ]
            for gesture in StaticGesture
        }
        self.active_sprites_for_different_positions: Dict[
            StaticGesture, List[arcade.Sprite]
        ] = {
            gesture: [
                arcade.Sprite(
                    filename=self.game_core.sprites_collection.active_gestures_cards_paths[
                        gesture
                    ]
                )
                for _ in self.cards_center_locations
            ]
            for gesture in StaticGesture
        }

    def _draw_current_gesture_progress_bar(self) -> None:
        if self.current_gesture_selection is None:
            return
        draw_progression_as_rectangle_part(
            rectangle=self.progress_bar_rectangles[self.shown_gesture_index],
            color=self.time_progress_bar_color,
            progression_part=self.current_gesture_selection.proportion_of_completed_time,
        )

    def _draw_progress_bars_backgrounds(self) -> None:
        for i, progress_bar_rectgangle in enumerate(self.progress_bar_rectangles):
            current_color: arcade.Color
            if i <= self.shown_gesture_index:
                current_color = self.time_progress_bar_background_color
            else:
                current_color = self.time_progress_bar_color
            draw_rectangle(rectangle=progress_bar_rectgangle, color=current_color)

    def _update_scene_cards(self) -> None:
        sprite: arcade.Sprite
        sprite_list: List[arcade.Sprite]
        gesture: StaticGesture
        target_sprite: arcade.Sprite

        for sprite_list in self.not_active_sprites_for_different_positions.values():
            for sprite in sprite_list:
                sprite.remove_from_sprite_lists()
        for sprite_list in self.active_sprites_for_different_positions.values():
            for sprite in sprite_list:
                sprite.remove_from_sprite_lists()
        for i, sprite_position in enumerate(self.cards_center_locations):
            if i < self.shown_gesture_index:
                target_sprite = self.not_active_sprites_for_different_positions[
                    StaticGesture.BACKGROUND
                ][i]
            if i == self.shown_gesture_index:
                if self.current_gesture_selection is None:
                    gesture = StaticGesture.BACKGROUND
                else:
                    gesture = self.current_gesture_selection.selected_object
                target_sprite = self.not_active_sprites_for_different_positions[
                    gesture
                ][i]
            if i > self.shown_gesture_index:
                gesture = self.shown_gestures[i]
                target_sprite = self.active_sprites_for_different_positions[gesture][i]

            target_sprite.center_x = sprite_position.x
            target_sprite.center_y = sprite_position.y
            self.game_core.scene.add_sprite(
                self.main_sprite_list_name,
                target_sprite,
            )

    def _update_current_gesture_selection(self) -> None:
        active_geture_selection: Optional[
            TimeTrackedEntity[StaticGesture]
        ] = self.game_core.hand_detection_state.active_gesture_time_track
        if active_geture_selection is None:
            self.current_gesture_selection = None
            return
        shown_gesture: StaticGesture = active_geture_selection.entity
        if shown_gesture == StaticGesture.BACKGROUND:
            self.current_gesture_selection = None
            return
        self.current_gesture_selection = TimeBasedSelection(
            selected_object=shown_gesture,
            accumulated_time_seconds=active_geture_selection.tracked_time_seconds,
            time_requirement_seconds=self.gesture_demonstration_time_requirement_seconds,
        )

    def setup(self) -> None:
        self.shown_gesture_index = MEMORIZATION_GAME_GESTURES_NUM - 1
        self.shown_gestures = [None for _ in range(MEMORIZATION_GAME_GESTURES_NUM)]
        self.game_core.recreate_scene()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        self.game_core.sprites_collection.memory_game_gesture_demo_bg.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.memory_game_gesture_demo_bg.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.memory_game_gesture_demo_bg,
        )

    def on_draw(self) -> None:
        self.clear()
        self.game_core.scene.draw()
        self._draw_progress_bars_backgrounds()
        self._draw_current_gesture_progress_bar()
        self.game_core.draw_hands_detections_on_web_camera()

    def on_update(self, delta_time: float) -> None:
        self.game_core.update_inner_state(delta_time)
        self._update_current_gesture_selection()
        self._update_scene_cards()
        if self.current_gesture_selection is None:
            return
        if self.current_gesture_selection.is_active:
            self.game_core.hand_detection_state.nullify_gesture_selection()
            self.shown_gestures[
                self.shown_gesture_index
            ] = self.current_gesture_selection.selected_object
            self.shown_gesture_index -= 1
        if self.shown_gesture_index < 0:
            self.game_manager.go_to_results(user_gestures=self.shown_gestures)


from MVP.memory_game.memory_game_manager import (
    MemoryGameManager,
    MEMORIZATION_GAME_GESTURES_NUM,
)
