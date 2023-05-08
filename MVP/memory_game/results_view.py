from __future__ import annotations
import arcade
from MVP.ui_const import SCREEN_WIDTH, SCREEN_HEIGHT
from arcade.application import Window
from MVP.game_core import GameCore
from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.gesture_detection import GestureDetection
from static_gesture_classification.static_gesture import StaticGesture
from typing import Optional, List, Dict, Generic, TypeVar
from MVP.data_structures.rect import Rect
from MVP.memory_game.memory_game_result_kind import MemoryGameResultKind
from MVP.draw_utils import draw_progression_as_rectangle_part, draw_rectangle
from MVP.time_based_gesture_options_tracker import TimeBasedGesturesOptionsTracker


class MemoryGameResultsView(arcade.View):
    def __init__(
        self,
        game_core: GameCore,
        game_manager: MemoryGameManager,
    ) -> None:
        super().__init__()
        self.game_core: GameCore = game_core
        self.game_manager: MemoryGameManager = game_manager
        self.main_sprite_list_name: str = "MemoryGameResultsView"
        self.ground_true_cards_positions: List[arcade.NamedPoint] = [
            arcade.NamedPoint(x=179.5, y=98),
            arcade.NamedPoint(x=305.5, y=98),
            arcade.NamedPoint(x=431.5, y=98),
            arcade.NamedPoint(x=557.5, y=98),
            arcade.NamedPoint(x=683.5, y=98),
        ]
        self.user_input_cards_positions: List[arcade.NamedPoint] = [
            arcade.NamedPoint(x=179.5, y=265),
            arcade.NamedPoint(x=305.5, y=265),
            arcade.NamedPoint(x=431.5, y=265),
            arcade.NamedPoint(x=557.5, y=265),
            arcade.NamedPoint(x=683.5, y=265),
        ]
        self.predictions_num_card_position: arcade.NamedPoint = arcade.NamedPoint(
            x=321, y=466
        )
        self.title_position_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=431.5, y=634
        )
        self._menu_options_confirmation_time_seconds: float = 2
        self.option_progress_bars_rectangles: Dict[str, Rect] = {
            "menu": Rect(top_left_x=1048, top_left_y=624, width=130, height=12),
            "replay": Rect(top_left_x=1048, top_left_y=520, width=130, height=12),
        }
        self._menu_options_tracker: TimeBasedGesturesOptionsTracker[
            str
        ] = TimeBasedGesturesOptionsTracker(
            gesture_to_option_mapping={
                StaticGesture.OKEY: "replay",
                StaticGesture.FIST: "menu",
            },
            required_time_seconds=self._menu_options_confirmation_time_seconds,
        )
        self.progress_bar_color: arcade.Color = [255, 145, 103]

    def _draw_current_option_progress_bar(self) -> None:
        active_option_selection: Optional[
            TimeBasedSelection[str]
        ] = self._menu_options_tracker.current_selection
        if active_option_selection is None:
            return
        draw_progression_as_rectangle_part(
            rectangle=self.option_progress_bars_rectangles[
                active_option_selection.selected_object
            ],
            progression_part=active_option_selection.proportion_of_completed_time,
            color=self.progress_bar_color,
        )

    def setup(
        self,
        user_gestures: List[StaticGesture],
        gt_gestures: List[StaticGesture],
        correct_answers_number: int,
        result_kind: MemoryGameResultKind,
    ) -> None:
        self._menu_options_tracker.reset()
        self.game_core.recreate_scene()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        # add all sprites
        self.game_core.scene.add_sprite_list(self.main_sprite_list_name)
        self.game_core.sprites_collection.memory_game_results_bg.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.memory_game_results_bg.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.memory_game_results_bg,
        )
        position: arcade.NamedPoint
        gesture: StaticGesture
        sprite: arcade.Sprite
        for position, gesture in zip(self.user_input_cards_positions, user_gestures):
            sprite = arcade.Sprite(
                filename=self.game_core.sprites_collection.not_active_gestures_cards_paths[
                    gesture
                ]
            )
            sprite.center_x = position.x
            sprite.center_y = position.y
            self.game_core.scene.add_sprite(self.main_sprite_list_name, sprite)

        for position, gesture in zip(self.ground_true_cards_positions, gt_gestures):
            sprite = arcade.Sprite(
                filename=self.game_core.sprites_collection.not_active_gestures_cards_paths[
                    gesture
                ]
            )
            sprite.center_x = position.x
            sprite.center_y = position.y
            self.game_core.scene.add_sprite(self.main_sprite_list_name, sprite)

        self.game_core.sprites_collection.memory_game_reults[
            result_kind
        ].center_x = self.title_position_arcade.x
        self.game_core.sprites_collection.memory_game_reults[
            result_kind
        ].center_y = self.title_position_arcade.y

        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.memory_game_reults[result_kind],
        )

        self.game_core.sprites_collection.numbers_cards[
            correct_answers_number
        ].center_x = self.predictions_num_card_position.x
        self.game_core.sprites_collection.numbers_cards[
            correct_answers_number
        ].center_y = self.predictions_num_card_position.y
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.numbers_cards[correct_answers_number],
        )

    def on_draw(self) -> None:
        self.clear()
        self.game_core.scene.draw()
        self._draw_current_option_progress_bar()
        self.game_core.draw_hands_detections_on_web_camera()

    def on_update(self, delta_time: float):
        self.game_core.update_inner_state(delta_time)
        self._menu_options_tracker.update_inner_state(
            active_gesture_selection=self.game_core.hand_detection_state.active_gesture_time_track
        )
        selected_menu_option: Optional[str] = self._menu_options_tracker.active_option
        if selected_menu_option == "menu":
            self.game_manager.return_to_menu()
        if selected_menu_option == "replay":
            self.game_manager.replay()


from MVP.memory_game.memory_game_manager import (
    MemoryGameManager,
)
