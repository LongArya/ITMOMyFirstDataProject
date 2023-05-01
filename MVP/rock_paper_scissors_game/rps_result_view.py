from __future__ import annotations
import arcade
from MVP.game_core_protocol import GameCoreProtocol
from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from static_gesture_classification.static_gesture import StaticGesture
from MVP.rock_paper_scissors_game.game_mechanics import (
    RPSGameOption,
    EndGameResult,
)
from typing import Dict, Optional
from MVP.ui_const import SCREEN_HEIGHT, SCREEN_WIDTH
from MVP.data_structures.gesture_detection import GestureDetection
from MVP.data_structures.rect import Rect
from MVP.draw_utils import draw_progression_as_rectangle_part
from MVP.time_based_gesture_options_tracker import TimeBasedGesturesOptionsTracker


class RPSResultsView(arcade.View):
    def __init__(
        self, game_core: GameCoreProtocol, game_manager: RockPaperScissorsGameManager
    ) -> None:
        super().__init__()
        self.game_core = game_core
        self.game_manager = game_manager
        self.hero_option: Optional[RPSGameOption] = None
        self.enemy_option: Optional[RPSGameOption] = None
        self.outcome: Optional[EndGameResult] = None
        self.main_sprite_list_name = "RPSResults"
        self.option_selection_time_requirement_seconds: float = 2
        self.hero_option_draw_position_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=163, y=341
        )
        self.enemy_option_draw_position_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=678, y=341
        )
        self.outcome_draw_position_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=421, y=583.5
        )
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
            required_time_seconds=self.option_selection_time_requirement_seconds,
        )
        self.option_progress_bar_color: arcade.Color = [255, 145, 103]

    def setup(
        self,
        hero_option: RPSGameOption,
        enemy_option: RPSGameOption,
        outcome: EndGameResult,
    ):
        self._menu_options_tracker.reset()
        self.hero_option = hero_option
        self.enemy_option = enemy_option
        self.outcome = outcome
        # scene
        self.game_core.recreate_scene()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        self.game_core.scene.add_sprite_list(self.main_sprite_list_name)
        # background
        self.game_core.sprites_collection.RPS_results_backgrond.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.RPS_results_backgrond.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.RPS_results_backgrond,
        )
        # hero option
        self.game_core.sprites_collection.RPS_not_selected_options[
            hero_option
        ].center_x = self.hero_option_draw_position_arcade.x
        self.game_core.sprites_collection.RPS_not_selected_options[
            hero_option
        ].center_y = self.hero_option_draw_position_arcade.y
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.RPS_not_selected_options[hero_option],
        )
        # enemy option
        self.game_core.sprites_collection.RPS_enemy_turn[
            enemy_option
        ].center_x = self.enemy_option_draw_position_arcade.x
        self.game_core.sprites_collection.RPS_enemy_turn[
            enemy_option
        ].center_y = self.enemy_option_draw_position_arcade.y
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.RPS_enemy_turn[enemy_option],
        )
        # result title
        self.game_core.sprites_collection.RPS_outcomes[
            outcome
        ].center_x = self.outcome_draw_position_arcade.x
        self.game_core.sprites_collection.RPS_outcomes[
            outcome
        ].center_y = self.outcome_draw_position_arcade.y
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.RPS_outcomes[outcome],
        )
        # web camera
        self.game_core.setup_web_camera_preview_in_scene()

    def _draw_current_option_selection(self) -> None:
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
            color=self.option_progress_bar_color,
        )

    def on_draw(self) -> None:
        self.clear()
        self.game_core.scene.draw()
        self._draw_current_option_selection()
        if self.game_core.hand_detection_state.active_track is not None:
            gesture_detection: GestureDetection = (
                self.game_core.hand_detection_state.active_track.last.object
            )
            self.game_core.draw_gesture_detection_in_web_camera(
                gesture_detection=gesture_detection, active=True
            )

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


from MVP.rock_paper_scissors_game.game_manager import RockPaperScissorsGameManager
