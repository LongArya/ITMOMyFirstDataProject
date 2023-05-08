from __future__ import annotations
import arcade
from MVP.game_core import GameCore
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


class RPSGestureSelectionView(arcade.View):
    def __init__(
        self, game_core: GameCore, game_manager: RockPaperScissorsGameManager
    ) -> None:
        super().__init__()
        self.game_core: GameCore = game_core
        self.game_manager: RockPaperScissorsGameManager = game_manager
        self.main_sprite_list_name: str = "RPSGestureSelection"
        self.rps_option_selection_time_requirement_seconds: float = 5
        self.rps_option_tracker: TimeBasedGesturesOptionsTracker[
            RPSGameOption
        ] = TimeBasedGesturesOptionsTracker(
            gesture_to_option_mapping={
                StaticGesture.TWO: RPSGameOption.SCISSORS,
                StaticGesture.FIVE: RPSGameOption.PAPER,
                StaticGesture.FIST: RPSGameOption.ROCK,
            },
            required_time_seconds=self.rps_option_selection_time_requirement_seconds,
        )
        self.rps_draw_positions_arcade: Dict[RPSGameOption, arcade.NamedPoint] = {
            RPSGameOption.PAPER: arcade.NamedPoint(x=163, y=340),
            RPSGameOption.SCISSORS: arcade.NamedPoint(x=420, y=340),
            RPSGameOption.ROCK: arcade.NamedPoint(x=678, y=340),
        }
        self.progress_bar_rect_arcade: Rect = Rect(
            top_left_x=72, top_left_y=631, width=699, height=37
        )
        self.progress_bar_color: arcade.Color = [255, 145, 103]

    def _add_not_selected_rps_options_to_scene(self) -> None:
        for rps_option in RPSGameOption:
            target_sprite: arcade.Sprite = (
                self.game_core.sprites_collection.RPS_not_selected_options[rps_option]
            )
            target_sprite.center_x = self.rps_draw_positions_arcade[rps_option].x
            target_sprite.center_y = self.rps_draw_positions_arcade[rps_option].y
            self.game_core.scene.add_sprite(self.main_sprite_list_name, target_sprite)

    def setup(self) -> None:
        self.rps_option_tracker.reset()
        self.game_core.recreate_scene()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        self.game_core.sprites_collection.RPS_geture_selection_background.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.RPS_geture_selection_background.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite_list(self.main_sprite_list_name)
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.RPS_geture_selection_background,
        )
        self._add_not_selected_rps_options_to_scene()
        # web camera
        self.game_core.setup_web_camera_preview_in_scene()

    def _update_scene_based_on_selected_rps_option(self) -> None:
        # remove all rps options sprites
        sprite: arcade.Sprite
        for sprite in self.game_core.sprites_collection.RPS_selected_options.values():
            sprite.remove_from_sprite_lists()
        for (
            sprite
        ) in self.game_core.sprites_collection.RPS_not_selected_options.values():
            sprite.remove_from_sprite_lists()

        if self.rps_option_tracker.current_selection is None:
            self._add_not_selected_rps_options_to_scene()
            return

        # add sprites
        rps_option: RPSGameOption
        for rps_option in RPSGameOption:
            target_sprite: arcade.Sprite = (
                (self.game_core.sprites_collection.RPS_selected_options[rps_option])
                if rps_option
                == self.rps_option_tracker.current_selection.selected_object
                else self.game_core.sprites_collection.RPS_not_selected_options[
                    rps_option
                ]
            )
            target_sprite.center_x = self.rps_draw_positions_arcade[rps_option].x
            target_sprite.center_y = self.rps_draw_positions_arcade[rps_option].y
            self.game_core.scene.add_sprite(self.main_sprite_list_name, target_sprite)

    def _draw_current_rps_selection_progress(self) -> None:
        if self.rps_option_tracker.current_selection is None:
            return
        draw_progression_as_rectangle_part(
            rectangle=self.progress_bar_rect_arcade,
            progression_part=self.rps_option_tracker.current_selection.proportion_of_completed_time,
            color=self.progress_bar_color,
        )

    def on_draw(self) -> None:
        self.clear()
        self.game_core.scene.draw()
        self._draw_current_rps_selection_progress()
        self.game_core.draw_hands_detections_on_web_camera()

    def on_update(self, delta_time: float):
        self.game_core.update_inner_state(delta_time)
        self.rps_option_tracker.update_inner_state(
            active_gesture_selection=self.game_core.hand_detection_state.active_gesture_time_track
        )
        self._update_scene_based_on_selected_rps_option()
        selected_rps_object: Optional[
            RPSGameOption
        ] = self.rps_option_tracker.active_option
        if selected_rps_object is None:
            return
        self.game_manager.switch_to_results(hero_option=selected_rps_object)


from MVP.rock_paper_scissors_game.game_manager import RockPaperScissorsGameManager
