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


class MemoryGameManager:
    pass


class MemoryGamePreparationView(arcade.View):
    def __init__(self, game_core: GameCoreProtocol, game_manager: MemoryGameManager):
        self.game_core = game_core
        self.preparation_confirmation_selection: Optional[
            TimeBasedSelection[StaticGesture]
        ] = None
        self.preparation_confirmation_time_seconds: float = 2
        self.confirmation_progress_bar_rectangle_arcade: Rect = Rect(
            top_left_x=1048, top_left_y=593, height=12, width=130
        )
        self.confirmation_progress_bar_color: arcade.Color = [255, 145, 103]
        self.main_sprite_list_name: str = "MemGamePrepStage"
        super().__init__()

    def setup(self) -> None:
        self.game_core.recreate_scene()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        self.game_core.scene.add_sprite_list(self.main_sprite_list_name)
        self.game_core.sprites_collection.memory_game_preparation_bg.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.memory_game_preparation_bg.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.memory_game_preparation_bg,
        )

    def _draw_preparation_confirmation_status(self) -> None:
        if self.preparation_confirmation_selection is None:
            return
        draw_progression_as_rectangle_part(
            rectangle=self.confirmation_progress_bar_rectangle_arcade,
            progression_part=self.preparation_confirmation_selection.proportion_of_completed_time,
            color=self.confirmation_progress_bar_color,
        )

    def on_draw(self) -> None:
        self.clear()
        self.game_core.scene.draw()
        self._draw_preparation_confirmation_status()
        if self.game_core.hand_detection_state.active_track is not None:
            gesture_detection: GestureDetection = (
                self.game_core.hand_detection_state.active_track.last.object
            )
            self.game_core.draw_gesture_detection_in_web_camera(
                gesture_detection=gesture_detection, active=True
            )

    def _update_preparation_confimation(self) -> None:
        active_geture_selection: Optional[
            TimeTrackedEntity[StaticGesture]
        ] = self.game_core.hand_detection_state.active_gesture_time_track
        if active_geture_selection is None:
            self.preparation_confirmation_selection = None
            return
        if active_geture_selection.entity != StaticGesture.OKEY:
            self.preparation_confirmation_selection = None
            return
        self.preparation_confirmation_selection = TimeBasedSelection(
            selected_object=StaticGesture.OKEY,
            accumulated_time_seconds=active_geture_selection.tracked_time_seconds,
            time_requirement_seconds=self.preparation_confirmation_time_seconds,
        )

    def on_update(self, delta_time: float):
        self.game_core.update_inner_state(delta_time)
        self._update_preparation_confimation()
