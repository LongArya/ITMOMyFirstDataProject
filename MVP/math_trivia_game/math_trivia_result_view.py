from __future__ import annotations
import arcade
from MVP.ui_const import SCREEN_WIDTH, SCREEN_HEIGHT
from arcade.application import Window
from MVP.game_core_protocol import GameCoreProtocol
from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.gesture_detection import GestureDetection
from static_gesture_classification.static_gesture import StaticGesture
from MVP.time_based_gesture_options_tracker import TimeBasedGesturesOptionsTracker
from MVP.draw_utils import draw_progression_as_rectangle_part
from MVP.data_structures.rect import Rect
from typing import Optional, List, Dict


class MathTriviaResultsView(arcade.View):
    def __init__(
        self, game_core: GameCoreProtocol, game_manager: MathTriviaGameManager
    ) -> None:
        super().__init__()
        self.game_core = game_core
        self.game_manager = game_manager
        self.result_text: arcade.Text
        self.result_text_center: arcade.NamedPoint = arcade.NamedPoint(x=430, y=287)
        self.text_font_size: int = 60
        self.text_color: arcade.Color = [56, 55, 50]

        self.option_progress_bars_rectangles: Dict[str, Rect] = {
            "menu": Rect(top_left_x=1048, top_left_y=624, width=130, height=12),
            "replay": Rect(top_left_x=1048, top_left_y=520, width=130, height=12),
        }
        self._menu_options_confirmation_time_seconds: float = 2
        self._menu_options_tracker: TimeBasedGesturesOptionsTracker[
            str
        ] = TimeBasedGesturesOptionsTracker(
            gesture_to_option_mapping={
                StaticGesture.OKEY: "replay",
                StaticGesture.FIST: "menu",
            },
            required_time_seconds=self._menu_options_confirmation_time_seconds,
        )
        self.current_option_selection: Optional[TimeBasedSelection[str]] = None
        self.main_sprite_list_name: str = "MathTriviaResultsView"
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

    def _setup_title_text_object(self, text: str) -> None:
        self.result_text = arcade.Text(
            text=text,
            start_x=self.result_text_center.x,
            start_y=self.result_text_center.y,
            color=self.text_color,
            font_size=self.text_font_size,
        )
        self.result_text.x -= self.result_text.content_width / 2

    def setup(self, result_text: str) -> None:
        self.game_core.recreate_scene()
        self._setup_title_text_object(result_text)
        self._menu_options_tracker.reset()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        self.game_core.scene.add_sprite_list(self.main_sprite_list_name)
        self.game_core.sprites_collection.math_trivia_results_background.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.math_trivia_results_background.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.math_trivia_results_background,
        )

    def on_draw(self) -> None:
        self.clear()
        self.game_core.scene.draw()
        self.result_text.draw()
        self._draw_current_option_progress_bar()
        if self.game_core.hand_detection_state.active_track is not None:
            gesture_detection: GestureDetection = (
                self.game_core.hand_detection_state.active_track.last.object
            )
            self.game_core.draw_gesture_detection_in_web_camera(
                gesture_detection=gesture_detection, active=True
            )

    def on_update(self, delta_time: float) -> None:
        self.game_core.update_inner_state(delta_time)
        self._menu_options_tracker.update_inner_state(
            active_gesture_selection=self.game_core.hand_detection_state.active_gesture_time_track
        )
        selected_menu_option: Optional[str] = self._menu_options_tracker.active_option
        if selected_menu_option == "menu":
            self.game_manager.return_to_menu()
        if selected_menu_option == "replay":
            self.game_manager.replay()


from MVP.math_trivia_game.math_trivia_game_manager import MathTriviaGameManager
