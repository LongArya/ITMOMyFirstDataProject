from __future__ import annotations
import arcade
from MVP.ui_const import SCREEN_WIDTH, SCREEN_HEIGHT
from arcade.application import Window
from MVP.game_core import GameCore
from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.gesture_detection import GestureDetection
from static_gesture_classification.static_gesture import StaticGesture
from MVP.time_based_gesture_options_tracker import TimeBasedGesturesOptionsTracker
from MVP.draw_utils import draw_progression_as_rectangle_part
from MVP.data_structures.rect import Rect
from typing import Optional, List, Dict
from MVP.game_core import GameCore
from MVP.math_trivia_game.trivia_question import TriviaQuestion


class MathQuestionView(arcade.View):
    def __init__(
        self, game_core: GameCore, game_manager: MathTriviaGameManager
    ) -> None:
        super().__init__()
        self.trivia_question: TriviaQuestion
        self.text_objects: List[arcade.Text]
        self.option_confirmation_time_seconds: float = 2
        self.answers_options_tracker: TimeBasedGesturesOptionsTracker[
            int
        ] = TimeBasedGesturesOptionsTracker(
            gesture_to_option_mapping={
                StaticGesture.ONE: 1,
                StaticGesture.TWO: 2,
                StaticGesture.THREE: 3,
                StaticGesture.FOUR: 4,
            },
            required_time_seconds=self.option_confirmation_time_seconds,
        )
        self.menu_options_tracker: TimeBasedGesturesOptionsTracker[
            str
        ] = TimeBasedGesturesOptionsTracker(
            gesture_to_option_mapping={StaticGesture.FIST: "end_game"},
            required_time_seconds=self.option_confirmation_time_seconds,
        )
        self.answers_progress_bar_rectangles: Dict[int, Rect] = {
            1: Rect(top_left_x=187, top_left_y=316, height=11, width=159),
            2: Rect(top_left_x=537, top_left_y=316, height=11, width=159),
            3: Rect(top_left_x=187, top_left_y=176, height=11, width=159),
            4: Rect(top_left_x=537, top_left_y=176, height=11, width=159),
        }
        self.question_text_position: arcade.NamedPoint = arcade.NamedPoint(x=420, y=500)
        self.answers_centers_positions: List[arcade.NamedPoint] = [
            arcade.NamedPoint(x=263, y=256),
            arcade.NamedPoint(x=613, y=256),
            arcade.NamedPoint(x=263, y=113),
            arcade.NamedPoint(x=613, y=113),
        ]
        self.menu_options_progress_bar_rectgangles: Dict[str, Rect] = {
            "end_game": Rect(top_left_x=1048, top_left_y=584, height=12, width=130)
        }
        self.game_manager: MathTriviaGameManager = game_manager
        self.game_core: GameCore = game_core
        self.main_sprite_list_name: str = "MathQuestionView"
        self.progress_bar_color: arcade.Color = [255, 145, 103]
        self.text_color: arcade.Color = [56, 55, 50]
        self.question_font_size: int = 42
        self.text_font_size: int = 36

    def _init_text_objects_from_trivia_question(
        self, trivia_question: TriviaQuestion
    ) -> None:
        self.text_objects = []
        question_text = arcade.Text(
            text=trivia_question.question_text,
            start_x=self.question_text_position.x,
            start_y=self.question_text_position.y,
            color=self.text_color,
            font_size=self.question_font_size,
        )
        question_text.x -= question_text.content_width / 2
        self.text_objects.append(question_text)

        for position, possible_answer in zip(
            self.answers_centers_positions, trivia_question.all_answers
        ):
            text_object = arcade.Text(
                text=possible_answer,
                start_x=position.x,
                start_y=position.y,
                color=self.text_color,
                font_size=self.text_font_size,
            )
            text_object.x -= text_object.content_width / 2
            self.text_objects.append(text_object)

    def setup(self, trivia_question: TriviaQuestion) -> None:
        self.menu_options_tracker.reset()
        self.answers_options_tracker.reset()
        self.trivia_question = trivia_question
        self.game_core.recreate_scene()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        self._init_text_objects_from_trivia_question(trivia_question)
        self.game_core.scene.add_sprite_list(self.main_sprite_list_name)
        self.game_core.sprites_collection.math_trivia_background.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.math_trivia_background.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_core.sprites_collection.math_trivia_background,
        )

    def _draw_selected_answer_progress_bar(self) -> None:
        active_answer_selection: Optional[
            TimeBasedSelection[int]
        ] = self.answers_options_tracker.current_selection
        if active_answer_selection is None:
            return
        draw_progression_as_rectangle_part(
            rectangle=self.answers_progress_bar_rectangles[
                active_answer_selection.selected_object
            ],
            progression_part=active_answer_selection.proportion_of_completed_time,
            color=self.progress_bar_color,
        )

    def _draw_menu_option_progress_bar(self) -> None:
        active_option_selection: Optional[
            TimeBasedSelection[str]
        ] = self.menu_options_tracker.current_selection
        if active_option_selection is None:
            return
        draw_progression_as_rectangle_part(
            rectangle=self.menu_options_progress_bar_rectgangles[
                active_option_selection.selected_object
            ],
            progression_part=active_option_selection.proportion_of_completed_time,
            color=self.progress_bar_color,
        )

    def on_draw(self):
        self.clear()
        self.game_core.scene.draw()
        self._draw_selected_answer_progress_bar()
        self._draw_menu_option_progress_bar()
        for text in self.text_objects:
            text.draw()
        self.game_core.draw_hands_detections_on_web_camera()

    def on_update(self, delta_time: float):
        self.game_core.update_inner_state(delta_time)
        self.menu_options_tracker.update_inner_state(
            active_gesture_selection=self.game_core.hand_detection_state.active_gesture_time_track
        )
        self.answers_options_tracker.update_inner_state(
            active_gesture_selection=self.game_core.hand_detection_state.active_gesture_time_track
        )
        selected_answer_index: Optional[
            int
        ] = self.answers_options_tracker.active_option
        if selected_answer_index is not None:
            selected_answer_index -= 1
            selected_answer: str = self.trivia_question.all_answers[
                selected_answer_index
            ]
            self.game_manager.end_level(user_answer=selected_answer)

        selected_menu_option: Optional[str] = self.menu_options_tracker.active_option
        if selected_menu_option == "end_game":
            self.game_manager.end_game()


from MVP.math_trivia_game.math_trivia_game_manager import MathTriviaGameManager
