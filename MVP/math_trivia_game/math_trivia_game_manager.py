from __future__ import annotations
import arcade
from MVP.ui_const import SCREEN_WIDTH, SCREEN_HEIGHT
from arcade.application import Window
from MVP.game_core_protocol import GameCoreProtocol
from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.gesture_detection import GestureDetection
from MVP.math_trivia_game.trivia_question import TriviaQuestion
from static_gesture_classification.static_gesture import StaticGesture
from typing import Optional, List, Dict
import random


TRIVIA_ANSWERS_COUNT: int = 4


def generate_random_trivia_question() -> TriviaQuestion:
    a = random.randrange(1000)
    b = random.randrange(1000)
    correct_answer_index: int = random.randrange(TRIVIA_ANSWERS_COUNT)
    correct_answer = a + b
    wrong_answers: List[int] = list(
        range(correct_answer - 150, correct_answer + 150, 1)
    )
    wrong_answers = list(filter(lambda number: number > 0, wrong_answers))
    all_answers: List[int] = random.sample(wrong_answers, TRIVIA_ANSWERS_COUNT)
    all_answers[correct_answer_index] = correct_answer
    return TriviaQuestion(
        question_text=f"{a} + {b} = ?",
        all_answers=[f"{a}" for a in all_answers],
        correct_answer=f"{correct_answer}",
    )


class MathTriviaGameManager(arcade.View):
    def __init__(self, game_core: GameCoreProtocol, menu_view: MenuManagerView):
        super().__init__()
        self.game_core = game_core
        self.menu_view = menu_view
        self.completed_levels_num: int
        self.correct_answers_num: int
        self.current_question: TriviaQuestion
        self.math_question_view = MathQuestionView(
            game_core=game_core, game_manager=self
        )
        self.results_view = MathTriviaResultsView(
            game_core=self.game_core, game_manager=self
        )

    def setup(self) -> None:
        self.completed_levels_num = 0
        self.correct_answers_num = 0
        self.current_question = generate_random_trivia_question()
        self.math_question_view.setup(self.current_question)
        self.window.show_view(self.math_question_view)

    def end_level(self, user_answer: str) -> None:
        # register result
        self.completed_levels_num += 1
        if self.current_question.is_answer_correct(user_answer):
            self.correct_answers_num += 1
        self.current_question = generate_random_trivia_question()
        self.math_question_view.setup(self.current_question)
        self.window.show_view(self.math_question_view)

    def end_game(self) -> None:
        end_game_result_msg: str = (
            f"{self.correct_answers_num} / {self.completed_levels_num}"
        )
        self.results_view.setup(result_text=end_game_result_msg)
        self.window.show_view(self.results_view)

    def return_to_menu(self) -> None:
        self.menu_view.setup()
        self.window.show_view(self.menu_view)

    def replay(self) -> None:
        self.setup()


from MVP.math_trivia_game.math_question_view import MathQuestionView
from MVP.math_trivia_game.math_trivia_result_view import MathTriviaResultsView
from MVP.menu_manager_view import MenuManagerView
