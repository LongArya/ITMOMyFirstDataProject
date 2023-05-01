from __future__ import annotations
import arcade
import random
from MVP.game_core_protocol import GameCoreProtocol
from MVP.memory_game.memory_game_result_kind import MemoryGameResultKind
from static_gesture_classification.static_gesture import StaticGesture
from typing import List, Dict


MEMORIZATION_GAME_GESTURES_NUM: int = 5
RESULT_TO_SCORE_MAPPING: Dict[int, MemoryGameResultKind] = {
    0: MemoryGameResultKind.BAD,
    1: MemoryGameResultKind.BAD,
    2: MemoryGameResultKind.BAD,
    3: MemoryGameResultKind.OKAY,
    4: MemoryGameResultKind.GOOD,
    5: MemoryGameResultKind.GOOD,
}


class MemoryGameManager(arcade.View):
    def __init__(self, game_core: GameCoreProtocol, menu_view: MenuManagerView):
        super().__init__()
        self.memorization_gestures: List[StaticGesture]
        self.game_core = game_core
        self.menu_view = menu_view
        self.preparation_stage_view = MemoryGamePreparationView(
            game_core=self.game_core, game_manager=self
        )
        self.memorization_stage_view = MemoryGameMemorizationStage(
            game_core=self.game_core, game_manager=self
        )
        self.gestures_demonstration_stage_view = MemoryGameGesturesDemonstrationView(
            game_core=self.game_core, game_manager=self
        )
        self.results_view = MemoryGameResultsView(
            game_core=self.game_core, game_manager=self
        )

    def _generate_gestures_for_memorization(self) -> List[StaticGesture]:
        memorization_candidates = [
            g for g in StaticGesture if g != StaticGesture.BACKGROUND
        ]
        memorization_gestures = random.sample(
            population=memorization_candidates, k=MEMORIZATION_GAME_GESTURES_NUM
        )
        return memorization_gestures

    def setup(self) -> None:
        self.memorization_gestures = self._generate_gestures_for_memorization()
        self.preparation_stage_view.setup()
        self.window.show_view(self.preparation_stage_view)

    def start_memorization_stage(self) -> None:
        self.memorization_stage_view.setup(
            memorization_gestures=self.memorization_gestures
        )
        self.window.show_view(self.memorization_stage_view)

    def start_gestures_demonstration_stage(self) -> None:
        self.gestures_demonstration_stage_view.setup()
        self.window.show_view(self.gestures_demonstration_stage_view)

    def go_to_results(self, user_gestures: List[StaticGesture]) -> None:
        user_score: int = 0
        for user_gesture, gt_gesture in zip(user_gestures, self.memorization_gestures):
            if user_gesture == gt_gesture:
                user_score += 1
        result: MemoryGameResultKind = RESULT_TO_SCORE_MAPPING[user_score]
        self.results_view.setup(
            user_gestures=user_gestures,
            gt_gestures=self.memorization_gestures,
            correct_answers_number=user_score,
            result_kind=result,
        )
        self.window.show_view(self.results_view)

    def replay(self) -> None:
        self.setup()

    def return_to_menu(self) -> None:
        self.menu_view.setup()
        self.window.show_view(self.menu_view)


from MVP.menu_manager_view import MenuManagerView
from MVP.memory_game.gestures_demonstration_view import (
    MemoryGameGesturesDemonstrationView,
)
from MVP.memory_game.gestures_memorization_view import MemoryGameMemorizationStage
from MVP.memory_game.game_preparation_view import MemoryGamePreparationView
from MVP.memory_game.results_view import MemoryGameResultsView
