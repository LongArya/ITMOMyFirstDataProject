from __future__ import annotations
import arcade
import random
from MVP.game_core_protocol import GameCoreProtocol
from static_gesture_classification.static_gesture import StaticGesture
from typing import List


MEMORIZATION_GAME_GESTURES_NUM: int = 5


class MemoryGameManager(arcade.View):
    def __init__(self, game_core: GameCoreProtocol, menu_view: MenuManagerView):
        self.game_core = game_core
        self.menu_view = menu_view
        self.memorization_gestures: List[StaticGesture] = random.sample(
            population=list(StaticGesture), k=MEMORIZATION_GAME_GESTURES_NUM
        )

    def setup(self) -> None:
        self.memorization_gestures = random.sample(
            population=list(StaticGesture), k=MEMORIZATION_GAME_GESTURES_NUM
        )

    def start_memorization_stage(self):
        pass


from MVP.menu_manager_view import MenuManagerView
