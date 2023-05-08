from __future__ import annotations
import arcade
from MVP.rock_paper_scissors_game.game_mechanics import (
    RPSGameOption,
    EndGameResult,
    rock_paper_scissors_rule,
)
from MVP.game_core import GameCore
import random


class RockPaperScissorsGameManager(arcade.View):
    def __init__(
        self,
        game_core: GameCore,
        menu_view: MenuManagerView,
    ) -> None:
        super().__init__()
        self.menu_manager = menu_view
        self.game_core = game_core
        self.gesture_selection = RPSGestureSelectionView(
            game_core=self.game_core, game_manager=self
        )
        self.results_screen = RPSResultsView(
            game_core=self.game_core, game_manager=self
        )

    def setup(self) -> None:
        self.gesture_selection.setup()
        self.window.show_view(self.gesture_selection)

    def switch_to_results(self, hero_option: RPSGameOption) -> None:
        enemy_option: RPSGameOption = random.choice(list(RPSGameOption))
        result: EndGameResult = rock_paper_scissors_rule(
            hero_option=hero_option, enemy_option=enemy_option
        )
        self.results_screen.setup(
            hero_option=hero_option, enemy_option=enemy_option, outcome=result
        )
        self.window.show_view(self.results_screen)

    def replay(self) -> None:
        self.setup()

    def return_to_menu(self) -> None:
        self.menu_manager.setup()
        self.window.show_view(self.menu_manager)


from MVP.menu_manager_view import MenuManagerView
from MVP.rock_paper_scissors_game.rps_result_view import (
    RPSResultsView,
)
from MVP.rock_paper_scissors_game.rps_gesture_selection_view import (
    RPSGestureSelectionView,
)
