import arcade
import PIL
from PIL import Image
import cv2
from dataclasses import dataclass
from static_gesture_classification.static_gesture import StaticGesture
from typing import Optional, Dict, List
from enum import Enum, auto
import random

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 650
SCREEN_TITLE = "Rock Paper Scissors Game"


class RPSGameOption(Enum):
    ROCK = auto()
    PAPER = auto()
    SCISSORS = auto()


class EndGameResult(Enum):
    Win = auto()
    Lose = auto()
    Tie = auto()


@dataclass
class GestureSelection:
    """Class for tracking time spent with gesture"""

    gesture: StaticGesture
    accumulated_time_seconds: float
    time_requirement_seconds: float

    def accumulate_time(self, time: float) -> None:
        self.accumulated_time_seconds += time
        self.accumulated_time_seconds = min(
            self.accumulated_time_seconds, self.time_requirement_seconds
        )

    def is_gesture_compatible(self, gesture: StaticGesture) -> bool:
        return self.gesture == gesture

    @property
    def is_active(self) -> bool:
        return self.accumulated_time_seconds == self.time_requirement_seconds

    @property
    def proportion_of_completed_time(self) -> float:
        return self.accumulated_time_seconds / self.time_requirement_seconds


def kb_symbol_to_gesture(symbol_code: Optional[int]) -> StaticGesture:
    if symbol_code is None:
        return StaticGesture.BACKGROUND
    gesture_code = symbol_code - 49
    try:
        return StaticGesture(gesture_code)
    except ValueError:
        return StaticGesture.BACKGROUND


def rock_paper_scissors_rule(
    hero_option: RPSGameOption, enemy_option: RPSGameOption
) -> EndGameResult:
    if hero_option == RPSGameOption.ROCK:
        if enemy_option == RPSGameOption.ROCK:
            return EndGameResult.Tie
        if enemy_option == RPSGameOption.PAPER:
            return EndGameResult.Win
        if enemy_option == RPSGameOption.SCISSORS:
            return EndGameResult.Lose
    if hero_option == RPSGameOption.PAPER:
        if enemy_option == RPSGameOption.ROCK:
            return EndGameResult.Win
        if enemy_option == RPSGameOption.PAPER:
            return EndGameResult.Tie
        if enemy_option == RPSGameOption.SCISSORS:
            return EndGameResult.Lose
    if hero_option == RPSGameOption.SCISSORS:
        if enemy_option == RPSGameOption.ROCK:
            return EndGameResult.Lose
        if enemy_option == RPSGameOption.PAPER:
            return EndGameResult.Win
        if enemy_option == RPSGameOption.SCISSORS:
            return EndGameResult.Tie


class RockPaperScissorsGameManagerView(arcade.View):
    def __init__(self) -> None:
        super().__init__()
        self.gesture_selection_view = RPSGestureSelectionView(game_manager=self)
        self.opponent_turn_view = RPSOpponentWaitingView(
            game_manager=self, waiting_time_requirement_seconds=2
        )
        self.result_screen_view = RPSResultScreen(self)
        self.window.show_view(self.gesture_selection_view)
        self.hero_option: Optional[RPSGameOption] = None

    def switch_to_gesture_selection_view(self) -> None:
        self.window.show_view(self.gesture_selection_view)

    def switch_to_op_turn(self) -> None:
        self.window.show_view(self.opponent_turn_view)

    def switch_to_result_turn(self) -> None:
        enemy_option: RPSGameOption = random.choice(list(RPSGameOption))
        result: EndGameResult = rock_paper_scissors_rule(
            hero_option=self.hero_option, enemy_option=enemy_option
        )
        self.result_screen_view.setup(
            hero_option=self.hero_option, enemy_option=enemy_option, result=result
        )
        self.window.show_view(self.result_screen_view)

    def on_draw(self):
        self.clear()
        arcade.set_background_color(arcade.csscolor.SKY_BLUE)


class RPSGestureSelectionView(arcade.View):
    def __init__(self, game_manager: RockPaperScissorsGameManagerView) -> None:
        super().__init__()
        self._gesture_time_requirement_seconds: float = 2.0
        self.current_selection: GestureSelection = self._init_gesture_selection(
            StaticGesture.BACKGROUND
        )
        self.selected_keyboard_symbol: Optional[int] = None
        self.gesture_rps_option_mapping: Dict[StaticGesture, RPSGameOption] = {
            StaticGesture.TWO: RPSGameOption.SCISSORS,
            StaticGesture.FIVE: RPSGameOption.PAPER,
            StaticGesture.FIST: RPSGameOption.ROCK,
        }
        self.game_manager: RockPaperScissorsGameManagerView = game_manager

    def _init_gesture_selection(self, gesture: StaticGesture):
        return GestureSelection(
            gesture=gesture,
            accumulated_time_seconds=0,
            time_requirement_seconds=self._gesture_time_requirement_seconds,
        )

    def _handle_selected_gesture(self, gesture: StaticGesture, delta_time: float):
        if self.current_selection.is_gesture_compatible(gesture):
            self.current_selection.accumulate_time(delta_time)
        else:
            self.current_selection = self._init_gesture_selection(gesture)

    def _draw_rps_option_selection_as_filled_arc(
        self, gesture_selection: GestureSelection, center_point: arcade.NamedPoint
    ) -> None:
        current_rps_option: Optional[
            RPSGameOption
        ] = self.gesture_rps_option_mapping.get(gesture_selection.gesture, None)
        if current_rps_option is None:
            raise ValueError("")
        gesture_selection_arc_start = 90
        arc_span_degrees = gesture_selection.proportion_of_completed_time * 360
        arcade.draw_text(
            f"{current_rps_option.name}",
            start_x=center_point.x,
            start_y=center_point.y,
            color=arcade.color.GREEN,
            font_size=20,
            width=SCREEN_WIDTH,
            align="center",
        )
        arcade.draw_arc_filled(
            center_x=center_point.x,
            center_y=center_point.y,
            width=50,
            height=50,
            color=(0, 255, 0),
            start_angle=gesture_selection_arc_start,
            end_angle=gesture_selection_arc_start + arc_span_degrees,
        )

    def _draw_allowed_gesture_warning(self) -> None:
        arcade.draw_text(
            f"Show one of TWO, FIVE, FIST",
            start_x=200,
            start_y=200,
            color=arcade.color.GREEN,
            font_size=20,
            width=SCREEN_WIDTH,
            align="center",
        )

    def setup(self) -> None:
        pass

    def on_draw(self):
        self.clear()
        current_rps_option: Optional[
            RPSGameOption
        ] = self.gesture_rps_option_mapping.get(self.current_selection.gesture, None)
        if current_rps_option is None:
            self._draw_allowed_gesture_warning()
        else:
            self._draw_rps_option_selection_as_filled_arc(
                gesture_selection=self.current_selection,
                center_point=arcade.NamedPoint(200, 200),
            )
        return super().on_draw()

    def _read_gesture(self) -> StaticGesture:
        return kb_symbol_to_gesture(self.selected_keyboard_symbol)

    def on_key_press(self, symbol: int, modifiers: int):
        self.selected_keyboard_symbol = symbol
        return super().on_key_press(symbol, modifiers)

    def on_key_release(self, _symbol: int, _modifiers: int):
        self.selected_keyboard_symbol = None
        return super().on_key_release(_symbol, _modifiers)

    def read_gesture(self) -> StaticGesture:
        return self.selected_keyboard_symbol

    def on_update(self, delta_time: float):
        gesture: StaticGesture = self._read_gesture()
        self._handle_selected_gesture(gesture=gesture, delta_time=delta_time)
        if self.current_selection.is_active:
            current_rps_option: Optional[
                RPSGameOption
            ] = self.gesture_rps_option_mapping.get(
                self.current_selection.gesture, None
            )
            if current_rps_option is not None:
                self.game_manager.hero_option = current_rps_option
                self.game_manager.switch_to_op_turn()
        return super().on_update(delta_time)


class RPSOpponentWaitingView(arcade.View):
    def __init__(
        self,
        waiting_time_requirement_seconds: float,
        game_manager: RockPaperScissorsGameManagerView,
    ) -> None:
        super().__init__()
        self.game_manager: RockPaperScissorsGameManagerView = game_manager
        self.waiting_time_requirement_seconds: float = waiting_time_requirement_seconds
        self.waited_time_seconds: float = 0

    def on_update(self, delta_time: float):
        self.waited_time_seconds += delta_time
        if self.waited_time_seconds >= self.waiting_time_requirement_seconds:
            self.game_manager.switch_to_result_turn()
        return super().on_update(delta_time)

    def on_draw(self):
        self.clear()
        arcade.draw_text(
            f"Waiting for opponent",
            start_x=200,
            start_y=200,
            color=arcade.color.GREEN,
            font_size=20,
            width=SCREEN_WIDTH,
            align="center",
        )
        return super().on_draw()


class RPSResultScreen(arcade.View):
    def __init__(self, game_manager: RockPaperScissorsGameManagerView) -> None:
        super().__init__()
        self.game_manager: RockPaperScissorsGameManagerView = game_manager
        self.hero_option: Optional[RPSGameOption] = None
        self.enemy_option: Optional[RPSGameOption] = None
        self.result: Optional[EndGameResult] = None

    def setup(
        self,
        hero_option: RPSGameOption,
        enemy_option: RPSGameOption,
        result: EndGameResult,
    ) -> None:
        self.hero_option = hero_option
        self.enemy_option = enemy_option
        self.result = result

    def draw_text(self, text: str):
        arcade.draw_text(
            text,
            start_x=200,
            start_y=200,
            color=arcade.color.GREEN,
            font_size=20,
            width=SCREEN_WIDTH,
            align="center",
        )

    def on_draw(self):
        self.clear()
        text: str = (
            f"{self.hero_option.name} VS {self.enemy_option.name} : {self.result.name}"
        )
        self.draw_text(text)


def main():
    """Main function"""
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
    rps_game = RockPaperScissorsGameManagerView()
    arcade.run()


if __name__ == "__main__":
    # print(random.choice(list(RPSGameOption)))
    main()
