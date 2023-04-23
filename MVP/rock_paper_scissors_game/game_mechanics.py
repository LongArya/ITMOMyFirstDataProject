from enum import Enum, auto


class RPSGameOption(Enum):
    ROCK = auto()
    PAPER = auto()
    SCISSORS = auto()


class EndGameResult(Enum):
    Win = auto()
    Lose = auto()
    Tie = auto()


def rock_paper_scissors_rule(
    hero_option: RPSGameOption, enemy_option: RPSGameOption
) -> EndGameResult:
    """Determines game's outcome from hero perspective"""
    if hero_option == RPSGameOption.ROCK:
        if enemy_option == RPSGameOption.ROCK:
            return EndGameResult.Tie
        if enemy_option == RPSGameOption.PAPER:
            return EndGameResult.Win
        if enemy_option == RPSGameOption.SCISSORS:
            return EndGameResult.Lose
    elif hero_option == RPSGameOption.PAPER:
        if enemy_option == RPSGameOption.ROCK:
            return EndGameResult.Win
        if enemy_option == RPSGameOption.PAPER:
            return EndGameResult.Tie
        if enemy_option == RPSGameOption.SCISSORS:
            return EndGameResult.Lose
    elif hero_option == RPSGameOption.SCISSORS:
        if enemy_option == RPSGameOption.ROCK:
            return EndGameResult.Lose
        if enemy_option == RPSGameOption.PAPER:
            return EndGameResult.Win
        if enemy_option == RPSGameOption.SCISSORS:
            return EndGameResult.Tie
    else:
        raise ValueError(f"Undefined hero option {hero_option}")
