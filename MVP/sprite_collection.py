import os
from static_gesture_classification.static_gesture import StaticGesture
from MVP.rock_paper_scissors_game.game_mechanics import RPSGameOption, EndGameResult
from MVP.data_structures.game_kind import GameKind
from MVP.memory_game.memory_game_result_kind import MemoryGameResultKind
import arcade
from typing import Dict


class SpriteCollection:
    def __init__(self, sprites_root: str):
        self.sprites_root = sprites_root
        self.active_gestures_sprites: Dict[
            StaticGesture, arcade.Sprite
        ] = self.load_getures_sprites(state="active")
        self.not_active_gestures_sprites: Dict[
            StaticGesture, arcade.Sprite
        ] = self.load_getures_sprites(state="active")
        self.web_camera_preview = arcade.Sprite(
            os.path.join(self.sprites_root, "web_camera_preview.png")
        )
        self.RPS_geture_selection_background = arcade.Sprite(
            os.path.join(self.sprites_root, "RPS", "RPSGestureSelectionBackground.png")
        )
        self.RPS_results_backgrond = arcade.Sprite(
            os.path.join(self.sprites_root, "RPS", "RPSResultsBackground.png")
        )
        self.RPS_selected_options: Dict[
            RPSGameOption, arcade.Sprite
        ] = self.load_RPS_options(state="selected")
        self.RPS_not_selected_options: Dict[
            RPSGameOption, arcade.Sprite
        ] = self.load_RPS_options(state="not_selected")
        self.RPS_enemy_turn: Dict[RPSGameOption, arcade.Sprite] = self.load_RPS_options(
            state="enemy"
        )
        self.RPS_outcomes: Dict[
            EndGameResult, arcade.Sprite
        ] = self.load_RPS_end_game_results()
        self.selected_menu_cards: Dict[
            GameKind, arcade.Sprite
        ] = self.load_menu_game_cards(state="selected")
        self.not_selected_menu_cards: Dict[
            GameKind, arcade.Sprite
        ] = self.load_menu_game_cards(state="not_selected")
        self.menu_cards_placeholders = self.load_menu_game_cards(state="placeholder")
        self.menu_background_sprite: arcade.Sprite = arcade.Sprite(
            os.path.join(self.sprites_root, "Menu", "MenuBackground.png")
        )
        self.menu_cards_with_rules = self.load_menu_game_cards(state="rules")
        self.closed_menu_cursor: arcade.Sprite = arcade.Sprite(
            os.path.join(self.sprites_root, "Menu", "closed_cursor.png")
        )
        self.open_menu_cursor: arcade.Sprite = arcade.Sprite(
            os.path.join(self.sprites_root, "Menu", "open_cursor.png")
        )
        # MEMORY GAME
        self.memory_game_preparation_bg: arcade.Sprite = arcade.Sprite(
            os.path.join(self.sprites_root, "MemoryGame", "MemoryGamePreparation.png")
        )
        self.memory_game_memorization_bg: arcade.Sprite = arcade.Sprite(
            os.path.join(
                self.sprites_root, "MemoryGame", "MemoryGameMemorizationStage.png"
            )
        )
        self.memory_game_results_bg: arcade.Sprite = arcade.Sprite(
            os.path.join(self.sprites_root, "MemoryGame", "MemoryGameResults.png")
        )
        self.memory_game_gesture_demo_bg: arcade.Sprite = arcade.Sprite(
            os.path.join(
                self.sprites_root, "MemoryGame", "MemoryGameGesturesDemonstration.png"
            )
        )
        self.memory_game_reults: Dict[MemoryGameResultKind, arcade.Sprite] = {
            kind: arcade.Sprite(
                filename=os.path.join(
                    self.sprites_root, "MemoryGame", "results", f"{kind.name}.png"
                )
            )
            for kind in MemoryGameResultKind
        }
        self.numbers_cards: Dict[int, arcade.Sprite] = {
            number: arcade.Sprite(
                os.path.join(
                    self.sprites_root, "MemoryGame", "numbers", f"{number}.png"
                )
            )
            for number in range(6)
        }
        self.not_active_gestures_cards_paths: Dict[
            StaticGesture, str
        ] = self.get_gestures_cards_paths(state="not_active")
        self.active_gestures_cards_paths: Dict[
            StaticGesture, str
        ] = self.get_gestures_cards_paths(state="active")

    def get_gestures_cards_paths(self, state: str) -> Dict[StaticGesture, str]:
        gestures_cards_root = os.path.join(self.sprites_root, "MemoryGame", state)
        gestures_cards_sprites: Dict[StaticGesture, str] = {}
        for static_gesture in StaticGesture:
            gestures_cards_sprites[static_gesture] = os.path.join(
                gestures_cards_root, f"{static_gesture.name}.png"
            )
        return gestures_cards_sprites

    def load_getures_sprites(self, state: str) -> Dict[StaticGesture, arcade.Sprite]:
        gestures_root = os.path.join(self.sprites_root, "GesturesDetections", state)
        gestures_sprites: Dict[StaticGesture, arcade.Sprite] = {}
        for gesture in StaticGesture:
            gestures_sprites[gesture] = arcade.Sprite(
                filename=os.path.join(gestures_root, f"{gesture.name}.png")
            )
        return gestures_sprites

    def load_RPS_options(self, state: str) -> Dict[RPSGameOption, arcade.Sprite]:
        rps_options_root = os.path.join(
            self.sprites_root, "RPS", "RPSGameOption", state
        )
        rps_options_sprites: Dict[RPSGameOption, arcade.Sprite] = {}
        for rps_option in RPSGameOption:
            rps_options_sprites[rps_option] = arcade.Sprite(
                os.path.join(rps_options_root, f"{rps_option.name}.png")
            )
        return rps_options_sprites

    def load_RPS_end_game_results(self) -> Dict[EndGameResult, arcade.Sprite]:
        rps_results_root = os.path.join(self.sprites_root, "RPS", "Outcomes")
        rps_results_sprites: Dict[EndGameResult, arcade.Sprite] = {}
        for endgame_result in EndGameResult:
            rps_results_sprites[endgame_result] = arcade.Sprite(
                os.path.join(rps_results_root, f"{endgame_result.name}.png")
            )
        return rps_results_sprites

    def load_menu_game_cards(self, state) -> Dict[GameKind, arcade.Sprite]:
        menu_options_root = os.path.join(self.sprites_root, "Menu", "game_cards", state)
        menu_cards_sprites: Dict[GameKind, arcade.Sprite] = {}
        for game_kind in GameKind:
            menu_cards_sprites[game_kind] = arcade.Sprite(
                os.path.join(menu_options_root, f"{game_kind.name}.png")
            )
        return menu_cards_sprites
