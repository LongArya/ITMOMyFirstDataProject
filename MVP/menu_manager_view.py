import arcade
from MVP.ui_const import SCREEN_HEIGHT, SCREEN_WIDTH
from MVP.game_core_protocol import GameCoreProtocol
from typing import Dict, Optional
from MVP.data_structures.game_kind import GameKind
from MVP.geometry_utils import (
    project_point_to_rectangle,
    get_box_center,
    change_point_origin,
    from_opencv_coordinate_system_to_arcade,
)
from MVP.data_structures.track import Track
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.gesture_detection import GestureDetection
from static_gesture_classification.static_gesture import StaticGesture


class MenuManagerView(arcade.View):
    def __init__(self, game_core: GameCoreProtocol) -> None:
        super().__init__()
        self.game_core = game_core
        self.games: Dict[GameKind, arcade.View] = {
            GameKind.ROCK_PAPER_SCISSORS: RockPaperScissorsGameManager(
                game_core=self.game_core, menu_view=self
            ),
            GameKind.MEMORY_GAME: MemoryGameManager(
                game_core=self.game_core, menu_view=self
            ),
            GameKind.MATH_TRIVIA_GAME: MathTriviaGameManager(
                game_core=self.game_core, menu_view=self
            ),
        }
        self.cards_sprite_list: str = "cards_sprite_list"
        self.cursor_sprite_list: str = "cursor_sprite_list"
        self.selected_game: Optional[GameKind] = None
        self.is_cursor_closed_on_current_frame: bool = False
        self.is_cursor_closed_on_prev_frame: bool = False
        self.cursor_default_position_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=SCREEN_WIDTH / 2, y=SCREEN_HEIGHT / 2
        )
        self.cursor_center_current_position_arcade: arcade.NamedPoint = (
            self.cursor_default_position_arcade
        )
        self.selection_area_origin_opencv = arcade.NamedPoint(0, 0)
        self.selection_area_dimensions: arcade.NamedPoint = arcade.NamedPoint(
            x=840, y=SCREEN_HEIGHT
        )
        self.games_cards_positions_arcade: Dict[GameKind, arcade.NamedPoint] = {
            GameKind.ROCK_PAPER_SCISSORS: arcade.NamedPoint(x=164, y=522),
            GameKind.MEMORY_GAME: arcade.NamedPoint(x=420, y=522),
            GameKind.MATH_TRIVIA_GAME: arcade.NamedPoint(x=676, y=522),
        }
        self.rule_card_position: arcade.NamedPoint = arcade.NamedPoint(x=1077, y=521.5)

    def start_game(self, game_kind: GameKind) -> None:
        self.games[game_kind].setup()

    def setup(self) -> None:
        self.game_core.recreate_scene()
        self.game_core.hand_detection_state.nullify_gesture_selection()
        self.game_core.scene.add_sprite_list(self.cards_sprite_list)
        self.game_core.scene.add_sprite_list(self.cursor_sprite_list)
        # add background
        self.game_core.sprites_collection.menu_background_sprite.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_core.sprites_collection.menu_background_sprite.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_core.scene.add_sprite(
            self.cards_sprite_list,
            self.game_core.sprites_collection.menu_background_sprite,
        )
        # add blank game cards and not active game cards
        for game_kind in GameKind:
            card_center_position = self.games_cards_positions_arcade[game_kind]
            self.game_core.sprites_collection.menu_cards_placeholders[
                game_kind
            ].center_x = card_center_position.x
            self.game_core.sprites_collection.menu_cards_placeholders[
                game_kind
            ].center_y = card_center_position.y
            self.game_core.sprites_collection.not_selected_menu_cards[
                game_kind
            ].center_x = card_center_position.x
            self.game_core.sprites_collection.not_selected_menu_cards[
                game_kind
            ].center_y = card_center_position.y
            self.game_core.sprites_collection.selected_menu_cards[
                game_kind
            ].center_x = card_center_position.x
            self.game_core.sprites_collection.selected_menu_cards[
                game_kind
            ].center_y = card_center_position.y
            self.game_core.scene.add_sprite(
                self.cards_sprite_list,
                self.game_core.sprites_collection.menu_cards_placeholders[game_kind],
            )
            self.game_core.scene.add_sprite(
                self.cards_sprite_list,
                self.game_core.sprites_collection.not_selected_menu_cards[game_kind],
            )
        # set rules cards position
        for game_kind in GameKind:
            self.game_core.sprites_collection.menu_cards_with_rules[
                game_kind
            ].center_x = self.rule_card_position.x
            self.game_core.sprites_collection.menu_cards_with_rules[
                game_kind
            ].center_y = self.rule_card_position.y

    def on_draw(self) -> None:
        self.clear()
        self.game_core.scene.draw()
        if self.game_core.hand_detection_state.active_track is not None:
            gesture_detection: GestureDetection = (
                self.game_core.hand_detection_state.active_track.last.object
            )
            self.game_core.draw_gesture_detection_in_web_camera(
                gesture_detection=gesture_detection, active=True
            )

    def update_game_cards_on_scene(self) -> None:
        # remove all active and inactive game cards
        for game_kind in GameKind:
            self.game_core.sprites_collection.not_selected_menu_cards[
                game_kind
            ].remove_from_sprite_lists()
            self.game_core.sprites_collection.selected_menu_cards[
                game_kind
            ].remove_from_sprite_lists()
            self.game_core.sprites_collection.menu_cards_with_rules[
                game_kind
            ].remove_from_sprite_lists()

        # add wrt current selected game
        for game_kind in GameKind:
            target_sprite = (
                self.game_core.sprites_collection.selected_menu_cards[game_kind]
                if game_kind == self.selected_game
                else self.game_core.sprites_collection.not_selected_menu_cards[
                    game_kind
                ]
            )
            self.game_core.scene.add_sprite(self.cards_sprite_list, target_sprite)
        if self.selected_game is not None:
            self.game_core.scene.add_sprite(
                self.cards_sprite_list,
                self.game_core.sprites_collection.menu_cards_with_rules[
                    self.selected_game
                ],
            )

    def update_cursor_position(self) -> None:
        active_track: Optional[
            Track[GestureDetection]
        ] = self.game_core.hand_detection_state.active_track
        if active_track is None:
            return
        if self.is_cursor_closed_on_current_frame:
            gesture_detection: GestureDetection = active_track.last.object
            box_center: arcade.NamedPoint = get_box_center(
                xyxy_box=gesture_detection.xyxy_box
            )
            box_center = project_point_to_rectangle(
                point=box_center,
                projection_dimensions=self.selection_area_dimensions,
                original_space_dimensions=arcade.NamedPoint(
                    x=self.game_core.camera_frame_width,
                    y=self.game_core.camera_frame_height,
                ),
            )
            box_center = change_point_origin(
                opencv_like_point=box_center,
                current_origin=self.selection_area_origin_opencv,
                new_origin=arcade.NamedPoint(0, 0),
            )
            box_center = from_opencv_coordinate_system_to_arcade(
                point_in_opencv_system=box_center,
                width=SCREEN_WIDTH,
                height=SCREEN_HEIGHT,
            )
            self.cursor_center_current_position_arcade = box_center
            # FIXME add clips to prevent out of boundaries

    def place_current_cursor_on_scene(self) -> None:
        # remove present cursors
        self.game_core.sprites_collection.open_menu_cursor.remove_from_sprite_lists()
        self.game_core.sprites_collection.closed_menu_cursor.remove_from_sprite_lists()

        target_sprite: arcade.Sprite = (
            self.game_core.sprites_collection.closed_menu_cursor
            if self.is_cursor_closed_on_current_frame
            else self.game_core.sprites_collection.open_menu_cursor
        )
        target_sprite.center_x = self.cursor_center_current_position_arcade.x
        target_sprite.center_y = self.cursor_center_current_position_arcade.y
        self.game_core.scene.add_sprite(self.cursor_sprite_list, target_sprite)

    def update_current_game_selection(self) -> None:
        if not self.is_cursor_closed_on_current_frame:
            return
        any_cards_touched: bool = False
        for game_kind in GameKind:
            cursor_touches_game: bool = arcade.check_for_collision(
                self.game_core.sprites_collection.closed_menu_cursor,
                self.game_core.sprites_collection.menu_cards_placeholders[game_kind],
            )
            if cursor_touches_game:
                self.selected_game = game_kind
                any_cards_touched = True
                break
        if not any_cards_touched:
            self.selected_game = None

    def on_update(self, delta_time: float) -> None:
        self.game_core.update_inner_state(delta_time)
        # update cursor states
        self.is_cursor_closed_on_prev_frame = self.is_cursor_closed_on_current_frame
        active_geture_selection: Optional[
            TimeTrackedEntity[StaticGesture]
        ] = self.game_core.hand_detection_state.active_gesture_time_track
        self.is_cursor_closed_on_current_frame = (
            active_geture_selection is not None
            and active_geture_selection.entity == StaticGesture.FIST
        )

        self.update_cursor_position()
        self.place_current_cursor_on_scene()
        self.update_current_game_selection()
        self.update_game_cards_on_scene()
        # switch to game
        if (
            not self.is_cursor_closed_on_current_frame
            and self.is_cursor_closed_on_prev_frame
            and self.selected_game is not None
        ):
            self.start_game(self.selected_game)


from MVP.rock_paper_scissors_game.game_manager import RockPaperScissorsGameManager
from MVP.memory_game.memory_game_manager import MemoryGameManager
from MVP.math_trivia_game.math_trivia_game_manager import MathTriviaGameManager
