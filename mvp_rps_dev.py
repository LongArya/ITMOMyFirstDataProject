import cv2
import logging
import random
from general.data_structures.data_split import DataSplit
import hydra
import arcade
import os
import numpy as np
from const import (
    STATIC_GESTURE_CFG_NAME,
    STATIC_GESTURE_CFG_ROOT,
    YOLO_V7_HAND_DETECTION,
    YOLO_V7_INPUT_RESOLUTION,
)
from yolo_detection import YoloInferece
from hydra.core.config_store import ConfigStore
from static_gesture_classification.config import StaticGestureConfig
from MVP.sprite_collection import SpriteCollection
from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.ui_const import GAME_RESOUSES_DIR, SCREEN_HEIGHT, SCREEN_WIDTH
from MVP.geometry_utils import (
    change_point_origin,
    from_opencv_coordinate_system_to_arcade,
    project_point_to_rectangle,
    get_box_center,
)
from typing import Optional, Protocol, Dict
from MVP.hands_detection_state import HandDetectionState
from MVP.data_structures.gesture_detection import GestureDetection
from MVP.rock_paper_scissors_game.game_mechanics import (
    RPSGameOption,
    EndGameResult,
    rock_paper_scissors_rule,
)
from static_gesture_classification.static_gesture import StaticGesture
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
    init_augmentations_from_config,
)

os.environ["HYDRA_FULL_ERROR"] = "1"
cs = ConfigStore.instance()
cs.store(name=STATIC_GESTURE_CFG_NAME, node=StaticGestureConfig)

arcade.configure_logging(level=logging.ERROR)


class GameManagerProtocol(Protocol):
    hand_detection_state: HandDetectionState
    sprites_collection: SpriteCollection
    scene: arcade.Scene

    def recreate_scene(self):
        pass

    def update_inner_state(self, time_delta: float) -> None:
        pass

    def setup_web_camera_preview_in_scene(self) -> None:
        pass

    def draw_gesture_detection_in_web_camera(
        self, gesture_detection: GestureDetection, active: bool
    ) -> None:
        pass


class GameManagerPlaceHolder:
    def __init__(
        self, hand_detection: HandDetectionState, sprites_collection: SpriteCollection
    ):
        self.cap = cv2.VideoCapture(0)
        self.hand_detection_state = hand_detection
        self.sprites_collection = sprites_collection
        self.frame_number = 0
        self.scene = arcade.Scene()
        self.web_camera_sprite_list_name = "web_camera_preview"

    def recreate_scene(self):
        self.scene = arcade.Scene()

    def update_inner_state(self, time_delta: float) -> None:
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.fliplr(frame)
        self.hand_detection_state.update_inner_state(
            image=frame, time_delta=time_delta, frame_number=self.frame_number
        )
        self.frame_number += 1

    def setup_web_camera_preview_in_scene(self) -> None:
        self.scene.add_sprite_list(self.web_camera_sprite_list_name)
        self.sprites_collection.web_camera_preview.center_x = (
            SCREEN_WIDTH - self.sprites_collection.web_camera_preview.width // 2
        )
        self.sprites_collection.web_camera_preview.center_y = (
            self.sprites_collection.web_camera_preview.height // 2
        )
        self.scene.add_sprite(
            name=self.web_camera_sprite_list_name,
            sprite=self.sprites_collection.web_camera_preview,
        )

    def _remove_gestures_sprites_from_web_camera_preview(self) -> None:
        gesture_sprite: arcade.Sprite
        for gesture_sprite in self.sprites_collection.active_gestures_sprites.values():
            gesture_sprite.remove_from_sprite_lists()
        for (
            gesture_sprite
        ) in self.sprites_collection.not_active_gestures_sprites.values():
            gesture_sprite.remove_from_sprite_lists()

    def draw_gesture_detection_in_web_camera(
        self, gesture_detection: GestureDetection, active: bool
    ) -> None:
        self._remove_gestures_sprites_from_web_camera_preview()
        # select_target_sprite
        target_sprite: arcade.Sprite
        if active:
            target_sprite = self.sprites_collection.active_gestures_sprites[
                gesture_detection.gesture
            ]
        else:
            target_sprite = self.sprites_collection.not_active_gestures_sprites[
                gesture_detection.gesture
            ]
        # count gesture position in web camera preview
        box_center: arcade.NamedPoint = get_box_center(
            xyxy_box=gesture_detection.xyxy_box
        )
        box_center = project_point_to_rectangle(
            point=box_center,
            projection_dimensions=arcade.NamedPoint(
                x=self.sprites_collection.web_camera_preview.width,
                y=self.sprites_collection.web_camera_preview.height,
            ),
            original_space_dimensions=arcade.NamedPoint(
                x=640, y=480
            ),  # FIXME read actual web camera dimension values,
        )
        box_center = change_point_origin(
            opencv_like_point=box_center,
            current_origin=arcade.NamedPoint(
                x=SCREEN_WIDTH - self.sprites_collection.web_camera_preview.width,
                y=SCREEN_HEIGHT - self.sprites_collection.web_camera_preview.height,
            ),  # FIXME temporary
            new_origin=arcade.NamedPoint(0, 0),
        )
        box_center = from_opencv_coordinate_system_to_arcade(
            point_in_opencv_system=box_center, width=SCREEN_WIDTH, height=SCREEN_HEIGHT
        )
        target_sprite.center_x = box_center.x
        target_sprite.center_y = box_center.y
        self.scene.add_sprite(self.web_camera_sprite_list_name, target_sprite)


class RPSGestureSelectionView(arcade.View):
    def __init__(self, game_manager: GameManagerProtocol) -> None:
        super().__init__()
        self.game_manager = game_manager
        self.main_sprite_list_name: str = "RPSGestureSelection"
        self.rps_option_selection_time_requirement_seconds: float = 5
        self.gesture_rps_option_mapping: Dict[StaticGesture, RPSGameOption] = {
            StaticGesture.TWO: RPSGameOption.SCISSORS,
            StaticGesture.FIVE: RPSGameOption.PAPER,
            StaticGesture.FIST: RPSGameOption.ROCK,
        }
        self.rps_option_selection: Optional[TimeBasedSelection[RPSGameOption]] = None
        self.rps_draw_positions_arcade: Dict[RPSGameOption, arcade.NamedPoint] = {
            RPSGameOption.PAPER: arcade.NamedPoint(x=163, y=340),
            RPSGameOption.SCISSORS: arcade.NamedPoint(x=420, y=340),
            RPSGameOption.ROCK: arcade.NamedPoint(x=678, y=340),
        }
        self.progress_bar_left_rop_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=72, y=631
        )
        self.progress_bar_height: int = 37
        self.progress_bar_max_width = 699
        self.progress_bar_color: arcade.Color = [255, 145, 103]

    # FIXME move scene setup to separate method

    def _add_not_selected_rps_options_to_scene(self) -> None:
        for rps_option in RPSGameOption:
            target_sprite: arcade.Sprite = (
                self.game_manager.sprites_collection.RPS_not_selected_options[
                    rps_option
                ]
            )
            target_sprite.center_x = self.rps_draw_positions_arcade[rps_option].x
            target_sprite.center_y = self.rps_draw_positions_arcade[rps_option].y
            self.game_manager.scene.add_sprite(
                self.main_sprite_list_name, target_sprite
            )

    def setup(self) -> None:
        self.game_manager.recreate_scene()  # recreate scene
        self.game_manager.hand_detection_state.nullify_gesture_selection()
        self.game_manager.sprites_collection.RPS_geture_selection_background.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_manager.sprites_collection.RPS_geture_selection_background.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_manager.scene.add_sprite_list(self.main_sprite_list_name)
        self.game_manager.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_manager.sprites_collection.RPS_geture_selection_background,
        )
        self._add_not_selected_rps_options_to_scene()
        # web camera
        self.game_manager.setup_web_camera_preview_in_scene()

    def _update_selected_rps_option(self) -> None:
        active_geture_selection: Optional[
            TimeTrackedEntity[StaticGesture]
        ] = self.game_manager.hand_detection_state.active_gesture_time_track
        if active_geture_selection is None:
            self.rps_option_selection = None
            return
        rps_option = self.gesture_rps_option_mapping.get(
            active_geture_selection.entity, None
        )
        if rps_option is None:
            self.rps_option_selection = None
            return
        self.rps_option_selection = TimeBasedSelection(
            selected_object=rps_option,
            accumulated_time_seconds=active_geture_selection.tracked_time_seconds,
            time_requirement_seconds=self.rps_option_selection_time_requirement_seconds,
        )

    def _update_scene_based_on_selected_rps_option(self) -> None:
        # remove all rps options sprites
        sprite: arcade.Sprite
        for (
            sprite
        ) in self.game_manager.sprites_collection.RPS_selected_options.values():
            sprite.remove_from_sprite_lists()
        for (
            sprite
        ) in self.game_manager.sprites_collection.RPS_not_selected_options.values():
            sprite.remove_from_sprite_lists()

        if self.rps_option_selection is None:
            self._add_not_selected_rps_options_to_scene()
            return

        # add sprites
        rps_option: RPSGameOption
        for rps_option in RPSGameOption:
            target_sprite: arcade.Sprite = (
                (self.game_manager.sprites_collection.RPS_selected_options[rps_option])
                if rps_option == self.rps_option_selection.selected_object
                else self.game_manager.sprites_collection.RPS_not_selected_options[
                    rps_option
                ]
            )
            target_sprite.center_x = self.rps_draw_positions_arcade[rps_option].x
            target_sprite.center_y = self.rps_draw_positions_arcade[rps_option].y
            self.game_manager.scene.add_sprite(
                self.main_sprite_list_name, target_sprite
            )

    def _draw_current_rps_selection_progress(self) -> None:
        if self.rps_option_selection is None:
            return
        arcade.draw_lrtb_rectangle_filled(
            left=self.progress_bar_left_rop_arcade.x,
            top=self.progress_bar_left_rop_arcade.y,
            bottom=self.progress_bar_left_rop_arcade.y - self.progress_bar_height,
            right=self.progress_bar_left_rop_arcade.x
            + self.progress_bar_max_width
            * self.rps_option_selection.proportion_of_completed_time,
            color=self.progress_bar_color,
        )

    def on_draw(self) -> None:
        self.clear()
        self.game_manager.scene.draw()
        self._draw_current_rps_selection_progress()
        if self.game_manager.hand_detection_state.active_track is not None:
            gesture_detection: GestureDetection = (
                self.game_manager.hand_detection_state.active_track.last.object
            )
            self.game_manager.draw_gesture_detection_in_web_camera(
                gesture_detection=gesture_detection, active=True
            )

    def on_update(self, delta_time: float):
        self.game_manager.update_inner_state(delta_time)
        self._update_selected_rps_option()
        self._update_scene_based_on_selected_rps_option()
        if self.rps_option_selection is None:
            return
        if self.rps_option_selection.is_active:
            self.game_manager.switch_to_results(
                hero_option=self.rps_option_selection.selected_object
            )


class RPSResultsView(arcade.View):
    def __init__(self, game_manager: GameManagerProtocol) -> None:
        super().__init__()
        self.game_manager = game_manager
        self.hero_option: Optional[RPSGameOption] = None
        self.enemy_option: Optional[RPSGameOption] = None
        self.outcome: Optional[EndGameResult] = None
        self.main_sprite_list_name = "RPSResults"
        self.option_selection_time_requirement_seconds: float = 2
        self.hero_option_draw_position_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=163, y=341
        )
        self.enemy_option_draw_position_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=678, y=341
        )
        self.outcome_draw_position_arcade: arcade.NamedPoint = arcade.NamedPoint(
            x=421, y=583.5
        )
        self.option_progress_bar_draw_positions: Dict[str, arcade.NamedPoint] = {
            "menu": arcade.NamedPoint(x=1048, y=624),
            "replay": arcade.NamedPoint(x=1048, y=520),
        }
        self.gesture_option_mapping: Dict[StaticGesture, str] = {
            StaticGesture.OKEY: "replay",
            StaticGesture.FIST: "menu",
        }
        self.option_progress_bar_max_width = 130
        self.option_progress_bar_height = 12
        self.option_progress_bar_color: arcade.Color = [255, 145, 103]
        self.current_option_selection: Optional[TimeBasedSelection[str]] = None

    def setup(
        self,
        hero_option: RPSGameOption,
        enemy_option: RPSGameOption,
        outcome: EndGameResult,
    ):
        self.current_option_selection = None
        self.hero_option = hero_option
        self.enemy_option = enemy_option
        self.outcome = outcome
        # scene
        self.game_manager.recreate_scene()
        self.game_manager.hand_detection_state.nullify_gesture_selection()
        self.game_manager.scene.add_sprite_list(self.main_sprite_list_name)
        # background
        self.game_manager.sprites_collection.RPS_results_backgrond.center_x = (
            SCREEN_WIDTH / 2
        )
        self.game_manager.sprites_collection.RPS_results_backgrond.center_y = (
            SCREEN_HEIGHT / 2
        )
        self.game_manager.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_manager.sprites_collection.RPS_results_backgrond,
        )
        # hero option
        self.game_manager.sprites_collection.RPS_not_selected_options[
            hero_option
        ].center_x = self.hero_option_draw_position_arcade.x
        self.game_manager.sprites_collection.RPS_not_selected_options[
            hero_option
        ].center_y = self.hero_option_draw_position_arcade.y
        self.game_manager.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_manager.sprites_collection.RPS_not_selected_options[hero_option],
        )
        # enemy option
        self.game_manager.sprites_collection.RPS_enemy_turn[
            enemy_option
        ].center_x = self.enemy_option_draw_position_arcade.x
        self.game_manager.sprites_collection.RPS_enemy_turn[
            enemy_option
        ].center_y = self.enemy_option_draw_position_arcade.y
        self.game_manager.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_manager.sprites_collection.RPS_enemy_turn[enemy_option],
        )
        # result title
        self.game_manager.sprites_collection.RPS_outcomes[
            outcome
        ].center_x = self.outcome_draw_position_arcade.x
        self.game_manager.sprites_collection.RPS_outcomes[
            outcome
        ].center_y = self.outcome_draw_position_arcade.y
        self.game_manager.scene.add_sprite(
            self.main_sprite_list_name,
            self.game_manager.sprites_collection.RPS_outcomes[outcome],
        )
        # web camera
        self.game_manager.setup_web_camera_preview_in_scene()

    def _draw_current_option_selection(self) -> None:
        if self.current_option_selection is None:
            return
        option = self.current_option_selection.selected_object
        lt_arcade_pos = self.option_progress_bar_draw_positions[option]
        arcade.draw_lrtb_rectangle_filled(
            left=lt_arcade_pos.x,
            top=lt_arcade_pos.y,
            bottom=lt_arcade_pos.y - self.option_progress_bar_height,
            right=lt_arcade_pos.x
            + self.option_progress_bar_max_width
            * self.current_option_selection.proportion_of_completed_time,
            color=self.option_progress_bar_color,
        )

    def _update_selected_menu_option(self) -> None:
        active_geture_selection: Optional[
            TimeTrackedEntity[StaticGesture]
        ] = self.game_manager.hand_detection_state.active_gesture_time_track
        if active_geture_selection is None:
            self.rps_option_selection = None
            return
        menu_option = self.gesture_option_mapping.get(
            active_geture_selection.entity, None
        )
        if menu_option is None:
            self.rps_option_selection = None
            return
        self.current_option_selection = TimeBasedSelection(
            selected_object=menu_option,
            accumulated_time_seconds=active_geture_selection.tracked_time_seconds,
            time_requirement_seconds=self.option_selection_time_requirement_seconds,
        )

    def on_draw(self) -> None:
        self.clear()
        self.game_manager.scene.draw()
        self._draw_current_option_selection()
        if self.game_manager.hand_detection_state.active_track is not None:
            gesture_detection: GestureDetection = (
                self.game_manager.hand_detection_state.active_track.last.object
            )
            self.game_manager.draw_gesture_detection_in_web_camera(
                gesture_detection=gesture_detection, active=True
            )

    def on_update(self, delta_time: float):
        self.game_manager.update_inner_state(delta_time)
        self._update_selected_menu_option()
        if self.current_option_selection is None:
            return
        if self.current_option_selection.is_active:
            if self.current_option_selection.selected_object == "replay":
                self.game_manager.replay()


class RockPaperScissorsGameManager(arcade.View):
    def __init__(
        self,
        game_manager: GameManagerProtocol,
    ) -> None:
        super().__init__()
        self.game_manager = game_manager
        self.gesture_selection = RPSGestureSelectionView(game_manager=self)
        self.results_screen = RPSResultsView(game_manager=self)

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

    # Game Manager Protocol Implementation
    def update_inner_state(self, time_delta: float) -> None:
        self.game_manager.update_inner_state(time_delta=time_delta)

    def setup_web_camera_preview_in_scene(self) -> None:
        self.game_manager.setup_web_camera_preview_in_scene()

    def draw_gesture_detection_in_web_camera(
        self, gesture_detection: GestureDetection, active: bool
    ) -> None:
        self.game_manager.draw_gesture_detection_in_web_camera(
            gesture_detection=gesture_detection, active=active
        )

    def recreate_scene(self):
        self.game_manager.recreate_scene()

    @property
    def hand_detection_state(self):
        return self.game_manager.hand_detection_state

    @property
    def sprites_collection(self):
        return self.game_manager.sprites_collection

    @property
    def scene(self):
        return self.game_manager.scene


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def test_game(cfg: StaticGestureConfig):
    """Main function"""
    window = arcade.Window(SCREEN_WIDTH, SCREEN_HEIGHT, "Demo")
    yolo_hand_detector = YoloInferece(
        model_path=YOLO_V7_HAND_DETECTION, input_resolution=YOLO_V7_INPUT_RESOLUTION
    )
    gesture_classifier = StaticGestureClassifier.load_from_checkpoint(
        "E:\\dev\\MyFirstDataProject\\training_results\\STAT-87\\checkpoints\\checkpoint_epoch=12-val_weighted_F1=0.68.ckpt",
        cfg=cfg,
        results_location=None,
    )
    gesture_classifier.eval()
    gesture_classifier.to("cuda")
    val_augs = init_augmentations_from_config(augs_cfg=cfg.augs)[DataSplit.VAL]

    hands_det_state = HandDetectionState(
        gesture_classifier=gesture_classifier,
        hand_detector=yolo_hand_detector,
        gesture_classifier_preprocessing=val_augs,
        tracks_buffer_size=20,
    )
    game_manager = GameManagerPlaceHolder(
        hand_detection=hands_det_state,
        sprites_collection=SpriteCollection(GAME_RESOUSES_DIR),
    )
    # gesture_view = RPSGestureSelectionView(game_manager=game_manager)
    # gesture_view.setup()
    # gesture_view = RPSResultsView(game_manager=game_manager)
    # gesture_view.setup(
    #     hero_option=RPSGameOption.SCISSORS,
    #     enemy_option=RPSGameOption.PAPER,
    #     outcome=EndGameResult.Win,
    # )
    # window.show_view(gesture_view)
    rps_game = RockPaperScissorsGameManager(game_manager=game_manager)
    rps_game.setup()
    arcade.run()


test_game()
