import os
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import torch
from PIL import Image
import numpy as np
from typing import Tuple, List, Callable, Iterable
from const import (
    STATIC_GESTURE_CFG_NAME,
    STATIC_GESTURE_CFG_ROOT,
)
from static_gesture_classification.static_gesture import StaticGesture
import hydra
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
    init_augmentations_from_config,
)
from hydra.core.config_store import ConfigStore
from general.data_structures.data_split import DataSplit
from static_gesture_classification.config import StaticGestureConfig

os.environ["HYDRA_FULL_ERROR"] = "1"
cs = ConfigStore.instance()
cs.store(name=STATIC_GESTURE_CFG_NAME, node=StaticGestureConfig)
from general.utils import TorchNormalizeInverse
from train_static_gestures import neutralize_image_normalization


class RealTimeDemo:
    def __init__(self):
        self.web_camera_capturer = cv2.VideoCapture(0)
        self._window_name = "StaticGestureRecorder"
        self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_FULLSCREEN)
        self.frame_width, self.frame_height = self._get_web_camera_frame_resolution()
        self._box_side_pixel_length = 10
        self._box_x_position = 0
        self._box_y_position = 0
        self._init_trackbars()

    def _get_web_camera_frame_resolution(self) -> Tuple[int, int]:
        ret, frame = self.web_camera_capturer.read()
        height, width = frame.shape[:2]
        return (width, height)

    def _init_trackbars(self):
        def box_side_trackbar_callback(value: int) -> None:
            self._box_side_pixel_length = value

        def box_x_pos_trackbar_callback(value: int) -> None:
            self._box_x_position = value

        def box_y_pos_trackbar_callback(value: int) -> None:
            self._box_y_position = value

        cv2.createTrackbar(
            "Box side length",
            self._window_name,
            0,
            min(self.frame_height, self.frame_width),
            box_side_trackbar_callback,
        )

        cv2.createTrackbar(
            "Box X",
            self._window_name,
            0,
            self.frame_width - 1,
            box_x_pos_trackbar_callback,
        )

        cv2.createTrackbar(
            "Box Y",
            self._window_name,
            0,
            self.frame_height - 1,
            box_y_pos_trackbar_callback,
        )

    @property
    def current_rectangle(self) -> List[int]:
        x1 = np.clip(self._box_x_position, 0, self.frame_width - 1)
        y1 = np.clip(self._box_y_position, 0, self.frame_height - 1)
        x2 = np.clip(
            self._box_x_position + self._box_side_pixel_length, 0, self.frame_width - 1
        )
        y2 = np.clip(
            self._box_y_position + self._box_side_pixel_length, 0, self.frame_height - 1
        )
        return [x1, y1, x2, y2]

    def _get_frame_visualuzation(
        self, frame: np.ndarray, gesture: StaticGesture, prob: float
    ) -> np.ndarray:
        viz_frame = frame.copy()
        x1, y1, x2, y2 = self.current_rectangle
        viz_frame = cv2.rectangle(
            img=viz_frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2
        )
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.frame_height, self.frame_height))
        viz_frame = np.concatenate((viz_frame, crop), axis=1)
        status_bar_image = np.zeros((190, viz_frame.shape[1], 3), dtype=viz_frame.dtype)
        status_bar_image = cv2.putText(
            img=status_bar_image,
            text=f"Prediction: {gesture.name}, Score: {prob:.2f}",
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(0, 255, 0),
        )
        viz_frame = np.concatenate((status_bar_image, viz_frame), axis=0)
        return viz_frame

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        preprocessed_frame = frame.copy()
        preprocessed_frame = np.fliplr(preprocessed_frame)
        return preprocessed_frame

    def _get_prediction_on_crop(
        self,
        frame: np.ndarray,
        model: StaticGestureClassifier,
        val_pipeline: Callable[[Image.Image], torch.Tensor],
    ) -> Tuple[StaticGesture, float]:
        image: Image.Image = Image.fromarray(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mode="RGB"
        )
        crop = image.crop(self.current_rectangle)
        input: torch.Tensor = val_pipeline(crop)
        # backed_input = neutralize_image_normalization(
        #     input, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # )
        # plt.imshow(backed_input)
        # plt.show()
        return model.get_gesture_prediction_for_single_input(input)

    def run(
        self,
        model: StaticGestureClassifier,
        val_pipeline: Callable[[Image.Image], torch.Tensor],
    ):
        while True:
            # read frame
            ret, frame = self.web_camera_capturer.read()
            if not ret:
                raise ValueError("Invalid frame")
            frame = self._preprocess_frame(frame)
            gesture, prob = self._get_prediction_on_crop(frame, model, val_pipeline)
            viz = self._get_frame_visualuzation(frame, gesture, prob)
            cv2.imshow(self._window_name, viz)
            cv2.waitKey(1)


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def run_real_time_demo(cfg: StaticGestureConfig):
    model = StaticGestureClassifier.load_from_checkpoint(
        "E:\\dev\\MyFirstDataProject\\training_results\\STAT-87\\checkpoints\\checkpoint_epoch=12-val_weighted_F1=0.68.ckpt",
        cfg=cfg,
        results_location=None,
    )
    model.eval()
    model.to("cuda")
    val_augs = init_augmentations_from_config(augs_cfg=cfg.augs)[DataSplit.VAL]
    demo = RealTimeDemo()
    demo.run(model, val_pipeline=val_augs)


run_real_time_demo()
