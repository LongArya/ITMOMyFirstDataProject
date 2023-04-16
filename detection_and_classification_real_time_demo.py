import torch
from typing import Tuple
import matplotlib.pyplot as plt
import onnxruntime
import numpy as np
import cv2
from yolo_detection import YoloInferece
from const import YOLO_V7_INPUT_RESOLUTION, YOLO_V7_HAND_DETECTION
from static_gesture_classification.config import StaticGestureConfig
import os
import hydra
from hydra.core.config_store import ConfigStore
from const import STATIC_GESTURE_CFG_NAME, STATIC_GESTURE_CFG_ROOT
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
    init_augmentations_from_config,
)
from typing import Callable, List, Iterable
from PIL import Image
from static_gesture_classification.static_gesture import StaticGesture
from general.data_structures.data_split import DataSplit

os.environ["HYDRA_FULL_ERROR"] = "1"
cs = ConfigStore.instance()
cs.store(name=STATIC_GESTURE_CFG_NAME, node=StaticGestureConfig)


def get_minimum_square_box_containing_box(xyxy_box: Iterable[float]) -> List[int]:
    x1, y1, x2, y2 = xyxy_box
    box_width = x2 - x1
    box_height = y2 - y1
    square_side = int(max(box_width, box_height))
    box_mid_x = (x1 + x2) / 2
    box_mid_y = (y1 + y2) / 2
    new_x1 = int(box_mid_x - square_side / 2)
    new_y1 = int(box_mid_y - square_side / 2)
    new_x2 = new_x1 + square_side
    new_y2 = new_y1 + square_side
    return [new_x1, new_y1, new_x2, new_y2]


def get_prediction_from_hand_detection(
    rgb_frame: np.ndarray,
    model: StaticGestureClassifier,
    detected_box: np.ndarray,
    val_pipeline: Callable[[Image.Image], torch.Tensor],
) -> Tuple[StaticGesture, float]:
    image: Image.Image = Image.fromarray(rgb_frame, mode="RGB")
    squared_box = get_minimum_square_box_containing_box(detected_box)
    crop = image.crop(squared_box)
    input: torch.Tensor = val_pipeline(crop)
    return model.get_gesture_prediction_for_single_input(input)


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def real_time_demo(cfg: StaticGestureConfig):
    window = cv2.namedWindow("DetectorRealTimeDemo", cv2.WINDOW_FULLSCREEN)
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

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.fliplr(frame)

        predictions = yolo_hand_detector(frame)
        boxes = predictions[:, :4].astype(np.int32)
        pred_viz = frame.copy()
        for box in boxes:
            x1, y1, x2, y2 = box
            gesture, prob = get_prediction_from_hand_detection(
                rgb_frame=frame,
                model=gesture_classifier,
                val_pipeline=val_augs,
                detected_box=box,
            )
            pred_viz = cv2.rectangle(pred_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            pred_viz = cv2.putText(
                img=pred_viz,
                text=f"{gesture.name}_{prob:.2f}",
                org=(x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(0, 255, 0),
            )
        pred_viz = cv2.cvtColor(pred_viz, cv2.COLOR_RGB2BGR)
        cv2.imshow("DetectorRealTimeDemo", pred_viz)
        cv2.waitKey(1)


real_time_demo()
