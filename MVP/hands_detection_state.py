from yolo_detection import YoloInferece
import numpy as np
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
)
from typing import TypeVar, Generic, List, Optional
from dataclasses import dataclass
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.track import Track
from MVP.data_structures.gesture_detection import GestureDetection

T = TypeVar("T")


ACTIVE_HAND_STEAL_TIME_FRAMES = 5


class HandDetectionState:
    def __init__(
        self,
        hand_detector: YoloInferece,
        gesture_classifier: StaticGestureClassifier,
        tracks_buffer_size: Optional[int],
    ):
        self.gesture_detections_tracks: List[Track[GestureDetection]] = []
        self.hand_detector: YoloInferece = hand_detector
        self.gesture_classifier: StaticGestureClassifier = gesture_classifier
        self.tracks_buffer_size: Optional[int] = tracks_buffer_size

    def _choose_active_track_based_on_detection_height(self, N: int) -> Optional[int]:
        """Active track is the one where hand where the highest last N frames"""
        # choose active track ID
        if not self.gesture_detections_tracks:
            return None
        highest_box_history: List[int] = []
        return 0

    def _produce_gesture_detections(self, image: np.ndarray) -> List[GestureDetection]:
        return []

    def update_inner_state(self, image: np.ndarray) -> None:
        # spawn gesture detections (inference detector, inference classifier)
        gesture_detections: List[GestureDetection] = self._produce_gesture_detections(
            image
        )
        # match with current tracks, spawn new/extend current/remove stale
        # select active track
        pass

    @property
    def active_gesture_detection(self) -> GestureDetection:
        # return active gesture detection (one with the highest hand position/the longest/allow only one)
        # possibly smooth gesture based on track history/confidence scores (avoid temporarily mulfanctioning of detector/classifier)
        pass

    @property
    def nonactive_gesture_detections(self) -> List[GestureDetection]:
        return []
