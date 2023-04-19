from yolo_detection import YoloInferece
import numpy as np
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
)
from typing import TypeVar, Generic, List, Optional, Callable, Tuple, Iterable
from dataclasses import dataclass
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from MVP.data_structures.track import Track, TrackedObject
from MVP.data_structures.gesture_detection import GestureDetection
from MVP.tracking_utils import (
    get_hungarian_algorithm_matches,
    get_fp_instances,
    get_fn_instances,
)
from MVP.data_structures.match import Match
from general.utils import get_xyxy_boxes_iou, timing_decorator
from copy import deepcopy
from PIL import Image
import torch
from static_gesture_classification.static_gesture import StaticGesture

T = TypeVar("T")


ACTIVE_HAND_STEAL_TIME_FRAMES = 5


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


class HandDetectionState:
    def __init__(
        self,
        hand_detector: YoloInferece,
        gesture_classifier: StaticGestureClassifier,
        gesture_classifier_preprocessing: Callable[[Image.Image], torch.Tensor],
        tracks_buffer_size: Optional[int],
    ):
        self.gesture_classifier_preprocessing = gesture_classifier_preprocessing
        self.gesture_detections_tracks: List[Track[GestureDetection]] = []
        self.hand_detector: YoloInferece = hand_detector
        self.gesture_classifier: StaticGestureClassifier = gesture_classifier
        self.tracks_buffer_size: Optional[int] = tracks_buffer_size
        self.tracks_matcher: Callable[
            [TrackedObject[GestureDetection], Track[GestureDetection]],
            float,
        ] = lambda tracked_obj, track: -get_xyxy_boxes_iou(
            tracked_obj.object.xyxy_box, track.last.object.xyxy_box
        )
        self.match_iou_thrd: float = 0.6
        self.tracks_TTL_after_dissapearance_frames: int = 10
        self.active_track: Optional[Track[GestureDetection]] = None
        self.max_height_active_requirement_frames: int = 10

    @property
    def new_track_id(self) -> int:
        current_max_id: int
        try:
            current_max_id = max(
                [track.track_id for track in self.gesture_detections_tracks]
            )
        except ValueError:
            current_max_id = 0
        return current_max_id + 1

    def _choose_active_track_based_on_detection_height(self, N: int):
        """Active track is the one where hand where the highest last N frames"""
        # choose active track ID
        pass

    @timing_decorator
    def _produce_gesture_detections(
        self, image: np.ndarray, frame: int
    ) -> List[TrackedObject[GestureDetection]]:
        new_tracked_objects: List[TrackedObject[GestureDetection]] = []
        predictions = self.hand_detector(image)
        boxes = predictions[:, :4].astype(np.int32)
        for box in boxes:
            gesture, prob = get_prediction_from_hand_detection(
                rgb_frame=image,
                model=self.gesture_classifier,
                val_pipeline=self.gesture_classifier_preprocessing,
                detected_box=box,
            )
            gesture_detection: GestureDetection = GestureDetection(
                gesture=gesture, xyxy_box=box, score=prob
            )
            new_tracked_objects.append(
                TrackedObject(object=gesture_detection, frame=frame)
            )
        return new_tracked_objects

    @timing_decorator
    def _remove_stale_tracks(self, current_frame: int) -> None:
        stale_track_indexes = [
            i
            for i, track in enumerate(self.gesture_detections_tracks)
            if track.is_stale(
                current_frame=current_frame,
                expiration_frame_num=self.tracks_TTL_after_dissapearance_frames,
            )
        ]
        self.gesture_detections_tracks = [
            track
            for i, track in enumerate(self.gesture_detections_tracks)
            if i not in stale_track_indexes
        ]

    @timing_decorator
    def _update_tracks_with_tracked_objects(
        self,
        tracked_objects: List[TrackedObject[GestureDetection]],
    ) -> None:
        matches: List[Match]
        weights_matrix: np.ndarray
        matches, weights_matrix = get_hungarian_algorithm_matches(
            ground_true=self.gesture_detections_tracks,
            predictions=tracked_objects,
            weight_measurer=self.tracks_matcher,
        )
        matches = list(
            filter(
                lambda match: -weights_matrix[match.gt_value][match.pred_value]
                >= self.match_iou_thrd,
                matches,
            )
        )
        for match in matches:
            self.gesture_detections_tracks[match.gt_value].extend_track(
                tracked_objects[match.pred_value]
            )
        fp_tracked_objects = get_fp_instances(tracked_objects, matches)
        for fp_object in fp_tracked_objects:
            new_track: Track[GestureDetection] = Track(
                buffer_size=self.tracks_buffer_size, track_id=self.new_track_id
            )
            new_track.extend_track(fp_object)
            self.gesture_detections_tracks.append(new_track)

    def update_tracks_with_linearly_interpolated_boxes(self) -> None:
        # update all tracks with interpolated boxes
        pass

    def update_inner_state(self, image: np.ndarray, frame: int) -> None:
        # spawn gesture detections (inference detector, inference classifier)
        gesture_detections: List[
            TrackedObject[GestureDetection]
        ] = self._produce_gesture_detections(image=image, frame=frame)
        # match with current tracks, spawn new/extend current/remove stale
        self._update_tracks_with_tracked_objects(tracked_objects=gesture_detections)
        self._remove_stale_tracks(current_frame=frame)
        # select active track
        self._choose_active_track_based_on_detection_height(
            N=self.max_height_active_requirement_frames
        )

    @property
    def active_gesture_detection(self) -> GestureDetection:
        # return active gesture detection (one with the highest hand position/the longest/allow only one)
        # possibly smooth gesture based on track history/confidence scores (avoid temporarily mulfanctioning of detector/classifier)
        raise NotImplementedError

    @property
    def nonactive_gesture_detections(self) -> List[GestureDetection]:
        return []
