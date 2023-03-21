import cv2
import json
import os
import numpy as np
from typing import Tuple, List, Optional
from general.utils import get_new_pattern_name_folder
from static_gesture_classification.static_gesture import StaticGesture
from static_gesture_classification.custom_static_gesure_record import (
    CustomStaticGestureRecord,
)


class StaticGestureRecorder:
    def __init__(self):
        self.web_camera_capturer = cv2.VideoCapture(0)
        self._window_name = "StaticGestureRecorder"
        self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_FULLSCREEN)
        self.frame_width, self.frame_height = self._get_web_camera_frame_resolution()
        self._box_side_pixel_length = 10
        self._box_x_position = 0
        self._box_y_position = 0
        self._is_recording = False
        self._current_gesture: StaticGesture = StaticGesture.BACKGROUND
        self._output_directory_pattern = "record"
        self._current_record: Optional[CustomStaticGestureRecord] = None
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

        def gesture_callback(value: int) -> None:
            self._current_gesture = StaticGesture(value)

        def record_callback(value: int) -> None:
            self._is_recording = bool(value)

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

        cv2.createTrackbar(
            "Gesture",
            self._window_name,
            1,
            len(StaticGesture),
            gesture_callback,
        )

        cv2.createTrackbar(
            "Record",
            self._window_name,
            0,
            1,
            record_callback,
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

    def _save_meta_for_current_record(self) -> None:
        if self._current_record is None:
            raise ValueError("Cannot save meta without established record")
        json_box = list(map(int, self.current_rectangle))
        meta_data = {
            "bbox": json_box,
            "gesture": self._current_gesture.name,
        }
        with open(self._current_record.meta_path, "w") as f:
            json.dump(meta_data, f)

    def _get_frame_visualuzation(self, frame: np.ndarray) -> np.ndarray:
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
            text=f"Record: {self._is_recording}",
            org=(10, 50),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(0, 255, 0),
        )
        status_bar_image = cv2.putText(
            img=status_bar_image,
            text=f"Gesture: {self._current_gesture.name}",
            org=(10, 100),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(0, 255, 0),
        )
        written_frames = (
            len(os.listdir(self._current_record.images_folder))
            if self._current_record is not None
            else 0
        )
        status_bar_image = cv2.putText(
            img=status_bar_image,
            text=f"Written frames: {written_frames}",
            org=(10, 150),
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

    def run(self, save_folder: str):
        prev_recording_state: bool = False
        while True:
            # read frame
            ret, frame = self.web_camera_capturer.read()
            if not ret:
                raise ValueError("Invalid frame")
            frame = self._preprocess_frame(frame)
            viz = self._get_frame_visualuzation(frame)
            current_recording_state = self._is_recording
            # handle start record
            if current_recording_state and not prev_recording_state:
                new_record_root = get_new_pattern_name_folder(
                    root=save_folder, pattern=self._output_directory_pattern
                )
                self._current_record = CustomStaticGestureRecord(new_record_root)
                self._save_meta_for_current_record()
            # handle continuous recording
            if current_recording_state and prev_recording_state:
                cv2.imwrite(self._current_record.new_image_path, frame)
            # handle end record
            if not current_recording_state and prev_recording_state:
                self._current_record = None
            prev_recording_state = current_recording_state
            cv2.imshow(self._window_name, viz)
            cv2.waitKey(1)


def main():
    recorder = StaticGestureRecorder()
    recorder.run("E:\\dev\\MyFirstDataProject\\Data\\custom_data")


if __name__ == "__main__":
    main()
