import cv2
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from typing import Tuple, List, Optional
from general.utils import get_new_pattern_name_folder
from static_gesture_classification.static_gesture import StaticGesture
from static_gesture_classification.custom_static_gesure_record import (
    CustomStaticGestureRecord,
)
from static_gesture_classification.data_loading.custom_dataset import (
    CustomRecordDataset,
)
from const import (
    CUSTOM_TRAIN_ROOT,
    CUSTOM_VAL_ROOT,
    CUSTOM_DATA_ROOT,
    CUSTOM_VAL_VIZ_ROOT,
    CUSTOM_TRAIN_VIZ_ROOT,
    CUSTOM_PRESPLIT_ROOT,
)
from glob import glob
from matplotlib.axes._axes import Axes
from general.utils import plot_images_in_grid
from sklearn.model_selection import train_test_split
import shutil


def generate_plot_for_records(
    records: List[CustomStaticGestureRecord], axes: List[List[Axes]]
) -> Axes:
    col_num = len(axes)
    row_num = len(axes[0])
    images: List[np.ndarray] = []
    ds: CustomRecordDataset
    for record in records:
        ds = CustomRecordDataset(record)
        middle_sample = ds[len(ds) // 2]
        images.append(middle_sample["image"])
    plot_images_in_grid(axes=axes, images=images)
    for grid_tile_index, record in enumerate(records):
        ds = CustomRecordDataset(record)
        i, j = np.unravel_index(grid_tile_index, (col_num, row_num))
        axes[i][j].set_title(f"{len(ds)}")


def spawn_record_from_images_and_meta(
    images_paths: List[str], meta_path: str, new_record_parent_folder: str
):
    new_record_root = get_new_pattern_name_folder(
        root=new_record_parent_folder, pattern="record"
    )
    new_record = CustomStaticGestureRecord(new_record_root)
    shutil.copy(src=meta_path, dst=new_record.meta_path)
    for image_path in images_paths:
        shutil.move(src=image_path, dst=new_record.new_image_path)


def split_record_on_train_val(
    record: CustomStaticGestureRecord, train_part_size: float
):
    images_paths = record.images_paths
    train_images, val_images = train_test_split(
        images_paths, train_size=train_part_size
    )
    spawn_record_from_images_and_meta(
        images_paths=train_images,
        meta_path=record.meta_path,
        new_record_parent_folder=CUSTOM_TRAIN_ROOT,
    )
    spawn_record_from_images_and_meta(
        images_paths=val_images,
        meta_path=record.meta_path,
        new_record_parent_folder=CUSTOM_VAL_ROOT,
    )


def spawn_new_train_val_records_from_presplit_material(train_part_size: float):
    presplit_records_roots = glob(os.path.join(CUSTOM_PRESPLIT_ROOT, "*"))
    presplit_records = [
        CustomStaticGestureRecord(root) for root in presplit_records_roots
    ]
    for record in presplit_records:
        split_record_on_train_val(record=record, train_part_size=train_part_size)
    shutil.rmtree(CUSTOM_PRESPLIT_ROOT)
    os.makedirs(CUSTOM_PRESPLIT_ROOT)


def visualize_records_content_for_each_gesture(records_root: str, output_root: str):
    """For each recorded gesture write"""
    grid_size = (5, 5)
    grid_volume = grid_size[0] * grid_size[1]
    records_paths: List[str] = glob(os.path.join(records_root, "*"))
    records: List[CustomStaticGestureRecord] = [
        CustomStaticGestureRecord(record_path) for record_path in records_paths
    ]
    for gesture in StaticGesture:
        gesture_records: List[CustomStaticGestureRecord] = list(
            filter(lambda record: record.meta_gesture == gesture, records)
        )
        if len(gesture_records) > grid_volume:
            raise NotImplementedError(
                f"{len(gesture_records)} plots for {gesture.name} will not be drawn, due to grid volume limit {grid_volume}"
            )
        if not gesture_records:
            continue
        fig, axes = plt.subplots(*grid_size)
        generate_plot_for_records(gesture_records, axes)
        plt.savefig(fname=os.path.join(output_root, f"{gesture.name}.png"))
        plt.close(fig)


def visualize_current_custom_dataset_content():
    if os.path.exists(CUSTOM_TRAIN_VIZ_ROOT):
        shutil.rmtree(CUSTOM_TRAIN_VIZ_ROOT)
    os.makedirs(CUSTOM_TRAIN_VIZ_ROOT)
    if os.path.exists(CUSTOM_VAL_VIZ_ROOT):
        shutil.rmtree(CUSTOM_VAL_VIZ_ROOT)
    os.makedirs(CUSTOM_VAL_VIZ_ROOT)
    visualize_records_content_for_each_gesture(
        records_root=CUSTOM_TRAIN_ROOT, output_root=CUSTOM_TRAIN_VIZ_ROOT
    )
    visualize_records_content_for_each_gesture(
        records_root=CUSTOM_VAL_ROOT, output_root=CUSTOM_VAL_VIZ_ROOT
    )


class StaticGestureRecorder:
    def __init__(self):
        self.web_camera_capturer = cv2.VideoCapture(0)
        self._window_name = "StaticGestureRecorder"
        self._window = cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
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
            146,
            min(self.frame_height, self.frame_width),
            box_side_trackbar_callback,
        )

        cv2.createTrackbar(
            "Box X",
            self._window_name,
            80,
            self.frame_width - 1,
            box_x_pos_trackbar_callback,
        )

        cv2.createTrackbar(
            "Box Y",
            self._window_name,
            222,
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


def write_for_train():
    recorder = StaticGestureRecorder()
    recorder.run(CUSTOM_TRAIN_ROOT)


def write_for_val():
    recorder = StaticGestureRecorder()
    recorder.run(CUSTOM_VAL_ROOT)


def write_for_presplit():
    recorder = StaticGestureRecorder()
    recorder.run(CUSTOM_PRESPLIT_ROOT)


if __name__ == "__main__":
    spawn_new_train_val_records_from_presplit_material(train_part_size=0.8)
    # write_for_presplit()
    # visualize_current_custom_dataset_content()
