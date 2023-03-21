import os
import json
from glob import glob
from dataclasses import dataclass
from typing import List
from general.utils import get_new_pattern_name_folder
from static_gesture_classification.static_gesture import StaticGesture


@dataclass
class CustomStaticGestureRecord:
    root_folder: str

    def __post_init__(self) -> None:
        os.makedirs(self.images_folder, exist_ok=True)

    @property
    def meta_path(self) -> str:
        return os.path.join(self.root_folder, "meta.json")

    @property
    def images_folder(self) -> str:
        return os.path.join(self.root_folder, "images")

    def _read_meta_field(self, field: str):
        with open(self.meta_path, "r") as f:
            meta = json.load(f)
        return meta[field]

    @property
    def meta_xyxy_box(self) -> List[int]:
        box = self._read_meta_field("bbox")
        return box

    @property
    def meta_gesture(self) -> StaticGesture:
        gesture_name: str = self._read_meta_field("gesture")
        gesture: StaticGesture = StaticGesture[gesture_name]
        return gesture

    @property
    def new_image_path(self) -> str:
        new_frame_path = get_new_pattern_name_folder(self.images_folder, "image")
        new_frame_path = f"{new_frame_path}.png"
        return new_frame_path

    @property
    def images_paths(self):
        return sorted(glob(os.path.join(self.images_folder, "*.png")))
