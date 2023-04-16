import os
import shutil
from general.data_structures.data_split import DataSplit
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import json
from general.datasets.read_meta_dataset import (
    ReadMetaDataset,
    ReadMetaConcatDataset,
)
from typing import List, Optional, Dict
from static_gesture_classification.static_gesture import StaticGesture
from const import HAGRID_IMAGES_ROOT, HAGRID_META_ROOT
from general.utils import traverse_all_files
from typing import Any, Dict


HAGRID_GESTURES_MAPPING: Dict[str, StaticGesture] = {
    "like": StaticGesture.THUMBS_UP,
    "two_up": StaticGesture.TWO,
    "three2": StaticGesture.THREE,
    "four": StaticGesture.FOUR,
    "rock": StaticGesture.BACKGROUND,
    "stop_inverted": StaticGesture.FIVE,
    "three": StaticGesture.THREE,
    "two_up_inverted": StaticGesture.TWO,
    "no_gesture": StaticGesture.BACKGROUND,
    "one": StaticGesture.ONE,
    "palm": StaticGesture.FIVE,
    "fist": StaticGesture.FIST,
    "call": StaticGesture.BACKGROUND,
    "peace_inverted": StaticGesture.TWO,
    "peace": StaticGesture.TWO,
    "mute": StaticGesture.ONE,
    "ok": StaticGesture.OKEY,
    "stop": StaticGesture.FIVE,
    "dislike": StaticGesture.THUMBS_DOWN,
}


class HagridDataset(ReadMetaDataset):
    def __init__(self, image_folder: str, json_file: str) -> None:
        self.meta_items: List[Dict] = self._prepare_meta(
            image_folder=image_folder, json_file=json_file
        )

    def _find_single_matching_image_by_name(
        self, image_folder: str, image_name: str
    ) -> Optional[str]:
        image_folder_content = traverse_all_files(image_folder)
        matching_images = list(
            filter(
                lambda path: os.path.splitext(os.path.basename(path))[0] == image_name,
                image_folder_content,
            )
        )
        if len(matching_images) != 1:
            return None
        return matching_images[0]

    def _prepare_meta(self, image_folder: str, json_file: str) -> List[Dict]:
        with open(json_file, "r") as f:
            meta_data: Dict = json.load(f)
        prepared_meta: List[Dict] = []
        for image_name, meta in meta_data.items():
            image_path: Optional[str] = self._find_single_matching_image_by_name(
                image_folder=image_folder, image_name=image_name
            )
            if image_path is None:
                continue
            for box, label in zip(meta["bboxes"], meta["labels"]):
                gesture: StaticGesture = HAGRID_GESTURES_MAPPING[label]
                meta_item = {
                    "image_path": image_path,
                    "original_label": label,
                    "bbox": box,
                    "label": gesture.value,
                }
                prepared_meta.append(meta_item)
        return prepared_meta

    def __len__(self) -> int:
        return len(self.meta_items)

    def read_meta(self, index: int) -> Any:
        return self.meta_items[index]

    def _get_rgb_crop_from_meta(self, meta) -> np.ndarray:
        image = Image.open(meta["image_path"])
        try:
            rgb_image = image.convert("RGB")
        except:
            print(meta["image_path"])
        x1, y1, w, h = meta["bbox"]
        width, height = image.size
        x1 *= width
        y1 *= height
        w *= width
        h *= height
        x2 = x1 + w
        y2 = y1 + h
        rgb_crop = rgb_image.crop([x1, y1, x2, y2])
        return rgb_crop

    def __getitem__(self, index) -> Dict[str, Any]:
        sample = self.read_meta(index)
        crop = self._get_rgb_crop_from_meta(sample)
        sample["image"] = crop
        return sample


def split_folder_content_as_train_val(folder: str, train_size: float):
    folder_content = traverse_all_files(folder)
    folder_files = list(filter(lambda path: os.path.isfile(path), folder_content))
    if not folder_files:
        return
    train_paths, val_paths = train_test_split(folder_files, train_size=train_size)
    train_root = os.path.join(folder, DataSplit.TRAIN.name)
    val_root = os.path.join(folder, DataSplit.VAL.name)
    os.makedirs(train_root)
    os.makedirs(val_root)
    for train_path in train_paths:
        shutil.move(train_path, train_root)
    for val_path in val_paths:
        shutil.move(val_path, val_root)
