import cv2
import numpy as np
from general.datasets.read_meta_dataset import ReadMetaDataset
from static_gesture_classification.custom_static_gesure_record import (
    CustomStaticGestureRecord,
)
from typing import Any
from PIL import Image


class CustomRecordDataset(ReadMetaDataset):
    def __init__(self, custom_record: CustomStaticGestureRecord) -> None:
        self._record: CustomStaticGestureRecord = custom_record
        self._bbox = self._record.meta_xyxy_box
        self._gesture_index = self._record.meta_gesture.value
        self._images_paths = self._record.images_paths

    def read_meta(self, index: int) -> Any:
        return {
            "bbox": self._bbox,
            "image_path": self._images_paths[index],
            "label": self._gesture_index,
        }

    def _get_rgb_crop_from_meta(self, meta) -> np.ndarray:
        image = Image.open(meta["image_path"])
        try:
            rgb_image = image.convert("RGB")
        except:
            print(meta["image_path"])
        rgb_crop = rgb_image.crop(meta["bbox"])
        return rgb_crop

    def __len__(self) -> int:
        return len(self._images_paths)

    def __getitem__(self, index: int) -> Any:
        sample = self.read_meta(index)
        crop = self._get_rgb_crop_from_meta(sample)
        sample["image"] = crop
        return sample
