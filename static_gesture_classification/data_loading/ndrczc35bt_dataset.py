import pandas as pd
import os
from typing import List
from general.datasets.read_meta_dataset import (
    ReadMetaDataset,
    ReadMetaConcatDataset,
    ReadMetaSubset,
)
from typing import Dict, Any
import cv2
from PIL import Image
import numpy as np
from general.utils import traverse_all_files, keep_files_with_extension
from static_gesture_classification.static_gesture import StaticGesture

ZERO_BBOX = "[0 0 0 0]"
NDRCZC_BANNED_LABELS = [
    "XSign",
    "Collab",
    "TimeOut",
]
NDRCZC_PREFIX_TO_GESTURES_MAPPING: Dict[str, StaticGesture] = {
    "Eight": StaticGesture.BACKGROUND,
    "Span": StaticGesture.BACKGROUND,
    "Horiz": StaticGesture.BACKGROUND,  # FIXME consider using it as FIVE with corresponding rotation
    "Seven": StaticGesture.BACKGROUND,
    "One": StaticGesture.ONE,
    "Two": StaticGesture.TWO,
    "Six": StaticGesture.THREE,
    "Three": StaticGesture.THREE,
    "Four": StaticGesture.FOUR,
    "Five": StaticGesture.FIVE,
    "Punch": StaticGesture.FIST,
    "Nine": StaticGesture.OKEY,
}


class NdrczcMarkupTable(pd.DataFrame):
    @property
    def image() -> pd.Series:
        ...

    @property
    def gesture() -> pd.Series:
        ...

    @property
    def bbox() -> pd.Series:
        ...


class NdrczczDatasetMetaResolver(ReadMetaDataset):
    """Resolves Ndrczcz gesture labels"""

    def __init__(self, ndrczc_dataset: ReadMetaDataset):
        self._base_dataset = ndrczc_dataset

    def _resolve_gesture_label(self, gesture_label: str) -> int:
        """Transforms original gesture label from ndrczc markup"""
        gesture_prefix = gesture_label.split("_")[0]
        gesture: StaticGesture = NDRCZC_PREFIX_TO_GESTURES_MAPPING[gesture_prefix]
        return gesture.value

    def read_meta(self, index) -> Any:
        meta = self._base_dataset.read_meta(index)
        meta["label"] = self._resolve_gesture_label(meta["label"])
        return meta

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._base_dataset[index]
        sample["label"] = self._resolve_gesture_label(sample["label"])
        return sample


class NdrczcDataset(ReadMetaDataset):
    def __init__(self, txt_meta_path: str) -> None:
        self.meta_table = read_ndrczc_markup_in_train_ready_format(txt_meta_path)

    def read_meta(self, index) -> Any:
        row = self.meta_table.iloc[index]
        return {
            "bbox": parse_xyxy_bbox_from_string(row["bbox"]),
            "image_path": row["image"],
            "label": row["gesture"],
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
        return len(self.meta_table)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.read_meta(index)
        crop = self._get_rgb_crop_from_meta(sample)
        sample["image"] = crop
        return sample


def parse_xyxy_bbox_from_string(bbox_str: str) -> List[float]:
    """Decipher box coordinates from string encoding in xyxy format"""
    bbox_without_braces = bbox_str.replace("[", "")
    bbox_without_braces = bbox_without_braces.replace("]", "")
    x1, y1, w, h = map(float, bbox_without_braces.split(" "))
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def get_absolute_path_to_the_image(txt_path: str, relative_path: str) -> str:
    """Gets absolute path to the image from ndrczc dataset,
    paths in markup are incorrect so they require additional postprocessing"""
    csv_name = os.path.basename(txt_path)
    csv_name = os.path.splitext(csv_name)[0]
    meta_root = os.path.dirname(os.path.dirname(txt_path))

    norm_path = os.path.normpath(relative_path)
    folders_split = norm_path.split(os.sep)
    folders_split[-3] = csv_name
    folders_split[-2] = folders_split[-3]
    fixed_relative_path = os.path.join(*folders_split[1:])
    absolute_path = os.path.join(meta_root, fixed_relative_path)
    return absolute_path


def read_ndrczc_markup_in_train_ready_format(txt_path: str) -> NdrczcMarkupTable:
    """Reads ndrczc markup"""
    csv_table = pd.read_csv(txt_path)
    modified_table_data: List[List[str]] = []
    gesture_columns = csv_table.columns[2:]
    for _, data in csv_table.iterrows():
        for gesture in gesture_columns:
            bbox_str = data[gesture]
            if bbox_str != ZERO_BBOX:
                absolute_image_path = get_absolute_path_to_the_image(
                    txt_path=txt_path, relative_path=data["rgb"]
                )
                if not os.path.exists(absolute_image_path):
                    continue

                row = [absolute_image_path, gesture, bbox_str]
                modified_table_data.append(row)
    modified_table = pd.DataFrame(
        modified_table_data, columns=["image", "gesture", "bbox"]
    )
    return modified_table


def compose_ndrczc_dataset_for_static_gesture_classification(
    ndrczc_meta_root: str,
) -> ReadMetaDataset:
    """Composes all ndrczc data, removes labels that are not fit for static gesture classification"""
    all_files = traverse_all_files(ndrczc_meta_root)
    all_txt_files = keep_files_with_extension(all_files, extension=".txt")
    ndrczc_datasets = [NdrczcDataset(txt_file) for txt_file in all_txt_files]
    ndrczc_dataset = ReadMetaConcatDataset(ndrczc_datasets)
    # filter samples with banned labels
    kept_indexes: List[int] = []
    for index in range(len(ndrczc_dataset)):
        meta = ndrczc_dataset.read_meta(index)
        label = meta["label"]
        if label not in NDRCZC_BANNED_LABELS:
            kept_indexes.append(index)

    ndrczc_dataset = ReadMetaSubset(ndrczc_dataset, indices=kept_indexes)
    ndrczc_dataset = NdrczczDatasetMetaResolver(ndrczc_dataset)
    return ndrczc_dataset
