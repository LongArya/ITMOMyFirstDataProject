import pandas as pd
import os
from typing import List
from general.datasets.read_meta_dataset import ReadMetaDataset
from typing import Dict, Any
import cv2
import numpy as np

ZERO_BBOX = "[0 0 0 0]"


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
        bgr_image = cv2.imread(meta["image_path"])
        try:
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        except:
            print(meta["image_path"])
        x1, y1, x2, y2 = map(int, meta["bbox"])
        rgb_crop = rgb_image[y1:y2, x1:x2]
        return rgb_crop

    def __len__(self) -> int:
        return len(self.meta_table)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.read_meta(index)
        crop = self._get_rgb_crop_from_meta(sample)
        sample["image"] = crop
        return sample


def parse_xyxy_bbox_from_string(bbox_str: str):
    bbox_without_braces = bbox_str.replace("[", "")
    bbox_without_braces = bbox_without_braces.replace("]", "")
    x1, y1, w, h = map(float, bbox_without_braces.split(" "))
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]


def get_absolute_path_to_the_image(csv_path: str, relative_path: str) -> str:
    csv_name = os.path.basename(csv_path)
    csv_name = os.path.splitext(csv_name)[0]
    meta_root = os.path.dirname(os.path.dirname(csv_path))

    norm_path = os.path.normpath(relative_path)
    folders_split = norm_path.split(os.sep)
    folders_split[-3] = csv_name
    folders_split[-2] = folders_split[-3]
    fixed_relative_path = os.path.join(*folders_split[1:])
    absolute_path = os.path.join(meta_root, fixed_relative_path)
    return absolute_path


def read_ndrczc_markup_in_train_ready_format(csv_path: str) -> NdrczcMarkupTable:
    csv_table = pd.read_csv(csv_path)
    modified_table_data: List[List[str]] = []
    gesture_columns = csv_table.columns[2:]
    for _, data in csv_table.iterrows():
        for gesture in gesture_columns:
            bbox_str = data[gesture]
            if bbox_str != ZERO_BBOX:
                absolute_image_path = get_absolute_path_to_the_image(
                    csv_path=csv_path, relative_path=data["rgb"]
                )
                if not os.path.exists(absolute_image_path):
                    continue

                row = [absolute_image_path, gesture, bbox_str]
                modified_table_data.append(row)
    modified_table = pd.DataFrame(
        modified_table_data, columns=["image", "gesture", "bbox"]
    )
    return modified_table


def compose_ndrczc_dataset():
    pass
