import os
from tqdm import tqdm
import numpy as np
from typing import List
from static_gesture_classification.data_loading.ndrczc35bt_dataset import (
    NdrczcMarkupTable,
    parse_xyxy_bbox_from_string,
    read_ndrczc_markup_in_train_ready_format,
)
from const import DATA_ROOT
import cv2
import matplotlib.pyplot as plt
import pandas as p
from general.utils import traverse_all_files, keep_files_with_extension
from torch.utils.data import Dataset, Subset, ConcatDataset
from typing import Any, Dict


class ReadMetaDataset(Dataset):
    """Dataset that requires separate method for meta reading,
    In order to not read image when we need only meta, which should drastically analytic"""

    def read_meta(self, index) -> Any:
        raise NotImplementedError


class NdrczcDataset(ReadMetaDataset):
    def __init__(self, csv_meta_path: str) -> None:
        self.meta_table = read_ndrczc_markup_in_train_ready_format(csv_meta_path)

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


def draw_sample(absolute_path: str, box: str, label: str) -> None:
    image = cv2.imread(absolute_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    x, y, w, h = map(int, parse_xyxy_bbox_from_string(box))
    crop = image[y : y + h, x : x + w]
    plt.imshow(crop)
    plt.title(label)
    plt.show()


def plot_samples(csv_path: str):
    transformed_table = read_ndrczc_markup_in_train_ready_format(csv_path)
    for _, data in transformed_table.iterrows():
        draw_sample(
            absolute_path=data["image"], box=data["bbox"], label=data["gesture"]
        )
        break


def draw_samples():
    meta_root = "E:\\dev\\MyFirstDataProject\\Data\\ndrczc35bt-1"
    all_files = traverse_all_files(meta_root)
    all_txt_files = keep_files_with_extension(all_files, extension=".txt")
    for txt_file in all_txt_files:
        print(txt_file)
        plot_samples(csv_path=txt_file)


def summary_length():
    meta_root = "E:\\dev\\MyFirstDataProject\\Data\\ndrczc35bt-1"
    all_files = traverse_all_files(meta_root)
    all_txt_files = keep_files_with_extension(all_files, extension=".txt")
    length = 0
    for txt_file in all_txt_files:
        table = read_ndrczc_markup_in_train_ready_format(txt_file)
        length += len(table)
    print(f"summary length = {length}")
