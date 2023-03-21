import os
from tqdm import tqdm
import numpy as np
from typing import List
from static_gesture_classification.data_loading.ndrczc35bt_dataset import (
    NdrczcMarkupTable,
    parse_xyxy_bbox_from_string,
    read_ndrczc_markup_in_train_ready_format,
    NdrczcDataset,
)
from static_gesture_classification.custom_static_gesure_record import (
    CustomStaticGestureRecord,
)
from static_gesture_classification.data_loading.custom_dataset import (
    CustomRecordDataset,
)

from const import DATA_ROOT
import cv2
import matplotlib.pyplot as plt
import pandas as p
from general.utils import traverse_all_files, keep_files_with_extension
from typing import Any, Dict


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
