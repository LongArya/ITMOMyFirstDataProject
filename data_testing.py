import os
from glob import glob
from tqdm import tqdm
from general.datasets.read_meta_dataset import ReadMetaSubset, ReadMetaDataset
from typing import List, Optional, Dict
from static_gesture_classification.data_loading.ndrczc35bt_dataset import (
    parse_xyxy_bbox_from_string,
    read_ndrczc_markup_in_train_ready_format,
    compose_ndrczc_dataset_for_static_gesture_classification,
)
from static_gesture_classification.custom_static_gesure_record import (
    CustomStaticGestureRecord,
)
from static_gesture_classification.data_loading.custom_dataset import (
    CustomRecordDataset,
)
from static_gesture_classification.static_gesture import StaticGesture
from static_gesture_classification.data_loading.load_datasets import load_train_dataset

from const import DATA_ROOT
import cv2
import matplotlib.pyplot as plt
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


def draw_dataset_sample(sample, save_path: Optional[str] = None) -> None:
    crop = sample["image"]
    label = sample["label"]
    try:
        label = StaticGesture(label).name
    except ValueError:
        pass
    plt.imshow(crop)
    plt.title(label)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


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


def get_dataset_subset_with_label(dataset, label):
    indexes = []
    for i in range(len(dataset)):
        meta = dataset.read_meta(i)
        if meta["label"] == label:
            indexes.append(i)
    return ReadMetaSubset(dataset, indexes)


def get_dataset_unique_labels(dataset: ReadMetaDataset) -> List[int]:
    labels: List[int] = []
    for i in range(len(dataset)):
        meta = dataset.read_meta(i)
        labels.append(meta["label"])
    unique_labels = list(set(labels))
    return unique_labels


def test_dataset():
    meta_root = os.path.join(DATA_ROOT, "ndrczc35bt-1")
    dataset = compose_ndrczc_dataset_for_static_gesture_classification(meta_root)
    print(len(dataset))


def check_nines():
    meta_root = os.path.join(DATA_ROOT, "ndrczc35bt-1")
    dataset = compose_ndrczc_dataset_for_static_gesture_classification(meta_root)
    subset = get_dataset_subset_with_label(dataset, "Nine_VFL")
    for sample in subset:
        draw_dataset_sample(sample)


def plot_dataset(dataset: CustomRecordDataset, save_path: str):
    fig, axis = plt.subplots(1, 3)
    for column_index, index in enumerate([0, len(dataset) // 2, len(dataset) - 1]):
        sample = dataset[index]
        axis[column_index].imshow(sample["image"])
        title = StaticGesture(sample["label"]).name
        axis[column_index].set_title(title)

    fig.savefig(save_path)


def plot_custom_records_datasets() -> None:
    records_root = "E:\\dev\\MyFirstDataProject\\Data\\custom_data"
    reocords_paths = sorted(glob(os.path.join(records_root, "*")))
    for record_path in reocords_paths:
        record_name = os.path.basename(record_path)
        record = CustomStaticGestureRecord(record_path)
        record_dataset = CustomRecordDataset(record)
        plot_dataset(dataset=record_dataset, save_path=f"{record_name}.png")


dataset = load_train_dataset()
print(len(dataset))
unique_labels = get_dataset_unique_labels(dataset)
print(len(unique_labels))
for label in unique_labels:
    subset = get_dataset_subset_with_label(dataset=dataset, label=label)
    print(StaticGesture(label), len(subset))
