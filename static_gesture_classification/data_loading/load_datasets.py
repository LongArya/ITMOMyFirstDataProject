import os
import pandas as pd
from glob import glob
from static_gesture_classification.data_loading.ndrczc35bt_dataset import (
    prepare_ndrczc_dataset_for_static_gesture_classification,
    NdrczcDataset,
    NdrczcMarkupTable,
)
from const import (
    CUSTOM_DATA_ROOT,
    NDRCZC_DATA_ROOT,
    HAGRID_META_ROOT,
    HAGRID_IMAGES_ROOT,
)
from static_gesture_classification.data_loading.custom_dataset import (
    CustomRecordDataset,
    CustomStaticGestureRecord,
)
from general.utils import traverse_all_files
from general.data_structures.data_split import DataSplit
from static_gesture_classification.data_loading.hagrid_dataset import HagridDataset
from general.datasets.read_meta_dataset import ReadMetaDataset, ReadMetaConcatDataset
from static_gesture_classification.static_gesture import StaticGesture
from general.datasets.read_meta_dataset import ReadMetaSubset
from typing import List, Tuple


def get_dataset_subset_with_gesture(dataset, label: StaticGesture):
    indexes = []
    for i in range(len(dataset)):
        meta = dataset.read_meta(i)
        if StaticGesture(meta["label"]) == label:
            indexes.append(i)
    return ReadMetaSubset(dataset, indexes)


def get_dataset_unique_labels(dataset: ReadMetaDataset) -> List[int]:
    labels: List[int] = []
    for i in range(len(dataset)):
        meta = dataset.read_meta(i)
        labels.append(meta["label"])
    unique_labels = list(set(labels))
    return unique_labels


def compose_custom_dataset_from_records(records_root: str) -> ReadMetaDataset:
    custom_records_paths: List[str] = glob(os.path.join(records_root, "*"))
    custom_records: List[CustomStaticGestureRecord] = [
        CustomStaticGestureRecord(path) for path in custom_records_paths
    ]
    custom_datasets: List[CustomRecordDataset] = [
        CustomRecordDataset(record) for record in custom_records
    ]
    custom_dataset = ReadMetaConcatDataset(custom_datasets)
    return custom_dataset


def load_ndrczc_train_dataset() -> ReadMetaDataset:
    train_split_path = os.path.join(
        NDRCZC_DATA_ROOT, "CUSTOM_TRAIN_TEST_SPLIT", "train.csv"
    )
    train_markup_table: NdrczcMarkupTable = pd.read_csv(train_split_path, index_col=0)
    train_dataset = NdrczcDataset(train_markup_table)
    train_dataset = prepare_ndrczc_dataset_for_static_gesture_classification(
        train_dataset
    )
    return train_dataset


def load_ndrczc_val_dataset() -> ReadMetaDataset:
    val_split_path = os.path.join(
        NDRCZC_DATA_ROOT, "CUSTOM_TRAIN_TEST_SPLIT", "val.csv"
    )
    val_markup_table: NdrczcMarkupTable = pd.read_csv(val_split_path, index_col=0)
    val_dataset = NdrczcDataset(val_markup_table)
    val_dataset = prepare_ndrczc_dataset_for_static_gesture_classification(val_dataset)
    return val_dataset


def load_custom_train_dataset() -> ReadMetaDataset:
    custom_train_root: str = os.path.join(CUSTOM_DATA_ROOT, "train")
    custom_train_dataset = compose_custom_dataset_from_records(
        records_root=custom_train_root
    )
    return custom_train_dataset


def load_custom_val_dataset() -> ReadMetaDataset:
    custom_val_root: str = os.path.join(CUSTOM_DATA_ROOT, "val")
    custom_val_dataset = compose_custom_dataset_from_records(
        records_root=custom_val_root
    )
    return custom_val_dataset


def compose_hagrid_dataset(data_split: DataSplit):
    hagrid_json_files = traverse_all_files(HAGRID_META_ROOT)
    hagrid_datasets: List[ReadMetaDataset] = []
    for json_file in hagrid_json_files:
        json_name = os.path.splitext(os.path.basename(json_file))[0]
        image_folder = os.path.join(HAGRID_IMAGES_ROOT, json_name, data_split.name)
        dataset = HagridDataset(image_folder=image_folder, json_file=json_file)
        hagrid_datasets.append(dataset)
    combined_dataset = ReadMetaConcatDataset(hagrid_datasets)
    return combined_dataset


def load_train_dataset() -> ReadMetaDataset:
    custom_train = load_custom_train_dataset()
    ndrczc_train = load_ndrczc_train_dataset()
    hagrid_train = compose_hagrid_dataset(DataSplit.TRAIN)
    train_ds = ReadMetaConcatDataset([custom_train, ndrczc_train, hagrid_train])
    return train_ds


def load_val_dataset() -> ReadMetaDataset:
    custom_val = load_ndrczc_val_dataset()
    ndrczc_val = load_custom_val_dataset()
    hagrid_val = compose_hagrid_dataset(DataSplit.VAL)
    val_ds = ReadMetaConcatDataset([custom_val, ndrczc_val, hagrid_val])
    return val_ds
