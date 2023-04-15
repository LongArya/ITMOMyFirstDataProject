import os
from glob import glob
from static_gesture_classification.data_loading.ndrczc35bt_dataset import (
    compose_ndrczc_dataset_for_static_gesture_classification,
)
from const import CUSTOM_DATA_ROOT, NDRCZC_DATA_ROOT
from static_gesture_classification.data_loading.custom_dataset import (
    CustomRecordDataset,
    CustomStaticGestureRecord,
)
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


def load_custom_train_dataset() -> ReadMetaDataset:
    custom_train_root: str = os.path.join(CUSTOM_DATA_ROOT, "train")
    custom_train_records_paths: List[str] = glob(os.path.join(custom_train_root, "*"))
    custom_records: List[CustomStaticGestureRecord] = [
        CustomStaticGestureRecord(path) for path in custom_train_records_paths
    ]
    custom_datasets: List[CustomRecordDataset] = [
        CustomRecordDataset(record) for record in custom_records
    ]
    custom_train_dataset = ReadMetaConcatDataset(custom_datasets)
    return custom_train_dataset


def load_full_gesture_dataset() -> ReadMetaDataset:
    ndrczc_dataset = compose_ndrczc_dataset_for_static_gesture_classification(
        NDRCZC_DATA_ROOT
    )
    custom_train_dataset = load_custom_train_dataset()
    train_dataset = ReadMetaConcatDataset([ndrczc_dataset, custom_train_dataset])
    return train_dataset


def load_gesture_datasets(
    amount_per_gesture_train: int, amount_per_gesture_val: int
) -> Tuple[ReadMetaDataset, ReadMetaDataset]:
    train_dataset_materials: List[ReadMetaDataset] = []
    val_dataset_materials: List[ReadMetaDataset] = []
    full_dataset = load_full_gesture_dataset()
    for gesture in StaticGesture:
        gesture_subset = get_dataset_subset_with_gesture(
            dataset=full_dataset, label=gesture
        )
        train_indexes = list(range(amount_per_gesture_train))
        val_indexes = list(
            range(
                amount_per_gesture_train,
                amount_per_gesture_train + amount_per_gesture_val,
            )
        )
        train_dataset_materials.append(ReadMetaSubset(gesture_subset, train_indexes))
        val_dataset_materials.append(ReadMetaSubset(gesture_subset, val_indexes))

    train_dataset = ReadMetaConcatDataset(train_dataset_materials)
    val_dataset = ReadMetaConcatDataset(val_dataset_materials)
    return train_dataset, val_dataset
