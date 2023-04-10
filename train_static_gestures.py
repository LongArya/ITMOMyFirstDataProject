import os
from torchmetrics.classification import BinaryPrecisionRecallCurve
from static_gesture_classification.metrics_utils import (
    compute_pr_curve_for_gesture,
    get_pr_curves_for_gestures,
    get_pr_curve_plot,
    generate_confusion_matrix_plot_from_classification_results,
)
from neptune.types import File
from matplotlib.axes._axes import Axes
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, precision_recall_curve
import torch.nn as nn
import numpy as np
import pandas as pd
import hydra
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Callable
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, NeptuneLogger
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
)
from static_gesture_classification.static_gesture import StaticGesture
from static_gesture_classification.config import StaticGestureConfig
from const import (
    STATIC_GESTURE_CFG_NAME,
    STATIC_GESTURE_CFG_ROOT,
    DATA_ROOT,
    TRAIN_RESULTS_ROOT,
)
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, default_collate
from general.data_structures.model_line import ModelLine
from general.data_structures.data_split import DataSplit
import torchvision.transforms as tf
from static_gesture_classification.data_loading.load_datasets import load_train_dataset
from general.datasets.read_meta_dataset import (
    ReadMetaSubset,
    ReadMetaDataset,
    ReadMetaConcatDataset,
)
from static_gesture_classification.classification_results_dataframe import (
    ClassificationResultsDataframe,
)

os.environ["HYDRA_FULL_ERROR"] = "1"

cs = ConfigStore.instance()
cs.store(name=STATIC_GESTURE_CFG_NAME, node=StaticGestureConfig)


TEST_TRAIN_TRANSFORM = tf.Compose(
    [
        tf.ToTensor(),
        tf.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # FIXME fill with relevant values
        tf.Resize((224, 224)),
    ]
)


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated_batch: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = default_collate([sample[key] for sample in batch])
    return collated_batch


class TransformApplier(ReadMetaDataset):
    def __init__(
        self, dataset: ReadMetaDataset, transformation: Callable[[Any], Any]
    ) -> None:
        self.dataset = dataset
        self.transformation = transformation

    def __len__(self) -> int:
        return len(self.dataset)

    def read_meta(self, index: int) -> Any:
        return self.dataset.read_meta()

    def __getitem__(self, index) -> Any:
        sample = deepcopy(self.dataset[index])
        sample["image"] = self.transformation(sample["image"])
        return sample


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


def get_mini_train_dataset() -> ReadMetaDataset:
    dataset = load_train_dataset()
    unique_labels = get_dataset_unique_labels(dataset)
    dataset_concat_target = []
    for label in unique_labels:
        label_subset = get_dataset_subset_with_label(dataset=dataset, label=label)
        dataset_concat_target.append(ReadMetaSubset(label_subset, [0, 1]))
    concat_dataset = ReadMetaConcatDataset(dataset_concat_target)
    result_dataset = TransformApplier(
        dataset=concat_dataset, transformation=TEST_TRAIN_TRANSFORM
    )
    return result_dataset


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def test_gestures_pipeline(cfg: StaticGestureConfig):
    dummy_model_line_path = os.path.join(TRAIN_RESULTS_ROOT, "dummy_line1")
    dummy_ml = ModelLine(dummy_model_line_path)
    lightning_classifier = StaticGestureClassifier(cfg, results_location=dummy_ml)
    lightning_classifier.model.eval()
    csv_logger = CSVLogger(save_dir=dummy_ml.root)
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmEzZDc3Mi1kNDg4LTQ2MjgtOGU4MS1jZDlhZDM2OTkyM2MifQ==",
        project="longarya/StaticGestureClassification",
        tags=["training", "resnet18"],
        log_model_checkpoints=False,
    )

    dataset = get_mini_train_dataset()
    trainer = pl.Trainer(logger=[neptune_logger], log_every_n_steps=1)
    train_dataloader = DataLoader(
        dataset=dataset, batch_size=16, num_workers=0, collate_fn=custom_collate
    )
    val_dataloader = DataLoader(
        dataset=dataset, batch_size=16, num_workers=0, collate_fn=custom_collate
    )
    trainer.fit(
        model=lightning_classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


def test_logging_of_conf_matrix():
    fig, ax = plt.subplots()
    dataframe = pd.read_csv(
        "E:\\dev\\MyFirstDataProject\\training_results\\dummy_line1\\train_predictions\\0002.csv",
        index_col=0,
    )
    ax = generate_confusion_matrix_plot_from_classification_results(dataframe, ax)

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmEzZDc3Mi1kNDg4LTQ2MjgtOGU4MS1jZDlhZDM2OTkyM2MifQ==",
        project="longarya/StaticGestureClassification",
        tags=["training", "resnet18"],
        log_model_checkpoints=False,
    )

    neptune_logger.experiment["train/misclassified_images"].upload(File.as_image(fig))
    plt.close(fig)


test_gestures_pipeline()
