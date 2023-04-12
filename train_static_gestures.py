import os
from matplotlib.axes._axes import Axes
import json
from torchmetrics.classification import BinaryPrecisionRecallCurve
from neptune.types import File
from static_gesture_classification.metrics_utils import (
    compute_pr_curve_for_gesture,
    get_pr_curves_for_gestures,
    get_pr_curve_plot,
    generate_confusion_matrix_plot_from_classification_results,
    get_combined_pr_curves_plot,
    compute_AP_for_gesture,
    get_f1_curve_plot,
    get_f1_curve_values_from_pr_curve,
)
from pytorch_lightning.callbacks import ModelCheckpoint
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
from static_gesture_classification.data_loading.load_datasets import (
    load_train_dataset,
    get_dataset_subset_with_gesture,
)
from general.datasets.read_meta_dataset import (
    ReadMetaSubset,
    ReadMetaDataset,
    ReadMetaConcatDataset,
)
from static_gesture_classification.classification_results_dataframe import (
    ClassificationResultsDataframe,
)
from pytorch_lightning.callbacks import Callback
from neptune.new.run import Run

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
        return self.dataset.read_meta(index)

    def __getitem__(self, index) -> Any:
        sample = deepcopy(self.dataset[index])
        sample["image"] = self.transformation(sample["image"])
        return sample


class DummyInferenceCallback(Callback):
    """Callback that runs model on dummy input in order to assert reproducibility later"""

    def __init__(self, dummy_input: torch.Tensor, save_root: str):
        super().__init__()
        self.dummy_input = dummy_input
        self.save_root = save_root

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        logits = pl_module.model(self.dummy_input)
        json_serializible_logits = logits.tolist()
        output_path: str = os.path.join(
            self.save_root, f"{pl_module.current_epoch:04d}.json"
        )
        with open(output_path, "w") as f:
            json.dump({"dummy_input_logits": json_serializible_logits}, f, indent=2)
        # save logits to neptune
        return super().on_validation_epoch_end(trainer, pl_module)


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


# TODO log stats about data and thresholds
# TODO init augmentations wrt config params
# TODO train first version
def generate_gesture_distribution_plot(
    plot_axis: Axes, gesture_distribution: Dict[StaticGesture, int]
):
    # Fixing random state for reproducibility
    gestures_labels: List[str] = [g.name for g in StaticGesture]
    y_pos = np.arange(len(StaticGesture))
    gestures_amounts_in_dataset = [
        gesture_distribution.get(gesture, 0) for gesture in StaticGesture
    ]
    plot_axis.barh(y_pos, gestures_amounts_in_dataset, align="center")
    plot_axis.set_yticks(y_pos, labels=gestures_labels)
    plot_axis.set_xlabel("N")
    return plot_axis


def log_dataset_distribution_to_neptune(
    neptune_run: Run,
    log_path: str,
    dataset: ReadMetaDataset,
):
    # collect gesture distribution
    gesture_distribution: Dict[StaticGesture, int] = {}
    for gesture in StaticGesture:
        gesture_subset_dataset = get_dataset_subset_with_gesture(
            dataset=dataset, label=gesture
        )
        gesture_distribution[gesture] = len(gesture_subset_dataset)
    fig, ax = plt.subplots()
    ax = generate_gesture_distribution_plot(
        plot_axis=ax, gesture_distribution=gesture_distribution
    )
    ax.set_title(f"Gesture distribution of {len(dataset)} samples")
    neptune_run[log_path].upload(File.as_image(fig))
    plt.close(fig)


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def train_static_gesture(cfg: StaticGestureConfig):
    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmEzZDc3Mi1kNDg4LTQ2MjgtOGU4MS1jZDlhZDM2OTkyM2MifQ==",
        project="longarya/StaticGestureClassification",
        tags=["training", "resnet18"],
        log_model_checkpoints=True,
    )
    run_id: str = neptune_logger.experiment["sys/id"].fetch()
    model_line_path: str = os.path.join(TRAIN_RESULTS_ROOT, run_id)
    model_line: ModelLine = ModelLine(model_line_path)
    lightning_classifier = StaticGestureClassifier(cfg, results_location=model_line)

    dataset = get_mini_train_dataset()
    log_dataset_distribution_to_neptune(
        neptune_run=neptune_logger.experiment,
        dataset=dataset,
        # log_path=os.path.join("datasets", "train"),
        log_path="datasets/train",
    )

    dummy_output_directory = os.path.join(model_line.root, "dummy_output")
    os.makedirs(dummy_output_directory, exist_ok=True)

    dummy_inference_callback = DummyInferenceCallback(
        dummy_input=torch.zeros(1, 3, 224, 224), save_root=dummy_output_directory
    )
    model_ckpt_callback = ModelCheckpoint(
        monitor="mAP",
        dirpath=model_line.checkpoints_root,
        mode="max",
        auto_insert_metric_name=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        filename="checkpoint_{epoch:02d}-{mAP:.2f}",
    )
    trainer = pl.Trainer(
        logger=[neptune_logger],
        log_every_n_steps=1,
        max_epochs=3,
        callbacks=[dummy_inference_callback, model_ckpt_callback],
    )
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
    dataframe = pd.read_csv(
        "E:\\dev\\MyFirstDataProject\\training_results\\dummy_line1\\train_predictions\\0002.csv",
        index_col=0,
    )
    # ax = generate_confusion_matrix_plot_from_classification_results(dataframe, ax)
    pr_curve = compute_pr_curve_for_gesture(dataframe, StaticGesture.OKEY)
    f1_curve_values = get_f1_curve_values_from_pr_curve(pr_curve=pr_curve)

    fig, ax = plt.subplots()
    ax = get_f1_curve_plot(
        plot_axis=ax,
        f1_score_values=f1_curve_values,
        thresholds=pr_curve.thresholds,
    )
    plt.show()
    return

    neptune_logger = NeptuneLogger(
        api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNmEzZDc3Mi1kNDg4LTQ2MjgtOGU4MS1jZDlhZDM2OTkyM2MifQ==",
        project="longarya/StaticGestureClassification",
        tags=["training", "resnet18"],
        log_model_checkpoints=False,
    )
    mAP = 1
    neptune_logger.experiment["val/mAP"].append(mAP)
    # plt.close(fig)


# train_static_gesture()
print(os.path.join("a", "b"))
