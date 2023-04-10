from torchvision.models import resnet18
from neptune.types import File
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from static_gesture_classification.config import StaticGestureConfig
from general.data_structures.model_line import ModelLine
import pandas as pd
from typing import Dict, List
from collections import defaultdict
from general.data_structures.data_split import DataSplit
from static_gesture_classification.static_gesture import StaticGesture
from static_gesture_classification.classification_results_dataframe import (
    ClassificationResultsDataframe,
)
from static_gesture_classification.metrics_utils import (
    compute_pr_curve_for_gesture,
    get_pr_curves_for_gestures,
    get_pr_curve_plot,
    generate_confusion_matrix_plot_from_classification_results,
)


def init_static_gesture_classifier(cfg: StaticGestureConfig) -> nn.Module:
    if cfg.model.architecture == "resnet18":
        model = resnet18(pretrained=cfg.model.use_pretrained)
        model.fc = nn.Linear(model.fc.in_features, len(StaticGesture))
        return model
    else:
        raise NotImplementedError(f"Unknown architecture: {cfg.model.architecture}")


def init_lr_scheduler(optimizer, cfg: StaticGestureConfig):
    if cfg.train_hyperparams.scheduler_type == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=cfg.train_hyperparams.lr_reduction_factor,
            patience=cfg.train_hyperparams.patience_epochs_num,
        )
        return scheduler
    else:
        raise NotImplementedError(
            f"Unknown scheduler type: {cfg.train_hyperparams.scheduler_type}"
        )


def init_classification_results_dataframe() -> ClassificationResultsDataframe:
    return pd.DataFrame(
        columns=["image_path", "ground_true", "prediction", "prediction_score"]
        + [gesture.name for gesture in StaticGesture]
    )


class StaticGestureClassifier(pl.LightningModule):
    def __init__(
        self,
        cfg: StaticGestureConfig,
        results_location: ModelLine,
    ):
        super().__init__()
        self.cfg = cfg
        # FIXME use focal loss? / specify loss in cfg
        self.criterion = nn.CrossEntropyLoss()
        self.model = init_static_gesture_classifier(self.cfg)
        self.predictions_on_datasets: Dict[
            DataSplit, ClassificationResultsDataframe
        ] = defaultdict(init_classification_results_dataframe)
        self.results_location = results_location

    def _append_predictions_to_split(
        self,
        gt_classes: torch.Tensor,
        logits: torch.Tensor,
        images_paths: List[str],
        split: DataSplit,
    ):
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)
        pred_classes = torch.argmax(probs, dim=1, keepdim=True)
        predictions_scores = torch.take_along_dim(probs, pred_classes, dim=1)

        batch_predictions = []
        for path, gt_class, pred_class, pred_score, single_image_probabilities in zip(
            images_paths, gt_classes, pred_classes, predictions_scores, probs
        ):
            gt_label = StaticGesture(gt_class.item()).name
            pred_label = StaticGesture(pred_class.item()).name
            # TODO append scores of all classes
            batch_predictions.append(
                [path, gt_label, pred_label, pred_score.item()]
                + [prob.item() for prob in single_image_probabilities]
            )
        batch_prediction_dataframe: ClassificationResultsDataframe = pd.DataFrame(
            batch_predictions,
            columns=self.predictions_on_datasets[split].columns,
        )
        self.predictions_on_datasets[split] = pd.concat(
            [self.predictions_on_datasets[split], batch_prediction_dataframe],
            ignore_index=True,
        )

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        gt_labels = batch["label"]
        images_paths = batch["image_path"]
        pred_labels = self.model(inputs)
        self._append_predictions_to_split(
            gt_classes=gt_labels,
            logits=pred_labels,
            images_paths=images_paths,
            split=DataSplit.TRAIN,
        )

        loss = self.criterion(pred_labels, gt_labels)
        # specify batch size, since in might be incorrectly inferred from dict by lightning
        batch_size = batch["image"].size(dim=0)
        self.log("train_loss", loss, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        gt_labels = batch["label"]
        images_paths = batch["image_path"]
        pred_labels = self.model(inputs)
        # save gt, preds to table
        self._append_predictions_to_split(
            gt_classes=gt_labels,
            logits=pred_labels,
            images_paths=images_paths,
            split=DataSplit.VAL,
        )

    def on_train_epoch_end(self) -> None:
        # save results
        save_path = os.path.join(
            self.results_location.train_predictions_root,
            f"{self.current_epoch:04d}.csv",
        )
        self.predictions_on_datasets[DataSplit.TRAIN].to_csv(save_path)
        # refresh results
        self.predictions_on_datasets[
            DataSplit.TRAIN
        ] = init_classification_results_dataframe()
        return super().on_train_epoch_end()

    def _log_conf_matrix_to_neptune(
        self,
        classification_results: ClassificationResultsDataframe,
        log_path: str,
    ) -> None:
        fig, ax = plt.subplots()
        ax = generate_confusion_matrix_plot_from_classification_results(
            classification_results, ax
        )
        self.logger.experiment[log_path].upload(File.as_image(fig))
        plt.close(fig)

    def on_validation_epoch_end(self) -> None:
        save_path = os.path.join(
            self.results_location.val_predictions_root, f"{self.current_epoch:04d}.csv"
        )
        conf_matrix_neptune_path = (
            f"val/figures/confusion_matrices/{self.current_epoch:04d}"
        )
        self._log_conf_matrix_to_neptune(
            classification_results=self.predictions_on_datasets[DataSplit.VAL],
            log_path=conf_matrix_neptune_path,
        )
        self.predictions_on_datasets[DataSplit.VAL].to_csv(save_path)

        # refresh results
        self.predictions_on_datasets[
            DataSplit.VAL
        ] = init_classification_results_dataframe()

        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.train_hyperparams.learinig_rate,
            momentum=self.cfg.train_hyperparams.momentun,
        )
        scheduler = init_lr_scheduler(optimizer=optimizer, cfg=self.cfg)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "train_loss",
        }


# failure cases
