from torchvision.models import resnet18
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


# TODO move to other place
@dataclass
class ClassificationResultsDataframe(pd.DataFrame):
    @property
    def image_path(self) -> str:
        ...

    @property
    def ground_true(self) -> str:
        ...

    @property
    def prediction(self) -> str:
        ...

    @property
    def prediction_score(self) -> float:
        ...


def init_model(cfg: StaticGestureConfig) -> nn.Module:
    if cfg.model.architecture == "resnet18":
        model = resnet18(pretrained=cfg.model.use_pretrained)
        model.fc = nn.Linear(model.fc.in_features, cfg.train_hyperparams.num_classes)
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


# TODO test this on mnist
# TODO add metrics calculations
# Metrics calculations -> requires train and val predictions?
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
        self.model = init_model(self.cfg)
        self.predictions_on_datasets: Dict[
            DataSplit, ClassificationResultsDataframe
        ] = defaultdict(
            lambda: pd.DataFrame(
                columns=["image_path", "ground_true", "prediction", "prediction_score"]
            )
        )
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
        for path, gt_class, pred_class, pred_score in zip(
            images_paths, gt_classes, pred_classes, predictions_scores
        ):
            gt_label = self.cfg.train_hyperparams.classes_names[gt_class.item()]
            pred_label = self.cfg.train_hyperparams.classes_names[pred_class.item()]
            batch_predictions.append([path, gt_label, pred_label, pred_score.item()])
        batch_prediction_dataframe: ClassificationResultsDataframe = pd.DataFrame(
            batch_predictions,
            columns=["image_path", "ground_true", "prediction", "prediction_score"],
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
        self.predictions_on_datasets[DataSplit.TRAIN] = pd.DataFrame(
            columns=["ground_true", "prediction", "image_path"]
        )
        return super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        # save results
        save_path = os.path.join(
            self.results_location.val_predictions_root, f"{self.current_epoch:04d}.csv"
        )
        self.predictions_on_datasets[DataSplit.VAL].to_csv(save_path)
        # refresh results
        self.predictions_on_datasets[DataSplit.VAL] = pd.DataFrame(
            columns=["ground_true", "prediction", "image_path"]
        )
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.cfg.train_hyperparams.learinig_rate,
            momentum=self.cfg.train_hyperparams.momentun,
        )
        scheduler = init_lr_scheduler(optimizer=optimizer, cfg=self.cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}
