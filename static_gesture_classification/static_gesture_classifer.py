from torchvision.models import resnet18
from neptune.types import File
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from static_gesture_classification.config import StaticGestureConfig
from general.data_structures.model_line import ModelLine
import pandas as pd
from PIL import Image
from typing import Dict, List, Iterable, Tuple, Callable, Optional
from collections import defaultdict
from general.data_structures.data_split import DataSplit
from static_gesture_classification.static_gesture import StaticGesture
from static_gesture_classification.classification_results_dataframe import (
    ClassificationResultsDataframe,
)
from static_gesture_classification.metrics_utils import (
    log_dict_like_structure_to_neptune,
)
from static_gesture_classification.metrics_utils import (
    generate_confusion_matrix_plot_from_classification_results,
)
from static_gesture_classification.config import AugsConfig, TrainHyperparameters
from static_gesture_classification.focall_loss import FocalLoss
import torchvision.transforms as tf


def init_static_gesture_classifier(cfg: StaticGestureConfig) -> nn.Module:
    if cfg.model.architecture == "resnet18":
        model = resnet18(pretrained=cfg.model.use_pretrained)
        model.fc = nn.Linear(model.fc.in_features, len(StaticGesture))
        return model
    else:
        raise NotImplementedError(f"Unknown architecture: {cfg.model.architecture}")


def init_lr_scheduler(optimizer, cfg: StaticGestureConfig) -> Optional[nn.Module]:
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


def init_augmentations_from_config(
    augs_cfg: AugsConfig,
) -> Dict[DataSplit, Callable[[Image.Image], torch.Tensor]]:
    """Inits augmentations for each"""
    resize = tf.Resize(augs_cfg.input_resolution)
    normalization = tf.Compose(
        [
            tf.ToTensor(),
            tf.Normalize(
                mean=augs_cfg.normalization_mean, std=augs_cfg.normalization_std
            ),
        ]
    )
    augmentation_transform = tf.Compose(
        [
            tf.ColorJitter(
                brightness=augs_cfg.brightness_factor,
                contrast=augs_cfg.contrast_factor,
                saturation=augs_cfg.saturation_contrast,
                hue=augs_cfg.hue_contrast,
            ),
            tf.RandomAffine(
                degrees=augs_cfg.rotation_range_angles_degrees,
                translate=augs_cfg.translation_range_imsize_fractions,
                scale=augs_cfg.scaling_range_factors,
                shear=augs_cfg.shear_x_axis_degrees_range,
            ),
        ]
    )
    val_aug = tf.Compose([resize, normalization])
    train_aug = tf.Compose(
        [
            resize,
            tf.RandomApply(
                transforms=[augmentation_transform], p=augs_cfg.augmentation_probability
            ),
            normalization,
        ]
    )
    return {DataSplit.TRAIN: train_aug, DataSplit.VAL: val_aug}


def init_loss_from_config(train_config: TrainHyperparameters) -> Optional[nn.Module]:
    if train_config.loss == "cross-entropy":
        return nn.CrossEntropyLoss()
    elif train_config.loss == "focal":
        return FocalLoss(
            class_num=len(StaticGesture), gamma=train_config.focal_loss_gamma
        )
    else:
        raise NotImplementedError(f"Loss {train_config.loss} is not implemented")


class StaticGestureClassifier(pl.LightningModule):
    def __init__(
        self,
        cfg: StaticGestureConfig,
        results_location: ModelLine,
    ):
        super().__init__()
        self.cfg = cfg
        self.criterion = init_loss_from_config(cfg.train_hyperparams)
        self.model = init_static_gesture_classifier(self.cfg)
        self.predictions_on_datasets: Dict[
            DataSplit, ClassificationResultsDataframe
        ] = defaultdict(init_classification_results_dataframe)
        self.results_location = results_location

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

    def get_gesture_prediction_for_single_input(
        self, single_input: torch.Tensor
    ) -> Tuple[StaticGesture, float]:
        input_is_single_image: bool = single_input.ndim == 3 or (
            single_input.ndim == 4 and single_input.shape[0] == 1
        )
        if not input_is_single_image:
            raise ValueError("Should be used with single image only")
        network_input: torch.Tensor = (
            single_input
            if single_input.ndim == 4
            else torch.unsqueeze(single_input, dim=0)
        )
        network_input = network_input.to(self.device)
        logits = self.model(network_input).cpu()
        softmax = nn.Softmax(dim=1)
        probs = softmax(logits)[0]
        pred_class_index = torch.argmax(probs, dim=0, keepdim=True).item()
        prediction_probability: float = probs[pred_class_index].item()
        predicted_gesture: StaticGesture = StaticGesture(pred_class_index)
        return predicted_gesture, prediction_probability

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
        """"""
        fig, ax = plt.subplots()
        ax = generate_confusion_matrix_plot_from_classification_results(
            classification_results, ax
        )
        self.logger.experiment[log_path].upload(File.as_image(fig))
        plt.close(fig)

    def _log_validation_metrics(
        self, val_predictions: ClassificationResultsDataframe
    ) -> None:
        """"""
        # log confusion matrix
        conf_matrix_neptune_path = (
            f"val/figures/confusion_matrices/{self.current_epoch:04d}"
        )
        self._log_conf_matrix_to_neptune(
            classification_results=val_predictions,
            log_path=conf_matrix_neptune_path,
        )
        # log classification report
        gt = val_predictions.ground_true.tolist()
        pred = val_predictions.prediction.tolist()
        labels = [g.name for g in StaticGesture]
        val_classification_report = classification_report(
            y_true=gt, y_pred=pred, labels=labels, output_dict=True
        )
        log_dict_like_structure_to_neptune(
            dict_like_structure=val_classification_report,
            neptune_root="val/classification_report",
            neptune_run=self.logger.experiment,
            log_as_sequence=True,
        )
        self.log(
            "val_weighted_F1", val_classification_report["weighted avg"]["f1-score"]
        )

    def on_validation_epoch_end(self) -> None:
        # log metrics to neptune
        self._log_validation_metrics(
            val_predictions=self.predictions_on_datasets[DataSplit.VAL]
        )
        # save predictions locally
        save_path = os.path.join(
            self.results_location.val_predictions_root, f"{self.current_epoch:04d}.csv"
        )
        self.predictions_on_datasets[DataSplit.VAL].to_csv(save_path)
        # refresh predictions
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
