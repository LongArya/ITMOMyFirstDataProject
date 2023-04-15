import os
from neptune.utils import stringify_unsupported
import random
from omegaconf import DictConfig
import cv2
from PIL import Image
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
import torch
import numpy as np
import pandas as pd
import hydra
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Callable, Iterable
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
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
    IMAGENET_MEAN,
    IMAGENET_STD,
    RESNET18_INPUT_SIZE,
)
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, default_collate
from general.data_structures.model_line import ModelLine
from general.data_structures.data_split import DataSplit
import torchvision.transforms as tf
from static_gesture_classification.data_loading.load_datasets import (
    load_full_gesture_dataset,
    get_dataset_subset_with_gesture,
    get_dataset_unique_labels,
    load_gesture_datasets,
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
from static_gesture_classification.config import AugsConfig
from general.data_structures.data_split import DataSplit
from general.utils import TorchNormalizeInverse
from static_gesture_classification.data_loading.transform_applier import (
    TransformApplier,
)
from hydra.core.hydra_config import HydraConfig
import matplotlib

matplotlib.use("Agg")


os.environ["HYDRA_FULL_ERROR"] = "1"

cs = ConfigStore.instance()
cs.store(name=STATIC_GESTURE_CFG_NAME, node=StaticGestureConfig)


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


def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    collated_batch: Dict[str, Any] = {}
    keys = batch[0].keys()
    for key in keys:
        collated_batch[key] = default_collate([sample[key] for sample in batch])
    return collated_batch


def neutralize_image_normalization(
    image_tensor: torch.Tensor, mean: Iterable[float], std: Iterable[float]
) -> np.ndarray:
    inverse_normalization = TorchNormalizeInverse(mean=mean, std=std)
    denormalized_tensor = inverse_normalization(image_tensor)
    denormalized_tensor *= 255
    denormalized_tensor = torch.permute(denormalized_tensor, (1, 2, 0))
    image: np.ndarray = denormalized_tensor.numpy()
    image = image.astype(np.uint8)
    return image


class DummyInferenceCallback(Callback):
    """Callback that runs model on dummy input in order to assert reproducibility later"""

    def __init__(self, dummy_input: torch.Tensor, save_root: str):
        super().__init__()
        self.dummy_input = dummy_input
        self.save_root = save_root

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        logits = pl_module.model(self.dummy_input.to(pl_module.device))
        json_serializible_logits = logits.cpu().tolist()
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


def get_mini_train_dataset() -> ReadMetaDataset:
    dataset = load_full_gesture_dataset()
    dataset_concat_target = []
    for gesture in StaticGesture:
        label_subset = get_dataset_subset_with_gesture(dataset=dataset, label=gesture)
        dataset_concat_target.append(ReadMetaSubset(label_subset, [0, 1]))
    concat_dataset = ReadMetaConcatDataset(dataset_concat_target)
    return concat_dataset


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


def log_dataset_samples_to_neptune(
    neptune_run: Run,
    neptune_root: str,
    samples_num: int,
    dataset: ReadMetaDataset,
    augs_config: AugsConfig,
):
    """Logs visualizations of N random samples to neptune run"""
    log_indexes: List[int] = random.sample(
        population=list(range(len(dataset))), k=samples_num
    )
    for logged_example_index, index in enumerate(log_indexes):
        sample = dataset[index]
        fig, ax = plt.subplots()
        image = sample["image"]
        image = neutralize_image_normalization(
            image,
            mean=augs_config.normalization_mean,
            std=augs_config.normalization_std,
        )
        gesture = StaticGesture(sample["label"])
        ax.imshow(image)
        ax.set_title(gesture.name)
        neptune_run[f"{neptune_root}/sample_{logged_example_index}"].upload(
            File.as_image(fig)
        )
        plt.close(fig)


def log_info_about_datasets_to_neptune(
    train_dataset: ReadMetaDataset,
    val_dataset: ReadMetaDataset,
    neptune_run: Run,
    logged_samples_amount: int,
    augs_config: AugsConfig,
):
    log_dataset_distribution_to_neptune(
        neptune_run=neptune_run,
        dataset=train_dataset,
        log_path="datasets/train_distribution",
    )
    log_dataset_distribution_to_neptune(
        neptune_run=neptune_run,
        dataset=val_dataset,
        log_path="datasets/val_distribution",
    )
    log_dataset_samples_to_neptune(
        neptune_run=neptune_run,
        neptune_root="datasets/train_samples",
        samples_num=logged_samples_amount,
        dataset=train_dataset,
        augs_config=augs_config,
    )
    log_dataset_samples_to_neptune(
        neptune_run=neptune_run,
        neptune_root="datasets/val_samples",
        samples_num=logged_samples_amount,
        dataset=val_dataset,
        augs_config=augs_config,
    )


def log_dict_config_to_neptune(cfg: DictConfig, prefix: str, neptune_run: Run):
    if not isinstance(cfg, DictConfig):
        neptune_run[prefix] = stringify_unsupported(cfg)
        return
    for k, v in cfg.items():
        extended_prefix = prefix + "/" + k if prefix else k
        log_dict_config_to_neptune(v, extended_prefix, neptune_run)


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
        log_model_checkpoints=False,
    )
    run_id: str = neptune_logger.experiment["sys/id"].fetch()
    model_line_path: str = os.path.join(TRAIN_RESULTS_ROOT, run_id)
    model_line: ModelLine = ModelLine(model_line_path)
    lightning_classifier = StaticGestureClassifier(cfg, results_location=model_line)
    log_dict_config_to_neptune(
        cfg=cfg, prefix="conf", neptune_run=neptune_logger.experiment
    )
    train_dataset, val_dataset = load_gesture_datasets(
        amount_per_gesture_train=200, amount_per_gesture_val=70
    )
    augmentations = init_augmentations_from_config(augs_cfg=cfg.augs)

    train_dataset = TransformApplier(
        dataset=train_dataset, transformation=augmentations[DataSplit.TRAIN]
    )
    val_dataset = TransformApplier(
        dataset=val_dataset, transformation=augmentations[DataSplit.VAL]
    )
    log_info_about_datasets_to_neptune(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        neptune_run=neptune_logger.experiment,
        logged_samples_amount=10,
        augs_config=cfg.augs,
    )

    dummy_output_directory = os.path.join(model_line.root, "dummy_output")
    os.makedirs(dummy_output_directory, exist_ok=True)

    dummy_inference_callback = DummyInferenceCallback(
        dummy_input=torch.zeros(1, 3, *cfg.augs.input_resolution),
        save_root=dummy_output_directory,
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
        logger=neptune_logger,
        log_every_n_steps=1,
        max_epochs=100,
        callbacks=[dummy_inference_callback, model_ckpt_callback],
        num_sanity_val_steps=0,
        gpus=[0],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=16, num_workers=0, collate_fn=custom_collate
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=16, num_workers=0, collate_fn=custom_collate
    )
    trainer.fit(
        model=lightning_classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def test_output(cfg: StaticGestureConfig):
    model = StaticGestureClassifier.load_from_checkpoint(
        "E:\\dev\\MyFirstDataProject\\training_results\\STAT-54\\checkpoints\\checkpoint_epoch=00-mAP=0.87.ckpt",
        cfg=cfg,
        results_location=None,
    )
    x = torch.zeros((1, 3, 224, 224))
    model.eval()
    y = model(x)
    print(y)


train_static_gesture()
