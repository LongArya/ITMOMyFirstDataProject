import os
from itertools import product
import random
from sklearn.metrics import classification_report
from omegaconf import DictConfig
import cv2
from PIL import Image
from matplotlib.axes._axes import Axes
import json
from torchmetrics.classification import BinaryPrecisionRecallCurve
from neptune.types import File
from general.utils import plot_images_in_grid, get_sample_with_image_path
from static_gesture_classification.metrics_utils import (
    log_dict_like_structure_to_neptune,
)
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from neptune.types import File
from matplotlib.axes._axes import Axes
import torch
import numpy as np
import pandas as pd
import hydra
from copy import deepcopy
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Callable, Iterable, Mapping, Union, Optional
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
    init_augmentations_from_config,
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
from torch.utils.data import Dataset, DataLoader, default_collate
from general.data_structures.model_line import ModelLine
from general.data_structures.data_split import DataSplit
import torchvision.transforms as tf
from static_gesture_classification.data_loading.load_datasets import (
    get_dataset_subset_with_gesture,
    get_dataset_unique_labels,
    load_train_dataset,
    load_val_dataset,
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
from static_gesture_classification.classification_results_views import (
    get_gt_pred_combination_view,
    get_fails_view,
)
import matplotlib

matplotlib.use("Agg")


os.environ["HYDRA_FULL_ERROR"] = "1"

cs = ConfigStore.instance()
cs.store(name=STATIC_GESTURE_CFG_NAME, node=StaticGestureConfig)


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


def generate_failure_cases_images(
    root: str,
    dataframe_predictions: ClassificationResultsDataframe,
    dataset: ReadMetaSubset,
) -> None:
    grid_size = (5, 5)
    grid_images_num = grid_size[0] * grid_size[1]
    for gt, pred in product(list(StaticGesture), list(StaticGesture), repeat=1):
        if gt == pred:
            continue
        fails_view = get_gt_pred_combination_view(dataframe_predictions, gt, pred)
        failed_images_paths: List[str] = fails_view.image_path.tolist()
        samples_with_failed_images: List[Dict] = [
            get_sample_with_image_path(dataset=dataset, image_path=image_path)
            for image_path in failed_images_paths
        ]
        failed_images = [sample["image"] for sample in samples_with_failed_images]
        if len(samples_with_failed_images) == 0:
            continue

        title = f"{gt.name}_as_{pred.name}"
        save_dir = os.path.join(root, title)
        os.makedirs(save_dir, exist_ok=True)

        for plot_num, index in enumerate(range(0, len(failed_images), grid_images_num)):
            grid_images = failed_images[index : index + grid_images_num]
            fig, axes = plt.subplots(*grid_size)
            plot_images_in_grid(axes=axes, images=grid_images)
            fig.suptitle(title)
            plt.savefig(fname=os.path.join(save_dir, f"{title}_{plot_num}.png"))
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
        tags=[cfg.model.architecture],
        log_model_checkpoints=False,
    )
    run_id: str = neptune_logger.experiment["sys/id"].fetch()
    model_line_path: str = os.path.join(TRAIN_RESULTS_ROOT, run_id)
    model_line: ModelLine = ModelLine(model_line_path)
    lightning_classifier = StaticGestureClassifier(cfg, results_location=model_line)
    log_dict_like_structure_to_neptune(
        dict_like_structure=cfg,
        neptune_root="conf",
        neptune_run=neptune_logger.experiment,
        log_as_sequence=False,
    )
    train_dataset = load_train_dataset()
    val_dataset = load_val_dataset()
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
    lr_monitor_callback = LearningRateMonitor(
        logging_interval="step", log_momentum=True
    )
    model_ckpt_callback = ModelCheckpoint(
        monitor="val_weighted_F1",
        dirpath=model_line.checkpoints_root,
        mode="max",
        auto_insert_metric_name=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
        filename="checkpoint_{epoch:02d}-{val_weighted_F1:.2f}",
        save_top_k=3,
    )
    trainer = pl.Trainer(
        logger=neptune_logger,
        log_every_n_steps=1,
        max_epochs=100,
        callbacks=[dummy_inference_callback, model_ckpt_callback, lr_monitor_callback],
        num_sanity_val_steps=0,
        gpus=[0],
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=custom_collate,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=16,
        num_workers=0,
        collate_fn=custom_collate,
        shuffle=False,
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
        "E:\\dev\\MyFirstDataProject\\training_results\\STAT-81\\checkpoints\\checkpoint_epoch=06-val_weighted_F1=0.87.ckpt",
        cfg=cfg,
        results_location=None,
    )
    model.eval()
    model.to("cuda")
    val_transform = init_augmentations_from_config(cfg.augs)[DataSplit.VAL]
    full_ds = load_full_gesture_dataset()
    full_ds = TransformApplier(dataset=full_ds, transformation=val_transform)
    im_path = (
        "e:\dev\MyFirstDataProject\Data\\ndrczc35bt-1\Subject3\Subject3\\523_color.png"
    )
    sample = get_sample_with_image_path(dataset=full_ds, image_path=im_path)
    gesture, prob = model.get_gesture_prediction_for_single_input(sample["image"])
    print(gesture, prob)
    plt.imshow(
        neutralize_image_normalization(
            sample["image"],
            mean=cfg.augs.normalization_mean,
            std=cfg.augs.normalization_std,
        )
    )
    plt.show()


# @hydra.main(
#     config_path=STATIC_GESTURE_CFG_ROOT,
#     config_name=STATIC_GESTURE_CFG_NAME,
#     version_base=None,
# )
# def failure_cases_generation(cfg: StaticGestureClassifier):
#     dataframe_predictions = pd.read_csv(
#         "E:\\dev\\MyFirstDataProject\\training_results\\STAT-81\\train_predictions\\0006.csv",
#         index_col=0,
#     )

#     dataset = load_full_gesture_dataset()
#     generate_failure_cases_images(
#         root="E:\\dev\\MyFirstDataProject\\training_results\\STAT-81\\FC_train",
#         dataframe_predictions=dataframe_predictions,
#         dataset=dataset,
#     )


if __name__ == "__main__":
    train_static_gesture()
