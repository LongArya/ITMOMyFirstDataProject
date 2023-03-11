import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import hydra
import matplotlib.pyplot as plt
from typing import Dict, Any
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import pytorch_lightning as pl
from static_gesture_classification.static_gesture_classifer import (
    StaticGestureClassifier,
)
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

os.environ["HYDRA_FULL_ERROR"] = "1"

cs = ConfigStore.instance()
cs.store(name=STATIC_GESTURE_CFG_NAME, node=StaticGestureConfig)


class HWC2CHW_Transpose:
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        print(f"{image.shape} before transpose")
        return torch.permute(image, (2, 0, 1))


transform = tf.Compose(
    [
        tf.ToTensor(),
        tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        tf.Resize((224, 224)),
    ]
)


class MNISTAdapter(Dataset):
    @staticmethod
    def collate_sample(samples):
        collated_images = default_collate([sample["image"] for sample in samples])
        collated_labels = default_collate([sample["label"] for sample in samples])
        collated_images_paths = [sample["image_path"] for sample in samples]
        collated_sample = {
            "image": collated_images,
            "label": collated_labels,
            "image_path": collated_images_paths,
        }
        return collated_sample

    def __init__(self, mnist_root: str, train: bool):
        self.dataset = MNIST(root=mnist_root, train=train, download=True)

    def __len__(self) -> str:
        return len(self.dataset)

    # TODO write collate fn
    def __getitem__(self, index: int) -> Dict[str, Any]:
        PIL_image, label = self.dataset[index]
        PIL_image = PIL_image.convert("RGB")
        np_image = np.array(PIL_image)
        tnz_image = transform(np_image)
        sample = {
            "image": tnz_image,
            "label": label,
            "image_path": f"mnist_{index:05d}.png",
        }
        return sample


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def mnist_example(cfg: StaticGestureConfig):
    val_mnist = MNISTAdapter(mnist_root=DATA_ROOT, train=False)
    train_mnist = MNISTAdapter(mnist_root=DATA_ROOT, train=True)
    train_loader = DataLoader(
        dataset=train_mnist,
        batch_size=8,
        num_workers=4,
        collate_fn=MNISTAdapter.collate_sample,
    )
    val_loader = DataLoader(
        dataset=val_mnist,
        batch_size=8,
        num_workers=4,
        collate_fn=MNISTAdapter.collate_sample,
    )
    dummy_model_line_path = os.path.join(TRAIN_RESULTS_ROOT, "dummy_line")
    dummy_ml = ModelLine(dummy_model_line_path)
    # TODO that results can be repeated
    lightning_classifier = StaticGestureClassifier(cfg, results_location=dummy_ml)
    trainer = pl.Trainer(gpus=[0], log_every_n_steps=1)
    trainer.fit(
        model=lightning_classifier,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def test_pipeline(cfg: StaticGestureConfig):
    train_mnist = MNISTAdapter(mnist_root=DATA_ROOT, train=True)
    train_loader = DataLoader(
        dataset=train_mnist,
        batch_size=8,
        collate_fn=MNISTAdapter.collate_sample,
        num_workers=4,
    )
    dummy_model_line_path = os.path.join(TRAIN_RESULTS_ROOT, "dummy_line")
    dummy_ml = ModelLine(dummy_model_line_path)
    lightning_classifier = StaticGestureClassifier(cfg, results_location=dummy_ml)
    lightning_classifier.model.eval()
    batch = next(iter(train_loader))
    # lightning_classifier.training_step(batch, [])
    # print(batch["label"])
    # print(lightning_classifier.predictions_on_datasets[DataSplit.TRAIN])
    lightning_classifier.validation_step(batch, [])
    print(batch["label"])
    print(lightning_classifier.predictions_on_datasets[DataSplit.VAL])


mnist_example()
# test_pipeline()
