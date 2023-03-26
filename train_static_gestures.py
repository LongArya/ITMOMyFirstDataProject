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


@hydra.main(
    config_path=STATIC_GESTURE_CFG_ROOT,
    config_name=STATIC_GESTURE_CFG_NAME,
    version_base=None,
)
def test_gestures_pipeline(cfg: StaticGestureConfig):

    dummy_model_line_path = os.path.join(TRAIN_RESULTS_ROOT, "dummy_line")
    dummy_ml = ModelLine(dummy_model_line_path)
    lightning_classifier = StaticGestureClassifier(cfg, results_location=dummy_ml)
    lightning_classifier.model.eval()
    # lightning_classifier.training_step(batch, [])
    # print(batch["label"])
    # print(lightning_classifier.predictions_on_datasets[DataSplit.TRAIN])
