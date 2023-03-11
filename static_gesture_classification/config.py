from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class AugsConfig:
    normalization_mean: List[float]
    normalization_std: List[float]
    input_resolution: Tuple[int, int]
    rotation_range_angles_degrees: Tuple[float, float]
    translation_range_imsize_fractions: Tuple[float, float]
    scaling_range_factors: Tuple[float, float]


@dataclass
class TrainHyperparameters:
    num_classes: int
    classes_names: List[str]
    device: str
    learinig_rate: float
    momentun: float
    scheduler_type: str
    # optimizer_type: str  # TODO add later
    patience_epochs_num: int
    lr_reduction_factor: float


@dataclass
class ModelConfig:
    architecture: str
    use_pretrained: bool


@dataclass
class StaticGestureConfig:
    augs: AugsConfig
    train_hyperparams: TrainHyperparameters
    model: ModelConfig
