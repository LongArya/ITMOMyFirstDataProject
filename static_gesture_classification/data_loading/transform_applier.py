from PIL import Image
import torch
from typing import Any
from copy import deepcopy
from typing import Callable
from general.datasets.read_meta_dataset import ReadMetaDataset


class TransformApplier(ReadMetaDataset):
    def __init__(
        self,
        dataset: ReadMetaDataset,
        transformation: Callable[[Image.Image], torch.Tensor],
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
