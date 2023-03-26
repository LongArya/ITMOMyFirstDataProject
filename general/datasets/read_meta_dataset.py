from torch.utils.data import Dataset, ConcatDataset, Subset
import bisect
from typing import Any


class ReadMetaDataset(Dataset):
    """Dataset that requires separate method for meta reading,
    In order to not read image when we need only meta, which should drastically analytic"""

    def read_meta(self, index: int) -> Any:
        raise NotImplementedError


class ReadMetaConcatDataset(ConcatDataset, ReadMetaDataset):
    """Concat dataset extended with read meta functionality"""

    def read_meta(self, idx: int) -> Any:
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx].read_meta(sample_idx)


class ReadMetaSubset(Subset, ReadMetaDataset):
    """Subset extended with read meta funcionaluty"""

    def read_meta(self, idx: int):
        return self.dataset.read_meta(self.indices[idx])
