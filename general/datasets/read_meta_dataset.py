from torch.utils.data import Dataset
from typing import Any


class ReadMetaDataset(Dataset):
    """Dataset that requires separate method for meta reading,
    In order to not read image when we need only meta, which should drastically analytic"""

    def read_meta(self, index) -> Any:
        raise NotImplementedError
