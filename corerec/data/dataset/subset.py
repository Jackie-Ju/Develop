import time
from typing import TypeVar, Sequence
from torch.utils.data import Dataset

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class Subset(Dataset[T_co]):
    r"""
    Subset of a dataset with weights at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        tmp_batch = self.dataset[self.indices[idx]]
        return tmp_batch

    def __len__(self):
        return len(self.indices)