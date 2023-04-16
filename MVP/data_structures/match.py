from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class Match(Generic[T]):
    gt_value: T
    pred_value: T

    @property
    def is_correct(self) -> bool:
        """Match is considered correct if ground true value is the same as predicted"""
        return self.gt_value == self.pred_value
