from collections import deque
from typing import TypeVar, Generic, List, Deque, Optional
from dataclasses import dataclass
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity

T = TypeVar("T")


class Track(Generic[T]):
    """Class for keeping series of some objects"""

    def __init__(self, buffer_size: Optional[int], track_id: int):
        self.track_id = track_id
        self.tracked_series: Deque[T] = deque(maxlen=buffer_size)

    def extend(self, entity: T) -> None:
        self.tracked_series.append(entity)

    @property
    def last(self) -> T:
        return self.tracked_series[-1]
