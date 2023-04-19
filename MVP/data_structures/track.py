from collections import deque
from typing import TypeVar, Generic, List, Deque, Optional
from dataclasses import dataclass
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity

T = TypeVar("T")


@dataclass
class TrackedObject(Generic[T]):
    frame: int
    object: T


class Track(Generic[T]):
    """Class for keeping series of some objects"""

    def __init__(self, buffer_size: Optional[int], track_id: int):
        self.track_id = track_id
        self.tracked_series: Deque[TrackedObject[T]] = deque(maxlen=buffer_size)
        self._tracked_history: int = 0

    def extend_track(self, tracked_frame: TrackedObject[T]) -> None:
        self.tracked_series.append(tracked_frame)
        self._tracked_history += 1

    @property
    def last(self) -> TrackedObject[T]:
        return self.tracked_series[-1]

    @property
    def last_updated_frame(self) -> int:
        return self.tracked_series[-1].frame

    @property
    def tracked_history(self):
        return self._tracked_history

    def is_stale(self, current_frame: int, expiration_frame_num: int) -> bool:
        is_stale = current_frame - self.last_updated_frame > expiration_frame_num
        return is_stale
