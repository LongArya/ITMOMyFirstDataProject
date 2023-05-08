from collections import deque
from typing import TypeVar, Generic, List, Deque, Optional
from dataclasses import dataclass
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity

T = TypeVar("T")


@dataclass
class TrackedObject(Generic[T]):
    """Representation of object at time point"""

    time_stamp: int
    object: T


class Track(Generic[T]):
    """Class for keeping time series of some objects"""

    def __init__(self, buffer_size: Optional[int], track_id: int):
        self.track_id = track_id
        self.tracked_series: Deque[TrackedObject[T]] = deque(maxlen=buffer_size)
        self._tracks_updates_number: int = 0

    def extend_track(self, tracked_frame: TrackedObject[T]) -> None:
        self.tracked_series.append(tracked_frame)
        self._tracks_updates_number += 1

    @property
    def last(self) -> TrackedObject[T]:
        return self.tracked_series[-1]

    @property
    def last_updated_frame(self) -> int:
        return self.tracked_series[-1].time_stamp

    @property
    def tracks_updates_number(self):
        return self._tracks_updates_number

    def is_stale(self, current_frame: int, expiration_frame_num: int) -> bool:
        is_stale = current_frame - self.last_updated_frame > expiration_frame_num
        return is_stale
