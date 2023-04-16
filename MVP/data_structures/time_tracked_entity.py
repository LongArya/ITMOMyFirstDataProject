from typing import TypeVar, Generic
from dataclasses import dataclass

T = TypeVar("T")


@dataclass
class TimeTrackedEntity(Generic[T]):
    entity: T
    tracked_time_seconds: float

    def accumulate_time(self, time_seconds: float) -> None:
        self.tracked_time_seconds += time_seconds
