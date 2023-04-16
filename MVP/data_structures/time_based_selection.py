from typing import TypeVar, Generic
from dataclasses import dataclass
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity

T = TypeVar("T")


@dataclass
class TimeBasedSelection(Generic[T]):
    """Class for working with time based selections"""

    selected_object: T
    accumulated_time_seconds: float
    time_requirement_seconds: float

    def accumulate_time(self, time: float) -> None:
        self.accumulated_time_seconds += time
        self.accumulated_time_seconds = min(
            self.accumulated_time_seconds, self.time_requirement_seconds
        )

    @property
    def is_active(self) -> bool:
        return self.accumulated_time_seconds == self.time_requirement_seconds

    @property
    def proportion_of_completed_time(self) -> float:
        return self.accumulated_time_seconds / self.time_requirement_seconds
