from MVP.data_structures.time_based_selection import TimeBasedSelection
from MVP.data_structures.time_tracked_entity import TimeTrackedEntity
from static_gesture_classification.static_gesture import StaticGesture
from typing import Generic, TypeVar, Optional, Dict


T = TypeVar("T")


class TimeBasedGesturesOptionsTracker(Generic[T]):
    def __init__(
        self,
        gesture_to_option_mapping: Dict[StaticGesture, T],
        required_time_seconds: float,
    ) -> None:
        self.gesture_to_option_mapping = gesture_to_option_mapping
        self.required_time_seconds = required_time_seconds
        self._current_option_selection: Optional[TimeBasedSelection[T]] = None

    def reset(self) -> None:
        self._current_option_selection = None

    def update_inner_state(
        self, active_gesture_selection: Optional[TimeTrackedEntity[StaticGesture]]
    ) -> None:
        if active_gesture_selection is None:
            self._current_option_selection = None
            return
        option: Optional[T] = self.gesture_to_option_mapping.get(
            active_gesture_selection.entity, None
        )
        if option is None:
            self._current_option_selection = None
            return
        self._current_option_selection = TimeBasedSelection(
            selected_object=option,
            accumulated_time_seconds=active_gesture_selection.tracked_time_seconds,
            time_requirement_seconds=self.required_time_seconds,
        )

    @property
    def current_selection(self) -> Optional[TimeBasedSelection[T]]:
        return self._current_option_selection

    @property
    def active_option(self) -> Optional[T]:
        if self._current_option_selection is None:
            return None
        if not self._current_option_selection.is_active:
            return None
        return self._current_option_selection.selected_object
