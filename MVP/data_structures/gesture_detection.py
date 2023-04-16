from dataclasses import dataclass
from typing import Iterable
from static_gesture_classification.static_gesture import StaticGesture


@dataclass
class GestureDetection:
    gesture: StaticGesture
    xyxy_box: Iterable[float]
    score: float
