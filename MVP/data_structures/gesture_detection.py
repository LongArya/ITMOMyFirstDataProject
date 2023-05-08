from dataclasses import dataclass
from typing import List
from static_gesture_classification.static_gesture import StaticGesture


@dataclass
class GestureDetection:
    """Representaion of hand gesture detection"""

    gesture: StaticGesture
    xyxy_box: List[float]
    score: float
