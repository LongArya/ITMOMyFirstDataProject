from dataclasses import dataclass
import pandas as pd


@dataclass
class ClassificationResultsDataframe(pd.DataFrame):
    @property
    def image_path(self) -> str:
        ...

    @property
    def ground_true(self) -> str:
        ...

    @property
    def prediction(self) -> str:
        ...

    @property
    def prediction_score(self) -> float:
        ...
