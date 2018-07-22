from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase


class TwoClassScoreBase(ScoreActualPredictedBase):
    def __init__(self, positive_class: object):
        super().__init__()
        self._positive_class = positive_class

    @property
    def positive_class(self):
        return self._positive_class

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _better_than(self, this: float, other: float) -> bool:
        pass

    @abstractmethod
    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:
        pass
