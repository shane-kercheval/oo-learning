from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from oolearning.enums.Metric import Metric
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class AucPrecisionRecallScore(UtilityFunctionMixin, ScoreBase):
    def __init__(self, positive_class: object):
        super().__init__()
        self._positive_class = positive_class

    @property
    def name(self) -> str:
        return Metric.AUC_PRECISION_RECALL.value

    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:

        return average_precision_score(y_true=[1 if x == self._positive_class else 0 for x in actual_values],
                                       y_score=predicted_values[self._positive_class])
