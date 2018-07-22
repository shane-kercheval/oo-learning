from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase
from oolearning.enums.Metric import Metric
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class AccuracyScore(UtilityFunctionMixin, ScoreActualPredictedBase):
    """
    For classification problems, calculates simple "accuracy"; uses sklearn's `accuracy_score()` function:
        http://scikit-learn.org/stable/modules/generated/sklearn.score_names.accuracy_score.html
    """
    def __init__(self, converter: ContinuousToClassConverterBase):
        super().__init__()
        self._converter = converter

    @property
    def name(self) -> str:
        return Metric.ACCURACY.value

    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:
        predicted_classes = self._converter.convert(values=predicted_values)
        return accuracy_score(y_true=actual_values, y_pred=predicted_classes)
