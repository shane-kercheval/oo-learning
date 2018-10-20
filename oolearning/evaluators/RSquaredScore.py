import numpy as np
from sklearn.metrics import r2_score

from oolearning.enums.Metric import Metric
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase


class RSquaredScore(UtilityFunctionMixin, ScoreActualPredictedBase):
    @property
    def name(self) -> str:
        return Metric.R_SQUARED.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return r2_score(y_true=actual_values, y_pred=predicted_values)
