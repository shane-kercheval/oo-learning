import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase


class MseScore(CostFunctionMixin, ScoreActualPredictedBase):
    @property
    def name(self) -> str:
        return Metric.MEAN_SQUARED_ERROR.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return float(np.mean(np.square(actual_values - predicted_values)))
