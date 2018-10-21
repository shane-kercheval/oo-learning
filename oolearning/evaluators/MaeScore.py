import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin


class MaeScore(CostFunctionMixin, ScoreActualPredictedBase):
    @property
    def name(self):
        return Metric.MEAN_ABSOLUTE_ERROR.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return float(np.mean(np.abs(actual_values-predicted_values)))
