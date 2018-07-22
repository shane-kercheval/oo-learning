import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.RegressionEvaluator import RegressionEvaluator
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase


class RmseScore(CostFunctionMixin, ScoreActualPredictedBase):
    @property
    def name(self) -> str:
        return Metric.ROOT_MEAN_SQUARE_ERROR.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return RegressionEvaluator().evaluate(actual_values=actual_values,
                                              predicted_values=predicted_values).root_mean_squared_error
