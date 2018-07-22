import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.RegressionEvaluator import RegressionEvaluator


class MaeScore(CostFunctionMixin, ScoreActualPredictedBase):
    @property
    def name(self):
        return Metric.MEAN_ABSOLUTE_ERROR.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        # noinspection PyTypeChecker
        return RegressionEvaluator().evaluate(actual_values=actual_values,
                                              predicted_values=predicted_values).mean_absolute_error
