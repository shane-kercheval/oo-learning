from typing import Union

import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase


class RmsleScore(CostFunctionMixin, ScoreActualPredictedBase):
    """
    https://www.coursera.org/learn/competitive-data-science/lecture/qhRmV/regression-metrics-review-ii

    RMSLE, like MAPE/MSPE cares more about relative errors than absolute errors

    "From the perspective of RMSLE, it is always better to predict more (over the target value) than the same
    amount less than target" i.e. it will tend to over-predict rather than under-predict
    """
    def __init__(self, constant: Union[float, None]=1):
        """
        :param constant: log(0) gives an error, so we use log(x + constant)
        """
        super().__init__()
        self._constant = constant

    @property
    def name(self) -> str:
        return Metric.ROOT_MEAN_SQUARE_LOGARITHMIC_ERROR.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        adjusted_actuals = actual_values + self._constant if self._constant else actual_values
        adjusted_predictions = predicted_values + self._constant if self._constant else predicted_values

        return np.sqrt(np.mean(np.square(np.log(adjusted_actuals) - np.log(adjusted_predictions))))
