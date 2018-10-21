from typing import Union

import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase


class MspeScore(CostFunctionMixin, ScoreActualPredictedBase):
    """
    https://www.coursera.org/learn/competitive-data-science/lecture/qhRmV/regression-metrics-review-ii

    NOTE: if any ``actual_values` contain zero, the original formula will have a divide-by-zero error.
        Therefore, the option to add a constant is given. However, it is still possible that the any of the
        actual_values plus the constant will equal a zero.

        You also have to be very careful of the constant you add. If you add a very small constant, then the
        denominator will be a very small number, which will result in a very high error.
        A reasonable constant might be `1`, but it will still give a very different outcome than not adding a
        constant. Therefore, use with caution. It is recommended for datasets where an outcome of zero is not
        possible.
    """
    def __init__(self, constant: Union[float, None]=None):
        """
        :param constant: this is the value of the constant that is added to the actual and predicted values
            to (help) avoid a divide-by-zero problem.
        """
        super().__init__()
        self._constant = constant

    @property
    def name(self) -> str:
        return Metric.MEAN_SQUARED_PERCENTAGE_ERROR.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        # needed because if the actual_values contains zero, we get NA; so we add a very small amount to
        # each value. This will have varying degrees of influence on the final calculated score depending on
        # how large the actual & predicted values are. Additionally, there is still a chance that the original
        # value plus the constant equals 0, in which case we would still get NA.
        adjusted_actuals = actual_values + self._constant if self._constant else actual_values
        adjusted_predictions = predicted_values + self._constant if self._constant else predicted_values

        return float(np.mean(np.square((adjusted_actuals - adjusted_predictions) / adjusted_actuals)))
