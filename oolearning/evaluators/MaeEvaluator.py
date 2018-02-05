from typing import Tuple

import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class MaeEvaluator(CostFunctionMixin, EvaluatorBase):
    @property
    def metric_name(self):
        return Metric.MEAN_ABSOLUTE_ERROR.value

    # noinspection PyTypeChecker,PyMethodMayBeStatic
    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return np.mean(np.abs(predicted_values - actual_values)), None
