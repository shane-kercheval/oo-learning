from typing import Tuple

import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class MaeEvaluator(EvaluatorBase):
    def __init__(self):
        super().__init__(better_than=lambda this, other: this < other)  # smaller MAE is better

    @property
    def metric_name(self):
        return Metric.MEAN_ABSOLUTE_ERROR.value

    # noinspection PyTypeChecker,PyMethodMayBeStatic
    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return np.mean(np.abs(predicted_values - actual_values)), None
