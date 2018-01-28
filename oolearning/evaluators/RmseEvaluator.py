from typing import Tuple

import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class RmseEvaluator(EvaluatorBase):
    def __init__(self):
        super().__init__(better_than=lambda this, other: this < other)  # smaller RMSE is better

    @property
    def metric_name(self) -> str:
        return Metric.ROOT_MEAN_SQUARE_ERROR.value

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return np.sqrt(np.mean(np.square(predicted_values - actual_values))), None
