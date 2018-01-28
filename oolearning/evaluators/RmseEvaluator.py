from typing import Tuple

import numpy as np

from .EvaluatorBase import EvaluatorBase
from ..enums.Metric import Metric


class RmseEvaluator(EvaluatorBase):
    def __init__(self):
        super().__init__(better_than=lambda this, other: this < other)  # smaller RMSE is better

    @property
    def metric_name(self) -> str:
        return Metric.ROOT_MEAN_SQUARE_ERROR.value

    def _calculate_accuracy(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return np.sqrt(np.mean(np.square(predicted_values - actual_values))), None
