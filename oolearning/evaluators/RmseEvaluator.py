from typing import Tuple

import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class RmseEvaluator(CostFunctionMixin, EvaluatorBase):
    @property
    def metric_name(self) -> str:
        return Metric.ROOT_MEAN_SQUARE_ERROR.value

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return np.sqrt(np.mean(np.square(predicted_values - actual_values))), None
