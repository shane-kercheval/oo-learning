from typing import Tuple

import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator


class ErrorRateTwoClassEvaluator(CostFunctionMixin, TwoClassEvaluator):
    @property
    def metric_name(self) -> str:
        return Metric.ERROR_RATE.value

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return self.confusion_matrix.error_rate, self._confusion_matrix
