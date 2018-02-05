from typing import Tuple
import numpy as np
from oolearning.enums.Metric import Metric
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class SensitivityEvaluator(UtilityFunctionMixin, TwoClassEvaluator):
    @property
    def metric_name(self) -> str:
        return Metric.SENSITIVITY.value

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return self.confusion_matrix.sensitivity, self._confusion_matrix
