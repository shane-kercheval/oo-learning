from typing import Tuple

import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class AucEvaluator(UtilityFunctionMixin, TwoClassEvaluator):
    @property
    def metric_name(self) -> str:
        return Metric.AREA_UNDER_CURVE.value

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return self.auc, self._confusion_matrix
