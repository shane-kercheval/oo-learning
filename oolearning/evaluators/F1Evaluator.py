from typing import Tuple
import numpy as np
from oolearning.enums.Metric import Metric
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator


class F1Evaluator(UtilityFunctionMixin, TwoClassEvaluator):
    @property
    def metric_name(self) -> str:
        return Metric.F1_SCORE.value

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return self.confusion_matrix.f1_score, self._confusion_matrix
