import numpy as np
from sklearn.metrics import accuracy_score

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.evaluators.ScoreBase import SupportsAnyClassificationMixin


class ErrorRateScore(SupportsAnyClassificationMixin, CostFunctionMixin, ScoreBase):
    @property
    def name(self) -> str:
        return Metric.ERROR_RATE.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return 1 - accuracy_score(y_true=actual_values, y_pred=predicted_values)
