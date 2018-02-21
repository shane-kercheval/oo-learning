import numpy as np
from sklearn.metrics import accuracy_score

from oolearning.enums.Metric import Metric
from oolearning.evaluators.ScoreBase import SupportsAnyClassificationMixin, ClassificationScoreBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class AccuracyScore(SupportsAnyClassificationMixin, UtilityFunctionMixin, ClassificationScoreBase):
    @property
    def name(self) -> str:
        return Metric.ACCURACY.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return accuracy_score(y_true=actual_values, y_pred=predicted_values)
