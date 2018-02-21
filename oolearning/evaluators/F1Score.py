import numpy as np
from sklearn.metrics import f1_score

from oolearning.enums.Metric import Metric
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator
from oolearning.evaluators.ScoreBase import ClassificationScoreBase, SupportsTwoClassClassificationMixin
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class F1Score(SupportsTwoClassClassificationMixin, UtilityFunctionMixin, ClassificationScoreBase):
    @property
    def name(self) -> str:
        return Metric.F1_SCORE.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return TwoClassEvaluator.from_classes(actual_classes=actual_values,
                                              predicted_classes=predicted_values,
                                              positive_category=self._positive_class).f1_score
