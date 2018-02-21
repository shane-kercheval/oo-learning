import numpy as np

from oolearning.enums.Metric import Metric
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator
from oolearning.evaluators.ScoreBase import SupportsTwoClassClassificationMixin, \
    ClassificationScoreBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class SpecificityScore(SupportsTwoClassClassificationMixin, UtilityFunctionMixin, ClassificationScoreBase):
    @property
    def name(self) -> str:
        return Metric.SPECIFICITY.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return TwoClassEvaluator.from_classes(actual_classes=actual_values,
                                              predicted_classes=predicted_values,
                                              positive_category=self._positive_class).true_negative_rate
