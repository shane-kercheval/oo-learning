import numpy as np
from sklearn.metrics import cohen_kappa_score

from oolearning.enums.Metric import Metric
from oolearning.evaluators.ScoreBase import ScoreBase, SupportsTwoClassClassificationMixin, \
    ClassificationScoreBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class KappaScore(SupportsTwoClassClassificationMixin, UtilityFunctionMixin, ClassificationScoreBase):
    @property
    def name(self) -> str:
        return Metric.KAPPA.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return cohen_kappa_score(y1=actual_values, y2=predicted_values)
