import numpy as np
from sklearn.metrics import roc_auc_score

from oolearning.enums.Metric import Metric
from oolearning.evaluators.ScoreBase import SupportsTwoClassProbabilitiesMixin, ClassificationScoreBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


class AucScore(SupportsTwoClassProbabilitiesMixin, UtilityFunctionMixin, ClassificationScoreBase):
    @property
    def name(self) -> str:
        return Metric.AREA_UNDER_CURVE.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return roc_auc_score(y_true=actual_values, y_score=predicted_values)
