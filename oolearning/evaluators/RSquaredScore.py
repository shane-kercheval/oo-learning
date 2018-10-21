import numpy as np
from sklearn.metrics import r2_score

from oolearning.enums.Metric import Metric
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin
from oolearning.evaluators.ScoreActualPredictedBase import ScoreActualPredictedBase


class RSquaredScore(UtilityFunctionMixin, ScoreActualPredictedBase):
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html
        "Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
        A constant model that always predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0."
    """
    @property
    def name(self) -> str:
        return Metric.R_SQUARED.value

    def _calculate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        return r2_score(y_true=actual_values, y_pred=predicted_values)
