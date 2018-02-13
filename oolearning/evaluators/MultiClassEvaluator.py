from typing import Tuple

import numpy as np
import pandas as pd

from oolearning.enums.Metric import Metric
from oolearning.evaluators.ConfusionMatrixMC import ConfusionMatrixMC
from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin


# noinspection PyAbstractClass
class MultiClassEvaluator(EvaluatorBase):
    def __init__(self):
        self._confusion_matrix = None
        super().__init__()

    @staticmethod
    def _get_predicted_categories(self, probabilities: pd.DataFrame) -> np.ndarray:
        pass

    def evaluate(self, actual_values: np.ndarray, predicted_values: pd.DataFrame):
        """
        # TODO
        """
        predicted_classes = self._get_predicted_categories(probabilities=predicted_values)
        self._confusion_matrix = ConfusionMatrixMC(actual_classes=actual_values,
                                                   predicted_classes=predicted_classes)
        return super().evaluate(actual_values=actual_values, predicted_values=predicted_classes)


class AccuracyMultiClassEvaluator(UtilityFunctionMixin, MultiClassEvaluator):
    def _better_than(self, this: float, other: float) -> bool:
        pass

    @property
    def metric_name(self) -> str:
        return Metric.ACCURACY.value

    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> \
            Tuple[float, object]:
        return self._confusion_matrix.accuracy, self._confusion_matrix

