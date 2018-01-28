from typing import Tuple
import numpy as np

from .TwoClassEvaluator import TwoClassEvaluator
from ..enums.Metric import Metric


class KappaEvaluator(TwoClassEvaluator):
    def __init__(self,
                 positive_category,
                 negative_category,
                 use_probabilities: bool=True,
                 threshold: float=0.5):
        super().__init__(better_than=lambda this, other: this > other,  # larger Kappa is better
                         positive_category=positive_category,
                         negative_category=negative_category,
                         use_probabilities=use_probabilities,
                         threshold=threshold)

    @property
    def metric_name(self) -> str:
        return Metric.KAPPA.value

    def _calculate_accuracy(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        return self.confusion_matrix.kappa, self._confusion_matrix
