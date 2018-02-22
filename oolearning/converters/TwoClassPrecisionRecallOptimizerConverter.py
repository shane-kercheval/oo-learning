import math
from typing import Tuple

import numpy as np
import pandas as pd

from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase
from oolearning.converters.TwoClassThresholdConverter import TwoClassThresholdConverter
from oolearning.evaluators.TwoClassConfusionMatrix import TwoClassConfusionMatrix


class TwoClassPrecisionRecallOptimizerConverter(TwoClassConverterBase):
    """
    Converts continuous values to classes based on the calculating the threshold that minimizes the
        distance to the upper left corner of an ROC curve.
    """
    def __init__(self, positive_class, actual_classes):
        super().__init__(positive_class=positive_class)
        self._ideal_threshold = None
        self._positive_predictive_value = None
        self._true_positive_rates = None
        self._actual_classes = actual_classes

    @property
    def ideal_threshold(self) -> float:
        return self._ideal_threshold

    @property
    def positive_predictive_values(self) -> np.ndarray:
        return self._positive_predictive_value

    @property
    def true_positive_rates(self) -> np.ndarray:
        return self._true_positive_rates

    def convert(self, values: pd.DataFrame) -> np.ndarray:
        self._positive_predictive_value, self._true_positive_rates, self._ideal_threshold = \
            self._calculate_ppv_tpr_ideal_threshold(potential_cutoff_values=np.arange(0.0, 1.01, 0.01),
                                                    actual_classes=self._actual_classes,
                                                    predicted_probabilities=values,
                                                    positive_class=self.positive_class)
        return TwoClassThresholdConverter(threshold=self._ideal_threshold,
                                          positive_class=self.positive_class).convert(values=values)

    @staticmethod
    def _calculate_ppv_tpr_ideal_threshold(potential_cutoff_values: np.ndarray,
                                           actual_classes: np.ndarray,
                                           predicted_probabilities: pd.DataFrame,
                                           positive_class: object) -> Tuple[np.ndarray, np.ndarray, float]:

        def get_ppv_tpr(threshold):
            converter = TwoClassThresholdConverter(threshold=threshold, positive_class=positive_class)
            converted_classes = converter.convert(values=predicted_probabilities)
            matrix = TwoClassConfusionMatrix(actual_classes=actual_classes,
                                             predicted_classes=converted_classes,
                                             positive_class=positive_class)

            return matrix.positive_predictive_value, matrix.true_positive_rate

        ppv_tpr_tuple = [get_ppv_tpr(threshold=x) for x in potential_cutoff_values]  # list of rates
        # remove Nones caused by divide by zero for e.g. FPR/TPR
        ppv_tpr_tuple = [x for x in ppv_tpr_tuple if x[0] is not None and x[1] is not None]
        positive_predictive_values, true_positive_rates = zip(*ppv_tpr_tuple)

        # calculate distance from upper right (1, 1)
        def distance_formula(x, y):
            return math.sqrt((1 - x) ** 2 + (1 - y) ** 2)  # i.e. pythagorean theorem

        # calculate distance for each point on the ROC graph
        # precision (i.e. positive_predictive_values) on Y-axis and Recall (i.e. Sensitivity) on X-axis
        distances = [distance_formula(x=tpr, y=ppv) for ppv, tpr in ppv_tpr_tuple]
        # index is the index of cutoff value in the range that has the minimum distance from the upper left
        val, index = min((val, index) for (index, val) in enumerate(distances))
        ideal_threshold = potential_cutoff_values[index]

        return positive_predictive_values, true_positive_rates, ideal_threshold
