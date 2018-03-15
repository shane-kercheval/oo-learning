import math
from typing import Tuple

import numpy as np
import pandas as pd

from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase
from oolearning.converters.TwoClassThresholdConverter import TwoClassThresholdConverter
from oolearning.evaluators.TwoClassConfusionMatrix import TwoClassConfusionMatrix


class TwoClassRocOptimizerConverter(TwoClassConverterBase):
    """
    Converts continuous values to classes based on the calculating the threshold that minimizes the
        distance to the upper left corner of an ROC curve.
    """

    def __init__(self, positive_class, actual_classes):
        super().__init__(positive_class=positive_class)
        self._ideal_threshold = None
        self._false_positive_rates = None
        self._true_positive_rates = None
        self._actual_classes = actual_classes

    @property
    def ideal_threshold(self) -> float:
        return self._ideal_threshold

    @property
    def false_positive_rates(self) -> np.ndarray:
        return self._false_positive_rates

    @property
    def true_positive_rates(self) -> np.ndarray:
        return self._true_positive_rates

    def convert(self, values: pd.DataFrame) -> np.ndarray:
        self._false_positive_rates, self._true_positive_rates, self._ideal_threshold = \
            self._calculate_fpr_tpr_ideal_threshold(potential_cutoff_values=np.arange(0.0, 1.01, 0.01),
                                                    actual_classes=self._actual_classes,
                                                    predicted_probabilities=values,
                                                    positive_class=self.positive_class)
        return TwoClassThresholdConverter(threshold=self._ideal_threshold,
                                          positive_class=self.positive_class).convert(values=values)

    @staticmethod
    def _calculate_fpr_tpr_ideal_threshold(potential_cutoff_values: np.ndarray,
                                           actual_classes: np.ndarray,
                                           predicted_probabilities: pd.DataFrame,
                                           positive_class: object) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        helper method to calculate false positive and true positive (sensitivity) rates for possible
        cutoff values between 0 and 1
        :return: array of false positive rates, true positive rates, and an ideal threshold (which
            is calculated by finding the the point on the ROC curve that has minimum distance to the
            upper left point (i.e. a perfect predictor)
        """
        def get_fpr_tpr(threshold):
            converter = TwoClassThresholdConverter(threshold=threshold, positive_class=positive_class)
            converted_classes = converter.convert(values=predicted_probabilities)
            matrix = TwoClassConfusionMatrix(actual_classes=actual_classes,
                                             predicted_classes=converted_classes,
                                             positive_class=positive_class)
            return matrix.false_positive_rate, matrix.true_positive_rate

        fpr_tpr_tuple = [get_fpr_tpr(threshold=x) for x in potential_cutoff_values]  # list of rates
        # remove Nones caused by divide by zero for e.g. FPR/TPR
        fpr_tpr_tuple = [x for x in fpr_tpr_tuple if x[0] is not None and x[1] is not None]
        false_positive_rates, true_positive_rates = zip(*fpr_tpr_tuple)

        # calculate distance from upper left (0, 1)
        def distance_formula(x, y):
            return math.sqrt((0 - x) ** 2 + (1 - y) ** 2)  # i.e. pythagorean theorem

        # calculate distance for each point on the ROC graph
        distances = [distance_formula(x=fpr, y=tpr) for fpr, tpr in fpr_tpr_tuple]
        # index is the index of cutoff value in the range that has the minimum distance from the upper left
        val, index = min((val, index) for (index, val) in enumerate(distances))
        ideal_threshold = potential_cutoff_values[index]

        # round to 2 because sometimes machines fuck up decimal points, but should always be 2
        return false_positive_rates, true_positive_rates, round(ideal_threshold, 2)
