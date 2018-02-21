import numpy as np
import pandas as pd

from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase


class TwoClassThresholdConverter(TwoClassConverterBase):
    """
    Converts continuous values to classes based on a customized threshold.
    """
    def __init__(self, threshold=None):
        self._threshold = threshold

    def convert(self,
                predicted_probabilities: pd.DataFrame,
                positive_class: object) -> np.ndarray:

        assert predicted_probabilities.shape[1] == 2

        column_names = predicted_probabilities.columns.values
        negative_class = column_names[0] if column_names[1] == positive_class else column_names[1]

        return np.array([positive_class if x > self._threshold else negative_class
                         for x in predicted_probabilities[positive_class]])
