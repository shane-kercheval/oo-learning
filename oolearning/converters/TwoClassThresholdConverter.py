import numpy as np
import pandas as pd

from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase


class TwoClassThresholdConverter(TwoClassConverterBase):
    """
    Converts continuous values to classes based on a customized threshold.
    """
    def __init__(self, positive_class, threshold=0.5):
        super().__init__(positive_class=positive_class)
        self._threshold = threshold

    def convert(self, values: pd.DataFrame) -> np.ndarray:

        assert values.shape[1] == 2

        column_names = values.columns.values
        negative_class = column_names[0] if column_names[1] == self.positive_class else column_names[1]

        return np.array([self.positive_class if x > self._threshold else negative_class
                         for x in values[self.positive_class]])
