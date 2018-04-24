import numpy as np
import pandas as pd

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase


class HighestValueConverter(ContinuousToClassConverterBase):
    """
    Converts continuous values to classes based on the column/class that has the highest value.
    """
    def convert(self, values: pd.DataFrame) -> np.ndarray:
        """
        :param values: the (DataFrame) output of a model's `.predict()`, which has column names as class names
        :return: an array of class predictions.
        """
        return values.idxmax(axis=1).values
