import numpy as np
import pandas as pd

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase


class HighestValueConverter(ContinuousToClassConverterBase):
    """
    Converts continuous values to classes based on a customized threshold.
    """
    def convert(self, values: pd.DataFrame) -> np.ndarray:
        """
        :param values:
            pd.DataFrame with column names as class names
        :return: list of class predictions
        """
        return values.idxmax(axis=1).values
