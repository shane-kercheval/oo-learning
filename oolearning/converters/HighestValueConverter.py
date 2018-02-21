from typing import Union, List

import numpy as np
import pandas as pd

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase


class HighestValueConverter(ContinuousToClassConverterBase):
    """
    Converts continuous values to classes based on a customized threshold.
    """
    def convert(self, values: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        :param values:
            pd.DataFrame with column names as class names
        :param kwargs:
        :return: list of class predictions
        """
        if len(kwargs) != 0:
            raise ValueError('should not pass any other parameters to `convert` except for `values`')

        return list(values.idxmax(axis=1).values)
