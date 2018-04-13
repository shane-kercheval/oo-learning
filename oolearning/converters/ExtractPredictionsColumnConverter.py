import numpy as np
import pandas as pd

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase


class ExtractPredictionsColumnConverter(ContinuousToClassConverterBase):
    def __init__(self, column):
        """
        :param column: the column name of the predictions to return (i.e. name of the class of interest)
        """
        self._column = column

    def convert(self, values: pd.DataFrame) -> np.ndarray:
        return values[self._column]
