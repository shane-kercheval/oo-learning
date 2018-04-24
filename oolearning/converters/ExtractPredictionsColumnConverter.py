import numpy as np
import pandas as pd

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase


class ExtractPredictionsColumnConverter(ContinuousToClassConverterBase):
    """
    A converter that simply returns the predictions of a particular column. Unlike other Converters, this
        Converter does not convert the predicted values to specific classes. This class is primarily used with
        the ModelStacker class; the ModelStacker uses the predictions (of the positive class in two-class
        classification) of a base model to train the stacking model.
    """
    def __init__(self, column: str):
        """
        :param column: the column name of the predictions to return (i.e. name of the class of interest)
        """
        self._column = column

    def convert(self, values: pd.DataFrame) -> np.ndarray:
        """
        :param values:
            pd.DataFrame with column names as class names
        :return: the specified column
        """
        return values[self._column]
