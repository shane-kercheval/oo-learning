from typing import Union

import numpy as np
import pandas as pd

from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase


class TwoClassThresholdConverter(TwoClassConverterBase):
    """
    Converts continuous values to classes based on a customized threshold corresponding to the 'positive'
        class's values. Any value that is greater than `threshold` will be converted to a positive class,
        otherwise it will be converted to a negative class. The class names are the names of the columns in
        the `values` DataFrame passed to `.convert()`
    """
    def __init__(self, positive_class: Union[str, int], threshold: float=0.5):
        """
        :param positive_class: the value of the positive class of the target variable in the corresponding
            data-set being trained/predicted/evaluated.

            For example, for the titanic data-set, the positive class might be `lived`, or might be `1`

        :param threshold: the threshold to use when converting the continuous prediction to a class prediction
        """
        super().__init__(positive_class=positive_class)
        self._threshold = threshold

    def convert(self, values: pd.DataFrame) -> np.ndarray:
        """
        :param values: the (DataFrame) output of a model's `.predict()`, which has column names as class names
        :return: an array of class predictions.
        """
        assert values.shape[1] == 2

        column_names = values.columns.values
        negative_class = column_names[0] if column_names[1] == self.positive_class else column_names[1]

        return np.array([self.positive_class if x > self._threshold else negative_class
                         for x in values[self.positive_class]])
