from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from oolearning.evaluators.ScoreBase import ScoreBase


class ScoreActualPredictedBase(ScoreBase):
    def calculate(self,
                  actual_values: np.ndarray,
                  predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        given the actual and predicted values, this function calculates the corresponding value/score
        :param actual_values: actual values
        :param predicted_values: predicted values (from a trained model)
        :return: calculated score
        """
        assert self._value is None  # we don't want to be able to reuse test_evaluators
        self._value = self._calculate(actual_values, predicted_values)
        assert isinstance(self._value, float) or isinstance(self._value, int)
        return self._value

    @abstractmethod
    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:
        """
        This method calculates the value of the metric/Score.
        :param actual_values: the actual values of the target variable
        :param predicted_values: the predicted values of the target variable
        :return: the Score's value
        """
        pass
