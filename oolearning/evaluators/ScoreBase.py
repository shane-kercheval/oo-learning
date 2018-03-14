import copy
from abc import ABCMeta, abstractmethod
from typing import Union

import numpy as np
import pandas as pd


class ScoreBase(metaclass=ABCMeta):
    def __init__(self):
        self._value = None

    def clone(self):
        """
        when, for example, resampling, an Evaluator will have to be cloned several times (before using)
        :return: a clone of the current object
        """
        assert self._value is None  # only intended on being called before evaluating
        return copy.deepcopy(self)

    @property
    def value(self) -> float:
        return self._value

    def better_than(self, other: 'ScoreBase') -> bool:
        assert isinstance(other, ScoreBase)
        return self._better_than(this=self.value, other=other.value)

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

    def __lt__(self, other):
        return self.better_than(other=other)

    ##########################################################################################################
    # Abstract Methods
    ##########################################################################################################
    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: 'friendly' name to identify the metric such as 'RMSE'
        """
        pass

    @abstractmethod
    def _better_than(self, this: float, other: float) -> bool:
        pass

    @abstractmethod
    def _calculate(self,
                   actual_values: np.ndarray,
                   predicted_values: Union[np.ndarray, pd.DataFrame]) -> float:
        pass
