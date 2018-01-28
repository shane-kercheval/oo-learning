import copy
from abc import ABCMeta, abstractmethod
from typing import Tuple, Callable

import numpy as np


class EvaluatorBase(metaclass=ABCMeta):
    """
    An 'evaluator' is responsible for calculating the "value" given a set of actual and predicted values.
    An 'evaluator' has an overall 'value' such as 'kappa' or 'AUC', for example, but can also have other
        detailed information such an object that contains a confusion confusion_matrix and other metrics
        (e.g. true positive rate), defined by the inheriting class
    """

    def __init__(self, better_than: Callable[[float, float], bool]):
        """
        :param better_than: function that takes two floats and, specific to the type of metric, returns True
            if the first float is "better" than the second float, otherwise False.
            e.g. when comparing Kappas the larger number is "better", when comparing RMSE smaller numbers are
                "better"
        """
        self._value = None
        self._details = None
        self._better_than = better_than

    def clone(self):
        """
        when, for example, resampling, an Evaluator will have to be cloned several times (before using)
        :return: a clone of the current object
        """
        assert self._value is None  # only intended on being called before evaluating
        return copy.deepcopy(self)

    @property
    def value(self) -> float:
        """
        :return: the calculated value based on the actual & predicted values. You can think of this as the
            "accuracy"; however, "accuracy" isn't the appropriate term since an Evaluator can be based on an
            "error rate".
        """
        assert isinstance(self._value, float)
        return self._value

    @property
    def better_than_function(self):
        return self._better_than

    def better_than(self, other: 'EvaluatorBase') -> bool:
        assert isinstance(other, EvaluatorBase)
        return self._better_than(self.value, other.value)

    def evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        """
        given the actual and predicted values, this function calculates the corresponding value, storing
        the value and object associated with detailed information, and returning the value as a
        convenience for users of the method
        :param actual_values: actual values
        :param predicted_values: predicted values (from a trained model)
        :return: value
        """
        assert self._value is None  # we don't want to be able to reuse test_evaluators
        assert actual_values.shape == predicted_values.shape
        self._value, self._details = self._evaluate(actual_values, predicted_values)
        assert isinstance(self._value, float)
        return self._value

    def __lt__(self, other):
        return self.better_than(other=other)

    ##########################################################################################################
    # Abstract Methods
    ##########################################################################################################
    @property
    @abstractmethod
    def metric_name(self) -> str:
        """
        :return: 'friendly' name to identify the metric such as 'RMSE'
        """
        pass

    @abstractmethod
    def _evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        """
        method for the inheriting class to override that calculates the value e.g. "accuracy" and prepares any
            "details" object necessary
        :return: a tuple containing the value, and an object defining additional value information,
            such as a confusion confusion_matrix
        """
        pass
