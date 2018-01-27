import copy
from abc import ABCMeta, abstractmethod
from typing import Tuple, Callable

import numpy as np


class EvaluatorBase(metaclass=ABCMeta):
    """
    An 'evaluator' is responsible for calculating the "accuracy" given a set of actual and predicted values.
    An 'evaluator' has an overall 'accuracy' such as 'kappa' or 'AUC', for example, but can also have other
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
        self._accuracy = None
        self._details = None
        self._better_than = better_than

    def clone(self):
        """
        when, for example, resampling, an Evaluator will have to be cloned several times (before using)
        :return: a clone of the current object
        """
        assert self._accuracy is None  # only intended on being called before evaluating
        return copy.deepcopy(self)

    @property
    def accuracy(self) -> float:
        """
        :return: the calculated accuracy based on the actual & predicted values
        """
        assert isinstance(self._accuracy, float)
        return self._accuracy

    @property
    def better_than_function(self):
        return self._better_than

    def better_than(self, other: 'EvaluatorBase') -> bool:
        assert isinstance(other, EvaluatorBase)
        return self._better_than(self.accuracy, other.accuracy)

    @property
    def details(self) -> object:
        """
        :return: additional information such an object that contains a confusion confusion_matrix and/or other
            metrics (e.g. true positive rate), defined by the inheriting class
        """
        return self._details

    def evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray) -> float:
        """
        given the actual and predicted values, this function calculates the corresponding accuracy, storing
        the accuracy and object associated with detailed information, and returning the accuracy as a
        convenience for users of the method
        :param actual_values: actual values
        :param predicted_values: predicted values (from a trained model)
        :return: accuracy
        """
        assert self._accuracy is None  # we don't want to be able to reuse test_evaluators
        assert actual_values.shape == predicted_values.shape
        self._accuracy, self._details = self._calculate_accuracy(actual_values, predicted_values)
        assert isinstance(self._accuracy, float)
        return self._accuracy

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
    def _calculate_accuracy(self, actual_values: np.ndarray, predicted_values: np.ndarray) ->\
            Tuple[float, object]:
        """
        method for the inheriting class to override that does the accuracy calculation and prepares any
            "details" object necessary
        :return: a tuple containing the accuracy, and an object defining additional accuracy information,
            such as a confusion confusion_matrix
        """
        pass
