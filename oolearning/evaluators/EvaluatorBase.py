from abc import ABCMeta, abstractmethod

import numpy as np


class EvaluatorBase(metaclass=ABCMeta):
    """
    An Evaluator object takes the predictions of a model, as well as the actual values, and evaluates the
        model across many metrics.
    """
    @abstractmethod
    def evaluate(self, actual_values: np.ndarray, predicted_values: object):
        """
        Takes the actual and predicted values, and calculates the associated model performance/quality
            metrics.
        :param actual_values: the actual values associated with the target variable (e.g. of the holdout set)
        :param predicted_values: the predicted values from the model (e.g. of the holdout set); most likely
            an array for regression problems and a DataFrame from classification problems.
        """
        pass

    @property
    @abstractmethod
    def all_quality_metrics(self) -> dict:
        """
        :return: dictionary with all the metrics and associated values
        """
        pass
