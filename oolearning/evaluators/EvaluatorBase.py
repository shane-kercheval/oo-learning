from abc import ABCMeta, abstractmethod

import numpy as np

from oolearning.OOLearningHelpers import OOLearningHelpers


class EvaluatorBase(metaclass=ABCMeta):
    """
    An Evaluator object takes the predictions of a model, as well as the actual values, and evaluates the
        model across many score_names.
    """
    @abstractmethod
    def evaluate(self, actual_values: np.ndarray, predicted_values: object):
        """
        Takes the actual and predicted values, and calculates the associated model performance/quality
            score_names.
        :param actual_values: the actual values associated with the target variable (e.g. of the holdout set)
        :param predicted_values: the predicted values from the model (e.g. of the holdout set); most likely
            an array for regression problems and a DataFrame from classification problems.
        """
        pass

    @property
    @abstractmethod
    def all_quality_metrics(self) -> dict:
        """
        :return: dictionary with all the score_names and associated values
        """

    def __str__(self):
        return str(OOLearningHelpers.round_dict(self.all_quality_metrics)).replace(", ", "\n ")
