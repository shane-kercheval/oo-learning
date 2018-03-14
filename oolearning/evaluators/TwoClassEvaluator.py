from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning.evaluators.TwoClassConfusionMatrix import TwoClassConfusionMatrix
from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class TwoClassEvaluator(EvaluatorBase):
    """
    Class representing a confusion confusion_matrix for two-class (or 2 category) classifiers.
    """
    def __init__(self, positive_class: object):
        """
             |                  | Predicted Negative | Predicted Positive |
             | ---------------- | ------------------ | ------------------ |
             | Actual Negative  | True Negative      | False Positive     |
             | Actual Positive  | False Negative     | True Positive      |

             if `positive_class` is None, then the confusion matrix is not arranged by category.

             For multi-class problems, `positive_class` is not applicable.
         """
        self._positive_class = positive_class
        self._confusion_matrix = None

    def evaluate(self, actual_values: np.ndarray, predicted_values: np.ndarray):

        self._confusion_matrix = TwoClassConfusionMatrix(actual_classes=actual_values,
                                                         predicted_classes=predicted_values,
                                                         positive_class=self._positive_class)

    @classmethod
    def from_classes(cls,
                     actual_classes: np.ndarray,
                     predicted_classes: np.ndarray,
                     positive_class) -> 'TwoClassEvaluator':
        evaluator = TwoClassEvaluator(positive_class=positive_class)
        evaluator.evaluate(actual_values=actual_classes, predicted_values=predicted_classes)
        return evaluator

    @property
    def matrix(self) -> pd.DataFrame:
        return self._confusion_matrix.matrix

    @property
    def confusion_matrix(self) -> TwoClassConfusionMatrix:
        return self._confusion_matrix

    @property
    def total_observations(self) -> int:
        return self._confusion_matrix.total_observations

    @property
    def sensitivity(self) -> Union[float, None]:
        return self._confusion_matrix.sensitivity

    @property
    def specificity(self) -> Union[float, None]:
        return self._confusion_matrix.specificity

    @property
    def true_positive_rate(self) -> Union[float, None]:
        return self._confusion_matrix.true_positive_rate

    @property
    def true_negative_rate(self) -> Union[float, None]:
        return self._confusion_matrix.true_negative_rate

    @property
    def false_negative_rate(self) -> Union[float, None]:
        return self._confusion_matrix.false_negative_rate

    @property
    def false_positive_rate(self) -> Union[float, None]:
        return self._confusion_matrix.false_positive_rate

    @property
    def accuracy(self) -> Union[float, None]:
        return self._confusion_matrix.accuracy

    @property
    def error_rate(self) -> Union[float, None]:
        return self._confusion_matrix.error_rate

    @property
    def positive_predictive_value(self) -> Union[float, None]:
        return self._confusion_matrix.positive_predictive_value

    @property
    def negative_predictive_value(self) -> Union[float, None]:
        return self._confusion_matrix.negative_predictive_value

    @property
    def prevalence(self) -> Union[float, None]:
        return self._confusion_matrix.prevalence

    @property
    def kappa(self) -> Union[float, None]:
        return self._confusion_matrix.kappa

    @property
    def f1_score(self) -> Union[float, None]:
        return self._confusion_matrix.f1_score

    @property
    def all_quality_metrics(self) -> dict:
        return self._confusion_matrix.all_quality_metrics

    def plot_all_quality_metrics(self):
        # noinspection PyTypeChecker
        x = pd.DataFrame.from_dict([self.all_quality_metrics])
        x = x[list(self.all_quality_metrics.keys())].drop(columns='Total Observations')

        fig, ax = plt.subplots()
        p = x.plot(kind='box',
                   rot=20,
                   title='Quality Scores',
                   yticks=np.linspace(start=0, stop=1, num=21),
                   medianprops=dict(linewidth=4),
                   grid=True,
                   ax=ax)
        plt.xticks(ha='right')

        return p
