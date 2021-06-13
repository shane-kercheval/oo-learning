from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning.evaluators.TwoClassConfusionMatrix import TwoClassConfusionMatrix
from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class TwoClassEvaluator(EvaluatorBase):
    """
    Evaluates models for two-class classification problems.
    """
    def __init__(self, positive_class: object):
        """
        :param positive_class: if `positive_class` is None, then the confusion matrix is not arranged by
            category
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

    def __str__(self):
        val = super().__str__()
        val += "\n\nConfusion Matrix\n----------------\n\n" + self.matrix.to_string()

        return val

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

    def plot_all_quality_metrics(self, comparison_evaluator: "TwoClassEvaluator" = None):
        """
        Creates a plot that shows all of the quality score_names in this class.

        :param comparison_evaluator: adds additional points to the plot for the score_names associated with
            the `comparison_evaluator`; allows the user to compare two different evaluators (e.g. from two
            different models
        """
        # convert diction to dataframe, without "Total Observations" which will fuck up axis
        # noinspection PyTypeChecker
        metrics_dataframe = pd.DataFrame.from_dict([self.all_quality_metrics])
        metrics_dataframe = metrics_dataframe[list(self.all_quality_metrics.keys())].\
            drop(columns='Total Observations')

        x_values = np.linspace(1, metrics_dataframe.shape[1], metrics_dataframe.shape[1])
        self_y_values = metrics_dataframe.iloc[0].values
        ax = plt.gca()

        if comparison_evaluator is not None:
            # convert diction to dataframe, without "Total Observations" which will fuck up axis
            # noinspection PyTypeChecker
            comparison_metrics_dataframe = pd.DataFrame.from_dict([comparison_evaluator.all_quality_metrics])
            comparison_metrics_dataframe = comparison_metrics_dataframe[
                list(comparison_evaluator.all_quality_metrics.keys())].drop(columns='Total Observations')
            comparison_y_values = comparison_metrics_dataframe.iloc[0].values
            plt.scatter(x_values, comparison_y_values, color='r', alpha=0.7, marker='o', s=75)
            for i, v in enumerate([0] + list(comparison_y_values)):
                if i != 0:
                    ax.text(i + 0.1, v - 0.05,
                            '{0}%'.format(round(v*100, 1)),
                            color='r',
                            ha='center')

        plt.scatter(x_values, self_y_values, color='g', alpha=0.7, marker='o', s=75)

        metrics_list = list(self.all_quality_metrics.keys())
        metrics_list.remove('Total Observations')
        plt.xticks(ticks=np.arange(metrics_dataframe.shape[1]+1),
                   labels=[''] + metrics_list,
                   rotation=17,
                   ha='right')
        plt.yticks(np.linspace(start=0, stop=1, num=21))
        #        plt.scatter(X, Y2, color='g')
        for i, v in enumerate([0] + list(self_y_values)):
            # noinspection PyUnboundLocalVariable
            if i != 0:
                ax.text(i + 0.1, v + 0.025, '{0}%'.format(round(v*100, 1)), color='g', ha='center')

        plt.title('Quality Scores')
        plt.grid()
        plt.tight_layout()
