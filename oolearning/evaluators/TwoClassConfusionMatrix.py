from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning.evaluators.ConfusionMatrix import ConfusionMatrix


class TwoClassConfusionMatrix(ConfusionMatrix):
    """
    Class representing a confusion matrix for two-class (or 2 category) classifiers.

        |                  | Predicted Negative | Predicted Positive |
        | ---------------- | ------------------ | ------------------ |
        | Actual Negative  | True Negative      | False Positive     |
        | Actual Positive  | False Negative     | True Positive      |
    """
    def __init__(self,
                 actual_classes: np.ndarray,
                 predicted_classes: np.ndarray,
                 positive_class: object):
        unique_classes = list(set(actual_classes))  # get unique values then convert to list
        assert len(unique_classes) == 2
        negative_class = unique_classes[0] if positive_class == unique_classes[1] else unique_classes[1]

        super().__init__(actual_classes=actual_classes,
                         predicted_classes=predicted_classes,
                         class_order=[negative_class, positive_class])

        self._positive_class = positive_class
        self._negative_class = negative_class
        category_list = [self._negative_class, self._positive_class]

        self._actual_positives = self.matrix.loc[self._positive_class][category_list].sum()
        self._actual_negatives = self.matrix.loc[self._negative_class][category_list].sum()

        self._true_positives = self.matrix.loc[self._positive_class, self._positive_class]
        self._true_negatives = self.matrix.loc[self._negative_class, self._negative_class]
        self._false_positives = self.matrix.loc[self._negative_class, self._positive_class]
        self._false_negatives = self.matrix.loc[self._positive_class, self._negative_class]

    @property
    def sensitivity(self) -> float:
        """
        :return: a.k.a true positive rate
        """
        return 0 if self._actual_positives == 0 else self._true_positives / self._actual_positives

    @property
    def specificity(self) -> float:
        """
        :return: a.k.a false positive rate
        """
        return 0 if self._actual_negatives == 0 else self._true_negatives / self._actual_negatives

    @property
    def true_positive_rate(self) -> float:
        return self.sensitivity

    @property
    def true_negative_rate(self) -> float:
        return self.specificity

    @property
    def false_negative_rate(self) -> float:
        return 0 if self._actual_positives == 0 else self._false_negatives / self._actual_positives

    @property
    def false_positive_rate(self) -> float:
        return 0 if self._actual_negatives == 0 else self._false_positives / self._actual_negatives

    @property
    def accuracy(self) -> Union[float, None]:
        return None if self.total_observations == 0 else \
            (self._true_negatives + self._true_positives) / self.total_observations

    @property
    def error_rate(self) -> Union[float, None]:
        return None if self.total_observations == 0 else \
            (self._false_positives + self._false_negatives) / self.total_observations

    @property
    def positive_predictive_value(self) -> float:
        return 0 if (self._true_positives + self._false_positives) == 0 else \
            self._true_positives / (self._true_positives + self._false_positives)

    @property
    def negative_predictive_value(self) -> float:
        return 0 if (self._true_negatives + self._false_negatives) == 0 else \
            self._true_negatives / (self._true_negatives + self._false_negatives)

    @property
    def prevalence(self) -> Union[float, None]:
        return None if self.total_observations == 0 else \
            (self._true_positives + self._false_negatives) / self.total_observations

    @property
    def kappa(self) -> Union[float, None]:
        if self.total_observations == 0 or \
                ((self._true_negatives + self._false_negatives) / self.total_observations) == 0:
            return None
        # proportion of the actual agreements
        # add the proportion of all instances where the predicted type and actual type agree
        pr_a = (self._true_negatives + self._true_positives) / self.total_observations
        # probability of both predicted and actual being negative
        p_negative_prediction_and_actual = \
            ((self._true_negatives + self._false_positives) / self.total_observations) * \
            ((self._true_negatives + self._false_negatives) / self.total_observations)
        # probability of both predicted and actual being positive
        p_positive_prediction_and_actual = \
            self.prevalence * ((self._false_positives + self._true_positives) / self.total_observations)
        # probability that chance alone would lead the predicted and actual values to match, under the
        # assumption that both are selected randomly (i.e. implies independence) according to the observed
        # proportions (probability of independent events = P(A & B) == P(A) * P(B)
        pr_e = p_negative_prediction_and_actual + p_positive_prediction_and_actual
        return (pr_a - pr_e) / (1 - pr_e)

    @property
    def f1_score(self) -> float:
        return self.fbeta_score(beta=1)

    def fbeta_score(self, beta: float) -> float:
        """
        :param beta: The `beta` parameter determines the weight of precision in the combined score.
            `beta < 1` lends more weight to precision, while
            `beta > 1` favors recall
            (`beta -> 0` considers only precision, `beta -> inf` only recall).
            http://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html
        :return:
        """
        if self.positive_predictive_value is None or self.sensitivity is None or \
                (self.positive_predictive_value + self.sensitivity) == 0:
            return 0

        return (1 + (beta**2)) * (self.positive_predictive_value * self.sensitivity) / \
            (((beta**2) * self.positive_predictive_value) + self.sensitivity)

    @property
    def all_quality_metrics(self) -> dict:
        """
        :return: dictionary with all the score_names and associated values
        """
        return {'Kappa': self.kappa,
                'F1 Score': self.f1_score,
                'Two-Class Accuracy': self.accuracy,
                'Error Rate': self.error_rate,
                'True Positive Rate': self.sensitivity,
                'True Negative Rate': self.specificity,
                'False Positive Rate': self.false_positive_rate,
                'False Negative Rate': self.false_negative_rate,
                'Positive Predictive Value': self.positive_predictive_value,
                'Negative Predictive Value': self.negative_predictive_value,
                'Prevalence': self.prevalence,
                'No Information Rate': max(self.prevalence, 1-self.prevalence),  # i.e. largest class %
                'Total Observations': self.total_observations}

    def plot_all_quality_metrics(self, comparison_matrix: "TwoClassConfusionMatrix" = None):
        """
        Creates a plot that shows all of the quality score_names in this class.

        :param comparison_matrix: adds additional points to the plot for the score_names associated with the
            `comparison_matrix`; allows the user to compare two different confusion matrices (e.g. from two
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

        if comparison_matrix is not None:
            # convert diction to dataframe, without "Total Observations" which will fuck up axis
            # noinspection PyTypeChecker
            comparison_metrics_dataframe = pd.DataFrame.from_dict([comparison_matrix.all_quality_metrics])
            comparison_metrics_dataframe = comparison_metrics_dataframe[
                list(comparison_matrix.all_quality_metrics.keys())].drop(columns='Total Observations')
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
