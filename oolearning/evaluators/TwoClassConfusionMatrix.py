from typing import Union

import numpy as np

from oolearning.evaluators.ConfusionMatrix import ConfusionMatrix


class TwoClassConfusionMatrix(ConfusionMatrix):
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
    def sensitivity(self) -> Union[float, None]:
        """
        :return: a.k.a true positive rate
        """
        return None if self._actual_positives == 0 else self._true_positives / self._actual_positives

    @property
    def specificity(self) -> Union[float, None]:
        """
        :return: a.k.a false positive rate
        """
        return None if self._actual_negatives == 0 else self._true_negatives / self._actual_negatives

    @property
    def true_positive_rate(self) -> Union[float, None]:
        return self.sensitivity

    @property
    def true_negative_rate(self) -> Union[float, None]:
        return self.specificity

    @property
    def false_negative_rate(self) -> Union[float, None]:
        return None if self._actual_positives == 0 else self._false_negatives / self._actual_positives

    @property
    def false_positive_rate(self) -> Union[float, None]:
        return None if self._actual_negatives == 0 else self._false_positives / self._actual_negatives

    @property
    def accuracy(self) -> Union[float, None]:
        return None if self.total_observations == 0 else \
            (self._true_negatives + self._true_positives) / self.total_observations

    @property
    def error_rate(self) -> Union[float, None]:
        return None if self.total_observations == 0 else \
            (self._false_positives + self._false_negatives) / self.total_observations

    @property
    def positive_predictive_value(self) -> Union[float, None]:
        return None if (self._true_positives + self._false_positives) == 0 else \
            self._true_positives / (self._true_positives + self._false_positives)

    @property
    def negative_predictive_value(self) -> Union[float, None]:
        return None if (self._true_negatives + self._false_negatives) == 0 else \
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
    def f1_score(self) -> Union[float, None]:
        if self.positive_predictive_value is None or \
                self.sensitivity is None or \
                (self.positive_predictive_value + self.sensitivity) == 0:
            return None

        return 2 * (self.positive_predictive_value * self.sensitivity) / \
            (self.positive_predictive_value + self.sensitivity)

    @property
    def all_quality_metrics(self) -> dict:
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
