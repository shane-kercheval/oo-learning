from typing import Union

import numpy as np
import pandas as pd


class ConfusionMatrix2C:
    """
    Class representing a confusion confusion_matrix for two-class (or 2 category) classifiers.
    """

    def __init__(self, confusion_matrix: pd.DataFrame, positive_category, negative_category):
        category_list = [negative_category, positive_category]

        self._confusion_matrix = confusion_matrix
        self._positive_category = positive_category
        self._negative_category = negative_category

        self._total_observations = confusion_matrix.loc['Total', 'Total']
        self._actual_positives = confusion_matrix.loc[self._positive_category][category_list].sum()
        self._actual_negatives = confusion_matrix.loc[self._negative_category][category_list].sum()

        self._true_positives = confusion_matrix.loc[self._positive_category, self._positive_category]
        self._true_negatives = confusion_matrix.loc[self._negative_category, self._negative_category]
        self._false_positives = confusion_matrix.loc[self._negative_category, self._positive_category]
        self._false_negatives = confusion_matrix.loc[self._positive_category, self._negative_category]

    @classmethod
    def from_values(cls,
                    true_positives: int,
                    true_negatives: int,
                    false_positives: int,
                    false_negatives: int,
                    positive_category='pos',
                    negative_category='neg') -> 'ConfusionMatrix2C':
        """
        takes the individual values/frequencies from a 'cross tabulation' of actual vs predicted values
        :param true_positives:
        :param true_negatives:
        :param false_positives:
        :param false_negatives:
        :param positive_category:
        :param negative_category:
        :return: ConfusionMatrix2C object
        """
        unique_categories = [negative_category, positive_category]
        confusion_matrix = pd.DataFrame({negative_category: [true_negatives, false_negatives],
                                         positive_category: [false_positives, true_positives]},
                                        index=unique_categories, columns=unique_categories)
        confusion_matrix['Total'] = confusion_matrix.sum(axis=1)
        confusion_matrix.loc['Total'] = confusion_matrix.sum(axis=0)

        confusion_matrix.index.name = 'actual'
        confusion_matrix.columns.name = 'predicted'

        return cls(confusion_matrix=confusion_matrix,
                   positive_category=positive_category,
                   negative_category=negative_category)

    @classmethod
    def from_predictions(cls,
                         actual_classes: np.ndarray,
                         predicted_classes: np.ndarray,
                         positive_category,
                         negative_category) -> \
            'ConfusionMatrix2C':
        """
        takes the actual/predicted values and creates a confusion confusion_matrix
        :param actual_classes:
        :param predicted_classes:
        :param positive_category:
        :param negative_category:
        :return: ConfusionMatrix2C object
        """
        assert len(actual_classes) == len(predicted_classes)
        # ensure that all the unique predicted values are in the actual values (e.g. ensure that we aren't
        # using predicted probabilities /etc.; this could cause problems if all the actual values are 1 class;
        # but I would imagine this is an extreme corner case
        assert all([x in np.unique(actual_classes) for x in np.unique(predicted_classes)])

        expected_indexes = [negative_category, positive_category] + ['All']
        new_indexes = [negative_category, positive_category] + ['Total']

        # |                  | Predicted Negative | Predicted Positive |
        # | ---------------- | ------------------ | ------------------ |
        # | Actual Negative  | True Negative      | False Positive     |
        # | Actual Positive  | False Negative     | True Positive      |
        confusion_matrix = pd.crosstab(actual_classes, predicted_classes, margins=True)
        confusion_matrix = confusion_matrix.reindex(index=expected_indexes)
        confusion_matrix.index = new_indexes
        confusion_matrix.index.name = 'actual'

        confusion_matrix = confusion_matrix.reindex(columns=expected_indexes)
        confusion_matrix.columns = new_indexes
        confusion_matrix.columns.name = 'predicted'
        # NaN values could be found when e.g. predictions are all from single class
        confusion_matrix.fillna(value=0, inplace=True)

        return cls(confusion_matrix=confusion_matrix,
                   positive_category=positive_category,
                   negative_category=negative_category)

    @property
    def matrix(self) -> pd.DataFrame:
        """
        :return: pandas DataFrame with the structure of:
        # |                  | Predicted Negative | Predicted Positive |
        # | ---------------- | ------------------ | ------------------ |
        # | Actual Negative  | True Negative      | False Positive     |
        # | Actual Positive  | False Negative     | True Positive      |
        """
        return self._confusion_matrix

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
    def two_class_accuracy(self) -> Union[float, None]:
        return None if self._total_observations == 0 else \
            (self._true_negatives + self._true_positives) / self._total_observations

    @property
    def error_rate(self) -> Union[float, None]:
        return None if self._total_observations == 0 else \
            (self._false_positives + self._false_negatives) / self._total_observations

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
        return None if self._total_observations == 0 else \
            (self._true_positives + self._false_negatives) / self._total_observations

    @property
    def kappa(self) -> Union[float, None]:
        if self._total_observations == 0 or \
                ((self._true_negatives + self._false_negatives) / self._total_observations) == 0:
            return None
        # proportion of the actual agreements
        # add the proportion of all instances where the predicted type and actual type agree
        pr_a = (self._true_negatives + self._true_positives) / self._total_observations
        # probability of both predicted and actual being negative
        p_negative_prediction_and_actual = \
            ((self._true_negatives + self._false_positives) / self._total_observations) * \
            ((self._true_negatives + self._false_negatives) / self._total_observations)
        # probability of both predicted and actual being positive
        p_positive_prediction_and_actual = \
            self.prevalence * ((self._false_positives + self._true_positives) / self._total_observations)
        # probability that chance alone would lead the predicted and actual values to match, under the
        # assumption that both are selected randomly (i.e. implies independence) according to the observed
        # proportions (probability of independent events = P(A & B) == P(A) * P(B)
        pr_e = p_negative_prediction_and_actual + p_positive_prediction_and_actual
        return (pr_a - pr_e) / (1 - pr_e)

    @property
    def f1_score(self) -> float:
        return 2 * (self.positive_predictive_value * self.sensitivity) / \
               (self.positive_predictive_value + self.sensitivity)

    @property
    def all_quality_metrics(self) -> dict:
        return {'Kappa': self.kappa,
                'F1 Score': self.f1_score,
                'Two-Class Accuracy': self.two_class_accuracy,
                'Error Rate': self.error_rate,
                'True Positive Rate': self.sensitivity,
                'True Negative Rate': self.specificity,
                'False Positive Rate': self.false_positive_rate,
                'False Negative Rate': self.false_negative_rate,
                'Positive Predictive Value': self.positive_predictive_value,
                'Negative Predictive Value': self.negative_predictive_value,
                'Prevalence': self.prevalence,
                'No Information Rate': self._actual_negatives / self._total_observations,
                'Total Observations': self._total_observations}