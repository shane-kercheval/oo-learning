from typing import Tuple, Union, Callable
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from oolearning.evaluators.ClassificationEvaluator import ClassificationEvaluator
from oolearning.evaluators.ConfusionMatrix import ConfusionMatrix


# noinspection PyAbstractClass
class TwoClassEvaluator(ClassificationEvaluator):
    """
    Generic class to evaluate metrics for two-class (or 2 category) classifiers.
    `value` prop returns a generic 'value' calculation:
        ((true_negatives + true_positives) / total_observations)
    other Evaluators can inherit and override this functionality and return, e.g. Kappa/AUC calculations
    """

    def __init__(self,
                 better_than: Callable[[float, float], bool],
                 positive_category,
                 negative_category,
                 use_probabilities: bool=True,
                 threshold: float=0.5):
        """
        :param positive_category: value of the positive category
        :param negative_category: value of the negative category
        :param use_probabilities: rather than an predicted_values as a categorical prediction, we might have
            probabilities that are passed in, in which case we would want to set this value to True
        :param threshold: if threshold is specified in constructor, that value is used to create the
            confusion confusion_matrix; only used if use_probabilities is True;
            otherwise, if `use_probabilities` is True and `threshold` is None, the ideal threshold will be
                calculated based on the the point (and corresponding threshold) that is closest to the upper
                left corner of the ROC graph.
        # TODO: document performance hit when needed to calculate the ideal threshold (now caching value)
        """
        super().__init__(better_than=better_than,
                         categories=[positive_category, negative_category],
                         use_probabilities=use_probabilities,
                         threshold=threshold)
        self._positive_category = positive_category
        self._negative_category = negative_category
        self._confusion_matrix = None
        self._fpr = None
        self._tpr = None
        self._ideal_threshold = None

    @property
    def auc(self) -> float:
        pos_predictions = self._predicted_values.loc[:, self._positive_category]
        return roc_auc_score(y_true=self._actual_values, y_score=pos_predictions)

    @property
    def confusion_matrix(self) -> pd.DataFrame:
        return self._confusion_matrix

    @property
    def threshold(self) -> float:
        """
        :return: if threshold was specified in constructor, that value is used/returned, otherwise
            it is calculated
        """
        return self._threshold

    def _get_predicted_categories(self, threshold) -> np.ndarray:
        """
        helper method to generate categorical predictions from predicted probabilities based on a
        specified threshold
        :param threshold: cutoff value; probabilities that are greater than the threshold will be
            categorized as the positive_category
        :return: array of categorical predictions
        """
        pos_predictions = self._predicted_values.loc[:, self._positive_category]
        return np.where(pos_predictions > threshold, self._positive_category, self._negative_category)

    # TODO: cache this value
    def _calculate_fpr_tpr_ideal_threshold(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        helper method to calculate false positive and true positive (sensitivity) rates for possible
        cutoff values between 0 and 1
        :return: array of false positive rates, true positive rates, and an ideal threshold (which
            is calculated by finding the the point on the ROC curve that has minimum distance to the
            upper left point (i.e. a perfect predictor)
        """
        def get_fpr_tpr(threshold):
            confusion_matrix = \
                ConfusionMatrix.from_predictions(actual_values=self._actual_values,
                                                 predicted_values=self._get_predicted_categories(
                                                     threshold=threshold),
                                                 positive_category=self._positive_category,
                                                 negative_category=self._negative_category)
            return confusion_matrix.false_positive_rate, confusion_matrix.sensitivity

        potential_cutoff_values = np.arange(0.0, 1.01, 0.01)  # all possible cutoffs (precision of 2)
        fpr_tpr_tuple = [get_fpr_tpr(threshold=x) for x in potential_cutoff_values]  # list of rates
        false_positive_rates, true_positive_rates = zip(*fpr_tpr_tuple)

        # calculate distance from upper left (0, 1)
        def distance_formula(x, y):
            return math.sqrt((0 - x) ** 2 + (1 - y) ** 2)  # i.e. pythagorean theorem

        # calculate distance for each point on the ROC graph
        distances = [distance_formula(fpr, tpr) for fpr, tpr in fpr_tpr_tuple]
        # index is the index of cutoff value in the range that has the minimum distance from the upper left
        val, index = min((val, index) for (index, val) in enumerate(distances))
        ideal_threshold = potential_cutoff_values[index]

        return false_positive_rates, true_positive_rates, ideal_threshold

    def get_roc_curve(self):
        """
        :return: an ROC curve, indicating the point (threshold) that has the minimum distance to the
            upper left corner (i.e. a perfect predictor). If a threshold is specified in the
            class constructor, then that threshold is also annotated on the graph.
        """
        if self._fpr is None or self._tpr is None or self._ideal_threshold is None:
            self._fpr, self._tpr, self._ideal_threshold = self._calculate_fpr_tpr_ideal_threshold()

        index_of_ideal = int(round(self._ideal_threshold, 2) * 100)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('square')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.plot(self._fpr, self._tpr)
        ax.annotate('Closest to Upper Left (threshold=' + str(round(self._ideal_threshold, 2)) + ')',
                    xy=(self._fpr[index_of_ideal], self._tpr[index_of_ideal] - 0.01),
                    xytext=(self._fpr[index_of_ideal], self._tpr[index_of_ideal] - 0.1),
                    arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
                    horizontalalignment='left', verticalalignment='top')

        if self._custom_threshold:  # if we are using a custom threshold, then annotate it on the graph
            index_of_custom_threshold = int(round(self._threshold, 2) * 100)
            ax.annotate('Chosen threshold=' + str(round(self._threshold, 2)) + ')',
                        xy=(self._fpr[index_of_custom_threshold],
                            self._tpr[index_of_custom_threshold] - 0.01), xytext=(
                    self._fpr[index_of_custom_threshold], self._tpr[index_of_custom_threshold] - 0.2),
                        arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
                        horizontalalignment='left', verticalalignment='top')

        ax.set(**{'title': 'ROC',
                  'xlabel': 'False Positive Rate (1 - Specificity)',
                  'ylabel': 'Sensitivity'})
        return fig

    def evaluate(self, actual_values: np.ndarray, predicted_values: Union[np.ndarray, pd.DataFrame]):
        """
        overriding `evaluate` in order to calculate threshold and categorical predictions if necessary
        # TODO: performance hit when needed to calculate the ideal threshold
        :param actual_values:
        :param predicted_values: predicted values should be an array when the "actual" i.e. class values are
            used and a DataFrame containing the predictions of each class when probabilities are used (i.e. a
            column for each class category; which is what is returned from the ModelWrappers)
        :return:
        """
        self._actual_values = actual_values
        self._predicted_values = predicted_values

        if self._use_probabilities is True:
            assert predicted_values.shape[0] == len(actual_values)
            assert predicted_values.shape[1] == 2  # one for each class
            assert self._positive_category in predicted_values.columns.values
            assert self._negative_category in predicted_values.columns.values

            if self._threshold is None:  # calculate threshold
                self._fpr, self._tpr, self._ideal_threshold = self._calculate_fpr_tpr_ideal_threshold()
                self._threshold = self._ideal_threshold

            # get predicted categories from the predicted probabilities and a specified threshold
            predicted_values = self._get_predicted_categories(threshold=self._threshold)

        self._confusion_matrix = ConfusionMatrix.from_predictions(actual_values=actual_values,
                                                                  predicted_values=predicted_values,
                                                                  positive_category=self._positive_category,
                                                                  negative_category=self._negative_category)

        return super().evaluate(actual_values=actual_values, predicted_values=predicted_values)
