import math
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from oolearning.evaluators.ClassificationEvaluator import ClassificationEvaluator
from oolearning.evaluators.ConfusionMatrix2C import ConfusionMatrix2C


# noinspection PyAbstractClass
class TwoClassEvaluator(ClassificationEvaluator):
    """
    Generic class to evaluate metrics for two-class (or 2 category) classifiers.
    `value` prop returns a generic 'value' calculation:
        ((true_negatives + true_positives) / total_observations)
    other Evaluators can inherit and override this functionality and return, e.g. Kappa/AUC calculations
    """

    def __init__(self,
                 positive_category,
                 negative_category,
                 use_probabilities: bool=True,
                 threshold: Union[float, None]=0.5):
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
        super().__init__(categories=[positive_category, negative_category],
                         use_probabilities=use_probabilities,
                         threshold=threshold)
        self._positive_category = positive_category
        self._negative_category = negative_category
        self._confusion_matrix = None
        self._fpr = None
        self._tpr = None
        self._ideal_threshold_roc = None
        self._ppv = None
        self._ideal_threshold_ppv_tpr = None

    @property
    def auc(self) -> float:
        pos_predictions = self._predicted_values.loc[:, self._positive_category]
        return roc_auc_score(y_true=self._actual_values, y_score=pos_predictions)

    @property
    def confusion_matrix(self) -> ConfusionMatrix2C:
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
    def _calculate_ppv_tpr_ideal_threshold(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        helper method to calculate false positive and true positive (sensitivity) rates for possible
        cutoff values between 0 and 1
        :return: array of false positive rates, true positive rates, and an ideal threshold (which
            is calculated by finding the the point on the ROC curve that has minimum distance to the
            upper left point (i.e. a perfect predictor)
        """
        def get_ppv_tpr(threshold):
            confusion_matrix = \
                ConfusionMatrix2C.from_predictions(actual_classes=self._actual_values,
                                                   predicted_classes=self._get_predicted_categories(
                                                     threshold=threshold),
                                                   positive_category=self._positive_category,
                                                   negative_category=self._negative_category)
            return confusion_matrix.positive_predictive_value, confusion_matrix.sensitivity

        potential_cutoff_values = np.arange(0.0, 1.01, 0.01)  # all possible cutoffs (precision of 2)
        ppv_tpr_tuple = [get_ppv_tpr(threshold=x) for x in potential_cutoff_values]  # list of rates
        # remove Nones caused by divide by zero for e.g. FPR/TPR
        ppv_tpr_tuple = [x for x in ppv_tpr_tuple if x[0] is not None and x[1] is not None]
        positive_predictive_values, true_positive_rates = zip(*ppv_tpr_tuple)

        # calculate distance from upper right (1, 1)
        def distance_formula(x, y):
            return math.sqrt((1 - x) ** 2 + (1 - y) ** 2)  # i.e. pythagorean theorem

        # calculate distance for each point on the ROC graph
        # precision (i.e. positive_predictive_value) on Y-axis and Recall (i.e. Sensitivity) on X-axis
        distances = [distance_formula(x=tpr, y=ppv) for ppv, tpr in ppv_tpr_tuple]
        # index is the index of cutoff value in the range that has the minimum distance from the upper left
        val, index = min((val, index) for (index, val) in enumerate(distances))
        ideal_threshold = potential_cutoff_values[index]

        return positive_predictive_values, true_positive_rates, ideal_threshold

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
                ConfusionMatrix2C.from_predictions(actual_classes=self._actual_values,
                                                   predicted_classes=self._get_predicted_categories(
                                                     threshold=threshold),
                                                   positive_category=self._positive_category,
                                                   negative_category=self._negative_category)
            return confusion_matrix.false_positive_rate, confusion_matrix.sensitivity

        potential_cutoff_values = np.arange(0.0, 1.01, 0.01)  # all possible cutoffs (precision of 2)
        fpr_tpr_tuple = [get_fpr_tpr(threshold=x) for x in potential_cutoff_values]  # list of rates
        # remove Nones caused by divide by zero for e.g. FPR/TPR
        fpr_tpr_tuple = [x for x in fpr_tpr_tuple if x[0] is not None and x[1] is not None]
        false_positive_rates, true_positive_rates = zip(*fpr_tpr_tuple)

        # calculate distance from upper left (0, 1)
        def distance_formula(x, y):
            return math.sqrt((0 - x) ** 2 + (1 - y) ** 2)  # i.e. pythagorean theorem

        # calculate distance for each point on the ROC graph
        distances = [distance_formula(x=fpr, y=tpr) for fpr, tpr in fpr_tpr_tuple]
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
        if self._fpr is None or self._tpr is None or self._ideal_threshold_roc is None:
            self._fpr, self._tpr, self._ideal_threshold_roc = self._calculate_fpr_tpr_ideal_threshold()

        return self._create_curve(x_coordinates=self._fpr,
                                  y_coordinates=self._tpr,
                                  threshold=self._threshold,
                                  ideal_threshold=self._ideal_threshold_roc,
                                  using_threshold_parameter=self._using_threshold_parameter,
                                  title='ROC (AUC={0})'.format(round(self.auc, 3)),
                                  x_label='False Positive Rate (1 - True Negative Rate)',
                                  y_label='True Positive Rate',
                                  corner='Left')

    def get_precision_recall_curve(self):
        """
        # TODO
        """
        return self.get_ppv_tpr_curve()

    def get_ppv_tpr_curve(self):
        """
        # TODO
        """
        if self._ppv is None or self._tpr is None or self._ideal_threshold_ppv_tpr is None:
            self._ppv, self._tpr, self._ideal_threshold_ppv_tpr = self._calculate_ppv_tpr_ideal_threshold()

        return self._create_curve(x_coordinates=self._tpr,
                                  y_coordinates=self._ppv,
                                  threshold=self._threshold,
                                  ideal_threshold=self._ideal_threshold_ppv_tpr,
                                  using_threshold_parameter=self._using_threshold_parameter,
                                  title='Positive Predictive Value vs. True Positive Rate',
                                  x_label='True Positive Rate',
                                  y_label='Positive Predictive Value',
                                  corner='Right')

    @staticmethod
    def _create_curve(x_coordinates, y_coordinates, threshold, ideal_threshold, using_threshold_parameter,
                      title, x_label, y_label, corner):
        index_of_ideal = int(round(ideal_threshold, 2) * 100)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('square')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.plot(x_coordinates, y_coordinates, drawstyle='steps-post')
        ax.annotate('Closest to Upper {0} (threshold={1})'.format(corner, str(round(ideal_threshold, 2))),
                    xy=(x_coordinates[index_of_ideal], y_coordinates[index_of_ideal] - 0.01),
                    xytext=(x_coordinates[index_of_ideal], y_coordinates[index_of_ideal] - 0.1),
                    arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
                    horizontalalignment='left', verticalalignment='top')

        if using_threshold_parameter:  # if we are using a custom threshold, then annotate it on the graph
            index_of_custom_threshold = int(round(threshold, 2) * 100)
            ax.annotate('Chosen threshold=' + str(round(threshold, 2)),
                        xy=(x_coordinates[index_of_custom_threshold],
                            y_coordinates[index_of_custom_threshold] - 0.01),
                        xytext=(x_coordinates[index_of_custom_threshold],
                                y_coordinates[index_of_custom_threshold] - 0.2),
                        arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
                        horizontalalignment=corner.lower(), verticalalignment='top')

        ax.set(**{'title': title,
                  'xlabel': x_label,
                  'ylabel': y_label})
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
                self._fpr, self._tpr, self._ideal_threshold_roc = self._calculate_fpr_tpr_ideal_threshold()
                self._threshold = self._ideal_threshold_roc

            # get predicted categories from the predicted probabilities and a specified threshold
            predicted_classes = self._get_predicted_categories(threshold=self._threshold)
        else:
            predicted_classes = predicted_values

        self._confusion_matrix = ConfusionMatrix2C.from_predictions(actual_classes=actual_values,
                                                                    predicted_classes=predicted_classes,
                                                                    positive_category=self._positive_category,
                                                                    negative_category=self._negative_category)

        return super().evaluate(actual_values=actual_values, predicted_values=predicted_classes)
