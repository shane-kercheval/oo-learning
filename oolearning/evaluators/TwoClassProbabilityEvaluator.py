"""
Evaluates 2-class classification problems, where "probabilities" are supplied as well as a Converted (i.e.
    an object that encapsulates the logic to convert the probabilities to classes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase
from oolearning.converters.TwoClassPrecisionRecallOptimizerConverter import \
    TwoClassPrecisionRecallOptimizerConverter
from oolearning.converters.TwoClassRocOptimizerConverter import TwoClassRocOptimizerConverter
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator


class TwoClassProbabilityEvaluator(TwoClassEvaluator):
    def __init__(self,
                 converter: TwoClassConverterBase,
                 positive_class: object):
        super().__init__(positive_class=positive_class)
        self._converter = converter
        self._actual_classes = None
        self._predicted_probabilities = None

        self._auc = None

        self._fpr = None
        self._tpr = None
        self._ideal_threshold_roc = None
        self._ppv = None
        self._ideal_threshold_ppv_tpr = None

    @property
    def auc(self):
        return self._auc

    def evaluate(self,
                 actual_values: np.ndarray, predicted_values: pd.DataFrame):
        self._actual_classes = actual_values
        self._predicted_probabilities = predicted_values

        self._auc = roc_auc_score(y_true=[1 if x == self._positive_class else 0 for x in actual_values],
                                  y_score=predicted_values[self._positive_class])

        predicted_classes = self._converter.convert(predicted_probabilities=predicted_values,
                                                    positive_class=self._positive_class)

        super().evaluate(actual_values=actual_values, predicted_values=predicted_classes)

    def get_roc_curve(self):
        """
        :return: an ROC curve, indicating the point (threshold) that has the minimum distance to the
            upper left corner (i.e. a perfect predictor). If a threshold is specified in the
            class constructor, then that threshold is also annotated on the graph.
        """
        # from sklearn.metrics import roc_curve
        # from sklearn.metrics import roc_auc_score
        # fpr, tpr, thresholds = roc_curve(y_true=self._actual_classes, y_score=self._predicted_probabilities[self._positive_class])
        # roc_auc = roc_auc_score(y_true=self._actual_classes, y_score=self._predicted_probabilities[self._positive_class])
        # # Plot ROC curve
        # import matplotlib.pyplot as plt
        # plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
        # plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.xlabel('False Positive Rate or (1 - Specifity)')
        # plt.ylabel('True Positive Rate or (Sensitivity)')
        # plt.title('Receiver Operating Characteristic')
        # plt.legend(loc="lower right")

        if self._fpr is None or self._tpr is None or self._ideal_threshold_roc is None:
            converter = TwoClassRocOptimizerConverter(actual_classes=self._actual_classes)
            converter.convert(predicted_probabilities=self._predicted_probabilities,
                              positive_class=self._positive_class)
            self._fpr = converter.false_positive_rates
            self._tpr = converter.true_positive_rates
            self._ideal_threshold_roc = converter.ideal_threshold

        return self._create_curve(x_coordinates=self._fpr,
                                  y_coordinates=self._tpr,
                                  threshold=0.5,
                                  ideal_threshold=self._ideal_threshold_roc,
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
            converter = TwoClassPrecisionRecallOptimizerConverter(actual_classes=self._actual_classes)
            converter.convert(predicted_probabilities=self._predicted_probabilities,
                              positive_class=self._positive_class)
            self._ppv = converter.positive_predictive_values
            self._tpr = converter.true_positive_rates
            self._ideal_threshold_ppv_tpr = converter.ideal_threshold

        return self._create_curve(x_coordinates=self._tpr,
                                  y_coordinates=self._ppv,
                                  threshold=0.5,
                                  ideal_threshold=self._ideal_threshold_ppv_tpr,
                                  title='Positive Predictive Value vs. True Positive Rate',
                                  x_label='True Positive Rate',
                                  y_label='Positive Predictive Value',
                                  corner='Right')

    @staticmethod
    def _create_curve(x_coordinates, y_coordinates, threshold, ideal_threshold,
                      title, x_label, y_label, corner):
        index_of_ideal = int(round(ideal_threshold, 2) * 100)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.axis('square')
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.plot(x_coordinates, y_coordinates, drawstyle='steps-post')
        ax.annotate('Closest to Upper {0}\n(threshold={1})'.format(corner, str(round(ideal_threshold, 2))),
                    xy=(x_coordinates[index_of_ideal], y_coordinates[index_of_ideal] - 0.01),
                    xytext=(x_coordinates[index_of_ideal], y_coordinates[index_of_ideal] - 0.1),
                    arrowprops=dict(facecolor='black', headwidth=4, width=2, headlength=4),
                    horizontalalignment='left', verticalalignment='top')

        index_of_custom_threshold = int(round(threshold, 2) * 100)
        ax.annotate('Default threshold=' + str(round(threshold, 2)),
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
