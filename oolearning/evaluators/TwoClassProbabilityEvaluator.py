import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from oolearning.converters.TwoClassConverterBase import TwoClassConverterBase
from oolearning.converters.TwoClassPrecisionRecallOptimizerConverter import \
    TwoClassPrecisionRecallOptimizerConverter
from oolearning.converters.TwoClassRocOptimizerConverter import TwoClassRocOptimizerConverter
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator


class TwoClassProbabilityEvaluator(TwoClassEvaluator):
    """
    Evaluates 2-class classification problems, where "probabilities" are supplied as well as a Converter (i.e.
      an object that encapsulates the logic to convert the probabilities to classes.
    """
    def __init__(self,
                 converter: TwoClassConverterBase):
        super().__init__(positive_class=converter.positive_class)
        self._converter = converter
        self._actual_classes = None
        self._predicted_probabilities = None

        self._auc_roc = None
        self._auc_precision_recall = None

        self._fpr = None
        self._tpr = None
        self._ideal_threshold_roc = None
        self._ppv = None
        self._ideal_threshold_ppv_tpr = None

    @property
    def auc_roc(self):
        return self._auc_roc

    @property
    def auc_precision_recall(self):
        return self._auc_precision_recall

    def evaluate(self,
                 actual_values: np.ndarray, predicted_values: pd.DataFrame):
        self._actual_classes = actual_values
        self._predicted_probabilities = predicted_values

        self._auc_roc = roc_auc_score(y_true=[1 if x == self._positive_class else 0 for x in actual_values],
                                      y_score=predicted_values[self._positive_class])
        # according to this (), average precision is same as auc of pr curve
        self._auc_precision_recall = average_precision_score(y_true=[1 if x == self._positive_class else 0
                                                                     for x in actual_values],
                                                             y_score=predicted_values[self._positive_class])

        predicted_classes = self._converter.convert(values=predicted_values)

        super().evaluate(actual_values=actual_values, predicted_values=predicted_classes)

    def plot_roc_curve(self):
        """
        :return: an ROC curve, indicating the point (threshold) that has the minimum distance to the
            upper left corner (i.e. a perfect predictor). If a threshold is specified in the
            class constructor, then that threshold is also annotated on the graph.
        """
        if self._fpr is None or self._tpr is None or self._ideal_threshold_roc is None:
            converter = TwoClassRocOptimizerConverter(actual_classes=self._actual_classes,
                                                      positive_class=self._converter.positive_class)
            converter.convert(values=self._predicted_probabilities)
            self._fpr = converter.false_positive_rates
            self._tpr = converter.true_positive_rates
            self._ideal_threshold_roc = converter.ideal_threshold

        self._create_curve(x_coordinates=self._fpr,
                           y_coordinates=self._tpr,
                           threshold=0.5,
                           ideal_threshold=self._ideal_threshold_roc,
                           title='ROC (AUC={0})'.format(round(self.auc_roc, 3)),
                           x_label='False Positive Rate (1 - True Negative Rate)',
                           y_label='True Positive Rate',
                           corner='Left')

    def plot_precision_recall_curve(self):
        """
        # TODO document
        """
        return self.plot_ppv_tpr_curve()

    def plot_ppv_tpr_curve(self):
        """
        # TODO document
        """
        if self._ppv is None or self._tpr is None or self._ideal_threshold_ppv_tpr is None:
            converter = TwoClassPrecisionRecallOptimizerConverter(actual_classes=self._actual_classes,
                                                                  positive_class=self._converter.positive_class)  # noqa
            converter.convert(values=self._predicted_probabilities)
            self._ppv = converter.positive_predictive_values
            self._tpr = converter.true_positive_rates
            self._ideal_threshold_ppv_tpr = converter.ideal_threshold

        self._create_curve(x_coordinates=self._tpr,
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

    @property
    def all_quality_metrics(self) -> dict:
        metrics = {'AUC ROC': self.auc_roc, 'AUC Precision/Recall': self.auc_precision_recall}
        metrics.update(super().all_quality_metrics)
        return metrics

    # def plot_all_quality_metrics(self):
    #     return self._confusion_matrix.plot_all_quality_metrics()
