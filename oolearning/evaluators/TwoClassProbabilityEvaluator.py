import matplotlib.pyplot as plt
import matplotlib.lines
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
        plt.tight_layout()

    def plot_precision_recall_curve(self):
        """
        # TODO document
        """
        self.plot_ppv_tpr_curve()

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

        plt.tight_layout()

    def plot_calibration(self):
        """
        :return: calibration plot. Predicted probabilities are matched with the actual class and binned by
            the prediction in intervals of 0.1. i.e. all probabilities/classes that have a prediction between
            0 to 0.1 are grouped together, > 0.1 <= 0.2 are grouped together, and so on. For each group, the
            percent of positive classes found is calculated. For example, in the group that has predicted
            probabilities between 0 and 0.1, we would expect the average probability to be 0.05, and therefore
            we would expect about 0.05 (i.e. 5%) of the group to be a positive class. The percentage of
            positive classes for each bin is plotted. If the points fall along a 45 degree line, the model
            has produced well-calibrated probabilities.
        """
        calibration_data = pd.concat([self._predicted_probabilities[self._positive_class],
                                      self._actual_classes], axis=1)
        calibration_data.columns = ['probabilities', 'actual_classes']
        bin_labels = ['[0, 0.1]', '(0.1, 0.2]', '(0.2, 0.3]', '(0.3, 0.4]', '(0.4, 0.5]', '(0.5, 0.6]',
                      '(0.6, 0.7]', '(0.7, 0.8]', '(0.8, 0.9]', '(0.9, 1.0]']
        # .cut maintains distribution shape
        bins = pd.cut(calibration_data.probabilities,
                      bins=np.arange(0.0, 1.1, 0.1),
                      include_lowest=True,
                      labels=bin_labels)
        calibration_data['bins'] = bins
        # calibration_data.bins.value_counts(ascending=True)
        # calibration_data.head()
        # calibration_data.sort_values(['bins', 'actual_classes'])

        def calibration_grouping(x):
            # noinspection PyTypeChecker
            number_positive_events = sum(x.actual_classes == self._positive_class)
            total_observations = len(x.actual_classes)
            d = {'Positive Events Found': number_positive_events,
                 'Total Observations': total_observations,
                 'Actual Calibration': 0 if total_observations == 0
                                         else number_positive_events / total_observations}
            return pd.Series(d, index=['Positive Events Found', 'Total Observations', 'Actual Calibration'])

        calibration_group_data = calibration_data.groupby('bins').apply(calibration_grouping)
        calibration_group_data['Perfect Calibration'] = np.arange(0.05, 1.05, 0.10)

        calibration_group_data[['Actual Calibration', 'Perfect Calibration']].plot(yticks=np.arange(0.0, 1.1, 0.1))  # noqa
        ax = plt.gca()
        ax.set_xticks(np.arange(len(bin_labels)))
        ax.set_xticklabels(labels=bin_labels, rotation=20, ha='right', size=9)
        ax.set_xlim(-0.5, len(bin_labels) - 0.5)
        ax.figure.set_size_inches(8, 8)
        ax.grid(which='major', alpha=0.1)
        for index in range(10):
            text = '({}/{} = {:.1%})'.format(calibration_group_data.iloc[index]['Positive Events Found'],
                                             calibration_group_data.iloc[index]['Total Observations'],
                                             calibration_group_data.iloc[index]['Actual Calibration'])
            ax.annotate(text,
                        xy=(index+0.15, calibration_group_data.iloc[index]['Actual Calibration'] - 0.005),
                        size=7)
        ax.scatter(x=np.arange(len(bin_labels)), y=calibration_group_data['Actual Calibration'].values, s=10)
        ax.set(**{'title': 'Calibration Chart',
                  'xlabel': 'Binned Probabilities',
                  'ylabel': 'Percent of Positive (Actual) Events in Bin'})
        plt.tight_layout()

    def plot_predicted_probability_hist(self):
        calibration_data = pd.concat([self._predicted_probabilities[self._positive_class],
                                      self._actual_classes], axis=1)
        calibration_data.columns = ['Predicted Probabilities', 'Actual Classes']

        calibration_data['Predicted Probabilities'].hist(by=calibration_data['Actual Classes'],
                                                         bins=20)
        axes = plt.gcf().get_axes()
        for ax in axes:
            ax.set_xticks(np.arange(0.0, 1.1, 0.1))
            ax.set(**{'xlabel': 'Predicted Probability (Positive Event)',
                      'ylabel': 'Count'})
        ax = plt.gca()
        ax.figure.set_size_inches(10, 6)
        plt.suptitle('Histogram of Predicted Probabilities, by Actual Outcome', fontsize=12)
        plt.tight_layout()

    @staticmethod
    def _create_gain_lift_data(predicted_probabilities, actual_classes, positive_class):

        raw_data = pd.concat([predicted_probabilities[positive_class],
                              actual_classes], axis=1)
        raw_data.columns = ['probabilities', 'actual_classes']
        raw_data.sort_values(['probabilities'], ascending=False)

        # .qcut gets percentiles
        bins = pd.qcut(x=raw_data['probabilities'], q=10, labels=list(range(100, 0, -10)))

        raw_data['percentiles'] = bins
        # probabilities_classes.sort_values('probabilities')

        def gain_grouping(x):
            # noinspection PyTypeChecker
            number_positive_events = sum(x.actual_classes == positive_class)
            d = {'Number of Observations': len(x.actual_classes),
                 'Number of Positive Events': number_positive_events
                 }
            return pd.Series(d, index=['Number of Observations', 'Number of Positive Events'])

        # noinspection PyTypeChecker
        number_of_positive_events = sum(actual_classes == positive_class)
        gain_lift_data = raw_data.groupby('percentiles').apply(gain_grouping)

        temp = pd.DataFrame({'Number of Observations': 0, 'Number of Positive Events': 0}, index=[0],)
        temp.index.names = ['percentiles']
        gain_lift_data = pd.concat([gain_lift_data, temp])
        gain_lift_data.sort_index(ascending=True, inplace=True)
        gain_lift_data['Cumulative Observations'] = gain_lift_data['Number of Observations'].cumsum()
        gain_lift_data['Cumulative Positive Events'] = gain_lift_data['Number of Positive Events'].cumsum()
        gain_lift_data['Percentage of Positive Events'] = gain_lift_data['Cumulative Positive Events'] / \
                                                           number_of_positive_events
        gain_lift_data['Random Gain'] = gain_lift_data.index.values
        gain_lift_data['Model Gain'] = gain_lift_data['Percentage of Positive Events'] * 100

        total_observations = len(actual_classes)

        gain_lift_data['Model Lift'] = (gain_lift_data['Model Gain'] / 100) / \
                                  (gain_lift_data['Cumulative Observations'] / total_observations)
        return gain_lift_data

    def plot_gain_chart(self):
        """
        :return: A Gain Chart shows the % of positive events we have 'captured' i.e. located by looking at the
            top x% of population of predictions such that the highest predictions are looked at first.
            So we can say we've captured X% of all the positive events by searching Y% of highest predictions.

            For example, if X (and the x-axis) is 20% and Y (and the y-axis) is 83%, and we are predicting
            the probability that a customer will purchase a widget, then we can say something like:
            "In the case of propensity to buy, we can say we can identify and target 83% of the customers
            who are likely to buy the product by target 20% of total customers (in the population we are
            looking at)". This gives us an idea of effort vs payoff in sales/marketing/etc. activities.
        """
        # noinspection PyTypeChecker
        gain_lift_group_data = self._create_gain_lift_data(predicted_probabilities=self._predicted_probabilities,  # noqa
                                                           actual_classes=self._actual_classes,
                                                           positive_class=self._positive_class)
        # get the percent of positive class proportion, call it X%
        # we want plot X%, such that ideally we've found i.e. predicted 100% of the positive events after
        # only searching X% of the population. In other words, if we ordered our predictions by probability
        # (in descending order) and there were e.g. 40 positive events out of a population of 100, then
        # a model with perfect gain would have all 40 events in the top 40 spots (after ordering by
        # probability)
        # noinspection PyTypeChecker
        number_of_positive_events = sum(self._actual_classes == self._positive_class)
        positive_class_proportion = number_of_positive_events / len(self._actual_classes) * 100
        perfect_gain_x_values = sorted(list(range(0, 110, 10)) + [positive_class_proportion])
        perfect_gain_y_values = np.array(perfect_gain_x_values) / positive_class_proportion * 100
        perfect_gain_y_values = [min(x, 100) for x in perfect_gain_y_values]

        #####################################
        gain_lift_group_data[['Model Gain', 'Random Gain']].plot(xticks=range(0, 110, 10),
                                                                 yticks=range(0, 110, 10))

        x_tick_labels = ['{}%'.format(x) for x in range(0, 110, 10)]
        ax = plt.gca()
        ax.figure.set_size_inches(8, 8)
        ax.set_xticklabels(labels=x_tick_labels, size=9)
        ax.set_yticklabels(labels=['{}%'.format(x) for x in range(0, 110, 10)], size=9)
        ax.set_xlim(-5, 105)
        ax.grid(which='major', alpha=10)
        for index in range(10, 110, 10):
            ax.annotate('{}%'.format(round(gain_lift_group_data.loc[index]['Model Gain'], 1)),
                        xy=(index + 1, gain_lift_group_data.loc[index]['Model Gain'] - 1),
                        size=10)

        ax.scatter(x=range(0, 110, 10), y=gain_lift_group_data['Model Gain'].values, s=10)
        line = matplotlib.lines.Line2D(perfect_gain_x_values,
                                       perfect_gain_y_values,
                                       color='green')
        ax.scatter(x=positive_class_proportion, y=100, s=10, color='green')
        ax.annotate('{}%'.format(round(positive_class_proportion, 1)),
                    xy=(positive_class_proportion + 1, 100 - 2.5),
                    size=10)
        ax.add_line(line)
        ax.legend(labels=['Model Gain', 'Random Gain', 'Perfect Gain'])
        ax.set(**{'title': 'Gain Chart',
                  'xlabel': 'Percentile (lower percentiles contain higher predicted probabilities)',
                  'ylabel': '% of Positive Events Captured'})
        plt.tight_layout()

    def plot_lift_chart(self):
        """
        :return: Lift chart shows when we are selecting the top X (x-axis) percent of the predictions such
            that the highest predictions are looked at first, we can expected Y-times (y-axis) the total
            number of positive events found than by randomly selecting X% of the data.
        """
        gain_lift_group_data = self._create_gain_lift_data(predicted_probabilities=self._predicted_probabilities,  # noqa
                                                           actual_classes=self._actual_classes,
                                                           positive_class=self._positive_class)
        # get the percent of positive class proportion, call it X%
        # we want plot X%, such that ideally we've found i.e. predicted 100% of the positive events after
        # only searching X% of the population. In other words, if we ordered our predictions by probability
        # (in descending order) and there were e.g. 40 positive events out of a population of 100, then
        # a model with perfect gain would have all 40 events in the top 40 spots (after ordering by
        # probability)
        # noinspection PyTypeChecker
        #####################################
        lift_data = gain_lift_group_data.loc[range(10, 110, 10)][['Model Lift']]
        lift_data['Random Lift'] = [1 for _ in range(10, 110, 10)]
        lift_data.plot(xticks=range(0, 110, 10))
        ax = plt.gca()
        ax.figure.set_size_inches(8, 8)
        ax.set_xticklabels(labels=['{}%'.format(x) for x in range(0, 110, 10)], size=9)
        ax.set_xlim(0, 105)
        ax.grid(which='major', alpha=10)
        for index in range(10, 110, 10):
            ax.annotate(round(gain_lift_group_data.loc[index]['Model Lift'], 1),
                        xy=(index - 1.5, gain_lift_group_data.loc[index]['Model Lift'] + 0.02),
                        size=10)

        ax.scatter(x=range(0, 110, 10), y=gain_lift_group_data['Model Lift'].values, s=10)
        ax.set(**{'title': 'Lift Chart',
                  'xlabel': 'Percentile (lower percentiles contain higher predicted probabilities)',
                  'ylabel': 'Lift'})
        plt.tight_layout()

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
