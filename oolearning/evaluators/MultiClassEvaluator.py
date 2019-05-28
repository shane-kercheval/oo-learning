import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from oolearning.converters.ContinuousToClassConverterBase import ContinuousToClassConverterBase
from oolearning.evaluators.ConfusionMatrix import ConfusionMatrix
from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator


class MultiClassEvaluator(EvaluatorBase):
    """
    Evaluates models for multi-class classification problems.
    """
    def __init__(self,
                 converter: ContinuousToClassConverterBase,
                 actual_classes: np.ndarray = None,
                 predicted_classes: np.ndarray = None):
        """
        :param converter: A Converter that converts a predictions DataFrame to the predicted classes.
        :param actual_classes:
        :param predicted_classes:
        """
        super().__init__()
        self._converter = converter

        if converter is not None:
            self._confusion_matrix = None
            self._total_observations = None
            self._kappa = None
            self._accuracy = None
            self._metrics_per_class = None
        else:
            if actual_classes is None or predicted_classes is None:
                raise ValueError('must pass in both `actual_classes` and `predicted_classes` if no converter')
            self.set_instance_values_from_values(actual_classes=actual_classes,
                                                 predicted_classes=predicted_classes)

    def __str__(self):
        val = super().__str__()
        val += "\n\nConfusion Matrix\n----------------\n\n" + self.matrix.to_string()

        return val

    def evaluate(self, actual_values: np.ndarray, predicted_values: object):

        # noinspection PyTypeChecker
        predicted_classes = self._converter.convert(values=predicted_values)
        self.set_instance_values_from_values(actual_classes=actual_values,
                                             predicted_classes=predicted_classes)

    def set_instance_values_from_values(self, actual_classes: np.ndarray, predicted_classes: np.ndarray):
        self._confusion_matrix = ConfusionMatrix(actual_classes=actual_classes,
                                                 predicted_classes=predicted_classes)

        self._total_observations = self._confusion_matrix.matrix.loc['Total', 'Total']
        self._kappa = cohen_kappa_score(y1=actual_classes, y2=predicted_classes)
        self._accuracy = accuracy_score(y_true=actual_classes, y_pred=predicted_classes)
        self._metrics_per_class = self.create_metrics_per_class(actual_classes=actual_classes,
                                                                predicted_classes=predicted_classes)

    @staticmethod
    def create_metrics_per_class(actual_classes: np.ndarray, predicted_classes: np.ndarray):
        # for each class, let's treat it as if it were the positive class and all others were the negative
        # then create a confusion matrix; and build up a DataFrame
        unique_classes = np.unique(actual_classes)
        pos_label = 'pos'
        neg_label = 'neg'
        metric_dataframe = None

        for target in unique_classes:
            actual_binary_classes = [pos_label if x == target else neg_label for x in actual_classes]
            predicted_binary_classes = [pos_label if x == target else neg_label for x in predicted_classes]
            bin_matrix = TwoClassEvaluator(positive_class=pos_label)
            bin_matrix.evaluate(actual_values=np.array(actual_binary_classes),
                                predicted_values=np.array(predicted_binary_classes))

            if metric_dataframe is None:
                metric_dataframe = pd.DataFrame(columns=bin_matrix.all_quality_metrics.keys())

            metric_dataframe = metric_dataframe.append(bin_matrix.all_quality_metrics, ignore_index=True)
        metric_dataframe.index = unique_classes
        return metric_dataframe.drop(columns='Total Observations')

    @classmethod
    def from_classes(cls, actual_classes: np.ndarray, predicted_classes: np.ndarray) -> 'MultiClassEvaluator':
        # noinspection PyTypeChecker
        return MultiClassEvaluator(converter=None,
                                   actual_classes=actual_classes,
                                   predicted_classes=predicted_classes)

    @property
    def matrix(self) -> pd.DataFrame:
        return self._confusion_matrix.matrix

    @property
    def confusion_matrix(self) -> ConfusionMatrix:
        return self._confusion_matrix

    @property
    def total_observations(self) -> int:
        return self._confusion_matrix.total_observations

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def kappa(self):
        return self._kappa

    @property
    def no_information_rate(self):
        return self._confusion_matrix.matrix.drop(index='Total')['Total'].max() / self._total_observations

    @property
    def metrics_per_class(self):
        return self._metrics_per_class

    @property
    def all_quality_metrics(self) -> dict:
        return {'Kappa': self.kappa,
                'Accuracy': self.accuracy,
                'Error Rate': 1 - self._accuracy,
                'No Information Rate': self.no_information_rate,  # i.e. largest class %
                'Total Observations': self._total_observations}
