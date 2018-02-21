import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from oolearning.evaluators.ConfusionMatrix import ConfusionMatrix
from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.evaluators.TwoClassEvaluator import TwoClassEvaluator


class MultiClassEvaluator(EvaluatorBase):
    def evaluate(self, actual_values: np.ndarray, predicted_values: object):
        pass

    def __init__(self, actual_classes: np.ndarray, predicted_classes: np.ndarray):
        """
        takes the actual/predicted values and creates a confusion confusion_matrix
        :param actual_classes:
        :param predicted_classes:
        :return: MultiClassEvaluator object
        """
        self._confusion_matrix = ConfusionMatrix(actual_classes=actual_classes,
                                                 predicted_classes=predicted_classes,
                                                 positive_class=None)

        self._total_observations = self.matrix.loc['Total', 'Total']
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
            bin_matrix = TwoClassEvaluator(actual_classes=np.array(actual_binary_classes),
                                           predicted_classes=np.array(predicted_binary_classes),
                                           positive_class=pos_label)
            if metric_dataframe is None:
                metric_dataframe = pd.DataFrame(columns=bin_matrix.all_quality_metrics.keys())

            metric_dataframe = metric_dataframe.append(bin_matrix.all_quality_metrics, ignore_index=True)
        metric_dataframe.index = unique_classes
        return metric_dataframe.drop(columns='Total Observations')

    @classmethod
    def from_probabilities(cls, actual_classes, predicted_probabilities: pd.DataFrame):
        """
        # TODO document
        chooses the class with the highest probability
        :param actual_classes:
        :param predicted_probabilities:
        :return:
        """
        # TODO: extract 'strategy'?   i.e. predicted_probabilities.idxmax(axis=1)
        predicted_classes = predicted_probabilities.idxmax(axis=1)
        return MultiClassEvaluator(actual_classes=actual_classes, predicted_classes=np.array(predicted_classes))

    @property
    def matrix(self) -> pd.DataFrame:
        return self._confusion_matrix.matrix

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def kappa(self):
        return self._kappa

    @property
    def no_information_rate(self):
        return self.matrix.drop(index='Total')['Total'].max() / self._total_observations

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
