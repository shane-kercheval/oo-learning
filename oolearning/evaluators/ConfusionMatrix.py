import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score

from oolearning.evaluators.ConfusionMatrix2C import ConfusionMatrix2C


class ConfusionMatrix:
    def __init__(self, actual_classes: np.ndarray, predicted_classes: np.ndarray):
        """
        takes the actual/predicted values and creates a confusion confusion_matrix
        :param actual_classes:
        :param predicted_classes:
        :return: ConfusionMatrix object
        """
        assert len(actual_classes) == len(predicted_classes)
        # ensure that all the unique predicted values are in the actual values
        assert all([x in np.unique(actual_classes) for x in np.unique(predicted_classes)])

        confusion_matrix = pd.crosstab(actual_classes, predicted_classes, margins=True)
        expected_indexes = list(confusion_matrix.index.values)
        new_indexes = [x if x != 'All' else 'Total' for x in expected_indexes]
        confusion_matrix.index = new_indexes
        confusion_matrix.index.name = 'actual'

        confusion_matrix = confusion_matrix.reindex(columns=expected_indexes)
        confusion_matrix.columns = new_indexes
        confusion_matrix.columns.name = 'predicted'
        # NaN values could be found when e.g. predictions are all from single class
        confusion_matrix.fillna(value=0, inplace=True)

        # self._actual_classes = actual_classes
        # self._predicted_classes = predicted_classes
        self._confusion_matrix = confusion_matrix
        self._total_observations = confusion_matrix.loc['Total', 'Total']

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
            bin_matrix = ConfusionMatrix2C.from_classes(actual_classes=np.array(actual_binary_classes),
                                                        predicted_classes=np.array(predicted_binary_classes),
                                                        positive_category=pos_label,
                                                        negative_category=neg_label)
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
        predicted_classes = predicted_probabilities.idxmax(axis=1)
        return ConfusionMatrix(actual_classes=actual_classes, predicted_classes=np.array(predicted_classes))

    @property
    def matrix(self):
        return self._confusion_matrix

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
