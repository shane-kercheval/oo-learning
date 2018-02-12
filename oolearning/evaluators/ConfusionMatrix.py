import numpy as np
import pandas as pd


class ConfusionMatrix:
    def __init__(self, confusion_matrix: pd.DataFrame):
        self._confusion_matrix = confusion_matrix

    @classmethod
    def from_probabilities(cls, actual_classes, predicted_probabilities: pd.DataFrame):
        predicted_classes = predicted_probabilities.idxmax(axis=1)
        return ConfusionMatrix.from_classes(actual_classes=actual_classes,
                                            predicted_classes=np.array(predicted_classes))

    @classmethod
    def from_classes(cls, actual_classes: np.ndarray, predicted_classes: np.ndarray) -> 'ConfusionMatrix':
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

        return cls(confusion_matrix=confusion_matrix)

    @property
    def matrix(self):
        return self._confusion_matrix

    def accuracy(self):
        pass








