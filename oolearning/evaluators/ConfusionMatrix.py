from typing import List

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt


class ConfusionMatrix:
    """
    Class that represents a confusion matrix for any number of classes.
    """
    def __init__(self,
                 actual_classes: np.ndarray,
                 predicted_classes: np.ndarray,
                 class_order: List[object]=None):
        assert len(actual_classes) == len(predicted_classes)
        # ensure that all the unique predicted values are in the actual values
        assert all([x in np.unique(actual_classes) for x in np.unique(predicted_classes)])

        confusion_matrix = pd.crosstab(pd.Series(list(actual_classes)),
                                       pd.Series(list(predicted_classes)),
                                       margins=True)

        if class_order is not None:
            expected_indexes = class_order + ['All']
        else:
            expected_indexes = list(confusion_matrix.index.values)

        new_indexes = [x if x != 'All' else 'Total' for x in expected_indexes]

        # change row order/names
        confusion_matrix = confusion_matrix.reindex(index=expected_indexes)
        confusion_matrix.index = new_indexes
        confusion_matrix.index.name = 'actual'
        # change column order/names
        confusion_matrix = confusion_matrix.reindex(columns=expected_indexes)
        confusion_matrix.columns = new_indexes
        confusion_matrix.columns.name = 'predicted'
        # NaN values could be found when e.g. predictions are all from single class
        confusion_matrix.fillna(value=0, inplace=True)

        self._confusion_matrix = confusion_matrix

    @property
    def matrix(self) -> pd.DataFrame:
        """
        :return: The matrix as a DataFrame.
        """
        return self._confusion_matrix

    @property
    def matrix_proportions(self) -> pd.DataFrame:
        """
        :return: The matrix as a DataFrame, showing frequencies rather than absolute values.
        """
        # noinspection PyTypeChecker
        return self.matrix / self.total_observations

    @property
    def total_observations(self) -> int:
        """
        :return: the total number of observations
        """
        return self.matrix.loc['Total', 'Total']

    def plot(self,
             include_totals: bool=False,
             proportions: bool=False):
        """
        :param include_totals: if `include_totals` is True, it includes a row and column for totals
        :param proportions: if `proportions` is True, uses the matrix_proportions
        :return: a heatmap plot of the confusion matrix
        """
        ax = plt.axes()
        matrix = round(self.matrix_proportions, 3) if proportions else self.matrix
        matrix = matrix if include_totals else matrix.drop(index='Total', columns='Total')
        sns.heatmap(ax=ax, data=matrix, annot=True, cmap="Blues")
        ax.set_title('Predicted vs Actual Classifications')
        plt.xticks(rotation=20, ha='right')
        plt.yticks(rotation=0)
        plt.xlabel('Predicted', labelpad=20)
        plt.ylabel('Actual', labelpad=20)
        plt.tight_layout()
