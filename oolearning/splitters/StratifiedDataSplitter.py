from abc import abstractmethod
from typing import Tuple, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

from oolearning.splitters.DataSplitterBase import DataSplitterBase


class StratifiedDataSplitter(DataSplitterBase):
    """
    Base class splitting the data in a 'stratified' manner (i.e. maintain 'balance' of the target variable,
    whether it is categorical balance in the case of a classification problem or maintain the distribution
    of the target variable in the case of a regression problem.
    """

    @abstractmethod
    def labels_to_stratify(self, target_values: np.ndarray) -> np.ndarray:
        """
        :param target_values:
        :return: the labels the `split` method will use to stratify
        """
        pass

    def split(self, target_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns indexes corresponding to the training and test sets based on stratified values from
            `target_values`
        :param target_values: the values to stratify
        :return: training and testing indexes based on the holdout_ratio passed into the constructor
        """
        train_indexes, holdout_indexes = self.split_monte_carlo(target_values=target_values,
                                                                samples=1,
                                                                seed=self._seed)

        # indexes will be a list of lists, and in this case, there will only be 1 list inside the list
        return train_indexes[0], holdout_indexes[0]

    def split_monte_carlo(self, target_values: np.ndarray, samples: int, seed: int=42) \
            -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        creates multiple samples (determined by `samples` parameter) of stratified training and test data
            e.g. if `samples=5`, the length of the `target_values` array is 100, and `holdout_ratio` passed
            into the constructor is 0.20, this function returns two tuples, each item being a list of indexes;
            the first item/list will have 5 arrays of 80 values, representing the training indexes, and
            the second item/list will have 5 arrays of 20 values, representing the test indexes
            (similar to description in Applied Predictive Modeling pg 71/72
        :param target_values: the values to stratify
        :param samples: the number of samples to return
        :param seed: seed used by the random number generator
        :return: list of training and testing indexes for each fold, and based on the holdout_ratio passed
        into the constructor
        """
        pre_labels = self.labels_to_stratify(target_values=target_values)  # get labels to stratify
        labels = LabelEncoder().fit(pre_labels).transform(pre_labels)  # en
        split = StratifiedShuffleSplit(n_splits=samples, test_size=self._holdout_ratio, random_state=seed)

        train_indexes = list()
        test_indexes = list()
        for train_ind, test_ind in split.split(np.zeros(len(target_values)), labels):
            assert len(train_ind) - round(len(target_values) * (1 - self._holdout_ratio)) < 2  # assert close
            assert len(test_ind) - round(len(target_values) * self._holdout_ratio) < 2  # assert close
            assert set(train_ind).isdisjoint(test_ind)

            train_indexes.append(train_ind.tolist())
            test_indexes.append(test_ind.tolist())

        return train_indexes, test_indexes
