from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy as np


class DataSplitterBase(metaclass=ABCMeta):
    """"
    Class that defines methods to split the data into training/test sets
    """

    def __init__(self, holdout_ratio: float, seed: int=42):
        """
        :param holdout_ratio: percentage of the dataset to assign to the holdout set
        :param seed: seed # for random 'state' to control consistently (up to inheritors to use correctly)
        """
        assert 0 < holdout_ratio < 1
        self._holdout_ratio = holdout_ratio
        self._seed = seed

    @abstractmethod
    def split(self, target_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param target_values: the target values (i.e. dependent variable)
        :return: numeric indexes for training set and test set (e.g. use .iloc for pandas DataFrame, not .loc)
        """
        pass
