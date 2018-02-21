from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy as np


class DataSplitterBase(metaclass=ABCMeta):
    """"
    Class that defines methods to split the data into training/test sets
    """

    def __init__(self, holdout_ratio: float):
        assert 0 < holdout_ratio < 1
        self._test_ratio = holdout_ratio

    @abstractmethod
    def split(self, target_values: np.ndarray, seed: int=42) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param target_values: the target values (i.e. dependent variable)
        :param seed: seed # for random 'state' to control consistently (up to inheritors to use correctly)
        :return: indexes for training set and test set
        """
        pass
