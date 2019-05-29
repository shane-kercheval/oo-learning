from abc import ABCMeta, abstractmethod
from typing import Tuple
import numpy as np
import pandas as pd


class DataSplitterBase(metaclass=ABCMeta):
    """"
    Class that defines methods to split the data into training/test sets
    """

    def __init__(self, holdout_ratio: float, seed: int = 42):
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

    def split_sets(self, data: pd.DataFrame, target_variable: str) -> Tuple[pd.DataFrame,
                                                                            pd.Series,
                                                                            pd.DataFrame,
                                                                            pd.Series]:
        """
        Rather than returning the indexes (letting the end-user then separate out into training/holdout matrix
            and values it returns
        :param data: a DataFrame to split
        :param target_variable: the target variable
        :return:
            4 objects
            1) Training Dataframe Containing Features
            2) Training array containing target values
            3) Holdout Dataframe Containing Features
            4) Holdout array containing target values
        """
        training_indexes, holdout_indexes = self.split(target_values=data[target_variable])

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns=target_variable)

        holdout_y = data.iloc[holdout_indexes][target_variable]
        holdout_x = data.iloc[holdout_indexes].drop(columns=target_variable)

        return training_x, training_y, holdout_x, holdout_y
