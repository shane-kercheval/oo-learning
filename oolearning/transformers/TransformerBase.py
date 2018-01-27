from abc import ABCMeta, abstractmethod

import copy
import pandas as pd


class TransformerBase(metaclass=ABCMeta):
    """
    A transformer is an object that transforms datasets by first `fitting` an an initial dataset, and saving
    the values necessary to consistently transform future datasets based on the fitted dataset.
    """

    def __init__(self):
        self._state = None

    def clone(self):
        """
        :return: a clone of the current object
        """
        assert self._state is None  # only intended on being called before transforming
        return copy.deepcopy(self)

    @property
    def state(self) -> dict:
        """
        :return: the 'state' saved during fitting.
            state is a dictionary of values; the values are used to consistently transform subsequent datasets
            e.g. the state might be the median value of each of the columns of the training set, used to
            impute future datasets using the same numbers/medians during each transformation
        """
        return self._state

    @abstractmethod
    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        """
        determines ('fits') the information necessary to transform future datasets
        :param data_x: data to fit
        :return: state (dictionary) to be saved for the next transformation
        """
        pass

    @abstractmethod
    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        """
        performs the transformations of `data` based on the previously fitted information
        :param data_x: the data (DataFrame) to transform
        :param state: the previous state saved based on the fitted data (i.e. values needed to do the
        transformation)
        :return: transformed DataFrame
        """
        pass

    def fit(self, data_x: pd.DataFrame):
        """
        saves the necessary information into _state to transform future datasets
        :param data_x: data to fit
        :return: None
        """
        assert self._state is None  # ensure that we have not fitted the data previously
        assert isinstance(data_x, pd.DataFrame)
        self._state = self._fit_definition(data_x=data_x)
        assert self._state is not None  # ensure after we have fitted the transformation, we have cached state

    def transform(self, data_x: pd.DataFrame) -> pd.DataFrame:
        """
        performs the transformations of `data` based on the previously fitted information
        :param data_x:
        :return: transformed DataFrame
        """
        assert self._state is not None  # make sure we have fitted the data and saved the state before we go
        data_x = data_x.copy()
        assert isinstance(data_x, pd.DataFrame)
        assert isinstance(self._state, dict)
        return self._transform_definition(data_x=data_x, state=self._state)

    def fit_transform(self, data_x: pd.DataFrame) -> pd.DataFrame:
        """
        convenience method that calls both fit & transform (e.g. could be used on the initial training set)
        :param data_x:
        :return:
        """
        self.fit(data_x=data_x)
        return self.transform(data_x=data_x)
