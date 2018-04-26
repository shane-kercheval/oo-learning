from abc import ABCMeta, abstractmethod

import copy
import pandas as pd


class TransformerBase(metaclass=ABCMeta):
    """
    A transformer is an object that transforms data-sets by first `fitting` an initial data-set, and saving
    the values necessary to consistently transform future data-sets based on the fitted data-set.
    """

    def __init__(self, check_dataframe_indexes_maintained=True):
        """
        :param check_dataframe_indexes_maintained: Flag to check that the indexes of the transformed DataFrame
            match the indexes of the DataFrame passed into `transform`/`fit_transform`
        """
        self._state = None
        self._check_dataframe_indexes_maintained = check_dataframe_indexes_maintained

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
            state is a dictionary of values; the values are used to consistently transform subsequent
            data-sets e.g. the state might be the median value of each of the columns of the training set,
            used to impute future data-sets using the same numbers/medians during each transformation
        """
        return self._state

    @abstractmethod
    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        """
        determines ('fits') the information necessary to transform future data-sets

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

    @abstractmethod
    def peak(self, data_x: pd.DataFrame):
        """
        There are times (for example when resampling) when the data is fitted with a subset of data, and if a
            rare value is not contained within the data when fitted, but shows up in a future data-set (i.e.
            during `transform`), then there are instances (for example when creating dummy variables) that the
            unexpected data creates problems. In the case of creating dummy variables, `transform` would add a
            new columns that didn't previously exist when fitting the original model, which would cause the
            model to crash. Therefore, sometimes, we will want to 'peak' at the data to make sure we
            incorporate future 'unseen' values. This should obviously not be used to do calculations (such
            as computing values to impute for the ImputationTransformer).

        Note, there is also risk of peaking in that the data when peaked at might not be the same as the
            data that is passed in from `fit`, e.g. from previous transformations.

        The TransformerPipeline calls `peak()` for each Transformer in the pipeline. It is up to each class
            to utilize or ignore this functionality. (Most will probably ignore.)

        :return: nothing
        """
        pass

    def fit(self, data_x: pd.DataFrame):
        """
        saves the necessary information into _state to transform future data-sets

        :param data_x: data to fit
        :return: None
        """
        assert self._state is None  # ensure that we have not fitted the data previously
        assert isinstance(data_x, pd.DataFrame)
        data_x = data_x.copy()
        # noinspection PyTypeChecker
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
        transformed_data = self._transform_definition(data_x=data_x, state=self._state)
        if self._check_dataframe_indexes_maintained:
            # noinspection PyTypeChecker
            assert all(transformed_data.index.values == data_x.index.values)
        return transformed_data

    def fit_transform(self, data_x: pd.DataFrame) -> pd.DataFrame:
        """
        convenience method that calls both fit & transform (e.g. could be used on the initial training set)

        :param data_x:
        :return:
        """
        self.fit(data_x=data_x)
        return self.transform(data_x=data_x)
