from typing import Callable

import pandas as pd

from oolearning.transformers.TransformerBase import TransformerBase


class StatelessColumnTransformer(TransformerBase):
    """
    Provides a way to pass functions (i.e. custom transformations) that do not need to save any type of
    state, so transformations are limited to those that do not require retaining any information (as required
    by, for example, imputation transformations). An example of a helper transformation would be to apply a
    log transformation to a numeric column. Unlike, for example, centering and scaling, calculating the log
    for each row (for a given column) does not depend on the values of any other rows.
    """
    def __init__(self, columns: list, custom_function: Callable):
        """
        :param columns: the columns to apply the `custom_function` to
        :param custom_function: a function that takes a pandas DataFrame, does a helper transformation,
            and returns a pandas DataFrame
        """
        super().__init__()
        self._columns = columns
        self._custom_function = custom_function

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        # ensure all columns exist in data
        assert all([column in data_x.columns.values for column in self._columns])
        # nothing to save into state
        return {}

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        transformed_data = data_x.copy()
        # noinspection PyUnresolvedReferences
        transformed_data[self._columns] = transformed_data[self._columns].apply(self._custom_function)
        # noinspection PyTypeChecker
        return transformed_data
