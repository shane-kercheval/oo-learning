from typing import Callable
import pandas as pd

from oolearning.transformers.TransformerBase import TransformerBase


class StatelessTransformer(TransformerBase):
    """
    Provides a way to pass functions (i.e. custom transformations) that do not need to save any type of
    state, so transformations are limited to those that do not require retaining any information (as required
    by, for example, imputation transformations). An example of a stateless transformation would be to remove
    certain columns. The transformation is done without any prior state or values/information.
    """

    def __init__(self, custom_function: Callable):
        """
        :param custom_function: a function that takes a pandas DataFrame, does a stateless transformation,
            and returns a pandas DataFrame
        """
        super().__init__()
        self._custom_function = custom_function

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        # nothing to save into state
        return {}

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        transformed_data = self._custom_function(data_x.copy())
        assert isinstance(transformed_data, pd.DataFrame)
        return transformed_data
