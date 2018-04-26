from typing import List
import pandas as pd

from oolearning.transformers.TransformerBase import TransformerBase


class RemoveColumnsTransformer(TransformerBase):
    """
    Removes the columns passed into the constructor.
    """
    def __init__(self, columns: List[str]):
        super().__init__()
        assert columns is not None
        assert isinstance(columns, list)
        self._columns_to_remove = columns

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        # ensure all columns exist in data
        assert all([column in data_x.columns.values for column in self._columns_to_remove])

        # nothing to save into state
        return {}

    # noinspection PyTypeChecker
    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        return data_x.drop(columns=self._columns_to_remove)
