from typing import List
import pandas as pd

from oolearning.transformers.TransformerBase import TransformerBase


class CategoricConverterTransformer(TransformerBase):
    """
    Converts columns to *unordered* pd.Categorical
    """

    def __init__(self, columns: List[str]):
        super().__init__()
        assert columns is not None
        assert isinstance(columns, list)
        self._columns_to_convert = columns

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        # save unique values for each column to convert, so that if converting a smaller (future) dataset that
        # doesn't contain all the values (e.g. a single observation), we recreate categories that have the
        # original categories
        unique_column_values = dict()

        for column in self._columns_to_convert:
            unique_column_values[column] = sorted(data_x[column].dropna().unique().tolist())

        return unique_column_values

    # noinspection PyTypeChecker,PyUnresolvedReferences
    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        data_x = data_x.copy()
        for column, unique_values in state.items():
            data_x[column] = pd.Categorical(data_x[column], categories=unique_values)

        return data_x
