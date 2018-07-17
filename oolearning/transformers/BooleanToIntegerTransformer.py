from typing import List, Union
import pandas as pd

from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.OOLearningHelpers import OOLearningHelpers


class BooleanToIntegerTransformer(TransformerBase):
    """
    Removes the columns passed into the constructor.
    """
    def __init__(self, columns: Union[List[str], None]=None):
        """

        :param columns: if `columns` is None, then any column that is detected to be `bool` is converted,
            otherwise only the columns specified
        """
        super().__init__()

        if columns is not None:
            assert isinstance(columns, list)
        self._columns_to_convert = columns

    def peak(self, data_x: pd.DataFrame):
        pass

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        if self._columns_to_convert is None:  # detect boolean columns
            self._columns_to_convert = [column for column in data_x.columns.values
                                        if OOLearningHelpers.is_series_boolean(data_x[column])]
        else:
            # ensure all columns exist in data and it is type boolean
            assert all([column in data_x.columns.values and OOLearningHelpers.is_series_boolean(data_x[column])  # noqa
                        for column in self._columns_to_convert])

        # nothing to save into state
        return {}

    # noinspection PyTypeChecker
    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        for column in self._columns_to_convert:
            data_x[column] = data_x[column].astype(int)

        return data_x
