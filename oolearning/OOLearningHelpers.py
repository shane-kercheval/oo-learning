from typing import List

import numpy as np
import pandas as pd


class OOLearningHelpers:

    _numeric_dtypes = [np.dtype('float64'), np.dtype('int64')]

    @staticmethod
    def is_series_numeric(variable: pd.Series):
        return variable.dtype in OOLearningHelpers._numeric_dtypes

    @staticmethod
    def is_series_dtype_numeric(dtype: np.dtype):
        return dtype in OOLearningHelpers._numeric_dtypes

    @staticmethod
    def get_columns_by_type(data_dtypes: List[np.dtype], target_variable: str=None):
        """returns numeric columns in first return, and string columns in second"""
        assert isinstance(data_dtypes, pd.Series)
        types_dictionary = dict(data_dtypes)

        if target_variable is not None:
            del types_dictionary[target_variable]

        numeric_columns = [key for key, value in types_dictionary.items()
                           if OOLearningHelpers.is_series_dtype_numeric(dtype=value) is True]
        non_numeric_columns = [key for key, value in types_dictionary.items()
                               if OOLearningHelpers.is_series_dtype_numeric(dtype=value) is False]

        return numeric_columns, non_numeric_columns
