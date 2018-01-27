import numpy as np
import pandas as pd


class ModelSearcherHelpers:
    @staticmethod
    def get_columns_by_type(data_dtypes, target_variable=None):
        """returns numeric columns in first return, and string columns in second"""
        assert isinstance(data_dtypes, pd.Series)
        types_dictionary = dict(data_dtypes)

        if target_variable is not None:
            del types_dictionary[target_variable]

        numeric_columns = [key for key, value in types_dictionary.items()
                           if value in [np.dtype('float64'), np.dtype('int64')]]
        non_numeric_columns = [key for key, value in types_dictionary.items() if key not in numeric_columns]

        return numeric_columns, non_numeric_columns