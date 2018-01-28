from typing import Callable

import numpy as np
import pandas as pd

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.transformers.TransformerBase import TransformerBase


class ImputationTransformer(TransformerBase):
    """
    Imputes the value for numerical columns based on the median value of that column.
    """
    def __init__(self,
                 numeric_imputation_function: Callable[[pd.Series], pd.Series]=np.nanmedian,
                 categoric_imputation_function: Callable[[pd.Series], pd.Series]=lambda x: pd.value_counts(x).index.values[0]):  # noqa
        """
        :param numeric_imputation_function:  default is a function that returns the median; setting this
            field to None will result in no numeric columns being imputed
            Callable takes a pd.Series and returns a pd.Series
        :param categoric_imputation_function: default is a function that returns the mode; setting this
            field to None will result in no categoric columns being imputed
        """
        super().__init__()
        self._numeric_imputation_function = numeric_imputation_function
        self._categoric_imputation_function = categoric_imputation_function

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        imputed_values = dict()
        # make sure the target/response column is ignored
        numeric_features, categoric_features = OOLearningHelpers.\
            get_columns_by_type(data_dtypes=data_x.dtypes, target_variable=None)

        if self._numeric_imputation_function is not None:
            for column in numeric_features:
                imputed_values[column] = self._numeric_imputation_function(data_x[column])

        if self._categoric_imputation_function is not None:
            for column in categoric_features:
                imputed_values[column] = self._categoric_imputation_function(data_x[column])

        return imputed_values

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        for column, imputed_value in state.items():
            data_x[column].fillna(value=imputed_value, inplace=True)

        return data_x
