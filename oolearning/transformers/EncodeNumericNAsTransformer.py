from typing import Union
import itertools
import numpy as np

import pandas as pd
from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.enums.CategoricalEncoding import CategoricalEncoding
from oolearning.transformers.TransformerBase import TransformerBase


class EncodeNumericNAsTransformer(TransformerBase):
    """
    The idea with this class/functionality is that, within a dataset, if the population that is associated
        with missing values has significantly different distributions of features (i.e. features that don't
        have missing values) when compared with the population that *doesn't* have missing values (or above/
        below some threshold), then simply imputing missing values might be overly misrepresentative. If, for
        example, you were clustering, and the distributions were the same and you had enough data, then
        you might simply remove the subset that has missing data and assume you aren't leaving out additional
        segments. If the distributions are different an alternative to imputing the values might be to create
        an indicator column for missing values, so that missing values is treated as its own
        feature/dimension. The downside to this approach is that the original column must still have a value
        for NA, for example `0`, which might be misleading in itself.

    Therefore, this transformation will create an additional column for each numeric column that contains
        np.nan values. So if the original column is `XXX` the additional column will be `XXX_NA`. Rows
        containing np.nan value in `XXX` will have a `1` in `XXX_NA` and will be replaced with a `0` (or a
        custom value) in `XXX`.

    """
    def __init__(self,
                 columns_to_encode: Union[list, None]=None,
                 replacement_value: int=0):
        """
        :param columns_to_encode: a list specific columns to encode. The list should only contain numeric
            features. If this parameter is set, then _NA columns will be generated for every specified column,
            even if no NA values are found. If this parameter is not set, only numeric columns that contain
            NA values will be converted.
        """
        super().__init__()

        if columns_to_encode is not None:
            assert isinstance(columns_to_encode, list)
            assert len(columns_to_encode) > 0

        self._columns_to_encode = columns_to_encode
        self._replacement_value = replacement_value

        self._columns_to_reindex = None
        self._encoded_columns = None
        self._peak_state = None

    def __generate_state(self, data_x) -> dict:
        """
        :return: {'columns': ...} where `...` contains the list of columns that are A) numeric and/or passed
            in, and B) include NA values
        """
        numeric_columns, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                   target_variable=None)

        # if columns_to_include is None, then we get the numeric columns that have missing values
        # if it already specified, then we simply make sure the columns are numeric, but they don't have to
        # contain missing values
        if self._columns_to_encode is None:
            na_columns = data_x.columns[data_x.isnull().any()].values
            # numeric columns that have missing values
            self._columns_to_encode = [x for x in numeric_columns if x in na_columns]
        else:
            # columns_to_include was specified or already set, verify that all columns are numeric
            assert all([x in numeric_columns for x in self._columns_to_encode])

        return {'columns': self._columns_to_encode}

    def peak(self, data_x: pd.DataFrame):
        """
        The problem is that when Encoding, specifically when resampling etc…. the data/Transformer is
        fitted with only a subset of values that it will eventually see, and if a rare value is not in the
        dataset that is fitted, but shows up in a future dataset (i.e. during `transform`), then getting the
        encoded columns would result in a dataset that contains columns that the model didn't see when
        fitting, and therefore, doesn't know what to do with. So, before transforming, we will allow
        the transformer to 'peak' at ALL the data.

        TransformerPipeline has a `peak` function which calls `peak` for each Transformer.
        So particular 'model_processor` classes may take advantage of that if used in the pipeline.
        But the class may be used directly, so in `_fit_definition`, if `_peak_state` is None, then we will
        call it manually. Technically, no "peak" happened, but it will be the same result in that scenario.
        """
        self._peak_state = self.__generate_state(data_x=data_x)

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        if self._peak_state is None:  # see comments in `peak` for an explanation.
            self.peak(data_x=data_x)

        state = self._peak_state

        return state

    @staticmethod
    def __flatten(l):
        return [item for sublist in l for item in sublist]

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:
        ######################################################################################################
        # based on the columns/categories found in the `fit` stage,
        # 1) ensure that all the categories being transformed are a SUBSET of what was previously found (i.e.
        # no new columns/values
        # 2) create a new DataFrame with the corresponding dummy columns
        # 4) reindex according to previously defined columns in `fit`
        ######################################################################################################
        found_state = self.__generate_state(data_x=data_x)
        found_columns_to_include = found_state['columns']
        assert self._columns_to_encode == found_columns_to_include

        # "list" containing all the columns (nested list for each column associated with XXX_NA)
        unflattened = [x if x not in self._columns_to_encode else [x + '_NA', x] for x in data_x.columns.values]  # noqa

        def flatten(items):
            return itertools.chain.from_iterable(itertools.repeat(x,1) if isinstance(x,str) else x for x in items)  # noqa

        self._columns_to_reindex = list(flatten(unflattened))

        for feature in self._columns_to_encode:
            non_na_indices = data_x.index[data_x[feature].notnull()].values
            feature_na = feature + '_NA'
            data_x[feature_na] = data_x[feature]
            # for _NA column, fill all NA with 1 and everything else with 0
            data_x[feature_na].fillna(value=1, inplace=True)
            data_x.loc[non_na_indices, feature_na] = 0
            # for regular column, fill NA with specified replacement value
            data_x[feature].fillna(value=self._replacement_value, inplace=True)

        # ensure all columns exist (e.g. if transforming single observation, ensure that columns associated
        # with values not found are still created; ensures columns are dropped consistently for dummy encoding
        return data_x.reindex(columns=self._columns_to_reindex, fill_value=0)
