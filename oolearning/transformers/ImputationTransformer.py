from typing import Callable, List, Union

import numpy as np
import pandas as pd

from oolearning.OOLearningHelpers import OOLearningHelpers
from oolearning.transformers.TransformerBase import TransformerBase


def value_counts(x):
    return pd.value_counts(x).index.values[0]


class ImputationTransformer(TransformerBase):
    """
    Imputes the value for numerical columns based on the median value of that column.
    """
    def __init__(self,
                 numeric_imputation_function: Callable[[pd.Series], pd.Series] = np.nanmedian,
                 categoric_imputation_function: Callable[[pd.Series], pd.Series] = value_counts,
                 group_by_column: str = None,
                 treat_zeros_as_na: bool = False,
                 columns_explicit: List[str] = None,
                 columns_to_ignore: List[str] = None):  # noqa
        """
        :param numeric_imputation_function: A function that will be used to compute the value used to impute
            for numeric features. The default is a function that returns the median; setting this
            field to None will result in no numeric columns being imputed
            Callable takes a pd.Series and returns a pd.Series
        :param categoric_imputation_function: A function that will be used to compute the value used to impute
            for categoric features. The default is a function that returns the mode; setting this
            field to None will result in no categoric columns being imputed
        :param group_by_column: takes a column name, and imputation will be per (or grouped by) that column.
            For example, if we had the titanic dataset, and were imputing the `fare`; rather than, for
            example, taking the median of all the fares, we might want to take the median of only
            the fares associated with the same class (e.g. if the associated class is "upper", we want the
            median of only the upper-class fares.

            If we are imputing the value of the column that is also the column we are grouping by, we will
            simply take the imputed value of all the data in that column.

            If we are imputing a value of a column (X) that is NOT the column we are grouping by, but the
            column we are grouping by is also NA, we will use the imputed value of all of the data in that
            column X.

        :param treat_zeros_as_na: For numeric columns, setting this to True will replace, not only NA values
            with the imputed value, but also values of `0` with the imputed value.
        """
        super().__init__()

        assert not (columns_explicit and columns_to_ignore)  # cannot use both parameters simultaneously

        self._numeric_imputation_function = numeric_imputation_function
        self._categoric_imputation_function = categoric_imputation_function
        self._treat_zeros_as_na = treat_zeros_as_na
        self._group_by_column = group_by_column
        self._columns_explicit = columns_explicit
        self._columns_to_ignore = columns_to_ignore

    def peak(self, data_x: pd.DataFrame):
        pass

    @staticmethod
    def _imputation_helper(data_x: pd.DataFrame,
                           imputation_function: Callable[[pd.Series], pd.Series],
                           column: str,
                           group_by_column: Union[str, None]):
        """
        :return: single value if not by group; otherwise a dictionary per unique value of the group
        """
        # if we are grouping by a specific column
        # and the current column isn't the same one we are grouping by, then ... else just impute
        if group_by_column and column != group_by_column:
            # run the imputation function on each subset of data (for each group), via apply
            values_per_group = data_x.groupby(group_by_column).apply(lambda x: imputation_function(x[column]))
            assert isinstance(values_per_group, pd.Series)
            assert set(values_per_group.index.values) == set(data_x[group_by_column].dropna().unique())
            imputation_value = values_per_group.to_dict()
            # in the event that we are imputing a value based on column X, and X's value is missing,
            # we will simply calculate the value to impute on all of the data (from the column belonging to
            # the value we are imputing)
            # noinspection PyUnresolvedReferences
            imputation_value['all'] = imputation_function(data_x[column])
        else:
            imputation_value = imputation_function(data_x[column])

        return imputation_value

    def _fit_definition(self, data_x: pd.DataFrame) -> dict:
        imputed_values = dict()  # single dictionary if not by group; nested dictionary if by group
        numeric_features, categoric_features = OOLearningHelpers.\
            get_columns_by_type(data_dtypes=data_x.dtypes, target_variable=None)

        # only keep features that are in _columns_explicit, if specified
        if self._columns_explicit:
            numeric_features = [x for x in numeric_features if x in self._columns_explicit]
            categoric_features = [x for x in categoric_features if x in self._columns_explicit]

        # remove features that are in _columns_to_ignore, if specified
        if self._columns_to_ignore:
            numeric_features = [x for x in numeric_features if x not in self._columns_to_ignore]
            categoric_features = [x for x in categoric_features if x not in self._columns_to_ignore]

        # impute and store numeric values, if numeric function exists
        if self._numeric_imputation_function is not None:
            for column in numeric_features:
                if self._treat_zeros_as_na:
                    # if we are treating zeros ans NAs, then we need to replace 0's with NA here so that the
                    # data is fitted correctly (i.e. as if the 0's were NAs), even though we won't be
                    # transforming the data
                    data_x[column].replace(0, np.nan, inplace=True)

                imputed_values[column] = self._imputation_helper(data_x=data_x,
                                                                 imputation_function=self._numeric_imputation_function,  # noqa
                                                                 column=column,
                                                                 group_by_column=self._group_by_column)
        # impute and store categoric values, if numeric function exists
        if self._categoric_imputation_function is not None:
            for column in categoric_features:
                imputed_values[column] = self._imputation_helper(data_x=data_x,
                                                                 imputation_function=self._categoric_imputation_function,  # noqa
                                                                 column=column,
                                                                 group_by_column=self._group_by_column)

        return imputed_values

    def _transform_definition(self, data_x: pd.DataFrame, state: dict) -> pd.DataFrame:

        if self._treat_zeros_as_na:
            numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=data_x.dtypes,
                                                                        target_variable=None)
            for column in state.keys():
                if column in numeric_features:
                    data_x[column].replace(0, np.nan, inplace=True)

        if self._group_by_column:
            # for each group, fill na values with corresponding imputation values from dictionary
            # the imputed_value is a dictionary with the corresponding values per group, unless the column
            # is the same as what we grouped by, then it is a single value
            for column, imputed_value in state.items():
                if column == self._group_by_column:
                    # we don't want to fill the "group by" column before any other column, because in the
                    # event we are imputing a value for column X, and the corresponding row also has a missing
                    # value in the "group by" column, it is (IMO) better to simply take the imputation (e.g.
                    # median value of the entire column X, then it is to first impute the group by column, and
                    # then take the imputed value based off of another imputed value. The "group by" column is
                    # supposed to provide additional insight. Using the insight when it is actually missing
                    # seems like a bad idea, better to just not use it.
                    pass
                else:
                    # first, fill in the values for any rows which are missing data in the "group by column"
                    indexes_of_group_by_na = data_x[data_x[self._group_by_column].isna()].index.values
                    if len(indexes_of_group_by_na) > 0:
                        data_x.loc[indexes_of_group_by_na, column] = \
                            data_x.loc[indexes_of_group_by_na, column].fillna(state[column]['all'])

                    # now fill the rest of the NA values
                    for value in data_x[self._group_by_column].dropna().unique():
                        indexes_of_group_by_value = data_x[data_x[self._group_by_column] == value].\
                            index.values
                        if len(indexes_of_group_by_value) > 0:
                            data_x.loc[indexes_of_group_by_value, column] = \
                                data_x.loc[indexes_of_group_by_value, column].fillna(state[column][value])

            # earlier, we ignored imputing the _group_by_column (if it was one of the columns we were
            # imputing), so it didn't interfere with what we do for NA values of the "group by" column.
            # Let's circle back and now impute the column if it is in fact a column we are imputing.
            if self._group_by_column in state.keys():
                data_x[self._group_by_column].fillna(value=state[self._group_by_column], inplace=True)

        else:
            data_x.fillna(value=self.state, inplace=True)

        return data_x
