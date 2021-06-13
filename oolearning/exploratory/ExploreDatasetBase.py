from abc import ABCMeta, abstractmethod
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pandas.plotting import scatter_matrix

from oolearning import OOLearningHelpers


class ExploreDatasetBase(metaclass=ABCMeta):
    """
    Base class (for e.g. classification and regression specific overriding) that gives convenience while
        exploring a new dataset by providing common functionality frequently needed during standard
        exploration.

    Note: a copy of the dataset is made and stored in this class.
    """
    def __init__(self, dataset: pd.DataFrame, target_variable: Union[str, None] = None):
        """
        :param dataset: dataset to explore. Note: a copy of the dataset is made and stored in this class.
        :param target_variable: the name of the target variable/column
        """
        self._dataset = dataset.copy()
        self._target_variable = target_variable

    def __str__(self):

        string = "" if self.target_variable is None \
            else "Target Variable: " + self.target_variable + "\n================\n\n"

        string += "Numeric Summary\n===============\n"
        string += self.numeric_summary().to_string()

        string += "\n\n\nCategoric Summary\n=================\n"
        string += self.categoric_summary().to_string()

        return string

    class NoFeaturesMatchThresholdException(Exception):
        pass

    @abstractmethod
    def plot_against_target(self, feature: str):
        """
        Shows a plot of the specific `feature` against, or compared with, the target variable. The type of
            graph depends on whether the target variable is numeric (regression) or categoric
            (classification).

        :param feature: feature to visualize and compare against the target
        """
        pass

    @classmethod
    def from_csv(cls,
                 csv_file_path: str,
                 target_variable: Union[str, None] = None,
                 skip_initial_space: bool = True,
                 separator: str = ',') -> 'ExploreDatasetBase':
        """
        Instantiates this class (via subclass) by first loading in a csv from `csv_file_path`.

        NOTE: this method sets non-numeric columns to `pd.Categorical` types.

        :param csv_file_path: path to the csv file
        :param target_variable: the name of the target variable/column
        :param skip_initial_space: Skip spaces after delimiter.
        :param separator: Delimiter to use. If sep is None, the C engine cannot automatically detect the
            separator ... https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
        :return: an instance of this class (i.e. subclass)
        """
        explore = cls(dataset=pd.read_csv(csv_file_path,
                                          skipinitialspace=skip_initial_space,
                                          sep=separator),
                      target_variable=target_variable)

        _, categoric_features = OOLearningHelpers.get_columns_by_type(data_dtypes=explore.dataset.dtypes,
                                                                      target_variable=target_variable)
        for feature in categoric_features:
            explore.dataset[feature] = explore.dataset[feature].astype('category')

        return explore

    @property
    def dataset(self) -> pd.DataFrame:
        """
        Returns the underlying dataset of interest.

        WARNING: The underlying dataset should be changed from these class methods (i.e. subclass), rather
        than changing directly, since this class caches information about the dataset. If changes are made,
        the user can call `._update_cache()` manually.
        """
        return self._dataset

    def drop(self, columns: List[str]):
        """
        :param columns: a list of column names to "drop" i.e. remove from the dataset
        """
        self._dataset.drop(columns=columns, inplace=True)

    @property
    def target_variable(self) -> str:
        """
        Returns the name of the target variable.
        """
        return self._target_variable

    @property
    def is_target_numeric(self) -> Union[List[str], None]:
        """
        Returns the names of the numeric features (columns without target variable).
        """
        if self.target_variable is None:
            return None

        return OOLearningHelpers.is_series_numeric(variable=self.dataset[self.target_variable])

    @property
    def numeric_features(self) -> List[str]:
        """
        Returns the names of the numeric features (columns without target variable).
        """
        numeric_features, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=self._dataset.dtypes,
                                                                    target_variable=self._target_variable)
        return numeric_features

    @property
    def categoric_features(self) -> List[str]:
        """
        Returns the names of the categoric features (columns without target variable).
        """
        _, categoric_features = OOLearningHelpers.get_columns_by_type(data_dtypes=self._dataset.dtypes,
                                                                      target_variable=self._target_variable)
        return categoric_features

    @property
    def numeric_columns(self) -> List[str]:
        numeric_columns, _ = OOLearningHelpers.get_columns_by_type(data_dtypes=self._dataset.dtypes,
                                                                   target_variable=None)
        return numeric_columns

    @property
    def categoric_columns(self) -> List[str]:
        _, categoric_columns = OOLearningHelpers.get_columns_by_type(data_dtypes=self._dataset.dtypes,
                                                                     target_variable=None)
        return categoric_columns

    def numeric_summary(self) -> Union[pd.DataFrame, None]:
        """
        Returns the following attributes (as columns) for each of the numeric features (as rows).

        `count`: The number of non-null values found for the given feature.
        `nulls`: The number of null values found for the given feature.
        `perc_nulls`: The percent of null values found (i.e. `nulls / (count + nulls)`) for a given feature.
        `num_zeros`: The number of values that equal `0`, found for the given feature.
        `perc_zeros`: The percent of `0`s found (i.e. `num_zeros / number of values in series`) for a given
            feature. Note: `number of values in series` is `count` + `nulls`, so this shows the percent of
            zeros found considering all of the values in the series, not just the non-null values.
        `mean`: The `mean` of all the values for a given feature.
        `st_dev`: The `standard deviation` of all the values for a given feature.
        `coef of var`: The `coefficient of variation (CV)`, is defined as the standard deviation divided by
            the mean, and describes the variability of the feature's values relative to its mean.

            We can use this metric to compare the variation of two different variables (i.e. features) that
            have different units or scales.
        `skewness`: "unbiased skew"; utilizes `pandas` DataFrame underlying `.skew()` function
        `kurtosis`: "unbiased kurtosis ... using Fisher’s definition of kurtosis"; utilizes `pandas` DataFrame
            underlying `.skew()` function
        `min`: minimum value found
        `10%`: the value found at the 10th percentile of data
        `25%`: the value found at the 25th percentile of data
        `50%`: the value found at the 50th percentile of data
        `75%`: the value found at the 75th percentile of data
        `90%`: the value found at the 90th percentile of data
        `max`: maximum value found
        """
        # if there aren't any numeric features and the target variable is not numeric, we don't have anything
        # to display, return None
        numeric_columns = self.numeric_columns

        if numeric_columns is None or len(numeric_columns) == 0:
            return None

        # column, number of nulls in column, percent of nulls in column
        null_data = [(column,
                      self._dataset[column].isnull().sum(),
                      round(self._dataset[column].isnull().sum() / len(self._dataset), 3))
                     for column in numeric_columns]
        columns, num_nulls, perc_nulls = zip(*null_data)

        # column, number of 0's, percent of 0's
        zeros_data = [(column, sum(self._dataset[column] == 0),
                       round(sum(self._dataset[column] == 0) / len(self._dataset), 3))
                      for column in numeric_columns]
        columns, num_zeros, perc_zeros = zip(*zeros_data)
        return pd.DataFrame({'count': [self._dataset[x].count() for x in numeric_columns],
                             'nulls': num_nulls,
                             'perc_nulls': perc_nulls,
                             'num_zeros': num_zeros,
                             'perc_zeros': perc_zeros,
                             'mean': [round(self._dataset[x].mean(), 3) for x in numeric_columns],
                             'st_dev': [round(self._dataset[x].std(), 3) for x in numeric_columns],
                             'coef of var': [round(self._dataset[x].std() / self._dataset[x].mean(), 3)
                                             if self._dataset[x].mean() != 0 else np.nan
                                             for x in numeric_columns],
                             'skewness': [round(self._dataset[x].skew(), 3) for x in numeric_columns],
                             'kurtosis': [round(self._dataset[x].kurt(), 3) for x in numeric_columns],
                             'min': [round(self._dataset[x].min(), 3) for x in numeric_columns],
                             '10%': [round(self._dataset[x].quantile(q=0.10), 3) for x in numeric_columns],
                             '25%': [round(self._dataset[x].quantile(q=0.25), 3) for x in numeric_columns],
                             '50%': [round(self._dataset[x].quantile(q=0.50), 3) for x in numeric_columns],
                             '75%': [round(self._dataset[x].quantile(q=0.75), 3) for x in numeric_columns],
                             '90%': [round(self._dataset[x].quantile(q=0.90), 3) for x in numeric_columns],
                             'max': [round(self._dataset[x].max(), 3) for x in numeric_columns]},
                            index=columns,
                            columns=['count', 'nulls', 'perc_nulls', 'num_zeros', 'perc_zeros', 'mean',
                                     'st_dev', 'coef of var', 'skewness', 'kurtosis', 'min', '10%', '25%',
                                     '50%', '75%', '90%', 'max'])

    def categoric_summary(self) -> Union[pd.DataFrame, None]:
        """
        Returns the following attributes (as columns) for each of the categoric features (as rows).

        `count`: The number of non-null values found for the given feature.
        `nulls`: The number of null values found for the given feature.
        `perc_nulls`: The percent of null values found (i.e. `nulls / (count + nulls)`) for a given feature.
        `top`: The most frequent value found for a given feature.
        `unique`: The number of unique values found for a given feature.
        `perc_unique`: The percent of unique values found (i.e. `unique` divided by the total number of values
            (null or non-null) for a given feature. 
        """
        # if there aren't any categoric features and the target variable is numeric, we don't have anything
        # to display, return None

        categoric_columns = self.categoric_columns
        if categoric_columns is None or len(categoric_columns) == 0:
            return None

        # column, number of nulls in column, percent of nulls in column
        null_data = [(column,
                      self._dataset[column].isnull().sum(),
                      round(self._dataset[column].isnull().sum() / len(self._dataset), 3))
                     for column in categoric_columns]
        columns, num_nulls, perc_nulls = zip(*null_data)
        return pd.DataFrame({'count': [self._dataset[x].count() for x in categoric_columns],
                             'nulls': num_nulls,
                             'perc_nulls': perc_nulls,
                             'top': [self._dataset[x].value_counts().index[0] for x in categoric_columns],
                             'unique': [len(self._dataset[x].dropna().unique()) for x in categoric_columns],
                             'perc_unique': [round(len(self._dataset[x].dropna().unique()) /
                                                   self._dataset[x].count(), 3) for x in categoric_columns]},
                            index=columns,
                            columns=['count', 'nulls', 'perc_nulls', 'top', 'unique', 'perc_unique'])

    def set_as_categoric(self, feature: str, mapping: dict, ordered: bool = False):
        """
        some features have a numeric type but are logically categorical variables. This method allows the
            variable to be set as a categoric feature, updating the values
        :param feature:
        :param mapping: dictionary containing the unique values of the codes/integers in the current data as
            the key, and the categoric string as the value.
        :param ordered:
        :return:
        """
        # pandas expects code to be `0, 1, 2, ...`, which won't always be the case. We need to map the actual
        # values to what pandas expects.
        actual_to_expected_mapping = dict(zip(mapping.keys(), np.arange(len(mapping))))
        codes = pd.Series(self._dataset[feature]).map(actual_to_expected_mapping).fillna(-1)
        self._dataset[feature] = pd.Categorical.from_codes(codes.astype(int),
                                                           mapping.values(),
                                                           ordered=ordered)

    def set_level_order(self, categoric_feature: str, levels: list):
        """
        Sets the "order" (via Pandas `.cat.reorder_categories()`) of the `categoric_feature to the `levels`.

        :param categoric_feature: the feature to change
        :param levels: the levels to set the feature to.
        """
        assert self._dataset[categoric_feature].dtype.name == 'category'  # must be a category
        self._dataset[categoric_feature].cat.reorder_categories(levels, inplace=True)

    def as_type(self, columns_types: dict):
        """
        Cycles through each column (keys of dictionary) and sets the columns to the type specified in the
            value of the dictionary.
        :param columns_types:
        """
        for column, dtype in columns_types.items():
            self._dataset[column] = self._dataset[column].astype(dtype=dtype)

    def unique_values(self, categoric_feature: str, sort_by_feature=False) -> pd.DataFrame:
        """
        Shows the unique values and corresponding frequencies.

        :param categoric_feature: the categoric feature of interest
        :param sort_by_feature: if `True`, then if the column is a `pd.Categorical`, the order of the
            rows will be based on the order; if `False`, the data will be ordered by frequency
        :return: a DataFrame showing the unique values (as rows) and frequencies.
        """
        # only for categoric features (and the target variable if it is categoric)
        assert categoric_feature in self.categoric_columns
        count_series = self._dataset[categoric_feature].value_counts(sort=not sort_by_feature)
        count_df = pd.DataFrame(count_series)
        count_df['perc'] = (count_series.values / count_series.values.sum()).round(3)
        count_df.columns = ['freq', 'perc']

        return count_df

    def plot_unique_values(self, categoric_feature: str, sort_by_feature=False):
        """
        A bar-chart visualization of `.unique_values()`

        :param categoric_feature: the categoric feature of interest
        :param sort_by_feature: if `True`, then if the column is a `pd.Categorical`, the order of the
            rows will be based on the order; if `False`, the data will be ordered by frequency
        """
        unique_values = self.unique_values(categoric_feature=categoric_feature,
                                           sort_by_feature=sort_by_feature)

        # noinspection PyUnresolvedReferences
        ax = unique_values.drop(columns='perc').plot(kind='bar', rot=10, title=categoric_feature)
        for idx, label in enumerate(list(unique_values.index)):
            freq = unique_values.loc[label, 'freq']
            perc = unique_values.loc[label, 'perc']

            ax.annotate(freq, (idx, freq), xytext=(-8, 2), textcoords='offset points')
            ax.annotate("{0:.0f}%".format(perc * 100), (idx, 2), xytext=(-8, 0), textcoords='offset points')

        ax.set_xticklabels(labels=unique_values.index.values, rotation=20, ha='right')
        plt.tight_layout()

    def plot_boxplot(self, numeric_feature: str):
        """
        Creates a Box-plot of the numeric_feature.
        """
        assert numeric_feature in self.numeric_columns
        ax = self._dataset[numeric_feature].plot(kind='box')
        ax.set_xlabel("")
        ax.set_ylabel(numeric_feature)
        plt.title("")
        plt.tight_layout()

    def plot_histogram(self,
                       numeric_feature: str,
                       num_bins: int = 10):
        """
        Creates a Histogram of the numeric_feature.
        """
        # only for numeric features (and the target variable if it is numeric)
        assert numeric_feature in self.numeric_columns
        self._dataset[numeric_feature].hist(bins=num_bins)
        plt.title(numeric_feature)
        plt.tight_layout()

    def plot_scatterplot_numerics(self, numeric_columns=None, figure_size: tuple = (12, 8)):
        """
        Creates a Scatter-plot among various numeric_features.
        :param numeric_columns: The numeric columns to include in the plot. If `numeric_columns` is none,
            all numeric columns will be plotted.
        :param figure_size:
        :return:
        """
        if numeric_columns is None:
            numeric_columns = self.numeric_columns
        scatter_matrix(self._dataset[numeric_columns], figsize=figure_size)
        plt.tight_layout()

    def plot_correlation_heatmap(self,
                                 numeric_features: Union[list, None] = None,
                                 threshold: Union[float, None] = None,
                                 figure_size: tuple = (10, 8),
                                 round_by: int = 2,
                                 features_to_highlight: Union[list, None] = None):
        """
        Creates a heatmap of the correlations between all of the numeric columns
            (feature + target if numeric).

        :param threshold: the heatmap only includes columns that have a correlation value, corresponding to at
            least one other column, where the absolute value is higher than the threshold.

            So for example, if the threshold is `0.8` then all columns that have a correlation (absolute)
                value of `>=.80` anywhere in the correlation matrix (i.e. with any other column), will show in
                the heatmap, even though a specific column might not have high correlations with every other
                column included.
        :param figure_size: width, height in inches.
        :param round_by: the number of decimal places to round to when showing the correlations in the heatmap
        :param numeric_features: columns to include in heatmap; must be numeric columns
        :param features_to_highlight: feature labels to highlight in red
        """
        if numeric_features is not None:
            # ensure is a numeric column
            valid_columns = self.numeric_columns
            assert all([x in valid_columns for x in numeric_features])
            correlations = self._dataset[numeric_features].corr()
        else:
            correlations = self._dataset.corr()

        if threshold is not None:

            features = correlations.columns.values
            correlation_matrix = np.abs(correlations.values)
            np.fill_diagonal(correlation_matrix, np.NaN)
            meets_threshold = np.apply_along_axis(lambda x: np.any(x >= threshold), 0, correlation_matrix)
            if not meets_threshold.any():
                raise self.NoFeaturesMatchThresholdException('correlation `threshold` set too high.')

            features_meets_threshold = features[meets_threshold]
            correlations = correlations.loc[features_meets_threshold, features_meets_threshold]

        OOLearningHelpers.plot_correlations(correlations=correlations,
                                            title='correlations',
                                            figure_size=figure_size,
                                            round_by=round_by,
                                            features_to_highlight=features_to_highlight)
