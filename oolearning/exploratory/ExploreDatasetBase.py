from abc import ABCMeta
from typing import List
from matplotlib import pyplot as pl
from pandas.plotting import scatter_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from oolearning import OOLearningHelpers


class ExploreDatasetBase(metaclass=ABCMeta):
    def __init__(self, dataset: pd.DataFrame, target_variable: str):
        """
        # TODO: DOCUMENT; NOTE: unlike from_csv, THIS METHOD DOES NOT NON-NUMERIC COLUMNS TO pd.Categorical types,
        # because unlike from_csv which is loading the data directly, we don't know if the user has already
        # formatted the dataset or not and don't want to undo anything that has been done
        :param dataset:
        :param target_variable:
        """
        self._dataset = dataset
        self._target_variable = target_variable

        self._numeric_features, self._categoric_features = \
            OOLearningHelpers.get_columns_by_type(data_dtypes=dataset.dtypes, target_variable=target_variable)

        self._is_target_numeric = OOLearningHelpers.is_series_numeric(self._dataset[self._target_variable])

    @classmethod
    def from_csv(cls, csv_file_path: str, target_variable: str) -> 'ExploreDatasetBase':
        """
        # TODO: DOCUMENT; THIS METHOD SETS NON-NUMERIC COLUMNS TO pd.Categorical types
        :param csv_file_path:
        :param target_variable:
        :return:
        """
        explore = cls(dataset=pd.read_csv(csv_file_path), target_variable=target_variable)

        _, categoric_features = OOLearningHelpers.get_columns_by_type(data_dtypes=explore.dataset.dtypes,
                                                                      target_variable=target_variable)
        for feature in categoric_features:
            explore.dataset[feature] = explore.dataset[feature].astype('category')

        return explore

    @property
    def dataset(self):
        return self._dataset

    @property
    def target_variable(self):
        return self._target_variable

    @property
    def numeric_features(self):
        return self._numeric_features

    @property
    def categoric_features(self):
        return self._categoric_features

    @property
    def numeric_summary(self):
        """
        #TODO: document
        https://www.quora.com/How-can-a-standard-deviation-divided-by-mean-be-useful
        :return:
        """

        numeric_columns = self._numeric_features + [self._target_variable] if self._is_target_numeric else self._numeric_features  # noqa

        # column, number of nulls in column, percent of nulls in column
        null_data = [(column, self._dataset[column].isnull().sum(), round(self._dataset[column].isnull().sum() / len(self._dataset), 3))  # noqa
                     for column in numeric_columns]
        columns, num_nulls, perc_nulls = zip(*null_data)

        # column, number of 0's, percent of 0's
        zeros_data = [(column, sum(self._dataset[column] == 0), round(sum(self._dataset[column] == 0) / len(self._dataset), 3))  # noqa
                      for column in numeric_columns]
        columns, num_zeros, perc_zeros = zip(*zeros_data)
        return pd.DataFrame({'count': [self._dataset[x].count() for x in numeric_columns],
                             'nulls': num_nulls,
                             'perc_nulls': perc_nulls,
                             'num_zeros': num_zeros,
                             'perc_zeros': perc_zeros,
                             'mean': [round(self._dataset[x].mean(), 3) for x in numeric_columns],
                             'st_dev': [round(self._dataset[x].std(), 3) for x in numeric_columns],
                             'coef of var': [round(self._dataset[x].std() / self._dataset[x].mean(), 3) for x in numeric_columns],  # noqa
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

    @property
    def categoric_summary(self):
        categoric_columns = self.categoric_features if self._is_target_numeric else self._categoric_features + [self._target_variable]  # noqa

        # column, number of nulls in column, percent of nulls in column
        null_data = [(column, self._dataset[column].isnull().sum(), round(self._dataset[column].isnull().sum() / len(self._dataset), 3))  # noqa
                     for column in categoric_columns]
        columns, num_nulls, perc_nulls = zip(*null_data)
        return pd.DataFrame({'count': [self._dataset[x].count() for x in categoric_columns],
                             'nulls': num_nulls,
                             'perc_nulls': perc_nulls,
                             'top': [self._dataset[x].value_counts().index[0] for x in categoric_columns],
                             'unique': [len(self._dataset[x].dropna().unique()) for x in categoric_columns],  # noqa
                             'perc_unique': [round(len(self._dataset[x].dropna().unique()) / self._dataset[x].count(), 3) for x in categoric_columns]},  # noqa
                            index=columns,
                            columns=['count', 'nulls', 'perc_nulls', 'top', 'unique', 'perc_unique'])

    def set_level_order(self, categoric_feature: str, levels: List):

        assert self._dataset[categoric_feature].dtype.name == 'category'  # must be a category
        self._dataset[categoric_feature].cat.reorder_categories(levels, inplace=True)

    def unique_values(self, categoric_feature: str, sort_by_feature=False):
        """
        # TODO: document
        :param categoric_feature:
        :param sort_by_feature: if `True`, then if the column is a `pd.Categorical`, then the order of the
            rows will be based on the order;
            If `False`, the data will be ordered by frequency
        :return:
        """
        # only for categoric features (and the target variable if it is categoric)
        valid_features = self._categoric_features if self._is_target_numeric else self._categoric_features + \
                                                                                  [self._target_variable]
        assert categoric_feature in valid_features
        count_series = self._dataset[categoric_feature].value_counts(sort=not sort_by_feature)
        count_df = pd.DataFrame(count_series)
        count_df['perc'] = (count_series.values / count_series.values.sum()).round(3)
        count_df.columns = ['freq', 'perc']

        return count_df

    def unique_values_bar(self, categoric_feature, sort_by_feature=False):
        """
        TODO: Document
        :param categoric_feature:
        :param sort_by_feature:
        :return:
        """
        unique_values = self.unique_values(categoric_feature=categoric_feature,
                                           sort_by_feature=sort_by_feature)

        # noinspection PyUnresolvedReferences
        ax = unique_values.drop(labels='perc', axis=1).plot(kind='bar', rot=10, title=categoric_feature)
        for idx, label in enumerate(list(unique_values.index)):
            freq = unique_values.loc[label, 'freq']
            perc = unique_values.loc[label, 'perc']

            ax.annotate(freq, (idx, freq), xytext=(-8, 2), textcoords='offset points')
            ax.annotate("{0:.0f}%".format(perc * 100), (idx, 2), xytext=(-8, 0), textcoords='offset points')

        # noinspection PyUnresolvedReferences
        # ax = unique_values.drop(labels='perc', axis=1).plot(kind='barh')
        # for idx, label in enumerate(list(unique_values.index)):
        #     freq = unique_values.loc[label, 'freq']
        #     perc = unique_values.loc[label, 'perc']
        #
        #     ax.annotate(freq, (freq, idx), xytext=(0, 0), textcoords='offset points')
        #     ax.annotate("{0:.0f}%".format(perc * 100), (0, idx - 0.03), xytext=(0, 0),
        #                 textcoords='offset points')

    def boxplot(self, numeric_feature):
        """
        # TODO:
        :param numeric_feature:
        :return:
        """
        valid_features = self._numeric_features + [self._target_variable] if self._is_target_numeric else self._numeric_features  # noqa
        assert numeric_feature in valid_features
        box_plot = self._dataset[numeric_feature].plot(kind='box')
        plt.title(numeric_feature)
        return box_plot

    def histogram(self, numeric_feature):
        # only for numeric features (and the target variable if it is numeric)
        valid_features = self._numeric_features + [self._target_variable] if self._is_target_numeric else self._numeric_features  # noqa
        assert numeric_feature in valid_features
        hist_plot = self._dataset[numeric_feature].hist()
        plt.title(numeric_feature)
        return hist_plot

    def scatter_plot_numerics(self, numeric_columns=None, figure_size=(12, 8)):
        """
        TODO document
        if numeric_columns is none, all numeric columns will be plotted.
        :param numeric_columns:
        :param figure_size:
        :return:
        """
        if numeric_columns is None:
            numeric_columns = self._numeric_features + [self._target_variable] if self._is_target_numeric else self._numeric_features  # noqa
        return scatter_matrix(self._dataset[numeric_columns], figsize=figure_size)

    def correlation_heatmap(self):
        f, ax = pl.subplots(figsize=(10, 8))
        corr = self._dataset.corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax,
                    center=0)
        plt.xticks(rotation=20)
        plt.title('correlations')
