from abc import ABCMeta

import pandas as pd

from oolearning import OOLearningHelpers


class ExploreDatasetBase(metaclass=ABCMeta):
    def __init__(self, dataset: pd.DataFrame, target_variable: str):
        self._dataset = dataset
        self._target_variable = target_variable

        self._numeric_features, self._categoric_features = \
            OOLearningHelpers.get_columns_by_type(data_dtypes=dataset.dtypes, target_variable=target_variable)

        self._is_target_numeric = OOLearningHelpers.is_series_numeric(self._dataset[self._target_variable])

    @classmethod
    def from_csv(cls, csv_file_path: str, target_variable: str) -> 'ExploreDatasetBase':
        return cls(dataset=pd.read_csv(csv_file_path), target_variable=target_variable)

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

    def unique_values(self, categoric_feature: str):
        assert (categoric_feature in self._categoric_features) or categoric_feature == self._target_variable
        count_series = self._dataset[categoric_feature].value_counts()
        count_df = pd.DataFrame(count_series)
        count_df['perc'] = (count_series.values / count_series.values.sum()).round(3)
        count_df.columns = ['freq', 'perc']
        return count_df
