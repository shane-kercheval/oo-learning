import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as pl

from oolearning.transformers.StatelessTransformer import StatelessTransformer
from oolearning.transformers.TransformerPipeline import TransformerPipeline


class OOLearningHelpers:

    _numeric_dtypes = [np.dtype('float64'), np.dtype('int64'), np.dtype('uint8')]

    @staticmethod
    def is_series_numeric(variable: pd.Series):
        return variable.dtype in OOLearningHelpers._numeric_dtypes

    @staticmethod
    def is_series_dtype_numeric(dtype: np.dtype):
        return dtype in OOLearningHelpers._numeric_dtypes

    @staticmethod
    def is_series_boolean(variable: pd.Series):
        return isinstance(variable.values[0], bool) or isinstance(variable.values[0], np.bool_)

    @staticmethod
    def get_columns_by_type(data_dtypes: pd.Series, target_variable: str=None):
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

    @staticmethod
    def plot_correlations(correlations: pd.DataFrame, title: str=None, mask_duplicates=True):
        if title is None:
            title = ''

        if mask_duplicates:
            mask = np.zeros_like(correlations)
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = np.zeros_like(correlations, dtype=np.bool)

        with sns.axes_style("white"):
            f, ax = pl.subplots(figsize=(10, 8))
            sns.heatmap(correlations,
                        mask=mask,
                        annot=True,
                        cmap=sns.diverging_palette(220, 10, as_cmap=True),
                        square=True, ax=ax,
                        center=0)
            plt.xticks(rotation=20, ha='right')
            plt.title(title)

    @staticmethod
    def get_final_datasets(data, target_variable, splitter, transformations):

        # if we have a splitter, split into training and holdout, else just do transformations on all data
        if splitter:
            training_indexes, holdout_indexes = splitter.split(target_values=data[target_variable])
        else:
            training_indexes, holdout_indexes = range(len(data)), []

        training_y = data.iloc[training_indexes][target_variable]
        training_x = data.iloc[training_indexes].drop(columns=target_variable)

        holdout_y = data.iloc[holdout_indexes][target_variable]
        holdout_x = data.iloc[holdout_indexes].drop(columns=target_variable)

        # transform on training data
        if transformations is not None:
            # before we train the data, we actually want to 'snoop' at what the expected columns will be with
            # ALL the data. The reason is that if we so some sort of dummy encoding, but not all the
            # categories are included in the training set (i.e. maybe only a small number of observations have
            # the categoric value), then we can still ensure that we will be giving the same expected columns/
            # encodings to the predict method with the holdout set.
            # noinspection PyTypeChecker
            expected_columns = TransformerPipeline.get_expected_columns(data=data.drop(columns=target_variable),  # noqa
                                                                        transformations=transformations)
            transformer = StatelessTransformer(custom_function=lambda x_df: x_df.reindex(columns=expected_columns,  # noqa
                                                                                         fill_value=0))
            transformations = transformations + [transformer]

        pipeline = TransformerPipeline(transformations=transformations)
        # before we fit the data, we actually want to 'peak' at what the expected columns will be with
        # ALL the data. The reason is that if we so some sort of encoding (dummy/one-hot), but not all
        # of the categories are included in the training set (i.e. maybe only a small number of
        # observations have the categoric value), then we can still ensure that we will be giving the
        # same expected columns/encodings to the `predict` method with the holdout set.

        # peak at all the data (except for the target variable of course)
        # noinspection PyTypeChecker
        pipeline.peak(data_x=data.drop(columns=target_variable))

        # fit on only the train data-set (and also transform)
        transformed_training_x = pipeline.fit_transform(training_x)

        if holdout_indexes:
            transformed_holdout_x = pipeline.transform(holdout_x)
        else:
            transformed_holdout_x = holdout_x

        return transformed_training_x, training_y, transformed_holdout_x, holdout_y, pipeline
