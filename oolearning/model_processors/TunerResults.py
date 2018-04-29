from collections import OrderedDict
from typing import List

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning.enums.Metric import Metric

from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.model_wrappers import HyperParamsGrid


# TODO: update documentation
class TunerResults:
    def __init__(self,
                 tune_results: pd.DataFrame,
                 time_results: pd.DataFrame,
                 params_grid: HyperParamsGrid):
        self._tune_results_objects = tune_results

        results_list = list()
        # for each metric, add the mean and standard deviation as a diction to the tune_results list
        for resampler_result in tune_results.resampler_object.tolist():
            metric_dictionary = OrderedDict()
            for metric in resampler_result.metrics:
                metric_dictionary.update(
                    {metric + '_mean': resampler_result.metric_means[metric],
                     metric + '_st_dev': resampler_result.metric_standard_deviations[metric],
                     metric + '_cv': resampler_result.metric_coefficient_of_variation[metric]})
            results_list.append(metric_dictionary)

        # noinspection PyUnresolvedReferences
        self._tune_results_values = pd.concat([tune_results.copy().drop(columns='resampler_object'),
                                               pd.DataFrame(results_list)], axis=1)
        self._time_results = time_results
        self._params_grid = params_grid

    @property
    def num_param_combos(self) -> int:
        return self._tune_results_objects.shape[0]

    @property
    def tune_results(self) -> pd.DataFrame:
        """
        :return: dataframe that has metrics for each tuned model (rows), the means, standard deviations, and
            coefficient of variations for the corresponding resampler results (e.g. cross validation results),
            which are represented in the DataFrame columns.

        `_mean`: mean i.e. average of the resampler results
        `_st_dev`: standard deviation of resampler results
        `_cv`: coefficient of variation of resampler results

            If `sample A` has a CV of 0.12 and `sample B` has a CV of 0.25, you would say that `sample B` has
                more  variation, relative to its mean.

        """
        return self._tune_results_values

    @property
    def time_results(self) -> pd.DataFrame:
        return self._time_results

    @property
    def sorted_best_indexes(self):
        # sort based off of the first score.. each Resampler will have the same scores
        return np.argsort(self._tune_results_objects.resampler_object.values)

    @property
    def best_index(self):
        return self.sorted_best_indexes[0]

    @property
    def sorted_best_models(self) -> pd.DataFrame:
        return self.tune_results.iloc[self.sorted_best_indexes]

    @property
    def best_model_resampler_object(self):
        # get the first index ([0]) of the best indexes, and use that index to get the resampler_object
        return self._tune_results_objects.resampler_object.values[self.best_index]

    @property
    def best_model(self) -> pd.Series:
        return self.sorted_best_models.iloc[0]

    @property
    def best_hyper_params(self) -> dict:
        return \
            None if self._params_grid is None \
            else self.sorted_best_models.loc[:, self._params_grid.hyper_params].iloc[0].to_dict()

    @property
    def resampler_decorators(self):
        # todo document, for each hyper-param combo that was resampled, return the associated decorators
        return [self._tune_results_objects.iloc[i].resampler_object.decorators for i in np.arange(0, self.num_param_combos)]  # noqa

    @property
    def resampler_decorators_first(self):
        # TODO document, returns the first decorator for each hyper-param combo that was resampled
        if self.resampler_decorators:
            return [x[0] for x in self.resampler_decorators]
        else:
            return None

    @staticmethod
    def columnwise_conditional_format(df,
                                      hyper_params,
                                      tuned_hyper_params,
                                      minimizers: List[bool],
                                      font_size: int=8):
        """
        # TODO: document... minimizers i.e. CostFunction are blue ... maximizers i.e. UtilityFunctions
        # are green..... darker colors are "better".. so for "maximizers" it will be higher numbers
        # for minimizers it will be lower numbers, except that all st-dev and CV numbers, lower is better
        code copied from:
            https://stackoverflow.com/questions/44017205/apply-seaborn-heatmap-columnwise-on-pandas-dataframe
        """

        score_columns = [x for x in df.columns.values if x not in hyper_params]
        score_values = df.loc[:, score_columns]

        num_rows = len(score_values)
        num_cols = len(score_values.columns)

        n = 3  # i.e. number of metrics e.g. _mean, _st_dev, _cv
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(num_cols):
            truths = [True] * num_cols
            truths[i] = False
            mask = np.array(num_rows * [truths], dtype=bool)
            color_values = np.ma.masked_where(mask, score_values)
            # TODO document this behavior
            # a smaller number for standard deviation and coefficient of variation is ALWAYS better,
            # regardless if the associated metric is a minimizer or maximizer (i.e. CostFunction or
            # UtilityFunction). In other words, smaller variation means more 'confidence' in the metric.
            # If you have a slightly higher mean, but the numbers are much more variable than another sample
            # with a slightly lower mean with more less variation, it may be desirable to choose the slightly
            # lower mean that has less variation.
            is_minimizer = minimizers[int(math.floor(i/n))]
            is_std_or_cv = '_st_dev' in score_columns[i] or '_cv' in score_columns[i]
            colors = 'Blues' if minimizers[int(math.floor(i/n))] else 'Greens'
            # "_r" value after color means invert colors (small values are darker)
            # we want small colors for everything except maximizer and if it is not standard deviation
            # or coefficient of variation
            colors = colors if not is_minimizer and not is_std_or_cv else colors + '_r'
            ax.pcolormesh(color_values, cmap=colors)

        # outer dict is higher/lower than mean (true/false); inner dict is minimizer/maximizer;
        # minimizer has dark color for low value
        cell_color_chart = {True: {True: 'black', False: 'w'}, False: {True: 'w', False: 'black'}}
        for column in range(score_values.shape[1]):
            # if the column is a standard deviation or coefficient of variation, it is a minimizer by
            # definition; if it's not, we need to see if the specific Score is a minimizer/maximizer
            is_column_minimizer = '_st_dev' in score_columns[column] or '_cv' in score_columns[column]
            if not is_column_minimizer:
                is_column_minimizer = minimizers[int(math.floor(column/n))]

            column_mean = score_values[score_columns[column]].mean()
            for row in range(score_values.shape[0]):
                cell_value = score_values.ix[row, column]
                plt.text(column + .5, row + .5, '%.3f' % cell_value,
                         fontsize=font_size,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=cell_color_chart[cell_value > column_mean][is_column_minimizer])

        ax.set_xticks(np.arange(start=0.5, stop=len(score_columns), step=1))
        ax.set_xticklabels(score_columns, rotation=35, ha='right', fontsize=font_size)

        param_combos = df.loc[:, tuned_hyper_params]
        labels = []
        for index in range(len(param_combos)):
            labels.append(str(dict(zip(tuned_hyper_params, param_combos.iloc[index].values.tolist()))))

        y_tick_positions = np.arange(start=0, stop=len(param_combos)) + 0.5
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(labels, fontsize=font_size)

        # the first Score (and therefore the first item in `minimizers`, and first column in score_values
        # determines what params are considered the "best"
        minimizer = minimizers[0]
        best = min if minimizer else max
        index_of_best_mean = list(score_values[score_columns[0]]).index(best(score_values[score_columns[0]]))
        plt.gca().get_yticklabels()[index_of_best_mean].set_color('red')
        ax.invert_yaxis()
        plt.tight_layout()
        return plt

    def get_heatmap(self, font_size: int=8):
        """
        NOTE: only shows the "tuned" hyper-params i.e. hyper-params that were tuned over >1 values.
        :return:
        """
        if self._params_grid is None:  # if there are no hyper-params, no need for a heatmap.
            return None
        scores = self._tune_results_objects.iloc[0].resampler_object.scores[0]
        # if the Score is a Cost Function it is a 'minimizer'
        minimizers = [isinstance(x, CostFunctionMixin) for x in scores]

        # .tuned_hyper_params ensures only hyper-params with >1 values
        self.columnwise_conditional_format(df=self.tune_results,
                                           hyper_params=self._params_grid.hyper_params,
                                           tuned_hyper_params=self._params_grid.tuned_hyper_params,
                                           minimizers=minimizers,
                                           font_size=font_size)

    def get_cross_validation_boxplots(self, metric: Metric):
        """
        NOTE: only shows the "tuned" hyper-params i.e. hyper-params that were tuned over >1 values.
        :return:
        """
        if self._params_grid is None:  # if there are no hyper-params, no need for a heatmap.
            return None

        metric_name = metric.value
        # build the dataframe that will be used to generate the boxplot; 1 column per resampled hyper-params
        resamples = pd.DataFrame()
        for index in range(self.num_param_combos):
            cross_val_scores = self._tune_results_objects.iloc[index].loc['resampler_object'].\
                cross_validation_scores[metric_name]
            # column name should be the hyper_params & values
            column_name_dict = dict()
            # .tuned_hyper_params ensures only hyper-params with >1 values
            for hyper_param in self._params_grid.tuned_hyper_params:
                column_name_dict[hyper_param] = self._tune_results_objects.iloc[index].loc[hyper_param]

            resamples[str(column_name_dict)] = pd.Series(data=cross_val_scores)

        # ensure correct number of models (columns in `resamples`, and rows in `tune_results`
        assert resamples.shape[1] == len(self.tune_results)
        # ensure correct number of resamples (rows in `resamples`, and row in `the underlying cross validation
        # scores (of resampled hyper-param))
        assert resamples.shape[0] == len(self._tune_results_objects.iloc[0].loc['resampler_object'].cross_validation_scores)  # noqa

        # get the means to determine the 'best' hyper-param combo
        resample_means = [resamples[column].mean() for column in resamples.columns.values]
        assert len(resample_means) == resamples.shape[1]

        # get the current score object so we can determine if it is a minimizer or maximizer
        score = [x for x in self._tune_results_objects.iloc[0].resampler_object.scores[0]
                     if x.name == metric_name]
        assert len(score) == 1  # we should just get the current score
        # if the `better_than` function returns True, 0 is "better than" 1 and we have a minimizer
        # for minimizers, we want to return the min, which is the best value, otherwise, return the max
        minimizer = isinstance(score[0], CostFunctionMixin)
        best = min if minimizer else max
        index_of_best_mean = resample_means.index(best(resample_means))

        resamples.boxplot(vert=False, figsize=(10, 10))
        plt.xlim(0.0, 1.0)
        plt.title('{0} ({1})'.format('Cross-Validation Scores Per Resampled Hyper-parameter', metric.name),
                  loc='right')
        plt.tight_layout()
        plt.gca().get_yticklabels()[index_of_best_mean].set_color('red')
        plt.gca().invert_yaxis()

    def get_profile_hyper_params(self, metric: Metric, x_axis, line=None, grid=None):
        if grid is not None:
            assert line is not None  # can't have grid without line

        metric_value = metric.value + '_mean'

        if line is None:  # then we also know grid is None as well
            hyper_params = [x_axis]
            df = self.tune_results[hyper_params + [metric_value]]
            df.groupby(hyper_params).mean().plot(figsize=(10, 7))

        elif grid is None:  # then we know line is NOT None, but grid is,
            hyper_params = [x_axis, line]
            df = self.tune_results[hyper_params + [metric_value]]
            df_grouped = df.groupby(hyper_params).mean()
            plot_df = df_grouped.unstack(line).loc[:, metric_value]
            ax = plot_df.plot(figsize=(10, 7))
            ax.set_ylabel(metric_value)

        else:  # line and grid are not None
            hyper_params = [grid, line, x_axis]
            df = self.tune_results
            df_grouped = df.groupby(hyper_params)[[metric_value]].mean()

            # We can ask for ALL THE AXES and put them into axes
            num_rows = math.ceil(len(hyper_params) / 2)
            # noinspection PyTypeChecker
            fig, axes = plt.subplots(nrows=num_rows, ncols=2, sharex=True, sharey=True, figsize=(10, 8))
            axes_list = [item for sublist in axes for item in sublist]

            for index in df_grouped.index.levels[0].values:
                ax = axes_list.pop(0)
                df_grouped.loc[index].unstack(line).loc[:, metric_value].plot(ax=ax)
                ax.set_ylabel(metric_value)
                ax.set_title('{0}={1}'.format(grid, index))
                # ax.tick_params(axis='both', which='both')

            # Now use the matplotlib .remove() method to
            # delete anything we didn't use
            for ax in axes_list:
                ax.remove()

            plt.tight_layout()
