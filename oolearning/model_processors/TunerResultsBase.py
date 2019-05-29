import math
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning.enums.Metric import Metric
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.model_processors.ResamplerResults import ResamplerResults


class TunerResultsBase(metaclass=ABCMeta):

    def __init__(self,
                 resampler_results: List[ResamplerResults],
                 hyper_params_combos: List[dict],
                 resampler_times: List[str]):

        self._resampler_results = resampler_results
        self._hyper_params_combos = hyper_params_combos
        self._resampler_times = resampler_times

        results_values = list()
        # for each score, add the mean and standard deviation as a diction to the resampled_stats list
        for resampler_result in resampler_results:
            score_dictionary = OrderedDict()
            for score in resampler_result.score_names:
                score_dictionary.update(
                    {score + '_mean': resampler_result.score_means[score],
                     score + '_st_dev': resampler_result.score_standard_deviations[score],
                     score + '_cv': resampler_result.score_coefficients_of_variation[score]})
            results_values.append(score_dictionary)

        params_combinations = pd.DataFrame({'hyper_params': ['None']}) \
            if hyper_params_combos is None or len(hyper_params_combos) == 0 \
            else pd.DataFrame(hyper_params_combos, columns=self.hyper_param_names)

        # # as a check, we want to make sure the tune_results and time_results dataframe doesn't contain any
        # # NAs; however, if we set a particular hyper-parameter to None, this will cause a false positive
        # # so, let's ignore any columns where the hyper-param is specifically set to None
        # if hyper_params_combos is not None and len(hyper_params_combos) > 0:
        #     # noinspection PyProtectedMember
        #     params_containing_none = [key for key, value in hyper_params_combos.items()
        #                               if value is None or (isinstance(value, list) and None in value)]
        # else:
        #     # might not have hyper-params (e.g. automating a process and encountering a model (e.g.
        #     # Regression) that doesn't have hyper-params
        #     params_containing_none = []

        self._tune_results_objects = pd.concat([params_combinations.copy(),
                                                pd.DataFrame(resampler_results,
                                                             columns=['resampler_object'])],
                                               axis=1)
        assert self._tune_results_objects.isnull().sum().sum() == 0

        self._time_results = pd.concat([params_combinations.copy(),
                                        pd.DataFrame(resampler_times, columns=['execution_time'])],
                                       axis=1)
        assert self._time_results.isnull().sum().sum() == 0

        self._tune_results_values = pd.concat([self._tune_results_objects.copy().drop(columns='resampler_object'),  # noqa
                                               pd.DataFrame(results_values)], axis=1)

    def __str__(self):
        val = "Best Hyper-Parameters\n=====================\n\n"
        val += str(self.best_hyper_params).replace(", ", "\n ")
        val += "\n\nTuner Results\n=============\n\n"

        temp = self.resampled_stats.copy().round(8)
        temp['rank'] = self._tune_results_objects.resampler_object.rank()

        return val + temp[['rank'] + list(self.resampled_stats.columns.values)].to_string()

    @property
    def number_of_cycles(self) -> int:
        return self._tune_results_objects.shape[0]

    @property
    def hyper_param_names(self):
        if self._hyper_params_combos:

            keys = [list(params.keys()) for params in self._hyper_params_combos]
            assert all([k == keys[0] for k in keys])  # all the keys should match
            names = keys[0]
        else:
            names = None

        return names

    @property
    def hyper_param_combos(self):
        return self._hyper_params_combos

    @property
    def resampled_stats(self) -> pd.DataFrame:
        """
        :return: dataframe that has score_names for each tuned model (rows), the means, standard deviations,
            and coefficient of variations for the corresponding resampler results (e.g. cross validation
            results), which are represented in the DataFrame columns.

        `_mean`: mean i.e. average of the resampler results
        `_st_dev`: standard deviation of resampler results
        `_cv`: coefficient of variation of resampler results

            If `sample A` has a CV of 0.12 and `sample B` has a CV of 0.25, you would say that `sample B` has
                more  variation, relative to its mean.

        """
        return self._tune_results_values

    @property
    def resampler_times(self) -> pd.DataFrame:
        return self._time_results

    @property
    def sorted_best_indexes(self):
        # sort based off of the first score.. each Resampler will have the same scores
        # works because __lt__ is implemented for ResamplerResults
        return np.argsort(self._tune_results_objects.resampler_object.values)

    @property
    def best_index(self):
        return self.sorted_best_indexes[0]

    @property
    def sorted_best_models(self) -> pd.DataFrame:
        return self.resampled_stats.iloc[self.sorted_best_indexes]

    @property
    def best_model_resampler_object(self):
        # get the first index ([0]) of the best indexes, and use that index to get the resampler_object
        return self._tune_results_objects.resampler_object.values[self.best_index]

    @property
    def best_model(self) -> pd.Series:
        return self.sorted_best_models.iloc[0]

    @property
    def resampler_decorators(self):
        # todo document, for each hyper-param combo that was resampled, return the associated decorators
        return [self._tune_results_objects.iloc[i].resampler_object.decorators for i in
                np.arange(0, self.number_of_cycles)]  # noqa

    @property
    def resampler_decorators_first(self):
        # TODO document, returns the first decorator for each hyper-param combo that was resampled
        if self.resampler_decorators:
            return [x[0] for x in self.resampler_decorators]
        else:
            return None

    @property
    @abstractmethod
    def best_hyper_params(self) -> Union[dict, None]:
        pass

    def plot_iteration_mean_scores(self):
        score_names = [score.name + "_mean" for score in self.best_model_resampler_object.scores[0]]
        stats = self.resampled_stats[score_names].copy()
        stats['Iteration'] = range(1, len(stats) + 1)
        stats.plot(x='Iteration')

    @staticmethod
    def columnwise_conditional_format(df,
                                      hyper_param_names,
                                      minimizers: List[bool],
                                      font_size: int = 8):
        """
        # TODO: document... minimizers i.e. CostFunction are blue ... maximizers i.e. UtilityFunctions
        # are green..... darker colors are "better".. so for "maximizers" it will be higher numbers
        # for minimizers it will be lower numbers, except that all st-dev and CV numbers, lower is better
        code copied from:
            https://stackoverflow.com/questions/44017205/apply-seaborn-heatmap-columnwise-on-pandas-dataframe
        """

        score_columns = [x for x in df.columns.values if x not in hyper_param_names]
        score_values = df.loc[:, score_columns]

        num_rows = len(score_values)
        num_cols = len(score_values.columns)

        n = 3  # i.e. number of score_names e.g. _mean, _st_dev, _cv
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(num_cols):
            truths = [True] * num_cols
            truths[i] = False
            mask = np.array(num_rows * [truths], dtype=bool)
            color_values = np.ma.masked_where(mask, score_values)
            # TODO document this behavior
            # a smaller number for standard deviation and coefficient of variation is ALWAYS better,
            # regardless if the associated score is a minimizer or maximizer (i.e. CostFunction or
            # UtilityFunction). In other words, smaller variation means more 'confidence' in the score.
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
                cell_value = score_values.iloc[row, column]
                plt.text(column + .5, row + .5, '%.3f' % cell_value,
                         fontsize=font_size,
                         horizontalalignment='center',
                         verticalalignment='center',
                         color=cell_color_chart[cell_value > column_mean][is_column_minimizer])

        ax.set_xticks(np.arange(start=0.5, stop=len(score_columns), step=1))
        ax.set_xticklabels(score_columns, rotation=35, ha='right', fontsize=font_size)

        param_combos = df.loc[:, hyper_param_names]
        # drop any columns (i.e. hyper-params) that only have 1 value (i.e. weren't really "tuned")
        unique_values_per_column = param_combos.apply(pd.Series.nunique)
        columns_to_drop = unique_values_per_column[unique_values_per_column == 1].index
        param_combos = param_combos.drop(columns_to_drop, axis=1)
        hyper_param_names = [x for x in hyper_param_names if x not in columns_to_drop]

        labels = []
        for index in range(len(param_combos)):
            labels.append(str(dict(zip(hyper_param_names, param_combos.iloc[index].values.tolist()))))

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

    def plot_resampled_stats(self, font_size: int = 8):
        """
        NOTE: only shows the "tuned" hyper-params i.e. hyper-params that were tuned over >1 values.
        :return:
        """

        if self.hyper_param_names is None or len(self.hyper_param_names) == 0:
            return None

        scores = self._tune_results_objects.iloc[0].resampler_object.scores[0]
        # if the Score is a Cost Function it is a 'minimizer'
        minimizers = [isinstance(x, CostFunctionMixin) for x in scores]

        # .tuned_hyper_params ensures only hyper-params with >1 values
        self.columnwise_conditional_format(df=self.resampled_stats.copy(),
                                           hyper_param_names=self.hyper_param_names,
                                           minimizers=minimizers,
                                           font_size=font_size)
        plt.tight_layout()

    def plot_resampled_scores(self,
                              metric: Metric = None,
                              score_name: str = None,
                              x_axis_limits: tuple = (0.0, 1.0),
                              show_one_ste_rule: bool = False):
        """
        NOTE: there is some odd/inconsistent reference lines/colors with this graph to be aware of:
            The params with the best MEAN are highlighted in red.
            The vertical line represents the params that give the best MEDIAN
            The blue vertical line represents one standard error below the MEAN.

        NOTE: only shows the "tuned" hyper-params i.e. hyper-params that were tuned over >1 values.
        :param metric: the metric (corresponding to the Score object) to display (use this parameter or
            `score_name`
        :param score_name: alternative to the `metric` parameter, you can specify the name of the score to
            retrieve; (the name corresponding to the `name` property of the Score object. While the `metric`
            parameter is a convenience when dealing with built in Scores, `score_name` can be used for custom
            score objects.
        :param x_axis_limits: limits for the x-axis
        :param show_one_ste_rule: show a blue line one standard error below the mean of the best model.
        """

        if self.hyper_param_names is None or len(self.hyper_param_names) == 0:
            return None

        assert metric is not None or score_name is not None

        param_combos = self.resampled_stats.loc[:, self.hyper_param_names]
        # drop any columns (i.e. hyper-params) that only have 1 value (i.e. weren't really "tuned")
        unique_values_per_column = param_combos.apply(pd.Series.nunique)
        columns_to_drop = unique_values_per_column[unique_values_per_column == 1].index.values

        metric_name = metric.value if score_name is None else score_name
        # build the dataframe that will be used to generate the boxplot; 1 column per resampled hyper-params
        resamples = pd.DataFrame()
        for index in range(self.number_of_cycles):
            cross_val_scores = self._tune_results_objects.iloc[index].loc['resampler_object'].\
                resampled_scores[metric_name]
            # column name should be the hyper_params & values
            column_name_dict = dict()
            # only want hyper-params with >1 values (i.e. that were actually "tuned")
            hyper_param_names = [x for x in self.hyper_param_names if x not in columns_to_drop]
            for hyper_param in hyper_param_names:
                column_name_dict[hyper_param] = self._tune_results_objects.iloc[index].loc[hyper_param]

            resamples[str(column_name_dict)] = pd.Series(data=cross_val_scores)

        # ensure correct number of models (columns in `resamples`, and rows in `resampled_stats`
        assert resamples.shape[1] == len(param_combos.drop_duplicates())
        # ensure correct number of resamples (rows in `resamples`, and row in `the underlying cross validation
        # scores (of resampled hyper-param))
        assert resamples.shape[0] == len(self._tune_results_objects.iloc[0].loc['resampler_object'].resampled_scores)  # noqa

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

        if show_one_ste_rule:
            # using means rather than medians because we are calculating standard error (from the mean)
            resamples_of_best_mean = resamples[resamples.columns.values[index_of_best_mean]].values
            one_standard_error_of_best = resamples_of_best_mean.std() / math.sqrt(len(resamples_of_best_mean))
            # noinspection PyUnresolvedReferences
            one_standard_error_rule = resamples_of_best_mean.mean() - one_standard_error_of_best

            resamples = resamples.iloc[:, resample_means > one_standard_error_rule]
            resamples.boxplot(vert=False, figsize=(10, 10))
            resample_medians = [resamples[column].median() for column in resamples.columns.values]
            plt.axvline(x=best(resample_medians), color='red', linewidth=1)
            plt.axvline(x=one_standard_error_rule, color='blue', linewidth=1)

            # have to recalculate resample_means and index_of_best_mean
            resample_means = [resamples[column].mean() for column in resamples.columns.values]
            index_of_best_mean = resample_means.index(best(resample_means))

        else:
            resamples.boxplot(vert=False, figsize=(10, 10))
            resample_medians = [resamples[column].median() for column in resamples.columns.values]
            plt.axvline(x=best(resample_medians), color='red', linewidth=1)

        # noinspection PyTypeChecker,PyUnresolvedReferences
        if (resamples <= 1).all().all():
            plt.xlim(x_axis_limits[0], x_axis_limits[1])
        plt.title('{0} ({1})'.format('Cross-Validation Scores Per Resampled Hyper-parameter',
                                     metric.name if score_name is None else score_name),
                  loc='right')
        plt.tight_layout()
        plt.gca().get_yticklabels()[index_of_best_mean].set_color('red')
        plt.gca().invert_yaxis()

    def plot_hyper_params_profile(self,
                                  x_axis,
                                  line=None,
                                  grid=None,
                                  metric: Metric = None,
                                  score_name: str = None):
        """
        :param x_axis: the hyper-parameter to place on the x-axis
        :param line: the hyper-parameter to show as lines on the graph
        :param grid: the hyper-parameter, such that when the plot displays, a grid is one, and one graph is
            created for each tuned value in the hyper-parameter.
        :param metric: the metric (corresponding to the Score object) to display (use this parameter or
            `score_name`
        :param score_name: alternative to the `metric` parameter, you can specify the name of the score to
            retrieve; (the name corresponding to the `name` property of the Score object. While the `metric`
            parameter is a convenience when dealing with built in Scores, `score_name` can be used for custom
            score objects.
        """

        if self.hyper_param_names is None or len(self.hyper_param_names) == 0:
            return None

        assert metric is not None or score_name is not None

        if grid is not None:
            assert line is not None  # can't have grid without line

        score_value = metric.value + '_mean' if score_name is None else score_name + '_mean'

        if line is None:  # then we also know grid is None as well
            hyper_params = [x_axis]
            df = self.resampled_stats[hyper_params + [score_value]]
            ax = df.groupby(hyper_params).mean().plot(figsize=(10, 7))
            x_limits = ax.get_xlim()
            padding = (x_limits[1] - x_limits[0]) * 0.05
            ax.set_xlim(x_limits[0] - padding, x_limits[1] + padding)

        elif grid is None:  # then we know line is NOT None, but grid is,
            hyper_params = [x_axis, line]
            df = self.resampled_stats[hyper_params + [score_value]]
            df_grouped = df.groupby(hyper_params).mean()
            plot_df = df_grouped.unstack(line).loc[:, score_value]
            ax = plot_df.plot(figsize=(10, 7))
            ax.set_ylabel(score_value)
            x_limits = ax.get_xlim()
            padding = (x_limits[1] - x_limits[0]) * 0.05
            ax.set_xlim(x_limits[0] - padding, x_limits[1] + padding)

        else:  # line and grid are not None
            hyper_params = [grid, line, x_axis]
            df = self.resampled_stats
            df_grouped = df.groupby(hyper_params)[[score_value]].mean()

            # We can ask for ALL THE AXES and put them into axes
            num_rows = math.ceil(len(hyper_params) / 2)
            # noinspection PyTypeChecker
            fig, axes = plt.subplots(nrows=num_rows, ncols=2, sharex=True, sharey=True, figsize=(10, 8))
            axes_list = [item for sublist in axes for item in sublist]

            for index in df_grouped.index.levels[0].values:
                ax = axes_list.pop(0)
                df_grouped.loc[index].unstack(line).loc[:, score_value].plot(ax=ax)
                ax.set_ylabel(score_value)
                ax.set_title('{0}={1}'.format(grid, index))
                x_limits = ax.get_xlim()
                padding = (x_limits[1] - x_limits[0]) * 0.05
                ax.set_xlim(x_limits[0] - padding, x_limits[1] + padding)
                # ax.tick_params(axis='both', which='both')

            # Now use the matplotlib .remove() method to
            # delete anything we didn't use
            for ax in axes_list:
                ax.remove()

            plt.tight_layout()
