from collections import OrderedDict
from typing import List

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from oolearning.enums.Metric import Metric


# TODO: update documentation
class TunerResults:
    def __init__(self, tune_results: pd.DataFrame, time_results: pd.DataFrame, hyper_params: list):
        self._tune_results_objects = tune_results

        results_list = list()
        # for each metric, add the mean and standard deviation as a diction to the tune_results list
        for resampler_result in tune_results.resampler_object.tolist():
            metric_dictionary = OrderedDict()
            for metric in resampler_result.metrics:
                metric_dictionary.update(
                    {metric + '_mean': resampler_result.metric_means[metric],
                     metric + '_st_dev': resampler_result.metric_standard_deviations[metric]})
            results_list.append(metric_dictionary)

        # noinspection PyUnresolvedReferences
        self._tune_results_values = pd.concat([tune_results.copy().drop(columns='resampler_object'),
                                               pd.DataFrame(results_list)], axis=1)
        self._time_results = time_results
        self._hyper_params = hyper_params

    @property
    def tune_results(self) -> pd.DataFrame:
        return self._tune_results_values

    @property
    def time_results(self) -> pd.DataFrame:
        return self._time_results

    @property
    def sorted_best_models(self) -> pd.DataFrame:
        # sort based off of the first Evaluator.. each Resampler will have the same evaluators
        indexes = np.argsort(self._tune_results_objects.resampler_object.values)
        return self.tune_results.iloc[indexes]

    @property
    def best_model(self) -> pd.Series:
        return self.sorted_best_models.iloc[0]

    @property
    def best_hyper_params(self) -> dict:
        return \
            None if self._hyper_params is None \
            else self.sorted_best_models.loc[:, self._hyper_params].iloc[0].to_dict()

    @staticmethod
    def columnwise_conditional_format(df, hyper_params, minimizers: List[bool]):
        """
        code copied from:
            https://stackoverflow.com/questions/44017205/apply-seaborn-heatmap-columnwise-on-pandas-dataframe
        """

        evaluator_columns = [x for x in df.columns.values if x not in hyper_params]
        evaluator_values = df.loc[:, evaluator_columns]

        num_rows = len(evaluator_values)
        num_cols = len(evaluator_values.columns)

        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(num_cols):
            truths = [True] * num_cols
            truths[i] = False
            mask = np.array(num_rows * [truths], dtype=bool)
            color_values = np.ma.masked_where(mask, evaluator_values)
            # "_r" value after color means invert colors (small values are darker)
            ax.pcolormesh(color_values, cmap='Blues_r' if minimizers[int(math.floor(i/2))] else 'Greens')

        for y in range(evaluator_values.shape[0]):
            for x in range(evaluator_values.shape[1]):
                plt.text(x + .5, y + .5, '%.3f' % evaluator_values.ix[y, x],
                         horizontalalignment='center',
                         verticalalignment='center')

        ax.set_xticks(np.arange(start=0.5, stop=len(evaluator_columns), step=1))
        ax.set_xticklabels(evaluator_columns, rotation=35, ha='right')

        param_combos = df.loc[:, hyper_params]
        labels = []
        for index in range(len(param_combos)):
            labels.append(str(dict(zip(hyper_params, param_combos.iloc[index].values.tolist()))))

        y_tick_positions = np.arange(start=0, stop=len(param_combos)) + 0.5
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(labels)
        len(labels)
        plt.tight_layout()
        return plt

    def get_heatmap(self):
        if self._hyper_params is None:  # if there are no hyper-params, no need for a heatmap.
            return None
        evaluators = self._tune_results_objects.iloc[0].resampler_object.evaluators[0]
        # if the `better_than` function returns True, 0 is "better than" 1 and we have a minimizer
        minimizers = [x.better_than_function(0, 1) for x in evaluators]

        return self.columnwise_conditional_format(df=self.tune_results,
                                                  hyper_params=self._hyper_params,
                                                  minimizers=minimizers)

    def get_cross_validation_boxplots(self, metric: Metric):
        metric_name = metric.value
        # build the dataframe that will be used to generate the boxplot; 1 column per resampled hyper-params
        resamples = pd.DataFrame()
        for index in range(len(self._tune_results_objects)):
            cross_val_scores = self._tune_results_objects.iloc[index].loc['resampler_object'].\
                cross_validation_scores[metric_name]
            # column name should be the hyper_params & values
            column_name_dict = dict()
            for hyper_param in self._hyper_params:
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

        # get the current evaluator object so we can determine if it is a minimizer or maximizer
        evaluator = [x for x in self._tune_results_objects.iloc[0].resampler_object.evaluators[0]
                     if x.metric_name == metric_name]
        assert len(evaluator) == 1  # we should just get the current evaluator
        # if the `better_than` function returns True, 0 is "better than" 1 and we have a minimizer
        # for minimizers, we want to return the min, which is the best value, otherwise, return the max
        minimizer = evaluator[0].better_than_function(0, 1)
        best = min if minimizer else max
        index_of_best_mean = resample_means.index(best(resample_means))

        resample_boxplot = resamples.boxplot(vert=False, figsize=(10, 10))
        plt.xlim(0.0, 1.0)
        plt.title('{0} ({1})'.format('Cross-Validation Scores Per Resampled Hyper-parameter', metric.name),
                  loc='right')
        plt.tight_layout()
        plt.gca().get_yticklabels()[index_of_best_mean].set_color('red')
        return resample_boxplot
