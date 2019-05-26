from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Union, List

import numpy as np
import pandas as pd

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
        else:
            keys = None

        return keys[0]

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
