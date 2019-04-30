from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Union

import numpy as np
import pandas as pd


class TunerResultsBase(metaclass=ABCMeta):

    def __init__(self,
                 tune_results: pd.DataFrame,
                 time_results: pd.DataFrame):
        self._tune_results_objects = tune_results

        results_list = list()
        # for each score, add the mean and standard deviation as a diction to the resampled_stats list
        for resampler_result in tune_results.resampler_object.tolist():
            score_dictionary = OrderedDict()
            for score in resampler_result.score_names:
                score_dictionary.update(
                    {score + '_mean': resampler_result.score_means[score],
                     score + '_st_dev': resampler_result.score_standard_deviations[score],
                     score + '_cv': resampler_result.score_coefficients_of_variation[score]})
            results_list.append(score_dictionary)

        # noinspection PyUnresolvedReferences
        self._tune_results_values = pd.concat([tune_results.copy().drop(columns='resampler_object'),
                                               pd.DataFrame(results_list)], axis=1)
        self._time_results = time_results

    @property
    def number_of_cycles(self) -> int:
        return self._tune_results_objects.shape[0]

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
