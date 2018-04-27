from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.DecoratorBase import DecoratorBase


class ResamplerResults:
    def __init__(self, scores: List[List[ScoreBase]], decorators: Union[List[DecoratorBase], None]):
        """
        :param scores: a list of list of holdout_scores.
            each outer list represents a resampling result (e.g. a single fold for a single repeat in repeated
                k-fold cross validation);
            each element of the inner list represents a specific score for the single resampling result
        #TODO: decorators are a list of decorators passed into the Resampler
        """
        self._scores = scores
        self._decorators = decorators

        # for each score, add the metric name/value to a dict to add to the ResamplerResults
        self._cross_validation_scores = list()

        for resample_eval_list in scores:
            results_dict = dict()
            for temp_eval in resample_eval_list:
                results_dict[temp_eval.name] = temp_eval.value

            self._cross_validation_scores.append(results_dict)

    @property
    def metrics(self) -> List[str]:
        assert self._cross_validation_scores is not None
        return list(self._cross_validation_scores[0].keys())

    @property
    def num_resamples(self) -> int:
        return len(self.cross_validation_scores)

    @property
    def scores(self) -> List[List[ScoreBase]]:
        return self._scores

    @property
    def cross_validation_scores(self) -> pd.DataFrame:
        return pd.DataFrame(self._cross_validation_scores, columns=self.metrics)

    def cross_validation_score_boxplot(self):
        plt.ylim(0.0, 1.0)
        self.cross_validation_scores.boxplot()
        plt.title('Cross-Validation Scores')

    @property
    def metric_means(self) -> dict:
        """
        :return: mean i.e. average for each metric/Evaluator
        :return:
        """
        return {metric: self.cross_validation_scores[metric].mean() for metric in self.metrics}

    @property
    def metric_standard_deviations(self) -> dict:
        """
        :return: standard deviation for each metric/Evaluator
        """
        return {metric: self.cross_validation_scores[metric].std() for metric in self.metrics}

    @property
    def metric_coefficient_of_variation(self) -> dict:
        """
        :return: `coefficient of variation` for each metric/Evaluator

         If `sample A` has a CV of 0.12 and `sample B` has a CV of 0.25, you would say that `sample B` has
            more variation, relative to its mean.
        """
        return {metric:
                np.nan if self.cross_validation_scores[metric].mean() == 0
                else round((self.cross_validation_scores[metric].std() /
                            self.cross_validation_scores[metric].mean()), 2)
                for metric in self.metrics}

    @property
    def decorators(self) -> List[DecoratorBase]:
        return self._decorators

    def __lt__(self, other):
        """
        # TODO: update documentation
        basically, i'm utilizing the `better_than` function passed to the Evaluator, in order to compare
            the means of all the Evaluators, in order to properly sort
             e.g. when comparing Kappas the larger number is "better", when comparing RMSE smaller numbers are
                "better"
        """
        # get the first score's `better_than` function, and utilize compare the means associated with the
        # first score
        # noinspection PyProtectedMember
        better_than_function = self._scores[0][0]._better_than
        # get the mean of the first (i.e. main) metric for *this* ResamplerResult
        this_mean = self.metric_means[self.metrics[0]]
        # get the mean of the first (i.e. main) metric for the *other* ResamplerResult
        other_mean = other.metric_means[other.metrics[0]]

        return better_than_function(this_mean, other_mean)
