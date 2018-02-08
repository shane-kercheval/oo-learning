from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from oolearning.evaluators.EvaluatorBase import EvaluatorBase


class ResamplerResults:
    def __init__(self, evaluators: List[List[EvaluatorBase]]):
        """
        :param evaluators: a list of list of holdout_evaluators.
            each outer list represents a resampling result (e.g. a single fold for a single repeat in repeated
                k-fold cross validation);
            each element of the inner list represents a specific evaluator for the single resampling result
        """
        self._evaluators = evaluators

        # for each evaluator, add the metric name/value to a dict to add to the ResamplerResults
        self._cross_validation_scores = list()

        for resample_eval_list in evaluators:
            results_dict = dict()
            for temp_eval in resample_eval_list:
                results_dict[temp_eval.metric_name] = temp_eval.value

            self._cross_validation_scores.append(results_dict)

    @property
    def metrics(self) -> List[str]:
        assert self._cross_validation_scores is not None
        return list(self._cross_validation_scores[0].keys())

    @property
    def num_resamples(self) -> int:
        return len(self.cross_validation_scores)

    @property
    def evaluators(self) -> List[List[EvaluatorBase]]:
        return self._evaluators

    @property
    def cross_validation_scores(self) -> pd.DataFrame:
        return pd.DataFrame(self._cross_validation_scores, columns=self.metrics)

    def cross_validation_score_boxplot(self):
        plt.ylim(0.0, 1.0)
        plot = self.cross_validation_scores.boxplot()
        plt.title('Cross-Validation Scores')
        return plot

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
            more  variation, relative to its mean.
        """
        return {metric: round((self.cross_validation_scores[metric].std() /
                               self.cross_validation_scores[metric].mean()), 2) for metric in self.metrics}

    def __lt__(self, other):
        """
        # TODO: update documentation
        basically, i'm utilizing the `better_than` function passed to the Evaluator, in order to compare
            the means of all the Evaluators, in order to properly sort
             e.g. when comparing Kappas the larger number is "better", when comparing RMSE smaller numbers are
                "better"
        """
        # get the first evaluator's `better_than` function, and utilize compare the means associated with the
        # first evaluator
        # noinspection PyProtectedMember
        better_than_function = self._evaluators[0][0]._better_than
        # get the mean of the first (i.e. main) metric for *this* ResamplerResult
        this_mean = self.metric_means[self.metrics[0]]
        # get the mean of the first (i.e. main) metric for the *other* ResamplerResult
        other_mean = other.metric_means[other.metrics[0]]

        return better_than_function(this_mean, other_mean)
