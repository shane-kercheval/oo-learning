from typing import List

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
    def metrics(self):
        assert self._cross_validation_scores is not None
        return list(self._cross_validation_scores[0].keys())

    @property
    def num_resamples(self):
        return len(self.cross_validation_scores)

    @property
    def evaluators(self):
        return self._evaluators

    @property
    def cross_validation_scores(self) -> pd.DataFrame:
        return pd.DataFrame(self._cross_validation_scores)

    @property
    def metric_means(self):
        return {metric: self.cross_validation_scores[metric].mean() for metric in self.metrics}

    @property
    def metric_standard_deviations(self):
        return {metric: self.cross_validation_scores[metric].std() for metric in self.metrics}

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
        better_than_function = self._evaluators[0][0].better_than_function
        # get the mean of the first (i.e. main) metric
        this_mean = self.metric_means[self.metrics[0]]
        # get the mean of the first (i.e. main) metric for the other ResamplerResult
        other_mean = other.metric_means[other.metrics[0]]

        return better_than_function(this_mean, other_mean)
