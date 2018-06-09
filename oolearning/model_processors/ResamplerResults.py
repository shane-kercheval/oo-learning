from typing import List, Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase


class ResamplerResults:
    """
    Class that encapsulates the results of resampling a model.
    """
    def __init__(self,
                 scores: List[List[ScoreBase]],
                 decorators: Union[List[DecoratorBase], None],
                 hyper_params: Union[HyperParamsBase, None]):
        """
        :param scores: a list of list of holdout_score_objects.
            each outer list represents a resampling result (e.g. a single fold for a single repeat in repeated
                k-fold cross validation);
            each element of the inner list represents a specific score for the single resampling result
        :param decorators: the list of decorators passed into the Resampler and evaluated after each
            model is trained
        """
        self._scores = scores
        self._decorators = decorators
        self._hyper_params = hyper_params

        # for each score, add the score name/value to a dict to add to the ResamplerResults
        self._resampled_scores = list()

        for resample_eval_list in scores:
            results_dict = dict()
            for temp_eval in resample_eval_list:
                results_dict[temp_eval.name] = temp_eval.value

            self._resampled_scores.append(results_dict)

    @property
    def hyper_params(self):
        return self._hyper_params

    @property
    def score_names(self) -> List[str]:
        """
        :return: the names of the score_names being evaluated
        """
        assert self._resampled_scores is not None
        return list(self._resampled_scores[0].keys())

    @property
    def num_resamples(self) -> int:
        """
        :return: the number of resamples i.e. number of times model was trained on resampled data
        """
        return len(self.resampled_scores)

    @property
    def scores(self) -> List[List[ScoreBase]]:
        """
        :return: a list of list of holdout_score_objects.
            each outer list represents a resampling result (e.g. a single fold for a single repeat in repeated
                k-fold cross validation);
            each element of the inner list represents a specific score for the single resampling result
        """
        return self._scores

    @property
    def resampled_scores(self) -> pd.DataFrame:
        """
        :return: a DataFrame showing the resampled scores for each Score object, per resample.
            For example, for a `RepeatedCrossValidationResampler` there should be 1 row for each fold,
            multiplied by the number of repeats (e.g. a 5-fold, 5-repeat cross validation resampler would have
             25 rows; with each column corresponding to the Score objects that were passed in as a list to the
             Resampler.
        """
        return pd.DataFrame(self._resampled_scores, columns=self.score_names)

    def plot_resampled_scores(self):
        """
        :return: boxplot visualization for Score
        """
        # noinspection PyTypeChecker,PyUnresolvedReferences
        # if all scores are >=1 (e.g. classification) then place on 0-1 scale
        if (self.resampled_scores <= 1).all().all():
            plt.ylim(0.0, 1.0)
        self.resampled_scores.boxplot()
        plt.title('Resampled Scores')

    @property
    def score_means(self) -> dict:
        """
        :return: mean for each Score across all the resampled results
        """
        return {score: self.resampled_scores[score].mean() for score in self.score_names}

    @property
    def score_standard_deviations(self) -> dict:
        """
        :return: standard deviation for each Score across all the resampled results
        """
        return {score: self.resampled_scores[score].std() for score in self.score_names}

    @property
    def score_coefficients_of_variation(self) -> dict:
        """
        :return: `coefficient of variation` for each Score across all the resampled results

         If `sample A` has a CV of 0.12 and `sample B` has a CV of 0.25, you would say that `sample B` has
            more variation, relative to its mean.
        """
        return {score:
                np.nan if self.resampled_scores[score].mean() == 0
                else round((self.resampled_scores[score].std() /
                            self.resampled_scores[score].mean()), 2)
                for score in self.score_names}

    @property
    def score_stats(self) -> pd.DataFrame:
        stats = pd.DataFrame([self.score_means,
                              self.score_standard_deviations,
                              self.score_coefficients_of_variation],
                             index=['means', 'standard deviations', 'coefficients of variation'])
        return stats.reindex(columns=self.score_names)

    @property
    def decorators(self) -> List[DecoratorBase]:
        """
        :return: the list of decorators passed into the Resampler and evaluated after each model is trained
        """
        return self._decorators

    def __lt__(self, other):
        """
        this method is used to sort Resampled results based on the whether the Score object (and therefore,
            how we would define the "best" results, as well as how we would sort them) is based on a `utility`
            function or a `cost` function.
             e.g. when comparing Kappas the larger number is "better", when comparing RMSE smaller numbers are
                "better"
        """
        # get the first score's `better_than` function, and utilize compare the means associated with the
        # first score
        # noinspection PyProtectedMember
        better_than_function = self._scores[0][0]._better_than
        # get the mean of the first (i.e. main) score for *this* ResamplerResult
        this_mean = self.score_means[self.score_names[0]]
        # get the mean of the first (i.e. main) score for the *other* ResamplerResult
        other_mean = other.score_means[other.score_names[0]]

        return better_than_function(this_mean, other_mean)
