from typing import List

import numpy as np
import pandas as pd

from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.TunerResults import TunerResults


class SearcherResults:
    def __init__(self,
                 model_descriptions: List[str],
                 model_names: List[str],
                 tuner_results: List[TunerResults],
                 holdout_scores=List[List[ScoreBase]]):
        """
        :param tuner_results: list of TunerResults (one per model)
        :param holdout_scores: list of (list of Scores). Each outer list item (one per model),
            contains a list of Scores (same Scores as the Tuner)
        """
        self._model_descriptions = model_descriptions
        self._model_names = model_names
        self._tuner_results = tuner_results
        self._holdout_scores = holdout_scores

    @property
    def tuner_results(self) -> List[TunerResults]:
        return self._tuner_results

    @property
    def holdout_scores(self) ->List[List[ScoreBase]]:
        """
        :return: List of Lists of Scores. Each item in the outer list corresponds to a specific model/
            model-description.  Each of those items has a list of Scores, corresponding to the
            Scores passed into the Searcher.
        """
        return self._holdout_scores

    @property
    def model_names(self) -> List[str]:
        return self._model_names

    @property
    def model_descriptions(self) -> List[str]:
        return self._model_descriptions

    @property
    def holdout_score_values(self) -> pd.DataFrame:
        """
        Evaluator values for the holdout sets
        # TODO: update documentation
        """
        # get all the scores for each model (each model will have the same Evaluators)
        scores = self._holdout_scores[0]
        # get all the columns that the tuner_results will have
        score_columns = [x.name for x in scores]
        holdout_accuracies = [[x.value for x in evaluator] for evaluator in self._holdout_scores]

        return pd.DataFrame(holdout_accuracies, columns=score_columns, index=self._model_descriptions)

    # noinspection PyUnresolvedReferences
    @property
    def best_tuned_results(self):
        """
        :return: a dataframe with each model + best tuned result as a row
        """
        # get all the scores for each model (each model will have the same Evaluators)
        scores = self.holdout_scores[0]
        # get all the columns that the tuner_results will have
        score_columns = [(x.name + '_mean', x.name + '_st_dev', x.name + '_cv')
                             for x in scores]
        score_columns = list(sum(score_columns, ()))  # flatten out list
        # get the best resampling results each model that was tuned
        best_model_series = [x.best_model for x in self.tuner_results]

        # since each model will have different tune results, we need to separate and put in its own column
        hyper_params = [x.loc[list(set(x.index.values).difference(score_columns))].to_dict()
                        for x in best_model_series]  # gets the hyper-params for each model (as a dict)

        assert len(self._model_names) == len(hyper_params)
        assert len(self._model_names) == len(best_model_series)

        best_tune_results_df = pd.DataFrame(columns=['model', 'hyper_params'] + score_columns)
        for index in range(len(self._model_names)):
            tune_results_series = best_model_series[index].\
                drop(list(set(best_model_series[index].index.values).difference(score_columns)))
            tune_results_series['hyper_params'] = hyper_params[index]
            tune_results_series['model'] = self._model_names[index]
            best_tune_results_df = best_tune_results_df.\
                append(tune_results_series.loc[['model', 'hyper_params'] + score_columns])

        best_tune_results_df.index = self._model_descriptions

        return best_tune_results_df

    @property
    def best_model_index(self):
        """
        :return: returns the index of the best model based on the holdout accuracies (i.e. based on the first
            Score specified when creating the Resampler passed into the ModelSearcher constructor.
            This index can then be used to index on the `tuner_results` and `holdout_scores` properties
        """

        # get the first score in each list (1 list per model)
        score_list = [x[0] for x in self._holdout_scores]
        # should all be the same type of score
        assert all([type(score_list[0]) == type(x) for x in score_list])
        # since Scores implement the __lt__ function based on the better_than function, sorting the
        # Scores gets the indexes of the best sorted
        indexes = np.argsort(score_list)
        return indexes[0]  # return the index of the first i.e. "best" model based on the sorted Scores

# RESAMPLED CROSS VALIDATION
# TODO: add boxplot for tuner result cross validations (pass in metric)
# TODO: add heatmap: shows each best tune result (rows) for each Score (columns) (means)

# HOLDOUT
# TODO: add heatmap: shows each best tune result (rows) for each Score (columns)
