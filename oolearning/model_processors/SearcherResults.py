from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
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


# single item is a model
# each model has been tuned so it (the ones with hyper-params at least) have multiple sub-models (i.e. 1 for each hyper-param combination)
# each model has a "best" hyper-params sub-model based on the tuned/resampled results
    # found in `best_tuned_results`, which the the best submodel per model, with associated holdout scores
    # after each "model" has tuned across all sub-model/hyper-param-combos, the best model is chosen, and the model & best hyper-params are refit on all of the training data, and scored on a holdout set
        # these holdout scores are found in `holdout_score_values`, these are single values so no mean/standard-dev associated with them
# each sub-model has been resampled, so the "best model" for the sub-model has associated resampled data (i.e. all the cross validation scores for each score)
    #

    def get_holdout_score_heatmap(self):
        """
        NOTE: only shows the "tuned" hyper-params i.e. hyper-params that were tuned over >1 values.
        :return:
        """
        scores = self.holdout_scores[0]
        # if the Score is a Cost Function it is a 'minimizer'
        minimizers = [isinstance(x, CostFunctionMixin) for x in scores]

        score_columns = [x.name for x in scores]
        score_values = self.holdout_score_values.loc[:, score_columns]

        num_rows = len(score_values)
        num_cols = len(score_values.columns)
        fig, ax = plt.subplots(figsize=(10, 10))
        for i in range(num_cols):
            truths = [True] * num_cols
            truths[i] = False
            mask = np.array(num_rows * [truths], dtype=bool)
            color_values = np.ma.masked_where(mask, score_values)
            # "_r" value after color means invert colors (small values are darker)
            ax.pcolormesh(color_values, cmap='Blues_r' if minimizers[i] else 'Greens')

        for y in range(score_values.shape[0]):
            for x in range(score_values.shape[1]):
                plt.text(x + .5, y + .5, '%.3f' % score_values.ix[y, x],
                         horizontalalignment='center',
                         verticalalignment='center')

        ax.set_xticks(np.arange(start=0.5, stop=len(score_columns), step=1))
        ax.set_xticklabels(score_columns, rotation=35, ha='right')

        labels = self.holdout_score_values.index.values

        y_tick_positions = np.arange(start=0, stop=len(labels)) + 0.5
        ax.set_yticks(y_tick_positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        plt.tight_layout()
        return plt
