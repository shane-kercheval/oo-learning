from typing import List

import numpy as np
import pandas as pd

from oolearning.evaluators.EvaluatorBase import EvaluatorBase
from oolearning.model_processors.TunerResults import TunerResults


class SearcherResults:
    def __init__(self,
                 model_descriptions: List[str],
                 model_names: List[str],
                 tuner_results: List[TunerResults],
                 holdout_evaluators=List[List[EvaluatorBase]]):
        """
        :param tuner_results: list of TunerResults (one per model)
        :param holdout_evaluators: list of (list of Evaluators). Each outer list item (one per model),
            contains a list of Evaluators (same Evaluators as the Tuner)
        """
        self._model_descriptions = model_descriptions
        self._model_names = model_names
        self._tuner_results = tuner_results
        self._holdout_evaluators = holdout_evaluators

    @property
    def tuner_results(self):
        return self._tuner_results

    @property
    def holdout_evaluators(self):
        return self._holdout_evaluators

    @property
    def model_names(self):
        return self._model_names

    @property
    def model_descriptions(self):
        return self._model_descriptions

    @property
    def holdout_eval_values(self):
        """
        Evaluator values for the holdout sets
        # TODO: update documentation
        """
        # get all the evaluators for each model (each model will have the same Evaluators)
        evaluators = self._holdout_evaluators[0]
        # get all the columns that the tuner_results will have
        evaluator_columns = [x.metric_name for x in evaluators]
        holdout_accuracies = [[x.value for x in evaluator] for evaluator in self._holdout_evaluators]

        return pd.DataFrame(holdout_accuracies, columns=evaluator_columns, index=self._model_descriptions)

    # noinspection PyUnresolvedReferences
    @property
    def best_tuned_results(self):
        """
        :return: a dataframe with each model + best tuned result as a row
        """
        # get all the evaluators for each model (each model will have the same Evaluators)
        evaluators = self.holdout_evaluators[0]
        # get all the columns that the tuner_results will have
        evaluator_columns = [(x.metric_name + '_mean', x.metric_name + '_st_dev') for x in evaluators]
        evaluator_columns = list(sum(evaluator_columns, ()))  # flatten out list
        # get the best resampling results each model that was tuned
        best_model_series = [x.best_model for x in self.tuner_results]

        # since each model will have different tune results, we need to separate and put in its own column
        hyper_params = [x.loc[list(set(x.index.values).difference(evaluator_columns))].to_dict()
                        for x in best_model_series]  # gets the hyper-params for each model (as a dict)

        assert len(self._model_names) == len(hyper_params)
        assert len(self._model_names) == len(best_model_series)

        best_tune_results_df = pd.DataFrame(columns=['model', 'hyper_params'] + evaluator_columns)
        for index in range(len(self._model_names)):
            tune_results_series = best_model_series[index].\
                drop(list(set(best_model_series[index].index.values).difference(evaluator_columns)))
            tune_results_series['hyper_params'] = hyper_params[index]
            tune_results_series['model'] = self._model_names[index]
            best_tune_results_df = best_tune_results_df.\
                append(tune_results_series.loc[['model', 'hyper_params'] + evaluator_columns])

        best_tune_results_df.index = self._model_descriptions

        return best_tune_results_df

    @property
    def best_model_index(self):
        """
        :return: returns the index of the best model based on the holdout accuracies (i.e. based on the first
            Evaluator specified when creating the Resampler passed into the ModelSearcher constructor.
            This index can then be used to index on the `tuner_results` and `holdout_evaluators` properties
        """

        # get the first evaluator in each list (1 list per model)
        evaluator_list = [x[0] for x in self._holdout_evaluators]
        # should all be the same type of evaluator
        assert all([type(evaluator_list[0]) == type(x) for x in evaluator_list])
        # since Evaluators implement the __lt__ function based on the better_than function, sorting the
        # Evaluators gets the indexes of the best sorted
        indexes = np.argsort(evaluator_list)
        return indexes[0]  # return the index of the first i.e. "best" model based on the sorted Evaluators
