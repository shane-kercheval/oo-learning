from typing import List, Union

import pandas as pd

from oolearning.model_processors.TunerResultsBase import TunerResultsBase


class BayesianHyperOptTunerResults(TunerResultsBase):

    def __init__(self,
                 tune_results: pd.DataFrame,
                 time_results: pd.DataFrame,
                 parameter_names: List[str],
                 trials_object: object):
        super().__init__(tune_results, time_results)
        self._parameter_names = parameter_names
        self._trials_object = trials_object

    @property
    def best_hyper_params(self) -> Union[dict, None]:
        sorted_best_parameters = self.sorted_best_models.loc[:, self._parameter_names]. \
            copy(deep=True)
        # reindex, so that the top row has index 0; we can't simply use .iloc because if there are a mix
        # of float/int parameters iloc changes everything to a float, which fucks up the model if this
        # field is used to pass parameters to a model (e.g. when retraining on entire dataset in Searcher)
        sorted_best_parameters.index = range(sorted_best_parameters.shape[0])
        return {param: sorted_best_parameters.at[0, param]
                for param in sorted_best_parameters.columns.values}

    # noinspection PyUnresolvedReferences
    @property
    def trials_object(self) -> object:
        return self._trials_object
