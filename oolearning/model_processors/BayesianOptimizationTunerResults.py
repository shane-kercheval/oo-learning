from typing import Union, List

import pandas as pd

from oolearning.model_processors.TunerResultsBase import TunerResultsBase


class BayesianOptimizationTunerResults(TunerResultsBase):

    def __init__(self,
                 tune_results: pd.DataFrame,
                 time_results: pd.DataFrame,
                 parameter_names: List[str],
                 optimizer: object):
        super().__init__(tune_results, time_results)
        self._parameter_names = parameter_names
        self._optimizer_object = optimizer

    @property
    def best_hyper_params(self) -> Union[dict, None]:
        params_nested_dict = self.resampled_stats[self._parameter_names].to_dict()
        return {key: value[self.best_index] for key, value in params_nested_dict.items()}

    # noinspection PyUnresolvedReferences
    @property
    def optimizer_results(self) -> pd.DataFrame:
        score_name = self.best_model_resampler_object.scores[0][0].name
        optimizer_results = pd.DataFrame([res['params'] for res in self._optimizer_object.res])
        optimizer_results[score_name] = [res['target'] for res in self._optimizer_object.res]

        return optimizer_results
