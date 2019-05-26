from typing import List, Union

from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_processors.TunerResultsBase import TunerResultsBase


class BayesianHyperOptTunerResults(TunerResultsBase):

    def __init__(self,
                 resampler_results: List[ResamplerResults],
                 hyper_params_combos: List[dict],
                 resampler_times: List[str],
                 trials_object: object,
                 transformations_objects: List[dict]):
        super().__init__(resampler_results=resampler_results,
                         hyper_params_combos=hyper_params_combos,
                         resampler_times=resampler_times)

        self._trials_object = trials_object
        self._transformations_objects = transformations_objects

    @property
    def best_hyper_params(self) -> Union[dict, None]:
        sorted_best_parameters = self.sorted_best_models.loc[:, self.hyper_param_names]. \
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

    @property
    def transformation_names(self) -> List[str]:
        return list(self._transformations_objects[0].keys())

    @property
    def model_hyper_param_names(self) -> List[str]:
        return [key for key in list(self._hyper_params_combos[0].keys())
                if key not in self.transformation_names]

    @property
    def best_model_hyper_params(self) -> dict:
        """
        `best_hyper_params` returns model hyper parameters and the Transformers that were tuned.
        `best_model_hyper_params` returns only the hyper parameters specific to the model
        :return:
        """
        return {key: value for key, value in self.best_hyper_params.items()
                if key in self.model_hyper_param_names}

    @property
    def best_transformations(self) -> list:
        return list(self._transformations_objects[self.best_index].values())
