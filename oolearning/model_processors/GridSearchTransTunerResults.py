from typing import List, Union

from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_processors.TunerResultsBase import TunerResultsBase
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid


class GridSearchTransTunerResults(TunerResultsBase):
    def __init__(self,
                 resampler_results: List[ResamplerResults],
                 hyper_params_combos: List[dict],
                 resampler_times: List[str],
                 transformations_objects: List[dict]
                 ):

        super().__init__(resampler_results=resampler_results,
                         hyper_params_combos=hyper_params_combos,
                         resampler_times=resampler_times)

        self._transformations_objects = transformations_objects

        # make sure none of the transformations_objects has been used
        # noinspection PyUnresolvedReferences
        assert all([not transformer.has_executed
                    for trans_object in transformations_objects
                    for transformer in trans_object.values()])

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

    @property
    def transformation_names(self) -> List[str]:
        return self.hyper_param_names

    @property
    def best_transformations(self) -> list:
        return list(self._transformations_objects[self.best_index].values())
