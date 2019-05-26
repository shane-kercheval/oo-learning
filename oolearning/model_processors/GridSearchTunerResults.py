from typing import List, Union

from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.model_processors.TunerResultsBase import TunerResultsBase
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid


class GridSearchTunerResults(TunerResultsBase):
    def __init__(self,
                 resampler_results: List[ResamplerResults],
                 params_grid: HyperParamsGrid,
                 resampler_times: List[str],
                 ):

        super().__init__(resampler_results=resampler_results,
                         hyper_params_combos=params_grid.params_grid.to_dict('records') if params_grid else None,  # noqa
                         resampler_times=resampler_times)

        self._params_grid = params_grid

    @property
    def best_hyper_params(self) -> Union[dict, None]:
        if self._params_grid is None:
            return None
        else:
            # get the sorted best parameters, create a copy so when we change the indexes it doesn't change
            # the original dataframe
            sorted_best_parameters = self.sorted_best_models.loc[:, self._params_grid.param_names].\
                copy(deep=True)
            # reindex, so that the top row has index 0; we can't simply use .iloc because if there are a mix
            # of float/int parameters iloc changes everything to a float, which fucks up the model if this
            # field is used to pass parameters to a model (e.g. when retraining on entire dataset in Searcher)
            sorted_best_parameters.index = range(sorted_best_parameters.shape[0])
            return {param: sorted_best_parameters.at[0, param]
                    for param in sorted_best_parameters.columns.values}
