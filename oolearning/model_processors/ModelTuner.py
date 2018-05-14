import time
from typing import List

import numpy as np
import pandas as pd

from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.TunerResults import TunerResults
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError


class ModelTuner:
    """
    A ModelTuner uses a Resampler for tuning a single model across various hyper-parameters.
    In other words, it runs a specified Resampler repeatedly over a combination of hyper-parameters, finding
    the "best" potential model as well as related information.
    """
    def __init__(self,
                 resampler: ResamplerBase,
                 hyper_param_object: HyperParamsBase,
                 resampler_decorators: List[DecoratorBase] = None,
                 persistence_manager: PersistenceManagerBase = None):
        """
        :param resampler:
        :param hyper_param_object: an object containing the default values for the corresponding
            hyper-parameters. Default in this case may be simply the parameterless constructor for the
            associated hyper-parameters class, or may be custom hyper-parameters values set in the
            constructor, which will not be tuned (since any hyper-parameter that will be tuned will
            be over-written.

            e.g if we are tuning a Random Forest classification model over `max_features`, we might pass in
            `RandomForestHP()` to retain all the defaults, or we could pass in
            `RandomForestHP(n_estimators=20)`, since `n_estimators` is not being tuned (i.e. over-written)
        :param resampler_decorators: a list of decorators that will be passed into the resampler object.
            They will be cloned for each resampler object.
        :param persistence_manager: a PersistenceManager defining how the underlying models should be cached,
            optional.
            NOTE: for each resampling (i.e. each set of hyper-params being resampled) the persistence_manager
            is cloned and passed to the resampler. The resampler already differentiates the file name based on
            the hyper-parameters (and e.g. repeat/fold indexes which should make it unique)
        """
        assert isinstance(resampler, ResamplerBase)

        self._resampler = resampler
        self._hyper_param_object = hyper_param_object
        self._results = None
        self._persistence_manager = persistence_manager
        self._resampler_decorators = resampler_decorators

    @property
    def results(self):
        if self._results is None:
            raise ModelNotFittedError()

        return self._results

    def tune(self, data_x: pd.DataFrame, data_y: np.ndarray, params_grid: HyperParamsGrid):
        """
        `resample` handles the logic of applying the pre-process transformations, as well as fitting the data
            based on teh resampling method and storing the resampled accuracies
        :param data_x: DataFrame to train the model on
        :param data_y: np.ndarray containing the target values to be trained on
        :type params_grid: object containing the sequences of hyper-parameters to tune. Each hyper-parameter
            will over-write the associated hyper-parameter passed into the constructor's
            `hyper_param_defaults` parameter.
        :return: None
        """
        # single row DataFrame if no params, or DataFrame of each combination of hyper-parameters
        params_combinations = pd.DataFrame({'hyper_params': ['None']}) \
            if params_grid is None else params_grid.params_grid
        assert len(params_combinations) > 0
        results_list = list()
        time_duration_list = list()

        for index in range(len(params_combinations)):
            # we are going to reuse the resampler and hyper_params for each combination, so clone first
            resampler_copy = self._resampler.clone()

            if self._resampler_decorators:
                resampler_copy.set_decorators(decorators=[x.clone() for x in self._resampler_decorators])

            if self._persistence_manager is not None:
                resampler_copy.set_persistence_manager(persistence_manager=self._persistence_manager.clone())
            hyper_params_copy = None if self._hyper_param_object is None else self._hyper_param_object.clone()

            if params_grid is not None and self._hyper_param_object is not None:
                # if we are tuning over hyper-params, update the params object with the current params
                params_dict = params_combinations.iloc[index, :].to_dict()  # dict of the current combination
                # print(params_dict)
                hyper_params_copy.update_dict(params_dict)  # default params, updated based on combo

            start_time = time.time()
            resampler_copy.resample(data_x=data_x, data_y=data_y, hyper_params=hyper_params_copy)
            execution_time = "{0} seconds".format(round(time.time() - start_time))
            # print(execution_time)
            time_duration_list.append(execution_time)

            results_list.append(resampler_copy.results)

        tune_results = pd.concat([params_combinations.copy(),
                                  pd.DataFrame(results_list, columns=['resampler_object'])], axis=1)
        assert tune_results.isnull().sum().sum() == 0

        time_results = pd.concat([params_combinations.copy(),
                                  pd.DataFrame(time_duration_list, columns=['execution_time'])],
                                 axis=1)
        assert time_results.isnull().sum().sum() == 0

        self._results = TunerResults(tune_results=tune_results,
                                     time_results=time_results,
                                     params_grid=params_grid)
