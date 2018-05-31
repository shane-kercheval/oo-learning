from typing import List
from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count

import time
import numpy as np
import pandas as pd

from oolearning.model_processors.ProcessingExceptions import CallbackUsedWithParallelizationError
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.TunerResults import TunerResults
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError


def single_tune(args):
    params_combo_index = args['params_combo_index']
    has_params = args['has_params']
    hyper_param_object = args['hyper_param_object']
    # we are going to reuse the resampler and hyper_params for each combination, so clone first
    resampler_copy = args['resampler_copy']
    decorators = args['decorators']
    persistence_manager = args['persistence_manager']
    data_x = args['data_x']
    data_y = args['data_y']

    if decorators:
        resampler_copy.set_decorators(decorators=decorators)

    if persistence_manager is not None:
        resampler_copy.set_persistence_manager(persistence_manager=persistence_manager.clone())

    if has_params and hyper_param_object is not None:
        # if we are tuning over hyper-params, update the params object with the current params
        params_dict = params_combo_index.to_dict()  # dict of the current combination
        # print(params_dict)
        hyper_param_object.update_dict(params_dict)  # default params, updated based on combo

    start_time_individual = time.time()
    resampler_copy.resample(data_x=data_x, data_y=data_y, hyper_params=hyper_param_object)
    execution_time_individual = "{0} seconds".format(round(time.time() - start_time_individual))
    # print(execution_time)
    return resampler_copy.results, execution_time_individual


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
                 persistence_manager: PersistenceManagerBase = None,
                 parallelization_cores: int=-1):
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
        :param parallelization_cores: the number of cores to use for parallelization. -1 is all, 0 or 1 is
            "off"
        """
        assert isinstance(resampler, ResamplerBase)

        self._resampler = resampler
        self._hyper_param_object = hyper_param_object
        self._results = None
        self._persistence_manager = persistence_manager
        self._resampler_decorators = resampler_decorators
        self._total_tune_time = None

        # noinspection PyProtectedMember
        # if there is a callback and we are using parallelization, raise error
        if resampler._train_callback is not None and (parallelization_cores != 0 and
                                                      parallelization_cores != 1):
            raise CallbackUsedWithParallelizationError()

        # save the map_function based on parallelization or not
        if parallelization_cores == 0 or parallelization_cores == 1:
            self._map_function = map
        else:
            cores = cpu_count() if parallelization_cores == -1 else parallelization_cores
            pool = ThreadPool(cores)
            self._map_function = pool.map

    @property
    def results(self):
        if self._results is None:
            raise ModelNotFittedError()

        return self._results

    @property
    def total_tune_time(self):
        """
        :return: total time
        """
        if self._results is None:
            raise ModelNotFittedError()

        return self._total_tune_time

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

        # map_function rather than a for loop so we can switch between parallelization and non-parallelization
        single_tune_args = [dict(params_combo_index=params_combinations.iloc[x, :],  # parameter combination
                                 has_params=params_grid is not None,
                                 hyper_param_object=self._hyper_param_object.clone() if self._hyper_param_object else None,  # noqa
                                 resampler_copy=self._resampler.clone(),  # resampler
                                 decorators=[y.clone() for y in self._resampler_decorators] if self._resampler_decorators else None,  # noqa
                                 persistence_manager=self._persistence_manager,
                                 data_x=data_x,
                                 data_y=data_y)  # decorators
                            for x in range(len(params_combinations))]
        start_time = time.time()
        results = list(self._map_function(single_tune, single_tune_args))
        self._total_tune_time = time.time() - start_time

        results_list = [x[0] for x in results]
        time_duration_list = [x[1] for x in results]

        # for index in range(len(params_combinations)):
        #     # we are going to reuse the resampler and hyper_params for each combination, so clone first
        #     resampler_copy = self._resampler.clone()
        #
        #     if self._resampler_decorators:
        #         resampler_copy.set_decorators(decorators=[x.clone() for x in self._resampler_decorators])
        #
        #     if self._persistence_manager is not None:
        #         resampler_copy.set_persistence_manager(persistence_manager=self._persistence_manager.clone())
        #     hyper_params_copy = None if self._hyper_param_object is None else self._hyper_param_object.clone()
        #
        #     if params_grid is not None and self._hyper_param_object is not None:
        #         # if we are tuning over hyper-params, update the params object with the current params
        #         params_dict = params_combinations.iloc[index, :].to_dict()  # dict of the current combination
        #         # print(params_dict)
        #         hyper_params_copy.update_dict(params_dict)  # default params, updated based on combo
        #
        #     start_time = time.time()
        #     resampler_copy.resample(data_x=data_x, data_y=data_y, hyper_params=hyper_params_copy)
        #     execution_time = "{0} seconds".format(round(time.time() - start_time))
        #     # print(execution_time)
        #     time_duration_list.append(execution_time)
        #
        #     results_list.append(resampler_copy.results)

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
