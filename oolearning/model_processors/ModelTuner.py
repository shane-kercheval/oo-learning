from typing import List
from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count

import time
import numpy as np
import pandas as pd

from oolearning.model_processors.ProcessingExceptions import CallbackUsedWithParallelizationError
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.TunerResults import TunerResults
from oolearning.model_wrappers.ModelExceptions import ModelNotFittedError


def resampler_results_build_cache_key(model: ModelWrapperBase,
                                      hyper_params: HyperParamsBase) -> str:
    """
    :return: returns a key that acts as, for example, the file name of the model being cached for the
        persistence manager; has the form:
            `repeat[repeat number]_fold[fold number]_[Model Class Name]`
    """
    model_name = 'resampler_results_' + model.name
    if hyper_params is None:
        key = model_name
    else:
        # if hyper-params, flatten out list of param names and values and concatenate/join them together
        hyper_params_long = '_'.join([str(x) + str(y) for x, y in hyper_params.params_dict.items()])
        key = '_'.join([model_name, hyper_params_long])

    return key


def single_tune(args):
    params_combo_index = args['params_combo_index']  # dict
    has_params = args['has_params']
    hyper_param_object = args['hyper_param_object']
    # we are going to reuse the resampler and hyper_params for each combination, so clone first
    resampler_copy = args['resampler_copy']
    decorators = args['decorators']
    model_persistence_manager = args['model_persistence_manager']
    resampler_persistence_manager = args['resampler_persistence_manager']
    data_x = args['data_x']
    data_y = args['data_y']

    if decorators:
        resampler_copy.set_decorators(decorators=decorators)

    if has_params and hyper_param_object is not None:
        # if we are tuning over hyper-params, update the params object with the current params
        # print(params_combo_index)
        hyper_param_object.update_dict(params_combo_index)  # default params, updated based on combo

    if model_persistence_manager is not None:
        resampler_copy.set_model_persistence_manager(persistence_manager=model_persistence_manager.clone())

    if resampler_persistence_manager is not None:
        cache_key = resampler_results_build_cache_key(model=resampler_copy.model,
                                                      hyper_params=hyper_param_object)
        resampler_persistence_manager.set_key(key=cache_key)
        resampler_copy.set_results_persistence_manager(persistence_manager=resampler_persistence_manager.clone())  # noqa

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
                 model_persistence_manager: PersistenceManagerBase = None,
                 resampler_persistence_manager: PersistenceManagerBase = None,
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
        :param model_persistence_manager: a PersistenceManager defining how the underlying models should be cached,
            optional.
            NOTE: for each resampling (i.e. each set of hyper-params being resampled) the model_persistence_manager
            is cloned and passed to the resampler. The resampler already differentiates the file name based on
            the hyper-parameters (and e.g. repeat/fold indexes which should make it unique)
        :param resampler_persistence_manager:  caches the entire results of the resampler (not individual
            models) so that the resampling is not run at all (if cached files are found). However, if, for
            example, the transformations are changed (which will change the results), the cache directory
            needs to be deleted (i.e. clear the cache).
        :param parallelization_cores: the number of cores to use for parallelization. -1 is all, 0 or 1 is
            "off"
        """
        assert isinstance(resampler, ResamplerBase)

        self._resampler = resampler
        self._hyper_param_object = hyper_param_object
        self._results = None
        self._model_persistence_manager = model_persistence_manager
        self._resampler_persistence_manager = resampler_persistence_manager
        self._resampler_decorators = resampler_decorators
        self._total_tune_time = None
        self._parallelization_cores = parallelization_cores

        # noinspection PyProtectedMember
        # if there is a callback and we are using parallelization, raise error
        if resampler._train_callback is not None and (parallelization_cores != 0 and
                                                      parallelization_cores != 1):
            raise CallbackUsedWithParallelizationError()

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

        # for `params_combo_index`, we can't simply use .iloc (which i previously did), because if a
        # combination happens to have a mix of float/int params types, iloc will convert everything to a
        # float, which can cause problems if a model expects and checks for an int (e.g. XGBoost); so we
        # build up a dictionary which retains the original param type.
        single_tune_args = [dict(params_combo_index={param: params_combinations.at[x, param] for param in params_combinations.columns.values},  # noqa
                                 has_params=params_grid is not None,
                                 hyper_param_object=self._hyper_param_object.clone() if self._hyper_param_object else None,  # noqa
                                 resampler_copy=self._resampler.clone(),  # resampler
                                 decorators=[y.clone() for y in self._resampler_decorators] if self._resampler_decorators else None,  # noqa
                                 model_persistence_manager=self._model_persistence_manager,
                                 resampler_persistence_manager=self._resampler_persistence_manager,
                                 data_x=data_x,
                                 data_y=data_y)  # decorators
                            for x in range(len(params_combinations))]

        start_time = time.time()
        if self._parallelization_cores == 0 or self._parallelization_cores == 1:
            results = list(map(single_tune, single_tune_args))
        else:
            cores = cpu_count() if self._parallelization_cores == -1 else self._parallelization_cores
            with ThreadPool(cores) as pool:
                results = list(pool.map(single_tune, single_tune_args))
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
        #     if self._model_persistence_manager is not None:
        #         resampler_copy.set_model_persistence_manager(persistence_manager=self._model_persistence_manager.clone())
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

        # as a check, we want to make sure the tune_results and time_results dataframe doesn't contain any
        # NAs; however, if we set a particular hyper-parameter to None, this will cause a false positive
        # so, let's ignore any columns where the hyper-param is specifically set to None
        if params_grid:
            # noinspection PyProtectedMember
            params_containing_none = [key for key, value in params_grid._params_dict.items()
                                      if value is None or (isinstance(value, list) and None in value)]
        else:
            params_containing_none = []

        tune_results = pd.concat([params_combinations.copy(),
                                  pd.DataFrame(results_list, columns=['resampler_object'])], axis=1)
        assert tune_results.drop(columns=params_containing_none).isnull().sum().sum() == 0

        time_results = pd.concat([params_combinations.copy(),
                                  pd.DataFrame(time_duration_list, columns=['execution_time'])],
                                 axis=1)
        assert time_results.drop(columns=params_containing_none).isnull().sum().sum() == 0

        self._results = TunerResults(tune_results=tune_results,
                                     time_results=time_results,
                                     params_grid=params_grid)
