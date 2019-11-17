import itertools
import time
# from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count
from multiprocessing import get_context

from typing import List

import numpy as np
import pandas as pd

from oolearning.model_processors.CloneableFactory import CloneableFactory
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.GridSearchTransTunerResults import GridSearchTransTunerResults
from oolearning.model_processors.ModelTunerBase import ModelTunerBase
from oolearning.model_processors.ProcessingExceptions import CallbackUsedWithParallelizationError
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase


def resampler_results_build_cache_key(model_name: str,
                                      hyper_params: HyperParamsBase) -> str:
    """
    :return: returns a key that acts as, for example, the file name of the model being cached for the
        persistence manager; has the form:
            `repeat[repeat number]_fold[fold number]_[Model Class Name]`
    """
    model_name = 'resampler_results_' + model_name
    if hyper_params is None:
        key = model_name
    else:
        # if hyper-params, flatten out list of param names and values and concatenate/join them together
        hyper_params_long = '_'.join([str(x) + str(y) for x, y in hyper_params.params_dict.items()])
        key = '_'.join([model_name, hyper_params_long])

    return key


def single_tune(args):
    transformations_dict = args['transformations_dict']  # dict
    hyper_param_object = args['hyper_param_object']
    resampler_copy = args['resampler_copy']
    model_persistence_manager = args['model_persistence_manager']
    resampler_persistence_manager = args['resampler_persistence_manager']
    data_x = args['data_x']
    data_y = args['data_y']

    resampler_copy.append_transformations(list(transformations_dict.values()))  # works for empty list

    if model_persistence_manager is not None:
        resampler_copy.set_model_persistence_manager(persistence_manager=model_persistence_manager)

    if resampler_persistence_manager is not None:
        # noinspection PyProtectedMember
        cache_key = resampler_results_build_cache_key(model_name=resampler_copy._model_factory.get_model().name,  # noqa
                                                      hyper_params=hyper_param_object)
        resampler_persistence_manager.set_key(key=cache_key)
        resampler_copy.set_results_persistence_manager(persistence_manager=resampler_persistence_manager)

    start_time_individual = time.time()
    resampler_copy.resample(data_x=data_x, data_y=data_y, hyper_params=hyper_param_object)
    execution_time_individual = "{0} seconds".format(round(time.time() - start_time_individual))

    return {'resampler_results_object': resampler_copy.results,
            'resampler_time_seconds': execution_time_individual,
            'transformations_objects': transformations_dict,
            'hyper_params_combos': {key: type(value).__name__ for key, value in transformations_dict.items()},
            'decorators': resampler_copy.decorators,
            'num_resamples': resampler_copy.results.num_resamples}


class GridSearchTransformationTuner(ModelTunerBase):
    """
    TODO Document
    """
    def __init__(self,
                 resampler: ResamplerBase,
                 transformations_space: dict,
                 hyper_param_object: HyperParamsBase,
                 model_persistence_manager: PersistenceManagerBase = None,
                 resampler_persistence_manager: PersistenceManagerBase = None,
                 parallelization_cores: int = -1):
        """

        """

        super().__init__()
        assert isinstance(resampler, ResamplerBase)

        self._resampler_factory = CloneableFactory(resampler)
        self._hyper_param_factory = CloneableFactory(hyper_param_object)
        self._transformations_space = transformations_space
        self._model_persistence_manager_factory = CloneableFactory(model_persistence_manager)
        self._resampler_persistence_manager_factory = CloneableFactory(resampler_persistence_manager)
        self._parallelization_cores = parallelization_cores
        self._resampler_decorators = None
        self._number_of_resamples = None
        # noinspection PyProtectedMember
        # if there is a callback and we are using parallelization, raise error
        if resampler._train_callback is not None and (parallelization_cores != 0 and
                                                      parallelization_cores != 1):
            raise CallbackUsedWithParallelizationError()

    @property
    def resampler_decorators(self) -> List[List[DecoratorBase]]:
        """
        :return: a nested list of Decorators
            The length of the entire/outer list will equal the number of cycles (i.e. number of different
            hyper-param combos tested)
            The length of each inner list will equal the number of Decorator objects passed in
        """
        return self._resampler_decorators

    @property
    def number_of_resamples(self) -> set:
        """
        :return: a set containing the number of resamples for each Resampler
            e.g. a 5-fold 5-repeat resampler should have 25 resamples
            the number should be the same for each Resampler object so the set returned should only have
            1 item (25
        """
        return self._number_of_resamples

    def _tune(self, data_x: pd.DataFrame, data_y: np.ndarray) -> GridSearchTransTunerResults:
        """
        """

        params_list = [y if isinstance(y, list) else [y] for x, y in self._transformations_space.items()]
        grid_df = pd.DataFrame(list(itertools.product(*params_list)))
        grid_df.columns = self._transformations_space.keys()

        # map_function rather than a for loop so we can switch between parallelization and non-parallelization

        # noinspection PyUnresolvedReferences
        single_tune_args = [dict(transformations_dict={param: grid_df.at[x, param].clone()
                                                       for param in grid_df.columns.values},  # noqa
                                 hyper_param_object=self._hyper_param_factory.get(),  # noqa
                                 resampler_copy=self._resampler_factory.get(),  # resampler
                                 model_persistence_manager=self._model_persistence_manager_factory.get(),
                                 resampler_persistence_manager=self._resampler_persistence_manager_factory.get(),  # noqa
                                 data_x=data_x,
                                 data_y=data_y)
                            for x in range(len(grid_df))]

        # if self._parallelization_cores == 0 or self._parallelization_cores == 1:
        #     results = list(map(single_tune, single_tune_args))
        # else:
        #     cores = cpu_count() if self._parallelization_cores == -1 else self._parallelization_cores
        #     # with ThreadPool(cores) as pool:
        #     # https://codewithoutrules.com/2018/09/04/python-multiprocessing/
        #     with get_context("spawn").Pool(cores) as pool:
        #         results = list(pool.map(single_tune, single_tune_args))
        results = list(map(single_tune, single_tune_args))

        results_list = [x['resampler_results_object'] for x in results]
        time_duration_list = [x['resampler_time_seconds'] for x in results]
        transformations_objects = [x['transformations_objects'] for x in results]
        hyper_params_combos = [x['hyper_params_combos'] for x in results]
        self._resampler_decorators = [x['decorators'] for x in results]
        self._number_of_resamples = set([x['num_resamples'] for x in results])

        # noinspection PyTypeChecker
        return GridSearchTransTunerResults(resampler_results=results_list,
                                           hyper_params_combos=hyper_params_combos,
                                           resampler_times=time_duration_list,
                                           transformations_objects=transformations_objects)
