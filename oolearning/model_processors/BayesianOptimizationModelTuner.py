import time
from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count
from typing import List, Callable, Union

from bayes_opt import BayesianOptimization

import numpy as np
import pandas as pd

from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin
from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_processors.TunerResultsBase import TunerResultsBase
from oolearning.model_processors.BayesianOptimizationTunerResults import BayesianOptimizationTunerResults
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.ModelTunerBase import ModelTunerBase
from oolearning.model_processors.ProcessingExceptions import CallbackUsedWithParallelizationError
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.GridSearchTunerResults import GridSearchTunerResults
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.HyperParamsGrid import HyperParamsGrid
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase


class BayesianOptimizationModelTuner(ModelTunerBase):
    """
    A BayesianOptimizationModelTuner is a wrapper around around github/fmfn/BayesianOptimization
        (https://github.com/fmfn/BayesianOptimization) that searches for the
    uses a Resampler for tuning a single model across various hyper-parameters.
    In other words, it runs a specified Resampler repeatedly over a combination of hyper-parameters, finding
    the "best" potential model as well as related information.
    """
    def __init__(self,
                 resampler: ResamplerBase,
                 hyper_param_object: HyperParamsBase,
                 parameter_bounds: dict,
                 init_points: int,
                 n_iter: int,
                 verbose: int = 2,
                 seed: int = 42
                 ):
        """

        :param resampler:
        :param hyper_param_object:
        :param objective_function:
        """
        super().__init__()

        assert isinstance(resampler, ResamplerBase)

        self._resampler = resampler
        self._hyper_param_object = hyper_param_object
        self._parameter_bounds = parameter_bounds
        self._init_points = init_points
        self._n_iter = n_iter
        self._verbose = verbose
        self._seed = seed

    def _tune(self, data_x: pd.DataFrame, data_y: np.ndarray) -> TunerResultsBase:

        global temp_resampler_results
        temp_resampler_results = list()

        global temp_resampler_times
        temp_resampler_times = list()

        global temp_hyper_params
        temp_hyper_params = list()

        # need global functions otherwise I get "function not defined" in `optimizer.maximize()`
        global temp_objective_function
        def temp_objective_function(locals_dictionary: dict):
            # this will be passed in a diction from `locals()` call in the dynamic objective function which
            # will contain a dictionary of parameters with the corresponding values, which is exactly what
            # `update_dict()` takes

            # if(locals_dictionary is None):
            #     return None

            local_hyper_params = self._hyper_param_object.clone()
            local_hyper_params.update_dict(locals_dictionary)
            temp_hyper_params.append(local_hyper_params)

            local_resampler = self._resampler.clone()

            resample_start_time = time.time()
            local_resampler.resample(data_x=data_x, data_y=data_y, hyper_params=local_hyper_params)
            temp_resampler_times.append(time.time() - resample_start_time)
            temp_resampler_results.append(local_resampler.results)

            first_score_object = local_resampler.results.scores[0][0]

            assert isinstance(first_score_object, CostFunctionMixin) or \
                   isinstance(first_score_object, UtilityFunctionMixin)

            resample_mean = local_resampler.results.score_means[first_score_object.name]

            # if the first score object passed in to the resampler is a Cost Function, we want to **Minimize**
            # the score, so we need to multiply it by negative 1 since we the optimizer maximizes
            if isinstance(local_resampler.results.scores[0][0], CostFunctionMixin):
                resample_mean = resample_mean * -1

            return resample_mean

        parameter_names = list(self._parameter_bounds.keys())

        objective_parameters = ", ".join(parameter_names)
        objective_function_string = "global objective_function\n" \
                                    "def objective_function({0}): return temp_objective_function(locals())"
        objective_function_string = objective_function_string.format(objective_parameters)
        exec(objective_function_string)

        # noinspection PyUnresolvedReferences
        optimizer = BayesianOptimization(f=objective_function,
                                         pbounds=self._parameter_bounds,
                                         verbose=self._verbose,
                                         random_state=self._seed)
        optimizer.maximize(init_points=self._init_points, n_iter=self._n_iter)

        assert len(temp_resampler_results) == self._n_iter + self._init_points

        # ensure the target values email the mean resampled score
        # TODO will not work for CostFunctionMixin, must multiply by negative 1
        assert [ob['target'] for ob in optimizer.res] == [result.score_means[temp_resampler_results[0].scores[0][0].name] for result in temp_resampler_results]

        # the hyper-params of each resampler
        all_params_every_resampler = [params_object.params_dict for params_object in temp_hyper_params]
        # extract the params corresponding to the parameters we are optimizing according to parameter_bounds
        optimizer_params_dict = [{x: the_dict[x] for x in parameter_names
                                  if x in the_dict} for the_dict in all_params_every_resampler]

        tune_results = pd.DataFrame({x: [y[x] for y in optimizer_params_dict] for x in parameter_names})
        tune_results['resampler_object'] = temp_resampler_results

        time_results = pd.DataFrame({x: [y[x] for y in optimizer_params_dict] for x in parameter_names})
        time_results['resample_time_seconds'] = temp_resampler_times

        return BayesianOptimizationTunerResults(tune_results=tune_results,
                                                time_results=time_results,
                                                parameter_names=parameter_names,
                                                optimizer=optimizer)
