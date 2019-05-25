import time

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization

from oolearning.model_processors.CloneableFactory import CloneableFactory
from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin
from oolearning.model_processors.BayesianOptimizationTunerResults import BayesianOptimizationTunerResults
from oolearning.model_processors.ModelTunerBase import ModelTunerBase
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.TunerResultsBase import TunerResultsBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase


class BayesianOptimizationModelTuner(ModelTunerBase):
    # noinspection SpellCheckingInspection
    """
        A BayesianOptimizationModelTuner is a wrapper around around github/fmfn/BayesianOptimization
            (https://github.com/fmfn/BayesianOptimization) that searches for the
        uses a Resampler for tuning a single model across various hyper-parameters.
        In other words, it runs a specified Resampler repeatedly over a combination of hyper-parameters,
        finding the "best" potential model as well as related information.
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
        """
        super().__init__()

        assert isinstance(resampler, ResamplerBase)

        self._resampler_factory = CloneableFactory(resampler)
        self._hyper_param_factory = CloneableFactory(hyper_param_object)
        self._parameter_bounds = parameter_bounds
        self._init_points = init_points
        self._n_iter = n_iter
        self._verbose = verbose
        self._seed = seed

    def _tune(self, data_x: pd.DataFrame, data_y: np.ndarray) -> TunerResultsBase:

        global resampler_results
        resampler_results = list()

        global resampler_times
        resampler_times = list()

        global temp_hyper_params
        temp_hyper_params = list()

        # need global functions otherwise I get "function not defined" in `optimizer.maximize()`
        # noinspection PyGlobalUndefined
        global temp_objective_function

        # noinspection PyUnusedLocal,PyRedeclaration
        def temp_objective_function(locals_dictionary: dict):
            # this will be passed in a diction from `locals()` call in the dynamic objective function which
            # will contain a dictionary of parameters with the corresponding values, which is exactly what
            # `update_dict()` takes

            # if(locals_dictionary is None):
            #     return None

            local_hyper_params = self._hyper_param_factory.get()
            local_hyper_params.update_dict(locals_dictionary)
            temp_hyper_params.append(local_hyper_params)

            local_resampler = self._resampler_factory.get()

            resample_start_time = time.time()
            local_resampler.resample(data_x=data_x, data_y=data_y, hyper_params=local_hyper_params)
            resampler_times.append(time.time() - resample_start_time)
            resampler_results.append(local_resampler.results)

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

        assert len(resampler_results) == self._n_iter + self._init_points

        # ensure the target values email the mean resampled score
        # if the score object is a CostFunctionMixin, must multiply by -1 since we had to multiple by -1
        # above to so that we can optimize for the smallest value (e.g. RMSE)
        multiplier = -1 if isinstance(resampler_results[0].scores[0][0], CostFunctionMixin) else 1
        assert [multiplier * ob['target'] for ob in optimizer.res] ==\
               [result.score_means[resampler_results[0].scores[0][0].name]
                for result in resampler_results]

        # the hyper-params of each resampler
        all_params_every_resampler = [params_object.params_dict for params_object in temp_hyper_params]
        # extract the params corresponding to the parameters we are optimizing according to parameter_bounds
        optimizer_params_dict = [{x: the_dict[x] for x in parameter_names
                                  if x in the_dict} for the_dict in all_params_every_resampler]

        return BayesianOptimizationTunerResults(resampler_results=resampler_results,
                                                hyper_params_combos=optimizer_params_dict,
                                                resampler_times=[str(round(x, 1)) + " Seconds" for x in resampler_times],  # noqa
                                                optimizer=optimizer)
