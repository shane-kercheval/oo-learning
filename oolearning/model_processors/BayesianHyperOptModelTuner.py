import time
from typing import List

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, tpe

from oolearning.evaluators.CostFunctionMixin import CostFunctionMixin
from oolearning.evaluators.UtilityFunctionMixin import UtilityFunctionMixin
from oolearning.model_processors.BayesianHyperOptTunerResults import BayesianHyperOptTunerResults
from oolearning.model_processors.CloneableFactory import CloneableFactory
from oolearning.model_processors.DecoratorBase import DecoratorBase
from oolearning.model_processors.ModelTunerBase import ModelTunerBase
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.TunerResultsBase import TunerResultsBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.transformers.TransformerBase import TransformerBase


class BayesianHyperOptModelTuner(ModelTunerBase):
    """
    Wrapper for HyperOpt
        https://hyperopt.github.io/hyperopt/
        https://github.com/hyperopt/hyperopt
        https://github.com/hyperopt/hyperopt/wiki/FMin

    A good article on hyperopt can be found here
        https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0

    Uses a Resampler for tuning a single model across various hyper-parameters.
    In other words, it runs a specified Resampler repeatedly over a combination of hyper-parameters, finding
    the "best" potential model as well as related information.
    """
    def __init__(self,
                 resampler: ResamplerBase,
                 hyper_param_object: HyperParamsBase,
                 space: dict,
                 max_evaluations: int,
                 seed: int = 42):
        """

        :param resampler:
        :param hyper_param_object:
        :param space: dictionary of Transformer objects or hp hyper-parameter types
            e.g. `hp.choice('max_depth', range(-1, 10))`
        :param max_evaluations:
        :param seed:
        """

        super().__init__()

        self._resampler_factory = CloneableFactory(resampler)
        self._hyper_param_factory = CloneableFactory(hyper_param_object)
        self._space = space
        self._max_evaluations = max_evaluations
        self._seed = seed
        self._resampler_decorators = None

    @property
    def resampler_decorators(self) -> List[List[DecoratorBase]]:
        """
        :return: a nested list of Decorators
            The length of the entire/outer list will equal the number of cycles (i.e. number of different
            hyper-param combos tested)
            The length of each inner list will equal the number of Decorator objects passed in
        """
        return self._resampler_decorators

    def _tune(self, data_x: pd.DataFrame, data_y: np.ndarray) -> TunerResultsBase:

        def objective(params):
            transformation_dictionary = {key: value.clone() for key, value in params.items() if isinstance(value, TransformerBase)}  # noqa
            transformations = [value.clone() for value in transformation_dictionary.values()]

            hyper_params_dict = {key: value for key, value in params.items() if not isinstance(value, TransformerBase)}  # noqa
            local_hyper_params = self._hyper_param_factory.get()
            local_hyper_params.update_dict(hyper_params_dict)

            local_resampler = self._resampler_factory.get()
            local_resampler.append_transformations(transformations)  # works for empty list

            resample_start_time = time.time()
            local_resampler.resample(data_x=data_x, data_y=data_y, hyper_params=local_hyper_params)
            resample_time = time.time() - resample_start_time

            first_score_object = local_resampler.results.scores[0][0]

            assert isinstance(first_score_object, CostFunctionMixin) or \
                   isinstance(first_score_object, UtilityFunctionMixin)

            resample_mean = local_resampler.results.score_means[first_score_object.name]

            # the hyperopt framework minimizes the "loss" function by default
            # so, if the first score object passed in to the resampler is a Utility Function, we want to
            # **Maximize** the score, so we need to multiply it by negative 1 since the optimizer *minimizes*
            if isinstance(local_resampler.results.scores[0][0], UtilityFunctionMixin):
                resample_mean = resample_mean * -1

            return {'loss': resample_mean,
                    'status': STATUS_OK,
                    'params': {key: value if not isinstance(value, TransformerBase) else type(value).__name__
                               for key, value in params.items()},
                    'resampler_object': local_resampler.results,
                    'resampler_time_seconds': resample_time,
                    'transformations': transformation_dictionary}

        trials = Trials()
        # I do not use the results of `fmin`, because for hp.choice results, fmin returns the index of the
        # best parameter, not the actual value, which is not ideal (although I assume it is required
        # since you could pass in an object and would need to know the index number of the object
        fmin(fn=objective,
             space=self._space,
             algo=tpe.suggest,
             max_evals=self._max_evaluations,
             trials=trials,
             rstate=np.random.RandomState(seed=self._seed))

        # these transformation objects will not have been used
        transformations_objects = [result['transformations'] for result in trials.results]

        # check that the Resampler scores have the same mean values as the loss values in trials
        # if the score object is a Utility, must multiply by -1 since we had to multiple by -1
        # above to so that we can optimize for the largest value (e.g. AUC)
        multiplier = -1 if isinstance(trials.results[0]['resampler_object'].scores[0][0], UtilityFunctionMixin) else 1  # noqa
        optimizer_values = [dictionary['loss'] * multiplier for dictionary in trials.results]
        resampler_means = [dictionary['resampler_object'].score_means[dictionary['resampler_object'].scores[0][0].name]  # noqa
                           for dictionary in trials.results]
        assert optimizer_values == resampler_means

        self._resampler_decorators = [x['resampler_object'].decorators for x in trials.results]

        resampler_results = [dictionary['resampler_object'] for dictionary in trials.results]
        hyper_params_combos = [dictionary['params'] for dictionary in trials.results]
        resampler_times = [str(round(dictionary['resampler_time_seconds'], 1)) + " Seconds" for dictionary in trials.results]  # noqa
        return BayesianHyperOptTunerResults(resampler_results=resampler_results,
                                            hyper_params_combos=hyper_params_combos,
                                            resampler_times=resampler_times,
                                            trials_object=trials,
                                            transformations_objects=transformations_objects)
