import numpy as np
import pandas as pd
from typing import List

from ..evaluators.EvaluatorBase import EvaluatorBase
from ..hyper_params.HyperParamsBase import HyperParamsBase
from ..model_wrappers.ModelWrapperBase import ModelWrapperBase
from ..model_processors.ResamplerBase import ResamplerBase
from ..model_processors.ResamplerResults import ResamplerResults
from ..splitters.StratifiedDataSplitter import StratifiedDataSplitter
from ..transformers.TransformerBase import TransformerBase

# TODO: NOT FINISHED, NOT TESTED


class StratifiedMonteCarloResampler(ResamplerBase):
    def __init__(self,
                 model: ModelWrapperBase,
                 model_transformations: List[TransformerBase],
                 stratified_splitter: StratifiedDataSplitter,
                 evaluators: List[EvaluatorBase],
                 repeats=30):
        """
        :param model:
        :param model_transformations:
        :param stratified_splitter: e.g. ClassificationStratifiedDataSplitter or
            RegressionStratifiedDataSplitter; i.e. this resampler needs to know how to stratify the data
            based off of the target values
        :param evaluators:
        :param repeats:
        """
        super().__init__(model=model, model_transformations=model_transformations, evaluators=evaluators)

        assert isinstance(repeats, int)

        self._repeats = repeats
        self._stratified_splitter = stratified_splitter

    def _resample(self,
                  data_x: pd.DataFrame,
                  data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:

        resampler_results = ResamplerResults()
        training_indexes, test_indexes = self._stratified_splitter.split_monte_carlo(target_values=data_y,
                                                                                     samples=self._repeats,
                                                                                     seed=42)
        for train_ind, test_ind in zip(training_indexes, test_indexes):
                train_x, test_x = data_x[train_ind], data_x[test_ind]
                train_y, test_y = data_y[train_ind], data_y[test_ind]

                model_copy = self._model.clone()  # need to reuse this object type for each fold/repeat
                model_copy.train(data_x=train_x, data_y=train_y, hyper_params=hyper_params)

                # for each evaluator, add the metric name/value to a dict to add to the ResamplerResults
                results_dict = dict()
                for evaluator in self._evaluators:
                    evaluator_copy = evaluator.clone()  # need to reuse this object type for each fold/repeat
                    evaluator_copy.evaluate(actual_values=test_y,
                                            predicted_values=model_copy.predict(data_x=test_x))
                    results_dict[evaluator_copy.metric_name] = evaluator_copy.accuracy

                resampler_results.add(result=results_dict)

        return resampler_results
