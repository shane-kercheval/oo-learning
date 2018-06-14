import numpy as np
import pandas as pd
from typing import List, Callable, Union

from oolearning.evaluators.ScoreBase import ScoreBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase
from oolearning.model_processors.ResamplerBase import ResamplerBase
from oolearning.model_processors.ResamplerResults import ResamplerResults
from oolearning.persistence.PersistenceManagerBase import PersistenceManagerBase
from oolearning.splitters.StratifiedDataSplitter import StratifiedDataSplitter
from oolearning.transformers.TransformerBase import TransformerBase
from oolearning.transformers.TransformerPipeline import TransformerPipeline


# TODO: NOT FINISHED, NOT TESTED
class StratifiedMonteCarloResampler(ResamplerBase):
    def __init__(self,
                 model: ModelWrapperBase,
                 transformations: List[TransformerBase],
                 stratified_splitter: StratifiedDataSplitter,
                 scores: List[ScoreBase],
                 model_persistence_manager: PersistenceManagerBase = None,
                 train_callback: Callable[[pd.DataFrame, np.ndarray,
                                           Union[HyperParamsBase, None]], None] = None,
                 repeats=30):
        """
        :param model:
        :param transformations:
        :param stratified_splitter: e.g. ClassificationStratifiedDataSplitter or
            RegressionStratifiedDataSplitter; i.e. this resampler needs to know how to stratify the data
            based off of the target values
        :param scores:
        :param repeats:
        """
        super().__init__(model=model,
                         transformations=transformations,
                         scores=scores,
                         model_persistence_manager=model_persistence_manager,
                         train_callback=train_callback)

        assert isinstance(repeats, int)

        self._repeats = repeats
        self._stratified_splitter = stratified_splitter

    def _resample(self,
                  data_x: pd.DataFrame,
                  data_y: np.ndarray,
                  hyper_params: HyperParamsBase = None) -> ResamplerResults:

        result_evaluators = list()
        training_indexes, test_indexes = self._stratified_splitter.split_monte_carlo(target_values=data_y,
                                                                                     samples=self._repeats,
                                                                                     seed=42)
        for train_ind, test_ind in zip(training_indexes, test_indexes):
                train_x_not_transformed, holdout_x_not_transformed = data_x[train_ind], data_x[test_ind]
                train_y, test_y = data_y[train_ind], data_y[test_ind]

                pipeline = TransformerPipeline(transformations=self._transformations)
                train_x_transformed = pipeline.fit_transform(data_x=train_x_not_transformed)
                holdout_x_transformed = pipeline.transform(data_x=holdout_x_not_transformed)

                if self._train_callback is not None:
                    self._train_callback(train_x_transformed, data_y, hyper_params)

                model_copy = self._model.clone()  # need to reuse this object type for each fold/repeat
                model_copy.train(data_x=train_x_not_transformed, data_y=train_y, hyper_params=hyper_params)

                # for each evaluator, add the metric name/value to a dict to add to the ResamplerResults
                fold_evaluators = list()
                for evaluator in self._scores:
                    evaluator_copy = evaluator.clone()  # need to reuse this object type for each fold/repeat
                    evaluator_copy.calculate(actual_values=test_y,
                                             predicted_values=model_copy.predict(data_x=holdout_x_transformed))  # noqa
                    fold_evaluators.append(evaluator_copy)
                result_evaluators.append(fold_evaluators)

        return ResamplerResults(scores=result_evaluators, decorators=None)
