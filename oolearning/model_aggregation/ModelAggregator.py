from typing import Union, List

import numpy as np
import pandas as pd

from oolearning.model_aggregation.AggregationStrategyBase import AggregationStrategyBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class ModelAggregator(ModelWrapperBase):
    """
    Simple Bridge Between ModelWrapperBase and AggregationStrategyBase. Allows this object to be treated
        like any other model; although there is no logic necessary in `.train()`.
    """
    def __init__(self, models: List[ModelWrapperBase], aggregation_strategy: AggregationStrategyBase):
        """
        :param models: pre-trained models used to predict on `data_x` (predictions will be aggregated
            according to child objects
        """
        super().__init__()
        assert len(models) >= 3
        self._models = models
        self._aggregation_strategy = aggregation_strategy

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self, data_x: pd.DataFrame, data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:
        """
        nothing to do in train(); models are already pre-trained
        """
        return 0

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        :param model_object: not used
        :param data_x: for each of the models passed into the constructor, the predictions are made with
        `data_x` and then are aggregated according to the specified strategy.
        :return: aggregated predictions.
        """
        model_predictions = [x.predict(data_x=data_x) for x in self._models]
        assert len(model_predictions) == len(self._models)
        if isinstance(model_predictions[0], pd.DataFrame):
            # need to ensure that all of the resulting prediction dataframes have the same indexes as `data_x`
            # because we rely on the indexes to calculate the means
            # noinspection PyTypeChecker
            assert all([all(x.index.values == data_x.index.values) for x in model_predictions])

        return self._aggregation_strategy.aggregate(model_predictions=model_predictions)
