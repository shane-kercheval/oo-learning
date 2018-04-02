from typing import Union, List

import numpy as np
import pandas as pd

from oolearning.model_aggregation.AggregationStrategyBase import AggregationStrategyBase
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class ModelAggregator(ModelWrapperBase):
    """
    Simple Bridge Between ModelWrapperBase and AggregationStrategyBase
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
        model_predictions = [x.predict(data_x=data_x) for x in self._models]
        # need to ensure that all of the resulting prediction dataframes have the same indexes as `data_x`,
        # because we rely on the indexes to calculate the means
        # noinspection PyTypeChecker
        assert all([all(x.index.values == data_x.index.values) for x in model_predictions])
        voting_predictions = self._aggregation_strategy.aggregate(model_predictions=model_predictions)

        return voting_predictions
