from typing import Union, List

import numpy as np
import pandas as pd

from oolearning.transformers.TransformerPipeline import TransformerPipeline
from oolearning.model_aggregation.AggregationStrategyBase import AggregationStrategyBase
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


class ModelAggregator(ModelWrapperBase):
    """
    Simple Bridge Between ModelWrapperBase and AggregationStrategyBase. Allows this object to be treated
        like any other model.
    """
    def __init__(self, base_models: List[ModelInfo], aggregation_strategy: AggregationStrategyBase):
        """
        :param base_models: pre-trained base_models used to predict on `data_x` (predictions will be aggregated
            according to child objects
        """
        super().__init__()
        assert len(base_models) >= 3
        self._base_models = base_models
        self._base_transformation_pipeline = list()
        self._aggregation_strategy = aggregation_strategy

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        for index, model_info in enumerate(self._base_models):
            if model_info.transformations:
                # ensure none of the Transformers have been used.
                assert all([x.state is None for x in model_info.transformations])

            self._base_transformation_pipeline.append(TransformerPipeline(model_info.transformations))
            transformed_data_x = self._base_transformation_pipeline[index].fit_transform(data_x=data_x)

            model_info.model.train(data_x=transformed_data_x,
                                   data_y=data_y,
                                   hyper_params=model_info.hyper_params)

        return ''

    def _predict(self, model_object: object, data_x: pd.DataFrame) -> Union[np.ndarray, pd.DataFrame]:
        """
        :param model_object: not used
        :param data_x: for each of the base_models passed into the constructor, the predictions are made with
        `data_x` and then are aggregated according to the specified strategy.
        :return: aggregated predictions.
        """
        model_predictions = list()
        for index, model_info in enumerate(self._base_models):
            transformed_data_x = self._base_transformation_pipeline[index].transform(data_x=data_x)
            model_predictions.append(model_info.model.predict(data_x=transformed_data_x))

        assert len(model_predictions) == len(self._base_models)
        if isinstance(model_predictions[0], pd.DataFrame):
            # need to ensure that all of the resulting prediction dataframes have the same indexes as `data_x`
            # because we rely on the indexes to calculate the means
            # noinspection PyTypeChecker
            assert all([all(x.index.values == data_x.index.values) for x in model_predictions])

        return self._aggregation_strategy.aggregate(model_predictions=model_predictions)
