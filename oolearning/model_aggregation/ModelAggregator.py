from typing import Union, List
from multiprocessing import Pool as ThreadPool
from multiprocessing import cpu_count

import numpy as np
import pandas as pd

from oolearning.transformers.TransformerPipeline import TransformerPipeline
from oolearning.model_aggregation.AggregationStrategyBase import AggregationStrategyBase
from oolearning.model_processors.ModelInfo import ModelInfo
from oolearning.model_wrappers.HyperParamsBase import HyperParamsBase
from oolearning.model_wrappers.ModelWrapperBase import ModelWrapperBase


def train_aggregator(args):
    model_info = args[0]
    data_x_local = args[1]
    data_y_local = args[2]

    if model_info.transformations:
        # ensure none of the Transformers have been used.
        assert all([x.state is None for x in model_info.transformations])

    # List of Pipelines to cache for `predict()`
    pipeline = TransformerPipeline(model_info.transformations)
    # fit/transform with current pipeline
    transformed_data_x_local = pipeline.fit_transform(data_x=data_x_local)
    model_info.model.train(data_x=transformed_data_x_local,
                           data_y=data_y_local,
                           hyper_params=model_info.hyper_params)
    return model_info, pipeline


class ModelAggregator(ModelWrapperBase):
    """
    Simple Bridge Between ModelWrapperBase and AggregationStrategyBase. Allows this object to be treated
        like any other model.
    """
    def __init__(self,
                 base_models: List[ModelInfo],
                 aggregation_strategy: AggregationStrategyBase,
                 parallelization_cores: int=-1):
        """

        :param base_models: list of ModelInfos describing the models to train
        :param aggregation_strategy: object defines how the the final predictions from the base_models should
            be aggregated
        :param parallelization_cores: the number of cores to use for parallelization. -1 is all, 0 or 1 is 
            "off"
        """
        super().__init__()
        assert len(base_models) >= 3
        self._base_models = base_models
        self._base_transformation_pipeline = list()
        self._aggregation_strategy = aggregation_strategy

        self._parallelization_cores = parallelization_cores

    @property
    def feature_importance(self):
        raise NotImplementedError()

    def _train(self,
               data_x: pd.DataFrame,
               data_y: np.ndarray,
               hyper_params: HyperParamsBase = None) -> object:

        assert hyper_params is None  # not used in Aggregator

        # map_function rather than a for loop so we can switch between parallelization and non-parallelization
        aggregator_args = [(model_info, data_x, data_y) for model_info in self._base_models]

        if self._parallelization_cores == 0 or self._parallelization_cores == 1:
            results = list(map(train_aggregator, aggregator_args))
        else:
            cores = cpu_count() if self._parallelization_cores == -1 else self._parallelization_cores
            with ThreadPool(cores) as pool:
                results = list(pool.map(train_aggregator, aggregator_args))

        self._base_models = [x[0] for x in results]
        # List of Pipelines to cache for `predict()`
        self._base_transformation_pipeline = [x[1] for x in results]

        # for index, model_info in enumerate(self._base_models):
        #
        #     if model_info.transformations:
        #         # ensure none of the Transformers have been used.
        #         assert all([x.state is None for x in model_info.transformations])
        #
        #     # List of Pipelines to cache for `predict()`
        #     self._base_transformation_pipeline.append(TransformerPipeline(model_info.transformations))
        #     # fit/transform with current pipeline
        #     transformed_data_x = self._base_transformation_pipeline[index].fit_transform(data_x=data_x)
        #
        #     model_info.model.train(data_x=transformed_data_x,
        #                            data_y=data_y,
        #                            hyper_params=model_info.hyper_params)

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
